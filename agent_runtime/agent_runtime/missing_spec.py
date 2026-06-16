"""Missing-specialist detection + the user notify/CHOICE surface (s4 M1, RC8).

Blueprint RC8 / §2b: when the planner authors a node that REQUIRES a specialist
but NO registered specialization fits, the node must NOT silently degrade to raw
LLM auto-completion (the round-1/s8 ``spec=""`` -> ``_compose_system()=None`` bug).
Instead the chat NOTIFIES the user and offers an explicit CHOICE:

  * ``sse_fallback``     — proceed anyway, running the node spec-less and streaming
                           its raw output to the chat (the user accepts a generic,
                           unspecialized answer for this step);
  * ``define_and_resume``— pause the plan, let the user DEFINE the missing
                           specialization (the existing spec-authoring chat
                           surface), then RESUME the same plan with the new spec
                           stamped onto the node.

This module is the BACKEND of that flow — pure data + detection + the deterministic
DAG rewrites the two resolutions apply. It carries NO model call and NO HTTP: the
live path (``chat_app.agentic``) calls :func:`detect_missing_specialists` before it
drives the runtime, emits :data:`EVENT_MISSING_SPECIALIST` (so the choice streams to
the chat over SSE), and — once the user picks — calls :func:`apply_resolution` to
get the DAG to actually run. Kept here (pure, import-light) so it stays trivially
testable and the engine owns the mechanic, not the web layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from .factory import PlanDAG, PlanNode

# The lifecycle event published on the in-process plane when a plan cannot run as
# authored because a node needs an unavailable specialist. The chat's SSE stream
# relays it (it is added to the streamed kinds) so the user is NOTIFIED live and
# shown the CHOICE — it is never a silent failure.
EVENT_MISSING_SPECIALIST = "agent_missing_specialist"

# The two resolutions the user may pick (the CHOICE offered alongside the notify).
CHOICE_SSE_FALLBACK = "sse_fallback"
CHOICE_DEFINE_AND_RESUME = "define_and_resume"
MISSING_SPEC_CHOICES: tuple[str, ...] = (CHOICE_SSE_FALLBACK, CHOICE_DEFINE_AND_RESUME)


@dataclass(frozen=True)
class MissingSpecialist:
    """One node that needs a specialist no registered specialization provides.

    ``needs`` is the planner's free-text descriptor of the required specialist
    (the ``PlanNode.needs_spec`` escape hatch); ``role`` is the node's role (if
    any) — together they describe the capability the user is asked to supply or
    waive."""

    node_id: str
    task: str
    needs: str
    role: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "task": self.task,
            "needs": self.needs,
            "role": self.role,
        }


def detect_missing_specialists(
    dag: PlanDAG, registered: Iterable[str]
) -> list[MissingSpecialist]:
    """Return every node that DECLARED a needed specialist none can satisfy.

    A node is missing-specialist when it carries a non-empty ``needs_spec`` AND
    has no ``effective_specs`` that resolve against ``registered`` (the registry's
    known spec names). A node whose ``effective_specs`` are all registered is fine
    (it has its grounding); a bare node with no ``needs_spec`` is an ordinary
    unspecialized step (the planner judged no specialist is required) and is NOT
    flagged — only an EXPLICIT unmet need is surfaced, so the notify never
    false-fires on routine bare steps."""
    known = set(registered)
    missing: list[MissingSpecialist] = []
    for n in dag.nodes:
        if not n.needs_spec:
            continue
        # If the planner also bound a registered spec, the need is already met.
        if any(s in known for s in n.effective_specs):
            continue
        missing.append(
            MissingSpecialist(
                node_id=n.id, task=n.task, needs=n.needs_spec, role=n.role
            )
        )
    return missing


def _sink_nodes(dag: PlanDAG) -> list[PlanNode]:
    """The DAG's SINK nodes (nothing depends on them) — the terminal/answer step(s).

    Falls back to the last node in topo order if every node has a dependent (a valid
    acyclic DAG always has at least one sink, so this is purely defensive)."""
    depended_on = {d for n in dag.nodes for d in n.depends_on}
    sinks = [n for n in dag.nodes if n.id not in depended_on]
    if sinks:
        return sinks
    order = dag.topo_order()
    return [order[-1]] if order else []


def missing_from_requested(
    dag: PlanDAG, requested: Iterable[str], registered: Iterable[str]
) -> list[MissingSpecialist]:
    """Surface a USER-REQUESTED specialization no registered spec provides.

    This is the STRUCTURAL scenario-3 trigger (s10-a8). The pre-a8 path relied on a
    node VOLUNTEERING a free-text ``needs_spec`` during incremental authoring — a
    signal the 4.6B model does not reliably emit (s10-a4), so the missing-specialist
    notify never fired live. Instead, the shape selector reliably EXTRACTS the
    specialization name(s) the user asked for (the proven a3 ``requested_specs``
    layer; the new ``unmet_specs`` field lets the model name one the registry does
    NOT have), and the TRIGGER becomes a DETERMINISTIC registry-membership check
    here: a requested name that is not in ``registered`` is a missing specialist.

    Each unmet spec is attached to the DAG's SINK node(s) — the terminal/answer step
    that produces the deliverable — so a ``define_and_resume`` stamps the
    newly-defined specialization exactly where the answer is written, and an
    ``sse_fallback`` runs that node spec-less (the same two resolutions a
    node-volunteered ``needs_spec`` offers, via the SAME notify + :func:`apply_resolution`
    path — only the trigger changed). Returns [] when every requested spec is
    registered (the common case → no false notify). Generic: a set-membership check,
    no scenario/keyword/topic matching."""
    known = set(registered)
    seen: list[str] = []
    for raw in requested:
        name = str(raw).strip()
        if name and name not in known and name not in seen:
            seen.append(name)
    if not seen:
        return []
    needs = ", ".join(seen)
    return [
        MissingSpecialist(node_id=n.id, task=n.task, needs=needs, role=n.role)
        for n in _sink_nodes(dag)
    ]


def missing_specialist_payload(
    missing: list[MissingSpecialist],
    *,
    resume_token: str,
) -> dict[str, Any]:
    """Build the :data:`EVENT_MISSING_SPECIALIST` payload (notify + the CHOICE).

    Carries the unmet nodes, the two offered choices, and the opaque
    ``resume_token`` the client echoes back when it picks a resolution — so the
    notify the user sees IS actionable, not a dead-end error."""
    return {
        "resume_token": resume_token,
        "choices": list(MISSING_SPEC_CHOICES),
        "missing": [m.as_dict() for m in missing],
    }


def apply_resolution(
    dag: PlanDAG,
    missing: list[MissingSpecialist],
    *,
    choice: str,
    defined_specs: Optional[Mapping[str, str]] = None,
) -> PlanDAG:
    """Rewrite ``dag`` so it can RUN under the user's chosen resolution.

    Deterministic, model-free DAG surgery applied to exactly the missing-spec
    nodes:

    * ``sse_fallback`` — clear each missing node's ``needs_spec`` so it runs as a
      plain spec-less step (raw LLM), its output streaming to the chat. The user
      knowingly accepts a generic answer for that step; it is no longer a blocking
      unmet need.
    * ``define_and_resume`` — stamp the now-defined specialization onto each
      missing node (``defined_specs`` maps ``node_id -> spec_name``; a single
      mapping value may also key by the empty string ``""`` to apply ONE newly
      defined spec to every missing node) and clear ``needs_spec``. The node now
      carries its grounding and runs specialized.

    Every other node is returned unchanged. Raises ``ValueError`` on an unknown
    choice, or on ``define_and_resume`` with no spec supplied for a missing node
    (fail-closed — never resume a still-unmet node silently)."""
    if choice not in MISSING_SPEC_CHOICES:
        raise ValueError(
            f"unknown missing-specialist choice {choice!r}; "
            f"expected one of {list(MISSING_SPEC_CHOICES)}"
        )
    missing_ids = {m.node_id for m in missing}
    defined = dict(defined_specs or {})
    rewritten: list[PlanNode] = []
    for n in dag.nodes:
        if n.id not in missing_ids:
            rewritten.append(n)
            continue
        if choice == CHOICE_SSE_FALLBACK:
            rewritten.append(_with(n, specs=n.specs, needs_spec=None))
            continue
        # define_and_resume: resolve the spec to stamp (per-node, else the global
        # "" fallback), fail-closed if none was supplied.
        spec_name = defined.get(n.id) or defined.get("")
        if not spec_name:
            raise ValueError(
                f"define_and_resume: no defined spec supplied for node {n.id!r} "
                f"(needed: {n.needs_spec!r})"
            )
        merged = tuple([*n.specs, spec_name]) if n.specs else (spec_name,)
        rewritten.append(_with(n, specs=merged, needs_spec=None))
    return PlanDAG(nodes=rewritten, rationale=dag.rationale, shape=dag.shape)


def _with(node: PlanNode, *, specs: tuple[str, ...], needs_spec: Optional[str]) -> PlanNode:
    """Return a copy of ``node`` with ``specs``/``needs_spec`` replaced (others kept)."""
    return PlanNode(
        id=node.id,
        task=node.task,
        spec=node.spec,
        specs=specs,
        depends_on=node.depends_on,
        tool=node.tool,
        tool_args=node.tool_args,
        role=node.role,
        needs_spec=needs_spec,
    )


__all__ = [
    "EVENT_MISSING_SPECIALIST",
    "CHOICE_SSE_FALLBACK",
    "CHOICE_DEFINE_AND_RESUME",
    "MISSING_SPEC_CHOICES",
    "MissingSpecialist",
    "detect_missing_specialists",
    "missing_from_requested",
    "missing_specialist_payload",
    "apply_resolution",
]
