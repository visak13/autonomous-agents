"""Reactive self-heal ROUTING — a node FAILURE → the planner's heal DECISION (§2e, d1).

A failed node must never leave a dead plan. This module is the seam that ROUTES a
node failure to the PLANNER's heal logic and maps its structured decision onto the
runtime's recovery primitive. The doctrine it upholds (d1) is precise:

    the reactive heal RULE only ROUTES the failure event; the PLANNER owns the
    control-flow DECISION (``Planner.heal_decision`` picks one of
    ``retry|pivot|extend|abort``); the RUNTIME mechanically ENACTS the routed
    action (retry re-dispatch / replan_subgraph / surface).

WHY A SEPARATE ROUTER — NOT the LambdaRegistry lambda
-----------------------------------------------------
A :class:`~reactive_tools.subscriptions.LambdaRegistry` lambda is advisory /
observe-only BY DOCTRINE ("rx observes, CRUD mutates") — it can NEVER perform heal
control flow or mutate DAG state. So the "heal rule registered on the EventPlane/
LambdaRegistry" is an OBSERVE-ONLY record of every routed failure (the reactive,
user-visible surface — :func:`register_heal_rule`); the actual decision lives in
the planner and the enactment lives in the runtime. The runtime publishes the
failure on the plane (so the advisory rule fires), this router asks the planner,
and an ``agent_heal_routed`` event records the chosen action — all observable, and
none of it mutating from inside a lambda.

In-process + dependency-light (d2/d10): a plain object holding the planner; the
``heal_decision`` it calls is the native structured Gemma enum call a2 built.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .planner import HEAL_ACTIONS
from .selfheal import MalformedOutputError

# --------------------------------------------------------------------------- #
# Plane vocabulary — the reactive heal rule observes these control-plane kinds.
# Both are LOW-FREQUENCY (one per node failure), so a lambda may reduce them with
# ``each`` (they are not the data-plane tool_call/tool_result kinds the anti-wake-
# storm guard rejects).
# --------------------------------------------------------------------------- #
EVENT_NODE_FAILURE_DETECTED = "agent_node_failure_detected"  # a node failed; routing begins
EVENT_HEAL_ROUTED = "agent_heal_routed"                      # the planner's heal decision

# The kinds the registered self-heal rule observes (the failure + the routed decision).
HEAL_RULE_KINDS: tuple[str, ...] = (EVENT_NODE_FAILURE_DETECTED, EVENT_HEAL_ROUTED)

# How each planner heal ACTION maps to the runtime's recovery KIND.
#   retry         → idempotent re-dispatch of the SAME failed node.
#   pivot, extend → replan_subgraph corrective sub-DAG (a different approach / an
#                   extra remediation step).
#   abort         → surface the failure to the user/neuron (no recovery).
_RETRY_ACTIONS = frozenset({"retry"})
_REPLAN_ACTIONS = frozenset({"pivot", "extend"})
_ABORT_ACTIONS = frozenset({"abort"})


def _kind_for(action: str) -> str:
    """Map a planner heal ACTION enum to the runtime recovery KIND.

    Unknown / unexpected actions map to ``replan`` — the safest recovery (a
    corrective sub-graph), never a silent give-up."""
    if action in _RETRY_ACTIONS:
        return "retry"
    if action in _REPLAN_ACTIONS:
        return "replan"
    if action in _ABORT_ACTIONS:
        return "abort"
    return "replan"


@dataclass
class HealRoute:
    """The routed heal outcome for ONE failed node — the planner DECISION + the
    runtime KIND it maps to.

    ``action`` is the planner's enum choice (one of :data:`HEAL_ACTIONS`).
    ``kind`` is the recovery primitive the runtime enacts: ``retry`` (idempotent
    re-dispatch of the SAME node), ``replan`` (replan_subgraph corrective sub-DAG,
    for ``pivot``/``extend``), or ``abort`` (surface to user/neuron). ``fallback``
    is True when no legal decision could be obtained (exhausted JSON repair) and
    the router defaulted to a safe ``replan`` — so heal is never WORSE than the
    pre-b4 unconditional replan."""

    action: str
    kind: str
    rationale: str = ""
    fallback: bool = False

    @property
    def is_retry(self) -> bool:
        return self.kind == "retry"

    @property
    def is_replan(self) -> bool:
        return self.kind == "replan"

    @property
    def is_abort(self) -> bool:
        return self.kind == "abort"

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "kind": self.kind,
            "rationale": self.rationale,
            "fallback": self.fallback,
        }


class HealRouter:
    """Planner-owned heal control flow: route a node FAILURE to a :class:`HealRoute`.

    Wired into the runtime's failure seam (``AgentRuntime(heal_router=…)``). On a
    node failure the runtime calls :meth:`route`; this asks the PLANNER for a
    structured heal decision (``planner.heal_decision`` — the native enum call a2
    built) and maps it onto the runtime recovery kind. The runtime then ENACTS the
    kind — so the *decision* stays planner-owned (d1) while the *enactment* (which
    mutates DAG state) stays inside the runtime's trust boundary.

    ``max_retries`` bounds the coarse re-dispatch: a ``retry`` decision once the
    per-node retry budget is spent is ESCALATED to ``replan`` so a permanently
    failing node can never spin. When ``heal_decision`` cannot yield a legal action
    (exhausted JSON repair → :class:`MalformedOutputError`), the router falls back
    to ``replan`` — the exact pre-b4 behaviour — so wiring the router is never a
    regression.
    """

    def __init__(self, planner: Any, *, max_retries: int = 1) -> None:
        self.planner = planner
        self.max_retries = max_retries

    async def route(
        self,
        failed_task: str,
        error: str,
        *,
        attempt: int = 0,
        completed: Optional[Sequence[str]] = None,
    ) -> HealRoute:
        """Ask the planner how to heal ``failed_task`` and map it to a recovery kind.

        ``attempt`` is how many coarse re-dispatches this node has already had
        (the runtime tracks it); ``completed`` is the already-DONE node ids (names
        only, for the model's awareness — never their outputs, d10)."""
        try:
            decision = await self.planner.heal_decision(
                failed_task,
                error,
                attempt=attempt,
                max_attempts=self.max_retries,
                completed=list(completed or []),
            )
        except MalformedOutputError as exc:
            # No legal enum after the bounded repair loop → safe default: replan
            # (the pre-b4 unconditional sub-graph re-plan), flagged as a fallback.
            return HealRoute(
                action="pivot",
                kind="replan",
                rationale=f"heal_decision unavailable ({exc}); defaulting to replan",
                fallback=True,
            )
        kind = _kind_for(decision.action)
        # Bounded coarse re-dispatch: a retry past the budget escalates to a replan
        # so a node that keeps failing can never loop forever on "retry".
        if kind == "retry" and attempt >= self.max_retries:
            kind = "replan"
        return HealRoute(action=decision.action, kind=kind, rationale=decision.rationale)


def register_heal_rule(
    registry: Any,
    *,
    run_id: str = "",
    source_plane: Any = None,
) -> Optional[str]:
    """Register the OBSERVE-ONLY reactive self-heal RULE on a LambdaRegistry (d1/d15).

    Creates an advisory lambda that observes the self-heal control-plane kinds
    (:data:`HEAL_RULE_KINDS`) on the run's plane — so every node-failure routing is
    visible on the read-only live-subscriptions surface the UI renders. The rule
    ONLY routes/observes; it never mutates state or enacts a heal (the planner +
    runtime do that). Returns the created lambda's ``sub_id``, or ``None`` if no
    registry was wired. Best-effort: a registry failure here must never break a run,
    so it is swallowed (the rule is observe-only).

    ``source_plane`` decouples WHERE events are observed (the per-run plane the
    failures flow on) from WHERE the rule is recorded (this shared registry the UI
    reads) — mirroring the per-run observability lambda."""
    if registry is None:
        return None
    try:
        rec = registry.create(
            list(HEAL_RULE_KINDS),
            label=f"self-heal-rule:{run_id}" if run_id else "self-heal-rule",
            reducer="each",  # control-plane kinds only → no wake-storm
            reaction="advisory",  # OBSERVE-ONLY: routes/advises, never mutates (d1)
            owner={"run_id": run_id, "kind": "self-heal", "auto": True},
            note="reactive self-heal rule: observes node-failure routing (advisory only)",
            source_plane=source_plane,
        )
        return rec.sub_id
    except Exception:  # noqa: BLE001 — advisory rule must never break a run
        return None


__all__ = [
    "HealRoute",
    "HealRouter",
    "register_heal_rule",
    "EVENT_NODE_FAILURE_DETECTED",
    "EVENT_HEAL_ROUTED",
    "HEAL_RULE_KINDS",
]
