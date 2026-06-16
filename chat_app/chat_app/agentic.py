"""LIVE agentic run for the chat server — the REAL planner+runtime path (s3/b2).

A user's chat message is now driven through the REAL Round-3 agent (RC1/RC2 fix):

1. SHAPE SELECTION (s3/b1) — a native structured Gemma call picks the plan SHAPE
   that fits THIS query (``linear`` / ``modular-parallel`` / the bounded cyclic
   ``deep-research`` / any text-file shape on disk), or escalates when unsure.
2. The DAG is DERIVED from the actual query — NOT a hard-coded two-report goal:
   * an acyclic shape → the REAL :class:`~agent_runtime.Planner` self-derives the
     DAG (``Planner.plan(query)``) and the REAL :class:`~agent_runtime.AgentRuntime`
     drives it on the LIVE Gemma transport, honouring the shape's execution
     discipline (``linear`` = strict sequential, ``modular-parallel`` = concurrent
     ready-wave — the s3/b1 :mod:`agent_runtime.scheduler` port);
   * the ``deep-research`` (cyclic) shape → UNROLLED by the generic
     :func:`~agent_runtime.unroll_shape` into a bounded acyclic role-tagged DAG
     (ONE specialization reused across ~10 rounds, role-differentiated, growing
     visibility) and driven by the SAME :class:`~agent_runtime.AgentRuntime` —
     there is no per-shape executor (a3 re-architecture).

What this REPLACES (RC1/RC2): the old fixed ``analyze → draft_md → draft_html``
stub DAG on the :class:`~agent_runtime.stub` FakeTransport, the hard-coded
TWO-REPORT (md+html) goal, and the ``ensure_grounded`` plan-rewrite that forced
EVERY request into ``research → md → html``. The planner now derives both the
shape and the nodes from the user's actual request — no template, no forced shape.

The minimal canonical specialization(s) the live path needs to inject (RC3) are
seeded at boot by ``chat_app.app.build_wiring`` (``markdown-writer`` +
``research-analyst``); the full specialization-management surface is s4's, not
here. d8 (the unattended-email safety invariant, b5): the planner is offered the
full six-bucket s2 node→tool surface (``OFFERED_TOOLS`` below), but the ONLY mail
tool is the recipient-hard-locked ``send_mail`` — the legacy free-``to``
``send_email`` is neither offered NOR registered on the hook, so the model can
never emit an arbitrary recipient through this path (nor the s5/s6 plans reusing
it).
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from reactive_tools import EventPlane, ToolHook
from specialization import SpecLoader, SpecRegistry
from specialization.seed import DEEP_RESEARCH_SPEC
from agent_runtime import (
    EVENT_MISSING_SPECIALIST,
    EVENT_NEEDS_CLARIFICATION,
    MISSING_SPEC_CHOICES,
    AbstractPlanFactory,
    clarification_payload,
    AgentRuntime,
    ExecutionMode,
    HealRouter,
    IncrementalPlanner,
    MalformedOutputError,
    PlanDAG,
    Planner,
    ROLE_SYNTHESIS,
    ROLE_VERIFY,
    SchemaToolArgEmitter,
    ShapeSelection,
    ShapeSelector,
    apply_missing_spec_resolution,
    default_node_verifier,
    detect_missing_specialists,
    execution_mode_for,
    missing_from_requested,
    missing_specialist_payload,
    unroll_shape,
)
from agent_runtime.tracing import get_tracer

# Back-compat: the names the s6 workflow producer (chat_app.workflow) + existing
# specs/tests still reference. ``markdown-writer`` / ``html-writer`` are the two
# round-2 writer rulesets; ``both_specs_registered`` is still the s6 live-brief
# pre-req gate. b2 no longer routes the chat path on these (it routes on live mode
# + the derived shape), but the symbols stay so s6 + its tests keep working.
MD_SPEC = "markdown-writer"
HTML_SPEC = "html-writer"

# The tools the planner may BIND to a node (the structured-output enum). b5 wires
# the FULL six-bucket s2 node→tool surface (web_search, web_fetch, file_read,
# file_write, the recipient-LOCKED send_mail, and cron_add/list/delete) so a node
# ANSWERS via tools rather than raw LLM auto-completion, plus the read-only
# observability tools b2 already offered.
#
# d8 (the unattended-email safety invariant): the ONLY mail tool offered is the
# recipient-hard-locked ``send_mail`` (its exposed schema has no ``to`` field and
# its adapter always targets ``SMTP_FROM_EMAIL``). The legacy free-``to``
# ``send_email`` is NOT in this enum AND is no longer registered on the hook
# (build_default_hook dropped it), so the model can never emit an arbitrary
# recipient through the routed chat path (or the s5/s6 plans that reuse it).
# Only names actually present on the hook are offered (see ``_run_acyclic``), so
# a tool absent from the wiring is silently skipped rather than enum-poisoning.
OFFERED_TOOLS = (
    "web_search",
    "web_fetch",
    "file_read",
    "file_write",
    "send_mail",
    "cron_add",
    "cron_list",
    "cron_delete",
    "create_subscription",
    "list_subscriptions",
)

# F5 'DO NOT SEARCH' ENFORCEMENT: the tools that touch the WEB. When the
# model-driven router judges a request forbids searching (``search_allowed=False``
# — e.g. "without searching, from your own knowledge"), these are STRIPPED from the
# tools offered to BOTH the incremental authorer and the self-heal re-planner, so a
# node can never bind one and the runtime can never fire one — a structural zero-
# web-call guarantee, not a phrasing/keyword check. (Identifying which registry
# tools are web tools is configuration, NOT scenario special-casing.)
WEB_TOOLS = ("web_search", "web_fetch")


def _filter_web_tools(tool_names: list[str], allow_web: bool) -> list[str]:
    """Drop the :data:`WEB_TOOLS` from ``tool_names`` when the web is disallowed."""
    if allow_web:
        return list(tool_names)
    web = set(WEB_TOOLS)
    return [t for t in tool_names if t not in web]


def _deep_research_spec(
    registry: SpecRegistry, requested_specs: Optional[list[str]]
) -> Optional[str]:
    """The SINGLE specialization the deep-research shape reuses across all rounds.

    F5: a user-EXPLICITLY-named (registered) specialization wins over the hard-coded
    ``research-analyst`` default — which previously made a named output spec
    structurally unreachable on this route (the F5(i) failure). Falls back to the
    seeded ``DEEP_RESEARCH_SPEC`` when the user named none, or ``None`` when even
    that is absent (role-framing only)."""
    for name in requested_specs or []:
        if name in registry:
            return name
    return DEEP_RESEARCH_SPEC if DEEP_RESEARCH_SPEC in registry else None


def build_plan_schema(spec_names: list[str], tool_names: list[str]) -> dict[str, Any]:
    """The plan schema (Ollama native ``format``) with ENUM-constrained spec/tool.

    'spec'/'specs' and 'tool' are enums of exactly the registered names (+ "" for
    none), so Gemma cannot put a tool name in the spec slot or invent a tool —
    while it still freely chooses the nodes, edges and bindings (autonomy
    preserved; only the vocabulary is constrained). This is the proven d1
    structured path.

    N-SPEC PER NODE (s4 M1, blueprint §2b): ``specs`` is an ARRAY whose items are
    the SAME registered-name enum, so the planner can COMPOSE several
    specializations onto one node (the runtime layers them into one assembled
    system — :meth:`SubAgent._compose_ruleset_stack`). Without exposing ``specs``
    in the native ``format`` the live planner was structurally limited to ONE spec
    per node even though the factory/runtime already supported N.

    MISSING-SPECIALIST SIGNAL (s4 M1, RC8): ``needs_spec`` is a FREE-TEXT string
    (deliberately NOT enum-constrained) so the planner can DECLARE a node needs a
    specialist that is absent from the lookup — instead of silently leaving it
    unspecialized. A node with ``needs_spec`` and no resolvable ``spec``/``specs``
    is surfaced to the user as a notify + CHOICE (see
    :mod:`agent_runtime.missing_spec`)."""
    spec_enum = [""] + list(spec_names)
    return {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "task": {"type": "string"},
                        "spec": {"type": "string", "enum": spec_enum},
                        "specs": {
                            "type": "array",
                            "items": {"type": "string", "enum": spec_enum},
                        },
                        "needs_spec": {"type": "string"},
                        "tool": {"type": "string", "enum": [""] + list(tool_names)},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id", "task"],
                },
            },
        },
        "required": ["rationale", "nodes"],
    }


def both_specs_registered(registry: SpecRegistry) -> bool:
    """True when the two writer specialists exist (the s6 live-brief pre-req).

    KEPT for back-compat with the s6 workflow producer + its tests; the b2 chat
    route no longer gates on it (it routes on live mode + the derived shape)."""
    return MD_SPEC in registry and HTML_SPEC in registry


def agentic_goal(query: str) -> str:
    """The GOAL handed to the planner — the user's ACTUAL request, verbatim (d6).

    RC2 fix: there is NO hard-coded two-report framing here any more. The user's
    own message IS the goal; the planner self-derives the shape (via the selector)
    and the nodes (via ``Planner.plan``) from it — no template, no forced shape."""
    return (query or "").strip()


# Clearly-delimited header bounding the prior-turn block the planner sees, so the
# model treats it as CONTEXT (not a new instruction) and continues the thread.
_PRIOR_CONTEXT_HEADER = (
    "PRIOR CONVERSATION (recent turns of THIS chat thread, for context — the "
    "user's new request continues it; do not re-answer the old turns):"
)
_CURRENT_REQUEST_HEADER = "CURRENT REQUEST:"


def goal_with_context(conversation_context: Optional[str], query: str) -> str:
    """Prefix the bounded prior-turn context (s5/a1) onto the planning GOAL (s5/a2).

    The shape selector, :class:`~agent_runtime.Planner`, and every node the planner
    derives all reason over the GOAL STRING; threading the per-chat conversation
    context in HERE — clearly delimited, and AFTER a1's recent-N/summary bound has
    already shrunk it to fit the small Gemma window — is what lets a chat-driven
    plan SEE prior turns instead of running memoryless. It is injected once, at the
    single goal the whole live path already shares, so no downstream component
    needs a new parameter.

    A blank/absent context returns the bare current request UNCHANGED, so a
    thread's FIRST turn (and any caller that passes no context) is byte-identical
    to the pre-a2 behaviour — no regression, and never an empty header block."""
    q = agentic_goal(query)
    ctx = (conversation_context or "").strip()
    if not ctx:
        return q
    return (
        f"{_PRIOR_CONTEXT_HEADER}\n{ctx}\n\n{_CURRENT_REQUEST_HEADER}\n{q}"
    )


def _nonempty_output(node, result) -> Optional[str]:
    """A minimal, format-AGNOSTIC result check: a node must produce some output.

    Replaces the old md/html-specific ``_report_validator`` (the nodes are no
    longer hard-coded markdown/HTML writers). The full in_progress→verifiable→done
    lifecycle gate + inline reviewer-fix on every node path is b3's dedicated
    job; b2 only keeps the runtime from silently accepting an empty node."""
    out = (getattr(result, "output", "") or "").strip()
    return None if out else "empty output"


@dataclass
class AgenticResult:
    """The outcome of one live agentic run for a chat message (s3/b2 normalized).

    The route handlers read the NORMALIZED fields (``final_response`` / ``outputs``
    / ``states`` / ``launch_order`` / ``ok`` / ``artifacts``) so the acyclic and
    deep-research paths surface uniformly. ``result`` (a
    :class:`~agent_runtime.RuntimeResult`) / ``deep_research`` carry the raw run
    for callers that want it; ``md_report`` / ``html_report`` are kept (best-effort
    populated from the final deliverable) so the s6 workflow producer keeps
    working unchanged."""

    dag: Any = None
    result: Any = None                  # RuntimeResult (acyclic OR deep-research) or None
    deep_research: Any = None           # cyclic-run summary dict (a3) or None
    md_report: Optional[str] = None     # back-compat: s6 workflow reads this
    html_report: Optional[str] = None
    rationale: str = ""
    shape: Optional[str] = None
    escalated: bool = False
    final_response: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    states: dict[str, dict] = field(default_factory=dict)
    launch_order: list[str] = field(default_factory=list)
    ok: bool = False
    # (filename, mime, body) tuples the route persists as downloadable artifacts.
    artifacts: list[tuple[str, str, str]] = field(default_factory=list)
    # MISSING-SPECIALIST PAUSE (s4 M1, RC8): True when the run could NOT proceed as
    # authored because a node needs a specialist no registered spec provides — the
    # run is PAUSED (the runtime was NOT driven) and the user must pick a CHOICE.
    # ``pending`` carries the notify/choice payload (resume_token + choices +
    # missing nodes) the chat surfaces; ``resume_agentic`` consumes it. ``ok`` is
    # False while paused (nothing ran) — this is an explicit, visible pause, never
    # a silent raw-LLM fallthrough.
    missing_specialist: bool = False
    pending: Optional[dict[str, Any]] = None
    # AMBIGUITY CLARIFICATION PAUSE (scenario-2): True when the planner judged the
    # request too underspecified to act on and is asking the user a clarifying
    # question BACK. Like the missing-specialist pause this is an explicit, visible
    # pause — nothing ran, ``ok`` is False, and ``pending`` carries the question +
    # resume_token the chat surfaces; the run resumes (on the user's answer) by
    # re-driving ``run_agentic`` with the answer folded into the goal. Never a
    # silent guess of the missing detail.
    needs_clarification: bool = False


# --------------------------------------------------------------------------- #
# the live entrypoint — shape selection → planner-derived DAG → live runtime
# --------------------------------------------------------------------------- #
async def run_agentic(
    query: str,
    *,
    transport,
    registry: SpecRegistry,
    hook: ToolHook,
    plane: EventPlane,
    timeout: float = 900.0,
    run_id: Optional[str] = None,
    max_iter_override: Optional[int] = None,
    shape_config: Optional[Any] = None,
    conversation_context: Optional[str] = None,
    clarification: Optional[str] = None,
    skip_ambiguity: bool = False,
) -> AgenticResult:
    """Drive the REAL Round-3 agent for ``query`` and return the normalized result.

    Selects the plan SHAPE for the query (Gemma structured enum, s3/b1), then
    either drives the planner-derived acyclic DAG on the live runtime (honouring
    the shape's execution discipline) or runs the bounded cyclic deep-research
    executor — NEVER the old fixed stub DAG / two-report rewrite.

    CONVERSATION MEMORY (s5/a2): ``conversation_context`` is the bounded prior-turn
    context block for THIS chat thread (assembled by the s5/a1
    :class:`~chat_app.conversation_memory.ConversationMemory`, strictly scoped to
    the chat_id). When supplied it is prefixed onto the planning GOAL via
    :func:`goal_with_context`, so shape selection, ``Planner.plan`` and every
    derived node see the prior turns and the run CONTINUES the conversation rather
    than answering memorylessly. A blank/absent value leaves the goal unchanged.

    SHAPE-CONFIG OVERRIDE (s4/a4, d5): ``shape_config`` is the UI's per-shape
    ``max_iter`` store (a :class:`~chat_app.shape_config.ShapeConfigStore`, or any
    object with ``get_max_iter(name) -> Optional[int]``). When supplied — and no
    explicit ``max_iter_override`` was passed by the caller — the SELECTED shape's
    stored override is read here, right after selection, and threaded into the
    deep-research unroll. So a value set on the Shapes screen (persisted to SQLite),
    NOT the shape file's default, bounds the rounds the runtime runs. An explicit
    ``max_iter_override`` still wins (e.g. a focused test); the executor clamps
    whatever it gets to the shape's ``hard_cap``.

    TRACING (s6/b2): an outer ``agent.session`` span wraps shape-selection +
    planning + running so the captured Phoenix trace gathers them under one root
    (the planner span + the agent.run span with its per-node + Gemma spans). The
    ``run_id`` threads through to ``runtime.run`` so the trace correlates back to
    the ``/runs/{run_id}`` the client polls."""
    # s5/a2: fold the bounded per-chat prior-turn context into the single goal the
    # shape selector + planner + nodes all reason over (no-op when context blank).
    query = goal_with_context(conversation_context, query)

    # CLARIFICATION RESUME (scenario-2): the user has answered the planner's
    # clarifying question — fold the answer into the goal so the shape selector,
    # planner and every node act on the CLARIFIED intent, and skip re-asking. The
    # block is clearly delimited so the model treats it as the resolved detail, not
    # a new instruction.
    clarification = (clarification or "").strip()
    if clarification:
        query = (
            f"{query}\n\nCLARIFICATION (you asked the user for a missing detail; "
            f"they answered — plan on this clarified intent):\n{clarification}"
        )
        skip_ambiguity = True

    tracer = get_tracer("chat_app.agentic")
    with tracer.start_as_current_span("agent.session") as session_span:
        session_span.set_attribute("session.query", query[:500])
        if run_id:
            session_span.set_attribute("session.run_id", str(run_id))

        # 0) AMBIGUITY GATE (scenario-2 clarification turn) — BEFORE any authoring,
        # the planner JUDGES whether the request is too underspecified to act on
        # without guessing a load-bearing detail. If so the run PAUSES and asks the
        # user ONE clarifying question back (mirrors the missing-specialist pause);
        # it resumes by re-calling run_agentic with the answer folded in (above).
        # Fail-open: assess_ambiguity returns "not ambiguous" on any malformed /
        # schema-blind reply, so a non-ambiguous request (and the offline seam,
        # which never calls this) proceeds exactly as before — no regression.
        if not skip_ambiguity:
            ambiguity = await _assess_ambiguity(
                query, transport=transport, registry=registry, hook=hook
            )
            if ambiguity.needs_clarification:
                resume_token = f"resume-{uuid.uuid4().hex[:12]}"
                payload = clarification_payload(
                    ambiguity.question, resume_token=resume_token, original_query=query
                )
                session_span.set_attribute("session.needs_clarification", True)
                # Publish on the chat's plane so the SSE stream NOTIFIES the user
                # live with the question (parity with the missing-specialist notify).
                await plane.publish(
                    EVENT_NEEDS_CLARIFICATION, dict(payload), source="chat_app.agentic"
                )
                return AgenticResult(
                    rationale=ambiguity.rationale,
                    ok=False,
                    needs_clarification=True,
                    pending=payload,
                )

        # 1) SHAPE SELECTION (s3/b1) — the one model-driven choice before authoring.
        # F5: the selector is given the registered spec names so it can extract, in
        # the SAME structured call, two intent signals the router was previously
        # blind to — ``search_allowed`` (may this request use the web?) and
        # ``requested_specs`` (which listed specializations did the user name?). The
        # model reads these from the goal (intent-faithful across phrasings), and the
        # routing below ENFORCES them structurally.
        selector = ShapeSelector(transport, spec_names=registry.names())
        catalog = selector.catalog()
        try:
            selection = await selector.select(query)
        except MalformedOutputError as exc:
            # A failed/low-confidence selection is an explicit signal, never a
            # silent mis-pick: fall back to the default acyclic path below (web
            # allowed, no named spec — the permissive default).
            selection = ShapeSelection(
                shape=None, escalate=True, rationale=f"shape selection failed: {exc}"
            )
        shape_spec = catalog.get(selection.shape) if selection.shape else None
        # F5 routing signals enforced below. Defensively re-filter requested_specs to
        # currently-registered specs (the selection may have been built on the failure
        # fallback, or names may have changed) so only a real spec ever binds.
        allow_web = bool(selection.search_allowed)
        registered = set(registry.names())
        requested_specs = [s for s in selection.requested_specs if s in registered]
        # STRUCTURAL MISSING-SPECIALIST TRIGGER (scenario-3, s10-a8). The selector
        # reliably extracts the specialization name(s) the user asked for; a name the
        # user requested that is NOT registered is the missing-specialist signal. The
        # TRIGGER is this DETERMINISTIC set-difference (not the per-node free-text
        # ``needs_spec`` the 4.6B model would not volunteer — s10-a4 — and not a
        # keyword/scenario match). A request needing an unavailable specialist is
        # routed to the ACYCLIC path below (so a define-and-resume has a terminal node
        # to stamp the newly-defined spec onto — the fileless deep-research shape is
        # the wrong target) and PAUSED for the user CHOICE before anything runs.
        unmet_specs = [s for s in selection.unmet_specs if s not in registered]
        session_span.set_attribute("session.shape", selection.shape or "escalate")
        session_span.set_attribute("session.escalate", selection.escalate)
        session_span.set_attribute("session.search_allowed", allow_web)
        session_span.set_attribute("session.requested_specs", requested_specs)
        session_span.set_attribute("session.unmet_specs", unmet_specs)

        # SHAPE-CONFIG OVERRIDE (s4/a4, d5): read the UI-set per-shape max_iter from
        # the store for the SELECTED shape, so the runtime honors the value the user
        # set on the Shapes screen (persisted to SQLite) instead of the text-file
        # default. An explicit caller-supplied max_iter_override still wins; the
        # executor clamps whatever it gets to the shape's hard_cap.
        if max_iter_override is None and shape_config is not None and selection.shape:
            max_iter_override = shape_config.get_max_iter(selection.shape)
            if max_iter_override is not None:
                session_span.set_attribute("session.max_iter_override", int(max_iter_override))

        # 2) CYCLIC (deep-research) → UNROLL the declarative template into a bounded
        # acyclic role-tagged DAG and drive it on the SAME generic runtime (a3). The
        # route keys off the shape's DECLARATIVE fields (does it declare round/final
        # roles → is_unrollable), NOT a hard-coded shape name, so adding a cyclic
        # shape is adding one text file (plug-n-play).
        #
        # F5 'DO NOT SEARCH' GUARD: the deep-research family is INHERENTLY a web
        # research shape (every research round searches+fetches). If the model
        # selected it BUT the request forbids the web (``allow_web`` False — a
        # contradiction the model occasionally emits, or a phrasing that over-routed),
        # we do NOT run the search shape: fall through to the acyclic path with the
        # web tools stripped, honoring the constraint over the topology pick.
        #
        # FILE-OUTPUT GUARD (d11/s7-a2 invariant, s10-a4): the deep-research family is
        # also inherently FILELESS — it streams research + synthesis but authors no
        # output node, so it can never satisfy a request that asks for the result
        # WRITTEN TO A FILE. When the model picked deep-research for such a request
        # (a 'research X … and write it as a markdown file' that over-routed on the
        # 'research' phrasing), we suppress it the same way as the no-search guard and
        # fall through to the acyclic path, which authors a terminal ``file_write``
        # node and produces a real, relatable-named workspace file (d11). The intent
        # signal is model-extracted (``wants_file``), not a keyword match (anti-rig).
        wants_file = bool(selection.wants_file)
        session_span.set_attribute("session.wants_file", wants_file)
        # MISSING-SPECIALIST GUARD (a8): when the user requested an unavailable
        # specialization, suppress the inherently-fileless/streamed deep-research
        # shape and fall through to the acyclic path — that path authors a terminal
        # answer/output node, which the missing-specialist pause below attaches the
        # unmet need to (so define-and-resume can stamp the newly-defined spec there).
        if (
            shape_spec is not None
            and shape_spec.is_unrollable
            and allow_web
            and not wants_file
            and not unmet_specs
        ):
            return await _run_deep_research(
                query,
                shape_spec,
                selection,
                transport=transport,
                registry=registry,
                hook=hook,
                plane=plane,
                timeout=timeout,
                run_id=run_id,
                max_iter_override=max_iter_override,
                requested_specs=requested_specs,
            )

        # 3) Acyclic shapes (linear / modular-parallel) or escalate → planner DAG.
        # When the deep-research shape was suppressed by the no-search guard above,
        # there is no acyclic shape discipline to honor → run as the default
        # concurrent acyclic plan (shape_spec=None) with web tools stripped.
        acyclic_shape = None if (shape_spec is not None and shape_spec.is_unrollable) else shape_spec
        return await _run_acyclic(
            query,
            acyclic_shape,
            selection,
            transport=transport,
            registry=registry,
            hook=hook,
            plane=plane,
            timeout=timeout,
            run_id=run_id,
            allow_web=allow_web,
            requested_specs=requested_specs,
            # a8: the user-requested specializations that are NOT registered — the
            # acyclic gate synthesizes the missing-specialist notify for them.
            unmet_specs=unmet_specs,
            # s5/a4: the acyclic planner PARAPHRASES the goal into node tasks, so the
            # answer node lost the concrete prior-turn facts even though the goal
            # carried them. Thread the raw bounded context to the runtime so every
            # node grounds in the thread. (The deep-research path already injects the
            # context-laden goal verbatim into each node's user turn, so it needs no
            # equivalent — only the paraphrasing acyclic path did.)
            conversation_context=conversation_context,
        )


async def _assess_ambiguity(
    goal: str, *, transport, registry: SpecRegistry, hook: ToolHook
):
    """Ask the planner whether ``goal`` is too underspecified to act on (s-2).

    Builds a lean planner over the SAME body-free factory the authoring planner
    uses (registry lookup + tool catalog; d10) and runs its native structured
    ambiguity assessment. Fail-open: any error returns a "not ambiguous" decision
    so a clear request — and any transport that does not understand the schema —
    proceeds unchanged. No plan is authored here; this is the single decision
    BEFORE shape selection."""
    factory = AbstractPlanFactory(registry.index(), tool_catalog=hook.registry.catalog())
    planner = Planner(transport, factory)
    return await planner.assess_ambiguity(goal)


def _build_acyclic_runtime(
    *,
    transport,
    registry: SpecRegistry,
    hook: ToolHook,
    plane: EventPlane,
    shape_spec,
    conversation_context: Optional[str] = None,
    allow_web: bool = True,
) -> tuple[AgentRuntime, Planner]:
    """Build the (runtime, planner) pair for the acyclic live path.

    Extracted so the initial run AND a missing-specialist RESUME construct an
    identical runtime (same lifecycle gate, self-heal, tool surface, call opts) —
    the resume must re-derive nothing about HOW the DAG runs, only WHICH DAG.

    F5 ``allow_web``: when False (the user forbade searching), the web tools are
    stripped from the SELF-HEAL re-planner's schema too — so a heal/replan can never
    re-introduce a web tool the initial authoring was forbidden to bind — and the
    search-then-read follow-through is disabled. Default True = the pre-F5 surface."""
    factory = AbstractPlanFactory(registry.index(), tool_catalog=hook.registry.catalog())
    offered_tools = _filter_web_tools(
        [t["name"] for t in hook.registry.catalog() if t["name"] in OFFERED_TOOLS],
        allow_web,
    )
    plan_schema = build_plan_schema(registry.names(), offered_tools)
    # gemma4 is a thinking model — think=False so the CoT trace does not eat the
    # token budget and leave EMPTY content; temp 0 deterministic; the richer
    # format=plan_schema (valid spec/tool enums, the N-spec ``specs`` array and the
    # free-text ``needs_spec`` missing-signal — s4 M1) is the proven s8 path.
    planner = Planner(
        transport,
        factory,
        call_opts={
            "think": False,
            "temperature": 0,
            "format": plan_schema,
            "max_tokens": 2048,
        },
    )
    # The shape's execution DISCIPLINE (s3/b1): linear → SEQUENTIAL (strict
    # single-file), modular-parallel → CONCURRENT (ready-wave). An escalate /
    # unknown shape falls back to CONCURRENT (the legacy runtime behaviour).
    mode = (
        execution_mode_for(shape_spec.execution)
        if shape_spec is not None
        else ExecutionMode.CONCURRENT
    )

    # Per-run hook bound to THIS chat's plane (reusing the SAME tool registry) so
    # the agent's live tool calls stream to this chat's SSE overlay.
    run_hook = ToolHook(plane, registry=hook.registry)

    async def replanner_adapter(node, err, completed):
        return await planner.replan_subgraph(
            node.task, err, spec=node.primary_spec, completed=completed
        )

    # REACTIVE SELF-HEAL (b4, blueprint §2e, d1): route a node FAILURE through the
    # planner's heal DECISION (Planner.heal_decision enum retry|pivot|extend|abort)
    # so recovery is automatic on the live runtime — retry re-dispatches the node,
    # pivot/extend re-derive a corrective sub-DAG (the replanner above), abort
    # surfaces. The planner owns the decision; the runtime enacts; the registered
    # LambdaRegistry rule only observes the routing.
    heal_router = HealRouter(planner)

    runtime = AgentRuntime(
        transport=transport,
        loader=SpecLoader(registry),
        hook=run_hook,
        plane=plane,
        # SEQUENTIAL also caps concurrency at 1 (belt-and-suspenders over the
        # scheduler's single-file dispatch); CONCURRENT is unbounded fan-out.
        max_concurrency=1 if mode == ExecutionMode.SEQUENTIAL else None,
        replanner=replanner_adapter,
        result_validator=_nonempty_output,
        # LIFECYCLE GATE (b3, d2): wire the in_progress→verifiable→done verify
        # gate onto this (live acyclic) node path so EVERY node crosses VERIFIABLE
        # with the same-spec reviewer ENCOURAGED to read the output and FIX IT
        # INLINE — not the old verifier=None trivial-pass that left the inline
        # reviewer-fix unreachable. The default gate is conservative (it rejects
        # only clearly-unusable output) so a real answer always passes.
        verifier=default_node_verifier,
        tool_arg_emitter=SchemaToolArgEmitter(transport, max_tokens=256),
        # d13 SEARCH-THEN-READ (a4): an open-shape node that runs ``web_search``
        # (e.g. a modular-parallel news step) FOLLOWS THROUGH and web_fetches the
        # top real result URLs, so the node summarises ACTUAL article content, not
        # the search-results page (the "describes the source list" failure d13
        # targets). web_fetch is actually invoked on real upstream URLs here too.
        # F5: 0 when the web is disallowed — a belt-and-suspenders over the stripped
        # tool enum (no web_search can be bound, so nothing to follow through).
        read_search_max_fetch=3 if allow_web else 0,
        # think=False on the producer nodes too (same gemma4 CoT risk); temp 0.4
        # for prose variety; raised num_predict so content is not truncated.
        subagent_call_opts={"num_predict": 1400, "temperature": 0.4, "think": False},
        max_heals=4,
        max_replans=2,
        execution=mode,
        lambda_registry=getattr(hook, "subscriptions", None),
        heal_router=heal_router,
        # CONVERSATION MEMORY (s5/a4): hand the bounded prior-turn context to the
        # runtime so EVERY node's sub-agent grounds its answer in the thread — not
        # just the planner (whose paraphrased node tasks dropped the concrete facts,
        # making the answer node hallucinate). None/blank => memoryless as before.
        conversation_context=conversation_context,
    )
    return runtime, planner


def _build_incremental_planner(
    *,
    transport,
    registry: SpecRegistry,
    hook: ToolHook,
    shape_spec,
    allow_web: bool = True,
    requested_specs: Optional[list[str]] = None,
) -> IncrementalPlanner:
    """Build the seed-then-fill authorer for the live acyclic path (d3 port).

    Mirrors :func:`_build_acyclic_runtime`'s planner construction — the SAME
    body-free factory (registry lookup + tool catalog; d10) and the SAME offered
    enum vocabulary (the ``OFFERED_TOOLS`` present on the hook + the registered spec
    names, identical to :func:`build_plan_schema`) — but authors the DAG node BY
    node instead of one-shotting it. The selected shape (its name + description) is
    threaded in so the per-node calls author edges that fit the shape (parallel vs
    chained). The resulting :class:`~agent_runtime.PlanResult` is a drop-in for the
    one-shot ``Planner.plan`` the caller would otherwise use."""
    factory = AbstractPlanFactory(registry.index(), tool_catalog=hook.registry.catalog())
    # F5: drop the web tools from the authoring enum when the request forbids the
    # web — a node then CANNOT bind web_search/web_fetch (the structural zero-web
    # guarantee). Default True = the full pre-F5 surface.
    offered_tools = _filter_web_tools(
        [t["name"] for t in hook.registry.catalog() if t["name"] in OFFERED_TOOLS],
        allow_web,
    )
    # F2 DEFAULT RESEARCH SPEC: the generic research specialization the authorer
    # stamps onto any null-spec GATHER node (web_search/web_fetch) so no parallel
    # sibling ships an ungrounded source-list-summary section (d13). It is the SAME
    # reused spec the deep-research shape uses (``research-analyst``) — passed in by
    # NAME (the generic authorer hard-codes none), and only when it is actually
    # registered (else the authorer no-ops the pass).
    default_research_spec = DEEP_RESEARCH_SPEC if DEEP_RESEARCH_SPEC in registry else ""
    return IncrementalPlanner(
        transport,
        factory,
        spec_names=registry.names(),
        tool_names=offered_tools,
        shape_name=(shape_spec.name if shape_spec is not None else ""),
        shape_description=(shape_spec.description if shape_spec is not None else ""),
        default_research_spec=default_research_spec,
        # F5: the user-NAMED specialization(s) the authorer must bind (told + a
        # terminal-node finalization guarantee inside IncrementalPlanner).
        requested_specs=requested_specs or [],
    )


async def _run_acyclic(
    query: str,
    shape_spec,
    selection: ShapeSelection,
    *,
    transport,
    registry: SpecRegistry,
    hook: ToolHook,
    plane: EventPlane,
    timeout: float,
    run_id: Optional[str],
    conversation_context: Optional[str] = None,
    allow_web: bool = True,
    requested_specs: Optional[list[str]] = None,
    unmet_specs: Optional[list[str]] = None,
) -> AgenticResult:
    """Planner self-derives the DAG; the live runtime drives it for the shape's mode.

    MISSING-SPECIALIST GATE (s4 M1, RC8): after the planner authors the DAG and
    BEFORE the runtime is driven, the plan is checked for nodes that DECLARED a
    needed specialist (``needs_spec``) that no registered specialization provides.
    If any exist the run is PAUSED — the runtime is NOT driven — and a
    notify+CHOICE (:data:`EVENT_MISSING_SPECIALIST`) is published on the plane (so
    it streams to the chat). The user picks ``sse_fallback`` or
    ``define_and_resume`` and the run continues via :func:`resume_agentic`. This is
    the explicit, visible alternative to the old silent ``spec=""`` raw-LLM
    fallthrough."""
    # The one-shot Planner is still built inside (it wires the self-heal re-plan +
    # heal decision onto the runtime); it just no longer authors the initial DAG.
    runtime, _planner = _build_acyclic_runtime(
        transport=transport, registry=registry, hook=hook, plane=plane,
        shape_spec=shape_spec, conversation_context=conversation_context,
        allow_web=allow_web,
    )
    # INCREMENTAL SEED-THEN-FILL AUTHORING (d3 re-architecture, the literal
    # eda-base3 port). EVERY authored acyclic shape — linear, modular-parallel, and
    # the escalate/no-shape fallback — is authored by the SINGLE generic incremental
    # authorer (a3): the DAG is filled node-by-node, each node a small reliable
    # native structured Gemma decision seeing goal + shape + already-authored nodes
    # (the create_plan→add_action-per-node mechanism from eda-base3's plan_fsm),
    # which dissolved the parallel-multi-tool shortfall WITHOUT a stronger model or
    # few-shot rigging (a2). There is now ONE authoring mechanism, not a per-shape
    # one-shot vs incremental split — the shape's name+description thread in so the
    # per-node calls author edges that fit the shape (parallel vs chained), while
    # the runtime's execution discipline (linear=SEQUENTIAL) still enforces
    # single-file dispatch regardless. Everything downstream — missing-spec
    # detection, the d11 SchemaToolArgEmitter tool-arg grounding, the lifecycle
    # gate, the runtime — consumes the resulting PlanDAG unchanged. (The one-shot
    # Planner is still built above for the self-heal sub-graph re-plan + the heal
    # decision; it just no longer authors the initial DAG.)
    authoring_planner = _build_incremental_planner(
        transport=transport, registry=registry, hook=hook, shape_spec=shape_spec,
        allow_web=allow_web, requested_specs=requested_specs,
    )
    plan_result = await authoring_planner.plan(query)
    dag = plan_result.dag

    # MISSING-SPECIALIST DETECTION + NOTIFY (s4 M1 + s10-a8): a node that needs a
    # specialist the registry cannot supply pauses the run with a user CHOICE —
    # never silent. TWO trigger sources, unified here:
    #   * per-node ``needs_spec`` the planner VOLUNTEERED during authoring (s4 M1);
    #   * the STRUCTURAL a8 trigger — a specialization the user REQUESTED (extracted
    #     by the shape selector) that is not registered, synthesized onto the DAG's
    #     sink node(s). This is the reliable path: the 4.6B model does not volunteer
    #     the per-node needs_spec (s10-a4), but the deterministic registry-membership
    #     check on the selector's reliable extraction always fires.
    # Merged + deduped by node id (a node already flagged by needs_spec is not
    # double-listed); both reach the SAME notify + resume mechanism unchanged.
    missing = detect_missing_specialists(dag, registry.names())
    if unmet_specs:
        flagged = {m.node_id for m in missing}
        missing = missing + [
            m
            for m in missing_from_requested(dag, unmet_specs, registry.names())
            if m.node_id not in flagged
        ]
    if missing:
        resume_token = f"resume-{uuid.uuid4().hex[:12]}"
        payload = missing_specialist_payload(missing, resume_token=resume_token)
        # Publish on the chat's plane so the SSE stream NOTIFIES the user live.
        await plane.publish(
            EVENT_MISSING_SPECIALIST, dict(payload), source="chat_app.agentic"
        )
        return AgenticResult(
            dag=dag,
            shape=selection.shape,
            escalated=selection.escalate,
            rationale=(selection.rationale or dag.rationale or ""),
            ok=False,
            missing_specialist=True,
            pending=payload,
        )

    result = await runtime.run(dag, timeout=timeout, run_id=run_id)
    return _agentic_from_runtime(
        dag,
        result,
        shape=selection.shape,
        escalated=selection.escalate,
        rationale=(selection.rationale or dag.rationale or ""),
    )


async def resume_agentic(
    dag: PlanDAG,
    choice: str,
    *,
    transport,
    registry: SpecRegistry,
    hook: ToolHook,
    plane: EventPlane,
    missing: Optional[list[dict[str, Any]]] = None,
    defined_specs: Optional[dict[str, str]] = None,
    shape_spec=None,
    shape: Optional[str] = None,
    rationale: str = "",
    timeout: float = 900.0,
    run_id: Optional[str] = None,
    conversation_context: Optional[str] = None,
) -> AgenticResult:
    """RESUME a missing-specialist-paused plan with the user's chosen resolution.

    ``choice`` is one of :data:`~agent_runtime.MISSING_SPEC_CHOICES`:

    * ``sse_fallback`` — the paused nodes' ``needs_spec`` is cleared so they run
      spec-less (raw LLM), their output streaming to the chat (the user accepts a
      generic answer for those steps);
    * ``define_and_resume`` — ``defined_specs`` (``node_id -> spec_name``, or the
      ``""`` key to apply one newly-defined spec to every paused node) is stamped
      onto the paused nodes, which now run specialized.

    The SAME runtime construction as the initial run drives the rewritten DAG, so
    the lifecycle gate / self-heal / tool surface are identical — only the DAG's
    spec bindings changed. The ``missing`` list (the pending payload's nodes) tells
    the rewrite which nodes to resolve; if omitted, every node still carrying
    ``needs_spec`` is resolved.

    CONVERSATION MEMORY ON RESUME (s7/a1, a6): the initial :func:`_run_acyclic`
    threads the bounded prior-turn ``conversation_context`` into the runtime so
    every node's sub-agent grounds in the thread (the s5/a4 executing-path fix).
    A missing-specialist RESUME re-drives the SAME paused nodes, so it must carry
    the SAME context or those nodes would run memoryless — a fidelity gap on the
    resume path (it affects o6 scenario 3, whose missing-specialist resolution —
    SSE-fallback OR define-and-resume — flows through here). The caller stashes the
    initial turn's context in ``pending_runs`` and passes it back here; it is
    threaded into :func:`_build_acyclic_runtime` exactly as the initial run did.
    None/blank => memoryless as before (first-turn / no-context resumes are
    byte-identical to the pre-fix behaviour)."""
    if choice not in MISSING_SPEC_CHOICES:
        raise ValueError(
            f"unknown resume choice {choice!r}; expected one of "
            f"{list(MISSING_SPEC_CHOICES)}"
        )
    from agent_runtime import MissingSpecialist

    if missing:
        ms = [
            MissingSpecialist(
                node_id=m["node_id"], task=m.get("task", ""),
                needs=m.get("needs", ""), role=m.get("role"),
            )
            for m in missing
        ]
    else:
        ms = detect_missing_specialists(dag, [])  # any node carrying needs_spec

    resolved_dag = apply_missing_spec_resolution(
        dag, ms, choice=choice, defined_specs=defined_specs
    )
    runtime, _planner = _build_acyclic_runtime(
        transport=transport, registry=registry, hook=hook, plane=plane,
        shape_spec=shape_spec,
        # a6 fix: thread the initial turn's bounded prior-turn context so the
        # resumed nodes ground in the conversation exactly as the initial run did.
        conversation_context=conversation_context,
    )
    result = await runtime.run(resolved_dag, timeout=timeout, run_id=run_id)
    return _agentic_from_runtime(
        resolved_dag,
        result,
        shape=shape,
        escalated=False,
        rationale=rationale or resolved_dag.rationale or "",
    )


def _agentic_from_runtime(
    dag: PlanDAG,
    result,
    *,
    shape: Optional[str],
    escalated: bool,
    rationale: str,
) -> AgenticResult:
    """Normalize a :class:`~agent_runtime.RuntimeResult` into an :class:`AgenticResult`."""
    outputs = {
        nid: _strip_fence((r.output or ""))[:600] for nid, r in result.results.items()
    }
    states = {
        nid: {
            "status": str(st.get("status")),
            "attempts": int(st.get("attempts", 0)),
            "error": st.get("error"),
        }
        for nid, st in result.states.items()
    }
    final = _final_output(dag, result)
    return AgenticResult(
        dag=dag,
        result=result,
        md_report=(final or None),
        rationale=rationale,
        shape=shape,
        escalated=escalated,
        final_response=final,
        outputs=outputs,
        states=states,
        launch_order=list(result.launch_order),
        ok=bool(result.ok),
        artifacts=_artifacts_for(final),
    )


def _offline_canned_plan(query: str) -> dict[str, Any]:
    """A spec-less, deterministic plan the OFFLINE seam feeds through ``Planner.plan``.

    RC1/RC2: ``_demo_dag`` (a hand-built :class:`PlanDAG` that bypassed the planner
    AND ran regardless of mode, pretending to be the agent) is GONE. The offline
    path now flows ``message → Planner.plan → AgentRuntime`` EXACTLY as the live
    path does — only the transport is the deterministic fake (the d12 pluggable
    seam). Spec-less so the offline harness needs no registry lookups or GPU."""
    subject = (query or "").strip()[:200]
    return {
        "rationale": "analyze the request, then answer it",
        "nodes": [
            {"id": "analyze", "task": f"analyze the request: {subject}", "depends_on": []},
            {
                "id": "answer",
                "task": f"answer the request: {subject}",
                "depends_on": ["analyze"],
            },
        ],
    }


async def run_offline(
    query: str,
    *,
    registry: SpecRegistry,
    hook: ToolHook,
    plane: EventPlane,
    run_id: Optional[str] = None,
    conversation_context: Optional[str] = None,
) -> AgenticResult:
    """The OFFLINE (stub) seam (d12): the planner-derived DAG on the stub transport.

    The SAME ``Planner.plan → AgentRuntime`` pipeline the live path uses, driven on
    the deterministic :mod:`agent_runtime.stub` transport so the app works with NO
    Ollama / GPU. This is NOT the old ``_demo_dag`` (a fixed runtime DAG that
    bypassed the planner): the request flows through the real planner, only the
    transport is fake.

    CONVERSATION MEMORY (s5/a2): like :func:`run_agentic`, ``conversation_context``
    (the s5/a1 bounded per-chat prior-turn block) is prefixed onto the planning
    goal via :func:`goal_with_context`, so the offline seam is conversation-aware
    too and the wiring is identical on both transports."""
    from agent_runtime import stub

    query = goal_with_context(conversation_context, query)
    factory = AbstractPlanFactory(registry.index(), tool_catalog=hook.registry.catalog())
    planner = Planner(stub.valid_plan_transport(plan=_offline_canned_plan(query)), factory)
    run_hook = ToolHook(plane, registry=hook.registry)
    runtime = AgentRuntime(
        transport=stub.subagent_transport(),
        loader=SpecLoader(registry),
        hook=run_hook,
        plane=plane,
        # LIFECYCLE GATE (b3, d2): the OFFLINE node path built neither a validator
        # nor a verifier, so its nodes crossed the gate trivially. Wire the SAME
        # in_progress→verifiable→done gate + empty-output validator here so the
        # offline (non-agentic) path is lifecycle-faithful too — every node goes
        # through VERIFIABLE with the inline reviewer-fix reachable.
        result_validator=_nonempty_output,
        verifier=default_node_verifier,
    )
    plan_result = await planner.plan(query)
    dag = plan_result.dag
    result = await runtime.run(dag, run_id=run_id)
    return _agentic_from_runtime(
        dag, result, shape="offline", escalated=False, rationale=dag.rationale
    )


async def _run_deep_research(
    query: str,
    shape_spec,
    selection: ShapeSelection,
    *,
    transport,
    registry: SpecRegistry,
    hook: ToolHook,
    plane: EventPlane,
    timeout: float = 900.0,
    run_id: Optional[str] = None,
    max_iter_override: Optional[int],
    requested_specs: Optional[list[str]] = None,
) -> AgenticResult:
    """Drive the cyclic deep-research shape on the GENERIC runtime (a3 re-arch).

    The shape is UNROLLED (declaratively, by :func:`~agent_runtime.unroll_shape`)
    into a bounded acyclic role-tagged DAG honoring the UI-set ``max_iter`` (d5),
    and that DAG is driven by the SAME :class:`~agent_runtime.AgentRuntime` as every
    other shape — there is no per-shape executor. ONE specialization (``§2c``) is
    bound to every node so only the node ROLE differs; the unroll's
    growing-visibility edges make each round depend on every prior node, so the
    runtime's upstream-inputs threading hands each node all earlier layers."""
    # The ONE specialization reused across every round (§2c). Seeded at boot (RC3);
    # fall back to no spec (role framing only) if it is somehow absent.
    #
    # F5 NAMED-SPEC HONORED: the deep-research shape reuses a SINGLE specialization
    # across all rounds (by design). If the user EXPLICITLY named a (registered)
    # specialization, that is the reused spec — instead of the hard-coded
    # research-analyst default that previously made a user-named output spec
    # structurally unreachable on this route (the F5(i) failure). Falls back to the
    # research-analyst default when the user named none.
    spec_name = _deep_research_spec(registry, requested_specs)
    effective_max_iter = shape_spec.effective_max_iter(max_iter_override)

    # UNROLL the declarative template → a bounded acyclic role-tagged DAG (generic).
    dag = unroll_shape(
        shape_spec, query, spec=spec_name, max_iter_override=max_iter_override
    )

    # d13 (a4): the research ROLE layers must READ real sources, not describe a
    # search-results page. A per-run hook (bound to THIS chat's plane, reusing the
    # shared registry's web_search/web_fetch) is wired so the runtime's research
    # executor can search → web_fetch real article URLs → fold the EXTRACTED text
    # into each layer (``read_search_max_fetch`` enables it). This replaces a3's
    # tool-less role path; the deleted DeepResearchExecutor is NOT involved.
    run_hook = ToolHook(plane, registry=hook.registry)
    runtime = AgentRuntime(
        transport=transport,
        loader=SpecLoader(registry),
        hook=run_hook,
        plane=plane,
        # think=False (gemma4 CoT risk) + temp 0 (deterministic judgment) + a
        # generous num_ctx so a late round's growing-visibility context (all prior
        # layers threaded as inputs) PLUS the fetched article text is not truncated.
        subagent_call_opts={"think": False, "temperature": 0, "num_ctx": 16384},
        # d13 SEARCH-THEN-READ: each research layer web_fetches up to 3 real result
        # URLs (a round-rotating window → different sources per round) and grounds
        # its findings in the extracted article text. web_fetch is ACTUALLY invoked
        # on real upstream URLs (not skipped) — the d13 bar.
        read_search_max_fetch=3,
        # The role nodes' own per-role schema + judgment verdict-repair (in
        # SubAgent._run_role) carry quality; no extra acyclic gate is wired here so
        # the verify ROLE node is the verification (the lifecycle still passes
        # through VERIFIABLE trivially).
        execution=ExecutionMode.CONCURRENT,
    )
    result = await runtime.run(dag, timeout=timeout, run_id=run_id)

    outputs: dict[str, str] = {}
    for nid, res in result.results.items():
        rendered = _render_parsed(res.parsed) if res.parsed is not None else (res.output or "")
        outputs[nid] = _strip_fence(rendered)[:600]

    # The final answer = the synthesis role node's output (fall back to verify, then
    # the last research layer). Role is carried on the result so we find it directly.
    final = _final_role_output(dag, result, ROLE_SYNTHESIS) or _final_role_output(
        dag, result, ROLE_VERIFY
    ) or _final_role_output(dag, result, "research") or ""

    rounds_executed = sum(
        1 for nid in result.results if nid.endswith("_research")
    )
    return AgenticResult(
        dag=dag,
        result=result,
        # Back-compat summary of the (now generic) cyclic run for callers/UI/tests.
        deep_research={
            "shape": shape_spec.name,
            "effective_max_iter": effective_max_iter,
            "rounds_executed": rounds_executed,
            "dag": dag.as_dict(),
        },
        md_report=(final or None),
        rationale=selection.rationale,
        shape=selection.shape,
        escalated=selection.escalate,
        final_response=final,
        outputs=outputs,
        states=result.states,
        launch_order=result.launch_order,
        ok=bool(result.ok and rounds_executed > 0),
        artifacts=_artifacts_for(final),
    )


def _final_role_output(dag: PlanDAG, result, role: str) -> str:
    """The rendered output of the LAST node carrying ``role`` (topo order), or ''."""
    by_topo = dag.topo_order()
    for node in reversed(by_topo):
        if node.role != role:
            continue
        res = result.results.get(node.id)
        if res is None:
            continue
        # for_user: the final answer the chat surfaces must carry the CONTENT
        # only — never the judgment scaffold (a synthesis/verify node's
        # "**verdict:** fail" header is internal lifecycle metadata, not part of
        # the answer; F4: that prefix must not leak into the user-facing reply).
        rendered = (
            _render_parsed(res.parsed, for_user=True)
            if res.parsed is not None
            else (res.output or "")
        )
        rendered = _strip_fence(rendered)
        if rendered.strip():
            return rendered
    return ""


# --------------------------------------------------------------------------- #
# small rendering / extraction helpers
# --------------------------------------------------------------------------- #
def _strip_fence(text: str) -> str:
    """Strip a leading/trailing ``` code fence the small model sometimes wraps."""
    s = (text or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _final_output(dag: PlanDAG, result) -> str:
    """The plan's final deliverable: the last topo node with non-empty output.

    Walk the deterministic topo order from the END (the sink) and return the first
    node that produced text — that is the answer the chat surfaces. Falls back to
    any non-empty node output."""
    by_topo = dag.topo_order()
    for node in reversed(by_topo):
        r = result.results.get(node.id)
        if r is not None and (r.output or "").strip():
            return _strip_fence(r.output)
    for r in result.results.values():
        if (r.output or "").strip():
            return _strip_fence(r.output)
    return ""


def _render_parsed(parsed: Any, *, for_user: bool = False) -> str:
    """Render a role node's parsed structured output to readable text.

    A worker emits ``{output}``; research emits ``{findings,sources,...}``;
    synthesis/verify emit ``{verdict,findings,...}``. Surface the most useful
    human-readable view without leaking the raw JSON when avoidable.

    ``for_user`` renders the answer the chat surfaces: the judgment SCAFFOLD a
    synthesis/verify node carries (its ``**verdict:**`` header) is internal
    lifecycle metadata, not part of the answer, so it is OMITTED — otherwise a
    deep-research run's verify verdict (e.g. ``**verdict:** fail``) leaks into the
    user-facing reply (F4). The per-node debug ``outputs`` map keeps the verdict
    (``for_user`` defaults False)."""
    if parsed is None:
        return ""
    if isinstance(parsed, str):
        return _strip_fence(parsed)
    if isinstance(parsed, Mapping):
        if parsed.get("output"):
            return _strip_fence(str(parsed["output"]))
        findings = parsed.get("findings")
        if isinstance(findings, (list, tuple)) and findings:
            lines = [f"- {f}" for f in findings]
            verdict = parsed.get("verdict")
            if verdict and not for_user:
                lines.insert(0, f"**verdict:** {verdict}\n")
            return "\n".join(str(x) for x in lines)
        try:
            return json.dumps(parsed, indent=2, ensure_ascii=False)[:2000]
        except (TypeError, ValueError):
            return str(parsed)[:2000]
    return str(parsed)[:2000]


def _artifacts_for(final: str) -> list[tuple[str, str, str]]:
    """One downloadable artifact (``report.md``) for a non-empty final answer.

    Format-agnostic (the planner is no longer a hard-coded md+html writer): the
    final deliverable is surfaced as a single markdown report the chat can
    download. Empty → no artifact."""
    body = (final or "").strip()
    if not body:
        return []
    return [("report.md", "text/markdown; charset=utf-8", body)]


__all__ = [
    "run_agentic",
    "run_offline",
    "resume_agentic",
    "AgenticResult",
    "both_specs_registered",
    "agentic_goal",
    "goal_with_context",
    "build_plan_schema",
    "MD_SPEC",
    "HTML_SPEC",
    "OFFERED_TOOLS",
]
