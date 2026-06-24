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
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence

from reactive_tools import EventPlane, ToolHook
from reactive_tools.file_tools import mime_for_path
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
    PlannerReactor,
    ROLE_SYNTHESIZER,
    ROLE_WORKER,
    DagGrower,
    N4_TREE_DEPTH_CEILING,
    ResearchState,
    Tree,
    TreeConfig,
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
from agent_runtime.synth_tools import (
    UNSUPPORTED_SECTION_INSTRUCTION,
    assemble_html_spa,
    collapse_duplicate_sections,
    collapse_outline_duplicate_sections,
    collect_fetched_sources_full,
    derive_output_path,
    enforce_single_html_document,
    explicit_filename,
    html_close_gap,
    is_detailed_task,
    render_source_catalog,
    unwrap_output_envelope,
)
from agent_runtime.shapes import ShapeSpec, load_shape
from agent_runtime.factory import PlanNode

# Back-compat: the names the s6 workflow producer (chat_app.workflow) + existing
# specs/tests still reference. ``markdown-writer`` / ``html-writer`` are the two
# round-2 writer rulesets; ``both_specs_registered`` is still the s6 live-brief
# pre-req gate. b2 no longer routes the chat path on these (it routes on live mode
# + the derived shape), but the symbols stay so s6 + its tests keep working.
MD_SPEC = "markdown-writer"
HTML_SPEC = "html-writer"

# DEEP-RESEARCH BREADTH (s9/N1, d60/c15 part-a). The per-research-node fetch CAP — how
# many real articles a deep-research gather node may web_fetch and READ. Lifted from the
# legacy hard-wired ``3`` to a genuine breadth budget (~8-12) so a detailed sourced
# report grounds in MANY real sources, not three (the c15 "breadth is narrow by
# construction" gap). This is a NON-FLOW cost/safety CEILING, never a flow gate: the
# research agent still REASONS about whether/which sources to fetch and stops the moment
# it has read enough (c5/d49 ReAct loop) — this only raises the ceiling so genuine
# breadth is REACHABLE. The paired ReAct turn ceiling in agent_runtime rises
# proportionally (RESEARCH_SEARCH_HEADROOM) so the loop can search several angles then
# read many. Env-overridable (``RA_RESEARCH_FETCH_BREADTH``) for live tuning per d60/D-C
# without a code edit. Composes with the UNCHANGED c13 write side (same findings,sources
# contract — the global source list just gets richer).
DEEP_RESEARCH_FETCH_BREADTH = max(1, int(os.getenv("RA_RESEARCH_FETCH_BREADTH", "10")))

# D97 (s13/B1) — the REPORT path (:func:`run_plan_chain`) PINS the research-tree per-leaf
# fetch BREADTH to 3, independent of the deep-research N1 breadth knob above (default 10).
# The user fixed breadth at 3 ("BREADTH stays FIXED at 3") so the agentic loop DEPTH — not
# a wide fan-of-fetches — is the quality lever on this served route; DEPTH stays the
# live-tuneable knob (``RA_TREE_DEPTH`` via :meth:`TreeConfig.from_env`; the shapes/specs
# depth wiring lands in B6). Hard-pinned (not env-overridable): breadth is a fixed contract
# here per D97, so it does not track the shared ``RA_RESEARCH_FETCH_BREADTH``.
PLAN_CHAIN_TREE_BREADTH = 3

# DEEP-RESEARCH SUBAGENT num_ctx (s9/N1 REVISE, d62/d63 Option B). The two
# deep-research call sites ran the generic runtime at num_ctx=16384, but the
# growing-visibility unroll has each research node RE-ACCUMULATE all prior fetched
# article text — at the lifted DEEP_RESEARCH_FETCH_BREADTH (10) the prompt inflates
# toward/past 16384, i.e. the d22 window-OVERFLOW regime (prompt truncated → empty
# thinking → thin/empty output) the recipe fixed at 32768 in s5/s11. Size the window
# to the proven 32768 regime: with E4B's sliding-window attention KV is nearly free
# (8k→32k ≈ +24MiB), the resident model already loads at ctx 32768, and there is no
# Shared-GPU spill on the 6GB card — so this is a safe, measured raise, not a guess.
# Env-overridable (``RA_RESEARCH_NUM_CTX``) for live tuning, mirroring the breadth
# knob. INTERIM STOPGAP: the durable fix for the re-accumulation is N2 (compact
# ArticleNote summaries) + N3 (chunked read) + N4 (pruning) — this only buys CoT/
# content headroom at the higher fetch count so breadth=10 stops tripping overflow.
DEEP_RESEARCH_NUM_CTX = max(8192, int(os.getenv("RA_RESEARCH_NUM_CTX", "32768")))


# P2-5c (d135 / d65 FLAG-FREE END-STATE) — the GENERIC shape-driven engine is now the served
# DEFAULT for the report path; the bespoke ``run_research_tree`` orchestrator + the reversible
# ``RA_GENERIC_REPORT_PATH`` scaffolding flag have been RETIRED (P2-5b parity HELD: within-run,
# same-budget, generic breadth >= tree, grounded — p2_5b_code_review.md APPROVE). There is no
# longer a flag to read or a tree branch to fall back to: ``run_plan_chain`` and the sibling
# ``_run_deep_research_sectioned`` BOTH run PHASE-1 research through the generic
# declarative-unroll + AgentRuntime growable engine (:func:`_run_generic_research_phase`), with
# the P2.2 event-driven reactor + framework-injected review LIVE on the served route (they were
# gated behind the same retired flag, so making generic the default brings them ON — resolving
# the P2-5-review flag-coupling). The duplicate-section-collapse + single-document backstops
# (P2-3-review) stay; the (findings, sources) write-side contract is byte-identical.

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


def _shape_file_research_depth(
    catalog: Mapping[str, "ShapeSpec"], shape_name: Optional[str]
) -> Optional[int]:
    """The research DEPTH (layer/iteration count) the report path plans, READ FROM THE
    DEEP-RESEARCH SHAPE FILE (d107(2)).

    The served report route (:func:`run_plan_chain`) runs the deep-research methodology,
    so its layer budget is the deep-research shape's declared iteration count — the shape
    FILE's ``max_iter`` — not an env-only knob. We read it from the on-disk shape the model
    SELECTED when that is a deep-research-family (unrollable) shape, else from the canonical
    ``deep-research`` shape file the report path embodies. So "reading the shape file ALONE"
    lets the agent plan up to N layers; ``run_plan_chain`` then CLAMPS the value to the hard
    ``N4_TREE_DEPTH_CEILING`` (≤10) and the agent may STOP EARLY (stop_research) before it.

    Returns None only when no deep-research shape is loadable — then ``run_plan_chain``'s env
    baseline (``RA_TREE_DEPTH``) stands. The per-shape UI ``depth`` override (B6) still wins
    over this file default; this is the declarative DEFAULT below that override."""
    spec = catalog.get(shape_name) if shape_name else None
    if spec is None or not getattr(spec, "is_unrollable", False):
        spec = catalog.get("deep-research")  # the canonical deep-research shape file
    if spec is not None:
        try:
            return int(spec.max_iter)
        except (TypeError, ValueError):
            return None
    return None


def _shape_file_completeness_stop(
    catalog: Mapping[str, "ShapeSpec"], shape_name: Optional[str]
) -> str:
    """The COMPLETENESS-DRIVEN stop SIGNAL the report path uses, READ FROM THE DEEP-RESEARCH
    SHAPE FILE (P2.4 / d131 / d132.D).

    The deep-research stop semantics ("keep poking the right gap-questions until every blank
    is filled, then STOP") are DEFINED IN THE SHAPE — the shape file's ``completeness_stop``
    string — not in a hard-coded prompt. :func:`run_plan_chain` hands this text to the
    research-tree decision node as its ``stop_criteria`` so the model REASONS over the shape's
    completeness test instead of halting at an arbitrary depth baked into
    ``research_tree._DECISION_INSTRUCTION``. Read from the SELECTED deep-research-family shape
    when that is what the model picked, else from the canonical ``deep-research`` shape file
    the report path embodies. Empty string when no shape (or no field) is loadable → the tree
    keeps its byte-identical baked-default stop wording (no behavioural change)."""
    spec = catalog.get(shape_name) if shape_name else None
    if spec is None or not getattr(spec, "is_unrollable", False):
        spec = catalog.get("deep-research")  # the canonical deep-research shape file
    return str(getattr(spec, "completeness_stop", "") or "") if spec is not None else ""


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
    # d39 OVERALL GOAL: the VERBATIM user request fed into every worker node's user
    # turn (so a Gemma node, which cannot discover context, serves the real objective
    # and not just the planner's paraphrase). It is the BARE current request — taken
    # BEFORE the prior-turn context fold below — because the conversation memory is
    # injected into each node SEPARATELY (``conversation_context``); folding it into
    # the goal too would duplicate it in the node user turn. Carried onto the DAG and
    # read by the runtime (PlanDAG.goal -> AgentRuntime._overall_goal).
    overall_goal = agentic_goal(query)

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
        clar_block = (
            f"\n\nCLARIFICATION (you asked the user for a missing detail; "
            f"they answered — plan on this clarified intent):\n{clarification}"
        )
        query = f"{query}{clar_block}"
        # The clarified intent is part of the verbatim goal too, so every node sees
        # the resolved detail (not just the planner during authoring).
        overall_goal = f"{overall_goal}{clar_block}"
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
        # acyclic DAG and drive it on the SAME generic runtime (a3). The route keys off
        # the shape's DECLARATIVE fields (does it declare round/final positions →
        # is_unrollable), NOT a hard-coded shape name, so adding a cyclic shape is
        # adding one text file (plug-n-play).
        #
        # FLAG #6 — RETIRED-AS-COMBINER / JUSTIFIED-AS-CONSTRAINTS (s9/c2, d48). The
        # route is now the model's REASONED shape selection (shape_selector is off
        # format=schema as of c2, so the emitted shape is faithful to the CoT — it no
        # longer over-routes a simple/file request to deep-research the way the
        # constrained enum sample did). The three sub-conditions below are NO LONGER a
        # defensive "combine 4 signals to pick the route" flag; they are
        # CONSTRAINT-HONORING guards on the model's OWN reasoned signals — each is the
        # JUSTIFIED class in the flag audit (a no-web request must not run a web shape;
        # a file request must terminate in a file; a missing-spec must pause). With
        # faithful selection they rarely fire; when they do they honor an explicit
        # user constraint, not override the model's reasoning.
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
                # d39: the verbatim overall goal, carried onto the unrolled DAG so
                # every research/synthesis node grounds in the real objective.
                overall_goal=overall_goal,
            )

        # 2b) PLAN-CHAINING (c1b/d49.4): a LARGE / multi-page FILE request is built
        # across TWO chained plans — plan1 RESEARCH → plan2 a write-file SHAPE whose
        # per-page nodes fill one file (decomposition in the authored DAG, not code;
        # accumulation = c1's read-back loop). Fires ONLY when the model judged the
        # output multi-page AND a file is wanted AND no specialist is missing (the
        # acyclic path owns the missing-spec pause). Every other request — including a
        # single-file write — falls through UNCHANGED, so c1 single-file reliability
        # cannot regress. The signal is model-extracted (``multi_page``), never a
        # keyword match (anti-rig, parity with ``wants_file``).
        multi_page = bool(getattr(selection, "multi_page", False))
        session_span.set_attribute("session.multi_page", multi_page)
        if wants_file and multi_page and not unmet_specs:
            # RESEARCH DEPTH RESOLUTION (s13/B6 + d107(2)). Precedence, highest first:
            #   1) the per-shape UI ``depth`` override from the shapes/specs store (B6) — the
            #      user's explicit choice on the Shapes screen (persisted in SQLite);
            #   2) the DEEP-RESEARCH SHAPE FILE's declared iteration count (d107(2)) — so
            #      "reading the shape file ALONE" sets how many layers the agent plans;
            #   3) (when both absent) None → run_plan_chain's env baseline (RA_TREE_DEPTH).
            # run_plan_chain CLAMPS whatever it gets to the hard N4_TREE_DEPTH_CEILING (≤10),
            # and the agent may STOP EARLY (stop_research) before the bound. get_depth is
            # hasattr-guarded so an older store degrades gracefully.
            research_depth = None
            if shape_config is not None and selection.shape and hasattr(shape_config, "get_depth"):
                research_depth = shape_config.get_depth(selection.shape)
            if research_depth is None:
                research_depth = _shape_file_research_depth(catalog, selection.shape)
            if research_depth is not None:
                session_span.set_attribute("session.research_depth", int(research_depth))
            # P2.4 (d131/d132.D) — the STOP SIGNAL read FROM THE DEEP-RESEARCH SHAPE FILE
            # (completeness_stop): "fill all the blanks", a completeness test the decision node
            # reasons over instead of an arbitrary depth cap. Empty when no shape field →
            # run_plan_chain keeps the byte-identical baked-default stop wording.
            completeness_stop = _shape_file_completeness_stop(catalog, selection.shape)
            if completeness_stop:
                session_span.set_attribute("session.completeness_stop", True)
            return await run_plan_chain(
                query,
                selection,
                transport=transport,
                registry=registry,
                hook=hook,
                plane=plane,
                timeout=timeout,
                run_id=run_id,
                conversation_context=conversation_context,
                overall_goal=overall_goal,
                allow_web=allow_web,
                requested_specs=requested_specs,
                research_depth=research_depth,
                completeness_stop=completeness_stop,
                # P2.5 — the shape catalog so the reversible generic-engine PHASE-1 can
                # resolve + unroll the deep-research shape (no-op when the flag is OFF).
                catalog=catalog,
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
            # d39: the verbatim overall goal, carried onto the authored DAG so every
            # node (esp. the downstream writer/synthesize node, whose paraphrased task
            # dropped the real objective) grounds in the user's actual request.
            overall_goal=overall_goal,
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
    research_fetch_breadth: int = 3,
    emit_article_notes: bool = False,
    chunked_read: bool = False,
    verify_lane: bool = False,
    enable_reactor: bool = False,
    subagent_num_ctx: Optional[int] = None,
    grower: Optional[Any] = None,
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
    # gemma4 is a thinking model. s1/a2 POC: enable native ``think=True`` on the
    # planner DAG emission so the model reasons before emitting the plan; the CoT
    # comes back in a SEPARATE message.thinking field (NOT prefixing content) and
    # the transport's JSON-extraction interceptor returns clean fenced JSON. The
    # CoT still COUNTS against the token budget, so ``max_tokens`` is raised from
    # 2048 to 4096 to keep message.content from truncating to empty (the s8/a2
    # failure mode). temp 0 deterministic; format=plan_schema enforces the schema.
    planner = Planner(
        transport,
        factory,
        call_opts={
            "think": True,
            "temperature": 0,
            "format": plan_schema,
            "max_tokens": 4096,
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

    # P2.2/P2.5 (d132.B, CF#2 from P2-2-review) — EVENT-DRIVEN PLANNER REACTION on the
    # SERVED route. When ``enable_reactor`` (the report route turns it on behind the
    # reversible P2.5 generic-report flag), a node failure is routed to the planner's
    # heal DECISION through a real EventPlane SUBSCRIBER (:class:`PlannerReactor`) instead
    # of the synchronous in-call ``heal_router`` — so a failed PARALLEL node is decided the
    # instant it fails (recover-before-the-join) and a worker clarification surfaces while
    # siblings keep running. The reactor WRAPS the SAME ``heal_router``, so its decision +
    # safe fallback are byte-identical to the synchronous path; only the trigger changes.
    # The runtime owns its lifecycle (``AgentRuntime.run`` starts/stops it). None (default)
    # keeps the proven Phase-1 synchronous heal path byte-for-byte.
    planner_reactor = PlannerReactor(heal_router, plane) if enable_reactor else None

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
        # s1/b1 REASONING ROLLOUT: SchemaToolArgEmitter now defaults think=True; the
        # explicit max_tokens is raised 256->4096 so the CoT does not starve the emitted
        # tool-arg JSON to EMPTY (a2-proven load-bearing bump).
        tool_arg_emitter=SchemaToolArgEmitter(transport, max_tokens=4096),
        # AGENTIC RESEARCH cap (s9/c5, d49/d50 — reframes the retired d13 gate): a
        # ``web_search`` node is now a TRUE AGENT that DECIDES to search and which
        # sources to read (SubAgent._run_research_loop), not a deterministic search-
        # then-read follow-through. This value is only the NON-FLOW fetch CAP (max
        # web_fetch calls the loop may make), not a flow gate. F5: 0 when the web is
        # disallowed (web tools are stripped, so the research loop is never reached).
        # d65: ``research_fetch_breadth`` is the configurable cap — the run_plan_chain
        # PHASE-1 research runtime raises it to the deep-research breadth budget so the
        # gather reads MANY real sources; every other acyclic caller keeps the legacy 3.
        read_search_max_fetch=research_fetch_breadth if allow_web else 0,
        # d65 served-route wiring: the note + chunked-read grounding lanes. Default OFF
        # so the short/headlines/csv acyclic paths are byte-identical (no regression);
        # run_plan_chain PHASE-1 turns them ON so its research grounds like deep-research.
        emit_article_notes=emit_article_notes,
        chunked_read=chunked_read,
        # N6/d72 served-route wiring: N5 verification lane. Default OFF so the
        # short/headlines/csv acyclic paths stay byte-identical (no regression);
        # the two REPORT-deliverable routes pass it =True UNCONDITIONALLY in code
        # (run_plan_chain / _run_deep_research_sectioned PHASE-1 generic research runtime ->
        # Seam A gather-more; run_section_write_phase write_runtime -> Seam B ground-or-remove).
        # No env/UI flag gates grounding (d65 end-state).
        verify_lane=verify_lane,
        # s1/b1 REASONING ROLLOUT: think=True on the producer nodes too so gemma4
        # reasons before writing (CoT in the SEPARATE message.thinking field); temp 0.4
        # for prose variety; num_predict raised 1400->4096 so the CoT does not starve
        # the produced content to EMPTY (a2-proven load-bearing bump). For a role node
        # this num_predict raises the role floor via max() in SubAgent._run_role.
        subagent_call_opts=(
            {"num_predict": 4096, "temperature": 0.4, "think": True}
            if subagent_num_ctx is None
            # P2.5 generic-research PHASE-1: size the sub-agent window to the proven
            # deep-research 32768 SWA regime (the bespoke tree leaf already runs at
            # ``config.num_ctx``); without it the research nodes would run at the transport
            # default and overflow on the source-laden prompts (the d22 regime). None
            # (every other caller) keeps the byte-identical opts above.
            else {"num_predict": 4096, "temperature": 0.4, "think": True,
                  "num_ctx": int(subagent_num_ctx)}
        ),
        max_heals=4,
        max_replans=2,
        execution=mode,
        lambda_registry=getattr(hook, "subscriptions", None),
        heal_router=heal_router,
        # P2.2/P2.5 CF#2 — wired only when ``enable_reactor`` (the report route behind the
        # reversible flag). When present the runtime takes the event-driven heal path; when
        # None it keeps the Phase-1 synchronous ``heal_router`` path (byte-compatible).
        planner_reactor=planner_reactor,
        # CONVERSATION MEMORY (s5/a4): hand the bounded prior-turn context to the
        # runtime so EVERY node's sub-agent grounds its answer in the thread — not
        # just the planner (whose paraphrased node tasks dropped the concrete facts,
        # making the answer node hallucinate). None/blank => memoryless as before.
        conversation_context=conversation_context,
        # P2.5b (d134/d135) — the GROWABLE-DAG grower. Wired ONLY by the generic research
        # PHASE-1 when the deep-research shape declares ``expand_on_gaps`` (the iterative
        # breadth lever); the runtime grows the seed DAG round-by-round on note gaps. None
        # (every other caller) keeps the single-pass drive byte-identical.
        grower=grower,
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
    inject_review: bool = False,
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
        # P2.2/P2.5 CF#2 — FRAMEWORK-INJECTED REVIEW. Default False (authored plan
        # byte-identical); the served report write phase turns it on UNCONDITIONALLY
        # (P2-5c/d65 flag-free end-state — formerly behind the reversible P2.5 flag) so
        # each authored write node gains a work->review pair + a final review (the
        # framework injects it, not the planner). The review emits RAW worker content,
        # never a verdict/findings envelope (d50: the enum-verdict judgment path is retired).
        inject_review=inject_review,
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
    overall_goal: Optional[str] = None,
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
    # d39 OVERALL GOAL: stamp the verbatim user request onto the authored DAG so the
    # runtime feeds it into every node's user turn (the planner only PARAPHRASES it
    # into per-node tasks). Carried on the plan so the missing-spec resume keeps it.
    dag.goal = overall_goal or ""

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


def _collect_findings(result, *, char_budget: int = 12000) -> str:
    """Concatenate the research plan's per-node prose into ONE findings block.

    Renders each node from its PARSED structured result (so a research role's
    ``{findings,…}`` envelope is unwrapped, not leaked) in launch order, bounded to
    ``char_budget`` total so the write-phase goal stays within num_ctx. This is the
    plan1→plan2 hand-off: the write-file plan is authored + grounded FROM this text."""
    order = list(getattr(result, "launch_order", [])) or list(result.results.keys())
    parts: list[str] = []
    for nid in order:
        r = result.results.get(nid)
        if r is None:
            continue
        text = _render_parsed(r.parsed) if r.parsed is not None else unwrap_output_envelope(r.output or "")
        text = _strip_fence(text or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)[:char_budget]


_WRITE_FILE_SHAPE = ShapeSpec(
    name="write-file",
    description=(
        "A WRITE-FILE plan: the document is decomposed into the pages/sections it "
        "needs, ONE file_write node PER page/section, CHAINED so each node continues "
        "the previous one (page 2 depends_on page 1, …). EVERY node writes to the "
        "SAME output file; the runtime appends each page onto it in order. Use this "
        "shape's sequential chain to fill a large/multi-page document one part at a time."
    ),
    execution="sequential",
)


def _normalize_write_dag(dag: PlanDAG, out_name: str) -> PlanDAG:
    """Enforce the WRITE-FILE shape's discipline on the authored plan2 (c1b).

    The planner DECIDES the document's sections (each node's task — the decomposition,
    LLM-driven, stays in the DAG). But a small model authors them inconsistently:
    independent nodes that each derive their OWN slug filename, or a non-writer node
    that emits structured JSON to disk (the live c1b smoke saw four files +
    ``{"process_name":…}`` on disk). The write-file SHAPE's contract is fixed, so it is
    enforced here, NOT left to chance: every section node becomes a ``file_write``
    writer (``role=None`` → routes to the shared RAW read-back loop, never a schema/
    JSON path), is told to write to the SAME ``out_name``, and the nodes are CHAINED
    LINEARLY (page N depends_on page N-1). The runtime's writer-chain contract then
    accumulates them — RAW — into one file, each section appended in order. The
    sections remain the planner's; only the single-file linear-append SHAPE is imposed."""
    ordered = dag.topo_order() if dag.nodes else []
    new_nodes: list[PlanNode] = []
    prev: Optional[str] = None
    for n in ordered:
        task = n.task if out_name in (n.task or "") else (
            f"{n.task}\n\nWrite this section to the file '{out_name}'."
        )
        new_nodes.append(
            PlanNode(
                id=n.id,
                task=task,
                spec=n.spec,
                specs=n.specs,
                depends_on=((prev,) if prev else ()),
                tool="file_write",
                role=None,
                # SOURCE-SCOPING (s9/c13, d56): preserve the planner's per-section
                # source→section assignment so each write node is fed only its own
                # sources near the cursor (the SWA fix). The write-file SHAPE imposes
                # the single-file linear chain; the source_ids stay the planner's.
                source_ids=n.source_ids,
            )
        )
        prev = n.id
    return PlanDAG(nodes=new_nodes, rationale=dag.rationale, goal=dag.goal)


# Word tokens for the cheap lexical match in :func:`_ensure_source_coverage` (no model
# call — deterministic, model-independent).
_COVERAGE_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _ensure_source_coverage(
    write_dag: PlanDAG, sources: Sequence[Mapping[str, str]]
) -> PlanDAG:
    """Guarantee every fetched source reaches the writer — NO silent drop (MSF/d89-c, seam ④).

    The write planner assigns each section its ``source_ids``, but a small model
    routinely leaves a fetched+cited source assigned to NO section — the d86/d87 Al
    Jazeera / CSIS *vanish* (the source was fetched, cited in a note, then dropped before
    the writer ever saw it). This deterministic pass UNIONS the assigned ids across all
    section nodes and APPENDS any unassigned fetched source to the BEST lexical-match
    section (falling back to the LAST section — the "trailing additional sources" home),
    so 100% of fetched sources reach the writer.

    Model-INDEPENDENT (no model call) and NO fabrication — it only re-homes REAL fetched
    sources the run already holds. A no-source report (empty ``sources``) or an UNSCOPED
    DAG (no section carries ``source_ids`` — the degenerate single-section path that falls
    back to the full upstream index) is a NO-OP → byte-identical."""
    n_sources = len(sources or [])
    if n_sources == 0:
        return write_dag
    nodes = list(write_dag.nodes)
    scoped = [nd for nd in nodes if nd.source_ids]
    if not scoped:
        return write_dag  # unscoped path: the node already sees the full upstream index
    assigned: set[int] = set()
    for nd in nodes:
        for sid in nd.source_ids:
            assigned.add(int(sid))
    missing = [i for i in range(1, n_sources + 1) if i not in assigned]
    if not missing:
        return write_dag
    # route each unassigned source to the section whose TASK best lexically overlaps the
    # source's title+text; ties / no-overlap fall back to the last scoped section.
    add_map: dict[str, list[int]] = {}
    for sid in missing:
        src = sources[sid - 1]
        src_text = f"{src.get('title', '')} {src.get('markdown', '')}".lower()
        best_node = scoped[-1]
        best_score = -1
        for nd in scoped:
            node_terms = {
                w for w in _COVERAGE_WORD_RE.findall((nd.task or "").lower()) if len(w) > 3
            }
            score = sum(1 for t in node_terms if t in src_text)
            if score > best_score:
                best_score = score
                best_node = nd
        add_map.setdefault(best_node.id, []).append(sid)
    new_nodes: list[PlanNode] = []
    for nd in nodes:
        extra = add_map.get(nd.id)
        if extra:
            merged = list(nd.source_ids) + [s for s in extra if s not in nd.source_ids]
            new_nodes.append(replace(nd, source_ids=tuple(merged)))
        else:
            new_nodes.append(nd)
    return PlanDAG(nodes=new_nodes, rationale=write_dag.rationale, goal=write_dag.goal)


def _flag_unsupported_sections(write_dag: PlanDAG) -> PlanDAG:
    """Flag write sections left with NO sources as UNSUPPORTED — no fabrication (d106 #6).

    EMPTY-NODE-NO-FABRICATE (d60-critical): a research node that yielded 0 sources / timed
    out (FX-loop's ``_unsupported_leaf``) leaves its write section unscoped. After
    :func:`_ensure_source_coverage` has re-homed every fetched source, a section that STILL
    carries no ``source_ids`` — while sibling sections ARE scoped — has nothing to write
    from, so its task is REWRITTEN to the deterministic ``UNSUPPORTED_SECTION_INSTRUCTION``:
    write a single UNSUPPORTED line, fabricate nothing (the runtime also feeds it no source
    text to copy). The B8a Timeline section (assigned to B1, which fetched 0 → invented
    every dated event from memory) is exactly this case.

    Decomposition stays the PLANNER's — this never adds/removes/reorders sections, it only
    forbids fabrication on a section the planner already authored without sources. Fires
    ONLY on the SCOPED path (some section carries source_ids); the UNSCOPED single-section /
    no-source-report path (every section empty → falls back to the full upstream index) is
    a NO-OP, byte-identical, so a normal short report and the d56 empty-outline fallback are
    untouched."""
    nodes = list(write_dag.nodes)
    scoped = [nd for nd in nodes if nd.source_ids]
    if not scoped or len(scoped) == len(nodes):
        return write_dag  # unscoped path, or every section already has its own sources
    new_nodes: list[PlanNode] = []
    for nd in nodes:
        if nd.source_ids:
            new_nodes.append(nd)
        else:
            task = f"{nd.task}\n\n{UNSUPPORTED_SECTION_INSTRUCTION}"
            new_nodes.append(replace(nd, task=task))
    return PlanDAG(nodes=new_nodes, rationale=write_dag.rationale, goal=write_dag.goal)


def _collect_chain_sources(result) -> list[dict[str, str]]:
    """Collect the run's global fetched SOURCES (``[{title,url,markdown}]``) (s9/c13).

    Walks every research node's raw ``tool_value`` (each a
    ``{"fetched":[{title,url,markdown},…]}``) in launch order and dedupes by URL,
    so the 1-based position is the STABLE global SOURCE id the write planner assigns
    per section. Empty when the research read nothing (a no-source report → the write
    phase authors plain per-section nodes, no scoping)."""
    order = list(getattr(result, "launch_order", [])) or list(result.results.keys())
    tool_values = []
    for nid in order:
        r = result.results.get(nid)
        if r is not None and getattr(r, "tool_value", None) is not None:
            tool_values.append(r.tool_value)
    return collect_fetched_sources_full(tool_values)


def _collect_article_notes(result) -> list[dict[str, Any]]:
    """Collect the run's accumulated N2 ArticleNotes from every research node (s9/N4w).

    Walks each research node's raw ``tool_value`` (``{"article_notes":[…]}`` when the
    note lane fired) in launch order. Empty when the note lane was OFF or the model
    emitted no note — so a path that did not wire the lane is byte-identical (no key)."""
    order = list(getattr(result, "launch_order", [])) or list(result.results.keys())
    notes: list[dict[str, Any]] = []
    for nid in order:
        r = result.results.get(nid)
        tv = getattr(r, "tool_value", None) if r is not None else None
        if isinstance(tv, Mapping):
            for note in tv.get("article_notes") or []:
                if isinstance(note, Mapping):
                    notes.append(dict(note))
    return notes


def _research_state_path(run_id: Optional[str]) -> str:
    """A run-scoped path for the tree loop's persisted research-state file (d49).

    Lives under a temp ``ra_research_state`` dir (no cwd assumption); the run id keys
    it so concurrent runs never share state. ``ResearchState`` truncates it on open so
    only THIS run's notes are read back by the decision node."""
    base = os.path.join(tempfile.gettempdir(), "ra_research_state")
    os.makedirs(base, exist_ok=True)
    rid = (run_id or uuid.uuid4().hex)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(rid))
    return os.path.join(base, f"{safe}.jsonl")


def _render_outline_hint(outline_hint: Optional[list[dict[str, str]]]) -> str:
    """The PRIMARY section-scaffold clause woven into write_goal from the agent-decided
    research outline (s13/B3). Empty/blank outline → "" so the write phase falls back to
    findings-driven decomposition unchanged (d56 — NO hard-coded / fabricated sections).

    The clause is placed ABOVE the RESEARCH FINDINGS block (NOT a trailing aside) so the
    document direction the agent settled on actually REACHES the writer as the scaffold it
    starts from — the planner still REASONS over it (refine/merge/rename, add a section the
    findings clearly need, drop one they cannot support), it is a starting scaffold, not a
    frozen template."""
    if not outline_hint:
        return ""
    lines = []
    for sec in outline_hint:
        if not isinstance(sec, dict):
            continue
        title = str(sec.get("title", "")).strip()
        if not title:
            continue
        covers = str(sec.get("covers", "")).strip()
        lines.append(f"  - {title}" + (f" — covers: {covers}" if covers else ""))
    if not lines:
        return ""
    return (
        "\n\nThe research phase decided this DOCUMENT OUTLINE (the agent's planned section "
        "direction). It is the PRIMARY scaffold and the COMPLETE section list: author "
        "EXACTLY ONE section per outline entry, IN THIS ORDER, and NOTHING ELSE. You may "
        "REFINE/merge/rename a title or DROP an entry the findings cannot support, but do "
        "NOT author a second, parallel, findings-driven set of sections, do NOT append "
        "extra sections after the last outline entry / after a conclusion, and do NOT emit "
        "the same section twice under a drifted title (this caused a duplicate section tail "
        "with conflicting numbering). Every heading in the document must correspond to one "
        "outline entry below:\n" + "\n".join(lines)
    )


def _compose_write_goal(
    query: str,
    out_name: str,
    findings: str,
    catalog: str,
    *,
    is_html: bool,
    outline_hint: Optional[list[dict[str, str]]] = None,
) -> str:
    """Compose the write-planner goal: the per-section decomposition instruction, the B3
    outline scaffold (PRIMARY, above findings), the optional nav-SPA clause, then the
    research findings + source catalog. Pure/string-only so the B3 wiring is unit-testable."""
    nav_clause = (
        "\n\nThis is an HTML document — make it a NAVIGABLE single-page report (SPA): "
        "the FIRST section writes the page shell plus a NAV menu that links to each "
        "later section's anchor, and each later section is ONE <section id=\"...\"> "
        "whose heading matches its nav link. Keep it ONE well-formed HTML document."
        if is_html else ""
    )
    # EMPTY-NODE-NO-FABRICATE (s13/FX d106 #6, d60-critical): every section must be
    # grounded in the findings/sources below. A section with no supporting source must be
    # DROPPED or written as an explicit UNSUPPORTED line — NEVER fabricated from memory.
    no_fabricate_clause = (
        "\n\nGROUND EVERY SECTION in the RESEARCH FINDINGS and AVAILABLE SOURCES below. "
        "If a planned section has NO supporting source/finding (its research yielded "
        "nothing), DROP it or write only a single line marking it UNSUPPORTED — do NOT "
        "invent timelines, dates, figures, names, quotes, or citations from memory. Do "
        "NOT write a placeholder or 'sources to be added later' section; cite only the "
        "REAL fetched URLs from AVAILABLE SOURCES."
    )
    return (
        f"{query}\n\nWrite the COMPLETE document to the file '{out_name}'. "
        "Decompose it into the sections it needs: author ONE file_write node PER "
        f"section, each writing to '{out_name}', CHAINED so each section depends_on "
        "the previous (section 2 depends_on section 1, …). For EACH section, set "
        "source_ids to the SOURCE NUMBERS (from AVAILABLE SOURCES below) whose "
        "facts/figures/URLs that section uses — assign every relevant source to the "
        "section it belongs to, so each section is given ONLY its own sources."
        + _render_outline_hint(outline_hint)
        + no_fabricate_clause
        + nav_clause
        + "\n\nRESEARCH FINDINGS (decide the sections AND which sources each uses "
        f"FROM these — do NOT re-research):\n{findings}"
        + (f"\n\n{catalog}" if catalog else "")
    )


async def run_section_write_phase(
    query: str,
    out_name: str,
    findings: str,
    sources: list[dict[str, str]],
    *,
    transport,
    registry: SpecRegistry,
    hook: ToolHook,
    plane: EventPlane,
    timeout: float,
    run_id: Optional[str],
    overall_goal: Optional[str] = None,
    requested_specs: Optional[list[str]] = None,
    outline_hint: Optional[list[dict[str, str]]] = None,
):
    """The SHARED per-section bounded-SPA write phase (s9/c13, d56/d57).

    The single synthesis stage reused by BOTH the deep-research path
    (:func:`_run_deep_research`) and plan-chaining (:func:`run_plan_chain` PHASE 2):
    the write planner reads the research FINDINGS + the numbered SOURCE catalog and
    authors ONE file_write section node per part, ASSIGNING each section its own
    ``source_ids`` (the d56 (R) model-authored source→section mapping). The runtime
    then feeds each section ONLY its assigned sources' real text+URLs, nearest the
    generation cursor (``SubAgent._scoped_source_block``), so they stay inside the
    model's ~512-token sliding window and the section reproduces real figures/URLs
    instead of fabricating placeholders (the d55 SWA fix). Decomposition + the
    source→section assignment are the planner's REASONING — no hard-coded section
    list, no code relevance-matcher (d48/d56 hard guard). Degrades gracefully: a
    short report authors ONE section (no scoping), byte-near the pre-c13 single-file
    path. Returns ``(write_dag, w_result)``.

    The document is built RAW across the per-section file_write chain (c1b
    accumulation); a final pass closes an HTML wrapper the terminal section left open."""
    requested_specs = requested_specs or []
    is_html = out_name.lower().rsplit(".", 1)[-1] in ("html", "htm")
    catalog = render_source_catalog(sources)
    # NAV SPA (d55/d57): an HTML deliverable is authored as a NAVIGABLE single-page
    # report (nav + one <section> per part). The shell+nav is the FIRST section; each
    # later section is one anchored <section> — the planner authors the anchors/links
    # (reasoning-driven, no template). Markdown/other formats keep plain sections.
    # s13/B3: the agent-decided research OUTLINE (outline_hint) is woven in as the PRIMARY
    # section scaffold (above the findings) so the document direction reaches the writer;
    # an empty outline keeps the findings-driven decomposition unchanged (d56).
    write_goal = _compose_write_goal(
        query, out_name, findings, catalog, is_html=is_html, outline_hint=outline_hint,
    )
    # P2-5c (d65 FLAG-FREE) — the SERVED report write phase now ALWAYS turns ON the
    # framework-injected review (the write planner authors work nodes; the FRAMEWORK adds a
    # work->review pair + a final review) AND the event-driven reactor on the write runtime.
    # These were gated behind the retired RA_GENERIC_REPORT_PATH flag; with the generic engine
    # the served default they are LIVE on the served route (resolving the P2-5-review
    # flag-coupling). The (findings, sources) contract + the per-section bounded SPA write are
    # otherwise unchanged.
    write_planner = _build_incremental_planner(
        transport=transport, registry=registry, hook=hook,
        shape_spec=_WRITE_FILE_SHAPE, allow_web=False, requested_specs=requested_specs,
        inject_review=True,
    )
    write_runtime, _ = _build_acyclic_runtime(
        transport=transport, registry=registry, hook=hook, plane=plane,
        shape_spec=_WRITE_FILE_SHAPE, conversation_context=None, allow_web=False,
        # N6/d72: N5 Seam B post-write verify over the FINAL page — every claim is
        # re-checked against chain_sources (fed below) and fabrications are
        # ground-or-removed. Self-skips when chain_sources is empty (csv/txt).
        verify_lane=True,
        enable_reactor=True,
    )
    # SOURCE-SCOPING (d56): hand the runtime the run's global source list so each
    # section node resolves its planner-assigned source_ids to the real text/URLs.
    write_runtime.chain_sources = sources
    w_plan = await write_planner.plan(write_goal)
    write_dag = _normalize_write_dag(w_plan.dag, out_name)
    # NO SILENT DROP (MSF/d89-c, seam ④): after the planner assigns source_ids per
    # section, re-home any fetched source it left unassigned so 100% of fetched+cited
    # sources reach the writer (kills the d86/d87 Al Jazeera / CSIS vanish). Deterministic,
    # model-independent, no fabrication; a no-source / unscoped DAG is a no-op.
    write_dag = _ensure_source_coverage(write_dag, sources)
    # EMPTY-NODE-NO-FABRICATE (s13/FX d106 #6): a section the planner left with no sources
    # after coverage (its research node yielded nothing) is flagged UNSUPPORTED so the
    # writer marks it instead of fabricating from memory (the B8a Timeline defect). No-op on
    # the unscoped / no-source path, so short reports + the d56 empty-outline fallback stay
    # byte-identical.
    write_dag = _flag_unsupported_sections(write_dag)
    write_dag.goal = f"{overall_goal or query}\n\nWrite the document to '{out_name}'."
    w_result = await write_runtime.run(write_dag, timeout=timeout, run_id=run_id)

    # ---- final well-formedness (s9/c13): for an HTML deliverable, normalise to ONE
    # well-formed document and assemble the NAVIGABLE single-page wrapper + nav (the
    # per-section bounded writers emit bare body fragments so sources stay in the SWA
    # window — no section emits the page wrapper/nav; this deterministic structural
    # pass supplies them FROM the model's own headings, fabricating no content).
    # Non-HTML deliverables only get the close-gap no-op (markdown/text/csv untouched).
    try:
        rb = await hook.invoke("file_read", path=out_name, max_bytes=4_000_000)
        if rb.ok and isinstance(rb.value, dict):
            text = str(rb.value.get("text") or "")
            if is_html:
                # c14 (d59): collapse any re-emitted body-level report pass / repeated
                # heading-FAMILY to EXACTLY ONE pass of each section BEFORE the nav is
                # built FROM the headings — so the assembled SPA (and its TOC) carries
                # one of each, not the 2–3 duplicate passes the long write loop
                # over-produced (c13r). enforce_single_html_document first normalises the
                # document wrapper; collapse then removes the body-level duplicates.
                # s13/FX d106 #7 — OUTLINE-AS-PRIMARY backstop: after the wrapper is
                # normalised and same-family re-emissions are collapsed, drop any LATER
                # heading that duplicates an outline section an earlier heading already
                # wrote (the B8a appended-tail / triple "Section 3" defect that survived the
                # family-only collapse because the wording drifted). No-op when the outline
                # is empty (d56 fallback) or no two headings share an outline slot.
                assembled = assemble_html_spa(
                    collapse_outline_duplicate_sections(
                        collapse_duplicate_sections(enforce_single_html_document(text)),
                        outline_hint,
                    ),
                    title=query[:80],
                )
                if assembled != text:
                    await hook.invoke(
                        "file_write", path=out_name, content=assembled,
                        append=False, overwrite=True,
                    )
            else:
                gaps = html_close_gap(text)
                if gaps:
                    await hook.invoke(
                        "file_write", path=out_name, content="\n" + "".join(gaps),
                        append=True, overwrite=True,
                    )
    except Exception:  # pragma: no cover - best-effort well-formedness only
        pass
    return write_dag, w_result


def _deep_research_shape(
    catalog: Optional[Mapping[str, "ShapeSpec"]], shape_name: Optional[str]
) -> "ShapeSpec":
    """The deep-research ShapeSpec the report path embodies (the SELECTED deep-research-family
    shape when the model picked one, else the canonical ``deep-research`` shape). Mirrors
    :func:`_shape_file_completeness_stop`'s resolution.

    P2-5c (d65 FLAG-FREE) — the generic engine is now the served report DEFAULT and there is no
    tree fallback, so this NEVER returns None: when the passed catalog can't supply the shape
    (no catalog, or a degenerate offline catalog), it loads the SHIPPED canonical
    ``deep-research`` shape from the agent_runtime shapes dir (which declares ``expand_on_gaps``
    so the growable engine reproduces the tree's iterative breadth)."""
    spec = None
    if catalog:
        spec = catalog.get(shape_name) if shape_name else None
        if spec is None or not getattr(spec, "is_unrollable", False):
            spec = catalog.get("deep-research")
    if spec is None:
        # No usable shape in the passed catalog → load the shipped canonical deep-research
        # shape directly (the served-route invariant: the report path always has a shape).
        spec = load_shape("deep-research")
    return spec


async def _run_generic_research_phase(
    query: str,
    *,
    transport,
    registry: SpecRegistry,
    hook: ToolHook,
    plane: EventPlane,
    timeout: float,
    run_id: Optional[str],
    overall_goal: Optional[str],
    requested_specs: Optional[list[str]],
    dr_shape: "ShapeSpec",
    research_depth: Optional[int],
    completeness_stop: Optional[str] = None,
) -> tuple[str, list[dict[str, str]], dict[str, Any]]:
    """P2.5/P2-5c — the GENERIC-engine PHASE-1 research, now the SERVED DEFAULT (d65 flag-free).

    Replaces the bespoke ``run_research_tree`` layer loop with the SAME generic engine the
    inline path uses (d115/d128): UNROLL the deep-research SHAPE into a bounded acyclic
    role-tagged DAG (:func:`unroll_shape`), STRIP it to the research/critic rounds
    (:func:`_research_only_dag`), and DRIVE it on the generic :class:`AgentRuntime` with the
    report-route grounding lanes ON (notes + chunked-read + verify) and the P2.2 event-driven
    reactor wired. The accumulated ``(findings, sources)`` hand to the SAME PHASE-2
    :func:`run_section_write_phase`, so ONLY the research ENGINE changes — the write side, the
    ``(findings, sources)`` contract and the d50/d60 grounding invariants are byte-identical.

    ITERATIVE BREADTH (P2-5b, d134/d135 — parity HELD): when the resolved shape declares
    ``expand_on_gaps`` the unroll emits only the SEED layer + tags the DAG ``growable``, and
    this phase builds a :class:`DagGrower` (below) that REUSES the SAME ``ResearchState`` +
    ``Tree`` + ``run_decision_node`` + ``completeness_stop`` the retired tree used. The runtime
    drive loop (:meth:`AgentRuntime._drive_growable`) then DECOMPOSE-FIRST-seeds the goal into
    scoped children and grows wave-by-wave on note gaps — reproducing ``run_research_tree``'s
    state-driven re-expansion WITHOUT a second engine. P2-5b proved within-run, same-budget,
    that generic breadth meets-or-exceeds the tree, grounded; per d65 the reversible flag was
    RETIRED in P2-5c and this is the served default on BOTH report routes."""
    spec_name = _deep_research_spec(registry, requested_specs)
    # Unroll the declarative shape; honor the shape/UI depth as max_iter (unroll_shape clamps
    # to the shape hard_cap). Per-leaf fetch breadth stays PINNED to the report contract (D97).
    research_dag = _research_only_dag(
        unroll_shape(
            dr_shape, overall_goal or query, spec=spec_name,
            max_iter_override=research_depth,
            # P2.5b — opt IN to growable mode: this path WIRES a grower below to drive the
            # iterative growth. When the shape declares ``expand_on_gaps`` the unroll then
            # emits the seed layer + tags ``growable``; otherwise it is the frozen unroll.
            grow=True,
        )
    )
    research_dag.goal = overall_goal or query
    # P2.5b (d134/d135) — ITERATIVE GAP-EXPANSION. When the shape declares ``expand_on_gaps``
    # the unroll emitted only the SEED layer + tagged the DAG ``growable``; build the GROWER
    # that the runtime drive loop uses to reproduce ``run_research_tree``'s iterative breadth.
    # It REUSES the SAME ResearchState + Tree + run_decision_node + completeness_stop the
    # bespoke tree uses — the gather already runs on the generic runtime, the grower just folds
    # it into state, runs the decision node on the gaps, and maps the new branches to research
    # nodes. None when the shape is NOT growable (frozen unroll, byte-identical to pre-P2.5b).
    grower = None
    if getattr(research_dag, "growable", False):
        # Mirror the tree path's config: pinned report breadth + UI/shape depth clamp.
        grow_config = replace(TreeConfig.from_env(), leaf_breadth=PLAN_CHAIN_TREE_BREADTH)
        if research_depth is not None:
            grow_config = replace(
                grow_config, depth=max(1, min(int(research_depth), N4_TREE_DEPTH_CEILING))
            )
        # P2-5c FORWARD HARDENING — make the SERVED generic growable loop WALL-CLOCK bounded by
        # DEFAULT so a full-depth live/UI run is time-bounded (graceful partial, not a hard
        # cancel). When the operator has NOT pinned RA_GROW_WALLCLOCK_BUDGET_S (env default 0),
        # derive the budget from THIS run's own timeout envelope, leaving ~10% headroom for the
        # PHASE-2 write side: the grow loop stops AUTHORING new layers at ~90% of the timeout and
        # returns the findings gathered so far with stop_reason='budget' (the runtime's
        # _drive_growable enforces this off grow_config.grow_wallclock_budget). An explicit env
        # budget (>0) always wins; timeout<=0 leaves it OFF (0).
        if grow_config.grow_wallclock_budget <= 0 and timeout and timeout > 0:
            grow_config = replace(grow_config, grow_wallclock_budget=float(timeout) * 0.9)
        methodology = (
            registry.load(spec_name).body if (spec_name and spec_name in registry) else ""
        )
        # fan_out: shape-declared (the iterative breadth cap), else the config default.
        fan_out = int(getattr(research_dag, "fan_out", 0)) or grow_config.fan_out
        grower = DagGrower(
            transport=transport,
            goal=overall_goal or query,
            spec=spec_name,
            config=grow_config,
            state=ResearchState(_research_state_path(run_id)),
            tree=Tree(fan_out=fan_out),
            methodology=methodology,
            # REUSE the shape's completeness_stop VERBATIM as the decision-node stop signal.
            stop_criteria=completeness_stop or getattr(dr_shape, "completeness_stop", "") or None,
            max_layers=int(getattr(research_dag, "max_layers", 0)),
        )
    runtime, _ = _build_acyclic_runtime(
        transport=transport, registry=registry, hook=hook, plane=plane,
        shape_spec=dr_shape, conversation_context=None, allow_web=True,
        # D97: the report contract pins per-leaf fetch breadth to 3 (depth is the lever).
        research_fetch_breadth=PLAN_CHAIN_TREE_BREADTH,
        # d65 report-route grounding lanes ON (parity with the tree leaf).
        emit_article_notes=True, chunked_read=True, verify_lane=True,
        # P2.5 CF#2: event-driven reactor on the served generic research runtime.
        enable_reactor=True,
        # Size the sub-agent window to the proven deep-research SWA regime.
        subagent_num_ctx=DEEP_RESEARCH_NUM_CTX,
        # P2.5b — the growable-DAG grower (None for a frozen unroll, byte-identical).
        grower=grower,
    )
    result = await runtime.run(research_dag, timeout=timeout, run_id=run_id)
    findings = _collect_findings(result)
    sources = _collect_chain_sources(result)
    # P2.5b — the iterative-growth trace (parallels the tree's layers) so the served report
    # path can PROVE the generic engine grew breadth round-by-round on note gaps.
    grow_trace: dict[str, Any] = {"growable": bool(grower is not None)}
    if grower is not None:
        grow_trace.update({
            "layers": list(grower.layers),
            "stop_reason": grower.stop_reason,
            "grow_layers": int(getattr(runtime, "_grow_layers", 0)),
            "max_layers": int(grower.max_layers),
        })
    return findings, sources, grow_trace


async def run_plan_chain(
    query: str,
    selection: ShapeSelection,
    *,
    transport,
    registry: SpecRegistry,
    hook: ToolHook,
    plane: EventPlane,
    timeout: float,
    run_id: Optional[str],
    conversation_context: Optional[str] = None,
    overall_goal: Optional[str] = None,
    allow_web: bool = True,
    requested_specs: Optional[list[str]] = None,
    research_depth: Optional[int] = None,
    completeness_stop: Optional[str] = None,
    catalog: Optional[Mapping[str, "ShapeSpec"]] = None,
) -> AgenticResult:
    """PLAN-CHAINING for a LARGE / multi-page output (c1b, d49.4 / d50.4(4)).

    Chains TWO plans, exactly as the neuron maintains a recipe across phases:

      * **plan1 = RESEARCH** — the incremental planner authors + the live runtime
        drives a research DAG that gathers the facts/sources (NO file written here);
      * **plan2 = WRITE-FILE shape** — a SECOND plan, authored as the ``write-file``
        sequential shape, whose per-page/section ``file_write`` nodes FILL one file.
        The decomposition into pages lives in THAT authored DAG (the planner reads the
        request + findings and emits one node per page, chained), NOT in code.

    The per-page ACCUMULATION is c1's shared raw-content read-back loop unchanged
    (``runtime._run_raw_file_loop``): the runtime's structural writer-chain contract
    makes each downstream page CONTINUE (append onto) the file its upstream page
    wrote, and defer the closing-tag gate to the final page. The research findings
    are fed verbatim into plan2's authoring goal + every write node's user turn (the
    d17/d38 context-feeding), so the writer grounds in the real research.

    A final well-formedness pass closes an HTML wrapper if the terminal page left it
    open (chain-scope parity with c1's per-node close gate)."""
    requested_specs = requested_specs or []
    tracer = get_tracer("chat_app.agentic")
    with tracer.start_as_current_span("agent.plan_chain") as span:
        # ---- PHASE 1: RESEARCH via the GENERIC growable engine (s13/B1; P2-5c, d135/d65) ----
        # The report path runs a real agentic, iteratively-growing research DAG + persisted
        # MEMORY, REPLACING the flat single-pass incremental-planner DAG that ran ONCE and
        # starved the writer (d93/d94): the generic engine decompose-first-seeds the goal into
        # scoped research nodes (note + chunked-read + breadth grounding lanes LIVE on the
        # generic research runtime — verify_lane / Seam-A gather-more INCLUDED), appends each
        # node's ArticleNotes + findings to a persisted ResearchState, and the grower's DECISION
        # NODE reads that state back (d49, real state not memory) and authors expand/prune by
        # REASONING (no template, no code fan-out) — growing wave-by-wave until the model stops
        # expanding, the depth bound, or the wall-clock budget is hit. This is the SAME generic
        # engine _run_deep_research_sectioned runs, the report path's default (d65: flag-free —
        # the bespoke run_research_tree orchestrator is RETIRED).
        # The accumulated (findings, sources) hand UNCHANGED to the c13 per-section write
        # phase below — PHASE-2 feeding is byte-identical (d89), so source-scoping / the SWA
        # fix / d50.1 raw-content + d60 0-fabrication grounding all carry over unchanged.
        #
        # D97 CONFIG MANDATE: BREADTH is PINNED to 3 on this path (PLAN_CHAIN_TREE_BREADTH,
        # NOT the deep-research N1 default of 10).
        #
        # DEPTH (s13/B6, d97): DEPTH is the live quality lever and is now USER-SETTABLE
        # through the shapes/specs flow — the Shapes config store carries a per-shape
        # ``depth`` override (sibling of ``max_iter``); run_agentic reads it for the
        # selected shape and hands it here as ``research_depth``. We map it onto
        # TreeConfig.depth, CLAMPED to [1, N4_TREE_DEPTH_CEILING] (the hard ≤10 the user
        # fixed) so a UI value can lower or raise depth but never exceed the safety bound.
        # When no override is set (``research_depth`` None) the env baseline stands
        # (RA_TREE_DEPTH via TreeConfig.from_env). BREADTH is unaffected — it stays pinned.
        # DEPTH / BREADTH trace config (B6/D97): leaf breadth is PINNED to
        # PLAN_CHAIN_TREE_BREADTH and depth is the user-settable lever clamped to
        # [1, N4_TREE_DEPTH_CEILING]. The generic phase builds its OWN grower config with these
        # same bounds (mirrored here ONLY for the served-route span attributes below).
        tree_config = replace(TreeConfig.from_env(), leaf_breadth=PLAN_CHAIN_TREE_BREADTH)
        if research_depth is not None:
            clamped_depth = max(1, min(int(research_depth), N4_TREE_DEPTH_CEILING))
            tree_config = replace(tree_config, depth=clamped_depth)
        # P2-5c (d135 / d65 FLAG-FREE) — PHASE-1 research runs through the GENERIC
        # declarative-unroll + AgentRuntime GROWABLE engine (the same engine the inline path
        # uses); the bespoke run_research_tree loop is RETIRED. The generic engine reproduces
        # the tree's iterative breadth via the DECOMPOSE-FIRST seed + the gap-expansion grower
        # (P2-5b parity HELD: within-run, same-budget, generic breadth >= tree, grounded). The
        # deep-research SHAPE is always resolvable (catalog, else the shipped canonical shape);
        # the generic path has no tree-authored outline, so PHASE-2 falls back to
        # findings-driven section decomposition (d56). The shape completeness_stop is handed to
        # the GROWABLE drive's decision node verbatim (the same "fill all the blanks" signal the
        # tree reasoned over, d131/d132.D).
        dr_shape = _deep_research_shape(catalog, selection.shape)
        findings, sources, grow_trace = await _run_generic_research_phase(
            query,
            transport=transport, registry=registry, hook=hook, plane=plane,
            timeout=timeout, run_id=run_id, overall_goal=overall_goal,
            requested_specs=requested_specs, dr_shape=dr_shape,
            research_depth=research_depth,
            completeness_stop=completeness_stop,
        )
        # The generic engine has no tree-authored outline channel → PHASE-2 falls back to
        # findings-driven section decomposition (d56).
        outline_hint: Optional[list[dict[str, str]]] = None
        # The grower ran iterative layers; the per-layer 'gathered' counts are the research
        # rounds executed (parity with the tree's old layer trace).
        rounds_executed = sum(int(l.get("gathered", 0) or 0) for l in grow_trace.get("layers", []))
        engine = "generic-unroll"
        # SOURCE-SCOPING (s9/c13, d56): the (findings, sources) contract feeds the SAME PHASE-2
        # write planner, which assigns each section ONLY its own sources (the SWA fix). The
        # served-path trace records the engine + research_nodes / depth, and (when the shape
        # grew) the grower's stop_reason / grow_layers — the iterative-breadth signal that
        # replaces the tree's layer trace.
        span.set_attribute("plan_chain.engine", engine)
        span.set_attribute("plan_chain.research_nodes", rounds_executed)
        span.set_attribute("plan_chain.findings_chars", len(findings))
        span.set_attribute("plan_chain.sources", len(sources))
        span.set_attribute("plan_chain.leaf_breadth", tree_config.leaf_breadth)
        # The CONFIGURED depth bound that drove the loop (B6: proves the shapes/specs depth
        # override reached the grower config on the SERVED route).
        span.set_attribute("plan_chain.tree_depth_configured", tree_config.depth)
        if grow_trace.get("growable"):
            span.set_attribute(
                "plan_chain.grow_stop_reason", str(grow_trace.get("stop_reason") or "")
            )
            span.set_attribute(
                "plan_chain.grow_layers", int(grow_trace.get("grow_layers", 0) or 0)
            )

        # ---- shared output file: the LLM-named file, else a derived relatable name ----
        out_name = explicit_filename(query) or derive_output_path(
            overall_goal or query, "", requested_specs or None
        )
        span.set_attribute("plan_chain.out_file", out_name)

        # ---- PHASE 2: the SHARED per-section bounded-SPA write phase (d56/d57) ----
        write_dag, w_result = await run_section_write_phase(
            query, out_name, findings, sources,
            transport=transport, registry=registry, hook=hook, plane=plane,
            timeout=timeout, run_id=run_id, overall_goal=overall_goal,
            requested_specs=requested_specs,
            # s13/B3 — the agent-decided outline becomes the PRIMARY section scaffold (the
            # tree authors it; the generic engine has none → None → findings-driven sections).
            outline_hint=outline_hint,
        )
        span.set_attribute("plan_chain.write_nodes", len(write_dag.nodes))

        agentic = _agentic_from_runtime(
            write_dag, w_result,
            shape=(selection.shape or "plan-chain"),
            escalated=selection.escalate,
            rationale=(selection.rationale or write_dag.rationale or "plan-chaining: research → write-file"),
        )
        # SERVED-route loop trace (s13/B1, P2-5c): carry the research-engine control snapshot on
        # the result so the s7 markdown trace PROVES which engine + grounding lanes ran on the
        # SERVED report path. The engine is always the GENERIC growable engine now (the bespoke
        # tree is retired); when the shape grew, its iterative trace (grow layers / stop reason)
        # replaces the tree's old layer snapshot — its DAG IS the research topology.
        agentic.deep_research = {
            "shape": (selection.shape or "plan-chain"),
            "engine": engine,
            "rounds_executed": rounds_executed,
            "leaf_breadth": tree_config.leaf_breadth,
            "write_dag": write_dag.as_dict(),
            "sources": len(sources),
            "sectioned": True,
            "plan_chain": True,
        }
        if grow_trace.get("growable"):
            # The GROWABLE generic engine's iterative trace: how many research layers grew, what
            # each decision layer expanded, and the stop reason (agent_sufficient / no_expansion
            # / depth_bound / budget — the P2-5c graceful wall-clock stop).
            agentic.deep_research.update({
                "growable": True,
                "layers": grow_trace.get("layers", []),
                "stop_reason": grow_trace.get("stop_reason"),
                "grow_layers": grow_trace.get("grow_layers", 0),
                "depth_reached": grow_trace.get("grow_layers", 0),
                "depth_configured": tree_config.depth,
                "max_layers": grow_trace.get("max_layers", 0),
            })
        return agentic


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
    # Render role-node output from its PARSED structured result, not the raw model
    # text (s8/b8): a role node emits schema-wrapped JSON ({output} for worker/
    # synthesis, {findings,…} for research), so surfacing ``r.output`` raw would leak
    # the JSON envelope into the chat (the o4 "answers with raw {…} JSON" defect on
    # the linear path). _render_parsed unwraps it; a bare (spec-less, role-less) node
    # has no parsed result and falls back to its plain text.
    outputs = {
        nid: _strip_fence(
            _render_parsed(r.parsed) if r.parsed is not None else unwrap_output_envelope(r.output or "")
        )[:600]
        for nid, r in result.results.items()
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
        artifacts=_artifacts_for(final, result),
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

    # d39: the bare current request is the verbatim overall goal fed to every node
    # (captured before the prior-turn context fold, which is injected separately).
    overall_goal = agentic_goal(query)
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
    dag.goal = overall_goal or ""  # d39: feed the verbatim goal to every node
    result = await runtime.run(dag, run_id=run_id)
    return _agentic_from_runtime(
        dag, result, shape="offline", escalated=False, rationale=dag.rationale
    )


_REPORT_DELIVERABLE_RE = re.compile(
    r"\b(report|document|dossier|write[\s-]?up|article|essay|paper|brief(?:ing)?|"
    r"web[\s-]?page|web[\s-]?site|page|site|html|htm|markdown|\.md\b|\.html?\b)\b",
    re.IGNORECASE,
)


def _is_report_deliverable(query: str, selection) -> bool:
    """True when a DETAILED request is also a written REPORT deliverable (s9/c13).

    The ROUTING gate that decides whether the deep-research synthesis runs as the
    per-section bounded write phase (a file report) vs the inline single-synthesis
    answer. Fires when the model already read a file/multi-page intent
    (``wants_file``/``multi_page``) OR the request names a written report deliverable
    (report/document/HTML/markdown/page/…). This is a deliverable-TYPE routing signal
    — the same class as the existing ``wants_file`` extraction — NOT the d56 source→
    section assignment (which stays model-authored). A bare 'research X in depth'
    (no report/file cue) is NOT sectioned, so an inline deep-research answer is
    unchanged (no regression)."""
    if getattr(selection, "wants_file", False) or getattr(selection, "multi_page", False):
        return True
    return bool(_REPORT_DELIVERABLE_RE.search(query or ""))


def _research_only_dag(dag: PlanDAG) -> PlanDAG:
    """The unrolled deep-research DAG MINUS its terminal synthesis/verify (s9/c13).

    The per-section write phase REPLACES the single synthesis node, so the research
    run keeps only the gather/critic rounds (node ids end ``_research``/``_critic``);
    the terminal ``_synthesis``/``_verify`` nodes are dropped. They are terminal (no
    node depends on them), so removing them leaves every remaining node's
    ``depends_on`` resolvable — the DAG stays valid. At least one ``_research`` node
    always remains (every round carries one)."""
    keep_ids = {
        n.id for n in dag.nodes
        if not (n.id.endswith("_synthesis") or n.id.endswith("_verify"))
    }
    kept = [
        PlanNode(
            id=n.id, task=n.task, spec=n.spec, specs=n.specs,
            # defensively drop any edge to a removed terminal node so the DAG stays
            # valid even if a future unroll wires a research/critic dep to one.
            depends_on=tuple(d for d in n.depends_on if d in keep_ids),
            tool=n.tool, tool_args=n.tool_args, role=n.role, needs_spec=n.needs_spec,
            source_ids=n.source_ids,
        )
        for n in dag.nodes if n.id in keep_ids
    ]
    return PlanDAG(
        nodes=kept, rationale=dag.rationale, shape=dag.shape, goal=dag.goal,
        # P2.5b — carry the growable tagging through the research-only strip so the runtime
        # still grows the DAG (a growable seed has no synthesis/verify to drop anyway).
        growable=getattr(dag, "growable", False),
        fan_out=getattr(dag, "fan_out", 0),
        max_layers=getattr(dag, "max_layers", 0),
    )


async def _run_deep_research_sectioned(
    query: str,
    shape_spec,
    selection: ShapeSelection,
    dag: PlanDAG,
    *,
    transport,
    registry: SpecRegistry,
    hook: ToolHook,
    plane: EventPlane,
    timeout: float,
    run_id: Optional[str],
    effective_max_iter: int,
    requested_specs: Optional[list[str]] = None,
    overall_goal: Optional[str] = None,
) -> AgenticResult:
    """Deep research (GENERIC GROWABLE ENGINE) → SHARED per-section bounded-SPA write phase.

    The grounded structured path for a DETAILED sourced report (s9/c13 + P2-5c, d65/d135):
    PHASE-1 research runs through the SAME generic declarative-unroll + AgentRuntime growable
    engine the report path uses (:func:`_run_generic_research_phase`) — the bespoke
    ``run_research_tree`` loop is RETIRED (flag-free end-state d65). The accumulated findings
    + fetched SOURCES hand to :func:`run_section_write_phase`, which authors one bounded
    file_write section per part — each fed ONLY its planner-assigned sources nearest the
    cursor so they stay inside the model's ~512-token sliding window. Decomposition +
    source→section assignment stay the planner's reasoning (no hard-coded section list, no
    code relevance-matcher). ``dag`` (the unrolled shape) is no longer the research topology
    here — the growable engine builds it — but is kept on the signature for the caller's
    uniform deep-research construction."""
    # P2-5c (d135 / d65 FLAG-FREE) — the GENERIC growable engine is the served default for the
    # detailed-report path too (the bespoke run_research_tree loop is retired). The SELECTED
    # deep-research shape IS the unroll source; the generic phase decompose-first-seeds + grows
    # on note gaps (reproducing the retired tree's iterative breadth), with the note +
    # chunked-read + breadth grounding lanes + the P2.2 event-driven reactor LIVE. The shape's
    # completeness_stop is reused VERBATIM as the grower's decision stop signal — the same "fill
    # all the blanks" signal the tree's decision node reasoned over (CF#4). The seeded spec
    # methodology is loaded INSIDE the generic phase (parity with the report route).
    completeness_stop = str(getattr(shape_spec, "completeness_stop", "") or "")
    findings, sources, grow_trace = await _run_generic_research_phase(
        query,
        transport=transport, registry=registry, hook=hook, plane=plane,
        timeout=timeout, run_id=run_id, overall_goal=overall_goal,
        requested_specs=requested_specs, dr_shape=shape_spec,
        # The detailed-report route's depth follows the shape's effective max_iter (the
        # UI/shape iteration count); the generic phase clamps it to the depth ceiling.
        research_depth=effective_max_iter,
        completeness_stop=completeness_stop,
    )

    out_name = explicit_filename(query) or derive_output_path(
        overall_goal or query, "", requested_specs or None
    )
    write_dag, w_result = await run_section_write_phase(
        query, out_name, findings, sources,
        transport=transport, registry=registry, hook=hook, plane=plane,
        timeout=timeout, run_id=run_id, overall_goal=overall_goal,
        requested_specs=requested_specs,
        # The generic engine has no tree-authored outline channel → findings-driven
        # section decomposition (d56).
        outline_hint=None,
    )

    # The grower's per-layer 'gathered' counts (incl. the decompose-first seed fold) are the
    # research rounds executed (parity with the retired tree's per-layer trace).
    rounds_executed = sum(int(l.get("gathered", 0) or 0) for l in grow_trace.get("layers", []))
    research_ok = bool((findings or "").strip()) and (rounds_executed > 0 or len(sources) > 0)
    agentic = _agentic_from_runtime(
        write_dag, w_result,
        shape=selection.shape, escalated=selection.escalate,
        rationale=(selection.rationale or "deep-research generic growable → per-section bounded SPA"),
    )
    # Preserve the deep-research summary contract + reflect BOTH phases in ``ok``. The research
    # topology is now the generic growable DAG; the grower's trace (layers / stop_reason /
    # grow_layers) replaces the tree's control snapshot — proof the iterative breadth ran on
    # the served route.
    agentic.deep_research = {
        "shape": shape_spec.name,
        "engine": "generic-unroll",
        "effective_max_iter": effective_max_iter,
        "rounds_executed": rounds_executed,
        "growable": bool(grow_trace.get("growable")),
        "layers": grow_trace.get("layers", []),
        "stop_reason": grow_trace.get("stop_reason"),
        "grow_layers": int(grow_trace.get("grow_layers", 0) or 0),
        "write_dag": write_dag.as_dict(),
        "sources": len(sources),
        "sectioned": True,
    }
    agentic.ok = bool(research_ok and w_result.ok)
    return agentic


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
    overall_goal: Optional[str] = None,
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
    # d39 OVERALL GOAL: stamp the verbatim user request onto the unrolled DAG so the
    # runtime feeds it into every research/synthesis node's user turn (uniform with
    # the acyclic path; the role nodes otherwise see only their unroll task prefix).
    dag.goal = overall_goal or ""

    # SPA-PER-SECTION BOUNDED SYNTHESIS (s9/c13, d55/d56/d57): a DETAILED sourced
    # report is the path that fails the E4B sliding-window ceiling (the single
    # synthesis node builds one giant doc; later sections fall outside the ~512-tok
    # SWA window → placeholder facts/URLs). For a detailed request, run the research
    # ROUNDS only, then hand their findings + fetched sources to the SHARED
    # per-section bounded write phase (each section fed ONLY its own sources, nearest
    # the cursor). is_detailed_task is the SAME reasoning SIGNAL c8 uses — a gate, not
    # a content template (d48-clean): a NON-detailed unrollable request (e.g.
    # headlines) is byte-identical to the pre-c13 single-synthesis path below (no
    # regression). The degenerate short-detailed case authors ONE section (graceful).
    if is_detailed_task(query) and _is_report_deliverable(query, selection):
        return await _run_deep_research_sectioned(
            query, shape_spec, selection, dag,
            transport=transport, registry=registry, hook=hook, plane=plane,
            timeout=timeout, run_id=run_id, effective_max_iter=effective_max_iter,
            requested_specs=requested_specs, overall_goal=overall_goal,
        )

    # d13 → d49/d50 (s9/c5): each deep-research layer must READ real sources, not
    # describe a search-results page. A per-run hook (bound to THIS chat's plane,
    # reusing the shared registry's web_search/web_fetch) is wired so each layer's
    # AGENTIC research loop (SubAgent._run_research_loop) can web_search → web_fetch
    # the real article URLs IT chooses → ground its findings in the EXTRACTED text.
    # ``read_search_max_fetch`` is the per-layer fetch CAP (non-flow), not a gate.
    run_hook = ToolHook(plane, registry=hook.registry)
    runtime = AgentRuntime(
        transport=transport,
        loader=SpecLoader(registry),
        hook=run_hook,
        plane=plane,
        # s1/b1 REASONING ROLLOUT: think=True so each deep-research role node (incl. the
        # JUDGMENT verdict) reasons in the SEPARATE message.thinking field + temp 0
        # (deterministic judgment) + a generous num_ctx so a late round's
        # growing-visibility context (all prior layers threaded as inputs) PLUS the
        # fetched article text is not truncated. num_predict=4096 raises the per-role
        # floor via max() in SubAgent._run_role so the CoT does not starve the role's
        # JSON content/verdict to EMPTY (a2-proven load-bearing bump). NOTE: the
        # deep-research CONTENT-quality fix (o4) is OUT OF SCOPE here — this only gives
        # the CoT token headroom; it is step s3.
        subagent_call_opts={"think": True, "temperature": 0,
                            "num_ctx": DEEP_RESEARCH_NUM_CTX, "num_predict": 4096},
        # AGENTIC RESEARCH cap (s9/c5, d49/d50): each deep-research layer is a TRUE
        # AGENT (SubAgent._run_research_loop) that DECIDES to web_search then web_fetch
        # the sources IT chooses, grounding its findings in the extracted article text.
        # This value is the NON-FLOW fetch CAP per layer (max web_fetch calls), not the
        # retired deterministic search-then-read follow-through.
        # BREADTH (s9/N1, d60/c15 part-a): lifted from the legacy hard-wired 3 to the
        # configurable breadth budget (~8-12) so a layer reads MANY real sources; the
        # ReAct turn ceiling rises proportionally (runtime RESEARCH_SEARCH_HEADROOM).
        read_search_max_fetch=DEEP_RESEARCH_FETCH_BREADTH,
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
    # d48: the deep-research final answer = the SYNTHESIZER node's output; fall back
    # to the last WORKER node (the verify/research positions are now worker nodes).
    final = _final_role_output(dag, result, ROLE_SYNTHESIZER) or _final_role_output(
        dag, result, ROLE_WORKER
    ) or ""

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
        artifacts=_artifacts_for(final, result),
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
    any non-empty node output. Role nodes are rendered from their PARSED result
    (``for_user`` strips any internal verdict scaffold) so the user-facing answer is
    clean CONTENT, never the raw schema-wrapped JSON envelope (s8/b8)."""
    def _render(r) -> str:
        rendered = (
            _render_parsed(r.parsed, for_user=True)
            if r.parsed is not None
            else unwrap_output_envelope(r.output or "")
        )
        return _strip_fence(rendered)

    by_topo = dag.topo_order()
    for node in reversed(by_topo):
        r = result.results.get(node.id)
        if r is not None and _render(r).strip():
            return _render(r)
    for r in result.results.values():
        if _render(r).strip():
            return _render(r)
    return ""


def _render_parsed(parsed: Any, *, for_user: bool = False) -> str:
    """Render a role node's parsed structured output to readable text.

    A worker or SYNTHESIS node emits ``{output}`` (the deliverable text, s8/b8);
    research emits ``{findings,sources,...}``; verify/reviewer emit
    ``{verdict,findings,...}``. Surface the most useful human-readable view without
    leaking the raw JSON when avoidable.

    ``for_user`` renders the answer the chat surfaces: the judgment SCAFFOLD a
    verify/reviewer node carries (its ``**verdict:**`` header) is internal
    lifecycle metadata, not part of the answer, so it is OMITTED — otherwise a
    deep-research run's verify verdict (e.g. ``**verdict:** fail``) leaks into the
    user-facing reply (F4). The per-node debug ``outputs`` map keeps the verdict
    (``for_user`` defaults False)."""
    if parsed is None:
        return ""
    if isinstance(parsed, str):
        return _strip_fence(unwrap_output_envelope(parsed))
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


def _written_filename(result) -> str:
    """The basename of the file the run wrote via ``file_write`` (LLM's chosen name).

    Scans the node results (in launch order, last wins → the terminal write) for a
    ``file_write``/``write_file`` tool result and returns its path's basename, or
    ``''`` when no file was written. This is how the LLM's chosen filename +
    extension reaches the artifact (c3/d49) instead of a hard-coded ``report.md``."""
    if result is None or not getattr(result, "results", None):
        return ""
    order = list(getattr(result, "launch_order", []) or list(result.results.keys()))
    chosen = ""
    for nid in order:
        res = result.results.get(nid)
        if res is None:
            continue
        if res.tool_used in ("file_write", "write_file") and isinstance(res.tool_value, Mapping):
            p = str(res.tool_value.get("path") or "").strip()
            if p:
                chosen = p  # keep the last written file (the terminal deliverable)
    if not chosen:
        return ""
    return chosen.replace("\\", "/").rsplit("/", 1)[-1]


def _default_artifact_name(body: str) -> str:
    """A relatable ``.md`` artifact name from the answer's title (no LLM filename).

    Only used when the run wrote NO file (e.g. a pure SSE answer), so there is no
    LLM-chosen name to passthrough — we derive a slug from the first heading/line
    and default to ``.md`` (a conventional text-report extension, NOT the old
    hard-coded literal ``report.md``)."""
    title = ""
    for line in (body or "").splitlines():
        s = line.strip().lstrip("#").strip()
        if s and not s.startswith("<") and not s.startswith("```"):
            title = s
            break
    slug = "".join(c if c.isalnum() else "-" for c in title.lower()).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return f"{slug[:60] or 'report'}.md"


def _artifacts_for(final: str, result=None) -> list[tuple[str, str, str]]:
    """One downloadable artifact for a non-empty final answer (LLM-named).

    The filename + mime are PASSTHROUGH from the name the LLM chose: if the run
    wrote a file via ``file_write`` the artifact carries THAT name + extension and a
    mime derived from it (c3/d49 — the LLM picks .html/.md/.txt/.csv/...; no
    hard-coded ``report.md``/``text/markdown``, no per-format template/sniffer).
    When no file was written, a relatable ``.md`` name is derived from the answer.
    Empty → no artifact."""
    body = (final or "").strip()
    if not body:
        return []
    name = _written_filename(result) or _default_artifact_name(body)
    return [(name, mime_for_path(name), body)]


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
