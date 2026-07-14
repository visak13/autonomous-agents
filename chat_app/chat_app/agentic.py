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
   * the ``deep-research`` shape → an ENGINE-OWNED GROWABLE SEED (a single tool-less
     self-selecting research node, :func:`_research_seed_dag`) whose research TOPOLOGY is
     AUTHORED at runtime by the :class:`~agent_runtime.research_tree.DagGrower` (decompose-first
     → grow on note gaps) and driven by the SAME :class:`~agent_runtime.AgentRuntime` — there is
     no per-shape executor and NO deterministic unroll (s16/a3 d239/d247: the unroll is retired).

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
from chat_app.curation import CURATED_SHAPES, CURATED_SPECS, curate_index
from agent_runtime import (
    EVENT_MISSING_SPECIALIST,
    EVENT_NEEDS_CLARIFICATION,
    MISSING_SPEC_CHOICES,
    AbstractPlanFactory,
    clarification_payload,
    AgentRuntime,
    ExecutionMode,
    HealRouter,
    HealLog,
    IncrementalPlanner,
    MalformedOutputError,
    SelfHeal,
    PlanDAG,
    Planner,
    PlannerReactor,
    DagGrower,
    N4_TREE_DEPTH_CEILING,
    ResearchState,
    # d285 SB-4: the per-branch memory resolver (SB-1) + the leaf record, so the served
    # research finalizer opens/continues each branch's memory by index and folds its leaf in.
    resolve_brief_memory,
    LeafResult,
    # s14/P3A — the research-owned compact-memory builders (single source of truth);
    # the in-research decision node + this write planner reason over the SAME builders.
    compose_research_narrative,
    render_verbatim_source_index,
    resolve_chunk,
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
)
from agent_runtime.tracing import get_tracer
from agent_runtime.synth_tools import (
    collect_fetched_sources_full,
    derive_output_path,
    explicit_filename,
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
# d178 (s15 thread-2) — the inline whole-doc deep-research runtime that consumed this knob
# directly was RETIRED with the monolithic synthesis fold; the deep-research path now runs
# PHASE-1 on the generic engine where per-leaf fetch breadth is the report-pinned
# ``PLAN_CHAIN_TREE_BREADTH``. This constant remains the canonical default of the same env
# knob (``RA_RESEARCH_FETCH_BREADTH``) that ``TreeConfig.from_env().leaf_breadth`` reads, so
# the env contract + its band are unchanged.
DEEP_RESEARCH_FETCH_BREADTH = max(1, int(os.getenv("RA_RESEARCH_FETCH_BREADTH", "10")))

# D97 SUPERSEDED (autonomy rebuild P4, owner decision 2026-07-13): breadth 3 starved the
# frontier — live Maratha forensics showed genuinely-needed facets (maps, modern influence)
# created in all 5 epochs and EXECUTED IN 0 because the hard cap silently swallowed them.
# The report path now defaults to 10 and tracks the SHARED env knob
# (``RA_RESEARCH_FETCH_BREADTH``), same contract as the deep-research N1 breadth above.
# The cap also becomes VISIBLE DATA to the model (the decision observation carries the
# remaining branch budget) instead of an invisible truncation.
PLAN_CHAIN_TREE_BREADTH = DEEP_RESEARCH_FETCH_BREADTH

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


# RP-6b (d359/d361) — a declarative SPEC-ROLE → the seeded default spec that fills it. The
# deep-research shape declares each phase's ``spec_role``; the engine maps that ROLE (not a
# spec NAME — a name conditional is banned) onto the seeded specialization for that class of
# node. The research phase's ``research`` role → the research/analysis output spec. The
# ``writer`` role is NOT resolved here: the write phase's writer spec flows through the write
# planner (requested_specs), so it never has to ride on the research seed (Bug A d355/d356).
_SPEC_ROLE_DEFAULTS: dict[str, str] = {"research": DEEP_RESEARCH_SPEC}


def _deep_research_spec(
    registry: SpecRegistry, *, shape: Optional["ShapeSpec"] = None
) -> Optional[str]:
    """The specialization the deep-research RESEARCH-phase seed carries (RP-6b spec-routing).

    The deep-research SHAPE DECLARES the research phase's ``spec_role`` (``[[phases]]``); the
    engine maps that role onto the seeded default spec (:data:`_SPEC_ROLE_DEFAULTS`) — the
    research role → the research-analysis output spec. This is the Bug A fix (d355/d356): a
    user-named WRITER/output spec is NO LONGER pulled onto the research seed (the old
    "first requested registered spec wins" F5 behaviour put a writer spec on a research node);
    it routes to the WRITE phase instead (``run_section_write_phase`` threads ``requested_specs``
    into the write planner, where a user-named output spec is reachable on the deliverable node).
    So the research node gets a research spec and the write node gets the writer spec — never
    crossed. Falls back to the research-analysis default for a shape with no phases, or ``None``
    when even that default is unregistered (role-framing only)."""
    role = shape.spec_role_for("research") if shape is not None else "research"
    name = _SPEC_ROLE_DEFAULTS.get(role, DEEP_RESEARCH_SPEC)
    return name if name in registry else None


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
    if spec is None or not getattr(spec, "is_deep_research", False):
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
    if spec is None or not getattr(spec, "is_deep_research", False):
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
    # d185 NOTES-ARCH Layer 2 — the per-research BRIEF (a thesis-level digest derived
    # from the research graph), populated on the report/deep-research path. The route
    # persists it at the chat-session level keyed by (chat_id, research_id) so multiple
    # researches in one chat coexist; None when the run did no growable research.
    research_brief: Optional[dict[str, Any]] = None
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
    # D4 (d215) TERMINAL SYNTHESIZER — a brief, human-facing summary of the finished run
    # (topic + sections + sources + artifact), composed by the terminal synthesizer stage
    # that runs ONCE after the planner loop exits and SSE-announced on the plane. None on
    # paths that do not run the synthesizer (acyclic / paused runs).
    synthesis_summary: Optional[str] = None


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
    session_id: Optional[str] = None,
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
    the ``/runs/{run_id}`` the client polls.

    SESSION BINDING (a17, d185 Layer 3): ``session_id`` is the CHAT SESSION id (the
    route passes ``chat_id``). When supplied it threads into ``run_plan_chain``'s
    PHASE-1 research so the persisted ``ResearchState`` is keyed by the session and
    STICKS — a follow-up turn in the SAME chat reads its prior notes/sources back
    (no re-research) while a different chat stays isolated. None (inline/offline) →
    the research state is run-scoped + truncated on open (byte-identical to pre-a17)."""
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
        # s15 ROUTING PURITY (d148/d151/d161): the shape choice is the model REASONING
        # over the shape / spec / TOOL descriptions — there is no downstream boolean
        # code-switch. So advertise the SPEC one-liners (what each specialization
        # DELIVERS — the output determinant, d161) and the TOOL one-liners (what a plan
        # can DO) to the selector alongside the shape descriptions, so the model can
        # decide the route from descriptions alone.
        spec_catalog = [
            {"name": e.name, "description": e.description} for e in registry.index()
        ]
        selector = ShapeSelector(
            transport,
            spec_names=registry.names(),
            spec_catalog=spec_catalog,
            tool_catalog=hook.registry.catalog(),
            # d230 registry scoping: the selector OFFERS only the curated required-now
            # shapes + specs (the raw load_shapes loader + the full registry are
            # untouched — this narrows only what the selector reasons over).
            exposed_shapes=CURATED_SHAPES,
            exposed_specs=CURATED_SPECS,
        )
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

        # 2) ROUTE BY THE SELECTED SHAPE (s15 ROUTING PURITY, d148/d151/d161).
        # The model's REASONED shape choice IS the route — there is NO wants_file /
        # multi_page boolean code-switch and NO `_is_report_deliverable` regex
        # deciding the path (those rigid structural gates are retired). Reliability
        # comes from PROMPT QUALITY (the selector now reasons over the shape / spec /
        # tool DESCRIPTIONS), not hard-coded branching (d148).
        #
        # The deep-research FAMILY is keyed off the EXECUTION DISCIPLINE token
        # (``execution == "deep-research"`` → ``is_deep_research``, s16/a3 d239/d247 — re-keyed
        # off the retired round/final positions), NOT a hard-coded shape name — so adding a
        # research-family shape is adding one text file. It is an
        # EXHAUSTIVE single-topic investigation whose natural deliverable is a sourced
        # report, so it routes to the proven report path (:func:`run_plan_chain`,
        # d162/d171). OUTPUT DELIVERY (a saved file vs a chat answer vs an email) is
        # the attached SPECIALIZATION's job (html-writer → file, sse → chat, mail →
        # email; d161) — NOT a router flag, NOT a new output-mode enum.
        #
        # The ONLY conditions that remain are the JUSTIFIED constraint class — they
        # HONOR an explicit user constraint, they never PICK the route from extracted
        # booleans:
        #   * NO-WEB: the deep-research family is INHERENTLY a web shape (every round
        #     searches + fetches). If the user forbade the web (``allow_web`` False)
        #     we cannot run it — fall through to the acyclic path with web tools
        #     stripped, honoring the constraint over the topology pick.
        #   * MISSING-SPEC: a request naming an UNAVAILABLE specialization must PAUSE
        #     to define-and-resume; the acyclic path authors the terminal node the
        #     missing-specialist pause attaches that need to, so fall through there.
        # wants_file / multi_page stay recorded as OBSERVABILITY (the selector still
        # extracts them) but no longer gate anything.
        session_span.set_attribute("session.wants_file", bool(selection.wants_file))
        session_span.set_attribute(
            "session.multi_page", bool(getattr(selection, "multi_page", False))
        )
        is_deep_research = shape_spec is not None and shape_spec.is_deep_research
        session_span.set_attribute("session.is_deep_research", bool(is_deep_research))

        # SESSION RESEARCH READ-BACK (s15/a22, a14 finding #3 — d148-clean, reasoning-led).
        # A follow-up turn in the SAME chat session must answer FROM the research this session
        # already did, not re-research from a blank slate. Rehydrate the persisted notes +
        # verbatim sources (a17) and the a16 brief into the NODE-GROUNDING context the answer
        # path reads — fed to the sub-agents' user turns (alongside the prior-turn chat memory),
        # so the answer is produced FROM the accumulated knowledge. It is folded HERE, AFTER
        # shape selection, so the read-back never reaches the shape selector's goal — the route
        # stays a pure reasoning over the user's OWN request (an unrelated same-session turn,
        # e.g. the anti-over-route haiku, is unaffected). A no-op when the session has no prior
        # research (first turn / unrelated session) → byte-identical to pre-a22. Constructing
        # the session-bound ResearchState here, on the follow-up turn, is what reads the prior
        # notes/sources back so a same-session follow-up genuinely answers from them.
        node_context = _with_session_readback(conversation_context, session_id, run_id)

        # 2) SEED THE ONE GENERIC LOOP (d214/d215/d239, FORK4 / S1). There is NO bespoke web
        # fork any more: EVERY shape routes through the SAME :func:`_run_generic_loop`; this
        # only computes which PLAN the loop is SEEDED with — a RESEARCH-first plan for the web
        # deep-research family, an ACYCLIC plan for every other shape. The loop then drives
        # research → (planner reasons) → write → (planner reasons) → done → synthesizer, or a
        # single acyclic plan → done → synthesizer. The retired ``agentic.py:691`` fork chose
        # between two SEPARATE engines; this chooses a SEED for one.
        #
        # The deep-research family is INHERENTLY a web shape (every round searches + fetches),
        # so the JUSTIFIED constraint guards still divert it to the acyclic SEED:
        #   * NO-WEB: the user forbade the web → cannot run deep-research → acyclic seed, web
        #     tools stripped (honor the constraint over the topology pick);
        #   * MISSING-SPEC: a request naming an UNAVAILABLE specialization must PAUSE to
        #     define-and-resume → acyclic seed (whose authored terminal node the pause attaches
        #     the need to). wants_file / multi_page stay OBSERVABILITY only — they gate nothing.
        route_research = bool(is_deep_research and allow_web and not unmet_specs)
        research_depth: Optional[int] = None
        completeness_stop: Optional[str] = None
        if route_research:
            # RESEARCH DEPTH RESOLUTION (s13/B6 + d107(2)). Precedence, highest first:
            #   1) the per-shape UI ``depth`` override from the shapes/specs store (B6);
            #   2) the DEEP-RESEARCH SHAPE FILE's declared iteration count (d107(2));
            #   3) (when both absent) None → the loop's env baseline (RA_TREE_DEPTH).
            # The loop CLAMPS whatever it gets to N4_TREE_DEPTH_CEILING and the agent may STOP
            # EARLY (stop_research) before the bound (S5). get_depth is hasattr-guarded.
            if shape_config is not None and selection.shape and hasattr(shape_config, "get_depth"):
                research_depth = shape_config.get_depth(selection.shape)
            if research_depth is None:
                research_depth = _shape_file_research_depth(catalog, selection.shape)
            if research_depth is not None:
                session_span.set_attribute("session.research_depth", int(research_depth))
            # P2.4 (d131/d132.D) — the completeness STOP read FROM THE DEEP-RESEARCH SHAPE FILE
            # ("fill all the blanks"), a reasoned stop the decision node uses (S5: the model's
            # stop_research is the primary stop; the depth ceiling is a non-deciding safety net).
            completeness_stop = _shape_file_completeness_stop(catalog, selection.shape)
            if completeness_stop:
                session_span.set_attribute("session.completeness_stop", True)
        # RP-6b (d359/d361) — the loop's FIRST phase is DECLARED BY THE DEEP-RESEARCH SHAPE, not
        # a hardcoded research-first seed. When we route to the deep-research family we SEED the
        # loop with the shape's FIRST declared phase kind (``ShapeSpec.first_phase_kind`` →
        # ``research``); otherwise the acyclic seed. The route DECISION (route_research) stays the
        # JUSTIFIED-constraint gate (deep-research + web + no missing spec); only the SEED KIND is
        # now read from the shape rather than baked as the literal ``"research"``.
        first_plan_kind = "acyclic"
        if route_research:
            first_plan_kind = _deep_research_shape(catalog, selection.shape).first_phase_kind
        # When a deep-research shape was suppressed by the no-web / missing-spec guard there is
        # no acyclic shape discipline to honor → the default concurrent acyclic plan (None).
        seed_shape = None if (route_research or is_deep_research) else shape_spec
        session_span.set_attribute("session.first_plan_kind", first_plan_kind)
        return await _run_generic_loop(
            query,
            selection,
            first_plan_kind=first_plan_kind,
            shape_spec=seed_shape,
            transport=transport,
            registry=registry,
            hook=hook,
            plane=plane,
            timeout=timeout,
            run_id=run_id,
            # a22: node-grounding context carries the SESSION READ-BACK (prior research) so a
            # same-session follow-up grounds in it on either seed.
            conversation_context=node_context,
            # d39: the verbatim overall goal, carried onto every authored DAG so each node
            # grounds in the user's actual request (not the planner's paraphrase).
            overall_goal=overall_goal,
            allow_web=allow_web,
            requested_specs=requested_specs,
            # a8: user-requested specializations NOT registered — the acyclic gate synthesizes
            # the missing-specialist notify for them.
            unmet_specs=unmet_specs,
            research_depth=research_depth,
            completeness_stop=completeness_stop,
            # the shape catalog so the research seed can resolve + unroll the deep-research shape.
            catalog=catalog,
            # a17 (d185 Layer 3) — the CHAT SESSION id so the research state STICKS per session.
            session_id=session_id,
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
    factory = AbstractPlanFactory(
        curate_index(registry.index(), CURATED_SPECS),  # d230 planner-facing spec scoping
        tool_catalog=hook.registry.catalog(),
    )
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
    enable_reactor: bool = False,
    subagent_num_ctx: Optional[int] = None,
    grower: Optional[Any] = None,
    node_finalizer: Optional[Any] = None,
) -> tuple[AgentRuntime, Planner]:
    """Build the (runtime, planner) pair for the acyclic live path.

    Extracted so the initial run AND a missing-specialist RESUME construct an
    identical runtime (same lifecycle gate, self-heal, tool surface, call opts) —
    the resume must re-derive nothing about HOW the DAG runs, only WHICH DAG.

    F5 ``allow_web``: when False (the user forbade searching), the web tools are
    stripped from the SELF-HEAL re-planner's schema too — so a heal/replan can never
    re-introduce a web tool the initial authoring was forbidden to bind — and the
    search-then-read follow-through is disabled. Default True = the pre-F5 surface."""
    factory = AbstractPlanFactory(
        curate_index(registry.index(), CURATED_SPECS),  # d230 planner-facing spec scoping
        tool_catalog=hook.registry.catalog(),
    )
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
        # RP-3c (d330): the ``verify_lane`` wiring is RETIRED. The flag-gated engine
        # verify/revise self-review lane it turned on is gone — the model self-review moved
        # to the definition-layer writer doctrine (self-review-before-finish), and the no-fab
        # research gather-more gate is de-flagged to an output-agnostic signal gate. No
        # grounding boolean is threaded (d65 end-state: grounding is a flag-free default).
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
        # UNIVERSAL FINALIZE (d285 SB-4): the per-node finalizer the served research phase wires
        # so each finished node emits its own ``(summary, memory_index)`` digest (SB-2) into the
        # inter-node handoff. None (every other caller) → no per-node finalize, byte-identical.
        node_finalizer=node_finalizer,
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
    authoring_directive: str = "",
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
    factory = AbstractPlanFactory(
        curate_index(registry.index(), CURATED_SPECS),  # d230 planner-facing spec scoping
        tool_catalog=hook.registry.catalog(),
    )
    # F5: drop the web tools from the authoring enum when the request forbids the
    # web — a node then CANNOT bind web_search/web_fetch (the structural zero-web
    # guarantee). Default True = the full pre-F5 surface.
    offered_tools = _filter_web_tools(
        [t["name"] for t in hook.registry.catalog() if t["name"] in OFFERED_TOOLS],
        allow_web,
    )
    # RP-3b (d311/d319/d328): the F2 default-research-spec stamp is RETIRED — the planner
    # assigns a research spec to its gather nodes ITSELF (measured 100% reliable on live
    # E4B). The engine no longer passes a default research spec to author onto null-spec
    # gather nodes.
    return IncrementalPlanner(
        transport,
        factory,
        spec_names=registry.names(),
        tool_names=offered_tools,
        shape_name=(shape_spec.name if shape_spec is not None else ""),
        shape_description=(shape_spec.description if shape_spec is not None else ""),
        # RP-4c/d341: thread the SELECTED shape's decompose_methodology so the incremental
        # authoring procedure is DEFINED by the shape (e.g. the schedule-leg SCHEDULE-ONLY
        # doctrine) with precedence over the generic gather→deliver recipe. Empty for shapes
        # without the field (linear/etc) → byte-identical authoring.
        shape_decompose_methodology=(
            getattr(shape_spec, "decompose_methodology", "") if shape_spec is not None else ""
        ),
        # F5: the user-NAMED specialization(s) the authorer must bind (told + a
        # terminal-node finalization guarantee inside IncrementalPlanner).
        requested_specs=requested_specs or [],
        # s14/a9 — phase-specific per-turn authoring directive (the write phase passes the
        # SOURCE-ID mandate; empty for the research/gather planner → byte-identical).
        authoring_directive=authoring_directive,
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
    """The ACYCLIC route = the generic loop SEEDED with a single acyclic plan (d214/d239).

    No longer a separate engine: a thin SEED into :func:`_run_generic_loop`
    (``first_plan_kind="acyclic"``). The incremental node-by-node authoring + the
    missing-specialist gate + the lifecycle/self-heal runtime are UNCHANGED (they live in
    :func:`_author_and_drive_acyclic_plan`); the loop drives ONE acyclic plan, the planner
    reasons the follow-up (a simple plan → done after one iteration, FORK4), and the run EXITS
    into the terminal synthesizer. A missing-specialist plan PAUSES for the user CHOICE exactly
    as before (the loop returns the pause with no follow-up / no synthesizer). Kept as a named
    entry for the direct-call tests; the WORK is the one generic loop."""
    return await _run_generic_loop(
        query, selection,
        first_plan_kind="acyclic",
        shape_spec=shape_spec,
        transport=transport, registry=registry, hook=hook, plane=plane,
        timeout=timeout, run_id=run_id,
        conversation_context=conversation_context, overall_goal=overall_goal,
        allow_web=allow_web, requested_specs=requested_specs, unmet_specs=unmet_specs,
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


# RP-6a (d359/d361): the write-file shape is a DECLARATIVE DISK SHAPE (shapes/write-file.toml),
# NOT a hardcoded in-engine constant. Per the d341/d319 charter (behavior lives in SHAPES on disk,
# not baked in engine code), the engine LOADS it from disk via the SAME load_shape mechanism used
# for the other shapes (e.g. deep-research below, schedule-leg, codebase-summary). Behavior-
# preserving relocation — the loaded spec's name/description/execution are byte-identical to the
# former constant; no new engine conditional, no flow/format branch.
_WRITE_FILE_SHAPE = load_shape("write-file")


def _normalize_write_dag(dag: PlanDAG, out_name: str) -> PlanDAG:
    """MINIMAL single-file DELIVERY normalization (SB-6/d299 — anti-fabrication retirement).

    SB-6 RETIRED the structure-authoring this normaliser used to do. It NO LONGER stamps the
    ``file_write`` tool, NO LONGER stamps ``role=None``, and NO LONGER re-chains the nodes into a
    forced linear order. Those were the engine AUTHORING the deliverable's write shape — the exact
    fabrication the SA-6 PART-2 lesson warns against (d278). The write METHODOLOGY now lives where
    it belongs: in the ``section-html-writer`` SPEC BODY (the writer reads it via ``_compose_system``)
    and the FILE bundle doctrine (surfaced when the node SELF-SELECTS ``file``). The PLANNER authors
    the section topology + the ``depends_on`` chain by REASONING over the shape + the data-complexity
    (d10/d246). A write node is a TOOL-LESS worker; the runtime routes it to the served writer by
    the write-phase-exclusive ``deliverable_path`` delivery-context signal (``chain_sources`` is the
    source-text seam a follow-up READER also carries, so it is NOT the route discriminator — d301),
    never an engine tool/role/spec stamp.

    What REMAINS here is PURE DELIVERY (d299 DP2), justified as tool dispatch — not structure
    authoring: name the output file in each node's task so the writer knows where to write (the
    runtime's ``deliverable_path`` is the authoritative path; this only names the target). RP-1
    (d319/d311) RETIRED the 'single output document' FRAMING — the write path is OUTPUT-AGNOSTIC,
    imposing no format/single-document assumption. The planner's topology (nodes + ``depends_on``),
    specs, source_ids, tool, and role are PRESERVED VERBATIM — the engine authors NO structure and
    imposes NO tool/role/chain."""
    new_nodes: list[PlanNode] = []
    for n in dag.nodes:
        task = n.task if out_name in (n.task or "") else (
            f"{n.task}\n\nWrite to the file '{out_name}'."
        )
        new_nodes.append(replace(n, task=task))
    return PlanDAG(nodes=new_nodes, rationale=dag.rationale, goal=dag.goal)


# RP-1 (d319/d311): the ENGINE-side DAG source-coverage backstop (``_ensure_source_coverage``
# + its ``_best_match_node``/``_COVERAGE_WORD_RE`` lexical matcher) and the LEAD-synthesis /
# UNSUPPORTED-section machinery (``LEAD_SYNTHESIS_INSTRUCTION``, ``_flag_unsupported_sections``)
# are RETIRED. They were the engine REASSIGNING source_ids across nodes and DROPPING/re-chaining
# sourceless sections — engine authoring of the plan's structure. Source→section assignment is
# the model/planner's job (grounded by the SOURCE-ID ASSIGNMENT MANDATE prompt); the writer
# cites its sources directly. That authoring methodology moves to RP-2's writer spec.


# RP-1 (d319/d311): the engine-derived SERVED-ROUTE OUTLINE (``_outline_from_authored_sections``
# + its ``_section_title_from_task`` heading-derivation + the ``_SHELL_SECTION_RE`` /
# ``_LEADING_IMPERATIVE_RE`` matchers) is RETIRED. It was the engine computing nav labels /
# a section outline FROM the model's authored DAG — engine authoring of document structure. The
# MODEL authors its own nav + section headings directly (methodology → RP-2's writer spec).


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


def _research_state_path(
    run_id: Optional[str], session_id: Optional[str] = None
) -> str:
    """A path for the tree loop's persisted research-state file (d49 / a17 session binding).

    Lives under a temp ``ra_research_state`` dir (no cwd assumption). The KEY decides the
    scope:

    * ``session_id`` present (the SERVED chat route, a17 d185 Layer 3) → key by the CHAT
      SESSION so the file STICKS across turns: a follow-up in the SAME session opens the
      SAME file and reads its prior notes/sources back (``ResearchState(session_bound=True)``
      does NOT truncate), while a different session keys a different file (isolation).
    * ``session_id`` absent (the inline/offline path) → key by the run id so concurrent runs
      never share state and ``ResearchState`` truncates on open (byte-identical to pre-a17).

    The session key is namespaced (``sess__``) so it can never collide with a run-id file."""
    base = os.path.join(tempfile.gettempdir(), "ra_research_state")
    os.makedirs(base, exist_ok=True)
    if session_id:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(session_id))
        return os.path.join(base, f"sess__{safe}.jsonl")
    rid = (run_id or uuid.uuid4().hex)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(rid))
    return os.path.join(base, f"{safe}.jsonl")


# SESSION RESEARCH READ-BACK (s15/a22, a14 finding #3 — d148-clean, reasoning-led). A
# follow-up turn in the SAME chat session must ANSWER FROM the research the session already
# did — not re-research from a blank slate (the a14 gate: a same-session follow-up routed
# LINEAR and never read the a17 session-bound ResearchState back, so
# ``session_readback_on_followup`` was false). The fix below rehydrates the persisted notes +
# verbatim sources (a17) and the per-research BRIEF (a16/d185 Layer 2) and renders a compact
# block the answer path grounds in. The block is GROUNDING the model REASONS over (clearly
# labelled "use only if the current request builds on it") — NOT a route flag / boolean
# code-switch (the d148 routing-purity invariant): it is injected into the DOWNSTREAM node
# grounding ONLY, never the shape-selector goal, so an UNRELATED same-session turn (e.g. the
# anti-over-route haiku leg) still routes purely on the user's own request.
#
# The linear/chat answer path binds no ``load_source`` tool, so — per the goldilocks feed
# lesson (a model that cannot PULL must be PUSHED the few scoped sources' bodies, bounded to a
# window fraction so it never saturates num_ctx) — the few prior sources are pushed as BOUNDED
# verbatim leads, not a bodies-omitted index map.
_READBACK_LEAD_CHARS = 700       # per-source verbatim lead excerpt pushed to the answer
_READBACK_LEAD_TOTAL = 6000      # total lead budget across all prior sources (window bound)
_READBACK_MAX_SOURCES = 12       # cap the prior-source list shown


def _session_readback(session_id: Optional[str], run_id: Optional[str]) -> str:
    """Rehydrate THIS chat session's prior research and render the compact READ-BACK block.

    Opens the session-bound :class:`~agent_runtime.ResearchState` (a17) READ-ONLY — session
    binding does NOT truncate, and we only ``read``/``sources``/``collect_notes`` + project the
    a16 brief, never append — so the prior leaf notes + verbatim sources are rehydrated without
    mutating the session. Returns ``""`` when the session has no prior research (the thread's
    first turn, or a session that never ran a growable research) — a no-op that leaves the
    answer path byte-identical to pre-a22, so only a genuine follow-up gets the read-back.

    The returned block carries: the per-research BRIEF digest (thesis-level orientation), the
    running NARRATIVE (the settled COVERED facts/figures + still-open gaps, each grounded in its
    ``[S#]``), and the PRIOR SOURCES list with bounded verbatim LEADS pushed (real URLs to cite
    by ``[S#]`` + enough body for the answer to surface the figures without re-fetching)."""
    if not session_id:
        return ""
    try:
        # session_bound=True → rehydrate (no truncate). Constructing it here, on the
        # follow-up turn, is ALSO what reads the prior notes/sources back so the linear
        # answer below can answer FROM them (not just a wiped slate).
        state = ResearchState(_research_state_path(None, session_id), session_bound=True)
    except OSError as exc:
        print(f"[session-readback] state open skipped: {exc}", flush=True)
        return ""
    records = state.read()
    sources = state.sources()
    # Read back ONLY when the session actually gathered prior research (notes AND sources):
    # a fresh/unrelated session has neither → "" → no injection, no regression.
    if not records or not sources:
        return ""
    notes = state.collect_notes()
    parts: list[str] = [
        "PRIOR RESEARCH FROM THIS SESSION (the report you already researched earlier in THIS "
        "same chat — its persisted findings and the REAL sources you fetched). If the CURRENT "
        "request builds on this research, ANSWER FROM this accumulated knowledge: reuse these "
        "established facts/figures and cite these same sources by [S#] — do NOT re-research "
        "from scratch and do NOT emit an empty placeholder shell. If the current request is "
        "unrelated to this research, ignore this block."
    ]
    try:
        brief = state.research_brief(topic="")
        digest = str((brief or {}).get("digest") or "").strip()
        if digest:
            parts.append("PRIOR RESEARCH BRIEF:\n" + digest)
    except Exception as exc:  # a derived digest must never break the answer path
        print(f"[session-readback] brief projection skipped: {exc}", flush=True)
    narrative = compose_research_narrative(notes, sources)
    if narrative:
        parts.append(narrative)
    src_lines = ["PRIOR SOURCES (the REAL fetched URLs — cite these by their [S#]):"]
    budget = _READBACK_LEAD_TOTAL
    for i, s in enumerate(sources[:_READBACK_MAX_SOURCES], 1):
        title = str(s.get("title") or "").strip()
        url = str(s.get("url") or "").strip()
        src_lines.append(f"[S{i}] {title or url} — {url}")
        body = re.sub(r"\s+", " ", str(s.get("markdown") or "")).strip()
        if body and budget > 0:
            lead = body[: min(_READBACK_LEAD_CHARS, budget)]
            budget -= len(lead)
            src_lines.append(f"    {lead}")
    parts.append("\n".join(src_lines))
    return "\n\n".join(parts)


def _with_session_readback(
    conversation_context: Optional[str], session_id: Optional[str], run_id: Optional[str]
) -> Optional[str]:
    """Append the session READ-BACK block (a22) to the node-grounding context.

    The bounded prior-turn chat memory (``conversation_context``) and the heavier research
    read-back are both NODE GROUNDING — they reach every sub-agent's user turn so the answer
    is produced FROM them. Folds the read-back AFTER the prior-turn context, clearly
    delimited; returns the context UNCHANGED when there is no prior research to read back
    (no-op for a first turn / unrelated session)."""
    readback = _session_readback(session_id, run_id)
    if not readback:
        return conversation_context
    ctx = (conversation_context or "").strip()
    return f"{ctx}\n\n{readback}" if ctx else readback


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


# === s14/P3A — research MEMORY fed to the write PLANNER. Stage B (a4) moved these
# compact-memory builders DOWN into agent_runtime.research_tree so research OWNS the
# pattern and the in-research DECISION node + the write PLANNER reason over the SAME
# builders (ONE source of truth, no divergent copies). render_verbatim_source_index +
# compose_research_narrative are imported from agent_runtime; the a2 name is aliased so
# this module's call sites stay unchanged.
_compose_research_narrative = compose_research_narrative  # back-compat alias (a2 name)


# s14/a9 — the WRITE planner's PER-TURN source-id directive (the prompt-quality fix for the
# a7 "planner authored 0 source_ids on live E4B"). Surfaced on every authoring turn by
# IncrementalPlanner (the strongest lever for E4B), in addition to the goal-level mandate.
# PROMPT TEXT only — the model still reasons which [S#] each section uses (d148: no seatbelt).
_WRITE_SOURCE_ID_DIRECTIVE = (
    "SOURCE-ID ASSIGNMENT (do this on EVERY step you author): each section step that "
    "WRITES or PRESENTS content MUST set 'source_ids' to a NON-EMPTY list of the [S#] "
    "SOURCE NUMBERS from the SOURCE INDEX whose facts/figures/URLs that section uses "
    "(e.g. source_ids: [1,4]) — never author a content section with source_ids: []. "
    "Route every [S#] in the index to the section that uses it; a source may serve more "
    "than one section. Only a pure gather/search step (none here — the sources are "
    "already fetched) leaves source_ids empty."
)


def _render_write_planning_event(event: Optional[Mapping[str, Any]]) -> str:
    """Render the research reviewer's WRITE-PLANNING EVENT as a role:'user' block (d221/d189).

    The research plan's last-step reviewer emits this event; on this route it reaches the
    WRITE planner as part of its user turn (d189 events-as-user-message). It carries the
    reviewer's hand-off: a digest of WHAT the research found, the OUTLINE-FIRST sectioning
    directive (planner-authored, never a writer directive — d211), and the research-memory
    handle the section nodes bind to. Empty string when there is no event (byte-identical to
    the pre-d221 goal)."""
    if not event:
        return ""
    # SB-6/d299: the engine WRITE-METHODOLOGY DIRECTIVE ("write sectioned, OUTLINE FIRST then
    # sections" + "Author an OUTLINE/LEAD node FIRST …") is RETIRED from this render — that was the
    # engine stamping the write topology into the planner goal. The methodology now lives in the
    # ``section-html-writer`` spec body, and the planner REASONS the section topology from the DATA
    # below (data-complexity) + the shape + the spec DESCRIPTION (d10/d246). Only grounding DATA
    # remains here.
    handle = str(event.get("memory_handle") or "").strip()
    digest = str(event.get("findings_digest") or "").strip()
    complexity = str(event.get("data_complexity") or "").strip()
    lines = ["\n\nWRITE-PLANNING EVENT (from the research reviewer — your planning brief):"]
    lines.append(
        "- The research is complete and held in the bound research memory; author the "
        "WRITE plan from it."
    )
    if complexity:
        # d237 — the reviewer's data-complexity read informs one-pass-vs-sectioned (the planner
        # REASONS the section count from it; it is reported, never a directive to write sectioned).
        lines.append(
            f"- DATA COMPLEXITY (the shape of the researched data): {complexity}. Reason the "
            "number of sections from this — a simple finding may be one pass; a multi-part topic "
            "warrants one section per part."
        )
    if handle:
        lines.append(
            f"- BOUND RESEARCH MEMORY: {handle}. Each section node reads its sources from "
            "this memory via its source tools (load_source / the source index) — never "
            "expect them dumped in full."
        )
    if digest:
        lines.append(f"- WHAT THE RESEARCH FOUND (digest):\n{digest}")
    return "\n".join(lines)


def _compose_write_goal(
    query: str,
    out_name: str,
    findings: str,
    catalog: str,
    *,
    outline_hint: Optional[list[dict[str, str]]] = None,
    narrative: str = "",
    source_index: str = "",
    write_planning_event: Optional[Mapping[str, Any]] = None,
) -> str:
    """Compose the write-planner goal: the research reviewer's WRITE-PLANNING EVENT (d221,
    delivered as the planner's role:'user' brief), the B3 outline scaffold (PRIMARY, above
    findings), then the research findings + source catalog. Pure/string-only so the wiring is
    unit-testable.

    RP-1 (d319/d311) — FORMAT PINS RETIRED: the nav-SPA single-page-HTML clause and the
    'single output document / one-node-per-section / chained / lead-first' decomposition
    framing are GONE. The write path is OUTPUT-AGNOSTIC — the model picks its own
    format/structure and the PLANNER authors the topology by reasoning over the write SHAPE
    (the decomposition methodology moves to RP-2's writer spec, not this engine prompt)."""
    # s14/P3A Stage A — COMPACT-MEMORY mode: when a verbatim SOURCE INDEX is supplied the
    # planner is grounded in the running NARRATIVE (covered/gaps/direction) + the verbatim
    # index INSTEAD of the raw 12k findings blob + positional catalog (the d146(3)(4) fix:
    # run2 died at the UNSCOPED planner ingesting the blob). Degrades to the legacy
    # findings/catalog grounding when no index is available (no sources / notes), so that
    # path stays byte-identical.
    compact = bool(source_index)
    grounding_noun = "RESEARCH NARRATIVE and the SOURCE INDEX" if compact else (
        "RESEARCH FINDINGS and AVAILABLE SOURCES"
    )
    source_noun = "SOURCE INDEX" if compact else "AVAILABLE SOURCES"
    # EMPTY-NODE-NO-FABRICATE (s13/FX d106 #6, d60-critical): every section must be
    # grounded in the context below. A section with no supporting source must be
    # DROPPED or written as an explicit UNSUPPORTED line — NEVER fabricated from memory.
    no_fabricate_clause = (
        f"\n\nGROUND EVERY SECTION in the {grounding_noun} below. "
        "If a planned section has NO supporting source/finding (its research yielded "
        "nothing), DROP it or write only a single line marking it UNSUPPORTED — do NOT "
        "invent timelines, dates, figures, names, quotes, or citations from memory. Do "
        f"NOT write a placeholder or 'sources to be added later' section; cite only the "
        f"REAL fetched URLs from the {source_noun}."
    )
    # s14/P3A Stage B item 4 — SOURCE-ID ASSIGNMENT MANDATE (the PRIMARY, prompt-driven fix
    # for the a3 collapse where the planner authored 0 source_ids on every write node and the
    # writers — source-starved — wrote literal "[Source Placeholder]" into a table's citation
    # column). In compact mode the [S#] index is right there to assign FROM, so the planner
    # MUST do it. This is reliability via prompt quality (d147/d148), NOT a deterministic
    # seatbelt — the coverage backstop below only redistributes REAL ids as a last resort and
    # is recorded separately so a reviewer can confirm THIS mandate worked on its own.
    source_ids_mandate = (
        f"\n\nSOURCE-ID ASSIGNMENT IS MANDATORY: EVERY write section MUST carry a "
        f"NON-EMPTY source_ids list of [S#] numbers taken from the {source_noun} — never "
        "author a section with source_ids: []. Each [S#] in the index grounds specific "
        "facts; route every [S#] to the section that uses it (a source may serve more than "
        "one section). When a section presents a TABLE or any per-row/per-figure citation "
        "column, fill EVERY citation cell with a REAL [S#] (and/or its URL) from that "
        "section's assigned sources — NEVER write a worded stand-in such as '[Source "
        "Placeholder]', 'Source Placeholder', 'URL Placeholder for X', 'Source N Title', "
        "'TBD', 'source here', or any similar filler. If you cannot ground a row in a real "
        "[S#], drop the row rather than placeholder it."
        if compact else
        "\n\nEVERY write section MUST carry a NON-EMPTY source_ids list (the SOURCE "
        "NUMBERS it uses); fill every table/citation cell with a REAL source — never write "
        "a worded stand-in such as '[Source Placeholder]', 'Source Placeholder', 'URL "
        "Placeholder for X', 'Source N Title', 'TBD', or any similar filler."
    )
    if compact:
        context_block = (
            (f"\n\n{narrative}" if narrative else "")
            + f"\n\n{source_index}"
        )
    else:
        context_block = (
            "\n\nRESEARCH FINDINGS (decide the sections AND which sources each uses "
            f"FROM these — do NOT re-research):\n{findings}"
            + (f"\n\n{catalog}" if catalog else "")
        )
    return (
        f"{query}\n\nWrite the deliverable to the file '{out_name}'. Do NOT bind a "
        "tool to the write nodes — each is a TOOL-LESS worker that self-selects its "
        "file-authoring tools at runtime and writes per its writer spec (SB-6/d299). "
        f"Set each write node's source_ids to the SOURCE NUMBERS (from the {source_noun} "
        "below) whose facts/figures/URLs it uses — assign every relevant source to the "
        "part it belongs to, so each write node is given ONLY its own sources. "
        "DISJOINT SECTIONS (live duplication catch): the nodes together write ONE "
        "document, so assign each node a DISTINCT, NON-OVERLAPPING set of sections — "
        "never task two nodes with the same topic/section, and never re-state another "
        "node's figures as its own section. Exactly ONE node — the FINAL one — closes "
        "the document (the single sources/references section and the wrapper close); "
        "no other node's task may include sources sections or closing the document."
        + _render_write_planning_event(write_planning_event)
        + _render_outline_hint(outline_hint)
        + no_fabricate_clause
        + source_ids_mandate
        + context_block
    )


def _compose_write_planner_inputs(
    query: str,
    out_name: str,
    findings: str,
    sources: list[dict[str, str]],
    *,
    outline_hint: Optional[list[dict[str, str]]] = None,
    research_notes: Optional[Sequence[Mapping[str, Any]]] = None,
    write_planning_event: Optional[Mapping[str, Any]] = None,
    research_memory_handle: Optional[str] = None,
) -> tuple[str, Optional[list[dict[str, Any]]], str]:
    """Compose the write PLANNER's inputs from the research output — the ONE source of truth
    shared by BOTH the two-drive write phase (:func:`run_section_write_phase`) and the ONE-DRIVE
    phase-transition hook (:func:`make_write_phase_author`, RP-6c B2 / d359/d361).

    Given the research ``(findings, sources, research_notes, memory handle)``, produce the
    ``(write_goal, prior_memory, source_id_directive)`` triple the write planner needs:

    * ``write_goal`` — the numbered SOURCE catalog + the byte-faithful VERBATIM ``[S#]`` SOURCE
      INDEX (s14/P3A) + the running research NARRATIVE (covered/gaps/direction) + the reviewer's
      WRITE-PLANNING EVENT, via :func:`_compose_write_goal`;
    * ``prior_memory`` — the SB-4 ``(summary, memory_index)`` pair binding the write to the SAME
      research memory the gather built (``None`` when there is no research memory handle);
    * ``source_id_directive`` — the per-turn SOURCE-ID mandate (empty when no sources).

    Pure/string-only + byte-identical to ``run_section_write_phase``'s former inline body, so the
    two-drive path is behaviour-preserving and the one-drive hook composes the write goal from the
    run's OWN live research state through the identical builders."""
    catalog = render_source_catalog(sources)
    # s14/P3A Stage A — the COMPACT research MEMORY the write PLANNER grounds in: a byte-faithful
    # VERBATIM SOURCE INDEX keyed by the SAME stable ``[S#]`` the writers resolve + a running
    # NARRATIVE (covered/gaps/direction, built from the research ArticleNotes — no new model call).
    # Degrades to the legacy findings/catalog grounding when there are no sources to index.
    source_index = render_verbatim_source_index(sources)
    narrative = _compose_research_narrative(research_notes, sources)
    write_goal = _compose_write_goal(
        query, out_name, findings, catalog, outline_hint=outline_hint,
        narrative=narrative, source_index=source_index,
        write_planning_event=write_planning_event,
    )
    # s14/a9 — the per-turn SOURCE-ID directive (the strongest lever for E4B); only when there is
    # a SOURCE INDEX to assign FROM. The model still reasons which ``[S#]`` each section uses.
    source_id_directive = _WRITE_SOURCE_ID_DIRECTIVE if sources else ""
    # UNIVERSAL FINALIZE HANDOFF (d285 SB-4), P2 de-fabrication: the handoff summary is
    # now the code-assembled bounded DIGEST (index + note gists + pull cursor) — not the
    # retired engine truncation of raw findings (`findings[:1200]`), which was the exact
    # engine-extracted push the owner rejected. The digest tells the write planner what
    # EXISTS and how nodes PULL it (read_notes/load_source).
    if research_memory_handle:
        from chat_app.digest import build_research_digest

        prior_memory = [{
            "summary": build_research_digest(sources, research_notes, token_budget=800),
            "memory_index": research_memory_handle,
        }]
    else:
        prior_memory = None
    return write_goal, prior_memory, source_id_directive


# RP-6c B2 (O4, DESIGN §b/§f) — the ONE-NODE write-authoring contract. The write-file shape's
# DESCRIPTION still biases the planner toward the N-section CHAIN (one whole-document node per
# section — Bug B; the stale write-file.toml text cleanup is RP-6d). Until that lands, the
# one-drive hook steers the planner to ONE coherent-document node via the per-turn AUTHORING
# DIRECTIVE (the sanctioned prompt lever, strongest on E4B) — NOT by editing the shape file and
# NOT by the engine authoring structure (the MODEL still authors the topology via
# ``IncrementalPlanner.plan``; this only DIRECTS it, like the source-id mandate). O4 permits an
# additional single ``final_review`` node; it forbids the N-way whole-document chain.
_ONE_WRITE_NODE_DIRECTIVE = (
    "AUTHOR THE WRITE PLAN AS ONE COHERENT DOCUMENT. The deliverable is a SINGLE file: author "
    "EXACTLY ONE write node that produces the WHOLE document — its own file_write -> file_read -> "
    "continue -> finish loop accumulates every part of that one document IN ORDER, so the "
    "'sections' are that single node's WITHIN-NODE structure, never separate nodes. Do NOT author "
    "one node per section, and do NOT chain N whole-document nodes (that re-emits and DUPLICATES "
    "the document). ALSO author EXACTLY ONE 'final_review' node with role='reviewer' that "
    "depends_on the write node, binding the SAME format spec as its review rubric: the reviewer "
    "READS the finished file itself (file_read), verifies it against the goal and that spec, "
    "FIXES any defect it finds ITSELF via file_update (grounded, minimal edits — never a rewrite "
    "from scratch), and reports an honest final status of what the document actually contains "
    "(and anything still missing). Author NO other nodes."
)


def make_write_phase_author(
    *,
    query: str,
    out_name: str,
    transport,
    registry: SpecRegistry,
    hook: ToolHook,
    requested_specs: Optional[list[str]] = None,
    outline_hint: Optional[list[dict[str, str]]] = None,
):
    """Build the REAL one-drive WRITE-AUTHORING hook (RP-6c B2 / d359/d361, DESIGN §b/§c).

    Returns a :data:`~agent_runtime.runtime.PhaseAuthor` callable ``(rt, dag, next_plan) ->
    list[PlanNode]`` — the LIVE implementation of B1's injected minimal hook that the one-drive
    phase transition (:class:`~agent_runtime.runtime.PhaseTransition`) invokes on research stop.
    It fills B1's mechanism with the real work:

    1. COMPOSE the write goal from the run's OWN LIVE research state (findings + the verbatim
       ``[S#]`` SOURCE INDEX + the research NARRATIVE + the SB-4 ``(summary, memory_index)`` pair),
       read off the SHARED runtime (``rt._cache`` research results via the SAME collectors the
       two-drive path uses; the research memory handle off the run's ``grower.state``) — NOT engine
       locals hand-carried across a runtime boundary.
    2. CALL ``IncrementalPlanner.plan(write_goal, prior_memory=…)`` mid-drive so the MODEL authors
       the write topology (the engine composes only DATA + INVOKES the planner — no engine
       structure/content authoring, anti-fab d310/d317/d319).
    3. Return ONE coherent-document write node (+ optional ``final_review`` per O4), steered by the
       :data:`_ONE_WRITE_NODE_DIRECTIVE` — NEVER the N-section whole-document chain (Bug B).

    The research→write handoff rides as the write node's CONTEXT from the SHARED ``ResearchState``,
    so the ``chain_sources``/``chain_notes`` cross-runtime bridge is DROPPED on this path (Bug C):
    the write node runs on the SAME runtime as the research, so its notes bind from the shared
    run's upstream research deps (``SubAgent._collect_upstream_notes``, keyed to the same ``[S#]``)
    and its sources are composed into the write goal from the run's own state — no engine glue and
    no separate ``write_runtime`` side-channel. (Reliable ``load_source`` FORCE-bind on the writer
    node is the RP-6d residual per DESIGN §f.)

    The drive (:meth:`AgentRuntime._author_next_phase`) STAMPS the per-node ``deliverable_path``
    (O1), wires the deps-less node(s) onto the research sinks, appends them to the LIVE dag and
    drives them in the SAME run — so this hook returns the model-authored node(s) UN-stamped and
    UN-wired (the drive owns that structural ordering, never this hook)."""
    _requested = list(requested_specs or [])

    async def _author_write_phase(rt, dag, next_plan):
        from types import SimpleNamespace

        # LIVE RESEARCH STATE off the SHARED run (DESIGN §c): the finished research nodes' results
        # are in ``rt._cache``; render them with the SAME collectors the two-drive path runs over
        # the final RunResult — findings prose, the run's global fetched SOURCES (stable ``[S#]``),
        # and the accumulated ArticleNotes — so the write goal is composed from the run's own state.
        research_ids = [n.id for n in dag.nodes if n.id in getattr(rt, "_cache", {})]
        live = SimpleNamespace(results=rt._cache, launch_order=research_ids)
        findings = _collect_findings(live)
        sources = _collect_chain_sources(live)
        research_notes = _collect_article_notes(live)
        # The shared ResearchState's stable memory handle (bind the write nodes to the SAME
        # research memory the gather built) — read from the run's OWN grower, never a passed local.
        grower = getattr(rt, "_grower", None)
        state = getattr(grower, "state", None) if grower is not None else None
        memory_handle = (getattr(state, "memory_handle", "") or "") if state is not None else ""
        write_planning_event = _build_write_planning_event(
            findings, sources, memory_handle=memory_handle,
        )
        write_goal, prior_memory, source_id_directive = _compose_write_planner_inputs(
            query, out_name, findings, sources, outline_hint=outline_hint,
            research_notes=research_notes, write_planning_event=write_planning_event,
            research_memory_handle=memory_handle,
        )
        # DIRECTIVE = the O4 ONE-NODE contract FIRST, then the per-turn source-id mandate.
        directive = _ONE_WRITE_NODE_DIRECTIVE + (
            "\n\n" + source_id_directive if source_id_directive else ""
        )
        write_planner = _build_incremental_planner(
            transport=transport, registry=registry, hook=hook,
            shape_spec=_WRITE_FILE_SHAPE, allow_web=False,
            requested_specs=_requested, authoring_directive=directive,
        )
        # The MODEL authors the write topology (engine composes only the DATA above + INVOKES the
        # planner). ``_normalize_write_dag`` names the output file in each node's task (pure
        # DELIVERY, d299) — it authors NO structure/tool/role/chain.
        w_plan = await write_planner.plan(write_goal, prior_memory=prior_memory)
        write_dag = _normalize_write_dag(w_plan.dag, out_name)
        # MEMORY-BY-HANDLE (d221): BIND every authored write node to the research memory so its
        # context names the handle and its source tools resolve against the run's OWN sources.
        if memory_handle:
            write_dag = replace(write_dag, nodes=[
                n if n.research_memory_handle
                else replace(n, research_memory_handle=memory_handle)
                for n in write_dag.nodes
            ])
        # Return the MODEL-authored node(s); the drive stamps deliverable_path (O1) + wires the
        # deps onto the research sinks + drives them in the SAME run. NO chain_sources/chain_notes
        # set on ``rt`` (Bug C: the handoff rides as node context from the shared ResearchState).
        return list(write_dag.nodes)

    return _author_write_phase


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
    research_notes: Optional[Sequence[Mapping[str, Any]]] = None,
    write_planning_event: Optional[Mapping[str, Any]] = None,
    research_memory_handle: Optional[str] = None,
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
    # RP-1 (d319/d311): the format-pinned NAV-SPA framing is RETIRED — the write path is
    # OUTPUT-AGNOSTIC. The model picks its own format/structure; the planner authors the
    # topology by reasoning over the write SHAPE. No is_html gate, no single-page-HTML clause.
    # s13/B3: the agent-decided research OUTLINE (outline_hint) is woven in as the PRIMARY
    # section scaffold (above the findings) so the document direction reaches the writer;
    # an empty outline keeps the findings-driven decomposition unchanged (d56).
    # s14/P3A Stage A — replace the write PLANNER's raw 12k findings blob with COMPACT
    # research MEMORY: a running NARRATIVE summary (covered/gaps/direction, built from the
    # research's own ArticleNotes — no new model call) + a byte-faithful VERBATIM SOURCE
    # INDEX keyed by the SAME stable [S#] the per-section writers resolve. The writers are
    # UNCHANGED (chain_sources below still feeds each section its full scoped text); only the
    # planner's grounding input changes. Degrades to the legacy findings/catalog when there
    # are no sources to index (source_index == "" → _compose_write_goal's legacy branch).
    #
    # RP-6c B2 (d359/d361): this composition (catalog + verbatim [S#] SOURCE INDEX + research
    # NARRATIVE + the SB-4 prior_memory + the per-turn source-id directive) is factored into
    # ``_compose_write_planner_inputs`` — the ONE source of truth this two-drive write phase and
    # the one-drive phase-transition hook (:func:`make_write_phase_author`) both compose through.
    # Behaviour is byte-identical to the former inline body.
    write_goal, _write_prior_memory, write_directive = _compose_write_planner_inputs(
        query, out_name, findings, sources, outline_hint=outline_hint,
        research_notes=research_notes, write_planning_event=write_planning_event,
        research_memory_handle=research_memory_handle,
    )
    # P2-5c (d65 FLAG-FREE) — the SERVED report write phase now ALWAYS turns ON the
    # framework-injected review (the write planner authors work nodes; the FRAMEWORK adds a
    # work->review pair + a final review) AND the event-driven reactor on the write runtime.
    # These were gated behind the retired RA_GENERIC_REPORT_PATH flag; with the generic engine
    # the served default they are LIVE on the served route (resolving the P2-5-review
    # flag-coupling). The (findings, sources) contract + the per-section bounded SPA write are
    # otherwise unchanged.
    # s14/a9 — PER-TURN SOURCE-ID DIRECTIVE for the WRITE planner (the strongest lever for E4B):
    # ``write_directive`` above is the source-id mandate composed by ``_compose_write_planner_inputs``
    # (empty when there are no sources to assign FROM) — the model still chooses which [S#] each
    # section uses (d148: no seatbelt).
    # SF-1 (d310/d311) — the framework review injection is RETIRED (the model authors the whole
    # document; no engine reviewer edits it), so the write planner authors the body-node DAG
    # with no review twins.
    # P2 (Gate-2d live catch): the write planner gets the ONE-write-node directive on
    # THIS (two-drive) path too — 4 write nodes × (~62s/call self-select + pull + write
    # turns) starve each node to ~5 turns inside the phase budget, yielding one thin
    # section apiece. ONE node accumulating the whole document through its own
    # write → read-back → continue loop (exactly the file doctrine's design) spends
    # the same budget on ONE coherent artifact. Planner-directive prompt lever
    # (sanctioned, same as the one-drive hook) — the model still authors the topology.
    write_planner = _build_incremental_planner(
        transport=transport, registry=registry, hook=hook,
        shape_spec=_WRITE_FILE_SHAPE, allow_web=False, requested_specs=requested_specs,
        authoring_directive=(_ONE_WRITE_NODE_DIRECTIVE + "\n\n" + write_directive
                             if write_directive else _ONE_WRITE_NODE_DIRECTIVE),
    )
    write_runtime, _ = _build_acyclic_runtime(
        transport=transport, registry=registry, hook=hook, plane=plane,
        shape_spec=_WRITE_FILE_SHAPE, conversation_context=None, allow_web=False,
        # RP-3c (d330): the post-write engine verify lane is retired — the model self-review
        # (ground-or-drop unbacked claims before finishing) now lives in the writer doctrine.
        enable_reactor=True,
    )
    # SOURCE-SCOPING (d56): hand the runtime the run's global source list so each
    # section node resolves its planner-assigned source_ids to the real text/URLs. NOTE: chain_sources
    # is NOT the write-route discriminator — a follow-up READER also carries chain_sources to resolve
    # prior sources (d301); the route keys on ``deliverable_path`` (set below), which is write-phase
    # exclusive.
    write_runtime.chain_sources = sources
    # SA-4 (d234/d235) — feed the run's research NOTES to the write runtime via the new
    # ``chain_notes`` seam (the mirror of ``chain_sources`` above). The runtime's
    # ``_node_run_ctx`` folds these into ``ctx['notes']`` so a write/review node that
    # SELF-SELECTS ``research_read`` binds ``read_notes`` (the CHEAP first leg of the read
    # hierarchy) through the working growth point — replacing the retired per-run read_notes
    # pre-registration below. Empty/None research_notes → no notes fed (byte-identical).
    write_runtime.chain_notes = research_notes
    # s15/a27 (fix b) — the single deliverable path the framework FINAL REVIEWER edits in place.
    # The write phase writes exactly ONE file (``out_name``); handing the runtime the authoritative
    # path makes the reviewer's target resolution deterministic instead of scanning the writer
    # cache for a ``tool_value['path']`` that a no-emit writer may never have produced (the
    # measured review.anchored==0 silent skip). The generic per-node review route does NOT set
    # this, so it keeps the cache-scan behaviour (byte-identical).
    #
    # ALL-WRITERS INVARIANT (SB-6/d301) — LOAD-BEARING for the write-route soundness. ``deliverable_path``
    # is ALSO the WRITE-ROUTE discriminator: a TOOL-LESS write node routes to the served writer
    # (_run_file_delivery) iff its runtime has deliverable_path set (runtime.SubAgent.run), now that
    # the engine no longer stamps tool=file_write to mark writers. This is SOUND because deliverable_path
    # is set on a DEDICATED write runtime here and NOWHERE else in production (research / gather /
    # follow-up read run in SEPARATE runtimes with no deliverable_path), AND the write PLANNER authors
    # ONE WRITE node per section (see _compose_write_goal) — no gather/analysis/summarise nodes. So
    # every non-review node in a deliverable_path runtime IS a section writer; routing them all to the
    # writer is correct. (chain_sources was NOT used as the discriminator precisely because a follow-up
    # READER carries it without being a writer — d301.) The write itself is the d49/d50 DELIVERY of the
    # model's spec-driven raw emission (the orchestration PERSISTS it; not self-select-GATED — gating a
    # pre-loop route on a self-select that happens INSIDE the loop is circular, d299). FUTURE RISK this
    # invariant guards: keep this runtime write-only — a NON-writer run in a deliverable_path runtime
    # would be forced to write a file.
    write_runtime.deliverable_path = out_name
    # A0 (SA-1) — the per-run ``load_source`` PRE-REGISTRATION IS RETIRED. It was the mask that
    # papered over the registry-wiring gap (hook.registry was a base ToolRegistry with no .add, so
    # bundle self-select silently no-op'd). With the A0 fix (hook.registry is the GrowableToolRegistry,
    # register_agentic_tools assigns it back), the write/review nodes that SELF-SELECT the
    # ``research_read`` bundle now genuinely register ``load_source`` through the working growth point —
    # bound to THIS run's verbatim sources via the runtime's ``_node_run_ctx`` (which feeds
    # ``ctx['sources']`` from ``write_runtime.chain_sources``, set above). No per-run hook.register and
    # no flag/fallback (d186): the served web write path obtains load_source the same way every bundle
    # does.
    # SA-4 — the per-run ``read_notes`` PRE-REGISTRATION IS RETIRED (it was the SA-1-deferred
    # bridge). The matching ``chain_notes`` seam now feeds ``write_runtime.chain_notes`` (set
    # above) → the runtime's ``_node_run_ctx`` exposes ``ctx['notes']`` → a write/review node
    # that SELF-SELECTS ``research_read`` registers ``read_notes`` through the working growth
    # point, bound to THIS run's notes and keyed to the SAME global [S#] as load_source (via the
    # sources list the bundle also reads from ctx). So both legs of the d234/d235 read hierarchy
    # (read_notes CHEAP → load_source EXPENSIVE) bind by self-select with NO per-run pre-reg and
    # NO remaining mask — the served web write path obtains read_notes the same way every bundle
    # obtains its tools.
    # UNIVERSAL FINALIZE HANDOFF (d285 SB-4): ``_write_prior_memory`` above is the research
    # phase's ``(summary, memory_index)`` pair (composed by ``_compose_write_planner_inputs``) —
    # so the write planner reasons over that research line and can CONTINUE its index on the
    # section nodes it authors, binding the write to the SAME research memory the gather built.
    # Absent (no research memory / no findings) → None → the SB-3 seam renders nothing.
    w_plan = await write_planner.plan(write_goal, prior_memory=_write_prior_memory)
    # s15 thread-2 (HOLLOW-RENDER FIX): GUARANTEE a SHELL-ONLY lead + one real BODY-author node
    # per outline section (the section LIST stays the model-decided outline; each body's CONTENT
    # is authored live from its scoped sources — no hardcoded prose), THEN wrap each body with
    # its framework review twin (relocated from the planner build). No-op when there is no usable
    # multi-section outline (short reports keep the planner's own decomposition byte-identical).
    #
    # s15/a8 — SERVED-ROUTE TRIGGER: the served generic-engine routes (run_plan_chain /
    # _run_deep_research_sectioned) supply NO research outline_hint, so this guarantee used to
    # no-op there and the report went hollow (a7). When no research outline is supplied, derive
    # the model-decided section list from the write planner's OWN authored DAG so the body-author
    # nodes FIRE on the served route too. A real outline_hint (the tree path) still wins, so that
    # path is byte-identical.
    # D2 (d216) — EMERGENT SECTIONING: the write planner's OWN authored DAG IS the section
    # decomposition (it authored one file_write node per section it decided from the research).
    # The prescribed lead+body scaffold (``_ensure_section_body_dag``) is DELETED (d216/d218) — no
    # deterministic 'shell-then-one-body-node-per-section' structure and no 'start with a lead
    # page / you are a sectioned writer' directive is imposed on the model. SF-1 (d310/d311)
    # also RETIRES the deterministic HTML assembly/surgery + the framework review injection:
    # the MODEL authors the whole document (skeleton + content + nav + Sources) and the engine
    # authors/fixes nothing (SF-2/SF-3 harden the skeleton-then-fill spec so the report is
    # coherent by construction; the write phase is degraded-but-functional until then).
    write_dag = _normalize_write_dag(w_plan.dag, out_name)
    # MEMORY-BY-HANDLE (d221, write side): BIND every write/review node to the research
    # memory so its context names the handle and it reads sources via its tools (the runtime
    # renders "Binded research memory: <handle>"). A node that already carries a handle (none
    # here — the write planner authors none) is left untouched. No-op when no handle was
    # supplied (the non-report callers), keeping those paths byte-identical.
    if research_memory_handle:
        write_dag = replace(write_dag, nodes=[
            n if n.research_memory_handle
            else replace(n, research_memory_handle=research_memory_handle)
            for n in write_dag.nodes
        ])
    # s14/P3A Stage B item 4 — OBSERVABILITY: capture how many sections the PLANNER itself
    # authored with non-empty source_ids (the PRIMARY prompt fix working) BEFORE the backstop
    # touches anything. Recorded on the span next to the backstop's own fired/mode signal so a
    # reviewer/live gate can tell "the LLM authored real source_ids" apart from "the backstop
    # papered it over" (d147/d148 — no hardcoded behavior may mask weak LLM behavior).
    planner_authored_source_id_nodes = sum(1 for nd in write_dag.nodes if nd.source_ids)
    n_write_nodes_pre = len(write_dag.nodes)
    # s15/a27 (fix b) — COUNT the framework review nodes wired into the executed write DAG. The
    # d184 served report carries EXACTLY ONE ``final_review`` over the writer sinks; surfacing the
    # count lets a gate assert reviewer_node_count == 1 (the final reviewer is present + wired
    # AFTER the writers) without a live run, separate from whether it then fired (the
    # ``review.anchored`` span).
    reviewer_node_count = sum(
        1 for nd in write_dag.nodes
        if str(nd.id).endswith("_review") or str(nd.id) == "final_review"
    )
    # RP-1 (d319/d311): the ENGINE source-coverage backstop (``_ensure_source_coverage`` —
    # redistributing source_ids across nodes) and the UNSUPPORTED-section flag/drop pass
    # (``_flag_unsupported_sections`` — dropping sourceless sections + re-chaining the DAG) are
    # RETIRED. Both were the engine AUTHORING/reassigning the plan's structure. The MODEL/planner
    # owns source→section assignment (grounded by the SOURCE-ID ASSIGNMENT MANDATE prompt) and
    # the writer cites its sources directly (methodology → RP-2's writer spec). Only OBSERVABILITY
    # of the planner's own authoring remains — no engine reassignment/drop.
    try:
        from opentelemetry import trace as _otel_trace
        _span = _otel_trace.get_current_span()
        _span.set_attribute("write_phase.planner_authored_source_id_nodes",
                            int(planner_authored_source_id_nodes))
        _span.set_attribute("write_phase.write_nodes", int(n_write_nodes_pre))
        _span.set_attribute("write_phase.reviewer_node_count", int(reviewer_node_count))
    except Exception:  # pragma: no cover - observability must never break the write path
        pass
    # D1 (d216) — the per-section NOTE/SOURCE PUSH (``_inject_section_notes``) is DELETED (d216/d218). The
    # served write worker is now a TOOL-CALLING READER (runtime ``_run_tool_calling_writer``):
    # it PULLS the verbatim text of the sources its section needs with load_source on demand and
    # writes from that REAL text, instead of being handed a pushed notes/source blob in its task
    # (a bounded push IS the verbatim dump the user rejected, d202/d205/d211). The reviewer keeps
    # its own on-demand load_source pull. ``research_notes`` still feeds the write PLANNER's
    # compact NARRATIVE (covered/gaps/direction) above; only the per-writer task-push is gone.
    write_dag.goal = f"{overall_goal or query}\n\nWrite the document to '{out_name}'."
    w_result = await write_runtime.run(write_dag, timeout=timeout, run_id=run_id)

    # RP-1 (d319/d311): the final well-formedness close-gap pass is RETIRED — the engine
    # authors/fixes NOTHING. The deliverable ships EXACTLY as the write phase produced it
    # (raw model output; coherence + well-formedness are the model's own responsibility,
    # hardened via the writer SPEC in RP-2, never an engine post-processing fix).
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
        if spec is None or not getattr(spec, "is_deep_research", False):
            spec = catalog.get("deep-research")
    if spec is None:
        # No usable shape in the passed catalog → load the shipped canonical deep-research
        # shape directly (the served-route invariant: the report path always has a shape).
        spec = load_shape("deep-research")
    return spec


def _research_seed_dag(shape: "ShapeSpec", goal: str, *, spec: Optional[str]) -> PlanDAG:
    """The ENGINE-owned GROWABLE research SEED (s16/a3 d239/d247 — RETIRE THE UNROLL).

    Replaces the deleted ``shapes.unroll_shape`` on the research path: the SHAPE no longer
    pre-bakes any node graph nor binds a gather tool. The engine emits ONE research SEED node
    carrying ONLY the goal (position-framed for the model to investigate) and tags the
    :class:`PlanDAG` ``growable`` — the :class:`~agent_runtime.research_tree.DagGrower` then
    AUTHORS the real topology by REASONING (decompose-first → grow on note gaps via
    ``run_decision_node``), bounded by the shape's ``fan_out`` / ``max_layers`` / ``max_sources``.

    d242 TRUE SELF-SELECT: the seed is TOOL-LESS — NEITHER the shape NOR this builder binds
    ``web_search``. Like every in-plan node (as2/d242) it starts with only ``get_bundles`` +
    ``finish`` and SELF-SELECTS its research bundle (which carries the configured search/fetch
    tools) at runtime. as4 DE-WEB (d227/d241): the seed is :data:`ROLE_RESEARCHER` (the gather
    node TYPE, d213) — so even tool-less it ROUTES to the runtime's research loop AND the
    DagGrower folds it source-agnostically (HEADSUP1: the rare decompose-empty FALLBACK path
    where this whole-goal seed is the gather node, not replaced). In the normal served path the
    grower's decompose-first ``seed_layer`` REPLACES this whole-goal seed with the scoped facet
    nodes before any gather — this seed is the breadth-fallback the grower keeps when the model
    authors no decomposition (never empty)."""
    from agent_runtime.roles import ROLE_WORKER, position_framing
    from specialization.seed import RESEARCH_METHODOLOGY_SPEC

    # SB-RR (d292/d293): the gather SEED is a WORKER-default node (d273) carrying the
    # research-METHODOLOGY spec — research is a SPECIALIZATION, not a role. That spec's body
    # ("self-select your gather bundle first…") makes the seed worker self-select the gather
    # bundle and reach the unified worker loop's gather behavior; ROLE_RESEARCHER is RETIRED.
    # Compose the methodology AHEAD of the round's output-quality spec; dedup if equal.
    extra = (spec,) if spec and spec != RESEARCH_METHODOLOGY_SPEC else ()
    specs = (RESEARCH_METHODOLOGY_SPEC,) + extra
    seed = PlanNode(
        id="r1_research",
        task=f"[research · round 1] {position_framing('research')}\n\n{goal}",
        spec=RESEARCH_METHODOLOGY_SPEC,
        specs=specs,
        depends_on=(),
        # WORKER-default (d273): every spawned node is a worker; the gather behavior comes from
        # the research-methodology SPEC self-selecting the gather bundle, NOT a role. The grower
        # still folds this fallback seed source-agnostically via the research_memory_handle it
        # binds on (as4 HEADSUP1), bound by the inject path, not a role.
        role=ROLE_WORKER,
        # TOOL-LESS (d242): no web_search bind — the node self-selects its research bundle.
        tool=None,
        tool_args={},
    )
    return PlanDAG(
        nodes=[seed],
        rationale=(
            f"{shape.name} growable research seed (1 tool-less self-selecting node; the grower "
            f"authors topology by reasoning — decompose-first then grow on note gaps, "
            f"max_layers={shape.max_layers or 'cfg'}, fan_out={shape.fan_out or 'cfg'})"
        ),
        shape=shape.name,
        growable=True,
        fan_out=int(shape.fan_out),
        max_layers=int(shape.max_layers),
        max_sources=int(shape.max_sources),
    )


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
    # The run's user-requested specs — part of the phase's uniform input contract (mirrors
    # the write phase). RP-6b (d359/d361 / Bug A d355/d356): the research phase DELIBERATELY
    # does NOT route these onto the research seed — its spec is the shape's declared research
    # role (``_deep_research_spec(..., shape=dr_shape)`` below). A user-named WRITER/output spec
    # reaches the WRITE phase instead, so a writer spec never lands on a research node.
    requested_specs: Optional[list[str]],
    dr_shape: "ShapeSpec",
    research_depth: Optional[int],
    completeness_stop: Optional[str] = None,
    session_id: Optional[str] = None,
) -> tuple[str, list[dict[str, str]], dict[str, Any]]:
    """P2.5/P2-5c — the GENERIC-engine PHASE-1 research, now the SERVED DEFAULT (d65 flag-free).

    Replaces the bespoke ``run_research_tree`` layer loop with the SAME generic engine the
    inline path uses (d115/d128): build the ENGINE-OWNED GROWABLE SEED (:func:`_research_seed_dag`
    — a single tool-less self-selecting research node, s16/a3 d239/d247; no ``unroll_shape``, no
    shape-bound ``web_search``), and DRIVE it on the generic :class:`AgentRuntime` with the
    report-route grounding lanes ON (notes + chunked-read + verify) and the P2.2 event-driven
    reactor wired. The accumulated ``(findings, sources)`` hand to the SAME PHASE-2
    :func:`run_section_write_phase`, so ONLY the research ENGINE changes — the write side, the
    ``(findings, sources)`` contract and the d50/d60 grounding invariants are byte-identical.

    ITERATIVE BREADTH (P2-5b, d134/d135 — parity HELD): the seed is tagged ``growable``, and
    this phase builds a :class:`DagGrower` (below) that REUSES the SAME ``ResearchState`` +
    ``Tree`` + ``run_decision_node`` + ``completeness_stop`` the retired tree used. The runtime
    drive loop (:meth:`AgentRuntime._drive_growable`) then DECOMPOSE-FIRST-seeds the goal into
    scoped children and grows wave-by-wave on note gaps — reproducing ``run_research_tree``'s
    state-driven re-expansion WITHOUT a second engine. P2-5b proved within-run, same-budget,
    that generic breadth meets-or-exceeds the tree, grounded; per d65 the reversible flag was
    RETIRED in P2-5c and this is the served default on BOTH report routes."""
    # RP-6b (d359/d361) — the research-phase spec is ROUTED BY THE SHAPE's declared spec_role
    # (research → the research-analysis spec), NOT by grabbing a requested writer spec (Bug A).
    spec_name = _deep_research_spec(registry, shape=dr_shape)
    # ENGINE-OWNED GROWABLE SEED (s16/a3 d239/d247 — unroll_shape RETIRED). The shape pre-bakes
    # NO node graph and binds NO gather tool: the engine emits ONE tool-less self-selecting
    # research seed (d242) + tags the DAG ``growable``; the DagGrower wired below AUTHORS the
    # topology by reasoning (decompose-first → grow on note gaps). The shape/UI depth is honored
    # by the grow_config depth clamp below; per-leaf fetch breadth stays PINNED to the report
    # contract (D97). No _research_only_dag strip needed — the seed is already research-only.
    research_dag = _research_seed_dag(dr_shape, overall_goal or query, spec=spec_name)
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
        # d235 — the decision node's investigative METHODOLOGY comes from the ROLE/RUNTIME (the
        # research bundle doctrine), NOT the research-analyst SPEC (now output-quality only). This
        # fixes the spec-vs-role blur: the spec shapes how the answer READS; the research loop
        # methodology (decompose-first → grow out → expand/prune → stop) drives how it WORKS.
        from agent_runtime.bundles.research import RESEARCH_METHODOLOGY
        methodology = RESEARCH_METHODOLOGY
        # fan_out: shape-declared (the iterative breadth cap), else the config default.
        fan_out = int(getattr(research_dag, "fan_out", 0)) or grow_config.fan_out
        grower = DagGrower(
            transport=transport,
            goal=overall_goal or query,
            spec=spec_name,
            config=grow_config,
            # a17 (d185 Layer 3) — SESSION-BOUND research state on the served chat route:
            # when session_id (=chat_id) is present the state file is keyed by the session
            # and STICKS (no truncate), so a follow-up turn reads prior notes/sources back.
            # Absent (inline/offline) → run-scoped + truncate-on-open (byte-identical pre-a17).
            state=ResearchState(
                _research_state_path(run_id, session_id),
                session_bound=bool(session_id),
            ),
            tree=Tree(fan_out=fan_out),
            methodology=methodology,
            # REUSE the shape's completeness_stop VERBATIM as the decision-node stop signal.
            stop_criteria=completeness_stop or getattr(dr_shape, "completeness_stop", "") or None,
            # s14/a15 (d160/d161) — REUSE the shape's decompose_methodology VERBATIM as the
            # DECOMPOSE-FIRST seed's breadth doctrine, so the seed RELIABLY authors ≥3 scoped
            # facets as a SHAPE PROPERTY the model reasons over (curing the d160 thin-report
            # 1-source collapse). None → the baked default decompose wording (byte-identical).
            decompose_criteria=getattr(dr_shape, "decompose_methodology", "") or None,
            max_layers=int(getattr(research_dag, "max_layers", 0)),
        )
    # UNIVERSAL FINALIZE (d285 SB-4) — the per-node finalizer wired onto the SERVED research
    # runtime. After each research node finishes, it (1) OPENS/continues that branch's research
    # memory by the node's brief ``memory_index`` via SB-1's ``resolve_brief_memory`` (the
    # per-branch opening SB-3 deferred) and folds the node's gathered leaf into it — so the
    # handed-off index names a REAL, pullable store — and (2) asks SB-2's
    # ``Planner.finalize_node`` for the NODE's OWN model digest, fed the node's real output as
    # ``work_digest`` (so the summary is genuinely that node's, not an orchestration blurb). The
    # ``(summary, memory_index)`` pair becomes the SOLE inter-node context a downstream node
    # receives (``SubAgent._compose_task`` drops the clipped prose + folded fetched bodies). The
    # grower's ``self.state`` cross-layer aggregate is folded UNCHANGED (decision loop untouched);
    # this writes the SAME leaf additionally into the per-branch memory. d49-clean: the writer is
    # PUSHED its scoped sources via a SEPARATE path (chain_sources) — this never model-PULLs.
    _finalize_planner = _loop_reasoning_planner(transport, registry, hook)
    _branch_mem_root = os.path.join(tempfile.gettempdir(), "ra_branch_memory")
    _research_goal = overall_goal or query

    async def _node_finalizer(node, result):
        idx = (getattr(node, "research_memory_handle", "") or "")
        n_sources = 0
        try:
            tv = getattr(result, "tool_value", None)
            notes: list[dict[str, Any]] = []
            fetched: list[dict[str, Any]] = []
            if isinstance(tv, Mapping):
                notes = [dict(x) for x in (tv.get("article_notes") or tv.get("notes") or [])
                         if isinstance(x, Mapping)]
                fetched = [dict(x) for x in (tv.get("fetched") or tv.get("records")
                           or tv.get("chunks") or []) if isinstance(x, Mapping)]
            n_sources = len(fetched)
            # PER-BRANCH OPENING (SB-1 resolver): index → continue that memory; <<NEW>>/unset →
            # mint a fresh one. Fold this node's leaf in so the index is a real, pullable store.
            bstate = resolve_brief_memory(_branch_mem_root, getattr(node, "memory_index", None))
            bstate.append_leaf(
                LeafResult(
                    branch_id=str(node.id),
                    question=(getattr(node, "task", "") or _research_goal),
                    findings=(getattr(result, "output", "") or ""),
                    notes=notes, fetched=fetched,
                ),
                layer=0,
            )
            idx = bstate.memory_handle or idx
        except Exception:  # noqa: BLE001 - per-branch opening must never break a node's run
            pass
        nf = await _finalize_planner.finalize_node(
            getattr(node, "task", "") or _research_goal,
            memory_index=idx,
            work_digest=(getattr(result, "output", "") or "")[:4000],
            sources=n_sources,
        )
        return {"summary": nf.summary, "memory_index": nf.memory_index}

    runtime, _ = _build_acyclic_runtime(
        transport=transport, registry=registry, hook=hook, plane=plane,
        shape_spec=dr_shape, conversation_context=None, allow_web=True,
        node_finalizer=_node_finalizer,
        # D97: the report contract pins per-leaf fetch breadth to 3 (depth is the lever).
        research_fetch_breadth=PLAN_CHAIN_TREE_BREADTH,
        # d65 report-route grounding lanes ON (parity with the tree leaf).
        emit_article_notes=True, chunked_read=True,
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
    # d163 SHAPE BACKSTOP — cap the fetched sources fed to the write phase at the shape's
    # ``max_sources`` ceiling (richness comes from the writer USING each source's figures, not
    # from more sources — a thinner, fully-mined 6-8 beats 11 half-used). Keep the FIRST N in
    # launch order: the decompose-first seed fetches the breadth-covering facets first, so the
    # retained head still spans the facets. 0 / unset => uncapped (byte-identical). [S#]
    # numbering is assigned downstream from THIS capped list, so citations stay consistent.
    _src_ceiling = int(getattr(research_dag, "max_sources", 0) or 0)
    if _src_ceiling > 0 and len(sources) > _src_ceiling:
        print(f"[source-ceiling] capping {len(sources)} fetched sources to "
              f"{_src_ceiling} (d163 shape backstop)", flush=True)
        sources = sources[:_src_ceiling]
    # s14/P3A Stage A — carry the run's accumulated ArticleNotes alongside (findings, sources)
    # so the shared PHASE-2 write phase can build the running NARRATIVE summary
    # (covered/gaps/direction) it feeds the write PLANNER in place of the raw 12k blob. Stashed
    # on grow_trace (a dict already returned) to avoid changing the tuple arity — empty list
    # when the note lane emitted nothing (the write phase then degrades to legacy findings).
    grow_trace: dict[str, Any] = {
        "growable": bool(grower is not None),
        "article_notes": _collect_article_notes(result),
        # MEMORY-BY-HANDLE (d221): the stable handle of the research memory this phase
        # wrote into, so the write plan can BIND its section nodes to it (read-via-tools).
        "memory_handle": (grower.state.memory_handle if grower is not None else ""),
    }
    if grower is not None:
        grow_trace.update({
            "layers": list(grower.layers),
            "stop_reason": grower.stop_reason,
            "grow_layers": int(getattr(runtime, "_grow_layers", 0)),
            "max_layers": int(grower.max_layers),
        })
        # d185 NOTES-ARCH Layer 2 — the per-research BRIEF: a thesis-level digest
        # DERIVED from a15's concern graph over the SAME persisted state the loop
        # walked (no new LLM call, no fabrication). Stashed on grow_trace (a dict
        # already returned) so the arity is unchanged; run_plan_chain bubbles it up
        # on AgenticResult and the route persists it at the chat-session level keyed
        # by (chat_id, research_id) so multiple researches in one chat coexist.
        try:
            grow_trace["research_brief"] = grower.state.research_brief(
                topic=overall_goal or query, tree=grower.tree
            )
        except Exception as exc:  # a derived digest must never fail the research run
            print(f"[research-brief] skipped (projection error: {exc})", flush=True)
    return findings, sources, grow_trace


# === D3/D4 (d214/d215) — the ITERATIVE PLANNER LOOP + TERMINAL SYNTHESIZER. ============ #
# The served report route is ONE generic mechanism (AGENT_ARCHITECTURE §2), not a fixed
# pipeline: a PLANNER authors a plan, it EXECUTES, its LAST-STEP REVIEWER emits a FINAL
# STATUS, and the planner reads that status to decide the follow-up plan — looping until no
# further plan is needed, then EXITING into the terminal synthesizer. ==================== #
EVENT_RUN_SYNTHESIS = "agent_run_synthesis"  # D4 SSE: terminal synthesizer summary + artifact
# d221 — the synthesizer STREAMS its summary to the shell: incremental delta frames precede the
# terminal summary+artifact event so a streaming-aware shell renders the summary progressively
# instead of in one batch. The terminal EVENT_RUN_SYNTHESIS still carries the FULL summary +
# artifact (back-compat: the held UI, d98, consumes only that and is unaffected by the deltas).
EVENT_RUN_SYNTHESIS_DELTA = "agent_run_synthesis_delta"


def _build_write_planning_event(
    findings: str, sources: Sequence[Mapping[str, Any]],
    *, memory_handle: str = "", data_complexity: str = "",
) -> dict[str, Any]:
    """Assemble the research reviewer's WRITE-PLANNING EVENT — the GROUNDING the write planner
    reads (d192-3 / d221 / d237).

    This is the DATA the next (write) planner needs, read as a ``role:'user'`` write-planning
    message (d189). It is NO LONGER the loop's follow-up DECISION (that is the planner's real
    :meth:`~agent_runtime.Planner.decide_followup` reasoning over the real reviewer status —
    the retired ``_research_plan_final_status`` faked that decision); this only HANDS the write
    planner the grounding once the planner has decided to write:
      1. ``findings_digest`` — WHAT the research found (a bounded digest, so the planner decides
         the sections from real content, not just a source count);
      2. ``memory_handle`` — the research memory the write nodes BIND to (read-via-tools);
      3. ``data_complexity`` — the research reviewer's read of the data shape over N points
         (d237), so the planner reasons one-pass-vs-sectioned. Sectioning stays EMERGENT.

    SB-6/d299: the engine WRITE-METHODOLOGY strings (``write_directive``/``output_desired``) are
    RETIRED from this event — the methodology lives in the ``section-html-writer`` spec body now;
    the planner reasons the section topology from the DATA above + the shape + the spec
    DESCRIPTION (d10/d246), not an engine-stamped directive."""
    n_src = len(sources or [])
    digest = (findings or "").strip()
    # SB-6/d299: the engine-stamped WRITE-METHODOLOGY strings (``output_desired`` = "sectioned
    # report" + ``write_directive`` = "write sectioned, OUTLINE FIRST then sections") are RETIRED
    # from this event — that methodology now lives in the ``section-html-writer`` SPEC BODY (the
    # writer reads it), and the PLANNER reasons sectioned-vs-single + the section count from the
    # DATA (``data_complexity``) + the shape + the spec DESCRIPTION (d246), never an engine string.
    # Only the GROUNDING DATA the planner reasons over remains.
    return {
        "kind": "write_plan",
        "sections_basis": "research_nodes",
        "sources": n_src,
        "memory_handle": memory_handle or "",
        "findings_digest": digest[:1200],
        "data_complexity": (data_complexity or "").strip(),
    }


def _compose_reviewer_summary(status) -> str:
    """Compose the research reviewer's OVERALL SUMMARY (the (summary, index) pair text) from its
    ONE model emission — SB-5 (d285/d289).

    PURE PLUMBING: the reviewer's model-emitted ``rationale`` + the model-emitted
    ``data_complexity`` rendered as the reviewer's summary TEXT (the model authors the
    assessment; the engine only concatenates). The planner's
    :meth:`~agent_runtime.Planner.decide_followup` REASONS over this SINGLE non-divergent signal
    (the data-complexity rides INSIDE it), so the structured ``data_complexity`` field is no
    longer a competing second signal for that decision (d289). The SAME single ``review_research``
    emission still feeds ``_build_write_planning_event`` byte-identically — one source, no second
    model call, no divergence. Empty when there is no reviewed status (offline / acyclic)."""
    if status is None:
        return ""
    rationale = str(getattr(status, "rationale", "") or "").strip()
    complexity = str(getattr(status, "data_complexity", "") or "").strip()
    parts: list[str] = []
    if rationale:
        parts.append(rationale)
    if complexity:
        parts.append(f"Data complexity: {complexity}")
    return " ".join(parts).strip()


def _write_reviewer_status(w_result) -> str:
    """READ the write plan's deliverable status — the REVIEWER's own words when it ran.

    AUTONOMY REBUILD P3: the write planner now authors a model-driven ``final_review``
    node (role='reviewer', unified loop, same-spec rubric, fixes via file_update) — when
    that node produced output, its model-authored status prose IS the deliverable status
    (parse-to-read: the engine extracts, never composes). It grounds the finalize summary
    in what the artifact ACTUALLY contains (killing the counted-not-read overclaiming).
    Fallback (no reviewer node / no output): the run-outcome heuristic —
    ``deliverable_complete`` when the write plan ran ok, ``deliverable_thin`` on failure."""
    if w_result is None:
        return "deliverable_thin"
    try:
        results = getattr(w_result, "results", None) or {}
        for node_id, res in results.items():
            role = str(getattr(res, "role", "") or "")
            if role == "reviewer" or "review" in str(node_id).lower():
                prose = str(getattr(res, "output", "") or "").strip()
                if prose:
                    return prose[:2000]
    except Exception:  # noqa: BLE001 — status read is best-effort, never breaks the run
        pass
    return "deliverable_complete" if getattr(w_result, "ok", True) else "deliverable_thin"


async def _run_terminal_synthesizer(
    *, plane, query: str, out_name: str, sources: Sequence[Mapping[str, Any]],
    write_dag, plans_authored: Sequence[str], span,
    planner: Optional[Planner] = None, overall_goal: Optional[str] = None,
    reviewer_status: str = "",
) -> str:
    """The TERMINAL SYNTHESIZER stage (D4, d215): runs ONCE after the planner loop EXITS.

    It is NOT an in-plan ``add_step`` node — a direct terminal STAGE call. It delivers a brief,
    human-facing summary of the finished run and STREAMS it on the chat plane: incremental
    :data:`EVENT_RUN_SYNTHESIS_DELTA` frames (so a streaming-aware shell renders the summary
    progressively, d221), then a terminal :data:`EVENT_RUN_SYNTHESIS` carrying the FULL summary
    (+ the artifact name/mime when a downloadable file was produced, so the held UI offers the
    download). It touches NO content/coherence (the writer + reviewer own those, d218).

    SUMMARY CONTENT (d240/d221, FORK3): the summary is the PLANNER's LLM-generated finalize
    digest (:meth:`~agent_runtime.Planner.finalize_summary`) — NOT a hardcoded string (the fixed
    string was the faked fabrication the av map flagged). When ``planner`` is None (the offline
    streaming-unit seam) it falls back to a minimal DERIVED factual one-liner so the SSE stream
    still works; the LIVE served route always passes the planner so the summary is real."""
    n_sections = sum(
        1 for n in (write_dag.nodes if write_dag is not None else [])
        if not str(n.id).endswith("_review") and str(n.id) != "final_review"
    )
    n_sources = len(sources or [])
    has_artifact = bool(out_name)
    if planner is not None:
        # FORK3 — the real LLM finalize digest (fail-safe to a derived one-liner inside).
        summary = await planner.finalize_summary(
            overall_goal or query,
            plans_authored=list(plans_authored),
            sources=n_sources, sections=n_sections,
            artifact=out_name if has_artifact else "",
            # P3 GROUNDED FINALIZE: the reviewer's model-authored account of what the
            # artifact ACTUALLY contains — the summary must claim ⊆ this, not counts.
            memory_index=(reviewer_status or "").strip(),
        )
    else:
        # Offline streaming seam (no planner) — a minimal DERIVED factual summary.
        summary = (
            f"Complete for: {query.strip()[:160]}. "
            f"Authored {n_sections} section(s) grounded in {n_sources} source(s) across "
            f"{len(plans_authored)} plan(s) ({' → '.join(plans_authored)})."
            + (f" Your report is ready to download: {out_name}." if has_artifact else "")
        )
    span.set_attribute("plan_chain.synthesizer_ran", True)
    span.set_attribute("plan_chain.synthesizer_in_plan_node", False)
    span.set_attribute("plan_chain.synthesizer_llm", planner is not None)
    span.set_attribute("plan_chain.synthesizer_summary_chars", len(summary))
    # STREAM the summary to the shell as ordered delta frames (d221). Chunk on word
    # boundaries so each frame is a readable increment; the shell concatenates by ``seq``.
    chunks = _chunk_summary_for_stream(summary)
    span.set_attribute("plan_chain.synthesizer_stream_frames", len(chunks))
    try:
        for seq, delta in enumerate(chunks):
            await plane.publish(
                EVENT_RUN_SYNTHESIS_DELTA,
                {"seq": seq, "delta": delta, "done": False},
                source="chat_app.agentic",
            )
        # Terminal frame: the FULL summary + (when a file was produced) the downloadable
        # artifact (back-compat for the held UI). A chat-answer turn carries no artifact.
        terminal: dict[str, Any] = {"summary": summary, "streamed": True}
        if has_artifact:
            terminal["artifact"] = {"name": out_name, "mime": mime_for_path(out_name)}
        await plane.publish(
            EVENT_RUN_SYNTHESIS, terminal, source="chat_app.agentic",
        )
    except Exception:  # pragma: no cover - the SSE announce is best-effort
        pass
    return summary


def _chunk_summary_for_stream(summary: str, *, max_chars: int = 48) -> list[str]:
    """Split ``summary`` into ordered, word-boundary delta chunks for streaming (d221).

    Pure/string-only (unit-testable). Each chunk is <= ``max_chars`` where a word boundary
    allows it; the concatenation of all chunks reproduces ``summary`` byte-for-byte so the
    shell can rebuild it exactly. A short summary yields a single chunk."""
    text = summary or ""
    if not text:
        return []
    chunks: list[str] = []
    cur = ""
    for word in text.split(" "):
        piece = word if not cur else cur + " " + word
        if len(piece) > max_chars and cur:
            chunks.append(cur + " ")
            cur = word
        else:
            cur = piece
    if cur:
        chunks.append(cur)
    return chunks


def _loop_reasoning_planner(transport, registry: SpecRegistry, hook: ToolHook) -> Planner:
    """The single :class:`Planner` the generic loop uses for its REASONING calls — the
    follow-up decision, the research-plan reviewer, and the terminal finalize summary.

    Built on the SAME body-free factory (curated registry lookup + tool catalog; d10) as the
    authoring/heal planner, so the loop's reasoning is scoped identically. ``think=True`` +
    ``temperature=0`` are the per-call defaults; each reasoning method overrides its own
    ``format``/``num_predict`` (the methods carry their own native options)."""
    factory = AbstractPlanFactory(
        curate_index(registry.index(), CURATED_SPECS), tool_catalog=hook.registry.catalog()
    )
    return Planner(transport, factory, call_opts={"think": True, "temperature": 0})


async def _author_and_drive_acyclic_plan(
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
    conversation_context: Optional[str],
    overall_goal: Optional[str],
    allow_web: bool,
    requested_specs: Optional[list[str]],
    unmet_specs: Optional[list[str]],
) -> tuple[Optional[AgenticResult], Optional[PlanDAG], Any]:
    """Author + drive ONE acyclic plan (the incremental node-by-node authoring path, d3).

    Returns ``(pause, dag, result)``. ``pause`` is a non-None :class:`AgenticResult` when the
    plan is HELD for a missing specialist (the user CHOICE surface) — the generic loop returns
    it immediately (no follow-up, no terminal synthesizer). Otherwise ``pause`` is None and
    ``(dag, result)`` is the driven plan the loop reviews + reasons a follow-up over. Extracted
    verbatim from the retired standalone ``_run_acyclic`` so the missing-spec gate, the d11
    tool-arg grounding, the lifecycle gate + self-heal are IDENTICAL — only the call site moved
    into the one engine (FORK4 / d239: no separate acyclic fast path)."""
    runtime, _planner = _build_acyclic_runtime(
        transport=transport, registry=registry, hook=hook, plane=plane,
        shape_spec=shape_spec, conversation_context=conversation_context,
        allow_web=allow_web,
    )
    authoring_planner = _build_incremental_planner(
        transport=transport, registry=registry, hook=hook, shape_spec=shape_spec,
        allow_web=allow_web, requested_specs=requested_specs,
    )
    # RP-4c (d338/d339): BOUNDED RE-PROMPT on a malformed-empty authoring turn.
    # IncrementalPlanner.plan() raises MalformedOutputError when the tool-driven authorer
    # produced no usable nodes — a TRANSIENT empty-authoring turn on E4B (RP-4b's ~1/6
    # final-landed schedule-only miss). Its docstring assumes an "outer self-heal can re-plan
    # exactly as it does for a malformed one-shot plan", but this acyclic call site never wired
    # one, so an empty turn surfaced as a user-visible failure. Wrap it in the canonical
    # SelfHeal so a malformed-empty (or a still-invalid assembled-DAG) turn RE-LAUNCHES the
    # authoring: the MODEL re-authors from scratch each attempt. This is d310-legal — the engine
    # injects/alters/fabricates NOTHING; the only touch is re-prompt-on-malformed (the ONE
    # permitted output operation). Bounded (max_heals=2 → up to 3 authoring turns).
    _author_heal_log = HealLog(label="acyclic_authoring")
    plan_result = await SelfHeal(max_heals=2).run(
        lambda: authoring_planner.plan(query),
        label="acyclic_authoring",
        log=_author_heal_log,
    )
    try:  # OBSERVABILITY (doctrine: hardening is MEASURED) — record the re-author on the span.
        from opentelemetry import trace as _otel_trace
        _sp = _otel_trace.get_current_span()
        _sp.set_attribute("acyclic.authoring_heal_attempts", len(_author_heal_log.attempts))
        _sp.set_attribute("acyclic.authoring_healed", bool(_author_heal_log.healed))
    except Exception:  # pragma: no cover - observability must never break authoring
        pass
    dag = plan_result.dag
    dag.goal = overall_goal or ""

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
        await plane.publish(
            EVENT_MISSING_SPECIALIST, dict(payload), source="chat_app.agentic"
        )
        pause = AgenticResult(
            dag=dag,
            shape=selection.shape,
            escalated=selection.escalate,
            rationale=(selection.rationale or dag.rationale or ""),
            ok=False,
            missing_specialist=True,
            pending=payload,
        )
        return pause, dag, None

    result = await runtime.run(dag, timeout=timeout, run_id=run_id)
    return None, dag, result


async def _run_generic_loop(
    query: str,
    selection: ShapeSelection,
    *,
    first_plan_kind: str,
    transport,
    registry: SpecRegistry,
    hook: ToolHook,
    plane: EventPlane,
    timeout: float,
    run_id: Optional[str],
    shape_spec=None,
    conversation_context: Optional[str] = None,
    overall_goal: Optional[str] = None,
    allow_web: bool = True,
    requested_specs: Optional[list[str]] = None,
    unmet_specs: Optional[list[str]] = None,
    research_depth: Optional[int] = None,
    completeness_stop: Optional[str] = None,
    catalog: Optional[Mapping[str, "ShapeSpec"]] = None,
    session_id: Optional[str] = None,
    max_iterations: int = 6,
) -> AgenticResult:
    """The ONE generic ITERATIVE PLANNER LOOP for EVERY shape (d214/d215/d239, FORK4 / S1).

    There is no longer a bespoke web fork: ALL shapes route through this single loop, seeded by
    ``first_plan_kind`` (``"research"`` for the web deep-research family, ``"acyclic"`` for
    every other shape — linear / modular / escalate / no-web / missing-spec). One generic
    mechanism (AGENT_ARCHITECTURE §2):

      1. the planner AUTHORS a plan (research = the growable unroll engine; write = the section
         writer; acyclic = the incremental node-by-node authorer);
      2. the plan EXECUTES;
      3. its LAST-STEP REVIEWER emits a real STATUS — a REAL ``review_research`` LLM call for the
         research plan, the ``final_review`` node's emitted status for the write plan, the run
         outcome for an acyclic plan (the retired ``_research/_write_plan_final_status`` faked
         these);
      4. the PLANNER REASONS the follow-up via :meth:`~agent_runtime.Planner.decide_followup`
         (research → write, write → done, a single acyclic plan → done) — replacing the retired
         hardcoded research->write->done while-loop;
      5. repeat until ``done``, then EXIT into the TERMINAL SYNTHESIZER (d215), which runs ONCE
         and STREAMS the planner's LLM finalize summary (+ artifact when a file was produced).

    The reviewer/decision/synthesizer LLM calls are all FAIL-SAFE to safe baselines (the offline
    FakeTransport seam) so the suite + offline paths stay green while the live thinking model
    gets the real reasoning. A missing-specialist acyclic plan PAUSES for the user CHOICE (no
    follow-up, no synthesizer). The loop is bounded by ``max_iterations`` (a runaway backstop)."""
    requested_specs = requested_specs or []
    tracer = get_tracer("chat_app.agentic")
    reasoning_planner = _loop_reasoning_planner(transport, registry, hook)
    with tracer.start_as_current_span("agent.plan_chain") as span:
        span.set_attribute("plan_chain.first_plan_kind", first_plan_kind)
        plans_authored: list[str] = []
        # shared accumulators across plans
        findings: str = ""
        sources: list[dict[str, str]] = []
        grow_trace: dict[str, Any] = {}
        research_status = None                 # ResearchReviewStatus (the real reviewer)
        reviewer_summary = ""                  # SB-5: the reviewer's (summary, index) pair text
        write_planning_event: Optional[dict[str, Any]] = None
        memory_handle = ""
        out_name = ""
        write_dag = None
        w_result = None
        acyclic_dag = None
        acyclic_result = None
        produced = "acyclic" if first_plan_kind == "acyclic" else "write"
        engine = "generic-unroll"
        tree_config = replace(TreeConfig.from_env(), leaf_breadth=PLAN_CHAIN_TREE_BREADTH)
        if research_depth is not None:
            tree_config = replace(
                tree_config, depth=max(1, min(int(research_depth), N4_TREE_DEPTH_CEILING))
            )
        rounds_executed = 0

        # RP-6b (d359/d361) — the deep-research SHAPE whose DECLARED PHASE ORDER drives the loop's
        # default phase transitions (``research → write → done``). Resolved once here so each
        # phase's safe-baseline ``default_next`` is READ from the shape (``next_phase_plan``)
        # rather than baked as the literal ``"write_plan"`` / ``"done"``.
        dr_shape = _deep_research_shape(catalog, selection.shape)

        next_kind = first_plan_kind
        iters = 0
        # ACCUMULATED RESEARCH (live US-Iran verification catch): each research plan used
        # to OVERWRITE (findings, sources, notes) with its OWN fresh yield, so the
        # reviewer + follow-up decision judged only the LATEST plan — a re-research whose
        # fresh yield dwindled to 0 read as "research_thin" forever (the research loop),
        # and a write after a thin phase grounded in almost nothing. The loop now
        # ACCUMULATES across research plans so the reviewer, the follow-up decision and
        # the write phase all ground in the session's WHOLE gathered research. This is
        # DATA wiring (d192 context-flow); the decisions stay the model's reasoning.
        cum_findings = ""
        cum_sources: list[Any] = []
        cum_source_keys: set[str] = set()
        cum_notes: list[Any] = []
        fresh_source_count = 0
        grow_trace: dict[str, Any] = {}
        deliverable_doc = ""
        out_name = ""
        write_reviewer_prose = ""  # P3: the final_review node's model-authored status
        while next_kind != "done" and iters < max_iterations:
            iters += 1
            if next_kind in ("research", "research_plan"):
                # === RESEARCH PLAN === the generic growable engine gathers into persisted memory.
                findings, sources, grow_trace = await _run_generic_research_phase(
                    query,
                    transport=transport, registry=registry, hook=hook, plane=plane,
                    timeout=timeout, run_id=run_id, overall_goal=overall_goal,
                    requested_specs=requested_specs, dr_shape=dr_shape,
                    research_depth=research_depth, completeness_stop=completeness_stop,
                    session_id=session_id,
                )
                plans_authored.append("research")
                # Fold this plan's fresh yield into the run's ACCUMULATED research (dedup
                # sources by url) and rebind (findings, sources) to the accumulated view —
                # every downstream consumer (reviewer / decide / write grounding) sees the
                # whole session's research, while fresh_source_count keeps the NEW yield
                # visible to the follow-up decision (the diminishing-returns signal).
                fresh_source_count = len(sources)
                for _s in sources:
                    _k = str((_s.get("url") if isinstance(_s, Mapping) else "") or id(_s))
                    if _k not in cum_source_keys:
                        cum_source_keys.add(_k)
                        cum_sources.append(_s)
                cum_findings = (cum_findings + "\n\n" + (findings or "")).strip()[-16000:]
                cum_notes.extend(grow_trace.get("article_notes") or [])
                findings, sources = cum_findings, list(cum_sources)
                rounds_executed = sum(
                    int(l.get("gathered", 0) or 0) for l in grow_trace.get("layers", [])
                )
                memory_handle = str(grow_trace.get("memory_handle") or session_id or run_id or "")
                span.set_attribute("plan_chain.engine", engine)
                span.set_attribute("plan_chain.research_nodes", rounds_executed)
                span.set_attribute("plan_chain.findings_chars", len(findings))
                span.set_attribute("plan_chain.sources", len(sources))
                span.set_attribute("plan_chain.leaf_breadth", tree_config.leaf_breadth)
                span.set_attribute("plan_chain.tree_depth_configured", tree_config.depth)
                if grow_trace.get("growable"):
                    stop_reason = str(grow_trace.get("stop_reason") or "")
                    span.set_attribute("plan_chain.grow_stop_reason", stop_reason)
                    span.set_attribute(
                        "plan_chain.grow_layers", int(grow_trace.get("grow_layers", 0) or 0)
                    )
                    # S5 (d240): the MODEL's stop_research is the PRIMARY stop — the grow loop
                    # breaks on the model's no-expansion (stop_reason 'agent_sufficient'/
                    # 'no_expansion') BEFORE the max-layers ceiling. The depth ceiling
                    # (config.depth = the shape file's max_iter = the N4 high ceiling, 10 on the
                    # served route) + the wall-clock budget (timeout*0.9, set in the research
                    # phase) are a NON-DECIDING safety net guarding runaway growth ONLY; a
                    # 'depth_bound'/'budget' stop_reason is the safety net firing (the exception),
                    # not normal operation. completeness_stop stays reasoned doctrine.
                    span.set_attribute(
                        "plan_chain.stop_primary_is_model",
                        stop_reason not in ("depth_bound", "budget"),
                    )
                    span.set_attribute("plan_chain.depth_ceiling_role", "safety_net")
                    span.set_attribute(
                        "plan_chain.depth_ceiling", int(grow_trace.get("max_layers", 0) or 0)
                    )
                # REAL last-step reviewer (replaces the faked _research_plan_final_status): a
                # genuine LLM reviewer REASONS the status + the data complexity (d214/d237).
                research_status = await reasoning_planner.review_research(
                    overall_goal or query, findings, sources=len(sources)
                )
                span.set_attribute("plan_chain.research_reviewer_status", research_status.status)
                span.set_attribute("plan_chain.research_memory_handle", memory_handle)
                span.set_attribute(
                    "plan_chain.research_data_complexity", research_status.data_complexity[:200]
                )
                # the write-planning GROUNDING the planner hands the write plan (data only; the
                # DECISION below is decide_followup's real reasoning, not this).
                write_planning_event = _build_write_planning_event(
                    findings, sources, memory_handle=memory_handle,
                    data_complexity=research_status.data_complexity,
                )
                last_plan_kind = "research"
                reviewer_status_str = research_status.status
                findings_digest = findings[:1200]
                # SB-5 (d285/d289): the planner's decide REASONS over the reviewer's OVERALL
                # SUMMARY (the (summary, memory_index) pair), composed from review_research's ONE
                # emission. The bare structured data_complexity is no longer passed SEPARATELY to
                # decide (it rides INSIDE the summary) — write_planning_event above still reads the
                # SAME single emission, so there is ONE non-divergent data-complexity source.
                reviewer_summary = _compose_reviewer_summary(research_status)
                # RP-6b (d359/d361) — the safe-baseline next plan is the SHAPE's DECLARED next
                # phase after ``research`` (``research → write``), not a hardcoded ``"write_plan"``.
                default_next = dr_shape.next_phase_plan("research")
            elif next_kind in ("write_plan", "write"):
                # === WRITE PLAN === sections EMERGE from the research via the (write) planner.
                # s17 (flex output — no engine format stamp): the MODEL names the deliverable
                # (extension included) via one structured call; an explicit user-named file
                # still wins, and the neutral derived stem is only the fail-safe fallback.
                out_name = explicit_filename(query)
                if not out_name:
                    out_name = await reasoning_planner.name_deliverable(
                        overall_goal or query, requested_specs=requested_specs or None
                    )
                if not out_name:
                    out_name = derive_output_path(
                        overall_goal or query, "", requested_specs or None
                    )
                span.set_attribute("plan_chain.out_file", out_name)
                # STALENESS SNAPSHOT (autonomy rebuild P2 — the Gate-2e lesson): the
                # workspace sandbox persists across runs and chats, so a target file
                # left by a PREVIOUS session can sit at ``out_name``. Snapshot its
                # bytes BEFORE the write plan; if the plan finishes with the content
                # unchanged, no deliverable was produced THIS run — report that
                # honestly instead of shipping last session's file as fresh work.
                pre_plan_doc = ""
                try:
                    _rb0 = await hook.invoke(
                        "file_read", path=out_name, max_bytes=4_000_000
                    )
                    if getattr(_rb0, "ok", False) and isinstance(_rb0.value, Mapping):
                        pre_plan_doc = str(_rb0.value.get("text") or "")
                except Exception:  # noqa: BLE001 — snapshot is best-effort
                    pre_plan_doc = ""
                write_dag, w_result = await run_section_write_phase(
                    query, out_name, findings, sources,
                    transport=transport, registry=registry, hook=hook, plane=plane,
                    # WRITE-PHASE BUDGET (live run-8 catch): a multi-part write (4 nodes
                    # × several ~60s turns on the 6GB card) does not fit the research
                    # phase's cap — the FINAL part (sources + close) was cancelled at
                    # the wire. The write phase gets a proportionate 1.5× slice; the
                    # research phases keep the base budget.
                    timeout=float(timeout) * 1.5, run_id=run_id, overall_goal=overall_goal,
                    requested_specs=requested_specs, outline_hint=None,
                    research_notes=(cum_notes or grow_trace.get("article_notes")),
                    write_planning_event=write_planning_event,
                    research_memory_handle=memory_handle,
                )
                plans_authored.append("write")
                produced = "write"
                span.set_attribute("plan_chain.write_nodes", len(write_dag.nodes))
                # PULL-WRITER (autonomy rebuild P2): the deliverable now lives ONLY on
                # disk, authored by the model driving file_write itself — read the real
                # bytes back for artifact persistence (engine reads, never edits). An
                # absent/empty file is surfaced honestly downstream, never synthesized.
                deliverable_doc = ""
                try:
                    _rb = await hook.invoke(
                        "file_read", path=out_name, max_bytes=4_000_000
                    )
                    if getattr(_rb, "ok", False) and isinstance(_rb.value, Mapping):
                        deliverable_doc = str(_rb.value.get("text") or "")
                except Exception:  # noqa: BLE001 — persistence read is best-effort
                    deliverable_doc = ""
                # STALENESS GUARD (pairs with the pre-plan snapshot above): identical
                # bytes before and after the write plan = the plan never touched the
                # file → there is NO deliverable from this run. Dropping it here keeps
                # the downstream honest (no artifact card, no final_response document)
                # instead of dressing a previous session's file up as fresh output.
                if deliverable_doc and deliverable_doc == pre_plan_doc:
                    span.set_attribute("plan_chain.deliverable_stale", True)
                    deliverable_doc = ""
                span.set_attribute(
                    "plan_chain.deliverable_bytes", len(deliverable_doc)
                )
                # READ the write plan's final_review node status (replaces the faked
                # _write_plan_final_status hardcoded deliverable_complete).
                reviewer_status_str = _write_reviewer_status(w_result)
                write_reviewer_prose = reviewer_status_str  # P3: grounds the finalize summary
                span.set_attribute("plan_chain.write_reviewer_status", reviewer_status_str)
                last_plan_kind = "write"
                findings_digest = ""
                # after a WRITE plan the post-write decide reasons over the write reviewer STATUS
                # (deliverable_complete/thin); no research summary applies → empty (fall-through).
                reviewer_summary = ""
                # RP-6b (d359/d361) — the safe-baseline after the LAST declared phase (``write``)
                # is the shape's terminal ``done`` (``next_phase_plan`` returns it for the tail).
                default_next = dr_shape.next_phase_plan("write")
            elif next_kind == "acyclic":
                # === ACYCLIC PLAN (the folded _run_acyclic single-plan case) ===
                pause, acyclic_dag, acyclic_result = await _author_and_drive_acyclic_plan(
                    query, shape_spec, selection,
                    transport=transport, registry=registry, hook=hook, plane=plane,
                    timeout=timeout, run_id=run_id,
                    conversation_context=conversation_context, overall_goal=overall_goal,
                    allow_web=allow_web, requested_specs=requested_specs,
                    unmet_specs=unmet_specs,
                )
                if pause is not None:
                    # Missing specialist → HELD for the user CHOICE (no follow-up, no synthesizer).
                    return pause
                plans_authored.append("acyclic")
                produced = "acyclic"
                reviewer_status_str = (
                    "answer_complete" if getattr(acyclic_result, "ok", True) else "answer_thin"
                )
                span.set_attribute("plan_chain.acyclic_reviewer_status", reviewer_status_str)
                last_plan_kind = "acyclic"
                findings_digest = ""
                reviewer_summary = ""
                default_next = "done"
            else:
                # review_plan or an unknown kind — no dedicated builder in as1; EXIT safely (the
                # write plan already carries its own final_review). Bounded, never spins.
                span.set_attribute("plan_chain.unbuilt_followup", str(next_kind))
                break

            # ---- the PLANNER REASONS the follow-up (real LLM; fail-safe to default_next) ----
            decision = await reasoning_planner.decide_followup(
                overall_goal or query,
                last_plan_kind=last_plan_kind,
                reviewer_status=reviewer_status_str,
                # SB-5 (d285/d289) — the SINGLE signal: the reviewer's (summary, memory_index)
                # pair. The data-complexity rides INSIDE reviewer_summary; the bare structured
                # field is no longer passed separately (it would be a competing second signal).
                reviewer_summary=reviewer_summary,
                memory_index=memory_handle,
                findings_digest=findings_digest,
                sources=len(sources),
                fresh_sources=(
                    fresh_source_count if last_plan_kind == "research" else None
                ),
                plans_so_far=plans_authored,
                default_next=default_next,
            )
            span.set_attribute(f"plan_chain.followup_after_{last_plan_kind}", decision.next_plan)
            next_kind = decision.next_plan

        span.set_attribute("plan_chain.planner_loop_iterations", len(plans_authored))
        span.set_attribute("plan_chain.plans_authored", ",".join(plans_authored))

        # ---- assemble the AgenticResult from the produced deliverable ----
        if produced == "acyclic":
            agentic = _agentic_from_runtime(
                acyclic_dag, acyclic_result,
                shape=selection.shape,
                escalated=selection.escalate,
                rationale=(selection.rationale or (acyclic_dag.rationale if acyclic_dag else "")),
            )
        elif w_result is None or write_dag is None:
            # RUNAWAY-BACKSTOP EXIT (live US-Iran catch): the loop hit max_iterations
            # WITHOUT ever authoring a write plan (every iteration re-researched), so
            # there is no write DAG/result to normalize — surface an HONEST failed
            # result carrying the research trail instead of crashing on the absent
            # write result ("'NoneType' object has no attribute 'results'").
            span.set_attribute("plan_chain.exhausted_without_write", True)
            agentic = AgenticResult(
                dag=None, result=None,
                shape=(selection.shape or "plan-chain"),
                escalated=selection.escalate,
                rationale=(selection.rationale
                           or "plan-chain: loop ceiling reached with no write plan"),
                final_response=(
                    "The research phase ran to the loop ceiling without producing the "
                    "written deliverable. The gathered research is preserved in this "
                    "session's memory — ask again to write the report from it."
                ),
                ok=False,
            )
            agentic.research_brief = grow_trace.get("research_brief")
        else:
            agentic = _agentic_from_runtime(
                write_dag, w_result,
                shape=(selection.shape or "plan-chain"),
                escalated=selection.escalate,
                rationale=(selection.rationale or (write_dag.rationale if write_dag else "")
                           or "plan-chain: research → write (iterative planner loop)"),
            )
            # PULL-WRITER (autonomy rebuild P2): the deliverable is the FILE the model
            # wrote (read back above), not a node's chat emission — surface it as the
            # response/artifact from the real bytes. (Phase 3 swaps final_response to
            # the synthesizer summary; the artifact stays the file.)
            if deliverable_doc and out_name:
                agentic.final_response = deliverable_doc
                agentic.md_report = deliverable_doc
                agentic.html_report = (
                    deliverable_doc if out_name.lower().endswith((".html", ".htm"))
                    else agentic.html_report
                )
                agentic.artifacts = [
                    (out_name, mime_for_path(out_name), deliverable_doc)
                ]
            # SERVED-route loop trace (s13/B1) — the generic growable engine control snapshot.
            agentic.deep_research = {
                "shape": (selection.shape or "plan-chain"),
                "engine": engine,
                "rounds_executed": rounds_executed,
                "leaf_breadth": tree_config.leaf_breadth,
                "write_dag": write_dag.as_dict() if write_dag is not None else {},
                "sources": len(sources),
                "sectioned": True,
                "plan_chain": True,
                "planner_loop_iterations": len(plans_authored),
                "plans_authored": list(plans_authored),
            }
            agentic.research_brief = grow_trace.get("research_brief")
            if grow_trace.get("growable"):
                agentic.deep_research.update({
                    "growable": True,
                    "layers": grow_trace.get("layers", []),
                    "stop_reason": grow_trace.get("stop_reason"),
                    "grow_layers": grow_trace.get("grow_layers", 0),
                    "depth_reached": grow_trace.get("grow_layers", 0),
                    "depth_configured": tree_config.depth,
                    "max_layers": grow_trace.get("max_layers", 0),
                })

        # ---- TERMINAL SYNTHESIZER (d215): runs ONCE after the loop EXITS. NOT an add_step node.
        # It STREAMS the planner's LLM finalize summary (+ the artifact when a file was produced).
        # P3 GROUNDED FINALIZE: the write reviewer's model-authored status prose (what the
        # artifact ACTUALLY contains) rides the finalize call's memory_index input, so the
        # summary claims what was reviewed-in, never bare counts (kills the overclaiming).
        agentic.synthesis_summary = await _run_terminal_synthesizer(
            plane=plane, query=query, out_name=out_name, sources=sources,
            write_dag=write_dag, plans_authored=plans_authored, span=span,
            planner=reasoning_planner, overall_goal=overall_goal,
            reviewer_status=write_reviewer_prose,
        )
        # P3 CHAT TURN = SUMMARY + DOWNLOAD CARD (owner decision): when a file deliverable
        # exists, the persisted chat response is the model-authored finalize SUMMARY; the
        # document itself stays artifact-only (the artifact list drives the download card).
        # A run with no fresh deliverable keeps its honest node-output response unchanged.
        if deliverable_doc and out_name and agentic.synthesis_summary:
            agentic.final_response = agentic.synthesis_summary
        return agentic


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
    session_id: Optional[str] = None,
) -> AgenticResult:
    """The served REPORT route = the generic loop SEEDED with a research-first plan (d214/d239).

    This is no longer a bespoke engine — it is a thin SEED into :func:`_run_generic_loop`
    (``first_plan_kind="research"``): research → (planner reasons) → write → (planner reasons) →
    done → terminal synthesizer. Kept as a named entry so the deep-research family + the direct
    report-route tests have a stable seed; the WORK is the one generic loop."""
    return await _run_generic_loop(
        query, selection,
        first_plan_kind="research",
        transport=transport, registry=registry, hook=hook, plane=plane,
        timeout=timeout, run_id=run_id,
        conversation_context=conversation_context, overall_goal=overall_goal,
        allow_web=allow_web, requested_specs=requested_specs,
        research_depth=research_depth, completeness_stop=completeness_stop,
        catalog=catalog, session_id=session_id,
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
    factory = AbstractPlanFactory(
        curate_index(registry.index(), CURATED_SPECS),  # d230 planner-facing spec scoping
        tool_catalog=hook.registry.catalog(),
    )
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
            memory_index=getattr(n, "memory_index", ""),  # d285 SB-3: carry the brief choice
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
        max_sources=getattr(dag, "max_sources", 0),
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
    session_id: Optional[str] = None,
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
        session_id=session_id,
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
        # s14/P3A Stage A — research ArticleNotes drive the write planner's NARRATIVE summary
        # (covered/gaps/direction) in place of the raw 12k findings blob.
        research_notes=grow_trace.get("article_notes"),
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
    """Drive the cyclic deep-research shape → the SHARED per-section bounded write phase.

    s15 thread-2 (d178) — SECTIONED-WRITE UNIFICATION. The deep-research path now writes
    EXCLUSIVELY per-section: the monolithic single-shot ``synthesis.react_file`` whole-doc
    fold — the topic-independent corruption root that truncated a ~70KB document mid-table
    when the late sections fell outside E4B's ~512-token sliding window — is RETIRED. There
    is no longer an inline whole-doc synthesis branch (the ``sectioned`` parameter is gone);
    EVERY report-producing request writes per-section.

    This function UNROLLS the declarative shape into a bounded acyclic role-tagged DAG
    (honoring the UI-set ``max_iter``, d5) and hands it to
    :func:`_run_deep_research_sectioned`, which runs PHASE-1 research on the generic growable
    engine then authors the report ONE bounded ``file_write`` section at a time over the run's
    findings + scoped sources (each section fed ONLY its planner-assigned sources nearest the
    cursor, so figures/URLs stay inside the model's sliding window). The served route reaches
    the report path via :func:`run_plan_chain`; this sibling shares the IDENTICAL PHASE-1
    generic engine + PHASE-2 :func:`run_section_write_phase`, so the two report paths stay
    unified on the single per-section write substrate."""
    # The research-phase specialization, ROUTED BY THE SHAPE (RP-6b d359/d361). The
    # deep-research shape declares the research phase's ``spec_role`` (``research``), which the
    # engine maps onto the seeded research-analysis spec. A user-named WRITER/output spec is NOT
    # pulled onto the research seed (Bug A d355/d356) — it routes to the WRITE phase, where
    # ``run_section_write_phase`` threads ``requested_specs`` into the write planner and the
    # named output spec is reachable on the deliverable node. Falls back to no spec (role
    # framing only) if the research-analysis default is somehow unregistered.
    spec_name = _deep_research_spec(registry, shape=shape_spec)
    effective_max_iter = shape_spec.effective_max_iter(max_iter_override)

    # ENGINE-OWNED GROWABLE SEED (s16/a3 d239/d247 — unroll_shape RETIRED). Kept on the
    # sectioned signature for the caller's uniform deep-research construction: the growable
    # engine AUTHORS the real research topology by reasoning inside the sectioned phase; this
    # DAG is the construction handle (a tool-less self-selecting research seed, growable-tagged).
    dag = _research_seed_dag(shape_spec, query, spec=spec_name)
    # d39 OVERALL GOAL: stamp the verbatim user request onto the unrolled DAG so the
    # runtime feeds it into every research node's user turn (uniform with the acyclic path;
    # the role nodes otherwise see only their unroll task prefix).
    dag.goal = overall_goal or ""

    # PER-SECTION BOUNDED WRITE (s9/c13 → d178): research ROUNDS only, then hand the findings
    # + fetched sources to the SHARED per-section bounded write phase. This is the SOLE write
    # path for deep-research now — the inline single-synthesis whole-doc fold is retired.
    return await _run_deep_research_sectioned(
        query, shape_spec, selection, dag,
        transport=transport, registry=registry, hook=hook, plane=plane,
        timeout=timeout, run_id=run_id, effective_max_iter=effective_max_iter,
        requested_specs=requested_specs, overall_goal=overall_goal,
    )


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
