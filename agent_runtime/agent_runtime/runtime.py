"""The in-process agent runtime — hardened DAG executor + agent lifecycle (d2).

This is the d2 concurrency model made concrete and PRODUCTION-hardened (Stage B):
each DAG node is launched as a tracked ``asyncio`` task on the SAME event loop
(NO shell forking, NO broker/pool, NO subprocess). On top of the Stage-A core
this adds:

- an explicit per-node STATE MACHINE (:class:`~agent_runtime.status.NodeState` /
  :class:`~agent_runtime.status.NodeStatus`) so completion, skipping, and
  cancellation are unambiguous and observable;
- BOUNDED CONCURRENCY (an optional semaphore) so a fan-out of ready nodes cannot
  swamp the single shared GPU/phi;
- full AGENT LIFECYCLE — start, track, await, and **cancel**: a timeout (or an
  explicit :meth:`AgentRuntime.cancel_all`) cancels every in-flight tracked task
  and awaits its teardown, so no node task is ever orphaned;
- an IDEMPOTENT result cache: a node whose result is already known is served from
  cache and NEVER re-executed (the no-double-execution guarantee that makes the
  sub-graph re-plan safe);
- SUB-GRAPH SELF-HEAL: when a node exhausts its node-level self-heal, the runtime
  asks an injected ``replanner`` to re-derive a MINIMAL corrective sub-graph for
  just that node and runs it in-process (sharing the cache), recovering the node
  without redoing the rest of the DAG. Bounded by ``max_replans`` with a clean
  give-up-and-surface path.

Context-scoping is enforced BY CONSTRUCTION (d10): the runtime owns the
``SpecLoader`` and resolves each node's single spec into a
:class:`~agent_runtime.scope.ScopedSpec` (one ``{name, body}``) which is the ONLY
specialization view the :class:`SubAgent` is handed — the sub-agent holds no
loader, so it structurally cannot reach a second spec.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

from llm_framework import Chain, Context, Transport
from llm_framework.stages import call_stage, prompt_assembly, structured_output
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode
from reactive_tools import EventPlane, LambdaRegistry, ToolHook
from specialization.loader import SpecLoader

from .tracing import get_tracer, run_blocking_in_span

from .article_note import coerce_article_note
from .chunked_read import chunked_read as _chunked_read
from .claim_verify import (
    REVIEWER_TOOL_SPECS,
    research_answered_from_memory,
    verify_and_revise,
)
from .research_tree import first_native_call, make_tool_spec
from .collision import (
    Collision,
    CollisionResolver,
    CollisionUnresolved,
    apply_resolution,
    detect_collision,
    strip_directives,
)
from .factory import PlanDAG, PlanNode
from .identity import with_identity
from .heal_router import (
    EVENT_HEAL_ROUTED,
    EVENT_NODE_FAILURE_DETECTED,
    HealRouter,
    register_heal_rule,
)
from .reactor import PlannerReactor
from .roles import (
    READ_NOT_DESCRIBE,
    ROLE_SYNTHESIZER,
    role_framing,
)
from .synth_tools import (
    DONE_SENTINEL,
    _strip_fence as _strip_synth_fence,
    anchored_insert_args,
    begins_html_document,
    choose_section_anchor,
    collapse_duplicate_sections,
    collect_fetched_sources,
    derive_output_path,
    enforce_single_html_document,
    ensure_source_coverage,
    has_duplicate_html_structure,
    has_truncation_marker,
    html_close_gap,
    is_detailed_task,
    plant_section_anchor,
    reconcile_doc_structure,
    render_scoped_sources,
    resolve_writer_source_budget,
    READ_RANKING_CHUNK_CHARS,
    read_content_char_budget,
    section_reemission,
    select_relevant_chunks,
    strip_ungrounded_urls,
    render_source_index,
    sanitize_write_path,
    split_done_signal,
    strip_section_anchor,
    strip_wrapper_closers,
    strip_wrapper_openers,
    top_level_html_doc_count,
)
from .scheduler import ExecutionMode, next_dispatch
from .scope import ScopedSpec
from .selfheal import (
    HealLog,
    InvalidStepError,
    MalformedOutputError,
    SelfHeal,
    ToolFailureError,
)
from .status import NodeState, NodeStatus

# Lifecycle event kinds published on the in-process plane (observable by the
# planner / UI / a smoke subscriber — the reactive plane in action, d2).
EVENT_NODE_LAUNCHED = "agent_node_launched"
EVENT_NODE_DONE = "agent_node_done"
EVENT_NODE_FAILED = "agent_node_failed"
EVENT_NODE_HEALED = "agent_node_healed"
EVENT_NODE_CANCELLED = "agent_node_cancelled"
EVENT_NODE_REPLANNED = "agent_node_replanned"
EVENT_NODE_SKIPPED = "agent_node_skipped"
# Stage-B run-engine lifecycle events: the produce→gate→done crossing + the
# CODER=REVIEWER inline-fix path (all observable on the in-process EventPlane so
# the UI DAG view + the freeze-safe stream see every transition).
EVENT_NODE_VERIFIABLE = "agent_node_verifiable"        # produce finished; entering the verify gate
EVENT_NODE_REVIEW = "agent_node_review"                # gate rejected → same-spec inline review starting
EVENT_NODE_INLINE_FIXED = "agent_node_inline_fixed"    # inline review fix carried the node past the gate
EVENT_NODE_VERIFY_FAILED = "agent_node_verify_failed"  # gate still failing after inline-fix budget
# DAG SPEC-COLLISION lifecycle events (d11): a genuine conflict among a node's
# specs PAUSES the node in-flight at the HITL resolution gate, then resumes with
# the user-resolved composition (both observable on the in-process plane so the UI
# can show the node waiting for a decision and then proceeding).
EVENT_NODE_COLLISION = "agent_node_collision"                   # paused; awaiting HITL resolution
EVENT_NODE_COLLISION_RESOLVED = "agent_node_collision_resolved" # resumed with the resolved composition

# P2-5c — GROWABLE-loop per-layer progress (the iterative gap-expansion drive, _drive_growable).
# One event per grown wave so a long live/UI report run is OBSERVABLE: the layer index, how many
# research nodes the wave dispatched, the cumulative source/node count so far, the elapsed
# wall-clock, and the stop_reason when growth ends (agent_sufficient / no_expansion / depth_bound
# / budget). Advisory-only (best-effort emit; never gates the loop).
EVENT_GROW_LAYER = "agent_grow_layer"

# The CONTROL-PLANE node-lifecycle kinds an auto-created per-run observability
# lambda watches (s9/a2 — closes the s7 F1 HIGH gap / honors d15). Every one is a
# low-frequency control signal, so the lambda may reduce with ``each`` safely —
# NOT the data-plane ``tool_call``/``tool_result`` kinds, which would wake-storm
# and which the LambdaRegistry's anti-wake-storm guard rejects under ``each``.
RUN_LIFECYCLE_KINDS: tuple[str, ...] = (
    EVENT_NODE_LAUNCHED,
    EVENT_NODE_DONE,
    EVENT_NODE_FAILED,
    EVENT_NODE_HEALED,
    EVENT_NODE_CANCELLED,
    EVENT_NODE_REPLANNED,
    EVENT_NODE_SKIPPED,
    EVENT_NODE_VERIFIABLE,
    EVENT_NODE_REVIEW,
    EVENT_NODE_INLINE_FIXED,
    EVENT_NODE_VERIFY_FAILED,
    EVENT_NODE_COLLISION,
    EVENT_NODE_COLLISION_RESOLVED,
)

# Stepwise SYNTHESIS tuning (s9/c1, D1 fix): the terminal deliverable is built one
# section at a time via tool calls (no ``format=<schema>`` — see synth_tools). Each
# call need only emit ONE bounded section, so a per-call ``num_predict`` that clears
# the think=True CoT + a section is enough; the TOTAL document is unbounded across
# calls (that is the whole point — it removes the single-call output ceiling that
# truncated D1 mid-document). The write node runs deterministic (temp=0, d35).
SYNTH_MAX_SECTIONS = 16
SYNTH_NUM_PREDICT = 4096
# s9/N5: the largest deliverable (chars) the verify lane will re-persist from a SINGLE
# whole-document revise turn. A revise turn re-emits the corrected doc in one call,
# bounded by SYNTH_NUM_PREDICT (4096 tok, ~shared with think CoT) — beyond ~this many
# chars a one-turn rewrite would truncate, so the lane surfaces the unbacked verdict
# but does NOT auto-rewrite (never trade a fabrication-flag for a truncated file).
_VERIFY_REVISE_MAX_CHARS = 9000

# WRITER-SOURCE WINDOW SIZING (MSF/d89): the write phase does NOT pass num_ctx in its
# subagent_call_opts — it relies on the value BAKED into the model's Modelfile (E4B =
# 32768). So when a section node sizes its per-source budget against the window and the
# call opts carry no num_ctx, fall back to this default (env RA_WRITE_NUM_CTX). The
# window-fit cap keeps the RAISED writer feed (12k chars × several sources) under a
# fraction of the window so the section prompt + num_predict output never overflow it
# (the d22 overflow→empty-thinking failure the raise must not reintroduce).
try:
    _DEFAULT_WRITE_NUM_CTX = max(2048, int(os.environ.get("RA_WRITE_NUM_CTX", "32768")))
except (TypeError, ValueError):
    _DEFAULT_WRITE_NUM_CTX = 32768
# Approx chars/token to convert a num_ctx token window into a char budget for the
# byte-oriented per-source cap (matches the d22 ~3.5 chars/token measurement).
_CHARS_PER_TOKEN = 3.5

# READ-SIDE relevance embedder (d109): the research read path RANKS a fetched source's
# paragraph chunks by MiniLM 384-d similarity to the node's sub-question (NOT lexical),
# REUSING the memory store's ``CpuEmbedder``. The model loads ONCE per process (fastembed
# ONNX, CPU-pinned) — heavy, so it is built lazily and cached here and shared across every
# research sub-agent the runner builds. ``None`` means fastembed/memory is unavailable →
# the read falls back to the bounded map/reduce (never lexical).
_UNSET_EMBEDDER: Any = object()
_READ_EMBEDDER: Any = _UNSET_EMBEDDER


def _load_read_embedder() -> Any:
    """The shared MiniLM ``CpuEmbedder`` (memory store's), lazily built once; ``None`` if
    fastembed/memory cannot load (caller then uses the map/reduce fallback)."""
    global _READ_EMBEDDER
    if _READ_EMBEDDER is _UNSET_EMBEDDER:
        try:
            from memory.embedder import CpuEmbedder

            _READ_EMBEDDER = CpuEmbedder()
        except Exception:  # noqa: BLE001 - missing optional embedder must not crash a read
            _READ_EMBEDDER = None
    return _READ_EMBEDDER
# Fraction of the write window the TOTAL of a section's source excerpts may occupy,
# leaving the rest for the section prompt scaffolding + the num_predict output.
_WRITE_SOURCE_WINDOW_FRACTION = 0.6

# AGENTIC RESEARCH loop bounds (s9/c5, d49/d50 — retires flags #1/#3). A web_search
# node is no longer driven by a deterministic search-then-read EXECUTOR; it is a TRUE
# AGENT that DECIDES to search and which sources to read via lightweight tool calls
# (``web_search``/``web_fetch`` — small args the small model emits reliably, unlike
# content-laden JSON, d49). These are NON-FLOW cost/safety bounds (a cap on a loop the
# model drives), NOT flow gates: ``RESEARCH_MAX_TURNS`` caps total ReAct turns and
# ``RESEARCH_DEFAULT_FETCH_CAP`` is the fetch cap used when a caller wired none.
RESEARCH_MAX_TURNS = 12
RESEARCH_DEFAULT_FETCH_CAP = 5
# BREADTH (s9/N1, d60/c15 part-a): the total ReAct turn ceiling RISES PROPORTIONALLY
# with the fetch cap so a high-breadth gather can search several angles AND read MANY
# sources without the flat turn ceiling clipping it (a cap of ~10 needs >12 turns to
# search-then-read-then-write). ``RESEARCH_SEARCH_HEADROOM`` is the non-fetch turns a
# gather spends — search angles, nudges and the final findings turn. Still a NON-FLOW
# bound: the model stops the instant it has read enough; this only raises the ceiling
# so genuine breadth is REACHABLE (it never forces more work). For the legacy cap (≤5)
# the effective ceiling stays RESEARCH_MAX_TURNS (max() floor), so narrow paths are
# byte-identical to before.
RESEARCH_SEARCH_HEADROOM = 6

# The research-agent instruction (d38/d39/d50 prompt-quality mandate: crisp, anti-
# hallucination). Appended to the assembled USER turn so the worker knows it must
# gather REAL evidence via its tools before answering, and that FINDINGS are RAW prose
# (never JSON — content is RAW on every route, d50.1).
_RESEARCH_LOOP_INSTRUCTION = (
    "----\n"
    "You are a RESEARCH AGENT with two tools. Gather REAL evidence with them before "
    "you answer — do not rely on memory.\n\n"
    "To call a tool, reply with ONLY a JSON object and NOTHING else:\n"
    '  {{"tool": "web_search", "args": {{"query": "<search terms>"}}}}\n'
    '  {{"tool": "web_fetch", "args": {{"url": "<a result URL to read in full>"}}}}\n\n'
    "Workflow: search the topic, then READ the most relevant results by fetching their "
    "URLs (up to {fetch_cap} fetches), then write your findings. Fetch a source before "
    "relying on it; never invent facts or cite a page you have not read. Issue ONE tool "
    "call per turn.\n\n"
    "When you have read enough, STOP calling tools and write your FINDINGS as plain "
    "prose (NOT JSON): the key facts, figures and events, each attributed to the source "
    "URL you read it from. Be specific and substantive."
)
_RESEARCH_NUDGE = (
    "Reply with EITHER a single tool-call JSON object (to search or fetch), OR your "
    "FINDINGS as plain prose — nothing else."
)
_RESEARCH_FINALIZE = (
    "Stop searching. Write your FINDINGS now as plain prose, drawn from the sources you "
    "have already read — the key facts and figures, each attributed to its source URL. "
    "Output ONLY the findings (no JSON, no preamble)."
)
# s9/N5 (d62/c15 part-e): the no-fabrication GATHER-MORE nudge — sent when the
# ``verify_lane`` is ON and a research stage wrote substantive findings with ZERO real
# fetches (it answered FROM MEMORY; E4B does this on the bare ReAct path). The model
# MUST search + read a real source before its findings are accepted. Bounded by
# ``_RESEARCH_GATHER_MORE_MAX`` (NON-FLOW): a genuinely unfetchable topic cannot loop
# forever — after the budget the findings stand and the deliverable verify lane still
# re-checks each claim against the (still 0) fetched sources.
_RESEARCH_GATHER_MORE_MAX = 2
_RESEARCH_GATHER_MORE = (
    "You answered from MEMORY — you fetched NO real source. A research task may NOT be "
    "answered from memory: every fact must come from a source you actually read. SEARCH "
    "now and FETCH at least one relevant result URL (up to {fetch_cap} fetches), THEN "
    "write your findings from what you read. Reply with ONE tool-call JSON object "
    '({{"tool":"web_search",...}} or {{"tool":"web_fetch",...}}) — nothing else.'
)
# s9/N2 (d60/c15 part-b): the per-article CONTROL-NOTE clause, appended to the research
# instruction ONLY when ``emit_article_notes`` is enabled (default OFF → byte-identical
# to N1). It asks the agent to record a SHORT structured note after reading each source
# — a lightweight control record (like the c5 tool args), NOT the deliverable document
# (content stays RAW, d50.1). The note steers the NEXT search via ``gaps_or_followups``.
_RESEARCH_NOTE_CLAUSE = (
    "\n\nAfter you READ a source (web_fetch), record a SHORT note about it before moving "
    "on — reply with ONLY this JSON and nothing else:\n"
    '  {"tool": "note", "args": {"url": "<the source URL you just read>", '
    '"summary": "<2-3 sentence summary>", "category": "<topic>", '
    '"source_trust": "primary|secondary|reference-untrusted", '
    '"key_claims": ["<short fact>", "..."], "relevance": "<why it serves the goal>", '
    '"gaps_or_followups": ["<what to search next>", "..."]}}\n'
    "Keep it SHORT — this note STEERS your next search; it is NOT the final answer. Treat "
    "Wikipedia as reference-untrusted (citable only if attributed, never sole backing)."
)


def _first_json_object(text: str) -> Optional[str]:
    """Return the first balanced top-level ``{...}`` substring of ``text``, or None.

    A tiny, dependency-free, string-literal/escape-aware scan so a research turn's
    lightweight tool call can be recovered even if the model wraps it in a fence. A
    truncated/unbalanced object yields None (treated as not-a-tool-call → findings)."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _is_writer_node(node: PlanNode) -> bool:
    """A node whose job is to AUTHOR the file deliverable via the shared raw-content
    loop — the deep-research SYNTHESIS role, or an acyclic ``file_write`` tool node.

    Plan-chaining (c1b/d49.4) keys the multi-page accumulation off this: a write-file
    plan's per-page/section nodes are all writers, chained via ``depends_on``, so a
    writer whose UPSTREAM is itself a writer CONTINUES that file (appends the next
    page) and a writer with a DOWNSTREAM writer is NON-final (must not close the
    document yet). The discriminator is purely structural — the decomposition into
    pages lives in the authored DAG, never in code."""
    return bool(
        getattr(node, "role", None) == ROLE_SYNTHESIZER
        or (node.tool or "") in ("file_write", "write_file")
    )


# A result validator decides whether a finished node's output is LOGICALLY valid;
# returning a non-empty reason string marks the step logically-invalid (the third
# self-heal failure mode). Returning None/empty means the result is accepted.
ResultValidator = Callable[[PlanNode, "SubAgentResult"], Optional[str]]

# A replanner re-derives a MINIMAL corrective sub-graph for a failed node. It is
# injected (typically ``Planner.replan_subgraph``) so the runtime stays decoupled
# from the planner. Args: (failed_node, error_text, already_completed_ids).
Replanner = Callable[[PlanNode, str, list[str]], Awaitable[PlanDAG]]

# A per-node VERIFY GATE (Stage-B run engine). Given the finished node + its
# produced result it decides whether the output is acceptable — the
# ``verifiable → done`` gate. It may return a bare ``bool`` (True = pass) or a
# ``(ok, reason)`` tuple (the reason is fed to the inline reviewer on a fail).
# Sync or awaitable. Injected so the engine stays decoupled from what "correct"
# means for a given node/spec; when absent the gate trivially passes (every node
# still traverses VERIFIABLE — the lifecycle is mandatory, the gate is optional).
NodeVerifier = Callable[[PlanNode, "SubAgentResult"], Any]

# The SHAPING framing (d1): a specialization body is an OUTPUT-SHAPING RULESET,
# not a task and not a skill how-to. At produce time it is injected as the SYSTEM
# turn ABOVE the spec body, and the REAL task content + tool findings ride the
# USER turn (see ``SubAgent._compose_task``). This preamble makes the separation
# explicit so the model DOES the task and only shapes the FORM of its output by
# the ruleset — it must never describe the skill instead of doing the task (the
# round-1 Iran->markdown-how-to bug). When a node has NO spec the produce step
# has no system prompt at all (a bare step), so this framing is only added
# alongside a real ruleset body.
_SHAPING_FRAMING = (
    "The text below is an OUTPUT-SHAPING RULESET, not the task. DO the task in the "
    "user message (using its inputs and tool findings), then shape the FORM of "
    "your answer to follow the ruleset. The ruleset governs structure/format only; "
    "the content must be the real task result. Never describe the ruleset or write "
    "about the skill instead of doing the task."
)

# STRUCTURED-DATA -> PRESENTATION SEPARATION (s13/P1-report, d115 parity). A small
# writer over-produces a WHOLE mini-document on the first section (its own title + a
# consolidated figures/Sources block), then later sections repeat that shell — the
# B8a2 thematic duplicate-tail (two <h1>s, cost material twice) and the mid-section
# truncation. The cause is conflating DATA (the shared figures/timeline/sources) with
# PRESENTATION (the prose document). This guidance separates the two WITHOUT scripting
# a pipeline — the agent still structures its own work: it MAY first capture the key
# facts as a structured DATA file (a .json, written with file_write — legitimate data,
# never the deliverable's prose), then author the report from it. The deliverable
# itself stays RAW prose/HTML (d50.1, never a JSON envelope). The invariants: ONE
# document title written once, each later part a sub-section (never a second title),
# the shared figures/Sources block written ONCE (never repeated per section).
_REPORT_SEPARATION_GUIDANCE = (
    " SEPARATE DATA FROM PRESENTATION: the shared facts (key figures, the timeline, the "
    "SOURCES list) are the DOCUMENT'S, written ONCE — never repeated in a later section. "
    "You MAY first save those key facts to a small structured data file (a .json, via "
    "file_write) and then write the report prose from it, but the report itself is RAW "
    "content, never JSON. Give the document ONE title (a single top-level heading) on "
    "this first section; every later section is a SUB-section under it — never open a "
    "second top-level title or a second figures/sources block. Finish each sentence you "
    "start — never stop a section mid-sentence."
)

# DAG SPEC-COMPOSITION (d2/d11): a node may carry 1+ specs (``node.effective_specs``).
# When it carries MORE THAN ONE, their ruleset bodies are LAYERED into a single
# composed stack — the produce SYSTEM — in a DETERMINISTIC, DOCUMENTED ORDER:
#
#   COMPOSITION ORDER = the order the node lists its specs (``effective_specs``,
#   i.e. exactly the planner's emitted ``specs`` list order). It is stable and
#   reproducible: same node -> same stack, every run.
#
# Each spec's body is introduced by a labelled separator header so the model (and
# a human reading a trace) can see the layer boundaries; the single ``_SHAPING_
# FRAMING`` preamble wraps the WHOLE stack ONCE (it is the shaping contract for
# every layer, not per-layer). This mirrors eda-base3's ``assemble_ruleset``
# layering. A SINGLE-spec node is composed WITHOUT any header — byte-for-byte the
# pre-composition ``{framing}\n\n{body}`` — so single-spec behaviour never changed.
_RULESET_LAYER_HEADER = "===== Ruleset {i}/{n}: {name} ====="

# MULTI-SPEC RECONCILIATION (d47-req4): when a node carries 2+ specs they are
# COMBINED guidelines, applied in the listed (priority) ORDER — earlier = higher
# priority. If two rulesets CONFLICT, the worker RECONCILES in the moment by
# REASONING (NOT app-refusal, NOT a rigid first-wins drop): blend them where they
# can both be honored, and where they genuinely cannot, lean toward the
# higher-priority (earlier) ruleset's intent while preserving as much of the other
# as possible. This preamble leads a multi-spec stack so the small model treats the
# layers as one reconciled contract, never as contradictory orders to stall on.
_RULESET_RECONCILE_PREAMBLE = (
    "You have MORE THAN ONE output-shaping ruleset below, listed in PRIORITY ORDER "
    "(Ruleset 1 = highest priority). Follow ALL of them as COMBINED guidelines for "
    "the FORM of your answer. Where two rules CONFLICT, RECONCILE them yourself by "
    "reasoning: satisfy both wherever possible, and where you truly cannot, favor the "
    "higher-priority (earlier) ruleset while keeping as much of the lower one as you "
    "can. Never refuse or stall over a conflict — produce the best reconciled result."
)

# CONVERSATION MEMORY (s5/a4): the bounded prior-turn context (assembled per
# chat_id by chat_app's ConversationMemory) is injected into the produce-step USER
# turn so the NODE that authors the user-visible answer SEES prior turns — not just
# the planner. Threading it only into the planning goal (s5/a2) let the planner
# read the history but the planner-PARAPHRASED node tasks dropped the concrete
# facts, so the answering node hallucinated (e.g. invented a project codename).
# Carried on the USER turn (the task content), clearly delimited, BEFORE the task
# so the model grounds its answer in the thread. Empty/None => omitted entirely
# (byte-identical to the pre-fix user turn — no regression for a first turn).
_PRIOR_CONVERSATION_HEADER = (
    "PRIOR CONVERSATION (the user is continuing this thread — use these earlier "
    "turns to answer; treat any facts the user stated as authoritative):"
)

# OVERALL GOAL (d38/d39): the verbatim user request the WHOLE plan serves, fed into
# EVERY worker node's user turn. The probe confirmed the central gap — today only
# the planner sees the goal, and the per-node task it emits is a PARAPHRASE, so a
# downstream node works toward a lossy restatement and never the real objective.
# A Gemma node cannot DISCOVER the goal (no file/grep access like an eda-base3/
# Claude-Code worker), so it must be CONSTRUCTED and fed. It leads the user turn so
# the node keeps its specific task aligned to the real intent; empty => omitted
# (byte-identical to the pre-d39 user turn — no regression for a goal-less caller).
_OVERALL_GOAL_HEADER = (
    "OVERALL GOAL (the user's full request this whole plan serves — keep your work "
    "aligned to it; your specific task below is one part of achieving it):"
)

# The CODER=REVIEWER framing (d10): the SAME spec body that produced a node's
# output is re-used, now in a REVIEW posture, to correct the output INLINE when
# the gate rejects it — without re-triggering the produce step or the DAG loop.
_REVIEWER_FRAMING = (
    "You are now REVIEWING your own previous output for CORRECTNESS against the "
    "task above. An automated verification gate REJECTED that output for the "
    "stated reason. Apply the MINIMAL correction that fixes the problem and makes "
    "the output pass — do not restart the task from scratch and do not add "
    "commentary. Return ONLY the corrected output."
)

# A tool-arg emitter (re-)derives a node's tool kwargs just before the tool fires
# (s8/b1 phi hardening — typically :class:`~agent_runtime.toolargs.SchemaToolArgEmitter`).
# It is given the node and returns the kwargs to call the tool with (sync or
# awaitable). Injected so the runtime stays decoupled from the emission strategy.
ToolArgEmitter = Callable[[PlanNode], Any]


@dataclass
class SubAgentResult:
    """The output of one launched sub-agent."""

    node_id: str
    spec: Optional[str]
    output: Optional[str]
    tool_used: Optional[str] = None
    tool_value: Any = None
    heal: Optional[dict[str, Any]] = None
    replanned: bool = False
    # The FULL ordered list of spec names composed onto this node (d2/d11). For a
    # single-spec node this is ``(spec,)``; for a bare node it is empty; ``spec``
    # stays the PRIMARY (first) one for back-compat display.
    specs: tuple[str, ...] = ()
    # ROLE-NODE structured output (a3 generic role execution). When the node
    # carries a ``role``, its phi call is schema-constrained (the per-role output
    # schema) so the parsed JSON is available here; a judgment role additionally
    # carries the validated enum ``verdict`` and the verdict-repair count. For a
    # plain (role-less) node these stay None/empty — byte-identical to before.
    role: Optional[str] = None
    parsed: Any = None
    verdict: Optional[str] = None
    verdict_repairs: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "spec": self.spec,
            "specs": list(self.specs),
            "output": self.output,
            "tool_used": self.tool_used,
            "replanned": self.replanned,
            "role": self.role,
            "verdict": self.verdict,
        }


class SubAgent:
    """One launched agent — scoped to ONE task + ONE compiled spec body (d10).

    Stage-B context-scoping is BY CONSTRUCTION: the agent is handed a
    :class:`~agent_runtime.scope.ScopedSpec` (a single resolved ``{name, body}``)
    — NOT a loader/registry — so it cannot reach another spec's body. (A legacy
    ``loader=`` is still accepted for back-compat; it is used ONCE in ``__init__``
    to resolve the single body and is NOT retained, so the by-construction
    guarantee holds either way — see :meth:`~agent_runtime.scope.ScopedSpec.assert_no_loader`.)

    The optional ``hook`` lets it invoke its one hinted tool; a failing tool
    surfaces as :class:`ToolFailureError`. An optional ``result_validator`` lets
    the runtime reject a logically-invalid result as :class:`InvalidStepError`.
    """

    def __init__(
        self,
        node: PlanNode,
        *,
        transport: Transport,
        scope: Optional[ScopedSpec] = None,
        scopes: Optional[Sequence[ScopedSpec]] = None,
        loader: Optional[SpecLoader] = None,
        hook: Optional[ToolHook] = None,
        result_validator: Optional[ResultValidator] = None,
        tool_arg_emitter: Optional["ToolArgEmitter"] = None,
        call_opts: Optional[Mapping[str, Any]] = None,
        conversation_context: Optional[str] = None,
        overall_goal: Optional[str] = None,
        upstream_tool_values: Optional[Mapping[str, Any]] = None,
        max_repair_attempts: int = 2,
        max_verdict_repairs: int = 2,
        read_search_max_fetch: int = 0,
        search_tool: str = "web_search",
        fetch_tool: str = "web_fetch",
        emit_article_notes: bool = False,
        note_tool: str = "note",
        chunked_read: bool = False,
        verify_lane: bool = False,
        fetched_char_budget: int = 2000,
        upstream_input_char_budget: int = 4000,
        writer_source_budget: Optional[int] = None,
        chain_continue_path: Optional[str] = None,
        chain_is_final: bool = True,
        chain_sources: Optional[Sequence[Mapping[str, str]]] = None,
        read_embedder: Any = None,
    ) -> None:
        self.node = node
        self.transport = transport
        self.hook = hook
        # READ-SIDE relevance embedder (d109): the MiniLM ``CpuEmbedder`` used to rank a
        # fetched source's chunks against this node's sub-question (RELEVANCE-SELECT read).
        # The served deep-research route wires the shared embedder (``_load_read_embedder``);
        # ``None`` (e.g. a unit test) makes the read use the bounded map/reduce fallback.
        self._read_embedder = read_embedder
        # SOURCE-SCOPING (s9/c13, d56): the run's GLOBAL fetched-source list
        # (``[{title,url,markdown}, …]`` in stable 1-based order). A synthesis/write
        # node carrying ``node.source_ids`` is fed ONLY its assigned sources' real
        # text + URLs, placed NEAREST the generation cursor (see
        # :meth:`_scoped_source_block`) so they sit inside the model's ~512-token
        # sliding window — the d55/d56 SWA fix. None / a node with no source_ids =
        # the pre-c13 full-index behaviour (graceful 1-section degenerate case).
        self._chain_sources = list(chain_sources) if chain_sources else []
        # PLAN-CHAINING (c1b/d49.4): when this writer node is a CONTINUATION page of a
        # multi-page write-file plan, ``chain_continue_path`` is the file an UPSTREAM
        # writer already started — this node APPENDS its page/section onto it (reads
        # the real on-disk tail first, never overwrites). ``chain_is_final`` is False
        # when a DOWNSTREAM node is itself a writer (more pages follow), so this node
        # must NOT close the document wrapper yet (the closing-tag gate is deferred to
        # the final page). Both are computed structurally by the runtime from the DAG
        # topology (see :func:`_is_writer_node`) — None/True for every non-chained run,
        # so a single-file deliverable is byte-for-byte the pre-c1b behaviour.
        self._chain_continue_path = (chain_continue_path or "").strip() or None
        self._chain_is_final = bool(chain_is_final)
        # FETCH CAP (s9/c5, d49/d50 — reframes the retired d13 search-then-read GATE):
        # this is NO LONGER a flow gate that FORCES a web_search node to auto-fetch.
        # A web_search node is now a TRUE AGENT (:meth:`_run_research_loop`) that DECIDES
        # whether and which sources to read; this value only CAPS how many ``web_fetch``
        # calls that loop may make — a NON-FLOW cost/safety bound, not a flow switch.
        # 0 here means "no caller-set cap" → the loop applies ``RESEARCH_DEFAULT_FETCH_CAP``
        # so the agent can still read sources. (The old 0=OFF deterministic-skip semantics
        # are gone with the executor.)
        self._read_search_max_fetch = max(0, int(read_search_max_fetch))
        self._search_tool = search_tool
        self._fetch_tool = fetch_tool
        # ARTICLE-NOTE CONTROL LANE (s9/N2, d60/c15 part-b): when enabled, the research
        # ReAct loop offers a third lightweight tool (``note``) so the agent records a
        # per-article structured-CONTROL record (summary/category/source-trust/claims/
        # follow-ups) after reading a source — carried ADDITIVELY in
        # ``tool_value['article_notes']`` to DIRECT the next node's search (N4) and weight
        # provenance (N5). Default OFF → byte-identical to N1 (the note tool is absent, no
        # extra turns, no tool_value change); the deep-research orchestration opts in (N6).
        self._emit_article_notes = bool(emit_article_notes)
        self._note_tool = note_tool
        # CHUNKED READ (s9/N3, d62/c15 part-d): the READ-side analog of the c13/d55
        # write-side 512-token SWA window. When enabled, a fetched source LONGER than
        # ``fetched_char_budget`` is read via a map/reduce summary (each in-window chunk
        # summarized, the running summary flowing forward) instead of the flat
        # ``md[:budget]`` truncation — so the WHOLE article reaches the window, no
        # truncation, no fabrication. The summary is stored ADDITIVELY as ``summary`` on
        # the fetched dict; the full real ``markdown`` is UNTOUCHED (the c13 write-side
        # verbatim-citation path reads ``markdown`` and is unchanged). Default OFF →
        # byte-identical truncation, no summarizer call, no ``summary`` key; the deep-
        # research orchestration opts in (N6) and N3r leg-probes with the flag ON.
        self._chunked_read = bool(chunked_read)
        # NO-FABRICATION VERIFICATION LANE (s9/N5, d62/c15 part-e — the PRIMARY no-fab
        # mechanism per the neuron ruling). When enabled, this sub-agent runs the
        # REASONING claim->source provenance check (``agent_runtime.claim_verify``) at
        # two seams: (1) a RESEARCH node that wrote substantive findings with ZERO real
        # fetches (answered FROM MEMORY) is forced to GATHER-MORE before its findings are
        # accepted; (2) a terminal SYNTHESIS deliverable is re-checked claim-by-claim
        # against the run's fetched sources and the model is forced to GROUND or
        # REVISE/REMOVE any unbacked claim (the c13r B2 narrative-fabrication gap). It is
        # REASONING, never a regex/string content-filter (d14/d48). Default OFF →
        # byte-identical (no verify turn, no gather-more nudge, no rewrite); the served
        # deep-research / plan-chain gather turns it ON for every sub-agent it builds (N6).
        self._verify_lane = bool(verify_lane)
        # Per-article char budget when the FETCHED article text is folded into the
        # user turn (the generic 1200-char tool-output cap is far too small to carry
        # real article content to synthesis). Each fetched source is truncated to
        # this; with ~3 sources it fits the deep-research num_ctx comfortably.
        self._fetched_char_budget = max(400, int(fetched_char_budget))
        # INTER-NODE CONTEXT (o4 fix): per-upstream-dependency char budget when an
        # upstream node's produced PROSE is folded into this node's user turn. The
        # legacy hard-clip was 800 chars (~200 tok) — far too small to carry real
        # research to a downstream writer/synthesize node, so reports came out thin.
        # Parameterized with a sensibly LARGER default (not a new blind magic
        # constant); the final budget is confirmed against num_ctx in s5/s6. Bounded
        # below so a misconfigured tiny value can't silently re-introduce the clip.
        self._upstream_input_char_budget = max(200, int(upstream_input_char_budget))
        # WRITER SOURCE BUDGET (MSF/d89, fixes the BINDING seam ②): chars of EACH
        # section-assigned source's real article text fed to the writer. The legacy 700
        # starved the writer to ~0.3% of a long article (d87) → thin reports on every
        # model. None ⇒ the configured ``resolve_writer_source_budget()`` (env
        # RA_WRITER_SOURCE_BUDGET, default 12000). :meth:`_scoped_source_block` SIZES the
        # effective per-source budget to the num_ctx window before rendering so the
        # raised feed never reintroduces the d22 overflow→empty-thinking failure. Only
        # the report write path (source_ids + chain_sources present) consumes it →
        # default-safe, byte-identical elsewhere.
        self._writer_source_budget = (
            resolve_writer_source_budget()
            if writer_source_budget is None
            else max(120, int(writer_source_budget))
        )
        self._validate = result_validator
        # ROLE EXECUTION (a3): bounds on the per-role structured JSON parse/repair
        # and, for a JUDGMENT role, the enum-verdict re-emit loop (the b3 hardening,
        # now generic). Only consulted for a role-carrying node.
        self._max_repair_attempts = max_repair_attempts
        self._max_verdict_repairs = max_verdict_repairs
        # a2-recipe (s7/a2) TOOL-ARG GROUNDING: the RAW upstream tool VALUES (e.g.
        # the web_search results dict, keyed by dep node id) so the tool-arg emitter
        # can ground a derived-from-upstream arg in real data — web_fetch.url picked
        # from an actual search result, not hallucinated. The emitter ALSO sees the
        # upstream produced prose (passed to it as ``inputs`` at run time) so
        # file_write.content can be the real upstream report. None => the emitter
        # grounds from nothing (back-compat: identical to the pre-fix behaviour).
        self._upstream_tool_values = dict(upstream_tool_values or {})
        # CONVERSATION MEMORY (s5/a4): bounded prior-turn context for THIS chat
        # thread, injected into the produce USER turn so this node grounds its
        # answer in the conversation. None/blank => omitted (no regression).
        self._conversation_context = (conversation_context or "").strip()
        # OVERALL GOAL (d39): the verbatim user request the whole plan serves,
        # injected at the TOP of the produce/role USER turn so this node grounds its
        # specific (planner-paraphrased) task in the real objective. Blank => omitted
        # (no regression). Fed verbatim — the goal is the authoritative intent and is
        # small relative to num_ctx 32768; it is NOT clipped like upstream prose.
        self._overall_goal = (overall_goal or "").strip()
        # Options forwarded to the transport on the node's phi call (e.g.
        # ``num_predict``/``max_tokens`` so a writer node can produce a detailed
        # report rather than a truncated default). None = transport defaults.
        self._call_opts = dict(call_opts or {})
        # Optional schema-constrained tool-arg emitter (s8/b1 phi hardening): when
        # set, the node's tool args are (re-)derived through a JSON-schema phi call
        # before the tool fires, so a plan-time empty ``tool_args`` no longer hard-
        # fails the tool. None = use the node's own args verbatim (back-compat).
        self._tool_arg_emitter = tool_arg_emitter
        # Resolve the node's 1+ spec bodies up front (the sub-agent's whole
        # grounding). Precedence: an already-scoped set (``scopes``) > a single
        # scope (``scope``, back-compat) > resolve every ``effective_specs`` name
        # via the loader HERE and DO NOT retain the loader (by-construction d10).
        # Composition ORDER follows ``node.effective_specs`` (the planner's list).
        resolved: list[ScopedSpec]
        if scopes is not None:
            resolved = list(scopes)
        elif scope is not None:
            resolved = [scope]
        elif node.effective_specs and loader is not None:
            resolved = [ScopedSpec.resolve(loader, name) for name in node.effective_specs]
        else:
            resolved = []
        self.scopes: tuple[ScopedSpec, ...] = tuple(resolved)
        # Back-compat introspection: the PRIMARY (first) resolved scope, or None.
        self.scope: Optional[ScopedSpec] = self.scopes[0] if self.scopes else None
        # The composed shaping body (layered stack of every scope's ruleset, in
        # order). For a single spec this is exactly that one body (no headers).
        self.spec_body: str = self._compose_ruleset_stack()
        # Structural proof: this agent holds no loader/registry it could enumerate
        # (a tuple of frozen ScopedSpec bodies carries none).
        ScopedSpec.assert_no_loader(self)

    def _compose_ruleset_stack(self) -> str:
        """Layer every resolved spec body into ONE deterministic ruleset stack.

        ORDER = ``node.effective_specs`` order (the order the scopes were handed
        in). A SINGLE non-empty body is returned verbatim (no separator — exact
        single-spec back-compat). TWO OR MORE bodies are joined under labelled
        ``_RULESET_LAYER_HEADER`` separators so each layer's boundary is explicit.
        Empty-bodied scopes are skipped; an all-empty/zero set yields ``""``."""
        bodies = [(s.name, s.body) for s in self.scopes if (s.body or "").strip()]
        if not bodies:
            return ""
        if len(bodies) == 1:
            return bodies[0][1]
        n = len(bodies)
        parts = [
            f"{_RULESET_LAYER_HEADER.format(i=i + 1, n=n, name=name)}\n{body}"
            for i, (name, body) in enumerate(bodies)
        ]
        # d47-req4: lead the multi-spec stack with the reconciliation contract so the
        # worker follows the layers as COMBINED guidelines and reasons through any
        # conflict (never refuses / never blindly drops a layer).
        return _RULESET_RECONCILE_PREAMBLE + "\n\n" + "\n\n".join(parts)

    @property
    def spec_names(self) -> tuple[str, ...]:
        """The ordered names of every composed spec (empty for a bare node)."""
        return tuple(s.name for s in self.scopes)

    def _compose_task(self, inputs: Mapping[str, Any], tool_value: Any) -> str:
        """Assemble this node's USER turn — a DELIBERATE per-node context assembler.

        d38/d39 promotes this from an inputs-folder to the WORKER CONTEXT-ASSEMBLY
        ARCHITECTURE: a Gemma node cannot DISCOVER its context (no file/grep access
        like an eda-base3/Claude-Code worker), so everything it needs is CONSTRUCTED
        and fed here, in a fixed, legible order:

          1. OVERALL GOAL   — the verbatim user request the whole plan serves (d39),
                              so the node serves the real objective, not just the
                              planner's PARAPHRASED per-node task;
          2. PRIOR CONVERSATION — the bounded prior-turn memory (s5/a4);
          3. CURRENT TASK   — this node's specific (paraphrased) task;
          4. INPUTS         — each DIRECT dependency's produced PROSE (d17, budget-
                              clipped to ``_upstream_input_char_budget``);
          5. SOURCES & FINDINGS — each DIRECT dependency's raw tool VALUE (the key
                              sources / search results), budget-bounded in
                              :meth:`_render_tool_value`;
          6. this node's OWN tool output (if any).

        Dependency-scoped, NON-transitive (direct deps only, d17). The spec ruleset
        never enters here: the USER turn carries ONLY task content + context +
        findings; the SHAPING ruleset rides the SYSTEM turn (d1, see
        :meth:`_compose_system`). When neither a goal NOR prior conversation is set
        the output is byte-identical to the pre-d39 user turn (no regression)."""
        parts: list[str] = []
        # 1+2) Preamble: the overall goal (d39) then the prior-turn memory (s5/a4),
        # each clearly delimited so the model reads them as grounding CONTEXT, not as
        # new instructions. When a preamble is present, the node's own task is then
        # introduced by an explicit CURRENT TASK: header (exactly as the prior s5/a4
        # conversation-only behaviour did) so the boundary stays unambiguous.
        preamble: list[str] = []
        if self._overall_goal:
            preamble.append(f"{_OVERALL_GOAL_HEADER}\n{self._overall_goal}")
        if self._conversation_context:
            preamble.append(
                f"{_PRIOR_CONVERSATION_HEADER}\n{self._conversation_context}"
            )
        if preamble:
            parts.append("\n\n".join(preamble) + "\n\nCURRENT TASK:")
        # 3) This node's specific task.
        parts.append(self.node.task)
        if inputs:
            parts.append("\nINPUTS FROM PRIOR STEPS:")
            for k, v in inputs.items():
                parts.append(f"- {k}: {str(v)[: self._upstream_input_char_budget]}")
        # INTER-NODE CONTEXT (o4 fix, part 2): a downstream writer/synthesize node
        # has no tool of its OWN, so the rich fetched-source text its research
        # dependency retrieved only ever rendered into THAT node's turn — the writer
        # saw clipped prose and never the sources, producing thin/empty reports.
        # Fold each DIRECT dependency's raw tool value (the key sources / search
        # results) into this node's user turn via the same source-rendering path, so
        # synthesis grounds in the actual research. ``_upstream_tool_values`` is built
        # from ``node.depends_on`` only — DIRECT deps, NOT transitive accumulation
        # (out of scope). Each rendered value is already budget-bounded inside
        # :meth:`_render_tool_value` (fetched sources to ``_fetched_char_budget``,
        # other values to the compact cap), so this stays safe against num_ctx.
        for dep, uv in self._upstream_tool_values.items():
            if uv is None:
                continue
            parts.append(
                f"\nSOURCES & FINDINGS FROM PRIOR STEP {dep} "
                "(use this content directly):"
            )
            parts.append(self._render_tool_value(uv))
        if tool_value is not None:
            parts.append(self._render_tool_value(tool_value))
        return "\n".join(parts)

    def _render_tool_value(self, tool_value: Any) -> str:
        """Fold a tool's output into the user turn.

        d13 (a4): when the value carries FETCHED article content (a search node
        that followed through to ``web_fetch`` real URLs, or the research executor),
        render each source's EXTRACTED text under an explicit READ-NOT-DESCRIBE
        header with a generous per-source budget — so the node synthesises from the
        actual article text, not a 1200-char dump of the search-results page. Any
        other tool value keeps the original compact rendering (back-compat)."""
        if isinstance(tool_value, Mapping) and tool_value.get("fetched"):
            fetched = tool_value.get("fetched") or []
            parts = [
                "\nFETCHED SOURCE CONTENT — you have ALREADY retrieved and read the "
                "real articles below. State the actual findings from this text. "
                + READ_NOT_DESCRIBE
            ]
            for i, art in enumerate(fetched, 1):
                title = str(art.get("title") or "").strip() or "(untitled)"
                url = str(art.get("url") or "").strip()
                # Prefer the N3 in-window map/reduce summary (whole-document, window-safe)
                # when present; otherwise fall back to the budget-bounded raw markdown
                # (back-compat — no ``summary`` means chunked-read was OFF or unneeded).
                summary = str(art.get("summary") or "").strip()
                body = summary or (
                    str(art.get("markdown") or "").strip()[: self._fetched_char_budget]
                )
                parts.append(f"\n--- SOURCE {i}: {title} <{url}> ---\n{body}")
            return "\n".join(parts)
        return f"\nTOOL OUTPUT ({self.node.tool}):\n{str(tool_value)[:1200]}"

    def _with_source_index(self, user: str) -> str:
        """Append the authoritative REAL-SOURCE-URL index to a synthesizer USER turn (c12 #5).

        ROOT-CAUSE fix for fabricated/placeholder citations on the long path: the real
        fetched URLs DO reach the synthesizer (inside each upstream source's article
        body), but scattered through a huge prompt the small model cannot reliably
        reconstruct them and invents ``[Name, 2025]`` placeholders. Assemble the ACTUAL
        fetched URLs the orchestration already holds (``_upstream_tool_values``) into ONE
        compact, prominent, cite-ONLY-from-this list and append it — d17 context-feeding
        + d46/d49 no-fabrication (real data fed, the model still reasons about placement;
        NOT a citation template). No fetched sources (headlines, a haiku) => unchanged.

        SOURCE-SCOPING (s9/c13, d56): when THIS node carries planner-assigned
        ``source_ids`` against the run's global ``_chain_sources``, the full index is
        SUPPRESSED here — its scoped subset is instead placed NEAREST the generation
        cursor by :meth:`_scoped_source_block` (the SWA fix), so a section is not
        handed the whole 18k-74k-tok source block. A node without source_ids keeps the
        full-index behaviour (the single-node synth / 1-section degenerate case)."""
        if self.node.source_ids and self._chain_sources:
            return user
        block = render_source_index(
            collect_fetched_sources(self._upstream_tool_values.values())
        )
        return f"{user}\n\n----\n{block}" if block else user

    def _scoped_source_block(self) -> str:
        """This node's assigned sources, rendered TIGHT for nearest-cursor placement (c13).

        Returns the compact per-section source block (real article excerpts + URLs +
        the cite-verbatim/no-fabrication instruction) for a write/synthesis node that
        carries ``node.source_ids`` against the run's global ``_chain_sources`` — the
        (F) feed-scoping half of d56. Empty string when the node is unscoped or the
        global source list is absent (graceful no-op), so the block is appended at the
        very END of the section's first user turn ONLY when it has scoped sources.

        MSF/d89: the per-source budget is RAISED from 700 to ``self._writer_source_budget``
        (default 12k) but SIZED to the write window so a section's TOTAL source bytes stay
        under a fraction of num_ctx — ``min(per_source, window_chars*0.6/n_sources)``,
        window_chars ≈ num_ctx*3.5 — leaving room for the section prompt + num_predict
        output (never reintroducing the d22 overflow). The section's own task is passed as
        the relevance topic so each source's excerpt is SECTION-RELEVANT, not first-N-raw."""
        if not (self.node.source_ids and self._chain_sources):
            return ""
        # count THIS section's in-range assigned sources (the bytes that compete for the
        # window); at least 1 so a single-source section keeps the full per-source budget.
        n_sources = sum(
            1 for i in self.node.source_ids
            if isinstance(i, int) and 1 <= i <= len(self._chain_sources)
        )
        n_sources = max(1, n_sources)
        num_ctx = int(self._call_opts.get("num_ctx") or 0) or _DEFAULT_WRITE_NUM_CTX
        window_chars = num_ctx * _CHARS_PER_TOKEN
        fit = int(window_chars * _WRITE_SOURCE_WINDOW_FRACTION / n_sources)
        # never size BELOW the legacy floor — the cap only ever trims the RAISE down to
        # what the window holds, it does not re-starve a section with many sources.
        per_source = max(700, min(self._writer_source_budget, fit))
        return render_scoped_sources(
            self._chain_sources,
            self.node.source_ids,
            excerpt_budget=per_source,
            section_topic=self.node.task or "",
        )

    # URL extensions that ``web_fetch`` cannot turn into readable article TEXT:
    # Trafilatura is HTML-only, so a PDF/office/media URL decodes to binary garbage
    # and a research layer reports "unreadable binary data" instead of findings (the
    # max_iter=10 live finding). Skip these up front so the layer reads real prose.
    _NON_ARTICLE_EXT = (
        ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".csv",
        ".zip", ".gz", ".tar", ".png", ".jpg", ".jpeg", ".gif", ".svg",
        ".mp4", ".mp3", ".mov", ".avi", ".bin",
    )

    @classmethod
    def _looks_like_article_url(cls, url: str) -> bool:
        """A public http(s) URL that is plausibly a readable HTML page (not a file)."""
        if not url.startswith(("http://", "https://")):
            return False
        path = url.split("?", 1)[0].split("#", 1)[0].lower()
        return not path.endswith(cls._NON_ARTICLE_EXT)

    @staticmethod
    def _is_readable_fetch(val: Mapping[str, Any]) -> bool:
        """True if a web_fetch result carries READABLE article text (not binary).

        Trust the tool's ``extracted`` flag (Trafilatura produced article markdown);
        otherwise require a text-ish content type. A PDF/binary fetch (``extracted``
        False + a non-text content type) is rejected so its garbage never reaches the
        research call — the fix for the live "uninterpretable binary data" failure."""
        if val.get("extracted"):
            return True
        ctype = str(val.get("content_type") or "").lower()
        if "pdf" in ctype:
            return False
        return any(t in ctype for t in ("text/", "html", "json", "xml")) and bool(ctype)

    # ------------------------------------------------------------------ #
    # AGENTIC RESEARCH loop (s9/c5, d49/d50 — retires flags #1/#3)
    # ------------------------------------------------------------------ #
    async def _research_emit(
        self, system: Optional[str], convo: list[Mapping[str, Any]],
        opts: Mapping[str, Any],
    ) -> tuple[str, Optional[list[dict[str, Any]]]]:
        """ONE research-agent turn → ``(raw_text, native_tool_calls)`` (NO ``format=<schema>``).

        When ``opts`` carries ``tools=[...]`` the model can answer with a NATIVE
        ``message.tool_calls`` (surfaced on ``ctx.tool_calls`` by ``call_stage``); the
        raw text is also returned so a FINDINGS turn stays free prose (content RAW, d50.1)
        and a non-native reply still feeds the balanced-brace string fallback (s13).
        Offloaded off the event loop with the otel context re-attached so each turn's llm
        span nests under the research span (the never-freeze fix, mirrors
        :meth:`_run_raw_file_loop`)."""
        chain = Chain()
        chain.use(prompt_assembly())
        chain.use(call_stage(self.transport, **dict(opts)))
        ctx = Context(system=system, history=list(convo), transport=self.transport)
        ctx = await run_blocking_in_span(chain.run, ctx)
        return (ctx.raw_output or ""), getattr(ctx, "tool_calls", None)

    def _parse_research_call(
        self, raw: str
    ) -> Optional[tuple[str, dict[str, Any]]]:
        """Recover a lightweight ``(tool, args)`` research call from a turn, or None.

        A TOOL turn is a bare JSON object (the instruction asks for ONLY the JSON), so
        a fence-stripped turn that STARTS with ``{`` is parsed as a possible call; any
        other turn is the model's FINDINGS prose (returns None → loop ends). Only the
        research tools (``web_search``/``web_fetch``, plus ``note`` when article-note
        emission is enabled, N2) are accepted; an unparseable or unknown object is treated
        as findings, never silently dispatched."""
        accepted = (self._search_tool, self._fetch_tool)
        if self._emit_article_notes:
            accepted = accepted + (self._note_tool,)
        s = _strip_synth_fence(raw or "").strip()
        if not s.startswith("{"):
            return None
        blob = _first_json_object(s)
        if not blob:
            return None
        try:
            parsed = json.loads(blob)
        except (ValueError, TypeError):
            return None
        if not isinstance(parsed, Mapping):
            return None
        tool = parsed.get("tool") or parsed.get("name") or parsed.get("tool_name")
        args = parsed.get("args") or parsed.get("arguments") or parsed.get("parameters")
        if not (isinstance(tool, str) and tool.strip()):
            # Bare {<tool_name>: {...args}} slip.
            for key, val in parsed.items():
                if str(key).strip() in accepted:
                    tool, args = str(key).strip(), val
                    break
        name = str(tool).strip() if isinstance(tool, str) else ""
        if name not in accepted:
            return None
        if not isinstance(args, Mapping):
            args = {
                k: v for k, v in parsed.items()
                if k not in ("tool", "name", "tool_name", "args", "arguments", "parameters")
            }
        return name, dict(args)

    def _research_tool_specs(self, accepted: Sequence[str]) -> list[dict[str, Any]]:
        """Native tool schemas (s13) for the research ReAct tools the agent may call —
        keyed by the CONFIGURED tool names so a renamed search/fetch/note tool still maps.
        Passed as ``tools=[...]`` so the model returns real ``message.tool_calls`` instead
        of a "reply with ONLY JSON" prose turn the string parser could drop."""
        specs: list[dict[str, Any]] = []
        for name in accepted:
            if name == self._search_tool:
                specs.append(make_tool_spec(
                    name,
                    "STEP 1 — find candidate sources. Search the web for a focused "
                    "question to IDENTIFY reliable primary sources before reading. "
                    "Use query OPERATORS to sharpen results: \"exact phrase\", "
                    "site:domain / -site:domain, OR, leading - to exclude, "
                    "intitle:, filetype:pdf. Returns ranked {title,url,snippet} rows; "
                    "Wikipedia is excluded automatically. Then web_fetch the most "
                    "promising URLs.",
                    {"query": {"type": "string"}}, ["query"]))
            elif name == self._fetch_tool:
                specs.append(make_tool_spec(
                    name,
                    "STEP 2 — read a source. Fetch ONE result URL and READ its full "
                    "article text before you rely on it (never cite a page you have "
                    "not read). If it FAILS the result says WHY — forbidden (403), "
                    "not-found (404), timeout, or a denied domain — so pick a "
                    "DIFFERENT source rather than re-trying a dead link.",
                    {"url": {"type": "string"}}, ["url"]))
            elif name == self._note_tool:
                specs.append(make_tool_spec(
                    name,
                    "Record a SHORT control note about a source you just READ.",
                    {"url": {"type": "string"}, "summary": {"type": "string"},
                     "category": {"type": "string"}, "source_trust": {"type": "string"},
                     "key_claims": {"type": "array", "items": {"type": "string"}},
                     "relevance": {"type": "string"},
                     "gaps_or_followups": {"type": "array", "items": {"type": "string"}}},
                    ["url", "summary"]))
        return specs

    async def _dispatch_research_tool(
        self, tool: str, args: Mapping[str, Any],
        fetched: list[dict[str, str]], seen_urls: set[str],
    ) -> str:
        """Execute ONE model-chosen research tool call → an observation string.

        A ``web_search`` returns its top candidate rows (title/url/snippet); a
        ``web_fetch`` returns the EXTRACTED article markdown (and the source is appended
        to ``fetched`` so it can later ground a downstream node, d17). The hook publishes
        tool_call/tool_result on each invoke, so the live trace shows the model's real
        search/fetch decisions (the observability bar). A failed/dead/binary call yields
        a short non-fatal note, never an exception (a research turn must not crash)."""
        if tool == self._search_tool:
            query = str(
                args.get("query") or args.get("q") or args.get("search") or ""
            ).strip()
            if not query:
                return "web_search needs a non-empty \"query\". Try again."
            try:
                res = await self.hook.invoke(self._search_tool, query=query)
            except Exception as exc:  # noqa: BLE001 - a failed search must not crash the node
                return f"web_search failed: {exc}. Try a different query or write your findings."
            if not getattr(res, "ok", False):
                return f"web_search returned no results ({getattr(res, 'error', '')}). Try another query."
            rows = (res.value or {}).get("results") if isinstance(res.value, Mapping) else None
            rows = rows or []
            if not rows:
                return "web_search returned 0 results. Try a broader query."
            lines = ["SEARCH RESULTS (choose URLs to web_fetch and read):"]
            for row in rows[:8]:
                if not isinstance(row, Mapping):
                    continue
                url = str(row.get("url") or "").strip()
                title = str(row.get("title") or "").strip() or "(untitled)"
                snip = str(row.get("snippet") or "").strip()[:200]
                lines.append(f"- {title} <{url}>\n  {snip}")
            return "\n".join(lines)

        # web_fetch
        url = str(args.get("url") or args.get("link") or "").strip()
        if not url:
            return "web_fetch needs a non-empty \"url\". Choose one from the search results."
        if url in seen_urls:
            return f"Already read <{url}>. Fetch a DIFFERENT source or write your findings."
        seen_urls.add(url)
        if not self._looks_like_article_url(url):
            return (
                f"<{url}> is not a readable HTML article (PDF/file/binary). "
                "Choose a different source."
            )
        try:
            res = await self.hook.invoke(self._fetch_tool, url=url)
        except Exception as exc:  # noqa: BLE001 - a dead link must not fail the node
            return f"Could not fetch <{url}>: {exc}. Try another source."
        if not getattr(res, "ok", False):
            return f"Could not fetch <{url}>. Try another source."
        val = res.value if isinstance(res.value, Mapping) else {}
        # web_fetch surfaces a STRUCTURED failure (ok=False + a DISTINCT error_kind)
        # so the agent reacts correctly to WHY a read failed: a 403/blocked/denied
        # page will not yield to a retry (pick another source); a 404 is a dead link;
        # a deny-listed domain (e.g. Wikipedia) must never be cited. Relay the exact
        # reason instead of a single generic "try another source".
        if val.get("ok") is False:
            kind = str(val.get("error_kind") or "error")
            detail = str(val.get("error") or "").strip()
            return (f"Could not read <{url}> [{kind}]: {detail} "
                    "Choose a DIFFERENT source from the search results.")
        md = str(val.get("markdown") or "").strip()
        if not md or not self._is_readable_fetch(val):
            return f"<{url}> had no readable article text. Try another source."
        title = str(val.get("title") or "").strip() or url.rsplit("/", 1)[-1]
        final_url = str(val.get("final_url") or url)
        record: dict[str, str] = {"title": title, "url": final_url, "markdown": md}
        # READ the source into the window (N3): a long article is map/reduced into an
        # in-window factual summary instead of being truncated to the first budget chars
        # (which dropped the rest of the document); short sources pass through verbatim.
        body, summary, read_signal = await self._read_fetched(md, title, final_url)
        if summary is not None:
            record["summary"] = summary  # additive; full ``markdown`` stays untouched
        fetched.append(record)
        # COVERAGE SIGNAL (MSF/d89-b, fixes seam ⑤): tell the model HOW MUCH of the
        # source it actually has so it reasons about coverage instead of treating the
        # sliver as the whole article. A whole-doc map/reduce summary (chunked-read ON)
        # IS complete coverage; a flat truncation is NOT — say so and invite a follow-up.
        full_chars = len(md)
        if read_signal is not None:
            # d109 HONEST signal: counts + provenance for the RELEVANCE-SELECT read —
            # replaces the vague "there is MORE" nudge with the real M-found/X-read numbers
            # and which sources are now in hand, across the node's fetched docs.
            src_names = [
                (f.get("title") or f.get("url") or "(source)") for f in fetched
            ]
            provenance = ", ".join(src_names[-3:])
            read_note = (
                f"FETCHED <{title}> <{final_url}> — found {read_signal['found']} relevant "
                f"passages in this source; reading the top {read_signal['read']} "
                f"({read_signal['chars']} chars) most relevant to your question. You have "
                f"now read {len(fetched)} source(s): {provenance}. "
            )
        elif summary is not None:
            read_note = (
                f"FETCHED <{title}> <{final_url}> — you have now READ this WHOLE source "
                f"(a grounded factual summary covering all {full_chars} chars). "
            )
        elif len(body) < full_chars:
            read_note = (
                f"FETCHED <{title}> <{final_url}> — showing the first {len(body)} of "
                f"{full_chars} chars; this source has MORE. Note follow-ups or fetch it "
                "again to cover the rest. "
            )
        else:
            read_note = f"FETCHED <{title}> <{final_url}> — you have now READ this source. "
        return read_note + READ_NOT_DESCRIBE + f"\n\n{body}"

    def _read_content_budget(self) -> int:
        """Per-source char budget for the d109 RELEVANCE-SELECT read (FX0 token-bounded).

        FX0/d108 (swa_test.md): keep the node's TOTAL relevant content under ~20k tokens
        so content + question + history + a generation reserve stay under the 32768 window
        guard. The read is one-source-at-a-time and ACCUMULATES in the ReAct convo, so the
        ~20k-token content budget is SHARED across the node's fetched sources — divide it by
        the fetch cap so several sources coexist in one window. Floored at the legacy
        per-source ``_fetched_char_budget`` (never shrinks a configured read) and capped at
        the full ~20k-token budget (a single-source read can use it all)."""
        fetch_cap = (
            self._read_search_max_fetch
            if self._read_search_max_fetch > 0
            else RESEARCH_DEFAULT_FETCH_CAP
        )
        total = read_content_char_budget()  # ~20k tokens × 4 chars/token
        per_source = total // max(1, fetch_cap)
        return max(self._fetched_char_budget, min(per_source, total))

    async def _read_fetched(
        self, md: str, title: str, url: str
    ) -> tuple[str, Optional[str], Optional[dict]]:
        """Read a fetched source INTO the window → ``(body, summary_or_None, signal)``.

        d109 RELEVANCE-SELECT-then-SINGLE-READ (supersedes the d62/c15 whole-doc map/reduce
        framing for the READ path). With chunked-read OFF or a source already within
        ``_fetched_char_budget``, this is the legacy flat ``md[:budget]`` read (``summary``
        and ``signal`` ``None``) — byte-identical to before. With chunked-read ON and a LONG
        source, the source's paragraph-granular chunks are RANKED by MiniLM 384-d embedding
        similarity to THIS node's sub-question (``self.node.task``) and the TOP relevant
        passages are assembled (verbatim, no model call) up to the FX0 token budget, then
        read in ONE in-window call — replacing the per-source 75-chunk map/reduce. ``signal``
        carries ``{found, read, chars}`` for the honest coverage note. FALLBACK to the
        bounded map/reduce summary ONLY when no embedder is wired or selection yields
        nothing (e.g. the relevant content still overflows) — never a lexical rank."""
        budget = self._fetched_char_budget
        if not self._chunked_read or len(md) <= budget:
            return md[:budget], None, None

        # ── d109 primary: RELEVANCE-SELECT (embedding rank, single in-window read) ──────
        sub_question = (self.node.task or self._overall_goal or "").strip()
        embedder = self._read_embedder
        if embedder is not None and sub_question:
            try:
                excerpt, found, read = select_relevant_chunks(
                    md,
                    sub_question,
                    embedder.embed,
                    chunk_chars=READ_RANKING_CHUNK_CHARS,
                    char_budget=self._read_content_budget(),
                )
            except Exception:  # noqa: BLE001 - an embedding hiccup falls back, never crashes
                excerpt, found, read = "", 0, 0
            if excerpt:
                return excerpt, None, {"found": found, "read": read, "chars": len(excerpt)}

        # ── FALLBACK: bounded map/reduce whole-document summary (legacy N3) ─────────────
        system = self._compose_system()
        opts = dict(self._call_opts)
        opts.pop("format", None)

        async def _summarize(prompt: str) -> str:
            raw, _ = await self._research_emit(
                system, [{"role": "user", "content": prompt}], opts
            )
            return _strip_synth_fence(raw or "").strip()

        summary = (
            await _chunked_read(
                md, summarize=_summarize, title=title, url=url, char_budget=budget
            )
        ).strip()
        if not summary:  # graceful: never an empty read
            return md[:budget], None, None
        return summary, summary, None

    def _record_article_note(
        self, args: Mapping[str, Any],
        fetched: list[dict[str, str]], notes: list[dict[str, Any]],
    ) -> str:
        """Record ONE per-article CONTROL note (s9/N2) → an observation that STEERS next.

        The note can only describe a source the agent has REALLY read (anti-fabrication):
        with no fetched source yet, it is refused. The note's PROVENANCE
        (``source_id``/``url``/``title``) is taken from the matching fetched source — the
        model's ``url`` arg picks WHICH read source, but the canonical url/title/id come
        from the runtime, never the model. The reasoning fields (summary/category/claims/
        relevance/follow-ups) are the model's, coerced + bounded. The returned ack surfaces
        the note's ``gaps_or_followups`` so the next turn can search those angles (the
        artifact DIRECTING the next search)."""
        if not fetched:
            return (
                "Record a NOTE only AFTER you web_fetch and READ a source. "
                "Search or fetch a source first."
            )
        want = str(args.get("url") or args.get("link") or "").strip()
        src, idx = fetched[-1], len(fetched) - 1  # default: the source just read
        if want:
            for i, art in enumerate(fetched):
                if str(art.get("url") or "").strip() == want:
                    src, idx = art, i
                    break
        note = coerce_article_note(
            args, source_id=idx + 1,
            url=str(src.get("url") or ""), title=str(src.get("title") or ""),
        )
        if note is None:
            return (
                "Note not recorded (malformed). Continue: read another source or write "
                "your FINDINGS."
            )
        notes.append(note.model_dump())
        followups = "; ".join(note.gaps_or_followups) or "none noted"
        return (
            f"Note recorded for <{note.title or note.url}> (trust: {note.source_trust}). "
            f"Open follow-ups: {followups}. Continue: read another source, search a new "
            "angle for those follow-ups, or write your FINDINGS."
        )

    async def _run_research_loop(self, inputs: Mapping[str, Any]) -> "SubAgentResult":
        """Run a ``web_search`` node as a TRUE AGENT (d49/d50) — RETIRES flags #1/#3.

        The deterministic search-then-read EXECUTOR (a web_search node auto-following
        through to ``web_fetch`` whenever ``read_search_max_fetch`` > 0) is replaced by
        a ReAct loop in which the WORKER ITSELF decides to search and which sources to
        read. Each turn the model emits EITHER a lightweight tool call
        (``{"tool":"web_search","args":{"query":...}}`` /
        ``{"tool":"web_fetch","args":{"url":...}}`` — small args the small model emits
        reliably, unlike content-laden JSON, d49) OR its FINDINGS as RAW prose (no
        ``format=<schema>``; content RAW, d50.1). The loop EXECUTES each call against the
        real hook, feeds the REAL observation back, and ends when the model writes
        findings (or a NON-FLOW bound trips). ``read_search_max_fetch`` survives ONLY as
        the fetch CAP here (a cost/safety bound on how many web_fetch calls the agent may
        make), NOT a gate that forces the flow.

        Readable sources the agent chose to read are attached to the result
        ``tool_value`` as ``{"fetched": [...]}`` so a downstream synthesizer/writer node
        still grounds in the real article text via the inter-node SOURCES & FINDINGS feed
        (d17, rendered by :meth:`_render_tool_value`). The findings prose is the node
        ``output``. The result-validator gate runs as in :meth:`run`."""
        system = self._compose_system()
        # The fetch CAP is a NON-FLOW bound: the agent decides WHETHER/WHICH to fetch;
        # the cap only bounds total fetches. A caller that wired none (0) gets a sane
        # default so the agent can still read sources.
        fetch_cap = (
            self._read_search_max_fetch if self._read_search_max_fetch > 0
            else RESEARCH_DEFAULT_FETCH_CAP
        )
        # Reasoned tool calls survive without a wire schema (d34); strip any inherited
        # format so a turn is free to be a tool call OR raw findings.
        opts = dict(self._call_opts)
        opts.pop("format", None)
        # s13 NATIVE tool-call path: offer the research tools as real schemas so the model
        # answers with ``message.tool_calls`` (drop-immune) instead of a "reply with ONLY
        # JSON" prose turn. ``accepted`` mirrors :meth:`_parse_research_call` (the fallback).
        accepted = (self._search_tool, self._fetch_tool)
        if self._emit_article_notes:
            accepted = accepted + (self._note_tool,)
        tool_specs = self._research_tool_specs(accepted)
        if tool_specs:
            opts["tools"] = tool_specs

        # The per-article CONTROL-note clause is appended ONLY when notes are enabled
        # (N2); default OFF keeps the instruction (and the whole loop) byte-identical.
        instruction = _RESEARCH_LOOP_INSTRUCTION.format(fetch_cap=fetch_cap)
        if self._emit_article_notes:
            instruction += _RESEARCH_NOTE_CLAUSE
        base_user = self._compose_task(inputs, None)
        convo: list[dict[str, Any]] = [{
            "role": "user",
            "content": base_user + "\n\n" + instruction,
        }]
        fetched: list[dict[str, str]] = []
        notes: list[dict[str, Any]] = []  # per-article CONTROL notes (N2; additive)
        seen_urls: set[str] = set()
        searches = fetches = unproductive = 0
        gather_more = 0  # N5: how many times we forced a 0-fetch stage to gather (bounded)
        findings = ""

        # The turn ceiling rises proportionally with the fetch cap so a high-breadth
        # gather can read MANY sources without the flat ceiling clipping it (N1). A
        # narrow legacy cap keeps the original RESEARCH_MAX_TURNS floor (no change). When
        # article notes are on (N2), allow up to one extra (note) turn per fetch so the
        # note turns do not starve the fetch budget — still a NON-FLOW ceiling, and OFF
        # (note_budget=0) it is the exact N1 formula.
        note_budget = fetch_cap if self._emit_article_notes else 0
        max_turns = max(
            RESEARCH_MAX_TURNS, fetch_cap + RESEARCH_SEARCH_HEADROOM + note_budget
        )

        tracer = get_tracer("agent_runtime.research")
        with tracer.start_as_current_span("research.react") as span:
            span.set_attribute("research.node", str(self.node.id))
            span.set_attribute("research.fetch_cap", fetch_cap)
            span.set_attribute("research.max_turns", max_turns)
            for turn in range(max_turns):
                raw, tool_calls = await self._research_emit(system, convo, opts)
                # s13: prefer the NATIVE tool call (it rides its own channel, so leading
                # prose can never swallow it); fall back to the balanced-brace string parser
                # for a non-native reply. NEITHER → the model wrote its FINDINGS as prose.
                call = first_native_call(tool_calls, accepted) or self._parse_research_call(raw)
                convo.append({"role": "assistant", "content": raw or (
                    json.dumps({"tool": call[0], "args": call[1]}) if call else "")})
                if call is None:
                    # No tool call → the model wrote its FINDINGS (RAW prose) = done.
                    findings = _strip_synth_fence(raw or "").strip()
                    if findings:
                        # N5 no-fab GATHER-MORE: substantive findings with ZERO real
                        # fetches = answered FROM MEMORY (E4B does this on the bare ReAct
                        # path) — a no-fabrication FAILURE. Force the stage to actually
                        # search+read a real source before accepting, bounded so an
                        # unfetchable topic cannot loop forever. Default OFF (verify_lane
                        # False) → this block is skipped, the branch is byte-identical.
                        if (
                            self._verify_lane
                            and gather_more < _RESEARCH_GATHER_MORE_MAX
                            and research_answered_from_memory(findings, fetches)
                        ):
                            gather_more += 1
                            findings = ""
                            span.set_attribute(
                                f"research.turn.{turn + 1}", "gather_more"
                            )
                            convo.append({
                                "role": "user",
                                "content": _RESEARCH_GATHER_MORE.format(fetch_cap=fetch_cap),
                            })
                            continue
                        span.set_attribute(f"research.turn.{turn + 1}", "findings")
                        break
                    unproductive += 1
                    if unproductive >= 2:
                        break
                    convo.append({"role": "user", "content": _RESEARCH_NUDGE})
                    continue
                tool, args = call
                if tool == self._note_tool:
                    # A CONTROL-note turn (N2): record the per-article note and feed back
                    # an ack that surfaces its follow-ups to steer the next search. Does
                    # NOT touch the fetch/search budget (it is not a gather call).
                    obs = self._record_article_note(args, fetched, notes)
                    span.set_attribute(f"research.turn.{turn + 1}.tool", "note")
                    convo.append({"role": "user", "content": obs})
                    continue
                if tool == self._fetch_tool and fetches >= fetch_cap:
                    convo.append({"role": "user", "content": (
                        f"Fetch limit ({fetch_cap}) reached. Write your FINDINGS now "
                        "from the sources you have read."
                    )})
                    continue
                obs = await self._dispatch_research_tool(tool, args, fetched, seen_urls)
                if tool == self._search_tool:
                    searches += 1
                else:
                    fetches += 1
                span.set_attribute(f"research.turn.{turn + 1}.tool", tool)
                convo.append({"role": "user", "content": obs})

            # Fallback: the model never wrote findings (kept calling tools / stalled) →
            # salvage ONE final emission grounded in whatever it has read.
            if not findings:
                convo.append({"role": "user", "content": _RESEARCH_FINALIZE})
                final_raw, _ = await self._research_emit(system, convo, opts)
                findings = _strip_synth_fence(final_raw or "").strip()
            span.set_attribute("research.searches", searches)
            span.set_attribute("research.fetches", fetches)
            span.set_attribute("research.sources", len(fetched))
            span.set_attribute("research.notes", len(notes))
            span.set_attribute("research.gather_more", gather_more)  # N5 no-fab forces
            span.set_attribute("research.chars", len(findings))

        # Attach the read sources so a downstream node grounds in them (d17). The
        # findings prose is the node output (RAW, d50.1). The per-article CONTROL notes
        # (N2) ride ADDITIVELY alongside ``fetched`` — the c13 write-side path reads only
        # ``fetched``/``fetched_count`` (UNCHANGED); the notes lane DIRECTS the next
        # research node (N4) and weights provenance (N5). No notes (default OFF, or the
        # model emitted none) → no ``article_notes`` key → byte-identical to before.
        tool_value: Any = None
        if fetched:
            tool_value = {"fetched": fetched, "fetched_count": len(fetched)}
            if notes:
                tool_value["article_notes"] = notes
        result = SubAgentResult(
            node_id=self.node.id,
            spec=self.node.primary_spec,
            specs=self.spec_names,
            output=findings,
            tool_used=self._search_tool,
            tool_value=tool_value,
            role=self.node.role,
        )
        if self._validate is not None:
            reason = self._validate(self.node, result)
            if reason:
                raise InvalidStepError(
                    f"node {self.node.id!r} produced a logically-invalid result: {reason}",
                    node_id=self.node.id,
                    reason=reason,
                )
        return result

    def _compose_system(self) -> Optional[str]:
        """Compose the produce-step SYSTEM turn: the SHAPING layer (d1).

        When the node carries 1+ specs, the system is the single SHAPING framing
        followed by the COMPOSED ruleset stack (``_compose_ruleset_stack``: every
        spec body layered in ``effective_specs`` order, d2/d11) — so the model
        treats the stack as instructions for the FORM of its answer to the real
        task (carried on the user turn), never as the task itself or a skill
        how-to. A node with no spec is a bare step: the system is ``None`` (no
        ruleset to apply). This is the one seam where TASK-DOING (user) and
        RULESET-SHAPING (system) are composed.

        ROLE FRAMING (a3 generic role execution): when the node carries a ``role``
        (e.g. an unrolled deep-research node — research/critic/synthesis/verify),
        the role-prompt TEMPLATE is appended BELOW the shaping-framed spec body, so
        the SAME spec behaves differently per node, differentiated ONLY by role
        (the §2c mechanic, formerly in the deleted per-shape executor). A role with
        no spec body is just the role framing (a bare-role call).

        UNIVERSAL IDENTITY (d11 / s3-a3): the capable-agent persona
        (:func:`agent_runtime.identity.with_identity`) is prepended to EVERY node
        call — so even a bare, spec-less step now carries the identity as its
        system turn (it rode only the planner/specs before). It leads the system
        turn, with the shaping framing + ruleset (+ role) below it."""
        role = self.node.role
        framing = role_framing(role) if role else None
        if not self.spec_body:
            # bare step (identity only) or bare-role call (identity + framing).
            return with_identity(framing)
        base = f"{_SHAPING_FRAMING}\n\n{self.spec_body}"
        shaping = f"{base}\n\n{framing}" if framing else base
        return with_identity(shaping)

    async def run(self, inputs: Optional[Mapping[str, Any]] = None) -> "SubAgentResult":
        """Execute the node: optional tool call, then a scoped phi call.

        A failed tool call raises :class:`ToolFailureError`; a result the
        validator rejects raises :class:`InvalidStepError` (both self-heal
        classes). A transport-level error propagates unchanged."""
        inputs = inputs or {}
        # ROUTE-INDEPENDENCE (d50): a terminal FILE-DELIVERY node (tool=file_write/
        # write_file) is a TRUE AGENT — it authors the deliverable via the SAME raw-
        # content read-back loop the synthesizer uses (:meth:`_run_raw_file_loop`),
        # NOT a single deterministic content-dump or a schema-serialized content arg.
        # So file output is reliable on the acyclic web_search->file_write TOOL path
        # exactly as it is for the deep-research synthesizer node — content stays
        # RAW on every route (d50/d49). A SYNTHESIZER node owns its own file loop
        # (dispatched below); a no-hook/offline caller falls through to the legacy
        # path.
        if (
            self.hook is not None
            and self.node.role != ROLE_SYNTHESIZER
            and (self.node.tool or "") in ("file_write", "write_file")
        ):
            return await self._run_file_delivery(inputs)
        # AGENTIC RESEARCH (s9/c5, d49/d50): a ``web_search`` node is a TRUE AGENT. It
        # DECIDES to search and which sources to read via lightweight tool calls — the
        # deterministic search-then-read EXECUTOR (flags #1/#3: ``read_search_max_fetch``
        # > 0 forcing ``_read_search_results`` auto-fetch) is RETIRED in favour of the
        # ReAct loop (:meth:`_run_research_loop`). ``read_search_max_fetch`` survives
        # only as a NON-FLOW fetch CAP inside that loop, not a flow gate. (Flag #2, the
        # role-research gate, was already retired with the research role in B2/d48.)
        if (
            self.hook is not None
            and self.node.role != ROLE_SYNTHESIZER
            and self.node.tool == self._search_tool
        ):
            return await self._run_research_loop(inputs)
        tool_value: Any = None
        # A non-research tool node (e.g. a standalone web_fetch step) invokes its single
        # tool through the generic path below (keyed on the tool, not a role).
        if self.node.tool and self.hook is not None:
            # Derive the tool args. With the schema-constrained emitter wired in
            # (s8/b1), this (re-)emits the args through a JSON-schema phi call when
            # the plan-time args are missing/empty; otherwise the node's own args
            # are used verbatim. The emitter may be sync or async.
            tool_args = dict(self.node.tool_args)
            if self._tool_arg_emitter is not None:
                # a2-recipe (s7/a2): hand the emitter the upstream produced prose
                # (``inputs``) AND the raw upstream tool values so a derived arg
                # (web_fetch.url, file_write.content) is grounded in REAL upstream
                # data, never a hallucinated placeholder. Call defensively: a
                # legacy emitter accepting only ``node`` still works (the runtime's
                # own SchemaToolArgEmitter accepts the kwargs).
                try:
                    emitted = self._tool_arg_emitter(
                        self.node, inputs=inputs, tool_values=self._upstream_tool_values
                    )
                except TypeError:
                    emitted = self._tool_arg_emitter(self.node)
                if hasattr(emitted, "__await__"):
                    emitted = await emitted
                if emitted is not None:
                    tool_args = dict(emitted)
            res = await self.hook.invoke(self.node.tool, **tool_args)
            if not res.ok:
                raise ToolFailureError(
                    f"tool {self.node.tool!r} failed: {res.error}",
                    tool=self.node.tool,
                    call_id=res.call_id,
                )
            tool_value = res.value
            # NOTE (s9/c5): the deterministic SEARCH-THEN-READ auto-fetch (flags #1/#3)
            # that used to enrich a ``web_search`` value here is RETIRED — a web_search
            # node is now routed to the agentic :meth:`_run_research_loop` above, where
            # the worker ITSELF decides which sources to fetch. This generic path now
            # only serves non-research single-tool nodes.

        # Scoped chain: the SHAPING-framed spec ruleset is the system turn (d1/
        # d10), the real task content + findings is the user turn — TASK-DOING and
        # RULESET-SHAPING composed at the produce step, never conflated.
        system = self._compose_system()
        user = self._compose_task(inputs, tool_value)
        if self.node.role == ROLE_SYNTHESIZER:
            # SYNTHESIZER (d48): the terminal output stage. It authors the deliverable
            # via the shared raw-content read-back file loop (:meth:`_run_synthesis` →
            # :meth:`_run_raw_file_loop`), emitting RAW content (no ``format=<schema>``,
            # d50.1) so a long report can never truncate as an escaped JSON string. It
            # surfaces the written file as a ``file_write`` tool result so the chat
            # artifact carries the CHOSEN filename+extension (cats.html stays cats.html).
            # c12 #5: append the authoritative REAL-SOURCE-URL index so the deliverable
            # cites the actual fetched URLs verbatim, never a fabricated placeholder.
            user = self._with_source_index(user)
            raw, parsed, _verdict, _repairs = await self._run_synthesis(system, user)
            tool_used = self.node.tool
            result_tool_value = tool_value
            if isinstance(parsed, Mapping):
                syn_path = parsed.get("written_path")
                if syn_path:
                    tool_used = "file_write"
                    result_tool_value = {"path": str(syn_path)}
            result = SubAgentResult(
                node_id=self.node.id,
                spec=self.node.primary_spec,
                specs=self.spec_names,
                output=raw,
                tool_used=tool_used,
                tool_value=result_tool_value,
                role=self.node.role,
                parsed=parsed,
            )
        else:
            # WORKER (d48) or a bare (role=None) producer step: the role-execution
            # SWITCH (flag #5) is retired — a worker emits RAW free-text (no per-role
            # output schema, no enum-verdict path; d50.1 content is RAW). Its behavior
            # comes from its SPEC(s) + the task framing + reasoning, NOT a code switch.
            chain = Chain()
            chain.use(prompt_assembly())
            chain.use(call_stage(self.transport, **self._call_opts))
            ctx = Context(system=system, user=user, transport=self.transport)
            # FREEZE FIX (decouple): chain.run drives the SYNCHRONOUS, blocking phi
            # HTTP round-trip (call_stage -> transport.chat). This node coroutine
            # runs ON the shared event loop, so calling it inline would block the
            # loop for the whole phi latency and freeze the server (incl /health).
            # Offload to a worker thread so the loop keeps serving other nodes, SSE
            # streams and /health while phi works (mirrors toolargs.py). TRACING
            # (s6/b2): run_blocking_in_span re-attaches the active OTel context (the
            # per-node span) inside that worker thread, so the b1 per-phi-call LLM
            # span nests UNDER this node's span instead of detaching into a root trace.
            ctx = await run_blocking_in_span(chain.run, ctx)
            result = SubAgentResult(
                node_id=self.node.id,
                spec=self.node.primary_spec,
                specs=self.spec_names,
                output=ctx.raw_output,
                tool_used=self.node.tool,
                tool_value=tool_value,
                role=self.node.role,
            )
        if self._validate is not None:
            reason = self._validate(self.node, result)
            if reason:
                raise InvalidStepError(
                    f"node {self.node.id!r} produced a logically-invalid result: {reason}",
                    node_id=self.node.id,
                    reason=reason,
                )
        return result

    async def _run_file_delivery(self, inputs: Mapping[str, Any]) -> "SubAgentResult":
        """Run a terminal FILE-DELIVERY tool node as a TRUE AGENT (d50).

        The acyclic ``web_search``->``file_write`` route's terminal writer is now
        route-independent with the deep-research synthesizer: it authors the
        deliverable via the SAME shared raw-content read-back loop
        (:meth:`_run_raw_file_loop`) instead of a single deterministic content-dump.

        The SHAPING-framed spec ruleset is the system turn (``_compose_system``) and
        the user turn (``_compose_task``) carries the OVERALL GOAL + the upstream
        INPUTS prose + the upstream SOURCES & FINDINGS (the research the loop writes
        FROM) — so the node grounds in the real research, RAW, never a JSON envelope.
        The CHOSEN filename+extension (``derive_output_path``) reaches disk so the
        artifact carries the requested type verbatim (cats.html stays cats.html) on
        the tool route exactly as on the synthesis route. The written path is surfaced
        as a ``file_write`` result so the chat artifact name is correct (parity with
        the synthesis role surfacing). The result-validator gate runs as in
        :meth:`run`."""
        system = self._compose_system()
        user = self._compose_task(inputs, None)
        # c12 #5: carry the REAL fetched source URLs into the deliverable's citations on
        # the acyclic web_search->file_write route too (parity with the synthesis role).
        user = self._with_source_index(user)
        # PLAN-CHAINING (c1b): a continuation page appends to the file an upstream
        # writer started; otherwise the chosen filename is derived as before.
        out_path = self._chain_continue_path or derive_output_path(
            self._overall_goal, self.node.task, self.spec_names
        )
        doc, written_path, finished = await self._run_raw_file_loop(
            system, user, out_path,
            continue_existing=self._chain_continue_path is not None,
            is_final=self._chain_is_final,
        )
        result = SubAgentResult(
            node_id=self.node.id,
            spec=self.node.primary_spec,
            specs=self.spec_names,
            output=doc,
            tool_used="file_write",
            tool_value=({"path": written_path} if written_path else None),
            parsed={"output": doc, "written_path": written_path, "converged": finished},
        )
        if self._validate is not None:
            reason = self._validate(self.node, result)
            if reason:
                raise InvalidStepError(
                    f"node {self.node.id!r} produced a logically-invalid result: {reason}",
                    node_id=self.node.id,
                    reason=reason,
                )
        return result

    async def _run_synthesis(
        self, system: Optional[str], user: str
    ) -> tuple[Optional[str], Any, Optional[str], int]:
        """SYNTHESIS role: author the terminal deliverable via the SHARED agentic
        raw-content file loop (:meth:`_run_raw_file_loop`), then wrap its result in
        the role-return tuple.

        The CHOSEN filename+extension (``derive_output_path``) reaches disk, so the
        artifact carries the requested type (cats.html, not findings.md). Returns
        ``(raw_doc, {"output": doc, "written_path": path|None, "converged": bool},
        None, 0)`` so the downstream ``_render_parsed`` rendering is unchanged and
        ``run`` can surface the written path as a ``file_write`` result."""
        # PLAN-CHAINING (c1b): a continuation page appends to the file an upstream
        # writer started; otherwise the chosen filename is derived as before.
        out_path = self._chain_continue_path or derive_output_path(
            self._overall_goal, self.node.task, self.spec_names
        )
        doc, written_path, finished = await self._run_raw_file_loop(
            system, user, out_path,
            continue_existing=self._chain_continue_path is not None,
            is_final=self._chain_is_final,
        )
        return doc, {"output": doc, "written_path": written_path, "converged": finished}, None, 0

    async def _run_raw_file_loop(
        self,
        system: Optional[str],
        user: str,
        out_path: str,
        *,
        continue_existing: bool = False,
        is_final: bool = True,
    ) -> tuple[str, Optional[str], bool]:
        """The SHARED agentic raw-content file loop — ROUTE-INDEPENDENT (d49 → d50).

        A PLANNER-IN-THE-LOOP ReAct loop the ORCHESTRATION drives over the REAL file
        tools (``reactive_tools.file_tools`` — ``file_write``/``file_read``). It is
        called by BOTH terminal-deliverable routes so file output is reliable
        REGARDLESS of how the planner authored the writer (d50):

          * the deep-research SYNTHESIS *role* node (:meth:`_run_synthesis`);
          * the acyclic ``web_search``->``file_write`` *tool* node
            (:meth:`_run_file_delivery`).

        MEASURED on E4B (d49): asking a small model to EMIT ``file_write``/``file_read``
        JSON tool calls with embedded content fails for a real deliverable (0 parseable
        calls — the same escaped-string friction as D1). So the model emits its
        STRENGTH — RAW content sections (NO JSON, no ``format=<schema>``) — and the loop
        ACTS ON each emission: it WRITES the section to the real file (``file_write``
        append) and READS THE FILE BACK (``file_read`` tail), feeding the ACTUAL on-disk
        state to the next turn. The model continues from what it actually sees and
        signals completion with ``<<DONE>>`` — judged from the real file (ground truth),
        not a hard-coded heuristic (d48). Even when the model one-shots the whole
        document, the loop still WRITES it then READS IT BACK before the model confirms
        — so read-back fires unconditionally and kills BOTH false-finish and truncation
        (a truncated emission shows its real end on the read-back, and the next turn
        appends the rest).

        temp=0 (d35) and ``num_predict`` is floored. If the model emits nothing usable,
        ONE raw fallback emission is salvaged AND persisted to the chosen path. With no
        tool hook wired (offline unit callers) it degrades to that single raw emission.
        Returns ``(doc, written_path|None, finished)`` — the deliverable text read from
        the REAL file (ground truth), the path it landed at, and whether the loop
        converged on a clean ``<<DONE>>``."""
        # Off the schema role-path: drop any inherited format schema, force temp=0 for
        # write determinism (d35), and floor num_predict so the CoT + one section never
        # starve. ``think`` stays whatever the caller set (deep-research: think=True).
        base_opts = dict(self._call_opts)
        base_opts.pop("format", None)
        base_opts["temperature"] = 0
        base_opts["num_predict"] = max(
            int(base_opts.get("num_predict", 0) or 0), SYNTH_NUM_PREDICT
        )

        async def emit(
            convo: list[Mapping[str, Any]], num_predict: Optional[int] = None
        ) -> str:
            """ONE raw model turn (NO format schema): return the model's text verbatim.

            The model emits RAW content (its strength), not a JSON tool call — so
            there is no escaped-string friction. The loop, not the model, drives the
            file tools."""
            opts = dict(base_opts)
            if num_predict is not None:
                opts["num_predict"] = int(num_predict)
            chain = Chain()
            chain.use(prompt_assembly())
            chain.use(call_stage(self.transport, **opts))
            ctx = Context(system=system, history=list(convo), transport=self.transport)
            ctx = await run_blocking_in_span(chain.run, ctx)
            return ctx.raw_output or ""

        async def raw_emission() -> str:
            """ONE full schema-less emission (the fallback / no-hook path)."""
            return _strip_synth_fence(
                await emit(
                    [
                        {
                            "role": "user",
                            "content": (
                                user
                                + "\n\n----\nWrite the COMPLETE deliverable now, in full, "
                                "in the format the task asks for. Output ONLY the "
                                "deliverable content itself — no preamble, no JSON, no "
                                + DONE_SENTINEL + "."
                            ),
                        }
                    ]
                )
            )

        tracer = get_tracer("agent_runtime.synthesis")
        with tracer.start_as_current_span("synthesis.react_file") as span:
            span.set_attribute("synthesis.node", str(self.node.id))
            span.set_attribute("synthesis.out_path", out_path)

            # No file hook (offline unit callers): degrade to a single raw emission —
            # there is no real file to read back, but the deliverable is still produced.
            if self.hook is None:
                span.set_attribute("synthesis.mode", "raw_no_hook")
                doc = await raw_emission()
                span.set_attribute("synthesis.chars", len(doc))
                return doc, None, False

            span.set_attribute("synthesis.mode", "react_file")
            span.set_attribute("synthesis.chain_continue", bool(continue_existing))
            span.set_attribute("synthesis.chain_final", bool(is_final))

            writes = 0
            unproductive = 0
            written_path: Optional[str] = None
            finished = False
            size: Any = 0
            # s13/P2.3 (d130/d132.C): the ROOT-CAUSE structural fix for the
            # duplicate-tail — the section-write loop builds the document by an
            # ANCHORED read->targeted-insert->write (each section inserted just before
            # one unique terminal anchor) instead of a BLIND ``file_write(append=True)``
            # that concatenates a re-emitted chunk AFTER the closed document. This flag
            # tracks whether THIS node has done its first physical write yet (the write
            # that plants the anchor); subsequent writes insert before it.
            anchor_planted = False
            # R2 (d79, MS3): the on-disk text ALREADY on the file when THIS node starts
            # writing. A continuation page appends after it, so this node's OWN section is
            # ``final_file[len(chain_prefix_text):]`` — the unit the per-section verify
            # grounds (each section <9000 chars, so it is never bypassed by the whole-doc
            # _VERIFY_REVISE_MAX_CHARS cap). Empty for a fresh first page (prefix = "").
            chain_prefix_text = ""

            # R1 (c1r): the chosen type drives the closing-tag well-formedness gate —
            # but ONLY on the FINAL page (c1b). A non-final page of a multi-page chain
            # must NOT close the document wrapper (more pages follow); the gate is
            # deferred to the terminal page so the assembled file is closed exactly once.
            _ext = ("." + out_path.rsplit(".", 1)[-1].lower()) if "." in out_path else ""
            _is_html_ext = _ext in (".html", ".htm")
            # c10 #4: a CSV deliverable is TABULAR-ONLY. The multi-turn "keep going /
            # close with a SOURCES section listing URLs" continuation machinery (built
            # for prose/HTML reports) injects explanatory prose + a "Source 1: https://"
            # tail INTO the .csv. So a CSV is treated as single-shot: emit the full table
            # in one turn, no detailed/sources nudge, accept once the rows are written.
            _is_csv_ext = _ext == ".csv"
            # The close-tag well-formedness GATE runs only on the FINAL page (a
            # non-final page must leave the wrapper open); but the wrapper-closer STRIP
            # below (c1b) keys off ``_is_html_ext`` so it fires on every NON-final HTML
            # page, regardless of final status.
            is_html = _is_html_ext and is_final
            close_fix_sent = False
            # c8 R2: the markdown/text analogue of the HTML close-gap gate. A
            # detailed/thorough task that the model one-shots + <<DONE>> in a single
            # turn very likely dropped a requested section + the sources list (c8r);
            # there is no tag to balance, so detect the detailed INTENT from the task
            # and, on a first-turn finish, send ONE continuation nudge before
            # accepting. Scoped to a fresh single-file final deliverable (not a
            # continuation page, which may legitimately be short).
            md_completeness_eligible = (
                not is_html and not _is_csv_ext and is_final and not continue_existing
                and is_detailed_task(user)
            )
            md_continue_sent = False
            # The closing instruction is only given when this page may finish the
            # document; a non-final chain page is told to leave the wrapper open.
            _close_clause = (
                "close every HTML tag you open." if is_final
                else "do NOT close the document wrapper — later parts continue it."
            )

            # PLAN-CHAINING (c1b): a CONTINUATION page resumes the file an upstream
            # writer already started. Read the real on-disk tail and frame the model to
            # APPEND the NEXT page/section (never re-open/re-close the wrapper, never
            # repeat earlier pages). Mark the file already-written so the first write
            # APPENDS (append = written_path is not None) instead of clobbering it.
            if continue_existing:
                tail0 = ""
                rb0 = await self.hook.invoke("file_read", path=out_path, tail=1200)
                if rb0.ok and isinstance(rb0.value, Mapping):
                    tail0 = str(rb0.value.get("text") or "")
                    size = rb0.value.get("size", size)
                    written_path = out_path
                # R2 (d79): capture the FULL prior text so this node's own appended
                # section can be isolated for the per-section grounding verify below.
                rbp = await self.hook.invoke(
                    "file_read", path=out_path, max_bytes=4_000_000
                )
                if rbp.ok and isinstance(rbp.value, Mapping):
                    chain_prefix_text = str(rbp.value.get("text") or "")
                intro = (
                    "\n\n----\nA document is being written ACROSS PARTS; earlier "
                    "pages/sections are ALREADY on the file. Its current end is:\n-----\n"
                    f"{tail0}\n-----\nWrite the NEXT page/section NOW as RAW content in "
                    "the document's format (real HTML tags / markdown / plain text) — NO "
                    "JSON, no tool call, no preamble. Continue seamlessly from the end "
                    "shown; do NOT repeat earlier content; " + _close_clause
                    + " When THIS page/section is fully written, reply with EXACTLY "
                    + DONE_SENTINEL + "."
                )
            elif _is_csv_ext:
                # c10 #4: CSV is tabular-only and single-shot. NO multi-turn section
                # build, NO "add a SOURCES section" nudge (that is what appended the
                # "Source 1: https://" prose to the .csv). Ask for clean rows only.
                intro = (
                    "\n\n----\nWrite the deliverable as a CSV file. Output ONLY valid "
                    "CSV — a header row, then one comma-separated data row per record, "
                    "and NOTHING else. Do NOT add any prose, explanation, commentary, "
                    "title, markdown, code fence, or a 'Sources' section — the file must "
                    "be tabular data ONLY. Emit the COMPLETE table now in one turn, then "
                    "reply with EXACTLY " + DONE_SENTINEL + " on the next turn."
                )
            else:
                intro = (
                    "\n\n----\nWrite the deliverable to a file, built up ACROSS SEVERAL "
                    "TURNS — one section per turn, not all at once. Emit the FIRST section "
                    "NOW as RAW content in the final format the task asks for (real HTML "
                    "tags / markdown / plain text) — NO JSON, no tool call, no preamble, "
                    "just the content itself. Write ONE complete section this turn (start "
                    "with the headline + introduction), then STOP and wait: after each "
                    "section I WRITE IT to the file and show you the file's current end, so "
                    "you continue exactly where it left off (never repeat what is already "
                    "written). Do NOT put " + DONE_SENTINEL + " in a turn that contains "
                    "content, and do NOT compress a detailed report into one short turn. "
                    "CARRY THE SOURCES THROUGH: when the research above attributes facts to "
                    "source URLs, cite them using the REAL source URLs provided VERBATIM — "
                    "never a fabricated publication name, date, or '[Name, 2025]'-style "
                    "placeholder, and never a URL you were not given — and close with a "
                    "SOURCES section listing the URLs you used (never drop them). Reply with EXACTLY "
                    + DONE_SENTINEL + " and nothing else ONLY once the WHOLE deliverable is "
                    "on the file — for a 'detailed'/'thorough'/'in-depth' report that means "
                    "a substantive intro, the full body (timeline, figures/table), the "
                    "fallout/analysis, AND the sources are all written, not a short "
                    "summary; " + _close_clause + _REPORT_SEPARATION_GUIDANCE
                )

            # SOURCE-SCOPING (s9/c13, d56): for a section node carrying source_ids,
            # append its scoped sources block at the VERY END of the turn — nearest the
            # generation cursor — so the real figures/URLs sit inside the model's
            # ~512-tok sliding window during this section's generation (the SWA fix).
            # Empty for an unscoped node (degenerate 1-section / single-synth path).
            scoped_block = self._scoped_source_block()
            scoped_suffix = ("\n\n----\n" + scoped_block) if scoped_block else ""
            convo: list[dict[str, Any]] = [
                {"role": "user", "content": user + intro + scoped_suffix}
            ]
            max_turns = SYNTH_MAX_SECTIONS + 4

            async def closing_gap() -> list[str]:
                """The missing top-level closing tags on the REAL file (HTML only)."""
                if not is_html or written_path is None:
                    return []
                rb = await self.hook.invoke(
                    "file_read", path=written_path, max_bytes=4_000_000
                )
                if rb.ok and isinstance(rb.value, Mapping):
                    return html_close_gap(str(rb.value.get("text") or ""))
                return []

            async def accept_done() -> bool:
                """Gate ``<<DONE>>``: accept it, OR (R1) send ONE close-continuation.

                Before accepting the model's finish, check the REAL file for unclosed
                top-level HTML tags (ground truth, d48-clean). If any are open and we
                have not already nudged once, queue ONE "append only the closing tags"
                continuation and return False (keep looping). On the markdown/text path
                (no tags to balance), a DETAILED task finished in a single turn
                (``writes<=1``) very likely dropped a requested section + the sources
                list — read the real file back and queue ONE completeness continuation
                (also once). Otherwise accept."""
                nonlocal close_fix_sent, md_continue_sent
                # (R1) HTML close-tag gate.
                if not close_fix_sent:
                    gaps = await closing_gap()
                    if gaps:
                        close_fix_sent = True
                        convo.append(
                            {
                                "role": "user",
                                "content": (
                                    "The document is NOT closed — it is missing the "
                                    f"closing tag(s) {' '.join(gaps)}. Append ONLY "
                                    f"{' '.join(gaps)} now (exactly those closing tags, "
                                    "nothing else), then reply EXACTLY "
                                    + DONE_SENTINEL + "."
                                ),
                            }
                        )
                        return False
                # (R2) markdown/text first-turn completeness gate (c8). A
                # detailed/thorough task that finished in ONE turn bypasses the per-turn
                # continuation lever (which only fires on turn>=2); read the REAL file
                # (ground truth, d48-clean) and nudge ONCE to finish the remaining
                # requested parts + a Sources list. No content/citation template — the
                # model decides what is still missing from what it actually wrote (d49).
                if (
                    md_completeness_eligible
                    and not md_continue_sent
                    and writes <= 1
                ):
                    md_continue_sent = True
                    tail_text = ""
                    if written_path is not None:
                        rb = await self.hook.invoke(
                            "file_read", path=written_path, tail=900
                        )
                        if rb.ok and isinstance(rb.value, Mapping):
                            # Hide the planted anchor sentinel from the model-facing tail.
                            tail_text = strip_section_anchor(str(rb.value.get("text") or ""))
                    convo.append(
                        {
                            "role": "user",
                            "content": (
                                "You signalled done after a single turn, but a "
                                "detailed/thorough report is not complete in one pass. "
                                "The file currently ENDS with:\n-----\n"
                                f"{tail_text}\n-----\nContinue from exactly there (do NOT "
                                "repeat what is shown): write any remaining requested "
                                "parts the report is still missing — the full timeline, "
                                "the key figures, and the analysis/fallout — AND close "
                                "with a SOURCES section listing the FULL source URLs you "
                                "used. Reply EXACTLY " + DONE_SENTINEL + " only once all "
                                "of that is on the file."
                            ),
                        }
                    )
                    return False
                return True

            for _turn in range(max_turns):
                # R2 (c1r): on the LAST allowed turn, nudge a wrap-up BEFORE the ceiling
                # so the loop converges (emits <<DONE>> on a closed file) instead of
                # silently churning to the cap and shipping a thin, never-finished file.
                if _turn == max_turns - 1 and not finished and convo and convo[-1]["role"] == "user":
                    convo[-1]["content"] += (
                        "\n\nThis is the FINAL part — you cannot continue after this. "
                        "Write whatever remains and CLOSE the document now (close every "
                        "tag you opened), then reply EXACTLY " + DONE_SENTINEL + "."
                    )
                raw = await emit(convo)
                convo.append({"role": "assistant", "content": raw or ""})
                done, content = split_done_signal(_strip_synth_fence(raw))
                content = content.strip()
                if content and _is_html_ext:
                    # E4B intermittently ESCAPES a tag close as ``\>`` (e.g. ``</body\>``)
                    # — a model output artifact, never intended in HTML. De-escape it so
                    # the on-disk document is well-formed and the R1 close-tag gate reads
                    # the true tag balance (one stray backslash must not survive to disk).
                    content = content.replace("\\>", ">")
                    # PLAN-CHAINING wrapper hygiene (c1b): a NON-FINAL page must not close
                    # the document wrapper — the deferred-close contract closes it exactly
                    # once on the terminal page. But the small model writes </body></html>
                    # into every page it finishes, leaving a duplicate INTERIOR close
                    # mid-document (c1br defect). Strip those wrapper closers from each
                    # non-final page's RAW content BEFORE it is written, using the
                    # chain_is_final signal already at the node — so exactly one trailing
                    # </body></html> survives. Final page + single-file path untouched.
                    if not is_final:
                        content = strip_wrapper_closers(content)

                if content:
                    # c10 #2 RE-EMISSION GUARD (HTML): a small model nudged to "continue"
                    # an ALREADY-complete, closed top-level document responds by emitting a
                    # FRESH <!DOCTYPE>…</html> document. Appending it concatenates a SECOND
                    # document into the file — the duplicate-document defect that tag-BALANCE
                    # (2 opens + 2 closes) does NOT catch. If this chunk BEGINS a new
                    # document AND the real file already holds a complete, closed one, the
                    # deliverable is done: DROP the re-emission and STOP (ground truth read
                    # from the file, d48-clean — never the model's memory). A genuine next
                    # section (which does not open a new <html>) is unaffected.
                    if (
                        _is_html_ext
                        and written_path is not None
                        and begins_html_document(content)
                    ):
                        rb_full = await self.hook.invoke(
                            "file_read", path=written_path, max_bytes=4_000_000
                        )
                        prior = (
                            str(rb_full.value.get("text") or "")
                            if rb_full.ok and isinstance(rb_full.value, Mapping)
                            else ""
                        )
                        if top_level_html_doc_count(prior) >= 1 and not html_close_gap(prior):
                            finished = True
                            break
                    # c14 (d59) BODY-LEVEL re-emission guard: the long-report loop also
                    # re-emits an ALREADY-WRITTEN section set WITHOUT a fresh <!DOCTYPE>
                    # (so the document-level guard above misses it) — repeating the
                    # <h1>/<h2> heading FAMILY the file already holds (c13r 4/4 long runs).
                    # If THIS chunk carries headings whose families are ALL already on the
                    # real file (ground truth, d48-clean) it adds no new section: DROP it
                    # rather than appending a duplicate pass. Finish if the file is a
                    # complete closed document; else count it unproductive and nudge for
                    # the NEXT, not-yet-written section.
                    if (
                        _is_html_ext
                        and written_path is not None
                        and ("<h1" in content.lower() or "<h2" in content.lower())
                    ):
                        rb_dup = await self.hook.invoke(
                            "file_read", path=written_path, max_bytes=4_000_000
                        )
                        prior_doc = (
                            str(rb_dup.value.get("text") or "")
                            if rb_dup.ok and isinstance(rb_dup.value, Mapping)
                            else ""
                        )
                        if section_reemission(content, prior_doc):
                            if not html_close_gap(prior_doc):
                                finished = True
                                break
                            unproductive += 1
                            if unproductive >= 2:
                                break
                            convo.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "Those sections are ALREADY written to the file — "
                                        "do not repeat them. Write the NEXT, not-yet-written "
                                        "section's real content now as raw HTML, or reply "
                                        "EXACTLY " + DONE_SENTINEL + " if the deliverable "
                                        "is complete."
                                    ),
                                }
                            )
                            continue
                    # The ORCHESTRATION acts on the emission: WRITE it to the real file
                    # (append after the first), then READ IT BACK so the next turn sees
                    # the ACTUAL on-disk state — read-back fires on every part, even when
                    # the model one-shots the whole document.
                    append = written_path is not None
                    # c12 #2b: an APPENDED HTML section must contribute body content ONLY,
                    # never RE-OPEN the document — strip any re-emitted document-wrapper
                    # OPEN tags (a fresh <!DOCTYPE>/<html>/<head>/<body>) the small model
                    # habitually prepends to each section, the stray-sibling-opens defect.
                    # The FIRST write (append=False) keeps its opener — it establishes the
                    # document; and wrapper CLOSES are left intact here so the final
                    # section's legitimate </body></html> (and the deferred-close contract)
                    # survives — a doubled-close TAIL from sections that each closed is
                    # collapsed losslessly by the final single-document normaliser below.
                    # Runs AFTER the re-emission guard above, so a full-document
                    # re-emission onto an already-complete file still STOPS rather than
                    # being silently appended.
                    if append and _is_html_ext:
                        content = strip_wrapper_openers(content)
                        if not content.strip():
                            # The whole chunk was wrapper scaffolding, no real section
                            # content — nothing to add; nudge for the next part.
                            unproductive += 1
                            convo.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "That part contained only document-wrapper tags, "
                                        "no new content. Write the NEXT section's real "
                                        "content now as raw HTML, or reply EXACTLY "
                                        + DONE_SENTINEL + " if the deliverable is complete."
                                    ),
                                }
                            )
                            if unproductive >= 2:
                                break
                            continue
                    # s13/P2.3 (d130/d132.C): ANCHORED read->targeted-insert->write
                    # REPLACES the blind ``file_write(append=True)``. The FIRST physical
                    # write of this node creates (or first-appends, for a continuation
                    # page) the content WITH the terminal anchor planted at the document
                    # end; every LATER section is inserted JUST BEFORE that single unique
                    # anchor via ``file_update``. Because nothing is ever blind-appended
                    # PAST the document's end, a re-emitted chunk cannot form the
                    # duplicate-tail / 2nd top-level document — the structural prevention.
                    if not anchor_planted:
                        res = await self.hook.invoke(
                            "file_write", path=out_path,
                            content=plant_section_anchor(content),
                            append=append, overwrite=True,
                        )
                        if res.ok:
                            anchor_planted = True
                    else:
                        target = written_path or out_path
                        anchor = None
                        rba = await self.hook.invoke("file_read", path=target, tail=8192)
                        if rba.ok and isinstance(rba.value, Mapping):
                            anchor = choose_section_anchor(
                                str(rba.value.get("text") or ""), _is_html_ext
                            )
                        if anchor is not None:
                            old, new = anchored_insert_args(anchor, content)
                            res = await self.hook.invoke(
                                "file_update", path=target, old=old, new=new, count=1,
                            )
                            if not res.ok:
                                # The anchor vanished / is no longer unique: never lose
                                # the section — guarded append + REPLANT the anchor so
                                # the next section inserts cleanly again.
                                res = await self.hook.invoke(
                                    "file_write", path=target,
                                    content=plant_section_anchor(content),
                                    append=True, overwrite=True,
                                )
                        else:
                            # No unique anchor present (e.g. a continuation page whose
                            # prior node left the wrapper open and its sentinel stripped)
                            # — append + replant rather than risk a wrong-span edit.
                            res = await self.hook.invoke(
                                "file_write", path=target,
                                content=plant_section_anchor(content),
                                append=True, overwrite=True,
                            )
                    if res.ok and isinstance(res.value, Mapping):
                        writes += 1
                        unproductive = 0
                        written_path = str(res.value.get("path") or out_path)
                        size = res.value.get("size", size)
                        tail_text = ""
                        rb = await self.hook.invoke("file_read", path=written_path, tail=900)
                        if rb.ok and isinstance(rb.value, Mapping):
                            # Hide the planted anchor sentinel from the tail shown back to
                            # the model (it is internal bookkeeping, stripped at finalize).
                            tail_text = strip_section_anchor(str(rb.value.get("text") or ""))
                            size = rb.value.get("size", size)
                        if done:
                            # R1: accept finish only if the real file's top-level tags
                            # are closed; else accept_done queued ONE close-continuation.
                            if await accept_done():
                                finished = True
                                break
                            continue
                        if _is_csv_ext:
                            # c10 #4: a CSV is tabular-only and single-shot — the table is
                            # on the file. Do NOT nudge for "more sections / a SOURCES
                            # list" (that nudge is what appended the "Source 1: https://"
                            # prose to the .csv). Accept the deliverable now.
                            finished = True
                            break
                        convo.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Saved part {writes}. The file is now {size} bytes; "
                                    f"it ENDS with:\n-----\n{tail_text}\n-----\nContinue "
                                    "the deliverable from exactly there — write the NEXT "
                                    "part as raw content (do NOT repeat anything shown "
                                    "above). A detailed/thorough report is NOT complete "
                                    "after a single section: keep going until the full body "
                                    "(timeline, the key figures/table), the analysis/fallout, "
                                    "AND a SOURCES section listing the source URLs are all on "
                                    "the file — cite each claim's source as you go using the "
                                    "REAL SOURCE URLS verbatim (never a fabricated '[Name, 2025]' "
                                    "placeholder or a URL not in that list). Reply "
                                    "with EXACTLY " + DONE_SENTINEL + " ONLY once the WHOLE "
                                    "deliverable (all of those parts) is on the file."
                                ),
                            }
                        )
                    else:
                        unproductive += 1
                        convo.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Saving failed ({getattr(res, 'error', '')!r}). "
                                    "Re-send the next part as raw content."
                                ),
                            }
                        )
                else:
                    # No content this turn. DONE (with something written) ends the loop;
                    # a DONE before ANY write is rejected (an empty deliverable is never
                    # acceptable — the only floor, NOT a structural heuristic, d48).
                    if done and written_path is not None:
                        # R1: same close-tag gate on a content-less DONE turn.
                        if await accept_done():
                            finished = True
                            break
                        continue
                    unproductive += 1
                    if unproductive >= 2:
                        break
                    convo.append(
                        {
                            "role": "user",
                            "content": (
                                "Write the next part of the deliverable now as RAW "
                                "content (no JSON), or reply with EXACTLY "
                                + DONE_SENTINEL + " if every part is already on the file."
                            ),
                        }
                    )

                if unproductive >= 2:
                    break

            # s13/P2.3 (d130/d132.C): FINALIZE — strip the planted anchor sentinel so it
            # never reaches disk / the served document. Runs BEFORE the per-section verify
            # and doc assembly below (which read the real file as ground truth), so they
            # see the clean document. No-op (byte-identical) when no anchor was planted
            # (e.g. a raw-fallback one-shot), and the rewrite only fires when the sentinel
            # was actually present.
            if anchor_planted and written_path is not None:
                rb_fin = await self.hook.invoke(
                    "file_read", path=written_path, max_bytes=4_000_000
                )
                if rb_fin.ok and isinstance(rb_fin.value, Mapping):
                    raw_doc = str(rb_fin.value.get("text") or "")
                    clean_doc = strip_section_anchor(raw_doc)
                    if clean_doc != raw_doc:
                        w_fin = await self.hook.invoke(
                            "file_write", path=written_path,
                            content=clean_doc, append=False, overwrite=True,
                        )
                        if w_fin.ok and isinstance(w_fin.value, Mapping):
                            written_path = str(w_fin.value.get("path") or written_path)

            span.set_attribute("synthesis.writes", writes)
            span.set_attribute("synthesis.finished", finished)
            # R2 (c1r): SURFACE non-convergence — the loop hit the ceiling without the
            # model ever signalling <<DONE>> on a closed file. Never silently shipped as
            # a clean finish: a False here flags the trace (and rides the returned parsed
            # so the chat layer can see the deliverable may be incomplete).
            span.set_attribute("synthesis.converged", finished)
            if not finished:
                span.set_attribute("synthesis.non_convergence", True)

            # R2 SECTION-SCOPED PER-PAGE VERIFY (d79, MS3): ground THIS node's OWN
            # section against ITS scoped sources, INSIDE the write loop — so a long
            # multi-section report is actively grounded section-by-section instead of
            # relying on the whole-doc final verify, which is BYPASSED on >9000-char docs
            # (_VERIFY_REVISE_MAX_CHARS). Each section is < that cap, so a single revise
            # turn can safely re-emit it. REASONING-not-regex (verify_and_revise: the
            # model judges groundedness and rewrites; d14/d48). Scoped to the node's
            # planner-assigned source_ids (the d56 section→source map), falling back to
            # the full chain source set for an unscoped node. Default OFF (verify_lane) →
            # skipped, byte-identical short/headlines/csv path; the served deep-research
            # route turns it ON (N6). The whole-doc final verify below then SKIPS this
            # node (``section_verified``) so a clean report is not double-charged, and the
            # MSF-fed per-source budget is reused so a raised-budget-grounded claim is not
            # flagged by a starved verify excerpt.
            section_verified = False
            if (
                self._verify_lane
                and writes > 0
                and written_path is not None
                and self._chain_sources
            ):
                sec_full = ""
                rbs = await self.hook.invoke(
                    "file_read", path=written_path, max_bytes=4_000_000
                )
                if rbs.ok and isinstance(rbs.value, Mapping):
                    sec_full = str(rbs.value.get("text") or "")
                section_text = sec_full[len(chain_prefix_text):] if sec_full else ""
                if section_text.strip():
                    node_ids = [
                        i for i in (self.node.source_ids or [])
                        if isinstance(i, int) and 1 <= i <= len(self._chain_sources)
                    ]
                    section_sources = (
                        [self._chain_sources[i - 1] for i in node_ids]
                        if node_ids else list(self._chain_sources)
                    )

                    async def _sec_verify_turn(prompt: str) -> str:
                        return await emit([{"role": "user", "content": prompt}])

                    # s13/P1 (d118): the VERDICT rides NATIVE message.tool_calls — drop-
                    # immune to leading prose — using the SAME native helper the decision
                    # loop uses (_research_emit + REVIEWER_TOOL_SPECS). The RAW revise turn
                    # stays the text emit above (corrected document is RAW, d50). A non-
                    # native reply falls back to the kept balanced-brace parser.
                    async def _sec_verify_native(prompt: str):
                        opts = dict(base_opts)
                        opts["tools"] = list(REVIEWER_TOOL_SPECS)
                        return await self._research_emit(
                            system, [{"role": "user", "content": prompt}], opts
                        )

                    sec_rev = await verify_and_revise(
                        section_text, section_sources,
                        verify=_sec_verify_turn,
                        verify_native=_sec_verify_native,
                        # FIX-C: the verify lane is the GENERIC REVIEWER — feed it the
                        # producing node's SAME composed spec so it reviews against the
                        # rules the section was built to satisfy (empty → spec-blind).
                        spec=self.spec_body,
                        goal=self._overall_goal or self.node.task or "",
                        max_passes=2,  # verify → revise → RE-VERIFY
                        excerpt_budget=self._writer_source_budget,
                    )
                    section_verified = True
                    span.set_attribute("synthesis.section_verify", True)
                    span.set_attribute("synthesis.section_verify_chars", len(section_text))
                    span.set_attribute("synthesis.section_verify_grounded", bool(sec_rev.grounded))
                    span.set_attribute("synthesis.section_verify_unbacked", len(sec_rev.unbacked))
                    span.set_attribute("synthesis.section_verify_passes", sec_rev.passes)
                    if (
                        sec_rev.revised
                        and sec_rev.document.strip()
                        and sec_rev.document != section_text
                    ):
                        revised_section = sec_rev.document
                        # Re-apply the loop's wrapper hygiene to the re-emitted section so
                        # a continuation page never re-opens the document and a non-final
                        # page never closes the wrapper early (the c1b/c12 invariants).
                        if _is_html_ext:
                            if chain_prefix_text:
                                revised_section = strip_wrapper_openers(revised_section)
                            if not is_final:
                                revised_section = strip_wrapper_closers(revised_section)
                        new_full = chain_prefix_text + revised_section
                        if _is_html_ext:
                            if has_duplicate_html_structure(new_full):
                                new_full = enforce_single_html_document(new_full)
                            new_full = collapse_duplicate_sections(new_full)
                        # s13/P1-report: a revise turn capped at num_predict can come back
                        # CUT MID-SENTENCE; never persist a rewrite that truncates content
                        # the original did not — surface the verdict, keep the fuller file.
                        _revise_truncates = (
                            has_truncation_marker(new_full)
                            and not has_truncation_marker(sec_full)
                        )
                        if new_full.strip() and not _revise_truncates:
                            w = await self.hook.invoke(
                                "file_write", path=written_path,
                                content=new_full, append=False, overwrite=True,
                            )
                            if w.ok and isinstance(w.value, Mapping):
                                written_path = str(w.value.get("path") or written_path)
                        elif _revise_truncates:
                            span.set_attribute("synthesis.section_verify_revise_skipped_truncation", True)
                            span.set_attribute("synthesis.section_verify_revised", True)

            # Assemble the deliverable from the REAL FILE (ground truth) — not from the
            # model's reported sections, so the chat surfaces exactly what was written.
            doc = ""
            if written_path is not None:
                rb = await self.hook.invoke(
                    "file_read", path=written_path, max_bytes=4_000_000
                )
                if rb.ok and isinstance(rb.value, Mapping):
                    # Defensive: the finalize step above already stripped the planted
                    # anchor sentinel from disk; strip again here so the assembled doc is
                    # clean even if that rewrite was skipped/failed.
                    doc = strip_section_anchor(str(rb.value.get("text") or ""))

            # c10 #2 / c12 #2b SINGLE-DOCUMENT GATE (safety net): the re-emission guard +
            # the per-append wrapper strip prevent the common cases, but a model can still
            # cram TWO complete <!DOCTYPE>…</html> documents into a SINGLE emission (or
            # leave stray sibling <html>/<body> opens / a doubled close). Assert STRICT
            # single-document-ness on the assembled bytes (d48-clean: real file, not
            # memory) and, if duplicate top-level structure is present, normalise to
            # exactly ONE <!DOCTYPE>/<html>/<head>/<body>…</body></html> and REWRITE the
            # file so exactly one well-formed document reaches disk AND the chat artifact.
            # c14 (d59): also collapse a re-emitted BODY-LEVEL report pass / repeated
            # heading-FAMILY (a duplicate that carries no second <!DOCTYPE>, so
            # has_duplicate_html_structure alone misses it). enforce normalises the
            # document wrapper first; collapse then drops the duplicate section passes —
            # both d48-clean (real bytes, no fabricated content), no-op on a clean file.
            if _is_html_ext:
                normalized = doc
                if has_duplicate_html_structure(normalized):
                    normalized = enforce_single_html_document(normalized)
                normalized = collapse_duplicate_sections(normalized)
                if normalized != doc and normalized.strip():
                    w = await self.hook.invoke(
                        "file_write", path=written_path or out_path,
                        content=normalized, append=False, overwrite=True,
                    )
                    if w.ok and isinstance(w.value, Mapping):
                        written_path = str(w.value.get("path") or written_path or out_path)
                    doc = normalized
                    span.set_attribute("synthesis.docs_deduped", True)

            if not doc.strip():
                # FALLBACK: the model never wrote a usable file — salvage ONE raw
                # emission AND persist it so the chosen extension still lands on disk.
                span.set_attribute("synthesis.fallback_raw", True)
                doc = await raw_emission()
                if doc.strip():
                    w = await self.hook.invoke(
                        "file_write", path=out_path, content=doc, append=False, overwrite=True
                    )
                    if w.ok and isinstance(w.value, Mapping):
                        written_path = str(w.value.get("path") or out_path)

            # N5 (d62/c15 part-e) — the REASONING no-fabrication VERIFICATION lane over
            # the FINAL deliverable. Re-check every claim against the run's FETCHED
            # sources via claim->source provenance and force the model to GROUND or
            # REVISE/REMOVE any unbacked claim (the c13r B2 narrative-fabrication gap,
            # e.g. the fabricated 17 USC 107(5) / CTEA-1998). REASONING, never a
            # regex/string filter (d14/d48): the model judges groundedness AND rewrites;
            # this loop only orchestrates the turns and persists the corrected file.
            # Runs ONLY on the FINAL page of a deliverable that HAS fetched sources to
            # check against — a source-less creative deliverable is never nagged/stripped
            # (the steer's "do not strip valid content"). The whole-doc REWRITE is gated
            # on a size a single revise turn can safely re-emit (the verdict still
            # surfaces beyond it) + a retention floor in verify_and_revise, so a long
            # report is never truncated/blanked. Default OFF → skipped, byte-identical
            # c13 write side; the served deep-research route turns it ON (N6).
            # R2 (d79, MS3): SKIP the whole-doc verify when the per-section verify above
            # already grounded this node's content (``section_verified``). For a
            # single-section report that IS the whole doc (no double-charge); for a long
            # multi-section report every page was grounded by its own node's per-section
            # pass, so the whole-doc rewrite (bypassed >9000 chars anyway) adds nothing.
            # Falls through to the whole-doc lane only when no per-section verify ran
            # (e.g. an unscoped/source-less final page) — the prior behaviour, unchanged.
            if (
                self._verify_lane
                and is_final
                and not section_verified
                and doc.strip()
                and self._chain_sources
                and written_path is not None
            ):
                async def _verify_turn(prompt: str) -> str:
                    return await emit([{"role": "user", "content": prompt}])

                # s13/P1 (d118): native verdict turn (drop-immune to leading prose),
                # same helper + reviewer tool surface as the per-section pass above.
                async def _verify_native_turn(prompt: str):
                    opts = dict(base_opts)
                    opts["tools"] = list(REVIEWER_TOOL_SPECS)
                    return await self._research_emit(
                        system, [{"role": "user", "content": prompt}], opts
                    )

                rev = await verify_and_revise(
                    doc, self._chain_sources,
                    verify=_verify_turn,
                    verify_native=_verify_native_turn,
                    # FIX-C: spec-aware generic reviewer (empty spec → spec-blind).
                    spec=self.spec_body,
                    goal=self._overall_goal or self.node.task or "",
                    max_passes=2,  # verify → revise → RE-VERIFY (confirm the fix grounded it)
                    # MSF/d89 lockstep: Seam-B judges against the SAME per-source budget
                    # the writer was fed, so a claim grounded in raised-budget text is not
                    # flagged unbacked by a starved 700-char verify excerpt.
                    excerpt_budget=self._writer_source_budget,
                )
                span.set_attribute("synthesis.verify_lane", True)
                span.set_attribute("synthesis.verify_grounded", bool(rev.grounded))
                span.set_attribute("synthesis.verify_unbacked", len(rev.unbacked))
                span.set_attribute("synthesis.verify_passes", rev.passes)
                # Persist a real rewrite only when it came back substantive and within a
                # single-turn output budget (a long doc would truncate on one revise
                # turn — surface the verdict, don't gut the file). The retention floor in
                # verify_and_revise already rejects a catastrophically short rewrite.
                if (
                    rev.revised
                    and rev.document.strip()
                    and rev.document != doc
                    and len(doc) <= _VERIFY_REVISE_MAX_CHARS
                ):
                    normalized = rev.document
                    if _is_html_ext:
                        if has_duplicate_html_structure(normalized):
                            normalized = enforce_single_html_document(normalized)
                        normalized = collapse_duplicate_sections(normalized)
                    # s13/P1-report: never replace the file with a rewrite that ends
                    # MID-SENTENCE (a num_predict cut) when the original did not — keep
                    # the fuller verified file and surface the verdict, don't truncate it.
                    _wd_truncates = (
                        has_truncation_marker(normalized)
                        and not has_truncation_marker(doc)
                    )
                    if normalized.strip() and not _wd_truncates:
                        w = await self.hook.invoke(
                            "file_write", path=written_path,
                            content=normalized, append=False, overwrite=True,
                        )
                        if w.ok and isinstance(w.value, Mapping):
                            written_path = str(w.value.get("path") or written_path)
                        doc = normalized
                        span.set_attribute("synthesis.verify_revised", True)
                    elif _wd_truncates:
                        span.set_attribute("synthesis.verify_revise_skipped_truncation", True)
                elif rev.revised:
                    # A rewrite was produced but the doc is too long to safely re-persist
                    # in one turn — keep the verified-but-unrewritten file; surface it.
                    span.set_attribute("synthesis.verify_revise_skipped_size", True)

            # FINAL-DOCUMENT NO-FAB URL GUARD (d84/d89, MS3): a DETERMINISTIC post-pass
            # over the assembled deliverable — every URL is checked against the run's
            # FETCHED-source set (each source's URL + every URL embedded verbatim in the
            # fetched article text), and any ungrounded/fabricated URL is REMOVED (its
            # anchor unwrapped to the visible text, bare occurrences dropped). Makes the
            # d60 no-fabrication guarantee deterministic rather than model-luck: even if
            # the reasoning verify lane misses a hallucinated link, no fabricated URL can
            # survive to the delivered file, for ANY model. Content-preserving (strips
            # only the offending token, never invents prose) + idempotent (a clean report
            # is byte-identical). Runs ONLY on the served report path (verify_lane on) +
            # final page + with fetched sources to check against → default-safe; a
            # source-less or short/headlines/csv deliverable is untouched.
            if (
                self._verify_lane
                and is_final
                and doc.strip()
                and self._chain_sources
                and written_path is not None
            ):
                guarded, removed_urls = strip_ungrounded_urls(doc, self._chain_sources)
                span.set_attribute("synthesis.url_guard", True)
                span.set_attribute("synthesis.url_guard_removed", len(removed_urls))
                if removed_urls and guarded.strip() and guarded != doc:
                    if _is_html_ext:
                        if has_duplicate_html_structure(guarded):
                            guarded = enforce_single_html_document(guarded)
                        guarded = collapse_duplicate_sections(guarded)
                    if guarded.strip():
                        w = await self.hook.invoke(
                            "file_write", path=written_path,
                            content=guarded, append=False, overwrite=True,
                        )
                        if w.ok and isinstance(w.value, Mapping):
                            written_path = str(w.value.get("path") or written_path)
                        doc = guarded
                        span.set_attribute("synthesis.url_guard_applied", True)

            # DETERMINISTIC SOURCE-COVERAGE NET (s13/B5c, design §4C): a final pass UNDER
            # the loop — appends an "Additional sources" reference block for every fetched
            # source the PHASE-2 write planner assigned to NO section AND that is not
            # already present (cited/listed) in the assembled doc, so a source the planner
            # skipped cannot silently vanish (the d87 dropped-source risk; the
            # ``_ensure_source_coverage`` d89 specified but never shipped, a1 Fact 4d).
            # d60-safe: adds ONLY a title+URL reference for material the run ACTUALLY
            # fetched — never invents content. Runs BEFORE the structure reconcile so a
            # newly-added <h2> is folded into the re-derived ToC; the in-loop agent stays
            # the PRIMARY coverage mechanism, this is only the net for the one it skipped.
            # Gated by the run's chain sources (present ONLY on the report write path) +
            # final page → default-safe; a source-less / short / headlines / csv
            # deliverable has no chain sources and is untouched. Independent of the verify
            # lane: coverage is a structural guarantee, not a no-fabrication concern.
            if (
                is_final
                and doc.strip()
                and self._chain_sources
                and written_path is not None
            ):
                covered, added_sources = ensure_source_coverage(
                    doc,
                    self._chain_sources,
                    self.node.source_ids or [],
                    is_html=_is_html_ext,
                )
                span.set_attribute("synthesis.coverage_net", True)
                span.set_attribute("synthesis.coverage_added", len(added_sources))
                if added_sources and covered.strip() and covered != doc:
                    w = await self.hook.invoke(
                        "file_write", path=written_path,
                        content=covered, append=False, overwrite=True,
                    )
                    if w.ok and isinstance(w.value, Mapping):
                        written_path = str(w.value.get("path") or written_path)
                    doc = covered
                    span.set_attribute("synthesis.coverage_applied", True)

            # FINAL DOC-STRUCTURE INTEGRITY BACKSTOP (s13/B5, design §4B): the LAST pass
            # over the fully-assembled HTML — after every section write, the dedup/verify
            # lanes and the URL guard, so it sees the document EXACTLY as it will be
            # served. A real stdlib parser re-derives the ToC from the actual final
            # h1..h3 set (so a late-appended section is navigable — the d93 ToC miss),
            # renames any duplicate element id, and balances the wrapper. Deterministic +
            # content-preserving (d48/d60-clean: re-derives navigation, never fabricates
            # prose) + idempotent (a clean, complete-ToC doc is byte-identical). HTML only;
            # a markdown/text/csv deliverable is returned untouched by the function itself.
            if _is_html_ext and doc.strip():
                # s13/P1-report: single-title (thematic duplicate-tail) enforcement runs
                # ONLY on the sourced deep-research REPORT (where the B8a2 two-<h1> shell
                # occurs); a multi-page file-delivery doc keeps its legitimate per-page
                # <h1>s. The truncation trim inside reconcile is always safe and runs
                # regardless. The report path is identified by its fetched chain sources.
                reconciled = reconcile_doc_structure(
                    doc, single_title=bool(self._chain_sources)
                )
                if reconciled != doc and reconciled.strip():
                    w = await self.hook.invoke(
                        "file_write", path=written_path or out_path,
                        content=reconciled, append=False, overwrite=True,
                    )
                    if w.ok and isinstance(w.value, Mapping):
                        written_path = str(w.value.get("path") or written_path or out_path)
                    doc = reconciled
                    span.set_attribute("synthesis.doc_reconciled", True)

            if written_path is not None:
                span.set_attribute("synthesis.written_path", written_path)
            span.set_attribute("synthesis.chars", len(doc))

        return doc, written_path, finished

    async def review_and_fix(
        self,
        result: "SubAgentResult",
        reason: str,
        inputs: Optional[Mapping[str, Any]] = None,
    ) -> "SubAgentResult":
        """CODER=REVIEWER inline fix: the SAME spec reviews+corrects the output.

        The ``verifiable → done`` self-repair (d10): the node's verify gate
        rejected ``result`` for ``reason``; instead of re-launching the node (and
        WITHOUT re-running its tool or re-entering the DAG loop), the same spec
        body is re-used in a REVIEW posture to produce a corrected output. It is a
        single scoped phi call — and, exactly like :meth:`run`, the blocking phi
        round-trip is offloaded with ``asyncio.to_thread`` so the shared event
        loop keeps serving other nodes / SSE / ``/health`` while the review runs
        (the freeze-fix doctrine applies to the review pass too)."""
        inputs = inputs or {}
        # Re-use the SAME shaping-framed ruleset the producer ran with (d1), now in
        # a REVIEW posture — so the inline fix still shapes the FORM by the ruleset.
        shaping_system = self._compose_system()
        review_system = (
            (shaping_system + "\n\n" + _REVIEWER_FRAMING) if shaping_system else _REVIEWER_FRAMING
        )
        review_user = (
            self._compose_task(inputs, result.tool_value)
            + "\n\nYOUR PREVIOUS OUTPUT (to review):\n"
            + str(result.output or "")
            + "\n\nVERIFY GATE REJECTION REASON:\n"
            + str(reason or "(unspecified)")
            + "\n\nReturn ONLY the corrected output."
        )
        chain = Chain()
        chain.use(prompt_assembly())
        chain.use(call_stage(self.transport, **self._call_opts))
        ctx = Context(
            system=review_system or None,
            user=review_user,
            transport=self.transport,
        )
        # FREEZE FIX (decouple): same rationale as run() — offload the blocking
        # phi review call off the single asyncio event loop. TRACING (s6/b2):
        # re-attach the node span context inside the worker thread so the inline
        # review's phi span also nests under this node's span.
        ctx = await run_blocking_in_span(chain.run, ctx)
        return SubAgentResult(
            node_id=self.node.id,
            spec=self.node.primary_spec,
            specs=self.spec_names,
            output=ctx.raw_output,
            tool_used=result.tool_used,
            tool_value=result.tool_value,
            heal={"inline_review_fix": True, "reason": reason},
        )


@dataclass
class RuntimeResult:
    """The outcome of running a whole DAG in-process."""

    results: dict[str, SubAgentResult] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    launch_order: list[str] = field(default_factory=list)
    heal_logs: dict[str, dict[str, Any]] = field(default_factory=dict)
    states: dict[str, dict[str, Any]] = field(default_factory=dict)
    timed_out: bool = False
    replans_used: int = 0

    @property
    def ok(self) -> bool:
        return not self.failed

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "timed_out": self.timed_out,
            "replans_used": self.replans_used,
            "launch_order": self.launch_order,
            "results": {
                nid: {"spec": r.spec, "specs": list(r.specs), "tool_used": r.tool_used,
                      "output": r.output, "replanned": r.replanned}
                for nid, r in self.results.items()
            },
            "failed": dict(self.failed),
            "heal_logs": self.heal_logs,
            "states": self.states,
        }


class AgentRuntime:
    """Launches + tracks each DAG node as an in-process asyncio task (d2).

    The runtime owns the shared in-process machinery (event plane, tool hook,
    spec loader, transport) and drives a :class:`PlanDAG` to completion with an
    explicit per-node state machine, bounded concurrency, full task lifecycle
    (start/track/await/cancel), an idempotent result cache, and sub-graph
    re-plan self-heal.
    """

    def __init__(
        self,
        *,
        transport: Transport,
        loader: Optional[SpecLoader] = None,
        hook: Optional[ToolHook] = None,
        plane: Optional[EventPlane] = None,
        max_heals: int = 2,
        max_concurrency: Optional[int] = None,
        max_replans: int = 2,
        replanner: Optional[Replanner] = None,
        result_validator: Optional[ResultValidator] = None,
        verifier: Optional[NodeVerifier] = None,
        max_inline_fixes: int = 1,
        max_verdict_repairs: int = 2,
        tool_arg_emitter: Optional[ToolArgEmitter] = None,
        subagent_call_opts: Optional[Mapping[str, Any]] = None,
        collision_resolver: Optional[CollisionResolver] = None,
        lambda_registry: Optional[LambdaRegistry] = None,
        heal_router: Optional[HealRouter] = None,
        planner_reactor: Optional[PlannerReactor] = None,
        max_heal_retries: int = 1,
        execution: ExecutionMode = ExecutionMode.CONCURRENT,
        conversation_context: Optional[str] = None,
        read_search_max_fetch: int = 0,
        emit_article_notes: bool = False,
        chunked_read: bool = False,
        verify_lane: bool = False,
        fetched_char_budget: int = 2000,
        upstream_input_char_budget: int = 4000,
        writer_source_budget: Optional[int] = None,
        grower: Optional[Any] = None,
    ) -> None:
        self.transport = transport
        self.loader = loader
        # d13 SEARCH-THEN-READ (a4): when >0, every sub-agent FOLLOWS a web_search
        # through to web_fetch-ing the top real result URLs (and a research-role node
        # runs the full search→fetch→read executor), so the output carries ACTUAL
        # fetched article content, not a description of the search-results page. The
        # per-source fold budget bounds how much extracted text reaches the call.
        # 0 = OFF (back-compat: every pre-a4 path runs the single-tool-then-LLM node).
        self.read_search_max_fetch = max(0, int(read_search_max_fetch))
        # ARTICLE-NOTE + CHUNKED-READ LANES (s9/N4w, d65 served-route wiring): the two
        # grounding lanes N2 (per-article CONTROL note → the N4 decision node reads it)
        # and N3 (in-window map/reduce of long sources, no truncation). Plumbed through
        # the runtime exactly like ``read_search_max_fetch`` so the deep-research /
        # plan-chain gather can turn them ON for EVERY research-loop sub-agent it builds
        # (producer AND inline reviewer) — making the grounded structured path the LIVE
        # default on the served deep-research route, not a dark constructor flag. Default
        # False keeps every OTHER shape's runtime (acyclic short/headlines/csv) byte-
        # identical — the lanes only light up where a gather wires them (d65, no regression).
        self.emit_article_notes = bool(emit_article_notes)
        self.chunked_read = bool(chunked_read)
        # NO-FAB VERIFICATION LANE (s9/N5, d62/c15 part-e): plumbed exactly like the N2/N3
        # lanes so the deep-research / plan-chain gather can turn the reasoning
        # claim->source verify ON for EVERY sub-agent it builds (research producer +
        # synthesis + inline reviewer) — the served-route LIVE default (N6), not a dark
        # constructor flag. Default False keeps every other shape byte-identical.
        self.verify_lane = bool(verify_lane)
        self.fetched_char_budget = max(400, int(fetched_char_budget))
        # INTER-NODE CONTEXT (o4 fix): per-upstream-dependency prose budget handed to
        # every node's sub-agent (producer + inline reviewer) so a downstream
        # writer/synthesize node receives FULLER upstream content than the legacy
        # 800-char clip. Larger sensible default; final budget confirmed vs num_ctx
        # in s5/s6. None of the per-node call paths override it (single source).
        self.upstream_input_char_budget = max(200, int(upstream_input_char_budget))
        # WRITER SOURCE BUDGET (MSF/d89): the per-source writer feed budget handed to
        # every section sub-agent (producer + inline reviewer). None ⇒ the configured
        # ``resolve_writer_source_budget()`` (env RA_WRITER_SOURCE_BUDGET, default 12000).
        # Each section sub-agent sizes it to the num_ctx window before rendering. Only the
        # report write path consumes it (source_ids present) → default-safe elsewhere.
        self.writer_source_budget = (
            resolve_writer_source_budget()
            if writer_source_budget is None
            else max(120, int(writer_source_budget))
        )
        # CONVERSATION MEMORY (s5/a4): the bounded prior-turn context for THIS chat
        # run, handed to EVERY node's sub-agent (producer + inline reviewer) so the
        # answer-authoring node sees prior turns — closing the gap where only the
        # planner saw the history and paraphrased node tasks dropped the facts.
        # None => the runtime is memoryless exactly as before (every non-chat path).
        self.conversation_context = (conversation_context or "").strip() or None
        # OVERALL GOAL (d39): the verbatim user request the running plan serves. It
        # travels ON the PlanDAG (``dag.goal``) and is stashed here at :meth:`run`
        # time, then handed to EVERY node's sub-agent (producer + inline reviewer) so
        # each worker grounds in the real objective, not just its paraphrased task.
        # None until a goal-carrying DAG is run (and for every goal-less caller).
        self._overall_goal: Optional[str] = None
        self.hook = hook
        self.plane = plane or (hook.plane if hook is not None else EventPlane())
        self.max_heals = max_heals
        self.max_concurrency = max_concurrency
        # EXECUTION DISCIPLINE (s3/b1, blueprint §2a): how this runtime DISPATCHES a
        # shape's ready nodes — CONCURRENT (modular-parallel: launch the whole ready
        # wave at once, the legacy behaviour and the DEFAULT) or SEQUENTIAL (linear:
        # one node in flight at a time, strict single-file order). The decision is a
        # pure-python port of eda-base3's readiness gate / dispatch FSM and lives in
        # :mod:`agent_runtime.scheduler`; this field only selects the mode it runs in.
        # Defaulting to CONCURRENT keeps every existing acyclic/deep-research path
        # byte-compatible (the deep-research executor never uses this loop).
        self.execution = execution
        self.max_replans = max_replans
        self.replanner = replanner
        self.result_validator = result_validator
        # Per-node VERIFY GATE + its inline CODER=REVIEWER fix budget (Stage-B run
        # engine). ``verifier`` None = the gate trivially passes (every node still
        # traverses VERIFIABLE). ``max_inline_fixes`` bounds same-spec inline
        # corrections before a gate failure falls through to re-plan / FAILED.
        self.verifier = verifier
        self.max_inline_fixes = max_inline_fixes
        # JUDGMENT-ROLE verdict-repair budget (a3, the generic b3 hardening): how
        # many times a role node re-emits when its enum verdict is null/invalid.
        self.max_verdict_repairs = max_verdict_repairs
        # Optional schema-constrained tool-arg emitter handed to every sub-agent
        # (s8/b1 phi hardening). None = sub-agents use node args verbatim.
        self.tool_arg_emitter = tool_arg_emitter
        # Options forwarded to each sub-agent's phi call (e.g. a higher token
        # budget so writer nodes produce detailed reports). None = defaults.
        self.subagent_call_opts = dict(subagent_call_opts or {})
        # DAG SPEC-COLLISION escalation (d11): the awaitable resolver consulted when
        # a node's 2+ specs GENUINELY conflict (same shaping axis, different value).
        # None = no escalation channel — a detected conflict surfaces as a clean
        # node FAILURE (CollisionUnresolved) rather than a silent auto-pick.
        self.collision_resolver = collision_resolver
        # AGENT-AUTHORS-LAMBDA, COMPLETED STRUCTURALLY (s9/a2, d15): the optional
        # shared :class:`LambdaRegistry` the UI's read-only live-subscriptions
        # surface reads. When wired, EVERY ``run()`` AUTO-CREATES a genuine
        # observability lambda (a reactive subscription) that observes THIS run's
        # node-lifecycle on ``self.plane`` and is recorded on this registry — so
        # the lambda tab populates on every live run regardless of how small the
        # model is (the agent is the author, the user only observes). None = no
        # auto-lambda (the legacy behaviour; the offline/stub paths leave it off).
        self.lambda_registry = lambda_registry
        # REACTIVE SELF-HEAL (b4, blueprint §2e, d1): when wired, a node FAILURE is
        # ROUTED to the planner's heal logic — ``heal_router.route`` asks the planner
        # for a structured heal DECISION (``Planner.heal_decision`` enum
        # retry|pivot|extend|abort) and the runtime ENACTS the routed action (retry
        # re-dispatch / replan_subgraph / abort-surface). The planner owns the
        # control-flow decision; the runtime owns the (state-mutating) enactment; the
        # registered EventPlane/LambdaRegistry rule only OBSERVES the routing. ``None``
        # keeps the legacy path: a failure goes straight to the unconditional
        # sub-graph re-plan (byte-compatible with every pre-b4 caller/test).
        self.heal_router = heal_router
        # EVENT-DRIVEN PLANNER REACTION (P2.2, d129.2): when wired, a node FAILURE is
        # routed to the planner's heal decision through a real EventPlane SUBSCRIBER
        # (:class:`PlannerReactor`) instead of the synchronous in-call ``heal_router``.
        # The runtime emits the failure event, the reactor (subscribed) reacts and
        # resolves the per-node decision, and the runtime enacts it — so the planner
        # genuinely subscribes-and-reacts (recover a parallel node the instant it
        # fails, before the join) rather than being called in the runtime's own stack.
        # None keeps the Phase-1 synchronous heal path (byte-compatible). The reactor
        # also surfaces a worker CLARIFICATION while sibling workers keep running.
        self.planner_reactor = planner_reactor
        # Bounds the coarse re-dispatch a ``retry`` decision triggers, per node.
        self.max_heal_retries = max_heal_retries
        # Per-node count of coarse re-dispatches already spent (the retry budget).
        self._heal_retries: dict[str, int] = {}
        # The auto-registered observe-only self-heal rule's sub_id (per run).
        self._heal_rule_sub_id: Optional[str] = None
        # Per-node resolved+cleaned scope cache. A node's scopes are resolved (and
        # any collision escalated) EXACTLY ONCE; the producer, every self-heal
        # retry, and the inline coder=reviewer all read this same composition — so
        # escalation never double-fires and the reviewer matches the producer.
        self._resolved_scopes: dict[str, list[ScopedSpec]] = {}
        # Tracked, observable set of live/finished node tasks (d2: in-process).
        self.tracked: dict[str, asyncio.Task] = {}
        # TRACING (s6/b2): the live per-node OTel span, keyed by node id. A node's
        # span is opened when it transitions to RUNNING (a child of the "agent.run"
        # span) and stays OPEN across the worker task AND the verify gate — which
        # run in different methods — so it can record the full pending -> in-progress
        # -> verifiable -> done lifecycle as span EVENTS. It is ended (and removed)
        # exactly once at the node's terminal transition (done/failed/cancelled).
        self._node_spans: dict[str, trace.Span] = {}
        # Per-node explicit state machine + idempotent successful-result cache.
        self.states: dict[str, NodeState] = {}
        self._cache: dict[str, SubAgentResult] = {}
        self._heal_logs: dict[str, dict[str, Any]] = {}
        self._launch_seq = 0
        self._replans_used = 0
        self._sem: Optional[asyncio.Semaphore] = None
        # GROWABLE-DAG GROWER (P2.5b, d134/d135): the injected dependency that relaxes the
        # ONE invariant "node set fixed at unroll time". When a ``growable`` DAG is run and a
        # grower is wired, the drive loop GROWS the DAG round-by-round on the decision node's
        # gap-driven branches (:meth:`_drive_growable`) — reproducing run_research_tree's
        # iterative breadth in the GENERIC engine. None (every other path) → the legacy
        # single-pass drive, byte-identical. The grower is duck-typed: it exposes
        # ``max_layers`` and ``async grow(dag, cache, layer) -> (new_nodes, stop_reason)``.
        self._grower = grower
        # The number of research layers the growable drive actually ran (seed=1 + grown),
        # for the run trace. 0 until a growable drive completes.
        self._grow_layers = 0

    # ------------------------------------------------------------------ #
    # small helpers over the shared state
    # ------------------------------------------------------------------ #
    def _state(self, node_id: str) -> NodeState:
        st = self.states.get(node_id)
        if st is None:
            st = NodeState(node_id=node_id)
            self.states[node_id] = st
        return st

    def _done_ids(self) -> set[str]:
        return set(self._cache)

    def _is_blocking(self, node_id: str) -> bool:
        st = self.states.get(node_id)
        return st is not None and st.status in (
            NodeStatus.FAILED, NodeStatus.SKIPPED, NodeStatus.CANCELLED
        )

    async def _emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        await self.plane.publish(kind, dict(payload), source="agent_runtime")

    # ------------------------------------------------------------------ #
    # per-node OTel span lifecycle (s6/b2) — opened at RUNNING, carried across
    # the worker task + verify gate, ended once at the terminal transition.
    # ------------------------------------------------------------------ #
    def _start_node_span(self, node: PlanNode) -> "trace.Span":
        """Open the per-node child span (parent = the active "agent.run" span).

        Started while the DAG driver runs UNDER the "agent.run" span, so the new
        span's parent is "agent.run" by context (one run trace). Carries the node
        id, task, and the ordered composed spec names; records the lifecycle entry
        events ``pending`` -> ``in-progress`` (the node WAS pending; it is now
        launched/running). Stored in :attr:`_node_spans` so the verify gate and the
        terminal transition — both OUTSIDE the worker task — can add later events
        and end it."""
        tracer = get_tracer("agent_runtime.runtime")
        span = tracer.start_span("agent.node")
        span.set_attribute("node.id", node.id)
        span.set_attribute("node.task", str(node.task)[:1000])
        specs = list(node.effective_specs)
        if specs:
            span.set_attribute("node.spec_names", specs)
        if node.tool:
            span.set_attribute("node.tool", str(node.tool))
        span.add_event("pending")
        span.add_event("in-progress")
        self._node_spans[node.id] = span
        return span

    def _node_span_event(self, node_id: str, name: str) -> None:
        """Record a lifecycle event on a node's live span (no-op if absent)."""
        span = self._node_spans.get(node_id)
        if span is not None and span.is_recording():
            span.add_event(name)

    def _end_node_span(
        self, node_id: str, *, event: str, error: Optional[str] = None
    ) -> None:
        """Record the terminal lifecycle event and END the node's span (once).

        ``event`` is the terminal lifecycle name (``done`` / ``failed`` /
        ``cancelled`` / ``skipped``). On an error path the span status is set to
        ERROR with the reason; otherwise OK. Popped from :attr:`_node_spans` so a
        second call is a safe no-op."""
        span = self._node_spans.pop(node_id, None)
        if span is None:
            return
        span.add_event(event)
        if error is not None:
            span.set_attribute("node.error", str(error)[:600])
            span.set_status(Status(StatusCode.ERROR, str(error)[:300]))
        else:
            span.set_status(Status(StatusCode.OK))
        span.end()

    def _resolve_scopes(self, node: PlanNode) -> Optional[list[ScopedSpec]]:
        """Resolve a node's 1+ spec NAMES into ordered ``ScopedSpec`` bodies (d2/d11).

        Resolution happens HERE — in the runtime's trust boundary (it owns the
        loader) — and the loader is NOT handed to the sub-agent (by-construction
        d10). Order follows ``node.effective_specs`` (the planner's list), which
        is exactly the composition order. Returns ``None`` for a bare node or when
        there is no loader (a single empty/None spec stays a bare step)."""
        if not (node.effective_specs and self.loader is not None):
            return None
        return [ScopedSpec.resolve(self.loader, name) for name in node.effective_specs]

    async def _scopes_for(self, node: PlanNode) -> Optional[list[ScopedSpec]]:
        """Resolve a node's scopes, ESCALATING a genuine spec collision (d11).

        The single seam between a4's autonomous composition and a5's HITL escalation
        — and it runs EXACTLY ONCE per node (cached). For a node with 2+ specs it
        detects a genuine conflict (same shaping axis, different value); on a
        conflict it PAUSES the node by ``await``-ing ``collision_resolver`` (the node
        coroutine suspends here, staying RUNNING/in-flight — it is not failed) and,
        once the user's resolution arrives, reorders/drops the scopes to the chosen
        composition. A compatible 2-spec node (different axes, or the same value) is
        composed autonomously with NO escalation. Finally directive tags are stripped
        so the model sees clean shaping prose. Returns the resolved+cleaned scopes
        (or ``None`` for a bare node). With no resolver wired, a genuine conflict
        raises :class:`CollisionUnresolved` — a clean failure, never a silent pick."""
        cached = self._resolved_scopes.get(node.id)
        if cached is not None:
            return cached or None
        raw = self._resolve_scopes(node)
        if raw is None:
            self._resolved_scopes[node.id] = []
            return None
        final = raw
        if len(raw) >= 2:
            collision = detect_collision(node.id, raw)
            if collision is not None:
                if self.collision_resolver is None:
                    # No escalation channel for a genuine conflict → fail cleanly.
                    raise CollisionUnresolved(node.id, collision)
                await self._emit(EVENT_NODE_COLLISION, collision.as_dict())
                resolution = await self.collision_resolver(collision)
                final = apply_resolution(raw, resolution)
                await self._emit(
                    EVENT_NODE_COLLISION_RESOLVED,
                    {"node_id": node.id, "challenge": collision.challenge,
                     "order": [s.name for s in final], "note": resolution.note},
                )
        # Strip directive tags so the produce SYSTEM carries clean shaping prose,
        # not the collision-detection metadata (a no-op for untagged bodies, so the
        # a4 composition is byte-identical for them).
        cleaned = [ScopedSpec.of(s.name, strip_directives(s.body)) for s in final]
        self._resolved_scopes[node.id] = cleaned
        return cleaned

    # ------------------------------------------------------------------ #
    # single-node execution (idempotent + node-level self-heal)
    # ------------------------------------------------------------------ #
    async def _run_node(self, node: PlanNode) -> SubAgentResult:
        """Run ONE node with node-level self-heal — idempotent against the cache.

        If the node's successful result is already cached, it is returned WITHOUT
        re-executing (``cache_hit``, ``attempts`` unchanged) — the no-double-
        execution guarantee. Otherwise each real attempt increments ``attempts``
        and the logic is wrapped in :class:`SelfHeal`."""
        st = self._state(node.id)
        if node.id in self._cache:
            st.cache_hit = True
            return self._cache[node.id]

        inputs = {
            dep: self._cache[dep].output for dep in node.depends_on if dep in self._cache
        }
        # a2-recipe (s7/a2): the RAW upstream tool values (e.g. web_search's results
        # dict) so the tool-arg emitter can ground a derived arg in real data — the
        # produced prose alone (``inputs``) often drops the result URLs.
        upstream_tool_values = {
            dep: self._cache[dep].tool_value
            for dep in node.depends_on
            if dep in self._cache and self._cache[dep].tool_value is not None
        }
        heal_log = HealLog(label=node.id)

        # PLAN-CHAINING (c1b/d49.4): for a WRITER node, decide whether it CONTINUES a
        # file an upstream writer already started (a multi-page chain) and whether it
        # is the FINAL page. ``chain_continue_path`` is the upstream writer's on-disk
        # path (this node appends the next page); ``chain_is_final`` is False when a
        # downstream node is itself a writer (more pages follow → don't close yet). For
        # any non-writer node, or a lone single-file writer (upstream=research, no
        # downstream writer), this stays (None, True) → byte-identical pre-c1b path.
        chain_continue_path: Optional[str] = None
        chain_is_final = True
        writer_ids = getattr(self, "_writer_ids", set())
        if node.id in writer_ids:
            for dep in node.depends_on:
                cached = self._cache.get(dep)
                if (
                    dep in writer_ids
                    and cached is not None
                    and isinstance(cached.tool_value, Mapping)
                    and cached.tool_value.get("path")
                ):
                    chain_continue_path = str(cached.tool_value["path"])
                    break
            dependents = getattr(self, "_dependent_ids", {}).get(node.id, set())
            chain_is_final = not any(d in writer_ids for d in dependents)

        async def logic() -> SubAgentResult:
            st.attempts += 1  # only a REAL execution counts (idempotency proof)
            # Resolve the node's 1+ spec bodies in the runtime's trust boundary and
            # hand the sub-agent only those scopes (by-construction d10). N specs
            # compose into one layered system at produce time (d2/d11); a GENUINE
            # spec collision is escalated to the HITL gate here (d11) — the node
            # pauses in-flight until the user resolves it, then proceeds.
            scopes = await self._scopes_for(node)
            agent = SubAgent(
                node,
                transport=self.transport,
                scopes=scopes,
                hook=self.hook,
                result_validator=self.result_validator,
                tool_arg_emitter=self.tool_arg_emitter,
                # CARRY-FORWARD FIX (s8/b1, a2-recipe): the runtime stored
                # ``subagent_call_opts`` (e.g. num_predict=1400, think=False) but
                # never handed it to the sub-agent, so writer nodes silently dropped
                # their token budget and ran on transport defaults. Thread it through
                # so the node's phi call honours the configured budget/knobs.
                call_opts=self.subagent_call_opts,
                conversation_context=self.conversation_context,
                # OVERALL GOAL (d39): the verbatim user request, so the producer node
                # grounds its paraphrased task in the real objective.
                overall_goal=self._overall_goal,
                upstream_tool_values=upstream_tool_values,
                max_verdict_repairs=self.max_verdict_repairs,
                read_search_max_fetch=self.read_search_max_fetch,
                # d65 served-route wiring: the note + chunked-read grounding lanes the
                # deep-research / plan-chain gather turned ON flow to the producer.
                emit_article_notes=self.emit_article_notes,
                chunked_read=self.chunked_read,
                # READ-SIDE relevance embedder (d109): when chunked-read is ON, the research
                # read ranks a long source's chunks by MiniLM similarity to the node's
                # sub-question (single in-window read) instead of the 75-chunk map/reduce.
                read_embedder=_load_read_embedder() if self.chunked_read else None,
                # N5: the reasoning no-fab verify lane (research gather-more + deliverable
                # claim->source check) runs on the producer when the gather turned it ON.
                verify_lane=self.verify_lane,
                fetched_char_budget=self.fetched_char_budget,
                upstream_input_char_budget=self.upstream_input_char_budget,
                # WRITER SOURCE BUDGET (MSF/d89): the producer section node is fed its
                # assigned sources at the raised, window-sized budget (vs the legacy 700).
                writer_source_budget=self.writer_source_budget,
                # PLAN-CHAINING (c1b): the multi-page continuation contract computed
                # above from the DAG topology (no-op for a single-file deliverable).
                chain_continue_path=chain_continue_path,
                chain_is_final=chain_is_final,
                # SOURCE-SCOPING (s9/c13, d56): the run's global fetched-source list,
                # set on the runtime by the per-section write phase. A section node
                # with ``source_ids`` is fed only its assigned subset near the cursor.
                chain_sources=getattr(self, "chain_sources", None),
            )
            return await agent.run(inputs)

        healer = SelfHeal(max_heals=self.max_heals)
        res = await healer.run(logic, label=node.id, log=heal_log)
        res.heal = heal_log.as_dict()
        self._heal_logs[node.id] = heal_log.as_dict()
        if heal_log.healed:
            st.healed = True
            await self._emit(EVENT_NODE_HEALED, {"node_id": node.id, "heal": heal_log.as_dict()})
        self._cache[node.id] = res
        return res

    async def _node_task(
        self, node: PlanNode, span: Optional["trace.Span"] = None
    ) -> SubAgentResult:
        """The tracked task body — bounded by the concurrency semaphore.

        TRACING (s6/b2): the node's span (created by the driver under "agent.run")
        is made the CURRENT otel context for the whole node execution, so every
        phi call inside :class:`SubAgent` (offloaded to a worker thread via
        ``run_blocking_in_span``) nests under THIS node span — the cross-thread
        propagation that keeps the run trace one tree. The attach is scoped to this
        asyncio task's copied context, so concurrent nodes never clobber each
        other's current span; it is detached in ``finally`` (the span itself stays
        OPEN — the driver ends it at the terminal transition)."""
        if span is None:
            if self._sem is not None:
                async with self._sem:
                    return await self._run_node(node)
            return await self._run_node(node)
        token = otel_context.attach(trace.set_span_in_context(span))
        try:
            if self._sem is not None:
                async with self._sem:
                    return await self._run_node(node)
            return await self._run_node(node)
        finally:
            otel_context.detach(token)

    # ------------------------------------------------------------------ #
    # per-node VERIFY GATE + CODER=REVIEWER inline fix (verifiable → done)
    # ------------------------------------------------------------------ #
    def _build_subagent(self, node: PlanNode) -> SubAgent:
        """Build a sub-agent scoped to this node's 1+ specs (by-construction d10).

        Mirrors the scope resolution in :meth:`_run_node`'s ``logic`` so the
        inline reviewer re-uses the SAME composed spec stack the producer ran with
        — the CODER=REVIEWER guarantee, now over an N-spec composition (d2/d11). The
        producer already resolved+escalated this node's scopes (cached), so the
        reviewer reads that exact resolved composition — the HITL resolution is NOT
        re-asked, and the reviewer never diverges from what was produced."""
        scopes = self._resolved_scopes.get(node.id)
        if scopes is None:  # defensive: producer always populates this first
            scopes = self._resolve_scopes(node)
        else:
            scopes = scopes or None
        return SubAgent(
            node,
            transport=self.transport,
            scopes=scopes,
            hook=self.hook,
            result_validator=self.result_validator,
            tool_arg_emitter=self.tool_arg_emitter,
            # CARRY-FORWARD FIX (s8/b1): the inline CODER=REVIEWER must run with the
            # SAME call opts (budget/think) as the producer, so it does not review a
            # full-budget output with a default-budget transport call.
            call_opts=self.subagent_call_opts,
            # Same prior-turn grounding the producer had (s5/a4), so the inline
            # reviewer corrects against the conversation, not a memoryless reading.
            conversation_context=self.conversation_context,
            # OVERALL GOAL (d39): the inline reviewer must compose the SAME goal-led
            # user turn the producer did, so it corrects against the real objective.
            overall_goal=self._overall_goal,
            # d13 (a4): the inline reviewer of a research node must read the SAME
            # fetched-article context the producer did, so it corrects against real
            # sources, not a memoryless re-read.
            read_search_max_fetch=self.read_search_max_fetch,
            # d65: the inline CODER=REVIEWER of a research node must run the SAME note +
            # chunked-read lanes the producer did, so it corrects against the same
            # grounded state (notes recorded, long sources mapped in-window), not a
            # truncated re-read.
            emit_article_notes=self.emit_article_notes,
            chunked_read=self.chunked_read,
            # READ-SIDE relevance embedder (d109): the inline CODER=REVIEWER of a research
            # node reads via the SAME relevance-select path the producer did, so it corrects
            # against the same top-ranked passages, not a re-read.
            read_embedder=_load_read_embedder() if self.chunked_read else None,
            # N5: the inline CODER=REVIEWER runs the SAME no-fab verify lane as the
            # producer so it corrects against the same grounded state, not a memoryless re-read.
            verify_lane=self.verify_lane,
            fetched_char_budget=self.fetched_char_budget,
            # INTER-NODE CONTEXT (o4 fix): the inline reviewer must compose the SAME
            # fuller-upstream user turn the producer did, so it corrects against the
            # real upstream sources, not the legacy 800-char clip.
            upstream_input_char_budget=self.upstream_input_char_budget,
            # WRITER SOURCE BUDGET (MSF/d89): the inline CODER=REVIEWER reviews against the
            # SAME raised, window-sized per-source feed the producer wrote from.
            writer_source_budget=self.writer_source_budget,
        )

    async def _run_verifier(
        self, node: PlanNode, result: SubAgentResult
    ) -> tuple[bool, Optional[str]]:
        """Run the per-node verify gate; normalise its verdict to ``(ok, reason)``.

        No verifier → the gate trivially passes. The verifier may return a bare
        bool or an ``(ok, reason)`` tuple, sync or awaitable."""
        if self.verifier is None:
            return True, None
        verdict = self.verifier(node, result)
        if hasattr(verdict, "__await__"):
            verdict = await verdict
        if isinstance(verdict, tuple):
            ok, reason = (verdict + (None,))[:2]
            return bool(ok), (reason if not ok else None)
        ok = bool(verdict)
        return ok, (None if ok else "verify gate rejected the node output")

    async def _verify_and_finalize(
        self, node: PlanNode, result: SubAgentResult
    ) -> tuple[bool, Optional[str]]:
        """Drive a produced node through the verify gate (``verifiable → done``).

        Transitions the node into VERIFIABLE (observable on the plane), runs the
        per-node gate, and — on a gate REJECTION — applies up to
        ``max_inline_fixes`` CODER=REVIEWER inline corrections: the SAME spec
        reviews+fixes the output (one scoped phi call each, phi offloaded off the
        loop), the corrected output REPLACES the node's cached result, and the
        gate is re-run — all WITHOUT re-launching the node or re-entering the DAG
        loop (no tool re-call, no upstream re-run). Returns ``(passed, reason)``;
        the caller crosses VERIFIABLE → DONE on pass, or tries re-plan / FAILED on
        a final fail. Idempotent w.r.t. the cache: the cache always holds the
        latest (corrected) output so downstream dependents read the fixed value."""
        st = self._state(node.id)
        # Idempotent on re-entry: a ``retry`` heal re-dispatch (b4) re-runs produce
        # and re-enters this gate while the node is ALREADY VERIFIABLE — and
        # VERIFIABLE→VERIFIABLE is not a legal transition. Only cross RUNNING→
        # VERIFIABLE the first time; a re-verify keeps the node VERIFIABLE.
        if st.status != NodeStatus.VERIFIABLE:
            st.transition(NodeStatus.VERIFIABLE)
        # TRACING (s6/b2): the third lifecycle event on the node's span. The span
        # is still open (the driver ends it at the terminal transition).
        self._node_span_event(node.id, "verifiable")
        await self._emit(
            EVENT_NODE_VERIFIABLE,
            {"node_id": node.id, "spec": node.primary_spec, "specs": list(node.effective_specs)},
        )
        ok, reason = await self._run_verifier(node, result)
        if ok:
            st.verified = True
            return True, None

        # Gate rejected → inline CODER=REVIEWER fix loop (same spec, no re-loop).
        inputs = {
            dep: self._cache[dep].output
            for dep in node.depends_on
            if dep in self._cache
        }
        # TRACING (s6/b2): this gate runs in the DRIVER context (where "agent.run"
        # is current), not the node task. Re-attach the node's span as current for
        # the inline review so the review's phi span (offloaded to a worker thread
        # via run_blocking_in_span) ALSO nests under THIS node's span, exactly like
        # the produce-step phi span — not under "agent.run".
        node_span = self._node_spans.get(node.id)
        token = (
            otel_context.attach(trace.set_span_in_context(node_span))
            if node_span is not None
            else None
        )
        try:
            agent = self._build_subagent(node)
            attempt = 0
            while not ok and attempt < self.max_inline_fixes:
                attempt += 1
                await self._emit(
                    EVENT_NODE_REVIEW,
                    {"node_id": node.id, "reason": reason, "attempt": attempt},
                )
                fixed = await agent.review_and_fix(result, reason or "", inputs)
                # The corrected output REPLACES the node result in the shared cache —
                # downstream dependents (and the aggregated result) see the fix, and
                # the DAG loop is NOT restarted.
                self._cache[node.id] = fixed
                result = fixed
                st.inline_fixes = attempt
                ok, reason = await self._run_verifier(node, fixed)
                if ok:
                    st.verified = True
                    st.inline_fixed = True
                    await self._emit(
                        EVENT_NODE_INLINE_FIXED,
                        {"node_id": node.id, "attempt": attempt},
                    )
                    return True, None

            await self._emit(
                EVENT_NODE_VERIFY_FAILED, {"node_id": node.id, "reason": reason}
            )
            return False, reason
        finally:
            if token is not None:
                otel_context.detach(token)

    # ------------------------------------------------------------------ #
    # sub-graph re-plan self-heal
    # ------------------------------------------------------------------ #
    def _namespace_subdag(self, subdag: PlanDAG, failed_id: str) -> PlanDAG:
        """Return ``subdag`` with its node ids prefixed so they cannot collide
        with the parent DAG's node states/cache (re-plan isolation).

        Intra-subdag ``depends_on`` edges are remapped to the prefixed ids; any
        edge to an id OUTSIDE the subdag (e.g. an already-completed parent node)
        is dropped — a corrective sub-graph for one step is self-contained, and a
        parent node it referenced is already satisfied via the shared cache, so
        the edge is redundant (and keeping it would fail DAG validation, which
        requires every ``depends_on`` to resolve within the graph)."""
        prefix = f"{failed_id}~rp{self._replans_used}~"
        idmap = {n.id: f"{prefix}{n.id}" for n in subdag.nodes}
        return PlanDAG(
            nodes=[
                PlanNode(
                    id=idmap[n.id],
                    task=n.task,
                    spec=n.spec,
                    specs=n.specs,  # carry the N-spec composition through re-plan
                    depends_on=tuple(idmap[d] for d in n.depends_on if d in idmap),
                    tool=n.tool,
                    tool_args=n.tool_args,
                    role=n.role,  # carry the node role through re-plan
                    needs_spec=n.needs_spec,  # carry the missing-spec signal too
                )
                for n in subdag.nodes
            ],
            rationale=subdag.rationale,
            shape=subdag.shape,
        )

    async def _try_replan(self, node: PlanNode, error: str) -> bool:
        """Re-derive + run a minimal corrective sub-graph for a failed node.

        Returns True if the re-plan recovered the node (its result is now cached
        under ``node.id``); False to surface the failure (no replanner, budget
        exhausted, or the corrective sub-graph also failed). Idempotent: the
        sub-graph drive shares the cache, so already-succeeded nodes never re-run.
        """
        if self.replanner is None or self._replans_used >= self.max_replans:
            return False
        self._replans_used += 1
        completed = sorted(self._done_ids())
        await self._emit(
            EVENT_NODE_REPLANNED,
            {"node_id": node.id, "error": error, "replan_attempt": self._replans_used},
        )
        try:
            subdag = await self.replanner(node, error, completed)
        except MalformedOutputError:
            return False  # the re-plan itself could not be derived → give up
        # NAMESPACE the corrective sub-graph's node ids before driving it. The
        # re-planner is a fresh phi call that naturally re-emits ids like 'n1'
        # (the node-schema example), which would COLLIDE with the parent DAG's
        # still-live node states/cache — re-transitioning a parent node that is
        # currently RUNNING (IllegalTransition: running → running). Prefixing the
        # subdag's own ids isolates its state machine from the parent's while the
        # shared cache/idempotency still holds (the parent ids the subdag may
        # depend on stay un-prefixed, so an already-done parent node still
        # satisfies the edge; intra-subdag edges are remapped consistently).
        subdag = self._namespace_subdag(subdag, node.id)
        # Run the corrective sub-graph in-process, sharing cache + states.
        await self._drive_dag(subdag)
        if not all(n.id in self._cache for n in subdag.nodes):
            return False  # the corrective sub-graph itself failed → surface
        # Map the sub-graph's terminal output back onto the failed node id so
        # downstream dependents in the parent DAG can proceed.
        terminal = subdag.topo_order()[-1]
        terminal_res = self._cache[terminal.id]
        self._cache[node.id] = SubAgentResult(
            node_id=node.id,
            spec=node.primary_spec,
            specs=tuple(node.effective_specs),
            output=terminal_res.output,
            tool_used=terminal_res.tool_used,
            tool_value=terminal_res.tool_value,
            heal={"replanned_via": [n.id for n in subdag.nodes]},
            replanned=True,
        )
        self._state(node.id).replanned = True
        return True

    # ------------------------------------------------------------------ #
    # reactive self-heal routing (b4, §2e, d1) — the failure seam
    # ------------------------------------------------------------------ #
    async def _heal_failed_node(self, node: PlanNode, error: str) -> bool:
        """ROUTE a node FAILURE through the planner's heal decision, then ENACT it.

        Returns True if the node was recovered (its result is now cached under
        ``node.id``); False to surface the failure. This is the single failure seam
        the DAG driver calls for both a node-level-heal-exhausted exception and a
        verify-gate-final-fail.

        Legacy path (``heal_router`` not wired): an unconditional sub-graph re-plan
        — byte-identical to the pre-b4 behaviour, so every existing caller/test is
        unchanged. Reactive path (router wired): publish the failure on the plane
        (so the observe-only heal RULE fires), ask the PLANNER for the heal decision
        (``retry|pivot|extend|abort``), record the routed action, then ENACT it:

          * retry  → idempotent re-dispatch of the SAME node (done nodes preserved);
          * replan → ``_try_replan`` corrective sub-DAG (pivot/extend);
          * abort  → surface the failure (no recovery).

        The planner owns the DECISION (d1); the runtime owns the state-mutating
        enactment.

        EVENT-DRIVEN PATH (P2.2): when a ``planner_reactor`` is wired the heal DECISION
        is obtained by EMITTING the failure event and AWAITING the reactor's reaction
        (the planner subscribes-and-reacts) rather than calling ``heal_router.route``
        synchronously in this stack. The enactment below is identical either way."""
        if self.heal_router is None and self.planner_reactor is None:
            return await self._try_replan(node, error)
        attempt = self._heal_retries.get(node.id, 0)
        completed = sorted(self._done_ids())
        if self.planner_reactor is not None:
            # Register the waiter BEFORE emitting, so the subscribed reactor has
            # somewhere to deliver its decision. The SAME failure event still drives
            # the advisory observe-only rule; now it also drives the planner reaction.
            self.planner_reactor.expect(node.id)
            await self._emit(
                EVENT_NODE_FAILURE_DETECTED,
                {"node_id": node.id, "task": node.task, "error": error,
                 "attempt": attempt, "completed": completed},
            )
            route = await self.planner_reactor.await_route(node.id)
        else:
            # The node-FAILURE event hits the plane → the registered advisory heal rule
            # observes it (reactive routing). This is the event the rule "routes".
            await self._emit(
                EVENT_NODE_FAILURE_DETECTED,
                {"node_id": node.id, "error": error, "attempt": attempt},
            )
            route = await self.heal_router.route(
                node.task, error, attempt=attempt, completed=completed
            )
        # Record the planner's routed decision on the plane (observable; the rule
        # also watches this kind).
        await self._emit(EVENT_HEAL_ROUTED, {"node_id": node.id, **route.as_dict()})
        if route.is_abort:
            return False  # unrecoverable → surface to the user/neuron
        if route.is_retry:
            self._heal_retries[node.id] = attempt + 1
            if await self._retry_dispatch(node):
                return True
            # The coarse re-dispatch also failed → escalate to a corrective replan
            # rather than surfacing a transient as terminal.
            return await self._try_replan(node, error)
        # pivot / extend (and any unmapped action) → corrective sub-graph re-plan.
        return await self._try_replan(node, error)

    async def _retry_dispatch(self, node: PlanNode) -> bool:
        """Idempotent re-dispatch of the SAME failed node (the ``retry`` route).

        Clears ONLY this node's cached (failed/partial) result so it re-executes;
        every already-DONE upstream node stays cached and is NEVER re-run — the
        "keep done nodes, replace only the failed node" guarantee. Re-runs the
        produce step (node-level self-heal applies again) and re-enters the verify
        gate. Returns True if the node now passes; False (a raised re-run, or a gate
        that still fails) so the caller can escalate to a replan."""
        self._cache.pop(node.id, None)
        try:
            res = await self._run_node(node)
        except Exception:  # noqa: BLE001 — re-dispatch exhausted → caller escalates
            return False
        passed, _ = await self._verify_and_finalize(node, res)
        return passed

    # ------------------------------------------------------------------ #
    # the DAG driver (re-entrant: also runs re-plan sub-graphs)
    # ------------------------------------------------------------------ #
    async def _drive_dag(self, dag: PlanDAG) -> None:
        """Drive one DAG to completion against the shared state/cache.

        Launches every ready node concurrently as a tracked asyncio task,
        honouring ``depends_on``; records each into the shared state machine; on
        a node failure attempts sub-graph re-plan before marking it FAILED; and
        promptly SKIPs nodes blocked by an upstream failure."""
        dag.validate()
        node_by_id = dag.by_id
        remaining = {n.id: n for n in dag.nodes if n.id not in self._cache}
        # Nodes already in cache are DONE for this drive (idempotent short-circuit).
        for n in dag.nodes:
            if n.id in self._cache:
                st = self._state(n.id)
                st.cache_hit = st.cache_hit or st.attempts == 0
                if st.status not in (NodeStatus.DONE,):
                    # PENDING → mark done via RUNNING-less path is illegal; set directly
                    st.status = NodeStatus.DONE
        running: dict[asyncio.Task, str] = {}

        while remaining or running:
            # SHAPE-FAITHFUL DISPATCH (s3/b1): the deterministic, model-independent
            # scheduler (ported from eda-base3's `_first_ready_action` /
            # `plan_next_action`) decides WHICH ready nodes may launch this turn,
            # honouring the execution discipline — CONCURRENT launches the whole
            # ready wave (modular-parallel, the legacy behaviour); SEQUENTIAL launches
            # only `first_ready_action` and only when nothing is in flight (linear,
            # strict single-file). `running.values()` are the in-flight node ids, so a
            # node already launched is never re-dispatched.
            dispatch = next_dispatch(
                dag,
                self._done_ids(),
                running.values(),
                mode=self.execution,
            )
            for node in dispatch.nodes:
                if node.id not in remaining:
                    continue
                blocked = [d for d in node.depends_on if self._is_blocking(d)]
                if blocked:
                    st = self._state(node.id)
                    st.transition(NodeStatus.SKIPPED)
                    st.error = f"skipped: upstream blocked {blocked}"
                    await self._emit(EVENT_NODE_SKIPPED, {"node_id": node.id, "blocked_by": blocked})
                    del remaining[node.id]
                    continue
                st = self._state(node.id)
                self._launch_seq += 1
                st.launch_seq = self._launch_seq
                st.transition(NodeStatus.RUNNING)
                # TRACING (s6/b2): open the node's child span HERE (under the
                # active "agent.run" span) at the RUNNING transition so its start
                # aligns with EVENT_NODE_LAUNCHED, and hand it to the task so the
                # node's phi spans nest under it.
                node_span = self._start_node_span(node)
                task = asyncio.create_task(
                    self._node_task(node, node_span), name=f"agent:{node.id}"
                )
                self.tracked[node.id] = task
                running[task] = node.id
                await self._emit(
                    EVENT_NODE_LAUNCHED,
                    {"node_id": node.id, "spec": node.primary_spec,
                     "specs": list(node.effective_specs), "depends_on": list(node.depends_on)},
                )
                del remaining[node.id]

            if not running:
                # Nothing running and nothing launchable → the rest are blocked.
                for nid in list(remaining):
                    st = self._state(nid)
                    if st.status == NodeStatus.PENDING:
                        st.transition(NodeStatus.SKIPPED)
                        st.error = "skipped: blocked by upstream failure"
                        await self._emit(EVENT_NODE_SKIPPED, {"node_id": nid, "blocked_by": "upstream"})
                    del remaining[nid]
                break

            completed, _ = await asyncio.wait(
                running.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in completed:
                nid = running.pop(task)
                st = self._state(nid)
                if task.cancelled():
                    if st.status == NodeStatus.RUNNING:
                        st.transition(NodeStatus.CANCELLED)
                    self._end_node_span(nid, event="cancelled", error="cancelled")
                    continue
                exc = task.exception()
                if exc is None:
                    # Produce step finished → enter the per-node VERIFY GATE
                    # (verifiable), applying the CODER=REVIEWER inline fix if the
                    # gate rejects the output. Only a passing gate crosses to DONE.
                    passed, vreason = await self._verify_and_finalize(
                        node_by_id[nid], self._cache[nid]
                    )
                    if passed:
                        st.transition(NodeStatus.DONE)
                        self._end_node_span(nid, event="done")
                        await self._emit(EVENT_NODE_DONE, {"node_id": nid, "spec": node_by_id[nid].primary_spec, "specs": list(node_by_id[nid].effective_specs)})
                        continue
                    # Gate failed even after inline-fix → ROUTE through the planner's
                    # heal decision (retry re-dispatch / replan / abort), exactly like
                    # a hard failure; else surface FAILED.
                    recovered = await self._heal_failed_node(
                        node_by_id[nid], f"verify gate failed: {vreason}"
                    )
                    if recovered:
                        st.transition(NodeStatus.DONE)
                        self._end_node_span(nid, event="done")
                        await self._emit(EVENT_NODE_DONE, {"node_id": nid, "spec": node_by_id[nid].primary_spec, "specs": list(node_by_id[nid].effective_specs), "via": "replan"})
                    else:
                        st.transition(NodeStatus.FAILED)
                        st.error = f"verify gate failed: {vreason}"
                        self._end_node_span(nid, event="failed", error=st.error)
                        await self._emit(EVENT_NODE_FAILED, {"node_id": nid, "error": st.error})
                    continue
                # Node-level self-heal was exhausted → ROUTE through the planner's
                # heal decision (retry re-dispatch / replan / abort).
                err = f"{type(exc).__name__}: {exc}"
                recovered = await self._heal_failed_node(node_by_id[nid], err)
                if recovered:
                    st.transition(NodeStatus.DONE)
                    self._end_node_span(nid, event="done")
                    await self._emit(EVENT_NODE_DONE, {"node_id": nid, "spec": node_by_id[nid].primary_spec, "specs": list(node_by_id[nid].effective_specs), "via": "replan"})
                else:
                    st.transition(NodeStatus.FAILED)
                    st.error = err
                    self._end_node_span(nid, event="failed", error=err)
                    await self._emit(EVENT_NODE_FAILED, {"node_id": nid, "error": err})

    async def _drive_growable(self, dag: PlanDAG) -> None:
        """Drive a GROWABLE DAG: seed wave, then GROW round-by-round on note gaps (P2.5b).

        Relaxes EXACTLY ONE invariant — "the node set is fixed at unroll time". The seed
        layer is driven by the SAME :meth:`_drive_dag` as any plan; then, while under the
        grower's ``max_layers`` bound, after each wave reaches completion this asks the
        grower (which reuses ``research_tree.run_decision_node`` + ``ResearchState``) for the
        next layer's research nodes, APPENDS them to the live DAG, and drives that next wave.
        ``_drive_dag`` short-circuits the already-cached (done) seed/earlier nodes, so each
        pass runs ONLY the freshly-appended wave with full growing-visibility into the prior
        layers. Growth STOPS on ``stop_research`` (agent_sufficient) / ``no_expansion`` /
        ``max_layers`` (depth_bound) — identical to ``run_research_tree``'s loop; no unbounded
        growth (every exit is a bound the model reasons over or a hard layer ceiling)."""
        grower = self._grower
        # DECOMPOSE-FIRST SEED (mirrors the tree's seed_only_root, d106 #3): if the grower can
        # decompose, REPLACE the unrolled whole-goal seed with the model's scoped children so
        # breadth is FRONT-LOADED before the first stop_research judgement (the tree's breadth
        # source). Fall back to the unrolled seed when the model authors no child (never empty).
        if hasattr(grower, "seed_layer"):
            try:
                seed_nodes = await grower.seed_layer()
            except Exception:  # noqa: BLE001 — decompose is best-effort; keep the unrolled seed
                seed_nodes = []
            if seed_nodes:
                dag.nodes = list(seed_nodes)
                # Recompute the writer-chain maps over the replaced node set (these are research
                # nodes → no writers, but keep the maps consistent with the live DAG).
                self._writer_ids = {n.id for n in dag.nodes if _is_writer_node(n)}
                self._dependent_ids = {}
                for n in dag.nodes:
                    for dep in n.depends_on:
                        self._dependent_ids.setdefault(dep, set()).add(n.id)
        # P2-5c FORWARD HARDENING — a per-engine WALL-CLOCK budget gives the growable loop a
        # GRACEFUL stop: when set (>0, via TreeConfig.grow_wallclock_budget) the loop stops
        # AUTHORING further growth layers once the budget elapses and returns the findings
        # GATHERED SO FAR with stop_reason='budget' — a PARTIAL, never an exception/abort — so a
        # full-depth live/UI run is time-bounded. 0 = OFF (the loop still stops on
        # agent_sufficient / no_expansion / max_layers).
        budget = float(
            getattr(getattr(grower, "config", None), "grow_wallclock_budget", 0.0) or 0.0
        )
        loop_clock = asyncio.get_running_loop().time
        t0 = loop_clock()

        def _sources_so_far() -> int:
            """Cumulative fetched-source count across the gathered research nodes (cheap; the
            grow loop is low-frequency so a per-layer scan of the cache is fine)."""
            total = 0
            for n in dag.nodes:
                r = self._cache.get(n.id)
                tv = getattr(r, "tool_value", None) if r is not None else None
                if isinstance(tv, Mapping):
                    total += len(tv.get("fetched") or [])
            return total

        async def _emit_grow_layer(idx: int, dispatched: int, stop: Optional[str]) -> None:
            # P2-5c — advisory per-layer progress so a long live run is OBSERVABLE (layer index,
            # nodes dispatched this wave, cumulative nodes/sources, elapsed wall-clock, stop).
            # Best-effort: observability never gates or breaks the drive.
            try:
                await self._emit(EVENT_GROW_LAYER, {
                    "layer": idx,
                    "nodes_dispatched": dispatched,
                    "nodes_total": len(dag.nodes),
                    "sources_so_far": _sources_so_far(),
                    "elapsed_s": round(loop_clock() - t0, 1),
                    "stop_reason": stop,
                })
            except Exception:  # noqa: BLE001 — observability only; never breaks the drive
                pass

        # Seed wave (layer 1): the decomposed children (or the unrolled whole-goal seed).
        await self._drive_dag(dag)
        await _emit_grow_layer(1, len(dag.nodes), None)
        max_layers = max(1, int(getattr(grower, "max_layers", 1) or 1))
        layer = 1
        budget_hit = False
        while layer < max_layers:
            # WALL-CLOCK budget check BEFORE authoring the next layer → graceful partial stop.
            if budget > 0 and (loop_clock() - t0) >= budget:
                budget_hit = True
                break
            # M1 (P2-5b-review) — wrap grow() so a transport exception MID-GROWTH stops growth
            # GRACEFULLY with the partial findings already gathered, instead of propagating up
            # and aborting the run (the seed + earlier waves' findings/sources stand).
            try:
                new_nodes, stop_reason = await grower.grow(dag, self._cache, layer)
            except Exception:  # noqa: BLE001 — graceful-partial on mid-growth failure
                if getattr(grower, "stop_reason", None) is None:
                    try:
                        grower.stop_reason = "grow_error"
                    except Exception:  # noqa: BLE001 — duck-typed grower; advisory trace only
                        pass
                await _emit_grow_layer(layer + 1, 0, getattr(grower, "stop_reason", "grow_error"))
                break
            if not new_nodes:
                # agent_sufficient / no_expansion — the grower recorded the reason.
                await _emit_grow_layer(layer + 1, 0, getattr(grower, "stop_reason", stop_reason))
                break
            # APPEND the next wave onto the live DAG (the relaxed invariant) and drive it.
            # validate() (run at _drive_dag entry) re-asserts acyclicity over the grown set.
            dag.nodes.extend(new_nodes)
            await self._drive_dag(dag)
            layer += 1
            await _emit_grow_layer(layer, len(new_nodes), None)
        else:
            # Reached the hard layer ceiling without an explicit model stop.
            if getattr(grower, "stop_reason", None) is None:
                try:
                    grower.stop_reason = "depth_bound"
                except Exception:  # noqa: BLE001 — duck-typed grower; advisory trace only
                    pass
            await _emit_grow_layer(layer + 1, 0, getattr(grower, "stop_reason", "depth_bound"))
        if budget_hit:
            # GRACEFUL wall-clock stop: the partial findings gathered so far stand (no abort).
            try:
                grower.stop_reason = "budget"
            except Exception:  # noqa: BLE001 — duck-typed grower; advisory trace only
                pass
            await _emit_grow_layer(layer + 1, 0, "budget")
        self._grow_layers = layer

    # ------------------------------------------------------------------ #
    # lifecycle: cancel + clean teardown
    # ------------------------------------------------------------------ #
    async def cancel_all(self) -> None:
        """Cancel every still-running tracked task and await its teardown.

        The explicit 'cancel' lifecycle verb (also the timeout path): no node
        task is left orphaned — each live task is cancelled and awaited so the
        single event loop has no dangling agents (d2)."""
        live = [(nid, t) for nid, t in self.tracked.items() if not t.done()]
        for nid, t in live:
            t.cancel()
        for nid, t in live:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            st = self.states.get(nid)
            if st is not None and st.status == NodeStatus.RUNNING:
                st.status = NodeStatus.CANCELLED
                self._end_node_span(nid, event="cancelled", error="cancelled")
                await self._emit(EVENT_NODE_CANCELLED, {"node_id": nid})
        # Defensive: end any node span still open (e.g. a node whose task finished
        # but whose terminal transition path did not run) so no span is leaked
        # un-exported on teardown.
        for nid in list(self._node_spans):
            self._end_node_span(nid, event="cancelled", error="incomplete")

    # ------------------------------------------------------------------ #
    # per-run observability lambda (agent-AUTHORED, user-OBSERVES — s9/a2, d15)
    # ------------------------------------------------------------------ #
    async def _start_run_observability(self, run_id: str) -> Optional[str]:
        """Auto-create the per-run observability lambda; return its sub_id (or None).

        The structural guarantee (d15): the AGENT (this runtime) authors a genuine
        reactive subscription per run that observes the control-plane node
        lifecycle on ``self.plane`` — so the read-only lambda tab populates on
        every live run, never depending on a (possibly tiny) model choosing to
        call ``create_subscription``. Recorded on ``self.lambda_registry`` (the
        shared registry the UI surface reads) while observing THIS run's plane via
        ``source_plane``. Best-effort + advisory: a registry failure here must
        never break the run (the lambda is observe-only), so it is captured and
        the run proceeds without it."""
        if self.lambda_registry is None:
            return None
        try:
            rec = self.lambda_registry.create(
                list(RUN_LIFECYCLE_KINDS),
                label=f"run-observability:{run_id}",
                reducer="each",  # control-plane kinds only → no wake-storm
                reaction="advisory",
                owner={"run_id": run_id, "auto": True, "kind": "observability"},
                note=f"auto-created per-run observability lambda for run {run_id}",
                source_plane=self.plane,
            )
            return rec.sub_id
        except Exception:  # noqa: BLE001 — advisory observability never breaks a run
            return None

    async def _stop_run_observability(self, sub_id: Optional[str]) -> None:
        """Tear the per-run observability lambda down cleanly (no orphaned task).

        Gives the lambda's driver a few scheduling turns to drain queued lifecycle
        events into fires first (best-effort liveness for the tab), then closes it
        — :meth:`LambdaRegistry.close` cancels + awaits the driver, so no task is
        left on the loop. The registered record persists in the read-only snapshot
        (closed, with its fire count), so a run that finished still shows it."""
        if sub_id is None or self.lambda_registry is None:
            return
        try:
            for _ in range(4):
                await asyncio.sleep(0)
            await self.lambda_registry.close(sub_id)
        except Exception:  # noqa: BLE001 — advisory teardown never breaks a run
            pass

    # ------------------------------------------------------------------ #
    # public entrypoint
    # ------------------------------------------------------------------ #
    async def run(
        self,
        dag: PlanDAG,
        *,
        timeout: Optional[float] = None,
        run_id: Optional[str] = None,
    ) -> RuntimeResult:
        """Drive ``dag`` to completion, honouring ``depends_on`` (d2).

        Bounded concurrency, idempotent caching, sub-graph re-plan, and clean
        cancellation on timeout. Returns an aggregated :class:`RuntimeResult`
        (it does not raise on a node failure or a timeout — those are recorded in
        the result so the caller can inspect the full per-node state machine).

        TRACING (s6/b2): the WHOLE drive is wrapped in the parent ``agent.run``
        span — the root of the run's trace tree. Every node span (and, through the
        cross-thread context propagation, every phi span) nests under it because
        the driver and the node tasks all run inside this span's active context.
        ``run_id`` is recorded as ``run.id`` (the job/run correlation id); when the
        caller does not supply one it defaults to this span's 128-bit trace id so
        the attribute is always populated and lines up with the trace Phoenix
        stores. ``node_count`` / ``timeout`` are recorded up front and
        ``replans_used`` / ``timed_out`` once the drive completes."""
        if self.max_concurrency is not None and self.max_concurrency > 0:
            self._sem = asyncio.Semaphore(self.max_concurrency)
        # OVERALL GOAL (d39): read the verbatim goal off the DAG and stash it so every
        # sub-agent built during this drive (producer + inline reviewer + any self-heal
        # sub-graph node, which all read ``self._overall_goal``) feeds it into the node
        # user turn. Single source of truth — it travels with the plan, so the missing-
        # spec resume (which re-runs a rebuilt DAG carrying the same goal) is covered
        # too. Blank/absent => None => omitted everywhere (byte-identical to pre-d39).
        self._overall_goal = (getattr(dag, "goal", "") or "").strip() or None
        # PLAN-CHAINING (c1b/d49.4): map the writer-chain from the authored DAG so a
        # multi-page write-file plan's per-section nodes ACCUMULATE into one file. The
        # set of writer node ids + the dependents map are read once here (the
        # decomposition lives in the DAG topology, not in code) and consulted in
        # :meth:`_run_node` to decide each writer's continuation path + finality.
        self._writer_ids = {n.id for n in dag.nodes if _is_writer_node(n)}
        self._dependent_ids = {}
        for n in dag.nodes:
            for dep in n.depends_on:
                self._dependent_ids.setdefault(dep, set()).add(n.id)
        out = RuntimeResult()
        tracer = get_tracer("agent_runtime.runtime")
        with tracer.start_as_current_span("agent.run") as span:
            rid = run_id or format(span.get_span_context().trace_id, "032x")
            span.set_attribute("run.id", rid)
            span.set_attribute("run.node_count", len(dag.nodes))
            if timeout is not None:
                span.set_attribute("run.timeout_s", float(timeout))
            if self.max_concurrency is not None:
                span.set_attribute("run.max_concurrency", self.max_concurrency)
            # AGENT-AUTHORED observability lambda (s9/a2, d15): created BEFORE the
            # drive so it observes every node-lifecycle event this run emits, and
            # so the lambda tab is populated for the whole run — not just at the
            # end. Best-effort; recorded so the run carries it on the span.
            obs_sub_id = await self._start_run_observability(rid)
            if obs_sub_id is not None:
                span.set_attribute("run.observability_lambda", obs_sub_id)
            # REACTIVE SELF-HEAL RULE (b4, d1/d15): register the observe-only rule
            # that watches node-failure routing on this run's plane, so the heal
            # routing is visible on the read-only live-subscriptions surface. Only
            # when a heal_router is actually wired (else there is no routing to
            # observe). Best-effort + advisory — never breaks the run.
            self._heal_rule_sub_id = None
            if self.heal_router is not None:
                self._heal_rule_sub_id = register_heal_rule(
                    self.lambda_registry, run_id=rid, source_plane=self.plane
                )
                if self._heal_rule_sub_id is not None:
                    span.set_attribute("run.self_heal_rule", self._heal_rule_sub_id)
            # EVENT-DRIVEN PLANNER REACTION (P2.2): start the reactor's plane
            # subscription BEFORE the drive so it reacts to every failure /
            # clarification this run emits, and tear it down cleanly in the finally
            # (no orphaned subscription task). Best-effort: a start failure must never
            # break the run (the failure seam falls back to the synchronous path).
            if self.planner_reactor is not None:
                try:
                    await self.planner_reactor.start()
                    span.set_attribute("run.planner_reactor", True)
                except Exception:  # noqa: BLE001 — advisory; never breaks the run
                    pass
            # GROWABLE DRIVE (P2.5b): a ``growable`` DAG with a wired grower runs the
            # round-by-round growth loop; every other plan runs the single-pass drive
            # (byte-identical). One coroutine either way, so the timeout wrapping is unchanged.
            if getattr(dag, "growable", False) and self._grower is not None:
                driver = self._drive_growable(dag)
            else:
                driver = self._drive_dag(dag)
            try:
                if timeout is not None:
                    try:
                        await asyncio.wait_for(driver, timeout=timeout)
                    except asyncio.TimeoutError:
                        out.timed_out = True
                else:
                    await driver
            finally:
                # Clean teardown: cancel + await any task still in flight (timeout,
                # exception, or a re-plan branch that was abandoned).
                await self.cancel_all()
                # Tear the per-run observability lambda down cleanly too (no
                # orphaned driver task); its record persists in the read-only
                # snapshot for post-run inspection.
                await self._stop_run_observability(obs_sub_id)
                # Tear the self-heal rule down cleanly too (same clean-teardown
                # contract; its closed record persists for post-run inspection).
                await self._stop_run_observability(self._heal_rule_sub_id)
                # Tear the event-driven planner reactor's subscription down cleanly
                # (no orphaned subscription task on the single loop, d2).
                if self.planner_reactor is not None:
                    try:
                        await self.planner_reactor.stop()
                    except Exception:  # noqa: BLE001 — teardown never breaks a run
                        pass

            span.set_attribute("run.replans_used", self._replans_used)
            span.set_attribute("run.timed_out", out.timed_out)
            return self._aggregate(dag, out)

    def _aggregate(self, dag: PlanDAG, out: RuntimeResult) -> RuntimeResult:
        """Read the shared state machine + cache into a :class:`RuntimeResult`."""
        out.replans_used = self._replans_used
        # launch order by launch_seq (nodes that actually launched).
        launched = [s for s in self.states.values() if s.launch_seq >= 0]
        out.launch_order = [s.node_id for s in sorted(launched, key=lambda s: s.launch_seq)]
        for nid, st in self.states.items():
            out.states[nid] = st.as_dict()
            if st.status == NodeStatus.DONE and nid in self._cache:
                out.results[nid] = self._cache[nid]
            elif st.status in (NodeStatus.FAILED, NodeStatus.SKIPPED, NodeStatus.CANCELLED):
                out.failed[nid] = st.error or st.status.value
            if nid in self._heal_logs:
                out.heal_logs[nid] = self._heal_logs[nid]
        return out


__all__ = [
    "SubAgent",
    "SubAgentResult",
    "AgentRuntime",
    "RuntimeResult",
    "ResultValidator",
    "NodeVerifier",
    "Replanner",
    "EVENT_NODE_LAUNCHED",
    "EVENT_NODE_DONE",
    "EVENT_NODE_FAILED",
    "EVENT_NODE_HEALED",
    "EVENT_NODE_CANCELLED",
    "EVENT_NODE_REPLANNED",
    "EVENT_NODE_SKIPPED",
    "EVENT_NODE_VERIFIABLE",
    "EVENT_NODE_REVIEW",
    "EVENT_NODE_INLINE_FIXED",
    "EVENT_NODE_VERIFY_FAILED",
    "EVENT_NODE_COLLISION",
    "EVENT_NODE_COLLISION_RESOLVED",
    "EVENT_NODE_FAILURE_DETECTED",
    "EVENT_HEAL_ROUTED",
]
