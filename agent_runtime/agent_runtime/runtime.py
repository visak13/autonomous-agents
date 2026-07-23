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
import re
import sys
import threading
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

from llm_framework import Chain, Context, Transport
from llm_framework.stages import call_stage, prompt_assembly, structured_output
from llm_framework.tokens import estimate_message_tokens, estimate_tokens
from llm_framework.context import Conversation, deterministic_summary
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode
from reactive_tools import EventPlane, LambdaRegistry, ToolHook
from specialization.loader import SpecLoader

from .tracing import get_tracer, run_blocking_in_span

from .article_note import coerce_article_note
from .chunked_read import chunked_read as _chunked_read
from .research_tree import (
    first_native_call,
    make_tool_spec,
    render_scoped_source_index,
    repair_model_json,
)
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
    # SB-RR (d293): ROLE_RESEARCHER + ROLE_WORKER are no longer referenced here — gather is a
    # self-selected specialization (not a role) and every worker shares ONE unified loop, so the
    # engine branches on neither. Only ROLE_SYNTHESIZER (terminal delivery, d215) survives —
    # plus ROLE_REVIEWER as the DATA the target-artifact gate reads: a reviewer verifies the
    # deliverable, it is not the node that must produce it (P3; planner-declared role).
    ROLE_REVIEWER,
    ROLE_SYNTHESIZER,
    role_framing,
)
# OO tool-bundle layer (d190/d212/d221): the canonical {tools + doctrine} per CAPABILITY
# DOMAIN. NODE-SELF-SELECT (d221): a node SELF-SELECTS the bundle(s) its task needs at
# runtime (the per-node ``_loaded_bundles`` set, grown via the get_bundles tool /
# :meth:`SubAgent._load_bundle`); ``object`` is the only always-on floor. The runtime
# unions the loaded bundles' tools + doctrine; the research/file doctrine TEXTS live in
# the bundles (single source of truth), re-bound here under their prior private names.
from .bundles import (
    BUNDLE_FILE,
    BUNDLE_OBJECT,
    BUNDLE_RESEARCH,
    BUNDLE_RESEARCH_READ,
    bundles_catalog_text,
    compose_doctrine,
    compose_tool_specs,
    expand_bundle,
    get_bundle,
)
from .bundles.base import AGENT_OPERATING_PROTOCOL
from .bundles.file import REPORT_SEPARATION_GUIDANCE as _REPORT_SEPARATION_GUIDANCE
# SoC ENGINE-THIN (SA-5/d254): the web URL/article/readability/record DISPATCH+INGEST
# semantics are OWNED by the web bundle now; the engine imports the bundle's adapter +
# the URL-grounding predicate and DELEGATES (it hardcodes no web semantics of its own).
from .bundles.research import WebGatherAdapter as _WebGatherAdapter
from .bundles.web_ingest import url_offered as _url_offered
from .synth_tools import (
    DONE_SENTINEL,
    _section_headings,
    _strip_fence as _strip_synth_fence,
    collect_fetched_sources,
    derive_output_path,
    document_restart,
    html_close_gap,
    render_scoped_sources,
    resolve_writer_source_budget,
    READ_RANKING_CHUNK_CHARS,
    read_content_char_budget,
    section_reemission,
    select_relevant_chunks,
    render_source_index,
    sanitize_write_path,
    split_done_signal,
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
# RP-3c (d330): ``_VERIFY_REVISE_MAX_CHARS`` (the single-turn whole-document revise size
# cap of the retired engine verify lane) is removed with the lane.

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
_EMBEDDER_WARM_LOCK = threading.Lock()
_EMBEDDER_WARM_THREAD: Any = None


def _warm_read_embedder_async() -> None:
    """Kick the embedder build on a DAEMON thread (idempotent; live 2026-07-13 catch:
    onnxruntime's native DLL load was measured at ~20 MINUTES on this host — a
    synchronous first-use build would freeze a live research read for that long).
    While warming, :func:`_load_read_embedder` returns ``None`` and callers use the
    bounded map/reduce fallback; once the thread lands, later reads get the real
    dense relevance-select. Call this at process startup (the app wiring does) so
    the cost runs off the critical path."""
    global _EMBEDDER_WARM_THREAD
    with _EMBEDDER_WARM_LOCK:
        if _READ_EMBEDDER is not _UNSET_EMBEDDER or _EMBEDDER_WARM_THREAD is not None:
            return

        def _build() -> None:
            global _READ_EMBEDDER
            try:
                from memory.embedder import CpuEmbedder

                built = CpuEmbedder()
            except Exception:  # noqa: BLE001 - optional embedder must never crash
                built = None
            with _EMBEDDER_WARM_LOCK:
                _READ_EMBEDDER = built

        _EMBEDDER_WARM_THREAD = threading.Thread(
            target=_build, name="read-embedder-warm", daemon=True
        )
        _EMBEDDER_WARM_THREAD.start()


def _load_read_embedder() -> Any:
    """The shared MiniLM ``CpuEmbedder``, built ONCE on a background warm thread;
    ``None`` while warming or if fastembed/memory cannot load (caller then uses the
    bounded map/reduce fallback — reads NEVER block on the embedder build)."""
    if _READ_EMBEDDER is _UNSET_EMBEDDER:
        _warm_read_embedder_async()
        return None
    return _READ_EMBEDDER
# Fraction of the write window the TOTAL of a section's source excerpts may occupy,
# leaving the rest for the section prompt scaffolding + the num_predict output.
_WRITE_SOURCE_WINDOW_FRACTION = 0.6

# d162 — the per-source LEAD cap for the UNSCOPED terminal writer (a single-section
# synthesis the planner left without ``source_ids``). It is fed the bounded compact SOURCE
# INDEX + a LEAD excerpt over EVERY source (so it has real grounding to write a substantive
# report) instead of the raw fetched-body fold that ballooned the served write input to
# 137KB / 107% of num_ctx at 11 sources (a15, silently truncated → thin report). Generous
# enough for substance, but the renderer's window-sized ``lead_total`` is the hard bound:
# more sources => thinner per-source leads, the TOTAL stays a fraction of the window.
_UNSCOPED_LEAD_CHARS = 3600

# AGENTIC RESEARCH loop bounds (s9/c5, d49/d50 — retires flags #1/#3). A web_search
# node is no longer driven by a deterministic search-then-read EXECUTOR; it is a TRUE
# AGENT that DECIDES to search and which sources to read via lightweight tool calls
# (``web_search``/``web_fetch`` — small args the small model emits reliably, unlike
# content-laden JSON, d49). These are NON-FLOW cost/safety bounds (a cap on a loop the
# model drives), NOT flow gates: ``RESEARCH_MAX_TURNS`` caps total ReAct turns and
# ``RESEARCH_DEFAULT_FETCH_CAP`` is the fetch cap used when a caller wired none.
RESEARCH_MAX_TURNS = 12
RESEARCH_DEFAULT_FETCH_CAP = 5
# d157 (latency-first) — tool-layer chunking on the RESEARCH.react READ PATH. Each web_fetch
# returns the 1+ MOST RELEVANT CHUNKS (the top embedding-ranked passages for THIS sub-question,
# d184 — several passages assembled up to the budget, not the single top chunk) to the research
# worker, NOT the whole 8-27KB article body — so a multi-turn react loop's accumulated input
# stays compact and the per-node latency drops. The full verbatim ``markdown`` is STILL stored
# on the source record (the writer's citation / load_source path is untouched); this only bounds
# what the WORKER ingests per fetch. Env-tunable for live tuning; the relevance-select keeps the
# MOST relevant passages so the bound trims raw bulk, not the useful content. A bounded compact
# representation, NOT an app truncation cap.
RESEARCH_READ_CHUNK_CHARS = max(1500, int(os.environ.get("RA_RESEARCH_READ_CHUNK_CHARS", "5000")))
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
# BUDGET-RESERVING SEED SLICE (live 6GB catch): the FRACTION of the growable drive's
# wall-clock budget the SEED wave may spend before the dispatch gate stops launching
# further gather nodes (see AgentRuntime._dispatch_deadline). Reserves the remainder
# for the grow/decision loop (expand/prune/stop reasoning), which was previously
# starved when a wide seed consumed the whole budget and the outer run timeout
# cancelled it mid-flight. Env-tunable; a NON-FLOW resource bound (d240).
RESEARCH_SEED_BUDGET_FRACTION = min(
    0.95, max(0.2, float(os.environ.get("RA_RESEARCH_SEED_BUDGET_FRACTION", "0.6")))
)
_SEED_BUDGET_FRACTION = RESEARCH_SEED_BUDGET_FRACTION

# The research-agent instruction (d38/d39/d50 prompt-quality mandate: crisp, anti-
# hallucination) — the worker must gather REAL evidence via its tools before answering,
# and FINDINGS are RAW prose (never JSON — content is RAW on every route, d50.1). The TEXT
# lives in the ResearchBundle (d190 — the bundle owns the doctrine) and is surfaced to the
# model EXACTLY ONCE, in the ``get_bundles`` LOAD observation when the node self-selects the
# bundle (its ``own_doctrine``, carried forward by the convo window). d229/d263: the research
# ReAct loop NEVER re-pastes that doctrine into the per-turn task message — that was a pure
# ~400-500-token-per-prompt duplication (trace 620d38fa); d263 further retired the pinned-head
# copy, so the doctrine now rides the load observation ONCE and nothing re-pastes it.
# CoT-autonomy P1: ONE fact-only recovery line for an unusable turn (no tool call,
# no output). The reply channel's two legal forms live in the OPERATING PROTOCOL on
# the system turn (single owner); this observation only states what happened.
_UNUSABLE_TURN_NOTE = (
    "Your last reply was neither a single tool-call JSON object nor your final output."
)
# Back-compat aliases (call sites collapsed to the one neutral note).
_RESEARCH_NUDGE = _UNUSABLE_TURN_NOTE
_WORKER_NUDGE = _UNUSABLE_TURN_NOTE
# CoT-autonomy P3: the finalize prompts are a RESOURCE FACT — the loop is ending;
# what a good conclusion looks like is the spec's knowledge, not a per-turn command.
_TURN_BUDGET_NOTE = (
    "Turn budget exhausted — this is your final turn; no further tool call will "
    "execute."
)
# Back-compat aliases (both call sites collapsed to the one budget fact).
_RESEARCH_FINALIZE = _TURN_BUDGET_NOTE
_WORKER_FINALIZE = _TURN_BUDGET_NOTE
# CoT-autonomy P3 (owner ruling): the no-fab GATHER-MORE bounce-gate is DELETED —
# no engine turn re-prompts a conclusion. Grounding lives in the research spec.

# CoT-autonomy P3: the TARGET-ARTIFACT bounce-gate is DELETED (owner ruling).
# Delivery honesty is downstream: the staleness guard + truthful deliverable_bytes.

# MALFORMED TOOL-CALL feedback bound (CoT-autonomy P2): how many tool-shaped-but-
# unparseable replies get the parse-error fact before the reply stands as prose.
_MALFORMED_CALL_MAX = 3
# CoT-autonomy P3: the per-run NOTE CLAUSE and the NOTE GATE are DELETED — note
# discipline's owners are the note tool description + the research doctrine.



def _unescape_lenient(s: str) -> str:
    """Decode standard JSON string escapes, passing an INVALID escape through verbatim
    (the tolerant half of the lenient tool-call recovery — never raises)."""
    out: list[str] = []
    i = 0
    mapping = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/",
               "b": "\b", "f": "\f"}
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            if n in mapping:
                out.append(mapping[n])
                i += 2
                continue
            if n == "u" and i + 6 <= len(s):
                try:
                    out.append(chr(int(s[i + 2:i + 6], 16)))
                    i += 6
                    continue
                except ValueError:
                    pass
            # invalid escape: keep the model's bytes verbatim
            out.append(c)
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _lenient_content_call(raw: str) -> Optional[tuple[str, dict]]:
    """CHANNEL-ROBUSTNESS recovery (CoT-autonomy P6, live promptlab catch): a
    tool-shaped reply carrying a MULTI-KB ``content`` string frequently breaks strict
    JSON on a single bad escape (E4B one-shots whole documents), and the model retries
    the same shape after the parse-error fact — losing the write entirely. When the
    call's INTENT is unambiguous ({"tool": …, "args": {… "content": "…" …}}), recover
    it: small fields via exact matches, the content span VERBATIM with a tolerant
    unescape. The recovered bytes are the model's own — nothing is composed or fixed;
    this is the tool channel accepting its own format leniently, like any real
    function-calling runtime. Returns None when the shape is not unambiguous."""
    # NORMALIZE channel junk first (live failure modes): a markdown fence around
    # the call, and trailing non-JSON tokens after the close — e.g. a stray ')'
    # (the model writes pythonic call syntax tails on big calls).
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.lstrip("`").lstrip()
        if raw[:4].lower() == "json":
            raw = raw[4:]
        raw = raw.rstrip("`").rstrip()
    # XML-ish channel wrappers (live variant: the reply ends '})</tool_call>').
    raw = re.sub(r"^<[a-zA-Z_]+>\s*", "", raw)
    raw = re.sub(r"(\s*</?[a-zA-Z_]+>\s*|[)\];`\s])+$", "", raw)
    # The trailing close tolerates a MISSING outer brace (a live failure mode: the
    # model truncates the final '}' on a multi-KB call).
    m = re.match(
        r'\s*\{\s*"tool"\s*:\s*"([A-Za-z_]\w*)"\s*,\s*"args"\s*:\s*\{(.*?)\}?\s*\}\s*$',
        raw, re.S,
    )
    if not m:
        return None
    tool, body = m.group(1), m.group(2)
    if '"content"' not in body:
        return None
    args: dict[str, Any] = {}
    for k in ("path", "sid", "url", "reason"):
        km = re.search(r'"%s"\s*:\s*"([^"\n]*)"' % k, body)
        if km:
            args[k] = km.group(1)
    for k in ("append", "overwrite"):
        km = re.search(r'"%s"\s*:\s*(true|false)' % k, body)
        if km:
            args[k] = km.group(1) == "true"
    cm = re.search(
        r'"content"\s*:\s*"(.*)"(?=\s*(?:,\s*"\w+"\s*:|\s*$))', body, re.S,
    )
    if not cm:
        return None
    args["content"] = _unescape_lenient(cm.group(1))
    return tool, args


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

# ONE-DRIVE PHASE-TRANSITION AUTHORING HOOK (RP-6c B1). Given the live runtime, the live DAG, and
# the next phase's PLAN KIND, author (MODEL-authored, via ``IncrementalPlanner.plan``) the next
# phase's sub-DAG and return its node(s). It is the mid-drive analogue of ``DagGrower.grow`` — the
# grower authors the next RESEARCH wave; this authors the next PHASE (research → write). The engine
# does NOT synthesize the structure; it only INVOKES the authorer the shape's declared phase names
# and appends the returned model-authored nodes. Async; returns an empty sequence to author nothing.
PhaseAuthor = Callable[["AgentRuntime", PlanDAG, str], Awaitable[Sequence[PlanNode]]]


@dataclass
class PhaseTransition:
    """The OPTIONAL, additive wiring that turns the growable research drive into the ONE-DRIVE
    phase-transition drive (RP-6c B1). Set on :attr:`AgentRuntime._phase_transition`; consulted by
    :meth:`AgentRuntime._drive_growable` AFTER the research grow loop stops.

    The shape OWNS the phase sequencing (RP-6b's ``[[phases]]`` → ``next_phase_plan`` /
    ``spec_role_for``); the engine drive READS that declaration and performs the mid-run authoring;
    the MODEL authors the actual write topology + content. There is NO engine ``if phase == 'write'``
    structural authoring beyond "call the authorer the shape's phase names and append its nodes".

    Fields
    ------
    next_plan:
        The shape's ``next_phase_plan`` bound method (reads the DECLARED phase order). Given the
        last completed phase kind (``first_kind``), returns the next plan kind (e.g. ``"write_plan"``)
        or ``"done"`` at the terminal phase. When it returns ``"done"`` the drive STOPS at research
        (no transition), byte-identical to a research-only run.
    author:
        The MODEL-authoring hook (:data:`PhaseAuthor`) invoked to author the next phase's sub-DAG.
        For B1 an injected minimal hook exercises the mechanism; the live hook (composing the write
        goal from live run state + ``IncrementalPlanner.plan``) lands in B2.
    deliverable_path:
        O1 — the write-phase delivery target STAMPED per-node onto every authored write-phase node
        (``PlanNode.deliverable_path``) so ONLY those nodes take the writer route in this SHARED
        runtime (research nodes, which carry none, keep the research route). Decided by the caller
        from the shape's declared ``spec_role_for(next_kind) == 'writer'`` — not a spec-name branch.
        Empty → the authored nodes are appended UNSTAMPED (they must self-carry a delivery target or
        route as plain workers).
    first_kind:
        The phase kind the research grow loop represents (``"research"``). Passed to ``next_plan`` to
        find the phase to transition INTO.
    headroom_fraction:
        O2 — the fraction of the run's timeout envelope RESERVED for the write phase (authoring +
        the write node run) within this one drive. The research grow budget is already set to
        ``timeout * (1 - headroom_fraction)`` upstream (agentic ``_run_generic_research_phase``), so
        the grow loop stops AUTHORING new research layers with this headroom left; the drive records
        the reservation for the trace/tests. Default 0.1 (the established ~10% write headroom).
    """

    next_plan: Callable[[str], str]
    author: PhaseAuthor
    deliverable_path: str = ""
    first_kind: str = "research"
    headroom_fraction: float = 0.1

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
# The TEXT now lives in the FileBundle (d190/d212 — the bundle owns the doctrine) and is
# imported above as ``_REPORT_SEPARATION_GUIDANCE`` (byte-identical), so the write
# loop is unchanged while the bundle is the single source of truth.

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
    # UNIVERSAL FINALIZE PAIR (d285 SB-4): the ``(summary, memory_index)`` this node emitted
    # when it finished — SB-2's :meth:`~agent_runtime.Planner.finalize_node` digest (the
    # NODE's own model summary of its real work) plus the index of the research memory it
    # used. Produced by the runtime's injected ``node_finalizer`` (the served orchestration
    # wires it to the planner + SB-1's resolver); ``None`` when no finalizer is wired
    # (offline/unit path) — then the inter-node handoff is byte-identical to pre-SB-4. The
    # downstream node's :meth:`SubAgent._compose_task` consumes the pair as the SOLE
    # inter-node payload (the clipped-prose input + the folded fetched value are dropped).
    summary: Optional[str] = None
    memory_index: Optional[str] = None

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
        upstream_memory: Optional[Mapping[str, Mapping[str, str]]] = None,
        max_repair_attempts: int = 2,
        max_verdict_repairs: int = 2,
        read_search_max_fetch: int = 0,
        search_tool: str = "web_search",
        fetch_tool: str = "web_fetch",
        emit_article_notes: bool = False,
        note_tool: str = "note",
        chunked_read: bool = False,
        fetched_char_budget: int = 2000,
        upstream_input_char_budget: int = 4000,
        writer_source_budget: Optional[int] = None,
        chain_continue_path: Optional[str] = None,
        chain_is_final: bool = True,
        chain_sources: Optional[Sequence[Mapping[str, str]]] = None,
        chain_notes: Optional[Sequence[Mapping[str, Any]]] = None,
        read_embedder: Any = None,
        deliverable_path: Optional[str] = None,
    ) -> None:
        self.node = node
        self.transport = transport
        self.hook = hook
        # NODE-SELF-SELECT (d221): the bundle(s) THIS node has loaded. Starts at the
        # always-on ``object`` floor only — there is no role/tool -> bundle table; a node
        # grows this set by SELF-SELECTING bundles at runtime (the get_bundles tool /
        # :meth:`_load_bundle`). Each load surfaces that bundle's doctrine ONCE in its load
        # observation (d263), so the model's active doctrine reflects exactly the
        # capabilities it actually loaded.
        self._loaded_bundles: set[str] = {BUNDLE_OBJECT}
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
        # CHAIN-NOTES seam (SA-4/d234/d235 — folds in the SA-1-deferred read_notes bridge):
        # the SEPARATE WRITE RUNTIME feeds its served research NOTES here (the gather notes
        # arrive as a write_report_spa PARAM, not via the DAG, so ``_collect_upstream_notes``
        # is empty on the write runtime). :meth:`_node_run_ctx` folds these into ``ctx['notes']``
        # so a write/review node that SELF-SELECTS ``research_read`` binds ``read_notes`` — the
        # CHEAP first leg of the d234/d235 read-hierarchy — WITHOUT the retired per-run
        # pre-registration. The exact mirror of ``chain_sources`` for the sources leg. None /
        # empty (every non-report caller, and the gather DAG itself) → byte-identical (no notes
        # fed; the gather runtime keeps reading its upstream notes via the DAG as before).
        self._chain_notes = list(chain_notes) if chain_notes else []
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
        # s14/P3A §3D — a FRAMEWORK-INJECTED review node (id ``*_review`` / ``final_review``)
        # SB-6/d301 — the report write phase's single DELIVERABLE path (the runtime's
        # ``deliverable_path``, set ONLY on the write runtime by chat_app.run_section_write_phase).
        # This is the WRITE-PHASE-EXCLUSIVE delivery-context signal the dispatch routes the TOOL-LESS
        # write node on (NOT ``chain_sources``, which a follow-up READER also carries to resolve
        # prior sources — routing on chain_sources alone over-broadly forced a non-writer to write a
        # file). None for every non-write-phase node (research/gather/follow-up), so those stay on
        # the unified self-select loop. The all-writers invariant (only writers run in a runtime with
        # deliverable_path set) makes routing all such non-review nodes to the writer SOUND.
        self._deliverable_path = (deliverable_path or "").strip() or None
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
        # SoC ENGINE-THIN (SA-5/d254): the WEB bundle owns the web_search/web_fetch dispatch +
        # all URL/article/readability/record semantics. The engine constructs the bundle's
        # gather adapter (keyed by THIS run's configured tool names) and delegates a web tool
        # call to it (:meth:`_dispatch_research_tool`), so the engine itself keeps only generic
        # by-name dispatch. Built directly from the bundle class (the single owner) — no web
        # logic lives in this module.
        self._web_adapter = _WebGatherAdapter(search_tool, fetch_tool, note_tool)
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
        # RP-3c (d330): the flag-gated NO-FAB VERIFY/REVISE self-review lane
        # (``self._verify_lane``) is RETIRED. The model self-review MOVED to the definition
        # layer (the writer specs' _COHERENT_ARTIFACT_DOCTRINE self-review-before-finish
        # clause); the no-fab research GATHER-MORE gate is KEPT but DE-FLAGGED to an
        # output-agnostic signal gate (see the write loop). No ``verify_lane`` boolean
        # survives (d311 no-hardcoded-flags; d319 the engine authors/decides/fixes nothing).
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
        # UNIVERSAL FINALIZE HANDOFF (d285 SB-4): the DIRECT-upstream ``(summary, memory_index)``
        # pairs — ``{dep_id: {"summary": str, "memory_index": str}}`` — the runtime built from
        # each finished dependency's :class:`SubAgentResult` finalize pair (SB-2). This is the
        # SOLE inter-node context payload (d285): for any dep present here, ``_compose_task``
        # renders ITS (summary, index) pair and DROPS that dep's clipped-prose input (ch1) AND
        # its directly-folded fetched value (ch3) — the detail lives in the research memory,
        # read by index on demand (the writer's scoped-source PUSH stays a SEPARATE delivery
        # path, not a re-fold here). Built from ``node.depends_on`` ONLY (d15 direct-upstream).
        # Empty (no finalizer wired / a seed node) => the handoff is byte-identical to pre-SB-4.
        self._upstream_memory = {
            str(k): dict(v) for k, v in dict(upstream_memory or {}).items()
            if isinstance(v, Mapping)
        }
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
        # MEMORY-BY-HANDLE (d221): when this node is BOUND to a research memory, name the
        # handle so the model knows the research lives there and READS it via its tools
        # (load_source / the on-demand source index) — NEVER expecting a verbatim dump in
        # this turn (d192/d202). The line is grounding CONTEXT, not a new instruction; it
        # is omitted entirely when the node carries no handle (byte-identical to pre-d221).
        handle = getattr(self.node, "research_memory_handle", None)
        if handle:
            parts.append(
                f"\nBinded research memory: {handle}\n"
                "Your research is held in this memory — READ it with your source tools "
                "(load_source by [S#], or the source index) when you need a fact, figure, "
                "or URL; do not expect it pasted in full here, and never invent a source."
            )
            # P2 pull-targeting (Gate-2b live catch): telling the model to "load_source
            # by [S#]" without SHOWING which [S#] exist left it nothing to aim at — it
            # wrote from its own knowledge and even emitted placeholder timeline rows.
            # Render the COMPACT index (titles+urls ONLY, never bodies — the s14 read
            # contract) so every pull has a concrete target. Data, not steering.
            if self._chain_sources:
                from .research_tree import render_verbatim_source_index

                parts.append(
                    "\n" + render_verbatim_source_index(self._chain_sources)
                )
        # UNIVERSAL FINALIZE HANDOFF (d285 SB-4): the SOLE inter-node context payload is the
        # DIRECT-upstream ``(summary, memory_index)`` PAIR(s). For every dep that carries a pair
        # we render the pair here and DROP that dep's clipped-prose input (ch1) AND its folded
        # fetched value (ch3): the engine no longer passes raw upstream bodies as context — it
        # passes the model's own per-node SUMMARY + the INDEX of the memory the detail lives in,
        # which the node reads on demand by index (the writer's scoped-source PUSH is a separate
        # delivery path, NOT a re-fold here). ``_upstream_memory`` is built from ``depends_on``
        # only, so this stays DIRECT-upstream-only (d15). Empty => byte-identical to pre-SB-4.
        paired_deps = {d for d, p in self._upstream_memory.items() if p}
        if paired_deps:
            parts.append(self._upstream_pair_block(paired_deps))
        if inputs:
            # ch1 COLLAPSE: a paired dep's prose output is REPLACED by its (summary, index) pair
            # above — render only the NON-paired deps' inputs (byte-identical when no pair).
            shown = [(k, v) for k, v in inputs.items() if k not in paired_deps]
            if shown:
                parts.append("\nINPUTS FROM PRIOR STEPS:")
                for k, v in shown:
                    parts.append(
                        f"- {k}: {str(v)[: self._upstream_input_char_budget]}"
                    )
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
        # d156/d162: a write-phase node fed via the BOUNDED SOURCE INDEX + LEAD +
        # ``load_source`` (compact + chunked on demand) must NOT also get the upstream FETCHED
        # full bodies folded here — that re-introduces the 81-137KB raw-body dump the
        # tool-layer chunking exists to eliminate (a15 measured 137KB / 107% of num_ctx at 11
        # sources for the UNSCOPED single-section synthesis, silently truncated → thin report).
        # :meth:`_feeds_via_source_index` covers BOTH a scoped node (``source_ids``) AND the
        # unscoped terminal writer/review now routed through the compact full-index feed; an
        # ordinary worker with no chain sources keeps its fetched fold (its only source feed).
        feeds_via_index = self._feeds_via_source_index()
        for dep, uv in self._upstream_tool_values.items():
            if uv is None:
                continue
            # ch3 COLLAPSE (d285 SB-4): a dep handed off via its (summary, memory_index) pair
            # does NOT also get its raw fetched bodies folded as inter-node context — the detail
            # is read from that memory by index (NOT pasted). Drop it here.
            if dep in paired_deps:
                continue
            # s14/a8 (d149): a WRITER dependency's tool_value is just the deliverable PATH
            # reference (``{"path": ...}``), not source content. Folding it rendered a
            # "SOURCES & FINDINGS FROM PRIOR STEP … TOOL OUTPUT (file_write): {'path': …}"
            # block that a small writer then ECHOED into the document (the scaffolding leak).
            # Skip a path-only value — it carries no findings for this node to use (the real
            # sources arrive via the scoped SOURCE INDEX / the research fetched-content feed).
            if isinstance(uv, Mapping) and uv.get("path") and not uv.get("fetched"):
                continue
            if feeds_via_index and isinstance(uv, Mapping) and uv.get("fetched"):
                continue
            parts.append(
                f"\nSOURCES & FINDINGS FROM PRIOR STEP {dep} "
                "(use this content directly):"
            )
            parts.append(self._render_tool_value(uv))
        if tool_value is not None:
            parts.append(self._render_tool_value(tool_value))
        return "\n".join(parts)

    def _upstream_pair_block(self, paired_deps: set[str]) -> str:
        """Render the d285 SB-4 inter-node payload — the DIRECT-upstream ``(summary, index)`` pair(s).

        This is the SOLE inter-node context the engine passes (the clipped-prose input + the
        folded fetched bodies are dropped for these deps): each upstream's own per-node SUMMARY
        (SB-2's ``finalize_node`` digest) plus the INDEX of the research memory holding its
        detail, which the node reads on demand BY INDEX (never expecting it pasted here).

        UNIFORM rendering — NO role/spec conditional (d285 anti-fabrication: zero spec/role-name
        conditionals). The worker-vs-reviewer distinction the design describes EMERGES from the
        DAG, not a code branch: a downstream WORKER usually has ONE direct upstream, so it sees
        that one previous step's pair; a downstream REVIEWER/synthesizer joins MANY branches, so
        it naturally sees the FULL set of their (summary, index) pairs — i.e. the joined overall.
        The engine authors no structure and folds no raw upstream bodies — it names the summary
        and the index. Deterministic ``depends_on`` order for a stable, legible turn."""
        rows = []
        for d in self.node.depends_on:
            if d not in paired_deps:
                continue
            p = self._upstream_memory.get(d) or {}
            summ = str(p.get("summary") or "").strip() or "(no summary)"
            idx = str(p.get("memory_index") or "").strip() or "(unset)"
            rows.append(f"- prior step {d} [research memory: {idx}]: {summ}")
        if not rows:
            return ""
        return (
            "\nUPSTREAM RESEARCH (what the previous step(s) produced — each a (summary, "
            "research-memory index) pair; the full DETAIL is held in that memory, read it BY "
            "INDEX with your source tools when you need a fact/figure/URL, never expect it "
            "pasted here):\n" + "\n".join(rows)
        )

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

    def _feeds_via_source_index(self) -> bool:
        """True when this node receives the run's sources via the BOUNDED SOURCE INDEX feed
        (so the raw fetched-body fold must be suppressed — d156/d162).

        Fires for the write-phase nodes that route through :meth:`_scoped_source_block`:
        a SCOPED node (``source_ids``), or an UNSCOPED terminal writer / file-delivery /
        review node (which now gets the compact full-index feed instead of the
        num_ctx-saturating raw-body dump). An ordinary worker step with no chain sources
        keeps its fetched fold (its only source feed). Guarded on ``_chain_sources`` so a
        research/worker node — which never has chain sources set — is never starved."""
        if not self._chain_sources:
            return False
        if self.node.source_ids:
            return True
        if self.node.role == ROLE_SYNTHESIZER:
            return True
        if (self.node.tool or "") in ("file_write", "write_file"):
            return True
        return False

    def _bound_chat_convo(
        self,
        convo: list[dict[str, Any]],
        system: Optional[str],
        *,
        reserve_predict: int = SYNTH_NUM_PREDICT,
    ) -> list[dict[str, Any]]:
        """Window a multi-turn write/review convo so a served ``llm.chat`` input never
        approaches num_ctx (d162).

        The on-disk deliverable is the ground truth — each turn already re-feeds the
        document's bounded TAIL via ``file_read`` (writer) or a clamped slice (reviewer), and
        the model is told to continue from THAT, not from its own accumulated section
        emissions sitting in history. So when the assembled window (system + convo) would
        crowd num_ctx, DROP the oldest MIDDLE turns while KEEPING ``convo[0]`` (the task + the
        bounded SOURCE INDEX) and the most-recent turns (the latest file tail + the
        continue-from-here instruction). The dropped text is not lost — it is on the file and
        re-surfaced via the bounded tail — so the bound emerges from the chunked/compact
        representation, NOT a blunt truncation of the live feed. Short runs (≤3 turns) and
        runs already inside budget are returned unchanged (byte-identical)."""
        if len(convo) <= 3:
            return convo
        num_ctx = int(self._call_opts.get("num_ctx") or _DEFAULT_WRITE_NUM_CTX)
        # Reserve the model's own output (num_predict) + a safety margin so prompt + output
        # never reach the num_ctx ceiling; the rest is the INPUT budget.
        budget = max(2048, num_ctx - int(reserve_predict) - 2048)
        sys_msgs = (
            [{"role": "system", "content": system}] if system else []
        )

        def total(msgs: list[dict[str, Any]]) -> int:
            return estimate_message_tokens(sys_msgs + msgs)

        if total(convo) <= budget:
            return convo
        head, tail = convo[:1], convo[1:]
        # Drop the oldest middle turn until the window fits, always retaining the last two
        # turns (the most recent emission + the file-tail observation to continue from).
        dropped = 0
        while len(tail) > 2 and total(head + tail) > budget:
            tail = tail[1:]
            dropped += 1
        # OBSERVABILITY (d162 hard line): this safety net is NON-LOAD-BEARING — the tool/feed
        # push-leads + fold suppression must bound EVERY input BY CONSTRUCTION so this never
        # fires. If it ever does, SURFACE it (it is not a silent app-cap: it drops only the
        # OLDEST middle turns, whose content is on disk and re-fed via the bounded tail — the
        # task turn + SOURCE INDEX + the recent file tail are always retained, never a figure
        # or source dropped). A non-zero count on the trace flags that the primary bound
        # leaked and must be tightened at the tool/feed layer.
        if dropped:
            try:
                span = trace.get_current_span()
                span.set_attribute("convo_bound.fired", True)
                span.set_attribute("convo_bound.dropped_middle_turns", dropped)
            except Exception:  # noqa: BLE001 - observability must never break the write loop
                pass
        return head + tail

    # ------------------------------------------------------------------ #
    # PER-NODE bounded MULTI-TURN MEMORY (d263 — pin/SWA-tail retired): the simple
    # middle-turn compaction that wraps a per-node loop's convo (see _node_history).
    # ------------------------------------------------------------------ #
    def _node_bundle_doctrine(self) -> str:
        """The composed doctrine of the bundle(s) THIS node has LOADED (d212/d221).

        NODE-SELF-SELECT (d221): the doctrine is the union over ``self._loaded_bundles``
        — the ``object`` floor plus whatever the node self-selected at runtime — NOT a
        role/tool -> bundle table lookup. The union is the 'active bundle defs' of the
        capabilities the node loaded. Never raises — a mis-wired bundle degrades to no
        doctrine rather than crashing the node."""
        try:
            names = getattr(self, "_loaded_bundles", None) or {BUNDLE_OBJECT}
            return compose_doctrine(names)
        except Exception:  # noqa: BLE001 - doctrine composition must never break a node
            return ""

    def _load_bundle(self, name: str) -> dict[str, Any]:
        """SELF-SELECT a bundle at runtime (d221) — the effect of ``get_bundles(name=…)``.

        Records ``name`` in :attr:`_loaded_bundles` (so the node's active doctrine now
        carries its usage text) and registers its handler-backed ToolDefs onto this node's tool
        registry when the run supplies one (the real GrowableToolRegistry growth point,
        bound to this run's fetched sources). Returns the bundle's ``{loaded, summary,
        doctrine, tools}``; an unknown name degrades to ``{error, …}`` rather than
        crashing the node."""
        from .bundles import UnknownBundleError

        registry = getattr(self.hook, "registry", None) if self.hook is not None else None
        ctx: dict[str, Any] = {}
        if self._chain_sources:
            ctx["sources"] = self._chain_sources
        try:
            result = expand_bundle(str(name).strip(), registry, ctx)
        except UnknownBundleError as exc:
            return {"error": str(exc)}
        self._loaded_bundles.add(result["loaded"])
        return result

    # ------------------------------------------------------------------ #
    # NODE-SELF-SELECT scaffolding (d221/d242 — TRUE self-select): every in-plan
    # node starts TOOL-LESS (only get_bundles + finish) and MUST self-select the
    # bundle(s) its task needs to OBTAIN its domain tools. There is NO role->bundle
    # table and NO pre-mounted domain tool — tools + doctrine + ctx-sources all arrive
    # on bundle LOAD. The same scaffolding drives every loop (research/review/write/
    # synthesis/file-delivery + the linear chat worker), exactly as the planner
    # self-selects via the same get_bundles tool.
    # ------------------------------------------------------------------ #
    def _node_run_ctx(self) -> dict[str, Any]:
        """The per-run binding ctx a self-selected bundle needs to register its tools.

        GENERIC seam (as4): supplies the run's fetched SOURCES (so research_read binds
        load_source to the real prior research) AND the prior gather NOTES (so research_read
        also binds read_notes — the CHEAP first leg of the cost hierarchy) plus the configured
        research tool names + note flag (so the research bundle binds the configured search/
        fetch/note schemas). A DOMAIN-AGNOSTIC memory-read (as4/d241) is realized HERE: the
        sources + notes are collected SOURCE-AGNOSTICALLY (web ``fetched``/``article_notes`` OR
        generic ``records``/``notes``), so a non-web complex-memory type (codebase, vector-db)
        extends THIS provider — and reaches every self-selecting node, INCLUDING the LINEAR/chat
        worker answering a follow-up over prior research — without touching any loop."""
        ctx: dict[str, Any] = {
            "search_tool": self._search_tool,
            "fetch_tool": self._fetch_tool,
            "note_tool": self._note_tool,
            "emit_notes": self._emit_article_notes,
        }
        if self._chain_sources:
            ctx["sources"] = self._chain_sources
        # DOMAIN-AGNOSTIC memory-read (as4/d241): bind read_notes (the cheap gist leg) — not
        # only load_source — for ANY self-selecting node by supplying the prior gather NOTES.
        # SA-4: the WRITE RUNTIME's served notes (``_chain_notes``, the retired read_notes
        # pre-reg's replacement) ride FIRST, then any in-DAG upstream notes — so the write/
        # review path binds read_notes via self-select (its DAG upstream is empty) AND the
        # gather DAG keeps reading its upstream notes. None fed on either side → no key.
        notes = list(self._chain_notes) + self._collect_upstream_notes()
        if notes:
            ctx["notes"] = notes
        return ctx

    def _collect_upstream_notes(self) -> list[Mapping[str, Any]]:
        """The prior gather NOTES this node can READ (as4 domain-agnostic memory-read, d241).

        Folds the structured note artifact every upstream gather dependency emitted into this
        node's ``tool_value`` — under the web key (``article_notes``) or a generic key
        (``notes``) — so ``read_notes`` binds for ANY self-selecting node (incl. the linear
        worker) over ANY source. Source-agnostic + order-preserving; ``read_notes`` keys each
        gist to the SAME ``[S#]`` as the supplied ``sources``."""
        out: list[Mapping[str, Any]] = []
        for tv in self._upstream_tool_values.values():
            if isinstance(tv, Mapping):
                for x in (tv.get("article_notes") or tv.get("notes") or []):
                    if isinstance(x, Mapping):
                        out.append(x)
        return out

    def _get_bundles_handler(self) -> Callable[..., dict[str, Any]]:
        """The ``get_bundles`` handler bound to THIS node (memoized).

        Lists the catalog, or LOADS a bundle by name — registering its tools onto the
        node's LIVE registry (so they become callable), binding this run's ctx, and
        recording the load in :attr:`_loaded_bundles` (so the node's active doctrine grows). This
        is the SAME ``make_get_bundles`` the planner registers, wired with the node's
        registry + ctx + the load hook (d242)."""
        handler = getattr(self, "_get_bundles_handler_cached", None)
        if handler is None:
            from .discovery_tools import make_get_bundles

            registry = getattr(self.hook, "registry", None) if self.hook is not None else None
            handler = make_get_bundles(
                registry=registry,
                ctx_provider=self._node_run_ctx,
                on_load=self._loaded_bundles.add,
            )
            self._get_bundles_handler_cached = handler
        return handler

    def _get_bundles_spec(self) -> dict[str, Any]:
        """The native ``get_bundles`` tool schema — the SELF-SELECT surface every in-plan
        node is offered (d242). The description is the d186 selection lever: it frames that
        the node starts tool-less and MUST load the bundle its task needs FIRST."""
        return make_tool_spec(
            "get_bundles",
            "SELF-SELECT your tools. Call with NO args to LIST the capability bundles you "
            "can load ({name, summary}); call name=\"<NAME>\" to LOAD one — its tools become "
            "callable and its doctrine guides you. You START with ONLY get_bundles + finish, "
            "so you MUST load the bundle your task needs (e.g. 'research' to search/fetch the "
            "web, 'file' to author a document, 'research_read' to read an already-fetched "
            "source) BEFORE you can use its tools. Load the right bundle FIRST, then work.",
            {"name": {"type": "string"}},
            [],
        )

    def _offered_tool_specs(
        self, only: Optional[Sequence[str]] = None
    ) -> list[dict[str, Any]]:
        """The tool schemas offered to the model THIS turn (d242 TRUE self-select):
        get_bundles (always) + the base finish + the tools of every bundle the node has
        SELF-SELECTED so far. Before any self-select this is EXACTLY {get_bundles, finish}
        — no domain tool is pre-offered; the node obtains its tools only by loading a
        bundle. Recomputed each turn so a freshly-loaded bundle's tools appear next turn.

        ``only`` (an optional allow-list of domain tool names) CURATES the loaded bundles'
        surface down to the subset THIS phase uses (the d212 'runtime selects the subset a
        phase needs' filter — e.g. a reviewer that loads 'file' but is offered file_read +
        file_update, never file_write). It NEVER pre-mounts a tool: a name in ``only`` whose
        bundle is not loaded yet simply does not appear. get_bundles + the base finish are
        always offered regardless of ``only``."""
        allow = set(only) if only is not None else None
        specs: list[dict[str, Any]] = [self._get_bundles_spec()]
        seen = {"get_bundles"}
        for spec in compose_tool_specs(self._loaded_bundles, self._node_run_ctx()):
            try:
                fname = spec["function"]["name"]
            except (KeyError, TypeError):
                fname = None
            if fname and fname in seen:
                continue
            # the base finish is always available; domain tools may be curated to ``only``.
            if fname and fname != "finish" and allow is not None and fname not in allow:
                continue
            if fname:
                seen.add(fname)
            specs.append(spec)
        return specs

    def _offered_tool_names(self, only: Optional[Sequence[str]] = None) -> tuple[str, ...]:
        """The names of the tools offered this turn — the ``accepted`` set the call
        parsers gate on (recomputed each turn as bundles load, curated to ``only``)."""
        return tuple(s["function"]["name"] for s in self._offered_tool_specs(only))

    async def _handle_self_select(
        self, tool: str, args: Mapping[str, Any]
    ) -> Optional[str]:
        """If ``tool`` is ``get_bundles``, run the self-select (LIST or LOAD) and return its
        observation string; else return None (not a self-select call → the loop dispatches
        its own domain tool). The observation tells the model exactly which tools it just
        unlocked so it acts on them next turn. Never raises — a self-select failure degrades
        to a 'list the bundles' nudge rather than crashing the node."""
        if tool != "get_bundles":
            return None
        handler = self._get_bundles_handler()
        name = args.get("name") if isinstance(args, Mapping) else None
        try:
            result = handler(name=name)
        except Exception as exc:  # noqa: BLE001 - self-select must never crash a node
            # Fact-only (CoT-autonomy P2): the failure + the no-args list contract
            # (get_bundles' own description carries its usage).
            return f"get_bundles failed: {exc}. Called with no arguments it lists the bundles."
        if "loaded" in result:
            loaded_tools = list(result.get("tools") or [])
            tools = ", ".join(loaded_tools) or "(none)"
            obs = f"Loaded bundle '{result['loaded']}'. Tools now available: {tools}."
            # PER-RUN DATA AT THE MOMENT OF RELEVANCE (P3/promptlab catch): the web
            # fetch budget is stated when the bundle providing the fetch tool loads —
            # never on every brief, where it nudged non-gather nodes into gathering.
            # Keyed on the TOOL the bundle provides (data), not a bundle-name switch.
            if self._fetch_tool in loaded_tools:
                cap = (
                    self._read_search_max_fetch
                    if self._read_search_max_fetch > 0 else RESEARCH_DEFAULT_FETCH_CAP
                )
                obs += f" Web fetch budget this task: {cap} sources."
            # d263: the bundle's DOCTRINE (its how-to: the research ReAct loop, the file
            # author/read doctrine, …) rides this LOAD observation ONCE — delivered in-band
            # when the node self-selects the bundle, carried forward by the convo window —
            # rather than re-pasted every turn as the retired pinned head did. (GET_BUNDLES
            # already promises 'you get back its doctrine'; this makes that true.)
            doctrine = str(result.get("doctrine") or "").strip()
            if doctrine:
                obs += f"\n\nHOW TO USE THIS BUNDLE:\n{doctrine}"
            return obs
        rows = result.get("bundles") or []
        listing = "; ".join(f"{r['name']} ({r['summary']})" for r in rows) or "(none)"
        err = result.get("error")
        prefix = f"{err}. " if err else ""
        # Fact-only (P2): catalog rows + the load syntax as a contract fact.
        return (f"{prefix}Bundles available — {listing}. "
                'get_bundles(name="<NAME>") loads one.')

    # How many turns a RAW-emission loop (file-delivery / synthesis) may spend SELF-SELECTING
    # its bundles before the write phase. Small — a node typically loads 1-2 bundles.
    _SELF_SELECT_FRONT_TURNS = 4

    async def _invoke_loaded_tool(
        self, tool: str, args: Mapping[str, Any]
    ) -> tuple[str, Any]:
        """Invoke a self-selected loaded tool → ``(observation_str, raw_value_or_None)``.

        The value-returning core of :meth:`_dispatch_loaded_tool`: a caller that only needs
        the observation uses the wrapper; the generic GATHER path (:meth:`_run_research_loop`,
        SA-4) needs the RAW value too so it can shape a downstream-pullable record. ``None``
        value on any unavailable/failed call (the observation still explains why)."""
        try:
            res = await self.hook.invoke(tool, **dict(args))
        except Exception as exc:  # noqa: BLE001 - a tool call must not crash the node
            return (f"'{tool}' failed: {exc}.", None)
        if not getattr(res, "ok", False):
            return (f"{tool} returned no usable result: {getattr(res, 'error', '')}.", None)
        return (f"{tool} result: {res.value}", res.value)

    async def _dispatch_loaded_tool(self, tool: str, args: Mapping[str, Any]) -> str:
        """Dispatch a tool the node SELF-SELECTED (its bundle registered it on the hook) → an
        observation string. GENERIC (d242 / as4): any loaded bundle's tool flows through the
        hook BY NAME, so a future domain-agnostic memory-read bundle (session/complex-memory,
        codebase, vector-db) needs NO loop change — it just registers more tools + widens
        :meth:`_node_run_ctx`. Never raises — an unavailable/failed tool degrades to a note."""
        obs, _ = await self._invoke_loaded_tool(tool, args)
        return obs

    @staticmethod
    def _gather_record(
        tool: str, args: Mapping[str, Any], value: Any
    ) -> dict[str, str]:
        """Shape one self-selected NON-WEB gather call into a generic source-like RECORD
        (SA-4/d254). Mirrors a web ``fetched`` entry — ``{title, url, markdown}`` — so a
        downstream reader/writer grounds in it through the SAME ``chain_sources`` harvest the
        web path uses (:func:`collect_fetched_sources_full`), with ZERO web semantics in the
        engine. Pure + source-agnostic: the ``url`` is a STABLE synthetic id (e.g.
        ``read_file://path``) so the writer's URL-dedup keys it; the body is whatever textual
        content the tool returned. Works for any bundle (codebase/vector-db/bash/future)."""
        text = ""
        ident = ""
        title = ""
        if isinstance(value, Mapping):
            for k in ("markdown", "text", "content", "body", "value"):
                v = value.get(k)
                if v:
                    text = str(v)
                    break
            ident = str(
                value.get("path") or value.get("url") or value.get("id") or ""
            ).strip()
            title = str(
                value.get("title") or value.get("name") or value.get("path") or ""
            ).strip()
        if not text:
            text = "" if value is None else str(value)
        if not ident:
            a = args if isinstance(args, Mapping) else {}
            ident = str(
                a.get("path") or a.get("url") or a.get("id") or a.get("query") or title or ""
            ).strip()
        if not ident:
            ident = (text[:48].strip() or tool)
        url = ident if "://" in ident else f"{tool}://{ident}"
        return {"title": title or ident or tool, "url": url, "markdown": text}

    # NOTE (SB-RR, d293): the former ``_run_linear_worker`` (the LINEAR/CHAT self-select loop)
    # is RETIRED — its job is subsumed by the UNIFIED worker loop (:meth:`_run_research_loop`),
    # which every hooked tool-less worker now enters. A trivial chat message is answered in ONE
    # emission there (the model selects no actionable bundle and writes its prose); a research
    # FOLLOW-UP self-selects ``research_read`` and answers from prior sources via the SAME loop
    # (whose generic non-web tool dispatch routes read_notes/load_source through
    # :meth:`_invoke_loaded_tool`).

    async def _self_select_front(self, system: Optional[str], *, suggest: str) -> None:
        """Run a bounded SELF-SELECT phase (d242) for a loop whose MAIN phase is RAW emission
        (file-delivery / synthesis): the node LOADS the bundle(s) it needs so its doctrine
        (and any read tools) arrive via self-select, NOT a prime. The raw write phase that
        follows offers the model no domain tools (it emits content; the loop drives the file
        tools), so this front is the loop's only self-select opportunity — its job is to pin
        the right doctrine before authoring.

        ``suggest`` is a PROMPT HINT naming the bundles this kind of node typically needs (a
        d186 framing lever, NOT a hardcoded load — the MODEL issues the get_bundles calls).
        Grows :attr:`_loaded_bundles`; never raises (a model that loads nothing degrades to
        the object floor, and raw emission still produces the deliverable)."""
        if self.hook is None:
            return
        convo: list[dict[str, Any]] = [{"role": "user", "content": (
            "Before you author the deliverable, SELF-SELECT your tools. " + suggest +
            " You start with only get_bundles + finish — call get_bundles(name=\"<NAME>\") "
            "once for EACH bundle you need, then reply READY to begin writing.")}]
        # Match the raw write phase's determinism (d35): temp=0, no inherited format schema,
        # so a self-select turn is consistent with the authoring turns that follow.
        opts = dict(self._call_opts)
        opts.pop("format", None)
        opts["temperature"] = 0
        for _ in range(self._SELF_SELECT_FRONT_TURNS):
            opts["tools"] = self._offered_tool_specs()
            accepted = self._offered_tool_names()
            raw, tool_calls = await self._research_emit(system, convo, opts)
            call = (first_native_call(tool_calls, accepted)
                    or self._parse_lightweight_call(raw, accepted))
            if call is None:
                break  # READY / prose → the node has loaded what it wants
            convo.append({"role": "assistant", "content": raw or (
                json.dumps({"tool": call[0], "args": call[1]}))})
            if call[0] == "get_bundles":
                obs = await self._handle_self_select(call[0], call[1])
                convo.append({"role": "tool", "content": obs or ""})
                continue
            break  # finish or any other reply → stop loading, proceed to write

    def _parse_lightweight_call(
        self, raw: str, accepted: Sequence[str]
    ) -> Optional[tuple[str, dict[str, Any]]]:
        """Recover a lightweight ``(tool, args)`` call for ANY name in ``accepted`` from a
        prose turn — the balanced-brace STRING fallback paired with :func:`first_native_call`
        for the self-select loops (so a non-native model can still call get_bundles or a
        loaded tool). Generic sibling of :meth:`_parse_research_call`; an unparseable or
        unknown object is treated as 'no call' (the turn is the model's content)."""
        s = _strip_synth_fence(raw or "").strip()
        if not s.startswith("{"):
            return None
        blob = _first_json_object(s)
        if not blob:
            return None
        try:
            parsed = json.loads(blob)
        except (ValueError, TypeError):
            try:
                parsed = json.loads(repair_model_json(blob))
            except (ValueError, TypeError):
                return None
        if not isinstance(parsed, Mapping):
            return None
        tool = parsed.get("tool") or parsed.get("name") or parsed.get("tool_name")
        args = parsed.get("args") or parsed.get("arguments") or parsed.get("parameters")
        accepted_set = set(accepted)
        if not (isinstance(tool, str) and tool.strip()):
            for key, val in parsed.items():
                if str(key).strip() in accepted_set:
                    tool, args = str(key).strip(), val
                    break
        name = str(tool).strip() if isinstance(tool, str) else ""
        if name not in accepted_set:
            return None
        if not isinstance(args, Mapping):
            args = {
                k: v for k, v in parsed.items()
                if k not in ("tool", "name", "tool_name", "args", "arguments", "parameters")
            }
        return name, dict(args)

    def _fetch_output_override(self) -> str:
        """The web_fetch output-message override from a LOADED bundle's context (d221).

        Scans the bundle(s) this node has self-selected for one that OVERRIDES the
        ``web_fetch`` observation (the research bundle prompts take-a-note); returns its
        suffix, or '' for a plain (non-research) context that loaded no such override.
        Never raises — an override-scan failure degrades to no suffix."""
        ctx = {"fetch_tool": self._fetch_tool}
        for name in getattr(self, "_loaded_bundles", ()):
            try:
                override = get_bundle(name).tool_output_override(self._fetch_tool, ctx)
            except Exception:  # noqa: BLE001 - an override scan must never break a fetch
                override = None
            if override:
                return override
        return ""

    def _node_history(
        self,
        convo: list[Mapping[str, Any]],
        system: Optional[str],
        *,
        compact: bool = True,
        reserve_predict: int = 0,
        keep_recent: int = 6,
    ) -> list[Mapping[str, Any]]:
        """Bounded history for a per-node multi-turn loop (d263 — pin/SWA-tail REMOVED).

        Returns the transport-ready history block to feed as ``Context(system=system,
        history=…)``. d263 retired the failed pinned-head + SWA-tail re-injection (which
        re-pasted the goal + bundle doctrine + task as always-in-view blocks on EVERY
        turn): the goal now rides the loop's FIRST turn (``convo[0]``, the
        :meth:`_compose_task` user turn) ONCE, the active doctrine arrives ONCE in the
        ``get_bundles`` load observation, and the SHAPING system rides ``Context(system=…)``
        ONCE — none of them re-pasted per turn. What remains here is the simple
        middle-turn compaction this module always provided (``Conversation.compact``):
        when ``compact`` is True and the window crosses num_ctx, the MIDDLE turns are
        folded into an offline summary while ``convo[0]`` (the goal/task turn) is KEPT
        verbatim and the most-recent ``keep_recent`` turns are preserved — so a long node
        never blows the window AND never loses its goal. When ``compact`` is False (a loop
        that already bounds its own convo, e.g. the write loop's :meth:`_bound_chat_convo`)
        the convo is returned unchanged.

        The summariser is the OFFLINE :func:`deterministic_summary` (no transport) on
        purpose: compaction here is non-blocking context hygiene (d4), so it never fires
        a live model round-trip inline on the event loop (the freeze hazard the rest of
        this module is careful to offload)."""
        history = list(convo)
        # compact=False, or too short to have any compactible middle (goal turn + recent),
        # → byte-identical passthrough.
        if not compact or len(history) <= keep_recent + 1:
            return history
        num_ctx = int(self._call_opts.get("num_ctx") or _DEFAULT_WRITE_NUM_CTX)
        reserve = int(reserve_predict) + 2048
        if system:
            reserve += estimate_tokens(system)
        threshold = max(1, num_ctx - reserve)
        # KEEP convo[0] (the goal/task turn) out of the compactible window — the goal-guard
        # the retired pin used to provide, now done by simply not folding the first turn —
        # and fold the MIDDLE of the rest via Conversation.compact (system=None, so its
        # .messages is [running summary, *recent]).
        head, rest = history[:1], history[1:]
        conv = Conversation(
            summarizer=deterministic_summary,
            compaction_threshold=threshold,
            keep_recent=keep_recent,
            auto_compact=False,
        )
        conv.extend(rest)
        conv.maybe_compact()
        return head + conv.messages

    def _scoped_source_block(self, *, full_index: bool = False) -> str:
        """This node's sources, fed by ROLE (d170 calibration — bounded-but-NOT-starved).

        The deep-research write phase has TWO source-consumer roles with OPPOSITE feeds,
        because of the d49 reality (the raw-file WRITER cannot emit tool calls, so it CANNOT
        pull source text on demand — only the REVIEWER, which dispatches ``load_source``, can):

        * **WRITER** (``full_index=False``) — PUSH the FULL figure-bearing BODIES of this
          section's assigned sources (``render_scoped_sources``: whole-doc summary + a
          figure/date-rich verbatim excerpt per source). This is the good-run calibration
          (trace bc7cef17: per-section writer inputs 20-39KB, ~80 figures). The d167 compact
          ``load_source``-nudge LEADS (~2400 chars/src) STARVED the writer to a thin report —
          same thinness as the d164 137KB over-feed, opposite cause. Now that ``max_sources``
          caps the run and the planner SCOPES each section to ~2-4 sources, the full bodies of
          those few sources are ~20-40KB — which FITS num_ctx un-truncated. The per-source
          budget is the writer budget, and the TOTAL is capped to a fraction of the window so
          even a many-source section stays bounded (thinner per source, never truncated). An
          UNSCOPED single-section writer feeds ALL (``max_sources``-capped) sources the same
          way. The d162 invariant holds by CONSTRUCTION: total source text <= window fraction.

        * **REVIEWER** (``full_index=True``) — the COMPACT full INDEX MAP only (every ``[S#]``
          for citation resolution, no bodies); it reads the live file by bounded region and
          PULLS any source's verbatim text on demand via ``load_source`` (it CAN call tools).

        Empty string when the run has no sources (caller degrades to the legacy path)."""
        if not self._chain_sources:
            return ""
        ids = [i for i in (self.node.source_ids or []) if isinstance(i, int)]
        if full_index:
            # REVIEWER: compact full INDEX (map for [S#] resolution) — bodies pulled on demand.
            block = render_scoped_source_index(
                self._chain_sources, ids,
                section_topic=self.node.task or "", full_index=True,
            )
            if block:
                block += (
                    "\n\nIf load_source returns a 'BUDGET REACHED' note, finish from what you "
                    "have already loaded; cite only the [S#]/URLs shown; do not placeholder a "
                    "citation you could not load."
                )
            return block
        # WRITER (scoped per-section, or unscoped single-section): PUSH full figure-bearing
        # bodies. Scoped → its 2-4 assigned ids; unscoped → all sources (max_sources-capped).
        feed_ids = ids if ids else list(range(1, len(self._chain_sources) + 1))
        if not feed_ids:
            return ""
        num_ctx = int(self._call_opts.get("num_ctx") or _DEFAULT_WRITE_NUM_CTX)
        # TOTAL source-text budget = a fraction of the char envelope; split across the feed
        # sources but capped per source at the writer budget — so the typical 2-4-source
        # section gets near-full bodies (~good-run 20-40KB) while a many-source section stays
        # bounded (the d162 no-truncation guarantee is structural, not an app cap).
        total_cap = int(num_ctx * _CHARS_PER_TOKEN * _WRITE_SOURCE_WINDOW_FRACTION)
        per_source = max(2500, min(resolve_writer_source_budget(), total_cap // len(feed_ids)))
        return render_scoped_sources(
            self._chain_sources, feed_ids,
            excerpt_budget=per_source, section_topic=self.node.task or "",
        )

    # SoC ENGINE-THIN (SA-5/d254): the web URL/article/readability semantics
    # (``NON_ARTICLE_EXT`` / ``looks_like_article_url`` / ``url_offered`` /
    # ``is_readable_fetch``) used to live HERE. They now belong to the WEB bundle
    # (``bundles.web_ingest``) — the engine imports the ``url_offered`` grounding predicate
    # for the loop's defense-in-depth guard and DELEGATES the rest via ``_web_adapter``.

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
        # PER-NODE bounded memory (d263): window the research ReAct convo with the simple
        # middle-turn compaction — the goal/task rides convo[0] ONCE and the doctrine rides
        # the get_bundles load observation ONCE (the retired pinned head + SWA tail no longer
        # re-paste them every turn); convo[0] is kept verbatim and older middle turns fold
        # once the window crosses num_ctx, so a long node never blows the window or loses its
        # goal. This is the canonical long multi-turn path the compaction subsystem drives.
        history = self._node_history(
            convo, system, compact=True,
            reserve_predict=int(opts.get("num_predict", 0) or 0),
        )
        ctx = Context(system=system, history=history, transport=self.transport)
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
            # d222 — retry after repairing E4B's common malformations (illegal \' escape /
            # stray special token). The trace showed a research NOTE dropped TWICE to a
            # ``\'`` escape — this recovers it so the note lands instead of being re-asked.
            try:
                parsed = json.loads(repair_model_json(blob))
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

        NOTE (d242): NOT wired into the loop anymore — under TRUE self-select the research
        tools are offered ONLY after the node loads the 'research' bundle, via
        :meth:`_offered_tool_specs` → ``compose_tool_specs``. Retained as the reference for
        the per-phase native schema construction (the configured-name binding + ``accepted``
        filtering); it is NOT a tool-pre-offer path.

        DELEGATES to the ResearchBundle (d190/d212) — the bundle is the single source of
        the schemas. The bundle exposes ONE ``tool_specs(ctx)`` catalog (no role-phase
        method, d212 #2); the runtime SELECTS the subset it offers this phase by filtering
        to exactly the ``accepted`` names, in ``accepted`` order — so the research ReAct
        loop is byte-identical to the prior inline construction."""
        bundle = get_bundle("research")
        ctx = {
            "search_tool": self._search_tool,
            "fetch_tool": self._fetch_tool,
            "note_tool": self._note_tool,
            "emit_notes": True,
        }
        catalog = {s["function"]["name"]: s for s in bundle.tool_specs(ctx)}
        return [catalog[name] for name in accepted if name in catalog]

    async def _dispatch_research_tool(
        self, tool: str, args: Mapping[str, Any],
        fetched: list[dict[str, str]], seen_urls: set[str],
        offered_urls: Optional[set[str]] = None,
    ) -> str:
        """DELEGATE one model-chosen WEB tool call to the web bundle's gather adapter.

        SoC ENGINE-THIN (SA-5/d254): the web_search/web_fetch dispatch + all URL/article/
        readability/record/coverage-note semantics are OWNED by the WEB bundle now
        (:class:`~agent_runtime.bundles.web_ingest.WebGatherAdapter`). The engine no longer
        hardcodes any of it — it hands the adapter THIS run's hook ``invoke`` closure, the
        ``read_fetched`` read closure (which still holds the engine's embedder + budgets), and
        the bundle-sourced web_fetch take-a-note suffix, and the adapter fires the web logic +
        appends readable sources to ``fetched``. Behaviour is byte-identical to the prior
        engine method (the served web path is the contrastive byte-comparable gate); only the
        OWNER moved from the engine into the bundle that owns the web tools."""
        return await self._web_adapter.dispatch(
            tool, args,
            invoke=self.hook.invoke,
            fetched=fetched,
            seen_urls=seen_urls,
            offered_urls=offered_urls,
            read_fetched=self._read_fetched,
            emit_article_notes=self._emit_article_notes,
            fetch_note_suffix=self._fetch_output_override(),
        )

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
        # d157: bound each per-source research read to a compact relevant CHUNK
        # (``RESEARCH_READ_CHUNK_CHARS``) so a multi-turn react loop's ACCUMULATED input stays
        # small and per-node latency drops — the full verbatim body is still stored for the
        # writer's citation / load_source path, and the relevance-select picks the MOST
        # relevant passages so the cap trims raw bulk, not the useful content.
        return max(self._fetched_char_budget, min(per_source, total, RESEARCH_READ_CHUNK_CHARS))

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
            # Fact-only (P2): the state; a note's contract lives in its description.
            return (
                "Note not recorded: no source has been read this task — a note "
                "records a source you actually read."
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
            return "Note not recorded: malformed arguments."
        notes.append(note.model_dump())
        followups = "; ".join(note.gaps_or_followups) or "none noted"
        # Fact-only (P2): the recorded state + coverage counts + this note's gap lane.
        # What findings should look like is the research spec/doctrine's knowledge.
        noted_ids = {int(n.get("source_id") or 0) for n in notes}
        return (
            f"Note recorded for <{note.title or note.url}> (trust: {note.source_trust}) — "
            f"{len(noted_ids)} of {len(fetched)} read source(s) now have notes. "
            f"This note's open follow-ups: {followups}."
        )

    async def _run_research_loop(self, inputs: Mapping[str, Any]) -> "SubAgentResult":
        """The UNIFIED WORKER LOOP (SB-RR, d273/d292/d293) — every hooked tool-less WORKER.

        ONE self-select ReAct loop now serves what used to be three separate paths: the
        GATHER node (the retired ROLE_RESEARCHER), the LINEAR/chat worker, and the bare
        producer. The node starts TOOL-LESS and the WORKER ITSELF decides — driven by its
        SPECIALIZATION, never a role — which bundle(s) to self-select, and the SELECTED bundle
        determines behavior:

        * a node carrying the research-METHODOLOGY spec self-selects the ``research`` bundle
          and GATHERS — each turn it emits a lightweight tool call
          (``{"tool":"web_search","args":{"query":...}}`` / ``web_fetch`` — small args the
          small model emits reliably, d49) OR its FINDINGS as RAW prose (no ``format=<schema>``,
          d50.1). ``read_search_max_fetch`` survives ONLY as the fetch CAP (a cost/safety bound),
          never a flow gate; the GATHER-specific gates below key on the SELF-SELECTED research
          bundle, not a role.
        * a research FOLLOW-UP self-selects ``research_read`` and answers from prior sources.
        * a TRIVIAL node self-selects NO actionable bundle and writes its answer in ONE
          emission (the old single-shot worker — its prose IS the output).

        The loop EXECUTES each call against the real hook, feeds the REAL observation back, and
        ends when the model writes its prose (or a NON-FLOW bound trips). Readable sources the
        agent chose to read are attached to ``tool_value`` as ``{"fetched": [...]}`` so a
        downstream node grounds in the real article text (d17, :meth:`_render_tool_value`); a
        non-gathering worker attaches none. The prose is the node ``output``; the result-
        validator gate runs as in :meth:`run`."""
        # NODE-SELF-SELECT (d242 — TRUE self-select): the worker starts TOOL-LESS (only
        # get_bundles + finish). To GATHER it MUST self-select the 'research' bundle (search/
        # fetch/note + the web_fetch take-a-note override) and, to read a fetched source,
        # 'research_read'; to answer from prior session research it self-selects 'research_read'
        # alone; a trivial worker self-selects nothing. There is NO role->bundle prime and NO
        # pre-mounted domain tool; the tools + doctrine + ctx all arrive on LOAD (the catalog is
        # advertised in the system prompt). The offered tools and the ``accepted`` set are
        # recomputed each turn from the bundles loaded so far (:meth:`_offered_tool_specs`),
        # exactly as the planner self-selects.
        system = self._compose_system()
        # The fetch CAP is a NON-FLOW bound: the agent decides WHETHER/WHICH to fetch;
        # the cap only bounds total fetches. A caller that wired none (0) gets a sane
        # default so the agent can still read sources.
        fetch_cap = (
            self._read_search_max_fetch if self._read_search_max_fetch > 0
            else RESEARCH_DEFAULT_FETCH_CAP
        )
        # Reasoned tool calls survive without a wire schema (d34); strip any inherited
        # format so a turn is free to be a tool call OR raw findings. The tool surface is
        # set PER TURN inside the loop (it grows as the node self-selects), not once here.
        opts = dict(self._call_opts)
        opts.pop("format", None)
        accepted: tuple[str, ...] = ()

        # d229 DOCTRINE DE-DUP (token hygiene; NO flags): the research DOCTRINE
        # (``RESEARCH_LOOP_INSTRUCTION`` — the 'Workflow: search…' loop how-to) is carried
        # ONCE, in the ``get_bundles`` LOAD observation the model gets when it self-selects the
        # 'research' bundle (the bundle's ``own_doctrine``, carried forward by the convo
        # window). d229/d263: it is NEVER re-pasted into the per-turn task message — doing so
        # duplicated the same ~400-500 tokens of doctrine in every prompt (trace forensic
        # 620d38fa, ~11-14% of early-turn tokens = PURE duplication), and d263 retired the
        # pinned-head copy that re-injected it every turn. The per-turn task therefore carries
        # ONLY the node task plus the per-RUN operational bits the shared doctrine cannot hold:
        # the CONCRETE fetch CAP (the bundle doctrine shows the literal ``{fetch_cap}``
        # placeholder, since the bundle text is shared and run-agnostic), and the article-note
        # clause when note emission is enabled (N2 — a separate operational instruction).
        base_user = self._compose_task(inputs, None)
        # CoT-autonomy P3: the operational suffix is RUN DATA only. The bundle-selection
        # script (the gather-first tool sequence it used to dictate) and the
        # article-note clause are DELETED — how to
        # operate lives in the OPERATING PROTOCOL (system turn), which bundles serve
        # what in the catalog advert, and note discipline in the note tool description +
        # research doctrine. What remains is the declared deliverable file (delivery
        # data, d299 DP2 — a fact, never a load-this-then-do-that sequence). The
        # concrete FETCH CAP now rides the research-bundle LOAD ack instead (delivered
        # at the moment of relevance — a live promptlab catch: naming a web-fetch
        # budget on every brief nudged WRITE nodes into gathering).
        operational = ""
        if self._deliverable_path and self._deliverable_path not in (self.node.task or ""):
            operational = f"DELIVERABLE FILE: '{self._deliverable_path}'."
        convo: list[dict[str, Any]] = [{
            "role": "user",
            "content": (base_user + "\n\n" + operational) if operational else base_user,
        }]
        fetched: list[dict[str, str]] = []
        notes: list[dict[str, Any]] = []  # per-article CONTROL notes (N2; additive)
        # SoC ENGINE-THIN (SA-4/d254): a NON-WEB gather node's generic source-like artifacts
        # (one per self-selected loaded-tool call), shaped by :meth:`_gather_record`. Mirrors
        # ``fetched`` for a different source; harvested into the writer's chain_sources by
        # :func:`collect_fetched_sources_full`. Empty on the web path (byte-identical).
        records: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        # s15/a25 LEVER 3 (d186): the REAL URLs web_search surfaced this node — the grounding
        # set a web_fetch url is validated against. The small model fabricates plausible-but-dead
        # URLs instead of copying a real one; an un-offered fetch returns a role:tool error so the
        # model RE-GROUNDS (robust tool feedback, not a hard seatbelt).
        offered_urls: set[str] = set()
        searches = fetches = unproductive = 0
        # CoT-autonomy P3: the gather-more / note-gate / target-artifact bounce state is
        # DELETED with the gates. ``wrote_target`` survives as trace DATA only (whether
        # a write-shaped result landed on the declared target — read by the span attrs).
        pending_findings = ""
        findings = ""
        wrote_target = False
        # MALFORMED-CALL channel feedback bound (CoT-autonomy P2).
        malformed_calls = 0

        # The turn ceiling rises proportionally with the fetch cap so a high-breadth
        # gather can read MANY sources without the flat ceiling clipping it (N1). A
        # narrow legacy cap keeps the original RESEARCH_MAX_TURNS floor (no change). When
        # article notes are on (N2), allow up to one extra (note) turn per fetch so the
        # note turns do not starve the fetch budget — still a NON-FLOW ceiling, and OFF
        # (note_budget=0) it is the exact N1 formula.
        note_budget = fetch_cap if self._emit_article_notes else 0
        # +2 SELF-SELECT headroom (d242): the node spends a turn (or two) loading its
        # bundle(s) before it gathers, so the ceiling must not clip the gather budget.
        max_turns = max(
            RESEARCH_MAX_TURNS, fetch_cap + RESEARCH_SEARCH_HEADROOM + note_budget
        ) + 2
        # DELIVERABLE headroom (autonomy rebuild P2): a node writing a declared file
        # spends turns on pulls (read_notes/load_source) plus one file_write per part —
        # the gather-sized ceiling would clip a whole-document write loop. Data-gated
        # on the declared target; still a NON-FLOW cap.
        if self._deliverable_path:
            max_turns += 10

        tracer = get_tracer("agent_runtime.research")
        with tracer.start_as_current_span("research.react") as span:
            span.set_attribute("research.node", str(self.node.id))
            span.set_attribute("research.fetch_cap", fetch_cap)
            span.set_attribute("research.max_turns", max_turns)
            for turn in range(max_turns):
                # d242 TRUE self-select: offer get_bundles + finish + the tools of every
                # bundle loaded so far; recompute each turn so a freshly-loaded bundle's
                # tools become callable next turn. Before any load it is only get_bundles +
                # finish — no gather tool is pre-mounted.
                opts["tools"] = self._offered_tool_specs()
                accepted = self._offered_tool_names()
                raw, tool_calls = await self._research_emit(system, convo, opts)
                # s13: prefer the NATIVE tool call (it rides its own channel, so leading
                # prose can never swallow it); fall back to the balanced-brace string parser
                # for a non-native reply (the research parser for gather calls, then the
                # generic lightweight parser for get_bundles/finish). NEITHER → the model
                # wrote its FINDINGS as prose.
                call = (first_native_call(tool_calls, accepted)
                        or self._parse_research_call(raw)
                        or self._parse_lightweight_call(raw, accepted))
                convo.append({"role": "assistant", "content": raw or (
                    json.dumps({"tool": call[0], "args": call[1]}) if call else "")})
                # SELF-SELECT turn: a get_bundles call LOADS a bundle (its tools appear next
                # turn); it is not a gather action, so handle + continue. Fed role:'tool' —
                # the transport renders it as an ENVELOPED user turn ([TOOL RESULT]…), so the
                # model both SEES it (d199's intent) and can tell it is a tool observation,
                # not the user speaking (the messaging-layer fix).
                if call is not None and call[0] == "get_bundles":
                    obs = await self._handle_self_select(call[0], call[1])
                    span.set_attribute(f"research.turn.{turn + 1}.tool", "get_bundles")
                    convo.append({"role": "tool", "content": obs or ""})
                    continue
                # A finish call is the model signalling done → fold into the 'no tool call'
                # path so the findings-acceptance logic runs (research output is RAW prose).
                # PARSE-TO-READ (Gate-2f wart): when the finish call carries a textual
                # reason/summary, THAT model-authored text is the node's findings — not the
                # raw tool-call JSON syntax it rode in on (which leaked verbatim into the
                # node output on the live run). Extraction only; no engine wording.
                if call is not None and call[0] == "finish":
                    _fin_args = call[1] or {}
                    _fin_text = str(
                        _fin_args.get("reason") or _fin_args.get("summary")
                        or _fin_args.get("result") or ""
                    ).strip()
                    if _fin_text:
                        raw = _fin_text
                    call = None
                # TRUE self-select guard (d242): a gather tool the node has NOT yet loaded is
                # not callable — the string parser can name one before its bundle is loaded.
                # Fact-only (P2): state what is loaded; the protocol + catalog own the how.
                if call is not None and call[0] not in accepted:
                    span.set_attribute(f"research.turn.{turn + 1}.tool", "unloaded")
                    convo.append({"role": "tool", "content": (
                        f"'{call[0]}' is not loaded — no loaded bundle provides it. "
                        f"Currently loaded tools: {', '.join(accepted) or '(none)'}.")})
                    continue
                # MALFORMED TOOL CALL (CoT-autonomy P2, live promptlab catch): a reply
                # that LOOKS like a tool call ({"tool": …) but parses as nothing must
                # not be silently accepted as findings — that is the engine MISREADING
                # the model (a 9KB file_write with one escaping slip became "findings"
                # and the write was lost). Report the real parse error as a TOOL fact —
                # channel feedback, not a next-action command; the model decides how to
                # recover (retry, smaller parts, different form). Bounded: after
                # _MALFORMED_CALL_MAX the reply stands as prose (never an infinite loop).
                # LENIENT RECOVERY first (P6): an unambiguous tool-shaped reply whose
                # big content string broke strict JSON is recovered verbatim — the
                # model's own bytes dispatch instead of dying to a parse error.
                if call is None and '"tool"' in (raw or "").lstrip()[:40]:
                    _len_call = _lenient_content_call(raw or "")
                    if _len_call is not None and _len_call[0] in accepted:
                        call = _len_call
                        span.set_attribute(
                            f"research.turn.{turn + 1}.lenient_parse", True
                        )
                if call is None and '"tool"' in (raw or "").lstrip()[:40]:
                    if malformed_calls < _MALFORMED_CALL_MAX:
                        malformed_calls += 1
                        try:
                            json.loads(raw)
                            perr = "parsed as JSON but not as a {tool, args} call"
                        except Exception as exc:  # noqa: BLE001 — the error IS the data
                            perr = f"{type(exc).__name__}: {str(exc)[:140]}"
                        span.set_attribute(
                            f"research.turn.{turn + 1}", "malformed_call"
                        )
                        convo.append({"role": "tool", "content": (
                            "That reply was NOT dispatched: it looks like a tool call "
                            f"but is not valid JSON ({perr})."
                        )})
                        continue
                if call is None:
                    # No tool call → the model wrote its FINDINGS (RAW prose) = done.
                    findings = _strip_synth_fence(raw or "").strip()
                    # CoT-autonomy P3 (owner ruling — no babysitting): the three bounce-
                    # gates that re-prompted a conclusion here (no-fab GATHER-MORE, the
                    # NOTE GATE, the TARGET-ARTIFACT gate) are DELETED. The model's own
                    # conclusion stands; a postcondition failure surfaces HONESTLY
                    # downstream instead (the persistence-side staleness guard ships no
                    # stale artifact, deliverable_bytes stays truthful, and the reviewer
                    # node reads the real file). The burden moved to the operating
                    # protocol + tool descriptions + specs, iterated in promptlab.
                    if findings:
                        span.set_attribute(f"research.turn.{turn + 1}", "findings")
                        break
                    unproductive += 1
                    if unproductive >= 2:
                        break
                    # ONE fact-only recovery line for an unusable turn (P1).
                    convo.append({"role": "user", "content": _UNUSABLE_TURN_NOTE})
                    continue
                tool, args = call
                if tool == self._note_tool:
                    # A CONTROL-note turn (N2): record the per-article note and feed back
                    # an ack that surfaces its follow-ups to steer the next search. Does
                    # NOT touch the fetch/search budget (it is not a gather call).
                    obs = self._record_article_note(args, fetched, notes)
                    span.set_attribute(f"research.turn.{turn + 1}.tool", "note")
                    # Observation fed role:'tool' — the transport renders it as an ENVELOPED
                    # user turn ([TOOL RESULT]…), so the model grounds on the ack (d199's
                    # intent) AND can tell it is tool output, not the user (messaging fix).
                    convo.append({"role": "tool", "content": obs})
                    continue
                # SoC ENGINE-THIN (SA-4/d254 — the Tier-2 gap fix): a self-selected gather tool
                # that is NOT the configured web search/fetch/note is a NON-WEB bundle's tool
                # (codebase / vector-db / bash / future). The engine does NOT hardcode its
                # semantics or mis-dispatch it as a web_fetch — it FALLS THROUGH to the GENERIC
                # by-name on-load hook dispatch (:meth:`_invoke_loaded_tool`), captures a generic
                # source-like RECORD so a downstream reader/writer pulls it via the SAME
                # chain_sources harvest web ``fetched`` uses, feeds the observation back, and
                # continues. The web search/fetch branch below is UNREACHED for a non-web tool,
                # so it stays byte-identical (contrastive gate, SA-4 (a)).
                if tool not in (self._search_tool, self._fetch_tool, self._note_tool):
                    obs, value = await self._invoke_loaded_tool(tool, args)
                    # Capture a downstream-pullable RECORD only for a READ-shaped value
                    # (it carries content). A WRITE-shaped result — {path, bytes, …},
                    # e.g. the pull-writer's file_write acks — is an action receipt,
                    # not a source; recording it would pollute chain_sources with
                    # pseudo-sources. Shape-based (data), never a tool-name gate.
                    write_shaped = (
                        isinstance(value, Mapping)
                        and "bytes" in value
                        and not (value.get("markdown") or value.get("text"))
                    )
                    if write_shaped and self._deliverable_path:
                        _wp = str((args or {}).get("path") or "").strip()
                        if _wp and Path(_wp).name == Path(self._deliverable_path).name:
                            wrote_target = True
                    if value is not None and not write_shaped:
                        records.append(self._gather_record(tool, args, value))
                    span.set_attribute(f"research.turn.{turn + 1}.tool", tool)
                    # Observation fed role:'tool' → transport-enveloped ([TOOL RESULT]…),
                    # mirroring the web search/fetch observation feed (messaging fix).
                    convo.append({"role": "tool", "content": obs})
                    continue
                if tool == self._fetch_tool:
                    # s15/a25 grounding guard (d186; d199-consistent): VALIDATE the chosen url is
                    # one web_search actually offered. The small model invents plausible-but-dead
                    # URLs (live: every fetch failed, 0 sources); when it targets a url NOT in
                    # ``offered_urls`` (and the node HAS been offered some), return an ERROR
                    # observation listing the real candidates so it RE-GROUNDS and retries — robust
                    # single-retry actionable feedback (a TOOL error result, so fed role:'tool' →
                    # transport-enveloped), NOT a hard seatbelt; it does NOT burn the fetch cap
                    # (a fabricated attempt never reached the network). Bounded by ``max_turns``.
                    want = str(args.get("url") or args.get("link") or "").strip()
                    if offered_urls and want and not _url_offered(want, offered_urls):
                        span.set_attribute(
                            f"research.turn.{turn + 1}.tool", "fetch_ungrounded"
                        )
                        cand = "\n".join(f"- {u}" for u in list(offered_urls)[:8])
                        convo.append({"role": "tool", "content": (
                            f"web_fetch refused <{want}> [ungrounded_url]: this URL was "
                            "not returned by any search this task; only returned URLs "
                            f"load. URLs actually returned include:\n{cand}"
                        )})
                        continue
                    if fetches >= fetch_cap:
                        convo.append({"role": "tool", "content": (
                            f"web_fetch refused [fetch_limit]: {fetch_cap} of {fetch_cap} "
                            "fetches used this task; no further fetch will execute."
                        )})
                        continue
                obs = await self._dispatch_research_tool(
                    tool, args, fetched, seen_urls, offered_urls
                )
                if tool == self._search_tool:
                    searches += 1
                else:
                    fetches += 1
                span.set_attribute(f"research.turn.{turn + 1}.tool", tool)
                # The web_search RESULTS and web_fetch BODY are observations the model must
                # GROUND on. Fed role:'tool': the transport renders them as ENVELOPED user
                # turns ([TOOL RESULT]…[/TOOL RESULT]) so the prompt-only model both SEES
                # them (d199's grounding intent — a bare role:'tool' turn is invisible to
                # gemma's template) AND can tell tool output from the user speaking (the
                # messaging-layer fix). In-memory history keeps the semantic tool label.
                convo.append({"role": "tool", "content": obs})

            # Fallback: the model never wrote findings (kept calling tools / stalled) →
            # ONE final emission after the turn-budget FACT (P3: the note states only
            # that the loop is ending; the spec owns what a conclusion looks like).
            if not findings and pending_findings:
                findings = pending_findings
            if not findings:
                convo.append({"role": "user", "content": (
                    _RESEARCH_FINALIZE if BUNDLE_RESEARCH in self._loaded_bundles
                    else _WORKER_FINALIZE
                )})
                final_raw, _ = await self._research_emit(system, convo, opts)
                findings = _strip_synth_fence(final_raw or "").strip()
            span.set_attribute("research.searches", searches)
            span.set_attribute("research.fetches", fetches)
            # SA-4 (the as3 lesson): SOURCES counts BOTH web fetched AND non-web records, so a
            # non-web gather's leaf-capture is visible in the trace (fetches==0 with records>0
            # is a real gather, not a regression); ``research.records`` is the distinct count.
            span.set_attribute("research.sources", len(fetched) + len(records))
            span.set_attribute("research.records", len(records))
            span.set_attribute("research.notes", len(notes))
            span.set_attribute("research.chars", len(findings))

        # Attach the read sources so a downstream node grounds in them (d17). The
        # findings prose is the node output (RAW, d50.1). The per-article CONTROL notes
        # (N2) ride ADDITIVELY alongside ``fetched`` — the c13 write-side path reads only
        # ``fetched``/``fetched_count`` (UNCHANGED); the notes lane DIRECTS the next
        # research node (N4) and weights provenance (N5). No notes (default OFF, or the
        # model emitted none) → no ``article_notes`` key → byte-identical to before.
        tool_value: Any = None
        if fetched or records:
            tv: dict[str, Any] = {}
            if fetched:
                tv["fetched"] = fetched
                tv["fetched_count"] = len(fetched)
                if notes:
                    tv["article_notes"] = notes
            # GENERIC records-emission (SA-4/d254): a non-web gather attaches its artifacts
            # under the source-agnostic ``records`` key (mirrors web ``fetched``) so the
            # writer's chain_sources harvest pulls them the SAME way. When only ``fetched`` is
            # present (the web path) ``tv`` is byte-identical to the prior dict (no records key).
            if records:
                tv["records"] = records
            tool_value = tv
        result = SubAgentResult(
            node_id=self.node.id,
            spec=self.node.primary_spec,
            specs=self.spec_names,
            output=findings,
            # SB-RR (d293): only report the search tool as used when the worker ACTUALLY
            # gathered (fetched/recorded a source). A trivial/follow-up worker that selected no
            # gather bundle reports the node's own tool (``None``), not a phantom search.
            tool_used=self._search_tool if (fetched or records) else self.node.tool,
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
        # CoT-autonomy P1 — THE OPERATING PROTOCOL rides the SYSTEM turn once per
        # role-carrying node (its single owner): a node that has not yet loaded any
        # bundle still knows the reason→act→observe loop and the reply channel. The
        # model's reasoning sequences the work; the protocol states only the channel.
        protocol = AGENT_OPERATING_PROTOCOL if role else ""
        # NODE-SELF-SELECT (d221): advertise the bundle catalog in every ROLE-carrying
        # node's system prompt (the in-plan agent roles: researcher/worker/reviewer +
        # the terminal synthesizer) so the model can REASON about which capability
        # bundle(s) its task needs and LOAD them at runtime with get_bundles (the planner
        # sets only role + spec; the node selects its own tools). A role-less plain
        # producer step is not an agent that self-selects tools, so it is left untouched
        # (back-compat). Appended LAST so it sits closest to the task.
        catalog = self._bundle_catalog_advert() if role else ""
        if not self.spec_body:
            # bare step (identity only) or bare-role call (identity + protocol + framing).
            body = "\n\n".join(p for p in (protocol, framing, catalog) if p) or None
            return with_identity(body)
        base = f"{_SHAPING_FRAMING}\n\n{self.spec_body}"
        if protocol:
            base = f"{protocol}\n\n{base}"
        shaping = f"{base}\n\n{framing}" if framing else base
        if catalog:
            shaping = f"{shaping}\n\n{catalog}"
        return with_identity(shaping)

    def _bundle_catalog_advert(self) -> str:
        """The 'List of bundles:' advertisement embedded in every node prompt (d221).

        Delegates to :func:`agent_runtime.bundles.bundles_catalog_text` (single source
        of truth) so the catalog the model reasons over stays in lockstep with the
        registered bundles. Never raises — a catalog failure degrades to no advert."""
        try:
            return bundles_catalog_text().strip()
        except Exception:  # noqa: BLE001 - the advert must never break a node
            return ""

    async def run(self, inputs: Optional[Mapping[str, Any]] = None) -> "SubAgentResult":
        """Execute the node: optional tool call, then a scoped phi call.

        A failed tool call raises :class:`ToolFailureError`; a result the
        validator rejects raises :class:`InvalidStepError` (both self-heal
        classes). A transport-level error propagates unchanged."""
        inputs = inputs or {}
        # AUTONOMY REBUILD Phase 2 (owner charter): the d299 `deliverable_path` routing
        # flag and the legacy explicit-file_write raw-loop route are DELETED. A write
        # node is a WORKER like any other: it enters the unified self-select loop below,
        # loads the `file` (+ `research_read`) bundles itself, PULLS its grounding
        # (read_notes → load_source) and DRIVES file_write/file_update per its
        # specialization — the live probe measured E4B emitting 8/8 well-formed
        # section-sized file_write tool calls, falsifying the d49 premise the raw push
        # loop was built on. The deliverable target is TASK DATA (the planner names the
        # file in the node task); `sanitize_write_path` at the tool boundary remains the
        # only path guard. No role/spec/tool-name conditional routes anything.
        # UNIFIED WORKER LOOP (d273/d292/d293 — SB-RR retires ROLE_RESEARCHER): every spawned
        # node is a WORKER (d273); gather is a SELF-SELECTED specialization, never a role. A
        # WORKER that did not bind a terminal tool (the common case: gather nodes, research
        # follow-ups, and plain producers are all tool-less) enters ONE self-select loop
        # (:meth:`_run_research_loop`) and the SELF-SELECTED BUNDLE drives behavior — load
        # ``research`` to GATHER (search/fetch/note), ``research_read`` to read prior sources,
        # or select NO actionable bundle to just answer in a single emission (the old
        # trivial/linear/bare-producer path folds in here). There is NO role gate: a node is
        # not branched on its worker-ness (d273), and the retired ROLE_RESEARCHER /
        # ROLE_WORKER-routing / role=None producer branches are all collapsed into this one
        # loop. The gather-specific gates inside the loop key on the SELF-SELECTED research
        # bundle, not a role. A legacy explicit ``web_search`` node folds in (tool ==
        # search_tool). Offline (no hook) callers and explicit non-search single-tool nodes fall
        # through to the bounded produce path below. (Terminal SYNTHESIZER delivery (d215) is
        # the ONLY structural role dispatch that survives — handled above/below.)
        if (
            self.hook is not None
            and (self.node.tool == self._search_tool or not self.node.tool)
        ):
            # SYNTHESIZER FOLD (autonomy rebuild P2C): the terminal delivery role no
            # longer routes to the raw push loop (_run_synthesis → _run_raw_file_loop —
            # both deleted) — it is a WORKER in the same unified self-select loop as
            # everyone else: it loads the file (+ research_read) bundles itself and
            # DRIVES file_write per its spec. Its target resolves exactly as
            # _run_synthesis resolved it (the c1b continuation path, else the derived
            # name) and rides as DELIVERY DATA (deliverable_path + the operational
            # delivery line, d299 DP2) — the target-artifact gate verifies delivery.
            if self.node.role == ROLE_SYNTHESIZER and not self._deliverable_path:
                self._deliverable_path = self._chain_continue_path or derive_output_path(
                    self._overall_goal, self.node.task, self.spec_names
                )
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
                    # The run's overall USER GOAL rides along (live catch: a node task
                    # that drops a user-stated detail — a time, a recipient — otherwise
                    # leaves the arg model anchoring on a schema example).
                    emitted = self._tool_arg_emitter(
                        self.node,
                        inputs=inputs,
                        tool_values=self._upstream_tool_values,
                        goal=self._overall_goal,
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
        # SYNTHESIZER FOLD (autonomy rebuild P2C): the ROLE_SYNTHESIZER special case
        # (_with_source_index → _run_synthesis → _run_raw_file_loop, the engine-steered
        # raw push loop with its riders/flags) is DELETED. A hooked tool-less synthesizer
        # routes to the unified self-select loop ABOVE with its target as deliverable
        # DATA; an offline or single-tool synthesizer produces prose here like any node.
        #
        # PRODUCE fall-through. After SB-RR (d293) every HOOKED, tool-less WORKER — gather,
        # research follow-up, or trivial producer — is routed to the unified worker loop
        # ABOVE (self-select decides its behavior), so this path is NOT a role=None /
        # ROLE_WORKER routing branch. It is reached ONLY by (a) an OFFLINE node (no hook —
        # it has no tool surface to self-select, so it stays a single bounded chain call,
        # back-compat) or (b) an explicit non-search single-tool node that already invoked
        # its one tool above and now produces prose from that ``tool_value``. The
        # role-execution SWITCH (flag #5) stays retired — it emits RAW free-text (no per-role
        # output schema, no enum-verdict path; d50.1 content is RAW); behavior comes from its
        # SPEC(s) + task framing + reasoning, NOT a code switch.
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

    # AUTONOMY REBUILD P2C — the served RAW WRITE LOOP IS DELETED.
    # _run_file_delivery, _tool_calling_writer_tool_specs, _dispatch_writer_tool,
    # _parse_writer_call, _run_synthesis and _run_raw_file_loop (with their
    # engine-authored riders: shell imperative, figures/table mandate, scope-faithful
    # completion, sources-only-final, section inventory, per-turn continuation
    # directives, is_detailed_task forced continuation, _is_csv_ext/_is_html_ext
    # branches, strip_internal_scaffolding write-path edits) are all removed. EVERY
    # node — write, synthesizer-terminal, gather, trivial — runs the ONE unified
    # self-select loop; write methodology lives in the writer SPECS and the file
    # BUNDLE doctrine; delivery is verified by the target-artifact gate; the tool
    # boundary (sanitize_write_path, guard_write_content) is the only engine touch.

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
        fetched_char_budget: int = 2000,
        upstream_input_char_budget: int = 4000,
        writer_source_budget: Optional[int] = None,
        grower: Optional[Any] = None,
        node_finalizer: Optional[Callable[[PlanNode, "SubAgentResult"], Any]] = None,
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
        # RP-3c (d330): the ``verify_lane`` plumbing is RETIRED (the flag-gated engine
        # verify/revise self-review lane it turned on is gone — the model self-review moved
        # to the definition-layer writer doctrine; the no-fab gather-more gate is de-flagged
        # to an output-agnostic signal gate). No boolean is threaded to the sub-agents.
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
        # ONE-DRIVE PHASE TRANSITION (RP-6c B1): the injected, OPTIONAL wiring that generalizes the
        # growable drive from "grow research waves" to "drive an ordered sequence of PHASES the
        # SHAPE declares, authoring the next phase's sub-DAG on transition". When set (a
        # :class:`PhaseTransition`), :meth:`_drive_growable` — AFTER the research grow loop stops —
        # consults the shape's declared ``next_plan(first_kind)``; if it names a further phase, it
        # calls the MODEL-authoring ``author(...)`` hook (which authors the write sub-DAG via
        # ``IncrementalPlanner.plan``), stamps the returned write-phase node(s) with their delivery
        # target (O1 — so ONLY they take the writer route in this SHARED runtime), appends them to
        # the LIVE dag, and drives them in the SAME run. None (every existing growable research
        # run) → the drive stops at the research grow loop, byte-identical to pre-RP-6c. Duck-typed
        # and additive: the research wave loop is UNCHANGED; only the post-stop behaviour branches.
        self._phase_transition: Optional["PhaseTransition"] = None
        # The number of research layers the growable drive actually ran (seed=1 + grown),
        # for the run trace. 0 until a growable drive completes.
        self._grow_layers = 0
        # BUDGET-RESERVING DISPATCH DEADLINE (live 6GB catch): when the growable drive
        # sets this (the seed wave's SLICE of the wall-clock budget), a passed deadline
        # stops LAUNCHING further nodes in _drive_dag — in-flight nodes finish and their
        # findings stand, the un-launched rest are SKIPPED — so the grow/decision loop
        # is never starved of budget by an over-wide seed wave (previously the outer
        # run timeout cancelled the whole seed mid-flight and growth never ran). A
        # NON-DECIDING resource ceiling (d240): it bounds WHEN gathering stops, never
        # WHAT the model decides. None (the default) = no gate, byte-identical drive.
        self._dispatch_deadline: Optional[float] = None
        # ONE-DRIVE PHASE-TRANSITION TRACE (RP-6c B1): the plan kind the transition authored into
        # the live drive on research stop (e.g. "write_plan"), and the ids of the write-phase
        # node(s) it appended (each carrying a per-node ``deliverable_path`` — the O1 writer-route
        # discriminator). Empty/"" when no transition fired (no phase wiring, or the shape's next
        # phase was terminal). Read by the run trace + tests; never gates the drive.
        self._phase_authored_plan: str = ""
        self._phase_authored_node_ids: list[str] = []
        # s15/a21 — the SURFACED grow-error: when a grow() round raises (previously a SILENT
        # swallow that masqueraded as an early-stop), the drive loop records "<Type>: <msg>"
        # here AND logs the full traceback to stderr, so the served grow_trace + the gate SEE
        # the crash instead of reading a clean stop. None when no round raised.
        self._grow_error: Optional[str] = None
        # RP-6c B1 — the SURFACED phase-transition error: when the write-authoring hook raises,
        # the transition stops GRACEFULLY (the research findings stand) but the error is RECORDED
        # here + the full traceback logged to stderr (never a silent swallow, d186). None when no
        # transition ran or the hook succeeded.
        self._phase_error: Optional[str] = None
        # UNIVERSAL FINALIZE WIRING (d285 SB-4): the injected per-node finalizer. The runtime
        # holds NO Planner/factory, so the ORCHESTRATION supplies this — a coroutine
        # ``finalizer(node, result) -> {"summary": str, "memory_index": str}`` that (served)
        # opens/continues the node's research memory by its brief index via SB-1's
        # ``resolve_brief_memory`` and asks SB-2's ``Planner.finalize_node`` for the NODE's own
        # model digest (fed the node's real output as ``work_digest``). After each node
        # finishes, :meth:`_run_node` calls it and stamps the ``(summary, memory_index)`` pair
        # onto the cached :class:`SubAgentResult`; a downstream node then receives ONLY that
        # pair as its inter-node context (``_compose_task``). None (offline/unit, or a route
        # that does not opt in) => no pair is produced and the handoff is byte-identical.
        self._node_finalizer = node_finalizer

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

    def _is_report_writer(self, node: PlanNode) -> bool:
        """A TOOL-LESS section writer of the SINGLE report deliverable (SB-6/d299/d301).

        DELIVERY-CONTEXT membership: on the report write runtime (``deliverable_path`` is set by
        chat_app.run_section_write_phase), every non-review, non-synthesizer node authors a section
        of the one deliverable. Used ALONGSIDE the module-level :func:`_is_writer_node` to keep the
        writer-chain accumulation (continuation path + finality) working now that report write nodes
        are tool-less (the engine no longer stamps tool=file_write). Keyed on ``deliverable_path``
        (write-phase EXCLUSIVE — set only on the write runtime), NOT ``chain_sources`` (a follow-up
        READER also carries chain_sources to resolve prior sources; keying on it would over-broadly
        treat a non-writer as a writer — d301). Pure DATA, NOT a spec/role/tool-name conditional.
        False on any runtime without deliverable_path (research/gather/follow-up), so those stay
        byte-identical.

        RP-6c B1 (O1) — RE-SCOPED to the SHARED runtime: the write-phase target is now ALSO carried
        PER-NODE (``node.deliverable_path``, stamped by the one-drive phase transition). In a shared
        runtime where research + write coexist, the runtime-global ``deliverable_path`` is UNSET, so
        membership keys on the NODE's own target — ONLY the write-phase node(s) are report writers;
        the research nodes (no per-node target) are not. The legacy dedicated write_runtime (global
        set, per-node unset) is unchanged."""
        return bool(
            (getattr(self, "deliverable_path", None) or getattr(node, "deliverable_path", None))
            and node.role != ROLE_SYNTHESIZER
        )

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
        # UNIVERSAL FINALIZE HANDOFF (d285 SB-4): each DIRECT upstream's ``(summary, memory_index)``
        # pair (stamped by the injected finalizer below when that dep finished). This is the SOLE
        # inter-node context payload — ``_compose_task`` renders it and drops the dep's clipped
        # prose + folded fetched bodies. Built from ``depends_on`` only (d15 direct-upstream-only);
        # a dep with no summary (no finalizer wired) is absent => byte-identical pre-SB-4 handoff.
        upstream_memory = {
            dep: {
                "summary": self._cache[dep].summary or "",
                "memory_index": self._cache[dep].memory_index or "",
            }
            for dep in node.depends_on
            if dep in self._cache and getattr(self._cache[dep], "summary", None)
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
                # d285 SB-4: the direct-upstream (summary, memory_index) pairs — the SOLE
                # inter-node context payload (collapses the clipped-prose + fetched-fold channels).
                upstream_memory=upstream_memory,
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
                # CHAIN-NOTES (SA-4): the run's served research NOTES the write runtime carries
                # (set by agentic.write_report_spa), so a write/review node that self-selects
                # research_read binds read_notes via ctx['notes'] — replacing the retired
                # per-run read_notes pre-registration. None for every non-report runtime.
                chain_notes=getattr(self, "chain_notes", None),
                # SB-6/d301: the write phase's single deliverable path — the delivery-context signal
                # the TOOL-LESS write node routes to the served writer on (not chain_sources, which a
                # follow-up reader also carries). RP-6c B1 (O1): PREFER the NODE's own target (stamped
                # by the one-drive phase transition) over the runtime-global, so in a SHARED runtime
                # ONLY the write-phase node(s) route to the writer and research nodes (no per-node
                # target) keep the research route. The legacy dedicated write_runtime (global set,
                # per-node unset) is byte-identical. None elsewhere.
                deliverable_path=(
                    getattr(node, "deliverable_path", None)
                    or getattr(self, "deliverable_path", None)
                ),
            )
            return await agent.run(inputs)

        healer = SelfHeal(max_heals=self.max_heals)
        res = await healer.run(logic, label=node.id, log=heal_log)
        res.heal = heal_log.as_dict()
        self._heal_logs[node.id] = heal_log.as_dict()
        if heal_log.healed:
            st.healed = True
            await self._emit(EVENT_NODE_HEALED, {"node_id": node.id, "heal": heal_log.as_dict()})
        # UNIVERSAL FINALIZE (d285 SB-4): now that the node has produced a result, ask the
        # injected finalizer for the NODE's own ``(summary, memory_index)`` digest (SB-2) and
        # stamp it onto the cached result, so a downstream node receives ONLY that pair as its
        # inter-node context. Fail-safe: a finalize error never breaks the run — the pair just
        # stays unset (the downstream handoff then degrades to the pre-SB-4 channels). No-op when
        # no finalizer is wired (offline/unit) or the node produced no usable output.
        if self._node_finalizer is not None and res is not None and res.output is not None:
            try:
                pair = await self._node_finalizer(node, res)
                if isinstance(pair, Mapping):
                    res.summary = str(pair.get("summary") or "").strip() or None
                    res.memory_index = str(pair.get("memory_index") or "").strip() or None
            except Exception:  # noqa: BLE001 - a finalize must never break a node's run
                pass
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
        # A node already SKIPPED or CANCELLED in a PRIOR drive of this same run (the
        # growable engine re-drives the SAME dag per grown wave) is TERMINAL — it must
        # not be re-collected as work: re-launching it is the illegal skipped→running
        # transition (live run-3 catch: a budget-skipped seed facet crashed the grown
        # wave's drive). The grower may still author a NEW gap node for that facet by
        # reasoning; the skipped node itself stays skipped.
        remaining = {
            n.id: n
            for n in dag.nodes
            if n.id not in self._cache
            and self._state(n.id).status
            not in (NodeStatus.SKIPPED, NodeStatus.CANCELLED)
        }
        # Nodes already in cache are DONE for this drive (idempotent short-circuit).
        for n in dag.nodes:
            if n.id in self._cache:
                st = self._state(n.id)
                st.cache_hit = st.cache_hit or st.attempts == 0
                if st.status not in (NodeStatus.DONE,):
                    # PENDING → mark done via RUNNING-less path is illegal; set directly
                    st.status = NodeStatus.DONE
        running: dict[asyncio.Task, str] = {}
        # Nodes actually LAUNCHED by this drive — the budget gate's at-least-one floor.
        launched_this_drive = 0

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
            # BUDGET-RESERVING DISPATCH GATE (see _dispatch_deadline): with a deadline
            # set (only the growable research drives set one) the drive runs SINGLE-FILE
            # — one node at a time, since concurrent gathers on the one shared local GPU
            # only serialize at the transport and then all die together at the outer
            # timeout — and, once the slice is spent AND at least one node has launched,
            # no further node launches: in-flight work finishes (its findings stand) and
            # the un-launched rest are SKIPPED with an explicit budget reason. The
            # at-least-one guarantee means a tiny budget still gathers SOMETHING (the
            # graceful-partial contract). No deadline (every non-growable drive) →
            # byte-identical dispatch.
            gated = self._dispatch_deadline is not None
            past_deadline = (
                gated and asyncio.get_running_loop().time() > self._dispatch_deadline
            )
            if past_deadline and launched_this_drive > 0 and not running and remaining:
                for nid in list(remaining):
                    st = self._state(nid)
                    if st.status == NodeStatus.PENDING:
                        st.transition(NodeStatus.SKIPPED)
                        st.error = "skipped: wall-clock slice spent (budget reserved for growth)"
                        await self._emit(
                            EVENT_NODE_SKIPPED, {"node_id": nid, "blocked_by": "budget"}
                        )
                    del remaining[nid]
                break
            launchable = dispatch.nodes
            if gated:
                if past_deadline and launched_this_drive > 0:
                    launchable = []
                elif running:
                    launchable = []  # single-file under a deadline: one node in flight
                else:
                    launchable = dispatch.nodes[:1]
            for node in launchable:
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
                launched_this_drive += 1
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
                # nodes → no writers, but keep the maps consistent with the live DAG). SB-6/d299:
                # the report-writer clause is inert here (research seed runtime has no chain_sources).
                self._writer_ids = {
                    n.id for n in dag.nodes
                    if _is_writer_node(n) or self._is_report_writer(n)
                }
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
                    # SOURCE-AGNOSTIC (as4 de-web): count web ``fetched`` OR a generic
                    # ``records``/``chunks`` artifact, so the counter works for any gather source.
                    total += len(tv.get("fetched") or tv.get("records") or tv.get("chunks") or [])
            return total

        async def _emit_grow_layer(
            idx: int, dispatched: int, stop: Optional[str],
            *, error: Optional[str] = None,
        ) -> None:
            # P2-5c — advisory per-layer progress so a long live run is OBSERVABLE (layer index,
            # nodes dispatched this wave, cumulative nodes/sources, elapsed wall-clock, stop).
            # s15/a21 — when a grow round RAISED, ``error`` carries the surfaced "<Type>: <msg>"
            # so the event stream shows the crash (no longer a silent early-stop).
            # Best-effort: observability never gates or breaks the drive.
            try:
                payload: dict[str, Any] = {
                    "layer": idx,
                    "nodes_dispatched": dispatched,
                    "nodes_total": len(dag.nodes),
                    "sources_so_far": _sources_so_far(),
                    "elapsed_s": round(loop_clock() - t0, 1),
                    "stop_reason": stop,
                }
                if error is not None:
                    payload["error"] = error
                await self._emit(EVENT_GROW_LAYER, payload)
            except Exception:  # noqa: BLE001 — observability only; never breaks the drive
                pass

        # Seed wave (layer 1): the decomposed children (or the unrolled whole-goal seed).
        # The seed gets a bounded SLICE of the wall-clock budget (live 6GB catch: an
        # over-wide seed used to consume the WHOLE budget and the outer run timeout then
        # cancelled it mid-flight — growth never ran and stop_reason stayed empty). Past
        # the slice, in-flight gathers finish and un-launched ones are SKIPPED (their
        # facets stand un-gathered; the grow loop can still author them as gap nodes if
        # the model reasons they matter). Fraction env-tunable; non-deciding (d240).
        seed_slice = budget * _SEED_BUDGET_FRACTION if budget > 0 else 0.0
        if seed_slice > 0:
            self._dispatch_deadline = t0 + seed_slice
        try:
            await self._drive_dag(dag)
        finally:
            self._dispatch_deadline = None
        await _emit_grow_layer(1, len(dag.nodes), None)
        max_layers = max(1, int(getattr(grower, "max_layers", 1) or 1))
        layer = 1
        budget_hit = False
        while layer < max_layers:
            # WALL-CLOCK budget check BEFORE authoring the next layer → graceful partial stop.
            if budget > 0 and (loop_clock() - t0) >= budget:
                budget_hit = True
                break
            # M1 (P2-5b-review) — wrap grow() so a failure MID-GROWTH stops growth GRACEFULLY with
            # the partial findings already gathered, instead of propagating up and aborting the run
            # (the seed + earlier waves' findings/sources stand).
            #
            # s15/a21 (d186 — NO swallowed crash): the previous wrap caught the exception SILENTLY
            # (no traceback), so a layer-2 grow() crash masqueraded as an ordinary early-stop —
            # grow_layers stuck at 1 — and cascaded into "under-measured" notes-graph / prune
            # signals at the a14 gate. The behaviour stays graceful (a transient transport blip mid
            # -growth must not abort the whole run), but it is NO LONGER SILENT: the FULL traceback
            # is logged to stderr and the surfaced error is RECORDED on the runtime + grower + the
            # grow-layer event, so the served grow_trace and the gate SEE the crash and route to the
            # concrete fix (never to "accept the ceiling"). The spec rule — never swallow an
            # exception — is honoured: it is surfaced, not hidden.
            try:
                new_nodes, stop_reason = await grower.grow(dag, self._cache, layer)
            except Exception as exc:  # noqa: BLE001 — graceful-partial, but SURFACED (never silent)
                err_text = traceback.format_exc()
                print(
                    f"[grow-error] DagGrower.grow raised while expanding to layer {layer + 1} "
                    f"(growth stops here; the seed + earlier waves' findings stand):\n{err_text}",
                    file=sys.stderr, flush=True,
                )
                self._grow_error = f"{type(exc).__name__}: {exc}"
                try:
                    grower.grow_error = self._grow_error  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001 — duck-typed grower; advisory only
                    pass
                if getattr(grower, "stop_reason", None) is None:
                    try:
                        grower.stop_reason = "grow_error"
                    except Exception:  # noqa: BLE001 — duck-typed grower; advisory trace only
                        pass
                await _emit_grow_layer(
                    layer + 1, 0, getattr(grower, "stop_reason", "grow_error"),
                    error=self._grow_error,
                )
                break
            if not new_nodes:
                # agent_sufficient / no_expansion — the grower recorded the reason.
                await _emit_grow_layer(layer + 1, 0, getattr(grower, "stop_reason", stop_reason))
                break
            # APPEND the next wave onto the live DAG (the relaxed invariant) and drive it.
            # validate() (run at _drive_dag entry) re-asserts acyclicity over the grown set.
            # The grown wave honors the WHOLE remaining budget as its dispatch deadline
            # (same budget-reserving gate as the seed slice) so it, too, ends gracefully
            # inside the budget instead of being cancelled by the outer run timeout.
            dag.nodes.extend(new_nodes)
            if budget > 0:
                self._dispatch_deadline = t0 + budget
            try:
                await self._drive_dag(dag)
            finally:
                self._dispatch_deadline = None
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
        # ONE-DRIVE PHASE TRANSITION (RP-6c B1) — ADDITIVE post-research-stop step. The research
        # grow loop ABOVE is BYTE-PRESERVED (P2-5b parity); ONLY this post-stop behaviour is new.
        # When a PhaseTransition is wired, on research stop the drive authors the shape's declared
        # NEXT phase's sub-DAG into this SAME live run (Bug B/Bug C dissolve — one drive, one write
        # node, research rides as node context). No wiring → no-op (byte-identical research drive).
        await self._author_next_phase(dag)

    async def _author_next_phase(self, dag: PlanDAG) -> None:
        """RP-6c B1 — the ONE-DRIVE phase-transition authoring step (see :class:`PhaseTransition`).

        On research grow-loop stop, if a :class:`PhaseTransition` is wired AND the SHAPE's declared
        phase order (``next_plan(first_kind)``) names a further phase, author that phase's sub-DAG
        (MODEL-authored via the injected ``author`` hook — the engine authors NO structure), stamp
        the write-phase delivery target onto each authored node (O1 — so ONLY these nodes take the
        writer route in this SHARED runtime; research nodes carry none and keep the research route),
        depend the deps-less authored nodes on the current research SINKS (growing visibility over
        all of research), append them to the LIVE dag, and DRIVE them in the SAME run
        (:meth:`_drive_dag` short-circuits the cached research nodes, so only the write phase runs).

        No wiring, or a TERMINAL next phase (``"done"``) → no-op, byte-identical to a research-only
        growable drive. A hook failure stops the transition GRACEFULLY (the research findings stand)
        and is SURFACED on ``self._phase_error`` + stderr (never a silent swallow, d186)."""
        pt = self._phase_transition
        if pt is None:
            return
        try:
            next_plan = pt.next_plan(pt.first_kind)
        except Exception:  # noqa: BLE001 — a shape-order read must never break the drive
            return
        if not next_plan or next_plan in ("done", pt.first_kind):
            # The shape declares no further phase after research → stop here (byte-identical).
            return
        # AUTHOR the next phase's sub-DAG (the MODEL authors the topology; the engine only INVOKES
        # the authorer the shape's phase names). A hook failure is graceful-but-SURFACED.
        try:
            authored = await pt.author(self, dag, next_plan)
        except Exception as exc:  # noqa: BLE001 — graceful-partial, but SURFACED (never silent)
            print(
                f"[phase-transition] author({next_plan!r}) raised (transition stops; the research "
                f"findings gathered so far stand):\n{traceback.format_exc()}",
                file=sys.stderr, flush=True,
            )
            self._phase_error = f"{type(exc).__name__}: {exc}"
            return
        new_nodes = [n for n in (authored or []) if isinstance(n, PlanNode)]
        if not new_nodes:
            return
        # Current research SINKS = nodes nothing (in the pre-transition DAG) depends on — the write
        # phase depends on these so it runs AFTER research with full visibility. Structural ORDERING
        # data (write follows research), never content authoring.
        new_ids = {n.id for n in new_nodes}
        depended = {dep for n in dag.nodes for dep in n.depends_on}
        sink_ids = tuple(n.id for n in dag.nodes if n.id not in depended and n.id not in new_ids)
        # O1 — STAMP the per-node write-phase delivery target (frozen PlanNode → dataclasses.replace)
        # so ONLY these nodes route to the writer; and wire deps-less nodes onto the research sinks.
        stamped: list[PlanNode] = []
        for n in new_nodes:
            changes: dict[str, Any] = {}
            if pt.deliverable_path and not n.deliverable_path:
                changes["deliverable_path"] = pt.deliverable_path
            if not n.depends_on and sink_ids:
                changes["depends_on"] = sink_ids
            stamped.append(replace(n, **changes) if changes else n)
        # APPEND onto the live DAG (the relaxed invariant, exactly as the grow loop appends a wave).
        dag.nodes.extend(stamped)
        # Keep the writer-chain / dependent maps consistent with the grown node set (mirrors the
        # seed_layer recompute above) so the appended write node(s) are seen as writers.
        self._writer_ids = {
            n.id for n in dag.nodes if _is_writer_node(n) or self._is_report_writer(n)
        }
        self._dependent_ids = {}
        for n in dag.nodes:
            for dep in n.depends_on:
                self._dependent_ids.setdefault(dep, set()).add(n.id)
        self._phase_authored_plan = next_plan
        self._phase_authored_node_ids = [n.id for n in stamped]
        # DRIVE the freshly-appended write phase in the SAME run (cached research short-circuits).
        await self._drive_dag(dag)

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
        # SB-6/d299 — REPORT WRITERS are now TOOL-LESS, so ``_is_writer_node`` (keyed on
        # tool=file_write / role=SYNTH) no longer recognizes them. Re-establish their writer-chain
        # membership by DELIVERY-CONTEXT DATA: on the report write runtime (``chain_sources`` set),
        # every non-review, non-synth node is a section writer of the SINGLE deliverable. This keeps
        # the continuation-path + append-ordering accumulation BYTE-IDENTICAL to the pre-SB-6
        # tool-stamped path (only the discriminator moves from the engine tool-stamp to the write
        # phase's own DATA). On the research runtime (no chain_sources) the extra clause is inert.
        self._writer_ids = {
            n.id for n in dag.nodes
            if _is_writer_node(n) or self._is_report_writer(n)
        }
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
                        # TRUTHFUL TRACE (live catch): a growable drive killed by the
                        # OUTER run timeout used to leave stop_reason EMPTY, which the
                        # trace then mis-read as a model stop. Record the budget stop.
                        if (
                            getattr(dag, "growable", False)
                            and self._grower is not None
                            and getattr(self._grower, "stop_reason", None) is None
                        ):
                            try:
                                self._grower.stop_reason = "budget"
                            except Exception:  # noqa: BLE001 — advisory trace only
                                pass
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
