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

from .collision import (
    Collision,
    CollisionResolver,
    CollisionUnresolved,
    apply_resolution,
    detect_collision,
    strip_directives,
)
from .factory import PlanDAG, PlanNode
from .heal_router import (
    EVENT_HEAL_ROUTED,
    EVENT_NODE_FAILURE_DETECTED,
    HealRouter,
    register_heal_rule,
)
from .roles import (
    JUDGMENT_REPAIR_BUMP,
    ROLE_RESEARCH,
    is_judgment_role,
    legal_verdict,
    role_framing,
    role_num_predict_floor,
    role_schema,
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
    "The text below is an OUTPUT-SHAPING RULESET, not the task itself. DO the "
    "task described in the user message — using the inputs and tool findings it "
    "carries — and then SHAPE the FORM of your answer to follow the ruleset. The "
    "ruleset governs structure/format ONLY; the content must be the real result "
    "of the task. Never describe or explain the ruleset, and never write about "
    "the skill instead of doing the task."
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
        upstream_tool_values: Optional[Mapping[str, Any]] = None,
        max_repair_attempts: int = 2,
        max_verdict_repairs: int = 2,
        read_search_max_fetch: int = 0,
        search_tool: str = "web_search",
        fetch_tool: str = "web_fetch",
        fetched_char_budget: int = 2000,
    ) -> None:
        self.node = node
        self.transport = transport
        self.hook = hook
        # d13 SEARCH-THEN-READ (a4): "a search is not research." When >0 and a hook
        # is wired, a node that searches the web FOLLOWS THROUGH and ``web_fetch``es
        # the top real result URLs so the produce/role call grounds in ACTUAL
        # article content (Trafilatura markdown), not the search-results page —
        # ``web_fetch`` is therefore ACTUALLY invoked on real upstream URLs (the d13
        # bar), not skipped. This is GENERIC: it fires for the ``research`` ROLE
        # (the deep-research layers) AND for any plain ``web_search`` node (the
        # open-shape news steps), never for a scenario/topic. 0 = OFF (back-compat:
        # the pre-a4 single-tool-then-LLM behaviour, every non-chat path).
        self._read_search_max_fetch = max(0, int(read_search_max_fetch))
        self._search_tool = search_tool
        self._fetch_tool = fetch_tool
        # Per-article char budget when the FETCHED article text is folded into the
        # user turn (the generic 1200-char tool-output cap is far too small to carry
        # real article content to synthesis). Each fetched source is truncated to
        # this; with ~3 sources it fits the deep-research num_ctx comfortably.
        self._fetched_char_budget = max(400, int(fetched_char_budget))
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
        return "\n\n".join(parts)

    @property
    def spec_names(self) -> tuple[str, ...]:
        """The ordered names of every composed spec (empty for a bare node)."""
        return tuple(s.name for s in self.scopes)

    def _compose_task(self, inputs: Mapping[str, Any], tool_value: Any) -> str:
        """Fold the task + upstream results + tool output into the user turn.

        This is still the *task at hand* (d10) — predecessor outputs are the
        step's inputs, not extra specializations or phased prompts. The spec
        ruleset never enters here: the USER turn carries ONLY the real task
        content + findings; the SHAPING ruleset rides the SYSTEM turn (d1, see
        :meth:`_compose_system`)."""
        parts: list[str] = []
        if self._conversation_context:
            parts.append(
                f"{_PRIOR_CONVERSATION_HEADER}\n{self._conversation_context}\n\n"
                "CURRENT TASK:"
            )
        parts.append(self.node.task)
        if inputs:
            parts.append("\nINPUTS FROM PRIOR STEPS:")
            for k, v in inputs.items():
                parts.append(f"- {k}: {str(v)[:800]}")
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
                "real articles below. Report the CONCRETE facts, figures, headlines "
                "and quotations found IN this text. Do NOT describe the search "
                "results or say a site 'has an article about' something — state the "
                "actual findings from the article text:"
            ]
            for i, art in enumerate(fetched, 1):
                title = str(art.get("title") or "").strip() or "(untitled)"
                url = str(art.get("url") or "").strip()
                body = str(art.get("markdown") or "").strip()[: self._fetched_char_budget]
                parts.append(f"\n--- SOURCE {i}: {title} <{url}> ---\n{body}")
            return "\n".join(parts)
        return f"\nTOOL OUTPUT ({self.node.tool}):\n{str(tool_value)[:1200]}"

    def _research_query(self) -> str:
        """The web-search query for a research-role node: this layer's topic.

        Strips the unroll's ``[research · round N] `` task prefix so the query is the
        clean goal/topic; the growing-visibility inputs (prior layers + critic
        follow-ups, threaded into the user turn) carry the DEPTH, while a rotating
        result window (see :meth:`_round_index`) gives each round fresh sources."""
        task = (self.node.task or "").strip()
        if task.startswith("[") and "]" in task:
            task = task.split("]", 1)[1].strip()
        return task or "(research topic)"

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

    async def _fetch_urls(
        self, urls: Sequence[str], *, limit: Optional[int] = None,
        max_attempts: Optional[int] = None,
    ) -> list[dict[str, str]]:
        """``web_fetch`` each URL (Trafilatura markdown); keep only READABLE articles.

        Each kept fetch yields ``{title, url, markdown}``. A fetch that fails, returns
        empty, or returns non-text/binary content (a PDF/file) is SKIPPED (a dead or
        unreadable link must not fail the node); the hook publishes tool_call/
        tool_result so the live trace shows ``web_fetch`` actually firing on each real
        URL (the d13 observability bar). ``limit`` stops once that many readable
        articles are collected; ``max_attempts`` bounds wasted fetches."""
        out: list[dict[str, str]] = []
        attempts = 0
        for url in urls:
            if limit is not None and len(out) >= limit:
                break
            if max_attempts is not None and attempts >= max_attempts:
                break
            attempts += 1
            try:
                res = await self.hook.invoke(self._fetch_tool, url=url)
            except Exception:  # noqa: BLE001 - a dead link must not fail the layer
                continue
            if not getattr(res, "ok", False):
                continue
            val = res.value if isinstance(res.value, Mapping) else {}
            md = str(val.get("markdown") or "").strip()
            if not md or not self._is_readable_fetch(val):
                continue
            out.append({
                "title": str(val.get("title") or ""),
                "url": str(val.get("final_url") or url),
                "markdown": md,
            })
        return out

    @classmethod
    def _result_urls(cls, search_value: Mapping[str, Any]) -> list[str]:
        """Ordered, de-duplicated, article-plausible http(s) URLs from a search."""
        urls: list[str] = []
        seen: set[str] = set()
        for row in (search_value.get("results") or []):
            if not isinstance(row, Mapping):
                continue
            url = str(row.get("url") or "").strip()
            if cls._looks_like_article_url(url) and url not in seen:
                seen.add(url)
                urls.append(url)
        return urls

    async def _read_search_results(
        self, search_value: Mapping[str, Any], max_fetch: int
    ) -> dict[str, Any]:
        """Follow a web_search through to reading its top READABLE results (d13).

        Fetches until ``max_fetch`` readable articles are collected (bounded attempts)
        and returns the search value ENRICHED with a ``fetched`` list of extracted
        articles. The original search rows are preserved so a node can still see the
        candidate list; ``fetched`` is what carries the REAL content into the call."""
        urls = self._result_urls(search_value)
        fetched = await self._fetch_urls(
            urls, limit=max_fetch, max_attempts=max_fetch * 2
        )
        return {**dict(search_value), "fetched": fetched, "fetched_count": len(fetched)}

    async def _emit_research_queries(self, inputs: Mapping[str, Any]) -> list[str]:
        """Author 1-2 FOCUSED keyword search queries for THIS research layer.

        d13 / the eda-base3 "small reliable decision per node": rather than feed the
        verbose instructional goal to DuckDuckGo (which returned off-topic junk in the
        live max_iter=10 run), the model emits short keyword queries from the GOAL +
        the prior layers' findings and the critic's follow-up questions — so each
        round searches something NEW and goes DEEPER (the critic genuinely drives the
        next fetch). A failed/empty emission falls back to a trimmed goal so the layer
        still searches. One small structured call (think=False, temp 0)."""
        goal = self._research_query()
        prior = "\n".join(f"{k}: {str(v)[:600]}" for k, v in inputs.items())[:2400]
        schema = {
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["queries"],
        }
        system = (
            "You plan web searches for ONE layer of an iterative research "
            "investigation. Output 1-2 SHORT keyword search queries (about 3-8 words "
            "each, NOT full sentences, NOT questions) that will surface NEW, on-topic "
            "articles to go DEEPER on the research goal. Use the prior layers' "
            "findings and the critic's follow-up questions (if shown) to decide what "
            "to search next; stay strictly on the goal's actual topic."
        )
        user = (
            f"RESEARCH GOAL: {goal}\n\nPRIOR LAYERS (findings + critic follow-ups):\n"
            f"{prior or '(none yet — this is the first layer)'}\n\n"
            "Return ONLY the JSON object with a 'queries' list of 1-2 keyword queries."
        )
        try:
            chain = Chain()
            chain.use(prompt_assembly())
            chain.use(call_stage(self.transport, think=False, temperature=0,
                                 num_predict=200, format=schema))
            chain.use(structured_output(self.transport, max_repair_attempts=1))
            ctx = Context(system=system, user=user, transport=self.transport)
            ctx = await run_blocking_in_span(chain.run, ctx)
            parsed = ctx.structured
            if isinstance(parsed, Mapping):
                qs = [str(q).strip() for q in (parsed.get("queries") or []) if str(q).strip()]
                if qs:
                    return qs[:2]
        except Exception:  # noqa: BLE001 - degrade to the goal query, never fail the layer
            pass
        return [goal[:120]]

    async def _research_read(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """The research-role executor: author focused queries, search, then READ (d13).

        The model authors 1-2 focused keyword queries for THIS layer (from the goal +
        prior layers + critic follow-ups), each ``web_search``ed; the readable (non-
        PDF) result URLs are de-duplicated and ``web_fetch``ed until ``max_fetch``
        real articles are read, whose EXTRACTED text is folded into the role call.
        web_fetch is therefore ACTUALLY invoked on real article URLs (not skipped),
        and successive rounds fetch DIFFERENT, deeper sources — generic to the
        research role, never a scenario/topic (d12)."""
        max_fetch = self._read_search_max_fetch
        queries = await self._emit_research_queries(inputs)
        urls: list[str] = []
        seen: set[str] = set()
        for q in queries:
            try:
                res = await self.hook.invoke(self._search_tool, query=q)
            except Exception:  # noqa: BLE001 - one bad search must not fail the layer
                continue
            if not getattr(res, "ok", False) or not isinstance(res.value, Mapping):
                continue
            for u in self._result_urls(res.value):
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
        fetched = await self._fetch_urls(urls, limit=max_fetch, max_attempts=max_fetch * 3)
        return {"queries": queries, "result_urls": urls,
                "fetched": fetched, "fetched_count": len(fetched)}

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
        no spec body is just the role framing (a bare-role call)."""
        role = self.node.role
        framing = role_framing(role) if role else None
        if not self.spec_body:
            return framing  # bare step (None) or bare-role call (framing only)
        base = f"{_SHAPING_FRAMING}\n\n{self.spec_body}"
        return f"{base}\n\n{framing}" if framing else base

    async def run(self, inputs: Optional[Mapping[str, Any]] = None) -> "SubAgentResult":
        """Execute the node: optional tool call, then a scoped phi call.

        A failed tool call raises :class:`ToolFailureError`; a result the
        validator rejects raises :class:`InvalidStepError` (both self-heal
        classes). A transport-level error propagates unchanged."""
        inputs = inputs or {}
        tool_value: Any = None
        if (
            self.node.role == ROLE_RESEARCH
            and self.hook is not None
            and self._read_search_max_fetch > 0
        ):
            # d13 RESEARCH EXECUTOR (a4): a research LAYER must REPORT THE ACTUAL
            # CONTENT of its sources, not describe a search-results page. Rather than
            # leave the node to hallucinate findings from parametric memory (the
            # pre-a4 deep-research role path had NO web access at all), DETERMINISTI-
            # CALLY search the web for this layer's topic and ``web_fetch`` real
            # article URLs, then hand the role call the EXTRACTED article text. The
            # model's only job is to read+synthesise — it never has to decide to call
            # a tool (which a 4.6B model does unreliably). Generic: keyed off the
            # research ROLE, never a scenario/topic (d12).
            tool_value = await self._research_read(inputs)
        elif self.node.tool and self.hook is not None:
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
            # d13 SEARCH-THEN-READ (a4): a plain ``web_search`` node (e.g. an
            # open-shape news step) returns only candidate {title,url,snippet} rows —
            # summarising THAT is the "describes the source list" failure. Follow
            # through and ``web_fetch`` the top real result URLs so the produce call
            # grounds in actual article text. Enriches (never replaces) the search
            # value with a ``fetched`` list; a fetch failure degrades gracefully.
            if (
                self._read_search_max_fetch > 0
                and self.node.tool == self._search_tool
                and isinstance(tool_value, Mapping)
            ):
                tool_value = await self._read_search_results(
                    tool_value, self._read_search_max_fetch
                )

        # Scoped chain: the SHAPING-framed spec ruleset is the system turn (d1/
        # d10), the real task content + findings is the user turn — TASK-DOING and
        # RULESET-SHAPING composed at the produce step, never conflated.
        system = self._compose_system()
        user = self._compose_task(inputs, tool_value)
        if self.node.role:
            # ROLE NODE (a3 generic role execution): a schema-constrained native
            # structured call (the per-role output schema as ``format=<schema>``,
            # raised num_predict floor) + a JUDGMENT-role enum-verdict repair loop —
            # the §2c/b3 mechanic, now GENERIC in the runtime (no per-shape engine).
            raw, parsed, verdict, repairs = await self._run_role(system, user)
            result = SubAgentResult(
                node_id=self.node.id,
                spec=self.node.primary_spec,
                specs=self.spec_names,
                output=raw,
                tool_used=self.node.tool,
                tool_value=tool_value,
                role=self.node.role,
                parsed=parsed,
                verdict=verdict,
                verdict_repairs=repairs,
            )
        else:
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

    async def _run_role(
        self, system: Optional[str], user: str
    ) -> tuple[Optional[str], Any, Optional[str], int]:
        """Run a ROLE node through the d1 structured path; return (raw, parsed,
        verdict, repairs) — the GENERIC role-execution mechanic (a3).

        The per-role OUTPUT SCHEMA is passed as native ``format=<schema>`` and the
        ``num_predict`` floor is raised for the role (judgment roles higher, so the
        enum verdict is not truncated away), with a bounded ``structured_output``
        JSON parse/repair. For a JUDGMENT role the parsed enum ``verdict`` is then
        validated against its legal set and, if missing/null/out-of-enum (the
        truncated-JSON failure mode the b3 hardening targets), the call is
        RE-ISSUED with a larger budget up to ``max_verdict_repairs`` — so the
        runtime never silently records a null verdict. The blocking phi round-trip
        is offloaded off the event loop (the freeze-fix doctrine)."""
        role = self.node.role
        schema = role_schema(role)
        base_opts = dict(self._call_opts)
        floor = role_num_predict_floor(role)
        base_opts["num_predict"] = max(int(base_opts.get("num_predict", 0) or 0), floor)
        base_opts["format"] = schema

        async def call(num_predict: Optional[int]) -> tuple[Any, Optional[str]]:
            opts = dict(base_opts)
            if num_predict is not None:
                opts["num_predict"] = int(num_predict)
            chain = Chain()
            chain.use(prompt_assembly())
            chain.use(call_stage(self.transport, **opts))
            chain.use(
                structured_output(
                    self.transport, max_repair_attempts=self._max_repair_attempts
                )
            )
            ctx = Context(system=system, user=user, transport=self.transport)
            ctx = await run_blocking_in_span(chain.run, ctx)
            return ctx.structured, ctx.raw_output

        parsed, raw = await call(None)
        if not is_judgment_role(role):
            return raw, parsed, None, 0
        # JUDGMENT role: validate + repair the enum verdict (never pass null).
        verdict = legal_verdict(role, parsed)
        repairs = 0
        budget = base_opts["num_predict"]
        while verdict is None and repairs < self._max_verdict_repairs:
            repairs += 1
            budget += JUDGMENT_REPAIR_BUMP
            parsed, raw = await call(budget)
            verdict = legal_verdict(role, parsed)
        return raw, parsed, verdict, repairs

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
        max_heal_retries: int = 1,
        execution: ExecutionMode = ExecutionMode.CONCURRENT,
        conversation_context: Optional[str] = None,
        read_search_max_fetch: int = 0,
        fetched_char_budget: int = 2000,
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
        self.fetched_char_budget = max(400, int(fetched_char_budget))
        # CONVERSATION MEMORY (s5/a4): the bounded prior-turn context for THIS chat
        # run, handed to EVERY node's sub-agent (producer + inline reviewer) so the
        # answer-authoring node sees prior turns — closing the gap where only the
        # planner saw the history and paraphrased node tasks dropped the facts.
        # None => the runtime is memoryless exactly as before (every non-chat path).
        self.conversation_context = (conversation_context or "").strip() or None
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
                upstream_tool_values=upstream_tool_values,
                max_verdict_repairs=self.max_verdict_repairs,
                read_search_max_fetch=self.read_search_max_fetch,
                fetched_char_budget=self.fetched_char_budget,
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
            # d13 (a4): the inline reviewer of a research node must read the SAME
            # fetched-article context the producer did, so it corrects against real
            # sources, not a memoryless re-read.
            read_search_max_fetch=self.read_search_max_fetch,
            fetched_char_budget=self.fetched_char_budget,
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
        enactment."""
        if self.heal_router is None:
            return await self._try_replan(node, error)
        attempt = self._heal_retries.get(node.id, 0)
        completed = sorted(self._done_ids())
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
            try:
                if timeout is not None:
                    try:
                        await asyncio.wait_for(self._drive_dag(dag), timeout=timeout)
                    except asyncio.TimeoutError:
                        out.timed_out = True
                else:
                    await self._drive_dag(dag)
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
