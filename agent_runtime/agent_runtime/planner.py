"""The autonomous planner — phi emits a custom DAG from factory + lookup ONLY.

The planner is the abstract-factory front end (d6/d10). Given a GOAL it:

1. asks the :class:`~agent_runtime.factory.AbstractPlanFactory` for the
   planner-context payload — which is factory description + the specialization
   LOOKUP index + a lean tool catalog, and NOTHING else (no spec bodies, no
   heavy phased prompts; d10). The payload is asserted body-free before the call.
2. runs that through the llm_framework :class:`~llm_framework.chain.Chain` with
   the :func:`~llm_framework.stages.structured_output` stage, so phi's reply is
   parsed as JSON with the built-in BOUNDED repair loop (first line of malformed
   -JSON self-heal).
3. parses the structured value into a validated
   :class:`~agent_runtime.factory.PlanDAG`.

The DAG is entirely model-DERIVED — the planner hard-codes no task list (d6).
The transport is PLUGGABLE: any llm_framework ``Transport`` (the deterministic
:class:`~llm_framework.transport.FakeTransport` for offline Stage-A runs, the
live ``OllamaTransport`` when the GPU frees — d7/d8).

The planner records its EXACT context payload on every plan() call (``last_context``)
so the smoke can prove context-scoping held (factory + lookup only).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from llm_framework import Chain, Context, Transport
from llm_framework.stages import call_stage, prompt_assembly, structured_output

from .factory import AbstractPlanFactory, PlanDAG, PlanError
from .identity import with_identity
from .selfheal import MalformedOutputError
from .tracing import get_tracer, run_blocking_in_span

# Heal-decision enum (blueprint §2e). When a node returns ``{status:"failed"}`` the
# planner makes a HEAL DECISION via a native structured Gemma call and the runtime
# ROUTES the chosen action through the planner's deterministic heal logic — the
# planner owns control flow (d1), the model only picks among these four:
#   * retry  — transient failure; re-launch the SAME step unchanged (re-dispatch).
#   * pivot  — the approach is wrong; re-derive a corrective sub-plan (replan).
#   * extend — the step needs an extra remediation step first (replan, augmenting).
#   * abort  — unrecoverable; give up this branch and surface to the user/neuron.
HEAL_ACTIONS: tuple[str, ...] = ("retry", "pivot", "abort", "extend")

# The per-call OUTPUT SCHEMA for the heal decision (d1 native structured path:
# ``enum`` on the action + ``required`` keys, passed as Ollama ``format=<schema>``).
_HEAL_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": list(HEAL_ACTIONS),
            "description": "how to heal the failed step",
        },
        "rationale": {"type": "string", "description": "one line: why this action"},
    },
    "required": ["action", "rationale"],
}

# Native structured-call options for the heal decision. s1/b1 REASONING ROLLOUT
# (planner steer): ``think=True`` so gemma4 reasons about how to heal a failed step
# (retry/pivot/extend/abort) in the SEPARATE message.thinking field before emitting
# the JSON decision — d1 wants think=True across the WHOLE structured pipeline, and
# the heal decision is a genuine structured step (a1 §3 site #1). The CoT competes
# with the content budget, so ``num_predict`` is raised 256->4096 (a2-proven
# load-bearing: at <=512 the content truncates to EMPTY). temp 0 deterministic; the
# response routes through the transport JSON-extraction interceptor like the others.
_HEAL_OPTS: dict[str, Any] = {
    "api": "native",
    "think": True,
    "temperature": 0,
    "num_predict": 4096,
    "format": _HEAL_DECISION_SCHEMA,
}

# AMBIGUITY ASSESSMENT (scenario-2 clarification turn). Before authoring a DAG the
# planner first DECIDES — via a native structured Gemma call — whether the user's
# request is too UNDERSPECIFIED to act on without guessing a load-bearing detail
# (e.g. "email me every morning" leaves the TOPIC and the TIME unstated). When it
# is, the planner asks ONE concise clarifying question BACK to the user instead of
# silently picking values; the chat pauses and resumes on the clarified intent.
# This is the model's own judgement (d6 — no hard-coded ambiguity rules); the JSON
# schema only constrains the SHAPE of the decision (enum-free booleans + a single
# question string), never which requests are ambiguous.
_AMBIGUITY_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "needs_clarification": {
            "type": "boolean",
            "description": (
                "true ONLY when the request is to SCHEDULE a recurring/future task "
                "and a scheduling detail (what to do, or when/how often) is missing; "
                "false for any normal one-shot request — proceed with sensible "
                "defaults, never interrogate"
            ),
        },
        "question": {
            "type": "string",
            "description": (
                "if needs_clarification, ONE short question for the missing "
                "scheduling detail; empty string otherwise"
            ),
        },
        "rationale": {"type": "string", "description": "one line: why"},
    },
    "required": ["needs_clarification", "question", "rationale"],
}

# s1/b1 REASONING ROLLOUT: gemma4 is a thinking model — enable native ``think=True``
# on the ambiguity gate so the model reasons (in the SEPARATE message.thinking field)
# about whether a request is load-bearingly underspecified before deciding. The CoT
# competes with the content token budget, so ``num_predict`` is raised 256->4096 (the
# a2-proven load-bearing bump: at <=512 the CoT alone fills the budget and the JSON
# ``content`` truncates to EMPTY). temp 0 deterministic; the transport JSON-extraction
# interceptor returns clean JSON.
_AMBIGUITY_OPTS: dict[str, Any] = {
    "api": "native",
    "think": True,
    "temperature": 0,
    "num_predict": 4096,
    "format": _AMBIGUITY_DECISION_SCHEMA,
}

# RESEARCH-PLAN LAST-STEP REVIEWER (d214/d221/d237). After the research plan executes, its
# last-step REVIEWER REASONS over the gathered research and emits a structured STATUS the
# planner reads: whether the research is COMPLETE or still THIN, and the DATA COMPLEXITY —
# the shape of the researched data over N points (how many concerns + how complex) — which
# the planner uses to reason one-pass-vs-sectioned for the write plan (d237). It does NOT
# dictate "write sectioned"; it reports. This replaces the retired hardcoded
# ``_research_plan_final_status`` pure-function that ALWAYS returned write-plan.
_RESEARCH_REVIEW_STATUSES: tuple[str, ...] = ("research_complete", "research_thin")

_RESEARCH_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": list(_RESEARCH_REVIEW_STATUSES),
            "description": "research_complete if the gathered research can support the "
            "deliverable; research_thin if a key part is unsupported / more gathering is needed",
        },
        "data_complexity": {
            "type": "string",
            "description": "one line: the shape of the researched data over N points — how "
            "many distinct concerns/parts and how complex (informs one-pass vs sectioned write)",
        },
        "rationale": {"type": "string", "description": "one line: why"},
    },
    "required": ["status", "data_complexity", "rationale"],
}

_RESEARCH_REVIEW_OPTS: dict[str, Any] = {
    "api": "native",
    "think": True,
    "temperature": 0,
    "num_predict": 4096,
    "format": _RESEARCH_REVIEW_SCHEMA,
}


# FOLLOW-UP PLAN DECISION (d214/d215/d221 — the ITERATIVE PLANNER LOOP). After a plan's
# LAST-STEP REVIEWER emits its status, the PLANNER REASONS whether another plan is needed
# and which kind — replacing the retired hardcoded research->write->done while-loop on the
# served report route. The model picks among:
#   * research_plan — more gathering is needed before the deliverable can be written.
#   * write_plan    — the research is settled; author the written (sectioned) deliverable.
#   * review_plan   — a standalone review/QA pass over the produced deliverable.
#   * done          — the deliverable is complete; EXIT the loop into the terminal synthesizer.
#
# RP-6b (d359/d361): this vocabulary is no longer a HARDCODED engine enum — the deep-research
# PHASES are DECLARED IN THE SHAPE (deep-research.toml ``[[phases]]``) and the plan kinds
# DERIVE from them (``ShapeSpec.followup_plans`` = the phase plan-kinds + the always-on
# review/done loop-controls). We read it from the shipped canonical deep-research shape at
# import; if that shape is unloadable (a degenerate checkout / offline path) we fall back to
# the byte-identical default the shape itself yields for no phases, so the enum is never empty.
def _followup_plans_from_shape() -> tuple[str, ...]:
    """The follow-up plan vocabulary, DERIVED from the deep-research shape's declared phases
    (RP-6b). Falls back to the shape's own no-phases default when the shape can't be loaded."""
    try:
        from .shapes import load_shape

        return load_shape("deep-research").followup_plans
    except Exception:  # noqa: BLE001 - a broken/absent shape must not break import; safe default
        from .shapes import _DEFAULT_FOLLOWUP_PLANS

        return _DEFAULT_FOLLOWUP_PLANS


FOLLOWUP_PLANS: tuple[str, ...] = _followup_plans_from_shape()

# The per-call OUTPUT SCHEMA for the follow-up decision (native structured path, like the
# heal/ambiguity decisions: ``enum`` on next_plan + ``required`` keys).
_FOLLOWUP_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "next_plan": {
            "type": "string",
            "enum": list(FOLLOWUP_PLANS),
            "description": "the next plan to author, or 'done' to finish",
        },
        "rationale": {"type": "string", "description": "one line: why this next plan"},
    },
    "required": ["next_plan", "rationale"],
}

# Native structured-call options for the follow-up decision — same think=True + raised
# num_predict regime as heal/ambiguity (the CoT competes with the content budget).
_FOLLOWUP_OPTS: dict[str, Any] = {
    "api": "native",
    "think": True,
    "temperature": 0,
    "num_predict": 4096,
    "format": _FOLLOWUP_DECISION_SCHEMA,
}

# s17 (user mandate: FLEX output — no engine format stamp): the MODEL names the
# deliverable file. One structured call; the engine parse-to-reads the filename and
# sanitizes it (never invents/repairs the name beyond the basename guard).
_DELIVERABLE_NAME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "filename": {
            "type": "string",
            "description": (
                "the deliverable's filename WITH the extension that fits the format "
                "the request asks for (e.g. report.html for an HTML report, "
                "notes.md for markdown, data.csv for tabular data)"
            ),
        },
        "rationale": {"type": "string", "description": "one line: why this name/format"},
    },
    "required": ["filename", "rationale"],
}
_DELIVERABLE_NAME_OPTS: dict[str, Any] = {
    "api": "native",
    "think": True,
    "temperature": 0,
    "num_predict": 2048,
    "format": _DELIVERABLE_NAME_SCHEMA,
}


@dataclass
class HealDecision:
    """The planner's structured heal decision for ONE failed node (blueprint §2e)."""

    action: str             # one of HEAL_ACTIONS
    rationale: str          # the model's one-line justification
    raw: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "rationale": self.rationale}


@dataclass
class AmbiguityDecision:
    """The planner's structured ambiguity assessment for one user request.

    ``needs_clarification`` is the model's judgement that the request omits a
    load-bearing detail it would otherwise have to guess; ``question`` is the ONE
    concise clarifying question to ask the user back (empty when not ambiguous)."""

    needs_clarification: bool
    question: str
    rationale: str = ""
    raw: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "needs_clarification": self.needs_clarification,
            "question": self.question,
            "rationale": self.rationale,
        }


@dataclass
class ResearchReviewStatus:
    """The research plan's last-step reviewer status (d214/d221/d237).

    ``status`` is one of :data:`_RESEARCH_REVIEW_STATUSES`; ``data_complexity`` is the
    reviewer's one-line read of the shape of the researched data over N points (how many
    concerns + how complex) which the planner reasons over for one-pass-vs-sectioned write."""

    status: str            # research_complete | research_thin
    data_complexity: str   # one-line data-shape read (d237)
    rationale: str = ""
    raw: Optional[str] = None

    @property
    def complete(self) -> bool:
        return self.status == "research_complete"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "data_complexity": self.data_complexity,
            "rationale": self.rationale,
        }


@dataclass
class FollowupDecision:
    """The planner's structured FOLLOW-UP decision after a plan's reviewer status (d214).

    ``next_plan`` is one of :data:`FOLLOWUP_PLANS` — the next plan the planner authors, or
    ``"done"`` to EXIT the loop into the terminal synthesizer. This is the model's own
    reasoning over the last plan's reviewer status + findings digest (d214/d221), replacing
    the retired hardcoded research->write->done while-loop."""

    next_plan: str             # one of FOLLOWUP_PLANS
    rationale: str             # the model's one-line justification
    raw: Optional[str] = None

    @property
    def done(self) -> bool:
        return self.next_plan == "done"

    def as_dict(self) -> dict[str, Any]:
        return {"next_plan": self.next_plan, "rationale": self.rationale}


@dataclass(frozen=True)
class NodeFinalization:
    """The UNIVERSAL node-finalize PAIR (d285 SB-2): a ``(summary, memory_index)`` emitted
    by EVERY node when it finishes — a research worker, a reviewer, and (as ONE caller) the
    terminal synthesizer alike — generalizing the prose-only terminal :meth:`finalize_summary`
    into the node contract.

    * ``summary`` is MODEL-emitted — a brief digest the node's own model writes of what it
      accomplished + the key findings/outcome (and, when its work assessed it, the SHAPE and
      COMPLEXITY of the data; a reviewer's digest then carries the d237 data-complexity signal
      as model-emitted TEXT, which the SB-5 planner reads). The engine authors NO template,
      structure, or fields inside it.
    * ``memory_index`` is the index of the research memory the node used (SB-1's store):
      finalize carries it through UNCHANGED, so passing it back to
      :func:`~agent_runtime.get_research_memory_store` / ``open_memory`` round-trips to the
      SAME memory the node opened/continued.

    Role-agnostic by construction — there is no spec-name / role-name field here; the pair is
    identical in shape for a worker and a reviewer."""

    summary: str        # MODEL-emitted digest of what the node did / found
    memory_index: str   # the index of the research memory the node used (SB-1)

    def as_dict(self) -> dict[str, Any]:
        return {"summary": self.summary, "memory_index": self.memory_index}


@dataclass
class PlanResult:
    """The outcome of one :meth:`Planner.plan` call.

    ``context`` is the exact (body-free) payload phi reasoned over — kept for the
    context-scoping proof. ``raw`` is phi's raw text; ``structured`` the parsed
    JSON; ``dag`` the validated plan."""

    dag: PlanDAG
    context: dict[str, Any]
    raw: Optional[str]
    structured: Any
    repair: dict[str, Any] = field(default_factory=dict)


class Planner:
    """Builds a custom DAG for a goal using phi, scoped to factory + lookup (d10).

    Parameters
    ----------
    transport:
        Any llm_framework ``Transport`` (pluggable; stub or live).
    factory:
        The :class:`AbstractPlanFactory` carrying the body-free lookup. The
        planner holds ONLY this — by construction it cannot reach a spec body.
    max_repair_attempts:
        Bound on the structured-output JSON repair loop (malformed-JSON heal,
        first line). Exhaustion surfaces as :class:`MalformedOutputError` for the
        outer :class:`~agent_runtime.selfheal.SelfHeal` to re-plan.
    call_opts:
        Extra options forwarded to the transport (e.g. ``temperature``,
        ``json=True`` to nudge JSON mode). ``json`` defaults on.
    """

    def __init__(
        self,
        transport: Transport,
        factory: AbstractPlanFactory,
        *,
        max_repair_attempts: int = 2,
        call_opts: Optional[dict[str, Any]] = None,
    ) -> None:
        self.transport = transport
        self.factory = factory
        self.max_repair_attempts = max_repair_attempts
        self.call_opts = {"json": True, **(call_opts or {})}
        # Captured each call for the context-scoping proof.
        self.last_context: Optional[dict[str, Any]] = None
        self.last_result: Optional[PlanResult] = None
        # Captured on the most recent re-plan call (also body-free, for the proof).
        self.last_replan_context: Optional[dict[str, Any]] = None

    def _build_chain(self) -> Chain:
        """The planner's own minimal chain: assemble → call → parse+repair.

        Deliberately NOT build_default_chain — the planner has no memory or tool
        seam; it is a single structured phi call. Lean for context (d10)."""
        chain = Chain()
        chain.use(prompt_assembly())
        chain.use(call_stage(self.transport, **self.call_opts))
        chain.use(
            structured_output(
                self.transport, max_repair_attempts=self.max_repair_attempts
            )
        )
        return chain

    async def plan(self, goal: str) -> PlanResult:
        """Emit a validated :class:`PlanDAG` for ``goal`` (raises on malformed).

        Async so it composes with the in-process runtime and the SelfHeal
        wrapper, though the underlying transport call is synchronous (the chain
        runs in-thread; phi I/O is fast and bounded). On exhausted JSON repair OR
        an unparsable plan it raises :class:`MalformedOutputError` so the outer
        self-heal can re-launch the plan step."""
        system, user = self.factory.planner_prompt(goal)
        # Re-capture the body-free context for the proof (planner_prompt already
        # asserted it body-free).
        self.last_context = self.factory.planner_context(goal)

        ctx = Context(system=system, user=user, transport=self.transport)
        # TRACING (s6/b2): the planning phase is one "planner.plan" span carrying
        # the goal, the bounded repair-attempt count, and the resulting node count.
        # It is opened as the CURRENT span so the phi structured call (the b1
        # transport span emitted inside the chain) nests UNDER it via the
        # cross-thread otel-context propagation below — not a detached root.
        tracer = get_tracer("agent_runtime.planner")
        with tracer.start_as_current_span("planner.plan") as span:
            span.set_attribute("planner.goal", goal[:1000])
            span.set_attribute("planner.max_repair_attempts", self.max_repair_attempts)
            # FREEZE FIX (decouple): the chain runs the SYNCHRONOUS, blocking phi
            # HTTP round-trip (call_stage -> transport.chat, plus any JSON-repair
            # transport.complete). Running it inline would block the one asyncio
            # event loop for the whole phi latency and freeze every request incl
            # /health. Offload to a worker thread (mirrors toolargs.py's emitter
            # offload) — and re-attach this span's otel context inside that thread
            # so the phi span nests under "planner.plan" (run_blocking_in_span).
            ctx = await run_blocking_in_span(self._build_chain().run, ctx)

            repair = ctx.meta.get("structured_output", {})
            span.set_attribute(
                "planner.repair_attempts", len(repair.get("attempts", []) or [])
            )
            if ctx.structured is None:
                raise MalformedOutputError(
                    f"planner JSON unparsable after {self.max_repair_attempts} "
                    f"repair attempts: {repair.get('final_error')}"
                )
            # SAFE FALLBACK (b2 hardening of the a2 4/5 graph-integrity finding):
            # with think=True the planner non-deterministically emits a node whose
            # depends_on names a PHANTOM id. ``parse_dag_safe`` REPAIRS that (drops
            # the dangling/self edge → graceful degrade) so it never surfaces as a
            # user-visible failure. Every OTHER invalidity (dup id / real cycle /
            # empty plan) still raises PlanError → MalformedOutputError, preserving
            # the outer self-heal's retry-on-reject backstop.
            try:
                dag, repairs = self.factory.parse_dag_safe(ctx.structured)
            except PlanError as exc:
                raise MalformedOutputError(f"planner emitted an invalid DAG: {exc}") from exc
            if repairs:
                span.set_attribute("planner.dag.repaired_edges", len(repairs))
                span.set_attribute("planner.dag.repairs", "; ".join(repairs)[:1000])

            span.set_attribute("planner.node_count", len(dag.nodes))
            result = PlanResult(
                dag=dag,
                context=self.last_context,
                raw=ctx.raw_output,
                structured=ctx.structured,
                repair=repair,
            )
            self.last_result = result
            return result

    async def assess_ambiguity(self, goal: str) -> AmbiguityDecision:
        """Decide whether ``goal`` is too underspecified to plan without guessing.

        A native structured Gemma call (s1/b1: ``think=True`` + raised
        ``num_predict``, ``temperature=0``, a JSON schema with a
        ``needs_clarification`` boolean + a single ``question`` string, per d1) —
        the SAME path as :meth:`heal_decision`. When the
        model judges the request omits a LOAD-BEARING detail (one it would have to
        guess to act correctly), it returns ``needs_clarification=True`` and ONE
        concise question to ask the user back; the live chat path then PAUSES and
        resumes on the clarified intent (mirroring the missing-specialist pause).
        Otherwise it returns ``needs_clarification=False`` and the run proceeds
        unchanged.

        This is the model's own judgement (d6 — no hard-coded ambiguity rules). It
        is intentionally FAIL-OPEN: any malformed/short reply that yields no legal
        boolean is treated as "not ambiguous" (proceed), so a transport that does
        not understand the schema never blocks a run — exactly the safe default the
        offline seam and existing callers rely on."""
        system = with_identity(
            self.factory.description
            + "\n\nBefore planning, decide if you must ask ONE clarifying question. "
            "Ask ONLY when the request is to SCHEDULE a recurring or future task "
            "(e.g. 'email me every morning', 'remind me daily at 9') AND a "
            "load-bearing scheduling detail is missing — WHAT to do, or WHEN/how "
            "often. Any normal one-shot request — even a broad one like 'write a "
            "report on X' — is NOT ambiguous: proceed with sensible defaults "
            "(general scope, full timeframe), never interrogate. If a scheduling "
            "detail is missing, set needs_clarification=true and ask ONE short "
            "question for it; otherwise needs_clarification=false and question=\"\". "
            "Emit STRICT JSON {\"needs_clarification\": <bool>, \"question\": "
            "<string>, \"rationale\": <one line>}."
        )
        user = f"USER REQUEST:\n{goal}\n\nReturn ONLY the JSON ambiguity decision."
        chain = Chain()
        chain.use(prompt_assembly())
        chain.use(call_stage(self.transport, **_AMBIGUITY_OPTS))
        chain.use(
            structured_output(self.transport, max_repair_attempts=self.max_repair_attempts)
        )
        ctx = Context(system=system, user=user, transport=self.transport)
        tracer = get_tracer("agent_runtime.planner")
        with tracer.start_as_current_span("planner.assess_ambiguity") as span:
            span.set_attribute("planner.ambiguity.goal", str(goal)[:1000])
            try:
                ctx = await run_blocking_in_span(chain.run, ctx)
            except Exception:  # noqa: BLE001 - fail-open: any transport error => proceed
                return AmbiguityDecision(False, "", rationale="assessment errored; proceeding")
            parsed = ctx.structured
            if not isinstance(parsed, Mapping) or not isinstance(
                parsed.get("needs_clarification"), bool
            ):
                # No legal boolean (malformed / schema-blind transport) => proceed.
                return AmbiguityDecision(False, "", rationale="no legal decision; proceeding")
            needs = bool(parsed.get("needs_clarification"))
            question = str(parsed.get("question", "") or "").strip()
            rationale = str(parsed.get("rationale", "") or "")
            # Fail-open consistency: "ambiguous" with no actual question to ask is
            # not actionable — proceed rather than pause on an empty prompt.
            if needs and not question:
                return AmbiguityDecision(False, "", rationale="ambiguous but no question; proceeding")
            span.set_attribute("planner.ambiguity.needs_clarification", needs)
            return AmbiguityDecision(
                needs_clarification=needs,
                question=question,
                rationale=rationale,
                raw=ctx.raw_output,
            )

    async def finalize_node(
        self,
        goal: str,
        *,
        memory_index: str = "",
        work_digest: str = "",
        plans_authored: Optional[list[str]] = None,
        sources: int = 0,
        sections: int = 0,
        artifact: str = "",
    ) -> NodeFinalization:
        """The UNIVERSAL node FINALIZE (d285 SB-2): a ROLE-AGNOSTIC ``(summary, memory_index)``
        pair every node emits when it finishes.

        Generalizes the prose-only terminal finalize into the NODE CONTRACT: a research worker,
        a reviewer, and (via :meth:`finalize_summary`) the terminal synthesizer all call THIS —
        the synthesizer is now ONE caller, not a special-cased path (d215 terminal kept).

        * the SUMMARY is MODEL-emitted (a short ``think=True``, low-temp Gemma digest the node's
          own model writes) — the engine authors NO template/structure/fields. The prompt invites
          every finishing node to digest WHAT it did + the key findings/outcome AND, when its work
          assessed it, the SHAPE and COMPLEXITY of the data; a reviewer's digest then naturally
          carries the d237 data-complexity signal as model-emitted TEXT (no engine flag/field),
          which the SB-5 planner reads. Nothing here branches on which role calls it.
        * the MEMORY_INDEX is the index of the research memory the node used (SB-1's store),
          carried through UNCHANGED so passing it back to
          :func:`~agent_runtime.get_research_memory_store` / ``open_memory`` resolves to the SAME
          memory the node opened/continued. ``finalize_node`` never opens or mutates the memory —
          it only digests + carries the index.

        FAIL-SAFE: an empty / errored reply yields a minimal DERIVED one-line summary (the offline
        FakeTransport seam) so a finalize never breaks a run — graceful degrade, not a fabricated
        narrative. The ``memory_index`` is preserved on the pair regardless of the summary path."""
        plans = list(plans_authored or [])
        index = str(memory_index or "").strip()
        # Minimal derived fallback (offline seam / empty reply) — a factual one-liner, not an
        # invented narrative; the LIVE thinking model produces the real digest.
        derived = (
            f"Finished: {goal.strip()[:160]}. "
            + (f"Authored {sections} section(s) " if sections else "")
            + (f"grounded in {sources} source(s). " if sources else "")
            + (f"Artifact ready: {artifact}." if artifact else "")
        ).strip()
        system = with_identity(
            self.factory.description
            + "\n\nYour part of the job is COMPLETE. Write a BRIEF (2-3 sentence) factual summary "
            "of what you accomplished — what your work covers/found, the key outcome (for a "
            "finished deliverable, that it is ready), and — WHEN your work assessed it — the SHAPE "
            "and COMPLEXITY of the data (roughly how many distinct parts/concerns and how complex). "
            "Be concrete and factual; do NOT invent facts not implied by the inputs, do NOT add a "
            "preamble, and do NOT restate these instructions. Plain prose only."
        )
        user = (
            f"USER REQUEST / GOAL:\n{goal}\n\n"
            + (f"WHAT YOU PRODUCED / FOUND:\n{work_digest}\n\n" if work_digest else "")
            + f"PLANS RUN: {', '.join(plans) or 'n/a'}\n"
            f"SECTIONS AUTHORED: {sections}\n"
            f"SOURCES GROUNDED: {sources}\n"
            + (f"DOWNLOADABLE ARTIFACT: {artifact}\n" if artifact else "")
            + "\nWrite the brief summary now."
        )
        chain = Chain()
        chain.use(prompt_assembly())
        chain.use(call_stage(
            self.transport, api="native", think=True, temperature=0.3, num_predict=1024,
        ))
        ctx = Context(system=system, user=user, transport=self.transport)
        tracer = get_tracer("agent_runtime.planner")
        with tracer.start_as_current_span("planner.finalize_node") as span:
            span.set_attribute("planner.finalize.goal", str(goal)[:500])
            span.set_attribute("planner.finalize.plans", ",".join(plans))
            span.set_attribute("planner.finalize.memory_index", index)
            try:
                ctx = await run_blocking_in_span(chain.run, ctx)
            except Exception:  # noqa: BLE001 - fail-safe: derived one-liner
                span.set_attribute("planner.finalize.fail_open", True)
                return NodeFinalization(summary=derived, memory_index=index)
            summary = (ctx.raw_output or "").strip()
            if not summary:
                span.set_attribute("planner.finalize.fail_open", True)
                return NodeFinalization(summary=derived, memory_index=index)
            span.set_attribute("planner.finalize.fail_open", False)
            span.set_attribute("planner.finalize.chars", len(summary))
            return NodeFinalization(summary=summary, memory_index=index)

    async def finalize_summary(
        self,
        goal: str,
        *,
        plans_authored: Optional[list[str]] = None,
        sources: int = 0,
        sections: int = 0,
        artifact: str = "",
        memory_index: str = "",
    ) -> str:
        """The TERMINAL SYNTHESIZER's finalize summary (d215/d221): a REAL LLM digest of the run.

        Now ONE CALLER of the universal :meth:`finalize_node` (d285 SB-2) — the synthesizer is no
        longer a special-cased finalize path; it shares the exact role-agnostic finalize contract
        every worker and reviewer uses, and returns just the model-emitted ``summary`` string for
        the SSE announce (the synthesizer streams prose, not the memory index). The synthesizer
        runs ONCE after the planner loop EXITS and delivers the final output — a brief, human-
        facing summary of what the run produced; per d221/d240 it is the model's LLM-generated
        digest, NOT a hardcoded string. It never edits content/coherence (writer + reviewer own
        those, d218).

        FAIL-SAFE: an empty / errored reply yields a minimal DERIVED one-line summary (the offline
        FakeTransport seam) so the SSE announce never breaks — graceful degrade, not a fabricated
        narrative."""
        result = await self.finalize_node(
            goal,
            memory_index=memory_index,
            plans_authored=plans_authored,
            sources=sources,
            sections=sections,
            artifact=artifact,
        )
        return result.summary

    async def review_research(
        self,
        goal: str,
        findings: str,
        *,
        sources: int = 0,
    ) -> ResearchReviewStatus:
        """The research plan's LAST-STEP REVIEWER: reason over the gathered research (d214/d237).

        A native structured Gemma call (same think=True regime as the other reviewer/decision
        calls) that REASONS over the accumulated findings + source count and emits a structured
        status — research_complete vs research_thin — plus the DATA COMPLEXITY (the shape of the
        researched data over N points), which the planner reads to reason one-pass-vs-sectioned
        for the write plan (d237). It replaces the retired ``_research_plan_final_status`` pure
        function that ALWAYS hardcoded write-plan; now a REAL reviewer reasons the status.

        FAIL-SAFE: any malformed / schema-blind reply (the offline FakeTransport seam) yields a
        DERIVED status — research_complete when there are findings/sources, else research_thin,
        with a source-count data-complexity note — so the served route + offline tests stay
        green while the live thinking model gets the real reviewed status."""
        n_src = int(sources or 0)
        derived = "research_complete" if ((findings or "").strip() or n_src) else "research_thin"
        derived_status = ResearchReviewStatus(
            status=derived,
            data_complexity=f"{n_src} source(s) gathered",
            rationale="derived (no reviewed decision)",
        )
        system = with_identity(
            self.factory.description
            + "\n\nYou are the research plan's LAST-STEP REVIEWER. Read the gathered research "
            "and emit a status:\n"
            "- 'research_complete': the research can SUPPORT the desired deliverable (the key "
            "parts are covered with sources).\n"
            "- 'research_thin': a key part is unsupported / more gathering is needed before "
            "writing.\n"
            "Also report DATA COMPLEXITY in one line: the shape of the researched data over N "
            "points — roughly how many distinct concerns/parts it covers and how complex — so "
            "the planner can reason whether the write should be one pass or sectioned. Do NOT "
            "dictate the write structure; just REPORT. Emit STRICT JSON {\"status\": "
            "<research_complete|research_thin>, \"data_complexity\": <one line>, \"rationale\": "
            "<one line>}."
        )
        user = (
            f"DESIRED DELIVERABLE / GOAL:\n{goal}\n\n"
            f"SOURCES GATHERED: {n_src}\n"
            f"GATHERED RESEARCH (bounded):\n{(findings or '')[:6000]}\n\n"
            "Return ONLY the JSON research review status."
        )
        chain = Chain()
        chain.use(prompt_assembly())
        chain.use(call_stage(self.transport, **_RESEARCH_REVIEW_OPTS))
        chain.use(
            structured_output(self.transport, max_repair_attempts=self.max_repair_attempts)
        )
        ctx = Context(system=system, user=user, transport=self.transport)
        tracer = get_tracer("agent_runtime.planner")
        with tracer.start_as_current_span("planner.review_research") as span:
            span.set_attribute("planner.research_review.goal", str(goal)[:500])
            span.set_attribute("planner.research_review.sources", n_src)
            try:
                ctx = await run_blocking_in_span(chain.run, ctx)
            except Exception:  # noqa: BLE001 - fail-safe: derive from what was gathered
                span.set_attribute("planner.research_review.status", derived_status.status)
                span.set_attribute("planner.research_review.fail_open", True)
                return derived_status
            parsed = ctx.structured
            status = (
                str(parsed.get("status")).strip().lower()
                if isinstance(parsed, Mapping) and parsed.get("status") is not None
                else None
            )
            if status not in _RESEARCH_REVIEW_STATUSES:
                span.set_attribute("planner.research_review.status", derived_status.status)
                span.set_attribute("planner.research_review.fail_open", True)
                return derived_status
            data_complexity = (
                str(parsed.get("data_complexity", "")).strip()
                if isinstance(parsed, Mapping)
                else ""
            ) or derived_status.data_complexity
            rationale = (
                str(parsed.get("rationale", "")) if isinstance(parsed, Mapping) else ""
            )
            span.set_attribute("planner.research_review.status", status)
            span.set_attribute("planner.research_review.fail_open", False)
            return ResearchReviewStatus(
                status=status,
                data_complexity=data_complexity,
                rationale=rationale,
                raw=ctx.raw_output,
            )

    async def name_deliverable(
        self,
        goal: str,
        *,
        requested_specs: Optional[Sequence[str]] = None,
    ) -> Optional[str]:
        """The MODEL names the deliverable file (s17 — flex output, no engine stamp).

        One native structured think=True call: the model reads the user's goal (and any
        requested output specs) and returns a filename WITH the extension fitting the
        format the request asks for. The engine parse-to-reads it (d311-8) and the
        caller sanitizes the basename; on ANY failure (offline seam, malformed reply)
        this returns ``None`` and the caller falls back to its neutral default — the
        naming is the model's, never repaired/invented by the engine."""
        system = with_identity(
            "Name the DELIVERABLE FILE for the request below. Pick a short, relatable "
            "kebab-case filename WITH the extension that fits the OUTPUT FORMAT the "
            "request asks for (an HTML report -> .html, markdown -> .md, tabular data "
            "-> .csv, code -> its language's extension, plain text -> .txt). Emit "
            'STRICT JSON {"filename": <name.ext>, "rationale": <one line>}.'
        )
        user = (
            f"REQUEST:\n{goal}\n"
            + (
                f"REQUESTED OUTPUT SPECS: {', '.join(requested_specs)}\n"
                if requested_specs
                else ""
            )
            + "\nReturn ONLY the JSON."
        )
        chain = Chain()
        chain.use(prompt_assembly())
        chain.use(call_stage(self.transport, **_DELIVERABLE_NAME_OPTS))
        chain.use(
            structured_output(self.transport, max_repair_attempts=self.max_repair_attempts)
        )
        ctx = Context(system=system, user=user, transport=self.transport)
        tracer = get_tracer("agent_runtime.planner")
        with tracer.start_as_current_span("planner.name_deliverable") as span:
            span.set_attribute("planner.name_deliverable.goal", str(goal)[:300])
            try:
                ctx = await run_blocking_in_span(chain.run, ctx)
            except Exception:  # noqa: BLE001 — fail-safe: caller falls back to its default
                span.set_attribute("planner.name_deliverable.fail_open", True)
                return None
            parsed = ctx.structured
            name = (
                str(parsed.get("filename")).strip()
                if isinstance(parsed, Mapping) and parsed.get("filename")
                else ""
            )
            # Parse-to-read guard only: a usable name is a bare basename with an
            # extension; anything else -> None (the caller's default names it).
            name = name.replace("\\", "/").rsplit("/", 1)[-1]
            if not name or "." not in name or name.startswith(".") or name.endswith("."):
                span.set_attribute("planner.name_deliverable.fail_open", True)
                return None
            span.set_attribute("planner.name_deliverable.filename", name[:120])
            return name

    async def decide_followup(
        self,
        goal: str,
        *,
        last_plan_kind: str,
        reviewer_status: str,
        reviewer_summary: str = "",
        memory_index: str = "",
        findings_digest: str = "",
        data_complexity: str = "",
        sources: int = 0,
        fresh_sources: Optional[int] = None,
        plans_so_far: Optional[list[str]] = None,
        default_next: str = "done",
    ) -> FollowupDecision:
        """REASON the next plan after a plan's last-step reviewer emits its status (d214/d215).

        The autonomous iterative planner loop: a plan executes, its last-step REVIEWER emits a
        FINAL STATUS, and the PLANNER reads that status to decide the follow-up plan — another
        ``research_plan`` / ``write_plan`` / ``review_plan``, or ``done`` to EXIT into the
        terminal synthesizer (d215). This is the model's own reasoning (a native structured
        Gemma call — the SAME think=True regime as :meth:`assess_ambiguity` / :meth:`heal_decision`),
        NOT a deterministic derivation from the status; it replaces the retired hardcoded
        research->write->done while-loop on the served report route.

        SB-5 (d285/d289) — the planner REASONS over the reviewer's OVERALL SUMMARY: the
        ``(reviewer_summary, memory_index)`` pair is the d285-faithful SINGLE signal the decision
        reads. ``reviewer_summary`` is the reviewer's model-emitted read of the gathered research,
        which CARRIES the d237 data-complexity AS TEXT — so the add-more-research-vs-write decision
        (and, downstream, the sectioned-vs-single emergence) reasons over ONE non-divergent source.
        The structured ``data_complexity`` is no longer consulted SEPARATELY here when a summary is
        present (it rides INSIDE the summary); it remains only the offline/back-compat fallback so a
        caller (or the FakeTransport seam) with no composed summary still degrades gracefully. The
        ``data-complexity`` is DATA the model reasons over (d10-clean) — never a hardcoded
        ``if N>k -> sectioned`` heuristic and never a spec-name/role-name branch.

        It is FAIL-SAFE to ``default_next`` (the caller's safe baseline — e.g. ``write_plan``
        after a research plan whose deliverable is a written report, ``done`` after a write
        plan or a single acyclic plan): any malformed / schema-blind reply (the offline
        FakeTransport seam, a confused model) yields ``default_next``, so the loop always
        makes safe forward progress and terminates — never spins. The LIVE thinking model
        gets the real reasoned decision."""
        default = default_next if default_next in FOLLOWUP_PLANS else "done"
        system = with_identity(
            self.factory.description
            + "\n\nA plan just finished and its last-step REVIEWER emitted a status. Decide "
            "the NEXT step of the overall job by reasoning over that status:\n"
            "- 'research_plan': the deliverable still needs MORE gathering before it can be "
            "written (the research is thin / a key part is unsupported).\n"
            "- 'write_plan': the research is settled and the desired deliverable is a written "
            "(sectioned) report/document — author it now.\n"
            "- 'review_plan': the deliverable is written but needs a standalone review/QA pass.\n"
            "- 'done': the deliverable is COMPLETE (the request is fully served) — FINISH.\n"
            "Reason from the actual status: after a RESEARCH plan whose research is complete and "
            "the user wants a written report, choose write_plan; after a WRITE plan whose "
            "reviewer reports the deliverable complete, choose done; if a simple request was "
            "already fully answered by the plan that just ran, choose done. Do NOT loop "
            "needlessly. DIMINISHING RETURNS: when research plans have ALREADY run and the "
            "latest one added few or NO NEW sources, another research_plan will not help — "
            "the accumulated research memory is what there is; choose write_plan and author "
            "the deliverable FROM it. Emit STRICT JSON {\"next_plan\": <research_plan|"
            "write_plan|review_plan|done>, \"rationale\": <one line>}."
        )
        # SB-5 (d285/d289): reason over the reviewer's OVERALL SUMMARY (the (summary, index)
        # pair) — the SINGLE non-divergent signal, carrying the data-complexity AS TEXT. The bare
        # structured ``data_complexity`` is the offline/back-compat fallback only (no summary).
        reviewer_view = (reviewer_summary or "").strip()
        idx = (memory_index or "").strip()
        complexity_block = (
            "REVIEWER SUMMARY (reason over THIS — the reviewer's overall read of the gathered "
            f"research; it carries the data-complexity as text):\n{reviewer_view}\n"
            if reviewer_view
            else f"DATA COMPLEXITY: {data_complexity or 'n/a'}\n"
        )
        user = (
            f"OVERALL GOAL:\n{goal}\n\n"
            f"LAST PLAN: {last_plan_kind}\n"
            f"ITS REVIEWER STATUS: {reviewer_status}\n"
            + complexity_block
            + (f"RESEARCH MEMORY INDEX: {idx}\n" if idx else "")
            + f"SOURCES GATHERED (accumulated across all research so far): {sources}\n"
            + (
                f"NEW SOURCES ADDED BY THE LAST PLAN: {int(fresh_sources)}\n"
                if fresh_sources is not None
                else ""
            )
            + f"PLANS AUTHORED SO FAR: {json.dumps(list(plans_so_far or []))}\n"
            f"FINDINGS DIGEST:\n{(findings_digest or '')[:1500]}\n\n"
            "Return ONLY the JSON follow-up decision."
        )
        chain = Chain()
        chain.use(prompt_assembly())
        chain.use(call_stage(self.transport, **_FOLLOWUP_OPTS))
        chain.use(
            structured_output(self.transport, max_repair_attempts=self.max_repair_attempts)
        )
        ctx = Context(system=system, user=user, transport=self.transport)
        tracer = get_tracer("agent_runtime.planner")
        with tracer.start_as_current_span("planner.decide_followup") as span:
            span.set_attribute("planner.followup.goal", str(goal)[:500])
            span.set_attribute("planner.followup.last_plan_kind", str(last_plan_kind))
            span.set_attribute("planner.followup.reviewer_status", str(reviewer_status)[:200])
            span.set_attribute("planner.followup.default_next", default)
            # SB-5 (d285/d289) — make the decide leg's SINGLE-signal source visible: it reasoned
            # over the reviewer (summary, memory_index) pair, not the bare structured field.
            span.set_attribute("planner.followup.reviewed_summary", bool(reviewer_view))
            span.set_attribute("planner.followup.reviewer_summary_chars", len(reviewer_view))
            span.set_attribute("planner.followup.memory_index", idx[:200])
            try:
                ctx = await run_blocking_in_span(chain.run, ctx)
            except Exception:  # noqa: BLE001 - fail-safe: any transport error => safe baseline
                span.set_attribute("planner.followup.next_plan", default)
                span.set_attribute("planner.followup.fail_open", True)
                return FollowupDecision(default, rationale="decision errored; safe baseline")
            parsed = ctx.structured
            nxt = (
                str(parsed.get("next_plan")).strip().lower()
                if isinstance(parsed, Mapping) and parsed.get("next_plan") is not None
                else None
            )
            if nxt not in FOLLOWUP_PLANS:
                # No legal decision (malformed / schema-blind transport) => safe baseline so the
                # loop always makes safe progress and terminates (offline seam stays green).
                span.set_attribute("planner.followup.next_plan", default)
                span.set_attribute("planner.followup.fail_open", True)
                return FollowupDecision(default, rationale="no legal decision; safe baseline")
            rationale = (
                str(parsed.get("rationale", "")) if isinstance(parsed, Mapping) else ""
            )
            span.set_attribute("planner.followup.next_plan", nxt)
            span.set_attribute("planner.followup.fail_open", False)
            return FollowupDecision(next_plan=nxt, rationale=rationale, raw=ctx.raw_output)

    async def heal_decision(
        self,
        failed_task: str,
        error: str,
        *,
        attempt: int = 0,
        max_attempts: int = 2,
        completed: Optional[list[str]] = None,
    ) -> HealDecision:
        """Decide how to heal a FAILED node (blueprint §2e) via a Gemma structured call.

        A node returning ``{status:"failed", reason}`` halts terminalization; rather
        than leave a dead plan, the planner makes a HEAL DECISION here — a native
        structured Gemma call (s1/b1: ``think=True`` + raised ``num_predict``,
        ``temperature=0``, a JSON schema with an ``enum`` action + ``required``
        keys, per d1). The returned
        :class:`HealDecision` is one of :data:`HEAL_ACTIONS`
        (``retry|pivot|abort|extend``) + a one-line rationale; the runtime ROUTES that
        action through its deterministic heal logic (retry → re-dispatch the same node;
        pivot/extend → :meth:`replan_subgraph`; abort → surface). The planner owns
        control flow — this call only MAKES the choice. Raises
        :class:`MalformedOutputError` if the repair loop cannot extract a legal enum
        action so the outer bound can give up."""
        system = with_identity(
            self.factory.description
            + "\n\nA step of an executing plan FAILED; the plan must not be left "
            "dead. Choose EXACTLY ONE heal action:\n"
            "- 'retry': TRANSIENT failure (timeout/one-off); re-launch the same step "
            "unchanged. Prefer this while attempts remain.\n"
            "- 'pivot': the APPROACH is wrong; re-derive a corrective sub-plan a "
            "different way. Use when retries are exhausted or the approach is "
            "structurally wrong.\n"
            "- 'extend': the step needs an EXTRA remediation step first; add it, "
            "then continue.\n"
            "- 'abort': UNRECOVERABLE; give up this branch and surface. Use only "
            "when nothing can recover.\n"
            "Emit STRICT JSON {\"action\": <retry|pivot|abort|extend>, \"rationale\": "
            "<one line>}."
        )
        user = (
            f"FAILED STEP: {failed_task}\n"
            f"ERROR: {error}\n"
            f"ATTEMPT: {attempt} of {max_attempts}\n"
            f"ALREADY COMPLETED (do not redo): {json.dumps(list(completed or []))}\n\n"
            "Return ONLY the JSON heal decision."
        )
        chain = Chain()
        chain.use(prompt_assembly())
        chain.use(call_stage(self.transport, **_HEAL_OPTS))
        chain.use(
            structured_output(self.transport, max_repair_attempts=self.max_repair_attempts)
        )
        ctx = Context(system=system, user=user, transport=self.transport)
        # TRACING (s6/b2): the heal decision is its own "planner.heal_decision" span so
        # the Gemma structured call nests under it (and joins the run trace when the
        # runtime calls this from within "agent.run"). Same cross-thread propagation
        # + freeze-fix offload as plan()/replan_subgraph.
        tracer = get_tracer("agent_runtime.planner")
        with tracer.start_as_current_span("planner.heal_decision") as span:
            span.set_attribute("planner.heal.failed_task", str(failed_task)[:500])
            span.set_attribute("planner.heal.error", str(error)[:500])
            span.set_attribute("planner.heal.attempt", int(attempt))
            ctx = await run_blocking_in_span(chain.run, ctx)
            parsed = ctx.structured
            action = (
                str(parsed.get("action")).strip().lower()
                if isinstance(parsed, Mapping) and parsed.get("action") is not None
                else None
            )
            if action not in HEAL_ACTIONS:
                repair = ctx.meta.get("structured_output", {})
                raise MalformedOutputError(
                    "heal decision produced no legal action "
                    f"(got {action!r}; need one of {list(HEAL_ACTIONS)}) after "
                    f"{self.max_repair_attempts} repair attempts: "
                    f"{repair.get('final_error')}"
                )
            rationale = (
                str(parsed.get("rationale", ""))
                if isinstance(parsed, Mapping)
                else ""
            )
            span.set_attribute("planner.heal.action", action)
            return HealDecision(action=action, rationale=rationale, raw=ctx.raw_output)

    async def replan_subgraph(
        self,
        failed_task: str,
        error: str,
        *,
        spec: Optional[str] = None,
        completed: Optional[list[str]] = None,
    ) -> PlanDAG:
        """Re-derive a MINIMAL corrective DAG for a single failed step (d6/o6).

        The runtime calls this when a node has exhausted its node-level self-heal:
        phi re-derives a small sub-graph that accomplishes only that step's intent
        with a different approach, scoped to the SAME body-free factory + lookup
        (d10) and parsed back into a validated :class:`PlanDAG`. Raises
        :class:`MalformedOutputError` on exhausted repair / invalid sub-plan so the
        runtime can bound the re-plan attempts and surface a give-up."""
        system, user = self.factory.replan_prompt(
            failed_task, error, spec=spec, completed=completed
        )
        self.last_replan_context = self.factory.replan_context(
            failed_task, error, spec=spec, completed=completed
        )
        ctx = Context(system=system, user=user, transport=self.transport)
        # TRACING (s6/b2): a corrective re-plan is its own "planner.replan" span so
        # the re-derive phi call nests under it (and, since the runtime calls this
        # from within the "agent.run" span, it joins the same run trace) rather
        # than detaching. Same cross-thread propagation seam as plan().
        tracer = get_tracer("agent_runtime.planner")
        with tracer.start_as_current_span("planner.replan") as span:
            span.set_attribute("planner.replan.failed_task", str(failed_task)[:500])
            span.set_attribute("planner.replan.error", str(error)[:500])
            if spec:
                span.set_attribute("planner.replan.spec", str(spec))
            # FREEZE FIX: offload the blocking phi chain off the event loop (same
            # rationale as Planner.plan above), re-attaching the span context inside
            # the worker thread so the re-derive phi span nests here.
            ctx = await run_blocking_in_span(self._build_chain().run, ctx)
            if ctx.structured is None:
                repair = ctx.meta.get("structured_output", {})
                raise MalformedOutputError(
                    f"re-plan JSON unparsable after {self.max_repair_attempts} "
                    f"repair attempts: {repair.get('final_error')}"
                )
            # SAFE FALLBACK (b2): a corrective sub-plan is authored by the SAME
            # think=True planner, so it can carry the same dangling-edge
            # malformation — repair it rather than fail the heal. Non-repairable
            # invalidity still raises for the runtime's bounded re-plan give-up.
            try:
                dag, repairs = self.factory.parse_dag_safe(ctx.structured)
            except PlanError as exc:
                raise MalformedOutputError(f"re-plan emitted an invalid DAG: {exc}") from exc
            if repairs:
                span.set_attribute("planner.replan.repaired_edges", len(repairs))
                span.set_attribute("planner.replan.repairs", "; ".join(repairs)[:1000])
            span.set_attribute("planner.replan.node_count", len(dag.nodes))
            return dag


__all__ = [
    "Planner",
    "PlanResult",
    "HealDecision",
    "HEAL_ACTIONS",
    "AmbiguityDecision",
    "FollowupDecision",
    "FOLLOWUP_PLANS",
    "ResearchReviewStatus",
]
