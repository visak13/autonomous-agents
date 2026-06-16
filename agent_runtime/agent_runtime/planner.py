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
from typing import Any, Mapping, Optional

from llm_framework import Chain, Context, Transport
from llm_framework.stages import call_stage, prompt_assembly, structured_output

from .factory import AbstractPlanFactory, PlanDAG, PlanError
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

# Native structured-call options for the heal decision (d1): the proven
# think=false / temp 0 path. ``think`` OFF so gemma emits the JSON decision
# directly instead of spending the budget on a CoT trace.
_HEAL_OPTS: dict[str, Any] = {
    "api": "native",
    "think": False,
    "temperature": 0,
    "num_predict": 256,
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
                "true ONLY when the request omits a detail you would otherwise have "
                "to GUESS to act correctly; false when it is clear enough to plan"
            ),
        },
        "question": {
            "type": "string",
            "description": (
                "if needs_clarification, ONE short question asking the user for the "
                "missing detail(s); empty string otherwise"
            ),
        },
        "rationale": {"type": "string", "description": "one line: why"},
    },
    "required": ["needs_clarification", "question", "rationale"],
}

# Same proven think=false / temp 0 native path as the heal decision.
_AMBIGUITY_OPTS: dict[str, Any] = {
    "api": "native",
    "think": False,
    "temperature": 0,
    "num_predict": 256,
    "format": _AMBIGUITY_DECISION_SCHEMA,
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
            try:
                dag = self.factory.parse_dag(ctx.structured)
            except PlanError as exc:
                # A structurally-invalid plan is also a malformed-output failure the
                # outer self-heal should re-plan.
                raise MalformedOutputError(f"planner emitted an invalid DAG: {exc}") from exc

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

        A native structured Gemma call (``think=False``, ``temperature=0``, a JSON
        schema with a ``needs_clarification`` boolean + a single ``question``
        string, per d1) — the SAME proven path as :meth:`heal_decision`. When the
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
        system = (
            self.factory.description
            + "\n\nBefore planning, JUDGE whether the user's request is clear enough "
            "to act on. A request is AMBIGUOUS only when it omits a LOAD-BEARING "
            "detail you would otherwise have to GUESS — e.g. 'email me every morning' "
            "does not say WHAT about or at WHAT time. A request that names its "
            "subject and intent is NOT ambiguous just because minor styling is "
            "unstated; do not over-ask. If ambiguous, set needs_clarification=true "
            "and write ONE short question gathering the missing detail(s). If clear, "
            "set needs_clarification=false and question to an empty string. Emit "
            "STRICT JSON {\"needs_clarification\": <bool>, \"question\": <string>, "
            "\"rationale\": <one line>}."
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
        structured Gemma call (``think=False``, ``temperature=0``, a JSON schema with
        an ``enum`` action + ``required`` keys, per d1). The returned
        :class:`HealDecision` is one of :data:`HEAL_ACTIONS`
        (``retry|pivot|abort|extend``) + a one-line rationale; the runtime ROUTES that
        action through its deterministic heal logic (retry → re-dispatch the same node;
        pivot/extend → :meth:`replan_subgraph`; abort → surface). The planner owns
        control flow — this call only MAKES the choice. Raises
        :class:`MalformedOutputError` if the repair loop cannot extract a legal enum
        action so the outer bound can give up."""
        system = (
            self.factory.description
            + "\n\nA SINGLE step of an executing plan FAILED and the plan must NOT be "
            "left dead. Decide how to HEAL it. Choose EXACTLY ONE action:\n"
            "  - 'retry': a TRANSIENT failure (timeout, a one-off error); re-launch "
            "the SAME step unchanged.\n"
            "  - 'pivot': the APPROACH is wrong; re-derive a corrective sub-plan that "
            "reaches the step's intent a DIFFERENT way.\n"
            "  - 'extend': the step needs an EXTRA remediation step before it can "
            "succeed; add it, then continue.\n"
            "  - 'abort': UNRECOVERABLE; give up this branch and surface to the user.\n"
            "Prefer 'retry' for a transient error while attempts remain; 'pivot' when "
            "retries are exhausted or the approach is structurally wrong; 'abort' only "
            "when nothing can recover. Emit STRICT JSON {\"action\": <one of "
            "retry|pivot|abort|extend>, \"rationale\": <one line>}."
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
            try:
                dag = self.factory.parse_dag(ctx.structured)
            except PlanError as exc:
                raise MalformedOutputError(f"re-plan emitted an invalid DAG: {exc}") from exc
            span.set_attribute("planner.replan.node_count", len(dag.nodes))
            return dag


__all__ = [
    "Planner",
    "PlanResult",
    "HealDecision",
    "HEAL_ACTIONS",
    "AmbiguityDecision",
]
