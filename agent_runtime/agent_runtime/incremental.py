"""Tool-driven plan authoring — the eda-base3 create_plan→add_action port (d39).

The one-shot :class:`~agent_runtime.planner.Planner` asks Gemma to emit the WHOLE
DAG in a SINGLE structured-output call. A 4.6B local model cannot reliably do that
for a parallel topology + multi-tool binding. The previous seed-then-fill authorer
improved on that by authoring nodes ONE AT A TIME — but each node was still a native
``format``-schema-CONSTRAINED call, which is the **d34 edge-drop** failure: constrained
decoding trades CONTENT fidelity for syntactic validity, silently dropping the small
model's correctly-reasoned ``depends_on`` edge (measured 0/3 connected WITH schema vs
3/3 WITHOUT; the edge is in ``message.thinking`` every time). A writer node then runs
DISCONNECTED from the research and the report is thin.

This module replaces that with the eda-base3 mechanism: the planner BUILDS the DAG by
ISSUING TOOL CALLS — ``seed_plan`` → ``add_step`` (one per node) → optional
``set_node_spec`` → ``finalize_plan`` — exactly as the EDP planner calls
``create_plan`` then ``add_action`` per action. Each call is prompt-elicited JSON the
loop parses + validates (NO ``format`` schema, ``think=True``), so a reasoned
``depends_on`` can never be schema-dropped (the specialist ruleset: load-bearing
reasoned fields go through discrete tool calls / prompt-JSON + validate-and-repair,
never constrained decoding). The tools live in :mod:`agent_runtime.plan_tools`
(:class:`PlanBuilder`) — pure data + validation; THIS module owns the transport loop.

The result is a validated :class:`~agent_runtime.factory.PlanDAG` wrapped in the SAME
:class:`~agent_runtime.planner.PlanResult` the one-shot planner returns, so it is a
drop-in for the live acyclic path (``chat_app.agentic._run_acyclic``): missing-
specialist detection, the SchemaToolArgEmitter tool-arg grounding, the lifecycle gate
and the runtime all consume the DAG unchanged — only HOW the DAG was authored changed.

Context-scoping (d10) is preserved: the planner reasons over the SAME body-free
factory context (factory description + the specialization LOOKUP index + the lean tool
catalog) — never a compiled spec body — asserted body-free before the loop.

RP-3b (d311/d319/d328): the three engine-authored STRUCTURE repairs (the F2 default-
research-spec pass, the d28 terminal-research-edge pass, the b5 output-format pass) are
RETIRED — the planner authors the gather-node spec, the writer<-research edge, and the
terminal format writer ITSELF (measured 100% reliable on live E4B). The engine authors
NO DAG/spec/format structure; reliability is a definition-layer property (the planner
prompt + the selected shape). Only F5 (honours a user-NAMED spec) + the d7 dangling-edge
backstop remain over the authored node set. RP-AUDIT F5 RETIRED the d50 filename echo —
filename-honoring is carried by ``PlanDAG.goal`` → ``derive_output_path`` (no engine-
authored task text, no baked file-writer sink).
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from llm_framework import Chain, Context, Transport
from llm_framework.stages import call_stage, prompt_assembly, structured_output

from .factory import AbstractPlanFactory, PlanDAG, PlanError
from .identity import with_identity
from .plan_tools import PLAN_TOOL_NAMES, PLAN_TOOLS_SPEC, PlanBuilder
from .planner import PlanResult
from .selfheal import MalformedOutputError
from .tracing import get_tracer, run_blocking_in_span

# Hard bound on the number of authored NODES — the planner's "stop calling add_step"
# is the ``finalize_plan`` call, but a runaway model that never finalizes must still
# terminate. A POC acyclic plan is a handful of steps; 12 is generous headroom over
# any realistic modular-parallel fan-out while capping the GPU cost of authoring.
DEFAULT_MAX_NODES = 12

# num_predict for ONE tool-call turn. The tool-call JSON is small, BUT ``think=True``
# (gemma4 reasons in the SEPARATE message.thinking field) competes with the content
# budget, so the cap is 4096 (the a2-proven load-bearing bump: at <=512 the CoT alone
# fills the budget and the JSON ``content`` truncates to EMPTY). temp 0 holds the tool
# call tight; this is headroom for the CoT, not a larger payload.
DEFAULT_NODE_NUM_PREDICT = 4096


class IncrementalPlanner:
    """Author a plan's DAG via the planner's tool calls (seed→add→set→finalize).

    Parameters
    ----------
    transport:
        Any ``llm_framework`` ``Transport`` (the live ``OllamaTransport`` or an
        offline ``FakeTransport``). Each tool-call turn goes through it with the
        native reasoning options (``api=native``, ``think=True``, ``temperature=0``,
        ``num_predict``) — and NO ``format`` schema, so the reasoned ``depends_on``
        edge is never constrained-decoding-dropped (d34).
    factory:
        The body-free :class:`AbstractPlanFactory` — the planner's ONLY world view
        (d10). Supplies the factory description + specialization LOOKUP + tool
        catalog for the system prompt, and parses the authored nodes back into a
        validated :class:`PlanDAG` (so the same validation as the one-shot path).
    spec_names / tool_names:
        The registered specialization names and the offered tool names. The
        :class:`PlanBuilder` validates each tool call's ``spec``/``specs``/``tool``
        against these sets (an unknown value is dropped, never crashes the loop) —
        the same vocabulary discipline the old per-node enum schema enforced, but as
        validation instead of constrained decoding (so it cannot drop ``depends_on``).
    shape_name / shape_description:
        The SELECTED shape (the seed's topology posture). Threaded into the system
        prompt so the model authors edges that fit the shape (parallel vs chained).
    max_nodes:
        Hard cap on authored nodes (:data:`DEFAULT_MAX_NODES`).
    node_num_predict:
        Output-token budget per tool-call turn (:data:`DEFAULT_NODE_NUM_PREDICT`).
    call_opts:
        Extra transport options merged OVER the proven native reasoning defaults.
    """

    def __init__(
        self,
        transport: Transport,
        factory: AbstractPlanFactory,
        *,
        spec_names: Sequence[str] = (),
        tool_names: Sequence[str] = (),
        shape_name: str = "",
        shape_description: str = "",
        shape_decompose_methodology: str = "",
        requested_specs: Sequence[str] = (),
        max_nodes: int = DEFAULT_MAX_NODES,
        node_num_predict: int = DEFAULT_NODE_NUM_PREDICT,
        max_repair_attempts: int = 2,
        call_opts: Optional[dict[str, Any]] = None,
        authoring_directive: str = "",
    ) -> None:
        self.transport = transport
        self.factory = factory
        # AUTHORING DIRECTIVE (s14/a9) — a phase-specific instruction surfaced on EVERY
        # per-turn authoring prompt (initial / observation / finalize), not just buried in
        # the goal. The write phase passes the SOURCE-ID mandate here so the per-turn lever
        # (the strongest one for E4B) actually reminds the model to set source_ids on each
        # section step. Empty for the research/gather planner → byte-identical behaviour.
        # This is PROMPT TEXT only — the model still decides which [S#] each section uses;
        # it is NOT a deterministic seatbelt (d148: reliability via prompt quality).
        self.authoring_directive = str(authoring_directive or "").strip()
        self.spec_names = [str(s) for s in spec_names if str(s).strip()]
        self.tool_names = [str(t) for t in tool_names if str(t).strip()]
        self.shape_name = str(shape_name or "")
        self.shape_description = str(shape_description or "")
        # SHAPE-DEFINED AUTHORING METHODOLOGY (RP-4c/d341). When the SELECTED shape supplies a
        # ``decompose_methodology`` (e.g. the schedule-leg shape's SCHEDULE-ONLY doctrine), it is
        # substituted into the authoring procedure and TAKES PRECEDENCE over the hardcoded generic
        # gather→combine→deliver recipe (which becomes a FALLBACK only for shapes with no
        # methodology). This mirrors research_tree's shape-methodology substitution (d161/d170):
        # behaviour lives in the SHAPE (definition layer); the substitution is generic +
        # shape-agnostic (no spec-name/flow conditional). Empty → byte-identical to the pre-d341
        # generic authoring procedure.
        self.shape_decompose_methodology = str(shape_decompose_methodology or "").strip()
        # F5 USER-REQUESTED SPECIALIZATIONS: the spec name(s) the user EXPLICITLY
        # named (extracted by the model-driven ShapeSelector, enum-constrained to
        # registered names). The authorer is TOLD about them (so the tool calls bind
        # them on the right node), and a finalization pass GUARANTEES they are
        # honored — if the small model forgot to bind any, they are stamped onto the
        # plan's terminal/delivery node(s). Kept only when actually registered.
        self.requested_specs = [
            str(s).strip()
            for s in (requested_specs or [])
            if str(s).strip() and str(s).strip() in set(str(n) for n in spec_names)
        ]
        self.max_nodes = max(1, int(max_nodes))
        self.node_num_predict = int(node_num_predict)
        self.max_repair_attempts = max_repair_attempts
        # Native reasoning path (mirrors the heal / ambiguity / shape-selection
        # calls): api=native so the top-level ``think=True`` reaches /api/chat and
        # gemma4 reasons about each tool call in the SEPARATE message.thinking field.
        # Those thinking tokens compete with the content budget, so ``num_predict``
        # (DEFAULT_NODE_NUM_PREDICT=4096) gives the CoT headroom. temp 0 for
        # deterministic authoring. CRUCIALLY: NO ``format`` schema — a constrained
        # decode would re-introduce the d34 edge-drop on ``depends_on``. An explicit
        # caller ``call_opts`` overrides.
        self.call_opts = {
            "api": "native",
            "think": True,
            "temperature": 0,
            "num_predict": self.node_num_predict,
            **(call_opts or {}),
        }
        # Captured each plan() call for the context-scoping proof (parity with
        # Planner.last_context) and the authored-node introspection.
        self.last_context: Optional[dict[str, Any]] = None
        self.last_result: Optional[PlanResult] = None
        self.last_nodes: list[dict[str, Any]] = []
        # The builder of the most recent plan() — exposes the per-call tool-call
        # audit trail (``.calls``) for the s7 trace / proof that the planner issued
        # tool calls rather than one-shotting a schema.
        self.last_builder: Optional[PlanBuilder] = None

    # ------------------------------------------------------------------ #
    # prompts (tool catalog system prompt + per-turn instruction)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tool_catalog_text() -> str:
        """Render the four plan-building tools + their args for the system prompt.

        DELEGATES to the PlanningBundle (d190) — the bundle owns the plan-tool catalog
        rendering (sourced from the same PLAN_TOOLS_SPEC), so the planner system prompt
        is byte-identical while the bundle is the single source of truth."""
        from .bundles import get_bundle

        return get_bundle("planning").plan_tool_catalog_text()

    @staticmethod
    def _upstream_memory_block(
        prior_memory: Optional[Sequence[Mapping[str, Any]]],
    ) -> str:
        """Render the UPSTREAM RESEARCH MEMORY the planner reasons over (d285 SB-3).

        ``prior_memory`` is the (summary, memory_index) pairs the planner RECEIVED from
        upstream nodes' finalize (SB-2) — the DATA it reasons over to choose, per step,
        whether to CONTINUE a prior research line (reuse its index) or start a FRESH one
        (``<<NEW>>``). d10-clean: this is DATA (a summary + an index), never a spec body.

        Empty string when there is no upstream (the SEED authoring — every step then
        opens a fresh ``<<NEW>>`` line), so the seed planner's turns stay byte-identical.
        SB-3 only surfaces this block as a SEAM; SB-4 wires the served handoff that
        POPULATES it from compose-task (a test injects it directly to prove the choice)."""
        pairs = [p for p in (prior_memory or []) if isinstance(p, Mapping)]
        rows = []
        for p in pairs:
            idx = str(p.get("memory_index") or "").strip()
            summ = str(p.get("summary") or "").strip()
            if not idx and not summ:
                continue
            rows.append(f"  - index={idx or '(unset)'}: {summ or '(no summary)'}")
        if not rows:
            return ""
        return (
            "\n\nUPSTREAM RESEARCH MEMORY (what earlier steps already gathered — each a "
            "(summary, index) pair). When a NEW step EXTENDS one of these research lines, "
            "set that step's 'memory_index' to the matching index to CONTINUE it; when a "
            "step starts a DISTINCT line, use \"<<NEW>>\". Reason over the summaries to "
            "decide:\n" + "\n".join(rows) + "\n"
        )

    def _directive_block(self) -> str:
        """The phase-specific authoring directive, rendered for a per-turn user prompt.

        Empty (no trailing text) when no directive was set — so the research/gather
        planner's turns stay byte-identical. The write phase sets it to the SOURCE-ID
        mandate so every authoring turn (initial / observation / finalize) carries the
        reminder, not just the goal (s14/a9 — the per-turn lever is the strongest for E4B)."""
        return f"{self.authoring_directive}\n\n" if self.authoring_directive else ""

    def _system(self, goal: str) -> str:
        """The body-free SEED context shared by every tool-call turn (asserted d10)."""
        ctx = self.factory.planner_context(goal)
        self.factory.assert_body_free(ctx)
        shape_line = ""
        if self.shape_name:
            shape_line = (
                f"\n\nPLAN SHAPE: {self.shape_name}. {self.shape_description}\n"
                "Author the steps and their 'depends_on' edges so they fit this "
                "shape: INDEPENDENT steps that can run at the same time share NO "
                "edge (depends_on []), while a step that needs an earlier step's "
                "result lists that step's id in depends_on. A request for a "
                "COMPOSITIONAL plan (e.g. 'linear plus modular parallel') means BOTH "
                "patterns coexist: some steps run in parallel (no edges between them) "
                "AND some run in sequence (chained by depends_on) — author the real "
                "mix, not a single flat line."
            )
        # F5: the user EXPLICITLY named these specialization(s) — a HARD instruction
        # to bind them (via add_step 'spec'/'specs' or a later set_node_spec). An
        # output-style spec belongs on the node that produces the FINAL deliverable;
        # a research spec on the gather node(s). (A finalization pass stamps any the
        # model still leaves unbound onto the terminal node, so this is the primary
        # mechanism + a safety net.)
        requested_line = ""
        if self.requested_specs:
            names = ", ".join(self.requested_specs)
            requested_line = (
                f"\n\nUSER-REQUESTED SPECIALIZATIONS: the user explicitly asked to "
                f"use [{names}]. You MUST set 'spec' (or 'specs') to this/these "
                "specialization(s) on the step(s) they apply to — for an "
                "output-style specialization, the step that produces the final "
                "deliverable; do not drop it in favour of another."
            )
        # PRECEDENCE (d341): when the SELECTED shape supplies a decompose_methodology, it DEFINES
        # the steps to author and REPLACES the generic gather→combine→deliver flow guidance below
        # (which becomes a fallback for methodology-less shapes). The model reasons over the
        # methodology; the engine authors nothing. Empty → byte-identical to the pre-d341 prompt.
        methodology = self.shape_decompose_methodology
        methodology_block = ""
        # NEUTRAL FALLBACK (RP-AUDIT F2) — this generic authoring guidance applies ONLY when the
        # selected shape supplies NO decompose_methodology. It is deliberately TOPOLOGY-NEUTRAL: it
        # tells the model to author the steps the GOAL actually needs and wire real dependencies,
        # WITHOUT baking a gather→combine→deliver mold. A read→write, single-step, or non-gather
        # goal is therefore NOT pushed into a gather shape. (The pre-F2 version hardcoded "a
        # modular-parallel plan is INDEPENDENT sub-tasks FOLLOWED BY one FINAL step that combines
        # their outputs" — a gather-specific assumption every methodology-less shape inherited; d341
        # forbids the engine baking a fixed flow. A shape whose flow IS gather→combine now carries
        # that in its own decompose_methodology, which takes precedence below.)
        flow_guidance = (
            "- Author the SMALLEST correct set of steps for THIS goal: decompose it into the "
            "distinct pieces of work its OWN outcome requires — no more, no fewer. Some goals "
            "need a single step; some need a step that reads or produces something and a later "
            "step that uses it; some name several independent sub-tasks. Do NOT force the goal "
            "into a fixed gather-then-combine shape.\n"
            "- Wire dependencies by REASONING: a step that needs an earlier step's result lists "
            "that step's id in depends_on; steps that need nothing from each other share NO edge "
            "(depends_on []) and may run at the same time. Never author a step that duplicates "
            "one already authored.\n"
            "- RESEARCH MEMORY (memory_index): a step that searches/fetches/reads works in a "
            "research memory. Set 'memory_index' to \"<<NEW>>\" to OPEN a fresh research line (the "
            "default for a new gather step), or to an existing INDEX to CONTINUE the research a "
            "prior step built — choose by reasoning over any UPSTREAM RESEARCH MEMORY you were "
            "given (continue the index whose summary this step extends).\n"
        )
        final_combine_guidance = (
            "- Make the LAST step the one that DELIVERS the goal's outcome; when it uses several "
            "earlier steps, its depends_on lists each of their ids. After authoring the "
            "delivering step, call finalize_plan.\n\n"
        )
        if methodology:
            methodology_block = (
                f"\n\nAUTHORING METHODOLOGY (selected shape '{self.shape_name}') — FOLLOW THIS. It "
                "DEFINES exactly which steps to author and TAKES PRECEDENCE over the generic "
                "gather/combine/deliver guidance below:\n" + methodology + "\n"
            )
            flow_guidance = ""
            final_combine_guidance = (
                "- After authoring the step(s) the methodology above calls for, call "
                "finalize_plan.\n\n"
            )
        # FORMAT-NEUTRAL BINDING (RP-AUDIT F6 / d352): the spec/tool selection GUIDANCE below
        # carries NO per-format example (no "an HTML request gets the HTML writer, a Markdown
        # request the Markdown writer" pin). The engine states the GENERIC rule — bind the writer
        # spec whose ADVERTISED format matches the goal's requested format — and the format→writer
        # MAPPING lives ONLY in the writer SPEC descriptions (html-writer / markdown-writer /
        # section-html-writer advertise which format each produces). The model reasons over those
        # descriptions to pick the writer; the engine pins no format (d317). The format-bleed
        # HYGIENE rule (never a document-format writer on a gather step) stays, kept format-neutral.
        return with_identity(
            ctx["factory"]["description"]
            + shape_line
            + requested_line
            + methodology_block
            + "\n\nBUILD THE PLAN BY CALLING TOOLS, one per reply. First reason about "
            "which SHAPE fits and which INPUT / PROCESSING / OUTPUT specializations "
            "to use, then issue tool calls to construct the DAG. Each reply is "
            "EXACTLY ONE tool call as STRICT JSON:\n"
            '{"tool": "<tool name>", "args": { ... }}\n'
            "No prose, no code fences — only that JSON object.\n\n"
            + self._tool_catalog_text()
            + "\n\nGUIDANCE:\n"
            "- Call seed_plan FIRST, then add_step for each step, then finalize_plan.\n"
            + flow_guidance
            + "- If a step's action is one an available tool performs (search, fetch, "
            "read/write a file, send email), set 'tool' to that tool's exact name.\n"
            "- Pick spec/specs only when a listed specialization genuinely fits the "
            "work THAT step does. A research/gather step (it searches, fetches, "
            "reads, takes notes) takes a research/analysis spec (e.g. "
            "research-analyst) or NONE — NEVER a document-format spec: putting "
            "a document-format writer on a gather step makes it emit the document "
            "instead of gathering notes. When the goal names a specific output FORMAT, "
            "bind — ONLY on the final WRITE step — the output-style writer spec whose "
            "ADVERTISED format matches the goal's requested format (read the writer "
            "specializations' own descriptions and pick the one that produces the "
            "requested format); never bind a document-format writer to a gather step.\n"
            "- DELIVERY: by default present the result in chat, or save it with "
            "file_write. Use send_mail ONLY when the goal EXPLICITLY asks to be "
            "emailed — never email unprompted.\n"
            + final_combine_guidance
            + "REGISTERED SPECIALIZATIONS (lookup — names + descriptions only):\n"
            + json.dumps(ctx["specializations"], indent=2)
            + "\n\nAVAILABLE TOOLS (names + descriptions only):\n"
            + json.dumps(ctx["tools"], indent=2)
        )

    def _initial_user(
        self, goal: str,
        prior_memory: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> str:
        """The first turn: the goal + the decision procedure for building the plan.

        ``prior_memory`` (d285 SB-3) is the upstream (summary, index) pairs the planner
        reasons over to choose each step's ``memory_index`` (continue vs <<NEW>>); empty/
        None → no upstream block (seed authoring, byte-identical)."""
        # PRECEDENCE (d341): when the selected shape supplies a decompose_methodology, it REPLACES
        # the generic gather→combine→deliver decision procedure (which stays the fallback for
        # methodology-less shapes). The model reasons over the methodology; the engine authors
        # nothing. Empty → byte-identical to the pre-d341 procedure.
        methodology = self.shape_decompose_methodology
        if methodology:
            procedure = (
                "Build the plan now, following THIS authoring methodology for the selected "
                f"'{self.shape_name}' shape — it DEFINES exactly which steps to author and TAKES "
                "PRECEDENCE over any generic gather/combine/deliver pattern:\n"
                + methodology + "\n\n"
                "Mechanics: call seed_plan first, then add_step for each step the methodology "
                "calls for (set 'tool' and 'spec' as it directs), then finalize_plan. ONE tool "
                "call per reply.\n"
            )
        else:
            # NEUTRAL FALLBACK (RP-AUDIT F2) — topology-neutral decision procedure for a
            # methodology-less shape. It guides the model to author the steps THIS goal needs and
            # wire real dependencies, WITHOUT the pre-F2 gather→combine→deliver recipe ("list the
            # distinct items … a gather step per item … a FINAL step that combines and delivers"),
            # which forced every methodology-less goal (incl. a read→write codebase task) into a
            # gather mold. A shape whose flow genuinely IS gather→combine carries that in its own
            # decompose_methodology (the branch above). Output-FORMAT / spec binding guidance stays
            # in the always-present _system GUIDANCE, so it is not lost by neutralizing here.
            procedure = (
                "Build the plan now. Decision procedure:\n"
                "1. Work out the steps THIS goal actually needs: read its outcome and break it "
                "into the distinct pieces of work required to reach it — no more, no fewer. Some "
                "goals need one step; some need a step that reads or produces something and a "
                "later step that uses it; some name several independent sub-tasks. Do NOT assume "
                "a gather-then-combine shape.\n"
                "2. Call seed_plan to open the plan.\n"
                "3. Call add_step for each step. For each: set 'depends_on' by reasoning (the ids "
                "of the steps whose result this step needs; [] for an independent step), set "
                "'tool' to the tool that performs its action (search/fetch, read/write a file, "
                "send email) or leave it blank for a reasoning/worker step, and set 'spec' only "
                "when a listed specialization genuinely fits that step's work — a research/gather "
                "step (it searches, fetches, reads, takes notes) takes a research/analysis spec "
                "or NONE, never a document-format writer spec (that makes it emit the document "
                "instead of gathering notes); bind an output-format writer spec ONLY on the final "
                "step that WRITES the deliverable. Make the LAST step the one that DELIVERS the "
                "goal's outcome (present it in chat, save it with file_write, or send_mail ONLY "
                "if the goal EXPLICITLY asked to be emailed).\n"
                "4. Call finalize_plan.\n"
            )
        return (
            f"GOAL: {goal}\n\n"
            + self._upstream_memory_block(prior_memory)
            + procedure
            + self._directive_block()
            + "Reply with ONE tool call now (start with seed_plan)."
        )

    def _observation_user(self, obs: Mapping[str, Any]) -> str:
        """Render a builder observation back to the model as the next user turn.

        NEUTRAL + METHODOLOGY-AWARE (RP-AUDIT F5, mirroring F1/F2/RP-4c d341). The
        per-step "issue the NEXT tool call" nudge is TOPOLOGY-NEUTRAL: it asks the model
        to author the next step THIS goal needs and to finalize once the plan has all the
        steps the goal needs — WITHOUT the pre-F5 gather framing ("finalize_plan if every
        distinct item is covered", which assumed a gather→combine flow and mis-framed a
        read→write / single-step / non-gather goal). When the SELECTED shape supplies a
        ``decompose_methodology`` the nudge DEFERS to it (author the next step that
        methodology still calls for, finalize when the methodology's steps are all
        present) — TAKING PRECEDENCE over the neutral guidance, exactly like ``_system`` /
        ``_initial_user`` / ``_finalize_user``. Generic + shape-agnostic (a presence check
        on the methodology field, NO shape-name/spec-name conditional); the engine renders
        shape text, the model authors. Empty methodology → the neutral topology-agnostic
        nudge."""
        steps = obs.get("steps") or []
        lines = [f"OBSERVATION: {obs.get('note', '')}"]
        if steps:
            lines.append("STEPS SO FAR:")
            for s in steps:
                dep = ", ".join(s.get("depends_on") or []) or "-"
                lines.append(
                    f"  {s['id']}: {s['task']}  "
                    f"[tool={s.get('tool') or '-'} spec={s.get('spec') or '-'} "
                    f"depends_on={dep}]"
                )
        else:
            lines.append("No steps authored yet.")
        if self.authoring_directive:
            lines.append(self.authoring_directive)
        # PRECEDENCE (RP-AUDIT F5 / d341): when the shape supplies a decompose_methodology,
        # the nudge defers to it (the model authors the next step it calls for); otherwise
        # a topology-neutral nudge that assumes NO gather→combine shape. The methodology
        # itself is already rendered in the system prompt (passed every turn), so this turn
        # only POINTS to it — it never re-authors or bakes a fixed flow.
        if self.shape_decompose_methodology:
            lines.append(
                "Issue the NEXT tool call, following the selected "
                f"'{self.shape_name}' shape's authoring methodology: add_step for the "
                "next step it calls for, set_node_spec to refine a step's "
                "specialization, or finalize_plan once the plan has every step the "
                "methodology calls for and is complete. Reply with ONE tool call."
            )
        else:
            lines.append(
                "Issue the NEXT tool call: add_step for the next step THIS goal needs, "
                "set_node_spec to refine a step's specialization, or finalize_plan once "
                "the plan has all the steps the goal needs and is complete. Reply with "
                "ONE tool call."
            )
        return "\n".join(lines)

    def _finalize_user(self, goal: str, builder: PlanBuilder) -> str:
        """Force the closing step when the model is repeating instead of finalizing.

        The 4.6B model cannot reliably do the 'which items are still uncovered'
        set-difference statelessly, so once it starts re-emitting an already-authored
        step (the duplicate signal) the gather phase is DONE — every distinct item is
        covered. This turn directs it to author ONLY the closing step(s) (the eda-base3
        FSM plays this controller role for the strong model). NOT few-shot coercion —
        it carries no example plan and works for any goal/shape.

        SHAPE-METHODOLOGY-AWARE (RP-4c/d341, F1): like ``_system`` / ``_initial_user``,
        when the selected shape supplies a ``decompose_methodology`` the turn RENDERS it
        and DEFERS to it (the model authors the closing step(s) per the shape's own flow),
        TAKING PRECEDENCE over — and REPLACING — the hardcoded gather→combine→deliver +
        output-format + delivery-tool recipe, which stays the FALLBACK for methodology-
        less gather shapes. Generic + shape-agnostic (presence check, no spec-name/flow
        conditional); the engine renders shape text, the model authors."""
        lines = [f"GOAL: {goal}", "", "STEPS ALREADY AUTHORED:"]
        for s in builder._state_summary():
            dep = ", ".join(s.get("depends_on") or []) or "-"
            lines.append(
                f"  {s['id']}: {s['task']}  [tool={s.get('tool') or '-'} "
                f"depends_on={dep}]"
            )
        # PRECEDENCE (RP-4c/d341, F1): when the SELECTED shape supplies a decompose_methodology,
        # this forced-finalize turn RENDERS that methodology and DEFERS to it — the model authors
        # the remaining/closing step(s) per the SHAPE's own flow, TAKING PRECEDENCE over (and
        # REPLACING) the hardcoded gather→combine→deliver + output-format-writer + delivery-tool
        # recipe below (which stays the FALLBACK only for methodology-LESS gather shapes). This
        # mirrors _system / _initial_user EXACTLY (same field, same precedence framing, same
        # research_tree d161/d170 substitution posture): behaviour lives in the SHAPE (definition
        # layer); the substitution is generic + shape-agnostic (a presence check on the methodology
        # field, NOT a spec-name/flow conditional); the engine renders shape text, the model authors
        # the step(s). Empty → byte-identical to the pre-F1 hardcoded finalize turn.
        methodology = self.shape_decompose_methodology
        if methodology:
            lines.extend(
                [
                    "",
                    "The plan is NOT yet finalized — author the remaining step(s) to "
                    "COMPLETE it, following THIS authoring methodology for the selected "
                    f"'{self.shape_name}' shape. It DEFINES which step(s) still need "
                    "authoring and TAKES PRECEDENCE over any generic gather/combine/"
                    "deliver pattern:\n"
                    + methodology + "\n\n"
                    "Author ONLY the step(s) the methodology still calls for (do NOT "
                    "repeat any step above; set 'tool'/'spec' exactly as the methodology "
                    "directs), then call finalize_plan. Reply with ONE tool call now.",
                ]
            )
        else:
            # NEUTRAL FALLBACK (RP-AUDIT F2) — topology-neutral closing turn for a methodology-less
            # shape. The pre-F2 version hardcoded "the FINAL step that combines the gathered results
            # … depends_on MUST list EVERY gather step id … the HTML writer for HTML" — a gather→
            # combine→deliver + output-format recipe the engine must NOT bake (d341). This directs
            # the model to author the remaining step(s) that finish the goal — usually the single
            # delivering step — wiring its real dependencies, without a gather assumption. Output-
            # FORMAT / spec binding guidance is carried by the always-present _system prompt (passed
            # on the same call), so neutralizing here does not lose it.
            lines.extend(
                [
                    "",
                    "The steps authored above do not yet COMPLETE the goal. Author the "
                    "remaining step(s) needed to finish it — most often the single step "
                    "that DELIVERS the goal's outcome using the earlier steps' results. "
                    "Set that step's depends_on to the ids of the earlier steps it uses; "
                    "set 'tool' to the delivery tool the goal asks for (file_write to save "
                    "a file; send_mail ONLY if the goal EXPLICITLY asked to be emailed; "
                    "else leave it \"\"). Do NOT repeat any step already authored above. "
                    "Reply with ONE add_step tool call.",
                ]
            )
        if self.authoring_directive:
            lines.append("")
            lines.append(self.authoring_directive)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # one tool-call turn
    # ------------------------------------------------------------------ #
    async def _call_model(
        self, system: str, convo: list[Mapping[str, Any]], opts: Mapping[str, Any]
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        """Run ONE native reasoning call → ``(parsed_tool_call, raw_text)``.

        Uses the assemble→call→parse+repair chain (so the bounded malformed-JSON
        self-heal still applies), offloaded off the event loop via
        :func:`run_blocking_in_span` (the never-freeze fix) with the otel context
        re-attached so each per-turn llm span nests under the authoring span. No
        ``format`` schema is passed — the model reasons freely (think=True) and the
        transport's JSON-extraction interceptor + the structured_output stage parse
        the emitted tool call, so a reasoned ``depends_on`` is never schema-dropped."""
        chain = Chain()
        chain.use(prompt_assembly())
        chain.use(call_stage(self.transport, **dict(opts)))
        chain.use(
            structured_output(self.transport, max_repair_attempts=self.max_repair_attempts)
        )
        ctx = Context(system=system, history=list(convo), transport=self.transport)
        ctx = await run_blocking_in_span(chain.run, ctx)
        parsed = ctx.structured
        return (dict(parsed) if isinstance(parsed, Mapping) else None), ctx.raw_output

    @staticmethod
    def _parse_tool_call(parsed: Mapping[str, Any]) -> Optional[tuple[str, dict[str, Any]]]:
        """Pull ``(tool_name, args)`` out of the model's parsed JSON, leniently.

        Accepts the canonical ``{"tool": <name>, "args": {...}}`` AND common slips:
        ``{"tool": <name>, ...other keys are the args}``, a bare
        ``{<tool_name>: {...args}}``, or — when the model skipped the wrapper and
        emitted the add_step args directly — a dict carrying a ``task`` is read as an
        add_step. Returns ``None`` only when no tool can be recovered."""
        if not isinstance(parsed, Mapping):
            return None
        tool = parsed.get("tool") or parsed.get("name") or parsed.get("tool_name")
        if isinstance(tool, str) and tool.strip():
            name = tool.strip()
            args = parsed.get("args") or parsed.get("arguments") or parsed.get("parameters")
            if not isinstance(args, Mapping):
                args = {
                    k: v
                    for k, v in parsed.items()
                    if k not in ("tool", "name", "tool_name", "args", "arguments", "parameters")
                }
            return name, dict(args)
        # Bare {<tool_name>: {...}} form.
        for key, val in parsed.items():
            if str(key).strip() in PLAN_TOOL_NAMES and isinstance(val, Mapping):
                return str(key).strip(), dict(val)
        # The model emitted add_step args directly (no wrapper).
        if "task" in parsed:
            return "add_step", dict(parsed)
        return None

    # ------------------------------------------------------------------ #
    # F5 / F2 finalization passes (over the authored node set) — unchanged
    # ------------------------------------------------------------------ #
    def _apply_requested_specs(
        self, authored: list[dict[str, Any]], span: Any
    ) -> int:
        """GUARANTEE a user-named specialization is bound somewhere (F5).

        The tool-call authoring is TOLD (in :meth:`_system`) that the user explicitly
        requested certain specialization(s), but the 4.6B model can forget to bind
        one. This finalization pass closes that gap STRUCTURALLY: if the user named
        spec(s) and NONE appears bound on any authored node, they are stamped onto the
        plan's TERMINAL node(s) — the node(s) nothing else depends on (the final
        deliverable an output-style specialization governs). Generic + structural; a
        no-op when the user named none, or when the model already bound at least one
        requested spec. Returns the number of nodes stamped."""
        requested = self.requested_specs
        if not requested:
            return 0
        wanted = set(requested)
        already = any(
            (record.get("spec") in wanted)
            or bool(wanted & set(record.get("specs") or []))
            for record in authored
        )
        if already:
            return 0  # the model bound a requested spec itself → honor its choice
        depended_on: set[str] = set()
        for record in authored:
            for dep in record.get("depends_on") or []:
                depended_on.add(str(dep))
        sinks = [r for r in authored if r["id"] not in depended_on] or authored[-1:]
        applied = 0
        for record in sinks:
            existing = list(record.get("specs") or [])
            if record.get("spec") and record["spec"] not in existing:
                existing.insert(0, record["spec"])
            for name in requested:
                if name not in existing:
                    existing.append(name)
            record["specs"] = existing
            record["spec"] = existing[0]
            record["needs_spec"] = None  # a named, registered spec is not "missing"
            applied += 1
            try:
                span.set_attribute(
                    f"planner.node.{record['id']}.requested_spec",
                    ", ".join(requested),
                )
            except Exception:
                pass
        if applied:
            try:
                span.set_attribute("planner.requested_specs", list(requested))
                span.set_attribute("planner.requested_spec_applied", applied)
            except Exception:
                pass
        return applied

    # ------------------------------------------------------------------ #
    # RP-3b (d311/d319/d328) — the three engine-authored STRUCTURE repairs are RETIRED.
    #
    # Three finalization passes used to CRUTCH the planner's authoring reliability by
    # having the ENGINE author DAG/spec/format structure the model was expected to
    # produce: _apply_default_research_spec (stamped a research spec on null-spec gather
    # nodes), _enforce_terminal_research_edge (auto-added the writer<-research edge on a
    # disconnected terminal), and _enforce_output_format_spec (stamped the format writer
    # on the terminal). Each is now retired, COUPLED to a live measure (RP-3b probe, 20
    # E4B :11434 trials): the planner authors all three axes ITSELF 100% reliably —
    # gather-node spec 12/12, terminal writer<-research edge 12/12, terminal format
    # writer 16/16 (incl. the deep-research write phase). Per d311/d319 the engine now
    # authors NO DAG/spec/format structure here; reliability comes from the planner
    # prompt + the selected shape (the definition layer), never an engine stamp/flag/
    # spec-name conditional. The helpers those passes used (_is_research, _ancestors,
    # _requested_output_format, _OUTPUT_FORMAT_WRITERS/_VARIANTS, default_research_spec)
    # are removed with them. F5 _apply_requested_specs (honours a user-NAMED spec) is OUT
    # of scope and preserved.
    #
    # RP-AUDIT F5 (d50 echo RETIRED, d319/d341): the ``_echo_literal_filename`` pass used
    # to ENGINE-AUTHOR a "write-the-file-as-<name>" sentence into the terminal node's TASK
    # and assumed a FILE-WRITER SINK (it classified sinks and skipped gather nodes). Both
    # are anti-fab violations — the engine authoring node task content + baking a delivery-
    # sink assumption. It is RETIRED. Filename-honoring is PRESERVED by the already-present
    # goal-carry: the user's named file is carried VERBATIM on ``PlanDAG.goal`` (= every
    # node's overall-goal, d39), and the writer derives its output path via
    # ``synth_tools.derive_output_path(overall_goal, node.task)`` — which reads the explicit
    # filename straight from the GOAL (the user's own words) regardless of the node task. So
    # ``cats.html`` still reaches disk as the user asked, with NO engine-authored task text
    # and NO file-sink assumption — the delivery sink is goal/shape-implied (the model
    # authors it; ``derive_output_path`` only applies when a node actually writes a file).

    # ------------------------------------------------------------------ #
    # the authoring loop (tool calls → assemble DAG)
    # ------------------------------------------------------------------ #
    async def plan(
        self, goal: str,
        *,
        prior_memory: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> PlanResult:
        """Author a validated :class:`PlanDAG` for ``goal`` via planner tool calls.

        ``prior_memory`` (d285 SB-3) is the OPTIONAL upstream (summary, memory_index)
        pairs the planner reasons over to choose each step's ``memory_index`` — CONTINUE a
        prior research line (reuse its index) or start a FRESH one (``<<NEW>>``). None/empty
        → no upstream block, the authoring is byte-identical to the seed path. This is the
        SB-3 SEAM; SB-4 wires the served handoff that populates it from compose-task (a test
        injects it directly to prove the choice resolves through SB-1's store pre-SB-4).

        Loops up to ``2*max_nodes + 4`` turns: each turn the model issues ONE tool
        call (a small native reasoning decision), the :class:`PlanBuilder` applies it
        and returns an observation, and the loop threads that observation back. The
        loop ends when the model calls ``finalize_plan`` (its own "plan complete"
        signal), the cap is hit, or it stalls. The authored nodes are parsed through
        the SAME :meth:`AbstractPlanFactory.parse_dag` the one-shot path uses, so the
        DAG is validated (unique ids, resolvable refs, acyclic) identically. Raises
        :class:`MalformedOutputError` if NO usable node was authored, so the outer
        self-heal can re-plan exactly as it does for a malformed one-shot plan."""
        system = self._system(goal)
        self.last_context = self.factory.planner_context(goal)
        opts = dict(self.call_opts)  # NO format schema (d34): reasoned edges survive.

        builder = PlanBuilder(
            spec_names=self.spec_names,
            tool_names=self.tool_names,
            shape_name=self.shape_name,
            shape_description=self.shape_description,
            max_nodes=self.max_nodes,
        )
        self.last_builder = builder
        max_turns = 2 * self.max_nodes + 4

        tracer = get_tracer("agent_runtime.incremental")
        with tracer.start_as_current_span("planner.incremental") as span:
            span.set_attribute("planner.goal", str(goal)[:1000])
            span.set_attribute("planner.shape", self.shape_name or "")
            span.set_attribute("planner.max_nodes", self.max_nodes)
            span.set_attribute("planner.authoring", "tool-driven")

            convo: list[dict[str, Any]] = [
                {"role": "user", "content": self._initial_user(goal, prior_memory)}
            ]
            tool_call_count = 0
            unproductive = 0  # consecutive turns that produced no usable progress
            saw_duplicate = False

            for turn in range(max_turns):
                parsed, raw = await self._call_model(system, convo, opts)
                convo.append({"role": "assistant", "content": raw or ""})
                call = self._parse_tool_call(parsed) if parsed is not None else None
                if call is None:
                    unproductive += 1
                    span.set_attribute(f"planner.turn.{turn + 1}.unparsed", True)
                    if unproductive >= 2:
                        break
                    convo.append(
                        {
                            "role": "user",
                            "content": (
                                "That was not a single JSON tool call. Reply with "
                                'EXACTLY {"tool": "<name>", "args": {...}} and nothing '
                                "else."
                            ),
                        }
                    )
                    continue

                tool, args = call
                tool_call_count += 1
                obs = builder.dispatch(tool, args)
                span.set_attribute(f"planner.turn.{turn + 1}.tool", tool)
                if obs.get("duplicate"):
                    saw_duplicate = True
                # Productive = a state-changing call (a new step, a spec set, a seed,
                # or an explicit finalize). A rejected/duplicate call is unproductive.
                if obs.get("ok"):
                    unproductive = 0
                else:
                    unproductive += 1

                if tool == "finalize_plan":
                    break
                if unproductive >= 2:
                    break
                # s15/a18 (d189): the builder.dispatch RESULT (the authored-steps observation)
                # is a tool function-result, NOT a fresh USER instruction — feed it back role
                # 'tool' so the planner reasons over the running plan state instead of reading
                # its own tool output as a new user turn. The "not a single JSON tool call"
                # nudge above and the finalize prompt stay role 'user' (genuine instructions).
                convo.append(
                    {"role": "tool", "content": self._observation_user(obs)}
                )

            span.set_attribute("planner.tool_calls", tool_call_count)

            # FORCED FINALIZE (deterministic loop control): the model stalled (kept
            # repeating / stopped issuing usable calls) without authoring a closing
            # combine step — make ONE explicit finalize call that authors only the
            # combine/deliver node. Only when there is something to combine (2+ steps)
            # and no node already depends on another (no combine sink yet).
            if (
                not builder.finalized
                and len(builder.nodes) >= 2
                and saw_duplicate
                and not any(n["depends_on"] for n in builder.nodes)
            ):
                parsed, raw = await self._call_model(
                    system,
                    [{"role": "user", "content": self._finalize_user(goal, builder)}],
                    opts,
                )
                call = self._parse_tool_call(parsed) if parsed is not None else None
                if call is not None:
                    tool, args = call
                    # Force the closing call to add_step even if the model labelled it
                    # otherwise (the prompt asked for exactly the final add_step).
                    if tool != "add_step" and "task" in args:
                        tool = "add_step"
                    obs = builder.dispatch(tool, args)
                    if obs.get("ok"):
                        span.set_attribute("planner.forced_finalize", True)

            span.set_attribute("planner.node_count", len(builder.nodes))

            # F5 REQUESTED-SPEC PASS: guarantee a user-NAMED spec (from the model-driven
            # ShapeSelector) is bound on the terminal/delivery node when the model forgot
            # to. Out of RP-3b scope (honours an explicit user request, not engine-authored
            # structure) — preserved.
            self._apply_requested_specs(builder.nodes, span)
            # RP-3b (d311/d319/d328): the F2 default-research-spec pass, the d28 terminal-
            # research-edge pass, and the b5 output-format pass are RETIRED — the planner
            # authors the gather-node spec, the writer<-research edge, and the format
            # writer ITSELF (measured 100% reliable on live E4B, RP-3b probe). The engine
            # authors NO DAG/spec/format structure here.
            # RP-AUDIT F5: the d50 ``_echo_literal_filename`` pass is RETIRED — filename-
            # honoring is preserved by the goal-carry (PlanDAG.goal → derive_output_path),
            # so the engine no longer authors task prose or assumes a file-writer sink.

            if not builder.nodes:
                raise MalformedOutputError(
                    "tool-driven authorer produced no usable nodes "
                    f"(shape={self.shape_name!r}, max_nodes={self.max_nodes})"
                )

            structured = builder.to_structured()
            # d7 SAFE PARSE: repair dangling/self depends_on edges (a phantom-id edge
            # the model may emit) instead of rejecting them — parse_dag_safe drops only
            # unresolvable/self refs (degrading gracefully) and still RAISES for genuine
            # invalidity (duplicate ids, a real cycle, an empty plan), which the outer
            # self-heal handles as retry-on-reject. The tool-call loop already clamps
            # deps backward (acyclic-by-construction), so ``dangling_repairs`` is
            # normally empty (byte-identical to the strict path) — this is the finalize
            # backstop guaranteeing a dangling edge never surfaces as a user-visible
            # failure (d7), validated across fresh model-loads.
            try:
                dag, dangling_repairs = self.factory.parse_dag_safe(structured)
            except PlanError as exc:
                # A node was still structurally invalid (dup id / real cycle) → surface
                # it as a malformed plan for the self-heal — never ship an invalid DAG.
                raise MalformedOutputError(
                    f"tool-driven authorer assembled an invalid DAG: {exc}"
                ) from exc
            if dangling_repairs:
                try:
                    span.set_attribute(
                        "planner.dag.dangling_repairs", len(dangling_repairs)
                    )
                    span.set_attribute("planner.dag.repairs", list(dangling_repairs))
                except Exception:
                    pass

            self.last_nodes = builder.nodes
            result = PlanResult(
                dag=dag,
                context=self.last_context,
                raw=json.dumps(builder.calls),
                structured=structured,
                repair={
                    "dangling_edges": list(dangling_repairs),
                },
            )
            self.last_result = result
            return result


__all__ = [
    "IncrementalPlanner",
    "DEFAULT_MAX_NODES",
    "DEFAULT_NODE_NUM_PREDICT",
]
