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
catalog) — never a compiled spec body — asserted body-free before the loop. The F2
default-research-spec and F5 requested-spec finalization passes run over the authored
node set exactly as before.
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
from .synth_tools import explicit_filename
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

# The tool names that mark a node as a RESEARCH/GATHER node (F2). A node that fires
# one of these reads the web — and the QUALITY of what it produces is governed by a
# research ruleset (d13: report the REAL findings by READING the fetched articles,
# never describe the search-results / source list). These are the nodes that, left
# spec-less by the per-node authorer, degrade to the source-list-summary anti-pattern
# (the empty/thin emailed news section). Delivery tools (send_mail/file_write) are
# deliberately NOT here: a delivery node is null-spec BY DESIGN — its content is
# grounded deterministically in upstream node text (d11), not researched — so the
# research ruleset does not apply to it. Generic + role-structural, never per-scenario.
DEFAULT_RESEARCH_TOOLS = ("web_search", "web_fetch")

# OUTPUT-FORMAT writers (s8/b5). When the GOAL explicitly names a deliverable
# format, the terminal write/synthesize node MUST carry that format's output-style
# spec so the deliverable comes back in the requested form (the d13/B2 fix: an HTML
# request must produce HTML, a Markdown request Markdown). The PROMPT tells the
# model to bind it (factory description + the per-turn final-step instruction), but
# E4B intermittently binds the analysis spec (research-analyst) on a "synthesize"
# node instead of the format writer — measured ~1/3 of runs, either direction. So,
# exactly like the F5 requested-spec guarantee, a finalization pass STAMPS the
# format writer when the model left it off. ``name -> (writer spec, goal regex)`` —
# the writer is only stamped when it is an actually-registered specialization, so a
# project without these seeds is a clean no-op. Mutually exclusive (an HTML request
# never keeps a Markdown writer and vice-versa).
_OUTPUT_FORMAT_WRITERS: tuple[tuple[str, str, str], ...] = (
    ("html", "html-writer", r"\bhtml\b|\.html\b|\bweb\s?page\b"),
    ("markdown", "markdown-writer", r"\bmarkdown\b|\.md\b"),
)


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
        default_research_spec: str = "",
        requested_specs: Sequence[str] = (),
        research_tools: Sequence[str] = DEFAULT_RESEARCH_TOOLS,
        max_nodes: int = DEFAULT_MAX_NODES,
        node_num_predict: int = DEFAULT_NODE_NUM_PREDICT,
        max_repair_attempts: int = 2,
        call_opts: Optional[dict[str, Any]] = None,
        inject_review: bool = False,
    ) -> None:
        self.transport = transport
        self.factory = factory
        # P2.2/P2.5 (d132.B) — FRAMEWORK-INJECTED REVIEW. When True, the authored plan
        # is passed through :func:`review_injection.inject_reviews` at build time (the
        # :class:`PlanBuilder` opt-in), so the FRAMEWORK (not the planner) turns each work
        # node into a work-then-review pair and appends a final review. Default False keeps
        # the authored plan byte-identical. The served flagship report route turns this ON
        # only behind the reversible P2.5 generic-report flag (parity-gated).
        self.inject_review = bool(inject_review)
        self.spec_names = [str(s) for s in spec_names if str(s).strip()]
        self.tool_names = [str(t) for t in tool_names if str(t).strip()]
        self.shape_name = str(shape_name or "")
        self.shape_description = str(shape_description or "")
        # F2 DEFAULT RESEARCH SPEC: the generic, role-appropriate research
        # specialization stamped onto any null-spec GATHER node (see
        # :data:`DEFAULT_RESEARCH_TOOLS`). The NAME is supplied by the caller (the
        # live route passes ``specialization.seed.DEEP_RESEARCH_SPEC`` —
        # ``research-analyst``), so this generic authorer hard-codes no spec name.
        # Applied only when the name is an actually-registered specialization.
        self.default_research_spec = str(default_research_spec or "").strip()
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
        self.research_tools = {
            str(t).strip().lower() for t in (research_tools or ()) if str(t).strip()
        }
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
        """Render the four plan-building tools + their args for the system prompt."""
        lines = ["PLAN-BUILDING TOOLS (call ONE per reply):"]
        for spec in PLAN_TOOLS_SPEC:
            lines.append(f"- {spec['name']}: {spec['description']}")
            for arg, meaning in spec["args"].items():
                lines.append(f"    {arg}: {meaning}")
        return "\n".join(lines)

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
        return with_identity(
            ctx["factory"]["description"]
            + shape_line
            + requested_line
            + "\n\nBUILD THE PLAN BY CALLING TOOLS, one per reply. First reason about "
            "which SHAPE fits and which INPUT / PROCESSING / OUTPUT specializations "
            "to use, then issue tool calls to construct the DAG. Each reply is "
            "EXACTLY ONE tool call as STRICT JSON:\n"
            '{"tool": "<tool name>", "args": { ... }}\n'
            "No prose, no code fences — only that JSON object.\n\n"
            + self._tool_catalog_text()
            + "\n\nGUIDANCE:\n"
            "- Call seed_plan FIRST, then add_step for each step, then finalize_plan.\n"
            "- Author the SMALLEST correct set of steps. A modular-parallel plan is "
            "INDEPENDENT sub-task steps (each depends_on=[], run at the same time) "
            "FOLLOWED BY one FINAL step that combines their outputs and delivers.\n"
            "- A gather/search step is a SOURCE: depends_on=[]. Never author a step "
            "that duplicates one already authored.\n"
            "- If a step's action is one an available tool performs (search, fetch, "
            "read/write a file, send email), set 'tool' to that tool's exact name.\n"
            "- Pick spec/specs only when a listed specialization genuinely fits. "
            "When the goal names an output FORMAT (HTML, Markdown, a .html/.md "
            "file), bind the output-style spec for THAT format on the final step — "
            "an HTML request gets the HTML writer, a Markdown request the Markdown "
            "writer; never the other.\n"
            "- DELIVERY: by default present the result in chat, or save it with "
            "file_write. Use send_mail ONLY when the goal EXPLICITLY asks to be "
            "emailed — never email unprompted.\n"
            "- The FINAL combine step's depends_on lists EVERY sub-task id it "
            "combines; after authoring it, call finalize_plan.\n\n"
            "REGISTERED SPECIALIZATIONS (lookup — names + descriptions only):\n"
            + json.dumps(ctx["specializations"], indent=2)
            + "\n\nAVAILABLE TOOLS (names + descriptions only):\n"
            + json.dumps(ctx["tools"], indent=2)
        )

    def _initial_user(self, goal: str) -> str:
        """The first turn: the goal + the decision procedure for building the plan."""
        return (
            f"GOAL: {goal}\n\n"
            "Build the plan now. Decision procedure:\n"
            "1. List the DISTINCT ITEMS the GOAL names — each separate topic, place, "
            "source, file or subject (e.g. 'climate change', 'space', 'AI' are "
            "THREE). Each gather step covers exactly ONE.\n"
            "2. Call seed_plan to open the plan.\n"
            "3. Call add_step for each gather step (one per distinct item; "
            "depends_on=[]; set 'tool' to the gather tool). Then add_step for the "
            "FINAL step that combines and delivers (depends_on every gather id; set "
            "'tool' to the delivery tool the goal asks for — file_write to save a "
            "file; send_mail ONLY if the goal EXPLICITLY asked to be emailed). If "
            "the goal names an output FORMAT (HTML, Markdown, a .html/.md file), "
            "also set the FINAL step's output-style 'spec' to the writer for THAT "
            "format — the HTML writer for HTML, the Markdown writer for Markdown.\n"
            "4. Call finalize_plan.\n"
            "Reply with ONE tool call now (start with seed_plan)."
        )

    def _observation_user(self, obs: Mapping[str, Any]) -> str:
        """Render a builder observation back to the model as the next user turn."""
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
        lines.append(
            "Issue the NEXT tool call: add_step for the next step, set_node_spec to "
            "refine a step's specialization, or finalize_plan if every distinct item "
            "is covered and the plan is complete. Reply with ONE tool call."
        )
        return "\n".join(lines)

    def _finalize_user(self, goal: str, builder: PlanBuilder) -> str:
        """Force the closing step when the model is repeating instead of finalizing.

        The 4.6B model cannot reliably do the 'which items are still uncovered'
        set-difference statelessly, so once it starts re-emitting an already-authored
        step (the duplicate signal) the gather phase is DONE — every distinct item is
        covered. This turn directs it to author ONLY the final combine/deliver step
        (the eda-base3 FSM plays this controller role for the strong model). NOT
        few-shot coercion — it carries no example plan and works for any goal/shape."""
        lines = [f"GOAL: {goal}", "", "STEPS ALREADY AUTHORED:"]
        for s in builder._state_summary():
            dep = ", ".join(s.get("depends_on") or []) or "-"
            lines.append(
                f"  {s['id']}: {s['task']}  [tool={s.get('tool') or '-'} "
                f"depends_on={dep}]"
            )
        lines.extend(
            [
                "",
                "Every distinct item the goal names is now COVERED — the gathering "
                "phase is COMPLETE. Call add_step ONCE for the FINAL step that "
                "combines the gathered results and delivers the goal's outcome. Its "
                "depends_on MUST list EVERY gather step id above it uses; set 'tool' "
                "to the delivery tool the goal asks for (file_write to save a file; "
                "send_mail ONLY if the goal EXPLICITLY asked to be emailed; else "
                "leave it \"\"). If the goal names an output FORMAT (HTML, Markdown, "
                "a .html/.md file), set this step's output-style 'spec' to the "
                "matching writer (the HTML writer for HTML, the Markdown writer for "
                "Markdown). Do NOT author another gather step or repeat any step "
                "above. Reply with ONE add_step tool call for this final step.",
            ]
        )
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

    def _apply_default_research_spec(
        self, authored: list[dict[str, Any]], span: Any
    ) -> int:
        """Stamp the default research spec onto every null-spec GATHER node (F2).

        A spec-less gather node has no ruleset, so it degrades to the d13 source-list-
        summary anti-pattern (the empty/thin emailed news section). This finalization
        pass binds :attr:`default_research_spec` to any node that has NO effective
        spec, did NOT declare ``needs_spec`` (so the missing-specialist hatch is never
        masked), AND is a GATHER node (its ``tool`` is one of :attr:`research_tools`).
        Sibling-consistent; a delivery node is intentionally left unbound (its content
        is upstream-grounded, d11). A no-op when the default is unset or not
        registered. Returns the number of nodes stamped."""
        default = self.default_research_spec
        if not default or default not in set(self.spec_names):
            return 0
        applied = 0
        for record in authored:
            has_spec = bool(record.get("spec")) or bool(record.get("specs"))
            if has_spec:
                continue
            if record.get("needs_spec"):
                continue  # declared a MISSING specialist → leave for the gate
            tool = str(record.get("tool") or "").strip().lower()
            if tool not in self.research_tools:
                continue
            record["spec"] = default
            applied += 1
            try:
                span.set_attribute(
                    f"planner.node.{record['id']}.default_spec", default
                )
            except Exception:  # tracing must never break authoring
                pass
        if applied:
            try:
                span.set_attribute("planner.default_research_spec", default)
                span.set_attribute("planner.default_spec_applied", applied)
            except Exception:
                pass
        return applied

    # ------------------------------------------------------------------ #
    # d28: terminal write/synthesize node MUST depend on the research node(s)
    # ------------------------------------------------------------------ #
    def _is_research(self, record: Mapping[str, Any]) -> bool:
        """Is this node a RESEARCH/GATHER node (its output IS the research)?

        Three signals, any of which is sufficient: it fires a research/gather tool
        (``web_search``/``web_fetch`` — :attr:`research_tools`); its node ROLE is
        ``research``; or it is bound to the :attr:`default_research_spec` (the F2
        pass may have just stamped that onto a null-spec gather node, so this is
        called AFTER F2). These are exactly the nodes whose output a downstream
        write/synthesize node must consume — never a delivery node (file_write /
        send_mail), whose content is grounded in upstream text, not researched."""
        tool = str(record.get("tool") or "").strip().lower()
        if tool in self.research_tools:
            return True
        if str(record.get("role") or "").strip().lower() == "research":
            return True
        specs = {str(s) for s in (record.get("specs") or [])}
        if record.get("spec"):
            specs.add(str(record["spec"]))
        if self.default_research_spec and self.default_research_spec in specs:
            return True
        return False

    @staticmethod
    def _ancestors(start_id: str, by_id: Mapping[str, Mapping[str, Any]]) -> set[str]:
        """Transitive ``depends_on`` closure of ``start_id`` (its upstream nodes)."""
        seen: set[str] = set()
        stack = [str(d) for d in (by_id[start_id].get("depends_on") or [])]
        while stack:
            d = stack.pop()
            if d in seen or d not in by_id:
                continue
            seen.add(d)
            stack.extend(str(x) for x in (by_id[d].get("depends_on") or []))
        return seen

    def _enforce_terminal_research_edge(
        self, authored: list[dict[str, Any]], span: Any
    ) -> list[str]:
        """d28: a TERMINAL write/synthesize node MUST depend on the RESEARCH node(s).

        E4B (and intermittently e2b) authors a FLAT zero-edge DAG — a research node
        and a write node with NO edge between them — so the writer runs DISCONNECTED
        from the research and the report comes back thin (the o4 / d34 emptiness).
        This pass repairs that STRUCTURALLY at finalize: for every TERMINAL node (a
        SINK — nothing depends on it) that is a WRITER (NOT itself a research/gather
        node, per :meth:`_is_research`) and that does NOT already (directly or
        transitively) depend on ANY research node, AUTO-ADD an edge to each research
        node it can reach acyclically — so the writer always sees the research.

        A no-op when there is no research node, the plan is a single step, or every
        terminal writer already sees the research (the healthy DAG the model authored
        stays BYTE-IDENTICAL — a connected combine/synthesize sink transitively
        depends on its gather nodes, so this never fires on it). Adding an edge from a
        SINK to a SOURCE only adds an ordering constraint and a sink has no dependents,
        so it can never introduce a cycle (guarded regardless). Returns repair notes.
        """
        if len(authored) < 2:
            return []
        research_ids = [r["id"] for r in authored if self._is_research(r)]
        if not research_ids:
            return []  # not a research task → nothing to connect a writer to
        by_id = {r["id"]: r for r in authored}
        depended_on = {
            str(d) for r in authored for d in (r.get("depends_on") or [])
        }
        sinks = [r for r in authored if r["id"] not in depended_on]
        notes: list[str] = []
        for w in sinks:
            if self._is_research(w):
                continue  # a research node that is itself terminal needs no upstream
            w_anc = self._ancestors(w["id"], by_id)
            if any(rid in w_anc for rid in research_ids):
                continue  # already sees the research (directly or transitively)
            deps = list(w.get("depends_on") or [])
            added: list[str] = []
            for rid in research_ids:
                if rid == w["id"] or rid in deps:
                    continue
                if w["id"] in self._ancestors(rid, by_id):
                    continue  # would cycle (defensive; a sink has no dependents)
                deps.append(rid)
                added.append(rid)
            if added:
                w["depends_on"] = deps
                notes.append(
                    f"node {w['id']!r}: auto-added research edge(s) {added} "
                    "(disconnected terminal writer → research)"
                )
                try:
                    span.set_attribute(
                        f"planner.node.{w['id']}.research_edge_added",
                        ", ".join(added),
                    )
                except Exception:
                    pass
        if notes:
            try:
                span.set_attribute("planner.research_edge_repairs", len(notes))
                span.set_attribute("planner.dag.connectivity_repaired", True)
            except Exception:
                pass
        return notes

    # ------------------------------------------------------------------ #
    # b5: GUARANTEE the requested OUTPUT FORMAT writer on the terminal node
    # ------------------------------------------------------------------ #
    @staticmethod
    def _requested_output_format(goal: str) -> Optional[tuple[str, str]]:
        """The ``(format, writer-spec)`` the GOAL explicitly asks for, or ``None``.

        Returns a single format ONLY when the goal names EXACTLY one of the known
        output formats (so an ambiguous goal that mentions both, or none, is a
        no-op — never a guess, mirroring the fail-open posture of the other
        signals). The match is on the verbatim goal text (the plan carries it),
        not the planner's paraphrase."""
        import re as _re

        text = str(goal or "")
        hits = [
            (fmt, writer)
            for fmt, writer, pat in _OUTPUT_FORMAT_WRITERS
            if _re.search(pat, text, _re.IGNORECASE)
        ]
        return hits[0] if len(hits) == 1 else None

    def _enforce_output_format_spec(
        self, authored: list[dict[str, Any]], goal: str, span: Any
    ) -> list[str]:
        """Guarantee the GOAL's requested output-format writer on the terminal node.

        The d13/B2 fix as a STRUCTURAL guarantee (the F5 pattern): when the goal
        names a deliverable format (HTML / Markdown / a .html/.md file) the PROMPT
        already tells the planner to bind that format's writer on the final step,
        but E4B intermittently binds the analysis spec (``research-analyst``) on a
        "synthesize" node instead — so the deliverable comes back in the wrong form.
        This pass closes that gap deterministically: for every TERMINAL writer node
        (a SINK that is not itself a research/gather node), it ensures the requested
        format's writer is the PRIMARY output-style spec, COMPOSING it ahead of any
        analysis spec already there (per the selection guidelines: an analysis spec +
        an output-format spec compose on the final node), and REMOVING the OTHER
        format's writer (the two are mutually exclusive). No-op (byte-identical) when
        the goal names no single format, the writer spec is not registered, or the
        terminal writer already leads with the right format spec. Returns repair
        notes.

        s9/c2 (d48) — JUSTIFIED, NOT a flow-forcing flag. This is a structural
        OUTPUT-FORMAT INVARIANT (parallel to the wants_file→file invariant): "an HTML
        request must terminate in the HTML writer." It does NOT pick the route or the
        plan shape (the model's reasoning does that). With c2's faithful selection
        (shape selection off format=schema), the planner usually binds the right writer
        itself, so this is a NO-OP safety net that only fires on an intermittent
        mis-bind — it never overrides a correctly-reasoned binding. Kept (justified)
        rather than removed so a single E4B mis-bind cannot silently ship the wrong
        format; it is the thin deterministic floor under the reasoned selection."""
        want = self._requested_output_format(goal)
        if not want:
            return []
        fmt, writer = want
        if writer not in set(self.spec_names):
            return []  # the project does not have this output-style seed → no-op
        other_writers = {
            w for _f, w, _p in _OUTPUT_FORMAT_WRITERS if w != writer
        }
        depended_on = {
            str(d) for r in authored for d in (r.get("depends_on") or [])
        }
        sinks = [r for r in authored if r["id"] not in depended_on]
        notes: list[str] = []
        for w in sinks:
            # Skip a pure GATHER sink (its job is fetching, not producing the
            # deliverable) — keyed on the TOOL/role, NOT on a spec binding: the
            # whole point of this pass is to fix a synthesize sink that was
            # mis-bound to the research-analyst spec, so ``_is_research`` (which
            # treats that binding as "research") must NOT gate it here.
            # d48: "research" is no longer a node role — a gather node is identified
            # by its TOOL (web_search/web_fetch), which is the reliable signal here.
            tool = str(w.get("tool") or "").strip().lower()
            if tool in self.research_tools:
                continue  # a gather node, not the deliverable writer
            existing = list(w.get("specs") or [])
            if not existing and w.get("spec"):
                existing = [w["spec"]]
            # Drop the wrong-format writer(s); keep analysis/other specs in order.
            kept = [s for s in existing if s not in other_writers]
            changed = kept != existing
            if writer in kept:
                if kept[0] != writer:  # present but not primary → promote it
                    kept = [writer] + [s for s in kept if s != writer]
                    changed = True
            else:
                kept = [writer] + kept  # missing → stamp it as the primary output style
                changed = True
            if changed:
                w["specs"] = kept
                w["spec"] = kept[0]
                w["needs_spec"] = None  # a registered writer is not "missing"
                notes.append(
                    f"node {w['id']!r}: enforced {fmt} output spec {writer!r} "
                    f"(specs now {kept})"
                )
                try:
                    span.set_attribute(
                        f"planner.node.{w['id']}.output_format_spec", writer
                    )
                except Exception:
                    pass
        if notes:
            try:
                span.set_attribute("planner.output_format", fmt)
                span.set_attribute("planner.output_format_repairs", len(notes))
            except Exception:
                pass
        return notes

    # ------------------------------------------------------------------ #
    # d50 point-4: echo an explicitly-named filename into the terminal task
    # ------------------------------------------------------------------ #
    def _echo_literal_filename(
        self, authored: list[dict[str, Any]], goal: str, span: Any
    ) -> list[str]:
        """Echo the GOAL's explicit filename into the TERMINAL writer node's task.

        Point-4 (d50): so ``cats.html`` resolves VERBATIM on ANY route. The
        deliverable path is derived TYPE-AGNOSTICALLY by
        :func:`~agent_runtime.synth_tools.derive_output_path`, which reads BOTH the
        carried overall-goal AND the node's task. The live path carries the goal, but
        echoing the literal name into the terminal task makes the verbatim name
        survive even when only the node task reaches the writer (a replan/standalone
        re-dispatch, or any future route) — the same robustness the output-format /
        research-edge finalization passes give. No-op (byte-identical) when the goal
        names no explicit file, or the terminal task already mentions it. Scoped to
        the deliverable-writer sink(s); a pure gather sink is skipped. Returns notes."""
        name = explicit_filename(goal)
        if not name:
            return []
        depended_on = {
            str(d) for r in authored for d in (r.get("depends_on") or [])
        }
        sinks = [r for r in authored if r["id"] not in depended_on]
        notes: list[str] = []
        for w in sinks:
            tool = str(w.get("tool") or "").strip().lower()
            role = str(w.get("role") or "").strip().lower()
            if tool in self.research_tools or role == "research":
                continue  # a gather sink, not the deliverable writer
            task = str(w.get("task") or "")
            if name.lower() in task.lower():
                continue  # already names the file → leave byte-identical
            w["task"] = (task.rstrip(". ") + f". Write the file as {name}.").strip()
            notes.append(
                f"node {w['id']!r}: echoed literal filename {name!r} into task"
            )
            try:
                span.set_attribute(f"planner.node.{w['id']}.literal_filename", name)
            except Exception:
                pass
        if notes:
            try:
                span.set_attribute("planner.literal_filename", name)
                span.set_attribute("planner.literal_filename_echoed", len(notes))
            except Exception:
                pass
        return notes

    # ------------------------------------------------------------------ #
    # the authoring loop (tool calls → assemble DAG)
    # ------------------------------------------------------------------ #
    async def plan(self, goal: str) -> PlanResult:
        """Author a validated :class:`PlanDAG` for ``goal`` via planner tool calls.

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
            inject_review=self.inject_review,
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
                {"role": "user", "content": self._initial_user(goal)}
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
                convo.append(
                    {"role": "user", "content": self._observation_user(obs)}
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

            # F5 REQUESTED-SPEC PASS (before F2): guarantee a user-NAMED spec is bound
            # on the terminal/delivery node when the model forgot to. Runs FIRST so the
            # F2 default-research pass below sees the now-bound node as already-specced.
            self._apply_requested_specs(builder.nodes, span)
            # F2 DEFAULT-RESEARCH-SPEC PASS: bind the generic research spec to any
            # null-spec gather node so no parallel sibling ships an ungrounded /
            # source-list-summary section (d13). Topology stays exactly as the model
            # authored it — only spec-less gather nodes gain the default ruleset.
            self._apply_default_research_spec(builder.nodes, span)
            # d28 TERMINAL-RESEARCH EDGE PASS (after F2 so default-spec'd gather nodes
            # are detectable): a terminal write/synthesize node MUST depend on the
            # research/gather node(s). E4B authors flat zero-edge DAGs EVERY run,
            # starving the writer → thin output (o4); this auto-adds the missing edge.
            # No-op (byte-identical) on a healthy DAG whose sink already sees research.
            edge_repairs = self._enforce_terminal_research_edge(builder.nodes, span)
            # b5 OUTPUT-FORMAT PASS: when the goal names a deliverable format
            # (HTML/Markdown/.html/.md), GUARANTEE that format's output-style writer
            # on the terminal write node (the prompt is the primary lever; E4B
            # intermittently binds research-analyst on a synthesize node instead, so
            # this stamps the format writer as primary + drops the wrong-format one).
            # No-op when the goal names no single format or the writer is unregistered.
            format_repairs = self._enforce_output_format_spec(builder.nodes, goal, span)
            # d50 POINT-4: echo an explicitly-named filename (cats.html) into the
            # terminal writer's task so it resolves VERBATIM on any route (the shared
            # agentic file loop derives the path from goal+task). No-op when the goal
            # names no file or the terminal task already mentions it.
            filename_echoes = self._echo_literal_filename(builder.nodes, goal, span)

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
                    "research_edges": edge_repairs,
                    "dangling_edges": list(dangling_repairs),
                    "output_format": format_repairs,
                    "literal_filename": filename_echoes,
                },
            )
            self.last_result = result
            return result


__all__ = [
    "IncrementalPlanner",
    "DEFAULT_MAX_NODES",
    "DEFAULT_NODE_NUM_PREDICT",
]
