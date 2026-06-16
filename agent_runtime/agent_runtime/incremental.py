"""Incremental seed-then-fill plan authoring — the literal eda-base3 port (d3).

The one-shot :class:`~agent_runtime.planner.Planner` asks Gemma to emit the WHOLE
DAG in a SINGLE structured-output call. A 4.6B local model cannot reliably do that
for a parallel topology + multi-tool binding (the scenario-2 "parallel needs a
stronger model" finding). eda-base3 — the EDP framework this very orchestration
runs on, and the Round-3 blueprint — does NOT one-shot: it SEEDS a plan (a shape +
goal, zero actions) and then FILLS the actions ONE AT A TIME, each ``add_action``
call a small, reliable decision the planner makes while SEEING the goal, the shape,
and every action authored so far (``fsm/plan_fsm.py`` + the ``create_plan`` /
``add_action`` tools). The readiness FSM then dispatches by ``depends_on``.

This module ports that mechanism for the local-Gemma runtime:

* SEED — the plan is the goal + the selected SHAPE (e.g. ``modular-parallel``),
  carried as context (no nodes yet). The shape is the topology *posture* (parallel
  vs sequential), exactly as eda-base3's ``plan.shape`` is.
* FILL — author nodes ONE AT A TIME. Each step is a SEPARATE native structured
  Gemma call (the proven d1 path: ``think=False`` top-level, a JSON schema with
  ``enum``+``required`` keys, ``temperature=0``, ``num_predict`` sized to hold ONE
  node object). The call sees the goal + shape + every already-authored node and
  emits just THIS node's fields + a ``more`` flag (mirroring the planner's own
  decision to stop calling ``add_action``). ``depends_on`` references only
  already-authored node ids, so the DAG is acyclic BY CONSTRUCTION (the
  ``add_action``-references-prior-actions invariant) — each parallel news node is a
  trivial ``depends_on=[]`` decision; the email node simply lists the three news
  ids. No whole-DAG one-shot, so the parallel-multi-tool "limitation" dissolves.

The result is a validated :class:`~agent_runtime.factory.PlanDAG` wrapped in the
SAME :class:`~agent_runtime.planner.PlanResult` the one-shot planner returns, so it
is a drop-in for the live acyclic path (``chat_app.agentic._run_acyclic``):
missing-specialist detection, the d11 :class:`SchemaToolArgEmitter` tool-arg
grounding, the lifecycle gate and the runtime all consume the DAG unchanged — only
HOW the DAG was authored changed.

Context-scoping (d10) is preserved: the planner reasons over the SAME body-free
factory context (factory description + the specialization LOOKUP index + the lean
tool catalog) — never a compiled spec body. The per-node payload is asserted
body-free before each call exactly as the one-shot planner asserts it.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from llm_framework import Chain, Context, Transport
from llm_framework.stages import call_stage, prompt_assembly, structured_output

from .factory import AbstractPlanFactory, PlanDAG, PlanError
from .planner import PlanResult
from .selfheal import MalformedOutputError
from .tracing import get_tracer, run_blocking_in_span

# Hard bound on the authoring loop — the planner's "stop calling add_action" is a
# ``more`` boolean, but a runaway model that never sets ``more=false`` must still
# terminate. A POC acyclic plan is a handful of steps; 12 is generous headroom over
# any realistic modular-parallel fan-out while capping the GPU cost of authoring.
DEFAULT_MAX_NODES = 12

# num_predict for ONE node object. A node is small (a task sentence + a few
# enum-constrained fields + a short depends_on array + the boolean), so this holds
# the whole object at temperature 0 without the small model running its output past
# the cap and truncating the JSON (the specialist-doc structured-output rule).
DEFAULT_NODE_NUM_PREDICT = 512

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


class IncrementalPlanner:
    """Author a plan's DAG node-by-node (seed-then-fill), Gemma deciding per node.

    Parameters
    ----------
    transport:
        Any ``llm_framework`` ``Transport`` (the live ``OllamaTransport`` or an
        offline ``FakeTransport``). Each per-node call goes through it with the d1
        native structured options.
    factory:
        The body-free :class:`AbstractPlanFactory` — the planner's ONLY world view
        (d10). Supplies the factory description + specialization LOOKUP + tool
        catalog for each node's context, and parses the assembled nodes back into a
        validated :class:`PlanDAG` (so the same validation as the one-shot path).
    spec_names / tool_names:
        The registered specialization names and the offered tool names. They become
        the per-node schema ``enum``s (+ ``""`` for none) so Gemma cannot invent a
        spec/tool or cross the two slots — identical vocabulary constraint to the
        one-shot :func:`chat_app.agentic.build_plan_schema`.
    shape_name / shape_description:
        The SELECTED shape (the seed's topology posture). Threaded into every node's
        context so the model authors edges that fit the shape (parallel vs chained).
    max_nodes:
        Hard cap on the authoring loop (:data:`DEFAULT_MAX_NODES`).
    node_num_predict:
        Output-token budget per node call (:data:`DEFAULT_NODE_NUM_PREDICT`).
    call_opts:
        Extra transport options merged OVER the proven native structured defaults
        (``api=native``, ``think=False``, ``temperature=0``, ``num_predict``).
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
    ) -> None:
        self.transport = transport
        self.factory = factory
        self.spec_names = [str(s) for s in spec_names if str(s).strip()]
        self.tool_names = [str(t) for t in tool_names if str(t).strip()]
        self.shape_name = str(shape_name or "")
        self.shape_description = str(shape_description or "")
        # F2 DEFAULT RESEARCH SPEC: the generic, role-appropriate research
        # specialization stamped onto any null-spec GATHER node (see
        # :data:`DEFAULT_RESEARCH_TOOLS`). The NAME is supplied by the caller (the
        # live route passes ``specialization.seed.DEEP_RESEARCH_SPEC`` —
        # ``research-analyst`` — the SAME spec the deep-research shape reuses), so
        # this generic authorer hard-codes no spec name. Applied only when the name
        # is an actually-registered specialization (else a no-op).
        self.default_research_spec = str(default_research_spec or "").strip()
        # F5 USER-REQUESTED SPECIALIZATIONS: the spec name(s) the user EXPLICITLY
        # named (extracted by the model-driven ShapeSelector, enum-constrained to
        # registered names). The authorer is TOLD about them (so the per-node calls
        # bind them on the right node), and a finalization pass GUARANTEES they are
        # honored — if the small model forgot to bind any of them, they are stamped
        # onto the plan's terminal/delivery node(s). Kept only when actually a
        # registered specialization (so an unknown name is a no-op, never poisons a
        # node). Empty => no named-spec handling, identical to the pre-F5 path.
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
        # The PROVEN d1 native structured path (mirrors the heal / ambiguity /
        # shape-selection calls): api=native so the dict ``format`` schema is
        # honoured AND the top-level ``think=False`` reaches /api/chat (gemma4 is a
        # thinking model — without it the CoT trace eats num_predict and the JSON
        # content comes back EMPTY). temp 0 for deterministic authoring.
        self.call_opts = {
            "api": "native",
            "think": False,
            "temperature": 0,
            "num_predict": self.node_num_predict,
            **(call_opts or {}),
        }
        # Captured each plan() call for the context-scoping proof (parity with
        # Planner.last_context) and the authored-node introspection a2 reads.
        self.last_context: Optional[dict[str, Any]] = None
        self.last_result: Optional[PlanResult] = None
        self.last_nodes: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # per-node OUTPUT SCHEMA (one node, not the whole DAG)
    # ------------------------------------------------------------------ #
    def _node_schema(self) -> dict[str, Any]:
        """The native ``format`` schema for ONE node (enum-constrained spec/tool).

        ``id`` is NOT model-authored — the authorer assigns canonical ``n1..nk``
        ids and SHOWS them to the model, so ``depends_on`` references are always
        resolvable and id collisions are impossible. The model emits the node's
        content (``task``), its bindings (``spec``/``specs``/``tool``), the
        free-text missing-specialist signal (``needs_spec``), its ``depends_on`` (a
        subset of the shown prior ids), and ``more`` (are further steps needed)."""
        spec_enum = [""] + list(self.spec_names)
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "spec": {"type": "string", "enum": spec_enum},
                "specs": {
                    "type": "array",
                    "items": {"type": "string", "enum": spec_enum},
                },
                "needs_spec": {"type": "string"},
                "tool": {"type": "string", "enum": [""] + list(self.tool_names)},
                "depends_on": {"type": "array", "items": {"type": "string"}},
                "more": {"type": "boolean"},
            },
            "required": ["task", "depends_on", "more"],
        }

    # ------------------------------------------------------------------ #
    # prompts (seed context + per-node fill instruction)
    # ------------------------------------------------------------------ #
    def _system(self, goal: str) -> str:
        """The body-free SEED context shared by every node call (asserted d10)."""
        ctx = self.factory.planner_context(goal)
        self.factory.assert_body_free(ctx)
        shape_line = ""
        if self.shape_name:
            shape_line = (
                f"\n\nPLAN SHAPE: {self.shape_name}. {self.shape_description}\n"
                "Author the nodes and their 'depends_on' edges so they fit this "
                "shape: INDEPENDENT steps that can run at the same time share NO "
                "edge (depends_on []), while a step that needs an earlier step's "
                "result lists that step's id in depends_on."
            )
        # F5: the user EXPLICITLY named these specialization(s) — a HARD instruction
        # to bind them, not few-shot coercion of any specific plan. An output-style
        # spec belongs on the node that produces the FINAL deliverable; a research
        # spec on the gather node(s). (A finalization pass stamps any the model still
        # leaves unbound onto the terminal node, so this is the primary mechanism +
        # a safety net.)
        requested_line = ""
        if self.requested_specs:
            names = ", ".join(self.requested_specs)
            requested_line = (
                f"\n\nUSER-REQUESTED SPECIALIZATIONS: the user explicitly asked to "
                f"use [{names}]. You MUST set 'spec' (or 'specs') to this/these "
                "specialization(s) on the node(s) they apply to — for an "
                "output-style specialization, the node that produces the final "
                "deliverable; do not drop it in favour of another."
            )
        return (
            ctx["factory"]["description"]
            + shape_line
            + requested_line
            + "\n\nYou author the plan ONE STEP AT A TIME. For the CURRENT step, "
            "emit STRICT JSON for that single node with keys: "
            '{"task": <the logical step, free text>, '
            '"spec": <one specialization name from the lookup, or "">, '
            '"specs": <list of specialization names to COMPOSE, or []>, '
            '"needs_spec": <free-text description of a REQUIRED specialist when NO '
            'listed specialization fits, else "">, '
            '"tool": <one tool name from the tool list, or "">, '
            '"depends_on": <list of ALREADY-AUTHORED step ids this step runs AFTER, '
            "or []>, "
            '"more": <true if MORE steps are still needed to fully accomplish the '
            "goal, false if this step COMPLETES the plan>}.\n\n"
            # PER-NODE GUIDANCE tuned live against gemma4-e2b-agent in s8/a2 (the
            # parallel-authoring gate). These are GENERIC modular-parallel + tool-
            # binding principles (independent sub-tasks have depends_on=[]; a
            # gather/search step is a SOURCE; the final step combines+delivers; bind
            # the tool when the step performs a tool action) — NOT few-shot coercion
            # of any specific plan (d3). Without them the 4.6B model deterministically
            # chained the parallel sub-tasks to the first node and under-bound tools.
            "Author the SMALLEST correct set of steps. A modular-parallel plan is a "
            "set of INDEPENDENT sub-task steps (each depends_on=[], so they run AT "
            "THE SAME TIME) FOLLOWED BY one FINAL step that COMBINES their outputs "
            "and delivers the result. So:\n"
            "- An independent sub-task step (e.g. researching one of several topics "
            "or sources) MUST have depends_on=[].\n"
            "- The FINAL step combines/delivers (e.g. assembling the results and "
            "emailing or saving them); its depends_on lists EVERY sub-task id it "
            "combines, and it sets more=false.\n"
            "- NEVER author a step that DUPLICATES one already authored. Each step "
            "must be distinct.\n"
            "- If the step's action is one an AVAILABLE TOOL performs (searching the "
            "web, fetching a page, reading or writing a file, sending email), you "
            "MUST set 'tool' to that tool's exact name — never describe a tool "
            "action while leaving 'tool' empty.\n"
            "- Pick a 'spec' (or 'specs') only when a listed specialization "
            "genuinely fits. Set 'more' to false on the final delivering step (and "
            "the moment the goal is fully covered).\n\n"
            "REGISTERED SPECIALIZATIONS (lookup — names + descriptions only):\n"
            + json.dumps(ctx["specializations"], indent=2)
            + "\n\nAVAILABLE TOOLS (names + descriptions only):\n"
            + json.dumps(ctx["tools"], indent=2)
        )

    def _user(self, goal: str, authored: list[dict[str, Any]], index: int) -> str:
        """The per-node turn: goal + every node so far + which step to author now."""
        lines = [f"GOAL: {goal}", ""]
        if authored:
            lines.append(
                "STEPS ALREADY AUTHORED (reference only — a new sub-task is usually "
                "INDEPENDENT of these and takes depends_on=[]; depend on an id ONLY "
                "to consume its output, e.g. a final combine step):"
            )
            for n in authored:
                dep = ", ".join(n["depends_on"]) if n["depends_on"] else "-"
                tool = n.get("tool") or "-"
                bound = n.get("spec") or (", ".join(n.get("specs") or []) or "-")
                lines.append(
                    f"  {n['id']}: {n['task']}  "
                    f"[tool={tool} spec={bound} depends_on={dep}]"
                )
        else:
            lines.append("No steps authored yet — this is the FIRST step.")
        lines.extend(
            [
                "",
                f"Now author STEP #{index + 1} (its id will be 'n{index + 1}'). "
                "Follow this DECISION PROCEDURE exactly:",
                "1. Read the GOAL and list the DISTINCT ITEMS it asks you to work "
                "on — each separate topic, place, source, file or subject it names "
                "(e.g. 'climate change', 'space exploration' and 'AI' are THREE "
                "distinct items; 'London', 'Tokyo', 'New York' are THREE). Each "
                "independent gather step covers exactly ONE such item.",
                "2. Compare that list to the steps ALREADY AUTHORED above — each one "
                "already covers ONE item. Work out which named items are NOT yet "
                "covered.",
                "3. IF at least one named item is still uncovered: author the "
                "gather/search step for the NEXT UNCOVERED item now. Its task MUST "
                "name a DIFFERENT item than every step above — NEVER repeat an item "
                "already covered, and never emit a step identical to one above. Set "
                "depends_on=[] (a step that searches, fetches or gathers is a SOURCE "
                "— it needs no input, so its depends_on is ALWAYS []; even when it "
                "resembles a step above it is INDEPENDENT, so it NEVER depends on "
                "another gather step). Set 'tool' to the exact tool that performs it "
                "(EVERY gather step, not just the first). Set more=true.",
                "4. IF every distinct item the goal names is ALREADY covered by a "
                "step above: do NOT author another gather step. Author the FINAL "
                "step that combines and delivers the result — its depends_on lists "
                "EVERY gather step id it merges; set 'tool' to the delivery tool the "
                "goal asks for (send_mail to email, file_write to save a file); set "
                "more=false.",
                "Return ONLY the JSON for this one step.",
            ]
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # one structured per-node call
    # ------------------------------------------------------------------ #
    async def _author_one(
        self, system: str, user: str, opts: Mapping[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Run ONE native structured Gemma call → the parsed node dict (or None).

        Uses the SAME assemble→call→parse+repair chain the one-shot planner uses
        (so the bounded malformed-JSON self-heal still applies), offloaded off the
        event loop via :func:`run_blocking_in_span` (the d4 never-freeze fix) with
        the otel context re-attached so each per-node phi span nests under the
        authoring span instead of detaching into a root trace."""
        chain = Chain()
        chain.use(prompt_assembly())
        chain.use(call_stage(self.transport, **dict(opts)))
        chain.use(
            structured_output(self.transport, max_repair_attempts=self.max_repair_attempts)
        )
        ctx = Context(system=system, user=user, transport=self.transport)
        ctx = await run_blocking_in_span(chain.run, ctx)
        parsed = ctx.structured
        return dict(parsed) if isinstance(parsed, Mapping) else None

    @staticmethod
    def _normalize_task(task: str) -> str:
        """Canonical form of a node task for EXACT-duplicate detection.

        Lower-cased, punctuation-stripped, whitespace-collapsed — so two steps the
        small model emitted with the SAME intent (e.g. 'Search for the latest news
        on climate change' authored twice) compare equal, while genuinely distinct
        items ('climate change' vs 'space exploration') do not. Deliberately STRICT
        (equality of the whole normalised string, not token overlap) so a real
        distinct-but-similar sub-task is NEVER dropped — only an actual repeat is."""
        import re as _re

        return _re.sub(r"[^a-z0-9]+", " ", str(task or "").lower()).strip()

    def _finalize_user(self, goal: str, authored: list[dict[str, Any]]) -> str:
        """Per-node turn that FORCES the closing step when the model is repeating.

        The 4.6B model cannot reliably do the 'which items are still uncovered'
        set-difference statelessly, so once it starts re-emitting an
        already-authored step (the exact-duplicate signal) the gather phase is
        DONE — every distinct item the goal names is covered. This turn directs it
        to author ONLY the final combine/deliver step (the eda-base3 FSM plays this
        controller role for the strong model; the local port supplies it here). NOT
        few-shot coercion of a specific plan — it carries no example plan and works
        for any goal/shape."""
        lines = [f"GOAL: {goal}", "", "STEPS ALREADY AUTHORED:"]
        for n in authored:
            dep = ", ".join(n["depends_on"]) if n["depends_on"] else "-"
            tool = n.get("tool") or "-"
            lines.append(f"  {n['id']}: {n['task']}  [tool={tool} depends_on={dep}]")
        lines.extend(
            [
                "",
                "Every distinct item the goal names is now COVERED by the steps "
                "above — the gathering phase is COMPLETE. Author ONLY the FINAL "
                "step that combines the gathered results and delivers the goal's "
                "outcome. Its depends_on MUST list EVERY gather step id above that "
                "it uses; set 'tool' to the delivery tool the goal asks for "
                "(send_mail to email, file_write to save to a file, else leave it "
                "\"\"); set more=false. Do NOT author another gather/search step and "
                "do NOT repeat any step above. Return ONLY the JSON for this one "
                "final step.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _clean_deps(raw: Any, valid_ids: set[str]) -> list[str]:
        """Keep only depends_on refs that point to an ALREADY-authored node.

        Dropping unknown / self / forward refs is what makes the DAG acyclic and
        resolvable BY CONSTRUCTION (the eda-base3 invariant that an action can only
        depend on actions authored before it). Order-preserving + de-duplicated."""
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            return []
        out: list[str] = []
        for d in raw:
            s = str(d).strip()
            if s and s in valid_ids and s not in out:
                out.append(s)
        return out

    def _apply_requested_specs(
        self, authored: list[dict[str, Any]], span: Any
    ) -> int:
        """GUARANTEE a user-named specialization is bound somewhere (F5).

        The per-node authoring call is TOLD (in :meth:`_system`) that the user
        explicitly requested certain specialization(s), but the 4.6B model can
        forget to bind one — exactly the F5(i) failure (a request naming
        ``markdown-writer`` ran with ``research-analyst`` on every node and the
        named spec NEVER bound). This finalization pass closes that gap
        STRUCTURALLY: if the user named spec(s) and NONE of them appears bound on any
        authored node, they are stamped onto the plan's TERMINAL node(s) — the
        node(s) nothing else depends on, i.e. the final deliverable an output-style
        specialization governs. Generic + structural — no per-scenario topic/spec is
        referenced; it keys only off 'is this requested spec bound anywhere?' and the
        DAG's sink set. A no-op when the user named none, or when the model already
        bound at least one requested spec (its own binding is honored). Returns the
        number of nodes stamped."""
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
        # The TERMINAL nodes: those no other authored node depends on (the sinks /
        # final deliverable). Stamp the requested spec(s) there.
        depended_on: set[str] = set()
        for record in authored:
            for dep in record.get("depends_on") or []:
                depended_on.add(str(dep))
        sinks = [r for r in authored if r["id"] not in depended_on] or authored[-1:]
        applied = 0
        for record in sinks:
            # COMPOSE onto any spec the node already carries (don't clobber), via the
            # N-spec ``specs`` slot the runtime layers (SubAgent._compose_ruleset_stack).
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

        The per-node authoring call decides each node's ``spec`` independently, so
        identical sibling gather sub-tasks diverge: one news node gets
        ``research-analyst`` and its siblings get NOTHING (the live a1 trace: 2 of 3
        parallel news nodes bound to no spec). A spec-less gather node has no
        ruleset, so it degrades to the d13 source-list-summary anti-pattern (the
        empty/thin emailed news section). This finalization pass closes that gap
        STRUCTURALLY: any node that

          * has NO effective spec (no ``spec`` and no ``specs``), AND
          * did NOT declare ``needs_spec`` (so the missing-specialist hatch — a node
            asking for an UNavailable specialist — is never masked by the default;
            that node still pauses the run for the user choice), AND
          * is a GATHER node (its ``tool`` is one of :attr:`research_tools`)

        is bound to :attr:`default_research_spec`. The result is sibling-consistent:
        every parallel gather node carries the same grounded research ruleset, so no
        sibling produces a thin/ungrounded section. Generic + role-structural — no
        per-scenario topic/spec/filename is referenced; a delivery node
        (send_mail/file_write) is intentionally left unbound (its content is
        upstream-grounded, d11). A no-op when the default is unset or not a
        registered specialization. Returns the number of nodes stamped."""
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
    # the authoring loop (seed → fill one node at a time → assemble DAG)
    # ------------------------------------------------------------------ #
    async def plan(self, goal: str) -> PlanResult:
        """Author a validated :class:`PlanDAG` for ``goal`` node-by-node.

        Loops up to ``max_nodes``: each iteration authors ONE node (a small native
        structured decision over goal + shape + nodes-so-far), assigns it the
        canonical id ``n{i+1}``, clamps its ``depends_on`` to already-authored ids,
        and stops when the model sets ``more=false`` (its own "plan complete"
        signal) or the cap is hit. The assembled nodes are parsed through the SAME
        :meth:`AbstractPlanFactory.parse_dag` the one-shot path uses, so the DAG is
        validated (unique ids, resolvable refs, acyclic) identically. Raises
        :class:`MalformedOutputError` if NO usable node was authored, so the outer
        self-heal can re-plan exactly as it does for a malformed one-shot plan."""
        system = self._system(goal)
        self.last_context = self.factory.planner_context(goal)
        opts = {**self.call_opts, "format": self._node_schema()}

        authored: list[dict[str, Any]] = []
        raw_nodes: list[dict[str, Any]] = []
        seen_tasks: dict[str, str] = {}  # normalised task -> node id (dedup)

        def _accept(node: Mapping[str, Any]) -> Optional[dict[str, Any]]:
            """Append a valid, NON-duplicate node; return it (or None if rejected)."""
            task = str(node.get("task") or "").strip()
            if not task:
                return None
            norm = self._normalize_task(task)
            if norm in seen_tasks:
                return None  # exact repeat of an already-authored step → reject
            nid = f"n{len(authored) + 1}"
            record = {
                "id": nid,
                "task": task,
                "spec": (str(node["spec"]) if node.get("spec") else None),
                "specs": [str(s) for s in (node.get("specs") or []) if str(s).strip()],
                "tool": (str(node["tool"]) if node.get("tool") else None),
                "needs_spec": (
                    str(node["needs_spec"]) if node.get("needs_spec") else None
                ),
                "depends_on": self._clean_deps(
                    node.get("depends_on"), {n["id"] for n in authored}
                ),
            }
            authored.append(record)
            seen_tasks[norm] = nid
            raw_nodes.append(dict(node))
            return record

        tracer = get_tracer("agent_runtime.incremental")
        with tracer.start_as_current_span("planner.incremental") as span:
            span.set_attribute("planner.goal", str(goal)[:1000])
            span.set_attribute("planner.shape", self.shape_name or "")
            span.set_attribute("planner.max_nodes", self.max_nodes)
            terminated = False     # the model authored its own final step (more=false)
            needs_finalize = False  # the model started repeating → force a closing step
            for i in range(self.max_nodes):
                # Index by nodes authored so far (NOT the loop counter) so the
                # "id will be n{k}" the prompt announces always matches the id
                # _accept() assigns, even after a skipped (empty) emission.
                user = self._user(goal, authored, len(authored))
                node = await self._author_one(system, user, opts)
                if node is None:
                    # Malformed/empty node: stop the loop. If earlier nodes were
                    # authored we still ship them (a partial-but-valid plan); if
                    # none were, the post-loop guard raises for the self-heal.
                    span.set_attribute(f"planner.node.{i + 1}.malformed", True)
                    break
                more = bool(node.get("more"))
                accepted = _accept(node)
                if accepted is None:
                    # An empty or DUPLICATE node. A duplicate is the model's "I have
                    # no new distinct item" signal — the small model cannot do the
                    # uncovered-items set-difference itself, so once it repeats, the
                    # gather phase is done. Stop authoring gathers; if the plan has
                    # NOT already terminated, force one closing/deliver step below.
                    if str(node.get("task") or "").strip():
                        # A duplicate's ``more`` flag is unreliable (the model often
                        # repeats a step while still asserting more=true). The repeat
                        # itself is the signal that no new distinct item remains, so
                        # always force the closing step unless the plan already
                        # terminated with its own final node.
                        needs_finalize = True
                        span.set_attribute(f"planner.node.{i + 1}.duplicate", True)
                        break
                    # A genuinely empty task: honour ``more`` so the model can still
                    # end the plan cleanly.
                    if not more:
                        break
                    continue
                if not more:
                    terminated = True
                    break

            # FORCED FINALIZE (deterministic loop control, d3): the model wanted more
            # steps but could only repeat, so it never authored the closing step. Make
            # ONE explicit finalize call that authors only the combine/deliver node.
            if needs_finalize and not terminated and authored:
                node = await self._author_one(
                    system, self._finalize_user(goal, authored), opts
                )
                if node is not None and _accept(node) is not None:
                    span.set_attribute("planner.forced_finalize", True)

            span.set_attribute("planner.node_count", len(authored))

            # F5 REQUESTED-SPEC PASS (before F2): guarantee a user-NAMED spec is
            # bound on the terminal/delivery node when the model forgot to. Runs
            # FIRST so the F2 default-research pass below sees the now-bound node as
            # already-specced and leaves it alone (no clobber).
            self._apply_requested_specs(authored, span)

            # F2 DEFAULT-RESEARCH-SPEC PASS: bind the generic research spec to any
            # null-spec gather node so no parallel sibling ships an ungrounded /
            # source-list-summary section (d13). Runs over the FINAL node set so the
            # bound spec is part of the authored DAG (visible in the trace + consumed
            # by the runtime unchanged). Topology stays exactly as the model authored
            # it — only spec-less gather nodes gain the default ruleset.
            self._apply_default_research_spec(authored, span)

            if not authored:
                raise MalformedOutputError(
                    "incremental authorer produced no usable nodes "
                    f"(shape={self.shape_name!r}, max_nodes={self.max_nodes})"
                )

            structured = {
                "rationale": (
                    f"incremental seed-then-fill authoring "
                    f"({self.shape_name or 'acyclic'}, {len(authored)} nodes)"
                ),
                "nodes": [
                    {
                        "id": n["id"],
                        "task": n["task"],
                        "spec": n["spec"],
                        "specs": n["specs"],
                        "tool": n["tool"],
                        "needs_spec": n["needs_spec"],
                        "depends_on": n["depends_on"],
                    }
                    for n in authored
                ],
                "shape": self.shape_name,
            }
            try:
                dag = self.factory.parse_dag(structured)
            except PlanError as exc:
                # Should not happen (deps are clamped backward → acyclic), but if a
                # node was still structurally invalid, surface it as a malformed
                # plan for the self-heal — never ship an invalid DAG to the runtime.
                raise MalformedOutputError(
                    f"incremental authorer assembled an invalid DAG: {exc}"
                ) from exc

            self.last_nodes = authored
            result = PlanResult(
                dag=dag,
                context=self.last_context,
                raw=json.dumps(raw_nodes),
                structured=structured,
                repair={},
            )
            self.last_result = result
            return result


__all__ = [
    "IncrementalPlanner",
    "DEFAULT_MAX_NODES",
    "DEFAULT_NODE_NUM_PREDICT",
]
