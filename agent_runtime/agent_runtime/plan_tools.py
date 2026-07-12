"""In-app plan-building TOOLS the planner LLM calls (s8/b1, d39 Role-1).

The seed-then-fill :class:`~agent_runtime.incremental.IncrementalPlanner` used to
author each node with a native ``format``-schema-constrained call. That is the
**d34 edge-drop** failure mode: constrained decoding trades CONTENT fidelity for
syntactic validity, so the small model's correctly-reasoned ``depends_on`` edge is
silently dropped (measured 0/3 connected WITH schema vs 3/3 WITHOUT; the edge is
present in ``message.thinking`` every time) — the writer node runs disconnected and
the report comes back thin. The specialist ruleset is explicit: load-bearing
reasoned fields (plan dependencies, spec/tool selection) must be built via DISCRETE
TOOL CALLS or prompt-elicited JSON + a validate-and-repair pass, with ``think=true`` —
never behind ``format``-schema.

This module is the eda-base3 ``create_plan`` + ``add_action`` pattern ported to the
local-Gemma runtime: the planner BUILDS the DAG by issuing one tool call per turn —

* ``seed_plan(shape, goal, rationale)`` — open the plan (records the topology
  posture; no nodes yet), exactly as eda-base3 ``create_plan`` seeds a shape+goal.
* ``add_step(task, depends_on, role, spec/specs/needs_spec, tool, id)`` — append ONE
  node; the builder assigns the canonical id ``n{k}`` and clamps ``depends_on`` to
  already-authored ids so the DAG is acyclic-by-construction (the eda-base3
  ``add_action``-references-prior-actions invariant). One reliable decision per call.
* ``set_node_spec(id, spec/specs/needs_spec)`` — refine the specialization on an
  already-authored node (select the INPUT / PROCESSING / OUTPUT specialist after the
  topology is laid down), without re-authoring the step.
* ``finalize_plan(rationale)`` — the planner's own "plan complete" signal (mirrors
  the model deciding to stop calling ``add_action``).

The builder is pure data + validation — NO model call (the loop in
:mod:`agent_runtime.incremental` owns the transport). It carries NO ``format`` schema
and never constrains the model's reasoned fields: each tool call arrives as
prompt-elicited JSON the loop parses + validates, so a ``depends_on`` edge can never
be schema-dropped. The accumulated nodes use the SAME record shape the incremental
authorer always produced, so the F2/F5 finalization passes and the
:class:`~agent_runtime.factory.AbstractPlanFactory.parse_dag` validation consume them
unchanged — only HOW the DAG is authored changed.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

# ROLE NORMALISATION (d213/d215): the in-plan node-role vocabulary is
# {researcher, worker, reviewer}; ``synthesizer`` is the framework-built terminal node
# (the planner should not author it) and ``planner`` is the stage. A small model still
# occasionally emits a RETIRED/legacy or stage role from stale prompting; map it to a
# valid in-plan node role HERE at the authoring boundary so a role-slip can never crash
# the DAG factory (which rejects an unknown role). The first-class in-plan roles pass
# through; the deep-research POSITION words map to the closest in-plan role
# (research -> researcher; critic/verify -> worker); synthesis/synthesize -> worker (a
# write worker — the terminal SYNTHESIZER stage is framework-built, never add_step'd);
# planner -> worker; any other unknown value -> worker (the safe fallback the
# shape-unroll already uses for an unknown position).
_LEGACY_ROLE_MAP: dict[str, str] = {
    "researcher": "researcher",
    "worker": "worker",
    "reviewer": "reviewer",
    "synthesizer": "synthesizer",
    "research": "researcher",
    "critic": "worker",
    "verify": "worker",
    "synthesis": "worker",
    "synthesize": "worker",
    "planner": "worker",
}


def _normalize_role(raw: Any) -> Optional[str]:
    """Map a model-emitted role to a valid in-plan node role, or None for a plain step."""
    role = str(raw or "").strip().lower()
    if not role:
        return None
    return _LEGACY_ROLE_MAP.get(role, "worker")


# ---------------------------------------------------------------------------- #
# d285 SB-3 — the STEP-BRIEF research-MEMORY field (memory_index | <<NEW>>).
# ---------------------------------------------------------------------------- #
# The TEXTUAL sentinel a step brief carries to mean "start a FRESH research memory".
# It is the DATA form of SB-1's ``NEW_MEMORY`` object: a brief is data the PLANNER
# authors as TEXT (one tool-call arg), so the sentinel it writes is a string, never a
# Python object. The :func:`~agent_runtime.research_tree.resolve_brief_memory` resolver
# maps this string onto SB-1's ``open_memory(NEW_MEMORY)``. Defined HERE (a dependency-
# free leaf module) so both the plan-step brief (this module) and the research brief
# (research_tree.to_brief) share one contract without an import cycle.
NEW_MEMORY_SENTINEL = "<<NEW>>"


def normalize_brief_memory_index(value: Any) -> str:
    """Canonicalize a step-brief ``memory_index`` field (d285 SB-3).

    A brief carries EITHER an existing research-memory INDEX (→ CONTINUE that memory)
    OR the textual NEW sentinel ``<<NEW>>`` (→ START a fresh, distinct memory). An
    empty / whitespace / ``None`` value means the planner chose nothing → treated as
    ``<<NEW>>`` (a fresh memory), never a silent blank handle. A real index is returned
    verbatim-stripped (SB-1's store applies its own filesystem-safe normalization when
    the index is actually opened). Pure-string + idempotent: the engine STAMPS no index
    here — it only canonicalizes the planner-authored choice (anti-fabrication, d285)."""
    s = "" if value is None else str(value).strip()
    if not s or s == NEW_MEMORY_SENTINEL:
        return NEW_MEMORY_SENTINEL
    return s

# The four plan-building tools, advertised to the planner in its system prompt.
# Data (not code) so the prompt and the dispatcher agree on exactly one surface.
# ``args`` lists each tool's accepted argument keys with a one-line meaning; the
# planner emits ONE call per turn as ``{"tool": <name>, "args": {...}}``.
PLAN_TOOLS_SPEC: tuple[dict[str, Any], ...] = (
    {
        "name": "seed_plan",
        "args": {
            "shape": "the chosen plan shape/topology posture (string, optional — "
                     "defaults to the pre-selected shape)",
            "rationale": "one line: why this shape fits the goal (optional)",
        },
        "description": "Open the plan once, before any steps. Records the topology "
                       "posture. Call this FIRST.",
    },
    {
        "name": "add_step",
        "args": {
            "task": "the logical step, free text (REQUIRED)",
            "depends_on": "list of ALREADY-AUTHORED step ids this runs AFTER (e.g. "
                          "[\"n1\",\"n2\"]); [] for an independent/source step",
            "tool": "one tool name from AVAILABLE TOOLS, or \"\"",
            "spec": "one specialization name from REGISTERED SPECIALIZATIONS, or \"\"",
            "specs": "list of specialization names to COMPOSE on this step, or []",
            "needs_spec": "free text describing a REQUIRED specialist when none "
                          "listed fits, else \"\"",
            "role": "in-plan node role: researcher (a gather step), worker (a normal "
                    "step, e.g. authors a section), or reviewer (make the LAST step a "
                    "reviewer that fixes the deliverable + emits the final status); "
                    "or \"\" for a plain step. Do NOT use planner/synthesizer (stages)",
            "source_ids": "list of [S#] SOURCE NUMBERS from the SOURCE INDEX / "
                          "AVAILABLE SOURCES whose facts/figures/URLs THIS step uses, "
                          "e.g. [1,4]. When a SOURCE INDEX / source list is provided, any "
                          "step that WRITES or PRESENTS content MUST set a NON-EMPTY list "
                          "(assign every [S#] the section draws on). Use [] ONLY for a "
                          "gather/search step or when no source list is provided",
            "memory_index": "the RESEARCH MEMORY this step works in (d285): pass the "
                            "INDEX a PRIOR step built (from the upstream summary + index "
                            "you received) to CONTINUE that research, or \"<<NEW>>\" / "
                            "leave empty to START a FRESH memory (a distinct new research "
                            "line). Reason over the upstream summary+index: continue when "
                            "this step extends that research, <<NEW>> when it begins a new "
                            "line. A gather/research step that opens a new line uses <<NEW>>",
            "id": "optional id you want to reference later; the builder assigns the "
                  "canonical id and maps yours to it",
        },
        "description": "Append ONE step to the plan. Independent steps take "
                       "depends_on=[]; a step that consumes earlier output lists "
                       "those step ids. Call once per step.",
    },
    {
        "name": "set_node_spec",
        "args": {
            "id": "the step id to refine (REQUIRED)",
            "spec": "one specialization name, or \"\"",
            "specs": "list of specialization names to COMPOSE, or []",
            "needs_spec": "free text for a REQUIRED missing specialist, or \"\"",
        },
        "description": "Refine the specialization on an already-authored step "
                       "(select its INPUT/PROCESSING/OUTPUT specialist) without "
                       "re-authoring it.",
    },
    {
        "name": "finalize_plan",
        "args": {"rationale": "one line summarising the finished plan (optional)"},
        "description": "Signal the plan is COMPLETE. Call this once every needed "
                       "step is authored.",
    },
)

PLAN_TOOL_NAMES: frozenset[str] = frozenset(t["name"] for t in PLAN_TOOLS_SPEC)


def _clean_source_ids(raw: Any) -> list[int]:
    """Keep only positive-int SOURCE numbers (deduped, order-preserving) (s9/c13).

    The model emits ``source_ids`` as a list of 1-based source numbers (or a bare
    number / a stringified int); anything non-int / non-positive is dropped so a
    malformed value can never crash the authoring loop. Empty list = no scoping."""
    if isinstance(raw, (int, str)):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[int] = []
    for s in raw:
        try:
            v = int(s)
        except (TypeError, ValueError):
            continue
        if v > 0 and v not in out:
            out.append(v)
    return out


def _normalize_task(task: str) -> str:
    """Canonical form of a node task for EXACT-duplicate detection.

    Lower-cased, punctuation-stripped, whitespace-collapsed — so two steps the
    small model emitted with the SAME intent ('Search for the latest news on
    climate change' authored twice) compare equal, while genuinely distinct items
    ('climate change' vs 'space exploration') do not. Deliberately STRICT (whole
    normalised string, not token overlap) so a real distinct-but-similar sub-task
    is NEVER dropped — only an actual repeat is. (Lifted verbatim from the prior
    IncrementalPlanner so dedup behaviour is byte-identical across the rework.)"""
    return re.sub(r"[^a-z0-9]+", " ", str(task or "").lower()).strip()


class PlanToolError(ValueError):
    """A tool call was structurally unusable (unknown tool, missing required arg)."""


class PlanBuilder:
    """Accumulates a plan DAG from the planner's tool calls (seed/add/set/finalize).

    Stateful but model-free: :meth:`dispatch` applies ONE parsed tool call and
    returns an OBSERVATION (a small dict the loop renders back to the model). The
    authored nodes use the same record shape the incremental authorer always built
    (``id/task/spec/specs/tool/needs_spec/depends_on``) so the downstream
    finalization passes + DAG validation are unchanged.

    Parameters
    ----------
    spec_names / tool_names:
        Registered specialization names and offered tool names. A ``spec`` /
        ``specs`` / ``tool`` value the model emits that is NOT in these sets is
        DROPPED (never crashes the loop) — the same vocabulary discipline the old
        per-node enum schema enforced, applied as validation instead of constrained
        decoding (so it can't drop the reasoned ``depends_on``).
    shape_name / shape_description:
        The pre-selected shape; ``seed_plan`` may override the name but defaults to
        this.
    max_nodes:
        Hard cap on authored steps; ``add_step`` beyond it is rejected with an
        observation rather than silently growing.
    """

    def __init__(
        self,
        *,
        spec_names: Sequence[str] = (),
        tool_names: Sequence[str] = (),
        shape_name: str = "",
        shape_description: str = "",
        max_nodes: int = 12,
    ) -> None:
        self._spec_names = {str(s) for s in spec_names if str(s).strip()}
        self._tool_names = {str(t) for t in tool_names if str(t).strip()}
        self.shape_name = str(shape_name or "")
        self.shape_description = str(shape_description or "")
        self.max_nodes = max(1, int(max_nodes))
        self.rationale = ""
        self.seeded = False
        self.finalized = False
        # The authored node records (incremental-authorer shape). Public so the
        # planner's F2/F5 finalization passes operate on it directly.
        self.nodes: list[dict[str, Any]] = []
        self._seen_tasks: dict[str, str] = {}  # normalised task -> canonical id
        self._alias: dict[str, str] = {}       # model-supplied id -> canonical id
        # Per-call audit trail (each {tool, ok, note}) for the trace + introspection.
        self.calls: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    @property
    def _known_ids(self) -> set[str]:
        return {n["id"] for n in self.nodes}

    def _resolve_id(self, raw: str) -> Optional[str]:
        """Map a model-referenced id (canonical or an alias it coined) to canonical."""
        s = str(raw).strip()
        if not s:
            return None
        if s in self._known_ids:
            return s
        return self._alias.get(s)

    def _clean_deps(self, raw: Any) -> list[str]:
        """Keep only depends_on refs that resolve to an ALREADY-authored node.

        Dropping unknown / self / forward refs is what makes the DAG acyclic and
        resolvable BY CONSTRUCTION (an action can only depend on actions authored
        before it). Order-preserving + de-duplicated. Aliases the model coined via
        ``add_step(id=...)`` resolve to their canonical id."""
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            return []
        out: list[str] = []
        for d in raw:
            rid = self._resolve_id(d)
            if rid and rid not in out:
                out.append(rid)
        return out

    def _clean_specs(self, raw: Any) -> list[str]:
        """Keep only specialization names that are actually registered."""
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            return []
        out: list[str] = []
        for s in raw:
            name = str(s).strip()
            if name and name in self._spec_names and name not in out:
                out.append(name)
        return out

    def _clean_spec(self, raw: Any) -> Optional[str]:
        name = str(raw or "").strip()
        return name if name and name in self._spec_names else None

    def _clean_tool(self, raw: Any) -> Optional[str]:
        name = str(raw or "").strip()
        return name if name and name in self._tool_names else None

    def _state_summary(self) -> list[dict[str, Any]]:
        """A compact view of the authored steps, for the observation echoed back."""
        out = []
        for n in self.nodes:
            out.append(
                {
                    "id": n["id"],
                    "task": n["task"],
                    "depends_on": list(n["depends_on"]),
                    "tool": n.get("tool") or "",
                    "spec": n.get("spec") or (", ".join(n.get("specs") or []) or ""),
                    "source_ids": n.get("source_ids") or [],
                    # d285 SB-3: echo the chosen memory line back so the planner SEES
                    # which index each step works in and can CONTINUE it on a later step.
                    "memory_index": n.get("memory_index") or NEW_MEMORY_SENTINEL,
                }
            )
        return out

    # ------------------------------------------------------------------ #
    # the four tools
    # ------------------------------------------------------------------ #
    def seed_plan(self, args: Mapping[str, Any]) -> dict[str, Any]:
        shape = str(args.get("shape") or "").strip()
        if shape:
            self.shape_name = shape
        rationale = str(args.get("rationale") or "").strip()
        if rationale:
            self.rationale = rationale
        self.seeded = True
        return {
            "ok": True,
            "note": (
                f"plan opened (shape={self.shape_name or 'acyclic'}). "
                "Author each step with add_step, then call finalize_plan."
            ),
            "steps": self._state_summary(),
        }

    def add_step(self, args: Mapping[str, Any]) -> dict[str, Any]:
        task = str(args.get("task") or "").strip()
        if not task:
            return {"ok": False, "note": "add_step needs a non-empty 'task'.",
                    "steps": self._state_summary()}
        if len(self.nodes) >= self.max_nodes:
            return {
                "ok": False,
                "note": f"step cap reached ({self.max_nodes}); call finalize_plan.",
                "steps": self._state_summary(),
            }
        norm = _normalize_task(task)
        if norm in self._seen_tasks:
            # The model re-emitted an existing step — its "no new distinct item"
            # signal. Reject the duplicate and nudge toward closing the plan.
            return {
                "ok": False,
                "duplicate": True,
                "note": (
                    f"that step already exists as {self._seen_tasks[norm]}. If every "
                    "distinct item the goal names is covered, author the FINAL "
                    "combine/deliver step (depends_on every gather id) or call "
                    "finalize_plan."
                ),
                "steps": self._state_summary(),
            }
        nid = f"n{len(self.nodes) + 1}"
        spec = self._clean_spec(args.get("spec"))
        specs = self._clean_specs(args.get("specs"))
        record: dict[str, Any] = {
            "id": nid,
            "task": task,
            "spec": spec,
            "specs": specs,
            "tool": self._clean_tool(args.get("tool")),
            "needs_spec": (str(args["needs_spec"]).strip() or None)
            if args.get("needs_spec")
            else None,
            "depends_on": self._clean_deps(args.get("depends_on")),
            # d48: coerce a legacy/unknown role to a valid node role (worker|
            # synthesizer) so a model role-slip can never crash the DAG factory.
            "role": _normalize_role(args.get("role")),
            # SOURCE-SCOPING (s9/c13, d56): the global SOURCE number(s) this section
            # uses — the planner's REASONED source→section assignment. Cleaned to a
            # deduped positive-int list; PlanNode re-validates. [] = no scoping.
            "source_ids": _clean_source_ids(args.get("source_ids")),
            # MEMORY-INDEX (d285 SB-3): the planner's REASONED choice of which research
            # memory this step works in — an existing index to CONTINUE, or "<<NEW>>"
            # to start fresh. Canonicalized (empty → <<NEW>>); resolved through SB-1's
            # store at run time. The engine stamps NO index — this is the planner's.
            "memory_index": normalize_brief_memory_index(args.get("memory_index")),
        }
        self.nodes.append(record)
        self._seen_tasks[norm] = nid
        # Register any model-coined id as an alias so later depends_on resolves.
        coined = str(args.get("id") or "").strip()
        if coined and coined != nid:
            self._alias[coined] = nid
        dropped = []
        if args.get("spec") and not spec:
            dropped.append("spec (not a registered specialization)")
        if args.get("tool") and not record["tool"]:
            dropped.append("tool (not an available tool)")
        note = f"added {nid}."
        if dropped:
            note += " dropped: " + "; ".join(dropped) + "."
        return {"ok": True, "id": nid, "note": note, "steps": self._state_summary()}

    def set_node_spec(self, args: Mapping[str, Any]) -> dict[str, Any]:
        rid = self._resolve_id(args.get("id") or "")
        if not rid:
            return {
                "ok": False,
                "note": f"set_node_spec: unknown step id {args.get('id')!r}.",
                "steps": self._state_summary(),
            }
        record = next(n for n in self.nodes if n["id"] == rid)
        spec = self._clean_spec(args.get("spec"))
        specs = self._clean_specs(args.get("specs"))
        if specs:
            record["specs"] = specs
            record["spec"] = specs[0]
        elif spec:
            record["spec"] = spec
            record["specs"] = []
        needs = str(args.get("needs_spec") or "").strip()
        if needs:
            record["needs_spec"] = needs
        bound = record.get("spec") or (", ".join(record.get("specs") or []) or "-")
        return {
            "ok": True,
            "note": f"{rid} specialization set to {bound}.",
            "steps": self._state_summary(),
        }

    def finalize_plan(self, args: Mapping[str, Any]) -> dict[str, Any]:
        rationale = str(args.get("rationale") or "").strip()
        if rationale:
            self.rationale = rationale
        self.finalized = True
        return {
            "ok": True,
            "done": True,
            "note": f"plan finalized with {len(self.nodes)} step(s).",
            "steps": self._state_summary(),
        }

    # ------------------------------------------------------------------ #
    # dispatch + export
    # ------------------------------------------------------------------ #
    def dispatch(self, tool: str, args: Optional[Mapping[str, Any]]) -> dict[str, Any]:
        """Apply ONE parsed tool call; return an observation dict for the loop.

        Never raises on a model mistake — an unknown tool / bad args returns an
        ``ok=False`` observation so the loop can echo it back and let the model
        correct, rather than aborting the whole authoring run."""
        name = str(tool or "").strip()
        kwargs = dict(args) if isinstance(args, Mapping) else {}
        if name not in PLAN_TOOL_NAMES:
            obs = {
                "ok": False,
                "note": (
                    f"unknown tool {name!r}; call one of "
                    f"{sorted(PLAN_TOOL_NAMES)}."
                ),
                "steps": self._state_summary(),
            }
        else:
            obs = getattr(self, name)(kwargs)
        self.calls.append({"tool": name, "ok": bool(obs.get("ok")), "note": obs.get("note", "")})
        return obs

    def to_structured(self) -> dict[str, Any]:
        """The ``{rationale, nodes, shape}`` dict the factory parses into a PlanDAG.

        Same shape the prior incremental authorer assembled, so
        :meth:`AbstractPlanFactory.parse_dag` consumes it unchanged."""
        structured = {
            "rationale": self.rationale
            or (
                f"tool-driven authoring "
                f"({self.shape_name or 'acyclic'}, {len(self.nodes)} nodes)"
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
                    "role": n.get("role"),
                    "source_ids": n.get("source_ids") or [],
                    # d285 SB-3: the planner's chosen research-memory line for this step
                    # (an index to continue, or <<NEW>>) — carried onto the PlanNode.
                    "memory_index": n.get("memory_index") or NEW_MEMORY_SENTINEL,
                }
                for n in self.nodes
            ],
            "shape": self.shape_name,
        }
        return structured


__all__ = [
    "PlanBuilder",
    "PlanToolError",
    "PLAN_TOOLS_SPEC",
    "PLAN_TOOL_NAMES",
    "NEW_MEMORY_SENTINEL",
    "normalize_brief_memory_index",
]
