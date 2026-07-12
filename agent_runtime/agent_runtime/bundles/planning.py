"""bundles.planning — the PlanningBundle (d190/d192/d194).

The CAPABILITY DOMAIN for AUTHORING a plan. The planner reasons about which SHAPE
(DAG topology) fits the goal, then composes the DAG by issuing DISCRETE TOOL CALLS —
``seed_plan`` → ``add_step`` × n → ``finalize_plan`` (the eda-base3
create_plan/add_action pattern) — never a one-shot ``format``-schema JSON DAG (that is
the d34 edge-drop). It DISCOVERS the available shapes + specializations via the
queryable ``get_shapes`` / ``get_specs`` tools, and per node it records a ROLE and a
SPECIALIZATION ('none' is a valid explicit choice, d194).

This bundle ORCHESTRATES the existing surfaces — :data:`~agent_runtime.plan_tools.PLAN_TOOLS_SPEC`
(dispatched by :class:`~agent_runtime.plan_tools.PlanBuilder`) and the discovery tools
:func:`~agent_runtime.discovery_tools.register_discovery_tools` — it reimplements
nothing. The plan-tool catalog text the incremental planner injects
(:meth:`plan_tool_catalog_text`) lives here as the single source of truth; the planner
delegates to it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..plan_tools import PLAN_TOOL_NAMES, PLAN_TOOLS_SPEC
from .base import ObjectBundle

# The planner doctrine (d235 — COMPRESSED for E4B legibility: the prior ~30-line prose was
# too verbose for the small model; this is the same rules tightened to the essentials + 3
# concrete examples). Pick ONE shape, compose the DAG via tool calls, record ROLE + SPEC per node.
_PLANNING_DOCTRINE = (
    "You are the PLANNER. DISCOVER first (get_shapes for the plan SHAPES, get_specs for the "
    "SPECIALIZATIONS), pick the ONE shape that fits, then BUILD the DAG with discrete tool "
    "calls: seed_plan once → add_step per node → finalize_plan.\n\n"
    "DAG: author the SMALLEST correct one. Independent nodes share NO edge (depends_on=[], they "
    "run in parallel); a node that uses an earlier node's output lists that node in depends_on. "
    "A compositional ask means BOTH patterns coexist — author the real mix, not a flat line.\n\n"
    "PER NODE set ROLE + SPECIALIZATION explicitly:\n"
    "- ROLE is one of: researcher (GATHERS evidence — search/fetch/note), worker (DOES a step, "
    "e.g. authors a section), reviewer. The LAST step is ALWAYS a reviewer (it inspects+fixes "
    "the deliverable and emits the plan's FINAL STATUS). NEVER author a 'planner' or "
    "'synthesizer' node — planning is THIS stage; final delivery is a terminal stage after your "
    "plans.\n"
    "- SPECIALIZATION is the behaviour ruleset; the SPEC FOLLOWS WHAT THE NODE DOES. A "
    "document-FORMAT spec (html-writer, markdown-writer) goes ONLY on the worker that writes the "
    "deliverable; a research/analysis spec goes on gather nodes. NEVER put a document-format spec "
    "on a research/gather node — that causes format-bleed (the gatherer emits HTML instead of "
    "notes). 'none' is a VALID explicit choice; a node may carry MORE THAN ONE spec.\n"
    "- You set ONLY role + spec, NOT tools/bundles — every node SELF-SELECTS its bundle(s) at "
    "runtime. Bind a node's 'tool' to a real available tool only when its action is one a tool "
    "performs (search, fetch, read/write a file, send mail).\n\n"
    "Examples (role / spec):\n"
    "- 'research the US-Iran conflict and write an HTML report' → researcher/research-analyst "
    "(gather) → worker/html-writer (write) → reviewer/none (final).\n"
    "- 'write a haiku' → worker/none → reviewer/none.\n"
    "- 'research X, Y and Z in parallel then summarise' → 3× researcher/research-analyst "
    "(depends_on=[]) → worker/none (depends_on=[those 3]) → reviewer/none.\n\n"
    "DELIVERY: present in chat or save with file_write by default; use send_mail ONLY when the "
    "goal EXPLICITLY asks to be emailed. Each reply is EXACTLY ONE tool call as strict JSON "
    '{"tool": "<name>", "args": { ... }} — no prose, no code fences.'
)


class PlanningBundle(ObjectBundle):
    """Plan-authoring capability: shape/spec discovery + tool-driven DAG construction."""

    name = "planning"
    summary = (
        "AUTHOR a plan DAG — discover shapes/specs, then seed_plan -> add_step xN -> "
        "finalize_plan, recording a role + specialization per node. The PLANNER stage "
        "loads this; in-plan nodes do not."
    )

    @property
    def own_doctrine(self) -> str:  # type: ignore[override]
        return f"{_PLANNING_DOCTRINE}\n\n{self.plan_tool_catalog_text()}"

    # ------------------------------------------------------------------ #
    # the plan-building tool catalog (rendered from PLAN_TOOLS_SPEC) — the
    # incremental planner's _tool_catalog_text delegates here.
    # ------------------------------------------------------------------ #
    def plan_tool_catalog_text(self) -> str:
        """Render the plan-building tools + their args for the planner system prompt.

        Reproduces the incremental planner's prior inline rendering, sourced from the
        existing :data:`~agent_runtime.plan_tools.PLAN_TOOLS_SPEC`."""
        lines = ["PLAN-BUILDING TOOLS (call ONE per reply):"]
        for spec in PLAN_TOOLS_SPEC:
            lines.append(f"- {spec['name']}: {spec['description']}")
            for arg, meaning in spec["args"].items():
                lines.append(f"    {arg}: {meaning}")
        return "\n".join(lines)

    def plan_tool_names(self) -> frozenset[str]:
        """The plan-building tool names (seed_plan / add_step / set_node_spec /
        finalize_plan), from :data:`~agent_runtime.plan_tools.PLAN_TOOL_NAMES`."""
        return PLAN_TOOL_NAMES

    # The plan-building tools are NOT native function schemas — they are dispatched by
    # the PlanBuilder loop. tool_specs() exposes the base finish only; the plan tools
    # are surfaced via plan_tool_catalog_text() (which the doctrine embeds).
    def tool_specs(self, ctx: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
        return super().tool_specs(ctx)

    def tool_names(self, ctx: Optional[Mapping[str, Any]] = None) -> list[str]:
        # The planner's real surface is the plan-building tools + the discovery tools.
        return sorted(set(self.plan_tool_names()) | {"get_shapes", "get_specs"})

    # ------------------------------------------------------------------ #
    # handler-backed discovery tools (get_shapes / get_specs).
    # ------------------------------------------------------------------ #
    def register(self, registry: Any, ctx: Optional[Mapping[str, Any]] = None) -> Any:
        """Add get_shapes + get_specs onto ``registry`` — orchestrates
        :func:`agent_runtime.discovery_tools.register_discovery_tools`. ``ctx`` may
        carry ``shapes_dir`` / ``specs_dir`` / ``spec_index_provider``."""
        from ..discovery_tools import register_discovery_tools

        ctx = ctx or {}
        shapes_dir = ctx.get("shapes_dir")
        specs_dir = ctx.get("specs_dir")
        return register_discovery_tools(
            registry,
            shapes_dir=Path(shapes_dir) if shapes_dir else None,
            specs_dir=Path(specs_dir) if specs_dir else None,
            spec_index_provider=ctx.get("spec_index_provider"),
        )


__all__ = ["PlanningBundle"]
