"""RP-4c / d341 — the SCHEDULE-ONLY authoring doctrine lives in a SHAPE's decompose_methodology,
and the incremental planner GENERICALLY substitutes the selected shape's methodology into its
authoring procedure, TAKING PRECEDENCE over the hardcoded generic gather→combine→deliver recipe.

Root cause (RP-4c forensic): the schedule-only mandate reached the authoring SYSTEM prompt (the
factory description) but was OVERRIDDEN by the concrete generic gather→deliver DECISION PROCEDURE
in ``IncrementalPlanner._initial_user`` / ``_system`` — which had no schedule branch — so the
planner authored run-now gather+email nodes alongside the cron_add leg. The d341 fix: carry the
schedule-only AUTHORING procedure in the ``schedule-leg`` shape's ``decompose_methodology`` and let
the planner substitute the SELECTED shape's methodology into the procedure with PRECEDENCE (the
generic recipe becomes a fallback for methodology-less shapes). Behaviour lives in the SHAPE
(definition layer); the substitution is generic + shape-agnostic (no spec-name/flow conditional),
mirroring the d161/d170 research_tree decompose_methodology substitution.

These tests are OFFLINE (no inference): they assert the shape carries the doctrine, and that the
planner PROMPTS render the methodology with precedence when present and fall back byte-identically
when absent. The live authoring-rate is measured separately (mixed phrasing, human-read).
"""
from __future__ import annotations

from agent_runtime.factory import AbstractPlanFactory
from agent_runtime.incremental import IncrementalPlanner
from agent_runtime.shapes import load_shape
from llm_framework import FakeTransport
from specialization.registry import SpecRegistry
from specialization.seed import seed_canonical_rulesets


_TOOL_CATALOG = [
    {"name": "cron_add", "description": "schedule a recurring task"},
    {"name": "web_search", "description": "search the web"},
    {"name": "send_mail", "description": "send an email to the user"},
]

# A verbatim fragment of the NEUTRAL generic fallback the methodology must REPLACE (RP-AUDIT F2
# neutralized this fallback: it is topology-neutral, no longer a gather→combine→deliver recipe).
_GENERIC_INITIAL = "Work out the steps THIS goal actually needs"
_GENERIC_SYSTEM_FLOW = "Author the SMALLEST correct set of steps for THIS goal"

# The pre-F2 gather-mold fragments the neutral fallback must NO LONGER contain (d341: the engine
# must not bake a fixed gather→combine→deliver flow into the methodology-less fallback).
_OLD_GATHER_MOLD = (
    "A modular-parallel plan is INDEPENDENT sub-task steps",
    "List the DISTINCT ITEMS the GOAL names",
    "combines and delivers",
    "The FINAL combine step's depends_on lists EVERY sub-task id",
)


def _planner(tmp_path, *, methodology: str) -> IncrementalPlanner:
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)
    factory = AbstractPlanFactory(reg.index(), tool_catalog=_TOOL_CATALOG)
    return IncrementalPlanner(
        FakeTransport([]),
        factory,
        spec_names=reg.names(),
        tool_names=[t["name"] for t in _TOOL_CATALOG],
        shape_name="schedule-leg",
        shape_description="schedule-only leg",
        shape_decompose_methodology=methodology,
    )


# --------------------------------------------------------------------------- #
# 1. The schedule-leg SHAPE carries the schedule-only authoring doctrine.
# --------------------------------------------------------------------------- #
def test_schedule_leg_shape_carries_schedule_only_methodology(tmp_path):
    shape = load_shape("schedule-leg")
    assert shape.name == "schedule-leg"
    dm = shape.decompose_methodology
    assert dm.strip(), "schedule-leg must carry a decompose_methodology (the authoring doctrine)"
    low = dm.lower()
    # The doctrine says: ONE cron_add node, recurring-scheduler, whole task, NO run-now.
    assert "cron_add" in low and "recurring-scheduler" in low
    assert "exactly one" in low
    assert "run-now" in low  # the explicit "no run-now" clause
    # recurring-scheduler is a REGISTERED spec (so the builder accepts the authored binding).
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)
    assert "recurring-scheduler" in set(reg.names())


# --------------------------------------------------------------------------- #
# 2. PRECEDENCE — the methodology REPLACES the generic recipe in BOTH prompts.
# --------------------------------------------------------------------------- #
def test_methodology_takes_precedence_over_generic_recipe(tmp_path):
    methodology = load_shape("schedule-leg").decompose_methodology
    p = _planner(tmp_path, methodology=methodology)
    goal = "Every day at 8am, research the AI news and email me a summary."

    sysmsg = p._system(goal)
    initial = p._initial_user(goal)

    # The shape's methodology is rendered into BOTH the system prompt and the first user turn.
    assert methodology.strip()[:60] in sysmsg
    assert methodology.strip()[:60] in initial
    # Precedence is stated explicitly.
    assert "PRECEDENCE" in sysmsg and "PRECEDENCE" in initial
    # The generic gather→deliver recipe is SUPPRESSED (fallback-only) when a methodology is present.
    assert _GENERIC_INITIAL not in initial, "generic decision procedure leaked despite a methodology"
    assert _GENERIC_SYSTEM_FLOW not in sysmsg, "generic gather-flow guidance leaked despite a methodology"


# --------------------------------------------------------------------------- #
# 3. FALLBACK — with NO methodology, the generic recipe is byte-identical to pre-d341.
# --------------------------------------------------------------------------- #
def test_no_methodology_falls_back_to_generic_recipe(tmp_path):
    p = _planner(tmp_path, methodology="")
    goal = "Research AI and climate news and save a brief."
    sysmsg = p._system(goal)
    initial = p._initial_user(goal)
    # The NEUTRAL generic fallback is present (the fallback for methodology-less shapes).
    assert _GENERIC_INITIAL in initial
    assert _GENERIC_SYSTEM_FLOW in sysmsg
    # RP-AUDIT F2: the fallback is TOPOLOGY-NEUTRAL — the pre-F2 gather→combine→deliver mold is GONE.
    for frag in _OLD_GATHER_MOLD:
        assert frag not in sysmsg and frag not in initial, f"gather-mold fragment leaked: {frag!r}"
    # No stray precedence/methodology scaffolding when there is no methodology.
    assert "AUTHORING METHODOLOGY (selected shape" not in sysmsg
