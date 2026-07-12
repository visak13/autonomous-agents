"""RP-AUDIT F1 — the FORCED-FINALIZE turn (``IncrementalPlanner._finalize_user``) is
SHAPE-METHODOLOGY-AWARE, mirroring the RP-4c/d341 fix already applied to ``_system`` /
``_initial_user``.

Finding F1 (HIGH): ``_finalize_user`` — the turn the engine forces when the small model
repeats instead of finalizing — HARDCODED a gather→combine→deliver closing step ('Call
add_step ONCE for the FINAL step that combines the gathered results … its depends_on MUST
list EVERY gather step id') PLUS an output-format binding recipe (HTML writer for HTML …)
PLUS a delivery-tool assumption (file_write / send_mail), and — UNLIKE ``_system`` /
``_initial_user`` (which RP-4c made methodology-aware) — NEVER consulted the selected
shape's ``decompose_methodology``. So a stalled authoring run injected a combine-every-
gather-step node even for a shape that supplied a two-step (e.g. read→write) methodology —
a d341-class violation (the engine bakes a fixed gather flow that overrides shape doctrine).

Fix (this file guards it): when the selected shape supplies a ``decompose_methodology``,
``_finalize_user`` RENDERS that methodology with PRECEDENCE and the hardcoded gather→
combine→deliver + output-format + delivery-tool recipe is SUPPRESSED (fallback-only for
methodology-less shapes). The substitution is generic + shape-agnostic (a presence check
on the methodology field, NO spec-name/flow conditional); the engine renders shape text,
the model authors the step(s).

These tests are OFFLINE (no inference): they assert the forced-finalize PROMPT renders the
methodology with precedence when present and falls back byte-identically when absent, and
that no spec-name/flow conditional was introduced (anti-fab, d341/d310/d319).
"""
from __future__ import annotations

import inspect

from agent_runtime.factory import AbstractPlanFactory
from agent_runtime.incremental import IncrementalPlanner
from agent_runtime.plan_tools import PlanBuilder
from agent_runtime.shapes import load_shape
from llm_framework import FakeTransport
from specialization.registry import SpecRegistry
from specialization.seed import seed_canonical_rulesets


_TOOL_CATALOG = [
    {"name": "cron_add", "description": "schedule a recurring task"},
    {"name": "web_search", "description": "search the web"},
    {"name": "file_write", "description": "write a file"},
    {"name": "send_mail", "description": "send an email to the user"},
]

# Verbatim fragments of the NEUTRAL finalize fallback the methodology must REPLACE (RP-AUDIT F2
# neutralized this fallback: it directs the model to author the remaining/delivering step(s)
# WITHOUT the pre-F2 gather→combine→deliver + output-format recipe).
_GENERIC_COMBINE = "the single step that DELIVERS the goal's outcome"
_GENERIC_EVERY_GATHER = "Do NOT repeat any step already authored above"
_GENERIC_FORMAT_BIND = "do not yet COMPLETE the goal"

# The pre-F2 gather-mold finalize fragments the neutral fallback must NO LONGER contain (d341).
_OLD_FINALIZE_MOLD = (
    "combines the gathered results",
    "depends_on MUST list EVERY gather step id",
    "the HTML writer for HTML",
    "the gathering phase is COMPLETE",
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


def _builder_with_two_steps(tmp_path) -> PlanBuilder:
    """A builder in the exact state the forced-finalize trigger fires on: 2+ authored
    gather steps, no combine sink yet (no depends_on edges)."""
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)
    builder = PlanBuilder(
        spec_names=reg.names(),
        tool_names=[t["name"] for t in _TOOL_CATALOG],
        shape_name="schedule-leg",
        shape_description="schedule-only leg",
    )
    builder.dispatch("seed_plan", {"goal": "g"})
    builder.dispatch("add_step", {"task": "gather AI news", "tool": "web_search", "depends_on": []})
    builder.dispatch("add_step", {"task": "gather climate news", "tool": "web_search", "depends_on": []})
    return builder


# --------------------------------------------------------------------------- #
# 1. PRECEDENCE — the methodology REPLACES the hardcoded finalize recipe.
# --------------------------------------------------------------------------- #
def test_finalize_methodology_takes_precedence_over_hardcoded_recipe(tmp_path):
    methodology = load_shape("schedule-leg").decompose_methodology
    assert methodology.strip(), "fixture: schedule-leg must carry a decompose_methodology"
    p = _planner(tmp_path, methodology=methodology)
    builder = _builder_with_two_steps(tmp_path)
    goal = "Every day at 8am, research the AI news and email me a summary."

    finalize = p._finalize_user(goal, builder)

    # The shape's methodology is rendered into the forced-finalize turn.
    assert methodology.strip()[:60] in finalize
    # Precedence is stated explicitly.
    assert "PRECEDENCE" in finalize
    # The hardcoded gather→combine→deliver + output-format + delivery recipe is SUPPRESSED.
    assert _GENERIC_COMBINE not in finalize, "hardcoded combine recipe leaked despite a methodology"
    assert _GENERIC_EVERY_GATHER not in finalize, "combine-every-gather-step recipe leaked despite a methodology"
    assert _GENERIC_FORMAT_BIND not in finalize, "output-format-writer binding leaked despite a methodology"


# --------------------------------------------------------------------------- #
# 2. FALLBACK — with NO methodology, the hardcoded recipe is byte-identical to pre-F1.
# --------------------------------------------------------------------------- #
def test_finalize_no_methodology_falls_back_to_hardcoded_recipe(tmp_path):
    p = _planner(tmp_path, methodology="")
    builder = _builder_with_two_steps(tmp_path)
    goal = "Research AI and climate news and save a brief."

    finalize = p._finalize_user(goal, builder)

    # The NEUTRAL finalize fallback IS what a methodology-less shape gets.
    assert _GENERIC_COMBINE in finalize
    assert _GENERIC_EVERY_GATHER in finalize
    assert _GENERIC_FORMAT_BIND in finalize
    # RP-AUDIT F2: the finalize fallback is TOPOLOGY-NEUTRAL — the pre-F2 gather-mold is GONE.
    for frag in _OLD_FINALIZE_MOLD:
        assert frag not in finalize, f"gather-mold finalize fragment leaked: {frag!r}"
    # No stray methodology/precedence scaffolding when there is no methodology.
    assert "PRECEDENCE" not in finalize
    assert "following THIS authoring methodology" not in finalize


# --------------------------------------------------------------------------- #
# 3. ANTI-FAB — the substitution is a generic presence check, not a spec-name/flow
#    conditional, and the engine renders shape text (it does not author structure).
# --------------------------------------------------------------------------- #
def test_finalize_substitution_is_generic_no_spec_name_conditional():
    src = inspect.getsource(IncrementalPlanner._finalize_user)
    # The branch is a presence check on the generic methodology FIELD …
    assert "self.shape_decompose_methodology" in src
    assert "if methodology:" in src
    # … NOT a shape/spec-NAME conditional (the d341/d278 fabrication smell).
    assert "schedule-leg" not in src
    assert 'shape_name ==' not in src and "shape_name==" not in src
    assert "== 'schedule" not in src and '== "schedule' not in src


# --------------------------------------------------------------------------- #
# 4. Mechanism parity — _finalize_user reads the SAME methodology field _system /
#    _initial_user do (the RP-4c fix applied identically to all three turns).
# --------------------------------------------------------------------------- #
def test_finalize_reads_same_methodology_field_as_system_and_initial():
    for method in (
        IncrementalPlanner._system,
        IncrementalPlanner._initial_user,
        IncrementalPlanner._finalize_user,
    ):
        assert "self.shape_decompose_methodology" in inspect.getsource(method)
