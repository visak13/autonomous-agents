"""RP-6b (d359/d361) — SELF-POLICING: the deep-research PHASES + per-phase SPEC-ROUTING are
DECLARED IN THE SHAPE, and the engine READS them (no hardcoded research-first seed, no fixed
phase enum, no engine flow/spec-name conditional). Anti-fabrication charter (d341/d319): the
deep-research FLOW must EMERGE from the definition layer (deep-research.toml ``[[phases]]``),
not be baked in engine code.

These assertions are deterministic + offline (they inspect the shape declaration, the derived
vocabulary, the spec-routing, and the ENGINE SOURCE — never a live model run). A test here that
FAILS is the charter catching a regression: a hardcoded seed/enum reintroduced, or a writer spec
routed onto a research node (Bug A d355/d356).
"""
from __future__ import annotations

import inspect
import re
import tempfile
from pathlib import Path

from agent_runtime.planner import FOLLOWUP_PLANS
from agent_runtime.shapes import load_shape
from specialization.registry import SpecRegistry
from specialization.seed import DEEP_RESEARCH_SPEC, seed_canonical_rulesets
from reactive_tools import EventPlane, build_default_hook, register_agentic_tools
from llm_framework import FakeTransport

import chat_app.agentic as ag
import agent_runtime.planner as planner_mod


def _seeded_registry() -> SpecRegistry:
    reg = SpecRegistry(Path(tempfile.mkdtemp()) / "specs")
    seed_canonical_rulesets(reg)  # research-analyst + markdown-writer + …
    return reg


# --------------------------------------------------------------------------- #
# 1) The SHAPE declares the phases + per-phase spec-routing (definition layer).
# --------------------------------------------------------------------------- #
def test_deep_research_shape_declares_ordered_phases_with_spec_roles():
    dr = load_shape("deep-research")
    # Ordered phases: GATHER (research) then AUTHOR (write) — the flow is in the shape.
    assert [p.kind for p in dr.phases] == ["research", "write"]
    # Per-phase spec-routing: research nodes → research role, write node → writer role.
    assert dr.spec_role_for("research") == "research"
    assert dr.spec_role_for("write") == "writer"
    # The first declared phase is what SEEDS the loop.
    assert dr.first_phase_kind == "research"
    # The phase ORDER drives the default transition (research → write → done).
    assert dr.next_phase_plan("research") == "write_plan"
    assert dr.next_phase_plan("write") == "done"


# --------------------------------------------------------------------------- #
# 2) The ENGINE READS the shape (the fixed enum + the hardcoded seed are GONE).
# --------------------------------------------------------------------------- #
def test_followup_plans_enum_derives_from_the_shape():
    # The planner's follow-up vocabulary is the shape's declared vocabulary — not a literal.
    assert FOLLOWUP_PLANS == load_shape("deep-research").followup_plans


def test_planner_source_has_no_hardcoded_phase_enum_literal():
    """The fixed ``FOLLOWUP_PLANS = ("research_plan", …)`` literal (planner.py) is RETIRED —
    the enum is assigned from the shape-derived helper."""
    src = inspect.getsource(planner_mod)
    # It must be sourced from the shape helper ...
    assert "_followup_plans_from_shape()" in src
    assert "FOLLOWUP_PLANS: tuple[str, ...] = _followup_plans_from_shape()" in src
    # ... and the retired hardcoded tuple literal assignment must NOT reappear.
    assert not re.search(
        r'FOLLOWUP_PLANS\s*:\s*tuple\[str,\s*\.\.\.\]\s*=\s*\(\s*"research_plan"', src
    )


def test_agentic_route_seeds_first_phase_from_the_shape():
    """The hardcoded research-first seed (``first_plan_kind="research" if route_research …``)
    is RETIRED — the seed reads the shape's first declared phase."""
    src = inspect.getsource(ag.run_agentic) if hasattr(ag, "run_agentic") else inspect.getsource(ag)
    # The retired hardcoded seed literal must be GONE ...
    assert 'first_plan_kind="research" if route_research else "acyclic"' not in src
    # ... replaced by reading the shape's first declared phase.
    assert ".first_phase_kind" in src


def test_loop_default_transitions_read_from_the_shape():
    """The loop's safe-baseline transitions come from the shape's phase order, not literals."""
    src = inspect.getsource(ag._run_generic_loop)
    assert 'dr_shape.next_phase_plan("research")' in src
    assert 'dr_shape.next_phase_plan("write")' in src
    # the retired hardcoded baselines must not reappear as the research/write default.
    assert 'default_next = "write_plan"' not in src


# --------------------------------------------------------------------------- #
# 3) Bug A dissolved — spec-routing follows the shape; no spec-name conditional.
# --------------------------------------------------------------------------- #
def test_bug_a_research_seed_never_carries_a_writer_spec():
    reg = _seeded_registry()
    dr = load_shape("deep-research")
    # The research seed carries the research-analysis spec (the shape's research role) ...
    assert ag._deep_research_spec(reg, shape=dr) == DEEP_RESEARCH_SPEC
    # ... and NEVER a user-named writer spec (the old F5 first-requested-wins = Bug A).
    assert ag._deep_research_spec(reg, shape=dr) != "markdown-writer"


def test_named_writer_reachable_on_the_write_authorer_its_correct_home():
    reg = _seeded_registry()
    hook = build_default_hook(EventPlane(), file_base=Path(tempfile.mkdtemp()))
    register_agentic_tools(hook, file_base=Path(tempfile.mkdtemp()), cron_data_dir=Path(tempfile.mkdtemp()))
    wp = ag._build_incremental_planner(
        transport=FakeTransport([]), registry=reg, hook=hook, shape_spec=None,
        requested_specs=["markdown-writer", "no-such-spec"],
    )
    # The user-named writer is threaded onto the WRITE authorer (reachable on the deliverable
    # node), so removing it from the research seed does NOT make it unreachable.
    assert wp.requested_specs == ["markdown-writer"]


def test_deep_research_spec_no_longer_scans_requested_specs():
    """The Bug A source (``for name in requested_specs: if name in registry: return name``) is
    RETIRED — the research spec is routed by the shape's declared role, not the first requested
    spec (a spec-NAME pick that let a writer land on a research node)."""
    src = inspect.getsource(ag._deep_research_spec)
    assert "for name in requested_specs" not in src
    assert "spec_role_for" in src  # it READS the shape's declared role instead


def test_engine_maps_role_not_spec_name_no_writer_name_conditional():
    """Anti-fab: the routing is a declarative ROLE→default-spec map (the role comes from the
    shape), NOT a spec-NAME conditional in the engine (banned d310/d311/d341)."""
    # The role map is keyed by ROLE, and only the research role is engine-resolved (the writer
    # role flows through the write planner).
    assert set(ag._SPEC_ROLE_DEFAULTS) == {"research"}
    src = inspect.getsource(ag._deep_research_spec)
    # No spec-name equality/membership branch on writer names in the routing helper.
    assert "markdown-writer" not in src
    assert "html-writer" not in src
    assert 'spec_role_for("research")' in src
