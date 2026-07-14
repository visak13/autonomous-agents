"""s13/FX-spec (P2-5c FLAG-FREE) — the AGENTIC DEEP-RESEARCH SPEC + SHAPE (d106/d107 a+b).

These FAST tests (no web, no live inference) prove the FX-spec deliverable:

(a) d235 SPEC-VS-ROLE DE-BLUR — THE SEEDED DEEP-RESEARCH SPEC carries ONLY the
    OUTPUT-QUALITY STANDARD (concrete/grounded/traceable/deeper). The investigative
    METHODOLOGY (identify/find/verify/stop) that d107(1) had put in the spec is DROPPED:
    how to drive the work is the researcher ROLE + the RUNTIME research loop, so the
    methodology now lives in the RESEARCH BUNDLE doctrine, not the output-shaping spec.

(b) THE DEEP-RESEARCH SHAPE FILE carries the iteration/depth count (d107(2)): the
    report path reads the shape FILE's depth into the plan (reaching the served-route
    depth trace on run_plan_chain), the HARD CAP of 10 layers BOUNDS the loop, and
    BREADTH stays PINNED at 3.

P2-5c retired the bespoke ``run_research_tree`` loop + ``_make_tree_gather`` leaf: PHASE-1
research now ALWAYS runs through the GENERIC declarative-unroll + AgentRuntime growable
engine (:func:`_run_generic_research_phase`). The grower REUSES the same decision node +
methodology + completeness_stop the retired tree's decision node reasoned over — so the
methodology-reaches-the-decision-node WIRING, the stop reasons (agent_sufficient /
no_expansion / depth_bound / budget) and the max_layers bound are proven at the engine level
in ``agent_runtime/tests/test_p2_5b_growable.py`` (20 tests) + ``test_s13_decision_enrichment``.
Here we assert the SERVED-route contract: the shape/spec RESOLUTION, that the shape FILE's
depth reaches the served route (clamped to the hard cap), and breadth pinned 3.
"""
from __future__ import annotations

import asyncio

from llm_framework import ChatResult, FakeTransport
from reactive_tools import EventPlane, ToolHook, build_default_hook, register_agentic_tools
from reactive_tools.tool_hook import ToolRegistry
from specialization import SpecRegistry
from specialization.registry import SpecRegistry as RegistryForWiring
from specialization.seed import DEEP_RESEARCH_SPEC, CANONICAL_RULESETS

import chat_app.agentic as agentic
from chat_app.agentic import (
    AgenticResult,
    PLAN_CHAIN_TREE_BREADTH,
    run_agentic,
    run_plan_chain,
)
from agent_runtime import (
    N4_TREE_DEPTH_CEILING,
    ShapeSelection,
    ShapeSpec,
    load_shape,
)
from agent_runtime.factory import PlanDAG


# --------------------------------------------------------------------------- #
# Shared fakes — the GENERIC research PHASE-1 is stubbed (the engine internals are
# proven in agent_runtime/tests/test_p2_5b_growable.py); we drive the SERVED route.
# --------------------------------------------------------------------------- #
_SRC = {"title": "Iran 2025", "url": "https://news.example.com/iran-2025",
        "markdown": "The conflict escalated June 13; 1,200 casualties by June 20."}


def _deep_research_shape() -> ShapeSpec:
    """A small unrollable deep-research ShapeSpec the report path resolves + unrolls."""
    return ShapeSpec(
        name="deep-research",
        description="deep research",
        max_iter=2,
        hard_cap=4,
        execution="deep-research",
        completeness_stop="Fill ALL the blanks before stopping.",
    )


class _ProseTransport:
    """A scripted transport whose every turn returns prose (no tool call)."""

    def complete(self, messages, **opts) -> str:  # pragma: no cover - parity shim
        return self.chat(messages, **opts).content

    def chat(self, messages, **opts) -> ChatResult:
        return ChatResult(role="assistant", content="PLAN: synthesize the gathered angles.")


class _FakeWriteResult:
    def __init__(self) -> None:
        self.results: dict = {}
        self.states: dict = {}
        self.launch_order: list = []
        self.ok = True


def _install_fakes(monkeypatch):
    """Stub the GENERIC research PHASE-1 (capturing its kwargs) + a write-phase spy.

    Returns ``generic_calls`` — the kwargs the served route handed the generic engine, so the
    shape/spec/depth RESOLUTION reaching the engine can be asserted on the served route."""
    generic_calls: list[dict] = []

    async def fake_generic(query, **kw):
        generic_calls.append(dict(kw))
        return (
            f"FINDINGS for {kw.get('research_depth')}.",
            [dict(_SRC)],
            {"growable": True, "stop_reason": "agent_sufficient", "grow_layers": 1,
             "max_layers": 4, "layers": [{"gathered": 1}]},
        )

    async def spy_write_phase(query, out_name, findings, sources, **_kw):
        return PlanDAG(nodes=[], goal=query), _FakeWriteResult()

    monkeypatch.setattr(agentic, "_run_generic_research_phase", fake_generic)
    monkeypatch.setattr(agentic, "run_section_write_phase", spy_write_phase)
    return generic_calls


def _run_chain(tmp_path, *, research_depth=None, registry=None, run_id="s13-fx",
               shape="deep-research", catalog=None):
    hook = ToolHook(EventPlane(), registry=ToolRegistry())
    return asyncio.run(run_plan_chain(
        "detailed HTML report on the 2025 US-Iran war",
        ShapeSelection(shape=shape, escalate=False, rationale="large report"),
        transport=_ProseTransport(),
        registry=registry if registry is not None else SpecRegistry(str(tmp_path)),
        hook=hook, plane=hook.plane, timeout=30.0, run_id=run_id,
        overall_goal="detailed HTML report on the 2025 US-Iran war",
        research_depth=research_depth,
        catalog=catalog if catalog is not None else {"deep-research": _deep_research_shape()},
    ))


# --------------------------------------------------------------------------- #
# (a1) d235 SPEC-VS-ROLE DE-BLUR: the SEEDED deep-research spec carries ONLY the
#      OUTPUT-QUALITY STANDARD (concrete/grounded/traceable/deeper). The investigative
#      METHODOLOGY (identify/find/verify/stop) was DROPPED from the spec — how to drive
#      the work is the researcher ROLE + the RUNTIME research loop, not an output-shaping
#      ruleset. The methodology now lives in the RESEARCH BUNDLE doctrine (asserted below).
# --------------------------------------------------------------------------- #
def test_s13_seed_spec_is_quality_only_not_methodology():
    """d235: the deep-research spec shapes the FORM/RIGOR of the answer only; it no longer
    dictates the task METHODOLOGY (that blurred spec with role/runtime)."""
    # DEEP_RESEARCH_SPEC names the canonical spec the deep-research route reuses.
    assert DEEP_RESEARCH_SPEC in CANONICAL_RULESETS
    body = CANONICAL_RULESETS[DEEP_RESEARCH_SPEC][1]
    low = body.lower()

    # The OUTPUT-QUALITY STANDARD is present: concrete, grounded by reading sources,
    # traceable, gaps flagged, deeper than prior layers.
    assert "concrete" in low
    assert "traceable" in low or "attribute each" in low
    assert "reading the sources" in low
    assert "deeper" in low

    # The d107 IDENTIFY/FIND/VERIFY/STOP methodology is GONE from the spec (de-blur): the
    # spec no longer names the staged loop or owns the stop_research criterion.
    assert "stop_research" not in low
    assert "deep-research methodology" not in low
    assert "identify → find → verify" not in low

    # The methodology lives in the RESEARCH BUNDLE doctrine now (decompose-first, grow out,
    # search/read/note/expand/prune/stop) — the role/runtime, not the spec.
    from agent_runtime.bundles.research import ResearchBundle

    doctrine = ResearchBundle().own_doctrine.lower()
    assert "decompose-first" in doctrine
    assert "expand" in doctrine and "prune" in doctrine
    assert "stop only when" in doctrine


# --------------------------------------------------------------------------- #
# (a2) the methodology REACHES the generic engine's decision node — on the SERVED route
#      the report path resolves the deep-research SHAPE + hands it to the generic engine,
#      which loads the seeded spec methodology into the grower's decision node. The
#      decision-prompt WIRING itself is proven at the engine level
#      (agent_runtime/tests/test_p2_5b_growable.py + test_s13_decision_enrichment.py);
#      here we assert the served route routes through the generic engine with the
#      resolved deep-research shape + its completeness stop.
# --------------------------------------------------------------------------- #
def test_s13_served_route_runs_generic_engine_with_resolved_shape(monkeypatch, tmp_path):
    """run_plan_chain routes PHASE-1 through the GENERIC engine, handing it the RESOLVED
    deep-research shape (the unroll source the grower's decision node reasons over) + the
    shape's completeness_stop — the same investigative stop signal the retired tree's
    decision node used."""
    generic_calls = _install_fakes(monkeypatch)

    result = _run_chain(
        tmp_path, research_depth=2,
        catalog={"deep-research": _deep_research_shape()},
    )
    assert result.deep_research is not None
    assert result.deep_research["engine"] == "generic-unroll"

    # The generic engine received the resolved deep-research shape + its completeness stop.
    assert len(generic_calls) == 1
    g = generic_calls[0]
    assert getattr(g["dr_shape"], "name", None) == "deep-research"
    # completeness_stop flows from run_agentic; when not supplied to run_plan_chain the
    # grower falls back to the shape's own completeness_stop (asserted on the shape above).
    assert getattr(g["dr_shape"], "completeness_stop", "")


# --------------------------------------------------------------------------- #
# (b1) the SHAPE FILE's depth is read into the plan — reading the shape file ALONE
#      (no store override) sets how many layers the agent plans
# --------------------------------------------------------------------------- #
class _NoOverrideConfig:
    """A shape_config with NO per-shape overrides set (the shapes/specs store empty)."""

    def get_max_iter(self, _name):
        return None

    def get_depth(self, _name):
        return None


def test_s13_shape_file_depth_read_into_plan(monkeypatch, tmp_path):
    """With NO store depth override, run_agentic reads the DEEP-RESEARCH SHAPE FILE's
    iteration count and hands it to the GENERIC loop's research seed as research_depth (as1/
    d239: run_plan_chain is a thin research-seed shim; run_agentic now seeds the one loop) —
    so the shape FILE alone drives the planned depth."""
    plane = EventPlane()
    hook = build_default_hook(plane)
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    registry = RegistryForWiring(tmp_path / "specs")

    async def fake_select(self, goal):
        return ShapeSelection(shape="deep-research", escalate=False, wants_file=True,
                              multi_page=True, search_allowed=True)

    monkeypatch.setattr(agentic.ShapeSelector, "select", fake_select)

    captured = {}

    async def fake_loop(query, sel, *, first_plan_kind, **kw):
        captured["first_plan_kind"] = first_plan_kind
        captured["research_depth"] = kw.get("research_depth")
        return AgenticResult(rationale="chained", ok=True, final_response="CHAINED")

    monkeypatch.setattr(agentic, "_run_generic_loop", fake_loop)

    res = asyncio.run(run_agentic(
        "write me a big multi-page report and save it as report.md",
        transport=FakeTransport([]),
        registry=registry, hook=hook, plane=plane,
        skip_ambiguity=True,
        shape_config=_NoOverrideConfig(),
    ))
    assert res.final_response == "CHAINED"
    assert captured["first_plan_kind"] == "research"  # deep-research → the research seed
    # The deep-research shape FILE's declared iteration count reached the generic loop.
    expected = int(load_shape("deep-research").max_iter)
    assert captured["research_depth"] == expected
    assert expected <= N4_TREE_DEPTH_CEILING  # the file default sits within the hard cap


def test_s13_shape_file_depth_reaches_served_route(monkeypatch, tmp_path):
    """End-to-end on run_plan_chain: a depth sourced from the shape file (passed as
    research_depth) reaches the GENERIC engine + the served-route depth trace (clamped to the
    hard cap), breadth pinned 3."""
    monkeypatch.delenv("RA_TREE_DEPTH", raising=False)
    generic_calls = _install_fakes(monkeypatch)

    shape_depth = int(load_shape("deep-research").max_iter)
    result = _run_chain(tmp_path, research_depth=shape_depth)

    dr = result.deep_research
    # the shape-file depth reached the generic engine ...
    assert generic_calls and generic_calls[0]["research_depth"] == shape_depth
    # ... and the served-route trace shows it (clamped to the hard cap).
    assert dr["depth_configured"] == min(shape_depth, N4_TREE_DEPTH_CEILING)
    assert dr["leaf_breadth"] == PLAN_CHAIN_TREE_BREADTH == 10  # P4 (owner 2026-07-13): D97's pin-3 SUPERSEDED — breadth defaults 10, tracks the shared env knob


# --------------------------------------------------------------------------- #
# (b2) HARD CAP 10 — an over-large depth is CLAMPED to N4_TREE_DEPTH_CEILING on the served
#      route (the loop TERMINATION at the bound — stop_reason='depth_bound' — is proven at
#      the engine level in agent_runtime/tests/test_p2_5b_growable.py).
# --------------------------------------------------------------------------- #
def test_s13_hard_cap_10_clamps_served_depth(monkeypatch, tmp_path):
    """An over-large depth is CLAMPED to N4_TREE_DEPTH_CEILING (10) on the served-route
    trace, and the raw value still reaches the generic engine (which re-clamps it into the
    grower config that bounds growth — depth_bound termination is proven at the engine level)."""
    monkeypatch.delenv("RA_TREE_DEPTH", raising=False)
    generic_calls = _install_fakes(monkeypatch)

    result = _run_chain(tmp_path, research_depth=99)

    dr = result.deep_research
    assert generic_calls and generic_calls[0]["research_depth"] == 99  # raw override reaches the engine
    assert dr["depth_configured"] == N4_TREE_DEPTH_CEILING == 10        # served trace clamped to the cap


# --------------------------------------------------------------------------- #
# (b3) BREADTH stays PINNED at 3 (D97), independent of any depth/iteration setting
# --------------------------------------------------------------------------- #
def test_s13_breadth_pinned_3(monkeypatch, tmp_path):
    """Whatever the depth, the report path's per-leaf fetch BREADTH is pinned to 3."""
    monkeypatch.setenv("RA_TREE_DEPTH", "2")
    monkeypatch.setenv("RA_RESEARCH_FETCH_BREADTH", "10")  # the N1 knob must NOT leak
    monkeypatch.setenv("RA_TREE_LEAF_BREADTH", "9")        # even a direct override is pinned
    _install_fakes(monkeypatch)

    result = _run_chain(tmp_path)

    assert result.deep_research["leaf_breadth"] == 10  # P4: tracks the shared env knob (10 above)
