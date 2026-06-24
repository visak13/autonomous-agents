"""s13/B6 (P2-5c FLAG-FREE) — research DEPTH is settable via the SHAPES/SPECS flow.

B6 makes the report path's research DEPTH a USER-controlled knob through the SAME
shapes/specs config store that already holds ``max_iter`` (not an env-only knob), while
BREADTH stays PINNED at 3 (D97).

P2-5c retired the bespoke ``run_research_tree`` loop: PHASE-1 research now ALWAYS runs
through the GENERIC declarative-unroll + AgentRuntime growable engine
(:func:`_run_generic_research_phase`). DEPTH still reaches the generic phase — run_plan_chain
clamps the override into ``tree_config`` (the served-route span trace) and passes
``research_depth`` to the generic phase, which builds the grower's config clamped to
``[1, N4_TREE_DEPTH_CEILING]``.

These FAST tests (no web, no live inference) prove:

* a per-shape ``depth=N`` set through :class:`ShapeConfigStore` (the shapes/specs path)
  reaches ``_run_generic_research_phase`` as ``research_depth`` and shows on the served-route
  ``deep_research.depth_configured`` trace (clamped), with breadth still 3;
* depth is CLAMPED to the hard ``N4_TREE_DEPTH_CEILING`` (≤10) the user fixed;
* with no override, the env baseline (``RA_TREE_DEPTH``) stands;
* ``run_agentic`` READS the depth override from the shape config and hands it to
  ``run_plan_chain`` (the wiring, driven through the real ``run_agentic``);
* the store round-trips the depth override durably and rejects nonsensical values;
* NO scaffolding/enable flag gates the GENERIC research engine on the default route (audit).
"""
from __future__ import annotations

import asyncio
import inspect
import re

import pytest

from llm_framework import ChatResult, FakeTransport
from reactive_tools import EventPlane, ToolHook, build_default_hook, register_agentic_tools
from reactive_tools.tool_hook import ToolRegistry
from specialization import SpecRegistry
from specialization.registry import SpecRegistry as RegistryForWiring

import chat_app.agentic as agentic
from chat_app.agentic import AgenticResult, run_agentic, run_plan_chain, PLAN_CHAIN_TREE_BREADTH
from chat_app.shape_config import ShapeConfigStore
from agent_runtime import (
    N4_TREE_DEPTH_CEILING,
    ShapeSelection,
    ShapeSpec,
    TreeConfig,
)
from agent_runtime.factory import PlanDAG


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
        round_roles=["research", "critic"],
        final_roles=["research", "synthesis", "verify"],
        completeness_stop="Fill ALL the blanks before stopping.",
    )


class _ProseTransport:
    """A scripted transport whose every turn returns prose (no tool call)."""

    def complete(self, messages, **opts) -> str:  # pragma: no cover - parity shim
        return self.chat(messages, **opts).content

    def chat(self, messages, **opts) -> ChatResult:
        return ChatResult(role="assistant", content="PLAN: gather then synthesize.")


class _FakeWriteResult:
    def __init__(self) -> None:
        self.results: dict = {}
        self.states: dict = {}
        self.launch_order: list = []
        self.ok = True


def _install_fakes(monkeypatch):
    """Stub the GENERIC research PHASE-1 (capturing the ``research_depth`` it received) +
    a write-phase spy. Returns ``seen_depth`` — the list of research_depth values the
    served route handed the generic engine."""
    seen_depth: list = []

    async def fake_generic(query, **kw):
        seen_depth.append(kw.get("research_depth"))
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
    return seen_depth


def _run_chain(tmp_path, *, research_depth):
    hook = ToolHook(EventPlane(), registry=ToolRegistry())
    return asyncio.run(run_plan_chain(
        "detailed HTML report on the 2025 US-Iran war",
        ShapeSelection(shape="plan-chain", escalate=False, rationale="large report"),
        transport=_ProseTransport(),
        registry=SpecRegistry(str(tmp_path)),
        hook=hook, plane=hook.plane, timeout=30.0, run_id="s13-b6",
        overall_goal="detailed HTML report on the 2025 US-Iran war",
        research_depth=research_depth,
        catalog={"deep-research": _deep_research_shape()},
    ))


# --------------------------------------------------------------------------- #
# (1) depth settable via the shapes/specs path → reaches the generic phase, breadth pinned 3
# --------------------------------------------------------------------------- #
def test_s13_depth_set_via_shape_store_reaches_generic_phase(monkeypatch, tmp_path):
    """A per-shape depth set through the SHAPES/SPECS store (the same path as max_iter)
    reaches ``_run_generic_research_phase`` as research_depth and shows (clamped) on the
    served-route ``depth_configured`` trace; breadth stays pinned at 3."""
    monkeypatch.delenv("RA_TREE_DEPTH", raising=False)
    seen_depth = _install_fakes(monkeypatch)

    # The shapes/specs flow: the user sets depth on a shape; the store persists it.
    store = ShapeConfigStore(tmp_path / "data")
    store.set_depth("plan-chain", 7)
    research_depth = store.get_depth("plan-chain")  # what run_agentic reads + hands down

    result = _run_chain(tmp_path, research_depth=research_depth)

    dr = result.deep_research
    assert dr is not None and dr.get("plan_chain") is True
    # depth from the shapes/specs store reached the GENERIC engine on the SERVED route.
    assert seen_depth == [7]
    # and it shows on the served-route depth_configured trace (within the cap).
    assert dr["depth_configured"] == 7
    # BREADTH stayed pinned at 3 (D97).
    assert dr["leaf_breadth"] == PLAN_CHAIN_TREE_BREADTH == 3
    store.close()


def test_s13_depth_clamped_to_ceiling(monkeypatch, tmp_path):
    """An out-of-range depth override is clamped to N4_TREE_DEPTH_CEILING (≤10) on the
    served-route trace (the generic phase clamps it again into its grower config)."""
    monkeypatch.delenv("RA_TREE_DEPTH", raising=False)
    seen_depth = _install_fakes(monkeypatch)
    result = _run_chain(tmp_path, research_depth=99)
    # the raw override reaches the generic phase (which clamps inside its grower config) ...
    assert seen_depth == [99]
    # ... and the served-route trace shows the clamped depth bound.
    assert result.deep_research["depth_configured"] == N4_TREE_DEPTH_CEILING == 10


def test_s13_no_depth_override_uses_env_baseline(monkeypatch, tmp_path):
    """With no shapes/specs override (research_depth=None), the env baseline stands."""
    monkeypatch.setenv("RA_TREE_DEPTH", "4")
    seen_depth = _install_fakes(monkeypatch)
    result = _run_chain(tmp_path, research_depth=None)
    # None reaches the generic phase (it falls back to the env baseline inside) ...
    assert seen_depth == [None]
    # ... and the served-route trace reflects the env baseline depth.
    assert result.deep_research["depth_configured"] == 4
    assert result.deep_research["leaf_breadth"] == 3  # breadth still pinned


# --------------------------------------------------------------------------- #
# (2) run_agentic READS the depth override from the shape config and passes it down
# --------------------------------------------------------------------------- #
class _DepthOnlyConfig:
    """A shape_config stub exposing the shapes/specs depth + max_iter read surface."""

    def __init__(self, depth):
        self._depth = depth

    def get_max_iter(self, _name):
        return None

    def get_depth(self, _name):
        return self._depth


def test_s13_run_agentic_reads_depth_from_shape_config(monkeypatch, tmp_path):
    """Driven through the REAL run_agentic: the multi-page file route reads the depth
    override from the shape config and hands it to run_plan_chain as research_depth."""
    plane = EventPlane()
    hook = build_default_hook(plane)
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    registry = RegistryForWiring(tmp_path / "specs")

    async def fake_select(self, goal):
        return ShapeSelection(shape="plan-chain", escalate=False, wants_file=True,
                              multi_page=True, search_allowed=True)

    monkeypatch.setattr(agentic.ShapeSelector, "select", fake_select)

    captured = {}

    async def fake_chain(query, sel, **kw):
        captured["research_depth"] = kw.get("research_depth")
        return AgenticResult(rationale="chained", ok=True, final_response="CHAINED")

    monkeypatch.setattr(agentic, "run_plan_chain", fake_chain)

    res = asyncio.run(run_agentic(
        "write me a big multi-page report and save it as report.md",
        transport=FakeTransport([]),
        registry=registry, hook=hook, plane=plane,
        skip_ambiguity=True,
        shape_config=_DepthOnlyConfig(6),
    ))
    assert res.final_response == "CHAINED"
    assert captured["research_depth"] == 6  # the shapes/specs depth reached run_plan_chain


# --------------------------------------------------------------------------- #
# (3) the store round-trips depth durably + rejects nonsensical values
# --------------------------------------------------------------------------- #
def test_s13_shape_config_store_depth_round_trip(tmp_path):
    """ShapeConfigStore persists/reads the depth override (durable, disjoint from
    max_iter) and rejects depth < 1; delete clears both override rows."""
    store = ShapeConfigStore(tmp_path / "data")
    assert store.get_depth("plan-chain") is None  # no override → None (env baseline)

    store.set_depth("plan-chain", 8)
    store.set_max_iter("plan-chain", 12)  # disjoint: a depth row coexists with max_iter
    assert store.get_depth("plan-chain") == 8
    assert store.get_max_iter("plan-chain") == 12
    assert store.all_depths() == {"plan-chain": 8}

    with pytest.raises(ValueError):
        store.set_depth("plan-chain", 0)

    # A fresh connection to the same db reads exactly what was written (durable).
    store.close()
    store2 = ShapeConfigStore(tmp_path / "data")
    assert store2.get_depth("plan-chain") == 8

    store2.delete("plan-chain")  # clears BOTH override rows, no orphan
    assert store2.get_depth("plan-chain") is None
    assert store2.get_max_iter("plan-chain") is None
    store2.close()


# --------------------------------------------------------------------------- #
# (4) AUDIT (P2-5c): the GENERIC research engine is the flag-free DEFAULT on run_plan_chain
# --------------------------------------------------------------------------- #
def test_s13_no_scaffolding_flag_gates_research_engine_on_default_route():
    """P2-5c audit: run_plan_chain drives the GENERIC research engine UNCONDITIONALLY (no
    enable/scaffolding flag gates it on the served report route), and depth is clamped to
    the proven [1, N4_TREE_DEPTH_CEILING] bound."""
    src = inspect.getsource(run_plan_chain)

    # The GENERIC research engine IS reached on this route (called, not behind a flag);
    # the bespoke tree runner is retired (not referenced).
    assert "_run_generic_research_phase(" in src
    assert "run_research_tree(" not in src

    # NO enable/scaffolding flag gates the research engine: no env-flag named
    # *ENABLE*/*SCAFFOLD*/*DISABLE* and no `if <…enable…>:` guard around the research call.
    assert not re.search(r"getenv\([^)]*(ENABLE|SCAFFOLD|DISABLE)", src, re.I), \
        "an enable/scaffold env flag must NOT gate the research engine on the default route"
    assert not re.search(r"\bif\b[^\n]*\b(enable|scaffold)\w*\b[^\n]*:", src, re.I), \
        "the research engine must not sit behind an enable/scaffold conditional"

    # The depth bound the served route clamps to is the hard ceiling the user fixed.
    assert "N4_TREE_DEPTH_CEILING" in src
    # The TreeConfig regime that backs the report route is the proven 32768/4096 contract.
    cfg = TreeConfig()
    assert cfg.num_ctx == 32768 and cfg.num_predict == 4096
