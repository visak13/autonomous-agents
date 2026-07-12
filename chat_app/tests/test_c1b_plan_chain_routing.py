"""s15 ROUTING PURITY (d148/d151/d161, as1 FORK4/d239) — run_agentic SEEDS THE ONE LOOP.

Locks the routing contract WITHOUT live inference. as1 retired the bespoke agentic.py:691
fork: there is no longer a SEPARATE report engine — EVERY shape routes through the single
``_run_generic_loop``; run_agentic only chooses which PLAN the loop is SEEDED with. The
deep-research FAMILY (keyed on ``execution == "deep-research"`` → ``is_deep_research``) seeds a
RESEARCH-first plan (``first_plan_kind="research"``), purely on the LLM-selected shape — the
retired ``wants_file`` / ``multi_page`` booleans no longer gate anything. An acyclic shape
(linear / modular-parallel) seeds an ACYCLIC plan, even when ``wants_file`` + ``multi_page``
are both True (proving those signals are no longer a routing gate). The only conditions that
divert a deep-research shape to the acyclic seed are the JUSTIFIED capability/hatch guards — a
no-web constraint and a missing specialist. The selector is stubbed to assert the seed branch
deterministically (the real on-disk catalog still resolves the shape); the loop itself is
proven live on E4B by the smoke.
"""
from __future__ import annotations

import asyncio

import chat_app.agentic as agentic
from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.shape_selector import ShapeSelection
from chat_app.agentic import AgenticResult, _normalize_write_dag, run_agentic


def test_normalize_write_dag_minimal_delivery_preserves_planner_topology():
    """SB-6/d299: _normalize_write_dag is now MINIMAL DELIVERY only. It adds the single-file
    out_name hint to each section's task and PRESERVES the planner's authored topology (tool, role,
    depends_on) VERBATIM. It NO LONGER stamps tool=file_write / role=None and NO LONGER re-chains the
    nodes — that engine structure-authoring (the SA-6 PART-2 fabrication) was retired: the PLANNER
    authors the section topology + chain, the runtime routes by delivery-context, and the shape's
    SEQUENTIAL execution appends the sections to one file."""
    dag = PlanDAG(
        nodes=[
            PlanNode(id="n1", task="Write the Introduction section", role="synthesizer"),
            PlanNode(id="n2", task="Write the Photovoltaic Effect section"),
            PlanNode(id="n3", task="Write the Sources section", tool="file_write"),
        ],
        goal="g",
    )
    nd = _normalize_write_dag(dag, "solar-report.md")
    assert [n.id for n in nd.nodes] == ["n1", "n2", "n3"]
    # the planner's authored tool/role are PRESERVED — the engine stamps NOTHING.
    assert nd.nodes[0].role == "synthesizer"
    assert (nd.nodes[1].tool or "") == "" and nd.nodes[1].role is None
    assert nd.nodes[2].tool == "file_write"
    # the ONLY thing normalize adds is the single-file DELIVERY hint — every section names out_name.
    assert all("solar-report.md" in n.task for n in nd.nodes)
    # NO engine re-chain — the planner's depends_on is preserved verbatim (here, none authored).
    assert all(n.depends_on == () for n in nd.nodes)
from llm_framework import FakeTransport
from reactive_tools import EventPlane, build_default_hook, register_agentic_tools
from specialization.registry import SpecRegistry


def _wiring(tmp_path):
    plane = EventPlane()
    hook = build_default_hook(plane)
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    registry = SpecRegistry(tmp_path / "specs")
    return registry, hook, plane


def _drive(monkeypatch, tmp_path, selection: ShapeSelection):
    """Run run_agentic with the selector stubbed to ``selection``; capture the SEED
    (``first_plan_kind``) run_agentic handed the ONE generic loop, and return (seed, result).

    as1 (FORK4/d239): the bespoke fork is retired — run_agentic no longer routes to two
    separate engines; it SEEDS ``_run_generic_loop``. The routing-purity contract is now
    'the right shape picks the right SEED', so we intercept the single loop and read the seed."""
    registry, hook, plane = _wiring(tmp_path)

    async def fake_select(self, goal):
        return selection

    monkeypatch.setattr(agentic.ShapeSelector, "select", fake_select)

    captured = {"seed": None}

    async def fake_loop(query, sel, *, first_plan_kind, **kw):
        captured["seed"] = first_plan_kind
        return AgenticResult(
            rationale=first_plan_kind, ok=True, final_response=first_plan_kind.upper()
        )

    monkeypatch.setattr(agentic, "_run_generic_loop", fake_loop)

    res = asyncio.run(
        run_agentic(
            "write me a big multi-page report and save it as report.md",
            transport=FakeTransport([]),
            registry=registry,
            hook=hook,
            plane=plane,
            skip_ambiguity=True,
        )
    )
    return captured["seed"], res


def test_deep_research_shape_seeds_research_plan(monkeypatch, tmp_path):
    # the LLM-selected deep-research (is_deep_research) shape → the RESEARCH seed of the one
    # loop, by SHAPE alone (wants_file/multi_page here are False — they no longer gate).
    sel = ShapeSelection(shape="deep-research", escalate=False, wants_file=False,
                         multi_page=False, search_allowed=True)
    seed, res = _drive(monkeypatch, tmp_path, sel)
    assert seed == "research"
    assert res.final_response == "RESEARCH"


def test_linear_shape_seeds_acyclic_even_with_file_signals(monkeypatch, tmp_path):
    # an acyclic shape seeds the ACYCLIC plan — even with wants_file AND multi_page both True,
    # proving the retired booleans are no longer a routing gate (s15).
    sel = ShapeSelection(shape="linear", escalate=False, wants_file=True,
                         multi_page=True, search_allowed=True)
    seed, res = _drive(monkeypatch, tmp_path, sel)
    assert seed == "acyclic"
    assert res.final_response == "ACYCLIC"


def test_deep_research_no_web_defers_to_acyclic(monkeypatch, tmp_path):
    # a no-web constraint (capability guard) diverts the deep-research shape — an
    # inherently web shape — to the acyclic seed with web tools stripped.
    sel = ShapeSelection(shape="deep-research", escalate=False, wants_file=False,
                         multi_page=False, search_allowed=False)
    seed, res = _drive(monkeypatch, tmp_path, sel)
    assert seed == "acyclic"


def test_deep_research_with_unmet_spec_defers_to_acyclic(monkeypatch, tmp_path):
    # a missing specialist must reach the acyclic missing-spec pause, not the research seed.
    sel = ShapeSelection(shape="deep-research", escalate=False, wants_file=False,
                         multi_page=False, search_allowed=True,
                         unmet_specs=["forensic-accountant"])
    seed, res = _drive(monkeypatch, tmp_path, sel)
    assert seed == "acyclic"
