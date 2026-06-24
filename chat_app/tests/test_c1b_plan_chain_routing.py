"""s9/c1b — run_agentic ROUTES a multi-page file request to plan-chaining.

Locks the trigger contract (d49.4) WITHOUT live inference: the model-extracted
``multi_page`` + ``wants_file`` signals route ``run_agentic`` to ``run_plan_chain``
(plan1 research → plan2 write-file shape), while EVERY other shape — including a
single-file write (multi_page False) — falls through UNCHANGED, so c1 single-file
reliability cannot regress. The selector is stubbed to assert the routing branch
deterministically; ``run_plan_chain`` itself is proven live on E4B by the smoke.
"""
from __future__ import annotations

import asyncio

import chat_app.agentic as agentic
from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.shape_selector import ShapeSelection
from chat_app.agentic import AgenticResult, _normalize_write_dag, run_agentic


def test_normalize_write_dag_enforces_single_file_linear_writers():
    """plan2's LLM-authored sections (mixed: a synthesis role, a bare node, a
    file_write) are normalized to a LINEAR chain of file_write writers all targeting
    the SAME file — the write-file shape's single-file append discipline (c1b)."""
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
    # every node is a RAW-loop file_write writer (role=None → no schema/JSON path)
    assert all(n.tool == "file_write" and n.role is None for n in nd.nodes)
    # all target the SAME file, so they accumulate into one deliverable
    assert all("solar-report.md" in n.task for n in nd.nodes)
    # chained linearly so the runtime continuation appends in order (one file)
    assert nd.nodes[0].depends_on == ()
    assert nd.nodes[1].depends_on == ("n1",)
    assert nd.nodes[2].depends_on == ("n2",)
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
    """Run run_agentic with the selector stubbed to ``selection``; capture whether
    run_plan_chain fired and return (chain_fired, result)."""
    registry, hook, plane = _wiring(tmp_path)

    async def fake_select(self, goal):
        return selection

    monkeypatch.setattr(agentic.ShapeSelector, "select", fake_select)

    fired = {"chain": False}

    async def fake_chain(query, sel, **kw):
        fired["chain"] = True
        return AgenticResult(rationale="chained", ok=True, final_response="CHAINED")

    async def fake_acyclic(query, shape_spec, sel, **kw):
        return AgenticResult(rationale="acyclic", ok=True, final_response="ACYCLIC")

    monkeypatch.setattr(agentic, "run_plan_chain", fake_chain)
    monkeypatch.setattr(agentic, "_run_acyclic", fake_acyclic)

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
    return fired["chain"], res


def test_multipage_file_request_routes_to_plan_chain(monkeypatch, tmp_path):
    sel = ShapeSelection(shape="linear", escalate=False, wants_file=True,
                         multi_page=True, search_allowed=True)
    chain_fired, res = _drive(monkeypatch, tmp_path, sel)
    assert chain_fired is True
    assert res.final_response == "CHAINED"


def test_single_file_request_does_not_chain(monkeypatch, tmp_path):
    # wants_file but NOT multi_page → the single-file acyclic path, never the chain.
    sel = ShapeSelection(shape="linear", escalate=False, wants_file=True,
                         multi_page=False, search_allowed=True)
    chain_fired, res = _drive(monkeypatch, tmp_path, sel)
    assert chain_fired is False
    assert res.final_response == "ACYCLIC"


def test_multipage_but_no_file_does_not_chain(monkeypatch, tmp_path):
    # multi_page but a chat-only answer (no file) → not the file-writing chain.
    sel = ShapeSelection(shape="linear", escalate=False, wants_file=False,
                         multi_page=True, search_allowed=True)
    chain_fired, res = _drive(monkeypatch, tmp_path, sel)
    assert chain_fired is False


def test_multipage_with_unmet_spec_defers_to_acyclic(monkeypatch, tmp_path):
    # a missing specialist must reach the acyclic missing-spec pause, not the chain.
    sel = ShapeSelection(shape="linear", escalate=False, wants_file=True,
                         multi_page=True, search_allowed=True,
                         unmet_specs=["forensic-accountant"])
    chain_fired, res = _drive(monkeypatch, tmp_path, sel)
    assert chain_fired is False
