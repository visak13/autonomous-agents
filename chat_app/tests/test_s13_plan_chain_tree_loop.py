"""s13/B1 (P2-5c FLAG-FREE END-STATE) — the REPORT path (run_plan_chain) drives the
GENERIC declarative-unroll research engine in PHASE-1.

P2-5c retired the bespoke ``run_research_tree`` loop + ``_make_tree_gather`` leaf: BOTH
``run_plan_chain`` and the sibling ``_run_deep_research_sectioned`` now ALWAYS run PHASE-1
research through :func:`_run_generic_research_phase` (the generic declarative-unroll +
AgentRuntime growable engine). There is no tree fallback and no flag branch — the reported
engine is always ``"generic-unroll"``.

This FAST integration test drives the real ``run_plan_chain`` with the generic PHASE-1
STUBBED (a fake returning a known ``(findings, sources, grow_trace)``) + a write-phase SPY,
and asserts:

* PHASE-1 runs through the GENERIC engine (the served result reports engine
  ``"generic-unroll"``), with the resolved deep-research shape + completeness_stop handed to
  the generic phase;
* the accumulated ``(findings, sources)`` FLOW INTO :func:`run_section_write_phase`
  (PHASE-2 feeding unchanged, d89);
* the per-leaf fetch BREADTH stays PINNED to 3 on this path (D97, via the served-route trace).

PHASE-1 + PHASE-2 are both stubbed (a fake + a spy) so the test stays fast + isolated to the
served-route chaining + the hand-off — it asserts the WIRING, not the engines' own behaviour.
"""
from __future__ import annotations

import asyncio

from llm_framework import ChatResult
from reactive_tools import EventPlane, ToolHook
from reactive_tools.tool_hook import ToolRegistry
from specialization import SpecRegistry

import chat_app.agentic as agentic
from chat_app.agentic import run_plan_chain, PLAN_CHAIN_TREE_BREADTH
from agent_runtime import ShapeSelection, ShapeSpec
from agent_runtime.factory import PlanDAG


_SRC = {"title": "Iran 2025", "url": "https://news.example.com/iran-2025",
        "markdown": "The conflict escalated June 13; 1,200 casualties by June 20."}


def _deep_research_shape() -> ShapeSpec:
    """A small unrollable deep-research ShapeSpec carrying the P2.4 completeness_stop."""
    return ShapeSpec(
        name="deep-research",
        description="deep research",
        max_iter=2,
        hard_cap=4,
        execution="deep-research",
        round_roles=["research", "critic"],
        final_roles=["research", "synthesis", "verify"],
        completeness_stop="Fill ALL the blanks across timeline/costs/impact before stopping.",
    )


class _ProseTransport:
    """A scripted transport whose every turn returns prose (no tool call).

    Enough for the wiring tests where the real research/decision turns are stubbed out;
    keeps the loops short and deterministic."""

    def complete(self, messages, **opts) -> str:  # pragma: no cover - parity shim
        return self.chat(messages, **opts).content

    def chat(self, messages, **opts) -> ChatResult:
        return ChatResult(role="assistant", content="PLAN: gather then synthesize.")


class _FakeWriteResult:
    """A minimal RuntimeResult stand-in satisfying _agentic_from_runtime."""

    def __init__(self) -> None:
        self.results: dict = {}
        self.states: dict = {}
        self.launch_order: list = []
        self.ok = True


def _install_fakes(monkeypatch):
    """Stub the GENERIC research PHASE-1 + a write-phase spy.

    Returns ``(generic_calls, write_calls)``: ``generic_calls`` records the kwargs the
    served route handed the generic phase (so we can assert the shape + stop wiring);
    ``write_calls`` records the (findings, sources) hand-off into PHASE-2."""
    generic_calls: list[dict] = []
    write_calls: list[dict] = []

    async def fake_generic(query, **kw):
        generic_calls.append(dict(kw))
        return (
            "FINDINGS for root: escalation + casualties.",
            [dict(_SRC)],
            {"growable": True, "stop_reason": "agent_sufficient", "grow_layers": 2,
             "max_layers": 4, "layers": [{"gathered": 1}, {"gathered": 1}]},
        )

    async def spy_write_phase(query, out_name, findings, sources, **_kw):
        write_calls.append({"query": query, "out_name": out_name,
                            "findings": findings, "sources": sources})
        return PlanDAG(nodes=[], goal=query), _FakeWriteResult()

    monkeypatch.setattr(agentic, "_run_generic_research_phase", fake_generic)
    monkeypatch.setattr(agentic, "run_section_write_phase", spy_write_phase)
    return generic_calls, write_calls


def test_s13_run_plan_chain_runs_generic_engine_and_feeds_write_phase(monkeypatch, tmp_path):
    """run_plan_chain PHASE-1 runs the GENERIC engine (engine == 'generic-unroll'), and the
    accumulated (findings, sources) flow into run_section_write_phase (d89)."""
    generic_calls, write_calls = _install_fakes(monkeypatch)

    hook = ToolHook(EventPlane(), registry=ToolRegistry())
    result = asyncio.run(run_plan_chain(
        "detailed HTML report on the 2025 US-Iran war",
        ShapeSelection(shape="plan-chain", escalate=False, rationale="large report"),
        transport=_ProseTransport(),
        registry=SpecRegistry(str(tmp_path)),
        hook=hook,
        plane=hook.plane,
        timeout=30.0,
        run_id="s13-test",
        overall_goal="detailed HTML report on the 2025 US-Iran war",
        research_depth=2,
        completeness_stop="Fill ALL the blanks across timeline/costs/impact before stopping.",
        catalog={"deep-research": _deep_research_shape()},
    ))

    # PHASE-1 ran through the GENERIC engine (no tree, no flag).
    dr = result.deep_research
    assert dr is not None and dr.get("plan_chain") is True
    assert dr["engine"] == "generic-unroll"
    # The served route handed the generic phase the resolved deep-research shape + the
    # shape's completeness_stop (the same "fill all the blanks" stop signal).
    assert len(generic_calls) == 1
    g = generic_calls[0]
    assert getattr(g["dr_shape"], "name", None) == "deep-research"
    assert "Fill ALL the blanks" in (g.get("completeness_stop") or "")

    # D97: breadth stays PINNED to 3 on this path (the served-route trace value).
    assert dr["leaf_breadth"] == PLAN_CHAIN_TREE_BREADTH == 3

    # (findings, sources) FLOWED into PHASE-2 (run_section_write_phase) — d89 hand-off.
    assert len(write_calls) == 1
    fed = write_calls[0]
    assert "FINDINGS for root" in fed["findings"]  # the generic engine's accumulated findings
    assert fed["sources"] and fed["sources"][0]["url"] == _SRC["url"]  # the deduped sources


def test_s13_breadth_pinned_3_ignores_n1_breadth_env(monkeypatch, tmp_path):
    """D97: the report path's per-leaf fetch breadth is FIXED at 3 even when the
    deep-research N1 breadth knob (RA_RESEARCH_FETCH_BREADTH) is set high — breadth is a
    fixed contract on this served route, not the shared knob."""
    monkeypatch.setenv("RA_RESEARCH_FETCH_BREADTH", "10")  # N1 knob high — must NOT leak
    _generic_calls, write_calls = _install_fakes(monkeypatch)

    hook = ToolHook(EventPlane(), registry=ToolRegistry())
    result = asyncio.run(run_plan_chain(
        "detailed report",
        ShapeSelection(shape="plan-chain", escalate=False, rationale="r"),
        transport=_ProseTransport(),
        registry=SpecRegistry(str(tmp_path)),
        hook=hook, plane=hook.plane, timeout=30.0, run_id="s13-breadth",
        overall_goal="detailed report",
        catalog={"deep-research": _deep_research_shape()},
    ))

    assert result.deep_research["leaf_breadth"] == PLAN_CHAIN_TREE_BREADTH == 3
    assert len(write_calls) == 1  # still completed end-to-end
