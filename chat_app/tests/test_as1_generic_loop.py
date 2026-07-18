"""as1 (S1, d214/d215/d239) — the ONE generic ITERATIVE PLANNER LOOP, real stages.

These tests lock the engine-unify contract WITHOUT live inference: ``_run_generic_loop`` drives
EVERY shape through one loop seeded by ``first_plan_kind``, and the three retired FAKED seams are
now REAL:

  * the research plan's last-step REVIEWER is a real ``Planner.review_research`` LLM call (not
    the retired ``_research_plan_final_status`` pure-fn that always hardcoded write-plan);
  * the follow-up is the planner's real ``decide_followup`` reasoning (not the hardcoded
    research->write->done while-loop) — the loop READS its decision;
  * the terminal synthesizer summary is the planner's real ``finalize_summary`` LLM digest (not
    a fixed string).

The research PHASE-1 + the write PHASE-2 are stubbed (fakes) so the test stays fast and isolated
to the LOOP control flow + the real reviewer/decision/synthesizer wiring. A scripted reasoning
transport returns the loop's structured decisions so we prove the loop READS them (research ->
write -> done), and a separate run proves the ACYCLIC seed folds into the same loop (one plan ->
done -> synthesizer).
"""
from __future__ import annotations

import asyncio
import json

from llm_framework import ChatResult
from reactive_tools import EventPlane, ToolHook
from reactive_tools.tool_hook import ToolRegistry
from specialization import SpecRegistry

import chat_app.agentic as agentic
from chat_app.agentic import _run_generic_loop, EVENT_RUN_SYNTHESIS
from agent_runtime import ShapeSelection, ShapeSpec
from agent_runtime.factory import PlanDAG


_SRC = {"title": "Iran 2025", "url": "https://news.example.com/iran-2025",
        "markdown": "The conflict escalated June 13; 1,200 casualties by June 20."}


def _deep_research_shape() -> ShapeSpec:
    return ShapeSpec(
        name="deep-research", description="deep research", max_iter=2, hard_cap=4,
        execution="deep-research",
        completeness_stop="Fill ALL the blanks before stopping.",
    )


class _ScriptedReasoning:
    """A transport that returns the loop's structured reasoning by inspecting the prompt.

    review_research → research_complete; decide_followup after a research plan → write_plan,
    after a write plan → done; finalize_summary → prose. This proves the loop READS the real
    reviewer status + the real follow-up decision (not a hardcoded rule)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat(self, messages, **opts) -> ChatResult:
        sys = " ".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "system"
        )
        usr = " ".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "user"
        )
        if "briefing its REVIEW node" in sys:
            # P6: the engine review_research method is retired — the planner AUTHORS
            # the review node's brief, and the node's prose is the review signal.
            self.calls.append("author_review_brief")
            return ChatResult(role="assistant", content=(
                "Review the gathered research against the goal; report coverage, "
                "gaps and the data's shape honestly."))
        if "ROLE: REVIEWER" in sys:
            # the P6 review NODE's model-authored prose — the single review signal.
            self.calls.append("review_node_turn")
            return ChatResult(role="assistant", content=(
                "The research covers 5 facets; complex data. Coverage is sufficient "
                "to support the deliverable; nothing material is missing."))
        if "LAST-STEP REVIEWER" in sys and "research_complete" in sys:
            self.calls.append("review_research")
            return ChatResult(role="assistant", content=json.dumps(
                {"status": "research_complete", "data_complexity": "5 facets; complex",
                 "rationale": "covered"}))
        if "Decide the NEXT step" in sys:
            self.calls.append("decide_followup")
            nxt = "write_plan" if "LAST PLAN: research" in usr else "done"
            return ChatResult(role="assistant", content=json.dumps(
                {"next_plan": nxt, "rationale": "reasoned"}))
        if "BRIEF" in sys and "summary" in sys:
            self.calls.append("finalize_summary")
            return ChatResult(role="assistant",
                              content="Your report on the 2025 conflict is ready.")
        # any other structured call (e.g. authoring) — harmless prose
        return ChatResult(role="assistant", content="PLAN: proceed.")

    def complete(self, messages, **opts) -> str:  # pragma: no cover - parity shim
        return self.chat(messages, **opts).content


class _FakeNodeResult:
    """A minimal node result satisfying _agentic_from_runtime (parsed/output/error)."""

    def __init__(self, output: str = "") -> None:
        self.parsed = None
        self.output = output
        self.error = None
        self.tool_used = ""
        self.tool_value = None


class _FakeWriteResult:
    def __init__(self) -> None:
        self.results = {
            "sec1": _FakeNodeResult("Section one body."),
            "final_review": _FakeNodeResult("Reviewed."),
        }
        self.states = {
            "sec1": {"status": "done", "attempts": 1, "error": None},
            "final_review": {"status": "done", "attempts": 1, "error": None},
        }
        self.launch_order = ["sec1", "final_review"]
        self.ok = True


class _RecordingPlane(EventPlane):
    def __init__(self) -> None:
        super().__init__()
        self.kinds: list[str] = []

    async def publish(self, kind, payload, source=None):
        self.kinds.append(kind)
        return await super().publish(kind, payload, source=source)


def _install_report_fakes(monkeypatch):
    calls = {"research": 0, "write": 0}

    async def fake_generic(query, **kw):
        calls["research"] += 1
        return (
            "FINDINGS: escalation + casualties.",
            [dict(_SRC)],
            {"growable": True, "stop_reason": "agent_sufficient", "grow_layers": 2,
             "max_layers": 10, "layers": [{"gathered": 1}, {"gathered": 1}],
             "memory_handle": "research_xyz"},
        )

    async def fake_write(query, out_name, findings, sources, **kw):
        calls["write"] += 1
        # capture the write-planning event so we prove the reviewer handoff reached the writer
        calls["write_event"] = kw.get("write_planning_event")
        return PlanDAG(nodes=[], goal=query), _FakeWriteResult()

    async def fake_review_node(goal, **kw):
        # P6: the planner-briefed review NODE runs on the real runtime in production;
        # this offline harness (empty registry/hook) fakes its model-authored prose,
        # like the other phases. The prose IS the review signal (SB-5).
        calls["review_node"] = calls.get("review_node", 0) + 1
        return ("The research covers 5 facets; complex data. Coverage is sufficient "
                "to support the deliverable; nothing material is missing.")

    monkeypatch.setattr(agentic, "_run_generic_research_phase", fake_generic)
    monkeypatch.setattr(agentic, "run_section_write_phase", fake_write)
    monkeypatch.setattr(agentic, "_run_research_review_node", fake_review_node)
    return calls


def test_research_seed_drives_research_then_write_then_done_with_real_stages(monkeypatch, tmp_path):
    """The research seed: research -> real review_research -> real decide_followup(write_plan) ->
    write -> read final_review status -> real decide_followup(done) -> terminal synthesizer."""
    calls = _install_report_fakes(monkeypatch)
    transport = _ScriptedReasoning()
    plane = _RecordingPlane()
    hook = ToolHook(plane, registry=ToolRegistry())

    result = asyncio.run(_run_generic_loop(
        "detailed HTML report on the 2025 US-Iran war",
        ShapeSelection(shape="deep-research", escalate=False, rationale="report"),
        first_plan_kind="research",
        transport=transport, registry=SpecRegistry(str(tmp_path)),
        hook=hook, plane=plane, timeout=30.0, run_id="as1-loop",
        overall_goal="detailed HTML report on the 2025 US-Iran war",
        research_depth=2, completeness_stop="Fill ALL the blanks before stopping.",
        catalog={"deep-research": _deep_research_shape()},
    ))

    # research + write each ran exactly once through the ONE loop
    assert calls["research"] == 1 and calls["write"] == 1
    # the loop authored research THEN write (the planner reasoned the follow-up)
    assert result.deep_research["plans_authored"] == ["research", "write"]
    assert result.deep_research["engine"] == "generic-unroll"
    # the REAL reasoning stages fired (not the retired pure-fns / hardcoded loop)
    assert calls.get("review_node") == 1  # P6: the briefed review node ran once
    assert transport.calls.count("decide_followup") == 2   # after research, after write
    assert transport.calls.count("finalize_summary") == 1  # the LLM terminal summary
    # the research reviewer's data-complexity handoff reached the write planner (d237)
    assert "5 facets; complex" in calls["write_event"]["data_complexity"]  # P6: review prose
    assert calls["write_event"]["memory_handle"] == "research_xyz"
    # the terminal synthesizer streamed + announced (its LLM summary is on the result)
    assert EVENT_RUN_SYNTHESIS in plane.kinds
    assert "ready" in (result.synthesis_summary or "").lower()
    # S5 trace: the model's stop is primary (stop_reason agent_sufficient, not depth_bound)
    assert result.deep_research["stop_reason"] == "agent_sufficient"


def test_acyclic_seed_folds_into_one_loop_single_plan_then_done(monkeypatch, tmp_path):
    """The acyclic seed folds into the SAME loop: one authored plan -> real decide_followup(done)
    -> terminal synthesizer (FORK4 — no separate fast path)."""
    drove = {"n": 0}

    async def fake_acyclic(query, shape_spec, selection, **kw):
        drove["n"] += 1
        dag = PlanDAG(nodes=[], goal=query)
        return None, dag, _FakeWriteResult()

    monkeypatch.setattr(agentic, "_author_and_drive_acyclic_plan", fake_acyclic)
    transport = _ScriptedReasoning()
    plane = _RecordingPlane()
    hook = ToolHook(plane, registry=ToolRegistry())

    result = asyncio.run(_run_generic_loop(
        "what is 2+2?",
        ShapeSelection(shape="linear", escalate=False, rationale="quick"),
        first_plan_kind="acyclic",
        transport=transport, registry=SpecRegistry(str(tmp_path)),
        hook=hook, plane=plane, timeout=30.0, run_id="as1-acyclic",
        overall_goal="what is 2+2?",
    ))

    assert drove["n"] == 1                      # one acyclic plan, then the loop exits
    assert result.ok is True
    # the acyclic plan went through the SAME loop → its follow-up was reasoned to 'done'
    assert transport.calls.count("decide_followup") == 1
    # the loop still ends in the terminal synthesizer (a chat answer carries NO artifact)
    assert EVENT_RUN_SYNTHESIS in plane.kinds
    assert result.synthesis_summary
