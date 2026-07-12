"""d221/d189 (as1 FORK2) — the RESEARCH-REVIEWER WRITE-PLANNING EVENT + EVENTS-AS-USER-MESSAGE.

The research plan's last-step reviewer hands the write planner a write-planning GROUNDING event
carrying (1) what the research found, (2) the research-memory handle, (3) the data-complexity
read (d237); that event reaches the WRITE planner as part of its role:'user' goal (d189).
SB-6/d299: the engine OUTLINE-FIRST sectioning DIRECTIVE was RETIRED from this event — that
methodology lives in the section-html-writer spec body now, and the planner reasons sectioning
from the data-complexity + the shape + the spec description. as1 retired the faked ``_research_plan_final_status``
pure-function (which ALSO faked the follow-up DECISION) — the follow-up decision is now the
planner's real ``decide_followup`` reasoning, and the GROUNDING DATA is assembled by
``_build_write_planning_event``. These tests cover the event assembly, its rendering, and that
the composed write goal carries it.
"""
from __future__ import annotations

from chat_app.agentic import (
    _build_write_planning_event,
    _render_write_planning_event,
    _compose_write_goal,
)


def test_write_planning_event_carries_the_handoffs():
    event = _build_write_planning_event(
        "casualties and damage figures were reported on both sides",
        [{"url": "u1"}, {"url": "u2"}],
        memory_handle="research_abc", data_complexity="3 concerns; moderate",
    )
    assert event["kind"] == "write_plan"
    # SB-6/d299: the engine WRITE-METHODOLOGY strings (write_directive / output_desired) are RETIRED
    # from the event — that methodology lives in the section-html-writer SPEC BODY now; only grounding
    # DATA remains, and the planner reasons sectioning from it (data_complexity) + the shape + spec.
    assert "write_directive" not in event
    assert "output_desired" not in event
    assert event["memory_handle"] == "research_abc"
    assert "casualties and damage" in event["findings_digest"]
    assert event["sources"] == 2
    assert event["data_complexity"] == "3 concerns; moderate"


def test_render_write_planning_event_block():
    block = _render_write_planning_event({
        "kind": "write_plan",
        "memory_handle": "research_abc",
        "findings_digest": "the conflict escalated in June",
        "data_complexity": "8 concerns; multi-faceted",
    })
    assert "WRITE-PLANNING EVENT" in block
    # SB-6/d299: the engine "OUTLINE/LEAD node FIRST" topology directive is RETIRED; sectioning is
    # reasoned by the planner from the DATA-COMPLEXITY (the legitimate data lever), not a stamp.
    assert "OUTLINE/LEAD node FIRST" not in block
    assert "DATA COMPLEXITY" in block
    assert "research_abc" in block
    assert "the conflict escalated in June" in block
    # No event → empty (byte-identical pre-d221 goal).
    assert _render_write_planning_event(None) == ""
    assert _render_write_planning_event({}) == ""


def test_compose_write_goal_carries_event_data():
    event = {
        "kind": "write_plan",
        "memory_handle": "research_abc",
        "findings_digest": "key figures and timeline",
        "data_complexity": "several concerns; multi-part",
    }
    goal = _compose_write_goal(
        "Write a report on the US-Iran conflict.", "report.html",
        "findings text", "SOURCES: [S1] ...",
        write_planning_event=event,
    )
    # (b) the reviewer's event reaches the planner as part of its user goal.
    assert "WRITE-PLANNING EVENT" in goal
    assert "research_abc" in goal
    # SB-6/d299: the EVENT's grounding DATA (data-complexity) reaches the planner goal so the planner
    # reasons sectioning from it. The engine WRITE-METHODOLOGY directive that used to ride the
    # write-planning EVENT (_render's "write sectioned, OUTLINE FIRST" block) is RETIRED. (The
    # planner-authored "OUTLINE/LEAD node FIRST + chain + source_ids" decomposition guidance that
    # lives in _compose_write_goal itself is KEPT — that is planner-authored TOPOLOGY, gate-approved.)
    assert "DATA COMPLEXITY" in goal
    assert "several concerns; multi-part" in goal


def test_compose_write_goal_without_event_is_unchanged_shape():
    """No event → no WRITE-PLANNING EVENT block (back-compat for non-report callers)."""
    goal = _compose_write_goal(
        "Write a report.", "report.html", "findings", "",
    )
    assert "WRITE-PLANNING EVENT" not in goal
