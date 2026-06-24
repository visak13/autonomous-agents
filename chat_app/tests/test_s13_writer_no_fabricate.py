"""s13 / FX-writer (d106 #6, #7) — write-path EMPTY-NODE-NO-FABRICATE + OUTLINE-AS-PRIMARY.

The B8a live run exposed two writer-side defects this file pins on the chat_app write path:

  #6  EMPTY-NODE-NO-FABRICATE (d60-critical): a research node that fetched 0 sources still
      got a section, which the writer fabricated from memory (the Timeline). After
      source-scoping + coverage, such a section carries NO source_ids; ``_flag_unsupported_
      sections`` rewrites its task to the deterministic UNSUPPORTED instruction so the writer
      marks it instead of inventing content. The scoped guard means short reports and the
      d56 empty-outline fallback are untouched.

  #7  OUTLINE-AS-PRIMARY: the agent outline must be the PRIMARY, COMPLETE section list, not
      a second parallel set appended after the conclusion. ``_render_outline_hint`` carries
      that instruction; an EMPTY outline yields "" so the write phase falls back to the
      findings-driven decomposition (d56), and ``_compose_write_goal`` always carries the
      anti-fabrication clause.
"""
from __future__ import annotations

from agent_runtime.factory import PlanDAG, PlanNode

from chat_app.agentic import (
    _compose_write_goal,
    _flag_unsupported_sections,
    _render_outline_hint,
)


_OUTLINE = [
    {"title": "Cost and Damage Assessment", "covers": "B2"},
    {"title": "Timeline of Key Events", "covers": "B1"},
]


# --------------------------------------------------------------------------- #
# #6 — EMPTY-NODE-NO-FABRICATE: a no-source section is flagged UNSUPPORTED.
# --------------------------------------------------------------------------- #
def test_no_source_section_is_flagged_unsupported_not_fabricated():
    """A scoped DAG with one section left without source_ids → that section is flagged
    UNSUPPORTED + told not to fabricate; the sourced sibling section is untouched."""
    dag = PlanDAG(nodes=[
        PlanNode(id="s1", task="Write the Cost section.", source_ids=(1, 2)),
        PlanNode(id="s2", task="Write the Timeline section.", source_ids=()),  # node fetched 0
    ])
    out = _flag_unsupported_sections(dag)
    cost = next(n for n in out.nodes if n.id == "s1")
    timeline = next(n for n in out.nodes if n.id == "s2")
    # the grounded section is byte-identical
    assert cost.task == "Write the Cost section."
    assert cost.source_ids == (1, 2)
    # the empty-source section is flagged UNSUPPORTED + anti-fabrication
    assert "UNSUPPORTED" in timeline.task
    assert "Write the Timeline section." in timeline.task
    low = timeline.task.lower()
    assert "do not invent" in low
    assert "no supporting sources" in low


def test_unscoped_path_is_noop():
    """The unscoped / no-source-report path (no section carries source_ids) is byte-identical
    — short reports + the d56 empty-outline fallback are never flagged."""
    dag = PlanDAG(nodes=[
        PlanNode(id="s1", task="Section one."),
        PlanNode(id="s2", task="Section two."),
    ])
    out = _flag_unsupported_sections(dag)
    assert [n.task for n in out.nodes] == ["Section one.", "Section two."]


def test_all_sections_scoped_is_noop():
    """When every section already has its own sources, nothing is flagged."""
    dag = PlanDAG(nodes=[
        PlanNode(id="s1", task="A.", source_ids=(1,)),
        PlanNode(id="s2", task="B.", source_ids=(2,)),
    ])
    out = _flag_unsupported_sections(dag)
    assert all("UNSUPPORTED" not in n.task for n in out.nodes)


# --------------------------------------------------------------------------- #
# #7 — OUTLINE-AS-PRIMARY: the outline hint instructs a single complete scaffold.
# --------------------------------------------------------------------------- #
def test_outline_hint_is_primary_complete_list_no_parallel_set():
    """The outline clause must state it is the COMPLETE / PRIMARY section list and forbid a
    second parallel findings-driven set + an appended tail (the B8a duplicate-tail cause)."""
    hint = _render_outline_hint(_OUTLINE)
    assert "PRIMARY" in hint
    low = hint.lower()
    assert "complete section list" in low
    assert "exactly one section per outline entry" in low
    assert "parallel" in low  # forbids the second parallel set
    assert "append" in low    # forbids the appended tail
    # the actual outline titles reach the writer
    assert "Cost and Damage Assessment" in hint
    assert "Timeline of Key Events" in hint


def test_empty_outline_hint_is_blank_d56_fallback():
    """d56: an empty/absent outline yields "" so the write phase keeps the findings-driven
    decomposition rather than forcing a zero-section / broken doc."""
    assert _render_outline_hint([]) == ""
    assert _render_outline_hint(None) == ""
    # an outline of only blank titles also degrades to "" (no scaffold to impose)
    assert _render_outline_hint([{"title": "", "covers": "x"}]) == ""


def test_compose_write_goal_carries_no_fabricate_clause():
    """The composed write goal always carries the EMPTY-NODE-NO-FABRICATE clause: ground
    every section, drop/flag unsupported ones, never fabricate, no placeholder sources."""
    goal = _compose_write_goal(
        "US-Iran report", "report.html", "findings text", "SOURCES: 1. BBC",
        is_html=True, outline_hint=_OUTLINE,
    )
    low = goal.lower()
    assert "ground every section" in low
    assert "unsupported" in low
    assert "do not" in low and "from memory" in low
    assert "placeholder" in low
    # outline-as-primary clause is present (composes with the no-fabricate clause)
    assert "PRIMARY" in goal


def test_compose_write_goal_empty_outline_still_has_no_fabricate_and_no_scaffold():
    """With no outline, the goal still carries the anti-fabrication clause but NO outline
    scaffold clause (d56 findings-driven fallback)."""
    goal = _compose_write_goal(
        "topic", "out.md", "findings", "", is_html=False, outline_hint=[],
    )
    assert "ground every section" in goal.lower()
    assert "PRIMARY scaffold" not in goal
