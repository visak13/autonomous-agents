"""s13 (design §3 / B3) — AGENT-DECIDED DOCUMENT DIRECTION: the outline channel.

Fast OFFLINE gate (no GPU, no network) for the B3 enrichment — the FAITHFULNESS LINCHPIN
that lets the research agent author the FINAL document's section plan and have it actually
REACH the document. Same scripted-transport + injected-gather seam as the B2 s13 tests.
Proves the full B3 chain:

* the NEW low-arity ``add_section`` / ``drop_section`` tools are wired through the SAME
  4-point seam as B2's ``stop_research`` — they PARSE (``parse_tree_call`` via ``TREE_TOOLS``),
  DISPATCH on their OWN explicit branch in ``run_decision_node`` (NOT mis-routed into the
  ``set_next_direction`` catch-all), and carry their ops on ``DecisionResult``;
* the decision emits ``add_section`` x2 → the OutlinePlan is PERSISTED to the ResearchState
  outline channel and READ BACK from disk (the anti-hallucination read-back), append-only
  (a ``drop`` removes; later add refines), and surfaces on ``ResearchState.read_outline()``;
* an EMPTY outline → the PHASE-2 write goal stays findings-driven (d56 — no hard-coded /
  fabricated sections), and a NON-empty outline REACHES ``write_goal`` as the PRIMARY
  scaffold ABOVE the findings (not a trailing aside).
"""
from __future__ import annotations

import asyncio

from agent_runtime.research_tree import (
    Branch,
    DecisionResult,
    LeafResult,
    ResearchState,
    Tree,
    TreeConfig,
    TREE_TOOLS,
    parse_tree_call,
    run_decision_node,
)

# The B3 write-goal weave lives in the served write phase (chat_app) — importable in the
# shared workspace .venv (d11). Pure/string-only helpers, no GPU/network.
from chat_app.agentic import _compose_write_goal, _render_outline_hint


# --------------------------------------------------------------------------- #
# Scripted transport + gather (same shape as the B2 s13 tests).
# --------------------------------------------------------------------------- #
class _ChatResult:
    def __init__(self, content: str) -> None:
        self.role = "assistant"
        self.content = content
        self.thinking = None
        self.raw = None


class _ScriptedTransport:
    def __init__(self, turns: list[str]) -> None:
        self._turns = list(turns)
        self.calls: list[str] = []

    def chat(self, messages, **opts) -> _ChatResult:
        user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        self.calls.append(user)
        i = len(self.calls) - 1
        content = self._turns[i] if i < len(self._turns) else "FINAL PLAN (fallback)."
        return _ChatResult(content)

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content


def _note(sid, trust, title, claims, gaps):
    return {
        "source_id": sid, "url": f"https://ex/{sid}", "title": title,
        "source_trust": trust, "category": "x", "summary": "s",
        "key_claims": claims, "relevance": "r", "gaps_or_followups": gaps,
    }


def _basic_gather():
    async def gather(branch: Branch, config: TreeConfig) -> LeafResult:
        return LeafResult(
            branch_id=branch.id, question=branch.question,
            findings=f"findings for {branch.question}",
            notes=[_note("1", "primary", "AP", ["a claim"], ["a gap"])],
            fetched=[{"title": "AP", "url": f"https://ex/{branch.id}", "markdown": "m"}],
        )
    return gather


GOAL = "Write a detailed sourced report on the June 2025 US-Iran conflict."


# =========================================================================== #
# B3 — add_section / drop_section PARSE via TREE_TOOLS (on-surface, low-arity).
# =========================================================================== #
def test_s13_parse_accepts_add_and_drop_section():
    assert "add_section" in TREE_TOOLS and "drop_section" in TREE_TOOLS
    assert parse_tree_call(
        '{"tool":"add_section","args":{"title":"Damage","covers":"S1,S2"}}'
    ) == ("add_section", {"title": "Damage", "covers": "S1,S2"})
    # bare-key slip is recovered too (mirrors the stop_research parse)
    assert parse_tree_call('{"drop_section": {"title": "Aftermath", "reason": "off-thesis"}}') \
        == ("drop_section", {"title": "Aftermath", "reason": "off-thesis"})


# =========================================================================== #
# B3 — the decision node emits add_section x2 on their OWN branch (NOT mis-routed
#       into set_next), carries the ops, and refines via drop_section (append-only).
# =========================================================================== #
def test_s13_decision_emits_add_section_x2_carried_not_misrouted():
    transport = _ScriptedTransport([
        '{"tool":"add_section","args":{"title":"Background","covers":"S1 origins"}}',
        '{"tool":"add_section","args":{"title":"Strike damage","covers":"S2,S3 figures"}}',
        "FINAL TREE PLAN prose.",
    ])
    tree = Tree(fan_out=5)
    res: DecisionResult = asyncio.run(run_decision_node(
        transport, goal=GOAL, state_render="notes", tree=tree,
        config=TreeConfig(decide_max_turns=10), parent_depth=0,
    ))
    # both sections landed in the effective outline, in order
    assert [s["title"] for s in tree.outline()] == ["Background", "Strike damage"]
    assert tree.outline()[1]["covers"] == "S2,S3 figures"
    # the ops are carried on the DecisionResult for the layer loop to persist (2 adds)
    assert [(o["op"], o["title"]) for o in res.outline_ops] == [
        ("add", "Background"), ("add", "Strike damage"),
    ]
    # add_section is NOT a next-direction (catch-all not hit) and NOT a stop
    assert res.next_direction is None and res.stop_research is None


def test_s13_drop_section_is_append_only_and_removes():
    # add A, add B, drop A → effective outline = [B]; the op log keeps all three (append-only)
    transport = _ScriptedTransport([
        '{"tool":"add_section","args":{"title":"A","covers":"x"}}',
        '{"tool":"add_section","args":{"title":"B","covers":"y"}}',
        '{"tool":"drop_section","args":{"title":"A","reason":"redundant"}}',
        "FINAL PLAN.",
    ])
    tree = Tree(fan_out=5)
    res = asyncio.run(run_decision_node(
        transport, goal=GOAL, state_render="notes", tree=tree,
        config=TreeConfig(decide_max_turns=10), parent_depth=0,
    ))
    assert [s["title"] for s in tree.outline()] == ["B"]
    assert [o["op"] for o in res.outline_ops] == ["add", "add", "drop"]


# =========================================================================== #
# B3 — the ResearchState OUTLINE CHANNEL persists ops + reads them BACK from disk
#       (anti-hallucination read-back), folding append-only (add upserts, drop removes).
# =========================================================================== #
def test_s13_research_state_outline_persists_and_reads_back(tmp_path):
    state = ResearchState(tmp_path / "s.jsonl")
    # the channel sidecar is distinct from the leaf-state file
    assert state.outline_path != state.path and state.outline_path.exists()
    state.append_outline_ops([
        {"op": "add", "title": "Background", "covers": "S1"},
        {"op": "add", "title": "Damage", "covers": "S2"},
        {"op": "drop", "title": "Background", "reason": "merged"},
        {"op": "add", "title": "Aftermath", "covers": "S4"},
    ])
    # read_outline READS THE FILE every call (no in-memory cache) — the bytes on disk fold to
    # the effective outline (add upserts, drop removes), proving persistence + read-back.
    reread = state.read_outline()
    assert [s["title"] for s in reread] == ["Damage", "Aftermath"]
    assert reread[0]["covers"] == "S2"
    # the raw sidecar bytes hold the FULL append-only op log (4 ops), not just the fold
    assert sum(1 for ln in state.outline_path.read_text(encoding="utf-8").splitlines() if ln.strip()) == 4
    # rendered for the decision prompt: non-empty shows the sections (read-back)
    rendered = state.render_outline_for_decision()
    assert "DOCUMENT OUTLINE" in rendered and "Damage" in rendered and "Aftermath" in rendered
    assert "Background" not in rendered  # the dropped section is gone


def test_s13_empty_outline_renders_propose_prompt(tmp_path):
    state = ResearchState(tmp_path / "s.jsonl")
    assert state.read_outline() == []
    rendered = state.render_outline_for_decision()
    assert "none yet" in rendered and "add_section" in rendered


# =========================================================================== #
# B3 — the decision node's outline ops PERSIST to the ResearchState channel and
#       READ BACK from disk (the (findings, sources, OUTLINE) hand-off to PHASE-2).
#       (Migrated off the retired run_research_tree layer loop: that loop simply
#       fed DecisionResult.outline_ops into ResearchState.append_outline_ops and
#       surfaced ResearchState.read_outline() on its result. We drive the kept
#       primitives — run_decision_node + the ResearchState outline channel — directly.)
# =========================================================================== #
def test_s13_decision_outline_persists_to_state_and_reads_back(tmp_path):
    # layer-1 decision authors the document direction (2 sections) then writes prose with no
    # expansion. The ops it carries are persisted to the outline channel and read back from
    # disk — exactly the hand-off the retired loop performed.
    state = ResearchState(tmp_path / "s.jsonl")
    transport = _ScriptedTransport([
        '{"tool":"add_section","args":{"title":"Background","covers":"S1 origins"}}',
        '{"tool":"add_section","args":{"title":"Strike damage","covers":"S2 Fordow"}}',
        "FINAL PLAN: the report covers background then strike damage.",
    ])
    tree = Tree(fan_out=5)
    res: DecisionResult = asyncio.run(run_decision_node(
        transport, goal=GOAL, state_render="notes", tree=tree,
        config=TreeConfig(decide_max_turns=10), parent_depth=0,
    ))
    # the decision carried the 2 add ops; the agent did NOT expand (final prose, no branches)
    assert res.new_branches == []
    assert res.outline_ops[0] == {"op": "add", "title": "Background", "covers": "S1 origins"}
    # persist this layer's ops to the channel (what the loop did) ...
    state.append_outline_ops(res.outline_ops)
    # ... then read the agent-decided outline BACK from disk (anti-hallucination read-back)
    assert [s["title"] for s in state.read_outline()] == ["Background", "Strike damage"]


def test_s13_empty_outline_when_agent_proposes_none(tmp_path):
    # an agent that never calls add_section → no outline ops → the channel stays EMPTY (no
    # fabricated sections); PHASE-2 will fall back to findings-driven decomposition (d56).
    state = ResearchState(tmp_path / "s.jsonl")
    transport = _ScriptedTransport(["FINAL PLAN with no outline calls."])
    tree = Tree(fan_out=5)
    res: DecisionResult = asyncio.run(run_decision_node(
        transport, goal=GOAL, state_render="notes", tree=tree,
        config=TreeConfig(decide_max_turns=10), parent_depth=0,
    ))
    assert res.outline_ops == []
    state.append_outline_ops(res.outline_ops)
    assert state.read_outline() == []


# =========================================================================== #
# B3 — outline_hint REACHES write_goal as the PRIMARY scaffold (above findings),
#       and an empty outline keeps the write goal findings-driven (d56 fallback).
# =========================================================================== #
def test_s13_outline_hint_reaches_write_goal_as_primary_scaffold():
    outline = [
        {"title": "Background", "covers": "S1 origins"},
        {"title": "Strike damage", "covers": "S2,S3 Fordow figures"},
    ]
    goal = _compose_write_goal(
        "Report on the conflict", "report.html",
        "FINDINGS_BODY_MARKER", "AVAILABLE SOURCES: ...",
        is_html=True, outline_hint=outline,
    )
    # both section titles reach the goal
    assert "Background" in goal and "Strike damage" in goal
    assert "S2,S3 Fordow figures" in goal           # the covers reach it too
    assert "PRIMARY scaffold" in goal               # framed as the scaffold, not an aside
    # PRIMARY = it appears ABOVE the research findings block (not a trailing aside)
    assert goal.index("Background") < goal.index("FINDINGS_BODY_MARKER")


def test_s13_empty_outline_keeps_write_goal_findings_driven():
    # d56 — no outline → NO scaffold clause; the goal stays findings-driven, unchanged.
    for empty in (None, []):
        goal = _compose_write_goal(
            "Report", "report.md", "FINDINGS_BODY", "SOURCES",
            is_html=False, outline_hint=empty,
        )
        assert "PRIMARY scaffold" not in goal and "DOCUMENT OUTLINE" not in goal
        assert "FINDINGS_BODY" in goal and "Decompose it into the sections it needs" in goal
    # a hint of only blank titles also degrades to the findings-driven goal
    assert _render_outline_hint([{"title": "  ", "covers": "x"}]) == ""
