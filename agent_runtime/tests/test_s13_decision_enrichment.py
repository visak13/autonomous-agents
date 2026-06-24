"""s13 (design §2 / B2) — ENRICHED REVIEW STEP: meaning-judged decision + agent stop.

Fast OFFLINE gate (no GPU, no network) for the B2 enrichment of the research-tree decision
node. Same scripted-transport + injected-gather seam as ``test_n4_research_tree.py``. Proves:

* (2b) the NEW low-arity ``stop_research`` tool is wired through the FULL seam — it PARSES
  (``parse_tree_call`` accepts it via ``TREE_TOOLS``), DISPATCHES on its own explicit branch
  in ``run_decision_node`` (NOT mis-routed into the ``set_next_direction`` catch-all), and
  the LAYER LOOP breaks EARLY with ``stop_reason='agent_sufficient'`` before the depth bound;
* (2c) ``expand_branch`` / ``prune_branch`` still parse + dispatch correctly (the meaning
  levers, unchanged) and a discrete ``set_next_direction`` is NOT confused with the new stop;
* (2a) ``render_for_decision`` emits the per-branch ``contributes: N claims, M sources,
  trust=T`` MEANING signal, computed from the persisted record alone (no new fetch).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

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


# --------------------------------------------------------------------------- #
# Scripted transport — replays a fixed sequence of decision-node turns (same
# shape as test_n4_research_tree._ScriptedTransport).
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
        self.calls: list[str] = []  # user turn of each chat call

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


def _make_gather(by_qhint):
    async def gather(branch: Branch, config: TreeConfig) -> LeafResult:
        for hint, proto in by_qhint.items():
            if hint in branch.question:
                return LeafResult(
                    branch_id=branch.id, question=branch.question,
                    findings=proto.findings, notes=list(proto.notes),
                    fetched=list(proto.fetched),
                )
        return LeafResult(
            branch_id=branch.id, question=branch.question,
            findings=f"findings for {branch.question}",
            notes=[_note("1", "primary", "t", ["c"], ["gap-x"])],
            fetched=[{"title": "t", "url": f"https://ex/{branch.id}", "markdown": "body"}],
        )
    return gather


GOAL = "Write a detailed sourced report on the June 2025 US-Iran conflict."


# =========================================================================== #
# 2b — stop_research PARSES via TREE_TOOLS (the new low-arity tool is on-surface).
# =========================================================================== #
def test_s13_parse_accepts_stop_research():
    assert "stop_research" in TREE_TOOLS
    assert parse_tree_call('{"tool":"stop_research","args":{"reason":"enough"}}') \
        == ("stop_research", {"reason": "enough"})
    # bare-key slip is recovered too
    assert parse_tree_call('{"stop_research": {"reason": "covered"}}') \
        == ("stop_research", {"reason": "covered"})


# =========================================================================== #
# 2b — the DECISION NODE dispatches stop_research on its OWN branch and STOPS,
#       and does NOT mis-route expand/prune into it (or set_next into stop).
# =========================================================================== #
def test_s13_decision_node_stop_research_breaks_and_is_carried():
    # The agent expands one grounded branch, then calls stop_research (enough). The decision
    # loop must BREAK on the stop call and carry the reason; the trailing turn is never read.
    transport = _ScriptedTransport([
        '{"tool":"expand_branch","args":{"parent":"root","question":"Fordow damage","rationale":"S1 gap"}}',
        '{"tool":"stop_research","args":{"reason":"thesis covered, no meaning-adding gap left"}}',
        "UNREACHED PROSE — the loop already stopped.",
    ])
    tree = Tree(fan_out=5)
    res: DecisionResult = asyncio.run(run_decision_node(
        transport, goal=GOAL, state_render="notes", tree=tree,
        config=TreeConfig(decide_max_turns=10), parent_depth=0,
    ))
    # stop carried; the expand BEFORE it still landed (meaning lever unchanged, 2c)
    assert res.stop_research == {"reason": "thesis covered, no meaning-adding gap left"}
    assert [b.question for b in res.new_branches] == ["Fordow damage"]
    # exactly 2 turns consumed — the loop broke on stop, did not read the 3rd turn
    assert res.turns == 2 and len(transport.calls) == 2
    # stop is NOT a next-direction (catch-all not hit)
    assert res.next_direction is None


def test_s13_expand_prune_still_dispatch_not_misrouted():
    # expand + prune + a discrete set_next must each route to their OWN handler — adding
    # stop_research to the menu must not capture or reorder them.
    transport = _ScriptedTransport([
        '{"tool":"expand_branch","args":{"parent":"root","question":"ceasefire terms","rationale":"S4 gap"}}',
        '{"tool":"prune_branch","args":{"branch":"S5","reason":"redundant with S1"}}',
        '{"tool":"set_next_direction","args":{"branch":"B1","reason":"highest value"}}',
        "FINAL TREE PLAN prose.",
    ])
    tree = Tree(fan_out=5)
    res = asyncio.run(run_decision_node(
        transport, goal=GOAL, state_render="notes", tree=tree,
        config=TreeConfig(decide_max_turns=10), parent_depth=1,
    ))
    assert [b.question for b in res.new_branches] == ["ceasefire terms"]
    assert res.pruned == [{"target": "S5", "reason": "redundant with S1"}]
    assert res.next_direction == {"branch": "B1", "reason": "highest value"}
    # set_next was honored as a next-direction, NOT swallowed as a stop signal
    assert res.stop_research is None


# =========================================================================== #
# 2b — the DECISION NODE surfaces the agent's explicit stop on its own channel.
#       (Migrated off the retired run_research_tree layer loop — the loop's
#       stop_reason='agent_sufficient' halt is DRIVEN by this DecisionResult signal,
#       which is the kept primitive. We prove the primitive directly.)
# =========================================================================== #
def test_s13_decision_node_surfaces_agent_sufficient_stop():
    # The layer-1 decision calls stop_research with a reason; the decision node must carry
    # that on DecisionResult.stop_research (the signal the layer loop reads to halt early
    # with 'agent_sufficient'), expand nothing, and never read the trailing turn.
    transport = _ScriptedTransport([
        '{"tool":"stop_research","args":{"reason":"the root overview already answers the thesis"}}',
        "UNREACHED.",
    ])
    tree = Tree(fan_out=5)
    res: DecisionResult = asyncio.run(run_decision_node(
        transport, goal=GOAL, state_render="root findings", tree=tree,
        config=TreeConfig(decide_max_turns=5), parent_depth=0,
    ))
    # the stop signal that DRIVES the loop's 'agent_sufficient' halt is carried on the result
    assert res.stop_research == {"reason": "the root overview already answers the thesis"}
    assert res.new_branches == []          # agent stopped instead of expanding
    # the loop broke on the stop call — exactly one turn consumed, trailing turn unread
    assert res.turns == 1 and len(transport.calls) == 1


def test_s13_agent_sufficient_wins_over_expansions_same_layer():
    # If the agent expands AND then stops in the same decision layer, the explicit stop is
    # carried ALONGSIDE the expansion — the stop is the decision (the layer loop ranks
    # 'agent_sufficient' over 'no_expansion'), and the expanded branch is still recorded.
    transport = _ScriptedTransport([
        '{"tool":"expand_branch","args":{"parent":"root","question":"casualties","rationale":"S1 gap"}}',
        '{"tool":"stop_research","args":{"reason":"enough"}}',
    ])
    tree = Tree(fan_out=5)
    res: DecisionResult = asyncio.run(run_decision_node(
        transport, goal=GOAL, state_render="notes", tree=tree,
        config=TreeConfig(decide_max_turns=10), parent_depth=0,
    ))
    # the explicit stop is the decision...
    assert res.stop_research == {"reason": "enough"}
    # ...AND the branch it expanded before stopping is still recorded on the result
    assert [b.question for b in res.new_branches] == ["casualties"]


# =========================================================================== #
# 2a — render_for_decision emits the per-branch 'contributes' MEANING line.
# =========================================================================== #
def test_s13_render_emits_per_branch_contributes_line(tmp_path):
    state = ResearchState(tmp_path / "state.jsonl")
    state.append_leaf(LeafResult(
        branch_id="root", question="overview",
        findings="f",
        notes=[
            _note("1", "primary", "AP", ["Fordow hit", "12 dead"], ["damage assessment"]),
            _note("2", "secondary", "Reuters", ["ceasefire signed"], []),
        ],
        fetched=[{"title": "AP", "url": "https://ap", "markdown": "m"}],
    ), layer=1)
    render = state.render_for_decision()  # reads BACK from disk (d49 real state)
    # 3 claims total (2 + 1), 2 note-bearing sources, both trust tiers present
    assert "contributes: 3 claims, 2 sources, trust=primary/secondary" in render
    # the underlying notes still render (the enrichment ADDS a line, removes nothing)
    assert "Fordow hit" in render and "damage assessment" in render


def test_s13_contributes_line_handles_empty_branch(tmp_path):
    state = ResearchState(tmp_path / "state.jsonl")
    state.append_leaf(LeafResult(
        branch_id="B1", question="dead-end", findings="", notes=[], fetched=[],
    ), layer=2)
    render = state.render_for_decision()
    assert "contributes: 0 claims, 0 sources, trust=n/a" in render


# =========================================================================== #
# P1-findings (s13) — FINDINGS BRIDGE. A leaf that gathered REAL findings but
# emitted 0 ArticleNotes must NEVER render "0 sources": the notes-only signal
# would make the decision node prune/stop on a branch full of real content. The
# bridge derives a NON-ZERO claims/sources signal from the persisted
# findings_digest + fetched_count, and surfaces the findings text so the planner
# decides on real data.
# =========================================================================== #
def test_s13_findings_bridge_nonzero_contribution_when_notes_empty(tmp_path):
    state = ResearchState(tmp_path / "state.jsonl")
    state.append_leaf(LeafResult(
        branch_id="B2", question="Fordow strike damage",
        # Real, substantive findings prose — but the small model emitted NO notes.
        findings=(
            "The Fordow enrichment site was struck on June 22, 2025. Satellite imagery "
            "shows two large craters over the underground halls. Iran acknowledged "
            "significant damage but disputed the US claim of total destruction. The IAEA "
            "reported it could not verify the centrifuge status."
        ),
        notes=[],
        fetched=[
            {"title": "AP", "url": "https://ap", "markdown": "m"},
            {"title": "Reuters", "url": "https://reuters", "markdown": "m"},
            {"title": "IAEA", "url": "https://iaea", "markdown": "m"},
        ],
    ), layer=1)
    render = state.render_for_decision()  # what the decision node actually sees
    # The branch is NOT rendered as empty — this is the core fix.
    assert "0 claims, 0 sources" not in render
    assert "0 sources" not in render
    # A non-zero contribution is shown, attributed to findings (not fabricated notes),
    # with the real fetched-source count (3) carried through.
    assert "claims (from findings)" in render
    assert "3 sources, trust=findings (notes not emitted)" in render
    # And the planner can SEE the actual findings content, not just a count.
    assert "Fordow enrichment site was struck" in render


def test_s13_findings_bridge_floors_sources_at_one_without_fetch_count(tmp_path):
    # A leaf with findings but neither notes NOR a recorded fetch count must still
    # read as a real contribution (>=1 claim, >=1 source) — it read SOMETHING to
    # write findings, so it is never shown as the empty 0/0 dead-end.
    state = ResearchState(tmp_path / "state.jsonl")
    state.append_leaf(LeafResult(
        branch_id="B3", question="ceasefire terms",
        findings="A ceasefire was brokered on June 24 with a phased withdrawal clause.",
        notes=[], fetched=[],
    ), layer=2)
    render = state.render_for_decision()
    assert "0 sources" not in render
    assert "1 sources, trust=findings (notes not emitted)" in render
    assert "claims (from findings)" in render


def test_s13_findings_bridge_does_not_disturb_noted_branches(tmp_path):
    # Regression guard: a branch WITH notes keeps the original notes-only signal
    # (no "from findings" wording, no findings line) — the bridge only fires when
    # notes are empty.
    state = ResearchState(tmp_path / "state.jsonl")
    state.append_leaf(LeafResult(
        branch_id="root", question="overview",
        findings="some findings prose that should NOT appear as a findings line here",
        notes=[_note("1", "primary", "AP", ["Fordow hit"], ["damage"])],
        fetched=[{"title": "AP", "url": "https://ap", "markdown": "m"}],
    ), layer=1)
    render = state.render_for_decision()
    assert "contributes: 1 claims, 1 sources, trust=primary" in render
    assert "from findings" not in render
    assert "findings: some findings prose" not in render
