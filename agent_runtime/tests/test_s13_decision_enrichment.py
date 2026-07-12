"""s13 (design §2 / B2) — ENRICHED REVIEW STEP: meaning-judged decision + agent stop.

Fast OFFLINE gate (no GPU, no network) for the B2 enrichment of the research-tree decision
node. Same scripted-transport + injected-gather seam as ``test_n4_research_tree.py``. Proves:

* (2b) the NEW low-arity ``stop_research`` tool is wired through the FULL seam — it PARSES
  (``parse_tree_call`` accepts it via ``TREE_TOOLS``), DISPATCHES on its own explicit branch
  in ``run_decision_node`` (NOT mis-routed into the ``set_next_direction`` catch-all), and
  the LAYER LOOP breaks EARLY with ``stop_reason='agent_sufficient'`` before the depth bound;
* (2c) ``expand_branch`` / ``prune_branch`` still parse + dispatch correctly (the meaning
  levers, unchanged) and a discrete ``set_next_direction`` is NOT confused with the new stop;
* (s14/P3A items 1+2) ``render_for_decision`` now renders the COMPACT RESEARCH MEMORY —
  a running NARRATIVE summary (COVERED grounded in stable [S#] / OPEN GAPS / DIRECTION) plus
  the VERBATIM SOURCE INDEX — and the raw per-branch ``contributes`` dump is RETIRED (it
  re-rendered all prior-layer state every layer → d146(2) unbounded context growth). The
  s13 findings-bridge is preserved INTO the narrative (a notes-empty branch surfaces its
  findings prose as COVERED so the decision node is never blind).
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


def _note_at(url, trust, title, claims, gaps):
    """A note whose ``url`` MATCHES a fetched source url, so the narrative resolves its
    claims to that source's stable [S#] (the s14/P3A index↔narrative join)."""
    return {
        "source_id": 1, "url": url, "title": title,
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
# s14/P3A items 1+2 — render_for_decision now renders the COMPACT RESEARCH MEMORY
# (running NARRATIVE summary + VERBATIM SOURCE INDEX) and the raw per-branch
# 'contributes' dump is RETIRED (kills the d146(2) unbounded linear context growth).
# The decision node reasons over COVERED (grounded in stable [S#]) + OPEN GAPS, and
# a stable [S#] verbatim source index — NOT a re-render of all prior-layer state.
# =========================================================================== #
def test_s14_render_emits_narrative_and_verbatim_index(tmp_path):
    state = ResearchState(tmp_path / "state.jsonl")
    state.append_leaf(LeafResult(
        branch_id="root", question="overview",
        findings="f",
        notes=[
            # note urls MATCH the fetched-source urls so the claim grounds to a real [S#]
            _note_at("https://ap", "primary", "AP", ["Fordow hit", "12 dead"], ["damage assessment"]),
            _note_at("https://reuters", "secondary", "Reuters", ["ceasefire signed"], []),
        ],
        fetched=[
            {"title": "AP", "url": "https://ap", "markdown": "# Strike\nFordow hit"},
            {"title": "Reuters", "url": "https://reuters", "markdown": "ceasefire signed"},
        ],
    ), layer=1)
    render = state.render_for_decision()  # reads BACK from disk (d49 real state)
    # the OLD raw per-branch dump is RETIRED
    assert "ARTICLE NOTES (already gathered)" not in render
    assert "contributes:" not in render
    # the NEW compact memory: a narrative + a verbatim [S#] index
    assert "RESEARCH NARRATIVE" in render and "SOURCE INDEX" in render
    # the notes' claims surface as COVERED bullets, grounded in the stable [S#] of the
    # source whose url matches the note (the index↔narrative join)
    assert "Fordow hit" in render and "ceasefire signed" in render
    assert "[S1]" in render and "[S2]" in render
    # the note's gap becomes an OPEN GAP the decision node can expand from
    assert "damage assessment" in render
    # the verbatim index carries the real fetched urls (never paraphrased)
    assert "https://ap" in render and "https://reuters" in render


def test_s14_render_handles_empty_branch(tmp_path):
    # A genuinely empty branch (no notes, no findings, no sources) renders an honest
    # 'none yet' for both artifacts — no crash, no fabricated content.
    state = ResearchState(tmp_path / "state.jsonl")
    state.append_leaf(LeafResult(
        branch_id="B1", question="dead-end", findings="", notes=[], fetched=[],
    ), layer=2)
    render = state.render_for_decision()
    assert "none yet" in render or "no sources" in render
    assert "contributes:" not in render


# =========================================================================== #
# FINDINGS BRIDGE (preserved into the narrative): a leaf that gathered REAL findings
# but emitted 0 ArticleNotes must NEVER leave the decision node blind. The narrative
# (built from notes) is then empty, so render derives COVERED bullets from the real
# findings prose — the decision node still reasons over what was gathered.
# =========================================================================== #
def test_s14_findings_bridge_surfaces_findings_when_notes_empty(tmp_path):
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
    # the branch is NOT blind — the findings prose is surfaced as COVERED (the bridge)
    assert "Fordow enrichment site was struck" in render
    assert "note lane emitted nothing" in render  # the bridge header
    # the verbatim index still lists the 3 real fetched sources by stable [S#]
    assert "[S1]" in render and "[S2]" in render and "[S3]" in render


def test_s14_findings_bridge_without_any_sources(tmp_path):
    # A leaf with findings but neither notes NOR fetched sources still surfaces its
    # findings as COVERED (it read SOMETHING to write them) — never an empty dead-end —
    # while the source index honestly reports no sources fetched yet.
    state = ResearchState(tmp_path / "state.jsonl")
    state.append_leaf(LeafResult(
        branch_id="B3", question="ceasefire terms",
        findings="A ceasefire was brokered on June 24 with a phased withdrawal clause.",
        notes=[], fetched=[],
    ), layer=2)
    render = state.render_for_decision()
    assert "A ceasefire was brokered on June 24" in render
    assert "no sources fetched yet" in render


def test_s14_noted_branch_prefers_narrative_over_findings_bridge(tmp_path):
    # Regression guard: a branch WITH notes uses the NOTE-derived narrative (claims as
    # COVERED grounded in [S#]); the findings-bridge fallback header must NOT appear.
    state = ResearchState(tmp_path / "state.jsonl")
    state.append_leaf(LeafResult(
        branch_id="root", question="overview",
        findings="some findings prose that should NOT appear as a covered bullet here",
        notes=[_note_at("https://ap", "primary", "AP", ["Fordow hit"], ["damage"])],
        fetched=[{"title": "AP", "url": "https://ap", "markdown": "m"}],
    ), layer=1)
    render = state.render_for_decision()
    # the note's claim is COVERED, grounded in the matching source's [S1]
    assert "Fordow hit" in render and "[S1]" in render
    # the note-driven narrative is used, NOT the findings-bridge fallback
    assert "note lane emitted nothing" not in render
    assert "some findings prose that should NOT appear" not in render
