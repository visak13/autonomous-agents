"""s13 (design §3 / B3) — the outline channel, RETIRED from the served decision surface (s14/a12).

Fast OFFLINE gate (no GPU, no network). s13/B3 added low-arity ``add_section`` / ``drop_section``
outline tools to the decision node. s14/a12 (d154) REMOVED them from the served surface: the
generic engine discards the tree-authored outline (PHASE-2 runs ``outline_hint=None``, d56), so
offering the outline tools gave the model a SECOND, silently-dropped surface to author
``source_ids`` on — it routed source_ids there (dropped) instead of onto the consumed file_write
write-planner. This module now proves the RETIREMENT and the kept inert plumbing:

* ``add_section`` / ``drop_section`` are NO LONGER OFFERED — absent from ``TREE_TOOLS`` and
  ``TREE_TOOL_SPECS``, rejected by ``parse_tree_call``, and ``run_decision_node`` emits NO
  outline ops even when the model tries to call them;
* the lower-level outline plumbing is KEPT, inert and unit-tested in isolation: ``Tree``'s
  outline methods are append-only, and the ``ResearchState`` outline channel still persists +
  reads back from disk (so the channel can be re-enabled without a code change);
* the PHASE-2 write goal stays findings-driven on an EMPTY outline (d56), and a NON-empty
  ``outline_hint`` still REACHES ``write_goal`` as the PRIMARY scaffold above the findings — the
  write-side mechanic is unchanged; the generic served path simply always feeds it ``None``.
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
    TREE_TOOL_SPECS,
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
# s14/a12 (d154) — add_section / drop_section are REMOVED from the offered surface:
#   absent from TREE_TOOLS + TREE_TOOL_SPECS, and parse_tree_call rejects them.
# =========================================================================== #
def test_s14_outline_tools_removed_from_decision_surface():
    assert "add_section" not in TREE_TOOLS and "drop_section" not in TREE_TOOLS
    spec_names = {s["function"]["name"] for s in TREE_TOOL_SPECS}
    assert "add_section" not in spec_names and "drop_section" not in spec_names
    # a model reply that still names the retired tool is NOT recognized as a tool call
    # (parse_tree_call gates on TREE_TOOLS) → it is treated as prose, never dispatched.
    assert parse_tree_call(
        '{"tool":"add_section","args":{"title":"Damage","covers":"S1,S2"}}'
    ) is None
    assert parse_tree_call('{"drop_section": {"title": "Aftermath", "reason": "off"}}') is None


# =========================================================================== #
# s14/a12 — the decision node no longer routes the retired outline tools: even when
#   the model emits add_section/drop_section, NO outline ops are produced (the dead
#   source_id channel is gone). The replies fall through as prose (loop ends cleanly).
# =========================================================================== #
def test_s14_decision_node_does_not_route_outline_tools():
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
    # the retired tools authored NOTHING — no effective outline, no carried ops
    assert tree.outline() == []
    assert res.outline_ops == []
    # and the run still terminates normally (no expansion / no spurious next-direction)
    assert res.next_direction is None and res.stop_research is None


# =========================================================================== #
# s14/a12 — the lower-level Tree outline methods are KEPT (inert plumbing) and stay
#   append-only: a DIRECT method call still upserts/drops, so the channel can be
#   re-enabled without a code change. (These are no longer reachable via the model.)
# =========================================================================== #
def test_s14_tree_outline_methods_still_append_only():
    tree = Tree(fan_out=5)
    tree.add_section({"title": "A", "covers": "x"})
    tree.add_section({"title": "B", "covers": "y"})
    tree.drop_section({"title": "A", "reason": "redundant"})
    # effective outline folds (add upserts, drop removes); op log keeps all three (append-only)
    assert [s["title"] for s in tree.outline()] == ["B"]
    assert [o["op"] for o in tree.outline_ops] == ["add", "add", "drop"]


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
# s14/a12 — the decision node hands NO outline ops to the channel now (the tools are
#   retired), so persisting its result is a no-op; the ResearchState outline channel
#   itself still persists + reads back DIRECT ops from disk (the kept inert plumbing).
# =========================================================================== #
def test_s14_decision_emits_no_ops_but_channel_persists_direct(tmp_path):
    state = ResearchState(tmp_path / "s.jsonl")
    transport = _ScriptedTransport([
        '{"tool":"add_section","args":{"title":"Background","covers":"S1 origins"}}',
        "FINAL PLAN: the report covers background then strike damage.",
    ])
    tree = Tree(fan_out=5)
    res: DecisionResult = asyncio.run(run_decision_node(
        transport, goal=GOAL, state_render="notes", tree=tree,
        config=TreeConfig(decide_max_turns=10), parent_depth=0,
    ))
    # the retired tools produced NO ops → persisting the decision is a no-op (channel empty)
    assert res.outline_ops == []
    state.append_outline_ops(res.outline_ops)
    assert state.read_outline() == []
    # the channel plumbing still works when fed ops DIRECTLY (re-enable without a code change)
    state.append_outline_ops([{"op": "add", "title": "Background", "covers": "S1 origins"}])
    assert [s["title"] for s in state.read_outline()] == ["Background"]


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
        outline_hint=outline,
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
            outline_hint=empty,
        )
        assert "PRIMARY scaffold" not in goal and "DOCUMENT OUTLINE" not in goal
        # RP-1 (d319/d311): the 'Decompose it into the sections it needs' single-document
        # framing is RETIRED — the goal stays findings-driven and output-agnostic.
        assert "FINDINGS_BODY" in goal
    # a hint of only blank titles also degrades to the findings-driven goal
    assert _render_outline_hint([{"title": "  ", "covers": "x"}]) == ""
