"""s15/a22 (a14 finding #3) — a same-session FOLLOW-UP answers FROM persisted research.

Fast OFFLINE proof (no GPU, no network, no live transport). The a14 gate found a
same-session follow-up routed LINEAR and never read the a17 session-bound
``ResearchState`` back, so it answered from a blank slate (``session_readback_on_followup``
was false). a22 fixes the CHAT/ANSWER path: when a chat session already did research,
:func:`chat_app.agentic._session_readback` rehydrates that session's persisted notes +
verbatim sources (a17) and the per-research BRIEF (a16/d185 Layer 2) into a compact block
that :func:`~chat_app.agentic._with_session_readback` folds into the DOWNSTREAM
node-grounding context — so the follow-up answer is produced FROM the accumulated knowledge,
not a re-research.

These tests prove the READ-BACK MECHANISM directly (the a14/a23 live gate proves it on the
served route end-to-end):

1. READ-BACK SURFACES PRIOR RESEARCH — a session that researched turn-1 yields a block
   carrying the prior figures (the notes' settled claims), the REAL source URLs to cite,
   and the brief digest.
2. NO PRIOR RESEARCH → NO-OP — a fresh/unrelated session reads back ``""`` (so an
   ordinary first turn is byte-identical to pre-a22; no injection).
3. FOLD IS ADDITIVE — ``_with_session_readback`` appends the read-back AFTER the
   prior-turn chat memory and is a no-op when there is nothing to read back.
4. ISOLATION — a DIFFERENT session never reads another session's research back.
"""
from __future__ import annotations

import tempfile

from agent_runtime.research_tree import LeafResult, ResearchState

from chat_app.agentic import (
    _research_state_path,
    _session_readback,
    _with_session_readback,
)


def _leaf(bid: str, question: str, claim: str, url: str, body: str) -> LeafResult:
    return LeafResult(
        branch_id=bid,
        question=question,
        findings=f"{claim} (digest)",
        notes=[{"summary": claim, "key_claims": [claim], "url": url, "title": bid}],
        fetched=[{"title": bid, "url": url, "markdown": body}],
    )


def _seed_session(monkeypatch, tmp_path, session_id: str) -> None:
    """Isolate the research-state dir to ``tmp_path`` and seed a turn-1 research for the
    session (so a follow-up has prior notes + verbatim sources to read back)."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    turn1 = ResearchState(_research_state_path(None, session_id), session_bound=True)
    turn1.append_leaf(
        _leaf("b1", "US-Iran economic cost", "US economic cost reached $113.3B",
              "https://example.test/iran-econ", "ECONOMIC DAMAGE: $113.3B in 2025-2026."),
        layer=1,
    )
    turn1.append_leaf(
        _leaf("b2", "US-Iran casualties", "Combined casualties were 4175",
              "https://example.test/iran-cas", "CASUALTIES: 4175 killed and wounded."),
        layer=1,
    )


# =========================================================================== #
# 1. THE HEADLINE — the read-back surfaces the prior figures + sources + brief.
# =========================================================================== #
def test_readback_surfaces_prior_research(monkeypatch, tmp_path):
    sid = "a22-sess-1"
    _seed_session(monkeypatch, tmp_path, sid)

    block = _session_readback(sid, None)

    assert block, "a follow-up in a researched session must read prior research back"
    # the prior FIGURES (settled note claims) are present — the answer can reuse them
    assert "$113.3B" in block
    assert "4175" in block
    # the REAL source URLs are present so the answer cites the SAME sources (no re-fetch)
    assert "https://example.test/iran-econ" in block
    assert "https://example.test/iran-cas" in block
    # the [S#] ids the answer cites by are surfaced
    assert "[S1]" in block and "[S2]" in block
    # it is clearly labelled as prior-research grounding (reasoning-led, not a route flag)
    assert "PRIOR RESEARCH FROM THIS SESSION" in block
    # the a16 brief digest is folded in (thesis-level orientation)
    assert "PRIOR RESEARCH BRIEF" in block


# =========================================================================== #
# 2. NO PRIOR RESEARCH → NO-OP (an ordinary first turn is unchanged).
# =========================================================================== #
def test_fresh_session_reads_back_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    assert _session_readback("a22-fresh", None) == ""
    # and no session_id at all (inline/offline path) is likewise a no-op
    assert _session_readback(None, "run-xyz") == ""


# =========================================================================== #
# 3. THE FOLD is additive and a no-op when there is nothing to read back.
# =========================================================================== #
def test_with_session_readback_folds_after_prior_turn_context(monkeypatch, tmp_path):
    sid = "a22-sess-3"
    _seed_session(monkeypatch, tmp_path, sid)

    prior_turns = "USER: research US-Iran\nASSISTANT: (produced the report)"
    folded = _with_session_readback(prior_turns, sid, None)
    assert folded is not None
    # prior-turn chat memory stays FIRST, the heavier research read-back is appended AFTER
    assert folded.startswith(prior_turns)
    assert "$113.3B" in folded
    assert folded.index(prior_turns) < folded.index("PRIOR RESEARCH FROM THIS SESSION")

    # no prior research → the context is returned UNCHANGED (no injection)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "other"))
    assert _with_session_readback(prior_turns, "a22-empty", None) == prior_turns
    assert _with_session_readback(None, "a22-empty", None) is None


# =========================================================================== #
# 4. ISOLATION — a different session never reads another session's research back.
# =========================================================================== #
def test_cross_session_isolation(monkeypatch, tmp_path):
    sid = "a22-sess-A"
    _seed_session(monkeypatch, tmp_path, sid)

    # session A reads its own research back ...
    assert "$113.3B" in _session_readback(sid, None)
    # ... but a DIFFERENT session keyed in the SAME dir sees nothing of A's.
    assert _session_readback("a22-sess-B", None) == ""
