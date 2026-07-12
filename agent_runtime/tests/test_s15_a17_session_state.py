"""s15/a17 (d185) — NOTES-ARCH Layer 3: SESSION-BOUND research state.

Fast OFFLINE gate (no GPU, no network). Proves the raw ``ResearchState`` file:

* DEFAULT (run-scoped) is byte-identical to pre-a17 — opening truncates, so a fresh
  state never carries a prior run's notes/sources (the inline/offline path);
* SESSION-BOUND STICKS — a follow-up state opened on the SAME session path READS BACK
  the prior leaf notes AND the verbatim sources (so it can build on / ``load_source``
  them instead of re-researching);
* CROSS-SESSION ISOLATED — a state keyed by a different session never sees them;
* the persisted verbatim source body is re-loadable by ``load_source`` after read-back
  (the leaf records alone never carried the markdown — the sidecar does), with the
  stable ``[S#]`` id preserved across the reopen.
"""
from __future__ import annotations

from agent_runtime.research_tree import LeafResult, ResearchState
from agent_runtime.source_tools import make_load_source


def _leaf(bid: str, question: str, claim: str, url: str, body: str) -> LeafResult:
    return LeafResult(
        branch_id=bid,
        question=question,
        findings=f"{claim} (digest)",
        notes=[{"summary": claim, "key_claims": [claim], "url": url, "title": bid}],
        fetched=[{"title": bid, "url": url, "markdown": body}],
    )


# =========================================================================== #
# 1. DEFAULT run-scoped state truncates on open (pre-a17 byte-identical).
# =========================================================================== #
def test_default_state_truncates_on_open(tmp_path):
    p = tmp_path / "run.jsonl"
    s1 = ResearchState(p)  # session_bound defaults False
    s1.append_leaf(_leaf("b1", "q1", "fact one", "https://x/1", "BODY ONE 12.3B"), layer=1)
    assert len(s1.read()) == 1

    # A NEW run-scoped state on the SAME path WIPES it (only THIS run is read back).
    s2 = ResearchState(p)
    assert s2.read() == []
    assert s2.sources() == []


# =========================================================================== #
# 2. THE HEADLINE: a SESSION-BOUND follow-up reads prior notes + sources back.
# =========================================================================== #
def test_session_bound_reads_back_prior_notes_and_sources(tmp_path):
    p = tmp_path / "sess__abc.jsonl"

    # Turn 1 — research, persisted under the session key.
    t1 = ResearchState(p, session_bound=True)
    t1.append_leaf(_leaf("b1", "US-Iran costs", "US economic cost $113.3B", "https://x/iran", "ECON $113.3B body"), layer=1)
    assert len(t1.read()) == 1 and len(t1.sources()) == 1

    # Turn 2 — a FOLLOW-UP in the SAME session opens the SAME file and READS BACK.
    t2 = ResearchState(p, session_bound=True)
    recs = t2.read()
    assert len(recs) == 1                                   # prior leaf record present
    assert recs[0]["branch_id"] == "b1"
    notes = t2.collect_notes()
    assert any("113.3B" in (n.get("summary") or "") for n in notes)  # prior NOTE read back
    srcs = t2.sources()
    assert len(srcs) == 1                                   # prior verbatim SOURCE read back
    assert srcs[0]["url"] == "https://x/iran"
    assert "ECON $113.3B" in srcs[0]["markdown"]            # the body the leaf record never held

    # and load_source RESOLVES that prior source's body by its stable [S#] (no re-fetch).
    load = make_load_source(t2.sources())
    out = load("S1")
    assert "$113.3B" in out["text"]

    # a NEW leaf in turn 2 APPENDS (accumulates), keeping the prior [S#] stable.
    t2.append_leaf(_leaf("b2", "follow-up", "casualties 4175", "https://x/cas", "CAS 4175 body"), layer=1)
    assert len(t2.read()) == 2
    assert t2.sources()[0]["url"] == "https://x/iran"       # S1 unchanged
    assert t2.sources()[1]["url"] == "https://x/cas"        # S2 appended


# =========================================================================== #
# 3. cross-session isolation — a different session key never sees another's state.
# =========================================================================== #
def test_cross_session_isolation(tmp_path):
    pa = tmp_path / "sess__A.jsonl"
    pb = tmp_path / "sess__B.jsonl"

    a = ResearchState(pa, session_bound=True)
    a.append_leaf(_leaf("b1", "qa", "secret A fact", "https://a/1", "A BODY"), layer=1)

    # session B (different key) opens cleanly — sees NOTHING of A.
    b = ResearchState(pb, session_bound=True)
    assert b.read() == []
    assert b.sources() == []
    assert b.collect_notes() == []

    # re-opening A still has A's state (isolation is symmetric, A intact).
    a2 = ResearchState(pa, session_bound=True)
    assert len(a2.read()) == 1
    assert a2.sources()[0]["markdown"] == "A BODY"
