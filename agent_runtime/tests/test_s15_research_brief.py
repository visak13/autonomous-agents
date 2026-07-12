"""s15/a16 (d185) — NOTES-ARCH Layer 2: the per-research BRIEF.

Fast OFFLINE gate (no GPU, no network). Proves the BRIEF is a thesis-level digest
DERIVED from a15's per-concern graph (Layer 1) over the SAME persisted ResearchState:

* DERIVED + no fabrication — every thesis line is a key_claim a settled concern's note
  actually carried; every source is an already-resolved [S#] (never a minted id);
* THESIS-LEVEL — the dominant cross-concern claims lead (corroboration ranks them);
* ADDRESSABLE — a compact JSON dict (topic + thesis + concerns + sources + open gaps +
  counts + a one-glance digest) a chat session stores per research;
* DISTINCT per research — two different researches project two different briefs.
"""
from __future__ import annotations

from agent_runtime.research_tree import (
    LeafResult,
    ResearchState,
    build_research_brief,
)


def _note(url, claims, gaps, *, trust="secondary", title="t"):
    return {
        "source_id": 1, "url": url, "title": title, "source_trust": trust,
        "category": "x", "summary": "s", "key_claims": claims, "relevance": "r",
        "gaps_or_followups": gaps,
    }


def _seed_iran_state(tmp_path) -> ResearchState:
    """Two gathered concerns; 'ceasefire signed' is corroborated by TWO concerns."""
    state = ResearchState(tmp_path / "state.jsonl")
    state.append_leaf(LeafResult(
        branch_id="s1_B1", question="overview of the conflict",
        findings="f",
        notes=[
            _note("https://ap", ["Fordow hit", "ceasefire signed"], ["damage unresolved"]),
            _note("https://reuters", ["ceasefire signed"], []),
        ],
        fetched=[
            {"title": "AP", "url": "https://ap", "markdown": "Fordow hit"},
            {"title": "Reuters", "url": "https://reuters", "markdown": "ceasefire signed"},
        ],
    ), layer=1)
    state.append_leaf(LeafResult(
        branch_id="s1_B2", question="timeline of key events",
        findings="f2",
        notes=[_note("https://nyt", ["June 22 strikes", "ceasefire signed"],
                     ["casualty figures unverified"])],
        fetched=[{"title": "NYT", "url": "https://nyt", "markdown": "June 22 strikes"}],
    ), layer=1)
    return state


# =========================================================================== #
# 1. the brief is the GRAPH projected to a thesis-level digest (shape + content).
# =========================================================================== #
def test_brief_shape_and_thesis_from_settled_concerns(tmp_path):
    state = _seed_iran_state(tmp_path)
    brief = state.research_brief(topic="US-Iran conflict report")

    assert brief["shape"] == "per_research_brief"
    assert brief["topic"] == "US-Iran conflict report"
    # thesis = the distinct established claims (no fabrication beyond the notes).
    note_claims = {"Fordow hit", "ceasefire signed", "June 22 strikes"}
    assert set(brief["thesis"]) <= note_claims
    assert "ceasefire signed" in brief["thesis"]
    # both settled concerns are present, each carrying its source ids.
    assert {c["concern_id"] for c in brief["concerns"]} == {"s1_B1", "s1_B2"}
    assert brief["settled_count"] == 2


# =========================================================================== #
# 2. the cross-concern claim LEADS (corroboration ranks the thesis).
# =========================================================================== #
def test_corroborated_claim_ranks_first(tmp_path):
    state = _seed_iran_state(tmp_path)
    brief = state.research_brief()
    # 'ceasefire signed' is carried by BOTH concerns → highest weight → first.
    assert brief["thesis"][0] == "ceasefire signed"


# =========================================================================== #
# 3. sources are the DISTINCT resolved [S#] (no minted id, stable order).
# =========================================================================== #
def test_sources_are_distinct_resolved_sids(tmp_path):
    state = _seed_iran_state(tmp_path)
    brief = state.research_brief()
    sids = [s["sid"] for s in brief["sources"]]
    assert sids == sorted(sids)
    assert sids == [1, 2, 3]            # AP, Reuters, NYT in fetch order
    assert all(isinstance(s, int) for s in sids)
    assert brief["source_count"] == 3
    # every sid is real (present in the verbatim index), never minted.
    assert all(1 <= s <= len(state.sources()) for s in sids)


# =========================================================================== #
# 4. open gaps are surfaced (what THIS research did not yet close).
# =========================================================================== #
def test_open_gaps_surfaced(tmp_path):
    state = _seed_iran_state(tmp_path)
    brief = state.research_brief()
    gaps = " | ".join(brief["open_gaps"]).lower()
    assert "damage" in gaps
    assert "casualty" in gaps
    assert brief["open_gap_count"] >= 2


# =========================================================================== #
# 5. a one-glance digest renders deterministically (the addressable summary).
# =========================================================================== #
def test_digest_renders_topic_thesis_sources(tmp_path):
    state = _seed_iran_state(tmp_path)
    brief = state.research_brief(topic="US-Iran conflict report")
    digest = brief["digest"]
    assert "US-Iran conflict report" in digest
    assert "ceasefire signed" in digest
    assert "S1" in digest and "S3" in digest


# =========================================================================== #
# 6. an EMPTY research yields an honest empty brief — never a fabricated summary.
# =========================================================================== #
def test_empty_research_is_honest(tmp_path):
    state = ResearchState(tmp_path / "empty.jsonl")
    brief = state.research_brief(topic="nothing yet")
    assert brief["thesis"] == []
    assert brief["sources"] == []
    assert brief["settled_count"] == 0
    assert "nothing gathered yet" in brief["digest"].lower()


# =========================================================================== #
# 7. DISTINCT researches project DISTINCT briefs (the addressability premise).
# =========================================================================== #
def test_two_researches_produce_distinct_briefs(tmp_path):
    iran = _seed_iran_state(tmp_path / "a")
    econ = ResearchState(tmp_path / "b" / "state.jsonl")
    econ.append_leaf(LeafResult(
        branch_id="s1_B1", question="inflation outlook",
        findings="f",
        notes=[_note("https://imf", ["CPI up 4.1%"], ["wage data missing"])],
        fetched=[{"title": "IMF", "url": "https://imf", "markdown": "CPI up 4.1%"}],
    ), layer=1)

    b1 = iran.research_brief(topic="US-Iran conflict")
    b2 = econ.research_brief(topic="inflation outlook")
    assert b1["topic"] != b2["topic"]
    assert set(b1["thesis"]).isdisjoint(set(b2["thesis"]))
    assert b1["digest"] != b2["digest"]


# =========================================================================== #
# 8. the module-level builder matches the ResearchState convenience method.
# =========================================================================== #
def test_build_research_brief_matches_state_method(tmp_path):
    state = _seed_iran_state(tmp_path)
    via_fn = build_research_brief(state.read(), state.sources(), topic="X")
    via_method = state.research_brief(topic="X")
    assert via_fn == via_method
