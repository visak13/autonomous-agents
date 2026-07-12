"""d235 — DEFINITION-TEXT + READ-TOOL COHESION: the two-tier READ on a COST HIERARCHY.

Proves the read-side deliverable of action ``at`` (definition-layer only, no flags):

1. ``read_notes`` (the CHEAP first leg) returns the structured ARTICLE-NOTE gist keyed by the
   GLOBAL ``[S#]`` (resolved by URL against the run's sources list, so a note gist and a
   ``load_source`` pull name the SAME source), and returns an explicit not-found for an unknown id.
2. The COST HIERARCHY is advertised in BOTH tool descriptions: ``read_notes`` is CHEAP / use
   FIRST; ``load_source`` is EXPENSIVE / use read_notes first / only for an exact figure to cite.
3. ``load_source`` is TUNED per d234 (top_n=2, per_call_cap=3000) — values that fit the E4B window.
4. The ``research_read`` bundle ADVERTISES both tools and its doctrine sequences them cheap-first,
   with NO overlap with the ``research`` (GATHER) bundle.

Fully OFFLINE — pure construction of the tools + bundles, no transport/network.
"""
from __future__ import annotations

from agent_runtime.bundles.research import ResearchBundle
from agent_runtime.bundles.research_read import (
    ResearchReadBundle,
    _LOAD_SOURCE_SPEC,
    _READ_NOTES_SPEC,
)
from agent_runtime.source_tools import (
    _DEFAULT_LOAD_MAX_CHARS,
    _DEFAULT_LOAD_TOP_N,
    _LOAD_SOURCE_DESCRIPTION,
    _READ_NOTES_DESCRIPTION,
    make_read_notes,
    make_read_notes_tool,
)


_NOTES = [
    {
        "source_id": 1,  # per-research-node id — NOT the global [S#]
        "url": "https://reuters.com/iran",
        "title": "Reuters Iran",
        "summary": "Economic damage put at $113.3B.",
        "key_claims": ["$113.3B economic damage", "4175 casualties reported"],
        "gaps_or_followups": ["who first reported the casualty figure?"],
        "source_trust": "secondary",
    }
]
# The GLOBAL source list: the matching source is [S2] here, so read_notes must re-key the note
# from its own source_id=1 to the global [S2] via URL match.
_SOURCES = [
    {"url": "https://apnews.com/x", "title": "AP", "markdown": "# AP\nbody"},
    {"url": "https://reuters.com/iran", "title": "Reuters Iran", "markdown": "# R\nbody"},
]


def test_read_notes_returns_gist_keyed_by_global_sid():
    rn = make_read_notes(_NOTES, _SOURCES)
    all_notes = rn()
    assert all_notes["count"] == 1
    entry = all_notes["notes"][0]
    # Re-keyed to the GLOBAL [S#] (S2), not the note's own source_id (1).
    assert entry["sid"] == "S2"
    assert "$113.3B economic damage" in entry["key_claims"]
    assert entry["gaps_or_followups"] == ["who first reported the casualty figure?"]
    # A single-source read by that global id resolves the same note.
    one = rn("S2")
    assert one["found"] is True
    assert one["sid"] == "S2"
    assert one["summary"].startswith("Economic damage")


def test_read_notes_unknown_sid_is_explicit_not_found():
    rn = make_read_notes(_NOTES, _SOURCES)
    miss = rn("S9")
    assert miss["found"] is False
    assert "NO NOTE" in miss["note"]


def test_read_notes_falls_back_to_own_id_without_sources():
    # No global sources to match against → the note keeps its own 1-based id (best-effort).
    rn = make_read_notes(_NOTES, [])
    entry = rn()["notes"][0]
    assert entry["sid"] == "S1"


def test_read_notes_empty_when_no_notes():
    rn = make_read_notes([], _SOURCES)
    out = rn()
    assert out["count"] == 0 and out["notes"] == []


def test_cost_hierarchy_is_advertised_in_both_descriptions():
    # read_notes — CHEAP, use FIRST.
    low_rn = _READ_NOTES_DESCRIPTION.lower()
    assert "cheap" in low_rn and "first" in low_rn
    # load_source — EXPENSIVE, read_notes first, only for the exact figure/quote to cite.
    low_ls = _LOAD_SOURCE_DESCRIPTION.lower()
    assert "expensive" in low_ls
    assert "read_notes" in low_ls
    assert "cite" in low_ls


def test_load_source_tuned_for_e4b_window():
    # d234 — tightened now that read_notes carries the breadth gist.
    assert _DEFAULT_LOAD_TOP_N == 2
    assert _DEFAULT_LOAD_MAX_CHARS == 3000


def _spec_name(spec):
    return spec.get("function", {}).get("name") or spec.get("name")


def test_research_read_bundle_advertises_both_read_tools():
    specs = ResearchReadBundle().tool_specs({})
    names = {_spec_name(s) for s in specs}
    assert "read_notes" in names
    assert "load_source" in names
    # The native schemas carry the cost-hierarchy framing.
    assert "CHEAP" in _READ_NOTES_SPEC["function"]["description"]
    assert "EXPENSIVE" in _LOAD_SOURCE_SPEC["function"]["description"]


def test_read_doctrine_sequences_cheap_first_and_owns_only_the_read_domain():
    doctrine = ResearchReadBundle().own_doctrine
    low = doctrine.lower()
    # read_notes FIRST, then load_source.
    assert "read_notes first" in low
    assert low.index("read_notes") < low.index("load_source")
    # research_read is the READ domain — it does NOT gather (no search/fetch here).
    assert "search" not in low and "web_fetch" not in low


def test_make_read_notes_tool_shape():
    t = make_read_notes_tool(_NOTES, _SOURCES)
    assert t.name == "read_notes"
    assert "CHEAP" in t.description
    # sid is OPTIONAL (read all when omitted).
    assert "sid" not in (_READ_NOTES_SPEC["function"]["parameters"]["required"])


def test_gather_bundle_note_is_the_structured_article_note():
    # 'note' in the GATHER bundle is the structured ARTICLE NOTE artifact, not a prose memo.
    specs = ResearchBundle().tool_specs({"emit_notes": True})
    note_spec = next(s for s in specs if _spec_name(s) == "note")
    desc = note_spec["function"]["description"].lower()
    assert "structured article note" in desc
    assert "key_claims" in desc and "gaps_or_followups" in desc
    assert "not a prose memo" in desc
