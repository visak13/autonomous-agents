"""s15/a15 (d185) — NOTES-ARCH Layer 1: the EXPLICIT per-concern RESEARCH GRAPH.

Fast OFFLINE gate (no GPU, no network). Proves the research memory is now a per-concern
GRAPH (concern NODE → its NOTES → cited SOURCES, gaps_or_followups as EDGES that branch to
the next concern) rather than a flat note blob — and that the graph is:

* a DERIVED projection over the SAME persisted ResearchState records + verbatim source index
  (no new persistence — a17 owns that): a CITES edge resolves a note's url to an EXISTING
  [S#] (never a minted id); a gap edge is a note's own follow-up;
* OBSERVABLE — ``to_dict`` serializes the SHAPE (``per_concern_graph`` + per-concern nodes
  with note→source + gap edges) so the a14 harness can assert ``notes_graph_shaped``;
* DRIVING the loop — ``expand_branch`` adds a LIVE concern node via a gap edge and
  ``prune_branch`` COLLAPSES a concern node (the graph folds the live Tree's walk);
* COMPATIBLE with the existing decision render — the CONCERN GRAPH block is ADDITIVE; the
  running NARRATIVE, the VERBATIM SOURCE INDEX and the OPEN-GAP LENS are all preserved.
"""
from __future__ import annotations

from agent_runtime.research_tree import (
    Branch,
    ConcernGraph,
    GapEdge,
    LeafResult,
    ResearchState,
    SourceRef,
    Tree,
    build_concern_graph,
)


def _note(url, claims, gaps, *, trust="secondary", title="t"):
    return {
        "source_id": 1, "url": url, "title": title, "source_trust": trust,
        "category": "x", "summary": "s", "key_claims": claims, "relevance": "r",
        "gaps_or_followups": gaps,
    }


def _seed_two_concern_state(tmp_path) -> ResearchState:
    """Two gathered concerns: 'overview' (2 sources, 1 gap) + 'timeline' (1 source, 1 gap)."""
    state = ResearchState(tmp_path / "state.jsonl")
    state.append_leaf(LeafResult(
        branch_id="s1_B1", question="overview of the conflict",
        findings="f",
        notes=[
            _note("https://ap", ["Fordow hit", "12 dead"], ["damage assessment unresolved"]),
            _note("https://reuters", ["ceasefire signed"], []),
        ],
        fetched=[
            {"title": "AP", "url": "https://ap", "markdown": "# Strike\nFordow hit"},
            {"title": "Reuters", "url": "https://reuters", "markdown": "ceasefire signed"},
        ],
    ), layer=1)
    state.append_leaf(LeafResult(
        branch_id="s1_B2", question="timeline of key events",
        findings="f2",
        notes=[_note("https://nyt", ["June 22 strikes"], ["casualty figures unverified"])],
        fetched=[{"title": "NYT", "url": "https://nyt", "markdown": "June 22 strikes"}],
    ), layer=1)
    return state


# =========================================================================== #
# 1. the projection is GRAPH-shaped: per-concern nodes → note→source CITES edges
#    + gap edges, resolved against the EXISTING [S#] index (no minted ids).
# =========================================================================== #
def test_graph_is_per_concern_with_source_and_gap_edges(tmp_path):
    state = _seed_two_concern_state(tmp_path)
    graph = state.concern_graph()
    assert isinstance(graph, ConcernGraph)
    # ONE concern node per gathered leaf — NOT a flat blob.
    assert [n.concern_id for n in graph.concerns()] == ["s1_B1", "s1_B2"]
    overview = graph.nodes["s1_B1"]
    timeline = graph.nodes["s1_B2"]
    # concern → NOTES (granular, one per source) → cited SOURCES resolved to the stable [S#].
    assert len(overview.notes) == 2
    assert overview.source_ids == [1, 2]          # https://ap → S1, https://reuters → S2
    assert timeline.source_ids == [3]             # https://nyt → S3 (the run index order)
    # the note's gaps_or_followups are the OUTGOING gap EDGES (tied to the source that raised them).
    assert [g.text for g in overview.gaps] == ["damage assessment unresolved"]
    assert overview.gaps[0].from_concern == "s1_B1" and overview.gaps[0].source_sid == 1
    assert [g.text for g in timeline.gaps] == ["casualty figures unverified"]
    # the COVERED meaning is the concern's distinct claims.
    assert "Fordow hit" in overview.claims and "ceasefire signed" in overview.claims


def test_graph_flags_single_source_concern(tmp_path):
    state = _seed_two_concern_state(tmp_path)
    graph = state.concern_graph()
    # the 2-source concern is corroborated; the 1-source concern is flagged single_source
    # (the breadth!=depth signal the decision node expands on).
    assert graph.nodes["s1_B1"].single_source is False
    assert graph.nodes["s1_B2"].single_source is True


# =========================================================================== #
# 2. OBSERVABLE — to_dict serializes the SHAPE so the a14 gate can assert
#    notes_graph_shaped (per-concern graph, not a flat blob).
# =========================================================================== #
def test_graph_to_dict_is_observable_shape(tmp_path):
    state = _seed_two_concern_state(tmp_path)
    d = state.concern_graph().to_dict()
    assert d["shape"] == "per_concern_graph"
    assert d["concern_count"] == 2 and d["settled_count"] == 2
    # explicit directed edges: concern→source CITES + concern→gap edges are enumerable.
    cites = [e for e in d["edges"] if e["kind"] == "cites"]
    gaps = [e for e in d["edges"] if e["kind"] == "gap"]
    assert {(e["from"], e["to_source"]) for e in cites} == {
        ("s1_B1", 1), ("s1_B1", 2), ("s1_B2", 3)}
    assert {e["text"] for e in gaps} == {
        "damage assessment unresolved", "casualty figures unverified"}
    # each concern in the serialization carries its notes-as-graph fields (not a flat blob).
    for c in d["concerns"]:
        assert {"concern_id", "source_ids", "gaps", "claims", "single_source"} <= set(c)


# =========================================================================== #
# 3. the loop WALKS the graph — expand adds a LIVE concern via a gap edge,
#    prune COLLAPSES a concern node (the live Tree folds into the projection).
# =========================================================================== #
def test_expand_adds_live_concern_and_follows_gap_edge(tmp_path):
    state = _seed_two_concern_state(tmp_path)
    tree = Tree(fan_out=5)
    # the model EXPANDS the overview's open gap into a new concern (a real Tree.expand call).
    tree.expand(
        {"parent": "s1_B1", "question": "Fordow damage assessment", "rationale": "damage assessment gap"},
        depth=2,
    )
    graph = state.concern_graph(tree=tree)
    # the authored-but-ungathered concern is a LIVE node in the graph (no notes yet).
    child = next(n for n in graph.concerns() if n.question == "Fordow damage assessment")
    assert child.status == "live" and child.notes == [] and child.parent == "s1_B1"
    # the parent concern's OPEN gap was FOLLOWED by that child (the edge the loop walked).
    overview = graph.nodes["s1_B1"]
    assert overview.gaps[0].followed_by == child.concern_id
    assert overview.gaps[0].is_open is False
    # the timeline concern's gap stays OPEN (nothing expanded it) — honest, no fabricated link.
    assert graph.nodes["s1_B2"].open_gaps and graph.nodes["s1_B2"].open_gaps[0].is_open


def test_prune_collapses_a_concern_node(tmp_path):
    state = _seed_two_concern_state(tmp_path)
    tree = Tree(fan_out=5)
    tree.prune({"branch": "s1_B2", "reason": "redundant with s1_B1"})
    graph = state.concern_graph(tree=tree)
    # the pruned concern is COLLAPSED (closed without further gather), reason recorded.
    assert graph.nodes["s1_B2"].status == "collapsed"
    assert graph.nodes["s1_B2"].rationale == "redundant with s1_B1"
    assert [n.concern_id for n in graph.collapsed()] == ["s1_B2"]
    # a collapsed concern's gaps no longer count as open breadth.
    assert all(g.from_concern != "s1_B2" for g in graph.open_gaps())


# =========================================================================== #
# 4. COMPATIBLE — render_for_decision folds in the CONCERN GRAPH block additively;
#    the narrative + verbatim index + open-gap lens are all preserved.
# =========================================================================== #
def test_render_for_decision_folds_graph_and_preserves_existing(tmp_path):
    state = _seed_two_concern_state(tmp_path)
    # no-tree render: both concerns are live → graph block + the SINGLE SOURCE corroboration flag.
    render = state.render_for_decision()
    assert "CONCERN GRAPH" in render
    assert "[s1_B1]" in render and "open gap: damage assessment unresolved" in render
    assert "[s1_B2]" in render and "SINGLE SOURCE" in render  # the 1-source concern is flagged
    # the existing memory is UNTOUCHED — narrative + verbatim index + the open-gap lens.
    assert "RESEARCH NARRATIVE" in render and "SOURCE INDEX" in render
    assert "OPEN-GAP LENS" in render
    assert "https://ap" in render and "[S1]" in render
    # a render WITH the live tree folds the loop's walk in: a pruned concern shows COLLAPSED.
    tree = Tree(fan_out=5)
    tree.prune({"branch": "s1_B2", "reason": "redundant"})
    pruned_render = state.render_for_decision(tree=tree)
    assert "COLLAPSED" in pruned_render and "redundant" in pruned_render
    assert "[s1_B2]" not in pruned_render.split("COLLAPSED")[0].split("CONCERN GRAPH")[1]


def test_render_skips_graph_on_empty_branch(tmp_path):
    # a genuinely empty branch (no notes/sources/gaps) → no graph noise (and no crash).
    state = ResearchState(tmp_path / "s.jsonl")
    state.append_leaf(LeafResult(
        branch_id="B1", question="dead-end", findings="", notes=[], fetched=[],
    ), layer=2)
    render = state.render_for_decision()
    # the empty concern has a question but no notes/sources/gaps → graph renders a node head
    # only if it has content; a question-only live concern is allowed but carries no edges.
    graph = state.concern_graph()
    assert graph.nodes["B1"].notes == [] and graph.nodes["B1"].source_ids == []
    # render must not crash and the existing honest 'none yet' memory still shows.
    assert "RESEARCH NARRATIVE" in render or "none yet" in render


# =========================================================================== #
# 5. build_concern_graph is pure/derived — no fabrication (an unresolved note url
#    carries NO [S#] edge; a tree-only branch carries no notes).
# =========================================================================== #
def test_unresolved_note_url_carries_no_source_edge():
    records = [{
        "layer": 1, "branch_id": "s1_B1", "question": "q",
        "notes": [_note("https://not-in-index", ["a claim"], ["a gap"])],
    }]
    sources = [{"title": "Other", "url": "https://other", "markdown": "m"}]
    graph = build_concern_graph(records, sources)
    node = graph.nodes["s1_B1"]
    # the claim/gap are kept, but NO [S#] is minted for a url that is not in the index.
    assert node.claims == ["a claim"] and [g.text for g in node.gaps] == ["a gap"]
    assert node.source_ids == [] and node.gaps[0].source_sid is None


def test_dataclasses_are_constructible_directly():
    # the public graph dataclasses are importable + directly constructible (test surface).
    g = ConcernGraph()
    assert g.to_dict()["concern_count"] == 0
    edge = GapEdge(text="t", from_concern="c1")
    assert edge.is_open and edge.to_dict()["open"] is True
    ref = SourceRef(sid=4, url="u", title="t", trust="primary")
    assert ref.to_dict()["sid"] == 4
    b = Branch(id="B1", parent="root", question="q", rationale="r", depth=1)
    assert b.id == "B1"
