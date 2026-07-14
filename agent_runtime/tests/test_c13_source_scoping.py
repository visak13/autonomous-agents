"""s9/c13 — SPA-per-section bounded synthesis: per-section SOURCE-SCOPING (d55/d56/d57).

The E4B sliding-window-attention ceiling (gemma4 ``sliding_window=512``): the single
synthesis node builds one giant document with an 18k-74k-tok source block, so as
generation moves to later sections the REAL sources fall OUTSIDE the local attention
window → placeholder facts (``$XX billion``) + placeholder citations (``URL 1``). The
fix decomposes the long sourced report into ONE bounded write node PER section, each
fed ONLY its planner-assigned sources NEAREST the generation cursor (so they stay
inside the ~512-tok window). The source→section assignment is the planner's REASONING
(``source_ids`` per node), NOT a code relevance-matcher (d56 hard guard); the runtime
only DELIVERS the assigned subset (d17-style feed-scoping).

These tests cover the path-independent core: the source-collection/render helpers, the
``source_ids`` plumbing (PlanNode/parse_dag/PlanBuilder), and the SubAgent scoped feed
(scoped node → only its sources, nearest the cursor; unscoped node → the full index,
byte-identical to the pre-c13 degenerate path).
"""
import asyncio

from agent_runtime.factory import AbstractPlanFactory, PlanDAG, PlanNode
from agent_runtime.plan_tools import PlanBuilder
from agent_runtime.runtime import SubAgent
from agent_runtime.synth_tools import (
    collect_fetched_sources_full,
    render_scoped_sources,
    render_source_catalog,
)
from llm_framework import FakeTransport
from reactive_tools import EventPlane, ToolHook, register_agentic_tools


def _run(coro):
    return asyncio.run(coro)


# d242 TRUE self-select: a raw-emission synthesis/file loop starts TOOL-LESS and runs a
# SELF-SELECT FRONT — the model loads the 'file' bundle (then replies READY) before it
# authors. Scripts therefore lead with this exchange so the front consumes it, not content.
_SS_FILE = '{"tool": "get_bundles", "args": {"name": "file"}}'


_SOURCES = [
    {"title": "BBC", "url": "https://bbc.com/iran",
     "markdown": "Brent crude rose above $100 after the strike."},
    {"title": "Al Jazeera", "url": "https://aljazeera.com/losses",
     "markdown": "Each side reported military losses overnight."},
    {"title": "CFR", "url": "https://cfr.org/timeline",
     "markdown": "A detailed timeline of the escalation."},
]

_FETCHED_TV = [
    {"fetched": [_SOURCES[0], _SOURCES[1]]},
    {"fetched": [_SOURCES[1], _SOURCES[2]]},  # AJ repeats — must dedupe by URL
]


# --------------------------------------------------------------------------- #
# 1) collect_fetched_sources_full — stable, URL-deduped, retains markdown
# --------------------------------------------------------------------------- #
def test_collect_full_dedupes_by_url_in_first_seen_order_keeping_markdown():
    full = collect_fetched_sources_full(_FETCHED_TV)
    assert [s["url"] for s in full] == [
        "https://bbc.com/iran", "https://aljazeera.com/losses", "https://cfr.org/timeline",
    ]
    # markdown retained (the per-section excerpt source)
    assert "Brent crude" in full[0]["markdown"]
    # a non-mapping / no-fetched value contributes nothing
    assert collect_fetched_sources_full([{"no": "fetched"}, "x", None]) == []


# --------------------------------------------------------------------------- #
# 2) render_source_catalog — numbered title—url for the write planner (no bodies)
# --------------------------------------------------------------------------- #
def test_catalog_is_numbered_titles_urls_without_article_bodies():
    cat = render_source_catalog(_SOURCES)
    assert "[1] BBC — https://bbc.com/iran" in cat
    assert "[2] Al Jazeera — https://aljazeera.com/losses" in cat
    assert "[3] CFR — https://cfr.org/timeline" in cat
    # the planner catalog is the index ONLY — no article excerpt leaks in
    assert "Brent crude" not in cat
    assert render_source_catalog([]) == ""


# --------------------------------------------------------------------------- #
# 3) render_scoped_sources — ONLY the assigned ids, tight, with the cite rule
# --------------------------------------------------------------------------- #
def test_scoped_render_keeps_only_assigned_ids_and_carries_the_cite_rule():
    block = render_scoped_sources(_SOURCES, [2])
    # ONLY source 2's url + excerpt — sources 1 and 3 are NOT in this section's turn
    assert "https://aljazeera.com/losses" in block
    assert "Each side reported military losses" in block
    assert "https://bbc.com/iran" not in block
    assert "https://cfr.org/timeline" not in block
    # the anti-fabrication / cite-verbatim instruction rides the scoped block
    assert "VERBATIM" in block
    assert "URL 1" in block  # the instruction explicitly bans the 'URL 1' placeholder


def test_scoped_render_tight_excerpt_budget_and_out_of_range_and_empty():
    big = [{"title": "T", "url": "https://x/y", "markdown": "Z" * 5000}]
    block = render_scoped_sources(big, [1], excerpt_budget=300)
    assert block.count("Z") <= 300  # excerpt capped tight (SWA-friendly)
    # out-of-range / empty selection → no block (graceful; node falls back to full index)
    assert render_scoped_sources(_SOURCES, [99]) == ""
    assert render_scoped_sources(_SOURCES, []) == ""
    assert render_scoped_sources([], [1]) == ""


# --------------------------------------------------------------------------- #
# 4) source_ids plumbing — parse_dag + PlanNode normalisation
# --------------------------------------------------------------------------- #
def test_plannode_normalises_source_ids():
    # deduped, positive-int only, order-preserving; invalid/non-positive dropped
    n = PlanNode(id="n1", task="intro", source_ids=(1, "2", 2, 0, "x", -3))
    assert n.source_ids == (1, 2)
    # absent → empty tuple (no scoping)
    assert PlanNode(id="n2", task="body").source_ids == ()


def test_build_dag_carries_source_ids_to_plannodes():
    # the static DAG builder the parsed planner output flows through
    node_dicts = [
        {"id": "n1", "task": "intro", "spec": None, "specs": (), "depends_on": [],
         "tool": "file_write", "tool_args": {}, "role": None, "needs_spec": None,
         "source_ids": [1, "2", 2]},
        {"id": "n2", "task": "body", "spec": None, "specs": (), "depends_on": ["n1"],
         "tool": "file_write", "tool_args": {}, "role": None, "needs_spec": None},
    ]
    dag = AbstractPlanFactory._build_dag(node_dicts, "", "")
    assert dag.nodes[0].source_ids == (1, 2)
    assert dag.nodes[1].source_ids == ()


# --------------------------------------------------------------------------- #
# 5) PlanBuilder.add_step carries source_ids into the authored plan
# --------------------------------------------------------------------------- #
def test_plan_builder_add_step_records_source_ids():
    b = PlanBuilder(tool_names=["file_write"], shape_name="write-file")
    b.seed_plan({})
    b.add_step({"task": "Military Losses section", "tool": "file_write", "source_ids": [2, 2, "3"]})
    structured = b.to_structured()
    assert structured["nodes"][0]["source_ids"] == [2, 3]


# --------------------------------------------------------------------------- #
# 6) SubAgent scoped feed — scoped node sees ONLY its sources; unscoped → full index
# --------------------------------------------------------------------------- #
def _sub(node: PlanNode, *, chain_sources=None) -> SubAgent:
    return SubAgent(node, transport=FakeTransport([]), chain_sources=chain_sources)


def test_scoped_node_suppresses_full_index_and_blocks_only_its_sources():
    node = PlanNode(id="s2", task="Military Losses", tool="file_write", source_ids=(2,))
    sub = _sub(node, chain_sources=_SOURCES)
    # the full-index append is SUPPRESSED for a scoped node (it goes nearest the cursor)
    assert sub._with_source_index("USER") == "USER"
    block = sub._scoped_source_block()
    assert "https://aljazeera.com/losses" in block
    assert "https://bbc.com/iran" not in block and "https://cfr.org/timeline" not in block


_BIG_SOURCES = [
    {"title": "BBC", "url": "https://bbc.com/iran",
     "markdown": "# Oil\n" + ("Brent crude rose above $100 after the strike. " * 400)},
    {"title": "Al Jazeera", "url": "https://aljazeera.com/losses",
     "markdown": "# Losses\n" + ("Each side reported military losses overnight. " * 400)},
    {"title": "CFR", "url": "https://cfr.org/timeline",
     "markdown": "# Timeline\n" + ("A detailed timeline of the escalation. " * 400)},
]


def test_d170_scoped_writer_block_pushes_full_figure_bearing_body_not_a_starving_lead():
    # d170 (supersedes d156 FOR THE WRITER): the raw-file writer CANNOT emit tool calls (d49),
    # so it cannot PULL source text on demand — its turn is therefore PUSHED the FULL
    # figure-bearing BODY of its (few) assigned sources (the good-run calibration, trace
    # bc7cef17), NOT a compact index + a ~2400-char lead, which STARVED the writer to a thin
    # report (the d167 mis-calibration). The REVIEWER (full_index, which CAN call load_source)
    # still gets the compact index — see the next test.
    node = PlanNode(id="s1", task="Military Losses", tool="file_write", source_ids=(2,))
    sub = _sub(node, chain_sources=_BIG_SOURCES)
    block = sub._scoped_source_block()
    assert "[S2]" in block and "https://aljazeera.com/losses" in block
    # the REAL verbatim body is pushed (so the writer has the figures to quote), not a stub.
    assert "military losses overnight" in block
    # only THIS section's assigned source — sibling urls are not dumped into the writer turn.
    assert "https://bbc.com/iran" not in block and "https://cfr.org/timeline" not in block
    # NOT starved: carries a substantial slice of the body (far more than the d167 ~2400 lead).
    assert len(block) > 8000
    # still BOUNDED by CONSTRUCTION: a single section's feed stays well under the num_ctx
    # char envelope (the d162 no-truncation guarantee), never the all-bodies 137KB dump.
    assert len(block) < int(32768 * 3.5 * 0.6)


def test_d156_reviewer_full_index_lists_all_sids_for_citation_resolution():
    # d156 citation-persistence root: the anchored reviewer resolves against EVERY source's
    # [S#] (full_index) so a writer's valid cross-section citation never falsely "does not
    # resolve" and gets deleted — even though this review node is scoped to a subset.
    node = PlanNode(id="s2_review", task="Military Losses", tool="file_update", source_ids=(2,))
    sub = _sub(node, chain_sources=_BIG_SOURCES)
    full = sub._scoped_source_block(full_index=True)
    # the MAP lists all three [S#] for resolution
    assert "[S1]" in full and "[S2]" in full and "[S3]" in full
    # the writer-scoped block (no full_index) maps ONLY the assigned id
    scoped = sub._scoped_source_block()
    assert "[S1]" not in scoped and "[S3]" not in scoped and "[S2]" in scoped


def test_d157_research_read_is_bounded_to_a_compact_chunk():
    # d157: the per-source research.react read is bounded to a compact relevant CHUNK so a
    # multi-turn react loop's accumulated input stays small (the full body is still stored).
    from agent_runtime.runtime import RESEARCH_READ_CHUNK_CHARS
    node = PlanNode(id="r1", task="oil prices")
    sub = SubAgent(node, transport=FakeTransport([]), chunked_read=True)
    assert sub._read_content_budget() <= RESEARCH_READ_CHUNK_CHARS


def test_unscoped_node_keeps_full_upstream_index_and_no_scoped_block():
    # the degenerate 1-section / single-synth path: no source_ids → full index, no scoped
    # block (byte-identical to the pre-c13 behaviour; no regression).
    node = PlanNode(id="syn", task="synthesize", role="synthesizer")
    sub = SubAgent(
        node, transport=FakeTransport([]),
        upstream_tool_values={"n1": {"fetched": [_SOURCES[0]]}},
    )
    out = sub._with_source_index("USER")
    assert "https://bbc.com/iran" in out  # full index appended
    assert sub._scoped_source_block() == ""  # no scoping when unscoped / no chain_sources


# --------------------------------------------------------------------------- #
# 7) end-to-end: a scoped synthesis node writes the section to a REAL file
# --------------------------------------------------------------------------- #
def _hook(tmp_path) -> ToolHook:
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    return hook



# --------------------------------------------------------------------------- #
# AUTONOMY REBUILD P2C — the raw write loop behavior tests below are RETIRED.
# The served raw-content write loop (_run_synthesis/_run_raw_file_loop/
# _run_file_delivery and its riders: stepwise continuation, done-redirect,
# close-continuation, is_detailed_task completeness nudge, re-emission drop,
# csv-tabular rider) is DELETED — every node runs the unified self-select loop
# and AUTHORS its file via file_write; delivery is verified by the
# target-artifact gate (test_target_artifact_gate.py) and duplicate/restart
# writes are governed at the TOOL BOUNDARY (file_write no-clobber refusal).
# --------------------------------------------------------------------------- #
