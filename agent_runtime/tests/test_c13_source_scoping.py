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
    assemble_html_spa,
    collect_fetched_sources_full,
    render_scoped_sources,
    render_source_catalog,
)
from llm_framework import FakeTransport
from reactive_tools import EventPlane, ToolHook, register_agentic_tools


def _run(coro):
    return asyncio.run(coro)


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
# 6b) assemble_html_spa — wrap a per-section fragment into ONE navigable SPA
# --------------------------------------------------------------------------- #
def test_assemble_spa_wraps_fragment_and_builds_nav_from_headings():
    frag = (
        "<h2>Military Losses</h2><p>14 batteries. "
        "<a href='https://aljazeera.com/losses'>src</a></p>"
        "<h2>Economic Impact</h2><p>Brent $104.</p>"
    )
    spa = assemble_html_spa(frag, title="US-Iran")
    low = spa.lower()
    # exactly one well-formed document wrapper
    assert low.count("<!doctype") == 1 and low.count("<html") == 1
    assert low.count("</html>") == 1 and low.count("<body") == 1
    # a nav built FROM the model's own headings, linking to injected anchors
    assert "<nav" in low
    assert 'href="#military-losses"' in low and 'href="#economic-impact"' in low
    assert 'id="military-losses"' in low and 'id="economic-impact"' in low
    # real content + the model's cited URL are PRESERVED verbatim (no truncation)
    assert "14 batteries" in spa and "https://aljazeera.com/losses" in spa


def test_assemble_spa_idempotent_on_existing_wrapper_and_noop_on_non_html():
    # an already-wrapped doc gets a nav but NOT a second wrapper
    full = "<!DOCTYPE html><html><head><title>x</title></head><body><h2>A</h2><p>z</p></body></html>"
    out = assemble_html_spa(full)
    assert out.lower().count("<html") == 1 and out.lower().count("</html>") == 1
    assert "<nav" in out.lower()
    # a heading-less fragment / non-HTML string is returned unchanged (no nav forced)
    assert assemble_html_spa("just some markdown text") == "just some markdown text"


# --------------------------------------------------------------------------- #
# 7) end-to-end: a scoped synthesis node writes the section to a REAL file
# --------------------------------------------------------------------------- #
def _hook(tmp_path) -> ToolHook:
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    return hook


def test_scoped_synthesis_writes_section_to_real_file(tmp_path):
    node = PlanNode(id="s1", task="Write the Military Losses section to report.md",
                    role="synthesizer", source_ids=(2,))
    section = "## Military Losses\nEach side reported losses. [Al Jazeera](https://aljazeera.com/losses)"
    sub = SubAgent(
        node,
        transport=FakeTransport([section, "<<DONE>>"]),
        hook=_hook(tmp_path),
        chain_sources=_SOURCES,
    )
    raw, parsed, _v, _r = _run(sub._run_synthesis(None, "Write the report."))
    written = parsed.get("written_path")
    assert written is not None
    text = (tmp_path / written).read_text(encoding="utf-8") if not (tmp_path / written).is_absolute() else open(written, encoding="utf-8").read()
    assert "Military Losses" in text
    assert "https://aljazeera.com/losses" in text  # the real cited URL landed


# --------------------------------------------------------------------------- #
# 8) MS3 R2 — section-scoped PER-PAGE verify grounds THIS node's section inside
#    the write loop (not bypassed by the whole-doc >9000-char cap) and rewrites it.
# --------------------------------------------------------------------------- #
def _read_written(tmp_path, written: str) -> str:
    p = tmp_path / written
    return p.read_text(encoding="utf-8") if not p.is_absolute() else open(written, encoding="utf-8").read()


def test_section_verify_grounds_or_removes_unbacked_claim_in_the_write_loop(tmp_path):
    """With verify_lane ON, the section the node wrote is fact-checked against ITS scoped
    sources INSIDE the loop: an unbacked claim is flagged, a revise turn removes it, and
    the corrected section is re-persisted — so a long report is grounded per-section even
    though the whole-doc verify is bypassed past _VERIFY_REVISE_MAX_CHARS."""
    node = PlanNode(id="s1", task="Write the Military Losses section to report.md",
                    role="synthesizer", source_ids=(2,))
    # The writer emits the real Al-Jazeera claim PLUS a fabricated statute claim.
    section = ("## Military Losses\nEach side reported losses [Al Jazeera]"
               "(https://aljazeera.com/losses). Under the fictional Pact of 1887 the "
               "strike was lawful.")
    grounded = "## Military Losses\nEach side reported losses [Al Jazeera](https://aljazeera.com/losses)."
    sub = SubAgent(
        node,
        # writer: section, <<DONE>>; then per-section verify: flag → revise → re-verify ok
        transport=FakeTransport([
            section,
            "<<DONE>>",
            '{"verdict":"revise","unbacked":[{"claim":"Pact of 1887","reason":"no source"}]}',
            grounded,
            '{"verdict":"ok"}',
        ]),
        hook=_hook(tmp_path),
        chain_sources=_SOURCES,
        verify_lane=True,
    )
    raw, parsed, _v, _r = _run(sub._run_synthesis(None, "Write the report."))
    written = parsed.get("written_path")
    assert written is not None
    text = _read_written(tmp_path, written)
    assert "Pact of 1887" not in text              # the fabrication was ground-or-removed
    assert "https://aljazeera.com/losses" in text  # the real grounded content was kept


def test_section_verify_is_default_off_byte_identical(tmp_path):
    """verify_lane OFF (the default) → NO verify turn fires; the section is written
    verbatim. The extra scripted verify responses are never consumed."""
    node = PlanNode(id="s1", task="Write the Military Losses section to report.md",
                    role="synthesizer", source_ids=(2,))
    section = "## Military Losses\nLosses reported [AJ](https://aljazeera.com/losses). Fictional claim X."
    sub = SubAgent(
        node,
        transport=FakeTransport([section, "<<DONE>>", '{"verdict":"revise"}', "should-not-be-used"]),
        hook=_hook(tmp_path),
        chain_sources=_SOURCES,
        verify_lane=False,
    )
    raw, parsed, _v, _r = _run(sub._run_synthesis(None, "Write the report."))
    text = _read_written(tmp_path, parsed.get("written_path"))
    assert "Fictional claim X." in text  # untouched: no verify lane ran
