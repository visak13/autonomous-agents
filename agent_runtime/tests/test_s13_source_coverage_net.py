"""s13 / B5c (design §4C) — BACKSTOP C: DETERMINISTIC SOURCE-COVERAGE NET.

``ensure_source_coverage`` is the ``_ensure_source_coverage`` d89 specified but never
shipped (a1 Fact 4d): a FINAL deterministic pass UNDER the loop that appends an
"Additional sources" reference block for every fetched source the PHASE-2 write planner
assigned to NO section AND that is not already present (cited/listed) in the assembled
doc — so a fetched+cited source the planner skipped cannot silently vanish (the d87
dropped-source / Al-Jazeera-disappearance risk). Like the other backstops it is d60-safe
(adds ONLY a title+URL reference for material the run ACTUALLY fetched; never invents
content) and idempotent.

The fast unit tests prove the function's contract; the final test drives the REAL served
file-save report route end-to-end (a ``FakeTransport`` emits the assembled document, the
shared raw-file loop writes + finalizes it) and asserts an UNASSIGNED fetched source
still appears in the served doc's sources — i.e. the net actually FIRES on the route the
deliverable takes, not just in isolation.
"""
from __future__ import annotations

import asyncio

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.runtime import AgentRuntime
from agent_runtime.synth_tools import DONE_SENTINEL, ensure_source_coverage
from llm_framework import FakeTransport
from reactive_tools import EventPlane, ToolHook, register_agentic_tools


# A realistic global fetched-source list (1-based ids 1, 2, 3).
_SOURCES = [
    {"title": "Al Jazeera 12 days", "url": "https://aljazeera.com/news/12-days",
     "markdown": "Damage estimates rose over the conflict."},
    {"title": "CSIS cost analysis", "url": "https://csis.org/analysis/cost",
     "summary": "Total cost reached $16.5 billion by Day 12."},
    {"title": "Reuters ceasefire", "url": "https://reuters.com/world/ceasefire",
     "markdown": "A ceasefire was reached on Day 12."},
]


# --------------------------------------------------------------------------- #
# Pure-function contract (FAST).
# --------------------------------------------------------------------------- #
def test_s13_unassigned_fetched_source_is_appended_as_reference():
    # Sources 1 and 2 were assigned to a section (and cited inline); source 3 (Reuters)
    # was fetched but the write planner assigned it to NO section, so it is absent from
    # the body. The coverage net must append it so it does not silently vanish.
    doc = (
        "<!DOCTYPE html><html><head><title>R</title></head><body>"
        '<h2>Damage</h2><p>Per <a href="https://aljazeera.com/news/12-days">AJ</a>.</p>'
        '<h2>Cost</h2><p>See https://csis.org/analysis/cost.</p>'
        "</body></html>"
    )
    out, added = ensure_source_coverage(doc, _SOURCES, assigned_ids=[1, 2])
    assert added == ["https://reuters.com/world/ceasefire"]
    # the skipped source now appears in the served doc's sources (the B5c guarantee)...
    assert "https://reuters.com/world/ceasefire" in out
    assert "Additional sources" in out
    assert "Reuters ceasefire" in out
    # ...inside the document body, before the wrapper close.
    assert out.index("Additional sources") < out.rindex("</body>")
    # the assigned/cited sources are NOT re-listed (no duplication).
    assert out.count("https://csis.org/analysis/cost") == 1
    assert out.count("https://aljazeera.com/news/12-days") == 1


def test_s13_assigned_source_is_not_relisted_even_if_uncited():
    # Source 2 is ASSIGNED to a section but the model did not cite it inline (not in the
    # doc). It is the in-loop agent's job, NOT this net's — so it must NOT be appended.
    doc = '<body><h2>Damage</h2><p>https://aljazeera.com/news/12-days</p></body>'
    out, added = ensure_source_coverage(doc, _SOURCES, assigned_ids=[1, 2])
    # only source 3 (fetched, unassigned, absent) is recovered; source 2 (assigned) is not.
    assert added == ["https://reuters.com/world/ceasefire"]
    assert "https://csis.org/analysis/cost" not in out


def test_s13_already_cited_unassigned_source_is_not_duplicated():
    # Source 3 is assigned to NO section but its URL is ALREADY present in the doc (cited
    # by another writer node / section). Presence ⇒ covered ⇒ NOT re-listed.
    doc = (
        "<body><p>https://aljazeera.com/news/12-days</p>"
        "<p>ceasefire https://reuters.com/world/ceasefire</p></body>"
    )
    out, added = ensure_source_coverage(doc, _SOURCES, assigned_ids=[1])
    # source 2 (unassigned, absent) is recovered; source 3 (unassigned but already cited)
    # is left alone — no duplicate reference.
    assert added == ["https://csis.org/analysis/cost"]
    assert out.count("https://reuters.com/world/ceasefire") == 1


def test_s13_presence_normalisation_grounds_across_fragment_and_query():
    # An unassigned source whose URL appears in the doc only with a #fragment/?query is
    # the SAME resource → already present → not re-listed (no false duplicate).
    doc = "<body><p>https://reuters.com/world/ceasefire?utm=x#top</p></body>"
    out, added = ensure_source_coverage(doc, _SOURCES, assigned_ids=[1, 2])
    assert added == []
    assert out == doc  # idempotent / unchanged


def test_s13_markdown_path_appends_markdown_reference_block():
    doc = "# Report\n\nDamage per <https://aljazeera.com/news/12-days>.\n"
    out, added = ensure_source_coverage(doc, _SOURCES, assigned_ids=[1])
    # markdown deliverable → a markdown heading + list, not HTML.
    assert "## Additional sources" in out
    assert "[CSIS cost analysis](https://csis.org/analysis/cost)" in out
    assert "[Reuters ceasefire](https://reuters.com/world/ceasefire)" in out
    assert "<section" not in out
    assert set(added) == {
        "https://csis.org/analysis/cost",
        "https://reuters.com/world/ceasefire",
    }


def test_s13_reference_only_never_invents_content():
    # d60-safe: the block carries ONLY the fetched title + URL — no figures, no prose
    # about the topic. The only added URLs are real fetched-source URLs.
    doc = "<body><p>https://aljazeera.com/news/12-days</p></body>"
    out, added = ensure_source_coverage(doc, _SOURCES, assigned_ids=[1])
    fetched_urls = {s["url"] for s in _SOURCES}
    for u in added:
        assert u in fetched_urls


def test_s13_all_assigned_or_present_returns_unchanged():
    doc = (
        "<body>"
        "<p>https://aljazeera.com/news/12-days</p>"
        "<p>https://csis.org/analysis/cost</p>"
        "<p>https://reuters.com/world/ceasefire</p></body>"
    )
    # every source is present (cited) → covered → no change.
    assert ensure_source_coverage(doc, _SOURCES, assigned_ids=[]) == (doc, [])


def test_s13_is_idempotent():
    doc = '<body><p><a href="https://aljazeera.com/news/12-days">AJ</a></p></body>'
    once, added1 = ensure_source_coverage(doc, _SOURCES, assigned_ids=[1])
    twice, added2 = ensure_source_coverage(once, _SOURCES, assigned_ids=[1])
    assert added1  # something was appended the first time
    assert added2 == []  # the appended references are now present → no second append
    assert twice == once


def test_s13_no_sources_and_empty_doc_are_noops():
    assert ensure_source_coverage("<body>x</body>", []) == ("<body>x</body>", [])
    assert ensure_source_coverage("", _SOURCES) == ("", [])
    assert ensure_source_coverage("   ", _SOURCES) == ("   ", [])


# --------------------------------------------------------------------------- #
# SERVED-ROUTE firing — the net fires on the REAL file-save report route.
# --------------------------------------------------------------------------- #
def _run(coro):
    return asyncio.run(coro)


def _hook(tmp_path) -> ToolHook:
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    return hook


def test_s13_unassigned_source_appears_in_served_doc(tmp_path):
    """Drive the served file-save route once: a writer node assigned ONLY source 1 emits
    a report citing source 1; the run's global chain sources also hold source 3 (fetched,
    assigned to no section). The served on-disk document must still list source 3."""
    emitted = (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>Conflict report</title></head><body>"
        "<h1 id=\"conflict-report\">Conflict report</h1>"
        "<h2 id=\"damage\">Damage</h2>"
        "<p>Per <a href=\"https://aljazeera.com/news/12-days\">Al Jazeera</a>, "
        "damage estimates rose.</p>"
        "</body></html>"
    )

    def reply(messages, **opts):
        n = sum(1 for m in messages if m.get("role") == "assistant")
        return emitted if n == 0 else DONE_SENTINEL

    dag = PlanDAG(
        nodes=[
            PlanNode(id="w", task="Write the conflict report to report.html",
                     tool="file_write", depends_on=(), source_ids=(1,)),
        ],
        goal="Write a detailed HTML report to report.html",
    )
    rt = AgentRuntime(transport=FakeTransport([reply]), hook=_hook(tmp_path),
                      max_concurrency=1)
    # The deep-research / plan-chain gather sets the run's global fetched-source list on
    # the runtime; emulate that here so the writer node's _run_raw_file_loop has it.
    rt.chain_sources = _SOURCES
    out = _run(rt.run(dag))
    assert out.ok, out.failed

    served = (tmp_path / "report.html").read_text(encoding="utf-8")
    # source 1 (assigned + cited) is in the body as before...
    assert "https://aljazeera.com/news/12-days" in served
    # ...and the UNASSIGNED, uncited fetched sources (2 and 3) are recovered by the net.
    assert "Additional sources" in served
    assert "https://csis.org/analysis/cost" in served
    assert "https://reuters.com/world/ceasefire" in served
    # reference-only: titles are listed, no second copy of the assigned source.
    assert served.count("https://aljazeera.com/news/12-days") == 1
