"""s13/P1-report — WARCOSTS-PARITY via STRUCTURED-DATA -> HTML-SPECIALIZATION separation.

Fast, fully OFFLINE structural tests (no transport, no GPU, no network) for the three
checkable Phase-1 invariants of P1-report:

  (1) DATA-vs-HTML separation — a structured ``.json`` DATA file is legitimate data and
      passes the write guard verbatim, while the HTML deliverable's CONTENT stays RAW
      prose (a JSON envelope is unwrapped, a bare envelope refused). This is the d50.1
      invariant: JSON lives in a data file / the tool-call layer, never as the report's
      content. The synthesis framing carries the separation directive (one title, shared
      facts once, optional data file) without scripting a pipeline.

  (2) NO THEMATIC DUPLICATE-TAIL — the B8a2 residual where the first write node
      over-produces a whole mini-document (its own title + figures/Sources) and a later
      section opens a SECOND ``<h1>`` (a second document shell) is collapsed: the served
      document carries exactly ONE top-level title and one pass of each section family.

  (3) NO TRUNCATION MARKER — a final section cut MID-SENTENCE at ``num_predict``
      ("…On February</body></html>") is detected and a short dangling fragment trimmed,
      so the served document ends cleanly; a verify/revise rewrite that would truncate
      the file is never persisted.
"""

from __future__ import annotations

import re

import pytest

from agent_runtime.runtime import _REPORT_SEPARATION_GUIDANCE
from agent_runtime.synth_tools import (
    collapse_duplicate_sections,
    enforce_single_h1,
    has_truncation_marker,
    reconcile_doc_structure,
    trim_dangling_sentence,
)
from reactive_tools import ToolInputError
from reactive_tools.file_tools import guard_write_content


# --------------------------------------------------------------------------- #
# (1) DATA-vs-HTML SEPARATION — structured data file is data; report content is raw.
# --------------------------------------------------------------------------- #
def test_s13_json_data_file_is_legitimate_data() -> None:
    """A structured ``.json`` DATA file passes the write guard VERBATIM (it is data, not
    the deliverable's prose) — the separation's data side is legitimate (d50 forbids JSON
    for the deliverable CONTENT, not for a data file)."""
    data = '{"figures": {"us_kia": 15, "wounded": 538}, "sources": ["https://www.warcosts.org/conflicts/iran-2026"]}'
    assert guard_write_content(data, "iran_data.json") == data


def test_s13_html_deliverable_content_stays_raw_prose() -> None:
    """The HTML deliverable's CONTENT stays RAW: a ``{"output": "<h1>…"}`` envelope is
    UNWRAPPED to the raw HTML, and a bare ``{"findings": [...]}`` wrapper with no
    deliverable text is REFUSED — so a JSON envelope can never become the report (d50.1)."""
    wrapped = '{"output": "<h1>US-Iran War (2026)</h1><p>Substantive prose.</p>"}'
    unwrapped = guard_write_content(wrapped, "report.html")
    assert unwrapped == "<h1>US-Iran War (2026)</h1><p>Substantive prose.</p>"
    assert not unwrapped.lstrip().startswith("{")

    with pytest.raises(ToolInputError):
        guard_write_content('{"findings": ["a", "b"]}', "report.html")


def test_s13_separation_guidance_directs_data_vs_presentation() -> None:
    """The report framing carries the separation directive: ONE title, shared facts
    written ONCE, an optional structured ``.json`` data file, and RAW (never JSON)
    content — the mechanism that stops the first node over-producing a whole mini-doc."""
    g = _REPORT_SEPARATION_GUIDANCE.lower()
    assert ".json" in g                     # the optional structured DATA file
    assert "one title" in g or "single top-level heading" in g
    assert "once" in g                       # shared facts/sources written once
    assert "raw" in g and "never json" in g  # deliverable content stays raw prose
    assert "mid-sentence" in g               # finish each sentence (anti-truncation)


# --------------------------------------------------------------------------- #
# (2) NO THEMATIC DUPLICATE-TAIL — single document title; one pass of each section.
# --------------------------------------------------------------------------- #
def test_s13_enforce_single_h1_demotes_later_titles() -> None:
    """The FIRST ``<h1>`` is kept as the title; every later ``<h1>`` is demoted to
    ``<h2>`` (a section) — content preserved, only the level changes; idempotent."""
    doc = (
        "<h1>The US-Iran Conflict and War News</h1><p>intro.</p>"
        "<h1>Iran War (2026) Conflict Analysis</h1><p>analysis.</p>"
    )
    out = enforce_single_h1(doc)
    assert out.lower().count("<h1") == 1
    assert "<h2>Iran War (2026) Conflict Analysis</h2>" in out
    # no content lost — only a heading level changed.
    assert "intro." in out and "analysis." in out
    # idempotent: a single-title document is byte-identical.
    assert enforce_single_h1(out) == out
    assert enforce_single_h1("<h1>Only Title</h1><p>x.</p>") == "<h1>Only Title</h1><p>x.</p>"


def test_s13_reconcile_kills_two_h1_thematic_duplicate_tail() -> None:
    """The exact B8a2 symptom: a doc with TWO ``<h1>``s (the first node's whole mini-doc
    shell + a later section node's second shell) repeating the Sources / Key-Figures
    blocks. After reconcile: exactly ONE ``<h1>``, the repeated families collapsed to one
    pass each, and the first (grounded) content kept."""
    doc = (
        "<!DOCTYPE html><html><head><title>US-Iran</title></head><body>\n"
        '<nav class="spa-nav"><ul><li><a href="#x">x</a></li></ul></nav>\n'
        "<h1>The US-Iran Conflict and War News</h1>\n"
        "<p>The conflict opened with a strike on Fordow.</p>\n"
        "<h2>Key Figures</h2><p>15 US KIA; 538 wounded.</p>\n"
        '<h2>Sources</h2><ul><li><a href="https://www.warcosts.org/conflicts/iran-2026">warcosts</a></li></ul>\n'
        # --- the SECOND document shell appended as the thematic duplicate-tail ---
        "<h1>Iran War (2026) Conflict Analysis</h1>\n"
        "<p>A second shell re-introduces the report.</p>\n"
        "<h2>Human and Financial Cost</h2><p>Total US cost was about $34 billion.</p>\n"
        "<h2>Key Figures</h2><p>15 US KIA; 538 wounded.</p>\n"
        '<h2>Sources</h2><ul><li><a href="https://www.warcosts.org/conflicts/iran-2026">warcosts</a></li></ul>\n'
        "</body></html>"
    )
    # single_title=True is the sourced deep-research REPORT path (where the B8a2
    # duplicate-tail lives); a multi-page file-delivery doc keeps its per-page titles.
    out = reconcile_doc_structure(doc, single_title=True)
    # exactly ONE top-level title — the two-<h1> defect is gone.
    assert out.lower().count("<h1") == 1
    # the repeated section families collapsed to a single pass each (the reconcile pass
    # stamps an id on each heading, so match the heading family, not the bare tag).
    assert len(re.findall(r"<h2[^>]*>Key Figures</h2>", out)) == 1
    assert len(re.findall(r"<h2[^>]*>Sources</h2>", out)) == 1
    # the first node's grounded content survives; the unique cost section survives.
    assert "The conflict opened with a strike on Fordow." in out
    assert "Total US cost was about $34 billion." in out
    # the second shell's title is preserved as a SECTION, not a second document title.
    assert "Iran War (2026) Conflict Analysis" in out
    # idempotent — a second reconcile pass is a fixed point.
    assert reconcile_doc_structure(out) == out


def test_s13_multipage_two_h1_preserved_when_not_single_title() -> None:
    """BOUNDARY: single-title enforcement is OFF by default, so a legitimate MULTI-PAGE
    document (the file-delivery / plan-chain path, one ``<h1>`` per page) keeps BOTH
    titles — only the sourced deep-research report (``single_title=True``) collapses them."""
    doc = (
        "<!DOCTYPE html><html><body>"
        "<h1>Page 1</h1><p>intro that ends cleanly.</p>"
        "<h1>Page 2</h1><p>more content that ends cleanly.</p>"
        "</body></html>"
    )
    out = reconcile_doc_structure(doc)  # default single_title=False
    assert "<h1>Page 1</h1>" in out and "<h1>Page 2</h1>" in out


def test_s13_clean_single_title_doc_is_unchanged() -> None:
    """A well-formed single-title report with no repeated family is a fixed point of the
    thematic-duplicate-tail passes (no false collapse of distinct sections)."""
    doc = (
        "<h1>Report</h1><p>intro that ends cleanly.</p>"
        "<h2>Economic Impact</h2><p>economy.</p>"
        "<h2>Environmental Impact</h2><p>environment.</p>"
    )
    assert enforce_single_h1(doc) == doc
    assert collapse_duplicate_sections(doc) == doc  # 0.5 overlap < threshold → both kept


# --------------------------------------------------------------------------- #
# (3) NO TRUNCATION MARKER — a mid-sentence num_predict cut is detected and trimmed.
# --------------------------------------------------------------------------- #
def test_s13_has_truncation_marker_detects_mid_sentence_cut() -> None:
    """A substantial document ending mid-sentence (a letter, behind the wrapper closes)
    is flagged; a document ending on sentence-terminating punctuation, a complete figure,
    or a short terse deliverable is NOT (no false positives)."""
    lead = (
        "After the opening strike on Fordow the conflict widened across the region, with "
        "casualties mounting on both sides over the following weeks and months. "
    )
    truncated = (
        "<html><body><h1>T</h1><p>" + lead + "The IRGC killed thousands of "
        "demonstrators in over 100 cities. On February</body></html>"
    )
    assert has_truncation_marker(truncated) is True

    clean = (
        "<html><body><h1>T</h1><p>" + lead + "The IRGC killed thousands of "
        "demonstrators in over 100 cities.</p></body></html>"
    )
    assert has_truncation_marker(clean) is False

    # a trailing complete figure ("130") is not a mid-word truncation.
    figure_end = (
        "<html><body><h1>T</h1><p>" + lead + "Oil rose from seventy-five dollars to "
        "over one hundred and thirty in forty-eight hours: 130</p></body></html>"
    )
    assert has_truncation_marker(figure_end) is False

    # a short, terse deliverable is never faulted (below the visible-text floor).
    assert has_truncation_marker("<p>Headlines</p>") is False


def test_s13_trim_dangling_sentence_removes_short_cutoff() -> None:
    """A SHORT mid-sentence cut-off tail is trimmed back to the last complete sentence,
    the wrapper closes are preserved, and the result carries no truncation marker;
    idempotent and a no-op on a clean document."""
    lead = (
        "After the opening strike on Fordow the conflict widened across the region, with "
        "casualties mounting on both sides over the following weeks and months. "
    )
    truncated = (
        "<html><body><h1>T</h1><p>" + lead + "The IRGC killed thousands of "
        "demonstrators in over 100 cities. On February</body></html>"
    )
    out = trim_dangling_sentence(truncated)
    assert "On February" not in out
    assert "over 100 cities." in out
    assert has_truncation_marker(out) is False
    assert out.lower().rstrip().endswith("</body></html>")
    # idempotent + no-op on the already-clean result.
    assert trim_dangling_sentence(out) == out


def test_s13_trim_leaves_a_long_dangling_block_untouched() -> None:
    """A LARGE dangling block (a whole unfinished section, > max_trim) is NOT silently
    dropped — real content is never lost; the verify/continuation lanes own that case."""
    long_fragment = "word " * 200  # ~1000 chars, no sentence terminator
    doc = "<html><body><h1>T</h1><p>Intro sentence. " + long_fragment + "</body></html>"
    assert trim_dangling_sentence(doc) == doc  # left intact (only flagged, never gutted)


def test_s13_reconcile_strips_final_truncation_marker() -> None:
    """The assembled-doc reconcile pass removes a final-section truncation marker so the
    served document ends cleanly."""
    truncated = (
        "<!DOCTYPE html><html><body><h1>US-Iran</h1>"
        "<p>After the opening strike on Fordow the conflict widened across the region, "
        "with casualties mounting on both sides. The IRGC killed thousands of "
        "demonstrators in over 100 cities. On February"
        "</body></html>"
    )
    out = reconcile_doc_structure(truncated)
    assert has_truncation_marker(out) is False
    assert "over 100 cities." in out
