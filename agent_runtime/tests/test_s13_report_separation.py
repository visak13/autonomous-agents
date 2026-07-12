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
from agent_runtime.synth_tools import has_truncation_marker
from reactive_tools import ToolInputError
from reactive_tools.file_tools import guard_write_content

# RP-1 (d319/d311): the thematic-duplicate-tail passes (enforce_single_h1,
# collapse_duplicate_sections, reconcile_doc_structure) are RETIRED engine structure-fixing —
# their tests are removed, as is trim_dangling_sentence (it EDITED the model's output). The
# DATA-vs-HTML separation guidance + the DETECT-ONLY truncation-marker predicate are KEPT below.


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


def test_s13_trim_dangling_sentence_stays_retired() -> None:
    """RP-1 (d319/d311): ``trim_dangling_sentence`` EDITED the model's output and is
    RETIRED — self-policing that no future change reintroduces it."""
    import agent_runtime.synth_tools as st

    assert not hasattr(st, "trim_dangling_sentence")
    assert not hasattr(st, "repair_table_cells")
