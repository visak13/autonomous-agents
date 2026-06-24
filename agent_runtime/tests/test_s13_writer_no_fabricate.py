"""s13 / FX-writer (d106 #6, #7) — OUTLINE-AS-PRIMARY backstop + EMPTY-NODE-NO-FABRICATE.

The B8a live run surfaced two writer-side quality defects:

  #7  The agent outline landed as a SECOND, parallel section set appended after the
      conclusion (three conflicting "Section 3"s) instead of being the PRIMARY scaffold.
  #6  A research node that fetched 0 sources (B1) still got a section, which the writer
      fabricated from memory (the Timeline of invented dated events).

This file covers the deterministic, drop-only structural pieces that live in
``synth_tools``: ``collapse_outline_duplicate_sections`` (the #7 backstop that collapses a
later heading duplicating an outline section an earlier heading already wrote) and the
``UNSUPPORTED_SECTION_INSTRUCTION`` text the #6 flag stamps onto a no-source section. Both
are d48/d60-clean — they read the real bytes and fabricate NOTHING. The empty-outline and
clean-doc no-op cases prove the d56 fallback and idempotency are preserved.
"""
from __future__ import annotations

from agent_runtime.synth_tools import (
    UNSUPPORTED_SECTION_INSTRUCTION,
    collapse_duplicate_sections,
    collapse_outline_duplicate_sections,
)

_OUTLINE = [
    {"title": "Introduction and Scope", "covers": "overview"},
    {"title": "Cost and Damage Assessment", "covers": "B2, B4"},
    {"title": "Geopolitical Analysis", "covers": "B3"},
]


def _headings(doc: str) -> list[str]:
    import re

    return re.findall(r"<h[12][^>]*>(.*?)</h[12]>", doc, re.IGNORECASE | re.DOTALL)


def test_outline_duplicate_tail_is_collapsed():
    """The B8a defect: a findings-driven set THEN the same outline sections as a tail.

    "Cost and Damage Assessment" and "Geopolitical Analysis" each appear twice (the second
    pass is the appended duplicate tail). The later occurrence of each outline slot is
    dropped; the first grounded occurrence is kept, and a section the outline does not name
    is left alone."""
    doc = (
        "<h1>Introduction and Scope</h1><p>intro</p>"
        "<h2>Cost and Damage Assessment</h2><p>real figures</p>"
        "<h2>Geopolitical Analysis</h2><p>real analysis</p>"
        "<h2>Conclusion</h2><p>wrap up</p>"
        # --- the appended duplicate tail (the parallel outline set) ---
        "<h2>Cost and Damage Assessment</h2><p>repeated</p>"
        "<h2>Geopolitical Analysis</h2><p>repeated</p>"
    )
    out = collapse_outline_duplicate_sections(doc, _OUTLINE)
    heads = [h.strip() for h in _headings(out)]
    assert heads == [
        "Introduction and Scope",
        "Cost and Damage Assessment",
        "Geopolitical Analysis",
        "Conclusion",
    ]
    # each outline section appears exactly once; "Conclusion" (not in the outline) survives
    assert out.count("Cost and Damage Assessment") == 1
    assert out.count("Geopolitical Analysis") == 1
    assert "repeated" not in out
    assert "real figures" in out and "real analysis" in out


def test_drifted_outline_duplicate_still_collapsed():
    """A duplicate that drifted in wording but still matches the SAME outline slot.

    "Cost & Damage Assessment Details" matches the outline's "Cost and Damage Assessment"
    family, so the later, drifted re-write is recognised as the same planned section and
    dropped (the family-only collapse would miss this because the two body headings differ).
    """
    doc = (
        "<h1>Cost and Damage Assessment</h1><p>first grounded pass</p>"
        "<h2>Geopolitical Analysis</h2><p>analysis</p>"
        "<h2>Cost and Damage Assessment Details</h2><p>second drifted pass</p>"
    )
    out = collapse_outline_duplicate_sections(doc, _OUTLINE)
    assert "first grounded pass" in out
    assert "second drifted pass" not in out


def test_section_not_in_outline_is_kept():
    """Conservative: a heading matching NO outline slot is never dropped (real new section)."""
    doc = (
        "<h1>Cost and Damage Assessment</h1><p>a</p>"
        "<h2>An Entirely Unrelated Appendix</h2><p>b</p>"
    )
    out = collapse_outline_duplicate_sections(doc, _OUTLINE)
    assert "An Entirely Unrelated Appendix" in out
    assert out == doc  # no two headings share a slot → byte-identical no-op


def test_empty_outline_is_noop_d56_fallback():
    """d56 fallback: an empty/absent outline leaves the findings-driven doc UNCHANGED."""
    doc = "<h1>Section A</h1><p>x</p><h2>Section B</h2><p>y</p>"
    assert collapse_outline_duplicate_sections(doc, []) == doc
    assert collapse_outline_duplicate_sections(doc, None) == doc


def test_clean_doc_following_outline_is_noop():
    """A doc that already follows the outline (one section per entry) is untouched."""
    doc = (
        "<h1>Introduction and Scope</h1><p>a</p>"
        "<h2>Cost and Damage Assessment</h2><p>b</p>"
        "<h2>Geopolitical Analysis</h2><p>c</p>"
    )
    assert collapse_outline_duplicate_sections(doc, _OUTLINE) == doc


def test_family_collapse_and_outline_collapse_compose():
    """The two passes compose: family-collapse kills identical re-emissions, the outline
    pass then kills the drifted same-slot duplicate the family pass left behind."""
    doc = (
        "<h1>Cost and Damage Assessment</h1><p>first</p>"
        "<h2>Cost and Damage Assessment</h2><p>identical reemit</p>"
        "<h2>Geopolitical Analysis</h2><p>geo</p>"
        "<h2>Cost & Damage Assessment Overview</h2><p>drifted reemit</p>"
    )
    after_family = collapse_duplicate_sections(doc)
    out = collapse_outline_duplicate_sections(after_family, _OUTLINE)
    # the Cost outline slot survives exactly once; both duplicate re-emissions are gone
    assert len([h for h in _headings(out) if "cost" in h.lower()]) == 1
    assert "identical reemit" not in out
    assert "drifted reemit" not in out
    assert "first" in out and "geo" in out


def test_unsupported_instruction_is_anti_fabrication():
    """The #6 flag text must mark UNSUPPORTED and explicitly forbid fabrication."""
    text = UNSUPPORTED_SECTION_INSTRUCTION
    assert "UNSUPPORTED" in text
    lowered = text.lower()
    assert "no supporting sources" in lowered
    assert "do not invent" in lowered
    # names the kinds of content that must NOT be fabricated (the B8a Timeline case)
    for token in ("dates", "figures", "timelines", "citations"):
        assert token in lowered
