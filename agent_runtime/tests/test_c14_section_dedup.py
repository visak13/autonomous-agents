"""s9/c14 — SECTION/REPORT de-duplication (d59): collapse a re-emitted body-level
report pass / repeated heading-FAMILY to EXACTLY ONE pass of each section.

The c13 SPA-per-section synthesis fixed citation fidelity + the SWA depth ceiling, but
the long write loop still OVER-PRODUCES: it re-emits ``<header>/<nav>/<h1>`` and the
whole section sequence again WITHOUT a fresh ``<!DOCTYPE>`` (c13r 4/4 long runs: 2–3
full passes; "Public Law History" ×3). ``enforce_single_html_document`` only dedups
duplicate DOCUMENT wrappers, so these body-level re-emissions slip past it.

``collapse_duplicate_sections`` is the bounded, model-agnostic, d48/d60-clean structural
fix: it segments at the ``<h1>``/``<h2>`` headings, keeps the FIRST (grounded,
source-scoped) occurrence of each heading FAMILY, and drops every later re-emission —
judged by heading FAMILY (significant-token overlap), NOT string equality, because the
wording DRIFTS across passes. ``section_reemission`` is the loop-side guard the ReAct
write loop uses to stop re-appending an already-written section set.

These tests cover the pure structural core (path-independent), mirroring the real c13r
``run1.html`` defect shape (drifting duplicate headings + a real source URL that must
survive the collapse, so the c13 citation win is never dropped).
"""
from agent_runtime.synth_tools import (
    assemble_html_spa,
    collapse_duplicate_sections,
    enforce_single_html_document,
    section_reemission,
    top_level_html_doc_count,
)


# A compact fixture in the SHAPE of c13r run1.html: a full report pass, then a SECOND
# pass whose headings DRIFT in wording (the c13r gotcha) — same sections, re-emitted.
# The grounded source URL lives in the FIRST Economic section and MUST survive.
_GROUNDED_URL = "https://www.visionofhumanity.org/the-hidden-price-of-the-iran-war/"
_DUP_REPORT = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>US-Iran</title></head>
<body>
    <header>
        <h1>The 2025 US-Iran Conflict: A Detailed Report</h1>
    </header>
    <nav><ul><li><a href="#overview">Overview</a></li></ul></nav>
    <div class="section" id="overview">
        <h2>Conflict Overview: Escalation Dynamics</h2>
        <p>The 2025 conflict escalated rapidly after the February strikes.</p>
    </div>
    <div class="section" id="military">
        <h2>Military Losses and Casualties Assessment</h2>
        <p>Both sides reported significant materiel losses.</p>
    </div>
    <div class="section" id="economic">
        <h2>Economic and Damage Assessment</h2>
        <p>Brent crude rose above $100 (source: {_GROUNDED_URL}).</p>
    </div>
    <div class="section" id="sources">
        <h2>Sources and Citations</h2>
        <ul><li><a href="{_GROUNDED_URL}">Vision of Humanity</a></li></ul>
    </div>
    <header>
        <h1>The 2025 US-Iran Conflict: A Detailed Report</h1>
    </header>
    <nav><ul><li><a href="#overview">Overview</a></li></ul></nav>
    <div class="section" id="overview">
        <h2>Conflict Overview: Escalation Dynamics and Context</h2>
        <p>A second, re-emitted pass of the same overview.</p>
    </div>
    <div class="section" id="military">
        <h2>Military Losses and Casualties</h2>
        <p>A re-emitted military section with drifted wording.</p>
    </div>
    <div class="section" id="economic">
        <h2>Economic and Damage Assessment: The Parallel War</h2>
        <p>A re-emitted economic section.</p>
    </div>
    <div class="section" id="sources">
        <h2>Sources</h2>
        <ul><li><a href="{_GROUNDED_URL}">Vision of Humanity</a></li></ul>
    </div>
</body>
</html>"""


def _families(doc: str) -> list[str]:
    """The visible text of each kept <h2> (lowercased), for assertion readability."""
    import re

    return [
        re.sub(r"<[^>]+>", "", m).strip().lower()
        for m in re.findall(r"<h2[^>]*>(.*?)</h2>", doc, flags=re.IGNORECASE | re.DOTALL)
    ]


# --------------------------------------------------------------------------- #
# 1) the core fix — a re-emitted body-level pass collapses to ONE of each section
# --------------------------------------------------------------------------- #
def test_collapse_removes_reemitted_body_level_pass():
    out = collapse_duplicate_sections(_DUP_REPORT)
    h2s = _families(out)
    # EXACTLY one of each section (the four families), despite the wording drift.
    assert len(h2s) == 4, h2s
    assert sum("overview" in h for h in h2s) == 1
    assert sum("military" in h for h in h2s) == 1
    assert sum("economic" in h for h in h2s) == 1
    assert sum("sources" in h for h in h2s) == 1
    # Only ONE <h1> and ONE <nav> survive (the duplicate header/nav pass is gone).
    assert out.lower().count("<h1") == 1
    assert out.lower().count("<nav") == 1


def test_collapse_keeps_first_grounded_occurrence_citation_win():
    out = collapse_duplicate_sections(_DUP_REPORT)
    # The grounded source URL (the c13 citation win) MUST survive the collapse.
    assert _GROUNDED_URL in out
    # The FIRST (grounded) Economic heading is the one kept, not the drifted re-emission.
    assert "Economic and Damage Assessment</h2>" in out
    assert "The Parallel War" not in out


def test_collapse_balances_wrapper_when_tail_section_dropped():
    # The LAST section ("Sources") is a duplicate and is dropped together with the
    # trailing </body></html>; html_close_gap must re-add exactly one closing pair.
    out = collapse_duplicate_sections(_DUP_REPORT)
    assert out.lower().count("</body>") == 1
    assert out.lower().count("</html>") == 1
    assert top_level_html_doc_count(out) == 1


def test_collapse_judges_by_family_not_string_equality():
    # "…Casualties Assessment" and "…Casualties" are DIFFERENT strings but the SAME
    # section family — string-equality would miss the second; family-overlap collapses it.
    out = collapse_duplicate_sections(_DUP_REPORT)
    mil = [h for h in _families(out) if "military" in h]
    assert mil == ["military losses and casualties assessment"]


# --------------------------------------------------------------------------- #
# 2) safety — no over-collapse, idempotent, no-op on clean / non-HTML input
# --------------------------------------------------------------------------- #
def test_clean_single_pass_report_is_unchanged():
    clean = (
        "<!DOCTYPE html><html><head><title>x</title></head><body>"
        "<h2>Overview</h2><p>a</p>"
        "<h2>Timeline</h2><p>b</p>"
        "<h2>Sources</h2><p>c</p>"
        "</body></html>"
    )
    assert collapse_duplicate_sections(clean) == clean


def test_distinct_sections_sharing_one_generic_word_are_not_collapsed():
    doc = (
        "<body><h2>Economic Impact</h2><p>a</p>"
        "<h2>Environmental Impact</h2><p>b</p></body>"
    )
    out = collapse_duplicate_sections(doc)
    assert "Economic Impact" in out and "Environmental Impact" in out


def test_numbered_pages_are_distinct_families_not_collapsed():
    # A legit multi-page chain ("Page 1"/"Page 2") must NOT collapse — the trailing
    # digit is the distinguishing token (regression for the c1b plan-chaining path).
    doc = (
        "<body><h1>Page 1</h1><p>intro</p>"
        "<h1>Page 2</h1><p>more</p></body>"
    )
    out = collapse_duplicate_sections(doc)
    assert "<h1>Page 1</h1>" in out and "<h1>Page 2</h1>" in out
    existing = "<h1>Page 1</h1><p>intro</p>"
    assert section_reemission("<h1>Page 2</h1><p>more</p>", existing) is False


def test_collapse_is_idempotent():
    once = collapse_duplicate_sections(_DUP_REPORT)
    twice = collapse_duplicate_sections(once)
    assert once == twice


def test_collapse_noop_on_fragment_and_non_html():
    assert collapse_duplicate_sections("just some markdown text") == "just some markdown text"
    assert collapse_duplicate_sections("") == ""
    one = "<body><h2>Only Section</h2><p>x</p></body>"
    assert collapse_duplicate_sections(one) == one


def test_enumerated_legal_headings_collapse():
    # The c13r legal defect: the four fair-use factors / a Public-Law section repeated,
    # carrying roman-numeral enumerators that must not defeat family matching.
    doc = (
        "<body>"
        "<h2>III. The Four Factors of Fair Use</h2><p>first</p>"
        "<h2>IV. Public Law History and Context</h2><p>first</p>"
        "<h2>V. The Four Factors of Fair Use</h2><p>repeat</p>"
        "<h2>VI. Public Law History and Context</h2><p>repeat</p>"
        "</body>"
    )
    out = collapse_duplicate_sections(doc)
    h2s = _families(out)
    assert len(h2s) == 2, h2s


# --------------------------------------------------------------------------- #
# 3) pipeline order — collapse THEN assemble yields a single-pass navigable SPA
# --------------------------------------------------------------------------- #
def test_collapse_then_assemble_produces_single_pass_spa():
    deduped = collapse_duplicate_sections(enforce_single_html_document(_DUP_REPORT))
    spa = assemble_html_spa(deduped, title="US-Iran")
    assert top_level_html_doc_count(spa) == 1
    assert spa.lower().count("<h1") == 1
    # one of each section family in the assembled SPA body
    assert len(_families(spa)) == 4
    # the grounded source survives the full pipeline
    assert _GROUNDED_URL in spa


# --------------------------------------------------------------------------- #
# 4) the loop-side guard — section_reemission
# --------------------------------------------------------------------------- #
def test_section_reemission_true_when_all_families_already_written():
    existing = "<h2>Overview</h2><p>a</p><h2>Military Losses</h2><p>b</p>"
    chunk = "<h2>Conflict Overview: Escalation</h2><p>again</p>"
    assert section_reemission(chunk, existing) is True


def test_section_reemission_false_when_a_new_section_appears():
    existing = "<h2>Overview</h2><p>a</p>"
    chunk = "<h2>Overview</h2><p>x</p><h2>Brand New Timeline</h2><p>new</p>"
    assert section_reemission(chunk, existing) is False


def test_section_reemission_false_for_plain_prose_and_empty_file():
    assert section_reemission("<p>just more prose, no heading</p>", "<h2>X</h2>") is False
    assert section_reemission("<h2>Overview</h2>", "") is False
