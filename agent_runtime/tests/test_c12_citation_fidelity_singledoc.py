"""s9/c12 — long-path CITATION FIDELITY (#5) + STRICT single-document (#2b).

Two residual defects the c6 re-run (`.s9_probe/acceptance_matrix_final.md` §3/§7)
left OPEN on the LONG concurrent-multi-node deep-research path:

* **#5 fabricated/placeholder citations (HIGH, d47 central gate).** The research
  nodes fetch REAL source URLs (bbc/aljazeera/cfr/…) and those URLs DO reach the
  synthesizer — but scattered through a huge prompt, buried inside each source's
  2000-char article body. Building section by section the small model could not
  reliably reconstruct them and invented generic ``[CyberWatch Report, 2025]``
  placeholders (or dropped citations). ROOT-CAUSE fix (d17 context-feeding +
  d46/d49 no-fabrication): assemble the ACTUAL fetched URLs the orchestration
  already holds into ONE compact, prominent, cite-ONLY-from-this index and feed it
  to the synthesizer. Real data fed — the model still reasons about placement; NOT
  a hard-coded citation template.
* **#2b stray structural tags (MED).** Each node emits a full ``<!DOCTYPE>…``
  document, so the section-by-section append left stray inline ``<html>``/``<body>``
  OPEN tags at section boundaries + a doubled ``</body></html>`` tail — which the
  ``</html>``-CLOSE-count dedup misses. Fixed by stripping document-wrapper OPENS
  from appended sections + a strict final single-document normaliser.

Same zero-inference FakeTransport-over-a-real-file harness as the c1/c10 tests.
"""
import asyncio
import re

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.runtime import AgentRuntime, SubAgent
from agent_runtime.synth_tools import (
    DONE_SENTINEL,
    collect_fetched_sources,
    enforce_single_html_document,
    has_duplicate_html_structure,
    render_source_index,
    strip_wrapper_openers,
)
from llm_framework import FakeTransport
from reactive_tools import EventPlane, ToolHook, register_agentic_tools


def _run(coro):
    return asyncio.run(coro)


def _hook(tmp_path) -> ToolHook:
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    return hook


_FETCHED = {
    "fetched": [
        {"title": "BBC", "url": "https://bbc.com/world/iran-strike",
         "markdown": "A strike on the facility was reported overnight."},
        {"title": "Al Jazeera", "url": "https://aljazeera.com/news/iran-fallout",
         "markdown": "Regional fallout escalated through the week."},
    ]
}


# --------------------------------------------------------------------------- #
# #5 — source-index helpers (pure)
# --------------------------------------------------------------------------- #
def test_collect_fetched_sources_dedupes_in_order():
    a = {"fetched": [
        {"title": "BBC", "url": "https://bbc.com/x", "markdown": "..."},
        {"title": "CFR", "url": "https://cfr.org/y", "markdown": "..."},
    ]}
    b = {"fetched": [
        {"title": "BBC again", "url": "https://bbc.com/x", "markdown": "..."},  # dup URL
        {"title": "CSIS", "url": "https://csis.org/z", "markdown": "..."},
    ]}
    got = collect_fetched_sources([a, b, "not-a-mapping", {"no": "fetched"}])
    assert got == [
        ("BBC", "https://bbc.com/x"),
        ("CFR", "https://cfr.org/y"),
        ("CSIS", "https://csis.org/z"),
    ]


def test_render_source_index_lists_urls_and_forbids_placeholders():
    block = render_source_index([("BBC", "https://bbc.com/x"), ("", "https://cfr.org/y")])
    assert "https://bbc.com/x" in block and "https://cfr.org/y" in block
    assert "REAL SOURCE URLS" in block
    assert "[Name, 2025]" in block            # the anti-fabrication instruction
    assert "VERBATIM" in block
    # a URL-less title falls back to the URL as its label
    assert "https://cfr.org/y — https://cfr.org/y" in block
    # no sources => no empty stub
    assert render_source_index([]) == ""


# --------------------------------------------------------------------------- #
# #2b — wrapper-open strip + strict single-document (pure)
# --------------------------------------------------------------------------- #
def test_strip_wrapper_openers_keeps_section_content():
    chunk = ("<!DOCTYPE html><html lang='en'><head><title>x</title></head>"
             "<body style='m'><h2>Timeline</h2><p>Day 1.</p></body></html>")
    out = strip_wrapper_openers(chunk)
    assert "<!doctype" not in out.lower()
    assert not re.search(r"<html", out, re.I)
    assert not re.search(r"<head", out, re.I)
    assert not re.search(r"<body(?:\s[^>]*)?>", out, re.I)
    # the real section content survives (only the wrapper opens were removed)
    assert "<h2>Timeline</h2>" in out and "Day 1." in out


def test_enforce_single_html_document_strips_sibling_opens_and_doubled_close():
    # the #2b symptom: stray inline <html>/<body> opens mid-doc + a doubled close tail
    doc = (
        "<!DOCTYPE html><html><head><title>R</title></head><body>"
        "<h1>Report</h1><p>Intro.</p>"
        "<html><body><h2>Timeline</h2><p>Day 1.</p>"   # stray sibling opens
        "</body></html></body></html>"                  # doubled close tail
    )
    assert has_duplicate_html_structure(doc) is True
    fixed = enforce_single_html_document(doc)
    assert fixed.lower().count("<!doctype") == 1
    assert len(re.findall(r"<html(?:\s[^>]*)?>", fixed, re.I)) == 1
    assert len(re.findall(r"<body(?:\s[^>]*)?>", fixed, re.I)) == 1
    assert fixed.lower().count("</html>") == 1
    assert fixed.lower().count("</body>") == 1
    # content from both "sections" survives
    assert "Intro." in fixed and "Timeline" in fixed and "Day 1." in fixed


def test_two_full_documents_reduced_to_first():
    d1 = "<!DOCTYPE html><html><head><title>A</title></head><body><p>first</p></body></html>"
    d2 = "<!DOCTYPE html><html><head><title>B</title></head><body><p>second</p></body></html>"
    assert has_duplicate_html_structure(d1 + d2) is True
    fixed = enforce_single_html_document(d1 + d2)
    assert fixed.lower().count("</html>") == 1
    assert "first" in fixed and "second" not in fixed


def test_single_clean_document_is_unchanged():
    clean = "<!DOCTYPE html><html><head><title>A</title></head><body><p>only</p></body></html>"
    assert has_duplicate_html_structure(clean) is False
    assert enforce_single_html_document(clean) == clean
    # a fragment / markdown is never flagged
    assert has_duplicate_html_structure("<section>x</section>") is False
    assert has_duplicate_html_structure("# Title\n\ntext") is False


# --------------------------------------------------------------------------- #
# #5 — the REAL fetched URLs reach the synthesizer + are cited (integration)
# --------------------------------------------------------------------------- #
def test_source_index_reaches_synthesizer_and_real_url_is_cited(tmp_path):
    captured: dict[str, str] = {}

    def reply(messages, **opts):
        n = sum(1 for m in messages if m.get("role") == "assistant")
        if n == 0:
            captured["user"] = "\n".join(
                str(m.get("content", "")) for m in messages if m.get("role") == "user"
            )
            # the model cites the REAL URL it was handed (no placeholder)
            return (
                "<!DOCTYPE html><html><head><title>US-Iran</title></head><body>"
                "<h1>The 2025 US-Iran Conflict</h1>"
                "<p>A strike was reported "
                "<a href=\"https://bbc.com/world/iran-strike\">BBC</a>.</p>"
                "<h2>Sources</h2><ul>"
                "<li>https://bbc.com/world/iran-strike</li>"
                "<li>https://aljazeera.com/news/iran-fallout</li>"
                "</ul></body></html>"
            )
        return DONE_SENTINEL

    node = PlanNode(
        id="s1", task="Write a detailed HTML report to us-iran.html.", role="synthesizer"
    )
    agent = SubAgent(
        node,
        transport=FakeTransport([reply]),
        hook=_hook(tmp_path),
        overall_goal="Write a detailed HTML report on the 2025 US-Iran conflict; cite sources.",
        upstream_tool_values={"r1": _FETCHED},
        call_opts={"think": False, "temperature": 0},
    )
    out = _run(agent.run({"r1": "research prose summarising the conflict"}))
    assert out is not None

    # the authoritative real-URL index reached the model's user turn
    user = captured["user"]
    assert "REAL SOURCE URLS" in user
    assert "https://bbc.com/world/iran-strike" in user
    assert "https://aljazeera.com/news/iran-fallout" in user
    assert "[Name, 2025]" in user            # the anti-fabrication instruction is present

    # the deliverable cites the REAL URL — no fabricated placeholder
    on_disk = (tmp_path / "us-iran.html").read_text(encoding="utf-8")
    assert "https://bbc.com/world/iran-strike" in on_disk
    assert "[BBC Report, 2025]" not in on_disk
    assert "[Source Name" not in on_disk


def test_no_fetched_sources_no_index_block(tmp_path):
    # a no-source task (a haiku) must NOT get an empty source stub injected.
    captured: dict[str, str] = {}

    def reply(messages, **opts):
        n = sum(1 for m in messages if m.get("role") == "assistant")
        if n == 0:
            captured["user"] = "\n".join(
                str(m.get("content", "")) for m in messages if m.get("role") == "user"
            )
            return "Red leaves drift downward / a quiet stream carries them / autumn says goodbye"
        return DONE_SENTINEL

    node = PlanNode(id="s1", task="Write a haiku to poem.txt.", role="synthesizer")
    agent = SubAgent(
        node,
        transport=FakeTransport([reply]),
        hook=_hook(tmp_path),
        overall_goal="Give me a haiku in a .txt file.",
        upstream_tool_values={},
        call_opts={"think": False, "temperature": 0},
    )
    _run(agent.run({}))
    # the authoritative source-INDEX block (its unique header) is NOT injected
    assert "these are the COMPLETE, exact URLs already fetched" not in captured["user"]


# --------------------------------------------------------------------------- #
# #2b — an appended section that RE-OPENS the wrapper yields ONE clean doc
# --------------------------------------------------------------------------- #
def test_appended_section_wrapper_opens_stripped_single_doc(tmp_path):
    head = (
        "<!DOCTYPE html><html><head><title>US-Iran</title></head><body>"
        "<h1>The 2025 US-Iran Conflict</h1><h2>Overview</h2><p>Intro.</p>"
    )  # wrapper left OPEN (no close) — first section
    sect2 = (
        "<!DOCTYPE html><html><body><h2>Timeline</h2><p>Day 1.</p></body></html>"
    )  # RE-OPENS the wrapper mid-document (the #2b defect trigger)

    def reply(messages, **opts):
        last_user = next(
            (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        n = sum(1 for m in messages if m.get("role") == "assistant")
        if "Append ONLY" in last_user:        # the close-gap continuation
            return "</body></html>"
        if n == 0:
            return head
        if n == 1:
            return sect2
        return DONE_SENTINEL

    rt = AgentRuntime(
        transport=FakeTransport([reply]),
        hook=_hook(tmp_path),
        subagent_call_opts={"think": False, "temperature": 0},
    )
    dag = PlanDAG(
        nodes=[PlanNode(id="s1", task="Write a detailed HTML report to report.html.",
                        role="synthesizer")],
        goal="Write a detailed HTML report on the 2025 US-Iran conflict.",
    )
    out = _run(rt.run(dag))
    assert out.ok

    on_disk = (tmp_path / "report.html").read_text(encoding="utf-8")
    # STRICTLY one top-level structure (was: stray opens + doubled close)
    assert on_disk.lower().count("<!doctype") == 1
    assert len(re.findall(r"<html(?:\s[^>]*)?>", on_disk, re.I)) == 1
    assert len(re.findall(r"<body(?:\s[^>]*)?>", on_disk, re.I)) == 1
    assert on_disk.lower().count("</html>") == 1
    assert on_disk.lower().count("</body>") == 1
    # the second section's CONTENT survived (only its wrapper was stripped)
    assert "Overview" in on_disk and "Timeline" in on_disk and "Day 1." in on_disk
