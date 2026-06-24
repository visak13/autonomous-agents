"""s13 / B5 (design §4B) — DOC-STRUCTURE INTEGRITY BACKSTOP via a real parser.

``reconcile_doc_structure`` is the FINAL deterministic pass over the fully-assembled
HTML deliverable: a real ``html.parser`` scan re-derives the table of contents from
the actual final ``<h1>``/``<h2>``/``<h3>`` set (so a section appended AFTER the
build-time nav is navigable — the d93 late-section ToC miss), renames any duplicate
element ``id``, and balances the wrapper. It generates NO content (d48/d60-clean) and
is idempotent.

The backstop must not just exist — it must FIRE on the served file-save report. So the
ToC and duplicate-id tests drive the REAL served route end-to-end: a ``FakeTransport``
emits the assembled document, the shared raw-file loop (:meth:`_run_raw_file_loop`)
writes it to a real file hook and finalizes it, and the assertions read the document
off DISK exactly as it is served. The wrapper-repair and idempotency tests exercise the
reconcile function directly on the kind of bytes that route produces.
"""
from __future__ import annotations

import asyncio

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.runtime import AgentRuntime
from agent_runtime.synth_tools import (
    DONE_SENTINEL,
    html_close_gap,
    reconcile_doc_structure,
)
from llm_framework import FakeTransport
from reactive_tools import EventPlane, ToolHook, register_agentic_tools


def _run(coro):
    return asyncio.run(coro)


def _hook(tmp_path) -> ToolHook:
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    return hook


def _served_html(tmp_path, emitted: str, filename: str = "report.html") -> str:
    """Drive the REAL served file-save route once and return the on-disk document.

    A single ``file_write`` writer node: the FakeTransport emits ``emitted`` on the
    first turn and ``<<DONE>>`` on the next, so the loop writes the document, reads it
    back, and runs the full finalization chain (ending in the reconcile backstop)."""

    def reply(messages, **opts):
        n = sum(1 for m in messages if m.get("role") == "assistant")
        return emitted if n == 0 else DONE_SENTINEL

    dag = PlanDAG(
        nodes=[
            PlanNode(id="w", task=f"Write the report to {filename}",
                     tool="file_write", depends_on=()),
        ],
        goal=f"Write an HTML report to {filename}",
    )
    rt = AgentRuntime(transport=FakeTransport([reply]), hook=_hook(tmp_path),
                      max_concurrency=1)
    out = _run(rt.run(dag))
    assert out.ok, out.failed
    return (tmp_path / filename).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# (i) A LATE-APPENDED section enters the ToC on the SERVED assembly (d93).
# --------------------------------------------------------------------------- #
# The assembled doc carries a nav built BEFORE the final section ("Conclusion") was
# appended — so the ToC lists Introduction + Background but not Conclusion. The served
# finalization must route through reconcile so the delivered ToC lists ALL sections.
_LATE_SECTION_DOC = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Report</title></head>
<body>
<nav class="spa-nav">
  <ul>
    <li class="toc-h2"><a href="#introduction">Introduction</a></li>
    <li class="toc-h2"><a href="#background">Background</a></li>
  </ul>
</nav>
<h1 id="report-title">Report</h1>
<h2 id="introduction">Introduction</h2><p>The opening section.</p>
<h2 id="background">Background</h2><p>Context for the work.</p>
<h2>Conclusion</h2><p>The late-appended final section.</p>
</body></html>"""


def test_s13_late_section_enters_toc_on_served_assembly(tmp_path):
    served = _served_html(tmp_path, _LATE_SECTION_DOC)
    nav = served[served.lower().index("<nav"): served.lower().index("</nav>")]
    # the late "Conclusion" heading is now in the re-derived ToC (slugged anchor)...
    assert 'href="#conclusion"' in nav
    assert ">Conclusion<" in nav
    # ...alongside the headings that were already present.
    assert 'href="#introduction"' in nav and 'href="#background"' in nav
    # the body section itself is preserved (no content fabricated or dropped).
    assert "The late-appended final section." in served
    assert "<h2 id=\"conclusion\">Conclusion</h2>" in served
    # exactly one nav (the stale partial one was replaced, not duplicated).
    assert served.lower().count("<nav") == 1


# --------------------------------------------------------------------------- #
# (ii) A DUPLICATE element id is renamed on the SERVED assembly.
# --------------------------------------------------------------------------- #
_DUP_ID_DOC = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Report</title></head>
<body>
<h1 id="overview">Overview</h1>
<figure id="figures"><figcaption>Chart A</figcaption></figure>
<p>Discussion of the data.</p>
<div id="figures"><figcaption>Chart B</figcaption></div>
</body></html>"""


def test_s13_dup_id_figures_renamed_on_served_assembly(tmp_path):
    served = _served_html(tmp_path, _DUP_ID_DOC)
    # the collision is resolved: the first keeps id="figures", the second is renamed.
    assert served.count('id="figures"') == 1
    assert 'id="figures-2"' in served
    # both elements survive (rename only, no element dropped).
    assert "Chart A" in served and "Chart B" in served


# --------------------------------------------------------------------------- #
# (iii) A MALFORMED (unclosed) wrapper is closed by the reconcile pass.
# --------------------------------------------------------------------------- #
_MALFORMED_WRAPPER = (
    '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>R</title></head>'
    '<body>\n<h1 id="t">Title</h1>\n<h2>Body</h2><p>Some real content.</p>'
)  # NOTE: no </body></html> — a truncated emission.


def test_s13_malformed_wrapper_closed(tmp_path):
    out = reconcile_doc_structure(_MALFORMED_WRAPPER)
    # the wrapper is now balanced (the deterministic close-gap repair fired)...
    assert html_close_gap(out) == []
    assert out.count("</body>") == 1 and out.count("</html>") == 1
    assert out.rstrip().endswith("</html>")
    # ...and the real content was preserved, never truncated.
    assert "Some real content." in out
    assert "<h1 id=\"t\">Title</h1>" in out


def test_s13_reconcile_is_idempotent(tmp_path):
    once = reconcile_doc_structure(_LATE_SECTION_DOC)
    twice = reconcile_doc_structure(once)
    assert once == twice  # a clean, complete-ToC, unique-id document is a fixed point.
    assert html_close_gap(once) == []


def test_s13_non_html_returned_unchanged(tmp_path):
    md = "# A markdown report\n\nNo HTML structure here.\n"
    assert reconcile_doc_structure(md) == md
    assert reconcile_doc_structure("") == ""
