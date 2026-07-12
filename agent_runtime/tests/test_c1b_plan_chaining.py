"""s9/c1b (d49.4) — PLAN-CHAINING multi-page accumulation in the shared file loop.

c1b adds LARGE / multi-page output via PLAN CHAINING: a write-file plan is ITSELF a
shape (linear OR parallel) whose per-page/section nodes FILL one file. The
decomposition lives in the authored DAG (the page nodes + their ``depends_on`` chain),
NOT in code — the runtime only provides the structural accumulation contract:

  * a WRITER node (synthesis role / ``file_write`` tool) whose UPSTREAM is itself a
    writer CONTINUES that file — it reads the real on-disk tail and APPENDS the next
    page/section (never clobbers the earlier pages);
  * a writer with a DOWNSTREAM writer is NON-final, so it does NOT close the document
    wrapper (the closing-tag gate is deferred to the terminal page) → the assembled
    HTML doc is closed exactly once, at the end.

Both are computed structurally from the DAG topology (``_is_writer_node`` + the
dependents map), so a lone single-file deliverable is byte-for-byte the pre-c1b path
(proven by the unchanged ``test_c1_stepwise_synthesis`` suite). These tests script a
:class:`FakeTransport` over a REAL file hook bound to a tmp sandbox — the whole chain
runs in-process with zero inference, building on c1's loop + c3's decomposed tools.
"""
from __future__ import annotations

import asyncio

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.runtime import AgentRuntime, _is_writer_node
from agent_runtime.synth_tools import (
    DONE_SENTINEL,
    html_close_gap,
)
from llm_framework import FakeTransport
from reactive_tools import EventPlane, ToolHook, register_agentic_tools


def _run(coro):
    return asyncio.run(coro)


def _hook(tmp_path) -> ToolHook:
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    return hook


_CONT_MARK = "earlier pages/sections are ALREADY on the file"


def test_is_writer_node_discriminator():
    assert _is_writer_node(PlanNode(id="a", task="t", role="synthesizer"))
    assert _is_writer_node(PlanNode(id="b", task="t", tool="file_write"))
    assert _is_writer_node(PlanNode(id="c", task="t", tool="write_file"))
    assert not _is_writer_node(PlanNode(id="d", task="t", role="worker"))
    assert not _is_writer_node(PlanNode(id="e", task="t", tool="web_search"))
    assert not _is_writer_node(PlanNode(id="f", task="t"))


# --------------------------------------------------------------------------- #
# A two-page MARKDOWN report: plan2's per-page nodes accumulate ONE file.
# --------------------------------------------------------------------------- #
_MD_P1 = "# Big Report\n\n## Page 1 — Introduction\n\nThe opening section, in full."
_MD_P2 = "## Page 2 — Findings\n\nThe second section, appended after the first."


def test_multipage_markdown_accumulates_across_chained_write_nodes(tmp_path):
    """A linear write-file shape (page1 -> page2) fills one report.md: page2 APPENDS
    onto page1's file (never overwrites), so both sections are present in order."""

    def reply(messages, **opts):
        convo = "\n".join(str(m.get("content") or "") for m in messages)
        n = sum(1 for m in messages if m.get("role") == "assistant")
        page = _MD_P2 if _CONT_MARK in convo else _MD_P1
        return page if n == 0 else DONE_SENTINEL

    dag = PlanDAG(
        nodes=[
            PlanNode(id="p1", task="Write page 1 (Introduction) of report.md",
                     tool="file_write", depends_on=()),
            PlanNode(id="p2", task="Write page 2 (Findings) of report.md",
                     tool="file_write", depends_on=("p1",)),
        ],
        goal="Write a multi-page report to report.md",
    )
    rt = AgentRuntime(transport=FakeTransport([reply]), hook=_hook(tmp_path),
                      max_concurrency=1)
    out = _run(rt.run(dag))
    assert out.ok, out.failed

    written = tmp_path / "report.md"
    assert written.is_file()
    body = written.read_text(encoding="utf-8")
    # BOTH pages present, page1 BEFORE page2 — the second node appended, did not clobber.
    assert _MD_P1 in body and _MD_P2 in body
    assert body.index(_MD_P1) < body.index(_MD_P2)
    # the terminal page's result is the WHOLE assembled file (ground truth).
    assert out.results["p2"].parsed.get("written_path")
    assert _MD_P1 in (out.results["p2"].output or "")


# --------------------------------------------------------------------------- #
# A two-page HTML doc: the wrapper opens on page1, stays open, closes ONCE on the
# final page (the deferred-close contract).
# --------------------------------------------------------------------------- #
_HTML_P1 = "<!DOCTYPE html><html><head><title>Doc</title></head><body><h1>Page 1</h1><p>intro</p>"
_HTML_P2 = "<h1>Page 2</h1><p>more</p></body></html>"


def test_multipage_html_defers_close_to_final_page(tmp_path):
    """Non-final page1 must NOT be nagged to close the wrapper; page2 (final) closes
    it. Result: exactly one well-formed document, both pages present, no double-close."""
    calls: list[dict] = []

    def reply(messages, **opts):
        calls.append(dict(opts))
        convo = "\n".join(str(m.get("content") or "") for m in messages)
        n = sum(1 for m in messages if m.get("role") == "assistant")
        page = _HTML_P2 if _CONT_MARK in convo else _HTML_P1
        return page if n == 0 else DONE_SENTINEL

    dag = PlanDAG(
        nodes=[
            PlanNode(id="h1", task="Write page 1 of doc.html", tool="file_write",
                     depends_on=()),
            PlanNode(id="h2", task="Write page 2 of doc.html", tool="file_write",
                     depends_on=("h1",)),
        ],
        goal="Write a multi-page HTML document to doc.html",
    )
    rt = AgentRuntime(transport=FakeTransport([reply]), hook=_hook(tmp_path),
                      max_concurrency=1)
    out = _run(rt.run(dag))
    assert out.ok, out.failed

    body = (tmp_path / "doc.html").read_text(encoding="utf-8")
    assert "<h1>Page 1</h1>" in body and "<h1>Page 2</h1>" in body
    # closed EXACTLY once at the very end — page1 did not prematurely close the wrapper.
    assert html_close_gap(body) == []
    assert body.count("</html>") == 1 and body.count("</body>") == 1
    assert body.rstrip().endswith("</html>")
    assert all("format" not in c for c in calls)


# --------------------------------------------------------------------------- #
# RP-AUDIT F3 (d319/d341): ``strip_wrapper_closers`` was a DEAD HTML-format-baked
# output-MODIFIER (its MAIN-loop call site was retired by RP-2/d319; no shipped
# caller remained). It has been DELETED from synth_tools along with its unit test
# — the engine strips/rewrites nothing. (test_sf1_reactive_coherence_retired now
# fails if it is re-defined or re-exported.)
# --------------------------------------------------------------------------- #
def test_single_file_writer_is_unaffected_no_premature_close(tmp_path):
    """REGRESSION FLOOR: a lone writer (no upstream/downstream writer) keeps the exact
    pre-c1b behaviour — fresh overwrite, the close-gate fires (final), one document."""
    whole = _HTML_P1 + _HTML_P2

    def reply(messages, **opts):
        n = sum(1 for m in messages if m.get("role") == "assistant")
        return whole if n == 0 else DONE_SENTINEL

    dag = PlanDAG(
        nodes=[PlanNode(id="only", task="Write the page to doc.html",
                        role="synthesizer")],
        goal="Write an HTML document to doc.html",
    )
    rt = AgentRuntime(transport=FakeTransport([reply]), hook=_hook(tmp_path))
    out = _run(rt.run(dag))
    assert out.ok, out.failed
    body = (tmp_path / "doc.html").read_text(encoding="utf-8")
    assert html_close_gap(body) == []
    assert body.count("</html>") == 1
