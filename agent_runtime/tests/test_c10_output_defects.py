"""s9/c10 — synthesizer OUTPUT defects #2/#3/#4 (from the c6 acceptance matrix).

These prove the three fixes, each through the SAME FakeTransport-over-a-real-file
harness the c1 stepwise-synthesis tests use (zero inference):

* **#2 DUPLICATE-DOCUMENT (HIGH).** A small model nudged to "continue" an
  ALREADY-complete, closed HTML document re-emits a FRESH ``<!DOCTYPE>…</html>``
  document; appended, the file held TWO complete documents concatenated (tag-BALANCE
  passed it — 2 opens + 2 closes look "balanced"). Fixed by a re-emission guard (drop
  a fresh-document chunk when the file already holds a complete closed one) PLUS a
  final-assembly single-document gate (dedupe to the first complete document and
  rewrite the file) for the two-docs-in-one-emission case.
* **#3 ``.txt`` not honored (MED).** The extension regex ``\\b\\.txt\\b`` failed when a
  space preceded ``.txt`` ("save to a .txt file" → ``.md``). Fixed in ``_ext_for``
  (shared by ``derive_output_path`` and ``deliverable_extension``).
* **#4 CSV trailing prose (LOW).** The "keep going / close with a SOURCES section
  listing URLs" continuation nudge injected ``Source 1: https://`` prose INTO the
  ``.csv``. Fixed by treating a CSV as single-shot tabular-only (no detailed/sources
  nudge).
"""
import asyncio

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.runtime import AgentRuntime
from agent_runtime.synth_tools import (
    DONE_SENTINEL,
    begins_html_document,
    dedupe_html_documents,
    deliverable_extension,
    derive_output_path,
    top_level_html_doc_count,
)
from llm_framework import FakeTransport
from reactive_tools import EventPlane, ToolHook, register_agentic_tools


def _run(coro):
    return asyncio.run(coro)


def _hook(tmp_path) -> ToolHook:
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    return hook


_FULL_DOC_1 = (
    "<!DOCTYPE html><html><head><title>US-Iran</title></head><body>"
    "<h1>The 2025 US-Iran Conflict</h1><h2>Timeline</h2><p>Day 1: strike.</p>"
    "<h2>Sources</h2><ul><li>https://example.com/a</li></ul></body></html>"
)
_FULL_DOC_2 = (
    "<!DOCTYPE html><html><head><title>US-Iran</title></head><body>"
    "<h1>The 2025 US-Iran Conflict</h1><h2>Analysis</h2><p>Different second pass.</p>"
    "</body></html>"
)


# --------------------------------------------------------------------------- #
# #2 — single-document helpers (pure)
# --------------------------------------------------------------------------- #
def test_top_level_html_doc_count():
    assert top_level_html_doc_count(_FULL_DOC_1) == 1
    assert top_level_html_doc_count(_FULL_DOC_1 + _FULL_DOC_2) == 2
    # a fragment / markdown / csv has no </html> → 0 (never flagged)
    assert top_level_html_doc_count("<section>x</section>") == 0
    assert top_level_html_doc_count("name,moons\nEarth,1") == 0
    assert top_level_html_doc_count("") == 0


def test_dedupe_html_documents_keeps_first():
    # two concatenated docs → keep ONLY the first complete one
    deduped = dedupe_html_documents(_FULL_DOC_1 + _FULL_DOC_2)
    assert deduped == _FULL_DOC_1
    assert top_level_html_doc_count(deduped) == 1
    # a single document (or fragment) is returned unchanged
    assert dedupe_html_documents(_FULL_DOC_1) == _FULL_DOC_1
    assert dedupe_html_documents("<section>frag</section>") == "<section>frag</section>"


def test_begins_html_document():
    assert begins_html_document("  <!DOCTYPE html><html>...") is True
    assert begins_html_document("<html lang='en'>...") is True
    assert begins_html_document("<h2>Next section</h2>") is False
    assert begins_html_document("name,moons\nEarth,1") is False
    assert begins_html_document("") is False


# --------------------------------------------------------------------------- #
# #2 — the loop: re-emission of a fresh document is DROPPED (one doc on disk)
# --------------------------------------------------------------------------- #
def _html_dag(task: str, goal: str = "Write a detailed HTML report on US-Iran.") -> PlanDAG:
    return PlanDAG(nodes=[PlanNode(id="s1", task=task, role="synthesizer")], goal=goal)


def test_reemitted_full_document_is_dropped_single_doc_on_disk(tmp_path):
    # turn0: a COMPLETE closed document; turn1 (after the continuation nudge): a SECOND
    # fresh full document (the defect trigger); turn2: DONE. The re-emission guard must
    # recognise the file already holds a complete closed doc and STOP — so the second
    # document never lands and the file holds EXACTLY ONE top-level document.
    def reply(messages, **opts):
        n = sum(1 for m in messages if m.get("role") == "assistant")
        if n == 0:
            return _FULL_DOC_1
        if n == 1:
            return _FULL_DOC_2  # re-emission of a whole new document
        return DONE_SENTINEL

    transport = FakeTransport([reply])
    rt = AgentRuntime(transport=transport, hook=_hook(tmp_path),
                      subagent_call_opts={"think": True, "temperature": 0})
    out = _run(rt.run(_html_dag("Write a detailed HTML report to us-iran.html.")))

    assert out.ok
    doc = out.results["s1"].output or ""
    written = tmp_path / "us-iran.html"
    assert written.is_file()
    # EXACTLY ONE top-level document reached disk (the defect was 2)
    assert top_level_html_doc_count(written.read_text(encoding="utf-8")) == 1
    assert top_level_html_doc_count(doc) == 1
    assert "Different second pass" not in doc  # the duplicate was dropped


def test_two_documents_in_one_emission_are_deduped_and_rewritten(tmp_path):
    # The model crams TWO complete documents into a SINGLE emission (the guard cannot
    # pre-empt this — written_path is None on the first write). The final single-document
    # gate must dedupe to the first AND rewrite the real file so the artifact is clean.
    def reply(messages, **opts):
        n = sum(1 for m in messages if m.get("role") == "assistant")
        return (_FULL_DOC_1 + _FULL_DOC_2) if n == 0 else DONE_SENTINEL

    transport = FakeTransport([reply])
    rt = AgentRuntime(transport=transport, hook=_hook(tmp_path),
                      subagent_call_opts={"think": False, "temperature": 0})
    out = _run(rt.run(_html_dag("Write the report to combined.html.")))

    assert out.ok
    doc = out.results["s1"].output or ""
    on_disk = (tmp_path / "combined.html").read_text(encoding="utf-8")
    assert top_level_html_doc_count(doc) == 1
    assert top_level_html_doc_count(on_disk) == 1   # the FILE was rewritten, not just the chat
    assert doc == on_disk
    assert "Different second pass" not in on_disk


# --------------------------------------------------------------------------- #
# #3 — the .txt extension is honored (was downgraded to .md)
# --------------------------------------------------------------------------- #
def test_txt_extension_is_honored():
    # the exact phrasings the c6 matrix proved failing → now .txt
    for text in ("save to a .txt file", "give me a .txt file", "a plain text file",
                 "txt file please", "save as text file"):
        assert deliverable_extension(None, text) == ".txt", text
    # no regression to the other formats / the .md default
    assert deliverable_extension(None, "a detailed markdown report") == ".md"
    assert deliverable_extension(None, "give me a CSV of planets") == ".csv"
    assert deliverable_extension(None, "an HTML report") == ".html"
    assert deliverable_extension(None, "just answer the question") == ".md"
    # the full path derivation carries the .txt through (no content-derived .md)
    assert derive_output_path("write a haiku", "save it to a .txt file", None).endswith(".txt")


# --------------------------------------------------------------------------- #
# #4 — a CSV is tabular-only (the SOURCES nudge no longer fires)
# --------------------------------------------------------------------------- #
def test_csv_is_tabular_only_no_trailing_prose(tmp_path):
    # turn0: clean CSV rows. turn1 WOULD be the prose+sources the old continuation nudge
    # provoked — but a CSV is single-shot, so the loop accepts after the rows and never
    # asks for more. The file must be tabular-only (no "Source 1: https://" tail).
    rows = "Planet,Moons\nMercury,0\nVenus,0\nEarth,1\nMars,2\nJupiter,95"
    prose = "\nThe distribution of moons varies widely. Source 1: https://example.com/x"

    def reply(messages, **opts):
        n = sum(1 for m in messages if m.get("role") == "assistant")
        if n == 0:
            return rows
        if n == 1:
            return prose  # the trailing-prose defect — must never be requested/written
        return DONE_SENTINEL

    transport = FakeTransport([reply])
    rt = AgentRuntime(transport=transport, hook=_hook(tmp_path),
                      subagent_call_opts={"think": False, "temperature": 0})
    dag = PlanDAG(
        nodes=[PlanNode(id="s1", task="Give me a CSV of the first 5 planets to planets.csv.",
                        role="synthesizer")],
        goal="Give me a CSV of the first 5 planets and their moons.",
    )
    out = _run(rt.run(dag))

    assert out.ok
    on_disk = (tmp_path / "planets.csv").read_text(encoding="utf-8")
    assert "Mercury,0" in on_disk and "Jupiter,95" in on_disk   # the real rows are there
    assert "Source 1" not in on_disk                            # NO trailing prose
    assert "distribution of moons" not in on_disk
    assert out.results["s1"].output.strip() == rows
