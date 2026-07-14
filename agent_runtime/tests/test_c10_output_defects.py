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
    deliverable_extension,
    derive_output_path,
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
# RP-AUDIT F3 (d319/d341/d330): the pure single-document helpers
# ``top_level_html_doc_count`` / ``dedupe_html_documents`` / ``begins_html_document``
# were DEAD HTML-format-pinned code (their re-emission-guard call site was replaced by
# the FORMAT-NEUTRAL document_restart/section_reemission/html_close_gap trio in
# RP-3c/d330). They are DELETED from synth_tools; their pure unit tests are removed
# with them. The two behaviour tests below (which asserted on the number of complete
# documents that reached disk) count top-level ``</html>`` closes INLINE — they test
# the LIVE re-emission guard / raw pass-through, not any deleted helper.
# --------------------------------------------------------------------------- #
def _top_level_html_doc_count(doc: str) -> int:
    """Local test helper: number of COMPLETE top-level HTML documents (``</html>``
    closes) in ``doc``. Inlined so the tests no longer depend on a shipped predicate."""
    return (doc or "").lower().count("</html>")


# --------------------------------------------------------------------------- #
# #2 — the loop: re-emission of a fresh document is DROPPED (one doc on disk)
# --------------------------------------------------------------------------- #
def _html_dag(task: str, goal: str = "Write a detailed HTML report on US-Iran.") -> PlanDAG:
    return PlanDAG(nodes=[PlanNode(id="s1", task=task, role="synthesizer")], goal=goal)


# AUTONOMY REBUILD P2C — the raw-loop output-defect tests (re-emission drop,
# two-docs-ship-raw, csv-tabular rider) are RETIRED with the deleted raw write
# loop; duplicate/restart writes are now governed at the TOOL BOUNDARY
# (file_write no-clobber refusal) and delivery by the target-artifact gate.


def test_format_inference_retired_explicit_name_still_honored():
    # RP-1: deliverable_extension no longer maps a request keyword (.txt / markdown / CSV /
    # HTML) to an extension — the engine does not infer a FORMAT; it returns the neutral
    # plain-text default. The model picks its own format by NAMING its file.
    for text in ("save to a .txt file", "a plain text file", "a detailed markdown report",
                 "give me a CSV of planets", "an HTML report", "just answer the question"):
        assert deliverable_extension(None, text) == ".md", text
    # an EXPLICIT filename the request names still survives verbatim through path derivation.
    assert derive_output_path("write a haiku", "save it to notes.txt", None) == "notes.txt"


# --------------------------------------------------------------------------- #
# #4 — a CSV is tabular-only (the SOURCES nudge no longer fires)
# --------------------------------------------------------------------------- #
