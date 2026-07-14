"""s14 / a8 (d149) — the per-section ANCHORED-REVIEW leak fix, proven FAST + OFFLINE.

ROOT CAUSE (measured from the a7 trace): the write-phase DAG normaliser stamps
``tool=file_write`` onto EVERY write node INCLUDING the framework-injected review nodes
(``*_review`` / ``final_review``). In :meth:`SubAgent.run` the ``file_write`` WRITER branch
was checked BEFORE the anchored-review branch, so a review node ran the raw-content writer
loop with the REVIEW task and physically WROTE its ``file_update(...)`` tool-call text into the
deliverable (21 file_update + 1 file_read literals in run1). Two further leaks: a writer
echoed its USER-turn scaffolding (``SOURCES & FINDINGS FROM PRIOR STEP``, ``TOOL OUTPUT
(file_write): {path}``, the ACROSS-PARTS continuation note) into the document.

The fix, proven here without a GPU or network:

* (routing) a review node is route-classified FIRST and routed to the bounded anchored-edit
  reviewer (its tool calls EXECUTE, never render) — even when it carries the normaliser's
  ``tool=file_write``; it never enters the raw writer loop.
* (dispatch) the anchored-review loop dispatches a tool call emitted as TEXT via the same
  balanced-brace string fallback the research loop uses, so a live-E4B-style text call EXECUTES.
* (scaffolding) ``strip_internal_scaffolding`` removes any internal scaffolding / tool-call
  text from a raw emission BEFORE it is written, so none can reach the deliverable.
"""
from __future__ import annotations

import asyncio

from llm_framework import FakeTransport

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.runtime import AgentRuntime
from agent_runtime.synth_tools import DONE_SENTINEL, strip_internal_scaffolding
from reactive_tools import EventPlane, ToolHook, register_agentic_tools


def _run(coro):
    return asyncio.run(coro)


def _hook(tmp_path) -> ToolHook:
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    return hook


_SOURCES = [
    {"title": "BBC", "url": "https://www.bbc.com/news/x", "source_trust": "secondary",
     "key_claims": ["$200bn requested"], "markdown": "The Pentagon requested $200bn."},
]

# the exact span the writer puts on disk and the reviewer fixes (a real, matchable anchor).
_CLAIM = "<p>The Pentagon is asking for another $200bn in funding for the war.</p>"
_FIXED = "<p>The Pentagon is asking for another $200bn in funding (Source: https://www.bbc.com/news/x).</p>"


def _routing_dag() -> PlanDAG:
    """n1 WRITES report.html; n1_review (stamped tool=file_write like the normaliser) reviews."""
    return PlanDAG(
        nodes=[
            PlanNode(id="n1", task="Write the report to report.html.", tool="file_write"),
            PlanNode(
                id="n1_review",
                task=("Review the output of the step 'Write the report'. The deliverable is "
                      "ALREADY written to the file — fix any unsupported claim in place."),
                tool="file_write",                    # the normaliser stamp that caused the leak
                depends_on=("n1",),
            ),
        ],
        rationale="r",
        goal="Write a report to report.html.",
    )


def _routing_transport() -> FakeTransport:
    """Writer one-shots the claim; the REVIEW turn emits a file_update as TEXT (a JSON string,
    not a native tool_call) — exactly how live E4B with think=True surfaces the call."""
    def reply(messages, **opts):
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
        # REVIEW turns: the anchored-review intro is review-only.
        if "The deliverable is ALREADY written to the file" in user:
            # d242 TRUE self-select: the reviewer loads the 'file' bundle FIRST, then edits.
            if n_assistant == 0:
                return '{"tool": "get_bundles", "args": {"name": "file"}}'
            if n_assistant == 1:
                # a TEXT (string-JSON) file_update — must be parsed+dispatched, never rendered.
                return ('{"tool": "file_update", "args": {"old": ' + _json(_CLAIM)
                        + ', "new": ' + _json(_FIXED) + '}}')
            return "DONE"
        # WRITER turns: emit the claim once, then confirm.
        if n_assistant == 0:
            return "<!DOCTYPE html><html><body>" + _CLAIM + "</body></html>"
        return DONE_SENTINEL
    return FakeTransport([reply])


def _json(s: str) -> str:
    import json
    return json.dumps(s)


# AUTONOMY REBUILD P2: test_writer_scaffolding_never_reaches_the_file RETIRED — it exercised the deleted raw write loop /
# deliverable_path routing (write nodes now run the unified self-select pull-writer;
# see test_sb6_write_fold_antifab.py::test_write_route_has_no_flag_every_worker_takes_the_unified_loop).


def test_strip_internal_scaffolding_unit():
    """The deterministic sanitizer drops scaffolding/tool-call text (whole-line, mid-line, and
    orphan multi-line args) while keeping real content verbatim."""
    raw = (
        "<h2>Cost Analysis</h2>\n"
        "<li>Q3 Tension: [URL Placeholder]file_update(\n"
        'old="x", new="y")\n'
        '            <td>The overall death toll is estimated by Hrana\n'
        "\n"
        "SOURCES & FINDINGS FROM PRIOR STEP n2 (use this content directly):\n"
        "TOOL OUTPUT (file_write):\n"
        "{'path': 'C:/proj/report.html'}\n"
        "A document is being written ACROSS PARTS; earlier pages are on the file.\n"
        "<p>Real paragraph about energy markets.</p>"
    )
    out = strip_internal_scaffolding(raw)
    for leak in ("file_update", "SOURCES & FINDINGS", "TOOL OUTPUT", "ACROSS PARTS",
                 "{'path'", "old=", "new="):
        assert leak not in out, f"leaked: {leak!r}"
    assert "Cost Analysis" in out
    assert "The overall death toll is estimated by Hrana" in out
    assert "Real paragraph about energy markets." in out
    assert "Q3 Tension: [URL Placeholder]" in out          # the placeholder text before the cut survives
