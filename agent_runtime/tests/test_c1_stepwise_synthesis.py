"""s9/c1 (d49 RE-SCOPE) — SYNTHESIS via PLANNER-IN-THE-LOOP ReAct over REAL files.

The first-pass synthesizer built the deliverable with an in-memory write_section/
finish loop + a structural ``_synthesis_incomplete`` heuristic — UNRELIABLE on E4B.
A measured-on-E4B follow-up showed that asking the small model to EMIT file_write/
file_read JSON tool calls with embedded content ALSO fails (0 parseable calls — the
same escaped-string friction as D1). So the model emits its STRENGTH — RAW content
sections — and the ORCHESTRATION (the planner-in-the-loop) ACTS ON each emission:
it WRITES the section to the real file and READS IT BACK, feeding the actual on-disk
state to the next turn; the model signals completion with ``<<DONE>>`` (judged from
the real file, not memory). Read-back fires unconditionally — even when the model
one-shots the whole document — killing both false-finish and truncation. The CHOSEN
filename+extension reaches disk so the artifact carries the requested type.

These tests script a :class:`FakeTransport` with the synthesizer's RAW emissions over
a REAL file hook bound to a tmp sandbox — the whole loop runs in-process with zero
inference. They prove:

1. the DONE-sentinel splitter + the type-agnostic path derivation (unit);
2. the GENERIC runtime drives the loop, assembling a MULTI-SECTION document on disk
   (no truncation), surfaced as a ``file_write`` result so the artifact name carries
   the chosen extension; NO ``format`` schema is ever sent and temperature is 0;
3. a one-shot emission is still WRITTEN then READ BACK before the model confirms DONE;
4. a DONE issued before anything is written is redirected (never an empty deliverable);
5. when the model emits nothing usable, the RAW fallback salvages a deliverable AND
   persists it to the chosen path; with no hook wired it degrades to a single emission.
"""
from __future__ import annotations

import asyncio

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.runtime import AgentRuntime
from agent_runtime.synth_tools import (
    DONE_SENTINEL,
    derive_output_path,
    deliverable_extension,
    explicit_filename,
    html_close_gap,
    sanitize_write_path,
    split_done_signal,
    unwrap_output_envelope,
)
from llm_framework import FakeTransport
from reactive_tools import EventPlane, ToolHook, register_agentic_tools


def _run(coro):
    return asyncio.run(coro)


def _hook(tmp_path) -> ToolHook:
    """A hook with the real agentic tools; file_write/file_read sandboxed to tmp."""
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    return hook


# --------------------------------------------------------------------------- #
# 1) pure parsing + path derivation (no model)
# --------------------------------------------------------------------------- #
def test_split_done_signal():
    # a plain section is content, not done
    assert split_done_signal("<h1>Hello</h1>") == (False, "<h1>Hello</h1>")
    # the bare sentinel / a DONE-only reply ends the loop with no content
    assert split_done_signal(DONE_SENTINEL) == (True, "")
    assert split_done_signal("  done. ") == (True, "")
    assert split_done_signal("**DONE**") == (True, "")
    # a final chunk followed by a trailing sentinel keeps the chunk
    d, c = split_done_signal("<p>last</p>\n<<DONE>>")
    assert d is True and c == "<p>last</p>"
    # prose that merely mentions 'done' is NOT a completion signal
    assert split_done_signal("The work here is done and dusted, more below.")[0] is False
    assert split_done_signal("") == (False, "")


def test_derive_output_path_is_type_agnostic():
    # an explicit filename the request names survives verbatim (c3r/d49) — the model's own choice
    assert derive_output_path("write cats.html for me", "deliver", None) == "cats.html"
    assert derive_output_path("a goal", "save it to data.csv", None) == "data.csv"
    # RP-1 (d319/d311): format INFERENCE is retired — with NO explicit filename the engine no
    # longer maps a bound writer spec or a request keyword to a format; the neutral plain-text
    # .md extension is used (the model picks its own format by naming its file).
    assert derive_output_path("detailed report", "write it", ["html-writer"]).endswith(".md")
    assert derive_output_path("a report", "write it", ["markdown-writer"]).endswith(".md")
    assert derive_output_path("make an HTML page about cats", "write", None).endswith(".md")
    assert derive_output_path("summarize the news", "write a summary", None).endswith(".md")
    # the stem is a relatable slug, never a generic 'report'/'output'
    assert derive_output_path("US-Iran conflict overview", "write", ["html-writer"]).startswith("us-iran")


def test_unwrap_output_envelope():
    # the bare-node {"output": "..."} leak the live csv case hit → inner deliverable
    csv = "Name,Diameter_km,Moons\nEarth,12756,1\nMars,6779,2"
    assert unwrap_output_envelope('{"output": "' + csv.replace("\n", "\\n") + '"}') == csv
    assert unwrap_output_envelope('{\n"output": "hello"\n}') == "hello"
    # a real deliverable that merely CONTAINS json is untouched
    body = "# Report\nSome prose with a {\"k\": 1} snippet inside."
    assert unwrap_output_envelope(body) == body
    # plain content untouched; an envelope with extra non-output keys is left alone
    assert unwrap_output_envelope("just text") == "just text"
    assert unwrap_output_envelope('{"foo": 1, "bar": 2}') == '{"foo": 1, "bar": 2}'


def test_html_close_gap_detects_unclosed_top_level_tags():
    # the c1r R1 failure: a file ending at </section> with the containers still open
    assert html_close_gap("<html><body><section>x</section>") == ["</body>", "</html>"]
    # a fully-closed document has no gap
    assert html_close_gap("<html><body><section>x</section></body></html>") == []
    # only one container left open → only that closing tag
    assert html_close_gap("<!doctype html><html><body>hi</body>") == ["</html>"]
    # a bare HTML fragment (no top-level container) is NOT faulted — no false nag
    assert html_close_gap("<section>fragment only</section>") == []
    assert html_close_gap("") == []


def test_extension_and_filename_helpers():
    assert explicit_filename("please write cats.html") == "cats.html"
    assert explicit_filename("no file here") is None
    # RP-1 (d319/d311): format INFERENCE is retired — deliverable_extension no longer maps a
    # writer spec or a request keyword to a format; it returns the neutral plain-text default.
    assert deliverable_extension(["html-writer"], "anything") == ".md"
    assert deliverable_extension(None, "give me CSV data") == ".md"
    assert deliverable_extension(None, "plain prose") == ".md"
    # sanitize keeps a model-chosen extension, borrows the default otherwise
    assert sanitize_write_path("cats.html", "report.md") == "cats.html"
    assert sanitize_write_path("my-report", "x.html") == "my-report.html"
    assert sanitize_write_path("", "fallback.md") == "fallback.md"
    # a path is reduced to its basename (the tool sandboxes the directory)
    assert sanitize_write_path("/etc/passwd.txt", "x.md") == "passwd.txt"


# --------------------------------------------------------------------------- #
# 2) GENERIC runtime drives the orchestration ReAct synthesis loop
# --------------------------------------------------------------------------- #
_SECTIONS = [
    "<!DOCTYPE html><html><head><title>US-Iran</title></head><body><h1>Headlines</h1>",
    "<h2>Timeline</h2><ul><li>Day 1: strike</li><li>Day 2: response</li></ul>",
    "<h2>Damage</h2><table><tr><th>Side</th><th>Casualties</th></tr><tr><td>A</td><td>1,190</td></tr></table>",
    "<h2>Sources</h2><ul><li>https://example.com/a</li></ul></body></html>",
]


def _synth_dag(
    task: str = "Write a detailed HTML report on the US-Iran situation to us-iran.html.",
    goal: str = "Write a detailed HTML report on the US-Iran situation.",
) -> PlanDAG:
    return PlanDAG(
        nodes=[PlanNode(id="s1", task=task, role="synthesizer")],
        goal=goal,
    )



# --------------------------------------------------------------------------- #
# AUTONOMY REBUILD P2C — the raw write loop behavior tests below are RETIRED.
# The served raw-content write loop (_run_synthesis/_run_raw_file_loop/
# _run_file_delivery and its riders: stepwise continuation, done-redirect,
# close-continuation, is_detailed_task completeness nudge, re-emission drop,
# csv-tabular rider) is DELETED — every node runs the unified self-select loop
# and AUTHORS its file via file_write; delivery is verified by the
# target-artifact gate (test_target_artifact_gate.py) and duplicate/restart
# writes are governed at the TOOL BOUNDARY (file_write no-clobber refusal).
# --------------------------------------------------------------------------- #
