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


def test_runtime_writes_real_file_section_by_section_no_format_temp0(tmp_path):
    seen_opts: list[dict] = []

    def reply(messages, **opts):
        seen_opts.append(dict(opts))
        n = sum(1 for m in messages if m.get("role") == "assistant")
        if n < len(_SECTIONS):
            return _SECTIONS[n]
        return DONE_SENTINEL

    transport = FakeTransport([reply])
    # think=True like the deep-research role nodes; temp set non-zero to prove the
    # synthesis path FORCES temp=0 regardless of the inherited call opts.
    rt = AgentRuntime(
        transport=transport, hook=_hook(tmp_path),
        subagent_call_opts={"think": True, "temperature": 0.4},
    )
    out = _run(rt.run(_synth_dag()))

    assert out.ok
    syn = out.results["s1"]
    doc = syn.output or ""
    # EVERY section assembled in full on disk — the long document did NOT truncate
    for sec in _SECTIONS:
        assert sec in doc, f"missing section: {sec[:40]}"
    assert doc.strip().endswith("</body></html>")

    # the REAL file was written with the CHOSEN extension, and equals the output
    written = tmp_path / "us-iran.html"
    assert written.is_file()
    assert written.read_text(encoding="utf-8") == doc

    # surfaced as a file_write result so the chat artifact carries the .html name
    assert syn.tool_used == "file_write"
    assert syn.tool_value["path"].replace("\\", "/").endswith("us-iran.html")
    assert isinstance(syn.parsed, dict) and syn.parsed.get("output") == doc
    assert syn.parsed.get("written_path")

    # STRUCTURAL guarantee: no synthesis call ever carried a format schema, and the
    # write node ran deterministic (temp=0, d35) — never the traced temp=0.4.
    assert seen_opts, "no synthesis calls recorded"
    assert all("format" not in o for o in seen_opts)
    assert all(o.get("temperature") == 0 for o in seen_opts)


def test_one_shot_emission_is_written_then_read_back(tmp_path):
    # The model writes the WHOLE document in one turn, then confirms DONE — the loop
    # still WRITES it and READS IT BACK before accepting completion (read-back fires).
    whole = "".join(_SECTIONS)

    def reply(messages, **opts):
        n = sum(1 for m in messages if m.get("role") == "assistant")
        return whole if n == 0 else DONE_SENTINEL

    transport = FakeTransport([reply])
    rt = AgentRuntime(transport=transport, hook=_hook(tmp_path),
                      subagent_call_opts={"think": False, "temperature": 0})
    out = _run(rt.run(_synth_dag()))

    assert out.ok
    doc = out.results["s1"].output or ""
    assert doc == whole
    assert (tmp_path / "us-iran.html").read_text(encoding="utf-8") == whole
    assert out.results["s1"].tool_used == "file_write"


def test_done_before_any_write_is_redirected(tmp_path):
    # The model says DONE before writing anything; the loop redirects it to write
    # first (an empty deliverable is never acceptable).
    def reply(messages, **opts):
        n = sum(1 for m in messages if m.get("role") == "assistant")
        if n == 0:
            return DONE_SENTINEL  # premature → redirected
        if n == 1:
            return "# Brief\nThe full brief body with the key facts."
        return DONE_SENTINEL

    transport = FakeTransport([reply])
    rt = AgentRuntime(transport=transport, hook=_hook(tmp_path),
                      subagent_call_opts={"think": False, "temperature": 0})
    out = _run(rt.run(_synth_dag("Write a brief to brief.md.")))

    assert out.ok
    doc = out.results["s1"].output or ""
    assert "the key facts" in doc.lower()
    assert (tmp_path / "brief.md").is_file()
    assert out.results["s1"].tool_used == "file_write"


def test_raw_fallback_when_nothing_usable_persists_to_chosen_path(tmp_path):
    # The model emits nothing usable in the loop (blank turns) → the loop gives up,
    # the RAW fallback salvages a deliverable AND persists it to the chosen path.
    prose = "Here is the complete plain prose answer."

    def reply(messages, **opts):
        last_user = [m["content"] for m in messages if m.get("role") == "user"][-1]
        if "Write the COMPLETE deliverable now" in last_user:
            return prose  # the fallback prompt
        return "   "       # blank loop turns

    transport = FakeTransport([reply])
    rt = AgentRuntime(transport=transport, hook=_hook(tmp_path),
                      subagent_call_opts={"think": False, "temperature": 0})
    out = _run(rt.run(_synth_dag("Write notes to notes.txt.")))

    assert out.ok
    syn = out.results["s1"]
    assert prose in (syn.output or "")
    assert (tmp_path / "notes.txt").is_file()
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8").strip() == prose.strip()
    assert syn.tool_used == "file_write"


def test_r1_unclosed_html_gets_one_close_continuation_before_done(tmp_path):
    # R1 (c1r): the model emits an HTML body then says DONE while the top-level tags
    # are STILL OPEN. The orchestration reads the REAL file, finds the gap, and sends
    # ONE "append only the closing tags" continuation instead of accepting the finish;
    # the document ends correctly closed.
    def reply(messages, **opts):
        last_user = [m["content"] for m in messages if m.get("role") == "user"][-1]
        if "missing the closing" in last_user:
            return "</body></html>\n" + DONE_SENTINEL  # the close-continuation
        n = sum(1 for m in messages if m.get("role") == "assistant")
        if n == 0:
            # unclosed: <html>/<body> opened, never closed, yet the model signals DONE
            return "<!DOCTYPE html><html><body><h1>Title</h1><p>body</p>\n" + DONE_SENTINEL
        return DONE_SENTINEL

    transport = FakeTransport([reply])
    rt = AgentRuntime(transport=transport, hook=_hook(tmp_path),
                      subagent_call_opts={"think": False, "temperature": 0})
    out = _run(rt.run(_synth_dag("Write an HTML report to report.html.")))

    assert out.ok
    syn = out.results["s1"]
    doc = syn.output or ""
    # the false-finish was caught: the real file is now closed
    assert doc.strip().endswith("</body></html>")
    assert html_close_gap(doc) == []
    written = tmp_path / "report.html"
    assert written.read_text(encoding="utf-8").strip().endswith("</body></html>")
    assert syn.parsed.get("converged") is True


def test_r2_non_convergence_is_surfaced_not_silently_finished(tmp_path):
    # R2 (c1r): the model writes a section every turn but NEVER signals DONE. The loop
    # bounds the churn at the ceiling and SURFACES non-convergence (converged=False) —
    # it must not silently report a clean finish, and the written content is preserved.
    def reply(messages, **opts):
        return "<p>another small part</p>"  # endless content, never <<DONE>>

    transport = FakeTransport([reply])
    rt = AgentRuntime(transport=transport, hook=_hook(tmp_path),
                      subagent_call_opts={"think": False, "temperature": 0})
    out = _run(rt.run(_synth_dag("Write an HTML report to churn.html.")))

    assert out.ok
    syn = out.results["s1"]
    # non-convergence is SURFACED on the parsed result (and the trace span), not hidden
    assert syn.parsed.get("converged") is False
    # the deliverable is still the real on-disk content (not empty / not a stub)
    assert "another small part" in (syn.output or "")
    assert (tmp_path / "churn.html").is_file()


def test_c8_md_detailed_first_turn_done_gets_one_completeness_continuation(tmp_path):
    # c8 R2: on the markdown/text path the R1 HTML close-gap gate is silent (no tags to
    # balance), so a DETAILED task that the model one-shots + <<DONE>> in a SINGLE turn
    # — dropping a requested section + the sources list — would be accepted as-is. The
    # orchestration detects the detailed intent + first-turn finish, reads the REAL
    # file, and sends ONE completeness continuation (remaining parts + a SOURCES list)
    # before accepting. No content/citation template — the model decides what is still
    # missing from what it actually wrote (d48/d49).
    first = (
        "# US-Iran Report\n\nThe US struck Iranian facilities; 24 reported killed "
        "(reuters.com).\n\n## Timeline\n- 18 June: airstrikes.\n" + DONE_SENTINEL
    )
    rest = (
        "\n## Fallout\nThe EU called for an immediate ceasefire; Brent could top $100 "
        "with Strait of Hormuz disruption (bbc.com).\n\n## Sources\n"
        "- https://www.reuters.com/world/x\n- https://www.bbc.com/news/y\n" + DONE_SENTINEL
    )

    def reply(messages, **opts):
        last_user = [m["content"] for m in messages if m.get("role") == "user"][-1]
        if "report is not complete in one pass" in last_user:
            return rest  # the completeness continuation
        n = sum(1 for m in messages if m.get("role") == "assistant")
        return first if n == 0 else DONE_SENTINEL

    transport = FakeTransport([reply])
    rt = AgentRuntime(transport=transport, hook=_hook(tmp_path),
                      subagent_call_opts={"think": False, "temperature": 0})
    out = _run(rt.run(_synth_dag(
        "Write a DETAILED report on the US-Iran situation to report.md.",
        goal="Write a detailed report on the US-Iran situation.",
    )))

    assert out.ok
    syn = out.results["s1"]
    doc = syn.output or ""
    # the premature single-turn finish was caught: the dropped fallout + the SOURCES
    # list (with FULL URLs) landed on the real file
    assert "Fallout" in doc and "ceasefire" in doc.lower()
    assert "## Sources" in doc
    assert "https://www.bbc.com/news/y" in doc
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == doc
    assert syn.parsed.get("converged") is True


def test_c8_md_simple_first_turn_done_accepted_in_one_turn(tmp_path):
    # d46 guard: a SIMPLE (non-detailed) task answered in one turn is NOT nagged to
    # expand — the completeness gate only fires on detailed-intent tasks, so
    # "headlines -> headlines", never forced into paragraphs. Proven by the model being
    # asked EXACTLY ONCE (no continuation turn).
    once = "- Strike on Iran\n- 24 reported killed\n- Oil up 4%\n" + DONE_SENTINEL
    calls = {"n": 0}

    def reply(messages, **opts):
        calls["n"] += 1
        n = sum(1 for m in messages if m.get("role") == "assistant")
        return once if n == 0 else DONE_SENTINEL

    transport = FakeTransport([reply])
    rt = AgentRuntime(transport=transport, hook=_hook(tmp_path),
                      subagent_call_opts={"think": False, "temperature": 0})
    out = _run(rt.run(_synth_dag(
        "Give me the headlines as a bullet list to headlines.md.",
        goal="Give me the latest US-Iran headlines.",
    )))

    assert out.ok
    syn = out.results["s1"]
    doc = syn.output or ""
    assert "Strike on Iran" in doc
    # accepted in ONE WRITE turn — the completeness gate did NOT fire (no second write emit).
    # d242: call 1 is the raw loop's SELF-SELECT front (the node loads its bundles), call 2 is
    # the single converged write turn; the gate firing again would show as a 3rd call.
    assert calls["n"] == 2
    assert syn.parsed.get("converged") is True


def test_no_hook_degrades_to_single_raw_emission():
    # No tool hook wired (offline unit caller) → a single raw emission, no file write.
    prose = "A concise conversational answer, no file."

    def reply(messages, **opts):
        return prose

    transport = FakeTransport([reply])
    rt = AgentRuntime(transport=transport, subagent_call_opts={"think": False, "temperature": 0})
    out = _run(rt.run(_synth_dag("Just answer in chat.")))

    assert out.ok
    syn = out.results["s1"]
    assert prose in (syn.output or "")
    assert isinstance(syn.parsed, dict) and syn.parsed.get("written_path") is None
    assert syn.tool_used in (None, "")
