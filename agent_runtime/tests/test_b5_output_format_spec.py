"""s8/b5 — OUTPUT-FORMAT spec guarantee on the terminal node (fully OFFLINE).

The d13/B2 fix as a STRUCTURAL guarantee (the F5 pattern). When the GOAL names a
deliverable format (HTML / Markdown / a .html/.md file), the terminal write node
MUST carry that format's output-style writer so the deliverable comes back in the
requested form. The PROMPT is the primary lever (factory description + the
per-turn final-step instruction), but E4B intermittently binds the analysis spec
(``research-analyst``) on a "synthesize" node instead of the format writer
(measured ~1/3 of live runs, either direction). So
:meth:`IncrementalPlanner._enforce_output_format_spec` STAMPS the requested
format's writer as the PRIMARY output style on the terminal writer — composing it
ahead of any analysis spec and removing the OTHER format's writer (the two are
mutually exclusive). It is a NO-OP (byte-identical) when the goal names no single
format, the writer spec is unregistered, or the terminal writer already leads with
the right format spec.

These tests exercise the enforcement helper + the goal-format detector directly,
so they run with zero inference.
"""
from __future__ import annotations

from agent_runtime.factory import AbstractPlanFactory
from agent_runtime.incremental import IncrementalPlanner
from llm_framework import FakeTransport
from specialization.registry import SpecRegistry
from specialization.seed import DEEP_RESEARCH_SPEC, seed_canonical_rulesets

_TOOL_CATALOG = [
    {"name": "web_search", "description": "search the web for candidate pages"},
    {"name": "web_fetch", "description": "fetch and extract a page's article text"},
    {"name": "file_write", "description": "write content to a file"},
]


def _planner(tmp_path, *, spec_names=None) -> IncrementalPlanner:
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)  # registers markdown-writer + html-writer + research-analyst
    factory = AbstractPlanFactory(reg.index(), tool_catalog=_TOOL_CATALOG)
    return IncrementalPlanner(
        FakeTransport([]),
        factory,
        spec_names=reg.names() if spec_names is None else spec_names,
        tool_names=[t["name"] for t in _TOOL_CATALOG],
        default_research_spec=DEEP_RESEARCH_SPEC,
    )


class _NullSpan:
    def set_attribute(self, *_a, **_k):  # tracing must never break authoring
        pass


def _node(nid, task, *, tool=None, spec=None, specs=None, role=None, depends_on=()):
    return {"id": nid, "task": task, "tool": tool, "spec": spec,
            "specs": list(specs or []), "needs_spec": None, "role": role,
            "depends_on": list(depends_on)}


# --------------------------------------------------------------------------- #
# goal-format detection
# --------------------------------------------------------------------------- #
def test_requested_output_format_detects_each_format():
    f = IncrementalPlanner._requested_output_format
    assert f("Write a report and produce it as an HTML file.") == ("html", "html-writer")
    assert f("Save the report as a .html document") == ("html", "html-writer")
    assert f("Build a single web page about X") == ("html", "html-writer")
    assert f("Save it as a Markdown (.md) document.") == ("markdown", "markdown-writer")
    assert f("Format the result as markdown") == ("markdown", "markdown-writer")


def test_requested_output_format_noop_on_none_or_both():
    f = IncrementalPlanner._requested_output_format
    assert f("Research the US-Iran conflict and summarise it.") is None  # no format
    assert f("Give me HTML or Markdown, your choice.") is None           # ambiguous → no guess
    assert f("") is None


# --------------------------------------------------------------------------- #
# enforcement helper
# --------------------------------------------------------------------------- #
def test_html_goal_stamps_html_writer_primary_over_research_analyst(tmp_path):
    planner = _planner(tmp_path)
    # The E4B failure: a synthesize sink bound research-analyst, NOT the format writer.
    authored = [
        _node("n1", "gather", tool="web_search", spec="research-analyst"),
        _node("n2", "synthesize report", spec="research-analyst", role="synthesizer",
              depends_on=["n1"]),
    ]
    notes = planner._enforce_output_format_spec(
        authored, "Write a report and produce it as an HTML file.", _NullSpan())
    assert len(notes) == 1 and "n2" in notes[0]
    # html-writer is PRIMARY; the analysis spec is kept as a composed secondary layer.
    assert authored[1]["spec"] == "html-writer"
    assert authored[1]["specs"] == ["html-writer", "research-analyst"]
    # the gather node is untouched (format spec belongs on the writer, not a gather node)
    assert authored[0]["spec"] == "research-analyst"


def test_markdown_goal_stamps_markdown_writer(tmp_path):
    planner = _planner(tmp_path)
    authored = [
        _node("n1", "gather", tool="web_search", spec="research-analyst"),
        _node("n2", "synthesize", spec="research-analyst", role="synthesizer",
              depends_on=["n1"]),
    ]
    notes = planner._enforce_output_format_spec(
        authored, "Save it as a Markdown (.md) document.", _NullSpan())
    assert len(notes) == 1
    assert authored[1]["spec"] == "markdown-writer"
    assert authored[1]["specs"] == ["markdown-writer", "research-analyst"]


def test_noop_when_terminal_already_leads_with_right_format(tmp_path):
    planner = _planner(tmp_path)
    authored = [
        _node("n1", "gather", tool="web_search", spec="research-analyst"),
        _node("n2", "write html", spec="html-writer", depends_on=["n1"]),
    ]
    before = [dict(n) for n in authored]
    notes = planner._enforce_output_format_spec(
        authored, "Produce it as an HTML file.", _NullSpan())
    assert notes == []                  # byte-identical no-op
    assert authored == before


def test_removes_wrong_format_writer_and_stamps_requested(tmp_path):
    planner = _planner(tmp_path)
    # The exact B2 bug: an HTML goal whose terminal node bound the MARKDOWN writer.
    authored = [
        _node("n1", "gather", tool="web_search", spec="research-analyst"),
        _node("n2", "write", spec="markdown-writer", depends_on=["n1"]),
    ]
    notes = planner._enforce_output_format_spec(
        authored, "Produce it as an HTML file.", _NullSpan())
    assert len(notes) == 1
    assert authored[1]["spec"] == "html-writer"
    assert "markdown-writer" not in authored[1]["specs"]  # wrong-format writer removed


def test_promotes_present_but_secondary_format_spec(tmp_path):
    planner = _planner(tmp_path)
    # html-writer present but NOT primary → promote it to primary (it governs form).
    authored = [
        _node("n1", "gather", tool="web_search", spec="research-analyst"),
        _node("n2", "write", specs=["research-analyst", "html-writer"],
              spec="research-analyst", depends_on=["n1"]),
    ]
    notes = planner._enforce_output_format_spec(
        authored, "Produce it as an HTML file.", _NullSpan())
    assert len(notes) == 1
    assert authored[1]["spec"] == "html-writer"
    assert authored[1]["specs"] == ["html-writer", "research-analyst"]


def test_noop_when_goal_names_no_format(tmp_path):
    planner = _planner(tmp_path)
    authored = [
        _node("n1", "gather", tool="web_search", spec="research-analyst"),
        _node("n2", "synthesize", spec="research-analyst", depends_on=["n1"]),
    ]
    before = [dict(n) for n in authored]
    notes = planner._enforce_output_format_spec(
        authored, "Research the US-Iran conflict and summarise it.", _NullSpan())
    assert notes == [] and authored == before


def test_does_not_stamp_a_research_gather_sink(tmp_path):
    planner = _planner(tmp_path)
    # A single research sink (web_search) is NOT a writer → never gets the format spec.
    authored = [_node("n1", "gather", tool="web_search", spec="research-analyst")]
    notes = planner._enforce_output_format_spec(
        authored, "Produce it as an HTML file.", _NullSpan())
    assert notes == []
    assert authored[0]["spec"] == "research-analyst"


def test_noop_when_writer_unregistered(tmp_path):
    # A project without the html-writer seed → the pass cannot stamp it (clean no-op).
    planner = _planner(tmp_path, spec_names=["research-analyst"])
    authored = [
        _node("n1", "gather", tool="web_search", spec="research-analyst"),
        _node("n2", "write", spec="research-analyst", depends_on=["n1"]),
    ]
    before = [dict(n) for n in authored]
    notes = planner._enforce_output_format_spec(
        authored, "Produce it as an HTML file.", _NullSpan())
    assert notes == [] and authored == before
