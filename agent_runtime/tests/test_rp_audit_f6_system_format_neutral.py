"""RP-AUDIT F6 / d352 — the always-present ``IncrementalPlanner._system`` spec-selection
GUIDANCE carries NO per-format EXAMPLE. F2 kept a format-hygiene sentence in ``_system`` that
still PINNED the mapping by example ("an HTML request gets the HTML writer, a Markdown request
the Markdown writer"). The USER decided (d352) to GENERALIZE that last remnant (over accepting
it): the engine prose states the FORMAT-NEUTRAL rule only — bind the writer spec whose ADVERTISED
format matches the goal's requested format — and the format→writer MAPPING lives ONLY in the
writer SPEC descriptions (html-writer / markdown-writer / section-html-writer advertise which
format each produces). This is the d317 "don't pin format in the engine" principle applied to the
last per-format remnant.

These tests are OFFLINE (no inference). They assert:
  • ``_system`` carries NO per-format example / no format pin (the removed literals are gone);
  • the FORMAT-NEUTRAL binding rule is PRESENT;
  • the format→writer MAPPING lives in the WRITER SPEC descriptions (spec-advertised);
  • the format-bleed HYGIENE rule (never a document-format writer on a gather node) stays intact;
  • no format/shape/spec-name FLOW conditional was introduced in incremental.py (anti-fab
    d310/d311/d319/d341/d278);
  • NOT-REGRESSED: a format goal still binds the right writer (html-writer) on the write node and
    leaves the gather node format-free, driven through the REAL parse path (mirrors F4).
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re

from agent_runtime import incremental as incremental_mod
from agent_runtime.factory import AbstractPlanFactory
from agent_runtime.incremental import IncrementalPlanner
from specialization.registry import SpecRegistry
from specialization.seed import CANONICAL_RULESETS, seed_canonical_rulesets
from llm_framework import FakeTransport


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _run(coro):
    return asyncio.run(coro)


_TOOL_CATALOG = [
    {"name": "web_search", "description": "search the web"},
    {"name": "web_fetch", "description": "fetch a page"},
    {"name": "file_write", "description": "write a file"},
    {"name": "send_mail", "description": "send an email"},
]


def _planner(tmp_path, *, shape_name, methodology, replies=()) -> IncrementalPlanner:
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)
    factory = AbstractPlanFactory(reg.index(), tool_catalog=_TOOL_CATALOG)
    return IncrementalPlanner(
        FakeTransport(list(replies)),
        factory,
        spec_names=reg.names(),
        tool_names=[t["name"] for t in _TOOL_CATALOG],
        shape_name=shape_name,
        shape_description="d",
        shape_decompose_methodology=methodology,
    )


def _seed():
    return json.dumps({"tool": "seed_plan", "args": {}})


def _add(task, *, tool="", spec="", depends_on=()):
    return json.dumps(
        {"tool": "add_step", "args": {"task": task, "tool": tool, "spec": spec,
                                      "specs": [], "depends_on": list(depends_on)}}
    )


def _finalize():
    return json.dumps({"tool": "finalize_plan", "args": {}})


# The per-format EXAMPLE / format-pin phrasings that must NO LONGER appear in engine prose.
_PER_FORMAT_EXAMPLE = (
    "an html request gets the html writer",
    "a markdown request the markdown writer",
    "the html writer for html",
    "the markdown writer for markdown",
    "html/markdown writer",
    "names an output format (html",
    "a .html/.md file",
    "bind the output-style spec for that format",
)


# --------------------------------------------------------------------------- #
# The _system guidance carries NO per-format example (format is not pinned in engine prose).
# --------------------------------------------------------------------------- #
def test_system_carries_no_per_format_example(tmp_path):
    p = _planner(tmp_path, shape_name="linear", methodology="")
    sysmsg = _norm(p._system("Research the US-Iran situation and write it to report.html."))
    for banned in _PER_FORMAT_EXAMPLE:
        assert banned not in sysmsg, f"per-format example leaked into _system: {banned!r}"


def test_system_states_the_format_neutral_binding_rule(tmp_path):
    """Neutral does NOT mean empty — the engine still directs the model to bind the correct
    writer, generically: match the goal's REQUESTED format to the writer's ADVERTISED format."""
    p = _planner(tmp_path, shape_name="linear", methodology="")
    sysmsg = _norm(p._system("Do the thing and write it out."))
    assert "advertised format matches the goal's requested format" in sysmsg
    assert "only on the final write step" in sysmsg
    # It tells the model to READ the writer spec descriptions (the mapping's real home).
    assert "read the writer specializations' own descriptions" in sysmsg


def test_system_keeps_format_bleed_hygiene(tmp_path):
    """The generic format-bleed HYGIENE rule stays: a research/gather node NEVER carries a
    document-format writer spec (kept format-neutral — no html/markdown pin)."""
    p = _planner(tmp_path, shape_name="linear", methodology="")
    sysmsg = _norm(p._system("Do the thing and write it out."))
    assert "never a document-format spec" in sysmsg
    assert "never bind a document-format writer to a gather step" in sysmsg
    # a research/gather step takes a research spec or NONE (the hygiene framing survives).
    assert "research-analyst" in sysmsg


# --------------------------------------------------------------------------- #
# The format→writer MAPPING lives in the WRITER SPEC descriptions (spec-advertised).
# --------------------------------------------------------------------------- #
def test_format_to_writer_mapping_is_spec_advertised():
    html_desc = _norm(CANONICAL_RULESETS["html-writer"][0])
    md_desc = _norm(CANONICAL_RULESETS["markdown-writer"][0])
    sec_desc = _norm(CANONICAL_RULESETS["section-html-writer"][0])
    # Each writer ADVERTISES which format it produces (so the model matches requested→advertised).
    assert "html" in html_desc and "not markdown" in html_desc
    assert "markdown" in md_desc and "not html" in md_desc
    assert "html" in sec_desc
    # …and each stays WRITE-node-only (the hygiene, carried by the spec, not the engine).
    for desc in (html_desc, md_desc, sec_desc):
        assert "never bind it to a research/gather/analysis node" in desc


# --------------------------------------------------------------------------- #
# ANTI-FAB — no format / shape-name / spec-name FLOW conditional in incremental.py (d341/d278).
# --------------------------------------------------------------------------- #
def test_incremental_module_has_no_format_or_name_flow_conditional():
    src = inspect.getsource(incremental_mod)
    for pat in (
        r"==\s*['\"]html-writer['\"]",
        r"==\s*['\"]markdown-writer['\"]",
        r"==\s*['\"]section-html-writer['\"]",
        r"==\s*['\"]html['\"]",
        r"==\s*['\"]markdown['\"]",
        r"if\s+[^\n]*\bshape_name\b[^\n]*==",
        r"if\s+[^\n]*\bspec\b[^\n]*==\s*['\"](html|markdown)",
    ):
        assert re.search(pat, src) is None, f"format/shape/spec-name flow conditional in incremental.py: {pat!r}"


# --------------------------------------------------------------------------- #
# NOT-REGRESSED — with the per-format example REMOVED, a format goal still binds the right
# writer on the write node and leaves the gather node format-free (real parse path; mirrors F4).
# --------------------------------------------------------------------------- #
def test_format_authoring_still_binds_html_writer_on_write_node(tmp_path):
    replies = [
        _seed(),
        _add("Search the US-Iran news", tool="web_search", depends_on=[]),
        _add("Write the HTML report to report.html", tool="file_write",
             spec="html-writer", depends_on=["n1"]),
        _finalize(),
    ]
    p = _planner(tmp_path, shape_name="linear", methodology="", replies=replies)
    dag = _run(p.plan("Research the US-Iran situation and write it to an HTML file report.html.")).dag
    by = dag.by_id
    assert len(dag.nodes) == 2
    # gather node stays format-free (no document-format spec bled onto it)…
    assert by["n1"].spec in (None, "")
    # …and the WRITE node carries the html-writer output-style spec.
    assert by["n2"].tool == "file_write"
    assert by["n2"].spec == "html-writer"
    assert by["n2"].depends_on == ("n1",)
