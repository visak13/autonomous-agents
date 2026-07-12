"""RP-AUDIT F4 / d341 — the planner FACTORY_DESCRIPTION carries ONLY the node schema +
the generic spec/tool SELECTION PRINCIPLE; it bakes NO scheduling recipe and NO
output-format-BINDING recipe. Those two authoring recipes used to ride in engine prose on
EVERY goal (the description is consumed by the legacy one-shot planner AND prepended into the
incremental authorer's ``_system``), duplicating definition-layer doctrine — the exact d341
hazard (engine prose baking a fixed flow/format that ignores the selected shape/spec).

FIX (self-policed here): thin the description; the doctrine lives in the DEFINITION LAYER —
  • the SCHEDULE-ONLY recipe in ``schedule-leg.toml``'s ``decompose_methodology`` (RP-4c), which
    the incremental authorer substitutes WITH PRECEDENCE; and
  • the OUTPUT-FORMAT-BINDING doctrine in the WRITER SPEC descriptions (html-writer /
    markdown-writer advertise which format they produce) the planner reasons over, plus the
    format-hygiene rule F2 keeps in the incremental ``_system`` guidance (NOT touched here).

These tests are OFFLINE (no inference). They assert the description is thinned (the two baked
recipes are gone), that the legitimate generic guidance + F2's format-bleed hygiene rule stay,
that the removed doctrine actually lives in the shapes/specs, and that no shape-name/spec-name
flow conditional was introduced (anti-fab d341/d278/d319).
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re

from agent_runtime import factory as factory_mod
from agent_runtime.factory import AbstractPlanFactory, FACTORY_DESCRIPTION
from agent_runtime.incremental import IncrementalPlanner
from agent_runtime.shapes import load_shape
from llm_framework import FakeTransport
from specialization.registry import SpecRegistry
from specialization.seed import CANONICAL_RULESETS, seed_canonical_rulesets


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _run(coro):
    return asyncio.run(coro)


_DRIVE_TOOLS = [
    {"name": "cron_add", "description": "schedule a recurring task"},
    {"name": "web_search", "description": "search the web"},
    {"name": "web_fetch", "description": "fetch a page"},
    {"name": "file_write", "description": "write a file"},
]


def _drive_planner(tmp_path, *, shape_name, methodology, replies) -> IncrementalPlanner:
    """An IncrementalPlanner over the (now-thinned) FACTORY_DESCRIPTION, so a drive through the
    real parse path proves the thinning did not regress authoring."""
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)
    factory = AbstractPlanFactory(reg.index(), tool_catalog=_DRIVE_TOOLS)  # uses FACTORY_DESCRIPTION
    return IncrementalPlanner(
        FakeTransport(list(replies)),
        factory,
        spec_names=reg.names(),
        tool_names=[t["name"] for t in _DRIVE_TOOLS],
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


# --------------------------------------------------------------------------- #
# The description is thinned — NO baked scheduling recipe, NO baked format-binding recipe.
# --------------------------------------------------------------------------- #
def test_description_bakes_no_scheduling_recipe():
    d = _norm(FACTORY_DESCRIPTION)
    # The SCHEDULE-ONLY authoring recipe (cron_add node / recurring-scheduler / cadence framing)
    # must NOT live in engine prose — it lives in schedule-leg.toml's decompose_methodology.
    for banned in (
        "cron_add",
        "recurring-scheduler",
        "schedule-only",
        "purely recurring",
        "every morning",
        "daily at 8am",
    ):
        assert banned not in d, f"baked scheduling recipe leaked into FACTORY_DESCRIPTION: {banned!r}"


def test_description_bakes_no_output_format_binding_recipe():
    d = _norm(FACTORY_DESCRIPTION)
    # The "if the goal names HTML ⇒ bind the HTML writer" recipe must NOT live in engine prose —
    # it lives in the writer SPEC descriptions the planner reasons over.
    for banned in (
        "output format: when the goal names",
        "bind the output-style spec for that format",
        "an html request gets the html writer",
        "a markdown request the markdown writer",
        "never substitute a different format's writer",
    ):
        assert banned not in d, f"baked output-format-binding recipe leaked into FACTORY_DESCRIPTION: {banned!r}"


# --------------------------------------------------------------------------- #
# The legitimate generic guidance + F2's format-bleed hygiene rule STAY (thin ≠ empty).
# --------------------------------------------------------------------------- #
def test_description_keeps_node_schema_and_selection_principle():
    d = _norm(FACTORY_DESCRIPTION)
    # Node schema (what a plan/node is).
    assert "decompose the goal into a dag" in d
    assert "depends_on" in d
    assert "'tool'" in d
    assert "'needs_spec'" in d or "needs_spec" in d
    # Generic selection principle (match spec to the WORK a node does).
    assert "match on the work a step produces" in d
    assert "output-style spec to the node that produces the deliverable" in d
    # F2's format-bleed hygiene rule (research/gather node never carries a doc-format spec) STAYS.
    assert "never put a document-format spec on a research/gather node" in d
    assert "format-bleed" in d
    assert "research-analyst" in d


# --------------------------------------------------------------------------- #
# The removed doctrine actually lives in the DEFINITION LAYER (shapes / specs).
# --------------------------------------------------------------------------- #
def test_scheduling_doctrine_lives_in_schedule_leg_shape():
    dm = _norm(load_shape("schedule-leg").decompose_methodology)
    assert dm, "schedule-leg must carry the SCHEDULE-ONLY decompose_methodology"
    # The recipe removed from the engine prose lives HERE, verbatim in the shape.
    assert "cron_add" in dm and "recurring-scheduler" in dm
    assert "schedule-only" in dm or "repeating schedule" in dm


def test_output_format_binding_doctrine_lives_in_writer_specs():
    html_desc = _norm(CANONICAL_RULESETS["html-writer"][0])
    md_desc = _norm(CANONICAL_RULESETS["markdown-writer"][0])
    # Each writer spec ADVERTISES the format it produces + the format condition the planner
    # reasons over — so binding "the HTML writer for an HTML request" is a MODEL choice over
    # spec descriptions, not an engine-baked recipe.
    assert "html" in html_desc and "not markdown" in html_desc
    assert "markdown" in md_desc and "not html" in md_desc
    # …and both stay WRITE-node-only (the hygiene the planner needs from the description).
    for desc in (html_desc, md_desc):
        assert "never bind it to a research/gather/analysis node" in desc


# --------------------------------------------------------------------------- #
# ANTI-FAB — no shape-name / spec-name FLOW conditional was introduced (d341/d278).
# --------------------------------------------------------------------------- #
def test_factory_module_has_no_shape_or_spec_name_flow_conditional():
    src = inspect.getsource(factory_mod)
    # A code branch keyed on a shape/spec NAME is the fabrication smell (the engine dictating
    # flow by name). FACTORY_DESCRIPTION is a plain constant string — there must be no such
    # conditional anywhere in the module.
    for pat in (
        r"==\s*['\"]schedule-leg['\"]",
        r"==\s*['\"]recurring-scheduler['\"]",
        r"==\s*['\"]html-writer['\"]",
        r"==\s*['\"]markdown-writer['\"]",
        r"if\s+[^\n]*\bshape_name\b[^\n]*==",
    ):
        assert re.search(pat, src) is None, f"shape/spec-name flow conditional in factory.py: {pat!r}"


# --------------------------------------------------------------------------- #
# AUTHORING-NOT-REGRESSED — drive the real parse path with the THINNED description.
# --------------------------------------------------------------------------- #
def test_schedule_authoring_still_produces_one_cron_add_node(tmp_path):
    """With the scheduling recipe REMOVED from FACTORY_DESCRIPTION, a schedule-leg goal still
    authors EXACTLY ONE cron_add / recurring-scheduler node (the doctrine now flows from the
    shape methodology, not engine prose) through the real IncrementalPlanner parse path."""
    dm = load_shape("schedule-leg").decompose_methodology
    replies = [
        _seed(),
        _add("research the latest AI news and email me a summary",
             tool="cron_add", spec="recurring-scheduler", depends_on=[]),
        _finalize(),
    ]
    p = _drive_planner(tmp_path, shape_name="schedule-leg", methodology=dm, replies=replies)
    dag = _run(p.plan("Every morning at 8am, research the AI news and email me a summary.")).dag
    assert len(dag.nodes) == 1
    only = dag.nodes[0]
    assert only.tool == "cron_add"
    assert only.spec == "recurring-scheduler"
    assert only.depends_on == ()


def test_format_authoring_still_binds_html_writer_on_write_node(tmp_path):
    """With the output-format-BINDING recipe REMOVED from FACTORY_DESCRIPTION, a format goal still
    authors a research→write DAG that binds html-writer on the WRITE node and leaves the gather
    node format-free (the binding now flows from the writer spec descriptions + F2's _system
    hygiene guidance) — proven through the real parse path."""
    replies = [
        _seed(),
        _add("Search the US-Iran news", tool="web_search", depends_on=[]),
        _add("Write the HTML report to report.html", tool="file_write",
             spec="html-writer", depends_on=["n1"]),
        _finalize(),
    ]
    p = _drive_planner(tmp_path, shape_name="linear", methodology="", replies=replies)
    dag = _run(p.plan("Research the US-Iran situation and write it to an HTML file report.html.")).dag
    by = dag.by_id
    assert len(dag.nodes) == 2
    # gather node stays format-free (no document-format spec bled onto it)…
    assert by["n1"].spec in (None, "")
    # …and the WRITE node carries the html-writer output-style spec.
    assert by["n2"].tool == "file_write"
    assert by["n2"].spec == "html-writer"
    assert by["n2"].depends_on == ("n1",)
