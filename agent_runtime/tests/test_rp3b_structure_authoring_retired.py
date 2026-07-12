"""RP-3b (d311/d319/d328) — the 3 engine-authored STRUCTURE repairs are RETIRED.

The incremental planner used to CRUTCH the model with three finalization passes that
had the ENGINE author DAG/spec/format structure:

* ``_apply_default_research_spec`` — stamped a research spec onto null-spec gather nodes,
* ``_enforce_terminal_research_edge`` — auto-added the writer<-research edge on a
  disconnected terminal writer,
* ``_enforce_output_format_spec`` — stamped the requested format's writer onto the terminal.

RP-3b RETIRED all three, COUPLED to a live measure (``.recipe-notes/rp3b_measure.py``,
20 E4B :11434 trials with the passes bypassed): the planner authors all three axes
ITSELF 100% reliably — gather-node spec 12/12, terminal writer<-research edge 12/12,
terminal format writer 16/16 (incl. the deep-research write phase). Per d311/d319 the
engine now authors NO DAG/spec/format structure here; reliability is a DEFINITION-LAYER
property (the planner prompt + the selected shape), never an engine stamp/flag/spec-name
conditional.

This is the SELF-POLICING test (d311 D):

1. STRUCTURAL — the three methods (and their now-dead helpers + the
   ``default_research_spec`` plumbing) are GONE from :class:`IncrementalPlanner`, and no
   engine structure-authoring remains in the finalize pipeline.
2. ANTI-FABRICATION — a scripted FLAT / spec-less / format-less authoring flows through
   UNREPAIRED: the disconnected writer STAYS disconnected, the spec-less gather STAYS
   spec-less, the missing format writer STAYS missing. The engine authors nothing.
3. AUTHORING RELIABILITY — a scripted correctly-authored plan (the shape the live model
   reliably produces) flows through unchanged: terminal connected + gather spec assigned
   + writer format spec bound, by the PLANNER not the engine.

These script a :class:`FakeTransport` with the planner's TOOL CALLS, so the whole
tool-driven authoring loop + finalize run in-process with zero inference.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from typing import Sequence

from agent_runtime.factory import AbstractPlanFactory
from agent_runtime.incremental import IncrementalPlanner
from llm_framework import FakeTransport
from specialization.registry import SpecRegistry
from specialization.seed import seed_canonical_rulesets


def _run(coro):
    return asyncio.run(coro)


_TOOL_CATALOG = [
    {"name": "web_search", "description": "search the web for candidate pages"},
    {"name": "web_fetch", "description": "fetch and extract a page's article text"},
    {"name": "file_write", "description": "write content to a file"},
]


def _seed(shape: str = "modular-parallel") -> str:
    return json.dumps({"tool": "seed_plan", "args": {"shape": shape}})


def _add(task: str, *, tool: str = "", spec: str = "", depends_on: Sequence[str] = ()) -> str:
    return json.dumps(
        {"tool": "add_step",
         "args": {"task": task, "tool": tool, "spec": spec, "specs": [],
                  "depends_on": list(depends_on)}}
    )


def _finalize() -> str:
    return json.dumps({"tool": "finalize_plan", "args": {}})


def _planner(replies, tmp_path) -> IncrementalPlanner:
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)  # registers research-analyst + html-writer + markdown-writer
    factory = AbstractPlanFactory(reg.index(), tool_catalog=_TOOL_CATALOG)
    return IncrementalPlanner(
        FakeTransport(list(replies)),
        factory,
        spec_names=reg.names(),
        tool_names=[t["name"] for t in _TOOL_CATALOG],
        shape_name="modular-parallel",
        shape_description="independent gather steps then one combine/deliver step",
    )


# --------------------------------------------------------------------------- #
# 1. STRUCTURAL — the retired passes + their plumbing are GONE
# --------------------------------------------------------------------------- #
def test_retired_structure_passes_are_absent():
    for gone in (
        "_apply_default_research_spec",
        "_enforce_terminal_research_edge",
        "_enforce_output_format_spec",
        "_is_research",
        "_ancestors",
        "_requested_output_format",
    ):
        assert not hasattr(IncrementalPlanner, gone), f"{gone} should be retired (RP-3b)"
    # the module-level output-format tables are gone too
    import agent_runtime.incremental as incr
    assert not hasattr(incr, "_OUTPUT_FORMAT_WRITERS")
    assert not hasattr(incr, "_OUTPUT_FORMAT_VARIANTS")
    # the default-research-spec plumbing is removed from the constructor
    params = inspect.signature(IncrementalPlanner.__init__).parameters
    assert "default_research_spec" not in params, (
        "the engine no longer carries a default research spec to author onto gather nodes"
    )


def test_no_engine_structure_authoring_source_remains():
    # Source-level guard: the finalize pipeline in plan() must not call any structure
    # repair. It may still call F5 (_apply_requested_specs — user-named spec, out of
    # scope) and the d50 filename echo (out of scope). No research-edge / default-spec /
    # output-format authoring may remain.
    src = inspect.getsource(IncrementalPlanner.plan)
    for banned in (
        "_apply_default_research_spec(",
        "_enforce_terminal_research_edge(",
        "_enforce_output_format_spec(",
    ):
        assert banned not in src, f"plan() still calls {banned} (RP-3b retirement incomplete)"


# --------------------------------------------------------------------------- #
# 2. ANTI-FABRICATION — a flat / spec-less / format-less plan flows through UNREPAIRED
# --------------------------------------------------------------------------- #
def _unrepaired_replies() -> list[str]:
    # The exact defects the retired passes used to fix: a spec-less gather node (n1)
    # and a disconnected, format-less writer (n2) — NO depends_on, NO spec — for an
    # HTML goal. Nothing must repair them anymore.
    return [
        _seed(),
        _add("Research the recent US-Iran conflict", tool="web_search"),   # n1: null spec
        _add("Write a detailed HTML report", tool="file_write"),           # n2: disconnected, no format
        _finalize(),
    ]


def test_engine_authors_nothing_disconnected_writer_stays_disconnected(tmp_path):
    planner = _planner(_unrepaired_replies(), tmp_path)
    result = _run(planner.plan("Write a detailed HTML report on the US-Iran conflict"))
    by = result.dag.by_id
    # the engine no longer adds the writer<-research edge — n2 stays a disconnected sink
    assert by["n2"].depends_on == (), "engine must NOT author the terminal-research edge"
    # the engine no longer stamps a research spec on the spec-less gather node
    assert by["n1"].effective_specs == (), "engine must NOT author a gather-node spec"
    # the engine no longer stamps the HTML writer on the format-less terminal
    assert "html-writer" not in by["n2"].effective_specs, "engine must NOT author the format writer"
    # and the retired repair-note keys are gone from the PlanResult
    assert "research_edges" not in result.repair
    assert "output_format" not in result.repair


# --------------------------------------------------------------------------- #
# 3. AUTHORING RELIABILITY — a correctly-authored plan flows through unchanged
# --------------------------------------------------------------------------- #
def _authored_replies() -> list[str]:
    # The shape the live model reliably produces (RP-3b measure): two gather nodes each
    # carrying a research spec, and a delivery writer that depends_on both and carries
    # the HTML writer spec — all authored by the PLANNER (here, the script).
    return [
        _seed(),
        _add("Research AI news", tool="web_search", spec="research-analyst"),      # n1
        _add("Research technology news", tool="web_search", spec="research-analyst"),  # n2
        _add("Compile and email an HTML brief", tool="file_write",
             spec="html-writer", depends_on=["n1", "n2"]),                          # n3
        _finalize(),
    ]


def test_planner_authored_axes_flow_through_unchanged(tmp_path):
    planner = _planner(_authored_replies(), tmp_path)
    result = _run(planner.plan("Research AI + tech news and save an HTML brief as brief.html"))
    by = result.dag.by_id
    # AXIS 1 — the terminal writer depends on both gather nodes (planner-authored edge)
    assert set(by["n3"].depends_on) == {"n1", "n2"}
    # AXIS 2 — every gather node carries the planner-authored research spec
    assert by["n1"].effective_specs == ("research-analyst",)
    assert by["n2"].effective_specs == ("research-analyst",)
    # AXIS 3 — the terminal writer carries the planner-authored HTML writer spec
    assert "html-writer" in by["n3"].effective_specs
    # nothing was engine-repaired
    assert "research_edges" not in result.repair
    assert "output_format" not in result.repair
