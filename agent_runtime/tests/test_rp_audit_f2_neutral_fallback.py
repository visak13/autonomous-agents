"""RP-AUDIT F2 — the methodology-LESS authoring FALLBACK is TOPOLOGY-NEUTRAL, and the
codebase-summary shape carries a READ→WRITE ``decompose_methodology`` that renders with
PRECEDENCE (so a non-web read→write task is not mis-authored as a gather→combine plan).

Finding F2 (HIGH): the generic authoring guidance in ``IncrementalPlanner._system`` /
``_initial_user`` / ``_finalize_user`` was NOT neutral — it baked a GATHER-shaped
gather→combine→deliver flow ("a modular-parallel plan is INDEPENDENT sub-tasks FOLLOWED
BY one FINAL step that combines their outputs", "list the DISTINCT ITEMS … a gather step
per item", "combines the gathered results … depends_on MUST list EVERY gather step id").
Every methodology-LESS shape inherited that mold. On the served route the acute case is
``codebase-summary``: its non-web read→write doctrine lived ONLY in the shape DESCRIPTION,
threaded into the same prompt ALONGSIDE the contradicting baked gather recipe — and a
DESCRIPTION does not replace the recipe (only a ``decompose_methodology`` does, via the
F1/RP-4c precedence mechanism).

Fix (two parts, both DEFINITION-layer / anti-fab d341):
  PART A — NEUTRALIZE the fallback: the ``_system`` / ``_initial_user`` / ``_finalize_user``
    generic guidance is now topology-neutral (author the steps the GOAL needs, wire real
    dependencies, end with the step that DELIVERS the outcome) with NO gather→combine mold.
  PART B — give ``codebase-summary`` a ``decompose_methodology`` (the read→write two-step
    doctrine, promoted from its DESCRIPTION) so it RENDERS WITH PRECEDENCE and REPLACES the
    (now-neutral) fallback.

These tests are OFFLINE (no inference): they assert the rendered prompts are neutral in the
fallback, that codebase-summary's methodology renders with precedence and drives read→write,
that a methodology-less shape still authors a VALID plan through the real parse path, and
that no shape-name/spec-name conditional was introduced (anti-fab d341/d310/d319).
"""
from __future__ import annotations

import asyncio
import inspect
import json
from typing import Sequence

from agent_runtime.factory import AbstractPlanFactory
from agent_runtime.incremental import IncrementalPlanner
from agent_runtime.plan_tools import PlanBuilder
from agent_runtime.shapes import load_shape
from llm_framework import FakeTransport
from specialization.registry import SpecRegistry
from specialization.seed import seed_canonical_rulesets


def _run(coro):
    return asyncio.run(coro)


_TOOL_CATALOG = [
    {"name": "web_search", "description": "search the web"},
    {"name": "web_fetch", "description": "fetch a page"},
    {"name": "file_write", "description": "write a file"},
    {"name": "send_mail", "description": "send an email"},
]

# The pre-F2 gather→combine→deliver mold that the neutral fallback must NO LONGER contain,
# across ALL THREE authoring turns (d341: the engine must not bake a fixed gather flow).
_GATHER_MOLD = (
    "A modular-parallel plan is INDEPENDENT sub-task steps",
    "FOLLOWED BY one FINAL step that combines their outputs",
    "List the DISTINCT ITEMS the GOAL names",
    "Each gather step covers exactly ONE",
    "combines the gathered results",
    "depends_on MUST list EVERY gather step id",
    "The FINAL combine step's depends_on lists EVERY sub-task id",
    "the HTML writer for HTML",
    "the gathering phase is COMPLETE",
)


def _planner(tmp_path, *, shape_name, methodology, replies: Sequence[str] = ()) -> IncrementalPlanner:
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


def _builder_two_gathers(tmp_path) -> PlanBuilder:
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)
    b = PlanBuilder(spec_names=reg.names(), tool_names=[t["name"] for t in _TOOL_CATALOG])
    b.dispatch("seed_plan", {"goal": "g"})
    b.dispatch("add_step", {"task": "gather A", "tool": "web_search", "depends_on": []})
    b.dispatch("add_step", {"task": "gather B", "tool": "web_search", "depends_on": []})
    return b


# --------------------------------------------------------------------------- #
# PART A — the methodology-LESS fallback is TOPOLOGY-NEUTRAL across all 3 turns.
# --------------------------------------------------------------------------- #
def test_fallback_is_topology_neutral_no_gather_mold(tmp_path):
    p = _planner(tmp_path, shape_name="linear", methodology="")
    goal = "Read the module at C:/proj/foo.py and write a summary to out.md."
    blob = (
        p._system(goal)
        + p._initial_user(goal)
        + p._finalize_user(goal, _builder_two_gathers(tmp_path))
    )
    for frag in _GATHER_MOLD:
        assert frag not in blob, f"gather-mold fragment leaked into the neutral fallback: {frag!r}"


def test_fallback_still_guides_authoring_generically(tmp_path):
    """Neutral does NOT mean empty — it still tells the model to author the goal's steps,
    wire dependencies, and end with the delivering step (generically useful)."""
    p = _planner(tmp_path, shape_name="linear", methodology="")
    goal = "Do the thing and save it."
    sysmsg, initial = p._system(goal), p._initial_user(goal)
    fin = p._finalize_user(goal, _builder_two_gathers(tmp_path))
    # System fallback: author the smallest correct set + wire real depends_on + deliver last.
    assert "Author the SMALLEST correct set of steps for THIS goal" in sysmsg
    assert "depends_on" in sysmsg
    assert "DELIVERS the goal's outcome" in sysmsg
    # Initial procedure: still a seed→add_step→finalize decision procedure.
    assert "Work out the steps THIS goal actually needs" in initial
    assert "seed_plan" in initial and "finalize_plan" in initial
    # Finalize turn: author the remaining/delivering step(s), no gather assumption.
    assert "DELIVERS the goal's outcome" in fin


# --------------------------------------------------------------------------- #
# PART B — codebase-summary carries a read→write methodology that RENDERS with precedence.
# --------------------------------------------------------------------------- #
def test_codebase_summary_shape_carries_read_write_methodology():
    shape = load_shape("codebase-summary")
    dm = shape.decompose_methodology
    assert dm.strip(), "codebase-summary must carry a decompose_methodology (read→write doctrine)"
    low = dm.lower()
    # The read→write two-step topology, non-web, grounded in on-disk files.
    assert "read" in low and "write" in low
    assert "read_dir" in low or "read_file" in low
    assert "file_write" in low
    assert "codebase" in low
    assert "no web" in low or "not a web" in low or "no gather" in low.replace("→", "")
    assert "two" in low  # exactly two steps


def test_codebase_methodology_takes_precedence_over_neutral_fallback(tmp_path):
    dm = load_shape("codebase-summary").decompose_methodology
    p = _planner(tmp_path, shape_name="codebase-summary", methodology=dm)
    goal = "Summarize the codebase at C:/proj and write summary.md."
    sysmsg, initial = p._system(goal), p._initial_user(goal)
    fin = p._finalize_user(goal, _builder_two_gathers(tmp_path))
    # The methodology is rendered into ALL THREE turns with explicit precedence.
    head = dm.strip()[:50]
    assert head in sysmsg and head in initial and head in fin
    assert "PRECEDENCE" in sysmsg and "PRECEDENCE" in initial and "PRECEDENCE" in fin
    # …and the neutral fallback guidance is suppressed when the methodology drives authoring.
    assert "Author the SMALLEST correct set of steps for THIS goal" not in sysmsg
    assert "Work out the steps THIS goal actually needs" not in initial


# --------------------------------------------------------------------------- #
# Drive proofs — a valid DAG authors on BOTH routes through the real parse path.
# --------------------------------------------------------------------------- #
def _seed():
    return json.dumps({"tool": "seed_plan", "args": {}})


def _add(task, *, tool="", spec="", depends_on=()):
    return json.dumps(
        {"tool": "add_step", "args": {"task": task, "tool": tool, "spec": spec,
                                      "specs": [], "depends_on": list(depends_on)}}
    )


def _finalize():
    return json.dumps({"tool": "finalize_plan", "args": {}})


def test_codebase_methodology_drives_valid_read_write_dag(tmp_path):
    """With the read→write methodology, the planner authors a valid 2-node read→write DAG
    (proving the methodology-driven topology parses through the real factory)."""
    dm = load_shape("codebase-summary").decompose_methodology
    replies = [
        _seed(),
        _add("Read the files under C:/proj", tool="", depends_on=[]),
        _add("Write the Markdown summary to summary.md", tool="file_write",
             spec="codebase-summary", depends_on=["n1"]),
        _finalize(),
    ]
    p = _planner(tmp_path, shape_name="codebase-summary", methodology=dm, replies=replies)
    dag = _run(p.plan("Summarize the codebase at C:/proj and write summary.md.")).dag
    by = dag.by_id
    assert len(dag.nodes) == 2
    # step 1 reads (no upstream); step 2 writes and depends_on step 1 (read→write chain).
    assert by["n1"].depends_on == ()
    assert by["n2"].depends_on == ("n1",)
    assert by["n2"].tool == "file_write"


def test_methodology_less_shape_still_authors_valid_plan(tmp_path):
    """VERIFY (F2): neutralizing the shared fallback did NOT break a methodology-less shape —
    a linear goal still authors a valid plan through seed→add_step→finalize with the neutral
    fallback prompts (the authoring mechanics are unaffected by the prompt neutralization)."""
    replies = [
        _seed(),
        _add("Search the AI news", tool="web_search", depends_on=[]),
        _add("Search the climate news", tool="web_search", depends_on=[]),
        _add("Write the combined brief", tool="file_write", depends_on=["n1", "n2"]),
        _finalize(),
    ]
    p = _planner(tmp_path, shape_name="linear", methodology="", replies=replies)
    result = _run(p.plan("Research AI and climate news and save a brief."))
    dag = result.dag
    assert len(dag.nodes) == 3
    assert set(dag.by_id["n3"].depends_on) == {"n1", "n2"}
    assert p.last_builder.finalized is True


# --------------------------------------------------------------------------- #
# ANTI-FAB — the whole mechanism stays a GENERIC presence check, no name conditional.
# --------------------------------------------------------------------------- #
def test_no_shape_or_spec_name_conditional_in_authoring_turns():
    for method in (
        IncrementalPlanner._system,
        IncrementalPlanner._initial_user,
        IncrementalPlanner._finalize_user,
    ):
        src = inspect.getsource(method)
        # The methodology substitution is a presence check on the generic FIELD …
        assert "self.shape_decompose_methodology" in src
        assert "if methodology:" in src
        # … NOT a shape/spec-NAME conditional (the d341/d278 fabrication smell).
        assert "codebase-summary" not in src
        assert "schedule-leg" not in src
        assert "shape_name ==" not in src and "shape_name==" not in src
