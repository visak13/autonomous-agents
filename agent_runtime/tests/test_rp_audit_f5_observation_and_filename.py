"""RP-AUDIT F5 — the two residual d341-class spots in ``IncrementalPlanner`` are
NEUTRALIZED (self-policing, anti-fab d310/d311/d319/d341):

(a) ``_observation_user`` (the per-step OBSERVATION turn fed back mid-authoring) leaned on
    GATHER FRAMING — it told the model to "finalize_plan if every distinct item is covered",
    which assumes a gather→combine flow and mis-frames a read→write / single-step / non-gather
    goal. F5 makes the turn TOPOLOGY-NEUTRAL and METHODOLOGY-AWARE: when the selected shape
    supplies a ``decompose_methodology`` the nudge DEFERS to it (mirroring F1/F2/RP-4c); with
    no methodology it uses a neutral nudge with NO gather assumption. A generic presence check
    on the methodology field, no shape-name/spec-name conditional.

(b) ``_echo_literal_filename`` (the d50 pass that, on an explicit user-named filename, wrote
    engine-authored prose — "Write the file as cats.html." — into the terminal node's TASK and
    assumed a FILE-WRITER SINK) is RETIRED. Both are anti-fab violations (engine authoring node
    task content + baking a delivery-sink assumption). Filename-honoring is PRESERVED by the
    already-present goal-carry: the user's named file rides ``PlanDAG.goal`` verbatim (d39) and
    the writer derives its path via ``derive_output_path(overall_goal, node.task)``, which reads
    the explicit filename straight from the GOAL — so ``cats.html`` still reaches disk with NO
    engine-authored task text and NO file-sink assumption.

These tests are OFFLINE (no inference): they assert the rendered observation turn is neutral in
the fallback / defers to a methodology when present, that the echo pass is gone and the terminal
task is left byte-identical (no engine prose), that filename-honoring survives via the goal-carry,
and that no shape-name/spec-name conditional was introduced.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from typing import Sequence

from agent_runtime import incremental as incremental_mod
from agent_runtime.factory import AbstractPlanFactory
from agent_runtime.incremental import IncrementalPlanner
from agent_runtime.synth_tools import derive_output_path
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

# The pre-F5 gather framing the neutralized observation turn must NO LONGER contain.
_GATHER_FRAMING = "every distinct item is covered"


def _planner(tmp_path, *, methodology: str, replies: Sequence[str] = ()) -> IncrementalPlanner:
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)
    factory = AbstractPlanFactory(reg.index(), tool_catalog=_TOOL_CATALOG)
    return IncrementalPlanner(
        FakeTransport(list(replies)),
        factory,
        spec_names=reg.names(),
        tool_names=[t["name"] for t in _TOOL_CATALOG],
        shape_name="linear",
        shape_description="d",
        shape_decompose_methodology=methodology,
    )


def _obs(note="one step done", steps=None):
    return {"note": note, "steps": steps or [
        {"id": "n1", "task": "gather AI news", "tool": "web_search",
         "spec": "", "depends_on": []},
    ]}


# --------------------------------------------------------------------------- #
# (a1) NEUTRAL — the methodology-less observation turn drops the gather framing.
# --------------------------------------------------------------------------- #
def test_observation_turn_is_topology_neutral_no_gather_framing(tmp_path):
    p = _planner(tmp_path, methodology="")
    turn = p._observation_user(_obs())
    assert _GATHER_FRAMING not in turn, "gather framing leaked into the neutralized observation turn"
    # Neutral ≠ empty: it still asks for the next step + finalize when the goal's steps are all present.
    assert "add_step for the next step THIS goal needs" in turn
    assert "finalize_plan once" in turn
    assert "the plan has all the steps the goal needs" in turn


# --------------------------------------------------------------------------- #
# (a2) METHODOLOGY-AWARE — with a decompose_methodology the turn DEFERS to it.
# --------------------------------------------------------------------------- #
def test_observation_turn_defers_to_shape_methodology_when_present(tmp_path):
    p = _planner(tmp_path, methodology="Step 1 read the files. Step 2 write the summary.")
    turn = p._observation_user(_obs())
    # Defers to the selected shape's authoring methodology …
    assert "authoring methodology" in turn
    assert "'linear'" in turn  # the shape name, rendered generically from self.shape_name
    # … and it does NOT fall back to the gather framing or the neutral generic nudge.
    assert _GATHER_FRAMING not in turn
    assert "add_step for the next step THIS goal needs" not in turn


# --------------------------------------------------------------------------- #
# (b1) ECHO RETIRED — the engine-authored-task-text + file-sink pass is GONE.
# --------------------------------------------------------------------------- #
def test_echo_literal_filename_pass_is_retired():
    assert not hasattr(IncrementalPlanner, "_echo_literal_filename"), (
        "the d50 engine-authored-task echo pass must be RETIRED (anti-fab d319/d341)"
    )
    src = inspect.getsource(incremental_mod)
    # No engine-authored node-task prose, and no file-writer-sink classification remains.
    assert "Write the file as" not in src, "engine-authored task prose still present"
    assert "self.research_tools" not in src, "file-writer-sink classification still present"
    assert "DEFAULT_RESEARCH_TOOLS" not in src


# --------------------------------------------------------------------------- #
# (b2) FILENAME-HONORING PRESERVED — the goal-carry resolves the user's named file
#      WITHOUT the echo, from the GOAL alone (the writer's real path-derivation input).
# --------------------------------------------------------------------------- #
def test_filename_honoring_preserved_via_goal_carry():
    # The user names cats.html; the writer derives its path from overall_goal + node.task.
    # Even when the node task carries NOTHING about the filename, the GOAL alone honors it —
    # exactly what the served route feeds (PlanDAG.goal = overall_goal, d39).
    assert derive_output_path("please write cats.html for me", "") == "cats.html"
    assert derive_output_path("please write cats.html for me", "deliver the report") == "cats.html"


def _seed():
    return json.dumps({"tool": "seed_plan", "args": {}})


def _add(task, *, tool="", spec="", depends_on=()):
    return json.dumps(
        {"tool": "add_step", "args": {"task": task, "tool": tool, "spec": spec,
                                      "specs": [], "depends_on": list(depends_on)}}
    )


def _finalize():
    return json.dumps({"tool": "finalize_plan", "args": {}})


def test_terminal_task_left_byte_identical_no_engine_prose(tmp_path):
    """DRIVE (b): with a goal that names cats.html, the authored terminal node's task is
    the MODEL's verbatim task — the engine appends NO 'Write the file as cats.html.' prose."""
    terminal_task = "Write the report on cats"
    replies = [
        _seed(),
        _add("Search for cat facts", tool="web_search", depends_on=[]),
        _add(terminal_task, tool="file_write", depends_on=["n1"]),
        _finalize(),
    ]
    p = _planner(tmp_path, methodology="", replies=replies)
    result = _run(p.plan("Research cats and write cats.html."))
    dag = result.dag
    term = dag.by_id["n2"]
    assert term.task == terminal_task, "engine mutated the terminal node's task (echo not retired)"
    assert "Write the file as" not in term.task
    # The repair envelope no longer carries a literal_filename echo record.
    assert "literal_filename" not in result.repair


# --------------------------------------------------------------------------- #
# ANTI-FAB — the observation-turn methodology branch is a generic presence check.
# --------------------------------------------------------------------------- #
def test_observation_substitution_is_generic_no_name_conditional():
    src = inspect.getsource(IncrementalPlanner._observation_user)
    # Presence check on the generic methodology FIELD (same field as _system/_initial/_finalize) …
    assert "self.shape_decompose_methodology" in src
    # … NOT a shape/spec-NAME conditional (the d341/d278 fabrication smell).
    assert "codebase-summary" not in src
    assert "schedule-leg" not in src
    assert "shape_name ==" not in src and "shape_name==" not in src


def test_observation_reads_same_methodology_field_as_other_turns():
    for method in (
        IncrementalPlanner._system,
        IncrementalPlanner._initial_user,
        IncrementalPlanner._finalize_user,
        IncrementalPlanner._observation_user,
    ):
        assert "self.shape_decompose_methodology" in inspect.getsource(method)
