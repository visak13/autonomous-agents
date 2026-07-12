"""s16/a3 (d239/d247) — RETIRE THE UNROLL: a shape is EXECUTION DISCIPLINE + DOCTRINE only.

This file previously proved the deterministic ``unroll_shape`` turned the deep-research
template (round/final POSITIONS + max_iter) into a fixed acyclic ``r{N}_research``/``r{N}_critic``
DAG that the generic runtime executed. That deterministic node population is RETIRED: a shape
NEVER pre-bakes a node graph and NEVER binds a gather tool. The deep-research research TOPOLOGY
is now AUTHORED at runtime by REASONING — the engine emits a TOOL-LESS self-selecting research
seed and the :class:`~agent_runtime.research_tree.DagGrower` (decompose-first → grow on note
gaps) authors the layers. The growable-engine EXECUTION (the seed + grow loop on the generic
``AgentRuntime``) is proven in ``test_p2_5b_growable.py``; this file locks the shape-as-discipline
contract at the ``agent_runtime`` layer.
"""
from __future__ import annotations

import pytest

from agent_runtime.shapes import ShapeSpec, load_shape


def test_unroll_shape_is_retired():
    # The deterministic unroll is DELETED — it is no longer importable from agent_runtime.shapes.
    with pytest.raises(ImportError):
        from agent_runtime.shapes import unroll_shape  # noqa: F401


def test_shape_declares_no_per_round_topology():
    # round_roles/final_roles are RETIRED from the dataclass: a shape carries NO per-node
    # topology in any posture (the deep-research topology is reasoned at runtime by the grower).
    spec = ShapeSpec(name="deep-research", execution="deep-research", max_iter=3, hard_cap=24)
    assert not hasattr(spec, "round_roles")
    assert not hasattr(spec, "final_roles")


def test_deep_research_identity_is_the_execution_token():
    # The deep-research FAMILY is keyed off the execution DISCIPLINE token, not declared
    # round/final positions (which no longer exist).
    on_disk = load_shape("deep-research")
    assert on_disk.is_deep_research
    assert on_disk.execution == "deep-research"
    # the growable marker → the engine builds a growable seed the grower authors topology from.
    assert on_disk.expand_on_gaps
    # a discipline shape is NOT deep-research (its DAG is authored by the incremental planner).
    assert not ShapeSpec(name="x", execution="concurrent").is_deep_research


def test_effective_max_iter_is_the_depth_ceiling_clamped_to_hard_cap():
    # The UI-overridable max_iter is now the research DEPTH ceiling the grower reasons within
    # (it no longer drives a fixed unrolled round count). Clamp-to-hard_cap is preserved.
    on_disk = load_shape("deep-research")
    assert on_disk.effective_max_iter(2) == 2
    assert on_disk.effective_max_iter(999) == on_disk.hard_cap
