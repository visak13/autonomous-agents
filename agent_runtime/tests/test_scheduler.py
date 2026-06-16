"""The deterministic, model-INDEPENDENT dispatch scheduler (s3/b1, blueprint §2a).

Ports of eda-base3's ``_first_ready_action`` (the wave-dispatch readiness gate) +
``plan_next_action`` (the dispatch FSM) as PURE PYTHON. These tests lock the
readiness gate and the per-turn dispatch decision for both execution disciplines
WITHOUT a runtime, transport, or GPU — the decision is over the DAG + the
done/running id sets only, so it is fully offline and reproducible.
"""
from __future__ import annotations

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.scheduler import (
    ExecutionMode,
    execution_mode_for,
    first_ready_action,
    is_complete,
    next_dispatch,
    ready_wave,
)


def _diamond() -> PlanDAG:
    # n1 → {n2, n3} → n4 : two independent middle nodes, one root, one join.
    return PlanDAG(
        nodes=[
            PlanNode(id="n1", task="root"),
            PlanNode(id="n2", task="mid-a", depends_on=("n1",)),
            PlanNode(id="n3", task="mid-b", depends_on=("n1",)),
            PlanNode(id="n4", task="join", depends_on=("n2", "n3")),
        ]
    )


# --------------------------------------------------------------------------- #
# first_ready_action — the port of eda-base3 `_first_ready_action`
# --------------------------------------------------------------------------- #
def test_first_ready_action_picks_first_in_node_order():
    dag = _diamond()
    # Nothing done → only the root is ready (deps of others unsatisfied).
    assert first_ready_action(dag, done=set()).id == "n1"
    # n1 done → n2 and n3 both ready; the gate returns the FIRST in node order.
    assert first_ready_action(dag, done={"n1"}).id == "n2"
    # n1,n2 done → n3 is the next ready (n4 still waits on n3).
    assert first_ready_action(dag, done={"n1", "n2"}).id == "n3"
    # all deps of n4 satisfied → n4.
    assert first_ready_action(dag, done={"n1", "n2", "n3"}).id == "n4"
    # everything done → no ready node.
    assert first_ready_action(dag, done={"n1", "n2", "n3", "n4"}) is None


def test_first_ready_action_skips_blocked_and_done():
    dag = _diamond()
    # n1 marked blocked (e.g. failed) → its dependents never become ready, and the
    # gate skips n1 itself → nothing ready.
    assert first_ready_action(dag, done=set(), blocked={"n1"}) is None
    # n1 done but n2 blocked → the gate returns n3, not the blocked n2.
    assert first_ready_action(dag, done={"n1"}, blocked={"n2"}).id == "n3"


# --------------------------------------------------------------------------- #
# ready_wave — every ready node (the concurrent dispatch set)
# --------------------------------------------------------------------------- #
def test_ready_wave_returns_all_independent_ready_nodes():
    dag = _diamond()
    assert [n.id for n in ready_wave(dag, done={"n1"})] == ["n2", "n3"]
    # node order is preserved/deterministic.
    assert [n.id for n in ready_wave(dag, done=set())] == ["n1"]
    assert ready_wave(dag, done={"n1", "n2", "n3", "n4"}) == []


# --------------------------------------------------------------------------- #
# next_dispatch — SEQUENTIAL launches ONE at a time; CONCURRENT launches the wave
# --------------------------------------------------------------------------- #
def test_concurrent_dispatch_launches_whole_wave():
    dag = _diamond()
    d = next_dispatch(dag, done={"n1"}, running=(), mode=ExecutionMode.CONCURRENT)
    assert [n.id for n in d.nodes] == ["n2", "n3"]  # both at once (modular-parallel)
    assert d.has_work


def test_concurrent_dispatch_excludes_already_running():
    dag = _diamond()
    # n2 already running → only n3 is newly launchable this turn.
    d = next_dispatch(dag, done={"n1"}, running={"n2"}, mode=ExecutionMode.CONCURRENT)
    assert [n.id for n in d.nodes] == ["n3"]


def test_sequential_dispatch_launches_only_first_ready():
    dag = _diamond()
    # Two nodes ready, nothing running → SEQUENTIAL launches ONLY the first.
    d = next_dispatch(dag, done={"n1"}, running=(), mode=ExecutionMode.SEQUENTIAL)
    assert [n.id for n in d.nodes] == ["n2"]


def test_sequential_dispatch_waits_while_one_in_flight():
    dag = _diamond()
    # A node is in flight → SEQUENTIAL launches NOTHING (strict single-file).
    d = next_dispatch(dag, done={"n1"}, running={"n2"}, mode=ExecutionMode.SEQUENTIAL)
    assert d.nodes == ()
    assert not d.has_work


# --------------------------------------------------------------------------- #
# is_complete — the terminal condition (nothing in flight, nothing launchable)
# --------------------------------------------------------------------------- #
def test_is_complete_only_when_settled():
    dag = _diamond()
    assert not is_complete(dag, done=set())                 # n1 still launchable
    assert not is_complete(dag, done={"n1"}, running={"n2"})  # n2 in flight
    assert is_complete(dag, done={"n1", "n2", "n3", "n4"})  # all done
    # A failed root that blocks the rest is also "settled" (nothing can launch).
    assert is_complete(dag, done=set(), blocked={"n1"})


# --------------------------------------------------------------------------- #
# execution_mode_for — token → mode mapping (with aliases + safe fallback)
# --------------------------------------------------------------------------- #
def test_execution_mode_for_maps_tokens_and_aliases():
    assert execution_mode_for("sequential") is ExecutionMode.SEQUENTIAL
    assert execution_mode_for("linear") is ExecutionMode.SEQUENTIAL
    assert execution_mode_for("concurrent") is ExecutionMode.CONCURRENT
    assert execution_mode_for("modular-parallel") is ExecutionMode.CONCURRENT
    # deep-research is not a dispatch mode → harmless CONCURRENT fallback.
    assert execution_mode_for("deep-research") is ExecutionMode.CONCURRENT
    # unknown / empty → CONCURRENT (legacy default, never crashes).
    assert execution_mode_for("nonsense") is ExecutionMode.CONCURRENT
    assert execution_mode_for(None) is ExecutionMode.CONCURRENT
