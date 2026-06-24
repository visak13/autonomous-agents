"""DAG schema-validation + SAFE FALLBACK for the planner emission path (s1/b2).

Builder-owned hardening of the a2 4/5 graph-integrity finding: with ``think=True``
the planner non-deterministically emits a DAG whose ``depends_on`` names a PHANTOM
node id (a dangling edge). ``parse_dag`` correctly REJECTS such a DAG (no silent bad
DAG) — but a hard reject surfaces as a user-visible failure, and 4/5 valid is not
good enough. This locks the two-part contract:

* STRICT path (``parse_dag``) keeps rejecting a dangling edge — the "no silent bad
  DAG" guarantee is unchanged.
* SAFE FALLBACK (``parse_dag_safe``, wired into ``Planner.plan`` /
  ``replan_subgraph``) REPAIRS a dangling/self edge by DROPPING it (graceful
  degrade), so a phantom-edge emission never fails the run; it reports what it
  dropped. Every OTHER invalidity (duplicate id / real cycle / empty plan) STILL
  raises, preserving the outer self-heal's retry-on-reject backstop.

Pure, in-process, offline — FakeTransport only (no GPU / :11435). Deterministic.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agent_runtime.factory import AbstractPlanFactory, PlanError
from agent_runtime.planner import Planner
from agent_runtime.selfheal import MalformedOutputError


def _factory() -> AbstractPlanFactory:
    return AbstractPlanFactory([])


# a 3-node valid chain n1 -> n2 -> n3
_VALID = {
    "rationale": "ok",
    "nodes": [
        {"id": "n1", "task": "gather"},
        {"id": "n2", "task": "analyse", "depends_on": ["n1"]},
        {"id": "n3", "task": "deliver", "depends_on": ["n2"]},
    ],
}

# the a2 malformation: n3 also depends on a PHANTOM 'n9' that no node defines.
_DANGLING = {
    "rationale": "phantom edge",
    "nodes": [
        {"id": "n1", "task": "gather"},
        {"id": "n2", "task": "analyse", "depends_on": ["n1"]},
        {"id": "n3", "task": "deliver", "depends_on": ["n2", "n9"]},
    ],
}


# --------------------------------------------------------------------------- #
# 1. valid DAG: both paths accept it, parse_dag_safe reports NO repairs
# --------------------------------------------------------------------------- #
def test_valid_dag_passes_both_paths() -> None:
    f = _factory()
    strict = f.parse_dag(_VALID)
    assert [n.id for n in strict.topo_order()] == ["n1", "n2", "n3"]

    safe, repairs = f.parse_dag_safe(_VALID)
    assert repairs == []  # nothing dangling → byte-identical topology
    assert [n.id for n in safe.topo_order()] == ["n1", "n2", "n3"]
    assert safe.as_dict() == strict.as_dict()


# --------------------------------------------------------------------------- #
# 2. STRICT parse_dag REJECTS a dangling edge (no silent bad DAG)
# --------------------------------------------------------------------------- #
def test_strict_parse_rejects_dangling_edge() -> None:
    f = _factory()
    with pytest.raises(PlanError, match="unknown node 'n9'"):
        f.parse_dag(_DANGLING)


# --------------------------------------------------------------------------- #
# 3. SAFE fallback REPAIRS a dangling edge: drops the phantom, keeps the rest
# --------------------------------------------------------------------------- #
def test_safe_fallback_repairs_dangling_edge() -> None:
    f = _factory()
    dag, repairs = f.parse_dag_safe(_DANGLING)
    # the phantom edge is gone, the REAL edge (n2) is preserved.
    assert dag.by_id["n3"].depends_on == ("n2",)
    # a valid topo order still exists (the DAG validated cleanly).
    assert [n.id for n in dag.topo_order()] == ["n1", "n2", "n3"]
    # exactly one repair, and it names the dropped phantom for observability.
    assert len(repairs) == 1
    assert "n3" in repairs[0] and "n9" in repairs[0]


# --------------------------------------------------------------------------- #
# 4. self-referential edge is also repaired (n -> n)
# --------------------------------------------------------------------------- #
def test_safe_fallback_repairs_self_edge() -> None:
    f = _factory()
    self_dep = {
        "nodes": [
            {"id": "n1", "task": "a"},
            {"id": "n2", "task": "b", "depends_on": ["n1", "n2"]},
        ]
    }
    # strict rejects the self-edge…
    with pytest.raises(PlanError, match="depends on itself"):
        f.parse_dag(self_dep)
    # …safe drops it, keeping the real n1 edge.
    dag, repairs = f.parse_dag_safe(self_dep)
    assert dag.by_id["n2"].depends_on == ("n1",)
    assert len(repairs) == 1 and "self" in repairs[0].lower()


# --------------------------------------------------------------------------- #
# 5. repair is NARROW: duplicate id / real cycle still RAISE (retry backstop)
# --------------------------------------------------------------------------- #
def test_safe_fallback_still_rejects_duplicate_id() -> None:
    f = _factory()
    dup = {"nodes": [{"id": "n1", "task": "a"}, {"id": "n1", "task": "b"}]}
    with pytest.raises(PlanError, match="duplicate node id"):
        f.parse_dag_safe(dup)  # not a dangling edge → repair must not mask it


def test_safe_fallback_still_rejects_real_cycle() -> None:
    f = _factory()
    # n1 <-> n2 is a real cycle among RESOLVABLE edges — repair must not touch it.
    cycle = {
        "nodes": [
            {"id": "n1", "task": "a", "depends_on": ["n2"]},
            {"id": "n2", "task": "b", "depends_on": ["n1"]},
        ]
    }
    with pytest.raises(PlanError, match="cycle"):
        f.parse_dag_safe(cycle)


def test_safe_fallback_still_rejects_empty_plan() -> None:
    f = _factory()
    with pytest.raises(PlanError):
        f.parse_dag_safe({"nodes": []})


# --------------------------------------------------------------------------- #
# 6. WIRED PATH: Planner.plan repairs a dangling-edge emission (no failure)
# --------------------------------------------------------------------------- #
def _planner(reply: dict) -> Planner:
    from llm_framework import FakeTransport

    return Planner(FakeTransport([json.dumps(reply)]), _factory())


def test_planner_plan_repairs_dangling_emission() -> None:
    """The a2 scenario end-to-end: a phantom-edge plan no longer fails the run."""
    result = asyncio.run(_planner(_DANGLING).plan("some goal"))
    dag = result.dag
    assert dag.by_id["n3"].depends_on == ("n2",)  # phantom dropped, run survives
    assert [n.id for n in dag.topo_order()] == ["n1", "n2", "n3"]


def test_planner_plan_still_raises_on_duplicate_id() -> None:
    """Retry-on-reject backstop: a malformation repair can't fix still surfaces."""
    dup = {"nodes": [{"id": "n1", "task": "a"}, {"id": "n1", "task": "b"}]}
    with pytest.raises(MalformedOutputError):
        asyncio.run(_planner(dup).plan("some goal"))
