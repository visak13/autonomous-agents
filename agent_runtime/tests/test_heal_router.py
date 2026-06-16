"""b4: reactive self-heal WIRING — a node FAILURE routes to the planner's heal
decision, and the runtime ENACTS the routed action (blueprint §2e, d1).

a2 proved ``Planner.heal_decision`` (the native enum retry|pivot|extend|abort) +
recovery MANUALLY. b4 wires it to fire on the live runtime: when a node fails,
the runtime ROUTES the failure to ``HealRouter`` (planner-owned decision) and
enacts the routed kind —

    retry  → idempotent RE-DISPATCH of the SAME node (done nodes preserved);
    pivot/extend → ``replan_subgraph`` corrective sub-DAG;
    abort  → surface the failure (no recovery).

The reactive heal RULE registered on the EventPlane/LambdaRegistry only OBSERVES
the routing (advisory; it never mutates — d1). Fully OFFLINE: FakeTransport, no
Ollama / network / GPU (d7/d8).
"""
from __future__ import annotations

import asyncio
import json

from agent_runtime import stub
from agent_runtime.factory import AbstractPlanFactory, PlanDAG, PlanNode
from agent_runtime.heal_router import (
    EVENT_HEAL_ROUTED,
    EVENT_NODE_FAILURE_DETECTED,
    HealRoute,
    HealRouter,
    register_heal_rule,
)
from agent_runtime.planner import HealDecision, Planner
from agent_runtime.runtime import AgentRuntime
from agent_runtime.selfheal import MalformedOutputError
from agent_runtime.status import NodeStatus
from llm_framework import FakeTransport
from reactive_tools import EventPlane, LambdaRegistry


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _planner(*replies: str) -> Planner:
    """A planner whose phi calls (heal_decision / replan) return ``replies`` in order,
    grounded on an EMPTY body-free spec index (d10)."""
    return Planner(FakeTransport(list(replies)), AbstractPlanFactory([]))


def _rejects_marker(marker: str = "FAIL"):
    """A result validator that rejects any output containing ``marker`` (so a node
    fails deterministically) — the InvalidStepError path of the runtime."""

    def validate(_node, result):
        return f"contains {marker!r}" if marker in (result.output or "") else None

    return validate


def _heal_runtime(subagent_transport, planner, **kw) -> AgentRuntime:
    """A runtime wired for reactive heal: a heal_router + the replanner adapter, with
    node-level self-heal OFF (``max_heals=0``) so a validator rejection propagates
    straight to the heal seam (not absorbed by the inner retry)."""

    async def replanner_adapter(node, err, completed):
        return await planner.replan_subgraph(
            node.task, err, spec=node.primary_spec, completed=completed
        )

    return AgentRuntime(
        transport=subagent_transport,
        replanner=replanner_adapter,
        result_validator=_rejects_marker(),
        heal_router=HealRouter(planner),
        max_heals=0,
        **kw,
    )


# --------------------------------------------------------------------------- #
# 1) retry → idempotent re-dispatch (no replan); DONE nodes preserved
# --------------------------------------------------------------------------- #
def test_retry_route_redispatches_same_node_and_recovers():
    # n1 produces a rejected output, then a good one on re-dispatch. The planner's
    # heal decision is 'retry' → the runtime re-runs the SAME node (no replan).
    subagent = FakeTransport(["FAIL once", "a good, real answer"])
    planner = _planner(json.dumps({"action": "retry", "rationale": "transient"}))
    rt = _heal_runtime(subagent, planner)
    dag = PlanDAG(nodes=[PlanNode(id="n1", task="Answer the question.")])

    out = asyncio.run(rt.run(dag))

    st = out.states["n1"]
    assert st["status"] == NodeStatus.DONE.value
    assert st["attempts"] == 2          # two REAL executions (fail + re-dispatch)
    assert st["replanned"] is False     # recovered by retry, NOT replan
    assert out.replans_used == 0
    assert out.results["n1"].output == "a good, real answer"
    assert planner.transport.call_count == 1  # exactly one heal_decision, no replan


def test_retry_preserves_already_done_upstream_node():
    # n1 succeeds (DONE, cached); n2 fails then recovers via a retry re-dispatch.
    # n1 must NOT be re-executed across the heal (done nodes preserved).
    subagent = FakeTransport(["n1 good output", "FAIL once", "n2 good output"])
    planner = _planner(json.dumps({"action": "retry", "rationale": "transient"}))
    rt = _heal_runtime(subagent, planner)
    dag = PlanDAG(
        nodes=[
            PlanNode(id="n1", task="First step."),
            PlanNode(id="n2", task="Second step.", depends_on=("n1",)),
        ]
    )

    out = asyncio.run(rt.run(dag))

    assert out.states["n1"]["status"] == NodeStatus.DONE.value
    assert out.states["n1"]["attempts"] == 1   # done node never re-run
    assert out.states["n2"]["status"] == NodeStatus.DONE.value
    assert out.states["n2"]["attempts"] == 2   # failed once, re-dispatched once
    assert out.replans_used == 0


# --------------------------------------------------------------------------- #
# 2) pivot → replan_subgraph corrective sub-DAG
# --------------------------------------------------------------------------- #
def test_pivot_route_recovers_via_replan_subgraph():
    # n1 fails; heal decision is 'pivot' → the runtime asks the planner for a
    # corrective sub-DAG (replan_subgraph) and runs it in-process.
    subagent = FakeTransport(["FAIL once", "recovered via the corrective sub-plan"])
    planner = _planner(
        json.dumps({"action": "pivot", "rationale": "approach is wrong"}),
        json.dumps(stub.canned_replan(task="do it a different way")),
    )
    rt = _heal_runtime(subagent, planner)
    dag = PlanDAG(nodes=[PlanNode(id="n1", task="Answer the question.")])

    out = asyncio.run(rt.run(dag))

    st = out.states["n1"]
    assert st["status"] == NodeStatus.DONE.value
    assert st["replanned"] is True
    assert out.replans_used == 1
    assert out.results["n1"].output == "recovered via the corrective sub-plan"
    assert planner.transport.call_count == 2  # heal_decision + one replan


# --------------------------------------------------------------------------- #
# 3) abort → surface FAILED (no recovery)
# --------------------------------------------------------------------------- #
def test_abort_route_surfaces_failed():
    subagent = FakeTransport(["FAIL once"])
    planner = _planner(json.dumps({"action": "abort", "rationale": "unrecoverable"}))
    rt = _heal_runtime(subagent, planner)
    dag = PlanDAG(nodes=[PlanNode(id="n1", task="Answer the question.")])

    out = asyncio.run(rt.run(dag))

    assert out.states["n1"]["status"] == NodeStatus.FAILED.value
    assert "n1" in out.failed
    assert out.replans_used == 0          # abort never replans
    assert not out.ok


# --------------------------------------------------------------------------- #
# 4) legacy path unchanged: NO heal_router → unconditional replan (byte-compat)
# --------------------------------------------------------------------------- #
def test_no_heal_router_keeps_legacy_unconditional_replan():
    # Without a heal_router the runtime must behave EXACTLY as pre-b4: a failure
    # goes straight to the sub-graph replan, no heal_decision call.
    subagent = FakeTransport(["FAIL once", "recovered output"])
    planner = _planner(json.dumps(stub.canned_replan(task="recover")))

    async def replanner_adapter(node, err, completed):
        return await planner.replan_subgraph(
            node.task, err, spec=node.primary_spec, completed=completed
        )

    rt = AgentRuntime(
        transport=subagent,
        replanner=replanner_adapter,
        result_validator=_rejects_marker(),
        max_heals=0,
        # heal_router omitted → legacy path
    )
    dag = PlanDAG(nodes=[PlanNode(id="n1", task="Answer.")])

    out = asyncio.run(rt.run(dag))

    assert out.states["n1"]["status"] == NodeStatus.DONE.value
    assert out.states["n1"]["replanned"] is True
    assert out.replans_used == 1
    assert planner.transport.call_count == 1  # only the replan — NO heal_decision


# --------------------------------------------------------------------------- #
# 5) HealRouter unit: action→kind mapping, budget escalation, fallback
# --------------------------------------------------------------------------- #
class _FixedPlanner:
    def __init__(self, action):
        self._action = action
        self.calls = 0

    async def heal_decision(self, *a, **k):
        self.calls += 1
        return HealDecision(action=self._action, rationale="r")


class _RaisingPlanner:
    async def heal_decision(self, *a, **k):
        raise MalformedOutputError("no legal enum after repair")


def test_router_maps_actions_to_kinds():
    assert asyncio.run(HealRouter(_FixedPlanner("retry")).route("t", "e")).is_retry
    assert asyncio.run(HealRouter(_FixedPlanner("pivot")).route("t", "e")).is_replan
    assert asyncio.run(HealRouter(_FixedPlanner("extend")).route("t", "e")).is_replan
    assert asyncio.run(HealRouter(_FixedPlanner("abort")).route("t", "e")).is_abort


def test_router_escalates_retry_past_budget_to_replan():
    r = HealRouter(_FixedPlanner("retry"), max_retries=1)
    assert asyncio.run(r.route("t", "e", attempt=0)).is_retry   # within budget
    over = asyncio.run(r.route("t", "e", attempt=1))            # budget spent
    assert over.is_replan and over.action == "retry"            # escalated, decision kept


def test_router_falls_back_to_replan_when_decision_unavailable():
    route = asyncio.run(HealRouter(_RaisingPlanner()).route("t", "e"))
    assert route.is_replan and route.fallback is True and route.action == "pivot"


def test_healroute_view_is_serialisable():
    v = HealRoute(action="retry", kind="retry", rationale="x").as_dict()
    assert v == {"action": "retry", "kind": "retry", "rationale": "x", "fallback": False}


# --------------------------------------------------------------------------- #
# 6) the reactive heal RULE on the LambdaRegistry — observe-only (d1/d15)
# --------------------------------------------------------------------------- #
def test_heal_rule_observes_routing_advisory_only():
    async def _run():
        plane = EventPlane()
        registry = LambdaRegistry(plane)
        sub_id = register_heal_rule(registry, run_id="run-x", source_plane=plane)
        assert sub_id is not None
        # The failure + the routed decision flow on the plane the rule observes.
        await plane.publish(EVENT_NODE_FAILURE_DETECTED, {"node_id": "n1", "error": "boom"})
        await plane.publish(EVENT_HEAL_ROUTED, {"node_id": "n1", "action": "retry", "kind": "retry"})
        for _ in range(20):  # let the lambda driver drain both events
            await asyncio.sleep(0)
        rec = registry.get(sub_id)
        await registry.close_all()
        return rec

    rec = asyncio.run(_run())
    assert rec is not None
    assert rec.fire_count >= 2          # observed BOTH routing events
    assert rec.reaction == "advisory"   # OBSERVE-ONLY: routes/advises, never mutates
    assert "self-heal-rule" in rec.label


def test_register_heal_rule_no_registry_is_noop():
    assert register_heal_rule(None) is None
