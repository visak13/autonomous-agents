"""P2.2 ORCHESTRATION — fast offline proofs (d129/d132.B, P2-2-orch).

The event-driven planner + the carry-forward, each proven OFFLINE (FakeTransport,
in-process EventPlane; no Ollama / network / GPU):

1. EVENT-DRIVEN PLANNER (``PlannerReactor``): the planner SUBSCRIBES to the EventPlane
   and REACTS — a node FAILURE event is decided by the reactor (not a synchronous
   in-call), recovering a PARALLEL node before the join; a worker CLARIFICATION is
   SURFACED while other workers keep being serviced.
4. CARRY-FORWARD: ``get_shapes`` / ``get_specs`` are reachable on the SERVED planner
   surface (built by ``chat_app.app.build_wiring``) + in the planner context.

(The framework-injected review parts (2/3) were retired in SF-1/d310/d311 — the model
authors the whole document; no engine reviewer injection or verdict fold remains.)
"""
from __future__ import annotations

import asyncio
import json

from agent_runtime import stub
from agent_runtime.clarification import EVENT_NEEDS_CLARIFICATION
from agent_runtime.factory import AbstractPlanFactory, PlanDAG, PlanNode
from agent_runtime.heal_router import EVENT_NODE_FAILURE_DETECTED, HealRouter
from agent_runtime.planner import HealDecision, Planner
from agent_runtime.reactor import EVENT_NODE_CLARIFICATION, PlannerReactor
from agent_runtime.runtime import AgentRuntime
from agent_runtime.status import NodeStatus
from llm_framework import FakeTransport
from reactive_tools import EventPlane


# --------------------------------------------------------------------------- #
# helpers (mirror test_heal_router.py's offline harness)
# --------------------------------------------------------------------------- #
class _FixedPlanner:
    """A planner whose heal_decision always returns ``action`` (no model call)."""

    def __init__(self, action: str = "retry") -> None:
        self._action = action
        self.calls = 0

    async def heal_decision(self, *a, **k):
        self.calls += 1
        return HealDecision(action=self._action, rationale="r")


def _planner(*replies: str) -> Planner:
    return Planner(FakeTransport(list(replies)), AbstractPlanFactory([]))


def _rejects_marker(marker: str = "FAIL"):
    def validate(_node, result):
        return f"contains {marker!r}" if marker in (result.output or "") else None

    return validate


# =========================================================================== #
# PART 1a — the reactor SUBSCRIBES to the plane and REACTS to a failure event
# =========================================================================== #
def test_reactor_reacts_to_failure_event_with_planner_decision():
    async def _run():
        plane = EventPlane()
        reactor = PlannerReactor(_FixedPlanner("retry"), plane)
        await reactor.start()
        # The runtime-side handshake: register the waiter, EMIT the failure event,
        # then await the reactor's reaction (the planner deciding via subscription).
        reactor.expect("n1")
        await plane.publish(
            EVENT_NODE_FAILURE_DETECTED,
            {"node_id": "n1", "task": "do x", "error": "boom", "attempt": 0, "completed": []},
        )
        route = await reactor.await_route("n1")
        await reactor.stop()
        return route

    route = asyncio.run(_run())
    # The decision came FROM a plane-event reaction, not a synchronous call.
    assert route.is_retry
    assert route.action == "retry"


def test_reactor_pivot_event_maps_to_replan():
    async def _run():
        plane = EventPlane()
        reactor = PlannerReactor(_FixedPlanner("pivot"), plane)
        await reactor.start()
        reactor.expect("n2")
        await plane.publish(
            EVENT_NODE_FAILURE_DETECTED,
            {"node_id": "n2", "task": "t", "error": "e", "attempt": 0, "completed": []},
        )
        route = await reactor.await_route("n2")
        await reactor.stop()
        return route

    assert asyncio.run(_run()).is_replan


# =========================================================================== #
# PART 1b — END-TO-END: the event-driven reactor recovers a PARALLEL node, and
# the JOIN consumes the recovered result (recover-before-the-join).
# =========================================================================== #
def test_reactor_recovers_parallel_node_before_join_on_real_runtime():
    # n1 ∥ n2 (independent), n3 joins both. n1 fails once then recovers; the heal
    # DECISION is obtained via the event-driven reactor (not heal_router). n3 must
    # run AFTER n1 recovered (the join sees the recovered parallel node).
    subagent = FakeTransport(["FAIL once", "n2 ok", "n1 recovered", "n3 joined output"])
    fixed = _FixedPlanner("retry")
    plane = EventPlane()

    async def replanner_adapter(node, err, completed):  # pragma: no cover - retry path
        return await _planner(json.dumps(stub.canned_replan(task="x"))).replan_subgraph(
            node.task, err, spec=node.primary_spec, completed=completed
        )

    rt = AgentRuntime(
        transport=subagent,
        plane=plane,
        replanner=replanner_adapter,
        result_validator=_rejects_marker(),
        planner_reactor=PlannerReactor(fixed, plane),
        max_heals=0,  # a validator rejection goes straight to the heal seam
    )
    dag = PlanDAG(nodes=[
        PlanNode(id="n1", task="Gather A."),
        PlanNode(id="n2", task="Gather B."),
        PlanNode(id="n3", task="Combine A and B.", depends_on=("n1", "n2")),
    ])

    out = asyncio.run(rt.run(dag))

    assert out.states["n1"]["status"] == NodeStatus.DONE.value
    assert out.states["n1"]["attempts"] == 2          # failed once, recovered via retry
    assert out.states["n2"]["status"] == NodeStatus.DONE.value
    assert out.states["n3"]["status"] == NodeStatus.DONE.value   # join consumed recovery
    assert out.replans_used == 0
    assert fixed.calls == 1                            # the planner REACTED exactly once


def test_no_reactor_keeps_synchronous_heal_path_unchanged():
    # Without a reactor the runtime uses the Phase-1 synchronous heal_router — byte
    # compatible (the reactor is strictly additive, no regression).
    subagent = FakeTransport(["FAIL once", "good answer"])
    fixed = _FixedPlanner("retry")
    rt = AgentRuntime(
        transport=subagent,
        result_validator=_rejects_marker(),
        heal_router=HealRouter(fixed),
        max_heals=0,
    )
    out = asyncio.run(rt.run(PlanDAG(nodes=[PlanNode(id="n1", task="Answer.")])))
    assert out.states["n1"]["status"] == NodeStatus.DONE.value
    assert fixed.calls == 1


# =========================================================================== #
# PART 1c — a worker CLARIFICATION is SURFACED while other workers keep going
# =========================================================================== #
def test_reactor_surfaces_clarification_without_blocking_other_work():
    async def _run():
        plane = EventPlane()
        surfaced: list[dict] = []
        sub = plane.subscribe([EVENT_NEEDS_CLARIFICATION])
        seen_cb: list[dict] = []
        reactor = PlannerReactor(
            _FixedPlanner("retry"), plane, on_clarification=lambda p: seen_cb.append(p)
        )
        await reactor.start()

        # A worker flags a clarification mid-run.
        await plane.publish(
            EVENT_NODE_CLARIFICATION,
            {"node_id": "nA", "question": "Which timeframe?"},
        )
        # The reactor must SURFACE it (publish the user-facing event) without blocking:
        # immediately after, a DIFFERENT failing worker (nB) is still serviced.
        reactor.expect("nB")
        await plane.publish(
            EVENT_NODE_FAILURE_DETECTED,
            {"node_id": "nB", "task": "t", "error": "e", "attempt": 0, "completed": []},
        )
        route_b = await reactor.await_route("nB")

        # Drain the surfaced clarification event.
        try:
            ev = await asyncio.wait_for(sub.get(), timeout=1.0)
            surfaced.append(dict(ev.payload))
        except asyncio.TimeoutError:
            pass
        await reactor.stop()
        return reactor.clarifications, surfaced, seen_cb, route_b

    clarifications, surfaced, seen_cb, route_b = asyncio.run(_run())
    assert any(c.get("node_id") == "nA" for c in clarifications)   # recorded
    assert any(s.get("question") == "Which timeframe?" for s in surfaced)  # surfaced to user
    assert seen_cb and seen_cb[0]["node_id"] == "nA"               # callback fired
    assert route_b.is_retry   # the OTHER worker was still serviced (not blocked)




# =========================================================================== #
# PART 4 — CARRY-FORWARD: get_shapes / get_specs reachable on the SERVED surface
# =========================================================================== #
def test_discovery_tools_live_on_served_planner_surface(tmp_path):
    from chat_app.app import build_wiring

    w = build_wiring(data_dir=str(tmp_path))
    catalog = {t["name"] for t in w.hook.registry.catalog()}
    # registered on the served hook => in the body-free tool catalog the planner sees
    assert "get_shapes" in catalog
    assert "get_specs" in catalog
    # the AbstractPlanFactory injects exactly that catalog into the planner CONTEXT
    factory = AbstractPlanFactory([], tool_catalog=w.hook.registry.catalog())
    tools = {t["name"] for t in factory.planner_context("goal")["tools"]}
    assert {"get_shapes", "get_specs"} <= tools
    # and both are dispatchable (reachable) through the hook on the event plane
    shapes = asyncio.run(w.hook.invoke("get_shapes"))
    specs = asyncio.run(w.hook.invoke("get_specs"))
    assert shapes.value["count"] >= 1          # the packaged shape catalog
    assert "specs" in specs.value
