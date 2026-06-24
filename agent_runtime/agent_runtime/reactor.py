"""EVENT-DRIVEN PLANNER REACTION — the planner SUBSCRIBES and REACTS (P2.2, d129.2/d132.B).

Phase-1 healed a failed node by calling the planner SYNCHRONOUSLY in the runtime's
own call stack (``_heal_failed_node`` -> ``HealRouter.route`` -> ``planner.heal_decision``
inline). The audit (d128/d153) named this the gap: the runtime is a one-way publisher
and the planner is an in-call collaborator, NOT an event-subscribing reactor.

This module closes that gap WITHOUT rewriting the proven enactment. :class:`PlannerReactor`
is a real EventPlane SUBSCRIBER: it runs its own background loop over the in-process
plane and REACTS to two control-plane signals (d129.2 — keep it simple):

* :data:`~agent_runtime.heal_router.EVENT_NODE_FAILURE_DETECTED` — a worker failed.
  The reactor asks the planner for the heal DECISION (via the same
  :class:`~agent_runtime.heal_router.HealRouter` mapping, so the decision logic and its
  safe fallback are byte-identical to Phase-1) and RESOLVES a per-node future the
  runtime awaits. Because the runtime emits this the instant a node task completes
  (``asyncio.wait(FIRST_COMPLETED)``), a failed PARALLEL node is decided the moment it
  fails — BEFORE a slow sibling or the join node runs (recover-before-the-join).

* :data:`EVENT_NODE_CLARIFICATION` — a worker flagged it needs a user clarification.
  The reactor SURFACES it (publishes :data:`~agent_runtime.clarification.EVENT_NEEDS_CLARIFICATION`
  on the plane + invokes an optional callback) and RETURNS — it never cancels the drive
  loop, so the OTHER workers keep running while the user is asked (d129.2 HITL).

The split honours d1: the planner owns the DECISION (here, reacting to an event); the
runtime owns the state-mutating ENACTMENT (retry / replan / abort). The reactor only
DECIDES and surfaces — it never mutates DAG state from inside the subscription.

In-process + dependency-light (d2/d10): one asyncio subscription task, in-memory
futures, no process boundary.
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional, Sequence

from .clarification import EVENT_NEEDS_CLARIFICATION
from .heal_router import (
    EVENT_NODE_FAILURE_DETECTED,
    HealRoute,
    HealRouter,
)

# A worker flags mid-run that it needs a user clarification (distinct from the
# pre-plan ambiguity gate). The reactor SURFACES it while siblings keep working.
EVENT_NODE_CLARIFICATION = "agent_node_clarification"

# The control-plane kinds the reactor subscribes to (both low-frequency).
REACTOR_KINDS: tuple[str, ...] = (EVENT_NODE_FAILURE_DETECTED, EVENT_NODE_CLARIFICATION)

ClarificationCallback = Callable[[dict[str, Any]], Any]


class PlannerReactor:
    """Subscribe to the EventPlane and REACT — the event-driven planner loop.

    Parameters
    ----------
    planner_or_router:
        Either a planner (a :class:`HealRouter` is built around it) or an
        already-built :class:`HealRouter`. Its ``route`` is the SAME heal decision +
        safe fallback Phase-1 used, so a wired reactor never decides WORSE than the
        synchronous path — only the trigger changes (event vs in-call).
    plane:
        The in-process :class:`~reactive_tools.event_plane.EventPlane` to subscribe on
        (the run's plane — the one the runtime emits failures on).
    on_clarification:
        Optional callback invoked with the clarification payload when a worker flags
        one (in addition to publishing :data:`EVENT_NEEDS_CLARIFICATION`).
    decision_timeout:
        Seconds the runtime's :meth:`await_route` waits for the reactor's decision
        before falling back to a safe replan (so a stalled planner can never wedge a
        run). ``None`` waits forever.
    """

    def __init__(
        self,
        planner_or_router: Any,
        plane: Any,
        *,
        max_retries: int = 1,
        on_clarification: Optional[ClarificationCallback] = None,
        decision_timeout: Optional[float] = 60.0,
    ) -> None:
        if isinstance(planner_or_router, HealRouter):
            self._router = planner_or_router
        else:
            self._router = HealRouter(planner_or_router, max_retries=max_retries)
        self.plane = plane
        self.on_clarification = on_clarification
        self.decision_timeout = decision_timeout
        self._sub: Any = None
        self._task: Optional[asyncio.Task] = None
        self._routes: dict[str, asyncio.Future] = {}
        # Observable record of every clarification surfaced this run.
        self.clarifications: list[dict[str, Any]] = []
        self._running = False

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        """Subscribe to the plane and launch the background reaction loop."""
        if self._running:
            return
        self._sub = self.plane.subscribe(REACTOR_KINDS)
        self._running = True
        self._task = asyncio.create_task(self._react_loop(), name="planner-reactor")

    async def stop(self) -> None:
        """Close the subscription and await the loop's teardown (no orphan task)."""
        if not self._running:
            return
        self._running = False
        if self._sub is not None:
            self._sub.close()
        if self._task is not None:
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None
        # Fail any still-pending route futures with a safe fallback so a runtime
        # awaiting one at teardown is never left hanging.
        for nid, fut in list(self._routes.items()):
            if not fut.done():
                fut.set_result(self._fallback_route("reactor stopped"))
        self._routes.clear()

    # ------------------------------------------------------------------ #
    # the subscription loop — REACT to plane events
    # ------------------------------------------------------------------ #
    async def _react_loop(self) -> None:
        try:
            async for event in self._sub:
                try:
                    if event.kind == EVENT_NODE_FAILURE_DETECTED:
                        await self._on_failure(dict(event.payload or {}))
                    elif event.kind == EVENT_NODE_CLARIFICATION:
                        await self._on_clarification(dict(event.payload or {}))
                except Exception:  # noqa: BLE001 — a reaction must never kill the loop
                    # On a failure-event reaction error, resolve the waiter with the
                    # safe fallback so the runtime never hangs.
                    nid = ""
                    try:
                        nid = str((event.payload or {}).get("node_id") or "")
                    except Exception:  # noqa: BLE001
                        nid = ""
                    fut = self._routes.get(nid)
                    if fut is not None and not fut.done():
                        fut.set_result(self._fallback_route("reaction error"))
        except (asyncio.CancelledError, StopAsyncIteration):
            return

    async def _on_failure(self, payload: dict[str, Any]) -> None:
        """React to a node FAILURE: ask the planner for the heal decision, resolve."""
        node_id = str(payload.get("node_id") or "")
        fut = self._routes.get(node_id)
        if fut is None or fut.done():
            # No runtime is awaiting this node (e.g. a failure event with no
            # corresponding await_route) — nothing to resolve; still observable.
            return
        route = await self._router.route(
            str(payload.get("task") or node_id),
            str(payload.get("error") or ""),
            attempt=int(payload.get("attempt") or 0),
            completed=list(payload.get("completed") or []),
        )
        if not fut.done():
            fut.set_result(route)

    async def _on_clarification(self, payload: dict[str, Any]) -> None:
        """SURFACE a worker clarification WITHOUT blocking the drive loop (d129.2).

        Records it, fires the optional callback, and publishes the user-facing
        :data:`EVENT_NEEDS_CLARIFICATION`. It returns immediately — the runtime's
        drive loop is never paused, so sibling workers keep running while the user is
        asked."""
        self.clarifications.append(dict(payload))
        if self.on_clarification is not None:
            try:
                res = self.on_clarification(payload)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:  # noqa: BLE001 — surfacing must never break the run
                pass
        try:
            await self.plane.publish(
                EVENT_NEEDS_CLARIFICATION, dict(payload), source="planner-reactor"
            )
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # the runtime-facing await surface
    # ------------------------------------------------------------------ #
    def expect(self, node_id: str) -> asyncio.Future:
        """Register (and return) the future the reactor resolves for ``node_id``.

        Called by the runtime BEFORE it emits the failure event, so the reaction has
        somewhere to deliver its decision."""
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._routes[node_id] = fut
        return fut

    async def await_route(self, node_id: str) -> HealRoute:
        """Await the reactor's heal decision for ``node_id`` (with the safe timeout).

        The runtime calls this after emitting :data:`EVENT_NODE_FAILURE_DETECTED`. If
        the decision does not arrive within ``decision_timeout`` it returns the safe
        replan fallback (never wedges the run)."""
        fut = self._routes.get(node_id)
        if fut is None:
            fut = self.expect(node_id)
        try:
            if self.decision_timeout is not None:
                route = await asyncio.wait_for(asyncio.shield(fut), timeout=self.decision_timeout)
            else:
                route = await fut
        except asyncio.TimeoutError:
            route = self._fallback_route("reactor decision timed out")
        finally:
            self._routes.pop(node_id, None)
        return route

    @staticmethod
    def _fallback_route(reason: str) -> HealRoute:
        """The safe default decision (a corrective replan) — mirrors HealRouter's."""
        return HealRoute(
            action="pivot",
            kind="replan",
            rationale=f"event-driven heal unavailable ({reason}); defaulting to replan",
            fallback=True,
        )


__all__ = [
    "PlannerReactor",
    "EVENT_NODE_CLARIFICATION",
    "REACTOR_KINDS",
]
