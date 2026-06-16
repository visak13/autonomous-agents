"""Deterministic, model-INDEPENDENT plan scheduler (ported from eda-base3 FSM).

eda-base3 decides what a plan runs NEXT with two pure functions that carry NO
model call — ``fsm/plan_fsm.py``'s :func:`_first_ready_action` (the wave-dispatch
readiness gate) and :func:`plan_next_action` (the deterministic dispatch FSM). The
blueprint's cross-cutting principle (a2): everything DETERMINISTIC in eda-base3
ports as PURE PYTHON unchanged — Gemma replaces ONLY the judgment points. Shape
SELECTION is a Gemma judgment point (see :mod:`agent_runtime.shape_selector`); the
DISPATCH that follows is deterministic and lives here.

This module is that port for the non-deep-research shapes. It turns a plan SHAPE's
execution DISCIPLINE into the exact set of nodes the runtime may LAUNCH on a given
turn:

* ``linear`` → :data:`ExecutionMode.SEQUENTIAL`: at most ONE ready node in flight
  at a time — the runtime launches the :func:`first_ready_action` (the first
  pending node whose dependencies are all done), waits for it, then the next.
  Strict single-file order, no fan-out.
* ``modular-parallel`` → :data:`ExecutionMode.CONCURRENT`: EVERY independent ready
  node launches at once — the wave the existing runtime already drives.

It consumes ONLY the :class:`~agent_runtime.factory.PlanDAG` + the done / running
id sets — no transport, no I/O, no model — so it is deterministic and fully
testable offline. The ``deep-research`` (cyclic) shape is UNROLLED into a bounded
acyclic role-tagged DAG by :func:`~agent_runtime.shapes.unroll_shape` and then
dispatched through THIS same scheduler like any other DAG (a3 re-architecture —
there is no per-shape executor); the unroll's growing-visibility edges make every
round depend on the prior ones, so the rounds run in order under either mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from .factory import PlanDAG, PlanNode


class ExecutionMode(str, Enum):
    """How the runtime dispatches a shape's ready nodes (blueprint §2a)."""

    SEQUENTIAL = "sequential"   # linear: one node in flight at a time
    CONCURRENT = "concurrent"   # modular-parallel: every ready node at once


# A shape's declared ``execution`` token → the runtime dispatch mode. Aliases are
# accepted so a shape file may name the shape OR its discipline. ``deep-research``
# maps to CONCURRENT: once UNROLLED its growing-visibility edges fully serialise
# the rounds, so the concurrent wave-dispatch runs them strictly in order anyway.
_MODE_BY_TOKEN: dict[str, ExecutionMode] = {
    "sequential": ExecutionMode.SEQUENTIAL,
    "linear": ExecutionMode.SEQUENTIAL,
    "concurrent": ExecutionMode.CONCURRENT,
    "modular-parallel": ExecutionMode.CONCURRENT,
    "modular": ExecutionMode.CONCURRENT,
    "parallel": ExecutionMode.CONCURRENT,
    "deep-research": ExecutionMode.CONCURRENT,
}


def execution_mode_for(value: Optional[str]) -> ExecutionMode:
    """Map a shape's ``execution`` token to an :class:`ExecutionMode`.

    Unknown/empty falls back to CONCURRENT (the legacy runtime behaviour), so a
    runtime built without a shape behaves exactly as before. A shape FILE's token
    is already validated fail-fast by :class:`~agent_runtime.shapes.ShapeSpec`."""
    if not value:
        return ExecutionMode.CONCURRENT
    return _MODE_BY_TOKEN.get(str(value).strip().lower(), ExecutionMode.CONCURRENT)


def first_ready_action(
    dag: PlanDAG, done: Iterable[str], blocked: Iterable[str] = ()
) -> Optional[PlanNode]:
    """The FIRST pending node whose every dependency is done — eda-base3's gate.

    A faithful pure-python port of ``fsm/plan_fsm.py:_first_ready_action``: scan
    the nodes IN ORDER and return the first whose ``depends_on`` are all satisfied
    (``done``) and which is neither already done nor ``blocked`` (an id to skip —
    e.g. a failed/skipped node). Deterministic: node-list order is the tie-break,
    so the same plan always yields the same next node. Returns ``None`` when no
    node is ready (all done, or the rest are waiting on unfinished/failed deps)."""
    done_set = set(done)
    blocked_set = set(blocked)
    for n in dag.nodes:
        if n.id in done_set or n.id in blocked_set:
            continue
        if all(d in done_set for d in n.depends_on):
            return n
    return None


def ready_wave(
    dag: PlanDAG, done: Iterable[str], blocked: Iterable[str] = ()
) -> list[PlanNode]:
    """EVERY ready node, in node order — the modular-parallel dispatch wave.

    The set form of :func:`first_ready_action`: all pending nodes whose
    dependencies are satisfied and which are not ``blocked``. This is the wave the
    runtime launches together under CONCURRENT execution."""
    done_set = set(done)
    blocked_set = set(blocked)
    return [
        n
        for n in dag.nodes
        if n.id not in done_set
        and n.id not in blocked_set
        and all(d in done_set for d in n.depends_on)
    ]


@dataclass(frozen=True)
class Dispatch:
    """The scheduler's decision for ONE driver turn (deterministic).

    ``nodes`` is the ordered tuple of nodes the runtime may LAUNCH this turn
    (possibly empty — the driver then waits on whatever is already running, or
    terminates if nothing is running and nothing is launchable)."""

    nodes: tuple[PlanNode, ...] = ()

    @property
    def has_work(self) -> bool:
        return bool(self.nodes)


def next_dispatch(
    dag: PlanDAG,
    done: Iterable[str],
    running: Iterable[str] = (),
    blocked: Iterable[str] = (),
    *,
    mode: ExecutionMode = ExecutionMode.CONCURRENT,
) -> Dispatch:
    """The nodes the runtime may LAUNCH on this turn, honouring the shape's mode.

    The deterministic dispatch decision — the port of ``plan_next_action``'s
    DISPATCH branch, generalised over the execution discipline:

    * SEQUENTIAL (linear): at most ONE node in flight. If anything is already
      ``running``, launch NOTHING (wait for it); otherwise launch ONLY the
      :func:`first_ready_action`. Strict single-file order.
    * CONCURRENT (modular-parallel): launch the WHOLE :func:`ready_wave` not
      already running — every independent ready node at once (byte-identical to
      the legacy runtime's dispatch).

    Already-``running`` nodes are never re-dispatched. Pure: no model, no I/O."""
    running_set = set(running)
    wave = [n for n in ready_wave(dag, done, blocked) if n.id not in running_set]
    if mode == ExecutionMode.SEQUENTIAL:
        if running_set:
            return Dispatch(())
        return Dispatch(tuple(wave[:1]))
    return Dispatch(tuple(wave))


def is_complete(
    dag: PlanDAG, done: Iterable[str], running: Iterable[str] = (), blocked: Iterable[str] = ()
) -> bool:
    """True when no node is running and none can be launched — the plan is settled.

    The terminal condition of ``plan_next_action`` (nothing in flight, nothing
    ready): every node is either done or permanently blocked by an unfinished/
    failed dependency. The runtime uses this to stop the drive loop."""
    if set(running):
        return False
    return not ready_wave(dag, done, blocked)


__all__ = [
    "ExecutionMode",
    "execution_mode_for",
    "first_ready_action",
    "ready_wave",
    "Dispatch",
    "next_dispatch",
    "is_complete",
]
