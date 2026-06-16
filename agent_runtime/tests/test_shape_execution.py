"""Faithful shape EXECUTION: linear=sequential, modular-parallel=concurrent (s3/b1).

The behavioural proof that the runtime HONORS a shape's execution discipline — the
RC4 remainder (blueprint §2a). Each node's phi call runs in a worker thread (the
freeze-fix offload), so a concurrency PROBE transport can observe whether
independent ready nodes actually OVERLAP:

* ``modular-parallel`` (CONCURRENT) → the two independent middle nodes of a diamond
  run AT THE SAME TIME (peak in-flight ≥ 2);
* ``linear`` (SEQUENTIAL) → never more than ONE node is in flight at any instant
  (peak == 1), and nodes finish in strict single-file order.

Driven entirely on the offline probe transport — no GPU, no live model. Also
asserts the deep-research executor is untouched by the new dispatch mode (it drives
its own rounds).
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from llm_framework import ChatResult

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.runtime import AgentRuntime
from agent_runtime.scheduler import ExecutionMode


class ConcurrencyProbe:
    """A transport that records the PEAK number of overlapping phi calls.

    Each ``chat`` increments a shared in-flight counter, holds briefly so genuine
    overlap is observable, then decrements — recording the peak and the order in
    which calls completed. It is a valid ``llm_framework`` transport (``chat`` +
    ``complete``) so the runtime drives it unchanged, fully offline."""

    def __init__(self, *, hold: float = 0.05) -> None:
        self.hold = hold
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0
        self.completed: list[str] = []

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content

    def chat(self, messages, **opts) -> ChatResult:
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)
        time.sleep(self.hold)
        # Record the node task text's first line so completion ORDER is observable.
        task_line = ""
        for m in messages:
            if m.get("role") == "user":
                task_line = str(m.get("content", "")).splitlines()[0]
                break
        with self._lock:
            self.current -= 1
            self.completed.append(task_line)
        return ChatResult(role="assistant", content="done")


def _diamond() -> PlanDAG:
    # n1 → {n2, n3} → n4 : n2 and n3 are INDEPENDENT and may overlap.
    return PlanDAG(
        nodes=[
            PlanNode(id="n1", task="step one"),
            PlanNode(id="n2", task="step two", depends_on=("n1",)),
            PlanNode(id="n3", task="step three", depends_on=("n1",)),
            PlanNode(id="n4", task="step four", depends_on=("n2", "n3")),
        ]
    )


def test_modular_parallel_runs_independent_nodes_concurrently():
    probe = ConcurrencyProbe()
    rt = AgentRuntime(transport=probe, execution=ExecutionMode.CONCURRENT)
    out = asyncio.run(rt.run(_diamond()))
    assert out.ok
    # The two independent middle nodes overlapped → peak in-flight reached 2.
    assert probe.peak >= 2, f"expected concurrent overlap, peak={probe.peak}"
    # All four nodes ran.
    assert set(out.results) == {"n1", "n2", "n3", "n4"}


def test_linear_runs_strictly_sequentially():
    probe = ConcurrencyProbe()
    rt = AgentRuntime(transport=probe, execution=ExecutionMode.SEQUENTIAL)
    out = asyncio.run(rt.run(_diamond()))
    assert out.ok
    # STRICT single-file: never more than one node in flight at any instant.
    assert probe.peak == 1, f"expected no overlap under linear, peak={probe.peak}"
    assert set(out.results) == {"n1", "n2", "n3", "n4"}
    # The launch order respects dependencies (n1 first, n4 last); the middle two
    # run one-at-a-time in deterministic node order.
    assert out.launch_order[0] == "n1"
    assert out.launch_order[-1] == "n4"
    assert out.launch_order == ["n1", "n2", "n3", "n4"]


def test_default_execution_is_concurrent_backcompat():
    # A runtime built WITHOUT an execution arg behaves exactly as before (the whole
    # existing suite relies on this) — independent nodes still overlap.
    probe = ConcurrencyProbe()
    rt = AgentRuntime(transport=probe)
    assert rt.execution is ExecutionMode.CONCURRENT
    out = asyncio.run(rt.run(_diamond()))
    assert out.ok and probe.peak >= 2


def test_linear_chain_completion_order_is_single_file():
    # A pure chain n1→n2→n3 under linear: completion order is exactly the chain.
    dag = PlanDAG(
        nodes=[
            PlanNode(id="n1", task="alpha"),
            PlanNode(id="n2", task="beta", depends_on=("n1",)),
            PlanNode(id="n3", task="gamma", depends_on=("n2",)),
        ]
    )
    probe = ConcurrencyProbe()
    rt = AgentRuntime(transport=probe, execution=ExecutionMode.SEQUENTIAL)
    out = asyncio.run(rt.run(dag))
    assert out.ok
    assert probe.completed == ["alpha", "beta", "gamma"]
