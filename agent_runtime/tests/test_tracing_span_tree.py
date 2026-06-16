"""Span-tree + cross-thread-propagation regression for the s6/b2 instrumentation.

b2 wired OpenTelemetry into the planner + runtime DAG. This test LOCKS the shape
that instrumentation must produce, fully OFFLINE (FakeTransport, an in-memory span
exporter — no Ollama / Phoenix / network / GPU, per d7/d8). It is the offline twin
of the b4 live Phoenix capture: same tree, proven without the collector.

What it proves:

1. **One trace tree.** ``agent.session`` (the orchestration wrapper that
   run_agentic opens) is the root; ``planner.plan`` and ``agent.run`` both nest
   under it — so planning and running land in ONE trace, not two disconnected ones.
2. **agent.run wraps the DAG.** Every per-node ``agent.node`` span is a direct
   child of ``agent.run``.
3. **Cross-thread context propagation (the load-bearing bit).** The phi call runs
   inside a worker thread (``asyncio.to_thread`` via ``run_blocking_in_span``). A
   span opened THERE — exactly as the real ``OllamaTransport.chat`` opens its
   per-call LLM span — nests under the *currently active* span (the node span for a
   node phi call, the ``planner.plan`` span for the plan call) in the SAME trace,
   instead of detaching into a separate root trace.
4. **Lifecycle as span events.** Each node span carries the
   ``pending -> in-progress -> verifiable -> done`` lifecycle as ordered events.
5. **Attributes.** ``agent.run`` carries a ``run.id`` + ``run.node_count``;
   ``planner.plan`` carries the resulting ``planner.node_count``.
"""
from __future__ import annotations

import asyncio
import json
import threading

from opentelemetry import trace as ot_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import agent_runtime.tracing as tracing
from agent_runtime.factory import AbstractPlanFactory, PlanDAG, PlanNode
from agent_runtime.planner import Planner
from agent_runtime.runtime import AgentRuntime
from agent_runtime.status import NodeStatus
from llm_framework import FakeTransport

_INDEX = [
    {"name": "markdown-writer", "description": "shape findings into GFM", "source": "seed"},
]

# A valid 2-node plan the FakeTransport hands the planner; parse_dag turns it into
# the PlanDAG the runtime then drives. Bare nodes (no spec) → no loader needed.
_PLAN_JSON = json.dumps(
    {
        "rationale": "two-step research then summarise",
        "nodes": [
            {"id": "n1", "task": "research the topic"},
            {"id": "n2", "task": "summarise the findings", "depends_on": ["n1"]},
        ],
    }
)


class _TracingFakeTransport(FakeTransport):
    """A FakeTransport that opens a ``phi.call`` span around every model call.

    This MIRRORS the real ``OllamaTransport.chat`` (which opens a per-call
    OpenInference LLM span via ``start_as_current_span``) so we can prove the
    cross-thread nesting WITHOUT live phi: ``start_as_current_span`` nests the
    span under whatever context is active when ``chat`` runs — and ``chat`` runs
    inside the worker thread, so the test fails unless b2's context re-attach put
    the node/planner span into that thread's context."""

    def chat(self, messages, **opts):  # type: ignore[override]
        tracer = tracing.get_tracer("test.phi")
        with tracer.start_as_current_span("phi.call") as sp:
            sp.set_attribute("test.thread", threading.current_thread().name)
            return super().chat(messages, **opts)


def _install_inmemory_exporter() -> InMemorySpanExporter:
    """Point agent_runtime.tracing's singleton at an in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Inject as the module singleton so get_tracer(...) returns THIS provider
    # (every span in the test lands in `exporter`). Context propagation uses the
    # global context API independently, so nesting still works.
    tracing._provider = provider
    return exporter


async def _drive_session(transport: _TracingFakeTransport) -> None:
    """Mirror run_agentic: an agent.session span wrapping plan() then run()."""
    factory = AbstractPlanFactory(_INDEX)
    planner = Planner(transport, factory)
    runtime = AgentRuntime(transport=transport)

    tracer = tracing.get_tracer("test.session")
    with tracer.start_as_current_span("agent.session"):
        plan_result = await planner.plan("research the topic")
        await runtime.run(plan_result.dag, timeout=30, run_id="run-test-b2")


def _by_id(spans):
    return {s.context.span_id: s for s in spans}


def test_b2_span_tree_nesting_and_lifecycle_events():
    exporter = _install_inmemory_exporter()
    transport = _TracingFakeTransport(
        [_PLAN_JSON, "research output n1", "summary output n2"]
    )
    asyncio.run(_drive_session(transport))

    spans = exporter.get_finished_spans()
    by_id = _by_id(spans)
    names = sorted(s.name for s in spans)

    def one(name):
        matches = [s for s in spans if s.name == name]
        assert len(matches) == 1, f"expected exactly one {name!r}, got {len(matches)} ({names})"
        return matches[0]

    def parent_of(span):
        return by_id.get(span.parent.span_id) if span.parent else None

    session = one("agent.session")
    planner_span = one("planner.plan")
    run_span = one("agent.run")
    node_spans = [s for s in spans if s.name == "agent.node"]
    phi_spans = [s for s in spans if s.name == "phi.call"]

    # 1. agent.session is the single root of the whole tree.
    assert session.parent is None

    # 2. planner.plan AND agent.run both nest under the SAME session → one trace.
    assert parent_of(planner_span) is session
    assert parent_of(run_span) is session
    assert planner_span.context.trace_id == run_span.context.trace_id == session.context.trace_id

    # 3. agent.run wraps the DAG: each node span is a direct child of agent.run.
    assert len(node_spans) == 2, names
    for ns in node_spans:
        assert parent_of(ns) is run_span
        assert ns.context.trace_id == session.context.trace_id

    # 4. CROSS-THREAD PROPAGATION: every phi.call (opened inside the worker thread)
    #    nests under the active span in the SAME trace — NOT a detached root.
    assert len(phi_spans) >= 3, f"expected >=3 phi.call spans, got {len(phi_spans)} ({names})"
    for ph in phi_spans:
        assert ph.parent is not None, "phi.call detached into a root trace (propagation broke)"
        assert ph.context.trace_id == session.context.trace_id
        # ...and it ran on a worker thread, not the event-loop thread.
        assert ph.attributes.get("test.thread") != threading.current_thread().name

    parents = {parent_of(ph).name for ph in phi_spans}
    # the planner's structured call nests under planner.plan; node calls under agent.node.
    assert "planner.plan" in parents
    assert "agent.node" in parents

    # 5. Each node span records the full lifecycle as ORDERED events.
    for ns in node_spans:
        ev = [e.name for e in ns.events]
        assert ev[:2] == ["pending", "in-progress"], ev
        assert "verifiable" in ev, ev
        assert ev[-1] == "done", ev
        assert ev.index("verifiable") < ev.index("done")

    # 6. Key attributes are populated.
    assert run_span.attributes.get("run.id") == "run-test-b2"
    assert run_span.attributes.get("run.node_count") == 2
    assert planner_span.attributes.get("planner.node_count") == 2
    assert planner_span.attributes.get("planner.goal")


def test_b2_inline_review_phi_also_nests_under_node_span():
    """The CODER=REVIEWER inline-fix phi call runs in the DRIVER context, not the
    node task. b2 re-attaches the node span there too, so BOTH the produce phi
    span and the inline-review phi span nest under the same node span (one trace),
    and the lifecycle still reads pending -> in-progress -> verifiable -> done."""
    exporter = _install_inmemory_exporter()
    # Produce a bad output (gate rejects), then a good one on inline review.
    transport = _TracingFakeTransport(["bad draft", "OK fixed draft"])

    def verifier(node, result):
        out = (result.output or "").lstrip()
        return (True, None) if out.startswith("OK") else (False, "must start with OK")

    node = PlanNode(id="n1", task="produce something")
    dag = PlanDAG(nodes=[node])
    runtime = AgentRuntime(transport=transport, verifier=verifier, max_inline_fixes=1)
    result = asyncio.run(runtime.run(dag))

    assert result.states["n1"]["status"] == NodeStatus.DONE.value
    assert result.states["n1"]["inline_fixed"] is True

    spans = exporter.get_finished_spans()
    by_id = _by_id(spans)
    run_span = next(s for s in spans if s.name == "agent.run")
    node_span = next(s for s in spans if s.name == "agent.node")
    phi_spans = [s for s in spans if s.name == "phi.call"]

    # Two phi calls (produce + one inline review) — BOTH under the node span.
    assert len(phi_spans) == 2, [s.name for s in spans]
    for ph in phi_spans:
        assert ph.parent is not None
        assert by_id[ph.parent.span_id] is node_span
        assert ph.context.trace_id == run_span.context.trace_id

    ev = [e.name for e in node_span.events]
    assert ev[:2] == ["pending", "in-progress"]
    assert "verifiable" in ev and ev[-1] == "done"
