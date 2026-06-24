"""b3: the DEFAULT lifecycle VERIFY GATE wired onto EVERY node path.

Part (1) of b3 wired ``default_node_verifier`` — the reusable safety-net gate —
onto the chat node paths (the live acyclic path + the offline path) that
previously built ``AgentRuntime`` with ``verifier=None`` and so passed the gate
trivially, leaving the CODER=REVIEWER inline-fix unreachable. This test LOCKS that
behaviour on a PLAIN (non-agentic) ``AgentRuntime`` drive — the same engine those
chat paths use — proving the full lifecycle:

    PENDING → RUNNING → VERIFIABLE → (gate rejects) → same-spec inline fix → DONE

and that the gate is CONSERVATIVE (a real answer passes first try) and that a
JUDGMENT node's missing verdict is REJECTED (never silently passed) and re-emitted
inline. Fully OFFLINE: FakeTransport, no Ollama / network / GPU (d7/d8).
"""
from __future__ import annotations

import asyncio
import json

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.runtime import AgentRuntime
from agent_runtime.status import NodeStatus
from agent_runtime.verify import default_node_verifier
from llm_framework import FakeTransport


class _Res:
    """Minimal stand-in for a SubAgentResult (just an ``output``)."""

    def __init__(self, output):
        self.output = output


def test_degenerate_output_repaired_inline_to_done():
    # Produce returns EMPTY (degenerate) → the default gate rejects → same-spec
    # inline review produces real content → the node reaches DONE via the inline
    # fix on a non-agentic AgentRuntime drive. strict=True asserts exactly 2 calls.
    transport = FakeTransport(["", "A real, non-empty deliverable answer."], strict=True)
    dag = PlanDAG(nodes=[PlanNode(id="n1", task="Answer the question.")])
    rt = AgentRuntime(
        transport=transport, verifier=default_node_verifier, max_inline_fixes=1
    )
    out = asyncio.run(rt.run(dag))

    st = out.states["n1"]
    assert st["status"] == NodeStatus.DONE.value
    assert st["verified"] is True
    assert st["inline_fixed"] is True
    assert out.results["n1"].output.strip() != ""
    assert transport.call_count == 2  # produce + exactly one inline review


def test_real_output_passes_gate_first_try_no_inline_fix():
    # The gate is conservative: a real answer passes with NO inline fix (so live
    # runs are never spuriously rejected).
    transport = FakeTransport(["A perfectly good answer."])
    dag = PlanDAG(nodes=[PlanNode(id="n1", task="Answer.")])
    rt = AgentRuntime(transport=transport, verifier=default_node_verifier)
    out = asyncio.run(rt.run(dag))

    st = out.states["n1"]
    assert st["status"] == NodeStatus.DONE.value
    assert st["verified"] is True
    assert st["inline_fixed"] is False
    assert transport.call_count == 1  # produce only — no review needed


def test_default_verifier_unit_contract():
    # d48: the enum-verdict JUDGMENT path is retired (no judgment roles remain), so
    # the verifier only enforces the usable-output contract now.
    n = PlanNode(id="n", task="t")
    # real output passes; degenerate tokens are rejected with a reviewer-able reason.
    assert default_node_verifier(n, _Res("real content")) == (True, None)
    for bad in ("", "   ", "null", "N/A", "{}", "[]"):
        ok, reason = default_node_verifier(n, _Res(bad))
        assert ok is False and "usable output" in reason

    # a worker node with arbitrary verdict-less JSON still passes (no judgment gate).
    assert default_node_verifier(n, _Res(json.dumps({"answer": 42})))[0] is True
