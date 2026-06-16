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


def test_judgment_node_missing_verdict_rejected_then_fixed_inline():
    # A judgment-role node whose produce OMITS the verdict is rejected by the gate
    # (never silently passed) and re-emits a legal verdict inline → DONE.
    #
    # NOTE (a3): a role node now ALSO repairs a null verdict at PRODUCE time inside
    # SubAgent._run_role (the generic move of the b3 hardening). That is a SECOND,
    # earlier safety net; this test isolates the GATE's inline-fix path by disabling
    # the produce-time repair (max_verdict_repairs=0), so the missing verdict reaches
    # the gate exactly as before and the CODER=REVIEWER inline fix handles it.
    produce = json.dumps({"findings": ["x"], "fixed_inline": []})            # no verdict
    review = json.dumps({"verdict": "pass", "findings": ["x"], "fixed_inline": []})
    transport = FakeTransport([produce, review], strict=True)
    dag = PlanDAG(nodes=[PlanNode(id="j1", task="Judge the answer.", role="verify")])
    rt = AgentRuntime(
        transport=transport, verifier=default_node_verifier, max_inline_fixes=1,
        max_verdict_repairs=0,
    )
    out = asyncio.run(rt.run(dag))

    st = out.states["j1"]
    assert st["status"] == NodeStatus.DONE.value
    assert st["inline_fixed"] is True
    assert transport.call_count == 2


def test_default_verifier_unit_contract():
    n = PlanNode(id="n", task="t")
    # real output passes; degenerate tokens are rejected with a reviewer-able reason.
    assert default_node_verifier(n, _Res("real content")) == (True, None)
    for bad in ("", "   ", "null", "N/A", "{}", "[]"):
        ok, reason = default_node_verifier(n, _Res(bad))
        assert ok is False and "usable output" in reason

    # judgment node: missing / out-of-enum verdict rejected; a legal verdict passes.
    j = PlanNode(id="j", task="t", role="critic")
    assert default_node_verifier(j, _Res(json.dumps({"gaps": []})))[0] is False
    assert default_node_verifier(j, _Res(json.dumps({"verdict": "maybe"})))[0] is False
    assert default_node_verifier(j, _Res(json.dumps({"verdict": "converged"}))) == (
        True,
        None,
    )
    # a non-judgment node with verdict-less JSON still passes (only judgment gated).
    assert default_node_verifier(n, _Res(json.dumps({"answer": 42})))[0] is True
