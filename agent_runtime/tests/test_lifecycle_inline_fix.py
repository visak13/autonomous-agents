"""Verify-gate → CODER=REVIEWER inline-fix LIFECYCLE regression test (a2 review).

a1 reworked the produce-step system turn (the SHAPING ruleset injection) and the
inline-review system turn (``SubAgent.review_and_fix``). This test LOCKS the
run-engine lifecycle that change rides through and proves it did not regress:

    PENDING → RUNNING → VERIFIABLE → (gate REJECTS) → same-spec inline review → DONE

and, critically, that a1's ``review_and_fix`` carries the SHAPING ruleset
(framing + body) into the inline-review system turn — not merely the reviewer
framing — so the inline fix still shapes the FORM by the ruleset (d1 + d10).

a1 committed no test for this path (``agent_runtime/tests`` held only
``test_inject.py``), so a future regression of the verify-gate / coder=reviewer
path, or of the shaping ruleset being dropped from the inline review, would have
gone uncaught. This is the inline-fix the a2 review adds under coder=reviewer
doctrine. Fully OFFLINE: FakeTransport, no Ollama / network / GPU (d7/d8).
"""
from __future__ import annotations

import asyncio

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.runtime import _REVIEWER_FRAMING, _SHAPING_FRAMING, AgentRuntime
from agent_runtime.status import NodeStatus
from llm_framework import FakeTransport
from specialization.loader import SpecLoader
from specialization.registry import SpecRegistry
from specialization.seed import MARKDOWN_WRITER_RULESET, seed_ruleset_spec


def test_verify_gate_inline_fix_carries_shaping_and_reaches_done(tmp_path):
    reg = SpecRegistry(tmp_path / "specs")
    seed_ruleset_spec(
        reg, "markdown-writer",
        "Shape findings into clean GFM (headings, lists, summary, sources).",
        MARKDOWN_WRITER_RULESET,
    )
    loader = SpecLoader(reg)

    # Produce a draft WITHOUT a level-1 heading (the gate rejects it), then a
    # corrected draft WITH one (the gate accepts) — the SAME-spec inline review
    # repairs it in place, no DAG re-loop. strict=True asserts an exact call count.
    transport = FakeTransport(
        [
            "Summary: talks stalled — but no level-1 heading here.",  # produce → rejected
            "# Iran nuclear-talks update\n\n**Summary** talks stalled.",  # inline review → accepted
        ],
        strict=True,
    )

    # Per-node verify gate: enforce a shaping rule (must open with a `# ` heading).
    def verifier(node, result):
        out = (result.output or "").lstrip()
        if out.startswith("# "):
            return True
        return (False, "output must open with a level-1 '# ' heading")

    node = PlanNode(
        id="n1",
        task="Research the latest Iran nuclear-talks update and report the findings.",
        spec="markdown-writer",
    )
    dag = PlanDAG(nodes=[node])

    runtime = AgentRuntime(
        transport=transport, loader=loader, verifier=verifier, max_inline_fixes=1
    )
    result = asyncio.run(runtime.run(dag))

    # --- LIFECYCLE: reached DONE via the inline fix (not re-plan, not FAILED) --- #
    st = result.states["n1"]
    assert st["status"] == NodeStatus.DONE.value
    assert st["inline_fixed"] is True
    assert st["verified"] is True
    assert "n1" in result.results
    assert result.results["n1"].output.lstrip().startswith("# ")  # corrected output cached

    # Exactly two phi calls: the produce step + one inline review.
    assert transport.call_count == 2
    produce_sys = next(
        m["content"] for m in transport.calls[0]["messages"] if m["role"] == "system"
    )
    review_sys = next(
        m["content"] for m in transport.calls[1]["messages"] if m["role"] == "system"
    )

    # --- PRODUCE carries the shaping layer (framing + ruleset), NOT the reviewer --- #
    assert _SHAPING_FRAMING in produce_sys
    assert MARKDOWN_WRITER_RULESET.strip() in produce_sys
    assert _REVIEWER_FRAMING not in produce_sys

    # --- INLINE REVIEW still carries the SAME shaping ruleset (a1's review_and_fix)
    #     AND the reviewer framing — the fix shapes by the ruleset too (d1 + d10) --- #
    assert _SHAPING_FRAMING in review_sys
    assert MARKDOWN_WRITER_RULESET.strip() in review_sys
    assert _REVIEWER_FRAMING in review_sys
