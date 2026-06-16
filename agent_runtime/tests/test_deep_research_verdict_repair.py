"""JUDGMENT-NODE verdict hardening — now a GENERIC runtime capability (a3).

The b3 caveat: a verbose small model spends its budget on the findings prose and
OVERRUNS ``num_predict``, truncating the verdict JSON so a judgment node's enum
verdict comes back NULL. The lifecycle must never silently pass a null verdict.

Previously this lived in the per-shape ``DeepResearchExecutor``. That executor is
DELETED (a3 re-architecture): the deep-research shape is now UNROLLED by the
generic :func:`~agent_runtime.unroll_shape` into a role-tagged DAG that the SAME
:class:`~agent_runtime.AgentRuntime` executes — so the verdict hardening moved into
the runtime's generic role execution (``SubAgent._run_role``). This test proves the
GENERIC runtime, driving an unrolled deep-research DAG:

* RAISES ``num_predict`` for judgment roles (a higher floor than the research
  default) so the verdict JSON is not truncated in the first place; and
* RETRIES/REPAIRS a null/invalid verdict — re-issuing the call with a larger
  budget until a LEGAL enum verdict arrives — recording the repair count on the
  node result (never fabricating a passing verdict).

Fully OFFLINE: a scripted FakeTransport, no Ollama / network / GPU.
"""
from __future__ import annotations

import asyncio
import json

from agent_runtime.factory import PlanDAG
from agent_runtime.roles import (
    ROLE_DEFAULT_NUM_PREDICT,
    JUDGMENT_NUM_PREDICT,
    ROLE_VERDICTS,
)
from agent_runtime.runtime import AgentRuntime
from agent_runtime.shapes import ShapeSpec, unroll_shape
from llm_framework import FakeTransport


def _one_round_shape() -> ShapeSpec:
    # max_iter=1 → only the FINAL round runs: research → synthesis → verify.
    return ShapeSpec(
        name="dr-test",
        max_iter=1,
        hard_cap=1,
        round_roles=("research", "critic"),
        final_roles=("research", "synthesis", "verify"),
        execution="deep-research",
    )


def _unrolled_one_round() -> PlanDAG:
    dag = unroll_shape(_one_round_shape(), "study the topic in depth")
    # The one-round unroll is exactly the final round's three role nodes.
    assert [n.role for n in dag.nodes] == ["research", "synthesis", "verify"]
    return dag


def _runtime(transport, **kw) -> AgentRuntime:
    # A lean runtime exactly like the deep-research path: no acyclic verify gate /
    # validator (the verify ROLE node is the verification), so the only structured
    # calls are the role nodes themselves.
    return AgentRuntime(transport=transport, **kw)


def test_null_synthesis_verdict_is_repaired_and_budget_is_raised():
    state = {"verdict_calls": 0, "judgment_num_predicts": [], "research_num_predicts": []}

    def reply(messages, **opts):
        fmt = opts.get("format") or {}
        props = fmt.get("properties", {})
        # research schema = findings/sources/open_questions (no verdict key).
        if "findings" in props and "verdict" not in props:
            state["research_num_predicts"].append(opts.get("num_predict"))
            return json.dumps(
                {"findings": ["f1"], "sources": ["s1"], "open_questions": ["q1"]}
            )
        # a verdict (review) schema → synthesis, then verify.
        state["verdict_calls"] += 1
        state["judgment_num_predicts"].append(opts.get("num_predict"))
        if state["verdict_calls"] == 1:
            # synthesis: OMIT the verdict (the truncated-JSON failure mode) → repair.
            return json.dumps({"findings": ["partial"], "fixed_inline": []})
        # the synthesis repair + the verify call both return a legal verdict.
        return json.dumps({"verdict": "pass", "findings": ["ok"], "fixed_inline": []})

    transport = FakeTransport([reply])  # callable reused for every call
    rt = _runtime(transport)
    out = asyncio.run(rt.run(_unrolled_one_round()))

    syn = out.results["r1_synthesis"]
    ver = out.results["r1_verify"]

    # The null synthesis verdict was REPAIRED to a legal enum value (never null).
    assert syn.verdict == "pass"
    assert syn.verdict in ROLE_VERDICTS["synthesis"]
    assert syn.verdict_repairs == 1
    # verify passed first try (no repair).
    assert ver.verdict == "pass" and ver.verdict_repairs == 0

    # Judgment calls carried the RAISED budget floor; the repair raised it further.
    assert all(np >= JUDGMENT_NUM_PREDICT for np in state["judgment_num_predicts"])
    assert max(state["judgment_num_predicts"]) > JUDGMENT_NUM_PREDICT  # the repair bump
    # research is NOT bumped — it keeps the default budget (the floor is judgment-only).
    assert all(np == ROLE_DEFAULT_NUM_PREDICT for np in state["research_num_predicts"])


def test_unrepairable_verdict_is_surfaced_not_silently_passed():
    # A judgment node that NEVER returns a legal verdict exhausts the bounded
    # repairs and surfaces verdict=None WITH the repair count — visible, not
    # silently passed as a valid verdict.
    def reply(messages, **opts):
        fmt = opts.get("format") or {}
        props = fmt.get("properties", {})
        if "findings" in props and "verdict" not in props:
            return json.dumps({"findings": ["f"], "sources": ["s"], "open_questions": ["q"]})
        return json.dumps({"findings": ["no verdict ever"], "fixed_inline": []})

    transport = FakeTransport([reply])
    rt = _runtime(transport, max_verdict_repairs=2)
    out = asyncio.run(rt.run(_unrolled_one_round()))

    syn = out.results["r1_synthesis"]
    assert syn.verdict is None              # never fabricated a passing verdict
    assert syn.verdict_repairs == 2         # the bounded repair budget was spent
