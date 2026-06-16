"""The a3 re-architecture: a cyclic shape UNROLLS generically and the SAME
AgentRuntime executes it — no per-shape executor (DeepResearchExecutor is deleted).

Proves end-to-end, fully OFFLINE (a recording FakeTransport, no GPU):

1. UNROLL is generic + declarative — :func:`unroll_shape` turns the deep-research
   template (round_roles/final_roles/max_iter) into a bounded ACYCLIC role-tagged
   DAG: rounds 1..n-1 = {research, critic}, final round = {research, synthesis,
   verify}; the SAME single spec bound to EVERY node; growing-visibility edges
   (each node depends on every prior node); honors the UI-set max_iter.
2. The GENERIC AgentRuntime EXECUTES that role-tagged DAG — role framing + per-role
   output schema are applied by the runtime's own SubAgent (not a per-shape engine),
   the d13 read-not-describe / critic-rejects-meta principles ride the role prompts,
   judgment roles carry a validated enum verdict, and a later round's node SEES the
   earlier rounds' outputs (growing visibility via the runtime's inputs threading).
"""
from __future__ import annotations

import asyncio
import json

from llm_framework import ChatResult

from agent_runtime.roles import ROLE_VERDICTS
from agent_runtime.runtime import AgentRuntime
from agent_runtime.scheduler import ExecutionMode
from agent_runtime.shapes import ShapeSpec, load_shape, unroll_shape

SPEC = "research-analyst"


def _shape(max_iter: int = 3) -> ShapeSpec:
    return ShapeSpec(
        name="deep-research",
        max_iter=max_iter,
        hard_cap=24,
        round_roles=("research", "critic"),
        final_roles=("research", "synthesis", "verify"),
        execution="deep-research",
    )


# --------------------------------------------------------------------------- #
# 1) the unroll is generic, declarative, bounded, acyclic, single-spec
# --------------------------------------------------------------------------- #
def test_unroll_is_bounded_acyclic_role_tagged_single_spec():
    dag = unroll_shape(_shape(3), "study the topic", spec=SPEC)
    # 2 non-final rounds × {research, critic} + 1 final × {research, synthesis, verify}
    assert [n.id for n in dag.nodes] == [
        "r1_research", "r1_critic",
        "r2_research", "r2_critic",
        "r3_research", "r3_synthesis", "r3_verify",
    ]
    assert [n.role for n in dag.nodes] == [
        "research", "critic", "research", "critic", "research", "synthesis", "verify",
    ]
    # the SAME single spec is bound to EVERY node (only the role differs — §2c)
    assert all(n.spec == SPEC and n.effective_specs == (SPEC,) for n in dag.nodes)
    # GROWING VISIBILITY: each node depends on EVERY previously-authored node, so the
    # rounds run in order and each node's inputs carry all earlier layers.
    assert dag.nodes[0].depends_on == ()
    assert dag.nodes[2].depends_on == ("r1_research", "r1_critic")
    assert dag.by_id["r3_verify"].depends_on == (
        "r1_research", "r1_critic", "r2_research", "r2_critic", "r3_research", "r3_synthesis",
    )
    # constructing the PlanDAG already validated acyclicity (no raise) — prove a topo
    # order exists over all nodes.
    assert len(dag.topo_order()) == len(dag.nodes)


def test_unroll_honors_max_iter_override_and_ships_on_disk_shape():
    on_disk = load_shape("deep-research")
    # the override drives the round count, clamped to the shape's hard_cap
    assert sum(1 for n in unroll_shape(on_disk, "g", max_iter_override=2).nodes
               if n.role == "research") == 2
    assert sum(1 for n in unroll_shape(on_disk, "g", max_iter_override=999).nodes
               if n.role == "research") == on_disk.hard_cap


# --------------------------------------------------------------------------- #
# 2) the GENERIC runtime executes the unrolled role-tagged DAG
# --------------------------------------------------------------------------- #
class _RecordingRoleTransport:
    """Records (system, user, schema) per call and answers each per-role schema."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content

    def chat(self, messages, **opts) -> ChatResult:
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        fmt = opts.get("format") or {}
        props = fmt.get("properties", {})
        self.calls.append({"system": system, "user": user, "props": set(props)})
        if "findings" in props and "verdict" not in props:  # research schema
            content = {"findings": ["a concrete fetched fact"], "sources": ["http://x/a"],
                       "open_questions": ["q"]}
        elif "verdict" in props:  # a judgment schema (critic / synthesis / verify)
            enum = props["verdict"].get("enum", [])
            content = {"verdict": ("converged" if "converged" in enum else "pass"),
                       "gaps": [], "weak_claims": [], "follow_up_queries": ["fq"],
                       "findings": ["f"], "fixed_inline": []}
        else:
            content = {"output": "x"}
        return ChatResult(role="assistant", content=json.dumps(content))


def _call_for(calls, role: str, round_no: int) -> dict:
    needle = f"[{role} · round {round_no}]"
    return next(c for c in calls if needle in c["user"])


def test_generic_runtime_executes_unrolled_dag_with_roles_and_growing_visibility():
    dag = unroll_shape(_shape(3), "study the topic", spec=SPEC)
    transport = _RecordingRoleTransport()
    # No loader → the single spec stays a NAME (role framing only), which is all this
    # test needs; the deep-research route passes a real loader to resolve the body.
    rt = AgentRuntime(
        transport=transport,
        execution=ExecutionMode.CONCURRENT,
        subagent_call_opts={"think": False, "temperature": 0},
    )
    out = asyncio.run(rt.run(dag))

    # every unrolled node ran and completed (the generic runtime drove the cyclic shape)
    assert out.ok
    assert set(out.results) == {n.id for n in dag.nodes}

    # ROLE FRAMING applied by the GENERIC runtime + d13 baked into the role prompts.
    research_call = _call_for(transport.calls, "research", 1)
    assert "ROLE: RESEARCH" in research_call["system"]
    assert "READ the actual source" in research_call["system"]  # d13: read, don't describe
    assert research_call["props"] == {"findings", "sources", "open_questions"}

    critic_call = _call_for(transport.calls, "critic", 1)
    assert "ROLE: CRITIC" in critic_call["system"]
    assert "NEEDS_MORE" in critic_call["system"]  # d13: critic rejects meta-summaries
    assert "verdict" in critic_call["props"]

    # JUDGMENT roles carry a validated enum verdict on the result.
    assert out.results["r1_critic"].verdict in ROLE_VERDICTS["critic"]
    assert out.results["r3_verify"].verdict in ROLE_VERDICTS["verify"]

    # GROWING VISIBILITY: the final verify node's user turn carries EVERY prior node's
    # output (threaded in as its inputs by the runtime), so each round provably builds
    # on all earlier layers.
    verify_call = _call_for(transport.calls, "verify", 3)
    for prior in ("r1_research", "r1_critic", "r2_research", "r3_research", "r3_synthesis"):
        assert prior in verify_call["user"], f"{prior} not visible to final verify node"
