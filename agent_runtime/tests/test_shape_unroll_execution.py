"""The a3 re-architecture (d48 update): a cyclic shape UNROLLS generically and the
SAME AgentRuntime executes it — no per-shape executor.

Proves end-to-end, fully OFFLINE (a recording FakeTransport, no GPU):

1. UNROLL is generic + declarative — :func:`unroll_shape` turns the deep-research
   template (round/final POSITIONS + max_iter) into a bounded ACYCLIC DAG: rounds
   1..n-1 = {research, critic}, final round = {research, synthesis, verify}; every
   position maps onto a worker/synthesizer NODE (d48), the SAME single spec bound to
   EVERY node; growing-visibility edges; honors the UI-set max_iter.
2. The GENERIC AgentRuntime EXECUTES that DAG — the per-POSITION behavior framing is
   injected into each node's TASK (prompting, not a role code-switch), a research
   position carries the web_search tool, the synthesizer writes the deliverable, and
   a later round's node SEES the earlier rounds' outputs (growing visibility).
"""
from __future__ import annotations

import asyncio

from llm_framework import ChatResult

from agent_runtime.roles import READ_NOT_DESCRIBE
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
    # d48: positions map onto the TWO node roles — synthesis → synthesizer, every
    # other position → worker (behavior comes from the position framing in the task).
    assert [n.role for n in dag.nodes] == [
        "worker", "worker", "worker", "worker", "worker", "synthesizer", "worker",
    ]
    # a research-position node carries the web_search tool (reads sources via the
    # generic search-then-read tool path — the retired role-research gate's job).
    assert dag.by_id["r1_research"].tool == "web_search"
    assert dag.by_id["r1_critic"].tool is None
    # the SAME single spec is bound to EVERY node
    assert all(n.spec == SPEC and n.effective_specs == (SPEC,) for n in dag.nodes)
    # the position framing is injected into the TASK (prompting drives behavior)
    assert "RESEARCH this layer" in dag.by_id["r1_research"].task
    assert "CRITIQUE" in dag.by_id["r1_critic"].task
    # GROWING VISIBILITY: each node depends on EVERY previously-authored node.
    assert dag.nodes[0].depends_on == ()
    assert dag.nodes[2].depends_on == ("r1_research", "r1_critic")
    assert dag.by_id["r3_verify"].depends_on == (
        "r1_research", "r1_critic", "r2_research", "r2_critic", "r3_research", "r3_synthesis",
    )
    assert len(dag.topo_order()) == len(dag.nodes)


def test_unroll_honors_max_iter_override_and_ships_on_disk_shape():
    on_disk = load_shape("deep-research")
    # the override drives the round count, clamped to the shape's hard_cap. Count the
    # research-position nodes (one per round) by their id suffix.
    assert sum(1 for n in unroll_shape(on_disk, "g", max_iter_override=2).nodes
               if n.id.endswith("_research")) == 2
    assert sum(1 for n in unroll_shape(on_disk, "g", max_iter_override=999).nodes
               if n.id.endswith("_research")) == on_disk.hard_cap


# --------------------------------------------------------------------------- #
# 2) the GENERIC runtime executes the unrolled DAG (worker/synthesizer + framing)
# --------------------------------------------------------------------------- #
class _RecordingTransport:
    """Records (system, user) per call and answers RAW text (d48/d50.1 — no schema)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content

    def chat(self, messages, **opts) -> ChatResult:
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        self.calls.append({"system": system, "user": user, "fmt": opts.get("format")})
        # Every node now emits RAW free-text (workers + the synthesizer's raw loop).
        return ChatResult(role="assistant", content="a concrete integrated answer")


def _call_for(calls, position: str, round_no: int) -> dict:
    needle = f"[{position} · round {round_no}]"
    return next(c for c in calls if needle in c["user"])


def test_generic_runtime_executes_unrolled_dag_with_positions_and_growing_visibility():
    dag = unroll_shape(_shape(3), "study the topic", spec=SPEC)
    transport = _RecordingTransport()
    rt = AgentRuntime(
        transport=transport,
        execution=ExecutionMode.CONCURRENT,
        subagent_call_opts={"think": False, "temperature": 0},
    )
    out = asyncio.run(rt.run(dag))

    # every unrolled node ran and completed (the generic runtime drove the cyclic shape)
    assert out.ok
    assert set(out.results) == {n.id for n in dag.nodes}

    # d48: NO per-call format=schema anywhere (content is RAW on every node).
    assert all(c["fmt"] is None for c in transport.calls)

    # POSITION framing reached the node TASK (prompting, not a role code-switch).
    research_call = _call_for(transport.calls, "research", 1)
    assert "RESEARCH this layer" in research_call["user"]
    assert READ_NOT_DESCRIBE in research_call["user"]  # d13: read, don't describe
    assert "ROLE: WORKER" in research_call["system"]

    critic_call = _call_for(transport.calls, "critic", 1)
    assert "CRITIQUE" in critic_call["user"]

    # the synthesis node ran as the SYNTHESIZER (its result carries that role).
    assert out.results["r3_synthesis"].role == "synthesizer"

    # GROWING VISIBILITY: the final verify node's user turn carries EVERY prior node's
    # output (threaded in as its inputs by the runtime).
    verify_call = _call_for(transport.calls, "verify", 3)
    for prior in ("r1_research", "r1_critic", "r2_research", "r3_research", "r3_synthesis"):
        assert prior in verify_call["user"], f"{prior} not visible to final verify node"
