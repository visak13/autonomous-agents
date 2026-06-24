"""s8/b1 — the planner builds the DAG by ISSUING TOOL CALLS (fully OFFLINE).

This is the Phase-2 #1 deliverable proof. The incremental authorer no longer makes
a per-node ``format``-schema-CONSTRAINED call (the d34 edge-drop: constrained
decoding silently drops the reasoned ``depends_on``). Instead the planner BUILDS the
plan by calling the in-app plan-building tools — ``seed_plan`` → ``add_step`` (one
per node) → optional ``set_node_spec`` → ``finalize_plan`` — exactly the eda-base3
``create_plan``/``add_action`` pattern. Each tool call is prompt-elicited JSON the
loop parses + validates (NO ``format`` schema, ``think=True``), so a reasoned edge
can never be schema-dropped.

These tests script a :class:`FakeTransport` with the planner's TOOL CALLS, so the
whole tool-driven loop runs in-process with zero inference. They prove:

1. the planner ISSUES TOOL CALLS (seed_plan / add_step / finalize_plan), recorded on
   the builder's call trail — not a single one-shot JSON DAG;
2. a 'linear PLUS modular parallel' request yields a COMPOSITIONAL DAG (independent
   parallel sources AND a multi-level chain), not a flat sequential line;
3. NO ``format`` schema is sent on the authoring calls (the structural guarantee that
   the reasoned ``depends_on`` is never constrained-decoding-dropped).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence

from agent_runtime.factory import AbstractPlanFactory
from agent_runtime.incremental import IncrementalPlanner
from agent_runtime.plan_tools import PlanBuilder
from llm_framework import FakeTransport
from specialization.registry import SpecRegistry
from specialization.seed import DEEP_RESEARCH_SPEC, seed_canonical_rulesets


def _run(coro):
    return asyncio.run(coro)


_TOOL_CATALOG = [
    {"name": "web_search", "description": "search the web for candidate pages"},
    {"name": "web_fetch", "description": "fetch and extract a page's article text"},
    {"name": "file_write", "description": "write content to a file"},
]


def _seed(shape: str = "linear-plus-modular-parallel") -> str:
    return json.dumps({"tool": "seed_plan", "args": {"shape": shape}})


def _add(task: str, *, tool: str = "", spec: str = "", depends_on: Sequence[str] = ()) -> str:
    return json.dumps(
        {
            "tool": "add_step",
            "args": {
                "task": task,
                "tool": tool,
                "spec": spec,
                "specs": [],
                "depends_on": list(depends_on),
            },
        }
    )


def _finalize() -> str:
    return json.dumps({"tool": "finalize_plan", "args": {}})


def _planner(replies: Sequence[str], tmp_path, **kw) -> IncrementalPlanner:
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)
    factory = AbstractPlanFactory(reg.index(), tool_catalog=_TOOL_CATALOG)
    return IncrementalPlanner(
        FakeTransport(list(replies)),
        factory,
        spec_names=reg.names(),
        tool_names=[t["name"] for t in _TOOL_CATALOG],
        shape_name="linear-plus-modular-parallel",
        shape_description="parallel gather steps, then a sequential combine→deliver chain",
        default_research_spec=DEEP_RESEARCH_SPEC,
        **kw,
    )


# A 'linear PLUS modular parallel' plan: 3 INDEPENDENT gathers (parallel, no edges)
# → a combine step that depends on all 3 → a final save step that depends on the
# combine (the LINEAR tail). Compositional: both patterns coexist.
def _compositional_replies() -> list[str]:
    return [
        _seed(),
        _add("Search the latest news on climate change", tool="web_search"),
        _add("Search the latest news on AI", tool="web_search"),
        _add("Search the latest news on space exploration", tool="web_search"),
        _add("Combine the three news summaries", depends_on=["n1", "n2", "n3"]),
        _add("Write the combined brief to a file", tool="file_write", depends_on=["n4"]),
        _finalize(),
    ]


def test_planner_issues_tool_calls_not_one_shot(tmp_path):
    planner = _planner(_compositional_replies(), tmp_path)
    result = _run(planner.plan("news on climate change, AI and space; combine then save"))

    builder = planner.last_builder
    assert builder is not None
    issued = [c["tool"] for c in builder.calls]
    # The DAG was BUILT by discrete tool calls, in the eda-base3 seed→add→finalize order.
    assert issued[0] == "seed_plan"
    assert issued.count("add_step") == 5
    assert issued[-1] == "finalize_plan"
    assert builder.finalized is True
    # PlanResult.raw carries the tool-call audit trail (not a one-shot JSON blob).
    assert json.loads(result.raw)[0]["tool"] == "seed_plan"


def test_linear_plus_modular_parallel_yields_compositional_dag(tmp_path):
    planner = _planner(_compositional_replies(), tmp_path)
    dag = _run(planner.plan("news on climate change, AI and space; combine then save")).dag
    by = dag.by_id

    # PARALLEL part: 3 independent source gathers, no edges between them.
    sources = [n for n in dag.nodes if not n.depends_on]
    assert len(sources) >= 3, "expected 3 independent parallel gather sources"

    # LINEAR part: a combine that fans IN all three, then a save that chains after it
    # — a real 2-level dependency chain, not a flat single line.
    assert set(by["n4"].depends_on) == {"n1", "n2", "n3"}   # combine fans in the parallel sources
    assert by["n5"].depends_on == ("n4",)                    # save chains after combine (linear tail)

    # COMPOSITIONAL, not flat: the DAG has BOTH a parallel fan-in AND a chained tail.
    has_parallel = any(len(n.depends_on) >= 2 for n in dag.nodes)
    has_chain = any(len(n.depends_on) == 1 for n in dag.nodes)
    assert has_parallel and has_chain, "DAG collapsed to a flat shape (the s6 bug)"
    # A genuinely flat sequential plan would have every node depend on exactly its
    # predecessor; assert this one does NOT (n1..n3 are all roots).
    assert len(sources) > 1


def test_no_format_schema_on_authoring_calls(tmp_path):
    # The structural d34 guarantee: the authoring turns carry NO ``format`` schema, so
    # the reasoned ``depends_on`` edge is never constrained-decoding-dropped. (The old
    # path passed ``format=self._node_schema()`` on every per-node call.)
    transport = FakeTransport(_compositional_replies())
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)
    factory = AbstractPlanFactory(reg.index(), tool_catalog=_TOOL_CATALOG)
    planner = IncrementalPlanner(
        transport,
        factory,
        spec_names=reg.names(),
        tool_names=[t["name"] for t in _TOOL_CATALOG],
        shape_name="linear-plus-modular-parallel",
    )
    _run(planner.plan("compose a plan"))
    assert transport.calls, "no transport calls recorded"
    for call in transport.calls:
        assert "format" not in call["opts"], (
            "an authoring call sent a 'format' schema — that re-introduces the d34 "
            "constrained-decoding edge-drop"
        )
        # the reasoning path is preserved (think=True), per the specialist ruleset.
        assert call["opts"].get("think") is True


def test_builder_drops_unknown_vocab_but_keeps_depends_on(tmp_path):
    # The PlanBuilder validates spec/tool against the registered vocab (an unknown
    # value is dropped, never crashes) — the same discipline the old enum schema gave,
    # as VALIDATION not constrained decoding, so depends_on (reasoned) always survives.
    b = PlanBuilder(
        spec_names=["research-analyst"],
        tool_names=["web_search"],
        shape_name="linear",
        max_nodes=12,
    )
    b.dispatch("seed_plan", {})
    b.dispatch("add_step", {"task": "gather", "tool": "web_search", "spec": "no-such-spec"})
    obs = b.dispatch(
        "add_step",
        {"task": "write", "tool": "made-up-tool", "spec": "research-analyst",
         "depends_on": ["n1"]},
    )
    assert obs["ok"] and obs["id"] == "n2"
    n1, n2 = b.nodes
    assert n1["tool"] == "web_search" and n1["spec"] is None      # unknown spec dropped
    assert n2["tool"] is None and n2["spec"] == "research-analyst"  # unknown tool dropped
    assert n2["depends_on"] == ["n1"]                              # reasoned edge KEPT


def test_legacy_role_is_coerced_to_valid_node_role(tmp_path):
    # d48 (s9/c5): the node-role vocab is bounded to {worker, synthesizer}. A small
    # model still occasionally emits a RETIRED legacy role (research/critic/.../verify
    # /synthesis) — the authoring boundary MUST coerce it (research/critic/reviewer/
    # verify -> worker; synthesis -> synthesizer; unknown -> worker) so a role-slip can
    # never crash the DAG factory (which rejects an unknown role).
    b = PlanBuilder(spec_names=[], tool_names=["web_search"], shape_name="linear",
                    max_nodes=12)
    b.dispatch("seed_plan", {})
    b.dispatch("add_step", {"task": "gather", "tool": "web_search", "role": "research"})
    b.dispatch("add_step", {"task": "critique", "role": "critic", "depends_on": ["n1"]})
    b.dispatch("add_step", {"task": "deliver", "role": "synthesis", "depends_on": ["n2"]})
    b.dispatch("add_step", {"task": "weird", "role": "made-up", "depends_on": ["n3"]})
    roles = [n["role"] for n in b.nodes]
    assert roles == ["worker", "worker", "synthesizer", "worker"]
    # the coerced records build VALID PlanNodes — no PlanError on the retired vocab
    # (PlanNode.__post_init__ rejects any role outside {worker, synthesizer}).
    from agent_runtime.factory import PlanNode
    built = [PlanNode(id=n["id"], task=n["task"], role=n["role"]) for n in b.nodes]
    assert {n.role for n in built} <= {"worker", "synthesizer"}


def test_unknown_tool_call_is_a_soft_observation_not_a_crash(tmp_path):
    b = PlanBuilder(spec_names=[], tool_names=[], max_nodes=12)
    obs = b.dispatch("frobnicate", {"x": 1})
    assert obs["ok"] is False
    assert "unknown tool" in obs["note"]


def test_alias_id_resolves_in_depends_on(tmp_path):
    # The model may coin its own id and reference it later; the builder maps it to the
    # canonical id so the reasoned edge still resolves.
    b = PlanBuilder(spec_names=[], tool_names=[], max_nodes=12)
    b.dispatch("add_step", {"task": "first", "id": "gather"})
    b.dispatch("add_step", {"task": "second", "depends_on": ["gather"]})
    assert b.nodes[1]["depends_on"] == ["n1"]  # alias 'gather' → canonical 'n1'
