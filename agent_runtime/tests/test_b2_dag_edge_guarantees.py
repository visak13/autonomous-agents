"""s8/b2 — DAG EDGE GUARANTEES enforced at finalize (fully OFFLINE).

Two model-agnostic structural guarantees the finalize step now upholds, so a bad
edge never surfaces as a user-visible failure and the writer is never starved of
the research:

* **d28 — terminal write/synthesize node MUST depend on the research/gather
  node(s).** E4B authors a FLAT zero-edge DAG EVERY run (a ``web_search`` research
  node and a ``file_write`` write node with NO edge between them), so the writer
  runs DISCONNECTED and the report comes back thin (o4 / d34). The finalize pass
  :meth:`IncrementalPlanner._enforce_terminal_research_edge` auto-adds the missing
  edge. It is a NO-OP (byte-identical) on a healthy DAG whose terminal sink already
  (transitively) depends on the research — proven below so a connected compositional
  plan is never mangled.

* **d7 — no dangling/phantom depends_on edge.** The finalize parse now goes through
  :meth:`AbstractPlanFactory.parse_dag_safe`, which REPAIRS an unresolvable / self
  ``depends_on`` ref instead of rejecting it, while still raising (→ self-heal
  retry-on-reject) for a genuine invalidity (duplicate id / real cycle).

These tests script a :class:`FakeTransport` with the planner's TOOL CALLS, so the
whole tool-driven finalize runs in-process with zero inference — and exercise the
enforcement helper directly for the edge cases.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence

from agent_runtime.factory import AbstractPlanFactory, PlanError
from agent_runtime.incremental import IncrementalPlanner
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


def _seed(shape: str = "research-then-write") -> str:
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
        shape_name="research-then-write",
        shape_description="gather, then write",
        default_research_spec=DEEP_RESEARCH_SPEC,
        **kw,
    )


# --------------------------------------------------------------------------- #
# d28 — the E4B flat zero-edge defect, repaired at finalize
# --------------------------------------------------------------------------- #
def _flat_zero_edge_replies() -> list[str]:
    """The exact E4B defect: a research node and a write node, NO edge between."""
    return [
        _seed(),
        _add("Research the recent US-Iran conflict", tool="web_search"),  # n1 (research)
        _add("Write a detailed HTML report", tool="file_write"),          # n2 (write) — NO depends_on
        _finalize(),
    ]


def test_d28_flat_zero_edge_dag_gets_write_to_research_edge(tmp_path):
    planner = _planner(_flat_zero_edge_replies(), tmp_path)
    result = _run(planner.plan("Write a detailed HTML report on the US-Iran conflict"))
    dag = result.dag
    by = dag.by_id

    # The model authored n1 (research) and n2 (write) with ZERO edges — exactly the
    # flat zero-edge DAG E4B emits every run. Finalize MUST have connected them.
    assert by["n2"].depends_on == ("n1",), (
        "the terminal write node was left disconnected from the research node — "
        "the writer would be starved → thin report (d28/o4 defect)"
    )
    # The whole DAG is now connected: exactly one source (the research) and the
    # write node fans in from it.
    sources = [n for n in dag.nodes if not n.depends_on]
    assert [n.id for n in sources] == ["n1"]
    # The repair is reported (auditable) on the PlanResult.
    assert result.repair["research_edges"], "edge repair not recorded"
    assert "n2" in result.repair["research_edges"][0]


def test_d28_is_a_noop_on_a_healthy_connected_dag(tmp_path):
    # A healthy compositional plan: 3 parallel research sources + a combine/write sink
    # that ALREADY depends on all three. The terminal sink transitively sees the
    # research, so the d28 pass must NOT touch the topology (byte-identical).
    replies = [
        _seed(),
        _add("Research climate change", tool="web_search"),
        _add("Research AI", tool="web_search"),
        _add("Research space", tool="web_search"),
        _add("Combine and write the brief", tool="file_write", depends_on=["n1", "n2", "n3"]),
        _finalize(),
    ]
    planner = _planner(replies, tmp_path)
    result = _run(planner.plan("news on climate, AI, space; combine then write"))
    by = result.dag.by_id

    # Topology UNCHANGED: the combine still fans in exactly its three authored sources
    # (no spurious extra edges), and the three sources stay independent roots.
    assert set(by["n4"].depends_on) == {"n1", "n2", "n3"}
    assert by["n1"].depends_on == () and by["n2"].depends_on == () and by["n3"].depends_on == ()
    # No research-edge repair fired on the already-connected plan.
    assert result.repair["research_edges"] == []


def test_d28_noop_when_no_research_node(tmp_path):
    # No gather/research node at all (two plain write steps) → nothing to connect a
    # writer to; the pass must be a no-op and never invent an edge.
    replies = [
        _seed(),
        _add("Draft section one", tool="file_write"),
        _add("Draft section two", tool="file_write"),
        _finalize(),
    ]
    planner = _planner(replies, tmp_path)
    result = _run(planner.plan("draft two independent sections"))
    assert result.repair["research_edges"] == []
    for n in result.dag.nodes:
        assert n.depends_on == ()


# --------------------------------------------------------------------------- #
# d28 — enforcement helper unit tests (the edge logic in isolation)
# --------------------------------------------------------------------------- #
def _bare_planner(tmp_path) -> IncrementalPlanner:
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)
    factory = AbstractPlanFactory(reg.index(), tool_catalog=_TOOL_CATALOG)
    return IncrementalPlanner(
        FakeTransport([]),
        factory,
        spec_names=reg.names(),
        tool_names=[t["name"] for t in _TOOL_CATALOG],
        default_research_spec=DEEP_RESEARCH_SPEC,
    )


class _NullSpan:
    def set_attribute(self, *_a, **_k):  # tracing must never break authoring
        pass


def test_enforce_detects_research_by_role_and_chains_transitively(tmp_path):
    planner = _bare_planner(tmp_path)
    # n1 research (role), n2 synthesize depends on n1, n3 write depends on n2.
    # The sink n3 transitively sees research via n2→n1 → NO repair.
    authored = [
        {"id": "n1", "task": "gather", "tool": None, "spec": None, "specs": [],
         "needs_spec": None, "role": "research", "depends_on": []},
        {"id": "n2", "task": "synthesize", "tool": None, "spec": None, "specs": [],
         "needs_spec": None, "role": "synthesis", "depends_on": ["n1"]},
        {"id": "n3", "task": "write", "tool": "file_write", "spec": None, "specs": [],
         "needs_spec": None, "role": None, "depends_on": ["n2"]},
    ]
    notes = planner._enforce_terminal_research_edge(authored, _NullSpan())
    assert notes == []
    assert authored[2]["depends_on"] == ["n2"]  # unchanged


def test_enforce_connects_only_the_disconnected_sink(tmp_path):
    planner = _bare_planner(tmp_path)
    # n1 research; n2 write depends on n1 (connected); n3 write disconnected (a second
    # terminal writer). Only n3 should gain the research edge; n2 untouched.
    authored = [
        {"id": "n1", "task": "gather", "tool": "web_search", "spec": None, "specs": [],
         "needs_spec": None, "role": None, "depends_on": []},
        {"id": "n2", "task": "write A", "tool": "file_write", "spec": None, "specs": [],
         "needs_spec": None, "role": None, "depends_on": ["n1"]},
        {"id": "n3", "task": "write B", "tool": "file_write", "spec": None, "specs": [],
         "needs_spec": None, "role": None, "depends_on": []},
    ]
    notes = planner._enforce_terminal_research_edge(authored, _NullSpan())
    assert len(notes) == 1 and "n3" in notes[0]
    assert authored[1]["depends_on"] == ["n1"]   # already-connected writer untouched
    assert authored[2]["depends_on"] == ["n1"]   # disconnected writer connected


# --------------------------------------------------------------------------- #
# d7 — finalize repairs a dangling/phantom edge instead of failing
# --------------------------------------------------------------------------- #
def test_d7_parse_dag_safe_repairs_dangling_edge_at_finalize(tmp_path):
    # The finalize path now parses via parse_dag_safe, which drops a phantom-id edge
    # and still builds a valid DAG (the d7 backstop) — verified here on the factory the
    # IncrementalPlanner uses. A genuine invalidity still raises (→ self-heal).
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)
    factory = AbstractPlanFactory(reg.index(), tool_catalog=_TOOL_CATALOG)
    structured = {
        "nodes": [
            {"id": "n1", "task": "gather", "depends_on": []},
            {"id": "n2", "task": "write", "depends_on": ["n1", "ghost_99"]},  # phantom ref
        ],
        "rationale": "dangling-edge plan",
    }
    dag, repairs = factory.parse_dag_safe(structured)
    assert repairs and "ghost_99" in repairs[0]
    assert dag.by_id["n2"].depends_on == ("n1",)  # phantom dropped, real edge kept

    # A real cycle is NOT silently repaired — finalize would surface it to the
    # self-heal as a malformed plan (retry-on-reject), not ship an invalid DAG.
    cyclic = {
        "nodes": [
            {"id": "n1", "task": "a", "depends_on": ["n2"]},
            {"id": "n2", "task": "b", "depends_on": ["n1"]},
        ],
    }
    try:
        factory.parse_dag_safe(cyclic)
        assert False, "a real cycle must still raise PlanError"
    except PlanError:
        pass
