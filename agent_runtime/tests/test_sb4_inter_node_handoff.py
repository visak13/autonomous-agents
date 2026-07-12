"""SB-4 (d285) — the COMPACT (summary, memory_index) inter-node handoff.

The d285 contract: the SOLE inter-node context payload a downstream node receives is the
DIRECT-upstream ``(summary, memory_index)`` PAIR — the upstream node's own model digest
(SB-2's ``finalize_node``) plus the index of the research memory its detail lives in. The
THREE prior channels collapse into that one pair: (1) the clipped-prose node output in the
inputs block, (2) the memory-handle string, and (3) the directly-folded upstream fetched
bodies. The DETAIL is pulled from the research memory BY INDEX on demand (SB-1's store),
never dumped into the turn.

These tests prove the gate on the REAL ``SubAgent._compose_task`` + SB-1's
``ResearchMemoryStore`` + SB-3's ``resolve_brief_memory``:

  * a downstream node turn CONTAINS the upstream (summary, index) pair;
  * the upstream's clipped prose (ch1) and folded fetched bodies (ch3) are DROPPED;
  * a fact is pullable from the memory BY INDEX while NOT dumped into the turn;
  * d15 direct-upstream-only is preserved (no transitive context);
  * the rendering is uniform (no spec-name / role-name conditional — anti-fabrication);
  * the per-branch memory OPENING is wired via SB-1's resolver (index → continue, <<NEW>> → fresh).
"""
from __future__ import annotations

import inspect

from agent_runtime import (
    NEW_MEMORY,
    ResearchMemoryStore,
    get_research_memory_store,
    resolve_brief_memory,
)
from agent_runtime.factory import PlanNode
from agent_runtime.research_tree import LeafResult, NEW_MEMORY_SENTINEL
from agent_runtime.runtime import SubAgent
from llm_framework.transport import FakeTransport


# Distinctive markers so a presence/absence assertion is unambiguous.
_PROSE_MARKER = "CLIPPED_PROSE_OUTPUT_MARKER_ch1"
_BODY_MARKER = "RAW_FETCHED_BODY_MARKER_ch3"
_SUMMARY_MARKER = "UPSTREAM_SUMMARY_DIGEST_MARKER"
_INDEX = "research-idx-A"


def _fetched_uv(marker: str) -> dict:
    """An upstream tool_value carrying a fetched source whose body holds ``marker``."""
    return {"fetched": [{"title": "t", "url": "https://example.com/a",
                         "markdown": f"verbatim body ... {marker} ... end"}]}


def _worker(dep_ids, *, upstream_memory=None, upstream_tool_values=None,
            node_id="w1", role="worker"):
    node = PlanNode(id=node_id, task="Write the impact section.",
                    role=role, depends_on=tuple(dep_ids))
    return SubAgent(
        node, transport=FakeTransport(["x"]),
        upstream_memory=upstream_memory,
        upstream_tool_values=upstream_tool_values,
    )


# --------------------------------------------------------------------------- #
# the COLLAPSE: pair rendered, ch1 prose + ch3 fetched bodies dropped
# --------------------------------------------------------------------------- #
def test_pair_rendered_and_prose_plus_fetched_dropped():
    agent = _worker(
        ["A"],
        upstream_memory={"A": {"summary": _SUMMARY_MARKER, "memory_index": _INDEX}},
        upstream_tool_values={"A": _fetched_uv(_BODY_MARKER)},
    )
    user = agent._compose_task({"A": _PROSE_MARKER}, tool_value=None)
    # The SOLE inter-node payload is the (summary, index) pair.
    assert _SUMMARY_MARKER in user
    assert _INDEX in user
    # ch1 (clipped prose) DROPPED for the paired dep.
    assert _PROSE_MARKER not in user
    # ch3 (folded fetched body) DROPPED for the paired dep — detail is pulled by index, not dumped.
    assert _BODY_MARKER not in user
    # It directs read-by-index, never a verbatim dump.
    assert "BY INDEX" in user


def test_byte_identical_channels_when_no_pair_injected():
    """With NO pair injected the pre-SB-4 channels are intact (no regression)."""
    agent = _worker(
        ["A"],
        upstream_memory=None,
        upstream_tool_values={"A": _fetched_uv(_BODY_MARKER)},
    )
    user = agent._compose_task({"A": _PROSE_MARKER}, tool_value=None)
    # The clipped prose (ch1) and the fetched body (ch3) are rendered as before.
    assert _PROSE_MARKER in user
    assert _BODY_MARKER in user
    # No SB-4 pair block appears.
    assert "UPSTREAM RESEARCH (what the previous step" not in user


# --------------------------------------------------------------------------- #
# pull-by-index: the detail is in the memory, NOT dumped into the turn
# --------------------------------------------------------------------------- #
def test_fact_pulled_by_index_not_dumped_into_turn(tmp_path):
    # Upstream node A gathered a fact (figure 4242) into a NEW research memory.
    store = ResearchMemoryStore(tmp_path)
    mem = store.open_memory(NEW_MEMORY)
    index = mem.memory_handle
    mem.append_leaf(
        LeafResult(branch_id="A", question="what happened",
                   findings="digest", notes=[{"claim": "key claim", "url": "u"}],
                   fetched=[{"title": "t", "url": "https://example.com/a",
                             "markdown": "verbatim body with the figure 4242."}]),
        layer=1,
    )
    # Downstream node B receives ONLY A's (summary, index) pair.
    agent = _worker(
        ["A"],
        upstream_memory={"A": {"summary": "A studied the impact", "memory_index": index}},
        upstream_tool_values={"A": {"fetched": [{"title": "t", "url": "u",
                                    "markdown": "verbatim body with the figure 4242."}]}},
    )
    user = agent._compose_task({"A": "prose"}, tool_value=None)
    # The index is named in B's turn so B can pull the detail.
    assert index in user
    # The actual figure is NOT pasted into B's turn (no full-content dump) ...
    assert "4242" not in user
    # ... but it IS readable from the memory BY INDEX (a fresh store instance = a later turn).
    later = ResearchMemoryStore(tmp_path).open_memory(index)
    assert any("4242" in s.get("markdown", "") for s in later.sources())


# --------------------------------------------------------------------------- #
# d15 — DIRECT-upstream-only (no transitive context)
# --------------------------------------------------------------------------- #
def test_direct_upstream_only_no_transitive_pair():
    # B depends only on A. Even if a TRANSITIVE ancestor's pair were handed in, only the
    # node's DIRECT depends_on entries are rendered — the block is built off depends_on.
    agent = _worker(
        ["A"],
        upstream_memory={
            "A": {"summary": "DIRECT_A_SUMMARY", "memory_index": "idx-A"},
            "ZZ_transitive": {"summary": "TRANSITIVE_LEAK", "memory_index": "idx-ZZ"},
        },
    )
    user = agent._compose_task({}, tool_value=None)
    assert "DIRECT_A_SUMMARY" in user
    assert "TRANSITIVE_LEAK" not in user  # a non-direct-dep pair is never rendered


# --------------------------------------------------------------------------- #
# the "overall / joined" reviewer view EMERGES from many direct upstreams (no role branch)
# --------------------------------------------------------------------------- #
def test_many_upstreams_join_into_one_block():
    agent = _worker(
        ["A", "B"], node_id="final_review", role="reviewer",
        upstream_memory={
            "A": {"summary": "SUMMARY_A", "memory_index": "idx-A"},
            "B": {"summary": "SUMMARY_B", "memory_index": "idx-B"},
        },
    )
    user = agent._compose_task({}, tool_value=None)
    # A reviewer joins MANY branches → it naturally sees the full set of (summary, index) pairs.
    for tok in ("SUMMARY_A", "SUMMARY_B", "idx-A", "idx-B"):
        assert tok in user


# --------------------------------------------------------------------------- #
# per-branch memory OPENING wired via SB-1's resolver (SB-3 deferred this to SB-4)
# --------------------------------------------------------------------------- #
def test_resolver_continue_vs_new(tmp_path):
    store = get_research_memory_store(tmp_path)
    # an existing index → CONTINUE that memory (prior detail readable back)
    a = resolve_brief_memory(store, "topic-A")
    a.append_leaf(
        LeafResult(branch_id="b", question="q", findings="d",
                   notes=[{"claim": "c", "url": "u"}],
                   fetched=[{"title": "t", "url": "u", "markdown": "body 9999"}]),
        layer=0,
    )
    again = resolve_brief_memory(store, "topic-A")
    assert again.memory_handle == "topic-A"
    assert any("9999" in s.get("markdown", "") for s in again.sources())
    # the <<NEW>> sentinel → a FRESH, distinct memory
    fresh = resolve_brief_memory(store, NEW_MEMORY_SENTINEL)
    assert fresh.memory_handle != "topic-A"
    # unset / empty also mints fresh
    fresh2 = resolve_brief_memory(store, None)
    assert fresh2.memory_handle not in ("topic-A", fresh.memory_handle)


# --------------------------------------------------------------------------- #
# ANTI-FABRICATION: the pair rendering carries NO spec-name / role-name conditional
# --------------------------------------------------------------------------- #
def test_pair_block_has_no_spec_or_role_conditional():
    src = inspect.getsource(SubAgent._upstream_pair_block)
    # strip the docstring so prose mentions of "role"/"spec" don't trip the check
    body = src.split('"""')[-1]
    lowered = body.lower()
    for banned in ("spec_id", "spec_name", "specialization", "== role",
                   "role ==", "is_review", "role_synthesizer", "if role"):
        assert banned not in lowered, f"anti-fabrication: found {banned!r} in pair rendering"


# --------------------------------------------------------------------------- #
# RUNTIME WIRING: the injected node_finalizer stamps the pair onto each result, and a
# downstream node's turn receives ONLY that pair (the run-loop seam, end to end on a fake).
# --------------------------------------------------------------------------- #
def test_runtime_finalizer_stamps_pair_and_downstream_receives_it(tmp_path):
    import asyncio

    from agent_runtime.runtime import AgentRuntime
    from agent_runtime.factory import PlanDAG
    from reactive_tools import EventPlane, ToolHook, register_agentic_tools

    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)

    # capture each node's composed USER turn
    seen: dict[str, str] = {}
    orig = SubAgent._compose_task

    def _spy(self, inputs, tool_value):
        out = orig(self, inputs, tool_value)
        seen[self.node.id] = out
        return out

    SubAgent._compose_task = _spy
    try:
        async def _finalizer(node, result):
            return {"summary": f"SUMMARY_OF_{node.id}", "memory_index": f"idx_{node.id}"}

        dag = PlanDAG(
            nodes=[
                PlanNode(id="g1", task="Gather facts about the topic.", depends_on=()),
                PlanNode(id="w2", task="Write a brief from the prior step.",
                         depends_on=("g1",)),
            ],
            goal="Research then write.",
        )
        rt = AgentRuntime(
            transport=FakeTransport(["found some facts", "the brief"]),
            hook=hook, max_concurrency=1, node_finalizer=_finalizer,
        )
        out = asyncio.run(rt.run(dag))
        assert out.ok, out.failed
    finally:
        SubAgent._compose_task = orig

    # the finalizer stamped each node's (summary, memory_index) pair onto its cached result
    assert out.results["g1"].summary == "SUMMARY_OF_g1"
    assert out.results["g1"].memory_index == "idx_g1"
    # the downstream node's turn carries g1's pair as its inter-node context
    assert "SUMMARY_OF_g1" in seen["w2"]
    assert "idx_g1" in seen["w2"]
