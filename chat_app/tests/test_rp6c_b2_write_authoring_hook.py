"""RP-6c B2 — the ONE-DRIVE WRITE-AUTHORING HOOK (d359/d361, DESIGN §b/§c/§f).

B1 landed the drive MECHANISM (``AgentRuntime.PhaseTransition`` + ``_author_next_phase`` +
``PlanNode.deliverable_path``) with a MINIMAL injected author hook. B2 FILLS that hook with the
REAL implementation: :func:`chat_app.agentic.make_write_phase_author` composes the write goal from
the run's OWN LIVE research state (findings + the verbatim ``[S#]`` SOURCE INDEX + the research
NARRATIVE + the SB-4 ``(summary, memory_index)`` pair), calls ``IncrementalPlanner.plan`` mid-drive
so the MODEL authors the write topology, and returns ONE coherent-document write node (+ optional
``final_review``) — NEVER the N-section whole-document chain (Bug B).

These tests prove, FULLY OFFLINE (no GPU, no live transport), the B2 contract:

1. The hook COMPOSES the write goal from LIVE run state (``rt._cache`` research results + the run's
   ``grower.state`` memory handle), INCLUDING the verbatim ``[S#]`` SOURCE INDEX built from the run's
   own fetched sources — not engine locals hand-carried across a runtime boundary.
2. It authors ONE write node (+ optional ``final_review``) via ``IncrementalPlanner.plan``, and
   returns EXACTLY the model-authored node(s) — the engine composes only DATA, it authors NO
   structure (no N-way whole-document chain). The O4 ONE-NODE directive is passed to the planner.
3. The research→write handoff rides as node context from the SHARED ``ResearchState``: the hook sets
   NO ``chain_sources``/``chain_notes`` cross-runtime bridge on the runtime (Bug C dissolves).
4. The shared composition ``_compose_write_planner_inputs`` (the single source of truth the two-drive
   write phase and this hook both compose through) produces the write goal / prior_memory / source-id
   directive faithfully.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from chat_app import agentic
from agent_runtime.factory import PlanDAG, PlanNode


# --------------------------------------------------------------------------- #
# fixtures — a fake SHARED runtime carrying finished research in ``_cache`` +
# a grower whose ResearchState exposes the run's stable memory handle.
# --------------------------------------------------------------------------- #
MEM_HANDLE = "research-mem-7f3a"
_SRC_A = {"title": "US-Iran conflict overview", "url": "https://example.com/iran-a",
          "markdown": "# Overview\nCasualties: 4175 killed. Cost: $113.3B.\n"}
_SRC_B = {"title": "Damage assessment", "url": "https://example.org/iran-b",
          "markdown": "## Damage\nFordow and Natanz struck June 2025.\n"}


def _research_result(output, fetched, notes):
    """A SubAgentResult-shaped object the agentic collectors read (parsed/output/tool_value)."""
    return SimpleNamespace(
        parsed=None, output=output,
        tool_value={"fetched": list(fetched), "article_notes": list(notes)},
    )


def _fake_rt(cache, *, memory_handle=MEM_HANDLE):
    return SimpleNamespace(
        _cache=dict(cache),
        _grower=SimpleNamespace(state=SimpleNamespace(memory_handle=memory_handle)),
    )


def _research_dag():
    # two finished research nodes; the hook reads their results out of rt._cache in dag order.
    return PlanDAG(
        nodes=[
            PlanNode(id="r1_research", task="research facet 1"),
            PlanNode(id="r2_research", task="research facet 2"),
        ],
        rationale="research", shape="deep-research", growable=True,
    )


def _live_rt_with_two_research_nodes():
    cache = {
        "r1_research": _research_result(
            "Iran suffered 4175 killed; economic cost $113.3B.",
            [_SRC_A],
            [{"claim": "4175 killed", "source_id": 1}],
        ),
        "r2_research": _research_result(
            "Fordow and Natanz were struck in June 2025.",
            [_SRC_B],
            [{"claim": "Fordow struck", "source_id": 2}],
        ),
    }
    return _fake_rt(cache), _research_dag()


class _CapturingPlanner:
    """Stands in for the real IncrementalPlanner: captures the goal/prior_memory/directive the
    hook composed, and returns a caller-supplied model-authored write DAG (the MODEL authoring)."""

    def __init__(self, authored_dag, captured, directive):
        self._authored = authored_dag
        self._captured = captured
        self._captured["directive"] = directive

    async def plan(self, goal, prior_memory=None):
        self._captured["goal"] = goal
        self._captured["prior_memory"] = prior_memory
        return SimpleNamespace(dag=self._authored)


def _patch_planner(monkeypatch, authored_dag, captured):
    def _fake_build(**kwargs):
        return _CapturingPlanner(authored_dag, captured, kwargs.get("authoring_directive", ""))
    monkeypatch.setattr(agentic, "_build_incremental_planner", _fake_build)


def _make_hook():
    return agentic.make_write_phase_author(
        query="Detailed HTML report on the June 2025 US-Iran conflict.",
        out_name="report.html",
        transport=None, registry=None, hook=None,  # only handed to _build_incremental_planner
    )


# =========================================================================== #
# 1. the hook composes the write goal from LIVE run state incl the verbatim [S#] index.
# =========================================================================== #
def test_hook_composes_write_goal_from_live_state_with_verbatim_source_index(monkeypatch):
    rt, dag = _live_rt_with_two_research_nodes()
    authored = PlanDAG(nodes=[PlanNode(id="w1_write", task="Write the whole report")],
                       rationale="write", goal="")
    captured: dict = {}
    _patch_planner(monkeypatch, authored, captured)

    asyncio.run(_make_hook()(rt, dag, "write_plan"))

    goal = captured["goal"]
    # the verbatim [S#] SOURCE INDEX was built from the RUN'S OWN fetched sources (live state).
    assert "SOURCE INDEX" in goal
    assert "[S1]" in goal and "[S2]" in goal
    assert "https://example.com/iran-a" in goal and "https://example.org/iran-b" in goal
    # the research findings prose reached the goal (composed from rt._cache, not an engine local).
    assert "4175" in goal and "Fordow" in goal
    # SB-4 prior_memory carries the SHARED ResearchState's memory handle (read off rt._grower.state).
    assert captured["prior_memory"] and captured["prior_memory"][0]["memory_index"] == MEM_HANDLE


def test_hook_reads_memory_handle_from_shared_grower_state_not_a_passed_local(monkeypatch):
    # a DIFFERENT handle on the run's grower must be the one that flows — proving it is read LIVE.
    rt, dag = _live_rt_with_two_research_nodes()
    rt._grower.state.memory_handle = "a-different-live-handle"
    authored = PlanDAG(nodes=[PlanNode(id="w1_write", task="Write it")], rationale="w", goal="")
    captured: dict = {}
    _patch_planner(monkeypatch, authored, captured)

    nodes = asyncio.run(_make_hook()(rt, dag, "write_plan"))

    assert captured["prior_memory"][0]["memory_index"] == "a-different-live-handle"
    # the authored write node is BOUND to that same live research memory (read-via-tools).
    assert nodes[0].research_memory_handle == "a-different-live-handle"


# =========================================================================== #
# 2. authors ONE write node (+ optional final_review); the MODEL authors the DAG.
# =========================================================================== #
def test_hook_returns_the_single_model_authored_write_node(monkeypatch):
    rt, dag = _live_rt_with_two_research_nodes()
    authored = PlanDAG(nodes=[PlanNode(id="w1_write", task="Write the whole document")],
                       rationale="write", goal="")
    captured: dict = {}
    _patch_planner(monkeypatch, authored, captured)

    nodes = asyncio.run(_make_hook()(rt, dag, "write_plan"))

    # EXACTLY the model-authored node — the engine authored NO extra structure (no N-chain).
    assert [n.id for n in nodes] == ["w1_write"]
    # PURE DELIVERY (d299): _normalize_write_dag only NAMES the output file in the task.
    assert "report.html" in nodes[0].task


def test_hook_allows_write_plus_final_review_but_returns_no_n_chain(monkeypatch):
    # O4: the planner MAY author write + a single final_review; the hook returns both VERBATIM.
    rt, dag = _live_rt_with_two_research_nodes()
    authored = PlanDAG(nodes=[
        PlanNode(id="w1_write", task="Write the whole document"),
        PlanNode(id="final_review", task="Review the finished document", depends_on=("w1_write",)),
    ], rationale="write+review", goal="")
    captured: dict = {}
    _patch_planner(monkeypatch, authored, captured)

    nodes = asyncio.run(_make_hook()(rt, dag, "write_plan"))

    assert [n.id for n in nodes] == ["w1_write", "final_review"]
    # the engine did not fabricate a per-section chain — the node set is the model's, verbatim.
    assert nodes[1].depends_on == ("w1_write",)


def test_hook_passes_the_one_node_directive_forbidding_the_n_chain(monkeypatch):
    rt, dag = _live_rt_with_two_research_nodes()
    authored = PlanDAG(nodes=[PlanNode(id="w1_write", task="Write it")], rationale="w", goal="")
    captured: dict = {}
    _patch_planner(monkeypatch, authored, captured)

    asyncio.run(_make_hook()(rt, dag, "write_plan"))

    directive = captured["directive"]
    # P5: the ONE-NODE contract rides the write-file SHAPE's decompose_methodology
    # (planner-only strategy layer); the per-turn directive is ONLY the source-id lever.
    from chat_app.agentic import _WRITE_FILE_SHAPE
    methodology = getattr(_WRITE_FILE_SHAPE, "decompose_methodology", "")
    assert "EXACTLY ONE write node" in methodology
    assert "final_review" in methodology  # the single reviewer node is part of the strategy
    assert "ONE COHERENT DOCUMENT" not in directive  # the engine constant stays deleted
    # the per-turn SOURCE-ID mandate is the directive (sources present → assign [S#]).
    assert "source_ids" in directive


def test_hook_invokes_model_authoring_and_never_authors_structure_itself(monkeypatch):
    # ANTI-FAB (d310/d317/d319): the ONLY way nodes appear is via IncrementalPlanner.plan (the
    # model authoring). If plan() is never reached, the hook has no nodes to return.
    rt, dag = _live_rt_with_two_research_nodes()
    calls = {"n": 0}

    class _Planner:
        async def plan(self, goal, prior_memory=None):
            calls["n"] += 1
            return SimpleNamespace(dag=PlanDAG(
                nodes=[PlanNode(id="w1_write", task="Write it")], rationale="w", goal=""))

    monkeypatch.setattr(agentic, "_build_incremental_planner", lambda **kw: _Planner())
    nodes = asyncio.run(_make_hook()(rt, dag, "write_plan"))

    assert calls["n"] == 1  # the model authored the topology, exactly once
    assert [n.id for n in nodes] == ["w1_write"]


# =========================================================================== #
# 3. the handoff rides as node context — NO chain_sources/chain_notes bridge is set on rt.
# =========================================================================== #
def test_hook_sets_no_chain_sources_or_chain_notes_bridge_on_the_runtime(monkeypatch):
    rt, dag = _live_rt_with_two_research_nodes()
    authored = PlanDAG(nodes=[PlanNode(id="w1_write", task="Write it")], rationale="w", goal="")
    captured: dict = {}
    _patch_planner(monkeypatch, authored, captured)

    asyncio.run(_make_hook()(rt, dag, "write_plan"))

    # DESIGN §c: research + write share ONE runtime + ONE ResearchState, so the write node reads
    # sources/notes from the shared state — the engine sets NO cross-runtime side-channel here.
    assert getattr(rt, "chain_sources", None) is None
    assert getattr(rt, "chain_notes", None) is None


# =========================================================================== #
# 4. the shared composition helper — the single source of truth.
# =========================================================================== #
def test_compose_write_planner_inputs_builds_goal_prior_memory_and_directive():
    sources = [_SRC_A, _SRC_B]
    write_goal, prior_memory, source_directive = agentic._compose_write_planner_inputs(
        "Report on the conflict.", "report.html",
        "Findings: 4175 killed; $113.3B cost.", sources,
        research_notes=[{"claim": "4175 killed", "source_id": 1}],
        write_planning_event={"kind": "write_plan", "findings_digest": "digest",
                              "memory_handle": MEM_HANDLE},
        research_memory_handle=MEM_HANDLE,
    )
    # the write goal carries the verbatim [S#] index + names the output file.
    assert "[S1]" in write_goal and "[S2]" in write_goal
    assert "report.html" in write_goal
    # prior_memory = the SB-4 (summary, memory_index) pair bound to the research memory.
    assert prior_memory[0]["memory_index"] == MEM_HANDLE
    # P2 de-fabrication: the summary is the code-assembled bounded DIGEST (verbatim
    # [S#] index + pull cursor), NOT the retired engine truncation of raw findings.
    digest = prior_memory[0]["summary"]
    assert "[S1]" in digest and "load_source" in digest
    assert "Findings: 4175 killed" not in digest[:40]  # no raw-findings truncation
    # sources present → the per-turn source-id directive fires (the model assigns [S#]).
    assert "source_ids" in source_directive


def test_compose_write_planner_inputs_no_sources_degrades_cleanly():
    write_goal, prior_memory, source_directive = agentic._compose_write_planner_inputs(
        "Report.", "out.html", "some findings", [],
        research_notes=None, write_planning_event=None, research_memory_handle=None,
    )
    assert "out.html" in write_goal
    assert prior_memory is None          # no memory handle → no SB-4 pair
    assert source_directive == ""        # no sources → no source-id mandate
