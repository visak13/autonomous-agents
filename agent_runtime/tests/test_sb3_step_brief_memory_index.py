"""SB-3 (d285) — the STEP BRIEF carries goal + a memory_index | <<NEW>>.

The d285 contract: each plan step's BRIEF carries its goal PLUS a ``memory_index`` —
an existing INDEX to CONTINUE a prior research memory, or the textual ``<<NEW>>``
sentinel to START a fresh one. The PLANNER chooses which by REASONING over the upstream
(summary, memory_index) it received; the chosen value RESOLVES through SB-1's
``get_research_memory_store`` / ``open_memory`` (an index continues that memory; <<NEW>>
mints a fresh one). Built ON SB-1 (the store) + SB-2 (the finalize pair) — no new store.

These tests prove the gate properties on the REAL surfaces:

  (a) a PLANNER-AUTHORED step (the IncrementalPlanner tool-call loop) CONTINUES a prior
      index when handed the upstream (summary, index) — and that index, passed through
      SB-1's tool, round-trips to the SAME memory (prior notes + sources readable back);
  (b) a PLANNER-AUTHORED step uses ``<<NEW>>`` (fresh) when there is no upstream — and
      that resolves to a DISTINCT new memory, isolated from any existing one;
  (c) the resolver maps a brief's memory_index THROUGH SB-1 (index→reuse / <<NEW>>→create);
  (d) the seed-layer (Tree.expand → the research node's brief) and the research-brief
      digest (``to_brief`` / ``build_research_brief``) both carry the field.

Anti-fabrication (d10-clean): the index is the PLANNER's choice over DATA — the engine
STAMPS no index and has ZERO spec-name / role-name conditionals on the brief field
(asserted by reading the resolver + the authoring boundary's own source). SB-3 is a SEAM:
the upstream block is INJECTED here (a test supplies it) exactly as SB-4 will later wire
it from compose-task; SB-3 does not build that handoff.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Sequence

from agent_runtime import (
    NEW_MEMORY,
    NEW_MEMORY_SENTINEL,
    ResearchMemoryStore,
    build_research_brief,
    get_research_memory_store,
    normalize_brief_memory_index,
    resolve_brief_memory,
)
from agent_runtime.factory import AbstractPlanFactory
from agent_runtime.incremental import IncrementalPlanner
from agent_runtime.research_tree import (
    DagGrower,
    LeafResult,
    Tree,
    TreeConfig,
    ResearchState,
)
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


def _add(task: str, *, tool: str = "", memory_index: str = "",
         depends_on: Sequence[str] = ()) -> str:
    args: dict[str, Any] = {"task": task, "tool": tool, "depends_on": list(depends_on)}
    if memory_index:
        args["memory_index"] = memory_index
    return json.dumps({"tool": "add_step", "args": args})


def _finalize() -> str:
    return json.dumps({"tool": "finalize_plan", "args": {}})


def _planner(replies: Sequence[str], tmp_path) -> IncrementalPlanner:
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
    )


def _leaf(*, branch_id: str, question: str, url: str, markdown: str, note: str) -> LeafResult:
    return LeafResult(
        branch_id=branch_id,
        question=question,
        findings=f"digest for {question}",
        notes=[{"claim": note, "url": url}],
        fetched=[{"title": f"title for {url}", "url": url, "markdown": markdown}],
    )


# --------------------------------------------------------------------------- #
# (a) a PLANNER-AUTHORED step CONTINUES a prior index (reuse), resolving via SB-1
# --------------------------------------------------------------------------- #
def test_planner_authored_step_continues_prior_index_reuse(tmp_path):
    # A prior step gathered into a memory; SB-2 finalize handed its (summary, index)
    # downstream. Seed that memory on disk so the reuse can be read back through SB-1.
    store = get_research_memory_store(tmp_path / "mem")
    prior = store.open_memory(NEW_MEMORY)
    prior_index = prior.memory_handle
    prior.append_leaf(
        _leaf(branch_id="b1", question="round 1",
              url="https://example.com/1", markdown="round-1 verbatim body 1234",
              note="prior-line claim"),
        layer=1,
    )

    upstream = [{"memory_index": prior_index,
                 "summary": "Gathered the casualty figures for the conflict."}]

    # The planner authors a FOLLOW-UP step that CONTINUES the prior research line — it
    # sets memory_index to the prior index (its own tool-call choice over the upstream).
    planner = _planner(
        [
            _seed(),
            _add("Deepen the casualty timeline", tool="web_search",
                 memory_index=prior_index),
            _finalize(),
        ],
        tmp_path,
    )

    # The planner REASONED over the upstream: the engine surface that feeds it the prompt
    # carries the upstream (summary, index) block as DATA (d10-clean), not a spec body.
    rendered = planner._initial_user("Deepen the report", upstream)
    assert prior_index in rendered
    assert "UPSTREAM RESEARCH MEMORY" in rendered

    result = _run(planner.plan("Deepen the report", prior_memory=upstream))
    nodes = {n.id: n for n in result.dag.nodes}
    # The authored gather step carries the PLANNER's chosen memory_index = the prior index.
    gather = nodes["n1"]
    assert gather.memory_index == prior_index

    # And that brief value RESOLVES through SB-1's tool back to the SAME memory — the
    # prior notes + verbatim sources read back (continue, not a fresh memory).
    continued = resolve_brief_memory(store, gather.memory_index)
    assert continued.memory_handle == prior_index
    claims = {n.get("claim") for n in continued.collect_notes()}
    assert "prior-line claim" in claims
    assert any("1234" in s["markdown"] for s in continued.sources())


# --------------------------------------------------------------------------- #
# (b) a PLANNER-AUTHORED step uses <<NEW>> when there is NO upstream (fresh)
# --------------------------------------------------------------------------- #
def test_planner_authored_step_uses_new_when_no_upstream(tmp_path):
    # No upstream memory → the planner opens a fresh research line. The model omits
    # memory_index entirely; the authoring boundary canonicalizes the absence to <<NEW>>.
    planner = _planner(
        [
            _seed(),
            _add("Research the new topic from scratch", tool="web_search"),
            _finalize(),
        ],
        tmp_path,
    )
    # With no upstream, the prompt carries NO upstream block (seed authoring, unchanged).
    assert "UPSTREAM RESEARCH MEMORY" not in planner._initial_user("Fresh report", None)

    result = _run(planner.plan("Fresh report"))
    gather = {n.id: n for n in result.dag.nodes}["n1"]
    assert gather.memory_index == NEW_MEMORY_SENTINEL

    # <<NEW>> resolves through SB-1 to a DISTINCT fresh memory, isolated from any existing.
    store = get_research_memory_store(tmp_path / "mem2")
    existing = store.open_memory("existing-line")
    existing.append_leaf(
        _leaf(branch_id="b", question="q", url="https://e/x", markdown="m", note="old claim"),
        layer=1,
    )
    fresh = resolve_brief_memory(store, gather.memory_index)
    assert fresh.memory_handle != "existing-line"
    assert fresh.collect_notes() == []  # the fresh line does not see the existing memory


# --------------------------------------------------------------------------- #
# (c) the resolver maps a brief's memory_index THROUGH SB-1 (reuse vs create)
# --------------------------------------------------------------------------- #
def test_resolve_brief_memory_reuse_and_new(tmp_path):
    store = ResearchMemoryStore(tmp_path)
    # An existing index → the SAME memory, read back across a FRESH store instance (disk).
    seeded = store.open_memory("alpha")
    seeded.append_leaf(
        _leaf(branch_id="b", question="q", url="https://e/a", markdown="body a 42", note="claim a"),
        layer=1,
    )
    later = ResearchMemoryStore(tmp_path)
    reused = resolve_brief_memory(later, "alpha")
    assert reused.memory_handle == "alpha"
    assert any(n.get("claim") == "claim a" for n in reused.collect_notes())

    # <<NEW>>, "" and None each mint a DISTINCT fresh memory (never a shared blank stem).
    a = resolve_brief_memory(store, NEW_MEMORY_SENTINEL)
    b = resolve_brief_memory(store, "")
    c = resolve_brief_memory(store, None)
    handles = {a.memory_handle, b.memory_handle, c.memory_handle, "alpha"}
    assert len(handles) == 4

    # The resolver also accepts a directory ROOT (→ the per-root singleton store).
    root_mem = resolve_brief_memory(tmp_path, "alpha")
    assert root_mem.memory_handle == "alpha"


def test_normalize_brief_memory_index_canonicalizes_absence_to_new():
    assert normalize_brief_memory_index(None) == NEW_MEMORY_SENTINEL
    assert normalize_brief_memory_index("") == NEW_MEMORY_SENTINEL
    assert normalize_brief_memory_index("   ") == NEW_MEMORY_SENTINEL
    assert normalize_brief_memory_index(NEW_MEMORY_SENTINEL) == NEW_MEMORY_SENTINEL
    assert normalize_brief_memory_index("  mem_abc  ") == "mem_abc"  # real index kept


# --------------------------------------------------------------------------- #
# (d) the seed-layer brief + the research-brief digest carry the field
# --------------------------------------------------------------------------- #
def test_seed_branch_brief_carries_memory_index(tmp_path):
    # The model authors a seed branch via expand_branch; its brief carries the choice.
    tree = Tree()
    tree.expand({"question": "facet one"})                         # default → <<NEW>>
    tree.expand({"question": "facet two", "memory_index": "mem_x"})  # continue a line
    by_q = {b.question: b for b in tree.branches.values()}
    assert by_q["facet one"].memory_index == NEW_MEMORY_SENTINEL
    assert by_q["facet two"].memory_index == "mem_x"

    # The grower maps each branch onto its research PlanNode, carrying the brief choice.
    state = ResearchState(tmp_path / "run.jsonl")
    grower = DagGrower(
        transport=None, goal="the goal", spec=None,
        config=TreeConfig(), state=state, tree=tree,
    )
    n_new = grower._research_node("s1_B1", "facet one", (), memory_index=by_q["facet one"].memory_index)
    n_cont = grower._research_node("s1_B2", "facet two", (), memory_index="mem_x")
    assert n_new.memory_index == NEW_MEMORY_SENTINEL
    assert n_cont.memory_index == "mem_x"
    # The node still binds the run's shared research memory handle (per-branch OPENING is SB-4).
    assert n_new.research_memory_handle == state.memory_handle


def test_research_brief_digest_carries_memory_index(tmp_path):
    brief = build_research_brief([], [], topic="t", memory_index="mem_q")
    assert brief["memory_index"] == "mem_q"
    # Absent/None canonicalizes to <<NEW>> (the brief is not yet bound to a memory line).
    assert build_research_brief([], [], topic="t")["memory_index"] == NEW_MEMORY_SENTINEL


# --------------------------------------------------------------------------- #
# anti-fabrication: the engine stamps NO index; zero spec/role conditionals
# --------------------------------------------------------------------------- #
def test_resolver_and_normalizer_have_no_spec_or_role_conditionals():
    src = inspect.getsource(resolve_brief_memory) + inspect.getsource(normalize_brief_memory_index)
    lowered = src.lower()
    for banned in ("spec_id", "spec_name", "role ==", "role==", "if role", "specialization"):
        assert banned not in lowered, f"brief-memory resolution must be domain-neutral; found {banned!r}"
    # The engine RELAYS the planner's choice to SB-1 — it never hardcodes/mints an index of
    # its own (no ``= \"mem_...\"`` literal assignment in the resolver); minting is SB-1's job.
    assert 'open_memory' in inspect.getsource(resolve_brief_memory)
    assert '"mem_' not in inspect.getsource(resolve_brief_memory)
    assert "'mem_" not in inspect.getsource(resolve_brief_memory)


def test_add_step_memory_index_is_not_engine_stamped(tmp_path):
    # The authoring boundary (PlanBuilder.add_step) carries the MODEL's value verbatim and
    # only canonicalizes absence — it never branches on spec/role to choose an index.
    from agent_runtime.plan_tools import PlanBuilder

    src = inspect.getsource(PlanBuilder.add_step)
    assert "memory_index" in src
    b = PlanBuilder(spec_names=[], tool_names=["web_search"])
    b.seed_plan({})
    b.add_step({"task": "gather", "tool": "web_search", "memory_index": "mem_chosen"})
    b.add_step({"task": "gather more", "tool": "web_search"})  # omitted → <<NEW>>
    mis = [n["memory_index"] for n in b.to_structured()["nodes"]]
    assert mis == ["mem_chosen", NEW_MEMORY_SENTINEL]
