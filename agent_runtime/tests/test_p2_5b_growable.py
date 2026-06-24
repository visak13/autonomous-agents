"""P2.5b (d134/d135) — the GENERIC engine reproduces run_research_tree's ITERATIVE breadth.

The frozen unroll built ALL nodes once (a static DAG the scheduler ran with no
re-frontier), so the generic engine gathered fewer scoped facets than the bespoke tree.
P2.5b relaxes EXACTLY ONE invariant — "node set fixed at unroll time": a shape declares
``expand_on_gaps`` and the runtime's drive loop GROWS the DAG round-by-round by REUSING the
tree's already-generic decision surface (``run_decision_node`` over a persisted
``ResearchState``). These tests prove, FULLY OFFLINE (scripted transport, no GPU):

1. a growable shape unrolls to ONLY the seed research layer + tags the DAG ``growable`` —
   and a NON-growing caller of the same shape still gets the full frozen unroll (no regress);
2. ``DagGrower.grow`` re-frontiers via REAL ``expand_branch`` tool calls GROUNDED in the
   gathered note's gap, mapping each new branch onto a growing-visibility research node;
3. growth STOPS on ``stop_research`` (agent_sufficient) and on ``no_expansion``;
4. the runtime's ``_drive_growable`` end-to-end grows wave-by-wave and is BOUNDED by
   ``max_layers`` (depth_bound) — no unbounded growth.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agent_runtime.research_tree import (
    DagGrower,
    ResearchState,
    Tree,
    TreeConfig,
)
from agent_runtime.runtime import AgentRuntime
from agent_runtime.scheduler import ExecutionMode
from agent_runtime.shapes import ShapeSpec, load_shape, unroll_shape

SPEC = "research-analyst"


def _growable_shape(max_layers: int = 5, fan_out: int = 5) -> ShapeSpec:
    return ShapeSpec(
        name="deep-research",
        max_iter=10,
        hard_cap=24,
        round_roles=("research", "critic"),
        final_roles=("research", "synthesis", "verify"),
        execution="deep-research",
        completeness_stop="STOP when every facet is filled.",
        expand_on_gaps=True,
        fan_out=fan_out,
        max_layers=max_layers,
    )


# --------------------------------------------------------------------------- #
# scripted transport — replays decision-node turns; records each user turn so a
# test can assert a note's gap reached the decision prompt (grounded growth).
# --------------------------------------------------------------------------- #
class _ChatResult:
    def __init__(self, content: str) -> None:
        self.role = "assistant"
        self.content = content
        self.thinking = None
        self.tool_calls = None
        self.raw = None


class _ScriptedDecisionTransport:
    """Replays a fixed list of decision-node turns (string JSON tool calls / prose).

    The decision node falls back to the balanced-brace string parser when a turn carries
    no native ``tool_calls`` (exactly the served defensive path), so a JSON-string turn
    drives the REAL ``Tree.expand`` / ``stop_research`` — genuine tree mutation, not a mock."""

    def __init__(self, turns: list[str]) -> None:
        self._turns = list(turns)
        self.calls: list[str] = []  # the user turn of each chat call

    def chat(self, messages, **opts) -> _ChatResult:
        user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        i = len(self.calls)
        self.calls.append(user)
        content = self._turns[i] if i < len(self._turns) else "FINAL PLAN (fallback)."
        return _ChatResult(content)

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content


def _note(gap: str) -> dict:
    return {
        "source_id": "1", "url": "https://ex/1", "title": "t",
        "source_trust": "primary", "category": "x", "summary": "s",
        "key_claims": ["a claim"], "relevance": "r", "gaps_or_followups": [gap],
    }


def _fake_result(output: str, *, note_gap: str | None = None):
    """A duck-typed SubAgentResult: the grower reads ``.output`` + ``.tool_value``."""
    tv = None
    if note_gap is not None:
        tv = {
            "article_notes": [_note(note_gap)],
            "fetched": [{"title": "t", "url": "https://ex/1", "markdown": "body"}],
        }
    return SimpleNamespace(output=output, tool_value=tv, parsed=None)


def _grower(transport, *, state_path, max_layers=5, fan_out=5) -> DagGrower:
    cfg = TreeConfig(depth=max_layers, fan_out=fan_out, decide_max_turns=6)
    return DagGrower(
        transport=transport,
        goal="Detailed report on the June 2025 US-Iran conflict.",
        spec=SPEC,
        config=cfg,
        state=ResearchState(state_path),
        tree=Tree(fan_out=fan_out),
        methodology="",
        stop_criteria="STOP when every facet is filled.",
        max_layers=max_layers,
    )


# =========================================================================== #
# 1. the unroll emits ONLY the seed layer for a growable shape (+ no regression)
# =========================================================================== #
def test_growable_unroll_emits_seed_layer_only_and_tags_dag():
    dag = unroll_shape(_growable_shape(), "study the topic", spec=SPEC, grow=True)
    # ONLY the seed RESEARCH node (the per-round critic is dropped — the decision node is
    # the critic, mirroring run_research_tree's gather-then-decide layer).
    assert [n.id for n in dag.nodes] == ["r1_research"]
    assert dag.by_id["r1_research"].tool == "web_search"
    assert dag.by_id["r1_research"].depends_on == ()
    # the DAG is tagged growable and carries the shape's growth bounds for the runtime.
    assert dag.growable is True
    assert dag.fan_out == 5 and dag.max_layers == 5


def test_non_growing_caller_of_same_shape_gets_full_frozen_unroll():
    shape = _growable_shape()
    # grow defaults False (the inline _run_deep_research route) → the FULL frozen unroll,
    # byte-identical to pre-P2.5b: the shape gaining expand_on_gaps never silently turns a
    # non-growing caller's plan into a seed-only DAG.
    frozen = unroll_shape(shape, "g", spec=SPEC)
    assert frozen.growable is False
    assert len(frozen.nodes) > 1
    # the research-position node count == the effective round count (every round has one).
    assert sum(1 for n in frozen.nodes if n.id.endswith("_research")) == shape.max_iter


def test_grow_flag_is_noop_on_a_shape_without_the_capability():
    plain = ShapeSpec(
        name="plain", max_iter=2, hard_cap=4,
        round_roles=("research", "critic"),
        final_roles=("research", "synthesis", "verify"),
        execution="deep-research",
    )  # no expand_on_gaps
    dag = unroll_shape(plain, "g", spec=SPEC, grow=True)
    assert dag.growable is False
    assert len(dag.nodes) > 1  # full frozen unroll — the capability must be DECLARED


def test_on_disk_deep_research_shape_declares_the_capability():
    shape = load_shape("deep-research")
    assert shape.expand_on_gaps is True
    assert shape.max_layers >= 1 and shape.fan_out >= 1
    # the shipped shape unrolls seed-only under grow=True.
    assert [n.id for n in unroll_shape(shape, "g", spec=SPEC, grow=True).nodes] == ["r1_research"]


# =========================================================================== #
# 2. DagGrower.grow re-frontiers via REAL expand_branch GROUNDED in the note gap
# =========================================================================== #
def test_grow_reads_gap_and_authors_growing_visibility_research_node(tmp_path):
    seed = unroll_shape(_growable_shape(), "g", spec=SPEC, grow=True)
    # the seed research node has been gathered — its note exposes a concrete GAP.
    cache = {"r1_research": _fake_result("seed findings", note_gap="Fordow damage extent unquantified")}
    # the model EXPANDS into that gap (a real expand_branch tool call, string-parsed).
    transport = _ScriptedDecisionTransport([
        '{"tool":"expand_branch","args":{"parent":"root","question":"Fordow strike damage extent","rationale":"note gap"}}',
        "FINAL PLAN: gathered the gap.",
    ])
    grower = _grower(transport, state_path=str(tmp_path / "state.jsonl"))

    new_nodes, stop = asyncio.run(grower.grow(seed, cache, layer=1))

    assert stop is None and len(new_nodes) == 1
    g = new_nodes[0]
    # the new node is a growing-visibility RESEARCH node grounded in the gap question.
    assert g.id == "g2_B1"
    assert g.tool == "web_search"
    assert "Fordow strike damage extent" in g.task
    assert g.tool_args["query"].startswith("Fordow strike damage extent")
    assert g.depends_on == ("r1_research",)  # depends on ALL prior nodes
    assert g.spec == SPEC
    # the gathered note's GAP actually reached the decision prompt (read-back grounding).
    assert "Fordow damage extent unquantified" in transport.calls[0]
    # the layer trace recorded the gather + expansion.
    assert grower.layers[0]["gathered"] == 1
    assert grower.layers[0]["expanded"] == ["B1"]


# =========================================================================== #
# 3. growth STOPS on stop_research and on no_expansion
# =========================================================================== #
def test_grow_stops_on_stop_research(tmp_path):
    seed = unroll_shape(_growable_shape(), "g", spec=SPEC, grow=True)
    cache = {"r1_research": _fake_result("findings", note_gap="some gap")}
    transport = _ScriptedDecisionTransport([
        '{"tool":"stop_research","args":{"reason":"every facet is filled"}}',
    ])
    grower = _grower(transport, state_path=str(tmp_path / "s.jsonl"))
    new_nodes, stop = asyncio.run(grower.grow(seed, cache, layer=1))
    assert new_nodes == [] and stop == "agent_sufficient"
    assert grower.stop_reason == "agent_sufficient"


def test_grow_stops_on_no_expansion(tmp_path):
    seed = unroll_shape(_growable_shape(), "g", spec=SPEC, grow=True)
    cache = {"r1_research": _fake_result("findings", note_gap="some gap")}
    # the model authors NO branch — only a final prose plan → no_expansion (no fabrication).
    transport = _ScriptedDecisionTransport(["FINAL PLAN: nothing more to expand."])
    grower = _grower(transport, state_path=str(tmp_path / "s.jsonl"))
    new_nodes, stop = asyncio.run(grower.grow(seed, cache, layer=1))
    assert new_nodes == [] and stop == "no_expansion"
    assert grower.stop_reason == "no_expansion"


# =========================================================================== #
# 4. the runtime _drive_growable grows wave-by-wave and is BOUNDED by max_layers
# =========================================================================== #
class _DriveTransport:
    """Drives the FULL runtime: a research-node call answers raw findings; a decision-node
    call (carrying ``tools=``) authors EXACTLY ONE child per layer then closes the layer
    with a prose plan. The decision node is keyed off the native ``tools=`` kwarg, so the
    research and decision turns never cross. Because every reachable decision layer expands,
    growth would run forever if unbounded — proving the ``max_layers`` ceiling terminates it."""

    def __init__(self) -> None:
        self.research_calls = 0
        self.decision_calls = 0

    def chat(self, messages, **opts):
        if opts.get("tools"):
            self.decision_calls += 1
            # Odd decision turn → expand one branch; even turn → a final prose plan that
            # ends THIS decision layer with exactly one new branch (deterministic growth).
            if self.decision_calls % 2 == 1:
                n = self.decision_calls
                return _ChatResult(
                    '{"tool":"expand_branch","args":{"parent":"root","question":'
                    '"deeper gap ' + str(n) + '","rationale":"r"}}'
                )
            return _ChatResult("FINAL PLAN: gathered this layer's gap.")
        self.research_calls += 1
        return _ChatResult("concrete grounded findings")

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content


def test_runtime_drive_growable_grows_and_is_bounded_by_max_layers(tmp_path):
    max_layers = 3
    seed = unroll_shape(_growable_shape(max_layers=max_layers), "study", spec=SPEC, grow=True)
    transport = _DriveTransport()
    grower = _grower(transport, state_path=str(tmp_path / "drive.jsonl"), max_layers=max_layers)
    rt = AgentRuntime(
        transport=transport,
        execution=ExecutionMode.CONCURRENT,
        subagent_call_opts={"think": False, "temperature": 0},
        grower=grower,
    )

    out = asyncio.run(rt.run(seed))

    assert out.ok
    # DECOMPOSE-FIRST: the seed layer is the model's decomposed children (s1_*), NOT the
    # unrolled whole-goal r1_research (which is replaced before the seed wave).
    assert any(nid.startswith("s1_") for nid in out.results)
    assert "r1_research" not in out.results
    # the node set GREW past the seed (the relaxed invariant): grown research nodes at
    # layer 2 (g2_*) and layer 3 (g3_*) — and NOTHING past the bound.
    grown_ids = sorted(nid for nid in out.results if nid.startswith("g"))
    assert grown_ids, "the DAG never grew past the seed"
    assert any(nid.startswith("g2_") for nid in grown_ids)
    assert any(nid.startswith("g3_") for nid in grown_ids)
    assert not any(nid.startswith("g4_") for nid in grown_ids)  # BOUNDED at max_layers=3
    # BOUNDED: exactly max_layers research layers ran, then the hard ceiling stopped it
    # (the model kept expanding every layer — only the bound terminates the loop).
    assert rt._grow_layers == max_layers
    assert grower.stop_reason == "depth_bound"
    # the decision node fired once per grown boundary (layers 1->2 and 2->3) = max_layers-1.
    assert len(grower.layers) == max_layers - 1


# =========================================================================== #
# 5. DECOMPOSE-FIRST seed — breadth is front-loaded (mirrors the tree's seed_only_root)
# =========================================================================== #
def test_seed_layer_decomposes_goal_into_independent_research_frontier(tmp_path):
    # the model decomposes the goal into THREE scoped sub-questions, then writes a prose plan.
    transport = _ScriptedDecisionTransport([
        '{"tool":"expand_branch","args":{"question":"ideological roots of the conflict"}}',
        '{"tool":"expand_branch","args":{"question":"timeline of key 2025 events"}}',
        '{"tool":"expand_branch","args":{"question":"damage and casualty figures"}}',
        "DECOMPOSED: three scoped sub-questions.",
    ])
    grower = _grower(transport, state_path=str(tmp_path / "seed.jsonl"))
    seed = asyncio.run(grower.seed_layer())

    # the seed frontier is the decomposed children — INDEPENDENT (gathered concurrently),
    # each a web_search research node carrying its scoped sub-question (no fabrication).
    assert [n.id for n in seed] == ["s1_B1", "s1_B2", "s1_B3"]
    assert all(n.tool == "web_search" and n.depends_on == () for n in seed)
    assert "ideological roots" in seed[0].task
    assert "timeline of key 2025 events" in seed[1].tool_args["query"]


def test_seed_layer_returns_empty_when_model_authors_no_child(tmp_path):
    # the model writes a plan WITHOUT decomposing → [] so the caller keeps the whole-goal seed.
    transport = _ScriptedDecisionTransport(["No decomposition — just research the whole goal."])
    grower = _grower(transport, state_path=str(tmp_path / "seed2.jsonl"))
    assert asyncio.run(grower.seed_layer()) == []


# =========================================================================== #
# 6. P2-5c FORWARD HARDENING — wall-clock budget gives a GRACEFUL partial stop
# =========================================================================== #
def _budget_grower(transport, *, state_path, max_layers, budget_s) -> DagGrower:
    """A grower whose config carries a wall-clock growth budget (the P2-5c knob)."""
    cfg = TreeConfig(
        depth=max_layers, fan_out=5, decide_max_turns=6,
        grow_wallclock_budget=budget_s,
    )
    return DagGrower(
        transport=transport,
        goal="Detailed report on the June 2025 US-Iran conflict.",
        spec=SPEC,
        config=cfg,
        state=ResearchState(state_path),
        tree=Tree(fan_out=5),
        methodology="",
        stop_criteria="STOP when every facet is filled.",
        max_layers=max_layers,
    )


def test_drive_growable_wallclock_budget_stops_gracefully_with_partial(tmp_path):
    """A tiny wall-clock budget stops the growable loop AFTER the seed wave with
    stop_reason='budget' — a GRACEFUL partial (the seed's findings/sources stand), NOT an
    exception/abort — even though the model would otherwise keep expanding every layer."""
    # max_layers=5 + a transport that expands every layer → would grow to the ceiling; the
    # budget must cut it short FIRST. Budget so small it is already exceeded by the time the
    # seed wave completes, so growth never authors a second wave.
    seed = unroll_shape(_growable_shape(max_layers=5), "study", spec=SPEC, grow=True)
    transport = _DriveTransport()
    grower = _budget_grower(
        transport, state_path=str(tmp_path / "budget.jsonl"), max_layers=5, budget_s=1e-9,
    )
    rt = AgentRuntime(
        transport=transport,
        execution=ExecutionMode.CONCURRENT,
        subagent_call_opts={"think": False, "temperature": 0},
        grower=grower,
    )

    out = asyncio.run(rt.run(seed))  # must NOT raise — graceful partial

    # GRACEFUL: the run completed, the seed wave's partial findings stand.
    assert out.ok
    assert any(nid.startswith("s1_") for nid in out.results), "the seed wave did not gather"
    # the budget cut growth short BEFORE the max_layers ceiling (no g4_/g5_ explosion).
    assert grower.stop_reason == "budget"
    assert rt._grow_layers == 1  # stopped right after the seed wave, no grown layers
    assert not any(nid.startswith("g2_") for nid in out.results)


def test_drive_growable_emits_per_layer_progress_events(tmp_path):
    """P2-5c — the growable drive emits an EVENT_GROW_LAYER progress event per wave (layer
    index, nodes dispatched, cumulative sources, elapsed wall-clock, stop_reason) so a long
    live run is observable. The seed wave + each grown wave + the terminal stop all emit."""
    from agent_runtime.runtime import EVENT_GROW_LAYER

    max_layers = 3
    seed = unroll_shape(_growable_shape(max_layers=max_layers), "study", spec=SPEC, grow=True)
    transport = _DriveTransport()
    grower = _grower(transport, state_path=str(tmp_path / "evt.jsonl"), max_layers=max_layers)
    rt = AgentRuntime(
        transport=transport,
        execution=ExecutionMode.CONCURRENT,
        subagent_call_opts={"think": False, "temperature": 0},
        grower=grower,
    )

    events: list[dict] = []
    orig_emit = rt._emit

    async def _recording_emit(kind, payload):
        if kind == EVENT_GROW_LAYER:
            events.append(dict(payload))
        return await orig_emit(kind, payload)

    rt._emit = _recording_emit  # type: ignore[method-assign]

    asyncio.run(rt.run(seed))

    # at least the seed event + the terminal stop event emitted.
    assert events, "no per-layer progress events emitted"
    # every event carries the observability fields (the long-run signal).
    for ev in events:
        assert set(ev) >= {"layer", "nodes_dispatched", "nodes_total", "sources_so_far",
                           "elapsed_s", "stop_reason"}
        assert isinstance(ev["elapsed_s"], (int, float))
    # the FIRST event is the seed wave (layer 1, no stop yet).
    assert events[0]["layer"] == 1 and events[0]["stop_reason"] is None
    # the LAST event carries the terminal stop reason (depth_bound at the max_layers ceiling).
    assert events[-1]["stop_reason"] == "depth_bound"
