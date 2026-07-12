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

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.research_tree import (
    DagGrower,
    ResearchState,
    Tree,
    TreeConfig,
)
from agent_runtime.roles import ROLE_RESEARCHER, ROLE_WORKER, position_framing
from agent_runtime.runtime import AgentRuntime, SubAgent
from agent_runtime.scheduler import ExecutionMode
from agent_runtime.shapes import ShapeSpec, load_shape
from agent_runtime.synth_tools import collect_fetched_sources_full
from specialization.seed import RESEARCH_METHODOLOGY_SPEC

SPEC = "research-analyst"


def _growable_shape(max_layers: int = 5, fan_out: int = 5) -> ShapeSpec:
    return ShapeSpec(
        name="deep-research",
        max_iter=10,
        hard_cap=24,
        execution="deep-research",
        completeness_stop="STOP when every facet is filled.",
        expand_on_gaps=True,
        fan_out=fan_out,
        max_layers=max_layers,
    )


def _seed_dag(shape: ShapeSpec, goal: str = "g") -> PlanDAG:
    """The layer-1 growable research frontier the grower INGESTS (s16/a3: the deterministic
    ``unroll_shape`` is RETIRED). In production the chat_app engine emits a TOOL-LESS
    self-selecting seed and ``_drive_growable`` REPLACES it with the grower's decompose-first
    children before the seed wave; ``DagGrower.grow`` then ingests research nodes the grower
    recognizes SOURCE-AGNOSTICALLY (as4 de-web, d227/d241): by the :data:`ROLE_RESEARCHER`
    gather-node TYPE (d213) — NOT by a ``web_search`` tool — so a TOOL-LESS self-selecting seed
    is folded (HEADSUP1). We construct one equivalent TOOL-LESS ``r1_research`` research node
    here to drive ``grow`` in isolation, tagged growable + carrying the shape's growth bounds."""
    seed = PlanNode(
        id="r1_research",
        task=f"[research · round 1] {position_framing('research')}\n\n{goal}",
        spec=SPEC,
        specs=(SPEC,),
        depends_on=(),
        # SOURCE-AGNOSTIC (as4 / SB-RR d292/d293): the gather node is a TOOL-LESS WORKER (research
        # is a SPECIALIZATION, not a role). The grower folds it for ingest by the research-MEMORY
        # HANDLE it binds on — exactly what real seeds carry (the grower sets it; the chat_app
        # seed gets it via the inject path), NOT a role. Mirror that here so ``grow`` ingests it.
        role=ROLE_WORKER,
        tool=None,
        tool_args={"query": str(goal)[:200]},
        research_memory_handle="r1mem",
    )
    return PlanDAG(
        nodes=[seed],
        rationale=f"{shape.name} growable seed",
        shape=shape.name,
        growable=True,
        fan_out=int(shape.fan_out),
        max_layers=int(shape.max_layers),
        max_sources=int(shape.max_sources),
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
# 1. the growable SEED + the on-disk capability (s16/a3: no deterministic unroll)
# =========================================================================== #
# NOTE (s16/a3 d239/d247): the former tests of the deterministic ``unroll_shape`` grow-branch
# output (seed-only emission) and of the FROZEN unroll (grow=False full-DAG, and the grow=False
# no-op on a shape lacking the capability) are DELETED — ``unroll_shape`` is retired and there is
# no frozen mode. The engine-owned TOOL-LESS growable seed (chat_app._research_seed_dag) is proven
# in chat_app/tests/test_p2_5_consolidation.py; the grower's growth is proven below.


def test_seed_dag_is_growable_and_carries_bounds():
    # The growable layer-1 frontier the grower drives: a single research node, tagged growable,
    # carrying the shape's growth bounds for the runtime (the grower authors the topology).
    dag = _seed_dag(_growable_shape())
    assert [n.id for n in dag.nodes] == ["r1_research"]
    assert dag.by_id["r1_research"].depends_on == ()
    assert dag.growable is True
    assert dag.fan_out == 5 and dag.max_layers == 5


def test_on_disk_deep_research_shape_declares_the_capability():
    shape = load_shape("deep-research")
    assert shape.expand_on_gaps is True
    assert shape.max_layers >= 1 and shape.fan_out >= 1
    # s16/a3: the shape is the deep-research family by its execution token; its research
    # topology is authored at runtime by the grower (no deterministic seed-only unroll to assert).
    assert shape.is_deep_research


# =========================================================================== #
# 2. DagGrower.grow re-frontiers via REAL expand_branch GROUNDED in the note gap
# =========================================================================== #
def test_grow_reads_gap_and_authors_growing_visibility_research_node(tmp_path):
    seed = _seed_dag(_growable_shape())
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
    # SOURCE-AGNOSTIC grown node (as4 de-web): TOOL-LESS (self-selects its gather bundle); a
    # WORKER-default node (SB-RR d292/d293 — research is a SPECIALIZATION, not a role) carrying
    # the research-methodology spec as its self-select lever; bound to the run's research memory.
    assert g.tool is None
    assert g.role == ROLE_WORKER
    assert g.research_memory_handle  # bound to the run's research/complex memory (d221)
    assert "Fordow strike damage extent" in g.task
    assert g.tool_args["query"].startswith("Fordow strike damage extent")
    assert g.depends_on == ("r1_research",)  # depends on ALL prior nodes
    # research-methodology LEADS the composed specs (self-select lever), then the round's
    # output-quality SPEC (research-analyst) — the methodology drives gather self-select.
    assert g.spec == RESEARCH_METHODOLOGY_SPEC
    assert g.specs == (RESEARCH_METHODOLOGY_SPEC, SPEC)
    # the gathered note's GAP actually reached the decision prompt (read-back grounding).
    assert "Fordow damage extent unquantified" in transport.calls[0]
    # the layer trace recorded the gather + expansion.
    assert grower.layers[0]["gathered"] == 1
    assert grower.layers[0]["expanded"] == ["B1"]


# =========================================================================== #
# 3. growth STOPS on stop_research and on no_expansion
# =========================================================================== #
def test_grow_stops_on_stop_research(tmp_path):
    seed = _seed_dag(_growable_shape())
    cache = {"r1_research": _fake_result("findings", note_gap="some gap")}
    transport = _ScriptedDecisionTransport([
        '{"tool":"stop_research","args":{"reason":"every facet is filled"}}',
    ])
    grower = _grower(transport, state_path=str(tmp_path / "s.jsonl"))
    new_nodes, stop = asyncio.run(grower.grow(seed, cache, layer=1))
    assert new_nodes == [] and stop == "agent_sufficient"
    assert grower.stop_reason == "agent_sufficient"


def test_grow_stops_on_no_expansion(tmp_path):
    seed = _seed_dag(_growable_shape())
    cache = {"r1_research": _fake_result("findings", note_gap="some gap")}
    # the model authors NO branch — only a final prose plan → no_expansion (no fabrication).
    transport = _ScriptedDecisionTransport(["FINAL PLAN: nothing more to expand."])
    grower = _grower(transport, state_path=str(tmp_path / "s.jsonl"))
    new_nodes, stop = asyncio.run(grower.grow(seed, cache, layer=1))
    assert new_nodes == [] and stop == "no_expansion"
    assert grower.stop_reason == "no_expansion"


def test_grow_honors_expand_contract_when_stop_called_same_pass(tmp_path):
    """d184 — THE ENGINE HONORS THE expand_branch CONTRACT. The rewritten expand_branch
    description PROMISES the model "this sub-topic WILL be gathered as a new round"; so when a
    decision layer authors an expand_branch AND THEN calls stop_research in the SAME pass, the
    engine GATHERS the authored branch (its round has not run yet — nothing is gathered to stop
    on) instead of silently dropping it and halting. This is the engine doing what the tool
    description says, NOT a stop/expand precedence seatbelt — and it is the broken-link fix
    (previously stop_research was checked BEFORE new_branches, so the expansion vanished)."""
    seed = _seed_dag(_growable_shape())
    cache = {"r1_research": _fake_result("findings", note_gap="Fordow damage unquantified")}
    # SAME decision layer: expand a concern, THEN (incoherently) call stop_research.
    transport = _ScriptedDecisionTransport([
        '{"tool":"expand_branch","args":{"parent":"root","question":"Fordow strike damage extent","rationale":"note gap"}}',
        '{"tool":"stop_research","args":{"reason":"premature — but a branch was just opened"}}',
    ])
    grower = _grower(transport, state_path=str(tmp_path / "contract.jsonl"))

    new_nodes, stop = asyncio.run(grower.grow(seed, cache, layer=1))

    # the engine GATHERS the expansion (the contract) rather than honoring the same-pass stop.
    assert stop is None, "a same-pass stop_research must NOT cancel an authored expansion"
    assert len(new_nodes) == 1 and new_nodes[0].id == "g2_B1"
    assert "Fordow strike damage extent" in new_nodes[0].task
    assert grower.stop_reason is None  # growth continues — the new round will gather
    # the trace records BOTH the expansion AND that a stop was raised (auditable), but the
    # expansion won — the round runs next.
    assert grower.layers[0]["expanded"] == ["B1"]
    assert grower.layers[0]["stop_research"] is not None


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
    seed = _seed_dag(_growable_shape(max_layers=max_layers), "study")
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
    # each a TOOL-LESS self-selecting research node carrying its scoped sub-question (as4
    # de-web): a WORKER-default node (SB-RR) whose research-methodology spec drives gather
    # self-select, not a web tool and not a role (no fabrication).
    assert [n.id for n in seed] == ["s1_B1", "s1_B2", "s1_B3"]
    assert all(
        n.tool is None and n.role == ROLE_WORKER and n.depends_on == ()
        and n.spec == RESEARCH_METHODOLOGY_SPEC for n in seed
    )
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
    seed = _seed_dag(_growable_shape(max_layers=5), "study")
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
    seed = _seed_dag(_growable_shape(max_layers=max_layers), "study")
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


# =========================================================================== #
# 7. s15/a21 — a gap-driven expansion GATHERS a layer-2 wave WITHOUT raising,
#    driven by NATIVE tool_calls (the live E4B surface), incl. a prune + multi-
#    source notes (the data shape the a14 gate's layer-2 grow ran on).
# =========================================================================== #
class _NativeDecisionTransport:
    """Replays decision turns as NATIVE ``message.tool_calls`` (the live Ollama path) — the
    string-JSON fallback is bypassed, so this exercises ``first_native_call`` end to end."""

    def __init__(self, turns: list) -> None:
        self._turns = list(turns)
        self.i = 0

    def chat(self, messages, **opts):
        turn = self._turns[self.i] if self.i < len(self._turns) else "FINAL PLAN."
        self.i += 1
        if isinstance(turn, str):
            return _ChatResult(turn)
        # a list of {"name","arguments"} → a native tool_calls reply (empty prose content).
        res = _ChatResult("")
        res.tool_calls = turn
        return res

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content


def _multi_source_result(nid: str, gap: str):
    """A gathered research node with TWO distinct sources + a gap (the layer-2 graph shape)."""
    notes = [
        {**_note(gap), "url": f"https://ex/{nid}/a", "key_claims": [f"claim A {nid}"]},
        {**_note("needs corroboration"), "url": f"https://ex/{nid}/b",
         "key_claims": [f"claim B {nid}"]},
    ]
    fetched = [
        {"title": "ta", "url": f"https://ex/{nid}/a", "markdown": "body a"},
        {"title": "tb", "url": f"https://ex/{nid}/b", "markdown": "body b"},
    ]
    return SimpleNamespace(output=f"findings {nid}", tool_value={
        "article_notes": notes, "fetched": fetched}, parsed=None)


def test_grow_gathers_layer2_wave_without_raising_native_calls(tmp_path):
    seed = _seed_dag(_growable_shape())
    grower = _grower(transport=None, state_path=str(tmp_path / "l2.jsonl"))
    # Layer-1 decision (NATIVE calls): expand two gaps + prune a seed branch, then a prose plan.
    grower.transport = _NativeDecisionTransport([
        [{"name": "expand_branch",
          "arguments": {"parent": "root", "question": "Fordow strike damage extent",
                        "rationale": "S1 gap"}}],
        [{"name": "expand_branch",
          "arguments": {"parent": "root", "question": "Casualty figures all parties",
                        "rationale": "S2 gap"}}],
        [{"name": "prune_branch",
          "arguments": {"branch": "r1_research", "reason": "off-thesis"}}],
        "FINAL PLAN: deepen the two gaps.",
    ])
    cache = {"r1_research": _multi_source_result("r1_research", "Fordow damage unquantified")}

    new1, stop1 = asyncio.run(grower.grow(seed, cache, layer=1))  # must NOT raise

    assert stop1 is None and [n.id for n in new1] == ["g2_B1", "g2_B2"]
    # the layer-1 grow took the per-concern GRAPH snapshot (the a14 notes_graph signal) AND
    # recorded the prune (the prune_reachable signal) — both were 'under-measured' only because
    # the swallowed crash cut the layer off; with no crash they are present.
    assert grower.layers[0]["graph_shape"] == "per_concern_graph"
    assert grower.layers[0]["expanded"] == ["B1", "B2"]
    assert grower.layers[0]["pruned"] == ["r1_research"]
    assert grower.grow_error is None  # the happy path records NO surfaced error

    # drive the grown wave into the cache and run a SECOND decision layer — still no raise.
    seed.nodes.extend(new1)
    for n in new1:
        cache[n.id] = _multi_source_result(n.id, "still open")
    grower.transport = _NativeDecisionTransport(["FINAL PLAN: settled."])
    new2, stop2 = asyncio.run(grower.grow(seed, cache, layer=2))

    assert new2 == [] and stop2 == "no_expansion"
    assert len(grower.layers) == 2 and grower.layers[1]["gathered"] == 2


# =========================================================================== #
# 8. s15/a21 — a grow() CRASH is SURFACED (full traceback + recorded error),
#    NOT silently swallowed: it must no longer masquerade as a clean early-stop.
# =========================================================================== #
class _RaisingGrower:
    """A duck-typed grower whose ``grow`` RAISES — to prove the drive loop SURFACES the crash
    (records the error, emits it on the grow-layer event) instead of silently breaking. Has no
    ``seed_layer`` (the unrolled seed wave drives normally) so only the GROW step fails."""

    def __init__(self, max_layers: int = 3) -> None:
        self.max_layers = max_layers
        self.config = None
        self.stop_reason = None
        self.grow_error = None
        self.layers: list = []

    async def grow(self, dag, cache, layer):
        raise RuntimeError("simulated layer-2 transport crash")


def test_grow_crash_is_surfaced_not_silently_swallowed(tmp_path, capsys):
    from agent_runtime.runtime import EVENT_GROW_LAYER

    seed = _seed_dag(_growable_shape(max_layers=3), "study")
    transport = _DriveTransport()
    grower = _RaisingGrower(max_layers=3)
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

    out = asyncio.run(rt.run(seed))  # GRACEFUL: the crash must not abort the run

    assert out.ok
    # the seed wave still gathered (partial findings stand).
    assert any(nid.startswith(("r1_", "s1_")) for nid in out.results)
    # SURFACED, not swallowed: the error is RECORDED on the runtime AND the grower, and the
    # stop_reason names the crash (no longer a clean 'agent_sufficient'/'depth_bound').
    assert rt._grow_error and "RuntimeError" in rt._grow_error
    assert grower.grow_error == rt._grow_error
    assert grower.stop_reason == "grow_error"
    assert rt._grow_layers == 1  # stuck at the seed — exactly the a14 symptom, now EXPLAINED
    # the grow-layer event carries the surfaced error (the gate/UI can see the crash).
    err_events = [e for e in events if e.get("error")]
    assert err_events and "RuntimeError" in err_events[-1]["error"]
    # the full traceback was logged to stderr (no longer silent).
    assert "simulated layer-2 transport crash" in capsys.readouterr().err


# =========================================================================== #
# 9. SA-4 (SoC ENGINE-THIN, d254) — the WITHIN-RUN PARITY PROBE: a NON-web gather
#    node dispatches its OWN self-selected tool through the GENERIC by-name hook,
#    emits source-agnostic RECORDS, and a downstream reader (the writer's
#    chain_sources harvest) PULLS them — while the WEB branch is UNCHANGED
#    (contrastive). The engine hardcodes NO web semantics in the gather loop.
# =========================================================================== #
class _ToolResult:
    """Duck-typed reactive_tools tool result (``.ok`` / ``.value`` / ``.error``)."""

    def __init__(self, ok: bool, value=None, error: str = "") -> None:
        self.ok = ok
        self.value = value
        self.error = error
        self.call_id = "c1"


_WEB_URL = "https://news.example.com/iran"


class _DualSourceHook:
    """One hook that dispatches BOTH a WEB tool (web_search/web_fetch) and a NON-WEB
    self-selected bundle tool (codebase ``read_file``) by name — the exact seam the gather
    loop drives (web via ``_dispatch_research_tool``, non-web via ``_invoke_loaded_tool``).
    Records every invocation so the probe proves which path actually fired. No registry is
    needed: in the gather loop the offered tools come from the loaded bundle's catalog
    (``compose_tool_specs``) and dispatch goes straight through ``invoke``."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def invoke(self, name: str, **args) -> _ToolResult:
        self.calls.append(name)
        if name == "web_search":
            return _ToolResult(True, {
                "query": args.get("query", ""),
                "results": [{"title": "Iran", "url": _WEB_URL, "snippet": "snip"}],
                "count": 1,
            })
        if name == "web_fetch":
            url = args.get("url", "")
            return _ToolResult(True, {
                "url": url, "final_url": url, "status": 200, "title": "Iran report",
                "markdown": "REAL WEB BODY: economic damage put at $113.3B.\n\nmore text.",
                "extracted": True,
            })
        if name == "read_file":
            return _ToolResult(True, {
                "path": args.get("path", ""), "found": True,
                "text": "def answer():\n    return 42\n",
                "chars": 27, "total_chars": 27, "truncated": False,
            })
        return _ToolResult(False, error=f"unknown tool {name}")


class _SelfSelectScript:
    """Replays a fixed sequence of agent turns; PREPENDS a get_bundles self-select for the
    named bundle (the node's opening move, d242), then the gather script. Records every user/
    tool turn so the probe can assert the real observation was fed back."""

    def __init__(self, bundle: str, turns: list[str]) -> None:
        lead = f'{{"tool": "get_bundles", "args": {{"name": "{bundle}"}}}}'
        self._turns = [lead] + list(turns)
        self.calls: list[str] = []

    def chat(self, messages, **opts):
        from llm_framework import ChatResult

        user = next(
            (m["content"] for m in reversed(messages) if m.get("role") in ("user", "tool")), ""
        )
        i = len(self.calls)
        self.calls.append(user)
        content = self._turns[i] if i < len(self._turns) else "FALLBACK FINDINGS."
        return ChatResult(role="assistant", content=content)

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content


def _research_node(nid: str, *, role: str, tool, query: str) -> PlanNode:
    return PlanNode(
        id=nid, task=f"[research] {query}", role=role, tool=tool,
        tool_args={"query": query}, spec=SPEC, specs=(SPEC,),
    )


def test_sa4_nonweb_gather_dispatches_own_tool_and_records_pull_into_chain_sources():
    """A NON-web gather node (ROLE_RESEARCHER, tool=None) self-selects the codebase bundle,
    the gather loop dispatches its read_file GENERICALLY (not mis-dispatched as web_fetch),
    emits source-agnostic RECORDS, and the writer's chain_sources harvest PULLS them — the
    full SA-4 spine end-to-end on the REAL SubAgent route, no web semantics involved."""
    hook = _DualSourceHook()
    transport = _SelfSelectScript("codebase", [
        '{"tool": "read_file", "args": {"path": "pkg/mod.py"}}',
        "FINDINGS: pkg/mod.py defines answer() returning 42.",
    ])
    # tool=None + ROLE_RESEARCHER routes to _run_research_loop (the gather route).
    node = _research_node("r1_research", role=ROLE_RESEARCHER, tool=None, query="summarize pkg")
    agent = SubAgent(node, transport=transport, hook=hook,
                     call_opts={"think": False, "temperature": 0})

    res = asyncio.run(agent.run({}))

    # (1) the node DISPATCHED ITS OWN tool by name — never the web fetcher.
    assert "read_file" in hook.calls
    assert "web_fetch" not in hook.calls and "web_search" not in hook.calls
    # (2) GENERIC records-emission: the artifact rides the source-agnostic ``records`` key —
    # NOT the web ``fetched`` key (the engine attached no web shape).
    tv = res.tool_value
    assert tv is not None and "records" in tv and "fetched" not in tv
    rec = tv["records"][0]
    assert rec["url"] == "read_file://pkg/mod.py"  # stable synthetic id (URL-deduped downstream)
    assert "return 42" in rec["markdown"]  # the REAL on-disk-shaped body, captured as a source
    # (3) DOWNSTREAM PULL: the writer's chain_sources harvest (the SAME collector web fetched
    # uses) pulls the non-web record — so a write/section node grounds in it via load_source,
    # mirroring write_report_spa for web. Measure SOURCES (leaf capture), not a fetch count.
    sources = collect_fetched_sources_full([tv])
    assert [s["url"] for s in sources] == ["read_file://pkg/mod.py"]
    assert "return 42" in sources[0]["markdown"]
    # the real read_file observation was fed back to the model (it grounded its findings on it).
    assert any("return 42" in c for c in transport.calls)
    assert "answer() returning 42" in (res.output or "")


def test_sa4_web_gather_branch_is_unchanged_contrastive():
    """CONTRASTIVE (the byte-comparable web branch): the SAME harness drives a WEB gather node
    — it still dispatches web_search → web_fetch and emits the web ``fetched``/``fetched_count``
    shape with NO ``records`` key, and the chain_sources harvest pulls the web source exactly as
    before. The SA-4 fallthrough is UNREACHED for a web tool, so the web path is unchanged."""
    hook = _DualSourceHook()
    transport = _SelfSelectScript("research", [
        '{"tool": "web_search", "args": {"query": "iran damage"}}',
        '{"tool": "web_fetch", "args": {"url": "' + _WEB_URL + '"}}',
        "FINDINGS: economic damage was $113.3B (" + _WEB_URL + ").",
    ])
    node = _research_node("r1_research", role="worker", tool="web_search", query="iran damage")
    agent = SubAgent(node, transport=transport, hook=hook,
                     read_search_max_fetch=3, call_opts={"think": False, "temperature": 0})

    res = asyncio.run(agent.run({}))

    # the web gather fired the web tools (the fallthrough never ran for a web tool).
    assert hook.calls.count("web_search") == 1 and hook.calls.count("web_fetch") == 1
    assert "read_file" not in hook.calls
    # the WEB shape is byte-identical: ``fetched`` + ``fetched_count``, and NO ``records`` key.
    tv = res.tool_value
    assert tv is not None and "fetched" in tv and "records" not in tv
    assert tv["fetched_count"] == 1
    assert {s["url"] for s in tv["fetched"]} == {_WEB_URL}
    # the chain_sources harvest still pulls the web source the same way.
    sources = collect_fetched_sources_full([tv])
    assert [s["url"] for s in sources] == [_WEB_URL]
    assert "$113.3B" in sources[0]["markdown"]


# --------------------------------------------------------------------------- #
# SB-RR (d292/d293): ROLE_RESEARCHER retirement — gather is a SELF-SELECTED specialization,
# every spawned node is a WORKER, and gather/trivial route through ONE unified worker loop.
# --------------------------------------------------------------------------- #
class _ProseOnlyTransport:
    """Every turn returns substantive prose with NO tool call — a trivial worker that needs no
    bundle. Records turns so we can assert it answered in a SINGLE emission (no forced gather)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat(self, messages, **opts):
        from llm_framework import ChatResult
        user = next(
            (m["content"] for m in reversed(messages) if m.get("role") in ("user", "tool")), ""
        )
        self.calls.append(user)
        return ChatResult(role="assistant", content="ANSWER: the capital of France is Paris.")

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content


def test_sbrr_worker_default_node_self_selects_gather_and_routes_to_loop():
    """SB-RR (d292/d293): with ROLE_RESEARCHER RETIRED, a WORKER-DEFAULT node (role=ROLE_WORKER,
    TOOL-LESS) carrying the research-methodology spec SELF-SELECTS the gather (``research``)
    bundle and reaches the unified worker loop's GATHER behavior — gather comes from the
    SELF-SELECTED BUNDLE, never a role. Proven on the REAL SubAgent.run route."""
    hook = _DualSourceHook()
    transport = _SelfSelectScript("research", [
        '{"tool": "web_search", "args": {"query": "iran damage"}}',
        '{"tool": "web_fetch", "args": {"url": "' + _WEB_URL + '"}}',
        "FINDINGS: economic damage was $113.3B (" + _WEB_URL + ").",
    ])
    # WORKER-default + tool-less + research-methodology spec — NO ROLE_RESEARCHER anywhere.
    node = PlanNode(
        id="r1_research", task="[research] iran damage", role=ROLE_WORKER, tool=None,
        tool_args={"query": "iran damage"},
        spec=RESEARCH_METHODOLOGY_SPEC, specs=(RESEARCH_METHODOLOGY_SPEC, SPEC),
    )
    agent = SubAgent(node, transport=transport, hook=hook,
                     read_search_max_fetch=3, call_opts={"think": False, "temperature": 0})

    res = asyncio.run(agent.run({}))

    # it GATHERED via the SELF-SELECTED research bundle: the web tools fired, a real source was
    # captured, and the findings ground in it — all from a WORKER node, no researcher role.
    assert node.role == ROLE_WORKER  # research is a SPECIALIZATION, not a role
    assert hook.calls.count("web_search") == 1 and hook.calls.count("web_fetch") == 1
    tv = res.tool_value
    assert tv is not None and tv.get("fetched_count") == 1
    assert {s["url"] for s in tv["fetched"]} == {_WEB_URL}
    assert "$113.3B" in (res.output or "")


def test_sbrr_trivial_worker_selects_no_bundle_single_emission_not_force_gathered():
    """SB-RR (d293): a trivial WORKER-default node that self-selects NO actionable bundle answers
    in ONE emission via the SAME unified loop — its prose IS the output, no fetch. The no-fab
    GATHER-MORE gate must NOT fire: it is keyed on the SELF-SELECTED ``research`` bundle (not a
    role), so a legitimate non-gathering worker is NEVER force-gathered. RP-3c (d330): the gate
    is now DE-FLAGGED (always-on, no ``verify_lane`` boolean), so this bundle-gating is the ONLY
    thing sparing a non-gathering worker — the invariant matters more, not less. This proves the
    gather mechanics are bundle-gated, the collapse of the old trivial/role=None producer path
    into the unified loop."""
    hook = _DualSourceHook()
    transport = _ProseOnlyTransport()
    node = PlanNode(id="n1", task="What is the capital of France?", role=ROLE_WORKER, tool=None)
    agent = SubAgent(node, transport=transport, hook=hook,
                     call_opts={"think": False, "temperature": 0})

    res = asyncio.run(agent.run({}))

    assert "Paris" in (res.output or "")
    assert hook.calls == []                 # never gathered (no web_search/web_fetch fired)
    assert res.tool_value is None           # no source attached — a pure prose answer
    assert len(transport.calls) == 1        # SINGLE emission — gather-more never forced a re-loop
