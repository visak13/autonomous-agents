"""RP-6c B1 — the ONE-DRIVE PHASE-TRANSITION mechanism (design sections b, e, g / O1 + O2).

RP-6c folds the two ``AgentRuntime.run`` drives (research ``runtime.run`` + a SEPARATE write
``write_runtime.run``, stitched by the engine ``_run_generic_loop`` while-loop) into ONE growable
drive: after the research grow loop stops, a PHASE-TRANSITION step authors the WRITE sub-DAG into
the SAME live run and drives it, so Bug B (N-chain triplication) and Bug C (engine-glued handoff)
dissolve by removing the seam.

B1 is the DRIVE-LEVEL MECHANISM only (B2 = the live write-goal composition + the real
``IncrementalPlanner.plan`` hook; B3 = removing the second ``AgentRuntime.run``). These tests prove,
FULLY OFFLINE (scripted transport, no GPU), the B1 contract:

1. BYTE-PARITY (P2-5b): the research wave loop is UNCHANGED — a growable run with a phase transition
   wired produces the SAME research nodes/results as one without; the transition only ADDS the write
   phase AFTER research stop (within-run contrastive parity).
2. The drive AUTHORS + APPENDS the model-authored write sub-DAG on the research→write transition and
   DRIVES it in the same run (the write node depends on the research sinks — growing visibility).
3. O1 — the writer-route discriminator keys on the NODE (its stamped ``deliverable_path``), NOT the
   runtime-global ``deliverable_path``: in a SHARED runtime (no global set) ONLY the write-phase
   node takes the writer route (``_run_file_delivery``); research nodes keep the research route.
4. O2 — write-phase headroom is reserved WITHIN the one drive: even when the research grow loop is
   stopped by its wall-clock budget (the ``timeout * (1 - headroom_fraction)`` reservation set
   upstream), the write phase STILL authors + runs in the reserved tail.
5. A TERMINAL next phase (``done``) → no transition (byte-identical research-only drive); an author
   hook failure stops the transition GRACEFULLY and is SURFACED (never a silent swallow).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.research_tree import DagGrower, ResearchState, Tree, TreeConfig
from agent_runtime.roles import ROLE_WORKER, position_framing
from agent_runtime.runtime import AgentRuntime, PhaseTransition, SubAgent
from agent_runtime.scheduler import ExecutionMode
from agent_runtime.shapes import ShapeSpec, load_shape
from agent_runtime.synth_tools import DONE_SENTINEL
from llm_framework import FakeTransport
from reactive_tools import EventPlane, ToolHook, register_agentic_tools

SPEC = "research-analyst"
WRITE_MARK = "RP6C_WRITE_NODE"
# a complete, well-formed single document — the write node's raw-file loop emits this then DONE.
_WRITE_DOC = (
    "<!DOCTYPE html><html><head><title>Report</title></head>"
    "<body><h1>US-Iran Report</h1><p>substantive body.</p></body></html>"
)


# --------------------------------------------------------------------------- #
# harness (mirrors test_p2_5b_growable) — a growable seed + a duck-typed grower.
# --------------------------------------------------------------------------- #
def _growable_shape(max_layers: int = 3, fan_out: int = 5) -> ShapeSpec:
    return ShapeSpec(
        name="deep-research", max_iter=10, hard_cap=24, execution="deep-research",
        completeness_stop="STOP when every facet is filled.",
        expand_on_gaps=True, fan_out=fan_out, max_layers=max_layers,
    )


def _seed_dag(shape: ShapeSpec, goal: str = "study") -> PlanDAG:
    seed = PlanNode(
        id="r1_research",
        task=f"[research · round 1] {position_framing('research')}\n\n{goal}",
        spec=SPEC, specs=(SPEC,), depends_on=(), role=ROLE_WORKER, tool=None,
        tool_args={"query": str(goal)[:200]}, research_memory_handle="r1mem",
    )
    return PlanDAG(
        nodes=[seed], rationale=f"{shape.name} growable seed", shape=shape.name,
        growable=True, fan_out=int(shape.fan_out), max_layers=int(shape.max_layers),
        max_sources=int(shape.max_sources),
    )


class _ChatResult:
    def __init__(self, content: str) -> None:
        self.role = "assistant"
        self.content = content
        self.thinking = None
        self.tool_calls = None
        self.raw = None


class _OneDriveTransport:
    """Drives the FULL one-drive run in a single transport: research grow (decision + gather) AND
    the write node's content loop. ``grow=True`` expands one branch per layer (real growth to the
    max_layers ceiling, mirroring test_p2_5b's _DriveTransport); ``grow=False`` never expands
    (seed-only research). The write node is detected by ``WRITE_MARK`` in its prompt: it emits ONE
    document then ``DONE_SENTINEL`` (works for both the raw-file loop and the unified worker loop)."""

    def __init__(self, *, grow: bool = True) -> None:
        self._grow = grow
        self.research_calls = 0
        self.decision_calls = 0
        self.write_calls = 0

    def chat(self, messages, **opts):
        convo = "\n".join(str(m.get("content") or "") for m in messages)
        if opts.get("tools"):
            self.decision_calls += 1
            if self._grow and self.decision_calls % 2 == 1:
                return _ChatResult(
                    '{"tool":"expand_branch","args":{"parent":"root","question":'
                    '"deeper gap ' + str(self.decision_calls) + '","rationale":"r"}}'
                )
            return _ChatResult("FINAL PLAN: gathered this layer's gap.")
        if WRITE_MARK in convo:
            n = sum(1 for m in messages if m.get("role") == "assistant")
            if n == 0:
                self.write_calls += 1
                return _ChatResult(_WRITE_DOC)
            return _ChatResult(DONE_SENTINEL)
        self.research_calls += 1
        return _ChatResult("concrete grounded findings")

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content


def _grower(transport, *, state_path, max_layers=3, fan_out=5, budget_s=0.0) -> DagGrower:
    cfg = TreeConfig(
        depth=max_layers, fan_out=fan_out, decide_max_turns=6, grow_wallclock_budget=budget_s,
    )
    return DagGrower(
        transport=transport, goal="Detailed report on the June 2025 US-Iran conflict.",
        spec=SPEC, config=cfg, state=ResearchState(state_path), tree=Tree(fan_out=fan_out),
        methodology="", stop_criteria="STOP when every facet is filled.", max_layers=max_layers,
    )


def _hook(tmp_path) -> ToolHook:
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    return hook


async def _author_one_write_node(rt, dag, next_plan):
    """A MINIMAL B1 author hook (B2 composes the real write goal + calls IncrementalPlanner.plan):
    returns ONE tool-less write node (no deliverable_path, no deps — the drive stamps + wires
    both). Mirrors what the planner would author for a single-file deliverable."""
    return [PlanNode(
        id="w1_write",
        task=f"{WRITE_MARK} Write the final report to report.html",
        role=None, tool=None,
    )]


def _phase_transition(*, deliverable_path="report.html", next_plan=None, author=None):
    # Default next_plan is the ON-DISK deep-research shape's DECLARED order (RP-6b): research →
    # write_plan. The shape OWNS the sequencing; the drive only READS it.
    shape = load_shape("deep-research")
    return PhaseTransition(
        next_plan=(next_plan or shape.next_phase_plan),
        author=(author or _author_one_write_node),
        deliverable_path=deliverable_path,
        first_kind="research",
    )


def _run(coro):
    return asyncio.run(coro)


def _research_ids(results):
    return sorted(nid for nid in results if nid != "w1_write")


# =========================================================================== #
# 0. the on-disk deep-research shape declares research → write (the DATA the drive reads).
# =========================================================================== #
def test_deep_research_shape_declares_research_to_write():
    shape = load_shape("deep-research")
    assert shape.next_phase_plan("research") == "write_plan"
    assert shape.spec_role_for("write") == "writer"
    # a terminal/unknown phase yields "done" (the drive then stops at research).
    assert shape.next_phase_plan("write") == "done"


# =========================================================================== #
# 1. BYTE-PARITY (P2-5b): the research wave is UNCHANGED; the transition only ADDS the write phase.
# =========================================================================== #
def test_research_wave_byte_parity_with_and_without_transition(tmp_path):
    # WITHOUT a phase transition — the pure growable research drive (the pre-RP-6c baseline).
    seed_a = _seed_dag(_growable_shape(max_layers=3))
    tp_a = _OneDriveTransport(grow=True)
    g_a = _grower(tp_a, state_path=str(tmp_path / "a.jsonl"), max_layers=3)
    rt_a = AgentRuntime(transport=tp_a, execution=ExecutionMode.CONCURRENT,
                        subagent_call_opts={"think": False, "temperature": 0}, grower=g_a)
    out_a = _run(rt_a.run(seed_a))

    # WITH a phase transition wired — identical research inputs.
    seed_b = _seed_dag(_growable_shape(max_layers=3))
    tp_b = _OneDriveTransport(grow=True)
    g_b = _grower(tp_b, state_path=str(tmp_path / "b.jsonl"), max_layers=3)
    rt_b = AgentRuntime(transport=tp_b, execution=ExecutionMode.CONCURRENT,
                        subagent_call_opts={"think": False, "temperature": 0}, grower=g_b)
    rt_b._phase_transition = _phase_transition()
    out_b = _run(rt_b.run(seed_b))

    assert out_a.ok and out_b.ok
    # RESEARCH BYTE-PARITY: the same research node set + the same grow depth + the same stop reason.
    assert _research_ids(out_a.results) == _research_ids(out_b.results)
    assert rt_a._grow_layers == rt_b._grow_layers == 3
    assert g_a.stop_reason == g_b.stop_reason == "depth_bound"
    # the WITHOUT run authored NO phase (byte-identical research-only drive)…
    assert rt_a._phase_authored_plan == "" and "w1_write" not in out_a.results
    # …and the WITH run ADDED exactly the write phase on top of the identical research.
    assert rt_b._phase_authored_plan == "write_plan"
    assert "w1_write" in out_b.results


# =========================================================================== #
# 2. the drive AUTHORS + APPENDS the write sub-DAG on transition, depending on the research sinks.
# =========================================================================== #
def test_transition_appends_write_node_depending_on_research_sinks(tmp_path):
    seed = _seed_dag(_growable_shape(max_layers=3))
    tp = _OneDriveTransport(grow=True)
    g = _grower(tp, state_path=str(tmp_path / "c.jsonl"), max_layers=3)
    rt = AgentRuntime(transport=tp, execution=ExecutionMode.CONCURRENT,
                      subagent_call_opts={"think": False, "temperature": 0}, grower=g)
    rt._phase_transition = _phase_transition()
    out = _run(rt.run(seed))

    assert out.ok
    # the model-authored write node was APPENDED to the LIVE dag + RAN in the same drive.
    assert rt._phase_authored_node_ids == ["w1_write"]
    assert "w1_write" in out.results
    w = seed.by_id["w1_write"]
    # it depends on the research SINKS (nodes nothing else depended on) — runs AFTER all research.
    research_ids = set(_research_ids(out.results))
    assert w.depends_on and set(w.depends_on) <= research_ids
    # every dep is a REAL research node (growing visibility), none dangling.
    assert all(dep in seed.by_id for dep in w.depends_on)


# =========================================================================== #
# 3. O1 — the writer route keys on the NODE's deliverable_path, NOT the runtime-global.
# =========================================================================== #
def test_o1_write_target_is_per_node_data_never_a_route(tmp_path):
    """AUTONOMY REBUILD P2 (supersedes the O1 writer-ROUTE contract): the per-node
    ``deliverable_path`` remains DATA the transition drive stamps (planner/tools read
    it), but it ROUTES nothing — the flag-routing to the raw writer loop is deleted;
    every node runs the unified self-select worker loop."""
    seed = _seed_dag(_growable_shape(max_layers=1))  # seed-only research (no growth)
    tp = _OneDriveTransport(grow=False)
    g = _grower(tp, state_path=str(tmp_path / "d.jsonl"), max_layers=1)
    rt = AgentRuntime(transport=tp, hook=_hook(tmp_path), execution=ExecutionMode.CONCURRENT,
                      subagent_call_opts={"think": False, "temperature": 0}, grower=g)
    assert getattr(rt, "deliverable_path", None) is None
    rt._phase_transition = _phase_transition(deliverable_path="report.html")
    out = _run(rt.run(seed))

    # O1a — the write node was STAMPED with its per-node delivery target by the drive.
    assert seed.by_id["w1_write"].deliverable_path == "report.html"
    # O1c — the research node carries NO delivery target.
    assert seed.by_id["r1_research"].deliverable_path is None
    # P2 — both nodes RAN through the unified worker loop (no special writer dispatch).
    assert "w1_write" in out.results and "r1_research" in out.results


def test_o1_route_ignores_spec_name_keys_only_on_delivery_data(tmp_path):
    """ANTI-FAB (d293/d319, neuron reinforcement 2): the writer route is keyed on the per-node
    DELIVERY-CONTEXT DATA (``deliverable_path``), NEVER on a spec NAME or output format. Two nodes
    carrying the SAME arbitrary spec differ ONLY in delivery data — and ONLY the one with a
    per-node ``deliverable_path`` arms the writer route. A node whose spec is NOT any known writer
    spec still routes when it carries the delivery data (the route reads no spec name)."""
    hook = _hook(tmp_path)
    same_spec = "an-arbitrary-non-writer-spec-name"
    with_data = PlanNode(id="a", task="write it", spec=same_spec, deliverable_path="out.html")
    without_data = PlanNode(id="b", task="write it", spec=same_spec)  # identical spec, no data
    a = SubAgent(with_data, transport=FakeTransport([lambda m, **o: DONE_SENTINEL]), hook=hook,
                 deliverable_path=(with_data.deliverable_path or None))
    b = SubAgent(without_data, transport=FakeTransport([lambda m, **o: "x"]), hook=hook,
                 deliverable_path=(without_data.deliverable_path or None))
    # SAME spec name → the route diverges PURELY on the delivery data, not the spec.
    assert a._deliverable_path == "out.html" and b._deliverable_path is None


def test_o1_subagent_discriminator_is_node_first(tmp_path):
    """The discriminator directly: on a runtime with NO global deliverable_path, a node CARRYING a
    per-node deliverable_path arms the writer route (``SubAgent._deliverable_path`` set); a node
    WITHOUT one does not. This is what routes ONLY the write-phase node in the shared runtime."""
    hook = _hook(tmp_path)
    write_node = PlanNode(id="w", task="write it", deliverable_path="out.html")
    research_node = PlanNode(id="r", task="research it", role=ROLE_WORKER)
    # runtime-global unset → the discriminator must come from the NODE.
    wa = SubAgent(write_node, transport=FakeTransport([lambda m, **o: DONE_SENTINEL]), hook=hook,
                  deliverable_path=(write_node.deliverable_path or None))
    ra = SubAgent(research_node, transport=FakeTransport([lambda m, **o: "x"]), hook=hook,
                  deliverable_path=(research_node.deliverable_path or None))
    assert wa._deliverable_path == "out.html"
    assert ra._deliverable_path is None


# =========================================================================== #
# 4. O2 — write-phase headroom is reserved WITHIN the one drive (write runs after a budget stop).
# =========================================================================== #
def test_o2_write_runs_after_research_budget_stop(tmp_path):
    # a tiny wall-clock budget stops the research grow loop right after the seed wave — the write
    # phase must STILL author + run in the reserved tail (headroom within the single drive).
    seed = _seed_dag(_growable_shape(max_layers=5))
    tp = _OneDriveTransport(grow=True)
    g = _grower(tp, state_path=str(tmp_path / "e.jsonl"), max_layers=5, budget_s=1e-9)
    rt = AgentRuntime(transport=tp, execution=ExecutionMode.CONCURRENT,
                      subagent_call_opts={"think": False, "temperature": 0}, grower=g)
    rt._phase_transition = _phase_transition()
    out = _run(rt.run(seed))

    assert out.ok
    # research stopped GRACEFULLY on the budget (partial, no grown layers)…
    assert g.stop_reason == "budget"
    assert rt._grow_layers == 1
    # …and the write phase STILL ran in the reserved headroom (O2: write is not starved by research).
    assert rt._phase_authored_plan == "write_plan"
    assert "w1_write" in out.results


def test_o2_headroom_fraction_default_is_ten_percent():
    # the ~10% write reservation is the established default (agentic sets grow budget = timeout*0.9).
    pt = PhaseTransition(next_plan=lambda k: "done", author=_author_one_write_node)
    assert pt.headroom_fraction == 0.1


# =========================================================================== #
# 5. edge cases — a TERMINAL next phase is a no-op; an author failure is graceful + SURFACED.
# =========================================================================== #
def test_terminal_next_phase_is_a_noop(tmp_path):
    seed = _seed_dag(_growable_shape(max_layers=3))
    tp = _OneDriveTransport(grow=True)
    g = _grower(tp, state_path=str(tmp_path / "f.jsonl"), max_layers=3)
    rt = AgentRuntime(transport=tp, execution=ExecutionMode.CONCURRENT,
                      subagent_call_opts={"think": False, "temperature": 0}, grower=g)
    # the shape says the next phase is terminal → the drive stops at research (no write authored).
    rt._phase_transition = _phase_transition(next_plan=lambda kind: "done")
    out = _run(rt.run(seed))

    assert out.ok
    assert rt._phase_authored_plan == ""
    assert "w1_write" not in out.results
    # the research drive is byte-identical to the no-wiring case.
    assert rt._grow_layers == 3 and g.stop_reason == "depth_bound"


def test_no_wiring_is_byte_identical(tmp_path):
    seed = _seed_dag(_growable_shape(max_layers=3))
    tp = _OneDriveTransport(grow=True)
    g = _grower(tp, state_path=str(tmp_path / "g.jsonl"), max_layers=3)
    rt = AgentRuntime(transport=tp, execution=ExecutionMode.CONCURRENT,
                      subagent_call_opts={"think": False, "temperature": 0}, grower=g)
    # _phase_transition defaults to None → the drive stops at the research grow loop.
    assert rt._phase_transition is None
    out = _run(rt.run(seed))

    assert out.ok
    assert rt._phase_authored_plan == "" and rt._phase_authored_node_ids == []
    assert "w1_write" not in out.results
    assert rt._grow_layers == 3 and g.stop_reason == "depth_bound"


def test_author_failure_is_graceful_and_surfaced(tmp_path, capsys):
    seed = _seed_dag(_growable_shape(max_layers=3))
    tp = _OneDriveTransport(grow=True)
    g = _grower(tp, state_path=str(tmp_path / "h.jsonl"), max_layers=3)
    rt = AgentRuntime(transport=tp, execution=ExecutionMode.CONCURRENT,
                      subagent_call_opts={"think": False, "temperature": 0}, grower=g)

    async def _raising_author(rt_, dag, next_plan):
        raise RuntimeError("simulated write-authoring crash")

    rt._phase_transition = _phase_transition(author=_raising_author)
    out = _run(rt.run(seed))  # GRACEFUL: the crash must not abort the run

    assert out.ok
    # the research findings stand (the transition failed AFTER research completed)…
    assert _research_ids(out.results) and "w1_write" not in out.results
    assert rt._grow_layers == 3
    # …and the error is SURFACED (recorded + full traceback to stderr), NEVER silently swallowed.
    assert rt._phase_error and "RuntimeError" in rt._phase_error
    assert rt._phase_authored_plan == ""
    assert "simulated write-authoring crash" in capsys.readouterr().err


def test_author_returning_nothing_is_a_noop(tmp_path):
    seed = _seed_dag(_growable_shape(max_layers=3))
    tp = _OneDriveTransport(grow=True)
    g = _grower(tp, state_path=str(tmp_path / "i.jsonl"), max_layers=3)
    rt = AgentRuntime(transport=tp, execution=ExecutionMode.CONCURRENT,
                      subagent_call_opts={"think": False, "temperature": 0}, grower=g)

    async def _empty_author(rt_, dag, next_plan):
        return []

    rt._phase_transition = _phase_transition(author=_empty_author)
    out = _run(rt.run(seed))

    assert out.ok
    assert rt._phase_authored_plan == "" and "w1_write" not in out.results
    assert rt._phase_error is None  # authoring nothing is legitimate, not an error


# =========================================================================== #
# 6. PlanNode.deliverable_path — the per-node O1 field (normalization + round-trip).
# =========================================================================== #
def test_plannode_deliverable_path_field():
    n = PlanNode(id="n", task="t", deliverable_path="report.html")
    assert n.deliverable_path == "report.html"
    assert n.as_dict()["deliverable_path"] == "report.html"
    # a blank/whitespace path is NO path (None) — a stray empty never mis-routes a research node.
    assert PlanNode(id="n", task="t", deliverable_path="   ").deliverable_path is None
    # default (every research/gather/follow-up node) is None → byte-identical routing.
    assert PlanNode(id="n", task="t").deliverable_path is None
