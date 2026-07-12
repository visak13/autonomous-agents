"""P2-5c — the GENERIC engine is the served report DEFAULT (flag-free end-state, d135/d65).

FAST offline tests (no Ollama, no web, no GPU) for the P2-5c retirement step:

* the reversible ``RA_GENERIC_REPORT_PATH`` flag + ``_generic_report_path_enabled`` are GONE,
  and the bespoke ``run_research_tree`` orchestrator + its ``_make_tree_gather`` leaf are
  retired (not attributes of ``chat_app.agentic`` any more);
* the served report path (``run_plan_chain``) runs the GENERIC declarative-unroll engine
  UNCONDITIONALLY (engine == "generic-unroll"), with NO tree branch to fall back to;
* CF#1 — the generic research path is DESCRIPTION-driven (an unrolled research-only DAG),
  not the hardcoded ``_DECISION_INSTRUCTION`` / ``_DECOMPOSE_INSTRUCTION`` tree prompts;
* CF#2 — the P2.2 event-driven reactor + framework-injected review are ALWAYS ON on the
  served report write phase (they were gated behind the retired flag → now LIVE by default);
* CF#3 — the duplicate-section-collapse backstops are RETAINED (not retired);
* CF#4 — the sibling ``_run_deep_research_sectioned`` route ALSO runs the generic engine and
  carries the shape completeness-STOP into the grower's decision node;
* (c) the framework-injected review renders RAW worker content, never a verdict-and-findings
  envelope (the P2-2 foldverify property, re-confirmed on this path);
* the HARD PARITY GATE logic (chat_app.parity) still computes (kept for the parity harness).
"""
from __future__ import annotations

import asyncio
import inspect

from llm_framework import ChatResult
from reactive_tools import EventPlane, ToolHook
from reactive_tools.tool_hook import ToolRegistry
from specialization import SpecRegistry

import chat_app.agentic as agentic
from chat_app.agentic import (
    PLAN_CHAIN_TREE_BREADTH,
    _build_acyclic_runtime,
    _run_generic_research_phase,
    run_plan_chain,
    run_section_write_phase,
)
from chat_app.parity import parity_metrics, parity_verdict
from agent_runtime import ShapeSelection, ShapeSpec
from agent_runtime.factory import PlanDAG, PlanNode


_SRC = {"title": "Iran 2025", "url": "https://news.example.com/iran-2025",
        "markdown": "The conflict escalated June 13; 1,200 casualties by June 20."}


def _deep_research_shape() -> ShapeSpec:
    """A small deep-research ShapeSpec carrying the P2.4 completeness_stop (s16/a3: identified
    by its execution token; its research topology is authored at runtime by the grower)."""
    return ShapeSpec(
        name="deep-research",
        description="deep research",
        max_iter=2,
        hard_cap=4,
        execution="deep-research",
        completeness_stop="Fill ALL the blanks across timeline/costs/impact before stopping.",
        expand_on_gaps=True,
        fan_out=5,
        max_layers=5,
    )


def test_research_seed_dag_is_tool_less_self_selecting_growable():
    """s16/a3 (d239/d247) read-verify bar (ii): the ENGINE-owned research seed builder emits
    a single TOOL-LESS self-selecting research node (no shape-bound web_search; d242) + tags
    the DAG growable, carrying the shape's growth bounds for the grower. The grower then
    AUTHORS the real topology by reasoning (decompose-first → grow on note gaps)."""
    shape = _deep_research_shape()
    dag = agentic._research_seed_dag(shape, "US-Iran June 2025 detailed report", spec="research-analyst")
    assert [n.id for n in dag.nodes] == ["r1_research"]
    seed = dag.nodes[0]
    # TOOL-LESS: neither the shape nor the seed binds a gather tool — the node self-selects
    # its research bundle at runtime (as2/d242). NO deterministic web_search position.
    assert seed.tool is None
    assert seed.tool_args == {}
    assert seed.depends_on == ()
    # SB-RR (d292/d293): the seed is a WORKER-default node (research is a SPECIALIZATION, not a
    # role) whose research-methodology spec LEADS its composed specs — that spec is the
    # self-select lever that makes the seed worker self-select its gather bundle.
    from agent_runtime.roles import ROLE_WORKER
    from specialization.seed import RESEARCH_METHODOLOGY_SPEC
    assert seed.role == ROLE_WORKER
    assert seed.effective_specs == (RESEARCH_METHODOLOGY_SPEC, "research-analyst")
    # growable + carries the shape's growth bounds (the grower drives the real topology).
    assert dag.growable is True
    assert dag.fan_out == 5 and dag.max_layers == 5


class _ProseTransport:
    """A scripted transport whose every turn returns prose (no tool call).

    Enough for the wiring tests where the real research/decision turns are stubbed out;
    keeps the loops short and deterministic."""

    def complete(self, messages, **opts) -> str:  # pragma: no cover - parity shim
        return self.chat(messages, **opts).content

    def chat(self, messages, **opts) -> ChatResult:
        return ChatResult(role="assistant", content="PLAN: gather then synthesize.")


class _FakeWriteResult:
    def __init__(self) -> None:
        self.results: dict = {}
        self.states: dict = {}
        self.launch_order: list = []
        self.ok = True


# --------------------------------------------------------------------------- #
# 1. the flag + the bespoke orchestrator are RETIRED (flag-free end-state, d65)
# --------------------------------------------------------------------------- #
def test_generic_report_flag_and_tree_orchestrator_are_retired():
    # The reversible flag is GONE — no toggle function, and the env is never READ any more
    # (a doc comment may still name it historically; what matters is no os.getenv on it, so
    # nothing can re-enable a tree branch).
    assert not hasattr(agentic, "_generic_report_path_enabled")
    src = inspect.getsource(agentic)
    # The flag name must NOT appear in any executable (non-comment) line — i.e. nothing READS
    # the env to re-enable a tree branch. Historical doc-comment mentions are allowed.
    code_lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    assert all("RA_GENERIC_REPORT_PATH" not in ln for ln in code_lines)
    # The bespoke orchestrator + its leaf gather are retired (not attributes any more).
    assert not hasattr(agentic, "run_research_tree")
    assert not hasattr(agentic, "_make_tree_gather")
    # The retired symbols are also gone from agent_runtime's public surface.
    import agent_runtime as ar
    assert not hasattr(ar, "run_research_tree")
    assert not hasattr(ar, "TreeRunResult")
    assert not hasattr(ar, "GatherFn")


# --------------------------------------------------------------------------- #
# 2. the served report path runs the GENERIC engine UNCONDITIONALLY (no flag)
# --------------------------------------------------------------------------- #
def _spy_write_phase():
    calls: list[dict] = []

    async def spy(query, out_name, findings, sources, **kw):
        calls.append({"findings": findings, "sources": sources,
                      "outline_hint": kw.get("outline_hint")})
        return PlanDAG(nodes=[], goal=query), _FakeWriteResult()

    return spy, calls


def test_plan_chain_uses_generic_engine_by_default(monkeypatch, tmp_path):
    """No flag, no env — the served report path runs the generic engine by default."""
    monkeypatch.delenv("RA_GENERIC_REPORT_PATH", raising=False)  # even if a stray env exists

    async def fake_generic(query, **kw):
        assert kw["dr_shape"].name == "deep-research"   # the resolved deep-research shape
        # the SAME shape completeness_stop is handed to the growable decision node.
        assert "Fill ALL the blanks" in (kw.get("completeness_stop") or "")
        return "GENERIC FINDINGS", [dict(_SRC)], {"growable": True, "stop_reason": "agent_sufficient",
                                                   "grow_layers": 1,
                                                   "layers": [{"gathered": 4}]}
    monkeypatch.setattr(agentic, "_run_generic_research_phase", fake_generic)

    spy, calls = _spy_write_phase()
    monkeypatch.setattr(agentic, "run_section_write_phase", spy)

    hook = ToolHook(EventPlane(), registry=ToolRegistry())
    result = asyncio.run(run_plan_chain(
        "detailed HTML report on the 2025 US-Iran war",
        ShapeSelection(shape="plan-chain", escalate=False, rationale="r"),
        transport=_ProseTransport(), registry=SpecRegistry(str(tmp_path)),
        hook=hook, plane=hook.plane, timeout=30.0, run_id="default",
        overall_goal="detailed HTML report on the 2025 US-Iran war",
        # run_agentic resolves the shape completeness_stop and forwards it; the report path
        # hands it verbatim to the growable decision node.
        completeness_stop="Fill ALL the blanks across timeline/costs/impact before stopping.",
        catalog={"deep-research": _deep_research_shape()},
    ))
    assert result.deep_research["engine"] == "generic-unroll"
    assert len(calls) == 1
    fed = calls[0]
    assert fed["findings"] == "GENERIC FINDINGS"
    assert fed["sources"][0]["url"] == _SRC["url"]
    # the generic engine has no tree-authored outline → PHASE-2 falls back to findings.
    assert fed["outline_hint"] is None


def test_plan_chain_resolves_shape_without_catalog(monkeypatch, tmp_path):
    """P2-5c — there is no tree fallback, so the report path must always resolve a shape:
    with NO catalog it loads the shipped canonical deep-research shape (never None)."""
    captured: dict = {}

    async def fake_generic(query, **kw):
        captured["dr_shape"] = kw.get("dr_shape")
        return "F", [dict(_SRC)], {"growable": True, "layers": [{"gathered": 1}]}
    monkeypatch.setattr(agentic, "_run_generic_research_phase", fake_generic)
    spy, _calls = _spy_write_phase()
    monkeypatch.setattr(agentic, "run_section_write_phase", spy)

    hook = ToolHook(EventPlane(), registry=ToolRegistry())
    result = asyncio.run(run_plan_chain(
        "detailed report", ShapeSelection(shape=None, escalate=False, rationale="r"),
        transport=_ProseTransport(), registry=SpecRegistry(str(tmp_path)),
        hook=hook, plane=hook.plane, timeout=30.0, run_id="nocat",
        overall_goal="detailed report", catalog=None,
    ))
    assert result.deep_research["engine"] == "generic-unroll"
    # a real ShapeSpec was resolved from the shipped shapes dir (not None).
    assert captured["dr_shape"] is not None
    assert getattr(captured["dr_shape"], "name", None) == "deep-research"


# --------------------------------------------------------------------------- #
# 3. CF#1 — the generic research path seeds a TOOL-LESS growable DAG + report-route config
#    (s16/a3 d239/d247: the frozen-unroll path is RETIRED — deep-research is always growable;
#    the grower authors the topology by reasoning. This test pins the seed shape + the
#    report-route grounding config the phase hands the runtime.)
# --------------------------------------------------------------------------- #
def test_generic_research_phase_seeds_growable_research_dag(monkeypatch, tmp_path):
    captured: dict = {}

    class _FakeRuntime:
        async def run(self, dag, **kw):
            captured["dag"] = dag
            return _FakeWriteResult()

    def fake_build_runtime(**kw):
        captured["enable_reactor"] = kw.get("enable_reactor")
        captured["emit_article_notes"] = kw.get("emit_article_notes")
        captured["research_fetch_breadth"] = kw.get("research_fetch_breadth")
        captured["subagent_num_ctx"] = kw.get("subagent_num_ctx")
        captured["grower"] = kw.get("grower")
        return _FakeRuntime(), None

    monkeypatch.setattr(agentic, "_build_acyclic_runtime", fake_build_runtime)

    findings, sources, grow_trace = asyncio.run(_run_generic_research_phase(
        "detailed report on the war",
        transport=_ProseTransport(), registry=SpecRegistry(str(tmp_path)),
        hook=ToolHook(EventPlane(), registry=ToolRegistry()),
        plane=EventPlane(), timeout=30.0, run_id="g",
        overall_goal="detailed report on the war", requested_specs=[],
        dr_shape=_deep_research_shape(), research_depth=2,
    ))
    # The deep-research shape ALWAYS seeds a growable plan now (no frozen unroll, s16/a3).
    assert grow_trace.get("growable") is True

    dag = captured["dag"]
    ids = [n.id for n in dag.nodes]
    # The engine emits a SINGLE TOOL-LESS self-selecting research seed (d242); the grower
    # then authors the real topology by reasoning. No terminal synthesis/verify in the seed.
    assert ids == ["r1_research"], "expected the single tool-less growable research seed"
    assert dag.nodes[0].tool is None, "the seed must be tool-less (self-selects its bundle)"
    assert all(not (i.endswith("_synthesis") or i.endswith("_verify")) for i in ids)
    # The seed task carries the research ROLE framing, NOT the hardcoded research-tree
    # decision/decompose prompt constants (those reach the grower, not the seed node) (CF#1).
    from agent_runtime import research_tree as rt
    joined = "\n".join(n.task for n in dag.nodes)
    assert rt._DECISION_INSTRUCTION not in joined
    assert rt._DECOMPOSE_INSTRUCTION not in joined
    # report-route config reached the runtime: grounding lanes + reactor + pinned breadth.
    assert captured["enable_reactor"] is True
    assert captured["emit_article_notes"] is True
    assert captured["research_fetch_breadth"] == PLAN_CHAIN_TREE_BREADTH
    # the growable seed wires a real grower (the grower authors the topology by reasoning).
    assert captured["grower"] is not None


def test_generic_research_phase_wires_grower_for_growable_shape(monkeypatch, tmp_path):
    """When the deep-research shape declares ``expand_on_gaps``, the generic PHASE-1 unrolls a
    SEED-ONLY growable DAG and WIRES a DagGrower into the runtime so the drive loop reproduces
    the retired tree's iterative breadth. The frozen path stays byte-identical (above)."""
    from agent_runtime import DagGrower

    # No env budget pinned → the served path DERIVES one from the run timeout (assert below).
    monkeypatch.delenv("RA_GROW_WALLCLOCK_BUDGET_S", raising=False)
    captured: dict = {}

    class _FakeRuntime:
        async def run(self, dag, **kw):
            captured["dag"] = dag
            return _FakeWriteResult()

    def fake_build_runtime(**kw):
        captured["grower"] = kw.get("grower")
        return _FakeRuntime(), None

    monkeypatch.setattr(agentic, "_build_acyclic_runtime", fake_build_runtime)

    growable_shape = ShapeSpec(
        name="deep-research", description="d", max_iter=2, hard_cap=4,
        execution="deep-research",
        completeness_stop="Fill ALL the blanks before stopping.",
        expand_on_gaps=True, fan_out=4, max_layers=3,
    )
    findings, sources, grow_trace = asyncio.run(_run_generic_research_phase(
        "detailed report on the war",
        transport=_ProseTransport(), registry=SpecRegistry(str(tmp_path)),
        hook=ToolHook(EventPlane(), registry=ToolRegistry()),
        plane=EventPlane(), timeout=30.0, run_id="grow",
        overall_goal="detailed report on the war", requested_specs=[],
        dr_shape=growable_shape, research_depth=2,
    ))

    # the DAG handed to the runtime is the SEED-ONLY growable plan.
    dag = captured["dag"]
    assert dag.growable is True
    assert [n.id for n in dag.nodes] == ["r1_research"]
    # a real DagGrower was wired, carrying the shape's completeness_stop (reused verbatim).
    grower = captured["grower"]
    assert isinstance(grower, DagGrower)
    assert "Fill ALL the blanks" in (grower.stop_criteria or "")
    # max_layers reflects the shape's 3, clamped to the (depth-overridden) config ceiling.
    assert grower.max_layers == min(3, grower.config.depth)
    # P2-5c FORWARD HARDENING — the served growable loop is WALL-CLOCK bounded by DEFAULT:
    # with no env budget pinned, the grower's config carries a budget derived from the run
    # timeout (~90%), so a full-depth live run is time-bounded (the runtime's _drive_growable
    # turns this into a graceful stop_reason='budget' partial). timeout=30 → 27.0s.
    assert grower.config.grow_wallclock_budget == 30.0 * 0.9
    # the returned trace marks the run growable (the served path reads it for the trace).
    assert grow_trace.get("growable") is True


# --------------------------------------------------------------------------- #
# 4. CF#2 — reactor + framework-review are ALWAYS ON on the served route (flag-free)
# --------------------------------------------------------------------------- #
def test_build_acyclic_runtime_reactor_gated(tmp_path):
    hook = ToolHook(EventPlane(), registry=ToolRegistry())
    reg = SpecRegistry(str(tmp_path))
    rt_off, _ = _build_acyclic_runtime(
        transport=_ProseTransport(), registry=reg, hook=hook,
        plane=hook.plane, shape_spec=None,
    )
    assert rt_off.planner_reactor is None  # default: synchronous heal (byte-compatible)
    rt_on, _ = _build_acyclic_runtime(
        transport=_ProseTransport(), registry=reg, hook=hook,
        plane=hook.plane, shape_spec=None, enable_reactor=True,
    )
    assert rt_on.planner_reactor is not None  # event-driven reactor wired


def test_write_phase_review_retired_and_reactor_always_on(monkeypatch, tmp_path):
    """SF-1 (d310/d311) — the framework REVIEW INJECTION is RETIRED: the served write phase hands
    the runtime the planner's OWN authored DAG with ZERO review nodes (no ``final_review``, no
    ``*_review`` twins) — the model authors the whole document; no engine reviewer edits it. The
    runtime reactor stays ON (unchanged). Proven on the FINAL DAG handed to the runtime."""
    seen: dict = {}

    def fake_planner(**kw):
        class _P:
            async def plan(self, goal, *, prior_memory=None):
                # ``prior_memory`` mirrors the real IncrementalPlanner.plan (SB-3 seam, populated
                # by SB-4's d285 write handoff) — accept + ignore it so this fake stays
                # signature-transparent to the live call.
                node = PlanNode(id="w1", task="write the report body", tool="file_write")
                return type("R", (), {"dag": PlanDAG(nodes=[node], goal=goal)})()
        return _P()

    def fake_runtime(**kw):
        seen["enable_reactor"] = kw.get("enable_reactor")

        class _R:
            chain_sources = None
            async def run(self, dag, **k):
                seen["dag"] = dag
                return _FakeWriteResult()
        return _R(), None

    async def fake_invoke(name, **kw):
        return type("Rb", (), {"ok": False, "value": None})()

    monkeypatch.setattr(agentic, "_build_incremental_planner", fake_planner)
    monkeypatch.setattr(agentic, "_build_acyclic_runtime", fake_runtime)
    # Identity-pass the NORMALIZE pass so the DAG the runtime sees is exactly the planner's
    # authored one (d216/d218 emergent sectioning — no lead+body scaffold is imposed, and SF-1
    # injects no review, so the planner's work node reaches the runtime unchanged). RP-1
    # (d319/d311): the engine coverage/flag DAG passes (_ensure_source_coverage /
    # _flag_unsupported_sections) are RETIRED, so there is nothing left to identity-pass.
    monkeypatch.setattr(agentic, "_normalize_write_dag", lambda dag, out: dag)

    hook = ToolHook(EventPlane(), registry=ToolRegistry())
    monkeypatch.setattr(hook, "invoke", fake_invoke)

    reg = SpecRegistry(str(tmp_path))

    async def drive():
        await run_section_write_phase(
            "q", "out.html", "findings", [],
            transport=_ProseTransport(), registry=reg,
            hook=hook, plane=hook.plane, timeout=5.0, run_id="w",
        )

    monkeypatch.delenv("RA_GENERIC_REPORT_PATH", raising=False)
    asyncio.run(drive())
    # The reactor is on (unchanged), AND the FINAL DAG the runtime ran carries NO framework review
    # node — review injection is retired (SF-1), proven on the served report DAG.
    assert seen["enable_reactor"] is True
    final_ids = [n.id for n in seen["dag"].nodes]
    assert not any(i == "final_review" or i.endswith("_review") for i in final_ids), final_ids


# RP-1 (d319/d311): the engine-derived served-route outline
# (``_outline_from_authored_sections``) is RETIRED — the model authors its own nav/headings —
# so its unit test is removed. The self-policing test below (surgery stays retired) is KEPT and
# now also covers ``collapse_duplicate_sections``.


def test_write_phase_html_surgery_retired():
    # SF-1 (d310/d311, self-policing) — the deterministic assemble_report_spa HTML surgery + the
    # coherence metrics are RETIRED from the served write phase: the engine authors/fixes NOTHING;
    # the MODEL authors the whole document (skeleton + content + nav + Sources). These must NOT
    # come back into run_section_write_phase.
    src = inspect.getsource(run_section_write_phase)
    assert "assemble_report_spa" not in src
    assert "collapse_duplicate_sections" not in src
    assert "collapse_outline_duplicate_sections" not in src
    assert "_coherence_metrics" not in src
    assert "_pre_surgery_path" not in src


# --------------------------------------------------------------------------- #
# 6. CF#4 — the sibling sectioned route ALSO runs the generic engine + carries the stop
# --------------------------------------------------------------------------- #
def test_sibling_route_runs_generic_engine_with_stop(monkeypatch, tmp_path):
    """_run_deep_research_sectioned (the detailed-report sibling route) now runs the GENERIC
    growable engine (run_research_tree retired) and hands the shape completeness_stop to it."""
    captured: dict = {}

    async def fake_generic(query, **kw):
        captured.update(kw)
        return "F", [dict(_SRC)], {"growable": True, "stop_reason": "agent_sufficient",
                                   "grow_layers": 1, "layers": [{"gathered": 3}]}

    async def spy_write(query, out_name, findings, sources, **kw):
        return PlanDAG(nodes=[], goal=query), _FakeWriteResult()

    monkeypatch.setattr(agentic, "_run_generic_research_phase", fake_generic)
    monkeypatch.setattr(agentic, "run_section_write_phase", spy_write)

    reg = SpecRegistry(str(tmp_path))
    shape = _deep_research_shape()
    hook = ToolHook(EventPlane(), registry=ToolRegistry())
    result = asyncio.run(agentic._run_deep_research_sectioned(
        "detailed report", shape,
        ShapeSelection(shape="deep-research", escalate=False, rationale="r"),
        PlanDAG(nodes=[], goal="g"),
        transport=_ProseTransport(), registry=reg, hook=hook, plane=hook.plane,
        timeout=30.0, run_id="sib", effective_max_iter=2,
        overall_goal="detailed report",
    ))
    # routed through the generic engine, with the selected shape AS the unroll source.
    assert captured.get("dr_shape") is shape
    assert captured.get("completeness_stop") == shape.completeness_stop
    assert result.deep_research["engine"] == "generic-unroll"
    assert result.deep_research["stop_reason"] == "agent_sufficient"


# --------------------------------------------------------------------------- #
# 8. the HARD PARITY GATE logic (kept for the parity harness)
# --------------------------------------------------------------------------- #
def test_parity_gate_holds_and_fails():
    class _R:
        def __init__(self, dr):
            self.deep_research = dr

    tree_doc = "<!doctype html><html><h1>War</h1>" + ("<h2>s</h2>x" * 50) + "</html>"
    tree = parity_metrics(_R({"engine": "research-tree", "sources": 5, "rounds_executed": 5}),
                          document=tree_doc)
    # generic with >= tree breadth, single doc, comparable size → HOLDS.
    good_doc = "<!doctype html><html><h1>War</h1>" + ("<h2>s</h2>x" * 48) + "</html>"
    good = parity_metrics(_R({"engine": "generic-unroll", "sources": 6, "rounds_executed": 4}),
                          document=good_doc)
    v_ok = parity_verdict(tree, good)
    assert v_ok["parity_holds"] is True

    # generic with collapsed breadth (1 source) + a dup tail → FAILS.
    bad_doc = "<!doctype html><html><h1>War</h1>x</html><!doctype html><html><h1>War2</h1></html>"
    bad = parity_metrics(_R({"engine": "generic-unroll", "sources": 1, "rounds_executed": 1}),
                         document=bad_doc)
    v_bad = parity_verdict(tree, bad)
    assert v_bad["parity_holds"] is False
    assert v_bad["gates"]["breadth_meets_phase1_bar"] is False
    assert v_bad["gates"]["no_dup_tail"] is False
