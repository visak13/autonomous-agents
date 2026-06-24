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
from agent_runtime.factory import PlanDAG


_SRC = {"title": "Iran 2025", "url": "https://news.example.com/iran-2025",
        "markdown": "The conflict escalated June 13; 1,200 casualties by June 20."}


def _deep_research_shape() -> ShapeSpec:
    """A small unrollable deep-research ShapeSpec carrying the P2.4 completeness_stop."""
    return ShapeSpec(
        name="deep-research",
        description="deep research",
        max_iter=2,
        hard_cap=4,
        execution="deep-research",
        round_roles=["research", "critic"],
        final_roles=["research", "synthesis", "verify"],
        completeness_stop="Fill ALL the blanks across timeline/costs/impact before stopping.",
    )


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
# 3. CF#1 — the generic research path is DESCRIPTION-driven (unrolled research-only DAG)
# --------------------------------------------------------------------------- #
def test_generic_research_phase_unrolls_research_only_dag(monkeypatch, tmp_path):
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
    # a non-growable shape yields a non-growable trace (the frozen unroll path).
    assert grow_trace.get("growable") is False

    dag = captured["dag"]
    ids = [n.id for n in dag.nodes]
    # research/critic rounds only — the terminal synthesis/verify are stripped (the write
    # phase replaces synthesis); the DAG IS the research topology (description-driven).
    assert ids, "unroll produced no research nodes"
    assert all(not (i.endswith("_synthesis") or i.endswith("_verify")) for i in ids)
    assert any(i.endswith("_research") for i in ids)
    # The node tasks carry the unrolled ROLE framing (description-driven), NOT the
    # hardcoded research-tree decision/decompose prompt constants (CF#1).
    from agent_runtime import research_tree as rt
    joined = "\n".join(n.task for n in dag.nodes)
    assert rt._DECISION_INSTRUCTION not in joined
    assert rt._DECOMPOSE_INSTRUCTION not in joined
    # report-route config reached the runtime: grounding lanes + reactor + pinned breadth.
    assert captured["enable_reactor"] is True
    assert captured["emit_article_notes"] is True
    assert captured["research_fetch_breadth"] == PLAN_CHAIN_TREE_BREADTH
    # a NON-growable shape wires NO grower (the frozen unroll, byte-identical).
    assert captured["grower"] is None


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
        round_roles=["research", "critic"], final_roles=["research", "synthesis", "verify"],
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


def test_write_phase_review_and_reactor_always_on(monkeypatch, tmp_path):
    """P2-5c — the served write phase ALWAYS builds its planner with inject_review and its
    runtime with the reactor (no flag): both are LIVE by default on the served report route."""
    seen: dict = {}

    def fake_planner(**kw):
        seen["inject_review"] = kw.get("inject_review")

        class _P:
            async def plan(self, goal):
                return type("R", (), {"dag": PlanDAG(nodes=[], goal=goal)})()
        return _P()

    def fake_runtime(**kw):
        seen["enable_reactor"] = kw.get("enable_reactor")

        class _R:
            chain_sources = None
            async def run(self, dag, **k):
                return _FakeWriteResult()
        return _R(), None

    async def fake_invoke(name, **kw):
        return type("Rb", (), {"ok": False, "value": None})()

    monkeypatch.setattr(agentic, "_build_incremental_planner", fake_planner)
    monkeypatch.setattr(agentic, "_build_acyclic_runtime", fake_runtime)
    monkeypatch.setattr(agentic, "_normalize_write_dag", lambda dag, out: dag)
    monkeypatch.setattr(agentic, "_ensure_source_coverage", lambda dag, s: dag)
    monkeypatch.setattr(agentic, "_flag_unsupported_sections", lambda dag: dag)

    hook = ToolHook(EventPlane(), registry=ToolRegistry())
    monkeypatch.setattr(hook, "invoke", fake_invoke)

    reg = SpecRegistry(str(tmp_path))

    async def drive():
        await run_section_write_phase(
            "q", "out.html", "findings", [],
            transport=_ProseTransport(), registry=reg,
            hook=hook, plane=hook.plane, timeout=5.0, run_id="w",
        )

    # No flag/env in play — review + reactor are on regardless of any env value.
    monkeypatch.delenv("RA_GENERIC_REPORT_PATH", raising=False)
    asyncio.run(drive())
    assert seen["inject_review"] is True and seen["enable_reactor"] is True


# --------------------------------------------------------------------------- #
# 5. CF#3 — the duplicate-section-collapse backstops are RETAINED
# --------------------------------------------------------------------------- #
def test_dup_collapse_backstops_retained():
    src = inspect.getsource(run_section_write_phase)
    # The structural fix (anchored insert, P2.3) prevents the dup TAIL but not an in-body
    # re-emitted section; the post-hoc collapse backstops MUST stay (P2-3-review CF#3).
    assert "collapse_duplicate_sections" in src
    assert "collapse_outline_duplicate_sections" in src
    assert "enforce_single_html_document" in src


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
# 7. (c) the framework-injected review renders RAW worker content, not a verdict envelope
# --------------------------------------------------------------------------- #
def test_injected_review_is_raw_worker_not_verdict():
    from agent_runtime.review_injection import inject_reviews
    structured = {"rationale": "r", "nodes": [
        {"id": "w1", "task": "write the report", "tool": "file_write"},
    ]}
    out = inject_reviews(structured)
    reviews = [n for n in out["nodes"] if n["id"] not in {"w1"}]
    assert reviews, "no review node injected"
    for rv in reviews:
        # worker role (or unset → worker), never a judgment/verdict role.
        assert rv.get("role", "worker") in ("worker", "", None)
        task = (rv.get("task") or "").lower()
        # the review must return corrected RAW content, never a verdict/findings envelope.
        assert "verdict" not in task or "not" in task or "raw" in task


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
