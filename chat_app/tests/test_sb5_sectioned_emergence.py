"""SB-5 (d285/d289) — sectioned-vs-single EMERGES from the reviewer summary by REASONING.

The SB-5 reconciliation (d289): there is ONE non-divergent data-complexity signal — the research
reviewer's OVERALL SUMMARY, composed from ``review_research``'s single model emission. The
planner's decide reasons over that summary; the SAME single emission feeds the write-planning
event byte-identically, so the sectioned-vs-single choice the WRITE planner authors emerges from
that one source.

These tests prove, OFFLINE and contrastively, that:

  (a) ONE SOURCE: ``_compose_reviewer_summary`` builds the reviewer summary from the reviewer's
      single emission, and ``_build_write_planning_event`` carries the SAME data-complexity — the
      summary the decide reasons over and the event the write planner reasons over are the one
      non-divergent signal (no second/divergent emission);
  (b) CONTRASTIVE EMERGENCE: driving the REAL write planner with a complexity-REASONING transport,
      a HIGH-complexity reviewer summary (carried into the write goal) routes the planner to a
      SECTIONED write plan (multiple section nodes); a SIMPLE one routes to a SINGLE one-pass node
      — the SAME reasoning logic, the only difference being the complexity from the reviewer
      summary, and the engine (``_normalize_write_dag``) preserves the planner's reasoned section
      count with NO heuristic;
  (c) ANTI-FABRICATION (the neuron watch item): the engine's write-DAG normalization imposes NO
      "if complexity/sections > k -> sectioned" heuristic and NO spec-name/role-name branch — the
      section COUNT is purely the planner's reasoned DAG (d216 emergent sectioning).

Native inference is NOT used — the transport scripts the authoring tool calls, reasoning over the
complexity carried in the prompt. The bounded LIVE confirm is a separate gate artifact.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re

from llm_framework import ChatResult
from reactive_tools import EventPlane, ToolHook, register_agentic_tools
from specialization import SpecRegistry
from specialization.seed import seed_canonical_rulesets

import chat_app.agentic as agentic
from chat_app.agentic import (
    _build_incremental_planner,
    _build_write_planning_event,
    _compose_reviewer_summary,
    _compose_write_goal,
    _normalize_write_dag,
    _WRITE_FILE_SHAPE,
)
from agent_runtime.planner import ResearchReviewStatus


_HIGH_COMPLEXITY = (
    "8 distinct concerns over a multi-week event — timeline, strikes and targets, casualty and "
    "damage figures, diplomacy, economic impact, oil markets, and outlook; multi-faceted and "
    "table-heavy"
)
_SIMPLE_COMPLEXITY = "a single simple finding — one short fact, no sub-parts"


# --------------------------------------------------------------------------- #
# (a) ONE SOURCE — the summary and the write event carry the SAME single emission
# --------------------------------------------------------------------------- #
def test_compose_reviewer_summary_folds_the_single_emission():
    status = ResearchReviewStatus(
        status="research_complete", data_complexity=_HIGH_COMPLEXITY,
        rationale="The research covers the conflict end to end and supports the report.",
    )
    summary = _compose_reviewer_summary(status)
    # the reviewer summary carries BOTH the model rationale AND the data-complexity AS TEXT
    assert "supports the report" in summary
    assert _HIGH_COMPLEXITY in summary
    assert "data complexity" in summary.lower()

    # the SAME single emission feeds the write-planning event (byte-identical field) — so the
    # decide (reads the summary) and the write planner (reads the event) share ONE source.
    event = _build_write_planning_event(
        "findings ...", [{"title": "S1", "url": "u", "markdown": "m"}],
        memory_handle="research-mem", data_complexity=status.data_complexity,
    )
    assert event["data_complexity"] == _HIGH_COMPLEXITY
    assert event["data_complexity"] in summary           # one non-divergent signal


def test_compose_reviewer_summary_empty_when_no_status():
    assert _compose_reviewer_summary(None) == ""


# --------------------------------------------------------------------------- #
# (b) CONTRASTIVE EMERGENCE — high summary -> SECTIONED, simple -> SINGLE
# --------------------------------------------------------------------------- #
class _SectionAuthorer:
    """Drives the write planner's seed->add->finalize protocol, REASONING the section COUNT from
    the data-complexity rendered in the goal (carried in from the reviewer summary): many distinct
    concerns -> multiple section nodes (SECTIONED); a single simple finding -> one node (SINGLE).
    The SAME logic produces both outcomes — only the complexity input differs."""

    def __init__(self) -> None:
        self._queue = None
        self._i = 0
        self.section_count = 0

    def _author_from_complexity(self, prompt: str) -> list[str]:
        text = prompt.lower()
        # REASON the number of sections from the complexity (one part per distinct concern).
        n = 1 if ("single simple finding" in text or "no sub-parts" in text) else 3
        self.section_count = n
        seq = [json.dumps({"tool": "seed_plan", "args": {"shape": "write-file"}})]
        for k in range(n):
            # SB-6/d299: the planner authors TOOL-LESS section workers carrying the writer SPEC
            # (it self-selects 'file' at runtime); the engine no longer stamps tool=file_write.
            # The planner authors the depends_on CHAIN (no engine re-chain).
            seq.append(json.dumps({"tool": "add_step", "args": {
                "task": f"Write section {k + 1} of the report to report.html.",
                "spec": "section-html-writer", "specs": ["section-html-writer"],
                "depends_on": ([f"n{k}"] if k else []),
            }}))
        seq.append(json.dumps({"tool": "finalize_plan", "args": {}}))
        return seq

    def chat(self, messages, **opts) -> ChatResult:
        user = " ".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")
        if self._queue is None:
            self._queue = self._author_from_complexity(user)
            self._i = 0
        if self._i < len(self._queue):
            reply = self._queue[self._i]
            self._i += 1
            return ChatResult(role="assistant", content=reply)
        return ChatResult(role="assistant",
                          content=json.dumps({"tool": "finalize_plan", "args": {}}))

    def complete(self, messages, **opts) -> str:  # pragma: no cover - parity shim
        return self.chat(messages, **opts).content


def _wiring(tmp_path):
    reg = SpecRegistry(str(tmp_path / "specs"))
    seed_canonical_rulesets(reg)
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    return reg, hook


def _author_write_plan(reviewer_summary_complexity: str, *, tmp_path):
    """Build a write goal whose write-planning event carries the reviewer summary's complexity,
    then drive the REAL write planner + the engine normalization; return the normalized DAG."""
    reg, hook = _wiring(tmp_path)
    status = ResearchReviewStatus(
        status="research_complete", data_complexity=reviewer_summary_complexity,
        rationale="The research supports the deliverable.",
    )
    # the reviewer summary the planner reasons over (decide) AND the event field (write) are ONE
    # source — both derived from this single emission.
    reviewer_summary = _compose_reviewer_summary(status)
    assert reviewer_summary_complexity in reviewer_summary
    event = _build_write_planning_event(
        "FINDINGS: the conflict in detail.", [{"title": "S1", "url": "u", "markdown": "m"}],
        memory_handle="research-mem", data_complexity=status.data_complexity,
    )
    write_goal = _compose_write_goal(
        "Write a detailed HTML report on the conflict.", "report.html",
        "FINDINGS: the conflict in detail.", "## S1\nm",
        write_planning_event=event,
    )
    # the complexity from the reviewer summary reached the write planner's goal (the routing input)
    assert reviewer_summary_complexity in write_goal

    transport = _SectionAuthorer()
    planner = _build_incremental_planner(
        transport=transport, registry=reg, hook=hook,
        shape_spec=_WRITE_FILE_SHAPE, allow_web=False, requested_specs=[],
        authoring_directive="",
    )
    w_plan = asyncio.run(planner.plan(write_goal))
    write_dag = _normalize_write_dag(w_plan.dag, "report.html")
    return write_dag, transport


def test_high_complexity_summary_routes_to_sectioned(tmp_path):
    write_dag, transport = _author_write_plan(_HIGH_COMPLEXITY, tmp_path=tmp_path)
    # the planner REASONED multiple sections from the high data-complexity -> SECTIONED
    assert transport.section_count >= 2
    assert len(write_dag.nodes) >= 2
    # SB-6/d299: the write nodes are TOOL-LESS (no engine file_write stamp) — the planner authored
    # the section topology + chain and _normalize_write_dag preserves it, adding only the single-file
    # DELIVERY hint to each task (the runtime routes them to the writer by delivery-context).
    assert all((n.tool or "") == "" for n in write_dag.nodes)
    assert all("report.html" in (n.task or "") for n in write_dag.nodes)
    # the planner authored the depends_on chain (not the engine) — sections 2..N continue section 1.
    assert any(n.depends_on for n in write_dag.nodes)


def test_simple_complexity_summary_routes_to_single(tmp_path):
    write_dag, transport = _author_write_plan(_SIMPLE_COMPLEXITY, tmp_path=tmp_path)
    # the SAME reasoning logic, fed a simple finding, authored ONE pass -> SINGLE
    assert transport.section_count == 1
    assert len(write_dag.nodes) == 1


def test_contrast_is_driven_only_by_the_complexity(tmp_path):
    """The contrast: identical pipeline + identical authoring logic, the ONLY difference being the
    complexity carried in from the reviewer summary, yields SECTIONED vs SINGLE."""
    high_dag, _ = _author_write_plan(_HIGH_COMPLEXITY, tmp_path=tmp_path / "hi")
    simple_dag, _ = _author_write_plan(_SIMPLE_COMPLEXITY, tmp_path=tmp_path / "lo")
    assert len(high_dag.nodes) > len(simple_dag.nodes)


# --------------------------------------------------------------------------- #
# (c) ANTI-FABRICATION (neuron watch item) — the engine imposes NO complexity heuristic
# --------------------------------------------------------------------------- #
def test_normalize_write_dag_imposes_no_complexity_heuristic():
    """The watch item: the engine's write-DAG normalization must NOT decide sectioned-vs-single
    by a heuristic. SB-6/d299: it no longer imposes ANY shape (tool/role/chain stamping retired) —
    it now only adds a single-file DELIVERY hint; the section COUNT + chain stay the planner's
    reasoned DAG (d216 emergent sectioning)."""
    src = inspect.getsource(_normalize_write_dag)
    lowered = src.split('"""')[-1].lower()  # strip the docstring so prose mentions don't trip
    # no numeric complexity/section heuristic
    assert not re.search(r"(complexity|sections?|facets?)\s*[<>]=?\s*\d", lowered)
    # no spec-name / role-name conditional decides the structure
    for banned in ("spec_id", "spec_name", "specialization", "role ==", "role==", "== role"):
        assert banned not in lowered, f"engine must not branch on {banned!r}"
    # the section count is whatever the planner authored — the normalizer iterates the authored
    # nodes (no fabricated sections), confirmed structurally by the contrastive tests above.
    assert "for n in ordered" in lowered or "for n in" in lowered


def test_compose_reviewer_summary_is_pure_plumbing():
    """``_compose_reviewer_summary`` is pure concatenation — no spec/role conditional, no
    heuristic; the model authored the assessment, the engine only renders it."""
    src = inspect.getsource(_compose_reviewer_summary)
    lowered = src.split('"""')[-1].lower()
    for banned in ("spec_id", "spec_name", "specialization", "role ==", "role==",
                   "== role", "if role"):
        assert banned not in lowered
    assert not re.search(r"(complexity|sections?|facets?)\s*[<>]=?\s*\d", lowered)
