"""as1 (FORK1, d214/d215) — Planner.decide_followup: the PLANNER reasons the next plan.

The iterative planner loop's follow-up decision is the model's own reasoning over the last
plan's reviewer status (NOT a hardcoded rule), with a FAIL-SAFE to the caller's safe baseline
so a non-reasoning transport (the offline FakeTransport seam) / a malformed reply never spins
the loop. These tests pin both halves: a reasoning transport's legal enum is honored; a
prose/blind transport falls back to ``default_next``."""
from __future__ import annotations

import asyncio
import json

from llm_framework import ChatResult
from agent_runtime.factory import AbstractPlanFactory
from agent_runtime.planner import Planner, FOLLOWUP_PLANS, ResearchReviewStatus


def _factory() -> AbstractPlanFactory:
    return AbstractPlanFactory([], tool_catalog=[])


class _ScriptedDecision:
    """A transport whose native chat returns a scripted JSON follow-up decision."""

    def __init__(self, next_plan: str) -> None:
        self._next = next_plan

    def chat(self, messages, **opts) -> ChatResult:
        return ChatResult(
            role="assistant",
            content=json.dumps({"next_plan": self._next, "rationale": "scripted"}),
        )

    def complete(self, messages, **opts) -> str:  # pragma: no cover - parity shim
        return self.chat(messages, **opts).content


class _ProseOnly:
    """A transport that only ever returns prose — no legal decision (the offline seam)."""

    def chat(self, messages, **opts) -> ChatResult:
        return ChatResult(role="assistant", content="I think we should keep going, maybe.")

    def complete(self, messages, **opts) -> str:  # pragma: no cover - parity shim
        return self.chat(messages, **opts).content


def test_decide_followup_honors_reasoned_enum():
    """A reasoning transport's legal enum decision is honored (the PLANNER reasons)."""
    for choice in FOLLOWUP_PLANS:
        planner = Planner(_ScriptedDecision(choice), _factory())
        decision = asyncio.run(
            planner.decide_followup(
                "write a report on X",
                last_plan_kind="research",
                reviewer_status="research_complete",
                findings_digest="found a, b, c",
                default_next="done",
            )
        )
        assert decision.next_plan == choice
        assert decision.done == (choice == "done")


def test_decide_followup_fails_open_to_safe_baseline():
    """A prose-only / schema-blind transport falls back to the caller's safe baseline so the
    loop always makes safe forward progress (research->write on the report route) and the
    offline seam stays green — never spins on an illegal reply."""
    planner = Planner(_ProseOnly(), _factory())
    # report route after research: safe baseline = write_plan (deliverable is the report)
    d1 = asyncio.run(
        planner.decide_followup(
            "detailed HTML report on X",
            last_plan_kind="research",
            reviewer_status="research_complete",
            default_next="write_plan",
        )
    )
    assert d1.next_plan == "write_plan"
    # after a write plan: safe baseline = done (the loop terminates)
    d2 = asyncio.run(
        planner.decide_followup(
            "detailed HTML report on X",
            last_plan_kind="write",
            reviewer_status="deliverable_complete",
            default_next="done",
        )
    )
    assert d2.next_plan == "done" and d2.done


class _ScriptedReview:
    """A transport returning a scripted research-review JSON status."""

    def __init__(self, status: str, complexity: str = "3 concerns; moderate") -> None:
        self._status = status
        self._complexity = complexity

    def chat(self, messages, **opts) -> ChatResult:
        return ChatResult(
            role="assistant",
            content=json.dumps(
                {"status": self._status, "data_complexity": self._complexity,
                 "rationale": "scripted"}
            ),
        )

    def complete(self, messages, **opts) -> str:  # pragma: no cover - parity shim
        return self.chat(messages, **opts).content


def test_review_research_honors_reasoned_status():
    """A reasoning transport's legal research-review status + data complexity are honored."""
    planner = Planner(_ScriptedReview("research_complete", "8 points; complex"), _factory())
    status = asyncio.run(
        planner.review_research("report on X", "found a, b, c", sources=8)
    )
    assert isinstance(status, ResearchReviewStatus)
    assert status.status == "research_complete" and status.complete
    assert "8 points" in status.data_complexity


def test_review_research_fails_open_to_derived():
    """A prose-only transport derives the status from what was gathered (complete with sources,
    thin without) so the served route + offline tests stay green."""
    planner = Planner(_ProseOnly(), _factory())
    with_src = asyncio.run(planner.review_research("report on X", "found facts", sources=5))
    assert with_src.status == "research_complete"
    no_src = asyncio.run(planner.review_research("report on X", "", sources=0))
    assert no_src.status == "research_thin"


class _ProseSummary:
    """A transport returning a scripted prose summary (the live synthesizer path)."""

    def __init__(self, text: str) -> None:
        self._text = text

    def chat(self, messages, **opts) -> ChatResult:
        return ChatResult(role="assistant", content=self._text)

    def complete(self, messages, **opts) -> str:  # pragma: no cover - parity shim
        return self._text


class _EmptyTransport:
    """A transport returning empty content (forces the derived fallback)."""

    def chat(self, messages, **opts) -> ChatResult:
        return ChatResult(role="assistant", content="")

    def complete(self, messages, **opts) -> str:  # pragma: no cover - parity shim
        return ""


def test_finalize_summary_uses_llm_prose():
    """The terminal synthesizer summary is the LLM's real prose digest, not a fixed string."""
    planner = Planner(_ProseSummary("Your report on the 2025 conflict is ready, covering the "
                                    "timeline and casualty figures."), _factory())
    summary = asyncio.run(
        planner.finalize_summary("report on the 2025 conflict", plans_authored=["research", "write"],
                                 sources=8, sections=4, artifact="report.html")
    )
    assert "ready" in summary.lower() and "fixed" not in summary
    assert summary.strip() != ""


def test_finalize_summary_fails_open_to_derived():
    """An empty reply (offline seam) yields a minimal factual derived one-liner, never a crash."""
    planner = Planner(_EmptyTransport(), _factory())
    summary = asyncio.run(
        planner.finalize_summary("report on X", plans_authored=["research", "write"],
                                 sources=3, sections=2, artifact="out.html")
    )
    assert "report on X" in summary and "out.html" in summary
