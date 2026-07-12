"""SB-5 (d285/d289) — the planner's decide_followup REASONS over the reviewer SUMMARY.

The d285 contract closes the agent-in-the-loop: the planner's follow-up decision (add-more-
research vs move-to-write) reasons over the reviewer's ``(summary, memory_index)`` PAIR — the
reviewer's model-emitted overall read of the gathered research, which CARRIES the d237 data-
complexity AS TEXT. The d289 reconciliation: this SUMMARY is the ONE non-divergent signal the
decision reads; the pre-existing structured ``ResearchReviewStatus.data_complexity`` is no longer
consulted SEPARATELY here (it rides INSIDE the summary) — it remains only the offline/back-compat
fallback when no composed summary is present.

These tests leg-probe the REAL :meth:`Planner.decide_followup` in isolation:

  (a) given a (reviewer_summary, memory_index) pair, the prompt the planner reasons over CONTAINS
      the summary text + the index and does NOT carry a separate competing "DATA COMPLEXITY:"
      line (the complexity is folded INTO the summary) — and the reasoned enum is honored;
  (b) with NO summary (the offline FakeTransport seam / a caller without one), the bare structured
      data_complexity is still rendered as the back-compat fallback, and a prose-only transport
      fails open to the caller's safe baseline (no regression);
  (c) self-policing anti-fabrication: decide_followup introduces ZERO new spec-name / role-name
      conditional and ZERO hardcoded complexity heuristic — the decision is the model REASONING
      over the summary DATA (d10-clean), asserted by reading its own source.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import json
import re
import textwrap

from llm_framework import ChatResult
from agent_runtime.factory import AbstractPlanFactory
from agent_runtime.planner import Planner


def _factory() -> AbstractPlanFactory:
    return AbstractPlanFactory([], tool_catalog=[])


class _CapturingDecision:
    """A reasoning transport that CAPTURES the decide prompt and returns a scripted enum."""

    def __init__(self, next_plan: str = "write_plan") -> None:
        self._next = next_plan
        self.system = ""
        self.user = ""

    def chat(self, messages, **opts) -> ChatResult:
        self.system = " ".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "system"
        )
        self.user = " ".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "user"
        )
        return ChatResult(
            role="assistant",
            content=json.dumps(
                {"next_plan": self._next, "rationale": "reasoned over the reviewer summary"}
            ),
        )

    def complete(self, messages, **opts) -> str:  # pragma: no cover - parity shim
        return self.chat(messages, **opts).content


class _ProseOnly:
    """A transport that only ever returns prose — no legal decision (the offline seam)."""

    def chat(self, messages, **opts) -> ChatResult:
        return ChatResult(role="assistant", content="let's just keep going I think")

    def complete(self, messages, **opts) -> str:  # pragma: no cover - parity shim
        return self.chat(messages, **opts).content


_HIGH_SUMMARY = (
    "The research covers the conflict end to end and supports the report. Data complexity: 8 "
    "distinct concerns over a multi-week event — timeline, strikes, casualty and damage "
    "figures, diplomacy, economic impact, oil markets, and outlook; multi-faceted and table-heavy."
)
_SIMPLE_SUMMARY = (
    "The single requested fact is gathered and supports the page. Data complexity: a single "
    "simple finding — one short fact, no sub-parts."
)


# --------------------------------------------------------------------------- #
# (a) the decision reasons over the (summary, index) pair — folded, not competing
# --------------------------------------------------------------------------- #
def test_decide_reasons_over_reviewer_summary_and_index():
    transport = _CapturingDecision("write_plan")
    planner = Planner(transport, _factory())
    decision = asyncio.run(
        planner.decide_followup(
            "detailed HTML report on the conflict",
            last_plan_kind="research",
            reviewer_status="research_complete",
            reviewer_summary=_HIGH_SUMMARY,
            memory_index="research-mem-abc",
            data_complexity="STRUCTURED_FIELD_SHOULD_NOT_APPEAR",
            sources=8,
            default_next="write_plan",
        )
    )
    # the planner reasoned over the reviewer SUMMARY (carrying the data-complexity as text)
    assert "REVIEWER SUMMARY" in transport.user
    assert "8 distinct concerns" in transport.user           # the complexity rides in the summary
    assert "research-mem-abc" in transport.user              # the memory index is named for pull
    # the bare structured field is NOT separately consulted when a summary is present (d289)
    assert "STRUCTURED_FIELD_SHOULD_NOT_APPEAR" not in transport.user
    assert "DATA COMPLEXITY:" not in transport.user          # no competing second signal line
    # the reasoned enum is honored
    assert decision.next_plan == "write_plan"


def test_decide_carries_simple_summary_for_routing():
    """A SIMPLE reviewer summary is the same single signal — it reaches the decision verbatim so
    the model can reason a one-pass write (contrast partner of the high-complexity case)."""
    transport = _CapturingDecision("write_plan")
    planner = Planner(transport, _factory())
    asyncio.run(
        planner.decide_followup(
            "short HTML page with today's gold price",
            last_plan_kind="research",
            reviewer_status="research_complete",
            reviewer_summary=_SIMPLE_SUMMARY,
            memory_index="research-mem-simple",
            default_next="write_plan",
        )
    )
    assert "single simple finding" in transport.user
    assert "DATA COMPLEXITY:" not in transport.user


# --------------------------------------------------------------------------- #
# (b) offline / no-summary fallback — bare complexity rendered, fails open safely
# --------------------------------------------------------------------------- #
def test_decide_falls_back_to_bare_complexity_without_summary():
    transport = _CapturingDecision("done")
    planner = Planner(transport, _factory())
    asyncio.run(
        planner.decide_followup(
            "report on X",
            last_plan_kind="research",
            reviewer_status="research_complete",
            data_complexity="3 concerns; moderate",
            default_next="write_plan",
        )
    )
    # with no composed summary the bare structured complexity is the back-compat fallback line
    assert "DATA COMPLEXITY: 3 concerns; moderate" in transport.user
    assert "REVIEWER SUMMARY" not in transport.user


def test_decide_fails_open_to_safe_baseline_with_summary():
    """A prose-only transport (the offline seam) still fails open to the caller's safe baseline
    even when a reviewer summary is supplied — the loop never spins."""
    planner = Planner(_ProseOnly(), _factory())
    d = asyncio.run(
        planner.decide_followup(
            "detailed HTML report on X",
            last_plan_kind="research",
            reviewer_status="research_complete",
            reviewer_summary=_HIGH_SUMMARY,
            memory_index="m",
            default_next="write_plan",
        )
    )
    assert d.next_plan == "write_plan"


# --------------------------------------------------------------------------- #
# (c) self-policing anti-fabrication: no new spec/role conditional, no heuristic
# --------------------------------------------------------------------------- #
def _code_without_docstring(func) -> str:
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    fn = tree.body[0]
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return "\n".join(ast.unparse(stmt) for stmt in body)


def test_decide_followup_has_no_spec_or_role_conditional_or_heuristic():
    code = _code_without_docstring(Planner.decide_followup)
    lowered = code.lower()
    # no branch keyed on a SPEC or a node ROLE — the decision is role-agnostic, reasoning over data
    for banned in ("spec_id", "spec_name", "specialization", "role ==", "role==", "== role",
                   "if role", '"role"', "'role'"):
        assert banned not in lowered, f"decide must be role/spec-agnostic; found {banned!r}"
    # no HARDCODED complexity heuristic (no "if N sections/complexity > k -> sectioned"); the
    # sectioned-vs-single choice EMERGES downstream in the write planner from the SAME reasoning,
    # never a numeric branch here.
    assert not re.search(r"(complexity|sections?|facets?)\s*[<>]=?\s*\d", lowered), \
        "sectioned-vs-single must EMERGE from reasoning, not a numeric complexity heuristic"
