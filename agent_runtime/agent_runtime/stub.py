"""Deterministic phi-transport stubs — exercise the harness with zero GPU (d7/d8).

The phi transport is PLUGGABLE (the llm_framework ``Transport`` protocol): the
live :class:`~llm_framework.transport.OllamaTransport` is used when the shared
GPU frees, and these deterministic stubs drive the WHOLE harness offline in the
meantime (d8/a1). Each stub is just a scripted
:class:`~llm_framework.transport.FakeTransport` returning canned-but-realistic
phi outputs, so the planner, runtime, and self-heal logic are all fully
exercised without live inference.

Three canned scenarios cover the Stage-A acceptance surface:

- :func:`valid_plan_transport` — phi emits a VALID DAG JSON (a real
  research → write-md → write-html shape) so the planner produces a runnable
  plan and the runtime launches+tracks it.
- :func:`malformed_then_valid_plan_transport` — phi first emits MALFORMED JSON,
  then valid JSON, exercising the structured-output repair loop (malformed-JSON
  self-heal).
- :func:`subagent_transport` — canned sub-agent replies (one per node) so a full
  DAG run produces realistic node outputs offline.

A live run swaps in ``OllamaTransport(base_url="http://localhost:11435",
model="phi4-mini")`` with no other code change — that is the whole point of the
pluggable seam.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from llm_framework import FakeTransport


# A realistic DAG for the canonical demonstrating goal (research → write two
# ways). It is *canned*, standing in for what phi would emit — the planner does
# NOT hard-code this; the stub does, purely so the offline harness has something
# DAG-shaped to parse and run (the live phi will author its own).
def canned_plan(
    topic: str = "the chosen topic",
    *,
    md_spec: str = "markdown-writer",
    html_spec: str = "html-writer",
    research_spec: str = "web-researcher",
) -> dict[str, Any]:
    """A valid research → write-md → write-html DAG as a plain dict."""
    return {
        "rationale": "research once, then render the same findings two ways in parallel",
        "nodes": [
            {
                "id": "n1",
                "task": f"research {topic} and gather key facts with sources",
                "spec": research_spec,
                "depends_on": [],
                "tool": "web_search",
                "tool_args": {"query": topic},
            },
            {
                "id": "n2",
                "task": f"write a detailed markdown report on {topic}",
                "spec": md_spec,
                "depends_on": ["n1"],
            },
            {
                "id": "n3",
                "task": f"write a detailed HTML report on {topic}",
                "spec": html_spec,
                "depends_on": ["n1"],
            },
        ],
    }


def canned_replan(
    task: str = "recovered step",
    *,
    spec: Optional[str] = None,
    tool: Optional[str] = None,
) -> dict[str, Any]:
    """A MINIMAL corrective sub-graph (one node) the planner re-derives for a
    failed step. Stands in for what live phi would emit on a re-plan call — the
    runtime does NOT hard-code it; the stub does, purely so the offline harness
    has a corrective DAG to parse and run for the sub-graph-self-heal path."""
    node: dict[str, Any] = {"id": "r1", "task": task, "spec": spec, "depends_on": []}
    if tool is not None:
        node["tool"] = tool
    return {"rationale": f"re-derive a corrective approach for: {task}", "nodes": [node]}


def valid_plan_transport(plan: Optional[Mapping[str, Any]] = None, **kw: Any) -> FakeTransport:
    """A transport whose first reply is a VALID DAG JSON (steady-state)."""
    payload = json.dumps(dict(plan) if plan is not None else canned_plan(**kw))
    return FakeTransport([payload])


def malformed_then_valid_plan_transport(
    plan: Optional[Mapping[str, Any]] = None, **kw: Any
) -> FakeTransport:
    """A transport that emits MALFORMED JSON first, then a VALID DAG JSON.

    Exercises the structured-output bounded repair loop: the planner's first
    parse fails, the repair re-prompt gets the valid plan. The malformed reply
    is a truncated object (an unbalanced brace) — exactly the failure the
    extractor classifies as malformed."""
    good = json.dumps(dict(plan) if plan is not None else canned_plan(**kw))
    malformed = '{"rationale": "oops", "nodes": [ {"id": "n1", "task": '  # truncated
    return FakeTransport([malformed, good])


def subagent_transport(replies: Optional[Sequence[str]] = None) -> FakeTransport:
    """Canned sub-agent replies (one consumed per node phi call).

    Defaults to three generic "report" bodies so a research→md→html DAG yields
    realistic node outputs. The transport repeats its last reply once exhausted,
    so an unexpected extra call still returns something sensible (non-strict)."""
    if replies is None:
        replies = [
            "Key facts gathered: point A, point B, point C (with sources).",
            "# Report\n\nA detailed markdown report covering A, B, and C.",
            "<h1>Report</h1><p>A detailed HTML report covering A, B, and C.</p>",
        ]
    return FakeTransport(list(replies))


def failing_tool_then_ok_subagent_transport(
    replies: Optional[Sequence[str]] = None,
) -> FakeTransport:
    """Sub-agent replies for the tool-error self-heal path.

    The tool failure happens in the TOOL, not the transport; this just supplies
    the phi reply for the successful re-launched attempt. One reply is enough
    (steady-state repeat), but a list lets a test script exact-count calls."""
    return subagent_transport(replies or ["Recovered: produced the step output after re-plan."])


__all__ = [
    "canned_plan",
    "canned_replan",
    "valid_plan_transport",
    "malformed_then_valid_plan_transport",
    "subagent_transport",
    "failing_tool_then_ok_subagent_transport",
]
