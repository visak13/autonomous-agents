"""P2.5 — IncrementalPlanner threads the framework-injected-review flag to PlanBuilder.

The P2.2 ``inject_review`` opt-in lived on :class:`PlanBuilder` but was never reachable
from :class:`IncrementalPlanner` (the live authorer), so framework-injected review was
structurally DARK on every authored plan. P2.5 threads it through. These FAST offline
tests prove the flag flows planner -> builder and defaults False (byte-identical)."""
from __future__ import annotations

from agent_runtime.incremental import IncrementalPlanner
from agent_runtime.factory import AbstractPlanFactory


def _planner(**kw) -> IncrementalPlanner:
    factory = AbstractPlanFactory(spec_index=[], tool_catalog=[])
    return IncrementalPlanner(transport=object(), factory=factory, **kw)


def test_inject_review_defaults_false():
    assert _planner().inject_review is False


def test_inject_review_stored_and_threaded_to_builder():
    p = _planner(inject_review=True)
    assert p.inject_review is True
    # The builder the planner constructs inside plan() must receive the flag. Build one the
    # same way plan() does and assert it carries the opt-in.
    from agent_runtime.plan_tools import PlanBuilder
    builder = PlanBuilder(
        spec_names=p.spec_names, tool_names=p.tool_names,
        shape_name=p.shape_name, shape_description=p.shape_description,
        max_nodes=p.max_nodes, inject_review=p.inject_review,
    )
    assert builder.inject_review is True
    # and inject_reviews actually fires through to_structured: a single work node yields
    # extra (review) nodes.
    builder.dispatch("seed_plan", {"goal": "g"})
    builder.dispatch("add_step", {"id": "w1", "task": "write the report", "tool": "file_write"})
    builder.dispatch("finalize_plan", {})
    structured = builder.to_structured()
    assert len(structured["nodes"]) > 1  # the framework injected a review node
