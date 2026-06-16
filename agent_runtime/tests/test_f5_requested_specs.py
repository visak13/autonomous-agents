"""F5 — a user-NAMED specialization is honored by the authorer (fully OFFLINE).

The model-driven :class:`~agent_runtime.shape_selector.ShapeSelector` extracts the
specialization(s) the user EXPLICITLY named into ``requested_specs``; the live route
threads them into the incremental authorer. The authorer is TOLD about them (in its
per-node system prompt) AND
:meth:`~agent_runtime.incremental.IncrementalPlanner._apply_requested_specs`
GUARANTEES they are bound: if the 4.6B model forgot to bind any of them, they are
stamped onto the plan's TERMINAL node(s) — exactly the F5(i) failure (a request
naming ``markdown-writer`` previously ran with the named spec NEVER bound).

These tests script a :class:`FakeTransport` with per-node JSON (one structured call
per node), so the whole authoring loop + the requested-spec pass run in-process with
zero inference, and assert the resulting DAG directly. The mechanism is generic +
structural (it keys only off 'is the requested spec bound anywhere?' and the DAG's
sink set) — no per-scenario topic/spec is referenced.
"""
from __future__ import annotations

import asyncio
import json
from typing import Sequence

from agent_runtime.factory import AbstractPlanFactory
from agent_runtime.incremental import IncrementalPlanner
from llm_framework import FakeTransport
from specialization.registry import SpecRegistry
from specialization.seed import DEEP_RESEARCH_SPEC, seed_canonical_rulesets

_MD_SPEC = "markdown-writer"  # seeded by seed_canonical_rulesets alongside research-analyst


def _run(coro):
    return asyncio.run(coro)


_TOOL_CATALOG = [
    {"name": "web_search", "description": "search the web for candidate pages"},
    {"name": "web_fetch", "description": "fetch and extract a page's article text"},
    {"name": "file_write", "description": "write content to a file"},
]


def _node(
    task: str,
    *,
    tool: str = "",
    spec: str = "",
    depends_on: Sequence[str] = (),
    more: bool = True,
) -> str:
    return json.dumps(
        {
            "task": task,
            "spec": spec,
            "specs": [],
            "needs_spec": "",
            "tool": tool,
            "depends_on": list(depends_on),
            "more": more,
        }
    )


def _planner(
    tmp_path,
    replies: Sequence[str],
    *,
    requested_specs: Sequence[str] = (),
    default_research_spec: str = DEEP_RESEARCH_SPEC,
) -> IncrementalPlanner:
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)  # registers research-analyst + markdown-writer
    factory = AbstractPlanFactory(reg.index(), tool_catalog=_TOOL_CATALOG)
    return IncrementalPlanner(
        FakeTransport(list(replies)),
        factory,
        spec_names=reg.names(),
        tool_names=[t["name"] for t in _TOOL_CATALOG],
        shape_name="linear",
        shape_description="a straight A→B chain",
        default_research_spec=default_research_spec,
        requested_specs=requested_specs,
    )


def test_named_spec_forgotten_by_model_is_stamped_on_terminal_node(tmp_path):
    # The model authors a research→write chain but binds the named output spec on
    # NEITHER node (the F5(i) failure mode). The requested-spec pass stamps it onto
    # the TERMINAL node (the writer/sink), so the user-named spec is honored.
    replies = [
        _node("research the history of the Eiffel Tower", tool="web_search"),
        _node("write the overview", depends_on=["n1"], more=False),
    ]
    planner = _planner(tmp_path, replies, requested_specs=[_MD_SPEC])
    dag = _run(planner.plan("overview of the Eiffel Tower using the markdown-writer specialization")).dag

    # the terminal node (nothing depends on n2) now carries the named spec ...
    assert _MD_SPEC in dag.by_id["n2"].effective_specs
    # ... and it is bound SOMEWHERE in the DAG (the F5(i) bar: not 0/N like before).
    assert any(_MD_SPEC in n.effective_specs for n in dag.nodes)


def test_named_spec_bound_by_model_is_honored_not_clobbered(tmp_path):
    # When the model DID bind the requested spec itself, the pass is a no-op — the
    # model's own (correct) binding stands; the named spec is not duplicated or moved.
    replies = [
        _node("research the topic", tool="web_search"),
        _node("write it up", spec=_MD_SPEC, depends_on=["n1"], more=False),
    ]
    planner = _planner(tmp_path, replies, requested_specs=[_MD_SPEC])
    dag = _run(planner.plan("write it up using markdown-writer")).dag
    assert dag.by_id["n2"].effective_specs == (_MD_SPEC,)  # exactly once, where the model put it


def test_no_requested_spec_is_a_no_op_baseline(tmp_path):
    # Contrastive baseline: the SAME unbound plan with NO requested spec leaves the
    # writer node spec-less — proving the binding above comes from the F5 pass, not
    # the authoring loop. (The research node n1 still gets the F2 default; the writer
    # node n2 is a non-gather delivery node, so it stays unbound.)
    replies = [
        _node("research the history of the Eiffel Tower", tool="web_search"),
        _node("write the overview", depends_on=["n1"], more=False),
    ]
    planner = _planner(tmp_path, replies, requested_specs=[])
    dag = _run(planner.plan("overview of the Eiffel Tower")).dag
    assert _MD_SPEC not in dag.by_id["n2"].effective_specs
    assert dag.by_id["n2"].effective_specs == ()


def test_unregistered_requested_spec_is_dropped(tmp_path):
    # An invented name the selector somehow passed is filtered at construction (only
    # registered specs survive) — it can never be stamped onto a node.
    planner = _planner(
        tmp_path,
        [_node("do the thing", more=False)],
        requested_specs=["no-such-spec"],
    )
    assert planner.requested_specs == []
    dag = _run(planner.plan("goal")).dag
    assert dag.by_id["n1"].effective_specs == ()


def test_named_spec_composes_over_existing_default_on_a_gather_sink(tmp_path):
    # If the sole/terminal node is a gather node that the F2 pass would default to
    # research-analyst, a user-named spec still reaches it: the requested-spec pass
    # runs FIRST and composes onto the sink, and F2 then leaves the now-specced node
    # alone (no clobber in either direction).
    replies = [_node("research and report on AI", tool="web_search", more=False)]
    planner = _planner(tmp_path, replies, requested_specs=[_MD_SPEC])
    dag = _run(planner.plan("research AI and write it using markdown-writer")).dag
    specs = dag.by_id["n1"].effective_specs
    assert _MD_SPEC in specs  # the user-named spec reached the single sink node
