"""F2 — default research spec on null-spec GATHER nodes (fully OFFLINE, no GPU).

The incremental authorer decides each node's ``spec`` in a SEPARATE per-node call,
so identical sibling gather sub-tasks diverge: one news node gets
``research-analyst`` while its siblings get nothing (the live a1 trace: 2 of 3
parallel news nodes bound to no spec). A spec-less gather node has no ruleset, so
it degrades to the d13 source-list-summary anti-pattern — the empty/thin emailed
news section. :meth:`IncrementalPlanner._apply_default_research_spec` closes that
gap STRUCTURALLY: every null-spec node that fires a research/gather tool
(``web_search``/``web_fetch``) is stamped with the generic research spec, so the
parallel siblings are sibling-consistent and all carry the grounded ruleset.

These tests script a :class:`FakeTransport` with per-node JSON (the authorer makes
one structured call per node), so the WHOLE authoring loop + the default-spec pass
run in-process with zero inference. They assert the resulting DAG directly.

The rule is generic + role-structural (no per-scenario topic/spec/filename):

* a null-spec GATHER node (tool in ``web_search``/``web_fetch``) -> stamped.
* a gather node that ALREADY has a spec -> unchanged (no override).
* a DELIVERY node (send_mail/file_write) -> left unbound (its content is
  upstream-grounded, d11; the research ruleset does not apply).
* a node that declared ``needs_spec`` -> left unbound (the missing-specialist
  hatch must still fire; the default never masks a declared missing capability).
* default unset / not a registered spec -> the pass is a no-op.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence

from agent_runtime.factory import AbstractPlanFactory
from agent_runtime.incremental import IncrementalPlanner
from llm_framework import FakeTransport
from specialization.registry import SpecRegistry
from specialization.seed import DEEP_RESEARCH_SPEC, seed_canonical_rulesets


def _run(coro):
    return asyncio.run(coro)


_TOOL_CATALOG = [
    {"name": "web_search", "description": "search the web for candidate pages"},
    {"name": "web_fetch", "description": "fetch and extract a page's article text"},
    {"name": "send_mail", "description": "email the result to the user"},
    {"name": "file_write", "description": "write content to a file"},
]


def _node(
    task: str,
    *,
    tool: str = "",
    spec: str = "",
    needs_spec: str = "",
    depends_on: Sequence[str] = (),
    more: bool = True,
) -> str:
    """One per-node structured reply, as the authorer's schema expects it."""
    return json.dumps(
        {
            "task": task,
            "spec": spec,
            "specs": [],
            "needs_spec": needs_spec,
            "tool": tool,
            "depends_on": list(depends_on),
            "more": more,
        }
    )


def _planner(
    tmp_path,
    replies: Sequence[str],
    *,
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
        shape_name="modular-parallel",
        shape_description="independent gather steps then one combine/deliver step",
        default_research_spec=default_research_spec,
    )


# A mixed parallel-gather plan that exercises EVERY branch of the F2 pass in one DAG.
def _mixed_plan_replies() -> list[str]:
    return [
        # n1: null-spec gather (web_search) -> SHOULD be defaulted.
        _node("Search the latest news on climate change", tool="web_search"),
        # n2: gather already bound to research-analyst -> unchanged (no override).
        _node("Search the latest news on AI", tool="web_search", spec="research-analyst"),
        # n3: null-spec gather via web_fetch -> SHOULD be defaulted.
        _node("Fetch the latest space-exploration coverage", tool="web_fetch"),
        # n4: null-spec but declares a MISSING specialist -> left unbound (hatch).
        _node(
            "Score the market sentiment of the AI coverage",
            tool="web_search",
            needs_spec="a finance-sentiment specialist",
        ),
        # n5: delivery node (send_mail) -> left unbound (content is upstream-grounded).
        _node(
            "Combine the gathered news and email a single summary",
            tool="send_mail",
            depends_on=["n1", "n2", "n3", "n4"],
            more=False,
        ),
    ]


def test_null_spec_gather_siblings_get_default_research_spec(tmp_path):
    planner = _planner(tmp_path, _mixed_plan_replies())
    dag = _run(planner.plan("news on climate change, AI, space exploration; email me")).dag
    by = dag.by_id

    # The two null-spec gather siblings now carry the SAME grounded research ruleset.
    assert by["n1"].effective_specs == (DEEP_RESEARCH_SPEC,)   # defaulted (web_search)
    assert by["n3"].effective_specs == (DEEP_RESEARCH_SPEC,)   # defaulted (web_fetch)
    # The already-bound gather node is untouched (idempotent, no double-binding).
    assert by["n2"].effective_specs == (DEEP_RESEARCH_SPEC,)

    # Sibling-consistency: every PARALLEL gather node (depends_on=[]) that gathers
    # now has a research spec — the empty-section failure cannot recur.
    gather_siblings = [
        n for n in dag.nodes
        if not n.depends_on and (n.tool in ("web_search", "web_fetch")) and not n.needs_spec
    ]
    assert gather_siblings, "expected parallel gather siblings in the authored DAG"
    assert all(n.effective_specs == (DEEP_RESEARCH_SPEC,) for n in gather_siblings)


def test_missing_specialist_node_is_not_masked_by_default(tmp_path):
    planner = _planner(tmp_path, _mixed_plan_replies())
    dag = _run(planner.plan("goal")).dag
    n4 = dag.by_id["n4"]
    # The default never overwrites a declared missing-specialist node: it stays
    # spec-less so detect_missing_specialists still pauses the run for the user.
    assert n4.effective_specs == ()
    assert n4.needs_spec == "a finance-sentiment specialist"


def test_delivery_node_is_left_unbound(tmp_path):
    planner = _planner(tmp_path, _mixed_plan_replies())
    dag = _run(planner.plan("goal")).dag
    n5 = dag.by_id["n5"]
    # A send_mail delivery node is null-spec BY DESIGN (its content is grounded in
    # upstream node text, d11) — the research ruleset must not be stamped onto it.
    assert n5.tool == "send_mail"
    assert n5.effective_specs == ()


def test_no_default_when_spec_unset_proves_the_pass_is_the_cause(tmp_path):
    # Contrastive baseline: with NO default configured, the same null-spec gather
    # nodes stay spec-less — proving the binding above comes from the F2 pass and
    # is not something the authoring loop did on its own.
    planner = _planner(tmp_path, _mixed_plan_replies(), default_research_spec="")
    dag = _run(planner.plan("goal")).dag
    assert dag.by_id["n1"].effective_specs == ()
    assert dag.by_id["n3"].effective_specs == ()
    assert dag.by_id["n2"].effective_specs == (DEEP_RESEARCH_SPEC,)  # model-chosen, kept


def test_unregistered_default_is_a_no_op(tmp_path):
    # A default name that is not a registered specialization is ignored (never
    # stamps an unresolvable spec name onto a node).
    planner = _planner(tmp_path, _mixed_plan_replies(), default_research_spec="no-such-spec")
    dag = _run(planner.plan("goal")).dag
    assert dag.by_id["n1"].effective_specs == ()
    assert dag.by_id["n3"].effective_specs == ()
