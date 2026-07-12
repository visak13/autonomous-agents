"""s9/c13 — deep-research → SHARED per-section bounded-SPA write phase (d55/d56/d57).

These tests cover the deterministic, transport-free seams of that wiring:

* ``_research_only_dag`` — the unrolled deep-research DAG minus its terminal
  synthesis/verify (the per-section write phase replaces synthesis), still valid;
* ``_collect_chain_sources`` — the run's global fetched-source list (the write
  planner's source catalog).

s15 ROUTING PURITY (d148/d151): the old ``_is_report_deliverable`` routing-gate
regex is RETIRED — the deep-research shape now routes by the LLM-SELECTED shape
(``run_agentic`` → :func:`run_plan_chain`), with no query-content gate on the served
path. The sectioned-vs-inline sub-branch on the test-only ``_run_deep_research``
sibling is an explicit ``sectioned`` parameter now, not a query sniff, so there is no
content-gate left to unit-test here.
"""
from agent_runtime.factory import PlanDAG, PlanNode
from chat_app.agentic import (
    _collect_chain_sources,
    _research_only_dag,
)


# --------------------------------------------------------------------------- #
# _research_only_dag — drops terminal synthesis/verify, keeps research, stays valid
# --------------------------------------------------------------------------- #
def test_research_only_dag_drops_synthesis_and_verify_and_validates():
    # a 2-round unrolled deep-research shape (growing-visibility edges)
    nodes = [
        PlanNode(id="r1_research", task="research", depends_on=()),
        PlanNode(id="r1_critic", task="critic", depends_on=("r1_research",)),
        PlanNode(id="r2_research", task="research", depends_on=("r1_research", "r1_critic")),
        PlanNode(id="r2_synthesis", task="synthesize",
                 depends_on=("r1_research", "r1_critic", "r2_research"), role="synthesizer"),
        PlanNode(id="r2_verify", task="verify",
                 depends_on=("r1_research", "r1_critic", "r2_research", "r2_synthesis")),
    ]
    dag = PlanDAG(nodes=nodes, rationale="r", goal="g")
    research = _research_only_dag(dag)  # constructs + validates a fresh PlanDAG
    kept = {n.id for n in research.nodes}
    assert kept == {"r1_research", "r1_critic", "r2_research"}
    # no remaining node references a dropped terminal node
    for n in research.nodes:
        assert all(d in kept for d in n.depends_on)
    assert research.goal == "g"


# --------------------------------------------------------------------------- #
# _collect_chain_sources — the run's global fetched-source list, deduped by URL
# --------------------------------------------------------------------------- #
class _R:
    def __init__(self, tool_value):
        self.tool_value = tool_value


class _Result:
    def __init__(self, results, order):
        self.results = results
        self.launch_order = order


def test_collect_chain_sources_walks_research_tool_values_in_launch_order():
    results = {
        "r1_research": _R({"fetched": [{"title": "BBC", "url": "https://bbc.com/a", "markdown": "x"}]}),
        "r2_research": _R({"fetched": [
            {"title": "BBC", "url": "https://bbc.com/a", "markdown": "x"},  # dup → dropped
            {"title": "CFR", "url": "https://cfr.org/b", "markdown": "y"},
        ]}),
        "r1_critic": _R(None),  # no tool value → contributes nothing
    }
    res = _Result(results, ["r1_research", "r1_critic", "r2_research"])
    sources = _collect_chain_sources(res)
    assert [s["url"] for s in sources] == ["https://bbc.com/a", "https://cfr.org/b"]
