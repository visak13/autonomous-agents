"""s9/c13 — deep-research → SHARED per-section bounded-SPA write phase (d55/d56/d57).

The acceptance scenario ("detailed HTML report on the 2025 US-Iran conflict … cite
sources") routes to ``_run_deep_research`` (``wants_file`` extracted False), so the
SWA fix MUST land on that path. These tests cover the deterministic, transport-free
seams of that wiring:

* the ROUTING gate (``_is_report_deliverable``) — a detailed REPORT deliverable is
  sectioned; a bare 'research X in depth' inline answer and the headlines path are
  NOT (no regression);
* ``_research_only_dag`` — the unrolled deep-research DAG minus its terminal
  synthesis/verify (the per-section write phase replaces synthesis), still valid;
* ``_collect_chain_sources`` — the run's global fetched-source list (the write
  planner's source catalog).
"""
from dataclasses import dataclass

from agent_runtime.factory import PlanDAG, PlanNode
from chat_app.agentic import (
    _collect_chain_sources,
    _is_report_deliverable,
    _research_only_dag,
)


@dataclass
class _Sel:
    wants_file: bool = False
    multi_page: bool = False


# --------------------------------------------------------------------------- #
# routing gate — detailed REPORT deliverable vs inline research / headlines
# --------------------------------------------------------------------------- #
def test_report_deliverable_gate_fires_on_a_detailed_report_request():
    # the acceptance prompt names an HTML report → sectioned (even with wants_file False)
    assert _is_report_deliverable(
        "Write a detailed HTML report on the 2025 US-Iran conflict and cite sources",
        _Sel(wants_file=False),
    )
    # a markdown/document deliverable also qualifies
    assert _is_report_deliverable("a thorough markdown document on solar power", _Sel())
    # the model's own file/multi-page intent qualifies regardless of wording
    assert _is_report_deliverable("an exhaustive write-up", _Sel(multi_page=True))


def test_report_deliverable_gate_excludes_inline_research_and_headlines():
    # bare 'research X in depth' with NO report/file cue → inline answer (NOT sectioned)
    assert not _is_report_deliverable("research the topic in depth", _Sel())
    # the headlines path is not even detailed; but it also carries no report cue
    assert not _is_report_deliverable("top news headlines for today", _Sel())


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
