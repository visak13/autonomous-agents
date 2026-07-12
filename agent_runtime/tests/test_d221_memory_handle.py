"""d221 — MEMORY-BY-HANDLE: a node is BOUND to a research memory by a stable handle, and
its context names that handle so it READS the research via tools (no verbatim dump).

Covers: the PlanNode field (normalisation + as_dict + parse round-trip) and the
``SubAgent._compose_task`` rendering of the "Binded research memory: <handle>" grounding line.
"""
from __future__ import annotations

from agent_runtime.factory import PlanNode, AbstractPlanFactory
from agent_runtime.runtime import SubAgent
from llm_framework.transport import FakeTransport


def test_plannode_handle_normalised_and_in_as_dict():
    node = PlanNode(id="n1", task="write", research_memory_handle="  research_abc  ")
    assert node.research_memory_handle == "research_abc"        # trimmed
    assert node.as_dict()["research_memory_handle"] == "research_abc"
    # A blank handle normalises to None (never an empty "Binded research memory:" line).
    blank = PlanNode(id="n2", task="t", research_memory_handle="   ")
    assert blank.research_memory_handle is None


def test_handle_round_trips_through_parse_dag():
    structured = {
        "rationale": "r",
        "shape": "linear",
        "nodes": [
            {"id": "n1", "task": "research", "role": "worker", "tool": "web_search",
             "research_memory_handle": "research_xyz"},
        ],
    }
    dag = AbstractPlanFactory([]).parse_dag(structured)
    assert dag.nodes[0].research_memory_handle == "research_xyz"


def test_compose_task_renders_binded_memory_line_when_handle_present():
    node = PlanNode(id="n1", task="Write the impact section.",
                    research_memory_handle="research_abc")
    agent = SubAgent(node, transport=FakeTransport(["x"]))
    user = agent._compose_task({}, tool_value=None)
    assert "Binded research memory: research_abc" in user
    # It directs read-via-tools, not a verbatim dump.
    assert "load_source" in user


def test_compose_task_omits_line_when_no_handle():
    node = PlanNode(id="n1", task="Write the impact section.")
    agent = SubAgent(node, transport=FakeTransport(["x"]))
    user = agent._compose_task({}, tool_value=None)
    assert "Binded research memory" not in user
