"""WORKER CONTEXT-ASSEMBLY ARCHITECTURE (d38/d39, s8/b3) — the OVERALL GOAL reaches
every worker node, and a downstream writer node's prompt carries the verbatim goal
PLUS the full dependency-scoped upstream research/sources.

The probe (d38) found the central gap: today only the planner sees the goal; the
per-node task it emits is a PARAPHRASE, so a downstream node works toward a lossy
restatement and never the real objective. A Gemma node cannot DISCOVER the goal
(no file/grep access like an eda-base3/Claude-Code worker), so it must be
CONSTRUCTED and fed. These tests lock that the goal — carried on ``PlanDAG.goal`` —
flows through ``AgentRuntime.run`` into EVERY node's user turn, and that
``_compose_task`` assembles goal -> prior conversation -> CURRENT TASK -> inputs ->
dependency-scoped sources, dependency-scoped and NON-transitive (direct deps only,
d17). Fully OFFLINE: FakeTransport, no Ollama / network / GPU.

It also locks the BACK-COMPAT contract (no goal + no context => byte-identical to
the pre-d39 user turn) and the d17/o4 budget FINALIZATION against num_ctx 32768
(upstream prose clipped to 4000 chars; the goal fed VERBATIM, never clipped).
"""
from __future__ import annotations

import asyncio

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.runtime import _OVERALL_GOAL_HEADER, AgentRuntime, SubAgent
from agent_runtime.status import NodeStatus
from llm_framework import FakeTransport
from reactive_tools import EventPlane, ToolHook


GOAL = "Create a detailed HTML report on the June 2026 US-Iran situation."


def _user_turn(transport: FakeTransport, *, contains: str) -> str:
    """The user turn of the produce call whose task text contains ``contains``."""
    for call in transport.calls:
        user = next(
            (m["content"] for m in call["messages"] if m["role"] == "user"), ""
        )
        if contains in user:
            return user
    raise AssertionError(f"no produce call whose user turn contains {contains!r}")


# --------------------------------------------------------------------------- #
# 1) END-TO-END: dag.goal flows through AgentRuntime into the node user turn.
# --------------------------------------------------------------------------- #
def test_overall_goal_reaches_node_user_turn_verbatim():
    transport = FakeTransport(["the answer"])
    node = PlanNode(id="n1", task="Write the report section.")
    dag = PlanDAG(nodes=[node], goal=GOAL)

    result = asyncio.run(AgentRuntime(transport=transport).run(dag))
    assert result.states["n1"]["status"] == NodeStatus.DONE.value

    user = _user_turn(transport, contains="Write the report section.")
    # The verbatim goal is present, under its clearly-delimited header, and LEADS
    # the user turn (before the node's own paraphrased task).
    assert _OVERALL_GOAL_HEADER in user
    assert GOAL in user                       # VERBATIM, not paraphrased
    assert "CURRENT TASK:" in user
    assert user.index(GOAL) < user.index("Write the report section.")


# --------------------------------------------------------------------------- #
# 2) THE DELIVERABLE: a downstream WRITER node's prompt contains the verbatim goal
#    + the FULL upstream research prose + the upstream SOURCES (tool value).
# --------------------------------------------------------------------------- #
def test_downstream_writer_prompt_contains_goal_and_upstream_sources():
    plane = EventPlane()
    hook = ToolHook(plane)

    research_prose = (
        "Iran and the US exchanged strikes in June 2026; casualties reported on "
        "both sides and damage to oil infrastructure. " * 4
    )
    source_md = (
        "On 13 June 2026 the headline figures were 24 dead and 1,200 displaced, "
        "per the wire services; the Abadan refinery sustained major damage."
    )

    def web_search(**kwargs):
        return {
            "results": [{"title": "Wire report", "url": "https://example.org/a",
                         "snippet": "casualties and damage"}],
            "count": 1,
        }

    def web_fetch(**kwargs):
        # The agentic research loop (s9/c5) collects the EXTRACTED article text from a
        # MODEL-chosen web_fetch into the node's tool_value, which the runtime then
        # renders as SOURCES & FINDINGS into the downstream writer's turn (d17).
        return {"url": kwargs.get("url"), "final_url": kwargs.get("url"),
                "title": "Wire report", "markdown": source_md, "extracted": True}

    hook.register("web_search", web_search, description="search the web")
    hook.register("web_fetch", web_fetch, description="fetch a URL")

    # n1 RESEARCH (a web_search node = a TRUE AGENT: it searches, fetches a source IT
    # chose, then writes findings — producing prose + a tool_value carrying sources);
    # n2 WRITER depends on n1 (no tool of its own — it must be FED the research).
    n1 = PlanNode(id="n1", task="Research the situation.", tool="web_search")
    n2 = PlanNode(id="n2", task="Write the final HTML report.", depends_on=["n1"])
    dag = PlanDAG(nodes=[n1, n2], goal=GOAL)

    # n1's agentic loop: search → fetch the chosen URL → write findings (3 turns);
    # then n2's single writer call. FakeTransport replays these in order.
    transport = FakeTransport([
        '{"tool": "web_search", "args": {"query": "US Iran June 2026 casualties"}}',
        '{"tool": "web_fetch", "args": {"url": "https://example.org/a"}}',
        "RESEARCH: " + research_prose,
        "<html>the report</html>",
    ])
    runtime = AgentRuntime(transport=transport, hook=hook)
    result = asyncio.run(runtime.run(dag))
    assert result.states["n2"]["status"] == NodeStatus.DONE.value

    writer_user = _user_turn(transport, contains="Write the final HTML report.")

    # (a) the VERBATIM overall goal is fed to the downstream writer.
    assert _OVERALL_GOAL_HEADER in writer_user
    assert GOAL in writer_user
    # (b) the FULL upstream research PROSE (n1's output) is folded in as an input.
    assert "INPUTS FROM PRIOR STEPS:" in writer_user
    assert "casualties reported on both sides" in writer_user
    # (c) the upstream SOURCES (n1's tool value) reach the writer under the
    #     dependency-scoped findings header — the writer SEES the real sources,
    #     not just clipped prose (the o4 thin/empty-report root cause).
    assert "SOURCES & FINDINGS FROM PRIOR STEP n1" in writer_user
    assert source_md in writer_user
    # (d) order: goal leads, then the node task, then inputs, then sources.
    assert writer_user.index(GOAL) < writer_user.index("Write the final HTML report.")
    assert (
        writer_user.index("INPUTS FROM PRIOR STEPS:")
        < writer_user.index("SOURCES & FINDINGS FROM PRIOR STEP n1")
    )


# --------------------------------------------------------------------------- #
# 3) BACK-COMPAT: no goal + no conversation context => byte-identical user turn.
# --------------------------------------------------------------------------- #
def test_backcompat_no_goal_no_context_user_turn_is_just_the_task():
    node = PlanNode(id="n1", task="Summarise the input.")
    agent = SubAgent(node, transport=FakeTransport(["x"]))
    # No goal, no context, no inputs -> the user turn is EXACTLY the task (pre-d39).
    assert agent._compose_task({}, tool_value=None) == "Summarise the input."

    # And via the runtime end to end (goal-less DAG): the user turn carries no
    # OVERALL GOAL header.
    transport = FakeTransport(["ok"])
    asyncio.run(AgentRuntime(transport=transport).run(PlanDAG(nodes=[node])))
    user = next(
        m["content"] for m in transport.calls[0]["messages"] if m["role"] == "user"
    )
    assert _OVERALL_GOAL_HEADER not in user
    assert user.startswith("Summarise the input.")


# --------------------------------------------------------------------------- #
# 4) BUDGET FINALIZATION vs num_ctx 32768: upstream prose clipped to 4000 chars;
#    the goal is fed VERBATIM (never clipped).
# --------------------------------------------------------------------------- #
def test_goal_verbatim_while_upstream_prose_clipped_to_budget():
    long_goal = "GOAL-SENTINEL " + ("g" * 6000)          # > the 4000 prose budget
    long_input = "UPSTREAM-SENTINEL " + ("u" * 6000)      # must be clipped to 4000
    node = PlanNode(id="n2", task="Write.", depends_on=["n1"])
    agent = SubAgent(
        node,
        transport=FakeTransport(["x"]),
        overall_goal=long_goal,
        upstream_input_char_budget=4000,
    )
    user = agent._compose_task({"n1": long_input}, tool_value=None)

    # The goal is present in FULL (verbatim, not clipped) — it is the authoritative
    # intent and is small relative to num_ctx 32768.
    assert long_goal in user
    # The upstream prose is clipped to the finalized 4000-char budget: the input
    # rendered for dep n1 is at most 4000 chars of value.
    dep_line = next(ln for ln in user.splitlines() if ln.startswith("- n1: "))
    rendered = dep_line[len("- n1: "):]
    assert len(rendered) == 4000
    assert rendered.startswith("UPSTREAM-SENTINEL ")
