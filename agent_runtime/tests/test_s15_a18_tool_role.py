"""s15 / a18 (d189) — the SYSTEMIC ``role:'user'`` tool-result mislabel fix, OFFLINE.

ROOT CAUSE (the user's unifying root of the whole saga): tool OUTPUTS were appended back
to the message history as ``role:'user'`` across EVERY react loop, so the model read its
OWN tool results as fresh USER instructions and kept responding — the haiku multi-write /
60KB loop, the never-terminating ``[finish]``, and the tool-marker / file_read-echo LEAK
into the deliverable. a6 fixed only the decision node (``research_tree.py``); this sweep
flips the remaining loops so EVERY tool RESULT is fed back ``role:'tool'`` (a native Ollama
:11434 function-result) while GENUINE instructions (nudges / finalize prompts) stay
``role:'user'``.

s15/a25 (d199) NARROWS a18 for the RESEARCH GATHER loop ONLY: live gemma4-e4b's
``{{ .Prompt }}`` template has NO role handling, so a role:'tool' SEARCH RESULTS / FETCHED
observation is IGNORED and the model fabricates dead urls (0 sources/notes land). The gather
loop therefore feeds the observations the model must GROUND on back ``role:'user'`` (test 1
below asserts this d199 contract). The write / reviewer / planner loops are UNCHANGED — their
observations are acknowledged, not grounded on, so they KEEP ``role:'tool'`` (tests 2–4).

These tests are fully OFFLINE (no GPU, no network): a scripted :class:`FakeTransport`
records every call's ``messages``, so we assert directly that each fed-back observation
rode ``role:'tool'`` (never ``role:'user'``) in each swept loop AND that the loop
TERMINATES on the model's stop signal (findings / DONE / finalize) instead of churning.

The transport's live acceptance of ``role:'tool'`` was verified against the resident
gemma4-e4b model on :11434 during the a18 build (the model grounded correctly on a
``role:'tool'`` message both with and without a preceding assistant ``tool_calls``).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from llm_framework import FakeTransport

from agent_runtime.factory import AbstractPlanFactory, PlanDAG, PlanNode
from agent_runtime.incremental import IncrementalPlanner
from agent_runtime.runtime import AgentRuntime, SubAgent
from agent_runtime.synth_tools import DONE_SENTINEL
from reactive_tools import EventPlane, ToolHook, register_agentic_tools
from specialization.registry import SpecRegistry
from specialization.seed import seed_canonical_rulesets


def _run(coro):
    return asyncio.run(coro)


def _file_hook(tmp_path) -> ToolHook:
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    return hook


def _roles(messages) -> list[str]:
    return [m.get("role") for m in messages]


def _by_role(messages, role: str) -> list[str]:
    return [str(m.get("content", "")) for m in messages if m.get("role") == role]


def _all_turns(transport: FakeTransport, role: str) -> list[str]:
    """Every distinct turn of ``role`` across ALL recorded calls (the union the model saw)."""
    seen: list[str] = []
    for call in transport.calls:
        for content in _by_role(call["messages"], role):
            if content not in seen:
                seen.append(content)
    return seen


# --------------------------------------------------------------------------- #
# helpers — a hook that serves canned search/fetch data (no network)
# --------------------------------------------------------------------------- #
class _ToolResult:
    def __init__(self, ok: bool, value: Any = None, error: str = "") -> None:
        self.ok = ok
        self.value = value
        self.error = error
        self.call_id = "c1"


class _FakeHook:
    def __init__(self, urls: dict[str, str]) -> None:
        self._urls = urls

    async def invoke(self, name: str, **args) -> _ToolResult:
        if name == "web_search":
            return _ToolResult(True, {
                "query": args.get("query", ""),
                "results": [
                    {"title": f"t{i}", "url": u, "snippet": "snip"}
                    for i, u in enumerate(self._urls)
                ],
                "count": len(self._urls),
            })
        if name == "web_fetch":
            url = args.get("url", "")
            return _ToolResult(True, {
                "url": url, "final_url": url, "status": 200,
                "title": url.rsplit("/", 1)[-1],
                "markdown": self._urls.get(url, "article body"),
                "extracted": True,
            })
        return _ToolResult(False, error=f"unknown tool {name}")


def _ws_node(nid: str = "r1_research") -> PlanNode:
    return PlanNode(id=nid, task="[research] crisis timeline",
                    role="worker", tool="web_search", tool_args={"query": "crisis"})


_SOURCES = [
    {"title": "BBC", "url": "https://www.bbc.com/news/x", "source_trust": "secondary",
     "key_claims": ["$200bn requested"], "markdown": "The Pentagon requested $200bn."},
]


# --------------------------------------------------------------------------- #
# 1. RESEARCH GATHER loop — search/fetch results ride role 'tool', findings stop it
# --------------------------------------------------------------------------- #
def test_gather_loop_feeds_tool_results_as_role_user_d199_and_terminates():
    # s15/a25 (d199, supersedes a18/d189 for THIS gather loop ONLY): live gemma4-e4b uses a
    # '{{ .Prompt }}' chat template with NO role handling, so a role:'tool' message is IGNORED —
    # fed as 'tool' the model never reads the SEARCH RESULTS / FETCHED article and FABRICATES dead
    # urls, so 0 sources/notes land. d199 feeds the observations the model must GROUND on back
    # role:'user' so it actually reads them. The fix is NARROW: only the research GATHER loop
    # flips — the write / reviewer / planner loops below KEEP role:'tool' (a18 stands; those
    # observations are acknowledged, not grounded on). This test now asserts the d199 contract.
    world = "https://news.example.com/world"
    hook = _FakeHook({world: "WORLD ARTICLE\n\nbody text."})
    transport = FakeTransport([
        # d242 TRUE self-select: the gather node starts TOOL-LESS and loads 'research' first.
        '{"tool": "get_bundles", "args": {"name": "research"}}',
        '{"tool": "web_search", "args": {"query": "world crisis"}}',
        f'{{"tool": "web_fetch", "args": {{"url": "{world}"}}}}',
        "FINDINGS: a real answer grounded in the source (" + world + ").",
    ])
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=5, call_opts={"think": False, "temperature": 0},
    )
    res = _run(agent.run({}))

    # TERMINATED on the model's findings — exactly 4 calls (self-select + search + fetch +
    # findings), no salvage-finalize 5th call.
    assert transport.call_count == 4
    assert "real answer grounded" in (res.output or "")

    # d199: the search + fetch RESULTS were fed back role 'user' (the grounding lever), so the
    # findings call sees both on the lane this model actually reads.
    final_msgs = transport.calls[-1]["messages"]
    user_turns = _by_role(final_msgs, "user")
    assert any("SEARCH RESULTS" in t for t in user_turns)
    assert any("WORLD ARTICLE" in t for t in user_turns)
    # They were NOT fed role 'tool' (which this model's template ignores) — the d199 inversion
    # of a18 for the gather loop. No grounded observation hides on the role:'tool' lane.
    tool_turns = "\n".join(_by_role(final_msgs, "tool"))
    assert "SEARCH RESULTS" not in tool_turns
    assert "WORLD ARTICLE" not in tool_turns


# --------------------------------------------------------------------------- #
# 2. SYNTHESIS / WRITE loop — the saved-file observation rides role 'tool', DONE stops it
# --------------------------------------------------------------------------- #
def test_write_loop_feeds_saved_state_as_role_tool_and_terminates(tmp_path):
    transport = FakeTransport([_write_reply])
    rt = AgentRuntime(
        transport=transport, hook=_file_hook(tmp_path),
        subagent_call_opts={"think": True, "temperature": 0},
    )
    rt.chain_sources = _SOURCES
    dag = PlanDAG(
        nodes=[PlanNode(id="w", task="Write the notes to notes.txt.", tool="file_write")],
        rationale="r", goal="Write notes to notes.txt.",
    )
    out = _run(rt.run(dag))
    assert out.ok

    on_disk = open(out.results["w"].tool_value["path"], encoding="utf-8").read()
    assert "First section" in on_disk and "Second section" in on_disk

    # The "Saved part N … file ENDS with …" file observation was fed back role 'tool'
    # (the file_read-echo / never-terminate root); it NEVER rode role 'user'.
    tool_turns = "\n".join(_all_turns(transport, "tool"))
    user_turns = "\n".join(_all_turns(transport, "user"))
    assert "Saved part 1" in tool_turns
    assert "Saved part" not in user_turns
    # The continuation DIRECTIVE that follows the observation DOES stay role 'user'.
    assert "Continue the deliverable" in user_turns


def _write_reply(messages, **opts):
    n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
    if n_assistant == 0:
        return "First section: an opening paragraph of the notes."
    return "Second section: the closing paragraph. " + DONE_SENTINEL


# --------------------------------------------------------------------------- #
# 3. REVIEWER anchored-edit loop — the edit/read result rides role 'tool', DONE stops it
# --------------------------------------------------------------------------- #
_CLAIM = "<p>The Pentagon is asking for another $200bn in funding for the war.</p>"
_FIXED = "<p>The Pentagon is asking for another $200bn (Source: https://www.bbc.com/news/x).</p>"


def _review_dag() -> PlanDAG:
    return PlanDAG(
        nodes=[
            PlanNode(id="n1", task="Write the report to report.html.", tool="file_write"),
            PlanNode(
                id="n1_review",
                task=("Review the output of the step 'Write the report'. The deliverable is "
                      "ALREADY written to the file — fix any unsupported claim in place."),
                tool="file_write",
                depends_on=("n1",),
            ),
        ],
        rationale="r",
        goal="Write a report to report.html.",
    )


def _review_reply(messages, **opts):
    user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
    if "The deliverable is ALREADY written to the file" in user:
        # REVIEW turns (d242 TRUE self-select): the reviewer starts TOOL-LESS, so it FIRST
        # loads the 'file' bundle (its edit tools), THEN emits a file_update as TEXT (the
        # live-E4B shape), then DONE.
        if n_assistant == 0:
            return '{"tool": "get_bundles", "args": {"name": "file"}}'
        if n_assistant == 1:
            return ('{"tool": "file_update", "args": {"old": ' + json.dumps(_CLAIM)
                    + ', "new": ' + json.dumps(_FIXED) + '}}')
        return "DONE"
    # WRITER turns: emit the claim once, then confirm done.
    if n_assistant == 0:
        return "<!DOCTYPE html><html><body>" + _CLAIM + "</body></html>"
    return DONE_SENTINEL


_PLANNER_TOOL_CATALOG = [
    {"name": "web_search", "description": "search the web for candidate pages"},
    {"name": "file_write", "description": "write content to a file"},
]


def _planner(replies, tmp_path) -> IncrementalPlanner:
    """A tool-driven incremental authorer over a scripted FakeTransport (no GPU/network)."""
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)
    factory = AbstractPlanFactory(reg.index(), tool_catalog=_PLANNER_TOOL_CATALOG)
    return IncrementalPlanner(
        FakeTransport(list(replies)),
        factory,
        spec_names=reg.names(),
        tool_names=[t["name"] for t in _PLANNER_TOOL_CATALOG],
        shape_name="linear-plus-modular-parallel",
        shape_description="parallel gather steps, then a sequential combine→deliver chain",
    )


def test_planner_loop_feeds_builder_observation_as_role_tool_and_finalizes(tmp_path):
    replies = [
        json.dumps({"tool": "seed_plan", "args": {"shape": "linear-plus-modular-parallel"}}),
        json.dumps({"tool": "add_step",
                    "args": {"task": "Search the news", "tool": "web_search",
                             "spec": "", "specs": [], "depends_on": []}}),
        json.dumps({"tool": "add_step",
                    "args": {"task": "Write the brief", "tool": "file_write",
                             "spec": "", "specs": [], "depends_on": ["n1"]}}),
        json.dumps({"tool": "finalize_plan", "args": {}}),
    ]
    planner = _planner(replies, tmp_path)
    transport = planner.transport  # the FakeTransport injected above
    result = _run(planner.plan("search the news, then write a brief"))

    # TERMINATED: the planner finalized the plan (the finalize call broke the loop).
    assert planner.last_builder is not None and planner.last_builder.finalized is True
    assert result.dag.nodes

    # The builder-dispatch OBSERVATIONS ("OBSERVATION: …") were fed back role 'tool',
    # never role 'user' — so the planner reasons over plan state, not its own tool output.
    tool_turns = "\n".join(_all_turns(transport, "tool"))
    user_turns = "\n".join(_all_turns(transport, "user"))
    assert "OBSERVATION:" in tool_turns
    assert "OBSERVATION:" not in user_turns
