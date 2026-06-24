"""s9/c5 (d49/d50): RESEARCH IS A TRUE AGENTIC LOOP — the worker DECIDES to fetch.

The deterministic search-then-read EXECUTOR (flags #1/#3: ``read_search_max_fetch``
> 0 forcing a ``web_search`` node to auto-``web_fetch`` its top hits) is RETIRED. A
``web_search`` node is now a TRUE AGENT (:meth:`SubAgent._run_research_loop`): it
emits lightweight tool calls (``web_search``/``web_fetch``) to gather REAL evidence,
the loop executes each against the real hook and feeds the observation back, and the
model writes its FINDINGS as RAW prose (no ``format=<schema>``, content RAW). The
``web_fetch`` calls are the MODEL's choices, not a deterministic top-k follow-through;
``read_search_max_fetch`` survives only as a NON-FLOW fetch CAP.

Fully OFFLINE: a fake hook records every web_search/web_fetch call and returns canned
article markdown; a SCRIPTED transport replays a sequence of agent turns (tool-call
JSON, then findings prose). No GPU, no network.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from llm_framework import ChatResult

from agent_runtime.factory import PlanNode
from agent_runtime.runtime import SubAgent

# Real-shaped article URLs the fake search returns; the fetcher gives each a distinct
# headline so we can prove WHICH ones the agent actually chose to read.
_URLS = [
    "https://news.example.com/world",
    "https://news.example.com/tech",
    "https://news.example.com/biz",
]
_HEADLINES = {
    _URLS[0]: "REAL HEADLINE WORLD: summit reaches accord",
    _URLS[1]: "REAL HEADLINE TECH: chip output up 12 percent",
    _URLS[2]: "REAL HEADLINE BIZ: index closes at record",
}


@dataclass
class _ToolResult:
    ok: bool
    value: Any = None
    error: str = ""
    call_id: str = "c1"


class _FakeHook:
    """Records web_search / web_fetch invocations; returns canned real-shaped data."""

    def __init__(self) -> None:
        self.searches: list[str] = []
        self.fetches: list[str] = []

    async def invoke(self, name: str, **args) -> _ToolResult:
        if name == "web_search":
            self.searches.append(args.get("query", ""))
            return _ToolResult(True, {
                "query": args.get("query", ""),
                "results": [
                    {"title": f"t{i}", "url": u, "snippet": "snip"}
                    for i, u in enumerate(_URLS)
                ],
                "count": len(_URLS),
            })
        if name == "web_fetch":
            url = args.get("url", "")
            self.fetches.append(url)
            return _ToolResult(True, {
                "url": url, "final_url": url, "status": 200,
                "title": url.rsplit("/", 1)[-1],
                "markdown": f"{_HEADLINES.get(url, 'article')}\n\nbody text for {url}.",
                "extracted": True,
            })
        return _ToolResult(False, error=f"unknown tool {name}")


class _ScriptedTransport:
    """Replays a fixed sequence of agent turns (one per chat call) + records the
    user turns the agent saw (so we can assert observations were fed back)."""

    def __init__(self, turns: list[str]) -> None:
        self._turns = list(turns)
        self.calls: list[str] = []  # the user turn of each chat call

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content

    def chat(self, messages, **opts) -> ChatResult:
        # The LAST user message is the freshest observation / instruction.
        user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        i = len(self.calls)
        self.calls.append(user)
        content = self._turns[i] if i < len(self._turns) else "FALLBACK FINDINGS."
        return ChatResult(role="assistant", content=content)


def _ws_node(nid: str = "r1_research", query: str = "morning world news") -> PlanNode:
    # d48: a research-POSITION node is a WORKER carrying the web_search tool.
    return PlanNode(id=nid, task="[research] morning world news briefing",
                    role="worker", tool="web_search", tool_args={"query": query})


def _full_convo(transport: _ScriptedTransport) -> str:
    return "\n\n".join(transport.calls)


def test_worker_decides_to_search_then_fetch_chosen_urls_and_fold_text():
    """The agent emits web_search → fetches the URLs IT chose → writes findings; the
    fetched article text is fed back, and the read sources land on tool_value (d17)."""
    hook = _FakeHook()
    transport = _ScriptedTransport([
        '{"tool": "web_search", "args": {"query": "world summit accord"}}',
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/world"}}',
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/tech"}}',
        "FINDINGS: a summit reached an accord (https://news.example.com/world) and "
        "chip output rose (https://news.example.com/tech).",
    ])
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=5, call_opts={"think": False, "temperature": 0},
    )
    res = asyncio.run(agent.run({}))

    # the MODEL's chosen query + URLs were executed (not a deterministic top-k)
    assert hook.searches == ["world summit accord"]
    assert hook.fetches == [_URLS[0], _URLS[1]]
    # the EXTRACTED article text was fed back to the agent (read-not-describe)
    convo = _full_convo(transport)
    assert "FETCHED" in convo
    assert _HEADLINES[_URLS[0]] in convo and _HEADLINES[_URLS[1]] in convo
    # findings prose is the RAW node output
    assert "summit reached an accord" in (res.output or "")
    # the read sources are attached for downstream grounding (d17)
    assert res.tool_value["fetched_count"] == 2
    assert {s["url"] for s in res.tool_value["fetched"]} == {_URLS[0], _URLS[1]}


def test_fetch_cap_is_a_non_flow_bound():
    """``read_search_max_fetch`` is a CAP, not a gate: the agent may try more fetches
    but only ``cap`` web_fetch calls are executed, then it is told to write findings."""
    hook = _FakeHook()
    transport = _ScriptedTransport([
        '{"tool": "web_search", "args": {"query": "news"}}',
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/world"}}',
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/tech"}}',  # over cap
        "FINDINGS from the one source I read.",
    ])
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=1, call_opts={"think": False},
    )
    res = asyncio.run(agent.run({}))
    assert hook.fetches == [_URLS[0]]  # the 2nd fetch was capped, never invoked
    assert "Fetch limit (1) reached" in _full_convo(transport)
    assert res.tool_value["fetched_count"] == 1


def test_non_article_url_is_rejected_before_fetch():
    """A model-chosen PDF/binary URL is filtered BEFORE the fetch (Trafilatura is
    HTML-only); the agent is told to pick another source — no wasted/garbage fetch."""
    hook = _FakeHook()
    transport = _ScriptedTransport([
        '{"tool": "web_search", "args": {"query": "report"}}',
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/report.pdf"}}',
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/world"}}',
        "FINDINGS from the readable article.",
    ])
    agent = SubAgent(_ws_node(), transport=transport, hook=hook,
                     read_search_max_fetch=5, call_opts={"think": False})
    asyncio.run(agent.run({}))
    # the .pdf URL was NEVER fetched; only the HTML article was
    assert hook.fetches == [_URLS[0]]
    assert "not a readable HTML article" in _full_convo(transport)


def test_worker_may_decide_not_to_fetch():
    """The worker's reasoning is honored: it may search and then answer WITHOUT
    fetching (no forced follow-through). The choice is the model's, not a flag."""
    hook = _FakeHook()
    transport = _ScriptedTransport([
        '{"tool": "web_search", "args": {"query": "quick facts"}}',
        "FINDINGS: a concise answer from the search snippets alone.",
    ])
    agent = SubAgent(_ws_node(), transport=transport, hook=hook,
                     read_search_max_fetch=5, call_opts={"think": False})
    res = asyncio.run(agent.run({}))
    assert hook.searches == ["quick facts"]
    assert hook.fetches == []  # the agent chose not to read full sources
    assert res.tool_value is None  # no sources read → nothing to attach
    assert "concise answer" in (res.output or "")


def test_findings_only_turn_ends_the_loop_with_no_tools():
    """If the very first turn is prose (no tool call), that prose IS the findings and
    the loop ends — the agent answered from its assembled context, no tool forced."""
    hook = _FakeHook()
    transport = _ScriptedTransport([
        "FINDINGS: I can answer this directly from the provided context.",
    ])
    agent = SubAgent(_ws_node(), transport=transport, hook=hook,
                     read_search_max_fetch=5, call_opts={"think": False})
    res = asyncio.run(agent.run({}))
    assert hook.searches == [] and hook.fetches == []
    assert "answer this directly" in (res.output or "")


# --------------------------------------------------------------------------- #
# s9/N1 (d60/c15 part-a) — BREADTH: the fetch cap lifts from the legacy 3 to a
# configurable ~8-12, and the ReAct TURN CEILING rises proportionally so a
# high-breadth gather can search several angles AND read MANY sources without the
# flat ceiling clipping it. Still a NON-FLOW bound: the model decides whether/which.
# --------------------------------------------------------------------------- #
class _ManyUrlHook:
    """A fake hook with MANY distinct article URLs so we can prove a single research
    node reads far more than the legacy 3-source ceiling."""

    def __init__(self, n: int) -> None:
        self.urls = [f"https://news.example.com/a{i}" for i in range(n)]
        self.searches: list[str] = []
        self.fetches: list[str] = []

    async def invoke(self, name: str, **args) -> _ToolResult:
        if name == "web_search":
            self.searches.append(args.get("query", ""))
            return _ToolResult(True, {
                "query": args.get("query", ""),
                "results": [
                    {"title": f"t{i}", "url": u, "snippet": "snip"}
                    for i, u in enumerate(self.urls)
                ],
                "count": len(self.urls),
            })
        if name == "web_fetch":
            url = args.get("url", "")
            self.fetches.append(url)
            return _ToolResult(True, {
                "url": url, "final_url": url, "status": 200,
                "title": url.rsplit("/", 1)[-1],
                "markdown": f"REAL ARTICLE {url}\n\nsubstantive body for {url}.",
                "extracted": True,
            })
        return _ToolResult(False, error=f"unknown tool {name}")


def test_breadth_turn_ceiling_rises_with_fetch_cap():
    """With a BREADTH cap (11), the agent reads ELEVEN real sources in one node — far
    past the legacy 3 — across TWO search angles. This run needs 14 turns (search,
    fetch×5, search, fetch×6, findings); under the OLD flat 12-turn ceiling the 11th
    fetch and the in-loop findings turn would be clipped, so proving all 11 URLs were
    fetched AND the in-loop findings were captured proves the ceiling scaled with the
    cap. The cap is still NON-FLOW — the model chose to search/fetch; nothing forced it."""
    hook = _ManyUrlHook(11)
    u = hook.urls
    turns = (
        ['{"tool": "web_search", "args": {"query": "angle one"}}']
        + [f'{{"tool": "web_fetch", "args": {{"url": "{u[i]}"}}}}' for i in range(5)]
        + ['{"tool": "web_search", "args": {"query": "angle two"}}']
        + [f'{{"tool": "web_fetch", "args": {{"url": "{u[i]}"}}}}' for i in range(5, 11)]
        + ["BREADTH FINDINGS: eleven sources synthesised, each attributed."]
    )
    assert len(turns) == 14  # genuinely exceeds the legacy RESEARCH_MAX_TURNS=12
    transport = _ScriptedTransport(turns)
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=11, call_opts={"think": False, "temperature": 0},
    )
    res = asyncio.run(agent.run({}))

    # All ELEVEN model-chosen sources were actually fetched (breadth, not capped at 3).
    assert hook.fetches == u
    assert res.tool_value["fetched_count"] == 11
    # The in-loop findings turn (turn 14) ran — proving the ceiling scaled past 12, not
    # the post-loop fallback salvage (which would have emitted a fetch-call JSON string).
    assert "BREADTH FINDINGS" in (res.output or "")


def test_narrow_cap_keeps_the_legacy_turn_floor():
    """A NARROW cap (≤5) leaves the turn ceiling at the original RESEARCH_MAX_TURNS
    floor — narrow/inline paths are unchanged by the breadth scaling (no regression)."""
    from agent_runtime.runtime import RESEARCH_MAX_TURNS, RESEARCH_SEARCH_HEADROOM

    # For any legacy cap the proportional ceiling stays clamped to the floor.
    for cap in (1, 3, 5):
        assert max(RESEARCH_MAX_TURNS, cap + RESEARCH_SEARCH_HEADROOM) == RESEARCH_MAX_TURNS
    # A breadth cap genuinely raises it.
    assert max(RESEARCH_MAX_TURNS, 10 + RESEARCH_SEARCH_HEADROOM) > RESEARCH_MAX_TURNS
