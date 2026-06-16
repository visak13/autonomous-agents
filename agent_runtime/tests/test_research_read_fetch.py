"""d13 (a4): RESEARCH READS, IT DOES NOT DESCRIBE.

A research layer (and any plain ``web_search`` node) must report the ACTUAL
content of its sources, not summarise the search-results page. The runtime's
search-then-read behaviour makes that structural: a research-role node SEARCHES
its topic then ``web_fetch``es real result URLs (a round-rotating window), and a
plain ``web_search`` node FOLLOWS THROUGH and fetches its top hits — so
``web_fetch`` is ACTUALLY invoked on real upstream URLs and the EXTRACTED article
text is folded into the produce/role call (never skipped).

Fully OFFLINE: a fake hook records every web_search/web_fetch call and returns
canned article markdown; a recording transport captures the user turn the node's
LLM call sees. No GPU, no network.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from llm_framework import ChatResult

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.runtime import AgentRuntime, SubAgent
from agent_runtime.scheduler import ExecutionMode

# Four real-shaped article URLs the fake search returns; the fetcher gives each a
# distinct headline so we can prove WHICH ones a layer actually read.
_URLS = [
    "https://news.example.com/world",
    "https://news.example.com/tech",
    "https://news.example.com/biz",
    "https://news.example.com/science",
]
_HEADLINES = {
    _URLS[0]: "REAL HEADLINE WORLD: summit reaches accord",
    _URLS[1]: "REAL HEADLINE TECH: chip output up 12 percent",
    _URLS[2]: "REAL HEADLINE BIZ: index closes at record",
    _URLS[3]: "REAL HEADLINE SCIENCE: probe returns samples",
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


class _ResearchTransport:
    """Captures the user turn; answers the research per-role schema."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content

    def chat(self, messages, **opts) -> ChatResult:
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        self.calls.append({"user": user, "props": set((opts.get("format") or {}).get("properties", {}))})
        return ChatResult(role="assistant", content=json.dumps(
            {"findings": ["f"], "sources": _URLS[:1], "open_questions": ["q"]}))


def _research_node(nid: str = "r1_research") -> PlanNode:
    return PlanNode(id=nid, task="[research · round 1] morning world news briefing",
                    role="research")


def test_research_role_node_fetches_real_urls_and_folds_article_text():
    """A research node SEARCHES then web_fetches real URLs; the article text reaches
    the LLM call under the read-not-describe header (d13)."""
    hook, transport = _FakeHook(), _ResearchTransport()
    agent = SubAgent(
        _research_node(), transport=transport, hook=hook,
        read_search_max_fetch=2, call_opts={"think": False, "temperature": 0},
    )
    res = asyncio.run(agent.run({}))

    assert hook.searches, "research node never called web_search"
    # web_fetch ACTUALLY invoked on the REAL result URLs (the d13 bar — not skipped)
    assert hook.fetches == _URLS[:2], hook.fetches
    # the EXTRACTED article text (real headlines) reached the LLM user turn
    user = transport.calls[-1]["user"]
    assert "FETCHED SOURCE CONTENT" in user
    assert _HEADLINES[_URLS[0]] in user and _HEADLINES[_URLS[1]] in user
    # the enriched tool_value carries the fetched articles for downstream layers
    assert res.tool_value["fetched_count"] == 2


def test_research_skips_pdf_and_binary_urls():
    """A research layer must NOT try to read PDFs/files (Trafilatura is HTML-only and
    they decode to binary garbage) — non-article URLs are filtered before fetch, so
    only readable HTML articles are fetched (the live max_iter=10 'binary data' fix)."""
    class _MixedHook(_FakeHook):
        async def invoke(self, name, **args):
            if name == "web_search":
                self.searches.append(args.get("query", ""))
                return _ToolResult(True, {"results": [
                    {"title": "html", "url": "https://news.example.com/world", "snippet": "s"},
                    {"title": "pdf", "url": "https://news.example.com/report.pdf", "snippet": "s"},
                    {"title": "html2", "url": "https://news.example.com/tech", "snippet": "s"},
                ]})
            return await super().invoke(name, **args)

    hook = _MixedHook()
    agent = SubAgent(_research_node(), transport=_ResearchTransport(), hook=hook,
                     read_search_max_fetch=3)
    asyncio.run(agent.run({}))
    # the .pdf URL is never fetched; only the two HTML articles are
    assert hook.fetches == ["https://news.example.com/world", "https://news.example.com/tech"]


def test_research_emits_focused_queries_from_layer_context():
    """The research layer AUTHORS its own search query (a structured call) rather than
    blindly searching the verbose goal — so it can stay on-topic and go deeper."""
    class _QueryTransport(_ResearchTransport):
        def chat(self, messages, **opts):
            props = set((opts.get("format") or {}).get("properties", {}))
            if props == {"queries"}:  # the query-authoring call
                user = next((m["content"] for m in messages if m.get("role") == "user"), "")
                self.calls.append({"user": user, "props": props})
                return ChatResult(role="assistant",
                                  content=json.dumps({"queries": ["solid state battery 2026"]}))
            return super().chat(messages, **opts)

    hook, transport = _FakeHook(), _QueryTransport()
    agent = SubAgent(_research_node(), transport=transport, hook=hook, read_search_max_fetch=2)
    asyncio.run(agent.run({}))
    # the model-authored keyword query (not the verbose task) drove the search
    assert hook.searches == ["solid state battery 2026"], hook.searches
    assert hook.fetches == _URLS[:2]


def test_plain_web_search_node_follows_through_to_fetch():
    """An open-shape node with tool=web_search auto-reads its top hits: web_fetch
    fires on real URLs and the article text (not just snippets) reaches the call."""
    hook, transport = _FakeHook(), _ResearchTransport()
    node = PlanNode(id="n1", task="get the latest tech news", tool="web_search",
                    tool_args={"query": "tech news"})
    agent = SubAgent(node, transport=transport, hook=hook, read_search_max_fetch=3)
    asyncio.run(agent.run({}))
    assert hook.searches == ["tech news"]
    assert hook.fetches == _URLS[:3]
    assert "FETCHED SOURCE CONTENT" in transport.calls[-1]["user"]
    assert _HEADLINES[_URLS[0]] in transport.calls[-1]["user"]


def test_read_search_off_by_default_is_back_compat():
    """With read_search_max_fetch=0 (the default) a web_search node does NOT fetch —
    the pre-a4 single-tool behaviour is unchanged for every non-chat path."""
    hook, transport = _FakeHook(), _ResearchTransport()
    node = PlanNode(id="n1", task="search", tool="web_search", tool_args={"query": "q"})
    # role=None + no read budget => plain producer (worker-style) call, no role schema
    agent = SubAgent(node, transport=transport, hook=hook)  # read_search_max_fetch=0
    asyncio.run(agent.run({}))
    assert hook.searches == ["q"] and hook.fetches == []  # searched, did NOT fetch
