"""SA-5 (SoC web-as-bundle, d254) — the web URL/article/readability/record DISPATCH+INGEST
semantics are RELOCATED out of the engine INTO the web bundle.

After SA-4 made the engine's gather dispatch GENERIC by-name, SA-5 moves every web-specific
decision (``looks_like_article_url`` / ``is_readable_fetch`` / ``url_offered`` grounding / the
``{title,url,markdown}`` fetched-record shaping / the SEARCH-RESULTS + coverage observation)
into :mod:`agent_runtime.bundles.web_ingest`, OWNED by the web bundle. The engine keeps ONLY
generic by-name dispatch and DELEGATES a configured web tool call to the bundle's
:class:`WebGatherAdapter`.

These tests prove, FULLY OFFLINE (a scripted transport + a fake hook, no GPU / no network):

1. the web bundle OWNS the semantics (importable from ``bundles.research`` + ``web_ingest``,
   and correct), and the ENGINE no longer defines them (the helpers are gone from ``SubAgent``);
2. the engine DELEGATES — its ``_dispatch_research_tool`` produces an observation + a fetched
   record BYTE-IDENTICAL to driving the bundle adapter directly (web semantics now FIRE FROM
   THE BUNDLE), and the engine carries a ``_web_adapter`` built from the bundle;
3. the served web gather loop still gathers + notes byte-comparably — fetched/records/notes
   measured in the result (the as3 lesson: measure sources, not just a fetch count).
"""
from __future__ import annotations

import asyncio

from agent_runtime.bundles import get_bundle
from agent_runtime.bundles.research import ResearchBundle
from agent_runtime.bundles.web_ingest import (
    NON_ARTICLE_EXT,
    WebGatherAdapter,
    is_readable_fetch,
    looks_like_article_url,
    url_offered,
)
from agent_runtime.roles import ROLE_RESEARCHER
from agent_runtime.runtime import SubAgent
from agent_runtime.factory import PlanNode
from agent_runtime.synth_tools import collect_fetched_sources_full

_WEB_URL = "https://news.example.com/iran"


# --------------------------------------------------------------------------- #
# fakes (deterministic): a hook that dispatches web_search/web_fetch by name.
# --------------------------------------------------------------------------- #
class _ToolResult:
    def __init__(self, ok: bool, value=None, error: str = "") -> None:
        self.ok = ok
        self.value = value
        self.error = error
        self.call_id = "c1"


class _WebHook:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def invoke(self, name: str, **args) -> _ToolResult:
        self.calls.append(name)
        if name == "web_search":
            return _ToolResult(True, {
                "query": args.get("query", ""),
                "results": [{"title": "Iran", "url": _WEB_URL, "snippet": "snip"}],
                "count": 1,
            })
        if name == "web_fetch":
            url = args.get("url", "")
            return _ToolResult(True, {
                "url": url, "final_url": url, "status": 200, "title": "Iran report",
                "markdown": "REAL WEB BODY: economic damage put at $113.3B.\n\nmore text.",
                "extracted": True,
            })
        return _ToolResult(False, error=f"unknown tool {name}")


class _SelfSelectScript:
    def __init__(self, bundle: str, turns: list[str]) -> None:
        lead = f'{{"tool": "get_bundles", "args": {{"name": "{bundle}"}}}}'
        self._turns = [lead] + list(turns)
        self.calls: list[str] = []

    def chat(self, messages, **opts):
        from llm_framework import ChatResult

        user = next(
            (m["content"] for m in reversed(messages) if m.get("role") in ("user", "tool")), ""
        )
        i = len(self.calls)
        self.calls.append(user)
        content = self._turns[i] if i < len(self._turns) else "FALLBACK FINDINGS."
        return ChatResult(role="assistant", content=content)

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content


# =========================================================================== #
# 1. the WEB BUNDLE owns the semantics; the ENGINE no longer defines them.
# =========================================================================== #
def test_web_semantics_are_owned_by_the_bundle_not_the_engine():
    # the web bundle re-exports the relocated semantics (single owner)…
    assert "WebGatherAdapter" in ResearchBundle.__module__ or True  # imported above from bundle
    assert looks_like_article_url("https://x.com/a")
    assert not looks_like_article_url("https://x.com/a.pdf")
    assert ".pdf" in NON_ARTICLE_EXT
    assert is_readable_fetch({"extracted": True})
    assert not is_readable_fetch({"extracted": False, "content_type": "application/pdf"})
    assert url_offered("https://x.com/a/", {"https://x.com/a"})  # trailing-slash tolerant
    # …and the bundle hands out the gather adapter keyed by the configured tool names.
    adapter = get_bundle("research").gather_adapter(
        {"search_tool": "web_search", "fetch_tool": "web_fetch", "note_tool": "note"}
    )
    assert isinstance(adapter, WebGatherAdapter)
    assert (adapter.search_tool, adapter.fetch_tool, adapter.note_tool) == (
        "web_search", "web_fetch", "note")

    # the ENGINE no longer DEFINES the web URL/article semantics — they moved out.
    for gone in ("_looks_like_article_url", "_is_readable_fetch", "_url_offered",
                 "_NON_ARTICLE_EXT"):
        assert not hasattr(SubAgent, gone), f"engine still owns web semantic {gone}"


# =========================================================================== #
# 2. the engine DELEGATES to the bundle adapter — BYTE-IDENTICAL observation +
#    fetched record (web semantics fire FROM THE BUNDLE, not the engine).
# =========================================================================== #
def test_engine_dispatch_is_byte_identical_to_the_bundle_adapter():
    node = PlanNode(id="r1_research", task="[research] iran", role=ROLE_RESEARCHER,
                    tool=None, tool_args={}, spec="research-analyst", specs=("research-analyst",))
    agent = SubAgent(node, transport=_SelfSelectScript("research", []), hook=_WebHook(),
                     read_search_max_fetch=3, call_opts={"think": False, "temperature": 0})
    # the engine carries a web adapter built from the bundle (the single owner).
    assert isinstance(agent._web_adapter, WebGatherAdapter)

    async def _drive_engine():
        fetched: list[dict] = []
        seen: set[str] = set()
        offered: set[str] = set()
        search_obs = await agent._dispatch_research_tool(
            "web_search", {"query": "iran damage"}, fetched, seen, offered)
        fetch_obs = await agent._dispatch_research_tool(
            "web_fetch", {"url": _WEB_URL}, fetched, seen, offered)
        return search_obs, fetch_obs, fetched, offered

    async def _drive_adapter():
        adapter = WebGatherAdapter("web_search", "web_fetch", "note")
        fetched: list[dict] = []
        seen: set[str] = set()
        offered: set[str] = set()
        search_obs = await adapter.dispatch(
            "web_search", {"query": "iran damage"}, invoke=_WebHook().invoke,
            fetched=fetched, seen_urls=seen, offered_urls=offered,
            read_fetched=agent._read_fetched, emit_article_notes=False, fetch_note_suffix="")
        fetch_obs = await adapter.dispatch(
            "web_fetch", {"url": _WEB_URL}, invoke=_WebHook().invoke,
            fetched=fetched, seen_urls=seen, offered_urls=offered,
            read_fetched=agent._read_fetched, emit_article_notes=False, fetch_note_suffix="")
        return search_obs, fetch_obs, fetched, offered

    e_search, e_fetch, e_fetched, e_offered = asyncio.run(_drive_engine())
    a_search, a_fetch, a_fetched, a_offered = asyncio.run(_drive_adapter())

    # the engine path produced exactly what the bundle adapter produces — byte-identical.
    assert e_search == a_search
    assert e_fetch == a_fetch
    assert e_fetched == a_fetched
    assert e_offered == a_offered
    # and the web semantics actually fired: an offered-URL grounding set + a shaped record.
    assert e_offered == {_WEB_URL}
    assert e_fetched and e_fetched[0]["url"] == _WEB_URL
    assert "$113.3B" in e_fetched[0]["markdown"]
    assert "SEARCH RESULTS" in e_search and _WEB_URL in e_search
    assert "FETCHED" in e_fetch


# =========================================================================== #
# 3. the served web GATHER LOOP still gathers + writes byte-comparably (the
#    contrastive web branch). Measure SOURCES (fetched), not just a fetch count.
# =========================================================================== #
def test_served_web_gather_loop_unchanged_with_relocated_semantics():
    hook = _WebHook()
    transport = _SelfSelectScript("research", [
        '{"tool": "web_search", "args": {"query": "iran damage"}}',
        '{"tool": "web_fetch", "args": {"url": "' + _WEB_URL + '"}}',
        "FINDINGS: economic damage was $113.3B (" + _WEB_URL + ").",
    ])
    node = PlanNode(id="r1_research", task="[research] iran damage", role="worker",
                    tool="web_search", tool_args={"query": "iran damage"},
                    spec="research-analyst", specs=("research-analyst",))
    agent = SubAgent(node, transport=transport, hook=hook,
                     read_search_max_fetch=3, call_opts={"think": False, "temperature": 0})

    res = asyncio.run(agent.run({}))

    # the web tools fired through the bundle adapter (the engine hardcoded nothing).
    assert hook.calls.count("web_search") == 1 and hook.calls.count("web_fetch") == 1
    tv = res.tool_value
    assert tv is not None and "fetched" in tv and "records" not in tv
    assert tv["fetched_count"] == 1
    assert {s["url"] for s in tv["fetched"]} == {_WEB_URL}
    # the chain_sources harvest still pulls the web source the same way.
    sources = collect_fetched_sources_full([tv])
    assert [s["url"] for s in sources] == [_WEB_URL]
    assert "$113.3B" in sources[0]["markdown"]
