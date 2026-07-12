"""Offline coverage for the s2/a2 web tools (web_search + web_fetch).

Network is never touched: ``web_search`` runs against an injected fake
:data:`SearchBackend`, and ``web_fetch`` runs against a monkeypatched
``_http_get_bytes`` that returns canned HTML — so the cache, backoff, normalised
result contract, Trafilatura→markdown extraction, the fallback path, and the
registry growth/dispatch are all proven deterministically with zero GPU and zero
HTTP. The live ``gemma4-e2b-agent`` proof of both tools is reported separately
(a6, no evidence files — d6).
"""
from __future__ import annotations

import asyncio

import pytest

from reactive_tools import (
    EventPlane,
    GrowableToolRegistry,
    ToolHook,
    WebFetchArgs,
    WebSearchArgs,
    WEB_FETCH_TOOL,
    WEB_SEARCH_TOOL,
    make_web_fetch,
    make_web_search,
    register_web_tools,
)
from reactive_tools import web_tools as wt
from reactive_tools.tools import ToolInputError
from reactive_tools.web_tools import ResultCache, ddgs_backend


# --------------------------------------------------------------------------- #
# ToolDef shape — each web tool is ONE Pydantic-typed registry entry
# --------------------------------------------------------------------------- #


def test_web_tools_are_single_pydantic_typed_tooldefs():
    assert WEB_SEARCH_TOOL.name == "web_search"
    assert WEB_SEARCH_TOOL.required_keys() == ["query"]
    assert WEB_SEARCH_TOOL.args_model is WebSearchArgs
    assert WEB_FETCH_TOOL.name == "web_fetch"
    assert WEB_FETCH_TOOL.required_keys() == ["url"]
    assert WEB_FETCH_TOOL.args_model is WebFetchArgs


# --------------------------------------------------------------------------- #
# web_search — normalised contract, cache, backoff
# --------------------------------------------------------------------------- #


def _fake_backend(rows):
    calls = {"n": 0}

    def backend(query, max_results, region, timeout):
        calls["n"] += 1
        return list(rows)[:max_results]

    backend.calls = calls  # type: ignore[attr-defined]
    return backend


def test_web_search_normalises_and_returns_contract():
    rows = [{"title": "A", "url": "https://a.example", "snippet": "sa"},
            {"title": "B", "url": "https://b.example", "snippet": "sb"}]
    search = make_web_search(backend=_fake_backend(rows))
    out = search("python asyncio", max_results=5)
    assert out["query"] == "python asyncio"
    assert out["count"] == 2
    assert out["cached"] is False
    assert out["results"][0] == {"title": "A", "url": "https://a.example", "snippet": "sa"}


def test_web_search_caches_repeat_query():
    backend = _fake_backend([{"title": "A", "url": "https://a.example", "snippet": "s"}])
    cache = ResultCache()
    search = make_web_search(backend=backend, cache=cache)
    first = search("dup query", max_results=3)
    second = search("dup query", max_results=3)
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["results"] == first["results"]
    assert backend.calls["n"] == 1  # backend hit exactly once


def test_web_search_cache_key_includes_args():
    backend = _fake_backend([{"title": "A", "url": "https://a.example", "snippet": "s"}])
    search = make_web_search(backend=backend, cache=ResultCache())
    search("q", max_results=3)
    search("q", max_results=7)        # different max_results -> not a cache hit
    search("q", max_results=3, region="uk-en")  # different region -> not a hit
    assert backend.calls["n"] == 3


def test_web_search_backoff_then_success():
    sleeps: list[float] = []
    state = {"n": 0}

    def flaky(query, max_results, region, timeout):
        state["n"] += 1
        if state["n"] < 3:
            raise wt.RatelimitException("429")
        return [{"title": "ok", "url": "https://ok.example", "snippet": "s"}]

    search = make_web_search(backend=flaky, max_retries=4, backoff_base=1.0,
                             sleep=sleeps.append)
    out = search("q")
    assert out["count"] == 1
    assert state["n"] == 3                 # 2 failures + 1 success
    assert sleeps == [1.0, 2.0]            # exponential backoff between retries


def test_web_search_backoff_exhausted_raises():
    def always_limited(query, max_results, region, timeout):
        raise wt.RatelimitException("429")

    search = make_web_search(backend=always_limited, max_retries=2,
                             backoff_base=0.0, sleep=lambda s: None)
    with pytest.raises(ToolInputError):
        search("q")


def test_web_search_rejects_empty_query():
    search = make_web_search(backend=_fake_backend([]))
    with pytest.raises(ToolInputError):
        search("   ")


def test_web_search_clamps_max_results():
    backend = _fake_backend([{"title": str(i), "url": f"https://{i}.example", "snippet": "s"}
                             for i in range(40)])
    search = make_web_search(backend=backend)
    out = search("q", max_results=999)     # clamped to 25
    assert out["count"] == 25


# --------------------------------------------------------------------------- #
# ResultCache — TTL expiry (injected clock)
# --------------------------------------------------------------------------- #


def test_result_cache_ttl_expiry_and_lru():
    now = {"t": 0.0}
    cache = ResultCache(ttl=10.0, max_entries=2, clock=lambda: now["t"])
    cache.put("k", [1])
    assert cache.get("k") == [1]
    now["t"] = 11.0
    assert cache.get("k") is None          # expired
    # LRU eviction bound
    now["t"] = 100.0
    cache.put("a", [1]); cache.put("b", [2]); cache.put("c", [3])
    assert len(cache) == 2
    assert cache.get("a") is None          # oldest evicted


# --------------------------------------------------------------------------- #
# ddgs_backend — normalises {title,href,body} -> {title,url,snippet}
# --------------------------------------------------------------------------- #


def test_ddgs_backend_normalises_rows(monkeypatch):
    class FakeDDGS:
        def __init__(self, *a, **k):
            ...

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, region, max_results, timelimit=None):
            return [
                {"title": "T1", "href": "https://t1.example", "body": "b1"},
                {"title": "T2", "href": "", "body": "b2"},        # dropped (no url)
                {"title": "T3", "href": "https://t3.example", "body": "b3"},
            ]

    monkeypatch.setattr(wt, "DDGS", FakeDDGS)
    out = ddgs_backend("q", 8, "us-en", 20.0)
    assert out == [
        {"title": "T1", "url": "https://t1.example", "snippet": "b1"},
        {"title": "T3", "url": "https://t3.example", "snippet": "b3"},
    ]


# --------------------------------------------------------------------------- #
# web_fetch — Trafilatura markdown extraction + fallback
# --------------------------------------------------------------------------- #


class _FakeResp:
    def __init__(self, url, status=200, content_type="text/html; charset=utf-8",
                 truncated=False):
        self.url = url
        self.status_code = status
        self.headers = {"content-type": content_type}
        self._reactive_truncated = truncated


_ARTICLE_HTML = """<html><head><title>Doc Title</title></head><body><article>
<h1>Main Heading</h1>
<p>This is a substantial first paragraph with enough words to clear the precision
threshold so trafilatura keeps it and renders it as real markdown content for the
converter to format properly with a heading above it.</p>
<h2>Subsection</h2>
<p>Another paragraph here, also reasonably long, discussing a topic in enough depth
that the extractor treats it as main body text and emits it under the heading.</p>
<ul><li>first bullet item that is long enough to keep</li><li>second bullet item also long enough</li></ul>
<p>See <a href="https://example.org/more">the link</a> for more on the subject above.</p>
</article></body></html>""".encode("utf-8")


def test_web_fetch_extracts_markdown(monkeypatch):
    resp = _FakeResp("https://example.com/a")
    monkeypatch.setattr(wt, "_http_get_bytes",
                        lambda url, **k: (_ARTICLE_HTML, resp))
    fetch = make_web_fetch()
    out = fetch("https://example.com/a")
    assert out["status"] == 200
    assert out["extracted"] is True
    assert out["final_url"] == "https://example.com/a"
    # real markdown markup is present
    assert "# Main Heading" in out["markdown"]
    assert "- first bullet item" in out["markdown"]
    assert "[the link](https://example.org/more)" in out["markdown"]
    assert out["title"] == "Doc Title"


def test_web_fetch_falls_back_when_not_extractable(monkeypatch):
    tiny = b"<html><body><p>hi</p></body></html>"
    resp = _FakeResp("https://example.com/tiny")
    monkeypatch.setattr(wt, "_http_get_bytes", lambda url, **k: (tiny, resp))
    # force trafilatura to return nothing so the stdlib fallback is exercised
    monkeypatch.setattr(wt.trafilatura, "extract", lambda *a, **k: None)
    out = fetch_result = make_web_fetch()("https://example.com/tiny")
    assert out["extracted"] is False
    assert "hi" in out["markdown"]         # stdlib text fallback still yields content


def test_web_fetch_passes_through_plain_text(monkeypatch):
    resp = _FakeResp("https://example.com/p.txt", content_type="text/plain")
    monkeypatch.setattr(wt, "_http_get_bytes",
                        lambda url, **k: (b"just plain text", resp))
    out = make_web_fetch()("https://example.com/p.txt")
    assert out["extracted"] is False
    assert out["markdown"] == "just plain text"


def test_web_fetch_caches_same_url(monkeypatch):
    """d221: a same-url re-fetch within the TTL is served from cache (no live HTTP)."""
    calls = {"n": 0}

    def _counting_get(url, **k):
        calls["n"] += 1
        return _ARTICLE_HTML, _FakeResp("https://example.com/a")

    monkeypatch.setattr(wt, "_http_get_bytes", _counting_get)
    fetch = make_web_fetch()
    first = fetch("https://example.com/a")
    second = fetch("https://example.com/a")
    assert calls["n"] == 1                    # only ONE live round-trip
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["markdown"] == first["markdown"]


def test_web_fetch_cache_key_includes_max_bytes(monkeypatch):
    """A larger re-fetch must NOT be served a smaller cached body (key has max_bytes)."""
    calls = {"n": 0}

    def _counting_get(url, **k):
        calls["n"] += 1
        return _ARTICLE_HTML, _FakeResp("https://example.com/a")

    monkeypatch.setattr(wt, "_http_get_bytes", _counting_get)
    fetch = make_web_fetch()
    fetch("https://example.com/a", max_bytes=1000)
    fetch("https://example.com/a", max_bytes=5000)   # different key → live fetch
    assert calls["n"] == 2


def test_web_fetch_cache_expires_on_ttl(monkeypatch):
    """An entry past the TTL is re-fetched (clock-driven, deterministic)."""
    now = {"t": 0.0}
    cache = ResultCache(ttl=10.0, clock=lambda: now["t"])
    calls = {"n": 0}

    def _counting_get(url, **k):
        calls["n"] += 1
        return _ARTICLE_HTML, _FakeResp("https://example.com/a")

    monkeypatch.setattr(wt, "_http_get_bytes", _counting_get)
    fetch = make_web_fetch(cache=cache)
    fetch("https://example.com/a")
    now["t"] = 11.0                           # advance past the TTL
    out = fetch("https://example.com/a")
    assert calls["n"] == 2                    # cache expired → fetched again
    assert out["cached"] is False


def test_web_fetch_does_not_cache_failures(monkeypatch):
    """A failure is never pinned — the next call retries the live fetch."""
    calls = {"n": 0}

    def _failing_get(url, **k):
        calls["n"] += 1
        raise wt.httpx.TimeoutException("boom")

    monkeypatch.setattr(wt, "_http_get_bytes", _failing_get)
    fetch = make_web_fetch(max_retries=0, sleep=lambda s: None)
    first = fetch("https://example.com/a")
    second = fetch("https://example.com/a")
    assert first["ok"] is False
    assert second["ok"] is False
    assert calls["n"] == 2                    # both attempted live, none cached


# --------------------------------------------------------------------------- #
# Registry growth — register_web_tools adds both as one ToolDef each, dispatchable
# --------------------------------------------------------------------------- #


def test_register_web_tools_growth_and_dispatch():
    registry = GrowableToolRegistry(ToolHook(EventPlane()))
    # inject a fake backend so the dispatched web_search makes no network call
    register_web_tools(registry, search_backend=_fake_backend(
        [{"title": "A", "url": "https://a.example", "snippet": "s"}]))
    assert "web_search" in registry
    assert "web_fetch" in registry
    # both appear in the structured-selection enum (selectable by the model)
    enum = registry.selection_schema()["properties"]["tool"]["enum"]
    assert "web_search" in enum and "web_fetch" in enum
    # dispatch web_search end-to-end through the hook
    res = asyncio.run(registry.hook.invoke("web_search", query="q", max_results=3))
    assert res.ok is True
    assert res.value["results"][0]["url"] == "https://a.example"
