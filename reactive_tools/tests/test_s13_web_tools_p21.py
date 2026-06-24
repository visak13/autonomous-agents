"""s13 / P2.1 TOOL-LAYER — web_search operators + region + max_results + source
deny-list (Wikipedia excluded); web_fetch 403/404/timeout DISTINCT + retry/alternate.

Fully OFFLINE: web_search runs against an injected fake :data:`SearchBackend`;
web_fetch runs against a monkeypatched ``_http_get_bytes`` that raises the real
httpx error types — so the deny-list filtering, operator/region/timelimit
pass-through, distinct failure classification, and retry/backoff are all proven
with zero network and zero GPU (d6 — the live proof is separate).
"""
from __future__ import annotations

import httpx
import pytest

from reactive_tools import web_tools as wt
from reactive_tools.tools import ToolInputError
from reactive_tools.web_tools import (
    DEFAULT_DENY_DOMAINS,
    ResultCache,
    make_web_fetch,
    make_web_search,
)


# --------------------------------------------------------------------------- #
# web_search — query OPERATORS + region + max_results pass through verbatim
# --------------------------------------------------------------------------- #


def _recording_backend(rows):
    seen = {}

    def backend(query, max_results, region, timeout, timelimit=None):
        seen["query"] = query
        seen["max_results"] = max_results
        seen["region"] = region
        seen["timelimit"] = timelimit
        return list(rows)[:max_results]

    backend.seen = seen  # type: ignore[attr-defined]
    return backend


def test_web_search_passes_operators_region_maxresults_verbatim():
    backend = _recording_backend(
        [{"title": "A", "url": "https://reuters.com/x", "snippet": "s"}])
    search = make_web_search(backend=backend, cache=ResultCache())
    q = 'Iran Fordow "battle damage assessment" site:gov OR site:iaea.org -opinion'
    out = search(q, max_results=12, region="uk-en")
    # the operator-laden query reaches the backend UNCHANGED (DDG interprets them)
    assert backend.seen["query"] == q
    assert backend.seen["max_results"] == 12
    assert backend.seen["region"] == "uk-en"
    assert out["count"] == 1


def test_web_search_timelimit_passed_through_when_set():
    backend = _recording_backend(
        [{"title": "A", "url": "https://reuters.com/x", "snippet": "s"}])
    search = make_web_search(backend=backend, cache=ResultCache())
    search("breaking news", timelimit="w")
    assert backend.seen["timelimit"] == "w"


# --------------------------------------------------------------------------- #
# web_search — Wikipedia DENY-LIST is tool-enforced (never returned)
# --------------------------------------------------------------------------- #


def test_web_search_excludes_wikipedia_by_default():
    rows = [
        {"title": "Wiki", "url": "https://en.wikipedia.org/wiki/Iran", "snippet": "w"},
        {"title": "Reuters", "url": "https://www.reuters.com/world/iran", "snippet": "r"},
        {"title": "Commons", "url": "https://commons.wikimedia.org/x", "snippet": "c"},
        {"title": "Wiktionary", "url": "https://en.wiktionary.org/x", "snippet": "d"},
    ]
    search = make_web_search(backend=lambda *a, **k: list(rows), cache=ResultCache())
    out = search("iran conflict")
    urls = [r["url"] for r in out["results"]]
    # all three wiki-family domains (incl. subdomains) are dropped at the tool layer
    assert urls == ["https://www.reuters.com/world/iran"]
    assert out["count"] == 1
    assert out["excluded_count"] == 3


def test_web_search_exclude_domains_arg_extends_denylist():
    rows = [
        {"title": "Reddit", "url": "https://www.reddit.com/r/x", "snippet": "a"},
        {"title": "Reuters", "url": "https://reuters.com/y", "snippet": "b"},
    ]
    search = make_web_search(backend=lambda *a, **k: list(rows), cache=ResultCache())
    out = search("topic", exclude_domains=["reddit.com"])
    assert [r["url"] for r in out["results"]] == ["https://reuters.com/y"]
    assert out["excluded_count"] == 1


def test_web_search_denylist_is_configurable_baseline():
    # A host can override the baseline deny-list entirely (the cross-cutting hook).
    rows = [{"title": "Wiki", "url": "https://en.wikipedia.org/x", "snippet": "w"}]
    search = make_web_search(backend=lambda *a, **k: list(rows),
                             cache=ResultCache(), deny_domains=())
    out = search("q")  # empty baseline -> wikipedia is NOT filtered
    assert out["count"] == 1
    assert "wikipedia.org" in DEFAULT_DENY_DOMAINS  # but the default still bans it


# --------------------------------------------------------------------------- #
# web_fetch — DISTINCT failure classification (403 vs 404 vs timeout) + retry
# --------------------------------------------------------------------------- #


def _raise_status(status):
    def _get(url, **k):
        req = httpx.Request("GET", url)
        resp = httpx.Response(status, request=req)
        raise httpx.HTTPStatusError(f"{status}", request=req, response=resp)
    return _get


def test_web_fetch_403_is_distinct_and_not_retried(monkeypatch):
    monkeypatch.setattr(wt, "_http_get_bytes", _raise_status(403))
    out = make_web_fetch(sleep=lambda s: None)("https://site.example/page")
    assert out["ok"] is False
    assert out["error_kind"] == "http_403"
    assert out["status"] == 403
    assert out["attempts"] == 1          # permanent -> tried once, no retry
    assert out["markdown"] == ""         # same key shape as success (safe to read)


def test_web_fetch_404_is_distinct(monkeypatch):
    monkeypatch.setattr(wt, "_http_get_bytes", _raise_status(404))
    out = make_web_fetch(sleep=lambda s: None)("https://site.example/missing")
    assert out["ok"] is False and out["error_kind"] == "http_404" and out["status"] == 404


def test_web_fetch_timeout_is_distinct_and_retried(monkeypatch):
    sleeps: list[float] = []

    def _timeout(url, **k):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(wt, "_http_get_bytes", _timeout)
    out = make_web_fetch(max_retries=2, backoff_base=1.0,
                         sleep=sleeps.append)("https://slow.example")
    assert out["ok"] is False and out["error_kind"] == "timeout"
    assert out["attempts"] == 3                 # 1 try + 2 retries
    assert sleeps == [1.0, 2.0]                 # exponential backoff between retries


def test_web_fetch_5xx_retries_then_succeeds(monkeypatch):
    state = {"n": 0}

    class _Resp:
        url = "https://flaky.example"
        status_code = 200
        headers = {"content-type": "text/plain"}
        _reactive_truncated = False

    def _get(url, **k):
        state["n"] += 1
        if state["n"] < 3:
            req = httpx.Request("GET", url)
            resp = httpx.Response(503, request=req)
            raise httpx.HTTPStatusError("503", request=req, response=resp)
        return (b"recovered body", _Resp())

    monkeypatch.setattr(wt, "_http_get_bytes", _get)
    out = make_web_fetch(max_retries=3, backoff_base=0.0,
                         sleep=lambda s: None)("https://flaky.example")
    assert out["ok"] is True
    assert "recovered body" in out["markdown"]
    assert state["n"] == 3                       # 2 transient 503s + 1 success


def test_web_fetch_network_error_is_distinct(monkeypatch):
    def _conn(url, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(wt, "_http_get_bytes", _conn)
    out = make_web_fetch(max_retries=0, sleep=lambda s: None)("https://down.example")
    assert out["ok"] is False and out["error_kind"] == "network_error"


def test_web_fetch_denied_domain_never_fetched(monkeypatch):
    called = {"n": 0}

    def _should_not_run(url, **k):
        called["n"] += 1
        raise AssertionError("a denied domain must never be fetched")

    monkeypatch.setattr(wt, "_http_get_bytes", _should_not_run)
    out = make_web_fetch()("https://en.wikipedia.org/wiki/Iran")
    assert out["ok"] is False and out["error_kind"] == "denied_domain"
    assert out["attempts"] == 0
    assert called["n"] == 0


def test_web_fetch_redirect_to_denied_domain_is_blocked(monkeypatch):
    # A non-denied URL that 30x-redirects ONTO a deny-listed domain must NOT be
    # fetched — the deny-list holds across redirects, not just on the first URL.
    # Drive the REAL _http_get_bytes redirect loop with a fake httpx.Client.
    class _RedirectResp:
        url = httpx.URL("https://shortlink.example/go")
        is_redirect = True
        headers = {"location": "https://en.wikipedia.org/wiki/Iran"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url):
            return _RedirectResp()

    monkeypatch.setattr(wt.httpx, "Client", _FakeClient)
    # SSRF guard must pass the (public) shortlink host so the loop reaches the hop.
    monkeypatch.setattr(wt, "_assert_public_http_url", lambda u: None)
    out = make_web_fetch(sleep=lambda s: None)("https://shortlink.example/go")
    assert out["ok"] is False
    assert out["error_kind"] == "denied_domain"


def test_web_fetch_blocked_host_is_distinct(monkeypatch):
    def _ssrf(url, **k):
        raise ToolInputError("host is not a public address")

    monkeypatch.setattr(wt, "_http_get_bytes", _ssrf)
    out = make_web_fetch(sleep=lambda s: None)("http://169.254.169.254/latest")
    assert out["ok"] is False and out["error_kind"] == "blocked"


# --------------------------------------------------------------------------- #
# Tool DESCRIPTIONS carry the flow (operators / failure reasons legible)
# --------------------------------------------------------------------------- #


def test_web_tool_descriptions_are_legible():
    sd = wt.WEB_SEARCH_TOOL.description.lower()
    assert "operator" in sd and "site:" in sd and "wikipedia" in sd
    fd = wt.WEB_FETCH_TOOL.description.lower()
    for token in ("403", "404", "timeout", "error_kind", "alternate"):
        assert token in fd


def test_web_fetch_success_carries_ok_true(monkeypatch):
    class _Resp:
        url = "https://ok.example"
        status_code = 200
        headers = {"content-type": "text/plain"}
        _reactive_truncated = False

    monkeypatch.setattr(wt, "_http_get_bytes",
                        lambda url, **k: (b"plain body", _Resp()))
    out = make_web_fetch()("https://ok.example")
    assert out["ok"] is True
    assert out["status"] == 200
    assert out["markdown"] == "plain body"
