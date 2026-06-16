"""Tests for the WEB-RESEARCH path (`specialization.research`).

Everything runs against a MOCK tool hook — NO live network (d10 / isolation).
The mock satisfies the small ``ToolInvoker`` contract (``await invoke(name,
**kwargs) -> result`` with ``.ok`` / ``.value``), so the real reactive-tools
HTTP layer is never imported here.

Load-bearing assertions:
- the trace SHAPE is ``{queries, sources, notes, ...}`` and is populated;
- the loop is HARD-BOUNDED — at most ``max_fetches`` fetches and at most
  ``max_calls`` total tool calls (the agentic-RAG ceiling);
- the trace persists to ``specs/<slug>/research_trace.json`` and round-trips.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest

from specialization.research import (
    HowNote,
    ResearchTrace,
    SourceRef,
    derive_queries,
    persist_trace,
    research,
    research_and_persist,
)


# --------------------------------------------------------------------------- #
# A mock ToolHook — records every invocation, returns canned web payloads.
# --------------------------------------------------------------------------- #
@dataclass
class _Result:
    ok: bool
    value: Any = None
    error: str | None = None


class MockHook:
    """In-memory stand-in for reactive_tools.ToolHook (no network)."""

    def __init__(self, *, fail_fetch_urls: set[str] | None = None,
                 fail_search: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail_fetch_urls = fail_fetch_urls or set()
        self.fail_search = fail_search

    async def invoke(self, name: str, /, **kwargs: Any) -> _Result:
        self.calls.append((name, kwargs))
        if name == "web_search":
            if self.fail_search:
                return _Result(ok=False, error="boom")
            q = kwargs.get("query", "")
            n = kwargs.get("max_results", 8)
            # Five canned results per query, deterministic + query-tagged.
            results = [
                {
                    "title": f"Result {i} for {q}",
                    "url": f"https://example.com/{abs(hash(q)) % 1000}/{i}",
                    "snippet": f"How to do step {i}: explanation for {q}.",
                }
                for i in range(5)
            ]
            return _Result(ok=True, value={"query": q, "results": results[:n], "count": min(n, 5)})
        if name == "web_fetch":
            url = kwargs.get("url", "")
            if url in self.fail_fetch_urls:
                return _Result(ok=False, error="fetch failed")
            return _Result(ok=True, value={
                "url": url,
                "final_url": url,
                "status": 200,
                "content_type": "text/html",
                "title": f"Title of {url}",
                "text": f"Detailed how-to content for {url}. " * 60,
                "truncated": False,
                "bytes": 1234,
            })
        return _Result(ok=False, error=f"unknown tool {name!r}")


def _run(coro):
    return asyncio.run(coro)


# ------------------------------ query derivation ------------------------------ #
def test_derive_queries_uses_skill_and_intent():
    qs = derive_queries("write markdown reports", "for a technical audience", max_queries=2)
    assert qs[0] == "how to write markdown reports"
    assert any("technical audience" in q for q in qs)
    assert len(qs) == 2


def test_derive_queries_rejects_empty_skill():
    with pytest.raises(ValueError):
        derive_queries("   ", "intent", max_queries=2)


# ------------------------------- trace shape -------------------------------- #
def test_research_returns_trace_with_queries_sources_notes():
    hook = MockHook()
    trace = _run(research("markdown reports", "concise", hook=hook,
                          max_fetches=3, max_calls=8))

    assert isinstance(trace, ResearchTrace)
    # The three load-bearing fields are present and populated.
    assert trace.queries and all(isinstance(q, str) for q in trace.queries)
    assert trace.sources and all(isinstance(s, SourceRef) for s in trace.sources)
    assert trace.notes and all(isinstance(n, HowNote) for n in trace.notes)

    # to_dict() is JSON-shaped with exactly the documented keys.
    d = trace.to_dict()
    assert {"skill", "intent", "queries", "sources", "notes", "stats",
            "errors", "created_at"} <= set(d)
    assert isinstance(d["sources"][0], dict) and "url" in d["sources"][0]
    assert isinstance(d["notes"][0], dict) and "how" in d["notes"][0]

    # Notes carry both search snippets and page extracts.
    kinds = {n.kind for n in trace.notes}
    assert "search_snippet" in kinds
    assert "page_extract" in kinds

    # Some sources were actually fetched.
    assert any(s.fetched for s in trace.sources)


# ------------------------- the agentic-RAG bound ---------------------------- #
def test_research_respects_max_fetches_hard_cap():
    hook = MockHook()
    trace = _run(research("topic", "intent", hook=hook, max_fetches=2, max_calls=99))
    fetch_calls = [c for c in hook.calls if c[0] == "web_fetch"]
    assert len(fetch_calls) == 2                      # HARD cap honored
    assert trace.stats["fetches"] == 2
    assert sum(1 for s in trace.sources if s.fetched) == 2


def test_research_respects_max_calls_ceiling():
    hook = MockHook()
    # 1 search consumes a call; ceiling of 3 leaves room for 2 fetches.
    trace = _run(research("topic", "", hook=hook, max_fetches=10,
                          max_calls=3, max_queries=1))
    assert len(hook.calls) <= 3
    assert trace.stats["calls"] <= 3
    assert trace.stats["searches"] == 1
    assert trace.stats["fetches"] == 2


def test_research_records_fetch_failure_and_continues():
    # Make the first candidate fetch fail; the loop must not crash and must
    # record the error while still producing a trace.
    hook = MockHook()
    # Discover candidate urls first to know one to fail.
    probe = _run(research("topic", "", hook=hook, max_fetches=0, max_calls=2,
                          max_queries=1))
    bad_url = probe.sources[0].url
    hook2 = MockHook(fail_fetch_urls={bad_url})
    trace = _run(research("topic", "", hook=hook2, max_fetches=3, max_calls=8,
                          max_queries=1))
    assert any("fetch failed" in e or "web_fetch" in e for e in trace.errors)
    # Despite the failure, research still returned a populated trace.
    assert trace.queries and trace.notes


def test_research_search_failure_is_recorded_not_raised():
    hook = MockHook(fail_search=True)
    trace = _run(research("topic", "intent", hook=hook, max_fetches=3, max_calls=8))
    assert trace.errors                       # the search failure was recorded
    assert trace.stats["fetches"] == 0        # nothing to fetch -> no fetches


# ------------------------------ persistence --------------------------------- #
def test_persist_trace_writes_json_under_specs_slug(tmp_path):
    hook = MockHook()
    trace = _run(research("Markdown Reports", "concise", hook=hook))
    path = persist_trace(trace, tmp_path / "specs")

    assert path.exists()
    assert path.name == "research_trace.json"
    # Slugged dir: spaces -> dashes.
    assert path.parent.name == "Markdown-Reports"

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["skill"] == "Markdown Reports"
    assert loaded["queries"] == trace.queries
    assert len(loaded["sources"]) == len(trace.sources)
    assert len(loaded["notes"]) == len(trace.notes)


def test_research_and_persist_round_trip(tmp_path):
    hook = MockHook()
    trace, path = _run(
        research_and_persist("topic", "intent", hook=hook, specs_dir=tmp_path / "specs")
    )
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["skill"] == "topic"
    assert loaded["stats"]["max_fetches"] == trace.stats["max_fetches"]
