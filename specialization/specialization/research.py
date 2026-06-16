"""The WEB-RESEARCH path of the specialization lifecycle (d8).

A specialization is DEFINED in the UI, then *researched* to discover the
"how" of a named skill, and only COMPILED on user approval. This module is the
**research** step: given a skill name and an intent, it runs a BOUNDED loop over
the FREE, KEY-LESS web tools (DuckDuckGo HTML search + page fetch/extract) built
in the reactive-tools layer, and returns a structured, replayable *research
trace* — the queries it issued, the source URLs it saw/read, and the extracted
"how" notes a later compile step distills into a :class:`CompiledSpec` body.

Decisions honored
------------------
- d2  — purely in-process. Research drives the in-process :class:`ToolHook`
  (asyncio); no broker/pool, no subprocess, no Claude. ``research`` is a
  coroutine because the hook entrypoint is ``async``.
- d6  — web access is FREE and key-LESS. We reuse ``reactive_tools``'
  ``web_search`` (DuckDuckGo HTML) and ``web_fetch`` (public GET + text
  extract) UNCHANGED, invoked by name through the single :class:`ToolHook` so
  every call + result still flows on the reactive event plane. No API keys.
- d8  — NO shell-command anything; trace persistence is plain ``pathlib`` JSON.
- agentic-RAG BOUND (from the RAG specialist) — the loop is HARD-BOUNDED: at
  most ``max_fetches`` page fetches (default 3) AND at most ``max_calls`` total
  tool invocations (search + fetch). A runaway agentic crawl is impossible by
  construction; the caps are explicit parameters recorded into the trace.

Context-scoping (d10)
---------------------
``reactive_tools`` is imported LAZILY (only inside :func:`build_research_hook`),
so importing this module — and the unit test that drives :func:`research`
against a *mock* hook — needs NO network and no reactive-tools import. The
research primitive depends only on the small ToolHook contract (``await
hook.invoke(name, **kwargs) -> result`` with ``.ok`` / ``.value``), which a
stub can satisfy in-memory.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

# Tool names registered by reactive_tools.register_core_tools — referenced by
# name only (the planner/agent never carries tool bodies; d10).
TOOL_WEB_SEARCH = "web_search"
TOOL_WEB_FETCH = "web_fetch"

# Default bounds (the agentic-RAG ceiling). HARD caps, overridable per call.
DEFAULT_MAX_FETCHES = 3          # at most this many web_fetch calls
DEFAULT_MAX_CALLS = 8            # at most this many TOTAL tool invocations
DEFAULT_MAX_QUERIES = 2          # at most this many distinct search queries
DEFAULT_RESULTS_PER_QUERY = 6    # search breadth per query
DEFAULT_NOTE_CHARS = 800         # how much extracted page text to keep per note


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# The ToolHook contract this module needs (a Protocol so a mock satisfies it)
# --------------------------------------------------------------------------- #
class _InvokeResult(Protocol):
    """The shape of a :class:`reactive_tools.ToolResult` we rely on."""

    ok: bool
    value: Any
    error: Optional[str]


class ToolInvoker(Protocol):
    """Minimal hook contract: ``await invoke(name, **kwargs) -> _InvokeResult``.

    The real :class:`reactive_tools.ToolHook` satisfies this, and so does any
    in-memory stub — which is exactly how the unit test exercises the loop with
    NO live network (d10 / test isolation)."""

    async def invoke(self, name: str, /, **kwargs: Any) -> _InvokeResult: ...


# --------------------------------------------------------------------------- #
# Trace data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SourceRef:
    """One source URL the research surfaced (from search and/or fetched)."""

    url: str
    title: str = ""
    via_query: str = ""
    fetched: bool = False


@dataclass(frozen=True)
class HowNote:
    """An extracted "how" note: where it came from + the distilled snippet."""

    source: str           # the URL (or "search:<query>" for a search snippet)
    kind: str             # "search_snippet" | "page_extract"
    how: str              # the note text (snippet, or trimmed page extract)
    title: str = ""


@dataclass
class ResearchTrace:
    """The replayable record of one :func:`research` run.

    The three load-bearing fields (asserted by the unit test and consumed by a
    later compile step) are ``queries``, ``sources`` and ``notes``."""

    skill: str
    intent: str
    queries: list[str] = field(default_factory=list)
    sources: list[SourceRef] = field(default_factory=list)
    notes: list[HowNote] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        """A plain JSON-serializable dict (for persistence / inspection)."""
        return {
            "skill": self.skill,
            "intent": self.intent,
            "queries": list(self.queries),
            "sources": [asdict(s) for s in self.sources],
            "notes": [asdict(n) for n in self.notes],
            "stats": dict(self.stats),
            "errors": list(self.errors),
            "created_at": self.created_at,
        }


# --------------------------------------------------------------------------- #
# Query derivation
# --------------------------------------------------------------------------- #
def derive_queries(skill: str, intent: str, max_queries: int) -> list[str]:
    """Derive up to ``max_queries`` search queries from skill + intent.

    Deterministic (no LLM here — phi's autonomous planning lives in the planner
    layer; this primitive is the reusable mechanical research step). The first
    query targets the "how"; the second folds in the intent for specificity."""
    skill = (skill or "").strip()
    intent = (intent or "").strip()
    if not skill:
        raise ValueError("research skill must be a non-empty string")
    candidates = [f"how to {skill}"]
    if intent:
        candidates.append(f"{skill} {intent} guide tutorial")
    else:
        candidates.append(f"{skill} guide tutorial best practices")
    # De-dupe while preserving order, then bound.
    seen: set[str] = set()
    out: list[str] = []
    for q in candidates:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[: max(1, max_queries)]


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


# --------------------------------------------------------------------------- #
# The bounded research loop
# --------------------------------------------------------------------------- #
async def research(
    skill: str,
    intent: str = "",
    *,
    hook: ToolInvoker,
    max_fetches: int = DEFAULT_MAX_FETCHES,
    max_calls: int = DEFAULT_MAX_CALLS,
    max_queries: int = DEFAULT_MAX_QUERIES,
    results_per_query: int = DEFAULT_RESULTS_PER_QUERY,
    note_chars: int = DEFAULT_NOTE_CHARS,
) -> ResearchTrace:
    """Research the "how" of ``skill`` over the free web tools; return a trace.

    Drives the in-process :class:`ToolHook` (``hook``) — invoking ``web_search``
    then ``web_fetch`` BY NAME so every call flows on the reactive plane. The
    loop is HARD-BOUNDED (agentic-RAG ceiling): no more than ``max_fetches``
    page fetches and no more than ``max_calls`` total tool invocations, whichever
    binds first. Tool failures are recorded into ``trace.errors`` and the loop
    continues (best-effort research, never a crash).

    Returns a :class:`ResearchTrace` with the issued ``queries``, the surfaced
    ``sources`` (URLs, with ``fetched`` marking the ones actually read), and the
    extracted "how" ``notes`` (search snippets + page extracts)."""
    trace = ResearchTrace(skill=skill.strip(), intent=intent.strip())
    calls = 0
    fetches = 0
    searches = 0

    queries = derive_queries(skill, intent, max_queries)

    # ---- phase 1: SEARCH to gather candidate sources --------------------- #
    # candidates: ordered, de-duped by url; each carries its discovery query.
    candidates: list[SourceRef] = []
    seen_urls: set[str] = set()
    for query in queries:
        if calls >= max_calls:
            break
        trace.queries.append(query)
        res = await hook.invoke(
            TOOL_WEB_SEARCH, query=query, max_results=results_per_query
        )
        calls += 1
        searches += 1
        if not getattr(res, "ok", False):
            trace.errors.append(f"web_search({query!r}) failed: {getattr(res, 'error', '?')}")
            continue
        payload = res.value or {}
        for r in payload.get("results", []):
            url = (r.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            title = (r.get("title") or "").strip()
            snippet = (r.get("snippet") or "").strip()
            candidates.append(SourceRef(url=url, title=title, via_query=query))
            if snippet:
                trace.notes.append(
                    HowNote(
                        source=f"search:{query}",
                        kind="search_snippet",
                        how=_clip(snippet, note_chars),
                        title=title,
                    )
                )

    # ---- phase 2: FETCH the top candidates (hard-capped) ----------------- #
    fetched_sources: list[SourceRef] = []
    for cand in candidates:
        if fetches >= max_fetches or calls >= max_calls:
            break
        res = await hook.invoke(TOOL_WEB_FETCH, url=cand.url)
        calls += 1
        fetches += 1
        if not getattr(res, "ok", False):
            trace.errors.append(f"web_fetch({cand.url!r}) failed: {getattr(res, 'error', '?')}")
            fetched_sources.append(cand)  # still record we tried it (unfetched)
            continue
        payload = res.value or {}
        page_title = (payload.get("title") or cand.title or "").strip()
        text = payload.get("text") or ""
        fetched_sources.append(
            SourceRef(url=cand.url, title=page_title, via_query=cand.via_query, fetched=True)
        )
        if text.strip():
            trace.notes.append(
                HowNote(
                    source=cand.url,
                    kind="page_extract",
                    how=_clip(text, note_chars),
                    title=page_title,
                )
            )

    # sources = the fetched ones first (with fetched=True), then any candidates
    # we surfaced but did not read (so the trace shows the full discovery set).
    fetched_urls = {s.url for s in fetched_sources}
    trace.sources = fetched_sources + [c for c in candidates if c.url not in fetched_urls]

    trace.stats = {
        "searches": searches,
        "fetches": fetches,
        "calls": calls,
        "candidates": len(candidates),
        "notes": len(trace.notes),
        "max_fetches": max_fetches,
        "max_calls": max_calls,
    }
    return trace


# --------------------------------------------------------------------------- #
# Persistence — replayable / inspectable trace under specs/<name>/
# --------------------------------------------------------------------------- #
def _slug(name: str) -> str:
    """Filesystem-safe stem for a spec name (matches registry._slug semantics)."""
    safe = (name or "").strip().replace(" ", "-").replace("/", "-").replace("\\", "-")
    if not safe or safe in (".", ".."):
        raise ValueError(f"invalid spec name {name!r}")
    return safe


def persist_trace(trace: ResearchTrace, specs_dir: str | Path, *,
                  filename: str = "research_trace.json",
                  subdir: Optional[str] = None) -> Path:
    """Persist ``trace`` to ``specs_dir/<slug>/research_trace.json``.

    Mirrors the registry's on-disk layout (``specs/<name>/…``) so the research
    trace sits beside the spec it will compile into, and is replayable /
    inspectable by hand (plain JSON, d8 — no shell, no service). Returns the
    path written.

    ``subdir`` overrides which value the per-spec directory is slugged from.
    The compiled spec stamps ``research_trace_ref: specs/<spec-NAME>/…`` (see
    ``engine._trace_ref``), but ``trace.skill`` is the research *subject* (the
    spec *description*, which drives the search queries) — so defaulting the
    directory to ``trace.skill`` made the persisted trace land under the
    description slug while the pointer referenced the name slug (they did not
    resolve). Callers that have the spec NAME pass it here so the trace lands
    exactly where ``research_trace_ref`` points."""
    spec_dir = Path(specs_dir) / _slug(subdir if subdir is not None else trace.skill)
    spec_dir.mkdir(parents=True, exist_ok=True)
    out = spec_dir / filename
    out.write_text(
        json.dumps(trace.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


async def research_and_persist(
    skill: str,
    intent: str,
    *,
    hook: ToolInvoker,
    specs_dir: str | Path,
    **kwargs: Any,
) -> tuple[ResearchTrace, Path]:
    """Convenience: run :func:`research` then :func:`persist_trace`.

    Returns ``(trace, path)``. Keyword args are forwarded to :func:`research`
    (e.g. ``max_fetches``, ``max_calls``)."""
    trace = await research(skill, intent, hook=hook, **kwargs)
    path = persist_trace(trace, specs_dir)
    return trace, path


# --------------------------------------------------------------------------- #
# Real-hook factory (LAZY reactive_tools import — keeps this module net-free
# to import, and the unit test mock-only) (d10)
# --------------------------------------------------------------------------- #
def build_research_hook(*, file_base: Any = None, http_timeout: float = 20.0):
    """Build a real :class:`ToolHook` pre-loaded with the core web tools.

    Imports ``reactive_tools`` lazily so merely importing :mod:`research` (and
    unit-testing it against a stub) never pulls in the HTTP tool layer. Use this
    for a LIVE research run; the bounded loop then hits the real free no-key web
    tools (d6)."""
    from reactive_tools import EventPlane, build_default_hook  # lazy (d10)

    plane = EventPlane()
    return build_default_hook(plane, file_base=file_base, http_timeout=http_timeout)


__all__ = [
    "research",
    "research_and_persist",
    "persist_trace",
    "derive_queries",
    "build_research_hook",
    "ResearchTrace",
    "SourceRef",
    "HowNote",
    "ToolInvoker",
    "TOOL_WEB_SEARCH",
    "TOOL_WEB_FETCH",
    "DEFAULT_MAX_FETCHES",
    "DEFAULT_MAX_CALLS",
]
