"""WEB TOOLS — ``web_search`` + ``web_fetch`` as growable-registry entries (s2/a2).

Two concrete tools that plug into the a1 SCAFFOLD
(:class:`~reactive_tools.tool_registry.GrowableToolRegistry`) — *each is exactly
ONE* :class:`~reactive_tools.tool_registry.ToolDef` (name + description + a
Pydantic args model + handler). The registry core is untouched; adding these is
the single ``registry.add(ToolDef(...))`` growth point (outcome o1).

Per decision **d4** (web-verified June 2026) and the Round-3 blueprint (§RC5):

- ``web_search`` = **DuckDuckGo via the** `ddgs` **library** — FULLY FREE, no API
  key, no paid provider. The legacy raw DDG-HTML scrape
  (``tools.py:make_web_search`` / ``_DDGResultParser``) is SUPERSEDED by this
  maintained library, which already handles DDG's redirect-decoding, backends and
  HTML churn. To survive rate limits under deep-research load (≈10 rounds × many
  queries) a small in-process **TTL result cache** + **exponential backoff** wrap
  every live call.

- ``web_fetch`` = **httpx retrieval + Trafilatura extraction to MARKDOWN**.
  Trafilatura (F1 0.958, native markdown output) replaces the stdlib
  ``html.parser`` text dump so nodes get clean, structured markdown (headings /
  lists / links preserved) instead of a flat blob.

Swappability (d1 — growable registry): ``web_search`` runs behind a small
:data:`SearchBackend` adapter. The default + only concrete adapter today is
:func:`ddgs_backend` (free DuckDuckGo); a paid provider (Tavily / Brave / Exa …)
could be added LATER as another backend WITHOUT touching the registry entry, the
caller, or any node — exactly the "keep web_search behind the registry interface
so a paid provider can be swapped in later" mandate.

Security is inherited, not reinvented: ``web_fetch`` reuses the reviewed SSRF
guard (:func:`~reactive_tools.tools._assert_public_http_url`) — http/https only,
private/loopback/link-local/reserved/multicast hosts rejected, redirects followed
MANUALLY and re-validated per hop, response size hard-bounded — from the a-series
file tools. ``d2`` holds: the only I/O is the tools' own outbound HTTP; no
broker/pool/subprocess.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Callable, Optional, Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field

import trafilatura

from .tool_registry import ToolDef
from .tools import (
    DEFAULT_FETCH_MAX_BYTES,
    DEFAULT_MAX_REDIRECTS,
    ToolInputError,
    _assert_public_http_url,
    _extract_text,
    _extract_title,
)

# ddgs raises typed exceptions we back off on; import lazily-tolerant so the
# module still imports if the lib layout shifts (the live path will surface it).
try:  # pragma: no cover - exercised by the live a6 proof
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
except Exception:  # noqa: BLE001 - keep import-time resilient
    DDGS = None  # type: ignore[assignment]

    class DDGSException(Exception):  # type: ignore[no-redef]
        ...

    class RatelimitException(DDGSException):  # type: ignore[no-redef]
        ...

    class TimeoutException(DDGSException):  # type: ignore[no-redef]
        ...


_USER_AGENT = "ReactiveAgents/0.1 (+in-process; httpx)"

# Defaults tuned for deep-research load: a short-lived cache so the SAME query
# fired across rounds is served once, and a bounded backoff so a transient DDG
# rate-limit self-heals instead of failing the node.
DEFAULT_SEARCH_TTL_SECONDS = 300.0   # 5 min — long enough to dedupe a research run
DEFAULT_CACHE_MAX_ENTRIES = 256
DEFAULT_MAX_RETRIES = 4              # 1 try + up to 4 backoff retries
DEFAULT_BACKOFF_BASE = 1.0          # seconds; doubles per retry (1,2,4,8)
DEFAULT_SEARCH_REGION = "us-en"

# --------------------------------------------------------------------------- #
# Source-domain DENY-LIST (cross-cutting source policy, tool-ENFORCED)
# --------------------------------------------------------------------------- #
# d131/d133 (user, EMPHATIC): Wikipedia is a hard, non-negotiable deny — never
# fetched (not even to mine primary sources) and never returned as a search
# result, and the same applies to its sibling Wikimedia/Wiktionary properties.
# This is the FIRST, baseline entry of a GENERAL, EXTENSIBLE source-policy
# mechanism: a deny-list is honored AT THE TOOL LAYER (both web_search and
# web_fetch), so a cross-cutting concern like "never use Wikipedia" is ENFORCED
# by the tools rather than merely requested in a prompt the model may ignore.
# Hosts are matched on the registrable-domain suffix, so every subdomain
# (en.wikipedia.org, simple.m.wikipedia.org, commons.wikimedia.org, …) is
# covered. Callers extend the baseline per-run via the shape/spec cross-cutting
# concerns (web_search ``exclude_domains`` arg / the ``deny_domains`` builder
# param) — the mechanism is generic; Wikipedia is just the first banned domain.
DEFAULT_DENY_DOMAINS: tuple[str, ...] = (
    "wikipedia.org",
    "wikimedia.org",
    "wiktionary.org",
)


def _host_of(url: str) -> str:
    """The lowercased hostname of ``url`` (empty string if unparseable)."""
    try:
        return (urlsplit(url).hostname or "").lower()
    except (ValueError, TypeError):
        return ""


def _normalise_deny(domains: Any) -> set[str]:
    """A clean lowercased set of bare registrable domains from any iterable.

    Tolerates ``None``, a single string, or a list; strips scheme/leading dots
    so ``'https://wikipedia.org/'`` and ``'.wikipedia.org'`` both normalise to
    ``'wikipedia.org'``."""
    if not domains:
        return set()
    if isinstance(domains, str):
        domains = [domains]
    out: set[str] = set()
    for d in domains:
        if not d:
            continue
        d = str(d).strip().lower()
        # accept a bare domain OR a full URL OR a leading-dot suffix
        host = _host_of(d) if "://" in d else d.lstrip(".")
        host = host.split("/")[0].strip()
        if host:
            out.add(host)
    return out


def _domain_denied(url: str, deny: set[str]) -> bool:
    """True when ``url``'s host equals or is a subdomain of any denied domain."""
    if not deny:
        return False
    host = _host_of(url)
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in deny)


# --------------------------------------------------------------------------- #
# TTL + LRU result cache (in-process, deep-research dedupe)
# --------------------------------------------------------------------------- #


class ResultCache:
    """A tiny in-process TTL + LRU cache for search results.

    Keyed by ``(query, max_results, region)``. Entries expire after ``ttl``
    seconds; the map is bounded to ``max_entries`` (oldest evicted). This is the
    rate-limit relief valve: a deep-research shape that re-issues the same query
    across rounds pays ONE live DDG call, not N. Purely in-memory (no disk, d2);
    ``clock`` is injectable so tests drive expiry deterministically.
    """

    def __init__(
        self,
        ttl: float = DEFAULT_SEARCH_TTL_SECONDS,
        max_entries: int = DEFAULT_CACHE_MAX_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl = ttl
        self.max_entries = max_entries
        self._clock = clock
        self._store: "OrderedDict[Any, tuple[float, Any]]" = OrderedDict()

    def get(self, key: Any) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if self._clock() >= expires_at:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)  # LRU touch
        return value

    def put(self, key: Any, value: Any) -> None:
        self._store[key] = (self._clock() + self.ttl, value)
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)  # evict oldest

    def __len__(self) -> int:
        return len(self._store)


# --------------------------------------------------------------------------- #
# Search backend adapter (provider-swappable; default = free DuckDuckGo/ddgs)
# --------------------------------------------------------------------------- #


class SearchBackend(Protocol):
    """A pluggable search provider. Returns a list of ``{title, url, snippet}``.

    The default :func:`ddgs_backend` is free DuckDuckGo; a paid provider could be
    dropped in later as another ``SearchBackend`` with NO change to the registry
    entry, the caller, or any node (d1 growable registry). ``timelimit`` (a DDG
    recency window: 'd'/'w'/'m'/'y') is passed only when set, so legacy backends
    with the 4-arg signature keep working unchanged."""

    def __call__(self, query: str, max_results: int, region: str, timeout: float,
                 timelimit: Optional[str] = None) -> list[dict[str, str]]:
        ...


def ddgs_backend(query: str, max_results: int, region: str, timeout: float,
                 timelimit: Optional[str] = None) -> list[dict[str, str]]:
    """Default search backend — free, key-LESS DuckDuckGo via the ``ddgs`` lib.

    Normalises ddgs' ``{title, href, body}`` rows to the tool's stable
    ``{title, url, snippet}`` contract. ``timelimit`` ('d'/'w'/'m'/'y') restricts
    results to the last day/week/month/year when set. Raises the ddgs typed
    exceptions (``RatelimitException`` / ``TimeoutException``) straight up so the
    caller's backoff can react to them."""
    if DDGS is None:  # pragma: no cover - only if the lib failed to import
        raise ToolInputError("ddgs library is not available; cannot run web_search")
    with DDGS(timeout=timeout) as ddgs:
        rows = ddgs.text(query, region=region, max_results=max_results,
                         timelimit=timelimit)
    out: list[dict[str, str]] = []
    for r in rows or []:
        url = r.get("href") or r.get("url") or ""
        if not url:
            continue
        out.append({
            "title": (r.get("title") or "").strip(),
            "url": url,
            "snippet": (r.get("body") or r.get("snippet") or "").strip(),
        })
    return out


# --------------------------------------------------------------------------- #
# web_search — ddgs + cache + exponential backoff
# --------------------------------------------------------------------------- #


class WebSearchArgs(BaseModel):
    """Args for :data:`WEB_SEARCH_TOOL`."""

    query: str = Field(
        ...,
        description=(
            "the search query. SUPPORTS DuckDuckGo OPERATORS — compose them to "
            "find RELIABLE PRIMARY sources fast: \"exact phrase\" (double quotes) "
            "for a verbatim match; site:domain to target one source "
            "(e.g. site:reuters.com) or -site:domain to exclude one; "
            "intitle:word to require a word in the page title; filetype:pdf for "
            "documents/reports; the uppercase OR for alternatives "
            "(sanctions OR embargo); a leading - to exclude a term (-opinion "
            "-blog). Example: 'Iran Fordow strike \"battle damage assessment\" "
            "site:gov OR site:iaea.org -opinion'."))
    max_results: int = Field(
        8, ge=1, le=25,
        description="maximum number of ranked results to return (1-25; raise it for a broad survey, lower it to stay focused)")
    region: str = Field(
        DEFAULT_SEARCH_REGION,
        description="DuckDuckGo region code, e.g. 'us-en', 'uk-en', 'wt-wt' (worldwide)")
    timelimit: Optional[str] = Field(
        None,
        description=(
            "OPTIONAL recency filter: 'd' (past day), 'w' (past week), 'm' (past "
            "month), 'y' (past year). Use it for fast-moving/current-events "
            "topics so stale pages are excluded; leave unset for background/"
            "historical questions."))
    exclude_domains: list[str] = Field(
        default_factory=list,
        description=(
            "OPTIONAL extra source domains to EXCLUDE from results (e.g. "
            "['reddit.com','medium.com'] to drop social/opinion sources). These "
            "ADD to the always-on baseline deny-list; Wikipedia/Wikimedia are "
            "ALREADY excluded by default and never returned."))


def make_web_search(
    *,
    backend: Optional[SearchBackend] = None,
    cache: Optional[ResultCache] = None,
    timeout: float = 20.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    sleep: Callable[[float], None] = time.sleep,
    deny_domains: Any = DEFAULT_DENY_DOMAINS,
) -> Callable[..., dict[str, Any]]:
    """Build the ``web_search`` handler (free DuckDuckGo, cache + backoff).

    ``backend`` defaults to :func:`ddgs_backend`; swap it for a paid provider
    later with no other change. ``cache`` is a shared :class:`ResultCache` (one is
    created if not given) so repeated queries within ``ttl`` are served without a
    live call. ``sleep`` is injectable so tests exercise the backoff path without
    real waits. ``deny_domains`` is the always-on baseline source deny-list
    (default: Wikipedia/Wikimedia/Wiktionary per d131/d133) — results from these
    hosts are dropped at the tool layer; a host extends it per-call via the
    ``exclude_domains`` arg (the shape/spec cross-cutting source policy)."""
    _backend: SearchBackend = backend or ddgs_backend
    _cache = cache if cache is not None else ResultCache()
    _baseline_deny = _normalise_deny(deny_domains)

    def web_search(query: str, max_results: int = 8,
                   region: str = DEFAULT_SEARCH_REGION,
                   timelimit: Optional[str] = None,
                   exclude_domains: Any = None) -> dict[str, Any]:
        """Free, key-LESS web search via DuckDuckGo (``ddgs``), cached + backed-off.

        Returns ``{"query", "results": [{title, url, snippet}], "count",
        "cached", "region", "excluded_count"}``. Supports DuckDuckGo query
        OPERATORS (phrase/site:/OR/-/intitle:/filetype:), a ``timelimit`` recency
        window, and a tool-ENFORCED source deny-list (baseline Wikipedia + any
        ``exclude_domains``) so banned domains never reach the model. A repeated
        query is served from the in-process TTL cache (``cached=True``); a
        transient DDG rate-limit is retried with exponential backoff before
        failing."""
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError("query must be a non-empty string")
        max_results = max(1, min(int(max_results), 25))
        deny = _baseline_deny | _normalise_deny(exclude_domains)
        # Cache key carries every arg that changes the result set (incl. the
        # effective deny-list + recency window) so distinct calls don't collide.
        key = (query.strip(), max_results, region, timelimit,
               tuple(sorted(deny)))
        cached = _cache.get(key)
        if cached is not None:
            return {"query": query, "results": cached, "count": len(cached),
                    "cached": True, "region": region,
                    "excluded_count": 0}

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                if timelimit:
                    raw_results = _backend(query.strip(), max_results, region,
                                           timeout, timelimit=timelimit)
                else:
                    raw_results = _backend(query.strip(), max_results, region,
                                           timeout)
                # Tool-ENFORCED deny-list: drop banned-domain rows BEFORE the
                # model ever sees them (defense-in-depth — independent of whether
                # the upstream provider honored a -site: operator).
                results = [r for r in (raw_results or [])
                           if not _domain_denied(r.get("url", ""), deny)]
                excluded = len(raw_results or []) - len(results)
                _cache.put(key, results)
                return {"query": query, "results": results, "count": len(results),
                        "cached": False, "region": region,
                        "excluded_count": excluded}
            except (RatelimitException, TimeoutException) as exc:
                # Transient — back off and retry. The cache makes the steady state
                # cheap; the backoff makes the spike survivable (deep-research load).
                last_exc = exc
                if attempt < max_retries:
                    sleep(backoff_base * (2 ** attempt))
                    continue
                break
            except DDGSException as exc:  # non-transient ddgs failure
                raise ToolInputError(f"web_search failed: {exc}") from exc
        raise ToolInputError(
            f"web_search rate-limited after {max_retries + 1} attempts: {last_exc}")

    return web_search


_WEB_SEARCH_DESC = (
    "Find SOURCES for a question. Free, no-key web search (DuckDuckGo via ddgs). "
    "Use it FIRST to IDENTIFY reliable primary sources before fetching: write a "
    "focused query with OPERATORS (\"exact phrase\", site:domain / -site:domain, "
    "OR, leading - to exclude, intitle:, filetype:pdf), set `timelimit` "
    "(d/w/m/y) for current events, and `exclude_domains` to drop unwanted "
    "sources. Returns ranked {title,url,snippet} results plus an `excluded_count` "
    "for deny-listed hits; Wikipedia/Wikimedia are ALWAYS excluded. Cached + "
    "rate-limit backoff. Then READ the most promising results with web_fetch.")


WEB_SEARCH_TOOL = ToolDef(
    name="web_search",
    description=_WEB_SEARCH_DESC,
    args_model=WebSearchArgs,
    handler=make_web_search(),
)


# --------------------------------------------------------------------------- #
# web_fetch — httpx retrieval + Trafilatura markdown extraction
# --------------------------------------------------------------------------- #


class WebFetchArgs(BaseModel):
    """Args for :data:`WEB_FETCH_TOOL`."""

    url: str = Field(..., description="the public http/https URL to fetch")
    max_bytes: int = Field(
        DEFAULT_FETCH_MAX_BYTES, ge=1,
        description="hard cap on response bytes read (defends against oversize/bomb responses)")
    max_redirects: int = Field(
        DEFAULT_MAX_REDIRECTS, ge=0, le=10,
        description="maximum redirect hops to follow (each hop is SSRF-revalidated)")


def _http_get_bytes(url: str, *, timeout: float, max_bytes: int,
                    max_redirects: int,
                    deny: Optional[set[str]] = None) -> tuple[bytes, "httpx.Response"]:
    """Fetch ``url`` as bytes, SSRF-guarded, deny-list-guarded and size-bounded.

    Redirects are followed MANUALLY (httpx auto-follow disabled) so EVERY hop's
    host is validated by :func:`_assert_public_http_url` BEFORE a request is
    issued to it — a public URL can't bounce the fetch onto an internal/metadata
    host. EVERY hop is ALSO checked against ``deny`` so a non-denied URL cannot
    redirect the fetch ONTO a deny-listed domain (e.g. a shortlink/AMP hop that
    lands on Wikipedia) — the deny-list holds on every path, not just the first.
    Reused, reviewed security primitive from the file/web a-series."""
    headers = {"User-Agent": _USER_AGENT, "Accept": "text/html,*/*;q=0.8"}
    current = url
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=False) as client:
        for _hop in range(max_redirects + 1):
            _assert_public_http_url(current)  # validate BEFORE issuing the request
            if deny and _domain_denied(current, deny):
                raise ToolInputError(
                    f"redirect landed on deny-listed domain {_host_of(current)!r}; "
                    "blocked (the source deny-list holds across redirects)")
            with client.stream("GET", current) as resp:
                location = resp.headers.get("location")
                if resp.is_redirect and location:
                    current = str(resp.url.join(location))
                    continue
                resp.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                truncated = False
                for chunk in resp.iter_bytes():
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= max_bytes:
                        truncated = True
                        break
                raw = b"".join(chunks)[:max_bytes]
                resp._reactive_truncated = truncated  # type: ignore[attr-defined]
                return raw, resp
    raise ToolInputError(f"too many redirects (> {max_redirects}) starting from {url!r}")


# Status codes worth a retry (transient): rate-limit + 5xx server errors. A 403
# (forbidden), 404 (not found), 410 (gone) etc. are PERMANENT for this URL —
# retrying wastes time; the agent should pick an alternate source instead.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _fetch_failure(url: str, *, status: Optional[int], error_kind: str,
                   error: str, attempts: int) -> dict[str, Any]:
    """A structured web_fetch FAILURE — same key shape as success (so existing
    readers of ``markdown``/``status`` don't crash) plus ``ok=False`` and a
    DISTINCT ``error_kind`` + human ``error`` so the agent knows EXACTLY why the
    fetch failed (forbidden vs not-found vs timeout vs blocked) and can react —
    try an ALTERNATE source rather than re-reading a dead URL."""
    return {
        "url": url,
        "final_url": url,
        "status": status,
        "content_type": "",
        "title": "",
        "markdown": "",
        "extracted": False,
        "truncated": False,
        "bytes": 0,
        "ok": False,
        "error_kind": error_kind,
        "error": error,
        "attempts": attempts,
    }


def make_web_fetch(*, timeout: float = 20.0,
                   max_retries: int = 2,
                   backoff_base: float = DEFAULT_BACKOFF_BASE,
                   sleep: Callable[[float], None] = time.sleep,
                   deny_domains: Any = DEFAULT_DENY_DOMAINS,
                   ) -> Callable[..., dict[str, Any]]:
    """Build the ``web_fetch`` handler (httpx retrieval + Trafilatura markdown).

    Transient failures (timeout / connection error / HTTP 429 / 5xx) are RETRIED
    with exponential backoff (like web_search); permanent ones (403/404/410, a
    deny-listed domain, an SSRF-blocked host) return immediately with a DISTINCT
    ``error_kind`` so the agent moves to an alternate source. ``deny_domains`` is
    the always-on baseline source deny-list (default Wikipedia) — a denied URL is
    NEVER fetched. ``sleep`` is injectable so tests exercise backoff without real
    waits."""
    _deny = _normalise_deny(deny_domains)

    def web_fetch(url: str, max_bytes: int = DEFAULT_FETCH_MAX_BYTES,
                  max_redirects: int = DEFAULT_MAX_REDIRECTS) -> dict[str, Any]:
        """Fetch a public URL and extract clean content as MARKDOWN (Trafilatura).

        On SUCCESS returns ``{"ok": True, "url", "final_url", "status",
        "content_type", "title", "markdown", "extracted", "truncated", "bytes"}``;
        ``markdown`` preserves headings/lists/links; ``extracted`` is True when
        Trafilatura produced article markdown, False on a plain-text fallback.
        On FAILURE returns ``{"ok": False, "error_kind", "error", "status", ...}``
        where ``error_kind`` is one of ``denied_domain`` / ``http_403`` /
        ``http_404`` / ``http_<code>`` / ``timeout`` / ``network_error`` /
        ``blocked`` / ``too_many_redirects`` — so the agent can DISTINGUISH a
        forbidden page from a missing one from a timeout and try an ALTERNATE
        source. Transient errors are retried with backoff first. SSRF-guarded +
        size-bounded (see module header)."""
        # Tool-ENFORCED deny-list — a banned domain (Wikipedia) is never fetched.
        if _domain_denied(url, _deny):
            return _fetch_failure(
                url, status=None, error_kind="denied_domain",
                error=(f"{_host_of(url)} is on the source deny-list and must not "
                       "be fetched or cited; use a different primary source."),
                attempts=0)

        last_kind = "network_error"
        last_err = ""
        last_status: Optional[int] = None
        for attempt in range(max_retries + 1):
            try:
                raw, resp = _http_get_bytes(
                    url, timeout=timeout, max_bytes=int(max_bytes),
                    max_redirects=int(max_redirects), deny=_deny)
            except httpx.HTTPStatusError as exc:
                last_status = exc.response.status_code
                last_kind = f"http_{last_status}"
                last_err = f"HTTP {last_status} fetching {url}"
                if last_status in _RETRYABLE_STATUS and attempt < max_retries:
                    sleep(backoff_base * (2 ** attempt))
                    continue
                return _fetch_failure(url, status=last_status,
                                      error_kind=last_kind, error=last_err,
                                      attempts=attempt + 1)
            except httpx.TimeoutException as exc:
                last_kind, last_err = "timeout", f"timed out fetching {url}: {exc}"
                if attempt < max_retries:
                    sleep(backoff_base * (2 ** attempt))
                    continue
                return _fetch_failure(url, status=None, error_kind="timeout",
                                      error=last_err, attempts=attempt + 1)
            except httpx.RequestError as exc:
                last_kind = "network_error"
                last_err = f"network error fetching {url}: {exc}"
                if attempt < max_retries:
                    sleep(backoff_base * (2 ** attempt))
                    continue
                return _fetch_failure(url, status=None, error_kind="network_error",
                                      error=last_err, attempts=attempt + 1)
            except ToolInputError as exc:
                # SSRF-blocked host / too-many-redirects / redirect-to-denied —
                # permanent, no retry.
                msg = str(exc)
                if "deny-list" in msg:
                    kind = "denied_domain"
                elif "too many redirects" in msg:
                    kind = "too_many_redirects"
                else:
                    kind = "blocked"
                return _fetch_failure(url, status=None, error_kind=kind,
                                      error=str(exc), attempts=attempt + 1)

            # ---- success: extract markdown -------------------------------- #
            truncated = bool(getattr(resp, "_reactive_truncated", False))
            content_type = resp.headers.get("content-type", "")
            final_url = str(resp.url)

            encoding = "utf-8"
            if "charset=" in content_type:
                encoding = content_type.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
            try:
                body = raw.decode(encoding, errors="replace")
            except (LookupError, ValueError):
                body = raw.decode("utf-8", errors="replace")

            is_html = "html" in content_type or (not content_type and "<" in body[:200])
            markdown: Optional[str] = None
            extracted = False
            title = ""
            if is_html:
                # Trafilatura → markdown (headings/lists/links preserved). favor_recall
                # keeps more of sparse pages; include_* turn on the markdown markup.
                markdown = trafilatura.extract(
                    body, url=final_url, output_format="markdown",
                    include_formatting=True, include_links=True, include_tables=True,
                    favor_recall=True)
                title = _extract_title(body)
                if markdown:
                    extracted = True
                else:
                    # Non-article / extraction-empty page: fall back to the reviewed
                    # stdlib text dump so a node still gets usable content.
                    markdown = _extract_text(body)
            else:
                markdown = body  # already plain text (e.g. text/plain, json)

            return {
                "ok": True,
                "url": url,
                "final_url": final_url,
                "status": resp.status_code,
                "content_type": content_type,
                "title": title,
                "markdown": markdown or "",
                "extracted": extracted,
                "truncated": truncated,
                "bytes": len(raw),
            }

        # Loop exhausted on a transient error (all retries failed).
        return _fetch_failure(url, status=last_status, error_kind=last_kind,
                              error=last_err or "fetch failed after retries",
                              attempts=max_retries + 1)

    return web_fetch


_WEB_FETCH_DESC = (
    "READ a source. Fetch ONE public URL and extract clean content as MARKDOWN "
    "(httpx + Trafilatura; headings/lists/links preserved). Read a source before "
    "relying on it — never cite a page you have not fetched. On failure it returns "
    "a DISTINCT reason (`error_kind`: http_403 forbidden / http_404 not-found / "
    "timeout / network_error / denied_domain / blocked) so you can tell a dead "
    "link from a blocked one and try an ALTERNATE source instead of retrying. "
    "Transient errors are retried with backoff; deny-listed domains (Wikipedia) "
    "are never fetched.")


WEB_FETCH_TOOL = ToolDef(
    name="web_fetch",
    description=_WEB_FETCH_DESC,
    args_model=WebFetchArgs,
    handler=make_web_fetch(),
)


# --------------------------------------------------------------------------- #
# Registration — add both web tools to a GrowableToolRegistry (one entry each)
# --------------------------------------------------------------------------- #


def register_web_tools(registry: Any, *, search_backend: Optional[SearchBackend] = None,
                       cache: Optional[ResultCache] = None, timeout: float = 20.0,
                       deny_domains: Any = DEFAULT_DENY_DOMAINS) -> Any:
    """Add ``web_search`` + ``web_fetch`` to a :class:`GrowableToolRegistry`.

    Each is one :class:`ToolDef`; registering is the single ``registry.add`` growth
    point (registry core untouched). Pass ``search_backend`` to swap the free
    DuckDuckGo provider for a paid one later. ``deny_domains`` sets the always-on
    baseline source deny-list applied by BOTH tools (default: Wikipedia/Wikimedia/
    Wiktionary per d131/d133) — a host can extend or override it. Returns the
    registry for chaining."""
    registry.add(ToolDef(
        name="web_search",
        description=WEB_SEARCH_TOOL.description,
        args_model=WebSearchArgs,
        handler=make_web_search(backend=search_backend, cache=cache, timeout=timeout,
                                deny_domains=deny_domains),
    ))
    registry.add(ToolDef(
        name="web_fetch",
        description=WEB_FETCH_TOOL.description,
        args_model=WebFetchArgs,
        handler=make_web_fetch(timeout=timeout, deny_domains=deny_domains),
    ))
    return registry


__all__ = [
    "WebSearchArgs",
    "WebFetchArgs",
    "ResultCache",
    "SearchBackend",
    "ddgs_backend",
    "make_web_search",
    "make_web_fetch",
    "WEB_SEARCH_TOOL",
    "WEB_FETCH_TOOL",
    "register_web_tools",
    "DEFAULT_SEARCH_TTL_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_DENY_DOMAINS",
]
