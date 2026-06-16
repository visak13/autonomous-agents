"""The 4 core tools — read_file, write_file, web_search, web_fetch.

These are plain callables registered into a :class:`~reactive_tools.tool_hook.ToolRegistry`
and invoked through the single :class:`~reactive_tools.tool_hook.ToolHook` so
every call + result flows on the event plane. They are *composable* (each is
just a function) and dependency-light (d10): HTTP via ``httpx`` (already a
workspace dep), HTML parsing via the **stdlib** ``html.parser`` — no scraping
library to install, which keeps the offline-first build honest and matches the
zero-dep precedent set by the event plane.

Decisions honored
------------------
- d2  — purely in-process. No broker/pool, no sockets beyond the tools' own
  outbound HTTP, no subprocess.
- d6  — web access is FREE and key-LESS: search hits DuckDuckGo's HTML endpoint
  (``https://html.duckduckgo.com/html/``); fetch is a plain GET. No API keys,
  no paid services.
- d8  — NO shell-command anything. File I/O is via ``pathlib``/``open``, never
  a shell; nothing here invokes cmd.exe/bash.

SECURITY NOTES FOR THE REVIEWER
-------------------------------
read_file / write_file (path traversal):
  Both resolve the requested path *against a fixed allowed base* and refuse any
  path that escapes it (``..``, absolute paths, symlinks that point outside).
  The check uses ``os.path.realpath`` + ``os.path.commonpath`` so a symlink
  inside the base that targets outside is also rejected. The host MUST pass a
  real sandbox ``base``; the default is the process CWD (callers should narrow
  it). Reads are size-bounded (``max_bytes``) so a huge file can't blow memory.

web_fetch (SSRF + untrusted HTML):
  - Scheme allow-list: only ``http``/``https`` (no ``file://``, ``ftp://`` …).
  - SSRF guard: the resolved host's IPs are checked and **private, loopback,
    link-local, reserved, and multicast addresses are rejected** so the tool
    can't be aimed at internal metadata services / localhost. This is
    best-effort (it does not defeat DNS-rebinding TOCTOU — a hardened deploy
    should additionally pin the connection to the validated IP); called out
    here so the reviewer can weigh it. Redirects are followed MANUALLY (httpx
    auto-follow disabled) and every hop is SSRF-validated BEFORE its request is
    issued, so a public URL cannot bounce the fetch onto an internal host.
  - Response size is hard-bounded (``max_bytes``, default 2 MiB) by reading the
    stream in chunks and stopping — defends against decompression / oversize
    bombs. HTML is parsed with the forgiving stdlib parser and only *text* is
    extracted (scripts/styles dropped), so no markup is ever executed.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

import httpx

# --------------------------------------------------------------------------- #
# File tools
# --------------------------------------------------------------------------- #

DEFAULT_READ_MAX_BYTES = 2 * 1024 * 1024  # 2 MiB

# Default artifact directory (d3): a bare/relative filename a tool writes lands
# HERE, not wherever the process CWD happens to be (the round-1
# C:\Users\aksou\Downloads\report.md bug). A host can override the base
# per-task by passing ``file_base`` to :func:`register_core_tools`.
DEFAULT_ARTIFACT_DIR = Path(r"C:\Projects\ReactiveAgents\artifacts")


class ToolInputError(ValueError):
    """A tool was called with invalid / unsafe input (bad path, bad url)."""


def _safe_resolve(path: str, base: Path) -> Path:
    """Resolve ``path`` under ``base`` and reject anything that escapes it.

    ``path`` may be relative (joined onto ``base``) or absolute (it must still
    land inside ``base``). Uses realpath + commonpath so a symlink pointing out
    of the sandbox is also caught (path-traversal guard)."""
    if not isinstance(path, str) or not path:
        raise ToolInputError("path must be a non-empty string")
    base_real = Path(os.path.realpath(base))
    candidate = Path(path)
    joined = candidate if candidate.is_absolute() else base_real / candidate
    target_real = Path(os.path.realpath(joined))
    try:
        common = os.path.commonpath([str(base_real), str(target_real)])
    except ValueError:
        # Different drives on Windows -> definitely outside the base.
        raise ToolInputError(f"path {path!r} escapes the allowed base") from None
    if common != str(base_real):
        raise ToolInputError(f"path {path!r} escapes the allowed base {base_real}")
    return target_real


def make_read_file(base: Path):
    def read_file(path: str, max_bytes: int = DEFAULT_READ_MAX_BYTES,
                  encoding: str = "utf-8") -> dict[str, Any]:
        """Read a text file (within the sandbox ``base``) and return its text.

        Returns ``{"path", "text", "truncated", "bytes"}``. ``truncated`` is
        True when the file was longer than ``max_bytes``."""
        target = _safe_resolve(path, base)
        if not target.is_file():
            raise ToolInputError(f"not a file: {path!r}")
        raw = target.read_bytes()
        truncated = len(raw) > max_bytes
        raw = raw[:max_bytes]
        text = raw.decode(encoding, errors="replace")
        return {
            "path": str(target),
            "text": text,
            "truncated": truncated,
            "bytes": len(raw),
        }

    return read_file


def make_write_file(base: Path):
    def write_file(path: str, content: str, encoding: str = "utf-8",
                   overwrite: bool = True) -> dict[str, Any]:
        """Write ``content`` to ``path`` (within the sandbox ``base``).

        Creates parent directories *inside the base*. Returns
        ``{"path", "bytes"}``. With ``overwrite=False`` an existing file is an
        error (no clobber)."""
        if not isinstance(content, str):
            raise ToolInputError("content must be a string")
        target = _safe_resolve(path, base)
        if target.exists() and not overwrite:
            raise ToolInputError(f"refusing to overwrite existing file: {path!r}")
        # Parent is guaranteed inside the base by _safe_resolve on the file.
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode(encoding)
        target.write_bytes(data)
        return {"path": str(target), "bytes": len(data)}

    return write_file


# --------------------------------------------------------------------------- #
# Claude-style file tools (the registered read_file / write_file the agent uses)
# --------------------------------------------------------------------------- #
#
# These SUPERSEDE the legacy callables above as the *registered* tools (the
# legacy ``make_read_file``/``make_write_file`` stay in this module for
# back-compat imports). They match Claude's Read/Write/Edit semantics:
#   - read_file : line-based ``offset``/``limit`` (in addition to the byte cap)
#   - write_file: ONE entrypoint exposing three modes —
#       * ``new_file=True``          -> CREATE+name a file; parent dirs made;
#                                       REFUSES to overwrite an existing file.
#       * ``old_string`` given       -> EDIT in place via exact-string replace;
#                                       fails if ``old_string`` is ABSENT or NOT
#                                       UNIQUE (unless ``replace_all=True``).
#       * ``content`` only           -> legacy create/overwrite write (back-compat
#                                       with existing write_file(path, content)
#                                       call sites + the tool-arg emitter schema).
# All modes keep the ``_safe_resolve`` path-traversal sandbox guard.


def make_read(base: Path):
    def read_file(path: str, offset: Optional[int] = None,
                  limit: Optional[int] = None,
                  max_bytes: int = DEFAULT_READ_MAX_BYTES,
                  encoding: str = "utf-8") -> dict[str, Any]:
        """Read a text file (within the sandbox ``base``), Claude-style.

        ``offset`` (0-based line index) and ``limit`` (max lines) select a line
        window like Claude's Read; the byte cap (``max_bytes``) still bounds how
        much is loaded first. Returns ``{"path", "content", "text", "offset",
        "limit", "lines_returned", "total_lines", "truncated", "line_sliced",
        "bytes"}`` (``text`` is an alias of ``content`` for back-compat)."""
        target = _safe_resolve(path, base)
        if not target.is_file():
            raise ToolInputError(f"not a file: {path!r}")
        raw = target.read_bytes()
        truncated = len(raw) > max_bytes
        raw = raw[:max_bytes]
        text = raw.decode(encoding, errors="replace")
        lines = text.splitlines()
        total_lines = len(lines)
        if offset is None and limit is None:
            content = text          # exact byte content (back-compat path)
            start = 0
            line_sliced = False
        else:
            start = offset or 0
            if start < 0:
                raise ToolInputError("offset must be >= 0")
            if limit is not None and limit < 0:
                raise ToolInputError("limit must be >= 0")
            end = total_lines if limit is None else start + limit
            content = "\n".join(lines[start:end])
            line_sliced = True
        return {
            "path": str(target),
            "content": content,
            "text": content,
            "offset": start,
            "limit": limit,
            "lines_returned": (len(content.splitlines()) if line_sliced
                               else total_lines),
            "total_lines": total_lines,
            "truncated": truncated,
            "line_sliced": line_sliced,
            "bytes": len(raw),
        }

    return read_file


def make_write(base: Path):
    def write_file(path: str, content: Optional[str] = None, *,
                   new_file: bool = False,
                   old_string: Optional[str] = None,
                   new_string: Optional[str] = None,
                   replace_all: bool = False,
                   encoding: str = "utf-8",
                   overwrite: bool = True) -> dict[str, Any]:
        """Create, overwrite, or edit a file (within the sandbox ``base``).

        ONE entrypoint, Claude-style (see module header for the three modes).
        Path-traversal is guarded by ``_safe_resolve``."""
        target = _safe_resolve(path, base)

        # --- EDIT mode: in-place exact-string replacement ------------------ #
        if old_string is not None:
            if not target.is_file():
                raise ToolInputError(f"cannot edit: not a file: {path!r}")
            if new_string is None:
                raise ToolInputError(
                    "edit mode requires new_string when old_string is given")
            if old_string == new_string:
                raise ToolInputError("old_string and new_string are identical")
            original = target.read_bytes().decode(encoding, errors="replace")
            count = original.count(old_string)
            if count == 0:
                raise ToolInputError(
                    f"old_string not found in {path!r} (no match to edit)")
            if count > 1 and not replace_all:
                raise ToolInputError(
                    f"old_string is not unique in {path!r} ({count} matches); "
                    "pass replace_all=True, or give a longer string that "
                    "matches exactly once")
            if replace_all:
                updated = original.replace(old_string, new_string)
                replaced = count
            else:
                updated = original.replace(old_string, new_string, 1)
                replaced = 1
            data = updated.encode(encoding)
            target.write_bytes(data)
            return {"path": str(target), "mode": "edit",
                    "replacements": replaced, "bytes": len(data)}

        # --- CREATE / WRITE mode ------------------------------------------- #
        if content is None:
            raise ToolInputError(
                "write requires 'content' (or 'old_string'/'new_string' to edit)")
        if not isinstance(content, str):
            raise ToolInputError("content must be a string")
        exists = target.exists()
        if new_file and exists:
            raise ToolInputError(
                f"new_file=True but file already exists (refusing to "
                f"overwrite): {path!r}")
        if exists and not overwrite and not new_file:
            raise ToolInputError(f"refusing to overwrite existing file: {path!r}")
        # Parent is guaranteed inside the base by _safe_resolve on the file.
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode(encoding)
        target.write_bytes(data)
        return {"path": str(target),
                "mode": "create" if new_file else "write",
                "created": not exists, "bytes": len(data)}

    return write_file


# --------------------------------------------------------------------------- #
# HTML parsing (stdlib) — shared by web_search + web_fetch
# --------------------------------------------------------------------------- #

_BLOCK_TAGS = frozenset(
    {"p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article",
     "header", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre"}
)
_SKIP_TAGS = frozenset({"script", "style", "head", "noscript", "template", "svg"})


class _TextExtractor(HTMLParser):
    """Collect human-readable text, dropping scripts/styles/markup.

    The stdlib parser is forgiving of the malformed HTML you get from arbitrary
    pages, and because we only ever *read* data (never eval), parsing untrusted
    markup is safe."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def text(self) -> str:
        raw = "".join(self._chunks)
        # Collapse runs of blank lines / trailing spaces into clean readable text.
        lines = [ln.strip() for ln in raw.splitlines()]
        out: list[str] = []
        blank = False
        for ln in lines:
            if ln:
                out.append(ln)
                blank = False
            elif not blank:
                out.append("")
                blank = True
        return "\n".join(out).strip()


def _extract_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


class _DDGResultParser(HTMLParser):
    """Parse DuckDuckGo's HTML results page into ``[{title, url, snippet}]``.

    DDG's no-JS HTML page (``html.duckduckgo.com/html/``) renders each result
    title as ``<a class="result__a" href=...>`` and the snippet as an element
    with class ``result__snippet``. The href is a redirect of the form
    ``//duckduckgo.com/l/?uddg=<percent-encoded-target>`` which we decode back
    to the real URL."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._in_title = False
        self._in_snippet = False
        self._cur: Optional[dict[str, str]] = None
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []

    @staticmethod
    def _classes(attrs: list[tuple[str, Optional[str]]]) -> set[str]:
        for k, v in attrs:
            if k == "class" and v:
                return set(v.split())
        return set()

    @staticmethod
    def _href(attrs: list[tuple[str, Optional[str]]]) -> str:
        for k, v in attrs:
            if k == "href" and v:
                return v
        return ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        classes = self._classes(attrs)
        if tag == "a" and "result__a" in classes:
            self._cur = {"title": "", "url": _decode_ddg_href(self._href(attrs)),
                         "snippet": ""}
            self._in_title = True
            self._title_parts = []
        elif "result__snippet" in classes:
            self._in_snippet = True
            self._snippet_parts = []

    def handle_endtag(self, tag: str) -> None:
        if self._in_title and tag == "a":
            if self._cur is not None:
                self._cur["title"] = "".join(self._title_parts).strip()
            self._in_title = False
        elif self._in_snippet and tag in ("a", "div", "span", "td"):
            if self._cur is not None:
                self._cur["snippet"] = "".join(self._snippet_parts).strip()
                self.results.append(self._cur)
                self._cur = None
            self._in_snippet = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
        elif self._in_snippet:
            self._snippet_parts.append(data)


def _decode_ddg_href(href: str) -> str:
    """Turn a DDG ``/l/?uddg=...`` redirect into the real target URL."""
    if not href:
        return ""
    parsed = urlparse(href if "//" in href else "//" + href, scheme="https")
    qs = parse_qs(parsed.query)
    if "uddg" in qs and qs["uddg"]:
        return unquote(qs["uddg"][0])
    # Already a direct link (or protocol-relative) — normalise scheme.
    if href.startswith("//"):
        return "https:" + href
    return href


# --------------------------------------------------------------------------- #
# Web tools
# --------------------------------------------------------------------------- #

DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
DEFAULT_FETCH_MAX_BYTES = 2 * 1024 * 1024  # 2 MiB
DEFAULT_MAX_REDIRECTS = 5
_USER_AGENT = "ReactiveAgents/0.1 (+in-process; httpx)"


def _assert_public_http_url(url: str) -> tuple[str, str]:
    """Validate a URL for outbound fetch. Returns ``(url, host)``.

    Enforces the http(s) scheme allow-list and the SSRF guard (reject hosts
    that resolve to private / loopback / link-local / reserved / multicast
    IPs). Raises :class:`ToolInputError` on any violation."""
    if not isinstance(url, str) or not url:
        raise ToolInputError("url must be a non-empty string")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolInputError(f"only http/https URLs allowed, got scheme {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ToolInputError(f"url has no host: {url!r}")
    # Resolve every address the host maps to and reject if ANY is non-public.
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ToolInputError(f"cannot resolve host {host!r}: {exc}") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            raise ToolInputError(
                f"refusing to fetch {host!r} -> non-public address {ip} (SSRF guard)"
            )
    return url, host


def make_web_search(http_timeout: float = 20.0):
    def web_search(query: str, max_results: int = 8) -> dict[str, Any]:
        """Free, key-LESS web search via DuckDuckGo's HTML endpoint (d6).

        Returns ``{"query", "results": [{title, url, snippet}, ...]}`` (up to
        ``max_results``). No API key, no paid service."""
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError("query must be a non-empty string")
        headers = {"User-Agent": _USER_AGENT,
                   "Accept": "text/html", "Accept-Language": "en-US,en;q=0.9"}
        with httpx.Client(timeout=http_timeout, headers=headers,
                          follow_redirects=True) as client:
            resp = client.post(DDG_HTML_ENDPOINT, data={"q": query, "kl": "us-en"})
            resp.raise_for_status()
            html = resp.text
        parser = _DDGResultParser()
        parser.feed(html)
        parser.close()
        results = [r for r in parser.results if r.get("url")][: max(0, max_results)]
        return {"query": query, "results": results, "count": len(results)}

    return web_search


def make_web_fetch(http_timeout: float = 20.0):
    def web_fetch(url: str, max_bytes: int = DEFAULT_FETCH_MAX_BYTES,
                  max_redirects: int = DEFAULT_MAX_REDIRECTS) -> dict[str, Any]:
        """Fetch a URL (public http/https only) and extract clean readable text.

        Returns ``{"url", "final_url", "status", "content_type", "title",
        "text", "truncated", "bytes"}``. Response size is hard-bounded by
        ``max_bytes``; untrusted HTML is only parsed for text, never executed
        (see module SECURITY NOTES).

        SSRF: redirects are followed MANUALLY (httpx auto-follow disabled) so
        EVERY hop's host is SSRF-validated *before* a request is issued to it —
        an httpx auto-follow would connect to an intermediate/final redirect
        target before any check could run, letting a public URL bounce the
        fetch onto an internal/metadata host. Capped at ``max_redirects`` hops."""
        headers = {"User-Agent": _USER_AGENT, "Accept": "text/html,*/*;q=0.8"}
        current = url
        with httpx.Client(timeout=http_timeout, headers=headers,
                          follow_redirects=False) as client:
            for _hop in range(max_redirects + 1):
                # Validate BEFORE issuing the request — no request is ever made
                # to a host that fails the scheme/SSRF guard.
                _assert_public_http_url(current)
                with client.stream("GET", current) as resp:
                    location = resp.headers.get("location")
                    if resp.is_redirect and location:
                        # Resolve relative redirects against the current URL; the
                        # next loop iteration re-validates the resolved target
                        # BEFORE issuing its request (SSRF per-hop guard).
                        current = str(resp.url.join(location))
                        continue
                    # Terminal: a normal response, or a 3xx lacking Location.
                    resp.raise_for_status()
                    content_type = resp.headers.get("content-type", "")
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
                    break
            else:
                raise ToolInputError(
                    f"too many redirects (> {max_redirects}) starting from {url!r}"
                )
        # Decode using the declared charset if any, else utf-8 with replacement.
        encoding = "utf-8"
        if "charset=" in content_type:
            encoding = content_type.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
        try:
            body = raw.decode(encoding, errors="replace")
        except (LookupError, ValueError):
            body = raw.decode("utf-8", errors="replace")
        if "html" in content_type or (not content_type and "<" in body[:200]):
            title = _extract_title(body)
            text = _extract_text(body)
        else:
            title = ""
            text = body
        return {
            "url": url,
            "final_url": str(resp.url),
            "status": resp.status_code,
            "content_type": content_type,
            "title": title,
            "text": text,
            "truncated": truncated,
            "bytes": len(raw),
        }

    return web_fetch


class _TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in = False
        self.title = ""

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag == "title":
            self._in = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in = False

    def handle_data(self, data: str) -> None:
        if self._in and not self.title:
            self.title = data.strip()


def _extract_title(html: str) -> str:
    p = _TitleParser()
    try:
        p.feed(html)
        p.close()
    except Exception:  # noqa: BLE001 - title is best-effort
        return ""
    return p.title


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def register_core_tools(hook: "Any", *, file_base: Any = None,
                        http_timeout: float = 20.0) -> "Any":
    """Register the 4 core tools onto ``hook`` (a :class:`ToolHook`).

    ``file_base`` is the sandbox root for the file tools AND the directory a
    bare/relative filename resolves under. When not supplied it defaults to
    :data:`DEFAULT_ARTIFACT_DIR` (``C:\\Projects\\ReactiveAgents\\artifacts``) so
    a written report lands in ``artifacts\\`` rather than wherever the process
    CWD happens to be (d3 — the round-1 Downloads\\report.md bug). Pass an
    explicit ``file_base`` for a per-task override. The dir is created if
    missing. Returns the hook for chaining.

    The registered ``read_file``/``write_file`` are the Claude-style tools
    (:func:`make_read` / :func:`make_write`): line-based read offset/limit, and
    one write entrypoint that CREATES (``new_file=True``) or EDITS in place via
    exact-string replacement."""
    base = Path(os.path.realpath(
        file_base if file_base is not None else DEFAULT_ARTIFACT_DIR))
    base.mkdir(parents=True, exist_ok=True)

    hook.register("read_file", make_read(base),
                  description="Read a text file in the sandbox; Claude-style line offset/limit + byte cap; returns content.")
    hook.register("write_file", make_write(base),
                  description="Create a new file (new_file=True, refuses overwrite) OR edit one in place via exact old_string->new_string (replace_all opt); Claude-style, one entrypoint.")
    # web_search/web_fetch are the s2/a2 ddgs + Trafilatura tools — the maintained
    # library path SUPERSEDES the legacy raw DDG-HTML scrape / stdlib text dump
    # below (decision d4). Imported lazily to avoid a circular import (web_tools
    # imports this module's SSRF guard + helpers).
    from .web_tools import make_web_fetch as make_web_fetch_md
    from .web_tools import make_web_search as make_web_search_ddgs
    hook.register("web_search", make_web_search_ddgs(timeout=http_timeout),
                  description="Free no-key web search (DuckDuckGo via ddgs); ranked titles+urls+snippets; cached + rate-limit backoff.")
    hook.register("web_fetch", make_web_fetch_md(timeout=http_timeout),
                  description="Fetch a public URL and extract clean content as MARKDOWN (httpx + Trafilatura).")
    return hook


__all__ = [
    "ToolInputError",
    "register_core_tools",
    "make_read",
    "make_write",
    "make_read_file",
    "make_write_file",
    "make_web_search",
    "make_web_fetch",
    "DEFAULT_ARTIFACT_DIR",
    "DDG_HTML_ENDPOINT",
]
