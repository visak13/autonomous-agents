"""bundles.web_ingest — the WEB bundle's DISPATCH + INGEST adapter (SA-5 / d254 SoC).

The web URL / article / readability / markdown-record semantics used to be baked into
the ENGINE (``runtime.SubAgent._dispatch_research_tool`` / ``_looks_like_article_url`` /
``_is_readable_fetch`` / ``_read_fetched`` / the fetched-record dict). After SA-4 made the
engine's gather dispatch GENERIC by-name, SA-5 RELOCATES every web-specific decision here,
into the bundle that OWNS ``web_search`` / ``web_fetch``. The engine now keeps ONLY the
generic by-name dispatch (``_invoke_loaded_tool``) and DELEGATES the configured web tools to
:class:`WebGatherAdapter` — so a web_search/web_fetch turn's URL grounding, readability gate,
article-record shaping and read-coverage note all FIRE FROM THE BUNDLE, not the engine.

The adapter is dependency-light and source-agnostic in its wiring: the engine supplies the
hook ``invoke`` closure, a ``read_fetched`` closure (which still holds the engine's embedder
+ budgets), and the web_fetch take-a-note suffix (already bundle-sourced via
``tool_output_override``). The LOGIC — what a readable article URL is, which fetch carries
real text, how a fetched source becomes a ``{title,url,markdown}`` record, and the coverage
read-note prose — lives HERE. Behaviour is byte-identical to the prior engine code (the
served web deep-research path is the contrastive byte-comparable gate), only its OWNER moved.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping, Optional

from ..roles import READ_NOT_DESCRIBE

# URL extensions that ``web_fetch`` cannot turn into readable article TEXT:
# Trafilatura is HTML-only, so a PDF/office/media URL decodes to binary garbage
# and a research layer reports "unreadable binary data" instead of findings (the
# max_iter=10 live finding). Skip these up front so the layer reads real prose.
NON_ARTICLE_EXT = (
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".csv",
    ".zip", ".gz", ".tar", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".mp4", ".mp3", ".mov", ".avi", ".bin",
)


def looks_like_article_url(url: str) -> bool:
    """A public http(s) URL that is plausibly a readable HTML page (not a file)."""
    if not url.startswith(("http://", "https://")):
        return False
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    return not path.endswith(NON_ARTICLE_EXT)


def url_offered(want: str, offered: set[str]) -> bool:
    """True when ``want`` is one of the real URLs a web_search surfaced this node.

    Matched ignoring a trailing slash so a verbatim copy with/without the slash still
    grounds (s15/a25 grounding guard — the model must fetch a URL it was OFFERED, not
    one it invented)."""
    w = want.rstrip("/")
    return any(want == o or w == o.rstrip("/") for o in offered)


def is_readable_fetch(val: Mapping[str, Any]) -> bool:
    """True if a web_fetch result carries READABLE article text (not binary).

    Trust the tool's ``extracted`` flag (Trafilatura produced article markdown);
    otherwise require a text-ish content type. A PDF/binary fetch (``extracted``
    False + a non-text content type) is rejected so its garbage never reaches the
    research call — the fix for the live "uninterpretable binary data" failure."""
    if val.get("extracted"):
        return True
    ctype = str(val.get("content_type") or "").lower()
    if "pdf" in ctype:
        return False
    return any(t in ctype for t in ("text/", "html", "json", "xml")) and bool(ctype)


class WebGatherAdapter:
    """The web bundle's gather DISPATCH + INGEST adapter (SA-5).

    Owns the web search/fetch dispatch keyed by the configured tool NAMES (the relocated
    construction-time ``_search_tool`` / ``_fetch_tool`` semantics) and every web ingest
    decision. :meth:`dispatch` executes ONE model-chosen web tool call against the engine's
    ``invoke`` closure and returns the observation string, appending readable fetched sources
    to ``fetched`` exactly as the prior engine method did."""

    def __init__(self, search_tool: str, fetch_tool: str, note_tool: str) -> None:
        self.search_tool = search_tool
        self.fetch_tool = fetch_tool
        self.note_tool = note_tool

    async def dispatch(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        invoke: Callable[..., Awaitable[Any]],
        fetched: list[dict[str, str]],
        seen_urls: set[str],
        offered_urls: Optional[set[str]] = None,
        read_fetched: Callable[[str, str, str], Awaitable[tuple]],
        emit_article_notes: bool = False,
        fetch_note_suffix: str = "",
    ) -> str:
        """Execute ONE model-chosen web tool call → an observation string.

        A ``web_search`` returns its top candidate rows (title/url/snippet); a
        ``web_fetch`` returns the EXTRACTED article markdown (and the source is appended
        to ``fetched`` so it can later ground a downstream node, d17). The caller's
        ``invoke`` publishes tool_call/tool_result on each invoke, so the live trace shows
        the model's real search/fetch decisions (the observability bar). A failed/dead/binary
        call yields a short non-fatal note, never an exception (a research turn must not
        crash)."""
        if tool == self.search_tool:
            query = str(
                args.get("query") or args.get("q") or args.get("search") or ""
            ).strip()
            if not query:
                return "web_search was not executed: it needs a non-empty \"query\"."
            try:
                res = await invoke(self.search_tool, query=query)
            except Exception as exc:  # noqa: BLE001 - a failed search must not crash the node
                return f"web_search failed: {exc}."
            if not getattr(res, "ok", False):
                return f"web_search returned no results ({getattr(res, 'error', '')})."
            rows = (res.value or {}).get("results") if isinstance(res.value, Mapping) else None
            rows = rows or []
            if not rows:
                return "web_search returned 0 results for this query."
            # s15/a25 LEVER 1 (d186): PRESENT the offered URLs as the EXPLICIT choice set so
            # the small model treats them as the actionable list to pick from (the role:tool
            # result was not clearly actionable, so the model fabricated dead URLs instead of
            # copying a real one). Each row's url is recorded in ``offered_urls`` (the grounding
            # set LEVER 3 validates the next web_fetch against).
            lines = [f"SEARCH RESULTS for \"{query}\" — only URLs returned by a search this task will load in web_fetch:"]
            urls_seen: list[str] = []
            displayed = 0
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                url = str(row.get("url") or "").strip()
                if not url:
                    continue
                # The GROUNDING set (LEVER 3) holds EVERY real url the search returned — a
                # fetch of any of them is legitimate. The DISPLAY is capped at 8 rows for token
                # economy; the offered set is NOT capped, so a fetch of a returned-but-not-shown
                # url still grounds (it was really returned, not invented).
                if offered_urls is not None:
                    offered_urls.add(url)
                urls_seen.append(url)
                if displayed < 8:
                    title = str(row.get("title") or "").strip() or "(untitled)"
                    snip = str(row.get("snippet") or "").strip()[:200]
                    lines.append(f"- {title} <{url}>\n  {snip}")
                    displayed += 1
            # Fact-only (CoT-autonomy P2): the rows ARE the data; the verbatim-URL
            # contract lives in web_fetch's tool description (its one owner) and the
            # header above states the grounding constraint as a fact.
            return "\n".join(lines)

        # web_fetch
        url = str(args.get("url") or args.get("link") or "").strip()
        if not url:
            return "web_fetch was not executed: it needs a non-empty \"url\"."
        if url in seen_urls:
            return f"Already read <{url}> this task."
        seen_urls.add(url)
        if not looks_like_article_url(url):
            return f"<{url}> is not a readable HTML article (PDF/file/binary)."
        try:
            res = await invoke(self.fetch_tool, url=url)
        except Exception as exc:  # noqa: BLE001 - a dead link must not fail the node
            return f"Could not fetch <{url}>: {exc}."
        if not getattr(res, "ok", False):
            return f"Could not fetch <{url}>."
        val = res.value if isinstance(res.value, Mapping) else {}
        # web_fetch surfaces a STRUCTURED failure (ok=False + a DISTINCT error_kind)
        # so the agent reacts correctly to WHY a read failed: a 403/blocked/denied
        # page will not yield to a retry (pick another source); a 404 is a dead link;
        # a deny-listed domain (e.g. Wikipedia) must never be cited. Relay the exact
        # reason instead of a single generic "try another source".
        if val.get("ok") is False:
            kind = str(val.get("error_kind") or "error")
            detail = str(val.get("error") or "").strip()
            return f"Could not read <{url}> [{kind}]: {detail}"
        md = str(val.get("markdown") or "").strip()
        if not md or not is_readable_fetch(val):
            return f"<{url}> had no readable article text."
        title = str(val.get("title") or "").strip() or url.rsplit("/", 1)[-1]
        final_url = str(val.get("final_url") or url)
        record: dict[str, str] = {"title": title, "url": final_url, "markdown": md}
        # READ the source into the window (N3): a long article is map/reduced into an
        # in-window factual summary instead of being truncated to the first budget chars
        # (which dropped the rest of the document); short sources pass through verbatim.
        body, summary, read_signal = await read_fetched(md, title, final_url)
        if summary is not None:
            record["summary"] = summary  # additive; full ``markdown`` stays untouched
        fetched.append(record)
        # COVERAGE SIGNAL (MSF/d89-b, fixes seam ⑤): tell the model HOW MUCH of the
        # source it actually has so it reasons about coverage instead of treating the
        # sliver as the whole article. A whole-doc map/reduce summary (chunked-read ON)
        # IS complete coverage; a flat truncation is NOT — say so and invite a follow-up.
        full_chars = len(md)
        if read_signal is not None:
            # d109 HONEST signal: counts + provenance for the RELEVANCE-SELECT read —
            # replaces the vague "there is MORE" nudge with the real M-found/X-read numbers
            # and which sources are now in hand, across the node's fetched docs.
            src_names = [
                (f.get("title") or f.get("url") or "(source)") for f in fetched
            ]
            provenance = ", ".join(src_names[-3:])
            read_note = (
                f"FETCHED <{title}> <{final_url}> — found {read_signal['found']} relevant "
                f"passages in this source; reading the top {read_signal['read']} "
                f"({read_signal['chars']} chars) most relevant to your question. You have "
                f"now read {len(fetched)} source(s): {provenance}. "
            )
        elif summary is not None:
            read_note = (
                f"FETCHED <{title}> <{final_url}> — you have now READ this WHOLE source "
                f"(a grounded factual summary covering all {full_chars} chars). "
            )
        elif len(body) < full_chars:
            read_note = (
                f"FETCHED <{title}> <{final_url}> — showing the first {len(body)} of "
                f"{full_chars} chars; {full_chars - len(body)} chars of this source "
                "are unread. "
            )
        else:
            read_note = f"FETCHED <{title}> <{final_url}> — you have now READ this source. "
        # Fact-only (CoT-autonomy P2): the coverage counts + the body ARE the
        # observation. The per-fetch READ_NOT_DESCRIBE doctrine append and the
        # take-a-note chain (WEB_FETCH_NOTE_OVERRIDE via ``fetch_note_suffix``) are
        # RETIRED — read-don't-describe lives ONCE in the research spec, and note
        # discipline lives in the note tool's description + the research bundle
        # doctrine (their single owners). ``emit_article_notes``/``fetch_note_suffix``
        # params are kept for signature stability but no longer append anything.
        del emit_article_notes, fetch_note_suffix  # explicitly unused (P2)
        observation = read_note + f"\n\n{body}"
        return observation


__all__ = [
    "NON_ARTICLE_EXT",
    "looks_like_article_url",
    "url_offered",
    "is_readable_fetch",
    "WebGatherAdapter",
]
