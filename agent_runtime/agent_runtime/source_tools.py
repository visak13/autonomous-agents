"""s14/P3A item 3 — the capped LOAD-ON-DEMAND source-retrieval tool (research-owned).

The write planner and per-section writers are handed the COMPACT verbatim SOURCE INDEX
(the *map*: stable ``[S#]`` + url + structure-aware chunk headings) rather than a wall of
prose.  When a role needs a source's ACTUAL text it calls ``load_source("S3", "S3.c2")`` and
gets back that chunk's VERBATIM text — bounded by a per-call cap and an overall budget, with
**degrade-to-write-from-loaded** behaviour when the budget is reached (so a cap-hit is a WRITE
trigger, never a stall, and the role is never handed an unbounded blob).

This module owns the tool *mechanism*; the byte-faithful chunk resolution lives in
``research_tree.resolve_chunk`` (the SAME structure-aware chunker the index is built from), so
a citation ``[S3.c2]`` resolves to the exact re-readable span.  No fabrication: an unknown
``[S#]`` returns an explicit not-found note, never invented text.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from reactive_tools.tool_registry import ToolDef

from .research_tree import resolve_chunk

# Default per-call cap (chars of verbatim text a single load_source returns) and the overall
# per-section budget multiple.  Both are bounded + deterministic; the caller sizes them to the
# num_ctx window (the write phase passes the same window-fraction it uses for the scoped block).
#
# d234/d235 READ-HIERARCHY TUNE — load_source is now the EXPENSIVE leg of a two-tier read: the
# CHEAP gist (which source has what — key_claims/summary/gaps) is read FIRST via ``read_notes``,
# so a load_source call is a TARGETED pull for an exact figure/quote to CITE, not the writer's
# only window into a source. With the gist available cheaply we TIGHTEN the verbatim pull toward
# precision over bulk (lower context cost per call, more calls afforded inside the E4B 32k
# window): _DEFAULT_LOAD_MAX_CHARS 4000->3000 (~750 tok/call) and _DEFAULT_LOAD_TOP_N 3->2 (lead
# + one following section is enough surrounding context for a cited figure; a writer that needs
# more pulls again by chunk id). The cumulative _DEFAULT_SECTION_BUDGET is unchanged (the overall
# per-section ceiling that triggers degrade-to-write-from-loaded).
_DEFAULT_LOAD_MAX_CHARS = 3000
_DEFAULT_SECTION_BUDGET = 12000

# A bare [S#] load (no specific chunk) returns the lead PLUS the next chunk(s) concatenated
# (TOP-N, >=1), so ONE pull hands the writer the cited passage WITH enough surrounding verbatim
# context to quote it faithfully. Each appended chunk keeps its own verbatim heading so figures
# stay attributable. Bounded by the per-call cap + the section budget exactly as a single chunk
# is — never an unbounded dump. A specific chunk request (chunk='S3.c2') still returns just that
# chunk. d234/d235: trimmed 3->2 now that read_notes carries the breadth gist (see above).
_DEFAULT_LOAD_TOP_N = 2


class LoadSourceArgs(BaseModel):
    """Arguments for ``load_source`` — pull ONE source's verbatim chunk on demand."""

    sid: str = Field(
        ...,
        description=(
            "The source id to load — either a source key like 'S3' (loads its lead chunk) "
            "or a specific chunk id like 'S3.c2' (loads that section's verbatim text)."
        ),
    )
    chunk: Optional[str] = Field(
        None,
        description=(
            "Optional explicit chunk id (e.g. 'S3.c2') when 'sid' is just the source key. "
            "Omit to load the source's lead chunk."
        ),
    )
    max_chars: int = Field(
        _DEFAULT_LOAD_MAX_CHARS,
        ge=1,
        description="Cap on the verbatim chars to return (the runtime clamps it to its budget).",
    )


def _parse_sid(sid: str, chunk: Optional[str]) -> tuple[Optional[int], Optional[str]]:
    """Split a 'S3' / 'S3.c2' reference into (numeric source index, chunk-id-or-None)."""
    raw = str(sid or "").strip()
    cid = str(chunk).strip() if chunk else None
    token = raw.lstrip("[").rstrip("]")
    if token.upper().startswith("S"):
        token = token[1:]
    head = token.split(".", 1)
    try:
        s_index = int(head[0])
    except (ValueError, IndexError):
        return None, cid
    if cid is None and "." in token:
        cid = f"S{token}"  # caller passed the full chunk id in `sid`
    return s_index, cid


def _next_cid(cid: Optional[str]) -> Optional[str]:
    """The cid of the chunk after ``cid`` (``S3.c1`` -> ``S3.c2``), or None when unparseable."""
    if not cid or "." not in str(cid):
        return None
    head, _, tail = str(cid).rpartition(".c")
    try:
        return f"{head}.c{int(tail) + 1}"
    except (ValueError, TypeError):
        return None


def make_load_source(
    sources: Sequence[Mapping[str, Any]],
    *,
    section_budget: int = _DEFAULT_SECTION_BUDGET,
    per_call_cap: int = _DEFAULT_LOAD_MAX_CHARS,
    top_n: int = _DEFAULT_LOAD_TOP_N,
):
    """Build a ``load_source`` handler bound to THIS run's verbatim source list (item 3).

    The handler enforces TWO caps so it never returns an unbounded blob:
      * a PER-CALL cap (``min(max_chars, per_call_cap)``) on the chars one call returns; and
      * a cumulative ``section_budget`` across calls — once spent, every further call returns
        the DEGRADE-TO-WRITE-FROM-LOADED payload (``{"capped": true, "text": "", "note":
        "BUDGET REACHED — write this section NOW from the sources you have already loaded; cite
        only those [S#]; do NOT wait for more."}``).

    Real-ids-only: an out-of-range / unparseable ``sid`` returns an explicit not-found note
    (``"more": false``, empty text) — never fabricated content.  A fresh handler is built per
    write phase, so the cumulative budget resets each run."""
    remaining = {"budget": max(per_call_cap, int(section_budget))}
    n = len(sources)

    def load_source(sid: str, chunk: Optional[str] = None,
                    max_chars: int = per_call_cap) -> dict[str, Any]:
        if remaining["budget"] <= 0:
            return {
                "sid": str(sid), "capped": True, "text": "", "more": False,
                "note": ("BUDGET REACHED — write this section NOW from the sources you have "
                         "already loaded; cite only those [S#]; do NOT wait for more."),
            }
        s_index, cid = _parse_sid(sid, chunk)
        if s_index is None or not (1 <= s_index <= n):
            return {
                "sid": str(sid), "capped": False, "text": "", "more": False,
                "note": (f"NO SUCH SOURCE {sid!r} — load only [S#] ids present in the SOURCE "
                         f"INDEX (1..{n}); never cite a source not in the index."),
            }
        cap = max(1, min(int(max_chars), per_call_cap, remaining["budget"]))
        markdown = str(sources[s_index - 1].get("markdown") or "")
        resolved = resolve_chunk(markdown, s_index, cid)
        text = str(resolved.get("text") or "")
        more = bool(resolved.get("more"))
        # Pass-D (d216) TOP-N: a BARE [S#] load (no explicit chunk requested) returns the lead
        # PLUS the next chunks concatenated, so one pull is SUBSTANTIVE — enough for the writer
        # to author a real section, not just the lede. A specific chunk request returns only it.
        bare_load = cid is None
        if bare_load and more and top_n > 1:
            cur = resolved.get("chunk")
            gathered = 1
            while gathered < top_n and len(text) < cap:
                nxt = _next_cid(cur)
                if not nxt:
                    break
                nres = resolve_chunk(markdown, s_index, nxt)
                if nres.get("chunk") != nxt:
                    break  # walked past the last chunk (resolver fell back to the lead)
                ntext = str(nres.get("text") or "")
                if not ntext:
                    break
                heading = str(nres.get("heading") or "").strip()
                text += (f"\n\n## {heading}\n" if heading else "\n\n") + ntext
                cur = nxt
                gathered += 1
                more = bool(nres.get("more"))
        truncated = len(text) > cap
        if truncated:
            text = text[:cap].rstrip()
        remaining["budget"] -= len(text)
        return {
            "sid": f"S{s_index}",
            "chunk": resolved.get("chunk"),
            "heading": resolved.get("heading", ""),
            "url": str(sources[s_index - 1].get("url") or ""),
            "text": text,
            "truncated": truncated,
            "more": more or truncated,
            "capped": False,
            "budget_remaining": remaining["budget"],
        }

    return load_source


# d234/d235 READ-HIERARCHY — load_source is the EXPENSIVE second leg of a two-tier read. Its
# description advertises the COST HIERARCHY plainly: read_notes (the cheap article-note gist)
# comes FIRST to learn which source has what; load_source is the EXPENSIVE verbatim pull, used
# ONLY for the exact figure/quote you will cite. A strong, concrete description is what makes a
# small model actually PULL its grounding for a citation (vs. write a figure from memory).
_LOAD_SOURCE_DESCRIPTION = (
    "EXPENSIVE — use read_notes FIRST. PULL the verbatim text of one already-fetched source "
    "ONLY when you need an EXACT figure, date or quotation to CITE word-for-word — read_notes "
    "already gave you each source's gist, so come here for the precise wording, not for an "
    "overview. Pass sid='S3' to get that source's opening AND its next section (the cited "
    "passage with enough surrounding verbatim text to quote it faithfully), or chunk='S3.c2' "
    "for one specific section. Returns the exact verbatim excerpt plus its url; 'more':true "
    "means further sections are available by chunk id. Bounded — when the budget is spent it "
    "returns a 'BUDGET REACHED' note, then write now from what you loaded. Cite only the [S#] "
    "ids that appear in the SOURCE INDEX."
)


# d234/d235 READ-HIERARCHY — read_notes is the CHEAP first leg. The structured ARTICLE-NOTE
# artifact (key_claims/summary/gaps_or_followups, one per fetched source) is the gist the writer
# and reviewer read BEFORE any verbatim pull: it tells them, compactly, which [S#] source covers
# what and where the gaps are, so they spend an expensive load_source call only on the source
# that actually holds the figure/quote they will cite. Bounded by construction (each note is the
# already-capped control record), so reading every note's gist costs a fraction of one verbatim
# chunk.
_DEFAULT_NOTES_SUMMARY_CHARS = 400
_DEFAULT_NOTES_CLAIMS = 6


class ReadNotesArgs(BaseModel):
    """Arguments for ``read_notes`` — read the CHEAP article-note gist of the fetched sources."""

    sid: Optional[str] = Field(
        None,
        description=(
            "Optional source id (e.g. 'S3') to read just that source's note. Omit to read the "
            "gist INDEX of every fetched source (which [S#] covers what + its open gaps)."
        ),
    )


def _note_url(note: Mapping[str, Any]) -> str:
    return str(note.get("url") or "").strip().lower().rstrip("/")


def make_read_notes(
    notes: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]] = (),
    *,
    summary_chars: int = _DEFAULT_NOTES_SUMMARY_CHARS,
    max_claims: int = _DEFAULT_NOTES_CLAIMS,
):
    """Build a ``read_notes`` handler over THIS run's ARTICLE-NOTES, keyed by GLOBAL ``[S#]``.

    Each fetched source contributed an :class:`~agent_runtime.article_note.ArticleNote` (the
    structured CONTROL record: ``summary`` / ``key_claims`` / ``gaps_or_followups`` /
    ``source_trust``). ``read_notes`` returns that gist so a writer/reviewer learns WHICH source
    has what BEFORE spending an expensive ``load_source`` verbatim pull.

    The note's own ``source_id`` is per-research-node (the index within the node that fetched it),
    NOT the global stable ``[S#]`` the writer cites by; we therefore re-key each note to the
    GLOBAL source index by URL-matching it against ``sources`` (the same list ``load_source``
    resolves), so a ``read_notes`` entry and a ``load_source('S#')`` pull name the SAME source. A
    note whose url matches no global source keeps its own 1-based id (best-effort, never invents
    one). Real-ids-only: an out-of-range / unknown ``sid`` returns an explicit not-found note."""
    # Map a global source index (1-based) → its canonical url, for re-keying notes.
    url_to_global: dict[str, int] = {}
    for i, s in enumerate(sources, 1):
        key = str(s.get("url") or "").strip().lower().rstrip("/")
        if key and key not in url_to_global:
            url_to_global[key] = i

    def _gist(note: Mapping[str, Any], sid_num: int) -> dict[str, Any]:
        claims = [str(c).strip() for c in (note.get("key_claims") or []) if str(c).strip()]
        gaps = [str(g).strip() for g in (note.get("gaps_or_followups") or []) if str(g).strip()]
        return {
            "sid": f"S{sid_num}",
            "url": str(note.get("url") or ""),
            "title": str(note.get("title") or ""),
            "source_trust": str(note.get("source_trust") or ""),
            "summary": str(note.get("summary") or "")[:summary_chars],
            "key_claims": claims[:max_claims],
            "gaps_or_followups": gaps[:max_claims],
        }

    # Pre-resolve every note to its global [S#] once.
    indexed: list[tuple[int, Mapping[str, Any]]] = []
    for note in notes:
        global_id = url_to_global.get(_note_url(note))
        if global_id is None:
            try:
                global_id = int(note.get("source_id") or 0) or 0
            except (TypeError, ValueError):
                global_id = 0
        indexed.append((int(global_id), note))

    def read_notes(sid: Optional[str] = None) -> dict[str, Any]:
        if not indexed:
            return {"notes": [], "count": 0,
                    "note": "No article notes were recorded for this research; "
                            "use load_source to read a source's verbatim text."}
        if sid:
            s_index, _ = _parse_sid(str(sid), None)
            match = next((n for gid, n in indexed if gid == s_index), None)
            if match is None:
                return {"sid": str(sid), "found": False,
                        "note": f"NO NOTE for {sid!r} — call read_notes() with no argument to "
                                f"see the gist index of every fetched [S#] source."}
            return {"found": True, **_gist(match, s_index)}
        gists = [_gist(n, gid) for gid, n in indexed if gid]
        return {"notes": gists, "count": len(gists)}

    return read_notes


_READ_NOTES_DESCRIPTION = (
    "CHEAP — call this FIRST, before load_source. Read the structured NOTE gist of the "
    "already-fetched sources: each fetched [S#] source's summary, key_claims and open gaps "
    "(the article-note control record), so you learn WHICH source has the figure or angle you "
    "need WITHOUT pulling any verbatim text. Call read_notes() with no argument for the gist "
    "index of every source, or read_notes(sid='S3') for one source's note. THEN spend an "
    "expensive load_source pull only on the [S#] that holds the exact figure/quote you will cite."
)


def make_read_notes_tool(
    notes: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]] = (),
    *,
    summary_chars: int = _DEFAULT_NOTES_SUMMARY_CHARS,
    max_claims: int = _DEFAULT_NOTES_CLAIMS,
) -> ToolDef:
    """The ``read_notes`` :class:`ToolDef`, ready to ``registry.add`` onto a write runtime."""
    return ToolDef(
        name="read_notes",
        description=_READ_NOTES_DESCRIPTION,
        args_model=ReadNotesArgs,
        handler=make_read_notes(
            notes, sources, summary_chars=summary_chars, max_claims=max_claims,
        ),
    )


def make_load_source_tool(
    sources: Sequence[Mapping[str, Any]],
    *,
    section_budget: int = _DEFAULT_SECTION_BUDGET,
    per_call_cap: int = _DEFAULT_LOAD_MAX_CHARS,
    top_n: int = _DEFAULT_LOAD_TOP_N,
) -> ToolDef:
    """The ``load_source`` :class:`ToolDef`, ready to ``registry.add`` onto a write runtime."""
    return ToolDef(
        name="load_source",
        description=_LOAD_SOURCE_DESCRIPTION,
        args_model=LoadSourceArgs,
        handler=make_load_source(
            sources, section_budget=section_budget, per_call_cap=per_call_cap, top_n=top_n,
        ),
    )


__all__ = [
    "LoadSourceArgs",
    "make_load_source",
    "make_load_source_tool",
    "ReadNotesArgs",
    "make_read_notes",
    "make_read_notes_tool",
]
