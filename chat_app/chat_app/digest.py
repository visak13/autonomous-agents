"""Code-assembled, token-budgeted RESEARCH DIGEST (autonomy rebuild, Phase 2B).

The eda-base3 pattern transplanted: downstream consumers (the write planner, the
follow-up decider) get a DETERMINISTIC, BOUNDED re-ground packet instead of raw
findings blobs — bodies only for load-bearing entries, an id+title index for
everything else, an explicit cursor naming what was withheld and HOW to pull it
(read_notes / load_source), and a progressive downshift so the packet cannot
outgrow its budget however large the research grows. NO model call, NO
fabrication: every line is projected from already-gathered artifacts.

This REPLACES the engine-extracted pushes the owner rejected (`_collect_findings`'
12k concatenation and the `findings[:1200]` truncation): the digest tells the
consumer WHAT exists and how to pull it; it does not spoon-feed content.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from agent_runtime.research_tree import (
    compose_research_narrative,
    render_verbatim_source_index,
)

__all__ = ["build_research_digest", "DIGEST_TOKEN_BUDGET"]

# ~4 chars/token heuristic — the same coarse ratio the runtime uses elsewhere.
_CHARS_PER_TOKEN = 4
# Default packet budget: comfortably inside one node's working context on the
# 32k window, leaving room for the goal + doctrine + generation.
DIGEST_TOKEN_BUDGET = 3000


def _note_gist(note: Mapping[str, Any], *, max_chars: int = 220) -> str:
    """One bounded line per note: url tail + summary head + open gap count."""
    url = str(note.get("url") or "")
    tail = url.rsplit("/", 1)[-1][:40] or url[:40]
    summary = " ".join(str(note.get("summary") or "").split())
    gaps = note.get("gaps_or_followups") or []
    line = f"- ({tail}) {summary}"
    if gaps:
        line += f" [open gaps: {len(gaps)}]"
    return line[:max_chars]


def build_research_digest(
    sources: Sequence[Mapping[str, Any]],
    notes: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    token_budget: int = DIGEST_TOKEN_BUDGET,
) -> str:
    """Assemble the bounded digest: narrative → source index → note gists → cursor.

    Progressive downshift when over budget (largest first): note gists collapse to
    a count line, then the narrative is dropped (the verbatim ``[S#]`` index is the
    LAST thing to shrink — citations must resolve). Every withheld layer is named
    in the cursor line so the consumer knows to PULL (read_notes / load_source)."""
    notes = list(notes or [])
    budget_chars = max(400, int(token_budget) * _CHARS_PER_TOKEN)

    narrative = compose_research_narrative(notes, sources) if notes else ""
    index = render_verbatim_source_index(sources)
    gists = [_note_gist(n) for n in notes]

    def _assemble(with_narrative: bool, gist_cap: Optional[int]) -> str:
        parts: list[str] = []
        if with_narrative and narrative:
            parts.append(narrative)
        if index:
            parts.append(index)
        shown = gists if gist_cap is None else gists[:gist_cap]
        withheld = len(gists) - len(shown)
        if shown:
            parts.append("ARTICLE-NOTE GISTS (pull any note in full via read_notes):\n"
                         + "\n".join(shown))
        cursor_bits: list[str] = []
        if withheld > 0:
            cursor_bits.append(f"{withheld} more note gist(s) withheld — read_notes")
        if not with_narrative and narrative:
            cursor_bits.append("narrative withheld for budget — read_notes carries the gaps")
        cursor_bits.append(
            "every [S#] source's verbatim text is pullable via load_source"
        )
        parts.append("MORE: " + "; ".join(cursor_bits) + ".")
        return "\n\n".join(p for p in parts if p)

    # Downshift ladder: full → capped gists → no gists → no narrative.
    for attempt in (
        lambda: _assemble(True, None),
        lambda: _assemble(True, 8),
        lambda: _assemble(True, 0),
        lambda: _assemble(False, 0),
    ):
        out = attempt()
        if len(out) <= budget_chars:
            return out
    # Last resort: hard-trim the final form, keeping its head (index-first layout
    # means the [S#] map survives; the trim point is announced).
    return out[: budget_chars - 40].rstrip() + "\n[digest trimmed at budget]"
