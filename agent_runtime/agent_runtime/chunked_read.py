"""s9/N3 (d62/c15 part-d): the READ-side chunked map/reduce of a long source.

READ-side analog of the write-side 512-token SWA window (d55/d59), same causal
model: a long unstructured source (a Wikipedia article, a long news piece) is far
larger than the in-window per-source budget, so the legacy flat ``md[:budget]``
read DROPPED everything past the cut — the agent (and any downstream synthesizer)
only ever saw the first ~2000 chars and formed thin findings from a fraction of
the document.

Instead, MAP each in-window chunk to a dense factual summary and REDUCE forward:
the running summary FLOWS into the next chunk's summarization (the classic
refine/rolling reduce). So the WHOLE document is read within the window, with NO
truncation and NO fabrication (grounded-only prompts; an empty model reply on a
chunk falls back to that chunk's own text so no section is silently lost).

This module is PURE and transport-agnostic: it takes an async ``summarize``
callback so it is unit-testable with a fake summarizer (mirroring how
:mod:`agent_runtime.article_note` is tested), and the runtime supplies a closure
that runs one real model turn. The full real ``markdown`` is stored UNCHANGED by
the caller, so the c13 write-side verbatim-citation path is untouched — this is a
READ-side summary lane, additive and composable.
"""
from __future__ import annotations

import re
from typing import Awaitable, Callable

# Chars of SOURCE consumed per map step. Sized well under the E4B 32768-token
# (~100k-char) window so a chunk + the running summary + the prompt scaffolding
# never overflow even at the largest realistic per-source budget.
DEFAULT_CHUNK_CHARS = 8000

# The async one-shot summarizer the caller injects: prompt -> model text.
Summarize = Callable[[str], Awaitable[str]]

# Grounded-only MAP prompts (anti-fabrication, d49/d50.1): the model summarizes
# ONLY from the provided text, never from outside knowledge, and RETAINS earlier
# facts as the summary flows forward.
_MAP_FIRST = (
    "You are reading a long source titled {title} <{url}> to extract its facts.\n"
    "Summarize the SECTION below into a dense, factual summary: capture every "
    "concrete fact, figure, date, name, quote and claim it states. Use ONLY what "
    "the text says — do NOT add outside knowledge and do NOT invent anything "
    "that is not present in the text. Keep it under ~{budget} characters.\n\n"
    "SECTION:\n{chunk}\n\nFACTUAL SUMMARY:"
)
_MAP_REFINE = (
    "You are reading a long source titled {title} <{url}> in parts to extract its "
    "facts.\nRUNNING FACTUAL SUMMARY of the earlier parts:\n{running}\n\n"
    "Here is the NEXT section. UPDATE the running summary so it covers BOTH the "
    "earlier parts AND this new section: keep the earlier facts and ADD the new "
    "concrete facts, figures, dates, names, quotes and claims. Use ONLY what the "
    "text says — do NOT add outside knowledge and do NOT invent anything not "
    "present. Keep it under ~{budget} characters and factual.\n\n"
    "NEXT SECTION:\n{chunk}\n\nUPDATED FACTUAL SUMMARY:"
)


def split_chunks(md: str, chunk_chars: int) -> list[str]:
    """Paragraph-aware split of ``md`` into chunks of at most ``chunk_chars``.

    Blank-line-separated paragraphs are packed greedily so a chunk stays whole
    sentences/paragraphs where possible; a single paragraph longer than the limit
    is hard-split so NO chunk ever exceeds ``chunk_chars`` (window safety). Empty
    input yields no chunks."""
    md = (md or "").strip()
    if not md:
        return []
    limit = max(1, int(chunk_chars))
    paras = re.split(r"\n\s*\n", md)
    chunks: list[str] = []
    cur = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(p) > limit:
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(p), limit):
                chunks.append(p[i : i + limit])
            continue
        if cur and len(cur) + 2 + len(p) > limit:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


async def chunked_read(
    markdown: str,
    *,
    summarize: Summarize,
    title: str = "",
    url: str = "",
    char_budget: int = 2000,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
) -> str:
    """Map/reduce read of ``markdown`` into an in-window factual summary.

    Short input (already within ``char_budget``) is returned VERBATIM with no LLM
    call — the short/clean-source path is unchanged. A longer source is split
    into ``chunk_chars`` chunks; each chunk is summarized in-window with the
    running summary flowing forward (refine/reduce), so the final summary reflects
    the WHOLE document rather than only its first ``char_budget`` characters.

    Anti-fabrication: the prompts mandate grounding in the provided text only. An
    empty model reply on a chunk does NOT drop that section — it falls back to
    the running summary plus a bounded slice of the chunk, so the read never loses
    content. The running summary is kept in-window (bounded to ``char_budget``)
    each step — the READ-side SWA window — and the final summary is
    returned bounded to ``char_budget``."""
    md = (markdown or "").strip()
    if len(md) <= char_budget:
        return md
    name = title or url or "(source)"
    chunks = split_chunks(md, max(int(char_budget), int(chunk_chars)))
    if not chunks:
        return md[:char_budget]
    running = ""
    for i, chunk in enumerate(chunks):
        if i == 0:
            prompt = _MAP_FIRST.format(
                title=name, url=url, chunk=chunk, budget=char_budget
            )
        else:
            prompt = _MAP_REFINE.format(
                title=name, url=url, running=running, chunk=chunk, budget=char_budget
            )
        try:
            out = (await summarize(prompt) or "").strip()
        except Exception:  # noqa: BLE001 - a summarizer hiccup must not lose the read
            out = ""
        if not out:
            # Never lose a section: carry the running summary + a bounded slice of
            # this chunk forward instead of dropping it.
            out = f"{running}\n{chunk}".strip()
        running = out[:char_budget]
    return running[:char_budget]
