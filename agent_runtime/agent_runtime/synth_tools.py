"""Stepwise SYNTHESIS over the REAL file tools (s9/c1 — d49 RE-SCOPE).

This REPLACES the s9/c1 first-pass synthesizer (the in-memory
``write_section``/``finish`` ``SynthesisBuilder`` + the ``_synthesis_incomplete``
completeness heuristic). That internal completeness loop was UNRELIABLE on E4B —
measured 2/4 complete; it dumped one blob then false-finished, leaving a dangling
``<section>`` — because the model was reasoning about its *imagined* memory of what
it had written, not the actual document.

d49 RE-SCOPE: the terminal SYNTHESIS deliverable is now built by a
PLANNER-IN-THE-LOOP ReAct loop over the REAL, sandboxed file tools
(``reactive_tools.file_tools`` — ``file_write``/``file_read``). The
synthesizer/agent ACTS ON EMISSIONS — the actual bytes on disk — exactly the way
the neuron drives a recipe file:

    file_write {path}                         create/open the file at the CHOSEN
                                              name+extension (fixes the type)
    file_write {path, content, append:true}   append the NEXT bounded section
    file_read  {path, tail:N}                 READ THE FILE BACK to observe the
                                              ACTUAL current state
    finish                                     the model's "complete" signal — taken
                                              only after it has read a complete file

Reading the real file (not the model's imagined memory) kills BOTH failure modes
at once:

* **false-finish** — the model cannot claim content the file does not contain; the
  read-back is ground truth.
* **truncation** — each append is ONE bounded section, so no single tool call ever
  serializes a long escaped JSON string (the D1 failure that sank the single
  ``format={"output": <string>}`` call); and when a call IS cut short the model
  sees where the file actually stopped and appends the rest.

Type-agnostic: the extension is whatever was chosen for the deliverable
(.md/.txt/.csv/.html/...); the file tool derives the mime as a pure passthrough.

Continuation/completeness lives in the ReAct ORCHESTRATION reading the REAL file —
NOT a hard-coded section counter or a structural completeness heuristic (d48
no-circumvention). This module is pure data + parsing + path derivation; the loop
and the transport round-trips live in :meth:`agent_runtime.runtime.SubAgent._run_synthesis`.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable, Mapping
from html.parser import HTMLParser
from typing import Any, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

from .chunked_read import split_chunks

# WRITER SOURCE BUDGET (MSF/d89): how many chars of each section-assigned source's
# REAL article text the WRITER is fed. The legacy 700 starved the writer to ~0.3% of
# a 200k-char Wikipedia article (d87 retention probe) — the binding constraint on
# thin reports across EVERY model. Raised to a generous default that the caller
# (``runtime.SubAgent._scoped_source_block``) SIZES to the num_ctx window so it never
# reintroduces the d22 overflow→empty-thinking failure. Env ``RA_WRITER_SOURCE_BUDGET``
# overrides the default (default-safe; only the report write path consumes it).
WRITER_SOURCE_BUDGET_ENV = "RA_WRITER_SOURCE_BUDGET"
DEFAULT_WRITER_SOURCE_BUDGET = 12000
# Granularity for the section-relevance chunk selection (``select_relevant_excerpt``):
# small enough that ranking by topic overlap is meaningful, large enough to keep whole
# paragraphs intact as VERBATIM slices (c13 verbatim-citation / d50.1 RAW preserved).
_RELEVANCE_CHUNK_CHARS = 1200
# Word tokens for the cheap lexical overlap rank (no extra model call).
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
# d163 — CONCRETE-FIGURE signal for the write-phase LEAD selection: currency, percentages,
# numbers-with-a-unit, and dated events. A chunk carrying these is exactly the
# figure/date-bearing material a substantive report must quote verbatim, so the lead
# selector PREFERS such chunks (so the writer is PUSHED the figures, never starved). Cheap
# regex, no model call; selection stays VERBATIM (c13/d50.1 RAW).
_FIGURE_RE = re.compile(
    r"\$\s?\d[\d,\.]*"
    r"|\b\d[\d,\.]*\s?%"
    r"|\b\d[\d,\.]*\s?(?:billion|million|trillion|bn|killed|dead|deaths|casualties|"
    r"wounded|injured|troops|soldiers|missiles|drones|barrels|tonnes|tons|km|miles|"
    r"aircraft|ships)\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}\b"
    r"|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"|\b\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)
# Common function words dropped from the relevance rank so they don't dominate the
# score (a topic like "casualties and damage" must not rank a chunk by its "and"s).
# Short CONTENT words (war/oil/gas/un) are kept — length alone is too blunt a filter.
_STOPWORDS = frozenset(
    "the and for with that this from into over under about above after before "
    "are was were has have had will would shall should can could may might must "
    "its their his her our your you they them then than which who whom whose what "
    "when where why how all any both each few more most other some such only own "
    "same too very not but its out off per via vs etc between during against".split()
)


def resolve_writer_source_budget(default: int = DEFAULT_WRITER_SOURCE_BUDGET) -> int:
    """The configured per-source WRITER budget — env ``RA_WRITER_SOURCE_BUDGET`` or default.

    A single resolver so the writer (``render_scoped_sources``) and the Seam-B verifier
    (``claim_verify``) read the SAME budget (d89 lockstep) — else verify would judge a
    claim against a 700-char excerpt the writer never saw. Floors at 120 chars; an
    unparseable env value falls back to ``default`` (never crashes the write path)."""
    raw = os.environ.get(WRITER_SOURCE_BUDGET_ENV)
    if raw is None or not str(raw).strip():
        return max(120, int(default))
    try:
        return max(120, int(str(raw).strip()))
    except (TypeError, ValueError):
        return max(120, int(default))


def select_relevant_excerpt(
    markdown: str, section_topic: str, budget: int, *, prefer_figures: bool = False
) -> str:
    """Up to ``budget`` chars of ``markdown`` most RELEVANT to ``section_topic`` (MSF/d89-a).

    Replaces the d87 flat ``markdown[:budget]`` slice (which only ever showed a long
    article's lede) with a SECTION-RELEVANT selection: rank the article's
    paragraph-aware chunks (``chunked_read.split_chunks``) by cheap LEXICAL overlap with
    the section heading/task and concatenate the top-scoring chunks — restored to
    original DOCUMENT order — up to ``budget``. Excerpts stay VERBATIM raw slices (c13
    verbatim-citation + d50.1 RAW). No extra model call. Falls back to ``markdown[:budget]``
    when there is no topic / no lexical signal / no chunks, and returns a short source
    whole.

    ``prefer_figures`` (d163, the write-phase LEAD): ADD a concrete-figure bonus to each
    chunk's score (currency / percentages / numbers-with-units / dated events) so the lead
    PUSHED to the writer carries the figures and dates a substantive sourced report must
    quote verbatim — instead of an off-figure lede. With ``prefer_figures`` a chunk that
    matched no topic term but DOES carry figures still ranks (so the writer is never starved
    of figures); selection stays bounded by ``budget`` and verbatim."""
    md = (markdown or "").strip()
    budget = max(120, int(budget))
    if len(md) <= budget:
        return md
    topic = (section_topic or "").strip()
    terms = {
        w.lower()
        for w in _WORD_RE.findall(topic)
        if len(w) > 1 and w.lower() not in _STOPWORDS
    }
    if not terms and not prefer_figures:
        return md[:budget]
    # Granularity ≤ budget so a small budget still selects RELEVANT chunks instead of
    # truncating one oversized chunk down to its (possibly off-topic) lede.
    chunk_size = max(200, min(_RELEVANCE_CHUNK_CHARS, budget))
    chunks = split_chunks(md, chunk_size)
    if not chunks:
        return md[:budget]
    # Score = topic-term overlap + (d163) a concrete-figure bonus when prefer_figures, so a
    # figure/date-dense chunk is pulled into the lead even if it is topic-light.
    scored = [
        (
            sum(ch.lower().count(t) for t in terms)
            + (len(_FIGURE_RE.findall(ch)) if prefer_figures else 0),
            idx,
            ch,
        )
        for idx, ch in enumerate(chunks)
    ]
    if not any(score for score, _, _ in scored):
        return md[:budget]  # nothing matched topic/figures — keep the lede deterministically
    picked: list[tuple[int, str]] = []
    used = 0
    for score, idx, ch in sorted(scored, key=lambda x: (-x[0], x[1])):
        if score <= 0 or used >= budget:
            break
        picked.append((idx, ch))
        used += len(ch) + 2
    if not picked:
        return md[:budget]
    picked.sort(key=lambda x: x[0])  # restore document order for readable verbatim flow
    return "\n\n".join(ch for _, ch in picked)[:budget]


# ── READ-SIDE relevance selection (d109): RELEVANCE-SELECT-then-SINGLE-READ ──────────
# The READ-path twin of the write-side ``select_relevant_excerpt`` above, but it RANKS
# by EMBEDDING similarity (the memory store's MiniLM 384-d ``CpuEmbedder``), NOT by the
# crude lexical overlap above — so a research node reading its fetched docs keeps the
# passages SEMANTICALLY closest to its sub-question. It REPLACES the per-source 75-chunk
# map/reduce read (``chunked_read``): instead of summarizing every chunk with a model
# call, split the source into SMALL paragraph-granular RANKING chunks, score each chunk
# against the sub-question, and assemble the TOP chunks (restored to document order) up
# to a token budget. The assembled excerpt is then read in ONE in-window call — FX0/d108
# (swa_test.md) proved attention is faithful across the whole ctx32k window, so the bound
# is the token budget, not attention. There is NO model call in this selection — it is
# pure embedding cosine, so it is cheap and verbatim (c13/d50.1 RAW preserved).

# Paragraph-granular RANKING-chunk size. Flat (no per-call adaptive sizing — the READ is
# one-doc-at-a-time): small enough that a chunk is a focused passage the ranker can score
# sharply, large enough to keep whole paragraphs intact as verbatim slices.
READ_RANKING_CHUNK_CHARS = 3000
# CONTENT token budget for the assembled read (FX0): keep the relevant content under
# ~20k tokens so content + question + history + a generation reserve all stay under the
# 32768 total window guard (swa_test.md: a hard truncation cliff sits at the ctx
# boundary — Ollama drops the front and output collapses once the prompt exceeds 32768).
READ_CONTENT_TOKEN_BUDGET = 20_000
READ_TOTAL_TOKEN_GUARD = 32_768
# Real post-HTML-strip article prose tokenizes ~4 chars/token (FX0 calibration, the
# conservative figure swa_test.md recommends for sizing real documents).
READ_CHARS_PER_TOKEN = 4


def read_content_char_budget(token_budget: int = READ_CONTENT_TOKEN_BUDGET) -> int:
    """Char budget for an assembled read of ``token_budget`` content tokens (FX0 ~4 c/t).

    Floored at one ranking chunk so a tiny configured budget still yields a whole
    passage rather than a sub-chunk sliver."""
    return max(READ_RANKING_CHUNK_CHARS, int(token_budget) * READ_CHARS_PER_TOKEN)


def select_relevant_chunks(
    markdown: str,
    sub_question: str,
    embed: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    *,
    chunk_chars: int = READ_RANKING_CHUNK_CHARS,
    char_budget: int,
) -> tuple[str, int, int]:
    """Top embedding-relevant passages of ``markdown`` for ``sub_question`` (d109).

    Split ``markdown`` into ~``chunk_chars`` paragraph-granular ranking chunks, embed the
    sub-question and every chunk with ``embed`` (MiniLM 384-d, L2-normalized → cosine ==
    dot product), rank chunks by similarity, and assemble the highest-scoring chunks —
    restored to DOCUMENT order — up to ``char_budget`` chars. Returns ``(excerpt,
    total_chunks, selected_chunks)`` so the caller can emit an honest coverage signal
    (M found / X read). The excerpt is VERBATIM (selected raw chunks, no summarization).

    ``embed`` maps a list of strings to an ``(n, d)`` matrix (``CpuEmbedder.embed``).
    Returns the leading ``char_budget`` slice (with ``selected == total``) when there is
    no sub-question or ≤1 chunk (nothing to rank). An embedding error is NOT caught here —
    it is the caller's signal to use the map/reduce fallback."""
    md = (markdown or "").strip()
    if not md:
        return "", 0, 0
    budget = max(chunk_chars, int(char_budget))
    chunks = split_chunks(md, chunk_chars)
    total = len(chunks)
    if total <= 1 or not (sub_question or "").strip():
        # Nothing to rank (single chunk / no query) — keep the leading slice deterministically.
        return md[:budget], total, total
    import numpy as np  # local: numpy rides in via fastembed/memory, not a synth_tools dep

    vecs = np.asarray(list(embed(chunks)), dtype=np.float32)
    qvec = np.asarray(list(embed([sub_question])), dtype=np.float32)[0]
    sims = vecs @ qvec  # cosine similarity (vectors are L2-normalized by CpuEmbedder)
    order = sorted(range(total), key=lambda i: (-float(sims[i]), i))
    picked: list[int] = []
    used = 0
    for i in order:
        clen = len(chunks[i]) + 2
        if picked and used + clen > budget:
            continue  # this chunk overflows — skip it, a smaller lower-ranked one may fit
        picked.append(i)
        used += clen
        if used >= budget:
            break
    picked.sort()  # restore document order for readable verbatim flow
    excerpt = "\n\n".join(chunks[i] for i in picked)[:budget]
    return excerpt, total, len(picked)


# The completion sentinel the synthesizer emits when the deliverable is fully on
# disk. The model emits RAW content sections (its strength — no JSON escaping
# friction); the ORCHESTRATION (the planner-in-the-loop) writes each emission to the
# real file and reads it back, and the model signals completion with this sentinel
# (judged from the ACTUAL file it was shown, not memory). Measured on E4B: asking the
# model to emit file_write/file_read JSON tool calls with embedded content fails for
# a real deliverable (0 parseable calls — the same D1 escaping friction), so the
# tools are driven by the loop, not authored by the model.
DONE_SENTINEL = "<<DONE>>"

# Recognise a bare completion reply (the sentinel, or a short DONE-only line) so a
# legitimate prose section that merely mentions the word "done" is NOT misread.
_DONE_ONLY_RE = re.compile(r"^[\[<*_\s]*done[\]>*_\s.!]*$", re.IGNORECASE)
# A trailing sentinel after a final chunk of content (some models append it).
_DONE_TAIL_RE = re.compile(r"\n*\s*<<\s*done\s*>>\s*\.?\s*$", re.IGNORECASE)


def split_done_signal(text: str) -> tuple[bool, str]:
    """Split a synthesizer turn into ``(is_done, content)``.

    The model writes the next RAW section, OR signals completion. Returns
    ``(True, "")`` when the WHOLE reply is the completion sentinel / a bare ``DONE``
    line; ``(True, <content>)`` when a final chunk is followed by a trailing
    ``<<DONE>>`` (the chunk is kept, the sentinel stripped); ``(False, <content>)``
    otherwise. Requiring the explicit ``<<DONE>>`` marker (or a reply that is ONLY
    the word done) means a section that merely contains the word "done" in prose is
    never mistaken for completion."""
    s = (text or "").strip()
    if not s:
        return False, ""
    if s == DONE_SENTINEL or _DONE_ONLY_RE.match(s):
        return True, ""
    m = _DONE_TAIL_RE.search(s)
    if m:
        return True, s[: m.start()].rstrip()
    return False, s


def unwrap_output_envelope(text: str) -> str:
    """Unwrap a spontaneous ``{"output": "<deliverable>"}`` JSON envelope to its inner
    string (defensive — s9/c1 live finding).

    Measured on E4B: a BARE node (no schema) sometimes wraps its produce output in a
    ``{"output": ...}`` envelope anyway, and on the acyclic ``web_search``->``file_write``
    path that envelope leaked verbatim into the written file (a ``.csv``/``.md`` whose
    bytes were ``{"output": "...."}`` — the o4 "raw JSON envelope" defect). When the
    text is exactly such an object, return the inner deliverable; otherwise return it
    unchanged (a real deliverable that merely contains JSON is untouched)."""
    s = (text or "").strip()
    if not (s.startswith("{") and '"output"' in s and s.endswith("}")):
        return text
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return text
    if isinstance(obj, dict) and isinstance(obj.get("output"), str) and len(obj) <= 3:
        return obj["output"]
    return text


# --------------------------------------------------------------------------- #
# CLOSING-TAG well-formedness (R1 / c1r) — the REAL-FILE gate on <<DONE>>
# --------------------------------------------------------------------------- #
# The read-back loop REDUCED but did not ELIMINATE the false-finish: the model can
# emit <<DONE>> with the top-level HTML container tags still open (c1r R1: a file
# ending at </section> with NO </body></html>, finished=True, 1/3 runs). Before
# accepting the finish, the ORCHESTRATION checks the REAL file (ground truth, d48-
# clean — it reads the actual bytes, NOT a memory heuristic) for unbalanced
# top-level container tags and, if any are open, sends ONE "close the document"
# continuation. Scoped to HTML, whose nested top-level containers are exactly what a
# truncated emission leaves dangling; a bare HTML fragment (no <html>/<body>) is not
# faulted, so a complete document is never nagged.
_HTML_CONTAINER_TAGS: tuple[str, ...] = ("body", "html")


def html_close_gap(doc: str) -> list[str]:
    """Closing tags an HTML deliverable OPENED but never CLOSED, in append order.

    For each top-level container (``<body>``, ``<html>``) whose OPENING tags
    outnumber its CLOSING tags in ``doc``, the missing ``</body>``/``</html>`` is
    returned — ordered innermost-first (``["</body>", "</html>"]``) so APPENDING them
    in that order closes the document correctly. Returns ``[]`` when the document is
    balanced OR uses no such container, so a complete (or fragment) file is never
    nagged. Reads ``doc`` (the real file's bytes) — not the model's memory (d48)."""
    low = (doc or "").lower()
    gaps: list[str] = []
    for tag in _HTML_CONTAINER_TAGS:
        opens = len(re.findall(rf"<{tag}(?:\s[^>]*)?>", low))
        closes = len(re.findall(rf"</{tag}\s*>", low))
        if opens > closes:
            gaps.append(f"</{tag}>")
    return gaps


# --------------------------------------------------------------------------- #
# DETAILED-INTENT signal (c8 R2) — the markdown/text first-turn completeness gate
# --------------------------------------------------------------------------- #
# The HTML path has html_close_gap to catch a premature <<DONE>> (unclosed tags).
# The markdown/text path has no structural tag to balance, so a small model can
# one-shot the whole report + <<DONE>> in ONE turn and DROP a requested section +
# the sources list, with nothing to catch it (c8r REVISE). The fix mirrors the
# AUTONOMY REBUILD P2C — ``is_detailed_task`` / ``_DETAILED_INTENT_RE`` are DELETED.
# The keyword flag gated an engine continuation nudge on the raw write loop (itself
# deleted): a hardcoded intent regex firing regardless of the model's own reasoning —
# the exact fabrication class the owner charter bans. Depth now comes from the
# writer SPECS + the model's judgment; delivery honesty from the target-artifact gate.


# --------------------------------------------------------------------------- #
# RP-AUDIT F3 (d319/d341/d330) — the DEAD HTML-format-pinned output helpers are
# REMOVED. Deleted here: the output-MODIFIERS ``strip_wrapper_closers``,
# ``strip_wrapper_openers`` and ``dedupe_html_documents`` (they stripped/deduped/
# rewrote the model's bytes) and the HTML-pinned read-only PREDICATES
# ``top_level_html_doc_count`` and ``begins_html_document`` (plus their now-orphaned
# ``_HTML_DOC_CLOSE_RE`` / ``_HTML_DOCTYPE_RE`` / ``_HTML_*_OPEN_RE`` /
# ``_HTML_HEAD_BLOCK_RE`` / ``_HTML_BODY_CLOSE_RE`` regexes). Their live call sites
# were already retired (runtime.py; RP-3c/d330 replaced the HTML-pinned re-emission
# guard with the FORMAT-NEUTRAL ``document_restart`` / ``section_reemission`` /
# ``html_close_gap`` trio, which STAYS). Retaining these behind exports + unit tests
# left format-baked, output-MODIFYING code "one import away" from being re-wired — a
# latent violation of the anti-fab charter (the engine authors/fixes/modifies
# NOTHING). ``test_sf1_reactive_coherence_retired`` now FAILS if any is re-DEFINED or
# re-EXPORTED, closing that gap. (``_HTML_CONTAINER_TAGS`` is KEPT — the surviving
# ``html_close_gap`` completeness gate reads it.)
# --------------------------------------------------------------------------- #


# s14/a8 (d149) — internal CONTEXT-ASSEMBLY scaffolding that must NEVER reach the rendered
# deliverable. The raw-content write loop folds these headers into the model's USER turn (the
# overall goal, prior-step findings, the across-parts continuation note, a tool-output path,
# a read-back file slice); a small writer intermittently ECHOES one back as if it were content,
# and the loop would then WRITE that scaffolding into the file. These exact internal phrases do
# not occur in a genuine report, so a line that begins with (or embeds) one is internal
# scaffolding and is dropped/cut.
_SCAFFOLDING_MARKERS: tuple[str, ...] = (
    "SOURCES & FINDINGS FROM PRIOR STEP",
    "TOOL OUTPUT (",
    "INPUTS FROM PRIOR STEPS:",
    "OVERALL GOAL (the user's full request",
    "PRIOR CONVERSATION (the user is continuing",
    "CURRENT TASK:",
    "A document is being written ACROSS PARTS",
    "The deliverable is ALREADY written to the file",
    "FETCHED SOURCE CONTENT",
    "FILE SLICE (offset",
    "SOURCE BUDGET REACHED",
)
# Literal tool-call invocation text a confused model may emit as prose; the write loop must
# never persist it as document content (the a7 leak: ``file_update(old=…, new=…)`` written
# verbatim). The reviewer/research tool names followed by ``(`` are the call signature.
_TOOLCALL_TOKENS: tuple[str, ...] = (
    "file_update(", "file_read(", "file_write(", "load_source(",
    "web_search(", "web_fetch(",
)
# An ORPHAN tool-call ARGUMENT line — the continuation of a multi-line call whose opener
# was already cut. A line starting ``old=`` / ``new=`` / ``path=`` / ``offset=`` / … is never
# legitimate report content (HTML/markdown/prose never opens a line with a bare ``key=``).
_TOOLCALL_ARG_RE = re.compile(
    r"^(old|new|path|offset|length|tail|sid|count|chunk|query|url|args|arguments)\s*=",
    re.IGNORECASE,
)


def strip_internal_scaffolding(content: str) -> str:
    """Remove internal context-assembly scaffolding / tool-call text from a raw emission.

    A small writer model intermittently ECHOES the headers the loop folded into its USER
    turn (overall-goal / prior-step findings / across-parts / tool-output-path / file-slice
    scaffolding) or emits a literal tool-call (``file_update(...)``) as if it were content.
    Without this, the shared raw-content file loop would WRITE that scaffolding into the
    deliverable (the s14/a8 d149 leak). Each line is checked: a line that BEGINS with an
    internal marker or a tool-call token is dropped entirely; a line that EMBEDS one mid-text
    is CUT at the earliest such point (keeping any real content before it); a bare
    ``{'path': …}`` dict left behind by a folded TOOL OUTPUT block is dropped. Everything else
    is kept verbatim, so legitimate report content is untouched."""
    if not content:
        return content
    cut_markers = _SCAFFOLDING_MARKERS + _TOOLCALL_TOKENS
    out: list[str] = []
    for line in content.splitlines():
        s = line.strip()
        if any(s.startswith(m) for m in cut_markers):
            continue
        # a path-only dict left by a folded TOOL OUTPUT block (never legit content).
        if s.startswith("{'path'") or s.startswith('{"path"'):
            continue
        # an ORPHAN tool-call argument line (the continuation of a multi-line call whose
        # opener was already cut, e.g. ``old="…", new="…")``) — never a legit content line.
        if _TOOLCALL_ARG_RE.match(s) or s == ")":
            continue
        # mid-line: cut at the earliest embedded internal marker / tool-call token.
        cut = len(line)
        for m in cut_markers:
            j = line.find(m)
            if j != -1 and j < cut:
                cut = j
        if cut < len(line):
            head = line[:cut].rstrip()
            if head:
                out.append(head)
            continue
        out.append(line)
    return "\n".join(out)


def _keep_first(text: str, pattern: "re.Pattern[str]") -> str:
    """Remove every match of ``pattern`` AFTER the first (keep the opener once)."""
    matches = list(pattern.finditer(text))
    if len(matches) <= 1:
        return text
    for m in reversed(matches[1:]):
        text = text[: m.start()] + text[m.end():]
    return text


def _keep_last(text: str, pattern: "re.Pattern[str]") -> str:
    """Remove every match of ``pattern`` BEFORE the last (keep the closer once)."""
    matches = list(pattern.finditer(text))
    if len(matches) <= 1:
        return text
    for m in reversed(matches[:-1]):
        text = text[: m.start()] + text[m.end():]
    return text


# RP-1 (d319/d311): repair_table_cells (the d168 cell-close surgery) RETIRED — engine no longer fixes/authors the model's output.


# RP-1 (d319/d311): has_duplicate_html_structure / enforce_single_html_document RETIRED — engine no longer fixes/authors the model's output.


# d171 — a per-section writer's shell template carries a scaffold PLACEHOLDER COMMENT
# (``<!-- Subsequent sections will be added here -->``) between the page shell and the
# appended sections; the comment is a write-time scaffold, never report content, but it
# survives every structural pass (the dedup/wrapper passes only touch tags, not comments)
# and leaks into the served document. This strips those scaffold/placeholder comments —
# matched by their scaffold WORDING so a genuine authored comment is never removed — as a
# cosmetic output cleanup. Idempotent; content-preserving (only comment nodes are touched).
_SCAFFOLD_COMMENT_RE = re.compile(
    r"[ \t]*<!--\s*(?:subsequent\s+sections?|sections?\s+(?:will\s+be|to\s+be)\s+added|"
    r"(?:more|additional|further|other)\s+sections?|insert\s+.*?\s+here|"
    r"placeholder|content\s+(?:goes|will\s+go)\s+here|add\s+(?:more|sections?)\b)"
    r".*?-->[ \t]*\n?",
    re.IGNORECASE | re.DOTALL,
)


def _html_escape_text(s: str) -> str:
    """Minimal HTML text escape (``&``/``<``/``>``) for verbatim source titles."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _html_escape_attr(s: str) -> str:
    """Minimal HTML attribute escape (``&``/``"``) for verbatim source URLs."""
    return str(s).replace("&", "&amp;").replace('"', "&quot;")


# --------------------------------------------------------------------------- #
# REAL SOURCE-URL INDEX (c12 #5) — carry the ACTUAL fetched URLs to the
# synthesizer so it cites them VERBATIM instead of fabricating placeholders.
# --------------------------------------------------------------------------- #
# ROOT CAUSE of #5: on the long concurrent-multi-node path the research nodes fetch
# real URLs, and those URLs DO reach the synthesizer — but scattered through a huge
# prompt, buried inside each source's 2000-char article body. Building section by
# section, the small model cannot reliably reconstruct them and invents generic
# ``[CyberWatch Report, 2025]`` placeholders (or drops citations). The fix is d17
# context-feeding (feed the node the data it cannot discover) + d46/d49 no-fabrication:
# assemble the ACTUAL fetched URLs the orchestration already holds into ONE compact,
# prominent, deduplicated list and hand it to the synthesizer as the authoritative,
# cite-ONLY-from-this source set. The model still REASONS about where/how to cite —
# this is real data fed, NOT a hard-coded citation template.
def collect_fetched_sources(
    tool_values: Optional[Iterable[object]],
) -> list[tuple[str, str]]:
    """Collect ``(title, url)`` for every real fetched source, deduped by URL.

    Walks the upstream tool VALUES (each a research node's
    ``{"fetched": [{"title", "url", "markdown"}, …]}``) and returns the distinct
    fetched sources in first-seen order. A value with no ``fetched`` list (a plain
    search-results value, a prose-only node) contributes nothing, so the index is
    built only from sources the agent actually READ."""
    sources: list[tuple[str, str]] = []
    seen: set[str] = set()
    for tv in tool_values or ():
        if not isinstance(tv, Mapping):
            continue
        for art in tv.get("fetched") or ():
            if not isinstance(art, Mapping):
                continue
            url = str(art.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append((str(art.get("title") or "").strip(), url))
    return sources


# --------------------------------------------------------------------------- #
# NAVIGABLE SINGLE-PAGE assembly (s9/c13, d55/d57) — wrap a per-section HTML
# fragment into ONE well-formed SPA with a nav built FROM the model's headings.
# --------------------------------------------------------------------------- #
# The per-section bounded write phase has each section emit ONLY its body content
# (a bare ``<h2>…`` fragment) so its sources stay inside the ~512-tok SWA window — so
# no section emits the page wrapper or a nav. This DETERMINISTIC structural pass (the
# same class as :func:`enforce_single_html_document` / :func:`html_close_gap`: it
# reads the real bytes, fabricates NO content) gives the assembled fragment exactly
# one ``<!DOCTYPE>/<html>/<head>/<body>`` wrapper and a nav menu built FROM the
# model-authored ``<h1>/<h2>`` headings (anchors injected where missing). Section
# content + headings stay 100% the model's; only navigation + the wrapper are assembled.
_HTML_HEADING_RE = re.compile(r"<(h[12])(\s[^>]*)?>(.*?)</\1\s*>", re.IGNORECASE | re.DOTALL)
# RP-3c (d330): OUTPUT-AGNOSTIC section detection. The re-emission guard must work for
# ANY output format, not just HTML, so a MARKDOWN top-level heading (a line opening with
# one or two ``#`` ATX markers) is recognised as a section landmark alongside the HTML
# ``<h1>/<h2>``. An HTML document has no line-start ``# `` marker and a Markdown document
# has no ``<h1>`` tag, so each format matches only its own idiom (no cross-format noise).
_MD_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,2}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
# RP-1 (d319/d311): _HTML_ID_ATTR_RE / _heading_slug RETIRED — engine no longer fixes/authors the model's output.


# RP-1 (d319/d311): _HTML_HEADING123_RE / _SPA_NAV_RE RETIRED — engine no longer fixes/authors the model's output.
_TAG_TOKEN_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)((?:\s[^>]*)?)(/?)>")
_LIST_CONTAINER_TAGS: frozenset[str] = frozenset({"ul", "ol"})


# RP-1 (d319/d311): _DocStructureParser RETIRED — engine no longer fixes/authors the model's output.


# RP-1 (d319/d311): _dedupe_element_ids / _wrap_orphan_list_items RETIRED — engine no longer fixes/authors the model's output.


# RP-1 (d319/d311): reconcile_doc_structure RETIRED — engine no longer fixes/authors the model's output.


# --------------------------------------------------------------------------- #
# SECTION-LEVEL de-duplication (s9/c14, d59) — collapse a re-emitted body-level
# report pass / repeated heading-FAMILY to EXACTLY ONE pass of each section.
# --------------------------------------------------------------------------- #
# :func:`enforce_single_html_document` only dedups duplicate ``<!DOCTYPE>``/``<html>``
# DOCUMENT wrappers. The long-report write loop ALSO over-produces at the BODY level:
# it re-emits ``<header>/<nav>/<h1>`` and the whole section sequence again WITHOUT a
# fresh document wrapper, so the duplicate pass slips past the wrapper-only gate (c13r
# 4/4 long runs: 2–3 full report passes; "Public Law History" ×3). This deterministic
# structural pass — the same class as :func:`enforce_single_html_document` /
# :func:`html_close_gap`: it reads the real bytes and fabricates NO content — keeps the
# FIRST occurrence of each section and drops every later re-emission of the same
# heading FAMILY. Duplication is judged by heading FAMILY (significant-token overlap),
# NOT string equality, because near-dup wording DRIFTS across passes (c13r: "Military
# Losses and Casualties Assessment" vs "…and Casualties"; "Sources and Citations" vs
# "Sources"). Keeping the FIRST (grounded, source-scoped) occurrence preserves the c13
# citation win and never invents content (d60 no-fabrication: this is drop-only — the
# active faithfulness/grounding verification is a SEPARATE concern, not a regex filter
# here).
_FAMILY_MATCH_RATIO = 0.67
_HEADING_STOPWORDS: frozenset[str] = frozenset(
    "a an the and or of to in on for with from by at as into over under after before "
    "during this that these those it its their his her our your is are was were be "
    "been being".split()
)
# A wrapper-OPEN tag (``<div …>``/``<section …>``/``<header>``/…) sitting immediately
# before a heading (only whitespace between) is absorbed into that heading's segment,
# so when the segment is dropped its own opener AND matching close go together and the
# container nesting stays balanced.
_WRAPPER_OPEN_TAIL_RE = re.compile(
    r"<(?:div|section|article|header|main)(?:\s[^>]*)?>\s*$", re.IGNORECASE
)
# A leading enumerator on a heading ("III.", "1)", "Section 2 —") is stripped before
# the family is computed so an enumerated re-emission still matches its first pass.
_LEADING_ENUMERATOR_RE = re.compile(
    r"^\s*(?:(?:section|part|chapter|appendix|article)\s+)?"
    r"(?:[0-9]+|[ivxlcdm]+)\s*[.)\]:\-–—]\s*",
    re.IGNORECASE,
)


def _heading_family(inner: str) -> frozenset[str]:
    """The FAMILY key of a heading — its significant tokens, drift-tolerant.

    Strips inner markup, drops a ``': subtitle'`` drift tail and a leading enumerator,
    lowercases, and returns the significant (non-stopword) word set. Two headings with
    DIFFERENT wording but the same core ("Military Losses and Casualties Assessment" vs
    "Military Losses and Casualties") yield overlapping families so
    :func:`_families_match` treats them as the same section (the c13r drift gotcha)."""
    bare = re.sub(r"<[^>]+>", " ", inner or "")
    bare = bare.split(":", 1)[0]
    bare = _LEADING_ENUMERATOR_RE.sub(" ", bare.lower())
    tokens = re.findall(r"[a-z0-9]+", bare)
    # Keep significant tokens: drop grammatical stopwords and stray single letters, but
    # KEEP digit tokens — a trailing "1"/"2" is the DISTINGUISHING part of "Page 1" vs
    # "Page 2" (a legit multi-page chain), so they must NOT collapse to one family.
    sig = frozenset(
        t for t in tokens
        if t not in _HEADING_STOPWORDS and (len(t) > 1 or t.isdigit())
    )
    return sig or frozenset(tokens)


def _families_match(a: frozenset[str], b: frozenset[str]) -> bool:
    """True when two heading families denote the SAME section (token containment).

    Equal-when-empty; otherwise the smaller family must be ``>= _FAMILY_MATCH_RATIO``
    contained in the larger. Containment (not Jaccard) so a drifted SUPERSET heading
    ("…Casualties Assessment" over "…Casualties") still matches, while two genuinely
    distinct sections that merely share one generic word ("Economic Impact" vs
    "Environmental Impact": 1/2 = 0.5) do NOT collapse."""
    if not a or not b:
        return a == b
    inter = len(a & b)
    if not inter:
        return False
    return inter / min(len(a), len(b)) >= _FAMILY_MATCH_RATIO


def _section_headings(doc: str) -> list[str]:
    """The inner text of every top-level section heading, in document order.

    OUTPUT-AGNOSTIC (RP-3c/d330): recognises BOTH an HTML ``<h1>``/``<h2>`` heading and a
    Markdown top-level (``#``/``##``) ATX heading, so the re-emission guard's section
    predicate works for an HTML OR a Markdown deliverable. Each format contributes only
    its own idiom's headings (an HTML doc has no line-start ``# `` marker; a Markdown doc
    has no ``<h1>`` tag), so a single-format document behaves exactly as before."""
    doc = doc or ""
    heads = [m.group(3) for m in _HTML_HEADING_RE.finditer(doc)]
    heads += [m.group(1) for m in _MD_HEADING_RE.finditer(doc)]
    return heads


# RP-1 (d319/d311): collapse_duplicate_sections RETIRED — engine no longer fixes/authors the model's output.


# RP-1 (d319/d311): the whole d173/d174 deterministic assembly layer (collapse_duplicate_section_ids /
# dedupe_source_lists / rebuild_section_nav / assemble_report_spa / enforce_single_h1 and their helper
# regexes) is RETIRED — the engine no longer fixes/authors the model's output; coherence lives in the
# writer/reviewer SPECS (d310/d313/d319).


# --------------------------------------------------------------------------- #
# FINAL-SECTION TRUNCATION marker (s13/P1-report) — a num_predict/section cut.
# --------------------------------------------------------------------------- #
# A deliverable cut off at ``num_predict`` ends MID-SENTENCE: the last VISIBLE character
# is a letter/comma (a dangling word), not sentence-terminating punctuation, even though
# the wrapper close tags follow (B8a2: "…the IRGC killed thousands … On February</body>
# </html>" — the final bio cut off mid-word). The structural close-gap/dedup passes do
# NOT see it (the tags balance); these two helpers detect the dangling-text symptom and
# trim a SHORT cut-off fragment so no truncation marker reaches the served document.
# Sentence-ENDING characters: ., !, ?, …, :, ;, closing quotes, and closing brackets.
_SENTENCE_END_CHARS: frozenset[str] = frozenset(".!?…:;\"'’”)]}")
# ONLY the trailing wrapper closes (``</body>``/``</html>``) the write loop auto-appends
# after a cut emission are peeled (+ whitespace), to reach the last authored character.
# A real mid-sentence truncation leaves the dangling text DIRECTLY before the wrapper
# close (the section's <p> never closed), so after peeling the last char is a letter. A
# COMPLETE final element — a closed ``</a></li></ul>`` Sources list, a closed ``</p>`` —
# leaves a ``>`` as the last char and is correctly read as NOT truncated (so a report
# ending in a URL list is never mistaken for a cut sentence).
_TRAILING_WRAPPER_CLOSE_RE = re.compile(
    r"(?:\s*</(?:body|html)\s*>)+\s*$", re.IGNORECASE
)


def _visible_body_split(doc: str) -> tuple[str, str]:
    """Split ``doc`` into (authored body, trailing ``</body></html>`` run). Trimmed."""
    s = (doc or "").rstrip()
    m = _TRAILING_WRAPPER_CLOSE_RE.search(s)
    if m:
        return s[: m.start()].rstrip(), s[m.start():].strip()
    return s, ""


def has_truncation_marker(doc: str, *, min_chars: int = 80) -> bool:
    """True when the deliverable ends MID-SENTENCE — a ``num_predict``/section truncation.

    Peels the trailing ``</body>``/``</html>`` the loop auto-appends and inspects the last
    authored character: a letter, comma or dash (a dangling word — the section's ``<p>``
    was never closed) signals a truncated final section. A COMPLETE final element ends in
    ``>`` (a closed ``</p>`` / Sources ``</a></li></ul>``), a sentence ends in terminating
    punctuation, and a complete figure ends in a digit — none of those are faulted. A
    short deliverable (headlines, a terse answer) is never faulted (``min_chars`` floor on
    the visible text). Reads the bytes only — no model call, deterministic, idempotent."""
    body, _closes = _visible_body_split(doc)
    if len(re.sub(r"<[^>]+>", "", body)) < min_chars:
        return False
    last = body[-1] if body else ""
    if not last:
        return False
    return last not in _SENTENCE_END_CHARS and not last.isdigit() and last != ">"


# RP-1 (d319/d311): trim_dangling_sentence RETIRED — it EDITED the model's output (trimming a
# truncated tail); the engine no longer fixes/authors output. ``has_truncation_marker`` above is
# KEPT: it is a detect-only predicate (reads bytes, informs a decision, never edits).


# --------------------------------------------------------------------------- #
# OUTLINE-AS-PRIMARY backstop (s13/FX-writer, d106 #7). collapse_duplicate_sections
# kills near-identical re-emissions (same heading FAMILY). The B8a failure was
# different: the writer authored a findings-driven section set AND a SECOND parallel
# set from the agent outline, appended as a tail — two headings that both denote the
# SAME OUTLINE SECTION but drifted in wording past the family-overlap threshold
# ("Cost Analysis" vs the outline's "Cost and Damage Assessment"), so the served doc
# carried three conflicting "Section 3"s. The PROMPT fix (outline is the COMPLETE
# section list, no parallel set) removes the cause; this is the deterministic,
# drop-only backstop for the residual close-wording case: each doc heading is mapped
# to the outline SLOT it matches, and a later heading claiming an already-claimed slot
# is dropped. Conservative on purpose — a heading that matches NO outline slot is KEPT
# (a genuinely new section the planner legitimately added is never dropped). d48/d60-
# clean: reads the real bytes, fabricates nothing.
# --------------------------------------------------------------------------- #
# EMPTY-NODE-NO-FABRICATE (s13/FX-writer, d106 #6). A research node that yielded 0
# sources / timed out (FX-loop's _unsupported_leaf) leaves its write-section node with
# NO assigned sources after source-scoping + coverage. The B8a Timeline section was
# exactly this — assigned to B1 which fetched 0 — and the writer invented every dated
# event from memory. The deterministic guarantee: such a section is marked UNSUPPORTED
# and its writer is told to fabricate NOTHING (the runtime also feeds it no source text
# to copy). This is the per-section task the write side stamps onto an unsupported node;
# it keeps decomposition the planner's (the planner chose the section) while making
# fabrication impossible (no sources fed + an explicit no-fabricate instruction).
UNSUPPORTED_SECTION_INSTRUCTION = (
    "This planned section has NO supporting sources — its research yielded nothing "
    "(an empty or timed-out research node). Write ONLY the section heading followed by "
    "a single explicit line marking it unsupported, exactly: "
    "'_No sources were found for this section; it is reported as UNSUPPORTED rather than "
    "fabricated._' Do NOT invent ANY content for it — no dates, figures, timelines, "
    "names, quotes, or citations (d60 no-fabrication). Write nothing beyond that one line."
)


def section_reemission(new_content: str, existing_doc: str) -> bool:
    """True when ``new_content`` only RE-EMITS sections already on the real file (c14).

    The loop-side guard for the per-section ReAct write loop: a chunk that carries one
    or more headings whose families are ALL already present in ``existing_doc`` adds NO
    new section — it is a re-run of the already-written set. The orchestration drops it
    rather than appending a duplicate pass (judged from the REAL file, d48-clean). A
    chunk that introduces ANY new heading family (a genuine next section) returns False,
    and a chunk with no heading at all returns False (it is ordinary continuation
    prose). Returns False when the file holds no headings yet (nothing to repeat)."""
    new_heads = _section_headings(new_content)
    if not new_heads:
        return False
    existing_fams = [_heading_family(h) for h in _section_headings(existing_doc)]
    if not existing_fams:
        return False
    for h in new_heads:
        fam = _heading_family(h)
        if not any(_families_match(fam, ef) for ef in existing_fams):
            return False
    return True


# --------------------------------------------------------------------------- #
# OUTPUT-AGNOSTIC document-restart signal (RP-3c, d330) — the format-neutral
# replacement for the HTML-pinned ``begins_html_document`` + ``top_level_html_doc_count``
# predicates the c10 re-emission guard used to DECIDE persist/stop/nudge.
# --------------------------------------------------------------------------- #
# A small model nudged to "continue" an ALREADY-written deliverable often RESTARTS the
# whole artifact from the top instead of writing the next section — re-emitting the same
# opening (``<!DOCTYPE html><html>…`` for HTML, ``# Title …`` for Markdown, the header row
# for a CSV, the first lines for code). The old guard detected this ONLY for HTML (a fresh
# ``<!doctype>``/``<html>`` plus a ``</html>`` count). ``document_restart`` detects it in a
# FORMAT-NEUTRAL way: the new chunk reproduces the opening the real file ALREADY begins with
# — no format token is hard-coded, so it works for any output shape. Like every guard
# predicate it reads the REAL bytes and fabricates nothing (d48-clean); it only informs the
# orchestration's persist/stop/nudge DECISION and never edits the model's output.
_RESTART_MIN_OVERLAP = 24
# A document RE-EMISSION re-writes the whole artifact, so the re-emitted chunk is
# DOCUMENT-SIZED. A short repeated FRAGMENT (e.g. the model churning ``<p>another small
# part</p>`` every turn without ever closing the document — the R2 non-convergence case)
# reproduces the file's opening too, but it is NOT a document restart: it is churn that must
# fall through to the loop's non-convergence path, not be dropped as a finished re-emission.
# This floor separates "re-emitted the whole document" from "repeated a tiny fragment"
# WITHOUT any format token (a document is bigger than a naked inline fragment in any format).
_RESTART_MIN_DOC_CHARS = 64


def _normalized_head(doc: str, window: int) -> str:
    """The document's leading ``window`` chars, whitespace-collapsed and lowercased —
    a format-neutral fingerprint of how the artifact OPENS."""
    return re.sub(r"\s+", " ", (doc or "").strip().lower())[:window]


def document_restart(
    new_content: str, existing_doc: str, window: int = 120
) -> bool:
    """True when ``new_content`` RE-OPENS the whole artifact ``existing_doc`` already begins with.

    Format-neutral (RP-3c/d330): compares the normalized leading fingerprint of each. When
    the new chunk reproduces the existing document's own opening (its first
    ``_RESTART_MIN_OVERLAP`` normalized chars match) AND is itself DOCUMENT-SIZED
    (``_RESTART_MIN_DOC_CHARS``), the model RESTARTED the whole artifact — a re-emission of
    the shell, not new content — so the orchestration can STOP or nudge rather than
    concatenate a second copy. Returns False when either side is too short to judge, when the
    new chunk opens differently (a genuine next section never reproduces the document's
    opening), or when the new chunk is a short repeated FRAGMENT rather than a whole-document
    re-emission (the R2 non-convergence churn case). Reads the real bytes, fabricates nothing
    (d48-clean)."""
    head_new = _normalized_head(new_content, window)
    head_old = _normalized_head(existing_doc, window)
    if len(head_new) < _RESTART_MIN_OVERLAP or len(head_old) < _RESTART_MIN_OVERLAP:
        return False
    if head_new[:_RESTART_MIN_OVERLAP] != head_old[:_RESTART_MIN_OVERLAP]:
        return False
    # a whole-document re-emission is document-sized, not a tiny repeated fragment
    return len((new_content or "").strip()) >= _RESTART_MIN_DOC_CHARS


# RP-1 (d319/d311): SECTION_ANCHOR / plant_section_anchor / choose_section_anchor / anchored_insert_args / strip_section_anchor RETIRED — engine no longer fixes/authors the model's output.


def collect_fetched_sources_full(
    tool_values: Optional[Iterable[object]],
) -> list[dict[str, str]]:
    """Collect ``{title, url, markdown}`` for every real fetched source (s9/c13, d56).

    The RICH analogue of :func:`collect_fetched_sources`: same first-seen,
    URL-deduped order (so a 1-based index into this list is the run's STABLE global
    SOURCE id the planner assigns per section), but it RETAINS each source's
    extracted article ``markdown`` so the per-section write phase can feed a section
    ONLY its assigned sources' real text — kept inside the model's ~512-token sliding
    window so it copies real figures/URLs instead of fabricating placeholders."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for tv in tool_values or ():
        if not isinstance(tv, Mapping):
            continue
        # SoC ENGINE-THIN (SA-4/d254): a gather node's source artifacts ride EITHER the web key
        # ``fetched`` OR the source-agnostic ``records`` key a NON-WEB bundle (codebase/vector-
        # db/future) emits — collected the SAME way (URL-deduped, markdown retained) so the
        # writer grounds in a non-web source through the identical chain_sources path. A web tv
        # carries only ``fetched`` (no ``records``), so its harvest is byte-identical.
        for art in list(tv.get("fetched") or ()) + list(tv.get("records") or ()):
            if not isinstance(art, Mapping):
                continue
            url = str(art.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            rec = {
                "title": str(art.get("title") or "").strip(),
                "url": url,
                "markdown": str(art.get("markdown") or "").strip(),
            }
            # CARRY THE WHOLE-DOC SUMMARY THROUGH (MSF/d89-a, fixes seam ③): the N3
            # chunked-read map/reduce summary (whole-document coverage) is set on the
            # fetched record by ``runtime._read_fetched``. The legacy collector dropped
            # it, so the writer only ever saw raw ``markdown[:700]`` and never the
            # whole-doc coverage. Retain it additively so ``render_scoped_sources`` can
            # prepend the grounded summary before the verbatim excerpt. Absent when
            # chunked-read was OFF/unneeded → byte-identical (no ``summary`` key).
            summary = str(art.get("summary") or "").strip()
            if summary:
                rec["summary"] = summary
            out.append(rec)
    return out


def render_source_catalog(sources: Sequence[Mapping[str, str]]) -> str:
    """A compact NUMBERED ``[i] title — url`` catalog for the WRITE PLANNER (s9/c13).

    Shown to the per-section write planner so it can REASON which source number(s)
    each section uses and record them as that node's ``source_ids`` (the d56 (R)
    model-authored source→section assignment). No article bodies — just the numbered
    title+url list, so the authoring prompt stays small. Returns ``""`` for no
    sources (a no-source report authors plain per-section nodes, no scoping)."""
    if not sources:
        return ""
    lines = ["AVAILABLE SOURCES (cite by these numbers — set each section's source_ids):"]
    for i, s in enumerate(sources, 1):
        title = str(s.get("title") or "").strip()
        url = str(s.get("url") or "").strip()
        lines.append(f"[{i}] {title or url} — {url}")
    return "\n".join(lines)


def render_scoped_sources(
    sources: Sequence[Mapping[str, str]],
    source_ids: Sequence[int],
    *,
    excerpt_budget: Optional[int] = None,
    section_topic: str = "",
) -> str:
    """Render ONLY this section's assigned sources, RICH, for its write turn (s9/c13, d56; MSF/d89).

    The (F) feed-scoping half of d56: given the run's global ``sources`` list and a
    section node's planner-assigned 1-based ``source_ids``, emit a block with each
    assigned source's number, title, URL and — MSF/d89, fixing the d87 700-char
    starvation — (1) the whole-doc grounded SUMMARY when carried through (coverage),
    (2) a SECTION-RELEVANT verbatim excerpt up to ``excerpt_budget`` chars (the binding
    constraint, raised from 700 to a window-sized budget), and (3) a "this source has
    MORE" marker when the article is longer than the excerpt — so the model reasons
    about coverage instead of treating the sliver as the whole article. ``excerpt_budget``
    None ⇒ the configured ``resolve_writer_source_budget()`` (the caller sizes it to the
    num_ctx window). Placed at the END of the section's user turn (nearest the cursor)
    so the real figures + URLs sit inside the model's sliding window — the SWA fix.
    Out-of-range ids are skipped; an empty/over-range selection returns ``""`` (the node
    falls back to the full upstream index — graceful for the 1-section degenerate case)."""
    if not sources or not source_ids:
        return ""
    budget = resolve_writer_source_budget() if excerpt_budget is None else max(120, int(excerpt_budget))
    picked: list[tuple[int, Mapping[str, str]]] = []
    for i in source_ids:
        if isinstance(i, int) and 1 <= i <= len(sources):
            picked.append((i, sources[i - 1]))
    if not picked:
        return ""
    lines = [
        "SOURCES FOR THIS SECTION — these are the REAL articles already fetched and "
        "read, each labelled by its [S#] number with its real URL. Write this section's "
        "facts/figures FROM the text below and cite each claim with that source's [S#] "
        "and its matching URL VERBATIM. If this section has a TABLE with a source/citation "
        "column, or a Sources list, fill EVERY such cell/entry with a REAL [S#] and its "
        "real URL from the list below — NEVER leave a worded stand-in like \"URL "
        "Placeholder\", \"Source Placeholder\", \"Source N Title\", \"[Name, 2025]\", a "
        "\"URL 1\"-style label, or a bare publication name/date. If you cannot ground a "
        "row in a real [S#] from this list, drop the row rather than placeholder it; and "
        "never cite a URL not listed here:",
    ]
    for i, s in picked:
        title = str(s.get("title") or "").strip()
        url = str(s.get("url") or "").strip()
        full = str(s.get("markdown") or "").strip()
        summary = str(s.get("summary") or "").strip()
        excerpt = select_relevant_excerpt(full, section_topic, budget, prefer_figures=True)
        block = [f"\n[S{i}] {title or url} — {url}"]
        if summary:
            block.append(
                "WHOLE-SOURCE SUMMARY (grounded factual overview of the FULL article — "
                "use it for COVERAGE of what the source establishes):"
            )
            block.append(summary)
            block.append(
                "RELEVANT EXCERPT (verbatim from the article — quote figures/dates/names "
                "from HERE):"
            )
        block.append(excerpt)
        if len(full) > len(excerpt):
            block.append(
                f"[showing the {len(excerpt)} most-relevant chars of {len(full)} total — "
                "this source has MORE; do not assume this excerpt is the complete article]"
            )
        lines.append("\n".join(block))
    return "\n".join(lines)


def render_source_index(sources: Sequence[tuple[str, str]]) -> str:
    """Render the authoritative SOURCE-URL index fed to the synthesizer (c12 #5).

    A compact, prominent list of the REAL fetched URLs with a strict anti-fabrication
    instruction: cite ONLY from this list, use each URL verbatim, never invent a
    citation / publication name / date / ``[Name, 2025]`` placeholder, and close with a
    SOURCES section listing them. Returns ``""`` when there are no fetched sources, so
    a no-source task (headlines, a haiku) is never given an empty stub."""
    if not sources:
        return ""
    lines = [
        "REAL SOURCE URLS — these are the COMPLETE, exact URLs already fetched and "
        "read for this report. They are the ONLY real sources you have. When you cite "
        "a fact, use the matching URL from THIS list VERBATIM, and close the document "
        "with a SOURCES section listing the URLs you used. NEVER invent a citation, a "
        "publication name, a date, or a \"[Name, 2025]\"-style placeholder, and never "
        "cite a URL that is not in this list:",
    ]
    for i, (title, url) in enumerate(sources, 1):
        lines.append(f"[{i}] {title or url} — {url}")
    return "\n".join(lines)


def _strip_fence(text: str) -> str:
    """Strip ONE leading/trailing ``` code fence the small model wraps a chunk in.

    Mirrors :func:`chat_app.agentic._strip_fence` so a section the model fenced
    (e.g. ```` ```html … ``` ````) does not leave a stray fence inside the file."""
    s = (text or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


# --------------------------------------------------------------------------- #
# TYPE-AGNOSTIC output-path derivation (c3r path-carry fix) — the CHOSEN
# filename + extension that must SURVIVE to disk, never a content-derived ``.md``.
# --------------------------------------------------------------------------- #
# An explicit filename the request names (e.g. "write cats.html") wins; else the
# extension comes from the bound output-format WRITER SPEC (html-writer -> .html);
# else from a format keyword in the request; else the conventional ``.md``. The
# stem is a relatable slug from the goal/task — never a generic ``report``/``output``.

# RP-1 (d319/d311): the spec-name->ext (``_WRITER_SPEC_EXT``) and format-keyword->ext
# (``_FORMAT_KEYWORD_EXT``) INFERENCE conditionals are RETIRED — the engine no longer
# guesses a deliverable FORMAT from the bound writer spec or a request keyword (and there is
# NO invented ``.html`` default — d318). The MODEL picks its own filename/extension/format:
# an explicit filename the request names survives verbatim, and a bare fallback stem gets the
# neutral plain-text ``.md`` extension. ``file_write`` is a pure passthrough.
_DEFAULT_OUTPUT_EXT = ".md"
# An explicit filename with one of the deliverable extensions, anywhere in the text.
_EXPLICIT_NAME_RE = re.compile(
    r"\b([A-Za-z0-9][\w\-]*\.(?:html?|md|markdown|csv|txt|json|xml|ya?ml))\b",
    re.IGNORECASE,
)


def _slug(text: str) -> str:
    """A relatable filename stem (kebab-case) from the request's first line."""
    first = ((text or "").strip().splitlines() or [""])[0]
    slug = re.sub(r"[^a-z0-9]+", "-", first.lower()).strip("-")
    return slug[:50].strip("-")


def derive_output_path(
    goal: Optional[str],
    task: Optional[str],
    specs: Optional[Sequence[str]] = None,
) -> str:
    """The deliverable's filename+extension, derived TYPE-AGNOSTICALLY (d49 / c3r).

    RP-1 (d319/d311): the engine no longer INFERS a format. Precedence: (1) an explicit
    filename the request names (``cats.html`` survives verbatim — the model's own choice);
    (2) a relatable slug stem from the goal/task with the neutral plain-text
    ``.md`` extension. ``specs`` is accepted for signature stability but no longer maps a
    writer spec to an extension (that spec-name->ext conditional was a retired format pin)."""
    text = f"{goal or ''}\n{task or ''}"
    m = _EXPLICIT_NAME_RE.search(text)
    if m:
        return m.group(1).replace("\\", "/").rsplit("/", 1)[-1]
    stem = _slug(goal or "") or _slug(task or "") or "report"
    return f"{stem}{_DEFAULT_OUTPUT_EXT}"


def explicit_filename(text: Optional[str]) -> Optional[str]:
    """The explicit deliverable filename named in ``text`` (basename), else None.

    Shared with the acyclic ``file_write``-node path (toolargs) so an explicitly
    named ``cats.html`` survives verbatim there too — one source of truth for the
    c3r path-carry fix."""
    m = _EXPLICIT_NAME_RE.search(text or "")
    return m.group(1).replace("\\", "/").rsplit("/", 1)[-1] if m else None


def deliverable_extension(specs: Optional[Sequence[str]], text: str) -> str:
    """RP-1 (d319/d311): format INFERENCE is RETIRED — returns the neutral plain-text default.
    ``specs``/``text`` are accepted for signature stability but no longer map a writer spec or a
    request keyword to a format (that was a retired format pin). The model picks its own
    filename/extension via explicit naming (``explicit_filename`` handles that)."""
    return _DEFAULT_OUTPUT_EXT


def sanitize_write_path(raw: Optional[str], default_path: str) -> str:
    """Keep the model's chosen filename when usable, else fall back to ``default_path``.

    Takes the basename (the file tool sandboxes the directory anyway). A name that
    already carries a real extension the model chose is kept verbatim (so a
    model-picked ``.html``/``.csv`` survives); a name with NO usable extension
    borrows ``default_path``'s extension; an empty name uses ``default_path``."""
    name = (raw or "").strip().replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name:
        return default_path
    if "." in name and not name.endswith("."):
        ext = name.rsplit(".", 1)[-1]
        if 1 <= len(ext) <= 6 and ext.isalnum():
            return name
    default_ext = "." + default_path.rsplit(".", 1)[-1] if "." in default_path else ".md"
    return f"{name}{default_ext}"


# ---------------------------------------------------------------------------- #
# FINAL-DOCUMENT NO-FAB URL GUARD (d84/d89 → MS3 R2 part 2).
# ---------------------------------------------------------------------------- #
# A DETERMINISTIC post-pass over the FINAL assembled deliverable: every URL the
# model emitted is checked against the run's set of ACTUALLY-FETCHED source URLs —
# the normalized (scheme/host/path, fragment + trailing slash dropped) ``url`` of
# each source the run genuinely fetched. A cited URL is grounded IFF its normalized
# form is IN that set. This closes the d92 404 leak: a URL that appears ONLY as an
# inline hyperlink INSIDE a fetched article (a secondary link the run never fetched
# and which may 404) is NO LONGER grounded just by being a substring of the article
# body — it must itself be a fetched source. Any URL NOT in the fetched-URL set is
# treated as ungrounded/fabricated and REMOVED —
# an ``<a href>`` is unwrapped to its visible anchor text (the prose stays, the
# fabricated link goes), and a bare ungrounded URL is dropped. This makes the d60
# no-fabrication guarantee DETERMINISTIC rather than model-luck: even if the
# reasoning verify lane misses a hallucinated link, no fabricated URL can survive to
# the delivered file, regardless of model. Structural + content-preserving
# (d48/d50.1): it strips ONLY the offending URL token, never invents or rewrites
# prose. A doc whose URLs are all grounded is returned UNCHANGED (idempotent), so a
# clean report stays byte-identical.
# RP-1 (d319/d311): _DOC_URL_RE RETIRED — engine no longer fixes/authors the model's output.
# Trailing punctuation that commonly clings to a URL in prose/markup but is not part
# of the address — stripped before the grounded-set membership test and removal.
_URL_TRAILING = ".,;:!?)]}\"'»>"


def _normalize_fetched_url(u: str) -> str:
    """Normalize a URL to its scheme/host/path identity for set-membership grounding.

    Lower-cases the whole address, drops the fragment AND query, and removes a
    trailing slash, so a cited URL matches the fetched source it came from regardless
    of ``#anchor``/``?tracking``/trailing-``/`` variance. Trailing prose punctuation is
    stripped first. Returns ``""`` for an empty/garbage URL. The SAME normalizer is
    applied to both the fetched-source ``url``s (building the grounded set) and every
    doc URL (the membership test), so a legitimately fetched-and-cited link always
    matches itself while an inline-only secondary link (different host/path) does not."""
    raw = (u or "").strip().rstrip(_URL_TRAILING)
    if not raw:
        return ""
    try:
        p = urlsplit(raw)
    except ValueError:
        return raw.rstrip("/").lower()
    if not p.scheme or not p.netloc:
        # Not a parseable absolute URL — fall back to the lenient normalisation so a
        # malformed token still dedups/compares consistently rather than vanishing.
        return raw.rstrip("/").lower()
    path = p.path.rstrip("/")
    return urlunsplit((p.scheme, p.netloc, path, "", "")).lower()


def fetched_url_set(sources: Sequence[Mapping[str, str]]) -> frozenset[str]:
    """The set of NORMALIZED URLs the run ACTUALLY FETCHED — each source's own ``url``,
    and ONLY that ``url`` (NOT URLs embedded inside the article text/summary/title).

    This is the grounded universe for the no-fab URL guard: a doc URL is grounded IFF
    its normalized form is in this set. Replaces the prior lenient substring corpus
    (url + markdown + summary + title concatenated) which falsely grounded any URL that
    merely appeared as an inline hyperlink inside a fetched article (the d92 404 leak)."""
    urls: set[str] = set()
    for s in sources or ():
        if not isinstance(s, Mapping):
            continue
        n = _normalize_fetched_url(str(s.get("url") or ""))
        if n:
            urls.add(n)
    return frozenset(urls)


# RP-1 (d319/d311): strip_ungrounded_urls RETIRED — engine no longer fixes/authors the model's output.


# --------------------------------------------------------------------------- #
# DETERMINISTIC SOURCE-COVERAGE NET (s13/B5c, design §4C) — the `_ensure_source_coverage`
# d89 specified but never shipped (a1 Fact 4d).
# --------------------------------------------------------------------------- #
# a1 Fact 4d: there is NO `_ensure_source_coverage` function, so a fetched+cited source
# whose 1-based id the PHASE-2 write planner assigned to NO section is silently DROPPED
# from the report (the d87 "dropped source" / Al-Jazeera-disappearance risk). d91/d92's
# "100% source coverage" was an OBSERVED outcome, not a code guarantee — §6 delegated
# coverage to the in-loop reviewer's reasoning, but the reviewer NOTING an unused source
# does not make the planner assign it. This is the deterministic net under the loop: a
# FINAL pass that appends an "Additional sources" reference block for every fetched
# source that (a) the write plan assigned to no section AND (b) is not already present
# (cited/listed) anywhere in the assembled doc. Like the other backstops it is d60-safe —
# it adds ONLY a title+URL reference for material the run ACTUALLY fetched; it never
# generates content or invents a source. The in-loop agent stays the PRIMARY coverage
# mechanism; this is the net for the source it skipped. Idempotent: a doc whose every
# fetched source is assigned or already present is returned UNCHANGED.
_ADDITIONAL_SOURCES_HEADING = "Additional sources"
_ADDITIONAL_SOURCES_CAPTION = (
    "Sources fetched for this report but not cited in a section above:"
)


def _esc_html_text(text: str) -> str:
    """Minimal HTML text escaping for a reference label (no content invented)."""
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _esc_html_attr(text: str) -> str:
    """Minimal HTML attribute escaping for a reference href."""
    return (text or "").replace("&", "&amp;").replace('"', "&quot;")


# RP-1 (d319/d311): _insert_before_close RETIRED — engine no longer fixes/authors the model's output.


# RP-1 (d319/d311): ensure_source_coverage RETIRED — engine no longer fixes/authors the model's output.


__all__ = [
    "DONE_SENTINEL",
    "split_done_signal",
    "html_close_gap",
    "UNSUPPORTED_SECTION_INSTRUCTION",
    "section_reemission",
    "strip_internal_scaffolding",
    "fetched_url_set",
    "collect_fetched_sources",
    "collect_fetched_sources_full",
    "render_source_catalog",
    "render_scoped_sources",
    "render_source_index",
    "derive_output_path",
    "explicit_filename",
    "deliverable_extension",
    "sanitize_write_path",
    "unwrap_output_envelope",
    "_strip_fence",
]
