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
from typing import Optional, Sequence
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


def select_relevant_excerpt(markdown: str, section_topic: str, budget: int) -> str:
    """Up to ``budget`` chars of ``markdown`` most RELEVANT to ``section_topic`` (MSF/d89-a).

    Replaces the d87 flat ``markdown[:budget]`` slice (which only ever showed a long
    article's lede) with a SECTION-RELEVANT selection: rank the article's
    paragraph-aware chunks (``chunked_read.split_chunks``) by cheap LEXICAL overlap with
    the section heading/task and concatenate the top-scoring chunks — restored to
    original DOCUMENT order — up to ``budget``. Excerpts stay VERBATIM raw slices (c13
    verbatim-citation + d50.1 RAW). No extra model call. Falls back to ``markdown[:budget]``
    when there is no topic / no lexical signal / no chunks, and returns a short source
    whole."""
    md = (markdown or "").strip()
    budget = max(120, int(budget))
    if len(md) <= budget:
        return md
    topic = (section_topic or "").strip()
    if not topic:
        return md[:budget]
    terms = {
        w.lower()
        for w in _WORD_RE.findall(topic)
        if len(w) > 1 and w.lower() not in _STOPWORDS
    }
    if not terms:
        return md[:budget]
    # Granularity ≤ budget so a small budget still selects RELEVANT chunks instead of
    # truncating one oversized chunk down to its (possibly off-topic) lede.
    chunk_size = max(200, min(_RELEVANCE_CHUNK_CHARS, budget))
    chunks = split_chunks(md, chunk_size)
    if not chunks:
        return md[:budget]
    scored = [
        (sum(ch.lower().count(t) for t in terms), idx, ch)
        for idx, ch in enumerate(chunks)
    ]
    if not any(score for score, _, _ in scored):
        return md[:budget]  # nothing matched the topic — keep the lede deterministically
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
# HTML R1 gate: when the task asked for a DETAILED/thorough deliverable and the
# model finished in a single turn, send ONE continuation nudge before accepting.
# This signal decides WHETHER to nag — it never injects content (d48-clean). A
# task with no detailed-intent keyword (e.g. "give me the headlines") returns
# False, so a legitimately short deliverable is accepted in one turn (d46:
# headlines->headlines, not paragraphs).
_DETAILED_INTENT_RE = re.compile(
    r"\b(detailed|thorough|in[\s-]?depth|comprehensive|exhaustive|elaborate|"
    r"extensive|deep[\s-]?dive|full(?:[\s-]+(?:report|write[\s-]?up|analysis))|"
    r"long(?:er)?[\s-]+(?:report|write[\s-]?up|analysis))\b",
    re.IGNORECASE,
)


def is_detailed_task(text: str) -> bool:
    """True when the task text asks for a DETAILED / thorough / in-depth deliverable.

    A reasoning SIGNAL, not a template: it only gates whether a single-turn
    ``<<DONE>>`` on the markdown/text path is nudged to continue — it never
    injects or templates content. Returns ``False`` for a task with no
    detailed-intent keyword so a legitimately short deliverable (headlines, a
    one-line answer) is accepted in one turn (d46)."""
    return bool(_DETAILED_INTENT_RE.search(text or ""))


def strip_wrapper_closers(content: str) -> str:
    """Remove document-wrapper closing tags from a NON-FINAL chain page's content (c1b).

    A multi-page HTML deliverable opens the ``<html>``/``<body>`` wrapper on the first
    page and closes it EXACTLY ONCE on the terminal page (the deferred-close contract).
    But the small model habitually writes ``</body></html>`` into EVERY page it
    finishes — so a non-final page leaves an interior ``</body></html>`` mid-document,
    yielding the structurally invalid duplicate-wrapper file the c1br review flagged
    (``</html><section …>`` appearing mid-file). Since a non-final page must NEVER
    close the wrapper, strip every ``</body>``/``</html>`` it emitted (the same
    ``_HTML_CONTAINER_TAGS`` the close-gap gate balances); the terminal page (plus the
    :func:`html_close_gap` gate) supplies the single trailing pair. Only the document
    wrapper closers are removed — ``</head>``, ``</section>`` and all body content are
    untouched — and only on non-final HTML pages, so the single-file and markdown paths
    are byte-identical. Trailing whitespace left by the removal is tidied."""
    out = content or ""
    for tag in _HTML_CONTAINER_TAGS:
        out = re.sub(rf"</{tag}\s*>", "", out, flags=re.IGNORECASE)
    return out.rstrip()


# --------------------------------------------------------------------------- #
# SINGLE-DOCUMENT well-formedness (c10 #2) — the full-document analogue of the
# c1b wrapper-dedup (:func:`strip_wrapper_closers`).
# --------------------------------------------------------------------------- #
# A small model asked to "continue" a deliverable that is ALREADY a complete,
# closed HTML document re-emits a FRESH ``<!DOCTYPE>…</html>`` document, so the
# appended file ends up holding TWO complete documents concatenated (the browser
# renders only the first; the rest trails after the first ``</html>``). Tag-BALANCE
# passes this (2 opens + 2 closes look "balanced"), so single-document-ness must be
# asserted SEPARATELY — count the top-level ``</html>`` closes (one per complete
# document) and, when there is more than one, keep only the first.
_HTML_DOC_CLOSE_RE = re.compile(r"</html\s*>", re.IGNORECASE)


def top_level_html_doc_count(doc: str) -> int:
    """How many COMPLETE top-level HTML documents ``doc`` contains.

    Counts ``</html>`` closes — each complete document ends with exactly one. A bare
    HTML fragment (no wrapper) or a markdown/text/csv file returns 0, so only a
    genuine multi-document concatenation (``> 1``) is ever flagged (c10 #2)."""
    return len(_HTML_DOC_CLOSE_RE.findall(doc or ""))


def dedupe_html_documents(doc: str) -> str:
    """Reduce a multi-document HTML concatenation to its FIRST complete document.

    When ``doc`` holds more than one top-level ``</html>`` (the c10 #2
    duplicate-document defect — the model re-emitted a fresh full document as its
    "next section"), keep everything up to and INCLUDING the first ``</html>`` and
    drop the trailing duplicate document(s): the browser renders only the first, and
    the first is the substantive (sourced) one the run built. A single document — or
    a fragment / non-HTML file (``<= 1`` close) — is returned unchanged. Reads the
    real bytes, never the model's memory (d48-clean)."""
    s = doc or ""
    matches = list(_HTML_DOC_CLOSE_RE.finditer(s))
    if len(matches) <= 1:
        return doc
    return s[: matches[0].end()].rstrip()


def begins_html_document(content: str) -> bool:
    """True when ``content`` STARTS a fresh top-level HTML document.

    Recognises a re-emission (a chunk that opens with ``<!DOCTYPE>`` or a top-level
    ``<html>``) so the orchestration can STOP rather than append a SECOND document on
    top of an already-complete one (c10 #2)."""
    s = (content or "").lstrip().lower()
    return s.startswith("<!doctype") or s.startswith("<html")


# --------------------------------------------------------------------------- #
# STRICT single-document well-formedness (c12 #2b) — strip SIBLING structural
# OPENS, not only count ``</html>`` CLOSES.
# --------------------------------------------------------------------------- #
# On the LONG concurrent-multi-node path each node emits a full ``<!DOCTYPE>…`` doc,
# so the section-by-section append leaves stray inline ``<html>``/``<body>`` OPEN tags
# at section boundaries (and a doubled ``</body></html>`` tail) — a structure
# tag-BALANCE and the ``</html>``-close dedup both miss. ``strip_wrapper_openers``
# removes the document-wrapper OPEN tags from an APPENDED (non-first) section so it
# contributes body content only; ``enforce_single_html_document`` is the final
# normaliser that guarantees exactly one top-level ``<!DOCTYPE>``/``<html>``/
# ``<head>``/``<body>`` … ``</body></html>`` on the assembled bytes (the c1b
# :func:`strip_wrapper_closers` is its CLOSE-tag analogue).
_HTML_DOCTYPE_RE = re.compile(r"<!doctype[^>]*>", re.IGNORECASE)
_HTML_HTML_OPEN_RE = re.compile(r"<html(?:\s[^>]*)?>", re.IGNORECASE)
_HTML_BODY_OPEN_RE = re.compile(r"<body(?:\s[^>]*)?>", re.IGNORECASE)
_HTML_HEAD_BLOCK_RE = re.compile(r"<head(?:\s[^>]*)?>.*?</head\s*>", re.IGNORECASE | re.DOTALL)
_HTML_BODY_CLOSE_RE = re.compile(r"</body\s*>", re.IGNORECASE)


def strip_wrapper_openers(content: str) -> str:
    """Remove document-wrapper OPEN tags from a NON-FIRST appended section (c12 #2b).

    A multi-section HTML deliverable opens the ``<!DOCTYPE>``/``<html>``/``<body>``
    wrapper on the FIRST write; every later section must contribute body content only.
    But the small model habitually re-emits a fresh ``<!DOCTYPE><html>…<body>`` at the
    start of each section it writes, so an APPEND leaves a stray inline ``<html>``/
    ``<body>`` open mid-document (the c12 #2b residual). Strip the ``<!DOCTYPE>``, the
    top-level ``<html>``/``<body>`` opens AND any re-emitted ``<head>…</head>`` block
    from the appended chunk — leaving its real section content. Only the document
    wrapper is removed; ``<section>``/``<h2>`` and all body content are untouched. Pair
    with :func:`strip_wrapper_closers` (the CLOSE-tag analogue) on each append so the
    single close-gap gate supplies the one trailing ``</body></html>``."""
    out = content or ""
    out = _HTML_DOCTYPE_RE.sub("", out)
    out = _HTML_HEAD_BLOCK_RE.sub("", out)
    out = _HTML_HTML_OPEN_RE.sub("", out)
    out = _HTML_BODY_OPEN_RE.sub("", out)
    return out.strip()


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


def has_duplicate_html_structure(doc: str) -> bool:
    """True when ``doc`` carries duplicate top-level HTML structure (c12 #2b).

    Flags more than one ``<!DOCTYPE>``, top-level ``<html>`` open, ``<body>`` open, or
    ``</html>`` close — the stray-sibling-tag malformation the long multi-node append
    leaves. A single well-formed document (or a fragment / non-HTML file) returns
    False, so :func:`enforce_single_html_document` only fires on a genuine defect."""
    return (
        len(_HTML_DOCTYPE_RE.findall(doc or "")) > 1
        or len(_HTML_HTML_OPEN_RE.findall(doc or "")) > 1
        or len(_HTML_BODY_OPEN_RE.findall(doc or "")) > 1
        or len(_HTML_DOC_CLOSE_RE.findall(doc or "")) > 1
    )


def enforce_single_html_document(doc: str) -> str:
    """Normalise ``doc`` to STRICTLY ONE top-level HTML document (c12 #2b).

    Two distinct malformations are reduced WITHOUT ever truncating real content:

    * **Duplicate DOCUMENT** — more than one document OPENER (``<!DOCTYPE>``/top-level
      ``<html>``), i.e. the model crammed two complete ``<!DOCTYPE>…</html>`` documents
      into one emission (or two writers targeted one file). The later copies are a
      re-emission of the same report, so keep the FIRST complete document and drop the
      rest (the :func:`dedupe_html_documents` semantics the c10 #2 gate established),
      then tidy any stray sibling opens/closes inside it.
    * **Stray sibling CLOSES** — exactly one document opener but several
      ``</body>``/``</html>`` closes (the doubled-tail from section appends that each
      carried ``</body></html>`` after the per-append opener-strip removed their opens).
      Here the body content between the closes is REAL and distinct, so removing the
      closes and re-appending exactly one ``</body></html>`` preserves every section.

    The result is exactly one ``<!DOCTYPE>``/``<html>``/``<head>``/``<body>`` …
    ``</body></html>``. Reads the real bytes, never the model's memory (d48-clean). A
    single well-formed document or a fragment / non-HTML file is returned unchanged."""
    s = doc or ""
    # Duplicate-DOCUMENT case: keep the FIRST complete document (drop re-emitted copies).
    if len(_HTML_HTML_OPEN_RE.findall(s)) > 1 or len(_HTML_DOCTYPE_RE.findall(s)) > 1:
        s = dedupe_html_documents(s)
        s = _keep_first(s, _HTML_DOCTYPE_RE)
        s = _keep_first(s, _HTML_HTML_OPEN_RE)
        s = _keep_first(s, _HTML_HEAD_BLOCK_RE)
        s = _keep_first(s, _HTML_BODY_OPEN_RE)
        s = _keep_last(s, _HTML_BODY_CLOSE_RE)
        s = _keep_last(s, _HTML_DOC_CLOSE_RE)
        return s.strip()
    # Stray-sibling-CLOSES case: one opener but several closes — the inter-close content
    # is real, so strip ALL wrapper closes and re-append exactly one (no truncation).
    if _HTML_HTML_OPEN_RE.search(s) or _HTML_BODY_OPEN_RE.search(s):
        s = _HTML_BODY_CLOSE_RE.sub("", s)
        s = _HTML_DOC_CLOSE_RE.sub("", s)
        s = s.rstrip() + "</body></html>"
    return s.strip()


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
_HTML_ID_ATTR_RE = re.compile(r'\bid\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)


def _heading_slug(text: str, used: set[str], idx: int) -> str:
    """A stable kebab-case anchor id from a heading's visible text (deduped)."""
    bare = re.sub(r"<[^>]+>", "", text or "")
    slug = re.sub(r"[^a-z0-9]+", "-", bare.lower()).strip("-")[:48].strip("-")
    slug = slug or f"section-{idx}"
    base, n = slug, 2
    while slug in used:
        slug = f"{base}-{n}"
        n += 1
    used.add(slug)
    return slug


def assemble_html_spa(doc: str, *, title: str = "Report") -> str:
    """Assemble a per-section HTML fragment into ONE navigable single-page document (c13).

    Idempotent + structural (d48-clean — reads real bytes, fabricates no content):

    * ensures every top-level ``<h1>/<h2>`` heading has an ``id`` anchor (slugged from
      its own visible text) so the nav can link to it;
    * builds a ``<nav>`` table-of-contents of those headings (the model's own section
      titles) and places it at the top of the body;
    * wraps the body in exactly one ``<!DOCTYPE>/<html>/<head>/<body>`` when the
      fragment has no document wrapper; if a wrapper is already present (the c1b
      path), only the nav is injected after ``<body>`` (no second wrapper).

    A non-HTML string, or a doc that already carries a ``<nav>``, is returned
    essentially unchanged (the nav is not duplicated). Never truncates real content."""
    s = doc or ""
    if not s.strip() or "<h1" not in s.lower() and "<h2" not in s.lower():
        return doc  # nothing to navigate (not an HTML report fragment)

    # 1) ensure an id on every h1/h2 (keep an existing id; else slug from its text).
    used: set[str] = set()
    for m in _HTML_ID_ATTR_RE.finditer(s):
        used.add(m.group(1))
    toc: list[tuple[str, str, str]] = []  # (level, id, visible-text)
    idx = 0

    def _stamp(m: "re.Match[str]") -> str:
        nonlocal idx
        idx += 1
        level, attrs, inner = m.group(1), (m.group(2) or ""), m.group(3)
        bare = re.sub(r"<[^>]+>", "", inner).strip()
        id_m = _HTML_ID_ATTR_RE.search(attrs)
        if id_m:
            hid = id_m.group(1)
        else:
            hid = _heading_slug(bare, used, idx)
            attrs = f'{attrs} id="{hid}"'
        toc.append((level.lower(), hid, bare))
        return f"<{level}{attrs}>{inner}</{level}>"

    body = _HTML_HEADING_RE.sub(_stamp, s)
    if not toc:
        return doc

    # 2) build the nav TOC from the model's own headings (skip if a nav already exists).
    if "<nav" not in body.lower():
        items = "\n".join(
            f'    <li class="toc-{lvl}"><a href="#{hid}">{txt}</a></li>'
            for lvl, hid, txt in toc
        )
        nav = f'<nav class="spa-nav">\n  <ul>\n{items}\n  </ul>\n</nav>\n'
    else:
        nav = ""

    # 3) wrap the fragment in one document, or inject the nav after an existing <body>.
    if "<html" in body.lower():
        if nav:
            body = re.sub(r"(<body(?:\s[^>]*)?>)", r"\1\n" + nav, body, count=1,
                          flags=re.IGNORECASE)
        return body
    head = (
        "<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{re.sub(r'<[^>]+>', '', title) or 'Report'}</title>\n"
        "<style>body{max-width:900px;margin:0 auto;padding:1.5rem;font-family:system-ui,"
        "Arial,sans-serif;line-height:1.5}nav.spa-nav{border:1px solid #ccc;padding:.75rem "
        "1rem;margin-bottom:1.5rem;background:#fafafa}nav.spa-nav ul{list-style:none;"
        "padding-left:0;margin:0}nav.spa-nav li.toc-h2{padding-left:1.25rem}"
        "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:.4rem}"
        "tr:nth-child(even){background:#f6f6f6}.sources{font-size:.9em;color:#555}</style>\n"
        "</head>"
    )
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n" + head + "\n<body>\n"
        + nav + body + "\n</body>\n</html>"
    )


# --------------------------------------------------------------------------- #
# DOC-STRUCTURE INTEGRITY BACKSTOP (s13/B5, design §4B) — a FINAL reconcile pass
# over the fully-assembled HTML, run ONCE as a safety net AFTER every other
# structural pass (assemble_html_spa / collapse_duplicate_sections / …).
# --------------------------------------------------------------------------- #
# The upstream passes are REGEX-scoped and each fires while the document is still
# being built: assemble_html_spa derives the nav from the h1/h2 headings PRESENT
# WHEN IT RUNS, so a section appended AFTER it (the late-section path, d93) is in
# the body but MISSING from the ToC; concurrent multi-node appends can also leave
# two elements sharing one ``id`` (a duplicate-anchor defect a regex heading-stamp
# never reconciles), or a wrapper left unbalanced by a truncated emission. This
# pass closes those gaps with a REAL stdlib parser (``html.parser.HTMLParser`` — no
# new dependency): it parses the ACTUAL assembled bytes to learn the true h1..h3
# set, every ``id`` in document order, and the wrapper/list balance, then applies
# only deterministic, content-preserving REPAIRS. Like its siblings it is d48/d60-
# clean — it reads the real bytes and fabricates NO content (it only RE-DERIVES
# navigation, RENAMES collided ids, and BALANCES the wrapper). It is idempotent:
# a well-formed document with a complete ToC and unique ids is returned unchanged.
_HTML_HEADING123_RE = re.compile(r"<(h[123])(\s[^>]*)?>(.*?)</\1\s*>", re.IGNORECASE | re.DOTALL)
_SPA_NAV_RE = re.compile(
    r"[ \t]*<nav\s+class=[\"']spa-nav[\"'][^>]*>.*?</nav\s*>\n?", re.IGNORECASE | re.DOTALL
)
_TAG_TOKEN_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)((?:\s[^>]*)?)(/?)>")
_LIST_CONTAINER_TAGS: frozenset[str] = frozenset({"ul", "ol"})


class _DocStructureParser(HTMLParser):
    """Read-only structural scan of assembled HTML via the stdlib parser (s13/B5).

    Records the true ``id`` multiset (to find collisions), whether a ``spa-nav``
    ToC is present, and whether any ``<li>`` sits OUTSIDE a ``<ul>``/``<ol>`` (an
    orphan list item). Used only to DECIDE which deterministic repairs to apply —
    the edits themselves are made on the original bytes, so the parser never
    re-serializes (and therefore never drops) real content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.id_counts: dict[str, int] = {}
        self.has_spa_nav = False
        self.orphan_li = False
        self._list_depth = 0

    def handle_starttag(self, tag: str, attrs: "list[tuple[str, Optional[str]]]") -> None:
        d = {k.lower(): (v or "") for k, v in attrs}
        hid = d.get("id", "").strip()
        if hid:
            self.id_counts[hid] = self.id_counts.get(hid, 0) + 1
        if tag == "nav" and "spa-nav" in d.get("class", ""):
            self.has_spa_nav = True
        if tag in _LIST_CONTAINER_TAGS:
            self._list_depth += 1
        elif tag == "li" and self._list_depth == 0:
            self.orphan_li = True

    def handle_startendtag(self, tag: str, attrs: "list[tuple[str, Optional[str]]]") -> None:
        # self-closing (e.g. <ul/>) opens no scope — only count its id.
        d = {k.lower(): (v or "") for k, v in attrs}
        hid = d.get("id", "").strip()
        if hid:
            self.id_counts[hid] = self.id_counts.get(hid, 0) + 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _LIST_CONTAINER_TAGS and self._list_depth > 0:
            self._list_depth -= 1


def _dedupe_element_ids(doc: str, duplicated: "frozenset[str]") -> str:
    """Rename the 2nd+ occurrence of each collided ``id`` value to ``<id>-2``, ``-3``….

    Keeps the FIRST occurrence's id intact (so existing inbound anchors to it still
    resolve) and makes every later collision unique. Operates only on the ``id="…"``
    attribute text, so no surrounding markup or content is touched."""
    if not duplicated:
        return doc
    counts: dict[str, int] = {}

    def _repl(m: "re.Match[str]") -> str:
        val = m.group(1)
        if val not in duplicated:
            return m.group(0)
        counts[val] = counts.get(val, 0) + 1
        if counts[val] == 1:
            return m.group(0)  # keep the first occurrence as-is
        new_val = f"{val}-{counts[val]}"
        return m.group(0).replace(val, new_val, 1)

    return _HTML_ID_ATTR_RE.sub(_repl, doc)


def _wrap_orphan_list_items(doc: str) -> str:
    """Wrap each run of top-level (orphan) ``<li>…</li>`` siblings in one ``<ul>``.

    A ``<li>`` outside any ``<ul>``/``<ol>`` is invalid; the conservative repair
    wraps a maximal run of CONSECUTIVE orphan items (only whitespace between them)
    in a single ``<ul>``. Items already inside a list are left untouched, so a
    well-formed list — and the rebuilt ``spa-nav`` ToC — is never re-wrapped."""
    spans: list[tuple[int, int]] = []          # (start, end) of each orphan <li>…</li>
    list_depth = 0
    open_li: list[Optional[int]] = []          # stack of orphan-li start offsets (None = inside a list)
    for m in _TAG_TOKEN_RE.finditer(doc):
        closing, name, self_close = m.group(1), m.group(2).lower(), m.group(4)
        if name in _LIST_CONTAINER_TAGS:
            if closing:
                list_depth = max(0, list_depth - 1)
            elif not self_close:
                list_depth += 1
        elif name == "li" and not self_close:
            if closing:
                if open_li:
                    start = open_li.pop()
                    if start is not None:
                        spans.append((start, m.end()))
            else:
                open_li.append(m.start() if list_depth == 0 else None)
    if not spans:
        return doc
    # group consecutive orphan spans separated only by whitespace into runs.
    runs: list[tuple[int, int]] = []
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if doc[cur_e:s].strip() == "":
            cur_e = e
        else:
            runs.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    runs.append((cur_s, cur_e))
    for s, e in reversed(runs):  # apply right-to-left so earlier offsets stay valid
        doc = doc[:s] + "<ul>\n" + doc[s:e] + "\n</ul>" + doc[e:]
    return doc


def reconcile_doc_structure(doc: str, *, title: str = "Report", single_title: bool = False) -> str:
    """FINAL doc-structure integrity backstop over assembled HTML (s13/B5, design §4B).

    Runs ONCE, last, as a safety net only — it generates NO content (d48/d60-clean:
    reads the real bytes, repairs structure deterministically). Three repairs, driven
    by a real ``html.parser`` scan of the ACTUAL assembled document:

    * **(i) ToC re-derivation** — when the document ALREADY carries a ``spa-nav`` table
      of contents (i.e. :func:`assemble_html_spa` ran), rebuilds it from the FULL set of
      ``<h1>``/``<h2>``/``<h3>`` headings present after every section is appended, so a
      late-appended section (d93) the build-time nav missed now appears. Every heading
      is given a stable slug ``id`` if it lacks one, and the nav links to those ids. A
      document that was never given a nav (a plain single-file report) is left with NO
      nav — the backstop COMPLETES an existing ToC, it does not fabricate one.
    * **(ii) duplicate-id rename** — any ``id`` value carried by more than one element
      (a collision the regex passes never reconcile) has its 2nd+ occurrences renamed
      ``<id>-2``, ``<id>-3``…; the first keeps the original so inbound anchors resolve.
    * **(iii) wrapper / list well-formedness** — collapses a stray duplicate wrapper
      (:func:`enforce_single_html_document`), wraps any orphan ``<li>`` in a ``<ul>``,
      and appends any missing ``</body>``/``</html>`` (:func:`html_close_gap`) so the
      document is balanced.

    A non-HTML string, or a document with no headings and no wrapper, is returned
    unchanged. Idempotent — a well-formed, complete-ToC, unique-id document is a
    fixed point."""
    s = doc or ""
    low = s.lower()
    if not s.strip() or (
        "<h1" not in low and "<h2" not in low and "<h3" not in low and "<html" not in low
    ):
        return doc  # not an HTML report — nothing to reconcile

    # 0) THEMATIC DUPLICATE-TAIL (s13/P1-report): demote a second document-shell title to
    # a section, then drop any now-duplicate heading family — BEFORE the ToC is re-derived
    # so the served nav reflects exactly ONE title and one pass of each section. Both are
    # drop-free/idempotent (enforce_single_h1 changes only a heading LEVEL; collapse keeps
    # the first grounded occurrence), so a clean single-title document is unchanged.
    # GATED to ``single_title`` (the sourced deep-research REPORT path, where the B8a2
    # duplicate-tail lives) so a legitimate MULTI-PAGE document — one ``<h1>`` per page on
    # the file-delivery/plan-chain path — keeps its per-page titles untouched.
    if single_title:
        s = enforce_single_h1(s)
        s = collapse_duplicate_sections(s)

    # 1) parse the real bytes to learn id collisions, orphan <li>, and nav presence.
    parser = _DocStructureParser()
    try:
        parser.feed(s)
        parser.close()
    except Exception:
        # A parser hiccup on pathological input must never sink the pipeline: fall
        # back to the regex-only repairs below (id collisions still seen via regex).
        parser = _DocStructureParser()
    duplicated = frozenset(v for v, c in parser.id_counts.items() if c > 1)
    if not duplicated:  # parser saw nothing (fallback) — detect collisions via regex.
        seen: dict[str, int] = {}
        for v in _HTML_ID_ATTR_RE.findall(s):
            seen[v] = seen.get(v, 0) + 1
        duplicated = frozenset(v for v, c in seen.items() if c > 1)

    # 2) make every id unique (keep the first of each collided value).
    s = _dedupe_element_ids(s, duplicated)

    # 3) wrap orphan list items before rebuilding the (well-formed) nav.
    if parser.orphan_li:
        s = _wrap_orphan_list_items(s)

    # 4) ToC re-derivation — ONLY when the document already carries a spa-nav (i.e.
    # assemble_html_spa ran). A plain single-file report that was never given a nav is
    # left untouched: the backstop COMPLETES an existing ToC, it never fabricates one
    # (so it does not alter the shape of a deliverable that intentionally had no nav).
    toc: list[tuple[str, str, str]] = []  # (level, id, visible-text)
    if parser.has_spa_nav:
        # stamp an id on every h1..h3 lacking one and collect the FULL heading list.
        used: set[str] = set(_HTML_ID_ATTR_RE.findall(s))
        idx = 0

        def _stamp(m: "re.Match[str]") -> str:
            nonlocal idx
            idx += 1
            level, attrs, inner = m.group(1), (m.group(2) or ""), m.group(3)
            bare = re.sub(r"<[^>]+>", "", inner).strip()
            id_m = _HTML_ID_ATTR_RE.search(attrs)
            if id_m:
                hid = id_m.group(1)
            else:
                hid = _heading_slug(bare, used, idx)
                attrs = f'{attrs} id="{hid}"'
            toc.append((level.lower(), hid, bare))
            return f"<{level}{attrs}>{inner}</{level}>"

        s = _HTML_HEADING123_RE.sub(_stamp, s)

    # 5) rebuild the ToC from the full heading set (drop a stale/partial spa-nav first).
    if toc:
        s = _SPA_NAV_RE.sub("", s)
        items = "\n".join(
            f'    <li class="toc-{lvl}"><a href="#{hid}">{txt}</a></li>'
            for lvl, hid, txt in toc
        )
        nav = f'<nav class="spa-nav">\n  <ul>\n{items}\n  </ul>\n</nav>\n'
        if re.search(r"<body(?:\s[^>]*)?>", s, re.IGNORECASE):
            # Consume any whitespace immediately after <body> so re-running the pass
            # (which strips then re-inserts the nav) is a fixed point, not a slowly
            # growing run of blank lines.
            s = re.sub(
                r"(<body(?:\s[^>]*)?>)\s*",
                lambda m: m.group(1) + "\n" + nav,
                s,
                count=1,
                flags=re.IGNORECASE,
            )
        else:
            s = nav + s

    # 6) wrapper well-formedness: one document, then balance the closing tags.
    if has_duplicate_html_structure(s):
        s = enforce_single_html_document(s)
    gaps = html_close_gap(s)
    if gaps:
        s = s.rstrip() + "\n" + "".join(gaps)
    # 7) FINAL-SECTION TRUNCATION (s13/P1-report): trim a short mid-sentence cut-off tail
    # (a num_predict/section truncation the tag-balance pass leaves intact) so the served
    # document never ends on a dangling fragment ("…On February</body></html>").
    s = trim_dangling_sentence(s)
    return s


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
    """The inner text of every top-level ``<h1>``/``<h2>`` heading, in document order."""
    return [m.group(3) for m in _HTML_HEADING_RE.finditer(doc or "")]


def collapse_duplicate_sections(doc: str) -> str:
    """Collapse re-emitted body-level report passes to ONE pass of each section (c14).

    Deterministic + structural (d48/d60-clean — reads the real bytes, fabricates NO
    content): segments ``doc`` at its ``<h1>``/``<h2>`` headings, keeps the FIRST
    occurrence of each heading FAMILY, and DROPS every later section whose family was
    already kept (a re-emitted duplicate pass). Each heading's segment is extended LEFT
    over an immediately-preceding wrapper-open tag so the section's own ``<div …>``
    opener and its matching close are removed together (container nesting stays
    balanced); any top-level ``</body>``/``</html>`` a dropped TAIL section carried is
    re-appended via :func:`html_close_gap` so the wrapper never ends up unbalanced.

    Keeping the FIRST occurrence preserves the grounded, source-scoped content (the c13
    citation win) and never invents text. A document with no repeated heading family —
    or fewer than two headings — is returned UNCHANGED (idempotent), so a clean
    single-pass report and a non-HTML/fragment string are untouched."""
    s = doc or ""
    heads = list(_HTML_HEADING_RE.finditer(s))
    if len(heads) < 2:
        return doc
    kept_fams: list[frozenset[str]] = []
    drop = [False] * len(heads)
    for i, m in enumerate(heads):
        fam = _heading_family(m.group(3))
        if any(_families_match(fam, kf) for kf in kept_fams):
            drop[i] = True
        else:
            kept_fams.append(fam)
    if not any(drop):
        return doc
    starts: list[int] = []
    for m in heads:
        start = m.start()
        wm = _WRAPPER_OPEN_TAIL_RE.search(s, 0, start)
        if wm:
            start = wm.start()
        starts.append(start)
    ends = starts[1:] + [len(s)]
    out = [s[: starts[0]]]
    for i in range(len(heads)):
        if not drop[i]:
            out.append(s[starts[i] : ends[i]])
    result = "".join(out)
    gaps = html_close_gap(result)
    if gaps:
        result = result.rstrip() + "\n" + "".join(gaps)
    return result


# --------------------------------------------------------------------------- #
# THEMATIC DUPLICATE-TAIL kill (s13/P1-report, d115 parity) — single document title.
# --------------------------------------------------------------------------- #
# The B8a2 residual: the first write node, tasked "shell + first section", over-produces
# a WHOLE mini-document — its own title <h1> + figures + a Sources block — then a later
# section node opens a SECOND <h1> (a second document shell, e.g. "The US-Iran Conflict…"
# then "Iran War (2026) Conflict Analysis"), so the served doc carries TWO top-level
# titles and repeats cost/casualty material. The wrapper gate (enforce_single_html_
# document) misses it — there is only ONE <!DOCTYPE>/<html> — and the family collapse
# misses it — the two titles differ in wording past the family-overlap threshold. The
# structured-data -> HTML separation removes the CAUSE (the shared figures/sources live
# once, not re-emitted per section); this is the deterministic structural backstop for
# the residual: a report has exactly ONE document title, so any later <h1> is a section.
_HTML_H1_RE = re.compile(r"<h1((?:\s[^>]*)?)>(.*?)</h1\s*>", re.IGNORECASE | re.DOTALL)


def enforce_single_h1(doc: str) -> str:
    """Keep the FIRST ``<h1>`` as the document title; demote every later ``<h1>`` to ``<h2>``.

    Drop-FREE (d48/d60-clean: no content is removed — only a heading LEVEL changes) and
    idempotent (a document with at most one ``<h1>`` is returned byte-identical). A later
    ``<h1>`` is a second document-shell title (the B8a2 thematic duplicate-tail); demoting
    it to ``<h2>`` collapses the two-title defect AND lets the existing family collapse /
    ToC re-derivation treat the second shell's body as ordinary sections. Attributes and
    inner markup of the demoted heading are preserved verbatim."""
    count = 0

    def _repl(m: "re.Match[str]") -> str:
        nonlocal count
        count += 1
        if count == 1:
            return m.group(0)  # the document title stays an <h1>
        return f"<h2{m.group(1)}>{m.group(2)}</h2>"

    return _HTML_H1_RE.sub(_repl, doc or "")


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


def trim_dangling_sentence(doc: str, *, max_trim: int = 400) -> str:
    """Remove a SHORT mid-sentence truncated tail so the served doc ends cleanly.

    Offline the cut-off content cannot be regenerated (that would fabricate, d60), but a
    dangling fragment is not a groundable claim. When the visible body ends mid-sentence
    (:func:`has_truncation_marker`), trim back to the last sentence-terminating
    punctuation and re-append the wrapper closes (:func:`html_close_gap`), so no
    truncation marker reaches the served document. CONSERVATIVE: only a SHORT trailing
    fragment (``<= max_trim`` chars, carrying no block-level heading) is trimmed — a large
    dangling block is LEFT untouched (the verify / continuation lanes own it) so real
    content is never silently dropped. No-op + idempotent on a clean document."""
    if not has_truncation_marker(doc):
        return doc
    body, closes = _visible_body_split(doc)
    cut = max((body.rfind(c) for c in _SENTENCE_END_CHARS), default=-1)
    if cut < 0:
        return doc  # no complete sentence to fall back to — leave it (flagged elsewhere)
    fragment = body[cut + 1:]
    if len(fragment) > max_trim or "<h" in fragment.lower():
        return doc  # not a short clean cut-off — do not drop a whole section
    trimmed = body[: cut + 1].rstrip()
    if not trimmed:
        return doc
    result = trimmed + (("\n" + closes) if closes else "")
    gaps = html_close_gap(result)
    if gaps:
        result = result.rstrip() + "\n" + "".join(gaps)
    return result


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
def _outline_families(
    outline: Optional[Sequence[Mapping[str, str]]],
) -> list[frozenset[str]]:
    """The heading FAMILY of each outline entry's title, in outline order (drift-tolerant,
    reusing :func:`_heading_family`). Blank/untitled entries are skipped."""
    fams: list[frozenset[str]] = []
    for sec in outline or ():
        if not isinstance(sec, Mapping):
            continue
        title = str(sec.get("title", "")).strip()
        if not title:
            continue
        fams.append(_heading_family(title))
    return fams


def collapse_outline_duplicate_sections(
    doc: str, outline: Optional[Sequence[Mapping[str, str]]]
) -> str:
    """Collapse a duplicate PARALLEL section set against the agent outline (s13/FX d106 #7).

    Deterministic + structural (d48/d60-clean — reads the real bytes, fabricates NO
    content): segments ``doc`` at its ``<h1>``/``<h2>`` headings; each heading is mapped
    to the FIRST outline entry whose family it matches (:func:`_families_match`). When two
    headings map to the SAME outline slot, the LATER one is a duplicate of that planned
    section (the B8a appended-tail / "three Section 3s" defect) and its segment is DROPPED;
    the first (grounded, source-scoped) occurrence is kept. A heading that matches no
    outline slot is KEPT (conservative — never drops a section the outline does not name).

    No-ops (returns the doc UNCHANGED) when the outline is empty, fewer than two headings
    are present, or no two headings share an outline slot — so a clean report, a non-HTML
    fragment, and a doc whose sections already follow the outline are all untouched
    (idempotent). Segment boundaries + the ``</body>``/``</html>`` re-append mirror
    :func:`collapse_duplicate_sections` so the wrapper never ends up unbalanced."""
    s = doc or ""
    outline_fams = _outline_families(outline)
    if not outline_fams:
        return doc
    heads = list(_HTML_HEADING_RE.finditer(s))
    if len(heads) < 2:
        return doc
    claimed: set[int] = set()
    drop = [False] * len(heads)
    for i, m in enumerate(heads):
        fam = _heading_family(m.group(3))
        slot = next(
            (j for j, of in enumerate(outline_fams) if _families_match(fam, of)), None
        )
        if slot is None:
            continue  # matches no outline section → keep (genuinely new)
        if slot in claimed:
            drop[i] = True  # a later heading for an already-written outline section
        else:
            claimed.add(slot)
    if not any(drop):
        return doc
    starts: list[int] = []
    for m in heads:
        start = m.start()
        wm = _WRAPPER_OPEN_TAIL_RE.search(s, 0, start)
        if wm:
            start = wm.start()
        starts.append(start)
    ends = starts[1:] + [len(s)]
    out = [s[: starts[0]]]
    for i in range(len(heads)):
        if not drop[i]:
            out.append(s[starts[i] : ends[i]])
    result = "".join(out)
    gaps = html_close_gap(result)
    if gaps:
        result = result.rstrip() + "\n" + "".join(gaps)
    return result


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
# ANCHORED SECTION INSERT (s13/P2.3, d130/d132.C) — the ROOT-CAUSE structural
# prevention of the duplicate-TAIL. The per-section write loop used to build the
# document by BLIND ``file_write(append=True)``: the small model re-emits an
# already-written chunk (a 2nd ``<!DOCTYPE>`` / a repeated section) and the blind
# append CONCATENATES it AFTER the closed document — the duplicate-tail / 2nd
# top-level document defect. The fix replaces blind append with an ANCHORED
# read->targeted-insert->write: the document always carries ONE unique terminal
# ANCHOR, and each new section is inserted JUST BEFORE that anchor via
# ``file_update(old=anchor, new=section+anchor)``. Because every section lands
# IN PLACE before the single anchor (never blind-appended after the document), no
# content can ever be concatenated past the document's end — the duplicate-tail
# cannot form structurally. ``file_update``'s own contract is what makes it safe:
# the ``old`` anchor must be present (a missing/ambiguous anchor REFUSES the write
# rather than silently appending), and a match REPLACES the span in place (it
# grows the doc by exactly the new section, not a blind concatenation).
#
# The anchor is a deterministic, UNIQUE, render-invisible HTML comment planted at
# the end of the document on creation (works for HTML, markdown AND plain text —
# an HTML comment renders to nothing in all three). For an HTML document that
# already carries its single ``</body>`` close, that close is an equally valid
# unique anchor, so :func:`choose_section_anchor` prefers it (a section then lands
# INSIDE the body). The planted sentinel is STRIPPED at finalize so it never
# reaches the served file.
SECTION_ANCHOR = "<!--__RA_SECTION_ANCHOR__-->"


def plant_section_anchor(content: str) -> str:
    """Append the unique terminal section ANCHOR to freshly-created section content.

    Used on the FIRST write of a node (create, or the first append of a
    continuation page): the document is left ending with the sentinel so every
    later section can be inserted just before it. The sentinel is a render-invisible
    HTML comment (no effect on HTML/markdown/plain-text output) and is removed by
    :func:`strip_section_anchor` at finalize."""
    return content + "\n" + SECTION_ANCHOR


def choose_section_anchor(file_text: str, is_html: bool) -> Optional[str]:
    """Pick the UNIQUE in-document insertion anchor, or None when none is unique.

    Prefers the planted :data:`SECTION_ANCHOR` sentinel when it is present exactly
    once (the normal path — it is planted at the document end on the first write).
    Falls back to the HTML ``</body>`` close when THAT is the only unique anchor
    (an already-closed HTML document with no sentinel — a section then inserts
    inside the body). Returns None when neither is uniquely present, so the caller
    degrades to a guarded append + replant rather than risk a wrong-span edit."""
    if file_text.count(SECTION_ANCHOR) == 1:
        return SECTION_ANCHOR
    if is_html and file_text.count("</body>") == 1:
        return "</body>"
    return None


def anchored_insert_args(anchor: str, section: str) -> tuple[str, str]:
    """``(old, new)`` for ``file_update`` to insert ``section`` JUST BEFORE ``anchor``.

    Replacing the unique ``anchor`` with ``section + anchor`` inserts the section in
    document ORDER and leaves the anchor unique and terminal for the next insert —
    the growable, no-blind-append invariant. A leading newline separates consecutive
    sections."""
    return anchor, section + "\n" + anchor


def strip_section_anchor(text: str) -> str:
    """Remove the planted section ANCHOR (and its leading newline) at finalize.

    Idempotent and content-preserving: a document that never carried the sentinel
    (e.g. a one-shot raw fallback) is returned byte-identical. Run once the write
    loop has finished, before the document is assembled/served, so the sentinel
    never reaches disk."""
    return text.replace("\n" + SECTION_ANCHOR, "").replace(SECTION_ANCHOR, "")


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
        for art in tv.get("fetched") or ():
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
        "read. Write this section's facts/figures FROM the text below and cite each "
        "claim with the matching URL VERBATIM. NEVER invent a figure, a citation, a "
        "publication name, a date, a \"[Name, 2025]\" placeholder, or a \"URL 1\"-style "
        "label, and never cite a URL not listed here:",
    ]
    for i, s in picked:
        title = str(s.get("title") or "").strip()
        url = str(s.get("url") or "").strip()
        full = str(s.get("markdown") or "").strip()
        summary = str(s.get("summary") or "").strip()
        excerpt = select_relevant_excerpt(full, section_topic, budget)
        block = [f"\n[{i}] {title or url} — {url}"]
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

# A model-bound output-format writer spec -> its file extension (the planner binds
# html-writer / markdown-writer on the terminal node; the extension must follow).
_WRITER_SPEC_EXT: dict[str, str] = {
    "html-writer": ".html",
    "markdown-writer": ".md",
}
# A format named in the request -> extension (checked when no explicit name / spec).
_FORMAT_KEYWORD_EXT: tuple[tuple[str, str], ...] = (
    (r"\bhtml\b|\bweb\s?page\b|\bweb\s?site\b", ".html"),
    (r"\bmarkdown\b|\bmd\b", ".md"),
    (r"\bcsv\b|\bspreadsheet\b|\bcomma[- ]separated\b", ".csv"),
    # ``.txt`` matching (c10 #3): the old ``\b\.txt\b`` FAILED whenever a space
    # preceded ``.txt`` (a space and ``.`` are both non-word, so no word boundary
    # sits between them) — "save to a .txt file" / "give me a .txt file" fell through
    # to the ``.md`` default. Match a literal ``.txt`` (no leading ``\b``), a bare
    # ``txt`` token, and the plain-text phrasings, so a reasoned ``.txt`` is honored.
    (r"\.txt\b|\btxt\b|\bplain[- ]?text\b|\btext file\b", ".txt"),
    (r"\bjson\b", ".json"),
)
# An explicit filename with one of the deliverable extensions, anywhere in the text.
_EXPLICIT_NAME_RE = re.compile(
    r"\b([A-Za-z0-9][\w\-]*\.(?:html?|md|markdown|csv|txt|json|xml|ya?ml))\b",
    re.IGNORECASE,
)


def _ext_for(specs: Optional[Sequence[str]], text: str) -> str:
    """Pick the deliverable extension from the bound writer spec, else the text."""
    for s in specs or ():
        ext = _WRITER_SPEC_EXT.get(str(s).strip().lower())
        if ext:
            return ext
    for pattern, ext in _FORMAT_KEYWORD_EXT:
        if re.search(pattern, text, re.IGNORECASE):
            return ext
    return ".md"


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

    Precedence: (1) an explicit filename the request names (``cats.html`` survives
    verbatim); (2) the bound output-format writer spec's extension (html-writer ->
    ``.html``); (3) a format keyword in the request; (4) ``.md``. The stem is a
    relatable slug from the goal/task. This is the chosen name the synthesizer
    writes to disk, so the LLM-chosen extension reaches the artifact — never the old
    content-derived ``.md`` fallback that turned ``cats.html`` into ``findings.md``."""
    text = f"{goal or ''}\n{task or ''}"
    m = _EXPLICIT_NAME_RE.search(text)
    if m:
        return m.group(1).replace("\\", "/").rsplit("/", 1)[-1]
    ext = _ext_for(specs, text)
    stem = _slug(goal or "") or _slug(task or "") or "report"
    return f"{stem}{ext}"


def explicit_filename(text: Optional[str]) -> Optional[str]:
    """The explicit deliverable filename named in ``text`` (basename), else None.

    Shared with the acyclic ``file_write``-node path (toolargs) so an explicitly
    named ``cats.html`` survives verbatim there too — one source of truth for the
    c3r path-carry fix."""
    m = _EXPLICIT_NAME_RE.search(text or "")
    return m.group(1).replace("\\", "/").rsplit("/", 1)[-1] if m else None


def deliverable_extension(specs: Optional[Sequence[str]], text: str) -> str:
    """The deliverable extension from the bound writer spec, else ``text``, else .md."""
    return _ext_for(specs, text)


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
_DOC_URL_RE = re.compile(r"https?://[^\s\"'<>)\]}]+", re.IGNORECASE)
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


def strip_ungrounded_urls(
    doc: str,
    sources: Sequence[Mapping[str, str]],
) -> tuple[str, list[str]]:
    """Remove every URL in ``doc`` that no fetched source backs (the no-fab URL guard).

    Deterministic ground-or-REMOVE over the final assembled deliverable: collect every
    distinct URL the model wrote and, for each one whose normalized (scheme/host/path)
    form is NOT in the run's fetched-URL SET, unwrap its ``<a href>`` anchor to the
    visible text and delete any bare occurrence. Grounding is now EXACT set membership
    of actually-fetched source URLs — an inline-only secondary link inside a fetched
    article is NOT grounded (closes the d92 404 leak). Longer URLs are removed first so
    a fabricated URL that is a prefix of another in-doc string is excised cleanly.
    Returns ``(new_doc, removed_urls)``; all-grounded → ``(doc, [])`` unchanged. Never
    invents or rewrites prose — only the ungrounded link token is excised, so a report
    that cites only fetched sources is untouched and the guarantee holds for any model."""
    s = doc or ""
    if not s.strip():
        return doc, []
    fetched = fetched_url_set(sources)
    seen: set[str] = set()
    removed: list[str] = []
    for m in _DOC_URL_RE.finditer(s):
        raw = m.group(0).rstrip(_URL_TRAILING)
        norm = _normalize_fetched_url(raw)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        if norm not in fetched:
            removed.append(raw)
    if not removed:
        return doc, []
    out = s
    for bad in sorted(set(removed), key=len, reverse=True):
        esc = re.escape(bad)
        # Unwrap an anchor whose href is the fabricated URL → keep the inner text.
        anchor = re.compile(
            r"<a\b[^>]*?href\s*=\s*[\"']?" + esc + r"[^\"'>]*[\"']?[^>]*>(.*?)</a>",
            re.IGNORECASE | re.DOTALL,
        )
        out = anchor.sub(lambda mm: mm.group(1), out)
        # Drop any remaining bare occurrence (with an optional trailing slash).
        out = re.sub(esc + r"/?", "", out)
    return out, removed


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


def _insert_before_close(doc: str, block: str) -> str:
    """Insert ``block`` just before the document's last ``</body>``/``</html>`` close.

    Keeps the reference block INSIDE the document body. Falls back to appending at the
    end when there is no wrapper close (a bare HTML fragment) — the final
    :func:`reconcile_doc_structure`/:func:`html_close_gap` pass then balances the wrapper.
    """
    for pat in (_HTML_BODY_CLOSE_RE, _HTML_DOC_CLOSE_RE):
        matches = list(pat.finditer(doc))
        if matches:
            i = matches[-1].start()
            return doc[:i].rstrip() + "\n" + block + "\n" + doc[i:]
    return doc.rstrip() + "\n" + block


def ensure_source_coverage(
    doc: str,
    sources: Sequence[Mapping[str, str]],
    assigned_ids: Optional[Iterable[object]] = None,
    *,
    is_html: Optional[bool] = None,
) -> tuple[str, list[str]]:
    """Append a reference for every fetched source the write plan left UNASSIGNED (Backstop C).

    The deterministic source-coverage net (s13/B5c, design §4C — the
    ``_ensure_source_coverage`` d89 specified but never shipped). Given the assembled
    deliverable ``doc``, the run's global fetched ``sources`` (1-based,
    ``[{title,url,…}]`` from :func:`collect_fetched_sources_full`), and the set of 1-based
    ``assigned_ids`` the PHASE-2 write planner gave to a section, append an "Additional
    sources" reference block listing every fetched source that was assigned to NO section
    AND whose URL is not already present (cited/listed) anywhere in ``doc`` — so a source
    the planner skipped cannot silently vanish (the d87 dropped-source risk).

    Presence is tested by the SAME normalized-URL identity as the no-fab URL guard
    (:func:`_normalize_fetched_url`), so a source cited under a different section — or by
    another writer node on the multi-page chain — counts as covered and is NOT re-listed.
    d60-safe: adds ONLY a title+URL reference for material the run actually fetched; never
    invents a source or generates content. ``is_html`` None ⇒ inferred from ``doc``.
    Idempotent — a doc whose every fetched source is assigned or already present is
    returned UNCHANGED. Returns ``(new_doc, added_urls)``."""
    s = doc or ""
    if not s.strip() or not sources:
        return doc, []
    assigned: set[int] = set()
    for i in assigned_ids or ():
        if isinstance(i, bool):
            continue
        if isinstance(i, int):
            assigned.add(i)
        elif isinstance(i, str) and i.strip().lstrip("-").isdigit():
            assigned.add(int(i.strip()))
    # URLs already present (cited or listed) in the assembled doc — normalized so a cited
    # source matches its fetched record regardless of #anchor / ?query / trailing-slash.
    present: set[str] = set()
    for m in _DOC_URL_RE.finditer(s):
        n = _normalize_fetched_url(m.group(0).rstrip(_URL_TRAILING))
        if n:
            present.add(n)
    missing: list[tuple[str, str]] = []
    seen: set[str] = set()
    for idx, src in enumerate(sources, 1):
        if not isinstance(src, Mapping) or idx in assigned:
            continue
        url = str(src.get("url") or "").strip()
        norm = _normalize_fetched_url(url)
        if not norm or norm in present or norm in seen:
            continue
        seen.add(norm)
        missing.append((str(src.get("title") or "").strip(), url))
    if not missing:
        return doc, []
    if is_html is None:
        low = s.lower()
        is_html = any(
            t in low
            for t in ("<!doctype", "<html", "<body", "<h1", "<h2", "<section", "<ul")
        )
    added = [u for _, u in missing]
    if is_html:
        items = "\n".join(
            f'      <li><a href="{_esc_html_attr(u)}">{_esc_html_text(t or u)}</a></li>'
            for t, u in missing
        )
        block = (
            '<section class="sources additional-sources">\n'
            f"  <h2>{_ADDITIONAL_SOURCES_HEADING}</h2>\n"
            f"  <p>{_ADDITIONAL_SOURCES_CAPTION}</p>\n"
            "  <ul>\n"
            f"{items}\n"
            "  </ul>\n"
            "</section>"
        )
        return _insert_before_close(s, block), added
    items = "\n".join(f"- [{t or u}]({u})" for t, u in missing)
    block = (
        f"\n\n## {_ADDITIONAL_SOURCES_HEADING}\n\n"
        f"{_ADDITIONAL_SOURCES_CAPTION}\n\n{items}\n"
    )
    return s.rstrip() + block, added


__all__ = [
    "DONE_SENTINEL",
    "split_done_signal",
    "html_close_gap",
    "is_detailed_task",
    "strip_wrapper_closers",
    "top_level_html_doc_count",
    "dedupe_html_documents",
    "begins_html_document",
    "strip_wrapper_openers",
    "has_duplicate_html_structure",
    "enforce_single_html_document",
    "ensure_source_coverage",
    "assemble_html_spa",
    "collapse_duplicate_sections",
    "collapse_outline_duplicate_sections",
    "UNSUPPORTED_SECTION_INSTRUCTION",
    "section_reemission",
    "SECTION_ANCHOR",
    "plant_section_anchor",
    "choose_section_anchor",
    "anchored_insert_args",
    "strip_section_anchor",
    "fetched_url_set",
    "strip_ungrounded_urls",
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
