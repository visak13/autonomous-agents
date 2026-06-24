"""s13 Stage B-FIX / FX-read (d109): RELEVANCE-SELECT-then-SINGLE-READ.

The research READ path no longer map/reduces a long fetched source into a whole-doc
summary (the s9/N3 75-chunk map/reduce). Instead it splits the source into SMALL
paragraph-granular RANKING chunks, RANKS them by EMBEDDING similarity (MiniLM 384-d, the
memory store's ``CpuEmbedder`` — NOT lexical overlap) to the node's sub-question, and
assembles the TOP relevant passages up to the FX0 token budget for a SINGLE in-window
read. These tests are FAST and OFFLINE — a deterministic bag-of-words fake stands in for
MiniLM, so the ranking is exercised without loading the real ONNX model.

* UNIT — ``select_relevant_chunks``: ranks chunks by similarity to the sub-question (a
  buried relevant passage beats the off-topic lede — NOT whole-doc / NOT first-budget);
  honors the char budget; returns the M-found / X-read counts; degrades safely with no
  query / a single chunk.
* UNIT — the FX0 token budget: the ~20k-token content budget stays under the 32768 total
  window guard; ranking chunks honor the flat ~3k bound.
* INTEGRATION — ``SubAgent._read_fetched`` with an embedder wired does the relevance-select
  read (verbatim excerpt, no map/reduce summary) and the research loop surfaces the HONEST
  signal (M relevant passages / top-X / N sources with provenance). With NO embedder wired
  the read falls back to the bounded map/reduce (the legacy path stays reachable).
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from llm_framework import ChatResult

from agent_runtime.factory import PlanNode
from agent_runtime.runtime import SubAgent
from agent_runtime.synth_tools import (
    READ_CHARS_PER_TOKEN,
    READ_CONTENT_TOKEN_BUDGET,
    READ_RANKING_CHUNK_CHARS,
    READ_TOTAL_TOKEN_GUARD,
    read_content_char_budget,
    select_relevant_chunks,
)


# --------------------------------------------------------------------------- #
# A deterministic, offline stand-in for the MiniLM CpuEmbedder.
# --------------------------------------------------------------------------- #
class _FakeEmbedder:
    """Bag-of-words embedder over a fixed vocab → L2-normalized vectors, so cosine
    similarity ranks a chunk by the topic words it SHARES with the query. A faithful
    proxy for ``CpuEmbedder.embed`` (same (n, d) float32 normalized contract) that keeps
    the read tests fast and model-free."""

    VOCAB = (
        "casualties damage strike missile oil refinery timeline diplomacy "
        "weather sports recipe cooking gardening"
    ).split()

    def embed(self, texts: Any) -> np.ndarray:
        rows = []
        for t in texts:
            tl = str(t).lower()
            v = np.array([float(tl.count(w)) for w in self.VOCAB], dtype=np.float32)
            n = float(np.linalg.norm(v))
            rows.append(v / n if n > 0 else v)  # zero vector → orthogonal to everything
        return np.asarray(rows, dtype=np.float32)


def _para(keyword: str, *, chars: int = 2800) -> str:
    """One paragraph (no blank line inside) dominated by ``keyword``, ~``chars`` long, so
    ``split_chunks`` at the 3k default keeps it as its OWN ranking chunk."""
    unit = f"{keyword} "
    return (unit * (chars // len(unit))).strip()


# Off-topic LEDE first, the topically-relevant passage BURIED second, a third off-topic
# tail — so selecting the relevant passage proves true ranking, not a lede/whole-doc read.
_LEDE = _para("weather sports recipe")
_RELEVANT = _para("casualties damage strike")
_TAIL = _para("gardening cooking")
_DOC = f"{_LEDE}\n\n{_RELEVANT}\n\n{_TAIL}"
_SUBQ = "casualties and damage figures from the strike"


# --------------------------------------------------------------------------- #
# UNIT — select_relevant_chunks
# --------------------------------------------------------------------------- #
def test_ranks_relevant_chunk_above_offtopic_lede():
    """A tight budget (one chunk fits) selects the BURIED relevant passage, NOT the
    off-topic lede — the read ranks by embedding similarity to the sub-question, it does
    not return the document's first chars."""
    excerpt, found, read = select_relevant_chunks(
        _DOC, _SUBQ, _FakeEmbedder().embed, char_budget=READ_RANKING_CHUNK_CHARS + 50
    )
    assert found == 3  # all three paragraphs were ranking chunks
    assert read == 1  # only the single most-relevant fit the tight budget
    assert "casualties" in excerpt and "damage" in excerpt
    assert "weather" not in excerpt and "gardening" not in excerpt  # off-topic excluded


def test_returns_counts_for_honest_signal():
    """Returns (excerpt, M_found, X_read): with a generous budget more chunks are read,
    but the count is still bounded by the total — the numbers feed the honest signal."""
    excerpt, found, read = select_relevant_chunks(
        _DOC, _SUBQ, _FakeEmbedder().embed, char_budget=10 * READ_RANKING_CHUNK_CHARS
    )
    assert found == 3
    assert 1 <= read <= 3
    assert "casualties" in excerpt  # the relevant passage is in the assembled read


def test_excerpt_honors_char_budget():
    """The assembled excerpt never exceeds the char budget (window safety)."""
    budget = READ_RANKING_CHUNK_CHARS + 100
    excerpt, _, _ = select_relevant_chunks(
        _DOC, _SUBQ, _FakeEmbedder().embed, char_budget=budget
    )
    assert len(excerpt) <= budget


def test_document_order_preserved_in_assembled_excerpt():
    """Selected chunks are restored to DOCUMENT order for readable verbatim flow: when
    the lede and the relevant passage are both selected, the lede text precedes the
    relevant text even though the relevant chunk ranked first."""
    # A budget big enough for two ~2.8k chunks.
    excerpt, _, read = select_relevant_chunks(
        f"{_RELEVANT}\n\n{_para('diplomacy timeline')}",
        _SUBQ,
        _FakeEmbedder().embed,
        char_budget=10 * READ_RANKING_CHUNK_CHARS,
    )
    assert read >= 1
    assert excerpt.startswith("casualties")  # the relevant (and first) chunk leads


def test_no_query_falls_back_to_leading_slice():
    """No sub-question → nothing to rank → the leading slice is returned deterministically
    (the function never raises; ranking is simply skipped)."""
    excerpt, found, read = select_relevant_chunks(
        _DOC, "", _FakeEmbedder().embed, char_budget=READ_RANKING_CHUNK_CHARS
    )
    assert excerpt == _DOC[:READ_RANKING_CHUNK_CHARS]
    assert found >= 1 and read == found


def test_empty_markdown_is_safe():
    """Empty input yields an empty read with zero counts (no crash, no model call)."""
    assert select_relevant_chunks("", _SUBQ, _FakeEmbedder().embed, char_budget=3000) == (
        "",
        0,
        0,
    )


# --------------------------------------------------------------------------- #
# UNIT — the FX0 token budget guard
# --------------------------------------------------------------------------- #
def test_content_budget_stays_under_total_window_guard():
    """FX0/d108: the ~20k-token CONTENT budget leaves headroom under the 32768 total
    window guard for the question + history + generation reserve."""
    assert READ_CONTENT_TOKEN_BUDGET == 20_000
    assert READ_CONTENT_TOKEN_BUDGET < READ_TOTAL_TOKEN_GUARD
    # at least ~12k tokens reserved for prompt + generation
    assert READ_TOTAL_TOKEN_GUARD - READ_CONTENT_TOKEN_BUDGET >= 12_000
    # the char budget is the token budget at the conservative ~4 chars/token calibration
    assert read_content_char_budget() == READ_CONTENT_TOKEN_BUDGET * READ_CHARS_PER_TOKEN


def test_ranking_chunk_is_flat_and_small_relative_to_budget():
    """The ranking chunk is a small flat ~3k passage — far below the assembled content
    budget, so many chunks compete for the top ranks (no per-call adaptive sizing)."""
    assert READ_RANKING_CHUNK_CHARS == 3000
    assert READ_RANKING_CHUNK_CHARS * 4 <= read_content_char_budget()


# --------------------------------------------------------------------------- #
# INTEGRATION — the research read path + honest signal
# --------------------------------------------------------------------------- #
@dataclass
class _ToolResult:
    ok: bool
    value: Any = None
    error: str = ""
    call_id: str = "c1"


class _FakeHook:
    def __init__(self, urls: dict[str, str]) -> None:
        self._urls = urls

    async def invoke(self, name: str, **args) -> _ToolResult:
        if name == "web_search":
            return _ToolResult(
                True,
                {
                    "results": [
                        {"title": f"t{i}", "url": u, "snippet": "s"}
                        for i, u in enumerate(self._urls)
                    ]
                },
            )
        if name == "web_fetch":
            url = args.get("url", "")
            return _ToolResult(
                True,
                {
                    "url": url,
                    "final_url": url,
                    "status": 200,
                    "title": url.rsplit("/", 1)[-1],
                    "markdown": self._urls.get(url, "body"),
                    "extracted": True,
                },
            )
        return _ToolResult(False, error=f"unknown tool {name}")


class _ScriptTransport:
    """Replays scripted AGENT turns; a map/reduce summarization call (its user turn
    contains ``FACTUAL SUMMARY``) is served out-of-band so it does not consume an agent
    turn — and counted, so a test can assert NO map/reduce happened on the select path."""

    def __init__(self, agent_turns: list[str]) -> None:
        self._turns = list(agent_turns)
        self.agent_calls: list[str] = []
        self.summarize_calls = 0

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content

    def chat(self, messages, **opts) -> ChatResult:
        user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        if "FACTUAL SUMMARY" in user:
            self.summarize_calls += 1
            return ChatResult(role="assistant", content="SUMMARY")
        i = len(self.agent_calls)
        self.agent_calls.append(user)
        content = self._turns[i] if i < len(self._turns) else "FINDINGS: done."
        return ChatResult(role="assistant", content=content)


def _research_node() -> PlanNode:
    return PlanNode(
        id="r1_research",
        task="casualties and damage figures from the strike",
        role="worker",
        tool="web_search",
        tool_args={"query": "strike casualties"},
    )


def _drive(transport, hook, **kw) -> Any:
    agent = SubAgent(
        _research_node(),
        transport=transport,
        hook=hook,
        read_search_max_fetch=5,
        chunked_read=True,
        call_opts={"think": False, "temperature": 0},
        **kw,
    )
    return asyncio.run(agent.run({})), agent


def test_read_fetched_selects_relevant_passages_no_mapreduce():
    """With an embedder wired, ``_read_fetched`` of a long source returns the relevance
    SELECTED excerpt (the buried relevant passage, NOT the lede) with ``summary=None`` and
    the M/X signal — and NO map/reduce summarizer fires."""
    # A high fetch cap squeezes the per-source budget down (20k tok / cap), so only the
    # single most-relevant chunk fits — surfacing the ranking through the real read path.
    agent = SubAgent(
        _research_node(),
        transport=_ScriptTransport([]),
        hook=_FakeHook({}),
        read_search_max_fetch=200,
        chunked_read=True,
        fetched_char_budget=500,  # << len(_DOC) so the long-source path is taken
        read_embedder=_FakeEmbedder(),
        call_opts={"think": False},
    )
    body, summary, signal = asyncio.run(
        agent._read_fetched(_DOC, "Strike report", "https://news.example.com/strike")
    )
    assert summary is None  # relevance-select, not a map/reduce summary
    assert signal is not None and signal["found"] == 3
    assert signal["read"] == 1  # tight per-source budget → only the top-ranked chunk fits
    assert "casualties" in body and "damage" in body  # the relevant passage was selected
    assert "weather" not in body and "gardening" not in body  # off-topic chunks ranked out


def test_research_loop_emits_honest_provenance_signal():
    """The fetch observation reports the HONEST coverage signal — M relevant passages,
    the top-X read, and N sources with provenance — replacing the vague 'there is MORE'
    nudge, and no map/reduce summarizer fires on the select path."""
    url = "https://news.example.com/strike"
    transport = _ScriptTransport(
        [
            '{"tool": "web_search", "args": {"query": "strike"}}',
            f'{{"tool": "web_fetch", "args": {{"url": "{url}"}}}}',
            "FINDINGS: casualties and damage figures recorded.",
        ]
    )
    res, _agent = _drive(
        transport, _FakeHook({url: _DOC}), fetched_char_budget=500, read_embedder=_FakeEmbedder()
    )
    fetch_obs = transport.agent_calls[-1]  # the user turn carrying the fetch observation
    assert "relevant passages in this source" in fetch_obs
    assert "reading the top" in fetch_obs
    assert "You have now read 1 source(s)" in fetch_obs
    assert "strike" in fetch_obs  # provenance: the source name is reported
    assert "found 3 relevant passages" in fetch_obs  # the M-found count is the real total
    assert transport.summarize_calls == 0  # the select read makes NO model summary call
    # the relevant passage really reached the window (verbatim selected chunks)
    assert "casualties" in fetch_obs and "damage" in fetch_obs


def test_no_embedder_falls_back_to_mapreduce():
    """With NO embedder wired, a long source still reads via the bounded map/reduce — the
    legacy fallback path stays reachable (never a lexical rank, never an empty read)."""
    url = "https://news.example.com/strike"
    transport = _ScriptTransport(
        [
            '{"tool": "web_search", "args": {"query": "strike"}}',
            f'{{"tool": "web_fetch", "args": {{"url": "{url}"}}}}',
            "FINDINGS: done.",
        ]
    )
    res, _agent = _drive(transport, _FakeHook({url: _DOC}), fetched_char_budget=500)
    assert transport.summarize_calls >= 1  # map/reduce fallback fired (no embedder)
    art = res.tool_value["fetched"][0]
    assert "summary" in art  # the whole-doc summary rode additively
    assert art["markdown"] == _DOC  # c13 verbatim path untouched
