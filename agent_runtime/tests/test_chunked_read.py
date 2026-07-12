"""s9/N3 (d62/c15 part-d): the READ-side chunked map/reduce — ``chunked_read``.

The write-side 512-token SWA window (d55/d59) has a READ-side twin: a long source
overflows the per-source window, so the flat ``md[:budget]`` read dropped everything
past the cut. ``chunked_read`` MAPs each in-window chunk to a factual summary and
REDUCEs forward (running summary flows into the next chunk), so the WHOLE document is
read within the window — no truncation, no fabrication. Fully OFFLINE.

* UNIT — the pure map/reduce (a fake async summarizer): short input is verbatim with
  NO model call; a long source is covered end-to-end; the summary flows forward; an
  empty reply never loses a section; the result stays in-window; the prompts are
  grounded-only; paragraph packing + oversized hard-split keep every chunk in-window.
* INTEGRATION — the research loop (:meth:`SubAgent._run_research_loop`) with
  ``chunked_read=True``: a LONG fetched source is read via the summary so content PAST
  the legacy truncation reaches the window (and rides additively as ``summary`` on the
  fetched dict, the c13 ``markdown`` path UNCHANGED). With the flag OFF (default) the
  read is the byte-identical ``md[:budget]`` truncation and no summarizer fires.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from llm_framework import ChatResult

from agent_runtime.chunked_read import chunked_read, split_chunks
from agent_runtime.factory import PlanNode
from agent_runtime.runtime import SubAgent


# --------------------------------------------------------------------------- #
# UNIT — chunked_read (pure map/reduce via a fake summarizer)
# --------------------------------------------------------------------------- #
class _FakeSummarizer:
    """Records each prompt; returns the ORDERED, deduped ``PART<n>MARK`` tokens it sees.

    Because the refine prompt carries the running summary (prior marks) PLUS the new
    chunk (the new mark), the returned summary accumulates the marks in order — so the
    final summary proves BOTH whole-document coverage and forward flow."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        seen: list[str] = []
        for m in re.findall(r"PART\d+MARK", prompt):
            if m not in seen:
                seen.append(m)
        return " ".join(seen)


def _marked_source(n: int, *, filler: int = 90) -> str:
    """``n`` blank-line-separated paragraphs, each ~100 chars with a unique mark."""
    return "\n\n".join(f"PART{i}MARK " + ("x" * filler) for i in range(1, n + 1))


def test_short_source_returned_verbatim_with_no_model_call():
    """A source already within budget is returned as-is and the summarizer NEVER fires —
    the short/clean-source path is unchanged (no needless LLM call)."""
    fake = _FakeSummarizer()
    md = "PART1MARK short body."
    out = asyncio.run(chunked_read(md, summarize=fake, char_budget=2000))
    assert out == md
    assert fake.prompts == []  # no summarization at all


def test_long_source_covers_whole_document():
    """Every chunk's content reaches the summary — not just the first budget chars: all
    marks survive (whole-document read) and the summarizer fires once per chunk."""
    fake = _FakeSummarizer()
    md = _marked_source(3)  # ~300 chars > budget → 3 chunks at chunk_chars=120
    out = asyncio.run(
        chunked_read(md, summarize=fake, char_budget=80, chunk_chars=120)
    )
    assert len(fake.prompts) == 3  # one map step per chunk
    for i in (1, 2, 3):
        assert f"PART{i}MARK" in out  # the LAST chunk is present, not truncated away


def test_summary_flows_forward():
    """REDUCE = the running summary flows into the next chunk's prompt: each refine step
    after the first carries the prior marks forward (so nothing earlier is dropped)."""
    fake = _FakeSummarizer()
    md = _marked_source(3)
    asyncio.run(chunked_read(md, summarize=fake, char_budget=80, chunk_chars=120))
    # first map step is the FIRST-chunk prompt (no running summary header)
    assert "RUNNING FACTUAL SUMMARY" not in fake.prompts[0]
    # the 2nd step carries chunk-1's mark forward as the running summary
    assert "RUNNING FACTUAL SUMMARY" in fake.prompts[1]
    assert "PART1MARK" in fake.prompts[1]
    # the 3rd step carries chunks 1+2 forward
    assert "PART1MARK" in fake.prompts[2] and "PART2MARK" in fake.prompts[2]


def test_empty_reply_never_loses_a_section():
    """Anti-loss: if the model returns nothing for a chunk, that chunk's text is carried
    forward instead of being dropped — the read is never silently empty/short."""

    async def _blank(_prompt: str) -> str:
        return ""

    md = _marked_source(2)
    out = asyncio.run(chunked_read(md, summarize=_blank, char_budget=4000, chunk_chars=120))
    assert out  # non-empty
    assert "PART1MARK" in out and "PART2MARK" in out  # both sections retained


def test_result_is_bounded_to_char_budget():
    """The returned summary stays in-window (bounded to the per-source budget)."""

    async def _flood(_prompt: str) -> str:
        return "y" * 5000

    md = _marked_source(3)
    out = asyncio.run(chunked_read(md, summarize=_flood, char_budget=200, chunk_chars=120))
    assert len(out) <= 200


def test_prompts_are_grounded_only_anti_fabrication():
    """Every map prompt mandates grounding in the provided text only (no fabrication)."""
    fake = _FakeSummarizer()
    md = _marked_source(2)
    asyncio.run(chunked_read(md, summarize=fake, char_budget=80, chunk_chars=120))
    for p in fake.prompts:
        assert "do NOT add outside knowledge" in p
        assert "Use ONLY what the" in p


def test_split_chunks_packs_paragraphs_and_hard_splits_oversized():
    """Paragraphs pack greedily; a paragraph longer than the limit is hard-split so NO
    chunk ever exceeds the limit (window safety)."""
    small = "a" * 50
    chunks = split_chunks(f"{small}\n\n{small}\n\n{small}", 120)
    assert all(len(c) <= 120 for c in chunks)
    # an oversized single paragraph is hard-split into <= limit pieces
    oversized = split_chunks("z" * 300, 120)
    assert len(oversized) == 3 and all(len(c) <= 120 for c in oversized)
    assert split_chunks("   ", 120) == []  # blank → no chunks


# --------------------------------------------------------------------------- #
# INTEGRATION — the research loop reads a long source via the summary
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
        self.fetches: list[str] = []

    async def invoke(self, name: str, **args) -> _ToolResult:
        if name == "web_search":
            return _ToolResult(True, {
                "results": [
                    {"title": f"t{i}", "url": u, "snippet": "s"}
                    for i, u in enumerate(self._urls)
                ],
            })
        if name == "web_fetch":
            url = args.get("url", "")
            self.fetches.append(url)
            return _ToolResult(True, {
                "url": url, "final_url": url, "status": 200,
                "title": url.rsplit("/", 1)[-1],
                "markdown": self._urls.get(url, "body"),
                "extracted": True,
            })
        return _ToolResult(False, error=f"unknown tool {name}")


class _SmartTransport:
    """Replays scripted AGENT turns; serves chunked-read summary turns out-of-band.

    A summarization call (its user turn contains ``FACTUAL SUMMARY``) does NOT consume
    a scripted agent turn — it returns the ordered ``*MARK`` tokens it sees, so the
    test can assert which source content reached the window."""

    def __init__(self, agent_turns: list[str]) -> None:
        # d242 TRUE self-select: the research node starts TOOL-LESS, so the script loads the
        # 'research' bundle first (its opening move) before any web_search/web_fetch.
        self._turns = ['{"tool": "get_bundles", "args": {"name": "research"}}'] + list(agent_turns)
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
            seen: list[str] = []
            for m in re.findall(r"[A-Z]+MARK", user):
                if m not in seen:
                    seen.append(m)
            return ChatResult(role="assistant", content="SUMMARY: " + " ".join(seen))
        i = len(self.agent_calls)
        self.agent_calls.append(user)
        content = self._turns[i] if i < len(self._turns) else "FALLBACK FINDINGS."
        return ChatResult(role="assistant", content=content)


def _ws_node() -> PlanNode:
    return PlanNode(id="r1_research", task="[research] crisis timeline",
                    role="worker", tool="web_search", tool_args={"query": "crisis"})


# A long source: TAILMARK sits PAST the 2000-char default budget, so the legacy
# md[:2000] read would never surface it.
_LONG = "HEADMARK intro. " + ("filler word " * 240) + " body. TAILMARK conclusion."


def test_chunked_read_on_reads_whole_long_source():
    """Flag ON: a long fetched source is read via the map/reduce summary, so content
    PAST the legacy truncation (TAILMARK) reaches the window and rides additively as
    ``summary`` on the fetched dict — the c13 ``markdown`` path keeping the full text."""
    assert _LONG.index("TAILMARK") > 2000  # the property under test
    url = "https://news.example.com/world"
    hook = _FakeHook({url: _LONG})
    transport = _SmartTransport([
        '{"tool": "web_search", "args": {"query": "world"}}',
        f'{{"tool": "web_fetch", "args": {{"url": "{url}"}}}}',
        "FINDINGS: a concise answer.",
    ])
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=5, chunked_read=True,
        call_opts={"think": False, "temperature": 0},
    )
    res = asyncio.run(agent.run({}))

    assert transport.summarize_calls >= 1  # the read was map/reduced, not truncated
    art = res.tool_value["fetched"][0]
    assert "TAILMARK" in art["summary"]  # whole-document read reached the window
    assert art["markdown"] == _LONG  # c13 verbatim path: full real text UNTOUCHED
    # the downstream d17 feed prefers the in-window summary (window-safe)
    rendered = agent._render_tool_value(res.tool_value)
    assert "TAILMARK" in rendered


def test_chunked_read_off_by_default_is_byte_identical_truncation():
    """Default (chunked_read=False): the read is the legacy md[:budget] truncation —
    no summarizer fires, no ``summary`` key, content past the cut never reaches the
    window. True no-regression for every standing research path."""
    url = "https://news.example.com/world"
    hook = _FakeHook({url: _LONG})
    transport = _SmartTransport([
        '{"tool": "web_search", "args": {"query": "world"}}',
        f'{{"tool": "web_fetch", "args": {{"url": "{url}"}}}}',
        "FINDINGS: a concise answer.",
    ])
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=5, call_opts={"think": False},
    )
    res = asyncio.run(agent.run({}))

    assert transport.summarize_calls == 0  # no map/reduce
    art = res.tool_value["fetched"][0]
    assert "summary" not in art
    assert art["markdown"] == _LONG  # full text still stored (for c13)
    # the in-loop observation was the budget-bounded truncation (no TAILMARK)
    fetch_obs = transport.agent_calls[-1]  # the user turn after the web_fetch
    assert "TAILMARK" not in fetch_obs
