"""s9/N2 (d60/c15 part-b): the per-article research-CONTROL artifact — ``ArticleNote``.

Two layers, fully OFFLINE (no GPU, no network):

* UNIT — the model + coercion: provenance is RUNTIME-owned (never the model's), the
  Wikipedia source-trust policy is FORCED (d60), and a loose/partial model dict coerces
  gracefully (never crashes a research turn).
* INTEGRATION — the leaf research loop (:meth:`SubAgent._run_research_loop`) with
  ``emit_article_notes=True``: the agent records a ``note`` after reading a source, the
  note rides ADDITIVELY on ``tool_value['article_notes']`` (the c13 ``fetched`` path is
  UNCHANGED), and its follow-ups are fed back to DIRECT the next search. With notes OFF
  (the default) the loop is byte-identical — the ``note`` tool is not even recognised.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from llm_framework import ChatResult

from agent_runtime.article_note import (
    ArticleNote,
    SOURCE_TRUST_TIERS,
    classify_source_trust,
    coerce_article_note,
)
from agent_runtime.factory import PlanNode
from agent_runtime.runtime import SubAgent


# --------------------------------------------------------------------------- #
# UNIT — classify_source_trust (d60 policy)
# --------------------------------------------------------------------------- #
def test_wikipedia_is_forced_reference_untrusted_even_if_model_claims_primary():
    """d60: Wikipedia is NOT trusted but citable-IF-attributed — a TIER, not a ban. The
    host policy overrides whatever tier the model claims (anti over-trust)."""
    url = "https://en.wikipedia.org/wiki/2025_crisis"
    assert classify_source_trust(url, "primary") == "reference-untrusted"
    assert classify_source_trust(url, None) == "reference-untrusted"
    assert classify_source_trust("https://commons.wikimedia.org/x", "secondary") == (
        "reference-untrusted"
    )


def test_non_wikipedia_honours_a_valid_claim_and_degrades_unknowns():
    assert classify_source_trust("https://gov.example/report", "primary") == "primary"
    assert classify_source_trust("https://news.example/x", "news") == "secondary"  # synonym
    assert classify_source_trust("https://x.example/y", "garbage-tier") == "secondary"
    assert classify_source_trust("https://x.example/y", "") == "secondary"
    assert classify_source_trust("not a url", "reference") == "reference-untrusted"
    # every resolved tier is in the fixed control vocabulary
    for claim in ("primary", "news", "wiki", "", None, "weird"):
        assert classify_source_trust("https://x.example/y", claim) in SOURCE_TRUST_TIERS


# --------------------------------------------------------------------------- #
# UNIT — coerce_article_note (provenance + lenient coercion)
# --------------------------------------------------------------------------- #
def test_provenance_is_runtime_owned_not_model_owned():
    """The model may LIE about which source a claim came from; the runtime supplies the
    canonical id/url/title and they win — a note always describes the REAL read source."""
    note = coerce_article_note(
        {"url": "https://evil.example/spoof", "title": "spoofed",
         "summary": "s", "source_trust": "primary"},
        source_id=3, url="https://real.example/article", title="Real Title",
    )
    assert isinstance(note, ArticleNote)
    assert note.source_id == 3
    assert note.url == "https://real.example/article"
    assert note.title == "Real Title"  # model's "spoofed" ignored


def test_loose_model_shapes_coerce_gracefully():
    """A small model is inconsistent: a 'list' field may arrive as a single string, key
    names vary, fields may be missing. All coerce; nothing raises."""
    note = coerce_article_note(
        {"abstract": "a summary", "topic": "tech",
         "claims": "fact one; fact two\nfact three",  # string, not a list
         "followups": "next angle"},
        source_id=1, url="https://x.example/a",
    )
    assert note.summary == "a summary"
    assert note.category == "tech"
    assert note.key_claims == ["fact one", "fact two", "fact three"]
    assert note.gaps_or_followups == ["next angle"]
    assert note.title == ""  # absent → default, not a crash


def test_malformed_note_is_dropped_not_crashed():
    assert coerce_article_note("not a mapping", source_id=1, url="u") is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# INTEGRATION — the research loop emits notes (additive, reasoning-populated)
# --------------------------------------------------------------------------- #
@dataclass
class _ToolResult:
    ok: bool
    value: Any = None
    error: str = ""
    call_id: str = "c1"


class _FakeHook:
    """Records web_search/web_fetch; returns canned real-shaped data for given URLs."""

    def __init__(self, urls: dict[str, str]) -> None:
        self._urls = urls  # url -> markdown body
        self.searches: list[str] = []
        self.fetches: list[str] = []

    async def invoke(self, name: str, **args) -> _ToolResult:
        if name == "web_search":
            self.searches.append(args.get("query", ""))
            return _ToolResult(True, {
                "query": args.get("query", ""),
                "results": [
                    {"title": f"t{i}", "url": u, "snippet": "snip"}
                    for i, u in enumerate(self._urls)
                ],
                "count": len(self._urls),
            })
        if name == "web_fetch":
            url = args.get("url", "")
            self.fetches.append(url)
            return _ToolResult(True, {
                "url": url, "final_url": url, "status": 200,
                "title": url.rsplit("/", 1)[-1],
                "markdown": self._urls.get(url, "article body"),
                "extracted": True,
            })
        return _ToolResult(False, error=f"unknown tool {name}")


class _ScriptedTransport:
    """Replays a fixed sequence of agent turns; records the user turn of each call.

    d242 TRUE self-select: a research node starts TOOL-LESS, so every script SELF-SELECTS
    the 'research' bundle first (the model's opening move) before its note-lane gather
    turns; pass ``self_select=None`` to drive the self-select turns explicitly."""

    def __init__(self, turns: list[str], *, self_select: str | None = "research") -> None:
        lead = [f'{{"tool": "get_bundles", "args": {{"name": "{self_select}"}}}}'] if self_select else []
        self._turns = lead + list(turns)
        self.calls: list[str] = []
        self.last_messages: list[dict] = []

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content

    def chat(self, messages, **opts) -> ChatResult:
        i = len(self.calls)
        user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        self.calls.append(user)
        # s15/a18 (d189): tool RESULTS now ride role 'tool' (not 'user'), so capture the
        # WHOLE message stream of the latest call — assertions about fed-back observations
        # must scan the tool channel, not just the user turns.
        self.last_messages = list(messages)
        content = self._turns[i] if i < len(self._turns) else "FALLBACK FINDINGS."
        return ChatResult(role="assistant", content=content)


def _ws_node(nid: str = "r1_research") -> PlanNode:
    return PlanNode(id=nid, task="[research] crisis timeline",
                    role="worker", tool="web_search", tool_args={"query": "crisis"})


def _full_convo(t: _ScriptedTransport) -> str:
    # s15/a18 (d189): the agent-facing conversation now carries fed-back tool RESULTS as
    # role 'tool' (not 'user'); include both so observation content (note acks, refusals,
    # search/fetch results) is visible to assertions, mirroring what the model actually sees.
    return "\n\n".join(
        str(m.get("content", ""))
        for m in t.last_messages
        if m.get("role") in ("user", "tool")
    )


def test_note_is_emitted_additively_and_directs_next_search():
    """With notes ON the agent records a per-article note after reading; it is reasoning-
    populated, rides on tool_value['article_notes'] WITHOUT disturbing the c13 'fetched'
    path, and its follow-ups are fed back to steer the next search."""
    world = "https://news.example.com/world"
    hook = _FakeHook({world: "WORLD ARTICLE\n\nbody text."})
    transport = _ScriptedTransport([
        '{"tool": "web_search", "args": {"query": "world crisis"}}',
        f'{{"tool": "web_fetch", "args": {{"url": "{world}"}}}}',
        '{"tool": "note", "args": {"url": "' + world + '", "summary": "a summit accord", '
        '"category": "world", "source_trust": "secondary", '
        '"key_claims": ["accord signed", "ceasefire holds"], '
        '"relevance": "directly on topic", "gaps_or_followups": ["search casualty figures"]}}',
        "FINDINGS: a summit reached an accord (" + world + ").",
    ])
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=5, emit_article_notes=True,
        call_opts={"think": False, "temperature": 0},
    )
    res = asyncio.run(agent.run({}))

    # c13 path UNCHANGED: the real source is still attached as fetched/fetched_count.
    assert hook.fetches == [world]
    assert res.tool_value["fetched_count"] == 1
    assert res.tool_value["fetched"][0]["url"] == world

    # the CONTROL note rides additively, reasoning-populated, provenance runtime-owned.
    notes = res.tool_value["article_notes"]
    assert len(notes) == 1
    n = notes[0]
    assert n["source_id"] == 1 and n["url"] == world
    assert n["summary"] == "a summit accord" and n["category"] == "world"
    assert n["key_claims"] == ["accord signed", "ceasefire holds"]
    assert n["source_trust"] == "secondary"
    assert n["gaps_or_followups"] == ["search casualty figures"]

    # the note STEERS the next search: its follow-ups are fed back into the loop.
    assert "open follow-ups: search casualty figures" in _full_convo(transport)
    # findings prose is still the RAW node output.
    assert "summit reached an accord" in (res.output or "")


def test_note_for_wikipedia_source_is_forced_reference_untrusted():
    """Even if the model tags a Wikipedia source 'primary', the recorded note's trust is
    reconciled to reference-untrusted (d60) — the policy holds end-to-end through the loop."""
    wiki = "https://en.wikipedia.org/wiki/Some_Event"
    hook = _FakeHook({wiki: "WIKI ARTICLE\n\nencyclopaedic body."})
    transport = _ScriptedTransport([
        '{"tool": "web_search", "args": {"query": "some event"}}',
        f'{{"tool": "web_fetch", "args": {{"url": "{wiki}"}}}}',
        '{"tool": "note", "args": {"url": "' + wiki + '", "summary": "overview", '
        '"source_trust": "primary", "key_claims": ["x"]}}',
        "FINDINGS: an overview attributed to Wikipedia (" + wiki + ").",
    ])
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=5, emit_article_notes=True,
        call_opts={"think": False},
    )
    res = asyncio.run(agent.run({}))
    assert res.tool_value["article_notes"][0]["source_trust"] == "reference-untrusted"


def test_note_before_any_read_is_refused():
    """Anti-fabrication: a note can only describe a source the agent has actually read.
    A note emitted before any fetch is refused (no phantom note), and the loop continues."""
    world = "https://news.example.com/world"
    hook = _FakeHook({world: "WORLD ARTICLE\n\nbody."})
    transport = _ScriptedTransport([
        '{"tool": "note", "args": {"url": "' + world + '", "summary": "premature"}}',
        '{"tool": "web_search", "args": {"query": "world"}}',
        f'{{"tool": "web_fetch", "args": {{"url": "{world}"}}}}',
        "FINDINGS from the one source I actually read.",
    ])
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=5, emit_article_notes=True, call_opts={"think": False},
    )
    res = asyncio.run(agent.run({}))
    assert "no source has been read this task" in _full_convo(transport)
    # no phantom note was recorded; the real fetch still produced its source.
    assert "article_notes" not in res.tool_value
    assert res.tool_value["fetched_count"] == 1


def test_notes_off_by_default_is_byte_identical_no_note_tool():
    """Default (emit_article_notes=False): the loop never advertises or accepts a 'note'
    tool and never adds an article_notes key — true no-regression for every standing path."""
    world = "https://news.example.com/world"
    hook = _FakeHook({world: "WORLD ARTICLE\n\nbody."})
    transport = _ScriptedTransport([
        '{"tool": "web_search", "args": {"query": "world"}}',
        f'{{"tool": "web_fetch", "args": {{"url": "{world}"}}}}',
        "FINDINGS: a concise answer.",
    ])
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=5, call_opts={"think": False},
    )
    res = asyncio.run(agent.run({}))
    assert res.tool_value["fetched_count"] == 1
    assert "article_notes" not in res.tool_value
    # the 'note' tool is not recognised when notes are off → a note JSON would be treated
    # as findings, never dispatched.
    assert agent._parse_research_call(
        '{"tool": "note", "args": {"summary": "x"}}'
    ) is None
