"""s9/N5 (d62/c15 part-e): the REASONING no-fabrication verify surface.

RP-3c (d330): the flag-gated engine VERIFY-AND-REVISE self-review lane (``verify_and_revise``
gated on ``verify_lane``) is RETIRED — the model self-review MOVED to the definition-layer
writer doctrine (``_COHERENT_ARTIFACT_DOCTRINE`` self-review-before-finish). What remains and
is exercised here is the MODEL-DRIVEN verify surface: ``verify_claims`` (the one reasoning
claim->source verify turn a node self-selects via the ``cross_verify`` research tool), the
verdict parse, the source-provenance render, and the ``research_answered_from_memory`` signal
that powers the DE-FLAGGED (always-on, output-agnostic) no-fab research gather-more gate.
Fully OFFLINE (a fake verifier / scripted transport).

* UNIT — the pure verify turn (a fake async verifier): the 0-fetch answered-from-memory
  PROVENANCE signal; the provenance render; the verdict parse (ok / revise / fenced /
  unreadable); ``verify_claims`` (grounded / unbacked / empty doc).
* INTEGRATION — the RESEARCH gather-more gate (runtime seam): a research-bundle node that
  answers FROM MEMORY (0 fetches) is force-gathered to read a real source before its
  findings are accepted — now DE-FLAGGED (no ``verify_lane`` boolean); a worker that did
  NOT self-select the research bundle answering from memory is NEVER force-gathered (the
  SB-RR d293 narrow gate survives the de-flag).
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from llm_framework import ChatResult

from agent_runtime.claim_verify import (
    UnbackedClaim,
    parse_verify_verdict,
    render_sources_for_verify,
    research_answered_from_memory,
    verify_claims,
)
from agent_runtime.factory import PlanNode
from agent_runtime.runtime import SubAgent


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# UNIT — (B-b) the 0-fetch / answered-from-memory PROVENANCE signal
# --------------------------------------------------------------------------- #
def test_research_answered_from_memory_signal():
    """0 fetches + substantive findings = answered FROM MEMORY (no-fab FAILURE); any
    real fetch, or a thin/empty stage, is NOT (no fabricated answer to force-revise)."""
    substantive = (
        "Iran struck back with 180 missiles on June 14, killing 12 people across "
        "several provinces and damaging infrastructure, the UN reported. " * 3
    )
    # 0 fetches + a real answer → it answered from memory (FAILURE)
    assert research_answered_from_memory(substantive, 0) is True
    # a real source was read → grounded, not a memory answer
    assert research_answered_from_memory(substantive, 2) is False
    # 0 fetches but ~empty findings → a genuinely aborted gather, not a fabrication
    assert research_answered_from_memory("", 0) is False
    assert research_answered_from_memory("too short", 0, min_chars=200) is False


# --------------------------------------------------------------------------- #
# UNIT — provenance render
# --------------------------------------------------------------------------- #
def test_render_sources_for_verify():
    """Each source renders its id/trust/title/url + key_claims + a bounded excerpt; an
    empty source set renders nothing (the caller treats a no-source doc as unbacked)."""
    assert render_sources_for_verify([]) == ""
    block = render_sources_for_verify(
        [
            {
                "title": "UN News",
                "url": "https://news.un.org/x",
                "source_trust": "secondary",
                "key_claims": ["180 missiles fired", "12 killed"],
                "markdown": "y" * 5000,
            },
            {"title": "Wiki", "url": "https://en.wikipedia.org/Iran",
             "source_trust": "reference-untrusted", "markdown": "encyclopedic"},
        ],
        excerpt_budget=300,
    )
    assert "[1] [secondary] UN News — https://news.un.org/x" in block
    assert "- 180 missiles fired" in block          # key_claims surface
    assert "[2] [reference-untrusted] Wiki" in block  # trust tier flagged (d60)
    # the long markdown is bounded to the excerpt budget (kept in-window)
    assert ("y" * 300) in block and ("y" * 301) not in block


# --------------------------------------------------------------------------- #
# UNIT — verdict parse
# --------------------------------------------------------------------------- #
def test_parse_verify_verdict():
    # a clean verdict → grounded, no unbacked claims
    ok = parse_verify_verdict('{"verdict":"ok"}')
    assert ok.grounded is True and ok.parsed is True and ok.unbacked == []
    # a revise verdict → not grounded, carries the flagged claims
    rev = parse_verify_verdict(
        '{"verdict":"revise","unbacked":[{"claim":"17 USC 107(5)","reason":"no source"}]}'
    )
    assert rev.grounded is False and len(rev.unbacked) == 1
    assert rev.unbacked[0].claim == "17 USC 107(5)" and rev.unbacked[0].reason == "no source"
    # a fenced verdict is still read (the model often fences JSON)
    fenced = parse_verify_verdict('```json\n{"verdict":"ok"}\n```')
    assert fenced.grounded is True and fenced.parsed is True
    # an unreadable reply → parsed False, NOT a silent grounded pass
    bad = parse_verify_verdict("I think it all looks fine to me.")
    assert bad.parsed is False and bad.grounded is False
    # "revise" with an EMPTY list → nothing to act on → grounded (never invent a claim)
    empty = parse_verify_verdict('{"verdict":"revise","unbacked":[]}')
    assert empty.grounded is True and empty.unbacked == []


# --------------------------------------------------------------------------- #
# UNIT — verify_claims (a fake verifier)
# --------------------------------------------------------------------------- #
_SOURCES = [
    {"title": "UN News", "url": "https://news.un.org/x", "source_trust": "secondary",
     "key_claims": ["180 missiles fired on June 14"], "markdown": "180 missiles fired."},
]


def test_verify_claims_empty_doc_is_grounded():
    """An empty deliverable is trivially grounded (nothing to fact-check) — no model call
    decides otherwise."""
    async def _never(_p: str) -> str:  # must not be called
        raise AssertionError("verify turn fired on empty doc")
    res = _run(verify_claims("", _SOURCES, verify=_never))
    assert res.grounded is True


def test_verify_claims_flags_unbacked_and_passes_clean():
    """The model REASONS groundedness: it flags a fabricated claim and passes a clean one.
    The prompt carries the source provenance + the report under a FACT-CHECK header."""
    seen: list[str] = []

    async def _fake(prompt: str) -> str:
        seen.append(prompt)
        if "17 USC 107(5)" in prompt:
            return '{"verdict":"revise","unbacked":[{"claim":"17 USC 107(5)","reason":"no fetched source mentions it"}]}'
        return '{"verdict":"ok"}'

    bad = _run(verify_claims("Under 17 USC 107(5) the strike was legal.", _SOURCES, verify=_fake))
    assert bad.grounded is False and bad.unbacked[0].claim == "17 USC 107(5)"
    clean = _run(verify_claims("180 missiles were fired on June 14.", _SOURCES, verify=_fake))
    assert clean.grounded is True
    # the verify prompt is provenance + report under the FACT-CHECK header (reasoning)
    assert "FETCHED SOURCES" in seen[0] and "REPORT TO FACT-CHECK" in seen[0]


# --------------------------------------------------------------------------- #
# INTEGRATION — the RESEARCH gather-more (answered-from-memory) gate
# --------------------------------------------------------------------------- #
@dataclass
class _ToolResult:
    ok: bool
    value: Any = None
    error: str = ""
    call_id: str = "c1"


class _FakeHook:
    def __init__(self, url: str) -> None:
        self.url = url
        self.fetches: list[str] = []

    async def invoke(self, name: str, **args) -> _ToolResult:
        if name == "web_search":
            return _ToolResult(True, {"results": [{"title": "t", "url": self.url, "snippet": "s"}]})
        if name == "web_fetch":
            url = args.get("url", "")
            self.fetches.append(url)
            return _ToolResult(True, {
                "url": url, "final_url": url, "status": 200, "title": "Iran report",
                "markdown": "Iran fired 180 missiles on June 14, the UN said.", "extracted": True,
            })
        return _ToolResult(False, error=f"unknown tool {name}")


class _MemoryThenGatherTransport:
    """Answers FROM MEMORY first; once nudged to gather, it searches → fetches → answers.

    Models the e4b-fetch-ceiling-diverges-by-path behaviour: the bare ReAct path answers
    from memory until the path STRUCTURALLY forces a real fetch."""

    def __init__(self) -> None:
        self.tool_calls = 0
        self._loaded = False

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content

    def chat(self, messages, **opts) -> ChatResult:
        # d242 TRUE self-select: a research node starts TOOL-LESS — load the 'research' bundle
        # FIRST (its opening move) so web_search/web_fetch become callable.
        if not self._loaded:
            self._loaded = True
            return ChatResult(role="assistant",
                              content='{"tool": "get_bundles", "args": {"name": "research"}}')
        # s15/a18 (d189): a fed-back tool RESULT now rides role 'tool' (the search/fetch
        # observations), while the gather-more nudge stays role 'user' — scan both.
        user = next(
            (m["content"] for m in reversed(messages)
             if m.get("role") in ("user", "tool")), ""
        )
        if "answered from MEMORY" in user:           # the N5 gather-more nudge → search now
            self.tool_calls += 1
            return ChatResult(role="assistant", content='{"tool":"web_search","args":{"query":"iran strike"}}')
        if "you have now READ" in user:              # a source was fetched → write findings
            return ChatResult(role="assistant",
                              content="FINDINGS: Iran fired 180 missiles on June 14 (https://news.un.org/iran).")
        if self.tool_calls == 1:                      # the search observation → fetch a result
            self.tool_calls += 1
            return ChatResult(role="assistant", content='{"tool":"web_fetch","args":{"url":"https://news.un.org/iran"}}')
        # the first turn (base task): answer entirely from memory, zero sources — a
        # substantive paragraph (the kind E4B fabricates from memory on the bare path).
        return ChatResult(role="assistant", content=(
            "FINDINGS: Iran and Israel exchanged strikes through June, escalating "
            "tensions sharply across the region. Several rounds of missile and drone "
            "attacks were reported, with both sides trading blame and warning of wider "
            "conflict as diplomatic channels stalled and casualties mounted steadily."))


def _research_node() -> PlanNode:
    return PlanNode(id="r1", task="[research] Iran-Israel June escalation",
                    role="worker", tool="web_search", tool_args={"query": "iran"})


def test_research_gather_more_deflagged_forces_fetch(tmp_path):
    """RP-3c (d330): the no-fab gather-more gate is DE-FLAGGED (no ``verify_lane`` boolean).
    A research-bundle node that answers FROM MEMORY (0 fetches) is STILL force-gathered — it
    then searches + reads a real source before its findings are accepted. The gate now fires
    on the output-agnostic signal alone (research bundle self-selected + answered-from-memory
    + under cap), the correct flag-free always-on end-state."""
    hook = _FakeHook("https://news.un.org/iran")
    transport = _MemoryThenGatherTransport()
    agent = SubAgent(
        _research_node(), transport=transport, hook=hook,
        read_search_max_fetch=5,
        call_opts={"think": False, "temperature": 0},
    )
    res = _run(agent.run({}))

    # the memory answer was rejected; the stage actually fetched a real source — WITHOUT any
    # flag being passed (the gate is now the flag-free default).
    assert hook.fetches == ["https://news.un.org/iran"]
    assert res.tool_value is not None and res.tool_value["fetched_count"] == 1
    assert "180 missiles" in (res.output or "")  # the grounded findings, not the memory one


class _MemoryNoBundleTransport:
    """Answers FROM MEMORY on the FIRST turn WITHOUT ever self-selecting the research bundle —
    a legitimate trivial/follow-up worker (it never loaded the gather tools)."""

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content

    def chat(self, messages, **opts) -> ChatResult:
        return ChatResult(role="assistant", content=(
            "FINDINGS: Iran and Israel exchanged strikes through June, escalating "
            "tensions sharply across the region as diplomatic channels stalled."))


def test_research_gather_more_narrow_gate_spares_non_research_worker(tmp_path):
    """The de-flagged gate stays NARROW (SB-RR d293): a worker that did NOT self-select the
    research bundle answering from memory is NEVER force-gathered — the always-on de-flag did
    not widen the gate to non-gathering workers. It answers in one turn, no nudge, no fetch."""
    hook = _FakeHook("https://news.un.org/iran")
    transport = _MemoryNoBundleTransport()
    agent = SubAgent(
        _research_node(), transport=transport, hook=hook,
        read_search_max_fetch=5,
        call_opts={"think": False, "temperature": 0},
    )
    res = _run(agent.run({}))

    assert hook.fetches == []           # never fetched — the bundle was never loaded
    assert res.tool_value is None        # no sources gathered
    assert "escalating tensions" in (res.output or "")  # the from-memory findings stood
