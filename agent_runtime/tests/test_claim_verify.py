"""s9/N5 (d62/c15 part-e): the REASONING no-fabrication VERIFICATION lane.

The PRIMARY no-fabrication mechanism: after gather + write, re-check every deliverable
claim against the FETCHED sources via claim->source PROVENANCE and force the model to
GROUND or REVISE/REMOVE any unbacked claim (the c13r B2 narrative-fabrication gap, e.g.
the fabricated ``17 USC 107(5)`` / ``CTEA-1998``). REASONING, NEVER a regex/string
content-filter (d14/d48): the model judges groundedness AND rewrites; the lane only
orchestrates the turns. Fully OFFLINE (a fake verifier / scripted transport).

* UNIT — the pure lane (a fake async verifier): the 0-fetch answered-from-memory
  PROVENANCE signal; the provenance render; the verdict parse (ok / revise / fenced /
  unreadable); ``verify_claims`` (grounded / unbacked / empty doc); ``verify_and_revise``
  (clean → untouched; unbacked → ground-or-remove rewrite; the retention floor rejects a
  truncated/over-pruned rewrite; an unreadable verdict surfaces without stripping;
  prompts are grounded-only).
* INTEGRATION — the runtime seams with ``verify_lane=True``: (a) a SYNTHESIS deliverable
  carrying a fabricated claim is re-checked, the model flags it, a revise turn grounds-or-
  removes it, and the corrected file is re-persisted (OFF default → byte-identical, no
  verify turn); (b) a RESEARCH node that answers FROM MEMORY (0 fetches) is forced to
  GATHER-MORE before its findings are accepted (OFF default → the memory answer stands).
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from llm_framework import ChatResult, FakeTransport

from agent_runtime.claim_verify import (
    UnbackedClaim,
    parse_verify_verdict,
    render_sources_for_verify,
    research_answered_from_memory,
    verify_and_revise,
    verify_claims,
)
from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.runtime import AgentRuntime, SubAgent
from agent_runtime.synth_tools import DONE_SENTINEL
from reactive_tools import EventPlane, ToolHook, register_agentic_tools


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
# UNIT — verify_and_revise (the GROUND-or-REVISE/REMOVE checkpoint)
# --------------------------------------------------------------------------- #
_REAL = (
    "On June 14 Iran fired 180 missiles, the UN reported. Twelve people were killed. "
    "Damage assessments are still ongoing across the affected provinces. "
)
_FAB = "Under 17 USC 107(5), enacted by the CTEA of 1998, the strike was deemed lawful. "


def test_verify_and_revise_clean_doc_is_untouched():
    """A grounded deliverable is returned UNCHANGED — never nagged or stripped (the
    steer's 'do not strip valid content'); the revise turn never fires."""
    async def _fake(prompt: str) -> str:
        assert "REPORT TO CORRECT" not in prompt  # no revise on a clean doc
        return '{"verdict":"ok"}'
    res = _run(verify_and_revise(_REAL, _SOURCES, verify=_fake))
    assert res.grounded is True and res.revised is False and res.document == _REAL


def test_verify_and_revise_grounds_or_removes_unbacked():
    """An unbacked claim is flagged, then a revise turn REMOVES it (grounded-only — never
    invents), and the re-verify confirms grounded. The corrected doc keeps the real
    content and drops the fabrication."""
    async def _fake(prompt: str) -> str:
        if "REPORT TO CORRECT" in prompt:
            return _REAL  # ground-or-remove → the fabrication is gone, real content kept
        # verify: flag while the fabrication is present, pass once it's gone
        if "17 USC 107(5)" in prompt:
            return '{"verdict":"revise","unbacked":[{"claim":"17 USC 107(5)","reason":"no source"}]}'
        return '{"verdict":"ok"}'

    res = _run(verify_and_revise(_REAL + _FAB, _SOURCES, verify=_fake, max_passes=2))
    assert res.revised is True and res.grounded is True
    assert "17 USC 107(5)" not in res.document and "180 missiles" in res.document


def test_verify_and_revise_rejects_truncated_rewrite():
    """SAFEGUARD: a revise turn that returns a catastrophically SHORT doc (a truncated /
    over-pruned rewrite) is REJECTED — the original stands, the deliverable is never
    blanked or gutted (the 'don't blank the deliverable' floor)."""
    async def _fake(prompt: str) -> str:
        if "REPORT TO CORRECT" in prompt:
            return "x."  # a truncated rewrite, far below the retention floor
        return '{"verdict":"revise","unbacked":[{"claim":"17 USC 107(5)","reason":"no source"}]}'

    res = _run(verify_and_revise(_REAL + _FAB, _SOURCES, verify=_fake, max_passes=1))
    assert res.revised is False and res.document == _REAL + _FAB  # original retained
    assert res.trace[-1].get("rejected_short") is True


def test_verify_and_revise_unreadable_verdict_surfaces_without_stripping():
    """An UNREADABLE verify reply surfaces the failure (grounded False) but NEVER strips
    or rewrites the document on noise."""
    async def _fake(prompt: str) -> str:
        assert "REPORT TO CORRECT" not in prompt  # never revise on an unreadable verdict
        return "honestly it reads fine"
    res = _run(verify_and_revise(_REAL, _SOURCES, verify=_fake))
    assert res.grounded is False and res.revised is False and res.document == _REAL


def test_revise_prompt_is_grounded_only_anti_fabrication():
    """The revise turn mandates ground-or-remove and FORBIDS inventing a fact/figure/
    citation to keep a claim (d60 no-fabrication)."""
    captured: list[str] = []

    async def _fake(prompt: str) -> str:
        captured.append(prompt)
        if "REPORT TO CORRECT" in prompt:
            return _REAL
        return ('{"verdict":"revise","unbacked":[{"claim":"17 USC 107(5)","reason":"x"}]}'
                if "17 USC 107(5)" in prompt else '{"verdict":"ok"}')

    _run(verify_and_revise(_REAL + _FAB, _SOURCES, verify=_fake, max_passes=2))
    revise_prompt = next(p for p in captured if "REPORT TO CORRECT" in p)
    assert "may NOT invent" in revise_prompt
    assert "REMOVING it" in revise_prompt and "GROUNDING it" in revise_prompt


# --------------------------------------------------------------------------- #
# INTEGRATION — the SYNTHESIS deliverable verify lane (runtime seam)
# --------------------------------------------------------------------------- #
def _hook(tmp_path) -> ToolHook:
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    return hook


_VERIFY_SOURCES = [
    {"title": "UN News", "url": "https://news.un.org/iran",
     "source_trust": "secondary", "markdown": "Iran fired 180 missiles on June 14."},
]
# A real-content deliverable PLUS a fabricated legal claim no source backs (the B2 gap).
_REPORT_REAL = (
    "# Iran-Israel Escalation\n\n"
    "On June 14 Iran fired 180 missiles, the UN reported. Twelve people were killed in "
    "the exchange. Damage assessments continue across several provinces, with officials "
    "warning of further escalation in the days ahead.\n\n"
)
_REPORT_FAB = "Legally, under 17 USC 107(5) enacted by the CTEA of 1998, the strike was deemed lawful.\n"


def _synth_dag() -> PlanDAG:
    return PlanDAG(
        nodes=[PlanNode(id="s1", task="Write a report on the Iran escalation to iran.md.",
                        role="synthesizer")],
        goal="Write a report on the Iran escalation.",
    )


def _synth_reply():
    """A reply fn: write the report (real + fabricated), then serve verify/revise turns.

    The verify turn flags the fabrication WHILE it is present and passes once removed;
    the revise turn returns the real content with the fabrication dropped."""
    def reply(messages, **opts):
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if "REPORT TO CORRECT" in user:
            return _REPORT_REAL                 # ground-or-remove → fabrication gone
        if "REPORT TO FACT-CHECK" in user:
            if "17 USC 107(5)" in user:
                return ('{"verdict":"revise","unbacked":[{"claim":"17 USC 107(5)",'
                        '"reason":"no fetched source backs this statute"}]}')
            return '{"verdict":"ok"}'
        # the write loop: one-shot the whole report, then confirm DONE
        n = sum(1 for m in messages if m.get("role") == "assistant")
        return (_REPORT_REAL + _REPORT_FAB) if n == 0 else DONE_SENTINEL
    return reply


def test_synthesis_verify_lane_on_grounds_or_removes_fabrication(tmp_path):
    """verify_lane ON: the written deliverable's fabricated claim is flagged, a revise
    turn removes it, and the CORRECTED file is re-persisted — the real content survives."""
    transport = FakeTransport([_synth_reply()])
    rt = AgentRuntime(
        transport=transport, hook=_hook(tmp_path),
        subagent_call_opts={"think": True, "temperature": 0},
        verify_lane=True,
    )
    rt.chain_sources = _VERIFY_SOURCES
    out = _run(rt.run(_synth_dag()))

    assert out.ok
    doc = out.results["s1"].output or ""
    # the fabrication is gone, the real content kept, and the REAL FILE matches
    assert "17 USC 107(5)" not in doc and "180 missiles" in doc
    on_disk = (tmp_path / "iran.md").read_text(encoding="utf-8")
    assert on_disk == doc and "CTEA" not in on_disk


def test_synthesis_verify_lane_off_by_default_is_byte_identical(tmp_path):
    """Default (verify_lane=False): NO verify turn fires and the deliverable ships AS
    WRITTEN (fabrication included) — true no-regression for the standing c13 write side."""
    seen_users: list[str] = []

    def reply(messages, **opts):
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        seen_users.append(user)
        n = sum(1 for m in messages if m.get("role") == "assistant")
        return (_REPORT_REAL + _REPORT_FAB) if n == 0 else DONE_SENTINEL

    transport = FakeTransport([reply])
    rt = AgentRuntime(
        transport=transport, hook=_hook(tmp_path),
        subagent_call_opts={"think": True, "temperature": 0},
    )
    rt.chain_sources = _VERIFY_SOURCES
    out = _run(rt.run(_synth_dag()))

    assert out.ok
    doc = out.results["s1"].output or ""
    assert "17 USC 107(5)" in doc  # shipped as written — no verify lane touched it
    assert not any("FACT-CHECK" in u for u in seen_users)  # the verify turn never fired


# --------------------------------------------------------------------------- #
# INTEGRATION — the RESEARCH gather-more (answered-from-memory) lane
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

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content

    def chat(self, messages, **opts) -> ChatResult:
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
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


def test_research_verify_lane_on_forces_gather_more(tmp_path):
    """verify_lane ON: a research stage that answers FROM MEMORY (0 fetches) is forced to
    GATHER-MORE — it then searches + reads a real source before its findings are accepted."""
    hook = _FakeHook("https://news.un.org/iran")
    transport = _MemoryThenGatherTransport()
    agent = SubAgent(
        _research_node(), transport=transport, hook=hook,
        read_search_max_fetch=5, verify_lane=True,
        call_opts={"think": False, "temperature": 0},
    )
    res = _run(agent.run({}))

    # the memory answer was rejected; the stage actually fetched a real source
    assert hook.fetches == ["https://news.un.org/iran"]
    assert res.tool_value is not None and res.tool_value["fetched_count"] == 1
    assert "180 missiles" in (res.output or "")  # the grounded findings, not the memory one


def test_research_verify_lane_off_accepts_memory_answer(tmp_path):
    """Default (verify_lane=False): the byte-identical path — a memory answer with 0
    fetches is accepted immediately, no gather-more nudge, no fetch."""
    hook = _FakeHook("https://news.un.org/iran")
    transport = _MemoryThenGatherTransport()
    agent = SubAgent(
        _research_node(), transport=transport, hook=hook,
        read_search_max_fetch=5,
        call_opts={"think": False, "temperature": 0},
    )
    res = _run(agent.run({}))

    assert hook.fetches == []           # never fetched — the memory answer stood
    assert res.tool_value is None        # no sources gathered
    assert "escalating tensions" in (res.output or "")  # the from-memory findings
