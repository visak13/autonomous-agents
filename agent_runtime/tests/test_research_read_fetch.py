"""s9/c5 (d49/d50): RESEARCH IS A TRUE AGENTIC LOOP — the worker DECIDES to fetch.

The deterministic search-then-read EXECUTOR (flags #1/#3: ``read_search_max_fetch``
> 0 forcing a ``web_search`` node to auto-``web_fetch`` its top hits) is RETIRED. A
``web_search`` node is now a TRUE AGENT (:meth:`SubAgent._run_research_loop`): it
emits lightweight tool calls (``web_search``/``web_fetch``) to gather REAL evidence,
the loop executes each against the real hook and feeds the observation back, and the
model writes its FINDINGS as RAW prose (no ``format=<schema>``, content RAW). The
``web_fetch`` calls are the MODEL's choices, not a deterministic top-k follow-through;
``read_search_max_fetch`` survives only as a NON-FLOW fetch CAP.

Fully OFFLINE: a fake hook records every web_search/web_fetch call and returns canned
article markdown; a SCRIPTED transport replays a sequence of agent turns (tool-call
JSON, then findings prose). No GPU, no network.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from llm_framework import ChatResult
from llm_framework.context import SUMMARY_HEADER

from agent_runtime.factory import PlanNode
from agent_runtime.runtime import SubAgent

# Real-shaped article URLs the fake search returns; the fetcher gives each a distinct
# headline so we can prove WHICH ones the agent actually chose to read.
_URLS = [
    "https://news.example.com/world",
    "https://news.example.com/tech",
    "https://news.example.com/biz",
]
_HEADLINES = {
    _URLS[0]: "REAL HEADLINE WORLD: summit reaches accord",
    _URLS[1]: "REAL HEADLINE TECH: chip output up 12 percent",
    _URLS[2]: "REAL HEADLINE BIZ: index closes at record",
}
# A .pdf the search ALSO surfaces — so the not-a-readable-article filter is exercised on a
# url that IS in the offered set (s15/a25: the offered-URL grounding guard would otherwise
# reject any un-offered url before the article-type filter runs).
_PDF_URL = "https://news.example.com/report.pdf"


@dataclass
class _ToolResult:
    ok: bool
    value: Any = None
    error: str = ""
    call_id: str = "c1"


class _FakeHook:
    """Records web_search / web_fetch invocations; returns canned real-shaped data."""

    def __init__(self) -> None:
        self.searches: list[str] = []
        self.fetches: list[str] = []

    async def invoke(self, name: str, **args) -> _ToolResult:
        if name == "web_search":
            self.searches.append(args.get("query", ""))
            rows = [
                {"title": f"t{i}", "url": u, "snippet": "snip"}
                for i, u in enumerate(_URLS)
            ]
            rows.append({"title": "tpdf", "url": _PDF_URL, "snippet": "snip"})
            return _ToolResult(True, {
                "query": args.get("query", ""),
                "results": rows,
                "count": len(rows),
            })
        if name == "web_fetch":
            url = args.get("url", "")
            self.fetches.append(url)
            return _ToolResult(True, {
                "url": url, "final_url": url, "status": 200,
                "title": url.rsplit("/", 1)[-1],
                "markdown": f"{_HEADLINES.get(url, 'article')}\n\nbody text for {url}.",
                "extracted": True,
            })
        return _ToolResult(False, error=f"unknown tool {name}")


# d242 TRUE self-select: a research node starts TOOL-LESS and MUST load the 'research'
# bundle before it can search/fetch. Every scripted transport therefore SELF-SELECTS first
# (the model's opening move), then runs its gather script. This models the real loop and is
# what keeps these MECHANISM tests exercising the live path; the self-select DECISION itself
# is asserted directly in test_research_loop_self_selects_before_gathering.
_LOAD_RESEARCH = '{"tool": "get_bundles", "args": {"name": "research"}}'


class _ScriptedTransport:
    """Replays a fixed sequence of agent turns (one per chat call) + records the
    user turns the agent saw (so we can assert observations were fed back).

    ``self_select`` (default ``"research"``) PREPENDS a get_bundles load turn so the node
    self-selects its gather tools before the script's first gather call (d242); pass
    ``None`` for a script that drives the self-select turns itself."""

    def __init__(self, turns: list[str], *, self_select: str | None = "research") -> None:
        lead = [f'{{"tool": "get_bundles", "args": {{"name": "{self_select}"}}}}'] if self_select else []
        self._turns = lead + list(turns)
        self.calls: list[str] = []  # the user turn of each chat call

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content

    def chat(self, messages, **opts) -> ChatResult:
        # The freshest fed-back turn the agent must react to — a tool RESULT (role 'tool',
        # s15/a18 d189) or a genuine instruction (role 'user'). Scan BOTH so a scripted
        # branch keyed on the observation still sees it now that results ride role 'tool'.
        user = next(
            (m["content"] for m in reversed(messages)
             if m.get("role") in ("user", "tool")), ""
        )
        i = len(self.calls)
        self.calls.append(user)
        content = self._turns[i] if i < len(self._turns) else "FALLBACK FINDINGS."
        return ChatResult(role="assistant", content=content)


def _ws_node(nid: str = "r1_research", query: str = "morning world news") -> PlanNode:
    # d48: a research-POSITION node is a WORKER carrying the web_search tool.
    return PlanNode(id=nid, task="[research] morning world news briefing",
                    role="worker", tool="web_search", tool_args={"query": query})


def _full_convo(transport: _ScriptedTransport) -> str:
    return "\n\n".join(transport.calls)


def test_worker_decides_to_search_then_fetch_chosen_urls_and_fold_text():
    """The agent emits web_search → fetches the URLs IT chose → writes findings; the
    fetched article text is fed back, and the read sources land on tool_value (d17)."""
    hook = _FakeHook()
    transport = _ScriptedTransport([
        '{"tool": "web_search", "args": {"query": "world summit accord"}}',
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/world"}}',
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/tech"}}',
        "FINDINGS: a summit reached an accord (https://news.example.com/world) and "
        "chip output rose (https://news.example.com/tech).",
    ])
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=5, call_opts={"think": False, "temperature": 0},
    )
    res = asyncio.run(agent.run({}))

    # the MODEL's chosen query + URLs were executed (not a deterministic top-k)
    assert hook.searches == ["world summit accord"]
    assert hook.fetches == [_URLS[0], _URLS[1]]
    # the EXTRACTED article text was fed back to the agent (read-not-describe)
    convo = _full_convo(transport)
    assert "FETCHED" in convo
    assert _HEADLINES[_URLS[0]] in convo and _HEADLINES[_URLS[1]] in convo
    # findings prose is the RAW node output
    assert "summit reached an accord" in (res.output or "")
    # the read sources are attached for downstream grounding (d17)
    assert res.tool_value["fetched_count"] == 2
    assert {s["url"] for s in res.tool_value["fetched"]} == {_URLS[0], _URLS[1]}


def test_fetch_cap_is_a_non_flow_bound():
    """``read_search_max_fetch`` is a CAP, not a gate: the agent may try more fetches
    but only ``cap`` web_fetch calls are executed, then it is told to write findings."""
    hook = _FakeHook()
    transport = _ScriptedTransport([
        '{"tool": "web_search", "args": {"query": "news"}}',
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/world"}}',
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/tech"}}',  # over cap
        "FINDINGS from the one source I read.",
    ])
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=1, call_opts={"think": False},
    )
    res = asyncio.run(agent.run({}))
    assert hook.fetches == [_URLS[0]]  # the 2nd fetch was capped, never invoked
    assert "Fetch limit (1) reached" in _full_convo(transport)
    assert res.tool_value["fetched_count"] == 1


def test_non_article_url_is_rejected_before_fetch():
    """A model-chosen PDF/binary URL is filtered BEFORE the fetch (Trafilatura is
    HTML-only); the agent is told to pick another source — no wasted/garbage fetch."""
    hook = _FakeHook()
    transport = _ScriptedTransport([
        '{"tool": "web_search", "args": {"query": "report"}}',
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/report.pdf"}}',
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/world"}}',
        "FINDINGS from the readable article.",
    ])
    agent = SubAgent(_ws_node(), transport=transport, hook=hook,
                     read_search_max_fetch=5, call_opts={"think": False})
    asyncio.run(agent.run({}))
    # the .pdf URL was NEVER fetched; only the HTML article was
    assert hook.fetches == [_URLS[0]]
    assert "not a readable HTML article" in _full_convo(transport)


def test_worker_may_decide_not_to_fetch():
    """The worker's reasoning is honored: it may search and then answer WITHOUT
    fetching (no forced follow-through). The choice is the model's, not a flag."""
    hook = _FakeHook()
    transport = _ScriptedTransport([
        '{"tool": "web_search", "args": {"query": "quick facts"}}',
        "FINDINGS: a concise answer from the search snippets alone.",
    ])
    agent = SubAgent(_ws_node(), transport=transport, hook=hook,
                     read_search_max_fetch=5, call_opts={"think": False})
    res = asyncio.run(agent.run({}))
    assert hook.searches == ["quick facts"]
    assert hook.fetches == []  # the agent chose not to read full sources
    assert res.tool_value is None  # no sources read → nothing to attach
    assert "concise answer" in (res.output or "")


def test_findings_only_turn_ends_the_loop_with_no_tools():
    """If the very first turn is prose (no tool call), that prose IS the findings and
    the loop ends — the agent answered from its assembled context, no tool forced."""
    hook = _FakeHook()
    transport = _ScriptedTransport([
        "FINDINGS: I can answer this directly from the provided context.",
    ])
    agent = SubAgent(_ws_node(), transport=transport, hook=hook,
                     read_search_max_fetch=5, call_opts={"think": False})
    res = asyncio.run(agent.run({}))
    assert hook.searches == [] and hook.fetches == []
    assert "answer this directly" in (res.output or "")


# --------------------------------------------------------------------------- #
# s9/N1 (d60/c15 part-a) — BREADTH: the fetch cap lifts from the legacy 3 to a
# configurable ~8-12, and the ReAct TURN CEILING rises proportionally so a
# high-breadth gather can search several angles AND read MANY sources without the
# flat ceiling clipping it. Still a NON-FLOW bound: the model decides whether/which.
# --------------------------------------------------------------------------- #
class _ManyUrlHook:
    """A fake hook with MANY distinct article URLs so we can prove a single research
    node reads far more than the legacy 3-source ceiling."""

    def __init__(self, n: int) -> None:
        self.urls = [f"https://news.example.com/a{i}" for i in range(n)]
        self.searches: list[str] = []
        self.fetches: list[str] = []

    async def invoke(self, name: str, **args) -> _ToolResult:
        if name == "web_search":
            self.searches.append(args.get("query", ""))
            return _ToolResult(True, {
                "query": args.get("query", ""),
                "results": [
                    {"title": f"t{i}", "url": u, "snippet": "snip"}
                    for i, u in enumerate(self.urls)
                ],
                "count": len(self.urls),
            })
        if name == "web_fetch":
            url = args.get("url", "")
            self.fetches.append(url)
            return _ToolResult(True, {
                "url": url, "final_url": url, "status": 200,
                "title": url.rsplit("/", 1)[-1],
                "markdown": f"REAL ARTICLE {url}\n\nsubstantive body for {url}.",
                "extracted": True,
            })
        return _ToolResult(False, error=f"unknown tool {name}")


def test_breadth_turn_ceiling_rises_with_fetch_cap():
    """With a BREADTH cap (11), the agent reads ELEVEN real sources in one node — far
    past the legacy 3 — across TWO search angles. This run needs 14 turns (search,
    fetch×5, search, fetch×6, findings); under the OLD flat 12-turn ceiling the 11th
    fetch and the in-loop findings turn would be clipped, so proving all 11 URLs were
    fetched AND the in-loop findings were captured proves the ceiling scaled with the
    cap. The cap is still NON-FLOW — the model chose to search/fetch; nothing forced it."""
    hook = _ManyUrlHook(11)
    u = hook.urls
    turns = (
        ['{"tool": "web_search", "args": {"query": "angle one"}}']
        + [f'{{"tool": "web_fetch", "args": {{"url": "{u[i]}"}}}}' for i in range(5)]
        + ['{"tool": "web_search", "args": {"query": "angle two"}}']
        + [f'{{"tool": "web_fetch", "args": {{"url": "{u[i]}"}}}}' for i in range(5, 11)]
        + ["BREADTH FINDINGS: eleven sources synthesised, each attributed."]
    )
    assert len(turns) == 14  # genuinely exceeds the legacy RESEARCH_MAX_TURNS=12
    transport = _ScriptedTransport(turns)
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=11, call_opts={"think": False, "temperature": 0},
    )
    res = asyncio.run(agent.run({}))

    # All ELEVEN model-chosen sources were actually fetched (breadth, not capped at 3).
    assert hook.fetches == u
    assert res.tool_value["fetched_count"] == 11
    # The in-loop findings turn (turn 14) ran — proving the ceiling scaled past 12, not
    # the post-loop fallback salvage (which would have emitted a fetch-call JSON string).
    assert "BREADTH FINDINGS" in (res.output or "")


def test_narrow_cap_keeps_the_legacy_turn_floor():
    """A NARROW cap (≤5) leaves the turn ceiling at the original RESEARCH_MAX_TURNS
    floor — narrow/inline paths are unchanged by the breadth scaling (no regression)."""
    from agent_runtime.runtime import RESEARCH_MAX_TURNS, RESEARCH_SEARCH_HEADROOM

    # For any legacy cap the proportional ceiling stays clamped to the floor.
    for cap in (1, 3, 5):
        assert max(RESEARCH_MAX_TURNS, cap + RESEARCH_SEARCH_HEADROOM) == RESEARCH_MAX_TURNS
    # A breadth cap genuinely raises it.
    assert max(RESEARCH_MAX_TURNS, 10 + RESEARCH_SEARCH_HEADROOM) > RESEARCH_MAX_TURNS


# --------------------------------------------------------------------------- #
# s15/a25 (d186 / d199) — OFFERED-URL GROUNDING. The live failure: with a18 (d189)
# feeding the search/fetch observations on role:'tool', and live Gemma using a
# '{{ .Prompt }}' chat template with NO role handling, the model IGNORED the role:'tool'
# SEARCH RESULTS and FABRICATED plausible-but-dead fetch URLs (or copied the instruction
# placeholder) so EVERY fetch failed and 0 sources/notes were captured. d199 (supersedes
# d189 for this model) feeds observations the model must act on back on role:'user' so it
# grounds; a defense-in-depth guard also validates the chosen fetch url against the URLs
# web_search actually offered and re-prompts on a fabricated/placeholder url. These tests
# exercise the REAL offered-URL flow so a fabricated/placeholder url FAILS the test (the
# regression a24's scripted, hardcoded-real-URL test could not catch).
# --------------------------------------------------------------------------- #
def test_fabricated_or_placeholder_fetch_url_is_rejected_and_regrounds():
    """A web_fetch url that is NOT one web_search offered (a fabricated domain, or the
    instruction's '<a result URL...>' placeholder) is rejected with a role:tool error that
    lists the REAL candidates; the model RE-GROUNDS on an offered url, which LANDS. The
    fabricated/placeholder urls are NEVER sent to the fetch tool, and they do NOT burn the
    fetch cap (the real, grounded fetch still runs)."""
    hook = _FakeHook()
    transport = _ScriptedTransport([
        '{"tool": "web_search", "args": {"query": "world news"}}',
        # the instruction PLACEHOLDER copied verbatim (the live 'copied the example' failure)
        '{"tool": "web_fetch", "args": {"url": "<a result URL to read in full>"}}',
        # a plausible-but-dead INVENTED url (the live 'cannot resolve host' failure)
        '{"tool": "web_fetch", "args": {"url": "https://www.invented-thinktank.org/made-up"}}',
        # finally a REAL offered url, copied verbatim
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/world"}}',
        "FINDINGS grounded in the one real source I read.",
    ])
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=5, call_opts={"think": False, "temperature": 0},
    )
    res = asyncio.run(agent.run({}))

    # NEITHER ungrounded url reached the fetch tool; only the offered one did.
    assert hook.fetches == [_URLS[0]]
    convo = _full_convo(transport)
    # the role:tool error fired for the ungrounded attempts and surfaced the real candidates
    assert "was NOT in the search results" in convo
    assert _URLS[0] in convo
    # capture LANDS once the model grounds on a real url (the d186 fix's whole point)
    assert res.tool_value["fetched_count"] == 1
    assert {s["url"] for s in res.tool_value["fetched"]} == {_URLS[0]}


def test_gather_lands_sources_and_notes():
    """d186 anti-regression (LIVE capture): after a gather, the read source AND the recorded
    article note LAND on tool_value (>0) — the exact leaf state the ResearchState folds into
    its source index + note lane. Proven by asserting capture LANDS, not merely that a clean
    final emitted (a14/a23 sources=notes=0 came from this lane being empty while the final
    still looked fine)."""
    hook = _FakeHook()
    transport = _ScriptedTransport([
        '{"tool": "web_search", "args": {"query": "world summit"}}',
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/world"}}',
        '{"tool": "note", "args": {"url": "https://news.example.com/world", '
        '"summary": "a summit reached an accord", '
        '"key_claims": ["the summit reached an accord"], '
        '"gaps_or_followups": ["economic terms not yet confirmed"]}}',
        "FINDINGS: a summit reached an accord (https://news.example.com/world).",
    ])
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=5, emit_article_notes=True,
        call_opts={"think": False, "temperature": 0},
    )
    res = asyncio.run(agent.run({}))

    # the read SOURCE landed (>0) — the verbatim source index / sidecar is non-empty
    assert res.tool_value["fetched_count"] == 1
    assert res.tool_value["fetched"][0]["url"] == _URLS[0]
    # the article NOTE landed (>0) — the gap lane the decision node reasons over
    assert res.tool_value["article_notes"]
    assert res.tool_value["article_notes"][0]["gaps_or_followups"] == [
        "economic terms not yet confirmed"
    ]


# --------------------------------------------------------------------------- #
# s15/a27 (d199 follow-on) — the NOTE GATE. The a25/d199 role:'user' fetch body +
# _FETCH_NOTE_CHAIN made noting POSSIBLE, but live Gemma still fetched ONE source and
# jumped STRAIGHT to findings, skipping the (optional) note (measured trace 1427f176:
# every single-fetch research node emitted notes=0 → empty write-from-notes substrate).
# The gate forces a note to LAND for each fetched-but-un-noted source before findings are
# accepted, bounded so an unwilling model cannot loop and never discarding real findings.
# --------------------------------------------------------------------------- #
def test_note_gate_forces_a_note_for_an_un_noted_fetched_source():
    """The model fetches a source then jumps STRAIGHT to findings without noting it. The
    NOTE GATE pushes it back to record THAT source's note first; once the note lands the
    findings are accepted. Proves a structured note LANDS per fetched source (>0), not the
    notes=0 the live trace showed."""
    hook = _FakeHook()
    transport = _ScriptedTransport([
        '{"tool": "web_search", "args": {"query": "world summit"}}',
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/world"}}',
        # the live failure: findings WITHOUT a note for the source just read
        "FINDINGS: a summit reached an accord (https://news.example.com/world).",
        # the gate pushed back → the model now records the missing note
        '{"tool": "note", "args": {"url": "https://news.example.com/world", '
        '"summary": "a summit reached an accord", '
        '"key_claims": ["the summit reached an accord"], '
        '"gaps_or_followups": ["economic terms not yet confirmed"]}}',
        # now every read source has a note → findings are accepted
        "FINDINGS: a summit reached an accord (https://news.example.com/world).",
    ])
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=5, emit_article_notes=True,
        call_opts={"think": False, "temperature": 0},
    )
    res = asyncio.run(agent.run({}))

    convo = _full_convo(transport)
    # the gate fired: the model was told its read source was still un-noted (role:'user')
    assert "have NOT recorded its note yet" in convo
    # the note LANDED (the whole point) — the gap lane / write substrate is non-empty
    assert res.tool_value["fetched_count"] == 1
    assert res.tool_value["article_notes"]
    assert res.tool_value["article_notes"][0]["gaps_or_followups"] == [
        "economic terms not yet confirmed"
    ]
    # the findings still stand (the gate adds a note turn, it does not drop the answer)
    assert "summit reached an accord" in (res.output or "")


def test_note_gate_is_bounded_and_never_discards_findings():
    """A model that REFUSES to note (keeps emitting findings) is gated only a BOUNDED number
    of times, then its findings are accepted from the salvaged emission — the gate can never
    loop forever nor cost the model its real findings. With fetch_cap=1 the bound is
    max(2, 1) == 2, so the loop terminates well inside the turn ceiling."""
    hook = _FakeHook()
    transport = _ScriptedTransport([
        '{"tool": "web_search", "args": {"query": "world summit"}}',
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/world"}}',
        # the model keeps answering without ever noting; the gate fires twice then yields
        "FINDINGS attempt one, no note.",
        "FINDINGS attempt two, still no note.",
        "FINDINGS attempt three, accepted.",
        # extra turns must NOT be needed — if the loop asks again the script falls back
    ])
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=1, emit_article_notes=True,
        call_opts={"think": False, "temperature": 0},
    )
    res = asyncio.run(agent.run({}))

    convo = _full_convo(transport)
    # the gate DID fire (bounded), surfacing the un-noted source
    assert "have NOT recorded its note yet" in convo
    # it terminated with the model's own findings, not the FALLBACK salvage or an empty answer
    assert "FINDINGS attempt" in (res.output or "")
    assert "FALLBACK FINDINGS" not in (res.output or "")
    # no note was ever recorded (the model refused) → the notes lane is simply empty, not looping
    assert res.tool_value["fetched_count"] == 1
    assert "article_notes" not in res.tool_value


# --------------------------------------------------------------------------- #
# s16/SA-3 (d263, refines d229) — PINNED-REBROADCAST REMOVAL. The failed pinned-head +
# SWA-tail re-injection (which re-pasted the goal + bundle doctrine + task as always-in-
# view blocks every turn) is REMOVED. The research DOCTRINE now rides the ``get_bundles``
# LOAD observation EXACTLY ONCE (delivered in-band when the node self-selects the bundle,
# carried forward by the convo window), is NEVER re-pasted into the per-turn task, and
# there is NO pinned-head / SWA-tail block anywhere. The goal rides convo[0] ONCE and the
# SHAPING system rides ``Context(system=…)`` ONCE — none re-pasted per turn. Proven on a
# REAL prompt build, with the research loop still firing search → fetch → findings.
# --------------------------------------------------------------------------- #
class _CapturingTransport(_ScriptedTransport):
    """A scripted transport that ALSO records the FULL message list of every chat call,
    so a test can inspect the real assembled prompt (system + turns)."""

    def __init__(self, turns: list[str]) -> None:
        super().__init__(turns)
        self.message_sets: list[list[dict]] = []

    def chat(self, messages, **opts) -> ChatResult:
        self.message_sets.append([dict(m) for m in messages])
        return super().chat(messages, **opts)


# A phrase that lives ONLY in the full research loop instruction (RESEARCH_LOOP_INSTRUCTION),
# never in the bundle's short catalog summary — so counting it across the whole prompt counts
# the DOCTRINE itself.
_DOCTRINE_PHRASE = "Workflow: search the topic"

# Retired markers (d263): no pinned head, no SWA tail, ever again.
_RETIRED_MARKERS = ("[Pinned context", "[Active context", "never lose sight", "REMINDER — stay")


def test_research_doctrine_rides_load_obs_once_no_pin_no_repaste():
    """d263: the research doctrine rides the get_bundles LOAD observation ONCE; no pin/SWA.

    Build a REAL research prompt and assert the doctrine phrase occurs EXACTLY ONCE across
    the whole assembled prompt, lives in the get_bundles LOAD observation (a role:tool turn,
    NOT a pinned-head system block — which no longer exists), appears in NO user turn, and
    that the per-turn task still carries the per-RUN operational fetch cap — all while the
    research loop still fires (search → fetch → findings)."""
    hook = _FakeHook()
    transport = _CapturingTransport([
        '{"tool": "web_search", "args": {"query": "world summit accord"}}',
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/world"}}',
        "FINDINGS: a summit reached an accord (https://news.example.com/world).",
    ])
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=5, call_opts={"think": False, "temperature": 0},
    )
    res = asyncio.run(agent.run({}))

    # the research loop STILL FIRES (search → fetch → findings), unchanged by the removal
    assert hook.searches == ["world summit accord"]
    assert hook.fetches == [_URLS[0]]
    assert "summit reached an accord" in (res.output or "")

    # Inspect the prompt build AFTER the node loads the 'research' bundle (the
    # _ScriptedTransport prepends the get_bundles load turn, so message_sets[1] is the first
    # build that carries the load observation).
    first = transport.message_sets[1]
    whole = "\n\n".join(str(m.get("content", "")) for m in first)
    # the doctrine is present EXACTLY ONCE across the entire assembled prompt
    assert whole.count(_DOCTRINE_PHRASE) == 1, (
        f"doctrine must appear exactly once; found {whole.count(_DOCTRINE_PHRASE)}"
    )
    # it rides the get_bundles LOAD observation (the "Loaded bundle '<name>'" turn), the
    # single in-band source of truth — NOT a pinned-head system block (the d263 removal).
    # (SA-2's transport role-normalization rewrites the tool function-result role to 'user'
    # at the wire, so the observation is identified by its content, not its role.)
    load_obs = [m for m in first if str(m.get("content", "")).startswith("Loaded bundle ")]
    assert load_obs, "expected a get_bundles load observation turn"
    assert _DOCTRINE_PHRASE in str(load_obs[0]["content"]), (
        "the doctrine must ride the get_bundles load observation"
    )
    # NO retired pinned-head / SWA-tail marker, and the doctrine is in NO system turn
    for marker in _RETIRED_MARKERS:
        assert marker not in whole, f"retired pin/SWA marker {marker!r} must be gone (d263)"
    system_turns = [m for m in first if m.get("role") == "system"]
    assert all(
        _DOCTRINE_PHRASE not in str(m.get("content", "")) for m in system_turns
    ), "the doctrine must NOT live in a (pinned-head) system turn anymore"
    # and the doctrine is NOT re-pasted into the per-turn TASK message (convo[0])
    task_turn = first[1]  # [0]=system, [1]=the _compose_task user turn
    assert _DOCTRINE_PHRASE not in str(task_turn.get("content", "")), (
        "the per-turn task message must NOT re-paste the research doctrine"
    )
    # the per-turn task DID keep the per-RUN operational fetch-cap bound (not the doctrine)
    assert "up to 5 of the source URLs" in str(task_turn.get("content", "")), (
        "the concrete per-run fetch cap should still reach the model on the task turn"
    )


def test_system_composed_once_goal_sent_once_across_turns():
    """d263 GATE: across the N research calls the SHAPING system turn is byte-IDENTICAL
    (composed once, carried — never re-composed/re-grown) and the OVERALL GOAL is sent
    EXACTLY ONCE per call (it rides convo[0] only — not also re-pasted as a pinned head and
    an SWA-tail reminder the way the retired mechanism did, which put the goal in the prompt
    THREE times every turn)."""
    hook = _FakeHook()
    transport = _CapturingTransport([
        '{"tool": "web_search", "args": {"query": "world summit accord"}}',
        '{"tool": "web_fetch", "args": {"url": "https://news.example.com/world"}}',
        "FINDINGS: a summit reached an accord (https://news.example.com/world).",
    ])
    goal = "GOALSENTINEL-write the morning world-news brief"
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook, overall_goal=goal,
        read_search_max_fetch=5, call_opts={"think": False, "temperature": 0},
    )
    asyncio.run(agent.run({}))

    # the gather builds (after the node self-selects 'research'): every one carries the same
    # system, and the goal appears exactly once in each.
    builds = transport.message_sets[1:]
    assert len(builds) >= 2, "expected multiple gather turns to compare"
    systems = [
        next((str(m["content"]) for m in b if m.get("role") == "system"), None)
        for b in builds
    ]
    assert all(s is not None for s in systems), "every build carries a system turn"
    assert len(set(systems)) == 1, (
        "the SHAPING system must be byte-identical across turns (composed once, carried), "
        "not re-composed/re-grown per call"
    )
    for b in builds:
        whole = "\n\n".join(str(m.get("content", "")) for m in b)
        assert whole.count(goal) == 1, (
            f"the overall goal must appear exactly once per call (rides convo[0] only); "
            f"found {whole.count(goal)} — a >1 count is the retired pin+SWA triple-injection"
        )


def test_node_history_compacts_middle_keeps_goal_turn():
    """d263 RETENTION: long-chat compaction still fires under the raised num_ctx (the simple
    middle-turn fold is KEPT), and it NEVER drops the goal — convo[0] (the goal/task turn) is
    retained verbatim while only the MIDDLE turns fold into an offline summary."""
    agent = SubAgent(
        _ws_node(), transport=_ScriptedTransport(["x"]),
        call_opts={"num_ctx": 600},  # small window so a long convo crosses the threshold
    )
    goal_turn = {"role": "user", "content": "GOAL-TASK TURN: " + "g" * 400}
    middle = [
        {"role": "assistant", "content": f"assistant turn {i} " + "m" * 400}
        for i in range(8)
    ]
    recent = [
        {"role": "tool", "content": f"RECENT-OBS-{i} " + "r" * 50} for i in range(3)
    ]
    convo = [goal_turn] + middle + recent

    out = agent._node_history(convo, system="SYS", compact=True, keep_recent=3)

    # the goal/task turn is retained VERBATIM as the head
    assert out[0] == goal_turn, "convo[0] (the goal/task turn) must be kept verbatim"
    # compaction fired: a running summary turn was injected and the window shrank
    assert any(
        m.get("role") == "system" and SUMMARY_HEADER in str(m.get("content", ""))
        for m in out
    ), "long-chat compaction must still fire (a summary turn appears)"
    assert len(out) < len(convo), "compaction must shrink the window"
    # the most-recent turns survive verbatim (not folded)
    assert out[-3:] == recent, "the most-recent keep_recent turns are preserved verbatim"


def test_node_history_short_convo_is_passthrough():
    """A convo with no compactible middle (goal turn + a few recent) is byte-identical."""
    agent = SubAgent(
        _ws_node(), transport=_ScriptedTransport(["x"]), call_opts={"num_ctx": 600},
    )
    convo = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "obs"},
    ]
    assert agent._node_history(convo, system="SYS", compact=True, keep_recent=6) == convo
    # compact=False is always a passthrough
    assert agent._node_history(convo, system="SYS", compact=False) == convo


def test_research_loop_self_selects_before_gathering():
    """d242 TRUE self-select: a research node starts TOOL-LESS — it CANNOT search until it
    LOADS the 'research' bundle. A script that tries to search before loading is nudged (no
    search fires); once it loads, the gather tools become callable and the loop runs."""
    # (A) tries to web_search on turn 1 WITHOUT loading → blocked, then loads + searches.
    hook = _FakeHook()
    transport = _ScriptedTransport(
        [
            '{"tool": "web_search", "args": {"query": "too early"}}',   # before any load → nudged
            '{"tool": "get_bundles", "args": {"name": "research"}}',    # NOW self-select
            '{"tool": "web_search", "args": {"query": "world summit accord"}}',
            "FINDINGS: a summit reached an accord.",
        ],
        self_select=None,  # this script drives its own (mis)ordered self-select
    )
    agent = SubAgent(
        _ws_node(), transport=transport, hook=hook,
        read_search_max_fetch=5, call_opts={"think": False, "temperature": 0},
    )
    res = asyncio.run(agent.run({}))
    # the premature search NEVER reached the hook; only the post-load one did.
    assert hook.searches == ["world summit accord"]
    convo = _full_convo(transport)
    assert "have not loaded the tool 'web_search'" in convo  # the self-select nudge fired
    assert "summit reached an accord" in (res.output or "")
