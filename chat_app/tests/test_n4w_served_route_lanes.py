"""s9/N4w (d65) → P2-5c — the note + chunked-read grounding lanes FIRE on the SERVED route.

review-knob-on-served-route discipline: it is NOT enough that the lane flags are wired at a
construction site — the lanes must actually FIRE on the request the served deep-research route
takes. After P2-5c the served report route (``run_plan_chain`` /
``_run_deep_research_sectioned``) gathers research through the GENERIC growable engine: each
research node runs on the same :class:`~agent_runtime.AgentRuntime` the report route builds via
``_build_acyclic_runtime(emit_article_notes=True, chunked_read=True,
research_fetch_breadth=PLAN_CHAIN_TREE_BREADTH)`` (the bespoke ``_make_tree_gather`` leaf is
retired; RP-3c/d330 retired the ``verify_lane`` flag — the no-fab self-review moved to the
writer doctrine and the gather-more gate is de-flagged). These tests drive that EXACT served
research config through the REAL AgentRuntime +
the REAL :class:`~reactive_tools.ToolHook` seam (web tools on a real ToolRegistry) with a
scripted transport, and assert BOTH lanes light up:

* emit_article_notes — the research node records a per-article CONTROL note (the gap signal the
  grower's decision node reasons over), surfacing on the node's ``tool_value.article_notes``;
* chunked_read read lane — a LONG fetched source is read via the d109 RELEVANCE-SELECT read
  (paragraph chunks ranked by the MiniLM embedder against the sub-question, top passages
  assembled for a single in-window read), NOT flat-truncated and NOT the old map/reduce
  summarizer (which d109 keeps only as the embedder-unavailable / overflow fallback).

A CONTROL test proves the same runtime with the lanes UNWIRED fires NEITHER lane (so the proof
measures the WIRING, not an always-on effect). The full LIVE E4B chat-request proof is N4r.

Test-only knob (``test_generic_research_lane_config``) mirrors the served lane config so a future
change to the report-route grounding lanes that DROPS one is caught here, not only live.
"""
from __future__ import annotations

import asyncio

from llm_framework import ChatResult
from reactive_tools import EventPlane, ToolHook
from reactive_tools.tool_hook import ToolRegistry

from agent_runtime import AgentRuntime, ExecutionMode
from agent_runtime.factory import PlanDAG, PlanNode


# A source LONGER than the per-source fetch budget (2000) so the long-source read engages.
_LONG_SOURCE = ("The 2025 conflict escalated on June 13. " * 120) + (
    "\n\nCasualty figures reached 1,200 by June 20 per the ministry. " * 60
)
_URL = "https://news.example.com/iran-2025"


def _web_seam():
    """A real ToolRegistry/ToolHook serving one search hit + a LONG article body."""
    registry = ToolRegistry()
    calls: dict[str, list] = {"searches": [], "fetches": []}

    def web_search(**args):
        calls["searches"].append(args.get("query", ""))
        return {"query": args.get("query", ""),
                "results": [{"title": "Iran 2025", "url": _URL, "snippet": "s"}],
                "count": 1}

    def web_fetch(**args):
        url = args.get("url", "")
        calls["fetches"].append(url)
        return {"url": url, "final_url": url, "status": 200,
                "title": "Iran 2025", "markdown": _LONG_SOURCE, "extracted": True}

    registry.register("web_search", web_search)
    registry.register("web_fetch", web_fetch)
    plane = EventPlane()
    return ToolHook(plane, registry=registry), plane, calls


class _LaneTransport:
    """Content-dispatching transport: the (now-fallback) map/reduce summarizer turn (the
    ``FACTUAL SUMMARY:`` map prompt) is answered with a canned factual summary and RECORDED;
    every other turn replays the next scripted research turn. Both the user turns AND the
    role='tool' OBSERVATIONS are captured: s15/a25 (d199) feeds the research GATHER loop's
    observations the model must ground on back as role='user' turns (live gemma4-e4b's
    '{{ .Prompt }}' template ignores role='tool'), so the d109 relevance-select honest signal
    rides in a USER turn — a test asserts it reached the research convo there. (a18/d189's
    role='tool' still holds for the write/reviewer/planner loops, which this seam doesn't drive.)"""

    def __init__(self, research_turns: list[str]) -> None:
        # d242 TRUE self-select: the research node starts TOOL-LESS, so it loads the
        # 'research' bundle first (its opening move) before any web_search/web_fetch.
        self._turns = ['{"tool": "get_bundles", "args": {"name": "research"}}'] + list(research_turns)
        self._i = 0
        self.summarizer_calls: list[str] = []
        self.user_turns: list[str] = []
        self.tool_turns: list[str] = []

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content

    def chat(self, messages, **opts) -> ChatResult:
        user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        self.user_turns.append(user)
        tool_obs = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "tool"), None
        )
        if tool_obs is not None:
            self.tool_turns.append(tool_obs)
        if "FACTUAL SUMMARY:" in user:  # the N3 map/reduce turn (d109 fallback only)
            self.summarizer_calls.append(user)
            return ChatResult(role="assistant",
                              content="June 13 escalation; 1,200 casualties by June 20.")
        turn = self._turns[self._i] if self._i < len(self._turns) else "FALLBACK FINDINGS."
        self._i += 1
        return ChatResult(role="assistant", content=turn)


def _research_turns() -> list[str]:
    return [
        '{"tool": "web_search", "args": {"query": "2025 Iran conflict casualties"}}',
        f'{{"tool": "web_fetch", "args": {{"url": "{_URL}"}}}}',
        ('{"tool": "note", "args": {"url": "' + _URL + '", "summary": "escalation + casualties", '
         '"category": "world", "source_trust": "secondary", '
         '"key_claims": ["June 13 escalation", "1,200 casualties"], '
         '"relevance": "on topic", "gaps_or_followups": ["search damage figures"]}}'),
        f"FINDINGS: the conflict escalated June 13; 1,200 casualties by June 20 ({_URL}).",
    ]


def _served_research_node() -> PlanNode:
    """A research (web_search-seeded) node — the position the generic engine gathers."""
    return PlanNode(id="r1_research", task="2025 Iran conflict",
                    depends_on=(), tool="web_search", role="worker")


def test_served_generic_research_fires_note_and_chunked_read_lanes(monkeypatch):
    """The EXACT served research config (emit_article_notes + chunked_read + breadth, the lanes
    ``_run_generic_research_phase`` wires) lights BOTH lanes through the real runtime + ToolHook
    seam: a note is recorded (the grower's decision-node input) AND the long source is read via
    the d109 relevance-select read (the honest coverage signal reaches the convo and the
    map/reduce summarizer does NOT fire — relevance-select replaced it).

    The shared read-embedder seam is patched to a fast bag-of-words fake: the REAL
    ``CpuEmbedder`` pulls onnxruntime's native DLL, whose load was measured at ~20
    MINUTES on this host (2026-07-13) — this one test was silently paying it. The
    lane wiring under test is identical; only the vector backend is faked."""
    import numpy as _np

    class _BagEmbedder:
        _VOCAB = "casualties conflict escalated june ministry figures".split()

        def embed(self, texts):
            rows = []
            for t in texts:
                tl = str(t).lower()
                v = _np.array([float(tl.count(w)) for w in self._VOCAB], dtype=_np.float32)
                n = _np.linalg.norm(v)
                rows.append(v / n if n else v)
            return _np.stack(rows) if rows else _np.empty((0, len(self._VOCAB)), dtype=_np.float32)

    import agent_runtime.runtime as _rt

    monkeypatch.setattr(_rt, "_READ_EMBEDDER", _BagEmbedder())
    hook, plane, calls = _web_seam()
    transport = _LaneTransport(_research_turns())
    runtime = AgentRuntime(
        transport=transport, loader=None, hook=hook, plane=plane,
        # the served generic report-route research lanes (parity with _run_generic_research_phase):
        read_search_max_fetch=6, emit_article_notes=True, chunked_read=True,
        execution=ExecutionMode.CONCURRENT,
    )
    result = asyncio.run(runtime.run(PlanDAG(nodes=[_served_research_node()], goal="g"),
                                     run_id="n4w-generic"))
    tv = result.results["r1_research"].tool_value or {}

    # breadth lane: the node actually fetched a real source through the served research config.
    assert calls["fetches"] == [_URL]
    assert tv.get("fetched") and tv["fetched"][0]["url"] == _URL
    # note lane FIRED: a per-article CONTROL note rode out of the served research node.
    notes = tv.get("article_notes") or []
    assert notes, "emit_article_notes lane did not fire on the served generic research node"
    assert notes[0]["key_claims"] == ["June 13 escalation", "1,200 casualties"]
    # read lane FIRED via d109 relevance-select: the honest coverage signal reached the convo
    # on the role='tool' lane (MESSAGING-LAYER contract, supersedes the d199 per-site
    # role='user' inversion: observations keep the semantic tool label in memory; the LIVE
    # OllamaTransport renders them as ENVELOPED user turns — [TOOL RESULT]… — so the model
    # both sees them and can tell tool output from the user; see
    # llm_framework/tests/test_observation_envelope.py). The old map/reduce summarizer did
    # NOT fire (relevance-select replaced it when the embedder is wired).
    assert any("relevant passages in this source" in t for t in transport.tool_turns), (
        "d109 relevance-select read did not fire on the served generic research node"
    )
    # The genuine USER lane carries no tool observation — the lanes are distinct now.
    assert not any("relevant passages in this source" in t for t in transport.user_turns), (
        "the relevance-select read leaked onto the role='user' lane (observations must ride "
        "role='tool'; the transport envelope owns their rendering)"
    )
    assert not transport.summarizer_calls, (
        "the map/reduce summarizer fired; relevance-select should have replaced it"
    )
    # findings are RAW prose (content, d50.1), grounded in the read source.
    assert "escalated June 13" in (result.results["r1_research"].output or "")


def test_lanes_dark_when_not_wired():
    """Control: a plain research run with the lane flags DEFAULT-OFF fires NEITHER lane —
    proving the prior test measures the WIRING, not an always-on effect (no-regression
    baseline for the short/headlines/csv acyclic paths)."""
    hook, plane, calls = _web_seam()
    transport = _LaneTransport(_research_turns())
    runtime = AgentRuntime(
        transport=transport, loader=None, hook=hook, plane=plane,
        read_search_max_fetch=6, execution=ExecutionMode.CONCURRENT,
        # emit_article_notes / chunked_read DEFAULT OFF.
    )
    result = asyncio.run(runtime.run(PlanDAG(nodes=[_served_research_node()], goal="g"),
                                     run_id="ctrl"))
    tv = result.results["r1_research"].tool_value or {}
    assert "article_notes" not in tv, "note lane fired with the flag OFF (regression)"
    assert not transport.summarizer_calls, "chunked read fired with the flag OFF (regression)"
