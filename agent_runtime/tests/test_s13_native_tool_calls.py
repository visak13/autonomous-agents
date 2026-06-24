"""s13 (design §P1) — NATIVE Ollama tool-call layer (d117 GO-NATIVE).

Fast OFFLINE gate (no GPU, no network) for the migration from the homegrown
``startswith('{')`` string-parse gate to NATIVE tool calls: the model's tool call
rides its OWN ``message.tool_calls`` channel (surfaced on ``ChatResult.tool_calls`` /
``Context.tool_calls``), so LEADING PROSE on the same turn can never swallow it (the
d114 probe measured native 19/19 vs string-parse 4/19 missed on the decision/outline
tools). Proves, all offline:

* the transport normaliser reads BOTH the native (object args) and the OpenAI-compat
  (JSON-string args) tool_calls shapes, and ``call_stage`` surfaces them onto the Context;
* a leading-PROSE native tool call is parsed from ``message.tool_calls`` and DISPATCHED
  through the real decision loop (the OLD string parser would have DROPPED it — asserted);
* ``run_decompose_node`` authors scoped children from such native replies;
* the balanced-brace string parser is KEPT as a defensive fallback and still recovers a
  call on a NON-native reply (a transport that returns no ``tool_calls``).

Guardrails honoured: content stays RAW (JSON only ever at the tool-call layer, d50); no
flags (native is the flag-free default, d65); the fallback parse logic is NOT deleted.
"""
from __future__ import annotations

import asyncio

from agent_runtime.research_tree import (
    Branch,
    DecisionResult,
    Tree,
    TreeConfig,
    TREE_TOOLS,
    first_native_call,
    parse_tree_call,
    run_decision_node,
    run_decompose_node,
)

GOAL = "Write a detailed sourced report on the June 2025 US-Iran conflict."


# --------------------------------------------------------------------------- #
# Fakes — a reply object with a native ``tool_calls`` channel, and two scripted
# transports: one NATIVE (surfaces tool_calls) and one NON-native (content only,
# like the legacy ``test_s13_decision_enrichment._ScriptedTransport``).
# --------------------------------------------------------------------------- #
class _Reply:
    """A ChatResult-shaped turn: prose ``content`` PLUS an optional native ``tool_calls``."""

    def __init__(self, content, tool_calls=None):
        self.role = "assistant"
        self.content = content
        self.thinking = None
        self.raw = None
        self.tool_calls = tool_calls


class _NativeScriptedTransport:
    """Replays turns that carry a NATIVE ``tool_calls`` field (the served api='native' shape)."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.calls = []  # opts of each chat call (to assert tools= was threaded)

    def chat(self, messages, **opts):
        self.calls.append(opts)
        i = len(self.calls) - 1
        return self._turns[i] if i < len(self._turns) else _Reply("FINAL PLAN (fallback).", None)

    def complete(self, messages, **opts):
        return self.chat(messages, **opts).content


class _NonNativeScriptedTransport:
    """Replays CONTENT-only turns (no ``tool_calls`` attr) — the defensive-fallback path."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.calls = []

    def chat(self, messages, **opts):
        self.calls.append(opts)
        i = len(self.calls) - 1
        content = self._turns[i] if i < len(self._turns) else "FINAL PLAN (fallback)."
        return _ContentOnly(content)

    def complete(self, messages, **opts):
        return self.chat(messages, **opts).content


class _ContentOnly:
    """A reply with NO ``tool_calls`` attribute at all (truly non-native transport)."""

    def __init__(self, content):
        self.role = "assistant"
        self.content = content
        self.thinking = None
        self.raw = None


# =========================================================================== #
# Transport-layer normaliser — reads BOTH provider shapes into a uniform list.
# =========================================================================== #
def test_s13_normalize_tool_calls_native_and_openai_shapes():
    from llm_framework.transport import _normalize_tool_calls

    # Native /api/chat: arguments is an OBJECT (dict) under message.tool_calls[].function.
    native = [{"function": {"name": "web_search", "arguments": {"query": "iran strikes"}}}]
    assert _normalize_tool_calls(native) == [
        {"name": "web_search", "arguments": {"query": "iran strikes"}}
    ]
    # OpenAI-compat: arguments is a JSON STRING.
    oai = [{"id": "c1", "type": "function",
            "function": {"name": "web_fetch", "arguments": '{"url": "https://ex/a"}'}}]
    assert _normalize_tool_calls(oai) == [
        {"name": "web_fetch", "arguments": {"url": "https://ex/a"}}
    ]
    # Absent / empty / malformed → None so the caller falls through to the string parser.
    assert _normalize_tool_calls(None) is None
    assert _normalize_tool_calls([]) is None
    assert _normalize_tool_calls([{"no_name": 1}]) is None


def test_s13_call_stage_surfaces_tool_calls_to_context():
    from llm_framework import Chain, Context
    from llm_framework.stages import call_stage, prompt_assembly
    from llm_framework.transport import ChatResult, FakeTransport

    tc = [{"name": "expand_branch", "arguments": {"question": "the timeline"}}]
    tp = FakeTransport([ChatResult(role="assistant", content="reasoning prose", tool_calls=tc)])
    ctx = Context(user="research the timeline", transport=tp)
    chain = Chain()
    chain.use(prompt_assembly())
    chain.use(call_stage(tp))
    ctx = chain.run(ctx)
    # raw content AND the native tool calls both ride the Context (the s13 seam).
    assert ctx.raw_output == "reasoning prose"
    assert ctx.tool_calls == tc


# =========================================================================== #
# first_native_call — selection + the fall-through-to-fallback contract.
# =========================================================================== #
def test_s13_first_native_call_selects_accepted_and_ignores_unknown():
    calls = [{"name": "note", "arguments": {"url": "https://x", "summary": "s"}}]
    assert first_native_call(calls, ("web_search", "web_fetch", "note")) == (
        "note", {"url": "https://x", "summary": "s"}
    )
    # An unknown tool name is NOT silently dispatched.
    assert first_native_call([{"name": "frobnicate", "arguments": {}}], ("note",)) is None
    # No native calls → None, so the caller uses the balanced-brace string fallback.
    assert first_native_call(None, ("note",)) is None
    assert first_native_call([], ("note",)) is None


# =========================================================================== #
# The decisive drop-immunity proof: a native tool call that is preceded by PROSE
# on the SAME turn is still dispatched — where the OLD string parser drops it.
# =========================================================================== #
def test_s13_native_decision_call_dispatched_even_with_leading_prose():
    prose_then_expand = "Here's my reasoning: the timeline is the biggest gap, so I'll expand it."
    expand_calls = [{
        "name": "expand_branch",
        "arguments": {"parent": "root", "question": "the timeline of strikes", "rationale": "S1 gap"},
    }]
    # CONTRAST: the legacy string parser DROPS this (prose leads, no '{' start) → returns None.
    assert parse_tree_call(prose_then_expand) is None

    transport = _NativeScriptedTransport([
        _Reply(prose_then_expand, expand_calls),
        _Reply("I think the gathered notes cover the thesis now.",
               [{"name": "stop_research", "arguments": {"reason": "thesis covered"}}]),
    ])
    tree = Tree(fan_out=5)
    res: DecisionResult = asyncio.run(run_decision_node(
        transport, goal=GOAL, state_render="notes", tree=tree,
        config=TreeConfig(decide_max_turns=10), parent_depth=0,
    ))
    # The native call landed DESPITE the leading prose, and the trailing stop carried.
    assert [b.question for b in res.new_branches] == ["the timeline of strikes"]
    assert res.stop_research == {"reason": "thesis covered"}
    assert res.turns == 2
    # The native tool schemas were actually threaded onto the transport call (tools=).
    assert any("tools" in opts for opts in transport.calls)
    assert res.next_direction is None  # stop was not mis-routed into a next-direction


def test_s13_native_decompose_authors_scoped_children():
    transport = _NativeScriptedTransport([
        _Reply("Let me decompose the goal into distinct facets.",
               [{"name": "expand_branch",
                 "arguments": {"parent": "root", "question": "the timeline of events",
                               "rationale": "facet: timeline"}}]),
        _Reply("And the human/material cost.",
               [{"name": "expand_branch",
                 "arguments": {"parent": "root", "question": "the casualty and damage figures",
                               "rationale": "facet: costs"}}]),
        _Reply("PLAN: research the timeline, then the costs.", None),  # prose → decomposition done
    ])
    tree = Tree(fan_out=5)
    children = asyncio.run(run_decompose_node(
        transport, goal=GOAL, tree=tree, config=TreeConfig(decide_max_turns=10),
    ))
    assert [b.question for b in children] == [
        "the timeline of events",
        "the casualty and damage figures",
    ]


# =========================================================================== #
# Defensive fallback (s13 condition 2) — the balanced-brace string parser is KEPT
# and still recovers a call on a NON-native reply (transport returns no tool_calls).
# =========================================================================== #
def test_s13_string_fallback_recovers_on_non_native_path():
    transport = _NonNativeScriptedTransport([
        # JSON leads, trailing prose is ignored by _first_json_object (the kept fallback).
        '{"tool":"expand_branch","args":{"parent":"root","question":"sanctions impact","rationale":"S2 gap"}}'
        " — expanding on the sanctions angle.",
        '{"tool":"stop_research","args":{"reason":"enough"}}',
    ])
    tree = Tree(fan_out=5)
    res = asyncio.run(run_decision_node(
        transport, goal=GOAL, state_render="notes", tree=tree,
        config=TreeConfig(decide_max_turns=10), parent_depth=0,
    ))
    assert [b.question for b in res.new_branches] == ["sanctions impact"]
    assert res.stop_research == {"reason": "enough"}


def test_s13_tree_tool_specs_cover_every_tree_tool():
    # Every dispatchable TREE tool is offered as a native schema (so the model can call it
    # natively) — no tool silently lacks a spec.
    from agent_runtime.research_tree import TREE_TOOL_SPECS

    spec_names = {s["function"]["name"] for s in TREE_TOOL_SPECS}
    assert spec_names == set(TREE_TOOLS)
    for spec in TREE_TOOL_SPECS:
        fn = spec["function"]
        assert spec["type"] == "function"
        assert isinstance(fn["parameters"]["properties"], dict)
        assert isinstance(fn["parameters"]["required"], list)
