"""P2.4 (d131 / d132.D / d133) — DEEP-RESEARCH SHAPE + SPEC: the stop signal is
DEFINED IN THE SHAPE (completeness "fill all the blanks", not a hard-coded depth
cap), workers emit reusable DATA POINTS whose gaps feed the next gap-question, the
deep-research SPEC seeds the investigative methodology, and NO-WIKIPEDIA is honored
by the tool-enforced deny-list the shape declares.

Fully OFFLINE (no GPU, no network): the shape is parsed from its on-disk TOML; the
decision node runs on the same scripted-transport + injected-gather seam as
``test_s13_decision_enrichment.py``; the deny-list runs the real web-search builder
against an injected fake backend. Proves each P2.4 requirement as a fast gate.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_runtime.research_tree import (
    Branch,
    LeafResult,
    Tree,
    TreeConfig,
    _DECISION_INSTRUCTION,
    _DECOMPOSE_INSTRUCTION,
    _DEFAULT_DECOMPOSE_SENTENCE,
    _DEFAULT_STOP_SENTENCE,
    _decision_instruction,
    _decompose_instruction,
    _methodology_block,
    run_decision_node,
    run_decompose_node,
)
from agent_runtime.shapes import load_shape, load_shapes

from reactive_tools.web_tools import ResultCache, make_web_search


# --------------------------------------------------------------------------- #
# Scripted transport — records the user turn it saw so we can assert what the
# model was actually shown (the shape stop signal / the seeded methodology).
# --------------------------------------------------------------------------- #
class _ChatResult:
    def __init__(self, content: str) -> None:
        self.role = "assistant"
        self.content = content
        self.thinking = None
        self.tool_calls = None
        self.raw = None


class _ScriptedTransport:
    def __init__(self, turns: list[str]) -> None:
        self._turns = list(turns)
        self.calls: list[str] = []  # the user turn of each chat() call

    def chat(self, messages, **opts) -> _ChatResult:
        user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        self.calls.append(user)
        i = len(self.calls) - 1
        content = self._turns[i] if i < len(self._turns) else "FINAL PLAN (fallback)."
        return _ChatResult(content)

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content


def _note(sid, trust, title, claims, gaps):
    return {
        "source_id": sid, "url": f"https://ex/{sid}", "title": title,
        "source_trust": trust, "category": "x", "summary": "s",
        "key_claims": claims, "relevance": "r", "gaps_or_followups": gaps,
    }


GOAL = "Write a detailed sourced report on the June 2025 US-Iran conflict."

# The deep-research spec doc the served route feeds in as ``methodology``.
_SPEC_DOC = (
    Path(__file__).resolve().parents[2]
    / "var" / "chat_app" / "specs" / "Deep-research.md"
)


# =========================================================================== #
# (1) The STOP SIGNAL is DEFINED IN THE SHAPE — a completeness test, NOT a depth cap.
# =========================================================================== #
def test_shape_declares_completeness_stop_not_a_depth_cap():
    shape = load_shape("deep-research")
    stop = shape.completeness_stop
    # The stop signal is real, free TEXT the model reasons over — not a number.
    assert isinstance(stop, str) and stop.strip()
    assert not stop.strip().isdigit()
    low = stop.lower()
    # It is a COMPLETENESS "fill all the blanks" criterion (gap-driven), and it
    # explicitly disclaims the arbitrary-depth stop.
    assert "blank" in low and ("gap" in low or "facet" in low)
    assert "depth" in low  # "do NOT halt at an arbitrary depth"
    # The shape ALSO declares the cross-cutting source deny-list (no Wikipedia).
    assert "wikipedia.org" in {d.lower() for d in shape.deny_domains}


def test_completeness_stop_is_read_from_shape_into_the_decision_instruction():
    shape = load_shape("deep-research")
    # No shape stop supplied → BYTE-IDENTICAL to the baked default (offline / no-shape
    # path is unchanged — no regression off the served route).
    assert _decision_instruction(None) == _DECISION_INSTRUCTION
    assert _decision_instruction("") == _DECISION_INSTRUCTION
    # Shape stop supplied → the default stop sentence is REPLACED by the shape's text,
    # so the stop SEMANTICS now live in the shape file, not this hard-coded prompt.
    woven = _decision_instruction(shape.completeness_stop)
    assert _DEFAULT_STOP_SENTENCE not in woven
    assert "DEFINED IN THE DEEP-RESEARCH SHAPE" in woven
    assert shape.completeness_stop.strip()[:40] in woven
    # everything ELSE in the instruction is preserved (only the stop clause changed).
    assert "expand_branch" in woven and "stop_research" in woven


def test_decision_node_shows_the_shape_stop_signal_to_the_model():
    # The served seam: run_decision_node(stop_criteria=shape.completeness_stop) must put
    # the shape's completeness stop in front of the model (the prompt it actually sees).
    shape = load_shape("deep-research")
    transport = _ScriptedTransport([
        '{"tool":"stop_research","args":{"reason":"every facet answered, blanks filled"}}',
    ])
    tree = Tree(fan_out=5)
    res = asyncio.run(run_decision_node(
        transport, goal=GOAL, state_render="notes so far", tree=tree,
        config=TreeConfig(decide_max_turns=4), parent_depth=0,
        stop_criteria=shape.completeness_stop,
    ))
    assert res.stop_research == {"reason": "every facet answered, blanks filled"}
    # The model was shown the shape-defined stop signal, not the baked default sentence.
    prompt = transport.calls[0]
    assert "DEFINED IN THE DEEP-RESEARCH SHAPE" in prompt
    assert _DEFAULT_STOP_SENTENCE not in prompt


# =========================================================================== #
# (2) [retired] The worker-data-point-gap-feeds-the-next-gap-question end-to-end
#     test drove run_research_tree's layer loop, which was retired with the
#     bespoke orchestrator (P2-5c). The data-point/gap MEANING signal that fed it
#     is still covered by the render_for_decision tests in
#     test_s13_decision_enrichment.py.
# =========================================================================== #


# =========================================================================== #
# (3) The deep-research SPEC seeds the investigative METHODOLOGY, and it is applied.
# =========================================================================== #
def test_spec_seeds_methodology_with_the_required_doctrine():
    body = _SPEC_DOC.read_text(encoding="utf-8").lower()
    # identify what/when/why/how
    for facet in ("what", "when", "why", "how"):
        assert facet in body
    # reliable sources FIRST + verify
    assert "reliable sources" in body and "verif" in body
    # fill the blanks / completeness stop
    assert "blank" in body
    # expand in the RIGHT dimensions
    for dim in ("timeline", "cost", "impact"):
        assert dim in body
    # avoid the wrong ones + NO wikipedia
    assert "wikipedia" in body and ("social" in body or "opinion" in body)


def test_methodology_is_fed_into_the_decision_prompt():
    # The served route passes the spec body as ``methodology`` → it must lead the decision
    # prompt so the agent reasons OVER the doctrine (d107(1)), here proven on the seam.
    methodology = "RELIABLE SOURCES FIRST. Identify what/when/why/how. NEVER use Wikipedia."
    transport = _ScriptedTransport([
        '{"tool":"stop_research","args":{"reason":"done"}}',
    ])
    asyncio.run(run_decision_node(
        transport, goal=GOAL, state_render="notes", tree=Tree(fan_out=5),
        config=TreeConfig(decide_max_turns=3), parent_depth=0,
        methodology=methodology,
    ))
    prompt = transport.calls[0]
    assert "RESEARCH METHODOLOGY" in prompt  # _methodology_block header
    assert "NEVER use Wikipedia" in prompt
    # empty methodology → no header injected (byte-identical offline path)
    assert _methodology_block("") == ""


# =========================================================================== #
# (4) NO WIKIPEDIA — the shape-declared deny-list is honored at the TOOL layer.
# =========================================================================== #
def test_shape_deny_list_keeps_wikipedia_out_at_the_tool():
    shape = load_shape("deep-research")
    # The shape's declared cross-cutting source policy is enforced by the P2.1 web tool:
    # feed the shape's deny_domains as the tool baseline and prove every wiki-family row
    # (incl. subdomains, across the wikimedia/wiktionary family) is dropped before the
    # model ever sees it — never fetched, never citable.
    rows = [
        {"title": "Wiki", "url": "https://en.wikipedia.org/wiki/Iran", "snippet": "w"},
        {"title": "Reuters", "url": "https://www.reuters.com/world/iran", "snippet": "r"},
        {"title": "Commons", "url": "https://commons.wikimedia.org/x", "snippet": "c"},
        {"title": "Wikt", "url": "https://en.wiktionary.org/x", "snippet": "d"},
    ]
    search = make_web_search(
        backend=lambda *a, **k: list(rows),
        cache=ResultCache(),
        deny_domains=shape.deny_domains,   # the SHAPE's declared policy, tool-enforced
    )
    out = search("iran conflict")
    urls = [r["url"] for r in out["results"]]
    assert urls == ["https://www.reuters.com/world/iran"]
    assert out["excluded_count"] == 3


def test_all_shapes_still_load_after_new_fields():
    # The two new optional fields must not break ANY existing shape file (they default
    # to empty); the whole catalog still parses.
    catalog = load_shapes()
    assert "deep-research" in catalog
    # shapes WITHOUT the field default cleanly (empty stop, empty deny-list).
    linear = catalog.get("linear")
    if linear is not None:
        assert linear.completeness_stop == "" and linear.deny_domains == ()


# =========================================================================== #
# s14/a15 (d160/d161) — BREADTH is a SHAPE PROPERTY: the deep-research shape declares
# a decompose_methodology doctrine that the DECOMPOSE-FIRST seed reasons over to author
# >=3 scoped facets (curing the d160 thin-report 1-source collapse), WITHOUT a hard-coded
# force-count. Mirrors the completeness_stop shape-property tests above, for the seed.
# =========================================================================== #
def test_shape_declares_decompose_methodology_not_a_force_count():
    shape = load_shape("deep-research")
    dm = shape.decompose_methodology
    # The breadth signal is real, free TEXT the model reasons over — not a number/force-count.
    assert isinstance(dm, str) and dm.strip()
    assert not dm.strip().isdigit()
    low = dm.lower()
    # It is a MULTI-DIMENSION breadth doctrine (scope the real facets), and it explicitly
    # frames the >=3 floor as doctrine ("three or more dimensions"), not "emit exactly N".
    assert "dimension" in low and "facet" in low
    assert "three or more" in low
    # It names the canonical facets the thesis spans (timeline / figures / causes / impact).
    assert "timeline" in low and ("figure" in low or "cost" in low)
    # It is NOT a code branch: the shape text never instructs a literal node count to emit.
    assert "exactly" not in low


def test_decompose_methodology_is_read_from_shape_into_the_decompose_instruction():
    shape = load_shape("deep-research")
    # No shape doctrine supplied → BYTE-IDENTICAL to the baked default (offline / no-shape
    # path is unchanged — no regression off the served route). Mirrors completeness_stop.
    assert _decompose_instruction(None) == _DECOMPOSE_INSTRUCTION
    assert _decompose_instruction("") == _DECOMPOSE_INSTRUCTION
    # Shape doctrine supplied → the default breadth sentence is REPLACED by the shape's text,
    # so the breadth SEMANTICS now live in the shape file, not this hard-coded prompt.
    woven = _decompose_instruction(shape.decompose_methodology)
    assert _DEFAULT_DECOMPOSE_SENTENCE not in woven
    assert "DEFINED IN THE DEEP-RESEARCH SHAPE" in woven
    assert shape.decompose_methodology.strip()[:40] in woven
    # everything ELSE in the decompose instruction is preserved (only the breadth clause changed).
    assert "expand_branch" in woven and "DECOMPOSE the goal" in woven


def test_decompose_node_shows_the_shape_breadth_doctrine_to_the_model():
    # The served seam: run_decompose_node(decompose_criteria=shape.decompose_methodology) must
    # put the shape's breadth doctrine in front of the model (the prompt it actually sees), and
    # the model's REAL expand_branch calls mutate the tree (no fabricated decomposition).
    shape = load_shape("deep-research")
    transport = _ScriptedTransport([
        '{"tool":"expand_branch","args":{"question":"timeline of the June 2025 events"}}',
        '{"tool":"expand_branch","args":{"question":"casualty and damage figures"}}',
        '{"tool":"expand_branch","args":{"question":"causes and regional drivers"}}',
        "DECOMPOSED: three scoped facets.",
    ])
    tree = Tree(fan_out=5)
    branches = asyncio.run(run_decompose_node(
        transport, goal=GOAL, tree=tree, config=TreeConfig(decide_max_turns=6),
        decompose_criteria=shape.decompose_methodology,
    ))
    # the model authored THREE scoped facets via real tree mutation (breadth front-loaded).
    assert len(branches) == 3
    # The model was shown the shape-defined breadth doctrine, not the baked default sentence.
    prompt = transport.calls[0]
    assert "DEFINED IN THE DEEP-RESEARCH SHAPE" in prompt
    assert _DEFAULT_DECOMPOSE_SENTENCE not in prompt
