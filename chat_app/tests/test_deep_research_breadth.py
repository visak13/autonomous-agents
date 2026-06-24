"""s9/N1 (d60/c15 part-a) — DEEP-RESEARCH BREADTH knob.

The deep-research gather nodes lift the legacy hard-wired ``read_search_max_fetch=3``
fetch cap to a configurable BREADTH budget (~8-12) so a detailed sourced report grounds
in MANY real sources, not three. The knob is a NON-FLOW ceiling (the agent still reasons
about whether/which to fetch) and is env-overridable for live tuning (d60/D-C). These
tests pin the band, the env-override, and that the two deep-research call sites WIRE the
knob (not a stray literal). The actual breadth-reachability behaviour is proven in
``agent_runtime/tests/test_research_read_fetch.py::test_breadth_turn_ceiling_rises_with_fetch_cap``.
"""
from __future__ import annotations

import importlib
import inspect

import chat_app.agentic as agentic


def test_breadth_default_is_in_band_and_beats_legacy_cap():
    # ~8-12 real articles (the c15 band), and strictly more than the retired 3.
    assert 8 <= agentic.DEEP_RESEARCH_FETCH_BREADTH <= 12
    assert agentic.DEEP_RESEARCH_FETCH_BREADTH > 3


def test_breadth_is_env_configurable(monkeypatch):
    # The knob is configurable for live tuning without a code edit (d60/D-C).
    monkeypatch.setenv("RA_RESEARCH_FETCH_BREADTH", "9")
    reloaded = importlib.reload(agentic)
    try:
        assert reloaded.DEEP_RESEARCH_FETCH_BREADTH == 9
    finally:
        # Restore the module to its default so later tests see the baseline.
        monkeypatch.delenv("RA_RESEARCH_FETCH_BREADTH", raising=False)
        importlib.reload(agentic)


def test_deep_research_sites_wire_the_breadth_knob_not_a_literal():
    # The INLINE deep-research path still wires the configurable knob directly; the
    # legacy hard-wired ``read_search_max_fetch=3`` must not appear on it.
    inline_src = inspect.getsource(agentic._run_deep_research)
    assert "read_search_max_fetch=DEEP_RESEARCH_FETCH_BREADTH" in inline_src
    assert "read_search_max_fetch=3" not in inline_src

    # P2-5c (d135 FLAG-FREE END-STATE) — the SECTIONED (report) path now routes PHASE-1
    # research through the GENERIC declarative-unroll + AgentRuntime growable engine
    # (``_run_generic_research_phase``); the bespoke ``run_research_tree`` loop +
    # ``_make_tree_gather`` are RETIRED. The sectioned function calls the generic phase and
    # wires no stray ``read_search_max_fetch=3`` literal.
    sectioned_src = inspect.getsource(agentic._run_deep_research_sectioned)
    assert "_run_generic_research_phase(" in sectioned_src
    assert "read_search_max_fetch=3" not in sectioned_src
    # The breadth knob is wired INSIDE the generic phase via the report-pinned breadth
    # (``research_fetch_breadth=PLAN_CHAIN_TREE_BREADTH``), NOT a literal.
    generic_src = inspect.getsource(agentic._run_generic_research_phase)
    assert "research_fetch_breadth=PLAN_CHAIN_TREE_BREADTH" in generic_src


def test_tree_leaf_breadth_tracks_the_research_breadth_env(monkeypatch):
    # Q-C tuneable: the sectioned (tree) path's leaf fetch breadth still tracks the
    # shared RA_RESEARCH_FETCH_BREADTH env knob (no code edit), via TreeConfig.
    from agent_runtime import TreeConfig
    monkeypatch.setenv("RA_RESEARCH_FETCH_BREADTH", "9")
    try:
        assert TreeConfig.from_env().leaf_breadth == 9
    finally:
        monkeypatch.delenv("RA_RESEARCH_FETCH_BREADTH", raising=False)


# --- s9/N1 REVISE (d62/d63): num_ctx sized for breadth so breadth=10 does not
# trip the d22 window-overflow regime (16384) the recipe fixed at 32768. -----------

def test_research_num_ctx_is_the_proven_32768_regime():
    # The d22 overflow regime (≤16384) must NOT be the default; 32768 is the proven
    # E4B regime (full deep-research prompt + CoT fits, SWA keeps KV nearly free).
    assert agentic.DEEP_RESEARCH_NUM_CTX >= 32768


def test_research_num_ctx_is_env_configurable(monkeypatch):
    # Mirrors the breadth knob: live-tunable without a code edit (RA_RESEARCH_NUM_CTX).
    monkeypatch.setenv("RA_RESEARCH_NUM_CTX", "65536")
    reloaded = importlib.reload(agentic)
    try:
        assert reloaded.DEEP_RESEARCH_NUM_CTX == 65536
    finally:
        monkeypatch.delenv("RA_RESEARCH_NUM_CTX", raising=False)
        importlib.reload(agentic)


def test_deep_research_sites_size_num_ctx_off_the_d22_overflow_regime():
    # The INLINE deep-research runtime runs at the sized num_ctx constant; the bare
    # ``num_ctx=16384`` literal (the d22 overflow window at breadth=10) is gone.
    inline_src = inspect.getsource(agentic._run_deep_research)
    assert '"num_ctx": DEEP_RESEARCH_NUM_CTX' in inline_src
    assert '"num_ctx": 16384' not in inline_src

    # P2-5c (d135 FLAG-FREE END-STATE) — the SECTIONED (report) path runs PHASE-1 through
    # the generic engine, which sizes the sub-agent window via
    # ``subagent_num_ctx=DEEP_RESEARCH_NUM_CTX`` (the proven 32768 SWA regime), off the d22
    # overflow regime. No bare ``num_ctx=16384`` literal on either function.
    sectioned_src = inspect.getsource(agentic._run_deep_research_sectioned)
    assert '"num_ctx": 16384' not in sectioned_src
    generic_src = inspect.getsource(agentic._run_generic_research_phase)
    assert '"num_ctx": 16384' not in generic_src
    assert "subagent_num_ctx=DEEP_RESEARCH_NUM_CTX" in generic_src
    from agent_runtime import TreeConfig
    assert TreeConfig().num_ctx >= 32768
