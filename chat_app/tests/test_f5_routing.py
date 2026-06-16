"""F5 — routing honors a named spec + 'do not search' (deterministic, OFFLINE).

The model-driven :class:`~agent_runtime.shape_selector.ShapeSelector` extracts two
intent signals from the goal (``search_allowed`` + ``requested_specs``);
``run_agentic`` enforces them STRUCTURALLY — no phrasing/keyword matcher:

* 'DO NOT SEARCH' → the web tools are stripped from BOTH the incremental authorer's
  enum and the self-heal re-planner's enum, AND the search-then-read follow-through
  is disabled. A node then CANNOT bind ``web_search``/``web_fetch`` and the runtime
  CANNOT fire one — a structural zero-web-call guarantee.
* USER-NAMED SPEC → on the deep-research route the named (registered) spec becomes
  the SINGLE reused spec (instead of the hard-coded ``research-analyst`` default that
  made a named output spec unreachable); on the acyclic route it is threaded into the
  incremental authorer (told + a terminal-node finalization guarantee).

These assertions are deterministic and model-free: they inspect the offered enums /
the reused spec the routing produces, not a live run.
"""
from __future__ import annotations

from reactive_tools import EventPlane, build_default_hook, register_agentic_tools
from specialization.registry import SpecRegistry
from specialization.seed import DEEP_RESEARCH_SPEC, seed_canonical_rulesets
from llm_framework import FakeTransport

from chat_app.agentic import (
    OFFERED_TOOLS,
    WEB_TOOLS,
    _build_acyclic_runtime,
    _build_incremental_planner,
    _deep_research_spec,
    _filter_web_tools,
)

_MD_SPEC = "markdown-writer"


def _seeded_registry(tmp_path) -> SpecRegistry:
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)  # research-analyst + markdown-writer
    return reg


def _hook(tmp_path):
    hook = build_default_hook(EventPlane(), file_base=tmp_path)
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    return hook


# --------------------------------------------------------------------------- #
# 1) 'do not search' — web tools are structurally unavailable to the authorer
# --------------------------------------------------------------------------- #
def test_filter_web_tools_drops_web_only_when_disallowed():
    tools = ["web_search", "web_fetch", "file_write", "send_mail"]
    assert _filter_web_tools(tools, allow_web=True) == tools  # untouched
    filtered = _filter_web_tools(tools, allow_web=False)
    assert "web_search" not in filtered and "web_fetch" not in filtered
    assert filtered == ["file_write", "send_mail"]  # non-web tools preserved


def test_no_search_strips_web_from_incremental_authoring_enum(tmp_path):
    reg, hook = _seeded_registry(tmp_path), _hook(tmp_path)
    allowed = _build_incremental_planner(
        transport=FakeTransport([]), registry=reg, hook=hook, shape_spec=None,
        allow_web=True,
    )
    blocked = _build_incremental_planner(
        transport=FakeTransport([]), registry=reg, hook=hook, shape_spec=None,
        allow_web=False,
    )
    # web tools offered normally ...
    assert set(WEB_TOOLS) <= set(allowed.tool_names)
    # ... and GONE when the user forbade the web (the node can never bind one).
    assert not (set(WEB_TOOLS) & set(blocked.tool_names))
    # non-web tools still offered (we only dropped the web ones).
    assert "send_mail" in blocked.tool_names and "file_write" in blocked.tool_names


def test_no_search_strips_web_from_selfheal_replanner_schema_and_fetch(tmp_path):
    reg, hook = _seeded_registry(tmp_path), _hook(tmp_path)
    runtime, planner = _build_acyclic_runtime(
        transport=FakeTransport([]), registry=reg, hook=hook, plane=EventPlane(),
        shape_spec=None, allow_web=False,
    )
    tool_enum = planner.call_opts["format"]["properties"]["nodes"]["items"][
        "properties"
    ]["tool"]["enum"]
    assert "web_search" not in tool_enum and "web_fetch" not in tool_enum
    # the d13 search-then-read follow-through is also disabled (belt-and-suspenders).
    assert runtime.read_search_max_fetch == 0


def test_search_allowed_keeps_web_surface_intact(tmp_path):
    # No regression: the default (allow_web=True) offers the full web surface and the
    # follow-through fetch budget, exactly as before F5.
    reg, hook = _seeded_registry(tmp_path), _hook(tmp_path)
    runtime, planner = _build_acyclic_runtime(
        transport=FakeTransport([]), registry=reg, hook=hook, plane=EventPlane(),
        shape_spec=None, allow_web=True,
    )
    tool_enum = planner.call_opts["format"]["properties"]["nodes"]["items"][
        "properties"
    ]["tool"]["enum"]
    assert "web_search" in tool_enum and "web_fetch" in tool_enum
    assert runtime.read_search_max_fetch == 3


# --------------------------------------------------------------------------- #
# 2) named spec — honored on the deep-research route (not the hardcoded default)
# --------------------------------------------------------------------------- #
def test_deep_research_reuses_user_named_spec(tmp_path):
    reg = _seeded_registry(tmp_path)
    # the user explicitly named markdown-writer → it is the reused spec ...
    assert _deep_research_spec(reg, [_MD_SPEC]) == _MD_SPEC
    # ... a request naming none falls back to the research-analyst default ...
    assert _deep_research_spec(reg, []) == DEEP_RESEARCH_SPEC
    assert _deep_research_spec(reg, None) == DEEP_RESEARCH_SPEC
    # ... and an UNregistered name is ignored (never reaches binding).
    assert _deep_research_spec(reg, ["no-such-spec"]) == DEEP_RESEARCH_SPEC


def test_named_spec_threaded_into_incremental_authorer(tmp_path):
    reg, hook = _seeded_registry(tmp_path), _hook(tmp_path)
    planner = _build_incremental_planner(
        transport=FakeTransport([]), registry=reg, hook=hook, shape_spec=None,
        requested_specs=[_MD_SPEC, "no-such-spec"],
    )
    # only the registered named spec survives onto the authorer (the guarantee pass).
    assert planner.requested_specs == [_MD_SPEC]


def test_web_tools_constant_subset_of_offered():
    # The web tools we strip are a real subset of the offered surface (config sanity).
    assert set(WEB_TOOLS) <= set(OFFERED_TOOLS)
