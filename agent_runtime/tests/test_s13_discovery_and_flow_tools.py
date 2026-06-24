"""s13 / P2.1 TOOL-LAYER — get_shapes/get_specs DISCOVERY tools + the research
FLOW SEMANTICS relocated INTO the tool descriptions (legible from the surface).

Fully OFFLINE (no GPU, no network):

* ``get_shapes`` reads the real packaged shapes catalog; ``get_specs`` reads a
  body-free index via an injected provider (no dependency on the specialization
  package) and honors the d10 descriptions-only contract.
* both register onto a :class:`GrowableToolRegistry` (selectable + dispatchable).
* the research TREE tool descriptions CARRY the identify -> expand(what/why/when/
  how) -> prune-bad-leads -> stop-when-sufficient flow, so an agent can drive the
  loop from the tool surface alone (d125/d126/d133 tool-drives-the-flow).
"""
from __future__ import annotations

import asyncio

from reactive_tools import EventPlane, GrowableToolRegistry, ToolHook

from agent_runtime.discovery_tools import (
    GET_SHAPES_TOOL,
    GET_SPECS_TOOL,
    make_get_shapes,
    make_get_specs,
    register_discovery_tools,
)
from agent_runtime.research_tree import TREE_TOOL_SPECS


# --------------------------------------------------------------------------- #
# get_shapes — lists the real packaged shape catalog with descriptions
# --------------------------------------------------------------------------- #


def test_get_shapes_lists_catalog_with_descriptions():
    out = make_get_shapes()()
    names = {r["name"] for r in out["shapes"]}
    assert "deep-research" in names                  # the flagship shape is listed
    assert out["count"] == len(out["shapes"]) >= 1
    dr = next(r for r in out["shapes"] if r["name"] == "deep-research")
    assert dr["description"]                          # description is returned
    assert "execution" in dr and "max_iter" in dr


def test_get_shapes_filter_narrows_catalog():
    out = make_get_shapes()(filter="deep-research")
    assert all("deep-research" in r["name"] for r in out["shapes"])
    assert out["count"] >= 1
    empty = make_get_shapes()(filter="no-such-shape-xyz")
    assert empty == {"shapes": [], "count": 0}


# --------------------------------------------------------------------------- #
# get_specs — body-free {name, description, source} rows (d10), filterable
# --------------------------------------------------------------------------- #


_FAKE_SPECS = [
    {"name": "html-writer", "description": "themed HTML report", "source": "ui", "body": "SECRET"},
    {"name": "pirate-tone", "description": "pirate voice", "source": "ui", "body": "SECRET"},
]


def test_get_specs_returns_body_free_rows():
    get_specs = make_get_specs(index_provider=lambda: list(_FAKE_SPECS))
    out = get_specs()
    assert out["count"] == 2
    rows = {r["name"]: r for r in out["specs"]}
    assert rows["html-writer"]["description"] == "themed HTML report"
    assert rows["html-writer"]["source"] == "ui"
    # d10: the compiled body is NEVER exposed by the discovery surface
    for r in out["specs"]:
        assert "body" not in r
    assert "SECRET" not in str(out)


def test_get_specs_filter_and_empty_dir():
    get_specs = make_get_specs(index_provider=lambda: list(_FAKE_SPECS))
    out = get_specs(filter="pirate")
    assert [r["name"] for r in out["specs"]] == ["pirate-tone"]
    # no specs dir + no provider -> empty catalog, not a crash
    assert make_get_specs(specs_dir=None)() == {"specs": [], "count": 0}


# --------------------------------------------------------------------------- #
# Registration — both discovery tools land on a registry (selectable+dispatchable)
# --------------------------------------------------------------------------- #


def test_register_discovery_tools_growth_and_dispatch():
    registry = GrowableToolRegistry(ToolHook(EventPlane()))
    register_discovery_tools(registry, spec_index_provider=lambda: list(_FAKE_SPECS))
    assert "get_shapes" in registry and "get_specs" in registry
    enum = registry.selection_schema()["properties"]["tool"]["enum"]
    assert "get_shapes" in enum and "get_specs" in enum
    res = asyncio.run(registry.hook.invoke("get_specs"))
    assert res.ok is True
    assert {r["name"] for r in res.value["specs"]} == {"html-writer", "pirate-tone"}


def test_discovery_tool_descriptions_explain_discovery():
    assert "DISCOVERY" in GET_SHAPES_TOOL.description
    assert "shape" in GET_SHAPES_TOOL.description.lower()
    assert "DISCOVERY" in GET_SPECS_TOOL.description
    assert "specialization" in GET_SPECS_TOOL.description.lower()


# --------------------------------------------------------------------------- #
# FLOW SEMANTICS legible from the research tool descriptions (d125/d126/d133)
# --------------------------------------------------------------------------- #


def _desc(name: str) -> str:
    for spec in TREE_TOOL_SPECS:
        fn = spec["function"]
        if fn["name"] == name:
            return fn["description"]
    raise AssertionError(f"no tree tool {name!r}")


def test_tree_tool_descriptions_carry_the_research_flow():
    # IDENTIFY/EXPAND along what/why/when/how lives in expand_branch
    expand = _desc("expand_branch").lower()
    assert "expand" in expand and "gap" in expand
    for facet in ("what", "why", "when", "how"):
        assert facet in expand
    # PRUNE bad leads
    prune = _desc("prune_branch").lower()
    assert "prune" in prune and ("redundant" in prune or "off-thesis" in prune)
    # STOP when sufficient (completeness-driven, not arbitrary depth)
    stop = _desc("stop_research").lower()
    assert "stop" in stop and "sufficient" in stop
    assert "gap" in stop or "blank" in stop


def test_tree_section_tools_shape_report_from_findings():
    add = _desc("add_section").lower()
    assert "section" in add and ("grounded" in add or "emerge" in add)
    drop = _desc("drop_section").lower()
    assert "drop" in drop and "support" in drop
