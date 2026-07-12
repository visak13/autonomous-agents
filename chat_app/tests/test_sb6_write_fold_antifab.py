"""SB-6/d299/d301 — SELF-POLICING anti-fabrication gate for the WRITE FOLD.

These assertions FAIL if a future edit reintroduces the SA-6-PART-2 fabrication on the served
write path (the USER: 'the edit is complete non-sense', d278): the ENGINE authoring the write
STRUCTURE (stamping ``tool=file_write`` / ``role=None`` / a forced linear chain), or routing the
write by a spec-name / role-name conditional.

The faithful write fold ((B)/d299/d301): the write METHODOLOGY lives in the ``section-html-writer``
SPEC BODY (the writer reads it via ``_compose_system``) + the FILE bundle doctrine (surfaced on
self-select); the PLANNER authors the section topology + the ``depends_on`` chain + the per-section
``source_ids`` by REASONING; the engine routes the TOOL-LESS write node to the served writer by
DELIVERY-CONTEXT DATA (the write-phase-exclusive ``deliverable_path``; ``chain_sources`` is the
source-text seam a follow-up reader also carries, so it is NOT the route discriminator — d301) and
only PERSISTS the model's spec-driven raw emission (the d49/d50 delivery design — content/structure
are the model's, not the engine's).

ALLOWED-SET: the ``deliverable_path`` delivery-context route; the framework REVIEW node-id route (kept UNCHANGED,
deferred to SB-7, d300). FORBIDDEN: engine ``tool=file_write`` routing for the report write /
``role=None`` stamp / any spec-name / role-name conditional deciding the write route or structure.
"""
from __future__ import annotations

import inspect

from chat_app.agentic import (
    _WRITE_FILE_SHAPE,
    _build_write_planning_event,
    _compose_write_goal,
    _normalize_write_dag,
    _render_write_planning_event,
)
from agent_runtime.runtime import SubAgent


def _code(fn) -> str:
    """Source with the leading docstring stripped, so PROSE mentions never trip the grep."""
    src = inspect.getsource(fn)
    parts = src.split('"""')
    return parts[2] if len(parts) >= 3 else src


# --- the engine authors NO write structure (the SA-6 PART-2 fabrication) --------------------- #
def test_normalize_write_dag_authors_no_structure():
    body = _code(_normalize_write_dag)
    flat = body.lower().replace(" ", "")
    assert 'tool="file_write"' not in body and "tool='file_write'" not in body, "no engine tool stamp"
    assert "role=none" not in flat, "no engine role=None stamp"
    assert "depends_on=" not in body, "no engine re-chain (the PLANNER authors the chain)"
    for banned in ("spec_id", "spec_name", "node.spec", "specialization", "role==", "==role"):
        assert banned not in flat, f"engine must not branch on {banned!r}"


# --- the write METHODOLOGY is OUT of the engine (migrated to the spec body) ------------------ #
def test_write_planning_event_carries_no_methodology_strings():
    ev = _build_write_planning_event("f", [{"url": "u"}], memory_handle="m", data_complexity="c")
    assert "write_directive" not in ev and "output_desired" not in ev
    # only grounding DATA the planner reasons over remains
    assert set(ev) <= {"kind", "sections_basis", "sources", "memory_handle",
                       "findings_digest", "data_complexity"}


def test_render_write_planning_event_has_no_topology_stamp():
    block = _render_write_planning_event({
        "kind": "write_plan", "memory_handle": "m",
        "findings_digest": "d", "data_complexity": "several concerns",
    })
    assert "OUTLINE/LEAD node FIRST" not in block  # engine topology stamp retired
    assert "write sectioned" not in block.lower()


# --- the planner is guided TOOL-NEUTRALLY (it authors TOOL-LESS write nodes) ----------------- #
def test_compose_write_goal_is_tool_neutral():
    goal = _compose_write_goal("Write a report.", "report.html", "findings",
                               "SOURCES: [S1] ...")
    assert "file_write" not in goal, "the planner must NOT be told to author file_write nodes (B)"
    # RP-1 (d319/d311): the engine 'CHAINED ... depends_on' topology framing is RETIRED — the
    # planner authors topology by reasoning over the write shape; the source_ids assignment
    # guidance (grounding, not structure) stays.
    assert "source_ids" in goal


def test_write_file_shape_description_is_tool_neutral():
    assert "file_write" not in _WRITE_FILE_SHAPE.description
    # the ordered single-file accumulation rides the shape's SEQUENTIAL execution (not an engine chain)
    assert _WRITE_FILE_SHAPE.execution == "sequential"


# --- the write ROUTE is DELIVERY-CONTEXT DATA, not a tool/spec stamp; the puller is dead ----- #
def test_write_route_is_delivery_context_not_spec_or_dead_puller():
    src = inspect.getsource(SubAgent.run)
    # the report write route keys on the WRITE-PHASE-EXCLUSIVE deliverable_path DATA signal — NOT
    # chain_sources (over-broad: a follow-up reader also carries it, d301) and NOT a tool/spec stamp
    assert "self._deliverable_path" in src
    # NO spec-name / spec-id conditional decides the route (code patterns, never in prose)
    for banned in ("node.spec ==", ".spec ==", "spec_id ==", "primary_spec ==", "spec_names =="):
        assert banned not in src, f"write route must not branch on {banned!r}"
    # the prod-dead _run_tool_calling_writer PULLER is NOT dispatched (callsite removed; SB-7 cleanup)
    assert "_run_tool_calling_writer(inputs)" not in src
