"""RP-6a (d359/d361) — SELF-POLICING gate: the write-file shape is a DISK SHAPE.

RP-6 re-architects the residual hardcoded structural spots so deep-research fully
EMERGES from the generic engine + disk shapes (USER d359). RP-6a is step 1: the
``write-file`` shape used to be a HARDCODED in-engine ``ShapeSpec`` constant
(``agentic._WRITE_FILE_SHAPE``). Per the d341/d319 charter (behavior lives in SHAPES
on disk, NOT baked in engine code) it now lives in ``shapes/write-file.toml`` and is
LOADED from disk via the SAME ``load_shape`` mechanism the other shapes use
(schedule-leg.toml / codebase-summary.toml / deep-research.toml).

These assertions FAIL if a future edit re-bakes the shape back into engine code, or if
the relocation silently changed the shape's behavior (name / execution / description).
Behavior-PRESERVING relocation — the loaded spec is byte-identical to the former constant.
"""
from __future__ import annotations

from pathlib import Path

from agent_runtime.shapes import SHAPES_DIR, load_shape, load_shapes
from chat_app.agentic import _WRITE_FILE_SHAPE
import chat_app.agentic as agentic


# The exact behavior the former hardcoded constant carried (relocate, don't rewrite).
_EXPECTED_DESCRIPTION = (
    "A WRITE-FILE plan: ONE node authors the whole document into a single deliverable "
    "file (accumulating it part by part within the node), plus ONE final_review "
    "reviewer node bound to the same specialization. Select for any goal whose outcome "
    "is a written document/file."
)


# --- the shape lives on disk and loads via the STANDARD mechanism --------------------------- #
def test_write_file_shape_file_exists_on_disk():
    path = Path(SHAPES_DIR) / "write-file.toml"
    assert path.is_file(), "write-file shape must be a declarative disk file in shapes/"


def test_write_file_shape_loads_via_standard_loader():
    # harvested by the SAME catalog loader the planner's shape-selection enum uses
    catalog = load_shapes()
    assert "write-file" in catalog, "write-file must be in the standard shape catalog"
    shape = load_shape("write-file")
    assert Path(shape.source).name == "write-file.toml"


# --- behavior is PRESERVED: the loaded spec is byte-identical to the former constant -------- #
def test_write_file_shape_behavior_preserved():
    shape = load_shape("write-file")
    assert shape.name == "write-file"
    assert shape.execution == "sequential"
    assert shape.description == _EXPECTED_DESCRIPTION


# --- the engine constant now COMES FROM DISK (not a hardcoded ShapeSpec literal) ------------ #
def test_engine_constant_is_the_disk_shape():
    # the module-level _WRITE_FILE_SHAPE is the disk-loaded spec — its source points at the toml
    assert Path(_WRITE_FILE_SHAPE.source).name == "write-file.toml", (
        "_WRITE_FILE_SHAPE must be LOADED from disk, not constructed from a hardcoded constant"
    )
    assert _WRITE_FILE_SHAPE.name == "write-file"
    assert _WRITE_FILE_SHAPE.execution == "sequential"
    assert _WRITE_FILE_SHAPE.description == _EXPECTED_DESCRIPTION


def test_hardcoded_shape_constant_is_gone_from_agentic():
    src = Path(agentic.__file__).read_text(encoding="utf-8")
    # the engine LOADS the shape from disk...
    assert 'load_shape("write-file")' in src, "the engine must load the write-file shape from disk"
    # ...and the hardcoded ShapeSpec constant + its inline description are GONE from agentic.py
    assert "_WRITE_FILE_SHAPE = ShapeSpec(" not in src, "the hardcoded ShapeSpec constant must be removed"
    assert "A WRITE-FILE plan:" not in src, "the shape's description now lives ONLY in the disk shape file"
