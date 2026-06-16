"""The declarative shape FILES + the execution-discipline field (s3/b1, d5).

Locks the two NEW text-file shapes (``linear`` / ``modular-parallel``) and the
``execution`` field added to :class:`ShapeSpec`: the field is parsed from disk,
validated fail-fast, and round-trips through ``as_dict``. The deep-research shape
stays intact (untouched executor).
"""
from __future__ import annotations

import pytest

from agent_runtime.shapes import (
    VALID_EXECUTION,
    ShapeError,
    ShapeSpec,
    load_shapes,
    shape_names,
)


def test_linear_and_modular_parallel_shapes_exist_on_disk():
    names = shape_names()
    assert "linear" in names
    assert "modular-parallel" in names
    # the deep-research shape is still present (not displaced).
    assert "deep-research" in names


def test_linear_shape_declares_sequential_execution():
    shape = load_shapes()["linear"]
    assert shape.execution == "sequential"
    assert shape.description  # human/LLM-facing one-liner for the selector enum


def test_modular_parallel_shape_declares_concurrent_execution():
    shape = load_shapes()["modular-parallel"]
    assert shape.execution == "concurrent"
    assert shape.description


def test_deep_research_shape_declares_its_own_discipline():
    shape = load_shapes()["deep-research"]
    assert shape.execution == "deep-research"
    # its cyclic mechanics are untouched.
    assert shape.round_roles == ("research", "critic")
    assert shape.final_roles == ("research", "synthesis", "verify")


# --------------------------------------------------------------------------- #
# the execution field: default, validation, round-trip
# --------------------------------------------------------------------------- #
def test_execution_defaults_to_concurrent():
    # A shape that declares no execution token defaults to concurrent (legacy).
    spec = ShapeSpec(name="x")
    assert spec.execution == "concurrent"


def test_execution_is_normalised_and_validated():
    assert ShapeSpec(name="x", execution="SEQUENTIAL").execution == "sequential"
    # an unknown token fails fast (a typo never silently degrades to a default).
    with pytest.raises(ShapeError):
        ShapeSpec(name="x", execution="bogus")


def test_execution_round_trips_through_as_dict():
    spec = ShapeSpec(name="x", execution="sequential")
    assert spec.as_dict()["execution"] == "sequential"


def test_valid_execution_tokens():
    assert VALID_EXECUTION == {"sequential", "concurrent", "deep-research"}
