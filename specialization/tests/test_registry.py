"""Tests for the specialization model + the d10 registry split.

The load-bearing assertions (d10): :meth:`SpecRegistry.index` carries NO body
(planner-facing lookup), while :meth:`SpecRegistry.load` returns the full body
(sub-agent loader). Plus markdown round-trip and the model guards.
"""
from __future__ import annotations

import dataclasses

import pytest

from specialization.model import (
    CompiledSpec,
    RawDefinition,
    SpecIndexEntry,
    parse_compiled_spec,
    parse_frontmatter_only,
)
from specialization.registry import SpecRegistry


# A representative compiled spec with a long-ish body (the thing the planner
# must NOT see, and the sub-agent MUST get).
BODY = (
    "# Markdown report specialist\n\n"
    "You write a detailed, well-structured markdown report on the given topic.\n"
    "Use ## sections, cite sources inline, and end with a Sources list.\n"
    "Never emit HTML; never invent citations.\n"
)


def _spec(name: str = "markdown-report", source: str = "ui") -> CompiledSpec:
    return CompiledSpec(
        name=name,
        description="Writes a detailed markdown report on a topic",
        source=source,
        research_trace_ref="research/run-001",
        created_at="2026-06-12T00:00:00+00:00",
        body=BODY,
    )


# ----------------------------- model: guards ----------------------------- #
def test_raw_definition_rejects_empty_name():
    with pytest.raises(ValueError):
        RawDefinition(name="  ", description="d", intent="i")


def test_compiled_spec_rejects_bad_source():
    with pytest.raises(ValueError):
        CompiledSpec(name="x", description="d", source="cli", body="b")


# ------------------------- model: markdown round-trip ------------------------- #
def test_markdown_round_trip_preserves_all_fields():
    spec = _spec()
    text = spec.to_markdown()
    back = parse_compiled_spec(text)
    assert back.name == spec.name
    assert back.description == spec.description
    assert back.source == spec.source
    assert back.research_trace_ref == spec.research_trace_ref
    assert back.created_at == spec.created_at
    assert back.body.strip() == spec.body.strip()


def test_frontmatter_only_parse_does_not_include_body():
    meta = parse_frontmatter_only(_spec().to_markdown())
    assert set(meta) >= {"name", "description", "source", "research_trace_ref", "created_at"}
    # The body text must not have leaked into any frontmatter value.
    assert all("Markdown report specialist" not in v for v in meta.values())


# ------------------ registry: the d10 split (load-bearing) ------------------ #
def test_index_carries_no_body(tmp_path):
    reg = SpecRegistry(tmp_path / "specs")
    reg.register(_spec("markdown-report"))
    reg.register(_spec("html-report"))

    rows = reg.index()
    assert {r.name for r in rows} == {"html-report", "markdown-report"}
    # Planner-facing lookup is EXACTLY {name, description, source} — body-free.
    for r in rows:
        assert isinstance(r, SpecIndexEntry)
        index_fields = {f.name for f in dataclasses.fields(r)}
        assert index_fields == {"name", "description", "source"}
        assert "body" not in index_fields
        # No body content reachable through any index field's value.
        for v in r.as_dict().values():
            assert "You write a detailed" not in v


def test_load_returns_full_body(tmp_path):
    reg = SpecRegistry(tmp_path / "specs")
    reg.register(_spec("markdown-report"))

    spec = reg.load("markdown-report")
    assert isinstance(spec, CompiledSpec)
    assert "You write a detailed, well-structured markdown report" in spec.body
    assert spec.description == "Writes a detailed markdown report on a topic"
    assert spec.source == "ui"


def test_register_persists_doc_and_round_trips_via_load(tmp_path):
    reg = SpecRegistry(tmp_path / "specs")
    path = reg.register(_spec("markdown-report"))
    assert path.exists()
    assert path.name == "markdown-report.md"
    # A fresh registry over the same dir loads the same body (on-disk durable).
    reg2 = SpecRegistry(tmp_path / "specs")
    assert "markdown-report" in reg2
    assert reg2.load("markdown-report").body.strip() == BODY.strip()


def test_load_unknown_raises_keyerror(tmp_path):
    reg = SpecRegistry(tmp_path / "specs")
    with pytest.raises(KeyError):
        reg.load("nope")


# ----------------- registry: re-editable persistence (s4/RC7) ---------------- #
def test_update_persists_edit_and_preserves_provenance(tmp_path):
    """The re-editable round trip the action requires: register → load-by-id →
    update → re-load shows the edit, with identity + provenance preserved."""
    from specialization.loader import SpecLoader

    reg = SpecRegistry(tmp_path / "specs")
    reg.register(_spec("markdown-report"))  # created_at/source/trace are set

    # fetch the SAME spec by id, then UPDATE its body + description.
    before = reg.load("markdown-report")
    updated = reg.update(
        "markdown-report",
        description="Writes an EDITED markdown report",
        body="# Edited\n\nEDITED RULESET BODY — lead with the outcome.\n",
    )

    # re-fetch shows the edit persisted...
    after = reg.load("markdown-report")
    assert "EDITED RULESET BODY" in after.body
    assert after.description == "Writes an EDITED markdown report"
    assert updated.body == after.body
    # ...while identity + provenance are preserved (NOT a fresh compile).
    assert after.name == before.name == "markdown-report"
    assert after.source == before.source
    assert after.research_trace_ref == before.research_trace_ref
    assert after.created_at == before.created_at

    # EFFECTIVE ON THE NEXT RUN: the SAME loader path a sub-agent composes its
    # spec body through now yields the edited body — and a FRESH registry over the
    # same dir agrees (durable on disk, not just in-memory).
    assert SpecLoader(reg).load_body("markdown-report") == after.body
    reg2 = SpecRegistry(tmp_path / "specs")
    assert "EDITED RULESET BODY" in reg2.load("markdown-report").body


def test_update_partial_keeps_unedited_fields(tmp_path):
    """Overlaying only ``description`` leaves the body untouched, and vice versa."""
    reg = SpecRegistry(tmp_path / "specs")
    reg.register(_spec("markdown-report"))

    reg.update("markdown-report", description="only the description changed")
    spec = reg.load("markdown-report")
    assert spec.description == "only the description changed"
    assert spec.body.strip() == BODY.strip()  # body unchanged


def test_update_unknown_raises_keyerror(tmp_path):
    reg = SpecRegistry(tmp_path / "specs")
    with pytest.raises(KeyError):
        reg.update("nope", body="x")


def test_update_rejects_blank_body(tmp_path):
    reg = SpecRegistry(tmp_path / "specs")
    reg.register(_spec("markdown-report"))
    with pytest.raises(ValueError):
        reg.update("markdown-report", body="   ")
