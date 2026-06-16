"""Per-query SHAPE SELECTION via the native structured Gemma path (s3/b1, d1).

Locks the shape-selection judgment point: the planner picks ONE shape per query
through a native structured call whose ``shape`` field is an ``enum`` HARVESTED
from the shape files PLUS the reserved ``escalate`` value, with ``required`` keys
and the d1 ``think=false`` / ``temp 0`` options. Driven on the offline
``FakeTransport`` (the model's reply is scripted) — no GPU, no live model.
"""
from __future__ import annotations

import asyncio

import pytest

from llm_framework import FakeTransport

from agent_runtime.selfheal import MalformedOutputError
from agent_runtime.shape_selector import (
    ESCALATE,
    ShapeSelector,
    build_selection_schema,
)
from agent_runtime.shapes import shape_names


def _reply(shape: str, rationale: str = "fits the query") -> str:
    return f'{{"shape": "{shape}", "rationale": "{rationale}"}}'


# --------------------------------------------------------------------------- #
# the OUTPUT SCHEMA: enum = shape names + escalate, required keys (d1)
# --------------------------------------------------------------------------- #
def test_schema_enum_is_harvested_names_plus_escalate():
    names = shape_names()
    schema = build_selection_schema(names)
    enum = schema["properties"]["shape"]["enum"]
    # every on-disk shape is selectable ...
    for n in names:
        assert n in enum
    # ... plus the reserved low-confidence escalation value, and nothing invented.
    assert ESCALATE in enum
    assert set(enum) == set(names) | {ESCALATE}
    # required keys force a complete, parseable decision — incl. the F5 intent
    # signals (search_allowed + requested_specs) so the small model reliably emits
    # them under native format= (required is the lever, not the prose).
    assert schema["required"] == [
        "shape",
        "rationale",
        "search_allowed",
        "requested_specs",
        "wants_file",
        "unmet_specs",
    ]
    assert schema["properties"]["search_allowed"]["type"] == "boolean"
    assert schema["properties"]["requested_specs"]["type"] == "array"
    # s10-a4 file-output signal (d11 invariant): a required boolean so the small
    # model reliably emits it under native format=.
    assert schema["properties"]["wants_file"]["type"] == "boolean"
    # s10-a8 missing-specialist signal: a required FREE-string array (NOT the
    # registered-name enum that locks requested_specs) so the model CAN name a
    # specialization the registry does not have — the structural scenario-3 trigger.
    assert schema["properties"]["unmet_specs"]["type"] == "array"
    assert schema["properties"]["unmet_specs"]["items"] == {"type": "string"}


def test_schema_requested_specs_enum_is_registered_spec_names():
    # F5: when spec names are supplied, the requested_specs items are enum-locked to
    # them (the model cannot invent a specialization); absent them, an unconstrained
    # string array (no empty-enum, which Ollama would reject).
    schema = build_selection_schema(["linear"], ["markdown-writer", "research-analyst"])
    item = schema["properties"]["requested_specs"]["items"]
    assert item.get("enum") == ["markdown-writer", "research-analyst"]
    bare = build_selection_schema(["linear"])
    assert "enum" not in bare["properties"]["requested_specs"]["items"]


def test_selector_advertises_d1_options():
    sel = ShapeSelector(FakeTransport([_reply("linear")]))
    # the proven d1 native path: think OFF top-level, deterministic temp 0.
    assert sel._call_opts["api"] == "native"
    assert sel._call_opts["think"] is False
    assert sel._call_opts["temperature"] == 0
    assert sel._call_opts["num_predict"] >= 128


# --------------------------------------------------------------------------- #
# selecting a concrete shape
# --------------------------------------------------------------------------- #
def test_selects_a_real_shape():
    sel = ShapeSelector(FakeTransport([_reply("modular-parallel", "splits into parallel sub-tasks")]))
    out = asyncio.run(sel.select("research three subtopics then combine them"))
    assert out.shape == "modular-parallel"
    assert out.escalate is False
    assert "parallel" in out.rationale
    # the call advertised the harvested enum (the proof the names came from disk).
    assert "modular-parallel" in sel.last_schema["properties"]["shape"]["enum"]


def test_selects_linear_shape():
    sel = ShapeSelector(FakeTransport([_reply("linear")]))
    out = asyncio.run(sel.select("do step A, then B which needs A, then C which needs B"))
    assert out.shape == "linear"
    assert out.escalate is False


def test_selects_deep_research_shape():
    sel = ShapeSelector(FakeTransport([_reply("deep-research")]))
    out = asyncio.run(sel.select("write an exhaustive multi-layer survey with critique"))
    assert out.shape == "deep-research"
    assert not out.escalate


# --------------------------------------------------------------------------- #
# the ESCALATE low-confidence signal
# --------------------------------------------------------------------------- #
def test_escalate_value_yields_no_shape():
    sel = ShapeSelector(FakeTransport([_reply(ESCALATE, "no shape clearly fits")]))
    out = asyncio.run(sel.select("?? ambiguous one-word goal"))
    assert out.escalate is True
    assert out.shape is None  # the caller routes this to a human / a default
    assert out.rationale


# --------------------------------------------------------------------------- #
# guards: an out-of-enum value or empty goal fails cleanly
# --------------------------------------------------------------------------- #
def test_out_of_enum_shape_raises_after_repair():
    # Valid JSON but an illegal (non-enum) shape → a clean MalformedOutputError,
    # never a silent mis-dispatch.
    sel = ShapeSelector(FakeTransport([_reply("banana"), _reply("banana")]))
    with pytest.raises(MalformedOutputError):
        asyncio.run(sel.select("a goal"))


def test_empty_goal_raises():
    sel = ShapeSelector(FakeTransport([_reply("linear")]))
    with pytest.raises(MalformedOutputError):
        asyncio.run(sel.select("   "))


# --------------------------------------------------------------------------- #
# F5 intent signals: search_allowed + requested_specs, extracted by the SAME call
# --------------------------------------------------------------------------- #
import json as _json


def _reply_f5(
    shape,
    *,
    search_allowed=True,
    requested_specs=None,
    wants_file=False,
    unmet_specs=None,
    rationale="fits",
):
    return _json.dumps(
        {
            "shape": shape,
            "rationale": rationale,
            "search_allowed": search_allowed,
            "requested_specs": list(requested_specs or []),
            "wants_file": wants_file,
            "unmet_specs": list(unmet_specs or []),
        }
    )


def test_search_allowed_false_is_parsed():
    sel = ShapeSelector(FakeTransport([_reply_f5("linear", search_allowed=False)]))
    out = asyncio.run(sel.select("explain photosynthesis without searching the web"))
    assert out.search_allowed is False
    assert out.shape == "linear"


def test_requested_specs_filtered_to_registered_names():
    # Only names in the selector's spec catalog survive (an invented one is dropped),
    # so an out-of-catalog name can never reach binding.
    sel = ShapeSelector(
        FakeTransport([_reply_f5("linear", requested_specs=["markdown-writer", "bogus"])]),
        spec_names=["markdown-writer", "research-analyst"],
    )
    out = asyncio.run(sel.select("write an overview using the markdown-writer specialization"))
    assert out.requested_specs == ["markdown-writer"]
    # the schema advertised the spec enum on requested_specs (the proof it was offered).
    item = sel.last_schema["properties"]["requested_specs"]["items"]
    assert item["enum"] == ["markdown-writer", "research-analyst"]


def test_f5_signals_default_permissive_when_omitted():
    # A reply WITHOUT the F5 fields (e.g. an older transport / canned plan) parses to
    # the permissive defaults — fail-open, byte-identical to the pre-F5 behaviour.
    sel = ShapeSelector(FakeTransport([_reply("linear")]), spec_names=["markdown-writer"])
    out = asyncio.run(sel.select("do step A then B"))
    assert out.search_allowed is True
    assert out.requested_specs == []
    # s10-a4: the file-output signal also defaults False when omitted (fail-open).
    assert out.wants_file is False
    # s10-a8: the missing-specialist signal defaults to [] when omitted (fail-open —
    # no false missing-spec notify on an older/omitting reply).
    assert out.unmet_specs == []


# --------------------------------------------------------------------------- #
# s10-a4 file-output signal (d11/s7-a2 invariant): wants_file, extracted by the
# SAME structured call, enforced by the caller (a file request must not go to the
# fileless deep-research path).
# --------------------------------------------------------------------------- #
def test_wants_file_true_is_parsed():
    sel = ShapeSelector(FakeTransport([_reply_f5("linear", wants_file=True)]))
    out = asyncio.run(sel.select("research X and write the findings as a markdown file"))
    assert out.wants_file is True
    assert out.shape == "linear"


def test_wants_file_false_for_chat_only_answer():
    # A plain answer request (no file) → wants_file False; contrastive baseline.
    sel = ShapeSelector(FakeTransport([_reply_f5("linear", wants_file=False)]))
    out = asyncio.run(sel.select("what are the main features of HTTP/3?"))
    assert out.wants_file is False


def test_wants_file_non_bool_is_failopen_false():
    # A non-boolean wants_file (a confused small-model reply) → False, never truthy
    # coercion, so the guard only fires on an explicit true.
    reply = _json.dumps(
        {
            "shape": "linear",
            "rationale": "fits",
            "search_allowed": True,
            "requested_specs": [],
            "wants_file": "yes",
        }
    )
    sel = ShapeSelector(FakeTransport([reply]))
    out = asyncio.run(sel.select("do step A then B"))
    assert out.wants_file is False


# --------------------------------------------------------------------------- #
# s10-a8 missing-specialist signal: unmet_specs is a FREE-string array (NOT the
# registered-name enum that locks requested_specs) so the model CAN name a
# specialization the registry does not have — the structural scenario-3 trigger.
# Parsing stays dumb (no membership filter); the CALLER's deterministic
# registry-membership check is the single authority on "missing".
# --------------------------------------------------------------------------- #
def test_unmet_specs_free_string_is_parsed_unfiltered():
    # A name the catalog does NOT contain still SURVIVES the parse (unlike
    # requested_specs, which drops out-of-catalog names) — that is the whole point:
    # the runtime must SEE the requested-but-unavailable spec to fire the notify.
    sel = ShapeSelector(
        FakeTransport([_reply_f5("linear", unmet_specs=["forensic-accountant"])]),
        spec_names=["markdown-writer", "research-analyst"],
    )
    out = asyncio.run(sel.select("write me a forensic-accountant report on the filing"))
    assert out.unmet_specs == ["forensic-accountant"]
    # the schema offered unmet_specs as a FREE string array (no enum) — the proof the
    # model was allowed to name a spec outside the catalog.
    item = sel.last_schema["properties"]["unmet_specs"]["items"]
    assert item == {"type": "string"}
    assert "enum" not in item


def test_unmet_specs_deduped_and_blanks_dropped():
    reply = _json.dumps(
        {
            "shape": "linear",
            "rationale": "fits",
            "search_allowed": True,
            "requested_specs": [],
            "wants_file": False,
            "unmet_specs": ["legal-brief", " legal-brief ", "", "patent-examiner"],
        }
    )
    sel = ShapeSelector(FakeTransport([reply]), spec_names=["markdown-writer"])
    out = asyncio.run(sel.select("draft a legal brief and a patent examiner note"))
    assert out.unmet_specs == ["legal-brief", "patent-examiner"]


def test_unmet_specs_non_list_is_failopen_empty():
    # A confused non-list reply → [] (no notify), never a crash or truthy coercion.
    reply = _json.dumps(
        {
            "shape": "linear",
            "rationale": "fits",
            "search_allowed": True,
            "requested_specs": [],
            "wants_file": False,
            "unmet_specs": "forensic-accountant",
        }
    )
    sel = ShapeSelector(FakeTransport([reply]), spec_names=["markdown-writer"])
    out = asyncio.run(sel.select("write a report"))
    assert out.unmet_specs == []
