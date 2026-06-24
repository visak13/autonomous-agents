"""Deterministic unit tests for the s1 transport JSON-extraction interceptor and
the JSON-safety helpers it reuses (``_strip_fences`` / ``_extract_json``).

CONTEXT (s1/a2 + s1/b1): gemma4-e2b-agent is a thinking model. With ``think=True``
the chain-of-thought comes back in a SEPARATE ``message.thinking`` field and
``message.content`` is (usually) markdown-FENCED JSON. The transport interceptor in
``OllamaTransport._chat_native`` fence-strips + walks out the first balanced JSON
object so BOTH the chain sites AND the two DIRECT ``json.loads`` sites
(toolargs / tool_registry) receive clean JSON. CRITICAL invariant: a TRUNCATED reply
(CoT ate ``num_predict`` → unbalanced/empty JSON) must stay VISIBLE downstream as a
parse failure — it must NEVER be silently masked into something that looks valid.

These tests are fully OFFLINE: the real :class:`OllamaTransport` is exercised with a
monkeypatched ``_post`` returning canned native ``/api/chat`` payloads, so the
interceptor runs on the real code path with zero GPU. ``FakeTransport`` deliberately
does NOT route through the interceptor, so it cannot be used here.

No async plugin is assumed (matching the repo): ``transport.chat`` is synchronous.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

import pytest

from llm_framework.transport import OllamaTransport
from llm_framework.stages import _strip_fences, _extract_json


# A small structured schema (its presence is what flips the call to "structured" so
# the interceptor fires); the schema content is irrelevant to the offline _post.
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"nodes": {"type": "array"}},
    "required": ["nodes"],
}


def _transport_returning(message: Mapping[str, Any]) -> OllamaTransport:
    """A native OllamaTransport whose ``_post`` returns a canned ``{message: ...}``.

    No HTTP, no GPU — the interceptor in ``_chat_native`` runs on the real payload.
    """
    tp = OllamaTransport(api="native")
    tp._post = lambda path, body: {"message": dict(message), "done_reason": "stop"}  # type: ignore[assignment]
    return tp


# --------------------------------------------------------------------------- #
# Pure helpers: _strip_fences
# --------------------------------------------------------------------------- #

def test_strip_fences_removes_json_fence():
    fenced = '```json\n{"a": 1}\n```'
    assert _strip_fences(fenced) == '{"a": 1}'


def test_strip_fences_removes_bare_fence():
    fenced = '```\n{"a": 1}\n```'
    assert _strip_fences(fenced) == '{"a": 1}'


def test_strip_fences_noop_on_unfenced():
    assert _strip_fences('{"a": 1}') == '{"a": 1}'


# --------------------------------------------------------------------------- #
# Pure helpers: _extract_json (balanced-extraction + truncation -> None)
# --------------------------------------------------------------------------- #

def test_extract_json_pulls_object_out_of_prose():
    text = 'Here is the plan you asked for:\n{"nodes": [1, 2]}\nHope that helps!'
    assert _extract_json(text) == '{"nodes": [1, 2]}'


def test_extract_json_handles_braces_inside_strings():
    # A closing brace inside a string literal must NOT end the object early.
    text = '{"q": "a } b", "n": 1}'
    assert _extract_json(text) == '{"q": "a } b", "n": 1}'


def test_extract_json_handles_fenced_object():
    text = '```json\n{"nodes": [{"id": "x"}]}\n```'
    assert json.loads(_extract_json(text)) == {"nodes": [{"id": "x"}]}


def test_extract_json_returns_none_on_truncated_unbalanced():
    # CoT ate the budget: the JSON is cut off mid-object (no matching close).
    truncated = '```json\n{"nodes": [{"id": "x", "task": "do '
    assert _extract_json(truncated) is None


def test_extract_json_returns_none_on_empty():
    assert _extract_json("") is None
    assert _extract_json("   ") is None


# --------------------------------------------------------------------------- #
# The interceptor on the real _chat_native path (monkeypatched _post)
# --------------------------------------------------------------------------- #

def test_interceptor_strips_fence_on_structured_call():
    # The real gemma4 shape: fenced JSON in content, CoT in a SEPARATE field.
    tp = _transport_returning(
        {"role": "assistant",
         "content": '```json\n{"nodes": []}\n```',
         "thinking": "1. analyze 2. emit"}
    )
    res = tp.chat([{"role": "user", "content": "plan"}], api="native", format=_SCHEMA)
    # Content is clean JSON the two DIRECT json.loads sites can parse without fences.
    assert json.loads(res.content) == {"nodes": []}
    # The CoT is surfaced separately for observability and never pollutes content.
    assert res.thinking == "1. analyze 2. emit"


def test_interceptor_extracts_json_from_surrounding_prose():
    tp = _transport_returning(
        {"role": "assistant",
         "content": 'Sure! {"nodes": [{"id": "n1"}]} done.'}
    )
    res = tp.chat([{"role": "user", "content": "plan"}], api="native", format=_SCHEMA)
    assert json.loads(res.content) == {"nodes": [{"id": "n1"}]}


def test_interceptor_leaves_truncation_visible_not_masked():
    # The decisive safety invariant: a truncated reply must NOT be silently turned
    # into valid-looking JSON. _extract_json returns None (unbalanced) → the
    # interceptor falls back to fence-strip ONLY, so the still-malformed text reaches
    # the caller and json.loads raises (a VISIBLE parse failure downstream).
    truncated = '```json\n{"nodes": [{"id": "n1", "task": "gath'
    tp = _transport_returning({"role": "assistant", "content": truncated})
    res = tp.chat([{"role": "user", "content": "plan"}], api="native", format=_SCHEMA)
    # NOT masked to empty (empty would hide the failure as "no output"); the broken
    # text is preserved (fence-stripped) so the failure is diagnosable.
    assert res.content.startswith('{"nodes"')
    with pytest.raises(json.JSONDecodeError):
        json.loads(res.content)


def test_interceptor_skips_unstructured_call():
    # No format/json → not a structured call → content passed through verbatim
    # (the interceptor must not mangle ordinary prose replies).
    prose = "Here is a plain answer with a stray { brace and no JSON intent."
    tp = _transport_returning({"role": "assistant", "content": prose})
    res = tp.chat([{"role": "user", "content": "hi"}], api="native")
    assert res.content == prose


def test_interceptor_handles_format_json_string_mode():
    # format="json" (syntax-only mode) is also a structured call → interceptor fires.
    tp = _transport_returning(
        {"role": "assistant", "content": '```\n{"ok": true}\n```'}
    )
    res = tp.chat([{"role": "user", "content": "x"}], api="native", json=True)
    assert json.loads(res.content) == {"ok": True}
