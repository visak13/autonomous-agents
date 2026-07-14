"""Messaging-layer fix — the OBSERVATION ENVELOPE at the transport chokepoint.

The owner's finding: rewriting role:tool→user (d262) made every tool observation
INDISTINGUISHABLE from a user request on this prompt-only model. The fix: the same
one-chokepoint rewrite now WRAPS the rewritten content in [TOOL RESULT]…[/TOOL RESULT]
and the agent identity declares the convention — so the model can tell an observation
it asked for from the user speaking. These tests are fully OFFLINE (monkeypatched
``_post`` captures the wire body).
"""
from __future__ import annotations

from typing import Any, Mapping

from llm_framework.transport import (
    AGENT_IDENTITY,
    OBS_ENVELOPE_CLOSE,
    OBS_ENVELOPE_OPEN,
    OllamaTransport,
)


def _capturing_transport(api: str = "native") -> tuple[OllamaTransport, list[dict[str, Any]]]:
    tp = OllamaTransport(api=api)
    captured: list[dict[str, Any]] = []

    def _fake_post(path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        captured.append(dict(body))
        if path == "/v1/chat/completions":
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        return {"message": {"role": "assistant", "content": "ok"}, "done_reason": "stop"}

    tp._post = _fake_post  # type: ignore[assignment]
    return tp, captured


def _wire_messages(captured: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assert captured, "no wire body captured"
    return list(captured[-1]["messages"])


def test_tool_turn_is_wrapped_and_rewritten() -> None:
    tp, captured = _capturing_transport()
    tp.chat([
        {"role": "user", "content": "find the figure"},
        {"role": "tool", "content": "SEARCH RESULTS\n- https://example.org"},
    ])
    msgs = _wire_messages(captured)
    tool_turn = msgs[-1]
    assert tool_turn["role"] == "user"  # visible to the prompt-only template
    assert tool_turn["content"].startswith(OBS_ENVELOPE_OPEN)
    assert tool_turn["content"].rstrip().endswith(OBS_ENVELOPE_CLOSE)
    assert "SEARCH RESULTS" in tool_turn["content"]


def test_user_and_assistant_turns_never_wrapped() -> None:
    tp, captured = _capturing_transport()
    tp.chat([
        {"role": "user", "content": "please fetch it"},
        {"role": "assistant", "content": "on it"},
        {"role": "user", "content": "thanks"},
    ])
    for m in _wire_messages(captured):
        if m.get("role") == "system":
            continue  # the identity legitimately DECLARES the marker convention
        assert OBS_ENVELOPE_OPEN not in str(m.get("content") or "")


def test_wrap_is_idempotent() -> None:
    tp, captured = _capturing_transport()
    already = f"{OBS_ENVELOPE_OPEN}\nalready wrapped\n{OBS_ENVELOPE_CLOSE}"
    tp.chat([{"role": "tool", "content": already}])
    tool_turn = _wire_messages(captured)[-1]
    assert tool_turn["content"].count(OBS_ENVELOPE_OPEN) == 1
    assert tool_turn["content"] == already


def test_identity_declares_the_convention() -> None:
    assert OBS_ENVELOPE_OPEN in AGENT_IDENTITY
    assert "never a request" in AGENT_IDENTITY or "never wrapped" in AGENT_IDENTITY


def test_openai_wire_path_also_wraps() -> None:
    tp, captured = _capturing_transport(api="openai")
    tp.chat([{"role": "tool", "content": "FETCHED body"}])
    tool_turn = _wire_messages(captured)[-1]
    assert tool_turn["role"] == "user"
    assert tool_turn["content"].startswith(OBS_ENVELOPE_OPEN)
