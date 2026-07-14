"""Deterministic unit tests for the transport-level role normalization (d262 /
d199 / d202).

CONTEXT: gemma4-e2b-agent's Ollama chat template is PROMPT-ONLY — it renders
``system`` / ``user`` / ``assistant`` turns but IGNORES ``role: "tool"`` entirely.
Any observation the runtime hands back as a tool message (plan acks, file-write
confirmations, reviewer file slices, self-select acks, research note-acks) is
therefore INVISIBLE to the model. ``OllamaTransport.chat`` normalises every
inbound ``role: "tool"`` turn to ``role: "user"`` at one chokepoint, BEFORE the
message list is dispatched to either wire path (``_chat_openai`` /
``_chat_native``) — each of which copies the list verbatim and never inspects
roles. This fixes all ~13 call sites at once and cannot regress at a new site.

These tests are fully OFFLINE: the real :class:`OllamaTransport` is exercised with
a monkeypatched ``_post`` that CAPTURES the wire body, so we assert on exactly the
``messages`` that would hit Ollama, with zero GPU/HTTP.
"""
from __future__ import annotations

from typing import Any, Mapping

from llm_framework.transport import OllamaTransport


def _capturing_transport(api: str) -> tuple[OllamaTransport, list[dict[str, Any]]]:
    """An OllamaTransport whose ``_post`` records the sent body and returns a
    minimal valid response for the given wire flavour. Returns ``(tp, captured)``
    where ``captured`` accumulates each sent body dict."""
    tp = OllamaTransport(api=api)
    captured: list[dict[str, Any]] = []

    def _fake_post(path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        captured.append(dict(body))
        if path == "/v1/chat/completions":
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        return {"message": {"role": "assistant", "content": "ok"}, "done_reason": "stop"}

    tp._post = _fake_post  # type: ignore[assignment]
    return tp, captured


def _roles(body: Mapping[str, Any]) -> list[str]:
    return [m.get("role") for m in body["messages"]]


# --------------------------------------------------------------------------- #
# Both wire paths: role:"tool" must arrive as role:"user"
# --------------------------------------------------------------------------- #

def test_tool_role_rewritten_to_user_on_native_path():
    tp, captured = _capturing_transport("native")
    tp.chat(
        [
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": "calling a tool"},
            {"role": "tool", "content": "OBSERVATION: file written to out.md"},
        ],
        api="native",
    )
    assert len(captured) == 1
    roles = _roles(captured[0])
    # No tool role survives to the wire; the observation now rides as a user turn.
    assert "tool" not in roles
    assert roles[-1] == "user"
    # The observation CONTENT is preserved INSIDE the envelope (messaging-layer fix:
    # the wrap lets the model tell tool output from the user speaking).
    wire = captured[0]["messages"][-1]["content"]
    assert "OBSERVATION: file written to out.md" in wire
    assert wire.startswith("[TOOL RESULT]") and wire.rstrip().endswith("[/TOOL RESULT]")


def test_tool_role_rewritten_to_user_on_openai_path():
    tp, captured = _capturing_transport("openai")
    tp.chat(
        [
            {"role": "user", "content": "do the thing"},
            {"role": "tool", "content": "OBSERVATION: plan accepted"},
        ],
        api="openai",
    )
    assert len(captured) == 1
    roles = _roles(captured[0])
    assert "tool" not in roles
    assert roles[-1] == "user"
    wire = captured[0]["messages"][-1]["content"]
    assert "OBSERVATION: plan accepted" in wire
    assert wire.startswith("[TOOL RESULT]")


def test_multiple_tool_turns_all_rewritten():
    tp, captured = _capturing_transport("native")
    tp.chat(
        [
            {"role": "tool", "content": "obs 1"},
            {"role": "user", "content": "continue"},
            {"role": "tool", "content": "obs 2"},
        ],
        api="native",
    )
    roles = _roles(captured[0])
    assert "tool" not in roles
    # Order is preserved; both observations are now visible ENVELOPED user turns,
    # while the genuine user turn stays bare.
    contents = [m["content"] for m in captured[0]["messages"] if m["role"] == "user"]
    assert any("obs 1" in c and c.startswith("[TOOL RESULT]") for c in contents)
    assert any("obs 2" in c and c.startswith("[TOOL RESULT]") for c in contents)
    assert "continue" in contents  # user text: bare, never wrapped


# --------------------------------------------------------------------------- #
# Non-tool roles are untouched; the caller's list is never mutated.
# --------------------------------------------------------------------------- #

def test_non_tool_roles_unchanged():
    tp, captured = _capturing_transport("native")
    tp.chat(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ],
        api="native",
    )
    roles = _roles(captured[0])
    # system/user/assistant all survive (a leading system turn also carries the
    # injected agent identity, but its role stays "system").
    assert roles.count("user") == 1
    assert roles.count("assistant") == 1
    assert "system" in roles
    assert "tool" not in roles


def test_caller_message_list_not_mutated():
    tp, captured = _capturing_transport("native")
    original = [
        {"role": "user", "content": "u"},
        {"role": "tool", "content": "obs"},
    ]
    tp.chat(original, api="native")
    # A chain may reuse the list; the tool turn must still read role:"tool" after.
    assert original[1]["role"] == "tool"
    assert original[1]["content"] == "obs"
