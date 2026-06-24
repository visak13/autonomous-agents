"""Unit + integration coverage for the growable Pydantic tool registry (s2/a1).

Covers the three scaffold pieces:
  - :class:`ToolDef`              — Pydantic-typed entry: schema derivation + arg validation.
  - :class:`GrowableToolRegistry` — add-one-entry growth, selection schema enum, hook wiring.
  - :class:`StructuredToolCaller` — the OFFER -> select -> validate -> DISPATCH path,
    driven OFFLINE through ``FakeTransport`` (scripted ``{tool, args}`` JSON) so the
    machinery is proven deterministically with zero GPU; the live ``gemma4-e2b-agent``
    proof is reported separately via record_action_status (no evidence files, d6).

No async test plugin is assumed (matching the repo): async paths are driven through
``asyncio.run`` from plain sync tests.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel, Field

from llm_framework import FakeTransport
from reactive_tools import (
    ECHO_TOOL,
    EchoArgs,
    EventPlane,
    GrowableToolRegistry,
    StructuredToolCaller,
    ToolDef,
    ToolHook,
    ToolRegistryError,
    build_tool_runtime,
)
from reactive_tools.tool_hook import EVENT_TOOL_CALL, EVENT_TOOL_RESULT


# --------------------------------------------------------------------------- #
# ToolDef — Pydantic schema derivation + arg validation
# --------------------------------------------------------------------------- #


def test_tooldef_args_schema_and_required_from_pydantic():
    schema = ECHO_TOOL.args_schema()
    assert schema["type"] == "object"
    assert "text" in schema["properties"]
    assert ECHO_TOOL.required_keys() == ["text"]


def test_tooldef_validate_args_coerces_and_drops_unknown():
    # unknown keys dropped (can't smuggle a junk kwarg into the handler); required kept.
    out = ECHO_TOOL.validate_args({"text": "hi", "bogus": 1})
    assert out == {"text": "hi"}


def test_tooldef_validate_args_missing_required_raises():
    with pytest.raises(ToolRegistryError):
        ECHO_TOOL.validate_args({})


def test_tooldef_rejects_non_basemodel_args():
    with pytest.raises(ToolRegistryError):
        ToolDef(name="x", description="d", args_model=dict, handler=lambda **k: k)


def test_tooldef_rejects_empty_name_and_noncallable_handler():
    with pytest.raises(ToolRegistryError):
        ToolDef(name="", description="d", args_model=EchoArgs, handler=_noop)
    with pytest.raises(ToolRegistryError):
        ToolDef(name="x", description="d", args_model=EchoArgs, handler=123)


def _noop(**kwargs):
    return kwargs


# --------------------------------------------------------------------------- #
# GrowableToolRegistry — add one entry => selectable + dispatchable
# --------------------------------------------------------------------------- #


def _fresh_registry() -> GrowableToolRegistry:
    return GrowableToolRegistry(ToolHook(EventPlane()))


def test_add_one_entry_makes_tool_selectable_and_registered_on_hook():
    reg = _fresh_registry()
    reg.add(ECHO_TOOL)
    assert "echo" in reg
    assert reg.names() == ["echo"]
    # registered on the hook too (dispatch seam) — one add, both effects.
    assert "echo" in reg.hook.registry


def test_selection_schema_enumerates_registered_names():
    reg = _fresh_registry()
    reg.add(ECHO_TOOL)
    schema = reg.selection_schema()
    assert schema["properties"]["tool"]["enum"] == ["echo"]
    assert schema["required"] == ["tool", "args"]


def test_selection_schema_empty_registry_raises():
    with pytest.raises(ToolRegistryError):
        _fresh_registry().selection_schema()


def test_growability_new_tool_is_one_entry():
    """The core o1 claim: adding a tool is exactly one ToolDef — no other change."""
    reg = _fresh_registry()
    reg.add(ECHO_TOOL)

    class AddArgs(BaseModel):
        a: int = Field(..., description="first addend")
        b: int = Field(..., description="second addend")

    reg.add(ToolDef(name="add", description="Add two integers.",
                    args_model=AddArgs, handler=lambda a, b: {"sum": a + b}))
    # immediately selectable (enum) AND dispatchable (hook) — no other code touched.
    assert reg.selection_schema()["properties"]["tool"]["enum"] == ["add", "echo"]
    assert "add" in reg.hook.registry
    res = asyncio.run(reg.hook.invoke("add", a=2, b=3))
    assert res.ok and res.value == {"sum": 5}


def test_offered_subset_filters_to_registered():
    reg = _fresh_registry()
    reg.add(ECHO_TOOL)
    assert reg.offered(["echo", "nope"]) == ["echo"]
    assert reg.selection_schema(["echo"])["properties"]["tool"]["enum"] == ["echo"]


# --------------------------------------------------------------------------- #
# StructuredToolCaller — OFFER -> select -> validate -> DISPATCH (offline)
# --------------------------------------------------------------------------- #


def _runtime_with_script(*replies):
    transport = FakeTransport(list(replies))
    rt = build_tool_runtime(transport=transport)
    return rt, transport


def test_select_parses_tool_and_args_and_uses_native_structured_opts():
    rt, transport = _runtime_with_script(json.dumps({"tool": "echo", "args": {"text": "hello"}}))
    sel = asyncio.run(rt.caller.select("say hello"))
    assert sel.tool == "echo" and sel.args == {"text": "hello"}
    # s1/b1 reasoning rollout ship-path opts: native api, think=True top-level (gemma4
    # reasons in the SEPARATE message.thinking field), temp 0, schema format, and a
    # max_tokens raised to give the CoT headroom (a2-proven: a small budget truncates).
    opts = transport.calls[-1]["opts"]
    assert opts["api"] == "native"
    assert opts["think"] is True
    assert opts["temperature"] == 0
    assert opts["format"]["properties"]["tool"]["enum"] == ["echo"]
    assert opts["max_tokens"] >= 4096


def test_call_dispatches_handler_and_returns_structured_result():
    rt, _ = _runtime_with_script(json.dumps({"tool": "echo", "args": {"text": "world"}}))
    out = asyncio.run(rt.caller.call("echo world back"))
    assert out.ok is True
    assert out.tool == "echo"
    assert out.value == {"echoed": "world", "length": 5}
    assert out.args == {"text": "world"}


def test_call_emits_tool_events_on_the_plane():
    plane = EventPlane()
    transport = FakeTransport([json.dumps({"tool": "echo", "args": {"text": "x"}})])
    rt = build_tool_runtime(plane=plane, transport=transport)

    async def drive():
        seen = []
        sub = plane.subscribe([EVENT_TOOL_CALL, EVENT_TOOL_RESULT])
        out = await rt.caller.call("echo x")
        # drain what was published during the dispatch
        for _ in range(2):
            seen.append(await asyncio.wait_for(sub.__anext__(), timeout=1.0))
        return out, seen

    out, seen = asyncio.run(drive())
    assert out.ok
    kinds = [e.kind for e in seen]
    assert EVENT_TOOL_CALL in kinds and EVENT_TOOL_RESULT in kinds


def test_call_rejects_tool_outside_enum():
    rt, _ = _runtime_with_script(json.dumps({"tool": "rm_rf", "args": {}}))
    with pytest.raises(ToolRegistryError):
        asyncio.run(rt.caller.call("delete everything"))


def test_call_bad_args_returns_structured_failure_not_crash():
    # model picks echo but omits the required 'text' -> validation failure surfaced
    # as ok=False (a healable signal), NOT a handler crash.
    rt, _ = _runtime_with_script(json.dumps({"tool": "echo", "args": {}}))
    out = asyncio.run(rt.caller.call("echo nothing"))
    assert out.ok is False
    assert "validation" in (out.error or "").lower()


def test_select_non_json_raises():
    rt, _ = _runtime_with_script("not json at all")
    with pytest.raises(ToolRegistryError):
        asyncio.run(rt.caller.select("anything"))


def test_build_tool_runtime_registers_smoke_tool_by_default():
    rt = build_tool_runtime()
    assert "echo" in rt.registry
    assert rt.caller is None  # no transport -> registry-only

    rt2 = build_tool_runtime(register_smoke=False)
    assert "echo" not in rt2.registry
