"""Cross-component bridge — serve the reactive ToolHook as llm_framework's tool_hook stage.

This module performs the **REAL cross-component import** that the integration
step (a3) exists to prove. ``reactive_tools`` imports ``llm_framework``'s chain
contract DIRECTLY — *not* a mocked or hand-copied interface. Both packages are
uv WORKSPACE MEMBERS resolved into the ONE shared root ``.venv`` (d11), so they
load in the SAME interpreter and this import resolves to the live
``llm_framework`` module object (d2 — everything in one in-process process).

What it does
------------
``llm_framework`` ships a ``tool_hook`` *seam* (``llm_framework.stages.tool_hook``)
— a documented no-op stage holding a stable slot in the chain so the reactive
layer (this component) can splice real behaviour in WITHOUT reshaping the
pipeline. This bridge:

1. Imports llm_framework's chain contract (``Chain`` / ``Context`` / ``Stage``)
   and the ``tool_hook`` seam itself — the contract reactive_tools fills.
2. Wraps a reactive :class:`~reactive_tools.tool_hook.ToolHook` as a chain
   :data:`~llm_framework.chain.Stage` (``(ctx) -> ctx``). When a tool call is
   requested on the context, the stage dispatches it through the in-process
   reactive hook (so the invocation + result flow as events on the event plane)
   and writes the result back onto the context.
3. :func:`install_reactive_tool_hook` splices that stage into a live ``Chain``
   in place of the no-op seam — the reactive hook then literally IS the chain's
   tool-hook stage, in one Python process.

A tool request is read from ``ctx.vars['tool_request']`` (``{"name", "args"}``)
if a stage/planner set one, else from a structured model reply of the shape
``{"tool": <name>, "args": {...}}`` (so phi can *decide* to call a tool and the
reactive hook executes it). No request this turn → the stage passes the context
through untouched, exactly like the seam it replaces.

IN-PROCESS (d2): the dispatch uses :meth:`ToolHook.invoke_sync` so it composes
with the synchronous ``Chain.run`` without a thread/loop hop. The event plane is
in-memory; nothing here crosses a process boundary.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

# ── THE REAL CROSS-COMPONENT IMPORT (d11/d2) ─────────────────────────────────
# reactive_tools importing llm_framework's chain contract DIRECTLY. This is the
# live sibling workspace package in the shared .venv, not a stub. If this import
# resolves, the two components are genuinely composable in one interpreter.
from llm_framework.chain import Chain, Context, Stage
from llm_framework.stages import tool_hook as _llm_tool_hook_seam

from .tool_hook import ToolHook, ToolResult

# The name of llm_framework's seam stage we fill (kept in sync via the import
# above — _SEAM_STAGE_NAME is derived from the real seam factory, not guessed).
_SEAM_STAGE_NAME = _llm_tool_hook_seam().__name__  # -> "tool_hook"

# ctx.vars slots the bridge reads/writes.
TOOL_REQUEST_KEY = "tool_request"  # {"name": str, "args": {...}} — set by a stage/planner
TOOL_RESULT_KEY = "tool_result"    # the bridge writes the ToolResult here


def _coerce_request(ctx: Context) -> Optional[dict[str, Any]]:
    """Find a tool-call request on the context, or ``None`` if none this turn.

    Two sources, in priority order:
      1. an explicit ``ctx.vars['tool_request']`` (a stage/planner asked for a tool);
      2. a structured model reply shaped ``{"tool": <name>, "args": {...}}`` —
         i.e. phi *decided* to call a tool through its JSON output.
    """
    req = ctx.get(TOOL_REQUEST_KEY)
    if req is None and isinstance(ctx.structured, Mapping) and "tool" in ctx.structured:
        req = {"name": ctx.structured.get("tool"),
               "args": ctx.structured.get("args", {})}
    if not req:
        return None
    name = req.get("name") or req.get("tool")
    args = req.get("args") or req.get("arguments") or {}
    if not name:
        return None
    return {"name": name, "args": dict(args)}


def reactive_tool_stage(hook: ToolHook) -> Stage:
    """Wrap a reactive :class:`ToolHook` as an llm_framework chain :data:`Stage`.

    Returns a ``(ctx: Context) -> Context`` callable — exactly the shape
    ``Chain.use``/``insert_*`` accept — that, when the context carries a tool
    request, dispatches it through ``hook`` (emitting the ``tool_call`` /
    ``tool_result`` events on the plane) and records the outcome on the context:
    ``ctx.vars['tool_result']`` plus a ``ctx.meta['tool_invocations']`` trail.
    """

    def tool_hook(ctx: Context) -> Context:  # name matches the seam it replaces
        req = _coerce_request(ctx)
        if req is None:
            return ctx  # no tool requested — pass-through, like the seam
        result: ToolResult = hook.invoke_sync(req["name"], **req["args"])
        ctx.set(TOOL_RESULT_KEY, {
            "name": result.name,
            "ok": result.ok,
            "call_id": result.call_id,
            "value": result.value,
            "error": result.error,
        })
        ctx.meta.setdefault("tool_invocations", []).append({
            "name": result.name,
            "call_id": result.call_id,
            "ok": result.ok,
        })
        return ctx

    return tool_hook


def install_reactive_tool_hook(
    chain: Chain,
    hook: ToolHook,
    *,
    after: Optional[str] = None,
) -> Chain:
    """Splice the reactive hook into ``chain``'s tool-hook slot, in place.

    Removes llm_framework's no-op ``tool_hook`` seam (if present) and inserts the
    reactive stage so the reactive hook IS the chain's tool-hook stage. The stage
    is placed AFTER the model speaks (the documented convention — the model emits
    a tool request, then the tool runs): by default AFTER ``structured_output``
    if the chain has it (so a parsed ``{"tool":...}`` reply is available), else
    AFTER ``call_stage``, else appended. Pass ``after`` to override. Returns the
    chain for fluent chaining.
    """
    stage = reactive_tool_stage(hook)
    names = chain.stage_names
    if _SEAM_STAGE_NAME in names:
        chain.remove(_SEAM_STAGE_NAME)
        names = chain.stage_names
    target = after
    if target is None:
        if "structured_output" in names:
            target = "structured_output"
        elif "call_stage" in names:
            target = "call_stage"
    if target is not None and target in names:
        chain.insert_after(target, stage, name=_SEAM_STAGE_NAME)
    else:
        chain.use(stage, name=_SEAM_STAGE_NAME)
    return chain


__all__ = [
    "reactive_tool_stage",
    "install_reactive_tool_hook",
    "TOOL_REQUEST_KEY",
    "TOOL_RESULT_KEY",
]
