"""Tool hook + registry — the single seam every tool is invoked through.

This is the ``reactive_tools`` tool layer (a2, on top of the a1 event plane).
The design doc calls for **composable tools registered by name, invoked through
ONE hook entrypoint**, where *each invocation and its result are emitted as
events on the event plane* so results flow back on the reactive plane and the
planner / agent runtime can react to them.

Shape
-----
- :class:`ToolRegistry` — a name -> callable map. Tools are registered by name
  (``register`` or the ``@registry.tool(...)`` decorator) and looked up by name.
  Registering by name is what makes tools *composable*: a tool body can call
  ``await hook.invoke("other_tool", ...)`` to build on another tool, and the
  planner only needs the *names* (context-scoping — it never carries tool
  bodies).
- :class:`ToolHook` — wraps a registry + an :class:`~reactive_tools.event_plane.EventPlane`.
  Its :meth:`invoke` is the ONE entrypoint. Every call:
    1. publishes a ``tool_call`` event ``{call_id, name, args}``,
    2. runs the tool (sync bodies go through :func:`asyncio.to_thread` so
       blocking file/HTTP I/O never stalls the single in-process event loop;
       coroutine tools are awaited directly),
    3. publishes a ``tool_result`` event ``{call_id, name, ok, value|error}``,
    4. returns a :class:`ToolResult`.
  ``call_id`` correlates the call with its result on the plane.

IN-PROCESS CONSTRAINT (d2 — load-bearing)
-----------------------------------------
Purely in-process: asyncio + the in-memory event plane only. NO broker/pool
HTTP, no sockets, no subprocess, no Claude. The tools themselves may do file or
network I/O (that is their job), but the *hook/registry/plane* machinery never
crosses a process boundary. The hook is the reactive seam; d8 (no shell-command
anything) is honored — no tool shells out.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Optional

from .event_plane import EventPlane

# A tool is any callable ``(**kwargs) -> value``. It may be a plain sync
# function or an async coroutine function; the hook handles both. Keeping the
# signature kwargs-only keeps call sites self-documenting and JSON-shaped (what
# a phi-driven planner emits) rather than positional-and-fragile.
ToolFunc = Callable[..., Any]

# Event kinds published on the plane for every invocation.
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"


class ToolError(RuntimeError):
    """Raised for tool-layer problems: unknown tool name, or (via
    :meth:`ToolResult.unwrap`) a tool that failed."""


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool: its name, callable, and a one-line description.

    The ``description`` is the *only* thing the planner needs to see to decide
    whether to use the tool — it is deliberately short so phi's context stays
    lean (context-scoping is a primary design constraint)."""

    name: str
    func: ToolFunc
    description: str = ""
    is_async: bool = False


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one :meth:`ToolHook.invoke` call.

    ``ok`` is ``True`` with ``value`` set on success, or ``False`` with
    ``error`` (the exception's message) on failure. ``call_id`` matches the
    ``tool_call``/``tool_result`` events emitted on the plane."""

    name: str
    ok: bool
    call_id: int
    value: Any = None
    error: Optional[str] = None

    def unwrap(self) -> Any:
        """Return ``value`` on success, or raise :class:`ToolError` on failure.

        Lets a call site choose between inspecting the result (default) and
        fail-fast semantics, without the hook ever swallowing the event flow."""
        if not self.ok:
            raise ToolError(f"tool {self.name!r} failed: {self.error}")
        return self.value


class ToolRegistry:
    """A by-name map of composable tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        func: ToolFunc,
        *,
        description: str = "",
    ) -> ToolSpec:
        """Register ``func`` under ``name``. Re-registering a name replaces it
        (last writer wins) so a host can override a default tool."""
        if not name or not isinstance(name, str):
            raise ValueError(f"tool name must be a non-empty str, got {name!r}")
        spec = ToolSpec(
            name=name,
            func=func,
            description=description,
            is_async=asyncio.iscoroutinefunction(func),
        )
        self._tools[name] = spec
        return spec

    def tool(self, name: str, *, description: str = "") -> Callable[[ToolFunc], ToolFunc]:
        """Decorator form of :meth:`register` — returns the function unchanged
        so it stays directly callable too."""

        def deco(func: ToolFunc) -> ToolFunc:
            self.register(name, func, description=description)
            return func

        return deco

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(
                f"unknown tool {name!r}; registered: {sorted(self._tools)}"
            ) from None

    def resolve(self, name: str) -> ToolSpec:
        """Resolve ``name`` to its dispatchable :class:`ToolSpec`.

        The DISPATCH seam :class:`ToolHook` uses to find a callable spec — kept
        SEPARATE from :meth:`get` so a registry whose ``get`` returns a different
        record (the :class:`~reactive_tools.tool_registry.GrowableToolRegistry`
        returns a selection ``ToolDef`` from ``get``) can still serve dispatch.
        For the base registry ``resolve`` is exactly ``get``."""
        return self.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def catalog(self) -> list[Mapping[str, str]]:
        """The lean ``[{name, description}]`` listing the planner sees — names +
        one-liners only, never tool bodies (context-scoping)."""
        return [
            {"name": s.name, "description": s.description}
            for s in sorted(self._tools.values(), key=lambda s: s.name)
        ]


class ToolHook:
    """The single entrypoint every tool is invoked through.

    Binds a :class:`ToolRegistry` to an :class:`EventPlane`. Invoking a tool
    emits a ``tool_call`` then a ``tool_result`` event on the plane, so every
    tool use (and its outcome) is observable on the reactive plane."""

    def __init__(self, plane: EventPlane, registry: Optional[ToolRegistry] = None) -> None:
        self.plane = plane
        self.registry = registry if registry is not None else ToolRegistry()
        self._call_seq = 0

    # -- registration passthroughs (convenience) -------------------------- #
    def register(self, name: str, func: ToolFunc, *, description: str = "") -> ToolSpec:
        return self.registry.register(name, func, description=description)

    def tool(self, name: str, *, description: str = "") -> Callable[[ToolFunc], ToolFunc]:
        return self.registry.tool(name, description=description)

    def _next_call_id(self) -> int:
        self._call_seq += 1
        return self._call_seq

    # -- the ONE entrypoint ----------------------------------------------- #
    async def invoke(self, name: str, /, **kwargs: Any) -> ToolResult:
        """Invoke the tool registered as ``name`` with keyword ``kwargs``.

        Always emits ``tool_call`` (before) and ``tool_result`` (after) on the
        plane. Returns a :class:`ToolResult`; a failing tool yields ``ok=False``
        (the error is on the event AND the result) rather than propagating —
        call ``.unwrap()`` for raise-on-error semantics. The lookup itself
        raises :class:`ToolError` for an unknown name (that is a programming
        error, not a tool failure, so it is not turned into an event)."""
        spec = self.registry.resolve(name)  # raises ToolError on unknown name
        call_id = self._next_call_id()

        await self.plane.publish(
            EVENT_TOOL_CALL,
            {"call_id": call_id, "name": name, "args": dict(kwargs)},
            source=f"tool:{name}",
        )

        try:
            if spec.is_async:
                value = await spec.func(**kwargs)
            else:
                # Run blocking tool bodies off the event loop so file/HTTP I/O
                # never stalls the single in-process loop (d2).
                value = await asyncio.to_thread(lambda: spec.func(**kwargs))
            result = ToolResult(name=name, ok=True, call_id=call_id, value=value)
        except Exception as exc:  # noqa: BLE001 - surface ANY tool failure as an event
            result = ToolResult(
                name=name, ok=False, call_id=call_id, error=f"{type(exc).__name__}: {exc}"
            )

        await self.plane.publish(
            EVENT_TOOL_RESULT,
            {
                "call_id": call_id,
                "name": name,
                "ok": result.ok,
                "value": result.value,
                "error": result.error,
            },
            source=f"tool:{name}",
        )
        return result

    def invoke_sync(self, name: str, /, **kwargs: Any) -> ToolResult:
        """Synchronous invoke for plain (non-async) call sites.

        Uses :meth:`EventPlane.publish_nowait` so it never awaits — subscribers
        pick the events up on their next loop turn. The tool body runs inline
        (no thread hop) since the caller is already synchronous."""
        spec = self.registry.resolve(name)
        if spec.is_async:
            raise ToolError(
                f"tool {name!r} is async; use `await hook.invoke({name!r}, ...)`"
            )
        call_id = self._next_call_id()
        self.plane.publish_nowait(
            EVENT_TOOL_CALL,
            {"call_id": call_id, "name": name, "args": dict(kwargs)},
            source=f"tool:{name}",
        )
        try:
            value = spec.func(**kwargs)
            result = ToolResult(name=name, ok=True, call_id=call_id, value=value)
        except Exception as exc:  # noqa: BLE001
            result = ToolResult(
                name=name, ok=False, call_id=call_id, error=f"{type(exc).__name__}: {exc}"
            )
        self.plane.publish_nowait(
            EVENT_TOOL_RESULT,
            {
                "call_id": call_id,
                "name": name,
                "ok": result.ok,
                "value": result.value,
                "error": result.error,
            },
            source=f"tool:{name}",
        )
        return result


def build_default_hook(
    plane: EventPlane,
    *,
    file_base: Any = None,
    http_timeout: float = 20.0,
    meta_plane: Any = None,
    smtp_config: Any = None,
) -> ToolHook:
    """Construct a :class:`ToolHook` pre-loaded with the core tools + the reactive
    lambda capability.

    ``file_base`` is the allowed root for the file tools (path-traversal guard);
    see :mod:`reactive_tools.tools`. The reactive-lambda tools (create / compose /
    list / close subscriptions) are registered into the SAME global hook (d12 —
    every agent / LLM call can reach them), backed by a
    :class:`~reactive_tools.subscriptions.LambdaRegistry` bound to ``plane`` and a
    separate ``meta_plane`` (the read-only live-subscriptions channel; a fresh one
    is created if not supplied). The registry is attached as ``hook.subscriptions``
    so a host can serve the observe-only UI surface. Imported lazily so the
    hook/registry machinery has no hard dependency on the concrete tool bodies.

    MAIL (d8 — the unattended-email safety invariant): the legacy free-``to``
    ``send_email`` tool is DELIBERATELY NOT registered here. It accepts an
    arbitrary recipient, so exposing it on the global hook would let a model emit
    an arbitrary ``to`` (the recipient hard-lock would be bypassable). The ONLY
    mail capability a node reaches is the recipient-LOCKED ``send_mail``, composed
    via :func:`register_agentic_tools` on the node→tool wiring path (s3/b5).
    ``smtp_config`` is retained for signature/back-compat and is threaded to that
    locked mail tool by the wiring layer, not to a free-``to`` tool here."""
    from .lambda_tools import register_lambda_tools
    from .subscriptions import LambdaRegistry
    from .tools import register_core_tools

    hook = ToolHook(plane)
    register_core_tools(hook, file_base=file_base, http_timeout=http_timeout)
    registry = LambdaRegistry(plane, meta_plane=meta_plane)
    register_lambda_tools(hook, registry)
    return hook


__all__ = [
    "ToolFunc",
    "ToolError",
    "ToolSpec",
    "ToolResult",
    "ToolRegistry",
    "ToolHook",
    "build_default_hook",
    "EVENT_TOOL_CALL",
    "EVENT_TOOL_RESULT",
]
