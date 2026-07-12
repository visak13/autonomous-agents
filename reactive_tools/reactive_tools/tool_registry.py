"""Growable, Pydantic-typed tool registry + native structured tool-calling dispatch.

This is the **s2/a1 SCAFFOLD** — the foundation every concrete tool (a2–a5) plugs
into. The Round-3 blueprint (RC5 / decision d1) calls for a *thin, growable,
Pydantic-typed tool registry* on top of the EXISTING engine
(:class:`~reactive_tools.tool_hook.ToolHook` + ``llm_framework.OllamaTransport``)
— NOT a monolithic agent framework, and NOT a rewrite of the working tool layer.

Three pieces, additive over the existing ``ToolHook`` seam:

1. :class:`ToolDef` — ONE registry entry. ``name`` + ``description`` + a
   **Pydantic args model** (``args_model``) + a ``handler`` callable. The args
   JSON schema (the ``required`` keys the model must fill, with types) is DERIVED
   from the Pydantic model — there is no second place to edit. *Adding a tool is
   exactly one ``ToolDef``* (the core of outcome o1's growability).

2. :class:`GrowableToolRegistry` — a by-name map of :class:`ToolDef`. ``add()`` is
   the single growth point: it records the def AND registers the handler onto the
   bound :class:`ToolHook`, so the new tool is immediately (a) **selectable** (it
   appears in the structured-selection enum) and (b) **dispatchable** (invocable
   through the hook, with its ``tool_call``/``tool_result`` events flowing on the
   event plane). No other code change is needed.

3. :class:`StructuredToolCaller` — the SINGLE place that OFFERS all registered
   tools to a node and DISPATCHES the model's chosen tool to its handler. One
   native ``/api/chat`` call with the proven Gemma structured-output settings
   (decision d1, s1/b1 reasoning rollout): ``think=True`` (TOP-LEVEL — gemma4
   reasons in the SEPARATE message.thinking field; ``num_predict`` is raised to
   4096 so the CoT cannot starve the JSON to EMPTY), ``temperature=0`` (deterministic
   selection), and ``format=<JSON schema>`` whose ``tool`` field is an **enum of
   the registered tool names** + ``required`` keys, with ``num_predict`` raised to
   fit the whole ``{tool, args}`` object. The chosen tool's args are then
   validated/coerced through that tool's Pydantic model BEFORE the handler runs,
   so the small local model can never drive a handler with a wrong-shape payload.

Decisions honored
------------------
- d1  — thin growable Pydantic registry on the existing ``llm_framework`` /
  ``reactive_tools``; the planner (not this module) owns control flow. No
  LangChain/LangGraph/PydanticAI/smolagents.
- spec — native structured outputs use a JSON SCHEMA in ``format`` (enum +
  required), NEVER ``format="json"``; ``think=True`` top-level (s1/b1, raised
  ``num_predict``); ``temperature=0``;
  ``num_predict`` raised to hold the whole object; per-request options only (the
  transport never mutates a global Ollama config).
- d2  — purely in-process: the registry/caller machinery crosses no process
  boundary. (A tool body may do its own I/O; that is a2–a5's concern.)

This module ships the registry + the dispatch path + ONE trivial smoke tool
(:data:`ECHO_TOOL`) so the path is exercised end-to-end on ``gemma4-e2b-agent``.
The six concrete tools land in a2–a5 — each is one :class:`ToolDef`.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError

from .event_plane import EventPlane
from .tool_hook import ToolHook, ToolRegistry, ToolResult, ToolSpec


class ToolRegistryError(RuntimeError):
    """A tool-registry / structured-dispatch problem: a bad :class:`ToolDef`,
    an unknown selected tool, args that fail the tool's Pydantic model, or a
    selection the model could not produce in the required shape."""


# --------------------------------------------------------------------------- #
# ToolDef — ONE registry entry (name + description + Pydantic args + handler)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolDef:
    """A single growable-registry entry — *adding a tool is exactly this object*.

    Attributes
    ----------
    name:
        The tool's unique name. This is the token the model selects (it is one of
        the enum values in the structured-selection schema) and the key the
        handler is dispatched by.
    description:
        A short one-liner shown to the model so it can decide whether to pick the
        tool. Kept lean (context-scoping): names + one-liners, never bodies.
    args_model:
        A **Pydantic** ``BaseModel`` subclass defining the tool's arguments. Its
        JSON schema (required fields + types) is what the model must satisfy, and
        it is the single source of truth — there is no separate hand-maintained
        schema table to keep in sync.
    handler:
        The callable that runs the tool. Invoked as ``handler(**validated_args)``
        where ``validated_args`` is the Pydantic-validated, model-dumped payload.
        May be sync or async — dispatch goes through :class:`ToolHook`, which runs
        sync bodies off the event loop and awaits coroutine bodies.
    """

    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[..., Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ToolRegistryError(f"ToolDef.name must be a non-empty str, got {self.name!r}")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ToolRegistryError(
                f"ToolDef.description must be a non-empty str (tool {self.name!r})")
        if not (isinstance(self.args_model, type) and issubclass(self.args_model, BaseModel)):
            raise ToolRegistryError(
                f"ToolDef.args_model must be a pydantic BaseModel subclass "
                f"(tool {self.name!r}), got {self.args_model!r}")
        if not callable(self.handler):
            raise ToolRegistryError(
                f"ToolDef.handler must be callable (tool {self.name!r}), got {self.handler!r}")

    def args_schema(self) -> dict[str, Any]:
        """The tool's argument JSON schema, derived from its Pydantic model.

        Used to TELL the model what to fill (in the selection prompt) and to
        document the tool. The required keys come straight from the model's
        required fields — no duplicate table."""
        return self.args_model.model_json_schema()

    def required_keys(self) -> list[str]:
        """The argument keys the model MUST fill, from the Pydantic schema."""
        return list(self.args_schema().get("required", []))

    def validate_args(self, raw: Optional[Mapping[str, Any]]) -> dict[str, Any]:
        """Validate/coerce ``raw`` through the Pydantic model → handler kwargs.

        Unknown keys are dropped (the model's default) and defaults are filled, so
        a small model emitting a slightly-off payload still yields a clean,
        in-shape kwargs dict — and can never smuggle a junk kwarg into the handler.
        Raises :class:`ToolRegistryError` (wrapping the Pydantic error) when a
        required arg is missing or mistyped, so the dispatch layer surfaces a
        structured failure the runtime/self-heal can react to."""
        try:
            model = self.args_model.model_validate(dict(raw or {}))
        except ValidationError as exc:
            raise ToolRegistryError(
                f"args for tool {self.name!r} failed validation: {exc.errors()}"
            ) from exc
        return model.model_dump()

    def catalog_row(self) -> dict[str, Any]:
        """A lean ``{name, description, args_schema, required}`` row for prompting."""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema(),
            "required": self.required_keys(),
        }


# --------------------------------------------------------------------------- #
# GrowableToolRegistry — add a ToolDef => selectable + dispatchable, no other change
# --------------------------------------------------------------------------- #


class GrowableToolRegistry:
    """A by-name map of :class:`ToolDef`, bound to a :class:`ToolHook`.

    The ONE growth point is :meth:`add`: it records the def and registers the
    handler onto the bound hook, so the tool is immediately selectable (in the
    enum) and dispatchable (through the hook, events on the plane). The registry
    is the source of truth for the structured-selection schema and the per-tool
    arg schemas.
    """

    def __init__(self, hook: ToolHook) -> None:
        if not isinstance(hook, ToolHook):
            raise ToolRegistryError(
                f"GrowableToolRegistry needs a ToolHook (event-plane dispatch seam), "
                f"got {hook!r}")
        self._hook = hook
        self._defs: dict[str, ToolDef] = {}
        # The DISPATCH store (A0): a base :class:`ToolRegistry` of dispatchable
        # ``ToolSpec`` s. We ADOPT the hook's EXISTING base registry so that the
        # static tools already registered on it (the core file/web tools and the
        # reactive-lambda tools registered by ``build_default_hook``) SURVIVE when
        # this growable becomes ``hook.registry`` — i.e. they stay dispatchable
        # through ``hook.invoke`` even though they are not structured-selectable
        # ``ToolDef`` s. Re-wrapping an existing growable reuses its base so the
        # spec store is never lost.
        base = getattr(hook, "registry", None)
        if isinstance(base, GrowableToolRegistry):
            base = base._base
        self._base: ToolRegistry = base if isinstance(base, ToolRegistry) else ToolRegistry()

    @property
    def hook(self) -> ToolHook:
        return self._hook

    # -- the single growth point ------------------------------------------ #

    def add(self, tool: ToolDef) -> ToolDef:
        """Register ``tool`` — the ONLY thing needed to add a tool.

        Records the def AND registers its handler onto the bound hook (so its
        invocation + result flow on the event plane). Re-adding a name replaces it
        (last writer wins) so a host can override a default tool. Returns the def."""
        if not isinstance(tool, ToolDef):
            raise ToolRegistryError(f"add() takes a ToolDef, got {tool!r}")
        # Record the handler in the dispatch store (a ToolSpec) AND the def in the
        # structured-selection map. Writing straight to ``_base`` is equivalent to
        # the old ``self._hook.register`` indirection (``hook.register`` delegates to
        # ``hook.registry`` which IS ``_base`` once this growable is assigned), and is
        # robust to whether the assign-back has happened yet.
        self._base.register(tool.name, tool.handler, description=tool.description)
        self._defs[tool.name] = tool
        return tool

    # -- dispatch / base passthroughs (so this can BE hook.registry, A0) -------- #

    def register(self, name: str, func, *, description: str = "") -> ToolSpec:
        """Register a raw handler as a dispatchable :class:`ToolSpec` (no ToolDef).

        The passthrough that keeps EXISTING ``hook.register`` call sites working once
        this growable is assigned to ``hook.registry``: ``hook.register`` delegates to
        ``self.registry.register``, which lands here and stores the spec in the base
        dispatch store. Such a tool is DISPATCHABLE (via :meth:`resolve` / the hook)
        but not structured-SELECTABLE (it has no :class:`ToolDef` arg schema) — exactly
        the right semantics for the core/lambda tools and the per-run write tools."""
        return self._base.register(name, func, description=description)

    def resolve(self, name: str) -> ToolSpec:
        """Resolve ``name`` to its dispatchable :class:`ToolSpec` (the hook's dispatch
        seam). Distinct from :meth:`get`, which returns the selection :class:`ToolDef`."""
        return self._base.resolve(name)

    def tool(
        self,
        name: str,
        *,
        description: str,
        args_model: type[BaseModel],
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form of :meth:`add` — returns the handler unchanged so it
        stays directly callable too."""

        def deco(handler: Callable[..., Any]) -> Callable[..., Any]:
            self.add(ToolDef(name=name, description=description,
                             args_model=args_model, handler=handler))
            return handler

        return deco

    # -- lookups ----------------------------------------------------------- #

    def get(self, name: str) -> ToolDef:
        try:
            return self._defs[name]
        except KeyError:
            raise ToolRegistryError(
                f"unknown tool {name!r}; registered: {self.names()}") from None

    def __contains__(self, name: object) -> bool:
        # Membership = DISPATCHABLE: a structured-selectable ToolDef OR a base
        # dispatch-only spec (core/lambda/per-run tools live only in ``_base``).
        return name in self._defs or name in self._base

    def __len__(self) -> int:
        return len(self._defs)

    def names(self) -> list[str]:
        # The structured-SELECTION names (the ``{tool, args}`` enum): only the
        # ToolDefs. Dispatch-only base tools are intentionally NOT selectable here.
        return sorted(self._defs)

    def catalog(self) -> list[dict[str, Any]]:
        """The lean ``[{name, description}]`` tool listing the planner/selector sees.

        Delegates to the base dispatch store so it is the FULL set of registered
        tools (the structured-selectable ToolDefs PLUS the dispatch-only core/lambda
        tools) — i.e. byte-identical to the pre-A0 ``hook.registry.catalog()`` when
        ``hook.registry`` was the base. Becoming ``hook.registry`` therefore does NOT
        shrink the planner's tool catalog; it only ADDS the working ``.add`` growth
        point. (The narrower structured-selection enum is :meth:`names` / :meth:`offered`.)"""
        return self._base.catalog()

    def arg_schemas(self, tool_names: Optional[Sequence[str]] = None) -> dict[str, Any]:
        """``{name: args_json_schema}`` for the offered tools (all if ``None``)."""
        return {n: self.get(n).args_schema() for n in self.offered(tool_names)}

    def offered(self, tool_names: Optional[Sequence[str]] = None) -> list[str]:
        """The names actually offered to the model: all registered, or the subset
        in ``tool_names`` that are registered (preserving the caller's order)."""
        if tool_names is None:
            return self.names()
        return [n for n in tool_names if n in self._defs]

    # -- the structured tool-SELECTION schema (Ollama-native ``format``) --- #

    def selection_schema(self, tool_names: Optional[Sequence[str]] = None) -> dict[str, Any]:
        """The JSON schema handed to Ollama ``format`` for ONE structured call.

        Constrains ``tool`` to an **enum of the registered tool names** and
        requires both the tool choice and an ``args`` object — so the model
        selects a real tool by name and emits its arguments in a single
        ``{tool, args}`` object. The per-tool argument shapes are conveyed in the
        prompt (and re-validated against the tool's Pydantic model after the call);
        ``args`` here is a free object because the required keys vary per tool."""
        offered = self.offered(tool_names)
        if not offered:
            raise ToolRegistryError("no tools registered to offer for selection")
        return {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "enum": offered,
                    "description": "the name of the single tool to call (must be one of the enum values)",
                },
                "args": {
                    "type": "object",
                    "description": "the arguments object for the chosen tool, matching that tool's arg schema",
                },
            },
            "required": ["tool", "args"],
        }


# --------------------------------------------------------------------------- #
# StructuredToolCaller — offer all tools, select one, dispatch it, return result
# --------------------------------------------------------------------------- #

# Default output budget for the selection call. The {tool, args} object is small,
# BUT s1/b1 enables ``think=True`` on the selection call (gemma4 reasons in the
# SEPARATE message.thinking field) and those thinking tokens compete with the content
# budget, so this is raised 512->4096 (the a2-proven load-bearing bump: at a small
# budget the CoT eats it and the JSON selection comes back EMPTY).
DEFAULT_SELECT_MAX_TOKENS = 4096


@dataclass(frozen=True)
class ToolSelection:
    """The model's structured choice from a :meth:`StructuredToolCaller.select`."""

    tool: str
    args: dict[str, Any]
    raw: str = ""


@dataclass(frozen=True)
class ToolCallResult:
    """The outcome of one :meth:`StructuredToolCaller.call` — selection + dispatch."""

    tool: str
    args: dict[str, Any]            # the validated args the handler ran with
    ok: bool
    value: Any = None
    error: Optional[str] = None
    selection: Optional[ToolSelection] = None   # the raw model choice (pre-validate)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "args": self.args,
            "ok": self.ok,
            "value": self.value,
            "error": self.error,
            "selection_raw": self.selection.raw if self.selection else None,
        }


class StructuredToolCaller:
    """Offer the registered tools to a node, let Gemma pick ONE, dispatch it.

    One native structured-output call selects ``{tool, args}`` (s1/b1 reasoning
    rollout: ``think=True`` top-level so gemma4 reasons in the SEPARATE
    message.thinking field, ``temperature=0``, ``format=<enum schema>``, raised
    ``num_predict`` so the CoT does not starve the JSON); the chosen tool's args
    are validated through its
    Pydantic model; the handler is dispatched through the bound :class:`ToolHook`
    (so the call + result are observable on the event plane); a structured
    :class:`ToolCallResult` is returned.
    """

    def __init__(
        self,
        transport: Any,
        registry: GrowableToolRegistry,
        *,
        max_tokens: int = DEFAULT_SELECT_MAX_TOKENS,
        temperature: float = 0.0,
        think: bool = True,
    ) -> None:
        if transport is None:
            raise ToolRegistryError("StructuredToolCaller needs a transport")
        if not isinstance(registry, GrowableToolRegistry):
            raise ToolRegistryError("StructuredToolCaller needs a GrowableToolRegistry")
        self.transport = transport
        self.registry = registry
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.think = think

    # -- prompt ------------------------------------------------------------ #

    def _messages(
        self,
        task: str,
        offered: Sequence[str],
        context: str,
    ) -> list[dict[str, str]]:
        """Build the (system, user) messages offering the tools + their arg shapes."""
        catalog = [self.registry.get(n).catalog_row() for n in offered]
        system = (
            "You select EXACTLY ONE tool to accomplish the task and emit its "
            "arguments. Respond ONLY as a JSON object {\"tool\": <one of the tool "
            "names>, \"args\": {<arguments for that tool>}}. The 'tool' MUST be one "
            "of the offered names. The 'args' MUST satisfy the chosen tool's arg "
            "schema (fill every required key). Reason it through privately first; "
            "your VISIBLE reply must be ONLY the JSON object — no prose, no code "
            "fences.\n\n"
            "OFFERED TOOLS (name, description, arg schema):\n"
            + json.dumps(catalog, indent=2)
        )
        user = (
            (f"CONTEXT:\n{context}\n\n" if context else "")
            + f"TASK: {task}\n\nReturn ONLY the JSON tool-call object."
        )
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    # -- structured selection (one native call) ---------------------------- #

    async def select(
        self,
        task: str,
        *,
        tool_names: Optional[Sequence[str]] = None,
        context: str = "",
    ) -> ToolSelection:
        """Make the ONE structured call and parse the model's ``{tool, args}``.

        Raises :class:`ToolRegistryError` if the model produces non-JSON, an
        object missing ``tool``, or a tool name outside the offered enum."""
        offered = self.registry.offered(tool_names)
        if not offered:
            raise ToolRegistryError("no tools offered for selection")
        schema = self.registry.selection_schema(tool_names)
        messages = self._messages(task, offered, context)
        # transport.chat is synchronous; run it off the event loop so the single
        # in-process loop is never stalled (d2). Native /api/chat carries the
        # top-level ``think`` control; ``format`` carries the enum schema.
        result = await asyncio.to_thread(
            lambda: self.transport.chat(
                messages,
                api="native",
                think=self.think,
                temperature=self.temperature,
                format=schema,
                max_tokens=self.max_tokens,
            )
        )
        raw = result.content or ""
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ToolRegistryError(
                f"tool-selection call did not return JSON: {raw!r}") from exc
        if not isinstance(parsed, Mapping):
            raise ToolRegistryError(f"tool-selection must be an object, got {parsed!r}")
        tool = parsed.get("tool")
        if not isinstance(tool, str) or not tool:
            raise ToolRegistryError(f"tool-selection missing a 'tool' name: {parsed!r}")
        if tool not in offered:
            raise ToolRegistryError(
                f"model selected tool {tool!r} not in offered {offered}")
        args = parsed.get("args") or {}
        if not isinstance(args, Mapping):
            raise ToolRegistryError(f"tool-selection 'args' must be an object, got {args!r}")
        return ToolSelection(tool=tool, args=dict(args), raw=raw)

    # -- select + validate + dispatch (the single OFFER->DISPATCH place) --- #

    async def call(
        self,
        task: str,
        *,
        tool_names: Optional[Sequence[str]] = None,
        context: str = "",
    ) -> ToolCallResult:
        """Offer the tools, select one, validate its args, dispatch the handler.

        The chosen tool's args are validated through its Pydantic model BEFORE the
        handler runs; the handler is invoked through the bound hook (events on the
        plane). Returns a structured :class:`ToolCallResult` carrying the handler's
        value (``ok=True``) or the failure (``ok=False``)."""
        selection = await self.select(task, tool_names=tool_names, context=context)
        tool_def = self.registry.get(selection.tool)
        # Validate/coerce the model's args through the tool's Pydantic schema —
        # a wrong-shape payload becomes a structured error, never a handler crash.
        try:
            args = tool_def.validate_args(selection.args)
        except ToolRegistryError as exc:
            return ToolCallResult(
                tool=selection.tool, args=dict(selection.args), ok=False,
                error=str(exc), selection=selection)
        # Dispatch through the hook (the single invoke seam; events on the plane).
        res: ToolResult = await self.registry.hook.invoke(selection.tool, **args)
        return ToolCallResult(
            tool=selection.tool, args=args, ok=res.ok, value=res.value,
            error=res.error, selection=selection)


# --------------------------------------------------------------------------- #
# The trivial smoke tool — exercises the whole path end-to-end (a1 deliverable)
# --------------------------------------------------------------------------- #


class EchoArgs(BaseModel):
    """Args for the :data:`ECHO_TOOL` smoke tool."""

    text: str = Field(..., description="the text to echo back verbatim")


def _echo(text: str) -> dict[str, Any]:
    """Echo ``text`` back — a trivial handler so the dispatch path is provable
    without any external dependency or side effect."""
    return {"echoed": text, "length": len(text)}


ECHO_TOOL = ToolDef(
    name="echo",
    description="Echo the given text back verbatim. A trivial smoke tool to exercise the tool-call path.",
    args_model=EchoArgs,
    handler=_echo,
)


@dataclass(frozen=True)
class ToolRuntime:
    """The wired pieces a host gets from :func:`build_tool_runtime`."""

    hook: ToolHook
    registry: GrowableToolRegistry
    caller: Optional[StructuredToolCaller]


def build_tool_runtime(
    plane: Optional[EventPlane] = None,
    *,
    hook: Optional[ToolHook] = None,
    transport: Any = None,
    register_smoke: bool = True,
    caller_opts: Optional[Mapping[str, Any]] = None,
) -> ToolRuntime:
    """Wire a :class:`GrowableToolRegistry` (+ optional caller) on a hook/plane.

    - ``hook`` is reused if given (so tool events land on the app's plane);
      otherwise a fresh :class:`ToolHook` on ``plane`` (or a new
      :class:`EventPlane`) is built.
    - ``register_smoke`` (default) registers :data:`ECHO_TOOL` so the path is
      immediately exercisable end-to-end.
    - When a ``transport`` is supplied, a :class:`StructuredToolCaller` is built
      (with ``caller_opts``); otherwise ``caller`` is ``None`` (registry-only).

    The six concrete tools (a2–a5) are added later with ``registry.add(ToolDef(...))``
    — one entry each, no change here.
    """
    if hook is None:
        hook = ToolHook(plane if plane is not None else EventPlane())
    registry = GrowableToolRegistry(hook)
    if register_smoke:
        registry.add(ECHO_TOOL)
    caller = (
        StructuredToolCaller(transport, registry, **dict(caller_opts or {}))
        if transport is not None else None
    )
    return ToolRuntime(hook=hook, registry=registry, caller=caller)


# --------------------------------------------------------------------------- #
# register_agentic_tools — compose the SIX s2 node→tool capabilities on a hook
# --------------------------------------------------------------------------- #

# The canonical node→tool surface a planner offers (s3/b5). Exactly the six s2
# registry buckets: web_search, web_fetch, file_read, file_write, the
# recipient-LOCKED send_mail, and the three cron tools. The legacy free-``to``
# ``send_email`` is deliberately ABSENT — a node may reach ONLY the locked mail
# tool (the d8 unattended-email safety invariant).
AGENTIC_TOOL_NAMES: tuple[str, ...] = (
    "web_search",
    "web_fetch",
    "file_read",
    "file_write",
    "send_mail",
    "cron_add",
    "cron_list",
    "cron_delete",
)


def register_agentic_tools(
    hook: ToolHook,
    *,
    file_base: Any = None,
    cron_db_path: Any = None,
    cron_data_dir: Any = None,
    smtp_config: Any = None,
    search_backend: Any = None,
    http_timeout: float = 20.0,
) -> GrowableToolRegistry:
    """Compose the SIX s2 node→tool capabilities onto ``hook`` (s3/b5).

    Builds a :class:`GrowableToolRegistry` bound to ``hook`` and registers the
    full node tool surface — ``web_search``, ``web_fetch``, ``file_read``,
    ``file_write`` (hard-sandboxed), the recipient-LOCKED ``send_mail``, and
    ``cron_add`` / ``cron_list`` / ``cron_delete`` — so a planner node can ANSWER
    via tools rather than raw LLM auto-completion. Each lands on the SAME hook
    (its calls + results ride the event plane), so the offered set is exactly
    :data:`AGENTIC_TOOL_NAMES`.

    SECURITY (d8 — the unattended-email safety invariant): the ONLY mail tool
    registered here is the recipient-hard-locked ``send_mail`` (its exposed schema
    carries no ``to`` field and its adapter always targets ``SMTP_FROM_EMAIL``).
    The legacy free-``to`` ``send_email`` is NEVER registered through this path,
    so a node can never emit an arbitrary recipient.

    Lazy imports keep the registry machinery free of any hard dependency on the
    concrete tool bodies (mirrors :func:`build_default_hook`). Returns the
    :class:`GrowableToolRegistry` (also reachable as the bound ``hook.registry``
    handlers)."""
    from .cron_tools import register_cron_tools
    from .file_tools import register_filesystem_tools
    from .send_mail_tool import register_send_mail
    from .web_tools import register_web_tools

    registry = GrowableToolRegistry(hook)
    register_web_tools(registry, search_backend=search_backend, timeout=http_timeout)
    register_filesystem_tools(registry, file_base)
    register_send_mail(registry, config=smtp_config)
    register_cron_tools(registry, cron_db_path, data_dir=cron_data_dir)
    # A0 FOUNDATION FIX: ASSIGN the growable back as ``hook.registry`` (the static
    # core/lambda tools already on the base were ADOPTED as the growable's dispatch
    # store at construction, so they survive the swap). This makes the docstring's
    # "also reachable as the bound hook.registry" TRUE and — crucially — gives every
    # bundle self-select path (runtime ``_load_bundle`` / ``get_bundles`` ->
    # ``expand_bundle`` -> ``registry.add``) a registry that actually HAS ``.add``,
    # so a self-selected bundle's handlers genuinely register + dispatch (d242/d265).
    hook.registry = registry
    return registry


__all__ = [
    "ToolRegistryError",
    "ToolDef",
    "GrowableToolRegistry",
    "StructuredToolCaller",
    "ToolSelection",
    "ToolCallResult",
    "ToolRuntime",
    "EchoArgs",
    "ECHO_TOOL",
    "build_tool_runtime",
    "DEFAULT_SELECT_MAX_TOKENS",
    "AGENTIC_TOOL_NAMES",
    "register_agentic_tools",
]
