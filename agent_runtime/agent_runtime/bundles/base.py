"""bundles.base — the OO tool-bundle FOUNDATION (d190).

A *bundle* is an object that exposes, for one capability domain, BOTH halves of
what a small model needs to act well:

* its **tools** — native tool schemas (the ``make_tool_spec`` shape the runtime
  already offers to the model as ``tools=[...]``), plus handler-backed
  :class:`~reactive_tools.tool_registry.ToolDef` s registered on demand; and
* its **doctrine** — the usage text that TEACHES the model how to operate those
  tools *together* (the loop, the discipline, the anti-fabrication rules).

:class:`ObjectBundle` is the BASE every categorized bundle EXTENDS (d190 #2). It
holds the ESSENTIAL, common surface — the object-level ``finish`` signal — and the
universal doctrine (*reason → call one tool → observe the REAL result → finish*).
A categorized bundle (research / writer / planning) adds its own tools + its own
doctrine ON TOP via :meth:`tool_specs` (calling ``super().tool_specs`` first) and
:attr:`own_doctrine`.

CRITICAL (d190): a bundle ORCHESTRATES the EXISTING tool functions
(:mod:`agent_runtime.research_tree`, :mod:`agent_runtime.plan_tools`,
:mod:`agent_runtime.source_tools`, :mod:`agent_runtime.discovery_tools`,
:mod:`agent_runtime.claim_verify`, :mod:`agent_runtime.synth_tools`) — it does NOT
reimplement them. The bundle TEXT is the SINGLE place a behaviour flavour lives
(d190/d191): a request gets a flavour ONLY if it LOADS the bundle that advertises
it, so behaviour is opt-in by bundle selection and cannot bleed onto other tasks.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from ..research_tree import make_tool_spec

# --------------------------------------------------------------------------- #
# The ESSENTIAL, common tool every node shares: the object-level finish signal.
# --------------------------------------------------------------------------- #
FINISH_TOOL = "finish"

_FINISH_SPEC: dict[str, Any] = make_tool_spec(
    "finish",
    "Signal this task is COMPLETE. Call it only AFTER you have produced the "
    "deliverable / finding and confirmed the REAL result (read back what you wrote, "
    "or the observations you actually gathered) — never from memory of what you "
    "intended. Give a one-line reason.",
    {"reason": {"type": "string"}},
    [],
)

# The universal doctrine every bundle inherits — the agentic-loop discipline + the
# RAW-content / lightweight-args rule (d49/d50.1). Kept SHORT (d38 prompt-quality).
_BASE_DOCTRINE = (
    "You are an autonomous agent operating a small, well-described TOOLSET. Work "
    "ITERATIVELY: reason → call ONE tool → observe the REAL result it returns → "
    "reason again → … → finish. Ground every step on what the tools ACTUALLY "
    "return, never on memory or assumption. The CONTENT you produce is RAW text "
    "(prose / HTML / code / CSV — whatever the task asks), never wrapped in JSON; "
    "tool calls carry only lightweight arguments (a filename, a query, a url, a "
    "done marker). When the task is genuinely done, call finish."
)


class ObjectBundle:
    """The base bundle: essential common tools + the universal agentic doctrine.

    Subclasses set :attr:`name` + :attr:`own_doctrine` and override
    :meth:`tool_specs` (calling ``super().tool_specs(ctx)`` to inherit the base
    tools) and/or :meth:`register` (to add handler-backed ToolDefs). The bundle is
    cheap + stateless — per-run binding (which source list, which configured tool
    names) is passed as the ``ctx`` mapping, not baked into the object."""

    #: The bundle's registry key (what ``get_bundle(name)`` resolves).
    name: str = "object"
    #: The categorized doctrine this bundle adds on top of the base doctrine.
    own_doctrine: str = ""
    #: A ONE-LINE advertisement (capability DOMAIN + doctrine summary) shown in the
    #: ``get_bundles`` catalog every node sees, so the model can REASON about which
    #: bundle to self-select (d221). Kept short + capability-framed (d186 — the
    #: description is the selection lever). The base ``object`` floor is always loaded,
    #: so it advertises itself as the universal default.
    summary: str = (
        "the always-on base floor — finish + the universal reason->act->observe loop "
        "(loaded for every node; you do not select it)."
    )

    # ------------------------------------------------------------------ #
    # doctrine
    # ------------------------------------------------------------------ #
    @property
    def base_doctrine(self) -> str:
        """The universal agentic-loop doctrine inherited by every bundle."""
        return _BASE_DOCTRINE

    @property
    def doctrine(self) -> str:
        """The bundle's full usage doctrine = base doctrine + this bundle's own.

        This is the text that teaches the model HOW to operate the bundle's tools
        together — what ``get_bundle(name).doctrine`` returns for a role to inject."""
        own = (self.own_doctrine or "").strip()
        return f"{self.base_doctrine}\n\n{own}" if own else self.base_doctrine

    # ------------------------------------------------------------------ #
    # tools (native schemas the model is offered)
    # ------------------------------------------------------------------ #
    def base_tool_specs(self) -> list[dict[str, Any]]:
        """The essential common tool schemas every bundle holds (object-level finish)."""
        return [dict(_FINISH_SPEC)]

    def tool_specs(self, ctx: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
        """The native tool schemas this bundle offers the model.

        BASE returns only the common tools; a categorized bundle returns
        ``super().tool_specs(ctx) + <its own>``. ``ctx`` carries per-run binding
        (e.g. the configured search/fetch/note names) when a tool's schema depends
        on it."""
        return self.base_tool_specs()

    def tool_names(self, ctx: Optional[Mapping[str, Any]] = None) -> list[str]:
        """The names the model sees for this bundle's offered tools."""
        return [s["function"]["name"] for s in self.tool_specs(ctx)]

    # ------------------------------------------------------------------ #
    # handler-backed tools (ToolDefs added to a GrowableToolRegistry)
    # ------------------------------------------------------------------ #
    def register(
        self, registry: Any, ctx: Optional[Mapping[str, Any]] = None
    ) -> Any:
        """Add this bundle's handler-backed :class:`ToolDef` s onto ``registry``.

        BASE registers nothing (its tools are native-schema-only). A categorized
        bundle that owns real handlers (e.g. the planning bundle's get_shapes /
        get_specs, the research bundle's load_source) overrides this to orchestrate
        the existing ``make_*`` / ``register_*`` factories. Returns the registry."""
        return registry

    # ------------------------------------------------------------------ #
    # tool OUTPUT-MESSAGE override (d221): a bundle's CONTEXT may OVERRIDE the
    # observation a BASE tool returns — e.g. the research context overrides
    # web_fetch's output to PROMPT take-a-note. The BASE (and a plain context)
    # overrides nothing, so a tool keeps its own message unless a LOADED bundle
    # chooses to extend it.
    # ------------------------------------------------------------------ #
    def tool_output_override(
        self, tool_name: str, ctx: Optional[Mapping[str, Any]] = None
    ) -> Optional[str]:
        """A suffix this bundle's context appends to ``tool_name``'s observation, or None.

        BASE returns None for every tool (no override). A categorized bundle overrides
        the message of a base tool it wraps when its doctrine needs the model prompted
        differently in that context (d221)."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<{type(self).__name__} name={self.name!r} "
            f"tools={self.tool_names()}>"
        )


__all__ = ["ObjectBundle", "FINISH_TOOL"]
