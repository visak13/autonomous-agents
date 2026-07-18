"""agent_runtime.bundles — the OO tool-bundle architecture (d190-d194 + d212 redraw).

``get_bundle(name)`` returns a BUNDLE object exposing, for one CAPABILITY DOMAIN,
``{tools + doctrine}`` — the tools to act with and the doctrine that teaches the model
how to operate them together. A BUNDLE IS A TOOL WRAPPER, NOT A ROLE (d212): it never
plans / researches / writes / reviews; its SOLE purpose is MEMORY MANAGEMENT — a node
loads ONLY the bundle(s) its task needs, so each node's context stays lean (E4B
determinism, d192). Every categorized bundle EXTENDS the base
:class:`~agent_runtime.bundles.base.ObjectBundle` (which holds the essential common
``finish`` tool + the universal agentic-loop doctrine).

A NODE (a role, d213) COMPOSES bundles (d212 #1): NODE-SELF-SELECT (d221) — the node
SELF-SELECTS the bundle(s) its task needs AT RUNTIME (the ``get_bundles`` tool /
:meth:`agent_runtime.runtime.SubAgent._load_bundle`, tracked in its ``_loaded_bundles``
set, ``object`` floor always on), and the runtime unions the loaded bundles' tools +
doctrine and offers that union to the model. There is NO hardcoded role -> bundle table.
No bundle exposes role-phase / actor methods (d212 #2): each exposes ONE
``tool_specs(ctx)`` + ``doctrine`` (+ an optional ``tool_output_override``, d221), and the
runtime/role selects whatever subset a phase needs.

Bundles ORCHESTRATE the existing tool functions (research_tree / plan_tools / source_tools /
discovery_tools / claim_verify / synth_tools) — they reimplement nothing. A behaviour
FLAVOUR lives ONLY in the bundle text (d190/d191): a request gets it only by LOADING that
bundle.

Registered CAPABILITY-DOMAIN bundles (d212 redraw):
  * ``object``        — the base (finish + the universal agentic-loop doctrine)
  * ``planning``      — shape/spec discovery + tool-driven DAG authoring
  * ``research``      — GATHER evidence: search/fetch/note + tree-decision + cross-verify
  * ``research_read`` — READ a fetched source's verbatim text on demand (load_source)
  * ``file``          — author a deliverable with generic, format-agnostic file tools
"""
from __future__ import annotations

from .base import FINISH_TOOL, ObjectBundle
from .codebase import CodebaseReadBundle
from .file import FileBundle
from .planning import PlanningBundle
from .research import ResearchBundle
from .research_read import ResearchReadBundle

# Bundle-name constants (use these instead of bare strings so a typo is a NameError).
BUNDLE_OBJECT = "object"
BUNDLE_PLANNING = "planning"
BUNDLE_RESEARCH = "research"
BUNDLE_RESEARCH_READ = "research_read"
BUNDLE_FILE = "file"
BUNDLE_CODEBASE = "codebase"

# The registry: name -> a SINGLETON bundle instance (bundles are stateless; per-run
# binding flows through ``ctx`` / the ``make_*`` factory args, never instance state).
_REGISTRY: dict[str, ObjectBundle] = {
    BUNDLE_OBJECT: ObjectBundle(),
    BUNDLE_PLANNING: PlanningBundle(),
    BUNDLE_RESEARCH: ResearchBundle(),
    BUNDLE_RESEARCH_READ: ResearchReadBundle(),
    BUNDLE_FILE: FileBundle(),
    BUNDLE_CODEBASE: CodebaseReadBundle(),
}


class UnknownBundleError(KeyError):
    """``get_bundle`` was asked for a name no bundle is registered under."""


def get_bundle(name: str) -> ObjectBundle:
    """Return the bundle registered under ``name`` (``{tools + doctrine}``).

    Falls through to a clear :class:`UnknownBundleError` listing the registered names
    so a mis-wired role surfaces loudly rather than silently degrading. An empty /
    'none' name resolves to the base :data:`ObjectBundle` (the sensible default for a
    node with no specialized capability domain)."""
    key = str(name or "").strip().lower()
    if not key or key == "none":
        return _REGISTRY[BUNDLE_OBJECT]
    try:
        return _REGISTRY[key]
    except KeyError:
        raise UnknownBundleError(
            f"unknown bundle {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def compose_doctrine(names) -> str:
    """The UNION doctrine for a SET of bundle names (d212 #1): each loaded capability's
    OWN doctrine, de-duplicated and in a stable order. CoT-autonomy P1: the base
    agentic-loop protocol is NO LONGER folded here — its single owner is the node's
    SYSTEM turn (``AGENT_OPERATING_PROTOCOL`` via ``_compose_system``), so the load
    observation delivers only the capability's domain knowledge (one-owner rule)."""
    bundles = [get_bundle(n) for n in sorted(set(names or []))]
    if not bundles:
        bundles = [_REGISTRY[BUNDLE_OBJECT]]
    parts: list[str] = []
    seen: set[str] = set()
    for b in bundles:
        own = (b.own_doctrine or "").strip()
        if own and own not in seen:
            parts.append(own)
            seen.add(own)
    return "\n\n".join(parts)


def compose_tool_specs(names, ctx=None) -> list:
    """The UNION tool catalog for a SET of bundle names (d212 #1): every loaded
    capability's ``tool_specs(ctx)``, de-duplicated by tool name (the base ``finish``
    appears once). This is the surface a composing node offers the model."""
    out: list = []
    seen: set[str] = set()
    for n in sorted(set(names or [])) or [BUNDLE_OBJECT]:
        for spec in get_bundle(n).tool_specs(ctx):
            try:
                fname = spec["function"]["name"]
            except (KeyError, TypeError):
                fname = None
            if fname and fname in seen:
                continue
            if fname:
                seen.add(fname)
            out.append(spec)
    return out


def bundle_names() -> list[str]:
    """The registered bundle names (sorted)."""
    return sorted(_REGISTRY)


# Bundles a PLANNER authors a plan with vs. ones an IN-PLAN NODE self-selects. The
# ``object`` floor is always loaded (never selected); ``planning`` is the planner
# stage's own bundle (an in-plan node never authors a plan). So the node-facing
# ``get_bundles`` catalog advertises everything EXCEPT these two by default (d221).
_NODE_CATALOG_EXCLUDE = frozenset({BUNDLE_OBJECT, BUNDLE_PLANNING})


def bundles_catalog(*, exclude=_NODE_CATALOG_EXCLUDE) -> list[dict[str, str]]:
    """The advertised bundle catalog — ``{name, summary}`` rows a node REASONS over to
    self-select its tools (d221), parallel to the get_shapes / get_specs catalogs.

    ``exclude`` drops always-on / stage-only bundles (default: the ``object`` floor +
    the planner's ``planning`` bundle), so an in-plan node sees only what it may load."""
    skip = set(exclude or ())
    rows: list[dict[str, str]] = []
    for name in sorted(_REGISTRY):
        if name in skip:
            continue
        rows.append({"name": name, "summary": (_REGISTRY[name].summary or "").strip()})
    return rows


def bundles_catalog_text(*, exclude=_NODE_CATALOG_EXCLUDE) -> str:
    """The 'List of bundles:' advertisement embedded in every node's prompt (d221).

    Single source of truth for the catalog text — names + capability-domain summaries
    + the instruction to expand one with ``get_bundles(name=...)`` at runtime. Returns
    '' when no bundle is selectable (degenerate, never in practice)."""
    rows = bundles_catalog(exclude=exclude)
    if not rows:
        return ""
    lines = [
        "List of bundles you can load — REASON about which your task needs, then call "
        "get_bundles(name=\"<NAME>\") to LOAD its tools at runtime (you start with only "
        "the base finish tool; load a bundle BEFORE you try to use its tools):",
    ]
    for r in rows:
        lines.append(f"- {r['name']}: {r['summary']}")
    return "\n".join(lines)


def expand_bundle(name: str, registry=None, ctx=None) -> dict:
    """LOAD a bundle by name at runtime (the ``get_bundles(name=...)`` effect, d221).

    Resolves the bundle, registers its handler-backed ToolDefs onto ``registry`` (when
    one is supplied — the real GrowableToolRegistry growth point), and returns the
    bundle's ``{loaded, summary, doctrine, tools}`` so the caller can offer the bundle's
    native tool schemas + pin its doctrine. Raises :class:`UnknownBundleError` for an
    unregistered name (the caller turns that into a model-visible 'unknown bundle' note,
    never a crash)."""
    bundle = get_bundle(name)  # raises UnknownBundleError on a real miss
    ctx = dict(ctx or {})
    if registry is not None:
        try:
            bundle.register(registry, ctx)
        except Exception:  # noqa: BLE001 - a handler-less bundle / missing ctx is fine
            pass
    return {
        "loaded": bundle.name,
        "summary": (bundle.summary or "").strip(),
        "doctrine": bundle.doctrine,
        "tools": bundle.tool_names(ctx),
    }


__all__ = [
    "ObjectBundle",
    "PlanningBundle",
    "ResearchBundle",
    "ResearchReadBundle",
    "FileBundle",
    "CodebaseReadBundle",
    "FINISH_TOOL",
    "BUNDLE_OBJECT",
    "BUNDLE_PLANNING",
    "BUNDLE_RESEARCH",
    "BUNDLE_RESEARCH_READ",
    "BUNDLE_FILE",
    "BUNDLE_CODEBASE",
    "UnknownBundleError",
    "get_bundle",
    "compose_doctrine",
    "compose_tool_specs",
    "bundle_names",
    "bundles_catalog",
    "bundles_catalog_text",
    "expand_bundle",
]
