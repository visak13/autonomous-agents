"""DISCOVERY tools — ``get_shapes`` + ``get_specs`` (s13 / P2.1, d126 #10 / d132.A).

The planner reasons over WHICH plan-shape (the methodology/flow template) and WHICH
specializations (the output-shaping rulesets) to seed a plan with. Today that
catalog is injected once as static text in ``factory.planner_context``; the user's
tool-layer thesis (d125/d126) wants the catalog to be QUERYABLE as tools so the
planner can look it up on demand and SEE mid-run changes (a newly compiled spec, a
new shape) without a restart.

Two read-only tools, each ONE :class:`~reactive_tools.tool_registry.ToolDef`:

- ``get_shapes`` → the available plan SHAPES as ``{name, description, execution,
  max_iter}`` rows, read fresh from the shapes TOML dir
  (:func:`agent_runtime.shapes.load_shapes`).
- ``get_specs``  → the available SPECIALIZATIONS as body-free ``{name, description,
  source}`` rows, read fresh from the on-disk spec registry. This honors the d10
  context-scoping split: the catalog surface returns DESCRIPTIONS ONLY, never a
  compiled body (a body reaches only a launched sub-agent via its spec_id).

Both accept an optional ``filter`` substring so the planner can narrow a large
catalog (matched case-insensitively against name + description). Pure reads: no
network, no LLM, no process boundary (d2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from reactive_tools.tool_registry import ToolDef

from .shapes import load_shapes


# --------------------------------------------------------------------------- #
# get_shapes — the plan-shape (methodology/flow) catalog
# --------------------------------------------------------------------------- #


class GetShapesArgs(BaseModel):
    """Args for :data:`GET_SHAPES_TOOL`."""

    filter: Optional[str] = Field(
        None,
        description=("OPTIONAL case-insensitive substring to narrow the catalog "
                     "(matched against a shape's name + description); omit to list all."))


def _matches(text_parts: list[str], needle: Optional[str]) -> bool:
    if not needle:
        return True
    hay = " ".join(p for p in text_parts if p).lower()
    return needle.strip().lower() in hay


def make_get_shapes(
    shapes_dir: Optional[Path] = None,
    *,
    exposed: Optional[Sequence[str]] = None,
) -> Callable[..., dict[str, Any]]:
    """Build the ``get_shapes`` handler reading from ``shapes_dir`` (default dir).

    ``exposed`` (an optional allow-list of shape names) CURATES which shapes this
    discovery surface advertises — the registry-scoping lever (d230). ``None``
    advertises every shape on disk (the generic default; tests + the growth phase
    rely on it). The raw :func:`load_shapes` loader is untouched, so a curated-OUT
    shape still exists for the executor — it is only hidden from the planner."""
    allow = set(exposed) if exposed is not None else None

    def get_shapes(filter: Optional[str] = None) -> dict[str, Any]:
        """List the available plan SHAPES (methodology/flow templates) the planner
        can seed a plan with — ``{name, description, execution, max_iter}`` rows."""
        catalog = load_shapes(shapes_dir)
        rows: list[dict[str, Any]] = []
        for name in sorted(catalog):
            if allow is not None and name not in allow:
                continue
            spec = catalog[name]
            if not _matches([name, spec.description], filter):
                continue
            rows.append({
                "name": name,
                "description": spec.description,
                "execution": spec.execution,
                "max_iter": spec.max_iter,
            })
        return {"shapes": rows, "count": len(rows)}

    return get_shapes


GET_SHAPES_TOOL = ToolDef(
    name="get_shapes",
    description=(
        "DISCOVERY: list the available plan SHAPES (the methodology/flow templates "
        "you seed a plan with, e.g. deep-research) as {name, description, execution} "
        "rows. Query this to pick the right shape for the goal; reflects shapes "
        "added since startup."),
    args_model=GetShapesArgs,
    handler=make_get_shapes(),
)


# --------------------------------------------------------------------------- #
# get_specs — the specialization (output-shaping ruleset) catalog (body-free)
# --------------------------------------------------------------------------- #


class GetSpecsArgs(BaseModel):
    """Args for :data:`GET_SPECS_TOOL`."""

    filter: Optional[str] = Field(
        None,
        description=("OPTIONAL case-insensitive substring to narrow the catalog "
                     "(matched against a spec's name + description); omit to list all."))


def _default_spec_index(specs_dir: Optional[Path]) -> list[Any]:
    """Read the body-free spec index from the on-disk registry (lazy import so
    agent_runtime carries no hard dependency on the specialization package)."""
    if specs_dir is None:
        return []
    try:
        from specialization.registry import SpecRegistry
    except Exception:  # noqa: BLE001 - registry unavailable: empty catalog, not a crash
        return []
    return list(SpecRegistry(specs_dir).index())


def _spec_row(entry: Any) -> dict[str, str]:
    """Normalise a SpecIndexEntry OR a plain mapping to a body-free row."""
    if isinstance(entry, dict):
        get = entry.get
    else:
        get = lambda k, d="": getattr(entry, k, d)  # noqa: E731
    return {
        "name": str(get("name", "") or ""),
        "description": str(get("description", "") or ""),
        "source": str(get("source", "") or ""),
    }


def make_get_specs(
    specs_dir: Optional[Path] = None,
    *,
    index_provider: Optional[Callable[[], list[Any]]] = None,
    exposed: Optional[Sequence[str]] = None,
) -> Callable[..., dict[str, Any]]:
    """Build the ``get_specs`` handler.

    ``index_provider`` (a zero-arg callable returning index rows/entries) overrides
    the default on-disk read — used in tests to avoid the registry dependency.
    ``exposed`` (an optional allow-list of spec names) CURATES which specs this
    discovery surface advertises (d230 registry scoping); ``None`` advertises every
    registered spec (the generic default). Curation hides a spec from the planner
    only — its body still loads by name if referenced."""
    provider = index_provider or (lambda: _default_spec_index(specs_dir))
    allow = set(exposed) if exposed is not None else None

    def get_specs(filter: Optional[str] = None) -> dict[str, Any]:
        """List the available SPECIALIZATIONS (output-shaping rulesets) as body-free
        ``{name, description, source}`` rows (d10 — never a compiled body)."""
        rows = [_spec_row(e) for e in (provider() or [])]
        if allow is not None:
            rows = [r for r in rows if r["name"] in allow]
        rows = [r for r in rows if _matches([r["name"], r["description"]], filter)]
        rows.sort(key=lambda r: r["name"])
        return {"specs": rows, "count": len(rows)}

    return get_specs


GET_SPECS_TOOL = ToolDef(
    name="get_specs",
    description=(
        "DISCOVERY: list the available SPECIALIZATIONS (output-shaping rulesets you "
        "attach to a node/worker, e.g. html-writer) as body-free {name, description, "
        "source} rows — descriptions only, never the ruleset body. Query this to "
        "pick per-node specs; reflects specs compiled since startup."),
    args_model=GetSpecsArgs,
    handler=make_get_specs(),
)


# --------------------------------------------------------------------------- #
# get_bundles — the bundle (tool-capability) catalog + RUNTIME loader (d221)
# --------------------------------------------------------------------------- #
#
# Parallel to get_shapes / get_specs, but with a SIDE EFFECT a node opts into: with no
# arg it LISTS the advertised bundle catalog ({name, summary} rows the node reasons
# over); with a ``name`` it LOADS that bundle's tools at runtime — registering its
# handler-backed ToolDefs onto the live registry (real GrowableToolRegistry growth) and
# returning its doctrine + tool names so the node can use them. This is the NODE-SELF-
# SELECT mechanism (d221): the planner sets only role + spec; each node expands the
# bundle(s) its task needs here at runtime. The ``object`` floor is always loaded.


class GetBundlesArgs(BaseModel):
    """Args for :data:`GET_BUNDLES_TOOL`."""

    name: Optional[str] = Field(
        None,
        description=("OPTIONAL bundle NAME to LOAD (expand its tools at runtime so you "
                     "can then call them); omit to LIST the available bundles and their "
                     "capability domains so you can choose."))


def make_get_bundles(
    *,
    registry: Optional[Any] = None,
    ctx_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
    on_load: Optional[Callable[[str], None]] = None,
) -> Callable[..., dict[str, Any]]:
    """Build the ``get_bundles`` handler.

    ``registry`` is the live :class:`GrowableToolRegistry` a load grows (handler-backed
    tools become selectable immediately). ``ctx_provider`` supplies the per-run ctx (e.g.
    this run's fetched ``sources``) a bundle needs to bind its handler tools. ``on_load``
    is an optional hook the runtime passes so a node can record the loaded bundle (e.g.
    grow its pinned doctrine + native tool schemas) when the model self-selects it."""
    from .bundles import UnknownBundleError, bundles_catalog, expand_bundle

    def get_bundles(name: Optional[str] = None) -> dict[str, Any]:
        """LIST the loadable tool bundles, or LOAD one by name to expand your tools.

        With no ``name``: returns the advertised ``{name, summary}`` catalog to reason
        over. With a ``name``: LOADS that bundle (registers its tools at runtime) and
        returns ``{loaded, summary, doctrine, tools}`` — call its tools on later turns."""
        if not (name and str(name).strip()):
            rows = bundles_catalog()
            return {"bundles": rows, "count": len(rows)}
        ctx = dict(ctx_provider() or {}) if ctx_provider else {}
        try:
            result = expand_bundle(str(name).strip(), registry, ctx)
        except UnknownBundleError:
            rows = bundles_catalog()
            return {
                "error": f"unknown bundle {name!r}",
                "bundles": rows,
                "count": len(rows),
            }
        if on_load:
            try:
                on_load(result["loaded"])
            except Exception:  # noqa: BLE001 - the load hook must never break the tool
                pass
        return result

    return get_bundles


GET_BUNDLES_TOOL = ToolDef(
    name="get_bundles",
    description=(
        "DISCOVERY + RUNTIME LOAD: the tool bundles (capability domains) you can load "
        "for your node. Call with NO args to LIST the bundles ({name, summary}) and "
        "REASON about which your task needs; call with name=\"<NAME>\" to LOAD that "
        "bundle — its tools are registered at runtime and become callable, and you get "
        "back its doctrine + tool names. You start with only the base finish tool, so "
        "LOAD the bundle(s) you need (e.g. research to gather, file to write a "
        "document) BEFORE using their tools."),
    args_model=GetBundlesArgs,
    handler=make_get_bundles(),
)


# --------------------------------------------------------------------------- #
# Registration — add the discovery tools to a GrowableToolRegistry
# --------------------------------------------------------------------------- #


def register_discovery_tools(
    registry: Any,
    *,
    shapes_dir: Optional[Path] = None,
    specs_dir: Optional[Path] = None,
    spec_index_provider: Optional[Callable[[], list[Any]]] = None,
    exposed_shapes: Optional[Sequence[str]] = None,
    exposed_specs: Optional[Sequence[str]] = None,
) -> Any:
    """Add ``get_shapes`` + ``get_specs`` + ``get_bundles`` to a :class:`GrowableToolRegistry`.

    ``shapes_dir`` / ``specs_dir`` point the tools at the live catalogs (defaults:
    the packaged shapes dir / no specs). ``spec_index_provider`` overrides the
    spec read (tests). ``exposed_shapes`` / ``exposed_specs`` (optional allow-lists)
    CURATE which shapes/specs the discovery surfaces advertise (d230 registry
    scoping); ``None`` (default) advertises the full catalog. ``get_bundles`` loads
    onto THIS same registry. Each tool is one ``ToolDef``; returns the registry."""
    registry.add(ToolDef(
        name="get_shapes",
        description=GET_SHAPES_TOOL.description,
        args_model=GetShapesArgs,
        handler=make_get_shapes(shapes_dir, exposed=exposed_shapes),
    ))
    registry.add(ToolDef(
        name="get_specs",
        description=GET_SPECS_TOOL.description,
        args_model=GetSpecsArgs,
        handler=make_get_specs(
            specs_dir, index_provider=spec_index_provider, exposed=exposed_specs
        ),
    ))
    registry.add(ToolDef(
        name="get_bundles",
        description=GET_BUNDLES_TOOL.description,
        args_model=GetBundlesArgs,
        handler=make_get_bundles(registry=registry),
    ))
    return registry


__all__ = [
    "GetShapesArgs",
    "GetSpecsArgs",
    "GetBundlesArgs",
    "make_get_shapes",
    "make_get_specs",
    "make_get_bundles",
    "GET_SHAPES_TOOL",
    "GET_SPECS_TOOL",
    "GET_BUNDLES_TOOL",
    "register_discovery_tools",
]
