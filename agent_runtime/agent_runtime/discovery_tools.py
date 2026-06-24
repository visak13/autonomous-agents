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
from typing import Any, Callable, Optional

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


def make_get_shapes(shapes_dir: Optional[Path] = None) -> Callable[..., dict[str, Any]]:
    """Build the ``get_shapes`` handler reading from ``shapes_dir`` (default dir)."""

    def get_shapes(filter: Optional[str] = None) -> dict[str, Any]:
        """List the available plan SHAPES (methodology/flow templates) the planner
        can seed a plan with — ``{name, description, execution, max_iter}`` rows."""
        catalog = load_shapes(shapes_dir)
        rows: list[dict[str, Any]] = []
        for name in sorted(catalog):
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
) -> Callable[..., dict[str, Any]]:
    """Build the ``get_specs`` handler.

    ``index_provider`` (a zero-arg callable returning index rows/entries) overrides
    the default on-disk read — used in tests to avoid the registry dependency."""
    provider = index_provider or (lambda: _default_spec_index(specs_dir))

    def get_specs(filter: Optional[str] = None) -> dict[str, Any]:
        """List the available SPECIALIZATIONS (output-shaping rulesets) as body-free
        ``{name, description, source}`` rows (d10 — never a compiled body)."""
        rows = [_spec_row(e) for e in (provider() or [])]
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
# Registration — add both discovery tools to a GrowableToolRegistry
# --------------------------------------------------------------------------- #


def register_discovery_tools(
    registry: Any,
    *,
    shapes_dir: Optional[Path] = None,
    specs_dir: Optional[Path] = None,
    spec_index_provider: Optional[Callable[[], list[Any]]] = None,
) -> Any:
    """Add ``get_shapes`` + ``get_specs`` to a :class:`GrowableToolRegistry`.

    ``shapes_dir`` / ``specs_dir`` point the tools at the live catalogs (defaults:
    the packaged shapes dir / no specs). ``spec_index_provider`` overrides the
    spec read (tests). Each tool is one ``ToolDef``; returns the registry."""
    registry.add(ToolDef(
        name="get_shapes",
        description=GET_SHAPES_TOOL.description,
        args_model=GetShapesArgs,
        handler=make_get_shapes(shapes_dir),
    ))
    registry.add(ToolDef(
        name="get_specs",
        description=GET_SPECS_TOOL.description,
        args_model=GetSpecsArgs,
        handler=make_get_specs(specs_dir, index_provider=spec_index_provider),
    ))
    return registry


__all__ = [
    "GetShapesArgs",
    "GetSpecsArgs",
    "make_get_shapes",
    "make_get_specs",
    "GET_SHAPES_TOOL",
    "GET_SPECS_TOOL",
    "register_discovery_tools",
]
