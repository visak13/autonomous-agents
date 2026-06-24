"""On-disk specialization registry with a STRICT d10 split.

The whole point of this module is the d10 context-scoping constraint: the
PLANNER may know only the abstract factory + a LOOKUP (an index/registry of
{name, description, source}), NEVER the full compiled bodies; a LAUNCHED
SUB-AGENT may know only the ONE compiled spec it executes with. Those are two
different reads of the same on-disk store, so they are two SEPARATE methods:

- :meth:`SpecRegistry.index`  → planner-facing. Returns ``SpecIndexEntry`` rows
  (``{name, description, source}``) and reads ONLY each doc's frontmatter — a
  body is never loaded, so the planner code path *cannot* see one.
- :meth:`SpecRegistry.load`   → sub-agent loader. Returns the FULL
  ``CompiledSpec`` (body included) for a single named spec.
- :meth:`SpecRegistry.register` → persists a ``CompiledSpec`` as its markdown
  doc under ``specs/<name>.md`` (the compile-on-approval write, d8).
- :meth:`SpecRegistry.update`   → re-editable persistence (s4/RC7): re-open an
  EXISTING spec, overlay edited ``description``/``body`` while preserving identity
  + provenance, and re-write the SAME doc the loader reads (so the edit is
  effective on the next run).

Storage is a plain directory of markdown-with-frontmatter docs (one per spec) —
no DB, no service, dependency-free (d10). The filename is derived from the spec
name so ``load`` is an O(1) path lookup while ``index`` is a frontmatter-only
scan of the directory.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from specialization.model import (
    CompiledSpec,
    SpecIndexEntry,
    parse_compiled_spec,
    parse_frontmatter_only,
)


def _slug(name: str) -> str:
    """Filesystem-safe stem for a spec name (the docs are ``<slug>.md``).

    Specs are named in kebab-case already; this only guards against stray path
    separators / whitespace so a name can never escape the specs dir."""
    safe = name.strip().replace(" ", "-")
    safe = safe.replace("/", "-").replace("\\", "-")
    if not safe or safe in (".", ".."):
        raise ValueError(f"invalid spec name {name!r}")
    return safe


class SpecRegistry:
    """A directory of compiled-spec markdown docs with the d10 lookup/load split."""

    def __init__(self, specs_dir: str | Path) -> None:
        self.specs_dir = Path(specs_dir)
        self.specs_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.specs_dir / f"{_slug(name)}.md"

    # ---- write side: compile-on-approval persists the doc (d8) ---- #
    def register(self, spec: CompiledSpec) -> Path:
        """Persist a compiled spec as its markdown-with-frontmatter doc.

        Overwrites an existing doc of the same name (recompile). Returns the
        path written."""
        p = self._path(spec.name)
        p.write_text(spec.to_markdown(), encoding="utf-8")
        return p

    # ---- re-editable persistence: re-open an existing spec and edit it (s4/RC7) ---- #
    def update(
        self,
        name: str,
        *,
        description: str | None = None,
        body: str | None = None,
    ) -> CompiledSpec:
        """Apply edits to an ALREADY-REGISTERED spec and re-persist it (s4/RC7).

        This is the write half of "re-open an existing specialization, edit it,
        and have the edit be effective on the NEXT run". It LOADS the current
        compiled spec, overlays only the supplied fields (``description`` and/or
        ``body``), and re-writes its doc — PRESERVING identity (the ``name`` key)
        and all provenance (``source``, ``research_trace_ref``, ``created_at``,
        plus any workflow ``schedule``/``delivery``). Returns the updated spec.

        "Effective next run" is automatic, not a second sync: the doc this
        overwrites under ``specs/<name>.md`` is the SAME file the sub-agent
        :class:`~specialization.loader.SpecLoader` reads (``load_body`` →
        :meth:`load`), so the next launched node composes the edited body. There
        is no separate store to keep in step.

        Raises ``KeyError`` if no spec named ``name`` is registered (you cannot
        edit what was never created); ``ValueError`` if a ``body`` is supplied but
        blank (an empty ruleset would silently neuter the specialist)."""
        if body is not None and not body.strip():
            raise ValueError("updated spec body must be non-empty")
        current = self.load(name)  # KeyError if the spec does not exist
        edited = replace(
            current,
            description=current.description if description is None else description,
            body=current.body if body is None else body,
        )
        self.register(edited)
        # Return the RELOADED spec so the caller gets exactly what persisted (the
        # doc write strips the body) — the return value then matches a later
        # :meth:`load` byte-for-byte, no stale in-memory copy.
        return self.load(name)

    # ---- delete: remove a spec's doc from the store (s4/a3) ---- #
    def delete(self, name: str) -> Path:
        """Remove a registered spec by unlinking its markdown doc. Returns the path.

        The store is STATELESS (every read — ``index``/``load``/``names``/
        ``__contains__`` — hits disk fresh), so a safe delete is just removing the
        ``specs/<slug>.md`` file: there is NO in-memory cache to invalidate. Uses
        ``_slug`` (via :meth:`_path`) so the name can never escape the specs dir.
        Raises ``KeyError`` if no spec named ``name`` is registered (the route maps
        that to 404), mirroring :meth:`load`."""
        p = self._path(name)
        if not p.exists():
            raise KeyError(f"no registered specialization named {name!r}")
        p.unlink()
        return p

    # ---- planner-facing lookup: index ONLY, never a body (d10) ---- #
    def index(self) -> list[SpecIndexEntry]:
        """Return the body-free lookup rows the PLANNER reasons over.

        Reads ONLY each doc's frontmatter (``parse_frontmatter_only`` never
        touches the body), so this surface is body-free by construction. Rows are
        ordered by name for a deterministic listing."""
        entries: list[SpecIndexEntry] = []
        for p in sorted(self.specs_dir.glob("*.md")):
            meta = parse_frontmatter_only(p.read_text(encoding="utf-8"))
            entries.append(
                SpecIndexEntry(
                    name=meta.get("name", ""),
                    description=meta.get("description", ""),
                    source=meta.get("source", ""),
                )
            )
        return entries

    def names(self) -> list[str]:
        """Just the registered spec names (a thin convenience over :meth:`index`)."""
        return [e.name for e in self.index()]

    # ---- sub-agent loader: the FULL compiled body for ONE spec ---- #
    def load(self, name: str) -> CompiledSpec:
        """Return the FULL ``CompiledSpec`` (body included) for one named spec.

        This is the ONLY surface that yields a body, and it yields exactly one —
        the single spec a launched sub-agent executes with (d10). Raises
        ``KeyError`` if no such spec is registered."""
        p = self._path(name)
        if not p.exists():
            raise KeyError(f"no registered specialization named {name!r}")
        return parse_compiled_spec(p.read_text(encoding="utf-8"))

    def __contains__(self, name: str) -> bool:
        try:
            return self._path(name).exists()
        except ValueError:
            return False
