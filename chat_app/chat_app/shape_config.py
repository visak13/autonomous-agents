"""Shape-config store + HTTP surface — per-shape ``max_iter`` overrides (s4/a4, d5).

WHAT THIS IS
------------
The plan SHAPES are declarative TEXT FILES on disk (``agent_runtime/shapes/*.toml``
— linear, modular-parallel, the bounded cyclic deep-research). They are the
runtime's source of truth for a shape's STRUCTURE (topology / round_roles /
final_roles / the default ``max_iter`` ceiling). What the s4 Shapes screen lets
the user change is ONE thing per shape: the ``max_iter`` cap — and that override
must PERSIST and be HONORED by the s3 runtime (d5).

This module is the BACKEND half of s4's Shapes screen:

* :class:`ShapeConfigStore` — durable per-shape ``max_iter`` overrides in the
  SHARED SQLite (``<data_dir>/chat.db``, the same db the chat store + cron tools
  use — d5 "persists in the shared SQLite"). The shape FILES stay the structure;
  only the override is stored. A row is ``(shape_name, max_iter, updated_at)``.
* :class:`ShapeConfigService` — merges the on-disk shape catalog
  (:func:`agent_runtime.load_shapes`) with the stored overrides into a single view
  per shape: its structure PLUS ``max_iter_override`` (the stored UI value, or
  ``None``) and ``effective_max_iter`` (what the runtime will actually run —
  ``override`` clamped to the shape's ``hard_cap`` via
  :meth:`~agent_runtime.shapes.ShapeSpec.effective_max_iter`).
* :func:`register_shape_routes` — mounts the read/WRITE HTTP surface
  (``GET /shapes``, ``GET /shapes/{name}``, ``PUT /shapes/{name}/max_iter``) onto
  the one app, mirroring the app-agnostic mount style of
  :func:`chat_app.specializations.register_specialization_routes`.

WHO READS THE OVERRIDE
----------------------
The runtime honors the override through :func:`chat_app.agentic.run_agentic`,
which (when handed this store) reads ``get_max_iter(selected_shape)`` after shape
selection and passes it as the deep-research executor's ``max_iter_override`` —
so a UI-set value, not the text-file default, bounds the unroll. The executor
already clamps to the shape's ``hard_cap`` (the shared-GPU safety bound), so the
store is free to hold a raw value and the clamp happens at read.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_runtime import ShapeSpec, load_shapes
from agent_runtime.shape_author import delete_shape as delete_shape_file

from chat_app.persistence import resolve_data_dir


# The 6 shipped built-in shapes (d13 no-regression): they live in the package
# shapes dir alongside user-authored shapes, so a delete is physically possible —
# but DELETE /shapes/{name} REFUSES them (409) so the user can only clear their OWN
# authored noise, never a shipped shape. Specs have no built-ins (any registered
# spec is deletable); this guard is shapes-only.
BUILTIN_SHAPES = frozenset(
    {
        "linear",
        "modular-parallel",
        "concurrent-multi-topic-gathering",
        "deep-research",
        "iterative-deep-research",
        "iterative-writing-improvement",
    }
)


def _now_iso() -> str:
    """UTC timestamp, second-precision ISO-8601 (sortable, timezone-explicit)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- #
# the durable store — per-shape max_iter overrides in the SHARED chat.db (d5)
# --------------------------------------------------------------------------- #
class ShapeConfigStore:
    """In-process, file-based store for per-shape ``max_iter`` overrides.

    Lives in the SAME SQLite database as the chat store + cron tools
    (``<data_dir>/chat.db``) so all shared state is one file (d5 "the shared
    SQLite"). It owns its OWN connection to that file — sqlite permits several
    connections to one database, and the ``shape_config`` table is disjoint from
    the chat tables, so there is no contention with :class:`~chat_app.persistence.ChatStore`.
    Every write commits immediately, so an override is durable on return — a fresh
    process (or another connection) reads exactly what was written.
    """

    DB_FILENAME = "chat.db"

    def __init__(self, data_dir: str | os.PathLike[str] | None = None) -> None:
        self.data_dir = resolve_data_dir(data_dir)
        self.db_path = self.data_dir / self.DB_FILENAME
        # check_same_thread=False: an async app may touch the store from uvicorn
        # worker thread(s); a process-wide lock serialises writes on the one
        # connection (mirrors ChatStore's discipline).
        self.db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS shape_config (
                    shape_name TEXT PRIMARY KEY,
                    max_iter   INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                -- s13/B6: the per-shape research-tree DEPTH override (sibling of
                -- max_iter), in a DISJOINT table so a shape may carry a depth override
                -- with no max_iter row (and vice-versa) — no NOT-NULL coupling, no
                -- migration of the existing shape_config rows.
                CREATE TABLE IF NOT EXISTS shape_depth (
                    shape_name TEXT PRIMARY KEY,
                    depth      INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self.db.commit()

    def set_max_iter(self, shape_name: str, max_iter: int) -> None:
        """Persist the per-shape ``max_iter`` override (upsert; durable on return).

        The stored value is the RAW UI value; the effective cap the runtime runs is
        the shape's ``effective_max_iter(override)`` (clamped to ``hard_cap`` at
        read). A value ``< 1`` is rejected — a nonsensical override never lands in
        the store (the route surfaces this as 422)."""
        name = (shape_name or "").strip()
        if not name:
            raise ValueError("shape_name must be non-empty")
        value = int(max_iter)
        if value < 1:
            raise ValueError(f"max_iter must be >= 1 (got {value})")
        with self._lock:
            self.db.execute(
                "INSERT INTO shape_config(shape_name, max_iter, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(shape_name) DO UPDATE SET "
                "max_iter = excluded.max_iter, updated_at = excluded.updated_at",
                (name, value, _now_iso()),
            )
            self.db.commit()

    def get_max_iter(self, shape_name: str) -> Optional[int]:
        """The stored override for ``shape_name``, or ``None`` if none is set.

        ``None`` means "no UI override" — the runtime then uses the shape file's
        default ``max_iter``."""
        row = self.db.execute(
            "SELECT max_iter FROM shape_config WHERE shape_name = ?",
            ((shape_name or "").strip(),),
        ).fetchone()
        return int(row["max_iter"]) if row is not None else None

    def set_depth(self, shape_name: str, depth: int) -> None:
        """Persist the per-shape research-tree ``depth`` override (s13/B6; upsert,
        durable on return). MIRRORS :meth:`set_max_iter` exactly — same shapes/specs
        store, a sibling override — so the user controls report-path tree DEPTH the
        same way they set ``max_iter``. The stored value is the RAW UI value; the
        run_plan_chain consumer clamps it to ``[1, N4_TREE_DEPTH_CEILING]`` (the
        hard ≤10 the user fixed) at read. A value ``< 1`` is rejected (the route
        surfaces it as 422)."""
        name = (shape_name or "").strip()
        if not name:
            raise ValueError("shape_name must be non-empty")
        value = int(depth)
        if value < 1:
            raise ValueError(f"depth must be >= 1 (got {value})")
        with self._lock:
            self.db.execute(
                "INSERT INTO shape_depth(shape_name, depth, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(shape_name) DO UPDATE SET "
                "depth = excluded.depth, updated_at = excluded.updated_at",
                (name, value, _now_iso()),
            )
            self.db.commit()

    def get_depth(self, shape_name: str) -> Optional[int]:
        """The stored research-tree ``depth`` override for ``shape_name``, or ``None``
        if none is set (s13/B6). ``None`` means "no UI override" — the runtime then
        uses the env baseline (``RA_TREE_DEPTH`` via ``TreeConfig.from_env``).
        Read by :func:`chat_app.agentic.run_agentic` for the SELECTED shape and
        handed to ``run_plan_chain`` as ``research_depth``."""
        row = self.db.execute(
            "SELECT depth FROM shape_depth WHERE shape_name = ?",
            ((shape_name or "").strip(),),
        ).fetchone()
        return int(row["depth"]) if row is not None else None

    def all_depths(self) -> dict[str, int]:
        """Every stored depth override as ``{shape_name: depth}`` (one read)."""
        rows = self.db.execute(
            "SELECT shape_name, depth FROM shape_depth"
        ).fetchall()
        return {r["shape_name"]: int(r["depth"]) for r in rows}

    def delete(self, shape_name: str) -> None:
        """Drop the per-shape ``max_iter`` AND ``depth`` override rows (idempotent;
        durable on return). Called when a shape file is deleted so no override row
        outlives the file as an orphan. Deleting a non-existent row is a no-op (a
        shape the user never overrode has no row to remove)."""
        name = (shape_name or "").strip()
        with self._lock:
            self.db.execute("DELETE FROM shape_config WHERE shape_name = ?", (name,))
            self.db.execute("DELETE FROM shape_depth WHERE shape_name = ?", (name,))
            self.db.commit()

    def all_overrides(self) -> dict[str, int]:
        """Every stored override as ``{shape_name: max_iter}`` (one read)."""
        rows = self.db.execute(
            "SELECT shape_name, max_iter FROM shape_config"
        ).fetchall()
        return {r["shape_name"]: int(r["max_iter"]) for r in rows}

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def __enter__(self) -> "ShapeConfigStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# request model (Pydantic v2 — house style; 422 is automatic on bad input)
# --------------------------------------------------------------------------- #
class SetMaxIterRequest(BaseModel):
    """Set a shape's ``max_iter`` override (the ONE field the Shapes screen edits).

    ``ge=1`` rejects a nonsensical floor at the wire; the upper sanity bound keeps
    a typo from storing an absurd value (the shape's own ``hard_cap`` still clamps
    the EFFECTIVE round count regardless of what is stored)."""

    max_iter: int = Field(ge=1, le=1000)


class SetDepthRequest(BaseModel):
    """Set a shape's research-tree ``depth`` override (s13/B6 — the sibling of
    ``max_iter`` on the Shapes screen).

    ``ge=1`` rejects a nonsensical floor at the wire; ``le=10`` is the hard
    ``N4_TREE_DEPTH_CEILING`` the user fixed (the run_plan_chain consumer also
    clamps, so the bound is enforced even if a raw value were stored)."""

    depth: int = Field(ge=1, le=10)


# --------------------------------------------------------------------------- #
# the service — merge the on-disk shape catalog with the stored overrides
# --------------------------------------------------------------------------- #
class ShapeConfigService:
    """Read the text-file shape catalog + the stored overrides into one view.

    The shape FILES are the structure (topology / round_roles / final_roles /
    default + hard_cap); this layers the persisted ``max_iter`` override on top and
    computes the ``effective_max_iter`` the runtime will run. ``shapes_dir`` is
    overridable for tests; it defaults to the package's on-disk catalog so a newly
    added shape file is surfaced with no code change.
    """

    def __init__(
        self, store: ShapeConfigStore, *, shapes_dir: Optional[Path] = None
    ) -> None:
        self._store = store
        self._shapes_dir = shapes_dir

    @property
    def store(self) -> ShapeConfigStore:
        return self._store

    def _catalog(self) -> dict[str, ShapeSpec]:
        return load_shapes(self._shapes_dir)

    @staticmethod
    def _view(
        spec: ShapeSpec,
        override: Optional[int],
        depth_override: Optional[int] = None,
    ) -> dict[str, Any]:
        """One shape's full structure + its overrides + the effective round count."""
        view = spec.as_dict()
        view["max_iter_override"] = override
        # The cap the runtime will actually run: the override clamped to hard_cap
        # (or, with no override, the shape's own default) — the single number the
        # s3 deep-research unroll honors.
        view["effective_max_iter"] = spec.effective_max_iter(override)
        # s13/B6: the per-shape research-tree DEPTH override (or None = env baseline).
        # The run_plan_chain consumer clamps it to the hard ceiling at read; the view
        # surfaces the raw stored value so the UI can render/edit it like max_iter.
        view["depth_override"] = depth_override
        return view

    def list_shapes(self) -> list[dict[str, Any]]:
        """Every text-file shape, each merged with its stored overrides (sorted)."""
        overrides = self._store.all_overrides()
        depths = self._store.all_depths()
        catalog = self._catalog()
        return [
            self._view(catalog[name], overrides.get(name), depths.get(name))
            for name in sorted(catalog)
        ]

    def get_shape(self, name: str) -> Optional[dict[str, Any]]:
        """One shape's view, or ``None`` if no such text-file shape exists."""
        spec = self._catalog().get(name)
        if spec is None:
            return None
        return self._view(
            spec, self._store.get_max_iter(name), self._store.get_depth(name)
        )

    def set_max_iter(self, name: str, max_iter: int) -> Optional[dict[str, Any]]:
        """Persist the override for ``name`` and return the updated view.

        Returns ``None`` if ``name`` is not a known text-file shape (the route maps
        that to 404) so the store never accumulates an override for a shape that
        does not exist. A ``max_iter < 1`` raises ``ValueError`` from the store
        (the route maps that to 422)."""
        spec = self._catalog().get(name)
        if spec is None:
            return None
        self._store.set_max_iter(name, max_iter)
        return self._view(spec, max_iter, self._store.get_depth(name))

    def set_depth(self, name: str, depth: int) -> Optional[dict[str, Any]]:
        """Persist the research-tree ``depth`` override for ``name`` and return the
        updated view (s13/B6 — the SAME shapes/specs path as :meth:`set_max_iter`).

        Returns ``None`` if ``name`` is not a known text-file shape (the route maps
        that to 404) so the store never accumulates a depth override for a shape that
        does not exist. A ``depth < 1`` raises ``ValueError`` from the store (the
        route maps that to 422)."""
        spec = self._catalog().get(name)
        if spec is None:
            return None
        self._store.set_depth(name, depth)
        return self._view(spec, self._store.get_max_iter(name), depth)

    def delete_shape(self, name: str) -> Optional[bool]:
        """Delete a USER-AUTHORED shape: remove its ``<name>.toml`` AND drop its
        ``shape_config`` override row (so no row outlives the file). Order matters —
        FILE THEN ROW. Returns ``None`` if no such text-file shape exists (route →
        404), ``True`` on success. Raises ``PermissionError`` for a shipped built-in
        (route → 409, d13 no-regression — only the user's own shapes are deletable).

        The structural store is stateless (load_shapes globs on every call), so the
        only durable extra state needing cleanup is the override row."""
        spec = self._catalog().get(name)
        if spec is None:
            return None
        if name in BUILTIN_SHAPES:
            raise PermissionError(
                f"shape {name!r} is a built-in and cannot be deleted"
            )
        # File then row: a leftover override row must never outlive its shape file.
        delete_shape_file(name, shapes_dir=self._shapes_dir)
        self._store.delete(name)
        return True


def register_shape_routes(
    app: FastAPI, service: ShapeConfigService
) -> ShapeConfigService:
    """Mount the Shapes config read/WRITE routes onto ``app`` (app-agnostic).

    Returns the service so a caller (the wiring, a test harness) can keep a handle.
    Mirrors :func:`chat_app.specializations.register_specialization_routes`.
    """

    @app.get("/shapes")
    async def list_shapes() -> dict:
        """List every text-file-defined shape with its structure + effective max_iter.

        This is what the s4 Shapes screen renders: each shape's topology
        (``execution``) / round_roles / final_roles / default + hard_cap, plus the
        user's ``max_iter_override`` and the ``effective_max_iter`` the runtime
        honors."""
        return {"shapes": service.list_shapes()}

    @app.get("/shapes/{name}")
    async def get_shape(name: str) -> dict:
        """View ONE shape's full structure + its current override (404 if unknown)."""
        view = service.get_shape(name)
        if view is None:
            raise HTTPException(status_code=404, detail=f"no shape {name!r}")
        return view

    @app.put("/shapes/{name}/max_iter")
    async def set_max_iter(name: str, req: SetMaxIterRequest) -> dict:
        """SET a shape's ``max_iter`` override (persisted to SQLite, honored by s3).

        404 if ``name`` is not a text-file shape; 422 (automatic + the store guard)
        on a value ``< 1``. Returns the updated shape view so the UI re-renders the
        new effective cap in one round-trip."""
        try:
            view = service.set_max_iter(name, req.max_iter)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if view is None:
            raise HTTPException(status_code=404, detail=f"no shape {name!r}")
        return view

    @app.put("/shapes/{name}/depth")
    async def set_depth(name: str, req: SetDepthRequest) -> dict:
        """SET a shape's research-tree ``depth`` override (s13/B6 — the SAME
        shapes/specs path as ``PUT /shapes/{name}/max_iter``; persisted to SQLite,
        honored by run_plan_chain on the report route).

        404 if ``name`` is not a text-file shape; 422 (automatic + the store guard)
        on a value out of ``[1, 10]``. Returns the updated shape view so the UI
        re-renders the new depth in one round-trip."""
        try:
            view = service.set_depth(name, req.depth)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if view is None:
            raise HTTPException(status_code=404, detail=f"no shape {name!r}")
        return view

    @app.delete("/shapes/{name}")
    async def delete_shape_route(name: str) -> dict:
        """DELETE a USER-AUTHORED shape (unlink its ``.toml`` + drop its override
        row). Additive — leaves list/get/set_max_iter/author UNCHANGED. 404 if no
        such shape; 409 for a shipped built-in (d13 — only the user's own shapes are
        deletable). The file + SQLite delete is offloaded off the event loop (d4).
        Returns ``{"ok": true, "deleted": name}`` so the UI can drop the row +
        refresh the list."""
        try:
            ok = await asyncio.to_thread(service.delete_shape, name)
        except PermissionError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        if ok is None:
            raise HTTPException(status_code=404, detail=f"no shape {name!r}")
        return {"ok": True, "deleted": name}

    return service


def build_shape_config_service(
    data_dir: str | os.PathLike[str] | None = None,
    *,
    shapes_dir: Optional[Path] = None,
) -> ShapeConfigService:
    """Convenience constructor: a service over a fresh store rooted at ``data_dir``.

    The wiring passes its SHARED store instead (so the API writes and the runtime
    reads see the same connection); this is for stand-alone callers/tests."""
    return ShapeConfigService(ShapeConfigStore(data_dir), shapes_dir=shapes_dir)


__all__ = [
    "ShapeConfigStore",
    "ShapeConfigService",
    "SetMaxIterRequest",
    "SetDepthRequest",
    "register_shape_routes",
    "build_shape_config_service",
]
