"""NL description -> Gemma-authored shape, exposed as an HTTP surface (s9/b1, d14(2)/d9).

WHAT THIS IS
------------
The BACKEND half of the Shapes screen's "create by describing" affordance: the user
DESCRIBES a plan shape in natural language and the local Gemma model AUTHORS the
declarative TOML file the runtime loads — exactly as a specialization is
chat-authored+compiled (:mod:`chat_app.spec_chat`). The user NEVER hand-writes a
shape file (d14(2)).

It MIRRORS the spec-chat authoring MECHANISM — a blocking native-structured Gemma
call the app offloads off its event loop, after which the artifact is persisted into
the store the runtime authoritatively reads — but it is the GENUINE shapes flow, not
a clone of the multi-turn spec conversation. A shape is a SINGLE small structured
decision (:class:`agent_runtime.shape_author.ShapeAuthor`), not the iterative
refinement a ruleset needs, so this is ONE ``describe -> author -> save`` call. It
returns the authored shape's full view — the SAME :class:`~chat_app.shape_config.ShapeConfigService`
view the list renders — so the new shape appears in the Shapes list immediately and
is loadable/runnable by the runtime (the selector harvests the shapes dir at call
time, the unroll/scheduler consume the file unchanged).

OWNERSHIP (s9 boundary)
-----------------------
This module owns only the ENDPOINT + its wiring. The authoring PROMPT + schema + the
family coercion live in :mod:`agent_runtime.shape_author` (a1) and the generation
QUALITY is tuned separately (b2) — this module does not touch either.

SAFETY
------
* Authoring requires the LIVE model; with no live transport the route returns 503
  (no silent stub-authored shape).
* The authored shape is written into the SAME shapes dir the
  :class:`~chat_app.shape_config.ShapeConfigService` reads, so the list and the
  authored file never diverge.
* A name that collides with an EXISTING shape is REJECTED (409), never silently
  overwritten — so a Gemma-picked name can never clobber a built-in shape file
  (universal [required]: never blind-overwrite).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_runtime.selfheal import MalformedOutputError
from agent_runtime.shape_author import ShapeAuthor, write_shape
from agent_runtime.shapes import ShapeError, load_shape

from chat_app.shape_config import ShapeConfigService

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# errors the service raises (the route maps each to an HTTP status)
# --------------------------------------------------------------------------- #
class ShapeAuthoringUnavailable(RuntimeError):
    """Authoring was requested but no LIVE model transport is configured.

    Shape authoring is a genuine Gemma call (d14(2)); the offline/stub seam cannot
    produce a real shape, so the route surfaces this as 503 rather than writing a
    bogus shape file."""


class ShapeNameConflict(ValueError):
    """The authored shape's name collides with an existing shape — refuse to clobber.

    The model chooses the shape name; if it picks the id of a shape that already
    exists (a built-in or a previously authored one), writing would OVERWRITE that
    file. We reject instead (mapped to 409) so authoring can never silently replace
    an existing shape (universal [required]: never blind-overwrite)."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(
            f"a shape named {name!r} already exists; describe it differently or "
            f"edit the existing shape instead of overwriting it"
        )


# --------------------------------------------------------------------------- #
# request model (Pydantic v2 — house style; 422 automatic on bad input)
# --------------------------------------------------------------------------- #
class AuthorShapeRequest(BaseModel):
    """Describe a shape in natural language; Gemma authors the declarative file.

    ``description`` is the ONLY thing the user supplies — the model picks the
    execution posture, roles and round ceiling. ``name_hint`` is an OPTIONAL nudge
    for the slug when the model leaves the name empty (it never overrides a name the
    model emits)."""

    description: str = Field(min_length=1, max_length=4000)
    name_hint: str = Field(default="", max_length=120)


# --------------------------------------------------------------------------- #
# the service — describe -> author -> save -> return the list view
# --------------------------------------------------------------------------- #
class ShapeAuthorService:
    """Turn one NL description into a saved, runnable shape and return its view.

    Composes a1's :class:`~agent_runtime.shape_author.ShapeAuthor` (the live Gemma
    authoring call) with the s4 :class:`~chat_app.shape_config.ShapeConfigService`
    (the catalog view the Shapes list renders). The authored file is written into
    the SAME shapes dir the config service reads, so the returned view and the
    on-disk catalog are always consistent.

    Parameters
    ----------
    config_service:
        The catalog/override service whose ``get_shape`` produces the returned view.
        Its shapes dir MUST match ``shapes_dir`` (both default to the package
        catalog) so an authored file is visible to the list.
    transport:
        The live ``llm_framework`` transport, or ``None`` when the app runs on the
        offline/stub seam — in which case authoring is UNAVAILABLE (503), never
        stubbed into a fake shape.
    shapes_dir:
        Where authored files are written; defaults (``None``) to the package
        :data:`~agent_runtime.shapes.SHAPES_DIR`, matching the config service's own
        default so the authored shape lands in the catalog the list reads.
    """

    def __init__(
        self,
        config_service: ShapeConfigService,
        *,
        transport: Optional[Any] = None,
        shapes_dir: Optional[Path] = None,
    ) -> None:
        self._config = config_service
        self._transport = transport
        self._shapes_dir = Path(shapes_dir) if shapes_dir is not None else None

    @property
    def available(self) -> bool:
        """True iff a live transport is wired (authoring needs the real model)."""
        return self._transport is not None

    async def author(self, description: str, *, name_hint: str = "") -> dict[str, Any]:
        """Author + save a shape from ``description``; return its full list view.

        Raises :class:`ShapeAuthoringUnavailable` (503) with no live transport,
        :class:`ShapeNameConflict` (409) if the authored name already exists,
        :class:`~agent_runtime.selfheal.MalformedOutputError` /
        :class:`~agent_runtime.shapes.ShapeError` (422) when the model fails to
        produce a usable shape."""
        if self._transport is None:
            raise ShapeAuthoringUnavailable(
                "shape authoring needs the live model (start the app with "
                "REACTIVE_AGENTS_LIVE=1); it is unavailable on the offline seam"
            )

        author = ShapeAuthor(self._transport, shapes_dir=self._shapes_dir)
        try:
            # One native-structured Gemma call -> a validated ShapeSpec (no write yet,
            # so the collision guard runs BEFORE anything touches disk).
            spec = await author.author(description, name_hint=name_hint)

            # Never clobber an existing shape: the model picks the name, so a collision
            # with a built-in / prior shape must be rejected, not silently overwritten.
            if self._config.get_shape(spec.name) is not None:
                raise ShapeNameConflict(spec.name)

            # Persist into the shapes dir the runtime loads, then re-load via the REAL
            # loader (the same round-trip guard author_and_write applies) so we never
            # ship a file the runtime would reject.
            path = write_shape(spec, shapes_dir=self._shapes_dir)
            reloaded = load_shape(spec.name, shapes_dir=self._shapes_dir)
            if (
                reloaded.execution != spec.execution
                or reloaded.is_unrollable != spec.is_unrollable
            ):
                raise ShapeError(
                    f"authored shape {spec.name!r} did not round-trip the loader "
                    f"(wrote execution={spec.execution!r}, read {reloaded.execution!r})"
                )
        except (MalformedOutputError, ShapeError):
            # Authoring failure (vague/unauthorable description, malformed JSON, or a
            # non-round-tripping write) — log with the cause, surface to the caller.
            logger.warning(
                "shape authoring failed for description %r", description[:200],
                exc_info=True,
            )
            raise

        view = self._config.get_shape(reloaded.name)
        if view is None:  # pragma: no cover - written-then-missing is an internal fault
            raise ShapeError(
                f"authored shape {reloaded.name!r} was written but is not in the "
                f"catalog the list reads — shapes-dir mismatch"
            )
        logger.info(
            "authored shape %r (execution=%s) from a description at %s",
            reloaded.name, reloaded.execution, path,
        )
        return view


def register_shape_author_routes(
    app: FastAPI, service: ShapeAuthorService
) -> ShapeAuthorService:
    """Mount the ``POST /shapes/author`` route onto ``app`` (app-agnostic).

    Returns the service so the wiring/test harness can keep a handle. Mirrors the
    mount style of :func:`chat_app.shape_config.register_shape_routes`."""

    @app.post("/shapes/author", status_code=201)
    async def author_shape(req: AuthorShapeRequest) -> dict:
        """DESCRIBE a shape -> Gemma authors the declarative file -> return its view.

        503 if no live model is wired; 409 if the authored name already exists; 422
        if the model cannot produce a usable shape from the description. On success
        returns the authored shape's full catalog view (so the UI lists it at once)
        with a 201."""
        try:
            return await service.author(req.description, name_hint=req.name_hint)
        except ShapeAuthoringUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except ShapeNameConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except (MalformedOutputError, ShapeError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"could not author a shape from that description: {exc}",
            )

    return service


__all__ = [
    "AuthorShapeRequest",
    "ShapeAuthorService",
    "ShapeAuthoringUnavailable",
    "ShapeNameConflict",
    "register_shape_author_routes",
]
