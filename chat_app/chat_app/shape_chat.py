"""s17 (d18a/d249, user-raised parity gap) — the CONVERSATIONAL shape-authoring chat.

The Shapes screen's describe/refine boxes were ONE-SHOT: create wrote a file from a
single description, refine rewrote the on-disk file per instruction — no transcript,
no draft stage, no approve gate. That is NOT the free-flowing iterative authoring the
spec chat gives specializations (d18a), and the user called the asymmetry out.

This module mirrors the spec chat's flow for shapes:

- ``open`` starts a session — a fresh CREATE draft, or a REFINE session seeded from an
  existing shape's on-disk definition.
- each ``message`` drives ONE live authoring turn: the first message AUTHORS a draft
  (``ShapeAuthor.author`` — catalog usage-context included, d249); every later message
  REFINES the IN-SESSION draft (``ShapeAuthor.refine`` on the in-memory prior — the
  file is NEVER touched mid-conversation), so each turn demonstrably builds on the
  previous version (o11).
- ``approve`` persists the draft through the SAME write + round-trip-loader guard the
  one-shot path uses (collision-checked for a create); ``deny`` discards it.

Authoring needs the LIVE model: on the offline/stub seam every drive route surfaces
503 (mirrors :class:`chat_app.shape_authoring.ShapeAuthorService`) — a shape is never
stubbed into existence. Sessions are in-memory (mirrors the spec chat's session map):
a draft is cheap to re-create, so process restart simply drops open drafts.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_runtime.selfheal import MalformedOutputError
from agent_runtime.shape_author import ShapeAuthor, write_shape
from agent_runtime.shapes import ShapeError, ShapeSpec, load_shape

from .shape_config import ShapeConfigService

logger = logging.getLogger(__name__)

__all__ = [
    "ShapeChatService",
    "ShapeChatUnavailable",
    "register_shape_chat_routes",
]


class ShapeChatUnavailable(RuntimeError):
    """Raised when a drive needs the live model but the app runs on the stub seam."""


@dataclass
class _Session:
    """One in-memory shape-authoring conversation."""

    session_id: str
    mode: str  # "create" | "refine"
    turns: list[dict[str, str]] = field(default_factory=list)
    draft: Optional[ShapeSpec] = None
    state: str = "open"  # open | approved | denied
    refine_of: Optional[str] = None  # the on-disk shape a refine session edits


def _draft_view(draft: Optional[ShapeSpec]) -> Optional[dict[str, Any]]:
    """The draft as the compact preview the chat renders (None before turn 1)."""
    if draft is None:
        return None
    return {
        "name": draft.name,
        "description": draft.description,
        "execution": draft.execution,
        "max_iter": int(draft.max_iter),
    }


# --------------------------------------------------------------------------- #
# request/response models (mirror the spec chat's shapes)
# --------------------------------------------------------------------------- #
class OpenShapeChatRequest(BaseModel):
    """Open a session. ``refine_of`` seeds the draft from that existing shape."""

    refine_of: Optional[str] = Field(default=None, max_length=120)


class ShapeChatMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ShapeChatView(BaseModel):
    session_id: str
    mode: str
    state: str
    refine_of: Optional[str] = None
    turns: list[dict[str, str]]
    draft: Optional[dict[str, Any]] = None


# --------------------------------------------------------------------------- #
# the service
# --------------------------------------------------------------------------- #
class ShapeChatService:
    """Drive a multi-turn shape-authoring conversation over an in-session DRAFT.

    Composes :class:`~agent_runtime.shape_author.ShapeAuthor` (the live authoring
    turns) with :class:`~chat_app.shape_config.ShapeConfigService` (the catalog the
    approve gate checks + the view the caller gets back). The on-disk file is
    touched ONLY at :meth:`approve`.
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
        self._sessions: dict[str, _Session] = {}

    @property
    def available(self) -> bool:
        return self._transport is not None

    # ---- session lifecycle ---- #
    def open_session(self, *, refine_of: Optional[str] = None) -> _Session:
        """Start a conversation — a fresh create, or a refine seeded from disk."""
        refine_of = (refine_of or "").strip() or None
        draft: Optional[ShapeSpec] = None
        mode = "create"
        if refine_of:
            if self._config.get_shape(refine_of) is None:
                raise HTTPException(status_code=404, detail=f"no shape {refine_of!r}")
            # Seed the draft from the REAL on-disk definition so turn 1 builds on it.
            draft = load_shape(refine_of, shapes_dir=self._shapes_dir)
            mode = "refine"
        sess = _Session(
            session_id=f"shapechat-{uuid.uuid4().hex[:12]}",
            mode=mode,
            draft=draft,
            refine_of=refine_of,
        )
        self._sessions[sess.session_id] = sess
        return sess

    def _session(self, session_id: str) -> _Session:
        sess = self._sessions.get(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"no shape chat {session_id!r}")
        if sess.state != "open":
            raise HTTPException(
                status_code=409, detail=f"shape chat {session_id!r} is {sess.state}"
            )
        return sess

    # ---- the conversational drive ---- #
    async def message(self, session_id: str, text: str) -> _Session:
        """One authoring turn: author the first draft, or refine the current one."""
        if self._transport is None:
            raise ShapeChatUnavailable(
                "shape authoring needs the live model (start the app with "
                "REACTIVE_AGENTS_LIVE=1); it is unavailable on the offline seam"
            )
        sess = self._session(session_id)
        text = str(text).strip()
        author = ShapeAuthor(self._transport, shapes_dir=self._shapes_dir)
        try:
            if sess.draft is None:
                draft = await author.author(text)
            else:
                # Refine the IN-SESSION draft (in-memory prior) — never the file. A
                # CREATE draft has no file to orphan, so a requested rename is honored
                # (live catch: "rename it to X" was silently kept at the old name);
                # a REFINE session keeps its on-disk name.
                draft = await author.refine(
                    sess.draft, text, keep_name=(sess.mode != "create")
                )
        except (MalformedOutputError, ShapeError):
            logger.warning(
                "shape chat turn failed (session %s, message %r)",
                session_id, text[:200], exc_info=True,
            )
            raise
        sess.turns.append({"role": "user", "text": text})
        sess.turns.append({
            "role": "assistant",
            "text": (
                f"Draft updated: '{draft.name}' [{draft.execution}] — "
                f"{draft.description}"
            ),
        })
        sess.draft = draft
        return sess

    # ---- the persistence gate ---- #
    def approve(self, session_id: str) -> dict[str, Any]:
        """Persist the draft through the same write + round-trip guard as one-shot."""
        sess = self._session(session_id)
        if sess.draft is None:
            raise HTTPException(status_code=409, detail="nothing drafted yet")
        draft = sess.draft
        # A CREATE must not clobber an existing shape; a REFINE overwrites ITS OWN
        # file (the draft keeps the seeded name — ShapeAuthor.refine forces it).
        if sess.mode == "create" and self._config.get_shape(draft.name) is not None:
            raise HTTPException(
                status_code=409,
                detail=f"a shape named {draft.name!r} already exists — ask the chat "
                "to rename the draft, then approve again",
            )
        path = write_shape(draft, shapes_dir=self._shapes_dir)
        reloaded = load_shape(draft.name, shapes_dir=self._shapes_dir)
        if reloaded.execution != draft.execution:
            raise ShapeError(
                f"approved shape {draft.name!r} did not round-trip the loader "
                f"(wrote execution={draft.execution!r}, read {reloaded.execution!r})"
            )
        sess.state = "approved"
        view = self._config.get_shape(reloaded.name)
        logger.info("shape chat approved %r at %s", reloaded.name, path)
        if view is None:  # pragma: no cover - written-then-missing internal fault
            raise ShapeError(
                f"approved shape {reloaded.name!r} missing from the catalog view"
            )
        return view

    def deny(self, session_id: str) -> None:
        """Discard the draft; nothing was ever written."""
        sess = self._session(session_id)
        sess.state = "denied"

    def view(self, session_id: str) -> ShapeChatView:
        sess = self._sessions.get(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"no shape chat {session_id!r}")
        return ShapeChatView(
            session_id=sess.session_id,
            mode=sess.mode,
            state=sess.state,
            refine_of=sess.refine_of,
            turns=list(sess.turns),
            draft=_draft_view(sess.draft),
        )


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
def register_shape_chat_routes(
    app: FastAPI, service: ShapeChatService
) -> ShapeChatService:
    """Mount the shape-chat routes (mirrors ``register_spec_chat_routes``)."""

    @app.post("/shape-chat", status_code=201)
    async def open_shape_chat(req: OpenShapeChatRequest) -> ShapeChatView:
        sess = service.open_session(refine_of=req.refine_of)
        return service.view(sess.session_id)

    @app.get("/shape-chat/{session_id}")
    async def get_shape_chat(session_id: str) -> ShapeChatView:
        return service.view(session_id)

    @app.post("/shape-chat/{session_id}/message")
    async def shape_chat_message(
        session_id: str, req: ShapeChatMessageRequest
    ) -> ShapeChatView:
        try:
            sess = await service.message(session_id, req.message)
        except ShapeChatUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (MalformedOutputError, ShapeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return service.view(sess.session_id)

    @app.post("/shape-chat/{session_id}/approve")
    async def approve_shape_chat(session_id: str) -> dict:
        try:
            view = service.approve(session_id)
        except ShapeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"approved": True, "shape": view}

    @app.post("/shape-chat/{session_id}/deny")
    async def deny_shape_chat(session_id: str) -> dict:
        service.deny(session_id)
        return {"denied": True}

    return service
