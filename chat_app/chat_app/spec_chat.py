"""STAGE B — the DISTINCT interactive spec-AUTHORING chat surface (s4/b1, d11).

The spec-definition chat is a surface SEPARATE from the task chat (d11): here the
USER drives a spec into shape over MULTIPLE back-and-forth turns — state intent,
read the drafted ruleset, critique it, watch it re-author, and finally approve to
compile + register a planner-loadable :class:`~specialization.model.CompiledSpec`.
The s7 React UI consumes exactly these routes; the legacy one-shot
``/specializations`` define→research→approve surface (``specializations.py``) is
left intact ALONGSIDE this one — this is the new interactive surface, not a
replacement.

It is built on a1's :class:`~specialization.conversation.SpecConversation` (the
genuine multi-turn re-author seam, NOT the single-shot ``compiler.condense_body``)
and reuses ``compiler.compile_spec`` + ``registry.register`` on approval — no
duplication of the compile/register write.

Surface (all in-process, d2 — no broker/pool, no second server):

- ``POST /spec-chats``                 — open a NEW authoring session → session id.
- ``POST /spec-chats/{id}/message``    — one user turn (the FIRST is the intent
                                         that authors draft 1; each later turn is
                                         a critique that RE-DRAFTS the body) →
                                         the revised draft preview + turn history.
- ``GET  /spec-chats/{id}``            — the full transcript: every user turn +
                                         every redrafted body + the current draft.
- ``POST /spec-chats/{id}/approve``    — compile + register the current body →
                                         the CompiledSpec, now loadable.
- ``POST /spec-chats/{id}/deny``       — discard the session without compiling.

Re-editable surface (s4/RC7 — re-open an EXISTING spec to view + edit, persisted
+ effective on the next run; the gap was that authoring could only make NEW specs):

- ``GET  /spec-chats/registered``        — LIST registered specs (body-free rows).
- ``GET  /spec-chats/registered/{name}`` — FETCH ONE by id: the full persisted
                                           spec (body + provenance) for the view.
- ``PUT  /spec-chats/registered/{name}`` — UPDATE + persist edits directly
                                           (provenance preserved; effective next run).
- ``POST /spec-chats/reopen``            — RE-OPEN an existing spec into an editable
                                           chat session (then refine via /message,
                                           re-register via /approve).

NEVER FREEZE (d4): the redraft (``SpecConversation.start`` / ``refine``) is a
SYNCHRONOUS chain call — on live phi4-mini it blocks on a slow, GPU-contended
round-trip. Every route that drives it (``/message``) and the compile-on-approval
(``/approve`` writes a file) is OFFLOADED off the asyncio event loop via
``asyncio.to_thread`` (specialist [required] rule: never block the one loop in an
``async def`` handler), so concurrent requests — incl ``/health`` and a second
chat's stream — stay responsive while a redraft is in flight.

Transport seam (d7/d8): the service is built with the wiring's transport — the
live :class:`~llm_framework.OllamaTransport` when ``transport_mode == "live"``,
else ``None`` so :class:`SpecConversation` uses its deterministic offline
FakeTransport seam (reproducible, GPU-free). Sessions are held in an in-process
service object on ``app.state.spec_chat_service`` (like ``SpecializationService``).
"""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm_framework import Transport
from specialization import RawDefinition, SpecRegistry
from specialization.conversation import (
    ConversationError,
    DraftPreview,
    SpecConversation,
)
from specialization.loader import SpecLoader


# --------------------------------------------------------------------------- #
# request / response models (Pydantic v2 — house style; 422 is automatic)
# --------------------------------------------------------------------------- #
class OpenSpecChatRequest(BaseModel):
    """Open a new spec-authoring session.

    ``name`` is the registry key the approved spec compiles under; ``description``
    is the planner-facing lookup text. The opening intent is the FIRST
    ``/message`` turn, not supplied here — so the user can open the surface and
    then converse."""

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=400)


class SpecChatMessageRequest(BaseModel):
    """One user turn in a spec-authoring chat.

    The first turn is the INTENT that authors draft 1; every later turn is a
    CRITIQUE that re-drafts the working body against the prior one."""

    message: str = Field(min_length=1, max_length=4000)


class TurnOut(BaseModel):
    """One recorded conversation turn (``role`` is ``"user"`` or ``"agent"``)."""

    role: str
    text: str


class DraftOut(BaseModel):
    """The working draft after a turn — the ruleset body plus the EXACT compiled
    markdown-with-frontmatter doc the user would approve (a real preview, not a
    paraphrase)."""

    name: str
    description: str
    body: str
    markdown: str
    turn: int  # how many author/refine rounds have produced this body (1-based)


class SpecChatView(BaseModel):
    """The full transcript of one session (the ``GET`` reopen view)."""

    session_id: str
    name: str
    description: str
    state: str
    started: bool
    draft: DraftOut | None  # the current working draft (None before the 1st turn)
    turns: list[TurnOut]


class MessageResponse(BaseModel):
    """The outcome of one user turn: the revised draft + the full turn history."""

    session_id: str
    state: str
    draft: DraftOut
    turns: list[TurnOut]


class ApproveResponse(BaseModel):
    """Receipt of a compile-on-approval: the spec is now registered + loadable."""

    session_id: str
    state: str
    name: str
    source: str
    registered: bool = True


class DenyResponse(BaseModel):
    """Receipt of a discard (the user rejected the body — nothing compiled)."""

    session_id: str
    state: str
    discarded: bool = True


# --- re-editable surface (s4/RC7): list / fetch-by-id / re-open / update --- #
class RegisteredSpecRow(BaseModel):
    """One row of the "pick a spec to re-open" list — body-free identity (d10)."""

    name: str
    description: str
    source: str


class RegisteredSpecOut(BaseModel):
    """The FULL persisted spec for the re-open VIEW (body + provenance included).

    This is the "fetch one by id for re-open" surface the SPA loads before
    editing — the planner-facing index is body-free, but the EDIT view needs the
    body and provenance."""

    name: str
    description: str
    source: str
    body: str
    research_trace_ref: str
    created_at: str


class ReopenSpecRequest(BaseModel):
    """Re-open an EXISTING registered spec into an editable chat session."""

    name: str = Field(min_length=1, max_length=120)


class UpdateSpecRequest(BaseModel):
    """Directly edit + persist an existing spec (no chat round-trip).

    Both fields are optional; only the supplied ones are overlaid (identity +
    provenance preserved). At least one must be present (enforced in the route)."""

    description: str | None = Field(default=None, max_length=400)
    body: str | None = Field(default=None, min_length=1)


# --------------------------------------------------------------------------- #
# the in-process session service
# --------------------------------------------------------------------------- #
class SpecChatService:
    """Holds + drives the live spec-authoring conversations (one per session).

    Each session is a :class:`SpecConversation` built over the SHARED registry
    and the configured transport (live phi or the offline seam). Held on
    ``app.state.spec_chat_service`` — in-process, no broker/pool (d2).

    The redraft methods here are SYNCHRONOUS (they run the blocking chain call);
    the route layer wraps them in ``asyncio.to_thread`` so the event loop never
    blocks (d4). Keeping them sync — rather than ``async`` with an internal
    offload — means the service stays a plain testable object and the offload
    decision lives at the one place that owns the loop (the route)."""

    def __init__(
        self,
        registry: SpecRegistry,
        *,
        transport: Optional[Transport] = None,
    ) -> None:
        self._registry = registry
        # The SAME context-scoped loader the runtime composes a node's spec body
        # through (SubAgent → SpecLoader.load_body → registry.load). Held so the
        # "effective on the next run" read goes through the REAL runtime path, not
        # a second registry handle.
        self._loader = SpecLoader(registry)
        self._transport = transport
        self._sessions: dict[str, SpecConversation] = {}
        self._counter = 0

    # -- open / lookup ------------------------------------------------------ #
    def open_session(self, name: str, description: str) -> str:
        """Open a new authoring session and return its id. Trivial + non-blocking
        (no chain call happens until the first ``/message``)."""
        self._counter += 1
        session_id = f"spec-chat-{self._counter}"
        raw = RawDefinition(name=name.strip(), description=description.strip(), intent="")
        self._sessions[session_id] = SpecConversation(
            raw, registry=self._registry, transport=self._transport
        )
        return session_id

    # -- re-editable surface (s4/RC7): list / fetch-by-id / re-open / update - #
    def list_registered(self) -> list[RegisteredSpecRow]:
        """Every REGISTERED spec as a body-free identity row (the "pick one to
        re-open" list). Reuses the registry's body-free index (d10)."""
        return [
            RegisteredSpecRow(name=e.name, description=e.description, source=e.source)
            for e in self._registry.index()
        ]

    def get_registered(self, name: str) -> RegisteredSpecOut:
        """The FULL persisted spec (body + provenance) for the re-open VIEW.

        Raises ``KeyError`` if no spec named ``name`` is registered."""
        spec = self._registry.load(name)  # KeyError if absent
        return RegisteredSpecOut(
            name=spec.name,
            description=spec.description,
            source=spec.source,
            body=spec.body,
            research_trace_ref=spec.research_trace_ref,
            created_at=spec.created_at,
        )

    def reopen_session(self, name: str) -> str:
        """Open an editable chat session SEEDED from an existing registered spec
        (RC7) and return its id. The session begins already-started with the
        existing body as its draft, so the next ``/message`` is a refine and
        ``/approve`` re-registers under the SAME name. ``KeyError`` if absent."""
        spec = self._registry.load(name)  # KeyError if absent
        self._counter += 1
        session_id = f"spec-chat-{self._counter}"
        self._sessions[session_id] = SpecConversation.reopen(
            spec, registry=self._registry, transport=self._transport
        )
        return session_id

    def update_spec(
        self,
        name: str,
        *,
        description: str | None = None,
        body: str | None = None,
    ) -> RegisteredSpecOut:
        """Directly edit + persist an existing spec (no chat round-trip) through
        ``SpecRegistry.update`` — identity + provenance preserved, effective on
        the next run (it overwrites the same doc the loader reads). ``KeyError``
        if absent; ``ValueError`` on a blank body. This is a file write, so the
        route offloads it off the event loop (d4)."""
        spec = self._registry.update(name, description=description, body=body)
        return RegisteredSpecOut(
            name=spec.name,
            description=spec.description,
            source=spec.source,
            body=spec.body,
            research_trace_ref=spec.research_trace_ref,
            created_at=spec.created_at,
        )

    def effective_body(self, name: str) -> str:
        """The body a node would load on its NEXT run — read through the SAME
        :class:`~specialization.loader.SpecLoader` path the runtime composes a
        spec with. Used to prove an edit is effective, not just stored."""
        return self._loader.load_body(name)

    def _conversation(self, session_id: str) -> SpecConversation:
        conv = self._sessions.get(session_id)
        if conv is None:
            raise KeyError(session_id)
        return conv

    # -- drive a turn (BLOCKING chain call — offloaded by the route) -------- #
    def drive_message(self, session_id: str, message: str) -> DraftPreview:
        """Apply one user turn: ``start`` the FIRST time (the intent authors draft
        1), ``refine`` every later time (the critique re-drafts the body).

        SYNCHRONOUS by design — this is the blocking phi round-trip on the live
        path. The route offloads it off the event loop (d4)."""
        conv = self._conversation(session_id)
        if not conv.started:
            return conv.start(message)
        return conv.refine(message)

    # -- terminal: compile + register (BLOCKING file write — offloaded) ----- #
    def approve(self, session_id: str) -> tuple[str, str, str]:
        """Compile + register the current body (reuses ``compiler.compile_spec`` +
        ``registry.register`` via the conversation's ``approve``). Returns
        ``(state, name, source)``. The registry write is file I/O, so the route
        offloads this too."""
        conv = self._conversation(session_id)
        spec = conv.approve()
        return conv.state, spec.name, spec.source

    def deny(self, session_id: str) -> str:
        """Discard the session without compiling. Returns the terminal state."""
        conv = self._conversation(session_id)
        conv.deny()
        return conv.state

    # -- read-only transcript ---------------------------------------------- #
    def view(self, session_id: str) -> SpecChatView:
        """The full transcript: identity, lifecycle state, current draft, history.

        Pure in-memory reads (no chain call), safe to run on the loop."""
        conv = self._conversation(session_id)
        return SpecChatView(
            session_id=session_id,
            name=conv.raw.name,
            description=conv.raw.description,
            state=conv.state,
            started=conv.started,
            draft=_draft_out(conv) if conv.started else None,
            turns=[TurnOut(role=t.role, text=t.text) for t in conv.history],
        )


def _rounds(conv: SpecConversation) -> int:
    """How many author/refine rounds have produced the current body — derived
    label-agnostically from the history (one ``agent`` turn per round)."""
    return sum(1 for t in conv.history if t.role == "agent")


def _draft_out(conv: SpecConversation) -> DraftOut:
    """Build a :class:`DraftOut` (body + compiled markdown preview) from a
    conversation's CURRENT body."""
    preview = DraftPreview(
        name=conv.raw.name,
        description=conv.raw.description,
        body=conv.body,
        turn=_rounds(conv),
    )
    return _draft_out_from_preview(preview)


def _draft_out_from_preview(preview: DraftPreview) -> DraftOut:
    """Project a :class:`DraftPreview` (the value ``start``/``refine`` return)
    into the wire model, including its compiled markdown doc."""
    return DraftOut(
        name=preview.name,
        description=preview.description,
        body=preview.body,
        markdown=preview.to_markdown(),
        turn=preview.turn,
    )


# --------------------------------------------------------------------------- #
# route registration — the app-agnostic mount entrypoint (mirrors
# specializations.register_specialization_routes so routes.register_routes mounts
# it alongside the legacy surface with no rewrite)
# --------------------------------------------------------------------------- #
def register_spec_chat_routes(
    app: FastAPI, service: SpecChatService
) -> SpecChatService:
    """Mount the open/message/transcript/approve/deny routes onto ``app``.

    Returns the service so a caller (test harness, or the unified app) keeps a
    handle (e.g. to stash on ``app.state``)."""

    @app.post("/spec-chats", status_code=201)
    async def open_spec_chat(req: OpenSpecChatRequest) -> dict:
        """Open a NEW spec-authoring session (no chain call yet) and return its
        id. The client then drives turns via ``POST /spec-chats/{id}/message``."""
        session_id = service.open_session(req.name, req.description)
        return {
            "session_id": session_id,
            "name": req.name.strip(),
            "description": req.description.strip(),
            "state": "open",
            "started": False,
        }

    @app.post("/spec-chats/{session_id}/message", response_model=MessageResponse)
    async def spec_chat_message(
        session_id: str, req: SpecChatMessageRequest
    ) -> MessageResponse:
        """One user turn → RE-DRAFT the body (intent authors draft 1; each later
        turn is a critique). The blocking redraft chain call is OFFLOADED off the
        event loop (d4 non-freeze), so the server stays responsive throughout."""
        try:
            preview = await asyncio.to_thread(
                service.drive_message, session_id, req.message
            )
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no spec-chat {session_id!r}")
        except ConversationError as exc:
            # Out-of-order / terminal-state authoring call.
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            # e.g. empty critique slipping past the min_length guard.
            raise HTTPException(status_code=422, detail=str(exc))
        view = service.view(session_id)
        return MessageResponse(
            session_id=session_id,
            state=view.state,
            draft=_draft_out_from_preview(preview),
            turns=view.turns,
        )

    # ---- re-editable surface (s4/RC7) — list / fetch-by-id / re-open / update.
    # The literal "registered" routes are declared BEFORE the "/spec-chats/
    # {session_id}" catch-all below so the path "registered" is never captured as
    # a session id (FastAPI matches in declaration order). ----
    @app.get("/spec-chats/registered", response_model=list[RegisteredSpecRow])
    async def list_registered_specs() -> list[RegisteredSpecRow]:
        """LIST every registered specialization as a body-free identity row — the
        "pick one to re-open" surface the SPA renders (d10: no bodies here)."""
        return service.list_registered()

    @app.get(
        "/spec-chats/registered/{name}", response_model=RegisteredSpecOut
    )
    async def get_registered_spec(name: str) -> RegisteredSpecOut:
        """FETCH ONE spec BY ID (name) for the re-open VIEW — the FULL persisted
        spec (body + provenance), what the SPA loads before editing."""
        try:
            return service.get_registered(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no registered spec {name!r}")

    @app.put(
        "/spec-chats/registered/{name}", response_model=RegisteredSpecOut
    )
    async def update_registered_spec(
        name: str, req: UpdateSpecRequest
    ) -> RegisteredSpecOut:
        """UPDATE + PERSIST an existing spec directly (no chat round-trip).

        Overlays the supplied ``description``/``body`` while preserving identity +
        provenance, and re-writes the SAME doc the loader reads — so the edit is
        EFFECTIVE ON THE NEXT RUN. The registry file write is offloaded off the
        event loop (d4)."""
        if req.description is None and req.body is None:
            raise HTTPException(
                status_code=422,
                detail="update requires at least one of 'description' or 'body'",
            )
        try:
            return await asyncio.to_thread(
                service.update_spec, name, description=req.description, body=req.body
            )
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no registered spec {name!r}")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/spec-chats/reopen", response_model=SpecChatView, status_code=201)
    async def reopen_spec_chat(req: ReopenSpecRequest) -> SpecChatView:
        """RE-OPEN an existing registered spec into an EDITABLE chat session
        (RC7). The returned view begins already-started with the existing body as
        its draft; the client then drives edits via ``POST /spec-chats/{id}/
        message`` (refine) and ``/approve`` re-registers under the SAME name."""
        try:
            session_id = service.reopen_session(req.name.strip())
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"no registered spec {req.name!r}"
            )
        return service.view(session_id)

    @app.get("/spec-chats/{session_id}", response_model=SpecChatView)
    async def get_spec_chat(session_id: str) -> SpecChatView:
        """The full transcript: every user turn + every redrafted body + the
        current draft (the s7 UI's reopen view). Pure in-memory read."""
        try:
            return service.view(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no spec-chat {session_id!r}")

    @app.post("/spec-chats/{session_id}/approve", response_model=ApproveResponse)
    async def approve_spec_chat(session_id: str) -> ApproveResponse:
        """Compile + register the current body (reuses ``compiler.compile_spec`` +
        ``registry.register``) → the spec is now planner-loadable. The registry
        file write is offloaded off the event loop (d4)."""
        try:
            state, name, source = await asyncio.to_thread(service.approve, session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no spec-chat {session_id!r}")
        except ConversationError as exc:
            # Nothing authored yet, or already decided.
            raise HTTPException(status_code=409, detail=str(exc))
        return ApproveResponse(
            session_id=session_id, state=state, name=name, source=source
        )

    @app.post("/spec-chats/{session_id}/deny", response_model=DenyResponse)
    async def deny_spec_chat(session_id: str) -> DenyResponse:
        """Discard the session without compiling (the user rejected the body)."""
        try:
            state = service.deny(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no spec-chat {session_id!r}")
        except ConversationError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return DenyResponse(session_id=session_id, state=state)

    return service


def build_spec_chat_service(
    registry: SpecRegistry,
    *,
    transport: Optional[Transport] = None,
) -> SpecChatService:
    """Convenience constructor: a service over the registry with the configured
    transport (live phi or the offline seam). ``routes.register_routes`` calls
    this from the wiring; a test harness calls it with no transport (offline)."""
    return SpecChatService(registry, transport=transport)


__all__ = [
    "OpenSpecChatRequest",
    "SpecChatMessageRequest",
    "TurnOut",
    "DraftOut",
    "SpecChatView",
    "MessageResponse",
    "ApproveResponse",
    "DenyResponse",
    "RegisteredSpecRow",
    "RegisteredSpecOut",
    "ReopenSpecRequest",
    "UpdateSpecRequest",
    "SpecChatService",
    "register_spec_chat_routes",
    "build_spec_chat_service",
]
