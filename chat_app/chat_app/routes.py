"""STAGE B — the complete HTTP surface of the unified chat app (s7/b3).

This module mounts the WHOLE backend the frontend (b2) consumes onto the ONE
:class:`fastapi.FastAPI` instance the a1 skeleton builds — wired to the REAL
in-process subsystems (d2/d11): the a2 SSE bridge over the shared reactive
:class:`~reactive_tools.EventPlane`, the b1 durable :class:`~chat_app.persistence.ChatStore`,
and the s6 :class:`~agent_runtime.AgentRuntime` driven on the deterministic STUB
transport (d12 — no live phi; the live Ollama transport swaps in at s8 with no
other change). Nothing here forks a shell or stands up a second service.

The surface (all on the single app, single uvicorn instance — d2/d8):

- ``POST   /chats``                  — open a NEW empty chat (id for the sidebar
                                       + the live stream the client opens next).
- ``GET    /chats``                  — list prior chats for the reopen sidebar.
- ``GET    /chats/{id}``             — one chat's FULL history: turns (user
                                       request + streamed reasoning events +
                                       final response) + artifact refs (reopen).
- ``POST   /chats/{id}/message``     — submit a request → drive the real s6
                                       AgentRuntime/planner DAG on the stub
                                       transport, persist the turn + any produced
                                       artifacts via b1, return the run summary.
- ``GET    /chats/{id}/stream``      — the a2 SSE stream of live reasoning/
                                       progress events FOR THIS CHAT.
- ``GET    /artifacts/{id}``         — download an artifact with correct mime +
                                       ``Content-Disposition`` (``?inline=1`` to
                                       render in-browser instead of downloading).

CRITICAL UNIFICATION (a2/a3 carry-forward, d2/d11): a3 deliberately built the
HITL approval gate as APP-AGNOSTIC modules (``approval.py`` + ``specializations.py``
exposing :func:`~chat_app.specializations.register_specialization_routes`). b3
REUSES them verbatim — :func:`register_routes` mounts s5's specialization
define/approve endpoints onto THIS SAME app (no reimplementation, no second
server). The stdlib ``http.server`` placeholder (``specialization/evidence/serve_ui.py``)
is replaced by a thin launcher for this one app, so there is exactly ONE server.

PER-CHAT STREAM SCOPING (b3 design note)
----------------------------------------
The in-process :class:`~reactive_tools.Event` carries no ``chat_id``, so a single
shared plane cannot be filtered down to one chat's events. b3 therefore scopes
streams with a :class:`ChatStreamHub`: each chat gets its OWN in-process
:class:`~reactive_tools.EventPlane`, and ``POST /chats/{id}/message`` builds its
per-turn :class:`~agent_runtime.AgentRuntime` with that chat's plane — so
``GET /chats/{id}/stream`` sees exactly (and only) this chat's runtime lifecycle.
a2's shared-plane ``/chat`` + ``/events`` endpoints are left intact for
back-compat (the all-chats firehose); the per-chat surface is additive.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from reactive_tools import EventPlane, Subscription
from agent_runtime import (
    CLARIFICATION_KIND,
    EVENT_MISSING_SPECIALIST,
    EVENT_NEEDS_CLARIFICATION,
    EVENT_NODE_CANCELLED,
    EVENT_NODE_COLLISION,
    EVENT_NODE_COLLISION_RESOLVED,
    EVENT_NODE_DONE,
    EVENT_NODE_FAILED,
    EVENT_NODE_HEALED,
    EVENT_NODE_INLINE_FIXED,
    EVENT_NODE_LAUNCHED,
    EVENT_NODE_REPLANNED,
    EVENT_NODE_REVIEW,
    EVENT_NODE_SKIPPED,
    EVENT_NODE_VERIFIABLE,
    EVENT_NODE_VERIFY_FAILED,
    load_shapes,
)
from chat_app.specializations import (
    build_specialization_service,
    register_specialization_routes,
)
from chat_app.spec_chat import (
    build_spec_chat_service,
    register_spec_chat_routes,
)
from chat_app.approval import HttpApprovalGate
from agent_runtime import stub
from chat_app.agentic import resume_agentic, run_agentic, run_offline
from chat_app.shape_config import ShapeConfigService, register_shape_routes
from chat_app.shape_authoring import (
    ShapeAuthorService,
    register_shape_author_routes,
)
from chat_app.runs import RunManager
from chat_app.workflow import make_live_report_producer, schedule_workflow_spec

# The runtime lifecycle kinds the per-chat SSE stream relays — the same set the
# a2 ``/events`` firehose uses (tool_call/tool_result ride the same plane so a
# tool-using DAG stays observable; the offline demo DAG is tool-less).
RUNTIME_EVENT_KINDS: tuple[str, ...] = (
    EVENT_NODE_LAUNCHED,
    EVENT_NODE_DONE,
    EVENT_NODE_FAILED,
    EVENT_NODE_HEALED,
    EVENT_NODE_CANCELLED,
    EVENT_NODE_REPLANNED,
    EVENT_NODE_SKIPPED,
    # The verify-gate + collision lifecycle kinds (d9 verify gate, d12 collision,
    # d16 produce->review->inline-fix). The runtime EMITS these (runtime.py
    # _emit), and the SPA's DAG/lifecycle reducer handles every one — but the
    # plane's `subscribe(kinds=...)` is a strict whitelist (event_plane.py), so
    # omitting them silently dropped the entire `verifiable` state and the
    # inline-fix/collision flows from the live stream. Forward them so the DAG
    # reflects the full pending->in-progress->verifiable->done lifecycle live.
    EVENT_NODE_VERIFIABLE,
    EVENT_NODE_REVIEW,
    EVENT_NODE_INLINE_FIXED,
    EVENT_NODE_VERIFY_FAILED,
    EVENT_NODE_COLLISION,
    EVENT_NODE_COLLISION_RESOLVED,
    # MISSING-SPECIALIST notify (s4 M1, RC8): a plan that needs an unavailable
    # specialist publishes this kind so the chat is NOTIFIED live and shown the
    # SSE-fallback / define-and-resume CHOICE — never a silent failure.
    EVENT_MISSING_SPECIALIST,
    # AMBIGUITY CLARIFICATION notify (scenario-2): a run that pauses to ask the
    # user a clarifying question publishes this kind so the chat is NOTIFIED live
    # and shown the question — never a silent guess of the missing detail.
    EVENT_NEEDS_CLARIFICATION,
    "tool_call",
    "tool_result",
)

# Per-connection queue bound + backpressure policy (house style: bound every
# per-connection queue with an explicit policy). The producer (the runtime) and
# the consumer (this SSE generator) share the ONE event loop, so a bounded queue
# here means LOSSLESS backpressure: if a slow client lets the queue fill, the
# in-process ``plane.publish`` simply awaits the drain rather than dropping an
# event. 512 is comfortably above the handful of events a single turn emits.
STREAM_QUEUE_MAXSIZE = 512


# --------------------------------------------------------------------------- #
# request/response models (Pydantic v2 — house style; 422 is automatic)
# --------------------------------------------------------------------------- #
class NewChatRequest(BaseModel):
    """Open a new chat. ``title`` is optional; a default is used if omitted."""

    title: str | None = Field(default=None, max_length=120)


class MessageRequest(BaseModel):
    """One submitted request in a chat. ``message`` is the user's prompt;
    ``topic`` optionally names the subject the demo DAG renders two ways."""

    message: str = Field(min_length=1, max_length=4000)
    topic: str | None = Field(default=None, max_length=400)


class NodeStateOut(BaseModel):
    """Terminal state of one DAG node, surfaced in the message response."""

    node_id: str
    status: str
    attempts: int
    error: str | None = None


class ArtifactOut(BaseModel):
    """An artifact ref surfaced after a run (download via ``GET /artifacts/{id}``)."""

    artifact_id: str
    filename: str
    mime: str
    size: int
    node_id: str | None = None


# --------------------------------------------------------------------------- #
# workflow-spec arming (Scenario B, s9/a4) — request models
# --------------------------------------------------------------------------- #
class ScheduleIn(BaseModel):
    """The schedule a workflow spec is armed with (maps to ``ScheduleSpec``).

    ``interval`` fires every ``interval_seconds`` while the app runs; ``one_shot``
    fires once after ``initial_delay``. ``max_fires`` bounds an interval (the safe
    self-test uses a small N so the recurrence never loops forever)."""

    kind: str = Field(default="interval", pattern="^(interval|one_shot)$")
    interval_seconds: float = Field(default=60.0, gt=0, le=86400.0)
    max_fires: int | None = Field(default=None, gt=0, le=100)
    initial_delay: float | None = Field(default=None, ge=0, le=86400.0)


class DeliveryIn(BaseModel):
    """The delivery channel a workflow spec is armed with (maps to ``DeliverySpec``).

    ``recipient`` defaults to ``None`` → the email tool sends to the configured
    own address (safe self-test); a value is honored but the safe-test keeps it
    self-addressed."""

    channel: str = Field(default="email", pattern="^(email)$")
    recipient: str | None = Field(default=None, max_length=320)


class ArmWorkflowRequest(BaseModel):
    """Arm an already-registered spec as a scheduled workflow (explicit, d8 safety).

    ``spec_name`` is a spec already registered (e.g. authored + approved via the
    interactive spec-chat). The schedule + delivery are LAYERED onto it at arm
    time — that is what turns a chat-defined output-shaping ruleset into a
    "daily brief"-style workflow the in-process scheduler fires and emails."""

    spec_name: str = Field(min_length=1, max_length=120)
    schedule: ScheduleIn = Field(default_factory=ScheduleIn)
    delivery: DeliveryIn = Field(default_factory=DeliveryIn)


class MessageResponse(BaseModel):
    """The outcome of driving the DAG for one submitted message.

    The live per-node lifecycle rode the chat's SSE stream; this is the final
    summary the POST returns once the in-process run completes + persisted.

    MISSING-SPECIALIST PAUSE (s4 M1, RC8): when the plan needs a specialist no
    registered spec provides, the run is PAUSED instead of run — ``ok`` is False,
    ``missing_specialist`` is True, and ``pending`` carries the notify/CHOICE
    payload (``resume_token`` + ``choices`` + the unmet ``missing`` nodes) the
    client echoes back to ``POST /chats/{id}/resume``. ``node_states``/``outputs``
    are empty (nothing ran). For a normal run ``missing_specialist`` is False and
    ``pending`` is null (back-compat: existing clients ignore the new fields)."""

    chat_id: str
    turn_index: int
    ok: bool
    launch_order: list[str]
    node_states: list[NodeStateOut]
    outputs: dict[str, str]
    artifacts: list[ArtifactOut]
    missing_specialist: bool = False
    pending: dict[str, Any] | None = None


class ResumeRequest(BaseModel):
    """Resolve a paused run (missing-specialist OR ambiguity-clarification).

    ``resume_token`` is the one the pause's ``pending`` payload carried.

    MISSING-SPECIALIST pause (s4 M1, RC8): ``choice`` is ``sse_fallback`` (run the
    unmet nodes spec-less, output streamed) or ``define_and_resume`` (stamp a
    now-defined spec onto them); for ``define_and_resume`` supply ``spec_name`` —
    the registered specialization to apply to every unmet node — or the per-node
    ``defined_specs`` map.

    AMBIGUITY-CLARIFICATION pause (scenario-2): the user ANSWERS the planner's
    clarifying question in ``answer`` (``choice`` is not used); the run resumes by
    re-driving the plan on the clarified intent. ``choice`` is therefore optional
    (validated per pause kind in the resume handler), not a fixed enum on the
    model, so one request type serves both pauses."""

    resume_token: str = Field(min_length=1, max_length=120)
    choice: str | None = Field(default=None, max_length=40)
    spec_name: str | None = Field(default=None, max_length=120)
    defined_specs: dict[str, str] | None = None
    # The user's answer to the planner's clarifying question (clarification pause).
    answer: str | None = Field(default=None, max_length=4000)


# --------------------------------------------------------------------------- #
# per-chat stream scoping
# --------------------------------------------------------------------------- #
class ChatStreamHub:
    """A registry of per-chat in-process event planes (b3 stream scoping).

    Each chat is handed its OWN :class:`~reactive_tools.EventPlane` (lazily
    created). The message run publishes a chat's runtime lifecycle onto that
    plane; the chat's SSE stream subscribes to the SAME plane — so a stream only
    ever sees its own chat's events. Held on ``app.state`` (one hub per app),
    constructed in :func:`register_routes`. Planes are lightweight (a list + an
    int); they are kept for the life of the process, bounded by the number of
    chats touched this run.
    """

    def __init__(self) -> None:
        self._planes: dict[str, EventPlane] = {}

    def plane_for(self, chat_id: str) -> EventPlane:
        plane = self._planes.get(chat_id)
        if plane is None:
            plane = EventPlane()
            self._planes[chat_id] = plane
        return plane


def _jsonable(obj: Any) -> Any:
    """Best-effort coerce an event payload to a JSON-serialisable form so a
    non-serialisable tool value degrades to ``repr`` instead of failing the
    stream (mirrors the a2 helper)."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        return repr(obj)


# --------------------------------------------------------------------------- #
# route registration — the b3 unification entrypoint
# --------------------------------------------------------------------------- #
def register_routes(app: FastAPI) -> None:
    """Mount the FULL b3 surface onto ``app`` (called from ``create_app``).

    Reads the shared in-process composition from ``app.state.wiring`` (built by
    the a1 skeleton) and mounts:

    1. the chat surface (chats / message / stream / artifacts), and
    2. the UNIFIED s5 specialization define/approve routes, by reusing a3's
       app-agnostic :func:`register_specialization_routes` against THIS app and
       the wiring's own engine + registry (no reimplementation, no second
       server).

    Side effects on ``app.state``: ``stream_hub`` (the per-chat plane registry)
    and ``spec_service`` (so the lifespan can ``cancel_all`` parked approvers at
    shutdown — see the lifespan note in ``specializations.py``)."""
    hub = ChatStreamHub()
    app.state.stream_hub = hub
    # Background-run registry (s1/a1 freeze decouple): POST /chats/{id}/runs
    # submits a run here and returns its id immediately; the lifespan calls
    # ``run_manager.shutdown()`` to cancel any in-flight run at teardown.
    run_manager = RunManager()
    app.state.run_manager = run_manager
    # MISSING-SPECIALIST PAUSE registry (s4 M1, RC8): a run that pauses on an
    # unavailable specialist stashes its paused PlanDAG + run context here keyed by
    # the pending payload's resume_token, so POST /chats/{id}/resume can pick it up
    # and continue with the user's chosen resolution. In-process, bounded by the
    # number of concurrent pauses (a paused token is popped on resume).
    app.state.pending_runs = {}
    # Append-only ledger of scheduler-fired workflow runs (Scenario B, s9/a4): the
    # live workflow producer records {run_id, topic, fired_at, report_chars, ok}
    # per fire so GET /workflows surfaces the fired run_ids (for Phoenix
    # correlation) without parsing the delivered email. Bounded by the number of
    # fires the safe self-test arms (a small max_fires).
    app.state.workflow_runs = []

    # --- (2) UNIFY: mount s5's specialization gate onto THIS app (a3 reuse) --- #
    w = app.state.wiring
    gate = HttpApprovalGate()
    spec_service = build_specialization_service(w.engine, w.registry, gate=gate)
    register_specialization_routes(app, spec_service)
    app.state.spec_service = spec_service  # lifespan calls gate.cancel_all()

    # --- (3) the DISTINCT interactive spec-AUTHORING chat surface (s4/b1, d11) -- #
    # A surface SEPARATE from the task chat: the user defines a spec over multiple
    # back-and-forth turns and approves to compile+register it. Mounted ALONGSIDE
    # the legacy one-shot /specializations routes above (not a replacement). Built
    # on the SAME registry; live phi transport when live, else the offline seam.
    spec_chat_service = build_spec_chat_service(
        w.registry,
        transport=w.live_transport if w.transport_mode == "live" else None,
    )
    register_spec_chat_routes(app, spec_chat_service)
    app.state.spec_chat_service = spec_chat_service

    # --- (4) the dedicated SHAPES config surface (s4/a4, d5) ------------------ #
    # The backend of the dedicated Shapes screen: list/view the text-file-defined
    # shapes (topology / round_roles / final_roles) and read/WRITE the per-shape
    # max_iter override. Built over the wiring's SHARED store so the value the API
    # writes is the SAME one run_agentic reads to bound the deep-research unroll.
    shape_service = ShapeConfigService(w.shape_config)
    register_shape_routes(app, shape_service)
    app.state.shape_service = shape_service

    # --- (5) the SHAPES "describe-a-shape" authoring surface (s9/b1, d14(2)/d9) -- #
    # The user DESCRIBES a plan shape; the live Gemma model authors the declarative
    # TOML the runtime loads (mirrors the spec-chat authoring mechanism, as the
    # genuine one-shot shapes flow). Built over the SAME ShapeConfigService catalog
    # so an authored shape lands in the list above; live transport only (else 503).
    shape_author_service = ShapeAuthorService(
        shape_service,
        transport=w.live_transport if w.transport_mode == "live" else None,
    )
    register_shape_author_routes(app, shape_author_service)
    app.state.shape_author_service = shape_author_service

    # ---------------------------- chat surface ---------------------------- #
    @app.post("/chats", status_code=201)
    async def new_chat(req: NewChatRequest) -> JSONResponse:
        """Open a NEW empty chat and return its id (durable immediately, O7).

        The frontend opens a chat — and its live SSE stream — BEFORE sending the
        first message, so the chat row must exist with zero turns up front."""
        w = app.state.wiring
        rec = await asyncio.to_thread(w.chat_store.create_chat, req.title)
        return JSONResponse(status_code=201, content=rec.to_dict())

    @app.get("/chats")
    async def list_chats() -> JSONResponse:
        """List prior chats (newest first) for the reopen sidebar — full records
        with their turns + artifact refs, reloaded straight from the durable
        store (survives a restart, O7)."""
        w = app.state.wiring
        chats = await asyncio.to_thread(w.chat_store.list_chats)
        return JSONResponse({"chats": [c.to_dict() for c in chats]})

    @app.get("/chats/{chat_id}")
    async def get_chat(chat_id: str) -> JSONResponse:
        """One chat's FULL history for reopen: ordered turns (user request +
        streamed reasoning events + final response) + artifact refs."""
        w = app.state.wiring
        rec = await asyncio.to_thread(w.chat_store.get_chat, chat_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"no chat {chat_id!r}")
        return JSONResponse(rec.to_dict())

    async def _execute_message_run(
        chat_id: str, req: MessageRequest, *, run_id: Optional[str] = None
    ) -> MessageResponse:
        """Drive the REAL planner+runtime for one message, persist, summarise (s3/b2).

        ``run_id`` is the RunManager job id (the decoupled ``/runs`` path threads
        it in so the Phoenix trace's ``agent.session``/``agent.run`` spans carry the
        SAME id the client polls at ``/runs/{run_id}`` — making a trace correlatable
        back to its run). The inline ``/message`` path leaves it ``None``.

        RC1/RC2 FIX: there is no fixed ``analyze → draft_md → draft_html`` stub DAG
        any more. The user's message drives :func:`~chat_app.agentic.run_agentic` —
        shape selection (s3/b1) → the planner-derived DAG on the LIVE Gemma runtime
        (or the bounded deep-research executor). Offline (stub mode) the SAME
        ``Planner.plan → AgentRuntime`` pipeline runs on the deterministic stub
        transport via :func:`~chat_app.agentic.run_offline` (the d12 seam) — NOT a
        hand-built demo DAG. The run is bound to THIS CHAT's plane so its lifecycle
        events reach only this chat's SSE stream; the turn (request + per-node
        events + final response) and any produced artifact are persisted via b1
        (O7).

        Shared by TWO entrypoints: the synchronous ``POST /chats/{id}/message``
        awaits it inline; the decoupled ``POST /chats/{id}/runs`` hands it to the
        :class:`RunManager` to run in the BACKGROUND. The caller has already
        verified the chat exists."""
        w = app.state.wiring
        plane = hub.plane_for(chat_id)  # per-chat plane (the SSE stream subscribes)

        # CONVERSATION MEMORY (s5/a1+a2): assemble THIS chat's bounded prior-turn
        # context BEFORE the run, so the plan CONTINUES the conversation instead of
        # running memoryless. It is read here (before save_turn below persists the
        # new turn), so it carries the PRIOR turns only — never this message — and is
        # strictly chat_id-scoped (one thread never sees another's). The sqlite read
        # is offloaded off the event loop so it never stalls other chats' SSE
        # streams. The FIRST turn of a chat assembles to "" → a no-op (the goal is
        # unchanged), so there is no regression for a brand-new thread.
        conversation_context = await asyncio.to_thread(
            w.conversation_memory.assemble_context, chat_id
        )

        if w.transport_mode == "live":
            agentic = await run_agentic(
                req.message,
                transport=w.live_transport,
                registry=w.registry,
                hook=w.hook,
                plane=plane,
                run_id=run_id,
                shape_config=w.shape_config,  # UI-set per-shape max_iter (s4/a4, d5)
                conversation_context=conversation_context,
            )
        else:
            agentic = await run_offline(
                req.message,
                registry=w.registry,
                hook=w.hook,
                plane=plane,
                run_id=run_id,
                conversation_context=conversation_context,
            )

        # AMBIGUITY CLARIFICATION PAUSE (scenario-2): the planner judged the request
        # too underspecified to act on and asked the user a clarifying question. The
        # notify (EVENT_NEEDS_CLARIFICATION) already streamed to the chat; PAUSE here
        # — stash the ORIGINAL request + bounded context keyed by the resume_token so
        # POST /chats/{id}/resume can re-drive the plan on the user's answer — and
        # return the question to the client instead of silently guessing the detail.
        # The turn is persisted so the pause is visible on reopen.
        if agentic.needs_clarification:
            pending = agentic.pending or {}
            token = pending.get("resume_token")
            if token:
                app.state.pending_runs[token] = {
                    "chat_id": chat_id,
                    "kind": CLARIFICATION_KIND,
                    # The user's ORIGINAL message (not the context-folded goal) — the
                    # resume re-runs run_agentic on it with the answer folded in, and
                    # run_agentic re-assembles its own conversation context.
                    "original_query": req.message,
                    "question": pending.get("question", ""),
                    "conversation_context": conversation_context,
                }
            paused_turn = await asyncio.to_thread(
                w.chat_store.save_turn,
                chat_id,
                req.message,
                [],
                f"Clarifying question: {pending.get('question', '')}",
            )
            return MessageResponse(
                chat_id=chat_id,
                turn_index=paused_turn.turn_index,
                ok=False,
                launch_order=[],
                node_states=[],
                outputs={},
                artifacts=[],
                missing_specialist=False,
                pending=pending,
            )

        # MISSING-SPECIALIST PAUSE (s4 M1, RC8): the plan needs a specialist no
        # registered spec provides. The notify (EVENT_MISSING_SPECIALIST) already
        # streamed to the chat; PAUSE here — stash the paused DAG + context keyed
        # by the resume_token so POST /chats/{id}/resume can continue it — and
        # return the CHOICE to the client instead of a silent failure. The turn is
        # persisted so the pause is visible on reopen.
        if agentic.missing_specialist:
            pending = agentic.pending or {}
            token = pending.get("resume_token")
            if token:
                app.state.pending_runs[token] = {
                    "chat_id": chat_id,
                    "dag": agentic.dag,
                    "shape": agentic.shape,
                    "rationale": agentic.rationale,
                    "missing": pending.get("missing", []),
                    # a6 fix (s7/a1): stash the SAME bounded prior-turn context the
                    # initial attempt assembled, so a missing-specialist RESUME drives
                    # the paused nodes with conversation memory — not memoryless. Reused
                    # (not re-assembled) so the resume sees exactly the PRIOR turns the
                    # initial run did (the paused turn save below must not leak in).
                    "conversation_context": conversation_context,
                }
            paused_turn = await asyncio.to_thread(
                w.chat_store.save_turn,
                chat_id,
                req.message,
                [],
                "Paused: a needed specialist is not available — choose a resolution.",
            )
            return MessageResponse(
                chat_id=chat_id,
                turn_index=paused_turn.turn_index,
                ok=False,
                launch_order=[],
                node_states=[],
                outputs={},
                artifacts=[],
                missing_specialist=True,
                pending=pending,
            )

        turn_events = [
            {
                "node_id": nid,
                "status": st["status"],
                "attempts": st["attempts"],
                "error": st["error"],
            }
            for nid, st in agentic.states.items()
        ]

        # Persist the turn (durable on return, O7). Blocking sqlite goes off the
        # event loop so the SSE generators serving other chats never stall.
        turn = await asyncio.to_thread(
            w.chat_store.save_turn,
            chat_id,
            req.message,
            turn_events,
            agentic.final_response,
        )

        # Surface the run's produced artifact(s) (the final deliverable): saved via
        # b1, refs returned + reloadable on reopen, downloadable at
        # GET /artifacts/{id}.
        artifacts: list[ArtifactOut] = []
        for filename, mime, body in agentic.artifacts:
            body = (body or "")
            if not body.strip():
                continue
            ref = await asyncio.to_thread(
                w.chat_store.save_artifact,
                chat_id,
                filename,
                body.encode("utf-8"),
                mime,
            )
            artifacts.append(
                ArtifactOut(
                    artifact_id=ref.artifact_id,
                    filename=ref.filename,
                    mime=ref.mime,
                    size=ref.size,
                    node_id=None,
                )
            )

        return MessageResponse(
            chat_id=chat_id,
            turn_index=turn.turn_index,
            ok=agentic.ok,
            launch_order=agentic.launch_order,
            node_states=[
                NodeStateOut(
                    node_id=nid,
                    status=st["status"],
                    attempts=st["attempts"],
                    error=st["error"],
                )
                for nid, st in agentic.states.items()
            ],
            outputs=agentic.outputs,
            artifacts=artifacts,
        )

    @app.post("/chats/{chat_id}/message", response_model=MessageResponse)
    async def post_message(chat_id: str, req: MessageRequest) -> MessageResponse:
        """SYNCHRONOUS submit (back-compat): drive the run and return its full
        summary once complete. Kept for existing callers; new UIs should prefer
        the decoupled ``POST /chats/{id}/runs`` so a slow run never blocks the
        request. The blocking phi is offloaded off the event loop either way, so
        even this inline path no longer freezes OTHER requests."""
        w = app.state.wiring
        rec = await asyncio.to_thread(w.chat_store.get_chat, chat_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"no chat {chat_id!r}")
        return await _execute_message_run(chat_id, req)

    async def _finish_resumed_turn(
        chat_id: str, user_label: str, agentic: Any
    ) -> MessageResponse:
        """Persist a resumed run's turn + artifacts and build its response (shared).

        Used by BOTH resume paths (missing-specialist and ambiguity-clarification)
        so a resolved pause persists and returns exactly like a fresh completed
        run. ``user_label`` is the synthetic user-turn label recorded for the
        resume (e.g. ``[resume:sse_fallback]`` or ``[clarify] <answer>``)."""
        w = app.state.wiring
        turn_events = [
            {"node_id": nid, "status": st["status"], "attempts": st["attempts"],
             "error": st["error"]}
            for nid, st in agentic.states.items()
        ]
        turn = await asyncio.to_thread(
            w.chat_store.save_turn, chat_id, user_label,
            turn_events, agentic.final_response,
        )
        artifacts: list[ArtifactOut] = []
        for filename, mime, body in agentic.artifacts:
            body = (body or "")
            if not body.strip():
                continue
            ref = await asyncio.to_thread(
                w.chat_store.save_artifact, chat_id, filename, body.encode("utf-8"), mime,
            )
            artifacts.append(ArtifactOut(
                artifact_id=ref.artifact_id, filename=ref.filename,
                mime=ref.mime, size=ref.size, node_id=None,
            ))
        return MessageResponse(
            chat_id=chat_id,
            turn_index=turn.turn_index,
            ok=agentic.ok,
            launch_order=agentic.launch_order,
            node_states=[
                NodeStateOut(node_id=nid, status=st["status"],
                             attempts=st["attempts"], error=st["error"])
                for nid, st in agentic.states.items()
            ],
            outputs=agentic.outputs,
            artifacts=artifacts,
        )

    @app.post("/chats/{chat_id}/resume", response_model=MessageResponse)
    async def resume_message(chat_id: str, req: ResumeRequest) -> MessageResponse:
        """Resolve a missing-specialist pause and CONTINUE the paused plan (s4 M1).

        Looks up the paused PlanDAG by ``resume_token`` and drives it under the
        user's chosen resolution (``sse_fallback`` runs the unmet nodes spec-less;
        ``define_and_resume`` stamps the now-registered ``spec_name`` onto them).
        Live mode only re-drives the real Gemma runtime; the run's lifecycle
        streams to this chat's SSE exactly like a fresh run. The turn + any
        artifact are persisted (O7). 404 if the token is unknown/already consumed;
        422 if ``define_and_resume`` is missing its spec or the spec is
        unregistered."""
        w = app.state.wiring
        rec = await asyncio.to_thread(w.chat_store.get_chat, chat_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"no chat {chat_id!r}")
        parked = app.state.pending_runs.get(req.resume_token)
        if parked is None or parked.get("chat_id") != chat_id:
            raise HTTPException(
                status_code=404, detail=f"no paused run {req.resume_token!r} for this chat"
            )

        # AMBIGUITY-CLARIFICATION RESUME (scenario-2): the parked pause is a
        # clarifying question; the user's ``answer`` resolves it. Re-drive a FRESH
        # plan via run_agentic with the answer folded into the goal (skip_ambiguity
        # is set internally so it does not re-ask) — NOT resume_agentic, which is for
        # a paused missing-specialist DAG. Then fall through to the shared
        # persist/return tail below.
        if parked.get("kind") == CLARIFICATION_KIND:
            answer = (req.answer or "").strip()
            if not answer:
                raise HTTPException(
                    status_code=422, detail="clarification resume requires an answer"
                )
            plane = hub.plane_for(chat_id)
            resume_context = parked.get("conversation_context")
            original_query = parked.get("original_query", "")
            if w.transport_mode == "live":
                agentic = await run_agentic(
                    original_query,
                    transport=w.live_transport,
                    registry=w.registry,
                    hook=w.hook,
                    plane=plane,
                    shape_config=w.shape_config,
                    conversation_context=resume_context,
                    clarification=answer,
                )
            else:
                # Offline seam (d12): the offline path never asks a clarifying
                # question (assess_ambiguity runs only on the live transport), so a
                # clarification resume here simply re-runs the plan with the answer
                # folded into the goal string.
                agentic = await run_offline(
                    f"{original_query}\n\nCLARIFICATION (the user answered your "
                    f"question):\n{answer}",
                    registry=w.registry,
                    hook=w.hook,
                    plane=plane,
                    conversation_context=resume_context,
                )
            app.state.pending_runs.pop(req.resume_token, None)
            return await _finish_resumed_turn(chat_id, f"[clarify] {answer}", agentic)

        # MISSING-SPECIALIST RESUME: ``choice`` is required and must be a known
        # resolution (the field is no longer a fixed enum on the model so the
        # clarification path can omit it — validate it here for this path).
        if req.choice not in ("sse_fallback", "define_and_resume"):
            raise HTTPException(
                status_code=422,
                detail="missing-specialist resume requires choice "
                "'sse_fallback' or 'define_and_resume'",
            )

        # Resolve the spec mapping for define_and_resume (the global spec_name
        # applies to every unmet node via the "" key; an explicit map wins).
        defined_specs = dict(req.defined_specs or {})
        if req.choice == "define_and_resume":
            if not defined_specs and req.spec_name:
                defined_specs = {"": req.spec_name}
            if not defined_specs:
                raise HTTPException(
                    status_code=422,
                    detail="define_and_resume requires spec_name or defined_specs",
                )
            for name in defined_specs.values():
                if name not in w.registry:
                    raise HTTPException(
                        status_code=422, detail=f"no registered spec {name!r}"
                    )

        plane = hub.plane_for(chat_id)
        # Reconstruct the paused plan's SHAPE SPEC from its name so the resume
        # re-derives the SAME execution discipline as the original run (a7 review
        # fix): without it ``resume_agentic`` defaulted shape_spec=None ->
        # CONCURRENT, so a paused ``linear`` (strict single-file) plan resumed as a
        # concurrent fan-out. The DAG's depends_on still serialised a chain (output
        # unaffected), but the discipline silently diverged from the initial run;
        # threading the spec back keeps linear-as-sequential on resume too. Unknown
        # shape -> None (the legacy CONCURRENT fallback), exactly as before.
        shape_name = parked.get("shape")
        shape_spec = load_shapes().get(shape_name) if shape_name else None
        # a6 fix (s7/a1): the bounded prior-turn context the initial attempt used,
        # stashed at pause time. Threaded back so the resumed nodes ground in the
        # conversation exactly as the initial run did (mirrors the s5/a4 fix).
        resume_context = parked.get("conversation_context")
        if w.transport_mode == "live":
            agentic = await resume_agentic(
                parked["dag"],
                req.choice,
                transport=w.live_transport,
                registry=w.registry,
                hook=w.hook,
                plane=plane,
                missing=parked.get("missing"),
                defined_specs=defined_specs or None,
                shape_spec=shape_spec,
                shape=parked.get("shape"),
                rationale=parked.get("rationale", ""),
                conversation_context=resume_context,
            )
        else:
            # Offline seam (d12): same resolution surgery on the stub transport.
            agentic = await resume_agentic(
                parked["dag"],
                req.choice,
                transport=stub.subagent_transport(),
                registry=w.registry,
                hook=w.hook,
                plane=plane,
                missing=parked.get("missing"),
                defined_specs=defined_specs or None,
                shape_spec=shape_spec,
                shape=parked.get("shape"),
                rationale=parked.get("rationale", ""),
                conversation_context=resume_context,
            )

        # Consume the token (a pause resolves once).
        app.state.pending_runs.pop(req.resume_token, None)
        return await _finish_resumed_turn(chat_id, f"[resume:{req.choice}]", agentic)

    @app.post("/chats/{chat_id}/runs", status_code=202)
    async def start_run(chat_id: str, req: MessageRequest) -> JSONResponse:
        """DECOUPLED submit (the freeze fix): start the agent run in the
        BACKGROUND and return its run id IMMEDIATELY (HTTP 202 Accepted).

        The handler does NOT await the run — it submits the same
        ``_execute_message_run`` body to the :class:`RunManager` and returns at
        once, so the request never hangs for the (possibly slow, GPU-contended)
        phi work. The client then (a) watches ``GET /chats/{id}/stream`` for the
        live per-node lifecycle, and (b) polls ``GET /runs/{run_id}`` for terminal
        status + the final summary. The server stays fully responsive (incl
        ``/health``) for the entire run."""
        w = app.state.wiring
        rec = await asyncio.to_thread(w.chat_store.get_chat, chat_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"no chat {chat_id!r}")

        # Pre-allocate the run id so it can be threaded into the run BEFORE the
        # body executes — without it the trace's agent.run/session spans fall back
        # to the trace id and cannot be correlated to the /runs/{id} the client
        # polls. submit honours this same id (no double-allocation).
        rid = f"run-{uuid.uuid4().hex[:12]}"

        async def _body() -> dict[str, Any]:
            resp = await _execute_message_run(chat_id, req, run_id=rid)
            return resp.model_dump()

        record = run_manager.submit(chat_id, _body, run_id=rid)
        return JSONResponse(
            status_code=202,
            content={
                "run_id": record.run_id,
                "chat_id": chat_id,
                "status": record.status,
                "stream": f"/chats/{chat_id}/stream",
                "status_url": f"/runs/{record.run_id}",
            },
        )

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str) -> JSONResponse:
        """Poll one background run's status + (once ``done``) its final summary.

        Returns the live ``running`` state or the terminal ``done`` / ``failed`` /
        ``cancelled`` state. The per-node lifecycle that produced it rode the
        chat's SSE stream; this is the coarse run-level status the client polls
        to know when to stop streaming."""
        record = run_manager.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
        return JSONResponse(record.to_dict())

    @app.get("/chats/{chat_id}/stream")
    async def stream_chat(chat_id: str, request: Request) -> EventSourceResponse:
        """Live SSE stream of THIS CHAT's runtime reasoning/progress (a2 bridge).

        Subscribes to the chat's per-chat in-process plane and relays every
        matching lifecycle event as a named SSE ``event:``. A ``connected`` event
        is yielded immediately so a client can wait for it before posting a
        message — the in-process plane has no replay buffer, so subscribe-before-
        publish is required and this handshake guarantees it.

        Teardown is clean + self-scoped (house style + d8): a client disconnect
        cancels the generator (``CancelledError``); the ``finally`` closes the
        per-connection :class:`Subscription`, so no dead queue is left for the
        producer to fill. ``ping`` heartbeats (sse-starlette default ~15s) keep an
        idle stream alive behind a buffering proxy. The per-connection queue is
        bounded (lossless in-process backpressure — see ``STREAM_QUEUE_MAXSIZE``).

        Resume is NOT supported: the plane is live/in-process with no history, so
        events missed during a reconnect gap are not replayed; ``id:`` carries the
        plane sequence purely for client-side ordering/debuggability."""
        plane = hub.plane_for(chat_id)
        sub: Subscription = plane.subscribe(
            kinds=RUNTIME_EVENT_KINDS, maxsize=STREAM_QUEUE_MAXSIZE
        )

        async def event_source() -> AsyncIterator[dict[str, Any]]:
            try:
                yield {
                    "event": "connected",
                    "data": json.dumps({"ok": True, "chat_id": chat_id}),
                }
                async for ev in sub:
                    if await request.is_disconnected():
                        break
                    yield {
                        "event": ev.kind,
                        "id": str(ev.seq),
                        "data": json.dumps(
                            {
                                "kind": ev.kind,
                                "seq": ev.seq,
                                "source": ev.source,
                                "payload": _jsonable(ev.payload),
                            }
                        ),
                    }
            finally:
                # House style: unsubscribe in finally so the producer never fills
                # a dead queue (disconnect arrives as CancelledError).
                sub.close()

        return EventSourceResponse(event_source())

    # ---------------------------- artifacts ---------------------------- #
    @app.get("/artifacts/{artifact_id}")
    async def get_artifact(
        artifact_id: str,
        inline: bool = Query(
            default=False,
            description="render in-browser (Content-Disposition: inline) instead "
            "of forcing a download (attachment)",
        ),
    ) -> FileResponse:
        """Download an artifact with the correct mime + ``Content-Disposition``.

        Resolves the artifact's metadata (filename, mime) and its on-disk bytes
        from the SAME b1 store, then serves the file with its stored content
        type. ``?inline=1`` renders it in the browser (e.g. preview an HTML/MD
        report); the default forces a download with the original filename."""
        w = app.state.wiring
        ref = await asyncio.to_thread(w.chat_store.get_artifact_ref, artifact_id)
        if ref is None:
            raise HTTPException(status_code=404, detail=f"no artifact {artifact_id!r}")
        path = await asyncio.to_thread(w.chat_store.get_artifact_path, artifact_id)
        if path is None or not path.is_file():
            raise HTTPException(
                status_code=404, detail=f"artifact {artifact_id!r} bytes missing"
            )
        return FileResponse(
            path,
            media_type=ref.mime,
            filename=ref.filename,
            content_disposition_type="inline" if inline else "attachment",
        )

    # ----------------------- workflow specs (Scenario B) ----------------------- #
    def _job_view(job: Any) -> dict[str, Any]:
        """A JSON view of a scheduled workflow job INCLUDING its last fire result.

        ``Scheduler.list_jobs`` snapshots deliberately omit the (arbitrary) fire
        result; the workflow surface wants it (delivered / smtp_code / message_id)
        so a fired send is provable from a poll alone — so this enriches the
        snapshot with ``last_result`` read off the live job object."""
        snap = job.snapshot()
        snap["last_result"] = _jsonable(job.last_result)
        return snap

    @app.post("/workflows", status_code=201)
    async def arm_workflow(req: ArmWorkflowRequest) -> JSONResponse:
        """Arm an already-registered spec as a SCHEDULED workflow (explicit, d8).

        This is the one EXPLICIT entrypoint that arms a scheduled send — nothing
        auto-starts a recurring email on boot (d8 safety). The named spec (e.g.
        authored via the interactive spec-chat) is loaded, the request's schedule
        + delivery are LAYERED onto it (turning a chat-defined output-shaping
        ruleset into a "daily brief"-style workflow), and it is scheduled on the
        in-process scheduler. In LIVE mode the fire runs a REAL Gemma-4 agentic
        run (traced to Phoenix); in stub mode it uses the deterministic stub
        producer. Returns the job id + the initial job snapshot."""
        from dataclasses import replace

        from specialization import DeliverySpec, ScheduleSpec

        w = app.state.wiring
        if req.spec_name not in w.registry:
            raise HTTPException(
                status_code=404, detail=f"no registered spec {req.spec_name!r}"
            )
        # Load off the event loop (registry read is file I/O).
        base = await asyncio.to_thread(w.registry.load, req.spec_name)
        spec = replace(
            base,
            schedule=ScheduleSpec(
                kind=req.schedule.kind,
                interval_seconds=req.schedule.interval_seconds,
                max_fires=req.schedule.max_fires,
                initial_delay=req.schedule.initial_delay,
            ),
            delivery=DeliverySpec(
                channel=req.delivery.channel, recipient=req.delivery.recipient
            ),
        )
        # LIVE producer (real Gemma-4 traced run) when live; else the stub. d8
        # safety: this only ARMS — the scheduler fires it on its own interval.
        if w.transport_mode == "live" and w.live_transport is not None:
            producer = make_live_report_producer(
                transport=w.live_transport,
                registry=w.registry,
                hook=w.hook,
                ledger=app.state.workflow_runs,
            )
        else:
            producer = None  # workflow.stub_produce_report (offline default)
        job_id = schedule_workflow_spec(
            w.scheduler, spec, w.hook, produce_report=producer
        )
        job = w.scheduler.get(job_id)
        return JSONResponse(
            status_code=201,
            content={
                "job_id": job_id,
                "spec": spec.name,
                "transport": w.transport_mode,
                "schedule": {
                    "kind": req.schedule.kind,
                    "interval_seconds": req.schedule.interval_seconds,
                    "max_fires": req.schedule.max_fires,
                    "initial_delay": req.schedule.initial_delay,
                },
                "delivery": {"channel": req.delivery.channel, "recipient": req.delivery.recipient},
                "snapshot": _job_view(job) if job is not None else None,
            },
        )

    @app.get("/workflows")
    async def list_workflows() -> JSONResponse:
        """Observe every armed workflow job + the scheduler-fired run ledger.

        Read-only: returns each job's live snapshot (status / fire_count /
        last_fired / last_result incl. smtp_code + message_id) plus the
        append-only ``workflow_runs`` ledger of fired run_ids (for Phoenix
        correlation). This is what proves the scheduler FIRED on its interval and
        the fire emailed the brief — without parsing the delivered mail."""
        w = app.state.wiring
        jobs = []
        for snap in w.scheduler.list_jobs():
            job = w.scheduler.get(snap["job_id"])
            if job is not None:
                jobs.append(_job_view(job))
        return JSONResponse(
            {
                "active": w.scheduler.active_count,
                "jobs": jobs,
                "runs": list(app.state.workflow_runs),
            }
        )


__all__ = [
    "register_routes",
    "ChatStreamHub",
    "RUNTIME_EVENT_KINDS",
    "MessageRequest",
    "MessageResponse",
    "NewChatRequest",
]
