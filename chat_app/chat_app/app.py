"""The single ASGI app that composes the whole ReactiveAgents stack in ONE process.

STAGE A skeleton (s7/a1) + the #1 integration risk retired (s7/a2)
-----------------------------------------------------------------
a1 proved the d2/d11 single-process guarantee: every component (reactive event
plane + tool hook, in-process memory, the specialization registry + engine, and
the agent_runtime planner + runtime) constructs and composes in ONE Python
interpreter.

a2 retires THE #1 integration risk — *the live SSE stream is fed from the REAL
s3 reactive EventPlane while the REAL s6 AgentRuntime drives an in-process DAG in
the same process*. Two endpoints make this concrete:

- ``POST /chat`` accepts a chat request and drives a SMALL real
  :class:`~agent_runtime.AgentRuntime` over a DAG on the STUB transport (d12 — no
  live phi). The runtime publishes its real lifecycle events
  (``agent_node_launched`` / ``agent_node_done`` / …) onto the SHARED plane.
- ``GET /events`` is the live SSE stream: it ``subscribe``s to that SAME
  :class:`~reactive_tools.EventPlane` and streams the runtime's real lifecycle
  events to the client as ``text/event-stream``. It tears down cleanly — the
  per-connection :class:`~reactive_tools.Subscription` is closed in ``finally``
  when the client disconnects (a disconnect surfaces as ``CancelledError``).

Transport (d12): everything still runs on the deterministic STUB/Fake transport.
NO live Ollama / phi4-mini call is made here; the live transport swaps in at s8
with no other code change (the llm_framework ``Transport`` protocol is the seam).
Self-scoped teardown (d8): a single in-process uvicorn instance is stopped only
by its own PID via graceful shutdown — never a name/image-wide kill.
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# --- sibling workspace members, all resolved into ONE interpreter (d2/d11) --- #
from reactive_tools import (
    EventPlane,
    Scheduler,
    Subscription,
    ToolHook,
    build_default_hook,
    register_agentic_tools,
    resolve_cron_db_path,
)
from memory import DurableFactStore, MemoryRecall
from specialization import SpecRegistry, SpecLoader, SpecializationEngine
from chat_app.persistence import ChatStore
from chat_app.conversation_memory import ConversationMemory
from chat_app.shape_config import ShapeConfigStore
from agent_runtime import (
    EVENT_NODE_CANCELLED,
    EVENT_NODE_DONE,
    EVENT_NODE_FAILED,
    EVENT_NODE_HEALED,
    EVENT_NODE_LAUNCHED,
    EVENT_NODE_REPLANNED,
    EVENT_NODE_SKIPPED,
    AbstractPlanFactory,
    AgentRuntime,
    Planner,
    stub,
)
from specialization.seed import seed_canonical_rulesets
from chat_app.agentic import run_agentic, run_offline
from chat_app.cron_scheduler import CronScheduler, make_cron_fire
# Tracing (s6): the a1 factory owns the single shared TracerProvider; b3 builds
# it eagerly at startup and tears it down on shutdown. config.load_tracing_env
# bridges the .env Phoenix endpoint/project keys into os.environ first.
from agent_runtime.tracing import get_tracer_provider, shutdown_tracer_provider
from reactive_tools.config import load_tracing_env

# The full set of runtime lifecycle kinds the SSE stream relays. Tool-layer
# events (``tool_call`` / ``tool_result``) ride the same plane and are included
# so a tool-using DAG is observable too — the demo DAG is tool-less, so in
# practice the node-lifecycle kinds are what stream.
RUNTIME_EVENT_KINDS: tuple[str, ...] = (
    EVENT_NODE_LAUNCHED,
    EVENT_NODE_DONE,
    EVENT_NODE_FAILED,
    EVENT_NODE_HEALED,
    EVENT_NODE_CANCELLED,
    EVENT_NODE_REPLANNED,
    EVENT_NODE_SKIPPED,
    "tool_call",
    "tool_result",
)


# --------------------------------------------------------------------------- #
# request / response models (Pydantic v2 — house style)
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    """A chat turn. ``message`` is the user's prompt; ``topic`` optionally names
    the subject the demo DAG renders two ways (defaults to the message)."""

    message: str = Field(min_length=1, max_length=4000)
    topic: str | None = Field(default=None, max_length=400)
    # Continue an existing chat by id (durable history, O7); omit to start a new
    # one. The id of the chat the turn landed in is returned on ChatResponse.
    chat_id: str | None = Field(default=None, max_length=120)


class NodeStateOut(BaseModel):
    """The terminal state of one DAG node, surfaced in the chat response."""

    node_id: str
    status: str
    attempts: int
    error: str | None = None


class ChatResponse(BaseModel):
    """The outcome of driving the DAG for one chat turn.

    The live per-node lifecycle is on the SSE stream; this is the final summary
    the POST returns once the in-process run completes."""

    run_id: str
    ok: bool
    chat_id: str  # the durable chat this turn was persisted into (O7)
    turn_index: int
    launch_order: list[str]
    node_states: list[NodeStateOut]
    event_kinds_emitted: list[str]
    outputs: dict[str, str]


def _jsonable(obj: Any) -> Any:
    """Best-effort coerce an event payload to a JSON-serialisable form.

    Lifecycle payloads are already plain dicts/strings; this only guards the
    general case (e.g. a tool_result carrying an arbitrary tool ``value``) so a
    non-serialisable object degrades to ``repr`` instead of failing the stream."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        return repr(obj)


def _data_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the on-disk home for the in-process subsystems' state.

    Override via the ``override`` arg or ``REACTIVE_AGENTS_DATA_DIR``; defaults to
    ``<repo>/var/chat_app``. Kept out of the package tree so a recompile/reinstall
    never wipes accumulated state.
    """
    if override is not None:
        root = Path(override)
    elif os.environ.get("REACTIVE_AGENTS_DATA_DIR"):
        root = Path(os.environ["REACTIVE_AGENTS_DATA_DIR"])
    else:
        # chat_app/chat_app/app.py -> parents[2] == the ReactiveAgents repo root.
        root = Path(__file__).resolve().parents[2] / "var" / "chat_app"
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class Wiring:
    """The shared in-process composition every later s7 action reuses.

    Holding the components on one object (attached to ``app.state``) makes the
    single-process composition introspectable without booting the server — that
    is exactly what the a1 import-smoke asserts.
    """

    plane: EventPlane
    hook: ToolHook
    memory: MemoryRecall
    registry: SpecRegistry
    engine: SpecializationEngine
    planner: Planner
    runtime: AgentRuntime
    chat_store: ChatStore  # durable chat history + artifacts (O7, s7/b1)
    conversation_memory: ConversationMemory  # bounded per-chat prior-turn context (s5/a1+a2)
    shape_config: ShapeConfigStore  # per-shape max_iter overrides (s4/a4, d5)
    scheduler: Scheduler  # in-process workflow-spec scheduler (s5/a3)
    cron_scheduler: CronScheduler  # always-on DB-backed cron firing service (s6)
    data_dir: Path
    transport_mode: str  # "stub" (default) or "live" (REACTIVE_AGENTS_LIVE=1, s8)
    # The shared live phi transport when transport_mode=="live" (s8 walkthrough);
    # None in stub mode. post_message builds a fresh live runtime sharing it.
    live_transport: Any = None

    def close(self) -> None:
        """Release the stateful resources this wiring owns.

        Resource-closed-where-opened: both the memory store and the chat store
        opened sqlite handles at construction, so the wiring closes them.
        Idempotent and failure-tolerant so shutdown never masks an earlier error.
        """
        closers = [
            lambda: self.memory.store.db.close(),
            self.conversation_memory.close,
            self.chat_store.close,
            self.shape_config.close,
        ]
        if self.live_transport is not None:
            closers.append(self.live_transport.close)
        for closer in closers:
            try:
                closer()
            except Exception:  # pragma: no cover - shutdown must not raise
                pass


def build_wiring(*, data_dir: str | os.PathLike[str] | None = None) -> Wiring:
    """Compose the WHOLE stack in ONE process on the STUB transport (d12).

    Every construction here is lightweight and offline: the CPU embedder is lazy
    (built only on first recall, never on import), the transports are
    deterministic Fakes, and no Ollama / GPU call is made.
    """
    data = _data_dir(data_dir)

    # 1) reactive core — one event plane, one tool hook (the 4 core tools). The
    #    file root is left to default to DEFAULT_ARTIFACT_DIR
    #    (C:\Projects\ReactiveAgents\artifacts) so written reports land in
    #    artifacts\ (d3 — fixes the round-1 Downloads\report.md bug) rather than
    #    inside the per-deploy data dir. The path-traversal guard still applies;
    #    pass file_base here for a per-task override.
    plane = EventPlane()
    hook = build_default_hook(plane)
    # NODE→TOOL WIRING (s3/b5): compose the SIX s2 node-callable tools onto the
    # SAME hook — web_search, web_fetch, file_read, file_write, the recipient-
    # LOCKED send_mail, and cron_add/list/delete — so a planned node ANSWERS via
    # specializations + tools rather than raw LLM auto-completion. d8 (the
    # unattended-email safety invariant): the ONLY mail tool a node can reach is
    # the recipient-hard-locked send_mail; the legacy free-``to`` send_email is
    # not registered on this hook at all (build_default_hook no longer adds it).
    # The cron tools share the SAME SQLite db (<data_dir>/chat.db) the chat store
    # uses, so a cron_add row is visible to the s6 firing scheduler.
    register_agentic_tools(hook, cron_data_dir=data)

    # 2) memory — in-process sqlite-vec store + the hardened recall facade (d3).
    #    The CPU MiniLM embedder is lazy, so this opens only the sqlite handle.
    store = DurableFactStore(data / "memory.db")
    memory = MemoryRecall(store)

    # LIVE MODE (s8): REACTIVE_AGENTS_LIVE=1 swaps the deterministic stub
    # transports for the real phi4-mini OllamaTransport — the whole point of the
    # pluggable Transport seam (d7). Default stays stub so the offline harness /
    # existing evidence are unaffected. keep_alive is short so the server does not
    # hog the SHARED GPU between turns (d8).
    live = os.environ.get("REACTIVE_AGENTS_LIVE", "").strip().lower() in (
        "1", "true", "yes", "on"
    )
    live_transport = None

    # 3) specialization — on-disk registry (the d10 lookup/load split) + the
    #    lifecycle engine. Offline deterministic condense by default (d7); LIVE
    #    phi condense when live mode is on (so an approved spec is phi-authored).
    registry = SpecRegistry(data / "specs")
    # RC3 (s3/b2): seed the MINIMAL canonical output-shaping rulesets so the LIVE
    # planner-derived path is REACHABLE OUT OF THE BOX — a planned node always has
    # a specialization to inject, and the bounded deep-research shape has its ONE
    # reused spec (``research-analyst``). Idempotent (re-seed overwrites the same
    # canonical names each boot). The full specialization-management surface
    # (authoring, re-edit, missing-spec fallback) is s4's job, not here.
    seed_canonical_rulesets(registry)
    if live:
        from llm_framework import OllamaTransport

        # s8/b1 swap (d17): the runtime model is the optimized Gemma-4 E2B custom
        # tag on the NATIVE Ollama :11434 (transport defaults), NOT phi4-mini on the
        # foreign Docker :11435. ``api="native"`` is required so the structured
        # planner call can pass the top-level ``think=False`` field. keep_alive is
        # WARM ("30m") so the cold ~8-10 s load is paid once and turns reuse the
        # resident model (warm TTFT ~0.38 s); the model's 1.44 GiB resident footprint
        # leaves ample headroom on the shared 6 GB GPU (a3 measured).
        live_transport = OllamaTransport(
            api="native", keep_alive="30m", timeout=300
        )
        engine = SpecializationEngine(
            registry, hook=hook, condense_transport=live_transport,
            specs_dir=data / "specs",
        )
    else:
        engine = SpecializationEngine(registry, hook=hook)

    # 4) agent runtime — the planner reasons over ONLY the body-free factory +
    #    spec lookup (d10); the runtime launches DAG nodes as in-process tasks on
    #    the SHARED plane/hook (d2 — no shell forking). Live phi or stub transport.
    factory = AbstractPlanFactory(
        registry.index(), tool_catalog=hook.registry.catalog()
    )
    if live:
        # s8/b1: the DECISIVE swap fix. gemma4 is a thinking model, so the
        # structured planner call passes ``think=False`` (suppress CoT) + temp 0 —
        # a3 proved 24/24 valid plan-DAGs this way; without it the CoT trace eats
        # num_predict and the JSON content comes back EMPTY (a2 measured 0%). The
        # Planner default ``json=True`` keeps format=json on. No few-shot anchor and
        # do NOT raise temperature (a2: both degrade DAG validity).
        planner = Planner(
            live_transport, factory, call_opts={"think": False, "temperature": 0}
        )
        runtime = AgentRuntime(
            transport=live_transport,
            loader=SpecLoader(registry),
            hook=hook,
            plane=plane,
        )
    else:
        planner = Planner(stub.valid_plan_transport(), factory)
        runtime = AgentRuntime(
            transport=stub.subagent_transport(),
            loader=SpecLoader(registry),
            hook=hook,
            plane=plane,
        )

    # 5) durable persistence (O7, s7/b1) — chat history + artifacts survive a
    #    SERVER RESTART. Rooted at the SAME data dir as everything else (chat.db +
    #    artifacts/ under <data_dir>), so the running app reads exactly what was
    #    written. In-process sqlite + on-disk files (d2/d8) — no standing service.
    chat_store = ChatStore(data)

    # 5a) CONVERSATION MEMORY (s5/a1+a2) — the bounded per-chat prior-turn context
    #    layer, built over the SAME shared chat_store so it reads exactly the turns
    #    the message path writes (strictly chat_id-scoped: one thread never sees
    #    another's). _execute_message_run assembles a chat's context here and threads
    #    it into run_agentic/run_offline so a chat-driven plan CONTINUES the
    #    conversation. Its running-summary table lives in the same chat.db (no new
    #    file); the wiring closes its sqlite handle at teardown.
    conversation_memory = ConversationMemory(chat_store)

    # 5b) per-shape max_iter overrides (s4/a4, d5) — the Shapes screen's backing
    #    store. Shares the SAME chat.db (the d5 "shared SQLite"); the runtime reads
    #    the override through run_agentic (handed this store below) so a UI-set
    #    max_iter, not the text-file default, bounds the deep-research unroll.
    shape_config = ShapeConfigStore(data)

    # 6) in-process workflow scheduler (s5/a3) — fires recurring/one-shot WORKFLOW
    #    specs while the app runs, on the SHARED plane (so the UI/lambda tab can
    #    observe scheduler_job_* events). Constructed with NO jobs: a real
    #    scheduled send is armed ONLY by an explicit schedule_workflow_spec call
    #    (the safe self-test) — nothing auto-starts a recurring email on boot (d8
    #    safety). The lifespan start()s it and shutdown()s its own tasks at teardown.
    scheduler = Scheduler(plane)

    # 6a) ALWAYS-ON DB-BACKED CRON FIRING SCHEDULER (s6) — the firing half of the
    #    cron capability the s2 store/tools left for s6. It reads the DUE cron rows
    #    from the SAME shared chat.db the cron tools write (resolved identically
    #    here), and on each tick fires each due job's prompt as a FRESH plan via
    #    run_agentic (live) / run_offline (stub seam) — never resume_agentic. Fire
    #    state (last_run_at / next_run_at / last_status) is persisted back to the
    #    DB so schedules + catch-up baselines SURVIVE A RESTART (re-read on boot);
    #    missed-fire catch-up is capped at the newest 3 windows (older dropped).
    #    The fire path reaches mail ONLY via the recipient-locked send_mail (d8),
    #    so an unattended cron send can never reach an arbitrary recipient. The
    #    lifespan start()s the tick loop and stop()s its own task at teardown.
    cron_db_path = str(resolve_cron_db_path(data_dir=data))
    cron_fire = make_cron_fire(
        transport_mode="live" if live else "stub",
        registry=registry,
        hook=hook,
        live_transport=live_transport,
        shape_config=shape_config,
    )
    cron_scheduler = CronScheduler(cron_db_path, cron_fire, plane=plane)

    return Wiring(
        plane=plane,
        hook=hook,
        memory=memory,
        registry=registry,
        engine=engine,
        planner=planner,
        runtime=runtime,
        chat_store=chat_store,
        conversation_memory=conversation_memory,
        shape_config=shape_config,
        scheduler=scheduler,
        cron_scheduler=cron_scheduler,
        data_dir=data,
        transport_mode="live" if live else "stub",
        live_transport=live_transport,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own teardown of the wiring's stateful resources (house-style lifespan).

    The wiring is composed eagerly in :func:`create_app` so the single-process
    guarantee is provable by import alone; the lifespan owns its teardown (the
    sqlite handles). Later actions move heavy startup (SSE hub, schema migration)
    in here, before the ``yield``.

    On startup it LOADS all persisted chats from the durable store (O7) so a
    fresh server boot rehydrates prior history — the count is stashed on
    ``app.state.persisted_chats`` for introspection / the UI's initial render.
    """
    w: Wiring = app.state.wiring
    app.state.persisted_chats = w.chat_store.list_chats()

    # Tracing (s6/b3): bridge the .env Phoenix config into os.environ, then build
    # the ONE shared tracer provider EAGERLY (the a1 factory singleton). Building
    # it here — before the first request — registers it as the OpenTelemetry
    # GLOBAL provider so the b1 phi-transport ``llm.chat`` span and the b2
    # planner/runtime DAG spans all resolve to this single provider (never a
    # second one). The provider's BatchSpanProcessor flushes spans on a BACKGROUND
    # thread, so an OTLP export to Phoenix can NEVER stall this asyncio event loop
    # or re-introduce the freeze (d4) — GET /health stays responsive while traces
    # export. Constructing the provider is pure object setup (no network), so
    # startup itself never blocks on Phoenix. /health is not traced: there is no
    # HTTP-server auto-instrumentation, and /health touches neither the phi
    # transport nor the DAG, so it produces no spans by construction.
    load_tracing_env()
    app.state.tracer_provider = get_tracer_provider()

    # Mark the in-process workflow scheduler live (s5/a3). It carries NO jobs at
    # boot — a scheduled send is armed only by an explicit schedule_workflow_spec
    # call (the safe self-test), never auto-started here (d8 safety).
    w.scheduler.start()

    # Start the always-on DB-backed cron FIRING scheduler (s6). It launches a tick
    # loop that, while the app runs, reads the due cron rows from the shared
    # chat.db and fires each as a fresh plan via run_agentic; its first tick runs
    # immediately so any windows MISSED while the app was down are caught up at
    # boot (capped at the newest 3). Schedules + fire state were persisted by the
    # cron tools / prior runs, so a fresh boot re-reads and resumes them. The
    # finally below stop()s only its own task (self-scoped teardown, d8).
    w.cron_scheduler.start()

    # s8/b1 startup warm-up: in LIVE mode pre-load the Gemma-4 model into VRAM so
    # the FIRST user turn does not pay the cold ~8-10 s load (warm TTFT ~0.38 s).
    # Fired as a DETACHED background task on a worker thread (asyncio.to_thread) so
    # startup never blocks on the load and /health is responsive immediately — the
    # same decouple discipline that fixed the freeze (d4). Best-effort: an
    # unreachable Ollama at boot must not crash the app (per-turn calls still
    # surface transport errors). think=False + 1 token keeps the probe trivial;
    # keep_alive="30m" leaves the model resident for the real turns.
    if w.transport_mode == "live" and w.live_transport is not None:
        async def _warm_model() -> None:
            try:
                await asyncio.to_thread(
                    w.live_transport.complete,
                    [{"role": "user", "content": "ok"}],
                    think=False, temperature=0, num_predict=1, keep_alive="30m",
                )
            except Exception:  # pragma: no cover - warm-up is best-effort
                pass
        app.state.warmup_task = asyncio.create_task(_warm_model())

    try:
        yield
    finally:
        # Cancel the s8/b1 warm-up probe if it is still in flight at shutdown so it
        # never outlives the server (self-scoped, failure-tolerant).
        wt = getattr(app.state, "warmup_task", None)
        if wt is not None and not wt.done():
            wt.cancel()
        # Self-scoped teardown (d8): cancel ONLY the scheduler's own tasks and
        # await their unwind, BEFORE the sqlite wiring close. Failure-tolerant so
        # shutdown never masks an earlier error; never a name/image-wide kill.
        try:
            await w.scheduler.shutdown()
        except Exception:  # pragma: no cover - shutdown must not raise
            pass
        # Stop the always-on cron firing scheduler's own tick task (s6) —
        # self-scoped + failure-tolerant, same discipline as above.
        try:
            await w.cron_scheduler.stop()
        except Exception:  # pragma: no cover - shutdown must not raise
            pass
        # Unwind any specialization compile parked at the HITL gate (b3 unifies
        # s5's gate onto this app): cancel_all raises CancelledError into each
        # awaiting engine.compile so no task is left blocked forever. Done before
        # the wiring close so the engine teardown runs first. Failure-tolerant so
        # shutdown never masks an earlier error.
        svc = getattr(app.state, "spec_service", None)
        if svc is not None:
            try:
                await svc.gate.cancel_all()
            except Exception:  # pragma: no cover - shutdown must not raise
                pass
        # Cancel any in-flight background agent run (s1/a1 decouple) so a slow
        # run does not outlive the server. Self-scoped + failure-tolerant.
        mgr = getattr(app.state, "run_manager", None)
        if mgr is not None:
            try:
                await mgr.shutdown()
            except Exception:  # pragma: no cover - shutdown must not raise
                pass
        # Tracing teardown (s6/b3): force-flush the BatchSpanProcessor so no
        # buffered span is lost, then shut the provider down. Done AFTER the run
        # manager is cancelled so any final spans from in-flight runs are captured.
        # Run OFF the event loop (force_flush makes a network call to Phoenix) and
        # failure-tolerant so an unreachable Phoenix can never hang the loop or
        # mask an earlier shutdown error.
        try:
            await asyncio.to_thread(shutdown_tracer_provider)
        except Exception:  # pragma: no cover - shutdown must not raise
            pass
        app.state.wiring.close()


def create_app(*, data_dir: str | os.PathLike[str] | None = None) -> FastAPI:
    """Build a fresh app with the whole stack wired onto ``app.state.wiring``."""
    app = FastAPI(
        title="ReactiveAgents chat_app",
        version="0.1.0",
        summary="Single-process reactive multi-agent web app (stub transport, s7/a1 skeleton).",
        lifespan=lifespan,
    )
    app.state.wiring = build_wiring(data_dir=data_dir)
    app.state.run_seq = 0  # monotonic run id source for POST /chat turns

    # b3 — mount the FULL chat surface (chats / message / stream / artifacts) +
    # the UNIFIED s5 specialization define/approve routes onto THIS one app.
    # Imported here (not at module top) so app.py and routes.py have no import
    # cycle. register_routes stashes app.state.stream_hub + app.state.spec_service.
    from chat_app.routes import register_routes

    register_routes(app)

    # b2 — serve the single-page chat frontend from THIS SAME app (d2): the SPA
    # at GET / and its assets under /static. ``StaticFiles`` carries no lifespan
    # of its own, so mounting it does not break the app's own lifespan (the
    # mount-lifespan caveat only bites sub-apps that need startup/shutdown).
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """The single-page chat UI (s7/b2), served by the one app at GET /."""
        return FileResponse(static_dir / "index.html", media_type="text/html")

    @app.get("/health")
    async def health() -> JSONResponse:
        """Liveness + a manifest of the composed in-process stack."""
        w: Wiring = app.state.wiring
        return JSONResponse(
            {
                "status": "ok",
                "transport": w.transport_mode,
                "components": {
                    "event_plane": type(w.plane).__name__,
                    "tool_hook": type(w.hook).__name__,
                    "tools": w.hook.registry.names(),
                    "memory": type(w.memory).__name__,
                    "spec_registry": type(w.registry).__name__,
                    "spec_engine": type(w.engine).__name__,
                    "planner": type(w.planner).__name__,
                    "agent_runtime": type(w.runtime).__name__,
                    "chat_store": type(w.chat_store).__name__,
                    "scheduler": type(w.scheduler).__name__,
                    "scheduler_jobs": w.scheduler.active_count,
                    "cron_scheduler": type(w.cron_scheduler).__name__,
                    "cron_scheduler_started": w.cron_scheduler.started,
                },
            }
        )

    @app.get("/debug/plane")
    async def debug_plane() -> JSONResponse:
        """Introspect the shared event plane — used by the e2e capture to PROVE
        clean SSE teardown: ``subscriber_count`` is the baseline before a stream
        connects, rises while it is live, and returns to baseline once the client
        disconnects and the generator's ``finally`` closes the subscription."""
        w: Wiring = app.state.wiring
        return JSONResponse(
            {"subscriber_count": w.plane.subscriber_count, "seq": w.plane._seq}
        )

    @app.get("/events")
    async def events(request: Request) -> EventSourceResponse:
        """Live SSE stream of the runtime's REAL lifecycle events (s7/a2).

        Subscribes to the SHARED in-process :class:`EventPlane` and relays every
        matching event to the client as a named SSE ``event:``. The subscription
        is created at the TOP of the generator and a ``connected`` event is
        yielded immediately, so a client can wait for it before triggering a run
        (the in-process plane has no replay buffer — subscribe-before-publish is
        required, and that handshake guarantees it).

        Teardown is clean and self-scoped (house style + d8): a client disconnect
        cancels the generator (``CancelledError``); the ``finally`` closes the
        per-connection :class:`Subscription`, so no dead queue is left for the
        producer to fill. ``ping`` heartbeats (sse-starlette default 15s) keep an
        idle stream alive behind a buffering proxy.

        Resume is NOT supported: the plane is live/in-process with no history, so
        events missed during a reconnect gap are not replayed. ``id:`` is set to
        the plane sequence purely for client-side ordering/debuggability.
        """
        w: Wiring = app.state.wiring
        sub: Subscription = w.plane.subscribe(kinds=RUNTIME_EVENT_KINDS)

        async def event_source() -> AsyncIterator[dict[str, Any]]:
            try:
                yield {"event": "connected", "data": json.dumps({"ok": True})}
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

    # ------------------------------------------------------------------ #
    # READ-ONLY reactive-lambda surface (s1/b2, d15) — the UI lambda-tab's
    # observe-only view of the subscriptions the AGENTS created + are using.
    # The user NEVER authors here: both routes are GET/read-only. The snapshot
    # is the registry query; the stream is the meta-plane live channel.
    # ------------------------------------------------------------------ #
    @app.get("/lambda/subscriptions")
    async def lambda_subscriptions(include_closed: bool = True) -> JSONResponse:
        """Read-only snapshot of every agent-created reactive lambda.

        Returns ``{"active", "total", "subscriptions": [...]}`` — each entry the
        observe-only projection (sub_id, what it observes, owner, status, live fire
        counters) the UI lambda-tab renders. There is deliberately no POST/PUT/
        DELETE counterpart: the user observes, the agent authors (d15)."""
        w: Wiring = app.state.wiring
        registry = getattr(w.hook, "subscriptions", None)
        if registry is None:
            return JSONResponse({"active": 0, "total": 0, "subscriptions": []})
        subs = registry.snapshot(include_closed=include_closed)
        return JSONResponse(
            {"active": registry.active_count, "total": len(registry.snapshot()),
             "subscriptions": subs}
        )

    @app.get("/lambda/stream")
    async def lambda_stream(request: Request) -> EventSourceResponse:
        """Live SSE of the reactive-lambda META plane — the read-only channel the
        UI lambda-tab subscribes to (``lambda_registered`` / ``lambda_fired`` /
        ``lambda_closed`` / ``lambda_observation``).

        Isolated from ``/events`` (the working plane) so the observe-only lambda
        view never mixes with run lifecycle. Same clean, self-scoped teardown as
        ``/events``: the per-connection subscription is closed in the ``finally``
        on client disconnect — no orphaned queue left on the meta plane."""
        w: Wiring = app.state.wiring
        registry = getattr(w.hook, "subscriptions", None)
        meta_plane = registry.meta_plane if registry is not None else w.plane
        sub: Subscription = meta_plane.subscribe()

        async def event_source() -> AsyncIterator[dict[str, Any]]:
            try:
                yield {"event": "connected", "data": json.dumps({"ok": True})}
                async for ev in sub:
                    if await request.is_disconnected():
                        break
                    yield {
                        "event": ev.kind,
                        "id": str(ev.seq),
                        "data": json.dumps(
                            {"kind": ev.kind, "seq": ev.seq, "source": ev.source,
                             "payload": _jsonable(ev.payload)}
                        ),
                    }
            finally:
                sub.close()

        return EventSourceResponse(event_source())

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        """Drive the REAL planner+runtime for one turn on the shared plane (s3/b2).

        RC1/RC2 FIX: the legacy a2 firehose endpoint no longer drives a fixed
        ``analyze → draft_md → draft_html`` stub DAG. The user's message now flows
        through :func:`~chat_app.agentic.run_agentic` — shape selection (s3/b1) →
        the planner-derived DAG on the LIVE Gemma runtime (or the bounded
        deep-research executor). Offline (stub mode) the SAME ``Planner.plan →
        AgentRuntime`` pipeline runs on the deterministic stub transport via
        :func:`~chat_app.agentic.run_offline` (the d12 seam) — NOT a hand-built
        demo DAG. The run is bound to the SHARED plane so its lifecycle reaches
        every ``/events`` subscriber; the turn is persisted durably (O7).

        ``chat_id=None`` starts a new chat; a supplied id continues one — the
        resolved id is returned so the client can keep the thread."""
        w: Wiring = app.state.wiring
        app.state.run_seq += 1
        run_id = f"run-{app.state.run_seq}"

        # CONVERSATION MEMORY (s5/a1+a2): this legacy firehose endpoint can CONTINUE
        # a thread (a supplied chat_id), so it too assembles the bounded prior-turn
        # context for that chat and threads it into the run — otherwise a continued
        # thread on this path would run memoryless. A new chat (chat_id=None) or a
        # chat with no prior turns assembles to "" → a no-op (goal unchanged).
        conversation_context = (
            await asyncio.to_thread(
                w.conversation_memory.assemble_context, req.chat_id
            )
            if req.chat_id
            else None
        )

        if w.transport_mode == "live":
            agentic = await run_agentic(
                req.message,
                transport=w.live_transport,
                registry=w.registry,
                hook=w.hook,
                plane=w.plane,  # the SHARED plane the SSE stream subscribes to
                run_id=run_id,
                shape_config=w.shape_config,  # UI-set per-shape max_iter (s4/a4, d5)
                conversation_context=conversation_context,
            )
        else:
            agentic = await run_offline(
                req.message,
                registry=w.registry,
                hook=w.hook,
                plane=w.plane,
                run_id=run_id,
                conversation_context=conversation_context,
            )

        kinds_emitted = sorted(
            {EVENT_NODE_LAUNCHED}
            | {
                EVENT_NODE_DONE if st["status"] == "done" else EVENT_NODE_FAILED
                for st in agentic.states.values()
            }
        )
        outputs = agentic.outputs

        # Persist this turn durably (O7, s7/b1) off the event loop: ``save_turn``
        # is a blocking sqlite call and this is an ``async def`` handler — running
        # it inline would freeze the one event loop and stall every live SSE stream.
        turn_events = [
            {
                "node_id": nid,
                "status": st["status"],
                "attempts": st["attempts"],
                "error": st["error"],
            }
            for nid, st in agentic.states.items()
        ]
        turn = await asyncio.to_thread(
            w.chat_store.save_turn,
            req.chat_id,
            req.message,
            events=turn_events,
            final_response=agentic.final_response,
            title=req.topic or None,
        )

        return ChatResponse(
            run_id=run_id,
            ok=agentic.ok,
            chat_id=turn.chat_id,  # the resolved id (minted if none was supplied)
            turn_index=turn.turn_index,
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
            event_kinds_emitted=kinds_emitted,
            outputs=outputs,
        )

    return app


# The module-level app the ASGI server (and the a1 import-smoke) loads.
app = create_app()
