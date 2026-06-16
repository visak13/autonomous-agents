"""The specialization define -> research -> approve HTTP surface (s7/a3).

This is the route layer that drives the d9 HITL compile gate over HTTP, built on
the genuine "wait for the user" awaitable in :mod:`chat_app.approval`. It is kept
**app-agnostic** — :func:`register_specialization_routes` mounts the routes onto
ANY :class:`fastapi.FastAPI` instance — for two reasons:

1. a3 must prove the gate end-to-end against a running app WITHOUT editing the
   ``app.py`` that a sibling action (a2's SSE work) is concurrently building.
2. The plan's b3 unifies these exact routes onto the ONE chat app; a mount helper
   is precisely what b3 calls, so the gate ships ready to unify (no rewrite).

The flow (all in-process, d2; stub/offline transport, d12):

- ``POST /specializations`` defines a specialist and launches an in-process
  ``engine.ui_specialize`` (or ``autonomous_specialize``) task. That task
  researches + authors a draft, then **blocks** inside the engine's compile gate
  awaiting the HTTP approver — nothing is compiled while it waits.
- ``GET  /specializations/pending`` lists the parked draft(s) with their
  ``to_html()`` preview and the ``challenge`` that keys the decision.
- ``POST /specializations/{challenge}/approve`` (or ``/deny``) delivers the real
  user decision over the wire, un-blocking the awaiting task: approve → compile +
  register; deny → :class:`~specialization.engine.ApprovalDenied`, nothing
  registered.
- ``GET  /specializations`` / ``GET /specializations/jobs/{job_id}`` expose the
  planner-facing registry index (body-free, d10) and per-job status so a client
  can observe the researching → awaiting_approval → registered transition.

Teardown note for the unifying app (b3): call ``gate.cancel_all()`` in the
lifespan ``finally`` so a draft still awaiting approval at shutdown unwinds its
suspended compile task cleanly (raises CancelledError into the awaiting compile).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from specialization import RawDefinition, SpecRegistry, SpecializationEngine
from specialization.engine import ApprovalDenied, SOURCE_AUTONOMOUS, SOURCE_UI

from chat_app.approval import ApprovalGateError, HttpApprovalGate


# --------------------------------------------------------------------------- #
# request model (Pydantic v2 — house style; 422 is automatic on bad input)
# --------------------------------------------------------------------------- #
class DefineRequest(BaseModel):
    """Define a specialist to research + (on approval) compile."""

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=400)
    intent: str = Field(default="", max_length=2000)
    source: str = Field(default=SOURCE_UI)


@dataclass
class _Job:
    """One launched specialization run, tracked so its task outlives the request
    and its status (researching → awaiting_approval → registered/denied/failed)
    can be observed."""

    job_id: str
    name: str
    source: str
    task: "asyncio.Task"


class SpecializationService:
    """Launches + tracks specialization jobs around the HTTP approval gate.

    Holds the engine (research + the compile gate), the registry (the
    planner-facing index readback), and the gate (the awaitable approver). One
    instance backs the route group; for the unified app (b3) it is built from the
    wiring's engine/registry + a shared gate.
    """

    def __init__(
        self,
        engine: SpecializationEngine,
        registry: SpecRegistry,
        gate: HttpApprovalGate,
    ) -> None:
        self._engine = engine
        self._registry = registry
        self._gate = gate
        self._jobs: dict[str, _Job] = {}
        self._counter = 0

    @property
    def gate(self) -> HttpApprovalGate:
        return self._gate

    # -- launch ------------------------------------------------------------- #
    def start(self, raw: RawDefinition, *, source: str) -> str:
        """Launch ``ui_specialize``/``autonomous_specialize`` as an in-process
        task and return its ``job_id``. Must be called on the running loop (it is
        — from an async route). The task is held in ``_jobs`` so it is not GC'd
        and survives the request that started it."""
        if source not in (SOURCE_UI, SOURCE_AUTONOMOUS):
            raise ValueError(f"source must be {SOURCE_UI!r} or {SOURCE_AUTONOMOUS!r}")
        self._counter += 1
        job_id = f"job-{self._counter}"
        entry = (
            self._engine.autonomous_specialize
            if source == SOURCE_AUTONOMOUS
            else self._engine.ui_specialize
        )
        # The approver is the gate's awaitable: the task will BLOCK inside the
        # engine's compile() until a real HTTP approve/deny resolves it (d9).
        task = asyncio.create_task(entry(raw, approver=self._gate.approver))
        self._jobs[job_id] = _Job(job_id=job_id, name=raw.name, source=source, task=task)
        return job_id

    def get_job(self, job_id: str) -> Optional[_Job]:
        return self._jobs.get(job_id)

    # -- status ------------------------------------------------------------- #
    async def job_view(self, job: _Job) -> dict:
        """Derive a job's current status without storing mutable flags.

        ``registered`` / ``denied`` / ``failed`` come from the finished task;
        while the task is still running its draft is either still being
        researched (``researching``) or parked at the gate awaiting a decision
        (``awaiting_approval``) — the latter detected by the gate's pending map,
        which is the genuine "the compile is blocked, waiting for the user"
        signal."""
        t = job.task
        base = {"job_id": job.job_id, "name": job.name, "source": job.source}
        if t.done():
            exc = t.exception()
            if exc is None:
                spec = t.result()
                return {**base, "status": "registered", "registered": spec.name,
                        "challenge": getattr(spec, "challenge", None)}
            if isinstance(exc, ApprovalDenied):
                return {**base, "status": "denied", "error": str(exc)}
            return {**base, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"}
        match = next(
            (p for p in await self._gate.pending() if p["name"] == job.name), None
        )
        return {
            **base,
            "status": "awaiting_approval" if match else "researching",
            "challenge": match["challenge"] if match else None,
        }

    async def all_jobs(self) -> list[dict]:
        return [await self.job_view(j) for j in self._jobs.values()]

    def index(self) -> list[dict]:
        """The planner-facing registry index — body-free rows only (d10)."""
        return [e.as_dict() for e in self._registry.index()]


def register_specialization_routes(
    app: FastAPI, service: SpecializationService
) -> SpecializationService:
    """Mount the define/pending/approve/deny + status routes onto ``app``.

    Returns the service so a caller (test harness, or b3's unified app) can keep a
    handle. The gate is reached via ``service.gate``.
    """
    gate = service.gate

    @app.post("/specializations")
    async def define_specialization(req: DefineRequest) -> dict:
        """DEFINE → launch research+author; the run then BLOCKS at the approval
        gate. Returns the ``job_id`` to poll and approve. Compiles NOTHING yet."""
        raw = RawDefinition(
            name=req.name.strip(),
            description=req.description.strip(),
            intent=req.intent.strip(),
        )
        try:
            job_id = service.start(raw, source=req.source)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"job_id": job_id, "name": raw.name, "source": req.source,
                "status": "researching"}

    @app.get("/specializations/pending")
    async def list_pending() -> dict:
        """The draft(s) awaiting approval — each with its ``to_html()`` preview
        and the ``challenge`` an approve/deny request names."""
        return {"pending": await gate.pending()}

    @app.get("/specializations")
    async def list_specializations() -> dict:
        """Planner-facing registry index (body-free, d10) + live job statuses."""
        return {"index": service.index(), "jobs": await service.all_jobs()}

    @app.get("/specializations/jobs/{job_id}")
    async def job_status(job_id: str) -> dict:
        job = service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        return await service.job_view(job)

    @app.post("/specializations/{challenge}/approve")
    async def approve(challenge: str) -> dict:
        """The HITL GATE over HTTP: a real POST resolves the awaiting approver
        with a granting token bound to THIS draft → compile + register."""
        try:
            return {"ok": True, **await gate.decide(challenge, approved=True)}
        except ApprovalGateError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/specializations/{challenge}/deny")
    async def deny(challenge: str) -> dict:
        """Decline the surfaced draft over HTTP → ApprovalDenied in the awaiting
        task; nothing is compiled or registered."""
        try:
            return {"ok": True, **await gate.decide(challenge, approved=False)}
        except ApprovalGateError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    return service


def build_specialization_service(
    engine: SpecializationEngine,
    registry: SpecRegistry,
    *,
    gate: Optional[HttpApprovalGate] = None,
) -> SpecializationService:
    """Convenience constructor: a service over an engine+registry with a fresh
    (or supplied) gate. b3 calls this from the wiring; the a3 harness calls it
    with an offline-hook engine."""
    return SpecializationService(engine, registry, gate or HttpApprovalGate())


__all__ = [
    "DefineRequest",
    "SpecializationService",
    "register_specialization_routes",
    "build_specialization_service",
]
