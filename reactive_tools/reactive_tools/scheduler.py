"""In-process job scheduler — fires jobs WHILE the app runs, OFF the loop.

This is the SCHEDULER half of the Scenario-B "workflow spec" capability (the
EMAIL half is :mod:`reactive_tools.email_tool`). A workflow spec (e.g. a "daily
brief") encodes a *schedule*; this scheduler is what fires it while the single
in-process app is alive, then hands the fire off to a callback (the chat_app
glue produces the report and calls ``send_email``).

Design constraints honored
--------------------------
- **d2 (in-process)** — pure ``asyncio`` + in-memory state. NO broker/pool HTTP,
  no subprocess, no shell, no second service. A job is an ``asyncio.Task`` the
  scheduler owns; intervals are ``asyncio.sleep`` loops; any *blocking* fire body
  is offloaded with :func:`asyncio.to_thread` so it NEVER stalls the one event
  loop (mirrors the s1 decouple decision d4 and the tool hook's own off-loop
  rule). The loop stays responsive (``/health`` answers) even while a job's
  blocking work runs.
- **Event-plane observability** — every lifecycle moment publishes onto the
  SHARED :class:`~reactive_tools.event_plane.EventPlane`
  (``scheduler_job_scheduled`` / ``_fired`` / ``_completed`` / ``_error`` /
  ``_cancelled``), so the UI/lambda tab can observe scheduled work live on the
  SAME plane that already relays run + tool lifecycle.
- **d8 (self-scoped teardown)** — :meth:`stop` / :meth:`shutdown` cancels ONLY
  the tasks THIS scheduler created and awaits their unwind. Never a name/image-
  wide kill, never another scheduler's tasks. Idempotent + failure-tolerant so a
  lifespan ``finally`` can call it without masking an earlier error.
- **dependency-light (d10)** — this module is GENERIC: it knows about timing,
  the event plane, and an arbitrary ``fire`` callback. It does NOT import the
  spec model, the tool hook, or the agent runtime — the workflow fire-path glue
  (produce a report -> ``send_email``) lives in the chat_app layer that already
  composes those pieces. That keeps reactive_tools free of upward dependencies.

A job's ``fire`` callback may be a coroutine function (awaited directly) or a
plain blocking function (run via :func:`asyncio.to_thread`). Either way the loop
is never blocked.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Union

from .event_plane import EventPlane

# A fire body: a zero-arg callable returning either a value or an awaitable. A
# coroutine function is awaited directly; a plain function is run off-loop via
# asyncio.to_thread so blocking work never stalls the single event loop (d2).
FireFunc = Callable[[], Union[Awaitable[Any], Any]]

# Job kinds.
KIND_INTERVAL = "interval"   # fire every ``interval_seconds`` while alive
KIND_ONE_SHOT = "one_shot"   # fire exactly once (after an optional initial delay)
JOB_KINDS = (KIND_INTERVAL, KIND_ONE_SHOT)

# Per-job lifecycle status (observable via :meth:`Scheduler.list_jobs`).
STATUS_PENDING = "pending"     # scheduled, not yet fired
STATUS_RUNNING = "running"     # a fire body is executing right now
STATUS_WAITING = "waiting"     # interval job between fires
STATUS_DONE = "done"           # one-shot fired / interval hit max_fires
STATUS_CANCELLED = "cancelled"
STATUS_ERROR = "error"         # last fire raised (interval jobs keep going)

# Event kinds published on the shared plane.
EVENT_JOB_SCHEDULED = "scheduler_job_scheduled"
EVENT_JOB_FIRED = "scheduler_job_fired"
EVENT_JOB_COMPLETED = "scheduler_job_completed"
EVENT_JOB_ERROR = "scheduler_job_error"
EVENT_JOB_CANCELLED = "scheduler_job_cancelled"

SCHEDULER_EVENT_KINDS: tuple[str, ...] = (
    EVENT_JOB_SCHEDULED,
    EVENT_JOB_FIRED,
    EVENT_JOB_COMPLETED,
    EVENT_JOB_ERROR,
    EVENT_JOB_CANCELLED,
)


@dataclass
class ScheduledJob:
    """A unit of scheduled work and its live lifecycle state.

    ``fire`` is the body the scheduler invokes on each tick (coroutine OR plain
    blocking callable — both are run off the event loop). For an ``interval``
    job it fires every ``interval_seconds`` until cancelled or ``max_fires`` is
    reached; for a ``one_shot`` it fires exactly once after ``initial_delay``.
    """

    fire: FireFunc
    kind: str = KIND_INTERVAL
    interval_seconds: float = 60.0
    # Optional bound on interval fires (None => unbounded while alive). The safe
    # self-test uses a small N so the test never loops forever.
    max_fires: Optional[int] = None
    # Delay before the FIRST fire (interval: also delays fire #1; one_shot: the
    # whole delay). None => 0 for one_shot, ``interval_seconds`` for interval.
    initial_delay: Optional[float] = None
    name: str = ""

    # ---- assigned/maintained by the scheduler (not set by the caller) ---- #
    job_id: str = ""
    status: str = STATUS_PENDING
    fire_count: int = 0
    last_fired: Optional[float] = None
    last_result: Any = None
    last_error: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.kind not in JOB_KINDS:
            raise ValueError(f"job kind {self.kind!r} not in {JOB_KINDS}")
        if self.kind == KIND_INTERVAL and self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0 for an interval job")
        if self.max_fires is not None and self.max_fires <= 0:
            raise ValueError("max_fires must be > 0 when set")

    def _first_delay(self) -> float:
        if self.initial_delay is not None:
            return max(0.0, float(self.initial_delay))
        return 0.0 if self.kind == KIND_ONE_SHOT else float(self.interval_seconds)

    def snapshot(self) -> dict[str, Any]:
        """A JSON-shaped read of this job's observable state."""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "kind": self.kind,
            "interval_seconds": self.interval_seconds,
            "max_fires": self.max_fires,
            "status": self.status,
            "fire_count": self.fire_count,
            "last_fired": self.last_fired,
            "last_error": self.last_error,
            "created_at": self.created_at,
        }


class Scheduler:
    """An in-process scheduler of recurring + one-shot jobs (one per app).

    Each scheduled job runs as a tracked :class:`asyncio.Task` the scheduler
    owns. Construct it with the SHARED event plane so its fire lifecycle is
    observable alongside run + tool events. Call :meth:`start` once (lifespan
    startup); :meth:`schedule` jobs while alive; :meth:`stop` / :meth:`shutdown`
    at teardown to cancel ONLY this scheduler's own tasks (d8).
    """

    def __init__(self, plane: EventPlane, *, source: str = "scheduler") -> None:
        self._plane = plane
        self._source = source
        self._jobs: dict[str, ScheduledJob] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._started = False

    # -- lifecycle ------------------------------------------------------- #
    def start(self) -> None:
        """Mark the scheduler live. Idempotent.

        Jobs added via :meth:`schedule` create their task immediately whether or
        not :meth:`start` was called; this flag exists so the lifespan can flip a
        clear started/stopped state and so :meth:`schedule` can refuse to add new
        work after :meth:`shutdown`."""
        self._started = True

    @property
    def started(self) -> bool:
        return self._started

    @property
    def active_count(self) -> int:
        """Number of jobs with a still-running task."""
        return len(self._tasks)

    # -- scheduling ------------------------------------------------------ #
    def schedule(self, job: ScheduledJob) -> str:
        """Schedule ``job`` and return its id. Starts its task at once.

        Must be called from within the running event loop (FastAPI handlers and
        the lifespan both run on it). Publishes ``scheduler_job_scheduled``."""
        if not self._started:
            # Be forgiving: scheduling implies we are live. (A host that calls
            # shutdown() first should not then schedule; see below.)
            self._started = True
        if not job.job_id:
            job.job_id = f"job-{uuid.uuid4().hex[:12]}"
        job.status = STATUS_PENDING
        self._jobs[job.job_id] = job
        task = asyncio.create_task(self._run_job(job), name=f"sched:{job.job_id}")
        self._tasks[job.job_id] = task
        self._plane.publish_nowait(
            EVENT_JOB_SCHEDULED, job.snapshot(), source=self._source
        )
        return job.job_id

    async def _run_job(self, job: ScheduledJob) -> None:
        """Drive one job's lifecycle. Never raises out (terminal status is the
        signal). Cancellation is the clean-stop path and is re-raised so the
        owning shutdown can await it."""
        try:
            await asyncio.sleep(job._first_delay())
            if job.kind == KIND_ONE_SHOT:
                await self._fire_once(job)
                job.status = STATUS_DONE
                return
            # interval
            while True:
                await self._fire_once(job)
                if job.max_fires is not None and job.fire_count >= job.max_fires:
                    job.status = STATUS_DONE
                    return
                job.status = STATUS_WAITING
                await asyncio.sleep(job.interval_seconds)
        except asyncio.CancelledError:
            # Clean self-scoped stop (cancel/stop/shutdown). Mark + announce, then
            # re-raise so the awaiting shutdown unwinds correctly.
            if job.status != STATUS_CANCELLED:
                job.status = STATUS_CANCELLED
                self._plane.publish_nowait(
                    EVENT_JOB_CANCELLED, job.snapshot(), source=self._source
                )
            raise
        finally:
            self._tasks.pop(job.job_id, None)

    async def _fire_once(self, job: ScheduledJob) -> None:
        """Invoke the fire body once, off the event loop, recording the outcome.

        A coroutine body is awaited directly; a blocking body is run via
        :func:`asyncio.to_thread` so it never stalls the loop (d2). Publishes
        ``scheduler_job_fired`` before and ``_completed`` / ``_error`` after. An
        error on an interval job is recorded but does NOT stop the schedule."""
        job.status = STATUS_RUNNING
        job.fire_count += 1
        job.last_fired = time.time()
        await self._plane.publish(
            EVENT_JOB_FIRED,
            {**job.snapshot(), "fire_seq": job.fire_count},
            source=self._source,
        )
        try:
            fire = job.fire
            if asyncio.iscoroutinefunction(fire):
                # An async body is awaited directly on the loop (it is expected to
                # await its own I/O rather than block).
                result = await fire()
            else:
                # A plain (possibly BLOCKING) body MUST run off the event loop, or
                # a slow fire would stall the single loop (d2 — the freeze fix's
                # core invariant). Offload it to a worker thread.
                result = await asyncio.to_thread(fire)
                # A sync wrapper may itself hand back an awaitable; resolve it on
                # the loop so the final result is a plain value.
                if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
                    result = await result
            job.last_result = result
            job.last_error = None
            await self._plane.publish(
                EVENT_JOB_COMPLETED,
                {**job.snapshot(), "result": _safe(result)},
                source=self._source,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface ANY fire failure
            job.last_error = f"{type(exc).__name__}: {exc}"
            job.status = STATUS_ERROR
            await self._plane.publish(
                EVENT_JOB_ERROR,
                {**job.snapshot(), "error": job.last_error},
                source=self._source,
            )

    # -- introspection --------------------------------------------------- #
    def get(self, job_id: str) -> Optional[ScheduledJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        """A snapshot list of every job's observable state (newest last)."""
        return [j.snapshot() for j in self._jobs.values()]

    # -- cancellation / teardown (self-scoped, d8) ----------------------- #
    def cancel(self, job_id: str) -> bool:
        """Cancel ONE job by id. Returns True if a live task was cancelled.

        Cancels only that job's task; the ``scheduler_job_cancelled`` event is
        published from the task's own unwind so the state is consistent."""
        task = self._tasks.get(job_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    def stop(self) -> None:
        """Synchronously request stop of EVERY job this scheduler owns (d8).

        Cancels each owned task and flips ``started`` off. Does NOT await the
        unwind — use :meth:`shutdown` from async code to await clean teardown.
        Provided for a sync call site that just needs the cancels requested."""
        self._started = False
        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()

    async def shutdown(self) -> None:
        """Cancel every owned job and AWAIT its unwind (lifespan teardown, d8).

        Self-scoped (only this scheduler's tasks) + failure-tolerant (never
        raises, so a lifespan ``finally`` cannot mask an earlier error). After it
        returns, no scheduler task is left running."""
        self._started = False
        tasks = list(self._tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks.clear()


def _safe(value: Any) -> Any:
    """Coerce a fire result to a small JSON-shaped form for the event payload.

    Dicts/lists/scalars pass through; anything else degrades to ``repr`` so an
    event payload never carries an unbounded/odd object onto the plane."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in list(value.items())[:50]}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in list(value)[:50]]
    return repr(value)


__all__ = [
    "Scheduler",
    "ScheduledJob",
    "FireFunc",
    "KIND_INTERVAL",
    "KIND_ONE_SHOT",
    "JOB_KINDS",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "STATUS_WAITING",
    "STATUS_DONE",
    "STATUS_CANCELLED",
    "STATUS_ERROR",
    "EVENT_JOB_SCHEDULED",
    "EVENT_JOB_FIRED",
    "EVENT_JOB_COMPLETED",
    "EVENT_JOB_ERROR",
    "EVENT_JOB_CANCELLED",
    "SCHEDULER_EVENT_KINDS",
]
