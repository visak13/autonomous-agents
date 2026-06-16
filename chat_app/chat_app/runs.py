"""In-process background-run registry — the freeze-decouple POC (s1/a1).

Round-1 freeze, root cause (two coupled faults):

1. The blocking phi HTTP round-trip ran INLINE on the single asyncio event loop
   (``chain.run`` -> ``transport.chat``), so one slow phi call (GPU shared with
   GeoGuessr) stalled the loop and froze every request — including ``/health``.
   That fault is fixed at the seam (``agent_runtime`` now offloads the chain run
   via ``asyncio.to_thread``; ``toolargs`` already did).

2. The POST handlers ``await``-ed the WHOLE agent run (planner + runtime, each a
   slow phi round-trip) before returning, so even with the loop unblocked the
   request itself hung for the full run duration — the UI had nothing to show
   and looked frozen.

This module retires fault 2: a handler SUBMITS the run as a tracked background
:class:`asyncio.Task` and RETURNS IMMEDIATELY with a run id. The run executes
OFF the request path and publishes its lifecycle onto the chat's existing
in-process :class:`~reactive_tools.EventPlane` — the SAME plane the SSE stream
already relays (the live-updates channel is REUSED, not duplicated, per the
action's "reuse the reactive event plane" constraint). The client watches that
SSE stream live and/or polls ``GET /runs/{run_id}`` for terminal status.

Self-contained + in-process (d2): no shell forking, no second service. The
manager is owned by ``app.state`` and constructed/torn down by the lifespan, so
outstanding background runs are cancelled cleanly on shutdown.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

# A run body is a zero-arg factory returning the awaitable that does the work.
# It is a factory (not a bare coroutine) so the manager owns task creation and
# nothing is awaited on the caller's request path.
RunBody = Callable[[], Awaitable[Any]]

# Terminal + live status values a run moves through.
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
_TERMINAL = frozenset({STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED})


@dataclass
class RunRecord:
    """The observable state of one background run.

    ``result`` holds the run body's return value once ``done``; ``error`` holds
    the stringified failure once ``failed``. ``started`` / ``ended`` are wall
    clock for a coarse duration read (the precise per-node lifecycle is on the
    SSE stream, not here)."""

    run_id: str
    chat_id: str
    status: str = STATUS_RUNNING
    result: Any = None
    error: Optional[str] = None
    started: float = field(default_factory=time.time)
    ended: Optional[float] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "chat_id": self.chat_id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "started": self.started,
            "ended": self.ended,
            "duration_s": (self.ended - self.started) if self.ended else None,
        }


class RunManager:
    """Submit-and-forget registry of background agent runs (one per app).

    ``submit`` returns a :class:`RunRecord` SYNCHRONOUSLY after only scheduling
    the work — the request handler returns that record's id immediately, never
    awaiting the run. The work runs as a tracked task whose completion callback
    flips the record to its terminal status; the task is dropped from the live
    set so finished runs do not accumulate references.
    """

    def __init__(self) -> None:
        self._runs: Dict[str, RunRecord] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

    def submit(self, chat_id: str, body: RunBody, *, run_id: Optional[str] = None) -> RunRecord:
        """Schedule ``body`` as a background task and return its record at once.

        The returned record is already ``running``; its status is updated in
        place by the task. Must be called from within the running event loop
        (it is — FastAPI handlers run on it)."""
        rid = run_id or f"run-{uuid.uuid4().hex[:12]}"
        record = RunRecord(run_id=rid, chat_id=chat_id)
        self._runs[rid] = record
        task = asyncio.create_task(self._drive(record, body), name=f"run:{rid}")
        self._tasks[rid] = task
        return record

    async def _drive(self, record: RunRecord, body: RunBody) -> None:
        """Run the body, recording its terminal status. Never raises out."""
        try:
            record.result = await body()
            record.status = STATUS_DONE
        except asyncio.CancelledError:
            record.status = STATUS_CANCELLED
            record.error = "run cancelled (server shutdown)"
            record.ended = time.time()
            raise  # let the cancellation propagate so shutdown awaits cleanly
        except Exception as exc:  # noqa: BLE001 - surface ANY failure as status
            record.status = STATUS_FAILED
            record.error = f"{type(exc).__name__}: {exc}"
        finally:
            if record.ended is None:
                record.ended = time.time()
            # Drop the finished task ref so completed runs don't pin memory.
            self._tasks.pop(record.run_id, None)

    def get(self, run_id: str) -> Optional[RunRecord]:
        return self._runs.get(run_id)

    def list_runs(self) -> list[RunRecord]:
        return list(self._runs.values())

    @property
    def active_count(self) -> int:
        return len(self._tasks)

    async def shutdown(self) -> None:
        """Cancel every outstanding run and await its unwind (lifespan teardown).

        Self-scoped + failure-tolerant: cancels only the tasks this manager
        owns, and never raises so shutdown cannot mask an earlier error."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks.clear()


__all__ = [
    "RunManager",
    "RunRecord",
    "RunBody",
    "STATUS_RUNNING",
    "STATUS_DONE",
    "STATUS_FAILED",
    "STATUS_CANCELLED",
]
