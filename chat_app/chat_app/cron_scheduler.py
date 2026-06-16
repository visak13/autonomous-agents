"""Always-on DB-backed cron FIRING scheduler service (s6).

The s2 layer built the cron *store* + tools (``cron_add`` / ``cron_list`` /
``cron_delete``) and persisted the schedule rows into the shared ``chat.db``,
deliberately leaving the bookkeeping columns (``last_run_at`` / ``next_run_at`` /
``last_status``) for s6. THIS is that s6 firing half: an always-on service that,
while the app runs, reads the DUE cron entries from that same SQLite store and,
for each due job, fires a FRESH plan through :func:`chat_app.agentic.run_agentic`
(the real shape-selection → planner-derived DAG → live runtime path — NOT
``resume_agentic``, which is only for a paused missing-specialist run).

It homes in ``chat_app`` on purpose (the action's explicit guidance): chat_app is
the one layer that already owns BOTH the cron store (via ``reactive_tools``) AND
``run_agentic`` (the app server process owns the runtime), so the firing service
can import both without inverting the dependency direction the lower layers keep.

What it guarantees
------------------
* **Fires on schedule** — a tick reads the due jobs and fires each via
  ``run_agentic`` (or the deterministic ``run_offline`` stub seam when the app is
  not in live mode — the SAME ``Planner.plan → AgentRuntime`` pipeline, only the
  transport differs).
* **Persists fire state in the DB** — after firing, the job's ``last_run_at`` /
  ``next_run_at`` / ``last_status`` are written back through
  :meth:`~reactive_tools.cron_store.CronStore.record_fire`. Because every tick
  RE-READS the schedule from SQLite (it holds no authoritative in-memory job
  list), schedules + fire state SURVIVE A RESTART: a fresh scheduler re-reads the
  rows on boot and resumes catch-up from the persisted ``last_run_at``.
* **Missed-fire catch-up, capped at 3** — on a tick (notably the first tick after
  a wake/restart) each due job's missed windows since its baseline are computed;
  AT MOST the :data:`MAX_CATCHUP` newest are replayed and any older windows are
  DROPPED, then ``last_run_at`` is advanced to the newest window. So a job that
  missed many fires while the app was down replays a bounded burst, never a flood.

Safety (d8 — the unattended-email invariant): a scheduler-fired plan reaches mail
ONLY through the recipient-hard-locked ``send_mail`` (the legacy free-``to``
``send_email`` is neither offered to nodes nor registered on the hook), so an
UNATTENDED cron fire can never send to an arbitrary recipient. Teardown is
self-scoped + failure-tolerant (it cancels only its OWN tick task), mirroring the
in-process :class:`reactive_tools.Scheduler`.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from reactive_tools import (
    CronJob,
    CronStore,
    EventPlane,
    iter_due_fire_times,
    next_fire_after,
    resolve_cron_db_path,
)

# Imported at MODULE level (not inside the fire callback) so the production fire
# path resolves these by global name on each call — a test can monkeypatch
# ``chat_app.cron_scheduler.run_agentic`` to a recorder and PROVE that a fire goes
# through run_agentic without a live GPU.
from chat_app.agentic import run_agentic, run_offline

# A FIRE body: given the due job + the specific window it is firing for, run the
# plan and return a small JSON-shaped outcome dict. Async so it can drive
# run_agentic directly.
CronFire = Callable[[CronJob, datetime], Awaitable[Any]]

# At most this many missed windows are replayed per job per tick; older windows
# are dropped (the action's catch-up cap). The newest are kept (most relevant).
MAX_CATCHUP = 3

# Default seconds between ticks. A minute is cron's finest granularity, so a
# sub-minute cadence guarantees no whole-minute window is skipped between ticks.
DEFAULT_TICK_SECONDS = 30.0

# Event kinds published on the shared plane when a job fires / its windows are
# caught up — so the UI/lambda tab can observe scheduled cron work live alongside
# run + tool lifecycle (parity with the in-process Scheduler's events).
EVENT_CRON_FIRED = "cron_job_fired"
EVENT_CRON_CAUGHT_UP = "cron_job_caught_up"
EVENT_CRON_ERROR = "cron_job_error"
CRON_SCHEDULER_EVENT_KINDS: tuple[str, ...] = (
    EVENT_CRON_FIRED,
    EVENT_CRON_CAUGHT_UP,
    EVENT_CRON_ERROR,
)


def _now_utc() -> datetime:
    """The default clock: timezone-aware UTC now (tests inject a fixed clock)."""
    return datetime.now(timezone.utc)


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    """Parse a stored ISO-8601 timestamp to an aware datetime, or None.

    The store writes UTC, second-precision, tz-explicit strings; a value that
    somehow lacks a tzinfo is treated as UTC so comparisons stay aware-vs-aware."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def make_cron_fire(
    *,
    transport_mode: str,
    registry: Any,
    hook: Any,
    live_transport: Any = None,
    shape_config: Any = None,
    timeout: float = 900.0,
) -> CronFire:
    """Build the production fire callback that runs a job's prompt as a fresh plan.

    In LIVE mode the job's prompt is driven through :func:`run_agentic` (real
    shape selection → planner-derived DAG → live Gemma runtime); otherwise the
    deterministic :func:`run_offline` stub seam runs the SAME planner→runtime
    pipeline with no GPU. Either way a FRESH plan is fired (never
    ``resume_agentic``). Each fire gets its own per-fire :class:`EventPlane` (no
    SSE consumer is attached to an unattended cron run) and a unique ``run_id``
    derived from the job + window so the run is traceable back to the fire."""

    async def fire(job: CronJob, fire_time: datetime) -> dict[str, Any]:
        plane = EventPlane()  # per-fire plane; an unattended run has no SSE viewer
        run_id = f"cron-{job.job_id}-{fire_time.strftime('%Y%m%dT%H%M%SZ')}"
        if transport_mode == "live" and live_transport is not None:
            agentic = await run_agentic(
                job.prompt,
                transport=live_transport,
                registry=registry,
                hook=hook,
                plane=plane,
                run_id=run_id,
                shape_config=shape_config,
                timeout=timeout,
                # UNATTENDED safety: a scheduler-fired run has NO user present to
                # answer a clarifying question, so the interactive ambiguity gate is
                # bypassed here — the job's prompt (authored when the user scheduled
                # it, already clarified) is acted on directly. Without this an
                # ambiguous-looking fired prompt could pause-and-never-deliver.
                skip_ambiguity=True,
            )
        else:
            agentic = await run_offline(
                job.prompt,
                registry=registry,
                hook=hook,
                plane=plane,
                run_id=run_id,
            )
        return {
            "ok": bool(getattr(agentic, "ok", False)),
            "run_id": run_id,
            "shape": getattr(agentic, "shape", None),
            "report_chars": len((getattr(agentic, "final_response", "") or "")),
        }

    return fire


class CronScheduler:
    """Always-on service that fires DB-persisted cron jobs while the app runs.

    Construct it with the resolved shared-DB path and a ``fire`` callback (the
    production one from :func:`make_cron_fire`, or a recorder in tests). Call
    :meth:`start` once at app startup to launch the tick loop and :meth:`stop` at
    teardown to cancel ONLY this scheduler's own task (self-scoped, d8). The
    schedule is read FRESH from SQLite each tick, so the service holds no
    authoritative state of its own and is inherently restart-safe.
    """

    def __init__(
        self,
        db_path: str | os.PathLike[str],
        fire: CronFire,
        *,
        clock: Optional[Callable[[], datetime]] = None,
        tick_seconds: float = DEFAULT_TICK_SECONDS,
        max_catchup: int = MAX_CATCHUP,
        plane: Optional[EventPlane] = None,
        source: str = "cron_scheduler",
    ) -> None:
        self._db_path = str(db_path)
        self._fire = fire
        self._clock = clock or _now_utc
        self._tick_seconds = float(tick_seconds)
        self._max_catchup = int(max_catchup)
        self._plane = plane
        self._source = source
        self._task: Optional[asyncio.Task] = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    # -- lifecycle ------------------------------------------------------- #
    def start(self) -> None:
        """Launch the recurring tick loop as a tracked task. Idempotent.

        Must be called from within the running event loop (the app lifespan is).
        The loop runs an immediate first tick (so a missed-fire catch-up happens
        right at boot/wake), then ticks every ``tick_seconds``."""
        if self._started:
            return
        self._started = True
        self._task = asyncio.create_task(self._loop(), name=self._source)

    async def stop(self) -> None:
        """Cancel ONLY this scheduler's tick task and await its unwind (d8).

        Self-scoped + failure-tolerant so a lifespan ``finally`` can call it
        without masking an earlier error. After it returns no tick task survives."""
        self._started = False
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _loop(self) -> None:
        """Drive ticks until cancelled. A tick failure never kills the loop."""
        try:
            while self._started:
                try:
                    await self.tick()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - a bad tick must not stop the loop
                    pass
                await asyncio.sleep(self._tick_seconds)
        except asyncio.CancelledError:
            raise

    # -- the one-pass logic (directly testable with an injected clock) --- #
    async def tick(self, now: Optional[datetime] = None) -> list[dict[str, Any]]:
        """Fire every due job once per kept window; return a per-job summary.

        For each ENABLED job: the missed windows since its baseline
        (``last_run_at`` if it has fired before, else its ``created_at`` — both
        persisted, so this resumes correctly after a restart) up to ``now`` are
        computed; the newest :attr:`_max_catchup` are FIRED (older dropped); the
        job's ``last_run_at`` is then advanced to the newest fired window and
        ``next_run_at`` recomputed. Reads the schedule FRESH from SQLite, so a
        restart picks up exactly the persisted rows + fire state."""
        now = now or self._clock()
        summaries: list[dict[str, Any]] = []
        # One short-lived store for the whole tick (no lock is held across the
        # awaited fires — only around each execute — so the cron tools' own
        # short-lived stores are never blocked for long).
        with CronStore(self._db_path) as store:
            jobs = store.list(enabled_only=True)
            for job in jobs:
                baseline = (
                    _parse_iso(job.last_run_at)
                    or _parse_iso(job.created_at)
                    or now
                )
                due = iter_due_fire_times(job.schedule, baseline, now)
                if not due:
                    continue
                kept = due[-self._max_catchup:]
                dropped = due[: len(due) - len(kept)]
                fired: list[str] = []
                last_status = "ok"
                for window in kept:
                    try:
                        await self._fire(job, window)
                        fired.append(window.isoformat())
                        await self._publish(EVENT_CRON_FIRED, job, window=window)
                    except Exception as exc:  # noqa: BLE001 - record, keep going
                        last_status = f"error: {type(exc).__name__}: {exc}"
                        await self._publish(
                            EVENT_CRON_ERROR, job, window=window, error=last_status
                        )
                # Advance persisted state to the NEWEST window (so a future tick
                # never refires it) regardless of per-fire errors — the error is
                # surfaced via last_status, never an infinite refire loop.
                newest = kept[-1]
                nxt = next_fire_after(job.schedule, now)
                store.record_fire(
                    job.job_id,
                    last_run_at=newest.isoformat(),
                    next_run_at=(nxt.isoformat() if nxt is not None else None),
                    last_status=last_status,
                )
                if dropped:
                    await self._publish(
                        EVENT_CRON_CAUGHT_UP, job, dropped=len(dropped),
                        fired=len(fired),
                    )
                summaries.append(
                    {
                        "job_id": job.job_id,
                        "name": job.name,
                        "fired": fired,
                        "fired_count": len(fired),
                        "dropped": len(dropped),
                        "last_run_at": newest.isoformat(),
                        "next_run_at": (nxt.isoformat() if nxt is not None else None),
                        "last_status": last_status,
                    }
                )
        return summaries

    async def _publish(self, kind: str, job: CronJob, **extra: Any) -> None:
        """Publish a cron lifecycle event on the shared plane (no-op if none)."""
        if self._plane is None:
            return
        payload: dict[str, Any] = {"job_id": job.job_id, "name": job.name}
        for k, v in extra.items():
            payload[k] = v.isoformat() if isinstance(v, datetime) else v
        try:
            await self._plane.publish(kind, payload, source=self._source)
        except Exception:  # noqa: BLE001 - observability must never break a fire
            pass


__all__ = [
    "CronScheduler",
    "CronFire",
    "make_cron_fire",
    "MAX_CATCHUP",
    "DEFAULT_TICK_SECONDS",
    "EVENT_CRON_FIRED",
    "EVENT_CRON_CAUGHT_UP",
    "EVENT_CRON_ERROR",
    "CRON_SCHEDULER_EVENT_KINDS",
]
