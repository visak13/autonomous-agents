"""Tests for the in-process :class:`reactive_tools.scheduler.Scheduler`.

Mirrors the house pattern (test_subscriptions): each async body is driven via
``asyncio.run`` from a plain sync test function — no pytest-asyncio dependency.

Coverage (the a3 acceptance points):
- an INTERVAL job fires N times then is cancelled / bounded;
- a ONE-SHOT job fires exactly once;
- ``stop()`` / ``shutdown()`` cancels cleanly (self-scoped, no leak);
- the EVENT PLANE sees the scheduler_job_* lifecycle;
- the EVENT LOOP is NEVER blocked — a /health-style probe stays responsive
  while a job's BLOCKING work runs (proving the to_thread off-loop offload).
"""
from __future__ import annotations

import asyncio
import time

from reactive_tools import (
    EVENT_JOB_COMPLETED,
    EVENT_JOB_FIRED,
    EVENT_JOB_SCHEDULED,
    EventPlane,
    ScheduledJob,
    Scheduler,
)
from reactive_tools.scheduler import (
    EVENT_JOB_CANCELLED,
    KIND_INTERVAL,
    KIND_ONE_SHOT,
    STATUS_CANCELLED,
    STATUS_DONE,
)


def test_one_shot_fires_exactly_once() -> None:
    async def body() -> None:
        plane = EventPlane()
        sched = Scheduler(plane)
        sched.start()
        calls: list[int] = []

        async def fire() -> str:
            calls.append(1)
            return "report-body"

        job_id = sched.schedule(
            ScheduledJob(fire=fire, kind=KIND_ONE_SHOT, initial_delay=0.0, name="brief")
        )
        # Give the one-shot task time to run; then assert it fired once and is done.
        await asyncio.sleep(0.1)
        job = sched.get(job_id)
        assert job is not None
        assert job.fire_count == 1
        assert calls == [1]
        assert job.status == STATUS_DONE
        assert job.last_result == "report-body"
        # No live task should remain (it completed and self-removed).
        assert sched.active_count == 0
        await sched.shutdown()

    asyncio.run(body())


def test_interval_fires_n_times_then_bounded() -> None:
    async def body() -> None:
        plane = EventPlane()
        sched = Scheduler(plane)
        sched.start()
        fires: list[float] = []

        async def fire() -> None:
            fires.append(time.time())

        # Short interval, bounded to 3 fires so the test terminates.
        job_id = sched.schedule(
            ScheduledJob(
                fire=fire,
                kind=KIND_INTERVAL,
                interval_seconds=0.02,
                initial_delay=0.0,
                max_fires=3,
                name="ticker",
            )
        )
        # Wait long enough for 3 fires (3 * 0.02 + slack).
        await asyncio.sleep(0.25)
        job = sched.get(job_id)
        assert job is not None
        assert job.fire_count == 3, f"expected exactly 3 fires, got {job.fire_count}"
        assert job.status == STATUS_DONE
        assert sched.active_count == 0
        await sched.shutdown()

    asyncio.run(body())


def test_interval_cancel_stops_firing() -> None:
    async def body() -> None:
        plane = EventPlane()
        sched = Scheduler(plane)
        sched.start()
        fires = {"n": 0}

        async def fire() -> None:
            fires["n"] += 1

        job_id = sched.schedule(
            ScheduledJob(fire=fire, kind=KIND_INTERVAL, interval_seconds=0.02,
                         initial_delay=0.0, name="unbounded")
        )
        await asyncio.sleep(0.07)  # let it tick a few times
        assert sched.cancel(job_id) is True
        await asyncio.sleep(0)  # let the cancel unwind
        count_at_cancel = fires["n"]
        assert count_at_cancel >= 1
        await asyncio.sleep(0.1)  # ensure NO further fires after cancel
        assert fires["n"] == count_at_cancel, "job kept firing after cancel"
        job = sched.get(job_id)
        assert job is not None and job.status == STATUS_CANCELLED
        # Cancelling an already-cancelled / unknown job is a no-op False.
        assert sched.cancel(job_id) is False
        assert sched.cancel("no-such-job") is False
        await sched.shutdown()

    asyncio.run(body())


def test_shutdown_is_self_scoped_and_clean() -> None:
    async def body() -> None:
        plane = EventPlane()
        sched = Scheduler(plane)
        sched.start()

        async def fire() -> None:
            await asyncio.sleep(0.005)

        for i in range(3):
            sched.schedule(
                ScheduledJob(fire=fire, kind=KIND_INTERVAL, interval_seconds=0.02,
                             initial_delay=0.0, name=f"job{i}")
            )
        # An UNRELATED task the scheduler does NOT own must survive shutdown.
        survivor_ran = {"after": False}

        async def survivor() -> None:
            await asyncio.sleep(0.2)
            survivor_ran["after"] = True

        other = asyncio.create_task(survivor())
        await asyncio.sleep(0.03)
        assert sched.active_count == 3
        await sched.shutdown()
        assert sched.active_count == 0  # all OWN tasks reaped
        assert sched.started is False
        assert other.done() is False  # the unrelated task was NOT cancelled
        await other
        assert survivor_ran["after"] is True
        # Idempotent: a second shutdown is a harmless no-op.
        await sched.shutdown()

    asyncio.run(body())


def test_event_plane_sees_job_lifecycle() -> None:
    async def body() -> None:
        plane = EventPlane()
        sched = Scheduler(plane)
        sched.start()
        seen: list[str] = []
        sub = plane.subscribe(
            kinds=(EVENT_JOB_SCHEDULED, EVENT_JOB_FIRED, EVENT_JOB_COMPLETED)
        )

        async def collect() -> None:
            async for ev in sub:
                seen.append(ev.kind)
                if seen.count(EVENT_JOB_COMPLETED) >= 1:
                    break

        collector = asyncio.create_task(collect())
        await asyncio.sleep(0)  # ensure the subscriber is parked before we publish

        async def fire() -> str:
            return "ok"

        sched.schedule(ScheduledJob(fire=fire, kind=KIND_ONE_SHOT, initial_delay=0.0))
        await asyncio.wait_for(collector, timeout=1.0)
        # The full ordered lifecycle reached the plane.
        assert EVENT_JOB_SCHEDULED in seen
        assert EVENT_JOB_FIRED in seen
        assert EVENT_JOB_COMPLETED in seen
        assert seen.index(EVENT_JOB_FIRED) < seen.index(EVENT_JOB_COMPLETED)
        sub.close()
        await sched.shutdown()

    asyncio.run(body())


def test_blocking_fire_does_not_block_event_loop() -> None:
    """A job whose fire body BLOCKS (time.sleep) must run off-loop so the loop
    stays responsive — a /health-style probe keeps ticking during the blocking
    work. Proves the to_thread offload (d2 — the freeze fix's core invariant)."""

    async def body() -> None:
        plane = EventPlane()
        sched = Scheduler(plane)
        sched.start()

        BLOCK = 0.3
        fire_done = {"v": False}

        def blocking_fire() -> str:
            # SYNCHRONOUS blocking work — the scheduler must NOT run this on the
            # event loop. Run as a plain (non-coroutine) callable so the scheduler
            # offloads it via asyncio.to_thread.
            time.sleep(BLOCK)
            fire_done["v"] = True
            return "blocked-then-done"

        # A lightweight async "probe" that ticks rapidly; if the loop were blocked
        # by the fire body, these ticks would stall for ~BLOCK seconds.
        probe_ticks = {"n": 0}

        async def health_probe() -> None:
            for _ in range(20):
                probe_ticks["n"] += 1
                await asyncio.sleep(0.01)

        probe = asyncio.create_task(health_probe())
        t0 = time.perf_counter()
        sched.schedule(
            ScheduledJob(fire=blocking_fire, kind=KIND_ONE_SHOT, initial_delay=0.0)
        )
        await probe  # ~0.2s of ticks; should NOT be delayed by the 0.3s block
        probe_elapsed = time.perf_counter() - t0

        # The probe completed its 20 ticks in roughly 0.2s — NOT serialized behind
        # the 0.3s blocking fire. Generous bound for CI jitter.
        assert probe_ticks["n"] == 20
        assert probe_elapsed < BLOCK + 0.15, (
            f"event loop appears blocked: probe took {probe_elapsed:.3f}s "
            f"(blocking fire was {BLOCK}s)"
        )
        # Let the blocking fire finish, then confirm it actually ran.
        await asyncio.sleep(BLOCK)
        assert fire_done["v"] is True
        await sched.shutdown()

    asyncio.run(body())
