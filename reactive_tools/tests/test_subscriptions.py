"""Unit coverage for the reactive-lambda registry — composition asserted with
INJECTED in-process events (no real I/O, no GPU): the in-memory EventPlane IS
the injected observable. Covers reducers, the anti-wake-storm guard, the
read-only surface, the advisory governed effect, composition, and clean teardown.

No async test plugin is assumed (matching the repo): each async body is driven
through ``asyncio.run`` from a plain sync test function.
"""
from __future__ import annotations

import asyncio

import pytest

from reactive_tools import EventPlane
from reactive_tools.event_plane import Event
from reactive_tools.subscriptions import (
    META_LAMBDA_CLOSED,
    META_LAMBDA_FIRED,
    META_LAMBDA_OBSERVATION,
    META_LAMBDA_REGISTERED,
    LambdaInputError,
    LambdaRegistry,
    build_reducer,
)


async def _drain(_plane: EventPlane) -> None:
    """Let queued driver coroutines run so injected events reach the lambdas."""
    for _ in range(5):
        await asyncio.sleep(0)


def test_reducers_are_stateful_and_timer_free():
    every = build_reducer("every:3")
    seen = [every(Event("k", {"i": i})) for i in range(7)]
    assert seen == [True, False, False, True, False, False, True]  # 1st, 4th, 7th

    distinct = build_reducer("distinct:status")
    vals = ["pending", "pending", "running", "running", "done"]
    got = [distinct(Event("k", {"status": v})) for v in vals]
    assert got == [True, False, True, False, True]  # only real changes

    match = build_reducer("match:ok=False")
    assert build_reducer("each")(Event("k"))
    assert match(Event("k", {"ok": False}))
    assert not match(Event("k", {"ok": True}))


def test_unknown_reducer_rejected():
    with pytest.raises(LambdaInputError):
        build_reducer("bogus:1")


def test_data_plane_kind_requires_a_reducer():
    async def body():
        plane = EventPlane()
        reg = LambdaRegistry(plane)
        # 'each' on a data-plane kind would wake-storm -> rejected.
        with pytest.raises(LambdaInputError):
            reg.create(["tool_result"], reducer="each")
        # With a reducer it is allowed.
        rec = reg.create(["tool_result"], reducer="every:2")
        assert rec.status == "active"
        await reg.close_all()

    asyncio.run(body())


def test_observe_all_each_is_rejected():
    # b3 reviewer inline fix: an observe-all lambda (no kinds) implicitly includes
    # the data-plane kinds, so 'each' over it must wake-storm-guard the same as a
    # literal data-plane kind would. A reducer makes observe-all legal.
    async def body():
        plane = EventPlane()
        reg = LambdaRegistry(plane)
        with pytest.raises(LambdaInputError):
            reg.create([], reducer="each")          # observe-all + each -> rejected
        with pytest.raises(LambdaInputError):
            reg.create(["*"], reducer="each")        # '*' is stripped -> observe-all
        rec = reg.create([], reducer="sample:5")    # observe-all WITH a reducer is fine
        assert rec.status == "active"
        await reg.close_all()

    asyncio.run(body())


def test_agent_creates_and_uses_lambdas_at_scale():
    async def body():
        plane = EventPlane()
        reg = LambdaRegistry(plane)

        # The agent creates several lambdas observing the b1 node-lifecycle kinds.
        a = reg.create(["agent_node_done"], label="done-watch", reducer="each")
        b = reg.create(["agent_node_launched", "agent_node_done"], label="lifecycle",
                       reducer="each", reaction="count")
        c = reg.create(["tool_result"], label="hot-sample", reducer="every:2")
        await _drain(plane)
        assert reg.active_count == 3

        # Inject events on the plane (the injected observable).
        await plane.publish("agent_node_launched", {"node_id": "n1"})
        await plane.publish("agent_node_done", {"node_id": "n1"})
        await plane.publish("agent_node_done", {"node_id": "n2"})
        for i in range(4):
            await plane.publish("tool_result", {"call_id": i, "ok": True})
        await _drain(plane)

        assert a.fire_count == 2           # two done events
        assert b.fire_count == 3           # one launched + two done
        assert c.fire_count == 2           # every:2 over 4 tool_results -> 1st, 3rd

        # Read-only surface: snapshot lists all, newest first, observe-only.
        snap = reg.snapshot()
        assert len(snap) == 3
        assert snap[0]["sub_id"] == c.sub_id
        assert snap[0]["observes"] == "tool_result [every:2]"
        assert all(set(v) >= {"sub_id", "observes", "owner", "status", "fire_count"}
                   for v in snap)

        await reg.close_all()
        await _drain(plane)
        assert reg.active_count == 0

    asyncio.run(body())


def test_meta_plane_is_the_readonly_live_channel():
    async def body():
        plane = EventPlane()
        meta = EventPlane()
        reg = LambdaRegistry(plane, meta_plane=meta)
        sub = meta.subscribe()  # the UI lambda-tab's live channel
        events: list[Event] = []

        async def collect():
            async for ev in sub:
                events.append(ev)

        task = asyncio.create_task(collect())

        rec = reg.create(["agent_node_done"], label="x", reducer="each", max_fires=1)
        await _drain(plane)
        await plane.publish("agent_node_done", {"node_id": "n1"})
        await _drain(plane)
        sub.close()
        await task

        kinds = [e.kind for e in events]
        assert META_LAMBDA_REGISTERED in kinds
        assert META_LAMBDA_FIRED in kinds
        assert META_LAMBDA_OBSERVATION in kinds   # the advisory governed effect
        assert META_LAMBDA_CLOSED in kinds        # max_fires=1 -> completion-shape close

        obs = next(e for e in events if e.kind == META_LAMBDA_OBSERVATION)
        assert obs.payload["idempotency_key"] == f"{rec.sub_id}:{obs.payload['source_seq']}"
        await reg.close_all()

    asyncio.run(body())


def test_compose_merges_kinds_with_lineage():
    async def body():
        plane = EventPlane()
        reg = LambdaRegistry(plane)
        a = reg.create(["agent_node_failed"], label="fail", reducer="each")
        b = reg.create(["agent_node_done"], label="done", reducer="each")
        merged = reg.compose([a.sub_id, b.sub_id], label="any-terminal", reducer="each")
        await _drain(plane)

        assert set(merged.kinds) == {"agent_node_failed", "agent_node_done"}
        assert merged.composed_from == (a.sub_id, b.sub_id)

        await plane.publish("agent_node_done", {"node_id": "n1"})
        await plane.publish("agent_node_failed", {"node_id": "n2"})
        await _drain(plane)
        assert merged.fire_count == 2  # observes the union as one stream
        await reg.close_all()

    asyncio.run(body())


def test_reaction_error_does_not_kill_driver(monkeypatch):
    async def body():
        plane = EventPlane()
        reg = LambdaRegistry(plane)
        rec = reg.create(["agent_node_done"], reducer="each")
        await _drain(plane)

        # Force the advisory emit to raise once; the driver must absorb it (domain
        # failure is a value, not a stream error) and keep observing.
        calls = {"n": 0}
        real = reg._emit_fire

        def boom(record, ev, note):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("advisory blew up")
            return real(record, ev, note)

        monkeypatch.setattr(reg, "_emit_fire", boom)
        await plane.publish("agent_node_done", {"node_id": "n1"})
        await _drain(plane)
        # If the error had killed the driver, the second event would not count.
        await plane.publish("agent_node_done", {"node_id": "n2"})
        await _drain(plane)
        assert rec.fire_count == 2
        assert rec.status == "active"
        await reg.close_all()

    asyncio.run(body())
