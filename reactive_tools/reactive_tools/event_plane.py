"""In-process reactive event plane — the bus tool results flow back on.

This is the ``reactive_tools`` subsystem from the design doc: the plane on
which tool invocations and their results are published, and on which the
planner / agent runtime subscribes to react.

IN-PROCESS CONSTRAINT (d2 — load-bearing, non-negotiable)
---------------------------------------------------------
This event plane is **purely in-process**. It is built on ``asyncio`` and
in-memory data structures ONLY. It deliberately does NOT — and must never —
use the eda-base3 broker/pool HTTP services, sockets, subprocesses, files, or
Claude. The whole concurrency model for this app is in-process (RxPY/asyncio
tasks + coroutines), not HTTP microservices or shell forking (d2). Anything
that crosses a process boundary belongs to a different layer, not here.

Design (dependency-light, d10)
------------------------------
A minimal asyncio pub/sub:

- :class:`Event`        — an immutable ``(kind, payload)`` carrier with a
  monotonically increasing sequence id and an optional ``source`` tag.
- :class:`Subscription` — an async-iterable handle backed by an
  ``asyncio.Queue``; close it to unsubscribe (also usable as an async context
  manager).
- :class:`EventPlane`   — the bus: :meth:`publish` fans an event out to every
  live subscriber; :meth:`subscribe` returns a :class:`Subscription`, optionally
  filtered by ``kind``.

Delivery is async and fan-out: every current subscriber whose filter matches
receives every published event, in publish order, each through its own queue
so a slow consumer cannot drop another consumer's events.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Iterable, Optional


@dataclass(frozen=True)
class Event:
    """An immutable event flowing on the plane.

    Attributes
    ----------
    kind:
        The event type (e.g. ``"tool_result"``, ``"tool_call"``). Subscribers
        filter on this.
    payload:
        Arbitrary in-process Python object carried with the event. Since the
        plane never crosses a process boundary (d2), this need not be
        serialisable.
    seq:
        Monotonic sequence id assigned by the plane at publish time; ``-1`` for
        an event that has not been published yet.
    source:
        Optional free-form tag naming who emitted the event.
    """

    kind: str
    payload: Any = None
    seq: int = -1
    source: Optional[str] = None


class Subscription:
    """A live subscription handle — an async-iterable stream of events.

    Iterate it with ``async for`` to receive matching events as they are
    published. Call :meth:`close` (or use it as an ``async with`` block) to
    unsubscribe; the plane then stops delivering to it and the iterator ends.
    """

    def __init__(
        self,
        plane: "EventPlane",
        *,
        kinds: Optional[frozenset[str]] = None,
        predicate: Optional[Callable[[Event], bool]] = None,
        maxsize: int = 0,
    ) -> None:
        self._plane = plane
        self._kinds = kinds
        self._predicate = predicate
        self._queue: asyncio.Queue[Optional[Event]] = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    # -- internal: called by the plane only ------------------------------
    def _matches(self, event: Event) -> bool:
        if self._kinds is not None and event.kind not in self._kinds:
            return False
        if self._predicate is not None and not self._predicate(event):
            return False
        return True

    async def _deliver(self, event: Event) -> None:
        if not self._closed:
            await self._queue.put(event)

    # -- public surface ---------------------------------------------------
    @property
    def closed(self) -> bool:
        return self._closed

    async def get(self) -> Event:
        """Await the next matching event. Raises if the subscription closed."""
        event = await self._queue.get()
        if event is None:  # sentinel pushed by close()
            raise StopAsyncIteration
        return event

    def close(self) -> None:
        """Unsubscribe. Idempotent; ends any in-flight ``async for``."""
        if self._closed:
            return
        self._closed = True
        self._plane._remove(self)
        # Wake any iterator parked on an empty queue.
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:  # pragma: no cover - maxsize=0 never full
            self._plane._loop_call_soon(self._queue.put_nowait, None)

    def __aiter__(self) -> AsyncIterator[Event]:
        return self

    async def __anext__(self) -> Event:
        event = await self._queue.get()
        if event is None:  # closed
            raise StopAsyncIteration
        return event

    async def __aenter__(self) -> "Subscription":
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.close()


class EventPlane:
    """An in-process publish/subscribe event bus.

    Not thread-safe by design: it is meant to be driven from a single asyncio
    event loop (the in-process agent runtime). All delivery is in-memory and
    in-process (d2).
    """

    def __init__(self) -> None:
        self._subscribers: list[Subscription] = []
        self._seq = 0

    # -- subscription management -----------------------------------------
    def subscribe(
        self,
        kinds: Optional[Iterable[str]] = None,
        *,
        predicate: Optional[Callable[[Event], bool]] = None,
        maxsize: int = 0,
    ) -> Subscription:
        """Register a subscriber and return its :class:`Subscription`.

        Parameters
        ----------
        kinds:
            If given, only events whose ``kind`` is in this set are delivered;
            ``None`` means every kind.
        predicate:
            Optional extra per-event filter applied after the ``kinds`` check.
        maxsize:
            Backing queue bound; ``0`` (default) is unbounded.
        """
        kind_set = frozenset(kinds) if kinds is not None else None
        sub = Subscription(self, kinds=kind_set, predicate=predicate, maxsize=maxsize)
        self._subscribers.append(sub)
        return sub

    def _remove(self, sub: Subscription) -> None:
        try:
            self._subscribers.remove(sub)
        except ValueError:
            pass

    def _loop_call_soon(self, fn: Callable[..., Any], *args: Any) -> None:
        try:
            asyncio.get_running_loop().call_soon(fn, *args)
        except RuntimeError:  # pragma: no cover - no loop running
            fn(*args)

    # -- publishing -------------------------------------------------------
    async def publish(
        self,
        kind: str,
        payload: Any = None,
        *,
        source: Optional[str] = None,
    ) -> Event:
        """Publish an event; fan it out to every matching live subscriber.

        Returns the stamped :class:`Event` (with its assigned ``seq``). Delivery
        is async: each subscriber receives the event through its own queue, in
        publish order, so one slow consumer cannot starve another.
        """
        self._seq += 1
        event = Event(kind=kind, payload=payload, seq=self._seq, source=source)
        # Snapshot so a subscriber closing mid-fanout doesn't mutate the list.
        for sub in list(self._subscribers):
            if sub.closed:
                continue
            if sub._matches(event):
                await sub._deliver(event)
        return event

    def publish_nowait(
        self,
        kind: str,
        payload: Any = None,
        *,
        source: Optional[str] = None,
    ) -> Event:
        """Synchronous publish for non-async callers (e.g. a tool body).

        Stamps and enqueues the event without awaiting; subscribers pick it up
        on their next loop turn. Useful when the producer is plain sync code.
        """
        self._seq += 1
        event = Event(kind=kind, payload=payload, seq=self._seq, source=source)
        for sub in list(self._subscribers):
            if sub.closed or not sub._matches(event):
                continue
            try:
                sub._queue.put_nowait(event)
            except asyncio.QueueFull:
                self._loop_call_soon(sub._queue.put_nowait, event)
        return event

    @property
    def subscriber_count(self) -> int:
        """Number of currently-live subscribers."""
        return len(self._subscribers)

    def close(self) -> None:
        """Close every subscription. Idempotent."""
        for sub in list(self._subscribers):
            sub.close()
