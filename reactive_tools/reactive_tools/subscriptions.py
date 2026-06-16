"""Reactive lambdas at scale — agent-CREATED + agent-USED subscriptions, with a
read-only live-subscriptions surface (d12 / d14 / d15).

WHAT THIS IS
------------
A *reactive lambda* is a subscription the AGENT/LLM itself creates, at runtime,
as suits a task — a first-class capability the agent composes AT SCALE, not just
an internal bus. Each lambda **observes** events on the shared in-process
:class:`~reactive_tools.event_plane.EventPlane` (the b1 run engine emits per-node
lifecycle events there; tool calls/results ride it too), optionally **reduces**
the stream, and on a fire emits an **advisory** observation. The USER never
authors a lambda — the only user interaction is OBSERVING, via the read-only
live-subscriptions surface (snapshot query + a meta-plane live channel) the UI
lambda-tab consumes (d15).

GROUNDING (spec-edp-reactive-framework-engineer craft, applied STANDALONE)
--------------------------------------------------------------------------
The eda-base3 RxPY event plane is borrowed as *concepts only* — no broker/pool/
driver coupling (d2: purely in-process asyncio). The load-bearing doctrines that
DO carry over and are enforced here:

- **rx observes, CRUD mutates.** A lambda NEVER mutates state. Its only sanctioned
  write-on-emit is a *governed, advisory-by-default* observation event — allow-kind,
  idempotency-keyed ``(sub_id, source_seq)``, never feeding back as a command.
- **Reduction, not transport, at high frequency.** A *data-plane* kind
  (``tool_call`` / ``tool_result``) must NOT be subscribed straight to a reaction —
  the first-tick burst would wake-storm. Creating a lambda on a data-plane kind
  REQUIRES an explicit reducing operator (``every:N`` / ``sample:N`` /
  ``distinct:<field>`` / ``match:<field>=<value>``); control-plane lifecycle kinds
  may use ``each``. (Concept parity with the spec's
  ``sample_ms``/``debounce_ms``/``distinct_until_changed``/``scan+filter`` —
  realised here as per-event-stateful, timer-free reducers so the build stays
  dependency-light and deterministic.)
- **Domain failures are values, not stream errors.** A reaction that raises is
  recorded as a fired-with-error observation; it never tears down the driver (one
  reaction crash must not kill the whole lambda).
- **Completion-shape + clean teardown.** Every lambda can bound itself
  (``max_fires``) and is torn down cleanly on BOTH normal close and failure: the
  driver task is cancelled + awaited, the backing subscription closed, and a
  ``lambda_closed`` meta event emitted — no orphaned task left on the loop.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from .event_plane import Event, EventPlane

# --------------------------------------------------------------------------- #
# Plane vocabulary
# --------------------------------------------------------------------------- #
# Kinds that are HIGH-FREQUENCY data-plane signals — they MUST be reduced before a
# lambda reacts (never subscribe straight to a reaction: the s6 wake-storm rule).
DATA_PLANE_KINDS = frozenset({"tool_call", "tool_result"})

# Meta-plane event kinds — the READ-ONLY live-subscriptions surface (d15). The UI
# lambda-tab subscribes to these to render the agents' live lambdas. They are
# emitted on a SEPARATE plane from the observed events so (a) a lambda observing
# "everything" can never recurse on its own meta-events and (b) the user-facing
# observe-only channel is cleanly isolated from the agent's working plane.
META_LAMBDA_REGISTERED = "lambda_registered"
META_LAMBDA_FIRED = "lambda_fired"
META_LAMBDA_CLOSED = "lambda_closed"
META_LAMBDA_OBSERVATION = "lambda_observation"  # the advisory reaction (governed)


class LambdaInputError(ValueError):
    """A reactive lambda was declared with invalid/unsafe parameters."""


# --------------------------------------------------------------------------- #
# Reducers — the reduction layer (spec: "the rx layer's primary job at high
# frequency is REDUCTION, not transport"). Each reducer is a STATEFUL per-event
# predicate ``(Event) -> bool``: True = let this event through to a fire, False =
# absorb it. Timer-free + deterministic (no real clock) so composition is unit-
# testable with injected events and the data-plane never wake-storms.
# --------------------------------------------------------------------------- #
def _reducer_each() -> Callable[[Event], bool]:
    """``each`` — let every matching event through (control-plane only)."""
    return lambda _ev: True


def _reducer_every(n: int) -> Callable[[Event], bool]:
    """``every:N`` — scan-threshold: fire once per N matching events (1st, N+1th…)."""
    if n < 1:
        raise LambdaInputError("every:N requires N >= 1")
    counter = itertools.count(0)

    def passes(_ev: Event) -> bool:
        return next(counter) % n == 0

    return passes


def _reducer_sample(n: int) -> Callable[[Event], bool]:
    """``sample:N`` — fire on the Nth, 2Nth… event (drop the first N-1; a sparse
    sample of a hot stream — concept parity with the spec's ``sample_ms``)."""
    if n < 1:
        raise LambdaInputError("sample:N requires N >= 1")
    counter = itertools.count(1)

    def passes(_ev: Event) -> bool:
        return next(counter) % n == 0

    return passes


def _reducer_distinct(field_path: str) -> Callable[[Event], bool]:
    """``distinct:<field>`` — distinct_until_changed on ``payload[field]``: fire
    only when the field's value actually CHANGES (real-change-only)."""
    if not field_path:
        raise LambdaInputError("distinct:<field> requires a field name")
    sentinel = object()
    last: list[Any] = [sentinel]

    def passes(ev: Event) -> bool:
        val = _payload_field(ev, field_path)
        if val != last[0] or last[0] is sentinel:
            last[0] = val
            return True
        return False

    return passes


def _reducer_match(field_path: str, value: str) -> Callable[[Event], bool]:
    """``match:<field>=<value>`` — filter: only events whose ``payload[field]``
    stringifies to ``value`` pass through."""
    if not field_path:
        raise LambdaInputError("match:<field>=<value> requires a field name")

    def passes(ev: Event) -> bool:
        return str(_payload_field(ev, field_path)) == value

    return passes


def _payload_field(ev: Event, field_path: str) -> Any:
    """Read ``ev.payload[field]`` defensively (payload may be any object)."""
    payload = ev.payload
    if isinstance(payload, Mapping):
        return payload.get(field_path)
    return getattr(payload, field_path, None)


def build_reducer(reducer: str) -> Callable[[Event], bool]:
    """Parse a reducer spec string into a stateful per-event predicate.

    Grammar (deliberately tiny + declarative so a phi-driven agent can emit it):
      ``each`` | ``every:<n>`` | ``sample:<n>`` | ``distinct:<field>``
      | ``match:<field>=<value>``
    """
    reducer = (reducer or "each").strip()
    if reducer == "each":
        return _reducer_each()
    head, _, tail = reducer.partition(":")
    if head == "every":
        return _reducer_every(int(tail))
    if head == "sample":
        return _reducer_sample(int(tail))
    if head == "distinct":
        return _reducer_distinct(tail)
    if head == "match":
        field_name, _, value = tail.partition("=")
        return _reducer_match(field_name, value)
    raise LambdaInputError(
        f"unknown reducer {reducer!r}; use each | every:N | sample:N | "
        "distinct:<field> | match:<field>=<value>"
    )


# --------------------------------------------------------------------------- #
# The lambda record — the observe-only view the UI lambda-tab reads
# --------------------------------------------------------------------------- #
@dataclass
class LambdaRecord:
    """The registry entry for ONE agent-created reactive lambda.

    This dataclass IS the read-only projection the live-subscriptions surface
    serves: ``sub_id`` (which subscription), ``observes`` (what it watches),
    ``owner`` (which node/run created it), ``status``, plus live fire counters."""

    sub_id: str
    label: str
    kinds: tuple[str, ...]
    reducer: str
    reaction: str
    owner: dict[str, Any]
    status: str = "active"          # active | closed
    created_seq: int = 0
    seen_count: int = 0             # matching events the lambda has seen
    fire_count: int = 0             # events that passed the reducer (reactions)
    last_event_kind: Optional[str] = None
    last_fired_seq: Optional[int] = None
    closed_seq: Optional[int] = None
    composed_from: tuple[str, ...] = ()

    @property
    def observes(self) -> str:
        """A one-line human description of WHAT this lambda observes (for the UI)."""
        base = "+".join(self.kinds) if self.kinds else "*"
        return f"{base} [{self.reducer}]"

    def as_view(self) -> dict[str, Any]:
        """The read-only dict the UI lambda-tab consumes (no internals leaked)."""
        return {
            "sub_id": self.sub_id,
            "label": self.label,
            "observes": self.observes,
            "kinds": list(self.kinds),
            "reducer": self.reducer,
            "reaction": self.reaction,
            "owner": dict(self.owner),
            "status": self.status,
            "created_seq": self.created_seq,
            "seen_count": self.seen_count,
            "fire_count": self.fire_count,
            "last_event_kind": self.last_event_kind,
            "last_fired_seq": self.last_fired_seq,
            "closed_seq": self.closed_seq,
            "composed_from": list(self.composed_from),
        }


# --------------------------------------------------------------------------- #
# The registry — create / list / compose / close lambdas at scale
# --------------------------------------------------------------------------- #
class LambdaRegistry:
    """The runtime home of the agents' reactive lambdas (d12 — one shared registry).

    Bound to the shared observed plane (where the run engine + tools publish) and
    a SEPARATE ``meta_plane`` carrying the read-only live-subscriptions surface
    (``lambda_registered`` / ``lambda_fired`` / ``lambda_closed`` /
    ``lambda_observation``). The agent calls :meth:`create` / :meth:`compose` to
    spin up lambdas and :meth:`snapshot` is the observe-only query; the user only
    ever reads — there is deliberately NO user-facing authoring path here.
    """

    def __init__(
        self,
        plane: EventPlane,
        *,
        meta_plane: Optional[EventPlane] = None,
    ) -> None:
        self.plane = plane
        # The read-only surface's channel. Isolated from ``plane`` so the UI's
        # observe-only stream never mixes with the agent's working events and a
        # watch-everything lambda cannot recurse on its own meta-events.
        self.meta_plane = meta_plane if meta_plane is not None else EventPlane()
        self._records: dict[str, LambdaRecord] = {}
        self._drivers: dict[str, asyncio.Task] = {}
        self._subs: dict[str, Any] = {}  # sub_id -> backing Subscription
        self._id_seq = 0

    # -- id minting ------------------------------------------------------- #
    def _next_id(self) -> str:
        self._id_seq += 1
        return f"lam-{self._id_seq:04d}"

    # -- create ----------------------------------------------------------- #
    def create(
        self,
        kinds: Sequence[str],
        *,
        label: str = "",
        reducer: str = "each",
        reaction: str = "advisory",
        owner: Optional[Mapping[str, Any]] = None,
        max_fires: Optional[int] = None,
        composed_from: Sequence[str] = (),
        note: str = "",
        source_plane: Optional[EventPlane] = None,
    ) -> LambdaRecord:
        """Create + start ONE agent reactive lambda. Returns its :class:`LambdaRecord`.

        ``kinds``   — event kinds to observe (``[]``/``["*"]`` = every kind).
        ``reducer`` — the reduction operator (see :func:`build_reducer`). A
                      data-plane kind REQUIRES a non-``each`` reducer (anti
                      wake-storm) — else :class:`LambdaInputError`.
        ``reaction``— ``advisory`` (emit a governed ``lambda_observation`` on each
                      fire) or ``count`` (count only, emit nothing extra). Both are
                      observe-only; a lambda NEVER mutates.
        ``max_fires``— completion-shape: auto-close after this many fires.
        ``source_plane`` — the plane to OBSERVE events on; defaults to this
                      registry's ``plane``. This decouples WHERE events are
                      observed from WHERE the lambda is recorded: a single shared
                      registry (the one the UI live-subscriptions surface reads)
                      can observe a PER-RUN plane (e.g. the agent runtime's
                      per-chat plane the node-lifecycle events actually flow on)
                      while still recording the lambda + emitting its meta events
                      on this registry's own ``meta_plane`` (s9/a2 — closes the
                      s7 F1 gap where runs emit on a per-chat plane the shared
                      registry never observed, so the lambda tab stayed empty).

        MUST be called from within the running event loop (it spawns the driver
        task) — the agent-facing tool wrapper is ``async`` so this holds.
        """
        kind_list = tuple(k for k in (kinds or ()) if k and k != "*")
        self._guard_reducer(kind_list, reducer)
        if reaction not in ("advisory", "count"):
            raise LambdaInputError(f"reaction must be 'advisory' or 'count', got {reaction!r}")

        # Observe on the requested source plane (the run's plane), else this
        # registry's own plane. Records + meta events still live on THIS registry,
        # so the read-only surface shows the lambda regardless of which plane it
        # observes.
        observed = source_plane if source_plane is not None else self.plane
        sub_id = self._next_id()
        rec = LambdaRecord(
            sub_id=sub_id,
            label=label or f"lambda-{sub_id}",
            kinds=kind_list,
            reducer=reducer,
            reaction=reaction,
            owner=dict(owner or {}),
            created_seq=observed._seq,
            composed_from=tuple(composed_from),
        )
        self._records[sub_id] = rec

        # Subscribe on the observed plane. None kinds => observe every kind.
        sub = observed.subscribe(kinds=kind_list or None)
        self._subs[sub_id] = sub
        predicate = build_reducer(reducer)
        task = asyncio.create_task(
            self._drive(rec, sub, predicate, max_fires=max_fires, note=note),
            name=f"lambda:{sub_id}",
        )
        self._drivers[sub_id] = task

        self.meta_plane.publish_nowait(
            META_LAMBDA_REGISTERED, rec.as_view(), source=f"lambda:{sub_id}"
        )
        return rec

    def _guard_reducer(self, kinds: tuple[str, ...], reducer: str) -> None:
        """Enforce the anti-wake-storm rule: a data-plane kind cannot be observed
        with ``each`` — it MUST first pass a reducing operator.

        An OBSERVE-ALL lambda (empty ``kinds`` => every kind) is treated as hot
        too: it necessarily includes the data-plane kinds (``tool_call`` /
        ``tool_result``), so ``each`` over observe-all would wake-storm exactly as
        a literal ``["tool_result"]`` would. The earlier guard only inspected an
        EXPLICIT kind list and so let observe-all + ``each`` slip through — closed
        here (b3 reviewer inline fix)."""
        if reducer.strip() == "each":
            if not kinds:
                raise LambdaInputError(
                    "an observe-all lambda (no kinds) cannot use reducer 'each' "
                    "(it includes the data-plane kinds tool_call/tool_result and "
                    "would wake-storm); add a reducer, e.g. every:10 / sample:25 / "
                    "distinct:<field> / match:<field>=<value>, or name explicit "
                    "control-plane kinds"
                )
            hot = [k for k in kinds if k in DATA_PLANE_KINDS]
            if hot:
                raise LambdaInputError(
                    f"data-plane kind(s) {hot} cannot be observed with reducer "
                    "'each' (would wake-storm); add a reducer, e.g. every:10 / "
                    "sample:25 / distinct:<field> / match:<field>=<value>"
                )
        # Validate the reducer string eagerly so a bad spec fails at create-time,
        # not silently inside the driver.
        build_reducer(reducer)

    # -- the driver (per-lambda) ------------------------------------------ #
    async def _drive(
        self,
        rec: LambdaRecord,
        sub: Any,
        predicate: Callable[[Event], bool],
        *,
        max_fires: Optional[int],
        note: str,
    ) -> None:
        """Consume the backing subscription, reduce, and fire (advisory).

        Teardown holds on BOTH clean completion and failure (the ``finally``
        closes the subscription, marks the record closed, and emits the meta
        ``lambda_closed`` — no orphaned task/subscription). A reaction that raises
        is captured as a fired-with-error observation (a domain failure is a
        value, NOT a stream error that would kill the driver)."""
        # Idempotency ledger for the governed advisory effect: a (sub_id, source
        # seq) pair fires at most once, so any replay can't double-emit.
        fired_keys: set[int] = set()
        try:
            async for ev in sub:
                rec.seen_count += 1
                rec.last_event_kind = ev.kind
                if not predicate(ev):
                    continue
                if ev.seq in fired_keys:
                    continue  # idempotency: never re-fire the same source event
                fired_keys.add(ev.seq)
                rec.fire_count += 1
                rec.last_fired_seq = ev.seq
                # A reaction is a DOMAIN action: a failure in it is a VALUE, not a
                # stream error. Guard it per-event so one bad reaction can never
                # tear the lambda down — the driver keeps observing.
                try:
                    self._emit_fire(rec, ev, note)
                except Exception as exc:  # noqa: BLE001
                    rec.last_event_kind = f"{ev.kind} (reaction-error: {type(exc).__name__})"
                if max_fires is not None and rec.fire_count >= max_fires:
                    break  # completion-shape: bounded lambda closes itself
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a transport/stream-level drop
            # Reserved for the stream itself failing (not a reaction): record it
            # as the close reason and tear down cleanly via the finally.
            note = f"{note} | driver-error: {type(exc).__name__}: {exc}".strip(" |")
        finally:
            self._finalize(rec, sub, note)

    def _emit_fire(self, rec: LambdaRecord, ev: Event, note: str) -> None:
        """Emit the live-channel fire meta-event and (if advisory) the governed
        observation — both on the meta-plane, idempotency-keyed, NEVER mutating."""
        # 1) live counter update for the UI lambda-tab.
        self.meta_plane.publish_nowait(
            META_LAMBDA_FIRED,
            {
                "sub_id": rec.sub_id,
                "label": rec.label,
                "source_seq": ev.seq,
                "source_kind": ev.kind,
                "fire_count": rec.fire_count,
                "seen_count": rec.seen_count,
            },
            source=f"lambda:{rec.sub_id}",
        )
        # 2) the agent's REACTION — advisory only, with an idempotency key. This is
        #    the sole sanctioned write-on-emit (governed effect): it never mutates
        #    state, only publishes a derived observation other agents may react to.
        if rec.reaction == "advisory":
            self.meta_plane.publish_nowait(
                META_LAMBDA_OBSERVATION,
                {
                    "sub_id": rec.sub_id,
                    "label": rec.label,
                    "idempotency_key": f"{rec.sub_id}:{ev.seq}",
                    "observed_kind": ev.kind,
                    "source_seq": ev.seq,
                    "fire_count": rec.fire_count,
                    "note": note or f"{rec.label} observed {ev.kind}",
                },
                source=f"lambda:{rec.sub_id}",
            )

    def _finalize(self, rec: LambdaRecord, sub: Any, note: str) -> None:
        """Idempotent teardown of one lambda — close sub, mark record, emit meta."""
        if rec.status == "closed":
            return
        try:
            sub.close()
        except Exception:  # noqa: BLE001 — close is best-effort + idempotent
            pass
        rec.status = "closed"
        rec.closed_seq = self.plane._seq
        self.meta_plane.publish_nowait(
            META_LAMBDA_CLOSED,
            {"sub_id": rec.sub_id, "label": rec.label, "reason": note or "completed",
             "fire_count": rec.fire_count, "seen_count": rec.seen_count},
            source=f"lambda:{rec.sub_id}",
        )

    # -- compose ---------------------------------------------------------- #
    def compose(
        self,
        sub_ids: Sequence[str],
        *,
        label: str = "",
        reducer: str = "each",
        reaction: str = "advisory",
        owner: Optional[Mapping[str, Any]] = None,
        max_fires: Optional[int] = None,
        source_plane: Optional[EventPlane] = None,
    ) -> LambdaRecord:
        """Compose existing lambdas into a NEW lambda over the UNION of their kinds.

        The merge combinator made concrete: the agent fuses several lambdas it
        created into a single higher-order lambda (e.g. "any node lifecycle change
        OR any failing tool"), reduced as one stream. Records ``composed_from`` so
        the read-only surface shows the lineage. ``source_plane`` is threaded
        through so a composed lambda observes the SAME run plane as its parts."""
        missing = [s for s in sub_ids if s not in self._records]
        if missing:
            raise LambdaInputError(f"cannot compose unknown lambda(s): {missing}")
        union: list[str] = []
        for s in sub_ids:
            for k in self._records[s].kinds:
                if k not in union:
                    union.append(k)
        return self.create(
            union,
            label=label or ("compose(" + "+".join(sub_ids) + ")"),
            reducer=reducer,
            reaction=reaction,
            owner=owner,
            max_fires=max_fires,
            composed_from=tuple(sub_ids),
            note=f"composed from {list(sub_ids)}",
            source_plane=source_plane,
        )

    # -- read-only surface ------------------------------------------------ #
    def snapshot(self, *, include_closed: bool = True) -> list[dict[str, Any]]:
        """The READ-ONLY live-subscriptions query the UI lambda-tab consumes.

        Returns the observe-only view of every agent-created lambda (id, what it
        observes, owner, status, live counters), newest first. Pure read — no
        mutation, no authoring path. This is the user's ONLY interaction (d15)."""
        recs = self._records.values()
        if not include_closed:
            recs = [r for r in recs if r.status == "active"]
        # Newest first. ``sub_id`` is a zero-padded monotonic id, so it orders by
        # creation even when several lambdas were created before any plane event
        # advanced ``created_seq`` (which can tie at 0).
        return [r.as_view() for r in sorted(recs, key=lambda r: r.sub_id, reverse=True)]

    def get(self, sub_id: str) -> Optional[LambdaRecord]:
        return self._records.get(sub_id)

    @property
    def active_count(self) -> int:
        return sum(1 for r in self._records.values() if r.status == "active")

    # -- close ------------------------------------------------------------ #
    async def close(self, sub_id: str) -> bool:
        """Close ONE lambda: cancel + await its driver (clean teardown, no orphan)."""
        rec = self._records.get(sub_id)
        if rec is None:
            return False
        task = self._drivers.get(sub_id)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        # The driver's ``finally`` finalises; ensure it ran even if never started.
        if rec.status != "closed":
            self._finalize(rec, self._subs.get(sub_id), "closed by request")
        return True

    async def close_all(self) -> None:
        """Tear down every live lambda — the shutdown path (no orphaned tasks)."""
        for sub_id in list(self._drivers):
            await self.close(sub_id)


__all__ = [
    "LambdaRegistry",
    "LambdaRecord",
    "LambdaInputError",
    "build_reducer",
    "DATA_PLANE_KINDS",
    "META_LAMBDA_REGISTERED",
    "META_LAMBDA_FIRED",
    "META_LAMBDA_CLOSED",
    "META_LAMBDA_OBSERVATION",
]
