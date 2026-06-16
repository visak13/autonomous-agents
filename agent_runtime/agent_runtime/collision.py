"""DAG spec COLLISION detection + HITL escalation (s3/Stage-B a5, d11).

a4 lets one DAG node carry N specs whose ruleset bodies are LAYERED into the
produce SYSTEM (compatible composition, applied autonomously). a5 closes the
other half of d11: *when two specs on a node GENUINELY CONFLICT, do not silently
pick one — PAUSE the node and ask the user which spec wins (HITL only on a real
conflict, never routinely).* This module is the standalone machinery for that:

1. A **deterministic, declarable conflict model.** A spec declares a *shaping
   directive* by embedding an explicit tag in its ruleset body::

       {{directive:axis=value}}

   ``axis`` names a shaping dimension (e.g. ``length``, ``format``, ``tone``) and
   ``value`` the stance this spec takes on it (e.g. ``verbose`` / ``terse``).
   Detection is a pure regex parse — NOT a fragile free-text NLP guess. Two specs
   on a node **collide** iff they declare directives on the SAME axis with
   DIFFERENT values. Same axis + same value (they agree) or different axes (they
   layer cleanly) is COMPATIBLE — composed autonomously, no escalation. A spec
   that declares no directive can never be the source of a declared conflict.

2. An **awaitable HITL resolution channel** (:class:`CollisionGate`) that mirrors
   the project's existing awaitable-approver pattern
   (``chat_app.approval.HttpApprovalGate``): the resolver injected into the
   runtime parks an :class:`asyncio.Future` keyed by the collision's deterministic
   ``challenge`` and returns it UN-resolved — so the node coroutine SUSPENDS at
   its ``await`` (the node stays in-flight / RUNNING; it is NOT failed and the
   run-engine lifecycle is intact). The future is resolved ONLY by a real
   out-of-band :meth:`CollisionGate.resolve` (the user's pick: which spec wins /
   the composition order). There is no in-process path that resolves it without
   that explicit decision — the "wait for the user" is structural, not a flag.

Kept STANDALONE in ``agent_runtime`` (it operates on the already-resolved
:class:`~agent_runtime.scope.ScopedSpec` bodies the runtime owns) so there is no
``chat_app`` → runtime coupling; an HTTP/UI surface can drive
:class:`CollisionGate` exactly as the FastAPI routes drive ``HttpApprovalGate``.
In-process, asyncio-native (d2/d10).
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Sequence

from .scope import ScopedSpec

# The declarable directive tag embedded in a ruleset body. ``axis`` and ``value``
# are short ``[A-Za-z0-9_-]`` tokens; whitespace around the parts is tolerated so a
# spec author can write ``{{directive: length = terse}}``. Deterministic + regex —
# never an NLP read of the prose.
_DIRECTIVE_RE = re.compile(
    r"\{\{\s*directive\s*:\s*([A-Za-z0-9_-]+)\s*=\s*([A-Za-z0-9_-]+)\s*\}\}"
)


class CollisionGateError(RuntimeError):
    """A resolution request that cannot be honored — no such pending collision, an
    already-decided one, or a resolution that names an unknown spec. Surfaced to an
    HTTP layer as a 404/409/422."""


class CollisionResolutionError(ValueError):
    """A :class:`CollisionResolution` cannot be applied to the node's scopes (it
    selected no known spec)."""


class CollisionUnresolved(RuntimeError):
    """A genuine collision was detected but no resolution channel is wired.

    Raised by the runtime so a real conflict surfaces as a clean, visible node
    FAILURE — never a silent auto-pick of one conflicting spec over the other."""

    def __init__(self, node_id: str, collision: "Collision") -> None:
        self.node_id = node_id
        self.collision = collision
        axes = ", ".join(a.axis for a in collision.axes)
        super().__init__(
            f"node {node_id!r} has a genuine spec collision on axis/axes [{axes}] "
            f"but no collision_resolver is wired to escalate it (d11)"
        )


# --------------------------------------------------------------------------- #
# The deterministic conflict model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Directive:
    """One declared shaping directive: a stance (``value``) on a shaping ``axis``."""

    axis: str
    value: str


def parse_directives(body: str) -> tuple[Directive, ...]:
    """Parse every ``{{directive:axis=value}}`` tag from a ruleset body.

    Returns the directives in first-seen order, de-duplicated (a spec repeating
    the same ``axis=value`` declares it once). ``axis``/``value`` are lower-cased
    so declarations are case-insensitive. A body with no tag yields ``()``."""
    out: list[Directive] = []
    seen: set[tuple[str, str]] = set()
    for m in _DIRECTIVE_RE.finditer(body or ""):
        d = Directive(axis=m.group(1).lower(), value=m.group(2).lower())
        key = (d.axis, d.value)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return tuple(out)


def strip_directives(body: str) -> str:
    """Remove directive tags so the model sees clean shaping prose, not metadata.

    A body that carries NO directive tag is returned BYTE-FOR-BYTE unchanged (so
    the a4 single-spec / no-tag composition is exactly preserved). When a tag is
    present it is removed and the small whitespace it leaves (trailing spaces, a
    run of blank lines) is tidied; the result is stripped of leading/trailing
    whitespace."""
    if not _DIRECTIVE_RE.search(body or ""):
        return body
    cleaned = _DIRECTIVE_RE.sub("", body)
    cleaned = re.sub(r"[ \t]+(\n)", r"\1", cleaned)  # trailing space before newline
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)      # double space the tag left mid-line
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)      # collapse blank-line runs
    return cleaned.strip()


@dataclass(frozen=True)
class ConflictAxis:
    """A single shaping axis on which 2+ specs declared DIFFERENT values.

    ``options`` is the ordered ``(spec_name, value)`` list the user chooses between
    (e.g. ``[("verbose-writer","verbose"), ("terse-editor","terse")]``)."""

    axis: str
    options: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Collision:
    """A genuine spec collision on one node — the escalation payload (d11).

    ``spec_names`` is every spec on the node in compose order; ``axes`` is the set
    of conflicting axes; ``challenge`` is a deterministic id that keys the awaitable
    resolution (so an out-of-band ``resolve`` names exactly THIS collision)."""

    node_id: str
    spec_names: tuple[str, ...]
    axes: tuple[ConflictAxis, ...]
    challenge: str

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "spec_names": list(self.spec_names),
            "challenge": self.challenge,
            "axes": [
                {"axis": a.axis, "options": [list(o) for o in a.options]}
                for a in self.axes
            ],
        }


def _challenge(node_id: str, spec_names: Sequence[str], axes: Sequence[ConflictAxis]) -> str:
    """A deterministic id for a collision (same inputs → same challenge).

    Hash of the node id + the ordered spec names + the conflicting axes/options, so
    the escalation key is stable and reproducible (no clock / randomness)."""
    canon = "|".join(
        [node_id, ",".join(spec_names)]
        + [f"{a.axis}={'/'.join(f'{s}:{v}' for s, v in a.options)}" for a in axes]
    )
    digest = hashlib.sha1(canon.encode("utf-8")).hexdigest()[:16]
    return f"collision-{node_id}-{digest}"


def detect_collision(node_id: str, scopes: Sequence[ScopedSpec]) -> Optional[Collision]:
    """Detect a GENUINE conflict among a node's resolved spec bodies (d11).

    Returns a :class:`Collision` iff two or more specs declare directives on the
    SAME axis with DIFFERENT values; otherwise ``None`` (compatible layering, to be
    composed autonomously). Fewer than two specs can never collide. Deterministic:
    same scopes → same verdict + same challenge."""
    if len(scopes) < 2:
        return None
    per_axis: dict[str, list[tuple[str, str]]] = {}
    for s in scopes:
        for d in parse_directives(s.body):
            per_axis.setdefault(d.axis, []).append((s.name, d.value))
    conflict_axes: list[ConflictAxis] = []
    for axis in sorted(per_axis):
        pairs = per_axis[axis]
        if len({v for _, v in pairs}) >= 2:  # 2+ DISTINCT values on one axis
            conflict_axes.append(ConflictAxis(axis=axis, options=tuple(pairs)))
    if not conflict_axes:
        return None
    spec_names = tuple(s.name for s in scopes)
    return Collision(
        node_id=node_id,
        spec_names=spec_names,
        axes=tuple(conflict_axes),
        challenge=_challenge(node_id, spec_names, conflict_axes),
    )


@dataclass(frozen=True)
class CollisionResolution:
    """The user's decision: the ordered subset of spec names to compose.

    ``order`` resolves both shapes of decision d11 names — "which spec wins" (drop
    the loser by omitting it) and "the ordering" (reorder the kept specs). The
    runtime composes ONLY these specs, in this order."""

    order: tuple[str, ...]
    note: str = ""


def apply_resolution(
    scopes: Sequence[ScopedSpec], resolution: CollisionResolution
) -> list[ScopedSpec]:
    """Reorder/filter ``scopes`` to the resolution's chosen order (drops the rest).

    Raises :class:`CollisionResolutionError` if the resolution selects no spec the
    node actually carries."""
    by_name = {s.name: s for s in scopes}
    chosen = [by_name[n] for n in resolution.order if n in by_name]
    if not chosen:
        raise CollisionResolutionError(
            f"resolution {list(resolution.order)!r} selects no spec on the node "
            f"(carries {list(by_name)!r})"
        )
    return chosen


# A collision resolver: given a detected collision, return the user's resolution.
# Injected into the runtime (typically :meth:`CollisionGate.resolver`) so the
# runtime stays decoupled from how the decision is surfaced/awaited.
CollisionResolver = Callable[[Collision], Awaitable[CollisionResolution]]


# --------------------------------------------------------------------------- #
# The awaitable HITL resolution channel (mirrors HttpApprovalGate)
# --------------------------------------------------------------------------- #
@dataclass
class _PendingCollision:
    """One collision parked at the gate, awaiting a real out-of-band decision."""

    collision: Collision
    future: "asyncio.Future[CollisionResolution]"


class CollisionGate:
    """Adapts the runtime's awaitable resolver to a real decision surface (d11).

    Structurally identical to ``chat_app.approval.HttpApprovalGate``: one gate
    instance backs a surface (e.g. held on ``app.state``). The resolver injected
    into :class:`~agent_runtime.runtime.AgentRuntime` parks the collision (keyed by
    its ``challenge``) and BLOCKS on a fresh :class:`asyncio.Future`; while blocked
    the node coroutine is suspended at its ``await`` — so the node stays in-flight
    (RUNNING) and NOTHING composes until a real decision arrives. The future is
    resolved ONLY by :meth:`resolve` (the user's pick over the wire). All access is
    on the single event loop, guarded by an :class:`asyncio.Lock`.
    """

    def __init__(self) -> None:
        self._pending: dict[str, _PendingCollision] = {}
        self._lock = asyncio.Lock()

    # -- the injected RESOLVER — a genuine "wait for the user" awaitable -- #
    async def resolver(self, collision: Collision) -> CollisionResolution:
        """The resolver injected into ``AgentRuntime(collision_resolver=...)``.

        Parks ``collision`` (keyed by ``challenge``) and BLOCKS until a real
        :meth:`resolve` sets the result. While blocked the node's produce step is
        suspended at its ``await`` — the node is paused in-flight, not failed.
        Raises :class:`CollisionGateError` if a collision with the same challenge is
        already parked (a duplicate surface would orphan a future)."""
        loop = asyncio.get_running_loop()
        fut: "asyncio.Future[CollisionResolution]" = loop.create_future()
        async with self._lock:
            if collision.challenge in self._pending:
                raise CollisionGateError(
                    f"collision {collision.challenge!r} is already awaiting resolution"
                )
            self._pending[collision.challenge] = _PendingCollision(collision, fut)
        try:
            return await fut  # the genuine block — control returns to the loop here
        finally:
            async with self._lock:
                self._pending.pop(collision.challenge, None)

    # -- the decision surface (driven by an HTTP route / a test) --------- #
    async def pending(self) -> list[dict]:
        """Snapshot of collisions awaiting resolution — what ``GET /…/collisions``
        would return: each carries the ``challenge`` (the resolve key) + the
        conflicting axes/options the user chooses between."""
        async with self._lock:
            return [p.collision.as_dict() for p in self._pending.values()]

    async def has_pending(self, challenge: str) -> bool:
        async with self._lock:
            return challenge in self._pending

    async def resolve(
        self, challenge: str, *, order: Sequence[str], note: str = ""
    ) -> dict:
        """Resolve a parked collision with the user's pick (the HITL decision).

        ``order`` is the chosen ordered subset of the collision's spec names (which
        spec(s) win + their compose order). Validates it is non-empty and names only
        specs on THAT collision, builds the :class:`CollisionResolution`, and sets it
        as the future's result — un-blocking the suspended produce step. Raises
        :class:`CollisionGateError` for an unknown/decided challenge or an order that
        names a spec not on the collision."""
        async with self._lock:
            p = self._pending.get(challenge)
            if p is None:
                raise CollisionGateError(
                    f"no collision pending resolution for challenge {challenge!r}"
                )
            if p.future.done():  # pragma: no cover - guarded by the pop in resolver
                raise CollisionGateError(
                    f"collision {challenge!r} has already been resolved"
                )
            chosen = tuple(str(s) for s in order)
            if not chosen:
                raise CollisionGateError(
                    f"resolution of {challenge!r} must pick at least one spec"
                )
            unknown = [s for s in chosen if s not in p.collision.spec_names]
            if unknown:
                raise CollisionGateError(
                    f"resolution of {challenge!r} names spec(s) {unknown!r} not on "
                    f"the collision (carries {list(p.collision.spec_names)!r})"
                )
            resolution = CollisionResolution(order=chosen, note=note)
            p.future.set_result(resolution)
            return {
                "challenge": challenge,
                "node_id": p.collision.node_id,
                "order": list(chosen),
                "note": note,
            }

    async def cancel_all(self, reason: str = "gate shutdown") -> int:
        """Cancel every still-parked resolver (lifespan teardown).

        A collision awaiting resolution at shutdown would leave a node coroutine
        blocked forever; cancelling the future raises :class:`asyncio.CancelledError`
        into the awaiting produce step so the run unwinds cleanly. Returns the count
        cancelled."""
        async with self._lock:
            pendings = list(self._pending.values())
        n = 0
        for p in pendings:
            if not p.future.done():
                p.future.cancel()
                n += 1
        return n


__all__ = [
    "Directive",
    "parse_directives",
    "strip_directives",
    "ConflictAxis",
    "Collision",
    "detect_collision",
    "CollisionResolution",
    "apply_resolution",
    "CollisionResolver",
    "CollisionGate",
    "CollisionGateError",
    "CollisionResolutionError",
    "CollisionUnresolved",
]
