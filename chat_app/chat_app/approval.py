"""The HTTP approval gate (s7/a3) — the #2 structural risk closed.

It adapts the specialization engine's *awaitable* approver to a REAL HTTP
surface, so the d9 HITL compile gate is unreachable without a genuine user
click that arrives as an HTTP request.

The engine's gate (see :mod:`specialization.engine`) compiles a draft ONLY
through an injected approver::

    Approver = async (SpecDraft) -> ApprovalToken

s5's UI minted that token *synchronously* inside the ``/approve`` handler
(define -> research -> a SECOND request approves the already-held draft). This
gate is stronger: the approver injected into ``engine.compile`` is a genuine
**"wait for the user" awaitable**. When the engine surfaces a draft to
:meth:`HttpApprovalGate.approver`, the gate parks an :class:`asyncio.Future`
keyed by the draft's ``challenge`` and returns it UN-resolved — so
``engine.compile`` SUSPENDS at its ``await`` (the spec is *not* compiled, the
coroutine is genuinely blocked). The future is resolved ONLY by a real HTTP
``POST /specializations/{challenge}/approve`` (grant) or ``/deny`` (deny).
There is no in-process code path that resolves it without an HTTP request — the
"wait" is structural, not a flag a caller can flip.

Two guarantees this preserves (both proven end-to-end in the a3 evidence):

1. **Unreachable without the surface.** No approver injected → ``compile``
   raises :class:`~specialization.engine.ApprovalRequired` and nothing is
   registered (the engine's own structural gate; this module never weakens it).
2. **Bound to THE surfaced draft, resolved only by a real request.** The future
   is keyed by ``draft.challenge``; an approve/deny request must name that exact
   challenge, and the granted token echoes it — so an approval is a decision
   about this specific surfaced draft delivered over the wire, never a blanket
   boolean.

In-process, asyncio-native (d2). The gate holds only ``asyncio`` primitives and
the live drafts; it owns no I/O of its own — the HTTP surface is the FastAPI app
that drives it.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from specialization.engine import ApprovalToken, SpecDraft


class ApprovalGateError(RuntimeError):
    """A decision request that cannot be honored (no such pending draft, or a
    draft already decided). Surfaced to the HTTP layer as a 404/409."""


@dataclass
class _Pending:
    """One draft parked at the gate, awaiting a real HTTP decision."""

    draft: SpecDraft
    future: "asyncio.Future[ApprovalToken]"


class HttpApprovalGate:
    """Adapts the engine's awaitable Approver to a real HTTP decision surface.

    One gate instance backs the whole app (held on ``app.state``). It is the
    seam between ``engine.compile`` (which ``await``\\ s the approver) and the
    FastAPI approve/deny routes (which resolve the awaited future). All access is
    on the single event loop, guarded by an :class:`asyncio.Lock` so a decision
    request and the parked approver never race on the pending map.
    """

    def __init__(self) -> None:
        self._pending: dict[str, _Pending] = {}
        self._lock = asyncio.Lock()

    # ----------------------------------------------------------------- #
    # The injected APPROVER — a genuine "wait for the user" awaitable.
    # ----------------------------------------------------------------- #
    async def approver(self, draft: SpecDraft) -> ApprovalToken:
        """The approver injected into ``engine.compile`` / ``ui_specialize`` /
        ``autonomous_specialize``.

        Parks ``draft`` as pending (keyed by its ``challenge``) and BLOCKS on a
        fresh :class:`asyncio.Future` until a real HTTP approve/deny request
        resolves it. While blocked, ``engine.compile`` is suspended at its
        ``await`` — so NOTHING is compiled or registered until the request
        arrives. This is the structural "wait for the user", not a flipped flag.

        Raises :class:`ApprovalGateError` if a draft with the same challenge is
        already parked (a duplicate surface would otherwise orphan a future)."""
        loop = asyncio.get_running_loop()
        fut: "asyncio.Future[ApprovalToken]" = loop.create_future()
        async with self._lock:
            if draft.challenge in self._pending:
                raise ApprovalGateError(
                    f"draft {draft.challenge!r} is already awaiting approval"
                )
            self._pending[draft.challenge] = _Pending(draft=draft, future=fut)
        try:
            # The genuine block: control returns to the event loop here and the
            # engine's compile() stays suspended until decide() sets this result.
            return await fut
        finally:
            # Whether approved, denied, or cancelled, the draft is no longer
            # pending — drop it so the pending map reflects only live waits.
            async with self._lock:
                self._pending.pop(draft.challenge, None)

    # ----------------------------------------------------------------- #
    # The HTTP-facing surface (driven by the FastAPI routes).
    # ----------------------------------------------------------------- #
    async def pending(self) -> list[dict]:
        """Snapshot of drafts awaiting approval — what ``GET /…/pending`` returns.

        Each entry carries the draft's ``challenge`` (the approve/deny key) and
        its ``to_html()`` preview (the exact ruleset the user reviews), plus the
        lookup text. Body-free of the registry concern — this is the *pending*
        surface, not the planner index."""
        async with self._lock:
            return [
                {
                    "challenge": challenge,
                    "name": p.draft.raw.name,
                    "description": p.draft.raw.description,
                    "source": p.draft.source,
                    "html": p.draft.to_html(),
                }
                for challenge, p in self._pending.items()
            ]

    async def has_pending(self, challenge: str) -> bool:
        async with self._lock:
            return challenge in self._pending

    async def decide(self, challenge: str, *, approved: bool) -> dict:
        """Resolve a parked approver with a real decision (the HTTP click).

        Grants (``approved=True``) or denies (``approved=False``) the pending
        draft named by ``challenge``: builds the matching :class:`ApprovalToken`
        for THAT exact draft and sets it as the future's result, un-blocking the
        suspended ``engine.compile``. Raises :class:`ApprovalGateError` if no
        draft with that challenge is pending (stale/forged id) or it was already
        decided. Returns a small JSON-able receipt."""
        async with self._lock:
            p = self._pending.get(challenge)
            if p is None:
                raise ApprovalGateError(
                    f"no draft pending approval for challenge {challenge!r}"
                )
            if p.future.done():  # pragma: no cover - guarded by the pop in approver
                raise ApprovalGateError(
                    f"draft {challenge!r} has already been decided"
                )
            token = (
                ApprovalToken.grant(p.draft)
                if approved
                else ApprovalToken.deny(p.draft)
            )
            # The token echoes draft.challenge — the engine re-checks it binds to
            # THE surfaced draft. set_result un-blocks the awaiting approver.
            p.future.set_result(token)
            return {
                "challenge": challenge,
                "name": p.draft.raw.name,
                "approved": approved,
                "token_challenge": token.challenge,
            }

    async def cancel_all(self, reason: str = "gate shutdown") -> int:
        """Cancel every still-parked approver (lifespan teardown).

        A draft awaiting approval when the app shuts down would otherwise leave a
        coroutine blocked forever; cancelling the future raises
        :class:`asyncio.CancelledError` into the awaiting ``engine.compile`` so
        the specialization task unwinds cleanly. Returns how many were cancelled.
        """
        async with self._lock:
            pendings = list(self._pending.values())
        n = 0
        for p in pendings:
            if not p.future.done():
                p.future.cancel()
                n += 1
        return n


__all__ = ["HttpApprovalGate", "ApprovalGateError"]
