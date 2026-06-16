"""Per-node status — the DAG executor's explicit state machine (Stage B).

Stage A tracked a node's progress implicitly across several sets (``remaining`` /
``running`` / ``done`` / ``failed``). A production executor needs ONE explicit
per-node state with a legal transition graph so completion, aggregation, and
cancellation are unambiguous and observable. That is :class:`NodeStatus` +
:class:`NodeState`.

The lifecycle borrows eda-base3's action FSM (CONCEPT only — standalone, no
coupling): a node moves ``pending → in-progress → verifiable → done`` with a
per-node VERIFY GATE on the ``verifiable → done`` edge (Stage-B run engine). The
historical status value for the in-progress phase is ``"running"`` (kept for the
shipped a1/a2 UI + smoke consumers); ``running`` IS the eda-base3 "in-progress"
phase. ``VERIFIABLE`` is the new gate phase: a node reaches it the instant its
produce step finishes, and only crosses to ``DONE`` once the gate passes (the
CODER=REVIEWER inline-fix may correct the output first — see the runtime). The
lifecycle (start → track → verify → await → cancel) maps onto these transitions::

    PENDING ──launch──▶ RUNNING ──produced──▶ VERIFIABLE ──gate passes──▶ DONE
        │                  │                       │  └──gate fails (inline-fix
        │                  │                       │      exhausted)──────▶ FAILED
        │                  │  └──fail (heal exhausted)──────────────────▶ FAILED
        │                  └──cancel──────────────────────────────────▶ CANCELLED
        └──upstream failed/blocked────────────────────────────────────▶ SKIPPED
    FAILED ──re-plan recovered──▶ DONE     (sub-graph self-heal)

(``RUNNING → DONE`` stays legal as well: a re-plan-recovered node — whose
corrective sub-graph already passed its own per-node gates — is finalised
directly without re-entering the gate.)

Pure data + a small guarded ``transition`` — no asyncio, no model call (d10
lean). The executor owns one :class:`NodeState` per node and reads the map back
into the :class:`~agent_runtime.runtime.RuntimeResult`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class NodeStatus(str, Enum):
    """The state of a single DAG node within one runtime drive."""

    PENDING = "pending"        # not yet launched
    RUNNING = "running"        # launched as a tracked asyncio task, in flight (eda-base3 "in-progress")
    VERIFIABLE = "verifiable"  # produce step finished; awaiting / running the per-node verify gate
    DONE = "done"              # completed successfully (gate passed; possibly after self-heal / inline-fix)
    FAILED = "failed"        # exhausted self-heal and re-plan; surfaced
    SKIPPED = "skipped"      # an upstream dependency failed; never launched
    CANCELLED = "cancelled"  # in-flight task cancelled (timeout / explicit cancel)


# Legal transitions — the executor asserts against this so a bug can never park a
# node in an impossible state (e.g. DONE → RUNNING).
_LEGAL: dict[NodeStatus, frozenset[NodeStatus]] = {
    NodeStatus.PENDING: frozenset(
        {NodeStatus.RUNNING, NodeStatus.SKIPPED, NodeStatus.CANCELLED}
    ),
    # RUNNING → VERIFIABLE is the normal produce→gate edge. RUNNING → DONE stays
    # legal for the re-plan-recovery path (the corrective sub-graph already gated
    # its own nodes, so the recovered node is finalised without re-gating).
    NodeStatus.RUNNING: frozenset(
        {NodeStatus.VERIFIABLE, NodeStatus.DONE, NodeStatus.FAILED, NodeStatus.CANCELLED}
    ),
    # VERIFIABLE → DONE when the per-node gate passes (possibly after an inline
    # CODER=REVIEWER fix); → FAILED when the gate still fails after inline-fix +
    # re-plan are exhausted; → CANCELLED on a timeout that fires mid-gate.
    NodeStatus.VERIFIABLE: frozenset(
        {NodeStatus.DONE, NodeStatus.FAILED, NodeStatus.CANCELLED}
    ),
    # FAILED → DONE is the sub-graph re-plan recovery path; FAILED is otherwise terminal.
    NodeStatus.FAILED: frozenset({NodeStatus.DONE}),
    NodeStatus.DONE: frozenset(),
    NodeStatus.SKIPPED: frozenset(),
    NodeStatus.CANCELLED: frozenset(),
}


class IllegalTransition(RuntimeError):
    """An executor attempted a state change outside the legal lifecycle graph."""


@dataclass
class NodeState:
    """The observable state of one DAG node across a runtime drive.

    Attributes
    ----------
    node_id:
        The node this state tracks.
    status:
        Current :class:`NodeStatus`.
    attempts:
        Times the node's logic was actually EXECUTED (incremented per real run;
        a cache short-circuit does NOT increment it — that is the idempotency
        proof: a re-launched, already-succeeded node stays at its prior count).
    launch_seq:
        Monotonic launch order index (the Nth node launched), or ``-1`` if never
        launched. Drives the deterministic ``launch_order`` aggregation.
    cache_hit:
        True if this node's result was served from the idempotent result cache
        instead of being re-executed (no double-execution proof).
    healed:
        True if node-level self-heal recovered a failure.
    replanned:
        True if the node was recovered by a re-derived sub-graph re-plan.
    verified:
        True once the per-node verify gate accepted the node's output (the
        ``verifiable → done`` crossing).
    inline_fixes:
        Number of CODER=REVIEWER inline fixes applied at the verify gate (each is
        a same-spec review+correct that did NOT re-trigger the DAG loop).
    inline_fixed:
        True if an inline review fix is what carried the node past the gate.
    error:
        The surfaced error string if the node ended FAILED.
    """

    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    attempts: int = 0
    launch_seq: int = -1
    cache_hit: bool = False
    healed: bool = False
    replanned: bool = False
    verified: bool = False
    inline_fixes: int = 0
    inline_fixed: bool = False
    error: Optional[str] = None

    def transition(self, to: NodeStatus) -> None:
        """Move to ``to`` if the transition is legal, else raise."""
        if to not in _LEGAL[self.status]:
            raise IllegalTransition(
                f"node {self.node_id!r}: illegal {self.status.value} → {to.value}"
            )
        self.status = to

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "attempts": self.attempts,
            "launch_seq": self.launch_seq,
            "cache_hit": self.cache_hit,
            "healed": self.healed,
            "replanned": self.replanned,
            "verified": self.verified,
            "inline_fixes": self.inline_fixes,
            "inline_fixed": self.inline_fixed,
            "error": self.error,
        }


__all__ = ["NodeStatus", "NodeState", "IllegalTransition", "_LEGAL"]
