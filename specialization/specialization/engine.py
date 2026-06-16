"""The specialization ENGINE: orchestrates the define -> research -> (approve) ->
compile -> register lifecycle, with the d9 **HITL compile gate** as a genuinely
user-facing approval signal.

Two entry points, ONE gate
--------------------------
- :meth:`SpecializationEngine.ui_specialize` — the **UI / HITL path**: define ->
  research -> author a condensed DRAFT -> **wait for explicit user approval** ->
  ONLY THEN compile + register. The "wait" is a real awaitable: the engine awaits
  an injected *user-facing approver* and surfaces the draft to it.
- :meth:`SpecializationEngine.autonomous_specialize` — the **AUTONOMOUS path**:
  define -> research -> AUTHOR the draft with no UI/human in the loop (used later
  by s6/s8 to satisfy o5b). Its COMPILE STILL routes through the SAME user-facing
  approval gate — there is deliberately **no auto-approve bypass**.

The approval gate (why it is not a flippable boolean)
-----------------------------------------------------
The HITL gate (d9) must represent a REAL human decision surfaced via the UI, not
an internal flag a caller flips. So approval is an **injectable, awaitable
user-decision**:

    Approver = async (SpecDraft) -> ApprovalToken

The engine NEVER mints approval itself. To compile it MUST ``await`` the injected
approver — handing it the draft to surface — and the approver returns an
:class:`ApprovalToken`. Two structural guarantees:

1. **Unreachable without an approver.** No approver injected → :func:`compile`
   raises :class:`ApprovalRequired`. Compile is impossible without the
   user-facing surface; there is no code path that compiles unprompted.
2. **Bound to THE surfaced draft.** Each draft carries a unique ``challenge``
   derived from its content; a valid token must echo that challenge AND say
   ``approved=True``. A stale/forged/denied token raises :class:`ApprovalDenied`.
   So approval is a decision ABOUT this specific surfaced draft, not a blanket
   boolean.

This makes the same gate usable at the s8 demo: the agent surfaces
autonomously-researched markdown+html drafts (see :meth:`SpecDraft.to_markdown`
/ :meth:`SpecDraft.to_html`) to the REAL user, who approves per d9.

No learnings flowback in either path (d8). In-process throughout (d2).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional, Union

from llm_framework import Transport

from specialization import compiler
from specialization.model import CompiledSpec, RawDefinition
from specialization.registry import SpecRegistry
from specialization.research import ResearchTrace, ToolInvoker, persist_trace, research

# Source tags (mirror model.SOURCES) — which lifecycle path authored the draft.
SOURCE_UI = "ui"
SOURCE_AUTONOMOUS = "autonomous"


# --------------------------------------------------------------------------- #
# Approval signal — the user-facing decision
# --------------------------------------------------------------------------- #
class ApprovalRequired(RuntimeError):
    """Raised when a compile is attempted with NO user-facing approver injected.

    This is the structural guarantee that compile is unreachable without the
    real approval surface (d9): there is no path to a registered spec that did
    not pass through an injected approver."""


class ApprovalDenied(RuntimeError):
    """Raised when the user-facing approver declined (or returned an invalid /
    stale token that does not match the surfaced draft)."""


@dataclass(frozen=True)
class ApprovalToken:
    """The decision an :data:`Approver` returns for a surfaced draft.

    ``challenge`` must echo the draft's :attr:`SpecDraft.challenge` — so the
    token is a decision about THE specific draft that was surfaced, not a
    detached boolean. ``approved`` is the human's yes/no."""

    challenge: str
    approved: bool

    @classmethod
    def grant(cls, draft: "SpecDraft") -> "ApprovalToken":
        """Helper a user-facing surface uses to APPROVE the given draft."""
        return cls(challenge=draft.challenge, approved=True)

    @classmethod
    def deny(cls, draft: "SpecDraft") -> "ApprovalToken":
        """Helper a user-facing surface uses to DECLINE the given draft."""
        return cls(challenge=draft.challenge, approved=False)


@dataclass(frozen=True)
class SpecDraft:
    """A condensed DRAFT surfaced for approval (post-research, pre-compile).

    Carries everything the user-facing surface needs to make a real decision: the
    definition, the research trace it was distilled from, the condensed ``body``
    the sub-agent would load, and the ``source`` path that authored it. The
    ``challenge`` binds an approval to exactly this draft."""

    raw: RawDefinition
    trace: ResearchTrace
    body: str
    source: str
    challenge: str = ""

    def __post_init__(self) -> None:
        if not self.challenge:
            object.__setattr__(self, "challenge", _draft_challenge(self.raw, self.body))

    # ---- preview surfaces for the user-facing approval UI (d9 / s8) ---- #
    def to_markdown(self) -> str:
        """The markdown the user reviews — the exact compiled doc they'd approve."""
        return compiler.compile_spec(
            self.raw, self.body, source=self.source,
            trace_ref=_trace_ref(self.raw),
        ).to_markdown()

    def to_html(self) -> str:
        """A minimal HTML preview of the draft for the approval surface (s8).

        Dependency-free (no markdown lib) — a ``<pre>`` wrap is enough for the
        user to read the proposed ruleset and decide; the demo's real UI styles
        it. Kept here so the draft is self-describing to any surface."""
        safe = (
            self.to_markdown()
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        return (
            f"<section class='spec-draft' data-source='{self.source}'>"
            f"<h2>Approve specialist: {self.raw.name}</h2>"
            f"<p>{self.raw.description}</p>"
            f"<pre class='spec-body'>{safe}</pre>"
            "</section>"
        )


def _draft_challenge(raw: RawDefinition, body: str) -> str:
    """A stable per-draft challenge derived from the definition + condensed body.

    Deterministic (content hash) so a test can reproduce it, and bound to the
    body so that re-authoring a different draft yields a different challenge — an
    approval can never be silently reused across drafts."""
    h = hashlib.sha256()
    h.update(raw.name.encode("utf-8"))
    h.update(b"\x00")
    h.update(body.encode("utf-8"))
    return h.hexdigest()[:16]


def _trace_ref(raw: RawDefinition) -> str:
    """The provenance pointer stamped into a compiled spec (where the trace lives)."""
    return f"specs/{raw.name}/research_trace.json"


# An injected user-facing approver: surfaces a draft, returns the user decision.
# Async because the d9 gate is a real awaitable "wait for the user". A plain
# (sync) callable returning a token is also accepted for convenience.
Approver = Callable[[SpecDraft], Union[ApprovalToken, Awaitable[ApprovalToken]]]


async def _await_decision(approver: Approver, draft: SpecDraft) -> ApprovalToken:
    """Invoke the approver and await its decision (sync or async approver)."""
    result = approver(draft)
    if hasattr(result, "__await__"):
        result = await result  # type: ignore[assignment]
    return result  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class SpecializationEngine:
    """Orchestrates the specialization lifecycle around the d9 HITL gate.

    Parameters
    ----------
    registry:
        The :class:`SpecRegistry` compiled specs are registered into (the
        compile-on-approval write, d8).
    hook:
        The in-process research :class:`ToolInvoker` (d2). Injected so the engine
        can be driven fully offline against a mock in tests.
    condense_transport:
        Optional transport for the condense chain. ``None`` (default) → OFFLINE
        deterministic condense (d7). Inject an ``OllamaTransport`` for the
        DEFERRED live phi authoring.
    specs_dir:
        Where research traces are persisted (replayable provenance). Defaults to
        the registry's directory so a spec and its trace sit together.
    """

    def __init__(
        self,
        registry: SpecRegistry,
        *,
        hook: ToolInvoker,
        condense_transport: Optional[Transport] = None,
        specs_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self._registry = registry
        self._hook = hook
        self._condense_transport = condense_transport
        self._specs_dir = Path(specs_dir) if specs_dir is not None else registry.specs_dir

    # ---- step: research + author the condensed DRAFT (no compile/register) ---- #
    async def author_draft(self, raw: RawDefinition, *, source: str) -> SpecDraft:
        """Define -> research -> author a condensed DRAFT. Does NOT compile.

        Runs the bounded web-research loop over the in-process hook, persists the
        replayable trace, then condenses a draft body through the chain
        (offline by default). The returned :class:`SpecDraft` is what a
        user-facing surface displays for approval — nothing has been compiled or
        registered yet."""
        if source not in (SOURCE_UI, SOURCE_AUTONOMOUS):
            raise ValueError(f"source must be 'ui' or 'autonomous', got {source!r}")
        skill = raw.description.strip() or raw.name
        trace = await research(skill, raw.intent, hook=self._hook)
        # Persist under the spec NAME slug so the on-disk trace lands exactly
        # where the compiled spec's research_trace_ref points (_trace_ref uses
        # raw.name). trace.skill is the research subject (the description) and
        # drives the queries — it is intentionally NOT the directory key.
        persist_trace(trace, self._specs_dir, subdir=raw.name)
        body = compiler.condense_body(raw, trace, transport=self._condense_transport)
        return SpecDraft(raw=raw, trace=trace, body=body, source=source)

    # ---- the GATE: compile a draft ONLY through the user-facing approver ---- #
    async def compile(self, draft: SpecDraft, *, approver: Optional[Approver]) -> CompiledSpec:
        """Compile + register a draft — but ONLY through a user-facing approval.

        Structurally:
        - ``approver is None`` → :class:`ApprovalRequired` (compile is
          unreachable without the user-facing surface — the d9 guarantee).
        - the approver is awaited with the draft surfaced to it; if it returns a
          token that does not match THIS draft's challenge or says
          ``approved=False`` → :class:`ApprovalDenied`.
        - only a matching, approving token reaches the compile + register write.

        Both lifecycle paths funnel through here; there is no second, ungated
        compile path."""
        if approver is None:
            raise ApprovalRequired(
                f"cannot compile {draft.raw.name!r}: no user-facing approver injected "
                "(the d9 HITL gate makes compile unreachable without one)"
            )
        token = await _await_decision(approver, draft)
        if not isinstance(token, ApprovalToken):
            raise ApprovalDenied(
                f"approver for {draft.raw.name!r} returned {type(token).__name__}, "
                "not an ApprovalToken"
            )
        if token.challenge != draft.challenge:
            raise ApprovalDenied(
                f"approval token does not match the surfaced draft for "
                f"{draft.raw.name!r} (challenge mismatch — stale or forged token)"
            )
        if not token.approved:
            raise ApprovalDenied(f"user declined to compile {draft.raw.name!r}")

        spec = compiler.compile_spec(
            draft.raw, draft.body, source=draft.source, trace_ref=_trace_ref(draft.raw)
        )
        self._registry.register(spec)
        return spec

    # ---- entry point (a): the UI / HITL path ---- #
    async def ui_specialize(self, raw: RawDefinition, *, approver: Optional[Approver]) -> CompiledSpec:
        """UI path: define -> research -> DRAFT -> wait for user approval ->
        compile + register. The "wait" is the awaited injected approver; without
        an approving decision nothing is compiled (d9)."""
        draft = await self.author_draft(raw, source=SOURCE_UI)
        return await self.compile(draft, approver=approver)

    # ---- entry point (b): the AUTONOMOUS path ---- #
    async def autonomous_specialize(
        self, raw: RawDefinition, *, approver: Optional[Approver]
    ) -> CompiledSpec:
        """Autonomous path: define -> research -> AUTHOR the draft with no human
        in the loop. The COMPILE still routes through the SAME user-facing gate —
        no auto-approve bypass (d9). For s6/s8: the agent researches+authors
        autonomously, then surfaces the draft to the REAL user for approval."""
        draft = await self.author_draft(raw, source=SOURCE_AUTONOMOUS)
        return await self.compile(draft, approver=approver)


__all__ = [
    "SpecializationEngine",
    "SpecDraft",
    "ApprovalToken",
    "Approver",
    "ApprovalRequired",
    "ApprovalDenied",
    "SOURCE_UI",
    "SOURCE_AUTONOMOUS",
]
