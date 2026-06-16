"""Conversational spec-authoring CORE — a USER-DRIVEN, multi-turn way to define
an OUTPUT-SHAPING RULESET interactively (d1/d2; the later s4 surface the seed
module deferred).

Where this sits in the lifecycle
--------------------------------
The :mod:`specialization.compiler` condense is a SINGLE-SHOT distillation of a
research trace into a body — it answers "given what research surfaced, what is
the ruleset?". This module is the OTHER authoring path the design calls for: a
back-and-forth where the USER drives the body into shape turn by turn
("define interactively in chat, back-and-forth until right"). The NEW seam is
:meth:`SpecConversation.refine` — it RE-AUTHORS the working body incorporating a
critique *against the PRIOR body*, which ``compiler.condense_body`` (single-shot,
trace-only) deliberately does not do.

What it authors (the d1 guard)
------------------------------
Every body this produces is an **OUTPUT-SHAPING RULESET**: a mission plus
concrete dos/don'ts that SHAPE THE FORM of a real task's output — never a
"how to <skill>" document (round-1's Iran->markdown-how-to bug). The guard is
carried in the system prompt of both the author and the refine chain, exactly
like :data:`specialization.seed.MARKDOWN_WRITER_RULESET` models it. A spec is
FLEXIBLE: the same conversation can define a pure formatting ruleset OR a fuller
workflow — only the body *content* differs, the mechanism here is one.

How the redraft runs (and why it is reproducible + GPU-free)
------------------------------------------------------------
The author/refine redraft is driven through the SAME ``llm_framework`` chain
spine the compiler uses (``build_default_chain`` -> ``prompt_assembly`` ->
``call_stage``), with the SAME swappable transport seam:

- ``transport=None`` (default) → a DETERMINISTIC offline
  :class:`~llm_framework.FakeTransport` scripted with a mechanical redraft
  (:func:`offline_author_body` / :func:`offline_refine_body`), so the whole
  conversation runs reproducibly with zero GPU (mirrors
  ``compiler.default_condense_transport``, including a non-empty fallback so a
  body is NEVER empty).
- inject an :class:`~llm_framework.OllamaTransport` (phi4-mini) → LIVE authoring.

Concurrency (d4): the redraft is a SYNCHRONOUS chain call (like the compiler's).
A caller that runs inside the asyncio event loop offloads it with
``asyncio.to_thread`` (the HTTP layer does exactly that) — this module never
blocks the loop itself, and never touches the broker/pool (in-process, d2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from llm_framework import (
    Context,
    FakeTransport,
    Transport,
    build_default_chain,
)

from specialization import compiler
from specialization.compiler import OFFLINE_MARKER
from specialization.model import CompiledSpec, RawDefinition
from specialization.registry import SpecRegistry

# Conversational authoring is a USER-driven define surface, so a body it
# compiles is tagged 'ui' (model.SOURCES) — the same origin as the engine's HITL
# UI path; this is the chat-authoring half of that surface.
SOURCE_UI = "ui"

# Conversation lifecycle states. ``open`` accepts start/refine/approve; the
# three terminal states block further authoring so a decided conversation can
# never be silently re-driven.
STATE_OPEN = "open"
STATE_APPROVED = "approved"
STATE_DENIED = "denied"
STATE_CANCELLED = "cancelled"
_TERMINAL = (STATE_APPROVED, STATE_DENIED, STATE_CANCELLED)


class ConversationError(RuntimeError):
    """Raised on an out-of-order call (e.g. refine before start, or any
    authoring call after the conversation has reached a terminal state)."""


# --------------------------------------------------------------------------- #
# Turn + preview value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Turn:
    """One recorded conversation turn (``role`` is ``"user"`` or ``"agent"``)."""

    role: str
    text: str


@dataclass(frozen=True)
class DraftPreview:
    """What :meth:`SpecConversation.start` / :meth:`refine` return for display.

    Carries the working body plus the planner-facing identity, and can render
    the EXACT compiled markdown-with-frontmatter doc the user would approve —
    so the UI surfaces a real preview, not a paraphrase."""

    name: str
    description: str
    body: str
    turn: int  # how many author/refine rounds have produced this body (1-based)

    def to_markdown(self) -> str:
        """The compiled doc the user reviews — identical to what approve() mints."""
        raw = RawDefinition(name=self.name, description=self.description, intent="")
        return compiler.compile_spec(raw, self.body, source=SOURCE_UI).to_markdown()


# --------------------------------------------------------------------------- #
# Prompt construction (what a LIVE phi sees; what the FakeTransport records)
# --------------------------------------------------------------------------- #
# The d1 guard, shared by both the author and the refine system prompt: the body
# is a ruleset that SHAPES a task's output, never a how-to about the skill.
_RULESET_GUARD = (
    "You are authoring an OUTPUT-SHAPING RULESET for a small autonomous agent. "
    "The body you write is the WHOLE grounding a sub-agent loads to SHAPE THE "
    "FORM of its answer to a REAL task: a mission line plus concrete dos and "
    "don'ts. The agent DOES the task first, THEN applies your rules. NEVER write "
    "a 'how to <skill>' tutorial or describe the skill itself — that is the bug "
    "this guard exists to prevent. Be lean (a small-context model loads this): "
    "no preamble, no restating the request. Output the ruleset body only, as "
    "raw markdown — do NOT wrap the whole body in a ``` code fence."
)

_AUTHOR_SYSTEM = _RULESET_GUARD

_REFINE_SYSTEM = (
    _RULESET_GUARD
    + " You are REVISING an existing ruleset. The user message contains the "
    "CURRENT ruleset body and a critique. Re-author the body so it incorporates "
    "the critique while preserving everything still correct. Return the full "
    "revised body, not a diff."
)


def build_author_messages(raw: RawDefinition, first_message: str) -> tuple[str, str]:
    """Build the ``(system, user)`` pair for the INITIAL author redraft."""
    lines = [
        f"SPECIALIST (ruleset name): {raw.name}",
        f"DESCRIPTION (planner-facing lookup): {raw.description}",
    ]
    intent = (first_message or raw.intent or "").strip()
    if intent:
        lines.append(f"WHAT THE USER WANTS THIS RULESET TO DO: {intent}")
    return _AUTHOR_SYSTEM, "\n".join(lines)


def build_refine_messages(
    raw: RawDefinition, prior_body: str, critique: str
) -> tuple[str, str]:
    """Build the ``(system, user)`` pair for a REFINE redraft.

    The user turn carries the PRIOR body AND the critique — this is the new seam:
    the redraft is conditioned on the existing body, not authored from scratch."""
    user = (
        f"RULESET NAME: {raw.name}\n"
        f"DESCRIPTION: {raw.description}\n\n"
        "CURRENT RULESET BODY:\n"
        f"{prior_body.strip()}\n\n"
        f"USER CRITIQUE TO INCORPORATE:\n{critique.strip()}"
    )
    return _REFINE_SYSTEM, user


# --------------------------------------------------------------------------- #
# Deterministic OFFLINE redrafts (the FakeTransport's scripted replies)
# --------------------------------------------------------------------------- #
def offline_author_body(raw: RawDefinition, first_message: str) -> str:
    """Deterministically author an initial output-shaping ruleset body (offline).

    Pure, no LLM, no network — the body the default offline
    :class:`FakeTransport` returns so the author chain has a real, deterministic
    reply to write to ``ctx.raw_output``. It is an honest OUTPUT-SHAPING ruleset
    (mission + dos/don'ts), never a skill how-to (d1)."""
    intent = (first_message or raw.intent or raw.description or raw.name).strip()
    parts = [
        f"# Output-shaping ruleset: {raw.name}",
        "",
        f"**Mission.** {intent} — do the task described in the user message using "
        "the inputs and tool findings provided there, then shape your answer to "
        "follow the rules below.",
        "",
        "## Rules",
        "- Produce the real result of the task; never explain or describe the "
        "skill instead of doing it.",
        "- Keep the output tight and well-structured — no preamble, no restating "
        "the task, no meta-commentary.",
        "- Lead with the key outcome, then the supporting detail.",
        "",
        OFFLINE_MARKER,
    ]
    return "\n".join(parts).strip()


def offline_refine_body(raw: RawDefinition, prior_body: str, critique: str) -> str:
    """Deterministically RE-AUTHOR ``prior_body`` to incorporate ``critique``.

    Mechanical, offline, and conditioned on the PRIOR body (the new seam): it
    folds the critique into a ``## Refinements`` section of the existing body.
    Guarantees, for the reproducible offline path, that (a) the body CHANGES and
    (b) the critique text appears in it — exactly what the live phi re-author
    would achieve, without a GPU."""
    critique = (critique or "").strip()
    # Re-author ON TOP of the prior body: strip the trailing offline marker,
    # append/extend the Refinements section, re-stamp the marker.
    body = prior_body.replace(OFFLINE_MARKER, "").rstrip()
    addition = f"- Apply this refinement: {critique}" if critique else "- (no-op refinement)"
    if "## Refinements" in body:
        body = f"{body}\n{addition}"
    else:
        body = f"{body}\n\n## Refinements\n{addition}"
    return f"{body}\n\n{OFFLINE_MARKER}".strip()


# --------------------------------------------------------------------------- #
# The conversation session
# --------------------------------------------------------------------------- #
class SpecConversation:
    """A stateful, USER-DRIVEN, multi-turn spec-authoring session (d1/d2).

    Per-session state:
      - the evolving :class:`RawDefinition` (name / description / intent),
      - the current working DRAFT body (the output-shaping ruleset so far),
      - the full turn history (``user`` / ``agent`` text).

    Lifecycle: :meth:`start` authors an initial body; :meth:`refine` re-authors
    it against a critique (repeatable); :meth:`approve` compiles + registers the
    body as a loadable :class:`CompiledSpec`; :meth:`deny` / :meth:`cancel` close
    the session without compiling. Everything is in-process (no broker/pool).

    Parameters
    ----------
    raw:
        The initial definition. ``name`` is the registry key; ``description`` is
        the planner-facing lookup text. ``intent`` may be empty and supplied as
        the ``first_message`` to :meth:`start` instead.
    registry:
        Where :meth:`approve` registers the compiled spec.
    transport:
        Optional chain transport. ``None`` (default) → deterministic offline
        redrafts (GPU-free, reproducible). Inject an ``OllamaTransport``
        (phi4-mini) for LIVE authoring.
    """

    def __init__(
        self,
        raw: RawDefinition,
        *,
        registry: SpecRegistry,
        transport: Optional[Transport] = None,
        source: str = SOURCE_UI,
        trace_ref: str = "",
    ) -> None:
        self._raw = raw
        self._registry = registry
        self._transport = transport
        # Provenance carried so a RE-OPENED spec (see :meth:`reopen`) re-compiles
        # under its ORIGINAL source/trace on approve instead of being silently
        # re-stamped 'ui'. A fresh authoring session defaults to 'ui' (the
        # chat-authoring origin), unchanged.
        self._source = source
        self._trace_ref = trace_ref
        self._body: str = ""
        self._history: List[Turn] = []
        self._rounds: int = 0          # author/refine rounds completed
        self._state: str = STATE_OPEN

    # -- re-open an EXISTING registered spec for editing (s4/RC7) ----------- #
    @classmethod
    def reopen(
        cls,
        spec: CompiledSpec,
        *,
        registry: SpecRegistry,
        transport: Optional[Transport] = None,
    ) -> "SpecConversation":
        """Open an editable conversation SEEDED from an already-registered spec.

        This is the conversational half of the re-editable surface (RC7): the gap
        was that the spec chat could only AUTHOR a new spec, never re-open an
        existing one to view + edit it. Unlike a fresh session (which must
        :meth:`start` to author draft 1), a reopened session begins ALREADY
        STARTED with the existing body as its working draft — so the very next
        user turn is a :meth:`refine` of the REAL persisted ruleset, and
        :meth:`approve` re-registers it under the SAME name (identity preserved)
        with its ORIGINAL source/provenance. The seeded history carries one
        ``agent`` turn holding the current body so the transcript shows exactly
        what is being edited."""
        raw = RawDefinition(name=spec.name, description=spec.description, intent="")
        conv = cls(
            raw,
            registry=registry,
            transport=transport,
            source=spec.source,
            trace_ref=spec.research_trace_ref,
        )
        conv._body = spec.body
        conv._rounds = 1  # already authored — next turn is a refine, not a start
        conv._history = [Turn(role="agent", text=spec.body)]
        return conv

    # -- read-only state ---------------------------------------------------- #
    @property
    def raw(self) -> RawDefinition:
        return self._raw

    @property
    def body(self) -> str:
        """The current working ruleset body (empty until :meth:`start`)."""
        return self._body

    @property
    def history(self) -> List[Turn]:
        """The full turn history (a copy — callers cannot mutate session state)."""
        return list(self._history)

    @property
    def state(self) -> str:
        return self._state

    @property
    def started(self) -> bool:
        return self._rounds > 0

    # -- the chain driver (shared by start + refine) ------------------------ #
    def _author_via_chain(self, system: str, user: str, *, offline_body: str) -> str:
        """Run the canonical chain and return the redrafted body.

        Mirrors ``compiler.condense_body``: ``transport=None`` wires the
        deterministic offline :class:`FakeTransport`; an injected transport drives
        live authoring. SYNCHRONOUS by design (d4) — a caller offloads it off the
        event loop. A live transport that returns empty/whitespace falls back to
        the deterministic body so a redraft is NEVER empty."""
        tp: Transport = self._transport or FakeTransport([offline_body])
        chain = build_default_chain(tp)
        ctx = chain.run(Context(system=system, user=user))
        # Strip a single OUTER ```markdown fence the small model often wraps the
        # whole body in — it is noise once composed into a sub-agent's system
        # prompt (shared with the compiler's live condense path).
        body = compiler.strip_code_fence(ctx.raw_output or "")
        if not body:
            body = offline_body
        return body

    def _ensure_open(self) -> None:
        if self._state in _TERMINAL:
            raise ConversationError(
                f"conversation for {self._raw.name!r} is {self._state}; no further "
                "authoring is possible"
            )

    # -- step: author the initial body -------------------------------------- #
    def start(
        self,
        first_message: str = "",
        *,
        description: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> DraftPreview:
        """Author the INITIAL output-shaping ruleset body and return a preview.

        ``first_message`` is the user's opening description of what the ruleset
        should do (it stands in for ``intent`` when the definition carried none).
        ``description`` / ``intent`` optionally evolve the definition at start
        (the frozen :class:`RawDefinition` is rebuilt). Idempotent guard: calling
        ``start`` twice is refused — use :meth:`refine` for subsequent turns."""
        self._ensure_open()
        if self._rounds > 0:
            raise ConversationError(
                f"conversation for {self._raw.name!r} already started; call refine()"
            )
        # Evolve the definition if the opening turn refined it.
        new_desc = description if description is not None else self._raw.description
        new_intent = intent if intent is not None else (first_message or self._raw.intent)
        self._raw = RawDefinition(
            name=self._raw.name, description=new_desc, intent=new_intent
        )

        opening = (first_message or self._raw.intent or "").strip()
        system, user = build_author_messages(self._raw, opening)
        body = self._author_via_chain(
            system, user, offline_body=offline_author_body(self._raw, opening)
        )
        self._body = body
        self._rounds += 1
        if opening:
            self._history.append(Turn(role="user", text=opening))
        self._history.append(Turn(role="agent", text=body))
        return self._preview()

    # -- step: re-author against a critique (THE new seam) ------------------ #
    def refine(self, user_critique: str) -> DraftPreview:
        """RE-AUTHOR the working body incorporating ``user_critique``.

        Conditioned on the PRIOR body (not authored from scratch) — the seam
        ``compiler.condense_body`` does not provide. Repeatable across turns;
        each call records the critique and the redrafted body on the history."""
        self._ensure_open()
        if self._rounds == 0:
            raise ConversationError(
                f"conversation for {self._raw.name!r} not started; call start() first"
            )
        if not (user_critique or "").strip():
            raise ValueError("refine requires a non-empty critique")

        system, user = build_refine_messages(self._raw, self._body, user_critique)
        body = self._author_via_chain(
            system,
            user,
            offline_body=offline_refine_body(self._raw, self._body, user_critique),
        )
        self._body = body
        self._rounds += 1
        self._history.append(Turn(role="user", text=user_critique.strip()))
        self._history.append(Turn(role="agent", text=body))
        return self._preview()

    # -- terminal: compile + register the current body ---------------------- #
    def approve(self) -> CompiledSpec:
        """Compile the current body into a :class:`CompiledSpec` and register it.

        Uses ``compiler.compile_spec`` + ``registry.register`` — the same
        compile+register write the engine's HITL gate performs, here driven by the
        user's explicit approval of the conversed body. A fresh session compiles
        under source ``"ui"``; a session opened via :meth:`reopen` compiles under
        the spec's ORIGINAL source/trace (provenance preserved). Returns the
        compiled spec; the conversation reaches the terminal ``approved`` state."""
        self._ensure_open()
        if self._rounds == 0 or not self._body.strip():
            raise ConversationError(
                f"nothing to approve for {self._raw.name!r}: call start() first"
            )
        spec = compiler.compile_spec(
            self._raw, self._body, source=self._source, trace_ref=self._trace_ref
        )
        self._registry.register(spec)
        self._state = STATE_APPROVED
        return spec

    def deny(self) -> None:
        """Close the conversation WITHOUT compiling (the user rejected the body)."""
        self._ensure_open()
        self._state = STATE_DENIED

    def cancel(self) -> None:
        """Abandon the conversation WITHOUT compiling (the user backed out)."""
        self._ensure_open()
        self._state = STATE_CANCELLED

    # -- internals ---------------------------------------------------------- #
    def _preview(self) -> DraftPreview:
        return DraftPreview(
            name=self._raw.name,
            description=self._raw.description,
            body=self._body,
            turn=self._rounds,
        )


__all__ = [
    "SpecConversation",
    "DraftPreview",
    "Turn",
    "ConversationError",
    "SOURCE_UI",
    "STATE_OPEN",
    "STATE_APPROVED",
    "STATE_DENIED",
    "STATE_CANCELLED",
    "build_author_messages",
    "build_refine_messages",
    "offline_author_body",
    "offline_refine_body",
]
