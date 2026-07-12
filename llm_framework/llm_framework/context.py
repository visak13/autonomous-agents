"""Claude-Code-style context management: token budgeting + compaction.

This module is the framework's answer to a small local model's bounded context
window (d10; today gemma4-e4b at num_ctx 32768 — phi4-mini historically). It keeps a live, in-window conversation and, Claude-Code-style, keeps it
from outgrowing the model's budget two ways (d4):

- **Auto-compaction** — whenever the estimated token count crosses a
  *configurable* threshold, the older turns are summarised into a compact
  summary and dropped from the window, preserving the system prompt and the
  most recent turns.
- **Manual ``compact()``** — the same operation on demand (the "/compact"
  command), independent of the threshold.

Compaction is **context hygiene, NOT a hard per-call pass/fail gate** (d4): it
never raises on a "too big" window and never blocks a call — it just keeps the
window lean. Every compaction emits a :class:`CompactionEvent`
(trigger reason, before/after token counts, the summary, which turns were
folded) so the demo and callers can prove exactly what happened.

Pluggability (so the cheap estimator can be swapped for a precise tokenizer):

- :class:`TokenCounter` is a tiny Protocol; the default
  :class:`HeuristicTokenCounter` wraps the dependency-light estimator from
  :mod:`llm_framework.tokens`. Pass any object with the same shape (e.g. a
  tiktoken-backed counter) to :class:`Conversation` to replace it.
- The *summariser* is injectable too. The default :class:`TransportSummarizer`
  calls the injected :class:`~llm_framework.transport.Transport` — with the
  scripted :class:`~llm_framework.transport.FakeTransport` this yields a
  deterministic canned summary, so the whole thing runs fully offline (d7/d8).

Integration with the Chain (a2): :meth:`Conversation.to_context` produces a
ready :class:`~llm_framework.chain.Context` (system prompt with the running
summary folded in, plus the live recent turns) for the canonical chain;
:meth:`Conversation.run_chain` runs a chain for one user turn and records both
sides back into the window, auto-compacting as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Protocol, Sequence, runtime_checkable

from .chain import Chain, Context
from .tokens import estimate_message_tokens, estimate_tokens
from .transport import Message, Transport

# --------------------------------------------------------------------------- #
# Pluggable token counter
# --------------------------------------------------------------------------- #


@runtime_checkable
class TokenCounter(Protocol):
    """The seam a token counter satisfies.

    Two methods so a counter can be exact about chat framing if it wants to;
    the heuristic default just adds a small per-message overhead. Swap in a
    precise tokenizer (e.g. tiktoken) by passing any object with this shape to
    :class:`Conversation` — the rest of the module is counter-agnostic.
    """

    def count_text(self, text: str) -> int:
        """Tokens in a bare string."""
        ...

    def count_messages(self, messages: Sequence[Message]) -> int:
        """Tokens in a list of chat messages (including role/framing overhead)."""
        ...


class HeuristicTokenCounter:
    """Default counter: the dependency-light estimator from :mod:`.tokens`.

    Deliberately approximate (its job is compaction budgeting, not billing —
    see :mod:`llm_framework.tokens`). ``per_message_overhead`` is forwarded to
    :func:`~llm_framework.tokens.estimate_message_tokens`.
    """

    def __init__(self, *, per_message_overhead: int = 4) -> None:
        self.per_message_overhead = per_message_overhead

    def count_text(self, text: str) -> int:
        return estimate_tokens(text)

    def count_messages(self, messages: Sequence[Message]) -> int:
        return estimate_message_tokens(
            messages, per_message_overhead=self.per_message_overhead
        )


# --------------------------------------------------------------------------- #
# Summarisers (injectable) — turn older turns into a compact summary
# --------------------------------------------------------------------------- #

# A summariser takes the turns to fold + any prior summary and returns the new
# compact summary text. Any callable with this shape is accepted.
Summarizer = Callable[[Sequence[Message], Optional[str]], str]

_DEFAULT_SUMMARY_SYSTEM = (
    "You compress a conversation so it can be dropped from the live context "
    "window while losing as little as possible. Write a concise summary that "
    "preserves the user's goals, every decision made, key facts, open "
    "questions, and anything needed to continue the task. Reply with ONLY the "
    "summary prose — no preamble, no headers, no code fences."
)

# Marker prefix the running summary is rendered under when re-injected, so it
# is visibly "earlier-conversation context" and not mistaken for a live turn.
SUMMARY_HEADER = "[Summary of earlier conversation]"


def _render_transcript(messages: Sequence[Message]) -> str:
    """Render messages as a plain ``role: content`` transcript for summarising."""
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "?")) if isinstance(msg, dict) else "?"
        content = str(msg.get("content", "")) if isinstance(msg, dict) else str(msg)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


class TransportSummarizer:
    """Summarise older turns by calling the injected transport (d4 path).

    With a real :class:`~llm_framework.transport.OllamaTransport` this is a live
    model summarisation call; with :class:`~llm_framework.transport.FakeTransport`
    its scripted reply is returned verbatim — a deterministic canned summary
    that needs no GPU (d7/d8). ``call_opts`` (temperature, keep_alive, …) are
    forwarded to ``transport.complete`` on every call.
    """

    def __init__(
        self,
        transport: Transport,
        *,
        system_prompt: str = _DEFAULT_SUMMARY_SYSTEM,
        **call_opts: Any,
    ) -> None:
        self.transport = transport
        self.system_prompt = system_prompt
        self.call_opts = call_opts

    def __call__(
        self, messages: Sequence[Message], prior_summary: Optional[str] = None
    ) -> str:
        transcript = _render_transcript(messages)
        user_parts: list[str] = []
        if prior_summary:
            user_parts.append(
                "A summary of the conversation BEFORE this excerpt already "
                f"exists; fold it into your new summary:\n\n{prior_summary}"
            )
        user_parts.append(
            "Summarise the following conversation excerpt:\n\n" + transcript
        )
        prompt: list[Message] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]
        return self.transport.complete(prompt, **self.call_opts).strip()


def deterministic_summary(
    messages: Sequence[Message], prior_summary: Optional[str] = None
) -> str:
    """Transport-free fallback summariser (fully offline, no model at all).

    Used only when a :class:`Conversation` is given neither a transport nor an
    explicit summariser. It is **bounded** — a fixed-shape header plus a capped
    gist of the folded turns — so it genuinely compresses a long history (and,
    by collapsing a prior summary to a marker, never grows unboundedly across
    repeated compactions). Deterministic, so unit tests stay reproducible.
    """
    gist_cap = 300
    pieces: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = " ".join(str(msg.get("content", "")).split())
        if content:
            pieces.append(content[:60])
    gist = " | ".join(pieces)[:gist_cap]
    parts = [SUMMARY_HEADER, f"Folded {len(messages)} earlier turn(s)."]
    if prior_summary:
        parts.append("(continues an earlier summary)")
    if gist:
        parts.append(f"Gist: {gist}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Compaction event — the exposed, capturable record
# --------------------------------------------------------------------------- #


@dataclass
class CompactionEvent:
    """One compaction, exposed so the demo can capture it (d4).

    The four headline fields the spec calls out — ``reason``, ``before_tokens``,
    ``after_tokens``, ``summary`` — plus a little provenance (how many turns were
    folded, the threshold in force, how many recent turns were kept).
    """

    reason: str  # "auto" (threshold crossed) | "manual" (compact() called)
    before_tokens: int
    after_tokens: int
    summary: str
    turns_summarized: int = 0
    kept_messages: int = 0
    threshold: Optional[int] = None

    @property
    def tokens_saved(self) -> int:
        return self.before_tokens - self.after_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "before_tokens": self.before_tokens,
            "after_tokens": self.after_tokens,
            "tokens_saved": self.tokens_saved,
            "summary": self.summary,
            "turns_summarized": self.turns_summarized,
            "kept_messages": self.kept_messages,
            "threshold": self.threshold,
        }


# --------------------------------------------------------------------------- #
# Conversation — the live in-window history + compaction
# --------------------------------------------------------------------------- #


DEFAULT_COMPACTION_THRESHOLD = 4000
"""Default token budget at which auto-compaction fires. Configurable per
Conversation — sized well under the model's window so recall + the new turn
still fit comfortably after the summary lands (d10)."""

DEFAULT_KEEP_RECENT = 4
"""Default number of most-recent messages preserved verbatim across a
compaction (≈ the last couple of exchanges)."""


class Conversation:
    """A live, in-window chat history with Claude-Code-style compaction.

    The window is: an optional ``system`` prompt, a running compaction
    ``summary`` (once anything has been compacted), then the live recent
    ``messages``. Append turns with :meth:`add_user` / :meth:`add_assistant`;
    when the estimated token count crosses ``compaction_threshold`` the older
    turns are folded into the summary automatically (if ``auto_compact``).
    Call :meth:`compact` to do it on demand.

    Parameters
    ----------
    system:
        Optional system prompt — always preserved across compactions.
    transport:
        Transport used by the default :class:`TransportSummarizer` (ignored if
        an explicit ``summarizer`` is given). If both are ``None`` the offline
        :func:`deterministic_summary` is used.
    token_counter:
        Pluggable :class:`TokenCounter`; defaults to :class:`HeuristicTokenCounter`.
    summarizer:
        Explicit summariser callable ``(messages, prior_summary) -> str``.
        Overrides ``transport``.
    compaction_threshold:
        Token budget that triggers auto-compaction (configurable, d4).
    keep_recent:
        How many most-recent messages to preserve verbatim on compaction.
    auto_compact:
        Whether appends trigger auto-compaction. ``compact()`` works regardless.
    """

    def __init__(
        self,
        *,
        system: Optional[str] = None,
        transport: Optional[Transport] = None,
        token_counter: Optional[TokenCounter] = None,
        summarizer: Optional[Summarizer] = None,
        compaction_threshold: int = DEFAULT_COMPACTION_THRESHOLD,
        keep_recent: int = DEFAULT_KEEP_RECENT,
        auto_compact: bool = True,
    ) -> None:
        if keep_recent < 0:
            raise ValueError("keep_recent must be >= 0")
        if compaction_threshold <= 0:
            raise ValueError("compaction_threshold must be > 0")
        self.system = system
        self.transport = transport
        self.counter: TokenCounter = token_counter or HeuristicTokenCounter()
        self.compaction_threshold = compaction_threshold
        self.keep_recent = keep_recent
        self.auto_compact = auto_compact

        # Resolve the summariser once: explicit > transport-backed > offline.
        if summarizer is not None:
            self._summarizer: Summarizer = summarizer
        elif transport is not None:
            self._summarizer = TransportSummarizer(transport)
        else:
            self._summarizer = deterministic_summary

        self._messages: List[Message] = []  # live recent turns (no system/summary)
        self._summary: Optional[str] = None  # running compaction summary
        self.events: List[CompactionEvent] = []  # every compaction, in order

    # -- appending turns --------------------------------------------------- #

    def add_message(self, role: str, content: str) -> Optional[CompactionEvent]:
        """Append a ``{role, content}`` turn; auto-compact if over budget.

        Returns the :class:`CompactionEvent` if this append triggered an
        auto-compaction, else ``None``.
        """
        self._messages.append({"role": role, "content": content})
        if self.auto_compact:
            return self.maybe_compact()
        return None

    def add_user(self, content: str) -> Optional[CompactionEvent]:
        return self.add_message("user", content)

    def add_assistant(self, content: str) -> Optional[CompactionEvent]:
        return self.add_message("assistant", content)

    def extend(self, messages: Sequence[Message]) -> Optional[CompactionEvent]:
        """Append several turns, then auto-compact at most once at the end."""
        for msg in messages:
            self._messages.append(
                {"role": str(msg.get("role", "user")), "content": str(msg.get("content", ""))}
            )
        if self.auto_compact:
            return self.maybe_compact()
        return None

    # -- the live window --------------------------------------------------- #

    @property
    def messages(self) -> List[Message]:
        """The full transport-ready window: ``system`` → running summary → recent turns.

        The optional ``system`` prompt and the running compaction ``summary`` lead the
        live recent turns (d263: the pinned head + SWA-tail re-injection are removed —
        the goal/doctrine ride the system/first turn ONCE and compaction carries the
        rest, rather than being re-pasted as always-in-view blocks every call)."""
        out: List[Message] = []
        if self.system:
            out.append({"role": "system", "content": self.system})
        if self._summary:
            out.append(
                {"role": "system", "content": f"{SUMMARY_HEADER}\n{self._summary}"}
            )
        out.extend(self._messages)
        return out

    @property
    def recent(self) -> List[Message]:
        """Just the live (un-summarised) recent turns."""
        return list(self._messages)

    @property
    def summary(self) -> Optional[str]:
        """The running compaction summary, or ``None`` if nothing compacted yet."""
        return self._summary

    def token_count(self) -> int:
        """Estimated tokens of the current window (system + summary + recent)."""
        return self.counter.count_messages(self.messages)

    def over_threshold(self) -> bool:
        return self.token_count() >= self.compaction_threshold

    # -- compaction -------------------------------------------------------- #

    def maybe_compact(self) -> Optional[CompactionEvent]:
        """Auto-compact iff the window is over the configured threshold (d4)."""
        if self.over_threshold():
            return self.compact(reason="auto")
        return None

    def compact(self, *, reason: str = "manual") -> Optional[CompactionEvent]:
        """Fold the older turns into the running summary; keep recent + system.

        Summarises everything except the last ``keep_recent`` messages (folding
        any prior summary so the running summary accumulates), replaces those
        older turns with the new summary, and records a :class:`CompactionEvent`
        with before/after token counts. Returns ``None`` (a no-op) when there is
        nothing older than the preserved window to compact — compaction never
        errors and never blocks (context hygiene, not a gate — d4).
        """
        if self.keep_recent:
            older = self._messages[: -self.keep_recent]
            recent = self._messages[-self.keep_recent :]
        else:
            older, recent = list(self._messages), []
        if not older:
            return None  # nothing to fold — recent window is all there is

        before_tokens = self.token_count()
        new_summary = self._summarizer(older, self._summary)

        self._summary = new_summary
        self._messages = recent
        after_tokens = self.token_count()

        event = CompactionEvent(
            reason=reason,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            summary=new_summary,
            turns_summarized=len(older),
            kept_messages=len(recent),
            threshold=self.compaction_threshold,
        )
        self.events.append(event)
        return event

    @property
    def last_compaction(self) -> Optional[CompactionEvent]:
        return self.events[-1] if self.events else None

    # -- Chain (a2) integration ------------------------------------------- #

    def to_context(
        self, user: Optional[str] = None, *, transport: Optional[Transport] = None
    ) -> Context:
        """Build a chain :class:`Context` from the current window.

        The running summary is folded into the system prompt (so the canonical
        ``prompt_assembly`` stage sees it as part of ``system``), the live recent
        turns become ``history``, and ``user`` is the new turn. The transport defaults
        to this conversation's own."""
        system = self.system
        if self._summary:
            block = f"{SUMMARY_HEADER}\n{self._summary}"
            system = f"{system}\n\n{block}" if system else block
        history = list(self._messages)
        return Context(
            system=system,
            history=history,
            user=user,
            transport=transport if transport is not None else self.transport,
        )

    def run_chain(self, chain: Chain, user: str) -> Context:
        """Run ``chain`` for one ``user`` turn and record both sides.

        Builds a context from the *current* window (so the model sees the
        summary + recent turns), runs the chain, then appends the user turn and
        the assistant's ``raw_output`` back into the window — auto-compacting if
        that pushes it over budget. Returns the resulting :class:`Context` so
        the caller can read ``raw_output`` / ``structured`` / ``meta``.
        """
        ctx = self.to_context(user=user)
        ctx = chain.run(ctx)
        self.add_user(user)
        self.add_assistant(ctx.raw_output or "")
        return ctx
