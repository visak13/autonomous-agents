"""Conversation memory — bounded prior-turn context per chat thread (s5/a1).

WHAT THIS IS
------------
A THIN read/assembly layer ON TOP of the existing durable
:class:`~chat_app.persistence.ChatStore`. Turns already persist there (the user
request + the agent's streamed events + the final response); this module does
NOT re-persist them. Given a ``chat_id`` — which IS the "thread" in this app (the
``chats`` table; one chat == one conversation thread) — it returns the prior-turn
history assembled into a BOUNDED conversation-context block suitable to inject
into the planner before a new turn runs.

WHY BOUNDED
-----------
The planner runs on a small local Gemma model with a modest context window, so
the assembled block must NEVER be unbounded history. Two configurable bounds
apply together:

* ``recent_turns`` — at most the last K turns are considered (most recent =
  most relevant to the next turn).
* ``max_chars`` — a hard character budget on the final block. When the recent
  turns overflow it, the OLDEST are dropped first (newest kept), and a final
  hard truncation guarantees ``len(block) <= max_chars`` even for a single
  oversized turn.

Optionally a compact per-chat RUNNING SUMMARY (the ``chat_summaries`` table) is
prepended, so long threads keep continuity with older turns the recent-N window
no longer covers — without growing the block. The summary is the caller's to
write (e.g. a future summarizer node); this layer only stores/retrieves it.

ISOLATION INVARIANT (load-bearing)
----------------------------------
Every read is STRICTLY scoped to one ``chat_id``. Turn reads go through
:meth:`ChatStore.get_chat`, which is keyed by ``chat_id``; the summary table is
keyed by ``chat_id``. One thread can therefore never read another thread's
turns or summary — proved by ``tests/test_chat_memory.py``.

PERSISTENCE (no new DB file, restart-durable)
---------------------------------------------
The running-summary table lives in the SAME shared ``<data_dir>/chat.db`` as the
chat store + shape-config + cron tools — there is NO second database file. It
owns its own connection to that file under its own lock, mirroring the proven
:class:`~chat_app.shape_config.ShapeConfigStore` discipline (sqlite permits
several connections to one database; the ``chat_summaries`` table is disjoint
from the chat tables, so there is no contention with
:class:`~chat_app.persistence.ChatStore`). Every write commits immediately, and
the schema is created idempotently (``CREATE TABLE IF NOT EXISTS``, no migration
framework), so a fresh process reopening the same file reads back exactly what
was written.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from chat_app.persistence import ChatStore, TurnRecord


def _now_iso() -> str:
    """UTC timestamp, second-precision ISO-8601 (sortable, timezone-explicit)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# Default bounds tuned for a small local Gemma context window. Both are
# overridable per-instance (constructor) and per-call (assemble_context args).
DEFAULT_RECENT_TURNS = 6
DEFAULT_MAX_CHARS = 4000

# Block formatting — plain, model-friendly role-prefixed lines.
USER_PREFIX = "User:"
ASSISTANT_PREFIX = "Assistant:"
SUMMARY_HEADER = "Summary of earlier conversation:"
TURN_SEP = "\n\n"


class ConversationMemory:
    """Bounded prior-turn context assembly for one chat thread, on top of ChatStore.

    Composition, not inheritance: a :class:`~chat_app.persistence.ChatStore`
    supplies the durable turns (read via its public :meth:`get_chat`), and this
    object adds (1) a bounded assembly API and (2) an optional per-chat running
    summary persisted in the SAME ``chat.db``. It is constructible standalone
    against any data dir (so a fresh process can reopen it after a restart).

    Thread-safety mirrors the rest of the app: the summary connection is guarded
    by a process-wide lock so the single connection stays consistent when an
    async app touches it from uvicorn worker threads.
    """

    DB_FILENAME = "chat.db"

    def __init__(
        self,
        store: ChatStore,
        *,
        recent_turns: int = DEFAULT_RECENT_TURNS,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        if recent_turns < 1:
            raise ValueError(f"recent_turns must be >= 1 (got {recent_turns})")
        if max_chars < 1:
            raise ValueError(f"max_chars must be >= 1 (got {max_chars})")
        self._store = store
        self.recent_turns = int(recent_turns)
        self.max_chars = int(max_chars)
        # Own connection to the SHARED chat.db (no new DB file) — same data dir as
        # the store, mirroring ShapeConfigStore's disjoint-table multi-connection
        # pattern.
        self.data_dir = store.data_dir
        self.db_path = self.data_dir / self.DB_FILENAME
        self.db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._create_schema()

    def _create_schema(self) -> None:
        """Create the running-summary table if absent (idempotent, restart-safe)."""
        with self._lock:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_summaries (
                    chat_id    TEXT PRIMARY KEY,
                    summary    TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self.db.commit()

    # ---- turn retrieval (strictly per chat_id, in order) ---- #
    def recent_turns_for(
        self, chat_id: str, limit: Optional[int] = None
    ) -> list[TurnRecord]:
        """The most recent turns of ``chat_id``, in CHRONOLOGICAL order.

        At most ``limit`` (default the instance ``recent_turns``) turns are
        returned — the last K of the chat — preserving ascending ``turn_index``
        so the planner reads them oldest→newest. An unknown chat returns ``[]``
        (never another chat's turns); a non-positive ``limit`` returns ``[]``.
        """
        k = self.recent_turns if limit is None else int(limit)
        if k <= 0:
            return []
        rec = self._store.get_chat(chat_id)
        if rec is None:
            return []
        # get_chat already returns turns ordered by turn_index ascending.
        return list(rec.turns[-k:])

    # ---- running summary (per chat_id, durable in the shared chat.db) ---- #
    def set_summary(self, chat_id: str, summary: str) -> None:
        """Persist (upsert) a compact running summary for ``chat_id``.

        Durable on return. Scoped to ``chat_id`` so it can never surface for
        another thread. The caller is responsible for keeping the summary
        compact — it is prepended to the assembled block subject to the same
        ``max_chars`` bound."""
        cid = (chat_id or "").strip()
        if not cid:
            raise ValueError("chat_id must be non-empty")
        with self._lock:
            self.db.execute(
                "INSERT INTO chat_summaries(chat_id, summary, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(chat_id) DO UPDATE SET "
                "summary = excluded.summary, updated_at = excluded.updated_at",
                (cid, summary, _now_iso()),
            )
            self.db.commit()

    def get_summary(self, chat_id: str) -> Optional[str]:
        """The stored running summary for ``chat_id``, or ``None`` if none set."""
        row = self.db.execute(
            "SELECT summary FROM chat_summaries WHERE chat_id = ?",
            ((chat_id or "").strip(),),
        ).fetchone()
        return row["summary"] if row is not None else None

    # ---- bounded assembly (the planner-injection surface) ---- #
    @staticmethod
    def _format_turn(turn: TurnRecord) -> str:
        """One turn as a role-prefixed block (assistant line dropped if empty)."""
        lines = [f"{USER_PREFIX} {turn.user_request}".rstrip()]
        if (turn.final_response or "").strip():
            lines.append(f"{ASSISTANT_PREFIX} {turn.final_response}".rstrip())
        return "\n".join(lines)

    def assemble_context(
        self,
        chat_id: str,
        *,
        recent_turns: Optional[int] = None,
        max_chars: Optional[int] = None,
        include_summary: bool = True,
    ) -> str:
        """Assemble a BOUNDED prior-turn context block for ``chat_id``.

        The block is an optional running summary followed by the most recent
        turns (oldest→newest), formatted as ``User:``/``Assistant:`` lines. It is
        bounded two ways and the return is GUARANTEED ``len(block) <= max_chars``:

        1. at most ``recent_turns`` turns are considered, and
        2. if those still overflow ``max_chars``, the OLDEST turns are dropped
           first (the most recent turn is always preferred), then a final hard
           truncation enforces the budget even for a single oversized turn/summary.

        Strictly scoped to ``chat_id`` (isolation invariant). An unknown chat with
        no summary yields ``""``.
        """
        k = self.recent_turns if recent_turns is None else int(recent_turns)
        budget = self.max_chars if max_chars is None else int(max_chars)
        if budget <= 0:
            return ""

        turns = self.recent_turns_for(chat_id, limit=k)
        summary = self.get_summary(chat_id) if include_summary else None

        summary_block = (
            f"{SUMMARY_HEADER}\n{summary}" if (summary and summary.strip()) else ""
        )

        # Fit the most recent turns within whatever budget the summary leaves,
        # adding newest→oldest and stopping before the budget is exceeded; the
        # summary always takes precedence (older context already distilled).
        sep_len = len(TURN_SEP)
        used = len(summary_block)
        kept_newest_first: list[str] = []
        for turn in reversed(turns):  # newest first
            block = self._format_turn(turn)
            extra = len(block) + (sep_len if (used > 0) else 0)
            if used + extra > budget:
                break
            kept_newest_first.append(block)
            used += extra

        ordered = ([summary_block] if summary_block else []) + list(
            reversed(kept_newest_first)  # back to chronological order
        )
        text = TURN_SEP.join(b for b in ordered if b)

        # Final safety net: a single turn (or the summary) larger than the whole
        # budget is hard-truncated so the bound holds unconditionally.
        if len(text) > budget:
            text = text[:budget]
        return text

    # ---- lifecycle ---- #
    def close(self) -> None:
        """Close the summary connection (the ChatStore is owned by the caller)."""
        with self._lock:
            self.db.close()

    def __enter__(self) -> "ConversationMemory":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def build_conversation_memory(
    data_dir: str | os.PathLike[str] | None = None,
    *,
    recent_turns: int = DEFAULT_RECENT_TURNS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> ConversationMemory:
    """Convenience constructor: a memory over a fresh ChatStore at ``data_dir``.

    The app wiring passes its SHARED :class:`ChatStore` instead (so reads see the
    same turns the message path writes); this is for stand-alone callers/tests
    that just need a memory rooted at a data dir."""
    return ConversationMemory(
        ChatStore(data_dir), recent_turns=recent_turns, max_chars=max_chars
    )


__all__ = [
    "ConversationMemory",
    "build_conversation_memory",
    "DEFAULT_RECENT_TURNS",
    "DEFAULT_MAX_CHARS",
]
