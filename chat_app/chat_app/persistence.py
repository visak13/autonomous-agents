"""Durable persistence for chat sessions + artifacts (s7/b1, O7).

WHAT THIS GUARANTEES
--------------------
Chat history and generated artifacts SURVIVE A SERVER RESTART. A running
``chat_app`` writes every turn (the user request + the agent's streamed
reasoning/progress events + the final response) and every artifact through
this layer; a FRESH process started later loads the exact same chats and
reproduces identical artifact bytes. That round-trip is proved by
``evidence/s7/b1/persist_roundtrip.json`` (process #1 writes + exits, a brand
new interpreter reloads).

DESIGN (honors the recipe's load-bearing decisions)
---------------------------------------------------
- **d2 (in-process), d8 (file-based):** the store is a stdlib :mod:`sqlite3`
  database plus on-disk artifact files — NO standing service, no extra
  process, nothing to connect to. It opens a file handle and that is all.
  This mirrors the house style already used by :class:`memory.DurableFactStore`
  (in-process sqlite, resource closed where opened).
- **DATA-DIR ALIGNMENT (critical):** the on-disk root is resolved by
  :func:`resolve_data_dir`, which is byte-for-byte identical to
  :func:`chat_app.app._data_dir` — same override arg, same
  ``REACTIVE_AGENTS_DATA_DIR`` env var, same ``<repo>/var/chat_app`` default.
  Persistence therefore lives UNDER the SAME root the running app uses
  (``<data_dir>/chat.db`` + ``<data_dir>/artifacts/``); there is never a
  second, divergent data dir that would silently break the restart-reopen
  proof. The resolver is duplicated rather than imported so this module stays
  importable on its own (the restart smoke spins a lean interpreter that need
  not pull the whole FastAPI/agent stack just to read the store).

PUBLIC SURFACE (the functions the action names)
-----------------------------------------------
- :meth:`ChatStore.save_turn`        — append one turn to a chat (creating it)
- :meth:`ChatStore.save_artifact`    — persist artifact bytes + its metadata
- :meth:`ChatStore.list_chats`       — every chat (id, title, created_at, …)
- :meth:`ChatStore.get_chat`         — one chat: full history + artifact refs
- :meth:`ChatStore.get_artifact_path`— on-disk path of an artifact's bytes

Module-level thin wrappers of the same names operate on a default store rooted
at :func:`resolve_data_dir` for callers that don't hold an instance.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# data-dir resolution — MUST stay identical to chat_app.app._data_dir
# --------------------------------------------------------------------------- #
def resolve_data_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the on-disk root for persisted state.

    Identical contract to :func:`chat_app.app._data_dir`: ``override`` arg →
    ``REACTIVE_AGENTS_DATA_DIR`` env → ``<repo>/var/chat_app`` default. Both
    ``app.py`` and ``persistence.py`` sit at ``chat_app/chat_app/<file>.py`` so
    ``parents[2]`` is the SAME ReactiveAgents repo root in each — the default
    path can never diverge. The directory is created if missing.
    """
    if override is not None:
        root = Path(override)
    elif os.environ.get("REACTIVE_AGENTS_DATA_DIR"):
        root = Path(os.environ["REACTIVE_AGENTS_DATA_DIR"])
    else:
        # chat_app/chat_app/persistence.py -> parents[2] == ReactiveAgents repo root.
        root = Path(__file__).resolve().parents[2] / "var" / "chat_app"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _now_iso() -> str:
    """UTC timestamp, second-precision ISO-8601 (sortable, timezone-explicit)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- #
# value shapes returned to callers (plain, JSON-friendly dataclasses)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TurnRecord:
    """One ordered turn in a chat: the user request, the agent's streamed
    reasoning/progress events, and the final response."""

    turn_index: int
    user_request: str
    events: list[dict[str, Any]]   # streamed reasoning / progress lifecycle events
    final_response: str
    created_at: str
    chat_id: str = ""   # the chat this turn landed in (minted if none supplied)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "turn_index": self.turn_index,
            "user_request": self.user_request,
            "events": self.events,
            "final_response": self.final_response,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ArtifactRef:
    """A persisted artifact's metadata. The bytes live on disk at ``path``
    (relative to the data root); :meth:`ChatStore.get_artifact_path` resolves
    it to an absolute path the caller can read."""

    artifact_id: str
    chat_id: str
    filename: str
    mime: str
    path: str          # path RELATIVE to the data root (portable across moves)
    size: int
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "chat_id": self.chat_id,
            "filename": self.filename,
            "mime": self.mime,
            "path": self.path,
            "size": self.size,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ResearchBriefRecord:
    """A per-research BRIEF persisted at the chat-session level (d185 Layer 2).

    One short thesis-level digest of a single research, addressable by
    ``(chat_id, research_id)`` so multiple researches in one chat coexist and are
    each looked up by their brief. ``brief`` is the DERIVED projection dict
    (``research_tree.build_research_brief``); ``topic`` is hoisted out of it for a
    cheap listing/lookup without re-parsing the JSON."""

    chat_id: str
    research_id: str
    topic: str
    brief: dict[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "research_id": self.research_id,
            "topic": self.topic,
            "brief": self.brief,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ChatRecord:
    """A chat session: identity + ordered turns + artifact refs (full history)."""

    chat_id: str
    title: str
    created_at: str
    turns: list[TurnRecord] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "title": self.title,
            "created_at": self.created_at,
            "turns": [t.to_dict() for t in self.turns],
            "artifacts": [a.to_dict() for a in self.artifacts],
        }


# --------------------------------------------------------------------------- #
# the durable store
# --------------------------------------------------------------------------- #
class ChatStore:
    """In-process, file-based durable store for chat sessions + artifacts.

    Layout under the data root (:func:`resolve_data_dir`):

        <data_dir>/chat.db          sqlite: chats, turns, artifact metadata
        <data_dir>/artifacts/<id>/<filename>   artifact bytes (one dir per id)

    sqlite gives durable, atomic, queryable metadata with zero standing service
    (d2/d8); artifact bytes are kept as plain files (binary blobs don't belong
    inline in the row, and a file path is what the web layer serves). Every
    ``save_*`` commits immediately, so a crash after any call leaves a complete,
    reloadable record — that is the restart guarantee O7 demands.
    """

    ARTIFACTS_SUBDIR = "artifacts"
    DB_FILENAME = "chat.db"

    def __init__(self, data_dir: str | os.PathLike[str] | None = None) -> None:
        self.data_dir = resolve_data_dir(data_dir)
        self.artifacts_dir = self.data_dir / self.ARTIFACTS_SUBDIR
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / self.DB_FILENAME
        # check_same_thread=False: a single in-process app may touch the store
        # from uvicorn's worker thread(s); a process-wide lock serialises writes
        # so the one connection stays consistent (no second DB process — d2).
        self.db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id    TEXT PRIMARY KEY,
                    title      TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    turn_pk        INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id        TEXT NOT NULL REFERENCES chats(chat_id),
                    turn_index     INTEGER NOT NULL,
                    user_request   TEXT NOT NULL,
                    events_json    TEXT NOT NULL,
                    final_response TEXT NOT NULL,
                    created_at     TEXT NOT NULL,
                    UNIQUE(chat_id, turn_index)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    chat_id     TEXT NOT NULL REFERENCES chats(chat_id),
                    filename    TEXT NOT NULL,
                    mime        TEXT NOT NULL,
                    rel_path    TEXT NOT NULL,
                    size        INTEGER NOT NULL,
                    created_at  TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_turns_chat
                    ON turns(chat_id, turn_index);
                CREATE INDEX IF NOT EXISTS idx_artifacts_chat
                    ON artifacts(chat_id);
                -- d185 NOTES-ARCH Layer 2 — per-research BRIEF, stored at the
                -- CHAT-SESSION level so MULTIPLE researches in one chat coexist,
                -- each addressable by (chat_id, research_id). ADDITIVE only
                -- (CREATE TABLE IF NOT EXISTS): leaves chats/turns/artifacts and
                -- all existing rows untouched; a17 owns the broader session
                -- binding + schema-intact verification. ``brief_json`` is the
                -- DERIVED brief projection (research_tree.build_research_brief),
                -- stored verbatim; (chat_id, research_id) is the addressable key.
                CREATE TABLE IF NOT EXISTS research_briefs (
                    chat_id     TEXT NOT NULL REFERENCES chats(chat_id),
                    research_id TEXT NOT NULL,
                    topic       TEXT NOT NULL,
                    brief_json  TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    PRIMARY KEY (chat_id, research_id)
                );
                CREATE INDEX IF NOT EXISTS idx_briefs_chat
                    ON research_briefs(chat_id);
                """
            )
            self.db.commit()

    # ---- write side (every call commits — durable on return) ---- #
    def create_chat(self, title: str | None = None) -> ChatRecord:
        """Mint a NEW, empty chat and return its record (durable immediately).

        Backs the b3 ``POST /chats`` route: the frontend opens a chat (and its
        live SSE stream) BEFORE the first message is sent, so the chat row must
        exist with zero turns. A fresh id is minted; the row is committed before
        returning. Idempotent only in the trivial sense (each call mints a new
        id) — there is intentionally no upsert here."""
        cid = f"chat-{uuid.uuid4().hex[:12]}"
        with self._lock:
            created = self._ensure_chat(cid, title or "New chat")
            self.db.commit()
        return ChatRecord(chat_id=cid, title=title or "New chat", created_at=created)

    def _ensure_chat(self, chat_id: str, title: str) -> str:
        """Insert the chat row if absent (idempotent). Returns its created_at."""
        row = self.db.execute(
            "SELECT created_at FROM chats WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        if row is not None:
            return row["created_at"]
        created = _now_iso()
        self.db.execute(
            "INSERT INTO chats(chat_id, title, created_at) VALUES (?, ?, ?)",
            (chat_id, title, created),
        )
        return created

    def save_turn(
        self,
        chat_id: str | None,
        user_request: str,
        events: list[dict[str, Any]] | None = None,
        final_response: str = "",
        *,
        title: str | None = None,
    ) -> TurnRecord:
        """Append one turn to ``chat_id`` (creating the chat on first turn).

        ``events`` is the agent's ordered streamed reasoning/progress lifecycle
        (the same shape the SSE stream relays) and is stored verbatim as JSON.
        The turn index is assigned server-side as the next slot for the chat, so
        ordering is authoritative regardless of caller. Pass ``chat_id=None`` to
        start a NEW chat (a fresh id is minted and returned on the record). The
        write commits before returning — the turn is durable immediately.
        """
        cid = chat_id or f"chat-{uuid.uuid4().hex[:12]}"
        events = events or []
        with self._lock:
            self._ensure_chat(cid, title or self._derive_title(user_request))
            row = self.db.execute(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 AS nxt "
                "FROM turns WHERE chat_id = ?",
                (cid,),
            ).fetchone()
            turn_index = int(row["nxt"])
            created = _now_iso()
            self.db.execute(
                "INSERT INTO turns(chat_id, turn_index, user_request, events_json, "
                "final_response, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    cid,
                    turn_index,
                    user_request,
                    json.dumps(events, ensure_ascii=False),
                    final_response,
                    created,
                ),
            )
            self.db.commit()
        return TurnRecord(
            turn_index=turn_index,
            user_request=user_request,
            events=events,
            final_response=final_response,
            created_at=created,
            chat_id=cid,
        )

    def save_artifact(
        self,
        chat_id: str,
        filename: str,
        data: bytes,
        mime: str = "application/octet-stream",
    ) -> ArtifactRef:
        """Persist artifact ``data`` (bytes) for ``chat_id`` and record its
        metadata. Bytes are written to ``<data_dir>/artifacts/<id>/<filename>``
        (one directory per artifact id so identical filenames never collide);
        the metadata row stores the path RELATIVE to the data root so the store
        survives the whole tree being moved. The chat must already exist (a
        turn is saved first in practice); a missing chat is created defensively
        so an artifact is never silently dropped.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"artifact data must be bytes, got {type(data).__name__}")
        artifact_id = f"art-{uuid.uuid4().hex[:12]}"
        safe_name = Path(filename).name or "artifact.bin"  # strip any path parts
        rel_dir = Path(self.ARTIFACTS_SUBDIR) / artifact_id
        abs_dir = self.data_dir / rel_dir
        abs_dir.mkdir(parents=True, exist_ok=True)
        abs_path = abs_dir / safe_name
        abs_path.write_bytes(bytes(data))
        rel_path = (rel_dir / safe_name).as_posix()
        created = _now_iso()
        with self._lock:
            # Defensive: ensure the parent chat exists for the FK.
            self._ensure_chat(chat_id, self._derive_title(safe_name))
            self.db.execute(
                "INSERT INTO artifacts(artifact_id, chat_id, filename, mime, "
                "rel_path, size, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (artifact_id, chat_id, safe_name, mime, rel_path, len(data), created),
            )
            self.db.commit()
        return ArtifactRef(
            artifact_id=artifact_id,
            chat_id=chat_id,
            filename=safe_name,
            mime=mime,
            path=rel_path,
            size=len(data),
            created_at=created,
        )

    # ---- read side (restart-safe: a fresh process reloads everything) ---- #
    def list_chats(self) -> list[ChatRecord]:
        """Every chat, newest first, WITH full turns + artifact refs.

        This is what the app calls on startup to rehydrate the UI: the returned
        records are the complete persisted history, reloaded straight from disk.
        """
        rows = self.db.execute(
            "SELECT chat_id FROM chats ORDER BY created_at DESC, chat_id DESC"
        ).fetchall()
        out: list[ChatRecord] = []
        for r in rows:
            rec = self.get_chat(r["chat_id"])
            if rec is not None:
                out.append(rec)
        return out

    def get_chat(self, chat_id: str) -> ChatRecord | None:
        """One chat's full history (ordered turns) + its artifact refs, or
        ``None`` if no such chat exists."""
        head = self.db.execute(
            "SELECT chat_id, title, created_at FROM chats WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
        if head is None:
            return None
        turn_rows = self.db.execute(
            "SELECT turn_index, user_request, events_json, final_response, "
            "created_at FROM turns WHERE chat_id = ? ORDER BY turn_index",
            (chat_id,),
        ).fetchall()
        turns = [
            TurnRecord(
                turn_index=int(t["turn_index"]),
                user_request=t["user_request"],
                events=json.loads(t["events_json"]),
                final_response=t["final_response"],
                created_at=t["created_at"],
                chat_id=chat_id,
            )
            for t in turn_rows
        ]
        art_rows = self.db.execute(
            "SELECT artifact_id, chat_id, filename, mime, rel_path, size, "
            "created_at FROM artifacts WHERE chat_id = ? ORDER BY created_at, "
            "artifact_id",
            (chat_id,),
        ).fetchall()
        artifacts = [
            ArtifactRef(
                artifact_id=a["artifact_id"],
                chat_id=a["chat_id"],
                filename=a["filename"],
                mime=a["mime"],
                path=a["rel_path"],
                size=int(a["size"]),
                created_at=a["created_at"],
            )
            for a in art_rows
        ]
        return ChatRecord(
            chat_id=head["chat_id"],
            title=head["title"],
            created_at=head["created_at"],
            turns=turns,
            artifacts=artifacts,
        )

    def get_artifact_ref(self, artifact_id: str) -> ArtifactRef | None:
        """The metadata of one artifact by id, or ``None`` if unknown.

        Backs the b3 ``GET /artifacts/{id}`` download route: the web layer needs
        the ``filename`` + ``mime`` (for ``Content-Disposition`` + ``Content-Type``)
        alongside the on-disk path. Kept beside :meth:`get_artifact_path` so the
        route can resolve metadata and bytes from the same store."""
        row = self.db.execute(
            "SELECT artifact_id, chat_id, filename, mime, rel_path, size, "
            "created_at FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            return None
        return ArtifactRef(
            artifact_id=row["artifact_id"],
            chat_id=row["chat_id"],
            filename=row["filename"],
            mime=row["mime"],
            path=row["rel_path"],
            size=int(row["size"]),
            created_at=row["created_at"],
        )

    def get_artifact_path(self, artifact_id: str) -> Path | None:
        """Absolute on-disk path of an artifact's bytes, or ``None`` if unknown.

        The stored path is relative to the data root, so it is resolved against
        :attr:`data_dir` here — the bytes are read back identically by any
        process that opens the same store."""
        row = self.db.execute(
            "SELECT rel_path FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()
        if row is None:
            return None
        return (self.data_dir / row["rel_path"]).resolve()

    # ---- per-research brief (d185 NOTES-ARCH Layer 2) ---- #
    def save_research_brief(
        self,
        chat_id: str,
        research_id: str,
        brief: dict[str, Any],
    ) -> ResearchBriefRecord:
        """Persist one research's BRIEF at the chat-session level (durable on return).

        Keyed by ``(chat_id, research_id)`` so MULTIPLE researches in one chat coexist
        and are each addressable by their brief; re-saving the SAME key UPSERTS (a
        re-run of the same research refreshes its digest, never duplicates). The chat
        row is ensured defensively for the FK so a brief is never silently dropped.
        ADDITIVE: touches only the ``research_briefs`` table (a17 owns wider keying)."""
        topic = str((brief or {}).get("topic") or "").strip()
        created = _now_iso()
        with self._lock:
            self._ensure_chat(chat_id, self._derive_title(topic or "Research"))
            self.db.execute(
                "INSERT INTO research_briefs(chat_id, research_id, topic, "
                "brief_json, created_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(chat_id, research_id) DO UPDATE SET "
                "topic=excluded.topic, brief_json=excluded.brief_json, "
                "created_at=excluded.created_at",
                (chat_id, research_id, topic,
                 json.dumps(brief or {}, ensure_ascii=False), created),
            )
            self.db.commit()
        return ResearchBriefRecord(
            chat_id=chat_id,
            research_id=research_id,
            topic=topic,
            brief=dict(brief or {}),
            created_at=created,
        )

    def get_research_brief(
        self, chat_id: str, research_id: str
    ) -> ResearchBriefRecord | None:
        """One research's brief by its addressable ``(chat_id, research_id)`` key."""
        row = self.db.execute(
            "SELECT chat_id, research_id, topic, brief_json, created_at "
            "FROM research_briefs WHERE chat_id = ? AND research_id = ?",
            (chat_id, research_id),
        ).fetchone()
        return self._brief_from_row(row)

    def list_research_briefs(self, chat_id: str) -> list[ResearchBriefRecord]:
        """Every research brief in one chat session, newest first.

        This is the addressable index a chat keeps: each distinct research surfaces
        as its own brief, so a follow-up can name + look up the one it means."""
        rows = self.db.execute(
            "SELECT chat_id, research_id, topic, brief_json, created_at "
            "FROM research_briefs WHERE chat_id = ? "
            "ORDER BY created_at DESC, research_id DESC",
            (chat_id,),
        ).fetchall()
        return [r for r in (self._brief_from_row(row) for row in rows) if r is not None]

    @staticmethod
    def _brief_from_row(row: sqlite3.Row | None) -> ResearchBriefRecord | None:
        if row is None:
            return None
        try:
            brief = json.loads(row["brief_json"])
        except (ValueError, TypeError):
            brief = {}
        if not isinstance(brief, dict):
            brief = {}
        return ResearchBriefRecord(
            chat_id=row["chat_id"],
            research_id=row["research_id"],
            topic=row["topic"],
            brief=brief,
            created_at=row["created_at"],
        )

    # ---- misc ---- #
    @staticmethod
    def _derive_title(text: str, limit: int = 60) -> str:
        """A readable default chat title from the first user request."""
        line = (text or "").strip().splitlines()[0] if (text or "").strip() else "Untitled chat"
        return line[:limit].strip() or "Untitled chat"

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def __enter__(self) -> "ChatStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        # Close the resource in the scope that opened it (house style).
        self.close()


# --------------------------------------------------------------------------- #
# module-level convenience wrappers over a default-rooted store
# --------------------------------------------------------------------------- #
def open_store(data_dir: str | os.PathLike[str] | None = None) -> ChatStore:
    """Open a :class:`ChatStore` rooted at the resolved data dir."""
    return ChatStore(data_dir)


def save_turn(
    chat_id: str | None,
    user_request: str,
    events: list[dict[str, Any]] | None = None,
    final_response: str = "",
    *,
    title: str | None = None,
    data_dir: str | os.PathLike[str] | None = None,
) -> TurnRecord:
    """Convenience wrapper: append a turn via a transient default-rooted store."""
    with open_store(data_dir) as store:
        return store.save_turn(
            chat_id, user_request, events, final_response, title=title
        )


def save_artifact(
    chat_id: str,
    filename: str,
    data: bytes,
    mime: str = "application/octet-stream",
    *,
    data_dir: str | os.PathLike[str] | None = None,
) -> ArtifactRef:
    """Convenience wrapper: persist an artifact via a transient store."""
    with open_store(data_dir) as store:
        return store.save_artifact(chat_id, filename, data, mime)


def list_chats(
    data_dir: str | os.PathLike[str] | None = None,
) -> list[ChatRecord]:
    """Convenience wrapper: load all chats via a transient store."""
    with open_store(data_dir) as store:
        return store.list_chats()


def get_chat(
    chat_id: str, data_dir: str | os.PathLike[str] | None = None
) -> ChatRecord | None:
    """Convenience wrapper: load one chat via a transient store."""
    with open_store(data_dir) as store:
        return store.get_chat(chat_id)


def get_artifact_path(
    artifact_id: str, data_dir: str | os.PathLike[str] | None = None
) -> Path | None:
    """Convenience wrapper: resolve an artifact path via a transient store."""
    with open_store(data_dir) as store:
        return store.get_artifact_path(artifact_id)
