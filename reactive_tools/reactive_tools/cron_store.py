"""Durable persistence for cron jobs — the shared-SQLite repository (s2/a5).

This is the PERSISTENCE half of the cron capability. The three node-callable
cron *tools* (``cron_add`` / ``cron_list`` / ``cron_delete``) live in
:mod:`reactive_tools.cron_tools`; this module is the single-responsibility
**repository** they round-trip entries through. The always-on FIRING scheduler
service and the at-most-3 missed-fire catch-up are explicitly s6 — they are NOT
built here. This module only stores/reads/deletes the schedule rows and keeps the
bookkeeping columns s6 will later read.

Where the rows live (the shared SQLite DB)
------------------------------------------
Cron entries persist into the SAME on-disk SQLite database the app uses for chat
memory: ``<data_dir>/chat.db`` (the file :class:`chat_app.persistence.ChatStore`
owns). The cron rows go in their OWN ``cron_jobs`` table created additively
(``CREATE TABLE IF NOT EXISTS``) — this module never touches the chat/turn/
artifact tables. Sharing the one database file (rather than a second db) is the
action's explicit contract ("the same DB used for chat memory in s5").

DATA-DIR ALIGNMENT (critical, and why it is DUPLICATED not imported)
--------------------------------------------------------------------
The data root is resolved by :func:`resolve_data_dir`, whose contract is
byte-for-byte identical to :func:`chat_app.persistence.resolve_data_dir` /
:func:`chat_app.app._data_dir`: an explicit ``override`` → the
``REACTIVE_AGENTS_DATA_DIR`` env var → ``<repo>/var/chat_app`` default. It is
**duplicated rather than imported** because ``reactive_tools`` is a LOWER layer
that must not depend upward on ``chat_app`` (the same no-upward-dependency rule
:mod:`reactive_tools.scheduler` calls out, and the same reason
``chat_app.persistence`` itself duplicated ``app._data_dir``). Both this file and
``chat_app/chat_app/persistence.py`` sit three levels under the repo root
(``<repo>/<pkg>/<pkg>/<file>.py``), so ``parents[2]`` is the SAME ReactiveAgents
root in each and the default path can never diverge.

Design / standards honored
--------------------------
- d2 (in-process), file-based — stdlib :mod:`sqlite3`, no standing service, no
  second process. The connection is opened and closed in the SAME scope (the
  store is a context manager); the cron tools open a short-lived store per call
  so no long-lived handle is leaked (resource closed where opened).
- Fail fast at the boundary — a schedule string is validated as a 5-field cron
  expression (:func:`validate_cron_expression`, no regex) before any row is
  written; a bad expression raises :class:`~reactive_tools.tools.ToolInputError`.
- Timeouts on outbound I/O — the sqlite connection carries an explicit busy
  ``timeout`` so a lock contended by the chat store can never hang unbounded.
- Bookkeeping for s6 — ``last_run_at`` / ``next_run_at`` / ``last_status`` columns
  exist so the s6 firing service can record fires and compute missed-fire
  catch-up; this module leaves them ``NULL`` (it does not fire anything).
"""
from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from .tools import ToolInputError

# The DB filename MUST match :attr:`chat_app.persistence.ChatStore.DB_FILENAME`
# so cron rows land in the SAME database file as chat memory. Duplicated (not
# imported) to keep reactive_tools free of an upward dependency on chat_app.
DB_FILENAME = "chat.db"

# Env override + default root — the SAME contract as
# chat_app.persistence.resolve_data_dir (duplicated, not imported; see module doc).
DATA_DIR_ENV = "REACTIVE_AGENTS_DATA_DIR"

# Explicit busy timeout (seconds) for the sqlite connection: a bounded wait for a
# lock the chat store may hold, never an unbounded hang (timeouts-on-I/O rule).
DB_BUSY_TIMEOUT_SECONDS = 5.0

# Standard 5-field cron layout: minute hour day-of-month month day-of-week, with
# each field's inclusive numeric bounds (used by the no-regex validator below).
CRON_FIELD_COUNT = 5
_CRON_FIELD_BOUNDS: tuple[tuple[int, int], ...] = (
    (0, 59),   # minute
    (0, 23),   # hour
    (1, 31),   # day of month
    (1, 12),   # month
    (0, 7),    # day of week (0 and 7 are both Sunday)
)
_CRON_FIELD_NAMES = ("minute", "hour", "day-of-month", "month", "day-of-week")


def resolve_data_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the on-disk root for persisted state (shared with chat memory).

    Precedence (identical to :func:`chat_app.persistence.resolve_data_dir`):
    an explicit ``override`` → the ``REACTIVE_AGENTS_DATA_DIR`` env var →
    ``<repo>/var/chat_app``. ``reactive_tools/reactive_tools/cron_store.py`` is
    three levels under the repo root, so ``parents[2]`` is the SAME root the chat
    store computes. The directory is created if missing.
    """
    if override is not None:
        root = Path(override)
    elif os.environ.get(DATA_DIR_ENV):
        root = Path(os.environ[DATA_DIR_ENV])
    else:
        # reactive_tools/reactive_tools/cron_store.py -> parents[2] == repo root.
        root = Path(__file__).resolve().parents[2] / "var" / "chat_app"
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_db_path(db_path: str | os.PathLike[str] | None = None,
                    data_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the shared SQLite db file (``<data_dir>/chat.db``).

    An explicit ``db_path`` wins (handy for tests pointing at a temp file);
    otherwise the file is ``DB_FILENAME`` under :func:`resolve_data_dir`."""
    if db_path is not None:
        return Path(db_path)
    return resolve_data_dir(data_dir) / DB_FILENAME


def _now_iso() -> str:
    """UTC, second-precision ISO-8601 timestamp (sortable, tz-explicit)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- #
# Cron-expression validation — 5-field standard cron, NO regex (spec rule #7)
# --------------------------------------------------------------------------- #
def _validate_cron_token(token: str, lo: int, hi: int, field_name: str) -> None:
    """Validate one comma-separated token of a cron field, in-range, no regex.

    Accepts ``*``, ``*/N``, ``A``, ``A-B``, ``A-B/N`` and ``A/N`` where A, B, N
    are integers; A/B must lie within ``[lo, hi]`` and N must be > 0. Raises
    :class:`~reactive_tools.tools.ToolInputError` on anything else."""
    if not token:
        raise ToolInputError(f"empty token in cron {field_name} field")
    # Split off an optional step (`base/step`).
    base, sep, step = token.partition("/")
    if sep:
        if not step.isdigit() or int(step) <= 0:
            raise ToolInputError(
                f"cron {field_name} step must be a positive integer, got {step!r}")
    if base == "*":
        return
    # A range `A-B` or a single value `A`.
    start, dash, end = base.partition("-")
    parts = (start, end) if dash else (start,)
    for p in parts:
        if not p.lstrip("-").isdigit():
            raise ToolInputError(
                f"cron {field_name} value must be an integer, got {p!r}")
        v = int(p)
        if v < lo or v > hi:
            raise ToolInputError(
                f"cron {field_name} value {v} out of range [{lo},{hi}]")
    if dash and int(start) > int(end):
        raise ToolInputError(
            f"cron {field_name} range start {start} exceeds end {end}")


def validate_cron_expression(expr: str) -> str:
    """Validate a standard 5-field cron expression and return it normalized.

    Fail-fast boundary guard: the schedule string must be exactly five
    whitespace-separated fields (minute hour day-of-month month day-of-week),
    each a comma-list of in-range tokens. Parsed with plain string ops + integer
    checks — **no regex** (spec standard #7: no regex without user approval).
    Returns the single-spaced normalized form; raises
    :class:`~reactive_tools.tools.ToolInputError` on a malformed expression."""
    if not isinstance(expr, str) or not expr.strip():
        raise ToolInputError("cron schedule must be a non-empty string")
    fields = expr.split()
    if len(fields) != CRON_FIELD_COUNT:
        raise ToolInputError(
            f"cron schedule must have {CRON_FIELD_COUNT} fields "
            f"(minute hour day-of-month month day-of-week), got {len(fields)}: {expr!r}")
    for field, (lo, hi), name in zip(fields, _CRON_FIELD_BOUNDS, _CRON_FIELD_NAMES):
        for token in field.split(","):
            _validate_cron_token(token, lo, hi, name)
    return " ".join(fields)


# --------------------------------------------------------------------------- #
# Cron-expression TIME MATH — match a datetime / enumerate due fire windows.
#
# The s2 store only VALIDATED an expression; the s6 firing scheduler needs to know
# WHEN an expression is due. This is the matching half, kept here next to the
# parser/validator (single home for all cron-expression logic, reusable, lower
# layer with no upward dependency). Pure stdlib + plain string ops — NO regex
# (standard #7), and NO third-party cron library (croniter is not a dependency).
# Granularity is one MINUTE: cron's finest field is the minute, so the scheduler
# reasons over minute-truncated timestamps.
# --------------------------------------------------------------------------- #

# A parsed expression: the five fields expanded to the explicit SETS of values a
# matching timestamp's (minute, hour, day-of-month, month, day-of-week) may take.
ParsedCron = tuple[frozenset[int], ...]

# Full expansions of the wildcard ``*`` per field — used to tell a RESTRICTED
# day-of-month / day-of-week field from an unrestricted one (the Vixie-cron rule
# below). day-of-week is normalised so Sunday is only ever 0 (cron's 7 -> 0).
_FULL_DOM: frozenset[int] = frozenset(range(1, 32))
_FULL_DOW: frozenset[int] = frozenset(range(0, 7))


def _expand_cron_field(field: str, lo: int, hi: int) -> frozenset[int]:
    """Expand ONE validated cron field to the explicit set of values it matches.

    Mirrors the token grammar :func:`_validate_cron_token` already accepts —
    ``*``, ``*/N``, ``A``, ``A-B``, ``A-B/N`` and ``A/N`` (the last meaning "from
    A to the field maximum, stepping by N", the standard cron reading). The field
    is assumed already validated by :func:`validate_cron_expression`."""
    values: set[int] = set()
    for token in field.split(","):
        base, sep, step = token.partition("/")
        stepv = int(step) if sep else 1
        if base == "*":
            start, end = lo, hi
        else:
            a, dash, b = base.partition("-")
            if dash:
                start, end = int(a), int(b)
            elif sep:
                # "A/N": a single base WITH a step means A..max step N (cron std).
                start, end = int(a), hi
            else:
                start = end = int(a)
        values.update(range(start, end + 1, stepv))
    return frozenset(values)


def parse_cron_expression(expr: str) -> ParsedCron:
    """Validate ``expr`` and expand its five fields to explicit value sets.

    Returns ``(minute, hour, day-of-month, month, day-of-week)`` frozensets ready
    for :func:`cron_matches`. day-of-week is normalised so a ``7`` (cron's
    alternate Sunday) collapses to ``0`` — matching against a Sunday is then
    uniform regardless of which form the expression used."""
    fields = validate_cron_expression(expr).split()
    parsed = tuple(
        _expand_cron_field(f, lo, hi)
        for f, (lo, hi) in zip(fields, _CRON_FIELD_BOUNDS)
    )
    dow = parsed[4]
    if 7 in dow:
        dow = frozenset((dow - {7}) | {0})
        parsed = parsed[:4] + (dow,)
    return parsed


def cron_matches(parsed: ParsedCron, dt: datetime) -> bool:
    """True when ``dt`` (minute granularity) satisfies the parsed expression.

    minute / hour / month must all match. The day match follows the standard
    Vixie-cron rule: when BOTH day-of-month and day-of-week are restricted (i.e.
    not ``*``), the day matches if EITHER does; otherwise (one is ``*``) both must
    match — and the wildcard one always does. ``dt`` is compared on its minute,
    hour, day, month and weekday (cron weekday: Sunday=0..Saturday=6)."""
    minute, hour, dom, month, dow = parsed
    if dt.minute not in minute or dt.hour not in hour or dt.month not in month:
        return False
    # Python isoweekday(): Mon=1..Sun=7; cron weekday: Sun=0..Sat=6 -> mod 7.
    cron_dow = dt.isoweekday() % 7
    dom_match = dt.day in dom
    dow_match = cron_dow in dow
    dom_restricted = dom != _FULL_DOM
    dow_restricted = dow != _FULL_DOW
    if dom_restricted and dow_restricted:
        return dom_match or dow_match
    return dom_match and dow_match


def _floor_minute(dt: datetime) -> datetime:
    """``dt`` truncated to the start of its minute (cron's finest granularity)."""
    return dt.replace(second=0, microsecond=0)


# How far back a single catch-up pass scans for missed windows. The firing
# scheduler keeps only the newest few fires anyway, so a window older than this
# (e.g. a job whose last fire was weeks ago) is inherently dropped — bounding the
# scan keeps a tick cheap no matter how stale ``last_run_at`` is.
DEFAULT_CATCHUP_SCAN_MINUTES = 7 * 24 * 60  # one week of minutes
# Horizon for computing the NEXT fire after "now" (bookkeeping only). A year is
# ample for any standard expression; an expression with no match in a year is
# pathological and simply yields ``None``.
DEFAULT_NEXT_SCAN_MINUTES = 366 * 24 * 60


def iter_due_fire_times(
    expr: str,
    after: datetime,
    upto: datetime,
    *,
    max_scan_minutes: int = DEFAULT_CATCHUP_SCAN_MINUTES,
) -> list[datetime]:
    """Every minute the expression is due in ``(after, upto]``, oldest → newest.

    The fire windows STRICTLY after ``after`` (the last fire / job creation
    baseline — exclusive, so a window is never fired twice) and up to and
    INCLUDING ``upto`` (now — so a window due exactly now fires this tick). The
    lookback is bounded to ``max_scan_minutes`` before ``upto`` so a long-stale
    ``after`` can never make a tick scan unboundedly; windows older than that are
    dropped (the scheduler caps catch-up anyway). All datetimes are minute-
    truncated; pass timezone-aware values (the store persists UTC)."""
    parsed = parse_cron_expression(expr)
    upto_m = _floor_minute(upto)
    after_m = _floor_minute(after)
    floor = upto_m - timedelta(minutes=max_scan_minutes)
    if after_m < floor:
        after_m = floor
    matches: list[datetime] = []
    one = timedelta(minutes=1)
    t = upto_m
    while t > after_m:
        if cron_matches(parsed, t):
            matches.append(t)
        t -= one
    matches.reverse()
    return matches


def next_fire_after(
    expr: str,
    after: datetime,
    *,
    max_scan_minutes: int = DEFAULT_NEXT_SCAN_MINUTES,
) -> Optional[datetime]:
    """The first minute the expression is due STRICTLY after ``after`` (or None).

    Forward scan from the minute after ``after`` up to ``max_scan_minutes`` ahead.
    Used purely for the ``next_run_at`` bookkeeping column."""
    parsed = parse_cron_expression(expr)
    one = timedelta(minutes=1)
    t = _floor_minute(after) + one
    limit = t + timedelta(minutes=max_scan_minutes)
    while t <= limit:
        if cron_matches(parsed, t):
            return t
        t += one
    return None


# --------------------------------------------------------------------------- #
# The stored shape
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CronJob:
    """One persisted cron entry: a schedule + the plan/prompt to run + bookkeeping.

    ``schedule`` is a validated 5-field cron expression; ``prompt`` is the
    plan/prompt the s6 firing service will run when the schedule is due. The
    ``last_run_at`` / ``next_run_at`` / ``last_status`` fields are bookkeeping
    s6 owns (left ``None`` by this layer)."""

    job_id: str
    name: str
    schedule: str
    prompt: str
    enabled: bool
    created_at: str
    updated_at: str
    last_run_at: Optional[str] = None
    next_run_at: Optional[str] = None
    last_status: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "schedule": self.schedule,
            "prompt": self.prompt,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_run_at": self.last_run_at,
            "next_run_at": self.next_run_at,
            "last_status": self.last_status,
        }


# --------------------------------------------------------------------------- #
# The repository
# --------------------------------------------------------------------------- #
class CronStore:
    """SQLite-backed CRUD for cron jobs in the shared ``chat.db`` (s2/a5).

    Single responsibility: persist / list / delete cron schedule rows. It owns
    ONLY the ``cron_jobs`` table, created additively so it coexists with the chat
    store's tables in the same database file. Use it as a context manager so the
    connection is closed in the scope that opened it; the cron tools open a
    short-lived store per call (cron ops are infrequent), so no long-lived handle
    is held."""

    TABLE = "cron_jobs"

    def __init__(self, db_path: str | os.PathLike[str] | None = None,
                 *, data_dir: str | os.PathLike[str] | None = None) -> None:
        self.db_path = str(resolve_db_path(db_path, data_dir))
        # check_same_thread=False: a single in-process app may touch the store
        # from uvicorn worker thread(s); a lock serialises writes. The explicit
        # busy timeout bounds any wait for a lock the chat store holds.
        self.db = sqlite3.connect(
            self.db_path, check_same_thread=False, timeout=DB_BUSY_TIMEOUT_SECONDS)
        self.db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock:
            self.db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    job_id      TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    schedule    TEXT NOT NULL,
                    prompt      TEXT NOT NULL,
                    enabled     INTEGER NOT NULL DEFAULT 1,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    last_run_at TEXT,
                    next_run_at TEXT,
                    last_status TEXT
                )
                """
            )
            self.db.commit()

    def _row_to_job(self, row: sqlite3.Row) -> CronJob:
        return CronJob(
            job_id=row["job_id"],
            name=row["name"],
            schedule=row["schedule"],
            prompt=row["prompt"],
            enabled=bool(row["enabled"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_run_at=row["last_run_at"],
            next_run_at=row["next_run_at"],
            last_status=row["last_status"],
        )

    # ---- write side (commits before returning — durable on return) ---- #
    def add(self, *, schedule: str, prompt: str, name: str = "",
            enabled: bool = True) -> CronJob:
        """Persist a new cron entry and return it (durable immediately).

        ``schedule`` is validated as a 5-field cron expression at this boundary
        (a bad expression raises before any write); ``prompt`` is the plan to run
        on fire and must be non-empty. A fresh ``cron-<hex>`` id is minted."""
        norm_schedule = validate_cron_expression(schedule)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ToolInputError("cron prompt (the plan to run) must be non-empty")
        job_id = f"cron-{uuid.uuid4().hex[:12]}"
        now = _now_iso()
        label = name.strip() if isinstance(name, str) and name.strip() else job_id
        with self._lock:
            self.db.execute(
                f"INSERT INTO {self.TABLE}(job_id, name, schedule, prompt, enabled, "
                f"created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job_id, label, norm_schedule, prompt, 1 if enabled else 0, now, now),
            )
            self.db.commit()
        return CronJob(
            job_id=job_id, name=label, schedule=norm_schedule, prompt=prompt,
            enabled=enabled, created_at=now, updated_at=now,
        )

    def delete(self, job_id: str) -> bool:
        """Delete the entry with ``job_id``. Returns True if a row was removed."""
        if not isinstance(job_id, str) or not job_id.strip():
            raise ToolInputError("job_id to delete must be a non-empty string")
        with self._lock:
            cur = self.db.execute(
                f"DELETE FROM {self.TABLE} WHERE job_id = ?", (job_id,))
            self.db.commit()
            return cur.rowcount > 0

    def record_fire(self, job_id: str, *, last_run_at: str,
                    next_run_at: Optional[str] = None,
                    last_status: Optional[str] = None) -> bool:
        """Advance a job's persisted FIRE state after the s6 scheduler fired it.

        Writes the bookkeeping columns the s2 layer deliberately left ``NULL``:
        ``last_run_at`` (the newest window this job has fired — the baseline the
        next tick's catch-up scans AFTER, so a window is never fired twice, and
        which SURVIVES a restart so a fresh scheduler resumes from here),
        ``next_run_at`` (the upcoming due window, informational) and
        ``last_status`` (the outcome of the most recent fire). Durable on return.
        Returns True if a row was updated (False if ``job_id`` is unknown)."""
        if not isinstance(job_id, str) or not job_id.strip():
            raise ToolInputError("job_id to record a fire for must be non-empty")
        now = _now_iso()
        with self._lock:
            cur = self.db.execute(
                f"UPDATE {self.TABLE} SET last_run_at = ?, next_run_at = ?, "
                f"last_status = ?, updated_at = ? WHERE job_id = ?",
                (last_run_at, next_run_at, last_status, now, job_id),
            )
            self.db.commit()
            return cur.rowcount > 0

    # ---- read side (restart-safe: a fresh process reloads everything) ---- #
    def get(self, job_id: str) -> Optional[CronJob]:
        """One cron entry by id, or ``None`` if unknown."""
        row = self.db.execute(
            f"SELECT * FROM {self.TABLE} WHERE job_id = ?", (job_id,)
        ).fetchone()
        return self._row_to_job(row) if row is not None else None

    def list(self, *, enabled_only: bool = False) -> list[CronJob]:
        """Every cron entry (oldest first), or only enabled ones."""
        sql = f"SELECT * FROM {self.TABLE}"
        if enabled_only:
            sql += " WHERE enabled = 1"
        sql += " ORDER BY created_at, job_id"
        return [self._row_to_job(r) for r in self.db.execute(sql).fetchall()]

    def count(self) -> int:
        return int(self.db.execute(
            f"SELECT COUNT(*) FROM {self.TABLE}").fetchone()[0])

    # ---- lifecycle ---- #
    def close(self) -> None:
        with self._lock:
            self.db.close()

    def __enter__(self) -> "CronStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        # Close the resource in the scope that opened it (house rule #10).
        self.close()


__all__ = [
    "DB_FILENAME",
    "DATA_DIR_ENV",
    "DB_BUSY_TIMEOUT_SECONDS",
    "CRON_FIELD_COUNT",
    "resolve_data_dir",
    "resolve_db_path",
    "validate_cron_expression",
    "ParsedCron",
    "parse_cron_expression",
    "cron_matches",
    "iter_due_fire_times",
    "next_fire_after",
    "DEFAULT_CATCHUP_SCAN_MINUTES",
    "DEFAULT_NEXT_SCAN_MINUTES",
    "CronJob",
    "CronStore",
]
