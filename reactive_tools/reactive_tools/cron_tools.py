"""cron_list / cron_add / cron_delete — the CRON tools, three registry entries (s2/a5).

These are the three node-callable cron capabilities, each expressed as **ONE**
:class:`~reactive_tools.tool_registry.ToolDef` on the a1
:class:`~reactive_tools.tool_registry.GrowableToolRegistry` (adding a tool = one
entry, decision d1 — no registry-core change). They sit BEHIND the a1 interface:
once added each is immediately selectable (in the structured-selection enum) and
dispatchable (through the bound :class:`~reactive_tools.tool_hook.ToolHook`, with
``tool_call`` / ``tool_result`` events on the event plane).

1. ``cron_add``    — register a schedule (5-field cron expression) + the
   plan/prompt to run when it fires. Persists a row; returns the new entry.
2. ``cron_list``   — list the registered cron entries (round-trips what ``cron_add``
   wrote, straight from the shared SQLite db).
3. ``cron_delete`` — remove an entry by ``job_id``.

Persistence is the shared SQLite db (``<data_dir>/chat.db``) via
:class:`reactive_tools.cron_store.CronStore` — the same database file the chat
store uses for chat memory. Each handler opens a SHORT-LIVED store per call (cron
ops are infrequent and not a hot path), so the db handle is closed in the same
scope it is opened (resource-closed-where-opened) and there is no long-lived
connection to leak.

SCOPE BOUNDARY (load-bearing)
-----------------------------
This action builds the list/add/delete TOOLS and their SQLite persistence ONLY.
The always-on FIRING scheduler service and the at-most-3 missed-fire catch-up are
**s6** — they are deliberately not built here. ``cron_add`` therefore stores the
schedule + prompt + bookkeeping columns (``last_run_at`` / ``next_run_at``) but
nothing in this module ever fires a job.

Decisions / standards honored
-----------------------------
- d1  — each tool is exactly one ``ToolDef`` on the growable registry; no
  framework, no control flow here.
- d2  — purely in-process stdlib sqlite; no broker/pool/subprocess.
- spec — the schedule is validated as a 5-field cron expression with NO regex
  (standard #7); a malformed schedule surfaces as a structured ``ok=False``
  (the registry validates args before dispatch); failures are never swallowed.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from .cron_store import CronStore, resolve_db_path, validate_cron_expression
from .tool_registry import GrowableToolRegistry, ToolDef
from .tools import ToolInputError

CRON_ADD_NAME = "cron_add"
CRON_LIST_NAME = "cron_list"
CRON_DELETE_NAME = "cron_delete"


# --------------------------------------------------------------------------- #
# Pydantic arg models — the single source of truth for each tool's schema
# --------------------------------------------------------------------------- #
class CronAddArgs(BaseModel):
    """Args for ``cron_add`` — a schedule + the plan/prompt to run on fire."""

    schedule: str = Field(
        ...,
        description=(
            "A standard 5-field cron expression 'minute hour day-of-month month "
            "day-of-week' (e.g. '0 9 * * *' = every day at 09:00)."
        ),
    )
    prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "The plan/prompt to run when the schedule fires (the agent task, e.g. "
            "'research today's AI news and email me a summary')."
        ),
    )
    name: str = Field(
        "",
        description="Optional human-readable label for the job (defaults to the job id).",
    )

    @field_validator("schedule")
    @classmethod
    def _check_schedule(cls, v: str) -> str:
        """Reject a malformed schedule at the schema boundary (fail-fast, no regex).

        Raises so the registry's ``validate_args`` surfaces a structured failure
        the runtime can react to, instead of writing a junk row."""
        return validate_cron_expression(v)


class CronListArgs(BaseModel):
    """Args for ``cron_list`` — optionally restrict to enabled jobs."""

    enabled_only: bool = Field(
        False,
        description="If True, list only enabled jobs; otherwise list every job.",
    )


class CronDeleteArgs(BaseModel):
    """Args for ``cron_delete`` — the id of the job to remove."""

    job_id: str = Field(
        ...,
        min_length=1,
        description="The id (as returned by cron_add / cron_list) of the job to delete.",
    )


# --------------------------------------------------------------------------- #
# Handlers — built bound to a resolved db path (short-lived store per call)
# --------------------------------------------------------------------------- #
def make_cron_add(db_path: str):
    """Build the ``cron_add`` handler bound to the shared-db ``db_path``."""

    def cron_add(schedule: str, prompt: str, name: str = "") -> dict[str, Any]:
        """Persist a new cron entry (schedule + plan) and return it.

        The schedule is re-validated defensively here too (the schema already
        validated it). Opens a short-lived :class:`CronStore` so the db handle is
        closed in this scope. Returns ``{ok, job_id, job}``; the firing of the
        job is s6's concern, not this tool's."""
        with CronStore(db_path) as store:
            job = store.add(schedule=schedule, prompt=prompt, name=name)
        return {"ok": True, "job_id": job.job_id, "job": job.to_dict()}

    return cron_add


def make_cron_list(db_path: str):
    """Build the ``cron_list`` handler bound to the shared-db ``db_path``."""

    def cron_list(enabled_only: bool = False) -> dict[str, Any]:
        """List the persisted cron entries (round-trips what ``cron_add`` wrote)."""
        with CronStore(db_path) as store:
            jobs = store.list(enabled_only=enabled_only)
        return {"ok": True, "count": len(jobs), "jobs": [j.to_dict() for j in jobs]}

    return cron_list


def make_cron_delete(db_path: str):
    """Build the ``cron_delete`` handler bound to the shared-db ``db_path``."""

    def cron_delete(job_id: str) -> dict[str, Any]:
        """Delete the entry with ``job_id``. ``deleted`` is False if no such job."""
        with CronStore(db_path) as store:
            deleted = store.delete(job_id)
        return {"ok": True, "deleted": deleted, "job_id": job_id}

    return cron_delete


# --------------------------------------------------------------------------- #
# The three registry entries (one ToolDef each) + registration helpers
# --------------------------------------------------------------------------- #
def build_cron_tools(db_path: str | os.PathLike[str] | None = None,
                     *, data_dir: str | os.PathLike[str] | None = None) -> list[ToolDef]:
    """Build the three :class:`ToolDef` entries (``cron_add``/``cron_list``/
    ``cron_delete``), all bound to the SAME resolved shared-db path.

    The db file is resolved once (see :func:`reactive_tools.cron_store.resolve_db_path`)
    and shared by all three handlers. Returns the three defs — each is exactly one
    registry entry, the whole point of d1's growability."""
    path = str(resolve_db_path(db_path, data_dir))
    add_def = ToolDef(
        name=CRON_ADD_NAME,
        description=(
            "Schedule a recurring task: store a 5-field cron expression (e.g. "
            "'0 9 * * *') plus the plan/prompt to run when it fires. Returns the "
            "new job's id. Does not run the job now — a background scheduler fires "
            "it on schedule."
        ),
        args_model=CronAddArgs,
        handler=make_cron_add(path),
    )
    list_def = ToolDef(
        name=CRON_LIST_NAME,
        description=(
            "List the scheduled (cron) jobs that have been registered, with their "
            "schedule, prompt and id. Returns {ok, count, jobs}."
        ),
        args_model=CronListArgs,
        handler=make_cron_list(path),
    )
    delete_def = ToolDef(
        name=CRON_DELETE_NAME,
        description=(
            "Delete a scheduled (cron) job by its id (from cron_add / cron_list). "
            "Returns {ok, deleted, job_id}; deleted is false if no such job exists."
        ),
        args_model=CronDeleteArgs,
        handler=make_cron_delete(path),
    )
    return [add_def, list_def, delete_def]


def register_cron_tools(registry: GrowableToolRegistry,
                        db_path: str | os.PathLike[str] | None = None,
                        *, data_dir: str | os.PathLike[str] | None = None) -> list[ToolDef]:
    """Add ``cron_add`` + ``cron_list`` + ``cron_delete`` to ``registry`` (the a1
    growth point).

    Each :meth:`GrowableToolRegistry.add` makes the tool immediately selectable
    (enum) AND dispatchable (hook). Returns the three added defs. The shared-db
    path is resolved once and shared by all three."""
    defs = build_cron_tools(db_path, data_dir=data_dir)
    for d in defs:
        registry.add(d)
    return defs


__all__ = [
    "CRON_ADD_NAME",
    "CRON_LIST_NAME",
    "CRON_DELETE_NAME",
    "CronAddArgs",
    "CronListArgs",
    "CronDeleteArgs",
    "make_cron_add",
    "make_cron_list",
    "make_cron_delete",
    "build_cron_tools",
    "register_cron_tools",
]
