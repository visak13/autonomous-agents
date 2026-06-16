"""Unit + integration coverage for the cron tools (s2/a5) — NO live model/network.

Proves the action's load-bearing properties entirely in-process against a temp
SQLite db:

- **Round-trip through SQLite** — ``cron_add`` -> ``cron_list`` shows it ->
  ``cron_delete`` removes it.
- **Persistence survives a process restart** — a SEPARATE :class:`CronStore`
  opened on the SAME db file still sees the entry (the restart guarantee).
- **Shared DB** — cron rows land in the same ``chat.db`` file; the ``cron_jobs``
  table is additive and does not disturb a chat ``ChatStore`` on the same file.
- **One registry entry each** — each cron tool is selectable (in the enum) and
  dispatchable (through the hook) after a single ``register_cron_tools``.
- **Cron validation, no regex** — a malformed schedule is rejected at the schema
  boundary and surfaces as a structured ``ok=False`` (never a junk row).
- **Scope boundary** — nothing here fires a job (the scheduler is s6).
"""
from __future__ import annotations

import asyncio

import pytest

from reactive_tools import EventPlane
from reactive_tools.tool_hook import ToolHook
from reactive_tools.tool_registry import GrowableToolRegistry, ToolRegistryError
from reactive_tools.cron_store import (
    CronStore,
    validate_cron_expression,
)
from reactive_tools.cron_tools import (
    CRON_ADD_NAME,
    CRON_DELETE_NAME,
    CRON_LIST_NAME,
    CronAddArgs,
    build_cron_tools,
    make_cron_add,
    make_cron_delete,
    make_cron_list,
    register_cron_tools,
)
from reactive_tools.tools import ToolInputError


@pytest.fixture
def db_path(tmp_path):
    """A temp shared-db file path (mirrors ``<data_dir>/chat.db``)."""
    return str(tmp_path / "chat.db")


# --------------------------------------------------------------------------- #
# cron-expression validation — 5-field standard cron, no regex
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "expr",
    [
        "0 9 * * *",          # every day at 09:00
        "*/5 * * * *",        # every 5 minutes
        "0 0 1 1 *",          # midnight on Jan 1
        "30 6,18 * * 1-5",    # 06:30 and 18:30 on weekdays
        "0 9 * * 0",          # Sundays at 09:00
        "15 14 1 * 1-5/2",    # range with a step
    ],
)
def test_valid_cron_expressions_accepted(expr):
    assert validate_cron_expression(expr) == " ".join(expr.split())


@pytest.mark.parametrize(
    "expr",
    [
        "",                   # empty
        "* * * *",            # too few fields
        "* * * * * *",        # too many fields
        "60 * * * *",         # minute out of range
        "* 24 * * *",         # hour out of range
        "* * 0 * *",          # day-of-month below range
        "* * * 13 *",         # month out of range
        "* * * * 8 ",         # day-of-week above range (7 max)
        "*/0 * * * *",        # zero step
        "abc * * * *",        # non-numeric
        "5-1 * * * *",        # inverted range
    ],
)
def test_invalid_cron_expressions_rejected(expr):
    with pytest.raises(ToolInputError):
        validate_cron_expression(expr)


def test_schema_validator_rejects_bad_schedule():
    """A malformed schedule is rejected at the Pydantic arg-model boundary."""
    with pytest.raises(Exception):
        CronAddArgs(schedule="not a cron", prompt="do x")


# --------------------------------------------------------------------------- #
# CronStore — CRUD round-trip + restart persistence
# --------------------------------------------------------------------------- #
def test_store_add_list_delete_round_trip(db_path):
    with CronStore(db_path) as store:
        assert store.count() == 0
        job = store.add(schedule="0 9 * * *", prompt="email me the news", name="daily")
        assert job.job_id.startswith("cron-")
        assert job.schedule == "0 9 * * *"
        assert job.name == "daily"
        # bookkeeping columns exist but are unset (firing is s6's job)
        assert job.last_run_at is None and job.next_run_at is None

        jobs = store.list()
        assert len(jobs) == 1 and jobs[0].job_id == job.job_id

        assert store.delete(job.job_id) is True
        assert store.list() == []
        # deleting a now-absent id reports False, not an error
        assert store.delete(job.job_id) is False


def test_store_persists_across_restart(db_path):
    """A SEPARATE store on the SAME file sees the entry — survives a restart."""
    with CronStore(db_path) as store1:
        job = store1.add(schedule="*/10 * * * *", prompt="poll the feed")
        job_id = job.job_id
    # proc #2: a brand-new connection to the same db file
    with CronStore(db_path) as store2:
        reloaded = store2.get(job_id)
        assert reloaded is not None
        assert reloaded.schedule == "*/10 * * * *"
        assert reloaded.prompt == "poll the feed"


def test_store_rejects_empty_prompt(db_path):
    with CronStore(db_path) as store:
        with pytest.raises(ToolInputError):
            store.add(schedule="0 9 * * *", prompt="   ")


def test_store_enabled_only_filter(db_path):
    with CronStore(db_path) as store:
        store.add(schedule="0 9 * * *", prompt="a", enabled=True)
        store.add(schedule="0 10 * * *", prompt="b", enabled=False)
        assert len(store.list()) == 2
        enabled = store.list(enabled_only=True)
        assert len(enabled) == 1 and enabled[0].prompt == "a"


def test_cron_rows_share_db_without_disturbing_other_tables(db_path):
    """The cron_jobs table is additive: a sibling table on the same file is
    untouched (proves cron rows coexist in the shared DB)."""
    import sqlite3

    # a sibling table (stand-in for the chat store's tables) on the same file
    side = sqlite3.connect(db_path)
    side.execute("CREATE TABLE IF NOT EXISTS chats (chat_id TEXT PRIMARY KEY)")
    side.execute("INSERT INTO chats(chat_id) VALUES ('chat-1')")
    side.commit()
    side.close()

    with CronStore(db_path) as store:
        store.add(schedule="0 9 * * *", prompt="x")
        assert store.count() == 1

    # the sibling table + its row are intact alongside cron_jobs
    check = sqlite3.connect(db_path)
    rows = check.execute("SELECT chat_id FROM chats").fetchall()
    tables = {
        r[0] for r in check.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    check.close()
    assert rows == [("chat-1",)]
    assert {"chats", "cron_jobs"} <= tables


# --------------------------------------------------------------------------- #
# The tools — handlers round-trip; structured results
# --------------------------------------------------------------------------- #
def test_tool_handlers_round_trip(db_path):
    add = make_cron_add(db_path)
    lst = make_cron_list(db_path)
    delete = make_cron_delete(db_path)

    res = add("0 9 * * *", "email me the news", "daily")
    assert res["ok"] is True
    job_id = res["job_id"]
    assert res["job"]["schedule"] == "0 9 * * *"

    listed = lst()
    assert listed["ok"] is True and listed["count"] == 1
    assert listed["jobs"][0]["job_id"] == job_id

    deleted = delete(job_id)
    assert deleted == {"ok": True, "deleted": True, "job_id": job_id}
    assert lst()["count"] == 0


# --------------------------------------------------------------------------- #
# ONE registry entry each — selectable + dispatchable after one register call
# --------------------------------------------------------------------------- #
def test_register_cron_tools_are_three_entries_selectable_and_dispatchable(db_path):
    hook = ToolHook(EventPlane())
    registry = GrowableToolRegistry(hook)
    defs = register_cron_tools(registry, db_path)
    assert {d.name for d in defs} == {CRON_ADD_NAME, CRON_LIST_NAME, CRON_DELETE_NAME}

    # selectable: each appears by name in the selection enum
    enum = set(registry.selection_schema()["properties"]["tool"]["enum"])
    assert {CRON_ADD_NAME, CRON_LIST_NAME, CRON_DELETE_NAME} <= enum

    # dispatchable: add -> list -> delete entirely through the bound hook
    added = asyncio.run(hook.invoke(
        CRON_ADD_NAME, schedule="0 9 * * *", prompt="do the thing"))
    assert added.ok is True
    job_id = added.value["job_id"]

    listed = asyncio.run(hook.invoke(CRON_LIST_NAME))
    assert listed.ok is True and listed.value["count"] == 1

    removed = asyncio.run(hook.invoke(CRON_DELETE_NAME, job_id=job_id))
    assert removed.ok is True and removed.value["deleted"] is True


def test_build_cron_tools_share_one_db(db_path):
    """All three defs bind to the SAME resolved db file."""
    defs = build_cron_tools(db_path)
    add = next(d for d in defs if d.name == CRON_ADD_NAME)
    lst = next(d for d in defs if d.name == CRON_LIST_NAME)
    add.handler(schedule="0 9 * * *", prompt="p")
    assert lst.handler()["count"] == 1


def test_bad_schedule_through_registry_is_structured_not_crash(db_path):
    """A malformed schedule routed through validate_args is a structured error
    (the dispatch layer can react), never a handler crash."""
    hook = ToolHook(EventPlane())
    registry = GrowableToolRegistry(hook)
    register_cron_tools(registry, db_path)
    tool = registry.get(CRON_ADD_NAME)
    with pytest.raises(ToolRegistryError):
        tool.validate_args({"schedule": "nonsense", "prompt": "x"})
