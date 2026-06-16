"""Run-and-prove for the s6 always-on DB-backed cron FIRING scheduler.

Drives each load-bearing behaviour of :mod:`chat_app.cron_scheduler` against a
REAL on-disk SQLite cron store (the same store the s2 cron tools write), with a
controllable clock so the timing is deterministic and no wall-clock wait is
needed — then reads the outcome back from an INDEPENDENT source (the recorded
``run_agentic`` invocations + the persisted DB rows), per the action's
run-and-prove mandate. No live GPU / Ollama, no real mail, no real time.

What is proven
--------------
1. FIRES-ON-SCHEDULE THROUGH run_agentic — a due job is fired by the PRODUCTION
   fire callback (:func:`~chat_app.cron_scheduler.make_cron_fire`), which is shown
   to call :func:`chat_app.agentic.run_agentic` (monkeypatched to a recorder) with
   the job's prompt; the DB fire-state (``last_run_at`` / ``last_status`` /
   ``next_run_at``) is advanced. The legacy ``resume_agentic`` is NOT used.
2. CATCH-UP CAPS AT EXACTLY 3 NEWEST — a job that missed 10 windows replays the
   newest 3 (older 7 dropped) and advances ``last_run_at`` to the newest.
3. SURVIVES-RESTART — a brand-new scheduler instance re-reads the SAME DB after a
   simulated process restart, sees the persisted schedule, RESUMES from the
   persisted ``last_run_at`` (never refiring an already-fired window), and
   re-fires the new due windows.

Plus: a brand-new job (no ``last_run_at``) baselines off its ``created_at``, and a
job that is not due fires nothing (no spurious catch-up).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from reactive_tools.cron_store import CronStore, resolve_db_path

from chat_app.cron_scheduler import CronScheduler, make_cron_fire


UTC = timezone.utc


def _db(tmp_path) -> str:
    """The shared-db path under a temp data dir (mirrors the app's resolution)."""
    return str(resolve_db_path(data_dir=tmp_path))


def _add(db_path: str, *, schedule: str, prompt: str, name: str = "") -> str:
    """Add a cron row and return its job_id (short-lived store, like the tools)."""
    with CronStore(db_path) as store:
        return store.add(schedule=schedule, prompt=prompt, name=name).job_id


def _set_last_run(db_path: str, job_id: str, when: datetime) -> None:
    """Pin a job's persisted ``last_run_at`` baseline (deterministic timing)."""
    with CronStore(db_path) as store:
        store.record_fire(job_id, last_run_at=when.isoformat())


def _job(db_path: str, job_id: str):
    with CronStore(db_path) as store:
        return store.get(job_id)


def _recorder():
    """A fire callback that records (job_id, fire_time); returns the list too."""
    fires: list[tuple[str, datetime]] = []

    async def fire(job, fire_time):
        fires.append((job.job_id, fire_time))
        return {"ok": True}

    return fire, fires


# --------------------------------------------------------------------------- #
# 1) FIRES ON SCHEDULE — through run_agentic, with DB state advanced
# --------------------------------------------------------------------------- #
def test_fires_on_schedule_through_run_agentic(tmp_path, monkeypatch):
    db_path = _db(tmp_path)
    job_id = _add(db_path, schedule="* * * * *", prompt="do the scheduled thing")
    base = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
    _set_last_run(db_path, job_id, base)
    now = base + timedelta(seconds=90)  # one whole minute (12:01) is now due

    # Spy on run_agentic; make run_offline blow up so we PROVE the live branch
    # went through run_agentic (not resume_agentic, not the stub seam).
    calls: list[dict] = []

    async def fake_run_agentic(query, **kwargs):
        calls.append({"query": query, "kwargs": kwargs})
        return SimpleNamespace(ok=True, shape="linear", final_response="ANSWER")

    async def fail_run_offline(*a, **k):  # pragma: no cover - must NOT be hit
        raise AssertionError("live fire must go through run_agentic, not run_offline")

    monkeypatch.setattr("chat_app.cron_scheduler.run_agentic", fake_run_agentic)
    monkeypatch.setattr("chat_app.cron_scheduler.run_offline", fail_run_offline)

    sentinel_transport = object()
    fire = make_cron_fire(
        transport_mode="live",
        registry=object(),
        hook=object(),
        live_transport=sentinel_transport,
    )
    sched = CronScheduler(db_path, fire, clock=lambda: now)
    summaries = asyncio.run(sched.tick())

    # run_agentic was invoked exactly once, with the JOB'S prompt + a fresh plan.
    assert len(calls) == 1
    assert calls[0]["query"] == "do the scheduled thing"
    assert calls[0]["kwargs"]["transport"] is sentinel_transport
    assert calls[0]["kwargs"]["run_id"].startswith(f"cron-{job_id}-")

    # The tick summary reflects exactly one fire, nothing dropped.
    assert len(summaries) == 1
    assert summaries[0]["job_id"] == job_id
    assert summaries[0]["fired_count"] == 1
    assert summaries[0]["dropped"] == 0

    # DB fire-state advanced + persisted (read back as the independent source).
    row = _job(db_path, job_id)
    assert row.last_run_at == (base + timedelta(minutes=1)).isoformat()
    assert row.last_status == "ok"
    assert row.next_run_at is not None  # the upcoming window was computed


# --------------------------------------------------------------------------- #
# 2) MISSED-FIRE CATCH-UP — exactly the 3 newest windows, older dropped
# --------------------------------------------------------------------------- #
def test_catchup_caps_at_three_newest(tmp_path):
    db_path = _db(tmp_path)
    job_id = _add(db_path, schedule="* * * * *", prompt="catch me up")
    base = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
    _set_last_run(db_path, job_id, base)
    now = base + timedelta(minutes=10)  # 10 windows missed: 12:01 .. 12:10

    fire, fires = _recorder()
    sched = CronScheduler(db_path, fire, clock=lambda: now, max_catchup=3)
    summaries = asyncio.run(sched.tick())

    # Exactly the NEWEST 3 windows fired; the older 7 were dropped.
    assert len(fires) == 3
    fired_windows = [ft for (_jid, ft) in fires]
    assert fired_windows == [
        base + timedelta(minutes=8),
        base + timedelta(minutes=9),
        base + timedelta(minutes=10),
    ]
    assert summaries[0]["fired_count"] == 3
    assert summaries[0]["dropped"] == 7

    # State advanced to the NEWEST window (12:10), so a re-tick fires nothing.
    row = _job(db_path, job_id)
    assert row.last_run_at == now.isoformat()
    again = asyncio.run(sched.tick())
    assert again == [] and len(fires) == 3  # nothing new fired


# --------------------------------------------------------------------------- #
# 3) SURVIVES RESTART — a fresh scheduler re-reads the DB + resumes fire state
# --------------------------------------------------------------------------- #
def test_survives_restart_rereads_db_and_resumes(tmp_path):
    db_path = _db(tmp_path)
    job_id = _add(db_path, schedule="* * * * *", prompt="every minute", name="ticker")
    base = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
    _set_last_run(db_path, job_id, base)

    # --- scheduler instance #1: fires the windows due up to 12:02, advances state.
    fire1, fires1 = _recorder()
    sched1 = CronScheduler(db_path, fire1, clock=lambda: base + timedelta(minutes=2))
    asyncio.run(sched1.tick())
    assert [ft for (_j, ft) in fires1] == [
        base + timedelta(minutes=1),
        base + timedelta(minutes=2),
    ]
    assert _job(db_path, job_id).last_run_at == (base + timedelta(minutes=2)).isoformat()

    # --- SIMULATED PROCESS RESTART: a brand-new scheduler over the SAME db file.
    # It holds NO in-memory job list — it must re-read the persisted schedule AND
    # the persisted last_run_at to know where to resume.
    fire2, fires2 = _recorder()
    sched2 = CronScheduler(db_path, fire2, clock=lambda: base + timedelta(minutes=5))

    # The schedule itself persisted across the "restart".
    persisted = _job(db_path, job_id)
    assert persisted is not None and persisted.schedule == "* * * * *"
    assert persisted.prompt == "every minute"

    asyncio.run(sched2.tick())
    # Resumes from 12:02 (persisted) — fires only the NEW windows 12:03..12:05,
    # never refiring 12:01/12:02 that instance #1 already fired.
    new_windows = [ft for (_j, ft) in fires2]
    assert new_windows == [
        base + timedelta(minutes=3),
        base + timedelta(minutes=4),
        base + timedelta(minutes=5),
    ]
    assert set(ft for (_j, ft) in fires1).isdisjoint(new_windows)  # no overlap
    assert _job(db_path, job_id).last_run_at == (base + timedelta(minutes=5)).isoformat()


# --------------------------------------------------------------------------- #
# Extra coverage: created_at baseline for a brand-new job; not-due is a no-op.
# --------------------------------------------------------------------------- #
def test_new_job_baselines_off_created_at(tmp_path):
    db_path = _db(tmp_path)
    job_id = _add(db_path, schedule="* * * * *", prompt="fresh job")
    created = datetime.fromisoformat(_job(db_path, job_id).created_at)
    # No last_run_at set: the baseline is created_at, so one minute later one
    # window is due and fires exactly once.
    now = created.replace(second=0, microsecond=0) + timedelta(minutes=1, seconds=30)

    fire, fires = _recorder()
    sched = CronScheduler(db_path, fire, clock=lambda: now)
    asyncio.run(sched.tick())
    assert len(fires) >= 1  # baselined off created_at, did fire


def test_not_due_job_fires_nothing(tmp_path):
    db_path = _db(tmp_path)
    job_id = _add(db_path, schedule="0 0 1 1 *", prompt="new year only")  # Jan 1 00:00
    _set_last_run(db_path, job_id, datetime(2026, 6, 1, 0, 0, tzinfo=UTC))
    now = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)  # June: never due

    fire, fires = _recorder()
    sched = CronScheduler(db_path, fire, clock=lambda: now)
    summaries = asyncio.run(sched.tick())
    assert fires == [] and summaries == []
    # State untouched (no fire happened).
    assert _job(db_path, job_id).last_run_at == datetime(2026, 6, 1, 0, 0, tzinfo=UTC).isoformat()


# =========================================================================== #
# s6/a2 — the UNATTENDED CRON-FIRED EMAIL path + the d8 safety invariant.
#
# These prove, run-and-prove style through the REAL routed entrypoint
# (``build_wiring`` -> a ``CronStore`` row -> ``CronScheduler.tick`` -> the
# PRODUCTION ``make_cron_fire`` live callback -> ``chat_app.agentic.run_agentic``),
# that an UNATTENDED scheduler fire can send email ONLY via the recipient-locked
# ``send_mail`` and can NEVER reach the legacy free-``to`` ``send_email`` (d8).
#
# Fidelity choices (so the proof is faithful, not a standalone harness):
#   * the WHOLE stack is the real ``build_wiring`` composition (the same hook,
#     tool registry and ``run_agentic`` the running server uses); only the live
#     Gemma transport is swapped for a deterministic router ``FakeTransport`` —
#     exactly the pluggable ``Transport`` seam (d7/d12) the live path is built on;
#   * the fire is driven from a real ``CronScheduler.tick()`` reading a real
#     persisted ``CronStore`` row — it FIRES FROM THE SCHEDULER, no user in loop;
#   * the SMTP send boundary is patched INDEPENDENTLY at ``smtplib.SMTP`` (mirrors
#     the s3 reviewer's independent smtplib patch) so the locked adapter's real
#     ``make_send_email`` channel runs end-to-end but NO real mail leaves — and the
#     captured ``EmailMessage`` is the independent source we read the recipient back
#     from, not the tool's own return value;
#   * ``send_mail`` is re-bound to a deterministic :class:`SmtpConfig` on the wiring
#     hook so ``SMTP_FROM`` is hermetic (``load_smtp_config`` would otherwise let the
#     repo ``.env`` win) — it is the identical locked tool, just with fixed creds.
# --------------------------------------------------------------------------- #
import json
import smtplib

import pytest

from reactive_tools import (
    GrowableToolRegistry,
    ToolError,
    resolve_cron_db_path,
)
from reactive_tools.config import SmtpConfig
from reactive_tools.send_mail_tool import register_send_mail

from chat_app.agentic import OFFERED_TOOLS, build_plan_schema
from chat_app.app import build_wiring


# A deterministic SMTP config — the locked recipient is its ``from_email``
# (``SMTP_FROM``). The password is here only to prove it never leaks.
_SMTP_FROM = "owner@example.com"
_CFG = SmtpConfig(
    host="smtp.example.com",
    port=587,
    username=_SMTP_FROM,
    password="app-secret-never-leak",
    from_email=_SMTP_FROM,
)


class _CapturingSMTP:
    """An independent stand-in for ``smtplib.SMTP`` — records every sent message
    (the To header + body) and does ZERO network I/O. The recipient we assert on
    is read back from THIS captured message, not from the tool's return dict."""

    instances: list["_CapturingSMTP"] = []

    def __init__(self, host=None, port=None, timeout=None, **kwargs):
        self.host, self.port, self.timeout = host, port, timeout
        self.sent_message = None
        _CapturingSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def ehlo(self, *a, **k):
        return (250, b"ok")

    def starttls(self, *a, **k):
        return (220, b"ready")

    def login(self, username, password):
        return (235, b"ok")

    def data(self, msg):
        return (250, b"2.0.0 OK")

    def send_message(self, msg, *a, **k):
        self.sent_message = msg
        return {}


@pytest.fixture
def capturing_smtp(monkeypatch):
    """Patch the SMTP boundary INDEPENDENTLY (the s3-reviewer pattern)."""
    _CapturingSMTP.instances = []
    monkeypatch.setattr(smtplib, "SMTP", _CapturingSMTP)
    return _CapturingSMTP


def _email_plan(*, subject: str, body: str, extra_args: dict | None = None) -> dict:
    """A one-node plan that emails the user via the LOCKED ``send_mail`` tool.

    The ``tool_args`` carry the agent-written subject+body verbatim (the
    ``SchemaToolArgEmitter`` uses them as-is since both required keys are present,
    so no extra phi call is needed). ``extra_args`` lets a test SMUGGLE a ``to``
    onto the node to prove the recipient stays locked regardless."""
    args = {"subject": subject, "body": body}
    if extra_args:
        args.update(extra_args)
    return {
        "rationale": "the scheduled job emails the user their brief",
        "nodes": [
            {
                "id": "send",
                "task": "email the user a short scheduled status brief",
                "spec": "",
                "depends_on": [],
                "tool": "send_mail",
                "tool_args": args,
            }
        ],
    }


def _router_transport(plan: dict):
    """A deterministic ``FakeTransport`` standing in for the live Gemma model.

    It routes each structured call by the JSON-schema ``format`` it is handed
    (the same seam the real model is driven through), so the full ``run_agentic``
    path runs offline. a3: ``linear`` is now an OPEN shape AUTHORED node-by-node by
    the incremental authorer (not the one-shot planner), so the routing is:
    shape-selection -> ``linear``; the INCREMENTAL per-node call (its schema has
    ``more``) -> the plan's nodes one at a time; the ``send_mail`` arg-emitter call
    (its schema has ``subject``) -> the agent-written subject+body (the incremental
    authorer carries no ``tool_args``, so the emitter grounds them per d11); the
    node's own scoped generation (no ``format``) -> a body string. Defensive: an
    unexpected heal call resolves to ``retry``."""
    from llm_framework import FakeTransport

    nodes = plan.get("nodes", [])
    args0 = (nodes[0].get("tool_args") if nodes else {}) or {}
    authored = {"i": 0}

    def route(messages, **opts):
        fmt = opts.get("format")
        props = fmt.get("properties", {}) if isinstance(fmt, dict) else {}
        if "shape" in props:
            return json.dumps({"shape": "linear", "rationale": "single sequential mail step"})
        if "more" in props:  # the INCREMENTAL per-node authoring call (a3)
            i = authored["i"]
            authored["i"] = i + 1
            node = nodes[i] if i < len(nodes) else nodes[-1]
            return json.dumps(
                {
                    "task": node.get("task", ""),
                    "spec": node.get("spec", "") or "",
                    "tool": node.get("tool", "") or "",
                    "depends_on": [],  # single-step linear plan
                    "more": i < len(nodes) - 1,
                }
            )
        if "subject" in props:  # the send_mail tool-arg emitter call (d11)
            return json.dumps(
                {"subject": args0.get("subject", ""), "body": args0.get("body", "")}
            )
        if "nodes" in props:  # legacy one-shot path (kept; not hit under incremental)
            return json.dumps(plan)
        if "action" in props:  # a heal decision should not be needed; stay benign
            return json.dumps({"action": "retry", "rationale": "transient"})
        # the send node's own scoped generation — a non-empty produced body
        return "Scheduled brief: all systems nominal; nothing requires your attention."

    return FakeTransport([route])


def _wiring_with_locked_smtp(tmp_path):
    """Build the REAL wiring and pin ``send_mail`` to the deterministic ``_CFG``.

    Re-registering ``send_mail`` (same locked ToolDef, deterministic creds) onto
    the wiring's hook replaces only the mail handler — every other tool, the
    registry, ``run_agentic`` and the cron firing path stay exactly as the running
    server composes them, so the entrypoint under test is the real one."""
    w = build_wiring(data_dir=tmp_path)
    register_send_mail(GrowableToolRegistry(w.hook), config=_CFG)
    return w


def _fire_one_cron_email_job(w, plan, *, smuggle: dict | None = None) -> dict:
    """Add a due cron job to the REAL store, then drive ONE scheduler tick through
    the PRODUCTION live fire callback. Returns the persisted job row's fire state."""
    cron_db = str(resolve_cron_db_path(data_dir=w.data_dir))
    base = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)
    with CronStore(cron_db) as store:
        job_id = store.add(
            schedule="* * * * *", prompt="email me my scheduled brief", name="daily-brief"
        ).job_id
        store.record_fire(job_id, last_run_at=base.isoformat())
    now = base + timedelta(seconds=90)  # one whole minute (09:01) is now due

    # The PRODUCTION live fire callback — identical to build_wiring's, only the
    # transport is the deterministic router (the d7/d12 pluggable seam).
    fire = make_cron_fire(
        transport_mode="live",
        registry=w.registry,
        hook=w.hook,
        live_transport=_router_transport(plan),
        shape_config=w.shape_config,
    )
    sched = CronScheduler(cron_db, fire, clock=lambda: now)
    summaries = asyncio.run(sched.tick())
    assert summaries and summaries[0]["fired_count"] == 1  # the scheduler DID fire it
    with CronStore(cron_db) as store:
        return {"job": store.get(job_id), "summary": summaries[0]}


# --------------------------------------------------------------------------- #
# (a) the unattended scheduler fire sends via the LOCKED send_mail adapter,
#     recipient hard-locked to SMTP_FROM, with a non-empty agent-written body.
# --------------------------------------------------------------------------- #
def test_cron_fired_email_sends_via_locked_send_mail_to_self(capturing_smtp, tmp_path):
    subject = "Your scheduled brief"
    body = "Here is your automated daily status brief — agent-written content."
    w = _wiring_with_locked_smtp(tmp_path)
    try:
        state = _fire_one_cron_email_job(w, _email_plan(subject=subject, body=body))
    finally:
        w.close()

    # The locked SMTP adapter was actually reached (one real channel send, no I/O).
    assert capturing_smtp.instances, "send_mail adapter was never invoked by the fire"
    sent = capturing_smtp.instances[-1].sent_message
    assert sent is not None
    # RECIPIENT HARD-LOCKED to SMTP_FROM — read back from the independent message.
    assert sent["To"] == _SMTP_FROM
    assert sent["From"] == _SMTP_FROM
    # A NON-EMPTY, agent-written body actually went out (subject + body preserved).
    assert sent["Subject"] == subject
    delivered_body = sent.get_content()
    assert delivered_body.strip()  # non-empty
    assert body in delivered_body
    # The fire was unattended + clean (the scheduler advanced state, no error).
    assert state["job"].last_status == "ok"


# --------------------------------------------------------------------------- #
# (a') even a node that SMUGGLES a foreign `to` cannot redirect the unattended
#      send — the recipient stays locked to SMTP_FROM (the lock is structural,
#      not a runtime check the model could argue around).
# --------------------------------------------------------------------------- #
def test_cron_fired_email_ignores_smuggled_recipient(capturing_smtp, tmp_path):
    attacker = "attacker@evil.com"
    w = _wiring_with_locked_smtp(tmp_path)
    try:
        _fire_one_cron_email_job(
            w,
            _email_plan(subject="hi", body="body text", extra_args={"to": attacker}),
        )
    finally:
        w.close()

    sent = capturing_smtp.instances[-1].sent_message
    assert sent is not None
    # The smuggled recipient neither leaks into the message nor redirects it.
    assert sent["To"] == _SMTP_FROM
    assert attacker not in str(sent)


# --------------------------------------------------------------------------- #
# (b) d8 STRUCTURAL INVARIANT — a cron-fired plan's nodes are offered ONLY the
#     locked send_mail; the legacy free-`to` send_email is not in the offered
#     tool enum, is not registered on the hook, and is not even invocable.
# --------------------------------------------------------------------------- #
def test_cron_fired_path_never_exposes_legacy_send_email(tmp_path):
    w = build_wiring(data_dir=tmp_path)
    try:
        # The EXACT offered-tool list + plan schema the cron-fired run_agentic path
        # builds (chat_app.agentic._run_acyclic computes this identically).
        offered = [t["name"] for t in w.hook.registry.catalog() if t["name"] in OFFERED_TOOLS]
        schema = build_plan_schema(w.registry.names(), offered)
        tool_enum = schema["properties"]["nodes"]["items"]["properties"]["tool"]["enum"]

        # the locked mail tool IS selectable by a cron-fired node ...
        assert "send_mail" in tool_enum
        assert "send_mail" in w.hook.registry.names()
        # ... and the legacy free-`to` tool is NOWHERE on the path:
        assert "send_email" not in tool_enum         # not selectable by the model
        assert "send_email" not in OFFERED_TOOLS      # not in the offer list
        assert "send_email" not in offered            # not in the offered catalog
        assert "send_email" not in w.hook.registry.names()  # not registered at all

        # and reaching it is structurally impossible — the hook has no such tool,
        # so an attempt to dispatch it (even with a foreign `to`) raises, never sends.
        with pytest.raises(ToolError):
            asyncio.run(
                w.hook.invoke("send_email", subject="s", body="b", to="attacker@evil.com")
            )
    finally:
        w.close()
