"""Tests for the workflow-spec fire path (chat_app.workflow).

Proves the Scenario-B glue WITHOUT live phi and WITHOUT live SMTP:
- a scheduled workflow spec FIRES -> produces a (stub) report -> calls the
  recipient-LOCKED ``send_mail`` tool via the hook -> delivers to the user's own
  locked address (``SMTP_FROM_EMAIL``);
- the spec's ``recipient`` is IGNORED — d8 (the unattended-email safety
  invariant, b5): an unattended scheduler-fired send can never reach an arbitrary
  recipient, so every fire delivers to self regardless of what the spec asked;
- a spec with NO schedule is rejected as "not a workflow spec" (back-compat:
  ordinary specs never get scheduled);
- the send rides the real tool hook (so it appears on the event plane, d12).

The SMTP layer is monkeypatched at the hook level: we register a FAKE
``send_mail`` that records its kwargs and returns a 250-style dict, so NO real
network/mail happens (mirrors the a2 email-tool test discipline).
"""
from __future__ import annotations

import asyncio

from reactive_tools import EventPlane, Scheduler, ToolHook
from reactive_tools.scheduler import KIND_ONE_SHOT
from specialization import CompiledSpec, DeliverySpec, ScheduleSpec

from chat_app.workflow import (
    make_workflow_fire,
    schedule_workflow_spec,
    stub_produce_report,
    workflow_job_from_spec,
)


LOCKED_ADDRESS = "self@example.com"


def _hook_with_fake_email(plane: EventPlane) -> tuple[ToolHook, list[dict]]:
    """A ToolHook whose ``send_mail`` is a fake recorder (no network).

    The fake mirrors the REAL recipient-locked ``send_mail`` signature — only
    ``subject`` + ``body`` (NO ``to``) — so the test can never even express a
    recipient. The returned ``to`` is always the locked own-address."""
    hook = ToolHook(plane)
    sent: list[dict] = []

    def fake_send_mail(*, subject, body):
        record = {"subject": subject, "body": body, "to": LOCKED_ADDRESS}
        sent.append(record)
        # mirror the real locked tool's 250 success shape (recipient = locked self)
        return {"ok": True, "to": LOCKED_ADDRESS, "subject": subject,
                "smtp_code": 250, "message_id": "<fake@local>", "bytes": len(body)}

    hook.register("send_mail", fake_send_mail, description="fake")
    return hook, sent


def _workflow_spec(recipient=None) -> CompiledSpec:
    return CompiledSpec(
        name="daily-brief",
        description="a daily news brief, markdown-shaped",
        source="seed",
        body="Structure findings with a heading, bullets, and a short summary.",
        schedule=ScheduleSpec(kind="one_shot", initial_delay=0.0),
        delivery=DeliverySpec(channel="email", recipient=recipient),
    )


def test_stub_report_references_spec() -> None:
    spec = _workflow_spec()
    report = stub_produce_report(spec)
    assert "daily-brief" in report
    assert "daily news brief" in report
    assert report.lstrip().startswith("#")  # markdown heading


def test_fire_produces_report_and_sends_email() -> None:
    async def body() -> None:
        plane = EventPlane()
        hook, sent = _hook_with_fake_email(plane)
        spec = _workflow_spec(recipient="me@example.com")
        fire = make_workflow_fire(spec, hook)
        result = await fire()

        assert result["spec"] == "daily-brief"
        assert result["report_chars"] > 0
        assert result["delivered"] is True
        assert result["channel"] == "email"
        assert result["smtp_code"] == 250
        # exactly one email sent; d8 — even though the spec named a DIFFERENT
        # recipient ("me@example.com"), the locked send_mail delivers to self.
        assert len(sent) == 1
        assert sent[0]["to"] == LOCKED_ADDRESS
        assert "daily-brief" in sent[0]["body"]
        assert sent[0]["subject"] == "[ReactiveAgents] daily-brief"

    asyncio.run(body())


def test_spec_recipient_is_ignored_and_locked_to_self() -> None:
    # d8 (the unattended-email safety invariant, b5): the spec's recipient is
    # NEVER honoured — a fire always delivers to the user's own locked address,
    # whether the spec named none or an arbitrary one.
    async def body() -> None:
        plane = EventPlane()
        hook, sent = _hook_with_fake_email(plane)
        spec = _workflow_spec(recipient="attacker@evil.com")
        result = await make_workflow_fire(spec, hook)()
        assert result["delivered"] is True
        assert sent[0]["to"] == LOCKED_ADDRESS
        assert sent[0]["to"] != "attacker@evil.com"

    asyncio.run(body())


def test_custom_producer_is_used() -> None:
    async def body() -> None:
        plane = EventPlane()
        hook, sent = _hook_with_fake_email(plane)
        spec = _workflow_spec()

        async def live_like_producer(s: CompiledSpec) -> str:
            return f"LIVE REPORT for {s.name}"

        result = await make_workflow_fire(spec, hook, produce_report=live_like_producer)()
        assert result["delivered"] is True
        assert sent[0]["body"] == "LIVE REPORT for daily-brief"

    asyncio.run(body())


def test_non_workflow_spec_is_rejected() -> None:
    # An ordinary (schedule-less) spec must NOT be schedulable as a workflow.
    ordinary = CompiledSpec(
        name="markdown-writer", description="md ruleset", source="seed",
        body="use GFM",
    )
    plane = EventPlane()
    hook, _ = _hook_with_fake_email(plane)
    try:
        workflow_job_from_spec(ordinary, hook)
        assert False, "expected ValueError for a schedule-less spec"
    except ValueError as exc:
        assert "no schedule" in str(exc)


def test_schedule_workflow_spec_fires_through_scheduler() -> None:
    async def body() -> None:
        plane = EventPlane()
        hook, sent = _hook_with_fake_email(plane)
        sched = Scheduler(plane)
        sched.start()
        spec = _workflow_spec(recipient="self@example.com")

        job_id = schedule_workflow_spec(sched, spec, hook)
        # one-shot, no delay — give it time to fire end to end
        await asyncio.sleep(0.15)
        job = sched.get(job_id)
        assert job is not None
        assert job.fire_count == 1
        assert isinstance(job.last_result, dict)
        assert job.last_result["delivered"] is True
        assert len(sent) == 1
        assert sent[0]["to"] == LOCKED_ADDRESS
        await sched.shutdown()

    asyncio.run(body())
