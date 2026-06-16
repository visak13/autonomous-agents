"""Workflow-spec fire path — schedule a "daily brief"-style spec and deliver it.

This is the GLUE that the generic :class:`reactive_tools.scheduler.Scheduler`
(timing + event plane only) and the recipient-locked ``send_mail`` tool meet at.
It lives in
``chat_app`` deliberately: chat_app already composes reactive_tools (the hook +
scheduler) AND specialization (the :class:`~specialization.CompiledSpec` workflow
model), so the upward dependency stays here and the scheduler/spec layers below
remain free of each other (d10).

What it does (Scenario B, s5)
-----------------------------
When a scheduled WORKFLOW SPEC fires, the scheduler invokes a callback that:

1. PRODUCES a report for the spec (its task output, shaped by the spec body).
   For s5 this is the deterministic STUB producer (``stub_produce_report``) — NO
   live phi, to avoid shared-GPU contention (d12); the full live report is s8.
   The producer is pluggable, so s8 swaps in a real agent run with no glue change.
2. DELIVERS the report through the spec's delivery channel. For ``email`` it
   calls the recipient-LOCKED ``send_mail`` tool via the hook — the recipient is
   hard-locked to ``SMTP_FROM_EMAIL`` (the user's own address), so an unattended
   scheduler-fired send can never reach an arbitrary recipient (d8, b5).

SAFETY (load-bearing, d8): nothing here AUTO-STARTS a real recurring send. A
caller must explicitly :func:`schedule_workflow_spec`; the build-time wiring
constructs the scheduler with NO jobs. The safe self-test schedules an immediate
one-shot (or a short bounded interval) to self — never a real 24h daily blast.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Awaitable, Callable, MutableSequence, Optional

from reactive_tools import EventPlane, ScheduledJob, Scheduler, ToolHook
from specialization import CompiledSpec, SpecRegistry

# A report producer: given the workflow spec, return the report text (or an
# awaitable of it). The default is the deterministic stub; s9 supplies the LIVE
# one (:func:`make_live_report_producer`) that runs a real Gemma-4 agentic run.
ReportProducer = Callable[[CompiledSpec], Any]


def stub_produce_report(spec: CompiledSpec) -> str:
    """Deterministic stub report for a workflow spec (s5, no live phi — d12).

    Produces a small markdown brief that references the spec's identity + body
    so a delivered email is recognisably "this spec's report", without any GPU
    call. s8 replaces this with a real agent run (same signature)."""
    fired_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    shaping = (spec.body or "").strip()
    shaping_note = shaping.splitlines()[0] if shaping else "(no output-shaping body)"
    return (
        f"# {spec.name}\n\n"
        f"_{spec.description}_\n\n"
        f"Generated at {fired_at} UTC by the in-process scheduler (stub report, s5).\n\n"
        f"## Summary\n\n"
        f"This is the scheduled workflow report for **{spec.name}**. The live "
        f"phi-authored report arrives at s8; for now this deterministic stub "
        f"proves the schedule -> produce -> deliver path end to end.\n\n"
        f"## Applied shaping\n\n"
        f"- ruleset: {shaping_note}\n"
    )


def _subject_for(spec: CompiledSpec) -> str:
    """The delivered email subject for a workflow spec's report."""
    return f"[ReactiveAgents] {spec.name}"


def make_workflow_fire(
    spec: CompiledSpec,
    hook: ToolHook,
    *,
    produce_report: Optional[ReportProducer] = None,
) -> Callable[[], Awaitable[dict[str, Any]]]:
    """Build the async ``fire`` callback for a workflow spec.

    The callback produces the report (stub by default) and, if the spec carries
    an ``email`` delivery channel, sends it via the recipient-LOCKED ``send_mail``
    tool on ``hook`` (so the send rides the event plane like any other tool call,
    d12; the recipient is hard-locked to ``SMTP_FROM_EMAIL`` — d8/b5).
    Returns a structured dict so the scheduler's ``scheduler_job_completed``
    event — and a caller awaiting the run — can PROVE what happened."""
    producer = produce_report or stub_produce_report

    async def fire() -> dict[str, Any]:
        outcome = producer(spec)
        # Support both a plain string and an awaitable producer (s8 live run).
        if hasattr(outcome, "__await__"):
            report = await outcome  # type: ignore[assignment]
        else:
            report = outcome
        report_text = str(report)

        result: dict[str, Any] = {
            "spec": spec.name,
            "report_chars": len(report_text),
            "delivered": False,
            "channel": None,
        }
        delivery = spec.delivery
        if delivery is not None and delivery.channel == "email":
            # d8 (the unattended-email safety invariant, b5): deliver via the
            # recipient-HARD-LOCKED ``send_mail`` tool — NOT the legacy free-``to``
            # ``send_email``. ``send_mail`` exposes only subject + body and its
            # adapter always targets ``SMTP_FROM_EMAIL``, so an UNATTENDED,
            # scheduler-fired (s6 cron) send can NEVER reach an arbitrary
            # recipient even if the spec's delivery.recipient was model-derived.
            # The spec's ``recipient`` is intentionally ignored: the only legal
            # destination is the user's own locked address.
            tool_result = await hook.invoke(
                "send_mail",
                subject=_subject_for(spec),
                body=report_text,
            )
            send = tool_result.value if tool_result.ok else None
            result["channel"] = "email"
            # send_mail returns {ok, to, smtp_code, message_id, ...} (``to`` is the
            # locked own-address); surface it so a scheduler-fired send is PROVABLE
            # (smtp 250 + the Message-ID the received copy carries) from the job's
            # last_result alone.
            result["delivered"] = bool(tool_result.ok and isinstance(send, dict) and send.get("ok"))
            result["to"] = (send or {}).get("to") if isinstance(send, dict) else None
            result["smtp_code"] = (send or {}).get("smtp_code") if isinstance(send, dict) else None
            result["message_id"] = (send or {}).get("message_id") if isinstance(send, dict) else None
            if not tool_result.ok:
                result["error"] = tool_result.error
            elif isinstance(send, dict) and not send.get("ok"):
                result["error"] = send.get("error")
        return result

    return fire


def _brief_topic(spec: CompiledSpec) -> str:
    """Derive a REAL, researchable topic for a workflow spec's live brief.

    The chat-defined workflow spec carries the brief's INTENT in its
    name/description (e.g. "daily-tech-brief" / "a concise daily brief of notable
    AI & software developments"). The live producer hands that to the planner as
    the research topic; the spec's markdown body shapes the written form."""
    subject = (spec.description or "").strip() or spec.name.replace("-", " ").strip()
    return subject[:400]


def make_live_report_producer(
    *,
    transport: Any,
    registry: SpecRegistry,
    hook: ToolHook,
    ledger: Optional[MutableSequence[dict[str, Any]]] = None,
) -> ReportProducer:
    """Build the LIVE report producer the s9 workflow fire uses (replaces the stub).

    This is the producer ``chat_app.workflow``'s module docstring promised — "s8
    replaces this stub with a real agent run (same signature)". When a workflow
    spec fires, it runs the SAME proven :func:`chat_app.agentic.run_agentic` live
    Gemma-4 path Scenario A uses: the real planner self-derives a research → write
    DAG, the real :class:`~agent_runtime.AgentRuntime` drives it on the live
    transport, and the whole run is traced (``agent.session → planner.plan →
    agent.run → agent.node`` + ``llm.chat`` Gemma-4 spans) by the GLOBAL tracer
    provider the app stood up at startup — so the FIRED run appears in the Phoenix
    ``reactive-agents`` project with Gemma-4 spans (Scenario-B requirement 4).

    Each fire gets a fresh per-fire :class:`EventPlane` (no SSE consumer is
    attached to a scheduler-fired run; tracing is global, so spans still export)
    and a unique ``run_id`` threaded into ``run_agentic`` so the trace's
    ``agent.session``/``agent.run`` spans carry it and are correlatable to the
    fire. The returned markdown brief is what the workflow's email delivers.

    ``ledger`` is an optional append-only list the route exposes at
    ``GET /workflows``: each fire records ``{run_id, topic, fired_at,
    report_chars, ok}`` so the scheduler-fired run_ids are observable for the
    Phoenix correlation without parsing the email body."""
    # Imported here (function-local) so the module has no hard import-time
    # dependency on the live agentic stack (keeps ``workflow`` importable in the
    # offline test harness, mirrors the lazy-seam discipline elsewhere).
    from chat_app.agentic import both_specs_registered, run_agentic

    async def produce(spec: CompiledSpec) -> str:
        topic = _brief_topic(spec)
        run_id = f"brief-{uuid.uuid4().hex[:12]}"
        fired_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not both_specs_registered(registry):
            raise RuntimeError(
                "live brief producer requires the markdown-writer + html-writer "
                "specialists to be registered (the writer rulesets that shape the "
                "report); register them before arming a live workflow"
            )
        plane = EventPlane()  # per-fire plane; global tracer still exports spans
        agentic = await run_agentic(
            topic,
            transport=transport,
            registry=registry,
            hook=hook,
            plane=plane,
            run_id=run_id,
        )
        report = (agentic.md_report or agentic.html_report or "").strip()
        if ledger is not None:
            ledger.append(
                {
                    "run_id": run_id,
                    "topic": topic,
                    "fired_at": fired_at,
                    "report_chars": len(report),
                    "ok": bool(agentic.ok),
                }
            )
        # Compose the delivered brief: the workflow spec's identity + the live
        # researched, markdown-shaped report + a provenance footer carrying the
        # run_id (so a received email is traceable back to its Phoenix run).
        return (
            f"# {spec.name}\n\n"
            f"_{spec.description}_\n\n"
            f"{report}\n\n"
            f"---\n"
            f"_Generated live at {fired_at} by the in-process scheduler via a "
            f"Gemma-4 agentic run (run_id `{run_id}`). Workflow spec applied: "
            f"`{spec.name}`._\n"
        )

    return produce


def workflow_job_from_spec(
    spec: CompiledSpec,
    hook: ToolHook,
    *,
    produce_report: Optional[ReportProducer] = None,
) -> ScheduledJob:
    """Build (but do NOT schedule) a :class:`ScheduledJob` from a workflow spec.

    Maps the spec's :class:`~specialization.ScheduleSpec` onto the generic job
    fields and binds the produce->deliver fire callback. Raises ``ValueError`` if
    the spec carries no schedule (it is not a workflow spec)."""
    sched = spec.schedule
    if sched is None:
        raise ValueError(f"spec {spec.name!r} has no schedule; not a workflow spec")
    fire = make_workflow_fire(spec, hook, produce_report=produce_report)
    return ScheduledJob(
        fire=fire,
        kind=sched.kind,
        interval_seconds=sched.interval_seconds,
        max_fires=sched.max_fires,
        initial_delay=sched.initial_delay,
        name=spec.name,
    )


def schedule_workflow_spec(
    scheduler: Scheduler,
    spec: CompiledSpec,
    hook: ToolHook,
    *,
    produce_report: Optional[ReportProducer] = None,
) -> str:
    """Schedule a workflow spec on ``scheduler`` and return the job id.

    This is the one EXPLICIT entrypoint that arms a scheduled send — there is no
    auto-start path (d8 safety). The safe self-test calls this with a one-shot /
    short-interval spec delivering to self."""
    job = workflow_job_from_spec(spec, hook, produce_report=produce_report)
    return scheduler.schedule(job)


__all__ = [
    "ReportProducer",
    "stub_produce_report",
    "make_live_report_producer",
    "make_workflow_fire",
    "workflow_job_from_spec",
    "schedule_workflow_spec",
]
