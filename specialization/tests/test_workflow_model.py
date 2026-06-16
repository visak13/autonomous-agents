"""Tests for the optional workflow-spec model fields (schedule + delivery).

The load-bearing property is BACK-COMPAT: an ordinary (pre-s5) spec must
round-trip byte-identically with NO new frontmatter keys, while a workflow spec
carries its schedule/delivery and reconstructs them exactly.
"""
from __future__ import annotations

from specialization.model import (
    CompiledSpec,
    DeliverySpec,
    ScheduleSpec,
    parse_compiled_spec,
)


def test_ordinary_spec_roundtrips_with_no_new_keys() -> None:
    spec = CompiledSpec(
        name="markdown-writer",
        description="output-shaping ruleset",
        source="seed",
        body="Structure findings with a heading and bullets.",
        created_at="2026-06-13T00:00:00+00:00",
    )
    md = spec.to_markdown()
    # No workflow keys leak into an ordinary spec's frontmatter (back-compat).
    assert "schedule_" not in md
    assert "delivery_" not in md
    back = parse_compiled_spec(md)
    assert back.schedule is None
    assert back.delivery is None
    assert back.is_workflow is False
    assert back.name == spec.name
    assert back.body == spec.body


def test_workflow_spec_roundtrips_schedule_and_delivery() -> None:
    spec = CompiledSpec(
        name="daily-brief",
        description="a daily brief",
        source="ui",
        body="Markdown sections + summary.",
        created_at="2026-06-13T00:00:00+00:00",
        schedule=ScheduleSpec(kind="interval", interval_seconds=120.0, max_fires=3),
        delivery=DeliverySpec(channel="email", recipient="me@example.com"),
    )
    md = spec.to_markdown()
    assert "schedule_kind: interval" in md
    assert "schedule_interval_seconds: 120.0" in md
    assert "schedule_max_fires: 3" in md
    assert "delivery_channel: email" in md
    assert "delivery_recipient: me@example.com" in md

    back = parse_compiled_spec(md)
    assert back.is_workflow is True
    assert back.schedule == spec.schedule
    assert back.delivery == spec.delivery


def test_one_shot_send_to_self_defaults() -> None:
    spec = CompiledSpec(
        name="one-shot-brief", description="d", source="seed", body="b",
        created_at="2026-06-13T00:00:00+00:00",
        schedule=ScheduleSpec(kind="one_shot"),
        delivery=DeliverySpec(channel="email"),  # recipient None => self
    )
    md = spec.to_markdown()
    # recipient None is NOT serialised (so it stays send-to-self on parse).
    assert "delivery_recipient" not in md
    assert "schedule_max_fires" not in md
    back = parse_compiled_spec(md)
    assert back.schedule.kind == "one_shot"
    assert back.delivery.recipient is None


def test_invalid_schedule_and_delivery_rejected() -> None:
    for bad in (
        lambda: ScheduleSpec(kind="weekly"),
        lambda: ScheduleSpec(kind="interval", interval_seconds=0),
        lambda: DeliverySpec(channel="sms"),
    ):
        try:
            bad()
            assert False, "expected ValueError"
        except ValueError:
            pass
