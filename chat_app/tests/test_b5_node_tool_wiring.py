"""b5 — node→tool wiring + the d8 unattended-email safety invariant (s3/b5).

Deterministic, model-free assertions for the two halves of b5:

1. NODE→TOOL OFFER — the planner offers the FULL six-bucket s2 node→tool surface
   (``web_search``, ``web_fetch``, ``file_read``, ``file_write``, the
   recipient-LOCKED ``send_mail``, and ``cron_add`` / ``cron_list`` /
   ``cron_delete``), so a node can ANSWER via tools rather than raw LLM
   auto-completion. ``register_agentic_tools`` composes them onto the SAME hook;
   ``chat_app.agentic.OFFERED_TOOLS`` + ``build_plan_schema`` put exactly those
   names into the planner's structured-output tool enum.

2. d8 — THE UNATTENDED-EMAIL SAFETY INVARIANT — a node may reach ONLY the
   recipient-hard-locked ``send_mail``. The legacy free-``to`` ``send_email`` is
   (a) NOT registered on the hook (``build_default_hook`` no longer wires it),
   (b) NOT in the offered tool enum, and (c) unreachable via any dispatch path —
   and the locked ``send_mail`` delivers to ``SMTP_FROM_EMAIL`` REGARDLESS of any
   smuggled ``to``, proven over a fake SMTP (no live network).

Everything here is in-process and offline: no live model, no real SMTP.
"""
from __future__ import annotations

import asyncio
import smtplib

import pytest

from reactive_tools import (
    AGENTIC_TOOL_NAMES,
    EventPlane,
    ToolHook,
    build_default_hook,
    register_agentic_tools,
)
from reactive_tools.config import SmtpConfig
from reactive_tools.send_mail_tool import SendMailArgs

from agent_runtime.toolargs import TOOL_ARG_SCHEMAS

from chat_app.agentic import OFFERED_TOOLS, build_plan_schema


# The six s2 buckets the mandate requires offered to nodes.
_SIX_S2_TOOLS = {
    "web_search",
    "web_fetch",
    "file_read",
    "file_write",
    "send_mail",
    "cron_add",
    "cron_list",
    "cron_delete",
}


# =========================================================================== #
# 1) NODE→TOOL OFFER — all six s2 buckets offered (incl. the locked send_mail)
# =========================================================================== #


def test_register_agentic_tools_offers_all_six_s2_buckets(tmp_path):
    hook = build_default_hook(EventPlane(), file_base=tmp_path)
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    names = set(hook.registry.names())
    # every s2 node→tool is registered + the canonical name set agrees
    assert _SIX_S2_TOOLS <= names
    assert set(AGENTIC_TOOL_NAMES) == _SIX_S2_TOOLS


def test_offered_tools_includes_six_s2_buckets():
    # the planner's offer list carries every s2 bucket (plus read-only observability)
    assert _SIX_S2_TOOLS <= set(OFFERED_TOOLS)


def test_plan_schema_tool_enum_offers_locked_send_mail_not_send_email(tmp_path):
    hook = build_default_hook(EventPlane(), file_base=tmp_path)
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    offered = [t["name"] for t in hook.registry.catalog() if t["name"] in OFFERED_TOOLS]
    schema = build_plan_schema(["spec-a"], offered)
    tool_enum = schema["properties"]["nodes"]["items"]["properties"]["tool"]["enum"]
    # the locked mail tool IS selectable; the free-`to` legacy tool is NOT
    assert "send_mail" in tool_enum
    assert "send_email" not in tool_enum
    assert _SIX_S2_TOOLS <= set(tool_enum)


# =========================================================================== #
# 2) d8 — the legacy free-`to` send_email is unreachable by the model
# =========================================================================== #


def test_build_default_hook_does_not_register_send_email(tmp_path):
    hook = build_default_hook(EventPlane(), file_base=tmp_path)
    assert "send_email" not in hook.registry.names()


def test_agentic_wiring_never_exposes_send_email(tmp_path):
    hook = build_default_hook(EventPlane(), file_base=tmp_path)
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    # not on the hook, not in the offer list, not in the offered catalog
    assert "send_email" not in hook.registry.names()
    assert "send_email" not in OFFERED_TOOLS
    offered = [t["name"] for t in hook.registry.catalog() if t["name"] in OFFERED_TOOLS]
    assert "send_email" not in offered


def test_send_mail_schemas_carry_no_recipient_field():
    # The model is shown NO `to` key anywhere on the send_mail path: not in the
    # tool's exposed Pydantic args, and not in the runtime's arg-emission schema.
    assert "to" not in SendMailArgs.model_fields
    assert set(SendMailArgs.model_fields) == {"subject", "body"}
    emit = TOOL_ARG_SCHEMAS["send_mail"]
    assert "to" not in emit["properties"]
    assert set(emit["properties"]) == {"subject", "body"}
    assert set(emit["required"]) == {"subject", "body"}


# --------------------------------------------------------------------------- #
# the core proof: the model cannot emit an arbitrary recipient via the runtime's
# node→tool dispatch path (`hook.invoke(node.tool, **tool_args)`), even with a
# smuggled `to` — the recipient stays locked to SMTP_FROM_EMAIL.
# --------------------------------------------------------------------------- #

_CFG = SmtpConfig(
    host="smtp.example.com",
    port=587,
    username="owner@example.com",
    password="app-secret-never-leak",
    from_email="owner@example.com",
)


class _FakeSMTP:
    """Minimal stand-in for ``smtplib.SMTP`` — records the sent message; no I/O."""

    instances: list["_FakeSMTP"] = []

    def __init__(self, host=None, port=None, timeout=None, **kwargs):
        self.host, self.port, self.timeout = host, port, timeout
        self.sent_message = None
        _FakeSMTP.instances.append(self)

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
def fake_smtp(monkeypatch):
    _FakeSMTP.instances = []
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    return _FakeSMTP


def test_node_dispatch_clean_args_locks_recipient_to_self(fake_smtp, tmp_path):
    """A normal send_mail node (subject+body) delivers to SMTP_FROM_EMAIL."""
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path,
                           smtp_config=_CFG)
    # exactly what runtime._SubAgent does: hook.invoke(node.tool, **tool_args)
    res = asyncio.run(hook.invoke("send_mail", subject="Brief", body="hello"))
    assert res.ok is True
    assert res.value["to"] == "owner@example.com"
    assert fake_smtp.instances[-1].sent_message["To"] == "owner@example.com"


def test_node_dispatch_smuggled_recipient_is_ignored_and_send_still_locks(fake_smtp, tmp_path):
    """The model CANNOT emit an arbitrary recipient: a smuggled ``to`` in the
    node's tool_args (the bypassed runtime dispatch path that does NOT run the
    registry's Pydantic validate_args) is silently discarded — the send still
    succeeds and still goes to SMTP_FROM_EMAIL, never the attacker address."""
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path,
                           smtp_config=_CFG)
    res = asyncio.run(
        hook.invoke("send_mail", subject="s", body="b", to="attacker@evil.com")
    )
    # the smuggled recipient neither leaks nor breaks the send
    assert res.ok is True
    assert res.value["to"] == "owner@example.com"
    sent = fake_smtp.instances[-1].sent_message
    assert sent["To"] == "owner@example.com"
    assert "attacker@evil.com" not in str(sent)


def test_send_mail_password_never_leaks(fake_smtp, tmp_path):
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path,
                           smtp_config=_CFG)
    res = asyncio.run(hook.invoke("send_mail", subject="s", body="b"))
    assert "app-secret-never-leak" not in repr(res.value)
