"""Unit coverage for the send_mail agentic tool (s2/a4) — NO real network.

The send path is exercised entirely in-process: either through a tiny in-memory
:class:`MailAdapter` stub, or through the default
:class:`SmtpAppPasswordAdapter` over a monkeypatched ``smtplib.SMTP``. The tests
prove the action's load-bearing properties:

- **Structural recipient lock (the named o1 safety bar):** the exposed args
  model (and thus the registry's structured-selection schema) has NO ``to``
  field — the model cannot express a recipient.
- A smuggled ``to`` in the raw args is dropped by the registry's
  ``validate_args`` and never reaches the channel.
- ``send_mail`` is ONE registry entry — selectable (in the enum) and
  dispatchable (through the hook) after a single ``register_send_mail``.
- The default adapter sends to ``SMTP_FROM_EMAIL`` only, over the existing
  SMTP+App-Password channel (reuse, d7), and never leaks the password.
- The adapter is swappable: an injected backend is used verbatim.
"""
from __future__ import annotations

import asyncio
import smtplib

import pytest

from reactive_tools import EventPlane
from reactive_tools.config import SmtpConfig
from reactive_tools.tool_hook import ToolHook
from reactive_tools.tool_registry import GrowableToolRegistry, ToolRegistryError
from reactive_tools.send_mail_tool import (
    MailAdapter,
    SendMailArgs,
    SmtpAppPasswordAdapter,
    make_send_mail_tool,
    register_send_mail,
)


_CFG = SmtpConfig(
    host="smtp.example.com",
    port=587,
    username="user@example.com",
    password="app-secret-xyz",
    from_email="user@example.com",
)


class _RecordingAdapter(MailAdapter):
    """In-memory swappable backend — records the send and returns a canned ok."""

    def __init__(self) -> None:
        self.sends: list[dict[str, str]] = []

    def send(self, *, subject: str, body: str) -> dict:
        self.sends.append({"subject": subject, "body": body})
        return {"ok": True, "to": "user@example.com", "subject": subject,
                "via": "recording-adapter"}


# --------------------------------------------------------------------------- #
# fake smtplib (reused shape from the email-tool tests) for the default adapter
# --------------------------------------------------------------------------- #


class _FakeSMTP:
    instances: list["_FakeSMTP"] = []

    def __init__(self, host=None, port=None, timeout=None, **kwargs):
        self.host, self.port, self.timeout = host, port, timeout
        self.calls: list[str] = []
        self.sent_message = None
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def ehlo(self, *a, **k):
        return (250, b"ok")

    def starttls(self, *a, **k):
        self.calls.append("starttls")
        return (220, b"ready")

    def login(self, username, password):
        self.calls.append("login")
        self.logged_in_with = (username, password)
        return (235, b"ok")

    def data(self, msg):
        return (250, b"2.0.0 OK accepted")

    def send_message(self, msg, *a, **k):
        self.sent_message = msg
        self.data(msg.as_string() if hasattr(msg, "as_string") else msg)
        return {}  # no refused recipients


@pytest.fixture
def fake_smtp(monkeypatch):
    _FakeSMTP.instances = []
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    return _FakeSMTP


# --------------------------------------------------------------------------- #
# STRUCTURAL recipient lock — the exposed schema has NO recipient field
# --------------------------------------------------------------------------- #


def test_args_model_has_no_recipient_field():
    fields = set(SendMailArgs.model_fields)
    assert fields == {"subject", "body"}
    assert "to" not in fields
    # the JSON schema the model is shown likewise carries no recipient key
    schema = SendMailArgs.model_json_schema()
    props = set(schema.get("properties", {}))
    assert props == {"subject", "body"}
    assert "to" not in props
    assert set(schema.get("required", [])) == {"subject", "body"}


def test_tool_args_schema_offered_has_no_recipient():
    tool = make_send_mail_tool(_RecordingAdapter())
    props = set(tool.args_schema().get("properties", {}))
    assert "to" not in props and props == {"subject", "body"}
    # the handler signature carries no `to` parameter either (defence in depth)
    import inspect
    params = set(inspect.signature(tool.handler).parameters)
    assert "to" not in params


def test_smuggled_recipient_is_dropped_before_handler():
    """A model that tries args={"subject","body","to"} cannot reach the channel:
    the registry's Pydantic validation strips the unknown ``to`` key."""
    adapter = _RecordingAdapter()
    tool = make_send_mail_tool(adapter)
    validated = tool.validate_args(
        {"subject": "s", "body": "b", "to": "attacker@evil.com"})
    assert validated == {"subject": "s", "body": "b"}
    assert "to" not in validated


# --------------------------------------------------------------------------- #
# ONE registry entry — selectable + dispatchable after a single register call
# --------------------------------------------------------------------------- #


def test_register_send_mail_is_one_entry_selectable_and_dispatchable():
    hook = ToolHook(EventPlane())
    registry = GrowableToolRegistry(hook)
    register_send_mail(registry, _RecordingAdapter())
    # selectable: appears by name in the offered set + selection enum
    assert "send_mail" in registry.names()
    enum = registry.selection_schema()["properties"]["tool"]["enum"]
    assert "send_mail" in enum
    # dispatchable: invocable through the bound hook
    result = asyncio.run(hook.invoke("send_mail", subject="hi", body="there"))
    assert result.ok is True
    assert result.value["ok"] is True


def test_dispatch_uses_injected_adapter():
    hook = ToolHook(EventPlane())
    registry = GrowableToolRegistry(hook)
    adapter = _RecordingAdapter()
    register_send_mail(registry, adapter)
    asyncio.run(hook.invoke("send_mail", subject="S", body="B"))
    assert adapter.sends == [{"subject": "S", "body": "B"}]


# --------------------------------------------------------------------------- #
# default adapter — SMTP+App-Password channel, recipient locked to SMTP_FROM
# --------------------------------------------------------------------------- #


def test_default_adapter_sends_to_self_only(fake_smtp):
    adapter = SmtpAppPasswordAdapter(_CFG)
    res = adapter.send(subject="Brief", body="hello")
    assert res["ok"] is True
    # locked recipient == the configured own address
    assert res["to"] == "user@example.com"
    msg = fake_smtp.instances[0].sent_message
    assert msg["To"] == "user@example.com"
    assert msg["From"] == "user@example.com"
    # the real STARTTLS+login channel was exercised (reuse, d7)
    assert "starttls" in fake_smtp.instances[0].calls
    assert fake_smtp.instances[0].logged_in_with == ("user@example.com", "app-secret-xyz")


def test_default_tool_end_to_end_locks_recipient(fake_smtp):
    """Even invoked through the hook with only subject+body, the message goes to
    SMTP_FROM_EMAIL — there is no path to set another recipient."""
    hook = ToolHook(EventPlane())
    registry = GrowableToolRegistry(hook)
    register_send_mail(registry, config=_CFG)
    result = asyncio.run(hook.invoke("send_mail", subject="s", body="b"))
    assert result.ok is True
    assert result.value["to"] == "user@example.com"
    assert fake_smtp.instances[0].sent_message["To"] == "user@example.com"


def test_password_never_leaks_in_result(fake_smtp):
    adapter = SmtpAppPasswordAdapter(_CFG)
    res = adapter.send(subject="s", body="b")
    assert "app-secret-xyz" not in repr(res)


def test_auth_failure_returns_structured_error_no_secret(monkeypatch):
    class _AuthFail(_FakeSMTP):
        def login(self, username, password):
            raise smtplib.SMTPAuthenticationError(535, b"5.7.8 bad creds")

    monkeypatch.setattr(smtplib, "SMTP", _AuthFail)
    adapter = SmtpAppPasswordAdapter(_CFG)
    res = adapter.send(subject="s", body="b")
    assert res["ok"] is False and res["error_type"] == "auth"
    assert "app-secret-xyz" not in repr(res)


# --------------------------------------------------------------------------- #
# input validation surfaces through the schema (blank subject rejected)
# --------------------------------------------------------------------------- #


def test_blank_subject_rejected_by_schema():
    tool = make_send_mail_tool(_RecordingAdapter())
    with pytest.raises(ToolRegistryError):
        tool.validate_args({"subject": "", "body": "b"})
