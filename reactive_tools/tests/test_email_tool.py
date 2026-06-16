"""Unit coverage for the send_email tool (s5/a2) — NO real network.

Every test monkeypatches ``smtplib.SMTP`` with an in-memory fake that records
the STARTTLS/login/send_message call sequence and returns a 250 from the DATA
command, so the message is built and "sent" entirely in-process. The tests
prove:

- the :class:`EmailMessage` is built correctly (From/To/Subject, plain body,
  optional HTML alternative) and ``to`` DEFAULTS to ``SMTP_FROM_EMAIL`` when
  omitted (the safe self-test needs no recipient);
- STARTTLS + login happen and the 250 DATA result is surfaced in the returned
  dict (``smtp_code``, ``accepted``, ``message_id``, ``bytes``);
- the password is NEVER present in the returned dict nor (on auth failure) in
  the error;
- auth / connection failures return a clear structured error, never a silent
  pass and never the secret;
- ``register_email_tool`` / ``build_default_hook`` put ``send_email`` into
  ``hook.registry.names()`` (d12 — global registry).
"""
from __future__ import annotations

import smtplib

import pytest

from reactive_tools import EventPlane, build_default_hook
from reactive_tools.config import SmtpConfig
from reactive_tools.email_tool import make_send_email, register_email_tool
from reactive_tools.tool_hook import ToolHook


_CFG = SmtpConfig(
    host="smtp.example.com",
    port=587,
    username="user@example.com",
    password="app-secret-xyz",
    from_email="user@example.com",
)


class _FakeSMTP:
    """In-memory stand-in for ``smtplib.SMTP`` — records the call sequence and
    captures the built message. Subclassable by the tool's capturing subclass.

    ``send_message`` routes through ``self.data`` (as real smtplib does via
    sendmail) so the tool's DATA-capture path is genuinely exercised, then
    returns an empty refused-recipients dict (every recipient accepted)."""

    instances: list["_FakeSMTP"] = []

    def __init__(self, host=None, port=None, timeout=None, **kwargs):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.calls: list[str] = []
        self.logged_in_with = None
        self.sent_message = None
        _FakeSMTP.instances.append(self)

    # context-manager protocol (the tool uses `with smtp_cls(...) as smtp:`)
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.calls.append("quit")
        return False

    def ehlo(self, *a, **k):
        self.calls.append("ehlo")
        return (250, b"ok")

    def starttls(self, *a, **k):
        self.calls.append("starttls")
        return (220, b"ready to start TLS")

    def login(self, username, password):
        self.calls.append("login")
        self.logged_in_with = (username, password)
        return (235, b"authentication succeeded")

    def data(self, msg):
        self.calls.append("data")
        return (250, b"2.0.0 OK message accepted")

    def send_message(self, msg, *a, **k):
        self.calls.append("send_message")
        self.sent_message = msg
        # mirror real smtplib: DATA is issued as part of the send
        self.data(msg.as_string() if hasattr(msg, "as_string") else msg)
        return {}  # no refused recipients


@pytest.fixture
def fake_smtp(monkeypatch):
    _FakeSMTP.instances = []
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    return _FakeSMTP


# --------------------------------------------------------------------------- #
# message building + `to` default
# --------------------------------------------------------------------------- #

def test_to_defaults_to_from_email_and_builds_message(fake_smtp):
    send_email = make_send_email(_CFG)
    res = send_email(subject="Daily Brief", body="hello world")
    # `to` omitted -> defaults to the config's own address (send-to-self)
    assert res["to"] == "user@example.com"
    msg = fake_smtp.instances[0].sent_message
    assert msg["From"] == "user@example.com"
    assert msg["To"] == "user@example.com"
    assert msg["Subject"] == "Daily Brief"
    assert msg.get_content().strip() == "hello world"


def test_explicit_recipient_is_used(fake_smtp):
    send_email = make_send_email(_CFG)
    res = send_email(subject="hi", body="b", to="someone@else.com")
    assert res["to"] == "someone@else.com"
    assert fake_smtp.instances[0].sent_message["To"] == "someone@else.com"


def test_html_alternative_is_attached(fake_smtp):
    send_email = make_send_email(_CFG)
    send_email(subject="s", body="plain", html="<p>rich</p>")
    msg = fake_smtp.instances[0].sent_message
    assert msg.is_multipart()
    subtypes = {part.get_content_subtype() for part in msg.iter_parts()}
    assert "html" in subtypes and "plain" in subtypes


# --------------------------------------------------------------------------- #
# STARTTLS + login + the 250 result dict
# --------------------------------------------------------------------------- #

def test_starttls_login_and_250_surfaced(fake_smtp):
    send_email = make_send_email(_CFG)
    res = send_email(subject="s", body="b")
    inst = fake_smtp.instances[0]
    # STARTTLS happened, login happened with the configured creds
    assert "starttls" in inst.calls
    assert inst.logged_in_with == ("user@example.com", "app-secret-xyz")
    # the 250 DATA result is captured and surfaced
    assert res["ok"] is True
    assert res["smtp_code"] == 250
    assert res["smtp_message"] == "2.0.0 OK message accepted"
    assert res["accepted"] == ["user@example.com"]
    assert res["refused"] == []
    assert res["message_id"].startswith("<") and res["message_id"].endswith(">")
    assert res["bytes"] > 0


# --------------------------------------------------------------------------- #
# password is never leaked
# --------------------------------------------------------------------------- #

def test_password_never_in_result(fake_smtp):
    send_email = make_send_email(_CFG)
    res = send_email(subject="s", body="b")
    assert "app-secret-xyz" not in repr(res)


def test_auth_failure_returns_structured_error_without_password(monkeypatch):
    class _AuthFailSMTP(_FakeSMTP):
        def login(self, username, password):
            raise smtplib.SMTPAuthenticationError(535, b"5.7.8 bad credentials")

    monkeypatch.setattr(smtplib, "SMTP", _AuthFailSMTP)
    send_email = make_send_email(_CFG)
    res = send_email(subject="s", body="b")
    assert res["ok"] is False
    assert res["error_type"] == "auth"
    assert res["smtp_code"] == 535
    # clear error, no secret
    assert "app-secret-xyz" not in repr(res)


def test_connection_error_returns_structured_error(monkeypatch):
    class _ConnRefusedSMTP(_FakeSMTP):
        def __init__(self, *a, **k):
            raise OSError("connection refused")

    monkeypatch.setattr(smtplib, "SMTP", _ConnRefusedSMTP)
    send_email = make_send_email(_CFG)
    res = send_email(subject="s", body="b")
    assert res["ok"] is False
    assert res["error_type"] == "connection"
    assert "connection refused" in res["error"]


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #

def test_blank_subject_rejected(fake_smtp):
    send_email = make_send_email(_CFG)
    res = send_email(subject="", body="b")
    assert res["ok"] is False and res["error_type"] == "input"
    # nothing was sent
    assert fake_smtp.instances == []


# --------------------------------------------------------------------------- #
# global registry wiring (d12)
# --------------------------------------------------------------------------- #

def test_register_email_tool_adds_to_registry():
    hook = ToolHook(EventPlane())
    register_email_tool(hook, _CFG)
    assert "send_email" in hook.registry.names()


def test_build_default_hook_does_not_register_send_email(tmp_path):
    # d8 (the unattended-email safety invariant, s3/b5): the legacy free-``to``
    # ``send_email`` is NO LONGER on the default hook — exposing a free-recipient
    # mail tool there would let a model emit an arbitrary recipient. The only mail
    # tool a node reaches is the recipient-locked ``send_mail`` (composed via
    # register_agentic_tools). ``send_email`` stays available as a direct callable
    # (test_register_email_tool_adds_to_registry) but is not wired by default.
    hook = build_default_hook(EventPlane(), file_base=tmp_path)
    assert "send_email" not in hook.registry.names()


def test_send_email_invokable_when_explicitly_registered(fake_smtp, tmp_path):
    # The capability still works when a host explicitly opts in via
    # register_email_tool (it is no longer auto-wired by build_default_hook).
    import asyncio

    hook = build_default_hook(EventPlane(), file_base=tmp_path)
    register_email_tool(hook, _CFG)
    result = asyncio.run(
        hook.invoke("send_email", subject="via-hook", body="b")
    )
    assert result.ok is True
    assert result.value["smtp_code"] == 250
    assert result.value["to"] == "user@example.com"
