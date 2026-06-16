"""Deterministic, model-free assertions for the TWO named safety bars (s2/a6).

These are the verify gate for the s2 "live end-to-end proof" action: the live
gemma4-e2b-agent run proves the tools are *callable*, but the safety bars must
hold regardless of any model and so are pinned here as fast, offline tests.

Safety bar 1 — file_write sandbox (a3, outcome o1)
    ``file_write`` REFUSES any path that resolves OUTSIDE the workspace sandbox
    root (``..``-traversal, absolute escape, and symlink/junction escape), and
    it refuses BEFORE any byte is written. Enforced in code (the audited
    ``_safe_resolve`` realpath+commonpath guard), not by prompt — so it cannot be
    argued around by a clever model.

Safety bar 2 — send_mail recipient lock (a4, outcome o1)
    ``send_mail``'s EXPOSED args schema has NO recipient field, so the model
    cannot even express a ``to``; a smuggled ``to`` is dropped before the handler;
    and the default SMTP adapter always targets ``SMTP_FROM_EMAIL`` (send-to-self).
    The lock is structural (field absence), not a runtime check.

Everything here is in-process and offline (no live model, no real SMTP/network):
the SMTP path is exercised over a monkeypatched ``smtplib.SMTP``.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import smtplib
from pathlib import Path

import pytest

from reactive_tools import EventPlane
from reactive_tools.config import SmtpConfig
from reactive_tools.tool_hook import ToolHook
from reactive_tools.tool_registry import GrowableToolRegistry
from reactive_tools.tools import ToolInputError
from reactive_tools.file_tools import (
    build_filesystem_tools,
    make_file_write,
    register_filesystem_tools,
    resolve_workspace_root,
)
from reactive_tools.send_mail_tool import (
    SendMailArgs,
    SmtpAppPasswordAdapter,
    make_send_mail_tool,
    register_send_mail,
)


# =========================================================================== #
# SAFETY BAR 1 — file_write hard sandbox
# =========================================================================== #


def _sandbox(tmp_path: Path) -> Path:
    """A realpath'd workspace root inside the test's tmp dir."""
    return resolve_workspace_root(tmp_path / "workspace")


def test_file_write_allows_path_inside_sandbox(tmp_path):
    """A normal relative path INSIDE the sandbox is written (the bar is not a
    blanket deny — it only refuses escapes)."""
    root = _sandbox(tmp_path)
    write = make_file_write(root)
    res = write(path="reports/out.md", content="inside the sandbox")
    written = Path(res["path"])
    assert written.is_file()
    assert written.read_text(encoding="utf-8") == "inside the sandbox"
    # the real location is genuinely under the real sandbox root
    assert os.path.commonpath([os.path.realpath(written), str(root)]) == str(root)


def test_file_write_refuses_dotdot_traversal(tmp_path):
    """A ``..`` traversal escape RAISES before any write and creates nothing
    outside the sandbox."""
    root = _sandbox(tmp_path)
    write = make_file_write(root)
    escape = tmp_path / "escaped_traversal.txt"
    with pytest.raises(ToolInputError):
        write(path="../escaped_traversal.txt", content="should never land")
    assert not escape.exists()


def test_file_write_refuses_absolute_escape(tmp_path):
    """An ABSOLUTE path outside the sandbox is refused before any write."""
    root = _sandbox(tmp_path)
    write = make_file_write(root)
    outside = tmp_path / "abs_escape.txt"
    with pytest.raises(ToolInputError):
        write(path=str(outside), content="should never land")
    assert not outside.exists()


def test_file_write_refuses_symlink_escape(tmp_path):
    """A symlink/junction whose real target is OUTSIDE the sandbox is refused:
    ``_safe_resolve`` realpath-follows the link before the containment check.

    Uses a directory symlink (falls back to a Windows junction when symlink
    creation needs privilege); the subcase is skipped only if neither link type
    can be created on the host — the realpath branch is the same one the ``..``
    and absolute cases also exercise, so the bar stays covered."""
    root = _sandbox(tmp_path)
    write = make_file_write(root)
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    link = root / "link_out"  # a path INSIDE the sandbox that points OUT
    try:
        os.symlink(outside_dir, link, target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        # No symlink privilege — try a directory junction (no privilege needed).
        rc = os.system(f'cmd /c mklink /J "{link}" "{outside_dir}" >NUL 2>&1')
        if rc != 0 or not link.exists():
            pytest.skip("host allows neither symlink nor junction creation")
    target = outside_dir / "evil.txt"
    with pytest.raises(ToolInputError):
        write(path="link_out/evil.txt", content="should never land outside")
    assert not target.exists()


def test_file_write_refusal_surfaces_through_dispatch(tmp_path):
    """Through the a1 registry/hook dispatch, an escaping write returns a
    structured ``ok=False`` (the refusal is observable, never a silent
    write-elsewhere)."""
    root = _sandbox(tmp_path)
    hook = ToolHook(EventPlane())
    registry = GrowableToolRegistry(hook)
    register_filesystem_tools(registry, root)
    result = asyncio.run(
        hook.invoke("file_write", path="../escape_via_dispatch.txt", content="x")
    )
    assert result.ok is False
    assert not (tmp_path / "escape_via_dispatch.txt").exists()


def test_file_write_def_is_built_for_the_sandbox_root(tmp_path):
    """Sanity: the built ToolDef pair is the file_read/file_write growable
    entries (one registry entry each)."""
    defs = build_filesystem_tools(_sandbox(tmp_path))
    names = {d.name for d in defs}
    assert names == {"file_read", "file_write"}


# =========================================================================== #
# SAFETY BAR 2 — send_mail recipient lock
# =========================================================================== #

_CFG = SmtpConfig(
    host="smtp.example.com",
    port=587,
    username="aksoulkar@gmail.com",
    password="app-secret-never-leak",
    from_email="aksoulkar@gmail.com",
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
        return {}  # no refused recipients


@pytest.fixture
def fake_smtp(monkeypatch):
    _FakeSMTP.instances = []
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    return _FakeSMTP


def test_send_mail_exposed_schema_has_no_recipient_field():
    """The structural lock: the model is shown a schema with ONLY subject+body."""
    fields = set(SendMailArgs.model_fields)
    assert fields == {"subject", "body"}
    assert "to" not in fields
    schema = SendMailArgs.model_json_schema()
    props = set(schema.get("properties", {}))
    assert props == {"subject", "body"}
    assert "to" not in props
    assert set(schema.get("required", [])) == {"subject", "body"}


def test_send_mail_tooldef_and_handler_have_no_recipient():
    """The ToolDef's offered arg schema AND the handler signature carry no ``to``."""
    tool = make_send_mail_tool(config=_CFG)
    props = set(tool.args_schema().get("properties", {}))
    assert props == {"subject", "body"} and "to" not in props
    params = set(inspect.signature(tool.handler).parameters)
    assert "to" not in params


def test_send_mail_smuggled_recipient_is_dropped(tmp_path):
    """A model that tries ``args={subject,body,to}`` cannot redirect mail: the
    registry's Pydantic validation strips the unknown ``to`` before the handler."""
    tool = make_send_mail_tool(config=_CFG)
    validated = tool.validate_args(
        {"subject": "s", "body": "b", "to": "attacker@evil.com"}
    )
    assert validated == {"subject": "s", "body": "b"}
    assert "to" not in validated


def test_send_mail_default_adapter_targets_smtp_from_only(fake_smtp):
    """The default SMTP adapter ALWAYS delivers to SMTP_FROM_EMAIL (send-to-self),
    proven over a fake SMTP — no recipient is ever forwarded."""
    adapter = SmtpAppPasswordAdapter(_CFG)
    res = adapter.send(subject="Brief", body="hello")
    assert res["ok"] is True
    assert res["to"] == "aksoulkar@gmail.com"
    msg = fake_smtp.instances[0].sent_message
    assert msg["To"] == "aksoulkar@gmail.com"
    assert msg["From"] == "aksoulkar@gmail.com"


def test_send_mail_through_dispatch_locks_recipient(fake_smtp):
    """Even invoked through the a1 hook with only subject+body, the message goes
    to SMTP_FROM_EMAIL — there is no path to set another recipient."""
    hook = ToolHook(EventPlane())
    registry = GrowableToolRegistry(hook)
    register_send_mail(registry, config=_CFG)
    # selectable + dispatchable as ONE entry
    assert "send_mail" in registry.names()
    assert "send_mail" in registry.selection_schema()["properties"]["tool"]["enum"]
    result = asyncio.run(hook.invoke("send_mail", subject="s", body="b"))
    assert result.ok is True
    assert result.value["to"] == "aksoulkar@gmail.com"
    assert fake_smtp.instances[0].sent_message["To"] == "aksoulkar@gmail.com"


def test_send_mail_password_never_leaks(fake_smtp):
    """The App-Password is never placed in the structured result."""
    adapter = SmtpAppPasswordAdapter(_CFG)
    res = adapter.send(subject="s", body="b")
    assert "app-secret-never-leak" not in repr(res)
