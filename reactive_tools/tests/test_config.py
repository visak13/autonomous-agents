"""Unit coverage for the SMTP config loader (s5/a1).

Proves :func:`reactive_tools.config.load_smtp_config`:
- reads the 5 SMTP_* vars from a temp ``.env`` into a typed ``SmtpConfig``;
- raises a clear ``SmtpConfigError`` (naming the key, never a value) on a
  missing required var;
- coerces ``SMTP_PORT`` to int and rejects a non-numeric port;
- never leaks the password in ``repr`` (no secret to logs/tracebacks).

No env vars are mutated globally — each test points the loader at its own temp
file, and the os.environ-fallback test scopes its change with monkeypatch.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from reactive_tools.config import (
    SmtpConfig,
    SmtpConfigError,
    load_smtp_config,
)


def _write_env(tmp_path: Path, body: str) -> Path:
    env = tmp_path / ".env"
    env.write_text(body, encoding="utf-8")
    return env


_FULL = (
    "# a comment line\n"
    "SMTP_HOST=smtp.example.com\n"
    "SMTP_PORT=587\n"
    'SMTP_USERNAME="user@example.com"\n'
    "SMTP_PASSWORD=app-secret-xyz\n"
    "export SMTP_FROM_EMAIL=user@example.com\n"
    "UNRELATED=ignored\n"
)


def test_loads_all_five_vars(tmp_path: Path):
    env = _write_env(tmp_path, _FULL)
    cfg = load_smtp_config(env)
    assert isinstance(cfg, SmtpConfig)
    assert cfg.host == "smtp.example.com"
    assert cfg.port == 587 and isinstance(cfg.port, int)
    assert cfg.username == "user@example.com"   # quotes stripped
    assert cfg.password == "app-secret-xyz"
    assert cfg.from_email == "user@example.com"  # 'export ' prefix tolerated


def test_missing_var_raises_naming_the_key(tmp_path: Path):
    body = _FULL.replace("SMTP_PASSWORD=app-secret-xyz\n", "")
    env = _write_env(tmp_path, body)
    with pytest.raises(SmtpConfigError, match="SMTP_PASSWORD"):
        load_smtp_config(env)


def test_blank_var_is_treated_as_missing(tmp_path: Path):
    body = _FULL.replace("SMTP_HOST=smtp.example.com\n", "SMTP_HOST=\n")
    env = _write_env(tmp_path, body)
    with pytest.raises(SmtpConfigError, match="SMTP_HOST"):
        load_smtp_config(env)


def test_non_numeric_port_raises(tmp_path: Path):
    body = _FULL.replace("SMTP_PORT=587\n", "SMTP_PORT=not-a-number\n")
    env = _write_env(tmp_path, body)
    with pytest.raises(SmtpConfigError, match="SMTP_PORT"):
        load_smtp_config(env)


def test_os_environ_is_the_fallback(tmp_path: Path, monkeypatch):
    # .env supplies 4 of 5; the 5th comes from the environment.
    env = _write_env(tmp_path, _FULL.replace(
        "export SMTP_FROM_EMAIL=user@example.com\n", ""))
    monkeypatch.setenv("SMTP_FROM_EMAIL", "fallback@example.com")
    cfg = load_smtp_config(env)
    assert cfg.from_email == "fallback@example.com"


def test_password_redacted_in_repr(tmp_path: Path):
    env = _write_env(tmp_path, _FULL)
    cfg = load_smtp_config(env)
    text = repr(cfg)
    assert "app-secret-xyz" not in text
    assert "<redacted>" in text
