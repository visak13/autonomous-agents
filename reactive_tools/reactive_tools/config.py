"""SMTP config loader — reads the 5 SMTP_* creds for ReactiveAgents email.

The email-sending workflow specs (Scenario B) need SMTP credentials. Those live
in the repo-root ``.env`` (gitignored), COPIED — not live-imported — from the
mcp-service repo, so there is no cross-repo coupling (a1, d7).

Decisions honored
-----------------
- d2  — purely in-process, stdlib only. The ``.env`` is parsed with a tiny
  hand-rolled stdlib parser (python-dotenv is NOT in the workspace), with an
  ``os.environ`` fallback per var. No new dependency, no shell, no subprocess.
- d8  — file I/O via ``pathlib``/``open`` only.

SECURITY
--------
Secret values (password especially) are NEVER printed or logged. ``SmtpConfig``
redacts the password in its ``repr`` so it cannot leak into a traceback/log line.
A missing required var raises a clear :class:`SmtpConfigError` naming ONLY the
key — never a value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# Repo root: config.py -> reactive_tools/ -> reactive_tools/ -> ReactiveAgents/
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = _REPO_ROOT / ".env"

# The 5 required keys (source key is SMTP_FROM_EMAIL, not FROM_EMAIL).
_REQUIRED_KEYS = (
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_USERNAME",
    "SMTP_PASSWORD",
    "SMTP_FROM_EMAIL",
)

# Tracing config keys. The agent_runtime.tracing factory (s6/a1) resolves the
# Phoenix OTLP endpoint + project name from ``os.environ`` so the process can be
# repointed without a code change. Unlike the SMTP creds — read on demand — these
# must be in ``os.environ`` BEFORE the tracer provider is first built, so they are
# bridged from ``.env`` into the environment at app startup (see load_tracing_env).
_TRACING_KEYS = (
    "REACTIVE_AGENTS_OTLP_ENDPOINT",
    "REACTIVE_AGENTS_PHOENIX_PROJECT",
)


class SmtpConfigError(RuntimeError):
    """Raised when a required SMTP var is missing or malformed (no secret in the message)."""


@dataclass(frozen=True)
class SmtpConfig:
    """Typed SMTP configuration. ``password`` is redacted in ``repr``."""

    host: str
    port: int
    username: str
    password: str
    from_email: str

    def __repr__(self) -> str:  # never leak the secret into logs/tracebacks
        return (
            "SmtpConfig(host={h!r}, port={p!r}, username={u!r}, "
            "password=<redacted>, from_email={f!r})".format(
                h=self.host, p=self.port, u=self.username, f=self.from_email
            )
        )


def _parse_env_file(path: Path) -> Dict[str, str]:
    """Tiny stdlib ``.env`` parser: ``KEY=VALUE`` per line.

    Skips blank lines and ``#`` comments, tolerates an optional ``export``
    prefix, splits on the FIRST ``=``, strips surrounding single/double quotes.
    Returns ``{}`` if the file does not exist (the ``os.environ`` fallback then
    applies).
    """
    if not path.is_file():
        return {}
    values: Dict[str, str] = {}
    # utf-8-sig tolerates a BOM if the file was saved by a Windows editor.
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key:
            values[key] = val
    return values


def load_smtp_config(env_path: Optional[Path | str] = None) -> SmtpConfig:
    """Load + validate the 5 SMTP_* creds into a typed :class:`SmtpConfig`.

    Resolution order per key: the ``.env`` file value wins, else ``os.environ``.
    Raises :class:`SmtpConfigError` (naming only the offending key, never a
    value) if a required var is missing/blank or ``SMTP_PORT`` is not an int.
    """
    path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    file_vals = _parse_env_file(path)

    resolved: Dict[str, str] = {}
    missing = []
    for key in _REQUIRED_KEYS:
        val = file_vals.get(key)
        if val is None or val == "":
            val = os.environ.get(key, "")
        if val == "":
            missing.append(key)
        else:
            resolved[key] = val

    if missing:
        raise SmtpConfigError(
            "Missing required SMTP config var(s): "
            + ", ".join(missing)
            + f" (looked in {path} and os.environ)"
        )

    try:
        port = int(resolved["SMTP_PORT"])
    except ValueError:
        raise SmtpConfigError(
            "SMTP_PORT must be an integer (got a non-numeric value)"
        ) from None

    return SmtpConfig(
        host=resolved["SMTP_HOST"],
        port=port,
        username=resolved["SMTP_USERNAME"],
        password=resolved["SMTP_PASSWORD"],
        from_email=resolved["SMTP_FROM_EMAIL"],
    )


def load_tracing_env(env_path: Optional[Path | str] = None) -> Dict[str, str]:
    """Bridge the ``.env`` tracing config keys into ``os.environ``.

    The :mod:`agent_runtime.tracing` factory builds the tracer provider from
    ``os.environ`` only (its proven defaults cover the unset case). Those keys
    must therefore be present in the environment BEFORE the provider is first
    built — which the app does eagerly at lifespan startup (s6/b3). This reads
    them from ``.env`` and sets them, with **a real ``os.environ`` value always
    winning** (standard dotenv semantics — an explicit env override is never
    clobbered by the file). Keys absent from both ``.env`` and the environment
    are left unset, so the factory falls back to its proven defaults.

    Idempotent and side-effect-only on the two tracing keys; never raises on a
    missing ``.env`` file. Returns the mapping it actually applied (for
    logging / introspection).
    """
    path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    file_vals = _parse_env_file(path)
    applied: Dict[str, str] = {}
    for key in _TRACING_KEYS:
        if os.environ.get(key):  # explicit env override wins — do not clobber
            continue
        val = file_vals.get(key)
        if val:
            os.environ[key] = val
            applied[key] = val
    return applied


__all__ = [
    "SmtpConfig",
    "SmtpConfigError",
    "load_smtp_config",
    "load_tracing_env",
    "DEFAULT_ENV_PATH",
]
