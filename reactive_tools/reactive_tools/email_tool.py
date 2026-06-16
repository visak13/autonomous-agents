"""send_email — a Claude-style outbound email tool on the GLOBAL tool hook.

This is the EMAIL capability behind the Scenario-B workflow specs (a "daily
brief" spec composes a research/write task with a *schedule* and an *email
delivery channel*). Like every other tool it is a plain callable registered by
name onto the single :class:`~reactive_tools.tool_hook.ToolHook`, so each send
(and its result) flows on the event plane and is reachable by EVERY agent / LLM
call — one global registry, no per-agent scoping (d12).

Decisions honored
-----------------
- d2  — purely in-process, stdlib only. Mail goes out over the tool's OWN
  outbound SMTP socket via stdlib :mod:`smtplib` + :mod:`email.message`; no
  broker/pool, no subprocess, no third-party email library.
- d7  — SMTP creds come from ReactiveAgents' OWN ``.env``/config (the a1
  :func:`reactive_tools.config.load_smtp_config`), COPIED from mcp-service — no
  live cross-repo coupling.
- d8  — no shell-command anything; smtplib only.
- d12 — registered onto the GLOBAL hook so any agent/LLM call can use it.

SAFE-TEST DISCIPLINE (load-bearing)
-----------------------------------
``to`` DEFAULTS to the config's own ``SMTP_FROM_EMAIL`` (the user's own
address) when omitted, so the later safe self-test (Scenario B) sends mail to
self and needs no recipient argument. This action does NOT send live mail — the
real send is proven separately under the safe-test action.

SECURITY
--------
The SMTP password is NEVER logged, echoed, or placed in the returned dict / any
error message. Auth and connection failures return a clear *structured* error
(``ok=False`` with a category), never a silent pass and never a secret. The
config object itself redacts its password in ``repr`` (see a1 config.py).
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .config import SmtpConfig, load_smtp_config

# How long (seconds) to wait on the SMTP socket before giving up. Bounded so a
# wedged server can never hang the single in-process event loop (the tool body
# already runs off-loop via the hook's asyncio.to_thread, but a bound is still
# correct hygiene).
DEFAULT_SMTP_TIMEOUT = 30.0


def _capturing_smtp_class() -> type:
    """Return a subclass of the *current* :class:`smtplib.SMTP` that records the
    ``(code, message)`` the server returns for the DATA command.

    Built at call time off ``smtplib.SMTP`` so a test that monkeypatches
    ``smtplib.SMTP`` is transparently subclassed too (the capture then exercises
    the fake's ``data`` path). ``send_message`` raises ``SMTPDataError`` unless
    the server returned a 250, so a clean return already *implies* 250 — this
    subclass simply surfaces the real code/text for the evidence dict instead of
    assuming it."""

    base = smtplib.SMTP

    class _CapturingSMTP(base):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.captured_code: Optional[int] = None
            self.captured_message: Optional[str] = None

        def data(self, msg: Any) -> Any:  # noqa: D401 - mirrors smtplib.SMTP.data
            code, resp = super().data(msg)
            self.captured_code = code
            self.captured_message = (
                resp.decode("ascii", "replace") if isinstance(resp, (bytes, bytearray))
                else str(resp)
            )
            return code, resp

    return _CapturingSMTP


def _add_attachments(msg: EmailMessage,
                     attachments: Optional[Sequence[Any]]) -> None:
    """Attach each item. An item is either a path string (read from disk) or a
    mapping ``{filename, content, maintype?, subtype?}`` (content is str or
    bytes). Defaults to ``application/octet-stream`` when the type is unknown."""
    for att in attachments or []:
        if isinstance(att, Mapping):
            filename = att.get("filename") or "attachment"
            content = att.get("content", b"")
            maintype = att.get("maintype", "application")
            subtype = att.get("subtype", "octet-stream")
            data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        else:  # treat as a path on disk
            p = Path(str(att))
            filename = p.name
            data = p.read_bytes()
            maintype, subtype = "application", "octet-stream"
        msg.add_attachment(data, maintype=maintype, subtype=subtype,
                           filename=filename)


def make_send_email(config: Optional[SmtpConfig] = None):
    """Build the ``send_email`` callable bound to ``config``.

    When ``config`` is ``None`` the SMTP config is loaded LAZILY on each call
    (via :func:`load_smtp_config`). That lets the tool be REGISTERED onto the
    global hook unconditionally — so ``send_email`` always appears in
    ``hook.registry.names()`` — without requiring a populated ``.env`` at
    hook-build time (``build_default_hook`` has many callers that pass none).
    A missing-cred situation then surfaces as a structured error at *call* time,
    not as a missing tool."""

    def send_email(*, subject: str, body: str, to: Optional[str] = None,
                   html: Optional[str] = None,
                   attachments: Optional[Sequence[Any]] = None) -> dict[str, Any]:
        """Send an email via stdlib smtplib over SMTP STARTTLS.

        ``to`` defaults to the config's own ``SMTP_FROM_EMAIL`` (send-to-self)
        when omitted — the safe self-test needs no recipient. ``subject`` and
        ``body`` are required (plain-text body); ``html`` adds an HTML
        alternative part; ``attachments`` is an optional list (see
        :func:`_add_attachments`).

        On success returns a structured dict::

            {ok, to, subject, accepted, refused, smtp_code, smtp_message,
             message_id, bytes}

        capturing the server's 250 result so a caller can PROVE the send. On an
        auth/connection/SMTP error returns ``{ok: False, error, error_type,
        to, subject}`` — a clear error, never a silent pass and never the
        password.
        """
        cfg = config if config is not None else load_smtp_config()
        recipient = to or cfg.from_email

        if not isinstance(subject, str) or not subject:
            return {"ok": False, "error_type": "input",
                    "error": "subject must be a non-empty string",
                    "to": recipient, "subject": subject}
        if not isinstance(body, str):
            return {"ok": False, "error_type": "input",
                    "error": "body must be a string",
                    "to": recipient, "subject": subject}

        msg = EmailMessage()
        msg["From"] = cfg.from_email
        msg["To"] = recipient
        msg["Subject"] = subject
        message_id = make_msgid()
        msg["Message-ID"] = message_id
        msg.set_content(body)
        if html:
            msg.add_alternative(html, subtype="html")
        try:
            _add_attachments(msg, attachments)
        except (OSError, ValueError) as exc:
            return {"ok": False, "error_type": "attachment",
                    "error": f"{type(exc).__name__}: {exc}",
                    "to": recipient, "subject": subject}

        raw_bytes = len(msg.as_bytes())
        smtp_cls = _capturing_smtp_class()
        try:
            with smtp_cls(cfg.host, cfg.port, timeout=DEFAULT_SMTP_TIMEOUT) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                smtp.login(cfg.username, cfg.password)
                refused = smtp.send_message(msg) or {}
                captured_code = getattr(smtp, "captured_code", None)
                captured_message = getattr(smtp, "captured_message", None)
        except smtplib.SMTPAuthenticationError as exc:
            # NB: exc carries an SMTP code + a server message, NOT the password.
            return {"ok": False, "error_type": "auth",
                    "error": f"SMTP authentication failed (code {exc.smtp_code})",
                    "smtp_code": exc.smtp_code,
                    "to": recipient, "subject": subject}
        except (smtplib.SMTPException,) as exc:
            return {"ok": False, "error_type": "smtp",
                    "error": f"{type(exc).__name__}: {exc}",
                    "to": recipient, "subject": subject}
        except (OSError,) as exc:  # connection refused / DNS / TLS socket error
            return {"ok": False, "error_type": "connection",
                    "error": f"{type(exc).__name__}: {exc}",
                    "to": recipient, "subject": subject}

        # send_message returns a (possibly empty) dict of REFUSED recipients;
        # empty => every recipient accepted. send_message would have raised
        # SMTPDataError on a non-250 DATA response, so reaching here means the
        # server accepted the message (250). Prefer the captured real code/text;
        # fall back to 250 on the proven success path.
        accepted = [recipient] if recipient not in refused else []
        smtp_code = captured_code if captured_code is not None else 250
        return {
            "ok": True,
            "to": recipient,
            "subject": subject,
            "accepted": accepted,
            "refused": list(refused.keys()),
            "smtp_code": smtp_code,
            "smtp_message": captured_message,
            "message_id": message_id,
            "bytes": raw_bytes,
        }

    return send_email


def register_email_tool(hook: Any, config: Optional[SmtpConfig] = None) -> Any:
    """Register ``send_email`` onto ``hook`` (a :class:`ToolHook`).

    ``config`` is an optional pre-loaded :class:`SmtpConfig`; when ``None`` the
    creds are loaded lazily per call (so the tool registers without an ``.env``
    present and only errors at send time). After this call ``send_email`` is in
    ``hook.registry.names()`` and shows in ``/health`` ``components.tools``
    (d12). Returns the hook for chaining."""
    hook.register(
        "send_email",
        make_send_email(config),
        description=(
            "Send an email via SMTP STARTTLS (stdlib smtplib). Args: subject, "
            "body (required); to (defaults to the configured own address), html, "
            "attachments. Returns the 250/result dict so a send can be proven."
        ),
    )
    return hook


__all__ = [
    "make_send_email",
    "register_email_tool",
    "DEFAULT_SMTP_TIMEOUT",
]
