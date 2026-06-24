"""send_mail — the agentic mail TOOL (s2/a4), one growable-registry entry.

This is the ``send_mail`` capability the local Gemma agent calls through the
s2/a1 :class:`~reactive_tools.tool_registry.GrowableToolRegistry`: adding it is
exactly ONE :class:`~reactive_tools.tool_registry.ToolDef`. It is *channel-only*
— the agent writes the ``subject`` and ``body``; the **recipient is hard-locked
to the user's own address** (``SMTP_FROM_EMAIL``) and is NOT something the model
can choose.

Two design rules carry the whole action:

1. **STRUCTURAL recipient lock (the named safety bar for outcome o1).** The
   exposed args model (:class:`SendMailArgs`) has ONLY ``subject`` + ``body`` —
   there is *no* ``to`` field. The structured-selection schema the registry
   derives from this Pydantic model (``args_model.model_json_schema()``) therefore
   carries no recipient key at all, so the local model *cannot even express* a
   recipient. The lock is not a runtime ``if`` the model could argue around — the
   field simply does not exist on the wire. Defence-in-depth: the handler takes no
   ``to`` kwarg either, and the registry's ``validate_args`` drops any unknown key,
   so a smuggled ``args={"to": ...}`` is stripped before the handler is reached.

2. **SWAPPABLE adapter (decision d1 — growable registry; d7 — reuse, don't
   rebuild).** The tool sends through a small :class:`MailAdapter` seam, so the
   backend can change later (e.g. a Gmail-API adapter) WITHOUT reworking any node
   or the tool wiring. The default :class:`SmtpAppPasswordAdapter` reuses the
   repo's existing, Round-2-proven SMTP + Gmail App-Password channel
   (:func:`reactive_tools.email_tool.make_send_email` +
   :func:`reactive_tools.config.load_smtp_config`) — it does not re-implement
   smtplib/STARTTLS (d7). The adapter *always* targets ``SMTP_FROM_EMAIL`` and
   never forwards a recipient.

Decisions honored
------------------
- d1  — one growable-registry entry (a ``ToolDef``) on the existing engine; the
  send path is behind a swappable adapter interface (no new framework).
- d7  — SMTP + Gmail App-Password, reusing the existing working channel
  (``email_tool``/``config``); NO Gmail-API migration, NO channel rebuild.
- spec (Local Gemma + Ollama) — universal standards: outbound call already has a
  bounded timeout (the channel's ``DEFAULT_SMTP_TIMEOUT``); no secret is ever
  placed in a return value or error; failures surface as a structured
  ``ok=False`` dict, never a silent pass and never the password.

The LIVE self-send + the recipient-lock proof on real mail are exercised in a6;
this module ships the tool *registered, callable, and with no recipient field in
its exposed schema*, proven offline (fake SMTP, no live network).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field

from .config import SmtpConfig
from .email_tool import make_send_email
from .tool_registry import GrowableToolRegistry, ToolDef


# --------------------------------------------------------------------------- #
# Exposed args — subject + body ONLY (the structural recipient lock)
# --------------------------------------------------------------------------- #


class SendMailArgs(BaseModel):
    """The arguments the model fills for ``send_mail`` — **no recipient field**.

    Only ``subject`` and ``body`` are exposed. Because the registry derives the
    structured-selection schema from this model, the local LLM has no ``to`` key
    to populate: the recipient lock to ``SMTP_FROM_EMAIL`` is enforced by the
    *absence* of the field, not by a runtime check the model could route around.

    Unknown keys are intentionally NOT ``forbid``-den: the registry's
    ``validate_args`` drops them (the Pydantic default), matching the engine's
    "a slightly-off small-model payload still yields clean kwargs" contract. A
    smuggled ``to`` is therefore stripped — never honoured, never a hard error —
    and could not reach the handler regardless (it takes no ``to`` parameter).
    """

    subject: str = Field(..., min_length=1,
                         description="the email subject line (the agent writes this)")
    body: str = Field(..., description="the plain-text email body (the agent writes this)")


# --------------------------------------------------------------------------- #
# Swappable mail backend — a thin seam so the channel can change later
# --------------------------------------------------------------------------- #


class MailAdapter(ABC):
    """A swappable mail backend.

    The tool talks ONLY to this seam, so a different backend (e.g. a future
    Gmail-API adapter) can be dropped in without touching the tool, the registry,
    or any node (d1 growability). The contract is deliberately tiny and
    recipient-free: an adapter is handed a ``subject`` + ``body`` and is
    responsible for delivering to the user's own locked address. It must NEVER
    accept or honour a caller-supplied recipient.
    """

    @abstractmethod
    def send(self, *, subject: str, body: str) -> dict[str, Any]:
        """Deliver ``subject``/``body`` to the locked own-address.

        Returns a structured result dict (``ok`` plus channel detail on success,
        or ``ok=False`` + a category on failure). Must never raise on an ordinary
        delivery failure and must never include a secret in the result."""
        raise NotImplementedError


class SmtpAppPasswordAdapter(MailAdapter):
    """Default backend: SMTP + Gmail App-Password, reusing the proven channel (d7).

    Delegates to :func:`reactive_tools.email_tool.make_send_email`, the existing
    Round-2 channel (stdlib ``smtplib`` STARTTLS + login, bounded timeout, secret
    never leaked). It is called WITHOUT a ``to`` argument, so the channel's
    send-to-self default targets ``SMTP_FROM_EMAIL`` — the recipient lock. When
    ``config`` is ``None`` the creds load lazily per send (so the tool registers
    without a populated ``.env`` and only errors at call time).
    """

    def __init__(self, config: Optional[SmtpConfig] = None) -> None:
        # Build the bound channel callable once; it loads creds lazily per call
        # when config is None (see make_send_email).
        self._send_email = make_send_email(config)

    def send(self, *, subject: str, body: str) -> dict[str, Any]:
        # No `to` passed → the channel locks the recipient to SMTP_FROM_EMAIL.
        # We do NOT forward any recipient here; the lock is intrinsic to the call.
        return self._send_email(subject=subject, body=body)


# --------------------------------------------------------------------------- #
# The send_mail tool — built on the adapter, recipient hard-locked
# --------------------------------------------------------------------------- #

SEND_MAIL_NAME = "send_mail"
SEND_MAIL_DESCRIPTION = (
    "Email the user's own inbox. Use ONLY when the user EXPLICITLY asks to be "
    "emailed; otherwise deliver the result in chat or via file_write, never by "
    "email. You write only the subject and body; the recipient is fixed to the "
    "user's own address and cannot be set. Returns a structured result proving "
    "the send."
)


def make_send_mail_handler(adapter: MailAdapter):
    """Build the ``send_mail`` handler bound to ``adapter``.

    The handler uses ONLY ``subject`` + ``body``; there is no ``to`` parameter, so
    even a payload that slipped a recipient past the schema could not reach the
    channel. It returns the adapter's structured result dict.

    DEFENSE-IN-DEPTH (s3/b5): the live runtime dispatches a node's tool via
    ``hook.invoke(node.tool, **tool_args)`` — which does NOT run the registry's
    Pydantic ``validate_args`` (that drops unknown keys). So the handler itself
    absorbs and DISCARDS any extra kwarg (``**_locked_out``): a smuggled
    ``to=...`` on that bypassed path is silently ignored — never forwarded, never
    a hard failure that would break a legitimate send — and the adapter still
    targets ``SMTP_FROM_EMAIL``. The recipient lock thus holds on EVERY path."""

    def send_mail(subject: str, body: str, **_locked_out: Any) -> dict[str, Any]:
        """Send ``subject``/``body`` to the locked own-address via the adapter.

        ``**_locked_out`` captures and ignores any extra kwarg (e.g. a smuggled
        ``to``) so the recipient can never be steered, on any dispatch path."""
        return adapter.send(subject=subject, body=body)

    return send_mail


def make_send_mail_tool(adapter: Optional[MailAdapter] = None,
                        *, config: Optional[SmtpConfig] = None) -> ToolDef:
    """Build the ``send_mail`` :class:`ToolDef` — *adding the tool is this object*.

    ``adapter`` is the swappable mail backend; when omitted a
    :class:`SmtpAppPasswordAdapter` (SMTP + App-Password, the proven channel) is
    built (passing ``config`` through, else lazy per-call load). The returned
    def's ``args_model`` is :class:`SendMailArgs` (subject + body only) so the
    exposed schema carries no recipient field."""
    backend = adapter if adapter is not None else SmtpAppPasswordAdapter(config)
    return ToolDef(
        name=SEND_MAIL_NAME,
        description=SEND_MAIL_DESCRIPTION,
        args_model=SendMailArgs,
        handler=make_send_mail_handler(backend),
    )


def register_send_mail(registry: GrowableToolRegistry,
                       adapter: Optional[MailAdapter] = None,
                       *, config: Optional[SmtpConfig] = None) -> ToolDef:
    """Register ``send_mail`` on ``registry`` — the single growth point (d1).

    After this call ``send_mail`` is selectable (in the registry's selection
    enum) and dispatchable (through the bound hook, events on the plane). Returns
    the registered :class:`ToolDef`."""
    return registry.add(make_send_mail_tool(adapter, config=config))


__all__ = [
    "SendMailArgs",
    "MailAdapter",
    "SmtpAppPasswordAdapter",
    "make_send_mail_handler",
    "make_send_mail_tool",
    "register_send_mail",
    "SEND_MAIL_NAME",
    "SEND_MAIL_DESCRIPTION",
]
