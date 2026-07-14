"""Transport layer for the standalone reactive agent.

A *transport* is the thin seam between the agent and a chat-completion model.
It exposes two methods:

- ``complete(messages, **opts) -> str``  — just the assistant text, for the
  common "give me the answer" call site.
- ``chat(messages, **opts) -> ChatResult`` — the full reply (role + content +
  raw provider payload), for call sites that need to inspect the role or the
  underlying response.

Two implementations ship here:

1. :class:`OllamaTransport` — the real runtime. Talks to a local Ollama at
   ``http://127.0.0.1:11434`` running the Gemma-4 edge **E4B** model
   ``gemma4-e4b-candidate-ctx32k`` (s8/b8 swap; was gemma4-e2b-agent at s8/b1,
   phi4-mini before that on :11435). It can drive
   EITHER the OpenAI-compatible endpoint (``/v1/chat/completions``) OR Ollama's
   native ``/api/chat`` (the native path also carries the ``think`` control —
   structured callers pass ``think=False``). It is ``keep_alive``-aware (d8 VRAM
   hygiene: keep the model warm across a burst, or unload promptly so it does not
   hog the shared GPU) and uses clean, explicit timeouts. ZERO Claude — pure
   local Ollama (d1).

2. :class:`FakeTransport` — a deterministic, scripted transport. You program
   it with a sequence of canned replies (strings, callables, or exceptions)
   and it returns/raises them in order. This lets the whole chain run fully
   OFFLINE with zero GPU use (d7/d8), including the important case of a
   malformed-JSON-then-valid-JSON sequence for testing parser self-heal.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

import httpx

# --------------------------------------------------------------------------- #
# Optional OpenTelemetry tracing (s6/b1)
# --------------------------------------------------------------------------- #
#
# Each phi call is instrumented as a CHILD OpenInference LLM span. Two design
# rules drive how the tracer is acquired here:
#
# 1. SINGLE SHARED PROVIDER, never a second one. agent_runtime.tracing (s6/a1)
#    builds the app's one TracerProvider AND registers it as the OpenTelemetry
#    *global* provider. We therefore attach to that global through
#    ``opentelemetry.trace.get_tracer(...)`` instead of importing
#    ``agent_runtime`` — a back-import would create a workspace dependency CYCLE
#    (agent_runtime already depends on llm_framework) and break this framework's
#    standalone / dependency-light (d10) identity.
#
# 2. OPTIONAL. If opentelemetry is not installed (this framework still runs on
#    httpx alone), tracing degrades to a no-op and ``chat()`` behaves exactly as
#    before. So the import is guarded.
try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import Status, StatusCode

    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - opentelemetry is an optional extra
    _OTEL_AVAILABLE = False

# OpenInference semantic-convention attribute keys. Hardcoded to the published
# ``openinference-semantic-conventions`` values so we do NOT pull that package
# into this lean framework (agent_runtime owns the heavyweight tracing deps).
_OI_SPAN_KIND = "openinference.span.kind"
_OI_LLM_KIND = "LLM"
_OI_MODEL_NAME = "llm.model_name"
_OI_INVOCATION_PARAMETERS = "llm.invocation_parameters"
_OI_TOKEN_PROMPT = "llm.token_count.prompt"
_OI_TOKEN_COMPLETION = "llm.token_count.completion"
_OI_TOKEN_TOTAL = "llm.token_count.total"
# Our own namespace for things OpenInference has no standard key for.
_ATTR_API_TYPE = "llm.provider.api_type"
_ATTR_LATENCY_MS = "llm.latency_ms"

# A chat message is a plain mapping {"role": ..., "content": ...}. We keep it
# as a dict alias rather than a model class to stay lean (d10) and to match
# both the OpenAI and Ollama wire shapes without translation.
Message = MutableMapping[str, Any]

# Runtime model swap (s8/b8, supersedes the gemma4-e2b-agent default of s8/b1 per
# d25/d31/d35). The app now drives Google's Gemma-4 edge **E4B** on the NATIVE
# Ollama at :11434, NOT the foreign Docker Ollama on :11435 (et-tu-brute, untouched).
# DEFAULT_MODEL is the custom Modelfile tag ``gemma4-e4b-candidate-ctx32k`` which
# BAKES the d36 baseline (num_ctx=32768, temperature=0, top_p=0.95, top_k=64) onto
# the TEXT-ONLY ``batiai/gemma4-e4b:q4`` base (~4.8 GB weights, fits the 6 GB card
# at 0% offload — s10/s11 measured E4B the only fit-passing upgrade beating e2b).
# WHY E4B over e2b: better instruction-following + strict-format, no deep-research
# over-routing; the e2b ``gemma4-e2b-agent`` tag is being retired (d42), so the app
# could not answer on its old default at all (e2b gone from Ollama at swap time).
# NOTE: E4B is the same gemma4 family/renderer as e2b — a THINKING model whose CoT
# lands in the SEPARATE ``message.thinking`` field with clean fenced JSON in
# ``content`` (d25 revalidated on E4B, s8/b8). Structured-output call sites still
# pass ``think=False`` so the CoT trace does not eat num_predict and return EMPTY
# content; reasoned call sites pass ``think=True`` + num_predict>=4096 (d6).
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma4-e4b-candidate-ctx32k:latest"

# --------------------------------------------------------------------------- #
# Universal agent identity (d15)
# --------------------------------------------------------------------------- #
#
# A single, short capable-agent identity shipped on EVERY real model call —
# planner, worker/runtime nodes, shape/spec authoring, the summariser, AND the
# chat runtime — by injecting it at THIS shared transport seam, so no call site
# has to remember it and every call carries the SAME identity. In the spirit of
# Claude Code's system prompt: it states WHO the agent is (capable, reasoning,
# grounded, context-holding), NOT how to code — coding craft lives in the planner
# + specialist docs. Kept tight (~90 tok) so the constant it adds to every prompt
# is negligible against num_ctx and never eats into the load-bearing
# num_predict=4096 OUTPUT budget (d6/d7): the identity is INPUT context, sized
# against num_ctx (8192+), so it cannot push generated content toward truncation.
#
# SINGLE SOURCE OF TRUTH (s3/a6 review fix): this is the ONE canonical identity
# text for the whole app. ``agent_runtime.identity`` RE-EXPORTS this constant (it
# cannot be defined there — llm_framework is the lower layer), so the call-site
# pre-fold ``agent_runtime.identity.with_identity`` and this transport-seam
# injection use BYTE-IDENTICAL text. That is what makes ``_inject_identity``'s
# startswith idempotency guard fire: a system turn a call site already folded the
# identity into is recognised here and NOT doubled. (Before the fix two DIFFERENT
# texts lived in the two layers, so the guard never matched and every with_identity
# call shipped TWO stacked personas — ~240 wasted tok + confused framing.) It also
# folds in the two universal output rules (ground-don't-hallucinate; when asked for
# JSON the visible reply is ONLY the JSON) so structured prompts can drop their own
# repeated "reason privately… no code fences" tails.
# The OBSERVATION ENVELOPE markers (messaging-layer fix): every tool observation the
# transport renders for this prompt-only model is wrapped in these, and the identity
# declares the convention — so the model can DISTINGUISH a tool result it asked for
# from the user speaking, even though both render as user turns on the wire.
OBS_ENVELOPE_OPEN = "[TOOL RESULT]"
OBS_ENVELOPE_CLOSE = "[/TOOL RESULT]"

AGENT_IDENTITY = (
    "You are a capable, autonomous agent. Reason about the user's real goal, then "
    "ACT — prefer doing the task well with sensible defaults over asking. Ground "
    "every answer in the inputs, tools and conversation you are given; never "
    "invent facts, sources or numbers. Treat the prior conversation as memory for "
    "multi-step work. Turns wrapped in [TOOL RESULT]…[/TOOL RESULT] are the outputs "
    "of tools YOU invoked — observations to reason over and act on, never a request "
    "from the user; the user's own words are never wrapped. When you are asked for "
    "JSON, your visible reply is ONLY that JSON — no prose, no code fences. Be "
    "concise and direct."
)


@dataclass
class ChatResult:
    """A single assistant reply."""

    role: str
    content: str
    # The raw provider payload (parsed JSON), kept for callers that need to
    # inspect usage, finish reasons, etc. Optional so FakeTransport stays light.
    raw: Mapping[str, Any] | None = None
    # The model's chain-of-thought trace, when ``think=True`` and the provider
    # returns it in a SEPARATE field (gemma4 native /api/chat -> message.thinking).
    # Surfaced here for observability so the CoT never has to pollute ``content``.
    thinking: str | None = None
    # NATIVE tool calls (s13): the model's structured tool calls when the request
    # carried ``tools=[...]`` and the provider returned them in a SEPARATE channel
    # (Ollama native ``message.tool_calls`` / OpenAI ``choices[].message.tool_calls``).
    # Normalised to ``[{"name": str, "arguments": dict}, ...]``; ``None`` when the
    # request passed no tools or the reply was plain prose. This is the drop-immune
    # layer that REPLACES the homegrown ``startswith('{')`` text parse — the tool call
    # rides its own field, so leading prose can never swallow it. The balanced-brace
    # string parser is kept as a defensive fallback for any non-native reply.
    tool_calls: list[dict[str, Any]] | None = None


def _normalize_tool_calls(raw_calls: Any) -> list[dict[str, Any]] | None:
    """Normalise a provider ``tool_calls`` array to ``[{"name", "arguments"}, ...]``.

    Ollama native (``/api/chat``) returns each call's ``arguments`` as an OBJECT
    (already a dict); the OpenAI-compat endpoint returns it as a JSON STRING. Both
    shapes nest the call under ``{"function": {"name", "arguments"}}`` (older Ollama
    builds put ``name``/``arguments`` at the top level — tolerated). A malformed,
    empty, or absent array returns ``None`` so the caller falls through to the
    balanced-brace string parser unchanged (the s13 defensive fallback)."""
    if not isinstance(raw_calls, (list, tuple)) or not raw_calls:
        return None
    out: list[dict[str, Any]] = []
    for tc in raw_calls:
        if not isinstance(tc, Mapping):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), Mapping) else tc
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        args: Any = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except (ValueError, TypeError):
                args = {}
        if not isinstance(args, Mapping):
            args = {}
        out.append({"name": name, "arguments": dict(args)})
    return out or None


@runtime_checkable
class Transport(Protocol):
    """The seam every transport satisfies.

    ``runtime_checkable`` so call sites can ``isinstance(x, Transport)`` in a
    pinch, though duck typing is the intended usage.
    """

    def complete(self, messages: Sequence[Message], **opts: Any) -> str:
        """Return only the assistant's text content."""
        ...

    def chat(self, messages: Sequence[Message], **opts: Any) -> ChatResult:
        """Return the full assistant reply (role + content + raw)."""
        ...


# --------------------------------------------------------------------------- #
# Real transport: local Ollama (gemma4-e4b-candidate-ctx32k on :11434)
# --------------------------------------------------------------------------- #


class TransportError(RuntimeError):
    """Raised when the model endpoint cannot be reached or returns an error."""


class OllamaTransport:
    """Real local-Ollama transport (Gemma-4 E4B; s8/b8 swap from gemma4-e2b-agent).

    Parameters
    ----------
    base_url:
        Ollama base URL. Defaults to the native serve ``http://127.0.0.1:11434``.
    model:
        Model tag. Defaults to ``gemma4-e4b-candidate-ctx32k`` (the custom
        Modelfile tag with the d36 baseline knobs baked in).
    api:
        ``"openai"`` -> POST ``/v1/chat/completions`` (OpenAI-compatible);
        ``"native"`` -> POST ``/api/chat`` (Ollama native). Both are fully
        supported; pick per call site or override per call via ``api=`` opt.
    keep_alive:
        How long Ollama keeps the model resident after the call. ``-1`` (an
        INTEGER, the default since s5/d21) keeps the model RESIDENT between
        calls — the GPU is now gemma's SOLE consumer, so the old evict-after-
        every-call default (``0``) was pure reload thrash (~1-2 s per call,
        sawtooth VRAM). Pass an integer ``0`` to unload immediately (VRAM
        hygiene, the original d8 intent when the 6 GB was shared), or e.g.
        ``"5m"`` to keep it warm for a bounded window.

        NOTE (s8/a1 finding): Ollama honours an *integer* ``0`` (seconds) as
        evict-now but does NOT honour the *string* ``"0"`` (the model stays
        resident ~24h). So any zero-valued string a caller passes is coerced
        to the integer ``0`` on the wire (see :meth:`_norm_keep_alive`) so the
        VRAM-hygiene behaviour still takes effect when explicitly requested.
        ``-1`` (keep-resident) passes through unchanged.
    timeout:
        Read timeout in seconds. Connect timeout is kept short separately so a
        dead endpoint fails fast rather than hanging the agent.
    client:
        Optional pre-built ``httpx.Client`` (e.g. for tests or connection
        reuse). If omitted, one is created lazily and owned by this instance.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        *,
        api: str = "openai",
        keep_alive: str | int | None = -1,
        timeout: float = 120.0,
        connect_timeout: float = 5.0,
        client: httpx.Client | None = None,
    ) -> None:
        if api not in ("openai", "native"):
            raise ValueError(f"api must be 'openai' or 'native', got {api!r}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api = api
        self.keep_alive = keep_alive
        self._timeout = httpx.Timeout(timeout, connect=connect_timeout)
        self._client = client
        self._owns_client = client is None

    # -- lifecycle --------------------------------------------------------- #

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "OllamaTransport":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- public API -------------------------------------------------------- #

    def complete(self, messages: Sequence[Message], **opts: Any) -> str:
        return self.chat(messages, **opts).content

    @staticmethod
    def _inject_identity(messages: Sequence[Message]) -> list[Message]:
        """Return a COPY of ``messages`` with :data:`AGENT_IDENTITY` (d15) ensured
        as the leading system content.

        The identity rides on every call routed through this transport. We never
        mutate the caller's list (a chain may reuse it), and we fold the identity
        INTO the first system turn rather than adding a second system message —
        gemma's chat template honours a single leading system turn best, so the
        identity comes first and the call site's task-specific system prompt
        follows it. Idempotent: if the identity is already the leading text (e.g.
        a re-entrant repair/summary call on an already-injected list) the messages
        are returned unchanged, so it is never doubled."""
        msgs = [dict(m) for m in messages]
        if msgs and msgs[0].get("role") == "system":
            head = str(msgs[0].get("content", ""))
            if head.startswith(AGENT_IDENTITY):
                return msgs  # already carries the identity — do not double up
            msgs[0]["content"] = f"{AGENT_IDENTITY}\n\n{head}" if head else AGENT_IDENTITY
        else:
            msgs.insert(0, {"role": "system", "content": AGENT_IDENTITY})
        return msgs

    @staticmethod
    def _normalize_tool_roles(messages: Sequence[Message]) -> list[Message]:
        """Return a COPY of ``messages`` with every ``role: "tool"`` turn rewritten
        to ``role: "user"`` AND its content wrapped in the OBSERVATION ENVELOPE
        (d262 / d199 / d202 + the messaging-layer fix).

        Gemma's Ollama chat template is PROMPT-ONLY: it renders ``system`` /
        ``user`` / ``assistant`` turns but IGNORES ``role: "tool"`` entirely, so
        any observation fed back as a tool message is INVISIBLE to the model.
        Rewriting tool→user at this one chokepoint makes every observation SEEN —
        but a bare rewrite made tool output INDISTINGUISHABLE from a user request
        (the owner's messaging-layer finding). The fix: every rewritten turn is
        wrapped ``[TOOL RESULT]\\n…\\n[/TOOL RESULT]`` — a convention the agent
        identity declares — so the model can tell an observation it asked for from
        the user speaking. User-authored turns are NEVER wrapped (the marker is
        exclusive to tool observations by construction), and the wrap is idempotent
        (already-wrapped content passes through). In-memory histories keep the
        semantic ``role:"tool"`` label; only the outgoing wire copy is rendered.
        We never mutate the caller's list (a chain may reuse it): tool turns are
        emitted as fresh dicts, other turns pass through unchanged."""
        out: list[Message] = []
        for m in messages:
            if m.get("role") == "tool":
                nm = dict(m)
                nm["role"] = "user"
                content = str(nm.get("content") or "")
                if not content.lstrip().startswith(OBS_ENVELOPE_OPEN):
                    nm["content"] = (
                        f"{OBS_ENVELOPE_OPEN}\n{content}\n{OBS_ENVELOPE_CLOSE}"
                    )
                out.append(nm)
            else:
                out.append(m)
        return out

    def chat(self, messages: Sequence[Message], **opts: Any) -> ChatResult:
        # d15: ship the universal agent identity on EVERY real model call. This is
        # the one shared seam every call site (planner, runtime nodes, shape/spec
        # authoring, summariser, chat) routes through, so injecting here — rather
        # than at each call site — guarantees a consistent identity everywhere.
        messages = self._inject_identity(messages)
        # d262/d199/d202: gemma's Ollama template ignores role:"tool", so any
        # observation handed back as a tool message is invisible to the model.
        # Rewrite tool roles to "user" at this one chokepoint (both wire paths copy
        # the message list verbatim and never inspect roles) so every tool
        # observation is SEEN by the model — fixing all ~13 call sites at once.
        messages = self._normalize_tool_roles(messages)
        api = opts.pop("api", self.api)
        # No tracing available (httpx-only install): behave exactly as before.
        if not _OTEL_AVAILABLE:
            return self._dispatch(api, messages, **opts)

        # Open ONE child OpenInference LLM span per model call. ``start_as_
        # current_span`` nests it under whatever span is currently active (the
        # per-node / per-run span the runtime opened) via context propagation —
        # so it is a CHILD, never a fresh root trace. We attach to the global
        # provider a1 registered; we never build a second one.
        tracer = _otel_trace.get_tracer(__name__)
        invocation = self._span_invocation_params(api, opts)
        started = time.perf_counter()
        with tracer.start_as_current_span("llm.chat") as span:
            span.set_attribute(_OI_SPAN_KIND, _OI_LLM_KIND)
            span.set_attribute(_OI_MODEL_NAME, self.model)
            span.set_attribute(_ATTR_API_TYPE, api)
            if invocation:
                # invocation holds ONLY non-secret generation knobs (model,
                # temperature, max_tokens, num_ctx) — never message content or
                # any api key/secret.
                span.set_attribute(_OI_INVOCATION_PARAMETERS, json.dumps(invocation))
            try:
                result = self._dispatch(api, messages, **opts)
            except BaseException as exc:
                span.set_attribute(
                    _ATTR_LATENCY_MS, (time.perf_counter() - started) * 1000.0
                )
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            latency_ms = (time.perf_counter() - started) * 1000.0
            span.set_attribute(_ATTR_LATENCY_MS, latency_ms)
            prompt_tokens, completion_tokens = self._extract_token_counts(result.raw)
            if prompt_tokens is not None:
                span.set_attribute(_OI_TOKEN_PROMPT, int(prompt_tokens))
            if completion_tokens is not None:
                span.set_attribute(_OI_TOKEN_COMPLETION, int(completion_tokens))
            if prompt_tokens is not None and completion_tokens is not None:
                span.set_attribute(
                    _OI_TOKEN_TOTAL, int(prompt_tokens) + int(completion_tokens)
                )
            # done_reason ("stop"/"length") — NON-secret, so it rides the span itself
            # (visible in BOTH Phoenix and the local trace's attributes). It is the
            # load-bearing signal that tells a clean finish from a num_predict-
            # truncated emission (c1r trace method / d19/o5), and is ALSO mirrored into
            # the local-only enrichment below for the markdown renderer.
            done_reason = self._extract_done_reason(result.raw)
            if done_reason:
                span.set_attribute("llm.response.done_reason", done_reason)
            # LOCAL-ONLY enrichment (s7 Stage-A): stash the full prompt + reasoning
            # the Phoenix span deliberately omits, keyed by THIS span's id, so the
            # local file exporter can merge it on export. Never a span attribute, so
            # Phoenix still gets no secrets — the no-secrets rule is honoured.
            self._record_local_capture(
                span, api, messages, result, invocation,
                prompt_tokens, completion_tokens, latency_ms, done_reason,
            )
            span.set_status(Status(StatusCode.OK))
            return result

    def _dispatch(self, api: str, messages: Sequence[Message], **opts: Any) -> ChatResult:
        """Route to the per-API implementation (shared by traced/untraced paths)."""
        if api == "openai":
            return self._chat_openai(messages, **opts)
        return self._chat_native(messages, **opts)

    @staticmethod
    def _span_invocation_params(api: str, opts: Mapping[str, Any]) -> dict[str, Any]:
        """Collect the safe, non-secret generation knobs for the LLM span.

        ONLY well-known invocation parameters land here — temperature, the
        max-output-tokens knob (under either spelling), and num_ctx — plus the
        api flavour. We deliberately exclude message content and never touch any
        credential, honouring the "no secrets in span attributes" rule."""
        params: dict[str, Any] = {"api_type": api, "model": opts.get("model", "")}
        if "temperature" in opts:
            params["temperature"] = opts["temperature"]
        if "max_tokens" in opts:
            params["max_tokens"] = opts["max_tokens"]
        elif "num_predict" in opts:
            params["max_tokens"] = opts["num_predict"]
        if "num_ctx" in opts:
            params["num_ctx"] = opts["num_ctx"]
        if not params.get("model"):
            params.pop("model", None)
        return params

    @staticmethod
    def _extract_token_counts(
        raw: Mapping[str, Any] | None,
    ) -> tuple[int | None, int | None]:
        """Pull (prompt, completion) token counts from a provider payload.

        Both wire shapes are supported: OpenAI-compat reports them under
        ``usage.{prompt_tokens, completion_tokens}``; Ollama native reports
        ``prompt_eval_count`` / ``eval_count`` at the top level. Returns
        ``(None, None)`` when the payload carries no usage — we never fabricate
        counts ("if available")."""
        if not isinstance(raw, Mapping):
            return None, None
        usage = raw.get("usage")
        if isinstance(usage, Mapping):
            p = usage.get("prompt_tokens")
            c = usage.get("completion_tokens")
            if p is not None or c is not None:
                return p, c
        return raw.get("prompt_eval_count"), raw.get("eval_count")

    @staticmethod
    def _extract_done_reason(raw: Mapping[str, Any] | None) -> "str | None":
        """The provider's stop reason, both wire shapes (never fabricated).

        Ollama native reports ``done_reason`` ("stop"/"length") at the top level;
        OpenAI-compat reports ``choices[0].finish_reason``. Returns None when the
        payload carries neither. A ``done_reason="length"`` is the load-bearing
        signal that distinguishes a num_predict-TRUNCATED emission from a clean
        finish — the fact the c1r trace method needed and could not read (d19/o5)."""
        if not isinstance(raw, Mapping):
            return None
        dr = raw.get("done_reason")
        if dr:
            return str(dr)
        choices = raw.get("choices")
        if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)) and choices:
            first = choices[0]
            if isinstance(first, Mapping) and first.get("finish_reason"):
                return str(first.get("finish_reason"))
        return None

    def _record_local_capture(
        self,
        span: Any,
        api: str,
        messages: Sequence[Message],
        result: "ChatResult",
        invocation: Mapping[str, Any],
        prompt_tokens: Any,
        completion_tokens: Any,
        latency_ms: float,
        done_reason: "str | None" = None,
    ) -> None:
        """Buffer this call's LOCAL-ONLY enrichment for the local file exporter.

        Records the data Phoenix omits — the full system+user prompt messages and
        the model's ``thinking`` reasoning block — alongside the same token/model/
        latency facts, keyed by the active span's hex id. This NEVER stamps any of
        it onto the span, so the OTLP→Phoenix export stays secret-free; only the
        local file exporter (``agent_runtime.local_trace``) pops it back out.

        Observability must never break a real model call, so this is best-effort:
        if opentelemetry or the capture buffer is unavailable, or anything fails,
        it is logged on the span (not the prompt) and swallowed.
        """
        try:
            from . import local_capture
        except Exception:  # pragma: no cover - capture buffer is an optional extra
            return
        try:
            ctx = span.get_span_context()
            span_id = format(ctx.span_id, "016x")
            trace_id = format(ctx.trace_id, "032x")
            payload = {
                "trace_id": trace_id,
                "span_id": span_id,
                "model": self.model,
                "api": api,
                "latency_ms": latency_ms,
                "invocation_parameters": dict(invocation) if invocation else {},
                # The FULL prompt actually sent (post identity-injection): every
                # system+user turn, role + content — the thing Phoenix cannot show.
                "messages": [dict(m) for m in messages],
                "response_content": result.content,
                # The CoT reasoning block (gemma native think=True -> message.thinking).
                "thinking": result.thinking,
                # The stop reason ("stop"/"length") — lets the renderer flag a
                # num_predict-truncated turn vs a clean finish (c1r trace method).
                "done_reason": done_reason,
                "token_counts": {
                    "prompt": int(prompt_tokens) if prompt_tokens is not None else None,
                    "completion": (
                        int(completion_tokens) if completion_tokens is not None else None
                    ),
                },
            }
            local_capture.record_llm_capture(span_id, payload)
        except Exception as exc:  # pragma: no cover - never break a model call
            # Note the failure on the span (which carries NO prompt content), so a
            # dropped capture is traceable without leaking the prompt we failed on.
            try:
                span.set_attribute("llm.local_capture_error", str(exc))
            except Exception:
                pass

    # -- option mapping ---------------------------------------------------- #
    #
    # We keep a small, explicit set of "nice" options and map them onto each
    # wire format. Unknown opts are passed through so callers aren't blocked,
    # but the common knobs (temperature, max tokens, num_ctx, json format) are
    # normalised across the two APIs.

    @staticmethod
    def _norm_keep_alive(value: Any) -> Any:
        """Coerce a zero-valued string keep_alive to the integer ``0`` (s8/a1).

        Ollama treats an INTEGER ``0`` (seconds) as evict-now but does NOT
        honour the STRING ``"0"`` / ``"0s"`` — the model then stays resident
        (~24h), defeating the d8 VRAM-hygiene intent. Coercing here means the
        eviction works regardless of whether a caller wrote ``0`` or ``"0"``,
        while non-zero strings like ``"5m"`` (keep-warm) pass through unchanged."""
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("0", "0s", "0m", "0h", "0ms"):
                return 0
        return value

    def _common_opts(self, opts: MutableMapping[str, Any]) -> dict[str, Any]:
        keep_alive = self._norm_keep_alive(opts.pop("keep_alive", self.keep_alive))
        return {"keep_alive": keep_alive, "opts": opts}

    def _chat_openai(self, messages: Sequence[Message], **opts: Any) -> ChatResult:
        common = self._common_opts(opts)
        rest = common["opts"]
        body: dict[str, Any] = {
            "model": rest.pop("model", self.model),
            "messages": list(messages),
            "stream": False,
        }
        if "temperature" in rest:
            body["temperature"] = rest.pop("temperature")
        # OpenAI uses max_tokens; accept either spelling.
        if "max_tokens" in rest:
            body["max_tokens"] = rest.pop("max_tokens")
        elif "num_predict" in rest:
            body["max_tokens"] = rest.pop("num_predict")
        if rest.pop("json", False) or rest.get("response_format"):
            body.setdefault("response_format", rest.pop("response_format", {"type": "json_object"}))
        # ``think`` is an Ollama-NATIVE control with no OpenAI-compat equivalent.
        # Drop it here so a caller that passes ``think=False`` uniformly (planner /
        # writer nodes) does not leak a bogus field into the OpenAI body. The
        # think=false structured-output fix takes effect on the native path below.
        rest.pop("think", None)
        # ``tools`` is a TOP-LEVEL field on the OpenAI-compat body too — pull it out
        # BEFORE the remainder folds into ``options`` so the native tool schemas are
        # not buried as a bogus option (s13). Parsed back from
        # ``choices[0].message.tool_calls`` below and normalised the same way as native.
        tools = rest.pop("tools", None)
        if tools:
            body["tools"] = list(tools)
        # Ollama honours keep_alive on the OpenAI-compat endpoint as an extra.
        if common["keep_alive"] is not None:
            body["keep_alive"] = common["keep_alive"]
        # num_ctx etc. -> Ollama reads these via "options" even on /v1.
        if rest:
            body.setdefault("options", {}).update(rest)

        data = self._post("/v1/chat/completions", body)
        try:
            msg = data["choices"][0]["message"]
            return ChatResult(role=msg.get("role", "assistant"),
                              content=msg.get("content", "") or "", raw=data,
                              tool_calls=_normalize_tool_calls(msg.get("tool_calls")))
        except (KeyError, IndexError, TypeError) as exc:
            raise TransportError(f"unexpected OpenAI-compat response: {data!r}") from exc

    def _chat_native(self, messages: Sequence[Message], **opts: Any) -> ChatResult:
        common = self._common_opts(opts)
        rest = common["opts"]
        options: dict[str, Any] = {}
        if "temperature" in rest:
            options["temperature"] = rest.pop("temperature")
        # Native uses num_predict for max output tokens.
        if "max_tokens" in rest:
            options["num_predict"] = rest.pop("max_tokens")
        elif "num_predict" in rest:
            options["num_predict"] = rest.pop("num_predict")
        if "num_ctx" in rest:
            options["num_ctx"] = rest.pop("num_ctx")
        # Pull the JSON-mode flags OUT of ``rest`` before folding the remainder
        # into ``options`` — otherwise ``json``/``format`` leak into Ollama's
        # options block as bogus keys (and a schema-dict ``format`` is silently
        # dropped instead of taking effect).
        want_json = bool(rest.pop("json", False))
        fmt = rest.pop("format", None)
        # ``think`` is a TOP-LEVEL /api/chat field (NOT an entry in ``options``).
        # Pull it out of ``rest`` before the remainder folds into ``options`` so it
        # lands on the body where Ollama reads it — otherwise it would silently
        # become a bogus option and the CoT trace would still fire. This is the
        # DECISIVE s8/b1 structured-output fix: gemma4 is a thinking model, so the
        # planner/structured calls pass ``think=False`` and the model emits the JSON
        # plan directly (a3 proved 24/24) instead of spending num_predict on CoT
        # and returning EMPTY content (a2 measured 0% without it).
        think = rest.pop("think", None)
        # ``tools`` is a TOP-LEVEL /api/chat field (NOT an entry in ``options``),
        # exactly like ``think``. Pull it out BEFORE the remainder folds into
        # ``options`` so the native tool schemas land on the body where Ollama reads
        # them and the model returns ``message.tool_calls`` (s13 native migration) —
        # otherwise it would become a bogus option and silently never take effect.
        tools = rest.pop("tools", None)
        options.update(rest.pop("options", {}))
        # Any remaining opts are treated as native options too.
        options.update(rest)

        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
        }
        if options:
            body["options"] = options
        # ``format`` forces structured output: "json" guarantees valid JSON
        # SYNTAX (not schema — see the agent's structured-output layer for
        # enum/required enforcement), while an explicit value (e.g. a JSON
        # schema dict, which Ollama supports natively) is honoured as-is.
        if fmt is not None:
            body["format"] = fmt
        elif want_json:
            body["format"] = "json"
        # Top-level think control (see above). Only set when the caller was explicit
        # — leaving it unset preserves the model's own default for non-structured
        # calls, so this never silently disables thinking where it is wanted.
        if think is not None:
            body["think"] = bool(think)
        # Native tool schemas (s13): when present, the model may answer with a
        # structured ``message.tool_calls`` instead of (or alongside) prose content.
        if tools:
            body["tools"] = list(tools)
        if common["keep_alive"] is not None:
            body["keep_alive"] = common["keep_alive"]

        data = self._post("/api/chat", body)
        try:
            msg = data["message"]
            content = msg.get("content", "") or ""
            # JSON-extraction interceptor (s1/a2): when the call was structured
            # (``format=json`` or a schema dict), strip Markdown code fences and
            # walk out the first balanced JSON object/array so the returned
            # ``content`` is clean JSON. This covers BOTH the chain sites (which
            # already strip via stages._extract_json) AND the two DIRECT
            # ``json.loads`` sites (toolargs.py / tool_registry.py) that break on
            # a fenced response. On a TRUNCATED reply (CoT ate num_predict ->
            # unbalanced/empty JSON) extraction returns None; we then fall back to
            # fence-stripping only, so the truncation stays VISIBLE downstream as a
            # parse failure rather than being masked. Lazy import avoids the
            # transport<->stages circular import.
            if fmt is not None or want_json:
                from .stages import _extract_json, _strip_fences

                extracted = _extract_json(content)
                content = extracted if extracted is not None else _strip_fences(content)
            return ChatResult(role=msg.get("role", "assistant"),
                              content=content, raw=data,
                              thinking=msg.get("thinking") or None,
                              tool_calls=_normalize_tool_calls(msg.get("tool_calls")))
        except (KeyError, TypeError) as exc:
            raise TransportError(f"unexpected native response: {data!r}") from exc

    # -- transport --------------------------------------------------------- #

    def _post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = self._http().post(url, json=body)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise TransportError(
                f"{path} -> HTTP {exc.response.status_code}: {exc.response.text[:500]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise TransportError(f"{path} -> transport failure: {exc}") from exc


# --------------------------------------------------------------------------- #
# Fake transport: deterministic, scripted, offline
# --------------------------------------------------------------------------- #

# A scripted reply can be:
#   - a str            -> returned as the assistant content
#   - a ChatResult     -> returned verbatim
#   - an Exception     -> raised (to simulate transport failures)
#   - a callable        -> called with (messages, **opts), result re-interpreted
ScriptedReply = "str | ChatResult | BaseException | Callable[..., Any]"


class FakeTransport:
    """Deterministic, scripted transport for fully-offline runs (d7/d8).

    Program it with an ordered list of replies; each ``chat``/``complete`` call
    consumes the next one. When the script is exhausted it repeats the LAST
    reply (so a steady-state stub keeps answering) unless ``strict=True``, in
    which case it raises ``IndexError`` — useful to assert an exact call count.

    Every call's ``messages`` and ``opts`` are recorded on ``.calls`` for test
    assertions.

    Example — malformed-JSON-then-valid sequence (exercises parser self-heal)::

        t = FakeTransport.malformed_then_valid('{"plan": [')  # bad, then good
        first = t.complete([{"role": "user", "content": "plan"}])   # '{"plan": ['
        second = t.complete([{"role": "user", "content": "retry"}]) # valid JSON
    """

    def __init__(
        self,
        responses: Sequence[Any] | None = None,
        *,
        strict: bool = False,
    ) -> None:
        self._responses: List[Any] = list(responses or [])
        self.strict = strict
        self._index = 0
        self.calls: List[dict[str, Any]] = []

    # -- programming ------------------------------------------------------- #

    def queue(self, *replies: Any) -> "FakeTransport":
        """Append one or more scripted replies. Returns self for chaining."""
        self._responses.extend(replies)
        return self

    def reset(self) -> None:
        """Rewind the script cursor and clear the recorded calls."""
        self._index = 0
        self.calls.clear()

    @classmethod
    def malformed_then_valid(
        cls,
        malformed: str = '{"oops": ',
        valid: str = '{"ok": true}',
        **kwargs: Any,
    ) -> "FakeTransport":
        """Convenience: a transport that returns broken JSON, then valid JSON."""
        return cls([malformed, valid], **kwargs)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    # -- public API -------------------------------------------------------- #

    def complete(self, messages: Sequence[Message], **opts: Any) -> str:
        return self.chat(messages, **opts).content

    def chat(self, messages: Sequence[Message], **opts: Any) -> ChatResult:
        self.calls.append({"messages": list(messages), "opts": dict(opts)})
        reply = self._next_reply()

        if callable(reply) and not isinstance(reply, type):
            reply = reply(messages, **opts)
        if isinstance(reply, type) and issubclass(reply, BaseException):
            raise reply()
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, ChatResult):
            return reply
        return ChatResult(role="assistant", content=str(reply))

    # -- internals --------------------------------------------------------- #

    def _next_reply(self) -> Any:
        if not self._responses:
            if self.strict:
                raise IndexError("FakeTransport has no scripted responses")
            return ""
        if self._index < len(self._responses):
            reply = self._responses[self._index]
            self._index += 1
            return reply
        if self.strict:
            raise IndexError(
                f"FakeTransport script exhausted after {len(self._responses)} replies"
            )
        return self._responses[-1]  # steady-state: repeat the last reply
