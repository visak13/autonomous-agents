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
   ``http://127.0.0.1:11434`` running the Gemma-4 edge E2B model
   ``gemma4-e2b-agent`` (s8/b1 swap; was phi4-mini on :11435). It can drive
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

# Runtime model swap (s8/b1, supersedes the phi4-mini default of d8 per d17).
# The app now drives Google's Gemma-4 edge E2B on a's NATIVE Ollama at :11434
# (v0.30.8), NOT the foreign Docker Ollama on :11435 (et-tu-brute, untouched).
# DEFAULT_MODEL is the custom Modelfile tag ``gemma4-e2b-agent`` which BAKES the
# s8-measured optimal knobs (num_ctx=8192, temperature=0, top_p=0.95, top_k=64,
# num_predict=1024) onto the ``gemma4:e2b-it-qat`` base; fall back to
# ``gemma4:e2b-it-qat`` + per-call params if the custom tag is not built.
# NOTE: gemma4 is a THINKING model — structured-output call sites MUST pass
# ``think=False`` (see the planner/agent wiring) or the CoT trace eats the token
# budget and the JSON ``content`` comes back EMPTY (the s8/a2 root-cause).
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma4-e2b-agent"


@dataclass
class ChatResult:
    """A single assistant reply."""

    role: str
    content: str
    # The raw provider payload (parsed JSON), kept for callers that need to
    # inspect usage, finish reasons, etc. Optional so FakeTransport stays light.
    raw: Mapping[str, Any] | None = None


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
# Real transport: local Ollama / phi4-mini
# --------------------------------------------------------------------------- #


class TransportError(RuntimeError):
    """Raised when the model endpoint cannot be reached or returns an error."""


class OllamaTransport:
    """Real local-Ollama transport (Gemma-4 E2B; s8/b1 swap from phi4-mini).

    Parameters
    ----------
    base_url:
        Ollama base URL. Defaults to the native serve ``http://127.0.0.1:11434``.
    model:
        Model tag. Defaults to ``gemma4-e2b-agent`` (the custom Modelfile tag
        with the s8-optimal knobs baked in).
    api:
        ``"openai"`` -> POST ``/v1/chat/completions`` (OpenAI-compatible);
        ``"native"`` -> POST ``/api/chat`` (Ollama native). Both are fully
        supported; pick per call site or override per call via ``api=`` opt.
    keep_alive:
        How long Ollama keeps the model resident after the call. ``0`` (an
        INTEGER, the default) unloads immediately — VRAM hygiene on the shared
        GPU (d8). Use e.g. ``"5m"`` to keep it warm across a burst of calls.

        NOTE (s8/a1 finding): Ollama honours an *integer* ``0`` (seconds) as
        evict-now but does NOT honour the *string* ``"0"`` (the model stays
        resident ~24h). The default is therefore the integer ``0``, and any
        zero-valued string a caller passes is coerced to the integer ``0`` on
        the wire (see :meth:`_norm_keep_alive`) so the documented VRAM-hygiene
        behaviour actually takes effect.
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
        keep_alive: str | int | None = 0,
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

    def chat(self, messages: Sequence[Message], **opts: Any) -> ChatResult:
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
            span.set_attribute(
                _ATTR_LATENCY_MS, (time.perf_counter() - started) * 1000.0
            )
            prompt_tokens, completion_tokens = self._extract_token_counts(result.raw)
            if prompt_tokens is not None:
                span.set_attribute(_OI_TOKEN_PROMPT, int(prompt_tokens))
            if completion_tokens is not None:
                span.set_attribute(_OI_TOKEN_COMPLETION, int(completion_tokens))
            if prompt_tokens is not None and completion_tokens is not None:
                span.set_attribute(
                    _OI_TOKEN_TOTAL, int(prompt_tokens) + int(completion_tokens)
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
                              content=msg.get("content", "") or "", raw=data)
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
        if common["keep_alive"] is not None:
            body["keep_alive"] = common["keep_alive"]

        data = self._post("/api/chat", body)
        try:
            msg = data["message"]
            return ChatResult(role=msg.get("role", "assistant"),
                              content=msg.get("content", "") or "", raw=data)
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
