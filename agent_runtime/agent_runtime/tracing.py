"""OpenTelemetry tracer factory for ReactiveAgents -> local docker Phoenix.

This module is the SINGLE place that builds the app's ``TracerProvider`` and is
the load-bearing de-risk of the whole tracing item (s6). Two facts make the
Phoenix export path non-obvious, and both are baked in here:

1. **Project naming.** Phoenix groups spans into a *project* by reading the
   OpenInference resource attribute ``openinference.project.name`` from the
   span's Resource -- NOT ``service.name``. We attach it via the
   ``ResourceAttributes.PROJECT_NAME`` constant so our spans land in their own
   ``reactive-agents`` project and never mix with the eda-ml traces already in
   this Phoenix instance.

2. **Transport.** Phoenix's OTLP gRPC/HTTP collector ports (4317 / 4318) are
   NOT exposed on this host (d5). The ONLY reachable ingest is Phoenix's own
   HTTP ``/v1/traces`` endpoint on :6006, so we use the OTLP **HTTP/proto**
   exporter pointed straight at ``http://localhost:6006/v1/traces``.

Both the endpoint and the project name are read from the environment so the
process can be repointed without code changes; the defaults below are the
proven-working values.

Usage::

    from agent_runtime.tracing import get_tracer_provider, get_tracer

    provider = get_tracer_provider()          # idempotent: built once
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("phi.call") as span:
        # Tag the span with the CONFIGURED model, never a hardcoded literal, so
        # the trace tracks whatever tag the transport actually drives (s8/b1 swap
        # to gemma4-e2b-agent; the live LLM span at transport.py sets
        # ``llm.model_name`` from ``self.model`` the same way).
        from llm_framework.transport import DEFAULT_MODEL
        span.set_attribute("model", DEFAULT_MODEL)
        ...
    # on shutdown:
    provider.force_flush(); provider.shutdown()
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any, Callable

from openinference.semconv.resource import ResourceAttributes
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Proven-working defaults (s6 POC). Phoenix only exposes :6006 /v1/traces here.
DEFAULT_PROJECT_NAME = "reactive-agents"
DEFAULT_OTLP_ENDPOINT = "http://localhost:6006/v1/traces"

# Env knobs (repoint without code changes).
ENV_PROJECT_NAME = "REACTIVE_AGENTS_PHOENIX_PROJECT"
ENV_OTLP_ENDPOINT = "REACTIVE_AGENTS_OTLP_ENDPOINT"

_provider_lock = threading.Lock()
_provider: TracerProvider | None = None


def _resolve_project_name() -> str:
    return os.environ.get(ENV_PROJECT_NAME, DEFAULT_PROJECT_NAME)


def _resolve_endpoint() -> str:
    return os.environ.get(ENV_OTLP_ENDPOINT, DEFAULT_OTLP_ENDPOINT)


def build_tracer_provider(
    *,
    project_name: str | None = None,
    endpoint: str | None = None,
) -> TracerProvider:
    """Build a fresh ``TracerProvider`` wired to Phoenix.

    The Resource carries ``openinference.project.name`` (the attribute Phoenix
    reads to name the project) plus a ``service.name`` for human-readability.
    A ``BatchSpanProcessor`` wraps the OTLP **HTTP** exporter pointed at
    Phoenix's ``/v1/traces``.

    This does NOT register the provider globally -- callers that want the
    process-wide singleton should use :func:`get_tracer_provider`.
    """
    project_name = project_name or _resolve_project_name()
    endpoint = endpoint or _resolve_endpoint()

    resource = Resource.create(
        {
            ResourceAttributes.PROJECT_NAME: project_name,
            "service.name": project_name,
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    # SECOND export path (s7 Stage-A): a LOCAL file exporter alongside — never
    # replacing — the Phoenix one above. It writes a readable, complete JSON per
    # trace under var/traces, merging in the full prompts + reasoning the
    # Phoenix span omits. Best-effort: a failure to wire it must never block the
    # (load-bearing) Phoenix path or app startup.
    try:
        from agent_runtime.local_trace import build_local_file_processor

        provider.add_span_processor(build_local_file_processor())
    except Exception:  # pragma: no cover - local capture is a debugging aid, not core
        pass
    # THIRD path (s7 Stage-B): render the just-written JSON to readable MARKDOWN
    # when a run's ROOT span ends. Registered AFTER the JSON processor above so the
    # trace's JSON is already on disk by the time it renders. Reads only — it never
    # touches the capture path; best-effort so it can never block Phoenix/startup.
    try:
        from agent_runtime.local_trace_md import MarkdownTraceProcessor

        provider.add_span_processor(MarkdownTraceProcessor())
    except Exception:  # pragma: no cover - markdown render is a debugging aid, not core
        pass
    return provider


def get_tracer_provider() -> TracerProvider:
    """Return the process-wide singleton provider, building it once.

    Also registers it as the OpenTelemetry global provider so any
    ``opentelemetry.trace.get_tracer(...)`` in the codebase resolves to it.
    Idempotent and thread-safe.
    """
    global _provider
    if _provider is not None:
        return _provider
    with _provider_lock:
        if _provider is None:
            _provider = build_tracer_provider()
            # Register globally so trace.get_tracer(...) anywhere uses our wiring.
            trace.set_tracer_provider(_provider)
    return _provider


def get_tracer(name: str = "agent_runtime") -> trace.Tracer:
    """Convenience accessor: ensure the provider exists and hand back a tracer."""
    return get_tracer_provider().get_tracer(name)


async def run_blocking_in_span(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a blocking ``fn`` in a worker thread WITH the current OTel context.

    This is the cross-thread span-propagation seam (s6/b2). The runtime and the
    planner offload their blocking phi chain to a worker thread via
    ``asyncio.to_thread`` (the never-freeze fix, d4). ``asyncio.to_thread`` DOES
    copy the caller's ``contextvars`` (so the OpenTelemetry context *would*
    propagate on its own), but we ALSO explicitly **capture the current OTel
    context here and re-attach it inside the worker thread** — belt-and-suspenders
    so that any span the blocking code opens (e.g. the b1 per-phi-call LLM span)
    nests under the currently-active span (the per-node / planner span) instead of
    detaching into a separate ROOT trace. The re-attach is the load-bearing line:
    without it a future change to the offload mechanism (a bare executor without
    ``copy_context``) would silently orphan every phi span.

    Capture happens on the calling coroutine's thread (where the node/planner span
    is active); the attach/detach is balanced inside the worker thread so the
    global context is left exactly as found.
    """
    captured = otel_context.get_current()

    def _runner() -> Any:
        token = otel_context.attach(captured)
        try:
            return fn(*args, **kwargs)
        finally:
            otel_context.detach(token)

    return await asyncio.to_thread(_runner)


def shutdown_tracer_provider() -> None:
    """Flush and tear down the singleton provider (call on app shutdown)."""
    global _provider
    if _provider is not None:
        try:
            _provider.force_flush()
        finally:
            _provider.shutdown()
            _provider = None
