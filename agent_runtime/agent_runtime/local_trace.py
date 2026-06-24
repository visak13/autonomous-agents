"""LOCAL file trace exporter — the readable, complete on-disk trace (s7 Stage-A).

Phoenix logging is effectively blind for debugging this app: the OTLP spans carry
token counts and latency but, by the "no secrets in span attributes" rule, NEVER
the prompt that was sent or the model's reasoning. This module is the SECOND
export path that fixes that locally, without touching the Phoenix one.

It plugs a :class:`LocalFileSpanExporter` into the app's existing
``TracerProvider`` as a SECOND span processor, alongside (never replacing) the
Phoenix ``BatchSpanProcessor``. On export it writes ONE JSON file per trace under
the local trace dir (``var/traces`` by default), and for every ``llm.chat`` span
it MERGES in the rich, local-only enrichment the transport stashed in
:mod:`llm_framework.local_capture` — the full system+user prompt messages, the
``thinking`` reasoning block, and the per-call token counts. That enrichment never
rode the span, so Phoenix still receives nothing secret; only this file path does.

The result is a single, self-contained JSON per run that Stage B can render to
markdown (prompts, reasoning, token cost per step) — the source of truth the user
and the neuron were missing.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

# Default local trace dir; repointable via env so the path is never hardcoded at
# the call site (matches tracing.py's endpoint/project env-knob convention).
DEFAULT_LOCAL_TRACE_DIR = os.path.join("C:\\", "Projects", "ReactiveAgents", "var", "traces")
ENV_LOCAL_TRACE_DIR = "REACTIVE_AGENTS_LOCAL_TRACE_DIR"

# The span the transport opens per model call (llm_framework.transport.chat). Only
# these carry enrichment; the name is the contract between the two layers.
_LLM_SPAN_NAME = "llm.chat"


def resolve_local_trace_dir() -> str:
    """Local trace dir from env, falling back to the proven default."""
    return os.environ.get(ENV_LOCAL_TRACE_DIR, DEFAULT_LOCAL_TRACE_DIR)


class LocalFileSpanExporter(SpanExporter):
    """Write each trace to ``<dir>/<trace_id>.json``, merging LLM enrichment.

    Spans of one trace end at different times, so this accumulates them in memory
    keyed by ``trace_id`` and rewrites that trace's file on every export — the file
    grows to the full span tree as outer spans close. Thread-safe; best-effort (a
    failed write is reported via the export result, never raised into the app).
    """

    def __init__(self, trace_dir: Optional[str] = None) -> None:
        self._dir = trace_dir or resolve_local_trace_dir()
        self._lock = threading.Lock()
        # trace_id -> {span_id -> span_dict}; rewritten to disk on each export.
        self._traces: Dict[str, Dict[str, Dict[str, Any]]] = {}
        os.makedirs(self._dir, exist_ok=True)

    # -- SpanExporter API -------------------------------------------------- #

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            touched: set[str] = set()
            with self._lock:
                for span in spans:
                    record = self._span_to_dict(span)
                    trace_id = record["trace_id"]
                    self._traces.setdefault(trace_id, {})[record["span_id"]] = record
                    touched.add(trace_id)
                for trace_id in touched:
                    self._write_trace(trace_id)
            return SpanExportResult.SUCCESS
        except Exception:  # pragma: no cover - never break the provider's export loop
            return SpanExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        # Writes are synchronous in export(); nothing is buffered to flush.
        return True

    def shutdown(self) -> None:
        # Files are written eagerly; no handles are held open between exports.
        return None

    # -- internals --------------------------------------------------------- #

    def _span_to_dict(self, span: ReadableSpan) -> Dict[str, Any]:
        ctx = span.get_span_context()
        trace_id = format(ctx.trace_id, "032x")
        span_id = format(ctx.span_id, "016x")
        parent_id = (
            format(span.parent.span_id, "016x") if span.parent is not None else None
        )
        start_ns = span.start_time or 0
        end_ns = span.end_time or 0
        record: Dict[str, Any] = {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_id,
            "name": span.name,
            "kind": span.kind.name if span.kind is not None else None,
            "start_time_ns": start_ns,
            "end_time_ns": end_ns,
            "duration_ms": (end_ns - start_ns) / 1_000_000.0 if end_ns and start_ns else None,
            "status": span.status.status_code.name if span.status is not None else None,
            # Phoenix-safe attributes only (model, tokens, latency, invocation) —
            # the prompt/reasoning live under "llm_capture" below, local-only.
            "attributes": {k: v for k, v in (span.attributes or {}).items()},
        }
        if span.name == _LLM_SPAN_NAME:
            enrichment = self._pop_enrichment(span_id)
            if enrichment is not None:
                record["llm_capture"] = enrichment
        return record

    @staticmethod
    def _pop_enrichment(span_id: str) -> Optional[Dict[str, Any]]:
        """Pull the transport's local-only enrichment for this LLM span, if any."""
        try:
            from llm_framework import local_capture
        except Exception:  # pragma: no cover - llm_framework always present in app
            return None
        return local_capture.pop_llm_capture(span_id)

    def _write_trace(self, trace_id: str) -> None:
        spans = sorted(
            self._traces.get(trace_id, {}).values(),
            key=lambda s: s.get("start_time_ns") or 0,
        )
        doc = {"trace_id": trace_id, "span_count": len(spans), "spans": spans}
        path = os.path.join(self._dir, f"{trace_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2, default=str)


def build_local_file_processor(trace_dir: Optional[str] = None) -> SimpleSpanProcessor:
    """Return a SimpleSpanProcessor wrapping the local file exporter.

    SimpleSpanProcessor (not Batch) so each span is written on the same thread it
    ends on, while the transport's enrichment for that span is still buffered —
    keeping the pop/merge deterministic for the POC.
    """
    return SimpleSpanProcessor(LocalFileSpanExporter(trace_dir))
