"""LOCAL-ONLY enrichment buffer for LLM-call observability (s7 Stage-A POC).

Phoenix spans deliberately carry NO message content and NO chain-of-thought
(the "no secrets in span attributes" rule in ``transport.chat``). That makes
the Phoenix traces unreadable for debugging: you can see token counts and
latency, but never the actual prompt that was sent or the model's reasoning.

This module is the seam that fixes that WITHOUT weakening the Phoenix rule. The
transport records the rich, local-only payload here — keyed by the active span's
hex ``span_id`` — instead of stamping it onto the span. The payload never becomes
a span attribute, so the OTLP exporter pointed at Phoenix never sees it; only the
LOCAL file exporter (``agent_runtime.local_trace``) pops it back out and merges it
into the on-disk trace JSON. Layering is respected: this lives in the lower
``llm_framework`` layer (no back-import of ``agent_runtime``), and the upper layer
reads from it.

The store is a small, self-bounded, thread-safe map. It is a hand-off buffer, not
a log: the file exporter ``pop``s each entry on span end, so in steady state it
holds only the handful of in-flight spans. The bound is a leak-guard for the case
where the local exporter is not installed (e.g. opentelemetry present but the
provider built without our processor) — then nothing pops, so we evict oldest.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

# Leak-guard: if no consumer is popping (local exporter not registered), keep at
# most this many in-flight enrichment records, evicting oldest-first. A normal
# run holds only a few at once because the exporter pops on span end.
_MAX_PENDING = 256

_lock = threading.Lock()
_pending: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def record_llm_capture(span_id: str, payload: Dict[str, Any]) -> None:
    """Buffer one LLM call's LOCAL-ONLY enrichment, keyed by hex ``span_id``.

    Called from ``transport.chat`` inside the active ``llm.chat`` span. ``payload``
    carries the full prompt messages, the ``thinking`` reasoning block, token
    counts, model tag and latency — everything Phoenix omits. Never raises into
    the caller: observability must not break a real model call.
    """
    if not span_id:
        return
    with _lock:
        _pending[span_id] = payload
        _pending.move_to_end(span_id)
        while len(_pending) > _MAX_PENDING:
            # Evict oldest un-popped entry — only reachable when no exporter consumes.
            _pending.popitem(last=False)


def pop_llm_capture(span_id: str) -> Optional[Dict[str, Any]]:
    """Remove and return the enrichment for ``span_id`` (the local exporter's read).

    Returns ``None`` if nothing was buffered for that span (e.g. a non-LLM span,
    or tracing without the transport seam). ``pop`` semantics keep the buffer
    drained so it never grows for the lifetime of the process.
    """
    if not span_id:
        return None
    with _lock:
        return _pending.pop(span_id, None)
