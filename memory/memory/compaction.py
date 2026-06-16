"""Bridge: llm_framework auto-compaction summaries → recallable memory chunks.

STAGE B / d4 — the SECOND memory source. When ``llm_framework.context``
compaction fires it emits a ``CompactionEvent`` (reason, before/after tokens,
the summary, how many turns it folded). This module turns such an event into a
:class:`~memory.store.CompactionSummaryRecord` that
:meth:`~memory.store.DurableFactStore.add_compaction_summary` indexes into the
SAME sqlite-vec store as durable facts — so recall draws from BOTH sources.

DECOUPLING (separation of concerns): the ``memory`` component must NOT depend on
``llm_framework`` (it is the lower layer in the single workspace). So the event
is read by DUCK TYPING — any object exposing ``summary`` / ``reason`` /
``before_tokens`` / ``after_tokens`` / ``turns_summarized`` works, including the
real :class:`llm_framework.context.CompactionEvent`. No import crosses the
boundary; the integration is wired by the caller (e.g. the demo/evidence
script), which legitimately knows both sides.
"""
from __future__ import annotations

from typing import Any, Iterable

from memory.store import CompactionSummaryRecord, DurableFactStore


def record_from_event(
    event: Any,
    *,
    conversation_id: str,
    event_index: int,
    source_path: str = "",
) -> CompactionSummaryRecord:
    """Build a :class:`CompactionSummaryRecord` from a duck-typed compaction event.

    Reads ``event.summary`` plus the provenance fields (``reason``,
    ``turns_summarized``, ``before_tokens``, ``after_tokens``) via ``getattr`` so
    the function never imports — or hard-depends on — ``llm_framework``. Missing
    attributes degrade to sensible defaults rather than raising, so a minimal
    stub event still bridges.
    """
    summary = getattr(event, "summary", None)
    if not summary:
        raise ValueError("compaction event has no summary text to index")
    return CompactionSummaryRecord(
        conversation_id=conversation_id,
        event_index=event_index,
        summary=str(summary),
        reason=str(getattr(event, "reason", "auto")),
        turns_summarized=int(getattr(event, "turns_summarized", 0) or 0),
        before_tokens=int(getattr(event, "before_tokens", 0) or 0),
        after_tokens=int(getattr(event, "after_tokens", 0) or 0),
        source_path=source_path,
    )


def index_compaction_event(
    store: DurableFactStore,
    event: Any,
    *,
    conversation_id: str,
    event_index: int,
    source_path: str = "",
) -> tuple[CompactionSummaryRecord, int]:
    """Bridge + index one compaction event into ``store`` (source #2).

    Returns the built record and the number of chunks indexed, so the caller can
    log exactly what landed for evidence."""
    record = record_from_event(
        event,
        conversation_id=conversation_id,
        event_index=event_index,
        source_path=source_path,
    )
    n = store.add_compaction_summary(record)
    return record, n


def index_conversation_compactions(
    store: DurableFactStore,
    events: Iterable[Any],
    *,
    conversation_id: str,
    source_path: str = "",
) -> list[tuple[CompactionSummaryRecord, int]]:
    """Index every compaction event a conversation produced (e.g.
    ``Conversation.events``), assigning each a stable event index. Returns the
    per-event (record, chunks_indexed) pairs."""
    out: list[tuple[CompactionSummaryRecord, int]] = []
    for i, event in enumerate(events):
        out.append(
            index_compaction_event(
                store, event, conversation_id=conversation_id,
                event_index=i, source_path=source_path,
            )
        )
    return out
