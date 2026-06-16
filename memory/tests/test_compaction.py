"""Unit + integration tests for the SECOND memory source (STAGE B / d4):
auto-compaction conversation summaries indexed alongside durable facts.

Unit: the summary record's citable id + provenance, the source-agnostic
chunker tagging the summary type, and the duck-typed llm_framework bridge.
Integration:
  - a persistent store holds BOTH sources; default recall draws from both and a
    type-scoped recall isolates the summary source; citations are present and
    distinguish the source; the cross-source recall survives a reopen WITHOUT
    re-embedding the corpus (restart contract preserved).
  - the REAL llm_framework.context compaction path (driven offline by a
    FakeTransport-backed Conversation) emits a CompactionEvent that the bridge
    indexes and recall then surfaces with a citation.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from memory.compaction import index_compaction_event, record_from_event
from memory.store import (
    COMPACTION_SUMMARY_TYPE,
    CompactionSummaryRecord,
    DurableFactStore,
    MemoryFact,
    chunk_compaction_summary,
)


# ------------------------------ unit: record ---------------------------- #
def test_summary_record_id_and_provenance():
    rec = CompactionSummaryRecord(
        conversation_id="sess-1", event_index=2, summary="the gist",
        reason="auto", turns_summarized=6, before_tokens=4200, after_tokens=900,
    )
    assert rec.source_id == "compaction:sess-1#2"
    prov = rec.provenance
    assert "auto compaction" in prov and "sess-1" in prov
    assert "folded 6 turn(s)" in prov and "4200->900" in prov
    assert rec.provenance_dict["before_tokens"] == 4200


def test_chunk_compaction_summary_tags_type_and_header():
    rec = CompactionSummaryRecord(
        conversation_id="c", event_index=0, summary="user wants a RAG demo built.",
    )
    chunks = chunk_compaction_summary(rec)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.fact_type == COMPACTION_SUMMARY_TYPE
    assert c.fact_name == "compaction:c#0"
    # the metadata header that gets embedded carries the summary type + id
    assert c.header.startswith("[fact: compaction:c#0 | type: compaction_summary")
    assert c.embed_text.startswith(c.header)


# ------------------------- unit: duck-typed bridge ---------------------- #
@dataclass
class _StubEvent:
    """A minimal stand-in for llm_framework's CompactionEvent (duck typing)."""
    summary: str
    reason: str = "auto"
    turns_summarized: int = 3
    before_tokens: int = 5000
    after_tokens: int = 800


def test_record_from_event_reads_duck_typed_event():
    ev = _StubEvent(summary="folded summary text", reason="manual")
    rec = record_from_event(ev, conversation_id="s9", event_index=1)
    assert rec.summary == "folded summary text"
    assert rec.reason == "manual" and rec.turns_summarized == 3
    assert rec.before_tokens == 5000 and rec.after_tokens == 800
    assert rec.source_id == "compaction:s9#1"


def test_record_from_event_requires_summary():
    with pytest.raises(ValueError):
        record_from_event(_StubEvent(summary=""), conversation_id="s", event_index=0)


# ---------------- integration: both sources in one store ---------------- #
def _seed_durable(store: DurableFactStore) -> None:
    for f in [
        MemoryFact(name="gil", description="cpython gil", type="reference",
                   body="The CPython GIL serializes bytecode so one thread runs Python at a time."),
        MemoryFact(name="rrf", description="reciprocal rank fusion", type="feedback",
                   body="Fuse BM25 and dense legs with reciprocal rank fusion, never a raw-score blend."),
    ]:
        store.add_fact(f)


def test_recall_draws_from_both_sources_and_scopes(tmp_path):
    db = tmp_path / "both.db"
    with DurableFactStore(db) as store:
        _seed_durable(store)
        store.add_compaction_summary(CompactionSummaryRecord(
            conversation_id="sess-42", event_index=0, reason="auto",
            turns_summarized=8, before_tokens=4300, after_tokens=950,
            summary="The user decided to build a hybrid RAG memory: a CPU MiniLM "
            "embedder with a sqlite-vec store, BM25 and dense legs fused with RRF. "
            "Durable facts and compaction summaries are two sources in one store.",
        ))
        built = store.chunk_count
        counts = store.source_counts()
    # Both sources are persisted in the SAME store.
    assert counts.get(COMPACTION_SUMMARY_TYPE, 0) >= 1
    assert sum(v for k, v in counts.items() if k != COMPACTION_SUMMARY_TYPE) >= 2

    # ---- reopen (restart) and recall across BOTH sources ---- #
    store2 = DurableFactStore(db)
    try:
        assert store2.chunk_count == built
        # Unscoped recall spans both sources; this query matches the summary best.
        hits = store2.recall("what did the user decide about building memory?", k=4)
        assert hits, "cross-source recall returned nothing"
        types = {h.citation["type"] for h in hits}
        assert COMPACTION_SUMMARY_TYPE in types, "summary source not recalled"
        # the summary should win rank-1 on its own content
        assert hits[0].citation["type"] == COMPACTION_SUMMARY_TYPE
        assert hits[0].fact_name == "compaction:sess-42#0"
        # restart contract intact: proc#2 re-embedded ZERO corpus chunks
        assert store2.embedded_chunk_count == 0
        # every hit carries an auditable citation
        for h in hits:
            assert set(h.citation) >= {"fact_name", "type", "path", "chunk_index"}

        # a durable-fact query still surfaces the durable source alongside.
        mixed = store2.recall("how to combine sparse and dense retrieval", k=4)
        assert any(h.citation["type"] != COMPACTION_SUMMARY_TYPE for h in mixed)

        # structure-first scope to the summary source alone.
        scoped = store2.recall("memory decision", k=3, type_filter=COMPACTION_SUMMARY_TYPE)
        assert scoped and all(
            h.citation["type"] == COMPACTION_SUMMARY_TYPE for h in scoped
        )
        assert scoped[0].classification == "hybrid"
    finally:
        store2.close()


# -------- integration: the REAL llm_framework compaction path ----------- #
def test_real_conversation_compaction_indexed_and_recalled(tmp_path):
    # Drive llm_framework.context compaction fully offline with a scripted
    # transport — no GPU, no live phi (live inference deferred per the brief).
    from llm_framework.context import Conversation
    from llm_framework.transport import FakeTransport

    scripted = (
        "The user is building ReactiveAgents and chose a sqlite-vec memory store "
        "with a CPU MiniLM embedder; they agreed RRF fusion over BM25+dense and "
        "want auto-compaction summaries recalled as a second memory source."
    )
    convo = Conversation(
        system="You are a planning assistant.",
        transport=FakeTransport([scripted]),
        compaction_threshold=80,   # tiny budget so a few turns trigger auto-compact
        keep_recent=2,
    )
    # Append enough turns to cross the threshold and fire an auto-compaction.
    for i in range(8):
        convo.add_user(f"Turn {i}: discuss the memory design in some detail here.")
        convo.add_assistant(f"Reply {i}: acknowledged, captured the decision.")
    convo.compact()  # force one more (manual) to be safe
    assert convo.events, "no compaction fired"
    event = convo.last_compaction
    assert event.summary == scripted  # the scripted summary was used verbatim

    db = tmp_path / "real.db"
    with DurableFactStore(db) as store:
        _seed_durable(store)
        rec, n = index_compaction_event(
            store, event, conversation_id="live-sess", event_index=0,
        )
        assert n >= 1 and rec.summary == scripted
        hits = store.recall("what memory store did the user choose?", k=4)
    # the live-path summary is recalled with a citation, alongside durable facts
    assert hits[0].citation["type"] == COMPACTION_SUMMARY_TYPE
    assert hits[0].fact_name == "compaction:live-sess#0"
