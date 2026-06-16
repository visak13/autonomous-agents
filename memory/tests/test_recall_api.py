"""Unit + integration tests for the PUBLIC recall API (STAGE B / b3).

Unit: the in-house no-regex token estimator, filter validation (fail-fast),
the structure-first classification gate, and the measure-first rerank skip.
Integration (over a persisted store holding BOTH sources):
  - recall returns classified, cited facts drawn from durable facts AND
    compaction summaries;
  - token-bounded SELECTIVE injection caps what re-enters the window (the d4
    lean-context lever) and reports a MEASURED per-recall token cost;
  - the structural path is a deterministic key lookup that does NO embedding;
  - a precise counter (llm_framework.tokens.estimate_tokens) can be injected
    WITHOUT memory importing llm_framework.
"""
from __future__ import annotations

import pytest

from memory.recall_api import (
    MemoryRecall,
    RerankDecision,
    estimate_tokens,
    rerank_decision,
)
from memory.store import (
    COMPACTION_SUMMARY_TYPE,
    CompactionSummaryRecord,
    DurableFactStore,
    MemoryFact,
)


# ------------------------------ unit: tokens ---------------------------- #
def test_estimate_tokens_basic():
    assert estimate_tokens("") == 0
    # word + punctuation pieces are counted; never below 1 for non-empty text.
    assert estimate_tokens("hello") >= 1
    assert estimate_tokens("hello, world!") >= 3
    # char/4 floor dominates for long unbroken strings.
    assert estimate_tokens("x" * 40) >= 10


def test_estimate_tokens_matches_framework_heuristic_shape():
    # The in-house no-regex counter should track the framework's regex heuristic
    # closely (same max(char/4, piece) blend) — proving the yardstick is the same
    # without importing llm_framework into memory.
    from llm_framework.tokens import estimate_tokens as fw

    for s in ["a short note", "RRF fuses BM25 + dense, never raw-score blend!",
              "compaction:sess-1#0 — conversation summary text here"]:
        assert abs(estimate_tokens(s) - fw(s)) <= 2


# ------------------------------ unit: rerank ---------------------------- #
def test_rerank_decision_is_a_documented_skip():
    d = rerank_decision(measured_precision_at_k=1.0)
    assert isinstance(d, RerankDecision)
    assert d.apply is False
    assert d.measured_precision_at_k == 1.0
    assert "torch" in d.reason and "d3" in d.reason


# --------------------------- unit: validation --------------------------- #
def _tiny_store(tmp_path) -> DurableFactStore:
    store = DurableFactStore(tmp_path / "api.db")
    store.add_fact(MemoryFact(
        name="gil", description="cpython gil", type="reference",
        body="The CPython GIL serializes bytecode so one thread runs Python at a time."))
    store.add_fact(MemoryFact(
        name="rrf", description="reciprocal rank fusion", type="feedback",
        body="Fuse BM25 and dense legs with reciprocal rank fusion, never a raw-score blend."))
    store.add_compaction_summary(CompactionSummaryRecord(
        conversation_id="sess-42", event_index=0, reason="auto",
        turns_summarized=8, before_tokens=4300, after_tokens=950,
        summary="The user decided to build a hybrid RAG memory: a CPU MiniLM "
        "embedder with a sqlite-vec store, BM25 and dense legs fused with RRF. "
        "Durable facts and compaction summaries are two sources in one store."))
    return store


def test_filter_validation_rejects_unknown_and_bad(tmp_path):
    with _tiny_store(tmp_path) as store:
        api = MemoryRecall(store)
        with pytest.raises(ValueError):
            api.recall("q", filters={"bogus": "x"})
        with pytest.raises(ValueError):
            api.recall("q", filters={"type": "not-a-type"})
        with pytest.raises(ValueError):
            api.recall("q", filters={"name": 123})
        with pytest.raises(ValueError):
            api.recall("q", k=0)
        with pytest.raises(ValueError):
            api.recall("q", token_budget=0)


def test_classification_gate(tmp_path):
    with _tiny_store(tmp_path) as store:
        api = MemoryRecall(store)
        # free-text, no filter -> semantic
        assert api.recall("how to combine retrieval legs").classification == "semantic"
        # filter + free-text -> hybrid
        assert api.recall("retrieval", filters={"type": "feedback"}).classification == "hybrid"
        # empty query + structural key -> structural (deterministic, no embed)
        assert api.recall("", filters={"name": "gil"}).classification == "structural"
        assert api.recall("", filters={"type": "reference"}).classification == "structural"


# --------- integration: both sources, citations, selective inject -------- #
def test_recall_spans_both_sources_with_citations(tmp_path):
    db = tmp_path / "both.db"
    with DurableFactStore(db) as store:
        store.add_fact(MemoryFact(
            name="gil", description="cpython gil", type="reference",
            body="The CPython GIL serializes bytecode so one thread runs Python at a time."))
        store.add_compaction_summary(CompactionSummaryRecord(
            conversation_id="sess-42", event_index=0, reason="auto",
            turns_summarized=8, before_tokens=4300, after_tokens=950,
            summary="The user decided to build a hybrid RAG memory store with RRF "
            "fusion over BM25 and dense legs; summaries are a second source."))
    # reopen (restart) and recall through the PUBLIC api
    store2 = DurableFactStore(db)
    try:
        api = MemoryRecall(store2)
        resp = api.recall("what did the user decide about building memory?", k=4)
        assert resp.facts, "no facts recalled"
        # the summary wins on its own content and is cited
        assert resp.facts[0].type == COMPACTION_SUMMARY_TYPE
        assert resp.facts[0].citation["fact_name"] == "compaction:sess-42#0"
        # every fact carries an auditable citation + a stable [D#] label
        for f in resp.facts:
            assert set(f.citation) >= {"fact_name", "type", "path", "chunk_index"}
            assert f.label.startswith("D")
        # context block renders the labels for re-injection
        block = resp.to_context_block()
        assert "[D1]" in block
        # restart contract intact: the corpus was NOT re-embedded
        assert store2.embedded_chunk_count == 0
    finally:
        store2.close()


def test_token_budget_bounds_injection(tmp_path):
    with _tiny_store(tmp_path) as store:
        api = MemoryRecall(store)
        # generous budget: nothing truncated, injected == considered
        big = api.recall("retrieval and memory design", k=3, token_budget=10_000)
        assert big.tokens_injected == big.tokens_considered
        assert big.tokens_injected == sum(f.tokens for f in big.facts)

        # tiny budget: only the top fact is injected, the rest are dropped.
        tiny = api.recall("retrieval and memory design", k=3, token_budget=1)
        assert len(tiny.facts) == 1
        assert tiny.truncated_by_budget is True
        assert tiny.tokens_injected <= tiny.tokens_considered
        assert tiny.tokens_saved == tiny.tokens_considered - tiny.tokens_injected
        # the measured cost equals the rendered block's cost (lean-context proof)
        assert tiny.tokens_injected == estimate_tokens(tiny.to_context_block())


def test_structural_lookup_is_deterministic_and_embeds_nothing(tmp_path):
    db = tmp_path / "struct.db"
    with DurableFactStore(db) as store:
        store.add_fact(MemoryFact(name="gil", description="d", type="reference",
                                  body="The CPython GIL serializes bytecode."))
        store.add_fact(MemoryFact(name="rrf", description="d", type="feedback",
                                  body="Fuse legs with RRF, never raw-score blend."))
    # reopen with a SENTINEL embedder that explodes if touched: the structural
    # path must answer from the schema with NO embedding at all.
    class _NoEmbed:
        def embed(self, *a, **k):
            raise AssertionError("structural path must not embed")
        def embed_one(self, *a, **k):
            raise AssertionError("structural path must not embed")

    store2 = DurableFactStore(db, embedder=_NoEmbed())  # type: ignore[arg-type]
    try:
        api = MemoryRecall(store2)
        resp = api.recall("", filters={"name": "gil"})
        assert resp.classification == "structural"
        assert resp.facts and resp.facts[0].fact_name == "gil"
        # determinism: same call, same answer
        again = api.recall("", filters={"name": "gil"})
        assert [f.fact_name for f in again.facts] == [f.fact_name for f in resp.facts]
    finally:
        store2.close()


def test_injectable_precise_counter(tmp_path):
    from llm_framework.tokens import estimate_tokens as fw

    with _tiny_store(tmp_path) as store:
        api = MemoryRecall(store, token_counter=fw)
        resp = api.recall("memory design", k=2, token_budget=10_000)
        # per-fact cost was measured by the INJECTED precise counter (fw), and
        # the injected total is the sum of selected per-fact costs.
        assert resp.tokens_injected == sum(f.tokens for f in resp.facts)
        for f in resp.facts:
            assert f.tokens == fw(f.render())
