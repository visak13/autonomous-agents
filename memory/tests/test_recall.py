"""Unit + integration tests for the structure-first / hybrid RRF recall layer.

Unit: BM25 ranking, RRF fusion (no raw-score blend), the no-regex tokenizer, and
the structure-first recency path. Integration: the end-to-end HybridRecaller over
a small in-memory corpus recalls the gold fact and classifies the path correctly.
"""
from __future__ import annotations

import numpy as np

from memory.recall import (
    BM25,
    Fact,
    HybridRecaller,
    QueryIntent,
    _tokenize,
    rrf_fuse,
)


# --------------------------- unit: tokenizer --------------------------- #
def test_tokenizer_splits_on_nonalnum_no_regex():
    assert _tokenize("HTTP-404: page_not found!") == ["http", "404", "page", "not", "found"]
    assert _tokenize("") == []


# ------------------------------ unit: BM25 ----------------------------- #
def test_bm25_ranks_exact_keyword_doc_first():
    docs = [
        "rust ownership and the borrow checker guarantee memory safety",
        "python asyncio runs coroutines on an event loop",
        "a database btree index speeds up lookups",
    ]
    bm = BM25(docs)
    scores = bm.scores("borrow checker memory safety")
    assert int(np.argmax(scores)) == 0  # the rust doc wins on exact keywords
    assert scores[1] == 0.0 and scores[2] == 0.0  # no shared terms -> zero


# ------------------------------ unit: RRF ------------------------------ #
def test_rrf_fuses_ranks_not_scores():
    # list A ranks 1 best; list B ranks 3 best. Item appearing high in BOTH wins.
    fused = rrf_fuse([1, 2, 3], [3, 1, 2])
    order = [fid for fid, _ in fused]
    assert order[0] == 1  # rank1+rank2 beats 3's rank3+rank1 and 2's rank2+rank3
    # fused score is a sum of 1/(k+rank) reciprocals, never a raw score blend
    top_score = fused[0][1]
    assert abs(top_score - (1 / 61 + 1 / 62)) < 1e-9


def test_rrf_single_list_preserves_order():
    fused = rrf_fuse([5, 6, 7])
    assert [fid for fid, _ in fused] == [5, 6, 7]


# ----------------- unit: structure-first recency path ------------------ #
def _toy_facts() -> list[Fact]:
    return [
        Fact(1, "p1.md", "Old py", "python", "py-old", "2025-01-01", "the cpython gil serializes threads"),
        Fact(2, "p2.md", "New py", "python", "py-new", "2025-06-01", "dataclasses reduce boilerplate"),
        Fact(3, "m1.md", "Embed", "ml", "ml-embed", "2025-05-01", "embeddings map text to dense vectors close by meaning"),
    ]


def test_structure_first_most_recent_is_deterministic_no_embedding():
    rec = HybridRecaller(_toy_facts())
    res = rec.recall(QueryIntent(text="latest python fact", topic="python", most_recent=True), k=3)
    assert res.classification == "structural"
    assert res.ranked[0]["fact_id"] == 2  # newest python fact by date
    assert "recency" in res.ranked[0]["why"]


# ----------------- integration: end-to-end hybrid recall --------------- #
def test_integration_semantic_recall_hits_gold_with_citation():
    rec = HybridRecaller(_toy_facts())
    res = rec.recall(QueryIntent(text="mapping sentences to points by meaning"), k=3)
    assert res.classification == "semantic"
    top_ids = [r["fact_id"] for r in res.ranked]
    assert 3 in top_ids  # the embeddings fact is recalled
    # every ranked hit carries an auditable citation
    for r in res.ranked:
        assert set(r["citation"]) == {"fact_id", "title", "path"}


def test_integration_hybrid_scope_limits_to_topic():
    rec = HybridRecaller(_toy_facts())
    res = rec.recall(QueryIntent(text="threads and the interpreter lock", topic="python"), k=3)
    assert res.classification == "hybrid"
    # scoped to python facts only -> no ml fact can appear
    assert all(r["fact_id"] in {1, 2} for r in res.ranked)
    assert res.ranked[0]["fact_id"] == 1  # the GIL fact
