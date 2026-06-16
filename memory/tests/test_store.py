"""Unit + integration tests for the durable Claude-memory fact store (STAGE B).

Unit: Claude-memory frontmatter parse/write round-trip, the type-schema guard,
and structure-aware chunking (multi-section -> multi-chunk, header attached).
Integration: build a persistent sqlite-vec db, CLOSE it, reopen in a *separate*
DurableFactStore instance and recall with citations WITHOUT re-embedding the
corpus (the restart contract, exercised in-process; the subprocess proof lives
in evidence/restart/prove_restart.py).
"""
from __future__ import annotations

import pytest

from memory.store import (
    DurableFactStore,
    MemoryFact,
    chunk_fact,
    parse_memory_fact,
    write_memory_fact,
)


# ----------------------------- unit: format ----------------------------- #
def test_parse_claude_memory_frontmatter_with_nested_type():
    text = (
        "---\n"
        "name: prefers-plain-language\n"
        "description: lead with plain framing\n"
        "metadata:\n"
        "  type: user\n"
        "---\n\n"
        "The user prefers plain language. See [[x]].\n"
    )
    fact = parse_memory_fact(text, path="p.md")
    assert fact.name == "prefers-plain-language"
    assert fact.description == "lead with plain framing"
    assert fact.type == "user"
    assert fact.body.startswith("The user prefers")
    assert fact.path == "p.md"


def test_invalid_type_fails_fast():
    with pytest.raises(ValueError):
        MemoryFact(name="x", description="d", type="notatype", body="b")


def test_write_then_parse_round_trips(tmp_path):
    fact = MemoryFact(
        name="rrf-fact", description="use rrf", type="feedback",
        body="**Why:** scales differ.\n\n**How to apply:** sum 1/(k+rank).",
    )
    p = write_memory_fact(fact, tmp_path / "rrf-fact.md")
    back = parse_memory_fact(p.read_text(encoding="utf-8"), path=str(p))
    assert (back.name, back.description, back.type) == ("rrf-fact", "use rrf", "feedback")
    assert "How to apply" in back.body


# ----------------------------- unit: chunking --------------------------- #
def test_short_fact_yields_single_chunk_with_header():
    fact = MemoryFact(name="s", description="d", type="reference", body="one short line.")
    chunks = chunk_fact(fact)
    assert len(chunks) == 1
    assert chunks[0].header.startswith("[fact: s | type: reference")
    assert chunks[0].embed_text.startswith(chunks[0].header)


def test_multi_section_long_fact_yields_multiple_chunks():
    para = ("word " * 200).strip()  # ~200 words ~= 266 tokens, near the target
    body = f"# Section A\n{para}\n\n# Section B\n{para}\n\n# Section C\n{para}"
    fact = MemoryFact(name="long", description="d", type="reference", body=body)
    chunks = chunk_fact(fact)
    assert len(chunks) >= 3  # at least one chunk per heading section
    # chunk indices are contiguous and sections are captured
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {"Section A", "Section B", "Section C"} <= {c.section for c in chunks}


# ------------- integration: persistence across a reopen ----------------- #
def test_persists_and_recalls_after_reopen_without_reembedding(tmp_path):
    facts = [
        MemoryFact(name="gil", description="cpython gil", type="reference",
                   body="The CPython GIL serializes bytecode so one thread runs Python at a time."),
        MemoryFact(name="rrf", description="reciprocal rank fusion", type="feedback",
                   body="Fuse BM25 and dense legs with reciprocal rank fusion, never a raw-score blend."),
        MemoryFact(name="pref", description="plain language", type="user",
                   body="The user prefers plain-language intake questions over jargon."),
    ]
    db = tmp_path / "facts.db"

    # ---- build process (proc#1 analogue): embed + persist, then CLOSE ---- #
    with DurableFactStore(db) as store:
        for f in facts:
            store.add_fact(f)
        built_chunks = store.chunk_count
        assert store.embedded_chunk_count == built_chunks > 0
    assert db.exists() and db.stat().st_size > 0

    # ---- fresh store over the SAME file (proc#2 analogue) ---- #
    store2 = DurableFactStore(db)
    try:
        assert store2.chunk_count == built_chunks  # read persisted chunks back
        hits = store2.recall("how do I combine sparse and dense retrieval?", k=2)
        assert hits, "recall returned nothing after reopen"
        assert hits[0].fact_name == "rrf"
        # restart contract: proc#2 re-embedded ZERO corpus chunks (query only)
        assert store2.embedded_chunk_count == 0
        # every hit carries an auditable citation
        for h in hits:
            assert set(h.citation) >= {"fact_name", "type", "path", "chunk_index"}
        # structure-first type scope narrows the class to 'hybrid'
        scoped = store2.recall("plain language", k=2, type_filter="user")
        assert scoped and scoped[0].classification == "hybrid"
        assert all(h.citation["type"] == "user" for h in scoped)
    finally:
        store2.close()
