"""SB-1 (d285) — the abstract, index-keyed research-memory SINGLETON tool.

The d285 memory model: nodes NEVER construct or manage a research store directly — they
only pass an INDEX to the tool, which owns creation and lookup. These tests prove the three
gate properties on the REAL :class:`ResearchMemoryStore`, built ON the existing
:class:`ResearchState` + ``memory_handle`` + the ``.sources.jsonl`` sidecar (no second store):

  (a) passing an EXISTING index returns/reuses the SAME memory — same handle, and the prior
      notes + verbatim sources are readable back (including across a FRESH store instance,
      proving the read-back is the real disk state, not just an in-process cache);
  (b) passing the NEW sentinel (or nothing) CREATES a distinct new memory;
  (c) a node interacts ONLY by passing an index — it never creates/manages the store.

Anti-fabrication: the tool is generic and domain-neutral — there is NO spec-name or
role-name conditional anywhere in it (asserted by reading the tool's own source).
"""
from __future__ import annotations

import inspect

from agent_runtime import (
    NEW_MEMORY,
    ResearchMemoryStore,
    ResearchState,
    get_research_memory_store,
)
from agent_runtime.research_tree import LeafResult


def _leaf(*, branch_id: str, question: str, url: str, markdown: str, note: str) -> LeafResult:
    """A minimal gathered leaf carrying one note + one verbatim fetched source."""
    return LeafResult(
        branch_id=branch_id,
        question=question,
        findings=f"digest for {question}",
        notes=[{"claim": note, "url": url}],
        fetched=[{"title": f"title for {url}", "url": url, "markdown": markdown}],
    )


# --------------------------------------------------------------------------- #
# (a) existing index → SAME memory; prior notes + sources readable back
# --------------------------------------------------------------------------- #
def test_existing_index_reuses_same_memory_same_handle(tmp_path):
    store = ResearchMemoryStore(tmp_path)
    mem = store.open_memory("topic-alpha")
    handle = mem.memory_handle
    assert handle == "topic-alpha"  # handle == the index (file stem)

    # Re-opening the same index returns the very SAME live memory (per-index singleton),
    # never a divergent copy.
    again = store.open_memory("topic-alpha")
    assert again is mem
    assert again.memory_handle == handle


def test_existing_index_reads_prior_notes_and_sources_back(tmp_path):
    # A first "node" opens a NEW memory and gathers into it.
    store = ResearchMemoryStore(tmp_path)
    mem = store.open_memory(NEW_MEMORY)
    index = mem.memory_handle
    mem.append_leaf(
        _leaf(
            branch_id="b1",
            question="what happened",
            url="https://example.com/a",
            markdown="The full verbatim body with the figure 1234.",
            note="a key claim",
        ),
        layer=1,
    )

    # A LATER node (a FRESH store instance over the same root — i.e. a different turn/process)
    # opens the SAME index and reads the prior notes + verbatim sources BACK from disk.
    later_store = ResearchMemoryStore(tmp_path)
    reopened = later_store.open_memory(index)
    assert reopened.memory_handle == index

    notes = reopened.collect_notes()
    assert any(n.get("claim") == "a key claim" for n in notes)

    sources = reopened.sources()
    assert len(sources) == 1
    assert sources[0]["url"] == "https://example.com/a"
    assert "1234" in sources[0]["markdown"]  # the verbatim body, read back from the sidecar


# --------------------------------------------------------------------------- #
# (b) NEW sentinel / nothing → a DISTINCT new memory
# --------------------------------------------------------------------------- #
def test_new_sentinel_and_none_create_distinct_memories(tmp_path):
    store = ResearchMemoryStore(tmp_path)
    a = store.open_memory(NEW_MEMORY)
    b = store.open_memory(NEW_MEMORY)
    c = store.open_memory()  # nothing → also a new memory
    d = store.open_memory(None)  # explicit None → also a new memory

    handles = {a.memory_handle, b.memory_handle, c.memory_handle, d.memory_handle}
    assert len(handles) == 4  # every NEW/none is a fresh, distinct index
    assert a is not b is not c is not d


def test_new_memory_is_isolated_from_an_existing_one(tmp_path):
    store = ResearchMemoryStore(tmp_path)
    first = store.open_memory(NEW_MEMORY)
    first.append_leaf(
        _leaf(
            branch_id="b1",
            question="q1",
            url="https://example.com/x",
            markdown="body x",
            note="first-memory claim",
        ),
        layer=1,
    )

    fresh = store.open_memory(NEW_MEMORY)
    assert fresh.memory_handle != first.memory_handle
    # The new memory does NOT see the first memory's notes or sources.
    assert fresh.collect_notes() == []
    assert fresh.sources() == []


def test_blank_index_is_treated_as_new_not_a_shared_blank(tmp_path):
    store = ResearchMemoryStore(tmp_path)
    a = store.open_memory("   ")
    b = store.open_memory("")
    assert a.memory_handle and b.memory_handle  # each got a minted handle
    assert a.memory_handle != b.memory_handle    # not a single shared blank-stem file
    assert a is not b


# --------------------------------------------------------------------------- #
# (c) a node interacts ONLY by passing an index
# --------------------------------------------------------------------------- #
def test_node_continues_research_via_index_only(tmp_path):
    """Simulate two nodes: node-1 opens NEW and gathers; node-2 is handed ONLY the index
    string and continues the SAME memory — neither node ever constructs ResearchState."""
    store = get_research_memory_store(tmp_path)

    def node_one_gather() -> str:
        mem = store.open_memory(NEW_MEMORY)  # passes the NEW sentinel — no construction
        mem.append_leaf(
            _leaf(
                branch_id="b1",
                question="round 1",
                url="https://example.com/1",
                markdown="round-1 verbatim body",
                note="round-1 claim",
            ),
            layer=1,
        )
        return mem.memory_handle  # the index it learned, to hand downstream

    def node_two_continue(index: str) -> ResearchState:
        mem = store.open_memory(index)  # passes ONLY the index — no construction
        mem.append_leaf(
            _leaf(
                branch_id="b2",
                question="round 2",
                url="https://example.com/2",
                markdown="round-2 verbatim body",
                note="round-2 claim",
            ),
            layer=2,
        )
        return mem

    idx = node_one_gather()
    mem2 = node_two_continue(idx)

    # Node-2 saw node-1's research (same memory) and added to it.
    claims = {n.get("claim") for n in mem2.collect_notes()}
    assert {"round-1 claim", "round-2 claim"} <= claims
    urls = {s["url"] for s in mem2.sources()}
    assert {"https://example.com/1", "https://example.com/2"} <= urls


def test_get_store_is_singleton_per_root(tmp_path):
    s1 = get_research_memory_store(tmp_path)
    s2 = get_research_memory_store(tmp_path)
    assert s1 is s2  # one store per root — the abstract tool is a genuine singleton


def test_has_memory_probe_does_not_create(tmp_path):
    store = ResearchMemoryStore(tmp_path)
    assert store.has_memory("never-seen") is False
    mem = store.open_memory("seen")
    mem.append_leaf(
        _leaf(branch_id="b", question="q", url="https://e/x", markdown="m", note="n"),
        layer=1,
    )
    # Persisted on disk → a fresh store can see it exists without opening it.
    assert ResearchMemoryStore(tmp_path).has_memory("seen") is True


# --------------------------------------------------------------------------- #
# anti-fabrication: the tool is generic — zero spec/role conditionals
# --------------------------------------------------------------------------- #
def test_tool_has_no_spec_or_role_conditionals():
    src = inspect.getsource(ResearchMemoryStore)
    lowered = src.lower()
    for banned in ("spec_id", "spec_name", "role ==", "role==", "if role", "specialization"):
        assert banned not in lowered, f"tool must be domain-neutral; found {banned!r}"
