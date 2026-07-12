"""SB-2 (d285) — the UNIVERSAL node FINALIZE contract.

The d285 contract: node FINALIZE is universal — EVERY worker AND EVERY reviewer (not only
the terminal synthesizer) emits a ``(summary, memory_index)`` pair when it finishes. The
prose-only terminal ``finalize_summary`` is generalized into ``Planner.finalize_node`` (the
node contract); the synthesizer is now ONE caller of that same generic finalize.

These tests prove the gate properties on the REAL :meth:`Planner.finalize_node`, built ON
SB-1's :class:`ResearchMemoryStore` (no second store):

  (a) an intermediate RESEARCH WORKER finalizes a ``(summary, memory_index)`` whose index,
      passed back to SB-1's ``get_research_memory_store`` / ``open_memory``, round-trips to the
      SAME memory the worker gathered into (across a FRESH store instance — real disk state);
  (b) a REVIEWER (continuing the SAME memory by index) finalizes a ``(summary, memory_index)``
      whose index round-trips to that SAME memory, and whose MODEL-emitted summary CARRIES the
      d237 data-complexity signal as text (no engine flag/field);
  (c) the SYNTHESIZER is ONE caller — ``finalize_summary`` delegates to ``finalize_node`` and
      returns just the summary string (d215 terminal path unchanged);
  (d) fail-safe: an empty reply yields a derived one-line summary, with the memory_index
      preserved on the pair regardless of the summary path.

Anti-fabrication: the finalize path is ROLE-AGNOSTIC — there is NO spec-name or role-name
conditional in ``finalize_node`` (asserted by reading its own source); the summary is
model-emitted and the engine authors no summary structure.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap

from llm_framework import ChatResult
from agent_runtime import (
    NEW_MEMORY,
    NodeFinalization,
    Planner,
    ResearchMemoryStore,
    get_research_memory_store,
)
from agent_runtime.factory import AbstractPlanFactory
from agent_runtime.research_tree import LeafResult


def _factory() -> AbstractPlanFactory:
    return AbstractPlanFactory([], tool_catalog=[])


class _ScriptedDigest:
    """A transport whose chat returns a fixed model-emitted finalize digest (the node's own
    model writing its summary — the engine never authors it)."""

    def __init__(self, text: str) -> None:
        self._text = text

    def chat(self, messages, **opts) -> ChatResult:
        return ChatResult(role="assistant", content=self._text)

    def complete(self, messages, **opts) -> str:  # pragma: no cover - parity shim
        return self._text


class _EmptyTransport:
    """A transport returning empty content (forces the derived fallback)."""

    def chat(self, messages, **opts) -> ChatResult:
        return ChatResult(role="assistant", content="")

    def complete(self, messages, **opts) -> str:  # pragma: no cover - parity shim
        return ""


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
# (a) a RESEARCH WORKER emits (summary, memory_index) round-tripping to its memory
# --------------------------------------------------------------------------- #
def test_research_worker_finalizes_summary_and_roundtripping_index(tmp_path):
    """An intermediate research worker opens a NEW memory, gathers into it, then FINALIZES a
    (summary, memory_index) pair; the index round-trips to the SAME memory via SB-1's tool."""
    store = get_research_memory_store(tmp_path)
    # the worker opens a NEW memory and gathers (it interacts ONLY by index — SB-1 contract)
    mem = store.open_memory(NEW_MEMORY)
    index = mem.memory_handle
    mem.append_leaf(
        _leaf(
            branch_id="b1",
            question="what happened",
            url="https://example.com/a",
            markdown="The full verbatim body with the figure 1234.",
            note="a key worker claim",
        ),
        layer=1,
    )

    planner = Planner(_ScriptedDigest("Gathered the escalation timeline and one casualty figure."),
                      _factory())
    fin = asyncio.run(planner.finalize_node(
        "detailed report on the conflict",
        memory_index=index,
        work_digest="found the escalation timeline + a casualty figure",
        sources=1,
    ))

    # the universal pair: a MODEL-emitted summary + the memory index the worker used
    assert isinstance(fin, NodeFinalization)
    assert fin.summary.strip() == "Gathered the escalation timeline and one casualty figure."
    assert fin.memory_index == index

    # the index round-trips through SB-1's tool to the SAME memory — even from a FRESH store
    # instance (a later node/turn), proving it resolves the real on-disk memory, not a cache.
    later = get_research_memory_store(tmp_path).open_memory(fin.memory_index)
    assert later.memory_handle == index
    claims = {n.get("claim") for n in later.collect_notes()}
    assert "a key worker claim" in claims
    srcs = later.sources()
    assert any(s["url"] == "https://example.com/a" and "1234" in s["markdown"] for s in srcs)


def test_research_worker_roundtrip_across_fresh_store_instance(tmp_path):
    """The strongest round-trip: a FRESH ResearchMemoryStore over the same root re-opens the
    finalized index and reads the worker's prior notes + verbatim sources back from disk."""
    store = ResearchMemoryStore(tmp_path)
    mem = store.open_memory(NEW_MEMORY)
    mem.append_leaf(
        _leaf(branch_id="b", question="q", url="https://e/x", markdown="body 9876", note="claim-x"),
        layer=1,
    )
    planner = Planner(_ScriptedDigest("Worker digest."), _factory())
    fin = asyncio.run(planner.finalize_node("goal", memory_index=mem.memory_handle))

    reopened = ResearchMemoryStore(tmp_path).open_memory(fin.memory_index)
    assert reopened.memory_handle == fin.memory_index
    assert any(n.get("claim") == "claim-x" for n in reopened.collect_notes())
    assert any("9876" in s["markdown"] for s in reopened.sources())


# --------------------------------------------------------------------------- #
# (b) a REVIEWER emits (summary, memory_index) carrying the d237 data-complexity
# --------------------------------------------------------------------------- #
def test_reviewer_finalizes_same_index_with_data_complexity_in_summary(tmp_path):
    """A reviewer CONTINUES the worker's memory by index and finalizes the SAME index; its
    MODEL-emitted summary carries the d237 data-complexity signal as text (no engine field)."""
    store = get_research_memory_store(tmp_path)
    # worker round (gathers + learns the index)
    worker_mem = store.open_memory(NEW_MEMORY)
    index = worker_mem.memory_handle
    worker_mem.append_leaf(
        _leaf(branch_id="b1", question="q1", url="https://e/1", markdown="b1", note="worker claim"),
        layer=1,
    )

    # reviewer round — opens the SAME memory by index ONLY (never constructs a store)
    reviewer_mem = store.open_memory(index)
    assert reviewer_mem.memory_handle == index  # the reviewer is on the worker's memory

    # the reviewer's model writes a digest that INCLUDES the data-complexity (model-emitted text,
    # not an engine flag) — exactly what the SB-5 planner will read for sectioned-vs-single.
    reviewer_digest = (
        "The research is complete and supports the report. Data complexity: 5 distinct "
        "concerns over a complex multi-section structure with large fetched content."
    )
    planner = Planner(_ScriptedDigest(reviewer_digest), _factory())
    fin = asyncio.run(planner.finalize_node(
        "detailed report on the conflict",
        memory_index=reviewer_mem.memory_handle,
        work_digest="reviewed sufficiency + gaps over the gathered research",
        sources=1,
    ))

    assert isinstance(fin, NodeFinalization)
    # the data-complexity rides INSIDE the model-emitted summary (no separate engine field)
    assert "data complexity" in fin.summary.lower()
    assert "5 distinct concerns" in fin.summary
    assert set(NodeFinalization("", "").as_dict().keys()) == {"summary", "memory_index"}

    # the reviewer's finalized index round-trips to the SAME memory the worker gathered into
    assert fin.memory_index == index
    later = get_research_memory_store(tmp_path).open_memory(fin.memory_index)
    assert later.memory_handle == index
    assert any(n.get("claim") == "worker claim" for n in later.collect_notes())


def test_worker_and_reviewer_finalize_to_one_shared_memory(tmp_path):
    """End-to-end: a worker and a reviewer each emit a (summary, memory_index); BOTH indices
    resolve to the ONE shared memory — the round-trip the gate requires."""
    store = get_research_memory_store(tmp_path)
    worker_mem = store.open_memory(NEW_MEMORY)
    index = worker_mem.memory_handle
    worker_mem.append_leaf(
        _leaf(branch_id="b", question="q", url="https://e/y", markdown="yy", note="c"),
        layer=1,
    )

    planner = Planner(_ScriptedDigest("digest"), _factory())
    worker_fin = asyncio.run(planner.finalize_node("g", memory_index=index))
    reviewer_fin = asyncio.run(
        planner.finalize_node("g", memory_index=store.open_memory(index).memory_handle)
    )

    assert worker_fin.memory_index == reviewer_fin.memory_index == index
    a = get_research_memory_store(tmp_path).open_memory(worker_fin.memory_index)
    b = get_research_memory_store(tmp_path).open_memory(reviewer_fin.memory_index)
    assert a is b  # the per-index singleton — both nodes finalized to ONE memory


# --------------------------------------------------------------------------- #
# (c) the synthesizer is ONE caller of the same generic finalize
# --------------------------------------------------------------------------- #
def test_synthesizer_is_one_caller_of_finalize_node():
    """``finalize_summary`` (the terminal synthesizer's path) returns the model digest string —
    it now delegates to the universal ``finalize_node`` (the synthesizer is not special-cased)."""
    planner = Planner(_ScriptedDigest("Your report on the conflict is ready, covering the "
                                      "timeline and casualty figures."), _factory())
    summary = asyncio.run(planner.finalize_summary(
        "report on the conflict", plans_authored=["research", "write"],
        sources=8, sections=4, artifact="report.html",
    ))
    assert isinstance(summary, str)
    assert "ready" in summary.lower()

    # and it accepts (and carries) a memory_index without changing its string return contract
    summary2 = asyncio.run(planner.finalize_summary(
        "report on the conflict", memory_index="mem_abc", artifact="report.html",
    ))
    assert "ready" in summary2.lower()


def test_finalize_summary_delegates_to_finalize_node(monkeypatch):
    """Prove the delegation seam: finalize_summary CALLS finalize_node and returns its
    ``.summary`` (so the synthesizer is literally one caller of the universal contract)."""
    planner = Planner(_ScriptedDigest("unused"), _factory())
    captured: dict[str, object] = {}

    async def _fake_finalize_node(goal, **kw):
        captured["goal"] = goal
        captured.update(kw)
        return NodeFinalization(summary="delegated summary", memory_index=kw.get("memory_index", ""))

    monkeypatch.setattr(planner, "finalize_node", _fake_finalize_node)
    out = asyncio.run(planner.finalize_summary("g", sources=3, sections=2,
                                               artifact="a.html", memory_index="mem_z"))
    assert out == "delegated summary"
    assert captured["memory_index"] == "mem_z"
    assert captured["sources"] == 3 and captured["sections"] == 2 and captured["artifact"] == "a.html"


# --------------------------------------------------------------------------- #
# (d) fail-safe: empty reply -> derived one-liner, index still preserved
# --------------------------------------------------------------------------- #
def test_finalize_node_fails_open_but_keeps_memory_index():
    """An empty reply (offline seam) yields a minimal factual derived summary, never a crash —
    and the memory_index is preserved on the pair regardless of the summary path."""
    planner = Planner(_EmptyTransport(), _factory())
    fin = asyncio.run(planner.finalize_node(
        "report on X", memory_index="mem_keepme", sources=3, sections=2, artifact="out.html",
    ))
    assert isinstance(fin, NodeFinalization)
    assert "report on X" in fin.summary and "out.html" in fin.summary
    assert fin.memory_index == "mem_keepme"


def test_finalize_summary_fails_open_to_derived():
    """Back-compat: the synthesizer caller still fails open to a derived one-liner (string)."""
    planner = Planner(_EmptyTransport(), _factory())
    summary = asyncio.run(planner.finalize_summary(
        "report on X", plans_authored=["research", "write"],
        sources=3, sections=2, artifact="out.html",
    ))
    assert "report on X" in summary and "out.html" in summary


# --------------------------------------------------------------------------- #
# anti-fabrication: the finalize path is ROLE-AGNOSTIC — zero spec/role conditionals
# --------------------------------------------------------------------------- #
def _code_without_docstring(func) -> str:
    """The function's CODE (statements minus its docstring), so the anti-fabrication check
    inspects real branching, not the prose explaining that there is none."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    fn = tree.body[0]
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # drop the leading docstring expression
    return "\n".join(ast.unparse(stmt) for stmt in body)


def test_finalize_node_has_no_spec_or_role_conditionals():
    code = _code_without_docstring(Planner.finalize_node).lower()
    # no conditional/field keyed on a spec or a node role — the finalize is role-agnostic
    for banned in ("spec_id", "spec_name", "role ==", "role==", "if role", "specialization",
                   "reviewer", "synthesizer", "worker", '"role"', "'role'"):
        assert banned not in code, f"finalize must be role-agnostic; found {banned!r} in code"
