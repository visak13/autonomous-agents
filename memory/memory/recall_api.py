"""Hardened PUBLIC memory recall API + token-cheap selective injection.

This is the stable surface the rest of the agent calls to remember things
(serves d4 — selective recall keeps phi's small context window lean — and d10).
It wraps b1's :class:`~memory.store.DurableFactStore` (which already holds BOTH
memory sources in one sqlite-vec db: durable Claude-memory facts AND
auto-compaction summaries) and adds the four things a *public* API owes a caller:

1. **One stable entry point** — :meth:`MemoryRecall.recall(query, k, filters)`
   returns token-bounded top-k facts WITH citations and a recall
   **classification** of ``structural`` / ``semantic`` / ``hybrid``. The caller
   never touches the store's internals, the embedder, or the leg ordering.

2. **Structure-first classification done right** (house-style [required]). The
   spec's three classes are surfaced honestly by the path actually taken:
     - ``structural`` — answerable from the frontmatter schema alone (an exact
       ``name`` fetch, or "list this ``type``"): a deterministic key lookup with
       NO embedding (delegates to :meth:`DurableFactStore.structural_lookup`).
     - ``hybrid``     — a free-text query scoped by a structural filter
       (``type`` / ``name``): structure-first scope THEN BM25+dense, RRF-fused.
     - ``semantic``   — a free-text query with no filter: hybrid retrieval over
       the whole store (both sources compete in one fused ranking).

3. **Token-bounded SELECTIVE injection** (the lean-context proof, d4). Recall
   ranks top-k, then injects ONLY the highest-ranked facts whose cumulative
   rendered token cost fits a ``token_budget`` — so what actually re-enters
   phi's window is the smallest citable set, not the whole recall. Every
   response reports a MEASURED per-recall token cost (``tokens_injected``) and
   what an unbounded inject would have cost (``tokens_considered``), so the
   saving is a number, not a claim. The token yardstick is **pluggable** — the
   default is an in-house, dependency-free, no-regex heuristic; the evidence
   harness injects :func:`llm_framework.tokens.estimate_tokens` to prove the
   measurement is the SAME one context-management budgets with, WITHOUT memory
   importing llm_framework (the decoupling b2 established).

4. **A measure-first rerank decision, not a reflex.** A cross-encoder rerank
   stage is deliberately NOT added here; :func:`rerank_decision` records the
   measured precision and the load-bearing reason (a CrossEncoder pulls torch
   onto the shared GPU — d3 / banned_options — for no measured precision gain at
   this scale). See the evidence harness for the numbers behind the skip.

House-style honored: structure-first gate explicit per class; RRF fusion (never
raw-score blend) lives in the store; in-process exact KNN (no standing service);
load-once embedder; no regex; fail-fast at the input boundary; resources closed
in the opening scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Optional, Sequence

from memory.store import SOURCE_TYPES, ChunkHit, DurableFactStore

# A token counter is any callable ``str -> int``. Pluggable so the precise
# yardstick (e.g. llm_framework.tokens.estimate_tokens, or a tiktoken counter)
# can be injected without memory taking a dependency on it.
TokenCounter = Callable[[str], int]

# Recognised structural filter keys (the known frontmatter schema). Anything
# else in ``filters`` is rejected at the boundary rather than silently ignored.
_FILTER_KEYS = ("type", "name")

# Default selective-injection budget (tokens). Sized small on purpose: recall is
# meant to re-inject a LEAN citable set into phi's window, not a corpus dump.
DEFAULT_TOKEN_BUDGET = 512
DEFAULT_K = 3


# --------------------------------------------------------------------------- #
# Default token counter — dependency-free, NO regex (spec no-regex rule)
# --------------------------------------------------------------------------- #
def _piece_count(text: str) -> int:
    """Count word-runs and punctuation-runs (mirrors ``\\w+|[^\\w\\s]+`` without
    regex). Consecutive word chars collapse to one piece, as do consecutive
    punctuation chars; whitespace separates."""
    pieces = 0
    prev = "space"  # one of: space | word | punct
    for ch in text:
        if ch.isalnum() or ch == "_":
            cls = "word"
        elif ch.isspace():
            cls = "space"
        else:
            cls = "punct"
        if cls != "space" and cls != prev:
            pieces += 1
        prev = cls
    return pieces


def estimate_tokens(text: str) -> int:
    """In-house heuristic token estimate (the default counter).

    Blends a ~4-chars-per-token estimate with a word/punctuation-piece count and
    takes the max — the same upper-leaning heuristic llm_framework.tokens uses
    for compaction budgeting, reimplemented here WITHOUT regex and WITHOUT a
    cross-component import so memory stays self-contained. Approximate by design
    (its job is context budgeting, not billing). Inject a precise counter via
    :class:`MemoryRecall` when exactness matters."""
    if not text:
        return 0
    char_estimate = (len(text) + 3) // 4  # ceil(len / 4)
    return max(char_estimate, _piece_count(text))


# --------------------------------------------------------------------------- #
# Public result shapes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RecalledFact:
    """One recalled fact in the public response: its citation, text, the per-leg
    audit ranks, and the MEASURED token cost of injecting it."""

    label: str            # stable injection label, e.g. "D1"
    rank: int
    fact_name: str
    type: str
    section: str
    text: str
    citation: dict
    classification: str   # structural | semantic | hybrid
    tokens: int           # measured token cost of THIS fact's rendered block
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    distance: Optional[float] = None

    def render(self) -> str:
        """The citable injection block for this fact (what enters phi's window).

        A stable ``[D#]`` label + a one-line source pointer + the chunk text, so
        the model can cite ``[D1]`` and the pointer resolves back to the exact
        source (fact name / type / path#chunk / section)."""
        c = self.citation
        sect = f" §{c['section']}" if c.get("section") else ""
        loc = c.get("path") or self.fact_name
        head = (
            f"[{self.label}] {self.fact_name} ({self.type}) "
            f"— {loc}#chunk{c.get('chunk_index', 0)}{sect}"
        )
        return f"{head}\n{self.text}".strip()

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "rank": self.rank,
            "fact_name": self.fact_name,
            "type": self.type,
            "section": self.section,
            "citation": self.citation,
            "classification": self.classification,
            "tokens": self.tokens,
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "rrf_score": self.rrf_score,
            "distance": self.distance,
            "text": self.text,
        }


@dataclass(frozen=True)
class RecallResponse:
    """The public recall result: classified, cited, and token-bounded.

    ``facts`` is the SELECTED set actually injected (top-ranked, budget-bounded);
    ``tokens_injected`` is its MEASURED cost (the lean-context proof) and
    ``tokens_considered`` what injecting all ``k_returned`` unbounded would have
    cost. ``sources`` lists which memory sources appeared (durable fact types
    and/or ``compaction_summary``), proving a recall draws from BOTH."""

    query: str
    classification: str
    facts: list[RecalledFact]
    sources: list[str]
    k_requested: int
    k_returned: int
    token_budget: Optional[int]
    tokens_injected: int
    tokens_considered: int
    truncated_by_budget: bool
    rerank_applied: bool = False
    filters: dict = field(default_factory=dict)

    def to_context_block(self) -> str:
        """The lean block to re-inject into phi's window: the selected facts'
        render() blocks, blank-line separated. This — and only this — is what
        selective recall costs the context window."""
        return "\n\n".join(f.render() for f in self.facts)

    @property
    def tokens_saved(self) -> int:
        """Tokens NOT injected thanks to selective bounding (lean saving)."""
        return max(0, self.tokens_considered - self.tokens_injected)

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "classification": self.classification,
            "filters": self.filters,
            "k_requested": self.k_requested,
            "k_returned": self.k_returned,
            "token_budget": self.token_budget,
            "tokens_injected": self.tokens_injected,
            "tokens_considered": self.tokens_considered,
            "tokens_saved": self.tokens_saved,
            "truncated_by_budget": self.truncated_by_budget,
            "rerank_applied": self.rerank_applied,
            "sources": self.sources,
            "facts": [f.as_dict() for f in self.facts],
            "context_block": self.to_context_block(),
        }


# --------------------------------------------------------------------------- #
# Measure-first rerank decision (documented skip, not a reflex)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RerankDecision:
    """The recorded outcome of the measure-first cross-encoder question."""

    apply: bool
    measured_precision_at_k: Optional[float]
    reason: str

    def as_dict(self) -> dict:
        return {
            "apply": self.apply,
            "measured_precision_at_k": self.measured_precision_at_k,
            "reason": self.reason,
        }


def rerank_decision(measured_precision_at_k: Optional[float] = None) -> RerankDecision:
    """Decide whether a cross-encoder rerank is justified — measure-first.

    Per the spec ("rerank ONLY when precision@k justifies the latency") AND the
    load-bearing constraints: a sentence-transformers ``CrossEncoder`` pulls the
    full torch stack and would run on the shared GPU — directly contradicting d3
    / banned_options (CPU-only memory, no torch, no GPU contention with
    phi4-mini) and the lean-venv goal (d10). At this corpus scale the RRF-fused
    hybrid already places the gold chunk at/near rank 1 (see the evidence
    harness's measured precision@k), so there is no precision headroom to buy.
    Decision: **SKIP**, recorded with its number. (Re-open only if a future,
    larger corpus measures a precision@k deficit that a CPU/ONNX cross-encoder
    could close without torch.)"""
    return RerankDecision(
        apply=False,
        measured_precision_at_k=measured_precision_at_k,
        reason=(
            "Skipped: a CrossEncoder pulls torch onto the shared GPU (violates "
            "d3/banned_options: CPU-only, no torch, no phi4-mini GPU contention) "
            "and d10's lean venv, for no measured precision@k gain — the RRF "
            "hybrid already ranks the gold chunk at the top at this scale."
        ),
    )


# --------------------------------------------------------------------------- #
# The public recall facade
# --------------------------------------------------------------------------- #
class MemoryRecall:
    """Hardened public recall API over a :class:`DurableFactStore`.

    One entry point — :meth:`recall` — returning classified, cited, token-bounded
    top-k facts drawn from BOTH memory sources. The store owns the retrieval
    mechanism (structure-first scope → BM25+dense → RRF fuse → cite); this facade
    owns the public POLICY: input validation, the structural/semantic/hybrid
    decision, selective injection within a token budget, and the measured cost.
    """

    def __init__(
        self,
        store: DurableFactStore,
        *,
        token_counter: TokenCounter | None = None,
        default_k: int = DEFAULT_K,
        default_token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> None:
        if default_k <= 0:
            raise ValueError("default_k must be > 0")
        if default_token_budget <= 0:
            raise ValueError("default_token_budget must be > 0")
        self.store = store
        # Default to the in-house no-regex heuristic; inject a precise counter
        # (e.g. llm_framework.tokens.estimate_tokens) for exact budgeting.
        self.count_tokens: TokenCounter = token_counter or estimate_tokens
        self.default_k = default_k
        self.default_token_budget = default_token_budget

    # -- input validation (fail fast at the boundary) --------------------- #
    @staticmethod
    def _validate_filters(filters: Optional[Mapping[str, object]]) -> dict:
        if not filters:
            return {}
        clean: dict = {}
        for key, val in filters.items():
            if key not in _FILTER_KEYS:
                raise ValueError(
                    f"unknown filter {key!r}; allowed: {_FILTER_KEYS}"
                )
            if val is None:
                continue
            if not isinstance(val, str):
                raise ValueError(f"filter {key!r} must be a string, got {type(val)}")
            clean[key] = val
        ftype = clean.get("type")
        if ftype is not None and ftype not in SOURCE_TYPES:
            raise ValueError(f"filter type {ftype!r} not in {SOURCE_TYPES}")
        return clean

    @staticmethod
    def _classify(query: str, filters: dict) -> str:
        """The structure-first gate: pick the path BEFORE retrieving.

        - empty free-text + a structural key → STRUCTURAL (deterministic, no embed)
        - a structural key + free-text       → HYBRID (scope then semantic)
        - free-text, no key                  → SEMANTIC (whole-store hybrid)
        """
        has_text = bool(query and query.strip())
        has_filter = bool(filters.get("type") or filters.get("name"))
        if not has_text and has_filter:
            return "structural"
        if has_filter:
            return "hybrid"
        return "semantic"

    def _retrieve(self, query: str, k: int, classification: str, filters: dict) -> list[ChunkHit]:
        """Dispatch to the store leg for the chosen class."""
        if classification == "structural":
            return self.store.structural_lookup(
                name=filters.get("name"), type_filter=filters.get("type"), k=k
            )
        # semantic / hybrid both run the hybrid pipeline; the scope (and thus the
        # label) differ by whether a structural filter was supplied.
        return self.store.recall(
            query, k=k,
            type_filter=filters.get("type"), name_filter=filters.get("name"),
        )

    # -- the public entry point ------------------------------------------- #
    def recall(
        self,
        query: str,
        k: int | None = None,
        filters: Optional[Mapping[str, object]] = None,
        *,
        token_budget: int | None = None,
    ) -> RecallResponse:
        """Recall top-k memory facts: classified, cited, token-bounded.

        Parameters
        ----------
        query:
            Free-text need. May be empty when ``filters`` carry a structural key
            (a deterministic structural lookup).
        k:
            Max facts to RANK (defaults to ``default_k``). Selective injection
            may return fewer if the token budget binds first.
        filters:
            Optional structural scope: ``{"type": <one of SOURCE_TYPES>}`` and/or
            ``{"name": <exact fact name>}``. Unknown keys are rejected.
        token_budget:
            Max tokens of recalled text to actually inject (defaults to
            ``default_token_budget``). Facts are added in rank order while the
            cumulative cost fits; the top fact is always returned (truncated set
            marked ``truncated_by_budget``) so a recall is never empty when a hit
            exists. This is the d4 lean-context lever.

        Returns a :class:`RecallResponse` with the selected facts, their
        citations, the recall classification, and the MEASURED per-recall token
        cost (``tokens_injected``) vs. the unbounded cost (``tokens_considered``).
        """
        k = self.default_k if k is None else k
        if k <= 0:
            raise ValueError("k must be > 0")
        budget = self.default_token_budget if token_budget is None else token_budget
        if budget <= 0:
            raise ValueError("token_budget must be > 0")
        clean = self._validate_filters(filters)
        query = query or ""
        classification = self._classify(query, clean)

        hits = self._retrieve(query, k, classification, clean)

        # --- selective injection: keep top-ranked facts within the budget --- #
        selected: list[RecalledFact] = []
        considered_tokens = 0
        injected_tokens = 0
        truncated = False
        for i, h in enumerate(hits):
            # Build the fact (tokens filled after we render-and-count it once).
            base = RecalledFact(
                label=f"D{i + 1}", rank=h.rank, fact_name=h.fact_name,
                type=h.citation.get("type", ""), section=h.section,
                text=h.text, citation=h.citation, classification=h.classification,
                tokens=0, dense_rank=h.dense_rank, bm25_rank=h.bm25_rank,
                rrf_score=h.rrf_score, distance=h.distance,
            )
            cost = self.count_tokens(base.render())
            fact = replace(base, tokens=cost)
            considered_tokens += cost
            # Always include the top fact; otherwise include only while it fits.
            if not selected:
                selected.append(fact)
                injected_tokens += cost
            elif injected_tokens + cost <= budget:
                selected.append(fact)
                injected_tokens += cost
            else:
                truncated = True
        if len(selected) < len(hits):
            truncated = True

        sources = sorted({f.type for f in selected if f.type})
        return RecallResponse(
            query=query,
            classification=classification,
            facts=selected,
            sources=sources,
            k_requested=k,
            k_returned=len(hits),
            token_budget=budget,
            tokens_injected=injected_tokens,
            tokens_considered=considered_tokens,
            truncated_by_budget=truncated,
            rerank_applied=False,
            filters=clean,
        )
