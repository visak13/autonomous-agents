"""Structure-first then hybrid (BM25 + dense MiniLM, RRF-fused) recall over
markdown-with-frontmatter memory facts, returning ranked facts WITH citations.

House-style (spec-python-ml-retrieval), in the spec's fixed pipeline order:
  1. STRUCTURE-FIRST  — scope the candidate set by known frontmatter keys
     (topic / name) and answer deterministic structural queries outright
     (e.g. "the most recent X fact" -> recency sort, NO embedding). Embedding
     is the SEMANTIC fallback, never the default.
  2. HYBRID retrieve  — BM25 (sparse lexical) + dense MiniLM (semantic) over the
     scoped set, in parallel.
  3. FUSE with RRF     — Reciprocal Rank Fusion, NEVER raw-score blending (BM25
     scores and cosine distances live on incompatible scales). [required, c1]
  4. CITE              — every hit carries its source frontmatter (path/title/
     fact_id) so the answer is auditable.

Each recall is classified structural / semantic / hybrid by the path actually
taken, recorded for the eval.

BM25 is implemented in-house (small corpus, d10 minimal-deps): no extra runtime
dependency and no regex (a plain str tokenizer), per the spec's no-regex rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from memory.embedder import CpuEmbedder

# RRF dampening constant. 60 is the canonical value from Cormack et al. 2009;
# it keeps any single list's #1 from dominating the fused order.
RRF_K = 60


# --------------------------------------------------------------------------- #
# Fact model + frontmatter parsing (structure-first metadata)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Fact:
    """One memory fact: its frontmatter keys + body, addressable for citation."""

    fact_id: int
    path: str
    title: str
    topic: str
    name: str
    date: str  # ISO yyyy-mm-dd; the recency key for structural queries
    body: str

    @property
    def citation(self) -> dict:
        """The auditable source pointer attached to every ranked hit."""
        return {"fact_id": self.fact_id, "title": self.title, "path": self.path}


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer WITHOUT regex (spec no-regex rule): split on
    any non-alphanumeric char so BM25 sees clean lexical terms."""
    out: list[str] = []
    cur: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split `---`-delimited YAML-ish frontmatter from the body. Minimal and
    dependency-free; values are read as plain strings."""
    meta: dict = {}
    body = text
    if text.startswith("---"):
        _, fm, body = text.split("---", 2)
        for line in fm.strip().splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                meta[key.strip()] = val.strip()
    return meta, body.strip()


def load_corpus(facts_dir: str | Path) -> list[Fact]:
    """Load every *.md frontmatter fact in a directory into Fact records."""
    facts: list[Fact] = []
    for path in sorted(Path(facts_dir).glob("*.md")):
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        facts.append(
            Fact(
                fact_id=int(meta["fact_id"]),
                path=str(path),
                title=meta.get("title", path.stem),
                topic=meta.get("topic", ""),
                name=meta.get("name", path.stem),
                date=meta.get("date", ""),
                body=body,
            )
        )
    return facts


# --------------------------------------------------------------------------- #
# BM25 (Okapi) — sparse lexical leg
# --------------------------------------------------------------------------- #
class BM25:
    """Okapi BM25 over a fixed fact corpus. Scores the body+title text of each
    fact against a tokenized query. In-house to avoid a runtime dep (d10)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens = [_tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.n = len(docs)
        # document frequency per term
        df: dict[str, int] = {}
        for toks in self.doc_tokens:
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        # BM25+ idf (always positive): log(1 + (N - df + 0.5)/(df + 0.5))
        self.idf = {
            t: math.log(1.0 + (self.n - dfi + 0.5) / (dfi + 0.5)) for t, dfi in df.items()
        }
        self.term_freqs: list[dict[str, int]] = []
        for toks in self.doc_tokens:
            tf: dict[str, int] = {}
            for term in toks:
                tf[term] = tf.get(term, 0) + 1
            self.term_freqs.append(tf)

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every doc for the query (index-aligned to the corpus)."""
        q_terms = _tokenize(query)
        out = np.zeros(self.n, dtype=np.float64)
        for i in range(self.n):
            tf = self.term_freqs[i]
            dl = self.doc_len[i]
            s = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * (freq * (self.k1 + 1)) / (denom or 1)
            out[i] = s
        return out


# --------------------------------------------------------------------------- #
# RRF fusion
# --------------------------------------------------------------------------- #
def rrf_fuse(*ranked_id_lists: list[int], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: combine N ranked lists of fact_ids into one fused
    order by summing 1/(k + rank). NEVER blends raw scores (incompatible scales).
    Returns (fact_id, fused_score) sorted by descending fused score."""
    fused: dict[int, float] = {}
    for ranked in ranked_id_lists:
        for rank, fact_id in enumerate(ranked, start=1):
            fused[fact_id] = fused.get(fact_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


# --------------------------------------------------------------------------- #
# Structured query intent (drives the structure-first gate)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QueryIntent:
    """A query plus the OPTIONAL structured constraints that scope it.

    - topic:        frontmatter filter (HYBRID scope, or part of a structural answer)
    - name:         exact frontmatter `name` key  -> deterministic structural lookup
    - most_recent:  answer from the recency key (date) within the scope -> structural
    """

    text: str
    topic: str | None = None
    name: str | None = None
    most_recent: bool = False


@dataclass
class RecallResult:
    """Ranked facts for one query, with the path classification + citations."""

    query: str
    classification: str  # "structural" | "semantic" | "hybrid"
    ranked: list[dict] = field(default_factory=list)  # fact_id, rank, citation, why


# --------------------------------------------------------------------------- #
# The recaller
# --------------------------------------------------------------------------- #
class HybridRecaller:
    """Structure-first then hybrid BM25+dense RRF recall with citations.

    The dense leg reuses the a1 CPU MiniLM embedder; embeddings are computed
    ONCE for the whole corpus at construction (load-once house-style)."""

    def __init__(self, facts: list[Fact], embedder: CpuEmbedder | None = None) -> None:
        self.facts = facts
        self.by_id = {f.fact_id: f for f in facts}
        self.embedder = embedder or CpuEmbedder()
        # Lexical leg indexes title + body so a title-word query still scores.
        self._bm25 = BM25([f"{f.title}\n{f.body}" for f in facts])
        # Dense leg: embed the corpus once; L2-normalized so dot == cosine.
        self._doc_vecs = self.embedder.embed([f.body for f in facts])

    # ---- structural helpers (no embedding) ---- #
    def _scope(self, intent: QueryIntent) -> list[Fact]:
        """Apply deterministic frontmatter filters to scope the candidate set."""
        cands = self.facts
        if intent.topic:
            cands = [f for f in cands if f.topic == intent.topic]
        if intent.name:
            cands = [f for f in cands if f.name == intent.name]
        return cands

    def _dense_rank(self, query: str, cands: list[Fact]) -> list[int]:
        """Dense cosine ranking of candidate fact_ids (best first)."""
        qv = self.embedder.embed_one(query)
        idx = {f.fact_id: i for i, f in enumerate(self.facts)}
        sims = [(f.fact_id, float(self._doc_vecs[idx[f.fact_id]] @ qv)) for f in cands]
        sims.sort(key=lambda kv: kv[1], reverse=True)
        return [fid for fid, _ in sims]

    def _bm25_rank(self, query: str, cands: list[Fact]) -> list[int]:
        """BM25 ranking of candidate fact_ids (best first)."""
        all_scores = self._bm25.scores(query)
        idx = {f.fact_id: i for i, f in enumerate(self.facts)}
        scored = [(f.fact_id, all_scores[idx[f.fact_id]]) for f in cands]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return [fid for fid, _ in scored]

    def recall(self, intent: QueryIntent, k: int = 3) -> RecallResult:
        """Run the structure-first -> hybrid pipeline for one query intent."""
        # ---- 1. STRUCTURE-FIRST: deterministic answers from frontmatter ---- #
        # A "most recent" query (optionally within a topic/name scope) is a pure
        # structural lookup on the recency key — no embedding at all.
        if intent.most_recent:
            cands = self._scope(intent)
            cands = sorted(cands, key=lambda f: f.date, reverse=True)
            ranked = [
                {
                    "rank": i + 1,
                    "fact_id": f.fact_id,
                    "citation": f.citation,
                    "why": "structural: recency sort on frontmatter `date`"
                    + (f" within topic={intent.topic}" if intent.topic else ""),
                }
                for i, f in enumerate(cands[:k])
            ]
            return RecallResult(intent.text, "structural", ranked)

        # An exact-name lookup with no free-text need is also structural.
        if intent.name and not intent.text.strip():
            cands = self._scope(intent)
            ranked = [
                {
                    "rank": i + 1,
                    "fact_id": f.fact_id,
                    "citation": f.citation,
                    "why": f"structural: exact frontmatter name={intent.name}",
                }
                for i, f in enumerate(cands[:k])
            ]
            return RecallResult(intent.text, "structural", ranked)

        # ---- 2/3. HYBRID: scope (if any) -> BM25 + dense -> RRF fuse ---- #
        scoped = bool(intent.topic or intent.name)
        cands = self._scope(intent)
        bm25_order = self._bm25_rank(intent.text, cands)
        dense_order = self._dense_rank(intent.text, cands)
        fused = rrf_fuse(bm25_order, dense_order)

        bm25_pos = {fid: r for r, fid in enumerate(bm25_order, 1)}
        dense_pos = {fid: r for r, fid in enumerate(dense_order, 1)}
        classification = "hybrid" if scoped else "semantic"
        ranked = []
        for i, (fid, score) in enumerate(fused[:k]):
            f = self.by_id[fid]
            ranked.append(
                {
                    "rank": i + 1,
                    "fact_id": fid,
                    "citation": f.citation,
                    "rrf_score": round(score, 6),
                    "bm25_rank": bm25_pos.get(fid),
                    "dense_rank": dense_pos.get(fid),
                    "why": (
                        ("hybrid: frontmatter scope then " if scoped else "semantic: ")
                        + "BM25+dense fused with RRF"
                    ),
                }
            )
        return RecallResult(intent.text, classification, ranked)
