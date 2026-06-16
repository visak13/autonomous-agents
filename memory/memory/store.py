"""In-process sqlite-vec KNN store for 384-d memory-fact embeddings.

House-style (spec-python-ml-retrieval): in-process vector store for a small/medium
corpus — no standing vector-DB service. Exact KNN (sqlite-vec brute force) is the
right index at this scale; ANN is unjustified until exact search misses a latency
bar. Each fact carries its frontmatter metadata (path/title) in a sibling table so
a structural filter can scope before the dense rank later.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sqlite_vec

from memory.embedder import DIM, CpuEmbedder
from memory.recall import BM25, rrf_fuse


class VecStore:
    """sqlite-vec backed store: vec0 virtual table + a metadata sidecar table."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.db = sqlite3.connect(self.db_path)
        # sqlite-vec ships as a loadable extension.
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._create_schema()

    def _create_schema(self) -> None:
        self.db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_facts "
            f"USING vec0(fact_id INTEGER PRIMARY KEY, embedding FLOAT[{DIM}])"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS facts "
            "(fact_id INTEGER PRIMARY KEY, path TEXT, title TEXT, body TEXT)"
        )
        self.db.commit()

    def add(self, fact_id: int, path: str, title: str, body: str, vec: np.ndarray) -> None:
        """Insert one fact: its metadata row + its embedding."""
        v = np.asarray(vec, dtype=np.float32)
        if v.shape != (DIM,):
            raise ValueError(f"expected ({DIM},) vector, got {v.shape}")
        self.db.execute(
            "INSERT INTO facts(fact_id, path, title, body) VALUES (?, ?, ?, ?)",
            (fact_id, path, title, body),
        )
        self.db.execute(
            "INSERT INTO vec_facts(fact_id, embedding) VALUES (?, ?)",
            (fact_id, sqlite_vec.serialize_float32(v.tolist())),
        )
        self.db.commit()

    def knn(self, query_vec: np.ndarray, k: int = 5) -> list[dict]:
        """Dense KNN: return the k nearest facts ranked by ascending distance."""
        q = np.asarray(query_vec, dtype=np.float32)
        rows = self.db.execute(
            """
            SELECT v.fact_id, v.distance, f.path, f.title, f.body
            FROM vec_facts v
            JOIN facts f ON f.fact_id = v.fact_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (sqlite_vec.serialize_float32(q.tolist()), k),
        ).fetchall()
        return [
            {
                "rank": i + 1,
                "fact_id": r[0],
                "distance": round(float(r[1]), 6),
                "path": r[2],
                "title": r[3],
                "body": r[4],
            }
            for i, r in enumerate(rows)
        ]

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "VecStore":
        return self

    def __exit__(self, *exc) -> None:
        # Close the resource in the scope that opened it.
        self.close()


# =========================================================================== #
# Durable Claude-memory fact store  (STAGE B)
# =========================================================================== #
# The Claude-memory fact format (distinct from a2's title/topic/fact_id facts):
#
#   ---
#   name: <short-kebab-slug>
#   description: <one-line summary used for relevance at recall>
#   metadata:
#     type: user | feedback | project | reference
#   ---
#   <body — the fact; for feedback/project, Why:/How-to-apply: lines; [[links]]>
#
# This layer is structure-FIRST (the frontmatter type/name is the known schema)
# and persists chunk embeddings into an on-disk sqlite-vec db so recall survives
# a real process restart with NO corpus re-embedding — proc#2 embeds only the
# query and reads the corpus vectors back from disk. House-style: in-process
# sqlite-vec (no standing service); MiniLM dense leg + a2 BM25 sparse leg fused
# with RRF (never raw-score blend, c1/[required]); load models once.

# Valid Claude-memory fact types (the known structural schema for this store).
FACT_TYPES = ("user", "feedback", "project", "reference")
# The SECOND memory source (STAGE B / d4): auto-compaction conversation summaries.
# These are NOT authored Claude-memory facts — they are runtime artifacts emitted
# by llm_framework.context compaction — so the type lives OUTSIDE FACT_TYPES (a
# MemoryFact can never masquerade as a summary) yet is indexed into the SAME
# sqlite-vec chunk store and competes in the SAME hybrid recall.
COMPACTION_SUMMARY_TYPE = "compaction_summary"
# Every type the store can hold / recall can scope to. Default recall (no scope)
# spans ALL of these, so a query draws from BOTH sources at once.
SOURCE_TYPES = FACT_TYPES + (COMPACTION_SUMMARY_TYPE,)
# Structure-aware chunk sizing. ~256-512 tokens; we size in WORDS and convert
# with a ~1.33 tokens/word heuristic (English subword average) to stay in band
# without dragging a tokenizer dep into the lean venv (d10).
TOKENS_PER_WORD = 1.33
TARGET_TOKENS = 384          # mid-band target
MAX_TOKENS = 512             # hard ceiling per chunk
_TARGET_WORDS = int(TARGET_TOKENS / TOKENS_PER_WORD)   # ~288
_MAX_WORDS = int(MAX_TOKENS / TOKENS_PER_WORD)         # ~385


@dataclass(frozen=True)
class MemoryFact:
    """One durable Claude-memory fact: frontmatter schema + body, citable by name."""

    name: str
    description: str
    type: str
    body: str
    path: str = ""

    def __post_init__(self) -> None:
        # Fail fast at the input boundary: the type is the structural key.
        if self.type not in FACT_TYPES:
            raise ValueError(f"fact type {self.type!r} not in {FACT_TYPES}")


@dataclass(frozen=True)
class Chunk:
    """A structure-aware slice of a fact, carrying a per-chunk metadata header.

    The header is PREPENDED to the embedded text so the dense vector and the
    sparse BM25 terms both see the fact's identity/section — and it is what a
    recalled hit cites back to (auditable source pointer)."""

    fact_name: str
    fact_type: str
    description: str
    source_path: str
    chunk_index: int
    section: str   # the markdown heading this chunk fell under ("" = preamble)
    text: str      # the chunk body (without the header line)

    @property
    def header(self) -> str:
        """The per-chunk metadata header line (structure-aware grounding)."""
        sect = f" | section: {self.section}" if self.section else ""
        return f"[fact: {self.fact_name} | type: {self.fact_type}{sect}]"

    @property
    def embed_text(self) -> str:
        """Header + body — what actually gets embedded / BM25-indexed."""
        return f"{self.header}\n{self.text}"

    @property
    def citation(self) -> dict:
        """Auditable source pointer attached to every ranked hit."""
        return {
            "fact_name": self.fact_name,
            "type": self.fact_type,
            "path": self.source_path,
            "chunk_index": self.chunk_index,
            "section": self.section,
        }


# --------------------------------------------------------------------------- #
# Claude-memory frontmatter parse + write  (no regex — spec no-regex rule)
# --------------------------------------------------------------------------- #
def parse_memory_fact(text: str, path: str = "") -> MemoryFact:
    """Parse a Claude-memory markdown-frontmatter fact (name/description/
    metadata.type + body). Dependency-free, no regex: a small indentation-aware
    line scanner handles the one nested block (`metadata:` -> `type:`)."""
    if not text.startswith("---"):
        raise ValueError("fact missing '---' frontmatter delimiter")
    _, fm, body = text.split("---", 2)
    name = description = ftype = ""
    in_metadata = False
    for raw in fm.splitlines():
        if not raw.strip():
            continue
        indented = raw[0] in (" ", "\t")
        key, _, val = raw.strip().partition(":")
        key, val = key.strip(), val.strip()
        if key == "metadata" and not val:
            in_metadata = True
            continue
        if in_metadata and indented:
            if key == "type":
                ftype = val
            continue
        in_metadata = False  # de-dented back to a top-level key
        if key == "name":
            name = val
        elif key == "description":
            description = val
        elif key == "type" and not ftype:  # tolerate a flat `type:` too
            ftype = val
    return MemoryFact(
        name=name, description=description, type=ftype, body=body.strip(), path=path
    )


def write_memory_fact(fact: MemoryFact, path: str | Path) -> Path:
    """Write a MemoryFact to disk in the canonical Claude-memory format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "---\n"
        f"name: {fact.name}\n"
        f"description: {fact.description}\n"
        "metadata:\n"
        f"  type: {fact.type}\n"
        "---\n\n"
        f"{fact.body}\n",
        encoding="utf-8",
    )
    return p


def load_memory_facts(facts_dir: str | Path) -> list[MemoryFact]:
    """Load every *.md Claude-memory fact in a directory."""
    out: list[MemoryFact] = []
    for p in sorted(Path(facts_dir).glob("*.md")):
        out.append(parse_memory_fact(p.read_text(encoding="utf-8"), path=str(p)))
    return out


# --------------------------------------------------------------------------- #
# Structure-aware chunking  (~256-512 tokens, per-chunk metadata header)
# --------------------------------------------------------------------------- #
def _word_count(text: str) -> int:
    return len(text.split())


def chunk_body(
    *, name: str, ftype: str, description: str, source_path: str, body: str
) -> list[Chunk]:
    """Structure-aware chunker (SOURCE-AGNOSTIC core).

    Split ``body`` on markdown headings into sections, then pack each section's
    paragraphs into ~TARGET_WORDS chunks (never exceeding _MAX_WORDS). A short
    body yields a single chunk; every chunk carries the source's metadata header.
    Splitting on STRUCTURE first (not a blind fixed window) keeps related lines
    together and the header intact. Both source types reuse this — a durable
    Claude-memory fact (:func:`chunk_fact`) and an auto-compaction summary
    (:func:`chunk_compaction_summary`) — so chunking has a SINGLE responsibility
    and stays identical across sources.
    """
    # Group lines into (section_heading, [paragraph, ...]).
    sections: list[tuple[str, list[str]]] = [("", [])]
    para: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):  # a markdown heading starts a new section
            if para:
                sections[-1][1].append("\n".join(para).strip())
                para = []
            sections.append((stripped.lstrip("# ").strip(), []))
        elif not stripped:  # blank line ends a paragraph
            if para:
                sections[-1][1].append("\n".join(para).strip())
                para = []
        else:
            para.append(line)
    if para:
        sections[-1][1].append("\n".join(para).strip())

    def mk(idx: int, section: str, text: str) -> Chunk:
        return Chunk(
            fact_name=name, fact_type=ftype, description=description,
            source_path=source_path, chunk_index=idx, section=section, text=text,
        )

    chunks: list[Chunk] = []
    idx = 0
    for heading, paras in sections:
        paras = [p for p in paras if p]
        buf: list[str] = []
        buf_words = 0
        for p in paras:
            pw = _word_count(p)
            # Flush if adding this paragraph would exceed the ceiling.
            if buf and buf_words + pw > _MAX_WORDS:
                chunks.append(mk(idx, heading, "\n\n".join(buf)))
                idx += 1
                buf, buf_words = [], 0
            buf.append(p)
            buf_words += pw
            # Flush once we've reached the mid-band target (keeps chunks ~in band).
            if buf_words >= _TARGET_WORDS:
                chunks.append(mk(idx, heading, "\n\n".join(buf)))
                idx += 1
                buf, buf_words = [], 0
        if buf:
            chunks.append(mk(idx, heading, "\n\n".join(buf)))
            idx += 1

    if not chunks:  # an empty body still yields one (header-only) chunk
        chunks.append(mk(0, "", ""))
    return chunks


def chunk_fact(fact: MemoryFact) -> list[Chunk]:
    """Chunk a durable Claude-memory fact (delegates to :func:`chunk_body`)."""
    return chunk_body(
        name=fact.name, ftype=fact.type, description=fact.description,
        source_path=fact.path, body=fact.body,
    )


# --------------------------------------------------------------------------- #
# Second memory source: auto-compaction conversation summaries  (STAGE B / d4)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CompactionSummaryRecord:
    """One auto-compaction summary to index as a recallable memory chunk.

    This is the memory-side, llm_framework-INDEPENDENT shape of a compaction
    summary (the bridge in :mod:`memory.compaction` builds one from a duck-typed
    llm_framework ``CompactionEvent`` — memory never imports llm_framework, so
    the two components stay decoupled). It carries the summary text PLUS the
    PROVENANCE the action calls for: which conversation it came from, which
    compaction event, why it fired (auto threshold / manual), how many turns it
    folded, and the before/after token counts.

    ``source_id`` becomes the chunk's citable ``fact_name`` and ``provenance``
    its one-line ``description``, so a recalled summary cites back to its exact
    origin alongside durable-fact citations.
    """

    conversation_id: str
    event_index: int          # 0-based index of this compaction within the convo
    summary: str
    reason: str = "auto"      # "auto" (threshold crossed) | "manual"
    turns_summarized: int = 0
    before_tokens: int = 0
    after_tokens: int = 0
    source_path: str = ""     # optional on-disk pointer (e.g. a transcript file)

    @property
    def source_id(self) -> str:
        """Stable, citable id for this summary chunk's ``fact_name``."""
        return f"compaction:{self.conversation_id}#{self.event_index}"

    @property
    def provenance(self) -> str:
        """One-line provenance string stored as the chunk ``description``."""
        return (
            f"{self.reason} compaction of conversation '{self.conversation_id}' "
            f"(event #{self.event_index}): folded {self.turns_summarized} turn(s), "
            f"{self.before_tokens}->{self.after_tokens} tokens"
        )

    @property
    def provenance_dict(self) -> dict:
        """Structured provenance for evidence dumps."""
        return {
            "conversation_id": self.conversation_id,
            "event_index": self.event_index,
            "reason": self.reason,
            "turns_summarized": self.turns_summarized,
            "before_tokens": self.before_tokens,
            "after_tokens": self.after_tokens,
            "source_path": self.source_path,
        }


def chunk_compaction_summary(record: CompactionSummaryRecord) -> list[Chunk]:
    """Chunk a compaction summary structure-aware, tagged ``compaction_summary``.

    Reuses the SAME source-agnostic :func:`chunk_body`, so a summary is sliced,
    headered and sized identically to a durable fact — only the ``fact_type``
    differs, marking it as the second source. The provenance rides along as the
    chunk ``description`` (and the source-id as the citable name)."""
    return chunk_body(
        name=record.source_id,
        ftype=COMPACTION_SUMMARY_TYPE,
        description=record.provenance,
        source_path=record.source_path,
        body=record.summary,
    )


# --------------------------------------------------------------------------- #
# Persistent chunk store + restart-safe recall
# --------------------------------------------------------------------------- #
@dataclass
class ChunkHit:
    """A recalled chunk with its citation + per-leg ranks (auditable)."""

    rank: int
    fact_name: str
    section: str
    citation: dict
    text: str
    classification: str
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    distance: float | None = None


class DurableFactStore:
    """Durable Claude-memory facts → structure-aware chunks → persistent
    sqlite-vec db, with restart-safe hybrid recall.

    PERSISTENCE CONTRACT (the STAGE B proof): the embeddings live on disk in
    sqlite-vec. A *building* process embeds each chunk ONCE and `add`s it; a
    SEPARATE later process opens the same db file and `recall`s — it embeds
    ONLY the query (one vector) and reads every corpus vector back from disk.
    No corpus re-embedding, no design churn (reuses a1 CpuEmbedder + a2 BM25/
    RRF). `embedded_chunk_count` exposes how many corpus chunks THIS instance
    embedded, so a restart proof can assert proc#2 embedded zero.
    """

    def __init__(self, db_path: str | Path, embedder: CpuEmbedder | None = None) -> None:
        self.db_path = str(db_path)
        self.db = sqlite3.connect(self.db_path)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._embedder = embedder
        # Count of CORPUS chunks this instance embedded (query embeds excluded).
        self.embedded_chunk_count = 0
        self._create_schema()

    # ---- lazy, load-once embedder (so a read-only reopen needn't build one
    # unless it actually issues a query) ---- #
    @property
    def embedder(self) -> CpuEmbedder:
        if self._embedder is None:
            self._embedder = CpuEmbedder()
        return self._embedder

    def _create_schema(self) -> None:
        self.db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks "
            f"USING vec0(chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{DIM}])"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "chunk_id INTEGER PRIMARY KEY, fact_name TEXT, fact_type TEXT, "
            "description TEXT, source_path TEXT, chunk_index INTEGER, "
            "section TEXT, text TEXT)"
        )
        self.db.commit()

    # ---- build side (proc#1) ---- #
    def _index_chunks(self, chunks: list[Chunk]) -> int:
        """Embed each chunk ONCE and persist its metadata row + vector. Shared by
        BOTH sources (durable facts and compaction summaries) so the embed/persist
        path is identical and single-responsibility. Returns chunks indexed."""
        if not chunks:
            return 0
        vecs = self.embedder.embed([c.embed_text for c in chunks])
        self.embedded_chunk_count += len(chunks)
        for c, v in zip(chunks, vecs):
            cur = self.db.execute(
                "INSERT INTO chunks(fact_name, fact_type, description, source_path, "
                "chunk_index, section, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (c.fact_name, c.fact_type, c.description, c.source_path,
                 c.chunk_index, c.section, c.text),
            )
            self.db.execute(
                "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                (cur.lastrowid, sqlite_vec.serialize_float32(v.tolist())),
            )
        self.db.commit()
        return len(chunks)

    def add_fact(self, fact: MemoryFact) -> int:
        """Index a durable Claude-memory fact (source #1). Returns chunks indexed."""
        return self._index_chunks(chunk_fact(fact))

    def add_compaction_summary(self, record: CompactionSummaryRecord) -> int:
        """Index an auto-compaction conversation summary (source #2, STAGE B/d4).

        The summary lands in the SAME sqlite-vec store as durable facts, tagged
        ``fact_type=compaction_summary`` with provenance, so :meth:`recall` draws
        from BOTH sources at once. Returns the number of chunks indexed."""
        return self._index_chunks(chunk_compaction_summary(record))

    def index_dir(self, facts_dir: str | Path) -> int:
        """Load + index every Claude-memory fact in a directory. Returns total
        chunks indexed."""
        total = 0
        for fact in load_memory_facts(facts_dir):
            total += self.add_fact(fact)
        return total

    def source_counts(self) -> dict[str, int]:
        """Persisted chunk count per ``fact_type`` — proves both sources coexist
        in one store (durable facts + compaction summaries)."""
        rows = self.db.execute(
            "SELECT fact_type, COUNT(*) FROM chunks GROUP BY fact_type"
        ).fetchall()
        return {r[0]: int(r[1]) for r in rows}

    @property
    def chunk_count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    # ---- recall side (proc#2, restart-safe) ---- #
    def _row(self, chunk_id: int) -> dict:
        r = self.db.execute(
            "SELECT fact_name, fact_type, description, source_path, chunk_index, "
            "section, text FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return {
            "fact_name": r[0], "fact_type": r[1], "description": r[2],
            "source_path": r[3], "chunk_index": r[4], "section": r[5], "text": r[6],
        }

    def _citation(self, row: dict) -> dict:
        return {
            "fact_name": row["fact_name"], "type": row["fact_type"],
            "path": row["source_path"], "chunk_index": row["chunk_index"],
            "section": row["section"],
        }

    def structural_lookup(
        self, *, name: str | None = None, type_filter: str | None = None, k: int = 3
    ) -> list[ChunkHit]:
        """STRUCTURAL path: a deterministic, NO-EMBEDDING lookup by the known
        frontmatter schema keys (``fact_name`` / ``fact_type``).

        The spec's structure-first gate (house-style [required]): when a query is
        answerable from the schema alone — an exact-name fetch, or "list this
        type" — return rows by key with NO dense or sparse leg and NO embedding
        at all. Rows are ordered deterministically (name, then chunk order).
        Hits are classified ``'structural'`` and carry their citation just like a
        hybrid hit, so the public API surfaces all three classes uniformly."""
        if type_filter is not None and type_filter not in SOURCE_TYPES:
            raise ValueError(f"type_filter {type_filter!r} not in {SOURCE_TYPES}")
        clauses: list[str] = []
        params: list[str] = []
        if name is not None:
            clauses.append("fact_name = ?")
            params.append(name)
        if type_filter is not None:
            clauses.append("fact_type = ?")
            params.append(type_filter)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.db.execute(
            "SELECT chunk_id FROM chunks" + where
            + " ORDER BY fact_name, chunk_index",
            params,
        ).fetchall()
        hits: list[ChunkHit] = []
        for i, (cid,) in enumerate(rows[:k]):
            row = self._row(cid)
            hits.append(ChunkHit(
                rank=i + 1,
                fact_name=row["fact_name"],
                section=row["section"],
                citation=self._citation(row),
                text=row["text"],
                classification="structural",
            ))
        return hits

    def recall(
        self,
        query: str,
        k: int = 3,
        type_filter: str | None = None,
        name_filter: str | None = None,
    ) -> list[ChunkHit]:
        """Restart-safe hybrid recall over the PERSISTED db, spanning BOTH sources.

        Pipeline (a2 house-style order): structure-first scope by frontmatter
        `type`/`name` (deterministic filter, no embedding) → dense leg =
        sqlite-vec KNN reading on-disk vectors (query embedded ONCE) → sparse leg
        = a2 BM25 over the scoped chunk text → RRF fuse (never raw-score blend) →
        cite. Classification reflects the path taken: 'hybrid' when a structural
        filter scoped the candidates, else 'semantic'. (The pure deterministic
        'structural' path — empty free-text query — is :meth:`structural_lookup`.)

        With both filters ``None`` the scope is EVERY chunk — durable facts AND
        compaction summaries — so the two sources compete in one fused ranking
        (STAGE B/d4). ``type_filter`` scopes to one source/fact-type;
        ``name_filter`` scopes to a single named fact (e.g. a hybrid lookup
        within one fact's chunks)."""
        # 1. STRUCTURE-FIRST scope (the known schema is the frontmatter
        #    `type`/`name`). Pre-filter; never post-filter what we can pre-filter.
        if type_filter is not None and type_filter not in SOURCE_TYPES:
            raise ValueError(f"type_filter {type_filter!r} not in {SOURCE_TYPES}")
        clauses: list[str] = []
        params: list[str] = []
        if type_filter is not None:
            clauses.append("fact_type = ?")
            params.append(type_filter)
        if name_filter is not None:
            clauses.append("fact_name = ?")
            params.append(name_filter)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        scope_rows = self.db.execute(
            "SELECT chunk_id FROM chunks" + where, params
        ).fetchall()
        scoped_ids = {r[0] for r in scope_rows}
        if not scoped_ids:
            return []
        classification = "hybrid" if (type_filter or name_filter) else "semantic"

        # 2. DENSE leg — KNN over PERSISTED vectors; only the query is embedded.
        qv = np.asarray(self.embedder.embed_one(query), dtype=np.float32)
        # sqlite-vec KNN matches over the WHOLE vec table and cannot inline-filter
        # to the structural scope, so we over-pull then intersect. If a filter
        # scopes to a SUBSET, a narrow global window can come back mostly
        # out-of-scope and starve the dense leg (post-filtering what we already
        # pre-filtered structurally). So widen the window to the full table when a
        # filter is active — cheap at this corpus scale (exact KNN, no ANN, per
        # house-style). With no filter the scope is everything, so the top-(k*4)
        # window is already the genuine candidate set.
        total = self.chunk_count
        knn_k = total if (type_filter or name_filter) else min(total, max(k * 4, k))
        dense_rows = self.db.execute(
            "SELECT chunk_id, distance FROM vec_chunks "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (sqlite_vec.serialize_float32(qv.tolist()), knn_k),
        ).fetchall()
        dense_order = [cid for cid, _ in dense_rows if cid in scoped_ids]
        dist_by_id = {cid: float(d) for cid, d in dense_rows}

        # 3. SPARSE leg — a2 BM25 over the scoped chunk text (no embedding).
        scoped = [(cid, self._row(cid)) for cid in scoped_ids]
        bm = BM25([f"{row['fact_name']} {row['section']}\n{row['text']}"
                   for _, row in scoped])
        bm_scores = bm.scores(query)
        bm_ranked = sorted(
            zip((cid for cid, _ in scoped), bm_scores),
            key=lambda kv: kv[1], reverse=True,
        )
        bm25_order = [cid for cid, _ in bm_ranked]

        # 4. RRF FUSE the two ranked id-lists (c1/[required]: never blend scores).
        fused = rrf_fuse(dense_order, bm25_order)
        dense_pos = {cid: r for r, cid in enumerate(dense_order, 1)}
        bm25_pos = {cid: r for r, cid in enumerate(bm25_order, 1)}

        # 5. CITE.
        hits: list[ChunkHit] = []
        for i, (cid, score) in enumerate(fused[:k]):
            row = self._row(cid)
            hits.append(ChunkHit(
                rank=i + 1,
                fact_name=row["fact_name"],
                section=row["section"],
                citation=self._citation(row),
                text=row["text"],
                classification=classification,
                dense_rank=dense_pos.get(cid),
                bm25_rank=bm25_pos.get(cid),
                rrf_score=round(float(score), 6),
                distance=round(dist_by_id[cid], 6) if cid in dist_by_id else None,
            ))
        return hits

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "DurableFactStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
