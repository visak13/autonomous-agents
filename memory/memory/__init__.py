"""In-process memory for the reactive agent.

CPU-only all-MiniLM-L6-v2 (384d) embedder + sqlite-vec KNN store. No torch-GPU,
no Ollama (d3 / banned_options). Structure-first then dense recall (this POC
exercises the dense leg; the frontmatter structural filter scopes it later).
"""
from memory.embedder import CpuEmbedder, MODEL_NAME, DIM
from memory.store import (
    VecStore,
    DurableFactStore,
    MemoryFact,
    Chunk,
    ChunkHit,
    CompactionSummaryRecord,
    FACT_TYPES,
    COMPACTION_SUMMARY_TYPE,
    SOURCE_TYPES,
    parse_memory_fact,
    write_memory_fact,
    load_memory_facts,
    chunk_fact,
    chunk_body,
    chunk_compaction_summary,
)
from memory.compaction import (
    record_from_event,
    index_compaction_event,
    index_conversation_compactions,
)
from memory.recall import (
    BM25,
    Fact,
    HybridRecaller,
    QueryIntent,
    RecallResult,
    load_corpus,
    parse_frontmatter,
    rrf_fuse,
)
from memory.recall_api import (
    MemoryRecall,
    RecallResponse,
    RecalledFact,
    RerankDecision,
    rerank_decision,
    estimate_tokens,
    DEFAULT_TOKEN_BUDGET,
    DEFAULT_K,
)

__all__ = [
    "CpuEmbedder",
    "VecStore",
    "DurableFactStore",
    "MemoryFact",
    "Chunk",
    "ChunkHit",
    "CompactionSummaryRecord",
    "FACT_TYPES",
    "COMPACTION_SUMMARY_TYPE",
    "SOURCE_TYPES",
    "parse_memory_fact",
    "write_memory_fact",
    "load_memory_facts",
    "chunk_fact",
    "chunk_body",
    "chunk_compaction_summary",
    "record_from_event",
    "index_compaction_event",
    "index_conversation_compactions",
    "MODEL_NAME",
    "DIM",
    "BM25",
    "Fact",
    "HybridRecaller",
    "QueryIntent",
    "RecallResult",
    "load_corpus",
    "parse_frontmatter",
    "rrf_fuse",
    "MemoryRecall",
    "RecallResponse",
    "RecalledFact",
    "RerankDecision",
    "rerank_decision",
    "estimate_tokens",
    "DEFAULT_TOKEN_BUDGET",
    "DEFAULT_K",
]
