"""Specialization engine for the reactive agent.

A specialization is DEFINED in the UI (or proposed autonomously), RESEARCHED,
and COMPILED on user approval (d8) into a markdown-with-frontmatter doc that a
launched sub-agent loads as its whole grounding. The registry enforces the d10
context-scoping split: the planner sees only a body-free :meth:`SpecRegistry.index`
lookup; a sub-agent loads exactly one full :meth:`SpecRegistry.load` body.

Workspace member (d11) resolving into the single shared root .venv so it runs in
the SAME interpreter as the rest of the app (d2 — one in-process process).
"""
from specialization.model import (
    DELIVERY_CHANNELS,
    INDEX_FIELDS,
    SCHEDULE_KINDS,
    SOURCES,
    CompiledSpec,
    DeliverySpec,
    RawDefinition,
    ScheduleSpec,
    SpecIndexEntry,
    parse_compiled_spec,
    parse_frontmatter_only,
    utc_now_iso,
)
from specialization.registry import SpecRegistry
from specialization.research import (
    HowNote,
    ResearchTrace,
    SourceRef,
    build_research_hook,
    derive_queries,
    persist_trace,
    research,
    research_and_persist,
)
from specialization import compiler
from specialization.engine import (
    ApprovalDenied,
    ApprovalRequired,
    ApprovalToken,
    Approver,
    SOURCE_AUTONOMOUS,
    SOURCE_UI,
    SpecDraft,
    SpecializationEngine,
)
from specialization.loader import SpecLoader
from specialization.conversation import (
    ConversationError,
    DraftPreview,
    SpecConversation,
    Turn,
)

__all__ = [
    "RawDefinition",
    "CompiledSpec",
    "ScheduleSpec",
    "DeliverySpec",
    "SpecIndexEntry",
    "SpecRegistry",
    "SOURCES",
    "SCHEDULE_KINDS",
    "DELIVERY_CHANNELS",
    "INDEX_FIELDS",
    "parse_compiled_spec",
    "parse_frontmatter_only",
    "utc_now_iso",
    # web-research path (a2)
    "research",
    "research_and_persist",
    "persist_trace",
    "derive_queries",
    "build_research_hook",
    "ResearchTrace",
    "SourceRef",
    "HowNote",
    # compiler + engine + loader (a3)
    "compiler",
    "SpecializationEngine",
    "SpecDraft",
    "ApprovalToken",
    "Approver",
    "ApprovalRequired",
    "ApprovalDenied",
    "SOURCE_UI",
    "SOURCE_AUTONOMOUS",
    "SpecLoader",
    # conversational spec-authoring core (s4/a1)
    "SpecConversation",
    "DraftPreview",
    "Turn",
    "ConversationError",
]
