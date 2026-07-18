"""bundles.research_read — the ResearchReadBundle (d190 + d212 capability-domain redraw).

The CAPABILITY DOMAIN for READING an already-fetched source's verbatim text ON DEMAND
(the ``load_source`` tool). This is a small, SEPARATELY-LOADABLE capability (d212 #3): a
node that must GROUND a fact in a real source — a researcher building findings, a writer
citing a figure, a reviewer checking a claim — loads THIS bundle, composed with whatever
else it needs (``research`` to gather, ``file`` to write). It is NOT a role: it carries
only the read tool + the read-grounding doctrine.

It is the READ half that the OLD ResearchBundle and WriterBundle each baked separately
(both called ``make_load_source_tool``); pulling it into one capability domain lets the
writer reuse the exact same read capability the researcher uses, without inheriting the
gather tools (search/fetch/tree) it does not need.

It ORCHESTRATES the existing :func:`agent_runtime.source_tools.make_load_source_tool` —
it reimplements nothing (d190).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ..research_tree import make_tool_spec
from ..source_tools import _DEFAULT_LOAD_MAX_CHARS, _DEFAULT_SECTION_BUDGET
from .base import ObjectBundle

# The read-grounding doctrine: a TWO-TIER read on a COST HIERARCHY (d234/d235). This bundle owns
# the ON-DEMAND READ-BACK of an already-fetched source — it does NOT gather (that is the research
# bundle: search/fetch/note/expand/prune); here you only READ what has already been gathered.
#   1. read_notes (CHEAP) FIRST — the article-note gist (summary/key_claims/gaps) per [S#] source,
#      so you learn which source has what without pulling any verbatim text.
#   2. load_source (EXPENSIVE) ONLY for the exact figure/quote you will CITE word-for-word.
# Capability doctrine (how to operate the two read tools), not a role identity.
# CoT-autonomy P4: cost KNOWLEDGE, not a prescribed call sequence — the model
# reasons its own read strategy from the economics.
_READ_DOCTRINE = (
    "SOURCE READING (cost hierarchy) — never write a fact from memory of a source. Two "
    "read tiers exist: read_notes is the CHEAP leg — each already-fetched [S#] source's "
    "note gist (summary, key_claims, open gaps), which tells you WHICH source holds the "
    "figure or angle you need for almost nothing; load_source is the EXPENSIVE leg — it "
    "pulls ONE source's bounded verbatim text (sid='S3' for its lead, chunk='S3.c2' for "
    "a specific section), most useful for the exact figure, date or quotation you will "
    "cite word-for-word. Cite every figure to the real [S#] / URL it came from, and cite "
    "ONLY [S#] ids that appear in the SOURCE INDEX. load_source reports when the "
    "section's source budget is spent."
)

# Native schema for read_notes — the CHEAP first leg (handler built per-run via make_read_notes_tool).
_READ_NOTES_SPEC: dict[str, Any] = make_tool_spec(
    "read_notes",
    "CHEAP — the inexpensive first leg of source reading. Returns the article-note gist "
    "of the already-fetched sources: each [S#] source's summary, key_claims and open "
    "gaps, so you learn WHICH source has what WITHOUT pulling verbatim text. "
    "read_notes() for the index of every source, or read_notes(sid='S3') for one.",
    {"sid": {"type": "string"}},
    [],
)

# Native schema for load_source — the EXPENSIVE second leg (handler-backed ToolDef built per-run
# via load_source_tool / make_load_source_tool; this is the inspectable surface).
_LOAD_SOURCE_SPEC: dict[str, Any] = make_tool_spec(
    "load_source",
    "EXPENSIVE — the costly second leg of source reading (read_notes is the cheap "
    "index). Loads the verbatim text of ONE already-fetched source on demand, by its "
    "[S#] id (sid='S3' for its lead, or chunk='S3.c2' for a specific section) — most "
    "useful for an exact figure, date or quote you will cite word-for-word. Returns a "
    "bounded verbatim excerpt and reports when the section's source budget is spent. "
    "Cite only [S#] ids from the SOURCE INDEX.",
    {"sid": {"type": "string"}, "chunk": {"type": "string"},
     "max_chars": {"type": "integer"}},
    ["sid"],
)


class ResearchReadBundle(ObjectBundle):
    """Read-a-fetched-source capability: the on-demand, capped ``load_source`` tool."""

    name = "research_read"
    summary = (
        "READ an already-fetched source on a COST HIERARCHY to ground a fact or citation: "
        "read_notes (CHEAP — the article-note gist of which [S#] has what) first, then "
        "load_source (EXPENSIVE — one source's verbatim text) only for the exact figure/quote "
        "you will cite. Load this when you must quote/cite real source text you (or an upstream "
        "node) already fetched."
    )

    @property
    def own_doctrine(self) -> str:  # type: ignore[override]
        return _READ_DOCTRINE

    def tool_specs(self, ctx: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
        return super().tool_specs(ctx) + [dict(_READ_NOTES_SPEC), dict(_LOAD_SOURCE_SPEC)]

    # ------------------------------------------------------------------ #
    # handler-backed tool: the capped load-on-demand load_source ToolDef, bound to
    # THIS run's fetched sources. ORCHESTRATES source_tools.make_load_source_tool.
    # ``register`` adds it when ctx supplies the per-run ``sources``.
    # ------------------------------------------------------------------ #
    def load_source_tool(
        self,
        sources: Sequence[Mapping[str, Any]],
        *,
        section_budget: int = _DEFAULT_SECTION_BUDGET,
        per_call_cap: int = _DEFAULT_LOAD_MAX_CHARS,
    ):
        """The capped load-on-demand ``load_source`` ToolDef bound to ``sources`` —
        orchestrates :func:`agent_runtime.source_tools.make_load_source_tool`."""
        from ..source_tools import make_load_source_tool

        return make_load_source_tool(
            sources, section_budget=section_budget, per_call_cap=per_call_cap
        )

    def register(self, registry: Any, ctx: Optional[Mapping[str, Any]] = None) -> Any:
        """Add the read tools onto ``registry`` from ``ctx`` (else a no-op — the native
        schemas still advertise the tools):

        * ``read_notes`` (CHEAP) whenever ``ctx`` supplies the run's ``article_notes``; and
        * ``load_source`` (EXPENSIVE) whenever ``ctx`` supplies the run's ``sources``.

        read_notes is keyed to the GLOBAL ``[S#]`` via the ``sources`` list (so a note gist and a
        load_source pull name the same source); it still registers without ``sources`` (the notes
        then keep their own ids)."""
        from ..source_tools import make_read_notes_tool

        ctx = ctx or {}
        sources = ctx.get("sources")
        notes = ctx.get("article_notes") or ctx.get("notes")
        if notes:
            registry.add(make_read_notes_tool(notes, sources or ()))
        if not sources:
            return registry
        tool = self.load_source_tool(
            sources,
            section_budget=int(ctx.get("section_budget", _DEFAULT_SECTION_BUDGET)),
            per_call_cap=int(ctx.get("per_call_cap", _DEFAULT_LOAD_MAX_CHARS)),
        )
        registry.add(tool)
        return registry


__all__ = ["ResearchReadBundle"]
