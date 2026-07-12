"""CLEAN-SLATE registry curation — the EXPOSED shapes/specs scope (d230 / d206).

The product is the SHAPE + SPEC + TOOL + ROLE definition layer; behaviour comes
only from those definitions, never a flag (d227). One real lever for E4B's choice
RELIABILITY is the SIZE of the choice space: a flat, fully-exposed registry of
every shape/spec lets a small model reasonably pick an off-target option (the a3
divert — the planner picked ``concurrent-multi-topic-gathering`` because it was
FREE to, d228).

d230 (user, "we need pure runs now") scopes the EXPOSED registry to exactly the
current acceptance cases (d206) and DEFERS the heavier vector-similarity
``get_shapes`` retrieval optimisation to a later registry-GROWTH phase. So this is
an HONEST required-now curation, NOT hide-to-force (d227/G3): the user explicitly
declared the d206 cases the genuine required-now set; removing the noise shapes is
scoping, not removing an alternative so deep-research "wins". The deferred
shapes/specs are NOT deleted — they remain on disk and in the raw loaders
(``load_shapes`` / ``SpecRegistry``); only the planner-facing ADVERTISEMENT
(``get_shapes`` / ``get_specs`` / the shape selector / the planner factory's
lookup) is narrowed to the curated set. Growing back out = widen these allow-lists.

CURATED SHAPES (cover every d206 case):
  * ``deep-research`` — the growable, exhaustive single-topic investigation:
    US-Iran report, pirate-history multi-page SPA, the US-Iran follow-up.
  * ``linear``        — the straight sequential path: haiku quick-exit, Java
    hello-world.
  * ``codebase-summary`` — read a LOCAL codebase/directory and write a summary
    (s16/aflex — the d239/d241 non-web generic-spine flex probe).

CURATED SPECS (the d206 output-shaping rulesets):
  * ``html-writer``      — HTML report / multi-page SPA deliverable.
  * ``section-html-writer`` — the SECTIONED-HTML variant for a complex/multi-section/
    data-heavy report built across turns (s16/ashw, d246); the planner selects it over
    ``html-writer`` by data complexity.
  * ``research-analyst`` — grounded findings on every research/gather node.
  * ``research-methodology`` — the DOMAIN-AGNOSTIC research METHODOLOGY (decompose →
    self-select-gather → note[claim+source+gap] → cross-verify → deepen → prune → stop →
    write) for a NON-web gather node (s16/SA-6, d258); the generic CORE the web/codebase/
    vectordb research variants share.
  * ``web-research``     — the LIVE-WEB variant of ``research-methodology``: the same method
    paired with the web gather bundle; the planner selects it over the CORE for a web
    investigation (s16/SA-6, d258).
  * ``pirate-tone``      — the pirate-voice house style (composed on write nodes).
  * ``claude-skill``     — the "research → a Claude skill in an MD file" deliverable.
  * ``codebase-summary`` — the Markdown summary of a local codebase (s16/aflex flex probe).
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

# The exposed allow-lists (d230 clean slate). Widen these to GROW the registry back
# out once the runs are stable (the deferred vector-retrieval tool is that phase).
CURATED_SHAPES: tuple[str, ...] = (
    "deep-research", "linear", "codebase-summary", "schedule-leg",
)
CURATED_SPECS: tuple[str, ...] = (
    "html-writer",
    "section-html-writer",
    "research-analyst",
    "research-methodology",
    "web-research",
    "pirate-tone",
    "claude-skill",
    "codebase-summary",
)


def curate_names(names: Iterable[str], allow: Sequence[str] = CURATED_SPECS) -> list[str]:
    """Keep only the names in ``allow`` (order-preserving) — the PLANNER-facing
    spec-name filter (the plan-schema enum / the authorer's spec_names)."""
    allowed = set(allow)
    return [n for n in names if n in allowed]


def curate_index(rows: Iterable[Any], allow: Sequence[str] = CURATED_SPECS) -> list[Any]:
    """Keep only the index rows whose ``name`` is in ``allow`` — the PLANNER-facing
    spec LOOKUP filter (the AbstractPlanFactory's body-free specialization index).

    Accepts ``SpecIndexEntry`` objects or ``{name, …}`` dicts. CURATION IS
    ADVERTISEMENT-ONLY (d230): apply this ONLY where the planner/selector reasons
    over the catalog — NEVER to the spec-management listing (``/spec-chats/
    registered``), the missing-specialist membership check, or body loading, which
    must see EVERY registered spec."""
    allowed = set(allow)

    def _name(r: Any) -> str:
        return str(r.get("name", "") if isinstance(r, dict) else getattr(r, "name", ""))

    return [r for r in rows if _name(r) in allowed]
