"""Programmatic SEED path for output-shaping ruleset specs (d1).

This is the direct way to stand a known **output-shaping ruleset** spec into the
registry WITHOUT the research -> HITL-compile lifecycle. It exists because of the
d1 redefinition: a specialization's compiled ``body`` is a *ruleset that shapes
the OUTPUT of a real task*, never a "how to <skill>" document. The research path
(:mod:`specialization.research` + :mod:`specialization.compiler`) is still here
and untouched, but it is NO LONGER the *definition* of a spec — it is one way to
author a body, and (as round-1 showed) a way that, pointed at a skill name,
yields a skill how-to. The interactive chat-authoring surface that lets a user
write such a ruleset back-and-forth is the LATER s4 step; for the POC (and for
the a3 live run) we seed the canonical rulesets directly here.

What a SEED produces
--------------------
:func:`seed_ruleset_spec` mints a :class:`~specialization.model.CompiledSpec`
whose ``body`` is the given ruleset text verbatim, tags it ``source="seed"``
(the honest origin — bypassed research + the compile gate), and registers it.
The body is the sub-agent's WHOLE grounding for SHAPING; the runtime composes it
as a shaping layer OVER the real task content at produce time (see
``agent_runtime.runtime.SubAgent``).

:data:`MARKDOWN_WRITER_RULESET` is the canonical markdown shaping ruleset — the
exact one round-1 got wrong (it researched "how to write markdown" instead of
shaping findings AS markdown). Here it is a pure shaping instruction set, with no
"how to" framing anywhere, so the negative check (system is not a skill-how-to)
holds by construction.

Dependency-free, in-process (d2/d10) — just the model + registry.
"""
from __future__ import annotations

from specialization.model import CompiledSpec
from specialization.registry import SpecRegistry

# The origin tag for a directly-seeded ruleset (model.SOURCES includes it).
SOURCE_SEED = "seed"


# --------------------------------------------------------------------------- #
# Canonical output-shaping rulesets (the POC seeds)
# --------------------------------------------------------------------------- #
# NOTE (d1, the load-bearing distinction): every line below shapes the FORM of
# the answer to whatever task the agent was given. There is deliberately NO
# "how to write markdown" content — the agent already DID the task (e.g. live
# research); this ruleset only tells it how to STRUCTURE the findings it has.
MARKDOWN_WRITER_RULESET = (
    "You are an OUTPUT-SHAPING ruleset, not a task. Do the task described in the "
    "user message using the inputs and tool findings provided there, then shape "
    "your answer to follow these rules:\n"
    "\n"
    "- Open with a single level-1 heading (`# `) naming the subject of the task.\n"
    "- Add a one-paragraph **Summary** of the key findings first.\n"
    "- Group the substantive findings under level-2 headings (`## `).\n"
    "- Use bullet lists (`- `) for discrete facts and points; use a numbered list "
    "only for ordered steps or rankings.\n"
    "- **Bold** the most important terms; use `inline code` for literal names, "
    "values, commands or identifiers.\n"
    "- Where the findings cite sources, list them under a final `## Sources` "
    "heading as `- [title](url)` links.\n"
    "- Use valid GitHub-Flavored Markdown throughout. Keep it tight — no preamble, "
    "no restating the task, no meta-commentary about markdown itself.\n"
    "\n"
    "Shape ONLY the form. The content must be the real findings from the task — "
    "never produce a tutorial about markdown itself instead of the findings."
)

# The canonical RESEARCH/ANALYST shaping ruleset (s3/b2). It is the ONE
# specialization the bounded deep-research shape reuses across every round
# (research → critic → … → synthesis → verify), differentiated only by node role
# (d2/§2c). Like every seed it shapes the FORM of the answer to the real task —
# there is no "how to research" how-to anywhere; the node already did the work,
# this ruleset only tells it how to present grounded, well-supported findings.
RESEARCH_ANALYST_RULESET = (
    "You are an OUTPUT-SHAPING ruleset for grounded analytical work, not a task. "
    "Do the task described in the user message using the inputs, prior layers and "
    "tool findings provided there, then shape your answer to follow these rules:\n"
    "\n"
    "- Lead with the substantive answer/findings — no preamble, no restating the "
    "task.\n"
    "- Make every claim CONCRETE and SPECIFIC; prefer facts, figures and named "
    "entities over generalities.\n"
    "- REPORT THE REAL FINDINGS BY READING THE SOURCES — never describe the search "
    "results or the source list. A search only names candidate pages; the answer "
    "must be the actual facts, figures, headlines and quotations taken from the "
    "fetched article text you were given. Writing 'site X has an article about Y' "
    "or 'Reuters covers Z' is a description of the sources, NOT a finding.\n"
    "- Keep claims TRACEABLE: attribute each finding to the source or prior layer "
    "it came from; never invent a citation.\n"
    "- Distinguish what is well-supported from what is uncertain or still open; "
    "flag gaps honestly rather than papering over them.\n"
    "- Build on the prior researched layers shown to you — go DEEPER, do not "
    "merely restate earlier layers.\n"
    "- Be concise and non-redundant; one well-supported point beats three vague "
    "ones.\n"
    "\n"
    "Shape ONLY the form and rigor. The content must be the real findings from the "
    "task — never produce a tutorial about how to do research instead of the "
    "findings themselves."
)

# The seed registry: name -> (description, ruleset body). The description is the
# planner-facing lookup text (body-free index, d10); the body is the shaping
# ruleset a sub-agent loads.
CANONICAL_RULESETS: dict[str, tuple[str, str]] = {
    "markdown-writer": (
        "Shape findings into a clean, well-structured GitHub-Flavored Markdown "
        "report (headings, lists, summary, sources).",
        MARKDOWN_WRITER_RULESET,
    ),
    "research-analyst": (
        "Shape work into concise, grounded, well-supported analytical findings "
        "with traceable claims and honest gaps (the deep-research shape's reused "
        "specialization).",
        RESEARCH_ANALYST_RULESET,
    ),
}

# The canonical specialization the deep-research shape reuses across all rounds
# (s3/b2). Exposed so the live chat route can pick it as the ONE spec without
# hard-coding the name at the call site.
DEEP_RESEARCH_SPEC = "research-analyst"


# --------------------------------------------------------------------------- #
# The seed API
# --------------------------------------------------------------------------- #
def make_ruleset_spec(name: str, description: str, ruleset: str) -> CompiledSpec:
    """Build (but do NOT register) a ``source="seed"`` output-shaping CompiledSpec.

    The ``ruleset`` text becomes the spec ``body`` verbatim — it is the WHOLE
    shaping grounding a sub-agent loads (d1/d10). No research, no compile gate;
    the origin is recorded as ``seed`` so provenance stays honest."""
    if not (ruleset or "").strip():
        raise ValueError(f"ruleset body for {name!r} must be non-empty")
    return CompiledSpec(
        name=name,
        description=description,
        source=SOURCE_SEED,
        body=ruleset.strip(),
        research_trace_ref="",  # seeded directly: there is no research trace
    )


def seed_ruleset_spec(
    registry: SpecRegistry, name: str, description: str, ruleset: str
) -> CompiledSpec:
    """Mint an output-shaping ruleset spec and REGISTER it. Returns the spec.

    The direct programmatic path (d1): stand a known shaping ruleset into the
    registry so the planner can mark a node with it and a sub-agent can load it —
    without the research/HITL lifecycle. Overwrites an existing spec of the same
    name (re-seed)."""
    spec = make_ruleset_spec(name, description, ruleset)
    registry.register(spec)
    return spec


def seed_canonical_rulesets(registry: SpecRegistry) -> list[CompiledSpec]:
    """Seed every :data:`CANONICAL_RULESETS` entry (e.g. ``markdown-writer``).

    A one-call bootstrap for the POC / the a3 live run so the known shaping
    rulesets exist in the registry. Returns the seeded specs."""
    return [
        seed_ruleset_spec(registry, name, description, ruleset)
        for name, (description, ruleset) in CANONICAL_RULESETS.items()
    ]


__all__ = [
    "SOURCE_SEED",
    "MARKDOWN_WRITER_RULESET",
    "RESEARCH_ANALYST_RULESET",
    "DEEP_RESEARCH_SPEC",
    "CANONICAL_RULESETS",
    "make_ruleset_spec",
    "seed_ruleset_spec",
    "seed_canonical_rulesets",
]
