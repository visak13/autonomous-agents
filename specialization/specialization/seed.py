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

# The canonical HTML shaping ruleset (s8/b5). The PARALLEL of the markdown writer
# for an HTML deliverable — it shapes the findings AS a complete HTML report, and
# (like every seed) carries NO "how to write HTML" how-to: the node already DID
# the task, this only tells it how to STRUCTURE the findings as one HTML document.
# It exists to fix the s8/b2 output-format divergence (an HTML request was routing
# to markdown-writer because the prior autonomous html-writer description was too
# generic to win the tie AND its body was a how-to-write-HTML guide that would have
# produced a tutorial, not a report). Promoting it to a canonical seed makes its
# selection-grade description authoritative and its body a real shaping ruleset.
#
# c11/d52 STYLING DEFAULT: the body also carries a reasoning-driven styling DEFAULT
# (readable typography, sensible spacing/margins, a STYLED table, source-citation
# styling). It is expressed as the QUALITIES the document should have — the LLM
# realizes them in its OWN inline `<style>` by reasoning — NOT a hard-coded CSS
# template, NOT a control flag, NOT app-side injection (honors d46/d14 behavior-via-
# prompting + d50 content-stays-RAW). The styling is a SENSIBLE DEFAULT that stays
# OVERRIDABLE + COMPOSABLE: when another spec (e.g. a user tone/house-style) is
# layered on the node it reconciles per the d47-req4 priority contract instead of
# being steamrolled by a fixed stylesheet.
HTML_WRITER_RULESET = (
    "You are an OUTPUT-SHAPING ruleset, not a task. Do the task described in the "
    "user message using the inputs and tool findings provided there, then shape "
    "your answer as ONE complete, self-contained HTML document:\n"
    "\n"
    "- Output a single valid HTML5 document and NOTHING else: begin at "
    "`<!DOCTYPE html>` and end at `</html>`. No Markdown, no code fences, no prose "
    "before or after the document.\n"
    "- Give it a `<head>` with `<meta charset=\"utf-8\">` and a `<title>` naming "
    "the subject.\n"
    "- Open the `<body>` with an `<h1>` naming the subject and a `<p>` lead "
    "summary of the key findings.\n"
    "- Use semantic structure: a `<section>` per topic, `<h2>`/`<h3>` headings in "
    "logical order, `<p>` for prose, `<ul>`/`<ol>`/`<li>` for lists, and `<table>` "
    "with `<th>`/`<td>` for tabular data.\n"
    "- Where the findings cite sources, end with a `<section>` whose heading is "
    "Sources, listing each as an `<a href=\"url\">title</a>` link.\n"
    "- Style the document for readability with ONE inline `<style>` block in the "
    "`<head>` (no external assets, scripts or images). Decide the actual CSS "
    "yourself by reasoning — do NOT paste a fixed boilerplate stylesheet — aiming "
    "for these QUALITIES:\n"
    "  - Typography: a clean, legible system/sans-serif font, a clear heading-to-"
    "body size scale, and comfortable line-height; constrain the body to a readable "
    "measure (e.g. a sensible max-width with auto side margins).\n"
    "  - Spacing: comfortable margins and padding so sections, headings, paragraphs "
    "and lists breathe — never a cramped wall of text.\n"
    "  - Tables: style every `<table>` so it reads as a real table — visible cell "
    "borders (collapsed), padded cells, a visually distinct header row, and "
    "zebra-striped alternating rows.\n"
    "  - Citations/Sources: style the Sources section and its links so citations "
    "are visually distinct and easy to scan (clear link styling; an offset or "
    "lighter-weight source list).\n"
    "- These styling choices are SENSIBLE DEFAULTS you apply by REASONING, not a "
    "rigid template. If another output-shaping ruleset is composed with this one "
    "(e.g. a tone, voice or house-style spec), RECONCILE and COMPOSE with it in the "
    "stated priority order: let it govern the axes it speaks to and apply these "
    "defaults only where it leaves the look unspecified. Never let this default "
    "styling steamroll or override another ruleset's intent.\n"
    "\n"
    "Shape ONLY the form. The content must be the real findings from the task — "
    "never produce a tutorial about HTML itself instead of the findings."
)

# The canonical RESEARCH/ANALYST shaping ruleset (s3/b2). It is the ONE
# specialization the bounded deep-research shape reuses across every round
# (research → critic → … → synthesis → verify), differentiated only by node role
# (d2/§2c). Like every seed it shapes the FORM of the answer to the real task —
# there is no "how to research" how-to anywhere; the node already did the work,
# this ruleset only tells it how to present grounded, well-supported findings.
#
# d107(1) — THE SEEDED DEEP-RESEARCH METHODOLOGY. The deep-research spec now also
# carries the INVESTIGATIVE MODEL the agent reasons over (DEEP_RESEARCH_SPEC points
# here): deep research = IDENTIFY the what/when/why/how the question needs → FIND
# answers by reading real sources → VERIFY each against those sources → STOP when
# the investigation is sufficiently answered AND verified. This is DISTINCT from
# question-answer (Q&A): Q&A returns one direct answer; an INVESTIGATION decomposes,
# gathers across angles, fact-checks, and decides for itself when it has gathered
# enough. The STOP CRITERION here is exactly what the research-planner agent reasons
# over to call ``stop_research`` (d95) — the spec, not a hard-coded flow, defines
# "enough". The depth/iteration ceiling is the SHAPE FILE's concern (d107(2)); this
# spec defines the methodology + the stop criteria, never a fabricated loop count.
RESEARCH_ANALYST_RULESET = (
    "You are an OUTPUT-SHAPING ruleset for grounded analytical work, not a task. "
    "Do the task in the user message (using its inputs, prior layers and tool "
    "findings), then shape your answer to these rules:\n"
    "- Lead with the substantive findings — no preamble, no restating the task.\n"
    "- Make every claim CONCRETE: facts, figures and named entities over "
    "generalities.\n"
    "- REPORT THE REAL FINDINGS BY READING THE SOURCES — never describe the search "
    "results or source list. The answer must be the actual facts, figures, "
    "headlines and quotations from the fetched article text; 'site X has an article "
    "about Y' is a description of sources, NOT a finding.\n"
    "- Keep claims TRACEABLE: attribute each to its source or prior layer; never "
    "invent a citation.\n"
    "- Distinguish well-supported from uncertain; flag gaps honestly.\n"
    "- Build on the prior layers — go DEEPER, do not restate them.\n"
    "- Be concise and non-redundant.\n"
    "\n"
    "DEEP-RESEARCH METHODOLOGY (this is an INVESTIGATION, not a question-answer). "
    "Drive the work as an investigative loop, not a single direct answer:\n"
    "1. IDENTIFY — decompose the question into what it actually needs: the WHAT, "
    "WHEN, WHY and HOW (the entities, the timeline, the causes, the mechanism). A "
    "deep-research request is an investigation to be opened up, not a fact to look "
    "up.\n"
    "2. FIND — look for the answers by searching and READING real sources, "
    "RELIABLE SOURCES FIRST: prefer primary, official and reputable reporting; "
    "gather across the distinct angles the question opened, not just the first hit. "
    "AVOID social-media posts, opinion pieces and vague global-outlook filler. As "
    "you fill the blanks, expand into the dimensions the question implies — timeline, "
    "key events, costs/figures, causes and impact.\n"
    "3. VERIFY — check each finding against the sources, corroborating critical "
    "facts across at least two independent reliable sources; reconcile "
    "contradictions, and drop anything you cannot support. Never fill a gap by "
    "inventing a fact or a citation.\n"
    "4. STOP — decide for yourself when the investigation is SUFFICIENTLY ANSWERED "
    "AND VERIFIED: every sub-question the question needed is covered by verified "
    "findings with no meaning-adding gap left. That sufficiency judgement is the "
    "criterion you reason over to STOP researching (stop_research) — stop then, do "
    "not pad with more rounds; keep investigating while a needed, unanswered or "
    "unverified angle remains (up to the shape's layer ceiling).\n"
    "This investigative model — identify → find → verify → stop-when-sufficient — is "
    "what separates deep research from a shallow Q&A answer.\n"
    "Shape ONLY the form and rigor; the content must be the real findings, never a "
    "tutorial about how to do research."
)

# The seed registry: name -> (description, ruleset body). The description is the
# planner-facing lookup text (body-free index, d10); the body is the shaping
# ruleset a sub-agent loads.
CANONICAL_RULESETS: dict[str, tuple[str, str]] = {
    "markdown-writer": (
        "Format the final deliverable as a clean GitHub-Flavored Markdown "
        "document — title heading, lead summary, sectioned headings, bullet/"
        "numbered lists and a Sources section. Bind to the node that PRODUCES a "
        "written report/document when the user wants structured, readable "
        "Markdown output (a .md document) — NOT HTML.",
        MARKDOWN_WRITER_RULESET,
    ),
    "html-writer": (
        "Format the final deliverable as ONE self-contained, semantic HTML5 "
        "document — `<h1>` title, lead summary, `<section>` topics with ordered "
        "`<h2>`/`<h3>` headings, lists, tables and a Sources section of links. "
        "Bind to the node that PRODUCES the final written report/document when the "
        "user wants HTML / a web page / a .html file — NOT Markdown.",
        HTML_WRITER_RULESET,
    ),
    "research-analyst": (
        "Shape the output as rigorous grounded analysis — concrete facts and "
        "figures READ from the sources (never a description of the search "
        "results), every claim traceable to its source, gaps and uncertainty "
        "flagged honestly. Bind to research/analysis nodes that must report real "
        "findings; the specialization the deep-research shape reuses each round.",
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
    "HTML_WRITER_RULESET",
    "RESEARCH_ANALYST_RULESET",
    "DEEP_RESEARCH_SPEC",
    "CANONICAL_RULESETS",
    "make_ruleset_spec",
    "seed_ruleset_spec",
    "seed_canonical_rulesets",
]
