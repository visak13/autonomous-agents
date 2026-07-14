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

# The SHARED output-agnostic COHERENCE + GROUNDING doctrine (RP-2, d319/d311/d326). RP-1 removed
# the engine's write-path authoring/fixing/composing; the MODEL now authors the whole artifact
# DIRECTLY (generic write tools), so the 7 coherence + grounding behaviours the engine used to
# impose live HERE, in the writer SPEC. Neuron d326 ruled OPTION A: EACH per-format writer spec
# carries these 7 points in ITS OWN format idiom — the model self-selects the format spec and the
# ENGINE pins NO format. To stay DRY without a second runtime mechanism (``CompiledSpec`` has no
# ``extends``; a node composes whole spec bodies), the doctrine is authored ONCE here and COMPOSED
# into every WRITER ruleset below — the same Python-level composition seed.py already uses for
# ``WEB_RESEARCH_RULESET``. It is OUTPUT-AGNOSTIC (it PINS no format — it names web / Markdown /
# code only as PARALLEL example idioms) and SOURCE-AGNOSTIC (a "source" is whatever grounded the
# task — a fetched URL, a file path read), so each writer applies it in its own idiom. GATHER
# specs (research-analyst / research-methodology / web-research) do NOT carry it — they never
# author the deliverable.
_COHERENT_ARTIFACT_DOCTRINE = (
    "\n\n"
    "AUTHOR A COHERENT, SELF-CONTAINED ARTIFACT — you own its structure and correctness; nothing "
    "downstream fixes, reformats, reorders or completes it for you. Uphold ALL of the following, "
    "each in the idiom your chosen output format provides:\n"
    "- OWN NAVIGATION FROM YOUR OWN HEADINGS: when the artifact is navigable, author its own "
    "table-of-contents / navigation DIRECTLY from the section headings you write (in-page anchor "
    "links for a web document, a heading-link table-of-contents for a Markdown document, an "
    "ordered structure for code); every navigation entry must point to a real section you "
    "actually author.\n"
    "- EMIT EXACTLY ONE WELL-FORMED, SELF-CONTAINED DOCUMENT: exactly ONE coherent artifact — a "
    "single root/title, every structural element you open properly closed and balanced, nothing "
    "left after the artifact ends, NO duplicate top-level structure and NO repeated section "
    "families. Author the whole thing in place; never restart, re-open or re-emit a document "
    "shell you have already written.\n"
    "- UNIQUE IDENTIFIERS, RESOLVING LINKS: give each section a UNIQUE identifier/anchor; every "
    "internal link or navigation entry RESOLVES to a real section; leave no dangling or duplicate "
    "reference.\n"
    "- NEVER EMPTY, NEVER A STUB: author every section IN FULL or omit it — never leave an empty "
    "section, a 'content to be added' placeholder or a stub. The lead / introduction is a REAL "
    "grounded synthesis of the key findings, never a placeholder.\n"
    "- GROUND EVERY SECTION, CITE ONLY REAL SOURCES: assign your own sources to each section and "
    "cite ONLY real sources you actually gathered or read for THIS task (a URL you fetched and "
    "opened, a file path you read). NEVER invent, guess or fabricate a source, and never build a "
    "citation from a source LABEL (an '[S3]'-style token is a label, NOT a source URL); never "
    "leave a worded citation placeholder ('[Source URL]', 'TBD', 'Source N'). If a claim or a "
    "table row cannot be grounded in a real source, DROP it rather than fabricate one — this "
    "no-ungrounded-source guarantee is YOURS to uphold.\n"
    "- FINISH YOUR SENTENCES: end every sentence, and the artifact as a whole, cleanly — never "
    "stop on a mid-sentence fragment or a truncated tail.\n"
    "- IMAGES ONLY FROM REAL GATHERED RECORDS: when the artifact embeds images, use ONLY image "
    "URLs actually returned by an image search/gather for THIS task — the record's image_url "
    "VERBATIM (attribute its source_url where the format supports it). NEVER invent, guess or "
    "placeholder an image path (no 'placeholder_map.jpg'); if no suitable image record was "
    "gathered, ship the artifact WITHOUT images rather than fabricate one.\n"
    "- SELF-REVIEW BEFORE YOU FINISH: before you emit your final token, RE-READ the whole artifact "
    "you just wrote and CHECK it against every point above, then CORRECT what you find — this review "
    "is the last, non-optional part of authoring, NOT a separate step anyone downstream runs for you. "
    "Concretely: any claim, figure or table row you cannot trace to a real source you actually "
    "gathered or read → GROUND it in that source or DROP it (never let an unbacked claim stand); any "
    "duplicate top-level structure, restarted document shell or repeated section family → remove the "
    "duplicate so exactly ONE coherent artifact remains; any navigation entry, anchor or internal "
    "link that does not resolve to a real section → fix or remove it; any empty section, stub or "
    "worded placeholder → author it in FULL or omit it; any mid-sentence or truncated tail → finish "
    "it cleanly. Only finish once the artifact passes your OWN review on all of the above.\n"
    "- YOU AUTHOR THE CONTENT, THE PLANNER CHOSE THE TOPOLOGY: the planner decided WHICH sections "
    "the artifact needs and how the work is split; YOU author each section's real content and the "
    "artifact's overall coherence per the above. The engine composes, fixes, reorders or "
    "reformats NOTHING — the coherent, self-contained artifact is entirely your own authorship."
)

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
    "- Emit ONE COMPLETE, WELL-FORMED document: every tag you open you also close, "
    "in order; it is VALID HTML5 that renders on its own with no external assets; "
    "write the whole report in this single pass — never stop mid-section, and never "
    "leave a trailing fragment, placeholder or note after the closing tag.\n"
    "\n"
    "Shape ONLY the form. The content must be the real findings from the task — "
    "never produce a tutorial about HTML itself instead of the findings."
)

# The canonical SECTIONED-HTML shaping ruleset (s16/ashw d246; RE-ARCHITECTED s16/SF-2
# d310/d311/d312). A VARIANT of html-writer HARDENED for a COMPLEX / multi-section /
# data-heavy report. It is a SEPARATE spec (d246 SEPARATION OF CONCERNS): base
# html-writer stays the one-pass writer; this carries the extra COHERENCE qualities a
# large sectioned document needs. The planner OPTS IN by SELECTING it (over
# html-writer) when the research is data-rich — sectioning is an emergent SPEC CHOICE,
# never a forced engine procedure (d211/d218/d246).
#
# RP-1 (d319/d311) — the SF-2 engine-compose SKELETON-THEN-FILL ruleset body and its
# ``SECTION_HTML_WRITER_SCHEMA`` structured-emission schema were RETIRED. That contract
# (author a {skeleton, sections} JSON object → a dumb engine k/v substitution composes the
# document) presumed an engine COMPOSE step; RP-1 made the write path OUTPUT-AGNOSTIC (the
# LLM authors its artifact DIRECTLY via generic write/edit tools; the engine composes/fixes
# NOTHING). RP-2 (d326, OPTION A) authors the REAL body here: section-html-writer is KEPT as
# the HARDENED long/multi-section HTML variant (a PEER format-spec the model self-selects over
# the one-pass ``html-writer`` when the report is large / data-heavy). Its body is HTML-format
# shaping ONLY — the 7-point coherence + grounding doctrine is COMPOSED in below from the shared
# ``_COHERENT_ARTIFACT_DOCTRINE`` (the model authors the document directly; the engine authors,
# decides, fixes or composes NOTHING). It differs from ``html-writer`` only by hardening for a
# long, many-section, table-heavy document that is prone to incoherence.
SECTION_HTML_WRITER_RULESET = (
    "You are an OUTPUT-SHAPING ruleset for a LARGE, MULTI-SECTION HTML report, not a task. "
    "Do the task described in the user message using its inputs and tool findings, then author "
    "the deliverable DIRECTLY with your file-authoring tools as ONE complete, self-contained "
    "HTML document:\n"
    "\n"
    "- Output a single valid HTML5 document and NOTHING else: begin at `<!DOCTYPE html>` and "
    "end at `</html>`. No Markdown, no code fences, no prose before or after the document.\n"
    "- Give it a `<head>` with `<meta charset=\"utf-8\">`, a `<title>` naming the subject, and "
    "ONE inline `<style>` block (no external assets, scripts or images).\n"
    "- Open the `<body>` with an `<h1>` naming the subject and a lead: a `<p>` (or short intro "
    "`<section>`) synthesising the key findings, plus — because this is a LONG, multi-section "
    "report — an in-page NAVIGATION list (`<nav>`) of links to each section you author.\n"
    "- Structure the body as MANY `<section>` blocks, each with a UNIQUE `id`, an `<h2>`/`<h3>` "
    "heading in logical order, and its own content: `<p>` prose, `<ul>`/`<ol>` lists, and "
    "`<table>` with `<th>`/`<td>` for the tabular / timeline / figures data a data-heavy report "
    "carries. Each `<nav>` link's target must match a real section `id` (resolving anchors).\n"
    "- Where the findings cite sources, end with ONE `<section>` headed Sources, listing each "
    "cited source exactly once as an `<a href=\"url\">title</a>` link.\n"
    "- Style the document for readability by REASONING out the CSS in that one `<style>` block "
    "(do NOT paste a fixed boilerplate): legible typography and a clear heading scale; a readable "
    "body measure with comfortable spacing; every `<table>` styled as a real table (collapsed "
    "visible cell borders, padded cells, a distinct header row, zebra-striped rows); and a "
    "distinct, easy-to-scan Sources/citation style. These are SENSIBLE DEFAULTS you apply by "
    "reasoning; if another output-shaping ruleset (a tone / house-style spec) is composed with "
    "this one, RECONCILE and COMPOSE in the stated priority order rather than steamrolling it.\n"
    "- A long report is where premature closes, sections after `</html>`, repeated nav / Sources "
    "blocks, duplicated sections and truncated tails creep in: author the WHOLE document in one "
    "coherent pass, close every tag you open, and let nothing follow `</html>`.\n"
    "\n"
    "Shape ONLY the form. The content must be the real findings from the task — never produce a "
    "tutorial about HTML itself instead of the findings."
)

# The canonical RESEARCH/ANALYST shaping ruleset (s3/b2). It is the ONE
# specialization the bounded deep-research shape reuses across every round
# (research → critic → … → synthesis → verify), differentiated only by node role
# (d2/§2c). Like every seed it shapes the FORM of the answer to the real task —
# there is no "how to research" how-to anywhere; the node already did the work,
# this ruleset only tells it how to present grounded, well-supported findings.
#
# d235 — SPEC-VS-ROLE DE-BLUR. The earlier d107(1) IDENTIFY/FIND/VERIFY/STOP
# investigative METHODOLOGY was DROPPED from this spec: how to drive the investigation
# (decompose, search, read, verify, when to stop_research) is the researcher ROLE's
# behaviour and the RUNTIME's research loop (the research bundle doctrine + the shape's
# layer ceiling), NOT an output-shaping ruleset. A spec that also dictated the task
# method blurred spec (HOW it presents) with role/runtime (HOW it works). This spec now
# carries ONLY the OUTPUT-QUALITY STANDARD — concrete, grounded, traceable, deeper.
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
    "Shape ONLY the form and rigor of the answer — concrete, grounded, traceable, "
    "deeper. HOW to drive the investigation (decompose, search, read, verify, when to "
    "stop) is the node's role and the runtime's job, not this spec; do not turn the "
    "answer into a tutorial about how to do research."
)

# The canonical RESEARCH-METHODOLOGY CORE spec (s16/SA-6, d256/d258). UNLIKE every other
# seed above (which are OUTPUT-SHAPING rulesets — they govern the FORM of the deliverable),
# this is a METHODOLOGY ruleset: it governs HOW a gather/research node DRIVES its
# investigation. It is the DEFINITION-LAYER home of the research method now that the
# researcher ROLE is being retired (d255/SA-7) — the investigative steering that used to be
# keyed to ROLE_RESEARCHER moves to a SELECTABLE SPEC (d240: the spec text is the lever).
#
# DOMAIN-AGNOSTIC BY CONSTRUCTION (d258): it NEVER names "web" — it describes the GENERIC
# craft (self-select your gather bundle first → decompose → gather + read → note as
# claim+source+gap → cross-verify → deepen on gaps → prune bad leads → question completeness
# → stop → write from notes), so a codebase-research / vectordb-research sibling later differs
# ONLY by its paired gather bundle, not by this method. The named ``web-research`` VARIANT
# below is this CORE plus a thin web-pairing note; the planner picks between them by their
# selection-lever DESCRIPTIONS in CANONICAL_RULESETS (mirroring html-writer ↔ section-html-
# writer). The reasoned doctrine (question-completeness, cross-verify across sources, expand
# the open/curiosity areas, prune bad leads) sharpens the bundle doctrine's completeness-stop +
# cross_verify + expand/prune (d191) — it is REASONING the agent applies over its self-selected
# bundle, NEVER a flag or a forced procedure (d211/d227/d240).
RESEARCH_METHODOLOGY_RULESET = (
    "You are a RESEARCH-METHODOLOGY ruleset — it shapes HOW you INVESTIGATE the task in "
    "the user message, not the format of the deliverable (a separate output-shaping spec "
    "governs that). Do the real task using its inputs and your tools. These are the reasoned "
    "QUALITIES of a thorough investigation, not a rigid script to recite:\n"
    "\n"
    "SELF-SELECT YOUR TOOLS FIRST. Before you gather, load the gather capability this task "
    "needs: FIRST call get_bundles to see the available capability bundles, THEN load the one "
    "that fits this investigation's domain — so your search / read / note tools are in hand "
    "before you start rather than improvised mid-run. (Good practice, not a hoop: a research "
    "task without its gather bundle has nothing to gather with.)\n"
    "\n"
    "DECOMPOSE FIRST, THEN GROW OUT. Break the goal into the distinct concerns it names (the "
    "what / why / when / how it must answer) BEFORE you drill any single one — breadth before "
    "depth. Cover every concern the goal raises before going deep on one of them.\n"
    "\n"
    "GATHER, THEN NOTE AS CLAIM + SOURCE + GAP. For each concern, use your self-selected tools "
    "to find and actually READ real sources, then record a structured NOTE: what you learned "
    "(the claim), WHERE you read it (the source), and — crucially — the GAP it left (a figure "
    "left unverified, an open question, the angle to chase next). The note, not loose prose, is "
    "what carries learning forward; its gaps direct the next round.\n"
    "\n"
    "CROSS-VERIFY ACROSS SOURCES. Before you rely on a claim, check it against the sources you "
    "have ACTUALLY pulled: attribute every fact to a real source you read; drop or qualify any "
    "claim no source backs; never cite something you only saw in a result list but never read. "
    "Where concern areas overlap, confirm a claim across more than one source rather than "
    "trusting a single one.\n"
    "\n"
    "DEEPEN ON GAPS, PRUNE BAD LEADS. EXPAND a concern that still has a missing meaning — an "
    "unanswered gap, or a curiosity/open area the goal invites — into another gathered round, "
    "REASONING over what your self-selected tools can still surface (do not guess the answer). "
    "PRUNE a lead that is adding no new meaning so it does not bloat the work.\n"
    "\n"
    "QUESTION COMPLETENESS, THEN STOP. Before you finish, ask yourself plainly: is this "
    "ACTUALLY complete? — is every concern the goal named settled in a note or honestly "
    "collapsed, with no open gap left unchased and no obvious area unexplored? Stop only when "
    "the answer is yes; do not stop early on a thin pass, and do not keep gathering once every "
    "concern is genuinely settled.\n"
    "\n"
    "WRITE FROM YOUR NOTES. Build the answer from the notes and sources you gathered — real "
    "facts, figures and attributions — never from memory or guesses; flag what stays uncertain "
    "honestly.\n"
    "\n"
    "This shapes HOW you investigate — the method, not the deliverable's format. The facts must "
    "come from the real sources you gather for THIS task; never turn the answer into a tutorial "
    "about how to do research."
)

# The named WEB-RESEARCH VARIANT (s16/SA-6, d258). It IS the generic CORE method above plus a
# thin WEB-PAIRING note — siblings differ ONLY by their paired gather bundle (d258), so the
# variant adds no second methodology, just names the web bundle as the domain it gathers from.
# Its selection-lever DESCRIPTION (in CANONICAL_RULESETS) is what the planner picks it by for a
# live-web investigation; the plain CORE stays selectable for a non-web (codebase / vector-store
# / file) brief.
_WEB_RESEARCH_PAIRING = (
    "\n\nFOR THIS VARIANT the domain is the LIVE WEB: the gather bundle you self-select is the "
    "web research bundle — web_search to find candidate sources, web_fetch to read them. Search "
    "focused questions, fetch and READ the real result URLs (copy a URL verbatim from the "
    "results; NEVER invent, guess or placeholder one), and note + cross-verify against the pages "
    "you actually read."
)
WEB_RESEARCH_RULESET = RESEARCH_METHODOLOGY_RULESET + _WEB_RESEARCH_PAIRING

# The canonical CLAUDE-SKILL shaping ruleset (d206 test 5 / d230 curated set). The
# output-style spec for the "research → provide a Claude skill in an MD file"
# deliverable: it shapes the findings AS a well-formed Claude Agent Skill markdown
# file (YAML frontmatter + an instruction body), carrying NO "how to write a skill"
# how-to beyond the structure — the node already did the task, this only tells it how
# to STRUCTURE the deliverable as a skill file.
CLAUDE_SKILL_RULESET = (
    "You are an OUTPUT-SHAPING ruleset, not a task. Do the task described in the "
    "user message using the inputs and tool findings provided there, then shape "
    "your answer as ONE complete Claude Agent Skill markdown file and NOTHING "
    "else:\n"
    "\n"
    "- Begin with a YAML frontmatter block delimited by `---` lines, containing at "
    "least `name:` (a short kebab-case skill id) and `description:` (one line: WHAT "
    "the skill does and WHEN to use it).\n"
    "- After the frontmatter, write the skill body as Markdown: an `#` H1 title, a "
    "short overview of what the skill does and when to use it, then clear, ordered "
    "INSTRUCTIONS the agent should follow, plus any usage notes or examples the "
    "findings support.\n"
    "- Keep instructions concrete and actionable (imperative steps), grounded in the "
    "real findings — never invent capabilities the task did not establish.\n"
    "- Output ONLY the markdown skill file: no code fences around the whole "
    "document, no prose before the frontmatter or after the body.\n"
    "\n"
    "Shape ONLY the form. The content must be the real findings/instructions from "
    "the task — never produce a tutorial about what a Claude skill is instead of the "
    "actual skill."
)

# The canonical CODEBASE-SUMMARY shaping ruleset (s16/aflex — d239/d241 generic-spine
# FLEX probe). The output-style spec for the "read a local codebase -> write a summary"
# deliverable: it shapes the findings (gathered by a worker that READ the real files via
# the codebase bundle) AS a clean Markdown codebase summary. Like every seed it carries NO
# "how to read code" how-to — the node already DID the reading; this only tells it how to
# STRUCTURE the summary, and (the load-bearing rule for an honest codebase doc) to attribute
# every claim to the real file PATH it was read from, never an invented file or symbol.
CODEBASE_SUMMARY_RULESET = (
    "You are an OUTPUT-SHAPING ruleset, not a task. Do the task described in the "
    "user message — summarize the codebase using the files the upstream node actually "
    "READ (their paths and contents are in the inputs/findings) — then shape your answer "
    "as a clean GitHub-Flavored Markdown codebase summary:\n"
    "\n"
    "- Open with a single level-1 heading (`# `) naming the directory/module summarized, "
    "then a one-paragraph **Overview** of what this part of the codebase does as a whole.\n"
    "- Add a `## Files` section: one bullet per file actually read, as "
    "`- `\\`path\\`` — what it defines and its role`, grounded in the real contents.\n"
    "- Where it helps, add a `## How it fits together` section describing the key "
    "relationships/flow between the files (imports, who calls/extends whom), only as the "
    "real source supports it.\n"
    "- Use `inline code` for file paths, class/function names and identifiers; **bold** the "
    "most important concepts. Keep it tight — no preamble, no restating the task.\n"
    "- ATTRIBUTE every claim to the real file PATH it came from; NEVER invent a file, path, "
    "class or function that was not in the files actually read. If something was not read, "
    "say so rather than guessing.\n"
    "\n"
    "Shape ONLY the form. The content must be the real findings from reading the code — "
    "never produce a tutorial about the language or about how to read code instead of the "
    "actual summary of THIS codebase."
)

# The canonical SCHEDULE-LEG methodology spec (RP-4, d322/d332). UNLIKE the output-shaping
# writers above (which shape the FORM of a deliverable), and like RESEARCH_METHODOLOGY_RULESET,
# this is a METHODOLOGY ruleset: it shapes HOW a node SCHEDULES a recurring/scheduled task. It is
# the DEFINITION-LAYER home of the whole-DAG scheduling doctrine (d322: the SCHEDULED UNIT is the
# WHOLE composed DAG re-running FRESH at the scheduled time — a recurring request SCHEDULES the
# whole task, nothing runs now). The planner STAMPS it by DESCRIPTION-MATCH on the single node that
# binds the generic ``cron_add`` tool; the MODEL then authors the ``cron_add`` ``prompt`` = the WHOLE
# self-contained recurring task, which the engine STORES VERBATIM (no engine surgery — this ruleset
# REPLACES the retired ``cron_prompt_from_task`` string-surgery that used to rewrite the prompt, a
# d310/d311/d319 fabrication). OUTPUT-AGNOSTIC / flow-agnostic: any recurring task, not just email.
# It carries NO ``_COHERENT_ARTIFACT_DOCTRINE`` (it authors no deliverable — the fired run does).
RECURRING_SCHEDULER_RULESET = (
    "You are a SCHEDULE-LEG methodology ruleset — it shapes HOW you SCHEDULE the recurring "
    "task in the user message, not the format of any deliverable. The user asked for a task to "
    "run on a REPEATING schedule (e.g. 'every morning at 7am research the AI news and email me'). "
    "Schedule it by making ONE `cron_add` tool call, reasoning as follows:\n"
    "\n"
    "THE SCHEDULED UNIT IS THE WHOLE TASK. Set the `cron_add` `prompt` argument to the ENTIRE "
    "recurring action the user wants performed on each fire — the whole task, self-contained, "
    "exactly as a fresh agent would need to read it to do the COMPLETE job (e.g. 'research the "
    "latest AI news and email me a summary'). The stored prompt is re-run FRESH at each scheduled "
    "time: the whole orchestration (research → write → deliver) re-composes and runs again from "
    "it. So it must be the COMPLETE task — never a single sub-step (not just 'send an email'), "
    "never a placeholder or stub, never empty.\n"
    "\n"
    "AUTHOR IT AS A DO-THE-WORK TASK, NOT A RE-SCHEDULE INSTRUCTION. The `prompt` describes the "
    "WORK to perform, with the scheduling wrapper stripped: do NOT carry the 'schedule a recurring "
    "task to …' framing or the cadence/time ('every morning at 7am') into the stored prompt — the "
    "cadence belongs in the `schedule` cron expression, not the prompt. A stored prompt that told "
    "the fired run to 'schedule' something would re-schedule itself instead of doing the work.\n"
    "\n"
    "SET THE `schedule` TO THE CADENCE. Translate the user's cadence into a standard 5-field cron "
    "expression 'minute hour day-of-month month day-of-week' (e.g. a daily 7am brief → '0 7 * * "
    "*').\n"
    "\n"
    "ONE SCHEDULE LEG, NO RUN-NOW. Scheduling the task is the whole job here: make the single "
    "`cron_add` call and stop. Do NOT also perform the task now — the user asked for it on a "
    "schedule, and the scheduled fire will run the whole thing. (Only if the user EXPLICITLY asked "
    "for it BOTH now AND recurring do you also author the run-now work.)\n"
    "\n"
    "This shapes HOW you schedule — the stored prompt must be the real, whole task so the fired run "
    "re-composes the complete orchestration; never a sub-step, a re-schedule instruction, or a "
    "placeholder."
)

# RP-2 (d326, OPTION A) — COMPOSE the shared coherence + grounding doctrine into EVERY per-format
# WRITER spec, so each carries all 7 points in its own format idiom (the model self-selects the
# format spec; the ENGINE pins nothing). Done here — BEFORE ``CANONICAL_RULESETS`` captures the
# bodies — via the same Python-level composition seed.py already uses (``WEB_RESEARCH_RULESET``).
# GATHER specs (research-analyst / research-methodology / web-research) are deliberately EXCLUDED:
# they never author the deliverable, so the artifact doctrine does not apply to them.
MARKDOWN_WRITER_RULESET = MARKDOWN_WRITER_RULESET + _COHERENT_ARTIFACT_DOCTRINE

# AUTONOMY REBUILD P2C — the destination for the DELETED engine CSV rider (the raw
# write loop's ``_is_csv_ext`` branch that pinned "tabular only, no prose" as an
# engine prompt injection). The discipline now lives where behavior belongs: a SPEC
# the planner binds to a CSV deliverable node. Format-shaping only; the model
# authors every byte (the engine composes/fixes nothing).
CSV_WRITER_RULESET = (
    "You are an OUTPUT-SHAPING ruleset, not a task. Do the task described in the "
    "user message using the inputs and tool findings provided there, then shape "
    "your answer to follow these rules:\n"
    "\n"
    "- Emit PURE tabular CSV data: a single header row naming the columns, then "
    "one data row per record — nothing else.\n"
    "- NO prose, no title line, no explanations, no markdown fences, no trailing "
    "commentary — a CSV file that a parser reads directly.\n"
    "- Quote a field with double quotes only when it contains a comma, quote or "
    "newline; escape embedded quotes by doubling them.\n"
    "- Keep every row's column count equal to the header's; leave a genuinely "
    "unknown value EMPTY rather than inventing one.\n"
    "\n"
    "Shape ONLY the form. The values must be the real findings from the task — "
    "never fabricate rows to fill the table."
)
HTML_WRITER_RULESET = HTML_WRITER_RULESET + _COHERENT_ARTIFACT_DOCTRINE
SECTION_HTML_WRITER_RULESET = SECTION_HTML_WRITER_RULESET + _COHERENT_ARTIFACT_DOCTRINE
CLAUDE_SKILL_RULESET = CLAUDE_SKILL_RULESET + _COHERENT_ARTIFACT_DOCTRINE
CODEBASE_SUMMARY_RULESET = CODEBASE_SUMMARY_RULESET + _COHERENT_ARTIFACT_DOCTRINE

# The seed registry: name -> (description, ruleset body). The description is the
# planner-facing lookup text (body-free index, d10); the body is the shaping
# ruleset a sub-agent loads.
CANONICAL_RULESETS: dict[str, tuple[str, str]] = {
    "markdown-writer": (
        "Format the final deliverable as a clean GitHub-Flavored Markdown "
        "document — title heading, lead summary, sectioned headings, bullet/"
        "numbered lists and a Sources section. Bind to the node that PRODUCES a "
        "written report/document when the user wants structured, readable "
        "Markdown output (a .md document) — NOT HTML. This is a document-FORMAT spec "
        "for the WRITE/deliverable node only; NEVER bind it to a research/gather/"
        "analysis node.",
        MARKDOWN_WRITER_RULESET,
    ),
    "csv-writer": (
        "Format the final deliverable as PURE tabular CSV — one header row naming "
        "the columns, one data row per record, correct quoting/escaping, and NO "
        "prose, titles or commentary around the data. Bind to the node that "
        "PRODUCES a .csv deliverable (a data table the user will open in a "
        "spreadsheet or parse). This is a document-FORMAT spec for the WRITE/"
        "deliverable node only; NEVER bind it to a research/gather/analysis node.",
        CSV_WRITER_RULESET,
    ),
    "html-writer": (
        "Format the final deliverable as ONE self-contained, semantic HTML5 "
        "document — `<h1>` title, lead summary, `<section>` topics with ordered "
        "`<h2>`/`<h3>` headings, lists, tables and a Sources section of links. "
        "Bind ONLY to the node that PRODUCES the final written report/document when "
        "the user wants HTML / a web page / a .html file — NOT Markdown. This is a "
        "document-FORMAT spec for the WRITE/deliverable node only; NEVER bind it to a "
        "research/gather/analysis node (it would make that gatherer emit HTML instead "
        "of gathering notes).",
        HTML_WRITER_RULESET,
    ),
    "section-html-writer": (
        "Format the final deliverable as ONE self-contained, semantic HTML5 document "
        "— like `html-writer`, but HARDENED for a LARGE, COMPLEX, MULTI-SECTION, "
        "DATA-HEAVY report (many topics, long timelines, several tables) that is prone "
        "to incoherence. The model authors the whole document DIRECTLY, emphasising the "
        "coherence a long report needs: its own in-page nav from its section headings, "
        "unique section ids that its nav links resolve to, one well-formed document "
        "(no premature close, nothing after `</html>`, no repeated nav / Sources / "
        "duplicated sections, no truncated tail), every section grounded and every "
        "citation a real fetched URL. SELECT THIS over `html-writer` when the research "
        "is data-rich / multi-part and the report is long or multi-section; use plain "
        "`html-writer` for a SIMPLE, short, single-pass HTML page. This is a document-"
        "FORMAT spec for the WRITE/deliverable node only; NEVER bind it to a research/"
        "gather/analysis node.",
        SECTION_HTML_WRITER_RULESET,
    ),
    "research-analyst": (
        "A gather-NODE analysis CAPABILITY (a specialization a node carries, NOT a plan "
        "shape): shape the output as rigorous grounded analysis — concrete facts and "
        "figures READ from the sources (never a description of the search "
        "results), every claim traceable to its source, gaps and uncertainty "
        "flagged honestly. Bind to research/analysis/GATHER nodes (search, fetch, "
        "read, take notes) that must report real findings — this is the spec for "
        "gather nodes, NOT a document-format/output-style spec; the specialization "
        "the deep-research shape reuses each round.",
        RESEARCH_ANALYST_RULESET,
    ),
    "research-methodology": (
        "A gather-NODE research CAPABILITY (a specialization a node carries, NOT a plan "
        "shape): a rigorous, DOMAIN-AGNOSTIC research METHODOLOGY — "
        "self-select your gather bundle first, decompose the goal into its concerns, "
        "gather and READ real sources, note each as claim+source+gap, cross-verify across "
        "sources, deepen on the gaps, prune bad leads, question whether it is actually "
        "complete, then write from your notes. This is the CORE method for any "
        "gather-then-report task; SELECT it for a NON-web investigation where you "
        "self-select the matching gather bundle (e.g. a local codebase or a vector store), "
        "or as the generic research method when no domain-specific variant fits. For a "
        "LIVE-WEB investigation, prefer the `web-research` variant. Bind to a "
        "research/gather node — NOT a final write/format/deliverable node.",
        RESEARCH_METHODOLOGY_RULESET,
    ),
    "web-research": (
        "A gather-NODE research CAPABILITY (a specialization a node carries, NOT a plan "
        "shape): the LIVE-WEB variant of the research METHODOLOGY — like "
        "`research-methodology`, but for gathering from the WEB (current events, news, "
        "public online sources): self-select the web gather bundle (search + fetch + "
        "read), decompose, gather and READ the real result pages, note as "
        "claim+source+gap, cross-verify against the pages you read, deepen on gaps, prune "
        "bad leads, question completeness, then write from your notes. SELECT THIS over "
        "`research-methodology` when the evidence must come from the LIVE WEB; use the "
        "plain core method for a non-web (codebase / vector-store / file) investigation. "
        "Bind to a research/gather node — NOT a final write/format/deliverable node.",
        WEB_RESEARCH_RULESET,
    ),
    "claude-skill": (
        "Format the final deliverable as a Claude Agent Skill — ONE Markdown file "
        "with YAML frontmatter (name + description) and an instruction body (title, "
        "overview, ordered instructions, usage notes). Bind ONLY to the node that "
        "WRITES the skill / SKILL.md file; NEVER to a research/gather/analysis node.",
        CLAUDE_SKILL_RULESET,
    ),
    "codebase-summary": (
        "Format the final deliverable as a clean Markdown CODEBASE SUMMARY — an "
        "overview, a per-file bullet list (`path` — what it defines), and how the files "
        "fit together, every claim attributed to the real file path read. Bind ONLY to "
        "the node that WRITES the summary of a local codebase/directory; NEVER to a "
        "web research/gather node.",
        CODEBASE_SUMMARY_RULESET,
    ),
    "recurring-scheduler": (
        "A SCHEDULE-LEG methodology (a specialization a node carries, NOT a plan shape): "
        "for a RECURRING / SCHEDULED request ('every morning research X and email me', "
        "'daily at 8am …', 'schedule a weekly …'), it drives the ONE node that calls "
        "`cron_add` to store the WHOLE user task as the scheduled prompt, so the ENTIRE "
        "orchestration re-runs FRESH at each fire (the scheduled unit is the whole task, "
        "not a sub-step). SELECT it for a request that must run on a repeating schedule and "
        "bind it to the single cron_add / schedule node. Author ONE schedule leg storing the "
        "whole task — schedule-only: do NOT ALSO run the deliverable now (unless the user "
        "EXPLICITLY asked for both now AND recurring), and do NOT schedule only a sub-step "
        "(e.g. just the email). Bind ONLY to the cron_add / schedule node, NEVER to a "
        "research / write / deliverable node.",
        RECURRING_SCHEDULER_RULESET,
    ),
}

# The canonical specialization the deep-research shape reuses across all rounds
# (s3/b2). Exposed so the live chat route can pick it as the ONE spec without
# hard-coding the name at the call site.
DEEP_RESEARCH_SPEC = "research-analyst"

# The DOMAIN-AGNOSTIC research-METHODOLOGY core spec (d256/d258). SB-RR (d292): now that
# the ROLE_RESEARCHER role is RETIRED, THIS is the lever that makes a WORKER-default node
# self-select its gather bundle — its body says "self-select your gather bundle first,
# decompose, gather + read, note as claim+source+gap, cross-verify, deepen on gaps, prune,
# question completeness, write from notes". A gather node carries it (composed ahead of the
# round's output-quality spec) so the gather posture comes from the SPECIALIZATION, never a
# role. Exposed so the engine/seed can compose it without hard-coding the name at the site.
RESEARCH_METHODOLOGY_SPEC = "research-methodology"

# The SCHEDULE-LEG methodology spec (RP-4, d322/d332). Exposed so the engine/seed/tests can
# reference the canonical name without hard-coding the string at the call site. The planner
# stamps it by DESCRIPTION-MATCH (no engine spec-name conditional) on the cron_add node.
RECURRING_SCHEDULER_SPEC = "recurring-scheduler"


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
    "SECTION_HTML_WRITER_RULESET",
    "RESEARCH_ANALYST_RULESET",
    "RESEARCH_METHODOLOGY_RULESET",
    "WEB_RESEARCH_RULESET",
    "CLAUDE_SKILL_RULESET",
    "CODEBASE_SUMMARY_RULESET",
    "RECURRING_SCHEDULER_RULESET",
    "DEEP_RESEARCH_SPEC",
    "RESEARCH_METHODOLOGY_SPEC",
    "RECURRING_SCHEDULER_SPEC",
    "CANONICAL_RULESETS",
    "make_ruleset_spec",
    "seed_ruleset_spec",
    "seed_canonical_rulesets",
]
