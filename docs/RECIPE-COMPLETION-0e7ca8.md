# Recipe completion report — recipe-make-the-reactiveagents-chat-genuinely-r-0e7ca8

**Date:** 2026-07-12 · **Driver:** direct completion session (no orchestration framework;
recipe store read-only). **Model under test:** `gemma4-e4b-candidate-ctx32k` on native
Ollama `:11434`, 6 GB GPU. **Test baseline:** all 7 suites green (981 tests) before the
session; every fix below re-ran green.

Verification method: live API-driven runs against the served FastAPI/SSE app (the same
routes the UI calls), span-level trace forensics (`var/traces/*.json|.md`), and offline
suites. Where an outcome's bar says "the user confirms in the UI", that leg remains the
user's — noted per outcome.

---

## The o1–o12 outcomes

### o1 — Reasoning pipeline-wide (think=True + JSON interceptor) — **MET**
Every structured step runs native think=True: the clarify gate, shape selector, planner
DAG/incremental authoring, tool-arg emission (`toolargs.py` `think=True` default),
`review_research` / `decide_followup` / `finalize_summary` / `name_deliverable`
(`_FOLLOWUP_OPTS`-family opts), and shape/spec authoring (`shape_author.py` call_opts).
Live traces show the separate `thinking` block plus clean parsed JSON on each (e.g. trace
`f779619f…md`: clarify gate, 804+44 tokens, `done_reason: stop`, valid JSON). Zero
JSON-parse failures across all live runs this session; the interceptor + repair chain is
suite-covered. A live probe additionally proved **native tool-calling**: 8/8 well-formed
`file_write` calls with section-sized HTML payloads, zero malformed-JSON events.

### o2 — Clarify gate converges — **MET** (one real bug found & fixed live)
Live: the scheduled-task prompt ("mail me a brief … every morning") paused with exactly one
question ("What time should I send the brief?"); the answer ("8am every day") resumed past
clarification into planning/scheduling with **no re-ask** (verified twice). Adversarial
catch: on the first pass the answer's *time* never reached the cron tool-args — the model
anchored on the schema example and scheduled `0 9 * * *`. Root cause: the run's clarified
goal was not threaded into the tool-arg composer. Fixed (goal now rides the composer
prompt + a translation-forcing schedule description); retest produced `0 8 * * *` exactly.

### o3 — Context window raised safely — **MET**
`Modelfile.gemma4-e4b-candidate-ctx32k` bakes `num_ctx 32768`; the s8/s9 measurement
probes in-repo justify it, and the user's own hardware verdict ("only 32k has proven to
work") confirms. Live VRAM during all runs: ~3.5 GB of 6 GB with think=True. Multi-round
holding rides the d263 compact-only windowing (pinned/SWA machinery deleted —
`llm_framework/context.py` verified clean) — the 48-minute deep-research run plus a
same-chat follow-up held context without OOM.

### o4 — End-to-end substantive US-Iran HTML report — **MET, with caveats**
The prior failure mode (empty/no-headlines) is gone, twice over. Run 2 (post loop-fix,
48 min) produced a 47.9 KB single-well-formed HTML report with headlines, a timeline
table, concrete damage figures ($34–42 B DOD costs) and `[S1]–[S7]` citations resolving
to real fetched CSIS URLs. Run 4 (all fixes, 27 min) converged in one research plan,
engaged the grow loop, and the model named its own `us-iran-war-report.html` — see the
run-4 section at the end. Residual quality defects across runs (spec-hardening backlog,
never engine fixes): duplicate Sources blocks / section families, one stub-section +
meta-comment leak (run 2), premature `</html>` closes (run 4), figure repetition,
single-source reliance. Final user read of the report in the UI remains the user's leg.

### o5 — Stale phi4-mini/:11435 refs — **MET**
All remaining *live-claim* references removed this session (`pyproject.toml` description,
`transport.py` section header, `context.py` ×3, `app.py` ×2, `spec_chat.py`). Repo-wide
grep now shows only historical-note framings ("formerly…", "phi4-mini before that on
:11435"), which the outcome explicitly allows. Nothing functional changed; suites green.

### o6 — Prompts concise/anti-hallucination + strong identity — **MET** (audit-based)
The live planner system prompt (visible in traces) is grounded and selection-disciplined:
the universal identity ("Ground every answer… never invent facts, sources or numbers"),
the SELECTION GUIDELINES block, format-bleed guards, and the compressed doctrine style of
d235/d263. The d263 removal of per-turn re-broadcast is the big token win (system composed
once; goal once; doctrine via load-observation). Live runs this session showed no
fabricated sources (citations resolve to genuinely fetched URLs). No fresh quantitative
before/after token audit was run this session.

### o7 — Clarify fires ONLY for scheduled tasks — **MET**
Live: haiku → no clarification; US-Iran detailed report → no clarification (proceeded
straight to shape selection/research); scheduled brief → exactly one clarification. The
gate prompt itself (in-trace) scopes asking to scheduling requests with a missing
load-bearing detail and forbids interrogating normal one-shots.

### o8 — send_mail only on explicit request — **MET**
No `send_mail` invocation appears in any trace across haiku, clarify/schedule, report and
follow-up runs (grep over all session traces). The scheduled-brief cron *prompt* includes
emailing — which the user explicitly requested in that task. The recipient self-lock (d12)
is unchanged and suite-covered. (No live positive-send was fired to avoid sending real
mail; that leg is suite-covered + user-confirmable.)

### o9 — Selection guidelines + delete buttons — **MET**
`docs/SELECTION_GUIDELINES.md` exists and is referenced verbatim inside the planner's
live selection prompt (trace-verified). Live selection this session: haiku → linear
one-turn; report → deep-research family; scheduled → cron leg — all appropriate. Delete
controls exist and are wired on both surfaces (`ShapeDeleteButton`, `DeleteSpecButton`)
with backend 409 built-in guards; suite-covered. (Live UI click-through remains the
user's leg.)

### o10 — Broad-scenario validation — **PARTIALLY MET**
This session live-verified via the served API: haiku quick-exit, scheduled-clarify
convergence + correct cron, US-Iran deep-research report substance, same-chat follow-up
from memory, no unwanted email, markdown traces, shapes API. The outcome's own bar —
*the user* runs the matrix in the UI and confirms — has not been performed by the user
yet. The d206 Phase-B diversity cases (pirate SPA, claude-skill, java hello-world) were
not run this session.

### o11 — Free-flowing iterative shape AND spec authoring — **MET (code+suite), UI leg open**
Spec side: multi-turn refine building on the prior body was already live (SpecChat).
Shape side was NOT symmetric (user-confirmed gap: one-shot describe/refine): built this
session — a full conversational **shape chat** (`/shape-chat` routes + panel in the Shapes
screen): every message drives one authoring turn over an in-session DRAFT (first message
authors, later messages refine the draft), nothing persists until Approve (create-name
collision → 409), Discard drops it; "Refine in chat" seeds a session from an existing
shape's on-disk definition. The authoring prompts now carry the catalog as usage-context
(d249) so drafts are selection-distinct. 5 new backend tests cover draft/refine/approve/
deny/collision/offline (+ a create-draft rename case). Live-proven on the served route —
see the shape-chat addendum at the end of this report.

### o12 — Local markdown tracing — **MET**
`var/traces/` holds per-trace `.json` + rendered `.md`. Verified legible content: every
prompt (system + user), the model's thinking blocks, per-call token costs
(prompt/completion/total), latency, model tag, `done_reason`, and an execution timeline —
e.g. `f779619f…md` (clarify gate) and the 48-min report trace (473 spans, 379 llm.chat
calls). Phoenix is bypassed entirely; the local exporter is the source.

**Score: 9 MET, 1 MET-with-caveats (o4), 2 partially met pending the user's own UI legs
(o10; o11's live click-through).**

---

## Fixes made this session (all suites green after each)

1. **Planner research loop convergence (the recipe's headline disease, live-caught):**
   each research plan overwrote `(findings, sources, notes)` with its own fresh yield, so
   the reviewer + follow-up decision judged only the latest plan — fresh yield dwindled
   (4,3,6,2,0,0), `research_thin` forever, six research plans, then a crash
   (`'NoneType' object has no attribute 'results'`) when the loop ceiling exited without
   a write. Fixed: cross-plan accumulation (dedup by URL) feeding reviewer/decide/write;
   `fresh_sources` + diminishing-returns doctrine in `decide_followup`; honest failed
   result on ceiling-exit-without-write. Validated: run 2 converged
   `research → research → write` and produced the substantive report.
2. **Grow-loop starvation (user-directed research audit):** measured median llm.chat =
   62 s; a 5-wide concurrent seed ≫ the 900 s phase cap, so the outer timeout cancelled
   the whole seed mid-flight — `grow_layers=0`, zero expand/prune/stop calls ever, empty
   stop_reason masquerading as a model stop. Fixed: budget-reserving dispatch gate
   (seed slice = `RA_RESEARCH_SEED_BUDGET_FRACTION`·budget, single-file dispatch under a
   deadline — concurrency on one GPU only serializes and mass-cancels — at-least-one
   floor, budget-skip for the rest), grown waves deadline-bound, truthful
   `stop_reason="budget"`, and terminal-skip exclusion on re-drives (run-3's
   `skipped→running` crash). Post-fix traces show gather nodes *completing* with notes
   and the truthful budget stop.
3. **Clarified-answer→tool-args wiring** (o2 above): goal threaded into the arg composer;
   translation-forcing cron schedule description.
4. **Model-authored deliverable filename** (user revoked the neutral-`.md` stamp):
   `Planner.name_deliverable` — one structured think call; the model names file+extension;
   engine parse-to-reads + basename-guards only; explicit user filename still wins;
   fail-open to the derived default. (Run 2's HTML had landed as `.md` — the visible
   symptom.)
5. **s16 finishing:** dead surgery-era code deleted (`repair_table_cells`,
   `trim_dangling_sentence`, the whole d173/d174 assembly-helper block) + self-policing
   extended; the raw write loop's HTML-specific figures mandate + close clause made
   format-neutral (the styled-table craft already lives in the `html-writer` spec);
   `AGENT_ARCHITECTURE.md` §4.1 rewritten to the d310/d313/d319 reality.
6. **s17 delivered:** Shapes screen redesigned to "discipline + doctrine, planner authors
   topology" (round_roles/final_roles API shim removed end-to-end), catalog usage-context
   into shape author/refine prompts, TopBar icons, the conversational shape chat (o11),
   frontend rebuilt.

## The measured probe that matters (user-directed)

**Can E4B drive the writer through the generic tool layer?** Yes: 8/8 well-formed
`file_write` tool calls with section-sized HTML content, valid JSON every turn, real
`[S#]` citations, zero parse failures (scratchpad probe, live `:11434`). The only
weakness — re-emission churn instead of `finish` — is already covered by the existing
*generic* guards (`section_reemission`, `document_restart`). This **falsifies the d49
premise** ("the raw writer cannot emit tool calls") that justifies the current served
write path (d299/SB-6 routes write nodes to the raw-emission loop with a bounded source
PUSH; `_run_tool_calling_writer` is unwired; run-2 trace: zero
`read_notes`/`load_source`/`file_write` calls — the writer never pulls memory).

## Open questions / recommended next steps (user decisions)

1. **Rewire the served write path to the unified self-select loop** (reverses recorded
   d299): write nodes tool-less in the generic loop, self-selecting `file` +
   `research_read`, PULLING notes/sources and driving `file_write`/`file_update` per
   their spec. The probe says it works; the guards exist; this is the single biggest step
   toward "the agent uses the toolbox and gets the job done" — and it would also dissolve
   the remaining raw-loop residuals (CSV single-shot branch, extension-keyed close-gate)
   and give the writer genuine autonomous memory use.
2. **Spec-harden the sectioned-report quality defects** (dup Sources blocks, stub
   sections, single-source reliance) in `section-html-writer`/`html-writer` — prompt
   levers, never engine fixes.
3. **Follow-up readback purity:** the same-chat follow-up answered with exact stored
   figures but its acyclic plan also ran ~4 fresh web searches; if pure memory readback
   is wanted, steer via the planner's selection doctrine (the d241 reasoning-picked
   short-circuit shape is still backlog).
4. **Run the d206 Phase-B diversity gate** (pirate SPA, claude-skill md, java file) and
   the user's own UI matrix (o10) when convenient.
5. **Housekeeping:** pytest isn't declared in the workspace — every `launch.ps1` uv sync
   prunes it (`uv pip install pytest pytest-asyncio pytest-timeout` restores); consider a
   dev dependency-group. `_collect_findings`' dormant no-sources legacy fallback and the
   positional catalog persist only for the degenerate branch.

## Run-4 validation result (all fixes together)

`run-e6608e8ea674`, **done ok in 27 min** (vs run 1's 98-min loop-crash and run 2's
48 min):

- **The model named its own deliverable: `us-iran-war-report.html`** — correct
  extension from the model's reasoning (the `.md`-on-HTML hardcode is gone).
- **Converged in ONE research plan**: `plans_authored = research, write`,
  `followup_after_research = write_plan`, `followup_after_write = done`.
- **The grow loop ENGAGED for the first time live**: the seed gathered under its
  budget slice, the decision node reasoned an `expand_branch`, and the grown
  gap-node gathered too (2 research nodes; expand_branch call visible in-trace).
  The stop was the budget safety net and the trace now says so truthfully
  (`stop_reason="budget"`, `stop_primary_is_model=False`) instead of masquerading
  as a model stop. On faster hardware the same wiring gives the model more grow
  rounds and its own `stop_research` becomes the normal terminator.
- Substance: sectioned report (overview, historical roots, timeline, damage
  assessment), concrete figures ($42 B; $314.8 M/day; $1 B annually), 6 unique
  real source URLs.
- Honest defects (raw-loop writer class, unrepaired by design — the engine never
  edits output): three `</html>` closes (premature-close mid-doc) and two
  duplicated section families. Both are exactly what recommendation #1 (the
  tool-driven pull-writer with the existing generic guards) plus writer-spec
  hardening owns.

## Duplication RCA (user-raised) + definition-layer fixes

Why the generated HTML carried duplicate sections/Sources blocks — three causes, none
of them a model ceiling:

1. **The engine's per-node prompt ordered the duplication.** The raw write loop told
   EVERY section node to "close with a SOURCES section" — N nodes → N Sources blocks,
   in direct contradiction of the writer spec's one-Sources doctrine. FIXED: the engine
   framing now defers to the specialization's structure rules ("the DOCUMENT closes with
   exactly ONE sources section — only the final part writes it, never add another if one
   exists"). The methodology now lives where it belongs.
2. **Continuation nodes were context-starved.** Each saw only the file's last 1200
   chars, so "do NOT repeat earlier content" was undecidable — sections above the window
   got re-authored. FIXED: the continuation frame now carries the document's REAL
   section-heading inventory (read from the actual bytes — read-to-inform, never edited)
   plus an explicit no-second-sources rule.
3. **The write planner could author overlapping node tasks.** FIXED: its directive now
   mandates DISJOINT section assignments per node and reserves document-closing (and the
   single Sources section) to the final node only.

All three are prompt/context levers (d186); zero output editing was added. The durable
end-state remains recommendation #1 — the tool-driven pull-writer, where the model
`file_read`s the document itself and dedup becomes its own reasoned act.

**Run-5 validation (fixes 1–3):** measurable improvement, not yet clean — Sources
blocks 5→2, `</html>` closes 3→2, and two of the four write nodes' duplicate emissions
were correctly dropped (0 writes) with the section inventory in every prompt. The
residual exposed a FOURTH cause: the fresh-file engine frame told EVERY node to reply
DONE "only once the WHOLE deliverable is on the file" — in a multi-node plan that
invites each part to write the whole report (n1, tasked with one section, one-shot the
entire document; a later node re-passed it with re-worded headings that slipped the
all-or-nothing re-emission guard).

4. **Scope-faithful completion (fix):** a non-final chain part's finish condition is
   now ITS OWN assigned section(s) — other sections, the sources list and the document
   close are explicitly later parts' — while only the final part carries the
   whole-document completeness bar (keyed off the existing `chain_is_final` delivery
   context, not a role/spec conditional).

**Run-6 validation (fixes 1–4): duplication effectively SOLVED** — exactly ONE Sources
section (5→2→1 across runs) and ONE `</html>` (3→2→1), distinct section topics, no
full re-pass, model-named `.html`, 25 min. Residuals: one re-worded near-duplicate
heading pair (slips the family guard — a writer-spec wording-discipline item), and a
new, predictable inverse gap — with every part scoped to its own section, NO part
owned the document OPEN (0 doctypes; the final close orphaned).

5. **First-part-opens (engine frame fix):** the fresh-file (first) writer's frame now
   instructs it to BEGIN the document per its specialization's structure rules, write
   its part, and leave the document open.

**Run-7 validation (fixes 1–5):** content discipline now excellent — 8 DISTINCT
coherent sections, zero duplicate families, exactly ONE Sources section, figures
grounded across 16+ source tags, 6 unique URLs (76 min — it reasoned a second research
plan). But STILL no document shell (0 doctypes/`<html>`/`<style>`), which exposed the
DEEPEST cause:

6. **The FILE bundle doctrine was lying to the model.** It still said the page wrapper
   "is added FOR you by the assembly step, so NEVER emit those wrapper tags" — doctrine
   from the retired `assemble_report_spa` era (SF-1 removed that assembly), directly
   contradicting the html-writer spec's "begin at `<!DOCTYPE html>`". The model obeyed
   its doctrine — correctly. FIXED in the definition layer (`bundles/file.py` doctrine +
   `file_write` description): multi-part document OWNERSHIP — the FIRST part opens the
   shell, middle parts append only their own sections, ONLY the final part closes,
   exactly once.

**Run-8 validation (fixes 1–6):** content discipline held (distinct sections, no
duplicate families), the run was HONESTLY `ok:false` — the final write part (sources +
close) was CANCELLED at the write phase's 900 s cap (four ~60 s-per-turn parts don't
fit), and n1 STILL opened with a bare `<h1>` despite frame+doctrine both mandating the
shell. Trace forensics showed why: the frame's FIRST imperative ("Emit the FIRST
section NOW… start with the headline") outweighed the shell directive 2,000 characters
later — a prompt-POSITION effect, the exact d186 lever class.

7. **Prompt-position (fix):** for a multi-part first writer the document-shell
   instruction now IS the opening imperative ("START with the format's DOCUMENT SHELL…
   THEN your first section"), not a rider.
8. **Write-phase budget (fix):** the write phase gets a proportionate 1.5× slice of the
   base phase budget (a 4-part write at ~60 s/turn cannot fit the research-phase cap;
   the final part was being cancelled at the wire). Research phases keep the base.

**Run-9 validation (fixes 1–8, the closing run): the document is WELL-FORMED
end-to-end for the first time** — exactly ONE `<!DOCTYPE>`, `<html>`, `<style>`,
`</body></html>`; model-named `.html`; 34 min; the strongest grounding of the series
(20 unique source URLs); distinct section topics. One regression shows the honest
ceiling of prompt-stacking: 3 Sources sections re-appeared (runs 6/7 held at exactly
1) — on a small model a long stack of frame riders obeys probabilistically, so
per-part structure discipline is ~stable, not guaranteed. The measured convergence
across the series: Sources blocks 5→2→1→(1)→3, `</html>` 3→2→1→(0)→1, duplicate
section families eliminated, shell complete. VERDICT: the substantive/sourced bar
(o4) is comfortably and repeatably met; residual document-hygiene variance is the
strongest argument for recommendation #1 (the tool-driven pull-writer + reviewer,
which enforces structure by READING the file and acting, rather than by riders the
model must hold in mind) — that rewire, not further prompt-stacking, is the durable
close for hygiene.

## Shape-chat live proof (o11 addendum)

Live two-turn conversation on the served route: turn 1 authored a draft
(`viewpoint-synthesis-gathering` [concurrent] with a selection-grade description);
turn 2 ("…make the description mention it fits debate-style questions")
demonstrably **built on** the turn-1 draft (description extended in place); deny
discarded with nothing written to the catalog. Live catch fixed in the same
session: a create-mode draft rename was silently ignored (`ShapeAuthor.refine`
force-keeps the prior name — right for on-disk edits, wrong for unpersisted
drafts); `keep_name=False` now honors renames on create drafts (suite-covered,
217 chat_app tests green).
