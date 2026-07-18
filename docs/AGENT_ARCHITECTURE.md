# ReactiveAgents — Agent Architecture (canonical)

**Status:** the single source of truth for the genuinely-agentic design. Consolidates
the settled decisions (d184, d185, d190–d219) so the design does not have to be
re-explained turn to turn. If code disagrees with this doc, the doc wins — fix the code
or update the doc with a recorded decision; do not let them silently drift.

> North star: behavior is driven by **SHAPES + SPECS + TOOLS** through reasoning.
> **No hardcoded, flag-based, or fabricated behavior.** Every behavior must emerge from
> a tool description, a spec, a shape, or the model's reasoning — never a control-flag
> that forces a path (d14/d60/d65). A failing result is a *design bug to fix*, never a
> "model ceiling" (d186).

---

## 0. The CoT-autonomy contract (2026-07-16 — supersedes any conflicting text below)

The owner's ruling after live-trace review: **the model's own chain of thought drives
the work; nothing spoon-feeds it.** No engine or tool text may command the next action
("search done, now fetch"), script a bundle/tool sequence, or re-prompt a conclusion
(all bounce-gates are deleted). Every behavior lives in exactly ONE owning text layer:

| Layer | Owns | Single home |
|---|---|---|
| Identity | who the agent is + channel protocol ([TOOL RESULT] envelope, JSON-only-when-asked) | `llm_framework/transport.py AGENT_IDENTITY` |
| Operating protocol | reason → ONE tool call → observe → … → finish; bootstrap via get_bundles | `bundles/base.py AGENT_OPERATING_PROTOCOL`, injected once per node system turn |
| Role | one drive statement per role (worker / reviewer / synthesizer) | `roles.py ROLE_FRAMINGS` |
| Shape | planner-only, step-by-step plan-AUTHORING strategy | `shapes/*.toml decompose_methodology` |
| Specialization | how to drive a brief to its output (business logic + quality bar) | `specialization/seed.py` |
| Bundle doctrine | domain knowledge, delivered once on the get_bundles LOAD ack | `bundles/*.py own_doctrine` |
| Tool description | what the tool does + how to use it well | ToolDef / spec `description` |
| Tool output | FACTS only — state, counts, caps, cursors, error_kind, real candidates | handlers / observation builders |
| Engine | orchestration, messaging, resource caps — zero instructional text | runtime / agentic |

Key consequences (each enforced by `agent_runtime/tests/test_no_steering_strings.py`
over the single registry `scripts/promptlab/retired_strings.py`):

- The first user turn carries only the brief + run DATA (a declared `DELIVERABLE
  FILE`, nothing else); the fetch budget is stated on the research-bundle load ack —
  the moment of relevance — never on every brief.
- The gather-more / note / target-artifact bounce-gates are DELETED. Honesty is
  downstream: the persistence-side staleness guard (unchanged bytes ⇒ no artifact),
  truthful trace attrs, and the reviewer node reading the real file.
- The write-plan strategy (one write node + one same-spec `final_review`, source-id
  assignment, ground-or-drop) is the write-file SHAPE's methodology — not engine text.
- The research review is a PLANNER-BRIEFED node: `Planner.author_review_brief` writes
  its brief, it pulls from the research memory with the gather workers' specs, and its
  prose is the single signal `decide_followup` reasons over (`review_research` is
  deleted).
- Channel robustness is not steering: a tool-shaped reply that breaks strict JSON gets
  a parse-error FACT, and an unambiguous multi-KB call with one bad escape is
  recovered verbatim (`_lenient_content_call`) — the model's own bytes, never composed.
- Prompt texts are validated with `scripts/promptlab/run_batch.py` — live batches per
  module, graded from traces including the model's captured thinking; a text ships at
  zero failures in a batch.

Sections below predating this contract remain accurate for structure (nouns, loops,
memory) but any nudge/gate/directive wording they describe is historical.

---

## 1. The core nouns (do not conflate them)

| Concept | What it is | What it is NOT |
|---|---|---|
| **Role** | the **node type** — *who* the actor is (d213) | not a bundle, not a spec |
| **Specialization (spec)** | the node's **doctrine/behavior** — *how* it acts (d194) | not a role, not tools |
| **Bundle** | a **tool wrapper**: topic/capability-specific tools + usage doctrine, loaded by a node (d212) | **NOT a role/actor**; it never plans/researches/writes/reviews |
| **Shape** | a plan's **execution DISCIPLINE + doctrine** (linear / modular / deep-research…) (d194; s16/a3 d239/d247) — the planner/grower **AUTHORS the topology by reasoning**; a shape NEVER pre-bakes a node graph or binds a tool | not a role; not a fixed DAG |

### Roles = node types (d213, d215)
The role **lives in the node type**. There are five, but they sit in three places:

- **PLANNER** — the planning **stage**. Authors each plan, drives the iterative loop,
  decides follow-up-plan-or-done. *Not an in-plan node.*
- **In-plan node roles** — **RESEARCHER, WORKER, REVIEWER**. These are the only roles the
  planner places *inside* a plan via `add_step`. **REVIEWER is the default last step of
  every plan** and emits the plan's final status.
- **SYNTHESIZER** — the **terminal** stage. Runs **once, after the planner loop exits**
  (no more plans). Delivers the final output (SSE + brief summary + downloadable
  artifact). *Never an in-plan node* (d215).

### Bundles = tool wrappers, drawn by CAPABILITY DOMAIN (d212)
`get_bundle(name)` returns `{ tools + doctrine }`. Bundles exist for **memory
management**: a node loads only the bundle(s) its task needs, so its context window holds
just those tools + doctrine, not every tool → lean per-node context → E4B determinism.

- A **base `object` bundle** (finish + the universal loop) is always included; every
  bundle extends it.
- Bundles are **capability domains**, not roles: `planning`, `research` (gather:
  search/fetch/note/expand/prune/stop + cross-verify), `research_read` (READ a fetched
  source on a **cost hierarchy**: `read_notes` — cheap article-note gist, *first* — then
  `load_source` — expensive verbatim, only for a figure/quote to cite), `file` (author a file).
- **A node SELECTS the bundle(s) it needs by REASONING over an advertised `get_bundles`
  catalog** (each bundle's name + capability **domain** + doctrine summary) — exactly as it
  picks a shape via `get_shapes` and a spec via `get_specs`. This is **NOT** a hardcoded
  role→bundle table. **Each agent self-selects its bundles at runtime** by calling
  `get_bundles(NAME)` to expand what its task needs (NODE-SELF-SELECT, d221); the **planner
  sets only role + spec per node, never bundles**. The only always-on default is the base
  `object` floor. A node may expand multiple bundles (e.g. a research node → WEB +
  note/decision overrides; a write node → `file` + `research_read`). There is **no
  `WriterBundle`**. A bundle can be **overridden/extended** in context — e.g. the research
  role expands the WEB bundle and overrides `web_fetch`'s output message to prompt
  note-taking (the plain WEB bundle has no such message).
- **TRUE self-select — every in-plan node starts TOOL-LESS (d242).** There is no
  role→bundle prime and **no pre-mounted domain tool anywhere in `runtime.py`**: every
  executor loop (research, anchored-review, tool-calling-writer, file-delivery, synthesis)
  and the **linear/chat worker** opens with only `get_bundles` + `finish` offered, and the
  node **OBTAINS its domain tools only by self-selecting** — `get_bundles(NAME)` registers
  that bundle's tools (`on_load`), pins its doctrine, and binds the run's ctx (the fetched
  sources). The offered tool surface is recomputed **each turn** from the bundles loaded so
  far (`SubAgent._offered_tool_specs`), so a freshly-loaded bundle's tools become callable
  next turn — exactly as the planner self-selects. (The two RAW-emission loops — file-delivery
  and synthesis — drive the file tools themselves, so they run a small **self-select front**
  to load their authoring doctrine before writing.) Reliability of self-select on the small
  model is a **tool-description + task-framing lever (d186)**, never a reason to pre-mount a
  tool or hardcode a load.
- **REGISTRY FOUNDATION — `hook.registry` IS the `GrowableToolRegistry` (A0, d265).** Self-select
  only registers a bundle's handlers because the served hook's registry has the `.add` growth
  point. `register_agentic_tools` builds the `GrowableToolRegistry` (bound to the hook) **and now
  ASSIGNS it back as `hook.registry`**, so `SubAgent._load_bundle` / `get_bundles` →
  `expand_bundle` → `registry.add(...)` genuinely records the handler + makes it dispatchable.
  The growable **adopts the hook's base `ToolRegistry` as an internal dispatch store** (the static
  core/lambda tools survive the swap), exposes a `.register` passthrough (so existing
  `hook.register` sites keep working) and a `.resolve` dispatch lookup (the hook's `invoke`
  resolves a `ToolSpec` via `resolve`, distinct from the growable's `get` which returns the
  selection `ToolDef`); `catalog()` delegates to that base so the planner's tool catalog is
  unchanged, while `names()`/the structured-selection enum stay the `ToolDef` set. **Before this
  fix** the returned growable was dropped and `hook.registry` stayed a base `ToolRegistry` with no
  `.add`: every bundle self-select silently no-op'd (the `AttributeError` swallowed by
  `expand_bundle`), invisible only because the served web write path **pre-registered**
  `load_source`/`read_notes` per run. That `load_source` per-run pre-registration is **retired**
  (a self-selecting write/review node now registers `load_source` via this growth point, bound to
  the run's `chain_sources`); the `read_notes` per-run pre-registration is kept as the **bridge**
  until the matching `chain_notes` feed to the write runtime's `_node_run_ctx` lands (so
  `read_notes` self-select binds there too) — a runtime-touching step folded into the SoC work.
- **The linear/chat worker is a node like any other (d241).** A trivial message is answered
  in one turn (prose, no tool call), but a follow-up that builds on prior research is **not
  special-cased out of self-select**: the worker self-selects the `research_read` memory-read
  bundle, reaches the session's prior sources/notes, and answers FROM them — still applying
  its spec. The loop is **generic** (it dispatches any loaded bundle's tool through the hook
  by name), so a future domain-agnostic memory-read bundle (codebase, vector-db) slots in
  with no loop change.
- **No role-phase methods on a bundle** (d212): a bundle exposes one `tool_specs(ctx)`
  catalog + `doctrine`.
- **FULL unrestricted self-select — no per-role tool subset anywhere** (d244, USER): a node
  that self-selects a bundle is offered **EVERY** tool of that bundle. A reviewer that loads
  `file` is offered `file_write` too; a writer that loads `file` is offered `file_update`
  too. There is **no per-role allow-list** curating the loaded surface down to a subset —
  the `only=review_tools`/`only=writer_tools` arguments at the anchored-review and writer
  loop callers of `_offered_tool_specs` are **gone** (the generic `only=` filter parameter
  remains for the runtime's own use, but no per-role subset is wired). The reviewer's
  **review + FIX-INLINE** behavior was its DESIGNED role all along (d245 — *not* a leak-guard;
  the `file_read`+`file_update` surface compensated for the small model's sectioned-writer
  unreliability), and it is preserved by the reviewer's **ROLE FRAMING + SPEC**
  (reasoning-enforced), never by a withheld tool. Loop dispatch is keyed on the **TOOL**
  (generic): any loaded-bundle tool flows through the `on_load` hook (`_dispatch_loaded_tool`),
  so a reviewer's `file_write` (or a writer's `file_update`) fires through the hook and works —
  file-family writes are forced to the single deliverable path (the single-deliverable
  invariant, applied uniformly, not a per-role restriction). If a write/edit literal ever
  leaks on live E4B, the fix is a **PROMPT/SPEC/description lever** (d186) — NEVER re-adding a
  per-role tool subset.

### Per-node anatomy
```
PLAN level:  SHAPE = execution discipline + doctrine  (get_shapes → linear | modular | deep-research…)
             (the planner/grower AUTHORS the topology by reasoning — no deterministic unroll)

each NODE  =  ROLE           (node type)               → WHAT actor      (d213)
           +  SPECIALIZATION (a spec; 'none' is valid) → HOW it behaves  (d194)
           +  BUNDLE(s)      (get_bundle → its tools)  → WHAT tools       (d212)

   behavior of a node  = its SPECIALIZATION
   structure of a plan = its SHAPE
   tools of a node     = the BUNDLE(s) it loads
```

---

## 2. Orchestration: the iterative planner loop (d214, d215)

There is **no fixed pipeline**. One generic mechanism:

```
[user goal]
   │
   ▼
PLANNER stage ───────────────────────────────────────────────┐
   get_bundle(planning) → get_shapes (pick SHAPE) + get_specs  │
   seed plan → add_step ×N (each node: ROLE + SPECIALIZATION,  │
              spec ALWAYS set, 'none' valid) — DAG is GROWABLE  │
   │                                                           │ final status
   ▼                                                           │ (from the plan's
   PLAN executes:                                              │  last-step reviewer)
     ( researcher | worker ) ×N  →  REVIEWER (default last     │
                                     step) → emit FINAL STATUS ─┤
   │                                                           │
   ▼                                                           │
   PLANNER: follow-up plan needed? (e.g. WRITE after RESEARCH) │
        YES ── author the next plan ───────────────────────────┘
        NO
        ▼
   loop EXITS
        ▼
   SYNTHESIZER (terminal, runs ONCE)
        SSE "artifact downloadable" + brief summary
```

- **Reviewer = the default last step of every plan**, not a wired stage. Its emitted
  status is what the planner reads to decide the next plan.
- The DAG is **growable**: for deep-research the engine emits a single **tool-less
  self-selecting** research seed (no shape-baked nodes, no shape-bound `web_search`; s16/a3
  d239/d247) and the **DagGrower grows the DAG** at runtime — decompose-first into scoped
  facets, then `expand_branch` on note gaps. The research **topology is authored by reasoning**,
  never unrolled from a fixed round graph.
- **The grow loop is SOURCE-AGNOSTIC (as4 — d227/d241/d186).** The grower owns **no web
  vocabulary**: every gather node it builds (the decompose-first seed children + the grown
  gap nodes) is **`ROLE_RESEARCHER` + TOOL-LESS** — it self-selects its gather bundle (web /
  vector-db / codebase-read / files), and that bundle's tool drives the gather. The grower
  drives expand/prune/stop over **whatever structured artifact** the bundle yields: it
  **recognizes** a gather node by its RESEARCHER role (or its bound research/complex-memory
  handle) — never by a `web_search` tool — so a **tool-less fallback seed is folded** (the
  rare decompose-empty path), and it **ingests** the gather artifact under either the web
  vocabulary (`article_notes`/`fetched`) or a generic key (`notes`/`records`). Adding a
  non-web gather source is a **bundle**, not a grower edit.

### 2.1 The realized engine (as1 / d239 — ONE loop, real stages, no fork)

There is **one** generic loop, `chat_app.agentic._run_generic_loop`, and **no bespoke web
fork**: EVERY shape routes through it (the retired `agentic.py:691` deep-research-AND-web fork is
gone). `run_agentic` only computes a **seed** — the first plan kind — and calls the one loop:

- **`first_plan_kind="research"`** for the web deep-research family (`is_deep_research` =
  `execution=="deep-research"`, s16/a3 — re-keyed off the retired round/final roles; + web-allowed +
  no unmet spec): research → (planner reasons) → write → (planner reasons) → done. The research
  plan is seeded by `_research_seed_dag` (a tool-less growable seed) and grown by the DagGrower —
  there is **no `unroll_shape`** and no shape-bound `web_search`.
- **`first_plan_kind="acyclic"`** for every other shape (linear / modular / escalate / no-web /
  missing-spec): the incremental node-by-node authorer authors ONE plan, then the planner
  reasons `done` after a single iteration. `run_plan_chain` / `_run_acyclic` remain only as thin
  **seed shims** into the one loop — not separate engines.

The three previously-**faked** seams are now **real** (the `_research_plan_final_status` /
`_write_plan_final_status` pure-functions and the hardcoded research→write→done while-loop are
retired):

1. **Last-step reviewer status is real.** The research plan's last step is a real
   `Planner.review_research` LLM call (emits `research_complete`/`research_thin` + the **data
   complexity** over N points, d237, + the memory handle); the write plan's status is **read**
   from its real `final_review` node. The loop READS these — it no longer hardcodes them.
2. **The follow-up is the planner reasoning.** `Planner.decide_followup` (a native think=True
   structured call over the reviewer status + findings digest) decides the next plan —
   `research_plan` / `write_plan` / `review_plan` / `done` — replacing the hardcoded while-loop.
3. **The terminal synthesizer summary is LLM-generated.** `Planner.finalize_summary` produces
   the human-facing digest the synthesizer streams (+ a downloadable artifact when a file was
   produced); it is **never a fixed string**.

All three LLM calls are **fail-safe to a safe baseline** (the offline `FakeTransport` seam / a
malformed reply → research→write→done / a derived summary), so the suite and offline paths stay
green while the live thinking model gets the real reasoning. A missing-specialist acyclic plan
still PAUSES for the user CHOICE (no follow-up, no synthesizer).

**S5 (d240) — the model decides when to stop.** The researcher's `stop_research` is the
**primary** stop: the growable drive loop breaks on the model's no-expansion *before* the
max-layers ceiling. The depth ceiling (`config.depth` = the shape file's `max_iter`, the N4
high ceiling = 10 on the served route) **+** the wall-clock budget (`timeout*0.9`) are a
**non-deciding safety net** guarding runaway growth only — a `depth_bound`/`budget` stop is the
exception, not normal operation. `completeness_stop` stays reasoned doctrine.

---

## 3. The research loop (d184) — inside a researcher node

Driven by **tool descriptions**, not flags. The canonical loop:

1. Identify the topic + its **concerns** (decompose-first template).
2. Search a concern.
3. **Read 1+ of the most relevant chunks** (read returns top-N chunks, not just one).
4. **Take a NOTE** — the *primary* act after each read (not "findings prose"): what was
   learned **and the gaps it left**.
5. For all concerns…
6. Verify the notes — anything still to clarify?
7. If yes → **`expand_branch`** = genuinely run a new search/read/note round for the
   sub-topic (expanding **commits to gather** — expand-then-stop is incoherent by tool
   semantics).
8. If the new data adds meaning → note it.
9. If not → **`prune_branch`** = collapse the concern (a real, used move).
10. **STOP only when every concern is settled-in-a-note OR collapsed, and no new concern
    remains.**

"Breadth is not depth": cover every concern the goal names before drilling one.

### Notes architecture (d185) — 3 layers
1. **Granular per-concern GRAPH notes** — one structured note per fetched article
   (claim/summary, source-trust, `gaps_or_followups`). Structured as a graph: concern
   nodes → notes → cited sources, with `gaps_or_followups` as **edges that spawn new
   concern nodes** (this is literally how expand/prune walk it). Not a single blob.
2. **Per-research BRIEF** — one short digest per research, at the **chat-session** level,
   so multiple researches in one chat coexist, each addressable by its brief.
3. **Session binding** — research state is **keyed to the chat session**, not truncated
   per run. A follow-up (“who reported that figure?”) **reads back** the exact note + its
   source instead of re-researching.

### Source-agnostic gather — THIN ENGINE, web is just one bundle (SA-4 / d254)

The gather loop (`SubAgent._run_research_loop`) hardcodes **no** source semantics. A gather
node self-selects its gather bundle and the loop dispatches whatever it loaded:

- **Web tool** (the configured `web_search` / `web_fetch` / `note`) → the engine **delegates
  to the WEB BUNDLE's gather adapter** (`bundles.web_ingest.WebGatherAdapter`, reached via
  `ResearchBundle.gather_adapter()`). SA-5 (d254) **relocated every web semantic OUT of the
  engine into that bundle**: offered-URL grounding (`url_offered`), the article-readability
  filter (`looks_like_article_url` / `is_readable_fetch` / `NON_ARTICLE_EXT`), the
  `{title, url, markdown}` fetched-record shaping, and the SEARCH-RESULTS + coverage
  observation prose. The engine's `_dispatch_research_tool` is now a thin **delegator** that
  hands the adapter this run's hook `invoke` closure, the `read_fetched` read closure (which
  still holds the engine's embedder + read budgets — a generic long-text read utility), and
  the bundle-sourced web_fetch take-a-note suffix. Behaviour is **byte-comparable** (the
  served web path is the contrastive gate); only the OWNER moved. The web tool
  **implementations** (`reactive_tools/web_tools.py`: ddgs search, httpx + Trafilatura→markdown
  fetch, SSRF-guarded, cached) already live in the web layer — **web is one self-contained
  bundle, the engine hardcodes none of it**.
- **Any OTHER self-selected tool** (a NON-web bundle: `codebase`, a future vector-db / bash)
  → the loop **falls through to the GENERIC by-name on-load hook dispatch**
  (`_invoke_loaded_tool` / `_dispatch_loaded_tool`). It is **never mis-dispatched as a
  web_fetch**. Each call is captured as a **source-agnostic RECORD** (`{title, url, markdown}`,
  where `url` is a stable synthetic id like `read_file://path`) attached to the node's
  `tool_value` under the generic **`records`** key — the mirror of the web `fetched` key.

A downstream writer grounds in a non-web record through the **SAME chain_sources harvest** the
web path uses: `collect_fetched_sources_full` reads **`fetched` OR `records`**, so a section
node resolves its `source_ids` to a codebase file's text exactly as it does a fetched article.
The trace counts `research.sources` = `fetched + records` (+ a distinct `research.records`), so
a non-web gather's leaf-capture is visible (`fetches==0` with `records>0` is a real gather, not
a regression). **Adding a new gather source is now "add a bundle", with zero engine edit** —
the d254 Tier-2 gap is closed.

### Read hierarchy binds by self-select on the write path too (SA-4 / d234 / d235)

The `read_notes` (CHEAP) → `load_source` (EXPENSIVE) read hierarchy now binds entirely via
**bundle self-select** on the served write path — **no per-run pre-registration** (the SA-1
registry-foundation fix retired the `load_source` pre-reg; SA-4 retires the `read_notes` one).
The seam: the runtime carries a **`chain_notes`** field (the mirror of `chain_sources`);
`write_report_spa` sets `write_runtime.chain_notes = research_notes`; `_node_run_ctx` folds it
into `ctx['notes']`. So when a write/review node self-selects `research_read`, the bundle
registers **both** `read_notes` (from `ctx['notes']`) and `load_source` (from `ctx['sources']`)
through the working growth point — keyed to the same global `[S#]`. No mask, no `unavailable`.

---

## 4. The write flow (d192, d211) — emergent, never prescribed

The write structure **emerges from the planner**, is **never dictated to the writer
model**, and the writer is **never spoon-fed** the research (d237).

1. The **research plan's last-step REVIEWER** (the *research-reviewer* — a reviewer
   extension that emits research info back to the planner) emits a **write-planning EVENT**
   carrying the research summary, the **DATA COMPLEXITY** (the shape of the researched data
   over *N* data points — how many concerns + how complex), and the research-memory
   **handle** (d221/d237). It *reports complexity*; it does **not** dictate "write sectioned".
2. The **PLANNER autonomously decides the write shape** — **one pass** (a single writer) vs
   **several passes** (sectioned: *N* section-worker nodes) — by reasoning over the research
   **complexity** + the **output format** (HTML, from the user query) + the **available spec**
   (`html-writer` exists). Sectioning is an *emergent response to complexity* (it exists
   because one-pass isn't always feasible), never a default (d237 / d192-4).
   - **The one-pass-vs-sectioned decision is REALIZED by SPEC-VARIANT SELECTION (d246, as4),
     never a procedural branch.** The planner selects the write **spec variant** by reasoning
     over the research-reviewer's DATA COMPLEXITY against the advertised `get_specs`
     descriptions: **`html-writer`** (one-pass) vs **`section-html-writer`** (the sectioned
     variant). The **`section-html-writer` spec now EXISTS** (s16/ashw): a SEPARATE canonical
     seed (`specialization/seed.py` `CANONICAL_RULESETS`, advertised via `CURATED_SPECS`), with
     a hardened ruleset GROUNDED on the MEASURED live-E4B sectioned-HTML failure trace (the
     shell turn closing `</body></html>` early so later sections orphan after `</html>`,
     re-inlined per-section styling, repeated nav/Sources blocks, fabricated `[S#]`→URL
     citations, dangling nav anchors) and a **selection-lever description** that tells the
     planner to pick it ONLY for a complex/multi-section/data-heavy report and keep plain
     `html-writer` for a simple one-pass page. The ruleset shapes the OUTPUT QUALITY of a
     sectioned doc (one open shell, one nav, one Sources, real URLs, complete sections); it does
     **NOT** force a "write N passes / lead-page-first" procedure (d211/d218/d246). Base
     `html-writer` stays the one-pass writer, untouched by sectioning (a light validity/
     completeness hardening only). The output-format floor (`_enforce_output_format_spec`) treats both as the same HTML
     **family**: it **respects** a planner-chosen variant (a complex-data `section-html-writer`
     is never overridden back to `html-writer`) and only stamps the one-pass default when the
     model bound **no** format writer at all. There is **NO** procedural `lead-then-sections`
     branch/flag in the engine (d211/d218/d246) — sectioning is the planner's emergent SPEC
     CHOICE realized as the authored topology (one `file_write` node, or *N*).
   - **s16/SA-6 PART 2 (skeleton-then-fill) was REJECTED and is CLOSED** (d316/d317/d319 — "fill
     the json body" is fabrication): there is NO skeleton/JSON-fill mechanism and NO engine
     compose. The LLM WRITES the artifact DIRECTLY per its spec via the generic file tools it
     drives; the engine orchestrates shapes + assigns specs + parse-to-reads, nothing more.
3. **No spoon-feeding (d237).** The writer is **never pre-fed** source bodies (the retired
   d49/d89/d170 ~73K push). Every write worker — lead *and* section — is a **tool-calling
   puller** reading on a **cost hierarchy**: **`read_notes`** (cheap article-note gist —
   summary/key_claims/gaps, which source has what) *first*, **`load_source`** (expensive
   verbatim, nearest-N chunked, length-tuned — top_n=2 / per_call_cap=3000 to fit the E4B
   window) only for an exact figure/quote it will cite (d233 / d234 / d235). The served write
   node routes to this puller (`_run_tool_calling_writer`, full-`INDEX` feed) whenever it has
   the run's chain sources + a deliverable path; the reviewer (`_run_anchored_review`) likewise
   pulls.
   - **Bounded d49 RAW-EMISSION EXCEPTION (as4).** The only residual full-body PUSH
     (`_scoped_source_block(full_index=False)` → `render_scoped_sources`) feeds **solely the
     RAW-emission loops** — `_run_file_delivery` (reached only for a write node WITHOUT chain
     sources, e.g. csv/txt/java → the scoped block is **empty**, so no research-body push) and
     `_run_synthesis` (a `ROLE_SYNTHESIZER` in-plan node, which the unified engine **no longer
     authors**). These loops emit RAW content and **cannot emit tool calls**, so they genuinely
     cannot pull (d49). On the served deep-research route the terminal synthesizer is
     `Planner.finalize_summary` (a digest over the findings + the already-authored deliverable,
     `synthesizer_in_plan_node=False`), so this full-body push is **not exercised** there. It is
     **kept, not restored/expanded** as the honest d49 exception; converting the raw loops to
     pullers is a larger d49 change, not in scope.
4. The writer is **never told** "you are a sectioned writer" or "start with a lead page"
   (d211); the `html-writer` spec supplies **format guidance only**.

**Forbidden (fabrication):** any hardcoded lead-then-sections scaffold, a
`_NOTES_FIRST_DIRECTIVE`, a "sectioned writer" directive reaching the model, or
**pre-feeding the writer the source bodies instead of letting it pull** (d237 — spoon-feeding).

### 4.1 Coherence — SPEC-driven, NEVER engine surgery (d310 / d313 / d319 — supersedes d196/d218/d219)

**SF-1/RP-1 (d310→d319) retired the whole coherence-era output-touching machinery**: the
anchored-review reviewer that edited HTML (`_run_anchored_review`, `_robust_match_spans`,
`review_injection.py`), `assemble_report_spa` + its folds (`rebuild_section_nav`,
`rebuild_sources_list`, `dedupe_source_lists`, `collapse_duplicate_section_ids`,
`ensure_source_coverage`/`best_match_node`), `_coherence_metrics`, `enforce_single_h1`,
`reconcile_doc_structure`, `trim_dangling_sentence`, `repair_table_cells` — all GONE
(self-policed by `test_sf1_reactive_coherence_retired.py`). The d219 "migrate deliberately"
window is CLOSED: no d173/d174 pass survives.

Coherence now comes from exactly two reasoning-driven places:

1. **Writer SPEC doctrine.** The writer specs (`html-writer`, `section-html-writer`,
   `markdown-writer`, …) carry the how-to-write hardening (`_COHERENT_ARTIFACT_DOCTRINE` in
   `specialization/seed.py`): one self-contained artifact, grounded citations to REAL fetched
   URLs only, never-empty/stub/truncated sections, no duplicate/overlapping passes, one clean
   Sources list. Bad output → harden the SPEC more; never patch the output.
2. **Detect-only guards.** The engine keeps ONLY read-and-decide predicates
   (`html_close_gap`, `has_truncation_marker`, `section_reemission`, `document_restart`) that
   inform the orchestration's persist/stop/nudge DECISION and never edit the model's bytes.
   The single permitted output-touch anywhere is parse-to-READ structured JSON (d311-8).

---

## 5. Memory model (d190, d192, d210, d263)

The determinism lever. **Context always flows to every node, but bounded per node.**

- **planner + synthesizer** context = the **chat** context.
- **researcher / worker / reviewer** context = **per-node** (not per-plan), fed prior-node
  context via the **DAG edges**.
- **research memory** is **stickied to the plan**, reached via **read-via-tools** on a cost
  hierarchy (`research_read`: `read_notes` cheap gist first, then `load_source` verbatim) —
  **never a verbatim dump** (d192/d202/d235).
- the **research brief** lives at the high (plan/chat) level so the planner picks the
  relevant research and a worker picks the most relevant note.
- **The memory-read is DOMAIN-AGNOSTIC (as4 — d241).** The per-node binding ctx
  (`_node_run_ctx`) supplies a self-selecting node BOTH the prior gather **sources** (so
  `load_source` binds) AND the prior gather **notes** (so `read_notes` — the cheap leg — binds),
  collected **source-agnostically** from the node's upstream gather artifacts under the web key
  (`article_notes`/`fetched`) OR a generic key (`notes`/`records`). So a non-web complex-memory
  type (codebase, vector-db) reaches the SAME `read_notes`→`load_source` interface with no loop
  change. The **linear/chat worker is NOT excluded** (d241): it self-selects `research_read` and
  answers a follow-up FROM prior research, still applying its spec (the d206 case = linear
  worker + spec + memory-lookup, **not** a fixed-reply short-circuit).
  - *Backlog (d241, NOT built here):* a **reasoning-PICKED short-circuit SHAPE** for a trivial
    follow-up is a later item the **planner selects via `get_shapes`** (never a flag / always-on
    branch), to be added only **after** the d206 6-test gate. A same-session follow-up today is
    grounded by the bounded `_session_readback` push (a22); migrating that push to a pure
    self-select **pull** (now that the linear worker is a tool-loop) is a deliberate later step,
    not a coherence/shortcut hack.

### Each node's context window (d263 — pinned-head + SWA-tail re-injection REMOVED)
```
┌──────────────────────────────────────────────┐
│ system: identity + shaping spec (+ catalog)   │  ← Context(system=…), composed ONCE
├──────────────────────────────────────────────┤
│ convo[0]: overall GOAL + this node's task     │  ← sent ONCE, kept across compaction
│           + upstream findings (_compose_task) │
├──────────────────────────────────────────────┤
│ get_bundles LOAD obs: the bundle's DOCTRINE   │  ← delivered ONCE, in-band, on self-select
├──────────────────────────────────────────────┤
│ older MIDDLE turns → AUTO-COMPACTED (summary) │  ← token-budgeted (Conversation.compact)
├──────────────────────────────────────────────┤
│ recent messages (chronological)               │  ← keep_recent kept verbatim
└──────────────────────────────────────────────┘
```
**d263 (SA-3, supersedes d200's pin-near-tail).** The earlier **pinned-head + SWA-tail
re-injection** was a *failed* mechanism: it re-pasted the user goal + bundle doctrine + the
node task as **always-in-view blocks on every turn**, but was wired into only the research
and raw-file loops; the other loops *also* re-composed and re-sent the full system + goal +
findings each call (trace: 3 consecutive calls with byte-identical 5k-token system + identical
goal/findings). Per the user verdict (**Option B**), the pin is **removed**, not extended:

- the **shaping system** (identity + spec + bundle catalog) rides `Context(system=…)` and is
  **composed once per node loop and carried** — not re-composed/re-grown per turn;
- the **overall goal + task** ride the loop's **first turn** (`convo[0]`, `_compose_task`)
  **once**, and `_node_history` keeps `convo[0]` verbatim so the goal survives compaction;
- a bundle's **doctrine** (its how-to) rides the **`get_bundles` LOAD observation once**, in-band
  when the node self-selects it (this is what `GET_BUNDLES` already promised — "you get back its
  doctrine"), carried forward by the convo window — never re-pasted per turn;
- what remains is the **simple middle-turn compaction** (`Conversation.compact`, the KEPT
  subsystem): once the window crosses `num_ctx`, the **middle** turns fold into an offline
  deterministic summary while `convo[0]` and the most-recent `keep_recent` turns stay verbatim.
  This **must not regress** the long-chat auto-compaction the raised `num_ctx` relies on.

The retired `PINNED_HEADER` / `SWA_TAIL_HEADER` constants, the `pinned`/`swa_tail` params on
`Conversation`, and the `bounded_window` seam are **deleted** from `llm_framework/context.py`;
`runtime._pinned_head` / `_swa_tail` are deleted and `_node_history` is the compact-only
windowing described above.

---

## 6. Model & message-role handling (Gemma E4B)

- Model: `gemma4-e4b-candidate-ctx32k` (Gemma 3n E4B-class, SWA), **native Ollama
  `:11434` only**, num_ctx 32768.
- **Message roles:** this model's chat template is `{{ .Prompt }}` — it does **not**
  honor `role:tool`. Empirically it **fabricates** when a tool result is fed as
  `role:tool` and **grounds** when fed as `role:user`. Therefore, in the **research
  gather loop**, tool **observations the model must act on go `role:user`**; genuine
  instructions/nudges/finalize also stay `role:user` (d199, narrowed by d202). Do not
  blanket-apply `role:tool` to this model.
- **Transport-level role normalization (d262):** the runtime still feeds many
  observations back as `role:tool` (plan acks, file-write confirmations, reviewer
  file slices, self-select acks, builder observations) across ~13 call sites
  (`runtime.py`, `research_tree.py`, `incremental.py`) — so the d199 fix above
  reached only the gather loop and the model was **blind** to those other
  observations on every run. The fix is a single chokepoint, **not** 13 per-site
  edits: `OllamaTransport.chat` runs `_normalize_tool_roles`, which rewrites **any**
  inbound `role:tool` turn to `role:user` **before** dispatch, on **both** wire
  paths (`_chat_openai` / `_chat_native` — each copies the message list verbatim
  and never inspects roles). This is **content-preserving** (only the label
  changes) and **layered**: the runtime's in-memory history keeps `role:tool` (so
  its bookkeeping/tests are unchanged), and only the outgoing wire copy is
  normalized — fixing all sites at once and unable to regress at a new 14th site.
  Verified live on E4B `:11434`: a fact placed **only** in a `role:tool` turn is
  invisible when sent raw (model: "I need the lookup result first") and **seen**
  once normalized (model returns the fact). The call-sites are left as-is.

---

## 7. Acceptance gate (d206) — neuron-run, live, no flag-fabrication

Diversity **is** the fabrication detector. **Phase A (harden first):** (1) US-Iran
deep-research report; (2) haiku → quick linear exit; (4) US-Iran **follow-up** answered
from research memory. **Only when all three pass → Phase B:** (3) pirate-history
**multi-page SPA** in pirate tone; (5) research → a Claude **skill** in markdown
(neuron-defined + judged); (6) Java hello-world file. The neuron runs these **live** and
**read-verifies the served evidence** (traces + artifacts), never on faith.

---

## 8. Standing constraints

- **No fabricated/flag/hardcoded behavior** (d14/d60/d65); **no "ceiling" framing** —
  every gap is a fixable lever: MEMORY, PROMPTS/DESCRIPTIONS, or TOOL OUTPUTS (d186).
- **Native Ollama `:11434` only** — never `:11435`/foreign PIDs/models.
- **`send_mail`** fires only when the user explicitly asks; recipient self-locked (d12).
- **UI is hard-held** until the dedicated UI step (s17): icons + re-theme + prove one
  deep-research + one simple run flow through the UI (d98).
- **Preserve `var/chat_app/{chat.db, memory.db, workspace}`** — extend / key by session,
  never clobber.
- **Git is neuron-owned**; show commit scope before any push; nothing pushed yet (d172).

---

## 9. Decision index (durable text in the recipe)

d184 canonical research loop · d185 notes architecture · d186 no-ceiling ·
d190 OO tool-bundles · d191 research-bundle template-grow flavor · d192 end-to-end
orchestration + memory model · d194 shape+spec per node · d199/d202 role-handling ·
d200 SWA tail *(superseded by d263)* · d206 6-test acceptance gate ·
d210 compaction/pin requirements *(pin removed by d263)* ·
d263 pinned-rebroadcast removal (compose system once, goal once, doctrine via load obs,
keep simple compaction) ·
d211 emergent write / no-fab · d212 bundle = tool wrapper (not a role) · d213 role =
node type · d214 iterative planner loop + reviewer = default last step · d215
synthesizer = terminal (after all plans, not in-plan) · d216 emergent sectioning +
tool-calling-reader writer (retire `_inject_section_notes`/`_ensure_section_body_dag`) ·
d218 coherence = writer doctrine + reviewer reasoning, NO deterministic HTML surgery ·
d219 migrate the pre-existing d173/d174 passes deliberately (keep until each is proven
redundant; measure pre/post-surgery coherence) · d233 read on a cost hierarchy
(`read_notes` cheap gist, then `load_source` verbatim) · d234 nearest-N load_source tune
(top_n=2 / per_call_cap=3000 to fit the E4B window) · d235 definition-text + read-tool
cohesion: `read_notes` tool + cost hierarchy in both descriptions; 'note' everywhere = the
structured article-note artifact; research-analyst spec = output-quality only (methodology
is role/runtime — the research bundle doctrine); research (gather) vs research_read (read)
sequenced no-overlap; planning doctrine compressed. ·
d227 spec-assignment doctrine (format spec on the writer; analysis + format compose) ·
d237 emergent write / no spoon-feeding (tool-calling pullers; reviewer reports data
complexity, planner decides one-pass-vs-sectioned) · d239 ONE generic engine, real stages,
no web fork · d240 generic-spine mandates (NO cheap fix — definition layer only; CONSULT
before any BIG/non-obvious fork; ceiling = non-deciding safety net, model's `stop_research`
decides) · d241 DOMAIN-AGNOSTIC complex-memory read (source-agnostic `read_notes`/`load_source`;
linear worker reaches it + applies spec; reasoning-picked short-circuit shape = backlog) ·
d242 TRUE self-select (every in-plan node starts tool-less) · d244/d245 full unrestricted
self-select (no per-role tool subset; reviewer review+fix-inline = role/spec) · d246
SPEC-VARIANT selection (`html-writer` one-pass vs `section-html-writer` sectioned, planner
selects by data complexity; output-format floor respects the chosen variant; NO procedural
sectioning branch) — **ashw authored the `section-html-writer` seed spec + selection-lever
description, GROUNDED on the measured live-E4B sectioned-HTML failure trace, added to
`CURATED_SPECS`; base `html-writer` kept one-pass (light validity hardening only)** · d247 shapes = discipline+doctrine, planner/grower authors topology
(cyclic-shape migration deferred) · **as4 — the grow loop is SOURCE-AGNOSTIC** (grower owns no
web vocabulary; gather nodes are `ROLE_RESEARCHER` + tool-less self-select; recognize/ingest
any source; fold the tool-less fallback seed).

**s16/SA-6 — generic research methodology (PART 1; PART 2 write-method re-scoping)** · d256/d258 the research
METHODOLOGY moves to the DEFINITION LAYER as a SELECTABLE spec (now that the researcher ROLE is
being retired, SA-7): a DOMAIN-AGNOSTIC **`research-methodology`** CORE seed spec (decompose →
self-select your gather bundle → gather + read → note as claim+source+gap → cross-verify →
deepen on gaps → prune bad leads → question completeness → stop → write from notes; NEVER names
'web') + a named **`web-research`** VARIANT = the CORE method + a thin web-pairing note (siblings
differ ONLY by the paired gather bundle, so codebase-research / vectordb-research are later
siblings of the SAME method). Both seeded in `specialization/seed.py` `CANONICAL_RULESETS` and
advertised via `CURATED_SPECS`; the selection-lever DESCRIPTIONS steer the planner to
`web-research` for a live-web brief and to the CORE for a non-web one. The reasoned doctrine
(question-completeness, cross-verify concern areas, expand the open/curiosity areas, prune bad
leads) also SHARPENS the research-bundle doctrine (`bundles/research.py`). Self-select framing in
the spec = GOOD PRACTICE only (d272/d273: `get_bundles` + tool-calling WORK — no reliability
overhaul). · **PART 2 (section-html-writer write method) is being RE-SCOPED** (d266→d278→d281→
d283/d284/d285): the faithful realization must be fully spec-body-driven + AGENT-IN-THE-LOOP (the
skeleton worker plans sections + a k/v handoff from the research memory; the planner loop authors
the fill workers dynamically) with a universal finalize-summary context substrate + a memory
singleton — pending the re-planned breakdown; NOT yet implemented.
