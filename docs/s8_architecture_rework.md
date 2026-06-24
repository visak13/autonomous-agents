# s8 — Architecture Rework: Reasoning-Driven, eda-base3-Style Plan→Dispatch (d39)

**Status:** Stage-A design probe (READ-ONLY). No source modified. This document is
the deliverable. It diagnoses the CURRENT ReactiveAgents Gemma pipeline, distills the
eda-base3 reference patterns, and proposes the d39 three-role architecture plus an
ordered Phase-2 implementation breakdown for the planner to author after approval.

**Scope grounding (load-bearing decisions):** the goal (d38) is to make the Gemma
flow *seamless and reasoning-driven*, modeled on eda-base3's own neuron + agentic-plan
prompt/flow architecture — three LLM roles in three scoped context windows, the planner
building the DAG by **calling exposed tools iteratively** (not one-shot schema JSON),
workers fed **dependency-scoped** context they cannot discover, a synthesizer in the
chat context. The fix is a dedicated prompt/flow/context-feeding rework, NOT narrow
patches. Behavior = pure reasoning, no structural override flags (d14); the only hard
structure that stays is the send_mail self-only recipient lock (d12). Target model =
Gemma-4 **E4B** text-only Q4 @ num_ctx 32768 / keep_alive -1 / num_predict 4096
(temp 0 on deterministic/write nodes) (d35); the gemma specialist is being re-trained
(d40) so Phase-2 workers inherit the measured ruleset.

**Environment:** base dir `C:\Projects\ReactiveAgents`; model on Ollama `:11434`
(never `:11435`). Live chat entrypoint = `chat_app/agentic.py:run_agentic`
(wired at `chat_app/app.py:724`).

> **Important current-state note.** The earlier `round3-port-blueprint.md` RCs (stub
> `_demo_dag`, hardcoded md+html two-report, `ensure_grounded` plan-rewrite) are
> ALREADY RETIRED: `run_agentic` now drives a real shape selection + the
> `IncrementalPlanner` (node-by-node authoring) on the live model, with search-then-read
> and inter-node context feeding. This doc documents the pipeline AS IT IS TODAY, not
> the round-3 starting point. Several d39 targets are *partially* realized already; the
> rework is to complete them coherently and replace the remaining schema-constrained
> authoring + hardcoded role/flag control flow with tool-driven, reasoning-driven flow.

---

## (1) PROMPT CATALOG — every driving prompt (file:line)

All structured calls run on the native Ollama `/api/chat` path. **`think` is a
per-call option passed by the call site** — it is NOT hard-forced by the transport.
The transport (`llm_framework/transport.py:538-570`) simply passes the caller's
`think` through to the top-level `/api/chat` `think` field. (The `think=false`
commentary at `transport.py:499-503,538-545` is STALE from the prior s8/b1 strategy;
the call sites below now pass `think=True` per the s1/b1 reasoning rollout, because
gemma4 returns CoT in a *separate* `message.thinking` field so content is not starved
as long as `num_predict` is high — see d1/d6.)

### 1.0 Universal identity (rides on every call)
- **`llm_framework/transport.py:134-141`** — `AGENT_IDENTITY` (~90 tok): *"You are a
  capable, autonomous agent. Reason about the user's real goal, then ACT — prefer doing
  the task well with sensible defaults over asking. Ground every answer… never invent
  facts/sources/numbers. Treat the prior conversation as memory… When asked for JSON,
  your visible reply is ONLY that JSON… Be concise and direct."* Injected once at the
  transport seam (`_inject_identity`, byte-identical to `agent_runtime.identity.with_identity`
  so the startswith idempotency guard prevents double-injection). Folds in the two
  universal output rules. This is the d15 universal identity — keep ~100-150 tok.

### 1.1 Planner / DAG-emission (one-shot)
- **`agent_runtime/factory.py:298-318`** — `FACTORY_DESCRIPTION` (the canonical planner
  template; body-free per d10): *"You are an autonomous planner. Decompose the GOAL into
  a DAG of logical steps — invent the steps yourself… each node has id/task/depends_on…
  Specialization (SELECTION GUIDELINES)… MATCH on the WORK a step PRODUCES… Tool: set
  'tool' + 'tool_args'…"*
- **`agent_runtime/factory.py:442-463`** — `planner_prompt(goal)` assembles system
  (FACTORY_DESCRIPTION + node schema `{"rationale","nodes":[…]}` + registered spec
  names + tool names) + user `"GOAL: {goal}\n\nReturn ONLY the JSON plan."`
- Driver: `agent_runtime/planner.py:Planner.plan()` (~L228-293). **One-shot,
  schema-constrained JSON DAG** — this is the d34 edge-drop / "linear plus modular
  parallel" collapse risk the d39 design replaces with tool-driven authoring.

### 1.2 Incremental planner / per-node authoring (THE LIVE chat authoring path)
- **`agent_runtime/incremental.py:216-282`** — `_system()`: body-free factory context +
  selected SHAPE description (guides depends_on edges: parallel vs chained) +
  user-requested specs (hard-bind instruction) + per-node schema
  (`task/spec/specs/needs_spec/tool/depends_on/more`) + authoring guidance.
- **`agent_runtime/incremental.py:284-327`** — `_user()` (per node) and
  **`:367-396`** `_finalize_user()` (final node): GOAL + already-authored steps +
  decision procedure (author next uncovered gather step, or the final combine step when
  all covered; gather steps are SOURCES `depends_on=[]`; bind tools; send_mail only if
  the goal explicitly asks).
- Defaults: **`incremental.py:59,67,78`** — `DEFAULT_MAX_NODES=12`,
  `DEFAULT_NODE_NUM_PREDICT=4096`, `DEFAULT_RESEARCH_TOOLS=("web_search","web_fetch")`.
- Driver: `IncrementalPlanner.plan()` → `_author_one()`. **This already authors
  node-by-node** (the d39 spirit) — but each node is still a *schema-constrained JSON
  call*, not the model *calling an exposed plan-building tool*. Closing that gap (and
  the F2/F5 finalization stamps below) is the core of the d39 planner rework.

### 1.3 Shape selector
- **`agent_runtime/shape_selector.py:267-359`** — `_system_prompt()`: select ONE shape
  from the disk-harvested catalog (name + one-line description) + `escalate`; choose by
  WORK not phrasing; reports intent signals `search_allowed`, `requested_specs`,
  `unmet_specs` (missing-specialist trigger), `wants_file`. Schema built by
  `build_selection_schema()` (`:128-216`). User: `"GOAL: {goal}\n\nReturn ONLY the JSON
  shape selection."` Driver: `ShapeSelector.select()`.

### 1.4 Ambiguity / clarify gate
- **`agent_runtime/planner.py:314-327`** — `assess_ambiguity()` system: *"…decide if you
  must ask ONE clarifying question. Ask ONLY when the request is to SCHEDULE a recurring
  or future task AND a load-bearing scheduling detail is missing… Any normal one-shot
  request — even a broad one like 'write a report on X' — is NOT ambiguous: proceed with
  sensible defaults, never interrogate…"* Emits `{needs_clarification, question,
  rationale}` (schema `:88-110`). **Fail-open** (`:341-348`: any malformed decision →
  not ambiguous). This already encodes the d9 clarify-only-for-scheduling policy via
  PROMPT (no logic gate) — d14-compliant.

### 1.5 Heal decision + replan-subgraph (failure recovery)
- **`agent_runtime/planner.py:387-401`** — `heal_decision()`: enum
  `{retry|pivot|abort|extend}` + rationale (schema `:50-61`). User carries failed task,
  error, attempt/max, already-completed.
- **`agent_runtime/factory.py:509-528`** — `replan_prompt()`: minimal corrective DAG for
  ONE failed node (same NODE_SCHEMA, body-free). Driver: `Planner.replan_subgraph()`.

### 1.6 Node / worker execution — role framings + output-shaping
- **`agent_runtime/roles.py:65-110`** — `role_framing()` (pure data, appended AFTER the
  spec body, BEFORE the task) for `research / critic / synthesis / verify / reviewer /
  worker`, each paired with a per-role OUTPUT SCHEMA `ROLE_SCHEMAS` (`:125-185`):
  research→`{findings,sources,open_questions}`; critic→`{gaps,weak_claims,
  follow_up_queries,verdict:enum[converged|needs_more]}`; synthesis/verify/reviewer→
  `{verdict:enum[pass|concerns|fail],findings,fixed_inline}`; worker→`{output}`.
  num_predict floors: **`roles.py:217-219`** `ROLE_DEFAULT_NUM_PREDICT=1200`,
  `JUDGMENT_NUM_PREDICT=1600`, `JUDGMENT_REPAIR_BUMP=600`.
- **`agent_runtime/runtime.py:156-162`** — `_SHAPING_FRAMING`: *"The text below is an
  OUTPUT-SHAPING RULESET, not the task. DO the task in the user message… then shape the
  FORM… Never describe the ruleset…"* (fixes the round-1 "Iran→markdown-how-to" bug).
  Prepended to a node's spec body; a bare (spec-less) node gets NO system prompt.
- **`agent_runtime/runtime.py:178`** — `_RULESET_LAYER_HEADER` `"===== Ruleset {i}/{n}:
  {name} ====="` for N-spec composition (single-spec node = no header, byte-identical).
- **`agent_runtime/runtime.py:197-203`** — `_REVIEWER_FRAMING` (coder-as-reviewer inline
  fix when the verify gate rejects).
- **`agent_runtime/runtime.py:189-192`** — `_PRIOR_CONVERSATION_HEADER` (threads bounded
  prior turns into the produce-step USER turn).

### 1.7 Tool-argument emission
- **`agent_runtime/toolargs.py:485-500`** — `SchemaToolArgEmitter` system/user: emit ONLY
  the JSON args for a tool, matching the schema (`TOOL_ARG_SCHEMAS` `:42-124`). Falls back
  to deterministic grounding from upstream data, then a deterministic fallback fn.
  **`toolargs.py:515-522`** — `cron_add` args are grounded DETERMINISTICALLY (no LLM):
  the fired job's prompt is extracted from the cron-step task.

### 1.8 Authoring prompts (shape + spec — the s6-folded free-flow targets)
- **`agent_runtime/shape_author.py:162-190`** — NL→shape JSON: emit one shape
  `{name,description,execution,max_iter,round_roles,final_roles}`; `execution` ∈
  `{sequential,concurrent,deep-research}`; roles only for deep-research; the
  `description` is **selection-critical** (discriminative one-liner). Schema
  `build_shape_schema()` (`:88-159`). Driver: `ShapeAuthor.author()`.
- **`specialization/compiler.py:95-119`** — `build_condense_messages()`: condense
  research+definition into a tight markdown ruleset body a sub-agent loads as its WHOLE
  grounding. Offline path = deterministic distillation (`offline_condense_body()`); live
  path wires real transport.
- Re-editable spec chat lives in **`chat_app/spec_chat.py`** (`build_spec_chat_service`/
  `register_spec_chat_routes`).

---

## (2) FLOW-FLAG CATALOG — hardcoded branches steering the LLM path (file:line)

These are the hardcoded constants / branches that currently STEER control flow instead
of the model reasoning. d38/d41 require replacing the *flow-shaping* ones incrementally
with Gemma reasoning (the per-call tuning constants in §1 stay — they are the d36
measured invocation baseline, not control-flow flags).

**Control-flow / routing flags (candidates for reasoning-driven replacement):**
- **`runtime.py:281,299`** `read_search_max_fetch:int=0` — gates SEARCH-THEN-READ. Chat
  sets it to **3** when web allowed (`agentic.py:645`, `agentic.py:1074`); 0 = OFF
  elsewhere.
- **`runtime.py:693-695`** — **hardcoded ROLE gate**: `if node.role==ROLE_RESEARCH and
  hook and read_search_max_fetch>0:` bypass the node's own tool and run `_research_read()`
  (search+fetch) deterministically.
- **`runtime.py:744-750`** — **hardcoded TOOL gate**: a non-research node whose tool is
  `web_search` auto-follows-through to `_read_search_results()` when fetch enabled.
- **`runtime.py:758-775`** — **hardcoded ROLE-EXECUTION gate**: `if node.role:` run
  `_run_role()` (schema-constrained + verdict-repair); else a bare Chain. This is the
  hardcoded role→behavior switch the d39 role reasoning should subsume.
- **`runtime.py:442 / 455`** — **fetched-content render gate**: `if tool_value.get(
  "fetched")` render full source blocks @ `fetched_char_budget`; else compact 1200-char
  cap.
- **`runtime.py:849-859`** — **verdict-repair loop** (judgment roles only): retry up to
  `max_verdict_repairs` with bumped num_predict.
- **`shape_author.py` / `shape_selector.py`** — the shape *catalog* and `execution` enum
  (`sequential|concurrent|deep-research`) is a fixed taxonomy; the
  "linear-plus-modular-parallel" compositional intent collapses to a flat shape (s6 bug)
  — a free-flow authoring/edit target.

**Per-call tuning constants (the d36 invocation baseline — KEEP, document, do NOT treat
as control-flow flags):**
- **`runtime.py:284,306`** `fetched_char_budget=2000` (min 400) — per fetched article.
- **`runtime.py:285,307-314`** `upstream_input_char_budget=4000` (min 200) — **NOTE: the
  d17 legacy 800-char clip is ALREADY RAISED to 4000 in current code** (the comment at
  `:307-310` documents the o4 fix). This is the d17/d18 inter-node fix, already landed;
  finalize against num_ctx in Phase 2.
- **`runtime.py:473-477`** `_NON_ARTICLE_EXT` — binary-URL skip list (fetch hygiene).
- **`incremental.py:59,67,78`** `DEFAULT_MAX_NODES=12`, `DEFAULT_NODE_NUM_PREDICT=4096`,
  research-tools whitelist.
- **Structured-call opts (think/temp/num_predict/format):** `planner.py:71-77`
  (`_HEAL_OPTS`), `planner.py:119-125` (`_AMBIGUITY_OPTS`), `runtime.py:604-609`
  (`_emit_research_queries`), `toolargs.py:460,474` (`think=True` default), `roles.py`
  floors — all `think=True, temperature=0, num_predict=4096` (s1/b1). These ride the
  transport pass-through (§1 intro); keep as the d36 per-task baseline, re-validated on
  E4B (d31/d35).

---

## (3) CONTEXT-FLOW MAP — node→node TODAY

The downstream node's user turn is assembled in **`runtime.py:_compose_task(inputs,
tool_value)` (~L391-431)**:

1. **Prior conversation** (`:400-404`) — bounded prior turns under
   `_PRIOR_CONVERSATION_HEADER`, when set.
2. **This node's task** (`:405`) — `node.task` (the planner-paraphrased description).
3. **Upstream PRODUCED PROSE** (`:406-409`) — each DIRECT dependency's produced output
   string, clipped to `upstream_input_char_budget` (**4000** chars, min 200):
   `parts.append(f"- {k}: {str(v)[:budget]}")`.
4. **Upstream TOOL VALUES** (`:421-428`) — each DIRECT dependency's raw tool result
   (search results / fetched sources / research findings), rendered via
   `_render_tool_value()`. Collected and passed in by the runtime at **`:1271-1276`**
   (`upstream_tool_values={dep.id: cache[dep.id].tool_value …}`).
5. **This node's own tool output** (`:429-430`).

**`_render_tool_value()` (`:433-455`):** if the value is a Mapping with a `"fetched"`
key → render a `FETCHED SOURCE CONTENT` section, each article's markdown clipped to
`fetched_char_budget` (**2000** chars) under READ_NOT_DESCRIBE framing (`:442-454`);
otherwise compact 1200-char cap (`:455`).

**Key facts:**
- **The overall GOAL does NOT flow to runtime nodes.** It lives only in the planner's
  prompts and is *paraphrased* into each node's `task` at author time
  (`planner.py`/`incremental.py`). A worker node never sees the verbatim user goal —
  only its own task string + direct-dep outputs. **This is a d39 gap:** workers should
  be fed the overall goal explicitly (they cannot discover it).
- **Fetched sources DO reach a downstream synthesize/write node** — via
  `upstream_tool_values` (the o4 fix at `:414-420`). Before it, a writer node (no tool
  of its own) saw only clipped prose and never the sources.
- **Direct-deps only** — context is NOT transitive (multi-hop accumulation is out of
  scope per d17). A node sees its immediate parents' outputs, not grandparents'.
- Query-rewriting: healthy / search-path only; the request reaches the planner verbatim
  (d17) — no change needed.

**Net:** the context-feeding mechanism exists and was hardened (4000/2000 budgets +
fetched-source folding), but it is (a) **goal-blind at the node** and (b) **assembled by
clipping** rather than by a deliberate per-node context-assembly that knows what each
role needs. d39 promotes this to a real **dependency-scoped context-assembly
architecture**.

---

## (4) EDA-BASE3 BLUEPRINT — the reference patterns to mirror

Studied: this repo's `/neuron` and `/agentic-plan` skills (`.claude/commands/`), the
planner-phase guides (`planner-phase-author`, `planner-phase-drive`), and
`architecture-vocabulary`. The patterns Gemma should mirror:

**B1. Separate shell + scoped context per role.** neuron / planner / worker each run in
their OWN shell with a context window scoped to their job: neuron = the recipe map +
routing only (it is a ROUTER, "not the brain" — it does not comprehend/code/verify);
planner = ONE step, loading ONE phase guide at a time (`ground → author → drive`) so it
never reads the whole job at once (the skill explicitly warns that loading everything
"is what makes a planner burn ~30k tokens and hallucinate"); worker = ONE action,
grounded by its brief + the compiled specialist doc.

**B2. Seed-plan-then-iteratively-add-steps via tool calls.** The planner does NOT hand-
author a monolithic plan object. It calls `create_plan(recipe_id, step_id, shape, goal)`
to SEED, then `add_action(plan_id, action_id, description, depends_on=[…],
executor_mode, acceptance_kind, verify={…}, specialization, concerns=[…])` — **one tool
call per action**. The guide is explicit: *"prefer `create_plan` + `add_action` — they
never have schema fights, the way `start_recipe`/`add_step` don't"*, and *"hand-authoring
a full plan object"* is an anti-pattern. This is the exact pattern d39 ports: the
planner BUILDS the DAG by calling exposed tools iteratively, which sidesteps one-shot
schema-constrained edge-drop (d34).

**B3. High-quality node DESCRIPTIONS drive workers.** `description` = WHAT to do;
`specialization` = WHO does it (the field, never a dispatch instruction in the
description). The worker's whole grounding is its description + the compiled spec doc +
stamped `injected_context` (`load_bearing_decisions` + `banned_options`). A strong,
self-contained description is the lever — exactly what d38 calls "the prompts are too
vague."

**B4. The CENTRAL GAP — discovery.** eda-base3 workers get a well-defined task AND can
DISCOVER their own inputs/outputs via Claude Code (read files, grep, explore). **Gemma
has NO discovery.** So the layer-level and node-to-node context that a Claude worker
would discover must instead be explicitly CONSTRUCTED and FED to each Gemma node. This
is why d17 is promoted from a clip-size tweak to a full context-feeding *architecture*:
every node must receive the upstream outputs + scoped context it needs, assembled for it
(overall goal + relevant upstream outputs when dependent).

**B5. Phased context-building.** Both neuron (phases a-e) and planner (ground/author/
drive) load one guide per phase on demand — gradual context building, not a single
overload. The Gemma flow should similarly build context in stages rather than stuff one
mega-prompt.

**B6. Done is gated, never self-declared (verify gate + reviewer leg).** Every checkable
action carries a real `acceptance.verify` (`file_exists` / `file_min_bytes` /
`glob_matches` / `command`); a failing gate PARKS the action in `verify` (visible,
re-checkable), never a false-done. Every plan includes a review/verify step; CODE work
gets a dedicated reviewer leg (not the builder self-blessing). ReactiveAgents already has
the analogue (`in_progress→verifiable→done` + inline reviewer-fix at `runtime.py:_verify_
and_finalize` / `SubAgent.review_and_fix`); keep mirroring it.

**B7. Specialization = compiled doc loaded as grounding.** A specialist action loads the
specialist's COMPILED doc as its whole grounding (no chat fork). ReactiveAgents'
`compose_specialist_docs` + per-node spec injection is the faithful port; the per-node
spec biases behavior (e.g. an HTML spec biases toward writing the HTML file).

---

## (5) PROPOSED DESIGN — d39 three-role architecture

Three LLM roles in three scoped context windows, reasoning-driven, app runnable at every
incremental step. Validated on E4B via the s7 markdown tracing.

### Role 1 — PLANNER (own context window)
**Reasons, then BUILDS the plan by calling exposed in-app plan-building tools
iteratively** — the eda-base3 `create_plan` + `add_step` pattern (B2), replacing the
one-shot schema-constrained JSON DAG (kills d34 edge-drop). Sequence:
1. **Reason about SHAPE fit** (shape-compatibility) — keep the shape reasoning but make
   it tool-recorded, and honor compositional/multi-pattern intent (fix the s6
   "linear plus modular parallel" → flat-DAG collapse).
2. **Select the INPUT specialist + PROCESSING specialist + OUTPUT specialist** by
   reasoning over their descriptions (per `SELECTION_GUIDELINES.md`).
3. **Piece them into a DAG** by calling tools — e.g. `seed_plan(shape, goal)`,
   `add_step(id, task, depends_on, role, spec/specs/needs_spec, tool)`, repeated, then
   `finalize_plan()`. Per node it records TASK + SPECIALIZATION. Implementation: expose
   these as native tool-calls the model issues (mirroring how Claude calls
   `create_plan`/`add_action`); the existing `IncrementalPlanner._author_one` becomes a
   tool-call loop rather than a per-node schema JSON emission. The plan structure is
   built by accumulation, so a dangling/dropped edge cannot silently corrupt the whole
   DAG (d7 dangling-edge + d28 missing-edge guarantees enforced at `add_step`/`finalize`,
   builder-owned).
- **d28 edge guarantee (MANDATORY on E4B):** validate at `finalize_plan` that a terminal
  write/synthesize node DEPENDS on the upstream research/gather node(s) for the same
  goal; auto-add the edge or re-author if missing. E4B authors flat zero-edge DAGs every
  run (d34), so this is what makes its report output substantive — it matters MORE than
  the model choice.
- **Crisp prompts (d11):** the planner system prompt is rewritten concise + anti-
  hallucination, grounded in how strong planners direct nodes via high-quality
  DESCRIPTIONS (B3) and the eda-base3 author phase guide.

### Role 2 — WORKER (own context window)
Executes ONE node independently. **Fed dependency-scoped context it cannot discover (B4):**
- The **OVERALL GOAL** (verbatim) — closes the §3 goal-blindness gap (today the node
  only sees its paraphrased task).
- The **PREVIOUS RELEVANT task output WHEN the node depends on it** — dependency-scoped
  (independent nodes get none). This is d17 promoted from a clip to a real per-node
  context-assembly: `_compose_task` becomes a deliberate assembler that knows what each
  role needs, with budgets finalized against num_ctx 32768 (the 4000/2000 budgets are
  the current floor).
- Its **compiled specialist doc** (B7) + `injected_context` (load-bearing decisions /
  banned options), wrapped by the `_SHAPING_FRAMING`.
- Node prompts rewritten crisp (d11); the universal identity (d15) rides every call.

### Role 3 — SYNTHESIZER (runs in the CHAT context window)
Terminal output stage. Emits one of: **FILE** (`file_write` MCP, sandboxed), **EMAIL**
(`send_mail` MCP — recipient self-only lock intact, d12), or **SSE** (fallback). **The
PLANNER decides which synthesizer fires by DESCRIBING it in that node's task
description** (d14 pure-reasoning — no `wants_email` strip, no clarify logic-gate). The
self-only recipient lock is the ONLY hard structure that stays (a small model slipping
to email-the-user-themselves is harmless under the lock — the accepted price of
free-flow).

### Incremental flag removal (app runnable each step)
Replace each flow-shaping flag (§2) only as its reasoning-driven replacement proves out
via the s7 markdown tracing — keep the LIVE app runnable at every step; the user
validates live in s9. Priority order: (a) planner one-shot→tool-driven authoring +
edge guarantees; (b) worker goal + dependency-scoped context assembly; (c) synthesizer
role decided by description; (d) fold the hardcoded role-execution / search-then-read
gates into role reasoning last (they are the most behavior-load-bearing). Big-bang is
too risky.

### Shape & spec FREE-FLOWING ITERATIVE authoring (s6 folded in)
- **Shapes:** add conversational EDIT/REFINE of an existing shape (today only
  describe→CREATE exists, no edit) and make the shape author honor compositional /
  multi-pattern intent (fix the flat-sequential collapse). UI edit affordance + backend
  refine-on-existing.
- **Specs:** ensure the multi-turn spec author ITERATES on the current spec (builds on
  it), not restart; re-openable spec chat persists the edit, effective next run.
- Both: free-flowing refine-on-existing via chat. Selection quality is driven by strong,
  selection-grade descriptions (per `SELECTION_GUIDELINES.md` §4) — and the AUTHORING
  prompts must explicitly require the model to produce selection-effective descriptions.

### Invocation baseline (d36) + model (d35) + specialist (d40)
- Per-task invocation from the sweep: deterministic/structured/write nodes → temp 0;
  creative/prose → higher-temp band; num_ctx 32768, keep_alive -1, num_predict 4096
  (raise to 8192 for long reports). Baked-into-Modelfile vs per-call HTTP override split
  per d36.
- Model = Gemma-4 E4B text-only Q4 (only fit-passing upgrade; @32k 0% offload; native
  think). Re-validate the transport think/JSON interceptor + thinking-capture for E4B
  (d25/d31) and re-tune E4B's over-ask-on-coding prompt behavior (d33).
- Phase-2 workers run under the re-trained `spec-local-gemma-ollama-llm-engineer` (d40)
  so they apply measured settings, not guesswork.

---

## (6) PHASE-2 ACTION BREAKDOWN (ordered; for the planner to author after approval)

Each: WHAT it changes · which flag/prompt it replaces · dependency · proof (s7 markdown
tracing) · reviewer leg. Not pre-fragmented beyond this clear breakdown.

1. **Expose plan-building tools + planner→tool-driven authoring.**
   *Changes:* add in-app `seed_plan` / `add_step` / `set_node_spec` / `finalize_plan`
   tools the planner LLM calls; convert `IncrementalPlanner._author_one`
   (`incremental.py`) from per-node schema JSON to a tool-call loop. *Replaces:* one-shot
   `factory.planner_prompt` JSON DAG (§1.1) + schema-constrained per-node authoring.
   *Dep:* none (first). *Proof:* trace shows the planner issuing tool calls; a
   "linear-plus-modular-parallel" request produces a compositional (not flat) DAG.
   *Reviewer:* planner/flow reviewer leg.
2. **DAG edge guarantees at finalize.**
   *Changes:* enforce d7 (no dangling/phantom edge) + d28 (terminal write/synthesize node
   MUST depend on research/gather node; auto-add or re-author). *Replaces:* implicit
   acyclic-only validation; the E4B flat zero-edge DAG defect (d34). *Dep:* #1. *Proof:*
   trace of an E4B run shows write→research edge present; report is substantive.
   *Reviewer:* same.
3. **Worker context-assembly architecture (goal + dependency-scoped upstream).**
   *Changes:* feed the verbatim overall goal to each worker; make `_compose_task`
   (`runtime.py:391-431`) a deliberate per-node assembler; finalize 4000/2000 budgets
   against num_ctx 32768. *Replaces:* §3 goal-blindness + clip-only assembly (d17
   promoted to architecture). *Dep:* #1. *Proof:* trace shows a downstream writer node's
   prompt containing the goal + full upstream research/sources. *Reviewer:* runtime
   reviewer leg.
4. **Synthesizer role decided by node description (file/email/SSE).**
   *Changes:* terminal node's output channel chosen from the planner's task description,
   running in chat context; keep send_mail self-only lock. *Replaces:* any
   `wants_email`/email-trigger logic-gate (d14); hardcoded output routing. *Dep:* #1,#3.
   *Proof:* trace shows email fired ONLY when the goal asked; file written otherwise; SSE
   fallback. *Reviewer:* security reviewer (recipient lock) + flow reviewer.
5. **Crisp prompt rewrite + universal identity sanity-check.**
   *Changes:* rewrite planner / node / ambiguity / tool-select / spec+shape selection
   prompts concise + anti-hallucination (d11); confirm the ~100-150 tok identity (d15)
   rides every call and isn't doubled. *Replaces:* vague prompts (the d38 core cause).
   *Dep:* #1-#4 (rewrite around the new flow). *Proof:* trace token counts down,
   thinking blocks populated, no hallucinated facts on the US-Iran probe. *Reviewer:*
   prompt-quality reviewer leg.
6. **Shape free-flow iterative authoring + edit/refine + compositional intent.**
   *Changes:* add shape EDIT/REFINE (UI + backend), honor multi-pattern intent (fix flat
   collapse) in `shape_author.py`. *Replaces:* describe-only create; the s6 collapse.
   *Dep:* #1. *Proof:* "linear plus modular parallel" round-trips through edit and
   selects the compositional shape. *Reviewer:* shape reviewer leg.
7. **Spec free-flow iterative authoring (refine-on-existing) + re-openable spec chat.**
   *Changes:* ensure multi-turn spec author iterates on the current spec; persist edits
   effective next run (`spec_chat.py`). *Replaces:* unclear restart-vs-iterate; no
   re-open flow. *Dep:* #1. *Proof:* trace shows a refined spec body persisted + applied
   next run. *Reviewer:* spec reviewer leg.
8. **Incremental flag-removal sweep + invocation baseline on E4B.**
   *Changes:* fold the remaining hardcoded role-execution / search-then-read gates
   (`runtime.py:693-695,744-750,758-775`) into role reasoning where proven; apply d36
   per-task invocation settings on E4B; re-validate think/JSON interceptor for E4B.
   *Replaces:* §2 control-flow flags. *Dep:* #1-#5. *Proof:* app runnable at each removal;
   trace shows reasoning-driven path. *Reviewer:* runtime + gemma-specialist reviewer.
9. **UI affordances: delete buttons for specializations + shapes.**
   *Changes:* add UI delete for noise specs/shapes. *Replaces:* n/a (additive). *Dep:*
   #6,#7. *Proof:* delete removes the item; selection catalog shrinks. *Reviewer:* UI/
   frontend reviewer leg.
10. **End-to-end live acceptance (s9 gate, the real bar).**
    *Changes:* none (validation). *Proof:* the US-Iran "detailed HTML report" task,
    driven LIVE in the chat UI by the user, converges past clarification (no re-ask
    except scheduled), produces a SUBSTANTIVE sourced HTML report (headlines, timeline,
    damage figures with sources — a thin/empty file = FAIL), no unwanted email, correct
    spec/shape selection, deletes work, multi-round chat holds. *Reviewer:* full domain
    reviewer pass + user sign-off.

---

### Acceptance for this Stage-A doc
Read-only; no source modified. The Phase-2 breakdown is a clear ordered list for the
planner to author after neuron→user approval — it is NOT pre-fragmented into final
actions. Implementation is reviewer-gated and proven live (s7 tracing + s9 UI).
