# ReactiveAgents Round-3 — Design / Port Blueprint (s1 deliverable)

**Goal:** turn the current "dummy chatbot that does nothing" into the intended
free-flowing, specialization-driven agent — by **porting eda-base3's proven
mechanics (shapes / specializations / node-roles / lifecycle / self-heal) to the
local Gemma (gemma4-e2b-agent / Ollama) runtime**, on top of the EXISTING engine.

This doc is the contract the build steps (s2–s8) follow. It synthesizes:
a1 (root-cause diagnosis), a2 (eda-base3 → Gemma port mapping), a3 (repo send_mail
status), a6 (evolving-deep-agent GCP inventory). **Settled architecture decisions
d1–d7 are binding** (thin growable Pydantic tool registry on the existing
llm_framework; planner owns the DAG; native structured tool-calling: `think=false`
top-level + JSON-schema enum+required + temp 0; ddgs + Trafilatura; shapes as text
files with UI-set max-iter; SMTP+App-Password mail; NO evidence files). Banned:
adopting LangChain/LangGraph/PydanticAI/smolagents wholesale.

**Headline finding (a1):** the `agent_runtime` ENGINE is NOT the problem — it
already implements planner-owns-DAG, spec-as-SYSTEM-ruleset injection, the
`in_progress→verifiable→done` lifecycle and inline reviewer-fix, sub-graph
self-heal, and N-spec compose. **The "dummy" behavior is a WIRING failure** (the
chat path bypasses the engine and runs a canned stub) plus missing shape/tool/memory
layers. **Prefer surgical WIRING + gap-fill over greenfield rewrites.**

---

## 1. Root causes of the "dummy chatbot" (file:symbol → today vs required → blocked outcome)

| # | Location | Today | Required | Blocks |
|---|----------|-------|----------|--------|
| RC1 | `chat_app/routes.py:_execute_message_run` (else ~L439-449) + `app.py:chat` (/chat ~L630-653); DAG from `_demo_dag` | User message runs a fixed `analyze→draft_md→draft_html` stub DAG on `stub.py:FakeTransport` returning canned strings; `w.planner` is built but **never called** | message → planner selects shape → real DAG on live Gemma with specs+tools | o2, o4, o6 |
| RC2 | `chat_app/agentic.py:run_agentic` / `agentic_goal` (L75) / `ensure_grounded` (L87) | Even "live" path is a hardcoded TWO-REPORT (md+html) demo; `ensure_grounded` REWRITES the plan into research→md→html regardless of the user's actual request | planner derives shape+nodes from the actual query; no hardcoded goal/shape rewrite | o2, o3, o6 |
| RC3 | `app.py:build_wiring` (L236-351) never calls `specialization.seed.seed_canonical_rulesets`; `specialization/seed.py:CANONICAL_RULESETS` defines only `markdown-writer` (no `html-writer`) | Live agentic path requires BOTH writer specs; neither is seeded at boot → always falls back to the stub. **Direct reason the shipped app is a dummy** | seed/define specs so the real path is reachable out of the box | o2, o3, o6 |
| RC4 | `agent_runtime/factory.py:FACTORY_DESCRIPTION` + `NODE_SCHEMA` (L236-260); `planner.py:Planner.plan`; `PlanNode` (no `role`/`shape`); `PlanDAG.validate/_kahn_order` ENFORCE ACYCLIC | No SHAPE concept; planner emits a generic acyclic free-DAG; no node ROLE; a cyclic deep-research loop is structurally unrepresentable; no max-iter | planner SELECTS a shape; runtime runs deep-research `~9×{research+critic}+1×{research+synthesis+verify}`, SAME spec differentiated by ROLE, each round sees prior layers, honors UI max-iter | o2, o6; partial o5 |
| RC5 | `reactive_tools/tool_hook.py:build_default_hook` (L263) + `tools.py:register_core_tools`; `chat_app/agentic.py:OFFERED_TOOLS` (L187) | Tools are plain callables (not Pydantic-typed); web_search = raw DDG-HTML scrape (no ddgs/cache/backoff); web_fetch = stdlib HTMLParser (not Trafilatura); NO cron tools (scheduler is a service, not node-callable); planner is offered only web_search+subscriptions — file/send_mail never offered to nodes; send_mail recipient NOT hard-locked | thin growable Pydantic registry; 6 node-callable tools per d1/d4; both safety bars | o1, o6 |
| RC6 | `app.py:build_wiring` constructs `MemoryRecall` (L257) but `routes._execute_message_run`, `app.chat`, `agentic.run_agentic` **never call** `w.memory.recall(...)` or write turns back | Turns persist to `ChatStore` for display only; never fed to planner/nodes; no per-thread isolation model | conversational chat, turn N-1→N context, SQLite memory, isolated threads, survives restart | o4, o6 |
| RC7 | `chat_app/spec_chat.py` (`build_spec_chat_service`/`register_spec_chat_routes`) | Authors a NEW spec then approves→compile→register; **no flow to re-open an EXISTING spec into an editable session and persist the edit** | re-openable spec chat: load existing spec → edit → persist → effective next run | o3 |
| RC8 | `agentic.py:build_plan_schema` spec enum `[""]+names`; a `spec=""` node → `SubAgent._compose_system()→None` (`runtime.py:315-328`) = raw LLM auto-completion | No missing-specialist fallback (no notify-user + SSE stream, no define-and-resume) | planner detects no matching specialist → notify user + SSE fallback OR define-and-resume | o3, o6 (scenario 3) |
| RC9 | Lifecycle gate `_verify_and_finalize` (`runtime.py:832`) + `SubAgent.review_and_fix` (L398) are IMPLEMENTED, but default chat branches build `AgentRuntime` with `verifier=None`/`result_validator=None`; only `run_agentic` wires `_report_validator` (agentic.py:122) | The `in_progress→verifiable→done` + inline reviewer-fix gate runs ONLY on the path that never executes out of the box | wire the lifecycle gate universally onto every node path | partial o5 |

**Reuse leverage (keep, do not rebuild):** the `agent_runtime` engine (DAG exec,
scope/spec injection, lifecycle, self-heal, N-spec compose), `EventPlane`/`LambdaRegistry`,
`ChatStore`, `SpecRegistry`/loader, `llm_framework.OllamaTransport` (api=native,
`think=false`). **Net build work = (a)** route chat through Planner+live runtime
instead of stub/`_demo_dag`; **(b)** add shape selection + node `role` field + cyclic
deep-research executor + max-iter honoring; **(c)** extend the tool registry
(ddgs/Trafilatura/locked send_mail/cron-tools, Pydantic-typed) and OFFER all tools to
nodes; **(d)** wire memory into the chat→planner loop; **(e)** add spec re-open/edit +
missing-spec fallback; **(f)** seed/define specs so the real path is reachable.

---

## 2. Gemma-port mapping (eda-base3 mechanism → local-Gemma realization)

**Cross-cutting principle (a2):** everything DETERMINISTIC in eda-base3 (FSMs,
ready-action selection, ruleset assembly, doc compose, the verify gate, liveness/heal)
ports as **PURE PYTHON unchanged** — it is already model-independent. **Gemma replaces
ONLY the Claude judgment points** (shape selection, action authoring, spec resolution,
node work, heal decision), each via native structured tool-calling per d1
(`think=false` top-level, JSON schema enum+required, temp 0 — the proven 24/24 path).

### 2a. SHAPES (select + execute faithfully)
- eda-base3: `schemas/plan.py:Plan.shape` (string id) + flat `actions[]` whose structure
  is the `depends_on` DAG; `fsm/plan_fsm.py:_first_ready_action` (wave-dispatch readiness
  gate) + `plan_next_action` (deterministic FSM). Shape is the planner LLM's choice.
- **Honest gap (a2/a3 assumption):** eda-base3 ships NO first-class ~10-round
  research+critic "deep-research" shape. Catalog = linear / modular / poc-iterate /
  diagnose-fix-verify / research-synthesize / creative-production / gather-validate.
  **The deep-research shape (d2) must be DEFINED NEW** — as a TEXT-FILE shape (d5),
  most naturally an extension of `research-synthesize` iterated to depth with an added
  critic role. **Owned by s3 (runtime) + s4 (Shapes screen).**
- Port: keep `shape` as a string; back each shape with a declarative **TEXT FILE**
  (node templates + edges + max-iter ceiling). Runtime reads + honors the UI-set max-iter
  cap, persisted in shared SQLite. Shape SELECTION = one Gemma structured call: schema
  with `shape` ENUM (names harvested from the shape files) + required, `think=false`,
  temp 0 (add an `escalate` enum value for low confidence). Port `_first_ready_action`
  and `plan_next_action` verbatim as pure Python; run independent ready nodes concurrently.

### 2b. SPECIALIZATION INJECTION (N-per-node)
- eda-base3: `ruleset.py:assemble_ruleset` (DFS post-order over `extends`, universal-first,
  dedup) → `compose.py:compose_specialist_docs` (N==1 → doc verbatim; N≥2 → banner +
  `# ===== Specialist stack: <id> =====` ordered concat, universal repeats per stack by
  design). `Action.spec_ids` (planner-stamped) + `Action.injected_context` (dispatcher
  stamps `load_bearing_decisions`+`banned_options`).
- Port: `assemble_ruleset` + `compose_specialist_docs` are deterministic string ops →
  **pure Python unchanged.** INJECTION = prompt-prefix construction: node's Gemma prompt =
  `compose_specialist_docs(spec_ids)` + `injected_context` + the node task. `spec_ids`
  stays a list (N-per-node, ordered concat, per-stack header). Resolve descriptors→spec_ids
  via a Gemma structured call (enum of available spec_ids) or embedding lookup; **fail-closed:**
  refuse to dispatch a specialist node whose stamped spec has no compiled doc. The schema
  constrains only the node OUTPUT, not the grounding text.

### 2c. NODE ROLES (same spec, different behavior)
- eda-base3: ROLE is NOT a schema field — it is the spawn PROTOCOL (pool sets `EDP_ROLE`,
  which selects the skill prompt: worker.md vs reviewer.md). SAME compiled doc loaded by
  both; `BranchReviewer` spawns a reviewer with the SAME neuron_id+spec_id.
- Port: since there is no Claude-Code pool, make `role` an **EXPLICIT node field**
  (`research|critic|worker|reviewer|synthesis|verify`). Role selects a **role-prompt
  template** (the local equivalent of the skill .md): same compiled doc prepended,
  different task framing + different OUTPUT SCHEMA:
  - research → `{findings[], sources[], open_questions[]}`
  - critic → `{gaps[], weak_claims[], follow_up_queries[], verdict: enum[converged|needs_more]}`
  - reviewer → `{verdict: enum[pass|concerns|fail], findings[], fixed_inline[]}`
  This IS the deep-research engine: ONE specialization reused across all rounds; each
  round's two nodes differ only by role-prompt + output-schema. The SHAPE (not the critic)
  bounds the loop count.

### 2d. LIFECYCLE `in_progress→verifiable→done` + reviewer-fixes-inline
- eda-base3: `fsm/state_machines.py:ACTION_TRANSITIONS` (`in_progress→[verify,done,failed,pending]`;
  `verify→[done,failed]`; the state name is **`verify`** = "a done-CLAIM whose acceptance
  gate has not passed — parked, visible, re-checkable, NOT failed"). `RecordActionStatus`
  requires evidence; runs the optional deterministic `verify` dict (`file_exists`/
  `file_min_bytes`/`glob_matches`/`command`): PASS→done, FAIL→parked in `verify`.
  reviewer.md Step 2.5: bounded in-session FIXES, then the gate re-runs.
- Port: state machine + verify gate = **pure Python** (stdlib + subprocess for the
  `command` check). `done` requires non-empty evidence. Expose legal transitions as the
  enum on any status-mutating structured call so Gemma can never emit an illegal state.
  The reviewer NODE is a Gemma `role` whose prompt authorizes safe in-session fixes via the
  file-write tool, then re-records status so the gate re-runs; output schema enumerates
  `verdict` + `fixed_inline[]`. For non-deterministic acceptance, the reviewer verdict IS
  the gate.

### 2e. SELF-HEAL / replan-on-failed-node
- eda-base3: `Reconcile→_advance_plan_liveness` probes `pool.liveness`; dead worker →
  tiered heal (`attempt<MAX` → attempt++ + reset to pending + idempotent re-dispatch; else
  surface `CHILD_CRASHED`). `_rollback_failed_dispatch` undoes a pre-stamped `in_progress`
  on spawn failure. `RecordPlan` overwrites the action set atomically (must re-include done
  actions). Self-heal Lambda on the EventPlane is a supported PATTERN.
- Port: detection + tiered re-dispatch = pure orchestration (no LLM): planner heartbeat
  probes each `in_progress` node's task/process liveness; dead + `attempt<MAX` → reset to
  pending + re-dispatch (idempotent); exhausted → surface to user/neuron. A node returning
  `{status:"failed", reason}` halts terminalization; the planner makes a heal DECISION via a
  Gemma structured call enum `{retry|pivot|abort|extend}` (`think=false`, temp 0). Prefer a
  DB UPDATE that keeps `done` rows + replaces pending/failed (safer than full-JSON re-send).
  Register the heal rule on the `reactive_tools` EventPlane/LambdaRegistry — the Lambda only
  ROUTES the event to the planner's deterministic heal logic; **the planner owns control flow**
  (d1).

---

## 3. SEND_MAIL channel — BOTH facts recorded; decision RESOLVED (d7)

**Fact A — ReactiveAgents repo (a3):** the existing mail path is **plain SMTP over Gmail**
(`smtplib` STARTTLS + a Gmail **App Password**), NOT GCP/Gmail-API. Creds live in
gitignored `C:\Projects\ReactiveAgents\.env`: `SMTP_HOST=smtp.gmail.com`, `SMTP_PORT=587`,
`SMTP_USERNAME`, `SMTP_PASSWORD` (16-char app password), `SMTP_FROM_EMAIL` (= username).
Loader `reactive_tools/config.py:load_smtp_config()` → frozen `SmtpConfig` (password
redacted in repr, BOM-tolerant). Send path `reactive_tools/email_tool.py:make_send_email`/
`register_email_tool` (name `send_email`) → STARTTLS→login→`send_message`, returns a
structured dict, never logs the password; proven in-process by unit tests + a live Round-2
send. **Usable as the channel as-is.**

**Fact B — evolving-deep-agent GCP inventory (a6):** the ONLY Google/GCP config that exists
is an **OAuth client-secret scoped to Calendar.events ONLY** (`GOOGLE_CALENDAR_CREDENTIALS_FILE`
→ a `client_secret_*.json` in `C:\Users\aksou\Downloads\`, OUTSIDE the repo; token file
`.memory/gcal_token.json` currently MISSING). **NO service-account JSON, NO Gmail-API code
path, NO `gmail.*` scope, NO GCP_* env vars.** That repo also sends mail via SMTP+App-Password
(`mcp-service/plugins/send_notification.py`). → There is **NO GCP/Gmail-API mail path anywhere.**

**DECISION (d7 — RESOLVED, gate CLEARED):** keep **SMTP + App-Password** (the working,
Round-2-proven channel). **No Gmail-API migration.** (This supersedes my action's "flag as
pending" wording, which predates d7.)

**Design mandate for s2:**
- `send_mail` is **CHANNEL-ONLY** — the agent/node writes the content; the tool only sends.
- **Recipient HARD-LOCKED** to the user's own address **`aksoulkar@gmail.com`** (= `SMTP_FROM`)
  regardless of backend. The existing `send_email` does NOT enforce this (it accepts an
  arbitrary `to=`, defaulting to `from_email`). **s2 MUST enforce the lock:** drop `to` from
  the exposed tool schema entirely (always send to `SMTP_FROM`), or reject/override any
  `to != from_email`. Never pass a caller-supplied recipient through — the local Gemma model
  must not be steerable to send elsewhere.
- Build behind a **small swappable adapter interface** so a different backend (e.g. a future
  Gmail-API adapter sourcing creds from elsewhere) can be added LATER without reworking nodes
  (growable registry, d1). Default + only concrete adapter today = SMTP+App-Password.
- Naming nit: describe the registry entry as **SMTP-over-Gmail**, not "GCP".

---

## 4. Per-build-step pointers (what this blueprint mandates for each step)

- **s2 — Tool registry + 6 tools.** Build the thin **Pydantic-typed registry** on the existing
  `llm_framework`/`reactive_tools` (adding a tool = one entry); tool selection via native
  structured outputs (`think=false`, JSON schema enum+required, temp 0). Replace web_search with
  **ddgs/DuckDuckGo** (free, cache+backoff) and web_fetch with **httpx+Trafilatura** (§RC5/d4).
  file read + file write **HARD-SANDBOXED** to a workspace root (refuse outside paths).
  **send_mail per §3** (channel-only, recipient hard-locked to `aksoulkar@gmail.com`, swappable
  adapter — gate CLEARED by d7). cron list/add/delete persist to shared SQLite (firing service =
  s6). **OFFER all 6 tools to nodes.** Prove each tool live on gemma4-e2b-agent + both safety bars +
  growability (a dummy tool via one entry becomes callable).

- **s3 — Shape-aware cyclic planner + runtime (the heart).** Add `shape` selection + the node
  **`role`** field (§2a/§2c); execute linear=sequential, modular-parallel=concurrent, and the NEW
  **deep-research** text-file shape (`~9×{research+critic}+1×{research+synthesis+verify}`, ONE spec,
  role-driven, growing visibility, **honor UI max-iter cap**). Wire the planner's **self-heal Lambda**
  (§2e) and the **universal lifecycle gate** onto every node path (fix RC9). Nodes invoke
  specializations+tools (s2), not raw LLM auto-completion. Route the real chat through Planner+live
  runtime (fix RC1/RC2). Prove each shape's topology by trace + a self-heal on a failed node.

- **s4 — Specializations + re-editable spec chat + Shapes screen.** N-spec injection per node (§2b,
  visible in trace). Missing-specialist path: notify user + SSE fallback OR define-and-resume (fix
  RC8). **Re-openable spec chat** to view/edit an existing spec, persisted + effective next run (fix
  RC7). NEW dedicated **'Shapes' screen** listing the text-file shapes + setting **max-iter per shape**
  via UI (persist SQLite; s3 runtime honors it — d5).

- **s5 — Interactive chat with memory.** Wire memory into the chat→planner loop (fix RC6): SQLite-
  persisted, multi-turn context (N references N-1), isolated threads, survives restart; user message
  drives a real plan, results stream over SSE.

- **s6 — DB-backed cron scheduler + catch-up.** Always-on service reading cron entries (created via
  s2 tools) from SQLite; fires plans on schedule; on wake catches up **AT MOST 3** missed fires (older
  dropped); survives restart; email scenario sends unattended via the locked send_mail.

- **s7 — End-to-end 3-scenario live acceptance.** Wire all primitives; run the 3 reference scenarios
  live on gemma4-e2b-agent (research→markdown file; every-morning-email w/ ambiguity check + cron;
  missing-specialist→SSE fallback), fixing integration gaps inline; final domain reviewer pass. No
  scenario hardcoded — each driven by planner shape selection.

- **s8 — Cleanup.** Remove noise evidence/proof/artifact/scratch files from prior rounds; keep source,
  config, workspace, genuine docs (incl. THIS doc). Report the cleaned list via record_* tools only.

**Discipline for ALL steps (d6):** do NOT write scratch evidence/proof files — the `record_*` MCP
tools ARE the completion channel; runtime proofs are reported THROUGH them. **Banned (x1):** adopting
LangChain/LangGraph/PydanticAI/smolagents wholesale — the planner owns control flow.
