# Autonomy Rebuild — Gate Evidence Report

**Dates:** 2026-07-13 → 2026-07-14 (autonomous multi-gate drive, owner-approved plan
`graceful-plotting-snowflake`). **Charter:** strategize → plan takes a shape → delegate
workers with dense skills → workers pull context bi-directionally → reviewer reads the
artifact and fixes it itself; the engine is a thin orchestrator + messenger; tools are
generic and single-purpose; the engine never fixes, fills, composes or assembles model
output.

All suites green at close: agent_runtime **497**, chat_app **221**, reactive_tools
**143**, specialization **62**.

---

## Phase 1 — Observation envelope, image tool, scheduled fire (Gate 1: PASSED)

- **Messaging layer:** every tool observation is wrapped `[TOOL RESULT]…[/TOOL RESULT]`
  at the single transport chokepoint (`OllamaTransport._normalize_tool_roles`);
  `AGENT_IDENTITY` declares the convention. Tool output is now distinguishable from the
  user's words; in-memory histories keep semantic `role:"tool"`.
- **Generic image tool:** `image_search` (ddgs `.images`, bounded records
  `{title,image_url,source_url,width,height}`, cache/backoff/deny-list). Zero
  report-awareness in the tool.
- **Scheduled fire proven live:** first-ever cron fire 2026-07-13 13:25 UTC →
  `send_mail` SMTP 250. `cron_add` arg composition hardened (exact minute, whole-task
  prompt). *Open defect: duplicate job registration observed once.*

## Phase 2 — Pull-writer in the unified loop; raw loop deleted (Gate 2: PASSED at 2f)

Every node (write, synthesizer-terminal, gather, trivial) now runs the ONE unified
self-select loop: it starts tool-less with `get_bundles` + `finish`, loads `file` /
`research_read` itself, PULLS its grounding, and drives `file_write` per its spec.

- **Deleted (989 lines):** `_run_file_delivery`, `_run_synthesis`, `_run_raw_file_loop`,
  `_dispatch_writer_tool`, `_parse_writer_call`, `_tool_calling_writer_tool_specs` and
  every rider they carried (shell imperative, figures/table mandate, scope-faithful
  completion, sources-only-final, section inventory, per-turn continuation directives,
  `is_detailed_task` forced continuation, `_is_csv_ext`/`_is_html_ext` branches,
  `strip_internal_scaffolding` write-path edits). `is_detailed_task` and its intent
  regex are deleted from `synth_tools`. Rider content moved to its owners first:
  writer-spec coherence doctrine, file-bundle multi-part ownership + pull discipline,
  the tool-authored `file_write` tail note, and a new `csv-writer` seed spec.
- **Engine push → node pull:** the write planner receives a code-assembled,
  token-budgeted digest (`chat_app/digest.py`) instead of a 12k findings blob; node
  briefs carry the verbatim `[S#]` source index; writers pull via
  `read_notes` → `load_source`.
- **Honesty gates added (verify, never author):**
  - *Target-artifact gate* — a node whose plan declared a deliverable file cannot
    conclude without a write-shaped tool result on it; bounded actionable tool-error
    bounce, prose salvaged on exhaustion. (Reviewers exempt: their job is verification.)
  - *Staleness guard* — the target file is snapshotted before the write plan; unchanged
    bytes ⇒ honest "no deliverable", never last session's file dressed as fresh.
- **Gate 2 ladder (each live run isolated one defect):** 2a mechanics ok / 0 pulls; 2b
  doctrine without targets; 2c `[S#]` targets → pulls but clobbered writes (file bundle
  had no `append` arg + `overwrite` defaulted true — both fixed); 2d append discipline →
  4 thin nodes; 2e ONE-write-node directive → **turn-1 prose, zero `file_write`, stale
  2d file shipped as the artifact** (the finding that produced both gates above);
  **2f PASSED** (36 min): one write node, `get_bundles(file)` → one 12.4KB `file_write`
  → finish; fresh bytes; **11/11 cited URLs grounded** in real fetched sources; max
  prompt 13K tokens (vs 76–96K chars pushed before); zero engine byte edits.

## Phase 3 — Reviewer owns the artifact; summary is the chat turn (Gate 3: MOSTLY PASSED)

- Write planner directive now requires ONE `final_review` node (role=`reviewer`,
  same format spec as its rubric) that `file_read`s the artifact, fixes defects itself
  via `file_update`, and reports an honest status.
- `finalize_summary` is grounded in the reviewer's model-authored prose
  (`memory_index` input); the persisted chat turn is the summary + artifact download
  card, the document stays artifact-only (plan-chain route).
- **Gate 3 live (28 min, Maratha report with images):**
  - n1 writer: `read_notes`×3 (memory pulls ✓) → **target gate fired** on a no-write
    conclusion → model loaded file bundle → `file_write`×6 → 29.4KB fresh artifact ✓
  - n2 reviewer: `file_read` → **one real `file_update`** → honest model-authored
    status ✓; grounded finalize ✓; chat turn = summary + card ✓; 3/3 citations grounded ✓
  - **NOT MET:** 3 `placeholder_map_*.jpg` img srcs returned — `image_search` was
    unreachable from the writer (fixed after the run: file-bundle doctrine
    "IMAGES ARE REAL OR ABSENT" — load the research bundle, src = verbatim
    `image_url`, omit if none). The reviewer's single fix missed the duplicated
    Sources/`</html>` tail and its status overclaimed "complete".

## Phase 4 — Frontier persistence, breadth, budget-as-data (Gate 4: PARTIAL)

- Breadth default 3 → 10 (owner decision; tracks `RA_RESEARCH_FETCH_BREADTH`).
- `expand_branch` acks carry the remaining branch budget as data.
- `ResearchState` gained a `.frontier.jsonl` sidecar; `DagGrower` persists each
  dispatched wave as the open frontier (cleared on a settled stop); `seed_layer`
  seeds **frontier-first** on follow-ups — the model's own open branches resume
  verbatim instead of B1→B5 re-decomposition.
- **Gate 4 live (9.5 min, same-chat follow-up "replace placeholder images + expand
  modern influence"):**
  - **MET:** `image_search` fired with real map records (the flex gap closed); the
    follow-up grounded real image URLs; run completed with honest node outputs.
  - **NOT MET / UNTESTED:** the planner chose an *acyclic* edit plan, so
    frontier-seeding (`s1_F*`) was not exercised; the existing artifact was never
    edited — the first node failed to `file_read` it through 10 self-heal attempts
    (the file existed at the sandbox root the whole time) and emitted an unverifiable
    "survey of the docs directory" claim; the run misnamed its deliverable
    (`findings-for-map-images.md`) and the acyclic route bypassed the
    summary-as-chat-turn rule (whole HTML persisted to the turn again).

## Open defects (honest backlog)

1. **Global workspace collision** — the file sandbox is app-global; identical
   model-chosen filenames leak across chats/runs. Needs per-chat namespacing (hook is
   built once at app boot). The two P2 gates make this honest, not fixed.
2. **Acyclic route parity** — *partially fixed post-Gate-4*: the summary+card chat
   turn now also applies when an acyclic run wrote a real file (`_written_filename`
   keys the swap), so the whole-document-in-chat leak is closed on both routes.
   Still open: acyclic nodes carry no deliverable-target data (so the target gate
   never arms there) and the acyclic artifact body is the node's text output, not
   the file's read-back bytes.
3. **Follow-up artifact resolution** — a follow-up edit task does not reliably locate
   the prior artifact (Gate 4 n1: 10 failed attempts against an existing file);
   the conversation memory should hand the follow-up the artifact's real path/handle.
4. **Reviewer thoroughness** — one `file_update` pass missed duplicated tail sections
   and the status overclaimed; consider the reviewer re-reading after its fix.
5. **Gap→seed threading** — the research reviewer's named gaps are not separately
   threaded into the next seed (frontier resume covers the budget-cancelled case only).
6. **Duplicate cron registration** — one duplicate job observed during Phase 1.

## Verdict

The fabrication layer the owner flagged is gone from the served write path: no engine
riders, no intent flags, no raw push loop, no engine-extracted context, no
tool-output-as-user confusion, no whole-document chat turns (plan-chain), and the
first live run where **plan → pull → write → review → fix → grounded summary** all
happened by model decision. What remains is listed above, honestly.

---

# Addendum — CoT-Autonomy Refactor (2026-07-16 → 07-18)

**Trigger:** the owner's second live-trace review: the agent's reasoning layer is
strong, but its chain of thought never drove the work — tool results commanded the
next action ("search done, now web_fetch"; "fetch done, now take notes"), the first
user turn scripted the exact bundle/tool sequence, engine bounce-gates re-prompted
conclusions with prescribed fixes, and role behavior lived in engine strings.
**Ruling:** no spoon-feeding, no babysitting anywhere — behavior lives in its owning
text layer; tool output is facts the model reasons over; prompt texts are validated
in live batches using the thinking channel in traces.

## Layer ownership (now enforced)

| Layer | Owns | Home |
|---|---|---|
| Identity | who the agent is + channel protocol | `transport.py AGENT_IDENTITY` |
| Operating protocol | reason → ONE tool call → observe → … → finish | `bundles/base.py AGENT_OPERATING_PROTOCOL` → every node's system turn |
| Role | one drive statement | `roles.py` |
| Shape | planner-only plan-authoring strategy | `shapes/*.toml decompose_methodology` |
| Specialization | brief→output business logic + quality bar | `specialization/seed.py` |
| Bundle doctrine | domain knowledge, delivered once at load | `bundles/*.py` |
| Tool description | what the tool does + how to use it well | ToolDef/spec descriptions |
| Tool output | facts only: state, counts, cursors, error_kind | handlers/observation builders |
| Engine | orchestration, messaging, resource caps — no instructional text | runtime/agentic |

## What changed (all landed, all suites green: 499/221/166/62)

- **P0** — `scripts/promptlab/`: the batch prompt-validation harness (isolated live
  module runs graded from traces, including the model's captured thinking) +
  `retired_strings.py`, one registry enforced by `test_no_steering_strings.py`
  (source grep both ways: enforced strings absent, pending strings still present).
- **P1** — one autonomous-agent identity ("your own reasoning drives the work —
  nothing else will sequence your steps"); the operating protocol on every node's
  system turn; minimal role drives; nudges collapsed to one fact.
- **P2** — every tool observation rewritten to facts; per-fetch doctrine appends and
  the take-a-note chain deleted; descriptions absorbed the teaching; tool-shaped but
  unparseable replies get a parse-error fact instead of silently becoming findings.
- **P3** — the first-turn tool-sequence script deleted; **all three bounce-gates
  deleted** (gather-more, note gate, target-artifact — owner ruling; honesty stays
  downstream in the persistence staleness guard + truthful trace attrs); caps and
  turn budgets are neutral resource facts; the ungrounded-URL check is a tool-layer
  refusal with error_kind + the real candidate rows; the fetch-cap datum moved to
  the research-bundle load ack after a live batch showed cap-on-every-brief nudged
  write nodes into gathering.
- **P4** — doctrine single-owners: read-don't-describe + the findings quality bar →
  research-methodology spec; read cost hierarchy rephrased from commands to cost
  knowledge; reviewer file mechanics → file bundle doctrine.
- **P5** — the write-plan strategy (one write node + one same-spec `final_review`,
  source-id assignment, ground-or-drop, no placeholders) moved into
  `shapes/write-file.toml`; `_compose_write_goal` is pure data;
  `_ONE_WRITE_NODE_DIRECTIVE` deleted (the per-turn source-id lever survives until a
  passing plan_author batch proves the shape alone holds).
- **P6** — `planner.review_research` deleted: the planner AUTHORS the review node's
  brief; the node runs the unified loop bound to the research memory with the gather
  workers' specs; its prose is the single signal `decide_followup` reasons over.
- **Channel robustness** (promptlab-driven): `_lenient_content_call` recovers an
  unambiguous multi-KB tool call broken by a single bad escape or a missing outer
  brace — the model's own bytes verbatim, nothing composed. Verified against the two
  real failed 9,030-char turns that had silently lost their writes.

## Live evidence so far (GPU-limited)

- finish_contract 1/1; write module 1/1 after the harness's missing memory-binding
  was fixed (the batch caught it: the node was told "ground in the research memory"
  while holding nothing, and reasonably went web-researching — a harness bug, not a
  model failure); write 2/5 before the lenient recovery landed (both failure modes
  diagnosed from the thinking channel: giant one-shot `file_write` JSON parse loss,
  and `[Source URL for X]` placeholder filler under whole-doc writes).

## Pending (GPU became unavailable 2026-07-16)

1. Batches at zero failures: write (with lenient recovery), gather, review,
   plan_author.
2. App restart onto this code, then **Live Gate B**: the full pipeline twice
   (research → briefed review node → decide → shape-driven write plan →
   final_review → synthesizer summary + artifact card).
3. `scripts/promptlab/trace_assert.py var/traces` — zero retired strings in live
   prompts; token-economy report vs `var/promptlab/baseline_pre_refactor.json`
   (pre-refactor: US-Iran mean 6,015 prompt tokens/call; Maratha mean 9,173).
4. Open judgment calls to validate: the source-id per-turn directive retirement;
   the placeholder-filler tendency under one-shot writes.
