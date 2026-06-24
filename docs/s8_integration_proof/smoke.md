# s8 Integration — SMOKE + HANDOFF (action b11)

**Type:** Quick smoke against the LIVE E4B app — **NOT** a self-blessed full acceptance.
The full scenario acceptance is the **NEURON's** to drive live (see Handoff below).
**Date:** 2026-06-19 · **Reviewer-gate prior:** b10 domain review = PASS on all 11 dimensions.

## App under test
- **URL / port:** `http://127.0.0.1:8000` (detached app, app_pid per b10 = 40260)
- **`/health`:** `status=ok`, `transport=live`, all components present
  (planner, agent_runtime, spec_registry, chat_store, scheduler, cron_scheduler).
- **Model (live, resident):** `gemma4-e4b-candidate-ctx32k:latest` on Ollama `:11434`.
  `/api/ps`: `size_vram == size` = 3,189,841,591 B (~3.19 GB), **0% offload / no shared-GPU
  spill**, `expires_at` far-future ⇒ `keep_alive=-1` (resident). This is the d35/d42 E4B swap, live.

## Smoke (a) — trivial chat turn  ✅ PASS
- **Request:** `"Hello! In one sentence, what can you help me with?"` → `POST /chats/{id}/message`
- **Result:** `ok=True`, `launch_order=[n1]`, `n1=done`, `missing_specialist=False` (no clarify loop /
  no re-ask on a normal turn), answered in ~52 s.
- **Answer:** *"I am an autonomous agent designed to synthesize information, execute complex tasks,
  and provide comprehensive, well-supported answers to your requests."* (identity present, on-task.)

## Smoke (b) — tool-driven flow end-to-end → output file  ✅ PASS
- **Request:** `"Research the health benefits of regular walking and write a short report,
  then save it to a file."`
- **Result:** `ok=True`, `launch_order=[n1, n2]`, both `done`, `missing_specialist=False`, ~144 s.
  - **n1 (research):** real, sourced content (e.g. Dr. Thomas Frieden / Harvard Health / CDC —
    "the closest thing we have to a wonder drug").
  - **n2 (write):** synthesized **the same research** into a structured report ⇒ the research
    reached the writer (d17/d28 dependency-scoped context + research→write edge working live).
  - **Output file:** artifact `report.md`, `text/markdown`, **2,929 bytes** of substantive sourced prose.
- **Trace evidence** (`var/traces/bdf78e235f5dfae053315de9824e1618.md`, `planner.incremental`):
  the DAG was authored by the **tool-call loop** — `seed_plan → add_step → finalize_plan`,
  one tool call per reply, with **reasoning/thinking blocks populated** (think=True). No one-shot
  schema-constrained DAG JSON. `file_write` selected as the delivery synthesizer by description.

## s8 feature presence checklist
Legend: ✅ observed live in this smoke · ☑ present & PASS per b10 domain review (not re-exercised here).

| # | Feature | Status |
|---|---------|--------|
| 1 | Tool-driven iterative plan authoring (seed_plan/add_step/finalize_plan, no format-schema) | ✅ |
| 2 | Edge guarantees — d7 (no dangling edge) + d28 (terminal write depends on research) | ✅ (n2→n1 edge live) · ☑ |
| 3 | Goal-to-worker context (verbatim goal + dependency-scoped upstream feed) | ✅ (n2 saw n1's research) |
| 4 | Synthesizer-by-description + send_mail recipient self-only lock intact | ✅ file_write fired · ☑ lock (4-layer) |
| 5 | Crisp prompts + single ~100–150-tok universal identity on every call | ☑ (identity in answer/trace) |
| 6 | Output-format spec selection (HTML→html-writer / MD→markdown-writer, d43) | ☑ (smoke used MD default) |
| 7 | Shape & spec free-flow edit/refine + compositional intent | ☑ |
| 8 | UI delete buttons (specializations & shapes) | ☑ (SPA served) |
| 9 | E4B swap + d25 revalidation (interceptor/thinking-capture, 6GB fit) | ✅ resident, size_vram==size |

## HANDOFF → NEURON (explicit)
The **FULL scenario acceptance is the NEURON's to drive LIVE** (human-like multi-turn, user's
final sign-off) per d4/d44 — **NOT** self-run or self-blessed here. Items the neuron must drive live:
- Substantive **sourced HTML report** for the US-Iran task (d43) as real HTML.
- Clarify **only** for scheduled tasks (no re-ask loop on a normal report).
- **No unrequested email** (send_mail only when explicitly asked; self-only lock).
- Correct **spec/shape selection**; **shape/spec edit-refine** via chat.
- **Multi-round context** holds; **delete buttons** work in the UI.

**Smoke verdict:** app is UP on E4B and answers; the tool-driven flow is wired end-to-end and
produces a substantive output file. **Handed off to the neuron for live acceptance.**
