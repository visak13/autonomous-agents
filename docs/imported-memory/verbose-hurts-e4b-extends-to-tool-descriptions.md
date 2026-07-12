---
name: verbose-hurts-e4b-extends-to-tool-descriptions
description: "RP-4b/d336 — rewriting a tool's DESCRIPTION (verbose OR concise) regressed E4B schedule-only 5/6→1/6; the desc is fed verbatim into the planner tool catalog so 'verbose hurts E4B' applies to tool descriptions, not just spec prose. Keep tools LEAN; put guidance in selector/planner doctrine. Prove regression-vs-load with a same-window control."
metadata: 
  node_type: memory
  type: project
  originSessionId: 5e94acad-6add-407f-a5f7-d5fc83ab0961
---

RP-4b (recipe-make-the-reactiveagents-chat-genuinely-r, s16). The d333 "write the
cron TOOLS properly → the LLM picks cron_add reliably" premise measured BACKWARDS on
gemma4-e4b: enriching the `cron_add` tool DESCRIPTION (verbose ~950ch AND concise ~618ch)
dropped schedule-only from **5/6 → 1/6** with MalformedOutputError ("no usable nodes") and
run-now misroutes (web_search→send_mail). The lean **~322-char original held 5/6**.

**Why:** the tool description is fed VERBATIM into the planner's AVAILABLE-TOOLS catalog
(agent_runtime incremental.py) AND the shape_selector tool_catalog — so more description
text confuses E4B's node authoring. The neuron's "verbose hurts E4B" prior (verbose spec
0/4, concise 4/6 — see [[writer-feed-goldilocks-push-full-bodies-of-few-scoped-sources]])
EXTENDS to tool descriptions, not just spec prose.

**How to apply:** keep tool descriptions LEAN. Put intent-separation / selection guidance
in the SELECTOR or planner DOCTRINE prompt (NOT the verbatim tool catalog) — RP-4b's
sharpened concise schedule-this-vs-do-now doctrine there measured non-regressing (5/6
doctrine-only). Landed: lean-original descriptions + sharpened doctrine + RP-4 anti-fab
intact (cron_prompt_from_task removed, verbatim store). Residual ~1/6 = a measured
small-model AUTHORING ceiling surfaced to the USER (d186, no unilateral ceiling).

**Measurement discipline (reusable):** to separate a code regression from a shared-Ollama
:11434 LOAD confound, run the ORIGINAL code in the SAME window as a control before
concluding. Here original=5/6 in the exact loaded window where my rewrite=1/6 → proved it
was the code, not load. Don't conclude load (or a ceiling) from a loaded-window number
alone. Related: [[coherence-is-nondeterministic-count-harness-false-passes]].
