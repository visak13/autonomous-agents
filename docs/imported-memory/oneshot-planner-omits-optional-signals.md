---
name: oneshot-planner-omits-optional-signals
description: "small one-shot local-LLM planner won't volunteer optional free-text DAG signals (needs_spec, parallel topology); prove the mechanism via planner-output-boundary injection + report the authoring limit honestly, don't fake a live trigger"
metadata: 
  node_type: memory
  type: project
  originSessionId: 49445ff6-d294-4c99-a837-0428403fa893
---

ReactiveAgents Round-3 (gemma4-e2b-agent port of eda-base3): the OLD-arch **one-shot whole-DAG** planner (d12) reliably AUTHORS valid DAGs but will NOT volunteer OPTIONAL signals the schema offers. Measured twice: a3 (b) couldn't produce parallel-topology+multi-tool; s7/a4 the planner NEVER emits the free-text `needs_spec` missing-specialist hatch (7 probes — markdown/sonnet/translate/legal × empty + research-only registries — always binds the catch-all spec or leaves `spec=None`). Root cause is the one-shot authoring, NOT a model ceiling or code bug; **s8** ports eda-base3's incremental seed-then-fill authorer and **s9** re-certifies genuine live triggering.

**How to prove a feature whose live trigger the small model declines** (planner-approved, NOT rigging): re-prove the MECHANISM end-to-end on the REAL routed entrypoint (run_agentic/resume_agentic via build_wiring live) by stamping the one declined signal at the **planner-OUTPUT boundary** (wrap `Planner.plan` to post-process its REAL output; node topology/tasks stay model-authored). That injects only the signal the model wouldn't volunteer — every downstream line (detect→notify→sse_fallback→a6 memory) runs untouched. Then report the live-authoring shortfall HONESTLY (option-1 style) — never fake a live planner trigger.

a4 proof: missing-spec notify (EVENT_MISSING_SPECIALIST + [sse_fallback,define_and_resume] choice, not silent) → sse_fallback resume ran real web_search + streamed node_launched→verifiable→done → a6 resume-memory PROVEN CONTRASTIVELY (private sentinel present WITH conversation_context, absent WITHOUT — isolates the threading through resume_agentic→_build_acyclic_runtime→each node's produce turn). See [[runandprove-must-drive-lifecycle-claims]], [[prove-ruleset-effect-contrastively]], [[reviewer-reprove-via-real-entrypoint-and-discipline-isolation]].
