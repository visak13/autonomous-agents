---
name: e4b-fetch-ceiling-diverges-by-path
description: "E4B's self-directed web_fetch ceiling depends on the RESEARCH PATH — 0 fetches on the agentic ReAct path, broad fetches on the structurally-driven path; a wired-path live breadth number can be unattainable for that reason, not a code bug."
metadata: 
  node_type: memory
  type: project
  originSessionId: e9e3f669-0a78-4c26-a5ee-d0715f48e51f
---

ReactiveAgents s9/N1r re-review: E4B (gemma4-e4b on Ollama :11434) does NOT
web_fetch on the agentic `_run_deep_research` ReAct path — it answers the whole
report from memory. Reproduced 2x: legprobe3 (US-Iran) = 1 react span, 62KB, 0
fetch; legprobe2 (FETCH-FORCING query: "you MUST web_fetch >=5 distinct sources,
your training is out of date, do not answer from memory") = 7 react spans, 70KB,
**web_fetch=0** (grep-confirmed across all probe traces), catalog_pool=0.

NOT slot contention: the probe forced `num_ctx=32768` via its own `build_wiring`
and ran to a full 70KB answer — truncation would yield EMPTY output, not 70KB. A
stale app pinning ctx 16384 on the single Ollama slot is irrelevant; clearing it
would not help, because the model simply declines to enter the fetch loop.

CONTRAST that pins the diagnosis: the SERVED path (run1, through
`run_plan_chain` PHASE-1, which structurally drives per-node fetching) DID fetch
**10 distinct articles / pool 25 LIVE** — breadth>3 demonstrated, just on the
NOT-knob-wired path. So E4B *can* fetch broadly where the path structurally
drives it; the agentic path leaves the fetch decision to E4B's own ReAct loop,
which it repeatedly declines.

**How to apply:** When asked to live-prove a breadth/fetch knob wired on the
agentic research path, expect a wired-path live number to be UNATTAINABLE on E4B
(the cap has nothing to bound at 0 fetches) even when the mechanism is
offline-correct. Measure the live number on the route that actually fetches;
render it as a model ceiling to WAIVE/re-scope (neuron decision), NOT a code
bounce — no source change makes E4B self-fetch. Ties
[[review-knob-on-served-route-not-just-wired-site]] and
[[e4b-long-report-citation-ceiling]].
