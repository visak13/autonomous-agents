---
name: served-shape-selector-diverts-off-run-plan-chain
description: "ReactiveAgents s16/a3 — served shape selector can route a deep report to a non-unrollable sibling, bypassing run_plan_chain; engine works when reached; two harness false-signals to avoid"
metadata: 
  node_type: memory
  type: project
  originSessionId: db45cad3-7a36-499b-9ce4-508b72099d17
---

ReactiveAgents s16/a3 (depth-2/breadth-3 served POC gate, US-Iran report on E4B :11434): the model-driven SHAPE SELECTOR non-deterministically picked `concurrent-multi-topic-gathering` (is_unrollable=False, execution=concurrent, "combine and email it") for the deep sourced report, so `run_agentic` routed to the generic planner-DAG path and BYPASSED `run_plan_chain` entirely — expand/grow/prune, research-memory, depth/breadth, pre-surgery snapshot never reached. Prior s15 runs DID route deep-research on the same query → non-deterministic mis-pick = a shape selector/description LEVER (d186), not a ceiling.

**Verify the ROUTE on the served path before trusting a per-engine readout** — a router diverts to a non-unrollable sibling that skips the engine under test (layers with [[review-knob-on-served-route-not-just-wired-site]], [[agentic-router-bypasses-semantic-graph-leg]]). To prove the engine itself, leg-probe the wired fn directly: building the real selector/catalog + a deep-research ShapeSelection and calling `run_plan_chain` directly showed the engine WORKS substantively 10/10 (grow fired to depth bound, 5 reasoned expansions, 2-wave gather 16 notes/8 sources/24 fetches, 2-iter planner loop + terminal synthesizer, d189 events-as-user + role:tool, coherence-clean writer+reviewer with pre==post so d173/d174 no-ops per d222, node-self-select).

**Two harness false-signals:** (1) `second_layer_gathered` indexing `dr['layers'][1]` false-fails because the SEED wave is not a separate entry (only the grow layer is) though grow_layers=2/depth_reached=2 with real gather — read the per-layer gathered + grow counters, not the index. (2) web_fetch `cache_hits=0` is EXPECTED when all fetches are distinct URLs (no repeat to serve) — gate "no excessive refetch" (no url live-fetched >1x), not hit-count.
