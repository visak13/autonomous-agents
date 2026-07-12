---
name: generic-engine-iterative-breadth-is-decompose-first
description: "making a generic unroll engine match a bespoke tree's iterative breadth — the lever is the decompose-first seed, not the grow loop; prove engine-parity within-run on E4B"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2d10d41d-95e1-49fe-924f-af42e14bbe22
---

ReactiveAgents s13 P2-5b-genloop: made the GENERIC declarative-unroll + AgentRuntime engine
reproduce `run_research_tree`'s iterative gap-driven breadth so the bespoke tree can be retired.

**The fix that worked:** relax EXACTLY ONE invariant — "node set fixed at unroll time." Shape
declares `expand_on_gaps`/`fan_out`/`max_layers`; `unroll_shape(grow=True)` emits a SEED-ONLY
`growable` DAG; a `DagGrower` (in research_tree.py) REUSES `run_decision_node` + `ResearchState`
+ `Tree` + `completeness_stop` VERBATIM (no bespoke python, no 2nd engine); runtime
`_drive_growable` appends each gap-authored layer, bounded by max_layers/no_expansion/stop_research.

**LOAD-BEARING insight (1st live run REFUTED a whole-goal seed):** breadth comes from the
DECOMPOSE-FIRST seed, NOT the grow loop. With a whole-goal seed the decision node DID
re-frontier (4 gap-grounded branches) but E4B called `stop_research` in the SAME layer → by the
tree's own "stop wins over same-layer expansion" rule those branches were discarded → generic 2
sources vs tree 4. Mechanism correct, seed shape wrong. Fix = `DagGrower.seed_layer` reuses
`run_decompose_node` (the tree's `seed_only_root`) to front-load breadth BEFORE the first stop
judgement. 2nd run: generic 6 ≥ tree 5, parity holds.

**Proving engine-parity on a non-deterministic small model:** E4B varied 9/4/5 tree sources
across 3 runs same query. Don't gate on an absolute number — the robust signal is the
SAME-BUDGET, WITHIN-RUN, two-engine comparison (`scripts/p2_5_parity.py` runs both engines back
to back, compares generic≥tree). Keep the consolidation behind the reversible
`RA_GENERIC_REPORT_PATH` flag pending the human UI quality spot-check (recipe's real gate).

Gate non-regression by gating the seed-only unroll on an engine `grow=True` opt-in so a shape
gaining the capability never silently turns a non-growing caller's full unroll into seed-only.
Related: [[e4b-fetch-ceiling-diverges-by-path]], [[review-knob-on-served-route-not-just-wired-site]].
