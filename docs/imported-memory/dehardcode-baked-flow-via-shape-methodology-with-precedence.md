---
name: dehardcode-baked-flow-via-shape-methodology-with-precedence
description: "ReactiveAgents — a baked generic authoring flow (gather→combine→deliver) that overrides a shape's intent is de-hardcoded by moving the procedure into the shape's decompose_methodology + generic substitution that REPLACES (not just precedes) the generic recipe"
metadata: 
  node_type: memory
  type: project
  originSessionId: be9469d1-a62e-4f85-ba1f-b8e43b57d4e0
---

RP-4c/d341 root cause + fix. The schedule-only run-now-misroute (planner authored `web_search`+`send_mail` run-now nodes alongside the `cron_add` leg) was NOT missing doctrine: the schedule mandate reached the authoring system prompt (`factory.py` description), but the concrete step-by-step **decision procedure** in `IncrementalPlanner._initial_user` (+ `_system` GUIDANCE) was a hardcoded generic "list distinct items → gather-per-item → combine → deliver" recipe with no schedule branch, and the concrete first-user-turn procedure OVERRODE the buried mandate — worst on imperative phrasing.

**Why:** engine-baked generic authoring scaffolds silently override shape/spec intent. The doctrine "behavior lives in shapes/specs" is violated when the engine's own prompt bakes a flow.

**How to apply:** de-hardcode by (1) putting the flow's authoring procedure into the SHAPE's `decompose_methodology` (a new dedicated shape, e.g. `schedule-leg.toml`, concise per [[verbose-hurts-e4b-extends-to-tool-descriptions]]); (2) making the engine GENERICALLY substitute the *selected* shape's `decompose_methodology` into the authoring prompt — mirroring `research_tree.py`'s d161/d170 substitution, no spec-name/flow conditional; (3) CRITICAL: the shape methodology must **REPLACE** the generic recipe when present (generic becomes fallback-only), not merely precede it — else the concrete generic recipe keeps winning (the actual bug). Route the selector to the new shape + add it to `CURATED_SHAPES`. Verified: run-now nodes 0/18 post-fix, imperative 9/9 (was misrouting). This is faithful/anti-fab (engine renders shape text, model authors) — see [[spec-name-conditional-and-engine-authored-structure-are-fabrication]].

**RP-AUDIT F1 (s16, RESOLVED the flagged sweep):** `_finalize_user` (the FORCED-FINALIZE turn the engine fires when the small model repeats instead of finalizing) carried the SAME baked class — a hardcoded gather→combine→deliver closing step + output-format-writer binding + delivery-tool assumption, never consulting the shape methodology. Fix = mirror this exact pattern: presence-check `self.shape_decompose_methodology`, render with PRECEDENCE, REPLACE the hardcoded recipe (fallback-only when absent; empty ⇒ byte-identical). All 3 authoring turns (`_system`/`_initial_user`/`_finalize_user`) now read the same field. Anti-fab test tip: assert the branch is `if methodology:` on the FIELD and grep the method SOURCE to prove NO shape-name/spec-name equality conditional. 880 tests green.
