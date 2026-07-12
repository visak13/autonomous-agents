---
name: declare-phases-as-structured-shape-data-engine-reads
description: "RP-6b — de-hardcode a baked flow's PHASE SEQUENCE + spec-routing via a dedicated structured shape field the engine READS (not a hardcoded seed/enum, not overloading an authoring text string)"
metadata: 
  node_type: memory
  type: project
  originSessionId: d13e39ef-8c67-430d-8bc0-2def4a5af6b0
---

RP-6b/d359-d361 (ReactiveAgents s16): removing the last hardcoded structural flow spots
(a research-first SEED in the agentic route + a fixed FOLLOWUP_PLANS phase enum in the
planner) so deep-research phases EMERGE from the shape.

**Pattern that worked:** declare the phase SEQUENCE + per-phase spec-routing as a
**dedicated structured field** (`[[phases]]` array in the shape TOML, each `{kind, spec_role}`),
NOT by overloading the shape's existing `decompose_methodology` TEXT string. Rationale (neuron
d220-approved): the decompose text is handed VERBATIM to the research tree for facet breadth —
conflating would risk regressing breadth; and phase sequence/transitions are engine
ORCHESTRATION data, so reading them as declarative DATA (like fan_out/max_layers) is the right
definition-layer fix and more anti-fab than model-authoring text. This differs from
[[dehardcode-baked-flow-via-shape-methodology-with-precedence]] where the fix WAS a
decompose_methodology string — because there the baked thing was an AUTHORING recipe; here it's
an engine phase sequence the engine legitimately needs to read.

Engine reads via ShapeSpec accessors: `first_phase_kind` (seeds the loop), `next_phase_plan`
(loop default_next transitions), `followup_plans` (the planner enum, derived at import with the
literal kept as a CLEARLY-MARKED offline-only fallback), `spec_role_for(kind)` (spec routing).

**Bug A (d355/d356) dissolved by the spec-routing:** research seed spec routed by the shape's
declared research ROLE via a generic role->seeded-default map (research-analyst), NOT by grabbing
the first user-requested registered spec (the old F5 behaviour that put a WRITER spec on a
research node). User-named writer reaches ONLY the write phase (write planner already gets
requested_specs). Role map, NOT a spec-name/shape-name conditional (banned). CompiledSpec has NO
gather-vs-writer role attr, so a finer per-requested-spec role-match = RP-6d residual.

**Gotcha caught:** `next_phase_plan` MUST fall back to the canonical research->write order for a
shape with NO declared phases, else it returns "done" after research and skips the write phase —
broke 10 served-route tests that inject a minimal catalog. Keep no-phase behaviour byte-identical
to the retired hardcoded default.

Boundary held: KEEP the 2-invocation drive (research phase + write phase) — that's RP-6c, not
RP-6b. Declarative-only change → a UNIT/self-policing Bug-A proof suffices (no live E4B run).
All 5 suites green (agent_runtime 500, chat_app 202=193+9, reactive_tools 137, spec 62, llm 18).
