---
name: spec-name-conditional-and-engine-authored-structure-are-fabrication
description: "reviewing definition-layer work: an engine 'if spec==X' conditional IS a flag, and engine code that authors structure is the app fabricating — not a guardrail"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e87ac66f-7792-49f2-bd76-00d5e1e988ab
---

When read-verifying a "definition-layer / no-flag" change, two patterns are
**fabrication** even when the worker labels them a guardrail, and you must catch
them by READING the code, not trusting the comment:

1. **A spec-NAME conditional in the engine is a flag.** `if spec == 'section-html-writer'`
   / `if 'X' in spec_names` = the engine branching its behavior on which spec was
   selected = hardcoded behavior keyed by a name. "Self-gated on the spec name so
   the base path is a NO-OP" is NOT a valid guardrail — it IS the flag.
2. **Engine code that AUTHORS structure is the app fabricating, not the model.**
   Code that computes section IDs, stamps a skeleton/task, or renders JSON→HTML is
   the application producing the deliverable's structure (d218). The MODEL must emit
   the output; the engine only delivers the instruction + dispatches tools.

**The faithful design** (the whole bundle/spec architecture rests on this):
methodology lives in the **SPEC TEXT** (the model does X because the spec says so,
d246/d254), the **PLANNER authors the topology/DAG** by reasoning from the spec
(d237/d255-d257), and the **engine stays generic** (zero spec-name conditionals;
the served write path is identical regardless of which spec is selected). Base
behavior comes from ITS OWN spec, not a code NO-OP. Move any pre-existing hardcoded
prompt directives OUT of the engine INTO the spec.

**Why:** s16/SA-6 (2026-06-28) re-authored section-html-writer to skeleton-then-fill
but did it with `_apply_skeleton_fill` self-gated on `spec=='section-html-writer'`
+ `_skeleton_section_id` code-authoring the skeleton + a `runtime.py` spec-name
conditional — labelling it "definition-layer, self-gated, base NO-OP". I surfaced it
to the neuron flagging only the runtime edit "for verify" and accepted the
self-gating as a guardrail. The **USER caught it as fabrication** and the neuron
verified in code; redo = d278 (methodology in spec, planner authors the DAG, zero
spec-name conditionals). I missed it by trusting the "no-flag" comment.

**How to apply:** before surfacing ANY definition-layer SA result, do your own
read-verify: grep the served path for `if ...spec... ==` / spec-name conditionals
(must be ZERO); confirm the engine doesn't compute/stamp the deliverable's
structure; confirm the methodology is in the spec text and the planner authors the
topology. A scope-exceeding CODE edit (beyond prompt/spec text) should CONSULT, not
self-judge (d240). Ties [[review-knob-on-served-route-not-just-wired-site]]
(prove/verify on the real served path, read the code not the comment).
