---
name: synthesis-role-misclassed-as-judgment
description: ReactiveAgents role pipeline — terminal SYNTHESIS deliverable node mis-classed as a JUDGMENT role returned a verdict scaffold not content
metadata: 
  node_type: memory
  type: project
  originSessionId: 440f6654-4be0-456a-9c07-c8cbbdd77ff0
---

ReactiveAgents (C:\Projects\ReactiveAgents) s8/b8: a role-tagged DAG node pipeline mis-classified the TERMINAL deliverable role (`synthesis`) as a JUDGMENT role, so the final user answer came back as `{"verdict":"pass","findings":[...],"fixed_inline":[]}` (a meta judgment scaffold) instead of the actual content. Root: `roles.py` had `ROLE_SYNTHESIS` in `ROLE_VERDICTS` + pointing at `_REVIEW_SCHEMA`; its framing literally said "Report your confidence as the verdict."

**Why:** This is the o4 "agent answers with a scaffold, not the deliverable" defect, and a clean instance of part-1 flag-removal (fold a hardcoded role-execution gate into reasoning).

**How to apply:** Synthesis = the d38 terminal SYNTHESIZER that PRODUCES the deliverable → give it a content schema (`{output}`, like worker) + content framing; leave `critic`/`verify`/`reviewer` as the judgment gates (verify stays the deep-research quality verdict). ALSO a second, independent leak: the LINEAR path's `_final_output`/`outputs` in `agentic.py` read raw `r.output` (the schema-wrapped JSON) instead of `_render_parsed(r.parsed, for_user=True)` → the raw `{...}` JSON envelope leaked into chat even for worker nodes; fix the render at BOTH the final-answer and per-node-outputs sites. Proven live on E4B: trivial turn went from a verdict JSON to "I'm online and ready to help!". Deep-research full re-validation is s9's neuron-driven job. Links [[no-project-coupled-specialist]] (this is project-internal, not specialist material).
