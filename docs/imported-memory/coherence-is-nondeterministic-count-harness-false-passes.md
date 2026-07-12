---
name: coherence-is-nondeterministic-count-harness-false-passes
description: "s14/a11 gate FAIL — served deep-research report coherence is run-to-run NON-DETERMINISTIC; the count-harness coherent=True is a FALSE PASS, only a human read catches the structural collapse"
metadata: 
  node_type: memory
  type: project
  originSessionId: 467ae640-5238-46d9-b446-82370af8bcde
---

s14/a11 FINAL live acceptance gate (2x served US-Iran HTML report on E4B) = STRUCTURAL-COLLAPSE FAIL. a17 run4 was clean but a11 run1 collapsed → coherence on the served deep-research report is RUN-TO-RUN VARIANCE (run4 was luck, never deterministic; neuron d173). CONTENT was strong (synthesized exec-summary, ~13-event dated timeline, 2 styled tables w/ $11.3B/$16.5B/3.2M/1492 figures, S1-S7 real cites, 0 wiki, 7/7 cited, planner-authored source_ids 9/11, bounded 63% envelope, fetch-recovery ok) — the ASSEMBLY collapsed.

The d168 lesson re-confirmed live: my automated `coherent=True` was a FALSE PASS. The next gate's structural asserts MUST cover the blind spots a title/keyword counter misses:
1. empty sections whose emptiness is an HTML COMMENT (`<!-- ...will be added in subsequent turns -->`), not "UNSUPPORTED"/"no sources" text.
2. duplicate section FAMILIES differing only by an id suffix like `-2` (#timeline-2/#costs-2/#analysis-2) where the EMPTY twin has no `<h2>` so a title-family counter sees each family once.
3. the "Sources Cited" footer block repeated N times (each section-writer re-emits its own), incl copies truncated mid-URL (those become the "non-resolving URL" false signal).
4. worded placeholders like "Source used in Executive Summary" replacing dropped URLs.
5. nav/ToC anchors pointing at the EMPTY twin ids (broken navigation).

ROOT (d173, NOT a gate concern): the shell pre-creates empty "content will be added" stub sections with nav ids; incremental section-writers APPEND new `-2` sections + their own sources block instead of FILLING the placeholders. Fix = deterministic assembly (forbid shell empty-stubs, collapse `-2` dups, one sources block, rebuild nav from real sections) — a NEW build action the planner authors. As the gate I STOP+surface, do NOT fix (hard-stop d148/d149). Batch A stays OPEN. Re-gate requires coherence on BOTH 2x runs by human-read + strengthened asserts. Builds on [[a14-chunking-unvalidated-at-full-breadth]] and [[writer-feed-goldilocks-push-full-bodies-of-few-scoped-sources]].

**s14/a18 RESOLVED (d174 neuron diagnosis):** built a pure idempotent served-assembly orchestrator (synth_tools `assemble_report_spa`, wired into chat_app agentic `run_section_write_phase`) — drop empty/stub sections; collapse `id=X` vs `id=X-2` to the FILLED base id; merge per-writer Sources to EXACTLY ONE block at END rebuilt from verbatim sources; rebuild nav from REAL section ids. The fix's REAL surprises came from the LIVE 2x gate, NOT the adversarial fixtures: FOUR served-write shapes a count-harness missed — (1) num_predict-truncated UNCLOSED `<section>`; (2) anchor-based `<p class="source-list"><a>` Sources with NO `<ol>/<li>` (a list-only detector missed it → the 0→1 net wrongly appended a duplicate `sources-2`); (3) leaked `<div style=display:none><<DONE>></div>` sentinels; (4) a writer double-id'ing a section → per-node reconcile minted a redundant `<h2 id=sources-2>`. Each was invisible to the 4 structural asserts AND/OR to the rendered page — only a manual human-READ of each render caught them. So: drive 2x live, human-READ each render, THEN harden the deterministic assembly to the OBSERVED shapes (added close-unclosed-sections, anchor-aware source detection, strip-done-sentinels, strip-heading-ids, relocate-sources-to-end). 19 adversarial determinism tests + 459 suite, 0 regress. Keep assembly-determinism (this action) STRICTLY separate from write-truncation report-quality (the lost table row = Batch B monolithic-vs-sectioned write, never fixed by assembly — fabrication forbidden d60). Reports: artifacts/a18_fresh_run3.html, a18_fresh_run4.html.
