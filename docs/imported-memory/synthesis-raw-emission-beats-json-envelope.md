---
name: synthesis-raw-emission-beats-json-envelope
description: "ReactiveAgents D1 — terminal synthesizer truncated because a long HTML doc was forced through a {output} format=schema escaped-JSON string; fix = raw stepwise emission + completeness backstop"
metadata: 
  node_type: memory
  type: project
  originSessionId: 487b394c-cd04-46e8-abf5-713e5a30057c
---

ReactiveAgents s9/c1 (D1 fix), proven live on E4B: the terminal SYNTHESIS node
truncated mid-document (684 tok / 2495 chars, JSON closed mid-sentence, no
timeline/table/sources) because the deliverable was emitted through a single
`format={"output": <string>}` schema call — a long HTML report serialized as ONE
escaped JSON string. Fix (`agent_runtime/synth_tools.py` SynthesisBuilder +
`runtime._run_synthesis`): drop the schema entirely, build the deliverable RAW via
stepwise `write_section`/`finish` tool calls (no format), temp=0 (d35),
num_predict 4096/call, num_ctx untouched; `parsed={"output": doc}` keeps downstream
render unchanged.

Two E4B realities that govern the design (the durable lesson — see
[[format-json-fixes-syntax-not-content-schema]] and
[[citation-fix-cant-fix-wrong-chunk-selection]]):
1. For a small model the value of "tool-driven" synthesis is the RAW UNESCAPED
   emission, NOT the JSON tool-call envelope. E4B is non-deterministic on the
   envelope for LONG HTML — it sometimes emits 0 parseable `write_section` calls
   (HTML-in-JSON escaping friction); a RAW single-call fallback then produces the
   full report. Both paths satisfy D1 because neither forces an escaped long string.
2. E4B sometimes calls `finish` after ONE section (dangling `<section>`, no
   timeline/table/sources). A one-shot completeness guard (`_synthesis_incomplete`:
   unbalanced `<section>/<table>/<ul>` or html-doc lacking `</html>`) nudges it to
   finish — it FIRED live (sections 1→5). It returns False for plain/markdown so a
   short reply still finishes immediately.

Live proof: 3/3 fresh US-Iran "detailed HTML report" runs COMPLETE + sourced
(4585 / 8533-via-fallback / 10128-via-stepwise+nudge chars). App on E4B :11434.

**s9/c1 d49 RE-SCOPE UPDATE (supersedes the write_section/finish builder + the
`_synthesis_incomplete` heuristic above; proven 6/6 live on E4B).** Two corrections,
both MEASURED: (a) asking the model to EMIT `file_write`/`file_read` JSON tool calls
with embedded content FAILS the same way the {output} envelope did (iran:
tool_calls=0, fell to fallback) — the friction is escaping ANY long string inside a
JSON envelope, tool-call or not. (b) The continuation/completeness loop must be
ORCHESTRATION-driven, not a hard-coded heuristic (d48). New design (kept the core
lesson "raw emission beats the envelope"): the model emits RAW content sections; the
LOOP does the real `file_write`(append) + `file_read`(tail) READ-BACK and feeds the
ACTUAL on-disk bytes back; the model confirms `<<DONE>>` judged from the real file.
Read-back fires UNCONDITIONALLY (even a one-shot emission is written then read back),
killing false-finish AND truncation. The LLM-chosen extension lands via
`derive_output_path` (explicit filename > bound writer-spec ext > format kw > .md);
`run()` surfaces the written path as a file_write result so the chat artifact carries
it (cats.html stays cats.html). `_synthesis_incomplete` + SynthesisBuilder DELETED.
Live: 6/6 (iran x3 .html complete+sourced, md, csv, cats.html), all mode=react_file
+ finished, no fallback. Also fixed a SEPARATE leak on the acyclic web_search->
file_write path: a bare node's spontaneous `{"output":...}` envelope leaked into the
written file — `unwrap_output_envelope` (synth_tools) wired into toolargs + agentic.
See [[incremental-authoring-needs-per-node-prompt-for-small-model]] (same "bare
mechanism under-binds on the small model; engineer the orchestration" lesson).
