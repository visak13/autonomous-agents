# ReactiveAgents

**A fully local, autonomous multi-agent system that runs on a 4B-parameter model and a
6 GB consumer GPU — where the model's own chain of thought drives the work, and the
engine never does.**

ReactiveAgents takes a goal like *"research the decline of the Ottoman Empire and write
me a detailed HTML report"* and autonomously plans, researches the live web, takes
structured notes, writes the document file by file, reviews and fixes its own artifact,
and delivers a grounded summary with the report as a download — end to end on a local
Gemma E4B via Ollama. No cloud APIs, no hardcoded pipelines.

---

## The design thesis: no spoon-feeding

Most small-model agent frameworks work by steering: tool results that command the next
action ("search done — now take notes"), prompt riders that script the tool sequence,
and engine gates that re-prompt the model until it complies. That produces demos, not
agents.

ReactiveAgents is built on the opposite bet: **a small model reasons well when every
layer informs and nothing sequences.** Every behavior lives in exactly one owning text
layer, and the engine is a thin orchestrator + messenger:

| Layer | Owns | Example |
|---|---|---|
| **Identity** | who the agent is + the channel protocol | "your own reasoning drives the work — nothing else will sequence your steps" |
| **Operating protocol** | the reason → act → observe → finish loop | one block on the system turn, once per node |
| **Roles** | one drive statement each | worker / reviewer / synthesizer |
| **Shapes** (TOML on disk) | planner-only plan-authoring strategy | "author ONE write node + ONE same-spec reviewer node" |
| **Specializations** | business logic + output quality bar | html-writer, research-methodology, csv-writer… |
| **Tool bundles** | domain knowledge, delivered once on load | web gathering, file authoring, source reading |
| **Tool descriptions** | what each tool does + how to use it well | "finish's reason is a receipt, never the product" |
| **Tool output** | *facts only* — counts, caps, cursors, error kinds | "file is now 4,212 bytes and ends with: …" |
| **Engine** | orchestration, messaging, resource caps | zero instructional text |

There are no intent-detection regexes, no role-based routing, no engine edits of model
output, and no re-prompting gates. A self-policing test greps the entire codebase for
every retired steering string so none of it can quietly return.

## What a run looks like

```
user goal
  → planner selects a SHAPE and authors a research plan
  → gather workers self-select the research bundle, search + read real sources,
    record structured notes (claim + source + gap) into persisted research memory
  → a growable research tree expands on the gaps the notes surface, prunes dead
    leads, and stops when the model judges coverage complete
  → a planner-BRIEFED review node pulls from the research memory and reports
    honestly whether it supports the deliverable
  → the planner reasons the next phase from that review; for a write phase it
    authors ONE write node + ONE reviewer node (the shape's strategy)
  → the writer pulls its evidence (cheap note gists first, verbatim source text
    for exact figures), then authors the document part by part via file_write
  → the reviewer reads the real file, fixes defects in place via file_update,
    and reports what the document actually contains
  → the chat turn is a grounded summary + the artifact as a download card
```

Context between nodes travels only through two channels: the persisted research memory
(notes + verbatim sources under stable `[S#]` ids, pulled on demand) and each node's
`finalize()` — a `(summary, memory_index)` pair. Nothing pushes 80 KB blobs into
prompts; a 32k context window sustains multi-hour runs.

## Engineering highlights

- **Runs on consumer hardware.** Gemma E4B (4B-class) on a 6 GB RTX 4050 via native
  Ollama, ~60 s/call, 32k context — every design decision is shaped by that budget.
- **Channel robustness without content edits.** Small models emit multi-KB
  `file_write` JSON that breaks on a single bad escape. A lenient recovery extracts
  the call's own bytes verbatim (fences, stray `)`s, missing braces, invalid escapes)
  — measured going from 40 % to 100 % write survival — while the engine still never
  composes or fixes a byte of content.
- **Prompt texts are tested like code.** `scripts/promptlab/` runs isolated modules
  (gather / write / review / plan-author) in live batches against the real model,
  grades each run from its trace — including the model's captured *thinking* — and a
  prompt text ships only at zero failures in a batch. Current scores: write 5/5,
  review 3/3, plan-author 3/3, gather 3/4.
- **Honesty over appearance.** A persistence staleness guard refuses to ship a stale
  file as fresh output; trace attributes report real byte counts; reviewer status is
  the model's own words. Failures surface, they don't get papered over.
- **Deep observability.** Every run emits a full local trace: every prompt, every tool
  call, every observation, and the model's reasoning channel — the raw material the
  whole iteration loop runs on.
- **948 offline tests** across the workspace (deterministic fake transport — no GPU
  needed), plus the live batch harness and end-to-end gate runs with trace forensics.

## Architecture

A [uv](https://docs.astral.sh/uv/) workspace of six in-process components (one
interpreter, one venv — no microservices):

```
llm_framework/    transport (Ollama native API, thinking channel, observation
                  envelope), prompt chains, token accounting
agent_runtime/    the engine: planner, unified self-select worker loop, research
                  tree with persisted frontier, tool bundles, roles, shapes (TOML)
specialization/   the spec registry — output-shaping rulesets the planner binds
                  per node
reactive_tools/   generic single-purpose tools: web search/fetch, image search,
                  sandboxed file read/write/update, cron, mail
memory/           embedding-backed recall (fastembed / MiniLM)
chat_app/         FastAPI app + SSE chat frontend, run orchestration, artifacts,
                  scheduled (cron) agentic runs
```

Every worker runs the same **unified self-select loop**: it starts with only
`get_bundles` + `finish`, reasons about which capability bundles its task needs, loads
them, and drives their tools. Gathering, writing, and reviewing are the *same loop*
with different briefs and specializations — there is no separate "writer pipeline".

## Running it

Requires Windows + Python 3.11+ + [uv](https://docs.astral.sh/uv/) + a local
[Ollama](https://ollama.com) build and a 6 GB-class GPU.

```powershell
# 1. start the model server (builds the custom Gemma tag on first run)
.\scripts\start-native-ollama.ps1

# 2. launch the app (uv sync + FastAPI on http://127.0.0.1:8000)
.\launch.ps1
```

Open `http://127.0.0.1:8000`, ask for something real — a researched HTML report, a CSV,
a scheduled morning news email — and watch the trace land in `var/traces/`.

```powershell
# offline test suites (no GPU)
uv pip install pytest pytest-asyncio pytest-timeout
.\.venv\Scripts\python.exe -m pytest agent_runtime/tests chat_app/tests reactive_tools/tests

# live prompt-validation batches (GPU)
.\.venv\Scripts\python.exe scripts\promptlab\run_batch.py --module write --n 5
```

## Documentation

- [`docs/AGENT_ARCHITECTURE.md`](docs/AGENT_ARCHITECTURE.md) — the canonical design,
  led by the layer-ownership contract
- [`docs/AUTONOMY-REBUILD.md`](docs/AUTONOMY-REBUILD.md) — the evidence report: live
  gate runs, trace forensics, what's proven and what's honestly still open
- [`scripts/promptlab/RUNBOOK.md`](scripts/promptlab/RUNBOOK.md) — the live validation
  procedure

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE).
