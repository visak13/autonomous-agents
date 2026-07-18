# Resume runbook — live validation after GPU unblocks

All code phases (P0–P6) are landed and offline-green. This is the exact remaining
sequence. Every command runs from `C:\Projects\ReactiveAgents` with the venv python.

## 1. Module batches (zero failures ships a prompt text)

~15–25 min each on the 6GB card. On any failure: read
`var/promptlab/<module>-<ts>/runN-thinking.txt` + `scoreboard.json`, edit exactly ONE
text layer (identity / role / spec / doctrine / description — never an engine string),
rerun.

```powershell
./.venv/Scripts/python.exe scripts/promptlab/run_batch.py --module write --n 5
./.venv/Scripts/python.exe scripts/promptlab/run_batch.py --module gather --n 4
./.venv/Scripts/python.exe scripts/promptlab/run_batch.py --module review --n 3
./.venv/Scripts/python.exe scripts/promptlab/run_batch.py --module plan_author --n 3
```

- `write` must hit 5/5 (the lenient tool-call recovery landed after the last 2/5).
- `plan_author` at 3/3 unlocks retiring the per-turn source-id directive
  (`_WRITE_SOURCE_ID_DIRECTIVE` in `chat_app/agentic.py`) — shape-only strategy.

## 2. Restart the app onto this code

```powershell
# stop the PID listening on :8000, then:
& .\launch.ps1
# launch.ps1's uv sync prunes pytest — reinstall for later suite runs:
uv pip install pytest pytest-asyncio pytest-timeout
```

## 3. Live Gate B — full pipeline, twice (two different topics)

```powershell
# per run:
#   POST /chats                       {"title": "gate-b <topic>"}
#   POST /chats/{chat_id}/runs        {"message": "Research <topic> and write me a detailed HTML report on it.", "agentic": true}
#   poll GET /runs/{run_id} until done (30–90 min)
```

Pass bar per run (forensics from the newest `var/traces/*.json`):
- research plan → **briefed review node** (`research_review` node id; planner-authored
  brief; prose output) → `decide_followup` acts on it,
- write plan = the shape's 2-node topology (one worker + one same-spec `final_review`),
- writer self-selects file (+ research_read) and writes real bytes (fresh vs the
  staleness snapshot; `plan_chain.deliverable_bytes` > 0),
- persisted chat turn = summary + artifact card (GET /chats/{id} → final_response),
- all cited URLs present in the trace's fetched sources.

## 4. Trace gate + token report

```powershell
./.venv/Scripts/python.exe scripts/promptlab/trace_assert.py var/traces
# token economy vs var/promptlab/baseline_pre_refactor.json
# (pre-refactor: US-Iran mean 6015 prompt tok/call; Maratha mean 9173) — compute the
# same stats over the Gate-B traces; expect a measurable per-call drop.
```

## 5. Close out

- Update `docs/AUTONOMY-REBUILD.md` addendum "Pending" → results (honest MET/NOT-MET).
- Commit + push.
