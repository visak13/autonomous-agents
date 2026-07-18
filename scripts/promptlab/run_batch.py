"""promptlab batch runner — validate ONE module's prompt stack against live gemma.

Usage (from the repo root, venv python):
    ./.venv/Scripts/python.exe scripts/promptlab/run_batch.py --module gather --n 5
    ./.venv/Scripts/python.exe scripts/promptlab/run_batch.py --module write --n 3

The workflow this exists for: edit exactly ONE text layer (identity / role drive /
spec / bundle doctrine / tool description) → run the batch → read the failing runs'
captured THINKING to see why the model chose wrong → iterate. A prompt text ships
only when a batch passes with ZERO failures.

Isolation per batch:
* traces  → var/promptlab/<module>-<ts>/traces (REACTIVE_AGENTS_LOCAL_TRACE_DIR)
* files   → var/promptlab/<module>-<ts>/work   (REACTIVE_AGENTS_WORKSPACE_ROOT)
* data    → var/promptlab/<module>-<ts>/data   (chat_app wiring data dir)
Per RUN, the grader receives only the trace files created during that run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True,
                    choices=["gather", "write", "review", "finish_contract", "plan_author"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    stamp = time.strftime("%m%d-%H%M%S")
    base = Path(args.out) if args.out else ROOT / "var" / "promptlab" / f"{args.module}-{stamp}"
    traces = base / "traces"
    work = base / "work"
    data = base / "data"
    for d in (traces, work, data):
        d.mkdir(parents=True, exist_ok=True)

    # Isolation env BEFORE any package import (both are read at wiring/tool build time).
    os.environ["REACTIVE_AGENTS_LOCAL_TRACE_DIR"] = str(traces)
    os.environ["REACTIVE_AGENTS_WORKSPACE_ROOT"] = str(work)
    os.environ.setdefault("REACTIVE_AGENTS_LIVE", "1")

    for sub in ("chat_app", "agent_runtime", "llm_framework", "reactive_tools",
                "specialization", "memory"):
        sys.path.insert(0, str(ROOT / sub))
    sys.path.insert(0, str(HERE))

    from chat_app.app import build_wiring  # noqa: E402
    from criteria import GRADERS, failing_thinking, load_traces  # noqa: E402
    from modules import MODULES  # noqa: E402

    wiring = build_wiring(data_dir=data)
    run_module = MODULES[args.module]
    grade = GRADERS[args.module]

    results = []
    for i in range(1, args.n + 1):
        before = set(traces.glob("*.json"))
        t0 = time.time()
        try:
            meta = asyncio.run(run_module(wiring, work))
            err = ""
        except Exception as exc:  # noqa: BLE001 — a crashed run is a failed run
            meta, err = {"output": ""}, f"{type(exc).__name__}: {exc}"
        dt = time.time() - t0
        new_traces = sorted(set(traces.glob("*.json")) - before, key=lambda p: p.stat().st_mtime)
        docs = load_traces(new_traces)
        fails = ([err] if err else []) + grade(docs, meta)
        verdict = "PASS" if not fails else "FAIL"
        results.append({
            "run": i, "verdict": verdict, "seconds": round(dt, 1), "failures": fails,
            "traces": [p.name for p in new_traces],
        })
        line = f"[{args.module} {i}/{args.n}] {verdict} ({dt:.0f}s)"
        if fails:
            line += " — " + "; ".join(fails)
        print(line, flush=True)
        if fails:
            think = failing_thinking(docs)
            (base / f"run{i}-thinking.txt").write_text(
                "\n\n=== TURN ===\n\n".join(think), encoding="utf-8")

    passed = sum(1 for r in results if r["verdict"] == "PASS")
    summary = {
        "module": args.module, "n": args.n, "passed": passed,
        "zero_failures": passed == args.n, "results": results, "dir": str(base),
    }
    (base / "scoreboard.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("module", "n", "passed", "zero_failures", "dir")}))
    return 0 if passed == args.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
