---
name: ollama-phantom-resident-confounds-marginal-ram-load
description: "an Ollama phantom-resident entry (size_vram set, VRAM 0, no backing server) blocks a clean reload and masquerades as RAM-OOM; clear it with `ollama stop` then attempt-and-measure — don't pre-gate live loads on a conservative free-RAM threshold"
metadata:
  node_type: memory
  type: reference
  originSessionId: 2dfcdfd4-b783-4b27-a55e-f178ab1b45d9
---

ReactiveAgents s9 (2026-06-21, DD live leg, native Ollama :11434, E4B gemma4-e4b-candidate-ctx32k). A live E4B load kept failing (silent early-exit / HTTP 000 / VRAM stays 0 / warmup timeouts) across N2r + two DD attempts. It LOOKED like host-RAM-OOM (free RAM had drifted 5.4→3.5GB). It was NOT.

**Root cause = a PHANTOM-RESIDENT Ollama registration.** `/api/ps` showed the model with `size_vram` set and `expires_at` far-future (keep_alive=-1) **but `nvidia-smi` VRAM = 0 and no backing llama-server** = a stuck registration (an earlier keep_alive=-1 run got evicted into this zombie state). It blocks a clean reload, so every subsequent load hangs → reads as OOM. It confounded EVERY prior "load failure."

**Fix (in-discipline, native :11434 ONLY, never the :11435 docker):**
1. `ollama stop gemma4-e4b-candidate-ctx32k:latest` clears the phantom → `/api/ps` becomes `{"models":[]}`.
2. Then **attempt-and-measure** — do NOT pre-gate/refuse the load on a conservative free-RAM threshold. The model's RESIDENT footprint is ~3.19GB (the ~5.3GB figure is DISK size, not required free host RAM), and VRAM was fully free. **Once the phantom was cleared, E4B loaded cleanly in 13s at ~4.13GB free RAM** (size_vram==size 3.19GB, offload=0, VRAM 3677MiB, backing llama-server present) — refuting the ">=6.4GB needed" gate I'd built. The user had said all along "E4B ran fine resident with these same apps at ~5GB free; it only broke when it got evicted into the phantom" — correct.

**Disciplines learned:**
- A `>=N GB free` PRE-GATE that REFUSES to even try is untested-conservative and can mask the real cause — replace "refuse-below-X" with "clear phantom + ONE controlled attempt + measure the EXACT failure signature (OOM-kill vs load-timeout vs early-exit code + exact free-RAM)". One attempt, not a thrash retry-loop (repeated marginal cold-loads are what CREATE the phantom).
- ROOT-FIX for a multi-leg live spine: after a clean load, PIN keep_alive=-1 and REUSE the one resident model across every leg; a per-leg cold reload at the margin re-creates the wedge. See [[smoke-resident-model-starves-detached-launch]], [[full-corpus-extract-eta-and-keepalive-intent]], [[alive-silent-worker-may-be-permission-blocked]] (the parallel "alive+silent ≠ broken" lesson — here alive+silent was a phantom-blocked load, there a permission wait).
- planner shell here CANNOT spawn PowerShell (EPERM) and its bash PATH lacks the `ollama` CLI; it CAN run curl (`/api/ps`), nvidia-smi, wmic/systeminfo. Host ops needing `ollama` must run in a WORKER shell (which has full access). Reap ONLY our own orphan probes, never user apps / foreign PIDs / :11435.
