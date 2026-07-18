"""promptlab live-trace gate — no retired steering string in any prompt/observation.

Usage:
    ./.venv/Scripts/python.exe scripts/promptlab/trace_assert.py var/traces
    ./.venv/Scripts/python.exe scripts/promptlab/trace_assert.py <trace.json> [...]

Walks every llm.chat span's captured messages in the given trace file(s)/dir and
fails (exit 1) on any ENFORCED string from ``retired_strings.py`` — the same single
registry the source grep-gate test uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from criteria import all_prompt_text, load_traces  # noqa: E402
from retired_strings import ENFORCED  # noqa: E402


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    paths: list[Path] = []
    for a in argv:
        p = Path(a)
        paths.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    docs = load_traces(paths)
    text = all_prompt_text(docs)
    hits = [s for s in ENFORCED if s in text]
    if hits:
        print(f"FAIL — retired steering string(s) in {len(docs)} trace(s):")
        for s in hits:
            print(f"  {s!r}")
        return 1
    print(f"OK — {len(docs)} trace(s), {len(ENFORCED)} enforced strings, zero hits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
