"""CoT-autonomy self-policing: retired ENGINE STEERING strings stay deleted.

The agent's own chain of thought drives the work. Engine/tool text may deliver DATA
(counts, caps, cursors, error facts, result payloads) and one-time KNOWLEDGE (bundle
doctrine, tool descriptions) — it may never COMMAND the model's next action. Every
next-action string retired by the refactor is registered in
``scripts/promptlab/retired_strings.py`` (the single registry, shared with the live
trace gate) and this test greps all five package source trees for regressions.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_REG = ROOT / "scripts" / "promptlab" / "retired_strings.py"
_spec = importlib.util.spec_from_file_location("retired_strings", _REG)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ENFORCED: list[str] = _mod.ENFORCED
PENDING: list[str] = _mod.PENDING

# The five package SOURCE trees (tests excluded — assertions may quote old strings).
SOURCE_TREES = [
    ROOT / "agent_runtime" / "agent_runtime",
    ROOT / "chat_app" / "chat_app",
    ROOT / "reactive_tools" / "reactive_tools",
    ROOT / "llm_framework" / "llm_framework",
    ROOT / "specialization" / "specialization",
]


def _iter_source():
    for tree in SOURCE_TREES:
        for p in tree.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            yield p, p.read_text(encoding="utf-8", errors="replace")
        # shape TOML files are model-facing text too
        for p in tree.rglob("*.toml"):
            yield p, p.read_text(encoding="utf-8", errors="replace")


def test_registry_is_wellformed():
    assert isinstance(ENFORCED, list) and isinstance(PENDING, list)
    dup = set(ENFORCED) & set(PENDING)
    assert not dup, f"strings in BOTH lists: {dup}"
    for s in ENFORCED + PENDING:
        assert isinstance(s, str) and len(s) >= 8, f"too-short/ambiguous entry: {s!r}"


def test_no_enforced_steering_string_in_package_source():
    hits: list[str] = []
    for path, text in _iter_source():
        for s in ENFORCED:
            if s in text:
                hits.append(f"{path.relative_to(ROOT)}: {s!r}")
    assert not hits, (
        "retired steering string(s) reappeared in package source:\n" + "\n".join(hits)
    )


def test_pending_strings_still_present_somewhere():
    """A PENDING entry that no longer matches any source is stale — it must be MOVED to
    ENFORCED (its phase landed) or corrected. Keeps the registry honest both ways."""
    remaining = set(PENDING)
    for _, text in _iter_source():
        remaining = {s for s in remaining if s not in text}
        if not remaining:
            return
    assert not remaining, (
        "PENDING entries match nothing — move to ENFORCED or fix: " + repr(sorted(remaining))
    )
