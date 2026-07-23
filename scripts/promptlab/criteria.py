"""promptlab trace grading — pure functions over the local trace JSONs.

A module's verdict is computed OFFLINE from (a) the trace files its run produced and
(b) the run's workdir artifacts. Every criterion returns a list of failure strings
(empty = pass), so a scoreboard can show exactly WHY a run failed, alongside the
model's captured ``thinking`` for the failing turns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from retired_strings import ENFORCED  # same-dir import (run_batch adds scripts/promptlab to sys.path)


# --------------------------------------------------------------------------- #
# trace loading / walking
# --------------------------------------------------------------------------- #
def load_traces(paths: Iterable[Path]) -> list[dict[str, Any]]:
    docs = []
    for p in paths:
        try:
            docs.append(json.loads(Path(p).read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 — a partial trace is skipped, not fatal
            pass
    return docs


def walk_spans(doc: dict[str, Any]):
    def _walk(spans):
        for sp in spans:
            yield sp
            yield from _walk(sp.get("children", []))
    yield from _walk(doc.get("spans", []))


def chat_captures(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every llm.chat span's llm_capture (messages + response_content + thinking)."""
    out = []
    for doc in docs:
        for sp in walk_spans(doc):
            cap = sp.get("llm_capture")
            if sp.get("name") == "llm.chat" and isinstance(cap, dict):
                out.append(cap)
    return out


def tool_calls(docs: list[dict[str, Any]]) -> list[str]:
    """Tool names in call order, read from the research.turn.N.tool span attrs."""
    calls: list[tuple[int, str]] = []
    for doc in docs:
        for sp in walk_spans(doc):
            for k, v in (sp.get("attributes") or {}).items():
                if k.startswith("research.turn.") and k.endswith(".tool"):
                    try:
                        n = int(k.split(".")[2])
                    except (IndexError, ValueError):
                        continue
                    calls.append((n, str(v)))
    return [t for _, t in sorted(calls)]


def all_prompt_text(docs: list[dict[str, Any]]) -> str:
    parts = []
    for cap in chat_captures(docs):
        for m in cap.get("messages") or []:
            parts.append(str(m.get("content") or ""))
    return "\n".join(parts)


def failing_thinking(docs: list[dict[str, Any]], limit: int = 3) -> list[str]:
    """The LAST few turns' thinking — the diagnostic for a failed run."""
    caps = chat_captures(docs)
    return [str(c.get("thinking") or "")[:2000] for c in caps[-limit:]]


# --------------------------------------------------------------------------- #
# shared criteria
# --------------------------------------------------------------------------- #
def no_enforced_strings(docs) -> list[str]:
    text = all_prompt_text(docs)
    return [f"retired steering string in prompts: {s!r}" for s in ENFORCED if s in text]


# --------------------------------------------------------------------------- #
# per-module verdicts — each returns list[str] failures (empty = pass)
# --------------------------------------------------------------------------- #
def grade_gather(docs, meta: dict[str, Any]) -> list[str]:
    fails = no_enforced_strings(docs)
    calls = tool_calls(docs)
    if calls.count("web_search") < 1:
        fails.append(f"no web_search (calls: {calls})")
    if calls.count("web_fetch") < 1:
        fails.append(f"no web_fetch (calls: {calls})")
    if calls.count("note") < 1:
        fails.append(f"no note recorded (calls: {calls})")
    out = str(meta.get("output") or "")
    if len(out) < 300:
        fails.append(f"findings too thin ({len(out)} chars)")
    # TRACEABILITY is a SYSTEM property (the pull architecture): the note store
    # carries runtime-stamped canonical URLs per read source; downstream pulls them
    # via read_notes/load_source. Inline URLs/[S#] in the prose are the strongest
    # form; name attributions ("Source: X") PASS when the URL-bearing notes exist.
    # No notes AND no inline URL/[S#] = genuinely untraceable output.
    attributed_inline = ("http" in out) or ("[S" in out)
    attributed_by_name = "Source:" in out or "source:" in out
    if not attributed_inline and not (attributed_by_name and calls.count("note") >= 1):
        fails.append("findings not traceable (no inline URL/[S#], and no URL-bearing note backs the name attributions)")
    return fails


def grade_write(docs, meta: dict[str, Any]) -> list[str]:
    fails = no_enforced_strings(docs)
    calls = tool_calls(docs)
    if calls.count("file_write") < 1:
        fails.append(f"no file_write (calls: {calls})")
    target = Path(meta["deliverable"])
    if not target.exists():
        fails.append(f"deliverable missing: {target}")
    else:
        body = target.read_text(encoding="utf-8", errors="replace")
        if len(body) < 800:
            fails.append(f"deliverable thin ({len(body)} bytes)")
        low = body.lower()
        for bad in ("placeholder", "tbd", "[source"):
            if bad in low:
                fails.append(f"placeholder-class filler in deliverable: {bad!r}")
    return fails


def grade_review(docs, meta: dict[str, Any]) -> list[str]:
    fails = no_enforced_strings(docs)
    calls = tool_calls(docs)
    if calls.count("file_read") < 1:
        fails.append(f"reviewer never read the file (calls: {calls})")
    out = str(meta.get("output") or "")
    if len(out) < 80:
        fails.append(f"status too thin ({len(out)} chars)")
    seeded = str(meta.get("seeded_defect") or "")
    fixed = False
    target = Path(meta["deliverable"])
    if seeded and target.exists():
        fixed = seeded not in target.read_text(encoding="utf-8", errors="replace")
    if seeded and not fixed and seeded.lower()[:24] not in out.lower():
        fails.append("seeded defect neither fixed (file unchanged) nor named in the status")
    return fails


def grade_finish_contract(docs, meta: dict[str, Any]) -> list[str]:
    fails = no_enforced_strings(docs)
    out = str(meta.get("output") or "")
    if not out.strip():
        fails.append("no output produced")
    if out.strip().startswith("{"):
        fails.append("output leaked tool-call JSON instead of prose")
    return fails


def grade_plan_author(docs, meta: dict[str, Any]) -> list[str]:
    fails = no_enforced_strings(docs)
    dag = meta.get("dag")
    if dag is None:
        fails.append("no DAG authored")
        return fails
    nodes = list(getattr(dag, "nodes", []))
    workers = [n for n in nodes if (n.role or "worker") != "reviewer"]
    reviewers = [n for n in nodes if (n.role or "") == "reviewer"]
    if len(workers) != 1:
        fails.append(f"expected exactly 1 write node, got {len(workers)}")
    if len(reviewers) != 1:
        fails.append(f"expected exactly 1 reviewer node, got {len(reviewers)}")
    if workers and not getattr(workers[0], "source_ids", None):
        fails.append("write node has empty source_ids")
    if workers and reviewers:
        wspecs = set(workers[0].specs or ([workers[0].spec] if workers[0].spec else []))
        rspecs = set(reviewers[0].specs or ([reviewers[0].spec] if reviewers[0].spec else []))
        if wspecs and not (wspecs & rspecs):
            fails.append(f"reviewer not bound to the writer's spec ({wspecs} vs {rspecs})")
    return fails


GRADERS = {
    "gather": grade_gather,
    "write": grade_write,
    "review": grade_review,
    "finish_contract": grade_finish_contract,
    "plan_author": grade_plan_author,
}
