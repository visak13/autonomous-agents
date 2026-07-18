"""promptlab MODULES — isolated node-level scenarios on the REAL served seams.

Each module is an async ``run(wiring, workdir) -> meta`` that drives ONE small
scenario through the real building blocks (live OllamaTransport, the real unified
worker loop via a one-node PlanDAG on ``_build_acyclic_runtime``'s AgentRuntime,
real bundles + hook). The batch runner isolates traces per run and hands the trace
docs + ``meta`` to the module's grader in ``criteria.py``.

A module is the smallest unit a prompt-text change can be validated against:
edit ONE text layer (identity / role / spec / doctrine / description) → rerun the
batch → zero failures ships it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agent_runtime.factory import PlanDAG, PlanNode


def _one_node_runtime(wiring, *, emit_article_notes: bool = True):
    """The served acyclic runtime (same lifecycle gate, self-heal, tool surface)."""
    from chat_app.agentic import _build_acyclic_runtime

    runtime, _planner = _build_acyclic_runtime(
        transport=wiring.runtime.transport,
        registry=wiring.registry,
        hook=wiring.hook,
        plane=wiring.plane,
        shape_spec=None,
        emit_article_notes=emit_article_notes,
    )
    return runtime


CANNED_SOURCES = [
    {
        "title": "Great Fire of London — contemporary account",
        "url": "https://example.org/great-fire-1666",
        "markdown": (
            "# The Great Fire of London (1666)\n\nThe fire began on 2 September 1666 "
            "in a bakery on Pudding Lane and burned for four days, destroying 13,200 "
            "houses and 87 parish churches, including St Paul's Cathedral. An estimated "
            "70,000 of the City's 80,000 inhabitants were made homeless. The rebuilding "
            "took over 30 years; Christopher Wren designed 51 new churches."
        ),
    },
    {
        "title": "Rebuilding acts and urban reform after 1666",
        "url": "https://example.org/rebuilding-acts",
        "markdown": (
            "# Rebuilding of London\n\nThe Rebuilding of London Act 1666 mandated brick "
            "and stone construction and wider streets. Insurance emerged as an industry: "
            "the first fire insurance company, the 'Fire Office', opened in 1681. The "
            "Monument to the Great Fire, 61 metres tall, was completed in 1677."
        ),
    },
]

CANNED_NOTES = [
    {"url": "https://example.org/great-fire-1666", "summary": "origin, scale, casualties of the 1666 fire",
     "key_claims": ["started 2 Sept 1666 Pudding Lane", "13,200 houses destroyed", "70,000 homeless"],
     "gaps_or_followups": ["economic cost figures"]},
    {"url": "https://example.org/rebuilding-acts", "summary": "rebuilding acts, insurance industry birth, Monument",
     "key_claims": ["Rebuilding Act 1666 brick/stone", "first fire insurance 1681", "Monument 1677, 61m"],
     "gaps_or_followups": []},
]


async def run_gather(wiring, workdir: Path) -> dict[str, Any]:
    """One worker with the research specs gathers live web evidence on a fixed topic."""
    runtime = _one_node_runtime(wiring, emit_article_notes=True)
    node = PlanNode(
        id="pl_gather",
        task=("Gather grounded evidence on the current status of the Artemis moon "
              "program: latest mission flown, next planned mission and its date, and "
              "the main schedule risks."),
        role="worker",
        specs=("research-methodology", "research-analyst"),
    )
    res = await runtime.run(PlanDAG(nodes=[node], goal=node.task), timeout=900.0,
                            run_id=f"promptlab-gather-{int(time.time())}")
    out = res.results.get(node.id)
    return {"output": getattr(out, "output", "") if out else "", "ok": res.ok}


async def run_write(wiring, workdir: Path) -> dict[str, Any]:
    """One worker writes an HTML report grounded in a canned source index."""
    runtime = _one_node_runtime(wiring, emit_article_notes=False)
    deliverable = "promptlab-report.html"
    runtime.chain_sources = CANNED_SOURCES
    runtime.chain_notes = CANNED_NOTES
    runtime.deliverable_path = deliverable
    node = PlanNode(
        id="pl_write",
        task=("Write a concise HTML report on the Great Fire of London and its "
              "aftermath, grounded in the research memory. "
              f"Write to the file '{deliverable}'."),
        role="worker",
        specs=("html-writer",),
        source_ids=(1, 2),
        # The [S#] index renders into the brief only for a memory-bound node — the
        # same binding the served write phase stamps (a live batch caught its absence:
        # the model was told 'ground in the research memory' while holding nothing,
        # and reasonably went to the web instead of writing).
        research_memory_handle="promptlab-canned",
    )
    res = await runtime.run(PlanDAG(nodes=[node], goal=node.task), timeout=900.0,
                            run_id=f"promptlab-write-{int(time.time())}")
    out = res.results.get(node.id)
    from reactive_tools.file_tools import resolve_workspace_root
    return {
        "output": getattr(out, "output", "") if out else "",
        "ok": res.ok,
        "deliverable": str(resolve_workspace_root() / deliverable),
    }


_FLAWED_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Great Fire of London</title></head>
<body>
<h1>The Great Fire of London</h1>
<p>The fire began on 2 September 1666 in a bakery on Pudding Lane.</p>
<p>It destroyed 99,999 houses and made 70,000 people homeless.</p>
<h2>Sources</h2>
<ul><li><a href="https://example.org/great-fire-1666">Contemporary account</a></li></ul>
</body></html>
"""


async def run_review(wiring, workdir: Path) -> dict[str, Any]:
    """A reviewer node inspects an on-disk artifact with a seeded factual defect."""
    from reactive_tools.file_tools import resolve_workspace_root

    deliverable = "promptlab-review.html"
    target = resolve_workspace_root() / deliverable
    target.write_text(_FLAWED_HTML, encoding="utf-8")

    runtime = _one_node_runtime(wiring, emit_article_notes=False)
    runtime.chain_sources = CANNED_SOURCES
    runtime.chain_notes = CANNED_NOTES
    runtime.deliverable_path = deliverable
    node = PlanNode(
        id="pl_review",
        task=(f"Review the finished report in the file '{deliverable}' against the "
              "research memory: verify every figure against the sources, fix any "
              "defect in place, and report an honest final status."),
        role="reviewer",
        specs=("html-writer",),
        source_ids=(1, 2),
        research_memory_handle="promptlab-canned",
    )
    res = await runtime.run(PlanDAG(nodes=[node], goal=node.task), timeout=900.0,
                            run_id=f"promptlab-review-{int(time.time())}")
    out = res.results.get(node.id)
    return {
        "output": getattr(out, "output", "") if out else "",
        "ok": res.ok,
        "deliverable": str(target),
        "seeded_defect": "99,999 houses",
    }


async def run_finish_contract(wiring, workdir: Path) -> dict[str, Any]:
    """A trivial worker answers in prose — no spurious tool use, no JSON leak."""
    runtime = _one_node_runtime(wiring, emit_article_notes=False)
    node = PlanNode(
        id="pl_trivial",
        task="Give three practical tips for writing clear git commit messages.",
        role="worker",
    )
    res = await runtime.run(PlanDAG(nodes=[node], goal=node.task), timeout=300.0,
                            run_id=f"promptlab-finish-{int(time.time())}")
    out = res.results.get(node.id)
    return {"output": getattr(out, "output", "") if out else "", "ok": res.ok}


async def run_plan_author(wiring, workdir: Path) -> dict[str, Any]:
    """The WRITE planner authors a plan for a file deliverable (authoring ONLY —
    mirrors the served two-drive composition; no nodes are driven)."""
    import chat_app.agentic as ag

    query = "Write a detailed HTML report on the Great Fire of London."
    out_name = "promptlab-plan.html"
    findings = ("The fire of 1666: began 2 Sept in Pudding Lane; 13,200 houses "
                "destroyed; 70,000 homeless; Rebuilding Act 1666; first fire "
                "insurance 1681; Monument completed 1677.")
    write_goal, prior_memory, write_directive = ag._compose_write_planner_inputs(
        query, out_name, findings, CANNED_SOURCES,
    )
    # The served directive: the shape methodology owns strategy post-P5; the engine
    # constant (pre-P5) is composed when it still exists — the module always mirrors
    # the CURRENT served path.
    one_node = getattr(ag, "_ONE_WRITE_NODE_DIRECTIVE", "")
    directive = (one_node + "\n\n" + write_directive).strip()
    planner = ag._build_incremental_planner(
        transport=wiring.runtime.transport, registry=wiring.registry,
        hook=wiring.hook, shape_spec=ag._WRITE_FILE_SHAPE, allow_web=False,
        authoring_directive=directive,
    )
    plan = await planner.plan(write_goal, prior_memory=prior_memory)
    dag = ag._normalize_write_dag(plan.dag, out_name)
    return {"dag": dag, "output": ""}


MODULES = {
    "gather": run_gather,
    "write": run_write,
    "review": run_review,
    "finish_contract": run_finish_contract,
    "plan_author": run_plan_author,
}
