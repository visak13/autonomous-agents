"""P2.5 — the HARD PARITY GATE for the generic-engine report-path consolidation.

HISTORICAL GATE (now satisfied): this gate governed the P2.5/P2.5b swap from the bespoke
``run_research_tree`` layer loop to the GENERIC declarative-unroll + AgentRuntime growable
engine on the flagship report path. The gate HELD (within-run, same-budget, generic breadth
>= tree, grounded), so P2-5c made the generic engine the served DEFAULT and RETIRED the
bespoke ``run_research_tree`` orchestrator + the reversible ``RA_GENERIC_REPORT_PATH``
scaffolding flag (flag-free end-state, d65). The criteria the gate measured on the live
US-Iran "detailed HTML report" query were:

  * BREADTH  — at least Phase-1's bar of >= 3 scoped sources/children, AND not fewer
               sources than the tree on the same budget;
  * NO DUP-TAIL — exactly one top-level document, one <h1>, no concatenated second doc;
  * GROUNDING — real fetched sources present (no zero-source fabrication);
  * DETAIL — warcosts-parity: at least ~80% of the tree's document size.

This module is PURE + side-effect-free and is KEPT as the offline-unit-testable parity-gate
logic (the one-off generic-vs-bespoke harness that drove it is retired with the bespoke
engine). The recorded live parity result lives in ``.s13_design/p2_5_parity.json``."""
from __future__ import annotations

import re
from typing import Any, Mapping, Optional

# Phase-1's proven breadth bar: >= 3 scoped children / sources (d122/d123).
PHASE1_BREADTH_BAR = 3
# warcosts-parity detail floor: the generic document must be at least this fraction of
# the tree document's size to count as "at least as detailed" (d115 parity bar).
DETAIL_FLOOR_FRACTION = 0.8


def parity_metrics(result: Any, *, document: str = "") -> dict[str, Any]:
    """Extract the parity-relevant metrics from one engine's run.

    ``result`` is the :class:`~chat_app.agentic.AgenticResult` (its ``deep_research`` trace
    carries the engine marker, the deduped source count and the research node count);
    ``document`` is the served report text (read from the written file) used for the
    dup-tail + detail measures. Pure: no I/O, so the same function scores either engine."""
    dr: Mapping[str, Any] = getattr(result, "deep_research", None) or {}
    doc = document or ""

    sources = int(dr.get("sources", 0) or 0)
    research_nodes = int(dr.get("rounds_executed", 0) or 0)

    # DUP-TAIL: a single well-formed document has exactly one DOCTYPE/<html>/<h1>. More than
    # one of any signals the duplicate-tail / concatenated-second-document defect.
    doctypes = len(re.findall(r"<!doctype", doc, re.IGNORECASE))
    html_open = len(re.findall(r"<html\b", doc, re.IGNORECASE))
    h1_count = len(re.findall(r"<h1\b", doc, re.IGNORECASE)) + len(re.findall(r"(?m)^#\s", doc))
    dup_tail = doctypes > 1 or html_open > 1 or h1_count > 1

    # DETAIL: heading count + raw size (warcosts-parity proxy).
    headings = len(re.findall(r"<h[1-6]\b", doc, re.IGNORECASE)) + len(
        re.findall(r"(?m)^#{1,6}\s", doc)
    )
    chars = len(doc)

    return {
        "engine": dr.get("engine"),
        "sources": sources,
        "research_nodes": research_nodes,
        "h1_count": h1_count,
        "dup_tail": dup_tail,
        "headings": headings,
        "chars": chars,
        "grounded": sources > 0 and bool(doc.strip()),
    }


def parity_verdict(
    tree: Mapping[str, Any],
    generic: Mapping[str, Any],
    *,
    breadth_bar: int = PHASE1_BREADTH_BAR,
    detail_floor: float = DETAIL_FLOOR_FRACTION,
) -> dict[str, Any]:
    """Apply the HARD PARITY GATE to the two engines' :func:`parity_metrics`.

    The generic engine must clear EVERY sub-gate to be eligible to replace the tree:
    the >= Phase-1 breadth bar, not-fewer-sources-than-the-tree, no dup-tail, grounded,
    and warcosts-parity detail. Returns the per-gate booleans + an overall ``parity_holds``
    and a recommendation. ``parity_holds=False`` means KEEP the flag OFF (tree stays served)
    and report the gap — never ship a regression (d132.E / d133)."""
    breadth_meets_bar = int(generic.get("sources", 0)) >= breadth_bar
    breadth_vs_tree = int(generic.get("sources", 0)) >= int(tree.get("sources", 0))
    no_dup_tail = not bool(generic.get("dup_tail", False))
    grounded = bool(generic.get("grounded", False))
    tree_chars = max(1, int(tree.get("chars", 0)))
    detail_ok = int(generic.get("chars", 0)) >= detail_floor * tree_chars

    holds = bool(
        breadth_meets_bar and breadth_vs_tree and no_dup_tail and grounded and detail_ok
    )
    return {
        "parity_holds": holds,
        "gates": {
            "breadth_meets_phase1_bar": breadth_meets_bar,
            "breadth_ge_tree": breadth_vs_tree,
            "no_dup_tail": no_dup_tail,
            "grounded": grounded,
            "detail_ge_floor": detail_ok,
        },
        "tree": dict(tree),
        "generic": dict(generic),
        "recommendation": (
            "PARITY HOLDS — the served generic growable engine meets the breadth/grounding/"
            "single-document bar (the bespoke engine is retired; this is now a regression check)."
            if holds
            else "PARITY DOES NOT HOLD — the served generic engine REGRESSED on this run; "
            "investigate and FIX the generic path (the bespoke engine is retired — there is no "
            "fallback to flip back to). Report the gap (do not ship a regression)."
        ),
    }
