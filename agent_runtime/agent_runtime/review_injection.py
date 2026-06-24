"""FRAMEWORK-INJECTED REVIEW — work=>work+review, finalize=>final-review (P2.2, d129.3/d132.B).

The user's orchestration model (d127/d128/d129) puts review injection in the
FRAMEWORK, not the planner: the planner only SEEDS the basic shape plan and fills
CONTEXT + per-node SPECIALIZATIONS; the framework is what turns each authored WORK
step into a ``work -> review`` pair and appends a FINAL review when the plan is
finalized. The reviewer node IDENTIFIES gaps AND FIXES them (it is an ordinary
worker node with the file read/write/update tools available at run time) and is
SPEC-AWARE — it inherits the work node's effective specializations so it applies the
SAME output-shaping ruleset when it corrects the deliverable.

This module is a PURE DAG transform (no model call, no I/O, d2) over the structured
plan dict the planner authors (``{rationale, nodes, shape}`` — the shape
:meth:`agent_runtime.plan_tools.PlanBuilder.to_structured` exports and
:meth:`agent_runtime.factory.AbstractPlanFactory.parse_dag` consumes). It is the
framework seam :class:`PlanBuilder` opts into (``inject_review=True``), kept here so
it is independently testable and never entangled with the authoring loop.

The two canonical flows (d129.3) fall out of one rule — insert a review after every
work node (re-pointing that node's consumers onto the review) and append one final
review over the resulting sinks:

* PARALLEL  ``n1, n2 (||), n3 (joins n1,n2)``
  => ``(n1->n1_review) || (n2->n2_review) -> n3(joins the reviews) -> n3_review -> final``
* LINEAR    ``n1 -> n2 -> n3``
  => ``n1 -> n1_review -> n2 -> n2_review -> n3 -> n3_review -> final``

Synthesizer fold (P2-2-foldverify constraints): a review node is a plain ``worker``
role, NOT a judgment role — it carries no verdict schema; its job is to read the
deliverable and return CORRECTED RAW output. The terminal that emits the deliverable
stays on the content-emission path; the review that follows it never re-frames it as
a ``{verdict, findings}`` envelope (constraints 1 & 2 of the foldverify gate).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

# Marker suffix the framework appends to a work node's id to name its review node.
# Also the idempotency guard: a node whose id already ends with this is itself a
# review node, so re-running the transform never reviews a review (no runaway).
REVIEW_SUFFIX = "_review"
FINAL_REVIEW_ID = "final_review"

# The review posture, injected as the review node's TASK text. It is a WORKER
# instruction (read the deliverable, find gaps, FIX them in place), NOT a judgment
# prompt — so the node returns corrected raw content, never a verdict envelope. The
# file read/write/update tools the reviewer uses are offered at run time by the
# runtime's tool surface; the task only has to direct the model to use them.
_REVIEW_TASK = (
    "Review the output of the step '{task}'. Read the actual produced content "
    "(read the file back if it was written to a file), identify any gaps, errors, "
    "missing requirements or unsupported claims against the goal, and FIX them in "
    "place by editing the content/file directly. Return the corrected deliverable "
    "as raw content — do not wrap it in a verdict or findings object."
)
_FINAL_REVIEW_TASK = (
    "Final review of the whole plan's deliverable. Read the actual final output "
    "(read the file back if one was written), check it fully meets the goal end to "
    "end, and FIX any remaining gaps in place. Return the corrected final "
    "deliverable as raw content — never a verdict or findings object."
)


def _effective_specs(node: Mapping[str, Any]) -> list[str]:
    """The work node's specs, for the SPEC-AWARE review (inherit the same ruleset)."""
    specs = node.get("specs") or []
    if isinstance(specs, (list, tuple)) and specs:
        return [str(s) for s in specs if str(s).strip()]
    spec = node.get("spec")
    return [str(spec)] if spec and str(spec).strip() else []


def _is_review_id(node_id: str) -> bool:
    return str(node_id).endswith(REVIEW_SUFFIX) or str(node_id) == FINAL_REVIEW_ID


def _review_node(work: Mapping[str, Any]) -> dict[str, Any]:
    """Build the review node for one WORK node (spec-aware, plain worker role)."""
    specs = _effective_specs(work)
    return {
        "id": f"{work['id']}{REVIEW_SUFFIX}",
        "task": _REVIEW_TASK.format(task=str(work.get("task") or work["id"])),
        # SPEC-AWARE: the reviewer applies the SAME output-shaping ruleset as the
        # work node it reviews, so a corrected HTML report stays HTML, etc.
        "spec": specs[0] if specs else None,
        "specs": list(specs),
        "tool": None,
        "needs_spec": None,
        "depends_on": [work["id"]],
        # WORKER, never a judgment role (foldverify constraint 2): the review
        # returns corrected RAW content, not a verdict enum.
        "role": "worker",
        "source_ids": list(work.get("source_ids") or []),
        # A non-schema annotation the runtime/trace can read to recognise an
        # injected review (ignored by parse_dag, which reads only the known keys).
        "review_of": work["id"],
    }


def inject_reviews(
    structured: Mapping[str, Any],
    *,
    add_final: bool = True,
    final_specs: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Return ``structured`` with a framework-injected review after every work node.

    Pure transform: for each ORIGINAL (non-review) node W, inserts ``W_review``
    depending on W and RE-POINTS every consumer that depended on W onto ``W_review``
    (so the review sits in the path: work -> review -> consumers). When ``add_final``
    is set, appends ONE :data:`FINAL_REVIEW_ID` node over the resulting sink reviews
    (the "finalize => final review" rule). Idempotent: nodes already named as reviews
    are passed through untouched, so applying the transform twice is a no-op beyond
    the first.

    ``final_specs`` overrides the final review's inherited specs (default: the specs
    of the sink work nodes it reviews, so the final pass is spec-aware too)."""
    nodes: list[dict[str, Any]] = [dict(n) for n in structured.get("nodes", [])]
    if not nodes:
        return {**dict(structured), "nodes": nodes}

    # IDEMPOTENCY: if the plan already carries any injected review node (a per-work
    # review or the final review), it was already transformed — return it unchanged
    # so applying the framework injection twice is a no-op (no duplicate reviews).
    if any(_is_review_id(n["id"]) for n in nodes):
        return {**dict(structured), "nodes": nodes}

    work_nodes = [n for n in nodes if not _is_review_id(n["id"])]

    # id -> its review id, for re-pointing consumers.
    review_of: dict[str, str] = {n["id"]: f"{n['id']}{REVIEW_SUFFIX}" for n in work_nodes}

    # 1) RE-POINT: a consumer that depended on a work node W now depends on W_review,
    #    inserting the review between the work and everything downstream of it.
    for n in work_nodes:
        n["depends_on"] = [review_of.get(d, d) for d in (n.get("depends_on") or [])]

    # 2) Build the per-work review nodes (depends_on = [W]).
    reviews = [_review_node(n) for n in work_nodes]

    augmented: list[dict[str, Any]] = work_nodes + reviews

    if add_final:
        # 3) FINAL review over the SINKS (nodes nothing depends on). After
        #    re-pointing, the sinks are the terminal review nodes.
        depended_on: set[str] = set()
        for n in augmented:
            depended_on.update(n.get("depends_on") or [])
        sinks = [n["id"] for n in augmented if n["id"] not in depended_on]
        # Inherit the sink work nodes' specs unless overridden (spec-aware final pass).
        if final_specs is not None:
            fspecs = [str(s) for s in final_specs if str(s).strip()]
        else:
            fspecs = []
            for r in reviews:
                if r["id"] in sinks:
                    for s in r.get("specs") or []:
                        if s not in fspecs:
                            fspecs.append(s)
        augmented.append({
            "id": FINAL_REVIEW_ID,
            "task": _FINAL_REVIEW_TASK,
            "spec": fspecs[0] if fspecs else None,
            "specs": list(fspecs),
            "tool": None,
            "needs_spec": None,
            "depends_on": sinks,
            "role": "worker",
            "source_ids": [],
            "review_of": "*",
        })

    return {**dict(structured), "nodes": augmented}


__all__ = [
    "inject_reviews",
    "REVIEW_SUFFIX",
    "FINAL_REVIEW_ID",
]
