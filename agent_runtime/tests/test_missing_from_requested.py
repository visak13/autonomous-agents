"""s10-a8 — the STRUCTURAL scenario-3 missing-specialist trigger (fully OFFLINE).

``missing_from_requested`` is the pure, model-free heart of the re-architected
scenario-3 trigger: the shape selector reliably EXTRACTS the specialization name(s)
the user asked for, and the TRIGGER is a DETERMINISTIC registry-membership check —
a requested name that is not registered is a missing specialist, attached to the
DAG's SINK node(s) so a define-and-resume stamps the newly-defined spec where the
answer is produced. This replaces the per-node ``needs_spec`` free-text the 4.6B
model would not volunteer (s10-a4). The mechanism is generic (a set-membership
check) — no scenario/keyword/topic matching anywhere.
"""
from __future__ import annotations

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.missing_spec import (
    CHOICE_DEFINE_AND_RESUME,
    CHOICE_SSE_FALLBACK,
    apply_resolution,
    missing_from_requested,
)


def _research_then_write() -> PlanDAG:
    """A research → write chain; n2 (the writer) is the SINK (nothing depends on it)."""
    return PlanDAG(
        nodes=[
            PlanNode(id="n1", task="research the filing", tool="web_search"),
            PlanNode(id="n2", task="write the report", depends_on=("n1",)),
        ]
    )


def test_unregistered_request_is_flagged_on_the_sink():
    dag = _research_then_write()
    missing = missing_from_requested(
        dag, ["forensic-accountant"], registered=["markdown-writer", "research-analyst"]
    )
    # attached to the SINK (the terminal writer node), not the upstream research node.
    assert [m.node_id for m in missing] == ["n2"]
    assert missing[0].needs == "forensic-accountant"


def test_registered_request_is_not_flagged():
    # The common case: every requested spec IS registered → no missing-spec notify.
    dag = _research_then_write()
    missing = missing_from_requested(
        dag, ["markdown-writer"], registered=["markdown-writer", "research-analyst"]
    )
    assert missing == []


def test_mixed_keeps_only_the_unregistered_ones_deduped():
    dag = _research_then_write()
    missing = missing_from_requested(
        dag,
        ["markdown-writer", "forensic-accountant", " forensic-accountant ", "legal-brief"],
        registered=["markdown-writer"],
    )
    assert len(missing) == 1  # one sink
    # the registered one is dropped; the unregistered ones dedup + join in order.
    assert missing[0].needs == "forensic-accountant, legal-brief"


def test_empty_when_no_request():
    dag = _research_then_write()
    assert missing_from_requested(dag, [], registered=["markdown-writer"]) == []


def test_all_sink_nodes_flagged_on_a_fan_out():
    # Two independent terminal nodes (a fan-out with no combine) → both are sinks and
    # both carry the unmet need, so a resume resolves every answer-producing node.
    dag = PlanDAG(
        nodes=[
            PlanNode(id="root", task="gather inputs", tool="web_search"),
            PlanNode(id="a", task="answer part A", depends_on=("root",)),
            PlanNode(id="b", task="answer part B", depends_on=("root",)),
        ]
    )
    missing = missing_from_requested(dag, ["legal-brief"], registered=[])
    assert sorted(m.node_id for m in missing) == ["a", "b"]


def test_resolution_round_trips_through_the_existing_mechanism():
    # The synthesized entries feed the UNCHANGED apply_resolution path: sse_fallback
    # clears the need (runs spec-less), define_and_resume stamps the now-defined spec
    # onto the sink. Only the TRIGGER is new; the two resolutions are reused verbatim.
    dag = _research_then_write()
    missing = missing_from_requested(dag, ["forensic-accountant"], registered=[])

    fb = apply_resolution(dag, missing, choice=CHOICE_SSE_FALLBACK)
    assert fb.by_id["n2"].needs_spec is None
    assert fb.by_id["n2"].effective_specs == ()

    dr = apply_resolution(
        dag, missing, choice=CHOICE_DEFINE_AND_RESUME,
        defined_specs={"": "forensic-accountant"},
    )
    assert "forensic-accountant" in dr.by_id["n2"].effective_specs
    assert dr.by_id["n2"].needs_spec is None
