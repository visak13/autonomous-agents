"""DAG VALIDITY backstop at finalize (fully OFFLINE).

RP-3b (d311/d319/d328) RETIRED the d28 terminal-research-edge STRUCTURE repair (the
engine used to auto-add a writer<-research edge on a disconnected terminal). The
planner now authors that edge ITSELF (measured 100% reliable on live E4B — see
``.recipe-notes/rp3b_measure.py``); the engine authors NO DAG structure. The retirement
+ the "engine authors nothing" proof live in
``test_rp3b_structure_authoring_retired.py``.

What REMAINS here is the d7 VALIDITY backstop, which is NOT structure-authoring: the
finalize parse goes through :meth:`AbstractPlanFactory.parse_dag_safe`, which REPAIRS an
unresolvable / self ``depends_on`` ref (a phantom-id edge the model may emit) instead of
rejecting it, while still raising (→ self-heal retry-on-reject) for a genuine invalidity
(duplicate id / real cycle). Dropping a phantom edge is graceful degradation of an
INVALID reference, not the engine authoring topology.
"""
from __future__ import annotations

from agent_runtime.factory import AbstractPlanFactory, PlanError
from specialization.registry import SpecRegistry
from specialization.seed import seed_canonical_rulesets

_TOOL_CATALOG = [
    {"name": "web_search", "description": "search the web for candidate pages"},
    {"name": "web_fetch", "description": "fetch and extract a page's article text"},
    {"name": "file_write", "description": "write content to a file"},
]


# --------------------------------------------------------------------------- #
# d7 — finalize repairs a dangling/phantom edge instead of failing
# --------------------------------------------------------------------------- #
def test_d7_parse_dag_safe_repairs_dangling_edge_at_finalize(tmp_path):
    # The finalize path parses via parse_dag_safe, which drops a phantom-id edge and
    # still builds a valid DAG (the d7 backstop). A genuine invalidity still raises
    # (→ self-heal). This is validity degradation, NOT structure authoring.
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)
    factory = AbstractPlanFactory(reg.index(), tool_catalog=_TOOL_CATALOG)
    structured = {
        "nodes": [
            {"id": "n1", "task": "gather", "depends_on": []},
            {"id": "n2", "task": "write", "depends_on": ["n1", "ghost_99"]},  # phantom ref
        ],
        "rationale": "dangling-edge plan",
    }
    dag, repairs = factory.parse_dag_safe(structured)
    assert repairs and "ghost_99" in repairs[0]
    assert dag.by_id["n2"].depends_on == ("n1",)  # phantom dropped, real edge kept

    # A real cycle is NOT silently repaired — finalize would surface it to the
    # self-heal as a malformed plan (retry-on-reject), not ship an invalid DAG.
    cyclic = {
        "nodes": [
            {"id": "n1", "task": "a", "depends_on": ["n2"]},
            {"id": "n2", "task": "b", "depends_on": ["n1"]},
        ],
    }
    try:
        factory.parse_dag_safe(cyclic)
        assert False, "a real cycle must still raise PlanError"
    except PlanError:
        pass
