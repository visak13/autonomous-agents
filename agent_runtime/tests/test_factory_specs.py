"""Planner-PARSE + body-free coverage for the a4 N-spec field (s3/Stage-B, d10/d11).

a4 added the ``specs`` key to ``NODE_SCHEMA`` and taught ``AbstractPlanFactory.
parse_dag`` to read it. test_compose / test_collision drive the RUNTIME composition
seam, but the PLANNER-layer parse of ``specs`` (and the d10 body-free guarantee with
``specs`` present) had no direct test. This locks it:

- ``parse_dag`` turns a node's ``specs`` list (or a bare string) into the ordered
  ``effective_specs``; the scalar ``spec`` still works (single-spec back-compat); a
  bare node has no specs.
- The composition ORDER is exactly the emitted list order (not alphabetised).
- The planner context/prompt the factory builds stays BODY-FREE even though the
  schema now advertises ``specs`` — the planner carries only NAMES; a body never
  enters (d10). The ``assert_body_free`` guard and the index body-guard still bite.

Pure, in-process, offline — no transport/loader/GPU: this is the factory/parser
surface only.
"""
from __future__ import annotations

import pytest

from agent_runtime.factory import (
    NODE_SCHEMA,
    AbstractPlanFactory,
    PlanError,
    PlanNode,
)

_INDEX = [
    {"name": "markdown-writer", "description": "shape findings into GFM", "source": "seed"},
    {"name": "brevity-editor", "description": "tighten the answer", "source": "seed"},
]


def _factory() -> AbstractPlanFactory:
    return AbstractPlanFactory(_INDEX)


# --------------------------------------------------------------------------- #
# parse_dag reads `specs` (list or bare str); scalar `spec` is back-compat.
# --------------------------------------------------------------------------- #
def test_parse_dag_reads_specs_list_in_order():
    dag = _factory().parse_dag(
        {
            "rationale": "r",
            "nodes": [
                {"id": "n1", "task": "research", "specs": ["markdown-writer", "brevity-editor"]},
                {"id": "n2", "task": "fmt", "specs": "markdown-writer", "depends_on": ["n1"]},
                {"id": "n3", "task": "plain", "spec": "brevity-editor"},
                {"id": "n4", "task": "bare"},
            ],
        }
    )
    by = dag.by_id
    # a list → the ordered effective_specs, list order PRESERVED (not alphabetised).
    assert by["n1"].effective_specs == ("markdown-writer", "brevity-editor")
    assert by["n1"].primary_spec == "markdown-writer"
    # a bare string `specs` is normalised to a one-tuple.
    assert by["n2"].effective_specs == ("markdown-writer",)
    # the scalar `spec` still resolves (single-spec back-compat) ...
    assert by["n3"].effective_specs == ("brevity-editor",)
    assert by["n3"].specs == ()  # scalar form leaves the specs tuple empty
    # ... and a bare node carries nothing.
    assert by["n4"].effective_specs == ()
    assert by["n4"].primary_spec is None


def test_parse_dag_specs_order_is_not_alphabetised():
    # Emit the names in non-alphabetical order; the parse must keep THAT order so
    # the runtime layers them as the planner intended (deterministic, d2/d11).
    dag = _factory().parse_dag(
        {"nodes": [{"id": "n1", "task": "t", "specs": ["brevity-editor", "markdown-writer"]}]}
    )
    assert dag.by_id["n1"].effective_specs == ("brevity-editor", "markdown-writer")


def test_parse_dag_drops_blank_spec_names():
    # Stray blank/empty entries must not become phantom specs (PlanNode cleans them).
    dag = _factory().parse_dag(
        {"nodes": [{"id": "n1", "task": "t", "specs": ["markdown-writer", "", "  "]}]}
    )
    assert dag.by_id["n1"].effective_specs == ("markdown-writer",)


def test_node_schema_advertises_specs():
    # The planner is TOLD it may emit `specs` (names only) — and the description
    # makes clear it carries names, not bodies (d10).
    assert "specs" in NODE_SCHEMA
    assert "name" in NODE_SCHEMA["specs"].lower()


# --------------------------------------------------------------------------- #
# d10: the planner layer stays BODY-FREE even with `specs` in the schema.
# --------------------------------------------------------------------------- #
def test_planner_context_is_body_free_with_specs_schema():
    f = _factory()
    ctx = f.planner_context("research a live topic and format it")
    # No exception = no 'body' anywhere in the planner payload (d10).
    f.assert_body_free(ctx)
    # The lookup the planner sees carries names/descriptions only — never a body.
    for row in ctx["specializations"]:
        assert set(row) <= {"name", "description", "source"}
        assert "body" not in row
    # The advertised node schema (which now includes `specs`) is itself body-free.
    f.assert_body_free({"node_schema": ctx["factory"]["node_schema"]})


def test_planner_prompt_carries_specs_key_but_no_body():
    system, user = _factory().planner_prompt("do X")
    # The planner prompt advertises the `specs` key to phi ...
    assert "specs" in system
    # ... but the registered-spec lookup embedded in it never leaks a body (d10).
    assert '"body"' not in system and "'body'" not in system
    assert user.startswith("GOAL: do X")


def test_index_body_guard_still_bites():
    # The d10 hard guard: an index row carrying a `body` is rejected outright, so a
    # body can never reach the planner-facing factory.
    with pytest.raises(PlanError):
        AbstractPlanFactory([{"name": "x", "description": "d", "body": "LEAK"}])


def test_assert_body_free_catches_a_planted_body():
    # The guard is not vacuous: a body planted anywhere in a payload is caught.
    with pytest.raises(PlanError):
        _factory().assert_body_free({"specializations": [{"name": "x", "body": "LEAK"}]})


# --------------------------------------------------------------------------- #
# round-trip: a specs-carrying node survives as_dict (used by re-plan carry-through).
# --------------------------------------------------------------------------- #
def test_plannode_as_dict_round_trips_specs():
    n = PlanNode(id="n1", task="t", specs=("markdown-writer", "brevity-editor"))
    d = n.as_dict()
    assert d["specs"] == ["markdown-writer", "brevity-editor"]
    assert d["spec"] is None  # scalar untouched; specs is the authoritative list
