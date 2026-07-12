"""d227/G1 — EMERGENT spec-assignment doctrine (research nodes never get a
document-format spec; an output-format spec binds ONLY to the write/deliverable
node). The TRUE root of the a3 format-bleed was the planner stamping html-writer on
RESEARCH nodes, so research leaves inherited HTML doctrine and emitted HTML instead
of notes. The fix is a DEFINITION-LAYER lever — the planner REASONS the assignment
from the doctrine + the advertised spec descriptions (d194/d227) — NOT a hardcoded
'if research node then spec=X' code rule. These tests pin that the anti-pattern is
present in the levers the planner reasons over: the factory doctrine, the
incremental authorer's per-node prompts, and the canonical writer/research specs'
descriptions. (A behavioural live-model proof is the neuron's a3 re-gate; this guards
the doctrine does not silently regress.)"""
from __future__ import annotations

from agent_runtime.factory import FACTORY_DESCRIPTION, AbstractPlanFactory
from agent_runtime.incremental import IncrementalPlanner
from llm_framework import FakeTransport
from specialization.seed import CANONICAL_RULESETS


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def test_factory_doctrine_forbids_format_spec_on_research_nodes():
    d = _norm(FACTORY_DESCRIPTION)
    # The positive rule stays (output-style spec on the deliverable node)…
    assert "output-style spec to the node that produces the deliverable" in d
    # …and the NEW anti-pattern: a research/gather node never carries a doc-format spec.
    assert "never put a document-format spec on a research/gather node" in d
    assert "research-analyst" in d
    assert "format-bleed" in d


def test_incremental_authorer_prompt_forbids_format_spec_on_gather_nodes():
    factory = AbstractPlanFactory([], tool_catalog=[])
    planner = IncrementalPlanner(
        FakeTransport([]),
        factory,
        spec_names=["html-writer", "research-analyst"],
        tool_names=["web_search", "file_write"],
        shape_name="deep-research",
        shape_description="exhaustive single-topic investigation",
    )
    system = _norm(planner._system("a report"))
    # The per-node authoring lever (strongest for E4B) carries the anti-pattern.
    assert "never a document-format spec" in system
    assert "instead of gathering notes" in system
    initial = _norm(planner._initial_user("write an html report on X"))
    assert "never a document-format writer spec" in initial


def test_canonical_writer_specs_advertise_write_node_only():
    html_desc = _norm(CANONICAL_RULESETS["html-writer"][0])
    md_desc = _norm(CANONICAL_RULESETS["markdown-writer"][0])
    research_desc = _norm(CANONICAL_RULESETS["research-analyst"][0])
    # The document-format specs advertise WRITE-node-only + the explicit anti-pattern,
    # so the planner reasoning over descriptions does not bind them to a gather node.
    for desc in (html_desc, md_desc):
        assert "never bind it to a research/gather/analysis node" in desc
    # The research spec advertises it is for gather nodes and is NOT a format spec.
    assert "not a document-format" in research_desc
    assert "gather" in research_desc
