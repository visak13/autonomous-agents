"""SA-4 (d234/d235) — the CHAIN-NOTES seam: the write runtime FEEDS the served research
NOTES into a self-selecting node's ``ctx['notes']``, so ``read_notes`` (the CHEAP first leg
of the read hierarchy) binds by SELF-SELECT — replacing the retired per-run read_notes
pre-registration (the SA-1-deferred bridge).

FULLY OFFLINE. These pin the runtime MECHANISM (chain_notes -> _node_run_ctx -> ctx['notes'])
that SA-4 added so the agentic write path needs only set ``write_runtime.chain_notes`` (the
exact mirror of ``chain_sources``). The served-hook self-select dispatch is proven in
chat_app/tests/test_sa1_registry_foundation_a0.py (the bundle binds read_notes from ctx
notes); SA-4 closes the loop by sourcing those notes from the runtime, not a per-run pre-reg.
"""
from __future__ import annotations

from agent_runtime.bundles import expand_bundle
from agent_runtime.factory import PlanNode
from agent_runtime.roles import ROLE_SYNTHESIZER
from agent_runtime.runtime import SubAgent
from reactive_tools.event_plane import EventPlane
from reactive_tools.tool_hook import ToolHook
from reactive_tools.tool_registry import GrowableToolRegistry

_SOURCES = [
    {"url": "https://reuters.com/iran", "title": "Reuters Iran",
     "markdown": "# Reuters\nEconomic damage was put at $113.3B."},
]
_NOTES = [
    {"source_id": 1, "url": "https://reuters.com/iran", "title": "Reuters Iran",
     "summary": "Economic damage put at $113.3B.", "key_claims": ["$113.3B economic damage"],
     "gaps_or_followups": ["who first reported it?"], "source_trust": "secondary"},
]


def _write_node() -> PlanNode:
    # a terminal write/synthesis node (the kind the write runtime builds) — it self-selects
    # research_read to ground its prose in the run's sources + notes.
    return PlanNode(id="w1", task="write the report from the research", role=ROLE_SYNTHESIZER)


def test_chain_notes_feeds_node_run_ctx_notes():
    """A SubAgent built with ``chain_notes`` exposes them as ``ctx['notes']`` — the seam the
    write runtime uses so read_notes self-selects on the write path (its DAG upstream is
    empty, so without this feed ctx['notes'] would be empty — the exact SA-1 deferral)."""
    agent = SubAgent(_write_node(), transport=None, hook=None,
                     chain_sources=_SOURCES, chain_notes=_NOTES)
    ctx = agent._node_run_ctx()
    assert ctx["notes"] == _NOTES, "chain_notes must surface as ctx['notes']"
    assert ctx["sources"] == _SOURCES  # the existing chain_sources leg still works


def test_no_chain_notes_leaves_ctx_notes_absent():
    """Byte-identical for every non-report runtime: no chain_notes fed AND no DAG upstream
    notes => no ``notes`` key at all (the pre-SA-4 behaviour for those paths)."""
    agent = SubAgent(_write_node(), transport=None, hook=None, chain_sources=_SOURCES)
    ctx = agent._node_run_ctx()
    assert "notes" not in ctx


def test_research_read_self_select_binds_read_notes_from_chain_notes_ctx():
    """END-TO-END seam: the ctx a chain_notes-fed node hands its self-select carries the notes,
    so research_read registers read_notes via the working growth point — no per-run pre-reg.
    Uses a GrowableToolRegistry (the served hook.registry type, SA-1) as the growth point."""
    agent = SubAgent(_write_node(), transport=None, hook=None,
                     chain_sources=_SOURCES, chain_notes=_NOTES)
    reg = GrowableToolRegistry(ToolHook(EventPlane()))
    # the exact ctx the node's get_bundles self-select passes to the bundle's register.
    ctx = agent._node_run_ctx()
    expand_bundle("research_read", reg, ctx)
    assert "read_notes" in reg, "research_read must register read_notes from chain_notes ctx"
    assert "load_source" in reg, "and load_source from the chain_sources ctx (both legs bind)"
