"""NODE-SELF-SELECT bundle mechanism (d221) — fully OFFLINE.

Proves the d221 redraw end to end, without Ollama / network:

* the hardcoded ROLE_BUNDLES / POSITION_BUNDLES / _TOOL_BUNDLES tables + the
  deterministic ``bundles_for_node`` / ``bundles_for_position`` assignment are GONE;
* ``get_bundles`` LISTS the advertised catalog and LOADS a bundle's tools at runtime
  (registers handler ToolDefs onto a real GrowableToolRegistry) — the runtime
  self-select surface, parallel to get_shapes / get_specs;
* the catalog is ADVERTISED in every ROLE-carrying node's system prompt (and NOT on a
  role-less producer step — back-compat);
* a node SELF-SELECTS its bundles at runtime (``SubAgent._load_bundle``), which grows
  its pinned doctrine and activates the research bundle's web_fetch OUTPUT-MESSAGE
  OVERRIDE (take-a-note) — a plain context has no such message.
"""
from __future__ import annotations

import agent_runtime.roles as roles
from agent_runtime.bundles import (
    BUNDLE_OBJECT,
    bundles_catalog,
    bundles_catalog_text,
    compose_doctrine,
    expand_bundle,
)
from agent_runtime.bundles.research import WEB_FETCH_NOTE_OVERRIDE
from agent_runtime.discovery_tools import make_get_bundles, register_discovery_tools
from agent_runtime.factory import PlanNode
from agent_runtime.runtime import SubAgent
from llm_framework import ChatResult, FakeTransport
from reactive_tools.event_plane import EventPlane
from reactive_tools.tool_hook import ToolHook
from reactive_tools.tool_registry import GrowableToolRegistry


# --------------------------------------------------------------------------- #
# (1) the hardcoded tables + bundles_for_node are REMOVED (d221 #3).
# --------------------------------------------------------------------------- #
def test_hardcoded_bundle_tables_are_removed():
    for gone in (
        "ROLE_BUNDLES", "POSITION_BUNDLES", "_TOOL_BUNDLES",
        "bundles_for_node", "bundles_for_position",
    ):
        assert not hasattr(roles, gone), f"{gone} must be removed (d221)"
    # the BUNDLE_* name constants stay (they are not tables).
    assert roles.BUNDLE_OBJECT == "object"
    assert roles.BUNDLE_RESEARCH == "research"


# --------------------------------------------------------------------------- #
# (2) the advertised catalog — node-selectable bundles only (object + planning out).
# --------------------------------------------------------------------------- #
def test_catalog_advertises_selectable_bundles_only():
    names = {r["name"] for r in bundles_catalog()}
    assert names == {"research", "research_read", "file", "codebase"}
    assert "object" not in names and "planning" not in names  # floor / planner-only
    text = bundles_catalog_text()
    assert "List of bundles" in text and 'get_bundles(name=' in text
    for n in ("research", "research_read", "file", "codebase"):
        assert f"- {n}:" in text


# --------------------------------------------------------------------------- #
# (3) get_bundles LISTS the catalog and LOADS a bundle at runtime.
# --------------------------------------------------------------------------- #
def test_get_bundles_lists_and_loads():
    get_bundles = make_get_bundles()
    listed = get_bundles()  # no name -> the catalog
    assert listed["count"] == 4
    assert {r["name"] for r in listed["bundles"]} == {
        "research", "research_read", "file", "codebase"}

    loaded = get_bundles(name="research")  # name -> LOAD it
    assert loaded["loaded"] == "research"
    assert "web_search" in loaded["tools"] and "web_fetch" in loaded["tools"]
    assert "GATHER" in loaded["doctrine"] or "gather" in loaded["doctrine"].lower()

    bogus = get_bundles(name="does-not-exist")  # unknown -> error + the catalog (no crash)
    assert "error" in bogus and bogus["count"] == 4


def test_get_bundles_load_registers_handler_tools_at_runtime():
    """expand_bundle('research_read', registry, {sources}) GROWS the registry with the
    load_source handler tool — the real runtime tool-load (GrowableToolRegistry.add)."""
    hook = ToolHook(EventPlane())
    registry = GrowableToolRegistry(hook)
    sources = [{"title": "T", "url": "http://x", "markdown": "body text"}]
    assert "load_source" not in registry.names()
    expand_bundle("research_read", registry, {"sources": sources})
    assert "load_source" in registry.names()  # loaded at runtime


def test_register_discovery_tools_includes_get_bundles():
    hook = ToolHook(EventPlane())
    registry = GrowableToolRegistry(hook)
    register_discovery_tools(registry)
    assert {"get_shapes", "get_specs", "get_bundles"} <= set(registry.names())


# --------------------------------------------------------------------------- #
# (4) the catalog is advertised on a ROLE node's prompt — not a role-less producer.
# --------------------------------------------------------------------------- #
def test_catalog_advertised_on_role_prompt_only():
    researcher = SubAgent(
        PlanNode(id="n1", task="investigate the topic", role="researcher"),
        transport=FakeTransport(["x"]),
    )
    sys_researcher = researcher._compose_system()
    assert "List of bundles" in sys_researcher  # advertised on the role prompt

    plain = SubAgent(
        PlanNode(id="n2", task="just produce"),  # no role, no spec
        transport=FakeTransport(["x"]),
    )
    assert "List of bundles" not in (plain._compose_system() or "")  # back-compat


# --------------------------------------------------------------------------- #
# (5) a node SELF-SELECTS at runtime: doctrine grows + the web_fetch override fires.
# --------------------------------------------------------------------------- #
def test_node_self_select_grows_doctrine_and_fetch_override():
    agent = SubAgent(
        PlanNode(id="n1", task="investigate", role="researcher"),
        transport=FakeTransport(["x"]),
    )
    # Floor only to start.
    assert agent._loaded_bundles == {BUNDLE_OBJECT}
    assert agent._node_bundle_doctrine() == compose_doctrine({BUNDLE_OBJECT})
    # No research loaded -> no web_fetch take-a-note override (plain context).
    assert agent._fetch_output_override() == ""

    agent._load_bundle("research")  # SELF-SELECT the gather capability
    assert "research" in agent._loaded_bundles
    doctrine = agent._node_bundle_doctrine()
    assert doctrine != compose_doctrine({BUNDLE_OBJECT})  # grew
    assert "note" in doctrine.lower()  # the gather/notes flavor is now pinned
    # The research CONTEXT now overrides web_fetch's output to prompt take-a-note.
    assert agent._fetch_output_override() == WEB_FETCH_NOTE_OVERRIDE


# --------------------------------------------------------------------------- #
# (6) d242 TRUE self-select: a node starts TOOL-LESS — the offered surface is EXACTLY
# {get_bundles, finish} until it loads a bundle; a domain tool appears only on LOAD.
# --------------------------------------------------------------------------- #
def test_offered_tools_are_only_get_bundles_and_finish_before_self_select():
    agent = SubAgent(
        PlanNode(id="n1", task="investigate", role="researcher"),
        transport=FakeTransport(["x"]),
    )
    # Before any self-select: NO domain tool is pre-offered (d242 verify (i)).
    assert set(agent._offered_tool_names()) == {"get_bundles", "finish"}
    # After loading 'research', its gather tools appear in the offered surface.
    agent._load_bundle("research")
    offered = set(agent._offered_tool_names())
    assert {"get_bundles", "finish", "web_search", "web_fetch"} <= offered
    # The curation filter (``only``) narrows the loaded surface to a phase subset while
    # always keeping get_bundles + finish — never pre-mounting an unloaded tool.
    curated = set(agent._offered_tool_names(only=("web_search",)))
    assert curated == {"get_bundles", "finish", "web_search"}  # web_fetch curated out


# --------------------------------------------------------------------------- #
# (7) d241 + d242: the LINEAR/chat worker is NOT special-cased out of self-select — a
# FOLLOW-UP message self-selects a memory-read bundle (research_read), reaches the prior
# sources, and answers from them; a trivial message still answers in ONE turn.
# --------------------------------------------------------------------------- #
class _MemHook:
    """A minimal hook the linear worker dispatches its self-selected memory-read tool
    through: records load_source/read_notes calls and returns canned prior-session text."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def invoke(self, name: str, **args):
        self.calls.append(name)

        class _R:
            ok = True
            error = ""
            value = "PRIOR SOURCE S1: the strike occurred on June 13; 1,200 casualties."

        return _R()


class _WorkerScript:
    """Replays a fixed sequence of worker turns by call index."""

    def __init__(self, turns: list[str]) -> None:
        self._turns = list(turns)
        self.calls = 0

    def complete(self, messages, **opts) -> str:
        return self.chat(messages, **opts).content

    def chat(self, messages, **opts) -> ChatResult:
        i = self.calls
        self.calls += 1
        content = self._turns[i] if i < len(self._turns) else "FALLBACK."
        return ChatResult(role="assistant", content=content)


def test_linear_worker_self_selects_memory_read_for_followup():
    import asyncio

    hook = _MemHook()
    # The follow-up worker: SELF-SELECT research_read → read a prior source → answer FROM it.
    transport = _WorkerScript([
        '{"tool": "get_bundles", "args": {"name": "research_read"}}',
        '{"tool": "load_source", "args": {"sid": "S1"}}',
        "Based on the prior research, the strike was on June 13 with ~1,200 casualties.",
    ])
    agent = SubAgent(
        PlanNode(id="w1", task="When did it happen again?", role="worker"),
        transport=transport, hook=hook,
        chain_sources=[{"title": "UN", "url": "http://u/n", "markdown": "June 13; 1,200."}],
    )
    out = asyncio.run(agent.run({}))
    # the worker reached the prior research via the self-selected memory-read tool ...
    assert "research_read" in agent._loaded_bundles
    assert "load_source" in hook.calls
    # ... and answered FROM it, still as the linear worker (its spec/role applied on system).
    assert "June 13" in (out.output or "")


def test_linear_worker_trivial_message_answers_in_one_turn():
    import asyncio

    hook = _MemHook()
    transport = _WorkerScript(["Hello! How can I help you today?"])  # plain prose, no tool call
    agent = SubAgent(
        PlanNode(id="w2", task="say hi", role="worker"),
        transport=transport, hook=hook,
    )
    out = asyncio.run(agent.run({}))
    assert transport.calls == 1            # answered in ONE turn (no spurious self-select)
    assert hook.calls == []                # no memory-read tool dispatched
    assert "How can I help" in (out.output or "")


# --------------------------------------------------------------------------- #
# as4 / d241 — DOMAIN-AGNOSTIC memory-read: _node_run_ctx supplies the prior gather
# NOTES (not only sources), so read_notes — the CHEAP first leg of the cost hierarchy —
# binds for ANY self-selecting node (incl. the linear worker). Notes are collected
# SOURCE-AGNOSTICALLY: under the web key (article_notes) OR a generic key (notes), so a
# non-web complex-memory type (codebase / vector-db) flows through the SAME seam.
# --------------------------------------------------------------------------- #
def test_node_run_ctx_supplies_upstream_notes_for_read_notes_binding():
    web_note = {"url": "http://u/n", "summary": "strike", "key_claims": ["June 13"]}
    generic_note = {"sid": "C1", "summary": "module foo", "key_claims": ["entrypoint main()"]}
    agent = SubAgent(
        PlanNode(id="w1", task="follow-up", role="worker"),
        transport=FakeTransport([]),
        # one upstream dep emitted WEB article_notes, another emitted GENERIC notes/records.
        upstream_tool_values={
            "dep_web": {"article_notes": [web_note], "fetched": [{"url": "http://u/n"}]},
            "dep_code": {"notes": [generic_note], "records": [{"sid": "C1"}]},
        },
        chain_sources=[{"title": "UN", "url": "http://u/n", "markdown": "June 13."}],
    )
    collected = agent._collect_upstream_notes()
    assert web_note in collected and generic_note in collected  # BOTH source vocabularies fold
    ctx = agent._node_run_ctx()
    assert ctx.get("notes") == collected                        # supplied to the bundle ctx
    assert ctx.get("sources")                                   # sources still flow (load_source)
    # the research_read bundle binds read_notes (CHEAP leg) when ctx carries notes — generic.
    reg = GrowableToolRegistry(ToolHook(EventPlane()))
    loaded = expand_bundle("research_read", registry=reg, ctx=ctx)
    assert "read_notes" in (loaded.get("tools") or [])


def test_node_run_ctx_no_upstream_notes_is_clean():
    # No upstream gather → no notes key (byte-identical to pre-as4; never a spurious empty bind).
    agent = SubAgent(
        PlanNode(id="w2", task="trivial", role="worker"),
        transport=FakeTransport([]),
    )
    assert agent._collect_upstream_notes() == []
    assert "notes" not in agent._node_run_ctx()
