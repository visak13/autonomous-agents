"""SA-1 / A0 — the REGISTRY-WIRING FOUNDATION fix, proven on the REAL served hook.

These tests are FULLY OFFLINE (stub transports; no Ollama / network). They drive the
SAME growth seam the runtime's node self-select uses — ``expand_bundle(name, hook.registry,
ctx)`` is exactly what ``SubAgent._load_bundle`` / the ``get_bundles`` tool call — against the
hook ``chat_app.app.build_wiring`` actually composes (a real ``GrowableToolRegistry``, NOT a
hand-built one), so they are the standing guard against the A0 regression class.

THE BUG (pre-fix): ``register_agentic_tools`` built a ``GrowableToolRegistry`` bound to the
hook, returned it, but NEVER assigned it back to ``hook.registry`` — which stayed a base
``ToolRegistry`` with no ``.add``. So a self-selected bundle's ``registry.add(...)`` raised
``AttributeError`` (swallowed by ``expand_bundle``) and the handlers NEVER registered; the
served write path only worked because it PRE-REGISTERED ``load_source`` / ``read_notes`` per
run (the mask).

THE FIX (A0): the growable adopts the hook's existing base registry as its internal dispatch
store (static core/lambda tools survive), gains a ``.register`` passthrough + a ``.resolve``
dispatch lookup, and ``register_agentic_tools`` ASSIGNS it back as ``hook.registry``. So
``hook.registry`` IS the growable: bundle self-select genuinely registers + dispatches, and the
``load_source`` per-run pre-reg is retired (a self-selecting node registers it via this seam,
bound to the run's ``chain_sources``). (The ``read_notes`` pre-reg stays as the existing bridge
until SA-4 adds the matching ``chain_notes`` feed to the write runtime's ``_node_run_ctx``.)
"""
from __future__ import annotations

import asyncio

from agent_runtime.bundles import expand_bundle
from chat_app.app import build_wiring
from reactive_tools.tool_hook import ToolRegistry
from reactive_tools.tool_registry import GrowableToolRegistry


# The GLOBAL verbatim source list a write/review node's load_source binds to (the runtime
# feeds ctx['sources'] from write_runtime.chain_sources). Shape mirrors the served path.
_SOURCES = [
    {"url": "https://reuters.com/iran", "title": "Reuters Iran",
     "markdown": "# Reuters\nEconomic damage was put at $113.3B in the report."},
]
# The prior gather NOTES (the read_notes gist), re-keyed to the global [S#] by URL.
_NOTES = [
    {
        "source_id": 1,
        "url": "https://reuters.com/iran",
        "title": "Reuters Iran",
        "summary": "Economic damage put at $113.3B.",
        "key_claims": ["$113.3B economic damage"],
        "gaps_or_followups": ["who first reported it?"],
        "source_trust": "secondary",
    }
]


def test_served_hook_registry_is_growable_with_add(tmp_path):
    """The served hook's registry IS a GrowableToolRegistry with the working growth point —
    AND the static core/lambda tools survived the swap (no catalog regression)."""
    w = build_wiring(data_dir=str(tmp_path))
    reg = w.hook.registry
    assert isinstance(reg, GrowableToolRegistry), (
        "register_agentic_tools must assign the growable back as hook.registry (A0)")
    # the growth + dispatch surface a self-select / hook.register path needs
    assert hasattr(reg, "add") and hasattr(reg, "register") and hasattr(reg, "resolve")
    catalog = {t["name"] for t in reg.catalog()}
    # the agentic node surface is advertised...
    assert {"web_search", "web_fetch", "file_read", "file_write", "send_mail"} <= catalog
    # ...AND the static core tools registered by build_default_hook SURVIVED (catalog()
    # delegates to the adopted base store, so the planner's tool_catalog did NOT shrink).
    assert "read_file" in catalog and "write_file" in catalog


def test_research_read_self_select_registers_load_source_dispatchable(tmp_path):
    """SERVED write-path load_source self-select: with the mask RETIRED, loading research_read
    on the real hook registers load_source bound to the run's sources and it DISPATCHES."""
    w = build_wiring(data_dir=str(tmp_path))
    reg = w.hook.registry
    assert "load_source" not in reg, "load_source must NOT be pre-registered (mask retired)"

    # the exact seam SubAgent._load_bundle / get_bundles invoke, with this run's sources in ctx
    result = expand_bundle("research_read", reg, {"sources": _SOURCES})
    assert result["loaded"] == "research_read"
    assert "load_source" in reg, "research_read self-select must register load_source via .add"

    res = asyncio.run(w.hook.invoke("load_source", sid="S1"))
    assert res.ok, f"load_source must dispatch through the hook: {res.error}"
    assert "$113.3B" in str(res.value), "load_source must return the REAL verbatim source text"


def test_research_read_binds_read_notes_when_ctx_has_notes(tmp_path):
    """The bundle mechanism for read_notes is CORRECT — it binds + dispatches whenever ctx
    carries the notes. On the served WRITE runtime ctx['notes'] is currently empty (the notes
    arrive as a write_report_spa param, not via the DAG), so the read_notes per-run pre-reg
    stays as the bridge until SA-4 adds the chain_notes feed. This test pins the mechanism so
    SA-4 only has to wire the feed, not re-prove the bundle."""
    w = build_wiring(data_dir=str(tmp_path))
    reg = w.hook.registry
    expand_bundle("research_read", reg, {"sources": _SOURCES, "article_notes": _NOTES})
    assert "read_notes" in reg, "research_read must register read_notes when ctx has notes"
    res = asyncio.run(w.hook.invoke("read_notes"))
    assert res.ok, f"read_notes must dispatch through the hook: {res.error}"


def test_fresh_codebase_bundle_self_selects_and_dispatches_end_to_end(tmp_path):
    """THE A0 GATE: a FRESH, NON-pre-registered bundle (codebase) self-selects on the REAL
    build_wiring hook and its tools — which would have returned 'unknown tool' before the fix —
    now register AND dispatch end-to-end. This proves the registry foundation independent of
    any per-run mask (codebase is never pre-registered anywhere)."""
    w = build_wiring(data_dir=str(tmp_path))
    reg = w.hook.registry
    # codebase's read_file shares a name with the core tool, but list_dir/read_dir are
    # genuinely fresh — assert those are absent before self-select.
    assert "list_dir" not in reg and "read_dir" not in reg, (
        "codebase tools must NOT exist before the bundle is self-selected")

    # make a tiny real codebase to read
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def answer():\n    return 42\n", encoding="utf-8")

    result = expand_bundle("codebase", reg, {"codebase_root": str(tmp_path)})
    assert result["loaded"] == "codebase"
    assert "list_dir" in reg and "read_dir" in reg and "read_file" in reg, (
        "codebase self-select must register its tools via the working .add growth point")

    # END-TO-END dispatch through the hook on the event plane — REAL on-disk read, not a stub.
    listed = asyncio.run(w.hook.invoke("list_dir", path="pkg"))
    assert listed.ok, f"list_dir must dispatch: {listed.error}"
    names = {e["name"] for e in listed.value["entries"]}
    assert "mod.py" in names

    read = asyncio.run(w.hook.invoke("read_file", path="pkg/mod.py"))
    assert read.ok and "return 42" in read.value["text"]


def test_base_registry_swallow_is_the_bug_the_fix_closes():
    """CONTRAST (documents the masked failure): a plain base ToolRegistry has no .add, so
    expand_bundle silently no-ops (the AttributeError is swallowed) and the tool stays
    undispatchable — the exact pre-fix served state. The served hook above does NOT exhibit
    this because hook.registry is now the growable."""
    base = ToolRegistry()
    assert not hasattr(base, "add")
    # expand_bundle swallows the AttributeError, returns the bundle dict, registers NOTHING.
    out = expand_bundle("codebase", base, {"codebase_root": "."})
    assert out["loaded"] == "codebase"
    assert "list_dir" not in base, "a base registry cannot grow — the gap the A0 fix closes"
