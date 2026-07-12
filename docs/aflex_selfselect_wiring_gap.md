# aflex finding — bundle self-select handler registration is broken on the served wiring (spine gap #2)

**Status:** surfaced by the s16/aflex generic-spine FLEX probe (d239/d240/d241). This is a
PREREQUISITE wiring fix for the whole bundle-self-select refactor — a SEPARATE BUILDER action
(not aflex; aflex is a pure probe). Neuron ruling: keep aflex a probe, characterize this gap,
route the fix to a builder with the brief below. It is NOT a definition-layer lever and must
NOT be hacked around in a bundle (that would leave the existing `research_read` self-select
broken — a d186 violation).

## Severity / blast radius

This blocks **TRUE self-select for every bundle** — `codebase` (this probe), the planned
`bash` (d259), web-`research`, and `research_read`. It sequences **FIRST** in the asd design
breakdown: all bundle self-select depends on it.

## Root cause (proven live + offline)

In the real wiring (`reactive_tools.tool_hook.build_default_hook` →
`chat_app.app.build_wiring`), `hook.registry` is a **base `ToolRegistry`** (it has
`catalog` / `register` / `get` / `names`) — **NOT** a `GrowableToolRegistry`. The base
`ToolRegistry` has **no `.add(ToolDef)`** method.

The node self-select growth point is:
`SubAgent._load_bundle` → `registry = getattr(self.hook, "registry", None)` →
`bundles.expand_bundle(name, registry, ctx)` → `bundle.register(registry, ctx)` →
`registry.add(ToolDef(...))`.

Because `registry` is a base `ToolRegistry`, `registry.add(...)` raises `AttributeError`,
and `expand_bundle` **SWALLOWS it** (`try/except Exception: pass`). So a self-selected
bundle's **handlers are never registered on the dispatch hook** — `get_bundles` reports the
bundle "loaded" (it returns the bundle's native tool *names*), the doctrine pins, but the
tools are not dispatchable. Calling one returns "unknown tool" from `_dispatch_loaded_tool`.

### Why this stayed hidden
- The d242 self-select tests (`test_bundle_self_select_d221.py`) construct an **explicit
  `GrowableToolRegistry`**, so they pass — they never exercise the REAL `build_wiring` hook.
- The served WEB write path doesn't rely on self-select for its read tool: the engine
  **PRE-REGISTERS** `load_source` per-run in `chat_app.agentic.write_report_spa` (and
  `read_notes` similarly), so the writer can call it regardless of whether self-select
  registered it. That pre-registration MASKS the gap for the one existing bundle that
  matters on the served route. A genuinely NEW bundle (codebase) has no pre-registration, so
  the gap is exposed.

### Evidence
- **Live (E4B, :11434):** the worker node self-SELECTED the new `codebase` bundle
  (`get_bundles(name="codebase")` succeeded; `loaded_bundles == ["codebase", "object"]`; the
  load ack listed `list_dir`/`read_dir`/`read_file`), but the next turn's `read_dir` /
  `list_dir` calls returned: *"… is not available … unknown tool 'read_dir'; registered:
  [… file_read, read_file, web_fetch, web_search, write_file …]"* — i.e. the codebase tools
  were never registered. (`read_file` in that list is the PRE-EXISTING `reactive_tools.tools`
  tool, not the bundle's.)
- **Offline, on the REAL `build_wiring` hook:** `SubAgent._load_bundle("research_read")` with
  chain sources → `load_source` registered = **False**, `read_notes` registered = **False**;
  `_load_bundle("codebase")` → `list_dir` registered = **False**. The same `expand_bundle`
  against a fresh `GrowableToolRegistry` registers all three correctly (so the bundle code +
  the mechanism are sound; only the wiring is wrong).
- `type(build_wiring().hook.registry).__name__ == "ToolRegistry"`, `hasattr(..., "add") ==
  False`, `hasattr(..., "catalog") == True`.

## Builder brief (the neuron's hand-off) — a SEPARATE action, sequenced FIRST

1. **Make `hook.registry` the growable registry.** `register_agentic_tools` ALREADY builds a
   `GrowableToolRegistry(hook)` (its docstring even says it is "reachable as the bound
   `hook.registry`") — the assignment is simply MISSING. Assign it, and add a `.register(...)`
   PASSTHROUGH on `GrowableToolRegistry` (delegating to its bound hook / underlying registry)
   so existing `hook.register(...)` call sites keep working after the swap. Mind the
   construction order in `build_wiring`/`build_default_hook` (the static tools are registered
   during `register_agentic_tools`, before the swap).
2. **Regression-safety on the served WEB path.** Once self-select genuinely registers
   handlers, RECONCILE/retire the `write_report_spa` per-run pre-registration of `load_source`
   (and `read_notes`) carefully: avoid double-registration and avoid regressing `load_source`
   on the served deep-research write route. (The web route must stay green.)
3. **A SERVED-PATH test (the gap that hid this).** Add a test that drives the REAL
   `build_wiring` hook (not a hand-built `GrowableToolRegistry`) and asserts self-select
   genuinely registers handlers: `research_read` self-select → `load_source` AND `read_notes`
   become dispatchable; a FRESH bundle (e.g. `codebase`) self-select → its tools dispatch
   end-to-end. A growable-registry-only unit test is NOT sufficient — that is exactly what let
   this regress.
4. **Verify SERVED self-select end-to-end** (not tests-only): a node on the real served route
   self-selects a non-pre-registered bundle and successfully calls its tool against live E4B.

## Relationship to the other gap

This is gap #2. Gap #1 (the gather-EXECUTION web-lock) is in
[aflex_tier2_spine_gap.md](aflex_tier2_spine_gap.md). Both are inputs to the asd design.
Ordering: **gap #2 (this wiring fix) FIRST** (all self-select depends on it), then gap #1
(source-agnostic gather-execution + records-emission) to close the fuller d241/d238 claim.
