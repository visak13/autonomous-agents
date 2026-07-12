# aflex finding — the non-web GATHER-EXECUTION web-lock (Tier-2 spine gap)

**Status:** surfaced by the s16/aflex generic-spine FLEX probe (d239/d240/d241). This is a
DEFINITION-LAYER LEVER for a SEPARATE engine-generalization action — **not** hacked in aflex
(aflex is a probe, not a builder; d186: a spine gap is a lever to fix properly, never a hack
or a waiver). Tier-1 (a new non-web capability via the definition layer alone, zero engine
edits) is PROVEN live by aflex; this doc characterizes the precise remaining gap so the
fuller d241/d238 claim can be built cleanly later.

## What aflex proved is already GENERIC (no engine edit — Tier-1)

A new **non-web capability** (a `codebase` bundle: `list_dir`/`read_dir`/`read_file`) runs
end-to-end through the existing spine by adding ONLY definition-layer pieces:

- a **bundle** (`agent_runtime/bundles/codebase.py`, registered in `bundles/__init__.py`),
- a **shape** (`agent_runtime/shapes/codebase-summary.toml`, `execution = "sequential"`),
- an output **spec** (`codebase-summary` in `specialization/seed.py`),
- **curation** entries (`CURATED_SHAPES` + `CURATED_SPECS` in `chat_app/curation.py`).

It rides these already-source-agnostic seams:

- **Bundle registry + catalog** auto-advertise the new bundle (`bundles_catalog_text`).
- **Generic loaded-tool dispatch** — `SubAgent._dispatch_loaded_tool` (runtime.py) invokes ANY
  loaded bundle's tool through the hook BY NAME; used by the linear-worker, tool-calling-writer,
  anchored-reviewer and the file/synthesis self-select fronts. Its docstring already names
  "codebase, vector-db" as the intended extension.
- **Source-agnostic read-binding** — `_node_run_ctx` / `_collect_upstream_notes` bind a node's
  upstream sources + notes under the web key (`fetched`/`article_notes`) OR a generic key
  (`records`/`notes`).
- **Source-agnostic grower** (as4) — recognizes/ingests gather nodes by RESEARCHER role / memory
  handle, not a web tool.

The Tier-1 demonstration uses a **WORKER** node (→ `_run_linear_worker` → `_dispatch_loaded_tool`)
to read the codebase, then a writer node to synthesize — so it never touches the web-locked
gather loop below.

## The GAP — the gather-EXECUTION loop and records-emission are WEB-LOCKED

The fuller d241/d238 claim — *the grower DEEPENS over the non-web source AND the writer PULLS over
the non-web complex-memory via the SAME `read_notes`/`load_source` interface* — cannot be reached
by definition layer alone today, because of three web-specific seams:

1. **The gather ReAct loop dispatches only web tools.**
   `SubAgent._run_research_loop` (runtime.py) is the loop a RESEARCHER node runs. It dispatches a
   gather call through `SubAgent._dispatch_research_tool` (runtime.py ~line 1717), which handles
   ONLY `self._search_tool` (`web_search`) and, as its unconditional `else`, `self._fetch_tool`
   (`web_fetch`) — with web semantics baked in: URL validation (`_looks_like_article_url`),
   offered-URL grounding (`offered_urls`), article-text detection (`_is_readable_fetch`), markdown
   extraction. **There is no generic fallthrough** to `_dispatch_loaded_tool`. So a RESEARCHER node
   that self-selects a NON-web gather bundle has its tools mis-dispatched as `web_fetch` (treated as
   a URL → "not a readable HTML article"). The note/fetch/search turns are also keyed on the
   construction-time names `self._note_tool`/`self._fetch_tool`/`self._search_tool`.

2. **Only the web loop emits a downstream-PULLABLE artifact.**
   `_run_research_loop` is the only loop that returns `tool_value = {"fetched": [...],
   "article_notes": [...]}` — the artifact `_collect_upstream_notes` + `chain_sources` later turn
   into a writer's `read_notes`/`load_source` pull. The GENERIC `_run_linear_worker` returns the
   worker's RAW prose answer only (`tool_value=None`); it accumulates no structured
   `records`/`notes`. So a worker-based non-web gather produces NOTHING for a downstream writer to
   `read_notes`/`load_source` over — the read-binding seam is generic, but nothing non-web ever
   FEEDS it.

3. **The writer's `chain_sources` is assembled by the web write pipeline.**
   `chat_app/agentic.py::write_report_spa` sets `write_runtime.chain_sources = sources`, where
   `sources` is the web research phase's fetched list. The acyclic (non-web) path does not harvest
   a non-web gather node's records into `chain_sources`.

## What "source-agnostic gather-execution + records-emission" requires (clean brief)

A future builder action (NOT aflex) should, at the ENGINE layer:

1. **Generic gather dispatch.** In `_run_research_loop`, when a self-selected gather tool is NOT
   the configured web search/fetch/note, fall through to `_dispatch_loaded_tool` (the generic hook
   dispatch) instead of `_dispatch_research_tool`. Keep the web branch byte-identical; add a
   source-agnostic branch keyed on "loaded gather tool that isn't a web tool". (Alternatively:
   make the gather bundle declare its own search/fetch/note tool names + a per-bundle dispatch/
   ingest adapter, so `_dispatch_research_tool` is selected by the bundle, not hardwired to web.)

2. **Source-agnostic records emission.** Have a non-web gather node attach a structured artifact to
   its result `tool_value` under the generic key (`records`/`notes`) — the same shape
   `_collect_upstream_notes` already reads — so a downstream writer's `read_notes`/`load_source`
   binds over it. This means either (a) a gather loop (not the prose-only linear worker) that
   accumulates per-source records for a non-web source, or (b) letting the codebase bundle's read
   tool record a per-file note that the node folds into `tool_value`.

3. **Source-agnostic chain_sources on the non-web write path.** Harvest a non-web gather node's
   `records` into the writer runtime's `chain_sources` (the acyclic/generic write path), mirroring
   what `write_report_spa` does for web `fetched`, so the writer can `load_source` real file
   content by `[S#]`.

With (1)–(3) the grower could deepen over a codebase (or vector-db) source and the writer would
pull over it through the SAME `read_notes`/`load_source` interface — closing the d241/d238 claim
for non-web complex-memory, still with the new SOURCE expressed as a bundle (definition layer),
the ENGINE generalization done once (not per source).

## Relationship to the other gap / ordering

This is gap #1. Gap #2 — bundle self-select handler registration is broken on the served wiring
(`hook.registry` is a base `ToolRegistry` with no `.add`, so `expand_bundle`'s growth point
silently no-ops) — is in [aflex_selfselect_wiring_gap.md](aflex_selfselect_wiring_gap.md).
**Gap #2 sequences FIRST** (ALL bundle self-select depends on it); this gap #1 (source-agnostic
gather-execution + records-emission) follows, to close the fuller d241/d238 claim.

## Scope / timing

The Tier-2 build TIMING (now as a new step vs. defer post-d206) is a neuron→user decision; it does
NOT block Tier-1. This doc is the brief for that action.
