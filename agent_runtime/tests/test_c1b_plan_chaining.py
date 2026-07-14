"""s9/c1b (d49.4) — PLAN-CHAINING multi-page accumulation in the shared file loop.

c1b adds LARGE / multi-page output via PLAN CHAINING: a write-file plan is ITSELF a
shape (linear OR parallel) whose per-page/section nodes FILL one file. The
decomposition lives in the authored DAG (the page nodes + their ``depends_on`` chain),
NOT in code — the runtime only provides the structural accumulation contract:

  * a WRITER node (synthesis role / ``file_write`` tool) whose UPSTREAM is itself a
    writer CONTINUES that file — it reads the real on-disk tail and APPENDS the next
    page/section (never clobbers the earlier pages);
  * a writer with a DOWNSTREAM writer is NON-final, so it does NOT close the document
    wrapper (the closing-tag gate is deferred to the terminal page) → the assembled
    HTML doc is closed exactly once, at the end.

Both are computed structurally from the DAG topology (``_is_writer_node`` + the
dependents map), so a lone single-file deliverable is byte-for-byte the pre-c1b path
(proven by the unchanged ``test_c1_stepwise_synthesis`` suite). These tests script a
:class:`FakeTransport` over a REAL file hook bound to a tmp sandbox — the whole chain
runs in-process with zero inference, building on c1's loop + c3's decomposed tools.
"""
from __future__ import annotations

import asyncio

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.runtime import AgentRuntime, _is_writer_node
from agent_runtime.synth_tools import (
    DONE_SENTINEL,
    html_close_gap,
)
from llm_framework import FakeTransport
from reactive_tools import EventPlane, ToolHook, register_agentic_tools


def _run(coro):
    return asyncio.run(coro)


def _hook(tmp_path) -> ToolHook:
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    return hook


_CONT_MARK = "earlier pages/sections are ALREADY on the file"


def test_is_writer_node_discriminator():
    assert _is_writer_node(PlanNode(id="a", task="t", role="synthesizer"))
    assert _is_writer_node(PlanNode(id="b", task="t", tool="file_write"))
    assert _is_writer_node(PlanNode(id="c", task="t", tool="write_file"))
    assert not _is_writer_node(PlanNode(id="d", task="t", role="worker"))
    assert not _is_writer_node(PlanNode(id="e", task="t", tool="web_search"))
    assert not _is_writer_node(PlanNode(id="f", task="t"))


# --------------------------------------------------------------------------- #
# A two-page MARKDOWN report: plan2's per-page nodes accumulate ONE file.
# --------------------------------------------------------------------------- #
_MD_P1 = "# Big Report\n\n## Page 1 — Introduction\n\nThe opening section, in full."
_MD_P2 = "## Page 2 — Findings\n\nThe second section, appended after the first."


# AUTONOMY REBUILD P2: test_multipage_markdown_accumulates_across_chained_write_nodes RETIRED — it exercised the deleted raw write loop /
# deliverable_path routing (write nodes now run the unified self-select pull-writer;
# see test_sb6_write_fold_antifab.py::test_write_route_has_no_flag_every_worker_takes_the_unified_loop).


# AUTONOMY REBUILD P2: test_multipage_html_defers_close_to_final_page RETIRED — it exercised the deleted raw write loop /
# deliverable_path routing (write nodes now run the unified self-select pull-writer;
# see test_sb6_write_fold_antifab.py::test_write_route_has_no_flag_every_worker_takes_the_unified_loop).


# AUTONOMY REBUILD P2: test_single_file_writer_is_unaffected_no_premature_close RETIRED — raw write loop deleted; the
# pull-writer owns document continuation per its spec.


