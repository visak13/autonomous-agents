"""RP-4b (d333/d334/d335/d336) — LEAN cron tools + a CONCISE schedule-vs-do-now doctrine.

RP-4b set out to push the schedule-only topology toward ~100% via the d333 "tool
advertisement" lever (a richer cron_add description). A same-window A/B measure REVERSED
that premise (recorded d336): rewriting the cron_add DESCRIPTION — verbose OR concise —
REGRESSED E4B authoring (1/6 vs the lean original's 5/6, same window), because the
description is fed verbatim into the planner's tool catalog and MORE text confuses the
small model. "Verbose hurts E4B" (verbose spec 0/4) extends to TOOL DESCRIPTIONS. So the
cron tools stay LEAN (the proven ~322-char 5/6 description), and the surviving RP-4b lever
is the CONCISE selector doctrine (the RP-3a-style schedule-this-vs-do-now kind-separation),
which measured NON-regressing at 5/6.

Self-policing tests (anti-fab charter, read/grep-style — no GPU, no network):

1. The cron tools stay LEAN + GENERIC + FLEX: cron_add is short, advertises WHEN (a
   recurring/cadence request) + WHAT (does NOT run now; re-runs FRESH) + ARGS (schedule
   cadence + the whole-task prompt), and carries NO deliverable/format/flow words. It is a
   generic capability that schedules ANY task (d335).
2. cron_list / cron_delete are self-explanatory (schedule-management pair, id-based).
3. The shape-selector carries a CONCISE, SHARP schedule-this-vs-do-now intent-separation
   (a recurring request is SCHEDULE-ONLY, one cron_add leg, no run-now; a 'do this now'
   request runs now with no cron_add), and it stays CONCISE (bounded length — verbose
   measured 0/4).
4. No engine flag / cron-tool / spec-name conditional drives the schedule routing — it is
   PROMPT doctrine only (behavior-via-prompting), never Python control flow.

The BEFORE/AFTER schedule-only LIVE reliability (lean 5/6; doctrine-kept 5/6; verbose &
concise-rewrite 1/6, the reversal) is documented in the RP-4b worklog + scratchpad measure.
"""
from __future__ import annotations

import inspect

from llm_framework import FakeTransport

from agent_runtime.shape_selector import ShapeSelector
import agent_runtime.shape_selector as shape_selector_mod
from reactive_tools.cron_tools import build_cron_tools


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cron_defs(tmp_path):
    return {d.name: d for d in build_cron_tools(data_dir=tmp_path)}


def _selector_prompt_with_cron():
    """Render the selector system prompt with cron_add advertised, so we can assert the
    schedule-this-vs-do-now doctrine is present + concise."""
    sel = ShapeSelector(
        FakeTransport([]),
        tool_catalog=[
            {"name": "cron_add", "description": "schedule a recurring task"},
            {"name": "web_search", "description": "search the web"},
        ],
    )
    return sel._system_prompt(sel.catalog())


# --------------------------------------------------------------------------- #
# 1) the cron tools stay LEAN + advertise when / what / args, GENERIC + FLEX
# --------------------------------------------------------------------------- #
def test_cron_add_description_is_lean():
    """LEAN WINS (d336): the description is the proven ~322-char capability blurb, NOT a
    bloated advertisement (a richer description measured a regression on E4B). Guard the
    length so it cannot silently balloon back into the regressing verbose form."""
    from reactive_tools.cron_tools import build_cron_tools as _b
    desc = {d.name: d for d in _b(data_dir=".")}["cron_add"].description
    assert len(desc) < 450, f"cron_add description grew to {len(desc)} chars — keep it lean"


def test_cron_add_description_advertises_when_what_args(tmp_path):
    low = _cron_defs(tmp_path)["cron_add"].description.lower()
    # WHEN — a recurring / cadence request (the trigger vocabulary is present)
    assert "recurring" in low
    assert "every morning" in low or "daily" in low or "schedule" in low
    # WHAT — it does NOT run now; the whole task re-runs FRESH each fire
    assert "not run now" in low or "is not run now" in low
    assert "fresh" in low
    # ARGS — the schedule cadence (cron expression) + the WHOLE self-contained task prompt
    assert "schedule" in low and "prompt" in low
    assert "cron expression" in low
    assert "whole" in low


def test_cron_tools_descriptions_are_generic_and_flex(tmp_path):
    defs = _cron_defs(tmp_path)
    flow_words = ("research", "report", "email", "html", "markdown", "summary", "news")
    for name in ("cron_add", "cron_list", "cron_delete"):
        low = defs[name].description.lower()
        for w in flow_words:
            assert w not in low, f"{name} description leaked flow word {w!r}"
    # FLEX — cron_add schedules ANY task (the '(for any …)' framing), not a fixed deliverable
    assert "any" in defs["cron_add"].description.lower()
    # and the arg schema is a plain schedule + prompt + name shape (no flow-keyed fields)
    props = set(defs["cron_add"].args_model.model_json_schema()["properties"])
    assert props == {"schedule", "prompt", "name"}


def test_cron_list_and_delete_are_self_explanatory(tmp_path):
    defs = _cron_defs(tmp_path)
    lst = defs["cron_list"].description.lower()
    dele = defs["cron_delete"].description.lower()
    assert ("scheduled" in lst or "cron" in lst) and "id" in lst
    assert ("delete" in dele or "cancel" in dele) and "id" in dele


# --------------------------------------------------------------------------- #
# 2) the selector carries a CONCISE, SHARP schedule-this-vs-do-now doctrine
# --------------------------------------------------------------------------- #
def test_selector_has_schedule_vs_donow_intent_separation():
    low = _selector_prompt_with_cron().lower()
    # the sharp binary is named + separated
    assert "schedule-this" in low and "do-this-now" in low
    assert "schedule-only" in low
    assert "cron_add" in low
    # a recurring request must NOT pick a run-now research shape
    assert "run-now" in low or "runs nothing now" in low
    # the explicit both-now-and-recurring carve-out
    assert "both now and" in low


def test_schedule_doctrine_is_concise():
    """CONCISE lever (d334): a SHARP separation, not a verbose dump (verbose measured 0/4).
    Guard the paragraph length so it cannot silently balloon back into prose bloat."""
    prompt = _selector_prompt_with_cron()
    marker = "SCHEDULE-THIS vs DO-THIS-NOW"
    assert marker in prompt
    start = prompt.index(marker)
    end = prompt.find("\n\n", start)
    para = prompt[start:] if end == -1 else prompt[start:end]
    assert len(para) < 900, f"schedule doctrine grew to {len(para)} chars (keep it concise)"


# --------------------------------------------------------------------------- #
# 3) no engine flag / cron-tool / spec-name conditional for the schedule routing
# --------------------------------------------------------------------------- #
def test_no_engine_cron_or_schedule_conditional_in_selector():
    """The schedule routing lives in PROMPT doctrine text only — never in Python control
    flow branching on the cron tool name or a schedule keyword (an engine hardcode, the
    anti-fab violation the charter forbids). Strip comment + string lines so doctrine prose
    naming cron_add is not a false match; assert against real CODE only."""
    src = inspect.getsource(shape_selector_mod)
    code = "\n".join(
        ln for ln in src.splitlines()
        if not ln.lstrip().startswith("#") and '"' not in ln and "'" not in ln
    )
    for banned in ("== cron_add", "cron_add ==", "if cron_add", "schedule_only ="):
        assert banned not in code, f"engine cron/schedule conditional found: {banned!r}"
