"""RP-4c (d338/d339) — SCHEDULE RELIABILITY: fix the two TRANSIENT schedule-only misses.

d338's trace-forensic reading of the RP-4b logs found the ~1/6 schedule-only miss is NOT a
schedule-vs-run-now authoring ceiling — the authoring is SOUND (every success authored a
perfect single cron_add node + the whole-task prompt verbatim). The misses are TRANSIENT +
HETEROGENEOUS:

  (i)  a bare ``TimeoutError`` — a transient :11434 latency spike exceeding the 300s per-call
       HTTP read timeout (perf d97/d101), and
  (ii) a ``MalformedOutputError`` "produced no usable nodes" — a transient EMPTY authoring turn
       from ``IncrementalPlanner.plan``.

Both are FIXABLE anti-fab-legally:

  (a) RE-PROMPT ON MALFORMED-EMPTY. ``IncrementalPlanner.plan``'s docstring assumes an "outer
      self-heal can re-plan exactly as it does for a malformed one-shot plan", but the acyclic
      authoring call site (``_author_and_drive_acyclic_plan``) never wired one, so an empty turn
      surfaced as a user-visible failure. RP-4c wraps that call in the canonical ``SelfHeal`` so a
      malformed-empty (or still-invalid assembled-DAG) turn RE-LAUNCHES the authoring — the MODEL
      re-authors from scratch each attempt. This is d310-LEGAL: the engine injects/alters/
      fabricates NOTHING; the only touch is re-prompt-on-malformed (the ONE permitted output op).
  (b) RAISE THE PER-CALL TIMEOUT 300 -> 600s (a CONFIG VALUE, not a behavior flag/gate) so the
      transient latency spike no longer surfaces as a bare read-timeout, and the long-report
      write phase (~360s) has headroom.

This is the SELF-POLICING test (d311 D):

1. BEHAVIOURAL — a scripted malformed-empty FIRST authoring turn is RECOVERED by the SelfHeal
   wrap: the model re-authors a valid single-node schedule plan on the retry, and the recovered
   node is EXACTLY what the model authored (the engine fabricated nothing).
2. ANTI-FABRICATION — on the recovery the engine adds no synthetic node/content: the DAG carries
   only the model-authored node, its task is the verbatim scripted task, and the heal log shows a
   genuine malformed-json RETRY (not a silent engine-authored placeholder).
3. STRUCTURAL — the acyclic authoring call site STAYS wrapped in ``SelfHeal`` (the fix can't
   silently regress to a bare ``plan()``), the per-call timeout is a raised VALUE (>=600), and no
   schedule/spec-name conditional was introduced into the authoring path (doctrine: behaviour
   lives in shapes/specs, never an engine spec-name/role-name conditional).

Parts 1-2 drive the REAL ``IncrementalPlanner`` + ``SelfHeal`` over a ``FakeTransport`` scripted
with the planner's tool calls, so the whole tool-driven loop + re-author run in-process with zero
inference. Part 3 inspects the shipped source of ``chat_app.agentic`` / ``chat_app.app``.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re
from typing import Sequence

from agent_runtime import HealLog, MalformedOutputError, SelfHeal
from agent_runtime.factory import AbstractPlanFactory
from agent_runtime.incremental import IncrementalPlanner
from llm_framework import FakeTransport
from specialization.registry import SpecRegistry
from specialization.seed import seed_canonical_rulesets

import importlib

import chat_app.agentic as agentic_mod
# NB: ``chat_app.app`` the ATTRIBUTE is the FastAPI instance (re-exported on the package), which
# shadows the submodule for a plain ``import ... as``; import the module object explicitly.
app_mod = importlib.import_module("chat_app.app")


def _run(coro):
    return asyncio.run(coro)


# A schedule-flavoured catalog: the recurring scheduler binds ``cron_add`` on its single node.
_TOOL_CATALOG = [
    {"name": "cron_add", "description": "schedule a recurring task"},
    {"name": "web_search", "description": "search the web for candidate pages"},
    {"name": "file_write", "description": "write content to a file"},
]

_SCHEDULE_TASK = "Every day at 9am, email me a summary of the AI news"


def _seed(shape: str = "single-step") -> str:
    return json.dumps({"tool": "seed_plan", "args": {"shape": shape}})


def _add(task: str, *, tool: str = "", spec: str = "", depends_on: Sequence[str] = ()) -> str:
    return json.dumps(
        {"tool": "add_step",
         "args": {"task": task, "tool": tool, "spec": spec, "specs": [],
                  "depends_on": list(depends_on)}}
    )


def _finalize() -> str:
    return json.dumps({"tool": "finalize_plan", "args": {}})


def _planner(transport: FakeTransport, tmp_path) -> IncrementalPlanner:
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)
    factory = AbstractPlanFactory(reg.index(), tool_catalog=_TOOL_CATALOG)
    return IncrementalPlanner(
        transport,
        factory,
        spec_names=reg.names(),
        tool_names=[t["name"] for t in _TOOL_CATALOG],
        shape_name="single-step",
        shape_description="a single scheduling step that authors one cron_add node",
    )


# The scripted transport: attempt 1 finalizes with ZERO steps authored (the transient empty
# turn -> MalformedOutputError "produced no usable nodes"); attempt 2 authors the real single
# cron_add schedule node. The SelfHeal wrap (exactly as _author_and_drive_acyclic_plan wires it)
# must consume attempt-1's failure and re-launch into attempt-2's valid authoring.
def _empty_then_valid_replies() -> list[str]:
    return [
        _finalize(),                               # attempt 1: finalize with no nodes -> empty
        _seed(),                                   # attempt 2: real authoring
        _add(_SCHEDULE_TASK, tool="cron_add"),
        _finalize(),
    ]


# --------------------------------------------------------------------------- #
# 1. BEHAVIOURAL — a malformed-empty first turn is RECOVERED by the SelfHeal re-author
# --------------------------------------------------------------------------- #
def test_malformed_empty_authoring_is_reauthored_via_selfheal(tmp_path):
    transport = FakeTransport(_empty_then_valid_replies())
    planner = _planner(transport, tmp_path)

    # The wrap is verbatim what _author_and_drive_acyclic_plan wires around the authoring call.
    heal_log = HealLog(label="acyclic_authoring")
    result = _run(
        SelfHeal(max_heals=2).run(
            lambda: planner.plan(_SCHEDULE_TASK),
            label="acyclic_authoring",
            log=heal_log,
        )
    )

    # RECOVERED: the retry authored a valid single-node schedule plan.
    assert result is not None and result.dag is not None
    assert len(result.dag.nodes) == 1, "the re-author should yield the one model-authored node"

    # The heal log proves a GENUINE retry happened (it did not give up, and did not fabricate a
    # plan on the first empty turn): exactly one malformed-json heal, then success.
    assert heal_log.healed is True
    assert len(heal_log.attempts) == 1
    assert heal_log.attempts[0].failure_type == "malformed_json"
    assert heal_log.exhausted is False


def test_first_empty_turn_alone_raises_without_the_wrap(tmp_path):
    # Sanity: WITHOUT the SelfHeal wrap the malformed-empty turn is exactly the user-visible
    # failure RP-4c fixes — so the wrap above is load-bearing, not decorative.
    transport = FakeTransport([_finalize()])  # finalize with zero nodes -> empty
    planner = _planner(transport, tmp_path)
    raised = False
    try:
        _run(planner.plan(_SCHEDULE_TASK))
    except MalformedOutputError as exc:
        raised = True
        assert "no usable nodes" in str(exc)
    assert raised, "a zero-node authoring turn must raise MalformedOutputError (the miss mode)"


# --------------------------------------------------------------------------- #
# 2. ANTI-FABRICATION — the recovered node is the MODEL's, engine fabricated nothing
# --------------------------------------------------------------------------- #
def test_reauthored_node_is_model_authored_engine_fabricates_nothing(tmp_path):
    transport = FakeTransport(_empty_then_valid_replies())
    planner = _planner(transport, tmp_path)
    result = _run(
        SelfHeal(max_heals=2).run(lambda: planner.plan(_SCHEDULE_TASK), label="acyclic_authoring")
    )
    node = result.dag.nodes[0]
    # The node TASK is the VERBATIM string the model authored on the retry — not an engine
    # placeholder, not augmented/reworded by the re-prompt machinery.
    assert node.task == _SCHEDULE_TASK
    # The model bound the cron_add tool it chose; the engine authored no extra node or spec.
    assert node.tool == "cron_add"
    assert node.depends_on == ()
    # The re-author is a fresh model turn: the builder's call trail is the model's own tool
    # calls (seed -> add_step -> finalize), not an engine-synthesised structure.
    trail = [c["tool"] for c in json.loads(result.raw)]
    assert trail == ["seed_plan", "add_step", "finalize_plan"]


# --------------------------------------------------------------------------- #
# 3. STRUCTURAL — the wiring stays: SelfHeal-wrapped authoring, raised timeout VALUE, no flag
# --------------------------------------------------------------------------- #
def test_acyclic_authoring_call_site_is_selfheal_wrapped():
    src = inspect.getsource(agentic_mod._author_and_drive_acyclic_plan)
    # The authoring call is wrapped in a bounded SelfHeal (the re-prompt-on-malformed re-author).
    assert "SelfHeal(max_heals=" in src, "the acyclic authoring lost its SelfHeal re-author wrap"
    assert "authoring_planner.plan(query)" in src
    # It must NOT be a BARE unwrapped call anymore (the RP-4b miss mode).
    assert not re.search(r"=\s*await\s+authoring_planner\.plan\(query\)", src), (
        "authoring_planner.plan(query) is awaited BARE again — the malformed-empty re-author "
        "wrap regressed"
    )
    # The re-author is bounded (a small N), not an unbounded loop.
    m = re.search(r"SelfHeal\(max_heals=(\d+)\)", src)
    assert m is not None and 1 <= int(m.group(1)) <= 4, "the re-author bound must be small"


def test_per_call_timeout_is_a_raised_config_value():
    src = inspect.getsource(app_mod)
    # The live native transport carries the RAISED per-call timeout VALUE (>=600), the RP-4c
    # fix for the transient :11434 read-timeout miss + long-report write headroom.
    m = re.search(r'api="native",\s*keep_alive=-1,\s*timeout=(\d+)', src)
    assert m is not None, "the live OllamaTransport timeout wiring changed shape unexpectedly"
    assert int(m.group(1)) >= 600, "the per-call timeout must be raised to >=600s (RP-4c)"


def test_no_schedule_or_spec_name_conditional_in_authoring():
    # DOCTRINE (d310/d311): behaviour lives in shapes/specs — the authoring path must carry NO
    # spec-name / role-name / schedule conditional that switches engine behaviour. The RP-4c fix
    # is a generic re-author + a timeout value; neither may branch on "this is a schedule task".
    src = inspect.getsource(agentic_mod._author_and_drive_acyclic_plan)
    lowered = src.lower()
    for banned in ("recurring-scheduler", "recurring_scheduler", "is_schedule", "cron_add"):
        assert banned not in lowered, (
            f"the authoring path introduced a schedule/spec-name conditional ({banned!r}); "
            "the re-author must stay generic"
        )
    # No literal string-equality branch on a shape/spec name in the added path.
    assert 'shape_name ==' not in src and "shape_name==" not in src
