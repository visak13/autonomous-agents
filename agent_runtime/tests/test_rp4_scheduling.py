"""RP-4 (d322/d332) — the SCHEDULE-LEG is SPEC-DRIVEN, the engine authors NOTHING.

Self-policing tests (the anti-fab charter's grep-style guards) for the RP-4 deliverable:

1. The ``recurring-scheduler`` scheduling SPEC exists, is selection-grade, and its body
   carries the whole-DAG scheduling doctrine (store the WHOLE task as the cron prompt;
   schedule-only; no sub-step; no re-schedule wrapper).
2. NO engine cron output-override: the retired ``cron_prompt_from_task`` string-surgery
   is GONE, and ``SchemaToolArgEmitter.__call__`` no longer branches on ``node.tool ==
   'cron_add'`` to rewrite the model's prompt. The engine stores the model's prompt VERBATIM.
3. NO engine spec-name conditional: ``recurring-scheduler`` appears in engine files only as
   PROMPT doctrine text (behavior-via-prompting), never in a code ``if ... ==`` conditional.
4. The ``cron_add`` TOOL stays GENERIC / flow-agnostic — its description + arg schema speak
   only of a schedule + a prompt, with no research/report/email/flow awareness.

These are behavior-free (no GPU, no network): they read source + the seeded spec/tool defs.
The whole-DAG FIRE (run_agentic(job.prompt) fresh) is proven in
``chat_app/tests/test_cron_scheduler_s6.py``; the live authoring reliability + the
model-authored whole-task prompt are proven by the RP-4 bounded live measure (worklog).
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import agent_runtime.toolargs as toolargs
from agent_runtime.toolargs import SchemaToolArgEmitter
from agent_runtime import factory as factory_mod
from agent_runtime import shape_selector as shape_selector_mod

from specialization.seed import (
    CANONICAL_RULESETS,
    RECURRING_SCHEDULER_SPEC,
    RECURRING_SCHEDULER_RULESET,
)
from reactive_tools.cron_tools import build_cron_tools


# --------------------------------------------------------------------------- #
# 1) the scheduling SPEC exists + is selection-grade + carries whole-DAG doctrine
# --------------------------------------------------------------------------- #
def test_recurring_scheduler_spec_exists_and_is_selection_grade():
    assert RECURRING_SCHEDULER_SPEC == "recurring-scheduler"
    assert RECURRING_SCHEDULER_SPEC in CANONICAL_RULESETS
    description, body = CANONICAL_RULESETS[RECURRING_SCHEDULER_SPEC]

    # selection-grade description: states WHEN to bind + the schedule-only contract,
    # and binds to the cron/schedule node only.
    low_d = description.lower()
    assert len(description) > 120
    assert "recurring" in low_d and "cron_add" in low_d
    assert "schedule-only" in low_d or "schedule only" in low_d
    assert "whole" in low_d  # stores the WHOLE task
    assert "bind" in low_d


def test_recurring_scheduler_body_carries_whole_dag_doctrine():
    body = RECURRING_SCHEDULER_RULESET
    low = body.lower()
    # whole self-contained task as the cron prompt (the scheduled unit is the whole task)
    assert "whole task" in low
    assert "cron_add" in low and "prompt" in low
    # schedule-only, no run-now
    assert "no run-now" in low or "do not also perform the task now" in low or "one schedule leg" in low
    # not a sub-step, not a re-schedule wrapper
    assert "sub-step" in low
    assert "re-schedule" in low or "reschedule" in low
    # it is a METHODOLOGY ruleset (shapes HOW you schedule), not an output-format spec
    assert "schedule-leg methodology" in low or "how you schedule" in low


def test_recurring_scheduler_is_not_an_output_writer_spec():
    """It authors no deliverable, so it must NOT carry the coherent-artifact writer
    doctrine (that belongs to the format-writer specs, not this methodology spec)."""
    body = RECURRING_SCHEDULER_RULESET
    # a sentinel phrase unique to _COHERENT_ARTIFACT_DOCTRINE
    assert "AUTHOR A COHERENT, SELF-CONTAINED ARTIFACT" not in body


# --------------------------------------------------------------------------- #
# 2) NO engine cron output-override — the string surgery is gone; emitter verbatim
# --------------------------------------------------------------------------- #
def test_cron_prompt_from_task_surgery_is_removed():
    # the function no longer exists (removed, not just unexported)
    assert not hasattr(toolargs, "cron_prompt_from_task")
    assert "cron_prompt_from_task" not in getattr(toolargs, "__all__", [])


def test_emitter_does_not_rewrite_cron_add_prompt():
    """SchemaToolArgEmitter.__call__ must not branch on the cron_add TOOL NAME to
    rewrite the model-emitted prompt (the retired d310/d311/d319 fabrication)."""
    src = inspect.getsource(SchemaToolArgEmitter.__call__)
    # strip comment lines so a doc-comment naming the RETIRED surgery is not a match;
    # we assert against real CODE only.
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    # no tool-name conditional that overrides the prompt for cron_add
    assert 'node.tool == "cron_add"' not in code
    assert "node.tool == 'cron_add'" not in code
    # no CALL to the retired string-surgery
    assert "cron_prompt_from_task(" not in code


# --------------------------------------------------------------------------- #
# 3) NO engine spec-name conditional — spec name is prompt doctrine only, not code
# --------------------------------------------------------------------------- #
def test_no_engine_spec_name_conditional_for_recurring_scheduler():
    """recurring-scheduler may appear as PROMPT doctrine text (behavior-via-prompting),
    but NEVER in a code conditional (`if ... == 'recurring-scheduler'`) — that would be
    the engine special-casing a spec name, the anti-fab violation the charter forbids."""
    engine_files = [
        Path(factory_mod.__file__),
        Path(shape_selector_mod.__file__),
        Path(toolargs.__file__),
    ]
    cond = re.compile(r"==\s*['\"]recurring-scheduler['\"]|['\"]recurring-scheduler['\"]\s*==|"
                      r"if\s+[^\n]*\brecurring-scheduler\b[^\n]*:")
    for f in engine_files:
        text = f.read_text(encoding="utf-8")
        for m in cond.finditer(text):
            # allow the match only if it is inside a string literal that is clearly a
            # prompt line (heuristic: the surrounding line is a quoted doctrine string,
            # not an if/comparison). A code conditional has no leading quote+text pattern.
            line = text[text.rfind("\n", 0, m.start()) + 1: text.find("\n", m.end())]
            assert line.lstrip().startswith(('"', "'", ")", "+")) or "if " not in line, (
                f"engine spec-name conditional for recurring-scheduler in {f.name}: {line.strip()!r}"
            )


# --------------------------------------------------------------------------- #
# 4) the cron_add TOOL stays generic / flow-agnostic
# --------------------------------------------------------------------------- #
def test_cron_add_tool_is_generic(tmp_path):
    defs = {d.name: d for d in build_cron_tools(data_dir=tmp_path)}
    add = defs["cron_add"]
    # The selection-facing TOOL DESCRIPTION advertises a GENERIC schedule + prompt
    # capability and carries NO flow/domain awareness. (An illustrative example inside
    # an ARG-field description — 'e.g. research …' — is fine: it shows the tool accepts
    # ANY task as its prompt, it is not flow LOGIC. So we assert on the tool description,
    # which is what the planner reasons over to pick the tool.)
    desc = add.description.lower()
    assert "schedule" in desc and ("prompt" in desc or "plan" in desc)
    for flow_word in ("research", "report", "email", "html", "markdown", "summary", "news"):
        assert flow_word not in desc, f"cron_add tool DESCRIPTION leaked flow word {flow_word!r}"
    # the arg schema stays a plain schedule + prompt + name shape (no flow-keyed fields)
    props = set(add.args_model.model_json_schema()["properties"])
    assert props == {"schedule", "prompt", "name"}
