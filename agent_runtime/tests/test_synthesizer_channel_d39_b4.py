"""s8/b4 — SYNTHESIZER role: terminal output channel decided by PURE REASONING (d39).

Phase-2 #4 deliverable proof. The terminal output stage (the d39 "synthesizer",
running in the chat run) emits one of THREE channels:

* **FILE**  — the planner authored ``tool=file_write`` on the terminal node;
* **EMAIL** — the planner authored ``tool=send_mail`` (recipient self-only lock, d12);
* **SSE**   — the planner authored NO delivery tool, so the terminal node's text is
  surfaced to the chat (the fallback).

THE WHOLE CHANNEL DECISION IS PURE REASONING (d14): the PLANNER picks the channel by
reasoning over the goal and recording it as the terminal node's ``tool`` — there is NO
``wants_email`` flag, no keyword email-trigger, no ``if ...: send_mail`` logic-gate in
the chat synthesizer path. (The ONLY ``channel == "email"`` branch in the codebase,
``chat_app/workflow.py:103``, is the SCHEDULED daily-brief path — an explicit
user-armed ``DeliverySpec`` — not this interactive synthesizer, so it is out of scope.)
The ONLY hard structure that stays is the ``send_mail`` recipient self-only lock.

These tests are FULLY OFFLINE (``FakeTransport`` + real registered tools over a fake
SMTP / a tmp sandbox; no Ollama / network / GPU). They prove, end to end:

PART A — DECISION is reasoning-driven (the REAL ``IncrementalPlanner``): three goals
  drive the SAME authoring code path; only the model's scripted reasoning differs, and
  the terminal node's channel comes out send_mail / file_write / none accordingly. No
  code branch forces it — proving the channel is the model's reasoning, not a gate.
PART B — EXECUTION + self-lock + SSE (the REAL ``AgentRuntime`` + real tools):
  EMAIL fires only for the email plan and locks the recipient to self EVEN with a
  smuggled ``to``; FILE writes into the sandbox otherwise; the no-tool plan invokes NO
  delivery tool and surfaces its text (SSE fallback).
"""
from __future__ import annotations

import asyncio
import json
import smtplib
from typing import Sequence

import pytest

from agent_runtime.factory import AbstractPlanFactory, PlanDAG, PlanNode
from agent_runtime.incremental import IncrementalPlanner
from agent_runtime.runtime import AgentRuntime
from agent_runtime.status import NodeStatus
from llm_framework import FakeTransport
from reactive_tools import EventPlane, ToolHook, register_agentic_tools
from reactive_tools.config import SmtpConfig
from specialization.registry import SpecRegistry
from specialization.seed import DEEP_RESEARCH_SPEC, seed_canonical_rulesets


def _run(coro):
    return asyncio.run(coro)


# =========================================================================== #
# PART A — the PLANNER decides the channel by REASONING (no logic-gate).
#
# The same tool-driven authoring loop runs for three goals; the scripted model
# reasoning differs only in the terminal step's ``tool``. The channel is whatever
# the model reasoned — there is no code that maps a keyword to a channel.
# =========================================================================== #

_TOOL_CATALOG = [
    {"name": "web_search", "description": "search the web for candidate pages"},
    {"name": "web_fetch", "description": "fetch and extract a page's article text"},
    {"name": "file_write", "description": "write content to a file"},
    {"name": "send_mail", "description": "email the user's own inbox (recipient locked)"},
]


def _seed(shape: str = "linear") -> str:
    return json.dumps({"tool": "seed_plan", "args": {"shape": shape}})


def _add(task: str, *, tool: str = "", depends_on: Sequence[str] = ()) -> str:
    return json.dumps(
        {"tool": "add_step",
         "args": {"task": task, "tool": tool, "spec": "", "specs": [],
                  "depends_on": list(depends_on)}}
    )


def _finalize() -> str:
    return json.dumps({"tool": "finalize_plan", "args": {}})


def _planner(replies: Sequence[str], tmp_path) -> IncrementalPlanner:
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)
    factory = AbstractPlanFactory(reg.index(), tool_catalog=_TOOL_CATALOG)
    return IncrementalPlanner(
        FakeTransport(list(replies)),
        factory,
        spec_names=reg.names(),
        tool_names=[t["name"] for t in _TOOL_CATALOG],
        shape_name="linear",
        shape_description="a gather step then a single deliver step",
    )


def _terminal_node(dag: PlanDAG) -> PlanNode:
    """The SINK node (nothing depends on it) — the synthesizer/deliver step."""
    depended = {d for n in dag.nodes for d in n.depends_on}
    sinks = [n for n in dag.nodes if n.id not in depended]
    assert len(sinks) == 1, f"expected one terminal node, got {[n.id for n in sinks]}"
    return sinks[0]


# Three goals → three reasoned channels. The gather step is identical; ONLY the
# terminal step's reasoned ``tool`` differs (the model's channel decision).
def _replies(deliver_tool: str) -> list[str]:
    return [
        _seed(),
        _add("Research the June 2026 US-Iran situation", tool="web_search"),
        _add("Write the final report and deliver it", tool=deliver_tool,
             depends_on=["n1"]),
        _finalize(),
    ]


def test_planner_reasons_email_channel_when_goal_asks_for_email(tmp_path):
    planner = _planner(_replies("send_mail"), tmp_path)
    dag = _run(planner.plan("Research the US-Iran situation and EMAIL me the report")).dag
    assert _terminal_node(dag).tool == "send_mail"


def test_planner_reasons_file_channel_for_a_written_report(tmp_path):
    planner = _planner(_replies("file_write"), tmp_path)
    dag = _run(planner.plan("Research the US-Iran situation and write it to an HTML file")).dag
    assert _terminal_node(dag).tool == "file_write"


def test_planner_reasons_no_channel_for_a_plain_chat_request(tmp_path):
    # No channel asked → the model leaves the terminal tool empty → SSE fallback.
    planner = _planner(_replies(""), tmp_path)
    dag = _run(planner.plan("Tell me about the June 2026 US-Iran situation")).dag
    assert _terminal_node(dag).tool in (None, "")


def test_channel_decision_is_reasoning_not_a_gate(tmp_path):
    """The authoring path is IDENTICAL across channels; only the reasoning differs.

    Same code, same goal-shape, same tool calls EXCEPT the terminal ``tool`` the model
    reasoned — and the authored channel tracks it 1:1. There is no keyword/flag branch
    that injects a channel; the channel is purely the model's recorded decision.
    """
    channels = {}
    for tool in ("send_mail", "file_write", ""):
        planner = _planner(_replies(tool), tmp_path)
        dag = _run(planner.plan("a report task")).dag
        channels[tool] = _terminal_node(dag).tool or ""
    assert channels == {"send_mail": "send_mail", "file_write": "file_write", "": ""}

    # And the planner PROMPT drives this by REASONING (d14): it instructs the model to
    # use send_mail only when explicitly asked — it is not a runtime logic-gate.
    planner = _planner(_replies(""), tmp_path)
    system = planner._system("a report task")
    initial = planner._initial_user("a report task")
    guidance = (system + initial).lower()
    assert "send_mail" in guidance
    assert "explicitly" in guidance and "email" in guidance
    assert "file_write" in guidance


# =========================================================================== #
# PART B — EXECUTION on the REAL runtime: each authored channel FIRES correctly,
# the send_mail recipient stays self-locked, and the no-tool plan falls back to SSE.
# =========================================================================== #

_CFG = SmtpConfig(
    host="smtp.example.com",
    port=587,
    username="owner@example.com",
    password="app-secret-never-leak",
    from_email="owner@example.com",
)


class _FakeSMTP:
    """Records the sent message; no network (mirrors the b5 wiring proof)."""

    instances: list["_FakeSMTP"] = []

    def __init__(self, host=None, port=None, timeout=None, **kwargs):
        self.host, self.port, self.timeout = host, port, timeout
        self.sent_message = None
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def ehlo(self, *a, **k):
        return (250, b"ok")

    def starttls(self, *a, **k):
        return (220, b"ready")

    def login(self, *a, **k):
        return (235, b"ok")

    def data(self, msg):
        return (250, b"2.0.0 OK")

    def send_message(self, msg, *a, **k):
        self.sent_message = msg
        return {}


@pytest.fixture
def fake_smtp(monkeypatch):
    _FakeSMTP.instances = []
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    return _FakeSMTP


def _hook(tmp_path) -> ToolHook:
    """A hook with the real agentic tools (send_mail self-locked, file_write→tmp)."""
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path,
                           smtp_config=_CFG)
    return hook


def _deliver_dag(deliver_tool: str, tool_args: dict) -> PlanDAG:
    """A 2-node plan: a worker writes the body, the terminal node delivers it."""
    n1 = PlanNode(id="n1", task="Write the report body.")
    n2 = PlanNode(id="n2", task="Deliver the report.", tool=deliver_tool,
                  tool_args=tool_args, depends_on=["n1"])
    return PlanDAG(nodes=[n1, n2], goal="Report on the US-Iran situation")


def test_email_channel_fires_and_locks_recipient_to_self(fake_smtp, tmp_path):
    """EMAIL plan → send_mail fires; a smuggled ``to`` is ignored (recipient = self)."""
    hook = _hook(tmp_path)
    dag = _deliver_dag(
        "send_mail",
        # The model can express only subject+body; we ALSO smuggle a recipient to
        # prove the structural lock holds on the live dispatch path.
        {"subject": "US-Iran report", "body": "the report body",
         "to": "attacker@evil.com"},
    )
    transport = FakeTransport(["the report body", "emailed."])
    result = _run(AgentRuntime(transport=transport, hook=hook).run(dag))

    assert result.states["n2"]["status"] == NodeStatus.DONE.value
    # the email actually went out, locked to the owner's own address
    assert _FakeSMTP.instances, "send_mail did not fire on the EMAIL plan"
    sent = _FakeSMTP.instances[-1].sent_message
    assert sent["To"] == "owner@example.com"
    assert "attacker@evil.com" not in str(sent)
    # the node recorded send_mail as the tool it used
    assert result.results["n2"].tool_used == "send_mail"
    assert result.results["n2"].tool_value["to"] == "owner@example.com"


def test_file_channel_writes_into_the_sandbox(fake_smtp, tmp_path):
    """FILE plan → the explicit ``file_write`` node writes the deliverable (P2).

    AUTONOMY REBUILD P2 (supersedes the raw read-back-loop mechanics, which are
    deleted with the deliverable_path routing): an acyclic node the planner
    explicitly bound to ``file_write`` takes the GENERIC single-tool path — its
    args are grounded from the upstream body (a2-recipe: ``content`` = the real
    upstream report, ``path`` from the plan) and the tool writes ONCE. The
    upstream-grounded content reaches disk raw, and NO email fires."""
    hook = _hook(tmp_path)
    dag = _deliver_dag("file_write", {"path": "us-iran.md"})
    dag.goal = "Write a report on the US-Iran situation to us-iran.md"

    def reply(messages, **opts):
        # n1 (the body worker) produces the report body; n2's post-tool scoped
        # call gets a plain ack emission.
        return "# US-Iran report\nthe body"

    transport = FakeTransport([reply])
    # Mirror the SERVED wiring: the schema emitter grounds file_write.content from
    # the REAL upstream body (a2-recipe) with no extra model call.
    from agent_runtime.toolargs import SchemaToolArgEmitter

    result = _run(AgentRuntime(
        transport=transport, hook=hook,
        tool_arg_emitter=SchemaToolArgEmitter(transport),
    ).run(dag))

    assert result.states["n2"]["status"] == NodeStatus.DONE.value
    assert result.results["n2"].tool_used == "file_write"
    written = tmp_path / "us-iran.md"
    assert written.is_file()
    # the upstream-grounded content landed on disk, raw (never a JSON envelope)
    body = written.read_text(encoding="utf-8")
    assert "US-Iran report" in body
    assert not body.lstrip().startswith("{"), "content must be RAW, never a JSON envelope"
    # the FILE channel never reaches the email channel
    assert not _FakeSMTP.instances, "a file plan must not send email"


def test_sse_fallback_when_no_channel_specified(fake_smtp, tmp_path):
    """No delivery tool → no email, no file; the terminal text is surfaced (SSE)."""
    hook = _hook(tmp_path)
    dag = _deliver_dag("", {})  # the synthesizer reasoned NO channel
    transport = FakeTransport(["the report body", "here is your report"])
    result = _run(AgentRuntime(transport=transport, hook=hook).run(dag))

    assert result.states["n2"]["status"] == NodeStatus.DONE.value
    # NEITHER delivery channel fired
    assert not _FakeSMTP.instances, "no channel asked → no email"
    assert not list(tmp_path.glob("*.md")), "no channel asked → no file written"
    # the terminal node used NO tool and produced the answer the chat streams (SSE)
    assert result.results["n2"].tool_used in (None, "")
    assert result.results["n2"].output and result.results["n2"].output.strip()
