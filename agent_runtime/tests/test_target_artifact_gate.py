"""TARGET-ARTIFACT acceptance gate (autonomy rebuild P2 — the Gate-2e lesson).

A node whose plan declared a deliverable FILE (``deliverable_path`` — model-named
DATA stamped by the write phase, never an engine regex) must not conclude as if it
delivered when it never wrote the file. Live Gate-2e: the one-node write plan
answered turn 1 with the whole document as PROSE — never loaded the file bundle —
and the run then shipped a STALE file from a previous session as the artifact.

The gate is the same KEEP-class as the no-fab GATHER-MORE gate: it verifies a
planner-declared postcondition with an actionable TOOL error observation; it never
edits or composes the model's bytes; it is bounded (on exhaustion the prose stands
and persistence reports honestly that no file was produced).
"""

import asyncio

from llm_framework.transport import ChatResult

from agent_runtime.factory import PlanNode
from agent_runtime.runtime import SubAgent, _TARGET_GATE_MAX


class _FileHook:
    """Minimal hook: acks a file_write with a WRITE-shaped value ({path, bytes})."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def invoke(self, name: str, **args):
        self.calls.append(name)
        path = args.get("path") or ""

        class _R:
            ok = True
            error = ""
            value = {"path": path, "bytes": 120}

        return _R()


class _Script:
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


def _write_node(task: str = "Write the report.\n\nWrite to the file 'report.html'."):
    return PlanNode(id="n1", task=task, role="worker")


def test_prose_only_conclusion_is_bounced_then_write_is_accepted():
    """Turn-1 prose on a deliverable node bounces with an actionable tool error;
    after a real file_write lands on the target, the conclusion is accepted."""
    hook = _FileHook()
    transport = _Script([
        "<!DOCTYPE html><html>the whole report as prose</html>",   # bounced
        '{"tool": "get_bundles", "args": {"name": "file"}}',
        '{"tool": "file_write", "args": {"path": "report.html", "content": "<!DOCTYPE html>..."}}',
        "Report written to report.html.",                          # accepted
    ])
    agent = SubAgent(
        _write_node(), transport=transport, hook=hook,
        deliverable_path="report.html",
    )
    out = asyncio.run(agent.run({}))
    assert "file_write" in hook.calls
    assert "Report written" in (out.output or "")
    # the bounce really happened (4 turns consumed, not 1)
    assert transport.calls == 4


def test_gate_error_names_the_declared_target_and_the_file_bundle():
    hook = _FileHook()
    seen: list[list] = []

    class _Spy(_Script):
        def chat(self, messages, **opts):
            seen.append([m for m in messages])
            return super().chat(messages, **opts)

    transport = _Spy([
        "prose conclusion without any write",   # bounced → error observation
        '{"tool": "get_bundles", "args": {"name": "file"}}',
        '{"tool": "file_write", "args": {"path": "report.html", "content": "x"}}',
        "done.",
    ])
    agent = SubAgent(
        _write_node(), transport=transport, hook=hook,
        deliverable_path="report.html",
    )
    asyncio.run(agent.run({}))
    # the SECOND call's history carries the gate observation as a TOOL turn
    flat = "\n".join(str(m) for m in seen[1])
    assert "report.html" in flat
    assert "get_bundles" in flat and "file_write" in flat
    assert "NOT saved to disk" in flat


def test_gate_is_bounded_and_salvages_the_prose_on_exhaustion():
    """A model that never writes is bounced _TARGET_GATE_MAX times, then its prose
    stands (honest downstream: persistence sees no fresh file and drops the artifact)."""
    hook = _FileHook()
    prose = "the report as prose, never written to disk"
    transport = _Script([prose] * (_TARGET_GATE_MAX + 1))
    agent = SubAgent(
        _write_node(), transport=transport, hook=hook,
        deliverable_path="report.html",
    )
    out = asyncio.run(agent.run({}))
    assert hook.calls == []                          # nothing was ever written
    assert transport.calls == _TARGET_GATE_MAX + 1   # bounded, no infinite loop
    assert prose in (out.output or "")               # prose salvaged, not discarded


def test_finish_reason_becomes_the_findings_not_the_call_json():
    """A finish tool call's model-authored reason is the node output (parse-to-read);
    the raw tool-call JSON syntax must not leak as the findings (live Gate-2f wart)."""
    hook = _FileHook()
    transport = _Script([
        '{"tool": "get_bundles", "args": {"name": "file"}}',
        '{"tool": "file_write", "args": {"path": "report.html", "content": "<!DOCTYPE html>..."}}',
        '{"tool": "finish", "args": {"reason": "Report authored and written to report.html."}}',
    ])
    agent = SubAgent(
        _write_node(), transport=transport, hook=hook,
        deliverable_path="report.html",
    )
    out = asyncio.run(agent.run({}))
    assert (out.output or "").strip() == "Report authored and written to report.html."


def test_no_deliverable_node_is_untouched_by_the_gate():
    """A plain worker (no declared deliverable) still answers in ONE prose turn."""
    hook = _FileHook()
    transport = _Script(["Hello! How can I help you today?"])
    agent = SubAgent(
        PlanNode(id="w1", task="say hi", role="worker"),
        transport=transport, hook=hook,
    )
    out = asyncio.run(agent.run({}))
    assert transport.calls == 1
    assert hook.calls == []
    assert "How can I help" in (out.output or "")
