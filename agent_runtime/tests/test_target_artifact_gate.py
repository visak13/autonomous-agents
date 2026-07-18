"""CoT-autonomy P3 — the TARGET-ARTIFACT bounce-gate is DELETED (owner ruling:
no babysitting; no engine turn ever re-prompts the model's conclusion).

What remains here:
* the finish-reason parse-to-read contract (a finish call's model-authored reason is
  the node output, never the raw tool-call JSON),
* the plain-worker path (one prose turn, no spurious tool use),
* self-policing that the gate stays deleted.

Delivery honesty now lives DOWNSTREAM: the persistence-side staleness guard (unchanged
bytes ⇒ no artifact), the truthful ``plan_chain.deliverable_bytes`` trace attr, and the
reviewer node reading the real file.
"""

import asyncio
import inspect

from llm_framework.transport import ChatResult

from agent_runtime.factory import PlanNode
from agent_runtime.runtime import SubAgent


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


def test_prose_conclusion_is_accepted_unbounced_even_with_a_declared_target():
    """P3: a node with a declared deliverable that concludes in prose is ACCEPTED on
    that turn — no engine bounce. The dishonesty is surfaced downstream (staleness
    guard / deliverable_bytes), never by re-prompting the model."""
    hook = _FileHook()
    prose = "the report as prose, never written to disk"
    transport = _Script([prose])
    agent = SubAgent(
        _write_node(), transport=transport, hook=hook,
        deliverable_path="report.html",
    )
    out = asyncio.run(agent.run({}))
    assert transport.calls == 1          # ONE turn — no bounce, no re-prompt
    assert hook.calls == []              # nothing was written (honest downstream)
    assert prose in (out.output or "")


def test_no_deliverable_node_is_untouched():
    """A plain worker (no declared deliverable) answers in ONE prose turn."""
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


def test_bounce_gates_stay_deleted():
    """Self-policing: no gate state or bounce constants return to the loop."""
    src = inspect.getsource(SubAgent._run_research_loop)
    for gone in ("target_gate", "note_gate", "gather_more",
                 "_RESEARCH_GATHER_MORE", "_RESEARCH_NOTE_GATE"):
        assert gone not in src, f"{gone} must stay deleted from the loop (P3)"


def test_lenient_recovery_dispatches_a_big_content_call_with_one_bad_escape():
    """P6 channel robustness: an unambiguous tool-shaped reply whose multi-KB content
    string breaks strict JSON (bad escape / missing outer brace) is recovered VERBATIM
    and dispatched — the model's own bytes, nothing composed."""
    from agent_runtime.runtime import _lenient_content_call

    BS = chr(92)
    raw = (
        '{"tool": "file_write", "args": {"append": false, "content": '
        '"<!DOCTYPE html>' + BS + 'n<p>a ' + BS + 'x bad escape and a '
        + BS + '"quote' + BS + '"</p>", "path": "r.html"}'  # missing outer brace too
    )
    r = _lenient_content_call(raw)
    assert r is not None
    tool, args = r
    assert tool == "file_write"
    assert args["path"] == "r.html" and args["append"] is False
    assert '<p>a ' + BS + 'x bad escape and a "quote"</p>' in args["content"]
    assert "\n" in args["content"]  # standard escapes decoded
