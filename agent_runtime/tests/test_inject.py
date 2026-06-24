"""Produce-step ruleset-INJECTION test (a1) — fully OFFLINE (FakeTransport, no GPU).

The d1 redefinition under test: a specialization body is an OUTPUT-SHAPING
RULESET the runtime injects as a SHAPING layer OVER the real task content — not a
"how to <skill>" document the agent recites (round-1's Iran->markdown-how-to
bug). This test drives the REAL produce step (``SubAgent.run``) with a scripted
:class:`FakeTransport` and inspects the exact messages the transport received:

- POSITIVE (shaping seam): the SYSTEM turn carries the seeded markdown ruleset +
  the shaping framing; the USER turn carries the real task content + the tool
  findings (and the ruleset never leaks into the user turn, nor the findings into
  the system turn).
- NEGATIVE (the bug is gone): the SYSTEM turn is NOT a skill how-to — it does not
  contain a "how to <skill>" description, and specifically is NOT the round-1
  research-distilled body (``# Specialist: ... / ## How (distilled from
  research)``) that caused the bug.

Everything is in-process + offline (d2/d7): no Ollama, no network, no GPU.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from agent_runtime.factory import PlanNode
from agent_runtime.identity import AGENT_IDENTITY
from agent_runtime.runtime import _SHAPING_FRAMING, SubAgent
from agent_runtime.scope import ScopedSpec
from llm_framework import FakeTransport
from specialization import compiler
from specialization.loader import SpecLoader
from specialization.model import RawDefinition
from specialization.registry import SpecRegistry
from specialization.research import HowNote, ResearchTrace, SourceRef
from specialization.seed import (
    MARKDOWN_WRITER_RULESET,
    SOURCE_SEED,
    seed_ruleset_spec,
)


def _run(coro):
    return asyncio.run(coro)


# A real (current) news-style task + findings — what the agent ACTUALLY researched.
_TOPIC = "the latest Iran nuclear-talks update"
_FINDINGS = (
    "Reuters reports a fresh round of indirect talks concluded on Tuesday with no "
    "breakthrough; both sides agreed to reconvene. AP notes inspectors regained "
    "limited site access. Key fact: sanctions relief remains the main sticking point."
)


# --------------------------------------------------------------------------- #
# A tiny in-process tool hook so the produce step has REAL tool findings to
# carry on the user turn (mirrors the runtime's ToolHook contract).
# --------------------------------------------------------------------------- #
@dataclass
class _ToolResult:
    ok: bool
    value: Any = None
    error: Optional[str] = None
    call_id: str = "call-1"


class _FindingsHook:
    """Returns the research findings for ``web_search``/``web_fetch`` (no network)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def invoke(self, name: str, /, **kwargs: Any) -> _ToolResult:
        self.calls.append((name, kwargs))
        return _ToolResult(ok=True, value=_FINDINGS)


def _markdown_node() -> PlanNode:
    """A produce node that runs ONE generic tool then shapes its output by the markdown
    ruleset. Uses ``web_fetch`` (a non-research single-tool node) so it exercises the
    GENERIC produce path's system/user shaping seam — a ``web_search`` node now runs the
    agentic research LOOP (s9/c5), covered separately in test_research_read_fetch.py."""
    return PlanNode(
        id="n1",
        task=f"Read about {_TOPIC} and report the findings.",
        spec="markdown-writer",
        tool="web_fetch",
        tool_args={"url": "https://example.org/iran-talks"},
    )


# --------------------------------------------------------------------------- #
# POSITIVE: the produce step injects the ruleset as a SHAPING layer over the
# REAL task content + findings.
# --------------------------------------------------------------------------- #
def test_produce_step_injects_ruleset_as_shaping_layer_over_real_task(tmp_path):
    # Seed the markdown-writer SHAPING ruleset via the programmatic seed path (d1).
    reg = SpecRegistry(tmp_path / "specs")
    seeded = seed_ruleset_spec(
        reg, "markdown-writer",
        "Shape findings into clean GFM (headings, lists, summary, sources).",
        MARKDOWN_WRITER_RULESET,
    )
    assert seeded.source == SOURCE_SEED            # honest origin, no research/gate
    assert seeded.body == MARKDOWN_WRITER_RULESET.strip()

    # Resolve the ONE spec body the runtime would hand the sub-agent (d10).
    loader = SpecLoader(reg)
    scope = ScopedSpec.resolve(loader, "markdown-writer")

    transport = FakeTransport(["# Iran nuclear-talks update\n\n**Summary.** ..."])
    node = _markdown_node()
    agent = SubAgent(node, transport=transport, scope=scope, hook=_FindingsHook())
    result = _run(agent.run())

    # Exactly one phi call was made; inspect the messages it received.
    assert transport.call_count == 1
    messages = transport.calls[0]["messages"]
    system = next(m["content"] for m in messages if m["role"] == "system")
    user = next(m["content"] for m in messages if m["role"] == "user")

    # --- SYSTEM carries the SHAPING ruleset (+ the explicit shaping framing) --- #
    assert _SHAPING_FRAMING in system               # the produce-step shaping seam
    assert MARKDOWN_WRITER_RULESET.strip() in system  # the seeded ruleset body
    # Distinctive shaping instructions are present (it governs FORM).
    assert "level-1 heading" in system and "## Sources" in system

    # --- USER carries the REAL task content + the tool findings --- #
    assert _TOPIC in user                            # the actual task
    assert "report the findings" in user
    assert "TOOL OUTPUT (web_fetch)" in user
    assert "sanctions relief remains the main sticking point" in user  # findings

    # --- separation of concerns: ruleset NOT in the user turn; findings NOT in
    # the system turn (the system shapes, the user is the task content) --- #
    assert "## Sources" not in user
    assert "sanctions relief" not in system
    assert _TOPIC not in system

    # The tool actually fired (the agent DID the task before shaping).
    assert agent.hook.calls and agent.hook.calls[0][0] == "web_fetch"
    assert result.tool_used == "web_fetch"


# --------------------------------------------------------------------------- #
# NEGATIVE: the SYSTEM turn is NOT a skill how-to — the round-1 failure mode is
# gone. The injected ruleset is a shaping layer, not a "how to write markdown".
# --------------------------------------------------------------------------- #
def test_injected_system_is_not_a_skill_how_to(tmp_path):
    reg = SpecRegistry(tmp_path / "specs")
    seed_ruleset_spec(
        reg, "markdown-writer",
        "Shape findings into clean GFM (headings, lists, summary, sources).",
        MARKDOWN_WRITER_RULESET,
    )
    scope = ScopedSpec.resolve(SpecLoader(reg), "markdown-writer")

    transport = FakeTransport(["# report"])
    agent = SubAgent(_markdown_node(), transport=transport, scope=scope, hook=_FindingsHook())
    _run(agent.run())
    system = next(
        m["content"] for m in transport.calls[0]["messages"] if m["role"] == "system"
    )

    # (a) No "how to <skill>" framing anywhere in the system turn — the ruleset
    #     shapes output, it does not teach the skill.
    low = system.lower()
    assert "how to write markdown" not in low
    assert "how to " not in low

    # (b) Construct the ROUND-1 research-distilled body for the SAME skill and
    #     prove the injected system is NOT that how-to document. round-1 compiled
    #     a spec by RESEARCHING "how to <skill>"; its body has the tell-tale
    #     "# Specialist: ... / ## How (distilled from research)" structure.
    raw = RawDefinition(
        name="markdown-writer",
        description="how to write markdown",
        intent="research the how of writing markdown",
    )
    how_to_trace = ResearchTrace(
        skill="how to write markdown", intent="",
        notes=[HowNote(source="search:how to write markdown", kind="search_snippet",
                       how="Use # for headings and - for bullet lists.")],
        sources=[SourceRef(url="https://example.com/md", title="Markdown guide", fetched=True)],
    )
    round1_body = compiler.offline_condense_body(raw, how_to_trace)
    assert "## How (distilled from research)" in round1_body  # it IS a how-to body
    # The injected system is the SHAPING ruleset, NOT that research how-to.
    assert "## How (distilled from research)" not in system
    assert round1_body not in system


# --------------------------------------------------------------------------- #
# A node with NO spec is a bare step: d11 now rides the UNIVERSAL IDENTITY as its
# system turn, but still NO shaping layer (the shaping framing only rides
# alongside a real ruleset body) — the test's real intent (no shaping leakage).
# --------------------------------------------------------------------------- #
def test_bare_node_has_only_identity_no_shaping_layer():
    transport = FakeTransport(["plain answer"])
    node = PlanNode(id="n1", task="Summarise the inputs.")  # no spec
    agent = SubAgent(node, transport=transport)
    _run(agent.run())
    system = next(
        m["content"] for m in transport.calls[0]["messages"] if m["role"] == "system"
    )
    assert system == AGENT_IDENTITY                  # identity only — no shaping layer
    assert _SHAPING_FRAMING not in system
