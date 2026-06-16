"""Tests for the COMPILER + ENGINE (HITL gate) + LOADER (a3).

Everything runs fully OFFLINE (d7): research is driven against a MockHook (no
network, like a2's test_research), and the condense uses the engine's default
deterministic offline transport (no GPU / no live phi). The load-bearing proofs
the action requires:

- the UI path does NOT compile without approval, but DOES after approval;
- compile-without-approval is REFUSED (compile is structurally unreachable
  without a user-facing approver) — the negative test;
- the AUTONOMOUS path authors a draft and compiles ONLY through the SAME
  user-facing gate (no auto-approve bypass);
- the loader returns ONE body.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from specialization import compiler
from specialization.engine import (
    ApprovalDenied,
    ApprovalRequired,
    ApprovalToken,
    SOURCE_AUTONOMOUS,
    SOURCE_UI,
    SpecDraft,
    SpecializationEngine,
)
from specialization.loader import SpecLoader
from specialization.model import RawDefinition
from specialization.registry import SpecRegistry


# --------------------------------------------------------------------------- #
# Offline mock tool hook (no network) — same contract as a2's MockHook.
# --------------------------------------------------------------------------- #
@dataclass
class _Result:
    ok: bool
    value: Any = None
    error: str | None = None


class MockHook:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def invoke(self, name: str, /, **kwargs: Any) -> _Result:
        self.calls.append((name, kwargs))
        if name == "web_search":
            q = kwargs.get("query", "")
            results = [
                {
                    "title": f"Guide {i} for {q}",
                    "url": f"https://example.com/{abs(hash(q)) % 1000}/{i}",
                    "snippet": f"How to do step {i} for {q}.",
                }
                for i in range(5)
            ]
            return _Result(ok=True, value={"query": q, "results": results, "count": 5})
        if name == "web_fetch":
            url = kwargs.get("url", "")
            return _Result(ok=True, value={
                "url": url, "title": f"Title of {url}",
                "text": f"Detailed how-to content for {url}. " * 40,
            })
        return _Result(ok=False, error=f"unknown tool {name!r}")


def _run(coro):
    return asyncio.run(coro)


def _engine(tmp_path):
    reg = SpecRegistry(tmp_path / "specs")
    return SpecializationEngine(reg, hook=MockHook(), specs_dir=tmp_path / "specs"), reg


_RAW = RawDefinition(
    name="markdown-reporter",
    description="write detailed markdown reports",
    intent="produce clear, well-structured markdown reports on a research topic",
)


# --------------------------------------------------------------------------- #
# Approvers — these stand in for the user-facing approval surface.
# --------------------------------------------------------------------------- #
async def _approve(draft: SpecDraft) -> ApprovalToken:
    """A user-facing surface that APPROVES the surfaced draft."""
    return ApprovalToken.grant(draft)


async def _deny(draft: SpecDraft) -> ApprovalToken:
    """A user-facing surface that DECLINES."""
    return ApprovalToken.deny(draft)


# --------------------------------------------------------------------------- #
# COMPILER — the chain-driven offline condense is genuinely exercised.
# --------------------------------------------------------------------------- #
def test_condense_body_runs_the_chain_offline():
    hook = MockHook()
    trace = _run(__import__("specialization.research", fromlist=["research"]).research(
        "markdown reports", "concise", hook=hook))
    body = compiler.condense_body(_RAW, trace)  # transport=None -> offline
    assert body.strip()
    assert "# Specialist: markdown-reporter" in body
    assert compiler.OFFLINE_MARKER in body          # offline, not a faked live reply
    # The chain really condensed the research notes into the body.
    assert "## How (distilled from research)" in body


def test_offline_condense_handles_empty_trace():
    from specialization.research import ResearchTrace
    body = compiler.offline_condense_body(_RAW, ResearchTrace(skill="x", intent="y"))
    assert "# Specialist:" in body and body.strip()


# --------------------------------------------------------------------------- #
# UI / HITL path — no compile without approval, compiles after approval.
# --------------------------------------------------------------------------- #
def test_ui_path_does_not_compile_without_approval(tmp_path):
    engine, reg = _engine(tmp_path)
    # approver=None -> compile is structurally unreachable (the d9 guarantee).
    with pytest.raises(ApprovalRequired):
        _run(engine.ui_specialize(_RAW, approver=None))
    # Nothing was registered.
    assert reg.names() == []


def test_ui_path_compiles_after_approval(tmp_path):
    engine, reg = _engine(tmp_path)
    spec = _run(engine.ui_specialize(_RAW, approver=_approve))
    assert spec.name == "markdown-reporter"
    assert spec.source == SOURCE_UI
    assert spec.body.strip()
    # It was actually registered (compile-on-approval write, d8).
    assert reg.names() == ["markdown-reporter"]
    assert reg.load("markdown-reporter").body == spec.body


def test_ui_path_declined_does_not_compile(tmp_path):
    engine, reg = _engine(tmp_path)
    with pytest.raises(ApprovalDenied):
        _run(engine.ui_specialize(_RAW, approver=_deny))
    assert reg.names() == []


# --------------------------------------------------------------------------- #
# The NEGATIVE test — compile-without-approval is REFUSED at the gate level.
# --------------------------------------------------------------------------- #
def test_compile_without_approver_is_refused(tmp_path):
    engine, reg = _engine(tmp_path)
    draft = _run(engine.author_draft(_RAW, source=SOURCE_UI))
    # A draft exists (researched + authored) but compile is unreachable with no
    # user-facing approver — there is NO code path to a registered spec.
    with pytest.raises(ApprovalRequired):
        _run(engine.compile(draft, approver=None))
    assert reg.names() == []


def test_stale_or_forged_token_is_refused(tmp_path):
    """Approval must be a decision about THIS draft — a token whose challenge
    does not match the surfaced draft is rejected (not a flippable boolean)."""
    engine, reg = _engine(tmp_path)
    draft = _run(engine.author_draft(_RAW, source=SOURCE_UI))

    async def forged(_draft: SpecDraft) -> ApprovalToken:
        return ApprovalToken(challenge="not-the-right-challenge", approved=True)

    with pytest.raises(ApprovalDenied):
        _run(engine.compile(draft, approver=forged))
    assert reg.names() == []


# --------------------------------------------------------------------------- #
# AUTONOMOUS path — authors a draft, but compile STILL routes the SAME gate.
# --------------------------------------------------------------------------- #
def test_autonomous_path_authors_draft_and_compiles_only_through_gate(tmp_path):
    engine, reg = _engine(tmp_path)
    # The autonomous path authors with NO human; the draft is source=autonomous.
    draft = _run(engine.author_draft(_RAW, source=SOURCE_AUTONOMOUS))
    assert draft.source == SOURCE_AUTONOMOUS
    assert draft.body.strip()  # authored autonomously, no UI involved

    # Compile still requires the user-facing gate (no auto-approve bypass).
    spec = _run(engine.compile(draft, approver=_approve))
    assert spec.source == SOURCE_AUTONOMOUS
    assert reg.names() == ["markdown-reporter"]


def test_autonomous_path_has_no_auto_approve_bypass(tmp_path):
    engine, reg = _engine(tmp_path)
    # Even fully autonomous, with no approver there is NO bypass — refused.
    with pytest.raises(ApprovalRequired):
        _run(engine.autonomous_specialize(_RAW, approver=None))
    assert reg.names() == []


def test_autonomous_end_to_end_through_gate(tmp_path):
    engine, reg = _engine(tmp_path)
    spec = _run(engine.autonomous_specialize(_RAW, approver=_approve))
    assert spec.source == SOURCE_AUTONOMOUS
    assert reg.load("markdown-reporter").source == SOURCE_AUTONOMOUS


def test_draft_renders_markdown_and_html_for_surface(tmp_path):
    engine, _ = _engine(tmp_path)
    draft = _run(engine.author_draft(_RAW, source=SOURCE_AUTONOMOUS))
    md = draft.to_markdown()
    html = draft.to_html()
    assert md.startswith("---")  # the compiled markdown-with-frontmatter doc
    assert "markdown-reporter" in md
    assert "<section class='spec-draft'" in html and "markdown-reporter" in html


# --------------------------------------------------------------------------- #
# LOADER — returns ONE body, scoped to a single named spec (d10).
# --------------------------------------------------------------------------- #
def test_loader_returns_one_body(tmp_path):
    engine, reg = _engine(tmp_path)
    spec = _run(engine.ui_specialize(_RAW, approver=_approve))

    loader = SpecLoader(reg)
    body = loader.load_body("markdown-reporter")
    assert body == spec.body
    assert isinstance(body, str) and body.strip()

    # The loader surface is single-spec-only: no index / names enumeration.
    assert not hasattr(loader, "index")
    assert not hasattr(loader, "names")
    # Unknown spec -> KeyError (it loads exactly the one you name, or nothing).
    with pytest.raises(KeyError):
        loader.load_body("does-not-exist")


def test_loader_full_spec_is_still_single(tmp_path):
    engine, reg = _engine(tmp_path)
    _run(engine.ui_specialize(_RAW, approver=_approve))
    loader = SpecLoader(reg)
    spec = loader.load("markdown-reporter")
    assert spec.name == "markdown-reporter"
    assert spec.body == loader.load_body("markdown-reporter")
