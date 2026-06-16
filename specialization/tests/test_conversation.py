"""Tests for the conversational spec-authoring CORE (conversation.py).

Fully OFFLINE (d7): no transport is injected, so the author/refine redrafts run
through the deterministic offline FakeTransport path — reproducible, GPU-free,
no live phi. The load-bearing proofs the action requires:

- start -> refine CHANGES the body;
- refine INCORPORATES the critique token into the body;
- approve COMPILES + REGISTERS a LOADABLE spec.

Plus guards: the d1 output-shaping (not skill-how-to) framing, lifecycle
ordering, terminal-state refusal, and the live-transport seam + empty-fallback.
"""
from __future__ import annotations

import pytest

from llm_framework import FakeTransport
from specialization.compiler import OFFLINE_MARKER
from specialization.conversation import (
    ConversationError,
    DraftPreview,
    SOURCE_UI,
    SpecConversation,
    STATE_APPROVED,
    STATE_CANCELLED,
    STATE_DENIED,
)
from specialization.model import RawDefinition
from specialization.registry import SpecRegistry


_RAW = RawDefinition(
    name="concise-brief",
    description="shape findings into a tight executive brief",
    intent="produce a short, skimmable executive brief of the findings",
)


def _conv(tmp_path, *, transport=None):
    reg = SpecRegistry(tmp_path / "specs")
    return SpecConversation(_RAW, registry=reg, transport=transport), reg


# --------------------------------------------------------------------------- #
# start: authors an initial OUTPUT-SHAPING ruleset (d1), returns a preview.
# --------------------------------------------------------------------------- #
def test_start_authors_output_shaping_ruleset(tmp_path):
    conv, _ = _conv(tmp_path)
    preview = conv.start("make a tight executive brief")
    assert isinstance(preview, DraftPreview)
    assert preview.turn == 1
    body = conv.body
    assert body.strip()
    # It is an output-shaping ruleset, not a 'how to' skill doc (d1).
    assert "Output-shaping ruleset" in body
    assert "**Mission.**" in body
    lowered = body.lower()
    assert "how to" not in lowered and "tutorial" not in lowered
    # Offline, deterministic — marked, not a faked live reply.
    assert OFFLINE_MARKER in body
    # The compiled preview is the real markdown-with-frontmatter doc.
    md = preview.to_markdown()
    assert md.startswith("---") and "concise-brief" in md


# --------------------------------------------------------------------------- #
# THE required proof: start -> refine CHANGES the body and incorporates the
# critique token.
# --------------------------------------------------------------------------- #
def test_refine_changes_body_and_incorporates_critique(tmp_path):
    conv, _ = _conv(tmp_path)
    conv.start("make a tight executive brief")
    before = conv.body

    critique = "ALWAYS-CAP-FINDINGS-AT-FIVE-BULLETS"
    preview = conv.refine(critique)

    after = conv.body
    assert after != before, "refine must RE-AUTHOR the body, not leave it unchanged"
    assert critique in after, "refine must incorporate the critique into the body"
    assert preview.turn == 2
    # The refine is conditioned on the PRIOR body — earlier content is preserved.
    assert "**Mission.**" in after


def test_refine_is_repeatable_and_accumulates(tmp_path):
    conv, _ = _conv(tmp_path)
    conv.start()
    conv.refine("first critique token AAA")
    body_after_one = conv.body
    conv.refine("second critique token BBB")
    body_after_two = conv.body
    assert "AAA" in body_after_one
    assert "AAA" in body_after_two and "BBB" in body_after_two
    assert body_after_two != body_after_one


# --------------------------------------------------------------------------- #
# THE required proof: approve COMPILES + REGISTERS a LOADABLE spec.
# --------------------------------------------------------------------------- #
def test_approve_compiles_and_registers_loadable_spec(tmp_path):
    conv, reg = _conv(tmp_path)
    conv.start("tight brief")
    conv.refine("bound it to five bullets")
    final_body = conv.body

    spec = conv.approve()
    assert spec.name == "concise-brief"
    assert spec.source == SOURCE_UI
    assert spec.body == final_body
    # Registered + loadable as ONE body (the d10 loader surface).
    assert reg.names() == ["concise-brief"]
    loaded = reg.load("concise-brief")
    assert loaded.body == final_body
    # Conversation is now terminal.
    assert conv.state == STATE_APPROVED


# --------------------------------------------------------------------------- #
# reopen: re-open an EXISTING registered spec into an editable session (s4/RC7).
# --------------------------------------------------------------------------- #
def test_reopen_seeds_existing_body_and_edits_in_place(tmp_path):
    """A reopened session begins ALREADY STARTED on the existing body, so the
    next turn is a refine of the REAL ruleset (not a from-scratch author), and
    approve re-registers under the SAME name with provenance preserved."""
    # First author + register a spec the normal way, with a non-'ui' provenance
    # so the preserve-on-reopen behaviour is observable.
    reg = SpecRegistry(tmp_path / "specs")
    seed = SpecConversation(
        RawDefinition(name="brief", description="a brief", intent="x"), registry=reg
    )
    seed.start("shape a tight brief")
    original = seed.approve()  # source 'ui'; capture its identity/trace
    # Stamp a distinct provenance on disk to prove reopen preserves it.
    from dataclasses import replace as _replace
    reg.register(_replace(original, source="autonomous", research_trace_ref="run-XYZ"))
    existing = reg.load("brief")

    # RE-OPEN it for editing.
    conv = SpecConversation.reopen(existing, registry=reg)
    assert conv.started is True            # already authored — no start() needed
    assert conv.body == existing.body      # the working draft IS the persisted body
    with pytest.raises(ConversationError):
        conv.start("should be refused")    # start on an already-started session

    # edit it: a refine folds in a sentinel critique.
    conv.refine("add token QQQ to the rules")
    assert "QQQ" in conv.body and conv.body != existing.body

    # approve re-registers under the SAME name with ORIGINAL provenance preserved.
    re_spec = conv.approve()
    assert re_spec.name == "brief"
    assert re_spec.source == "autonomous"          # NOT silently re-stamped 'ui'
    assert re_spec.research_trace_ref == "run-XYZ"
    assert "QQQ" in reg.load("brief").body         # edit persisted + loadable


def test_history_records_user_and_agent_turns(tmp_path):
    conv, _ = _conv(tmp_path)
    conv.start("opening message")
    conv.refine("a critique")
    roles = [t.role for t in conv.history]
    # user(opening), agent(body), user(critique), agent(body)
    assert roles == ["user", "agent", "user", "agent"]
    assert conv.history[0].text == "opening message"
    assert conv.history[2].text == "a critique"


# --------------------------------------------------------------------------- #
# Lifecycle ordering + terminal-state refusal.
# --------------------------------------------------------------------------- #
def test_refine_before_start_is_refused(tmp_path):
    conv, _ = _conv(tmp_path)
    with pytest.raises(ConversationError):
        conv.refine("too early")


def test_double_start_is_refused(tmp_path):
    conv, _ = _conv(tmp_path)
    conv.start("first")
    with pytest.raises(ConversationError):
        conv.start("again")


def test_empty_critique_is_refused(tmp_path):
    conv, _ = _conv(tmp_path)
    conv.start()
    with pytest.raises(ValueError):
        conv.refine("   ")


def test_approve_before_start_is_refused(tmp_path):
    conv, _ = _conv(tmp_path)
    with pytest.raises(ConversationError):
        conv.approve()


@pytest.mark.parametrize("closer,expected", [("deny", STATE_DENIED), ("cancel", STATE_CANCELLED)])
def test_deny_and_cancel_block_further_authoring(tmp_path, closer, expected):
    conv, reg = _conv(tmp_path)
    conv.start("body")
    getattr(conv, closer)()
    assert conv.state == expected
    # Nothing registered, and no further authoring/approve is possible.
    assert reg.names() == []
    with pytest.raises(ConversationError):
        conv.refine("nope")
    with pytest.raises(ConversationError):
        conv.approve()


# --------------------------------------------------------------------------- #
# The live-transport seam + the never-empty fallback.
# --------------------------------------------------------------------------- #
def test_injected_transport_drives_the_body(tmp_path):
    """An injected transport authors the body — proving the same chain seam the
    compiler uses (transport swap, nothing else)."""
    scripted = "# Output-shaping ruleset: concise-brief\n\n**Mission.** scripted body."
    conv, _ = _conv(tmp_path, transport=FakeTransport([scripted]))
    conv.start("anything")
    assert conv.body == scripted.strip() or conv.body == scripted
    assert "scripted body" in conv.body


def test_empty_transport_reply_falls_back_to_offline(tmp_path):
    """A live transport returning empty/whitespace must NEVER yield an empty body
    — it falls back to the deterministic offline redraft (mirrors the compiler)."""
    conv, _ = _conv(tmp_path, transport=FakeTransport(["   "]))
    conv.start("anything")
    assert conv.body.strip()
    assert OFFLINE_MARKER in conv.body  # the deterministic fallback was used
