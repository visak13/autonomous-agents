"""RP-3c (d330) — SELF-POLICING: the verify-lane de-flag + the output-agnostic re-emission
predicate.

The no-hardcoded-flags mandate (d311) forbids flag-gated engine behavior, and d319 says the
engine authors / decides / fixes NOTHING about the model's OUTPUT. RP-3c cleans the last two
flag/format couplings out of the orchestration:

(iii) DE-FLAG the model VERIFY/REVISE self-review lane. The ``verify_lane`` boolean and the two
      engine verify/revise turns it gated are GONE; the model self-review MOVED to the DEFINITION
      LAYER — the writer specs' shared ``_COHERENT_ARTIFACT_DOCTRINE`` now carries a
      SELF-REVIEW-BEFORE-FINISH clause (the MODEL re-reads + grounds-or-drops its own artifact).
      The no-fab research GATHER-MORE gate is KEPT but DE-FLAGGED to an output-agnostic signal
      gate (proven in test_claim_verify).

(iv)  DE-COUPLE the c10 re-emission guard's HTML predicate. The persist/stop/nudge DECISION is
      KEPT (legit orchestration — it never edits the model's bytes); the PREDICATE is now
      OUTPUT-AGNOSTIC (``document_restart`` + the markdown-aware ``section_reemission``), so the
      guard works for ANY output format, not just HTML.

These asserts are the self-policing test: they FAIL if a ``verify_lane`` flag / engine verify turn
is re-introduced, if the self-review leaves the doctrine, if the re-emission predicate re-pins to
HTML, or if the guard starts editing the model's bytes.
"""
from __future__ import annotations

import asyncio
import inspect
import re
from pathlib import Path

import agent_runtime.claim_verify as claim_verify
import agent_runtime.runtime as runtime_mod
from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.runtime import AgentRuntime
from agent_runtime.synth_tools import document_restart, section_reemission
from llm_framework import FakeTransport
from reactive_tools import EventPlane, ToolHook, register_agentic_tools
from specialization.seed import _COHERENT_ARTIFACT_DOCTRINE


def _run(coro):
    return asyncio.run(coro)


def _hook(tmp_path) -> ToolHook:
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    return hook


# =========================================================================== #
# (iii) SELF-POLICING — the verify-lane FLAG is gone; the self-review is in the doctrine
# =========================================================================== #
def test_verify_lane_flag_and_engine_turns_are_gone_from_runtime():
    """No ``verify_lane`` boolean survives anywhere in the runtime: no constructor param, no
    ``self._verify_lane`` / ``self.verify_lane`` assignment, no ``verify_lane=`` pass-through, and
    no ``verify_and_revise`` engine call. (Retirement COMMENTS may still name the flag in
    backticks — we match only actual CODE forms.)"""
    src = Path(runtime_mod.__file__).read_text(encoding="utf-8")
    # a constructor PARAM declaration
    assert re.search(r"^\s*verify_lane\s*:\s*bool", src, re.M) is None
    # an attribute ASSIGNMENT
    assert re.search(r"^\s*self\._?verify_lane\s*=", src, re.M) is None
    # a keyword pass-through to a sub-agent / grower
    assert re.search(r"verify_lane\s*=\s*(self\.|verify_lane|True|False)", src) is None
    # the retired engine ground-or-revise call
    assert "verify_and_revise(" not in src


def test_no_verify_lane_kwarg_on_the_served_agentic_builder():
    """The served chat_app builder no longer threads a ``verify_lane`` kwarg (the flag-free
    grounding end-state, d65)."""
    import chat_app.agentic as agentic
    src = Path(agentic.__file__).read_text(encoding="utf-8")
    assert re.search(r"^\s*verify_lane\s*:\s*bool", src, re.M) is None
    assert re.search(r"verify_lane\s*=\s*(True|False|verify_lane)", src) is None


def test_self_review_before_finish_lives_in_the_writer_doctrine():
    """The model self-review moved to the DEFINITION LAYER: the shared coherent-artifact
    doctrine carries an explicit self-review-before-finish step that grounds-or-drops unbacked
    claims and removes a re-emitted shell — the no-fab guarantee the engine lane used to enforce,
    now authored by the model itself."""
    d = _COHERENT_ARTIFACT_DOCTRINE
    assert "SELF-REVIEW BEFORE YOU FINISH" in d
    low = d.lower()
    # it re-reads the whole artifact and CORRECTS it as part of authoring
    assert "re-read" in low and "correct" in low
    # the no-fab grounding self-review (ground-or-drop an unbacked claim) is present
    assert "ground it" in low and "drop" in low
    # and it composes into the concrete writer rulesets (not just the shared string)
    from specialization.seed import HTML_WRITER_RULESET, MARKDOWN_WRITER_RULESET
    assert "SELF-REVIEW BEFORE YOU FINISH" in HTML_WRITER_RULESET
    assert "SELF-REVIEW BEFORE YOU FINISH" in MARKDOWN_WRITER_RULESET


def test_verify_and_revise_deleted_but_model_driven_verify_claims_kept():
    """The flag-gated engine ground-or-revise checkpoint is DELETED; the MODEL-DRIVEN verify
    surface (the ``cross_verify`` research tool's ``verify_claims`` + the de-flagged gather-more
    signal) is KEPT — self-review is model-driven, never a flag-gated engine turn."""
    assert not hasattr(claim_verify, "verify_and_revise")
    assert not hasattr(claim_verify, "RevisionResult")
    assert hasattr(claim_verify, "verify_claims")
    assert hasattr(claim_verify, "research_answered_from_memory")
    assert "verify_and_revise" not in getattr(claim_verify, "__all__", [])


# =========================================================================== #
# (iv) SELF-POLICING — the re-emission predicate is OUTPUT-AGNOSTIC
# =========================================================================== #
_MD_DOC = (
    "# The 2025 US-Iran Conflict\n\n"
    "A grounded summary of the escalation and its aftermath.\n\n"
    "## Timeline\n\n- Day 1: strike ([UN](https://news.un.org/x)).\n\n"
    "## Sources\n\n- https://news.un.org/x\n"
)
_HTML_DOC = (
    "<!DOCTYPE html><html><head><title>US-Iran</title></head><body>"
    "<h1>The 2025 US-Iran Conflict</h1><h2>Timeline</h2><p>Day 1: strike.</p></body></html>"
)


def test_document_restart_is_output_agnostic():
    """``document_restart`` detects a re-opened artifact in a FORMAT-NEUTRAL way — it fires for a
    Markdown restart AND an HTML restart (both reproduce the file's own opening), and does NOT
    fire for a genuine next section or against an empty file."""
    # Markdown: re-emitting the whole doc from the top IS a restart
    assert document_restart(_MD_DOC, _MD_DOC) is True
    # HTML: same doc re-opened is a restart (no HTML token is hard-coded — it is the head match)
    assert document_restart(_HTML_DOC, _HTML_DOC) is True
    # a GENUINE next Markdown section does NOT reproduce the opening → not a restart
    assert document_restart("## Aftermath\n\nNew grounded analysis of the region.", _MD_DOC) is False
    # nothing to restart against an empty / too-short file
    assert document_restart(_MD_DOC, "") is False
    assert document_restart("# x", _MD_DOC) is False


def test_section_reemission_recognises_markdown_headings():
    """``section_reemission`` is output-agnostic: a chunk repeating a Markdown heading family the
    file already holds is a re-emission; a chunk introducing a NEW Markdown heading is not."""
    assert section_reemission("## Timeline\n\n- Day 1 again.", _MD_DOC) is True
    assert section_reemission("## Casualties\n\n- New figures.", _MD_DOC) is False
    # no heading at all → ordinary continuation prose, never a re-emission
    assert section_reemission("Just some more prose with no heading.", _MD_DOC) is False


def _md_dag(task: str) -> PlanDAG:
    return PlanDAG(nodes=[PlanNode(id="s1", task=task, role="synthesizer")],
                   goal="Write a detailed report on the US-Iran conflict.")


def test_reemission_guard_is_output_agnostic_and_decision_only_on_markdown(tmp_path):
    """The re-emission guard fires on a NON-HTML (Markdown) deliverable: when the model RESTARTS
    the already-complete document, the guard DROPS the re-emission and STOPS. It is DECISION-ONLY
    — the on-disk bytes are EXACTLY the model's first-pass authored document (the guard never
    edited/reshaped them), proving the HTML pin is gone AND the engine still fixes nothing."""
    def reply(messages, **opts):
        n = sum(1 for m in messages if m.get("role") == "assistant")
        if n == 0:
            return _MD_DOC          # a complete markdown document
        if n == 1:
            return _MD_DOC          # RESTART: re-emit the whole document from the top
        return "<<DONE>>"

    transport = FakeTransport([reply])
    rt = AgentRuntime(transport=transport, hook=_hook(tmp_path),
                      subagent_call_opts={"think": True, "temperature": 0})
    out = _run(rt.run(_md_dag("Write a detailed report to us-iran.md.")))

    assert out.ok
    on_disk = (tmp_path / "us-iran.md").read_text(encoding="utf-8")
    # the restart was DROPPED — the document appears exactly ONCE (no concatenated duplicate)
    assert on_disk.count("# The 2025 US-Iran Conflict") == 1
    # DECISION-ONLY: the on-disk bytes are the model's first-pass document AS AUTHORED (the guard
    # dropped the duplicate but never edited/reshaped the content — only the write loop's standard
    # trailing-whitespace .strip() applied, exactly as it does for every format incl. HTML).
    assert on_disk == _MD_DOC.strip()
