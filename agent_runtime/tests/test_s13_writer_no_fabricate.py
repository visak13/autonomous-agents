"""s13 / FX-writer (d106 #6, #7) — OUTLINE-AS-PRIMARY backstop + EMPTY-NODE-NO-FABRICATE.

The B8a live run surfaced two writer-side quality defects:

  #7  The agent outline landed as a SECOND, parallel section set appended after the
      conclusion (three conflicting "Section 3"s) instead of being the PRIMARY scaffold.
  #6  A research node that fetched 0 sources (B1) still got a section, which the writer
      fabricated from memory (the Timeline of invented dated events).

This file covers the ``UNSUPPORTED_SECTION_INSTRUCTION`` text the #6 flag stamps onto a
no-source section (d48/d60-clean — it fabricates NOTHING). The #7 outline-duplicate collapse
(``collapse_outline_duplicate_sections``) was RETIRED in SF-1/d310/d311 with the rest of the
deterministic HTML assembly surgery — the model now authors the whole document.
"""
from __future__ import annotations

from agent_runtime.synth_tools import UNSUPPORTED_SECTION_INSTRUCTION


def test_unsupported_instruction_is_anti_fabrication():
    """The #6 flag text must mark UNSUPPORTED and explicitly forbid fabrication."""
    text = UNSUPPORTED_SECTION_INSTRUCTION
    assert "UNSUPPORTED" in text
    lowered = text.lower()
    assert "no supporting sources" in lowered
    assert "do not invent" in lowered
    # names the kinds of content that must NOT be fabricated (the B8a Timeline case)
    for token in ("dates", "figures", "timelines", "citations"):
        assert token in lowered
