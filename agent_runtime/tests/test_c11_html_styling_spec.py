"""c11/d52 — html-writer STYLING DEFAULT: reasoning-driven, overridable, composable.

Offline guard for the styling enrichment added to ``HTML_WRITER_RULESET`` (a SPEC
GUIDELINE, NOT a hard-coded CSS template / control flag / app-side injection).
Proves:
  - the html-writer ruleset carries the styling QUALITIES (typography, spacing/
    margins, a STYLED table — borders/zebra/padding, and source-citation styling);
  - it is framed as a reasoning-driven DEFAULT (decide the CSS yourself), not a
    fixed boilerplate stylesheet (honors d46/d14 + d50 — content stays RAW, the
    LLM writes the styled HTML itself);
  - composed with a user TONE spec it does NOT steamroll it: both bodies land, the
    d47-req4 reconcile preamble leads, and the tone body is fully preserved in
    either priority order.

Fully in-process / offline (no Ollama, network or GPU): drives the REAL
``SubAgent`` composition seam — the same one the runtime composes nodes through.
"""
from __future__ import annotations

from agent_runtime.factory import PlanNode
from agent_runtime.identity import AGENT_IDENTITY
from agent_runtime.runtime import (
    _RULESET_LAYER_HEADER,
    _RULESET_RECONCILE_PREAMBLE,
    _SHAPING_FRAMING,
    SubAgent,
)
from agent_runtime.scope import ScopedSpec
from llm_framework import FakeTransport
from specialization.seed import HTML_WRITER_RULESET


# A user-authored TONE spec layered ALONGSIDE html-writer — distinctive telltale
# strings absent from html-writer so the test can prove it is NOT steamrolled.
PIRATE_TONE_RULESET = (
    "You are an OUTPUT-SHAPING ruleset for VOICE. Write every word in the voice of "
    "a swashbuckling pirate — 'Arr', 'matey', nautical metaphors. TELLTALE-PIRATE-VOICE."
)


# --------------------------------------------------------------------------- #
# The styling DEFAULT is present and reasoning-driven (not a fixed template).
# --------------------------------------------------------------------------- #
def test_html_writer_carries_reasoning_driven_styling_default():
    body = HTML_WRITER_RULESET.lower()
    # an inline <style> block is mandated (the LLM authors it itself)...
    assert "<style>" in body
    # ...and the FOUR styling QUALITIES are all called for:
    assert "typograph" in body and "line-height" in body              # typography
    assert "spacing" in body and "margins" in body                    # spacing/margins
    assert "zebra" in body and "border" in body and "padded" in body  # styled table
    assert "citation" in body and "sources" in body                   # citation styling
    # ...expressed as a reasoning-driven DEFAULT, not a fixed/hard-coded template:
    assert "default" in body and "reason" in body
    assert "do not paste a fixed boilerplate" in body or "not a rigid template" in body


def test_styling_is_a_guideline_not_a_hardcoded_css_template():
    # d50/d46: it DESCRIBES qualities — it must NOT embed literal CSS rule blocks or
    # a stylesheet the model would just paste verbatim.
    assert "font-family:" not in HTML_WRITER_RULESET
    assert "{" not in HTML_WRITER_RULESET and "}" not in HTML_WRITER_RULESET  # no CSS rule blocks


def test_styling_default_is_overridable_and_composable_framing():
    body = HTML_WRITER_RULESET.lower()
    assert "reconcile" in body and "compose" in body
    assert "steamroll" in body or "override another" in body


# --------------------------------------------------------------------------- #
# Composes with a user TONE spec WITHOUT steamrolling it (the c11 CRITICAL req).
# --------------------------------------------------------------------------- #
def test_html_styling_composes_with_tone_spec_without_steamrolling():
    # Compose html-writer (Ruleset 1) + a user pirate-tone spec (Ruleset 2) through
    # the REAL SubAgent composition seam, html-writer FIRST (higher priority).
    scopes = [
        ScopedSpec.of("html-writer", HTML_WRITER_RULESET.strip()),
        ScopedSpec.of("pirate-tone", PIRATE_TONE_RULESET.strip()),
    ]
    node = PlanNode(id="n1", task="Write a detailed pirate report as HTML.",
                    specs=("html-writer", "pirate-tone"))
    agent = SubAgent(node, transport=FakeTransport(["x"]), scopes=scopes)
    system = agent._compose_system()

    # identity + ONE shaping framing + the reconcile contract lead the stack.
    assert system.startswith(AGENT_IDENTITY)
    assert system.count(_SHAPING_FRAMING) == 1
    assert _RULESET_RECONCILE_PREAMBLE in system
    # BOTH bodies land — the tone spec is NOT dropped/steamrolled by the styling.
    assert HTML_WRITER_RULESET.strip() in system
    assert PIRATE_TONE_RULESET.strip() in system
    assert "TELLTALE-PIRATE-VOICE" in system
    # layered in the specs-list (priority) ORDER.
    h1 = _RULESET_LAYER_HEADER.format(i=1, n=2, name="html-writer")
    h2 = _RULESET_LAYER_HEADER.format(i=2, n=2, name="pirate-tone")
    assert system.index(h1) < system.index(h2)


def test_user_tone_can_outrank_html_styling_in_reverse_order():
    # The user's own tone spec can be authored as the HIGHER-priority layer and
    # still composes — its voice WINS its axis; html-writer styling is not forced
    # to lead. Proves the styling default is genuinely OVERRIDABLE.
    scopes = [
        ScopedSpec.of("pirate-tone", PIRATE_TONE_RULESET.strip()),
        ScopedSpec.of("html-writer", HTML_WRITER_RULESET.strip()),
    ]
    node = PlanNode(id="n1", task="t.", specs=("pirate-tone", "html-writer"))
    agent = SubAgent(node, transport=FakeTransport(["x"]), scopes=scopes)
    system = agent._compose_system()
    assert _RULESET_RECONCILE_PREAMBLE in system
    assert system.index(PIRATE_TONE_RULESET.strip()) < system.index(HTML_WRITER_RULESET.strip())
    # and html-writer's styling default is still fully present (composed, not lost).
    assert "zebra" in system.lower() and "<style>" in system.lower()
