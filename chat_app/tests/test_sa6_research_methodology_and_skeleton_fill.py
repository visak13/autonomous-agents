"""s16/SA-6 PART 1 — the generic research-methodology CORE spec + web-research variant (d258).

(PART 2 — the section-html-writer SKELETON-THEN-FILL re-author — is DEFINITION-LAYER ONLY per
d278: the methodology lives in the SECTION_HTML_WRITER_RULESET spec TEXT and the PLANNER authors
the topology by reasoning from the spec; there is NO code stamping / spec-name conditional on the
write path to unit-test here. The spec-text + coherent-by-construction behaviour is proven by the
live human-read runs, not a code probe.)

PART 1 proves (definition-layer, the spec text is the lever — d240):
  * a DOMAIN-AGNOSTIC research-methodology CORE spec exists and NEVER names 'web';
  * a NAMED web-research VARIANT = the CORE method + a thin web-pairing note (siblings differ
    ONLY by the paired gather bundle, d258);
  * both are SEEDED + advertised in CURATED_SPECS, with selection-lever descriptions that steer
    the planner to web-research for a web brief and to the CORE for a non-web brief.
"""
from __future__ import annotations

from specialization.registry import SpecRegistry
from specialization.seed import (
    CANONICAL_RULESETS,
    RESEARCH_METHODOLOGY_RULESET,
    WEB_RESEARCH_RULESET,
    seed_canonical_rulesets,
)

from chat_app.curation import CURATED_SPECS


def test_core_methodology_is_domain_agnostic_never_names_web():
    body = RESEARCH_METHODOLOGY_RULESET.lower()
    assert "web" not in body, "the CORE methodology must be DOMAIN-AGNOSTIC (never names web)"
    for token in ("decompose", "note", "gap", "cross-verify", "expand",
                  "prune", "complete", "self-select"):
        assert token in body, f"CORE methodology missing the {token!r} doctrine"


def test_web_research_variant_is_core_plus_web_pairing():
    # siblings differ ONLY by the paired gather bundle (d258): the variant IS the CORE + a thin
    # web-pairing note, so codebase-research / vectordb-research are later siblings of the SAME
    # method.
    assert WEB_RESEARCH_RULESET.startswith(RESEARCH_METHODOLOGY_RULESET)
    pairing = WEB_RESEARCH_RULESET[len(RESEARCH_METHODOLOGY_RULESET):].lower()
    assert "web" in pairing and "web_search" in pairing and "web_fetch" in pairing


def test_both_specs_seeded_and_curated():
    assert "research-methodology" in CANONICAL_RULESETS
    assert "web-research" in CANONICAL_RULESETS
    assert "research-methodology" in CURATED_SPECS
    assert "web-research" in CURATED_SPECS


def test_specs_seed_into_a_registry(tmp_path):
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)
    assert "research-methodology" in reg.names()
    assert "web-research" in reg.names()
    # the CORE body loads back web-free; the variant body carries the web pairing.
    assert "web" not in reg.load("research-methodology").body.lower()
    assert "web_fetch" in reg.load("web-research").body.lower()


def test_selection_lever_descriptions_steer_web_vs_nonweb():
    """The DESCRIPTION is the planner's selection lever (the contrastive test's static half;
    the live planner-picks-X is the neuron/human read-verify gate)."""
    core_desc = CANONICAL_RULESETS["research-methodology"][0].lower()
    web_desc = CANONICAL_RULESETS["web-research"][0].lower()
    assert "web" in web_desc and "live" in web_desc
    assert "non-web" in core_desc and "domain-agnostic" in core_desc
    assert "gather node" in core_desc and "gather node" in web_desc
