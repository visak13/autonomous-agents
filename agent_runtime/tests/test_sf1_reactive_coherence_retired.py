"""SF-1 (d310/d311) SELF-POLICING — the reactive-coherence / output-fixing machinery STAYS
retired. The anti-fabrication charter says the engine authors/decides/fixes NOTHING and the
MODEL authors the whole document; SF-1 removed the FIRST fold site (the deterministic HTML
surgery + the whole-doc anchored-edit reviewer + the robust anchor-matcher + the coherence
metrics). These tests FAIL if any of that machinery is reintroduced.

Scope note (d313 → RP-1/d319/d311): SF-1 retired the FIRST fold site. RP-1 now also retires the
SECOND fold site — the inline engine output-fixing inside ``SubAgent._run_raw_file_loop``
(anchored-insert + enforce_single_html_document/collapse_duplicate_sections) — plus
``reconcile_doc_structure``, the ``strip_ungrounded_urls`` URL guard, the doc-side
``ensure_source_coverage`` net, the ``_flag_unsupported_sections`` / DAG ``_ensure_source_coverage``
/ ``_outline_from_authored_sections`` structure-authoring in ``chat_app.agentic``, the FORMAT
PINS (is_html gate / nav-SPA clause / single-output framing / spec-&-keyword→ext maps), and the
SF-2 engine-compose skeleton-then-fill spec artifacts. The RP-1 self-policing tests below assert
all of that stays gone. RP-2 (d319/d311/d326) additionally RETIRES the MAIN-loop plan-chaining
wrapper hygiene ``strip_wrapper_openers/closers`` (borderline-1: the writer spec now makes the
model author ONE well-formed self-contained artifact, so the engine no longer strips per-page
wrapper tags) AND authors the per-format writer-spec coherence + grounding doctrine (each format
writer carries all 7 points in its own idiom; the engine pins no format) — both asserted below.
(KEPT and NOT asserted gone: the model-driven verify/revise self-review CORE and the predicate-only
re-emission guard; verify-lane de-flagging is deferred to RP-3.)

Fully offline (import + behavioural checks; no GPU, no network)."""
from __future__ import annotations

import importlib
import inspect

import pytest

import agent_runtime.synth_tools as synth_tools
from agent_runtime.runtime import SubAgent


def test_assemble_report_spa_and_its_folds_are_gone_from_synth_tools():
    # The deterministic HTML SURGERY pass + every fold it composed is REMOVED — the engine
    # never assembles/rewrites the deliverable; the model authors it.
    for name in (
        "assemble_report_spa",
        "assemble_html_spa",
        "rebuild_section_nav",
        "rebuild_sources_list",
        "strip_scaffold_comments",
        "collapse_duplicate_section_ids",
        "collapse_outline_duplicate_sections",
        "dedupe_source_lists",
        "relocate_sources_to_end",
        "strip_done_sentinels",
    ):
        assert getattr(synth_tools, name, None) is None, f"{name} must stay retired (SF-1)"


def test_review_injection_module_is_gone():
    # The framework REVIEW INJECTION (work->review pairs + final_review) is retired wholesale.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_runtime.review_injection")


def test_anchored_edit_reviewer_is_gone_from_subagent():
    # No reviewer that EDITS the deliverable in place — the model owns coherence.
    for name in (
        "_run_anchored_review",
        "_review_intro",
        "_dispatch_review_tool",
        "_parse_review_call",
        "_anchored_review_tool_specs",
        "_is_review_node_id",
    ):
        assert not hasattr(SubAgent, name), f"SubAgent.{name} must stay retired (SF-1)"


def test_robust_anchor_matcher_is_gone_and_file_update_is_exact_only(tmp_path):
    # The (A) whitespace-tolerant / prefix...suffix robust matcher is removed; file_update now
    # matches EXACTLY. A whitespace-paraphrased 'old' must RAISE, not silently edit a guessed span.
    import reactive_tools.file_tools as file_tools

    assert getattr(file_tools, "_robust_match_spans", None) is None
    assert getattr(file_tools, "_strip_ws_index", None) is None

    f = tmp_path / "doc.txt"
    f.write_text("alpha    beta gamma", encoding="utf-8")
    update = file_tools.make_file_update(tmp_path)
    # exact snippet applies
    update(path=str(f), old="alpha    beta", new="ALPHA BETA")
    assert f.read_text(encoding="utf-8") == "ALPHA BETA gamma"
    # a whitespace-variant snippet is NOT resolved by any robust fallback — it raises.
    from reactive_tools.tools import ToolInputError

    with pytest.raises(ToolInputError):
        update(path=str(f), old="ALPHA   BETA", new="x")


def test_coherence_metrics_and_review_injection_are_gone_from_agentic():
    agentic = importlib.import_module("chat_app.agentic")
    for name in ("_coherence_metrics", "_pre_surgery_path", "_inject_section_reviews"):
        assert getattr(agentic, name, None) is None, f"chat_app.agentic.{name} must stay retired"


# =========================================================================== #
# RP-1 (d319/d311) SELF-POLICING — the engine authors/fixes NOTHING on the WRITE
# PATH: the 2nd fold site, reconcile, the URL guard, the doc-structure
# fabrications, the format pins and the SF-2 engine-compose spec stay gone.
# =========================================================================== #
def test_rp1_engine_output_fixing_and_ext_maps_gone_from_synth_tools():
    # Every engine output-FIXING / structure-AUTHORING function RP-1 retired, plus the
    # format-INFERENCE ext maps, must be absent from synth_tools.
    for name in (
        "reconcile_doc_structure",
        "enforce_single_html_document",
        "collapse_duplicate_sections",
        "has_duplicate_html_structure",
        "enforce_single_h1",
        "ensure_source_coverage",       # the doc-side sources-block insert
        "strip_ungrounded_urls",        # the no-fab URL guard (engine editing citations)
        "plant_section_anchor",
        "choose_section_anchor",
        "anchored_insert_args",
        "strip_section_anchor",
        "_DocStructureParser",
        "_dedupe_element_ids",
        "_wrap_orphan_list_items",
        "_WRITER_SPEC_EXT",             # spec-name -> ext format pin
        "_FORMAT_KEYWORD_EXT",          # request-keyword -> ext format pin
        "_ext_for",
    ):
        assert getattr(synth_tools, name, None) is None, f"{name} must stay retired (RP-1)"


def test_rp1_second_fold_output_fixing_gone_from_run_raw_file_loop():
    # AUTONOMY REBUILD P2C — the 2nd fold site no longer merely keeps its output-fixing
    # retired: the ENTIRE raw write loop is deleted from SubAgent. Every node (write,
    # synthesizer-terminal, gather, trivial) runs the ONE unified self-select loop and
    # authors its file via file_write; the strongest form of the RP-1 guarantee.
    for gone in ("_run_raw_file_loop", "_run_synthesis", "_run_file_delivery",
                 "_dispatch_writer_tool", "_parse_writer_call",
                 "_tool_calling_writer_tool_specs"):
        assert not hasattr(SubAgent, gone), f"SubAgent.{gone} must stay DELETED (P2C)"


def test_rp1_doc_structure_fabrications_and_format_pins_gone_from_agentic():
    agentic = importlib.import_module("chat_app.agentic")
    # the engine structure-authoring functions are gone as module attributes
    for name in (
        "_flag_unsupported_sections",
        "_ensure_source_coverage",
        "_outline_from_authored_sections",
        "_section_title_from_task",
    ):
        assert getattr(agentic, name, None) is None, f"agentic.{name} must stay retired (RP-1)"
    # _compose_write_goal no longer takes the is_html FORMAT flag (nav-SPA/single-doc pins gone).
    assert "is_html" not in inspect.signature(agentic._compose_write_goal).parameters
    # BEHAVIOURAL: the COMPOSED write goal for an .html deliverable carries NO format pin — no
    # nav-SPA single-page-HTML clause, no single-output-document framing. (Checked on the OUTPUT,
    # not source text, so retirement COMMENTS naming the removed pins never false-trip this.)
    goal = agentic._compose_write_goal("q", "report.html", "findings", "SOURCES: [S1]")
    for pin in ("single-page report (SPA)", "NAVIGABLE single-page",
                "single output document", "Write the COMPLETE document"):
        assert pin not in goal, f"format pin '{pin}' must stay retired from the write goal (RP-1)"
    # the engine doc-surgery calls stay gone from the write-phase orchestration.
    phase_src = inspect.getsource(agentic.run_section_write_phase)
    for banned in ("reconcile_doc_structure(", "enforce_single_html_document(",
                   "collapse_duplicate_sections(", "_flag_unsupported_sections(",
                   "_ensure_source_coverage("):
        assert banned not in phase_src, f"'{banned}' must stay retired (RP-1)"


def test_rp1_sf2_engine_compose_spec_gone_from_seed():
    seed = importlib.import_module("specialization.seed")
    # the structured skeleton-then-fill SCHEMA constant is gone
    assert getattr(seed, "SECTION_HTML_WRITER_SCHEMA", None) is None
    # the section-html-writer ENTRY survives (wiring) but carries NO engine-compose contract:
    # no {skeleton, sections} JSON / {{token}} substitution scaffold.
    ruleset = seed.SECTION_HTML_WRITER_RULESET
    low = ruleset.lower()
    assert "{{" not in ruleset and "skeleton" not in low and "\"sections\"" not in ruleset, \
        "the SF-2 engine-compose skeleton-then-fill contract must stay retired (RP-1)"


# The 7-point coherence + grounding doctrine RP-1 removed from the ENGINE, now owned by the
# MODEL via the writer SPEC (RP-2/d326). Each anchor is a distinctive phrase from ONE point of
# the shared ``_COHERENT_ARTIFACT_DOCTRINE``; asserting all seven are present in a writer body
# proves that writer carries all seven points.
_DOCTRINE_ANCHORS = (
    "author its own table-of-contents",                       # 1 own navigation
    "exactly ONE coherent artifact",                          # 2 one well-formed self-contained doc
    "give each section a UNIQUE identifier",                  # 3 unique ids / resolving links
    "never leave an empty",                                   # 4 never empty / never stub
    "cite ONLY real sources you actually gathered or read",   # 5 ground + cite real only
    "never stop on a mid-sentence fragment",                  # 6 finish sentences
    "the planner decided WHICH sections",                     # 7 planner topology / writer authors
)
# The WRITER / deliverable-format specs (author the artifact) vs the GATHER specs (never do).
_WRITER_SPECS = (
    "markdown-writer", "html-writer", "section-html-writer", "claude-skill", "codebase-summary",
)
_GATHER_SPECS = ("research-analyst", "research-methodology", "web-research")


def test_rp2_writer_specs_carry_coherence_doctrine_output_agnostic():
    # RP-2/d326 OPTION A: EVERY per-format writer spec carries ALL 7 coherence+grounding points
    # in its own idiom; the doctrine is OUTPUT-AGNOSTIC (pins no single format) so HTML is one
    # format among peers; GATHER specs never carry it (they do not author the deliverable).
    seed = importlib.import_module("specialization.seed")
    cr = seed.CANONICAL_RULESETS
    for name in _WRITER_SPECS:
        body = cr[name][1]
        missing = [a for a in _DOCTRINE_ANCHORS if a not in body]
        assert not missing, f"writer spec {name!r} is missing coherence-doctrine points: {missing}"
    for name in _GATHER_SPECS:
        body = cr[name][1]
        present = [a for a in _DOCTRINE_ANCHORS if a in body]
        assert not present, f"gather spec {name!r} must NOT carry the writer doctrine: {present}"
    # OUTPUT-AGNOSTIC: the shared doctrine names >=2 distinct format idioms as PARALLEL examples
    # (web + Markdown + code) — it PINS no single format. HTML-one-among-peers: the SAME doctrine
    # rides the Markdown writer and the HTML writers alike.
    doctrine = seed._COHERENT_ARTIFACT_DOCTRINE
    assert all(idiom in doctrine for idiom in ("web document", "Markdown document", "code")), \
        "the coherence doctrine must be output-agnostic (name multiple format idioms, pin none)"
    for name in ("markdown-writer", "html-writer", "section-html-writer"):
        assert doctrine in cr[name][1], f"{name!r} must carry the identical shared doctrine"


def test_rp2_main_loop_wrapper_hygiene_gone_from_run_raw_file_loop():
    # RP-2/d326 borderline-1 → AUTONOMY REBUILD P2C: the wrapper-hygiene strips are gone
    # in the strongest form — the raw write loop that hosted them is DELETED, and no
    # surviving SubAgent source calls them (the F3 test below separately keeps the
    # helpers themselves deleted from synth_tools).
    assert not hasattr(SubAgent, "_run_raw_file_loop")
    src = inspect.getsource(SubAgent)
    for banned in ("strip_wrapper_openers(", "strip_wrapper_closers("):
        assert banned not in src, f"{banned} must stay retired from SubAgent (RP-2/P2C)"


# =========================================================================== #
# RP-AUDIT F3 (d319/d341/d330) SELF-POLICING — the DEAD HTML-format-pinned output
# MODIFIERS / PREDICATES stay DELETED. RP-2 above only bans their CALL from the
# write loop; F3 additionally bans re-DEFINING or re-EXPORTING them from
# synth_tools, closing the "one import away from being re-wired" gap.
# =========================================================================== #
_F3_HTML_PINNED_OUTPUT_HELPERS = (
    "strip_wrapper_closers",     # output-MODIFIER: strips document-wrapper CLOSE tags
    "strip_wrapper_openers",     # output-MODIFIER: strips document-wrapper OPEN tags
    "dedupe_html_documents",     # output-MODIFIER: rewrites to the first complete doc
    "top_level_html_doc_count",  # HTML-pinned read-only PREDICATE (counts </html>)
    "begins_html_document",      # HTML-pinned read-only PREDICATE (<!DOCTYPE>/<html> start)
)


def test_f3_html_pinned_output_modifiers_gone_from_synth_tools():
    # Neither DEFINED as a module attribute NOR EXPORTED in __all__ — so re-introducing
    # any of these (defining OR exporting) FAILS here, not only calling one. The engine
    # authors/fixes/modifies NOTHING; the format-baked output-modifiers/predicates are
    # gone for good.
    exported = set(getattr(synth_tools, "__all__", ()))
    for name in _F3_HTML_PINNED_OUTPUT_HELPERS:
        assert getattr(synth_tools, name, None) is None, \
            f"{name} must stay DELETED from synth_tools (RP-AUDIT F3 dead HTML output helper)"
        assert name not in exported, \
            f"{name} must not be re-EXPORTED in synth_tools.__all__ (RP-AUDIT F3)"
    # KEPT (NOT banned): the FORMAT-NEUTRAL re-emission guard that replaced the deleted
    # HTML-pinned predicates (RP-3c/d330). Assert it survives so the deletion is scoped.
    for kept in ("document_restart", "section_reemission", "html_close_gap"):
        assert callable(getattr(synth_tools, kept, None)), \
            f"the format-neutral guard {kept} must stay (RP-AUDIT F3 keeps it)"
