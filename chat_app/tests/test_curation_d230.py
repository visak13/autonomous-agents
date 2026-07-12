"""d230 — CLEAN-SLATE registry curation: the planner-facing ADVERTISEMENT is
narrowed to the d206 required-now set, while the raw loaders / spec bodies are
untouched (honest scoping, not hide-to-force — see chat_app.curation).

Proves:
  * curate_index / curate_names narrow a registry's planner-facing lookup to ONLY
    CURATED_SPECS (markdown-writer + arbitrary UI specs drop out) — while the raw
    registry (management listing, missing-spec membership, body load) is untouched.
  * make_get_specs(exposed=…) / make_get_shapes(exposed=…) advertise only the
    curated set; the noise shapes (concurrent-multi-topic-gathering, …) are gone.
  * ShapeSelector(exposed_shapes=…) OFFERS only the curated shapes (catalog +
    spec_names), so the a3 divert option is not even on the menu — yet the raw
    load_shapes catalog still has every shape (the loader is untouched).
  * claude-skill is a seeded canonical spec, so the curated SPEC set is reachable
    out of the box.
"""
from __future__ import annotations

from agent_runtime.discovery_tools import make_get_shapes, make_get_specs
from agent_runtime.shape_selector import ShapeSelector
from agent_runtime.shapes import load_shapes
from llm_framework import FakeTransport
from specialization.registry import SpecRegistry
from specialization.seed import seed_canonical_rulesets

from chat_app.curation import (
    CURATED_SHAPES,
    CURATED_SPECS,
    curate_index,
    curate_names,
)


def _seed_specs(tmp_path):
    """Seed the canonical rulesets + an extra UI spec so curation has noise to drop."""
    reg = SpecRegistry(tmp_path / "specs")
    seed_canonical_rulesets(reg)  # html-writer, markdown-writer, research-analyst, claude-skill
    # an arbitrary user/ui spec NOT in the curated set + a curated UI spec (pirate-tone)
    (reg.specs_dir / "forensic-accountant.md").write_text(
        "---\nname: forensic-accountant\ndescription: detect fraud\nsource: ui\n---\nbody\n",
        encoding="utf-8",
    )
    (reg.specs_dir / "pirate-tone.md").write_text(
        "---\nname: pirate-tone\ndescription: pirate voice\nsource: ui\n---\nArr matey\n",
        encoding="utf-8",
    )
    return reg


def test_curate_helpers_narrow_planner_lookup_only(tmp_path):
    reg = _seed_specs(tmp_path)
    # The RAW registry (management listing / membership) still sees EVERY spec…
    raw = set(reg.names())
    assert {"markdown-writer", "forensic-accountant"} <= raw
    # …but the PLANNER-facing curated views advertise only the curated set.
    assert set(curate_names(reg.names(), CURATED_SPECS)) == set(CURATED_SPECS)
    idx_names = {e.name for e in curate_index(reg.index(), CURATED_SPECS)}
    assert idx_names == set(CURATED_SPECS)
    assert "markdown-writer" not in idx_names
    # A deferred spec's BODY still loads by name (curation is advertisement-only).
    assert reg.load("markdown-writer").name == "markdown-writer"


def test_claude_skill_is_seeded_and_curated(tmp_path):
    reg = _seed_specs(tmp_path)
    assert "claude-skill" in reg.names()
    assert "claude-skill" in set(curate_names(reg.names(), CURATED_SPECS))
    body = reg.load("claude-skill").body.lower()
    assert "frontmatter" in body and "skill" in body


def test_get_specs_tool_advertises_only_curated(tmp_path):
    reg = _seed_specs(tmp_path)
    # The discovery tool reads a fresh registry; the exposed allow-list curates it.
    rows = make_get_specs(
        index_provider=lambda: [
            {"name": n, "description": "", "source": "seed"} for n in
            ("html-writer", "section-html-writer", "markdown-writer",
             "research-analyst", "research-methodology", "web-research",
             "pirate-tone", "claude-skill", "codebase-summary",
             "forensic-accountant")
        ],
        exposed=CURATED_SPECS,
    )()
    got = {r["name"] for r in rows["specs"]}
    assert got == set(CURATED_SPECS)
    assert "markdown-writer" not in got and "forensic-accountant" not in got


def test_get_shapes_tool_advertises_only_curated():
    full = make_get_shapes()()  # no exposure → every shape on disk
    full_names = {r["name"] for r in full["shapes"]}
    assert "concurrent-multi-topic-gathering" in full_names  # the loader is untouched
    curated = make_get_shapes(exposed=CURATED_SHAPES)()
    names = {r["name"] for r in curated["shapes"]}
    assert names == set(CURATED_SHAPES)
    assert "concurrent-multi-topic-gathering" not in names
    assert "modular-parallel" not in names


def test_selector_offers_only_curated_shapes_but_loader_untouched():
    # The raw loader still has every shape (curation is advertisement-only).
    assert "concurrent-multi-topic-gathering" in load_shapes()
    sel = ShapeSelector(
        FakeTransport([]),
        spec_names=["html-writer", "markdown-writer", "research-analyst",
                    "pirate-tone", "claude-skill", "forensic-accountant"],
        spec_catalog=[{"name": "markdown-writer", "description": "md"},
                      {"name": "html-writer", "description": "html"}],
        exposed_shapes=CURATED_SHAPES,
        exposed_specs=CURATED_SPECS,
    )
    offered = set(sel.catalog())
    assert offered == set(CURATED_SHAPES)
    assert "concurrent-multi-topic-gathering" not in offered
    # The advertised spec CATALOG (the default routing surface) is curated…
    assert all(e["name"] in CURATED_SPECS for e in sel.spec_catalog)
    assert "markdown-writer" not in {e["name"] for e in sel.spec_catalog}
    # …but spec_names (the requested_specs enum) stays FULL so a user EXPLICITLY
    # naming a registered-but-deferred spec is still recognised (user-wins).
    assert "markdown-writer" in sel.spec_names
    assert "forensic-accountant" in sel.spec_names
