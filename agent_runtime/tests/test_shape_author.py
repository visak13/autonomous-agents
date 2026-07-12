"""NL description -> Gemma-authored declarative SHAPE file (s9, d14(2)).

Proves OFFLINE (a scripted FakeTransport, no GPU) that the shape-authoring
mechanism turns a natural-language description into a SCHEMA-VALID, runtime-loadable
declarative shape file — the shape-side equivalent of the chat-authored
specialization (the LIVE gemma4-e2b-agent proof rides the same code path; only the
transport differs):

1. the authoring call uses the b6/d34 PROMPT-JSON reasoning contract — think=True,
   temp 0, raised num_predict, and NO constrained `format` schema (the schema is
   the documented contract, not a wire constraint, so a reasoned posture is never
   schema-dropped);
2. a 'deep-research'-style description -> a deep-research FAMILY shape (execution
   token + max_iter depth ceiling + the growable marker); its research topology is
   AUTHORED at runtime by the grower, NOT a declared round/final node graph
   (s16/a3 d239/d247 — round_roles/final_roles are RETIRED);
3. a 'parallel gather + email' description -> a 'concurrent' DISCIPLINE shape
   (no per-node topology — its DAG is authored at plan time by the incremental
   planner);
4. a COMPOSITIONAL 'linear plus modular parallel' description does NOT collapse to
   a flat 'sequential' shape (b6/d18a — upgraded to 'concurrent');
5. the free-flow ITERATIVE REFINE builds on an existing shape (edit in place,
   keeps the name, overwrites the file);
6. the authored/refined file ROUND-TRIPS the real on-disk loader (load_shape) and
   is selectable from disk.
"""
from __future__ import annotations

import asyncio
import json

from llm_framework import FakeTransport

from agent_runtime.shape_author import (
    ShapeAuthor,
    build_shape_schema,
    write_shape,
)
from agent_runtime.shapes import (
    ShapeSpec,
    VALID_EXECUTION,
    load_shape,
)


def _deep_research_reply() -> str:
    # s16/a3: a deep-research shape is fully described by {name, description, execution,
    # max_iter} — it declares NO per-round node positions (round_roles/final_roles retired).
    return json.dumps(
        {
            "name": "Deep Dive",
            "description": "iterative depth-first research with a critic each round",
            "execution": "deep-research",
            "max_iter": 9,
        }
    )


def _parallel_reply() -> str:
    return json.dumps(
        {
            "name": "parallel-news",
            "description": "gather several sources at once then email the digest",
            "execution": "concurrent",
            "max_iter": 1,
        }
    )


# --------------------------------------------------------------------------- #
# 1) the authoring contract: native JSON schema, enum + required keys
# --------------------------------------------------------------------------- #
def test_schema_is_enum_constrained_and_all_keys_required():
    schema = build_shape_schema()
    props = schema["properties"]
    assert set(props["execution"]["enum"]) == set(VALID_EXECUTION)
    # s16/a3 (d239/d247): round_roles/final_roles are RETIRED from the authoring schema —
    # a shape declares NO per-node topology. EVERY remaining field is required.
    assert "round_roles" not in props and "final_roles" not in props
    assert set(schema["required"]) == {"name", "description", "execution", "max_iter"}


def test_author_uses_prompt_json_reasoning_options_no_format_schema():
    transport = FakeTransport([_parallel_reply()])
    author = ShapeAuthor(transport)
    asyncio.run(author.author("gather news from a few sources and email me"))
    opts = transport.calls[0]["opts"]
    assert opts["api"] == "native"
    # s1/b1 reasoning rollout: think ON (gemma4 reasons in the SEPARATE
    # message.thinking field); num_predict raised to give the CoT headroom.
    assert opts["think"] is True
    assert opts["temperature"] == 0
    assert opts["num_predict"] >= 4096
    # b6/d34: the reasoned posture must NOT sit behind a constrained `format` schema
    # (constrained decoding silently drops the reasoned field — the collapse this
    # action fixes). The JSON is prompt-elicited + interceptor/repair-validated.
    assert "format" not in opts
    # the schema is still the documented contract / proof artifact, just not a wire
    # constraint.
    assert author.last_schema == build_shape_schema()


# --------------------------------------------------------------------------- #
# 2) deep-research description -> deep-research FAMILY shape (topology reasoned at runtime)
# --------------------------------------------------------------------------- #
def test_deep_research_description_authors_deep_research_shape(tmp_path):
    author = ShapeAuthor(FakeTransport([_deep_research_reply()]))
    spec, path = asyncio.run(
        author.author_and_write(
            "research this topic in depth, going deeper each round with a critic",
            shapes_dir=tmp_path,
        )
    )
    assert spec.execution == "deep-research"
    assert spec.is_deep_research
    # s16/a3: NO declared round/final positions — the dataclass dropped those fields; the
    # research topology is authored at runtime by the grower from a tool-less growable seed.
    assert not hasattr(spec, "round_roles")
    assert not hasattr(spec, "final_roles")
    # the deep-research shape is GROWABLE (the engine builds a growable seed) + carries the
    # max_iter depth ceiling and the safety hard_cap.
    assert spec.expand_on_gaps
    assert spec.max_iter == 9 and spec.hard_cap >= 24


# --------------------------------------------------------------------------- #
# 3) parallel description -> concurrent DISCIPLINE shape (no per-node topology)
# --------------------------------------------------------------------------- #
def test_parallel_description_authors_concurrent_discipline_shape(tmp_path):
    author = ShapeAuthor(FakeTransport([_parallel_reply()]))
    spec, _ = asyncio.run(
        author.author_and_write(
            "gather news from three sources in parallel, then email me a digest",
            shapes_dir=tmp_path,
        )
    )
    assert spec.execution == "concurrent"
    # a discipline shape carries NO topology — the incremental planner authors its
    # DAG at plan time; it is NOT the deep-research family.
    assert not spec.is_deep_research
    assert spec.max_iter == 1


# --------------------------------------------------------------------------- #
# 4) the authored file round-trips the REAL on-disk loader + reconciles families
# --------------------------------------------------------------------------- #
def test_authored_file_round_trips_loader_and_is_clean_toml(tmp_path):
    author = ShapeAuthor(FakeTransport([_deep_research_reply()]))
    spec, path = asyncio.run(
        author.author_and_write("deep iterative research", shapes_dir=tmp_path)
    )
    # the file the runtime loads is exactly what load_shape parses. s16/a3: a deep-research
    # file carries the growable marker (expand_on_gaps), NOT round_roles/final_roles.
    text = path.read_text(encoding="utf-8")
    assert "expand_on_gaps" in text and "AUTHORED FROM A NATURAL-LANGUAGE DESCRIPTION" in text
    assert "round_roles" not in text and "final_roles" not in text
    reloaded = load_shape(spec.name, shapes_dir=tmp_path)
    assert reloaded.execution == spec.execution
    assert reloaded.is_deep_research and reloaded.expand_on_gaps


def test_discipline_shape_omits_topology_in_toml(tmp_path):
    # even if a model leaks round/final keys onto a concurrent shape, the coercion IGNORES
    # them (the retired fields are not read) and the written file carries no topology.
    leaky = json.dumps(
        {
            "name": "leaky-parallel",
            "description": "parallel gather",
            "execution": "concurrent",
            "max_iter": 5,
            "round_roles": ["research", "critic"],  # ignored — not read by coercion
            "final_roles": ["synthesis"],
        }
    )
    author = ShapeAuthor(FakeTransport([leaky]))
    spec, path = asyncio.run(author.author_and_write("parallel gather", shapes_dir=tmp_path))
    assert not spec.is_deep_research and spec.max_iter == 1
    text = path.read_text(encoding="utf-8")
    assert "round_roles" not in text and "final_roles" not in text
    assert not load_shape(spec.name, shapes_dir=tmp_path).is_deep_research


# --------------------------------------------------------------------------- #
# 5) COMPOSITIONAL intent (b6/d18a) — a "linear plus modular parallel" description
#    must NOT collapse to a flat 'sequential' shape. The deterministic safety-net
#    upgrades a model-emitted 'sequential' to 'concurrent' (the posture that
#    supports BOTH a chained sequential phase AND a parallel fan-out).
# --------------------------------------------------------------------------- #
def _collapsed_compositional_reply() -> str:
    """A model reply that COLLAPSED a compositional request to flat 'sequential'."""
    return json.dumps(
        {
            "name": "linear-plus-modular-parallel",
            "description": (
                "combines a strictly sequential foundation phase with a modular "
                "parallel phase exploring independent avenues, then combines them"
            ),
            "execution": "sequential",  # the collapse this action must prevent
            "max_iter": 1,
        }
    )


def test_compositional_description_does_not_collapse_to_sequential(tmp_path):
    author = ShapeAuthor(FakeTransport([_collapsed_compositional_reply()]))
    spec, _ = asyncio.run(
        author.author_and_write(
            "a linear foundation phase plus a modular parallel phase",
            shapes_dir=tmp_path,
        )
    )
    # The model emitted 'sequential' but the safety-net upgraded it to 'concurrent'
    # so the parallel phase is not flattened. Still a discipline shape (not deep-research).
    assert spec.execution == "concurrent"
    assert not spec.is_deep_research
    # round-trips the real loader as concurrent.
    assert load_shape(spec.name, shapes_dir=tmp_path).execution == "concurrent"


def test_pure_sequential_request_stays_sequential(tmp_path):
    # A genuinely linear description (no parallel cue) must NOT be upgraded.
    seq = json.dumps(
        {
            "name": "strict-chain",
            "description": "run the steps strictly one after another, never overlapping",
            "execution": "sequential",
            "max_iter": 1,
        }
    )
    author = ShapeAuthor(FakeTransport([seq]))
    spec, _ = asyncio.run(
        author.author_and_write("do the steps strictly in order", shapes_dir=tmp_path)
    )
    assert spec.execution == "sequential"


# --------------------------------------------------------------------------- #
# 6) free-flow ITERATIVE authoring (b6/d18a) — REFINE builds on an existing shape
# --------------------------------------------------------------------------- #
def _prior_concurrent_shape() -> ShapeSpec:
    return ShapeSpec(
        name="news-digest",
        description="gather several sources at once then combine into a digest",
        execution="concurrent",
        max_iter=1,
        source="<test>",
    )


def test_refine_builds_on_prior_and_keeps_name():
    # the refine USER turn must carry the PRIOR shape so the model edits, not restarts.
    refined_reply = json.dumps(
        {
            "name": "renamed-by-model",  # the model tried to rename — must be ignored
            "description": "gather several sources at once then email the digest",
            "execution": "concurrent",
            "max_iter": 1,
        }
    )
    transport = FakeTransport([refined_reply])
    author = ShapeAuthor(transport)
    prior = _prior_concurrent_shape()
    spec = asyncio.run(author.refine(prior, "email the digest instead of just combining"))
    # an edit edits IN PLACE — the prior name is preserved even though the model renamed.
    assert spec.name == "news-digest"
    # the prior definition was fed into the prompt (build-on-existing, not one-shot).
    user_turn = transport.calls[0]["messages"][-1]["content"]
    assert "CURRENT SHAPE" in user_turn and "news-digest" in user_turn
    # prompt-JSON path: no constrained format schema on a refine call either.
    assert "format" not in transport.calls[0]["opts"]


def test_refine_compositional_upgrade_sequential_to_concurrent(tmp_path):
    # start from a strict sequential shape on disk, then refine it to add a parallel
    # phase — the refined shape must become 'concurrent', overwriting the same file.
    prior = ShapeSpec(
        name="two-phase",
        description="run a foundation step then a follow-up step, one after another",
        execution="sequential",
        max_iter=1,
        source="<test>",
    )
    write_shape(prior, shapes_dir=tmp_path)
    # the model still emits 'sequential' (collapse) — the safety-net (seeing the
    # instruction's parallel intent) upgrades it.
    reply = json.dumps(
        {
            "name": "two-phase",
            "description": (
                "a sequential foundation phase, then a modular parallel phase of "
                "independent steps, then a combine step"
            ),
            "execution": "sequential",
            "max_iter": 1,
        }
    )
    author = ShapeAuthor(FakeTransport([reply]), shapes_dir=tmp_path)
    spec, path = asyncio.run(
        author.refine_and_write(
            "two-phase",
            "run the second phase as independent steps in parallel, then combine",
            shapes_dir=tmp_path,
        )
    )
    assert spec.name == "two-phase"
    assert spec.execution == "concurrent"
    # overwrote the SAME file (edit in place) and round-trips the real loader.
    assert path.name == "two-phase.toml"
    assert load_shape("two-phase", shapes_dir=tmp_path).execution == "concurrent"


def test_refine_empty_instruction_rejected():
    import pytest

    author = ShapeAuthor(FakeTransport([_parallel_reply()]))
    with pytest.raises(Exception):
        asyncio.run(author.refine(_prior_concurrent_shape(), "   "))
