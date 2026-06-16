"""NL description -> Gemma-authored declarative SHAPE file (s9, d14(2)).

Proves OFFLINE (a scripted FakeTransport, no GPU) that the shape-authoring
mechanism turns a natural-language description into a SCHEMA-VALID, runtime-loadable
declarative shape file — the shape-side equivalent of the chat-authored
specialization (the LIVE gemma4-e2b-agent proof rides the same code path; only the
transport differs):

1. the authoring call advertises the d1 native structured contract — a JSON SCHEMA
   (enum execution + role enums + EVERY key required), think=False, temp 0;
2. a 'deep-research'-style description -> a bounded research+critic unroll shape
   (round_roles + final_roles + max_iter), which the runtime's generic unroll
   expands into the correct role-tagged DAG;
3. a 'parallel gather + email' description -> a 'concurrent' DISCIPLINE shape
   (no per-node roles — its DAG is authored at plan time by the incremental
   planner);
4. the authored file ROUND-TRIPS the real on-disk loader (load_shape) and is
   selectable + unrollable from disk.
"""
from __future__ import annotations

import asyncio
import json

from llm_framework import FakeTransport

from agent_runtime.factory import VALID_ROLES
from agent_runtime.shape_author import (
    ShapeAuthor,
    build_shape_schema,
    shape_to_toml,
)
from agent_runtime.shapes import VALID_EXECUTION, load_shape, unroll_shape


def _deep_research_reply() -> str:
    return json.dumps(
        {
            "name": "Deep Dive",
            "description": "iterative depth-first research with a critic each round",
            "execution": "deep-research",
            "max_iter": 9,
            "round_roles": ["research", "critic"],
            "final_roles": ["research", "synthesis", "verify"],
        }
    )


def _parallel_reply() -> str:
    return json.dumps(
        {
            "name": "parallel-news",
            "description": "gather several sources at once then email the digest",
            "execution": "concurrent",
            "max_iter": 1,
            "round_roles": [],
            "final_roles": [],
        }
    )


# --------------------------------------------------------------------------- #
# 1) the authoring contract: native JSON schema, enum + required keys
# --------------------------------------------------------------------------- #
def test_schema_is_enum_constrained_and_all_keys_required():
    schema = build_shape_schema()
    props = schema["properties"]
    assert set(props["execution"]["enum"]) == set(VALID_EXECUTION)
    assert set(props["round_roles"]["items"]["enum"]) == set(VALID_ROLES)
    assert set(props["final_roles"]["items"]["enum"]) == set(VALID_ROLES)
    # EVERY field is required (the small model omits OPTIONAL signals).
    assert set(schema["required"]) == {
        "name", "description", "execution", "max_iter", "round_roles", "final_roles",
    }


def test_author_uses_d1_native_structured_options():
    transport = FakeTransport([_parallel_reply()])
    author = ShapeAuthor(transport)
    asyncio.run(author.author("gather news from a few sources and email me"))
    opts = transport.calls[0]["opts"]
    assert opts["api"] == "native"
    assert opts["think"] is False
    assert opts["temperature"] == 0
    assert opts["num_predict"] >= 256
    # the JSON SCHEMA (not format:"json") is what pins the enums + required keys.
    assert opts["format"] == build_shape_schema()


# --------------------------------------------------------------------------- #
# 2) deep-research description -> bounded research+critic unroll shape
# --------------------------------------------------------------------------- #
def test_deep_research_description_authors_unrollable_shape(tmp_path):
    author = ShapeAuthor(FakeTransport([_deep_research_reply()]))
    spec, path = asyncio.run(
        author.author_and_write(
            "research this topic in depth, going deeper each round with a critic",
            shapes_dir=tmp_path,
        )
    )
    assert spec.execution == "deep-research"
    assert spec.is_unrollable
    assert spec.round_roles == ("research", "critic")
    assert spec.final_roles == ("research", "synthesis", "verify")
    assert spec.max_iter == 9 and spec.hard_cap >= 24

    # the runtime's GENERIC unroll expands it into the correct bounded role-tagged DAG
    dag = unroll_shape(load_shape(spec.name, shapes_dir=tmp_path), "g", max_iter_override=3)
    assert [n.role for n in dag.nodes] == [
        "research", "critic", "research", "critic", "research", "synthesis", "verify",
    ]


# --------------------------------------------------------------------------- #
# 3) parallel description -> concurrent DISCIPLINE shape (no per-node roles)
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
    # DAG at plan time; the file must never look unrollable.
    assert not spec.is_unrollable
    assert spec.round_roles == () and spec.final_roles == () and spec.max_iter == 1


# --------------------------------------------------------------------------- #
# 4) the authored file round-trips the REAL on-disk loader + reconciles families
# --------------------------------------------------------------------------- #
def test_authored_file_round_trips_loader_and_is_clean_toml(tmp_path):
    author = ShapeAuthor(FakeTransport([_deep_research_reply()]))
    spec, path = asyncio.run(
        author.author_and_write("deep iterative research", shapes_dir=tmp_path)
    )
    # the file the runtime loads is exactly what load_shape parses.
    text = path.read_text(encoding="utf-8")
    assert "round_roles" in text and "AUTHORED FROM A NATURAL-LANGUAGE DESCRIPTION" in text
    reloaded = load_shape(spec.name, shapes_dir=tmp_path)
    assert reloaded.execution == spec.execution
    assert reloaded.round_roles == spec.round_roles


def test_discipline_shape_omits_roles_in_toml(tmp_path):
    # even if a model leaks roles onto a concurrent shape, the coercion strips them
    leaky = json.dumps(
        {
            "name": "leaky-parallel",
            "description": "parallel gather",
            "execution": "concurrent",
            "max_iter": 5,
            "round_roles": ["research", "critic"],  # must be dropped
            "final_roles": ["synthesis"],
        }
    )
    author = ShapeAuthor(FakeTransport([leaky]))
    spec, path = asyncio.run(author.author_and_write("parallel gather", shapes_dir=tmp_path))
    assert spec.round_roles == () and spec.final_roles == ()
    assert "round_roles" not in path.read_text(encoding="utf-8")
    assert not load_shape(spec.name, shapes_dir=tmp_path).is_unrollable
