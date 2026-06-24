"""Offline proof for the dedicated agentic AUTHORING tool surface (s9/b4, d47/d49).

The b4 acceptance: the SHAPE and SPEC authoring machinery is exposed as a DEDICATED,
named seed -> iterate -> freeze(+compile) tool surface, exactly parallel to the
planner's seed_plan -> add_step (:mod:`agent_runtime.plan_tools`) — reusing the
existing services (NOT a rebuild), NOT schema-constrained, each ``dispatch`` returning
an observation. All offline (scripted ``FakeTransport`` for shapes; the deterministic
``SpecConversation`` offline seam for specs — no live model, no GPU); the live E4B run
rides the SAME code path, only the transport differs.
"""
from __future__ import annotations

import asyncio
import json

from llm_framework import FakeTransport
from specialization import SpecRegistry

from chat_app.authoring_tools import (
    SHAPE_AUTHOR_TOOL_NAMES,
    SPEC_AUTHOR_TOOL_NAMES,
    ShapeAuthoringSession,
    SpecAuthoringSession,
    shape_author_catalog_text,
    spec_author_catalog_text,
)
from chat_app.shape_authoring import ShapeAuthorService
from chat_app.shape_config import ShapeConfigService, ShapeConfigStore
from chat_app.spec_chat import SpecChatService


# --------------------------------------------------------------------------- #
# scripted shape replies (the same JSON the live model authors)
# --------------------------------------------------------------------------- #
def _shape_reply(name: str, execution: str = "concurrent", description: str = "x") -> str:
    return json.dumps(
        {
            "name": name,
            "description": description,
            "execution": execution,
            "max_iter": 1,
            "round_roles": [],
            "final_roles": [],
        }
    )


def _shape_service(tmp_path, transport):
    config = ShapeConfigService(ShapeConfigStore(tmp_path), shapes_dir=tmp_path)
    return ShapeAuthorService(config, transport=transport, shapes_dir=tmp_path)


# --------------------------------------------------------------------------- #
# the tool surface vocabulary is the seed->iterate->freeze workflow
# --------------------------------------------------------------------------- #
def test_tool_surfaces_name_the_workflow():
    assert SHAPE_AUTHOR_TOOL_NAMES == {"seed_shape", "refine_shape", "freeze_shape"}
    assert SPEC_AUTHOR_TOOL_NAMES == {"seed_spec", "refine_spec", "freeze_spec"}
    shape_cat = shape_author_catalog_text()
    spec_cat = spec_author_catalog_text()
    for name in SHAPE_AUTHOR_TOOL_NAMES:
        assert name in shape_cat
    for name in SPEC_AUTHOR_TOOL_NAMES:
        assert name in spec_cat


# --------------------------------------------------------------------------- #
# SHAPE: seed -> iterate -> freeze over the real service (scripted transport)
# --------------------------------------------------------------------------- #
def test_shape_session_seed_refine_freeze(tmp_path):
    # 3 model calls: seed authors, refine re-authors, (freeze is a pure catalog read)
    deep_research_refine = json.dumps(
        {
            "name": "news-fanout",
            "description": "now iterate in deepening rounds with a critic",
            "execution": "deep-research",
            "max_iter": 6,
            "round_roles": ["research", "critic"],
            "final_roles": ["research", "synthesis", "verify"],
        }
    )
    transport = FakeTransport(
        [
            _shape_reply("news-fanout", "concurrent", "gather sources then combine"),
            deep_research_refine,
        ]
    )
    session = ShapeAuthoringSession(_shape_service(tmp_path, transport))
    assert session.available is True

    seed = asyncio.run(session.dispatch("seed_shape", {"description": "fan out then combine"}))
    assert seed["ok"] is True
    assert seed["name"] == "news-fanout"
    assert session.last_shape == "news-fanout"

    refine = asyncio.run(
        session.dispatch("refine_shape", {"instruction": "run iterative rounds instead"})
    )
    assert refine["ok"] is True
    assert refine["execution"] == "deep-research"  # the refine took effect

    freeze = asyncio.run(session.dispatch("freeze_shape", {}))
    assert freeze["ok"] is True and freeze["done"] is True
    assert session.frozen is True
    # audit trail records the stepwise tool calls (parallel to PlanBuilder.calls)
    assert [c["tool"] for c in session.calls] == ["seed_shape", "refine_shape", "freeze_shape"]


def test_shape_session_unknown_tool_and_unavailable(tmp_path):
    # unknown tool → ok=False observation, never raises
    session = ShapeAuthoringSession(_shape_service(tmp_path, FakeTransport([])))
    bad = asyncio.run(session.dispatch("author_everything", {}))
    assert bad["ok"] is False and "unknown tool" in bad["note"]

    # no live transport → unavailable observation (no stub-authored shape)
    offline = ShapeAuthoringSession(_shape_service(tmp_path, None))
    assert offline.available is False
    obs = asyncio.run(offline.dispatch("seed_shape", {"description": "anything"}))
    assert obs["ok"] is False and obs.get("unavailable") is True

    # freeze without a seed → ok=False (nothing selectable yet)
    nope = asyncio.run(session.dispatch("freeze_shape", {"name": "ghost"}))
    assert nope["ok"] is False


# --------------------------------------------------------------------------- #
# SPEC: seed -> iterate -> freeze+compile over the real service (offline seam)
# --------------------------------------------------------------------------- #
def test_spec_session_seed_refine_freeze_registers(tmp_path):
    service = SpecChatService(SpecRegistry(tmp_path / "specs"), transport=None)
    session = SpecAuthoringSession(service)

    seed = asyncio.run(
        session.dispatch(
            "seed_spec",
            {
                "name": "pirate-voice",
                "description": "answer in a swashbuckling pirate voice",
                "intent": "Shape every answer in a hearty pirate tone.",
            },
        )
    )
    assert seed["ok"] is True
    assert seed["name"] == "pirate-voice"
    body1 = seed["draft"]["body"]
    assert body1.strip()

    refine = asyncio.run(
        session.dispatch(
            "refine_spec",
            {"critique": "Always open with 'Arr!' and call the reader 'matey'."},
        )
    )
    assert refine["ok"] is True
    assert refine["turn"] == 2
    assert refine["draft"]["body"] != body1  # iterated on the prior draft

    freeze = asyncio.run(session.dispatch("freeze_spec", {}))
    assert freeze["ok"] is True and freeze["done"] is True
    assert freeze["name"] == "pirate-voice"

    # the frozen spec is REGISTERED + loadable on the next run (the real runtime path)
    assert any(row.name == "pirate-voice" for row in service.list_registered())
    assert service.effective_body("pirate-voice").strip()


def test_spec_session_ordering_guards(tmp_path):
    service = SpecChatService(SpecRegistry(tmp_path / "specs"), transport=None)
    session = SpecAuthoringSession(service)

    # refine / freeze before seed → ok=False, never raises
    assert asyncio.run(session.dispatch("refine_spec", {"critique": "x"}))["ok"] is False
    assert asyncio.run(session.dispatch("freeze_spec", {}))["ok"] is False

    # seed needs name + description + intent
    assert asyncio.run(session.dispatch("seed_spec", {"name": "x"}))["ok"] is False

    # a clean seed, then a second seed is rejected (one session per surface)
    ok = asyncio.run(
        session.dispatch(
            "seed_spec",
            {"name": "terse", "description": "a terse register", "intent": "be terse"},
        )
    )
    assert ok["ok"] is True
    dup = asyncio.run(
        session.dispatch(
            "seed_spec",
            {"name": "terse2", "description": "another", "intent": "again"},
        )
    )
    assert dup["ok"] is False
