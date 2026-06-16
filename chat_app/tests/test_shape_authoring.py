"""Backend proof for the s9/b1 describe-a-shape authoring surface (d14(2)/d9).

The b1 acceptance: *a user DESCRIBES a shape and the live Gemma model authors a
declarative shape file saved to the runtime's shapes dir, which then appears in the
Shapes list and is loadable/runnable by the runtime.* These tests prove that chain
end-to-end with NO live model (a scripted ``FakeTransport``; the live gemma4-e2b-agent
proof rides the SAME code path — only the transport differs):

1. SERVICE — :class:`ShapeAuthorService` authors a shape from a description, writes
   it into the shapes dir the catalog reads, and returns the SAME merged view the
   list renders (so the authored shape appears in the list at once).
2. UNAVAILABLE — with no live transport the service refuses (no stub-authored file).
3. NO CLOBBER — an authored name that collides with an existing shape is REJECTED,
   never silently overwritten (universal [required]).
4. AUTHORING FAILURE — an unauthorable description surfaces as a MalformedOutputError.
5. THE HTTP SURFACE — ``POST /shapes/author`` over a real app: 201 + the view on
   success (and ``GET /shapes`` then lists it), 503 with no live model, 409 on a
   name collision, 422 on an empty description / unauthorable reply.
"""
from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
from llm_framework import FakeTransport

from agent_runtime.selfheal import MalformedOutputError
from agent_runtime.shapes import load_shape

from chat_app.shape_authoring import (
    ShapeAuthorService,
    ShapeAuthoringUnavailable,
    ShapeNameConflict,
    register_shape_author_routes,
)
from chat_app.shape_config import (
    ShapeConfigService,
    ShapeConfigStore,
    register_shape_routes,
)


# --------------------------------------------------------------------------- #
# scripted model replies (the same shapes the live model authors)
# --------------------------------------------------------------------------- #
def _deep_research_reply(name: str = "deep-dive") -> str:
    return json.dumps(
        {
            "name": name,
            "description": "iterative depth-first research with a critic each round",
            "execution": "deep-research",
            "max_iter": 9,
            "round_roles": ["research", "critic"],
            "final_roles": ["research", "synthesis", "verify"],
        }
    )


def _parallel_reply(name: str = "parallel-news") -> str:
    return json.dumps(
        {
            "name": name,
            "description": "gather several sources at once then email the digest",
            "execution": "concurrent",
            "max_iter": 1,
            "round_roles": [],
            "final_roles": [],
        }
    )


def _bad_execution_reply() -> str:
    # valid JSON, but an execution the coercion rejects -> MalformedOutputError
    return json.dumps(
        {
            "name": "nope",
            "description": "x",
            "execution": "teleport",
            "max_iter": 1,
            "round_roles": [],
            "final_roles": [],
        }
    )


def _service(tmp_path, transport):
    """A ShapeAuthorService + its catalog service over an isolated tmp shapes dir.

    Both point at the SAME ``tmp_path`` so an authored file is visible to the catalog
    the list reads (the production invariant, isolated for the test)."""
    config = ShapeConfigService(ShapeConfigStore(tmp_path), shapes_dir=tmp_path)
    author = ShapeAuthorService(config, transport=transport, shapes_dir=tmp_path)
    return author, config


# --------------------------------------------------------------------------- #
# 1) the service authors -> writes -> returns the list view
# --------------------------------------------------------------------------- #
def test_service_authors_writes_and_returns_view(tmp_path):
    import asyncio

    author, config = _service(tmp_path, FakeTransport([_deep_research_reply()]))
    view = asyncio.run(author.author("research this topic in depth each round"))

    # the returned view IS the merged catalog view the list renders
    assert view["name"] == "deep-dive"
    assert view["execution"] == "deep-research"
    assert view["round_roles"] == ["research", "critic"]
    assert view["final_roles"] == ["research", "synthesis", "verify"]
    assert view["effective_max_iter"] == 9

    # the authored shape is on disk in the dir the runtime loads (round-trips loader)
    reloaded = load_shape("deep-dive", shapes_dir=tmp_path)
    assert reloaded.is_unrollable
    # and it appears in the catalog the Shapes list reads
    assert any(s["name"] == "deep-dive" for s in config.list_shapes())


# --------------------------------------------------------------------------- #
# 2) authoring needs the live model (no stub-authored file)
# --------------------------------------------------------------------------- #
def test_service_unavailable_without_transport(tmp_path):
    import asyncio

    author, _ = _service(tmp_path, None)
    assert author.available is False
    try:
        asyncio.run(author.author("anything"))
    except ShapeAuthoringUnavailable:
        return
    raise AssertionError("expected ShapeAuthoringUnavailable with no transport")


# --------------------------------------------------------------------------- #
# 3) a name collision is rejected, never silently overwritten
# --------------------------------------------------------------------------- #
def test_service_rejects_name_collision(tmp_path):
    import asyncio

    # two replies with the SAME authored name; the second must conflict, not clobber
    author, _ = _service(
        tmp_path, FakeTransport([_parallel_reply("dup"), _parallel_reply("dup")])
    )
    asyncio.run(author.author("gather news in parallel and email me"))
    try:
        asyncio.run(author.author("gather news in parallel and email me again"))
    except ShapeNameConflict as exc:
        assert exc.name == "dup"
        return
    raise AssertionError("expected ShapeNameConflict on a duplicate authored name")


# --------------------------------------------------------------------------- #
# 4) an unauthorable description surfaces as MalformedOutputError
# --------------------------------------------------------------------------- #
def test_service_surfaces_authoring_failure(tmp_path):
    import asyncio

    author, _ = _service(tmp_path, FakeTransport([_bad_execution_reply()]))
    try:
        asyncio.run(author.author("something the model can't shape"))
    except MalformedOutputError:
        return
    raise AssertionError("expected MalformedOutputError on an unauthorable reply")


# --------------------------------------------------------------------------- #
# 5) THE HTTP SURFACE — POST /shapes/author over a real app
# --------------------------------------------------------------------------- #
def _app(tmp_path, transport) -> FastAPI:
    app = FastAPI()
    config = ShapeConfigService(ShapeConfigStore(tmp_path), shapes_dir=tmp_path)
    register_shape_routes(app, config)
    register_shape_author_routes(
        app, ShapeAuthorService(config, transport=transport, shapes_dir=tmp_path)
    )
    return app


def test_post_author_creates_shape_and_lists_it(tmp_path):
    app = _app(tmp_path, FakeTransport([_deep_research_reply("ui-research")]))
    with TestClient(app) as client:
        resp = client.post(
            "/shapes/author",
            json={"description": "iteratively research deeply with a critic each round"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["name"] == "ui-research"
        assert body["execution"] == "deep-research"

        # the authored shape now appears in the catalog the Shapes screen lists
        listing = client.get("/shapes")
        assert listing.status_code == 200
        assert "ui-research" in {s["name"] for s in listing.json()["shapes"]}


def test_post_author_503_without_live_model(tmp_path):
    app = _app(tmp_path, None)
    with TestClient(app) as client:
        resp = client.post("/shapes/author", json={"description": "anything"})
        assert resp.status_code == 503


def test_post_author_409_on_name_collision(tmp_path):
    app = _app(
        tmp_path, FakeTransport([_parallel_reply("dup"), _parallel_reply("dup")])
    )
    with TestClient(app) as client:
        assert client.post(
            "/shapes/author", json={"description": "gather in parallel + email"}
        ).status_code == 201
        assert client.post(
            "/shapes/author", json={"description": "gather in parallel + email again"}
        ).status_code == 409


def test_post_author_422_on_empty_and_unauthorable(tmp_path):
    # empty description rejected at the wire by Pydantic (min_length=1)
    app_empty = _app(tmp_path / "a", FakeTransport([_deep_research_reply()]))
    with TestClient(app_empty) as client:
        assert client.post("/shapes/author", json={"description": ""}).status_code == 422

    # a reply the model can't shape -> 422 (MalformedOutputError mapped)
    app_bad = _app(tmp_path / "b", FakeTransport([_bad_execution_reply()]))
    with TestClient(app_bad) as client:
        assert client.post(
            "/shapes/author", json={"description": "unauthorable"}
        ).status_code == 422
