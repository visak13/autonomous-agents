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
    ShapeNotFound,
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
        }
    )


def _parallel_reply(name: str = "parallel-news") -> str:
    return json.dumps(
        {
            "name": name,
            "description": "gather several sources at once then email the digest",
            "execution": "concurrent",
            "max_iter": 1,
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
    # s17 (d248/d249): the transitional round_roles/final_roles empty-[] shim is REMOVED —
    # the view carries no fixed round topology; the deep-research identity is the execution token.
    assert "round_roles" not in view
    assert "final_roles" not in view
    assert view["effective_max_iter"] == 9

    # the authored shape is on disk in the dir the runtime loads (round-trips loader)
    reloaded = load_shape("deep-dive", shapes_dir=tmp_path)
    assert reloaded.is_deep_research
    # authored deep-research shapes are growable (the engine builds a tool-less growable seed)
    assert reloaded.expand_on_gaps
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


# --------------------------------------------------------------------------- #
# 6) REFINE — free-flow ITERATIVE edit of an EXISTING shape (s8/b6, d18a)
# --------------------------------------------------------------------------- #
def _collapsed_compositional_reply(name: str = "linear-plus-modular-parallel") -> str:
    # the model COLLAPSED a compositional refine to flat 'sequential'; the
    # safety-net must upgrade it to 'concurrent' so the parallel phase survives.
    return json.dumps(
        {
            "name": name,
            "description": (
                "a sequential foundation phase, then a modular parallel phase of "
                "independent avenues, then a combine step"
            ),
            "execution": "sequential",
            "max_iter": 1,
        }
    )


def test_service_refine_builds_on_prior_overwrites_in_place(tmp_path):
    import asyncio

    # seed an existing shape via the author flow, then refine it in place.
    author, config = _service(
        tmp_path,
        FakeTransport([_parallel_reply("news"), _collapsed_compositional_reply("news")]),
    )
    asyncio.run(author.author("gather news in parallel and email me"))
    assert load_shape("news", shapes_dir=tmp_path).execution == "concurrent"

    view = asyncio.run(
        author.refine("news", "add a sequential foundation phase before the parallel one")
    )
    # same shape (edit in place — no rename, no new file), posture stays/becomes
    # concurrent (compositional, never flattened to sequential).
    assert view["name"] == "news"
    assert view["execution"] == "concurrent"
    # exactly ONE shape file on disk (overwritten, not duplicated).
    assert len(list(tmp_path.glob("*.toml"))) == 1
    assert sum(1 for s in config.list_shapes() if s["name"] == "news") == 1


def test_service_refine_404_when_missing(tmp_path):
    import asyncio

    author, _ = _service(tmp_path, FakeTransport([_parallel_reply("whatever")]))
    try:
        asyncio.run(author.refine("does-not-exist", "tweak it"))
    except ShapeNotFound as exc:
        assert exc.name == "does-not-exist"
        return
    raise AssertionError("expected ShapeNotFound refining a missing shape")


def test_service_refine_unavailable_without_transport(tmp_path):
    import asyncio

    author, _ = _service(tmp_path, None)
    try:
        asyncio.run(author.refine("any", "tweak"))
    except ShapeAuthoringUnavailable:
        return
    raise AssertionError("expected ShapeAuthoringUnavailable refining with no transport")


def test_post_refine_edits_existing_and_returns_view(tmp_path):
    # author then refine over the real HTTP surface; 200 + updated view, file in place.
    app = _app(
        tmp_path,
        FakeTransport([_parallel_reply("flow"), _collapsed_compositional_reply("flow")]),
    )
    with TestClient(app) as client:
        assert client.post(
            "/shapes/author", json={"description": "gather in parallel + email"}
        ).status_code == 201
        resp = client.post(
            "/shapes/flow/refine",
            json={"instruction": "add a linear foundation phase before the parallel one"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["name"] == "flow"
        assert body["execution"] == "concurrent"  # compositional, not collapsed
        # still exactly one shape listed under that name
        listing = client.get("/shapes").json()["shapes"]
        assert sum(1 for s in listing if s["name"] == "flow") == 1


def test_post_refine_404_missing_and_503_no_model(tmp_path):
    # 404 when the shape doesn't exist
    app = _app(tmp_path / "a", FakeTransport([_parallel_reply("x")]))
    with TestClient(app) as client:
        assert client.post(
            "/shapes/ghost/refine", json={"instruction": "tweak"}
        ).status_code == 404

    # 503 with no live model
    app_off = _app(tmp_path / "b", None)
    with TestClient(app_off) as client:
        assert client.post(
            "/shapes/any/refine", json={"instruction": "tweak"}
        ).status_code == 503
