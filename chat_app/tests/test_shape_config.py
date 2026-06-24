"""Backend proof for the s4/a4 shape-config store + API + runtime wiring (d5).

The acceptance gate (s4/a4): *a per-shape ``max_iter`` set via the API persists to
SQLite and a deep-research unroll uses the override count (NOT the text-file
default).* These tests prove exactly that chain, end to end, with NO live model:

1. STORE DURABILITY — an override written through :class:`ShapeConfigStore` is in
   the SHARED ``chat.db`` and a FRESH connection reads it back (real persistence,
   not an in-memory cache).
2. UNROLL HONORS THE OVERRIDE — the generic :func:`~agent_runtime.unroll_shape`
   emits exactly ``override`` research rounds when given the stored value, vs the
   text-file default (10) when given none (a3: there is no per-shape executor).
3. SERVICE VIEW — the merged view surfaces the stored override + the EFFECTIVE cap
   (clamped to the shape's ``hard_cap``), the number the runtime actually runs.
4. API → SQLite — ``PUT /shapes/{name}/max_iter`` over the real app persists the
   value; a fresh store opened on the SAME data dir reads it.
5. THE FULL WIRING — set via the API, then drive :func:`run_agentic` (on a
   deterministic FakeTransport that selects ``deep-research``) handed the app's
   SAME store: the unroll runs the API-set count, not the file default. This is the
   real routed entrypoint, so it proves the override is actually honored, not just
   stored.
"""
from __future__ import annotations

import asyncio
import dataclasses

from fastapi import FastAPI
from fastapi.testclient import TestClient
from llm_framework import FakeTransport

from agent_runtime import load_shape, unroll_shape
from agent_runtime.shape_author import write_shape

from chat_app.app import create_app
from chat_app.agentic import run_agentic
from chat_app.shape_config import (
    ShapeConfigService,
    ShapeConfigStore,
    register_shape_routes,
)

DEEP_RESEARCH = "deep-research"
FILE_DEFAULT_MAX_ITER = 10  # the deep-research.toml default
HARD_CAP = 24               # the deep-research.toml absolute bound


def _rounds_in(dag) -> int:
    """The number of research ROUNDS in an unrolled deep-research DAG.

    One research node is emitted per round (final round included), so the count of
    research-POSITION nodes (id suffix ``_research``; d48 — they are worker nodes now)
    IS the effective round count the unroll ran."""
    return sum(1 for n in dag.nodes if n.id.endswith("_research"))


# --------------------------------------------------------------------------- #
# 1) the store persists to the SHARED SQLite (a fresh connection reads it back)
# --------------------------------------------------------------------------- #
def test_store_override_persists_across_connections(tmp_path):
    with ShapeConfigStore(tmp_path) as store:
        assert store.get_max_iter(DEEP_RESEARCH) is None  # nothing set yet
        store.set_max_iter(DEEP_RESEARCH, 3)

    # A brand-new connection (and a brand-new store object) on the SAME data dir
    # reads the committed value — durable, not in-memory.
    with ShapeConfigStore(tmp_path) as fresh:
        assert fresh.get_max_iter(DEEP_RESEARCH) == 3
        # and the db is the shared chat.db, not a separate file
        assert fresh.db_path.name == "chat.db"


def test_store_rejects_nonsensical_override(tmp_path):
    with ShapeConfigStore(tmp_path) as store:
        for bad in (0, -1):
            try:
                store.set_max_iter(DEEP_RESEARCH, bad)
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for max_iter={bad}")


# --------------------------------------------------------------------------- #
# 2) the deep-research UNROLL uses the override count, not the file default
# --------------------------------------------------------------------------- #
def test_build_dag_uses_override_round_count():
    shape = load_shape(DEEP_RESEARCH)
    # no override → the text-file default (10 rounds)
    default_dag = unroll_shape(shape, "research X deeply", max_iter_override=None)
    assert _rounds_in(default_dag) == FILE_DEFAULT_MAX_ITER

    # an override → exactly that many rounds (the s4 UI value drives the unroll)
    overridden = unroll_shape(shape, "research X deeply", max_iter_override=3)
    assert _rounds_in(overridden) == 3
    assert _rounds_in(overridden) != FILE_DEFAULT_MAX_ITER


def test_build_dag_override_is_clamped_to_hard_cap():
    shape = load_shape(DEEP_RESEARCH)
    # a UI value above the safety bound never exceeds hard_cap (shared-GPU guard)
    huge = unroll_shape(shape, "research X", max_iter_override=999)
    assert _rounds_in(huge) == HARD_CAP


# --------------------------------------------------------------------------- #
# 3) the service view merges the override + the effective (clamped) cap
# --------------------------------------------------------------------------- #
def test_service_view_surfaces_override_and_effective(tmp_path):
    service = ShapeConfigService(ShapeConfigStore(tmp_path))

    before = service.get_shape(DEEP_RESEARCH)
    assert before is not None
    assert before["max_iter_override"] is None
    assert before["effective_max_iter"] == FILE_DEFAULT_MAX_ITER
    # the structure the s4 screen renders is present
    assert before["round_roles"] == ["research", "critic"]
    assert before["final_roles"] == ["research", "synthesis", "verify"]
    assert before["execution"] == "deep-research"

    updated = service.set_max_iter(DEEP_RESEARCH, 4)
    assert updated is not None
    assert updated["max_iter_override"] == 4
    assert updated["effective_max_iter"] == 4

    # a value above hard_cap stores raw but the EFFECTIVE cap is clamped
    clamped = service.set_max_iter(DEEP_RESEARCH, 100)
    assert clamped["max_iter_override"] == 100
    assert clamped["effective_max_iter"] == HARD_CAP

    # an unknown shape is not stored (route maps to 404)
    assert service.set_max_iter("does-not-exist", 5) is None
    assert service.get_shape("does-not-exist") is None


# --------------------------------------------------------------------------- #
# 4) the API persists the override to SQLite (a fresh store reads it back)
# --------------------------------------------------------------------------- #
def test_api_put_max_iter_persists_to_sqlite(tmp_path):
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        # list surfaces every text-file shape with its (initially absent) override
        listing = client.get("/shapes")
        assert listing.status_code == 200
        names = {s["name"] for s in listing.json()["shapes"]}
        assert DEEP_RESEARCH in names

        resp = client.put(f"/shapes/{DEEP_RESEARCH}/max_iter", json={"max_iter": 6})
        assert resp.status_code == 200
        body = resp.json()
        assert body["max_iter_override"] == 6
        assert body["effective_max_iter"] == 6

        # unknown shape → 404; nonsensical value → 422 (Pydantic ge=1)
        assert client.put("/shapes/nope/max_iter", json={"max_iter": 6}).status_code == 404
        assert client.put(
            f"/shapes/{DEEP_RESEARCH}/max_iter", json={"max_iter": 0}
        ).status_code == 422

    # PROVE PERSISTENCE: a fresh store on the SAME data dir reads the API-set value
    with ShapeConfigStore(tmp_path) as fresh:
        assert fresh.get_max_iter(DEEP_RESEARCH) == 6


# --------------------------------------------------------------------------- #
# 5) THE FULL WIRING — API set → run_agentic reads the SAME store → unroll honors it
# --------------------------------------------------------------------------- #
def _deep_research_fake_transport() -> FakeTransport:
    """A transport that selects ``deep-research`` then answers every role node.

    Reply 1 is the shape-selection JSON; reply 2 is a generic role payload that
    FakeTransport repeats for every subsequent role call (research/critic/
    synthesis/verify) — enough to drive the bounded unroll to completion offline.
    The unroll's round COUNT (what a4 proves) is set by max_iter, not by anything
    the role nodes return."""
    # Reply 0 is the interactive AMBIGUITY assessment (scenario-2): run_agentic now
    # asks the planner whether the request is underspecified BEFORE shape selection.
    # A clear "research the topic in depth" request is not ambiguous, so this returns
    # needs_clarification=false and the run proceeds to selection (reply 1) unchanged.
    ambiguity = '{"needs_clarification": false, "question": "", "rationale": "clear"}'
    selection = '{"shape": "deep-research", "rationale": "iterative depth research"}'
    role = (
        '{"findings": ["f"], "sources": ["s"], "open_questions": ["q"], '
        '"gaps": [], "weak_claims": [], "follow_up_queries": ["nq"], '
        '"verdict": "converged", "fixed_inline": []}'
    )
    return FakeTransport([ambiguity, selection, role])


def test_run_agentic_honors_api_set_override_for_deep_research(tmp_path):
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        # 1) set the override THROUGH THE API (the s4 screen's write path)
        resp = client.put(f"/shapes/{DEEP_RESEARCH}/max_iter", json={"max_iter": 3})
        assert resp.status_code == 200

        # 2) drive the REAL routed entrypoint (run_agentic) on a deterministic
        #    transport, handed the app's SAME shared store — the exact object the
        #    live /chat path passes. No explicit max_iter_override: the value MUST
        #    come from the store the API just wrote.
        w = app.state.wiring
        result = asyncio.run(
            run_agentic(
                "research the topic in depth",
                transport=_deep_research_fake_transport(),
                registry=w.registry,
                hook=w.hook,
                plane=w.plane,
                shape_config=w.shape_config,
            )
        )

    assert result.shape == DEEP_RESEARCH
    dr = result.deep_research
    assert dr is not None
    # the unroll ran the API-SET count (3), NOT the text-file default (10). The
    # deep-research run summary is now a plain dict (a3: no DeepResearchResult).
    assert dr["effective_max_iter"] == 3
    assert dr["rounds_executed"] == 3
    # the unrolled DAG the generic runtime actually drove carries exactly 3 rounds
    assert _rounds_in(result.dag) == 3


# --------------------------------------------------------------------------- #
# 6) b9 (d13 UI delete) — DELETE a USER-AUTHORED shape: the selection catalog
#    (load_shapes, what the planner selects a shape from) SHRINKS, its persisted
#    max_iter override row is cleaned up (no orphaned row outlives the file), a
#    shipped BUILT-IN is REFUSED (409, only the user's own noise is clearable),
#    and the surviving shape is untouched (no orphaned reference breaks selection).
# --------------------------------------------------------------------------- #
def _seed_shapes_dir(tmp_path):
    """A temp shapes dir holding one shipped BUILT-IN (``linear``) plus one
    USER-AUTHORED shape (``my-test-shape``, a renamed copy so it is structurally
    valid). Returns the dir. Built on the production :func:`write_shape` author
    path, so the files load through the same :func:`load_shapes` the route reads."""
    shapes_dir = tmp_path / "shapes"
    shapes_dir.mkdir()
    write_shape(load_shape("linear"), shapes_dir=shapes_dir)  # a real built-in
    user = dataclasses.replace(
        load_shape("linear"),
        name="my-test-shape",
        description="a user-authored throwaway shape",
    )
    write_shape(user, shapes_dir=shapes_dir)
    return shapes_dir


def test_delete_user_shape_shrinks_catalog_builtin_protected(tmp_path):
    """The b9 acceptance over HTTP: ``DELETE /shapes/{name}`` removes a USER shape
    so the SELECTION CATALOG (``GET /shapes`` ← ``load_shapes``, what the planner
    picks a shape from) shrinks by exactly that shape; its persisted ``max_iter``
    override row is dropped (no orphaned row outlives the deleted file); the shipped
    BUILT-IN is REFUSED (409) so only the user's own shapes are clearable; and the
    surviving built-in stays selectable (no orphaned reference breaks selection).
    A re-delete / unknown name is a clean 404."""
    shapes_dir = _seed_shapes_dir(tmp_path)
    store = ShapeConfigStore(tmp_path)
    service = ShapeConfigService(store, shapes_dir=shapes_dir)
    app = FastAPI()
    register_shape_routes(app, service)

    with TestClient(app) as client:
        # the selection catalog holds BOTH the built-in and the user shape.
        before = {s["name"] for s in client.get("/shapes").json()["shapes"]}
        assert {"linear", "my-test-shape"} <= before

        # give the user shape a PERSISTED override so the delete must clean the row.
        assert (
            client.put("/shapes/my-test-shape/max_iter", json={"max_iter": 5}).status_code
            == 200
        )
        assert store.get_max_iter("my-test-shape") == 5

        # DELETE the user shape → 200 with the {ok, deleted} receipt the UI drops on.
        deleted = client.delete("/shapes/my-test-shape")
        assert deleted.status_code == 200, deleted.text
        assert deleted.json() == {"ok": True, "deleted": "my-test-shape"}

        # the catalog SHRANK by EXACTLY the user shape.
        after = {s["name"] for s in client.get("/shapes").json()["shapes"]}
        assert "my-test-shape" not in after
        assert before - after == {"my-test-shape"}
        # and NO orphaned override row outlives the unlinked file (FILE-then-ROW).
        assert store.get_max_iter("my-test-shape") is None

        # NO ORPHAN BREAK: the surviving built-in is still listed AND viewable.
        assert "linear" in after
        assert client.get("/shapes/linear").status_code == 200

        # BUILT-IN PROTECTED: deleting a shipped shape is refused (409), file kept.
        assert client.delete("/shapes/linear").status_code == 409
        assert "linear" in {
            s["name"] for s in client.get("/shapes").json()["shapes"]
        }

        # a re-delete / unknown shape is a clean 404, never a 500.
        assert client.delete("/shapes/my-test-shape").status_code == 404
        assert client.delete("/shapes/never-existed").status_code == 404

    store.close()
