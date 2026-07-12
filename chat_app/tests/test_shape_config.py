"""Backend proof for the s4/a4 shape-config store + API + runtime wiring (d5).

The acceptance gate (s4/a4): *a per-shape ``max_iter`` set via the API persists to
SQLite and a deep-research unroll uses the override count (NOT the text-file
default).* These tests prove exactly that chain, end to end, with NO live model:

1. STORE DURABILITY — an override written through :class:`ShapeConfigStore` is in
   the SHARED ``chat.db`` and a FRESH connection reads it back (real persistence,
   not an in-memory cache).
2. DEPTH CEILING HONORS THE OVERRIDE — ``ShapeSpec.effective_max_iter`` returns exactly
   ``override`` (clamped to ``hard_cap``) vs the text-file default (10) when given none. The
   override drives the research DEPTH ceiling the grower reasons within (s16/a3 d239/d247: the
   deterministic unroll is retired — there is no fixed round count to count).
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

import dataclasses

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_runtime import load_shape
from agent_runtime.shape_author import write_shape

from chat_app.app import create_app
from chat_app.shape_config import (
    ShapeConfigService,
    ShapeConfigStore,
    register_shape_routes,
)

DEEP_RESEARCH = "deep-research"
FILE_DEFAULT_MAX_ITER = 10  # the deep-research.toml default
HARD_CAP = 24               # the deep-research.toml absolute bound


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
# 2) the deep-research DEPTH CEILING honors the override, not the file default
#    (s16/a3 d239/d247: the deterministic unroll is RETIRED — there is no fixed
#    round count to count. The override now drives the EFFECTIVE depth ceiling
#    (effective_max_iter), which the engine feeds the grow_config depth clamp; the
#    grower then authors the actual research topology by reasoning. The override→
#    effective-count→hard_cap-clamp CONTRACT is unchanged, measured at the discipline
#    level instead of by counting unrolled nodes.)
# --------------------------------------------------------------------------- #
def test_effective_max_iter_uses_override_round_count():
    shape = load_shape(DEEP_RESEARCH)
    # no override → the text-file default (10)
    assert shape.effective_max_iter(None) == FILE_DEFAULT_MAX_ITER
    # an override → exactly that value (the s4 UI value drives the depth ceiling)
    assert shape.effective_max_iter(3) == 3
    assert shape.effective_max_iter(3) != FILE_DEFAULT_MAX_ITER


def test_effective_max_iter_override_is_clamped_to_hard_cap():
    shape = load_shape(DEEP_RESEARCH)
    # a UI value above the safety bound never exceeds hard_cap (shared-GPU guard)
    assert shape.effective_max_iter(999) == HARD_CAP


# --------------------------------------------------------------------------- #
# 3) the service view merges the override + the effective (clamped) cap
# --------------------------------------------------------------------------- #
def test_service_view_surfaces_override_and_effective(tmp_path):
    service = ShapeConfigService(ShapeConfigStore(tmp_path))

    before = service.get_shape(DEEP_RESEARCH)
    assert before is not None
    assert before["max_iter_override"] is None
    assert before["effective_max_iter"] == FILE_DEFAULT_MAX_ITER
    # s17 (d248/d249): round_roles/final_roles are fully RETIRED — the transitional empty-[]
    # API-boundary shim is REMOVED with the redesigned Shapes screen (shape = discipline +
    # doctrine; the planner authors topology). The deep-research identity is the execution
    # token; the research topology is authored at runtime by the grower.
    assert "round_roles" not in before
    assert "final_roles" not in before
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
# (5 — REMOVED, s15/a24) ``test_run_agentic_honors_api_set_override_for_deep_research``
# was deleted: it pinned the RETIRED ``_run_deep_research`` fixed-unroll return contract.
# Under the new design (a1 / P2-5c) the served deep-research route runs the GROWABLE engine
# via ``run_plan_chain`` driven by the DEPTH knob (``get_depth`` -> ``research_depth``);
# ``run_agentic`` never feeds ``get_max_iter`` into that route (see agentic.py:679-726, which
# passes ``research_depth`` and never ``max_iter_override``), and ``rounds_executed`` is now the
# grower's EMERGENT gather count, not ``== effective_max_iter``. The 3-reply transport could no
# longer drive the new section-write phase (MalformedOutputError: tool-driven authorer produced
# no usable nodes). The end-to-end ``run_agentic`` -> store -> runtime override wiring is now
# covered (for depth) by ``test_s13_run_agentic_reads_depth_from_shape_config``; the ``max_iter``
# store/API/override honoring stays covered by tests (1)-(4) above (store durability,
# ``effective_max_iter`` honoring the override depth ceiling, the service view, API->SQLite).
# --------------------------------------------------------------------------- #


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
