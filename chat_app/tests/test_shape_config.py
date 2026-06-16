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

from fastapi.testclient import TestClient
from llm_framework import FakeTransport

from agent_runtime import load_shape, unroll_shape
from agent_runtime.roles import ROLE_RESEARCH

from chat_app.app import create_app
from chat_app.agentic import run_agentic
from chat_app.shape_config import ShapeConfigService, ShapeConfigStore

DEEP_RESEARCH = "deep-research"
FILE_DEFAULT_MAX_ITER = 10  # the deep-research.toml default
HARD_CAP = 24               # the deep-research.toml absolute bound


def _rounds_in(dag) -> int:
    """The number of research ROUNDS in an unrolled deep-research DAG.

    One research node is emitted per round (final round included), so the count of
    research-role nodes IS the effective round count the unroll ran."""
    return sum(1 for n in dag.nodes if n.role == ROLE_RESEARCH)


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
