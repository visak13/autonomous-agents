"""Offline tests for the DISTINCT interactive spec-authoring chat surface (s4/b1).

All offline (deterministic SpecConversation FakeTransport seam — no live phi, no
GPU, d8). Two properties the action requires are proven here:

1. open → message(intent) → message(critique) → approve drives a REAL multi-turn
   definition and registers a PLANNER-LOADABLE spec (the body the planner loads
   equals the conversed body; the registry index lists it).
2. the redraft path is OFFLOADED off the event loop (d4 non-freeze): while a
   slow (blocking) redraft is in flight, a concurrent request still returns
   promptly — the one event loop is not frozen.
"""
from __future__ import annotations

import asyncio
import threading

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_framework import FakeTransport
from specialization import SpecRegistry
from specialization.compiler import OFFLINE_MARKER

from chat_app.app import create_app
from chat_app.spec_chat import (
    SpecChatService,
    build_spec_chat_service,
    register_spec_chat_routes,
)

# A collision-free sentinel rule the critique introduces — an improbable heading
# that the initial author draft will NOT contain, so its absent→present crossing
# is an unambiguous, label-agnostic proof the refine re-drafted the body.
SENTINEL_RULE = "Always include a section titled '## ZZZ-Sentinel-Citations'."
SENTINEL = "## ZZZ-Sentinel-Citations"


def test_open_message_critique_approve_registers_loadable_spec(tmp_path) -> None:
    """The full interactive flow over HTTP: open → intent → critique → approve,
    proving multi-turn refine + a planner-loadable compiled spec."""
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        # legacy one-shot surface is left intact ALONGSIDE the new one (d11).
        assert client.get("/specializations").status_code == 200

        # open
        opened = client.post(
            "/spec-chats",
            json={"name": "daily-brief-ruleset", "description": "shape a daily brief"},
        )
        assert opened.status_code == 201, opened.text
        session_id = opened.json()["session_id"]
        assert opened.json()["started"] is False

        # turn 1 — INTENT authors draft 1
        r1 = client.post(
            f"/spec-chats/{session_id}/message",
            json={"message": "Shape a concise daily news brief as an output ruleset."},
        )
        assert r1.status_code == 200, r1.text
        draft1 = r1.json()["draft"]["body"]
        assert draft1.strip()
        assert r1.json()["draft"]["turn"] == 1
        assert SENTINEL not in draft1  # sentinel absent BEFORE the critique

        # turn 2 — CRITIQUE re-drafts the body (the new seam)
        r2 = client.post(
            f"/spec-chats/{session_id}/message",
            json={"message": SENTINEL_RULE},
        )
        assert r2.status_code == 200, r2.text
        draft2 = r2.json()["draft"]["body"]
        assert r2.json()["draft"]["turn"] == 2
        # CONTRASTIVE, label-agnostic: the concrete sentinel rule is ABSENT in
        # draft1 and PRESENT in draft2 → the body genuinely changed per turn.
        assert SENTINEL in draft2
        assert draft2 != draft1

        # transcript carries BOTH user turns + BOTH redrafted bodies
        view = client.get(f"/spec-chats/{session_id}").json()
        roles = [t["role"] for t in view["turns"]]
        assert roles.count("user") == 2 and roles.count("agent") == 2
        assert view["started"] is True
        assert view["draft"]["body"] == draft2

        # approve → compile + register
        appr = client.post(f"/spec-chats/{session_id}/approve")
        assert appr.status_code == 200, appr.text
        assert appr.json()["registered"] is True
        assert appr.json()["state"] == "approved"
        assert appr.json()["name"] == "daily-brief-ruleset"

        # PLANNER-LOADABLE: the registry index lists it AND load() returns the
        # conversed body (the planner can load exactly what was approved).
        reg: SpecRegistry = app.state.wiring.registry
        assert "daily-brief-ruleset" in reg.names()
        loaded = reg.load("daily-brief-ruleset")
        assert loaded.body == draft2
        assert SENTINEL in loaded.body


# --------------------------------------------------------------------------- #
# s4/RC7 — the re-editable surface: create → fetch-by-id → update → re-fetch,
# AND effective on the next run (proven through the SAME loader the runtime uses).
# --------------------------------------------------------------------------- #
def _author_and_register(client, name: str, intent: str) -> str:
    """Helper: drive open → intent → approve so a spec is persisted; return the
    body that was registered."""
    sid = client.post(
        "/spec-chats", json={"name": name, "description": "d"}
    ).json()["session_id"]
    body = client.post(
        f"/spec-chats/{sid}/message", json={"message": intent}
    ).json()["draft"]["body"]
    appr = client.post(f"/spec-chats/{sid}/approve")
    assert appr.status_code == 200, appr.text
    return body


def test_reeditable_roundtrip_create_fetch_update_refetch_effective(tmp_path) -> None:
    """The action's acceptance gate over HTTP: a spec is CREATED+persisted, then
    FETCHED by id, UPDATED, and a RE-FETCH shows the edit — and the edit is
    EFFECTIVE on the next run via the SAME SpecLoader path the runtime composes a
    node's spec body through."""
    from specialization.loader import SpecLoader

    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        # CREATE + PERSIST.
        original_body = _author_and_register(
            client, "editable-ruleset", "shape a concise brief"
        )

        # LIST (body-free) shows it.
        listed = client.get("/spec-chats/registered")
        assert listed.status_code == 200, listed.text
        rows = {r["name"]: r for r in listed.json()}
        assert "editable-ruleset" in rows
        assert "body" not in rows["editable-ruleset"]  # d10: list is body-free

        # FETCH ONE BY ID — full spec (body + provenance) for the re-open view.
        full = client.get("/spec-chats/registered/editable-ruleset")
        assert full.status_code == 200, full.text
        assert full.json()["body"] == original_body
        assert full.json()["source"] == "ui"
        created_at_before = full.json()["created_at"]

        # UPDATE it (direct PUT — no chat round-trip).
        edited_body = "# Edited\n\nNEW-EDIT-TOKEN: lead with the outcome.\n"
        put = client.put(
            "/spec-chats/registered/editable-ruleset",
            json={"description": "an EDITED ruleset", "body": edited_body},
        )
        assert put.status_code == 200, put.text
        assert "NEW-EDIT-TOKEN" in put.json()["body"]

        # RE-FETCH shows the edit persisted, identity + provenance preserved.
        refetch = client.get("/spec-chats/registered/editable-ruleset").json()
        assert "NEW-EDIT-TOKEN" in refetch["body"]
        assert refetch["description"] == "an EDITED ruleset"
        assert refetch["source"] == "ui"                  # provenance preserved
        assert refetch["created_at"] == created_at_before  # not a fresh compile

        # EFFECTIVE NEXT RUN: the runtime composes a node's body via
        # SpecLoader.load_body(name) → registry.load. Read through that EXACT path
        # (not just the HTTP store) to prove the next launched node sees the edit.
        # (the doc write strips trailing whitespace — the persisted/effective
        # body is the stripped form, which is what a node would load.)
        reg: SpecRegistry = app.state.wiring.registry
        assert SpecLoader(reg).load_body("editable-ruleset") == edited_body.strip()
        # A FRESH registry over the SAME specs dir agrees → durable on disk, not
        # just in-memory (registry.load re-reads the doc each call).
        fresh = SpecRegistry(reg.specs_dir)
        assert "NEW-EDIT-TOKEN" in fresh.load("editable-ruleset").body


def test_reopen_existing_spec_into_editable_session(tmp_path) -> None:
    """RC7: re-open an EXISTING registered spec into an editable chat session —
    the session begins already-started on the existing body, a refine edits it,
    and approve re-registers it under the SAME name (effective next run)."""
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        original_body = _author_and_register(
            client, "reopenable-ruleset", "shape a concise brief"
        )

        # RE-OPEN → an editable session seeded with the existing body.
        reopened = client.post(
            "/spec-chats/reopen", json={"name": "reopenable-ruleset"}
        )
        assert reopened.status_code == 201, reopened.text
        view = reopened.json()
        assert view["started"] is True
        assert view["draft"]["body"] == original_body
        sid = view["session_id"]

        # EDIT via a refine turn, then APPROVE → re-registered under same name.
        r = client.post(
            f"/spec-chats/{sid}/message",
            json={"message": SENTINEL_RULE},
        )
        assert r.status_code == 200, r.text
        assert SENTINEL in r.json()["draft"]["body"]
        assert client.post(f"/spec-chats/{sid}/approve").status_code == 200

        reg: SpecRegistry = app.state.wiring.registry
        assert SENTINEL in reg.load("reopenable-ruleset").body  # edit persisted


def test_reeditable_unknown_spec_is_404_and_empty_update_is_422(tmp_path) -> None:
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        assert client.get("/spec-chats/registered/nope").status_code == 404
        assert (
            client.put("/spec-chats/registered/nope", json={"body": "x"}).status_code
            == 404
        )
        assert (
            client.post("/spec-chats/reopen", json={"name": "nope"}).status_code == 404
        )
        # an update with neither field is a 422 (nothing to change).
        _author_and_register(client, "present-ruleset", "shape it")
        assert (
            client.put("/spec-chats/registered/present-ruleset", json={}).status_code
            == 422
        )


def test_deny_blocks_further_authoring_and_does_not_register(tmp_path) -> None:
    """deny closes the session without compiling; a later message is refused."""
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        sid = client.post(
            "/spec-chats", json={"name": "throwaway-ruleset"}
        ).json()["session_id"]
        client.post(f"/spec-chats/{sid}/message", json={"message": "shape something"})

        denied = client.post(f"/spec-chats/{sid}/deny")
        assert denied.status_code == 200
        assert denied.json()["state"] == "denied"

        # authoring after a terminal state is a 409, and nothing was registered.
        again = client.post(f"/spec-chats/{sid}/message", json={"message": "more"})
        assert again.status_code == 409
        assert "throwaway-ruleset" not in app.state.wiring.registry.names()


def test_unknown_session_is_404(tmp_path) -> None:
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        assert client.get("/spec-chats/nope").status_code == 404
        assert (
            client.post("/spec-chats/nope/message", json={"message": "x"}).status_code
            == 404
        )
        assert client.post("/spec-chats/nope/approve").status_code == 404


def test_approve_before_any_turn_is_409(tmp_path) -> None:
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        sid = client.post(
            "/spec-chats", json={"name": "empty-ruleset"}
        ).json()["session_id"]
        assert client.post(f"/spec-chats/{sid}/approve").status_code == 409


# --------------------------------------------------------------------------- #
# d4 non-freeze: the redraft is offloaded off the event loop
# --------------------------------------------------------------------------- #
def _blocking_transport(gate: threading.Event) -> FakeTransport:
    """A transport whose chain call BLOCKS until ``gate`` is set — stands in for a
    slow, GPU-contended live phi round-trip so we can prove the redraft does not
    freeze the loop."""

    def _reply(messages, **opts):  # called by FakeTransport with (messages, **opts)
        gate.wait(timeout=10)
        return (
            "# Output-shaping ruleset: blocking\n\n"
            "**Mission.** do the task then shape the output per the rules.\n\n"
            "## Rules\n- Lead with the outcome.\n\n" + OFFLINE_MARKER
        )

    # repeats the last reply if the chain calls more than once.
    return FakeTransport([_reply])


def _bare_spec_chat_app(service: SpecChatService) -> FastAPI:
    app = FastAPI()
    register_spec_chat_routes(app, service)

    @app.get("/ping")
    async def ping() -> dict:  # a trivial loop-only route used as the liveness probe
        return {"ok": True}

    return app


def test_redraft_is_offloaded_off_the_event_loop(tmp_path) -> None:
    """While a blocking redraft is in flight, a concurrent request still returns
    promptly — proving ``/message`` offloads the blocking chain call off the loop
    (d4). If the handler ran the redraft inline, the loop would be frozen and the
    concurrent GET would not complete until the redraft finished."""

    async def _run() -> None:
        gate = threading.Event()
        registry = SpecRegistry(tmp_path / "specs")
        service = build_spec_chat_service(
            registry, transport=_blocking_transport(gate)
        )
        app = _bare_spec_chat_app(service)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            sid = (
                await client.post("/spec-chats", json={"name": "nonfreeze-ruleset"})
            ).json()["session_id"]

            # fire the blocking redraft; it parks in the threadpool on gate.wait()
            msg_task = asyncio.create_task(
                client.post(
                    f"/spec-chats/{sid}/message", json={"message": "shape it"}
                )
            )
            # give the handler a moment to enter to_thread (NOT to finish).
            await asyncio.sleep(0.2)
            assert not msg_task.done(), "redraft should still be blocked at the gate"

            # the loop is FREE: a concurrent request returns promptly while the
            # redraft is blocked. (If the redraft blocked the loop, this await
            # would itself hang until the 10s gate timeout.)
            pong = await asyncio.wait_for(client.get("/ping"), timeout=2.0)
            assert pong.status_code == 200 and pong.json()["ok"] is True
            assert not msg_task.done(), "redraft still blocked after the live probe"

            # release the redraft and confirm it completed correctly.
            gate.set()
            resp = await asyncio.wait_for(msg_task, timeout=5.0)
            assert resp.status_code == 200, resp.text
            assert resp.json()["draft"]["turn"] == 1

    asyncio.run(_run())
