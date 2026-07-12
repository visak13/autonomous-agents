"""s17 (d18a/d249 parity) — the CONVERSATIONAL shape chat: draft turns + approve gate.

The user-raised asymmetry: the spec chat authors a ruleset over a free-flowing
multi-turn conversation with a draft preview and an approve/deny gate, while the
Shapes screen only had one-shot describe/refine forms. These tests prove the new
shape chat closes that gap on the SAME scripted-transport seam the other authoring
tests use (the live model rides the identical code path):

1. CREATE FLOW — open → message authors a DRAFT (no file written) → a second
   message REFINES the draft BUILDING ON it (same name, updated posture) → approve
   writes the file into the catalog the list reads.
2. DENY — a denied session writes NOTHING and refuses further drives.
3. REFINE SESSION — open(refine_of=X) seeds the draft from the on-disk shape;
   approve overwrites that same shape (never a 409 for its own name).
4. CREATE COLLISION — approving a create-draft whose name already exists is a 409,
   never a silent clobber.
5. OFFLINE — with no live transport a message drive is 503 (never a stubbed shape).
"""
from __future__ import annotations

import asyncio
import json
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient
from llm_framework import FakeTransport

from chat_app.shape_chat import ShapeChatService, register_shape_chat_routes
from chat_app.shape_config import ShapeConfigService, ShapeConfigStore


def _reply(name: str, execution: str = "concurrent", max_iter: int = 1,
           description: str = "gather several independent angles then combine") -> str:
    return json.dumps({
        "name": name,
        "description": description,
        "execution": execution,
        "max_iter": max_iter,
    })


def _service(tmp_path, transport):
    config = ShapeConfigService(ShapeConfigStore(tmp_path), shapes_dir=tmp_path)
    return ShapeChatService(config, transport=transport, shapes_dir=tmp_path), config


def _app(service) -> TestClient:
    app = FastAPI()
    register_shape_chat_routes(app, service)
    return TestClient(app)


# --------------------------------------------------------------------------- #
# 1) create flow: draft → refine builds on it → approve persists
# --------------------------------------------------------------------------- #
def test_create_flow_drafts_refines_and_approves(tmp_path):
    transport = FakeTransport([
        _reply("news-fanout"),
        _reply("news-fanout", execution="sequential", max_iter=1,
               description="a strict one-after-another gather then a combine step"),
    ])
    service, config = _service(tmp_path, transport)
    client = _app(service)

    opened = client.post("/shape-chat", json={}).json()
    sid = opened["session_id"]
    assert opened["mode"] == "create" and opened["draft"] is None

    # turn 1 — a draft exists, but NO file is written yet
    v1 = client.post(f"/shape-chat/{sid}/message",
                     json={"message": "fan out independent news gathers"}).json()
    assert v1["draft"]["name"] == "news-fanout"
    assert v1["draft"]["execution"] == "concurrent"
    assert config.get_shape("news-fanout") is None
    assert not os.path.exists(tmp_path / "news-fanout.toml")

    # turn 2 — the refine BUILDS ON the in-session draft (same name, new posture)
    v2 = client.post(f"/shape-chat/{sid}/message",
                     json={"message": "make the gathers strictly sequential"}).json()
    assert v2["draft"]["name"] == "news-fanout"
    assert v2["draft"]["execution"] == "sequential"
    assert len(v2["turns"]) == 4  # 2 user + 2 assistant
    assert config.get_shape("news-fanout") is None  # still only a draft

    # approve — NOW it persists into the catalog the list reads
    approved = client.post(f"/shape-chat/{sid}/approve").json()
    assert approved["approved"] is True
    assert approved["shape"]["execution"] == "sequential"
    assert config.get_shape("news-fanout") is not None

    # the session is closed; further drives are 409
    assert client.post(f"/shape-chat/{sid}/message",
                       json={"message": "more"}).status_code == 409


# --------------------------------------------------------------------------- #
# 2) deny discards — nothing on disk, session closed
# --------------------------------------------------------------------------- #
def test_deny_discards_draft(tmp_path):
    service, config = _service(tmp_path, FakeTransport([_reply("throwaway")]))
    client = _app(service)
    sid = client.post("/shape-chat", json={}).json()["session_id"]
    client.post(f"/shape-chat/{sid}/message", json={"message": "anything"})
    assert client.post(f"/shape-chat/{sid}/deny").json() == {"denied": True}
    assert config.get_shape("throwaway") is None
    assert client.post(f"/shape-chat/{sid}/approve").status_code == 409


# --------------------------------------------------------------------------- #
# 3) refine session: seeded from disk, approve overwrites in place
# --------------------------------------------------------------------------- #
def test_refine_session_seeds_and_overwrites(tmp_path):
    seed_transport = FakeTransport([_reply("existing-shape")])
    service, config = _service(tmp_path, seed_transport)
    client = _app(service)
    sid = client.post("/shape-chat", json={}).json()["session_id"]
    client.post(f"/shape-chat/{sid}/message", json={"message": "make it"})
    client.post(f"/shape-chat/{sid}/approve")
    assert config.get_shape("existing-shape")["execution"] == "concurrent"

    # a NEW session opened ON that shape starts from its real definition
    service._transport = FakeTransport([
        _reply("existing-shape", execution="sequential"),
    ])
    opened = client.post("/shape-chat", json={"refine_of": "existing-shape"}).json()
    assert opened["mode"] == "refine"
    assert opened["draft"]["name"] == "existing-shape"
    sid2 = opened["session_id"]
    client.post(f"/shape-chat/{sid2}/message", json={"message": "run it in sequence"})
    approved = client.post(f"/shape-chat/{sid2}/approve").json()
    assert approved["shape"]["execution"] == "sequential"
    assert config.get_shape("existing-shape")["execution"] == "sequential"


# --------------------------------------------------------------------------- #
# 4) create collision -> 409, never a clobber
# --------------------------------------------------------------------------- #
def test_create_collision_is_409(tmp_path):
    service, config = _service(tmp_path, FakeTransport([_reply("taken")]))
    client = _app(service)
    sid = client.post("/shape-chat", json={}).json()["session_id"]
    client.post(f"/shape-chat/{sid}/message", json={"message": "make one"})
    client.post(f"/shape-chat/{sid}/approve")

    service._transport = FakeTransport([_reply("taken")])
    sid2 = client.post("/shape-chat", json={}).json()["session_id"]
    client.post(f"/shape-chat/{sid2}/message", json={"message": "make another"})
    resp = client.post(f"/shape-chat/{sid2}/approve")
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# 5) offline seam: drives are 503, no stub-authored shape
# --------------------------------------------------------------------------- #
def test_offline_message_is_503(tmp_path):
    service, config = _service(tmp_path, None)
    client = _app(service)
    sid = client.post("/shape-chat", json={}).json()["session_id"]
    resp = client.post(f"/shape-chat/{sid}/message", json={"message": "make one"})
    assert resp.status_code == 503
    assert list(config.list_shapes()) == []


# --------------------------------------------------------------------------- #
# 6) create-mode rename is honored (live catch); refine-mode keeps the name
# --------------------------------------------------------------------------- #
def test_create_draft_rename_is_honored(tmp_path):
    transport = FakeTransport([
        _reply("first-name"),
        _reply("better-name", execution="concurrent"),
    ])
    service, config = _service(tmp_path, transport)
    client = _app(service)
    sid = client.post("/shape-chat", json={}).json()["session_id"]
    v1 = client.post(f"/shape-chat/{sid}/message", json={"message": "make one"}).json()
    assert v1["draft"]["name"] == "first-name"
    v2 = client.post(f"/shape-chat/{sid}/message",
                     json={"message": "rename it to better-name"}).json()
    assert v2["draft"]["name"] == "better-name"  # create draft: rename honored
    approved = client.post(f"/shape-chat/{sid}/approve").json()
    assert approved["shape"]["name"] == "better-name"
    assert config.get_shape("first-name") is None
