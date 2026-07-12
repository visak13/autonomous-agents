"""s15/a17 (d185) — NOTES-ARCH Layer 3: SESSION BINDING wiring + chat.db safety.

Fast OFFLINE gate (no GPU, no network). Proves the chat-route side of session binding:

* the research-state PATH is keyed by the CHAT SESSION when present — same session →
  same file (sticks), different sessions → different files (isolation), and the session
  key is NAMESPACED so it can never collide with a run-id file;
* with NO session id the path stays run-scoped (byte-identical to pre-a17), and a missing
  run id still yields a valid unique path;
* SECURITY (the extend-not-clobber rule): the a17 change touches ONLY the tempdir research
  state — chat.db is untouched, so a chat's pre-existing turns/artifacts/briefs survive a
  fresh ChatStore open fully intact.
"""
from __future__ import annotations

from chat_app.agentic import _research_state_path
from chat_app.persistence import ChatStore


# =========================================================================== #
# 1. session-keyed path — same session sticks, different sessions isolate.
# =========================================================================== #
def test_path_keyed_by_session_when_present():
    a1 = _research_state_path("run-1", "chat-A")
    a2 = _research_state_path("run-2", "chat-A")   # different RUN, same SESSION
    b = _research_state_path("run-1", "chat-B")    # different SESSION

    assert a1 == a2                                 # same session → SAME sticky file
    assert a1 != b                                  # different session → isolated file
    assert a1.endswith("sess__chat-A.jsonl")        # namespaced session key
    assert b.endswith("sess__chat-B.jsonl")


def test_path_run_scoped_without_session():
    # No session id → run-scoped (pre-a17 behaviour): keyed by run id, NOT namespaced.
    r1 = _research_state_path("run-1", None)
    r2 = _research_state_path("run-2", None)
    assert r1 != r2
    assert "sess__" not in r1
    assert r1.endswith("run-1.jsonl")

    # A missing run id still yields a usable, unique path (uuid fallback) — never a crash.
    u1 = _research_state_path(None, None)
    u2 = _research_state_path(None, None)
    assert u1 != u2 and u1.endswith(".jsonl")


def test_session_key_cannot_collide_with_run_id():
    # A run named exactly like a session must NOT map to the session's sticky file.
    run_named_like_session = _research_state_path("chat-A", None)
    real_session = _research_state_path("run-x", "chat-A")
    assert run_named_like_session != real_session


# =========================================================================== #
# 2. SECURITY — a17 leaves chat.db schema + data fully intact (additive-safe).
# =========================================================================== #
def test_chat_db_data_and_schema_intact_after_a17(tmp_path):
    # Seed a chat with a turn, an artifact, and a research brief (pre-existing data).
    with ChatStore(tmp_path) as store:
        turn = store.save_turn(None, "tell me about X", [{"node_id": "n1"}], "answer X")
        cid = turn.chat_id
        ref = store.save_artifact(cid, "report.html", b"<html>hi</html>", "text/html")
        store.save_research_brief(cid, "run-1", {
            "shape": "per_research_brief", "topic": "X", "thesis": ["fact"],
            "concerns": [], "sources": [], "open_gaps": [],
            "concern_count": 0, "settled_count": 0, "source_count": 0,
            "open_gap_count": 0, "digest": "RESEARCH BRIEF — X",
        })

    # A fresh store over the same dir (re-runs schema create) reads EVERYTHING back intact.
    with ChatStore(tmp_path) as store2:
        rec = store2.get_chat(cid)
        assert rec is not None
        assert len(rec.turns) == 1
        assert rec.turns[0].user_request == "tell me about X"
        assert rec.turns[0].final_response == "answer X"
        assert len(rec.artifacts) == 1
        assert rec.artifacts[0].artifact_id == ref.artifact_id
        path = store2.get_artifact_path(ref.artifact_id)
        assert path is not None and path.read_bytes() == b"<html>hi</html>"
        brief = store2.get_research_brief(cid, "run-1")
        assert brief is not None and brief.brief["thesis"] == ["fact"]
