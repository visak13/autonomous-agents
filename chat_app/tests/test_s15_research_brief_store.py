"""s15/a16 (d185) — NOTES-ARCH Layer 2: the per-research BRIEF stored at the
CHAT-SESSION level.

Fast OFFLINE gate (no GPU, no network). Proves the chat-session store is:

* ADDRESSABLE per research — MULTIPLE researches in ONE chat coexist, each looked up
  by its (chat_id, research_id) key and listed distinctly (the headline assertion the
  a16 task requires);
* CROSS-SESSION ISOLATED — a different chat never sees another's briefs;
* UPSERT on re-run — re-saving the same key refreshes the digest, never duplicates;
* RESTART-SAFE — a brand-new ChatStore over the same dir reads the briefs back;
* ADDITIVE-SAFE — adding the research_briefs schema leaves existing chats/turns/
  artifacts and their rows fully intact (the SECURITY extend-not-clobber rule).
"""
from __future__ import annotations

from chat_app.persistence import ChatStore


def _brief(topic: str, thesis: list[str], sids: list[int]) -> dict:
    return {
        "shape": "per_research_brief",
        "topic": topic,
        "thesis": thesis,
        "concerns": [],
        "sources": [{"sid": s, "url": f"https://x/{s}", "title": f"S{s}"} for s in sids],
        "open_gaps": [],
        "concern_count": len(sids),
        "settled_count": len(sids),
        "source_count": len(sids),
        "open_gap_count": 0,
        "digest": f"RESEARCH BRIEF — {topic}",
    }


# =========================================================================== #
# 1. THE HEADLINE: multiple researches in ONE session → distinct addressable briefs.
# =========================================================================== #
def test_multiple_researches_one_session_distinct_addressable(tmp_path):
    with ChatStore(tmp_path) as store:
        chat = store.create_chat("research thread")
        cid = chat.chat_id

        store.save_research_brief(cid, "run-iran", _brief("US-Iran conflict", ["ceasefire"], [1, 2]))
        store.save_research_brief(cid, "run-econ", _brief("inflation outlook", ["CPI up"], [1]))

        # each is independently ADDRESSABLE by its (chat_id, research_id) key.
        iran = store.get_research_brief(cid, "run-iran")
        econ = store.get_research_brief(cid, "run-econ")
        assert iran is not None and econ is not None
        assert iran.topic == "US-Iran conflict"
        assert econ.topic == "inflation outlook"
        assert iran.brief["thesis"] == ["ceasefire"]
        assert econ.brief["thesis"] == ["CPI up"]
        assert iran.brief != econ.brief

        # both coexist in the session's brief index, distinct.
        listing = store.list_research_briefs(cid)
        assert {r.research_id for r in listing} == {"run-iran", "run-econ"}
        assert len({r.topic for r in listing}) == 2


# =========================================================================== #
# 2. cross-session isolation — a different chat never sees another's briefs.
# =========================================================================== #
def test_cross_session_isolation(tmp_path):
    with ChatStore(tmp_path) as store:
        a = store.create_chat("A").chat_id
        b = store.create_chat("B").chat_id
        store.save_research_brief(a, "run-1", _brief("topic A", ["a"], [1]))

        assert store.get_research_brief(b, "run-1") is None
        assert store.list_research_briefs(b) == []
        assert len(store.list_research_briefs(a)) == 1


# =========================================================================== #
# 3. UPSERT — re-running the same research refreshes, never duplicates.
# =========================================================================== #
def test_resave_same_key_upserts(tmp_path):
    with ChatStore(tmp_path) as store:
        cid = store.create_chat("c").chat_id
        store.save_research_brief(cid, "run-1", _brief("v1", ["first"], [1]))
        store.save_research_brief(cid, "run-1", _brief("v2", ["second"], [1, 2]))

        listing = store.list_research_briefs(cid)
        assert len(listing) == 1            # one row, not two
        rec = store.get_research_brief(cid, "run-1")
        assert rec.topic == "v2"
        assert rec.brief["thesis"] == ["second"]


# =========================================================================== #
# 4. restart-safe — a fresh store over the same dir reads the briefs back.
# =========================================================================== #
def test_survives_fresh_store_open(tmp_path):
    with ChatStore(tmp_path) as store:
        cid = store.create_chat("c").chat_id
        store.save_research_brief(cid, "run-1", _brief("persisted", ["claim"], [1]))

    with ChatStore(tmp_path) as store2:
        rec = store2.get_research_brief(cid, "run-1")
        assert rec is not None
        assert rec.topic == "persisted"
        assert rec.brief["thesis"] == ["claim"]


# =========================================================================== #
# 5. ADDITIVE-SAFE — the new schema leaves existing chats/turns/artifacts intact
#    (a store created BEFORE briefs existed still reads back, then takes briefs).
# =========================================================================== #
def test_brief_schema_add_is_additive_safe(tmp_path):
    # Seed a chat with a turn + an artifact FIRST (the pre-existing data).
    with ChatStore(tmp_path) as store:
        turn = store.save_turn(None, "tell me about X", [{"node_id": "n1"}], "the answer")
        cid = turn.chat_id
        ref = store.save_artifact(cid, "report.html", b"<html>hi</html>", "text/html")

    # A fresh store re-runs _create_schema (the IF NOT EXISTS brief table add).
    with ChatStore(tmp_path) as store2:
        # the pre-existing turn + artifact survive byte-for-byte.
        rec = store2.get_chat(cid)
        assert rec is not None
        assert len(rec.turns) == 1
        assert rec.turns[0].user_request == "tell me about X"
        assert rec.turns[0].final_response == "the answer"
        assert len(rec.artifacts) == 1
        assert rec.artifacts[0].artifact_id == ref.artifact_id
        path = store2.get_artifact_path(ref.artifact_id)
        assert path is not None and path.read_bytes() == b"<html>hi</html>"

        # and the same chat can now carry a research brief alongside its history.
        store2.save_research_brief(cid, "run-1", _brief("X", ["fact"], [1]))
        assert store2.get_research_brief(cid, "run-1").brief["thesis"] == ["fact"]
        # the brief add did not disturb the turn/artifact.
        rec2 = store2.get_chat(cid)
        assert len(rec2.turns) == 1 and len(rec2.artifacts) == 1
