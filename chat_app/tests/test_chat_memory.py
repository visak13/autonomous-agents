"""Backend proof for the s5/a1 conversation-memory layer.

The acceptance gate (s5/a1): a thin :class:`~chat_app.conversation_memory.ConversationMemory`
on top of the durable :class:`~chat_app.persistence.ChatStore` returns, per
``chat_id`` (the thread), a BOUNDED prior-turn context block, in order, strictly
isolated per chat, and surviving a fresh store open against the same DB file.
These tests prove exactly that — no live model, no second DB file:

1. ORDER — recent turns and the assembled block read oldest→newest.
2. BOUNDED — the assembled block never exceeds ``max_chars`` (oldest dropped
   first; a single oversized turn is hard-truncated) and honors ``recent_turns``.
3. ISOLATION — two different ``chat_id`` threads return non-overlapping context;
   neither reads the other's turns or summary.
4. RESTART-DURABLE — turns + a running summary written through one pair of
   stores are read back by a BRAND-NEW ChatStore + ConversationMemory on the
   SAME data dir; the summary lives in the shared ``chat.db`` (no new file).
5. SUMMARY — a per-chat running summary is prepended and is itself bounded.
"""
from __future__ import annotations

from chat_app.persistence import ChatStore
from chat_app.conversation_memory import (
    ASSISTANT_PREFIX,
    SUMMARY_HEADER,
    USER_PREFIX,
    ConversationMemory,
)


def _seed(store: ChatStore, chat_id: str | None, pairs: list[tuple[str, str]]) -> str:
    """Append (user_request, final_response) turns to a chat; return its id."""
    cid = chat_id
    for user, resp in pairs:
        rec = store.save_turn(cid, user, [], resp)
        cid = rec.chat_id  # first call mints the id when cid is None
    assert cid is not None
    return cid


# --------------------------------------------------------------------------- #
# 1) prior turns are returned IN ORDER (oldest -> newest)
# --------------------------------------------------------------------------- #
def test_recent_turns_in_order(tmp_path):
    with ChatStore(tmp_path) as store, ConversationMemory(store) as mem:
        cid = _seed(
            store,
            None,
            [("q0", "a0"), ("q1", "a1"), ("q2", "a2"), ("q3", "a3")],
        )

        turns = mem.recent_turns_for(cid)
        assert [t.user_request for t in turns] == ["q0", "q1", "q2", "q3"]
        assert [t.turn_index for t in turns] == [0, 1, 2, 3]

        # the assembled block preserves chronological order
        ctx = mem.assemble_context(cid)
        assert ctx.index("q0") < ctx.index("q1") < ctx.index("q2") < ctx.index("q3")
        assert f"{USER_PREFIX} q0" in ctx
        assert f"{ASSISTANT_PREFIX} a0" in ctx


def test_recent_turns_limit_keeps_newest(tmp_path):
    with ChatStore(tmp_path) as store, ConversationMemory(store, recent_turns=2) as mem:
        cid = _seed(store, None, [("q0", "a0"), ("q1", "a1"), ("q2", "a2")])

        # only the last 2 turns, still in chronological order
        turns = mem.recent_turns_for(cid)
        assert [t.user_request for t in turns] == ["q1", "q2"]

        ctx = mem.assemble_context(cid)
        assert "q0" not in ctx  # the oldest turn is outside the window
        assert "q1" in ctx and "q2" in ctx

        # a per-call override widens the window past the instance default
        assert len(mem.recent_turns_for(cid, limit=10)) == 3
        assert "q0" in mem.assemble_context(cid, recent_turns=10)


# --------------------------------------------------------------------------- #
# 2) the assembled context is BOUNDED
# --------------------------------------------------------------------------- #
def test_assembled_context_is_bounded(tmp_path):
    with ChatStore(tmp_path) as store:
        # many large turns; a small char budget must drop the oldest, keep newest
        big = "X" * 500
        pairs = [(f"u{i}-{big}", f"r{i}-{big}") for i in range(20)]
        cid = _seed(store, None, pairs)

        with ConversationMemory(store, recent_turns=20, max_chars=1200) as mem:
            ctx = mem.assemble_context(cid)
            assert len(ctx) <= 1200                  # the hard bound holds
            assert "u19" in ctx                       # newest turn is kept
            assert "u0" not in ctx                     # oldest is dropped first

        # a single turn larger than the WHOLE budget is hard-truncated, never over
        with ConversationMemory(store, recent_turns=20, max_chars=100) as mem:
            ctx = mem.assemble_context(cid)
            assert len(ctx) <= 100


# --------------------------------------------------------------------------- #
# 3) STRICT cross-chat isolation (no thread reads another thread's turns)
# --------------------------------------------------------------------------- #
def test_cross_chat_isolation(tmp_path):
    with ChatStore(tmp_path) as store, ConversationMemory(store) as mem:
        chat_a = _seed(store, None, [("apple-q", "apple-a"), ("avocado-q", "avocado-a")])
        chat_b = _seed(store, None, [("banana-q", "banana-a"), ("berry-q", "berry-a")])
        assert chat_a != chat_b

        ctx_a = mem.assemble_context(chat_a)
        ctx_b = mem.assemble_context(chat_b)

        # each context has ONLY its own thread's content
        assert "apple-q" in ctx_a and "avocado-q" in ctx_a
        assert "banana" not in ctx_a and "berry" not in ctx_a
        assert "banana-q" in ctx_b and "berry-q" in ctx_b
        assert "apple" not in ctx_b and "avocado" not in ctx_b

        # summaries are isolated too
        mem.set_summary(chat_a, "summary-for-A-only")
        assert mem.get_summary(chat_a) == "summary-for-A-only"
        assert mem.get_summary(chat_b) is None
        assert "summary-for-A-only" in mem.assemble_context(chat_a)
        assert "summary-for-A-only" not in mem.assemble_context(chat_b)

        # an unknown chat never bleeds another's turns
        assert mem.recent_turns_for("chat-does-not-exist") == []
        assert mem.assemble_context("chat-does-not-exist") == ""


# --------------------------------------------------------------------------- #
# 4) restart-durable: a FRESH store + memory on the SAME db reads it all back
# --------------------------------------------------------------------------- #
def test_survives_fresh_store_open(tmp_path):
    # process #1: write turns + a running summary, then close everything
    with ChatStore(tmp_path) as store, ConversationMemory(store) as mem:
        cid = _seed(store, None, [("first-q", "first-a"), ("second-q", "second-a")])
        mem.set_summary(cid, "earlier we discussed onboarding")
        # no second database file was created — it's all the shared chat.db
        assert mem.db_path.name == "chat.db"

    # process #2: brand-new objects on the SAME data dir reload everything
    with ChatStore(tmp_path) as store2, ConversationMemory(store2) as mem2:
        turns = mem2.recent_turns_for(cid)
        assert [t.user_request for t in turns] == ["first-q", "second-q"]
        assert mem2.get_summary(cid) == "earlier we discussed onboarding"

        ctx = mem2.assemble_context(cid)
        assert "earlier we discussed onboarding" in ctx
        assert "first-q" in ctx and "second-q" in ctx


# --------------------------------------------------------------------------- #
# 5) the running summary is prepended and itself bounded
# --------------------------------------------------------------------------- #
def test_summary_prepended_and_bounded(tmp_path):
    with ChatStore(tmp_path) as store, ConversationMemory(store, max_chars=4000) as mem:
        cid = _seed(store, None, [("recent-q", "recent-a")])
        mem.set_summary(cid, "compact rolling summary of older turns")

        ctx = mem.assemble_context(cid)
        # summary header comes before the recent turn block
        assert SUMMARY_HEADER in ctx
        assert ctx.index(SUMMARY_HEADER) < ctx.index("recent-q")

        # excluding the summary drops it entirely
        no_sum = mem.assemble_context(cid, include_summary=False)
        assert SUMMARY_HEADER not in no_sum
        assert "recent-q" in no_sum

        # even a large summary cannot blow the budget
        with ConversationMemory(store, max_chars=200) as tight:
            tight.set_summary(cid, "S" * 5000)
            assert len(tight.assemble_context(cid)) <= 200
