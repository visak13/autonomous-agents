"""s5/a2 — conversation memory wired into the REAL routed chat plan.

This locks the three proofs the action's acceptance gate requires, all OFFLINE
(the d12 deterministic stub transport — NO live Gemma / Ollama / GPU), so the
suite runs in seconds:

1. PRIOR-TURN CONTEXT REACHES THE PLAN — a chat's second turn drives a plan whose
   GOAL (the string the shape selector + ``Planner.plan`` + every derived node all
   reason over) CONTAINS the first turn's context. Proved two ways:
     * at the unit seam (:func:`~chat_app.agentic.goal_with_context`), and
     * end-to-end through ``run_offline`` (the prior context lands in the
       planner-derived DAG's node tasks).

2. STRICT PER-CHAT ISOLATION — driving a SECOND chat never receives the first
   chat's context (one thread can never read another's turns). Proved through the
   REAL routed entrypoint (``POST /chats/{id}/message`` → ``_execute_message_run``)
   with a capturing spy on the run, over two interleaved chats.

3. d8 (the unattended-email safety invariant, BINDING) — the chat-driven node→tool
   offering exposes the recipient-hard-locked ``send_mail`` and NEVER the legacy
   free-``to`` ``send_email``, on the same hook the routed plan uses.
"""
from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from reactive_tools import (
    EventPlane,
    build_default_hook,
    register_agentic_tools,
)

import chat_app.routes as routes
from chat_app.app import build_wiring, create_app
from chat_app.agentic import (
    OFFERED_TOOLS,
    build_plan_schema,
    goal_with_context,
    run_offline,
)

_SECRET = "ZORPAX42"  # a distinctive token a memoryless run could never invent


# =========================================================================== #
# 1) the unit seam: goal_with_context prefixes prior context onto the goal
# =========================================================================== #


def test_goal_with_context_prefixes_prior_turns():
    ctx = f"User: my secret code is {_SECRET}\nAssistant: noted"
    goal = goal_with_context(ctx, "what was my secret code")
    # the prior context AND the current request both reach the planner's goal,
    # current request LAST (the model continues the thread, not re-answers it).
    assert _SECRET in goal
    assert "what was my secret code" in goal
    assert goal.index(_SECRET) < goal.index("what was my secret code")


def test_goal_with_context_blank_is_bare_query_no_regression():
    # the FIRST turn (and any caller that passes no context) is byte-identical to
    # the pre-a2 behaviour: the bare, stripped query — never an empty header block.
    bare = goal_with_context(None, "  draft a poem  ")
    assert bare == "draft a poem"
    assert goal_with_context("", "draft a poem") == "draft a poem"
    assert goal_with_context("   ", "draft a poem") == "draft a poem"


# =========================================================================== #
# 1b) end-to-end: the prior context lands in the planner-DERIVED DAG node tasks
# =========================================================================== #


def _dag_text(agentic) -> str:
    """All node tasks of the run's planner-derived DAG, concatenated."""
    assert agentic.dag is not None
    return " ".join(n.task for n in agentic.dag.nodes)


def test_run_offline_threads_prior_context_into_the_planner_dag(tmp_path):
    w = build_wiring(data_dir=tmp_path)
    try:
        ctx = f"User: my secret code is {_SECRET}\nAssistant: noted"
        # WITH prior context → the planner-derived DAG's node tasks carry it.
        with_ctx = asyncio.run(
            run_offline(
                "recall it",
                registry=w.registry,
                hook=w.hook,
                plane=EventPlane(),
                conversation_context=ctx,
            )
        )
        assert _SECRET in _dag_text(with_ctx)

        # WITHOUT prior context (a fresh thread) → the same query never invents it.
        no_ctx = asyncio.run(
            run_offline(
                "recall it",
                registry=w.registry,
                hook=w.hook,
                plane=EventPlane(),
            )
        )
        assert _SECRET not in _dag_text(no_ctx)
    finally:
        w.close()


# =========================================================================== #
# 2) the REAL routed entrypoint: 2nd turn sees the 1st turn; chats are isolated
# =========================================================================== #


def test_routed_chat_run_sees_prior_turn_and_isolates_other_chats(tmp_path, monkeypatch):
    """Drive the actual ``POST /chats/{id}/message`` route over two interleaved
    chats and capture the conversation context each run is invoked with.

    The spy DELEGATES to the real offline run so each turn is genuinely persisted
    (``save_turn``) — which is what makes the NEXT turn's assembled context carry
    the prior turn. This exercises the s5/a2 wiring exactly as production does:
    ``_execute_message_run`` builds the per-chat context from the shared
    :class:`ConversationMemory` and threads it into the run."""
    monkeypatch.delenv("REACTIVE_AGENTS_LIVE", raising=False)
    app = create_app(data_dir=tmp_path)

    # (chat_id, conversation_context) recorded per run, in call order.
    captured: list[tuple[str, str | None]] = []
    real_run_offline = routes.run_offline

    async def _spy(query, **kwargs):
        # The route does not pass chat_id to run_offline; recover it from the plane
        # is overkill — instead tag by the captured order + the message content.
        captured.append((query, kwargs.get("conversation_context")))
        return await real_run_offline(query, **kwargs)

    monkeypatch.setattr(routes, "run_offline", _spy)

    with TestClient(app) as client:
        a = client.post("/chats", json={}).json()
        chat_a = a.get("chat_id") or a.get("id")
        b = client.post("/chats", json={}).json()
        chat_b = b.get("chat_id") or b.get("id")
        assert chat_a and chat_b and chat_a != chat_b

        # turn 1 on each chat (no prior turns yet → empty/None context).
        assert client.post(
            f"/chats/{chat_a}/message", json={"message": f"my secret code is {_SECRET}"}
        ).status_code == 200
        assert client.post(
            f"/chats/{chat_b}/message", json={"message": "hello there friend"}
        ).status_code == 200

        # turn 2 on each chat (now there IS a prior turn to assemble).
        assert client.post(
            f"/chats/{chat_a}/message", json={"message": "what was my secret code"}
        ).status_code == 200
        assert client.post(
            f"/chats/{chat_b}/message", json={"message": "say it again"}
        ).status_code == 200

    # four runs captured, in order: A1, B1, A2, B2.
    assert len(captured) == 4
    (a1_q, a1_ctx), (b1_q, b1_ctx), (a2_q, a2_ctx), (b2_q, b2_ctx) = captured

    # turn 1 of each chat had NO prior turns → no prior-turn context.
    assert not (a1_ctx or "").strip()
    assert not (b1_ctx or "").strip()

    # turn 2 of chat A CONTAINS chat A's first turn (the secret).
    assert a2_ctx and _SECRET in a2_ctx

    # ISOLATION: turn 2 of chat B carries chat B's OWN first turn, and NEVER chat
    # A's secret — one thread can never read another thread's turns.
    assert b2_ctx and "hello there friend" in b2_ctx
    assert _SECRET not in (b2_ctx or "")


# =========================================================================== #
# 3) d8 — the chat-driven offering exposes locked send_mail, not free-`to` email
# =========================================================================== #

_SIX_S2_TOOLS = {
    "web_search",
    "web_fetch",
    "file_read",
    "file_write",
    "send_mail",
    "cron_add",
    "cron_list",
    "cron_delete",
}


def test_chat_tool_offering_exposes_locked_send_mail_not_send_email(tmp_path):
    """The node→tool surface the chat-driven plan offers (the same hook
    ``build_wiring`` composes) selects the recipient-locked ``send_mail`` and never
    the legacy free-``to`` ``send_email`` (d8, BINDING)."""
    hook = build_default_hook(EventPlane(), file_base=tmp_path)
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)

    names = set(hook.registry.names())
    # the locked mail tool is registered; the legacy free-`to` tool is NOT.
    assert "send_mail" in names
    assert "send_email" not in names

    offered = [t["name"] for t in hook.registry.catalog() if t["name"] in OFFERED_TOOLS]
    schema = build_plan_schema(["spec-a"], offered)
    tool_enum = schema["properties"]["nodes"]["items"]["properties"]["tool"]["enum"]
    # the planner's structured-output tool enum can select send_mail, never send_email.
    assert "send_mail" in tool_enum
    assert "send_email" not in tool_enum
    assert "send_email" not in OFFERED_TOOLS
    # and the full six s2 buckets remain offered (no regression of the b5 surface).
    assert _SIX_S2_TOOLS <= set(tool_enum)


def test_wiring_hook_never_registers_send_email(tmp_path):
    """The SAME hook the running wiring builds never carries the legacy tool."""
    w = build_wiring(data_dir=tmp_path)
    try:
        assert "send_mail" in set(w.hook.registry.names())
        assert "send_email" not in set(w.hook.registry.names())
    finally:
        w.close()


# =========================================================================== #
# 4) a6 (s7) — a missing-specialist RESUME carries the conversation memory
# =========================================================================== #
# The initial _run_acyclic threads the bounded prior-turn context into the
# runtime so every node's sub-agent grounds in the thread (the s5/a4 fix). A
# missing-specialist RESUME re-drives the SAME paused nodes, so it must carry the
# SAME context or the resumed plan runs MEMORYLESS at the node level (the a6
# gap). Scenario 3's missing-specialist resolution — SSE-fallback OR
# define-and-resume — flows through resume_agentic, so this is the path the gap
# affected. Both proofs are OFFLINE (the deterministic stub transport).


def test_resume_agentic_threads_prior_context_into_runtime(tmp_path, monkeypatch):
    """resume_agentic threads ``conversation_context`` into the runtime it builds.

    Spies on the SHARED ``_build_acyclic_runtime`` the initial run AND the resume
    use, drives a missing-specialist resume on the stub transport, and asserts the
    resume handed the runtime the SAME prior-turn context (the secret) — proving
    the resumed nodes are no longer memoryless (the a6 fix in agentic.py)."""
    import chat_app.agentic as agentic_mod
    from agent_runtime import PlanDAG, PlanNode, stub

    captured: dict[str, str | None] = {}
    real_build = agentic_mod._build_acyclic_runtime

    def _spy(**kwargs):
        captured["ctx"] = kwargs.get("conversation_context")
        return real_build(**kwargs)

    monkeypatch.setattr(agentic_mod, "_build_acyclic_runtime", _spy)

    w = build_wiring(data_dir=tmp_path)
    try:
        # a paused DAG: one node DECLARED a needed specialist no spec provides.
        dag = PlanDAG(
            nodes=[PlanNode(id="n1", task="write the report", needs_spec="a markdown specialist")]
        )
        ctx = f"User: my secret code is {_SECRET}\nAssistant: noted"
        res = asyncio.run(
            agentic_mod.resume_agentic(
                dag,
                "sse_fallback",
                transport=stub.subagent_transport(),
                registry=w.registry,
                hook=w.hook,
                plane=EventPlane(),
                missing=[{"node_id": "n1", "task": "write the report", "needs": "a markdown specialist"}],
                conversation_context=ctx,
            )
        )
        # the resume path threaded the SAME prior-turn context into the runtime.
        assert captured.get("ctx") == ctx
        assert _SECRET in (captured.get("ctx") or "")
        # and the run actually drove the (now spec-less) node — a real resume.
        assert res.result is not None

        # CONTRAST (the pre-fix behaviour): a resume that passes NO context hands
        # the runtime None — so the assertion above is meaningful, not vacuous.
        captured.clear()
        asyncio.run(
            agentic_mod.resume_agentic(
                PlanDAG(nodes=[PlanNode(id="n1", task="write the report", needs_spec="x")]),
                "sse_fallback",
                transport=stub.subagent_transport(),
                registry=w.registry,
                hook=w.hook,
                plane=EventPlane(),
                missing=[{"node_id": "n1", "task": "write the report", "needs": "x"}],
            )
        )
        assert not (captured.get("ctx") or "")
    finally:
        w.close()


def test_routed_resume_carries_prior_context_end_to_end(tmp_path, monkeypatch):
    """End-to-end through the REAL routes: a paused turn STASHES the assembled
    prior-turn context in ``pending_runs`` and ``POST /chats/{id}/resume`` FORWARDS
    it to ``resume_agentic`` — so the resumed run is not memoryless (covers the a6
    fix in routes.py: stash + forward).

    A real missing-specialist pause cannot arise on the canned offline plan, so
    the 2nd turn's offline run is replaced with a PAUSE result; the rest of the
    route (assemble context → stash → resume → forward) runs exactly as
    production does."""
    from agent_runtime import MISSING_SPEC_CHOICES, PlanDAG, PlanNode
    from chat_app.agentic import AgenticResult

    monkeypatch.delenv("REACTIVE_AGENTS_LIVE", raising=False)
    app = create_app(data_dir=tmp_path)
    real_run_offline = routes.run_offline
    pause_token = "resume-test-a6"

    async def _maybe_pausing_run(query, **kwargs):
        ctx = kwargs.get("conversation_context") or ""
        if _SECRET in ctx:
            # turn 2 (its prior-turn context carries the secret) → PAUSE, so the
            # route stashes pending_runs with this very assembled context.
            dag = PlanDAG(nodes=[PlanNode(id="n1", task="report", needs_spec="x")])
            return AgenticResult(
                dag=dag,
                shape=None,
                ok=False,
                missing_specialist=True,
                pending={
                    "resume_token": pause_token,
                    "missing": [{"node_id": "n1", "task": "report", "needs": "x"}],
                    "choices": list(MISSING_SPEC_CHOICES),
                },
            )
        return await real_run_offline(query, **kwargs)

    monkeypatch.setattr(routes, "run_offline", _maybe_pausing_run)

    captured: dict[str, str | None] = {}

    async def _resume_spy(dag, choice, **kwargs):
        captured["ctx"] = kwargs.get("conversation_context")
        return AgenticResult(dag=dag, ok=True, final_response="resumed ok")

    monkeypatch.setattr(routes, "resume_agentic", _resume_spy)

    with TestClient(app) as client:
        c = client.post("/chats", json={}).json()
        chat = c.get("chat_id") or c.get("id")
        assert client.post(
            f"/chats/{chat}/message", json={"message": f"my secret code is {_SECRET}"}
        ).status_code == 200
        # turn 2 → pauses; the route stashes pending_runs with the prior context.
        r2 = client.post(
            f"/chats/{chat}/message", json={"message": "what was my secret code"}
        )
        assert r2.status_code == 200 and r2.json()["missing_specialist"] is True
        # the STASH carried the assembled prior-turn context (the secret).
        parked = app.state.pending_runs[pause_token]
        assert parked.get("conversation_context") and _SECRET in parked["conversation_context"]
        # resume → the route FORWARDS that context to resume_agentic.
        rr = client.post(
            f"/chats/{chat}/resume",
            json={"resume_token": pause_token, "choice": "sse_fallback"},
        )
        assert rr.status_code == 200
        assert captured.get("ctx") and _SECRET in captured["ctx"]
