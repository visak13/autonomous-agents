"""Ambiguity-clarification pause surface (scenario-2 clarification turn).

When the planner judges a user request too UNDERSPECIFIED to act on without
guessing a load-bearing detail (see :meth:`agent_runtime.Planner.assess_ambiguity`),
the live chat path must ASK the user a clarifying question BACK and resume on the
clarified intent — instead of silently picking values. This is the direct analogue
of the missing-specialist pause (:mod:`agent_runtime.missing_spec`): a model
DECISION (made by the planner) plus a small, pure NOTIFY payload + event kind that
the chat layer streams over SSE and resumes from.

Kept here — pure data, no model call and no HTTP — for the same reason
``missing_spec`` is: the engine owns the mechanic and it stays trivially testable.
The MODEL judgement lives on the planner; the PAUSE orchestration (publish the
event, stash the original request, re-run on the answer) lives in
:mod:`chat_app.agentic` / the chat routes, exactly mirroring the missing-spec flow.
"""
from __future__ import annotations

from typing import Any

# The lifecycle event published on the in-process plane when a run pauses to ask
# the user a clarifying question. The chat's SSE stream relays it (it is added to
# the streamed kinds) so the user is NOTIFIED live and shown the question — it is
# never a silent guess.
EVENT_NEEDS_CLARIFICATION = "agent_needs_clarification"

# The pending payload's ``kind`` discriminator (the chat stashes paused runs of
# several kinds keyed by resume_token; this distinguishes a clarification pause
# from a missing-specialist pause).
CLARIFICATION_KIND = "clarification"


def clarification_payload(
    question: str,
    *,
    resume_token: str,
    original_query: str,
) -> dict[str, Any]:
    """Build the :data:`EVENT_NEEDS_CLARIFICATION` payload (notify + the question).

    Carries the planner's ONE clarifying ``question``, the opaque ``resume_token``
    the client echoes back with its answer, the ``original_query`` the answer
    refines, and a ``kind`` discriminator so the resume route can tell a
    clarification pause apart from a missing-specialist pause. The notify the user
    sees IS actionable (answer the question), not a dead-end error."""
    return {
        "kind": CLARIFICATION_KIND,
        "resume_token": resume_token,
        "question": question,
        "original_query": original_query,
    }


__all__ = [
    "EVENT_NEEDS_CLARIFICATION",
    "CLARIFICATION_KIND",
    "clarification_payload",
]
