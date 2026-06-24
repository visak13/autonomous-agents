"""The UNIVERSAL agent identity prepended to every LLM call (d11 / s3-a3).

Like Claude Code's system prompt, ONE short capable-agent persona rides on EVERY
call the pipeline makes — the planner (plan / replan / ambiguity / heal), the
incremental authorer, and every runtime node (workers, role nodes, the chat
answer node, the inline reviewer). It gives the whole system a consistent,
capable-agent-first, reason-then-conform, memory-aware posture WITHOUT being
coding instructions (coding craft lives in the planner prompts + the compiled
specs) and WITHOUT re-litigating per-call behaviour.

Kept deliberately SHORT (~90 tokens): it is a constant added to every call's
token budget, so it must not eat into the load-bearing ``num_predict`` headroom
the think=True structured calls rely on (d6/d7). The two universal output rules
it folds in (ground-don't-hallucinate; when asked for JSON the visible reply is
ONLY the JSON) let the individual structured prompts DROP their own repeated
"reason privately… no code fences" tails — a net token saving.

This module imports ONLY the lower-layer ``llm_framework.transport`` (a legal
downward dependency, no cycle), from which it RE-EXPORTS the single canonical
:data:`AGENT_IDENTITY`. That single-source-of-truth is load-bearing (s3/a6 review
fix): the transport seam injects the SAME constant on every model call, and its
``_inject_identity`` idempotency guard recognises a system turn this module's
:func:`with_identity` already folded the identity into — so the identity is shipped
EXACTLY ONCE, never doubled. Defining a second (divergent) copy here is what caused
the double-injection bug, so we deliberately do not.
"""
from __future__ import annotations

from typing import Optional

# The ONE canonical universal persona, defined in (and re-exported from) the shared
# transport seam so the call-site pre-fold below and the transport-seam injection
# use byte-identical text. Capable-agent-first (act, don't interrogate), grounded
# (no invented facts/sources), memory-aware (use the conversation), JSON-clean
# (a structured call's visible reply is only its JSON). ~90 tokens.
from llm_framework.transport import AGENT_IDENTITY


def with_identity(system: Optional[str]) -> Optional[str]:
    """Prepend :data:`AGENT_IDENTITY` to a system turn (the universal-identity seam).

    ``None``/empty system → the identity alone becomes the system turn (so even a
    bare, spec-less node now carries the identity, d11). A non-empty system is
    joined below the identity with a blank line so the persona always leads."""
    if not system:
        return AGENT_IDENTITY
    return f"{AGENT_IDENTITY}\n\n{system}"


__all__ = ["AGENT_IDENTITY", "with_identity"]
