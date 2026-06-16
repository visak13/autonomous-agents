"""Default per-node VERIFY GATE(s) for the runtime lifecycle (verifiable → done).

The d2 lifecycle mandates that EVERY node path traverse the
``in_progress → verifiable → done`` gate with the CODER=REVIEWER reviewer
ENCOURAGED to read the node's output and FIX IT INLINE rather than merely
pass/fail. :class:`~agent_runtime.runtime.AgentRuntime` always runs the gate, but
when no ``verifier`` is injected the gate trivially passes — so the inline
reviewer-fix never gets a chance to fire. That was the gap on the chat node paths
(``run_offline`` wired neither verifier nor validator; ``_run_acyclic`` wired only
the empty-output validator): a degenerate node output was silently accepted, with
the reviewer never invited to repair it.

:func:`default_node_verifier` is the reusable, model-INDEPENDENT safety-net gate
wired onto every node path so a degenerate output is caught and handed to the
same-spec inline reviewer (``SubAgent.review_and_fix``) instead of passing
silently. It is deliberately CONSERVATIVE — it rejects only output that is clearly
unusable — so it never spuriously rejects a real answer on a live run:

* ANY node — a node must produce some usable output. Empty / whitespace-only, or a
  bare degenerate token (``null`` / ``none`` / ``n/a`` / ``{}`` / ``[]``) is
  rejected so the reviewer is asked to produce real content.
* A JUDGMENT node (one whose ``role`` carries an enum verdict —
  ``critic`` / ``synthesis`` / ``verify`` / ``reviewer``, see
  :data:`~agent_runtime.roles.ROLE_VERDICTS`) must additionally carry a verdict
  drawn from that role's legal enum. A missing / null / out-of-enum verdict is
  rejected with the legal set named in the reason, so the gate NEVER silently
  passes a null verdict (the s3/a2 measured caveat: a verbose small model overruns
  ``num_predict`` and truncates the verdict JSON). The reviewer re-emits a valid
  verdict inline.

Pure data + a best-effort JSON probe — no model call, no I/O — so it stays
trivially testable and import-light.
"""
from __future__ import annotations

import json
from typing import Any, Optional, Tuple

from .roles import ROLE_VERDICTS

# A bare token that is technically "output" but carries no usable content — the
# reviewer should be asked to produce a real answer rather than this being passed.
_DEGENERATE_TOKENS = {"", "null", "none", "n/a", "na", "nil", "{}", "[]"}


def _first_json_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` substring of ``text``, or ``None``.

    A tiny, dependency-free scan (string-literal + escape aware) so a verdict can
    be read out of output that wraps the JSON in prose or a code fence."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # unbalanced (truncated) → unreadable


def _extract_verdict(output: str) -> Optional[str]:
    """Best-effort: pull a ``verdict`` value out of a node's output text.

    The output may be raw JSON, JSON wrapped in prose / a code fence, or a plain
    string. Returns the verdict string if one can be read, else ``None`` (which
    the judgment gate treats as a missing verdict to repair)."""
    candidate = _first_json_object((output or "").strip())
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(obj, dict) and obj.get("verdict") is not None:
        return str(obj.get("verdict")).strip()
    return None


def default_node_verifier(node: Any, result: Any) -> Tuple[bool, Optional[str]]:
    """The default safety-net gate: reject only clearly-unusable node output.

    Returns ``(ok, reason)`` — the :data:`~agent_runtime.runtime.NodeVerifier`
    contract. ``ok=False`` hands ``reason`` to the inline CODER=REVIEWER fix; a
    real answer always passes (conservative by design)."""
    out = (getattr(result, "output", "") or "").strip()
    if out.lower() in _DEGENERATE_TOKENS:
        return False, "node produced no usable output — produce the real deliverable"

    role = getattr(node, "role", None)
    if role in ROLE_VERDICTS:
        legal = ROLE_VERDICTS[role]
        verdict = _extract_verdict(out)
        if verdict not in legal:
            return (
                False,
                f"{role} verdict {verdict!r} is missing or not one of {list(legal)}; "
                "re-emit the structured output with a valid verdict",
            )
    return True, None


__all__ = ["default_node_verifier"]
