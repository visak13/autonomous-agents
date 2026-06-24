"""Built-in chain stages.

Every stage here is a small **factory** returning a ``(ctx) -> ctx`` callable —
exactly the shape :class:`~llm_framework.chain.Chain.use` accepts, so a built-in
stage and a bare lambda compose identically. The factories let you bind a
transport / options at wire-up time while keeping the runtime call signature
uniform.

Shipped stages
--------------
- :func:`prompt_assembly`  — compose ``system`` + ``history`` + ``user`` into
  ``ctx.messages`` (the transport-ready list).
- :func:`call_stage`       — invoke the injected transport on ``ctx.messages``
  and store the raw assistant text on ``ctx.raw_output``.
- :func:`structured_output`— extract+parse JSON from the raw text; on malformed
  output, run a BOUNDED repair loop (re-prompt the transport, re-parse) capped
  at ``max_repair_attempts``.

Documented no-op SEAM stages (pass-through today, filled by later steps so the
chain *shape* is stable now):
- :func:`tool_hook`        — seam for s3 (reactive / tool calls).
- :func:`memory_injection` — seam for s4 (memory recall).

All stages are independently testable: each factory returns a plain callable
you can run against a hand-built :class:`Context`.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from .chain import Chain, Context, Stage
from .transport import Message, Transport, TransportError


# --------------------------------------------------------------------------- #
# 1. prompt assembly
# --------------------------------------------------------------------------- #


def prompt_assembly() -> Stage:
    """Compose ``system`` + ``history`` + ``user`` into ``ctx.messages``.

    The result is a flat transport-ready list: an optional ``system`` message,
    then the prior ``history`` turns verbatim, then the current ``user`` turn.
    Kept minimal (d10) — no templating, no role rewriting; that stays the
    caller's / a later stage's concern.
    """

    def prompt_assembly(ctx: Context) -> Context:
        messages: list[Message] = []
        if ctx.system:
            messages.append({"role": "system", "content": ctx.system})
        messages.extend(ctx.history)
        if ctx.user is not None:
            messages.append({"role": "user", "content": ctx.user})
        ctx.messages = messages
        return ctx

    return prompt_assembly


# --------------------------------------------------------------------------- #
# 2. model call
# --------------------------------------------------------------------------- #


def call_stage(transport: Optional[Transport] = None, **call_opts: Any) -> Stage:
    """Invoke the transport on ``ctx.messages`` and store its text.

    The transport is resolved at run time as ``transport or ctx.transport`` so a
    chain can be wired either with the transport bound here
    (``call_stage(my_transport)``) or carried on the context
    (``Context(transport=my_transport)``). ``call_opts`` (temperature, json,
    keep_alive, …) are forwarded to ``transport.chat`` on every call.

    Writes ``ctx.raw_output`` (assistant text) and records the call on
    ``ctx.meta['calls']``.
    """

    def call_stage(ctx: Context) -> Context:
        tp = transport or ctx.transport
        if tp is None:
            raise TransportError(
                "call_stage has no transport: pass one to call_stage(...) or set "
                "Context.transport"
            )
        if not ctx.messages:
            raise ValueError(
                "call_stage ran with empty ctx.messages — put prompt_assembly "
                "(or another message-producing stage) before it"
            )
        result = tp.chat(ctx.messages, **call_opts)
        ctx.raw_output = result.content
        # s13: surface the model's NATIVE tool calls (when the call passed ``tools=``)
        # so a ReAct loop can dispatch them directly instead of string-parsing prose.
        ctx.tool_calls = getattr(result, "tool_calls", None)
        ctx.meta.setdefault("calls", []).append(
            {
                "messages": list(ctx.messages),
                "opts": dict(call_opts),
                "output": result.content,
            }
        )
        return ctx

    return call_stage


# --------------------------------------------------------------------------- #
# 3. structured output: parse + bounded repair
# --------------------------------------------------------------------------- #

_DEFAULT_REPAIR_PROMPT = (
    "Your previous reply was not valid JSON. Reply with ONLY the corrected, "
    "strictly-valid JSON value and nothing else — no prose, no code fences."
)


def _strip_fences(text: str) -> str:
    """Drop a leading/trailing Markdown code fence if the model wrapped the JSON."""
    s = text.strip()
    if s.startswith("```"):
        # remove the opening fence line (``` or ```json) and the closing fence
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _extract_json(text: str) -> Optional[str]:
    """Return the first balanced JSON object/array substring, or None.

    Scans for the first ``{`` or ``[`` and walks forward honouring string
    literals + escapes until the matching close. This survives the common
    case of a model emitting prose around the JSON.
    """
    if not text:
        return None
    s = _strip_fences(text)
    start = -1
    opener = closer = ""
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            opener = ch
            closer = "}" if ch == "{" else "]"
            break
    if start == -1:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(s)):
        ch = s[i]
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
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None  # unbalanced (truncated JSON) -> treat as malformed


def _try_parse(text: Optional[str]) -> tuple[Any, Optional[str]]:
    """Best-effort parse: returns ``(value, None)`` or ``(None, error_message)``."""
    if text is None:
        return None, "no output to parse"
    candidate = _extract_json(text)
    if candidate is None:
        return None, "no JSON object/array found in output"
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"


def structured_output(
    transport: Optional[Transport] = None,
    *,
    max_repair_attempts: int = 2,
    repair_prompt: str = _DEFAULT_REPAIR_PROMPT,
    **repair_opts: Any,
) -> Stage:
    """Parse JSON from ``ctx.raw_output``; on malformed output, repair.

    1. Try to extract+parse JSON from ``ctx.raw_output``.
    2. If that fails, run a BOUNDED repair loop (at most ``max_repair_attempts``
       iterations): re-prompt the transport — feeding back the bad output and a
       fix instruction — take its new reply, and re-parse. ``repair_opts``
       (e.g. ``json=True``) are forwarded to the repair call so the transport
       can be nudged into JSON mode.

    On success ``ctx.structured`` holds the parsed value and ``ctx.raw_output``
    is updated to the text that parsed. On exhaustion ``ctx.structured`` stays
    ``None`` (this is context hygiene, NOT a hard pass/fail gate — d4); the full
    attempt trail is recorded on ``ctx.meta['structured_output']`` so the demo
    and callers can see exactly what happened.
    """

    def structured_output(ctx: Context) -> Context:
        tp = transport or ctx.transport
        text = ctx.raw_output
        value, error = _try_parse(text)

        record: dict[str, Any] = {
            "parsed_on_first_try": value is not None,
            "attempts": [],
            "repaired": False,
        }
        if value is not None:
            record["initial_error"] = None
        else:
            record["initial_error"] = error

        attempt = 0
        while value is None and attempt < max_repair_attempts and tp is not None:
            attempt += 1
            repair_messages: list[Message] = [
                {
                    "role": "system",
                    "content": repair_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        f"The following reply was not valid JSON:\n\n{text}\n\n"
                        "Return the corrected JSON only."
                    ),
                },
            ]
            try:
                text = tp.complete(repair_messages, **repair_opts)
            except Exception as exc:  # transport failure mid-repair: stop bounded
                record["attempts"].append({"attempt": attempt, "error": f"transport: {exc}"})
                break
            value, error = _try_parse(text)
            record["attempts"].append(
                {
                    "attempt": attempt,
                    "output": text,
                    "ok": value is not None,
                    "error": error,
                }
            )
            if value is not None:
                record["repaired"] = True

        if value is not None:
            ctx.structured = value
            ctx.raw_output = text  # the text that actually parsed
        else:
            ctx.structured = None
            record["final_error"] = error

        ctx.meta["structured_output"] = record
        return ctx

    return structured_output


# --------------------------------------------------------------------------- #
# 4. SEAM stages — documented no-ops, stable shape for s3 / s4
# --------------------------------------------------------------------------- #


def tool_hook() -> Stage:
    """SEAM for the reactive / tool layer (s3) — a documented NO-OP today.

    The chain shape is fixed now so later steps splice behaviour in WITHOUT
    reshaping the pipeline. When s3 lands, this stage will: inspect
    ``ctx.raw_output`` / ``ctx.structured`` for a tool-call request, dispatch
    the in-process tool (d2: in-process RxPY/asyncio, NOT shell forking), write
    the result back onto the ctx, and let the chain loop the model call. It sits
    AFTER the model call by convention (``insert_after('call_stage', ...)``).

    Until then it passes the context through untouched.
    """

    def tool_hook(ctx: Context) -> Context:
        return ctx

    return tool_hook


def memory_injection() -> Stage:
    """SEAM for the memory layer (s4) — a documented NO-OP today.

    When s4 lands, this stage will recall relevant durable facts + compaction
    summaries (in-process CPU MiniLM + sqlite-vec, hybrid BM25+dense RRF — d3)
    and inject ONLY the scoped-minimal slice into ``ctx.system`` / ``ctx.history``
    so phi's small window stays lean (d10 context-scoping: a launched sub-agent
    sees only the task at hand + its memory, never heavy phased prompts). It
    sits BEFORE ``prompt_assembly`` by convention so injected context is folded
    into the assembled messages.

    Until then it passes the context through untouched.
    """

    def memory_injection(ctx: Context) -> Context:
        return ctx

    return memory_injection


# --------------------------------------------------------------------------- #
# Convenience: the canonical chain shape
# --------------------------------------------------------------------------- #


def build_default_chain(
    transport: Optional[Transport] = None,
    *,
    structured: bool = False,
    call_opts: Optional[dict[str, Any]] = None,
    max_repair_attempts: int = 2,
) -> Chain:
    """Wire the canonical pipeline with all the seams in their stable slots.

    Shape (d10 — fixed now, behaviour filled later):

        memory_injection(no-op s4) -> prompt_assembly -> tool_hook(no-op s3)
        -> call_stage -> [structured_output]

    ``structured=True`` appends the parse/repair stage. Seams are present as
    no-ops so s3/s4 only need ``insert_*`` / swap, never a reshape.
    """
    chain = Chain()
    chain.use(memory_injection())
    chain.use(prompt_assembly())
    chain.use(tool_hook())
    chain.use(call_stage(transport, **(call_opts or {})))
    if structured:
        chain.use(
            structured_output(transport, max_repair_attempts=max_repair_attempts)
        )
    return chain
