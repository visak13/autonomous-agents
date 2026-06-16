"""The COMPILER: condense a RawDefinition + research trace into a CompiledSpec.

This is the distillation step of the specialization lifecycle (d8): once a
specialization has been DEFINED (:class:`~specialization.model.RawDefinition`)
and RESEARCHED (:class:`~specialization.research.ResearchTrace`), the compiler
condenses those into the tight prompt/ruleset BODY a launched sub-agent loads as
its whole grounding — a :class:`~specialization.model.CompiledSpec`.

How the condense runs (and why it is STUBBABLE)
-----------------------------------------------
The condense is driven through the **real** ``llm_framework`` chain — the same
``build_default_chain`` -> ``prompt_assembly`` -> ``call_stage`` spine the agent
runtime uses — so the compile flow is exercised exactly as it will run live, not
through a bespoke side path. The ONLY swappable seam is the transport:

- **OFFLINE (default, d7 "build-first-test-LLM-later")** — when no transport is
  injected, the compiler wires a :class:`~llm_framework.transport.FakeTransport`
  scripted with a *deterministic* condensation mechanically distilled from the
  trace (:func:`offline_condense_body`). The chain genuinely runs
  (prompt_assembly composes the messages, call_stage invokes the transport and
  writes ``raw_output``), so the whole compile path is fully evidenced WITHOUT
  touching the shared GPU. This is a real offline fallback, not a mock of phi.
- **LIVE (DEFERRED)** — a caller may inject a real
  :class:`~llm_framework.transport.OllamaTransport` (phi4-mini) to have the model
  author the body. That live-inference condense is **DEFERRED** here (the GPU is
  shared — d7): it is reachable by construction (same code path, different
  transport) but is NOT exercised/faked in this step's proof. It is left for a
  later live-inference run; we never fabricate a phi "reply" and call it live.

Nothing here is async — the chain is synchronous. Research (which IS async)
already produced the trace; the compiler just distills it.
"""
from __future__ import annotations

from typing import Optional

from llm_framework import (
    Context,
    FakeTransport,
    Transport,
    build_default_chain,
)

from specialization.model import CompiledSpec, RawDefinition
from specialization.research import ResearchTrace

# How many distilled "how" notes to fold into the offline body. A condensation,
# not a dump — keep the sub-agent's grounding lean (d10).
DEFAULT_MAX_NOTES = 6
DEFAULT_MAX_SOURCES = 5

# Marker line stamped into an OFFLINE-condensed body so a reader (and the s8
# demo) can tell at a glance the body was distilled deterministically and that
# the live phi4-mini condense is the deferred upgrade — not a faked live reply.
OFFLINE_MARKER = (
    "<!-- condensed offline (deterministic distillation); "
    "live phi4-mini condense DEFERRED (d7 shared-GPU) -->"
)


def strip_code_fence(text: str) -> str:
    """Drop a single OUTER markdown code fence the model often wraps a body in.

    A small model asked for "markdown" frequently returns the whole body inside a
    ```` ```markdown … ``` ```` fence; that fence is noise once the body is composed
    into a sub-agent's system prompt. Strips ONLY a leading fence line (``` or
    ```markdown/```md) plus a matching trailing fence, and only when the body
    actually starts with one — so a body that legitimately contains an inner code
    block (not wrapping the whole thing) is left untouched."""
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    first = lines[0].strip().lstrip("`").strip().lower()
    # Only treat as an OUTER wrapper when the opening fence has no/markdown info
    # string (```/```markdown/```md) — never a ```python etc. block that is content.
    if first and first not in {"markdown", "md", "text"}:
        return s
    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body).strip()


# --------------------------------------------------------------------------- #
# Prompt construction (what we'd send phi; what the FakeTransport sees)
# --------------------------------------------------------------------------- #
def build_condense_messages(raw: RawDefinition, trace: ResearchTrace) -> tuple[str, str]:
    """Build the ``(system, user)`` pair the condense chain runs on.

    The system prompt frames the task (distil research into a sub-agent ruleset);
    the user message carries the raw definition + the research notes/sources. The
    same messages are what a LIVE phi call would receive — so the offline and
    live paths are prompt-identical, only the transport differs."""
    system = (
        "You are compiling a SPECIALIST for a small autonomous agent. Condense the "
        "research below into a tight, self-contained ruleset the specialist loads as "
        "its WHOLE grounding: its mission, the distilled 'how', and concrete dos/"
        "don'ts. Be lean (a small-context model loads this) — no preamble, no "
        "restating the question. Output the ruleset body only, as raw markdown — "
        "do NOT wrap the whole body in a ``` code fence."
    )
    lines: list[str] = [
        f"SPECIALIST: {raw.name}",
        f"DESCRIPTION: {raw.description}",
        f"INTENT: {raw.intent}",
        "",
        "RESEARCH NOTES (distilled 'how'):",
    ]
    for note in trace.notes[:DEFAULT_MAX_NOTES]:
        title = f" [{note.title}]" if note.title else ""
        lines.append(f"- ({note.kind}{title}) {note.how}")
    fetched = [s for s in trace.sources if getattr(s, "fetched", False)]
    if fetched:
        lines.append("")
        lines.append("SOURCES READ:")
        for s in fetched[:DEFAULT_MAX_SOURCES]:
            lines.append(f"- {s.title or s.url} <{s.url}>")
    return system, "\n".join(lines)


# --------------------------------------------------------------------------- #
# The OFFLINE deterministic condensation (the FakeTransport's scripted reply)
# --------------------------------------------------------------------------- #
def offline_condense_body(raw: RawDefinition, trace: ResearchTrace) -> str:
    """Deterministically distil ``raw`` + ``trace`` into a ruleset body (offline).

    Pure, no LLM, no network — a mechanical condensation of exactly what the
    research surfaced. This is the body the default :class:`FakeTransport`
    returns, so the condense chain has a real, deterministic reply to write to
    ``ctx.raw_output`` (d7: full compile flow exercised with zero GPU)."""
    parts: list[str] = [
        f"# Specialist: {raw.name}",
        "",
        f"**Mission.** {raw.intent.strip() or raw.description.strip()}",
        "",
        "## How (distilled from research)",
    ]
    notes = trace.notes[:DEFAULT_MAX_NOTES]
    if notes:
        for note in notes:
            parts.append(f"- {note.how.strip()}")
    else:
        # No research notes (e.g. a search-only/empty trace) — still produce a
        # usable, honest body grounded in the definition itself.
        parts.append(f"- Apply best practices for: {raw.description.strip() or raw.name}.")

    fetched = [s for s in trace.sources if getattr(s, "fetched", False)]
    if fetched:
        parts.append("")
        parts.append("## Sources")
        for s in fetched[:DEFAULT_MAX_SOURCES]:
            parts.append(f"- {s.title or s.url} (<{s.url}>)")

    parts.append("")
    parts.append(OFFLINE_MARKER)
    return "\n".join(parts).strip()


def default_condense_transport(raw: RawDefinition, trace: ResearchTrace) -> FakeTransport:
    """A :class:`FakeTransport` scripted to return the offline-condensed body.

    Wiring the chain with this transport runs the WHOLE condense pipeline
    (prompt_assembly -> call_stage) end-to-end offline: the stub's single reply
    is the deterministic distillation, so ``ctx.raw_output`` is populated for
    real without any live inference."""
    return FakeTransport([offline_condense_body(raw, trace)])


# --------------------------------------------------------------------------- #
# The condense: drive the real chain, return the body
# --------------------------------------------------------------------------- #
def condense_body(
    raw: RawDefinition,
    trace: ResearchTrace,
    *,
    transport: Optional[Transport] = None,
) -> str:
    """Run the condense CHAIN and return the distilled ruleset BODY (a string).

    Drives the canonical ``llm_framework`` chain
    (``build_default_chain(transport)``): ``prompt_assembly`` composes the
    ``(system, user)`` messages and ``call_stage`` invokes the transport, whose
    text becomes the body.

    - ``transport=None`` (default) → OFFLINE: a deterministic
      :func:`default_condense_transport` is wired in, so the full chain runs with
      zero GPU and a reproducible body (d7).
    - ``transport=<OllamaTransport>`` → LIVE phi4-mini authoring. Reachable by
      construction (same chain), DEFERRED for a later live-inference run.
    """
    tp = transport or default_condense_transport(raw, trace)
    system, user = build_condense_messages(raw, trace)
    chain = build_default_chain(tp)
    ctx = chain.run(Context(system=system, user=user))
    body = strip_code_fence(ctx.raw_output or "")
    if not body:
        # A live transport could return empty/whitespace; never persist an empty
        # spec — fall back to the deterministic distillation so compile is total.
        body = offline_condense_body(raw, trace)
    return body


def compile_spec(
    raw: RawDefinition,
    body: str,
    *,
    source: str,
    trace_ref: str = "",
) -> CompiledSpec:
    """Assemble the final :class:`CompiledSpec` from a (already condensed) body.

    Kept separate from :func:`condense_body` so the engine can author a DRAFT
    body, surface it for approval, and only THEN mint the CompiledSpec — the
    registry key is ``raw.name`` and the lookup text is ``raw.description``."""
    return CompiledSpec(
        name=raw.name,
        description=raw.description,
        source=source,
        body=body,
        research_trace_ref=trace_ref,
    )


def condense(
    raw: RawDefinition,
    trace: ResearchTrace,
    *,
    source: str,
    transport: Optional[Transport] = None,
    trace_ref: str = "",
) -> CompiledSpec:
    """One-shot convenience: condense the body AND assemble the CompiledSpec.

    Used by call sites that compile directly (no separate draft/approval step);
    the engine instead splits this into :func:`condense_body` (author the draft)
    then :func:`compile_spec` (mint on approval)."""
    body = condense_body(raw, trace, transport=transport)
    return compile_spec(raw, body, source=source, trace_ref=trace_ref)


__all__ = [
    "build_condense_messages",
    "strip_code_fence",
    "offline_condense_body",
    "default_condense_transport",
    "condense_body",
    "compile_spec",
    "condense",
    "OFFLINE_MARKER",
    "DEFAULT_MAX_NOTES",
    "DEFAULT_MAX_SOURCES",
]
