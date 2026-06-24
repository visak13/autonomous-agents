"""s9/N5 (d62/c15 part-e) — the REASONING no-fabrication VERIFICATION lane.

The PRIMARY no-fabrication mechanism (neuron ruling): after the gather + write
phases produce a deliverable, this lane RE-CHECKS every deliverable claim against
the **fetched sources** via claim->source PROVENANCE and forces the model to GROUND
or REVISE/REMOVE any unbacked claim. It closes the c13r B2 gap — the UNSCOPED
narrative sections the per-section source-scoping never bound, where a small model
fabricated plausible-sounding specifics (the measured ``17 USC 107(5)`` /
``CTEA-1998`` invention) that no fetched source backs.

WHY REASONING, NEVER REGEX (neuron steer 3af4136c; d14/d48): a string/pattern
content-filter both MISSES novel fabrications and RISKS stripping valid content. So
groundedness is decided by the MODEL reasoning over the real sources — this module
only ORCHESTRATES the turns, renders the provenance the model reasons over, and
relays the model's verdict; it never decides what is "backed" by matching strings.

The lane treats THREE conditions as a no-fabrication FAILURE that forces
gather-more or revise/remove (the action's mandate + the N1r path-divergence
carry-forward):

* a deliverable CLAIM with NO backing fetched source (the c13r B2 narrative gap);
* a research stage that produced **0 fetches / answered from the model's OWN
  MEMORY** when it should have gathered — E4B does this on the bare ReAct path
  (the e4b-fetch-ceiling-diverges-by-path learning). This is a structural
  PROVENANCE signal (no real source was read), NOT a content regex.

DESIGN, mirroring the N2/N3 control lanes (``article_note`` / ``chunked_read``):

* PURE + transport-agnostic. The model turn is an injected async ``verify``
  callback (prompt -> text), so the lane is unit-testable with a fake verifier; the
  served wiring (N6) supplies a closure that runs one real E4B turn. No I/O here.
* The VERDICT is lightweight structured CONTROL the small model emits reliably
  (a short JSON list of unbacked claims — the same surface as the c5 tool args /
  the ``ArticleNote``), d50.1-clean. The REVISED DELIVERABLE is RAW content
  (never JSON-wrapped) — content is RAW on every route (d50/d50.1/d51).
* GROUNDED-ONLY prompts (anti-fabrication, d49/d60): the revise turn may ONLY
  ground a claim from the listed real sources or REMOVE it; it may NEVER invent a
  fact, figure, citation or source. "Not enough source" is always "gather more or
  drop," never pad.
* It COMPOSES with the UNCHANGED c13 write side: it runs as a checkpoint OVER the
  produced ``(document, sources)`` — strictly downstream, it changes no write-side
  signature. ``sources`` is the run's fetched-source list (``collect_fetched_sources_full``
  shape: ``{title, url, markdown}``), OPTIONALLY enriched with the N2
  ``ArticleNote`` provenance (``source_trust`` / ``key_claims``) so the check can
  weight a reference-untrusted (Wikipedia) source as never-sole-backing-for-a-hard-figure
  (d60).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

from .research_tree import first_native_call, make_tool_spec
from .synth_tools import resolve_writer_source_budget

# The injected async one-shot model turn: prompt -> raw model text. The served
# wiring (N6) runs one real E4B reasoning turn (think=True, temperature=0); the
# tests pass a scripted fake. Same injection shape as ``chunked_read.Summarize``.
Verify = Callable[[str], Awaitable[str]]

# s13/P1 (FIX-C + d118) — the NATIVE verdict turn: prompt -> ``(raw_text, tool_calls)``.
# Same shape as :meth:`agent_runtime.runtime.SubAgent._research_emit` (the native helper
# P1-impl built). The verify VERDICT is structured CONTROL (supported/issues), so per d114
# it rides the model's OWN ``message.tool_calls`` channel where LEADING PROSE can never
# swallow it; the raw text is also returned so a NON-native reply still feeds the kept
# balanced-brace parser (the defensive fallback, d117 condition 2). The RAW revise turn
# stays text-only (the corrected document is RAW content, never a tool call — d50).
VerifyNative = Callable[
    [str], Awaitable[tuple[str, Optional[Sequence[Mapping[str, Any]]]]]
]

# Bounds so the verify prompt stays inside the model's window (E4B 32768-tok / the
# 512 SWA reality). Each source contributes its provenance (id/trust/title/url) +
# its distilled key_claims (when an ArticleNote enriched it) + an excerpt of the real
# article text — enough to reason groundedness, never the whole corpus. MSF/d89: the
# default is RAISED in LOCKSTEP with the writer (``resolve_writer_source_budget`` — env
# RA_WRITER_SOURCE_BUDGET, default 12000) so Seam-B judges the SAME content the writer
# used; the runtime ALSO threads its per-source budget via ``excerpt_budget=`` at the
# call site (true lockstep regardless of import timing). A starved 700-char verify
# excerpt would otherwise flag a claim the writer correctly grounded in text it saw.
_SOURCE_EXCERPT_BUDGET = resolve_writer_source_budget()
_MAX_KEY_CLAIMS = 6
# Below this the "findings" of a 0-fetch research stage are too thin to be a
# substantive answer-from-memory (an empty/aborted leaf is not a fabrication).
_MEMORY_ANSWER_MIN_CHARS = 200


# ---------------------------------------------------------------------------- #
# (B-b) The research-stage no-fab PROVENANCE signal — 0 fetches / from memory.
# ---------------------------------------------------------------------------- #
def research_answered_from_memory(
    findings: str,
    fetched_count: int,
    *,
    min_chars: int = _MEMORY_ANSWER_MIN_CHARS,
) -> bool:
    """True when a research stage produced a substantive answer with **0 fetches**.

    A research stage's job is to GATHER: if it read **no** real source
    (``fetched_count <= 0``) yet emitted substantive findings (``>= min_chars`` of
    prose), the model answered from its OWN MEMORY — a no-fabrication FAILURE the
    orchestration must treat as "gather more or revise/remove" (the model may NOT
    answer a research task from memory; E4B does this on the bare ReAct path).

    This is a structural PROVENANCE count — *no source was read* — NOT a content
    regex over the claims (which the neuron steer forbids). An empty/near-empty
    leaf (a genuinely aborted gather) returns ``False``: there is no fabricated
    answer to force-revise, the orchestration simply re-dispatches the gather."""
    if fetched_count and fetched_count > 0:
        return False
    return len((findings or "").strip()) >= max(1, int(min_chars))


# ---------------------------------------------------------------------------- #
# Provenance rendering — the claim->source evidence the model reasons over.
# ---------------------------------------------------------------------------- #
def render_sources_for_verify(
    sources: Sequence[Mapping[str, Any]],
    *,
    excerpt_budget: int = _SOURCE_EXCERPT_BUDGET,
) -> str:
    """Render the fetched sources as a compact, numbered PROVENANCE block.

    Each source shows its 1-based id (the stable citation number the c13 write side
    already assigns), its trust tier (when an :class:`~agent_runtime.article_note.ArticleNote`
    enriched it — Wikipedia/reference-untrusted is flagged so the model never treats
    it as sole backing for a hard figure, d60), its title + URL, its distilled
    ``key_claims`` (when present), and a SHORT excerpt of the real article text. This
    is the evidence the verify turn reasons each deliverable claim against — kept
    bounded so the whole provenance set fits the model's window. Returns ``""`` for
    no sources (a no-source deliverable is judged entirely unbacked by the caller)."""
    if not sources:
        return ""
    budget = max(120, int(excerpt_budget))
    lines = ["FETCHED SOURCES (the ONLY real evidence — cite a claim only if one of "
             "these backs it):"]
    for i, s in enumerate(sources, 1):
        if not isinstance(s, Mapping):
            continue
        title = str(s.get("title") or "").strip()
        url = str(s.get("url") or "").strip()
        trust = str(s.get("source_trust") or "").strip()
        tag = f" [{trust}]" if trust else ""
        header = f"[{i}]{tag} {title or url} — {url}".rstrip(" —")
        lines.append(f"\n{header}")
        claims = s.get("key_claims")
        if isinstance(claims, (list, tuple)) and claims:
            for c in list(claims)[:_MAX_KEY_CLAIMS]:
                c = str(c).strip()
                if c:
                    lines.append(f"  - {c}")
        body = str(s.get("markdown") or s.get("excerpt") or "").strip()
        if body:
            lines.append(f"  text: {body[:budget]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------- #
# VERIFY turn — the model REASONS which deliverable claims are unbacked.
# ---------------------------------------------------------------------------- #
# The verify turn returns a LIGHTWEIGHT verdict (control, not content): a short
# JSON object listing the unbacked claims. This is the surface a small model emits
# reliably (like the c5 tool args / the ArticleNote) — the document itself is NEVER
# round-tripped through JSON here (that is the RAW revise turn's job).
_VERIFY_INSTRUCTION = (
    "You are a strict FACT-CHECKER enforcing a no-fabrication rule: EVERY factual "
    "claim in the report below must be backed by one of the FETCHED SOURCES — a "
    "real source that was actually read. A claim that no fetched source supports is "
    "UNBACKED (fabricated) and must be flagged, however plausible it sounds. Treat a "
    "specific statute/section number, a named law or case, a hard figure, a date, a "
    "quote or a proper name as a claim that NEEDS a backing source. A "
    "reference-untrusted source (e.g. Wikipedia) may NOT be the sole backing for a "
    "hard figure. Do NOT flag generic framing, transitions, or common knowledge.\n\n"
    "REASON over the report claim by claim against the sources. Then reply with ONLY "
    "a JSON object and NOTHING else:\n"
    '  {"verdict":"ok"}                       — if every claim is backed by a source\n'
    '  {"verdict":"revise","unbacked":[ {"claim":"<short verbatim snippet of the '
    'unbacked claim>","reason":"<which source it lacks / why no source backs it>"} ] }\n'
    "List EACH unbacked claim once, with a short verbatim snippet so it can be found. "
    "Judge by the SOURCES shown, not your own knowledge. ONLY the JSON object."
)

# ---------------------------------------------------------------------------- #
# (A) SPEC-AWARE GENERIC REVIEWER (FIX-C, d114) — the verify lane is no longer a
# spec-BLIND fixed fact-checker: it receives the WORKER'S SAME SPEC(s) (mirroring the
# per-node ``review_and_fix`` gate, which re-uses the producing node's spec body) so it
# reviews the deliverable against the rules it was BUILT to satisfy, in ADDITION to the
# no-fabrication source grounding. Empty spec → byte-identical to the legacy fact-checker
# (the default, so every standing offline test and the verify-lane-OFF path are unchanged).
# ---------------------------------------------------------------------------- #
def _review_spec_block(spec: str) -> str:
    """The REVIEW SPEC block injected at the TOP of the verify prompt (empty when no spec).

    Renders the worker's composed ruleset as the rubric the reviewer also checks the
    report against — so the same spec that SHAPED the deliverable now GRADES it. Returns
    ``""`` for no spec, leaving the prompt byte-identical to the spec-blind fact-checker."""
    s = (spec or "").strip()
    if not s:
        return ""
    return (
        "REVIEW SPEC (the rules THIS deliverable was built to satisfy — review the "
        "report against these AS WELL AS the sources below):\n"
        f"{s}\n\n"
    )


# When a spec is present the fact-checker becomes a GENERIC REVIEWER: it must flag a SPEC
# violation the SAME structured way it flags an unbacked claim, so one verdict covers both
# concerns. Appended to ``_VERIFY_INSTRUCTION`` only when a spec was supplied.
_SPEC_REVIEW_DIRECTIVE = (
    "In ADDITION to the no-fabrication source grounding above, you are a REVIEWER for "
    "this deliverable: also check the report against the REVIEW SPEC shown at the top. "
    "Treat a clear SPEC violation exactly like an unbacked claim — add it to the same "
    'list with {"claim":"<the offending span>","reason":"<which spec rule it breaks>"}. '
    "Return verdict \"ok\" ONLY when the report is both fully source-grounded AND "
    "spec-conformant; otherwise \"revise\" with every issue listed once."
)


def _verify_instruction(spec: str) -> str:
    """The verify instruction: the no-fab fact-checker, plus the spec-review directive
    when a worker spec was supplied (generic reviewer). Verdict JSON shape is unchanged
    either way, so the same parser reads both."""
    if (spec or "").strip():
        return f"{_VERIFY_INSTRUCTION}\n\n{_SPEC_REVIEW_DIRECTIVE}"
    return _VERIFY_INSTRUCTION


# ---------------------------------------------------------------------------- #
# (B) NATIVE verdict + (A) REVIEWER FILE TOOLS — the structured surfaces the reviewer
# gets. The VERDICT rides ``message.tool_calls`` (d114/d118: prose can never swallow it);
# the file READ/WRITE/UPDATE schemas are the proper file tools a generic reviewer needs to
# inspect and surgically correct the deliverable in place (file_write stays RAW, d50).
# ---------------------------------------------------------------------------- #
VERIFY_VERDICT_TOOL = "verify_verdict"
VERIFY_VERDICT_SPEC = make_tool_spec(
    VERIFY_VERDICT_TOOL,
    "Return the structured review verdict for the report: 'ok' when every claim is "
    "source-backed (and, if a review spec is given, spec-conformant), else 'revise' "
    "with each unbacked-or-violating span listed.",
    {
        "verdict": {"type": "string", "enum": ["ok", "revise"]},
        "unbacked": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["claim"],
            },
        },
    },
    ["verdict"],
)

# The reviewer's proper file tools (FIX-C): READ to inspect the deliverable, WRITE to
# re-emit it, UPDATE to surgically ground-or-remove ONE flagged span in place (the
# small-model-friendly edit that avoids a whole-document re-emission). Offered as native
# schemas so the served reviewer turn passes them as ``tools=[...]``.
REVIEWER_FILE_TOOL_SPECS: tuple[dict[str, Any], ...] = (
    make_tool_spec(
        "file_read",
        "Read the deliverable (or a slice) back from the workspace to inspect its "
        "current on-disk state.",
        {"path": {"type": "string"}, "tail": {"type": "integer"},
         "offset": {"type": "integer"}, "length": {"type": "integer"}},
        ["path"],
    ),
    make_tool_spec(
        "file_write",
        "Write/replace the deliverable file with corrected RAW content (no JSON wrapper).",
        {"path": {"type": "string"}, "content": {"type": "string"},
         "append": {"type": "boolean"}},
        ["path"],
    ),
    make_tool_spec(
        "file_update",
        "Surgically correct ONE span in place: replace the exact 'old' snippet with 'new' "
        "(empty 'new' removes it). The reviewer's ground-or-remove edit.",
        {"path": {"type": "string"}, "old": {"type": "string"},
         "new": {"type": "string"}, "count": {"type": "integer"}},
        ["path", "old"],
    ),
)

# The full reviewer tool surface threaded onto the native verify turn: the file tools the
# reviewer operates with PLUS the structured verdict it returns.
REVIEWER_TOOL_SPECS: tuple[dict[str, Any], ...] = REVIEWER_FILE_TOOL_SPECS + (
    VERIFY_VERDICT_SPEC,
)


@dataclass
class UnbackedClaim:
    """One deliverable claim the verify turn judged unsupported by any fetched source."""

    claim: str
    reason: str = ""


@dataclass
class VerifyResult:
    """The verify turn's verdict over a deliverable.

    ``grounded`` is True when the model found every claim backed by a fetched source
    (verdict ``ok``); otherwise ``unbacked`` carries the claims to ground-or-remove.
    ``parsed`` is False when the model's reply could not be read as a verdict (the
    caller treats an unreadable verdict CONSERVATIVELY — it does not fabricate a
    pass, and may re-prompt or skip revision rather than strip content)."""

    grounded: bool
    unbacked: list[UnbackedClaim] = field(default_factory=list)
    verdict: str = ""
    parsed: bool = True
    raw: str = ""


def _first_json_object(s: str) -> Optional[str]:
    """First balanced ``{...}`` object in ``s`` (string-literal/escape aware), or None.

    The same dependency-free scan the sibling lanes use (``verify._first_json_object`` /
    ``research_tree._first_json_object``) so a verdict can be read out of a reply that
    wraps the JSON in prose or a code fence; an unbalanced (truncated) object is
    unreadable -> None (never a half-parsed verdict)."""
    start = s.find("{")
    if start < 0:
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
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _strip_fence(s: str) -> str:
    """Strip a leading/trailing ``` code fence (the model often fences its JSON)."""
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s
        if s.endswith("```"):
            s = s[:-3]
        s = s.replace("```json", "").replace("```", "")
    return s.strip()


def _coerce_unbacked(raw_unbacked: Any) -> list[UnbackedClaim]:
    """Normalise a verdict's unbacked/issues list (from native args OR parsed JSON) into
    :class:`UnbackedClaim` records. Shared by the string parser and the native builder so
    both verdict surfaces read EXACTLY the same flagged-claim shape."""
    unbacked: list[UnbackedClaim] = []
    if isinstance(raw_unbacked, (list, tuple)):
        for item in raw_unbacked:
            if isinstance(item, Mapping):
                claim = str(item.get("claim") or item.get("text") or item.get("snippet") or "").strip()
                reason = str(item.get("reason") or item.get("why") or "").strip()
            else:
                claim, reason = str(item).strip(), ""
            if claim:
                unbacked.append(UnbackedClaim(claim=claim[:600], reason=reason[:400]))
    return unbacked


def _verdict_from_obj(obj: Mapping[str, Any], *, raw: str) -> VerifyResult:
    """Build a :class:`VerifyResult` from a verdict mapping (native tool args or parsed
    JSON). Grounded iff NO claim was flagged — the same gate for both surfaces."""
    verdict = str(obj.get("verdict") or "").strip().lower()
    raw_unbacked = (
        obj.get("unbacked") or obj.get("issues") or obj.get("claims")
        or obj.get("unsupported") or []
    )
    unbacked = _coerce_unbacked(raw_unbacked)
    # Grounded iff the verify turn flagged NO unbacked claim — there is nothing to
    # ground-or-remove. The verdict WORD is advisory (kept for the trace); what gates the
    # no-fab lane is whether a claim was flagged. An explicit "revise" with an EMPTY list
    # is therefore grounded (nothing actionable) — we never invent an unbacked claim, and
    # never strip without a flag.
    return VerifyResult(
        grounded=not unbacked,
        unbacked=unbacked,
        verdict=verdict,
        parsed=True,
        raw=raw,
    )


def verdict_from_native_args(args: Mapping[str, Any]) -> VerifyResult:
    """Build a :class:`VerifyResult` from the NATIVE ``verify_verdict`` tool-call args.

    The structured-verdict counterpart to :func:`parse_verify_verdict`: the verdict arrived
    on the model's ``message.tool_calls`` channel (so leading prose could never swallow it,
    d114/d118) and is already a parsed mapping — no JSON-from-prose extraction needed. Reads
    the SAME ``verdict`` + ``unbacked``/``issues`` shape, so a native and a fallback verdict
    are indistinguishable downstream."""
    if not isinstance(args, Mapping):
        return VerifyResult(grounded=False, parsed=False, raw="")
    return _verdict_from_obj(args, raw=json.dumps(dict(args), ensure_ascii=False))


def parse_verify_verdict(raw: str) -> VerifyResult:
    """Parse the verify turn's lightweight JSON verdict into a :class:`VerifyResult`.

    A reply of ``{"verdict":"ok"}`` (or any verdict with no unbacked list) means
    grounded. ``{"verdict":"revise","unbacked":[...]}`` carries the flagged claims.
    An unreadable / non-JSON reply yields ``parsed=False`` (and ``grounded=False``)
    so the caller never reads an unparseable verify turn as a silent PASS — but it
    also has no unbacked list to act on, so it will not strip content on noise.

    This is the DEFENSIVE FALLBACK kept from d117(2): when the verdict does NOT arrive on
    a native tool-call channel it is recovered from the reply text by the balanced-brace
    scan, so every non-native path keeps working unchanged."""
    text = _strip_fence(raw or "")
    blob = _first_json_object(text)
    if not blob:
        return VerifyResult(grounded=False, parsed=False, raw=raw or "")
    try:
        obj = json.loads(blob)
    except (ValueError, TypeError):
        return VerifyResult(grounded=False, parsed=False, raw=raw or "")
    if not isinstance(obj, Mapping):
        return VerifyResult(grounded=False, parsed=False, raw=raw or "")
    return _verdict_from_obj(obj, raw=raw or "")


async def verify_claims(
    document: str,
    sources: Sequence[Mapping[str, Any]],
    *,
    verify: Optional[Verify] = None,
    verify_native: Optional[VerifyNative] = None,
    goal: str = "",
    spec: str = "",
    excerpt_budget: int = _SOURCE_EXCERPT_BUDGET,
) -> VerifyResult:
    """Run ONE reasoning verify turn over a deliverable -> the unbacked-claim verdict.

    The model is shown the fetched-source provenance (:func:`render_sources_for_verify`)
    + the deliverable, and REASONS which claims no source backs (claim->source
    provenance, trust-weighted — d60). When a ``spec`` is supplied it ALSO reviews the
    report against that worker spec (the generic-reviewer rework, FIX-C). Returns the
    parsed :class:`VerifyResult`.

    The verdict is read NATIVE-FIRST (d118): when ``verify_native`` is given the verdict
    rides the model's ``message.tool_calls`` channel (:data:`VERIFY_VERDICT_TOOL`) where
    leading prose can never swallow it; a reply that carries no native call falls back to
    the kept balanced-brace parser over the reply text. With only ``verify`` (the legacy
    text turn) the reply is parsed by the balanced-brace path exactly as before.

    An empty deliverable is trivially grounded (nothing to check). A deliverable with
    NO fetched sources but real claims is judged entirely unbacked (every claim lacks
    a source) — the model still reasons it, but the caller must gather-or-remove."""
    if verify is None and verify_native is None:
        raise ValueError("verify_claims needs a verify or verify_native turn")
    doc = (document or "").strip()
    if not doc:
        return VerifyResult(grounded=True, verdict="ok")
    src_block = render_sources_for_verify(sources, excerpt_budget=excerpt_budget)
    if not src_block:
        src_block = "FETCHED SOURCES: (none — NO real source was read for this report)."
    goal_line = f"GOAL: {goal.strip()}\n\n" if goal and goal.strip() else ""
    spec_block = _review_spec_block(spec)
    prompt = (
        f"{goal_line}{spec_block}{src_block}\n\n"
        f"REPORT TO FACT-CHECK:\n{doc}\n\n{_verify_instruction(spec)}"
    )
    # NATIVE-FIRST verdict (B): the structured verdict on ``message.tool_calls`` is
    # drop-immune to leading prose; only when no native call arrives do we fall back to
    # the balanced-brace parser over the reply text (the kept d117(2) fallback).
    if verify_native is not None:
        try:
            text, tool_calls = await verify_native(prompt)
        except Exception:  # noqa: BLE001 - a verifier hiccup must not crash the pipeline
            return VerifyResult(grounded=False, parsed=False, raw="")
        native = first_native_call(tool_calls, (VERIFY_VERDICT_TOOL,))
        if native is not None:
            return verdict_from_native_args(native[1])
        return parse_verify_verdict(text or "")
    try:
        raw = (await verify(prompt) or "").strip()
    except Exception:  # noqa: BLE001 - a verifier hiccup must not crash the pipeline
        return VerifyResult(grounded=False, parsed=False, raw="")
    return parse_verify_verdict(raw)


# ---------------------------------------------------------------------------- #
# REVISE turn — force the model to GROUND or REVISE/REMOVE the unbacked claims.
# ---------------------------------------------------------------------------- #
# The revised deliverable comes back RAW (content is RAW on every route — d50/d50.1).
# Grounded-only: ground each flagged claim from the listed real sources, or REMOVE
# it; NEVER invent a fact/figure/citation/source to keep it. The whole document is
# re-emitted so the model can excise a claim cleanly in context.
_REVISE_INSTRUCTION = (
    "A fact-check found UNBACKED claims in your report — claims that NO fetched "
    "source supports. Produce a CORRECTED report that fixes EACH one by either:\n"
    "  (a) GROUNDING it — restate it to match what a listed FETCHED SOURCE actually "
    "says (and cite that source's URL), if a source backs it; OR\n"
    "  (b) REMOVING it — delete the claim (and any sentence that depends on it) if NO "
    "fetched source backs it.\n"
    "You may NOT invent a fact, figure, statute/section number, citation, date, "
    "publication or URL to keep a claim. Do NOT add new claims. Keep every backed "
    "claim and the report's structure/format intact. Output ONLY the full corrected "
    "report (raw, same format as the input) — no preamble, no JSON, no explanation."
)


def _render_unbacked(unbacked: Sequence[UnbackedClaim]) -> str:
    lines = ["UNBACKED CLAIMS to ground-or-remove:"]
    for i, u in enumerate(unbacked, 1):
        reason = f" — {u.reason}" if u.reason else ""
        lines.append(f"{i}. {u.claim}{reason}")
    return "\n".join(lines)


@dataclass
class RevisionResult:
    """The outcome of the verify-and-revise checkpoint over a deliverable.

    ``document`` is the deliverable AFTER the lane (the revised text when a revision
    fired, else the original unchanged). ``revised`` is True when a grounded-or-remove
    pass actually rewrote it. ``unbacked`` is what the final verify turn flagged;
    ``grounded`` is True when the deliverable ended with no unbacked claim. ``passes``
    counts the verify turns run; ``trace`` is per-pass detail for the live proof."""

    document: str
    revised: bool
    grounded: bool
    unbacked: list[UnbackedClaim] = field(default_factory=list)
    passes: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)


async def verify_and_revise(
    document: str,
    sources: Sequence[Mapping[str, Any]],
    *,
    verify: Verify,
    revise: Optional[Verify] = None,
    verify_native: Optional[VerifyNative] = None,
    goal: str = "",
    spec: str = "",
    max_passes: int = 1,
    excerpt_budget: int = _SOURCE_EXCERPT_BUDGET,
    min_retention_ratio: float = 0.5,
) -> RevisionResult:
    """The N5 checkpoint: VERIFY the deliverable, then force GROUND-or-REVISE/REMOVE.

    Runs :func:`verify_claims`; if it is already grounded, returns the document
    unchanged (a clean report is never nagged or stripped — the steer's "do not strip
    valid content"). On flagged unbacked claims it runs a RAW revise turn (``revise``,
    or ``verify`` reused) that grounds each claim from the listed sources or removes
    it (never invents), re-verifies the result, and repeats up to ``max_passes``
    times. Both the unbacked-claim judgement and the rewrite are the MODEL's reasoning
    — this orchestrator never edits the text itself (no regex content-filter, d14/d48).

    SAFEGUARD against a truncated / over-deleting revise turn (a single whole-document
    re-emission can be cut short by the output budget, or the model can over-prune):
    a revision shorter than ``min_retention_ratio`` of the original is REJECTED (the
    original stands, the unbacked verdict is surfaced) so the lane never blanks or
    guts a real deliverable. This is a "don't blank the deliverable" length FLOOR (the
    same class as the write loop's empty-doc floor), NOT a content regex over claims —
    grounding is still the model's reasoning; this only refuses a catastrophic rewrite.

    Returns a :class:`RevisionResult`: the final document, whether a revision fired,
    whether it ended grounded, the residual unbacked list, and a per-pass trace. If the
    verify turn is UNREADABLE (``parsed=False``) the document is returned UNCHANGED with
    ``grounded=False`` — the lane surfaces the failure rather than stripping on noise."""
    do_revise = revise or verify
    current = document or ""
    trace: list[dict[str, Any]] = []
    last: VerifyResult = VerifyResult(grounded=True, verdict="ok")
    revised_any = False
    ratio = max(0.0, min(1.0, float(min_retention_ratio)))

    passes = max(1, int(max_passes))
    for p in range(passes):
        last = await verify_claims(
            current, sources, verify=verify, verify_native=verify_native,
            goal=goal, spec=spec, excerpt_budget=excerpt_budget,
        )
        trace.append({
            "pass": p + 1,
            "grounded": last.grounded,
            "parsed": last.parsed,
            "verdict": last.verdict,
            "unbacked": [u.claim for u in last.unbacked],
        })
        if last.grounded or not last.parsed or not last.unbacked:
            # Grounded → done; unreadable verdict → surface, don't strip; no actionable
            # unbacked list → nothing to ground-or-remove (never invent a revision).
            break

        src_block = render_sources_for_verify(sources, excerpt_budget=excerpt_budget)
        if not src_block:
            src_block = "FETCHED SOURCES: (none — NO real source was read for this report)."
        prompt = (
            f"{src_block}\n\n{_render_unbacked(last.unbacked)}\n\n"
            f"REPORT TO CORRECT:\n{current}\n\n{_REVISE_INSTRUCTION}"
        )
        try:
            new_doc = (await do_revise(prompt) or "").strip()
        except Exception:  # noqa: BLE001 - a revise hiccup leaves the document as-is
            new_doc = ""
        # Accept a revision ONLY when it returned real content of a sane length: an
        # empty reply (model failed to revise) or a catastrophically short one (a
        # truncated/over-pruned rewrite) is REJECTED so the deliverable is never blanked
        # or gutted — the original stands and the unbacked verdict is surfaced instead.
        floor = int(len(current.strip()) * ratio)
        if new_doc and new_doc != current.strip() and len(new_doc) >= floor:
            current = new_doc
            revised_any = True
            trace[-1]["rewrote"] = True
        else:
            trace[-1]["rewrote"] = False
            if new_doc and len(new_doc) < floor:
                trace[-1]["rejected_short"] = True
            break

    return RevisionResult(
        document=current,
        revised=revised_any,
        grounded=bool(last.grounded),
        unbacked=list(last.unbacked),
        passes=len(trace),
        trace=trace,
    )


__all__ = [
    "Verify",
    "VerifyNative",
    "UnbackedClaim",
    "VerifyResult",
    "RevisionResult",
    "VERIFY_VERDICT_TOOL",
    "VERIFY_VERDICT_SPEC",
    "REVIEWER_FILE_TOOL_SPECS",
    "REVIEWER_TOOL_SPECS",
    "research_answered_from_memory",
    "render_sources_for_verify",
    "parse_verify_verdict",
    "verdict_from_native_args",
    "verify_claims",
    "verify_and_revise",
]
