"""Schema-constrained tool-argument emission (s8/b1 small-model hardening).

A live POC surfaced a small-model limitation: the small agent model, asked
to author a DAG, would name a tool on a node (e.g. ``web_search``) but emit an
EMPTY ``tool_args`` — so the node's tool call failed (``web_search`` needs a
``query``) and only recovered via the slow self-heal/re-plan path. The fix this
module ships (within the a1 tolerance: nudge the model, don't hard-gate it) is to
make tool-arg emission a SEPARATE, SCHEMA-CONSTRAINED call:

- a JSON SCHEMA per tool (required keys + types) is handed to Ollama's native
  ``format=<schema>`` structured-output mode, which forces the model to emit
  syntactically-valid JSON that satisfies the schema (the s8 probe measured this
  as reliable — valid ``{"query": ...}`` first try);
- ``max_tokens`` is raised so a small reasoning model does not truncate the JSON;
- the emission is BOUNDED (a few attempts) and, if the model still cannot produce
  usable args, a deterministic FALLBACK derives them from the node's task text —
  so a research node never hard-fails on a small-model hiccup (a1), while the model
  stays the PRIMARY author of the args (recorded per-call so the demo can prove
  which path ran).

This is additive + back-compat: an :class:`~agent_runtime.runtime.AgentRuntime`
built without a ``tool_arg_emitter`` behaves exactly as before (it uses the
node's own ``tool_args``). The emitter is reusable by the live chat server too.
In-process, dependency-light (d2/d10): one extra scoped phi call, no new service.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from llm_framework import Transport

from .factory import PlanNode
from .selfheal import ToolFailureError
from .synth_tools import deliverable_extension, explicit_filename, unwrap_output_envelope

# JSON schemas (Ollama-native ``format``) for the core tools' arguments. Required
# keys + types so the structured-output mode forces phi to fill them. Kept here
# as data so the emitter and any tool host agree on exactly one shape.
TOOL_ARG_SCHEMAS: dict[str, dict[str, Any]] = {
    "web_search": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "a concise web search query"},
            "max_results": {"type": "integer"},
        },
        "required": ["query"],
    },
    "web_fetch": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "an absolute http(s) URL to fetch"},
        },
        "required": ["url"],
    },
    "write_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
    "read_file": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    # s2/b5 node→tool surface. file_read/file_write are the GrowableToolRegistry
    # filesystem tools (hard-sandboxed); their args mirror read_file/write_file.
    "file_read": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    "file_write": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
    # send_mail is the recipient-LOCKED mail tool (s2/a4, d8 safety invariant).
    # Its schema exposes ONLY subject + body — there is NO ``to`` key, so the
    # model is structurally unable to emit a recipient at arg-emission time (it
    # is always delivered to SMTP_FROM_EMAIL by the adapter). Do NOT add ``to``.
    "send_mail": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "the email subject line"},
            "body": {"type": "string", "description": "the plain-text email body"},
        },
        "required": ["subject", "body"],
    },
    # cron tools (s2/a5) — schedule/list/delete recurring tasks in the shared db.
    "cron_add": {
        "type": "object",
        "properties": {
            "schedule": {
                "type": "string",
                "description": (
                    "a 5-field cron expression TRANSLATED from the time/cadence the "
                    "USER actually asked for, keeping the EXACT minute (e.g. 'every "
                    "day at 8am' -> '0 8 * * *'; 'daily at 13:14 UTC' -> "
                    "'14 13 * * *'); never a default or rounded time the user did "
                    "not state"
                ),
            },
            "prompt": {
                "type": "string",
                "description": (
                    "the WHOLE self-contained task to run each fire, INCLUDING the "
                    "delivery action the user asked for (e.g. 'research the top AI "
                    "news and EMAIL me a summary') — a fired run sees ONLY this "
                    "text, so a dropped verb (like emailing) is never performed"
                ),
            },
            "name": {"type": "string"},
        },
        "required": ["schedule", "prompt"],
    },
    "cron_list": {
        "type": "object",
        "properties": {"enabled_only": {"type": "boolean"}},
        "required": [],
    },
    "cron_delete": {
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    },
}


# A fallback derives args deterministically from the node when phi cannot. Returns
# a dict of args, or ``None`` if it cannot help (then the emitter surfaces a
# tool failure for the runtime's self-heal).
ArgFallback = Callable[[PlanNode], Optional[Mapping[str, Any]]]


def _clean_query(task: str) -> str:
    """Derive a search query from a node's free-text task (deterministic fallback).

    Strips the common instruction verbiage phi wraps a research step in
    ("research X and gather key facts with sources") down to the subject X, so a
    fallback ``web_search`` query is still on-topic without an LLM call."""
    q = (task or "").strip().lower()
    for lead in ("research ", "gather ", "find ", "search for ", "look up ",
                 "investigate ", "study "):
        if q.startswith(lead):
            q = q[len(lead):]
            break
    for tail in (" and gather key facts with sources", " and gather sources",
                 " with sources", " and summarize", " and gather facts",
                 " and collect sources"):
        if q.endswith(tail):
            q = q[: -len(tail)]
    return (q.strip() or task.strip())[:120]


def default_fallback(node: PlanNode) -> Optional[Mapping[str, Any]]:
    """Deterministic args for the search tool; ``None`` for tools we can't guess.

    ``web_search`` → a query distilled from the task. ``web_fetch`` needs a URL
    that cannot be invented, so it returns ``None`` (the emitter then raises a
    ToolFailureError the runtime self-heals)."""
    if node.tool == "web_search":
        return {"query": _clean_query(node.task)}
    return None


# --------------------------------------------------------------------------- #
# a2-recipe (s7/a2) UPSTREAM GROUNDING — derive a tool arg from REAL upstream data
# --------------------------------------------------------------------------- #
# The s7/a2 LIVE POC surfaced the #1 integration gap: the emitter saw ONLY
# ``node.task``, so a derived-from-upstream arg was hallucinated — ``web_fetch.url``
# came back as a placeholder like ``https://www.google.com/search?q=...`` and
# ``file_write.content`` as a stub ("This is the content of the generated report").
# The result was a fired-but-empty chain (the dummy-chatbot symptom). The fix:
# ground those args DETERMINISTICALLY in the upstream node outputs the runtime
# already holds — no LLM call (so no token-budget truncation, no hallucination).

_URL_RE = re.compile(r"https?://[^\s\"'<>)\]}]+")
# Generic, non-relatable filenames a small model defaults to — replace with a
# topic-derived slug so the workspace file has a RELATABLE name (o6/s1 bar).
_GENERIC_NAMES = {
    "report", "final_report", "output", "file", "document", "result",
    "generated_report", "summary", "untitled", "report_md",
}


def _strip_code_fence(text: str) -> str:
    """Drop a leading/trailing ``` fence the small model sometimes wraps output in."""
    s = (text or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _first_url_from_tool_values(tool_values: Optional[Mapping[str, Any]]) -> Optional[str]:
    """A REAL url from the upstream web_search results dict (preferred source).

    ``web_search`` returns ``{"results": [{"url": ...}, ...]}``; pick the first
    result's url — the page the planner's ``web_fetch`` step actually meant to read,
    instead of an invented one. Skips obvious non-article hosts (search redirects)."""
    for v in (tool_values or {}).values():
        if isinstance(v, Mapping):
            results = v.get("results")
            if isinstance(results, (list, tuple)):
                for r in results:
                    if isinstance(r, Mapping) and r.get("url"):
                        url = str(r["url"]).strip()
                        if url.startswith(("http://", "https://")):
                            return url
    return None


def _first_url_from_inputs(inputs: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Last-resort: scrape the first http(s) URL out of the upstream prose."""
    for v in (inputs or {}).values():
        m = _URL_RE.search(str(v))
        if m:
            return m.group(0).rstrip(".,);]")
    return None


def _normalize_report_value(v: Any) -> str:
    """One upstream value → readable deliverable text (R5 / c1r grounding side).

    Beyond the ``{"output": "<str>"}`` unwrap, flatten a bare ``{"findings": [...]}``
    wrapper (the exact shape that leaked onto disk as raw JSON) into readable bullet
    prose, so the writer node persists real text rather than a JSON envelope — the
    graceful counterpart to the file_write tool boundary's refuse (it never has to
    fire on this common shape)."""
    s = unwrap_output_envelope(str(v))
    st = s.strip()
    if st.startswith("{") and st.endswith("}") and ('"findings"' in st or '"output"' in st):
        try:
            obj = json.loads(st)
        except (ValueError, TypeError):
            return s
        if isinstance(obj, dict):
            out = obj.get("output")
            if isinstance(out, str) and out.strip():
                return out
            findings = obj.get("findings")
            if isinstance(findings, (list, tuple)) and findings:
                return "\n".join(f"- {x}" for x in findings)
    return s


def _report_text_from_inputs(inputs: Optional[Mapping[str, Any]]) -> str:
    """The upstream REPORT text to persist as ``file_write.content``.

    The writer node's OWN produced prose cannot reach a tool that fires BEFORE the
    produce step, so the real deliverable is the upstream node output (e.g. the
    summarize step). Prefer a markdown-looking body; among candidates pick the
    longest (the fullest report). Returns "" when there is nothing upstream."""
    texts = [
        _strip_code_fence(_normalize_report_value(v))
        for v in (inputs or {}).values()
        if str(v).strip()
    ]
    if not texts:
        return ""

    def _looks_markdown(t: str) -> bool:
        head = t.lstrip()
        return head.startswith("#") or head.startswith("- ") or "\n#" in t or "\n- " in t

    def _looks_html(t: str) -> bool:
        head = t.lstrip().lower()
        return head.startswith("<!doctype") or head.startswith("<html") or head.startswith("<")

    markdown = [t for t in texts if _looks_markdown(t) and not _looks_html(t)]
    non_html = [t for t in texts if not _looks_html(t)]
    pool = markdown or non_html or texts
    return max(pool, key=len)


def _relatable_filename(node: PlanNode, content: str) -> str:
    """A relatable filename for the deliverable, with the CHOSEN extension (c3r/d49).

    Precedence mirrors the synthesizer's :func:`~agent_runtime.synth_tools.derive_output_path`
    so both write paths agree: (1) an explicit filename the task names survives
    verbatim (``cats.html`` stays ``cats.html``); (2) otherwise the stem is a slug
    from the report's title (first markdown heading / non-tag line, else the task)
    and the EXTENSION comes from the bound output-format writer spec (html-writer ->
    ``.html``) or a format keyword in the task, defaulting to ``.md``. This replaces
    the old hard-coded ``.md`` that turned an html-writer ``cats.html`` request into
    ``findings.md`` when no usable path reached the write node."""
    named = explicit_filename(node.task)
    if named:
        return named
    title = ""
    for line in (content or "").splitlines():
        s = line.strip().lstrip("#").strip()
        # skip html tags / fences when picking a title line
        if s and not s.startswith("<") and not s.startswith("```"):
            title = s
            break
    if not title:
        title = (node.task or "report")
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
    ext = deliverable_extension(node.effective_specs, node.task or "")
    return f"{slug or 'report'}{ext}"


def _has_extension(path: str) -> bool:
    """True when the filename carries its own (short) extension the model chose.

    Used so a model-chosen ``.html``/``.csv``/... is preserved verbatim and never
    replaced by the relatable-name fallback (which only kicks in for a name with no
    extension of its own)."""
    name = (path or "").strip().strip("/\\").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if "." not in name or name.endswith("."):
        return False
    return 1 <= len(name.rsplit(".", 1)[-1]) <= 5


def _is_generic_name(path: str) -> bool:
    name = (path or "").strip().strip("/\\").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return (not stem) or re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_") in _GENERIC_NAMES


# RP-4 (d322/d332): the s7/a3 ``cron_prompt_from_task`` string-surgery that used to
# derive the fired job's ``prompt`` by stripping scheduling scaffolding from
# ``node.task`` was REMOVED. It was engine-authored output (it REWROTE the model's
# cron prompt), the exact d310/d311/d319 fabrication the re-plan retires, and it was
# fragile (empty / truncated on common phrasings). The whole-DAG scheduling doctrine
# now lives in the ``recurring-scheduler`` SPEC the planner stamps on the cron node,
# which drives the MODEL to author the whole self-contained task as the prompt; the
# engine stores it VERBATIM (parse-to-read only).


def _subject_from(node: PlanNode, body: str) -> str:
    """A concise email subject: the report's title line, else the node task.

    Mirrors the relatable-filename derivation — use the first markdown heading /
    first non-empty non-tag line of the grounded body as the subject so the email
    is sensibly titled; fall back to the node's task text. Bounded to a sane
    subject length."""
    for line in (body or "").splitlines():
        s = line.strip().lstrip("#").strip()
        if s and not s.startswith("<") and not s.startswith("```"):
            return s[:120]
    return (node.task or "Update")[:120]


def ground_args_from_upstream(
    node: PlanNode,
    have: Mapping[str, Any],
    inputs: Optional[Mapping[str, Any]],
    tool_values: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Deterministically derive a tool's args from real upstream data, or ``None``.

    Only the two derived-from-upstream tools are grounded; every other tool returns
    ``None`` (the emitter then runs its normal plan/phi/fallback path, byte-identical
    to the pre-fix behaviour). Returns ``None`` whenever there is no upstream data to
    ground from, so a first/standalone node is unaffected.

    * ``web_fetch`` → ``url`` from the upstream ``web_search`` results (real article
      url beats the model's invented one).
    * ``file_write`` / ``write_file`` → ``content`` = the upstream report text
      (the real deliverable, set without an LLM call so it is never a stub) and a
      relatable ``.md`` ``path`` (topic-derived when the model picked a generic name).
    """
    tool = node.tool or ""
    if tool == "web_fetch":
        url = _first_url_from_tool_values(tool_values) or _first_url_from_inputs(inputs)
        if url:
            return {**dict(have), "url": url}
        return None
    if tool in ("file_write", "write_file"):
        report = _report_text_from_inputs(inputs)
        if not report:
            return None
        path = str(have.get("path") or "").strip()
        # RESPECT the LLM's chosen filename + extension (c3/d49: the model picks ANY
        # extension — .md/.txt/.csv/.html/...). The old _ensure_md forced every
        # deliverable to .md; that hard-code is GONE. Only SYNTHESIZE a relatable
        # name when the model gave nothing usable (empty, or a generic stem WITH NO
        # extension of its own) — never strip/override an extension it did choose.
        if not path or (_is_generic_name(path) and not _has_extension(path)):
            path = _relatable_filename(node, report)
        return {"path": path, "content": report}
    if tool == "send_mail":
        # s7/a3: ground the email BODY in the upstream deliverable (the news/report
        # text the prior nodes produced) — exactly as file_write.content is grounded.
        # The action mandates "email content is real (not a stub)": left to the small
        # model's schema emission the body came back a ~38-char placeholder even when
        # an 8-result web_search + a summary ran upstream. Deterministic (no LLM): the
        # real upstream report is the body; the subject is the model's own if it set
        # one, else distilled from the report title / the node task. No upstream text
        # (e.g. a standalone send_mail) → None, so the plan/phi path runs unchanged.
        report = _report_text_from_inputs(inputs)
        if not report:
            return None
        subject = str(have.get("subject") or "").strip() or _subject_from(node, report)
        return {"subject": subject, "body": report}
    return None


@dataclass
class ToolArgEmission:
    """The recorded outcome of emitting one node's tool args (for the evidence)."""

    node_id: str
    tool: str
    final_args: dict[str, Any]
    source: str = "plan"           # plan | phi_schema | fallback
    phi_raw: Optional[str] = None
    attempts: int = 0
    error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "tool": self.tool,
            "final_args": self.final_args,
            "source": self.source,
            "phi_raw": self.phi_raw,
            "attempts": self.attempts,
            "error": self.error,
        }


class SchemaToolArgEmitter:
    """Emit a node's tool args via a schema-constrained phi call (+ fallback).

    Call it as ``await emitter(node)`` (it is the ``tool_arg_emitter`` the
    runtime/sub-agent invokes before a tool call). Behaviour:

    1. if the node already carries valid args (every required key present and
       non-empty) → use them as-is (``source='plan'``; phi emitted them at plan
       time);
    2. else make a BOUNDED, schema-constrained phi call (Ollama native
       ``format=<schema>`` + raised ``max_tokens``) and use the parsed args if
       they satisfy the schema's required keys (``source='phi_schema'``);
    3. else derive args with the deterministic ``fallback`` (``source='fallback'``);
    4. else raise :class:`ToolFailureError` so the runtime's node self-heal runs.

    Every emission is appended to ``self.log`` (a list of :class:`ToolArgEmission`)
    so the demo can prove phi authored the args (and how).
    """

    def __init__(
        self,
        transport: Transport,
        *,
        schemas: Optional[Mapping[str, Mapping[str, Any]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        max_attempts: int = 2,
        fallback: Optional[ArgFallback] = default_fallback,
        think: bool = True,
    ) -> None:
        self.transport = transport
        self.schemas = dict(schemas if schemas is not None else TOOL_ARG_SCHEMAS)
        self.max_tokens = max_tokens
        self.temperature = temperature
        # s1/b1 REASONING ROLLOUT: tool-arg emission is a SCHEMA-CONSTRAINED structured
        # call; gemma4 now reasons about the args in the SEPARATE message.thinking field
        # (``think`` defaults True). Because the CoT competes with the content budget,
        # the default ``max_tokens`` is raised to 4096 (a2-proven load-bearing: at a
        # small budget the CoT eats it and the JSON args come back EMPTY). The transport
        # JSON-extraction interceptor strips any fence before the direct json.loads at
        # :_resolve. The native transport drops ``think`` harmlessly on the OpenAI path /
        # older non-thinking models, so this stays back-compat.
        self.think = think
        self.max_attempts = max_attempts
        self.fallback = fallback
        self.log: list[ToolArgEmission] = []

    @staticmethod
    def _present(args: Mapping[str, Any], required: list[str]) -> bool:
        return all(
            k in args and args[k] not in (None, "", [], {}) for k in required
        )

    def _prompt(self, node: PlanNode, schema: Mapping[str, Any],
                have: Mapping[str, Any], goal: Optional[str] = None) -> tuple[str, str]:
        req = ", ".join(schema.get("required", [])) or "(none)"
        system = (
            f"You emit ONLY the JSON arguments for a '{node.tool}' tool call, "
            f"matching the given schema. Required keys: {req}. Reason it through "
            "privately first; your VISIBLE reply must be ONLY the JSON arguments "
            "object — no prose, no code fences."
        )
        # The run's USER GOAL rides the prompt when the caller supplies it (the
        # scheduled-brief live catch: the node task alone can drop a load-bearing
        # user-stated detail — a time, a recipient, a format — and the model then
        # anchors on a schema example instead of the user's words). Bounded: the
        # goal is the user's own short request text.
        goal_line = f"USER GOAL (ground every argument in the user's OWN words — a stated time, format or recipient is binding): {str(goal).strip()}\n" if goal and str(goal).strip() else ""
        user = (
            goal_line
            + f"TASK (the step this tool call serves): {node.task}\n"
            + (f"ALREADY-KNOWN ARGS: {json.dumps(dict(have))}\n" if have else "")
            + f"TOOL ARG SCHEMA: {json.dumps(dict(schema))}\n\n"
            "Return ONLY the JSON arguments object."
        )
        return system, user

    async def __call__(
        self,
        node: PlanNode,
        inputs: Optional[Mapping[str, Any]] = None,
        tool_values: Optional[Mapping[str, Any]] = None,
        goal: Optional[str] = None,
    ) -> Mapping[str, Any]:
        args = await self._resolve(node, inputs, tool_values, goal)
        # RP-4 (d322/d332): the cron_add PROMPT is authored by the MODEL and STORED
        # VERBATIM. The retired ``cron_prompt_from_task`` string-surgery that used to
        # REWRITE it here was engine-authored output — a tool-name conditional editing
        # the model's bytes, the exact d310/d311/d319 fabrication the re-plan retires
        # (and measured fragile: '' / truncated on common phrasings). The whole-DAG
        # scheduling doctrine now lives in the ``recurring-scheduler`` SPEC the planner
        # stamps on the cron node; the engine only parses-to-read, never rewrites.
        return args

    async def _resolve(
        self,
        node: PlanNode,
        inputs: Optional[Mapping[str, Any]] = None,
        tool_values: Optional[Mapping[str, Any]] = None,
        goal: Optional[str] = None,
    ) -> Mapping[str, Any]:
        schema = self.schemas.get(node.tool or "")
        base = dict(node.tool_args or {})
        # No schema for this tool → leave the node's own args untouched.
        if schema is None:
            self.log.append(ToolArgEmission(node.id, node.tool or "", base, source="plan"))
            return base

        required = list(schema.get("required", []))
        have = {k: v for k, v in base.items() if v not in (None, "", [], {})}

        # a2-recipe (s7/a2): GROUND a derived-from-upstream arg in real upstream data
        # FIRST — before the plan/phi path — so web_fetch.url and file_write.content
        # are the actual search-result url / the actual upstream report, never a
        # hallucinated placeholder. Takes precedence over a plan-time guess for these
        # two tools because the planner cannot know the real value at plan time. When
        # there is no upstream data (inputs/tool_values empty — e.g. a first node or a
        # legacy caller that passed none) this returns None and the original
        # plan/phi/fallback path runs UNCHANGED (full back-compat).
        grounded = ground_args_from_upstream(node, have, inputs, tool_values)
        if grounded is not None and self._present(grounded, required):
            self.log.append(
                ToolArgEmission(node.id, node.tool, dict(grounded), source="grounded")
            )
            return dict(grounded)

        if self._present(have, required):
            self.log.append(ToolArgEmission(node.id, node.tool, base, source="plan"))
            return base

        rec = ToolArgEmission(node.id, node.tool, dict(base), source="phi_schema")
        system, user = self._prompt(node, schema, have, goal)
        last_raw: Optional[str] = None
        for attempt in range(1, self.max_attempts + 1):
            rec.attempts = attempt
            try:
                # transport.chat is synchronous; run it off the event loop so the
                # single in-process loop is never stalled (d2).
                result = await asyncio.to_thread(
                    lambda: self.transport.chat(
                        [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
                        format=schema, max_tokens=self.max_tokens,
                        temperature=self.temperature, think=self.think,
                    )
                )
                last_raw = result.content
                parsed = json.loads(result.content)
            except Exception as exc:  # noqa: BLE001 - bounded; fall through to fallback
                rec.error = f"{type(exc).__name__}: {exc}"
                continue
            if isinstance(parsed, dict):
                merged = {**parsed, **have}  # explicit plan args win over emitted
                if self._present(merged, required):
                    rec.final_args = merged
                    rec.phi_raw = last_raw
                    self.log.append(rec)
                    return merged
            rec.error = f"emitted args missing required {required}: {parsed!r}"

        # phi could not produce usable args within the bound → deterministic fallback.
        rec.phi_raw = last_raw
        if self.fallback is not None:
            fb = self.fallback(node)
            if fb is not None:
                merged = {**dict(fb), **have}
                if self._present(merged, required):
                    rec.source = "fallback"
                    rec.final_args = merged
                    self.log.append(rec)
                    return merged
        # No way to satisfy the schema → a healable tool failure (the runtime
        # re-launches the node / re-plans the sub-graph).
        rec.source = "failed"
        self.log.append(rec)
        raise ToolFailureError(
            f"could not emit valid args for tool {node.tool!r} on node {node.id!r} "
            f"(required {required}); phi+fallback exhausted",
            tool=node.tool,
        )


__all__ = [
    "TOOL_ARG_SCHEMAS",
    "SchemaToolArgEmitter",
    "ToolArgEmission",
    "ArgFallback",
    "default_fallback",
    "ground_args_from_upstream",
]
