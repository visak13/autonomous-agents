"""LOCAL trace -> MARKDOWN renderer (s7 Stage-B).

Stage A (:mod:`agent_runtime.local_trace`) captures one self-contained JSON per
trace under ``var/traces`` — the full span tree plus, for every ``llm.chat`` span,
the local-only enrichment the transport stashed: the complete system+user prompt
messages, the model's ``thinking`` reasoning block, and per-call token counts.
Phoenix is blind to all of that, which is exactly what made the app impossible to
debug.

This module turns that JSON into a HUMAN-READABLE markdown file (``<trace_id>.md``
beside the JSON) so the user and the neuron can actually *read* a run: the span
hierarchy in execution order (``agent.node`` / ``planner.*`` -> ``llm.chat``), and
for each LLM call its prompt (system + user turns), its reasoning/thinking block,
and its token cost (prompt + completion + total), plus a per-run token total.

It is PURELY a renderer: it reads the JSON Stage A already wrote and never touches
the capture path, the transport, or the Phoenix export. Three entry points:

* :func:`render_trace_markdown` — pure (trace dict -> markdown string), no I/O.
* :func:`render_trace_file` — read one ``<trace_id>.json``, write ``<trace_id>.md``.
* :func:`render_all` — render every JSON in the trace dir.

It can be invoked automatically at trace end (see
:class:`MarkdownTraceProcessor`, wired in :mod:`agent_runtime.tracing` alongside —
never replacing — the JSON capture path) or by hand via the CLI::

    python -m agent_runtime.local_trace_md            # render every trace in the dir
    python -m agent_runtime.local_trace_md <file.json>  # render one trace
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional

# Reuse Stage A's env knob + default so the source/target dir is identical and
# never re-hardcoded — markdown lands right next to the JSON it was rendered from.
from agent_runtime.local_trace import resolve_local_trace_dir

# Base class for the at-trace-end processor. We subclass the REAL SDK
# ``SpanProcessor`` (not a bare ``object``) so we inherit ALL of its hooks —
# crucially ``_on_ending``, which the SDK's ``SynchronousMultiSpanProcessor``
# calls on every registered processor as a span ends (``Span.end`` ->
# ``_span_processor._on_ending(span)``). A duck-typed class that omits it raises
# ``AttributeError`` on the FIRST span end, before ``on_end`` ever runs — which
# breaks every run (caught in s7/b2 review). Subclassing also future-proofs us
# against new optional hooks. The import is guarded so the pure-renderer / CLI
# path still imports if OpenTelemetry is somehow absent (then this processor is
# never registered anyway — tracing.py only wires it where OTel is present).
try:
    from opentelemetry.sdk.trace import SpanProcessor as _SpanProcessor
except Exception:  # pragma: no cover - OTel is always present in agent_runtime
    _SpanProcessor = object  # type: ignore[assignment, misc]

_LLM_SPAN_NAME = "llm.chat"

# Per-span-name label hints: which attributes best name this step for a reader.
# Falls back gracefully to whatever attributes exist, so an unknown span name is
# still rendered (forest-safe, schema-tolerant) rather than dropped.
_LABEL_ATTRS: Dict[str, List[str]] = {
    "agent.node": ["node.id", "node.task", "node.spec_names"],
    "planner.assess_ambiguity": [
        "planner.ambiguity.goal",
        "planner.ambiguity.needs_clarification",
    ],
    "planner.select_shape": ["select.goal", "select.shape", "select.escalate"],
    "planner.plan": ["plan.goal", "plan.node_count"],
    "agent.session": ["session.goal", "session.id"],
}


def _fmt_int(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def _fence(text: Any, lang: str = "") -> str:
    """Render text in a fenced block, picking a fence long enough to be safe."""
    s = "" if text is None else str(text)
    fence = "```"
    while fence in s:
        fence += "`"
    return f"{fence}{lang}\n{s}\n{fence}"


def _build_forest(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Order spans as a parent->children forest, each level by start time.

    Spans whose ``parent_span_id`` is null OR points outside this file (a parent
    span that was exported to a different trace/never captured) are treated as
    roots, so nothing is ever silently dropped.
    """
    by_id = {s["span_id"]: s for s in spans}
    children: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for s in spans:
        parent = s.get("parent_span_id")
        if parent not in by_id:
            parent = None  # missing/foreign parent -> render as a root
        children.setdefault(parent, []).append(s)

    ordered: List[Dict[str, Any]] = []

    def _walk(node: Dict[str, Any], depth: int) -> None:
        node["_depth"] = depth
        ordered.append(node)
        kids = sorted(
            children.get(node["span_id"], []),
            key=lambda s: s.get("start_time_ns") or 0,
        )
        for kid in kids:
            _walk(kid, depth + 1)

    for root in sorted(children.get(None, []), key=lambda s: s.get("start_time_ns") or 0):
        _walk(root, 0)
    return ordered


def _span_label(span: Dict[str, Any]) -> str:
    """A short, reader-facing label for a span: name + its most telling attrs."""
    attrs = span.get("attributes") or {}
    name = span.get("name", "?")
    parts: List[str] = []
    for key in _LABEL_ATTRS.get(name, []):
        if key in attrs and attrs[key] not in (None, ""):
            short = key.split(".")[-1]
            val = str(attrs[key])
            if len(val) > 120:
                val = val[:117] + "..."
            parts.append(f"{short}={val}")
    return f"{name}" + (f" — {', '.join(parts)}" if parts else "")


def _llm_tokens(span: Dict[str, Any]) -> Dict[str, int]:
    """Prompt/completion/total tokens for an llm.chat span (attrs first, then capture)."""
    attrs = span.get("attributes") or {}
    cap = span.get("llm_capture") or {}
    counts = cap.get("token_counts") or {}
    prompt = attrs.get("llm.token_count.prompt", counts.get("prompt"))
    completion = attrs.get("llm.token_count.completion", counts.get("completion"))
    prompt = int(prompt) if prompt is not None else 0
    completion = int(completion) if completion is not None else 0
    total = attrs.get("llm.token_count.total")
    total = int(total) if total is not None else prompt + completion
    return {"prompt": prompt, "completion": completion, "total": total}


def _render_llm_capture(span: Dict[str, Any], lines: List[str]) -> None:
    """Render an llm.chat span's prompt, reasoning, response and token cost."""
    cap = span.get("llm_capture")
    toks = _llm_tokens(span)
    if not cap:
        lines.append(
            f"_(no local enrichment captured for this `llm.chat` span — "
            f"tokens prompt={_fmt_int(toks['prompt'])} / completion="
            f"{_fmt_int(toks['completion'])} / total={_fmt_int(toks['total'])})_"
        )
        lines.append("")
        return

    model = cap.get("model", span.get("attributes", {}).get("llm.model_name", "?"))
    api = cap.get("api", "?")
    latency = cap.get("latency_ms")
    latency_s = f"{float(latency):.1f} ms" if latency is not None else "?"
    lines.append(
        f"- **model:** `{model}` · **api:** `{api}` · **latency:** {latency_s}"
    )
    lines.append(
        f"- **tokens:** prompt **{_fmt_int(toks['prompt'])}** + completion "
        f"**{_fmt_int(toks['completion'])}** = total **{_fmt_int(toks['total'])}**"
    )
    inv = cap.get("invocation_parameters")
    if inv:
        lines.append(f"- **invocation:** `{json.dumps(inv, ensure_ascii=False)}`")
    done_reason = cap.get("done_reason") or span.get("attributes", {}).get(
        "llm.response.done_reason"
    )
    if done_reason:
        flag = "  ⚠️ TRUNCATED (hit num_predict)" if done_reason == "length" else ""
        lines.append(f"- **done_reason:** `{done_reason}`{flag}")
    lines.append("")

    messages = cap.get("messages") or []
    if messages:
        lines.append("**Prompt:**")
        lines.append("")
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            lines.append(f"_{role}:_")
            lines.append(_fence(content))
            lines.append("")

    thinking = cap.get("thinking")
    lines.append("**Reasoning / thinking:**")
    if thinking:
        lines.append(_fence(thinking))
    else:
        lines.append("_(empty — model returned no thinking block for this call)_")
    lines.append("")

    response = cap.get("response_content")
    if response is not None:
        lines.append("**Response:**")
        lines.append(_fence(response))
        lines.append("")


def render_trace_markdown(trace: Dict[str, Any]) -> str:
    """Render a loaded trace dict (Stage A's JSON shape) to a markdown string."""
    spans = trace.get("spans") or []
    ordered = _build_forest(spans)

    llm_spans = [s for s in spans if s.get("name") == _LLM_SPAN_NAME]
    run_tokens = {"prompt": 0, "completion": 0, "total": 0}
    for s in llm_spans:
        t = _llm_tokens(s)
        for k in run_tokens:
            run_tokens[k] += t[k]

    starts = [s.get("start_time_ns") for s in spans if s.get("start_time_ns")]
    ends = [s.get("end_time_ns") for s in spans if s.get("end_time_ns")]
    wall_ms = (max(ends) - min(starts)) / 1_000_000.0 if starts and ends else None

    out: List[str] = []
    out.append(f"# Trace `{trace.get('trace_id', '?')}`")
    out.append("")
    out.append(f"- **spans:** {trace.get('span_count', len(spans))}")
    out.append(f"- **llm.chat calls:** {len(llm_spans)}")
    out.append(
        f"- **run token total:** prompt **{_fmt_int(run_tokens['prompt'])}** + "
        f"completion **{_fmt_int(run_tokens['completion'])}** = "
        f"**{_fmt_int(run_tokens['total'])}**"
    )
    if wall_ms is not None:
        out.append(f"- **wall time:** {wall_ms:.1f} ms")
    out.append("")
    out.append("---")
    out.append("")
    out.append("## Execution timeline")
    out.append("")

    llm_index = 0
    for span in ordered:
        depth = span.get("_depth", 0)
        indent = "  " * depth
        label = _span_label(span)
        dur = span.get("duration_ms")
        dur_s = f" · {dur:.1f} ms" if isinstance(dur, (int, float)) else ""
        status = span.get("status")
        status_s = f" · {status}" if status and status != "OK" else ""

        if span.get("name") == _LLM_SPAN_NAME:
            llm_index += 1
            toks = _llm_tokens(span)
            out.append(
                f"{indent}- 🧠 **#{llm_index} {label}**{dur_s}{status_s} · "
                f"{_fmt_int(toks['total'])} tok"
            )
            out.append("")
            body: List[str] = []
            _render_llm_capture(span, body)
            for bl in body:
                out.append(f"{indent}  {bl}" if bl else "")
        else:
            out.append(f"{indent}- ▸ **{label}**{dur_s}{status_s}")

    out.append("")
    return "\n".join(out)


def render_trace_file(json_path: str, out_dir: Optional[str] = None) -> str:
    """Read one trace JSON and write ``<trace_id>.md`` beside it. Returns md path."""
    with open(json_path, "r", encoding="utf-8") as fh:
        trace = json.load(fh)
    md = render_trace_markdown(trace)
    trace_id = trace.get("trace_id") or os.path.splitext(os.path.basename(json_path))[0]
    out_dir = out_dir or os.path.dirname(json_path) or resolve_local_trace_dir()
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, f"{trace_id}.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)
    return md_path


def _md_is_current(json_path: str) -> bool:
    """True when ``<trace_id>.md`` already exists beside the JSON and is at least as
    new as it — i.e. nothing changed since it was last rendered. Lets a bulk render
    skip already-rendered traces instead of redoing all-time work (a shared trace dir
    accumulates tens of thousands of files; re-rendering every one on each shutdown
    turned teardown into a multi-minute stall)."""
    md_path = os.path.splitext(json_path)[0] + ".md"
    try:
        return os.path.getmtime(md_path) >= os.path.getmtime(json_path)
    except OSError:  # md missing (or json vanished) -> render it
        return False


def render_all(trace_dir: Optional[str] = None, *, force: bool = False) -> List[str]:
    """Render every ``*.json`` trace in the dir to markdown. Returns md paths.

    Skips traces whose ``.md`` is already up to date (``force=True`` re-renders all)
    so a full-directory pass stays O(new traces), not O(all-time traces) — the
    ``on_end`` hook already renders each trace as its root span closes, so a bulk
    pass only needs to fill in genuinely-unrendered files."""
    trace_dir = trace_dir or resolve_local_trace_dir()
    written: List[str] = []
    for json_path in sorted(glob.glob(os.path.join(trace_dir, "*.json"))):
        if not force and _md_is_current(json_path):
            continue
        try:
            written.append(render_trace_file(json_path))
        except Exception:  # one bad file must not abort the rest (debugging aid)
            continue
    return written


class MarkdownTraceProcessor(_SpanProcessor):
    """SpanProcessor that re-renders a trace's markdown when its root span ends.

    Wired in :mod:`agent_runtime.tracing` AFTER the JSON capture processor, so by
    the time a root span's ``on_end`` fires here the Stage-A JSON for that trace
    is already on disk and this just reads it and writes ``<trace_id>.md`` beside
    it. It NEVER touches the capture path and is fully best-effort — any failure
    is swallowed so a debugging aid can never break a model call or app startup.

    Subclasses the SDK ``SpanProcessor`` so it inherits every hook the provider
    calls — notably the no-op ``_on_ending`` the SDK invokes on each processor as
    a span ends. (A previous duck-typed version omitted ``_on_ending`` and raised
    ``AttributeError`` on the first span end, breaking the whole run; fixed in
    s7/b2.) We override only ``on_end`` (and ``shutdown`` as a backstop).
    """

    def __init__(self, trace_dir: Optional[str] = None) -> None:
        self._dir = trace_dir or resolve_local_trace_dir()

    def on_start(self, span: Any, parent_context: Any = None) -> None:  # noqa: D401
        return None

    def on_end(self, span: Any) -> None:
        try:
            # Only render once per run — when the ROOT span (no parent) closes.
            if getattr(span, "parent", None) is not None:
                return
            ctx = span.get_span_context()
            trace_id = format(ctx.trace_id, "032x")
            json_path = os.path.join(self._dir, f"{trace_id}.json")
            if os.path.exists(json_path):
                render_trace_file(json_path, self._dir)
        except Exception:  # pragma: no cover - rendering must never break export
            return None

    def shutdown(self) -> None:
        # Best-effort: on shutdown, render whatever JSON traces are on disk so a
        # run torn down before its root span flushed still gets a markdown file.
        try:
            render_all(self._dir)
        except Exception:  # pragma: no cover
            pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _main(argv: List[str]) -> int:
    args = argv[1:]
    if args:
        for path in args:
            print(render_trace_file(path))
    else:
        paths = render_all()
        for p in paths:
            print(p)
        if not paths:
            print("(no trace JSON found in", resolve_local_trace_dir(), ")")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
