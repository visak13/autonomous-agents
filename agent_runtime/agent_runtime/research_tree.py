"""s9/N4 (d62/c15/d69) — TREE-shaped research with PRUNING + the persisted-state DECISION NODE.

PHASE 1: the MECHANISM. This module is the iterative, decision-driven LAYER LOOP that
replaces the flat-linear deep-research topology with an ``a -> b1..bN -> b11..bNN``
research TREE: each layer GATHERS its live branches as leaf research nodes, a DECISION
NODE reads the PERSISTED file of ALL prior research (d49 — real state, not the model's
memory) and, by REASONING, authors this layer's tree mutations via discrete prompt-JSON
tool calls (``expand_branch`` / ``prune_branch`` + a SOFT next-direction), and the loop
STOPS cleanly when no live branches remain, no expansion is emitted, or the depth bound
is hit. The accumulated ``(findings, sources)`` then hand UNCHANGED to the c13 write side.

DESIGN INVARIANTS (the design doc ``.s9_probe/n4_design.md``, DD's proof, the recipe
decisions — honor them; they are load-bearing):

* **The tree is structured CONTROL state** (legitimate per d50.1 — only deliverable
  *content* is RAW). It is mutated **ONLY** by the model's tool calls: no template, no
  fabricated branch, no code-authored fan-out, no fallback that invents structure. If the
  model emits nothing actionable, the loop STOPS (graceful) — it never synthesizes a
  branch the model did not author. This is exactly DD's proven surface
  (``expand_branch`` / ``prune_branch``), measured 2/2 grounded + reasoned on real E4B.
* **Persisted research-state file = the single source of truth** (d49 / c1 raw read-back):
  each leaf APPENDS its ArticleNotes + a findings digest; the decision node READS it back
  to see the ACTUAL gathered state. This kills false-finish AND lets a long run survive
  without holding everything in one window.
* **Bounds are NON-FLOW** (cost/safety, never a flow gate, per d14/d48) AND **config-exposed**
  (Q-C HARD REQ — env/config, never hard-coded constants): DEPTH (baseline 5 / max 10), a
  per-layer fan-out cap (~4-5), and the leaf fetch breadth. The model DECIDES whether/where
  to expand within those ceilings.
* **Next-direction is SOFT** (DD constraint): E4B sometimes folds "what to pursue next"
  into its final plan PROSE rather than a discrete ``set_next_direction`` tool call. Accept
  it from the prose OR re-prompt ONCE; NEVER treat its absence as a planning failure. The
  expand/prune tools carry the topology; next-direction is advisory ordering.
* **temperature=0** for stable/reproducible tree authoring (DD: both trials chose the same
  4 expansions + 2 prunes).

REUSE boundary: this module is upstream of the c13 write side (UNCHANGED) and reuses the
N2 ``ArticleNote`` shape + the d49 file read/write pattern. The OFFLINE GATE is DD's
harness ``.s9_probe/dd_tree_authoring_probe.py`` (the same ``Tree`` + parse + 3-tool
surface this module ships); the unit tests here exercise the mechanism deterministically
with a scripted transport + an injected gather. The SERVED-ROUTE wiring (flag default-ON,
the agentic deep-research route) is N4w; the live >=2-layer E4B proof is N4r.

The leaf GATHER and the LLM TRANSPORT are DEPENDENCY-INJECTED so the orchestrator mechanism
is fully testable offline and the served wiring (N4w) supplies the real
``SubAgent._run_research_loop`` leaf + the resident keep_alive=-1 E4B transport.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# P2.5b (d134/d135) — the GROWABLE-DAG grower reuses these to map the decision node's
# gap-driven branches onto research PlanNodes. factory/roles/synth_tools are leaf modules
# (none import research_tree), so these top-level imports add no cycle.
from .factory import PlanDAG, PlanNode
from .roles import ROLE_WORKER, position_framing
from .synth_tools import unwrap_output_envelope

# ---------------------------------------------------------------------------- #
# Config — DEPTH, fan-out, leaf breadth are TUNEABLE via env (Q-C HARD REQ: NEVER
# hard-coded constants). Each is a NON-FLOW cost/safety ceiling, not a flow gate.
# ---------------------------------------------------------------------------- #
# The HARD depth ceiling the user fixed (max 10); the baseline default (5) sits under it.
N4_TREE_DEPTH_CEILING = 10


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    """Read an int env override, clamped to ``[lo, hi]`` (a malformed value falls back)."""
    try:
        val = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        val = default
    return max(lo, min(hi, val))


def _env_float(name: str, default: float, *, lo: float, hi: float) -> float:
    """Read a float env override, clamped to ``[lo, hi]`` (a malformed value falls back)."""
    try:
        val = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        val = default
    return max(lo, min(hi, val))


def _env_bool(name: str, default: bool) -> bool:
    """Read a bool env override (1/true/yes/on → True, 0/false/no/off → False)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class TreeConfig:
    """Config-exposed bounds for the research tree (Q-C: env/config, not constants).

    Every bound is a NON-FLOW cost/safety ceiling — the model REASONS about whether/where
    to expand within them; nothing here forces work. ``depth`` is clamped to the hard
    ``N4_TREE_DEPTH_CEILING`` (10) the user fixed; ``fan_out`` caps how many expansions a
    single decision layer may keep (~4-5, matching DD's 4); ``leaf_breadth`` is the per-leaf
    fetch budget handed to the leaf research node (the N1 breadth knob)."""

    depth: int = 5                 # baseline 5, clamped <= ceiling (max 10)
    fan_out: int = 5               # per-layer expansion cap (~4-5, DD authored 4)
    leaf_breadth: int = 10         # per-leaf web_fetch budget (N1 DEEP_RESEARCH_FETCH_BREADTH)
    decide_max_turns: int = 18     # ReAct turns the decision node may take (DD used 18)
    num_ctx: int = 32768           # E4B SWA window (d35/d22 — 32768 proven regime)
    num_predict: int = 4096        # d7: load-bearing with think=True (<=512 -> empty)
    # d106 #3 (SEED-ONLY/SCOPED ROOT): the root DECOMPOSES the goal into scoped sub-questions
    # FIRST (via expand_branch tool calls) and is NOT itself gathered as a whole-goal research
    # leaf — the B8a root burned 70min/0-yield fetching the entire goal. Flag-free DEFAULT (the
    # fix, d65): on when unset; RA_TREE_SEED_ONLY_ROOT=0 only to isolate loop-internals in tests.
    seed_only_root: bool = True
    # d106 #4 (SHORT per-node timeout): a single leaf research node is bounded WELL BELOW the
    # shared run timeout (~70min) so one stuck node cannot burn the whole budget; a node that
    # exceeds it is cancelled + marked UNSUPPORTED (fed to the writer, no fabrication). 420s=7min.
    node_timeout: float = 420.0
    # P2-5c FORWARD HARDENING — a per-engine WALL-CLOCK budget on the GROWABLE drive loop.
    # When >0, the runtime stops AUTHORING further growth layers once this many seconds have
    # elapsed since the seed wave and returns the findings GATHERED SO FAR with
    # ``stop_reason='budget'`` — a GRACEFUL partial, never an exception/abort. 0 = OFF (no
    # wall-clock bound; the loop still stops on agent_sufficient / no_expansion / max_layers).
    # The served generic report path derives a sensible budget from the run timeout when this
    # is unset; tests pin it explicitly to prove the graceful stop fires.
    grow_wallclock_budget: float = 0.0

    @classmethod
    def from_env(cls) -> "TreeConfig":
        """Build the config from env overrides (live-tuneable per d60/Q-C, no code edit).

        ``RA_TREE_DEPTH`` (baseline depth, clamped to [1, 10]), ``RA_TREE_FANOUT``
        (per-layer expansion cap), ``RA_TREE_LEAF_BREADTH`` (per-leaf fetch budget;
        defaults to the shared ``RA_RESEARCH_FETCH_BREADTH`` so it tracks N1 when unset),
        ``RA_TREE_DECIDE_MAX_TURNS``, ``RA_RESEARCH_NUM_CTX``, ``RA_TREE_NUM_PREDICT``."""
        leaf_default = _env_int("RA_RESEARCH_FETCH_BREADTH", 10, lo=1, hi=64)
        return cls(
            depth=_env_int("RA_TREE_DEPTH", 5, lo=1, hi=N4_TREE_DEPTH_CEILING),
            fan_out=_env_int("RA_TREE_FANOUT", 5, lo=1, hi=32),
            leaf_breadth=_env_int("RA_TREE_LEAF_BREADTH", leaf_default, lo=1, hi=64),
            decide_max_turns=_env_int("RA_TREE_DECIDE_MAX_TURNS", 18, lo=2, hi=64),
            num_ctx=_env_int("RA_RESEARCH_NUM_CTX", 32768, lo=8192, hi=131072),
            num_predict=_env_int("RA_TREE_NUM_PREDICT", 4096, lo=512, hi=16384),
            # d106 #3/#4 — the fix is the flag-free served DEFAULT (unset → on / 7-min bound).
            seed_only_root=_env_bool("RA_TREE_SEED_ONLY_ROOT", True),
            node_timeout=_env_float("RA_TREE_NODE_TIMEOUT", 420.0, lo=30.0, hi=3600.0),
            # P2-5c — wall-clock budget on the growable loop (0 = off; the served report
            # path derives one from the run timeout when this env is unset).
            grow_wallclock_budget=_env_float(
                "RA_GROW_WALLCLOCK_BUDGET_S", 0.0, lo=0.0, hi=86400.0
            ),
        )


# ---------------------------------------------------------------------------- #
# Branch + Tree — the structured CONTROL state, mutated ONLY by model tool calls.
# ---------------------------------------------------------------------------- #
@dataclass
class Branch:
    """One research direction the model authored (a node of the tree). ``depth`` is the
    layer it lives on (root = 0); ``rationale`` cites the note gap that justified it."""

    id: str
    parent: str
    question: str
    rationale: str = ""
    depth: int = 1


class Tree:
    """In-memory branch+prune research tree, mutated ONLY by the model's tool calls.

    Promoted verbatim in spirit from DD's proven ``Tree`` (the harness that measured 2/2
    grounded authoring on E4B), with per-branch DEPTH tracking added for the layer loop.
    There is NO method that authors a branch from anything but an ``expand``/``prune`` call
    — the orchestrator can only relay the model's calls here, never fabricate structure."""

    def __init__(self, *, fan_out: Optional[int] = None) -> None:
        self.branches: dict[str, Branch] = {}
        self.pruned: list[dict[str, str]] = []
        self.next_direction: Optional[dict[str, str]] = None
        self._fan_out = fan_out          # per-layer expansion cap (None = unbounded)
        self._n = 0
        self._layer_expansions = 0       # expansions kept in the CURRENT decision layer
        # s13/B3 — the OutlinePlan (AGENT-DECIDED DOCUMENT DIRECTION). ``_outline`` is the
        # EFFECTIVE section list (ordered by first add, a later add to the same title updates
        # its ``covers``, a drop removes it); ``outline_ops`` is the APPEND-ONLY op log the
        # layer loop persists to the ResearchState outline channel. The agent mutates this
        # ONLY via add_section/drop_section tool calls — no code-authored / template section.
        self._outline: dict[str, dict[str, str]] = {}
        self.outline_ops: list[dict[str, str]] = []

    def add_section(self, args: Mapping[str, Any]) -> str:
        """Author/refine ONE outline section from the model's ``add_section`` call → an ack.

        Append-only: records the op for persistence and upserts the EFFECTIVE outline (a
        repeat title refines its ``covers``). Never fabricates — the orchestrator only relays
        the model's call. A blank title is ignored (acknowledged, not kept)."""
        title = str(args.get("title", "")).strip()
        covers = str(args.get("covers", "")).strip()
        if not title:
            return (
                "add_section needs a non-empty title — not kept. Current outline: "
                f"{[s['title'] for s in self._outline.values()]}."
            )
        self.outline_ops.append({"op": "add", "title": title, "covers": covers})
        self._outline[title] = {"title": title, "covers": covers}
        return (
            f"OK section {title!r} (covers: {covers!r}) is in the outline. "
            f"Current outline: {[s['title'] for s in self._outline.values()]}. "
            f"Refine the outline, continue research, or write your plan."
        )

    def drop_section(self, args: Mapping[str, Any]) -> str:
        """Drop an outline section from the model's ``drop_section`` call → an ack.

        Append-only: records the drop op (persisted) and removes the title from the EFFECTIVE
        outline. Dropping an absent title is a harmless no-op (still recorded)."""
        title = str(args.get("title", "")).strip()
        reason = str(args.get("reason", "")).strip()
        self.outline_ops.append({"op": "drop", "title": title, "reason": reason})
        self._outline.pop(title, None)
        return (
            f"OK dropped section {title!r} ({reason!r}). "
            f"Current outline: {[s['title'] for s in self._outline.values()]}. "
            f"Refine the outline, continue research, or write your plan."
        )

    def outline(self) -> list[dict[str, str]]:
        """The EFFECTIVE outline (ordered list of ``{title, covers}``, drops removed)."""
        return [dict(s) for s in self._outline.values()]

    def begin_layer(self) -> None:
        """Reset the per-layer expansion counter (the fan-out cap is per decision layer)."""
        self._layer_expansions = 0

    def expand(self, args: Mapping[str, Any], *, depth: int = 1) -> str:
        """Author ONE branch from the model's ``expand_branch`` call → an observation ack.

        Honors the per-layer fan-out cap as a NON-FLOW ceiling: beyond the cap the call is
        acknowledged but the branch is NOT kept (the model is told the cap is reached), so
        the tree never grows past the configured breadth even if the model keeps asking."""
        if self._fan_out is not None and self._layer_expansions >= self._fan_out:
            return (
                f"Fan-out cap ({self._fan_out}) reached for this layer — that expansion was "
                f"not kept. Prune a weak note, set the next direction, or write your plan."
            )
        # s13 P1-review hardening: a native expand_branch whose args were empty/malformed
        # (first_native_call coerces non-dict args to {}) would otherwise author a
        # degenerate empty-question branch that later gets gathered for nothing. Treat a
        # blank question as a non-flow ack (mirror the fan-out-cap path) — keep no branch,
        # spend no counter — so the model retries with a real question or writes its plan.
        question = str(args.get("question", "")).strip()
        if not question:
            return (
                "That expand_branch had no question — nothing was added. Provide a "
                "question to research, prune a weak note, or write your plan."
            )
        self._n += 1
        self._layer_expansions += 1
        bid = f"B{self._n}"
        branch = Branch(
            id=bid,
            parent=str(args.get("parent", "root")).strip() or "root",
            question=question,
            rationale=str(args.get("rationale", "")).strip(),
            depth=depth,
        )
        self.branches[bid] = branch
        return (
            f"OK expanded {bid} under {branch.parent}: {branch.question!r}. "
            f"Live branches: {list(self.branches)}. Continue, or write your plan."
        )

    def prune(self, args: Mapping[str, Any]) -> str:
        """Kill a branch/note from the model's ``prune_branch`` call → an observation ack."""
        tgt = str(args.get("branch", "")).strip()
        reason = str(args.get("reason", "")).strip()
        self.pruned.append({"target": tgt, "reason": reason})
        self.branches.pop(tgt, None)
        return (
            f"OK pruned {tgt!r} ({reason!r}). Live branches: {list(self.branches)}. "
            f"Continue, or write your plan."
        )

    def set_next(self, args: Mapping[str, Any]) -> str:
        """Record the (advisory) next direction from a discrete ``set_next_direction`` call."""
        self.next_direction = {
            "branch": str(args.get("branch", "")).strip(),
            "reason": str(args.get("reason", "")).strip(),
        }
        return (
            f"OK next direction = {self.next_direction['branch']!r}. "
            f"Write your final tree plan now, or refine further."
        )

    def live_branches(self) -> list[Branch]:
        return list(self.branches.values())

    def snapshot(self) -> dict[str, Any]:
        return {
            "branches": [vars(b) for b in self.branches.values()],
            "pruned": list(self.pruned),
            "next_direction": self.next_direction,
            "outline": self.outline(),   # s13/B3 — the agent-decided document direction
        }


# ---------------------------------------------------------------------------- #
# Prompt-JSON parse — the 3 tree tools (replica of runtime._parse_research_call; the
# SAME parse DD proved). A non-tool turn is the model's FINAL prose plan (loop ends).
# ---------------------------------------------------------------------------- #
TREE_TOOLS: tuple[str, ...] = (
    "expand_branch", "prune_branch", "set_next_direction", "stop_research",
    # s13/B3 — the AGENT-DECIDED DOCUMENT DIRECTION channel (design §3, the faithfulness
    # linchpin): low-arity outline tools that author/refine the document's section plan.
    # They mutate the tree's OutlinePlan (NOT the research topology) and are persisted to the
    # ResearchState outline channel + read back each layer (same anti-hallucination read-back
    # as the leaf notes), then handed to PHASE-2 as the primary section scaffold.
    "add_section", "drop_section",
)


def _first_json_object(s: str) -> Optional[str]:
    """Return the first balanced ``{...}`` object in ``s`` (brace-depth scan), or None."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _strip_fence(s: str) -> str:
    """Strip a leading/trailing ```` ``` ```` code fence (the model often fences its JSON)."""
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s
        if s.endswith("```"):
            s = s[:-3]
        s = s.replace("```json", "").replace("```", "")
    return s.strip()


def parse_tree_call(raw: str) -> Optional[tuple[str, dict[str, Any]]]:
    """Recover a lightweight ``(tool, args)`` tree call from a turn, or None.

    A TOOL turn is a bare JSON object (the instruction asks for ONLY the JSON); any other
    turn is the model's FINAL prose plan (returns None → loop ends). Only the recognized
    ``TREE_TOOLS`` are accepted; an unparseable or unknown object is treated as prose, never a
    silently-dispatched call. Mirrors ``runtime._parse_research_call`` so the surface the
    DD harness proved is the surface that ships."""
    s = _strip_fence(raw or "").strip()
    if not s.startswith("{"):
        return None
    blob = _first_json_object(s)
    if not blob:
        return None
    try:
        parsed = json.loads(blob)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, Mapping):
        return None
    tool = parsed.get("tool") or parsed.get("name") or parsed.get("tool_name")
    args = parsed.get("args") or parsed.get("arguments") or parsed.get("parameters")
    if not (isinstance(tool, str) and tool.strip()):
        # Bare {<tool_name>: {...args}} slip.
        for key, val in parsed.items():
            if str(key).strip() in TREE_TOOLS:
                tool, args = str(key).strip(), val
                break
    name = str(tool).strip() if isinstance(tool, str) else ""
    if name not in TREE_TOOLS:
        return None
    if not isinstance(args, Mapping):
        args = {
            k: v for k, v in parsed.items()
            if k not in ("tool", "name", "tool_name", "args", "arguments", "parameters")
        }
    return name, dict(args)


# ---------------------------------------------------------------------------- #
# NATIVE tool-call layer (s13/P1) — pass real tool schemas to Ollama and READ
# ``message.tool_calls`` instead of string-parsing a "reply with ONLY JSON" prose
# turn. The tool call rides its OWN response channel, so leading prose can never
# swallow it (the d114 probe measured 19/19 native vs 4/19 missed on string parse).
# ``parse_tree_call`` above is KEPT as the defensive fallback for any non-native reply.
# ---------------------------------------------------------------------------- #
def make_tool_spec(
    name: str, description: str,
    properties: Mapping[str, Any], required: Sequence[str],
) -> dict[str, Any]:
    """Build ONE native Ollama/OpenAI tool schema (``{"type":"function","function":...}``)."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": dict(properties),
                "required": list(required),
            },
        },
    }


# The decision/decompose tools as native schemas (mirror TREE_TOOLS + _DECISION_INSTRUCTION
# arg shapes). Passed as ``tools=[...]`` so the model emits real tool_calls.
# These descriptions CARRY the research FLOW so it is legible from the tool surface
# itself (d125/d126/d133 tool-drives-the-flow): IDENTIFY the thesis -> EXPAND into
# the gaps along what/why/when/how -> PRUNE bad/off-thesis leads -> STOP when the
# notes answer the thesis with no meaning-adding gap left. The hardcoded
# _DECISION/_DECOMPOSE prompts are KEPT as a backstop, but the tools say what they
# are FOR so an agent can drive the loop from the descriptions alone.
TREE_TOOL_SPECS: tuple[dict[str, Any], ...] = (
    make_tool_spec(
        "expand_branch",
        "EXPAND the research into a GAP. Author one focused, scoped child question "
        "that fills a MISSING meaning the report needs — probe the next unanswered "
        "WHAT / WHY / WHEN / HOW of the thesis (e.g. a missing timeline event, an "
        "unquantified cost, an unexplained cause). Each child must be answerable by "
        "ONE focused research node; do not restate the whole goal.",
        {"parent": {"type": "string"}, "question": {"type": "string"},
         "rationale": {"type": "string"}},
        ["question"],
    ),
    make_tool_spec(
        "prune_branch",
        "PRUNE a bad lead. Cut a branch/note that ADDS NO MEANING — redundant, "
        "off-thesis, already answered, a dead end, or low-trust — so the budget goes "
        "to gaps that matter. Give the reason it fails the meaning test.",
        {"branch": {"type": "string"}, "reason": {"type": "string"}},
        ["branch"],
    ),
    make_tool_spec(
        "set_next_direction",
        "PRIORITISE the next move. Pick which live branch to pursue next — the one "
        "whose gap, once filled, most advances the thesis.",
        {"branch": {"type": "string"}, "reason": {"type": "string"}},
        ["branch"],
    ),
    make_tool_spec(
        "stop_research",
        "STOP when SUFFICIENT. Call this once the gathered notes answer the thesis "
        "with NO meaning-adding gap left (the blanks are filled) — not at an "
        "arbitrary depth. State briefly why coverage is now complete.",
        {"reason": {"type": "string"}},
        [],
    ),
    make_tool_spec(
        "add_section",
        "SHAPE the report. Add a section the final report should have, grounded in "
        "the notes it will cover — let the outline EMERGE from what the research "
        "actually found.",
        {"title": {"type": "string"}, "covers": {"type": "string"}},
        ["title"],
    ),
    make_tool_spec(
        "drop_section",
        "SHAPE the report. Drop an outlined section the gathered notes cannot "
        "support, so the report never promises a section it has no evidence for.",
        {"title": {"type": "string"}, "reason": {"type": "string"}},
        ["title"],
    ),
)


def first_native_call(
    tool_calls: Optional[Sequence[Mapping[str, Any]]],
    accepted: Sequence[str],
) -> Optional[tuple[str, dict[str, Any]]]:
    """Return the first ``(tool, args)`` from NATIVE ``message.tool_calls`` whose name is
    in ``accepted``, or None.

    ``tool_calls`` is the transport-normalised list (``[{"name","arguments"}, ...]``).
    None/empty (a plain-prose reply, or a transport that returned no tool_calls) → None,
    so the caller falls through to the balanced-brace string parser — the s13 defensive
    fallback that keeps every non-native path working unchanged."""
    if not tool_calls:
        return None
    accepted_set = set(accepted)
    for tc in tool_calls:
        if not isinstance(tc, Mapping):
            continue
        name = str(tc.get("name") or "").strip()
        if name in accepted_set:
            args = tc.get("arguments")
            return name, (dict(args) if isinstance(args, Mapping) else {})
    return None


# ---------------------------------------------------------------------------- #
# Leaf gather contract — what a leaf research node returns to the orchestrator.
# ---------------------------------------------------------------------------- #
@dataclass
class LeafResult:
    """The output of dispatching ONE branch as a leaf research node (the N4 GATHER step).

    ``findings`` = the leaf's RAW prose findings (content, d50.1); ``notes`` = the N2
    ArticleNotes (control lane the decision node reads); ``fetched`` = the read source
    records (``{title,url,markdown,...}``) that flow UNCHANGED to the c13 write side."""

    branch_id: str
    question: str
    findings: str = ""
    notes: list[dict[str, Any]] = field(default_factory=list)
    fetched: list[dict[str, Any]] = field(default_factory=list)


def _split_sentences(text: str) -> list[str]:
    """Split prose into coarse sentence-like segments (regex-free, deterministic).

    Used by the findings bridge (s13 P1-findings) to derive a *presence* claim signal
    from a leaf's findings_digest when it emitted no structured notes — not an exact
    parse, just enough to render a non-zero contribution the decision node can see."""
    segs: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in ".!?\n":
            seg = "".join(buf).strip()
            if seg:
                segs.append(seg)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        segs.append(tail)
    return segs


# ---------------------------------------------------------------------------- #
# Persisted research-state file — the single source of truth (d49 read-real-state).
# ---------------------------------------------------------------------------- #
class ResearchState:
    """The run-scoped persisted research-state file (d49 / c1 raw read-back pattern).

    Each leaf APPENDS its ArticleNotes + a findings digest as one JSON record per line
    (append-only, so a long run survives without holding everything in one window). The
    decision node READS the file back (real state, not the model's memory) and renders it
    compactly for its prompt. Provenance is owned by the runtime (the leaf supplies the
    notes the leaf's own research produced) — the decision node never invents a source."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate any stale prior-run file so the decision node reads only THIS run.
        self.path.write_text("", encoding="utf-8")
        self._records: list[dict[str, Any]] = []
        # s13/B3 — the OUTLINE channel: an append-only sidecar of the agent's document-
        # direction ops (add_section / drop_section), persisted ALONGSIDE the leaf state and
        # read BACK from disk into each decision layer (the same anti-hallucination read-back
        # as the leaf notes — the rendered outline comes from disk, never the model's memory).
        self.outline_path = self.path.parent / (self.path.stem + ".outline.jsonl")
        self.outline_path.write_text("", encoding="utf-8")

    def append_outline_ops(self, ops: Sequence[Mapping[str, Any]]) -> None:
        """Append the decision layer's outline ops (add_section/drop_section) to disk (B3)."""
        if not ops:
            return
        with self.outline_path.open("a", encoding="utf-8") as fh:
            for op in ops:
                fh.write(json.dumps(dict(op), default=str) + "\n")

    def read_outline(self) -> list[dict[str, str]]:
        """Read the outline ops BACK from disk and FOLD them into the effective outline.

        Append-only semantics (mirrors Tree.outline): ``add`` upserts a section (preserving
        first-seen order, refreshing ``covers``); ``drop`` removes it. Returns the ordered
        ``[{title, covers}]`` the document direction actually settled on — read from disk so
        the outline that reaches the prompt/write phase is the persisted state, not memory."""
        effective: dict[str, dict[str, str]] = {}
        if not self.outline_path.exists():
            return []
        for line in self.outline_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                op = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(op, Mapping):
                continue
            title = str(op.get("title", "")).strip()
            if not title:
                continue
            if op.get("op") == "drop":
                effective.pop(title, None)
            else:  # "add" (default)
                effective[title] = {"title": title, "covers": str(op.get("covers", "")).strip()}
        return list(effective.values())

    def render_outline_for_decision(
        self, outline: Optional[Sequence[Mapping[str, Any]]] = None
    ) -> str:
        """Render the current document outline for the decision prompt (read-back, B3).

        Reads from disk when ``outline`` is not supplied (d49 real state). An empty outline
        renders an explicit prompt to PROPOSE sections — never a fabricated default."""
        outline = list(outline) if outline is not None else self.read_outline()
        if not outline:
            return (
                "DOCUMENT OUTLINE (sections planned so far): (none yet — as the notes reveal "
                "the report's shape, propose sections with add_section / refine with drop_section)"
            )
        lines = ["DOCUMENT OUTLINE (sections planned so far):"]
        for i, sec in enumerate(outline, 1):
            title = str(sec.get("title", "")).strip()
            covers = str(sec.get("covers", "")).strip()
            lines.append(f"  {i}. {title}" + (f" — covers: {covers}" if covers else ""))
        return "\n".join(lines)

    def append_leaf(self, leaf: LeafResult, *, layer: int) -> None:
        """Append ONE leaf's gathered state (notes + findings digest) to the file (d49)."""
        record = {
            "layer": layer,
            "branch_id": leaf.branch_id,
            "question": leaf.question,
            "findings_digest": (leaf.findings or "")[:1200],
            "notes": list(leaf.notes or []),
            "fetched_count": len(leaf.fetched or []),
        }
        self._records.append(record)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def read(self) -> list[dict[str, Any]]:
        """Read the persisted state BACK from disk (d49 — the ACTUAL state, not memory)."""
        records: list[dict[str, Any]] = []
        if not self.path.exists():
            return records
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except (ValueError, TypeError):
                continue
        return records

    def render_for_decision(self, records: Optional[Sequence[Mapping[str, Any]]] = None) -> str:
        """Render the persisted notes compactly for the decision prompt (512-SWA-friendly).

        Shows each note's id, trust tier, key_claims and gaps — the signal the decision node
        reasons over to author expansions (grounded in gaps) and prunes (weak/redundant
        notes). Reads from disk when ``records`` is not supplied (d49 real state)."""
        records = list(records) if records is not None else self.read()
        if not records:
            return "ARTICLE NOTES (already gathered): (none yet)"
        lines = ["ARTICLE NOTES (already gathered):"]
        for rec in records:
            q = str(rec.get("question", "")).strip()
            notes = rec.get("notes") or []
            lines.append(f"# layer {rec.get('layer')} · branch {rec.get('branch_id')}: {q}")
            # Per-branch MEANING signal (2a) — computed from the PERSISTED record only (no
            # new fetch): how many claims/sources this branch contributed and at what trust.
            # Lets the decision node judge whether a branch ADDS MEANING or is prune-worthy.
            n_claims = sum(len(note.get("key_claims") or []) for note in notes)
            tiers = sorted({str(note.get("source_trust", "")).strip()
                            for note in notes if str(note.get("source_trust", "")).strip()})
            trust = "/".join(tiers) if tiers else "n/a"
            if notes:
                lines.append(
                    f"    contributes: {n_claims} claims, {len(notes)} sources, trust={trust}"
                )
            else:
                # s13 P1-findings — FINDINGS BRIDGE. A leaf can produce REAL findings (the
                # raw research prose) yet emit 0 ArticleNotes (the small model under-emits
                # the structured note tool). The notes-only signal above would then read
                # "0 claims, 0 sources" for a branch full of real content, and the decision
                # node would PRUNE/STOP blind. So when notes are empty, derive a NON-ZERO
                # contribution from the PERSISTED findings_digest + fetched_count (both real,
                # never fabricated): a node that gathered findings can NEVER read "0 sources".
                findings_digest = str(rec.get("findings_digest", "") or "").strip()
                fetched_count = int(rec.get("fetched_count", 0) or 0)
                if findings_digest:
                    # Coarse claim floor from the findings prose (sentence-like segments),
                    # at least 1 — a presence signal, not a fabricated note count.
                    segs = [s for s in _split_sentences(findings_digest) if len(s) > 20]
                    derived_claims = max(1, len(segs))
                    # Sources = the real fetched sources the leaf READ. If a leaf produced
                    # findings without recording a fetch count, still floor at 1 (it read
                    # SOMETHING to write findings) so the branch is never shown as empty.
                    derived_sources = max(1, fetched_count)
                    lines.append(
                        f"    contributes: {derived_claims} claims (from findings), "
                        f"{derived_sources} sources, trust=findings (notes not emitted)"
                    )
                    # Let the planner SEE the actual findings content it must decide on —
                    # a compact digest, so the decision is grounded in real data, not a 0.
                    digest = findings_digest if len(findings_digest) <= 500 \
                        else findings_digest[:500].rstrip() + "…"
                    lines.append(f"    findings: {digest}")
                else:
                    # Genuinely empty branch (no notes AND no findings) — honest 0/0.
                    lines.append(
                        f"    contributes: {n_claims} claims, {len(notes)} sources, trust={trust}"
                    )
            for note in notes:
                sid = note.get("source_id")
                trust = note.get("source_trust", "")
                title = str(note.get("title", "")).strip()
                claims = note.get("key_claims") or []
                gaps = note.get("gaps_or_followups") or []
                lines.append(
                    f"- S{sid} [{trust}] {title}\n"
                    f"    key_claims: {claims}\n"
                    f"    gaps_or_followups: {gaps}"
                )
        return "\n".join(lines)

    def all_fetched(self) -> list[dict[str, Any]]:
        """(Accumulated count helper.) Records hold ``fetched_count``; the real source
        records are carried by the orchestrator (below) for the c13 write-side contract."""
        return self._records


# ---------------------------------------------------------------------------- #
# The DECISION NODE — reasoning-driven, persisted-state-fed, DD-informed protocol.
# ---------------------------------------------------------------------------- #
_DECISION_INSTRUCTION = (
    "----\n"
    "You are a RESEARCH PLANNER growing a research TREE for a deep, well-sourced report. "
    "The ARTICLE NOTES below are everything gathered SO FAR; each branch shows what it "
    "CONTRIBUTES (claims, sources, trust). Judge every note/branch by ONE test — does it "
    "ADD MEANING to the report's thesis: a distinct claim, corroboration, a contradiction, "
    "or a concrete figure/date? PRUNE whatever is redundant, off-thesis, answered, "
    "dead-end, or low-trust. EXPAND only where a MISSING meaning would change the report. "
    "When the notes already cover the thesis with no meaning-adding gap left, STOP — do "
    "NOT pad with more branches.\n\n"
    "Call ONE tool per turn by replying with ONLY a JSON object and NOTHING else:\n"
    '  {"tool":"expand_branch","args":{"parent":"root","question":"<focused sub-question that fills a MISSING meaning>","rationale":"<the note id/gap it comes from, e.g. S1 gap>"}}\n'
    '  {"tool":"prune_branch","args":{"branch":"<a note id like S5 or a branch id like B2>","reason":"<redundant|off-thesis|answered|dead-end|low-trust>"}}\n'
    '  {"tool":"add_section","args":{"title":"<a section the FINAL report should have>","covers":"<the claims/notes this section will cover, e.g. S1,S3 damage figures>"}}\n'
    '  {"tool":"drop_section","args":{"title":"<an outlined section the notes cannot support>","reason":"<unsupported|redundant|off-thesis>"}}\n'
    '  {"tool":"set_next_direction","args":{"branch":"<branch id>","reason":"<why pursue next>"}}\n'
    '  {"tool":"stop_research","args":{"reason":"<why the gathered notes already answer the thesis — enough, no meaning-adding gap left>"}}\n\n'
    "As the notes reveal the report's shape, SHAPE ITS OUTLINE: add_section for each section "
    "the final report should have (grounded in the notes it will cover), drop_section for a "
    "planned section the notes cannot support. The outline below is YOUR document direction — "
    "refine it; it becomes the section scaffold the writer follows. "
    "Ground EVERY expansion in a specific note's gaps_or_followups (cite the note id). "
    "After you have expanded the meaning-adding directions and pruned the weak notes, "
    "EITHER call stop_research (enough) OR write your FINAL TREE PLAN as plain prose (NOT "
    "JSON): the live branches with their questions, the pruned items with reasons, and "
    "which branch to pursue first. ONE tool call per turn."
)

# P2.4 (d131/d132.D) — the EXACT default stop sentence inside _DECISION_INSTRUCTION.
# When the deep-research SHAPE supplies a ``completeness_stop`` criterion, this single
# sentence is REPLACED by the shape's text, so the STOP SIGNAL is DEFINED IN THE SHAPE
# (reasoned over by the model), not hard-coded here. It must remain a verbatim substring
# of _DECISION_INSTRUCTION; _decision_instruction() fails fast below if it ever drifts.
_DEFAULT_STOP_SENTENCE = (
    "When the notes already cover the thesis with no meaning-adding gap left, STOP — do "
    "NOT pad with more branches."
)
assert _DEFAULT_STOP_SENTENCE in _DECISION_INSTRUCTION, (
    "research_tree: _DEFAULT_STOP_SENTENCE drifted from _DECISION_INSTRUCTION — the "
    "shape-defined completeness_stop substitution would silently no-op"
)


def _decision_instruction(stop_criteria: Optional[str] = None) -> str:
    """The decision-node instruction, with the STOP SIGNAL sourced FROM THE SHAPE (P2.4).

    ``stop_criteria`` is the selected deep-research SHAPE's ``completeness_stop`` text
    (d131/d132.D — "keep poking the gap-questions until every blank is filled"). When
    present, it REPLACES the baked default stop sentence, so stop semantics live in the
    shape file (editable, no code change) instead of this hard-coded prompt. When empty /
    None (offline tests, a shape without the field) the instruction is BYTE-IDENTICAL to
    the original ``_DECISION_INSTRUCTION`` — no behavioural change off the served path."""
    sc = (stop_criteria or "").strip()
    if not sc:
        return _DECISION_INSTRUCTION
    shape_sentence = (
        "STOP CRITERION — this is DEFINED IN THE DEEP-RESEARCH SHAPE (your stop SIGNAL, "
        f"a COMPLETENESS test, NOT an arbitrary depth cap): {sc} When that criterion is "
        "met, STOP — do NOT pad with more branches."
    )
    return _DECISION_INSTRUCTION.replace(_DEFAULT_STOP_SENTENCE, shape_sentence, 1)


_DECISION_NUDGE = (
    "Reply with EITHER one tool-call JSON object (expand_branch / prune_branch / "
    "set_next_direction) OR your final tree plan as prose."
)

_NEXT_DIRECTION_REPROMPT = (
    "Before your final plan, if you have a clear next direction, emit it as a tool call: "
    '{"tool":"set_next_direction","args":{"branch":"<branch id>","reason":"<why>"}} — '
    "otherwise just state it in your plan prose (that is fine too)."
)


def _chat_to_text(transport: Any, messages: list[dict[str, Any]], config: TreeConfig) -> str:
    """One LLM turn → raw text. The transport auto-injects AGENT_IDENTITY (d15), so no
    system kwarg is passed — exactly as DD's harness and the served app do."""
    res = transport.chat(
        list(messages), think=True, temperature=0,
        num_ctx=config.num_ctx, num_predict=config.num_predict,
    )
    return getattr(res, "content", "") or ""


def _chat_turn(
    transport: Any, messages: list[dict[str, Any]], config: TreeConfig,
    *, tools: Optional[Sequence[Mapping[str, Any]]] = None,
) -> tuple[str, Optional[list[dict[str, Any]]]]:
    """One LLM turn → ``(raw_text, native_tool_calls)`` (s13 native tool-call path).

    Passes the native ``tools=[...]`` schemas so the model can answer with a real
    ``message.tool_calls`` the transport surfaces on ``ChatResult.tool_calls``. The
    text is still returned so the caller can fall back to the balanced-brace string
    parser on a non-native reply, and so a FINAL prose plan (no tool call) is detected
    exactly as before. Same offloaded-via-``to_thread`` call shape as ``_chat_to_text``."""
    kwargs: dict[str, Any] = dict(
        think=True, temperature=0,
        num_ctx=config.num_ctx, num_predict=config.num_predict,
    )
    if tools:
        kwargs["tools"] = list(tools)
    res = transport.chat(list(messages), **kwargs)
    text = getattr(res, "content", "") or ""
    calls = getattr(res, "tool_calls", None)
    return text, calls


def _methodology_block(methodology: Optional[str]) -> str:
    """d107(1) — lead a planning prompt with the SEEDED spec's investigative methodology so
    the agent reasons over the doctrine, not a hard-coded rule. Empty/None → byte-identical
    prompt (offline tests, no spec registered)."""
    if (methodology or "").strip():
        return (
            "RESEARCH METHODOLOGY (your operating doctrine — reason over it to decide "
            f"whether to keep researching or to STOP):\n{methodology.strip()}\n\n"
        )
    return ""


@dataclass
class DecisionResult:
    """What one decision layer produced: the new branches to gather next + the snapshot.

    ``stop_research`` is the d95 AGENT-decides-enough signal: when the model calls the
    ``stop_research`` tool it carries ``{"reason": ...}`` and the layer loop halts with
    ``stop_reason='agent_sufficient'`` (None when the agent did not call it)."""

    new_branches: list[Branch]
    pruned: list[dict[str, str]]
    next_direction: Optional[dict[str, str]]
    plan_prose: str
    turns: int
    stop_research: Optional[dict[str, str]] = None
    # s13/B3 — the APPEND-ONLY outline ops the agent authored THIS layer (add_section /
    # drop_section). The layer loop persists them to the ResearchState outline channel so the
    # next layer reads the refined document direction back (anti-hallucination read-back).
    outline_ops: list[dict[str, str]] = field(default_factory=list)


async def run_decision_node(
    transport: Any,
    *,
    goal: str,
    state_render: str,
    tree: Tree,
    config: TreeConfig,
    parent_depth: int,
    methodology: Optional[str] = None,
    stop_criteria: Optional[str] = None,
) -> DecisionResult:
    """Run ONE decision layer: read state → author expand/prune via tool calls → STOP.

    The tree is mutated ONLY by the model's tool calls (no fabrication). Next-direction is
    SOFT: the model may emit a discrete ``set_next_direction`` OR fold it into its final
    plan prose — if absent we re-prompt ONCE, then accept the prose (never a planning
    failure). Returns the branches expanded THIS layer (the next frontier), the prunes, the
    advisory next-direction, and the model's final plan prose. ``transport.chat`` is
    offloaded so a slow native call does not block the event loop."""
    tree.begin_layer()
    child_depth = parent_depth + 1
    before_ids = set(tree.branches)
    before_outline_ops = len(tree.outline_ops)  # s13/B3 — slice THIS layer's outline ops
    # d107(1) — the SEEDED DEEP-RESEARCH SPEC's investigative methodology + STOP criteria
    # lead the decision prompt so the agent's stop_research judgement is REASONED OVER THE
    # SPEC, not a hard-coded rule. Empty/None (offline tests, no spec) → byte-identical prompt.
    # P2.4 (d131/d132.D) — the STOP SIGNAL comes FROM THE SHAPE's completeness_stop when the
    # served route supplies it; else the baked default (byte-identical, offline/no-shape).
    convo: list[dict[str, Any]] = [{
        "role": "user",
        "content": f"{_methodology_block(methodology)}GOAL: {goal}\n\n{state_render}\n\n{_decision_instruction(stop_criteria)}",
    }]
    plan_prose = ""
    unproductive = 0
    reprompted_next = False
    stop_research: Optional[dict[str, str]] = None
    turns = 0
    for _ in range(config.decide_max_turns):
        raw, tool_calls = await asyncio.to_thread(
            _chat_turn, transport, convo, config, tools=TREE_TOOL_SPECS)
        turns += 1
        # s13: prefer the NATIVE tool call (it rides its own channel, so leading prose
        # can never swallow it); fall back to the balanced-brace string parser for any
        # non-native reply. A reply with NEITHER is the model's FINAL prose plan.
        call = first_native_call(tool_calls, TREE_TOOLS) or parse_tree_call(raw)
        # Echo the turn into the transcript. When a native call carried EMPTY prose
        # content, render the call so the conversation memory stays coherent next turn.
        convo.append({"role": "assistant", "content": raw or (
            json.dumps({"tool": call[0], "args": call[1]}) if call else "")})
        if call is None:
            stripped = _strip_fence(raw).strip()
            if stripped:
                # Final prose plan. KEEP it immediately (a re-prompt must never discard the
                # plan the model already wrote). Next-direction is SOFT: if the model never
                # emitted a discrete set_next_direction, re-prompt ONCE for it; otherwise
                # accept the prose. Its absence is NEVER a planning failure (DD constraint).
                plan_prose = stripped
                if tree.next_direction is None and not reprompted_next:
                    reprompted_next = True
                    convo.append({"role": "user", "content": _NEXT_DIRECTION_REPROMPT})
                    continue
                break
            unproductive += 1
            if unproductive >= 2:
                break
            convo.append({"role": "user", "content": _DECISION_NUDGE})
            continue
        name, args = call
        if name == "expand_branch":
            obs = tree.expand(args, depth=child_depth)
        elif name == "prune_branch":
            obs = tree.prune(args)
        elif name == "add_section":
            # s13/B3 — author/refine the document outline. Mutates the OutlinePlan (NOT the
            # research topology) and continues the loop (not a stop). MUST sit before the
            # set_next catch-all so an outline call is never mis-routed into a next-direction.
            obs = tree.add_section(args)
        elif name == "drop_section":
            obs = tree.drop_section(args)
        elif name == "stop_research":
            # d95 — the AGENT decided the gathered notes already answer the thesis. Record
            # the reason and STOP the decision loop now (the layer loop reads this and halts
            # with stop_reason='agent_sufficient'). MUST sit before the set_next catch-all so
            # an explicit stop is never mis-routed into a next-direction.
            stop_research = {"reason": str(args.get("reason", "")).strip()}
            break
        else:
            obs = tree.set_next(args)
        convo.append({"role": "user", "content": obs})

    new_branches = [b for bid, b in tree.branches.items() if bid not in before_ids]
    return DecisionResult(
        new_branches=new_branches,
        pruned=list(tree.pruned),
        next_direction=tree.next_direction,
        plan_prose=plan_prose,
        turns=turns,
        stop_research=stop_research,
        outline_ops=list(tree.outline_ops[before_outline_ops:]),
    )


# ---------------------------------------------------------------------------- #
# SEED-ONLY / DECOMPOSE-FIRST ROOT (d106 #3) — the root authors scoped sub-questions
# BEFORE any gathering; it never runs a whole-goal research pass.
# ---------------------------------------------------------------------------- #
_DECOMPOSE_INSTRUCTION = (
    "----\n"
    "You are OPENING a deep, well-sourced research investigation. NOTHING has been gathered "
    "yet. Do NOT try to research the whole goal in one pass (that wastes the budget on a "
    "single unfocused fetch). FIRST DECOMPOSE the goal into a few SCOPED sub-questions — each "
    "a distinct facet that ONE focused research node can answer (for example: the timeline, "
    "the key events/headlines, the costs/figures, the causes, the analysis).\n\n"
    "Call ONE tool per turn by replying with ONLY a JSON object and NOTHING else:\n"
    '  {"tool":"expand_branch","args":{"parent":"root","question":"<one focused, scoped sub-question>","rationale":"<which facet of the goal it covers>"}}\n\n'
    "Author 2-5 scoped sub-questions that TOGETHER cover the goal, then write a one-line "
    "plan as plain prose (NOT JSON). ONE tool call per turn."
)


async def run_decompose_node(
    transport: Any,
    *,
    goal: str,
    tree: Tree,
    config: TreeConfig,
    methodology: Optional[str] = None,
) -> list[Branch]:
    """Seed-only root (d106 #3): the root DECOMPOSES the goal into scoped sub-questions via
    ``expand_branch`` tool calls BEFORE any gathering — it never runs a whole-goal research
    pass. Returns the authored child branches (the layer-1 frontier). Empty when the model
    authored none → the caller falls back to gathering the root (never an empty research).

    Same agentic mechanism as the decision node (parse + ``tree.expand``, one tool per turn,
    ``transport.chat`` offloaded), so the tree is mutated ONLY by the model's calls — no
    fabricated/templated decomposition (d107 non-negotiable)."""
    tree.begin_layer()
    before_ids = set(tree.branches)
    convo: list[dict[str, Any]] = [{
        "role": "user",
        "content": f"{_methodology_block(methodology)}GOAL: {goal}\n\n{_DECOMPOSE_INSTRUCTION}",
    }]
    unproductive = 0
    for _ in range(config.decide_max_turns):
        raw, tool_calls = await asyncio.to_thread(
            _chat_turn, transport, convo, config, tools=TREE_TOOL_SPECS)
        # s13: prefer the NATIVE tool call, fall back to the string parser (defensive).
        call = first_native_call(tool_calls, TREE_TOOLS) or parse_tree_call(raw)
        convo.append({"role": "assistant", "content": raw or (
            json.dumps({"tool": call[0], "args": call[1]}) if call else "")})
        if call is None:
            # A non-tool reply is the model's one-line prose plan → decomposition is done.
            if _strip_fence(raw).strip():
                break
            unproductive += 1
            if unproductive >= 2:
                break
            convo.append({"role": "user", "content": _DECISION_NUDGE})
            continue
        name, args = call
        if name == "expand_branch":
            obs = tree.expand(args, depth=1)
        else:
            # At SEED time only expansion makes sense; acknowledge any other tool without
            # stopping (no notes exist yet to prune / stop over).
            obs = (
                "Seed step: author scoped sub-questions with expand_branch (one per turn), "
                "then write your one-line plan as prose."
            )
        convo.append({"role": "user", "content": obs})
    return [b for bid, b in tree.branches.items() if bid not in before_ids]


# ---------------------------------------------------------------------------- #
# P2.5b (d134/d135) — GROWABLE-DAG grower: the iterative breadth, in the GENERIC engine.
# ---------------------------------------------------------------------------- #
def _node_question(node: Any, fallback: str) -> str:
    """The research QUESTION a node carries (its seeded ``tool_args['query']``), else goal."""
    args = getattr(node, "tool_args", None) or {}
    q = str(args.get("query", "")).strip()
    return q or fallback


class DagGrower:
    """Grows a GROWABLE :class:`~agent_runtime.factory.PlanDAG` round-by-round on note gaps.

    This is the injected dependency that lets the GENERIC ``AgentRuntime`` reproduce
    ``run_research_tree``'s ITERATIVE breadth WITHOUT a second engine (d134/d135). After a
    research wave completes, :meth:`grow`:

      1. INGESTS the just-gathered research nodes' notes/findings/sources into a persisted
         :class:`ResearchState` (the d49 read-real-state source of truth);
      2. runs the SAME :func:`run_decision_node` over that state — the model authors
         ``expand_branch`` children GROUNDED IN each note's gaps (and may ``stop_research``);
      3. MAPS each new :class:`Branch` onto a research :class:`PlanNode` (growing-visibility
         edges → every new node depends on ALL prior nodes), which the runtime appends and
         dispatches as the next wave.

    Growth STOPS — and :meth:`grow` returns an empty node list with a ``stop_reason`` — on
    ``stop_research`` (``agent_sufficient``) or ``no_expansion``; the runtime caps the loop at
    ``max_layers`` (``depth_bound``). The grower owns NO bespoke research logic: it relays the
    model's tool calls into PlanNodes exactly as the bespoke tree's layer loop does — the SAME
    ``Tree`` / ``ResearchState`` / ``run_decision_node`` surface, driven by the generic runtime.
    """

    def __init__(
        self,
        *,
        transport: Any,
        goal: str,
        spec: Optional[str],
        config: TreeConfig,
        state: ResearchState,
        tree: Tree,
        methodology: Optional[str] = None,
        stop_criteria: Optional[str] = None,
        max_layers: int = 0,
    ) -> None:
        self.transport = transport
        self.goal = goal
        self.spec = spec
        self.config = config
        self.state = state
        self.tree = tree
        self.methodology = methodology
        self.stop_criteria = stop_criteria
        # The growth bound (research layers incl. the seed). 0 → the config depth ceiling;
        # always clamped to the config depth so a shape can never exceed the user-fixed
        # depth ceiling (termination safety — bounded growth, never unbounded).
        cap = int(max_layers) if int(max_layers) > 0 else int(config.depth)
        self.max_layers = max(1, min(cap, int(config.depth)))
        # node ids already folded into the ResearchState (idempotent across re-frontier waves).
        self._consumed: set[str] = set()
        # PARITY TRACE — one record per grown decision layer + the final stop reason, so the
        # served wiring can report exactly how breadth grew (gathered/expanded/turns/stop).
        self.layers: list[dict[str, Any]] = []
        self.stop_reason: Optional[str] = None

    def _research_node(self, nid: str, question: str, depends_on: tuple[str, ...]) -> PlanNode:
        """Build ONE research PlanNode (web_search-seeded, growing-visibility) for a branch."""
        q = (question or "").strip() or self.goal
        specs = (self.spec,) if self.spec else ()
        return PlanNode(
            id=nid,
            task=f"[research · gap] {position_framing('research')}\n\n{q}",
            spec=self.spec,
            specs=specs,
            depends_on=depends_on,
            role=ROLE_WORKER,
            tool="web_search",
            tool_args={"query": q[:200]},
        )

    async def seed_layer(self) -> list[PlanNode]:
        """DECOMPOSE-FIRST seed (mirrors the tree's ``seed_only_root``, d106 #3).

        The single biggest breadth lever: the tree DECOMPOSES the goal into scoped
        sub-questions via ``run_decompose_node`` BEFORE any gathering, then gathers ALL of
        them as the layer-1 frontier — so breadth is FRONT-LOADED before the model's first
        ``stop_research`` judgement. A whole-goal seed (gather one node, then decide) loses
        that breadth when the model stops right after decomposing. This reuses the SAME
        ``run_decompose_node`` the tree uses (no fabricated decomposition) and maps each
        authored child onto an INDEPENDENT research node (the layer-1 frontier gathers
        concurrently, exactly like the tree). Returns ``[]`` when the model authors no child
        → the caller keeps the unrolled whole-goal seed (never an empty research)."""
        branches = await run_decompose_node(
            self.transport,
            goal=self.goal,
            tree=self.tree,
            config=self.config,
            methodology=self.methodology,
        )
        if not branches:
            return []
        nodes: list[PlanNode] = []
        for b in branches:
            # seed frontier nodes are INDEPENDENT (depends_on=()) so they gather concurrently;
            # the decision node reads ALL their persisted notes back (cross-layer visibility).
            node = self._research_node(f"s1_{b.id}", b.question, ())
            self._consumed.discard(node.id)
            nodes.append(node)
        return nodes

    @staticmethod
    def _is_research_node(node: Any) -> bool:
        """A GATHER node (the research position) — the one seeded with a ``web_search``."""
        return getattr(node, "tool", None) == "web_search"

    def _ingest(self, nodes: Sequence[Any], cache: Mapping[str, Any], layer: int) -> int:
        """Fold every newly-completed research node's result into the ResearchState (d49).

        Reads each node's cached :class:`~agent_runtime.runtime.SubAgentResult`: its raw
        ``tool_value`` carries the N2 ``article_notes`` (the gaps the decision node reasons
        over) + the ``fetched`` sources; ``output`` is the prose findings digest. Idempotent —
        a node is folded exactly once even if ``grow`` is reached again. Returns how many
        research nodes were folded this layer."""
        gathered = 0
        for n in nodes:
            if n.id in self._consumed or not self._is_research_node(n):
                continue
            r = cache.get(n.id)
            if r is None:  # not done (skipped/failed) → nothing to fold, but mark consumed
                self._consumed.add(n.id)
                continue
            self._consumed.add(n.id)
            tv = getattr(r, "tool_value", None)
            notes: list[dict[str, Any]] = []
            fetched: list[dict[str, Any]] = []
            if isinstance(tv, Mapping):
                notes = [dict(x) for x in (tv.get("article_notes") or []) if isinstance(x, Mapping)]
                fetched = [dict(x) for x in (tv.get("fetched") or []) if isinstance(x, Mapping)]
            findings = unwrap_output_envelope(getattr(r, "output", "") or "")
            self.state.append_leaf(
                LeafResult(
                    branch_id=str(n.id),
                    question=_node_question(n, self.goal),
                    findings=findings,
                    notes=notes,
                    fetched=fetched,
                ),
                layer=layer,
            )
            gathered += 1
        return gathered

    async def grow(
        self, dag: PlanDAG, cache: Mapping[str, Any], layer: int
    ) -> tuple[list[PlanNode], Optional[str]]:
        """Ingest the completed wave, run ONE decision layer, return the next research nodes.

        Returns ``(new_nodes, stop_reason)``: a non-empty ``new_nodes`` (and ``None`` stop)
        means dispatch the next wave; an empty list + a ``stop_reason`` means growth is done.
        Mirrors ``run_research_tree``'s layer loop (gather → decide → stop?/expand?) exactly,
        but the GATHER already happened on the generic runtime (this just folds it in)."""
        gathered = self._ingest(dag.nodes, cache, layer)
        # Read the PERSISTED state back (d49 anti-hallucination) — notes + document outline.
        state_render = (
            self.state.render_for_decision()
            + "\n\n"
            + self.state.render_outline_for_decision()
        )
        decision = await run_decision_node(
            self.transport,
            goal=self.goal,
            state_render=state_render,
            tree=self.tree,
            config=self.config,
            parent_depth=layer,
            methodology=self.methodology,
            stop_criteria=self.stop_criteria,
        )
        self.state.append_outline_ops(decision.outline_ops)
        self.layers.append({
            "layer": layer,
            "gathered": gathered,
            "expanded": [b.id for b in decision.new_branches],
            "pruned": [p.get("target") for p in decision.pruned],
            "decision_turns": decision.turns,
            "stop_research": decision.stop_research,
        })
        # STOP — the agent decided the gathered notes already answer the thesis (d95). Honored
        # BEFORE no_expansion, exactly like the tree's layer loop (an explicit "enough" wins).
        if decision.stop_research is not None:
            self.stop_reason = "agent_sufficient"
            return [], self.stop_reason
        # STOP — the model authored no new branch this layer → done (no fabricated structure).
        if not decision.new_branches:
            self.stop_reason = "no_expansion"
            return [], self.stop_reason
        # MAP each model-authored branch → a research PlanNode. GROWING VISIBILITY: depend on
        # EVERY prior node so the new layer sees all earlier findings (the §2c semantic the
        # frozen unroll already used). The next layer's gather feeds the next decision node.
        prior_ids = tuple(n.id for n in dag.nodes)
        new_nodes = [
            self._research_node(f"g{layer + 1}_{b.id}", b.question, prior_ids)
            for b in decision.new_branches
        ]
        return new_nodes, None
