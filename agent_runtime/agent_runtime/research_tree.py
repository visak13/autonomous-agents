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
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# P2.5b (d134/d135) — the GROWABLE-DAG grower reuses these to map the decision node's
# gap-driven branches onto research PlanNodes. factory/roles/synth_tools are leaf modules
# (none import research_tree), so these top-level imports add no cycle.
from .factory import PlanDAG, PlanNode
from .plan_tools import NEW_MEMORY_SENTINEL, normalize_brief_memory_index
from .roles import ROLE_WORKER, position_framing
from specialization.seed import RESEARCH_METHODOLOGY_SPEC
from .synth_tools import select_relevant_excerpt, unwrap_output_envelope

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
    # d285 SB-3 — the research-MEMORY this seed branch's brief works in: an existing
    # index to CONTINUE, or the textual ``<<NEW>>`` sentinel to start a fresh line. The
    # model authors it on its ``expand_branch`` call (default <<NEW>> — a seed branch
    # opens a new research line); carried onto the branch's PlanNode brief. The actual
    # per-branch memory OPENING (vs the run's shared memory) is SB-4 — SB-3 only carries
    # the planner-authored choice on the brief surface.
    memory_index: str = NEW_MEMORY_SENTINEL


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
            # d285 SB-3 — the planner-authored research-memory choice for this branch's
            # brief (an index to continue, or <<NEW>>); canonicalized (empty → <<NEW>>).
            memory_index=normalize_brief_memory_index(args.get("memory_index")),
        )
        self.branches[bid] = branch
        # BUDGET-AS-DATA (autonomy rebuild P4): the fan-out cap is VISIBLE remaining
        # budget on every ack — the model prioritizes its highest-value expansions
        # instead of discovering the invisible cap only after a swallowed call.
        left = (
            "" if self._fan_out is None else
            f" Branch budget: {max(0, self._fan_out - self._layer_expansions)} more "
            "expansion(s) may be kept this layer — open the most valuable ones first."
        )
        return (
            f"OK expanded {bid} under {branch.parent}: {branch.question!r}. "
            f"Live branches: {list(self.branches)}.{left} Continue, or write your plan."
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
    # s14/a12 (d154) — the s13/B3 add_section / drop_section OUTLINE tools were REMOVED from the
    # served decision-node surface because they were a DEAD source_id channel. The generic
    # engine discards the tree-authored outline (PHASE-2 runs outline_hint=None, d56), so the
    # model was offered TWO surfaces to author source_ids on — add_section (whose source_ids arg
    # is SILENTLY DROPPED) and the file_write nodes (live, gate-checked) — and routed between
    # them arbitrarily (a7 vanished 0/9, a9 11/13-variable). Dropping the dead surface leaves the
    # file_write write-planner as the SOLE source_id surface, so authoring is deterministic. The
    # Tree.add_section / drop_section methods + ResearchState outline channel are KEPT (inert,
    # still unit-tested in isolation) but are no longer OFFERED to the model.
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


# d222 trace finding — E4B routes its tree/note tool calls through the STRING-parse path
# (not the native tool_calls channel), so a single malformed escape silently DROPS the call.
# The working trace (77b2…) showed a research note lost TWICE to ``the conflict\'s instability``
# (``\'`` is valid in Python/JS but ILLEGAL in JSON → json.loads fails → treated as prose →
# note_gate re-asks → the retry reproduces the SAME bad escape), plus a stray ``<tool_call|>``
# special token appended to the object. This repair makes those recoverable.
_MODEL_SPECIAL_TOKEN_RE = re.compile(
    r"<\|?/?(?:tool_call|end_of_turn|start_of_turn|eos|bos|im_end|im_start)\|?>",
    re.IGNORECASE,
)


def repair_model_json(blob: str) -> str:
    """Best-effort repair of a small model's malformed JSON tool call (d222).

    Fixes the two recurring E4B malformations the trace surfaced, so ``json.loads`` can
    recover the call instead of silently dropping it:
      * an illegal ``\\'`` escape inside a string → a literal apostrophe (``\\'`` is never
        valid JSON, so this is always safe);
      * a stray model special token (e.g. ``<tool_call|>``) left in the object.
    Idempotent and safe on already-valid JSON (neither pattern occurs in well-formed JSON)."""
    s = blob or ""
    s = _MODEL_SPECIAL_TOKEN_RE.sub("", s)
    s = s.replace("\\'", "'")
    return s


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
        # d222 — retry after repairing E4B's common malformations (illegal \' escape /
        # stray special token) so a single bad escape never silently drops a tree call.
        try:
            parsed = json.loads(repair_model_json(blob))
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
# d184 — these descriptions CARRY the CANONICAL RESEARCH LOOP so it is legible from the tool
# surface itself (tool-drives-the-flow, no force-N / no min-layers seatbelt): identify the
# concerns -> search -> read MULTIPLE relevant chunks -> NOTE what was learned + the gaps ->
# EXPAND a concern (which COMMITS to a new gathered round) -> PRUNE a concern that added no
# meaning -> STOP only when every concern is settled-noted OR collapsed and no new concern
# remains. The hardcoded _DECISION/_DECOMPOSE prompts are KEPT as a backstop, but the tools
# say what they are FOR — and, crucially, that expanding is a COMMITMENT TO GATHER (so a pass
# that expands cannot also stop) — so an agent can drive the loop from the descriptions alone.
TREE_TOOL_SPECS: tuple[dict[str, Any], ...] = (
    make_tool_spec(
        "expand_branch",
        "EXPAND a concern into the NEXT research ROUND. Author one focused, scoped child "
        "question for a MISSING meaning the report still needs — the next unanswered "
        "WHAT / WHY / WHEN / HOW (a missing timeline event, an unquantified cost, an "
        "unexplained cause, a figure only one source gave). Expanding COMMITS TO GATHER: this "
        "child WILL be run as a new research round (search -> read -> note) and its findings "
        "come back on a LATER turn — so you CANNOT expand and stop in the same pass; an "
        "expansion means there is still work to gather. Each child must be answerable by ONE "
        "focused research node; do not restate the whole goal. Set 'memory_index' to the "
        "research memory this branch works in: an existing INDEX to CONTINUE that research "
        "line, or \"<<NEW>>\" / leave empty to start a FRESH line (the default for a new "
        "seed facet).",
        {"parent": {"type": "string"}, "question": {"type": "string"},
         "rationale": {"type": "string"}, "memory_index": {"type": "string"}},
        ["question"],
    ),
    make_tool_spec(
        "prune_branch",
        "PRUNE — COLLAPSE a concern that added no meaning. A normal, expected move every "
        "layer: cut a branch/note that is redundant, off-thesis, already answered, a dead end, "
        "or low-trust so the budget funds the concerns that still matter. A concern is either "
        "settled by a note OR collapsed here — pruning is how you close one out without "
        "gathering it further. Name the reason it fails the meaning test.",
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
        "STOP — only when EVERY concern is SETTLED (its meaning is recorded in a note) OR has "
        "been COLLAPSED with prune_branch, AND no new concern remains. A concern resting on a "
        "single source, a figure without its surrounding detail, or an open follow-up is NOT "
        "settled — expand it instead. If you authored ANY expand_branch this pass you are NOT "
        "done: those rounds have not gathered yet, so there is nothing to stop on. State "
        "briefly why every concern is now settled or collapsed.",
        {"reason": {"type": "string"}},
        [],
    ),
    # s14/a12 (d154): add_section / drop_section specs REMOVED — the served generic engine
    # discards the tree-authored outline (outline_hint=None, d56), so offering these outline
    # tools only gave the model a SECOND, silently-dropped surface to author source_ids on.
    # The report's section structure now comes solely from PHASE-2 findings-driven decomposition
    # over the file_write write-planner (the surface that actually consumes source_ids).
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
# s14/P3A — the COMPACT RESEARCH MEMORY builders (the SINGLE source of truth).
#
# These were authored in Stage A (a2) on the chat_app write side; Stage B (a4) moves
# them DOWN here so research_tree OWNS the memory pattern and BOTH the in-research
# DECISION node (`ResearchState.render_for_decision`) and the post-research WRITE
# planner (`chat_app.agentic` re-exports these) reason over the SAME compact memory —
# one narrative builder + one verbatim-index builder, NO divergent copies (the d-dup
# hazard). Two artifacts with different disciplines:
#   * a SUMMARIZED running NARRATIVE (covered/gaps/direction) — the only thing summarized;
#   * a byte-faithful VERBATIM SOURCE INDEX keyed by a stable 1-based [S#] — never
#     paraphrased, structure-aware chunk MAP only (no bodies), so it stays compact while
#     the writers still resolve each [S#] to its full scoped text.
# Bounded, model-independent, deterministic; NO fabrication (an unresolved claim is
# rendered WITHOUT an [S#], never a minted id). research_tree is a leaf module (factory/
# roles/synth_tools only), so these add no import cycle.
# ---------------------------------------------------------------------------- #

# Caps so the memory stays compact (the whole point — the Stage A run2 died at the
# UNSCOPED planner ingesting a 12k blob). Bounded, model-independent, deterministic.
_NARRATIVE_COVERED_CAP = 24
_NARRATIVE_GAPS_CAP = 12
_INDEX_CHUNKS_PER_SOURCE = 8
_INDEX_HEADING_CHARS = 120

# d156 — the BOUNDED scoped feed for the section WRITER / anchored REVIEWER. The bodies are
# NOT dumped inline (that overfed the served write/review llm.chat input to 12-17k tok on
# trace b359a87f); a small, genuinely-compact LEAD excerpt grounds the turn and the verbatim
# bodies are pulled on demand via ``load_source``. ``_SCOPED_LEAD_CHARS`` (~512 tok) caps one
# source's lead; ``_SCOPED_LEAD_TOTAL`` caps the section's TOTAL lead across its sources — so
# the whole scoped block (index map + leads) stays a few KB, well under num_ctx, BECAUSE the
# representation is compact + chunked, not because of an app truncation cap.
# d163/d168 — raised from 1800/8000: run2 measured the served write/review feed at only 32% of
# the num_ctx envelope (12K of ~26K usable tok), so there is large headroom to PUSH more
# figure-bearing source text to the writer (good-run class ~dozens of figures). The renderer
# still total-caps at ``_SCOPED_LEAD_TOTAL`` so the feed stays source-count-stable and well
# under num_ctx by construction; the per-source cap rises so each source contributes more
# figure/date passages (select_relevant_excerpt prefer_figures), not just its lede.
_SCOPED_LEAD_CHARS = 3600
_SCOPED_LEAD_TOTAL = 18000

# A markdown ATX heading line (`# … `..`###### … `) — the structure boundary the verbatim
# index chunks a source on (structure-aware, NOT fixed byte windows).
_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)


def _normalize_index_url(url: Any) -> str:
    """Loose URL key for matching an ArticleNote's runtime-owned url to a global source.

    The note url and the source url both come from the SAME fetched article (both
    runtime-owned), so an exact match is the norm; this only guards trailing-slash / case
    drift. NEVER used to mint a citation — only to resolve which existing [S#] grounds a
    claim; an unresolved claim is rendered WITHOUT an [S#] (no fabricated id)."""
    return str(url or "").strip().lower().rstrip("/")


def _structure_aware_chunks(markdown: str, sid: int) -> list[dict[str, Any]]:
    """Break ONE source's verbatim markdown into structure-aware chunk METADATA.

    Splits on the source's own ATX headings (its real structure), not fixed byte windows.
    Each chunk records a stable hierarchical ``cid`` (``S{sid}.c{k}``), its nearest
    heading copied VERBATIM (bounded for compactness, never paraphrased), and the
    ``char_span`` [start,end) so the chunk resolves back to an exact re-readable region of
    the source. Returns the MAP only — no body text — so the planner/decision node stay
    compact while a writer still loads the full verbatim text via the scoped-source path or
    the ``load_source`` retrieval tool. A source with no headings yields one
    ``(full document)`` chunk spanning the whole text."""
    text = str(markdown or "")
    if not text.strip():
        return []
    matches = list(_MD_HEADING_RE.finditer(text))
    chunks: list[dict[str, Any]] = []
    if not matches:
        heading = "(full document)"
        chunks.append({"cid": f"S{sid}.c0", "heading": heading, "char_span": [0, len(text)]})
        return chunks
    # Leading material before the first heading, if any non-trivial.
    first_start = matches[0].start()
    k = 0
    if first_start > 0 and text[:first_start].strip():
        chunks.append({"cid": f"S{sid}.c{k}", "heading": "(intro)", "char_span": [0, first_start]})
        k += 1
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        heading = m.group(2).strip()[:_INDEX_HEADING_CHARS]
        chunks.append({"cid": f"S{sid}.c{k}", "heading": heading, "char_span": [start, end]})
        k += 1
    return chunks


def resolve_chunk(markdown: str, sid: int, cid: Optional[str]) -> dict[str, Any]:
    """Resolve a ``[S#]`` / ``[S#.cK]`` reference to its VERBATIM chunk text (s14/P3A §3C).

    The byte-faithful resolver behind the ``load_source`` retrieval tool: given a source's
    full markdown and an optional chunk id, return that chunk's exact ``text`` (a verbatim
    slice of the source — never paraphrased) plus its ``heading`` and span. With no/unknown
    ``cid`` it returns the FIRST chunk (the source's lead). ``more`` flags that the source
    has further chunks the caller can request by cid. Empty text for an empty source."""
    chunks = _structure_aware_chunks(markdown, sid)
    text = str(markdown or "")
    if not chunks:
        return {"sid": f"S{sid}", "chunk": None, "heading": "", "text": "", "more": False}
    target = None
    if cid:
        want = str(cid).strip()
        for c in chunks:
            if c["cid"] == want or want == f"S{sid}":
                target = c
                break
    if target is None:
        target = chunks[0]
    s, e = target["char_span"]
    return {
        "sid": f"S{sid}",
        "chunk": target["cid"],
        "heading": target["heading"],
        "text": text[s:e],
        "more": len(chunks) > 1,
    }


def render_verbatim_source_index(sources: Sequence[Mapping[str, Any]]) -> str:
    """The VERBATIM, stable-[S#]-keyed SOURCE INDEX (s14/P3A §3B) — the compact memory MAP.

    Replaces the raw findings blob + positional ``render_source_catalog`` as the grounding
    map for BOTH the decision node and the write planner. Each source is keyed by its STABLE
    1-based ``[S#]`` (identical to the id ``render_scoped_sources`` resolves a section's
    ``source_ids`` against — so a planner assignment lines up with the writer's scoped feed),
    with its URL + title copied VERBATIM (never paraphrased) and a structure-aware chunk map
    (cid + heading + span). Compact MAP only — no article bodies — so the prompt stays inside
    the window. Returns ``""`` for no sources (callers then degrade to the legacy path)."""
    if not sources:
        return ""
    lines = [
        "SOURCE INDEX (the REAL fetched sources — VERBATIM, never paraphrased). Each is keyed "
        "by its [S#] number; set each section's source_ids to the [S#] numbers whose "
        "facts/URLs that section uses. The chunk headings show each source's structure so "
        "you can route it to the right section — the section WRITER is fed each assigned "
        "source's full verbatim text, so you do NOT need the bodies here:",
    ]
    for i, s in enumerate(sources, 1):
        title = str(s.get("title") or "").strip()
        url = str(s.get("url") or "").strip()
        lines.append(f"\n[S{i}] {title or url}")
        lines.append(f"    url: {url}")
        chunks = _structure_aware_chunks(s.get("markdown") or "", i)
        if chunks:
            shown = chunks[:_INDEX_CHUNKS_PER_SOURCE]
            for c in shown:
                span = c["char_span"]
                lines.append(f"    {c['cid']} [{span[0]}:{span[1]}] {c['heading']}")
            if len(chunks) > len(shown):
                lines.append(f"    … (+{len(chunks) - len(shown)} more sections)")
    return "\n".join(lines)


def render_scoped_source_index(
    sources: Sequence[Mapping[str, Any]],
    source_ids: Sequence[int],
    *,
    section_topic: str = "",
    full_index: bool = False,
    lead_ids: Optional[Sequence[int]] = None,
    lead_chars: int = _SCOPED_LEAD_CHARS,
    lead_total: int = _SCOPED_LEAD_TOTAL,
) -> str:
    """The BOUNDED, tool-chunked scoped feed for a section WRITER / anchored REVIEWER (d156).

    Replaces the legacy full-body scoped dump (``render_scoped_sources`` × N sources, which
    overfed the served write/review ``llm.chat`` input to 12-17k tok — trace b359a87f, where
    ``load_source`` was bound but called 0× because the bodies were already dumped). Emits:

      * the SOURCE INDEX MAP for the relevant ``[S#]`` (global stable numbering preserved) —
        url + structure-aware chunk headings, **no bodies**;
      * a small, genuinely-compact per-source LEAD excerpt (section-relevant, total-capped at
        ``lead_total``) so a turn that makes no tool call still has real grounding; and
      * an explicit instruction to call ``load_source('S#'[, 'S#.cK'])`` for any further
        verbatim text — so deeper text is retrieved CHUNKED on demand instead of an 85KB raw
        dump, and no single input exceeds a sane bound BECAUSE the representation is compact +
        chunked (not an app truncation cap).

    ``full_index=True`` lists EVERY source's ``[S#]`` in the map (the reviewer's resolution
    view — so a valid cross-section citation always resolves to a real ``[S#]`` and is never
    falsely deleted), while the LEAD excerpts + ``section_topic`` stay focused on the node's
    assigned ``source_ids``. ``lead_ids`` overrides WHICH sources get a LEAD excerpt
    (defaults to the assigned ids): the d162 UNSCOPED terminal writer — a single-section
    synthesis the planner left without ``source_ids`` — passes EVERY shown id so it is fed
    real grounding over ALL sources (the scoped path's bounded substance, not the
    num_ctx-saturating raw-body fold), still total-capped by ``lead_total`` so the feed is
    bounded by the WINDOW and stays stable as the source count grows. Global ``[S#]``
    numbering is identical to
    :func:`render_verbatim_source_index` / :func:`resolve_chunk` / ``load_source`` so a
    citation written here resolves through the SAME index and cannot regress. Returns ``""``
    when there are no sources, or no assigned ids and ``full_index`` is False (caller
    degrades gracefully — the unscoped single-section path is unchanged)."""
    if not sources:
        return ""
    n = len(sources)
    assigned = [i for i in source_ids if isinstance(i, int) and 1 <= i <= n]
    if not assigned and not full_index:
        return ""
    # Map view: ALL sources for the reviewer's resolution view, else only assigned ids.
    shown_ids = list(range(1, n + 1)) if full_index else assigned
    # WHICH sources get a LEAD excerpt: ``lead_ids`` when given (d162 unscoped writer passes
    # every shown id), else the section's assigned ids (the scoped/reviewer default).
    if lead_ids is not None:
        lead_set = {i for i in lead_ids if isinstance(i, int) and 1 <= i <= n}
    else:
        lead_set = set(assigned)
    # The LEAD total is shared across the lead sources (the grounding floor); each lead is
    # bounded by ``lead_chars`` and by an even share of ``lead_total`` — so MORE sources mean
    # THINNER per-source leads while the TOTAL stays bounded by the window (source-count-stable).
    n_lead = max(1, len(lead_set))
    per_lead = max(400, min(int(lead_chars), int(lead_total) // n_lead))
    lines = [
        "SOURCE INDEX — the REAL fetched sources, each keyed by its stable [S#] (url + the "
        "section map below; VERBATIM, never paraphrased). The full bodies are NOT all inline: "
        "a compact LEAD excerpt is shown for grounding, and you CALL load_source('S#') (or "
        "load_source('S#','S#.cK') for a specific section) to read any source's further "
        "verbatim text ON DEMAND — request only the chunks you actually need. Cite each claim "
        "with a real [S#] and its URL VERBATIM; fill any table/Sources citation cell with a "
        "real [S#]+URL from this index or drop the row — never a worded stand-in like "
        "\"Source N\", \"URL Placeholder\", \"[Name, 2025]\", or a URL not listed here:",
    ]
    for i in shown_ids:
        s = sources[i - 1]
        title = str(s.get("title") or "").strip()
        url = str(s.get("url") or "").strip()
        full = str(s.get("markdown") or "").strip()
        lines.append(f"\n[S{i}] {title or url}")
        lines.append(f"    url: {url}")
        chunks = _structure_aware_chunks(full, i)
        if chunks:
            shown = chunks[:_INDEX_CHUNKS_PER_SOURCE]
            for c in shown:
                span = c["char_span"]
                lines.append(f"    {c['cid']} [{span[0]}:{span[1]}] {c['heading']}")
            if len(chunks) > len(shown):
                lines.append(
                    f"    … (+{len(chunks) - len(shown)} more sections — "
                    f"load_source('S{i}','S{i}.cK'))"
                )
        # Bounded LEAD excerpt only for the lead sources (the grounding floor; a reviewer's
        # extra full_index sources outside the lead set stay map-only and load on demand).
        if i in lead_set and full:
            # d163 — PUSH the figure/date-bearing passages into the lead (not just the lede),
            # so the writer is fed concrete figures to quote verbatim within the bounded feed.
            lead = select_relevant_excerpt(full, section_topic, per_lead, prefer_figures=True)
            if lead:
                lines.append(f"    LEAD [S{i}] (excerpt — load_source('S{i}') for the rest):")
                lines.append(lead)
                if len(full) > len(lead):
                    lines.append(
                        f"    [showing {len(lead)} of {len(full)} chars — this source has "
                        f"MORE; call load_source('S{i}') for the rest, do not assume this is "
                        "the whole article]"
                    )
    return "\n".join(lines)


def compose_research_narrative(
    research_notes: Optional[Sequence[Mapping[str, Any]]],
    sources: Sequence[Mapping[str, Any]],
) -> str:
    """The running NARRATIVE summary (covered / gaps / direction) (s14/P3A §3A).

    Built from already-gathered ArticleNotes (NO new model call, NO fabrication): each
    note's ``key_claims`` become COVERED bullets (grounded in the global ``[S#]`` resolved by
    matching the note's runtime-owned url to the source list — unresolved => no [S#], never a
    minted id), and ``gaps_or_followups`` become the OPEN GAPS that set DIRECTION. This is the
    ONLY summarized artifact; the SOURCE INDEX stays verbatim, so citations resolve through
    the index and cannot regress. Bounded for compactness. Returns ``""`` when no notes are
    available (callers then degrade to the legacy render)."""
    if not research_notes:
        return ""
    url2sid: dict[str, int] = {}
    for i, s in enumerate(sources or [], 1):
        key = _normalize_index_url(s.get("url"))
        if key and key not in url2sid:
            url2sid[key] = i
    covered: list[tuple[str, Optional[int]]] = []
    gaps: list[str] = []
    seen_claims: set[str] = set()
    seen_gaps: set[str] = set()
    for note in research_notes:
        if not isinstance(note, Mapping):
            continue
        sid = url2sid.get(_normalize_index_url(note.get("url")))
        for claim in note.get("key_claims") or []:
            c = str(claim).strip()
            key = c.lower()
            if not c or key in seen_claims:
                continue
            seen_claims.add(key)
            covered.append((c, sid))
        for gap in note.get("gaps_or_followups") or []:
            g = str(gap).strip()
            key = g.lower()
            if not g or key in seen_gaps:
                continue
            seen_gaps.add(key)
            gaps.append(g)
    if not covered and not gaps:
        return ""
    covered = covered[:_NARRATIVE_COVERED_CAP]
    gaps = gaps[:_NARRATIVE_GAPS_CAP]
    lines = [
        "RESEARCH NARRATIVE (a running SUMMARY of what the research established — use it to "
        "decide the section breakdown. It is a summary for ORIENTATION; cite facts/figures "
        "from the SOURCE INDEX below, not from this narrative):",
    ]
    if covered:
        lines.append("COVERED (settled points, each grounded in the source [S#] it came from):")
        for claim, sid in covered:
            tag = f" [S{sid}]" if sid else ""
            lines.append(f"  - {claim}{tag}")
    if gaps:
        lines.append("OPEN GAPS (questions the report still needs to address):")
        for gap in gaps:
            lines.append(f"  - {gap}")
        lines.append(
            "DIRECTION: ensure the sections cover the settled points above and, where the "
            "sources support it, address the open gaps; do NOT invent facts to fill a gap."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------- #
# s15/a15 (d185) — NOTES-ARCH Layer 1: the EXPLICIT per-concern RESEARCH GRAPH.
#
# The research memory was ALREADY granular (one ArticleNote per source — never a single
# blob, d50.1) but its STRUCTURE was IMPLICIT: notes lived in a flat per-leaf list, the
# concern a note belonged to was only the leaf's branch id, and a note's gaps_or_followups
# were a flat bullet list. This makes that structure an EXPLICIT GRAPH the decision loop
# WALKS:
#
#     concern NODE  ──has──▶  its NOTES  ──cite──▶  SOURCES [S#]
#         │
#         └── gap EDGE (a note's gaps_or_followups) ──▶  spawns a NEW concern NODE
#
# It is a DERIVED PROJECTION over the SAME persisted ResearchState records + verbatim source
# index + the Tree's live/pruned branch structure (d49 read-real-state). It adds NO new
# persistence (a17 owns session-bound persistence) and fabricates nothing: a concern node is
# a real gathered leaf (or a real live/pruned Tree branch), a source edge resolves a note's
# runtime-owned url to an EXISTING [S#] (never a minted id), a gap edge is a note's own
# follow-up. The decision node READS this graph (``render`` → folded into
# ``render_for_decision``, beside the gap lens it preserves) so the loop's ``expand_branch`` /
# ``prune_branch`` are DRIVEN BY it: expand follows an OPEN gap edge to a new concern node,
# prune collapses a concern node. ``to_dict`` serializes the whole shape so a smoke/gate can
# assert it is GRAPH-shaped (per-concern nodes with note→source + gap edges), not a flat blob.
# ---------------------------------------------------------------------------- #

# Caps so the graph render stays compact (the decision prompt is num_ctx-bounded).
_GRAPH_MAX_CONCERNS = 16
_GRAPH_MAX_GAPS_PER_CONCERN = 4
_GRAPH_MAX_CLAIMS_PER_CONCERN = 4
_GRAPH_MAX_COLLAPSED = 8

# Caps for the per-research BRIEF (d185 Layer 2) — a thesis-level digest stays short
# and addressable (not the whole graph): the dominant findings + the open gaps.
_BRIEF_MAX_THESIS = 8
_BRIEF_MAX_GAPS = 6


@dataclass
class SourceRef:
    """A CITED-SOURCE edge: a concern's note → a verbatim source (the stable [S#])."""

    sid: Optional[int]  # 1-based [S#] in the run's verbatim index; None if unresolved
    url: str = ""
    title: str = ""
    trust: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"sid": self.sid, "url": self.url, "title": self.title, "trust": self.trust}


@dataclass
class GapEdge:
    """An OPEN gap a concern surfaced — a candidate EDGE to the NEXT concern node.

    ``text`` is the follow-up (from a note's ``gaps_or_followups``); ``from_concern`` is the
    concern id that raised it; ``source_sid`` ties it to the source [S#] whose note surfaced
    it; ``followed_by`` is the child concern id an ``expand_branch`` spawned to pursue it
    (None while the gap is still OPEN — breadth the loop has NOT yet funded)."""

    text: str
    from_concern: str
    source_sid: Optional[int] = None
    followed_by: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return not self.followed_by

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "from_concern": self.from_concern,
            "source_sid": self.source_sid,
            "followed_by": self.followed_by,
            "open": self.is_open,
        }


@dataclass
class ConcernNode:
    """One research CONCERN (a facet/branch) — a node of the research graph.

    concern NODE → its NOTES → cited SOURCES, with the concern's open follow-ups as the gap
    EDGES that branch to the next concern. ``status``: ``live`` (authored via expand, not
    gathered yet), ``settled`` (gathered — has a persisted leaf record), ``collapsed``
    (pruned). ``parent`` + ``rationale`` describe the gap edge this concern was spawned from
    (the tail of its incoming edge)."""

    concern_id: str
    question: str = ""
    depth: int = 0
    status: str = "live"
    parent: Optional[str] = None
    rationale: str = ""
    notes: list[dict[str, Any]] = field(default_factory=list)
    sources: list[SourceRef] = field(default_factory=list)
    gaps: list[GapEdge] = field(default_factory=list)

    @property
    def source_ids(self) -> list[int]:
        """The distinct [S#] this concern's notes cite, in first-seen order."""
        seen: list[int] = []
        for s in self.sources:
            if s.sid and s.sid not in seen:
                seen.append(s.sid)
        return seen

    @property
    def single_source(self) -> bool:
        """A SETTLED concern resting on ≤1 distinct source is NOT yet corroborated — an
        open structural gap the decision node should expand (breadth != depth)."""
        return self.status == "settled" and len(self.source_ids) <= 1

    @property
    def open_gaps(self) -> list[GapEdge]:
        return [g for g in self.gaps if g.is_open]

    @property
    def claims(self) -> list[str]:
        """The distinct key_claims across this concern's notes (the COVERED meaning)."""
        out: list[str] = []
        seen: set[str] = set()
        for n in self.notes:
            for c in n.get("key_claims") or []:
                t = str(c).strip()
                k = t.lower()
                if t and k not in seen:
                    seen.add(k)
                    out.append(t)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "concern_id": self.concern_id,
            "question": self.question,
            "depth": self.depth,
            "status": self.status,
            "parent": self.parent,
            "rationale": self.rationale,
            "note_count": len(self.notes),
            "source_ids": self.source_ids,
            "sources": [s.to_dict() for s in self.sources],
            "gaps": [g.to_dict() for g in self.gaps],
            "claims": self.claims,
            "single_source": self.single_source,
        }


@dataclass
class ConcernGraph:
    """The explicit per-concern research GRAPH (d185) — a DERIVED projection, not persisted.

    ``nodes`` maps concern_id → :class:`ConcernNode` in stable first-seen order. The graph is
    the SHAPE the decision loop walks: ``expand_branch`` follows an OPEN gap edge to a new
    concern node, ``prune_branch`` collapses a concern node. ``to_dict`` serializes the whole
    shape so a smoke/gate can assert it is GRAPH-shaped (per-concern nodes with note→source +
    gap edges), NOT a flat note blob."""

    nodes: dict[str, ConcernNode] = field(default_factory=dict)

    def concerns(self) -> list[ConcernNode]:
        return list(self.nodes.values())

    def live(self) -> list[ConcernNode]:
        return [n for n in self.nodes.values() if n.status != "collapsed"]

    def settled(self) -> list[ConcernNode]:
        return [n for n in self.nodes.values() if n.status == "settled"]

    def collapsed(self) -> list[ConcernNode]:
        return [n for n in self.nodes.values() if n.status == "collapsed"]

    def open_gaps(self) -> list[GapEdge]:
        return [g for n in self.nodes.values() if n.status != "collapsed" for g in n.open_gaps]

    def edges(self) -> list[dict[str, Any]]:
        """Every directed edge: concern→source (``cites``) + concern→gap/child (``gap``)."""
        out: list[dict[str, Any]] = []
        for n in self.nodes.values():
            for sid in n.source_ids:
                out.append({"kind": "cites", "from": n.concern_id, "to_source": sid})
            for g in n.gaps:
                out.append(
                    {
                        "kind": "gap",
                        "from": g.from_concern,
                        "to_concern": g.followed_by,
                        "open": g.is_open,
                        "text": g.text,
                    }
                )
        return out

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph SHAPE (the queryable artifact the a14 gate reads)."""
        return {
            "shape": "per_concern_graph",
            "concern_count": len(self.nodes),
            "settled_count": len(self.settled()),
            "collapsed_count": len(self.collapsed()),
            "open_gap_count": len(self.open_gaps()),
            "concerns": [n.to_dict() for n in self.nodes.values()],
            "edges": self.edges(),
        }

    def render(
        self,
        *,
        max_concerns: int = _GRAPH_MAX_CONCERNS,
        max_gaps_per_concern: int = _GRAPH_MAX_GAPS_PER_CONCERN,
        max_claims_per_concern: int = _GRAPH_MAX_CLAIMS_PER_CONCERN,
        max_collapsed: int = _GRAPH_MAX_COLLAPSED,
    ) -> str:
        """Render the graph for the decision prompt (compact, bounded, deterministic).

        Returns ``""`` when the graph has no live concern with any content (the caller then
        skips the block — no empty noise on a genuinely-empty branch)."""
        live = [n for n in self.live() if n.question or n.notes or n.gaps or n.sources]
        if not live:
            return ""
        lines = [
            "CONCERN GRAPH (the research as a GRAPH the loop WALKS — each CONCERN node → the "
            "SOURCES [S#] that fed it → its OPEN GAPS, which are the EDGES that branch to the "
            "NEXT concern. A concern resting on a SINGLE source, or carrying an open gap, is "
            "NOT settled — EXPAND its gap. Prune a concern that added no meaning):"
        ]
        for n in live[:max_concerns]:
            sids = ", ".join(f"S{s}" for s in n.source_ids) or "none yet"
            flag = " — SINGLE SOURCE (corroborate)" if n.single_source else ""
            head = f"  ● [{n.concern_id}] {n.question or '(open concern)'}"
            lines.append(f"{head}  (sources: {sids}{flag})")
            claims = n.claims[:max_claims_per_concern]
            if claims:
                lines.append("      covered: " + "; ".join(claims))
            for g in n.open_gaps[:max_gaps_per_concern]:
                src = f" [from S{g.source_sid}]" if g.source_sid else ""
                lines.append(f"      → open gap: {g.text}{src}")
        collapsed = self.collapsed()[:max_collapsed]
        if collapsed:
            lines.append("  COLLAPSED (pruned concerns — closed without further gather):")
            for n in collapsed:
                why = f" ({n.rationale})" if n.rationale else ""
                lines.append(f"    ⊘ [{n.concern_id}] {n.question}{why}")
        return "\n".join(lines)

    # ----------------------------------------------------------------------- #
    # d185 NOTES-ARCH Layer 2 — per-research BRIEF (a thesis-level digest of THIS
    # research, DERIVED from the graph). Like the graph itself it is a pure
    # projection (d148-clean: data, no behavior flags, NO new LLM call) and
    # FABRICATES nothing — every thesis line is a key_claim a settled concern's
    # NOTE actually carried, every source is an already-resolved [S#]. Layer 1 is
    # the GRAPH the loop walks; Layer 2 is the short ADDRESSABLE digest a chat
    # session keeps so multiple researches coexist, each looked up by its brief.
    # ----------------------------------------------------------------------- #
    def thesis(self, *, limit: int = _BRIEF_MAX_THESIS) -> list[str]:
        """The thesis-level findings: the dominant distinct key_claims this research
        ESTABLISHED, taken across its SETTLED concerns in first-seen order (ranked by
        how many concerns corroborate each claim, so a cross-concern finding leads).

        No fabrication — every line is a claim a real note carried; an empty graph
        (nothing gathered) yields an empty thesis, never a placeholder."""
        order: list[str] = []
        weight: dict[str, int] = {}
        text: dict[str, str] = {}
        first_seen: dict[str, int] = {}
        for n in self.settled():
            for c in n.claims:
                key = c.lower()
                if key not in weight:
                    weight[key] = 0
                    text[key] = c
                    first_seen[key] = len(order)
                    order.append(key)
                weight[key] += 1
        order.sort(key=lambda k: (-weight[k], first_seen[k]))
        return [text[k] for k in order[:limit]]

    def to_brief(self, *, topic: str = "", memory_index: Any = None,
                 limit_thesis: int = _BRIEF_MAX_THESIS,
                 limit_gaps: int = _BRIEF_MAX_GAPS) -> dict[str, Any]:
        """Project the per-research BRIEF (d185 Layer 2) from the graph — a compact,
        JSON-serializable, ADDRESSABLE digest of what THIS research established.

        ``topic`` is the research question/goal (the graph holds per-concern questions
        but not the overall ask); when omitted it falls back to the first settled
        concern's question. The brief carries the thesis-level findings, the settled
        concerns, the distinct cited sources ([S#]+url+title), the still-OPEN gaps
        (what this research did NOT yet close), and headline counts. It is the unit a
        chat session stores + looks up; ``digest`` is its one-glance human form.

        ``memory_index`` (d285 SB-3) is the research-memory INDEX this brief addresses —
        the handle of the SB-1 memory it digests, so a downstream step can pass it back
        to CONTINUE the research. Canonicalized (empty/None → the ``<<NEW>>`` sentinel,
        meaning this brief is not yet bound to a persisted memory line)."""
        settled = self.settled()
        topic = (topic or "").strip() or (settled[0].question if settled else "")
        # distinct cited sources across settled concerns (stable [S#] order, no dup id)
        srcs: list[dict[str, Any]] = []
        seen_sids: set[int] = set()
        for n in settled:
            for s in n.sources:
                if s.sid and s.sid not in seen_sids:
                    seen_sids.add(s.sid)
                    srcs.append({"sid": s.sid, "url": s.url, "title": s.title})
        srcs.sort(key=lambda d: d["sid"])
        thesis = self.thesis(limit=limit_thesis)
        open_gaps: list[str] = []
        seen_gap: set[str] = set()
        for g in self.open_gaps():
            k = g.text.lower().strip()
            if g.text.strip() and k not in seen_gap:
                seen_gap.add(k)
                open_gaps.append(g.text.strip())
        concerns = [
            {
                "concern_id": n.concern_id,
                "question": n.question,
                "source_ids": n.source_ids,
                "single_source": n.single_source,
            }
            for n in settled
        ]
        brief: dict[str, Any] = {
            "shape": "per_research_brief",
            "topic": topic,
            # d285 SB-3: the research-memory INDEX this brief addresses (an index to
            # CONTINUE, or <<NEW>> when not yet bound to a persisted memory line).
            "memory_index": normalize_brief_memory_index(memory_index),
            "thesis": thesis,
            "concerns": concerns,
            "sources": srcs,
            "open_gaps": open_gaps[:limit_gaps],
            "concern_count": len(self.concerns()),
            "settled_count": len(settled),
            "source_count": len(srcs),
            "open_gap_count": len(open_gaps),
        }
        brief["digest"] = self.render_brief(brief)
        return brief

    @staticmethod
    def render_brief(brief: Mapping[str, Any]) -> str:
        """A compact one-glance text digest of a brief dict (the addressable summary).

        Deterministic + bounded; safe on a brief with no thesis/sources (an honest
        'research established nothing yet' rather than a fabricated summary)."""
        topic = str(brief.get("topic") or "").strip() or "(untitled research)"
        lines = [f"RESEARCH BRIEF — {topic}"]
        thesis = list(brief.get("thesis") or [])
        if thesis:
            lines.append("Established:")
            lines += [f"  • {t}" for t in thesis]
        else:
            lines.append("Established: (nothing gathered yet)")
        srcs = list(brief.get("sources") or [])
        if srcs:
            cited = ", ".join(f"S{s.get('sid')}" for s in srcs)
            lines.append(f"Sources ({len(srcs)}): {cited}")
        gaps = list(brief.get("open_gaps") or [])
        if gaps:
            lines.append("Open gaps:")
            lines += [f"  → {g}" for g in gaps]
        return "\n".join(lines)


def build_research_brief(
    records: Optional[Sequence[Mapping[str, Any]]],
    sources: Optional[Sequence[Mapping[str, Any]]],
    *,
    topic: str = "",
    memory_index: Any = None,
    tree: Optional["Tree"] = None,
) -> dict[str, Any]:
    """Project the per-research BRIEF (d185 Layer 2) straight from persisted state.

    Convenience over ``build_concern_graph(...).to_brief(...)`` for callers that hold
    the records + verbatim source index (and optionally the live Tree) but not a graph
    object. Derived, never persisted here (chat_app owns the session-level store).
    ``memory_index`` (d285 SB-3) is carried through to the brief unchanged."""
    return build_concern_graph(records, sources, tree=tree).to_brief(
        topic=topic, memory_index=memory_index
    )


def build_concern_graph(
    records: Optional[Sequence[Mapping[str, Any]]],
    sources: Optional[Sequence[Mapping[str, Any]]],
    *,
    tree: Optional["Tree"] = None,
) -> ConcernGraph:
    """Project the EXPLICIT per-concern graph (d185) from persisted state (+ the live Tree).

    The graph is DERIVED, never persisted (a17 owns persistence): a SETTLED concern node is a
    real gathered leaf ``record`` (its notes attached; each note's url resolved to an EXISTING
    [S#] as a CITES edge; each note's ``gaps_or_followups`` as a GAP edge); a LIVE concern node
    is a Tree branch the model authored via ``expand_branch`` but has not gathered yet; a
    COLLAPSED concern node is a Tree ``prune`` target. When ``tree`` is supplied, expand/prune
    are folded so the graph REFLECTS the loop's walk (a just-expanded gap → a live child node;
    a pruned concern → collapsed); a child branch whose parent is a graphed concern marks that
    parent's matching OPEN gap as ``followed_by`` (the edge the loop walked). No fabrication: a
    note that resolves to no source carries no [S#] edge; a tree-only branch carries no notes."""
    # url → stable [S#] (same join the narrative uses; never mints an id).
    url2sid: dict[str, int] = {}
    sid_meta: dict[int, dict[str, str]] = {}
    for i, s in enumerate(sources or [], 1):
        if not isinstance(s, Mapping):
            continue
        key = _normalize_index_url(s.get("url"))
        if key and key not in url2sid:
            url2sid[key] = i
        sid_meta[i] = {
            "url": str(s.get("url") or "").strip(),
            "title": str(s.get("title") or "").strip(),
        }
    graph = ConcernGraph()

    # (1) SETTLED concern nodes — the gathered leaf records (concern → notes → sources → gaps).
    for rec in records or []:
        if not isinstance(rec, Mapping):
            continue
        cid = str(rec.get("branch_id") or "").strip() or f"layer{rec.get('layer', '?')}"
        node = graph.nodes.get(cid)
        if node is None:
            node = ConcernNode(
                concern_id=cid,
                question=str(rec.get("question") or "").strip(),
                depth=int(rec.get("layer") or 0),
            )
            graph.nodes[cid] = node
        node.status = "settled"  # a persisted record means this concern was gathered
        seen_gaps: set[str] = {g.text.lower() for g in node.gaps}
        for raw in rec.get("notes") or []:
            if not isinstance(raw, Mapping):
                continue
            note = dict(raw)
            node.notes.append(note)
            sid = url2sid.get(_normalize_index_url(note.get("url")))
            if sid is not None and sid not in node.source_ids:
                meta = sid_meta.get(sid, {})
                node.sources.append(
                    SourceRef(
                        sid=sid,
                        url=meta.get("url", ""),
                        title=meta.get("title", ""),
                        trust=str(note.get("source_trust") or ""),
                    )
                )
            for gap in note.get("gaps_or_followups") or []:
                g = str(gap).strip()
                key = g.lower()
                if g and key not in seen_gaps:
                    seen_gaps.add(key)
                    node.gaps.append(GapEdge(text=g, from_concern=cid, source_sid=sid))

    # (2) LIVE concern nodes + parent/gap edges from the Tree (expand/prune WALK the graph).
    if tree is not None:
        for branch in tree.branches.values():
            bid = str(getattr(branch, "id", "")).strip()
            if not bid:
                continue
            node = graph.nodes.get(bid)
            if node is None:
                # An authored-but-not-yet-gathered concern: a live node (no notes yet).
                node = ConcernNode(
                    concern_id=bid,
                    question=str(getattr(branch, "question", "")).strip(),
                    depth=int(getattr(branch, "depth", 0) or 0),
                    status="live",
                )
                graph.nodes[bid] = node
            # Record the incoming gap edge (where this concern was spawned from).
            node.parent = str(getattr(branch, "parent", "") or "").strip() or node.parent
            node.rationale = str(getattr(branch, "rationale", "") or "").strip() or node.rationale
            # FOLLOW the edge: mark the parent concern's matching OPEN gap as followed by this
            # child (best-effort — exact text match, else the rationale's words, else the first
            # open gap). This is the loop's walk made explicit; unresolved → the gap stays open.
            parent = graph.nodes.get(node.parent or "")
            if parent is not None and parent is not node:
                _follow_gap(parent, node)
        # (3) COLLAPSED concern nodes — prune targets (a concern closed without more gather).
        for pr in getattr(tree, "pruned", []) or []:
            if not isinstance(pr, Mapping):
                continue
            tgt = str(pr.get("target") or "").strip()
            if not tgt:
                continue
            reason = str(pr.get("reason") or "").strip()
            node = graph.nodes.get(tgt)
            if node is None:
                node = ConcernNode(concern_id=tgt, rationale=reason)
                graph.nodes[tgt] = node
            node.status = "collapsed"
            if reason:
                node.rationale = reason
    return graph


def _follow_gap(parent: ConcernNode, child: ConcernNode) -> None:
    """Mark the parent concern's OPEN gap that ``child`` was spawned to pursue as followed.

    Best-effort edge resolution: prefer a gap whose text overlaps the child's rationale or
    question; else the first still-open gap. Idempotent (a gap already followed by this child
    is left alone); if nothing matches the gap simply stays OPEN — never a fabricated link."""
    opens = parent.open_gaps
    if not opens:
        return
    hay = f"{child.rationale} {child.question}".lower()
    chosen: Optional[GapEdge] = None
    for g in opens:
        words = [w for w in re.split(r"\W+", g.text.lower()) if len(w) > 3]
        if words and sum(1 for w in words if w in hay) >= max(1, len(words) // 3):
            chosen = g
            break
    if chosen is None and child.concern_id in {g.followed_by for g in parent.gaps}:
        return  # already linked to this child under some gap
    chosen = chosen or opens[0]
    chosen.followed_by = child.concern_id


# s15/a6 (d182) — the OPEN-GAP LENS appended to EVERY decision render. The gap lane now lives in
# the DATA the decision node reads (this render), not only in a distant prose doctrine wall: it
# tells the node to reason over the BLANKS each covered facet still has, so an early-stop-prone
# small model does not read a short COVERED list as "every facet is done" and stop after layer 1.
# DECISION-ONLY: it is appended in ``render_for_decision`` and is NEVER folded into the shared
# ``compose_research_narrative`` (which the WRITE planner also reads, where "go research more"
# would be the wrong signal).
_DECISION_GAP_LENS = (
    "OPEN-GAP LENS — reason over the BLANKS, not just the COVERED list above. For EACH covered "
    "facet, ask what is still UNVERIFIED or MISSING: a figure only one source gave, a date without "
    "its surrounding detail, a claim no second source corroborates, an unanswered follow-up. Each "
    "such blank is a GAP to EXPAND. A short COVERED list (or one with no OPEN GAPS listed yet) is "
    "the START of the investigation, NOT proof it is complete — keep expanding while real blanks "
    "remain."
)


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

    def __init__(
        self, path: str | os.PathLike[str], *, session_bound: bool = False
    ) -> None:
        # s15/a17 (d185) — NOTES-ARCH Layer 3: SESSION BINDING. By DEFAULT the state is
        # RUN-scoped — the file is truncated on open so the decision node reads only THIS
        # run (byte-identical to the pre-a17 behaviour for the inline/non-chat path). When
        # ``session_bound`` is True the file is keyed by the CHAT SESSION (see
        # chat_app._research_state_path) and STICKS: a follow-up turn in the SAME session
        # opens the SAME file and READS BACK the prior notes + verbatim sources (so it can
        # build on / load_source them instead of re-researching), while a DIFFERENT session
        # opens a DIFFERENT file and never sees them (isolation). Persistence is ADDITIVE —
        # a sidecar ``.sources.jsonl`` carries the verbatim source bodies (the leaf JSONL
        # only kept notes + a digest), so the read-back restores BOTH notes and sources.
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_bound = bool(session_bound)
        # The verbatim SOURCE INDEX sidecar — append-only, [S#]-ordered (line N == [S#] N).
        # Persisting it is what lets a session follow-up resolve a prior source's body via
        # load_source; the leaf records alone never carried the markdown.
        self.sources_path = self.path.parent / (self.path.stem + ".sources.jsonl")
        if not self.session_bound:
            # Truncate any stale prior-run file so the decision node reads only THIS run.
            self.path.write_text("", encoding="utf-8")
        self._records: list[dict[str, Any]] = []
        # s14/P3A Stage B (item 2) — the VERBATIM SOURCE INDEX: the run's fetched sources
        # accumulated in fetch order, URL-deduped, each holding {title,url,markdown}. Its
        # 1-based position is the STABLE [S#] id the decision render + the write side both
        # resolve against (identical id to chat_app's `_collect_chain_sources` dedup, which
        # also dedupes by url in launch order). Append-only: a re-fetched URL re-uses its
        # existing [S#] (cross-layer URL-dedup, closes the d76 gap) — an id is never reused.
        # This REPLACES the per-leaf `findings_digest[:500]` paraphrase as the source memory
        # the decision node reasons over (the raw bodies live here verbatim, never summarized).
        self._sources: list[dict[str, Any]] = []
        self._source_url_index: dict[str, int] = {}  # normalized-url -> 1-based [S#]
        # The OUTLINE channel: an append-only sidecar of the agent's document-
        # direction ops (add_section / drop_section), persisted ALONGSIDE the leaf state and
        # read BACK from disk into each decision layer (the same anti-hallucination read-back
        # as the leaf notes — the rendered outline comes from disk, never the model's memory).
        self.outline_path = self.path.parent / (self.path.stem + ".outline.jsonl")
        if not self.session_bound:
            self.outline_path.write_text("", encoding="utf-8")
        # FRONTIER sidecar (autonomy rebuild P4): the OPEN research branches (model-authored
        # via expand_branch / decompose, not yet gathered) persisted per decision layer. A
        # FOLLOW-UP research plan in the same session SEEDS from these instead of
        # re-decomposing the goal from scratch — the fix for the live prune/re-add loop
        # (B1→B5 re-decomposed 5×, duplicates grown to B12, starved facets executed in 0
        # of 5 epochs). Rewritten whole each layer (the frontier is current state, not a log).
        self.frontier_path = self.path.parent / (self.path.stem + ".frontier.jsonl")
        if not self.session_bound:
            self.frontier_path.write_text("", encoding="utf-8")
        if self.session_bound:
            # SESSION READ-BACK (a17): rehydrate this session's prior notes + verbatim
            # sources from disk so the decision loop reasons over (and load_source can pull
            # from) everything the session gathered so far — never a wiped slate. The leaf
            # records (notes/digest) live in self.path; the verbatim bodies in the sidecar.
            self._records = self.read()
            self._load_sources()

    @property
    def memory_handle(self) -> str:
        """The stable HANDLE of this research memory (d221 memory-by-handle).

        The file stem is the run/session-stable identifier (session-bound → the chat
        session key; run-scoped → the run id), so it is the natural name a downstream
        node is BOUND to and reads back via its source tools. Pure derivation, no I/O."""
        return self.path.stem

    def _load_sources(self) -> None:
        """Rehydrate the verbatim SOURCE INDEX from the sidecar (session read-back, a17).

        Append-only + [S#]-ordered on disk (line N == [S#] N), so reloading preserves the
        stable ids: each entry is re-appended in file order and its url re-indexed. Safe on
        a missing/partial sidecar (an unparseable line is skipped, never fabricated)."""
        self._sources = []
        self._source_url_index = {}
        if not self.sources_path.exists():
            return
        for line in self.sources_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(entry, Mapping):
                continue
            url = str(entry.get("url") or "").strip()
            rec = {
                "title": str(entry.get("title") or "").strip(),
                "url": url,
                "markdown": str(entry.get("markdown") or ""),
            }
            self._sources.append(rec)
            key = _normalize_index_url(url)
            if key and key not in self._source_url_index:
                self._source_url_index[key] = len(self._sources)  # 1-based [S#]

    def record_frontier(self, branches: Sequence[Mapping[str, Any]]) -> None:
        """Persist the CURRENT open frontier (P4) — model-authored branches not yet gathered.

        Rewritten whole each call (current state, not a log): each entry carries the
        branch's own id/question/rationale/memory_index verbatim (the model authored
        them; the engine records, never composes). An empty sequence CLEARS the file
        (the model settled every concern — nothing to seed a follow-up from)."""
        try:
            lines = [
                json.dumps({
                    "id": str(b.get("id") or ""),
                    "question": str(b.get("question") or ""),
                    "rationale": str(b.get("rationale") or ""),
                    "memory_index": str(b.get("memory_index") or ""),
                }, ensure_ascii=False)
                for b in branches
                if str(b.get("question") or "").strip()
            ]
            self.frontier_path.write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001 — frontier persistence must never break a layer
            pass

    def read_frontier(self) -> list[dict[str, Any]]:
        """Read back the persisted open frontier (P4) — [] when absent/cleared."""
        try:
            if not self.frontier_path.exists():
                return []
            out: list[dict[str, Any]] = []
            for line in self.frontier_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001 — skip a bad line, never fabricate
                    continue
                if isinstance(rec, Mapping) and str(rec.get("question") or "").strip():
                    out.append(dict(rec))
            return out
        except Exception:  # noqa: BLE001
            return []

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

    def _index_sources(self, fetched: Sequence[Mapping[str, Any]]) -> None:
        """Fold a leaf's fetched sources into the VERBATIM SOURCE INDEX (s14/P3A item 2).

        Append-only, URL-deduped: a fetched source whose (normalized) url is already indexed
        re-uses its existing [S#] (cross-layer dedup); a new url is appended and assigned the
        next 1-based [S#]. The full ``{title,url,markdown}`` is kept VERBATIM (never
        paraphrased) so the index resolves a citation back to an exact source. Ids are never
        reused or renumbered — the anti-regression spine."""
        for src in fetched or []:
            if not isinstance(src, Mapping):
                continue
            url = str(src.get("url") or "").strip()
            key = _normalize_index_url(url)
            if key and key in self._source_url_index:
                continue  # already indexed under its stable [S#]
            entry = {
                "title": str(src.get("title") or "").strip(),
                "url": url,
                "markdown": str(src.get("markdown") or ""),
            }
            self._sources.append(entry)
            if key:
                self._source_url_index[key] = len(self._sources)  # 1-based [S#]
            if self.session_bound:
                # PERSIST the verbatim body to the sidecar so a session follow-up can read
                # it back + load_source it (a17). Append-only, in [S#] order — the body the
                # leaf record never carried. Best-effort: a sidecar write must never break a
                # research run (the in-memory index is still authoritative for THIS run).
                try:
                    with self.sources_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(entry, default=str) + "\n")
                except OSError as exc:
                    print(f"[research-state] source-sidecar append skipped: {exc}", flush=True)

    def append_leaf(self, leaf: LeafResult, *, layer: int) -> None:
        """Append ONE leaf's gathered state (notes + findings digest) to the file (d49).

        s14/P3A item 2 — ALSO folds the leaf's fetched sources into the run's VERBATIM SOURCE
        INDEX (append-only, URL-deduped, stable [S#]); the index is the source memory the
        decision render now reasons over, replacing the per-leaf ``findings_digest`` slice."""
        self._index_sources(leaf.fetched or [])
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

    def sources(self) -> list[dict[str, Any]]:
        """The run's VERBATIM SOURCE INDEX entries (``[{title,url,markdown}]``), [S#]-ordered.

        Position is the stable 1-based [S#]. The same list the write side resolves
        ``source_ids`` against (chat_app re-collects it with the identical URL-dedup), so a
        decision-time citation and a write-time citation share one id space."""
        return list(self._sources)

    def collect_notes(self) -> list[dict[str, Any]]:
        """Every ArticleNote gathered so far, across all leaf records (decision memory input).

        Read-back from the persisted records (d49 real state) so the narrative is built from
        what was actually gathered, never the model's memory."""
        notes: list[dict[str, Any]] = []
        for rec in self.read():
            for note in rec.get("notes") or []:
                if isinstance(note, Mapping):
                    notes.append(dict(note))
        return notes

    def concern_graph(self, tree: Optional["Tree"] = None) -> ConcernGraph:
        """The EXPLICIT per-concern research GRAPH (d185) projected from the persisted state.

        A DERIVED view (no new persistence — a17 owns that): concern nodes are the gathered
        leaf records read back from disk (d49), notes→[S#] cites + gap edges from the verbatim
        source index. Pass the live ``tree`` to fold in authored-but-ungathered concerns
        (expand) and pruned concerns (collapse), so the graph REFLECTS the loop's walk. The
        object serializes via ``to_dict`` for a smoke/gate to assert the graph SHAPE."""
        return build_concern_graph(self.read(), self._sources, tree=tree)

    def render_concern_graph(self, tree: Optional["Tree"] = None) -> str:
        """Render the per-concern graph for the decision prompt (``""`` when empty)."""
        return self.concern_graph(tree=tree).render()

    def research_brief(
        self, *, topic: str = "", tree: Optional["Tree"] = None
    ) -> dict[str, Any]:
        """The per-research BRIEF (d185 Layer 2) — a thesis-level digest of THIS run.

        DERIVED from the same concern graph (Layer 1) over the persisted state (d49 read
        -back, no fabrication): the dominant findings this research established, its
        settled concerns, the distinct cited sources, and the still-open gaps. ``topic``
        is the research question/goal; pass the live ``tree`` to fold in the loop's walk.
        chat_app stores this dict at the CHAT-SESSION level keyed by a research id so
        multiple researches in one chat coexist, each addressable by its brief."""
        return self.concern_graph(tree=tree).to_brief(topic=topic)

    def _findings_fallback_covered(self, records: Sequence[Mapping[str, Any]]) -> str:
        """COVERED bullets derived from findings prose when NO ArticleNotes were emitted.

        s13 FINDINGS BRIDGE, preserved: a small model can produce REAL findings yet emit 0
        structured notes — the narrative (built from notes) would then be empty and the
        decision node would STOP/PRUNE blind. So when there are no notes, surface a compact
        COVERED list from each leaf's persisted ``findings_digest`` (real gathered prose, the
        [S#] left off because no note tied a claim to a source — never a minted id), so the
        decision node still reasons over what was gathered. Returns ``""`` when there is no
        findings prose either (an honest 'nothing yet')."""
        bullets: list[str] = []
        seen: set[str] = set()
        for rec in records:
            digest = str(rec.get("findings_digest", "") or "").strip()
            if not digest:
                continue
            for seg in _split_sentences(digest):
                s = seg.strip()
                if len(s) <= 20:
                    continue
                key = s.lower()
                if key in seen:
                    continue
                seen.add(key)
                bullets.append(s)
                if len(bullets) >= _NARRATIVE_COVERED_CAP:
                    break
            if len(bullets) >= _NARRATIVE_COVERED_CAP:
                break
        if not bullets:
            return ""
        lines = [
            "RESEARCH NARRATIVE (running SUMMARY from the gathered findings — the note lane "
            "emitted nothing, so these are settled points pulled from the raw findings prose):",
            "COVERED (settled points so far):",
        ]
        lines.extend(f"  - {b}" for b in bullets)
        return "\n".join(lines)

    def render_for_decision(
        self,
        records: Optional[Sequence[Mapping[str, Any]]] = None,
        *,
        tree: Optional["Tree"] = None,
    ) -> str:
        """Render the COMPACT research MEMORY for the decision prompt (s14/P3A items 1+2).

        RETIRES the old raw per-branch dump (which re-rendered ALL prior-layer state in full
        every layer → unbounded linear context growth, d146(2)). The decision node now reasons
        over the SAME compact memory the write planner uses (the a2 builders, now research-
        owned above):

          * the running NARRATIVE summary — COVERED (each grounded in its [S#]) / OPEN GAPS /
            DIRECTION, built from the gathered ArticleNotes (the ONLY summarized artifact); and
          * the VERBATIM SOURCE INDEX — stable [S#] + url + structure-aware chunk map, the raw
            sources never paraphrased.

        The node grows from the GAPS and prunes COVERED redundancy, grounded in stable ids, not
        a lossy 500-char excerpt. Reads notes/records from disk when ``records`` is not supplied
        (d49 real state); the source index is the run-accumulated verbatim list (``sources()``).
        Falls back to a findings-derived COVERED list when no notes were emitted (the s13
        findings-bridge, preserved) so the decision node is never blind on a notes-light run."""
        records = list(records) if records is not None else self.read()
        notes: list[dict[str, Any]] = []
        for rec in records:
            for note in rec.get("notes") or []:
                if isinstance(note, Mapping):
                    notes.append(dict(note))
        srcs = self._sources
        narrative = compose_research_narrative(notes, srcs)
        if not narrative:
            # No structured notes → derive COVERED from the real findings prose (bridge).
            narrative = self._findings_fallback_covered(records)
        index = render_verbatim_source_index(srcs)
        if not narrative and not index:
            return (
                "RESEARCH NARRATIVE (already gathered): (none yet)\n\n"
                "SOURCE INDEX: (no sources fetched yet)"
            )
        parts: list[str] = []
        parts.append(narrative or "RESEARCH NARRATIVE: (no settled points yet)")
        parts.append(index or "SOURCE INDEX: (no sources fetched yet)")
        # s15/a15 (d185) — fold in the EXPLICIT per-concern GRAPH so the decision node reasons
        # over the structure it WALKS (concern → notes → sources, gaps = edges to the next
        # concern), not only the flattened COVERED/GAPS narrative. Built from the SAME records
        # (+ the live tree when supplied, so authored/pruned concerns show); additive — the
        # narrative + index + gap lens are untouched. Empty (no graphable concern) → skipped.
        graph_render = build_concern_graph(records, srcs, tree=tree).render()
        if graph_render:
            parts.append(graph_render)
        # s15/a6 (d182) part (c) — surface the OPEN-GAP LENS prominently right beside the data so
        # the decision node reasons over blanks-to-fill, not COVERED-only. Decision-only (the shared
        # narrative builder stays clean for the write planner).
        parts.append(_DECISION_GAP_LENS)
        return "\n\n".join(parts)

    def all_fetched(self) -> list[dict[str, Any]]:
        """(Accumulated count helper.) Records hold ``fetched_count``; the real source
        records are carried by the orchestrator (below) for the c13 write-side contract."""
        return self._records


# ---------------------------------------------------------------------------- #
# The RESEARCH-MEMORY TOOL — an ABSTRACT, index-keyed create-and-lookup surface
# over the existing :class:`ResearchState` (d285 SB-1). NOT a second store.
# ---------------------------------------------------------------------------- #
class _NewMemorySentinel:
    """The ``NEW`` sentinel a caller passes to :meth:`ResearchMemoryStore.open_memory`
    to mean "start a FRESH research memory" (as opposed to reusing one by index).

    A distinct singleton object (identity-compared) so it can never collide with a real
    index string — passing it is unambiguous, where ``None``/``""`` could be an accident."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "NEW_MEMORY"


# The one sentinel value. ``open_memory(NEW_MEMORY)`` (or ``open_memory()`` / ``None``)
# creates a brand-new memory; ``open_memory("<index>")`` reuses an existing one.
NEW_MEMORY = _NewMemorySentinel()


def _normalize_memory_index(index: str) -> str:
    """Sanitize a memory INDEX into a filesystem-safe stem (the same rule the served path
    uses for run/session ids). Idempotent — a normalized index normalizes to itself, so a
    node that passes back a memory's ``memory_handle`` always resolves to the SAME file."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(index or "").strip())


class ResearchMemoryStore:
    """The abstract research-memory SINGLETON tool (d285 SB-1).

    The d285 memory model: nodes NEVER construct or manage a research store directly — they
    only pass an INDEX to this tool, and the tool owns creation and lookup:

    * an **existing index** → the SAME memory is returned (same ``memory_handle``, and its
      prior notes + verbatim sources are readable back), so a downstream node continues the
      research a prior node gathered;
    * the **NEW sentinel** (``NEW_MEMORY``) **or nothing** (``None``) → a brand-new, DISTINCT
      memory is created under a freshly-minted index, isolated from every existing one.

    It is "singleton" in two senses: there is ONE store per root (see
    :func:`get_research_memory_store`), and within a store there is ONE live memory per index
    (memoized) — so two nodes passing the same index share one memory rather than racing
    divergent copies. It is built ON the existing :class:`ResearchState` + its
    ``memory_handle`` + the ``.sources.jsonl`` sidecar (index-keyed memories open
    ``session_bound=True`` so the read-back of prior notes + sources is the real disk state),
    NOT a second/parallel store.

    Anti-fabrication: the tool is GENERIC and domain-neutral — it has zero spec-name or
    role-name conditionals and authors no behavior keyed to which caller opened it. The
    WHEN-to-reuse-vs-create-a-memory methodology lives in the research-methodology spec body
    + the planner's reasoning, never here; this tool provides only the
    index→reuse / NEW→create mechanism."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        # Where index-keyed memories live; one ``<index>.jsonl`` (+ sidecars) per memory.
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # index → the live ResearchState (per-index singleton for THIS store/process).
        self._memories: dict[str, ResearchState] = {}

    def _path_for(self, index: str) -> Path:
        return self.root / f"{index}.jsonl"

    def _mint_index(self) -> str:
        """A fresh, collision-free index for a NEW memory (distinct from any existing one,
        in-memory or on disk)."""
        while True:
            candidate = f"mem_{uuid.uuid4().hex}"
            if candidate not in self._memories and not self._path_for(candidate).exists():
                return candidate

    def open_memory(
        self, index: "str | _NewMemorySentinel | None" = None
    ) -> ResearchState:
        """Open a research memory by INDEX — the sole surface a node uses.

        ``index`` is one of: an existing index string (→ reuse that memory), ``NEW_MEMORY``
        or ``None`` (→ create a new distinct memory). Returns the :class:`ResearchState`;
        the caller never constructs one itself. Read ``.memory_handle`` to learn the index a
        new memory was minted under (so a later node can pass it back to continue)."""
        if index is None or index is NEW_MEMORY:
            resolved = self._mint_index()
        else:
            resolved = _normalize_memory_index(index)
            if not resolved:
                # An empty/whitespace index is not a usable handle → treat as NEW (a fresh
                # memory) rather than silently sharing one blank-stem file.
                resolved = self._mint_index()
        cached = self._memories.get(resolved)
        if cached is not None:
            return cached
        # Index-keyed memories are session_bound=True: opening an EXISTING index reads its
        # prior notes + verbatim sources back (no truncate); opening a never-seen index just
        # starts empty. Built ON ResearchState — no second store.
        state = ResearchState(self._path_for(resolved), session_bound=True)
        self._memories[resolved] = state
        return state

    def has_memory(self, index: str) -> bool:
        """Whether an index already names a memory (live in this store OR persisted on disk).
        A read-only probe; it never creates one."""
        resolved = _normalize_memory_index(index)
        if not resolved:
            return False
        return resolved in self._memories or self._path_for(resolved).exists()


# One store per root directory (process-wide), so "the research-memory tool" is a genuine
# singleton: every node that opens a memory under the same root shares the same store (and
# therefore the same per-index memories). Keyed by the resolved absolute root.
_MEMORY_STORES: dict[str, ResearchMemoryStore] = {}


def get_research_memory_store(root: str | os.PathLike[str]) -> ResearchMemoryStore:
    """Return THE research-memory store for ``root`` (created once, then reused)."""
    key = str(Path(root).resolve())
    store = _MEMORY_STORES.get(key)
    if store is None:
        store = ResearchMemoryStore(root)
        _MEMORY_STORES[key] = store
    return store


def resolve_brief_memory(
    root_or_store: "str | os.PathLike[str] | ResearchMemoryStore",
    memory_index: Any,
) -> ResearchState:
    """Resolve a STEP BRIEF's ``memory_index`` field THROUGH SB-1's store (d285 SB-3).

    The bridge from the planner's AUTHORED choice (the brief field — an index string or
    the textual ``<<NEW>>`` sentinel) onto SB-1's create-and-lookup mechanism:

    * an existing index → SB-1 returns the SAME memory (its prior notes + sources read
      back), so this step CONTINUES the research a prior step gathered;
    * the ``<<NEW>>`` sentinel (or an empty/unspecified field) → SB-1 mints a FRESH,
      distinct memory under a newly-minted index.

    ``root_or_store`` is either a directory ROOT (→ the per-root singleton store via
    :func:`get_research_memory_store`) or an already-built :class:`ResearchMemoryStore`.
    Returns the :class:`ResearchState`; read ``.memory_handle`` to learn the index a fresh
    memory was minted under (so a LATER step can pass it back to continue).

    Anti-fabrication (d285, d10-clean): the engine STAMPS no index here — it only RELAYS
    the planner-authored value to SB-1's index→reuse / <<NEW>>→create mechanism. The
    WHEN-to-continue-vs-create decision is the planner's reasoning over data, never code."""
    store = (
        root_or_store
        if isinstance(root_or_store, ResearchMemoryStore)
        else get_research_memory_store(root_or_store)
    )
    canonical = normalize_brief_memory_index(memory_index)
    if canonical == NEW_MEMORY_SENTINEL:
        return store.open_memory(NEW_MEMORY)
    return store.open_memory(canonical)


# ---------------------------------------------------------------------------- #
# The DECISION NODE — reasoning-driven, persisted-state-fed, DD-informed protocol.
# ---------------------------------------------------------------------------- #
_DECISION_INSTRUCTION = (
    "----\n"
    "You are a RESEARCH PLANNER growing a research TREE for a deep, well-sourced report. "
    "The RESEARCH MEMORY above — COVERED points (each with its source [S#]), OPEN GAPS, and the "
    "OPEN-GAP LENS — is everything gathered SO FAR. Your job each layer is to FILL THE BLANKS the "
    "memory shows: judge every note/branch by ONE test — does it ADD MEANING (a distinct claim, "
    "corroboration, a contradiction, a concrete figure/date)? EXPAND into the gaps where a missing "
    "meaning would change the report; PRUNE whatever is redundant, off-thesis, answered, dead-end, "
    "low-trust, or over-covered (call prune_branch and name the reason, so pruning is auditable and "
    "the depth budget funds real blanks instead of padding).\n\n"
    # s15/a6 (d182) parts (e)+(f) — the a3 deepening/prune doctrine WALL is shrunk to a lean
    # reminder. The gap signal now lives in the DATA (the OPEN GAPS + OPEN-GAP LENS the render puts
    # right above this prompt) and in the message-chaining (the gather loop records
    # gaps_or_followups per source, fed back as role 'tool'), so this prompt no longer carries the
    # whole argument. Intent preserved — breadth != depth, continue while gaps remain, prune every
    # layer — but NOT a min-layer floor / force-N / deterministic override (d14/d148).
    "Breadth is NOT depth: the first wave gave ONE first-pass note per facet — that OPENS each "
    "facet, it does not FILL it. A facet is STILL an open gap while it rests on a single source, "
    "names a figure/date without its surrounding detail, or carries an unanswered follow-up. So "
    "CONTINUING to expand the gaps is the normal path — calling stop_research right after the first "
    "breadth wave is almost always PREMATURE.\n\n"
    "When the notes already cover the thesis with no meaning-adding gap left, STOP — do "
    "NOT pad with more branches.\n\n"
    "Call ONE tool per turn by replying with ONLY a JSON object and NOTHING else:\n"
    '  {"tool":"expand_branch","args":{"parent":"root","question":"<focused sub-question that fills a MISSING meaning>","rationale":"<the note id/gap it comes from, e.g. S1 gap>"}}\n'
    '  {"tool":"prune_branch","args":{"branch":"<a note id like S5 or a branch id like B2>","reason":"<redundant|off-thesis|answered|dead-end|low-trust>"}}\n'
    '  {"tool":"set_next_direction","args":{"branch":"<branch id>","reason":"<why pursue next>"}}\n'
    '  {"tool":"stop_research","args":{"reason":"<ONLY in a SETTLED turn: you authored NO new expansion this turn AND every branch you expanded earlier already has its findings in the notes above, with no meaning-adding gap left>"}}\n\n'
    # s14/a12 (d154): the add_section / drop_section OUTLINE example calls + the "SHAPE ITS
    # OUTLINE" paragraph were REMOVED here. They invited the model to author the document
    # outline (and, with it, source_ids) during the RESEARCH phase — a surface the generic
    # engine discards (outline_hint=None, d56) while silently dropping the source_ids. The
    # report's sections now come solely from PHASE-2 findings-driven decomposition; source_ids
    # are authored ONLY on the consumed file_write write-planner surface.
    "Ground EVERY expansion in a specific note's gaps_or_followups (cite the note id). "
    "The flow is SEQUENTIAL, not a one-pass choice between expanding and stopping: a turn "
    "that authors ANY expand_branch ENDS RIGHT THERE — you have only OPENED work (the new "
    "child branches hold NO notes yet, so there is nothing to 'stop' on); those branches RUN "
    "as the next layer and return their findings, which you read on a LATER turn. So "
    "stop_research is NOT a co-equal alternative to expanding this turn — it is the move ONLY "
    "in a SETTLED turn: one where you authored NO new expansion AND every branch you expanded "
    "earlier already has its findings in the notes above. When the research has settled that "
    "way, EITHER call stop_research OR write your FINAL TREE PLAN as plain prose (NOT JSON): "
    "the live branches with their questions, the pruned items with reasons, and which branch "
    "to pursue first. ONE tool call per turn."
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
        # s14/a12 (d154): the add_section / drop_section dispatch branches were REMOVED — those
        # tools are no longer offered (see TREE_TOOLS / TREE_TOOL_SPECS), so the model cannot
        # author the dead outline/source_id surface here. ``outline_ops`` therefore stays empty.
        elif name == "stop_research":
            # d95 — the AGENT decided the gathered notes already answer the thesis. Record
            # the reason and STOP the decision loop now (the layer loop reads this and halts
            # with stop_reason='agent_sufficient'). MUST sit before the set_next catch-all so
            # an explicit stop is never mis-routed into a next-direction.
            stop_research = {"reason": str(args.get("reason", "")).strip()}
            break
        else:
            obs = tree.set_next(args)
        # s15/a6 (d182) part (d) — feed a TOOL observation back with role 'tool' (function-result),
        # NOT 'user', so the model does not conflate its OWN tool output (the expand/prune/set_next
        # ack) with a fresh USER instruction. Native Ollama :11434 accepts a role 'tool' message;
        # genuine instructions (_DECISION_NUDGE / _NEXT_DIRECTION_REPROMPT) stay role 'user'.
        convo.append({"role": "tool", "content": obs})

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

# s14/a15 (d160/d161) — the EXACT default breadth sentence inside _DECOMPOSE_INSTRUCTION.
# When the deep-research SHAPE supplies a ``decompose_methodology`` doctrine, this single
# sentence is REPLACED by the shape's text, so the BREADTH SIGNAL is DEFINED IN THE SHAPE
# (reasoned over by the model), not hard-coded here — exactly mirroring how
# ``completeness_stop`` overrides ``_DEFAULT_STOP_SENTENCE``. It must remain a verbatim
# substring of _DECOMPOSE_INSTRUCTION; _decompose_instruction() fails fast below if it drifts.
_DEFAULT_DECOMPOSE_SENTENCE = (
    "Author 2-5 scoped sub-questions that TOGETHER cover the goal, then write a one-line "
    "plan as plain prose (NOT JSON). ONE tool call per turn."
)
assert _DEFAULT_DECOMPOSE_SENTENCE in _DECOMPOSE_INSTRUCTION, (
    "research_tree: _DEFAULT_DECOMPOSE_SENTENCE drifted from _DECOMPOSE_INSTRUCTION — the "
    "shape-defined decompose_methodology substitution would silently no-op"
)


def _decompose_instruction(decompose_criteria: Optional[str] = None) -> str:
    """The decompose-node instruction, with the BREADTH DOCTRINE sourced FROM THE SHAPE (d161).

    ``decompose_criteria`` is the selected deep-research SHAPE's ``decompose_methodology``
    text (d160/d161 — "a detailed report spans MULTIPLE distinct dimensions; scope the real
    facets the thesis implies"). When present it REPLACES the baked default breadth sentence,
    so breadth semantics live in the SHAPE file (editable, no code change) — breadth is a SHAPE
    PROPERTY, not engine code. The substitution is DECLARATIVE METHODOLOGY THE MODEL REASONS
    OVER, never a hard-coded force-exactly-N branch. When empty / None (offline tests, a shape
    without the field) the instruction is BYTE-IDENTICAL to the original ``_DECOMPOSE_INSTRUCTION``
    — no behavioural change off the served path. Mirrors :func:`_decision_instruction` exactly."""
    dc = (decompose_criteria or "").strip()
    if not dc:
        return _DECOMPOSE_INSTRUCTION
    shape_sentence = (
        "BREADTH METHODOLOGY — this is DEFINED IN THE DEEP-RESEARCH SHAPE (your decompose "
        f"doctrine; reason over it to scope the real facets THIS goal implies): {dc} Author "
        "one scoped sub-question for EACH distinct facet you identify, then write a one-line "
        "plan as plain prose (NOT JSON). ONE tool call per turn."
    )
    return _DECOMPOSE_INSTRUCTION.replace(_DEFAULT_DECOMPOSE_SENTENCE, shape_sentence, 1)


async def run_decompose_node(
    transport: Any,
    *,
    goal: str,
    tree: Tree,
    config: TreeConfig,
    methodology: Optional[str] = None,
    decompose_criteria: Optional[str] = None,
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
        "content": f"{_methodology_block(methodology)}GOAL: {goal}\n\n{_decompose_instruction(decompose_criteria)}",
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
        # s15/a6 (d182) part (d) — the seed expand ack is the model's own tool output: feed it as
        # role 'tool', not 'user' (same anti-conflation chaining fix as the decision node).
        convo.append({"role": "tool", "content": obs})
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
        decompose_criteria: Optional[str] = None,
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
        # s14/a15 (d161) — the SHAPE's decompose_methodology doctrine, handed to the
        # DECOMPOSE-FIRST seed so breadth (≥3 scoped facets) is a SHAPE PROPERTY the model
        # reasons over (None → the baked-in default decompose wording, byte-identical).
        self.decompose_criteria = decompose_criteria
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
        # s15/a21 — the SURFACED grow-error "<Type>: <msg>" the drive loop records here when a
        # grow round RAISES (the runtime also logs the full traceback to stderr). None while no
        # round raised; the served grow_trace + the gate read it so a crash is VISIBLE, never a
        # silent early-stop (d186 — a grow failure is a fixable design bug to surface, not hide).
        self.grow_error: Optional[str] = None

    def _research_node(self, nid: str, question: str, depends_on: tuple[str, ...],
                       memory_index: str = NEW_MEMORY_SENTINEL) -> PlanNode:
        """Build ONE GATHER PlanNode (SOURCE-AGNOSTIC, growing-visibility) for a branch.

        as4 DE-WEB (d227/d241/d186): the node is TOOL-LESS — it does NOT bind ``web_search``
        (or any gather tool). Like every in-plan node (d242) it starts with only
        ``get_bundles`` + ``finish`` and SELF-SELECTS its gather bundle (web / vector-db /
        codebase-read / files); that bundle's tool drives the gather, and the grower drives
        expand/prune/stop over WHATEVER structured artifact it yields. The grower owns NO web
        vocabulary. SB-RR (d292/d293): the gather node is a WORKER-default node (d273) carrying
        the :data:`~specialization.seed.RESEARCH_METHODOLOGY_SPEC` METHODOLOGY spec — research
        is a SPECIALIZATION, NOT a role. That spec's body ("self-select your gather bundle
        first…") is what makes the worker self-select the gather bundle and reach the unified
        worker loop's gather behavior; the ROLE_RESEARCHER role is RETIRED, so nothing routes
        the node by a role. The question rides the task framing; ``tool_args['query']`` is
        carried bookkeeping only (read by ``_node_question`` for ingest/decision read-back) —
        NOT a tool binding, since ``tool`` is ``None``."""
        q = (question or "").strip() or self.goal
        # SB-RR (d292): the research-METHODOLOGY spec is the self-select lever; compose it AHEAD
        # of the round's output-quality spec (``self.spec``, e.g. research-analyst) so the gather
        # posture leads. Dedup if the caller already passed the methodology spec.
        extra = (self.spec,) if self.spec and self.spec != RESEARCH_METHODOLOGY_SPEC else ()
        specs = (RESEARCH_METHODOLOGY_SPEC,) + extra
        return PlanNode(
            id=nid,
            task=f"[research · gap] {position_framing('research')}\n\n{q}",
            spec=RESEARCH_METHODOLOGY_SPEC,
            specs=specs,
            depends_on=depends_on,
            # WORKER-default (d273): every spawned node is a worker; the gather behavior comes
            # from the research-methodology SPEC self-selecting the gather bundle, NOT a role.
            role=ROLE_WORKER,
            # TOOL-LESS (d242 / as4 de-web): self-selects its gather bundle; the grower does
            # not assume web_search. ``tool_args['query']`` is inert carried bookkeeping.
            tool=None,
            tool_args={"query": q[:200]},
            # MEMORY-BY-HANDLE (d221): bind each gather node to THIS run's research/complex
            # memory — also the SOURCE-AGNOSTIC marker the grower folds it by (not the tool).
            research_memory_handle=self.state.memory_handle,
            # MEMORY-INDEX (d285 SB-3): the planner-authored memory CHOICE the branch's brief
            # carried (an index to continue, or <<NEW>>). Carried onto the node's brief surface;
            # the run still gathers into the shared ``self.state`` memory (per-branch OPENING by
            # this index is SB-4) — SB-3 only surfaces the choice.
            memory_index=normalize_brief_memory_index(memory_index),
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
        # P4 FRONTIER-FIRST SEED: a follow-up research plan whose session already holds
        # an OPEN persisted frontier (branches the model authored but never gathered —
        # a budget-cancelled wave, or a prior plan's unexecuted expansions) RESUMES
        # those branches verbatim instead of re-decomposing the goal from scratch.
        # This kills the live prune/re-add loop (every plan rebuilt B1→B5, duplicated
        # to B12, and starved facets never executed). The questions are the MODEL's
        # own earlier authorship — the engine replays, never composes. A fresh goal
        # (no persisted frontier) keeps the decompose-first seed unchanged.
        persisted = self.state.read_frontier()
        if persisted:
            nodes: list[PlanNode] = []
            for i, rec in enumerate(persisted, start=1):
                node = self._research_node(
                    f"s1_F{i}",
                    str(rec.get("question") or ""),
                    (),
                    memory_index=str(rec.get("memory_index") or "") or NEW_MEMORY_SENTINEL,
                )
                self._consumed.discard(node.id)
                nodes.append(node)
            if nodes:
                return nodes
        branches = await run_decompose_node(
            self.transport,
            goal=self.goal,
            tree=self.tree,
            config=self.config,
            methodology=self.methodology,
            decompose_criteria=self.decompose_criteria,
        )
        if not branches:
            return []
        nodes = []
        for b in branches:
            # seed frontier nodes are INDEPENDENT (depends_on=()) so they gather concurrently;
            # the decision node reads ALL their persisted notes back (cross-layer visibility).
            node = self._research_node(
                f"s1_{b.id}", b.question, (), memory_index=b.memory_index)
            self._consumed.discard(node.id)
            nodes.append(node)
        # P4: the decomposed seed IS the initial open frontier — persist it so a plan
        # cancelled before its first grow layer still hands its facets to the follow-up.
        self.state.record_frontier([
            {"id": b.id, "question": b.question, "rationale": b.rationale,
             "memory_index": b.memory_index}
            for b in branches
        ])
        return nodes

    @staticmethod
    def _is_research_node(node: Any) -> bool:
        """A GATHER node — recognized SOURCE-AGNOSTICALLY (as4 de-web, d227/d241), never by a
        web tool and (SB-RR, d292/d293) never by a role. The :data:`ROLE_RESEARCHER` role is
        RETIRED; a gather node is now a WORKER carrying the research-methodology spec. The
        grower folds its OWN authored gather nodes by the research/complex-MEMORY HANDLE it
        binds onto each (``research_memory_handle`` — set in :meth:`_research_node` and the
        tool-less FALLBACK seed). This is the grower's INGEST recognizer (which completed nodes
        to fold into the ResearchState), NOT a runtime dispatch discriminator — the runtime
        routes purely by the node's self-selected bundle. A legacy ``web_search``-bound node
        still folds (back-compat)."""
        if getattr(node, "research_memory_handle", None):
            return True
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
                # SOURCE-AGNOSTIC gather artifacts (as4 de-web, d227/d241): fold the structured
                # note/record the node's SELF-SELECTED gather bundle yields under EITHER the web
                # vocabulary (``article_notes`` / ``fetched``) OR a generic key (``notes`` /
                # ``records``), so a non-web source (vector-db, codebase, files) folds through
                # the SAME ingest with no web-specific shape. The decision node reasons over the
                # note gaps regardless of which source produced them.
                notes = [
                    dict(x) for x in (tv.get("article_notes") or tv.get("notes") or [])
                    if isinstance(x, Mapping)
                ]
                fetched = [
                    dict(x)
                    for x in (tv.get("fetched") or tv.get("records") or tv.get("chunks") or [])
                    if isinstance(x, Mapping)
                ]
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
        # Read the PERSISTED state back (d49 anti-hallucination) — the gathered notes/sources.
        # s14/a12 (d154): the document-OUTLINE read-back (render_outline_for_decision) was
        # REMOVED from the decision prompt. It invited the model to author an outline (and route
        # source_ids into it) during research — a surface the generic engine discards
        # (outline_hint=None, d56). The report's sections come from PHASE-2 findings-driven
        # decomposition; source_ids live ONLY on the consumed file_write surface.
        # s15/a15 (d185) — render the decision state AS THE GRAPH the loop walks: pass the live
        # ``tree`` so authored-but-ungathered concerns (expand) and pruned concerns (collapse)
        # are folded in, and the decision node's next expand/prune is DRIVEN BY the graph (its
        # open gap edges + single-source flags), not only the flattened narrative.
        state_render = self.state.render_for_decision(tree=self.tree)
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
        # ``decision.outline_ops`` is now always empty (the outline tools are no longer offered);
        # the call is a harmless no-op kept so the channel can be re-enabled without a code change.
        self.state.append_outline_ops(decision.outline_ops)
        # s15/a15 (d185) — snapshot the per-concern GRAPH SHAPE this layer (a compact summary,
        # not the full serialization) so the served parity trace + the a14 gate can assert the
        # research is GRAPH-shaped (per-concern nodes with note→source + gap edges), not a flat
        # blob — and observe the graph the decision actually walked.
        graph = self.state.concern_graph(tree=self.tree)
        self.layers.append({
            "layer": layer,
            "gathered": gathered,
            "expanded": [b.id for b in decision.new_branches],
            "pruned": [p.get("target") for p in decision.pruned],
            "decision_turns": decision.turns,
            "stop_research": decision.stop_research,
            "graph_shape": "per_concern_graph",
            "graph_concerns": len(graph.nodes),
            "graph_settled": len(graph.settled()),
            "graph_open_gaps": len(graph.open_gaps()),
            "graph_collapsed": len(graph.collapsed()),
        })
        # d184 — THE ENGINE HONORS THE expand_branch CONTRACT. The rewritten expand_branch
        # description PROMISES the model "this sub-topic WILL be gathered as a new round"; the
        # broken link was the engine checking stop_research BEFORE new_branches, so an expansion
        # the model authored was silently dropped — the engine NOT delivering the tool's stated
        # contract. So whenever the model authored ANY new branch this layer, those branches RUN
        # as the next wave (the contract), and a stop_research raised in the SAME pass cannot
        # cancel them (the new branches hold NO notes yet — nothing is gathered to "stop" on).
        # This is the engine doing what the tool description says, NOT a stop/expand precedence
        # seatbelt: the stop signal is decided ONLY in a settled pass that authored no expansion.
        if not decision.new_branches:
            if decision.stop_research is not None:
                # STOP — the agent decided every concern is settled/collapsed (d95).
                self.stop_reason = "agent_sufficient"
            else:
                # STOP — no new branch authored this layer → done (no fabricated structure).
                self.stop_reason = "no_expansion"
            # P4: a settled stop CLEARS the persisted frontier (nothing left to seed a
            # follow-up plan from) — a budget stop outside this loop leaves the last
            # recorded frontier standing, which is exactly what the follow-up resumes.
            self.state.record_frontier([])
            return [], self.stop_reason
        # MAP each model-authored branch → a research PlanNode. GROWING VISIBILITY: depend on
        # EVERY prior node so the new layer sees all earlier findings (the §2c semantic the
        # frozen unroll already used). The next layer's gather feeds the next decision node.
        prior_ids = tuple(n.id for n in dag.nodes)
        new_nodes = [
            self._research_node(f"g{layer + 1}_{b.id}", b.question, prior_ids,
                                memory_index=b.memory_index)
            for b in decision.new_branches
        ]
        # P4 FRONTIER PERSISTENCE: the branches about to be dispatched ARE the open
        # frontier — persist them (model-authored, verbatim) so a follow-up research
        # plan that lands before they are settled SEEDS from them instead of
        # re-decomposing (the created-but-never-executed fix: a budget-cancelled wave
        # is picked up by the next plan, not rebuilt as B1→B5 duplicates).
        self.state.record_frontier([
            {"id": b.id, "question": b.question, "rationale": b.rationale,
             "memory_index": b.memory_index}
            for b in decision.new_branches
        ])
        return new_nodes, None

    def concern_graph(self) -> ConcernGraph:
        """The current EXPLICIT per-concern research GRAPH (d185) — the live projection over
        the persisted ResearchState + the loop's Tree (concern → notes → sources, gaps =
        edges). Exposed so a smoke/gate can serialize it (``to_dict``) and assert the research
        is GRAPH-shaped (per-concern nodes with note→source + gap edges), NOT a flat blob."""
        return self.state.concern_graph(tree=self.tree)
