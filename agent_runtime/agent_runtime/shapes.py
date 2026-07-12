"""Declarative plan-SHAPE definitions, loaded from text files (blueprint §2a/d5).

Plan SHAPES are defined as DECLARATIVE TEXT FILES on disk (d5) — TOML, so they
are human-editable and comment-friendly, and ``tomllib`` parses them with zero
third-party deps (Python ≥3.11). This module:

* :class:`ShapeSpec` — the parsed shape (name, description, role templates per
  round + final round, ``max_iter`` default ceiling, ``hard_cap`` safety bound,
  edge policy). It is the runtime's view of what a shape IS.
* :func:`load_shapes` / :func:`load_shape` — read the ``shapes/`` directory (or
  any directory) and return the parsed specs. The catalog of names is harvested
  here so a planner shape-SELECTION call can advertise the available shapes as a
  JSON-schema ``enum`` (§2a) without hard-coding them.
* :meth:`ShapeSpec.effective_max_iter` — honor a UI-set override (s4) BOUNDED by
  the shape's ``hard_cap`` (the runtime never exceeds it, whatever the UI sets).

Pure data + file I/O — no model call. A shape is an EXECUTION DISCIPLINE + optional
DOCTRINE only — it NEVER pre-bakes a node graph (s16/a3 d239/d247: RETIRE THE UNROLL).
The deep-research FAMILY (``execution == "deep-research"``) carries the iterative-loop
cadence (``max_iter``/``hard_cap`` depth ceiling), the breadth/stop DOCTRINE the model
reasons over (``decompose_methodology`` / ``completeness_stop``), and the growth knobs
(``fan_out`` / ``max_layers`` / ``max_sources``). Its research TOPOLOGY is AUTHORED at
runtime by REASONING — the engine emits a TOOL-LESS self-selecting research seed and the
:class:`~agent_runtime.research_tree.DagGrower` (decompose-first via ``run_decision_node``)
grows it on note gaps — NOT a deterministic ``unroll_shape`` populating a fixed DAG and
NOT a shape-bound ``web_search`` position (both DELETED in s16/a3). There is no per-shape
executor and no deterministic node population: a shape is fully described by
``{ name, description, execution, max_iter, hard_cap }`` + doctrine/grow fields.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .factory import PlanError

# Deep-research POSITIONS (d48). The bounded, fixed behavior vocabulary the research
# loop uses — DECOUPLED from the node-role vocabulary (:data:`~agent_runtime.factory.
# VALID_ROLES` = {worker, synthesizer}). The engine seed builder + the
# :class:`~agent_runtime.research_tree.DagGrower` map a position onto a worker/synthesizer
# NODE and inject the matching :data:`~agent_runtime.roles.POSITION_FRAMINGS` text into the
# node's TASK (behavior via PROMPTING, not a role code-switch). Positions are NOT node roles
# and are NOT LLM-extensible beyond this set. (s16/a3: the shape no longer DECLARES per-round
# position sequences — the research topology is reasoned at runtime, not unrolled.)
VALID_POSITIONS: frozenset[str] = frozenset(
    {"research", "critic", "synthesis", "verify", "worker"}
)

# The default on-disk shape directory (this package's ``shapes/``). Kept beside
# the code so a checkout ships the built-in shapes; the s4 UI may add more files.
SHAPES_DIR = Path(__file__).resolve().parent / "shapes"

# The canonical name of the deep-research shape (the one s3 must execute).
DEEP_RESEARCH = "deep-research"

# The execution DISCIPLINE a shape declares (blueprint §2a). It governs HOW the
# in-process runtime dispatches the planner-emitted nodes — NOT what the nodes are:
#   * "sequential"    — linear: at most ONE ready node in flight at a time (strict
#     single-file order; the `first_ready_action` wave-of-one);
#   * "concurrent"    — modular-parallel: EVERY independent ready node launches at
#     once (the wave the runtime already drives);
#   * "deep-research" — the iterative research family: the engine emits a TOOL-LESS
#     self-selecting research seed and the :class:`~agent_runtime.research_tree.DagGrower`
#     AUTHORS the topology by reasoning (decompose-first → grow on note gaps), bounded by
#     ``max_iter``/``max_layers``. There is NO deterministic unroll (s16/a3) — this token
#     IS the deep-research identity (see :meth:`ShapeSpec.is_deep_research`).
# Validated fail-fast (like the role names) so a typo never silently degrades to a
# default. The string→runtime mode mapping lives in :mod:`agent_runtime.scheduler`.
VALID_EXECUTION: frozenset[str] = frozenset(
    {"sequential", "concurrent", "deep-research"}
)

# RP-6b (d359/d361) — the deep-research FLOW is an ordered sequence of PHASES the SHAPE
# DECLARES (definition layer), NOT a research-first seed + a fixed engine phase enum baked
# in code (d341/d319). A phase declares a ``kind`` (what the phase does) and a ``spec_role``
# (which class of specialization its nodes carry) so the right spec lands on the right node
# (Bug A d355/d356): a RESEARCH phase's nodes carry the research/analysis spec, the WRITE
# phase's node carries the writer spec — never crossed. The engine READS these; a shape with
# NO phases (linear / schedule-leg / codebase-summary / write-file) keeps the byte-identical
# fallback vocabulary (:data:`_DEFAULT_FOLLOWUP_PLANS`).
VALID_PHASE_KINDS: frozenset[str] = frozenset({"research", "write"})
VALID_SPEC_ROLES: frozenset[str] = frozenset({"research", "writer"})

# A phase ``kind`` → the iterative-planner PLAN kind (the ``decide_followup`` vocabulary the
# generic loop reasons over). The loop accepts the bare phase kind on the SEED and the
# ``*_plan`` kind on a follow-up decision; both name the same phase.
_PHASE_PLAN_KIND: dict[str, str] = {"research": "research_plan", "write": "write_plan"}

# LOOP-CONTROL follow-ups that are ALWAYS available — a standalone QA/review pass and the
# terminal exit. These are NOT linear deep-research phases (the shape declares research →
# write); the engine appends them after the shape's declared phase plan-kinds.
_AUX_FOLLOWUP_PLANS: tuple[str, ...] = ("review_plan", "done")

# The offline/back-compat follow-up vocabulary a shape with NO declared phases yields — the
# byte-identical retired hardcoded enum (research → write → review → done).
_DEFAULT_FOLLOWUP_PLANS: tuple[str, ...] = (
    "research_plan", "write_plan", "review_plan", "done"
)


class ShapeError(PlanError):
    """A shape text file is missing/malformed or declares an unknown position."""


@dataclass(frozen=True)
class PhaseSpec:
    """One declared PHASE of a multi-phase shape (RP-6b / d359/d361).

    ``kind`` is what the phase DOES (a member of :data:`VALID_PHASE_KINDS`); ``spec_role``
    is the CLASS of specialization its nodes carry (:data:`VALID_SPEC_ROLES`) — the shape's
    per-phase spec-routing that keeps a writer spec off a research node and vice-versa (Bug A
    d355/d356). Validated fail-fast like the execution token."""

    kind: str
    spec_role: str

    def __post_init__(self) -> None:
        kind = str(self.kind or "").strip().lower()
        role = str(self.spec_role or "").strip().lower()
        if kind not in VALID_PHASE_KINDS:
            raise ShapeError(
                f"phase declares unknown kind {self.kind!r}; valid: {sorted(VALID_PHASE_KINDS)}"
            )
        if role not in VALID_SPEC_ROLES:
            raise ShapeError(
                f"phase {kind!r}: unknown spec_role {self.spec_role!r}; "
                f"valid: {sorted(VALID_SPEC_ROLES)}"
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "spec_role", role)

    @property
    def plan_kind(self) -> str:
        """The iterative-planner PLAN kind this phase maps onto (research → research_plan)."""
        return _PHASE_PLAN_KIND[self.kind]


@dataclass(frozen=True)
class ShapeSpec:
    """One declarative plan shape parsed from a text file (d5).

    Attributes
    ----------
    name:
        The shape id (the planner selects shapes by this name).
    description:
        Human/LLM-facing one-liner (used in the shape-selection enum's context).
    max_iter:
        The DEFAULT round ceiling — a UI-OVERRIDABLE value (d5). The deep-research
        shape ships ~10 (9 research+critic rounds + 1 final round).
    hard_cap:
        The absolute round bound the runtime never exceeds regardless of the UI
        override (shared-GPU safety). Defaults to ``max_iter`` when unset.
    edges:
        The declared edge policy (informational for the s4 UI + readers). The research
        topology is reasoned at runtime by the grower, not driven by this field.
    source:
        The file the spec was parsed from (provenance for the UI).
    completeness_stop:
        P2.4 (d131/d132.D) — the COMPLETENESS-DRIVEN stop SIGNAL the shape hands the
        LLM: "keep poking the right gap-questions until every blank is filled, then
        STOP". This is the deep-research shape's stop semantics DEFINED IN THE SHAPE
        (text the model reasons over), NOT an arbitrary depth cap hard-coded in
        ``research_tree._DECISION_INSTRUCTION``. Empty string → the runtime keeps its
        baked-in default stop wording (byte-identical, offline / shapes without it).
    decompose_methodology:
        s14/a15 (d160/d161) — the BREADTH doctrine the shape hands the DECOMPOSE-FIRST seed:
        "a detailed/exhaustive report spans MULTIPLE distinct dimensions, so open the
        investigation by scoping the real facets the thesis implies (timeline, key events,
        figures, causes, impact) — a single facet almost never covers a detailed report."
        This is the SHAPE-level breadth property (d161 — breadth is a SHAPE property, NOT
        engine code): it is substituted into ``research_tree._DECOMPOSE_INSTRUCTION`` (via
        ``_decompose_instruction``) so the seed RELIABLY authors ≥3 scoped sub-questions as
        METHODOLOGY THE MODEL REASONS OVER — NOT a hard-coded force-exactly-N branch and NOT
        an engine seed force-count (those are the d14/d148/d161-rejected hacks). Editing THIS
        string changes the breadth behaviour — no code change. Empty string → the runtime keeps
        its baked-in default decompose wording (byte-identical, offline / shapes without it).
        Mirrors :attr:`completeness_stop` exactly, for the seed instead of the stop.
    deny_domains:
        P2.4 (d131/d133) — the shape-level SOURCE deny-list, the cross-cutting source
        policy expressed at the SHAPE (TOOL-ENFORCED by the P2.1 web-tool baseline +
        the per-call ``exclude_domains`` arg). ``wikipedia.org`` (+ wikimedia /
        wiktionary) is the baseline; a shape names the domains the research must never
        fetch or cite. Empty tuple → only the tool's always-on baseline applies.
    expand_on_gaps:
        P2.5b (d134/d135) — the DECLARATIVE iterative-gap-expansion capability that lets the
        GENERIC engine reproduce ``run_research_tree``'s ITERATIVE breadth. The engine emits a
        TOOL-LESS self-selecting research SEED and tags the :class:`~agent_runtime.factory.PlanDAG`
        ``growable``; the runtime's drive loop then GROWS the DAG round-by-round by invoking the
        SAME ``research_tree.run_decision_node`` over the persisted ``ResearchState`` — each note's
        gaps author the next layer's research nodes (growing-visibility edges), bounded by
        ``max_layers`` / ``fan_out`` + ``no_expansion`` + the ``completeness_stop`` the model
        reasons over (``stop_research``). s16/a3 (d239/d247): the deterministic unroll is RETIRED,
        so growth is the ONLY research mode — this flag remains the declarative marker that the
        deep-research engine builds a growable seed (the canonical ``deep-research`` shape sets it
        True). The research topology is always REASONED, never a fixed node graph.
    fan_out:
        P2.5b — the per-decision-layer expansion cap handed to the grower's
        :class:`~agent_runtime.research_tree.Tree` (≤ this many ``expand_branch`` keeps per
        layer). 0 → fall back to the runtime's ``TreeConfig.fan_out``. Only consulted when
        ``expand_on_gaps`` is set.
    max_layers:
        P2.5b — the GROWTH bound: the maximum number of research layers the growable drive
        loop runs (seed = layer 1; growth adds up to ``max_layers - 1`` more). A hard
        termination-safety ceiling alongside ``no_expansion`` / ``stop_research``; further
        clamped by ``TreeConfig.depth`` (the user-fixed depth ceiling). 0 → fall back to the
        runtime's ``TreeConfig.depth``. Only consulted when ``expand_on_gaps`` is set.
    phases:
        RP-6b (d359/d361) — the ORDERED PHASES the shape DECLARES (definition layer), each a
        :class:`PhaseSpec` naming what the phase does (``kind``) and the class of specialization
        its nodes carry (``spec_role``). The engine READS this list instead of baking the flow in
        code: the FIRST phase SEEDS the iterative-planner loop (was the hardcoded research-first
        seed in ``agentic.route_research``), the phase ORDER drives the default phase transition,
        the follow-up plan vocabulary DERIVES from these kinds (:attr:`followup_plans` — was the
        fixed ``FOLLOWUP_PLANS`` enum in ``planner.py``), and the per-phase ``spec_role`` routes
        the right spec to the right node (Bug A d355/d356). Empty tuple (a non-phased shape) →
        the byte-identical fallback vocabulary (:data:`_DEFAULT_FOLLOWUP_PLANS`); non-deep-research
        shapes never read these, so an empty default is truthful and inert.
    """

    name: str
    description: str = ""
    max_iter: int = 1
    hard_cap: int = 0
    edges: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    execution: str = "concurrent"
    completeness_stop: str = ""
    decompose_methodology: str = ""
    deny_domains: tuple[str, ...] = ()
    expand_on_gaps: bool = False
    fan_out: int = 0
    max_layers: int = 0
    max_sources: int = 0
    phases: tuple[PhaseSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise ShapeError("shape file declares no 'name'")
        if int(self.max_iter) < 1:
            raise ShapeError(f"shape {self.name!r}: max_iter must be ≥1")
        # EXECUTION discipline (blueprint §2a): a known token only, so a typo in a
        # shape file fails fast instead of silently degrading to the default — the
        # same fail-fast posture the role names use. Normalised to lower-case.
        execution = str(self.execution or "").strip().lower() or "concurrent"
        if execution not in VALID_EXECUTION:
            raise ShapeError(
                f"shape {self.name!r}: unknown execution {self.execution!r}; "
                f"valid: {sorted(VALID_EXECUTION)}"
            )
        object.__setattr__(self, "execution", execution)
        # hard_cap defaults to max_iter (never below it) when unset/too small.
        if int(self.hard_cap) < int(self.max_iter):
            object.__setattr__(self, "hard_cap", int(self.max_iter))

    @property
    def is_deep_research(self) -> bool:
        """True iff this shape is the deep-research FAMILY — keyed on the EXECUTION
        DISCIPLINE token (s16/a3 d239/d247: re-keyed off the retired round/final roles).

        A ``deep-research`` shape's research TOPOLOGY is AUTHORED at runtime by reasoning
        (the engine's tool-less self-selecting research seed + the
        :class:`~agent_runtime.research_tree.DagGrower`'s decompose-first growth), NOT
        unrolled from a fixed node graph. Every OTHER shape (``sequential`` /
        ``concurrent``) is an execution DISCIPLINE whose acyclic DAG is AUTHORED
        node-by-node by the incremental planner. The router keys off this declarative
        execution token, never a hard-coded shape name — so shapes stay plug-n-play."""
        return self.execution == "deep-research"

    # --- RP-6b (d359/d361): the DECLARED-PHASE accessors the engine READS ------------------
    @property
    def first_phase_kind(self) -> str:
        """The kind of the FIRST declared phase — what the iterative-planner loop is SEEDED
        with (replaces ``agentic.route_research``'s hardcoded ``first_plan_kind="research"``).
        Falls back to ``"research"`` for a deep-research shape that declares no phases."""
        return self.phases[0].kind if self.phases else "research"

    @property
    def followup_plans(self) -> tuple[str, ...]:
        """The follow-up PLAN vocabulary DERIVED from the declared phases (+ the always-on
        loop-control ``review_plan`` / ``done``) — the source of ``planner.FOLLOWUP_PLANS``
        (replaces the fixed engine enum). Order-preserving, de-duplicated. A shape with no
        phases yields the byte-identical retired default (:data:`_DEFAULT_FOLLOWUP_PLANS`)."""
        if not self.phases:
            return _DEFAULT_FOLLOWUP_PLANS
        ordered: list[str] = []
        for kind in tuple(p.plan_kind for p in self.phases) + _AUX_FOLLOWUP_PLANS:
            if kind not in ordered:
                ordered.append(kind)
        return tuple(ordered)

    def spec_role_for(self, kind: str) -> str:
        """The ``spec_role`` the shape declares for the phase of the given ``kind`` — the
        per-phase spec-routing the engine reads to keep a writer spec off a research node and
        vice-versa (Bug A). Falls back to the role-conventional default for a non-phased shape
        (research kind → "research", otherwise "writer")."""
        for phase in self.phases:
            if phase.kind == kind:
                return phase.spec_role
        return "research" if kind == "research" else "writer"

    def next_phase_plan(self, current: str) -> str:
        """The DEFAULT next plan after the phase named by ``current`` completes — the shape's
        declared phase ORDER (replaces the loop's hardcoded ``default_next``). ``current`` may
        be a phase kind (``"research"``) or a plan kind (``"research_plan"``). Returns the NEXT
        phase's plan kind, or ``"done"`` when ``current`` is the last (or unknown) phase.

        A shape with NO declared phases falls back to the canonical deep-research phase ORDER
        (``research → write → done``) so a degenerate/offline shape keeps the byte-identical
        transition the retired hardcoded ``default_next`` produced (never skips the write phase)."""
        plan_to_kind = {v: k for k, v in _PHASE_PLAN_KIND.items()}
        cur_kind = plan_to_kind.get(current, current)
        kinds = [p.kind for p in self.phases] if self.phases else ["research", "write"]
        if cur_kind in kinds:
            nxt = kinds.index(cur_kind) + 1
            if nxt < len(kinds):
                return _PHASE_PLAN_KIND.get(kinds[nxt], "done")
        return "done"

    def effective_max_iter(self, override: Optional[int] = None) -> int:
        """The round count to run, honoring a UI override BOUNDED by ``hard_cap``.

        ``override`` is the s4 UI-set value (read from SQLite); ``None`` means use
        the shape default. The result is clamped to ``[1, hard_cap]`` so a UI
        value can lower OR raise the default but can NEVER exceed the safety bound
        (and a nonsensical ≤0 override falls back to the default)."""
        chosen = self.max_iter if override is None else int(override)
        if chosen < 1:
            chosen = self.max_iter
        return max(1, min(chosen, self.hard_cap))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "max_iter": self.max_iter,
            "hard_cap": self.hard_cap,
            # s17 (d248/d249): the transitional round_roles/final_roles empty-[] shim is
            # REMOVED — the Shapes screen now renders "shape = discipline + doctrine; the
            # planner authors topology" and reads no fixed round topology.
            "edges": dict(self.edges),
            "source": self.source,
            "execution": self.execution,
            "completeness_stop": self.completeness_stop,
            "decompose_methodology": self.decompose_methodology,
            "deny_domains": list(self.deny_domains),
            "expand_on_gaps": self.expand_on_gaps,
            "fan_out": self.fan_out,
            "max_layers": self.max_layers,
            "max_sources": self.max_sources,
            # RP-6b (d359/d361) — the declared phases + per-phase spec-routing (the readers /
            # the s4 UI see the flow the engine now reads from the shape, not baked in code).
            "phases": [
                {"kind": p.kind, "spec_role": p.spec_role} for p in self.phases
            ],
        }


def _parse_shape(path: Path) -> ShapeSpec:
    """Parse one TOML shape file into a :class:`ShapeSpec`."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ShapeError(f"cannot parse shape file {path}: {exc}") from exc
    # RP-6b (d359/d361) — parse the DECLARED phases (per-PhaseSpec validation fails fast on a
    # bad kind/spec_role). A shape with no ``[[phases]]`` table → empty tuple (non-phased).
    phases = tuple(
        PhaseSpec(kind=p.get("kind", ""), spec_role=p.get("spec_role", ""))
        for p in data.get("phases", []) or ()
    )
    return ShapeSpec(
        name=str(data.get("name") or path.stem),
        description=str(data.get("description", "")),
        max_iter=int(data.get("max_iter", 1)),
        hard_cap=int(data.get("hard_cap", 0)),
        edges=dict(data.get("edges", {})),
        source=str(path),
        execution=str(data.get("execution", "concurrent")),
        completeness_stop=str(data.get("completeness_stop", "")),
        decompose_methodology=str(data.get("decompose_methodology", "")),
        deny_domains=tuple(str(d) for d in data.get("deny_domains", ())),
        expand_on_gaps=bool(data.get("expand_on_gaps", False)),
        fan_out=int(data.get("fan_out", 0)),
        max_layers=int(data.get("max_layers", 0)),
        max_sources=int(data.get("max_sources", 0)),
        phases=phases,
    )


def load_shapes(shapes_dir: Optional[Path] = None) -> dict[str, ShapeSpec]:
    """Load every ``*.toml`` shape in ``shapes_dir`` (default :data:`SHAPES_DIR`).

    Returns ``{name: ShapeSpec}``. A missing directory yields an empty catalog
    (not an error — a checkout without custom shapes is valid). A malformed file
    raises :class:`ShapeError` so a broken shape never silently disappears."""
    directory = Path(shapes_dir) if shapes_dir is not None else SHAPES_DIR
    catalog: dict[str, ShapeSpec] = {}
    if not directory.is_dir():
        return catalog
    for path in sorted(directory.glob("*.toml")):
        spec = _parse_shape(path)
        catalog[spec.name] = spec
    return catalog


def load_shape(name: str, shapes_dir: Optional[Path] = None) -> ShapeSpec:
    """Load a single shape by name (raises :class:`ShapeError` if absent)."""
    catalog = load_shapes(shapes_dir)
    if name not in catalog:
        raise ShapeError(
            f"shape {name!r} not found in {shapes_dir or SHAPES_DIR} "
            f"(available: {sorted(catalog)})"
        )
    return catalog[name]


def shape_names(shapes_dir: Optional[Path] = None) -> list[str]:
    """The sorted catalog of shape names — the planner's shape-selection enum (§2a)."""
    return sorted(load_shapes(shapes_dir))


__all__ = [
    "ShapeSpec",
    "PhaseSpec",
    "ShapeError",
    "SHAPES_DIR",
    "DEEP_RESEARCH",
    "VALID_EXECUTION",
    "VALID_POSITIONS",
    "VALID_PHASE_KINDS",
    "VALID_SPEC_ROLES",
    "load_shapes",
    "load_shape",
    "shape_names",
]
