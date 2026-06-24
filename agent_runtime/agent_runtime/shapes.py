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

Pure data + file I/O — no model call. A shape that declares ``round_roles`` /
``final_roles`` (the bounded cyclic family, e.g. ``deep-research``) is UNROLLED
into a bounded acyclic role-tagged :class:`~agent_runtime.factory.PlanDAG` by the
GENERIC :func:`unroll_shape` here — which the SAME
:class:`~agent_runtime.runtime.AgentRuntime` then executes like any other DAG. So
there is NO per-shape executor: adding a cyclic shape is adding one text file
(a3 re-architecture; this replaced the deleted ``DeepResearchExecutor``).
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .factory import PlanDAG, PlanError, PlanNode
from .roles import ROLE_SYNTHESIZER, ROLE_WORKER, position_framing

# Deep-research ROUND POSITIONS (d48). A cyclic shape declares its per-round
# sequence as POSITIONS — a FIXED, bounded vocabulary, DECOUPLED from the node-role
# vocabulary (:data:`~agent_runtime.factory.VALID_ROLES` = {worker, synthesizer}).
# The unroll maps each position onto a worker/synthesizer NODE and injects the
# matching :data:`~agent_runtime.roles.POSITION_FRAMINGS` text into the node's TASK
# (behavior via PROMPTING, not a role code-switch). Positions are NOT node roles
# and are NOT LLM-extensible beyond this set.
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
#   * "deep-research" — the bounded cyclic family: the shape declares round/final
#     roles and is UNROLLED by :func:`unroll_shape` into an acyclic role-tagged DAG
#     the GENERIC runtime then drives (its growing-visibility edges serialise the
#     rounds). Declared here so the shape file documents its own discipline.
# Validated fail-fast (like the role names) so a typo never silently degrades to a
# default. The string→runtime mode mapping lives in :mod:`agent_runtime.scheduler`.
VALID_EXECUTION: frozenset[str] = frozenset(
    {"sequential", "concurrent", "deep-research"}
)


class ShapeError(PlanError):
    """A shape text file is missing/malformed or declares an unknown position."""


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
    round_roles:
        The node roles emitted each NON-FINAL round, in order (deep-research:
        ``["research", "critic"]``).
    final_roles:
        The node roles emitted in the single FINAL round, in order (deep-research:
        ``["research", "synthesis", "verify"]``).
    edges:
        The declared edge policy (informational for the s4 UI + readers; the
        unroll logic lives in the executor).
    source:
        The file the spec was parsed from (provenance for the UI).
    completeness_stop:
        P2.4 (d131/d132.D) — the COMPLETENESS-DRIVEN stop SIGNAL the shape hands the
        LLM: "keep poking the right gap-questions until every blank is filled, then
        STOP". This is the deep-research shape's stop semantics DEFINED IN THE SHAPE
        (text the model reasons over), NOT an arbitrary depth cap hard-coded in
        ``research_tree._DECISION_INSTRUCTION``. Empty string → the runtime keeps its
        baked-in default stop wording (byte-identical, offline / shapes without it).
    deny_domains:
        P2.4 (d131/d133) — the shape-level SOURCE deny-list, the cross-cutting source
        policy expressed at the SHAPE (TOOL-ENFORCED by the P2.1 web-tool baseline +
        the per-call ``exclude_domains`` arg). ``wikipedia.org`` (+ wikimedia /
        wiktionary) is the baseline; a shape names the domains the research must never
        fetch or cite. Empty tuple → only the tool's always-on baseline applies.
    expand_on_gaps:
        P2.5b (d134/d135) — the DECLARATIVE iterative-gap-expansion capability that lets
        the GENERIC engine reproduce ``run_research_tree``'s ITERATIVE breadth. When True
        the shape is NOT pre-unrolled into all its rounds: :func:`unroll_shape` emits ONLY
        the FIRST (seed) research layer and tags the :class:`~agent_runtime.factory.PlanDAG`
        ``growable``; the runtime's drive loop then GROWS the DAG round-by-round by invoking
        the SAME ``research_tree.run_decision_node`` over the persisted ``ResearchState`` —
        each note's gaps author the next layer's research nodes (growing-visibility edges),
        bounded by ``max_layers`` / ``fan_out`` + ``no_expansion`` + the ``completeness_stop``
        the model reasons over (``stop_research``). This relaxes EXACTLY ONE invariant ("node
        set fixed at unroll time") and REUSES the tree's already-generic tool-driven loop; it
        is the parity lever for retiring the bespoke tree. False → the legacy frozen unroll
        (every round emitted at unroll time, byte-identical to pre-P2.5b).
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
    """

    name: str
    description: str = ""
    max_iter: int = 1
    hard_cap: int = 0
    round_roles: tuple[str, ...] = ()
    final_roles: tuple[str, ...] = ()
    edges: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    execution: str = "concurrent"
    completeness_stop: str = ""
    deny_domains: tuple[str, ...] = ()
    expand_on_gaps: bool = False
    fan_out: int = 0
    max_layers: int = 0

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
        # Every declared round/final entry must be a known POSITION (d48 — these are
        # deep-research positions, not node roles). Fail fast on a typo.
        for position in tuple(self.round_roles) + tuple(self.final_roles):
            if position not in VALID_POSITIONS:
                raise ShapeError(
                    f"shape {self.name!r} declares unknown position {position!r}; "
                    f"valid positions: {sorted(VALID_POSITIONS)}"
                )

    @property
    def is_unrollable(self) -> bool:
        """True iff this shape declares round/final roles → it is UNROLLED (a3).

        A shape that names ``round_roles`` and/or ``final_roles`` (the bounded
        cyclic family, e.g. ``deep-research``) is expanded by :func:`unroll_shape`
        into a bounded acyclic role-tagged DAG the generic runtime executes. A
        shape WITHOUT them (``linear`` / ``modular-parallel``) is an execution
        DISCIPLINE only — its DAG is AUTHORED by the incremental planner instead.
        This declarative test is what makes shapes plug-n-play: the router keys
        off the shape file's fields, never a hard-coded shape name."""
        return bool(self.round_roles or self.final_roles)

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
            "round_roles": list(self.round_roles),
            "final_roles": list(self.final_roles),
            "edges": dict(self.edges),
            "source": self.source,
            "execution": self.execution,
            "completeness_stop": self.completeness_stop,
            "deny_domains": list(self.deny_domains),
            "expand_on_gaps": self.expand_on_gaps,
            "fan_out": self.fan_out,
            "max_layers": self.max_layers,
        }


def _parse_shape(path: Path) -> ShapeSpec:
    """Parse one TOML shape file into a :class:`ShapeSpec`."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ShapeError(f"cannot parse shape file {path}: {exc}") from exc
    return ShapeSpec(
        name=str(data.get("name") or path.stem),
        description=str(data.get("description", "")),
        max_iter=int(data.get("max_iter", 1)),
        hard_cap=int(data.get("hard_cap", 0)),
        round_roles=tuple(str(r) for r in data.get("round_roles", ())),
        final_roles=tuple(str(r) for r in data.get("final_roles", ())),
        edges=dict(data.get("edges", {})),
        source=str(path),
        execution=str(data.get("execution", "concurrent")),
        completeness_stop=str(data.get("completeness_stop", "")),
        deny_domains=tuple(str(d) for d in data.get("deny_domains", ())),
        expand_on_gaps=bool(data.get("expand_on_gaps", False)),
        fan_out=int(data.get("fan_out", 0)),
        max_layers=int(data.get("max_layers", 0)),
    )


# --------------------------------------------------------------------------- #
# GENERIC shape UNROLL: a cyclic template → a bounded acyclic role-tagged DAG
# --------------------------------------------------------------------------- #
def unroll_shape(
    shape: ShapeSpec,
    goal: str,
    *,
    spec: Optional[str] = None,
    max_iter_override: Optional[int] = None,
    grow: bool = False,
) -> PlanDAG:
    """Expand a cyclic ``shape`` into a bounded ACYCLIC, role-tagged DAG (a3).

    GROWABLE MODE (P2.5b, d134/d135): when ``grow=True`` AND the shape declares
    ``expand_on_gaps``, the unroll emits ONLY the FIRST (seed) RESEARCH layer and tags the
    returned :class:`PlanDAG` ``growable`` — the runtime's drive loop then GROWS the DAG
    round-by-round on note gaps via ``research_tree.run_decision_node`` (the iterative-breadth
    lever that lets the generic engine reproduce ``run_research_tree``). ``grow`` is the
    ENGINE's opt-in: a caller passes it ONLY when it wires a grower to drive the growth, so a
    shape gaining ``expand_on_gaps`` never silently turns a NON-growing caller's full unroll
    into a seed-only DAG (the inline ``_run_deep_research`` route keeps ``grow=False`` → the
    frozen unroll, byte-identical). ``grow=True`` on a shape WITHOUT ``expand_on_gaps`` is a
    no-op (the frozen unroll runs) — the capability must be declared on the shape.

    This is the GENERIC unroll that replaced the per-shape ``DeepResearchExecutor``
    (no per-shape python remains on the execution path). It is driven ENTIRELY by
    the shape's declarative fields — ``round_roles``, ``final_roles`` and the
    UI-overridable ``max_iter`` (bounded by ``hard_cap``) — so it works for ANY
    cyclic shape, not just ``deep-research``: adding such a shape is adding one
    text file.

    Rounds ``1..effective-1`` emit ``round_roles`` (e.g. {research, critic}); the
    single FINAL round emits ``final_roles`` (e.g. {research, synthesis, verify}).
    ``effective`` is :meth:`ShapeSpec.effective_max_iter` (honoring the UI
    override, clamped to ``hard_cap``) — so the runtime never runs more (or fewer)
    rounds than the shape/UI allow.

    EDGE POLICY — GROWING VISIBILITY (the §2c semantic, preserved): every node
    depends on EVERY previously-authored node. The rounds therefore run strictly
    in order, and — because the runtime threads each node's ``depends_on`` outputs
    in as its inputs — every node SEES all prior researched layers (their findings
    AND the critics' follow-ups), so each round genuinely builds on the ones
    before it. The same SINGLE ``spec`` is bound to every node; only the node ROLE
    differs (the role-prompt template + per-role output schema in
    :mod:`agent_runtime.roles`). Constructing the :class:`PlanDAG` validates
    acyclicity, so the cyclic shape provably never violates the acyclic invariant.

    A shape with neither ``round_roles`` nor ``final_roles`` is not unrollable
    (it is an execution discipline whose DAG is authored, not unrolled) and raises
    :class:`ShapeError`."""
    if not goal or not str(goal).strip():
        raise ShapeError(f"shape {shape.name!r} unroll needs a non-empty goal")
    if not shape.is_unrollable:
        raise ShapeError(
            f"shape {shape.name!r} declares no round_roles/final_roles — it is "
            "not an unrollable (cyclic) shape; its DAG is authored, not unrolled"
        )
    effective = shape.effective_max_iter(max_iter_override)
    specs = (spec,) if spec else ()
    nodes: list[PlanNode] = []
    prior_ids: list[str] = []  # EVERY node authored so far → full growing visibility
    # P2.5b (d134/d135) — ITERATIVE GAP-EXPANSION. When the shape declares ``expand_on_gaps``
    # we DO NOT pre-unroll every round (the frozen-DAG cause of the parity gap). Instead we
    # emit ONLY the SEED layer — the FIRST round's RESEARCH position(s) — and tag the DAG
    # ``growable``. The runtime's drive loop then grows the rest by re-frontiering on the
    # decision node's gap-driven ``expand_branch`` calls (the SAME tool surface the bespoke
    # tree uses), bounded by ``max_layers`` / ``fan_out`` + no_expansion + completeness_stop.
    # The seed drops the per-round CRITIC node: in growable mode the decision node IS the
    # critic, so the seed mirrors ``run_research_tree``'s gather-then-decide layer exactly.
    # Gated on BOTH the shape capability AND the engine's ``grow`` opt-in (so a non-growing
    # caller of an ``expand_on_gaps`` shape still gets the full frozen unroll — no regression).
    if grow and shape.expand_on_gaps:
        seed_positions = tuple(p for p in shape.round_roles if p == "research") or ("research",)
        for position in seed_positions:
            nid = f"r1_{position}"
            tool = "web_search" if position == "research" else None
            tool_args = {"query": str(goal)[:200]} if tool else {}
            nodes.append(
                PlanNode(
                    id=nid,
                    task=f"[{position} · round 1] {position_framing(position)}\n\n{goal}",
                    spec=spec,
                    specs=specs,
                    depends_on=tuple(prior_ids),
                    role=ROLE_SYNTHESIZER if position == "synthesis" else ROLE_WORKER,
                    tool=tool,
                    tool_args=tool_args,
                )
            )
            prior_ids.append(nid)
        return PlanDAG(
            nodes=nodes,
            rationale=(
                f"{shape.name} growable seed (1 layer; runtime grows on note gaps, "
                f"max_layers={shape.max_layers or 'cfg'}, fan_out={shape.fan_out or 'cfg'})"
            ),
            shape=shape.name,
            growable=True,
            fan_out=int(shape.fan_out),
            max_layers=int(shape.max_layers),
        )
    for r in range(1, effective + 1):
        is_final = r == effective
        positions = shape.final_roles if is_final else shape.round_roles
        for position in positions:
            nid = f"r{r}_{position}"
            # d48: map the POSITION onto a NODE ROLE (worker|synthesizer) and inject
            # the position's behavior framing into the TASK — so a research/critic/
            # verify node behaves as such via PROMPTING, not a per-role code switch.
            node_role = ROLE_SYNTHESIZER if position == "synthesis" else ROLE_WORKER
            # A research-position node reads real sources via the GENERIC
            # search-then-read TOOL path (the retired role-research gate's job, now
            # keyed on the tool, not a role): bind web_search + a seed query.
            tool = "web_search" if position == "research" else None
            tool_args = {"query": str(goal)[:200]} if tool else {}
            nodes.append(
                PlanNode(
                    id=nid,
                    task=f"[{position} · round {r}] {position_framing(position)}\n\n{goal}",
                    spec=spec,
                    specs=specs,
                    # Depend on every prior node: the rounds run in order and each
                    # node's inputs carry all earlier layers (growing visibility).
                    depends_on=tuple(prior_ids),
                    role=node_role,
                    tool=tool,
                    tool_args=tool_args,
                )
            )
            prior_ids.append(nid)
    return PlanDAG(
        nodes=nodes,
        rationale=f"{shape.name} unroll ({effective} rounds, position-framed)",
        shape=shape.name,
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
    "ShapeError",
    "SHAPES_DIR",
    "DEEP_RESEARCH",
    "VALID_EXECUTION",
    "VALID_POSITIONS",
    "unroll_shape",
    "load_shapes",
    "load_shape",
    "shape_names",
]
