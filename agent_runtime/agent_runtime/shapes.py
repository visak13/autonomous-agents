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

from .factory import PlanDAG, PlanError, PlanNode, VALID_ROLES

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
    """A shape text file is missing/malformed or declares an unknown role."""


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
        # Every declared role must be a known node role (fail fast on a typo).
        for role in tuple(self.round_roles) + tuple(self.final_roles):
            if role not in VALID_ROLES:
                raise ShapeError(
                    f"shape {self.name!r} declares unknown role {role!r}; "
                    f"valid roles: {sorted(VALID_ROLES)}"
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
) -> PlanDAG:
    """Expand a cyclic ``shape`` into a bounded ACYCLIC, role-tagged DAG (a3).

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
    for r in range(1, effective + 1):
        is_final = r == effective
        roles = shape.final_roles if is_final else shape.round_roles
        for role in roles:
            nid = f"r{r}_{role}"
            nodes.append(
                PlanNode(
                    id=nid,
                    task=f"[{role} · round {r}] {goal}",
                    spec=spec,
                    specs=specs,
                    # Depend on every prior node: the rounds run in order and each
                    # node's inputs carry all earlier layers (growing visibility).
                    depends_on=tuple(prior_ids),
                    role=role,
                )
            )
            prior_ids.append(nid)
    return PlanDAG(
        nodes=nodes,
        rationale=f"{shape.name} unroll ({effective} rounds, role-differentiated)",
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
    "unroll_shape",
    "load_shapes",
    "load_shape",
    "shape_names",
]
