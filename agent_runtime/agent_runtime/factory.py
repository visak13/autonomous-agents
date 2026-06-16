"""The abstract plan factory + the DAG it produces.

This module is the *shape* of an autonomous plan and the **abstract factory**
that builds the planner's context. It carries NO model call and NO execution —
it is pure data + validation so it stays lean for phi's small context (d10).

Two ideas live here:

1. :class:`PlanNode` / :class:`PlanDAG` — the model-DERIVED plan (d6). A node is
   a single *logical step* (``id`` + free-text ``task``), optionally bound to a
   specialization by NAME (looked up later) and to a primary tool by NAME. Edges
   are ``depends_on`` references. The DAG validates (unique ids, resolvable refs,
   acyclic) and exposes the scheduling primitives the in-process runtime needs
   (``ready`` / ``topo_order``). The DAG is whatever phi emits — there is no
   hard-coded task list anywhere (d6).

2. :class:`AbstractPlanFactory` — the planner's *only* world view (d10). It holds
   a short factory DESCRIPTION (what a plan is + the node schema phi must emit)
   and the specialization **LOOKUP index** (body-free ``SpecIndexEntry`` rows)
   plus a lean tool catalog (names + one-liners). From those it builds the
   planner-context payload — and it is constructed *so that a body can never
   enter it*: the only specialization data it accepts is the index. It also
   parses phi's emitted JSON back into a validated :class:`PlanDAG`.

Context-scoping (d10) is enforced HERE, by the surface: the factory is given the
index (never the registry), so the planner code path literally cannot reach a
compiled spec body.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence


class PlanError(ValueError):
    """A plan/DAG is structurally invalid (bad refs, cycle, duplicate id)."""


# Node ROLES (s3 / blueprint §2c). In eda-base3 the role is the spawn PROTOCOL
# (the pool sets EDP_ROLE → worker.md vs reviewer.md skill prompt over the SAME
# compiled spec doc). With no Claude-Code pool here, ``role`` is an EXPLICIT node
# field: the SAME specialization is reused, and the role selects a role-prompt
# template + a per-role OUTPUT SCHEMA (see ``agent_runtime.roles``). This is what
# lets the deep-research shape run ONE spec differentiated only by node role.
# Defined HERE (pure data, no model) so ``factory`` stays lean and ``roles`` can
# import it without a cycle.
VALID_ROLES: frozenset[str] = frozenset(
    {"research", "critic", "worker", "reviewer", "synthesis", "verify"}
)


# --------------------------------------------------------------------------- #
# The model-derived plan: nodes + DAG
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlanNode:
    """One logical step in a model-derived plan (d6).

    Attributes
    ----------
    id:
        Unique node id within the DAG (e.g. ``"n1"``). Edge targets reference it.
    task:
        The free-text logical step phi authored (e.g. "research <topic>"). This
        is the sub-agent's task-at-hand — the ONLY task it is told (d10).
    spec:
        Optional SINGLE specialization NAME to load for this node (the legacy
        single-spec form, looked up in the registry by a sub-agent). ``None`` =
        no specialization on this scalar field. Only a NAME is carried — never a
        body (d10). Kept for back-compat; for 1+ specs prefer ``specs``.
    specs:
        Optional list of specialization NAMES to COMPOSE (layer) onto this node
        (d2/d11 spec-composition: ONE task -> N specs). When non-empty it is the
        AUTHORITATIVE spec list and takes precedence over the scalar ``spec``;
        the runtime resolves every name's ruleset body and composes them into the
        produce SYSTEM in this list's order. Empty = fall back to the scalar
        ``spec`` (single-spec back-compat). Only NAMES are carried — never a body
        (d10): the planner stays body-free, bodies resolve at runtime.
    depends_on:
        Ids of nodes that must finish before this one launches.
    tool:
        Optional primary tool NAME the step is expected to use (a hint phi can
        emit). ``None`` = no tool. Only the name is carried (context-scoping).
    tool_args:
        Optional kwargs for the primary tool call (what phi would emit alongside
        a tool hint). Kept JSON-shaped.
    role:
        Optional NODE ROLE (blueprint §2c) — one of :data:`VALID_ROLES`
        (``research|critic|worker|reviewer|synthesis|verify``) or ``None``. The
        role does NOT change which specialization loads (that is ``specs``); it
        selects a role-prompt TEMPLATE + a per-role OUTPUT SCHEMA so the SAME spec
        behaves differently per node (the deep-research engine). ``None`` = a
        plain producer step (legacy behaviour, byte-compatible with every
        existing acyclic plan). Only a known role string is accepted.
    needs_spec:
        Optional FREE-TEXT descriptor of a specialist this step REQUIRES when NO
        registered specialization in the planner's lookup fits (s4 RC8 / blueprint
        §2b). Unlike ``spec``/``specs`` (enum-constrained to registered names),
        this is the planner's escape hatch to DECLARE a missing capability instead
        of silently leaving the node spec-less (which would degrade to raw LLM
        auto-completion). A node carrying ``needs_spec`` with NO resolvable
        ``effective_specs`` is the MISSING-SPECIALIST signal the chat surfaces as a
        user notify + CHOICE (SSE-fallback / define-and-resume) — see
        :mod:`agent_runtime.missing_spec`. ``None``/empty = the node needs no
        unavailable specialist (the common case).
    """

    id: str
    task: str
    spec: Optional[str] = None
    specs: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    tool: Optional[str] = None
    tool_args: Mapping[str, Any] = field(default_factory=dict)
    role: Optional[str] = None
    needs_spec: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise PlanError(f"PlanNode.id must be a non-empty str, got {self.id!r}")
        if not isinstance(self.task, str) or not self.task.strip():
            raise PlanError(f"PlanNode.task must be a non-empty str, got {self.task!r}")
        # ROLE (optional): None stays None (a plain producer step — exactly the
        # pre-role behaviour); a non-empty value MUST be a known role so a typo
        # can never silently select no template. Normalised to lower-case.
        if self.role is not None:
            role = str(self.role).strip().lower()
            if not role:
                object.__setattr__(self, "role", None)
            elif role not in VALID_ROLES:
                raise PlanError(
                    f"PlanNode.role {self.role!r} is not one of {sorted(VALID_ROLES)}"
                )
            else:
                object.__setattr__(self, "role", role)
        # Normalise ``specs`` to a tuple of clean, non-empty NAMES (a stray list
        # or None becomes a tuple; blanks are dropped). frozen dataclass → set via
        # object.__setattr__.
        raw_specs = self.specs or ()
        if isinstance(raw_specs, str):
            raw_specs = (raw_specs,)
        cleaned = tuple(str(s).strip() for s in raw_specs if str(s).strip())
        object.__setattr__(self, "specs", cleaned)
        # Normalise ``needs_spec``: a blank/whitespace descriptor is no descriptor
        # (None), so a stray empty string from the planner never trips the
        # missing-specialist detector.
        if self.needs_spec is not None:
            needs = str(self.needs_spec).strip()
            object.__setattr__(self, "needs_spec", needs or None)

    @property
    def effective_specs(self) -> tuple[str, ...]:
        """The ORDERED spec names that actually apply to this node (d2/d11).

        The authoritative list when ``specs`` is non-empty (composition order =
        this tuple's order); otherwise the single scalar ``spec`` as a one-tuple;
        otherwise empty (a bare step). This is the single source of truth both the
        runtime (body resolution + composition) and introspection read, so the
        single-spec and multi-spec forms never diverge."""
        if self.specs:
            return self.specs
        if self.spec:
            return (self.spec,)
        return ()

    @property
    def primary_spec(self) -> Optional[str]:
        """The first effective spec (or ``None``) — for display/result labelling.

        Back-compat: for a single-spec node this is exactly ``spec``; for a
        specs-only node it is ``specs[0]`` (so events/results still carry a
        meaningful spec name even when the scalar ``spec`` is unset)."""
        es = self.effective_specs
        return es[0] if es else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "spec": self.spec,
            "specs": list(self.specs),
            "depends_on": list(self.depends_on),
            "tool": self.tool,
            "tool_args": dict(self.tool_args),
            "role": self.role,
            "needs_spec": self.needs_spec,
        }


@dataclass
class PlanDAG:
    """A directed acyclic graph of :class:`PlanNode` — the plan phi emitted.

    Construction validates the graph eagerly (call :meth:`validate`, which the
    constructor runs) so an invalid DAG never reaches the runtime. The DAG is
    *model-derived* — its shape comes entirely from phi (d6); nothing here
    hard-codes a particular plan."""

    nodes: list[PlanNode]
    rationale: str = ""
    shape: str = ""

    def __post_init__(self) -> None:
        self.validate()

    # -- lookups ----------------------------------------------------------- #

    @property
    def by_id(self) -> dict[str, PlanNode]:
        return {n.id: n for n in self.nodes}

    def __len__(self) -> int:
        return len(self.nodes)

    # -- validation -------------------------------------------------------- #

    def validate(self) -> "PlanDAG":
        """Assert unique ids, resolvable ``depends_on`` refs, and acyclicity."""
        ids: set[str] = set()
        for n in self.nodes:
            if n.id in ids:
                raise PlanError(f"duplicate node id {n.id!r}")
            ids.add(n.id)
        for n in self.nodes:
            for d in n.depends_on:
                if d not in ids:
                    raise PlanError(
                        f"node {n.id!r} depends on unknown node {d!r}"
                    )
                if d == n.id:
                    raise PlanError(f"node {n.id!r} depends on itself")
        # Acyclicity via Kahn's algorithm (also gives a topo order for free).
        self._kahn_order()  # raises PlanError on a cycle
        return self

    def _kahn_order(self) -> list[str]:
        indeg: dict[str, int] = {n.id: 0 for n in self.nodes}
        adj: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for n in self.nodes:
            for d in n.depends_on:
                indeg[n.id] += 1
                adj[d].append(n.id)
        # Deterministic queue order (sorted) so the topo order is stable.
        ready = sorted([nid for nid, d in indeg.items() if d == 0])
        order: list[str] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            new_ready: list[str] = []
            for m in adj[nid]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    new_ready.append(m)
            ready = sorted(ready + new_ready)
        if len(order) != len(self.nodes):
            stuck = sorted(set(indeg) - set(order))
            raise PlanError(f"DAG has a cycle; nodes never become ready: {stuck}")
        return order

    def topo_order(self) -> list[PlanNode]:
        """Nodes in a deterministic dependency-respecting order."""
        by_id = self.by_id
        return [by_id[nid] for nid in self._kahn_order()]

    def ready(self, done: Iterable[str]) -> list[PlanNode]:
        """Nodes whose every dependency is in ``done`` and which are not in it.

        This is the scheduler primitive the in-process runtime polls each round
        to decide which nodes may launch next."""
        done_set = set(done)
        return [
            n
            for n in self.nodes
            if n.id not in done_set and all(d in done_set for d in n.depends_on)
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.as_dict() for n in self.nodes],
            "rationale": self.rationale,
            "shape": self.shape,
        }


# --------------------------------------------------------------------------- #
# The abstract plan factory: the planner's whole world view (d10)
# --------------------------------------------------------------------------- #

# The factory description handed to phi. It is intentionally compact (d10): it
# tells phi WHAT a plan is and the exact JSON node schema to emit — NOT how to
# solve any particular goal (no hard-coded task prompt, d6). phi reasons over
# the goal + the lookup to author the nodes itself.
FACTORY_DESCRIPTION = (
    "You are an autonomous planner. Decompose the GOAL into a DAG of logical "
    "steps. Each node is one step with a unique id and a free-text 'task'. Use "
    "'depends_on' to order steps (a node runs only after every id it lists has "
    "finished); independent steps share no edge and may run concurrently. If a "
    "step's work matches a registered specialization, set 'spec' to that "
    "specialization's name (from the lookup); if SEVERAL registered "
    "specializations apply to one step, list them ALL in 'specs' (they are "
    "composed/layered onto that step in the order you list them) instead of "
    "'spec'. If a step REQUIRES a specialist whose work is NOT covered by any "
    "registered specialization in the lookup, leave 'spec'/'specs' empty and "
    "DESCRIBE the needed specialist in 'needs_spec' (free text) — do NOT invent a "
    "name and do NOT silently leave it unspecialized. If a step "
    "needs a tool, set 'tool' to the tool name (from the tool list) and put its "
    "arguments in 'tool_args'. Emit ONLY the plan that fits THIS goal — invent "
    "the steps yourself; there is no template to follow."
)

# The node schema advertised to phi (kept as data so the planner prompt and the
# parser agree on exactly one shape).
NODE_SCHEMA: dict[str, str] = {
    "id": "unique short id, e.g. 'n1'",
    "task": "the logical step, free text",
    "spec": "ONE specialization name from the lookup, or null (single-spec form)",
    "specs": "list of specialization names from the lookup to COMPOSE on this "
             "step, in apply order (1+; use instead of 'spec' when more than one "
             "applies), or omitted/[] for none",
    "depends_on": "list of node ids that must finish first (may be empty)",
    "tool": "tool name from the tool list, or null",
    "tool_args": "object of arguments for the tool, or omitted",
    "role": "node role, one of research|critic|worker|reviewer|synthesis|verify, "
            "or null (null = a plain producer step)",
    "needs_spec": "free-text description of a REQUIRED specialist when NO listed "
                  "specialization fits, or null/omitted; set this (and leave "
                  "spec/specs empty) instead of guessing a name or running "
                  "unspecialized",
}


class AbstractPlanFactory:
    """Builds the planner's context and parses phi's DAG — body-free by design.

    The factory is constructed with the specialization **index** (body-free
    rows) and a lean tool catalog, NOT the registry. That is the d10 enforcement
    point: the planner, which only ever holds a factory, cannot reach a compiled
    spec body — the surface makes it impossible, not a matter of discipline.
    """

    def __init__(
        self,
        spec_index: Sequence[Mapping[str, Any]] | Sequence[Any],
        *,
        tool_catalog: Optional[Sequence[Mapping[str, str]]] = None,
        description: str = FACTORY_DESCRIPTION,
    ) -> None:
        # Normalise the index to plain body-free dicts. We accept SpecIndexEntry
        # objects (which carry only name/description/source) or already-dicts.
        self._spec_index: list[dict[str, str]] = [
            self._index_row(entry) for entry in spec_index
        ]
        self._tool_catalog: list[dict[str, str]] = [
            {"name": str(t["name"]), "description": str(t.get("description", ""))}
            for t in (tool_catalog or [])
        ]
        self.description = description

    @staticmethod
    def _index_row(entry: Any) -> dict[str, str]:
        """Coerce one index entry to a body-free ``{name, description, source}``.

        Hard guard: a dict carrying a ``body`` is rejected — a body must never
        reach the planner-facing factory (d10)."""
        if hasattr(entry, "as_dict"):
            entry = entry.as_dict()
        if isinstance(entry, Mapping):
            if "body" in entry:
                raise PlanError(
                    "spec index entry carries a 'body' — the factory is "
                    "body-free by design (d10); pass SpecRegistry.index() rows"
                )
            return {
                "name": str(entry.get("name", "")),
                "description": str(entry.get("description", "")),
                "source": str(entry.get("source", "")),
            }
        # A bare SpecIndexEntry without as_dict (defensive).
        return {
            "name": str(getattr(entry, "name", "")),
            "description": str(getattr(entry, "description", "")),
            "source": str(getattr(entry, "source", "")),
        }

    # -- the planner's context payload (factory + lookup ONLY, d10) -------- #

    def planner_context(self, goal: str) -> dict[str, Any]:
        """The EXACT payload the planner reasons over — factory + lookup + goal.

        Schema (and nothing more — no spec bodies, no phased prompts, d10)::

            {
              "factory": {"kind", "description", "node_schema"},
              "specializations": [{"name","description","source"}, ...],  # lookup
              "tools": [{"name","description"}, ...],                     # names only
              "goal": "<the user goal>"
            }

        :meth:`assert_body_free` can verify a built payload carries no body.
        """
        return {
            "factory": {
                "kind": "abstract-plan-factory",
                "description": self.description,
                "node_schema": dict(NODE_SCHEMA),
            },
            "specializations": [dict(r) for r in self._spec_index],
            "tools": [dict(t) for t in self._tool_catalog],
            "goal": goal,
        }

    @staticmethod
    def assert_body_free(payload: Mapping[str, Any]) -> None:
        """Raise if a planner-context payload contains a 'body' anywhere (d10).

        Used by the planner and the smoke as a hard, machine-checkable proof
        that context-scoping held — the planner never saw a compiled spec body."""

        def walk(obj: Any, path: str) -> None:
            if isinstance(obj, Mapping):
                for k, v in obj.items():
                    if str(k).lower() == "body":
                        raise PlanError(
                            f"planner context leaked a 'body' at {path}.{k} (d10)"
                        )
                    walk(v, f"{path}.{k}")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    walk(v, f"{path}[{i}]")

        walk(payload, "$")

    def planner_prompt(self, goal: str) -> tuple[str, str]:
        """Return ``(system, user)`` text for the planner's phi call.

        The system message is the factory description + node schema + the lookup
        and tool catalog (all body-free); the user message is just the goal. The
        payload is asserted body-free before it is serialised (d10)."""
        ctx = self.planner_context(goal)
        self.assert_body_free(ctx)
        system = (
            ctx["factory"]["description"]
            + "\n\nEmit STRICT JSON of the form: "
            + '{"rationale": "<one line>", "nodes": [ '
            + json.dumps(ctx["factory"]["node_schema"])
            + " ]}.\n\nNODE SCHEMA keys are described above; 'nodes' is a list of "
            "objects with those keys.\n\nREGISTERED SPECIALIZATIONS (lookup — "
            "names + descriptions only):\n"
            + json.dumps(ctx["specializations"], indent=2)
            + "\n\nAVAILABLE TOOLS (names + descriptions only):\n"
            + json.dumps(ctx["tools"], indent=2)
        )
        user = f"GOAL: {goal}\n\nReturn ONLY the JSON plan."
        return system, user

    # -- focused sub-graph re-plan (self-heal: re-derive only the failure) - #

    def replan_context(
        self,
        failed_task: str,
        error: str,
        *,
        spec: Optional[str] = None,
        completed: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        """Body-free context for re-deriving a corrective sub-graph for ONE step.

        Used by the runtime's sub-graph self-heal: a node has FAILED after its
        node-level retries; instead of replanning the whole goal, the planner is
        asked to re-derive a minimal corrective DAG for *just that step's intent*
        (an alternative approach / a small decomposition). Still factory + lookup
        ONLY — no spec bodies, no other nodes' work product (d10). ``completed``
        is just the list of already-DONE node ids (names, for the model's
        awareness that it must not redo them — never their outputs/bodies)."""
        return {
            "factory": {
                "kind": "abstract-plan-factory",
                "description": self.description,
                "node_schema": dict(NODE_SCHEMA),
            },
            "specializations": [dict(r) for r in self._spec_index],
            "tools": [dict(t) for t in self._tool_catalog],
            "failed_step": {"task": failed_task, "spec": spec, "error": error},
            "already_completed": list(completed or []),
        }

    def replan_prompt(
        self,
        failed_task: str,
        error: str,
        *,
        spec: Optional[str] = None,
        completed: Optional[Sequence[str]] = None,
    ) -> tuple[str, str]:
        """``(system, user)`` for the focused re-plan call (asserted body-free)."""
        ctx = self.replan_context(
            failed_task, error, spec=spec, completed=completed
        )
        self.assert_body_free(ctx)
        system = (
            self.description
            + "\n\nA SINGLE step of an existing plan FAILED and must be re-derived. "
            "Emit a MINIMAL corrective DAG (one or a few nodes) that accomplishes "
            "ONLY that step's intent with a different approach — do NOT replan the "
            "whole goal and do NOT redo already-completed steps. Emit STRICT JSON: "
            '{"rationale": "<one line>", "nodes": [ '
            + json.dumps(NODE_SCHEMA)
            + " ]}.\n\nREGISTERED SPECIALIZATIONS (lookup — names + descriptions "
            "only):\n"
            + json.dumps(ctx["specializations"], indent=2)
            + "\n\nAVAILABLE TOOLS (names + descriptions only):\n"
            + json.dumps(ctx["tools"], indent=2)
        )
        user = (
            f"FAILED STEP: {failed_task}\nERROR: {error}\n"
            f"ALREADY COMPLETED (do not redo): {json.dumps(list(completed or []))}\n\n"
            "Return ONLY the JSON corrective sub-plan."
        )
        return system, user

    # -- parse phi's emitted JSON back into a validated DAG --------------- #

    def parse_dag(self, structured: Any) -> PlanDAG:
        """Turn phi's parsed JSON (a dict/list) into a validated :class:`PlanDAG`.

        Accepts either ``{"nodes": [...], "rationale": ...}`` or a bare list of
        node objects. Raises :class:`PlanError` on a malformed/empty plan so the
        self-heal layer can catch it and repair/re-plan."""
        if structured is None:
            raise PlanError("planner produced no structured plan (None)")
        rationale = ""
        shape = ""
        if isinstance(structured, Mapping):
            raw_nodes = structured.get("nodes")
            rationale = str(structured.get("rationale", ""))
            shape = str(structured.get("shape", "") or "")
        elif isinstance(structured, list):
            raw_nodes = structured
        else:
            raise PlanError(f"plan must be an object or list, got {type(structured).__name__}")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise PlanError(f"plan has no 'nodes' list: {structured!r}")

        nodes: list[PlanNode] = []
        for i, raw in enumerate(raw_nodes):
            if not isinstance(raw, Mapping):
                raise PlanError(f"node[{i}] is not an object: {raw!r}")
            dep = raw.get("depends_on") or []
            if isinstance(dep, str):
                dep = [dep]
            # 1+ specs (d2/d11 composition): accept a 'specs' list (preferred) or a
            # bare string; PlanNode normalises/cleans it. The scalar 'spec' is kept
            # for single-spec back-compat. Only NAMES are read — never a body (d10).
            specs_raw = raw.get("specs") or ()
            if isinstance(specs_raw, str):
                specs_raw = [specs_raw]
            specs = tuple(str(s) for s in specs_raw if s)
            nodes.append(
                PlanNode(
                    id=str(raw.get("id") or f"n{i+1}"),
                    task=str(raw.get("task") or raw.get("description") or ""),
                    spec=(str(raw["spec"]) if raw.get("spec") else None),
                    specs=specs,
                    depends_on=tuple(str(d) for d in dep),
                    tool=(str(raw["tool"]) if raw.get("tool") else None),
                    tool_args=dict(raw.get("tool_args") or {}),
                    # ROLE (optional): a known role or null; PlanNode validates it.
                    role=(str(raw["role"]) if raw.get("role") else None),
                    # NEEDS_SPEC (optional, s4 RC8): the planner's free-text
                    # missing-specialist signal; PlanNode normalises blanks to None.
                    needs_spec=(
                        str(raw["needs_spec"]) if raw.get("needs_spec") else None
                    ),
                )
            )
        return PlanDAG(nodes=nodes, rationale=rationale, shape=shape)


__all__ = [
    "PlanError",
    "PlanNode",
    "PlanDAG",
    "AbstractPlanFactory",
    "FACTORY_DESCRIPTION",
    "NODE_SCHEMA",
    "VALID_ROLES",
]
