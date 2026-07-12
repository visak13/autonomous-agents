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

from .identity import with_identity


class PlanError(ValueError):
    """A plan/DAG is structurally invalid (bad refs, cycle, duplicate id)."""


# Node ROLES = node types (d213/d215, re-expanded from the d48 collapse). The ROLE
# lives in the node type; there are FIVE, sitting in three places:
#   * PLANNER     — the planning STAGE (shape selector + incremental planner). It
#                   drives the iterative loop; it is NOT an in-plan node, so it is
#                   intentionally ABSENT from VALID_ROLES.
#   * in-plan     — RESEARCHER, WORKER, REVIEWER: the ONLY roles the planner places
#                   inside a plan via add_step (d215). REVIEWER is the default LAST
#                   STEP of every plan and emits the plan's final status.
#   * SYNTHESIZER — the TERMINAL stage, materialised as a single role=synthesizer
#                   node that runs ONCE after the planner loop exits (d215). The
#                   planner never add_steps it; the framework builds it. It stays a
#                   legal PlanNode.role so that terminal node validates.
# Node BEHAVIOR still comes from the node's SPEC(s) + task framing + reasoning, NOT a
# per-role code switch (the 6-role enum {research,critic,worker,reviewer,synthesis,
# verify} and its per-role output schemas / verdict path stay RETIRED — all five
# roles emit RAW content). The deep-research per-round behaviors (research/critic/
# verify) remain POSITIONS the shape declares (``shapes.VALID_POSITIONS``), composed
# onto the in-plan nodes via the matching framing injected into the task (prompting).
# Roles are NOT LLM-extensible (Q-A: bounded) — this set is fixed. Defined HERE (pure
# data, no model) so ``factory`` stays lean and ``roles`` imports it without a cycle.
VALID_ROLES: frozenset[str] = frozenset(
    {"researcher", "worker", "reviewer", "synthesizer"}
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
        Optional NODE ROLE = node type (d213/d215) — one of :data:`VALID_ROLES`
        (``researcher`` | ``worker`` | ``reviewer`` | ``synthesizer``) or ``None``.
        The role does NOT change which specialization loads (that is ``specs``); it
        selects the node TYPE: ``researcher`` routes to the agentic research/gather
        loop, ``reviewer`` is the default last-step that inspects+fixes the
        deliverable and emits the plan's final status, ``synthesizer`` routes to the
        terminal raw-content file/answer loop, ``worker``/``None`` is a plain
        producer step whose behavior comes from its SPEC(s) + task + reasoning. The
        planner only ever add_steps ``researcher``/``worker``/``reviewer`` (d215);
        ``synthesizer`` is the framework-built terminal node. ``None`` = a plain
        producer step (byte-compatible with every existing acyclic plan). Only a
        known role string is accepted.
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
    source_ids: tuple[int, ...] = ()
    # MEMORY-BY-HANDLE (d221): the stable handle of the research memory this node is
    # BOUND to. A research node carries the handle of the memory it WRITES into; a
    # downstream write/review node carries the handle of the memory it READS from (via
    # the read-via-tools surface — load_source / research_read — NEVER a verbatim dump,
    # d192/d202). The runtime renders a "Binded research memory: <handle>" line into the
    # node's context (:meth:`SubAgent._compose_task`) so the model knows which memory to
    # read. ``None``/empty = the node is not bound to a research memory (the common case
    # for a plain worker), and the context line is omitted (byte-identical to pre-d221).
    research_memory_handle: Optional[str] = None
    # MEMORY-INDEX (d285 SB-3): the planner's REASONED CHOICE, carried on the step BRIEF,
    # of which research memory this step works in — an existing INDEX (→ CONTINUE the
    # memory a prior step built, reasoning over the upstream summary+index) or the textual
    # ``<<NEW>>`` sentinel (→ START a fresh, distinct research line). This is the AUTHORED
    # choice (data the planner emits), distinct from ``research_memory_handle`` (the
    # RESOLVED handle a node is bound to). It resolves THROUGH SB-1's store via
    # :func:`~agent_runtime.research_tree.resolve_brief_memory` (an index continues that
    # memory; <<NEW>> mints a fresh one). Empty string = unspecified (treated as <<NEW>>);
    # the engine stamps NO index here (anti-fabrication, d10-clean — the planner chooses).
    memory_index: str = ""
    # WRITE-PHASE DELIVERY TARGET (RP-6c B1 / O1). The single-file path this node delivers, set
    # by the one-drive phase-transition step (:class:`AgentRuntime._phase_transition`) on the
    # write-phase node(s) it appends — decided from the shape's declared ``spec_role_for('write')
    # == 'writer'`` (RP-6b), NOT a spec-name conditional. This RE-SCOPES the SB-6/d301 write-route
    # discriminator from the runtime-global ``deliverable_path`` to the NODE: in a SHARED runtime
    # where research nodes and the write node coexist, ONLY a node carrying this DATA routes to the
    # served writer (:meth:`SubAgent._run_file_delivery`); research nodes (no ``deliverable_path``)
    # keep the research route. ``None`` for every research/gather/follow-up node and every node on
    # the legacy dedicated write_runtime path (which still keys on the runtime-global signal), so
    # both are byte-identical. It is structural DELIVERY-CONTEXT DATA (this node delivers one file),
    # not a spec/role/tool-name branch — it passes the d293 anti-fabrication grep gate.
    deliverable_path: Optional[str] = None

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
        # SOURCE-SCOPING (s9/c13, d56): the 1-based indices into the run's global
        # fetched-SOURCE index that THIS (synthesis/write) node is responsible for —
        # the planner's REASONED source→section assignment, so each section's prompt
        # carries ONLY its own sources (kept inside the model's ~512-tok sliding
        # window). Normalised to a deduped, order-preserving tuple of positive ints; a
        # stray non-int / non-positive value is dropped. Empty = no scoping (the node
        # sees the full upstream source index, byte-identical to the pre-c13 path).
        raw_sids = self.source_ids or ()
        if isinstance(raw_sids, (int, str)):
            raw_sids = (raw_sids,)
        clean_sids: list[int] = []
        for s in raw_sids:
            try:
                v = int(s)
            except (TypeError, ValueError):
                continue
            if v > 0 and v not in clean_sids:
                clean_sids.append(v)
        object.__setattr__(self, "source_ids", tuple(clean_sids))
        # MEMORY-BY-HANDLE (d221): a blank/whitespace handle is no handle (None), so a
        # stray empty string never renders an empty "Binded research memory:" line.
        if self.research_memory_handle is not None:
            h = str(self.research_memory_handle).strip()
            object.__setattr__(self, "research_memory_handle", h or None)
        # MEMORY-INDEX (d285 SB-3): keep the planner-authored choice as a clean string.
        # A stray non-str / blank becomes "" (unspecified → the resolver treats it as
        # <<NEW>>); no sentinel logic here — canonicalization lives at the authoring
        # boundary (plan_tools) and the resolver (research_tree), so the node just carries.
        object.__setattr__(self, "memory_index", str(self.memory_index or "").strip())
        # WRITE-PHASE DELIVERY TARGET (RP-6c B1 / O1): a blank/whitespace path is no path
        # (None), so a stray empty string never mis-routes a research node to the writer.
        if self.deliverable_path is not None:
            dp = str(self.deliverable_path).strip()
            object.__setattr__(self, "deliverable_path", dp or None)

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
            "source_ids": list(self.source_ids),
            "research_memory_handle": self.research_memory_handle,
            "memory_index": self.memory_index,  # d285 SB-3: planner-authored choice
            "deliverable_path": self.deliverable_path,  # RP-6c B1/O1: write-phase delivery target
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
    # The VERBATIM overall goal this plan serves (d38/d39) — the user's actual
    # request, carried ON the plan so it travels with the DAG to the runtime and
    # survives the missing-specialist resume (which re-derives the DAG, not the
    # goal). The runtime reads it and feeds it to EVERY worker node's user turn,
    # because a Gemma node cannot DISCOVER the goal the way an eda-base3/Claude-Code
    # worker can (no file/grep access) — without it a node sees only the planner's
    # PARAPHRASED task. Empty => omitted everywhere (byte-identical to pre-d39).
    goal: str = ""
    # GROWABLE PLAN (P2.5b, d134/d135) — the ONE relaxed invariant: when a shape declares
    # ``expand_on_gaps`` the unroll emits only the SEED research layer and tags the DAG
    # ``growable``; the runtime's drive loop then APPENDS new research nodes round-by-round
    # via ``research_tree.run_decision_node`` (gaps → next layer), bounded by ``fan_out`` /
    # ``max_layers`` + no_expansion + completeness_stop. False/0 => a frozen DAG, byte-
    # identical to every pre-P2.5b plan (the node set is fixed at unroll/author time).
    growable: bool = False
    fan_out: int = 0      # per-layer expansion cap for the grower's Tree (0 => runtime default)
    max_layers: int = 0   # growth bound: max research layers incl. the seed (0 => runtime default)
    max_sources: int = 0  # d163 ceiling: max fetched sources fed to the write phase (0 => uncapped)

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
            "goal": self.goal,
            "growable": self.growable,
            "fan_out": self.fan_out,
            "max_layers": self.max_layers,
        }


# --------------------------------------------------------------------------- #
# The abstract plan factory: the planner's whole world view (d10)
# --------------------------------------------------------------------------- #

# The factory description handed to phi. It is intentionally compact (d10): it
# tells phi WHAT a plan is and the exact JSON node schema to emit — NOT how to
# solve any particular goal (no hard-coded task prompt, d6). phi reasons over
# the goal + the lookup to author the nodes itself.
#
# ANTI-FAB THINNING (RP-AUDIT F4 / d341): this description carries ONLY the node
# schema + the generic spec/tool SELECTION PRINCIPLE. It does NOT bake any
# flow/format/schedule AUTHORING RECIPE. Two such recipes used to live here and
# were removed because they duplicated definition-layer doctrine and rode on
# EVERY goal regardless of the selected shape (the d341 hazard):
#   • the SCHEDULE-ONLY recipe ("a recurring goal ⇒ one cron_add node …") now
#     lives ONLY in schedule-leg.toml's `decompose_methodology` (RP-4c), which the
#     incremental authorer substitutes with precedence; and
#   • the OUTPUT-FORMAT-BINDING recipe ("an HTML request gets the HTML writer …")
#     now lives in the WRITER SPEC descriptions (html-writer/markdown-writer say
#     which format they produce), which the planner reasons over — plus the
#     format-hygiene rule F2 keeps in the incremental `_system` guidance.
# What stays here is generic and flow-neutral: how to pick a spec by MATCHING the
# WORK a node produces vs each spec's advertised description (incl. the
# format-bleed hygiene that a gather node never carries a document-format spec).
FACTORY_DESCRIPTION = (
    "You are an autonomous planner. Decompose the GOAL into a DAG of logical "
    "steps — invent the steps yourself; there is no template.\n"
    "- Each node has a unique id, a free-text 'task', and 'depends_on' (ids that "
    "must finish first). Independent steps share no edge and run concurrently.\n"
    "- Specialization (SELECTION GUIDELINES, see docs/SELECTION_GUIDELINES.md): "
    "MATCH on the WORK a step PRODUCES vs each spec's description, not a shared "
    "keyword. Bind an output-style spec to the node that produces the deliverable, "
    "a role/analysis spec to the reasoning node. A node's specialization follows "
    "what THAT node DOES, not the goal's final format: a RESEARCH / GATHER / "
    "ANALYSIS step (it searches, fetches, reads, takes notes) takes a research or "
    "analysis spec (e.g. research-analyst) or NONE — its job is to gather grounded "
    "findings, not to produce the deliverable. A document-FORMAT spec (e.g. "
    "html-writer, markdown-writer) describes the FINAL written deliverable, so bind "
    "it ONLY to the node that WRITES that deliverable; NEVER put a document-format "
    "spec on a research/gather node — doing so makes that leaf emit the formatted "
    "document instead of gathering notes (format-bleed). Choose the spec whose "
    "advertised description best MATCHES each node's work — the registered "
    "specialization lookup (with each spec's description) is your guide. If a "
    "registered specialization fits, set 'spec' to its name. Use "
    "'specs' (applied in order) ONLY when 2+ "
    "genuinely COMPOSE (e.g. an analysis spec + an output-format spec on the same "
    "node) — keep it to 2-3 and never stack two conflicting output styles; never "
    "put a single name in 'specs' or set both. A user-requested spec wins over a "
    "default. If NO spec clearly fits, leave the node unspecialized (spec/specs "
    "empty) — do NOT force-fit a loosely related one. If a step REQUIRES a "
    "specialist no registered spec covers, leave spec/specs empty and describe it "
    "in 'needs_spec' — never invent a name or silently run it unspecialized.\n"
    "- Tool: if a step uses a tool, set 'tool' to its name and its arguments in "
    "'tool_args'.\n"
    "Emit ONLY the plan that fits THIS goal."
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
    "role": "in-plan node role, one of researcher|worker|reviewer, or null (null = a "
            "plain producer step; 'researcher' = a gather node; make the LAST node a "
            "'reviewer' that fixes the deliverable and emits the final status). Do NOT "
            "use 'planner' or 'synthesizer' — those are stages, not in-plan nodes",
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
        system = with_identity(
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
        system = with_identity(
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

    def _raw_node_dicts(
        self, structured: Any
    ) -> tuple[list[dict[str, Any]], str, str]:
        """Coerce phi's parsed JSON into clean per-node kwargs dicts (shared front
        half of :meth:`parse_dag` / :meth:`parse_dag_safe`).

        Validates only the ENVELOPE (plan is an object/list with a non-empty
        ``nodes`` list; each raw node is an object) and normalises each node's
        fields — ``depends_on`` is kept as a plain ``list[str]`` so the repair pass
        can edit it before the :class:`PlanNode`/:class:`PlanDAG` invariants
        (id/task/role field checks + ref/cycle/dup GRAPH checks) are enforced at
        build time. Raises :class:`PlanError` on a malformed envelope/node."""
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

        out: list[dict[str, Any]] = []
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
            out.append(
                {
                    "id": str(raw.get("id") or f"n{i+1}"),
                    "task": str(raw.get("task") or raw.get("description") or ""),
                    "spec": (str(raw["spec"]) if raw.get("spec") else None),
                    "specs": specs,
                    "depends_on": [str(d) for d in dep],
                    "tool": (str(raw["tool"]) if raw.get("tool") else None),
                    "tool_args": dict(raw.get("tool_args") or {}),
                    # ROLE (optional): a known role or null; PlanNode validates it.
                    "role": (str(raw["role"]) if raw.get("role") else None),
                    # NEEDS_SPEC (optional, s4 RC8): the planner's free-text
                    # missing-specialist signal; PlanNode normalises blanks to None.
                    "needs_spec": (
                        str(raw["needs_spec"]) if raw.get("needs_spec") else None
                    ),
                    # SOURCE-SCOPING (s9/c13, d56): the global SOURCE id(s) this
                    # section node owns; PlanNode normalises to positive-int tuple.
                    "source_ids": raw.get("source_ids") or (),
                    # MEMORY-BY-HANDLE (d221): the research-memory handle this node is
                    # bound to (read-via-tools); PlanNode normalises blanks to None.
                    "research_memory_handle": (
                        str(raw["research_memory_handle"])
                        if raw.get("research_memory_handle") else None
                    ),
                    # MEMORY-INDEX (d285 SB-3): the planner-authored research-memory
                    # CHOICE on the step brief (an index to continue, or <<NEW>>);
                    # PlanNode keeps it as a clean string ("" = unspecified).
                    "memory_index": str(raw.get("memory_index") or ""),
                }
            )
        return out, rationale, shape

    @staticmethod
    def _build_dag(
        node_dicts: Sequence[Mapping[str, Any]], rationale: str, shape: str
    ) -> PlanDAG:
        """Construct + EAGERLY VALIDATE a :class:`PlanDAG` from kwargs dicts.

        ``PlanNode`` enforces the per-node field invariants and ``PlanDAG``
        (via ``__post_init__`` → ``validate``) the graph invariants (unique ids,
        resolvable ``depends_on`` refs, acyclic) — so a structurally-invalid plan
        still raises :class:`PlanError` here exactly as before."""
        nodes = [
            PlanNode(
                id=d["id"],
                task=d["task"],
                spec=d["spec"],
                specs=tuple(d["specs"]),
                depends_on=tuple(d["depends_on"]),
                tool=d["tool"],
                tool_args=dict(d["tool_args"]),
                role=d["role"],
                needs_spec=d["needs_spec"],
                source_ids=tuple(d.get("source_ids") or ()),
                research_memory_handle=d.get("research_memory_handle"),
                memory_index=str(d.get("memory_index") or ""),  # d285 SB-3
            )
            for d in node_dicts
        ]
        return PlanDAG(nodes=nodes, rationale=rationale, shape=shape)

    @staticmethod
    def _repair_dangling_edges(node_dicts: Sequence[dict[str, Any]]) -> list[str]:
        """Drop, IN PLACE, every ``depends_on`` ref to an unknown node or to self.

        This is the NARROW b2 safe-fallback for the a2 live-observed think=True
        malformation: the planner non-deterministically emits a node whose
        ``depends_on`` names a PHANTOM id (no such node). A dangling edge only
        *adds* an unsatisfiable ordering constraint, so dropping it degrades the
        plan gracefully — the node simply loses that ordering hint (it may start
        earlier) instead of the whole run failing. Dropping edges only REMOVES
        constraints, so it can never introduce a cycle.

        Returns human-readable repair notes (empty list = nothing was dangling, so
        the result is byte-identical to the strict path). DELIBERATELY does NOT
        touch duplicate ids or a real cycle among RESOLVABLE edges — those are
        genuine ambiguities the strict validator must still reject so the planner's
        outer self-heal keeps its retry-on-reject backstop."""
        known = {d["id"] for d in node_dicts}
        repairs: list[str] = []
        for d in node_dicts:
            kept: list[str] = []
            for dep in d["depends_on"]:
                if dep == d["id"]:
                    repairs.append(
                        f"node {d['id']!r}: dropped self-referential depends_on"
                    )
                elif dep not in known:
                    repairs.append(
                        f"node {d['id']!r}: dropped dangling depends_on {dep!r}"
                    )
                else:
                    kept.append(dep)
            d["depends_on"] = kept
        return repairs

    def parse_dag(self, structured: Any) -> PlanDAG:
        """Turn phi's parsed JSON (a dict/list) into a validated :class:`PlanDAG`.

        Accepts either ``{"nodes": [...], "rationale": ...}`` or a bare list of
        node objects. STRICT: raises :class:`PlanError` on a malformed/empty plan —
        INCLUDING a dangling ``depends_on`` edge — so a caller that wants the "no
        silent bad DAG" guarantee gets it unchanged. The live planner emission path
        instead uses :meth:`parse_dag_safe`, which repairs dangling edges."""
        node_dicts, rationale, shape = self._raw_node_dicts(structured)
        return self._build_dag(node_dicts, rationale, shape)

    def parse_dag_safe(self, structured: Any) -> tuple[PlanDAG, list[str]]:
        """Parse into a validated DAG, REPAIRING dangling/self ``depends_on`` edges
        instead of rejecting them — the b2 safe fallback for the a2 finding.

        Returns ``(dag, repairs)`` where ``repairs`` lists every edge dropped
        (empty when the plan was already clean → byte-identical to
        :meth:`parse_dag`). Repair is intentionally NARROW (see
        :meth:`_repair_dangling_edges`): only unresolvable / self refs are dropped,
        so the node degrades gracefully rather than the run failing. Every OTHER
        invalidity (duplicate ids, a real cycle among resolvable edges, an empty
        plan, a bad field) still raises :class:`PlanError` — the planner's outer
        self-heal keeps its retry-on-reject backstop for malformations repair
        cannot safely fix."""
        node_dicts, rationale, shape = self._raw_node_dicts(structured)
        repairs = self._repair_dangling_edges(node_dicts)
        return self._build_dag(node_dicts, rationale, shape), repairs


__all__ = [
    "PlanError",
    "PlanNode",
    "PlanDAG",
    "AbstractPlanFactory",
    "FACTORY_DESCRIPTION",
    "NODE_SCHEMA",
    "VALID_ROLES",
]
