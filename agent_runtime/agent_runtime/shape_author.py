"""NL description -> Gemma-authored DECLARATIVE plan-SHAPE file (d14(2), s9).

The user will NEVER hand-write a shape file (d14(2)): the Shapes screen lets them
DESCRIBE a shape in natural language and Gemma AUTO-GENERATES the declarative TOML
the runtime loads — exactly as a specialization is chat-authored+compiled
(:mod:`chat_app.spec_chat` / :mod:`specialization.compiler`). This module is the
shape-side equivalent of that authoring mechanism: it turns one NL description into
a validated :class:`~agent_runtime.shapes.ShapeSpec` and writes it as a TOML file
into the ``shapes/`` directory the runtime's :func:`~agent_runtime.shapes.load_shapes`
reads — so an authored shape is immediately SELECTABLE (the shape-selection enum is
harvested from disk at call time) and RUNNABLE (the generic unroll / scheduler
consume it unchanged).

Why the s8 declarative format is clean enough for a small model to generate
---------------------------------------------------------------------------
A shape is fully described by a handful of fields
(``name, description, execution, max_iter`` — see
:class:`~agent_runtime.shapes.ShapeSpec`), so authoring it is a SINGLE small
structured decision, not the whole-DAG one-shot a 4.6B model struggles with. s16/a3
(d239/d247): a shape declares NO per-node topology in ANY family — round_roles/final_roles
are RETIRED. Two plug-n-play families, one schema:

* a DISCIPLINE shape (``execution`` = ``sequential`` / ``concurrent``) carries NO
  per-node topology — its DAG is authored node-by-node at plan time by the
  :class:`~agent_runtime.incremental.IncrementalPlanner`; the shape only declares
  the dispatch posture (``max_iter`` is 1).
* the deep-research family (``execution`` = ``deep-research``) ALSO carries no fixed
  topology: its research DAG is AUTHORED at runtime by the
  :class:`~agent_runtime.research_tree.DagGrower` (decompose-first → grow on note gaps) from
  an engine-owned tool-less growable seed. The shape carries only the ``max_iter`` depth
  ceiling + the ``expand_on_gaps`` growable marker (+ doctrine in the canonical file).

The call uses the PROMPT-JSON reasoning path (the SAME one
:class:`~agent_runtime.incremental.IncrementalPlanner` uses for tool-call
authoring): ``api=native``, ``think=True`` TOP-LEVEL (s1/b1 reasoning rollout —
gemma4 reasons in the SEPARATE message.thinking field; ``num_predict`` raised to
4096 so the CoT cannot starve the JSON content to EMPTY), ``temperature=0``
(deterministic), and the JSON elicited by the PROMPT (keys + enums spelled out in
:func:`_system_prompt`) — **NOT** a constrained ``format=<schema>``. This is the
b6 (d34/d18a) change: the local-gemma specialist [required] rule is that a
load-bearing REASONED field must never sit behind ``format``-schema constrained
decoding, because constrained decoding trades content fidelity for syntactic
validity and silently DROPS a correctly-reasoned value — exactly the failure that
collapsed a COMPOSITIONAL "linear plus modular parallel" description down to the
single most-restrictive ``execution`` enum value (``sequential``), losing the
parallel phase. The model now reasons the posture freely (think=True), emits the
JSON in ``message.content``, and the transport's fence-stripping JSON interceptor +
the ``structured_output`` repair stage + :func:`_coerce_spec`'s validate-and-repair
enforce the contract instead. :func:`build_shape_schema` is kept as the documented
contract (and the behavioural-proof artifact), not as a wire constraint.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional

from llm_framework import Chain, Context, Transport
from llm_framework.stages import call_stage, prompt_assembly, structured_output

from .identity import with_identity
from .selfheal import MalformedOutputError
from .shapes import (
    SHAPES_DIR,
    VALID_EXECUTION,
    ShapeError,
    ShapeSpec,
    load_shape,
    load_shapes,
)
from .tracing import get_tracer, run_blocking_in_span

# The default safety ceiling stamped onto an authored deep-research shape's
# ``hard_cap`` — the absolute round bound the runtime never exceeds regardless of a
# UI ``max_iter`` override (shared-GPU safety, mirrors the shipped deep-research
# shape's 24). Authoring never lets the model set this; it is policy, not content.
DEFAULT_DEEP_RESEARCH_HARD_CAP = 24

# num_predict for the WHOLE shape object. The shape JSON itself is small, BUT s1/b1
# enables ``think=True`` on this authoring call (gemma4 reasons in the SEPARATE
# message.thinking field) and those thinking tokens compete with the content budget,
# so the cap is raised 512->4096 (the a2-proven load-bearing bump: at <=512 the CoT
# alone fills the budget and the JSON ``content`` truncates to EMPTY). temp 0 still
# holds the actual shape object tight; this is headroom for the CoT, not a larger shape.
DEFAULT_NUM_PREDICT = 4096

# The reserved shape-selection escalation token — a shape file may never be named
# this (it is the selector's "no shape fits" signal), so authoring rejects it.
_RESERVED_NAMES = frozenset({"escalate"})

_UNROLLABLE = "deep-research"


def build_shape_schema() -> dict[str, Any]:
    """The native ``format`` schema for ONE authored shape (enum + required keys).

    ``execution`` is enum-constrained to :data:`~agent_runtime.shapes.VALID_EXECUTION` so
    Gemma can only emit a legal discipline vocabulary. EVERY field is ``required`` — the
    small model reliably fills a fully-specified object but omits OPTIONAL ones. s16/a3
    (d239/d247): a shape NO LONGER declares per-round node positions (round_roles/final_roles
    RETIRED) — the deep-research research topology is AUTHORED at runtime by reasoning, so a
    shape is fully described by { name, description, execution, max_iter }."""
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "a short kebab-case id for this shape (e.g. 'parallel-news')",
            },
            "description": {
                "type": "string",
                "description": (
                    "a STRONG, DISCRIMINATIVE one-liner the shape SELECTOR reads "
                    "to pick this shape: name the KIND of work it fits and WHEN to "
                    "choose it OVER the other shapes (e.g. 'choose for independent "
                    "sub-tasks gathered in parallel then combined') — never a bare "
                    "restatement of the name or posture"
                ),
            },
            "execution": {
                "type": "string",
                "enum": sorted(VALID_EXECUTION),
                "description": (
                    "the dispatch posture: 'sequential' = a strict one-after-another "
                    "chain; 'concurrent' = independent sub-tasks run at the same time "
                    "then combine (modular-parallel); 'deep-research' = bounded "
                    "iterative research that decomposes the topic then deepens "
                    "round-by-round, ending in synthesis+verify"
                ),
            },
            "max_iter": {
                "type": "integer",
                "description": (
                    "round/depth ceiling — ONLY meaningful for 'deep-research' (e.g. 10); "
                    "use 1 for 'sequential'/'concurrent'"
                ),
            },
        },
        "required": [
            "name",
            "description",
            "execution",
            "max_iter",
        ],
    }


def _authoring_guidance() -> str:
    """The shared family + rules block used by BOTH the create and refine prompts.

    Frames the shape vocabulary (keys, the three execution families, the role
    vocabulary), the COMPOSITIONAL/multi-pattern rule (the b6/d18a fix that stops a
    'linear plus modular parallel' description collapsing to a flat ``sequential``
    shape), and the d14 STRONG, selection-effective ``description`` mandate."""
    return (
        "A plan SHAPE is the EXECUTION POSTURE of a plan — NOT its individual steps. "
        "It is PLANNER INPUT: the planner reads this shape (its 'execution' posture "
        "and its 'description') to decide HOW to construct the per-task DAG — whether "
        "to chain steps sequentially, fan independent steps out in parallel, or run "
        "bounded deepening research rounds. Author it FOR THAT CONSUMER: condensed, "
        "actionable guidance the planner can apply directly, NOT a verbose essay. "
        "Every field MUST be SHORT and PRECISE — the planner loads this on every plan. "
        "Emit STRICT JSON for ONE shape with keys: "
        '{"name", "description", "execution", "max_iter"}.\n\n'
        "Choose 'execution' to fit the description:\n"
        "- 'deep-research': ITERATIVE, DEEPENING research — decompose the topic into facets, "
        "gather, then deepen round-by-round, ending in synthesis + verify. Set max_iter = the "
        "round/depth ceiling (~10 if unspecified). (The research topology is authored at "
        "runtime — you do NOT declare per-round node roles.)\n"
        "- 'concurrent': INDEPENDENT sub-tasks run AT THE SAME TIME then "
        "combine/deliver. Set max_iter=1.\n"
        "- 'sequential': steps run STRICTLY one after another, never overlapping. "
        "Set max_iter=1.\n\n"
        "COMPOSITIONAL / MULTI-PATTERN intent (IMPORTANT): when a description "
        "combines a SEQUENTIAL/linear phase WITH a parallel/modular phase — e.g. "
        "'linear plus modular parallel', or 'a foundation phase, THEN independent "
        "avenues explored in parallel, then combined' — choose execution='concurrent'. "
        "'concurrent' is the ONLY posture that supports BOTH patterns at once: the "
        "planner chains the sequential phase with depends_on edges AND fans out the "
        "independent parallel steps. NEVER choose 'sequential' for such a description "
        "— 'sequential' forbids ALL fan-out and would FLATTEN the parallel phase into "
        "a single file (the exact collapse this shape must avoid). For a compositional "
        "shape, write the 'description' to SPELL OUT the phased structure (the "
        "sequential foundation phase, THEN the parallel fan-out, THEN how the results "
        "combine) so the planner authors the real mix, not a flat line.\n\n"
        "RULES:\n"
        "- A shape NEVER declares per-node topology: for EVERY posture the steps/rounds are "
        "authored later (per-task at plan time, or — for 'deep-research' — by the research "
        "engine reasoning at runtime). The shape only sets the dispatch posture + (for "
        "deep-research) the max_iter depth ceiling.\n"
        "- 'name' is a short kebab-case id.\n"
        "- 'description' is SELECTION-CRITICAL — it is the ONLY text the shape "
        "SELECTOR reads to choose this shape over the others. Write a STRONG, "
        "DISCRIMINATIVE one-liner that names the KIND of work this shape fits and "
        "WHEN to pick it OVER the other shapes (for a compositional shape, name BOTH "
        "phases). A bare restatement of the name or the bare posture word is NOT "
        "acceptable — it makes the shape unselectable.\n"
    )


def _catalog_context(shapes_dir: Optional[Path] = None, *, limit: int = 12) -> str:
    """A compact EXISTING-SHAPES block for the author/refine prompts (s17, d249).

    Context-aware generation: the model sees the catalog its new shape will be
    selected AGAINST (each shape's name + discipline + selection description), so
    it authors a shape that is genuinely DISTINCT and selectable — not a duplicate
    of an existing posture under a new name. Bounded to ``limit`` entries (the
    catalog is small; the block must stay cheap on the 32k window). Empty string
    when the catalog cannot be read — authoring must never fail on catalog IO."""
    try:
        catalog = load_shapes(shapes_dir)
    except Exception:
        return ""
    if not catalog:
        return ""
    lines = [
        f"- {spec.name} [{spec.execution}]: {spec.description}"
        for _, spec in sorted(catalog.items())[:limit]
    ]
    return (
        "\n\nEXISTING SHAPES (the catalog the planner already selects from — these are "
        "CONCRETE EXAMPLES of the form, and your shape must be DISTINCT from all of "
        "them in name AND selection-description):\n" + "\n".join(lines)
    )


def _system_prompt() -> str:
    """Frame the CREATE task: author one shape from a fresh NL description."""
    return with_identity(
        "You AUTHOR a declarative plan-SHAPE from a natural-language description. "
        + _authoring_guidance()
        + "Return ONLY the JSON for this one shape."
    )


def _refine_system_prompt() -> str:
    """Frame the EDIT/REFINE task: emit the UPDATED shape, building on the prior one.

    The free-flow ITERATIVE authoring half of b6 (d18a): the user refines an
    EXISTING shape in plain language and the model emits the NEXT version BUILDING
    ON the current definition (not a one-shot create). Shares the exact same family
    + compositional + strong-description rules as :func:`_system_prompt` so a refined
    shape is held to the same contract a created one is."""
    return with_identity(
        "You REFINE an EXISTING declarative plan-SHAPE. You are given the shape's "
        "CURRENT definition and a refinement instruction; emit the UPDATED shape — "
        "BUILDING ON the current one, preserving every part the instruction does not "
        "change, applying the requested change, and KEEPING the same 'name'. This is "
        "an EDIT of an existing shape, not a fresh authoring.\n\n"
        + _authoring_guidance()
        + "Return ONLY the JSON for the one UPDATED shape."
    )


def _refine_user(prior: ShapeSpec, instruction: str) -> str:
    """The refine USER turn: the prior shape (as JSON) + the requested change."""
    prior_json = json.dumps(
        {
            "name": prior.name,
            "description": prior.description,
            "execution": prior.execution,
            "max_iter": int(prior.max_iter),
        },
        indent=2,
    )
    return (
        "CURRENT SHAPE (edit THIS, do not start over):\n"
        f"{prior_json}\n\n"
        f"REFINEMENT REQUESTED:\n{str(instruction).strip()}\n\n"
        "Return ONLY the updated shape JSON (same keys, same 'name')."
    )


# Lower-cased cue sets for the compositional safety-net (a deterministic backstop
# BEHIND the prompt, mirroring the d28/d7 finalize guarantees). The net only ever
# upgrades 'sequential' -> 'concurrent', which is behaviourally SAFE: under
# 'concurrent' dispatch a fully-chained DAG runs exactly as it would under
# 'sequential' (each node still waits on its single dependency), so re-enabling
# fan-out can never make a genuinely linear plan wrong — it only stops a
# compositional intent being flattened.
_PARALLEL_CUES = (
    "parallel",
    "concurrent",
    "simultaneous",
    "at the same time",
    "at once",
    "in parallel",
)
_SEQUENTIAL_CUES = (
    "sequential",
    "linear",
    "one after another",
    "one at a time",
    "in sequence",
    "step by step",
    "step-by-step",
)
_COMPOSITION_CUES = (
    "plus",
    " then ",
    "followed by",
    "phase",
    "stage",
    "combine",
    "both",
    "after that",
    "afterwards",
)


def _looks_compositional(text: str) -> bool:
    """True iff ``text`` signals BOTH a parallel phase AND a sequential/multi-phase one.

    Used as the deterministic backstop in :func:`_coerce_spec`: a description that
    explicitly couples a parallel/modular phase with a sequential or otherwise
    phased structure (e.g. 'linear plus modular parallel') is COMPOSITIONAL and must
    not be dispatched as flat ``sequential``. Requires a parallel cue (so a purely
    linear description never trips it) joined with either a sequential cue or a
    composition/phase cue."""
    t = f" {str(text or '').lower()} "
    has_parallel = any(cue in t for cue in _PARALLEL_CUES)
    if not has_parallel:
        return False
    has_sequence = any(cue in t for cue in _SEQUENTIAL_CUES)
    has_compose = any(cue in t for cue in _COMPOSITION_CUES)
    return has_sequence or has_compose


def _slugify(text: str, *, fallback: str) -> str:
    """A safe kebab-case shape id (lowercase, hyphen-joined, non-empty, non-reserved)."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug or slug in _RESERVED_NAMES:
        slug = re.sub(r"[^a-z0-9]+", "-", str(fallback).strip().lower()).strip("-")
    return slug or "authored-shape"


def _coerce_spec(
    parsed: Mapping[str, Any], *, name_hint: str, request_text: str = ""
) -> ShapeSpec:
    """Validate one parsed authoring reply into a consistent :class:`ShapeSpec`.

    Reconciles the two families so the on-disk file is always coherent: a
    discipline shape (sequential/concurrent) carries NO roles and a single round;
    a deep-research shape MUST carry round/final roles (an empty one is an
    authoring failure, surfaced for the self-heal/caller, never silently shipped as
    an inert shape).

    ``request_text`` is the originating NL request (the create description or the
    refine instruction + prior description). It feeds the COMPOSITIONAL safety-net
    (b6/d18a): if the model picked the most-restrictive ``sequential`` posture for a
    description that clearly couples a parallel phase with a sequential one, the
    posture is upgraded to ``concurrent`` so the parallel phase is not flattened (a
    deterministic backstop behind the prompt, same spirit as the d28/d7 finalize
    guarantees; never touches ``deep-research`` or an already-``concurrent`` shape)."""
    execution = str(parsed.get("execution", "")).strip().lower()
    if execution not in VALID_EXECUTION:
        raise MalformedOutputError(
            f"authored shape has unknown execution {execution!r}; "
            f"valid: {sorted(VALID_EXECUTION)}"
        )
    description = " ".join(str(parsed.get("description", "")).split())
    name = _slugify(parsed.get("name", ""), fallback=name_hint or execution)

    # COMPOSITIONAL safety-net: a 'sequential' posture for a description that names
    # BOTH a parallel and a sequential/phased structure is the flat-collapse this
    # action must prevent — upgrade it to 'concurrent' (which subsumes sequential
    # via depends_on edges, so the linear phase is preserved while the parallel
    # phase is re-enabled). Considers the model's own description AND the request.
    if execution == "sequential" and _looks_compositional(
        f"{request_text} {description}"
    ):
        execution = "concurrent"

    try:
        max_iter = int(parsed.get("max_iter") or 1)
    except (TypeError, ValueError):
        max_iter = 1

    # s16/a3 (d239/d247): a shape carries NO per-node topology in ANY posture — round_roles/
    # final_roles are RETIRED. A 'deep-research' shape is identified by the execution token
    # alone (ShapeSpec.is_deep_research) and its research topology is AUTHORED at runtime by
    # the grower; it only carries the max_iter depth ceiling (+ doctrine in the canonical file).
    # Every other posture is a discipline shape whose DAG the incremental planner authors at
    # plan time, so it pins a single round.
    if execution == _UNROLLABLE:
        max_iter = max(max_iter, 2)
        hard_cap = max(max_iter, DEFAULT_DEEP_RESEARCH_HARD_CAP)
        # Mark the authored deep-research shape growable so the engine builds a growable seed
        # (the canonical shipped shape sets this too); the research topology stays reasoned.
        expand_on_gaps = True
    else:
        max_iter = 1
        hard_cap = 1
        expand_on_gaps = False

    # ShapeSpec.__post_init__ re-validates execution + bounds, so a coerced spec is
    # structurally guaranteed before it is ever written.
    return ShapeSpec(
        name=name,
        description=description,
        max_iter=max_iter,
        hard_cap=hard_cap,
        execution=execution,
        expand_on_gaps=expand_on_gaps,
        source="<gemma-authored>",
    )


def shape_to_toml(spec: ShapeSpec) -> str:
    """Serialise a :class:`ShapeSpec` to a clean TOML document (round-trippable).

    Emits only the fields the loader reads, in a stable order, with a provenance
    header marking it Gemma-authored. String values go through :func:`json.dumps`
    whose escaping is a safe subset of TOML basic-string syntax, so a description
    with quotes/newlines round-trips through ``tomllib``. s16/a3 (d239/d247): a shape
    declares NO per-node topology — a 'deep-research' shape carries only the execution
    token + the max_iter depth ceiling (+ the growable marker); its research topology
    is authored at runtime by the grower."""
    lines = [
        "# =============================================================================",
        f"# {spec.name} shape — AUTHORED FROM A NATURAL-LANGUAGE DESCRIPTION by the",
        "# local Gemma model (d14(2)); the user never hand-writes this file. The runtime",
        "# loads it via agent_runtime.shapes.load_shapes like any built-in shape.",
        "# =============================================================================",
        f"name = {json.dumps(spec.name)}",
        f"description = {json.dumps(spec.description)}",
        f"execution = {json.dumps(spec.execution)}",
    ]
    if spec.is_deep_research:
        lines.append(f"max_iter = {int(spec.max_iter)}")
        lines.append(f"hard_cap = {int(spec.hard_cap)}")
        # The growable marker: the engine builds a tool-less self-selecting research seed and
        # the DagGrower authors the topology by reasoning (no deterministic unroll).
        lines.append(f"expand_on_gaps = {json.dumps(bool(spec.expand_on_gaps))}")
    return "\n".join(lines) + "\n"


def write_shape(spec: ShapeSpec, *, shapes_dir: Optional[Path] = None) -> Path:
    """Write ``spec`` as ``<name>.toml`` into the shapes dir the runtime loads.

    Defaults to the package's :data:`~agent_runtime.shapes.SHAPES_DIR` (production:
    the authored shape lands in the catalog the selector harvests). A caller may
    point at any directory (e.g. a UI's per-install shapes dir, or a temp dir in a
    smoke). Returns the written path."""
    directory = Path(shapes_dir) if shapes_dir is not None else SHAPES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec.name}.toml"
    path.write_text(shape_to_toml(spec), encoding="utf-8")
    return path


def delete_shape(name: str, *, shapes_dir: Optional[Path] = None) -> Path:
    """Delete a shape's ``<name>.toml`` from the shapes dir the runtime loads.

    The structural store is a plain directory globbed on every
    :func:`~agent_runtime.shapes.load_shapes` call (stateless), so removing a shape
    is just unlinking its file — a deleted shape is simply never offered to a fresh
    plan, and a resume referencing it falls back to CONCURRENT (handled in routes).
    Name→file is the direct ``<name>.toml`` map :func:`write_shape` writes. Defaults
    to the package :data:`~agent_runtime.shapes.SHAPES_DIR`. Raises
    :class:`~agent_runtime.shapes.ShapeError` if the file is absent (the route maps
    that to 404). Returns the unlinked path."""
    directory = Path(shapes_dir) if shapes_dir is not None else SHAPES_DIR
    path = directory / f"{name}.toml"
    if not path.exists():
        raise ShapeError(f"no shape file named {name!r} to delete")
    path.unlink()
    return path


class ShapeAuthor:
    """Author a declarative shape from an NL description via one native Gemma call.

    Parameters
    ----------
    transport:
        Any ``llm_framework`` ``Transport`` (the live ``OllamaTransport`` or an
        offline ``FakeTransport``). The authoring call goes through it with the d1
        native structured options.
    shapes_dir:
        Default output directory for :meth:`author_and_write` (defaults to the
        package :data:`~agent_runtime.shapes.SHAPES_DIR`).
    num_predict:
        Output-token budget for the one authoring call (:data:`DEFAULT_NUM_PREDICT`).
    call_opts:
        Extra transport options merged OVER the proven native structured defaults
        (``api=native``, ``think=True`` (s1/b1), ``temperature=0``, ``num_predict``).
    """

    def __init__(
        self,
        transport: Transport,
        *,
        shapes_dir: Optional[Path] = None,
        max_repair_attempts: int = 2,
        num_predict: int = DEFAULT_NUM_PREDICT,
        call_opts: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.transport = transport
        self.shapes_dir = Path(shapes_dir) if shapes_dir is not None else None
        self.max_repair_attempts = max_repair_attempts
        # s1/b1 REASONING ROLLOUT: ``think=True`` (gemma4 reasons about the shape in
        # the SEPARATE message.thinking field before emitting the JSON); ``num_predict``
        # (DEFAULT_NUM_PREDICT raised to 4096) gives the CoT headroom so content is not
        # starved. An explicit caller ``call_opts`` still overrides.
        self.call_opts = {
            "api": "native",
            "think": True,
            "temperature": 0,
            "num_predict": int(num_predict),
            **(dict(call_opts) if call_opts else {}),
        }
        # Captured each call for the behavioural proof (the schema advertised + the
        # raw text + the coerced spec).
        self.last_schema: Optional[dict[str, Any]] = None
        self.last_raw: Optional[str] = None
        self.last_spec: Optional[ShapeSpec] = None

    def schema(self) -> dict[str, Any]:
        """The native ``format`` schema advertised for one authoring call."""
        return build_shape_schema()

    async def author(self, description: str, *, name_hint: str = "") -> ShapeSpec:
        """Author a validated :class:`ShapeSpec` from ``description`` (live Gemma).

        Runs the SAME assemble→call→parse+repair chain the shape selector / planner
        use (so the bounded malformed-JSON self-heal applies), offloaded off the
        event loop via :func:`run_blocking_in_span` (the d4 never-freeze fix).
        Raises :class:`MalformedOutputError` when the model returns no usable shape
        after the bounded repair loop, so a caller can re-author exactly as it would
        for a malformed plan."""
        if not description or not str(description).strip():
            raise MalformedOutputError("shape authoring needs a non-empty description")
        # b6/d34: schema is the documented contract + proof artifact, but it is NOT
        # passed as a wire `format` constraint (constrained decoding drops the
        # reasoned posture). The prompt elicits the JSON; the interceptor + repair
        # stage + `_coerce_spec` enforce it.
        self.last_schema = self.schema()
        opts = dict(self.call_opts)

        chain = Chain()
        chain.use(prompt_assembly())
        chain.use(call_stage(self.transport, **opts))
        chain.use(
            structured_output(self.transport, max_repair_attempts=self.max_repair_attempts)
        )
        ctx = Context(
            system=_system_prompt(),
            # s17 (d249): the existing catalog rides the user turn as usage-context —
            # concrete examples of the form + the set the new shape must be distinct from.
            user=(
                f"DESCRIPTION:\n{str(description).strip()}"
                + _catalog_context(self.shapes_dir)
                + "\n\nReturn ONLY the shape JSON."
            ),
            transport=self.transport,
        )
        tracer = get_tracer("agent_runtime.shape_author")
        with tracer.start_as_current_span("planner.author_shape") as span:
            span.set_attribute("author.description", str(description)[:1000])
            ctx = await run_blocking_in_span(chain.run, ctx)
            self.last_raw = ctx.raw_output
            parsed = ctx.structured
            if not isinstance(parsed, Mapping):
                repair = ctx.meta.get("structured_output", {})
                raise MalformedOutputError(
                    "shape authoring produced no parseable JSON after "
                    f"{self.max_repair_attempts} repair attempts: "
                    f"{repair.get('final_error')}"
                )
            spec = _coerce_spec(parsed, name_hint=name_hint, request_text=description)
            self.last_spec = spec
            span.set_attribute("author.shape_name", spec.name)
            span.set_attribute("author.execution", spec.execution)
            span.set_attribute("author.deep_research", spec.is_deep_research)
            return spec

    async def refine(
        self, prior: ShapeSpec, instruction: str, *, keep_name: bool = True
    ) -> ShapeSpec:
        """Author the NEXT version of ``prior`` from a plain-language ``instruction``.

        The free-flow ITERATIVE half of b6 (d18a): an EDIT that BUILDS ON the
        existing shape rather than a one-shot create. The prior shape is fed into the
        prompt (:func:`_refine_user`) and the model emits the updated shape, held to
        the same family/compositional/strong-description contract as a created one.
        Runs the SAME prompt-JSON reasoning path as :meth:`author` (no ``format``
        constraint). The refined shape KEEPS the prior name (an edit edits in place;
        the model is never allowed to rename and orphan the file), so the caller can
        overwrite the same ``<name>.toml``. The compositional safety-net sees both the
        prior description and the instruction, so refining 'make it run the phases in
        parallel' off a linear shape upgrades the posture deterministically. Raises
        :class:`MalformedOutputError` when no usable shape survives the repair loop."""
        if not instruction or not str(instruction).strip():
            raise MalformedOutputError("shape refinement needs a non-empty instruction")
        self.last_schema = self.schema()
        opts = dict(self.call_opts)

        chain = Chain()
        chain.use(prompt_assembly())
        chain.use(call_stage(self.transport, **opts))
        chain.use(
            structured_output(self.transport, max_repair_attempts=self.max_repair_attempts)
        )
        ctx = Context(
            system=_refine_system_prompt(),
            # s17 (d249): the catalog rides the refine turn too — the updated shape's
            # description must stay discriminative against its siblings.
            user=_refine_user(prior, instruction) + _catalog_context(self.shapes_dir),
            transport=self.transport,
        )
        tracer = get_tracer("agent_runtime.shape_author")
        with tracer.start_as_current_span("planner.refine_shape") as span:
            span.set_attribute("refine.shape_name", prior.name)
            span.set_attribute("refine.instruction", str(instruction)[:1000])
            ctx = await run_blocking_in_span(chain.run, ctx)
            self.last_raw = ctx.raw_output
            parsed = ctx.structured
            if not isinstance(parsed, Mapping):
                repair = ctx.meta.get("structured_output", {})
                raise MalformedOutputError(
                    "shape refinement produced no parseable JSON after "
                    f"{self.max_repair_attempts} repair attempts: "
                    f"{repair.get('final_error')}"
                )
            spec = _coerce_spec(
                parsed,
                name_hint=prior.name,
                request_text=f"{prior.description} {instruction}",
            )
            # An edit of an ON-DISK shape edits IN PLACE: force the prior name so the
            # model can never rename-and-orphan the file. A caller refining an
            # UNPERSISTED draft (the s17 shape chat's create mode — no file exists)
            # passes keep_name=False so a requested rename is honored.
            if keep_name and spec.name != prior.name:
                spec = replace(spec, name=prior.name)
            self.last_spec = spec
            span.set_attribute("refine.execution", spec.execution)
            span.set_attribute("refine.deep_research", spec.is_deep_research)
            return spec

    async def author_and_write(
        self,
        description: str,
        *,
        shapes_dir: Optional[Path] = None,
        name_hint: str = "",
    ) -> tuple[ShapeSpec, Path]:
        """Author the shape AND write it to disk; return ``(spec, path)``.

        After writing, the file is RE-LOADED through the real
        :func:`~agent_runtime.shapes.load_shape` so the returned spec is proven to
        round-trip the on-disk loader (the same loader the runtime uses) — never a
        write the runtime would then reject."""
        spec = await self.author(description, name_hint=name_hint)
        target_dir = (
            Path(shapes_dir)
            if shapes_dir is not None
            else (self.shapes_dir if self.shapes_dir is not None else None)
        )
        path = write_shape(spec, shapes_dir=target_dir)
        # Round-trip guard: the runtime loads from disk, so prove THIS write parses.
        reloaded = load_shape(spec.name, shapes_dir=target_dir)
        if reloaded.execution != spec.execution or reloaded.is_deep_research != spec.is_deep_research:
            raise ShapeError(
                f"authored shape {spec.name!r} did not round-trip the loader "
                f"(wrote execution={spec.execution!r}, read {reloaded.execution!r})"
            )
        return reloaded, path

    async def refine_and_write(
        self,
        prior_name: str,
        instruction: str,
        *,
        shapes_dir: Optional[Path] = None,
    ) -> tuple[ShapeSpec, Path]:
        """Load ``prior_name``, REFINE it, and OVERWRITE its file; return ``(spec, path)``.

        The on-disk edit half of the free-flow iterative authoring: loads the current
        shape from the same dir the runtime reads (raising :class:`ShapeError` if it
        is absent), authors the next version BUILDING ON it (:meth:`refine`), and
        writes it back to the SAME ``<name>.toml`` (overwrite is correct here — this
        is an edit of an existing shape, not a name collision). The written file is
        re-loaded through the REAL loader so a non-round-tripping edit is never
        shipped."""
        target_dir = (
            Path(shapes_dir)
            if shapes_dir is not None
            else (self.shapes_dir if self.shapes_dir is not None else None)
        )
        prior = load_shape(prior_name, shapes_dir=target_dir)
        spec = await self.refine(prior, instruction)
        path = write_shape(spec, shapes_dir=target_dir)
        reloaded = load_shape(spec.name, shapes_dir=target_dir)
        if reloaded.execution != spec.execution or reloaded.is_deep_research != spec.is_deep_research:
            raise ShapeError(
                f"refined shape {spec.name!r} did not round-trip the loader "
                f"(wrote execution={spec.execution!r}, read {reloaded.execution!r})"
            )
        return reloaded, path


__all__ = [
    "ShapeAuthor",
    "build_shape_schema",
    "shape_to_toml",
    "write_shape",
    "delete_shape",
    "DEFAULT_NUM_PREDICT",
    "DEFAULT_DEEP_RESEARCH_HARD_CAP",
]
