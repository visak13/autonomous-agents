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
(``name, description, execution, max_iter, round_roles, final_roles`` — see
:class:`~agent_runtime.shapes.ShapeSpec`), so authoring it is a SINGLE small
structured decision, not the whole-DAG one-shot a 4.6B model struggles with. Two
plug-n-play families, one schema:

* a DISCIPLINE shape (``execution`` = ``sequential`` / ``concurrent``) carries NO
  per-node topology — its DAG is authored node-by-node at plan time by the
  :class:`~agent_runtime.incremental.IncrementalPlanner`; the shape only declares
  the dispatch posture. So ``round_roles`` / ``final_roles`` are forced ``[]`` and
  ``max_iter`` is 1.
* the bounded-cyclic family (``execution`` = ``deep-research``) DOES carry topology
  declaratively: ``round_roles`` (each non-final round) + ``final_roles`` (the
  final round) + ``max_iter`` (the round ceiling), which
  :func:`~agent_runtime.shapes.unroll_shape` expands into the role-tagged DAG.

The call uses the PROVEN d1 native-structured path (the same one
:class:`~agent_runtime.shape_selector.ShapeSelector` /
:class:`~agent_runtime.incremental.IncrementalPlanner` use): ``api=native``,
``think=False`` TOP-LEVEL (gemma4 is a thinking model — off so the whole budget
goes to the JSON, not a CoT trace that returns EMPTY content), ``temperature=0``
(deterministic), ``num_predict`` sized to hold the WHOLE object, and the schema
passed as Ollama-native ``format=<schema>`` with ``enum``+``required`` keys. Per
the local-gemma specialist [required] rule, a JSON SCHEMA (not ``format:"json"``)
is what pins the ``execution`` enum, the role enums and the required keys — and
EVERY field is ``required`` (no optionals) because this model reliably fills a
fully-specified object but tends to OMIT optional signals.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional

from llm_framework import Chain, Context, Transport
from llm_framework.stages import call_stage, prompt_assembly, structured_output

from .factory import VALID_ROLES
from .selfheal import MalformedOutputError
from .shapes import (
    SHAPES_DIR,
    VALID_EXECUTION,
    ShapeError,
    ShapeSpec,
    load_shape,
)
from .tracing import get_tracer, run_blocking_in_span

# The default safety ceiling stamped onto an authored deep-research shape's
# ``hard_cap`` — the absolute round bound the runtime never exceeds regardless of a
# UI ``max_iter`` override (shared-GPU safety, mirrors the shipped deep-research
# shape's 24). Authoring never lets the model set this; it is policy, not content.
DEFAULT_DEEP_RESEARCH_HARD_CAP = 24

# num_predict for the WHOLE shape object. A shape is small (a name, a one-line
# description, an execution enum, an int, and two short role arrays), so this holds
# it at temperature 0 without the verbose small model running its output past the
# cap and truncating the JSON (the specialist structured-output [required] rule).
DEFAULT_NUM_PREDICT = 512

# The reserved shape-selection escalation token — a shape file may never be named
# this (it is the selector's "no shape fits" signal), so authoring rejects it.
_RESERVED_NAMES = frozenset({"escalate"})

_UNROLLABLE = "deep-research"


def build_shape_schema() -> dict[str, Any]:
    """The native ``format`` schema for ONE authored shape (enum + required keys).

    ``execution`` is enum-constrained to :data:`~agent_runtime.shapes.VALID_EXECUTION`
    and the two role arrays to :data:`~agent_runtime.factory.VALID_ROLES`, so Gemma
    can only emit a legal discipline / role vocabulary (Ollama enforces it at the
    wire). EVERY field is ``required`` — the small model reliably fills a
    fully-specified object but omits OPTIONAL ones, so a discipline shape emits
    EMPTY role arrays + ``max_iter`` 1 rather than leaving them unset."""
    roles = sorted(VALID_ROLES)
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "a short kebab-case id for this shape (e.g. 'parallel-news')",
            },
            "description": {
                "type": "string",
                "description": "one line: what this shape's execution posture is",
            },
            "execution": {
                "type": "string",
                "enum": sorted(VALID_EXECUTION),
                "description": (
                    "the dispatch posture: 'sequential' = a strict one-after-another "
                    "chain; 'concurrent' = independent sub-tasks run at the same time "
                    "then combine (modular-parallel); 'deep-research' = bounded "
                    "iterative rounds that each go deeper, a critic checking each "
                    "round, ending in synthesis+verify"
                ),
            },
            "max_iter": {
                "type": "integer",
                "description": (
                    "round ceiling — ONLY meaningful for 'deep-research' (e.g. 10); "
                    "use 1 for 'sequential'/'concurrent'"
                ),
            },
            "round_roles": {
                "type": "array",
                "items": {"type": "string", "enum": roles},
                "description": (
                    "node roles emitted EACH non-final round — ONLY for "
                    "'deep-research' (use [\"research\",\"critic\"]); [] otherwise"
                ),
            },
            "final_roles": {
                "type": "array",
                "items": {"type": "string", "enum": roles},
                "description": (
                    "node roles emitted in the FINAL round — ONLY for "
                    "'deep-research' (use [\"research\",\"synthesis\",\"verify\"]); "
                    "[] otherwise"
                ),
            },
        },
        "required": [
            "name",
            "description",
            "execution",
            "max_iter",
            "round_roles",
            "final_roles",
        ],
    }


def _system_prompt() -> str:
    """Frame the authoring task: describe the shape format + the two families."""
    return (
        "You AUTHOR a declarative plan-SHAPE from a natural-language description. A "
        "plan SHAPE is the EXECUTION POSTURE of a plan — NOT its individual steps. "
        "Emit STRICT JSON for ONE shape with keys: "
        '{"name", "description", "execution", "max_iter", "round_roles", '
        '"final_roles"}.\n\n'
        "Choose 'execution' to fit the description:\n"
        "- 'deep-research': the task wants ITERATIVE, DEEPENING research — repeated "
        "rounds that each build on the last, a critic checking each round, finishing "
        "with a synthesis and a verification. Set round_roles=[\"research\",\"critic\"], "
        "final_roles=[\"research\",\"synthesis\",\"verify\"], and max_iter to the number "
        "of rounds (about 10 if unspecified).\n"
        "- 'concurrent': the task splits into INDEPENDENT sub-tasks that can run AT "
        "THE SAME TIME and are then combined/delivered (e.g. gather several things in "
        "parallel, then email or save them). Set round_roles=[], final_roles=[], "
        "max_iter=1.\n"
        "- 'sequential': the steps must run STRICTLY one after another, each needing "
        "the previous. Set round_roles=[], final_roles=[], max_iter=1.\n\n"
        "RULES:\n"
        "- round_roles and final_roles are ONLY for 'deep-research'. For "
        "'concurrent' and 'sequential' they MUST be the empty list [] (those shapes' "
        "actual steps are authored later, per-task — the shape only sets the dispatch "
        "posture).\n"
        "- roles may ONLY be: research, critic, synthesis, verify, worker, reviewer.\n"
        "- 'name' is a short kebab-case id (lowercase words joined by hyphens).\n"
        "Return ONLY the JSON for this one shape."
    )


def _slugify(text: str, *, fallback: str) -> str:
    """A safe kebab-case shape id (lowercase, hyphen-joined, non-empty, non-reserved)."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug or slug in _RESERVED_NAMES:
        slug = re.sub(r"[^a-z0-9]+", "-", str(fallback).strip().lower()).strip("-")
    return slug or "authored-shape"


def _coerce_spec(parsed: Mapping[str, Any], *, name_hint: str) -> ShapeSpec:
    """Validate one parsed authoring reply into a consistent :class:`ShapeSpec`.

    Reconciles the two families so the on-disk file is always coherent: a
    discipline shape (sequential/concurrent) carries NO roles and a single round;
    a deep-research shape MUST carry round/final roles (an empty one is an
    authoring failure, surfaced for the self-heal/caller, never silently shipped as
    an inert shape)."""
    execution = str(parsed.get("execution", "")).strip().lower()
    if execution not in VALID_EXECUTION:
        raise MalformedOutputError(
            f"authored shape has unknown execution {execution!r}; "
            f"valid: {sorted(VALID_EXECUTION)}"
        )
    description = " ".join(str(parsed.get("description", "")).split())
    name = _slugify(parsed.get("name", ""), fallback=name_hint or execution)

    # Keep only legal roles (the enum should already guarantee this, but a repair
    # reply could slip a stray token through the parser).
    round_roles = tuple(
        r for r in (parsed.get("round_roles") or []) if str(r) in VALID_ROLES
    )
    final_roles = tuple(
        r for r in (parsed.get("final_roles") or []) if str(r) in VALID_ROLES
    )
    try:
        max_iter = int(parsed.get("max_iter") or 1)
    except (TypeError, ValueError):
        max_iter = 1

    if execution == _UNROLLABLE:
        if not round_roles and not final_roles:
            raise MalformedOutputError(
                "authored a deep-research shape with NO round_roles/final_roles — "
                "an inert cyclic shape; re-author with the round + final roles"
            )
        max_iter = max(max_iter, 2)
        hard_cap = max(max_iter, DEFAULT_DEEP_RESEARCH_HARD_CAP)
    else:
        # A discipline shape carries no per-node topology (the incremental planner
        # authors its DAG at plan time); force the roles empty + a single round so
        # the file can never be mistaken for an unrollable shape.
        round_roles = ()
        final_roles = ()
        max_iter = 1
        hard_cap = 1

    # ShapeSpec.__post_init__ re-validates execution + roles + bounds, so a coerced
    # spec is structurally guaranteed before it is ever written.
    return ShapeSpec(
        name=name,
        description=description,
        max_iter=max_iter,
        hard_cap=hard_cap,
        round_roles=round_roles,
        final_roles=final_roles,
        execution=execution,
        source="<gemma-authored>",
    )


def shape_to_toml(spec: ShapeSpec) -> str:
    """Serialise a :class:`ShapeSpec` to a clean TOML document (round-trippable).

    Emits only the fields the loader reads, in a stable order, with a provenance
    header marking it Gemma-authored. String/array values go through
    :func:`json.dumps` whose escaping is a safe subset of TOML basic-string syntax,
    so a description with quotes/newlines round-trips through ``tomllib``."""
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
    if spec.is_unrollable:
        lines.append(f"max_iter = {int(spec.max_iter)}")
        lines.append(f"hard_cap = {int(spec.hard_cap)}")
        lines.append(f"round_roles = {json.dumps(list(spec.round_roles))}")
        lines.append(f"final_roles = {json.dumps(list(spec.final_roles))}")
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
        (``api=native``, ``think=False``, ``temperature=0``, ``num_predict``).
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
        self.call_opts = {
            "api": "native",
            "think": False,
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
        schema = self.schema()
        self.last_schema = schema
        opts = {**self.call_opts, "format": schema}

        chain = Chain()
        chain.use(prompt_assembly())
        chain.use(call_stage(self.transport, **opts))
        chain.use(
            structured_output(self.transport, max_repair_attempts=self.max_repair_attempts)
        )
        ctx = Context(
            system=_system_prompt(),
            user=f"DESCRIPTION:\n{str(description).strip()}\n\nReturn ONLY the shape JSON.",
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
            spec = _coerce_spec(parsed, name_hint=name_hint)
            self.last_spec = spec
            span.set_attribute("author.shape_name", spec.name)
            span.set_attribute("author.execution", spec.execution)
            span.set_attribute("author.unrollable", spec.is_unrollable)
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
        if reloaded.execution != spec.execution or reloaded.is_unrollable != spec.is_unrollable:
            raise ShapeError(
                f"authored shape {spec.name!r} did not round-trip the loader "
                f"(wrote execution={spec.execution!r}, read {reloaded.execution!r})"
            )
        return reloaded, path


__all__ = [
    "ShapeAuthor",
    "build_shape_schema",
    "shape_to_toml",
    "write_shape",
    "DEFAULT_NUM_PREDICT",
    "DEFAULT_DEEP_RESEARCH_HARD_CAP",
]
