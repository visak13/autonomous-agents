"""Per-query SHAPE SELECTION via a native structured Gemma call (blueprint §2a, d1).

Shape selection is the planner's ONE genuinely model-driven choice before the DAG
is authored: given a user GOAL, WHICH plan shape fits it — a strictly sequential
``linear`` chain, a ``modular-parallel`` fan-out, the bounded cyclic
``deep-research`` shape, or any other text-file shape on disk? Everything that
FOLLOWS the choice (the readiness gate, the dispatch FSM) is deterministic and
lives in :mod:`agent_runtime.scheduler`; the choice itself is a Gemma judgment
point, so it uses the proven d1 native structured path:

* the OUTPUT SCHEMA's ``shape`` field is an ``enum`` whose values are HARVESTED
  from the shape files (:func:`~agent_runtime.shapes.shape_names`) PLUS a reserved
  ``escalate`` value the model picks when it is NOT confident any shape fits — so
  a low-confidence selection is an explicit, structured signal the caller can
  route to a human / a default, never a silent mis-pick;
* ``think=False`` TOP-LEVEL (gemma is a thinking model — off so the whole budget
  goes to the JSON decision, not a CoT trace), ``temperature=0`` (deterministic),
  a raised ``num_predict``, and the schema passed as Ollama native
  ``format=<schema>`` with ``required`` keys — the exact 24/24 planner path (d1);
* driven through the existing ``llm_framework`` chain (``call_stage`` + bounded
  ``structured_output`` repair), with the blocking phi round-trip offloaded off the
  event loop and traced like the planner's other judgment calls.

The enum is rebuilt from the on-disk catalog at call time, so adding a shape file
(or the s4 UI adding one) makes it selectable with NO code change here — the
growable-shapes requirement, mirrored from the growable tool registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from llm_framework import Chain, Context, Transport
from llm_framework.stages import call_stage, prompt_assembly, structured_output

from .selfheal import MalformedOutputError
from .shapes import ShapeSpec, load_shapes
from .tracing import get_tracer, run_blocking_in_span

# The reserved enum value the model picks when no shape confidently fits. It is
# NOT a shape name (a shape file may never be named this) — it is the low-
# confidence ESCALATION signal (blueprint §2a). The caller routes it to a human /
# a default rather than dispatching a mis-selected shape.
ESCALATE = "escalate"

# Native structured-call options (d1): the proven think=false / temp 0 path. The
# per-call JSON schema (with the harvested shape enum) is added as ``format=`` in
# :meth:`ShapeSelector.select`. ``think`` OFF so gemma emits the JSON selection
# directly; ``num_predict`` raised so the rationale never truncates the JSON.
_SELECT_OPTS: dict[str, Any] = {
    "api": "native",
    "think": False,
    "temperature": 0,
    "num_predict": 256,
}


@dataclass
class ShapeSelection:
    """The planner's structured shape choice for ONE query (blueprint §2a).

    ``shape`` is the selected shape NAME, or ``None`` when the model escalated.
    ``escalate`` is True iff the model picked the reserved low-confidence value.
    ``rationale`` is the model's one-line justification; ``raw`` the raw text.

    F5 ROUTING SIGNALS (intent-faithful, model-extracted in the SAME structured
    call — not a phrase-matcher): ``search_allowed`` is the model's read of whether
    the request PERMITS web search/fetch (False ONLY when the user explicitly says
    not to search — answer from your own knowledge), and ``requested_specs`` is the
    specialization name(s) the user EXPLICITLY named. The caller (``run_agentic``)
    enforces them STRUCTURALLY: a ``search_allowed=False`` run is offered NO web
    tools and never the search shapes, and a named spec is bound to the plan rather
    than overridden by the deep-research default. Both default to the permissive /
    empty value so a selector reply (or transport) that omits them is byte-identical
    to the pre-F5 behaviour (fail-open, no regression).

    FILE-OUTPUT SIGNAL (d11/s7-a2 invariant, s10-a4): ``wants_file`` is the model's
    read of whether the user asked for the result WRITTEN TO A FILE (saved as a
    file/document/report on disk), again by intent across any phrasing — not a
    keyword match. The caller enforces the invariant STRUCTURALLY: a file request
    must terminate in a file-writing output node, so when ``wants_file`` is True the
    inherently-fileless deep-research family is suppressed in favour of the acyclic
    path (which authors a terminal ``file_write`` node). Defaults to False so a reply
    that omits it is byte-identical to the prior behaviour (fail-open).

    MISSING-SPECIALIST SIGNAL (scenario-3 STRUCTURAL trigger, s10-a8): ``unmet_specs``
    is the FREE-TEXT name(s) of any specialization / expert role / named output-style
    the user EXPLICITLY asked for that is NOT in the AVAILABLE SPECIALIZATIONS list
    the selector advertises — the model CLASSIFIES the request (an available spec
    goes in ``requested_specs``, an unavailable one in ``unmet_specs``). It is
    DELIBERATELY a free string, NOT the registered-name enum that locks
    ``requested_specs`` — so the model CAN name a specialization the registry does
    not have (the enum structurally prevented this, which is why scenario-3 could
    never fire from ``requested_specs``). The caller does NOT trust the model's
    classification blindly: it re-applies a DETERMINISTIC registry-membership check
    (a name that is actually registered is dropped) and fires the missing-specialist
    notify + SSE-fallback / define-and-resume on whatever remains. This replaces the
    per-node ``needs_spec`` free-text the 4.6B model would not reliably volunteer
    (s10-a4) with a reliable shape-selector extraction + a deterministic trigger.
    Defaults to empty so a reply that omits it is byte-identical to the pre-a8
    behaviour (fail-open, no notify)."""

    shape: Optional[str]
    escalate: bool
    rationale: str = ""
    raw: Optional[str] = None
    search_allowed: bool = True
    requested_specs: list[str] = field(default_factory=list)
    wants_file: bool = False
    unmet_specs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "escalate": self.escalate,
            "rationale": self.rationale,
            "search_allowed": self.search_allowed,
            "requested_specs": list(self.requested_specs),
            "wants_file": self.wants_file,
            "unmet_specs": list(self.unmet_specs),
        }


def build_selection_schema(
    shape_names: list[str], spec_names: Optional[list[str]] = None
) -> dict[str, Any]:
    """The per-call OUTPUT SCHEMA: a ``shape`` enum (names + escalate) + required keys.

    The enum is the harvested catalog of shape names PLUS :data:`ESCALATE` — so the
    model can ONLY return a real shape or the explicit escalation value; Ollama's
    native ``format=<schema>`` enforces that at the wire (d1). ``required`` makes
    every key mandatory so a parse can never silently omit the decision.

    F5: the schema also carries two INTENT signals the model fills by READING the
    goal (not a keyword list) — ``search_allowed`` (may this request use the web?)
    and ``requested_specs`` (which listed specializations did the user name?). They
    are ``required`` so the small model reliably emits them under native
    ``format=<schema>`` (a non-required key is the one Ollama may drop —
    output-control comes from ``required``, not from the prose). ``requested_specs``
    items are enum-constrained to the REGISTERED spec names (when supplied) so the
    model cannot invent a specialization."""
    spec_names = list(spec_names or [])
    spec_item: dict[str, Any] = {"type": "string"}
    if spec_names:
        spec_item = {"type": "string", "enum": spec_names}
    return {
        "type": "object",
        "properties": {
            "shape": {
                "type": "string",
                "enum": list(shape_names) + [ESCALATE],
                "description": (
                    "the single best-fitting plan shape for the query, or "
                    f"'{ESCALATE}' if you are NOT confident any shape fits"
                ),
            },
            "rationale": {
                "type": "string",
                "description": "one line: why this shape fits the query",
            },
            "search_allowed": {
                "type": "boolean",
                "description": (
                    "true for a NORMAL request; false ONLY when the user EXPLICITLY "
                    "forbids searching/browsing the web (e.g. 'do not search', "
                    "'without searching', 'just from what you already know', 'from "
                    "your own knowledge'). When false the plan must use NO web tools."
                ),
            },
            "requested_specs": {
                "type": "array",
                "items": spec_item,
                "description": (
                    "the specialization name(s) from the AVAILABLE SPECIALIZATIONS "
                    "list that the user EXPLICITLY asked to use by name; [] when the "
                    "user named none. Do NOT guess — only a name the user actually "
                    "requested."
                ),
            },
            "wants_file": {
                "type": "boolean",
                "description": (
                    "true when the user asks for the result to be WRITTEN TO A FILE "
                    "or saved as a document/report/file on disk (e.g. 'write a "
                    "markdown file', 'save it as a file', 'create a .md document'); "
                    "false when they just want an answer in the chat."
                ),
            },
            "unmet_specs": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "the specialization / expert role / named output-style the user "
                    "EXPLICITLY asked for that is NOT in the AVAILABLE SPECIALIZATIONS "
                    "list above (e.g. the user wants a 'forensic-accountant report' "
                    "but no such specialization is available). Name the missing "
                    "capability in the user's own terms. Use [] when the user named "
                    "none OR when every specialization they asked for IS available "
                    "(put those in requested_specs instead). Do NOT invent a need the "
                    "user did not express."
                ),
            },
        },
        "required": [
            "shape",
            "rationale",
            "search_allowed",
            "requested_specs",
            "wants_file",
            "unmet_specs",
        ],
    }


class ShapeSelector:
    """Select a plan shape for a goal via a native structured Gemma call (d1).

    Parameters
    ----------
    transport:
        Any ``llm_framework`` ``Transport`` (the live ``OllamaTransport`` or an
        offline ``FakeTransport`` for tests). The selection call goes through it
        with the d1 structured options.
    shapes_dir:
        Optional shapes directory; defaults to the package's on-disk catalog. The
        enum is harvested from it AT CALL TIME so a newly added shape file is
        selectable with no code change here.
    max_repair_attempts:
        Bound on the structured-output JSON parse/repair loop for the selection
        call.
    call_opts:
        Extra transport options merged over :data:`_SELECT_OPTS` (e.g. a different
        ``num_predict``); the d1 defaults win unless overridden here.
    """

    def __init__(
        self,
        transport: Transport,
        *,
        shapes_dir: Optional[Any] = None,
        spec_names: Optional[Sequence[str]] = None,
        max_repair_attempts: int = 2,
        call_opts: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.transport = transport
        self.shapes_dir = shapes_dir
        # F5: the registered specialization names the model may recognise as
        # USER-REQUESTED (the ``requested_specs`` enum + the prompt's advertised
        # list). Empty when the caller supplies none — then named-spec extraction is
        # simply unconstrained/unused, identical to the pre-F5 behaviour.
        self.spec_names = [str(s) for s in (spec_names or []) if str(s).strip()]
        self.max_repair_attempts = max_repair_attempts
        self._call_opts = {**_SELECT_OPTS, **dict(call_opts or {})}
        # Captured each call for the behavioural proof (the exact enum advertised +
        # the catalog the model chose from).
        self.last_schema: Optional[dict[str, Any]] = None
        self.last_selection: Optional[ShapeSelection] = None

    def catalog(self) -> dict[str, ShapeSpec]:
        """The on-disk shape catalog harvested for this selection (name → spec)."""
        return load_shapes(self.shapes_dir)

    def _system_prompt(self, catalog: Mapping[str, ShapeSpec]) -> str:
        """Describe the available shapes (name + one-line description) + the rule.

        The model is told the catalog (names + descriptions, harvested from the
        shape files) and instructed to pick the ONE best-fitting shape for the
        query, or :data:`ESCALATE` when unsure — never to invent a shape."""
        lines = [
            "You are a plan-shape SELECTOR. Given a user GOAL, choose the SINGLE "
            "plan shape that best fits how the work should be executed. Choose ONLY "
            "from the shapes listed below — do not invent one. If you are NOT "
            f"confident any shape fits, choose '{ESCALATE}'.",
            "",
            "AVAILABLE SHAPES:",
        ]
        for name in sorted(catalog):
            desc = " ".join(str(catalog[name].description or "").split())
            lines.append(f"  - {name}: {desc}")
        lines.append(
            f"  - {ESCALATE}: none of the above clearly fits; defer the choice."
        )
        # F5: choose the shape for the request's ACTUAL INTENT, not its phrasing —
        # the same informational need asked as a question, a 'describe…', or an
        # imperative is the SAME work and should route the same way. Pick the shape
        # by the WORK to be done (one straight pass = linear; independent parts to
        # gather and combine = modular-parallel; an exhaustive multi-round survey
        # with critique = a deep-research-style shape), never by the surface wording.
        lines.append(
            "\nChoose the shape by the WORK the request needs, NOT by its phrasing: "
            "a question, a 'describe…', and an imperative that ask for the SAME "
            "result must route to the SAME shape. Do not over-escalate a simple "
            "informational request to a heavy multi-round shape just because it is "
            "phrased as a question."
        )
        # F5 SIGNAL 1 — no-search constraint. The model JUDGES whether the request
        # forbids the web (intent, across any phrasing), not a keyword match.
        lines.append(
            "\nALSO decide 'search_allowed': true normally, but FALSE when the user "
            "explicitly says not to use the web (e.g. 'do not search', 'without "
            "searching', 'just from what you already know', 'from your own "
            "knowledge'). When it is false, prefer a non-search shape (linear/"
            "modular-parallel), never a web-research shape."
        )
        # F5 SIGNAL 2 — user-named specialization(s). Advertise the catalog so the
        # model can recognise a name the user actually requested.
        if self.spec_names:
            lines.append("\nAVAILABLE SPECIALIZATIONS (the user may name one):")
            for s in sorted(self.spec_names):
                lines.append(f"  - {s}")
            lines.append(
                "Set 'requested_specs' to the specialization name(s) above the user "
                "EXPLICITLY asked to use by name (e.g. 'using the markdown-writer "
                "specialization'); use [] when the user named none. Do not guess."
            )
            # MISSING-SPECIALIST SIGNAL (scenario-3 structural trigger, a8). The
            # model CLASSIFIES a requested specialization by whether it appears in
            # the list above: an AVAILABLE one goes in requested_specs, an
            # UNAVAILABLE one in unmet_specs. This is what lets the runtime fire the
            # missing-specialist notify by a deterministic membership check (it
            # cannot from requested_specs, whose enum is locked to the available
            # names). The model only NAMES the request; the runtime decides "missing".
            lines.append(
                "\nIf the user EXPLICITLY asks for a specialization / expert role / "
                "named output-style that is NOT in the AVAILABLE SPECIALIZATIONS list "
                "above (e.g. asks for a 'forensic-accountant report' when no such "
                "specialization is listed), set 'unmet_specs' to that needed "
                "capability in the user's own terms. Use [] when the user named none "
                "or every specialization they asked for IS in the list above. Do not "
                "invent a need the user did not express."
            )
        else:
            lines.append(
                "\nSet 'requested_specs' to [] (no specialization catalog supplied)."
            )
            lines.append(
                "\nSet 'unmet_specs' to [] (no specialization catalog supplied)."
            )
        # FILE-OUTPUT SIGNAL (d11/s7-a2 invariant). The model JUDGES whether the
        # request wants the result saved to a file (intent, across any phrasing) —
        # a file request must end in a written file, never a chat-only answer.
        lines.append(
            "\nALSO decide 'wants_file': true when the user asks for the result to "
            "be WRITTEN TO A FILE or saved as a document/report/file on disk (e.g. "
            "'write a markdown file', 'save it as a file', 'create a .md document'); "
            "false when they only want an answer shown in the chat."
        )
        lines.append(
            "\nEmit STRICT JSON {\"shape\": <one of the names above or "
            f"'{ESCALATE}'>, \"rationale\": <one line>, "
            "\"search_allowed\": <true|false>, \"requested_specs\": <list of names "
            "or []>, \"wants_file\": <true|false>, \"unmet_specs\": <list of needed "
            "specialization names not available, or []>}."
        )
        return "\n".join(lines)

    async def select(self, goal: str) -> ShapeSelection:
        """Select a shape for ``goal`` (raises :class:`MalformedOutputError` on a
        non-enum result after the bounded repair loop).

        Builds the per-call schema with the harvested shape enum + ``escalate``,
        runs the d1 native structured call (``think=False`` top-level, ``temp 0``,
        raised ``num_predict``, ``format=<schema>``) through the ``llm_framework``
        chain with bounded JSON repair, and parses the enum decision into a
        :class:`ShapeSelection`. The blocking phi round-trip is offloaded off the
        event loop (the freeze-fix doctrine) and the call is traced under a
        ``planner.select_shape`` span like the planner's other judgment points."""
        if not goal or not str(goal).strip():
            raise MalformedOutputError("shape selection needs a non-empty goal")
        catalog = self.catalog()
        names = sorted(catalog)
        schema = build_selection_schema(names, self.spec_names)
        self.last_schema = schema
        legal = set(names) | {ESCALATE}
        legal_specs = set(self.spec_names)

        system = self._system_prompt(catalog)
        user = f"GOAL: {goal}\n\nReturn ONLY the JSON shape selection."
        opts = {**self._call_opts, "format": schema}
        chain = Chain()
        chain.use(prompt_assembly())
        chain.use(call_stage(self.transport, **opts))
        chain.use(
            structured_output(self.transport, max_repair_attempts=self.max_repair_attempts)
        )
        ctx = Context(system=system, user=user, transport=self.transport)
        tracer = get_tracer("agent_runtime.shape_selector")
        with tracer.start_as_current_span("planner.select_shape") as span:
            span.set_attribute("select.goal", str(goal)[:1000])
            span.set_attribute("select.catalog", names)
            # FREEZE FIX (decouple): the chain drives the SYNCHRONOUS blocking phi
            # round-trip; offload it off the single event loop and re-attach this
            # span inside the worker thread so the phi span nests under it (same
            # seam as Planner.plan / heal_decision).
            ctx = await run_blocking_in_span(chain.run, ctx)
            parsed = ctx.structured
            choice = (
                str(parsed.get("shape")).strip()
                if isinstance(parsed, Mapping) and parsed.get("shape") is not None
                else None
            )
            if choice not in legal:
                repair = ctx.meta.get("structured_output", {})
                raise MalformedOutputError(
                    "shape selection produced no legal shape "
                    f"(got {choice!r}; need one of {sorted(legal)}) after "
                    f"{self.max_repair_attempts} repair attempts: "
                    f"{repair.get('final_error')}"
                )
            rationale = (
                str(parsed.get("rationale", "")) if isinstance(parsed, Mapping) else ""
            )
            escalate = choice == ESCALATE
            # F5 SIGNALS — parsed LENIENTLY so a reply (or transport) that omits
            # them is the permissive/empty default (fail-open, no regression):
            #   * search_allowed: only an explicit boolean false disables the web;
            #     anything else (missing / non-bool) → True (search allowed).
            #   * requested_specs: kept only when a registered spec name (so an
            #     invented name can never reach binding), order-preserving + deduped.
            raw_search = (
                parsed.get("search_allowed") if isinstance(parsed, Mapping) else None
            )
            search_allowed = raw_search if isinstance(raw_search, bool) else True
            requested_specs: list[str] = []
            raw_specs = (
                parsed.get("requested_specs") if isinstance(parsed, Mapping) else None
            )
            if isinstance(raw_specs, (list, tuple)):
                for s in raw_specs:
                    name = str(s).strip()
                    if not name or name in requested_specs:
                        continue
                    if legal_specs and name not in legal_specs:
                        continue
                    requested_specs.append(name)
            # FILE-OUTPUT signal parsed LENIENTLY (fail-open): only an explicit
            # boolean true marks the request as wanting a file; anything else
            # (missing / non-bool) → False, identical to the pre-a4 behaviour.
            raw_wants_file = (
                parsed.get("wants_file") if isinstance(parsed, Mapping) else None
            )
            wants_file = raw_wants_file if isinstance(raw_wants_file, bool) else False
            # MISSING-SPECIALIST signal (a8) parsed LENIENTLY (fail-open): kept as
            # FREE strings (NOT filtered to registered names — the whole point is to
            # carry a spec the registry does NOT have), order-preserving + deduped.
            # A name that happens to be registered is NOT dropped here — the caller's
            # deterministic membership check is the single authority on "missing", so
            # parsing stays dumb. Anything non-list → [] (no notify).
            unmet_specs: list[str] = []
            raw_unmet = (
                parsed.get("unmet_specs") if isinstance(parsed, Mapping) else None
            )
            if isinstance(raw_unmet, (list, tuple)):
                for s in raw_unmet:
                    name = str(s).strip()
                    if name and name not in unmet_specs:
                        unmet_specs.append(name)
            selection = ShapeSelection(
                shape=(None if escalate else choice),
                escalate=escalate,
                rationale=rationale,
                raw=ctx.raw_output,
                search_allowed=search_allowed,
                requested_specs=requested_specs,
                wants_file=wants_file,
                unmet_specs=unmet_specs,
            )
            span.set_attribute("select.shape", choice)
            span.set_attribute("select.escalate", escalate)
            span.set_attribute("select.search_allowed", search_allowed)
            span.set_attribute("select.requested_specs", requested_specs)
            span.set_attribute("select.wants_file", wants_file)
            span.set_attribute("select.unmet_specs", unmet_specs)
            self.last_selection = selection
            return selection


__all__ = [
    "ESCALATE",
    "ShapeSelector",
    "ShapeSelection",
    "build_selection_schema",
]
