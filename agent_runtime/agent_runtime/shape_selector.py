"""Per-query SHAPE SELECTION via a native structured Gemma call (blueprint §2a, d1).

Shape selection is the planner's ONE genuinely model-driven choice before the DAG
is authored: given a user GOAL, WHICH plan shape fits it — a strictly sequential
``linear`` chain, a ``modular-parallel`` fan-out, the bounded cyclic
``deep-research`` shape, or any other text-file shape on disk? Everything that
FOLLOWS the choice (the readiness gate, the dispatch FSM) is deterministic and
lives in :mod:`agent_runtime.scheduler`; the choice itself is a Gemma judgment
point, so it uses the proven d1 native structured path:

* the choice is the model's REASONED one (s9/c2, d46/d50.1): the selection is a
  REASONED field, so it does NOT ride ``format=<schema>`` — a single constrained
  enum sample lets the small model emit a *legal-but-different* value than its CoT
  concluded (the s9 RCA reason≠emit gap: thinking says ``linear``, the constrained
  emission samples ``modular-parallel``). Instead the JSON is elicited by the PROMPT
  and the emission is ANCHORED to the reasoning (the d39 remedy already proven on the
  DAG planner + the shape author): ``think=True`` so gemma reasons first, then it
  emits STRICT JSON whose values MUST equal its concluded choice, parsed + repaired by
  the ``llm_framework`` ``structured_output`` stage (fence-strip + balanced-JSON
  safety-net), and VALIDATED against the legal shape set with a bounded repair loop.
  The ``shape`` value may still ONLY be a real shape name or the reserved ``escalate``
  low-confidence signal — but that is enforced by a post-parse membership check, NOT a
  wire enum, so the reasoned value survives;
* ``temperature=0`` (deterministic), a raised ``num_predict`` (4096, so the CoT
  cannot starve the JSON decision to EMPTY);
* driven through the existing ``llm_framework`` chain (``call_stage`` + bounded
  ``structured_output`` repair), with the blocking phi round-trip offloaded off the
  event loop and traced like the planner's other judgment calls.

The legal shape set (and the per-call schema, kept only as the documented contract /
proof artifact in ``last_schema``) is rebuilt from the on-disk catalog at call time,
so adding a shape file (or the s4 UI adding one) makes it selectable with NO code
change here — the growable-shapes requirement, mirrored from the growable tool
registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from llm_framework import Chain, Context, Transport
from llm_framework.stages import call_stage, prompt_assembly, structured_output

from .identity import with_identity
from .selfheal import MalformedOutputError
from .shapes import ShapeSpec, load_shapes
from .tracing import get_tracer, run_blocking_in_span

# The reserved enum value the model picks when no shape confidently fits. It is
# NOT a shape name (a shape file may never be named this) — it is the low-
# confidence ESCALATION signal (blueprint §2a). The caller routes it to a human /
# a default rather than dispatching a mis-selected shape.
ESCALATE = "escalate"

# Native call options. s1/b1 REASONING ROLLOUT: ``think=True`` so gemma4 reasons
# about which shape fits (CoT in the SEPARATE message.thinking field) before emitting
# the JSON selection. s9/c2 (d46/d50.1): the selection is REASONED, so NO
# ``format=<schema>`` is added — the JSON is prompt-elicited and parse/repaired, then
# validated against the legal set, so the emitted value stays faithful to the CoT
# (a single constrained enum sample let the model emit a legal-but-different value
# than it reasoned — the s9 RCA reason≠emit gap). ``num_predict`` raised 256->4096
# (a2-proven load-bearing: thinking tokens compete with content, and at <=512 the
# content truncates to EMPTY). temp 0 deterministic.
_SELECT_OPTS: dict[str, Any] = {
    "api": "native",
    "think": True,
    "temperature": 0,
    "num_predict": 4096,
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
    # PLAN-CHAINING signal (c1b/d49.4): the model's read of whether the requested
    # file output is LARGE / multi-page (a multi-page report, a multi-section / "deep"
    # document) — so the router can route it through plan-chaining (research → a
    # write-file shape whose per-page nodes fill the file) instead of one writer node.
    # Fail-open: only an explicit true sets it (else False = the single-file path).
    multi_page: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "escalate": self.escalate,
            "rationale": self.rationale,
            "search_allowed": self.search_allowed,
            "requested_specs": list(self.requested_specs),
            "wants_file": self.wants_file,
            "unmet_specs": list(self.unmet_specs),
            "multi_page": self.multi_page,
        }


def build_selection_schema(
    shape_names: list[str], spec_names: Optional[list[str]] = None
) -> dict[str, Any]:
    """The per-call OUTPUT SCHEMA — kept as the documented CONTRACT / proof artifact.

    s9/c2 (d46/d50.1): this schema is NO LONGER passed to the wire as
    ``format=<schema>`` (the reasoned ``shape``/``requested_specs`` fields must stay
    faithful to the CoT — a constrained enum sample lets the model emit a
    legal-but-different value than it reasoned). It is now built only so the caller
    can (a) advertise the field contract in the PROMPT and (b) derive the legal shape
    set for the post-parse membership check + record ``last_schema`` as a proof
    artifact. The ``shape`` enum is the harvested catalog of shape names PLUS
    :data:`ESCALATE`; the post-parse check enforces that set, not the wire.

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
            "multi_page": {
                "type": "boolean",
                "description": (
                    "true when the user asks for a LARGE / MULTI-PAGE / multi-section "
                    "file — a multi-page report, a long/'detailed'/'in-depth' document "
                    "with several sections, a multi-page site/doc; false for a short or "
                    "single-section file, or a chat-only answer."
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
            "\nChoose by the WORK the request needs, NOT its phrasing (SELECTION "
            "GUIDELINES, docs/SELECTION_GUIDELINES.md): a question, a "
            "'describe…' and an imperative asking for the SAME result route to the "
            "SAME shape. Match the shape's weight to the work's — one straight pass "
            "= linear; independent parts to gather then combine = a parallel shape; "
            "an exhaustive multi-round survey of ONE topic = a deep-research shape. "
            "Do not over-escalate a simple informational request to a heavy "
            "multi-round shape just because it is phrased as a question."
        )
        # F5 SIGNAL 1 — no-search constraint. The model JUDGES whether the request
        # forbids the web (intent, across any phrasing), not a keyword match.
        lines.append(
            "\nDecide 'search_allowed': true normally, FALSE only when the user "
            "explicitly forbids the web (e.g. 'do not search', 'from your own "
            "knowledge'). When false, prefer a non-search shape, never a "
            "web-research shape."
        )
        # F5 SIGNAL 2 — user-named specialization(s). Advertise the catalog so the
        # model can recognise a name the user actually requested.
        if self.spec_names:
            lines.append("\nAVAILABLE SPECIALIZATIONS (the user may name one):")
            for s in sorted(self.spec_names):
                lines.append(f"  - {s}")
            lines.append(
                "Set 'requested_specs' to the listed specialization name(s) the user "
                "EXPLICITLY asked for by name; [] when none. Do not guess."
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
                "output-style NOT in the list above (e.g. a 'forensic-accountant "
                "report'), set 'unmet_specs' to that capability in the user's own "
                "terms; [] when none or all requested specs ARE listed. Do not "
                "invent a need."
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
            "\nDecide 'wants_file': true when the user asks for the result WRITTEN "
            "TO A FILE / saved as a document on disk (e.g. 'write a markdown file', "
            "'save it as a .md'); false when they only want an answer in the chat."
        )
        # PLAN-CHAINING signal (c1b). The model JUDGES whether the file is LARGE /
        # multi-page so a big document is built across chained plans (research → a
        # write-file shape filling it page by page), not crammed into one writer.
        lines.append(
            "\nDecide 'multi_page': true when the file is LARGE / multi-page / "
            "multi-section (a multi-page or 'detailed'/'in-depth' report, a doc with "
            "several sections); false for a short or single-section file."
        )
        lines.append(
            "\nEmit STRICT JSON {\"shape\": <one of the names above or "
            f"'{ESCALATE}'>, \"rationale\": <one line>, "
            "\"search_allowed\": <true|false>, \"requested_specs\": <list of names "
            "or []>, \"wants_file\": <true|false>, \"multi_page\": <true|false>, "
            "\"unmet_specs\": <list of needed specialization names not available, or []>}."
        )
        # ANCHOR EMISSION TO REASONING (s9/c2, d39/d50.1): the selection is now
        # prompt-JSON (no wire enum), so the ONE failure to guard is reason≠emit —
        # the model reasoning to one shape then emitting another legal value. Make the
        # JSON the faithful transcription of the conclusion, not a fresh guess.
        lines.append(
            "Your JSON values MUST be the SAME choice your reasoning concluded above: "
            "if you reasoned the shape is 'linear', emit \"shape\": \"linear\" — never "
            "a heavier shape you considered and rejected. Do not over-escalate a simple "
            "request. The JSON is the transcript of your decision, not a new one."
        )
        # The universal identity (prepended below) already requires a JSON-only
        # visible reply, so no per-prompt "reason privately / no fences" tail.
        return with_identity("\n".join(lines))

    async def select(self, goal: str) -> ShapeSelection:
        """Select a shape for ``goal`` (raises :class:`MalformedOutputError` on a
        non-enum result after the bounded repair loop).

        Builds the per-call schema with the harvested shape enum + ``escalate``,
        runs the d1 native structured call (s1/b1: ``think=True`` top-level, ``temp
        0``, raised ``num_predict``, ``format=<schema>``) through the ``llm_framework``
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
        # s9/c2 (d46/d50.1): NO ``format=schema`` on the wire — the reasoned shape /
        # requested_specs must stay faithful to the CoT (a constrained enum sample
        # emits a legal-but-different value than the model reasoned). The JSON is
        # prompt-elicited, parse/repaired by ``structured_output``, then VALIDATED
        # against ``legal`` below (membership check, not a wire enum). ``last_schema``
        # keeps the schema as the documented contract / proof artifact.
        opts = dict(self._call_opts)
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
            # PLAN-CHAINING signal (c1b) parsed LENIENTLY (fail-open): only an explicit
            # boolean true marks the request as multi-page; anything else → False (the
            # single-file path), so a reply omitting it is byte-identical to pre-c1b.
            raw_multi_page = (
                parsed.get("multi_page") if isinstance(parsed, Mapping) else None
            )
            multi_page = raw_multi_page if isinstance(raw_multi_page, bool) else False
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
                multi_page=multi_page,
            )
            span.set_attribute("select.shape", choice)
            span.set_attribute("select.escalate", escalate)
            span.set_attribute("select.search_allowed", search_allowed)
            span.set_attribute("select.requested_specs", requested_specs)
            span.set_attribute("select.wants_file", wants_file)
            span.set_attribute("select.multi_page", multi_page)
            span.set_attribute("select.unmet_specs", unmet_specs)
            self.last_selection = selection
            return selection


__all__ = [
    "ESCALATE",
    "ShapeSelector",
    "ShapeSelection",
    "build_selection_schema",
]
