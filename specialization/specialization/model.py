"""The specialization data model: a RawDefinition (pre-compile) and a
CompiledSpec (post-compile), serialized as a markdown-with-frontmatter doc.

Design (d8 lifecycle + d10 context-scoping):

- A specialization is DEFINED in the UI (or proposed autonomously) as a
  :class:`RawDefinition` — just a name, a one-line description, and the intent
  ("what this specialist is for"). Nothing is compiled yet.
- It is COMPILED only on user approval into a :class:`CompiledSpec`: the
  research-distilled prompt/ruleset a sub-agent loads, plus the provenance
  (where it came from, the research trace, when it was compiled).

The on-disk form is a markdown doc with a YAML-ish frontmatter block, mirroring
the Claude-memory fact format used across this app (``memory.store``) so the two
are visually and structurally consistent::

    ---
    name: <short-kebab-slug>
    description: <one-line summary — the planner-facing lookup text>
    source: ui | autonomous
    research_trace_ref: <pointer to the research run that produced the body>
    created_at: <ISO-8601 timestamp>
    ---

    <body — the condensed prompt/ruleset the sub-agent loads>

House-style match (memory.store): the frontmatter parser is dependency-free and
uses NO regex — a small line scanner — so the d10 "registry has zero runtime
deps" promise holds. Splitting the frontmatter from the body is what lets the
registry serve the planner-facing INDEX (frontmatter only, no body) separately
from the sub-agent LOADER (full body) — the d10 context-scoping split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

# A specialization's origin (frontmatter ``source``): defined interactively in
# the UI, proposed by the agent's own autonomous loop (d8), or SEEDED directly
# into the registry as an already-authored output-shaping ruleset (``seed`` —
# the programmatic path that bypasses research + the HITL compile gate, used to
# stand up known rulesets like ``markdown-writer`` for the POC; the interactive
# chat-authoring surface is the later s4 step).
SOURCES = ("ui", "autonomous", "seed")

# The frontmatter keys an INDEX entry exposes — and ONLY these. The body is
# deliberately absent here so the planner-facing lookup can never carry a body
# (d10). Kept as a module constant so the registry's index path and any
# serializer agree on exactly one schema.
INDEX_FIELDS = ("name", "description", "source")


def utc_now_iso() -> str:
    """An ISO-8601 UTC timestamp for ``created_at`` (compile time)."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RawDefinition:
    """A specialization as first DEFINED (pre-compile, d8).

    The UI (or the autonomous loop) produces this: a name, a one-line
    description, and the free-text intent describing what the specialist is for.
    It carries NO compiled body — research + the HITL compile gate turn it into a
    :class:`CompiledSpec`."""

    name: str
    description: str
    intent: str

    def __post_init__(self) -> None:
        # Fail fast at the boundary — the name is the registry key.
        if not self.name.strip():
            raise ValueError("RawDefinition.name must be non-empty")


@dataclass(frozen=True)
class SpecIndexEntry:
    """The planner-facing lookup row: frontmatter ONLY, never a body (d10).

    This is exactly what :meth:`registry.SpecRegistry.index` returns per spec —
    the abstract-factory/lookup view the planner reasons over to pick a
    specialist, kept body-free so phi's small context window stays lean."""

    name: str
    description: str
    source: str

    def as_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "source": self.source}


# Workflow-spec schedule kinds (mirror reactive_tools.scheduler.JOB_KINDS without
# importing it — the model stays dependency-free, d10). A spec with a SCHEDULE is
# a "workflow spec" (a daily-brief-style spec the in-process scheduler fires); a
# spec WITHOUT one is an ordinary output-shaping ruleset.
SCHEDULE_KINDS = ("interval", "one_shot")

# Delivery channels a workflow spec can carry. ``email`` routes the produced
# report through the ``send_email`` tool (recipient defaults to self when None).
DELIVERY_CHANNELS = ("email",)


@dataclass(frozen=True)
class ScheduleSpec:
    """A workflow spec's optional SCHEDULE — when the in-process scheduler fires it.

    ``kind`` is ``interval`` (every ``interval_seconds`` while the app runs) or
    ``one_shot`` (fire once after ``initial_delay``). ``max_fires`` optionally
    bounds an interval (used by the safe self-test so it never loops forever).
    Maps onto a :class:`reactive_tools.scheduler.ScheduledJob` at fire time —
    this is just the serialisable description, no behavior."""

    kind: str = "interval"
    interval_seconds: float = 86400.0  # default daily; safe-test overrides small
    max_fires: int | None = None
    initial_delay: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in SCHEDULE_KINDS:
            raise ValueError(f"schedule kind {self.kind!r} not in {SCHEDULE_KINDS}")
        if self.kind == "interval" and self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0 for an interval schedule")


@dataclass(frozen=True)
class DeliverySpec:
    """A workflow spec's optional DELIVERY channel — where the report goes.

    ``channel`` is currently ``email``; ``recipient`` defaults to ``None`` which
    the email tool resolves to the user's OWN address (send-to-self) — exactly
    what the safe self-test relies on."""

    channel: str = "email"
    recipient: str | None = None

    def __post_init__(self) -> None:
        if self.channel not in DELIVERY_CHANNELS:
            raise ValueError(f"delivery channel {self.channel!r} not in {DELIVERY_CHANNELS}")


@dataclass(frozen=True)
class CompiledSpec:
    """A specialization COMPILED (post-approval, d8): the sub-agent's ruleset.

    Serialized to / parsed from a markdown-with-frontmatter doc. The ``body`` is
    an **OUTPUT-SHAPING RULESET** (d1) — the instructions a launched sub-agent
    applies to SHAPE the form of its answer to the real task, NOT a how-to about
    a skill. A ``markdown-writer`` body says *"structure the FINDINGS with a
    heading, bullet lists and a short summary; use GFM syntax"* — it does NOT say
    *"how to write markdown"*. The runtime injects this body as the SHAPING layer
    of the produce step (the system turn), over the real task content + tool
    findings carried in the user turn; describing the skill instead of doing the
    task was round-1's Iran->markdown-how-to bug (d1).

    The frontmatter carries the lookup text (name/description/source) plus
    provenance (research trace, compile time).

    WORKFLOW SPECS (s5): a spec MAY additionally carry an optional ``schedule``
    (:class:`ScheduleSpec`) and ``delivery`` (:class:`DeliverySpec`) — that is
    what makes it a "daily brief"-style workflow the in-process scheduler fires
    and delivers. Both default to ``None``; an ORDINARY spec has neither, so
    every existing spec round-trips byte-identically (back-compat). They are
    serialised as flat frontmatter keys (``schedule_*`` / ``delivery_*``) only
    when present, and the planner-facing :class:`SpecIndexEntry` does NOT expose
    them (the d10 lookup stays name/description/source)."""

    name: str
    description: str
    source: str
    body: str
    research_trace_ref: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    # Optional workflow-spec extensions (s5). None for an ordinary spec.
    schedule: "ScheduleSpec | None" = None
    delivery: "DeliverySpec | None" = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("CompiledSpec.name must be non-empty")
        if self.source not in SOURCES:
            raise ValueError(f"source {self.source!r} not in {SOURCES}")

    @property
    def is_workflow(self) -> bool:
        """True if this spec carries a schedule (a daily-brief-style workflow)."""
        return self.schedule is not None

    # ---- the d10 split, at the model level ---- #
    @property
    def index_entry(self) -> SpecIndexEntry:
        """The body-free lookup view (what the planner is allowed to see)."""
        return SpecIndexEntry(
            name=self.name, description=self.description, source=self.source
        )

    # ---- serialization (markdown-with-frontmatter) ---- #
    def to_markdown(self) -> str:
        """Serialize to the canonical markdown-with-frontmatter doc.

        The optional workflow-spec keys (``schedule_*`` / ``delivery_*``) are
        emitted ONLY when this spec carries them, so an ordinary spec serialises
        exactly as before (back-compat — no new keys appear)."""
        lines = [
            "---",
            f"name: {self.name}",
            f"description: {self.description}",
            f"source: {self.source}",
            f"research_trace_ref: {self.research_trace_ref}",
            f"created_at: {self.created_at}",
        ]
        if self.schedule is not None:
            lines.append(f"schedule_kind: {self.schedule.kind}")
            lines.append(f"schedule_interval_seconds: {self.schedule.interval_seconds}")
            if self.schedule.max_fires is not None:
                lines.append(f"schedule_max_fires: {self.schedule.max_fires}")
            if self.schedule.initial_delay is not None:
                lines.append(f"schedule_initial_delay: {self.schedule.initial_delay}")
        if self.delivery is not None:
            lines.append(f"delivery_channel: {self.delivery.channel}")
            if self.delivery.recipient is not None:
                lines.append(f"delivery_recipient: {self.delivery.recipient}")
        return "---\n" + "\n".join(lines[1:]) + "\n---\n\n" + f"{self.body.strip()}\n"


def parse_frontmatter_only(text: str) -> dict:
    """Parse JUST the frontmatter of a compiled-spec doc — the body is NOT read.

    Dependency-free, NO regex (house-style, memory.store): a flat line scanner
    over the single frontmatter block. This is the d10 lookup path — reading a
    doc's identity WITHOUT pulling its (potentially large) body into context, so
    the planner-facing index stays body-free by construction, not by trimming.
    """
    if not text.startswith("---"):
        raise ValueError("compiled-spec doc missing '---' frontmatter delimiter")
    # Take only the text up to the closing delimiter; never touch the body.
    _, fm, _body = text.split("---", 2)
    meta: dict[str, str] = {}
    for raw in fm.splitlines():
        if not raw.strip():
            continue
        key, sep, val = raw.partition(":")
        if not sep:
            continue
        meta[key.strip()] = val.strip()
    return meta


def _parse_schedule(meta: dict) -> "ScheduleSpec | None":
    """Reconstruct a :class:`ScheduleSpec` from flat frontmatter, or None.

    Back-compat: a doc with NO ``schedule_kind`` key yields ``None`` (an ordinary
    spec), so every pre-s5 doc parses to a schedule-less spec unchanged."""
    kind = meta.get("schedule_kind")
    if not kind:
        return None
    interval = meta.get("schedule_interval_seconds")
    max_fires = meta.get("schedule_max_fires")
    initial_delay = meta.get("schedule_initial_delay")
    return ScheduleSpec(
        kind=kind,
        interval_seconds=float(interval) if interval else 86400.0,
        max_fires=int(max_fires) if max_fires else None,
        initial_delay=float(initial_delay) if initial_delay else None,
    )


def _parse_delivery(meta: dict) -> "DeliverySpec | None":
    """Reconstruct a :class:`DeliverySpec` from flat frontmatter, or None."""
    channel = meta.get("delivery_channel")
    if not channel:
        return None
    recipient = meta.get("delivery_recipient")
    return DeliverySpec(channel=channel, recipient=recipient or None)


def parse_compiled_spec(text: str) -> CompiledSpec:
    """Parse a full compiled-spec doc (frontmatter + body) back to a CompiledSpec.

    Reads the optional workflow-spec keys when present; absent => ``None`` so a
    pre-s5 doc round-trips to a schedule-less spec (back-compat)."""
    meta = parse_frontmatter_only(text)
    _, _fm, body = text.split("---", 2)
    return CompiledSpec(
        name=meta.get("name", ""),
        description=meta.get("description", ""),
        source=meta.get("source", ""),
        research_trace_ref=meta.get("research_trace_ref", ""),
        created_at=meta.get("created_at", ""),
        body=body.strip(),
        schedule=_parse_schedule(meta),
        delivery=_parse_delivery(meta),
    )
