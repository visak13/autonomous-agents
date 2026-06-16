"""Node ROLE templates + per-role OUTPUT SCHEMAS (blueprint §2c).

In eda-base3 a node's ROLE is the spawn PROTOCOL: the pool sets ``EDP_ROLE`` and
that selects the skill prompt (``worker.md`` vs ``reviewer.md``) loaded ABOVE the
SAME compiled spec doc. There is no Claude-Code pool here, so the local-Gemma
port makes ``role`` an explicit :class:`~agent_runtime.factory.PlanNode` field and
realises the same mechanic with two pure-data tables, defined here:

* :data:`ROLE_FRAMINGS` — the role-prompt TEMPLATE (the local equivalent of the
  skill ``.md``): the SAME compiled spec doc is prepended for every role; only
  this task-framing differs. This is exactly what makes the deep-research shape
  run ONE specialization differentiated only by node role.
* :data:`ROLE_SCHEMAS` — the per-role OUTPUT SCHEMA (a JSON Schema with ``enum``
  on the verdict fields + ``required`` keys). Passed as Ollama's native
  ``format=<schema>`` so a Gemma judgment point emits structured JSON the runtime
  can read deterministically — the d1 ``think=false`` + ``enum``+``required`` +
  ``temp 0`` path. (Ollama enforces the schema's SYNTAX/shape; the
  ``llm_framework`` structured-output stage still does a bounded parse/repair.)

Both tables are pure data (no model call, no I/O) so they stay trivially testable
and import-light. The role NAMES are the single source of truth in
:data:`~agent_runtime.factory.VALID_ROLES`; every role there has an entry here.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from .factory import VALID_ROLES

# Role name constants (mirror VALID_ROLES) — use these instead of bare strings so
# a typo is a NameError, not a silent miss.
ROLE_RESEARCH = "research"
ROLE_CRITIC = "critic"
ROLE_WORKER = "worker"
ROLE_REVIEWER = "reviewer"
ROLE_SYNTHESIS = "synthesis"
ROLE_VERIFY = "verify"


# --------------------------------------------------------------------------- #
# Role-prompt TEMPLATES (the local equivalent of eda-base3's skill .md files)
# --------------------------------------------------------------------------- #
#
# Each framing is appended AFTER the shaping-framed spec body (the SAME compiled
# doc for every role) and BEFORE the task — so the only thing that varies node to
# node is this paragraph + the output schema. The framing tells the model WHAT
# posture to take; it never restates the spec or the task.
ROLE_FRAMINGS: dict[str, str] = {
    ROLE_RESEARCH: (
        "ROLE: RESEARCH. Investigate the task at the CURRENT depth, building on "
        "every prior researched layer you are shown. "
        # d13 (READ, DON'T DESCRIBE): a research layer must report the ACTUAL "
        # content of the sources, not a list/description of which sources exist.
        "You MUST READ the actual source CONTENT, not describe the search "
        "results. A search only gives you a list of candidate pages — that is "
        "NOT research. SELECT the most relevant real article URLs from the "
        "search results, FETCH and read each one, and report the CONCRETE facts, "
        "figures, headlines and quotations you found IN those articles. NEVER "
        "write 'Wikipedia has an article about X' or 'Reuters covers Y' — that "
        "is a description of the source list, not a finding. A finding is a "
        "specific fact taken from the fetched article text. Produce concrete "
        "FINDINGS (specific, non-redundant facts that ADVANCE beyond the prior "
        "layers), the real SOURCES (the article URLs you actually read) they "
        "came from, and the OPEN QUESTIONS that remain. Go deeper than the "
        "previous layer — do not merely restate it."
    ),
    ROLE_CRITIC: (
        "ROLE: CRITIC. Adversarially review the research layer just produced "
        "(shown to you). "
        # d13: the critic's primary job is to REJECT meta-summaries and force a "
        # genuine fetch+read so each round accumulates real fetched substance.
        "FIRST, reject any 'research' that merely DESCRIBES or LISTS sources "
        "instead of reporting their actual content: if the findings say things "
        "like 'site X has information about Y' or just name/summarize the search "
        "results without concrete facts, figures or quotations taken from the "
        "fetched article text, that layer FAILED to read its sources — call it "
        "out explicitly and set the verdict to NEEDS_MORE. Then identify GAPS "
        "still unanswered, WEAK or UNSUPPORTED CLAIMS, and concrete FOLLOW-UP "
        "QUERIES the next research round should pursue (name the specific pages "
        "it should actually fetch and read). Judge whether the body of research "
        "has CONVERGED (concrete, well-sourced, read-not-described) or still "
        "NEEDS MORE depth. Be specific; do not rewrite the research."
    ),
    ROLE_SYNTHESIS: (
        "ROLE: SYNTHESIS. You are given ALL researched layers and their critic "
        "notes. Integrate them into ONE coherent, well-supported answer to the "
        "original task. "
        # d13: the synthesized answer must be built from real fetched content.
        "Build the answer from the CONCRETE facts, figures and quotations the "
        "research layers actually took from their fetched sources — NOT from a "
        "description of which sources were consulted. Resolve contradictions, "
        "keep only substantiated claims, and state the synthesized result "
        "plainly. Report your confidence as the verdict and list the key "
        "findings + anything you fixed inline."
    ),
    ROLE_VERIFY: (
        "ROLE: VERIFY. Independently check the synthesized answer against the "
        "researched layers for correctness, support, and completeness. "
        # d13: verification explicitly fails a source-list/meta answer.
        "Confirm the answer contains REAL fetched content — concrete facts, "
        "figures or quotations drawn from actual sources — and is NOT a list or "
        "description of which sources exist; an answer that only names/describes "
        "sources without their substance FAILS. Confirm claims trace to "
        "findings, flag anything unsupported, and return a pass/concerns/fail "
        "verdict with the specific findings behind it."
    ),
    ROLE_REVIEWER: (
        "ROLE: REVIEWER. Review the prior output for correctness against the "
        "task. Where it is wrong or incomplete, apply the MINIMAL inline fix and "
        "record it; otherwise pass it. Return a pass/concerns/fail verdict, the "
        "findings behind it, and any fixes you applied inline."
    ),
    ROLE_WORKER: (
        "ROLE: WORKER. Do the task directly and completely, using the inputs and "
        "any tool findings provided. Produce the deliverable itself — no "
        "meta-commentary."
    ),
}


# --------------------------------------------------------------------------- #
# Per-role OUTPUT SCHEMAS (JSON Schema; enum on verdicts + required keys)
# --------------------------------------------------------------------------- #
#
# These are passed verbatim as Ollama native ``format=<schema>`` (d1). A judgment
# role (critic/verify/synthesis/reviewer) carries an ``enum`` verdict + required
# keys so the loop can read a deterministic decision; research carries the
# findings/sources/open_questions triple per §2c.
def _arr_of_strings(desc: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "description": desc}


_RESEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": _arr_of_strings("concrete facts found at this depth"),
        "sources": _arr_of_strings("where each finding came from (urls/titles)"),
        "open_questions": _arr_of_strings("what remains unanswered"),
    },
    "required": ["findings", "sources", "open_questions"],
}

_CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "gaps": _arr_of_strings("unanswered gaps in the research so far"),
        "weak_claims": _arr_of_strings("claims that are unsupported or shaky"),
        "follow_up_queries": _arr_of_strings(
            "specific queries the next research round should pursue"
        ),
        "verdict": {
            "type": "string",
            "enum": ["converged", "needs_more"],
            "description": "whether the research has converged or needs more depth",
        },
    },
    "required": ["gaps", "weak_claims", "follow_up_queries", "verdict"],
}

# synthesis / verify / reviewer share the pass/concerns/fail verdict shape (§2c).
_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["pass", "concerns", "fail"],
            "description": "overall judgement of the output",
        },
        "findings": _arr_of_strings("the findings behind the verdict"),
        "fixed_inline": _arr_of_strings(
            "corrections applied inline to the output (empty if none)"
        ),
    },
    "required": ["verdict", "findings", "fixed_inline"],
}

# A plain worker produces a single free-text deliverable (no enum gate).
_WORKER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "output": {"type": "string", "description": "the deliverable itself"},
    },
    "required": ["output"],
}

ROLE_SCHEMAS: dict[str, dict[str, Any]] = {
    ROLE_RESEARCH: _RESEARCH_SCHEMA,
    ROLE_CRITIC: _CRITIC_SCHEMA,
    ROLE_SYNTHESIS: _REVIEW_SCHEMA,
    ROLE_VERIFY: _REVIEW_SCHEMA,
    ROLE_REVIEWER: _REVIEW_SCHEMA,
    ROLE_WORKER: _WORKER_SCHEMA,
}

# The verdict enum values per judgment role — exposed so callers (and the a2
# trace) can assert a returned verdict is legal without re-deriving the schema.
ROLE_VERDICTS: dict[str, tuple[str, ...]] = {
    ROLE_CRITIC: ("converged", "needs_more"),
    ROLE_SYNTHESIS: ("pass", "concerns", "fail"),
    ROLE_VERIFY: ("pass", "concerns", "fail"),
    ROLE_REVIEWER: ("pass", "concerns", "fail"),
}

# The roles that emit an enum ``verdict`` the runtime validates + repairs.
JUDGMENT_ROLES: frozenset[str] = frozenset(ROLE_VERDICTS)

# ------------------------------------------------------------------------- #
# Per-role OUTPUT BUDGET (num_predict) — GENERIC role-execution hardening
# (formerly baked into the deleted DeepResearchExecutor; now a property of the
# role itself so ANY runtime executing a role-tagged node gets it, a3 re-arch).
# ------------------------------------------------------------------------- #
#
# A role node emits structured JSON under a ``required`` schema. ``think`` is OFF
# (d1) so the whole budget goes to the JSON answer, not a CoT trace. A research
# layer's findings must not truncate, so the default floor is raised over the
# model's 1024 default. A JUDGMENT role additionally carries an enum ``verdict``:
# a VERBOSE small model spends the budget on the findings prose and OVERRUNS,
# truncating the JSON so the verdict key is dropped (parse-fail → null verdict).
# The runtime must never silently pass a null verdict, so judgment roles get a
# HIGHER floor (the whole verdict object fits) and, on each verdict-repair
# attempt, an even larger budget. These are FLOORS — an explicit larger
# ``num_predict`` in the call opts still wins.
ROLE_DEFAULT_NUM_PREDICT = 1200
JUDGMENT_NUM_PREDICT = 1600
JUDGMENT_REPAIR_BUMP = 600  # added per verdict-repair attempt


def is_judgment_role(role: Optional[str]) -> bool:
    """True for a role that emits an enum verdict the runtime validates."""
    return role in JUDGMENT_ROLES


def role_num_predict_floor(role: Optional[str]) -> int:
    """The minimum ``num_predict`` for ``role`` (judgment roles get the raised floor)."""
    return JUDGMENT_NUM_PREDICT if is_judgment_role(role) else ROLE_DEFAULT_NUM_PREDICT


def legal_verdict(role: Optional[str], parsed: Any) -> Optional[str]:
    """The LEGAL enum verdict in ``parsed`` for ``role``, else ``None``.

    Returns the verdict string ONLY when it is one of the role's legal enum
    values (:data:`ROLE_VERDICTS`); a missing, null, or out-of-enum verdict — or
    a non-mapping parse — returns ``None`` so the caller repairs it. Non-judgment
    roles always return ``None`` (they carry no verdict to validate)."""
    legal = ROLE_VERDICTS.get(role or "", ())
    if not legal or not isinstance(parsed, Mapping):
        return None
    raw = parsed.get("verdict")
    if raw is None:
        return None
    verdict = str(raw).strip()
    return verdict if verdict in legal else None


# Fail-fast invariant: every declared role has BOTH a framing and a schema. This
# runs at import so a future role added to VALID_ROLES without its tables here is
# caught immediately, not at the first run that selects it.
_missing_framing = VALID_ROLES - set(ROLE_FRAMINGS)
_missing_schema = VALID_ROLES - set(ROLE_SCHEMAS)
if _missing_framing or _missing_schema:  # pragma: no cover - structural guard
    raise RuntimeError(
        f"roles table incomplete: missing framing for {sorted(_missing_framing)}, "
        f"missing schema for {sorted(_missing_schema)}"
    )


def role_framing(role: str) -> str:
    """The role-prompt template for ``role`` (raises ``KeyError`` if unknown)."""
    return ROLE_FRAMINGS[role]


def role_schema(role: str) -> Mapping[str, Any]:
    """The per-role output JSON Schema for ``role`` (raises ``KeyError``)."""
    return ROLE_SCHEMAS[role]


__all__ = [
    "ROLE_RESEARCH",
    "ROLE_CRITIC",
    "ROLE_WORKER",
    "ROLE_REVIEWER",
    "ROLE_SYNTHESIS",
    "ROLE_VERIFY",
    "ROLE_FRAMINGS",
    "ROLE_SCHEMAS",
    "ROLE_VERDICTS",
    "JUDGMENT_ROLES",
    "ROLE_DEFAULT_NUM_PREDICT",
    "JUDGMENT_NUM_PREDICT",
    "JUDGMENT_REPAIR_BUMP",
    "is_judgment_role",
    "role_num_predict_floor",
    "legal_verdict",
    "role_framing",
    "role_schema",
]
