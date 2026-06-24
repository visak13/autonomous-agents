"""Node ROLE templates + the deep-research POSITION framings (d48 3-role collapse).

THREE roles, period (d48/d51 — the user's definitive architecture). At the NODE
level there are now only TWO node roles (the third, PLANNER, is the planning STAGE
— the shape selector + the incremental planner — not a per-node field):

* :data:`ROLE_WORKER` — a node that follows its 1+ specs (as combined guidelines)
  to hit its goal from a defined input → output. Behavior comes from the node's
  SPEC(s) + the task framing + reasoning, NOT a per-role code switch. A worker
  emits RAW free-text content (no ``format=<schema>`` wrapper — d50.1: content is
  RAW, never serialized).
* :data:`ROLE_SYNTHESIZER` — the terminal output stage: it writes the deliverable
  (to a file via the shared raw read-back loop, or to chat) — B1's domain, left
  intact. Also schema-less + RAW.

The 6-role enum that preceded this ({research, critic, worker, reviewer, synthesis,
verify}) is GONE: research/critic/reviewer/verify collapse into WORKER (their
distinct behavior now comes from a SPEC and/or the deep-research POSITION framing
below), and synthesis becomes SYNTHESIZER. There is no longer a per-role OUTPUT
SCHEMA or an enum-verdict judgment path — the role-execution switch (flag #5) and
the role-research fetch gate (flag #2) are retired (s9/c2, d48). Roles are NOT
LLM-extensible (Q-A: bounded) — :data:`~agent_runtime.factory.VALID_ROLES` is the
fixed two-element node-role vocabulary.

THE DEEP-RESEARCH POSITIONS. The deep-research SHAPE still runs ~10 rounds of
{research + critic} then a final {research + synthesis + verify}. Those per-round
*positions* are NOT node roles — they are a fixed, bounded POSITION vocabulary the
shape declares (:data:`~agent_runtime.shapes.VALID_POSITIONS`) and the unroll maps
onto worker/synthesizer NODES, injecting the matching :data:`POSITION_FRAMINGS`
text into the node's TASK (so the behavior is driven by PROMPTING, not a role
code-switch). A research-position node additionally carries the ``web_search`` tool
so it reads real sources via the generic search-then-read tool path (the deleted
role-research gate's job, now keyed on the TOOL, not a role).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from .factory import VALID_ROLES

# The TWO node-role constants (mirror VALID_ROLES) — use these instead of bare
# strings so a typo is a NameError, not a silent miss. PLANNER is a stage, not a
# node role, so it is intentionally NOT here.
ROLE_WORKER = "worker"
ROLE_SYNTHESIZER = "synthesizer"


# The CANONICAL "READ, DON'T DESCRIBE" guard (d13), hoisted to ONE constant and
# referenced by the research/synthesis/verify POSITION framings below AND by the
# runtime's fetched-source header. One idea, stated once.
READ_NOT_DESCRIBE = (
    "A search only lists candidate pages — that is NOT research. SELECT the most "
    "relevant real article URLs, FETCH and read them, and report the CONCRETE "
    "facts, figures, headlines and quotations found IN the article text. Never "
    "write 'site X has an article about Y' — that describes the source list, not a "
    "finding."
)


# --------------------------------------------------------------------------- #
# Node-role prompt TEMPLATES (worker + synthesizer ONLY)
# --------------------------------------------------------------------------- #
#
# Appended AFTER the shaping-framed spec body (the SAME composed ruleset for the
# node) and BEFORE the task. The framing tells the model WHAT posture to take; it
# never restates the spec or the task. A deep-research worker ALSO gets its
# position framing injected into the task (see POSITION_FRAMINGS) — the two
# compose: the generic worker posture here + the specific research/critic/verify
# behavior in the task.
ROLE_FRAMINGS: dict[str, str] = {
    ROLE_WORKER: (
        "ROLE: WORKER. Do the task directly and completely, following your "
        "specialization ruleset(s) as combined guidelines and using the inputs and "
        "any tool findings provided. Produce the deliverable itself — no "
        "meta-commentary, no preamble."
    ),
    ROLE_SYNTHESIZER: (
        "ROLE: SYNTHESIZER. You produce the FINAL DELIVERABLE itself — the actual "
        "answer the user receives. Integrate ALL upstream work (researched layers + "
        "critic notes, or the prior step's output) into ONE coherent, well-supported "
        "response to the original task, built from the concrete facts the layers took "
        "from their fetched sources, NOT from a description of which sources were "
        "consulted. PRESERVE AND CITE THE SOURCES: every upstream finding arrives "
        "attributed to the real URL it was read from — carry those source URLs/domains "
        "THROUGH into your deliverable (attribute the key facts and figures to the "
        "domain or URL they came from, and close with a Sources list of the URLs you "
        "actually used); never strip the sourcing into an unattributed summary. HONOR "
        "THE REQUESTED DEPTH: when the task asks for a detailed / thorough / in-depth "
        "report, write a SUBSTANTIVE one that covers every part the findings support — "
        "do not collapse rich researched material into a short paragraph. Resolve "
        "contradictions, keep only substantiated claims, and match the format/length "
        "the task asks for — no verdict, no meta-commentary, no "
        "'I will…' preamble. You write the deliverable to a FILE one bounded section "
        "per turn via the file_write tool (append each next section), READ THE FILE "
        "BACK with file_read to confirm the actual content, then call finish (the "
        "per-turn protocol + the file path are spelled out in the task). If the task "
        "is a simple conversational reply, write it in one file_write, read it back, "
        "and finish — directly and warmly."
    ),
}


# --------------------------------------------------------------------------- #
# DEEP-RESEARCH POSITION framings (prompting, NOT node roles)
# --------------------------------------------------------------------------- #
#
# Each deep-research round is a sequence of POSITIONS. The unroll maps every
# position onto a worker/synthesizer node and injects the matching framing below
# into that node's TASK text — so a "critic" node behaves like a critic because its
# TASK says so (prompting), not because a role enum routed it to different code.
# (These are the verbatim behavior texts that were the old per-role framings; they
# moved here unchanged so the deep-research quality is preserved.)
POSITION_RESEARCH = "research"
POSITION_CRITIC = "critic"
POSITION_SYNTHESIS = "synthesis"
POSITION_VERIFY = "verify"
POSITION_WORKER = "worker"

POSITION_FRAMINGS: dict[str, str] = {
    POSITION_RESEARCH: (
        "RESEARCH this layer: investigate the task at the CURRENT depth, building on "
        "every prior layer shown. " + READ_NOT_DESCRIBE + " Produce concrete "
        "FINDINGS that ADVANCE beyond the prior layers, the real SOURCES (the URLs "
        "you read), and the OPEN QUESTIONS that remain. Go deeper than the previous "
        "layer — do not merely restate it."
    ),
    POSITION_CRITIC: (
        "CRITIQUE the research layer just produced. FIRST: if it merely lists or "
        "describes sources instead of reporting concrete facts from the fetched "
        "text, it FAILED to read its sources — say so. Then name the unanswered "
        "GAPS, the WEAK or unsupported claims, and concrete FOLLOW-UP QUERIES (the "
        "specific pages the next round should fetch and read). Judge whether the "
        "research has CONVERGED or still NEEDS MORE depth. Be specific; do not "
        "rewrite the research."
    ),
    POSITION_SYNTHESIS: (
        "SYNTHESIZE the final deliverable from ALL the researched layers + critic "
        "notes above (see the SYNTHESIZER protocol). Carry the SOURCE URLs/domains "
        "from the findings THROUGH into the deliverable and cite them; honor the "
        "requested depth (a detailed report must be substantive, not a brief summary)."
    ),
    POSITION_VERIFY: (
        "VERIFY the synthesized answer against the researched layers for "
        "correctness, support and completeness. It must contain REAL fetched content "
        "(concrete facts, figures or quotations), NOT a list or description of which "
        "sources exist. Confirm claims trace to findings and flag anything "
        "unsupported, then restate the corrected, verified answer."
    ),
    POSITION_WORKER: (
        "Do this step directly and completely, using the inputs and any tool "
        "findings provided."
    ),
}


def position_framing(position: str) -> str:
    """The deep-research POSITION framing text injected into a node's task.

    Falls back to the generic worker framing for an unknown position so a future
    position never hard-fails the unroll (it just gets a plain worker task)."""
    return POSITION_FRAMINGS.get(position, POSITION_FRAMINGS[POSITION_WORKER])


# --------------------------------------------------------------------------- #
# Schemas / verdicts — RETIRED (d48). Kept as EMPTY tables + no-op helpers so the
# (few) importers still resolve while every role is now schema-less + RAW.
# --------------------------------------------------------------------------- #
#
# There is no per-role OUTPUT SCHEMA and no enum-verdict judgment path anymore:
# worker and synthesizer both emit RAW content (d50.1). ``ROLE_SCHEMAS`` is empty;
# both node roles are schema-less; there are no judgment roles, so the verdict
# tables are empty and the helpers are inert (return the permissive default).
ROLE_SCHEMAS: dict[str, dict[str, Any]] = {}

# BOTH node roles are schema-less by design (RAW emission). Exempt them so the
# import-time completeness guard stays meaningful (it now just asserts every node
# role has a framing).
SCHEMALESS_ROLES: frozenset[str] = frozenset({ROLE_WORKER, ROLE_SYNTHESIZER})

# No judgment roles remain (research/critic/verify/reviewer collapsed into worker,
# which emits RAW free-text). These stay defined + empty so verify.py / runtime.py
# imports resolve and their ``role in ROLE_VERDICTS`` checks are simply always
# False (the judgment path is dead, by design).
ROLE_VERDICTS: dict[str, tuple[str, ...]] = {}
JUDGMENT_ROLES: frozenset[str] = frozenset()

# Per-role OUTPUT BUDGET (num_predict) floor. With the judgment path retired these
# are a single generic floor; kept as named constants because the runtime + the
# deep-research opts still import/clear them.
ROLE_DEFAULT_NUM_PREDICT = 1200
JUDGMENT_NUM_PREDICT = 1600  # retained constant (no judgment role uses it now)
JUDGMENT_REPAIR_BUMP = 600   # retained constant (no verdict-repair loop now)


def is_judgment_role(role: Optional[str]) -> bool:
    """Always False — the enum-verdict judgment path is retired (d48)."""
    return role in JUDGMENT_ROLES


def role_num_predict_floor(role: Optional[str]) -> int:
    """The minimum ``num_predict`` for ``role`` (single generic floor now)."""
    return JUDGMENT_NUM_PREDICT if is_judgment_role(role) else ROLE_DEFAULT_NUM_PREDICT


def legal_verdict(role: Optional[str], parsed: Any) -> Optional[str]:
    """Always None — no role carries an enum verdict anymore (d48)."""
    legal = ROLE_VERDICTS.get(role or "", ())
    if not legal or not isinstance(parsed, Mapping):
        return None
    raw = parsed.get("verdict")
    if raw is None:
        return None
    verdict = str(raw).strip()
    return verdict if verdict in legal else None


# Fail-fast invariant: every declared NODE role has a framing. (Schemas are gone;
# both roles are schema-less, so only the framing guard remains meaningful.)
_missing_framing = VALID_ROLES - set(ROLE_FRAMINGS)
_missing_schema = VALID_ROLES - set(ROLE_SCHEMAS) - SCHEMALESS_ROLES
if _missing_framing or _missing_schema:  # pragma: no cover - structural guard
    raise RuntimeError(
        f"roles table incomplete: missing framing for {sorted(_missing_framing)}, "
        f"missing schema for {sorted(_missing_schema)}"
    )


def role_framing(role: str) -> str:
    """The node-role prompt template for ``role`` (raises ``KeyError`` if unknown)."""
    return ROLE_FRAMINGS[role]


def role_schema(role: str) -> Mapping[str, Any]:
    """RETIRED: no role carries an output schema (raises ``KeyError`` always)."""
    return ROLE_SCHEMAS[role]


__all__ = [
    "ROLE_WORKER",
    "ROLE_SYNTHESIZER",
    "ROLE_FRAMINGS",
    "READ_NOT_DESCRIBE",
    "POSITION_RESEARCH",
    "POSITION_CRITIC",
    "POSITION_SYNTHESIS",
    "POSITION_VERIFY",
    "POSITION_WORKER",
    "POSITION_FRAMINGS",
    "position_framing",
    "ROLE_SCHEMAS",
    "SCHEMALESS_ROLES",
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
