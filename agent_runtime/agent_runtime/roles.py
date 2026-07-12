"""Node ROLE templates + the deep-research POSITION framings (d213/d215 node types).

ROLE = node type (d213). There are FIVE, sitting in three places (d215):

* PLANNER — the planning STAGE (shape selector + incremental planner). It drives
  the iterative loop; NOT a per-node field, so it has no framing here.
* In-plan node roles — the roles the planner may place inside a plan via add_step:
  * :data:`ROLE_RESEARCHER` — RETIRED from the engine as a routing/dispatch discriminator
    (SB-RR, d292/d293): research is a SELF-SELECTED specialization, not a role. The symbol +
    its framing are KEPT only for BACK-COMPAT (a stale-prompted planner authoring this role
    degrades gracefully — a tool-less node routes through the SAME unified worker loop and
    gathers via its self-selected bundle). New gather nodes are WORKERS carrying the
    research-methodology spec (the self-select lever); no engine branch keys on this role.
  * :data:`ROLE_WORKER` — a node that follows its 1+ specs (as combined guidelines)
    to hit its goal from a defined input → output (e.g. a write worker authoring a
    section). Behavior comes from the node's SPEC(s) + task framing + reasoning.
  * :data:`ROLE_REVIEWER` — the DEFAULT LAST STEP of every plan: inspects+fixes the
    deliverable in place and emits the plan's FINAL STATUS (which the planner reads to
    decide the next plan).
* :data:`ROLE_SYNTHESIZER` — the TERMINAL stage: a single framework-built node that
  runs ONCE after the planner loop exits to deliver the result (file via the shared
  raw read-back loop, or chat). NOT add_step'd by the planner.

ALL FIVE emit RAW free-text content (no ``format=<schema>`` wrapper — d50.1: content
is RAW, never serialized). The 6-role enum that preceded this ({research, critic,
worker, reviewer, synthesis, verify}) is GONE: critic/verify remain deep-research
POSITIONS (prompting, below), and research/reviewer/synthesis became first-class node
types again (d213). There is no per-role OUTPUT SCHEMA or enum-verdict judgment path —
the role-execution switch (flag #5) and the role-research fetch gate (flag #2) stay
retired (s9/c2, d48). Roles are NOT LLM-extensible (Q-A: bounded) —
:data:`~agent_runtime.factory.VALID_ROLES` is the fixed node-role vocabulary
(researcher/worker/reviewer/synthesizer; planner is the stage, not a node role).

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

# The node-role constants (mirror VALID_ROLES) — use these instead of bare strings
# so a typo is a NameError, not a silent miss. PLANNER is the planning STAGE, not a
# per-node role, but its name is kept here so the planning stage can look up its bundle
# set the same way (it is intentionally NOT in VALID_ROLES). d213/d215.
ROLE_PLANNER = "planner"
ROLE_RESEARCHER = "researcher"
ROLE_WORKER = "worker"
ROLE_REVIEWER = "reviewer"
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
# Node-role prompt TEMPLATES — one per node type (researcher / worker / reviewer /
# synthesizer; planner is the STAGE, no per-node framing). d213/d215.
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
    ROLE_RESEARCHER: (
        "ROLE: RESEARCHER. You GATHER grounded evidence to answer your assigned "
        "concern, using the search / fetch / note tools your bundle gives you. "
        "Work the canonical loop: DECOMPOSE the concern, SEARCH it, READ the most "
        "relevant chunks of the real sources you fetch, and — your PRIMARY act after "
        "each read — take a NOTE that records what you learned AND the GAPS it left. "
        + READ_NOT_DESCRIBE + " Cover the breadth of the concern before drilling one "
        "part; expand a sub-concern only by actually running another search/read/note "
        "round, and STOP once every concern is settled-in-a-note or collapsed. Report "
        "concrete FINDINGS attributed to the real [S#]/URL you read them from, plus the "
        "OPEN QUESTIONS that remain — never write from memory and never cite a page you "
        "did not fetch."
    ),
    ROLE_REVIEWER: (
        "ROLE: REVIEWER — the DEFAULT LAST STEP of the plan. The deliverable is "
        "ALREADY produced by the upstream nodes; do NOT re-emit it. Inspect it by "
        "BOUNDED REGION (file_read a slice/tail — never the whole file at once) and "
        "GROUND your checks in the sources you can pull on demand (load_source against "
        "the SOURCE INDEX). Where you find a gap, an unsupported claim, a citation that "
        "does not resolve to a real [S#], or a coherence/structure defect, FIX it IN "
        "PLACE with a single anchored file_update (ground-or-remove; never fabricate a "
        "source). Make only targeted edits — never rewrite the whole document. THEN emit "
        "the plan's FINAL STATUS: a short, honest summary of what the plan accomplished, "
        "whether it met its goal, and — when the goal still needs a further plan — WHAT "
        "the next plan should produce (kind + shape, derived from the work just done, "
        "e.g. the desired sectioned output a write plan should author). Reply with that "
        "status as plain prose — never a verdict or findings object."
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
# BUNDLE NAME CONSTANTS (d212/d213). A bundle is a TOOL WRAPPER, NOT a role (d212):
# it carries one CAPABILITY DOMAIN's tools + doctrine. These mirror
# ``agent_runtime.bundles``'s BUNDLE_* constants and stay here as plain strings so this
# module is import-cycle-free.
#
# NODE-SELF-SELECT (d221): there is NO hardcoded role/position/tool -> bundle TABLE
# anymore. The planner sets ONLY a node's ROLE + SPECIALIZATION (never bundles, d194);
# each in-plan node SELF-SELECTS the bundle(s) its task needs AT RUNTIME by reasoning
# over the advertised ``get_bundles`` catalog and loading them (the runtime's
# ``get_bundles`` tool + the per-node ``_loaded_bundles`` set). The base ``object``
# bundle (finish + the universal loop) is the ONLY always-on floor. The retired
# ``ROLE_BUNDLES`` / ``POSITION_BUNDLES`` / ``_TOOL_BUNDLES`` tables + the deterministic
# ``bundles_for_node`` / ``bundles_for_position`` assignment are GONE — selection is
# REASONED (d14/d60/d65/d190), and selection reliability on the small model is a
# tool-DESCRIPTION lever (d186), never a reason to hardcode.
# --------------------------------------------------------------------------- #
BUNDLE_OBJECT = "object"
BUNDLE_PLANNING = "planning"
BUNDLE_RESEARCH = "research"
BUNDLE_RESEARCH_READ = "research_read"
BUNDLE_FILE = "file"

# (ROLE_PLANNER / ROLE_RESEARCHER / ROLE_WORKER / ROLE_REVIEWER / ROLE_SYNTHESIZER are
# defined once near the top of the module, before ROLE_FRAMINGS, so the framings table
# can reference them.)


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

# ALL node roles are schema-less by design (RAW emission, d48/d213). Exempt them so
# the import-time completeness guard stays meaningful (it now just asserts every node
# role has a framing). Researcher + reviewer joined the in-plan vocabulary (d213/d215);
# they emit RAW content too (findings prose / corrected content + a plain-prose status).
SCHEMALESS_ROLES: frozenset[str] = frozenset(
    {ROLE_WORKER, ROLE_SYNTHESIZER, ROLE_RESEARCHER, ROLE_REVIEWER}
)

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
    "BUNDLE_OBJECT",
    "BUNDLE_PLANNING",
    "BUNDLE_RESEARCH",
    "BUNDLE_RESEARCH_READ",
    "BUNDLE_FILE",
    "ROLE_PLANNER",
    "ROLE_RESEARCHER",
    "ROLE_REVIEWER",
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
