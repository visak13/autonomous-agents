"""bundles.research — the ResearchBundle (d190/d191 + d212 capability-domain redraw).

The CAPABILITY DOMAIN for GATHERING grounded evidence: search the web, READ the most
relevant chunks of a source, NOTE what was learned + the gaps, EXPAND / PRUNE / STOP the
research tree, and CROSS-VERIFY a claim against the sources actually pulled. It is a TOOL
WRAPPER, NOT a role (d212): it never "does the research" — it carries the gather tools +
the gather doctrine, and the researcher/critic/verify NODE TYPES (d213) load it (composed
with ``research_read`` so they can also pull a fetched source's verbatim text).

CAPABILITY-DOMAIN REDRAW (d212 #3): the READ capability (``load_source``) used to be baked
here AND in the writer bundle; it now lives in its own ``research_read`` bundle so a
writer can reuse it without inheriting the gather tools. This bundle keeps only the GATHER
domain (search / fetch / note / tree-decision / cross-verify).

It ORCHESTRATES the existing functions — :func:`agent_runtime.research_tree.make_tool_spec` /
:data:`~agent_runtime.research_tree.TREE_TOOL_SPECS` and
:func:`agent_runtime.claim_verify.verify_claims` — it reimplements nothing.

DOCTRINE (d191): the research bundle carries the *template-then-grow-out* flavor of COMPLEX
MEMORY — start from a decompose-first template and GROW OUT (expand the concerns the report
still needs). This flavor is SELF-CONTAINED here: a request only gets the research bias if
it LOADS this bundle, so a simple ask (which loads the base / file bundle) never sees the
template/grow treatment (d188 fix).

The exact text the runtime's research ReAct loop injects (:data:`RESEARCH_LOOP_INSTRUCTION`)
and the exact gather tool schemas live HERE as the single source of truth; ``runtime.py``
imports/delegates to them so behaviour is byte-identical while the bundle owns the doctrine.

NO ROLE-PHASE METHODS (d212 #2): the bundle exposes ONE :meth:`tool_specs` + :attr:`doctrine`.
It does NOT expose ``gather_tool_specs`` / ``decision_tool_specs`` (phase-shaped tool
subsets) — deciding which tools a given phase offers is the runtime's/role's job, which it
does by SELECTING the subset it wants out of this bundle's single catalog.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ..research_tree import TREE_TOOL_SPECS, make_tool_spec
from .base import ObjectBundle
from .web_ingest import (
    NON_ARTICLE_EXT,
    WebGatherAdapter,
    is_readable_fetch,
    looks_like_article_url,
    url_offered,
)

# --------------------------------------------------------------------------- #
# The canonical research ReAct-loop instruction (MOVED here from runtime.py so the
# bundle owns it; runtime re-imports it under its private name). The ``{fetch_cap}``
# placeholder is filled by the loop with ``.format(fetch_cap=...)`` — keep the doubled
# braces so a literal ``{ }`` in the JSON example survives ``str.format``.
# --------------------------------------------------------------------------- #
# CoT-autonomy P4: the loop instruction is DOMAIN KNOWLEDGE only. The reply-format
# protocol ("reply with ONLY a JSON object", "ONE tool call per turn") moved to the
# OPERATING PROTOCOL on the system turn; the findings quality bar (attribute each
# fact to its URL, close with what's missing) moved to the research-methodology
# SPEC; the concrete fetch cap is per-run data on the task turn. What remains is
# what only this bundle knows: how web evidence-gathering works.
RESEARCH_LOOP_INSTRUCTION = (
    "----\n"
    "WEB EVIDENCE GATHERING. web_search finds candidate sources; web_fetch reads one. "
    "Only a URL a search actually returned will load — a URL from memory is rejected "
    "as ungrounded, so copy fetch URLs exactly from the results. A search listing is "
    "not evidence: the facts come from sources you actually read. Never rely on a "
    "page you have not fetched, and never invent facts, figures or URLs."
)

# d191 — the template-then-grow-out flavor that distinguishes the research bundle. Added
# to the bundle DOCTRINE (not to the byte-exact loop instruction above), so it teaches the
# methodology without changing the proven loop text.
#
# d235 — this IS the research METHODOLOGY (decompose-first → grow out → expand/prune → stop),
# and it lives at the ROLE/RUNTIME layer (the research bundle), NOT in the research-analyst SPEC
# (which is now output-quality only). The served research grower's decision node is fed THIS text
# as its ``methodology`` (exported as :data:`RESEARCH_METHODOLOGY`) — so the investigative
# steering comes from the role/runtime, fixing the spec-vs-role blur.
_TEMPLATE_GROW_FLAVOR = (
    "RESEARCH METHODOLOGY — template, then grow out. Begin from a DECOMPOSE-FIRST "
    "template: break the goal into the distinct concerns it names (the WHAT / WHY / "
    "WHEN / HOW it must answer), then GROW OUT — for each concern: search → read the "
    "most relevant chunks → NOTE what you learned and the GAPS it left → EXPAND a "
    "concern that still needs a missing meaning — a gap left open or a curiosity the "
    "goal invites — REASONING over what your gather tools can still surface, not "
    "guessing the answer (expanding COMMITS to a new gathered round) → PRUNE a concern "
    "that added no meaning → and, before you finish, QUESTION completeness: ask plainly "
    "'is this actually complete?' — STOP only when EVERY concern is settled in a note OR "
    "honestly collapsed and no open gap or unexplored area remains. Breadth is not depth: "
    "cover every concern the goal names before drilling one. A note — not findings "
    "prose — is what carries learning forward; its gaps_or_followups drive the next "
    "round."
)

# d235 — the canonical research METHODOLOGY text, owned by the role/runtime (this bundle). The
# served research grower feeds it to the decision/decompose node as ``methodology`` so the
# investigative steering no longer comes from the research-analyst SPEC (now quality-only).
RESEARCH_METHODOLOGY = _TEMPLATE_GROW_FLAVOR

# CoT-autonomy P2 — the per-fetch take-a-note OVERRIDE is RETIRED (it commanded the
# model's next action after every read: "record a STRUCTURED note BEFORE you fetch
# another"). Note discipline's single owners are the ``note`` TOOL DESCRIPTION (what a
# note is and why the gap lane matters) and this bundle's doctrine — knowledge the
# model reasons over, delivered once, never a per-turn command. Kept as an empty
# string so the tool_output_override seam stays wire-compatible.
WEB_FETCH_NOTE_OVERRIDE = ""

_CROSS_VERIFY_FLAVOR = (
    "GROUNDING DISCIPLINE — before you rely on a claim, CROSS-VERIFY it against the "
    "sources you have ACTUALLY pulled (cross_verify). Attribute every fact to a "
    "fetched [S#] source; drop or qualify any claim no fetched source backs; never "
    "cite a URL you only saw in a search list but never read."
)

# --------------------------------------------------------------------------- #
# The cross-verify-against-sources tool (d190 #6 / d191) — orchestrates the existing
# claim_verify.verify_claims; it does NOT reimplement verification.
# --------------------------------------------------------------------------- #
CROSS_VERIFY_TOOL = "cross_verify"

_CROSS_VERIFY_SPEC: dict[str, Any] = make_tool_spec(
    "cross_verify",
    "CROSS-VERIFY a claim (or a draft passage) against ALL the sources you have "
    "already pulled, BEFORE you rely on it. Pass the claim text; get back whether the "
    "fetched [S#] sources back it and which parts are UNBACKED (asserted but supported "
    "by no source you actually read). Use it to drop or attribute any claim no fetched "
    "source supports, and to stop citing a URL you only saw in a search list.",
    {"claim": {"type": "string"}},
    ["claim"],
)


def make_cross_verify_tool(
    sources: Sequence[Mapping[str, Any]],
    *,
    verify: Any = None,
    verify_native: Any = None,
    goal: str = "",
    spec: str = "",
):
    """Build the ``cross_verify`` (spec, handler) bound to THIS run's fetched sources.

    The handler ORCHESTRATES :func:`agent_runtime.claim_verify.verify_claims` — one
    reasoning verify turn over the claim against the fetched-source provenance — and
    returns the grounded flag + the unbacked claims. The caller supplies a ``verify``
    (text) and/or ``verify_native`` (native tool-call) closure that runs one real
    model turn (same callbacks the runtime's verify lane uses), so no transport is
    baked in here. Returns ``(native_tool_spec, async_handler)``."""
    from ..claim_verify import verify_claims

    async def cross_verify(claim: str) -> dict[str, Any]:
        result = await verify_claims(
            str(claim or ""),
            sources,
            verify=verify,
            verify_native=verify_native,
            goal=goal,
            spec=spec,
        )
        return {
            "grounded": bool(result.grounded),
            "unbacked": [
                {"claim": u.claim, "reason": u.reason} for u in result.unbacked
            ],
            "source_count": len(sources),
        }

    return _CROSS_VERIFY_SPEC, cross_verify


class ResearchBundle(ObjectBundle):
    """Grounded evidence-GATHER capability: search/read/note + tree-decision + cross-verify."""

    name = "research"
    summary = (
        "GATHER grounded evidence from the live web — search for sources, fetch + READ "
        "the most relevant chunks, take a NOTE of what you learned and the gaps, "
        "expand/prune/stop the research, and cross-verify a claim. Load this when your "
        "task is to investigate or find facts from real sources."
    )

    @property
    def own_doctrine(self) -> str:  # type: ignore[override]
        return "\n\n".join(
            [
                _TEMPLATE_GROW_FLAVOR,
                _CROSS_VERIFY_FLAVOR,
                RESEARCH_LOOP_INSTRUCTION.replace("{{", "{").replace("}}", "}"),
            ]
        )

    # ------------------------------------------------------------------ #
    # PRIVATE catalog builders (d212 #2): these are NOT a public phase API —
    # they are how this ONE bundle assembles its single tool catalog. The runtime
    # SELECTS the subset it offers each phase out of tool_specs(); it does not call
    # phase-shaped methods on the bundle.
    # ------------------------------------------------------------------ #
    def _gather_specs(
        self,
        search_tool: str,
        fetch_tool: str,
        note_tool: str,
        *,
        emit_notes: bool = False,
    ) -> list[dict[str, Any]]:
        """The native schemas for the gather tools (search / fetch / optional note),
        keyed by the CONFIGURED tool names so a renamed search/fetch/note still maps."""
        specs: list[dict[str, Any]] = [
            make_tool_spec(
                search_tool,
                "STEP 1 — find candidate sources. Search the web for a focused "
                "question to IDENTIFY reliable primary sources before reading. "
                "Use query OPERATORS to sharpen results: \"exact phrase\", "
                "site:domain / -site:domain, OR, leading - to exclude, "
                "intitle:, filetype:pdf. Returns ranked {title,url,snippet} rows; "
                "Wikipedia is excluded automatically. Then web_fetch the most "
                "promising URLs.",
                {"query": {"type": "string"}},
                ["query"],
            ),
            make_tool_spec(
                fetch_tool,
                "STEP 2 — read a source. Fetch ONE OF the URLs from the search results "
                "above, COPIED VERBATIM — NEVER invent, guess, or placeholder a URL (a "
                "made-up url will not load; only a url web_search returned will). You get "
                "back the 1+ MOST RELEVANT chunks of its article text for your sub-question "
                "(the top embedding-ranked passages — not just the single top one, and not "
                "the raw whole page), so READ them before you rely on the source (never cite "
                "a page you have not read). The reply says how many relevant passages it "
                "FOUND vs READ — if you need more of a source, fetch it again or note a "
                "follow-up. If the fetch FAILS the result says WHY — not in the results, "
                "forbidden (403), not-found (404), timeout, or a denied domain — so pick a "
                "DIFFERENT url FROM THE SEARCH RESULTS rather than re-trying a dead link.",
                {"url": {"type": "string"}},
                ["url"],
            ),
            make_tool_spec(
                "image_search",
                "OPTIONAL — when the goal calls for VISUALS (maps, portraits, "
                "diagrams, photos), find REAL image URLs for a focused query. "
                "Returns {title, image_url, source_url, width, height} records. "
                "NOTE the image_url(s) you choose so downstream work can embed "
                "them VERBATIM — never invent, guess, or placeholder an image "
                "path; if no suitable record is returned, proceed without images.",
                {"query": {"type": "string"},
                 "max_results": {"type": "integer"}},
                ["query"],
            ),
        ]
        if emit_notes:
            specs.append(
                make_tool_spec(
                    note_tool,
                    "Your PRIMARY act after reading a source — do this BEFORE you fetch another "
                    "or write findings. Record a STRUCTURED ARTICLE NOTE (not a prose memo): its "
                    "summary, what you LEARNED (key_claims) AND, crucially, the GAPS this source "
                    "did NOT settle (gaps_or_followups: a figure left unverified, an open "
                    "question, the angle to search next). The structured NOTE — not "
                    "findings-prose — is what carries your learning forward: its gaps_or_followups "
                    "DIRECT the next research round, and a later writer reads its gist back "
                    "cheaply (read_notes) to decide which source to cite.",
                    {"url": {"type": "string"}, "summary": {"type": "string"},
                     "category": {"type": "string"}, "source_trust": {"type": "string"},
                     "key_claims": {"type": "array", "items": {"type": "string"}},
                     "relevance": {"type": "string"},
                     "gaps_or_followups": {"type": "array", "items": {"type": "string"}}},
                    ["url", "summary"],
                )
            )
        return specs

    def _decision_specs(self) -> list[dict[str, Any]]:
        """The research-tree decision tools (expand / prune / stop / set_next) —
        the existing :data:`~agent_runtime.research_tree.TREE_TOOL_SPECS`, verbatim."""
        return [dict(s) for s in TREE_TOOL_SPECS]

    def cross_verify_tool(
        self,
        sources: Sequence[Mapping[str, Any]],
        *,
        verify: Any = None,
        verify_native: Any = None,
        goal: str = "",
        spec: str = "",
    ):
        """The cross-verify-against-sources (spec, handler) — orchestrates
        :func:`make_cross_verify_tool`. A handler factory (per-run sources), NOT a
        phase-shaped tool-subset method."""
        return make_cross_verify_tool(
            sources, verify=verify, verify_native=verify_native, goal=goal, spec=spec
        )

    # ------------------------------------------------------------------ #
    # the SINGLE tool catalog (base finish + gather + tree-decision + cross_verify).
    # This is the bundle's ONLY tool-surface method (d212 #2); the runtime selects
    # whatever subset a phase needs out of it.
    # ------------------------------------------------------------------ #
    def tool_specs(self, ctx: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
        ctx = ctx or {}
        search = str(ctx.get("search_tool") or "web_search")
        fetch = str(ctx.get("fetch_tool") or "web_fetch")
        note = str(ctx.get("note_tool") or "note")
        emit_notes = bool(ctx.get("emit_notes", True))
        specs = super().tool_specs(ctx)
        specs += self._gather_specs(search, fetch, note, emit_notes=emit_notes)
        specs += self._decision_specs()
        specs.append(dict(_CROSS_VERIFY_SPEC))
        return specs

    def gather_adapter(
        self, ctx: Optional[Mapping[str, Any]] = None
    ) -> WebGatherAdapter:
        """The web DISPATCH + INGEST adapter (SA-5/d254) — the web bundle OWNS the
        web_search/web_fetch dispatch + all URL/article/readability/record semantics.

        The engine's generic gather loop delegates a configured-web-tool call to this
        adapter (:meth:`WebGatherAdapter.dispatch`) instead of hardcoding web semantics, so
        the engine keeps only generic by-name dispatch. ``ctx`` may carry the configured
        ``search_tool`` / ``fetch_tool`` / ``note_tool`` names so a renamed web tool still
        maps."""
        ctx = ctx or {}
        return WebGatherAdapter(
            str(ctx.get("search_tool") or "web_search"),
            str(ctx.get("fetch_tool") or "web_fetch"),
            str(ctx.get("note_tool") or "note"),
        )

    def tool_output_override(
        self, tool_name: str, ctx: Optional[Mapping[str, Any]] = None
    ) -> Optional[str]:
        """OVERRIDE web_fetch's observation to prompt take-a-note (d221).

        The research CONTEXT extends the base ``web_fetch`` output message so the model
        is steered to record a note for the source it just read (the gap lane). Any other
        tool — and a plain WEB / base context that has not loaded this bundle — gets no
        override (``None``). ``ctx`` may carry the configured ``fetch_tool`` name so a
        renamed fetch tool still matches."""
        ctx = ctx or {}
        fetch = str(ctx.get("fetch_tool") or "web_fetch")
        if tool_name == fetch:
            return WEB_FETCH_NOTE_OVERRIDE
        return None


__all__ = [
    "ResearchBundle",
    "RESEARCH_LOOP_INSTRUCTION",
    "RESEARCH_METHODOLOGY",
    "WEB_FETCH_NOTE_OVERRIDE",
    "CROSS_VERIFY_TOOL",
    "make_cross_verify_tool",
    # web dispatch + ingest semantics, now OWNED by the web bundle (SA-5/d254)
    "WebGatherAdapter",
    "NON_ARTICLE_EXT",
    "looks_like_article_url",
    "is_readable_fetch",
    "url_offered",
]
