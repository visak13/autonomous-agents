"""s9/N2 (d60/c15 part-b): the per-article research-CONTROL artifact — ``ArticleNote``.

A LEGITIMATE structured-CONTROL lane (d50.1/d51-clean: CONTROL, **never** deliverable
content). On each readable ``web_fetch`` the leaf research agent emits a small,
reasoning-populated record about the source it just read; the runtime coerces it into
an :class:`ArticleNote`. These notes:

* carry a per-article **summary** + **categorization** + **source-trust tier**, and
* DIRECT the next node's search via ``gaps_or_followups``.

WHY this does NOT violate "content is RAW, never serialized" (d50/d50.1): the note is
*lightweight control data* (short fields, like the c5 tool args the small model emits
reliably) — NOT the document. The RAW article markdown still flows untouched as the
``tool_value['fetched']`` source text for the UNCHANGED c13 write side; the notes are an
*additive* parallel lane (``tool_value['article_notes']``).

ANTI-FABRICATION discipline:
* ``source_id`` / ``url`` / ``title`` are owned by the RUNTIME (taken from the source the
  agent actually fetched), never trusted from the model — a note can only describe a
  source that was really read.
* ``source_trust`` for a Wikipedia source is FORCED to ``reference-untrusted`` (d60:
  Wikipedia is *not trusted but citable IF attributed* — a trust TIER, not a ban). This
  is source-trust POLICY, not template deliverable content; the model still reasons the
  summary / claims / relevance / followups.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

# The source-trust TIERS (a small, fixed control vocabulary — d60 provenance weighting):
#   primary             — first-hand/official source (gov, the org itself, a study).
#   secondary           — reputable reporting/analysis ABOUT a primary source.
#   reference-untrusted — encyclopaedic/aggregator (Wikipedia): citable ONLY if
#                         attributed, NEVER the sole backing for a hard figure (d60).
SOURCE_TRUST_TIERS: tuple[str, ...] = ("primary", "secondary", "reference-untrusted")
_DEFAULT_TIER = "secondary"

# Host substrings that are HARD-classified ``reference-untrusted`` regardless of what the
# model claims — the d60 Wikipedia policy (encyclopaedic, community-edited).
_REFERENCE_UNTRUSTED_HOSTS = ("wikipedia.org", "wikimedia.org", "wiktionary.org")

# Loose synonyms the small model may emit for a tier → the canonical tier.
_TIER_SYNONYMS = {
    "primary": "primary",
    "official": "primary",
    "firsthand": "primary",
    "first-hand": "primary",
    "gov": "primary",
    "government": "primary",
    "study": "primary",
    "secondary": "secondary",
    "news": "secondary",
    "reporting": "secondary",
    "analysis": "secondary",
    "media": "secondary",
    "reference": "reference-untrusted",
    "reference-untrusted": "reference-untrusted",
    "untrusted": "reference-untrusted",
    "encyclopedia": "reference-untrusted",
    "encyclopaedia": "reference-untrusted",
    "wiki": "reference-untrusted",
    "wikipedia": "reference-untrusted",
    "aggregator": "reference-untrusted",
}

# Bounds so a verbose small model cannot bloat the CONTROL lane (it stays lightweight,
# the whole point of a control artifact vs the RAW document).
_MAX_LIST_ITEMS = 8
_MAX_FIELD_CHARS = 600
_MAX_ITEM_CHARS = 300


def classify_source_trust(url: str, claimed: Any = None) -> str:
    """Resolve a source's trust TIER from its URL + the model's claimed tier.

    Wikipedia/Wikimedia hosts are FORCED to ``reference-untrusted`` (d60 policy — they
    are citable only if attributed, never sole backing). Otherwise the model's claimed
    tier is honoured when it normalises to a known tier (the model reasons about the
    source); an unknown/blank claim degrades to ``secondary`` (the safe middle tier)."""
    host = ""
    try:
        host = (urlparse(url or "").netloc or "").lower()
    except (ValueError, TypeError):
        host = ""
    if any(ref in host for ref in _REFERENCE_UNTRUSTED_HOSTS):
        return "reference-untrusted"
    token = str(claimed or "").strip().lower()
    if token in SOURCE_TRUST_TIERS:
        return token
    return _TIER_SYNONYMS.get(token, _DEFAULT_TIER)


class ArticleNote(BaseModel):
    """A per-article structured-CONTROL record (d50.1-clean: control, NOT content).

    Reasoning-populated by the leaf research agent about ONE source it has read, used to
    DIRECT the next node's search and to weight provenance in the verification lane (N5).
    Provenance fields (``source_id``/``url``/``title``) are RUNTIME-owned; the reasoning
    fields (``summary``/``category``/``key_claims``/``relevance``/``gaps_or_followups``)
    come from the model; ``source_trust`` is the model's claim reconciled with policy."""

    source_id: int = Field(..., description="1-based id of the read source (runtime-owned).")
    url: str = Field(..., description="Canonical fetched URL (runtime-owned).")
    title: str = Field(default="", description="Source title (runtime-owned).")
    source_trust: str = Field(
        default=_DEFAULT_TIER,
        description="Trust tier: primary | secondary | reference-untrusted.",
    )
    category: str = Field(default="", description="Topical category of the source.")
    summary: str = Field(default="", description="Short per-article summary (reasoned).")
    key_claims: list[str] = Field(
        default_factory=list, description="Short factual claims drawn from the source."
    )
    relevance: str = Field(default="", description="Why/whether this source serves the goal.")
    gaps_or_followups: list[str] = Field(
        default_factory=list,
        description="Open gaps / next search directions this source surfaces.",
    )


def _as_str(value: Any) -> str:
    return "" if value is None else str(value).strip()[:_MAX_FIELD_CHARS]


def _as_str_list(value: Any) -> list[str]:
    """Coerce a model-supplied value into a bounded list of short strings.

    A small model is inconsistent — it may emit a single string, a list, or None for a
    "list" field. Accept all shapes; split a stray newline/semicolon-joined string; drop
    blanks; bound the count and each item's length so the control lane stays lightweight."""
    if value is None:
        return []
    if isinstance(value, str):
        raw = [part for chunk in value.split("\n") for part in chunk.split(";")]
        items = [p.strip() for p in raw if p.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value if str(v).strip()]
    else:
        items = [str(value).strip()]
    return [item[:_MAX_ITEM_CHARS] for item in items[:_MAX_LIST_ITEMS]]


def coerce_article_note(
    data: Mapping[str, Any],
    *,
    source_id: int,
    url: str,
    title: str = "",
) -> Optional[ArticleNote]:
    """Build an :class:`ArticleNote` from a loose model-emitted dict + runtime provenance.

    Returns ``None`` only when ``data`` is not a mapping (a malformed note is simply
    dropped — never crashes a research turn). Provenance (``source_id``/``url``/``title``)
    is supplied by the RUNTIME and overrides anything the model put there (anti-fabrication
    of which source a claim came from); the reasoning fields are taken from the model and
    bounded; ``source_trust`` is reconciled with the d60 policy."""
    if not isinstance(data, Mapping):
        return None
    claimed_trust = (
        data.get("source_trust")
        or data.get("trust")
        or data.get("trust_tier")
        or data.get("tier")
    )
    return ArticleNote(
        source_id=int(source_id),
        url=str(url),
        title=_as_str(title),
        source_trust=classify_source_trust(url, claimed_trust),
        category=_as_str(
            data.get("category") or data.get("categorization") or data.get("topic")
        ),
        summary=_as_str(data.get("summary") or data.get("abstract")),
        key_claims=_as_str_list(
            data.get("key_claims") or data.get("claims") or data.get("facts")
        ),
        relevance=_as_str(data.get("relevance") or data.get("relevant")),
        gaps_or_followups=_as_str_list(
            data.get("gaps_or_followups")
            or data.get("gaps")
            or data.get("followups")
            or data.get("follow_ups")
            or data.get("next_searches")
        ),
    )
