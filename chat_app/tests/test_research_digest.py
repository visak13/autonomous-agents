"""Phase 2B — the code-assembled, token-budgeted research digest.

Deterministic (no model), bounded by construction, cursor names what was withheld
and how to pull it — the replacement for the retired findings-blob pushes.
"""
from __future__ import annotations

from chat_app.digest import build_research_digest


def _sources(n: int):
    return [
        {"title": f"Source {i}", "url": f"https://ex.org/a{i}",
         "markdown": f"body {i} " * 200}
        for i in range(1, n + 1)
    ]


def _notes(n: int):
    return [
        {"url": f"https://ex.org/a{i}", "summary": f"claims about topic {i}",
         "key_claims": [f"claim {i}"], "gaps_or_followups": ["dig deeper"]}
        for i in range(1, n + 1)
    ]


def test_digest_carries_index_gists_and_cursor() -> None:
    out = build_research_digest(_sources(3), _notes(3))
    assert "[S1]" in out and "[S3]" in out          # verbatim source index
    assert "ARTICLE-NOTE GISTS" in out              # pullable note gists
    assert "read_notes" in out and "load_source" in out  # the pull cursor
    assert "https://ex.org/a1" in out


def test_digest_respects_budget_with_downshift() -> None:
    out = build_research_digest(_sources(40), _notes(40), token_budget=800)
    assert len(out) <= 800 * 4
    # withheld layers are NAMED, never silently dropped
    assert "withheld" in out or "trimmed" in out


def test_digest_without_notes_still_has_index_and_cursor() -> None:
    out = build_research_digest(_sources(2), None)
    assert "[S2]" in out
    assert "load_source" in out
    assert "ARTICLE-NOTE GISTS" not in out


def test_digest_never_embeds_source_bodies() -> None:
    # The digest tells the consumer what exists — it never spoon-feeds the
    # verbatim article text (that is load_source's job, on demand).
    out = build_research_digest(_sources(3), _notes(3))
    assert "body 1 body 1" not in out
