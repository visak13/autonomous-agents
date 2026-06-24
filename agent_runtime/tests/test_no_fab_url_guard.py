"""MS3 (d84/d89) — the FINAL-DOCUMENT NO-FAB URL GUARD (``strip_ungrounded_urls``).

The deterministic complement to the reasoning verify lane: after the report is
assembled, every URL is checked against the run's set of ACTUALLY-FETCHED source URLs
(the normalized ``url`` of each fetched source — and ONLY that, not URLs embedded
inside the article body). A cited URL is grounded IFF its normalized form is in that
set; any URL NO fetched source backs is removed — its ``<a href>`` anchor unwrapped to
the visible text, bare occurrences dropped — so no ungrounded URL can survive the
deliverable, for any model (d60 made a guarantee, not luck). The guard is
content-preserving (strips ONLY the offending token, never invents prose) and
idempotent (a clean report is byte-identical). These prove the set-membership test, the
anchor unwrap, the bare-URL drop, prefix-safety, and the no-op-on-clean /
no-op-on-no-sources cases — plus the s13 (B4/§4A) closure of the d92 404 leak: a URL
present ONLY as an inline hyperlink inside a fetched article is now REMOVED.
"""
from __future__ import annotations

from agent_runtime.synth_tools import fetched_url_set, strip_ungrounded_urls

# A realistic fetched-source set: a top-level URL plus a secondary URL embedded in the
# article text (the real "cite-the-link-inside-the-article" case MSFr observed).
_SOURCES = [
    {
        "title": "Visualising 12 days",
        "url": "https://aljazeera.com/news/12-days",
        "markdown": "Damage estimates rose; see https://cnn.com/interactive/2025/iran for the map.",
    },
    {
        "title": "Cost of the war",
        "url": "https://csis.org/analysis/cost",
        "summary": "Total cost reached $16.5 billion by Day 12.",
    },
]


def test_grounded_urls_survive_untouched():
    # Both cited URLs are ACTUALLY-FETCHED source URLs (in the fetched-URL set) → no change.
    doc = (
        '<p>Per <a href="https://aljazeera.com/news/12-days">Al Jazeera</a>, '
        "see https://csis.org/analysis/cost.</p>"
    )
    out, removed = strip_ungrounded_urls(doc, _SOURCES)
    assert removed == []
    assert out == doc  # byte-identical: a clean report is never touched


def test_fabricated_anchor_is_unwrapped_to_its_text():
    doc = (
        '<p>Real <a href="https://csis.org/analysis/cost">CSIS</a> but '
        '<a href="https://made-up.example/fake-report">fabricated</a> link.</p>'
    )
    out, removed = strip_ungrounded_urls(doc, _SOURCES)
    assert removed == ["https://made-up.example/fake-report"]
    assert "made-up.example" not in out
    assert "fabricated" in out  # the visible anchor TEXT is preserved (prose kept)
    assert 'href="https://csis.org/analysis/cost"' in out  # grounded anchor intact


def test_bare_fabricated_url_is_dropped():
    doc = "See https://csis.org/analysis/cost and https://ghost.invalid/p for details."
    out, removed = strip_ungrounded_urls(doc, _SOURCES)
    assert removed == ["https://ghost.invalid/p"]
    assert "ghost.invalid" not in out
    assert "https://csis.org/analysis/cost" in out


def test_trailing_punctuation_does_not_falsely_flag_or_survive():
    # Grounded URL with a trailing period stays; fabricated one with trailing paren goes.
    doc = "Grounded https://aljazeera.com/news/12-days. Fake (https://nope.invalid/x)."
    out, removed = strip_ungrounded_urls(doc, _SOURCES)
    assert removed == ["https://nope.invalid/x"]
    assert "https://aljazeera.com/news/12-days" in out
    assert "nope.invalid" not in out


def test_removing_fabricated_url_does_not_corrupt_a_prefix_sharing_grounded_url():
    # A fabricated URL that SHARES A PREFIX with a grounded one must be excised without
    # damaging the grounded link (longer URLs removed first + exact-literal substitution).
    srcs = [{"url": "https://site.example/article/full-report", "markdown": ""}]
    doc = (
        'Good <a href="https://site.example/article/full-report">full</a>, '
        "bad https://site.example/article/ghost-fabrication."
    )
    out, removed = strip_ungrounded_urls(doc, srcs)
    assert removed == ["https://site.example/article/ghost-fabrication"]
    assert "ghost-fabrication" not in out
    assert "https://site.example/article/full-report" in out  # grounded link survives


def test_prefix_of_a_fetched_url_is_not_grounded_and_is_removed():
    # EXACT set membership (s13/§4A): a URL that is merely a PREFIX/substring of a real
    # fetched URL is NOT itself a fetched source → it is removed. This closes the lenient
    # substring loophole the old concatenated corpus had (where a prefix matched).
    srcs = [{"url": "https://site.example/article/full-report", "markdown": ""}]
    doc = "Prefix https://site.example/article is not itself a fetched source."
    out, removed = strip_ungrounded_urls(doc, srcs)
    assert removed == ["https://site.example/article"]
    assert "https://site.example/article" not in out
    # the real fetched URL is untouched (it never appears in this doc, but stays groundable)
    assert fetched_url_set(srcs) == {"https://site.example/article/full-report"}


def test_no_sources_means_every_url_is_ungrounded():
    doc = 'A <a href="https://x.example/a">link</a> and https://y.example/b here.'
    out, removed = strip_ungrounded_urls(doc, [])
    assert set(removed) == {"https://x.example/a", "https://y.example/b"}
    assert "x.example" not in out and "y.example" not in out
    assert "link" in out  # anchor text preserved even with no sources


def test_idempotent_second_pass_is_a_noop():
    doc = 'Keep https://csis.org/analysis/cost drop <a href="https://fake.invalid/z">z</a>.'
    once, removed1 = strip_ungrounded_urls(doc, _SOURCES)
    twice, removed2 = strip_ungrounded_urls(once, _SOURCES)
    assert removed1 == ["https://fake.invalid/z"]
    assert removed2 == []
    assert twice == once


def test_empty_doc_is_returned_unchanged():
    assert strip_ungrounded_urls("", _SOURCES) == ("", [])
    assert strip_ungrounded_urls("   ", _SOURCES) == ("   ", [])


def test_fetched_url_set_contains_only_actually_fetched_urls():
    # The grounded universe is the set of fetched source URLs ONLY — NOT URLs embedded
    # inside the article body/summary/title (those are secondary links, not fetched).
    s = fetched_url_set(_SOURCES)
    assert s == {
        "https://aljazeera.com/news/12-days",
        "https://csis.org/analysis/cost",
    }
    # the CNN link embedded in aljazeera's markdown is NOT in the fetched-URL set
    assert "https://cnn.com/interactive/2025/iran" not in s


# =========================================================================== #
# s13 (B4 / design §4A) — BACKSTOP A: citation guard → FETCHED-URL SET. Closes the
# d92 404 leak: a URL present ONLY as an inline hyperlink inside a fetched article
# (never itself fetched, may 404) is now UNGROUNDED and REMOVED. Was a false PASS
# under the old lenient substring corpus because it was a substring of the markdown.
# =========================================================================== #
def test_s13_inline_only_hyperlink_url_is_stripped():
    # aljazeera was fetched; the CNN link only appears INSIDE aljazeera's body (and as a
    # citation in the doc). Under the fetched-URL SET it is ungrounded → removed; the
    # fetched source URL it sits next to stays grounded. Anchor TEXT is preserved.
    sources = [
        {
            "title": "Al Jazeera 12 days",
            "url": "https://aljazeera.com/news/12-days",
            # body links out to a secondary page the run NEVER fetched (404-class)
            "markdown": 'Damage map at <a href="https://cnn.com/interactive/2025/iran">CNN</a>.',
        },
    ]
    doc = (
        '<p>Per <a href="https://aljazeera.com/news/12-days">Al Jazeera</a> and '
        'the <a href="https://cnn.com/interactive/2025/iran">CNN map</a>.</p>'
    )
    out, removed = strip_ungrounded_urls(doc, sources)
    assert removed == ["https://cnn.com/interactive/2025/iran"]  # inline-only → STRIPPED
    assert "cnn.com" not in out                                   # the 404-class link is gone
    assert "CNN map" in out                                       # anchor TEXT preserved (prose kept)
    assert 'href="https://aljazeera.com/news/12-days"' in out     # the FETCHED url stays grounded


def test_s13_bare_inline_only_url_is_stripped():
    # The same closure for a BARE (non-anchor) inline-only URL: csis was fetched; the
    # reuters link appears only inside csis's summary → ungrounded → dropped.
    sources = [
        {
            "title": "CSIS cost",
            "url": "https://csis.org/analysis/cost",
            "summary": "Per Reuters https://reuters.com/world/ceasefire the war cost $16.5B.",
        },
    ]
    doc = "Cost per https://csis.org/analysis/cost; ceasefire per https://reuters.com/world/ceasefire."
    out, removed = strip_ungrounded_urls(doc, sources)
    assert removed == ["https://reuters.com/world/ceasefire"]  # inline-in-summary only → stripped
    assert "reuters.com" not in out
    assert "https://csis.org/analysis/cost" in out             # fetched source url survives


def test_s13_fetched_url_set_excludes_body_embedded_links():
    # Direct proof of the set semantics that close the leak: a source whose body and
    # summary mention OTHER urls contributes ONLY its own fetched ``url`` to the set.
    sources = [
        {
            "url": "https://aljazeera.com/news/12-days",
            "markdown": "See https://cnn.com/interactive/2025/iran and https://bbc.com/x.",
            "summary": "Also https://nytimes.com/y covered it.",
        },
    ]
    s = fetched_url_set(sources)
    assert s == {"https://aljazeera.com/news/12-days"}
    for embedded in ("cnn.com", "bbc.com", "nytimes.com"):
        assert not any(embedded in u for u in s)


def test_s13_normalisation_grounds_across_fragment_query_and_trailing_slash():
    # A cited URL that differs from the fetched source ONLY by #fragment, ?query, or a
    # trailing slash is the SAME resource → still grounded (no false removal of a real
    # citation). The scheme/host/path normalisation is applied to both sides.
    sources = [{"url": "https://aljazeera.com/news/12-days/"}]  # fetched WITH trailing slash
    doc = (
        "A https://aljazeera.com/news/12-days#section-2 "
        "B https://aljazeera.com/news/12-days?utm=x "
        "C https://AlJazeera.com/news/12-days"
    )
    out, removed = strip_ungrounded_urls(doc, sources)
    assert removed == []          # all three normalise to the fetched resource
    assert out == doc             # a real, grounded report is untouched
