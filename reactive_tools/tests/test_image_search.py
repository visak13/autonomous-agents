"""image_search — the GENERIC single-purpose image-URL tool (autonomy rebuild P1).

Owner charter: the tool finds real image URLs and knows NOTHING about reports/HTML/
any use case. Offline tests with a fake backend: normalization, cap, deny-list on the
source domain, cache, backoff, and the never-placeholder contract living in the
DESCRIPTION (definition layer), not in code.
"""
from __future__ import annotations

import pytest

from reactive_tools import ToolInputError
from reactive_tools.web_tools import (
    IMAGE_SEARCH_TOOL,
    ImageSearchArgs,
    ResultCache,
    make_image_search,
)


def _rows(n: int = 3, host: str = "example.org"):
    return [
        {"title": f"img {i}", "image_url": f"https://cdn.{host}/i{i}.jpg",
         "source_url": f"https://{host}/page{i}", "width": 800, "height": 600}
        for i in range(n)
    ]


def test_normalized_records_and_count() -> None:
    calls: list[tuple] = []

    def backend(query, max_results, region, timeout):
        calls.append((query, max_results))
        return _rows(3)

    tool = make_image_search(backend=backend, cache=ResultCache())
    out = tool("maratha empire map")
    assert out["count"] == 3
    rec = out["results"][0]
    assert set(rec) == {"title", "image_url", "source_url", "width", "height"}
    assert rec["image_url"].startswith("https://cdn.")


def test_deny_list_drops_by_source_domain() -> None:
    def backend(query, max_results, region, timeout):
        return _rows(2) + [{
            "title": "wiki img", "image_url": "https://upload.wikimedia.org/x.jpg",
            "source_url": "https://en.wikipedia.org/wiki/X", "width": 1, "height": 1,
        }]

    tool = make_image_search(backend=backend, cache=ResultCache())
    out = tool("anything")
    assert out["count"] == 2
    assert all("wikipedia" not in r["source_url"] for r in out["results"])


def test_cache_serves_repeat_query() -> None:
    hits = {"n": 0}

    def backend(query, max_results, region, timeout):
        hits["n"] += 1
        return _rows(1)

    tool = make_image_search(backend=backend, cache=ResultCache())
    a = tool("q")
    b = tool("q")
    assert hits["n"] == 1
    assert a["cached"] is False and b["cached"] is True


def test_backoff_then_fail_is_actionable() -> None:
    from reactive_tools.web_tools import RatelimitException

    sleeps: list[float] = []

    def backend(query, max_results, region, timeout):
        raise RatelimitException("429")

    tool = make_image_search(
        backend=backend, cache=ResultCache(), max_retries=2,
        sleep=sleeps.append,
    )
    with pytest.raises(ToolInputError) as exc:
        tool("q")
    assert "rate-limited" in str(exc.value)
    assert len(sleeps) == 2  # backed off before failing


def test_empty_query_refused() -> None:
    tool = make_image_search(backend=lambda *a, **k: [], cache=ResultCache())
    with pytest.raises(ToolInputError):
        tool("   ")


def test_description_owns_the_never_placeholder_contract() -> None:
    # The behavior contract lives in the DEFINITION LAYER (tool description),
    # never in engine code: the description must carry the verbatim-embed +
    # never-placeholder doctrine, and the args model stays single-purpose.
    desc = IMAGE_SEARCH_TOOL.description
    assert "VERBATIM" in desc or "verbatim" in desc
    assert "placeholder" in desc
    assert set(ImageSearchArgs.model_fields) == {"query", "max_results", "region"}
