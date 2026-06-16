"""Tests for the define-UI surface (a4): the HITL approve gate + HTTP round-trip.

Two layers:
1. The :class:`SpecUIApp` logic directly — proves the d9 property that
   ``/api/research`` compiles NOTHING and only ``/api/approve`` registers a spec
   (and that approve is the single compile path). Net-free via a stub hook.
2. A real ``ThreadingHTTPServer`` round-trip over a localhost socket — proves the
   page renders and the JSON endpoints respond (the a4 self-check), still net-free.
"""
from __future__ import annotations

import json
import threading
import urllib.request
from typing import Any

import pytest

from specialization.registry import SpecRegistry
from specialization.ui import SpecUIApp, make_server


class _Result:
    def __init__(self, ok: bool, value: Any = None, error: str | None = None) -> None:
        self.ok, self.value, self.error = ok, value, error


class _StubHook:
    """Deterministic, net-free research hook (canned search + fetch)."""

    async def invoke(self, name: str, /, **kwargs: Any) -> _Result:
        if name == "web_search":
            return _Result(True, {"results": [
                {"url": "https://example.org/a", "title": "A", "snippet": "how to do it"},
                {"url": "https://example.org/b", "title": "B", "snippet": "best practices"},
            ]})
        if name == "web_fetch":
            return _Result(True, {"title": "Page", "text": "extracted how-to notes"})
        return _Result(False, error=f"unknown tool {name!r}")


def _app(tmp_path) -> SpecUIApp:
    return SpecUIApp(tmp_path / "specs", hook_factory=lambda: _StubHook())


# --------------------------------------------------------------------------- #
# Layer 1: the HITL gate property
# --------------------------------------------------------------------------- #
def test_research_authors_draft_but_compiles_nothing(tmp_path):
    app = _app(tmp_path)
    view = app.research("md-author", "Writes Markdown", "Author MD reports")
    # A draft was surfaced...
    assert view["draft_id"]
    assert view["name"] == "md-author"
    assert view["body"]                      # condensed draft present
    assert view["source"] == "ui"
    # ...but NOTHING is compiled/registered yet (the gate has not been passed).
    assert SpecRegistry(app.specs_dir).index() == []


def test_approve_is_the_only_compile_path(tmp_path):
    app = _app(tmp_path)
    view = app.research("md-author", "Writes Markdown", "Author MD reports")
    assert SpecRegistry(app.specs_dir).names() == []      # pre-click: empty

    result = app.approve(view["draft_id"])                # the click
    assert result["ok"] and result["registered"] == "md-author"

    reg = SpecRegistry(app.specs_dir)
    assert reg.names() == ["md-author"]                    # post-click: registered
    assert "Specialist: md-author" in reg.load("md-author").body


def test_approve_unknown_draft_refused(tmp_path):
    app = _app(tmp_path)
    with pytest.raises(KeyError):
        app.approve("not-a-real-draft-id")
    assert SpecRegistry(app.specs_dir).index() == []       # still nothing compiled


def test_deny_discards_without_compiling(tmp_path):
    app = _app(tmp_path)
    view = app.research("md-author", "Writes Markdown", "Author MD reports")
    app.deny(view["draft_id"])
    assert SpecRegistry(app.specs_dir).index() == []
    with pytest.raises(KeyError):                          # draft is gone
        app.approve(view["draft_id"])


def test_index_surface_is_body_free(tmp_path):
    app = _app(tmp_path)
    view = app.research("md-author", "Writes Markdown", "Author MD reports")
    app.approve(view["draft_id"])
    specs = app.specs()["specs"]                            # planner-facing index
    assert specs and set(specs[0]) == {"name", "description", "source"}  # no body key


# --------------------------------------------------------------------------- #
# Layer 2: a real HTTP round-trip (page renders + endpoints respond)
# --------------------------------------------------------------------------- #
def test_http_roundtrip_renders_and_responds(tmp_path):
    httpd = make_server(tmp_path / "specs", host="127.0.0.1", port=0,
                        hook_factory=lambda: _StubHook())
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        # GET / renders the page with the hit-testable Approve button.
        with urllib.request.urlopen(base + "/", timeout=5) as r:
            html = r.read().decode("utf-8")
            assert r.status == 200
            assert 'id="approve-btn"' in html and "pointer-events: auto" in html

        # POST /api/research authors a draft (compiles nothing).
        draft = _post(base + "/api/research",
                      {"name": "html-author", "description": "Writes HTML",
                       "intent": "Author HTML reports"})
        assert draft["draft_id"] and draft["body"]

        # GET /api/specs is still empty (no approve yet).
        assert _get(base + "/api/specs")["specs"] == []

        # POST /api/approve compiles + registers.
        appr = _post(base + "/api/approve", {"draft_id": draft["draft_id"]})
        assert appr["ok"] and appr["registered"] == "html-author"

        specs = _get(base + "/api/specs")["specs"]
        assert [s["name"] for s in specs] == ["html-author"]
        assert "body" not in specs[0]                       # index stays body-free
    finally:
        httpd.shutdown()
        httpd.server_close()


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))
