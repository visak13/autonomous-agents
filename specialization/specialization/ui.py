"""The minimal DEFINE-UI + HITL compile-gate SURFACE (a4, serves d8/d9).

This is where a specialization is **DEFINED** (d9): a small in-process web app
that lets a human

1. **DEFINE** a specialization — a form for ``name`` / ``description`` / ``intent``
   (a :class:`~specialization.model.RawDefinition`);
2. **RUN web-research** — a button that calls the engine's research+author path
   (:meth:`SpecializationEngine.author_draft`) and shows the surfaced *sources*
   and the condensed *DRAFT* body; **nothing is compiled yet**;
3. **APPROVE** — a real, hit-testable button that, *only when clicked*, supplies
   the user-approval token to :meth:`SpecializationEngine.compile`, which is the
   one and only path that compiles + registers the spec (the d9 HITL gate).

Why this is a GENUINE approval surface (neuron steer)
-----------------------------------------------------
The engine's compile gate is **structurally unreachable without an injected
user-facing approver** (see :mod:`specialization.engine`). This UI is that
surface: the ``/api/research`` endpoint authors a DRAFT and holds it server-side
but injects NO approver, so it *cannot* compile. The approver — an
:class:`~specialization.engine.ApprovalToken` granted for *exactly the held
draft* — is minted and handed to ``engine.compile`` **only inside the
``/api/approve`` handler**, which only runs when the Approve button is clicked.
So a compiled+registered spec is proof that a real click landed on the button;
there is no code path that compiles without one. The same surface is reused at
s8: the autonomous loop authors a draft, registers it here, and surfaces its
markdown+html to the REAL user for the same one-click approval.

In-process, zero new dependencies (d2/d8/d10)
---------------------------------------------
Built on the Python **stdlib** ``http.server`` (``ThreadingHTTPServer``) — no
FastAPI/Starlette/uvicorn (none are in the workspace; "reuse, don't add heavy
deps"). The async engine methods are driven via :func:`asyncio.run` inside each
request thread. No broker/pool, no sockets beyond the localhost HTTP listener,
no shell (d2/d8). The page is a single self-contained HTML document served from
``/``; the data flows over small JSON endpoints.
"""
from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional, Union

from specialization.engine import (
    ApprovalToken,
    SOURCE_AUTONOMOUS,
    SOURCE_UI,
    SpecDraft,
    SpecializationEngine,
)
from specialization.model import RawDefinition
from specialization.registry import SpecRegistry
from specialization.research import ToolInvoker

# A factory that builds the in-process research ToolHook (the d6 free no-key web
# tools). Injectable so a fast self-check can pass a deterministic stub and the
# live launch helper can pass the real DuckDuckGo-backed hook.
HookFactory = Callable[[], ToolInvoker]


def _default_hook_factory() -> ToolInvoker:
    """Build the REAL free no-key web-research hook (lazy import — keeps the UI
    module importable with no network / no reactive_tools for the stub path)."""
    from specialization.research import build_research_hook

    return build_research_hook()


# --------------------------------------------------------------------------- #
# The app: holds the registry dir, the hook factory, and the live draft store.
# --------------------------------------------------------------------------- #
class SpecUIApp:
    """In-process state for the define-UI: registry location + held drafts.

    A DRAFT authored by ``/api/research`` is held here (keyed by its content
    ``challenge``) until the user approves it — so the approver minted at
    ``/api/approve`` is bound to *exactly* the surfaced draft (the engine's d9
    gate rejects a mismatch). Thread-safe: ``ThreadingHTTPServer`` serves each
    request on its own thread."""

    def __init__(
        self,
        specs_dir: Union[str, Path],
        *,
        hook_factory: Optional[HookFactory] = None,
    ) -> None:
        self.specs_dir = Path(specs_dir)
        self.specs_dir.mkdir(parents=True, exist_ok=True)
        self._hook_factory = hook_factory or _default_hook_factory
        self._drafts: dict[str, SpecDraft] = {}
        self._lock = threading.Lock()

    # -- engine construction (fresh per request, inside the request's loop) -- #
    def _new_engine(self, hook: ToolInvoker) -> SpecializationEngine:
        registry = SpecRegistry(self.specs_dir)
        return SpecializationEngine(registry, hook=hook, specs_dir=self.specs_dir)

    # -- draft store -------------------------------------------------------- #
    def _put_draft(self, draft: SpecDraft) -> None:
        with self._lock:
            self._drafts[draft.challenge] = draft

    def _get_draft(self, draft_id: str) -> Optional[SpecDraft]:
        with self._lock:
            return self._drafts.get(draft_id)

    def _drop_draft(self, draft_id: str) -> None:
        with self._lock:
            self._drafts.pop(draft_id, None)

    # ------------------------------------------------------------------ #
    # Endpoint logic (pure-ish: take parsed input, return a JSON-able dict)
    # ------------------------------------------------------------------ #
    def research(self, name: str, description: str, intent: str, *, source: str = SOURCE_UI) -> dict[str, Any]:
        """DEFINE -> research -> author a DRAFT. Compiles NOTHING (no approver).

        Runs the engine's research+condense path and HOLDS the resulting draft
        server-side for a later approval. Returns the draft preview the UI
        renders (sources + condensed body) plus the ``draft_id`` the Approve
        button echoes back."""
        raw = RawDefinition(name=name.strip(), description=description.strip(), intent=intent.strip())

        async def _run() -> SpecDraft:
            hook = self._hook_factory()
            engine = self._new_engine(hook)
            return await engine.author_draft(raw, source=source)

        draft = asyncio.run(_run())
        self._put_draft(draft)
        return self._draft_view(draft)

    def approve(self, draft_id: str) -> dict[str, Any]:
        """The HITL GATE: compile + register the held draft ON USER CLICK.

        This is the ONLY place an approver is injected into ``engine.compile``.
        The approver grants an :class:`ApprovalToken` for the *held* draft, so
        the engine's challenge check passes; with no held draft (or a stale id)
        there is nothing to approve. Returns the registered spec doc."""
        draft = self._get_draft(draft_id)
        if draft is None:
            raise KeyError(f"no pending draft {draft_id!r} to approve")

        async def _run():
            engine = self._new_engine(_NullHook())  # compile() never touches the hook
            # The user-facing approver: granting a token for THE surfaced draft
            # is the digital form of the human clicking "Approve" on this draft.
            spec = await engine.compile(draft, approver=lambda d: ApprovalToken.grant(d))
            return spec

        spec = asyncio.run(_run())
        self._drop_draft(draft_id)
        path = self.specs_dir / f"{spec.name}.md"
        return {
            "ok": True,
            "registered": spec.name,
            "path": str(path),
            "source": spec.source,
            "markdown": spec.to_markdown(),
        }

    def deny(self, draft_id: str) -> dict[str, Any]:
        """Discard a held draft without compiling (the user declined)."""
        self._drop_draft(draft_id)
        return {"ok": True, "denied": draft_id}

    def specs(self) -> dict[str, Any]:
        """The planner-facing INDEX (d10): {name, description, source} only — no
        bodies. Backs the 'Registered specialists' panel."""
        registry = SpecRegistry(self.specs_dir)
        return {"specs": [{"name": e.name, "description": e.description, "source": e.source}
                          for e in registry.index()]}

    # -- view helpers ------------------------------------------------------- #
    def _draft_view(self, draft: SpecDraft) -> dict[str, Any]:
        return {
            "draft_id": draft.challenge,
            "name": draft.raw.name,
            "description": draft.raw.description,
            "source": draft.source,
            "body": draft.body,
            "markdown": draft.to_markdown(),
            "html": draft.to_html(),
            "sources": [
                {"url": s.url, "title": s.title, "fetched": getattr(s, "fetched", False)}
                for s in draft.trace.sources
            ],
            "queries": list(draft.trace.queries),
            "stats": dict(draft.trace.stats),
            "errors": list(draft.trace.errors),
        }


@dataclass
class _NullResult:
    ok: bool = False
    value: Any = None
    error: Optional[str] = "no-op hook (compile path does not research)"


class _NullHook:
    """A do-nothing ToolInvoker for the approve path — ``engine.compile`` never
    calls the hook, so this just satisfies the engine's required ``hook`` arg."""

    async def invoke(self, name: str, /, **kwargs: Any) -> _NullResult:  # pragma: no cover - never called
        return _NullResult()


# --------------------------------------------------------------------------- #
# The single self-contained HTML page
# --------------------------------------------------------------------------- #
PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Specialization — Define &amp; Approve</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font: 15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 2rem; max-width: 920px; margin-inline: auto;
         color: #1a1a1a; background: #fafafa; }
  h1 { font-size: 1.4rem; margin: 0 0 .25rem; }
  .sub { color: #666; margin: 0 0 1.5rem; }
  fieldset { border: 1px solid #ddd; border-radius: 10px; padding: 1rem 1.25rem;
             margin: 0 0 1.25rem; background: #fff; }
  legend { font-weight: 600; padding: 0 .4rem; }
  label { display: block; font-weight: 600; margin: .6rem 0 .2rem; }
  input, textarea { width: 100%; padding: .55rem .65rem; border: 1px solid #ccc;
                    border-radius: 7px; font: inherit; background: #fff; }
  textarea { min-height: 4.5rem; resize: vertical; }
  button { font: inherit; font-weight: 600; border: 0; border-radius: 8px;
           padding: .6rem 1.1rem; cursor: pointer; pointer-events: auto; }
  .btn-research { background: #2563eb; color: #fff; }
  .btn-research:hover { background: #1d4ed8; }
  /* The APPROVE control: a real, hit-testable button. pointer-events:auto and
     position:static (no overlay, nothing covering it) so a TRUSTED mouse click
     lands on it (a programmatic element.click() is explicitly NOT the proof). */
  #approve-btn { background: #16a34a; color: #fff; font-size: 1.05rem;
                 padding: .7rem 1.4rem; pointer-events: auto; position: relative;
                 z-index: 1; }
  #approve-btn:hover { background: #15803d; }
  #approve-btn:disabled { background: #9ca3af; cursor: not-allowed; }
  .btn-deny { background: #e5e7eb; color: #111; margin-left: .5rem; }
  .hidden { display: none; }
  pre { background: #0f172a; color: #e2e8f0; padding: 1rem; border-radius: 8px;
        overflow: auto; white-space: pre-wrap; word-break: break-word; }
  .src { font-size: .9rem; color: #334155; margin: .15rem 0; }
  .src .f { color: #16a34a; font-weight: 600; }
  .status { padding: .6rem .8rem; border-radius: 7px; margin: .8rem 0;
            font-weight: 600; }
  .ok { background: #dcfce7; color: #14532d; }
  .err { background: #fee2e2; color: #7f1d1d; }
  .pill { display: inline-block; font-size: .8rem; background: #eef2ff;
          color: #3730a3; border-radius: 999px; padding: .1rem .6rem; }
  ul.specs { list-style: none; padding: 0; margin: 0; }
  ul.specs li { border: 1px solid #eee; border-radius: 7px; padding: .5rem .75rem;
                margin: .35rem 0; background: #fff; }
</style>
</head>
<body>
  <h1>Specialization — Define, Research &amp; Approve</h1>
  <p class="sub">Define a specialist, run free web-research for the &ldquo;how&rdquo;,
     then <strong>Approve</strong> to compile &amp; register it (the HITL gate —
     compile happens only on a real click).</p>

  <fieldset>
    <legend>1 &middot; Define</legend>
    <label for="name">Name (kebab-case key)</label>
    <input id="name" placeholder="markdown-author" autocomplete="off">
    <label for="description">Description (one line — the planner-facing lookup text)</label>
    <input id="description" placeholder="Writes clean, well-structured Markdown documents" autocomplete="off">
    <label for="intent">Intent (what this specialist is for)</label>
    <textarea id="intent" placeholder="Produce a detailed, well-organized Markdown report on a given topic."></textarea>
    <div style="margin-top:1rem">
      <button id="research-btn" class="btn-research" type="button" onclick="runResearch()">Run web-research</button>
    </div>
  </fieldset>

  <div id="research-status"></div>

  <fieldset id="draft-section" class="hidden">
    <legend>2 &middot; Review draft &amp; Approve <span id="draft-source" class="pill"></span></legend>
    <p id="draft-meta" class="sub"></p>
    <h3 style="margin:.4rem 0">Sources</h3>
    <div id="sources"></div>
    <h3 style="margin:1rem 0 .4rem">Condensed draft (what a sub-agent would load)</h3>
    <pre id="draft-body"></pre>
    <div style="margin-top:1rem">
      <!-- Real hit-testable Approve button: only THIS click compiles+registers. -->
      <button id="approve-btn" type="button" onclick="approve()">Approve &amp; compile</button>
      <button id="deny-btn" class="btn-deny" type="button" onclick="deny()">Discard</button>
    </div>
  </fieldset>

  <div id="approve-status"></div>

  <fieldset>
    <legend>Registered specialists <span class="pill">planner-facing index — no bodies</span></legend>
    <ul id="specs" class="specs"></ul>
  </fieldset>

<script>
let currentDraftId = null;

function setStatus(id, msg, kind) {
  const el = document.getElementById(id);
  el.innerHTML = msg ? '<div class="status ' + kind + '">' + msg + '</div>' : '';
}
function esc(s){ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function runResearch() {
  const name = document.getElementById('name').value.trim();
  const description = document.getElementById('description').value.trim();
  const intent = document.getElementById('intent').value.trim();
  if (!name) { setStatus('research-status','Name is required.','err'); return; }
  const btn = document.getElementById('research-btn');
  btn.disabled = true;
  setStatus('research-status','Researching the &ldquo;how&rdquo; over the free web tools&hellip; (bounded loop)','ok');
  try {
    const r = await fetch('/api/research', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name, description, intent})});
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || ('HTTP '+r.status));
    renderDraft(data);
    setStatus('research-status','Draft authored — review and Approve to compile. (Nothing compiled yet.)','ok');
  } catch (e) {
    setStatus('research-status','Research failed: ' + esc(String(e.message||e)), 'err');
  } finally {
    btn.disabled = false;
  }
}

function renderDraft(d) {
  currentDraftId = d.draft_id;
  document.getElementById('draft-section').classList.remove('hidden');
  document.getElementById('draft-source').textContent = 'source: ' + d.source;
  const st = d.stats || {};
  document.getElementById('draft-meta').textContent =
    d.name + ' — ' + d.description + '  ['
    + (d.queries||[]).length + ' queries, '
    + (st.fetches||0) + ' fetched, ' + (d.sources||[]).length + ' sources]';
  const srcEl = document.getElementById('sources');
  srcEl.innerHTML = (d.sources||[]).map(s =>
    '<div class="src">' + (s.fetched ? '<span class="f">[read]</span> ' : '<span>[seen]</span> ')
    + esc(s.title || s.url) + ' &mdash; <span>' + esc(s.url) + '</span></div>').join('')
    || '<div class="src">(no sources surfaced)</div>';
  document.getElementById('draft-body').textContent = d.body;
  document.getElementById('approve-btn').disabled = false;
  setStatus('approve-status','','');
}

async function approve() {
  if (!currentDraftId) { setStatus('approve-status','No draft to approve.','err'); return; }
  const btn = document.getElementById('approve-btn');
  btn.disabled = true;
  try {
    const r = await fetch('/api/approve', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({draft_id: currentDraftId})});
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || ('HTTP '+r.status));
    setStatus('approve-status','✓ Compiled &amp; registered <strong>'
      + esc(data.registered) + '</strong> at ' + esc(data.path), 'ok');
    document.getElementById('draft-section').classList.add('hidden');
    currentDraftId = null;
    loadSpecs();
  } catch (e) {
    setStatus('approve-status','Approve failed: ' + esc(String(e.message||e)), 'err');
    btn.disabled = false;
  }
}

async function deny() {
  if (!currentDraftId) return;
  await fetch('/api/deny', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({draft_id: currentDraftId})});
  document.getElementById('draft-section').classList.add('hidden');
  currentDraftId = null;
  setStatus('approve-status','Draft discarded (not compiled).','ok');
}

async function loadSpecs() {
  try {
    const r = await fetch('/api/specs');
    const data = await r.json();
    const el = document.getElementById('specs');
    el.innerHTML = (data.specs||[]).map(s =>
      '<li><strong>' + esc(s.name) + '</strong> <span class="pill">' + esc(s.source) + '</span><br>'
      + '<span class="src">' + esc(s.description) + '</span></li>').join('')
      || '<li class="src">(none registered yet)</li>';
  } catch (e) { /* non-fatal */ }
}

loadSpecs();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# The HTTP handler / server
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    # The app is attached to the server instance (see make_server).
    server_version = "SpecUI/1.0"

    @property
    def app(self) -> SpecUIApp:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
        pass

    # ---- response helpers ---- #
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):  # pragma: no cover
            pass

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    # ---- routes ---- #
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/specs":
            self._json(200, self.app.specs())
        elif self.path in ("/healthz", "/api/health"):
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": f"not found: {self.path}"})

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/api/research":
                result = self.app.research(
                    payload.get("name", ""),
                    payload.get("description", ""),
                    payload.get("intent", ""),
                    source=payload.get("source", SOURCE_UI),
                )
                self._json(200, result)
            elif self.path == "/api/approve":
                result = self.app.approve(payload.get("draft_id", ""))
                self._json(200, result)
            elif self.path == "/api/deny":
                result = self.app.deny(payload.get("draft_id", ""))
                self._json(200, result)
            else:
                self._json(404, {"error": f"not found: {self.path}"})
        except KeyError as exc:
            self._json(404, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - surface as JSON, never crash the server
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def make_server(
    specs_dir: Union[str, Path],
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    hook_factory: Optional[HookFactory] = None,
) -> ThreadingHTTPServer:
    """Build (but do not start) the in-process define-UI HTTP server.

    The returned :class:`ThreadingHTTPServer` carries the :class:`SpecUIApp` on
    its ``.app`` attribute. Call ``.serve_forever()`` to run it, or drive it
    request-by-request in a test. ``hook_factory`` defaults to the real free
    no-key web-research hook; pass a stub for a net-free self-check."""
    app = SpecUIApp(specs_dir, hook_factory=hook_factory)
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.app = app  # type: ignore[attr-defined]
    return httpd


__all__ = [
    "SpecUIApp",
    "make_server",
    "PAGE_HTML",
    "HookFactory",
    "SOURCE_UI",
    "SOURCE_AUTONOMOUS",
]
