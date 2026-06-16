"""chat_app — the single-process ASGI web app for ReactiveAgents.

A uv WORKSPACE MEMBER (d11) resolving into the single shared root .venv so it
imports every sibling component in the SAME interpreter (d2 — one in-process
process). The composed stack lives on :data:`chat_app.app.app`.

Public surface
--------------
- :data:`app`           — the wired ASGI ``FastAPI`` instance
- :func:`create_app`    — build a fresh app (used by tests / alternate mounts)
- :class:`Wiring`       — the shared in-process composition (event plane, tool
  hook, memory, specialization registry+engine, planner+runtime)
- :func:`build_wiring`  — compose the whole stack on the stub transport (d12)
"""
from __future__ import annotations

from chat_app.app import Wiring, app, build_wiring, create_app

__all__ = ["app", "create_app", "Wiring", "build_wiring"]
