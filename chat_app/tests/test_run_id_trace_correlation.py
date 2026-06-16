"""Regression: the decoupled /runs path threads its RunManager run id into the
agentic run (s6/b5 review-leg fix).

THE DEFECT this locks: ``POST /chats/{id}/runs`` builds the run body closure and
hands it to :class:`RunManager`, which mints the ``run-xxxx`` id. Before the b5
fix that id was never threaded into ``run_agentic`` — so ``run_agentic`` saw
``run_id=None``, the ``agent.session`` span carried NO ``session.run_id`` and the
``agent.run`` span's ``run.id`` silently fell back to the 32-hex *trace id*
(observed in the b4 live capture). A Phoenix trace was therefore NOT correlatable
back to the ``/runs/{run_id}`` the client polls.

This proves, fully OFFLINE (no live phi / Ollama / Phoenix), that the id the
client receives from ``POST /runs`` is the SAME id ``run_agentic`` is invoked
with — i.e. the value that lands in ``session.run_id`` / ``agent.run.run.id``.
The span-level mapping (run_id -> agent.run.run.id) is already locked by
``agent_runtime/tests/test_tracing_span_tree.py::...`` (asserts
``run.id == "run-test-b2"``); together they cover the whole correlation chain.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

import chat_app.routes as routes
from chat_app.agentic import AgenticResult
from chat_app.app import create_app


def test_runs_path_threads_run_id_into_agentic(tmp_path, monkeypatch) -> None:
    app = create_app(data_dir=tmp_path)

    captured: dict[str, object] = {}

    async def _spy_run_agentic(topic, **kwargs):  # mirrors run_agentic's signature
        # Record the run_id the route threaded in; short-circuit before any phi.
        captured["run_id"] = kwargs.get("run_id")
        fake_result = SimpleNamespace(
            results={}, states={}, ok=True, launch_order=[]
        )
        return AgenticResult(
            dag=None, result=fake_result, md_report=None, html_report=None
        )

    # Force the live agentic branch in _execute_message_run without any real phi:
    # flip the wiring to "live" (s3/b2: the chat route now gates ONLY on live
    # mode, no longer on both_specs_registered) and swap run_agentic for the spy.
    monkeypatch.setattr(routes, "run_agentic", _spy_run_agentic)

    with TestClient(app) as client:
        app.state.wiring.transport_mode = "live"

        rec = client.post("/chats", json={}).json()
        chat_id = rec.get("chat_id") or rec.get("id")
        assert chat_id, rec

        resp = client.post(f"/chats/{chat_id}/runs", json={"message": "JWST"})
        assert resp.status_code == 202, resp.text
        run_id = resp.json()["run_id"]
        assert run_id.startswith("run-")

        # The run executes as a background RunManager task; poll its status (each
        # GET yields the event loop so the task can run) until terminal.
        for _ in range(50):
            status = client.get(f"/runs/{run_id}").json()["status"]
            if status in ("done", "failed", "cancelled"):
                break
            time.sleep(0.02)
        assert status == "done", status

    # THE ASSERTION: the id the client got from POST /runs is exactly the id
    # run_agentic (and thus the session/agent.run spans) was driven with — NOT
    # None, NOT a trace-id fallback.
    assert captured["run_id"] == run_id
