"""Tests for the explicit-arm workflow route (chat_app.routes /workflows, s9/a4).

Proves the Scenario-B arming surface WITHOUT live phi and WITHOUT real SMTP:
- POST /workflows arms an already-registered spec as a scheduled workflow,
  layering the request's schedule + delivery onto it (a chat-defined ruleset
  becomes a "daily brief"-style workflow);
- the in-process scheduler FIRES it (one-shot, no delay) and the fire delivers
  via the recipient-LOCKED send_mail tool — here a FAKE recorder, so NO real
  network/mail;
- GET /workflows surfaces the job's last_result (delivered + smtp_code +
  message_id) so a fired send is provable from a poll alone;
- arming an unregistered spec is a clean 404.

The app is built in STUB mode (default — no REACTIVE_AGENTS_LIVE), so the live
Gemma-4 producer is not exercised here (that is proven live under scenarioB/);
the stub producer drives the schedule->produce->deliver path deterministically.
The real send_mail on the wiring hook is REPLACED with a fake recorder so the
test never sends mail. d8 (b5): delivery is recipient-locked — the spec's
recipient is ignored and the fire always delivers to the user's own address.
"""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from specialization import CompiledSpec

from chat_app.app import create_app


LOCKED_ADDRESS = "self@example.com"


def _fake_email_hook(app) -> list[dict]:
    """Replace the wiring hook's send_mail with a fake recorder (no network).

    The fake mirrors the REAL recipient-locked ``send_mail`` signature (subject +
    body only, NO ``to``) so the recipient is always the locked own-address."""
    sent: list[dict] = []

    def fake_send_mail(*, subject, body):
        sent.append({"subject": subject, "body": body, "to": LOCKED_ADDRESS})
        return {
            "ok": True,
            "to": LOCKED_ADDRESS,
            "subject": subject,
            "smtp_code": 250,
            "message_id": "<fake-route@local>",
            "bytes": len(body),
        }

    app.state.wiring.hook.register("send_mail", fake_send_mail, description="fake")
    return sent


def _register_brief(app) -> None:
    """Register a daily-brief output-shaping spec (no schedule yet — armed later)."""
    app.state.wiring.registry.register(
        CompiledSpec(
            name="daily-brief",
            description="a concise daily brief, markdown-shaped",
            source="seed",
            body="Structure the findings with a heading, bullets, and a short summary.",
        )
    )


def test_arm_one_shot_fires_and_delivers(tmp_path) -> None:
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        sent = _fake_email_hook(app)
        _register_brief(app)

        resp = client.post(
            "/workflows",
            json={
                "spec_name": "daily-brief",
                "schedule": {"kind": "one_shot", "initial_delay": 0.0},
                "delivery": {"channel": "email", "recipient": "self@example.com"},
            },
        )
        assert resp.status_code == 201, resp.text
        job_id = resp.json()["job_id"]
        assert resp.json()["transport"] == "stub"

        # one-shot, no delay — poll GET /workflows until it has fired AND its
        # result is populated. (The scheduler bumps fire_count at fire START, so
        # polling on fire_count alone races the fire body; wait for last_result.)
        deadline = time.time() + 5.0
        snap = None
        while time.time() < deadline:
            jobs = client.get("/workflows").json()["jobs"]
            snap = next((j for j in jobs if j["job_id"] == job_id), None)
            if snap and snap["fire_count"] >= 1 and snap["last_result"] is not None:
                break
            time.sleep(0.05)
        assert snap is not None and snap["fire_count"] == 1, snap
        assert snap["last_result"]["delivered"] is True
        assert snap["last_result"]["smtp_code"] == 250
        assert snap["last_result"]["message_id"] == "<fake-route@local>"
        assert len(sent) == 1
        assert sent[0]["to"] == LOCKED_ADDRESS
        assert "daily-brief" in sent[0]["body"]


def test_arm_unregistered_spec_is_404(tmp_path) -> None:
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        _fake_email_hook(app)
        resp = client.post(
            "/workflows",
            json={"spec_name": "no-such-spec", "schedule": {"kind": "one_shot"}},
        )
        assert resp.status_code == 404
