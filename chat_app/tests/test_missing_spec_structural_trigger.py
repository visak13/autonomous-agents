"""s10-a8 — the STRUCTURAL scenario-3 missing-specialist trigger, through run_agentic.

Scenario 3 (a request needing a specialist no registered spec provides → notify +
SSE-fallback / define-and-resume) previously fired only on a per-node ``needs_spec``
free-text the 4.6B model would not reliably volunteer (s10-a4), so it never fired
live. The re-architecture moves the TRIGGER to a DETERMINISTIC registry-membership
check on the shape selector's reliable spec-name extraction: the new free-string
``unmet_specs`` lets the model NAME a specialization the registry does not have, and
``run_agentic`` fires the (unchanged) notify/pause when ``name not in registry``.

These tests drive the REAL ``run_agentic`` on the deterministic ``FakeTransport``
(no GPU / live model): one scripted reply for the shape-selection call, one for the
single-node incremental authoring call. ``skip_ambiguity=True`` bypasses the
pre-selection ambiguity call so the scripted replies line up. The trigger is generic
— a set-membership check, no scenario/keyword/topic matching.
"""
from __future__ import annotations

import asyncio
import json

from llm_framework import FakeTransport
from reactive_tools import EventPlane

from chat_app.agentic import run_agentic
from chat_app.app import build_wiring


def _selector_reply(shape: str, *, unmet_specs=None, requested_specs=None) -> str:
    return json.dumps(
        {
            "shape": shape,
            "rationale": "fits",
            "search_allowed": True,
            "requested_specs": list(requested_specs or []),
            "wants_file": False,
            "unmet_specs": list(unmet_specs or []),
        }
    )


def _node_reply(task: str, *, tool: str = "", spec: str = "", more: bool = False) -> str:
    return json.dumps(
        {
            "task": task,
            "spec": spec,
            "specs": [],
            "needs_spec": "",
            "tool": tool,
            "depends_on": [],
            "more": more,
        }
    )


def _run(transport, query, **kw):
    w = build_wiring()
    try:
        # the requested specialist must genuinely be absent for the trigger to be real.
        assert "forensic-accountant" not in set(w.registry.names())
        return asyncio.run(
            run_agentic(
                query,
                transport=transport,
                registry=w.registry,
                hook=w.hook,
                plane=EventPlane(),
                skip_ambiguity=True,
                **kw,
            )
        )
    finally:
        w.close()


def test_unregistered_requested_spec_fires_missing_specialist_pause():
    # The model names a specialization (forensic-accountant) that is NOT registered.
    # run_agentic must PAUSE with the missing-specialist notify — never run the node
    # spec-less — and attach the unmet need to the DAG's sink (answer) node.
    transport = FakeTransport(
        [
            _selector_reply("linear", unmet_specs=["forensic-accountant"]),
            _node_reply("write the forensic accounting report", more=False),
        ]
    )
    res = _run(transport, "write me a forensic-accountant report on this filing")

    assert res.missing_specialist is True
    assert res.ok is False
    pending = res.pending or {}
    assert pending.get("choices") == ["sse_fallback", "define_and_resume"]
    needs = " ".join(m.get("needs", "") for m in pending.get("missing", []))
    assert "forensic-accountant" in needs
    # the need is attached to a real DAG node (so a define-and-resume can stamp the
    # newly-defined spec there) — the sink/terminal node of the authored plan.
    flagged = {m["node_id"] for m in pending["missing"]}
    sink_ids = {
        n.id
        for n in res.dag.nodes
        if n.id not in {d for m in res.dag.nodes for d in m.depends_on}
    }
    assert flagged & sink_ids


def test_deep_research_is_suppressed_when_a_spec_is_unavailable():
    # Even if the model routes the request to the inherently-fileless/streamed
    # deep-research shape, an unavailable requested spec forces the ACYCLIC path and
    # the pause — deep-research is never run spec-less for a missing-specialist need.
    transport = FakeTransport(
        [
            _selector_reply("deep-research", unmet_specs=["forensic-accountant"]),
            _node_reply("research and report on the filing", more=False),
        ]
    )
    res = _run(transport, "do a deep forensic-accountant investigation of the filing")

    assert res.missing_specialist is True
    assert res.deep_research is None  # the deep-research path was NOT taken
    assert res.dag is not None        # an acyclic DAG was authored to attach the need


def test_registered_requested_spec_does_not_false_fire():
    # CONTRAST: the model names a spec that IS registered (markdown-writer) in
    # unmet_specs — the deterministic membership check drops it, so NO missing-spec
    # pause fires (the run proceeds to drive the node, not pause).
    w = build_wiring()
    try:
        assert "markdown-writer" in set(w.registry.names())
        transport = FakeTransport(
            [
                _selector_reply("linear", unmet_specs=["markdown-writer"]),
                _node_reply("write the overview", more=False),
                # the run proceeds to execute the single node; a generous tail of
                # produce/verify replies keeps the deterministic transport fed.
                *["the overview text" for _ in range(8)],
            ]
        )
        res = asyncio.run(
            run_agentic(
                "write an overview using the markdown-writer specialization",
                transport=transport,
                registry=w.registry,
                hook=w.hook,
                plane=EventPlane(),
                skip_ambiguity=True,
            )
        )
        # a registered name is NOT a missing specialist → no pause.
        assert res.missing_specialist is False
    finally:
        w.close()
