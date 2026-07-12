"""d221 — the TERMINAL SYNTHESIZER STREAMS its summary to the shell.

The synthesizer publishes ordered ``agent_run_synthesis_delta`` frames (so a streaming-aware
shell renders the summary progressively) followed by the terminal ``agent_run_synthesis`` event
carrying the FULL summary + the downloadable artifact (back-compat for the held UI). These
tests cover the pure chunker (byte-faithful split) and the streamed publish order/content.
"""
from __future__ import annotations

import asyncio

from chat_app.agentic import (
    EVENT_RUN_SYNTHESIS,
    EVENT_RUN_SYNTHESIS_DELTA,
    _chunk_summary_for_stream,
    _run_terminal_synthesizer,
)


class _RecordingPlane:
    """A minimal async event plane that records every published (kind, payload)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def publish(self, kind, payload, source=None):  # noqa: D401 - test double
        self.events.append((kind, dict(payload)))


class _Span:
    def set_attribute(self, *_a, **_k):
        pass


def test_chunker_is_byte_faithful_and_bounded():
    summary = (
        "Report complete for: the ongoing US-Iran conflict. Authored 4 section(s) grounded "
        "in 8 source(s) across 2 plan(s) (research → write). Your report is ready to "
        "download: us_iran_report.html."
    )
    chunks = _chunk_summary_for_stream(summary, max_chars=48)
    assert len(chunks) > 1                       # genuinely streamed in multiple frames
    assert "".join(chunks) == summary           # concatenation reproduces it byte-for-byte


def test_chunker_handles_short_and_empty():
    assert _chunk_summary_for_stream("") == []
    assert "".join(_chunk_summary_for_stream("short summary")) == "short summary"


def test_synthesizer_streams_deltas_then_terminal_event():
    plane = _RecordingPlane()
    summary = asyncio.run(_run_terminal_synthesizer(
        plane=plane, query="the ongoing US-Iran conflict",
        out_name="us_iran_report.html", sources=[{"url": "u1"}, {"url": "u2"}],
        write_dag=None, plans_authored=["research", "write"], span=_Span(),
    ))
    kinds = [k for k, _ in plane.events]
    # Deltas come FIRST, the terminal synthesis event LAST.
    assert kinds[-1] == EVENT_RUN_SYNTHESIS
    assert all(k == EVENT_RUN_SYNTHESIS_DELTA for k in kinds[:-1])
    assert kinds.count(EVENT_RUN_SYNTHESIS) == 1
    assert kinds.count(EVENT_RUN_SYNTHESIS_DELTA) >= 1

    # Delta frames are ordered and rebuild the streamed summary.
    deltas = [p for k, p in plane.events if k == EVENT_RUN_SYNTHESIS_DELTA]
    assert [d["seq"] for d in deltas] == list(range(len(deltas)))
    assert "".join(d["delta"] for d in deltas) == summary

    # Terminal event carries the full summary + the downloadable artifact.
    _, term = plane.events[-1]
    assert term["summary"] == summary
    assert term["streamed"] is True
    assert term["artifact"]["name"] == "us_iran_report.html"
