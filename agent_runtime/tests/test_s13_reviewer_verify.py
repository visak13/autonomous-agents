"""s13 / P1-reviewer (FIX-C, d114 + d118) — the deep-research VERIFY LANE reworked into a
SPEC-AWARE GENERIC REVIEWER with proper file tools, and its VERDICT migrated to NATIVE
Ollama tool-calls.

Two folded-together changes, proven FAST + OFFLINE (no GPU, no network):

* (A) SPEC-AWARE: the verify lane is no longer a spec-BLIND fixed fact-checker — when it
  receives the WORKER'S SAME SPEC it injects a REVIEW SPEC block + a reviewer directive and
  flags a SPEC violation the same structured way it flags an unbacked claim. With NO spec the
  prompt is byte-identical to the legacy fact-checker (the default / verify-lane-OFF path).
* (A) FILE TOOLS: the reviewer is given proper READ / WRITE / UPDATE file tools — a real
  ``file_update`` surgical-edit tool (new, in reactive_tools) PLUS native tool schemas; the
  ``file_update`` tool is exercised end-to-end through the real hook.
* (B) NATIVE VERDICT: the section_verify VERDICT now rides the model's OWN
  ``message.tool_calls`` channel (the SAME native helper the decision loop uses), so LEADING
  PROSE can never swallow it (the d111/d112 prose-drop is architecturally impossible on
  native). The balanced-brace string parser is KEPT as the defensive fallback for a
  non-native reply (d117 condition 2). Proven both as a unit and through the SERVED runtime
  section-verify seam.

Guardrails honoured: content stays RAW (the RAW revise turn is text, never a tool call — d50);
no flags (native is the flag-free default — d65); the fallback parse logic is NOT deleted; the
verify-lane-OFF path is byte-identical (no-regression).
"""
from __future__ import annotations

import asyncio

from agent_runtime.claim_verify import (
    REVIEWER_FILE_TOOL_SPECS,
    REVIEWER_TOOL_SPECS,
    VERIFY_VERDICT_TOOL,
    parse_verify_verdict,
    verdict_from_native_args,
    verify_claims,
)
from reactive_tools import EventPlane, ToolHook, register_agentic_tools


def _run(coro):
    return asyncio.run(coro)


_SOURCES = [
    {"title": "UN News", "url": "https://news.un.org/x", "source_trust": "secondary",
     "key_claims": ["180 missiles fired on June 14"], "markdown": "180 missiles fired."},
]


# =========================================================================== #
# (A) SPEC-AWARE — the verify lane receives + APPLIES the worker spec
# =========================================================================== #
_SPEC = "RULE: every report section MUST end with a 'Sources' list citing each URL."


def test_verify_lane_injects_and_applies_worker_spec():
    """With a worker spec the verify prompt carries a REVIEW SPEC block + a reviewer
    directive, and the reviewer flags a SPEC violation the same structured way it flags an
    unbacked claim — so the same spec that SHAPED the deliverable now GRADES it."""
    seen: list[str] = []

    async def _spec_aware(prompt: str) -> str:
        seen.append(prompt)
        # the reviewer applies the spec: flag the missing Sources list as an issue
        if "Sources" in prompt and "REVIEW SPEC" in prompt:
            return ('{"verdict":"revise","unbacked":[{"claim":"(no Sources list)",'
                    '"reason":"spec rule: each section must end with a Sources list"}]}')
        return '{"verdict":"ok"}'

    res = _run(verify_claims(
        "Iran fired 180 missiles on June 14.", _SOURCES, verify=_spec_aware, spec=_SPEC))
    # the spec text + the generic-reviewer directive both reached the prompt
    assert "REVIEW SPEC" in seen[0] and _SPEC in seen[0]
    assert "REVIEWER for this deliverable" in seen[0]
    # the spec violation was flagged (the lane is no longer spec-blind)
    assert res.grounded is False
    assert "spec rule" in res.unbacked[0].reason.lower()


def test_verify_lane_no_spec_is_byte_identical_fact_checker():
    """No spec → the prompt is the legacy spec-BLIND fact-checker: no REVIEW SPEC block, no
    reviewer directive (the default / verify-lane-OFF path stays unchanged)."""
    seen: list[str] = []

    async def _fake(prompt: str) -> str:
        seen.append(prompt)
        return '{"verdict":"ok"}'

    _run(verify_claims("Iran fired 180 missiles on June 14.", _SOURCES, verify=_fake))
    assert "REVIEW SPEC" not in seen[0]
    assert "REVIEWER for this deliverable" not in seen[0]
    # the classic fact-check framing is intact
    assert "FETCHED SOURCES" in seen[0] and "REPORT TO FACT-CHECK" in seen[0]


# =========================================================================== #
# (A) FILE TOOLS — read/write/update available + the update tool exercised
# =========================================================================== #
def test_reviewer_tool_specs_expose_read_write_update_and_verdict():
    """The reviewer's tool surface offers proper file READ / WRITE / UPDATE tools PLUS the
    structured verdict — each as a well-formed native schema."""
    names = {s["function"]["name"] for s in REVIEWER_TOOL_SPECS}
    assert {"file_read", "file_write", "file_update", VERIFY_VERDICT_TOOL} <= names
    file_names = {s["function"]["name"] for s in REVIEWER_FILE_TOOL_SPECS}
    assert file_names == {"file_read", "file_write", "file_update"}
    for s in REVIEWER_TOOL_SPECS:
        fn = s["function"]
        assert s["type"] == "function"
        assert isinstance(fn["parameters"]["properties"], dict)
        assert isinstance(fn["parameters"]["required"], list)


def test_file_update_tool_registered_and_exercised(tmp_path):
    """The new ``file_update`` tool is registered (selectable + dispatchable) and actually
    UPDATES a file in place through the real hook — the reviewer's ground-or-remove edit. A
    missing 'old' span is REFUSED (an honest no-match, never a silent no-op)."""
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)

    _run(hook.invoke(
        "file_write", path="r.md",
        content="Under 17 USC 107(5) the strike was legal. Real grounded text stays here."))
    # UPDATE: remove the fabricated span in place (empty 'new' = delete)
    res = _run(hook.invoke(
        "file_update", path="r.md",
        old="Under 17 USC 107(5) the strike was legal. ", new=""))
    assert res.ok and res.value["replaced"] == 1 and res.value["removed"] is True
    on_disk = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "17 USC 107(5)" not in on_disk and "Real grounded text stays here." in on_disk
    # a non-matching 'old' is refused (ok=False), not silently applied
    miss = _run(hook.invoke("file_update", path="r.md", old="no such span", new="x"))
    assert miss.ok is False


def test_file_update_replace_all_and_substitution(tmp_path):
    """count=0 replaces ALL occurrences; a non-empty 'new' substitutes (ground-or-correct)."""
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    _run(hook.invoke("file_write", path="r.md", content="X then X then X."))
    res = _run(hook.invoke("file_update", path="r.md", old="X", new="Y", count=0))
    assert res.ok and res.value["replaced"] == 3 and res.value["removed"] is False
    assert (tmp_path / "r.md").read_text(encoding="utf-8") == "Y then Y then Y."


# =========================================================================== #
# (B) NATIVE VERDICT — drop-immune to leading prose + the kept fallback
# =========================================================================== #
def test_native_verdict_dispatched_even_with_leading_prose():
    """The structured verdict rides the NATIVE ``message.tool_calls`` channel, so a turn that
    LEADS WITH PROSE still delivers the verdict — where the OLD string parser drops it."""
    prose = "Let me review this report against the sources and the review spec first."
    verdict_call = [{
        "name": VERIFY_VERDICT_TOOL,
        "arguments": {"verdict": "revise",
                      "unbacked": [{"claim": "17 USC 107(5)", "reason": "no fetched source"}]},
    }]

    async def _native(_prompt: str):
        return prose, verdict_call

    res = _run(verify_claims(
        "Under 17 USC 107(5) the strike was legal.", _SOURCES, verify_native=_native))
    assert res.parsed is True and res.grounded is False
    assert res.unbacked[0].claim == "17 USC 107(5)"
    # CONTRAST: the prose-only text would be DROPPED by the kept string parser (parsed False)
    assert parse_verify_verdict(prose).parsed is False


def test_native_verdict_ok_is_grounded():
    """A native ``ok`` verdict (no flagged claims) → grounded, even with co-emitted prose."""
    async def _native(_prompt: str):
        return "Looks fully grounded to me.", [
            {"name": VERIFY_VERDICT_TOOL, "arguments": {"verdict": "ok"}}]

    res = _run(verify_claims("180 missiles were fired on June 14.", _SOURCES, verify_native=_native))
    assert res.parsed is True and res.grounded is True and res.unbacked == []


def test_native_verdict_falls_back_to_balanced_brace_on_non_native_reply():
    """When the reply carries NO native tool_call (a non-native path) the verdict is recovered
    from the reply TEXT by the kept balanced-brace parser — the d117(2) defensive fallback."""
    async def _non_native(_prompt: str):
        # leading prose + a JSON verdict in the text, NO tool_calls channel
        return ('Here is my verdict: {"verdict":"revise",'
                '"unbacked":[{"claim":"CTEA 1998","reason":"unsourced"}]}'), None

    res = _run(verify_claims("Enacted by the CTEA of 1998.", _SOURCES, verify_native=_non_native))
    assert res.parsed is True and res.grounded is False
    assert res.unbacked[0].claim == "CTEA 1998"


def test_verdict_from_native_args_matches_string_parse_shape():
    """A native verdict and a fallback string verdict yield the SAME VerifyResult shape."""
    args = {"verdict": "revise", "unbacked": [{"claim": "c", "reason": "r"}]}
    native = verdict_from_native_args(args)
    parsed = parse_verify_verdict('{"verdict":"revise","unbacked":[{"claim":"c","reason":"r"}]}')
    assert native.grounded == parsed.grounded is False
    assert [u.claim for u in native.unbacked] == [u.claim for u in parsed.unbacked] == ["c"]
    # the 'issues' key is accepted as an alias for the flagged list
    alt = verdict_from_native_args({"verdict": "revise", "issues": [{"claim": "c2"}]})
    assert alt.grounded is False and alt.unbacked[0].claim == "c2"


# RP-3c (d330): the SERVED runtime section-verify INTEGRATION test is RETIRED with the
# engine verify/revise lane it drove (``verify_lane`` + ``verify_and_revise`` are gone). The
# NATIVE-verdict dispatch it proved is still covered by the ``verdict_from_native_args`` unit
# above and the model-driven ``verify_claims`` tests; the no-fab self-review now lives in the
# writer doctrine (proven by the RP-3c self-policing test + the bounded live measure).
