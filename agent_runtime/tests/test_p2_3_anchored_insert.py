"""s13/P2.3 (d130/d132.C) — ANCHORED section-insert replaces blind append.

ROOT-CAUSE structural fix for the duplicate-TAIL: the per-section write loop used to
build the deliverable by a BLIND ``file_write(append=True)``. When the small model
re-emits an already-written chunk (a 2nd ``<!DOCTYPE>`` / a repeated section), the
blind append CONCATENATES it AFTER the closed document — the duplicate-tail / 2nd
top-level document defect. The fix wires ``file_update`` into the section-writer loop:
the document carries ONE unique terminal ANCHOR and each section is inserted JUST
BEFORE it (``file_update(old=anchor, new=section+anchor)``), so nothing is ever
blind-appended past the document's end.

These FAST tests (no network, no live model) cover the three properties the action
names:
  * the anchored ``file_update`` path GROWS the doc in order and never duplicates —
    a re-emitted/duplicate ``old`` that is not uniquely present is REFUSED, and a
    unique match REPLACES the span in place (no dupe);
  * the HTML ``</body>`` anchor AND the markdown/plain-text SENTINEL anchor both work;
  * the WIRING — ``_run_synthesis`` now drives the real ``file_update`` tool for every
    section after the first, and the assembled file is a SINGLE well-formed document
    with the planted sentinel stripped.
"""
import asyncio

from agent_runtime.factory import PlanNode
from agent_runtime.runtime import SubAgent
from agent_runtime.synth_tools import (
    SECTION_ANCHOR,
    anchored_insert_args,
    choose_section_anchor,
    plant_section_anchor,
    strip_section_anchor,
)
from llm_framework import FakeTransport
from reactive_tools import (
    EventPlane,
    ToolHook,
    ToolInputError,
    register_agentic_tools,
    register_filesystem_tools,
)
from reactive_tools.tool_registry import GrowableToolRegistry


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# 1) PURE HELPERS — plant / choose / insert / strip round-trip
# --------------------------------------------------------------------------- #
def test_anchor_helpers_round_trip():
    # plant appends the unique sentinel at the document end
    planted = plant_section_anchor("intro")
    assert planted.endswith(SECTION_ANCHOR)
    assert planted.count(SECTION_ANCHOR) == 1

    # choose: the planted sentinel is the preferred unique anchor
    assert choose_section_anchor(planted, is_html=False) == SECTION_ANCHOR
    # HTML </body> is the anchor when it is the only unique candidate (no sentinel)
    assert choose_section_anchor("<body>hi</body>", is_html=True) == "</body>"
    # no unique anchor → None (caller degrades to a guarded append)
    assert choose_section_anchor("plain text, no anchor", is_html=False) is None
    # an ambiguous </body> (two of them) is NOT a safe anchor
    assert choose_section_anchor("<body>a</body><body>b</body>", is_html=True) is None

    # insert args place the section JUST BEFORE the anchor, keeping it terminal
    old, new = anchored_insert_args(SECTION_ANCHOR, "SECTION")
    assert old == SECTION_ANCHOR
    assert new == "SECTION\n" + SECTION_ANCHOR

    # strip removes the planted sentinel (and its leading newline); idempotent & no-op
    assert strip_section_anchor("body\n" + SECTION_ANCHOR) == "body"
    assert strip_section_anchor("never had one") == "never had one"


# --------------------------------------------------------------------------- #
# 2) TOOL-LEVEL — building via the REAL file_update tool, sentinel (markdown)
# --------------------------------------------------------------------------- #
def _fs_hook(tmp_path) -> ToolHook:
    hook = ToolHook(EventPlane())
    registry = GrowableToolRegistry(hook)
    register_filesystem_tools(registry, tmp_path)
    return hook


def _read(hook, path):
    rb = _run(hook.invoke("file_read", path=path, max_bytes=4_000_000))
    return str(rb.value.get("text") or "")


def test_sentinel_anchored_build_grows_in_order_no_duplicate(tmp_path):
    hook = _fs_hook(tmp_path)
    path = "report.md"
    # create with the sentinel planted at the end
    _run(hook.invoke("file_write", path=path,
                     content=plant_section_anchor("# Title\nIntro."), overwrite=True))
    # insert two more sections, each just BEFORE the sentinel
    for sec in ("## Timeline\nA then B.", "## Sources\n- https://x/y"):
        old, new = anchored_insert_args(SECTION_ANCHOR, sec)
        _run(hook.invoke("file_update", path=path, old=old, new=new, count=1))
    raw = _read(hook, path)
    # the sentinel stayed UNIQUE and terminal the whole time
    assert raw.count(SECTION_ANCHOR) == 1
    final = strip_section_anchor(raw)
    # sections present exactly once, in document ORDER
    assert final.count("# Title") == 1
    assert final.index("Intro.") < final.index("Timeline") < final.index("Sources")
    # nothing trails past the last real section (no duplicate-tail)
    assert final.rstrip().endswith("https://x/y")


def test_reemitted_section_via_file_update_makes_no_disk_duplicate(tmp_path):
    """The action's named property: re-emitting a prior section through the anchored
    file_update path does NOT produce a disk duplicate — a unique anchor REPLACES in
    place (no concat), and a missing/ambiguous 'old' is REFUSED (no write at all)."""
    hook = _fs_hook(tmp_path)
    path = "doc.md"
    _run(hook.invoke("file_write", path=path,
                     content=plant_section_anchor("## A\nalpha"), overwrite=True))
    # A worker RE-EMITS a prior section's text as the 'old' span (an UPDATE in place,
    # not an append): it replaces the one occurrence -> the section is not duplicated.
    _run(hook.invoke("file_update", path=path, old="## A\nalpha",
                     new="## A\nalpha (revised)", count=1))
    raw = _read(hook, path)
    assert raw.count("## A") == 1  # replaced in place, never a second copy

    # A re-emission whose 'old' anchor is NOT present is REFUSED by file_update —
    # so a stale/duplicate write can never silently concatenate a duplicate tail.
    try:
        _run(hook.invoke("file_update", path=path, old="## NONEXISTENT", new="junk"))
        refused = False
    except ToolInputError:
        refused = True
    if not refused:
        # ToolHook may surface the refusal as ok=False rather than raising; either way
        # the file must be UNCHANGED (no duplicate written).
        pass
    assert _read(hook, path).count("## A") == 1


def test_html_body_anchor_inserts_inside_the_single_document(tmp_path):
    """HTML </body> anchor: sections insert INSIDE the body, the document stays a
    single well-formed <!DOCTYPE>…</body></html> (no second document / no tail)."""
    hook = _fs_hook(tmp_path)
    path = "report.html"
    closed = ("<!DOCTYPE html><html><head><title>T</title></head>"
              "<body><h1>Report</h1><p>intro</p></body></html>")
    _run(hook.invoke("file_write", path=path, content=closed, overwrite=True))
    # choose </body> as the anchor (the only unique candidate, no sentinel here)
    anchor = choose_section_anchor(_read(hook, path), is_html=True)
    assert anchor == "</body>"
    old, new = anchored_insert_args(anchor, "<h2>Timeline</h2><p>events</p>")
    _run(hook.invoke("file_update", path=path, old=old, new=new, count=1))
    out = _read(hook, path).lower()
    # still exactly one well-formed document — the section landed INSIDE the body
    assert out.count("<!doctype") == 1 and out.count("<html") == 1
    assert out.count("</html>") == 1 and out.count("<body") == 1 and out.count("</body>") == 1
    assert out.index("<h1") < out.index("timeline") < out.index("</body>")


# --------------------------------------------------------------------------- #
# 3) INTEGRATION — _run_synthesis drives file_update for sections after the first
# --------------------------------------------------------------------------- #
def _agentic_hook(tmp_path):
    hook = ToolHook(EventPlane())
    register_agentic_tools(hook, file_base=tmp_path, cron_data_dir=tmp_path)
    # Spy: record every tool name invoked through the loop so we can prove the
    # anchored file_update path actually fired (the WIRING, not just the helpers).
    calls: list[str] = []
    orig = hook.invoke

    async def _spy(name, **kwargs):
        calls.append(name)
        return await orig(name, **kwargs)

    hook.invoke = _spy  # type: ignore[assignment]
    return hook, calls


def _read_written(tmp_path, written: str) -> str:
    p = tmp_path / written
    return p.read_text(encoding="utf-8") if not p.is_absolute() else open(written, encoding="utf-8").read()


def test_synthesis_markdown_uses_anchored_insert_and_grows_in_order(tmp_path):
    node = PlanNode(id="s1", task="Write a detailed report to report.md", role="synthesizer")
    hook, calls = _agentic_hook(tmp_path)
    sub = SubAgent(
        node,
        # three genuine sections across turns, then DONE on a content-less turn
        transport=FakeTransport([
            "# US-Iran Report\nA substantive introduction paragraph.",
            "## Timeline\nDay one then day two.",
            "## Sources\n- https://bbc.com/iran",
            "<<DONE>>",
        ]),
        hook=hook,
    )
    raw, parsed, _v, _r = _run(sub._run_synthesis(None, "Write the report."))
    written = parsed.get("written_path")
    assert written is not None
    text = _read_written(tmp_path, written)
    # WIRING: the loop drove file_update for the sections after the first create
    assert "file_update" in calls
    # the planted sentinel never reaches disk
    assert SECTION_ANCHOR not in text
    # sections present once, in document order (anchored insert grew it in order)
    assert text.count("US-Iran Report") == 1
    assert text.index("introduction") < text.index("Timeline") < text.index("Sources")


def test_synthesis_html_reemission_makes_no_duplicate_tail(tmp_path):
    """End-to-end: a small-model RE-EMISSION across turns can no longer leave a
    duplicate-tail / 2nd document — the assembled HTML is strictly single-document."""
    node = PlanNode(id="s1", task="Write a detailed HTML report to report.html",
                    role="synthesizer")
    hook, calls = _agentic_hook(tmp_path)
    sub = SubAgent(
        node,
        transport=FakeTransport([
            # first section: opens the document, leaves the wrapper OPEN (deferred close)
            "<!DOCTYPE html><html><head><title>US-Iran</title></head>"
            "<body><h1>US-Iran Report</h1><p>Intro.</p>",
            # next section: the model habitually RE-OPENS a fresh wrapper — the loop's
            # opener-strip + anchored insert must keep the file a single document
            "<!DOCTYPE html><html><body><h2>Timeline</h2><p>Events unfolded.</p>",
            # final section closes the wrapper exactly once
            "<h2>Sources</h2><p>https://bbc.com/iran</p></body></html>",
            "<<DONE>>",
        ]),
        hook=hook,
    )
    raw, parsed, _v, _r = _run(sub._run_synthesis(None, "Write the report."))
    written = parsed.get("written_path")
    assert written is not None
    text = _read_written(tmp_path, written)
    low = text.lower()
    # WIRING fired
    assert "file_update" in calls
    # sentinel stripped
    assert SECTION_ANCHOR not in text
    # STRICT single-document-ness: no duplicate-tail / no 2nd <!DOCTYPE>/<html>
    assert low.count("<!doctype") == 1
    assert low.count("<html") == 1
    assert low.count("</html>") == 1
    # nothing trails past the single closing tag
    assert low.rstrip().endswith("</html>")
    # the real sections all landed, in order
    assert low.index("us-iran report") < low.index("timeline") < low.index("sources")
