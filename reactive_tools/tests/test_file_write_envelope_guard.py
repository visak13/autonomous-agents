"""R5 (c1r) — the file_write TOOL boundary refuses/unwraps a raw JSON envelope.

The o4 "raw JSON envelope" defect leaked a wrapper ({"output": ...} / {"findings":
[...]}) verbatim onto disk on the acyclic web_search->file_write path. The fix is a
LAST line of defence at the file_write handler itself, so NO route can land a JSON
wrapper: a {"output": "<str>"} wrapper is UNWRAPPED to its deliverable; a wrapper
with no inner string ({"findings": [...]}) is REFUSED (visible ToolInputError);
.json targets and real deliverables that merely contain JSON are untouched.
"""
from __future__ import annotations

import pytest

from reactive_tools.file_tools import guard_write_content, make_file_write
from reactive_tools.tools import ToolInputError


def test_guard_unwraps_output_wrapper():
    csv = "Name,Diameter_km,Moons\nEarth,12756,1"
    assert guard_write_content('{"output": "' + csv.replace("\n", "\\n") + '"}', "data.csv") == csv
    assert guard_write_content('{\n"output": "hello"\n}', "a.html") == "hello"


def test_guard_refuses_findings_wrapper_for_non_json():
    with pytest.raises(ToolInputError):
        guard_write_content('{"findings": ["a", "b"]}', "report.html")


def test_guard_refuses_multi_key_research_scaffold():
    # the live leak shape: a full research/verify node scaffold dumped to a .md —
    # many keys, ALL of them internal-scaffold keys, no deliverable string inside.
    scaffold = (
        '{"findings": ["f"], "sources": ["s"], "open_questions": ["q"], "gaps": [], '
        '"weak_claims": [], "follow_up_queries": ["nq"], "verdict": "converged", '
        '"fixed_inline": []}'
    )
    with pytest.raises(ToolInputError):
        guard_write_content(scaffold, "research-the-topic-in-depth.md")
    # a verify scaffold carrying an inner `output` string IS unwrapped to it
    assert guard_write_content('{"verdict": "ok", "output": "the report body"}', "r.md") == "the report body"


def test_guard_exempts_json_targets_and_leaves_real_content():
    # a .json file legitimately holds JSON — never unwrapped/refused
    body = '{"findings": ["a", "b"]}'
    assert guard_write_content(body, "data.json") == body
    # a real deliverable that merely CONTAINS json is untouched
    doc = '# Report\n\nSome {"k": 1} snippet inline.'
    assert guard_write_content(doc, "r.md") == doc
    # a foreign-key object (not a wrapper) is left alone
    assert guard_write_content('{"foo": 1, "bar": 2}', "x.txt") == '{"foo": 1, "bar": 2}'


def test_file_write_handler_unwraps_and_refuses_on_disk(tmp_path):
    fw = make_file_write(tmp_path)
    # {"output": ...} is unwrapped → the deliverable, not the wrapper, lands on disk
    fw(path="out.html", content='{"output": "<html><body>hi</body></html>"}')
    assert (tmp_path / "out.html").read_text(encoding="utf-8") == "<html><body>hi</body></html>"

    # {"findings": [...]} is refused → nothing is written
    with pytest.raises(ToolInputError):
        fw(path="leak.html", content='{"findings": ["x", "y"]}')
    assert not (tmp_path / "leak.html").exists()

    # a .json target keeps its JSON content verbatim
    fw(path="data.json", content='{"findings": ["x", "y"]}')
    assert (tmp_path / "data.json").read_text(encoding="utf-8") == '{"findings": ["x", "y"]}'
