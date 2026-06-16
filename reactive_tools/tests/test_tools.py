"""Unit coverage for the Claude-style file tools (s2/a1) + the global registry.

The registered ``read_file``/``write_file`` are the Claude-style tools
(:func:`reactive_tools.tools.make_read` / :func:`make_write`): line-based read
``offset``/``limit``, and ONE write entrypoint that either CREATES a file
(``new_file=True``, refusing to overwrite) or EDITS in place via exact-string
``old_string`` -> ``new_string`` replacement (failing on absent / non-unique
matches unless ``replace_all``). The artifact-dir default and the single global
tool registry (d12) are covered too.

No async test plugin is assumed (matching the repo): the one hook/registry test
that needs the event loop is driven through ``asyncio.run`` from a plain sync
test. The tool callables themselves are sync, so most tests call them directly.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from reactive_tools import EventPlane, build_default_hook
from reactive_tools.tools import (
    DEFAULT_ARTIFACT_DIR,
    ToolInputError,
    make_read,
    make_write,
    register_core_tools,
)
from reactive_tools.tool_hook import ToolHook


# --------------------------------------------------------------------------- #
# write_file — CREATE (new_file) mode
# --------------------------------------------------------------------------- #

def test_new_file_create_makes_parent_dirs_and_writes(tmp_path: Path):
    write = make_write(tmp_path)
    res = write("nested/sub/report.md", content="# Hello\nbody", new_file=True)
    target = tmp_path / "nested" / "sub" / "report.md"
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == "# Hello\nbody"
    assert res["mode"] == "create"
    assert res["created"] is True
    assert Path(res["path"]) == target


def test_new_file_mode_refuses_to_overwrite_existing(tmp_path: Path):
    write = make_write(tmp_path)
    write("a.txt", content="first", new_file=True)
    with pytest.raises(ToolInputError, match="already exists"):
        write("a.txt", content="second", new_file=True)
    # original is untouched
    assert (tmp_path / "a.txt").read_text() == "first"


# --------------------------------------------------------------------------- #
# write_file — EDIT (in-place exact-string replacement) mode
# --------------------------------------------------------------------------- #

def test_exact_edit_replaces_unique_old_string(tmp_path: Path):
    write = make_write(tmp_path)
    write("doc.txt", content="alpha BETA gamma", new_file=True)
    res = write("doc.txt", old_string="BETA", new_string="DELTA")
    assert (tmp_path / "doc.txt").read_text() == "alpha DELTA gamma"
    assert res["mode"] == "edit"
    assert res["replacements"] == 1


def test_edit_fails_on_absent_old_string(tmp_path: Path):
    write = make_write(tmp_path)
    write("doc.txt", content="alpha beta", new_file=True)
    with pytest.raises(ToolInputError, match="not found"):
        write("doc.txt", old_string="ZZZ", new_string="x")
    # file unchanged
    assert (tmp_path / "doc.txt").read_text() == "alpha beta"


def test_edit_fails_on_non_unique_old_string(tmp_path: Path):
    write = make_write(tmp_path)
    write("doc.txt", content="x x x", new_file=True)
    with pytest.raises(ToolInputError, match="not unique"):
        write("doc.txt", old_string="x", new_string="y")
    # nothing changed because the edit was rejected before writing
    assert (tmp_path / "doc.txt").read_text() == "x x x"


def test_edit_replace_all_replaces_every_occurrence(tmp_path: Path):
    write = make_write(tmp_path)
    write("doc.txt", content="x x x", new_file=True)
    res = write("doc.txt", old_string="x", new_string="y", replace_all=True)
    assert (tmp_path / "doc.txt").read_text() == "y y y"
    assert res["replacements"] == 3


def test_same_tool_creates_then_edits_one_entrypoint(tmp_path: Path):
    """The Claude-exact requirement: ONE tool both creates+names and edits."""
    write = make_write(tmp_path)
    write("flow.txt", content="line one\nline two\n", new_file=True)
    write("flow.txt", old_string="line two", new_string="line TWO edited")
    assert (tmp_path / "flow.txt").read_text() == "line one\nline TWO edited\n"


# --------------------------------------------------------------------------- #
# read_file — Claude-style offset/limit
# --------------------------------------------------------------------------- #

def test_read_offset_limit_selects_line_window(tmp_path: Path):
    write = make_write(tmp_path)
    read = make_read(tmp_path)
    content = "\n".join(f"line{i}" for i in range(10))  # line0..line9
    write("lines.txt", content=content, new_file=True)

    res = read("lines.txt", offset=2, limit=3)
    assert res["content"] == "line2\nline3\nline4"
    assert res["lines_returned"] == 3
    assert res["total_lines"] == 10
    assert res["offset"] == 2
    assert res["line_sliced"] is True


def test_read_without_offset_limit_returns_full_text(tmp_path: Path):
    write = make_write(tmp_path)
    read = make_read(tmp_path)
    write("whole.txt", content="a\nb\nc", new_file=True)
    res = read("whole.txt")
    assert res["content"] == "a\nb\nc"
    assert res["text"] == "a\nb\nc"          # back-compat alias
    assert res["line_sliced"] is False


# --------------------------------------------------------------------------- #
# Path-traversal sandbox guard stays intact for the new tools
# --------------------------------------------------------------------------- #

def test_sandbox_guard_rejects_escape(tmp_path: Path):
    write = make_write(tmp_path)
    read = make_read(tmp_path)
    with pytest.raises(ToolInputError, match="escapes the allowed base"):
        write("../escape.txt", content="nope", new_file=True)
    with pytest.raises(ToolInputError, match="escapes the allowed base"):
        read("../escape.txt")


# --------------------------------------------------------------------------- #
# Artifact-dir default (d3): a bare filename lands under artifacts\
# --------------------------------------------------------------------------- #

def test_artifact_dir_is_the_default_base(tmp_path: Path):
    """With no explicit file_base, register_core_tools resolves bare names under
    DEFAULT_ARTIFACT_DIR (C:\\Projects\\ReactiveAgents\\artifacts) — the d3 fix."""
    hook = ToolHook(EventPlane())
    register_core_tools(hook)  # no file_base -> artifacts default
    write = hook.registry.get("write_file").func
    bare = "._s2_a1_artifact_default_probe.txt"
    res = write(bare, content="probe", new_file=True)
    produced = Path(res["path"])
    try:
        assert produced.parent == DEFAULT_ARTIFACT_DIR
        assert produced.is_file()
    finally:
        produced.unlink(missing_ok=True)


def test_explicit_file_base_overrides_default(tmp_path: Path):
    hook = ToolHook(EventPlane())
    register_core_tools(hook, file_base=tmp_path)
    write = hook.registry.get("write_file").func
    res = write("override.txt", content="x", new_file=True)
    assert Path(res["path"]).parent == tmp_path.resolve()


# --------------------------------------------------------------------------- #
# Global registry (d12): one shared registry holds every tool, reachable by name
# --------------------------------------------------------------------------- #

def test_default_hook_registry_catalog_has_all_tools():
    hook = build_default_hook(EventPlane(), file_base=None)
    names = {t["name"] for t in hook.registry.catalog()}
    # the 4 core tools the agent needs, all in the ONE registry
    assert {"read_file", "write_file", "web_search", "web_fetch"} <= names
    # every catalog entry carries a one-line description (what the planner sees)
    assert all(t["description"] for t in hook.registry.catalog())


def test_registry_is_reachable_through_the_one_hook(tmp_path: Path):
    """Tool AVAILABILITY is global: the SAME hook.invoke entrypoint reaches every
    registered tool by name (d12) — proven by round-tripping write then read."""
    hook = build_default_hook(EventPlane(), file_base=tmp_path)

    async def _drive():
        w = await hook.invoke("write_file", path="g.txt",
                              content="global registry", new_file=True)
        assert w.ok, w.error
        r = await hook.invoke("read_file", path="g.txt")
        assert r.ok, r.error
        return r.value["content"]

    assert asyncio.run(_drive()) == "global registry"
