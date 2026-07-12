"""bundles.codebase — the CodebaseReadBundle (d239/d241 generic-spine FLEX probe, s16/aflex).

A NON-WEB COMPLEX-MEMORY gather/read capability domain: READ a local codebase (a target
directory + its files) so a node can ground a summary in REAL on-disk source, exactly the
way the ``research`` + ``research_read`` bundles ground a report in fetched WEB sources —
but over a DIFFERENT, non-web source. It is a TOOL WRAPPER, NOT a role (d212): it carries
the codebase-read tools + the read doctrine; a WORKER node SELF-SELECTS it (``get_bundles``)
and the GENERIC self-select loop (:meth:`agent_runtime.runtime.SubAgent._run_linear_worker`
→ :meth:`._dispatch_loaded_tool`) drives its tools BY NAME — so no engine/loop change is
needed to add this source (the d239 spine claim; the dispatch docstring names "codebase,
vector-db" as the intended extension).

WHY THIS BUNDLE EXISTS (the probe, aflex): it is the live proof that a new capability that
exercises a DIFFERENT gather source than the web works by ADDING a bundle/shape/spec ONLY,
with ZERO orchestration edit. The codebase is a DIFFERENT complex-memory TYPE than
web-research, and the SAME domain-agnostic read interface (self-select → loaded-tool
dispatch) serves it.

It ORCHESTRATES the standard library (``pathlib``) only — it reimplements nothing and pulls
in no dependency (d190). Every read is BOUNDED (entry counts + per-file/total char caps) so a
node's lean context window (E4B determinism, d192) is never blown by a large file or a wide
tree.

NO ROLE-PHASE METHODS (d212 #2): the bundle exposes ONE :meth:`tool_specs` + :attr:`doctrine`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from reactive_tools.tool_registry import ToolDef

from ..research_tree import make_tool_spec
from .base import ObjectBundle

# Bounds (deterministic, window-safe). A directory listing is capped at _MAX_ENTRIES rows;
# a single file read at _PER_FILE_CHARS; a whole-directory digest at _DIGEST_FILES files of
# _DIGEST_HEAD_CHARS each. All are generous enough to gather a small module in ONE or TWO
# tool calls (the linear worker runs a tight turn budget) yet bounded so the context stays lean.
_MAX_ENTRIES = 200
_PER_FILE_CHARS = 6000
_DIGEST_FILES = 25
_DIGEST_HEAD_CHARS = 1400

# Extensions we treat as readable source/text. A path outside this set is reported as
# "binary/non-text (skipped)" rather than dumping bytes into the window.
_TEXT_SUFFIXES = frozenset({
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".rb", ".c",
    ".h", ".cc", ".cpp", ".hpp", ".cs", ".kt", ".swift", ".php", ".scala", ".sh",
    ".toml", ".cfg", ".ini", ".json", ".yaml", ".yml", ".md", ".rst", ".txt", ".sql",
    ".html", ".css", ".xml", ".gradle", ".tf", ".env", ".properties", "",
})


def _is_textlike(path: Path) -> bool:
    """True when ``path``'s suffix is a known text/source extension (cheap, by name)."""
    return path.suffix.lower() in _TEXT_SUFFIXES


def _resolve(root: Optional[str], raw: str) -> Path:
    """Resolve a model-supplied path. When the run bound a ``codebase_root`` (ctx), a
    RELATIVE path is taken under it; an absolute path is honored as-is. No ``root`` →
    resolve against the process cwd (the probe's served default)."""
    p = Path(str(raw or "").strip().strip('"').strip("'"))
    if root and not p.is_absolute():
        return (Path(root) / p)
    return p


def make_list_dir(root: Optional[str] = None):
    """Build a ``list_dir`` handler — list the entries of a directory (names + kind + size),
    bounded to :data:`_MAX_ENTRIES`. Real-paths-only: a missing/!dir path returns an explicit
    note, never invented content."""

    def list_dir(path: str = ".") -> dict[str, Any]:
        target = _resolve(root, path)
        if not target.exists():
            return {"path": str(target), "found": False,
                    "note": f"NO SUCH PATH {str(target)!r}. Give a directory path that exists."}
        if not target.is_dir():
            return {"path": str(target), "found": True, "is_dir": False,
                    "note": "This is a FILE, not a directory — call read_file on it instead."}
        entries: list[dict[str, Any]] = []
        try:
            children = sorted(target.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower()))
        except OSError as exc:
            return {"path": str(target), "found": True, "note": f"cannot list: {exc}"}
        for child in children[:_MAX_ENTRIES]:
            try:
                size = child.stat().st_size if child.is_file() else 0
            except OSError:
                size = 0
            entries.append({
                "name": child.name,
                "kind": "dir" if child.is_dir() else "file",
                "bytes": size,
                "textlike": child.is_file() and _is_textlike(child),
            })
        return {
            "path": str(target), "found": True, "is_dir": True,
            "entries": entries, "count": len(entries),
            "truncated": len(children) > _MAX_ENTRIES,
        }

    return list_dir


def make_read_file(root: Optional[str] = None):
    """Build a ``read_file`` handler — return ONE file's text, bounded to
    :data:`_PER_FILE_CHARS`. A binary/non-text or missing file returns an explicit note."""

    def read_file(path: str, max_chars: int = _PER_FILE_CHARS) -> dict[str, Any]:
        target = _resolve(root, path)
        if not target.exists() or not target.is_file():
            return {"path": str(target), "found": False,
                    "note": f"NO SUCH FILE {str(target)!r}. Use list_dir to see what exists."}
        if not _is_textlike(target):
            return {"path": str(target), "found": True, "text": "",
                    "note": "binary/non-text file (skipped) — summarize it by name/role only."}
        cap = max(1, min(int(max_chars or _PER_FILE_CHARS), _PER_FILE_CHARS))
        try:
            raw = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"path": str(target), "found": True, "text": "", "note": f"cannot read: {exc}"}
        text = raw[:cap]
        return {
            "path": str(target), "found": True, "text": text,
            "chars": len(text), "total_chars": len(raw),
            "truncated": len(raw) > cap,
        }

    return read_file


def make_read_dir(root: Optional[str] = None):
    """Build a ``read_dir`` handler — survey a directory AND digest each text file's head in
    ONE call (the workhorse for a tight turn budget). Returns up to :data:`_DIGEST_FILES`
    files, each truncated to :data:`_DIGEST_HEAD_CHARS`, so a small module is gathered in a
    single tool call. For an exact figure/signature, follow up with ``read_file`` on one path."""

    def read_dir(path: str = ".") -> dict[str, Any]:
        target = _resolve(root, path)
        if not target.exists():
            return {"path": str(target), "found": False,
                    "note": f"NO SUCH PATH {str(target)!r}. Give a directory path that exists."}
        if target.is_file():
            return make_read_file(root)(str(path))
        try:
            files = [c for c in sorted(target.iterdir(), key=lambda c: c.name.lower())
                     if c.is_file()]
        except OSError as exc:
            return {"path": str(target), "found": True, "note": f"cannot list: {exc}"}
        digests: list[dict[str, Any]] = []
        for f in files[:_DIGEST_FILES]:
            if not _is_textlike(f):
                digests.append({"path": str(f), "name": f.name, "head": "",
                                "note": "binary/non-text (skipped)"})
                continue
            try:
                raw = f.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                digests.append({"path": str(f), "name": f.name, "head": "", "note": f"unreadable: {exc}"})
                continue
            head = raw[:_DIGEST_HEAD_CHARS]
            digests.append({
                "path": str(f), "name": f.name, "head": head,
                "total_chars": len(raw), "truncated": len(raw) > _DIGEST_HEAD_CHARS,
            })
        return {
            "path": str(target), "found": True, "is_dir": True,
            "files": digests, "count": len(digests),
            "truncated": len(files) > _DIGEST_FILES,
            "note": ("Each file's HEAD is shown (truncated). For an exact signature/figure to "
                     "cite, call read_file on that file's path."),
        }

    return read_dir


# ---------------------------------------------------------------------------- #
# Native tool schemas (the inspectable surface the model is offered). Handlers are
# bound per-run in :meth:`CodebaseReadBundle.register`.
# ---------------------------------------------------------------------------- #
_LIST_DIR_SPEC: dict[str, Any] = make_tool_spec(
    "list_dir",
    "SURVEY a local directory — list its entries (name, file/dir, size) so you learn WHAT "
    "files the codebase holds before reading them. Pass a directory path. Then read_dir to "
    "digest them all, or read_file for one file's full text.",
    {"path": {"type": "string"}},
    ["path"],
)

_READ_DIR_SPEC: dict[str, Any] = make_tool_spec(
    "read_dir",
    "READ a whole directory in ONE call — returns each text file's name + the HEAD of its "
    "contents (truncated), so you can summarize a module without a read per file. Pass a "
    "directory path. Use this FIRST to gather a codebase area, then read_file only for the "
    "exact file whose full text or precise signature you must cite.",
    {"path": {"type": "string"}},
    ["path"],
)

_READ_FILE_SPEC: dict[str, Any] = make_tool_spec(
    "read_file",
    "READ ONE file's text in full (bounded) — use after read_dir when you need a specific "
    "file's complete contents or an exact signature/figure to cite. Pass the file path; cite "
    "what you learn to that path.",
    {"path": {"type": "string"}, "max_chars": {"type": "integer"}},
    ["path"],
)


class ListDirArgs(BaseModel):
    """Arguments for ``list_dir`` — survey a directory's entries."""

    path: str = Field(".", description="Directory path to list (relative to the codebase root, or absolute).")


class ReadDirArgs(BaseModel):
    """Arguments for ``read_dir`` — digest every text file's head in a directory."""

    path: str = Field(".", description="Directory path to digest.")


class ReadFileArgs(BaseModel):
    """Arguments for ``read_file`` — read ONE file's text."""

    path: str = Field(..., description="The file path to read (relative to the codebase root, or absolute).")
    max_chars: int = Field(
        _PER_FILE_CHARS, ge=1,
        description="Cap on the chars returned (the bundle clamps it to its per-file bound).",
    )


_READ_DOCTRINE = (
    "CODEBASE READING (a NON-WEB complex memory) — never describe a file from memory; READ the "
    "real on-disk source. Work on a cost hierarchy, cheap first: (1) list_dir(path) to SURVEY "
    "what a directory holds; (2) read_dir(path) to digest every text file's head in ONE call — "
    "this is your primary gather move, it covers a whole module cheaply; (3) read_file(path) only "
    "for the one file whose full text or exact signature/figure you must quote. Treat each file "
    "as a SOURCE: attribute every claim to the file PATH you read it from (e.g. 'bundles/file.py "
    "defines FileBundle …'), and never invent a file, path or symbol you did not actually read. "
    "Cover the directory's files for BREADTH before drilling into one for depth."
)


class CodebaseReadBundle(ObjectBundle):
    """Read a LOCAL CODEBASE (a non-web complex-memory source): list_dir / read_dir / read_file."""

    name = "codebase"
    summary = (
        "READ a LOCAL CODEBASE (files on disk) to ground a summary or answer in real source — "
        "list_dir to survey a directory, read_dir to digest a whole directory's files in one "
        "call, read_file for one file's full text. Load this when your task is to read, analyze "
        "or summarize files in a local directory/repository (NOT the web)."
    )

    @property
    def own_doctrine(self) -> str:  # type: ignore[override]
        return _READ_DOCTRINE

    def tool_specs(self, ctx: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
        return super().tool_specs(ctx) + [
            dict(_LIST_DIR_SPEC), dict(_READ_DIR_SPEC), dict(_READ_FILE_SPEC),
        ]

    def register(self, registry: Any, ctx: Optional[Mapping[str, Any]] = None) -> Any:
        """Add the codebase-read tools onto ``registry``. ``ctx`` may carry an optional
        ``codebase_root`` to scope relative paths; absent, paths resolve against the cwd.
        Handler-backed (orchestrates ``pathlib``), so the generic loaded-tool dispatch drives
        them by name with no loop change."""
        ctx = ctx or {}
        root = ctx.get("codebase_root")
        root = str(root) if root else None
        registry.add(ToolDef(
            name="list_dir", description=_LIST_DIR_SPEC["function"]["description"],
            args_model=ListDirArgs, handler=make_list_dir(root),
        ))
        registry.add(ToolDef(
            name="read_dir", description=_READ_DIR_SPEC["function"]["description"],
            args_model=ReadDirArgs, handler=make_read_dir(root),
        ))
        registry.add(ToolDef(
            name="read_file", description=_READ_FILE_SPEC["function"]["description"],
            args_model=ReadFileArgs, handler=make_read_file(root),
        ))
        return registry


__all__ = ["CodebaseReadBundle"]
