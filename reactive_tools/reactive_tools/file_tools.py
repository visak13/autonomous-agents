"""file_read + file_write — the FILESYSTEM tools, as two growable-registry entries.

These are the s2/a3 deliverable: the two filesystem capabilities the agent nodes
use, each expressed as **ONE** :class:`~reactive_tools.tool_registry.ToolDef` on the
a1 :class:`~reactive_tools.tool_registry.GrowableToolRegistry` (adding a tool = one
entry, decision d1 — no registry-core change). They sit BEHIND the a1 interface:
once added they are immediately selectable (in the structured-selection enum) and
dispatchable (through the bound :class:`~reactive_tools.tool_hook.ToolHook`, with
``tool_call``/``tool_result`` events on the event plane).

1. ``file_read``  — read a text file (stdlib), size-bounded, confined to the
   workspace sandbox.
2. ``file_write`` — write a text file, **HARD-SANDBOXED to a single workspace
   root**: it REFUSES any path that resolves outside the sandbox root
   (``..``-traversal, absolute escapes, and symlink escapes are rejected BEFORE
   any byte is written). This refusal is a NAMED SAFETY BAR for outcome o1 and is
   enforced **in code, not by prompt**.

The sandbox itself is the proven guard already shipped in :mod:`reactive_tools.tools`
— :func:`~reactive_tools.tools._safe_resolve` (``os.path.realpath`` +
``os.path.commonpath``), which resolves symlinks and rejects any target whose real
path is not under the real workspace root. Reusing it keeps a SINGLE source of truth
for the path-traversal guard (rather than a second, divergent implementation) and
means file_write's safety bar is exactly the audited one. A violation RAISES
:class:`~reactive_tools.tools.ToolInputError`, so the dispatch layer surfaces a hard,
provable ``ok=False`` refusal (and a ``tool_result`` error event) — never a silent
write-elsewhere.

Workspace root (the single configurable constant)
-------------------------------------------------
Per the Round-3 blueprint, the workspace root is ONE configurable constant — the
SAME directory where the s7 scenarios write their markdown files and where the
existing ``read_file``/``write_file`` already sandbox. We therefore default to
:data:`reactive_tools.tools.DEFAULT_ARTIFACT_DIR`
(``C:\\Projects\\ReactiveAgents\\artifacts``) and allow an environment override via
``REACTIVE_AGENTS_WORKSPACE_ROOT`` (and an explicit ``root=`` arg on the builders).
There is no second root to keep in sync.

Decisions honored
-----------------
- d1  — each tool is exactly one ``ToolDef`` on the growable registry; no
  framework, no control flow here.
- d2  — purely in-process, stdlib file I/O (``pathlib``); no broker/pool/subprocess.
- d6  — no scratch evidence files: the proof is reported through ``record_*``.
- o1  — the file_write sandbox refusal is the named safety bar, enforced in code.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .tool_registry import GrowableToolRegistry, ToolDef
from .tools import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_READ_MAX_BYTES,
    ToolInputError,
    _safe_resolve,
)

# The single configurable workspace-root constant. Defaults to the SAME directory
# the existing file tools sandbox to / the s7 scenarios write markdown into; an
# env override lets a deploy repoint it without a code change.
WORKSPACE_ROOT_ENV = "REACTIVE_AGENTS_WORKSPACE_ROOT"
DEFAULT_WORKSPACE_ROOT = DEFAULT_ARTIFACT_DIR


def resolve_workspace_root(root: Optional[Path | str] = None) -> Path:
    """Resolve the workspace sandbox root (the single configurable constant).

    Precedence: an explicit ``root`` arg, else the ``REACTIVE_AGENTS_WORKSPACE_ROOT``
    environment variable, else :data:`DEFAULT_WORKSPACE_ROOT`. The directory is
    created if missing and returned as a realpath (symlinks resolved) so the
    containment check in :func:`_safe_resolve` compares real path against real path.
    """
    chosen = (
        root
        if root is not None
        else os.environ.get(WORKSPACE_ROOT_ENV) or DEFAULT_WORKSPACE_ROOT
    )
    base = Path(os.path.realpath(Path(chosen)))
    base.mkdir(parents=True, exist_ok=True)
    return base


# --------------------------------------------------------------------------- #
# Pydantic arg models — the single source of truth for each tool's schema
# --------------------------------------------------------------------------- #


class FileReadArgs(BaseModel):
    """Args for ``file_read``."""

    path: str = Field(
        ...,
        description=(
            "Path of the text file to read, RELATIVE to the workspace root "
            "(it must resolve inside the sandbox)."
        ),
    )
    max_bytes: int = Field(
        DEFAULT_READ_MAX_BYTES,
        ge=1,
        description="Maximum bytes to read; the result is marked truncated beyond this.",
    )
    encoding: str = Field(
        "utf-8", description="Text encoding to decode the bytes with (errors replaced)."
    )


class FileWriteArgs(BaseModel):
    """Args for ``file_write`` (hard-sandboxed)."""

    path: str = Field(
        ...,
        description=(
            "Destination path RELATIVE to the workspace root. MUST resolve inside "
            "the sandbox; any '..'/absolute/symlink path that escapes is REFUSED."
        ),
    )
    content: str = Field(..., description="The text content to write to the file.")
    overwrite: bool = Field(
        True,
        description="If False, refuse to overwrite an existing file (no clobber).",
    )
    encoding: str = Field("utf-8", description="Text encoding to write the content with.")


# --------------------------------------------------------------------------- #
# Handlers — built bound to a sandbox root
# --------------------------------------------------------------------------- #


def make_file_read(root: Path):
    """Build the ``file_read`` handler bound to sandbox ``root``."""

    def file_read(
        path: str,
        max_bytes: int = DEFAULT_READ_MAX_BYTES,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        """Read a text file confined to the workspace sandbox and return its text.

        ``_safe_resolve`` rejects any ``path`` escaping the sandbox (so a read can
        never be aimed at an arbitrary file on disk). Returns
        ``{path, text, truncated, bytes}``; ``truncated`` is True when the file was
        larger than ``max_bytes``."""
        target = _safe_resolve(path, root)
        if not target.is_file():
            raise ToolInputError(f"not a file in the workspace: {path!r}")
        raw = target.read_bytes()
        truncated = len(raw) > max_bytes
        raw = raw[:max_bytes]
        return {
            "path": str(target),
            "text": raw.decode(encoding, errors="replace"),
            "truncated": truncated,
            "bytes": len(raw),
        }

    return file_read


def make_file_write(root: Path):
    """Build the ``file_write`` handler bound to sandbox ``root`` (the safety bar)."""

    def file_write(
        path: str,
        content: str,
        overwrite: bool = True,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        """Write ``content`` to ``path`` INSIDE the workspace sandbox.

        SAFETY BAR (o1): ``_safe_resolve`` resolves the path (realpath, following
        symlinks) and REFUSES — raising :class:`ToolInputError` BEFORE any write —
        any target whose real location is not under the real workspace root
        (``..``-traversal, absolute paths that escape, and symlinks pointing out).
        Only after that check passes are parent dirs (themselves inside the
        sandbox) created and the bytes written. Returns ``{path, bytes, created}``.
        With ``overwrite=False`` an existing file is refused (no clobber)."""
        if not isinstance(content, str):
            raise ToolInputError("content must be a string")
        # The single sandbox gate: any escaping path raises here, pre-write.
        target = _safe_resolve(path, root)
        existed = target.exists()
        if existed and not overwrite:
            raise ToolInputError(f"refusing to overwrite existing file: {path!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode(encoding)
        target.write_bytes(data)
        return {"path": str(target), "bytes": len(data), "created": not existed}

    return file_write


# --------------------------------------------------------------------------- #
# The two registry entries (one ToolDef each) + registration helpers
# --------------------------------------------------------------------------- #


def build_filesystem_tools(root: Optional[Path | str] = None) -> list[ToolDef]:
    """Build the two :class:`ToolDef` entries (``file_read``, ``file_write``).

    Both are bound to the SAME resolved workspace ``root`` (see
    :func:`resolve_workspace_root`). Returns ``[file_read_def, file_write_def]`` —
    each is exactly one registry entry, the whole point of d1's growability."""
    base = resolve_workspace_root(root)
    read_def = ToolDef(
        name="file_read",
        description=(
            "Read a UTF-8 text file from the workspace (relative path). Size-bounded; "
            "confined to the workspace sandbox. Returns {path, text, truncated, bytes}."
        ),
        args_model=FileReadArgs,
        handler=make_file_read(base),
    )
    write_def = ToolDef(
        name="file_write",
        description=(
            "Write a UTF-8 text file into the workspace (relative path). HARD-SANDBOXED: "
            "REFUSES any path escaping the workspace root (.., absolute, symlink). "
            "Returns {path, bytes, created}; overwrite=False refuses to clobber."
        ),
        args_model=FileWriteArgs,
        handler=make_file_write(base),
    )
    return [read_def, write_def]


def register_filesystem_tools(
    registry: GrowableToolRegistry, root: Optional[Path | str] = None
) -> list[ToolDef]:
    """Add ``file_read`` + ``file_write`` to ``registry`` (the a1 growth point).

    Each :meth:`GrowableToolRegistry.add` makes the tool immediately selectable
    (enum) AND dispatchable (hook). Returns the two added defs. The sandbox root is
    resolved once and shared by both."""
    defs = build_filesystem_tools(root)
    for d in defs:
        registry.add(d)
    return defs


__all__ = [
    "WORKSPACE_ROOT_ENV",
    "DEFAULT_WORKSPACE_ROOT",
    "resolve_workspace_root",
    "FileReadArgs",
    "FileWriteArgs",
    "make_file_read",
    "make_file_write",
    "build_filesystem_tools",
    "register_filesystem_tools",
]
