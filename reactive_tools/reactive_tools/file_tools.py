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

import json
import mimetypes
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


# --------------------------------------------------------------------------- #
# Extension → MIME PASSTHROUGH (c3/d49) — derive the type from the CHOSEN name
# --------------------------------------------------------------------------- #
# The file-output type is now LLM-driven: the model picks ANY filename/extension
# (.md/.txt/.csv/.html/...), and the mime is a PURE PASSTHROUGH derived from that
# name — NOT a per-format template or a content sniffer (the old _ensure_md/
# report.md hard-codes are gone). ``mimetypes`` is the stdlib extension table;
# the tiny supplement covers the couple of text types it misses on some hosts.
_MIME_SUPPLEMENT: dict[str, str] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".csv": "text/csv",
    ".txt": "text/plain",
}


def mime_for_path(path: str) -> str:
    """MIME type derived from the file's extension (passthrough, no sniffing).

    Uses the stdlib :mod:`mimetypes` table, supplemented for the few text types
    it misses (``.md``/``.csv``), and appends ``charset=utf-8`` for ``text/*`` so
    a downloaded text artifact is decoded correctly. Unknown extensions fall back
    to ``application/octet-stream`` — the honest "I don't know the type" answer,
    never a fabricated one."""
    name = (path or "").strip()
    ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] else ""
    mime = _MIME_SUPPLEMENT.get(ext) or mimetypes.guess_type(name)[0] or "application/octet-stream"
    if mime.startswith("text/") and "charset" not in mime:
        mime = f"{mime}; charset=utf-8"
    return mime


# --------------------------------------------------------------------------- #
# ENVELOPE GUARD at the WRITE boundary (R5 / c1r) — no raw JSON wrapper on disk
# --------------------------------------------------------------------------- #
# A bare node (no schema) can wrap its produce output in a {"output": ...} /
# {"findings": [...]} envelope; on the acyclic web_search->file_write path that
# wrapper leaked VERBATIM onto disk (c1r R5: a stray report.html holding raw
# {"findings": [...]}). This is the LAST line of defence, at the file_write TOOL
# itself — so NO route (synthesis, acyclic, or any future one) can land a JSON
# wrapper on disk. When the WHOLE content is such a wrapper we UNWRAP to the inner
# deliverable string if there is one, else REFUSE the write (a visible
# ToolInputError — the honest "this is not a deliverable" answer, never silent
# garbage). ``.json`` targets are EXEMPT (JSON is their legitimate content), and a
# real deliverable that merely CONTAINS JSON is untouched (the wrapper must be the
# entire body). This is the disk-boundary STRICT counterpart to the lenient
# surfaced-output ``agent_runtime.synth_tools.unwrap_output_envelope``; kept self-
# contained here so the low-level tool needs no upward import.
# The wrapper key whose value, when a non-empty string, IS the real deliverable.
_ENVELOPE_INNER_KEYS: tuple[str, ...] = (
    "output", "content", "report", "text", "result", "body", "deliverable",
)
# The FULL set of internal node-scaffold keys. A non-.json file whose entire body is
# a JSON object with ALL keys in this set is an internal role envelope (research
# {findings,sources,open_questions,...}, verify {verdict,findings,...}, the bare
# {output}/{findings} wrappers), NEVER a user deliverable — so it is unwrapped or
# refused. The "all keys in the set" test (not a key-count cap) is what keeps a
# genuine JSON data document — which would carry keys OUTSIDE this set — from ever
# being misread as an envelope.
_ENVELOPE_WRAPPER_KEYS = frozenset(
    _ENVELOPE_INNER_KEYS
    + (
        "findings", "data", "items", "sources", "verdict", "open_questions", "gaps",
        "weak_claims", "follow_up_queries", "fixed_inline", "summary", "title",
        "citations", "notes",
    )
)


def _bare_envelope_inner(content: str) -> tuple[bool, Optional[str]]:
    """Detect a bare node-scaffold JSON body → ``(is_envelope, inner_str_or_None)``.

    ``is_envelope`` is True ONLY when the entire stripped content is a JSON object
    whose keys are ALL known node-scaffold keys (:data:`_ENVELOPE_WRAPPER_KEYS`) — so
    a genuine JSON data document (with keys outside the set) is never misread. ``inner``
    is the first wrapper value that is a non-empty string (the real deliverable to
    unwrap to), or None when the scaffold holds only structured data (e.g. a
    ``findings`` list / a multi-field research or verify envelope) — the refuse case."""
    s = (content or "").strip()
    if not (s.startswith("{") and s.endswith("}")):
        return (False, None)
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return (False, None)
    if not isinstance(obj, dict) or not obj:
        return (False, None)
    if not set(obj).issubset(_ENVELOPE_WRAPPER_KEYS):
        return (False, None)
    for k in _ENVELOPE_INNER_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return (True, v)
    return (True, None)


def guard_write_content(content: str, path: str) -> str:
    """Sanitise ``content`` at the write boundary — UNWRAP or REFUSE a JSON envelope.

    Returns the content to actually write: unchanged when it is a real deliverable,
    or the unwrapped inner deliverable when the whole body was a ``{"output": ...}``
    wrapper. RAISES :class:`ToolInputError` when the body is a bare wrapper with no
    inner string (e.g. ``{"findings": [...]}``) — refusing keeps raw JSON off disk.
    ``.json`` targets are exempt (JSON is their legitimate content)."""
    if (path or "").strip().lower().endswith(".json"):
        return content
    is_env, inner = _bare_envelope_inner(content)
    if not is_env:
        return content
    if inner is not None:
        return inner
    raise ToolInputError(
        'refusing to write a bare JSON envelope ({"output"|"findings"|...}) as file '
        "content: no deliverable text inside (an upstream node leaked its raw wrapper)"
    )


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
    """Args for ``file_read`` (supports cheap state-checking: offset/range/tail)."""

    path: str = Field(
        ...,
        description=(
            "Path of the text file to read, RELATIVE to the workspace root "
            "(it must resolve inside the sandbox)."
        ),
    )
    offset: int = Field(
        0,
        ge=0,
        description="Byte offset to start reading from (0 = beginning). Ignored when 'tail' is set.",
    )
    length: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "Number of bytes to read from 'offset' (a range read). None = read to "
            "end (capped by max_bytes). Ignored when 'tail' is set."
        ),
    )
    tail: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Read only the LAST N bytes of the file (cheap end-of-file check for a "
            "react read-back loop). Overrides offset/length when set."
        ),
    )
    max_bytes: int = Field(
        DEFAULT_READ_MAX_BYTES,
        ge=1,
        description="Hard cap on bytes returned; 'truncated' is True if more remains beyond the window.",
    )
    encoding: str = Field(
        "utf-8", description="Text encoding to decode the bytes with (errors replaced)."
    )


class FileWriteArgs(BaseModel):
    """Args for ``file_write`` (hard-sandboxed, STEPWISE-decomposable).

    The write OPERATION is decomposed into composed steps (c3/d49) so an agent can
    drive it like the neuron drives a file: ``write(filename)`` first (just the
    ``path``, no ``content`` — creates/opens the file and fixes the chosen
    name/extension), then one or more ``write(content)`` calls (with ``append``
    to add the next section/page). Type-agnostic: the extension is whatever the
    model chose, and the mime is a passthrough from it."""

    path: str = Field(
        ...,
        description=(
            "Destination path RELATIVE to the workspace root, with the extension YOU "
            "choose (.md/.txt/.csv/.html/...). MUST resolve inside the sandbox; any "
            "'..'/absolute/symlink path that escapes is REFUSED."
        ),
    )
    content: str = Field(
        "",
        description=(
            "Text content to write. Omit (or empty) to just create/open the file at "
            "'path' as the first step; provide it to write the body."
        ),
    )
    append: bool = Field(
        False,
        description=(
            "If True, APPEND content to the end of the existing file (build a "
            "document section-by-section); if False, write/replace the file body."
        ),
    )
    overwrite: bool = Field(
        True,
        description="If False, refuse to overwrite an existing file (no clobber). Ignored when append=True.",
    )
    encoding: str = Field("utf-8", description="Text encoding to write the content with.")


class FileUpdateArgs(BaseModel):
    """Args for ``file_update`` — a TARGETED read-modify-write of an existing file.

    The reviewer's surgical edit primitive (s13/P1, FIX-C): correct ONE grounded
    span in place — replace an exact ``old`` snippet with ``new`` — instead of
    re-emitting the whole document (which a small model truncates). It reads the
    file, replaces the matched span, and writes the result back through the SAME
    hard sandbox + envelope guard as ``file_write``. RAW content only (d50): ``new``
    is verbatim text, never a JSON wrapper."""

    path: str = Field(
        ...,
        description=(
            "Path of the EXISTING text file to update, RELATIVE to the workspace root "
            "(it must resolve inside the sandbox)."
        ),
    )
    old: str = Field(
        ...,
        description=(
            "The exact text snippet to find and replace (verbatim, including its "
            "surrounding context if needed to make it unique)."
        ),
    )
    new: str = Field(
        "",
        description=(
            "The replacement text (raw content). Empty string DELETES the matched "
            "'old' span (ground-or-remove)."
        ),
    )
    count: int = Field(
        1,
        ge=0,
        description=(
            "How many occurrences of 'old' to replace (0 = ALL). Defaults to the "
            "first occurrence, the safe targeted edit."
        ),
    )
    encoding: str = Field("utf-8", description="Text encoding to read+write with.")


# --------------------------------------------------------------------------- #
# Handlers — built bound to a sandbox root
# --------------------------------------------------------------------------- #


def make_file_read(root: Path):
    """Build the ``file_read`` handler bound to sandbox ``root``."""

    def file_read(
        path: str,
        offset: int = 0,
        length: Optional[int] = None,
        tail: Optional[int] = None,
        max_bytes: int = DEFAULT_READ_MAX_BYTES,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        """Read a text file confined to the workspace sandbox (optionally a slice).

        ``_safe_resolve`` rejects any ``path`` escaping the sandbox (so a read can
        never be aimed at an arbitrary file on disk). Reads efficiently via ``seek``
        so a react loop can cheaply check the CURRENT file state (c3/d49.5):

        * ``tail=N`` → the LAST ``N`` bytes (overrides offset/length);
        * ``offset``/``length`` → a byte RANGE from ``offset`` (``length`` None reads
          to the end);
        * otherwise the whole file.

        Every window is still capped by ``max_bytes``. Returns
        ``{path, text, truncated, bytes, offset, size}`` where ``size`` is the FULL
        file size and ``truncated`` is True when bytes remain beyond the returned
        window (so a caller knows to read further / read the tail)."""
        target = _safe_resolve(path, root)
        if not target.is_file():
            raise ToolInputError(f"not a file in the workspace: {path!r}")
        size = target.stat().st_size
        # Resolve the requested window [start, start+want) within the file.
        if tail is not None and tail > 0:
            start = max(0, size - tail)
            want = size - start
        else:
            start = min(max(0, int(offset)), size)
            want = (size - start) if length is None else max(0, int(length))
            want = min(want, size - start)
        read_n = min(want, max_bytes)
        with target.open("rb") as fh:
            fh.seek(start)
            raw = fh.read(read_n)
        truncated = (start + len(raw)) < size
        return {
            "path": str(target),
            "text": raw.decode(encoding, errors="replace"),
            "truncated": truncated,
            "bytes": len(raw),
            "offset": start,
            "size": size,
        }

    return file_read


def make_file_write(root: Path):
    """Build the ``file_write`` handler bound to sandbox ``root`` (the safety bar)."""

    def file_write(
        path: str,
        content: str = "",
        append: bool = False,
        overwrite: bool = True,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        """Write ``content`` to ``path`` INSIDE the workspace sandbox (stepwise).

        STEPWISE (c3/d49): ``content`` is optional, so ``file_write(path=...)`` with
        no content is the first step — it creates/opens the file at the chosen name
        (the file is left empty if it did not exist) and fixes the extension; later
        ``file_write(path=..., content=..., append=True)`` calls fill it
        section-by-section. ``append`` adds to the end of the existing file;
        otherwise the body is written/replaced.

        SAFETY BAR (o1): ``_safe_resolve`` resolves the path (realpath, following
        symlinks) and REFUSES — raising :class:`ToolInputError` BEFORE any write —
        any target whose real location is not under the real workspace root
        (``..``-traversal, absolute paths that escape, and symlinks pointing out).
        Only after that check passes are parent dirs (themselves inside the
        sandbox) created and the bytes written.

        The type is LLM-driven: ``mime`` is a PASSTHROUGH derived from the chosen
        extension (no per-format template/sniffer). Returns
        ``{path, bytes, created, mime, size, appended}`` where ``bytes`` is the
        bytes written THIS call and ``size`` is the file's total size after — so a
        read-back / react loop sees the actual state. With ``overwrite=False`` an
        existing file is refused (no clobber); ``append`` ignores that (it adds)."""
        if not isinstance(content, str):
            raise ToolInputError("content must be a string")
        # R5 (c1r): no raw JSON envelope reaches disk on ANY route — unwrap a
        # {"output": "<deliverable>"} wrapper to its inner text, refuse a wrapper
        # with no deliverable string ({"findings": [...]}). .json targets exempt.
        content = guard_write_content(content, path)
        # The single sandbox gate: any escaping path raises here, pre-write.
        target = _safe_resolve(path, root)
        existed = target.exists()
        if existed and not append and not overwrite:
            raise ToolInputError(f"refusing to overwrite existing file: {path!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode(encoding)
        if append and existed:
            with target.open("ab") as fh:
                fh.write(data)
        else:
            # First step (create/open) or a full (over)write — both land here; an
            # empty ``content`` simply creates/truncates the file at the chosen name.
            target.write_bytes(data)
        return {
            "path": str(target),
            "bytes": len(data),
            "created": not existed,
            "mime": mime_for_path(str(target)),
            "size": target.stat().st_size,
            "appended": bool(append and existed),
        }

    return file_write


def make_file_update(root: Path):
    """Build the ``file_update`` handler bound to sandbox ``root`` (s13/P1, FIX-C).

    A TARGETED read-modify-write: replace an exact ``old`` snippet with ``new`` in an
    existing file and write the result back. This is the reviewer's surgical edit
    primitive — it grounds-or-removes ONE flagged span in place rather than forcing
    the small model to re-emit the whole document (which it truncates). It reuses the
    SAME sandbox gate + envelope guard as ``file_write`` (no second safety path)."""

    def file_update(
        path: str,
        old: str,
        new: str = "",
        count: int = 1,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        """Replace an exact ``old`` span with ``new`` inside an EXISTING sandboxed file.

        ``count`` bounds how many occurrences are replaced (``0`` = all, default the
        first). The matched span must be PRESENT — a missing ``old`` raises
        :class:`ToolInputError` (an honest "nothing matched", never a silent no-op that
        masquerades as an applied edit). The post-replacement body passes the same
        :func:`guard_write_content` envelope guard and the same ``_safe_resolve`` sandbox
        bar as :func:`make_file_write`, so no escaping path and no raw JSON wrapper can
        be written. Returns ``{path, replaced, bytes, size, removed}`` where ``replaced``
        is the number of occurrences changed and ``removed`` is True for a delete
        (empty ``new``)."""
        if not isinstance(old, str) or old == "":
            raise ToolInputError("file_update requires a non-empty 'old' snippet to find")
        if not isinstance(new, str):
            raise ToolInputError("file_update 'new' must be a string")
        target = _safe_resolve(path, root)
        if not target.is_file():
            raise ToolInputError(f"not a file in the workspace: {path!r}")
        text = target.read_bytes().decode(encoding, errors="replace")
        occurrences = text.count(old)
        if occurrences == 0:
            raise ToolInputError(
                f"file_update found no occurrence of the 'old' snippet in {path!r}"
            )
        n = occurrences if int(count) <= 0 else min(int(count), occurrences)
        updated = text.replace(old, new, n)
        # Same disk-boundary envelope guard as file_write — no raw JSON wrapper lands.
        updated = guard_write_content(updated, path)
        data = updated.encode(encoding)
        target.write_bytes(data)
        return {
            "path": str(target),
            "replaced": n,
            "bytes": len(data),
            "size": target.stat().st_size,
            "removed": new == "",
        }

    return file_update


# --------------------------------------------------------------------------- #
# The registry entries (one ToolDef each) + registration helpers
# --------------------------------------------------------------------------- #


def build_filesystem_tools(root: Optional[Path | str] = None) -> list[ToolDef]:
    """Build the :class:`ToolDef` entries (``file_read``, ``file_write``, ``file_update``).

    All bound to the SAME resolved workspace ``root`` (see
    :func:`resolve_workspace_root`). Returns ``[file_read_def, file_write_def,
    file_update_def]`` — each is exactly one registry entry, the whole point of d1's
    growability. ``file_update`` (s13/P1) is the reviewer's surgical in-place edit, a
    peer of the existing read/write so a generic reviewer node has proper READ/WRITE/
    UPDATE file tools."""
    base = resolve_workspace_root(root)
    read_def = ToolDef(
        name="file_read",
        description=(
            "Read a UTF-8 text file from the workspace (relative path); supports "
            "offset/length range reads and tail=N for a cheap end-of-file check. "
            "Sandbox-confined. Returns {path, text, truncated, bytes, offset, size}."
        ),
        args_model=FileReadArgs,
        handler=make_file_read(base),
    )
    write_def = ToolDef(
        name="file_write",
        description=(
            "Write a text file into the workspace at the filename+extension YOU choose "
            "(.md/.txt/.csv/.html/...). Stepwise: call with just a path to create the "
            "file, then with content (append=True to add sections). HARD-SANDBOXED: "
            "REFUSES any path escaping the workspace root (.., absolute, symlink). "
            "Returns {path, bytes, created, mime, size, appended}."
        ),
        args_model=FileWriteArgs,
        handler=make_file_write(base),
    )
    update_def = ToolDef(
        name="file_update",
        description=(
            "Update an EXISTING workspace text file in place: replace an exact 'old' "
            "snippet with 'new' (empty 'new' deletes it). The reviewer's surgical "
            "ground-or-remove edit — fix one flagged span without re-emitting the whole "
            "document. count=0 replaces all occurrences (default 1). Same hard sandbox "
            "as file_write; a missing 'old' is refused. Returns {path, replaced, bytes, "
            "size, removed}."
        ),
        args_model=FileUpdateArgs,
        handler=make_file_update(base),
    )
    return [read_def, write_def, update_def]


def register_filesystem_tools(
    registry: GrowableToolRegistry, root: Optional[Path | str] = None
) -> list[ToolDef]:
    """Add ``file_read`` + ``file_write`` + ``file_update`` to ``registry`` (a1 growth point).

    Each :meth:`GrowableToolRegistry.add` makes the tool immediately selectable
    (enum) AND dispatchable (hook). Returns the added defs. The sandbox root is
    resolved once and shared by all."""
    defs = build_filesystem_tools(root)
    for d in defs:
        registry.add(d)
    return defs


__all__ = [
    "WORKSPACE_ROOT_ENV",
    "DEFAULT_WORKSPACE_ROOT",
    "guard_write_content",
    "mime_for_path",
    "resolve_workspace_root",
    "FileReadArgs",
    "FileWriteArgs",
    "FileUpdateArgs",
    "make_file_read",
    "make_file_write",
    "make_file_update",
    "build_filesystem_tools",
    "register_filesystem_tools",
]
