"""bundles.file — the FileBundle (d190/d196/d197 + d212 capability-domain redraw).

The CAPABILITY DOMAIN for AUTHORING a file with GENERIC, format-agnostic file tools
(``file_write`` / ``file_read`` / ``file_update``). It is a TOOL WRAPPER, NOT a role
(d212): it never "writes the report" itself — it only carries the file-authoring tools
+ the doctrine that teaches the *write-a-section → read-it-back → continue* loop. The
WRITER / SYNTHESIZER / REVIEWER node types (d213) load THIS bundle (composed with the
``research_read`` bundle so they can also pull sources) — the bundle is not bound to any
one of them.

This is the file half of the OLD WriterBundle, which CONFLATED two capability domains —
file authoring AND source reading (``load_source``) — under one role-shaped bundle. The
redraw (d212 #3) splits them: ``file`` (here) is the write capability; ``research_read``
is the read-a-fetched-source capability; a writing node COMPOSES both.

NO BABYSITTING (d196/d197): this bundle deliberately carries NO deterministic HTML
assembly / regex normalization. Single-document coherence and any cleanup are the
autonomous author's + the reviewer's job, driven by doctrine + real file read-back — not
a regex pile post-processing the model's output. The file tools are GENERIC; FORMAT
(HTML / Markdown / code / CSV) comes from the node's SPEC, never from format-specific
core logic.

The data/presentation separation guidance the runtime's write loop injects
(:data:`REPORT_SEPARATION_GUIDANCE`) lives HERE as the single source of truth; the
runtime re-imports it under its private name so behaviour is byte-identical while the
bundle owns the doctrine (d190).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from ..research_tree import make_tool_spec
from .base import ObjectBundle

# --------------------------------------------------------------------------- #
# Data/presentation separation guidance (MOVED here from the old writer bundle;
# runtime re-imports it under ``_REPORT_SEPARATION_GUIDANCE``). Verbatim — byte-identical.
# --------------------------------------------------------------------------- #
REPORT_SEPARATION_GUIDANCE = (
    " SEPARATE DATA FROM PRESENTATION: the shared facts (key figures, the timeline, the "
    "SOURCES list) are the DOCUMENT'S, written ONCE — never repeated in a later section. "
    "You MAY first save those key facts to a small structured data file (a .json, via "
    "file_write) and then write the report prose from it, but the report itself is RAW "
    "content, never JSON. Give the document ONE title (a single top-level heading) on "
    "this first section; every later section is a SUB-section under it — never open a "
    "second top-level title or a second figures/sources block. Finish each sentence you "
    "start — never stop a section mid-sentence."
)

# The file-authoring doctrine (d196/d197): own what you write, format from the spec, check
# reality via read-back, the reviewer fixes — no regex babysitting. This is CAPABILITY
# doctrine (how to operate the file tools), not a role identity — the node's ROLE framing +
# SPEC supply who-is-writing-what; this teaches the section-at-a-time read-back loop.
_FILE_DOCTRINE = (
    "FILE AUTHORING — own what you write with the GENERIC file tools. Compose ONE bounded "
    "section per turn: write it with file_write (append the next section), then READ THE "
    "FILE BACK with file_read to see the ACTUAL current state (not your memory of it), then "
    "continue from exactly where the file ends, and call finish when the whole deliverable "
    "is on disk. The FORMAT — HTML, Markdown, code, CSV, plain text — comes from your SPEC "
    "and the task, never from a fixed template: emit the raw bytes of that format directly. "
    "Use file_update to surgically correct ONE span in place rather than rewriting the file. "
    "Do not post-process or 'assemble' your output with hidden machinery and do not babysit "
    "structure — write it correctly, read it back to confirm, and leave final polish to the "
    "reviewer. "
    "MULTI-PART DOCUMENT OWNERSHIP (SF-1: there is NO engine assembly — the model authors "
    "the WHOLE document): when a document is written across parts, the FIRST part OPENS it "
    "— the format's document shell (for HTML: <!DOCTYPE html>, <html>, <head> with title + "
    "styles, <body>) followed by that part's own section(s). Every MIDDLE part appends JUST "
    "its own complete section(s) — never a second shell, never a section another part owns, "
    "never a closing tag for the document. ONLY the FINAL part closes the document (for "
    "HTML: the single sources/references section, then </body></html>) — exactly once. "
    "WRITE ONLY DELIVERABLE CONTENT INTO THE FILE (d218): never put planning notes, "
    "status/meta commentary, TODO or 'to be added later' markers, or a <script> block "
    "carrying such a note into file_write — if you need to reason about what to write next, "
    "do it in your REPLY, not in the deliverable. "
    "GROUND FROM THE RESEARCH MEMORY, DON'T GUESS (P2 pull discipline): when your run "
    "carries gathered research (a source index / note gists were handed to you), ALSO load "
    "the research_read bundle (get_bundles name=\"research_read\") and PULL your material — "
    "read_notes first for each source's cheap gist, then load_source for the exact figures, "
    "dates and quotes you will cite. A substantive section is written FROM pulled evidence; "
    "writing a thin section from memory of the task text alone is the failure this rule "
    "exists to prevent. "
    "IMAGES ARE REAL OR ABSENT (no-fabrication): if the goal calls for images/maps/photos, "
    "load the research bundle (get_bundles name=\"research\") and use image_search — an <img> "
    "src must be an image_url COPIED VERBATIM from an image_search result. NEVER write a "
    "placeholder, invented or relative image path (placeholder_*.jpg is a defect); if no "
    "real image was found, OMIT the image and say so in the surrounding text. "
    "REVIEWING AN EXISTING FILE (P4 — the reviewer's file mechanics): read it in bounded "
    "regions (file_read offset/length, or tail=N for the end) rather than one whole-file "
    "blob; verify its claims against sources you can pull on demand; fix a defect with a "
    "targeted file_update on the exact span — ground-or-remove: a claim no source backs is "
    "corrected or removed, never invented around. Targeted edits, never a whole-document "
    "re-emission."
)


class FileBundle(ObjectBundle):
    """File-authoring capability: the generic, format-agnostic file tools + write-loop doctrine."""

    name = "file"
    summary = (
        "AUTHOR a deliverable file with the generic, format-agnostic file tools "
        "(file_write / file_read / file_update) in a write-section -> read-back -> "
        "continue loop. Load this when your task is to write/save a document (report, "
        "code, markdown, CSV) to disk."
    )

    @property
    def own_doctrine(self) -> str:  # type: ignore[override]
        return f"{_FILE_DOCTRINE}\n\n{REPORT_SEPARATION_GUIDANCE.strip()}"

    # ------------------------------------------------------------------ #
    # the GENERIC file tools (format-agnostic) — the single tool surface this
    # capability domain exposes. There are NO role-phase / actor methods here:
    # the bundle offers ONE tool_specs(ctx); the runtime/role decides usage.
    # ------------------------------------------------------------------ #
    def _file_specs(self) -> list[dict[str, Any]]:
        """Native schemas for the generic, format-agnostic file tools (file_write /
        file_read / file_update). Mirror the file tools the runtime's raw-file loop
        drives — declared here so the capability's surface is explicit and
        format-neutral (NO html-specific tool, d196)."""
        return [
            make_tool_spec(
                "file_write",
                "Write (or append) RAW content to the deliverable file by name. The "
                "content is the raw bytes of whatever FORMAT the task/spec calls for "
                "(.html/.md/.txt/.csv/code) — never a JSON wrapper. Append one bounded "
                "section per call. Document shell ownership: the FIRST part of a "
                "multi-part document opens the shell (e.g. <!DOCTYPE html>/<head>+styles), "
                "middle parts append ONLY their own section markup, and ONLY the final "
                "part closes the document — never re-open or re-close mid-document, and "
                "do NOT write planning/meta notes or <script> comments into the content. "
                "Set append=true on EVERY call after the file exists (only the very "
                "first write of a new file omits it) — a non-append write to an "
                "existing file is refused, never a silent overwrite.",
                {"path": {"type": "string"}, "content": {"type": "string"},
                 "append": {"type": "boolean"}},
                ["path", "content"],
            ),
            make_tool_spec(
                "file_read",
                "Read the deliverable (or a bounded slice / tail) back from disk to see "
                "its ACTUAL current state before continuing — check reality, do not rely "
                "on memory of what you wrote.",
                {"path": {"type": "string"}, "offset": {"type": "integer"},
                 "length": {"type": "integer"}, "tail": {"type": "integer"}},
                ["path"],
            ),
            make_tool_spec(
                "file_update",
                "Surgically correct ONE span in place: replace the exact 'old' snippet "
                "with 'new' (empty 'new' removes it). The ground-or-remove edit — no "
                "whole-file rewrite needed for a small fix.",
                {"path": {"type": "string"}, "old": {"type": "string"},
                 "new": {"type": "string"}},
                ["path", "old", "new"],
            ),
        ]

    def tool_specs(self, ctx: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
        return super().tool_specs(ctx) + self._file_specs()


__all__ = ["FileBundle", "REPORT_SEPARATION_GUIDANCE"]
