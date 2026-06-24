"""Dedicated agentic AUTHORING tools for shapes + specs (s9/b4, d47/d49).

WHAT THIS IS
------------
The plan-building planner authors a DAG by issuing DISCRETE TOOL CALLS —
``seed_plan`` → ``add_step`` → ``finalize_plan`` (:mod:`agent_runtime.plan_tools`):
stepwise, prompt-elicited, NEVER ``format``-schema constrained (the d34 fix). d47/d49
ask the SAME shape for AUTHORING: the user describes intent in plain language and the
local model authors a SHAPE (a plan topology) and a SPEC (a behaviour/tone ruleset) as
an agentic **seed → iterate → freeze + compile** workflow — *exactly parallel to*
``seed_plan → add_step``.

The authoring MACHINERY already exists and is already off ``format=schema`` (c0 probe):
:class:`~chat_app.shape_authoring.ShapeAuthorService` (over
:class:`agent_runtime.shape_author.ShapeAuthor`) and
:class:`~chat_app.spec_chat.SpecChatService` (over
:class:`specialization.conversation.SpecConversation`). This module does **NOT** rebuild
any of it — it EXPOSES that machinery as a dedicated, NAMED tool surface (the data
specs below + two thin session dispatchers), so authoring is genuinely tool-shaped and
stepwise, the same way :mod:`agent_runtime.plan_tools` exposes plan-building. The two
existing chat windows (``Shapes/`` and ``SpecChat/``) remain the separate UIs (d48);
this surface is the programmatic, agentic tool layer underneath the SAME services those
windows already drive — one source of truth (the services), one named tool vocabulary.

THE WORKFLOW (both surfaces)
----------------------------
* **seed** — open the artifact from a free-text description (the planner's
  ``seed_plan``: it records intent, no content yet beyond the first draft).
* **iterate** — refine the draft in plain language, BUILDING ON the prior draft
  (the planner's repeated ``add_step``). Repeatable across turns.
* **freeze (+ compile)** — finalize the artifact (the planner's ``finalize_plan``):
  a SHAPE is already persisted as a runnable TOML on each author/iterate, so freeze
  validates it round-trips the loader and is SELECTABLE; a SPEC is compiled +
  registered (``approve``) so it is loadable by a node on its next run.

NOT schema-constrained — the lever is PROMPT QUALITY. The authoring prompts (in
:mod:`agent_runtime.shape_author` and :mod:`specialization.conversation`) carry the
SHORT+PRECISE / planner-aware / tone-exemplar cues (d47); this surface only sequences
the calls. Each ``dispatch`` returns a small OBSERVATION dict (``{ok, note, ...}``),
exactly like :meth:`agent_runtime.plan_tools.PlanBuilder.dispatch`, so the same
loop/UI pattern can drive it and echo progress back.
"""
from __future__ import annotations

import asyncio
from typing import Any, Mapping, Optional

from chat_app.shape_authoring import (
    ShapeAuthoringUnavailable,
    ShapeAuthorService,
    ShapeNameConflict,
    ShapeNotFound,
)
from chat_app.spec_chat import SpecChatService

# --------------------------------------------------------------------------- #
# The dedicated authoring tool surfaces (data, mirroring PLAN_TOOLS_SPEC).
# Each tool: {name, args (key -> one-line meaning), description}. Advertised to
# the model/UI so the prompt and the dispatcher agree on exactly one surface.
# --------------------------------------------------------------------------- #
SHAPE_AUTHOR_TOOLS_SPEC: tuple[dict[str, Any], ...] = (
    {
        "name": "seed_shape",
        "args": {
            "description": "free-text description of the plan SHAPE to author "
                           "(its execution posture: sequential / parallel-modular / "
                           "iterative deep-research, or a composition) (REQUIRED)",
            "name_hint": "optional kebab-case nudge for the shape id",
        },
        "description": "Author a NEW plan shape from a natural-language description "
                       "and persist it (selectable immediately). Call this FIRST.",
    },
    {
        "name": "refine_shape",
        "args": {
            "name": "the shape id to edit in place (REQUIRED)",
            "instruction": "plain-language change to apply, building on the current "
                           "shape (REQUIRED)",
        },
        "description": "Iterate on an existing shape: apply a plain-language change "
                       "BUILDING ON the current version and overwrite it. Repeatable.",
    },
    {
        "name": "freeze_shape",
        "args": {"name": "the shape id to freeze (REQUIRED)"},
        "description": "Finalize the shape: confirm it round-trips the loader and is "
                       "SELECTABLE by the planner. Call once the shape is right.",
    },
)

SPEC_AUTHOR_TOOLS_SPEC: tuple[dict[str, Any], ...] = (
    {
        "name": "seed_spec",
        "args": {
            "name": "kebab-case spec/ruleset id (REQUIRED)",
            "description": "one-line, selection-effective description of what this "
                           "ruleset shapes (REQUIRED)",
            "intent": "free-text of what the user wants the ruleset to do "
                      "(e.g. 'answer in a pirate voice') (REQUIRED)",
        },
        "description": "Open a spec-authoring session and author DRAFT 1 of the "
                       "behaviour/tone ruleset from intent. Call this FIRST.",
    },
    {
        "name": "refine_spec",
        "args": {
            "critique": "plain-language change to apply to the current draft, "
                        "building on it (REQUIRED)",
        },
        "description": "Iterate on the current spec draft: re-author BOTH the body "
                       "and description incorporating the critique. Repeatable.",
    },
    {
        "name": "freeze_spec",
        "args": {},
        "description": "Freeze + COMPILE the spec: compile the current draft and "
                       "register it so a node can load it on its next run.",
    },
)

SHAPE_AUTHOR_TOOL_NAMES: frozenset[str] = frozenset(
    t["name"] for t in SHAPE_AUTHOR_TOOLS_SPEC
)
SPEC_AUTHOR_TOOL_NAMES: frozenset[str] = frozenset(
    t["name"] for t in SPEC_AUTHOR_TOOLS_SPEC
)


def _catalog_text(spec: tuple[dict[str, Any], ...], header: str) -> str:
    """Render a tool surface + its args for a system prompt (mirror plan_tools)."""
    lines = [header]
    for tool in spec:
        lines.append(f"- {tool['name']}: {tool['description']}")
        for arg, meaning in tool["args"].items():
            lines.append(f"    {arg}: {meaning}")
    return "\n".join(lines)


def shape_author_catalog_text() -> str:
    return _catalog_text(
        SHAPE_AUTHOR_TOOLS_SPEC,
        "SHAPE-AUTHORING TOOLS (seed -> iterate -> freeze; call ONE per reply):",
    )


def spec_author_catalog_text() -> str:
    return _catalog_text(
        SPEC_AUTHOR_TOOLS_SPEC,
        "SPEC-AUTHORING TOOLS (seed -> iterate -> freeze+compile; call ONE per reply):",
    )


class AuthoringToolError(ValueError):
    """A tool call was structurally unusable (unknown tool, missing required arg)."""


# --------------------------------------------------------------------------- #
# SHAPE authoring session — wraps ShapeAuthorService (REUSE, not rebuild)
# --------------------------------------------------------------------------- #
class ShapeAuthoringSession:
    """Drive the shape seed→iterate→freeze workflow over :class:`ShapeAuthorService`.

    Stateful only in the small sense that ``freeze_shape`` validates whatever the
    last seed/refine produced. The actual authoring (the live model call, the
    persist, the round-trip guard) is the existing service's — this just names the
    operations as tools and returns observations. :meth:`dispatch` never raises on a
    model/user mistake — it returns an ``ok=False`` observation so a loop can echo it
    back and let the caller correct (parallel to :meth:`PlanBuilder.dispatch`)."""

    def __init__(self, service: ShapeAuthorService) -> None:
        self._service = service
        # The most recently authored/refined shape name (the freeze target default).
        self.last_shape: Optional[str] = None
        self.frozen: bool = False
        self.calls: list[dict[str, Any]] = []

    @property
    def available(self) -> bool:
        return self._service.available

    async def seed_shape(self, args: Mapping[str, Any]) -> dict[str, Any]:
        description = str(args.get("description") or "").strip()
        if not description:
            return {"ok": False, "note": "seed_shape needs a non-empty 'description'."}
        try:
            view = await self._service.author(
                description, name_hint=str(args.get("name_hint") or "").strip()
            )
        except ShapeAuthoringUnavailable as exc:
            return {"ok": False, "unavailable": True, "note": str(exc)}
        except ShapeNameConflict as exc:
            return {"ok": False, "conflict": True, "note": str(exc)}
        except Exception as exc:  # MalformedOutputError / ShapeError → unauthorable
            return {"ok": False, "note": f"could not author a shape: {exc}"}
        self.last_shape = view.get("name")
        self.frozen = False
        return {
            "ok": True,
            "name": self.last_shape,
            "execution": view.get("execution"),
            "note": f"authored shape {self.last_shape!r}; refine it or freeze it.",
            "shape": view,
        }

    async def refine_shape(self, args: Mapping[str, Any]) -> dict[str, Any]:
        name = str(args.get("name") or self.last_shape or "").strip()
        instruction = str(args.get("instruction") or "").strip()
        if not name:
            return {"ok": False, "note": "refine_shape needs a 'name' (or seed first)."}
        if not instruction:
            return {"ok": False, "note": "refine_shape needs a non-empty 'instruction'."}
        try:
            view = await self._service.refine(name, instruction)
        except ShapeAuthoringUnavailable as exc:
            return {"ok": False, "unavailable": True, "note": str(exc)}
        except ShapeNotFound as exc:
            return {"ok": False, "missing": True, "note": str(exc)}
        except Exception as exc:
            return {"ok": False, "note": f"could not refine the shape: {exc}"}
        self.last_shape = view.get("name")
        self.frozen = False
        return {
            "ok": True,
            "name": self.last_shape,
            "execution": view.get("execution"),
            "note": f"refined shape {self.last_shape!r}; refine again or freeze it.",
            "shape": view,
        }

    async def freeze_shape(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """Finalize: confirm the shape is in the catalog the SELECTOR harvests.

        A shape is persisted as a runnable TOML on each seed/refine (the round-trip
        guard runs inside the service), so freeze is the explicit "this is done"
        signal — it asserts the shape is SELECTABLE (present in the config service's
        catalog) so the workflow has a clear terminal step parallel to
        ``finalize_plan``."""
        name = str(args.get("name") or self.last_shape or "").strip()
        if not name:
            return {"ok": False, "note": "freeze_shape needs a 'name' (or seed first)."}
        # The config service reads the SAME shapes dir the runtime selector harvests.
        view = self._service._config.get_shape(name)  # noqa: SLF001 (same-package reuse)
        if view is None:
            return {
                "ok": False,
                "note": f"shape {name!r} is not in the selectable catalog — "
                        "seed/refine it before freezing.",
            }
        self.last_shape = name
        self.frozen = True
        return {
            "ok": True,
            "done": True,
            "name": name,
            "execution": view.get("execution"),
            "note": f"shape {name!r} frozen — selectable and runnable.",
            "shape": view,
        }

    async def dispatch(
        self, tool: str, args: Optional[Mapping[str, Any]]
    ) -> dict[str, Any]:
        name = str(tool or "").strip()
        kwargs = dict(args) if isinstance(args, Mapping) else {}
        if name not in SHAPE_AUTHOR_TOOL_NAMES:
            obs = {
                "ok": False,
                "note": f"unknown tool {name!r}; call one of "
                        f"{sorted(SHAPE_AUTHOR_TOOL_NAMES)}.",
            }
        else:
            obs = await getattr(self, name)(kwargs)
        self.calls.append({"tool": name, "ok": bool(obs.get("ok")), "note": obs.get("note", "")})
        return obs


# --------------------------------------------------------------------------- #
# SPEC authoring session — wraps SpecChatService (REUSE, not rebuild)
# --------------------------------------------------------------------------- #
class SpecAuthoringSession:
    """Drive the spec seed→iterate→freeze+compile workflow over :class:`SpecChatService`.

    Holds the one ``session_id`` the service opens on ``seed_spec`` and threads every
    later turn through it, so the refine builds on the prior draft (the iterative seam
    in :class:`specialization.conversation.SpecConversation`). The service methods are
    SYNCHRONOUS blocking chain calls; this session offloads each off the event loop
    with :func:`asyncio.to_thread` (exactly as the HTTP route layer does), so a caller
    on the loop never blocks (d4). Observations mirror :meth:`PlanBuilder.dispatch`."""

    def __init__(self, service: SpecChatService) -> None:
        self._service = service
        self.session_id: Optional[str] = None
        self.compiled_name: Optional[str] = None
        self.frozen: bool = False
        self.calls: list[dict[str, Any]] = []

    async def seed_spec(self, args: Mapping[str, Any]) -> dict[str, Any]:
        name = str(args.get("name") or "").strip()
        description = str(args.get("description") or "").strip()
        intent = str(args.get("intent") or "").strip()
        if not name:
            return {"ok": False, "note": "seed_spec needs a kebab-case 'name'."}
        if not description:
            return {"ok": False, "note": "seed_spec needs a one-line 'description'."}
        if not intent:
            return {"ok": False, "note": "seed_spec needs the 'intent' (what it does)."}
        if self.session_id is not None:
            return {
                "ok": False,
                "note": "a spec session is already seeded; use refine_spec / freeze_spec.",
            }
        try:
            self.session_id = self._service.open_session(name, description)
            preview = await asyncio.to_thread(
                self._service.drive_message, self.session_id, intent
            )
        except Exception as exc:
            return {"ok": False, "note": f"could not author the spec draft: {exc}"}
        self.frozen = False
        return {
            "ok": True,
            "name": preview.name,
            "turn": preview.turn,
            "note": f"authored spec draft for {preview.name!r}; refine or freeze it.",
            "draft": {"description": preview.description, "body": preview.body},
        }

    async def refine_spec(self, args: Mapping[str, Any]) -> dict[str, Any]:
        critique = str(args.get("critique") or "").strip()
        if self.session_id is None:
            return {"ok": False, "note": "refine_spec needs a seeded session (seed_spec first)."}
        if not critique:
            return {"ok": False, "note": "refine_spec needs a non-empty 'critique'."}
        try:
            preview = await asyncio.to_thread(
                self._service.drive_message, self.session_id, critique
            )
        except Exception as exc:
            return {"ok": False, "note": f"could not refine the spec: {exc}"}
        self.frozen = False
        return {
            "ok": True,
            "name": preview.name,
            "turn": preview.turn,
            "note": f"refined spec {preview.name!r} (turn {preview.turn}); refine again or freeze.",
            "draft": {"description": preview.description, "body": preview.body},
        }

    async def freeze_spec(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if self.session_id is None:
            return {"ok": False, "note": "freeze_spec needs a seeded session (seed_spec first)."}
        try:
            state, name, source = await asyncio.to_thread(
                self._service.approve, self.session_id
            )
        except Exception as exc:
            return {"ok": False, "note": f"could not compile/register the spec: {exc}"}
        self.compiled_name = name
        self.frozen = True
        return {
            "ok": True,
            "done": True,
            "name": name,
            "state": state,
            "source": source,
            "note": f"spec {name!r} compiled + registered — loadable on the next run.",
        }

    async def dispatch(
        self, tool: str, args: Optional[Mapping[str, Any]]
    ) -> dict[str, Any]:
        name = str(tool or "").strip()
        kwargs = dict(args) if isinstance(args, Mapping) else {}
        if name not in SPEC_AUTHOR_TOOL_NAMES:
            obs = {
                "ok": False,
                "note": f"unknown tool {name!r}; call one of "
                        f"{sorted(SPEC_AUTHOR_TOOL_NAMES)}.",
            }
        else:
            obs = await getattr(self, name)(kwargs)
        self.calls.append({"tool": name, "ok": bool(obs.get("ok")), "note": obs.get("note", "")})
        return obs


__all__ = [
    "SHAPE_AUTHOR_TOOLS_SPEC",
    "SPEC_AUTHOR_TOOLS_SPEC",
    "SHAPE_AUTHOR_TOOL_NAMES",
    "SPEC_AUTHOR_TOOL_NAMES",
    "shape_author_catalog_text",
    "spec_author_catalog_text",
    "AuthoringToolError",
    "ShapeAuthoringSession",
    "SpecAuthoringSession",
]
