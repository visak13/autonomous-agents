"""The context-scoped sub-agent LOADER (d10).

When the planner launches a sub-agent to execute a task, that sub-agent must be
aware of ONLY the task at hand plus its ONE compiled specialization — never the
full registry, never other specs' bodies, never the planner-facing index. This
loader is exactly that narrow window: given a spec NAME, it returns that single
spec's compiled body and nothing else.

Why a separate class (not just ``registry.load``)
-------------------------------------------------
The :class:`~specialization.registry.SpecRegistry` carries BOTH the
planner-facing :meth:`~specialization.registry.SpecRegistry.index` (the lookup
the planner reasons over) AND the loader. Handing a sub-agent the whole registry
would hand it the index + every other spec by name — leaking the d10 scope. The
:class:`SpecLoader` wraps the registry and re-exposes ONLY the single-body load,
so a sub-agent that holds a loader *cannot* enumerate or read anything beyond the
one spec it is told to load. The scope is enforced by the surface, not by
discipline.

In-process, dependency-free (d2/d10).
"""
from __future__ import annotations

from specialization.model import CompiledSpec
from specialization.registry import SpecRegistry


class SpecLoader:
    """Loads EXACTLY ONE compiled spec body by name — the sub-agent's whole scope.

    Deliberately minimal: no ``index``, no ``names``, no listing. A sub-agent
    given a loader can load the single spec it was launched with and nothing
    else (d10)."""

    def __init__(self, registry: SpecRegistry) -> None:
        # Held privately so the wrapped registry's broader surface (index/names)
        # is NOT re-exposed through the loader.
        self._registry = registry

    def load_body(self, name: str) -> str:
        """Return ONLY the compiled BODY string for the one named spec.

        This is the sub-agent's whole grounding (the ruleset it executes with).
        Raises ``KeyError`` if no such spec is registered."""
        return self._registry.load(name).body

    def load(self, name: str) -> CompiledSpec:
        """Return the full :class:`CompiledSpec` for the one named spec.

        Still scoped to a SINGLE spec (you must name it) — provided for call
        sites that also need the frontmatter (description/source/provenance)
        alongside the body, e.g. to label the launched sub-agent."""
        return self._registry.load(name)


__all__ = ["SpecLoader"]
