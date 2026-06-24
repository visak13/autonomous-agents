"""Context-scoping enforced BY CONSTRUCTION (d10), not by convention.

Stage A scoped the two roles *by surface*: the planner held a factory built from
a body-free index, and a sub-agent held a :class:`~specialization.loader.SpecLoader`
but only loaded the one spec it was named. That is correct but it is still a
*discipline* — a sub-agent holding the whole loader could, with a code change,
load a second spec. Stage B closes that to a STRUCTURAL guarantee:

- :class:`ScopedSpec` — the capability a sub-agent is handed instead of a loader.
  It resolves EXACTLY ONE ``{name, body}`` at construction and exposes nothing
  that can enumerate or load another spec. A sub-agent that holds a ``ScopedSpec``
  literally cannot reach a second body — there is no loader, no registry, no
  index on its object graph (machine-checkable via :meth:`assert_no_loader`).
- :class:`PlannerScope` — the capability the planner is handed. It wraps ONLY the
  :class:`~agent_runtime.factory.AbstractPlanFactory` (which itself carries only
  the body-free lookup index), and re-exposes only the three planner verbs
  (context / prompt / parse). It holds no loader and no registry, so the planner
  code path cannot reach a compiled spec body. :meth:`assert_scoped` walks the
  object graph and proves it.

Both are frozen, dependency-light, in-process (d2/d10).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from .factory import AbstractPlanFactory, PlanDAG, PlanError


class ScopeViolation(PlanError):
    """A scope capability was found to reach beyond its single allowed view (d10)."""


class _SingleBodyLoader(Protocol):
    """The narrow surface :class:`ScopedSpec` needs: load ONE body by name.

    :class:`~specialization.loader.SpecLoader` satisfies this. We type against
    the method we use, not the concrete class, so the runtime can also resolve a
    ``ScopedSpec`` from any single-body source (a stub, a test double)."""

    def load_body(self, name: str) -> str: ...  # pragma: no cover - structural


# Attribute names that, if reachable on a sub-agent's object graph, would mean it
# can reach more than its one spec — the structural d10 violation we forbid.
_FORBIDDEN_SUBAGENT_ATTRS = ("_loader", "loader", "_registry", "registry")


@dataclass(frozen=True)
class ScopedSpec:
    """ONE resolved ``{name, body}`` — a sub-agent's entire specialization view.

    Constructed via :meth:`resolve` (from a single-body loader) or :meth:`of`
    (from an already-resolved body). Once built it is immutable and carries only
    the single body: there is no method to load, list, or reach another spec.
    This is the by-construction d10 enforcement for the sub-agent side — the
    runtime owns the loader and resolves exactly one body, handing the sub-agent
    only this capability.
    """

    name: str
    body: str

    @classmethod
    def resolve(cls, loader: _SingleBodyLoader, name: str) -> "ScopedSpec":
        """Resolve the single body for ``name`` through a single-body loader.

        The loader is used HERE (in the runtime's trust boundary) and is NOT
        retained on the returned capability — so the body is captured once and
        the loader never travels to the sub-agent."""
        if not name or not isinstance(name, str):
            raise ScopeViolation(f"ScopedSpec.resolve needs a spec name, got {name!r}")
        body = loader.load_body(name)
        return cls(name=name, body=str(body))

    @classmethod
    def of(cls, name: str, body: str) -> "ScopedSpec":
        """Build a scope from an already-resolved body (no loader involved)."""
        return cls(name=str(name), body=str(body))

    @staticmethod
    def assert_no_loader(agent: Any) -> None:
        """Raise :class:`ScopeViolation` if ``agent`` can reach a loader/registry.

        The machine-checkable proof that a launched sub-agent is scoped by
        CONSTRUCTION: it holds a single resolved body and NOTHING through which a
        second spec could be loaded. Checked against the live object's ``__dict__``
        (and slots), so it reflects what the agent can actually reach."""
        reachable: dict[str, Any] = {}
        reachable.update(getattr(agent, "__dict__", {}) or {})
        for slot in getattr(type(agent), "__slots__", ()) or ():
            if hasattr(agent, slot):
                reachable[slot] = getattr(agent, slot)
        for attr in _FORBIDDEN_SUBAGENT_ATTRS:
            val = reachable.get(attr)
            if val is not None:
                raise ScopeViolation(
                    f"sub-agent leaks a '{attr}' ({type(val).__name__}) — it can "
                    f"reach beyond its one spec (d10 by-construction violation)"
                )


class PlannerScope:
    """The planner's whole world: an :class:`AbstractPlanFactory` and nothing else.

    The factory is built from the body-free :meth:`SpecRegistry.index` rows, so
    by construction the planner cannot reach a spec body. ``PlannerScope`` makes
    that explicit and re-exposes ONLY the three planner verbs; it holds no
    loader and no registry. :meth:`assert_scoped` proves the object graph carries
    neither a loader/registry nor a spec body (mirrors
    :meth:`AbstractPlanFactory.assert_body_free` for the whole capability)."""

    __slots__ = ("_factory",)

    # Attribute names that would mean the planner can reach a body/loader.
    _FORBIDDEN = ("_loader", "loader", "_registry", "registry", "spec_body", "body")

    def __init__(self, factory: AbstractPlanFactory) -> None:
        if not isinstance(factory, AbstractPlanFactory):
            raise ScopeViolation(
                "PlannerScope wraps an AbstractPlanFactory (body-free lookup) only"
            )
        self._factory = factory

    @property
    def factory(self) -> AbstractPlanFactory:
        return self._factory

    # -- the only three verbs the planner needs (all body-free) ----------- #
    def planner_context(self, goal: str) -> dict[str, Any]:
        ctx = self._factory.planner_context(goal)
        self._factory.assert_body_free(ctx)  # belt-and-braces
        return ctx

    def planner_prompt(self, goal: str) -> tuple[str, str]:
        return self._factory.planner_prompt(goal)

    def parse_dag(self, structured: Any) -> PlanDAG:
        return self._factory.parse_dag(structured)

    def parse_dag_safe(self, structured: Any) -> tuple[PlanDAG, list[str]]:
        return self._factory.parse_dag_safe(structured)

    def assert_scoped(self) -> None:
        """Raise if this scope (or a built context) could reach a body/loader."""
        for attr in self._FORBIDDEN:
            if attr == "_factory":
                continue
            if getattr(self, attr, None) is not None:
                raise ScopeViolation(f"planner scope leaks '{attr}' (d10)")
        # The factory must itself be body-free for any goal it serves.
        self._factory.assert_body_free(self._factory.planner_context("<probe>"))


__all__ = [
    "ScopedSpec",
    "PlannerScope",
    "ScopeViolation",
]
