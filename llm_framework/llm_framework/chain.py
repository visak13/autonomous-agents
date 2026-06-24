"""Spring-style, lambda-chainable pipeline core.

The whole agent runtime is a single idea: a *chain of stages*. A **stage** is
ANY callable with the shape ``(ctx: Context) -> Context`` — a plain function, a
factory-produced closure, or a **bare lambda**. New behaviour is added by
``chain.use(stage)`` WITHOUT touching this module (open/closed, "Spring-style"):

    chain.use(lambda ctx: ctx.set("greeted", True))   # bare lambda, no core edit

The chain carries a :class:`Context` — a small mutable state object threaded
through every stage. Stages read from it and write back to it; the chain just
runs them in order. Stages can be appended OR inserted at runtime (``insert``,
``insert_before``, ``insert_after``) so the pipeline can be reshaped while the
agent is live (e.g. the reactive layer in s3 splicing a tool stage in).

This module is deliberately tiny and dependency-free — it is the stable spine
the rest of the framework (stages, context management, tools, memory) hangs
off, and it must stay lean for phi's small context window (d10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Imported only for type clarity / optional carry on the Context; the core does
# not call the transport itself — stages do.
from .transport import Message, Transport

# A stage is any callable (ctx) -> ctx. Returning ``None`` is tolerated and
# treated as "mutated ctx in place" so the very terse bare-lambda form works:
#     chain.use(lambda ctx: ctx.meta.update(seen=True))
Stage = Callable[["Context"], Optional["Context"]]


# --------------------------------------------------------------------------- #
# Context: the state threaded through the chain
# --------------------------------------------------------------------------- #


@dataclass
class Context:
    """Mutable state carried through a :class:`Chain` run.

    Fields are intentionally explicit (not a free-form dict) so the built-in
    stages have a stable contract, while ``vars``/``meta`` stay open for ad-hoc
    stages and lambdas. Everything here is lean by design (d10).

    Lifecycle of the built-in stages:
      - ``system`` / ``history`` / ``user`` are the INPUTS a caller sets.
      - ``prompt_assembly`` composes them into ``messages`` (transport-ready).
      - ``call_stage`` invokes the transport on ``messages`` and writes
        ``raw_output``.
      - ``structured_output`` parses ``raw_output`` JSON into ``structured``.
    """

    # -- inputs the caller sets -------------------------------------------- #
    system: Optional[str] = None
    """Optional system prompt."""
    history: List[Message] = field(default_factory=list)
    """Prior conversation turns, each ``{"role": ..., "content": ...}``."""
    user: Optional[str] = None
    """The current user input (the new turn being processed)."""

    # -- produced by the built-in stages ----------------------------------- #
    messages: List[Message] = field(default_factory=list)
    """Transport-ready message list, built by ``prompt_assembly``."""
    raw_output: Optional[str] = None
    """Raw assistant text from the model, set by ``call_stage``."""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    """Native tool calls from the model (s13), set by ``call_stage`` from
    ``ChatResult.tool_calls``. Normalised ``[{"name", "arguments"}, ...]``; ``None``
    when the request carried no ``tools`` or the reply was plain prose. Lets a ReAct
    loop read a structured tool call directly instead of string-parsing ``raw_output``."""
    structured: Any = None
    """Parsed structured output, set by ``structured_output`` (None if absent)."""

    # -- shared plumbing ---------------------------------------------------- #
    transport: Optional[Transport] = None
    """Optional transport carried on the ctx; stages may take one explicitly
    instead, but carrying it here lets a bare ``call_stage()`` find it (d2:
    everything is in-process, so passing the live transport along is cheap)."""

    # -- open extension points --------------------------------------------- #
    vars: Dict[str, Any] = field(default_factory=dict)
    """Free-form scratch for ad-hoc stages / lambdas."""
    meta: Dict[str, Any] = field(default_factory=dict)
    """Trace + diagnostics (stage trace, call records, repair info). The demo
    reads this to prove each stage's effect."""

    # -- tiny conveniences (keep lambdas terse) ---------------------------- #

    def set(self, key: str, value: Any) -> "Context":
        """Set ``vars[key] = value`` and return self (so a lambda can be one
        expression)."""
        self.vars[key] = value
        return self

    def get(self, key: str, default: Any = None) -> Any:
        return self.vars.get(key, default)

    def trace(self, event: str, **detail: Any) -> None:
        """Append a structured trace entry to ``meta['trace']``."""
        self.meta.setdefault("trace", []).append({"stage": event, **detail})


# --------------------------------------------------------------------------- #
# Chain: ordered, runtime-mutable composition of stages
# --------------------------------------------------------------------------- #


class Chain:
    """An ordered pipeline of stages run left-to-right over a :class:`Context`.

    Composition is fluent and non-destructive of the core: ``use`` appends, the
    ``insert*`` family splices at runtime. Each stage is stored with a name
    (its ``__name__`` by default, or one you pass) purely for tracing and for
    ``insert_before``/``insert_after`` targeting — lambdas get a synthetic name.
    """

    def __init__(self, stages: Optional[Sequence[Stage]] = None) -> None:
        self._stages: List[Tuple[str, Stage]] = []
        for stage in stages or ():
            self.use(stage)

    # -- introspection ----------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._stages)

    @property
    def stage_names(self) -> List[str]:
        return [name for name, _ in self._stages]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Chain({' -> '.join(self.stage_names) or 'empty'})"

    # -- composition ------------------------------------------------------- #

    def use(self, stage: Stage, *, name: Optional[str] = None) -> "Chain":
        """Append ``stage`` to the end. Returns self for fluent chaining."""
        self._stages.append((self._name_of(stage, name), stage))
        return self

    # Alias that reads well at call sites that think in terms of "append".
    append = use

    def insert(self, index: int, stage: Stage, *, name: Optional[str] = None) -> "Chain":
        """Insert ``stage`` at position ``index`` at runtime."""
        self._stages.insert(index, (self._name_of(stage, name), stage))
        return self

    def insert_before(
        self, target: str, stage: Stage, *, name: Optional[str] = None
    ) -> "Chain":
        """Insert ``stage`` immediately before the first stage named ``target``."""
        self._stages.insert(self._index_of(target), (self._name_of(stage, name), stage))
        return self

    def insert_after(
        self, target: str, stage: Stage, *, name: Optional[str] = None
    ) -> "Chain":
        """Insert ``stage`` immediately after the first stage named ``target``."""
        self._stages.insert(
            self._index_of(target) + 1, (self._name_of(stage, name), stage)
        )
        return self

    def remove(self, target: str) -> "Chain":
        """Drop the first stage named ``target`` (runtime un-splice)."""
        self._stages.pop(self._index_of(target))
        return self

    # -- execution --------------------------------------------------------- #

    def run(self, ctx: Optional[Context] = None) -> Context:
        """Run every stage in order over ``ctx`` (a fresh one if omitted).

        A stage that returns ``None`` is taken to have mutated ``ctx`` in place
        (so terse lambdas work); otherwise its return value becomes the new
        ctx threaded onward. Each stage is recorded on ``ctx.meta['trace']``.
        """
        if ctx is None:
            ctx = Context()
        for name, stage in list(self._stages):
            result = stage(ctx)
            if result is not None:
                ctx = result
            ctx.trace(name)
        return ctx

    __call__ = run

    # -- internals --------------------------------------------------------- #

    def _index_of(self, target: str) -> int:
        for i, (name, _) in enumerate(self._stages):
            if name == target:
                return i
        raise KeyError(f"no stage named {target!r} in {self.stage_names}")

    @staticmethod
    def _name_of(stage: Stage, name: Optional[str]) -> str:
        if name:
            return name
        fn_name = getattr(stage, "__name__", None)
        if fn_name and fn_name != "<lambda>":
            return fn_name
        return f"lambda@{id(stage):x}"
