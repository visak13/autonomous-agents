"""Self-heal: detect a failed logical step, correct it, re-launch in-process.

Self-heal (d2/o6) is the runtime's resilience seam. A *logical step* — the
planner's DAG emission, or a sub-agent's node execution — can fail two ways the
runtime is expected to recover from autonomously:

1. **Malformed phi JSON.** phi returns text that isn't the structured value the
   step needs. The first line of defence is the llm_framework
   :func:`structured_output` stage's BOUNDED repair loop (re-prompt → re-parse);
   the second line, here, is to detect that repair was exhausted
   (:class:`MalformedOutputError`) and re-launch the step from scratch.
2. **Tool error.** A tool invoked through the hook returns ``ok=False``
   (:class:`ToolFailureError`). The correction is a BOUNDED *re-plan* of that
   step's logic — a corrector callback adjusts the approach — and the step is
   re-launched.

:class:`SelfHeal` wraps ANY async "logic" callable with bounded, classified
retries. It is deliberately generic: the planner wraps its plan() call; the
runtime wraps each node's execution. Every heal is recorded on a
:class:`HealLog` so the smoke can prove a failure was detected, corrected, and
the logic re-launched with the task still completing.

In-process, dependency-free (d2/d10): re-launch is just calling the coroutine
again on the same event loop — never a process fork.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


# --------------------------------------------------------------------------- #
# Failure taxonomy
# --------------------------------------------------------------------------- #


class HealableError(Exception):
    """Base for failures the self-heal layer knows how to recover from."""


class MalformedOutputError(HealableError):
    """phi produced output that could not be parsed into the needed structure,
    even after the bounded structured-output repair loop. Correction: re-launch
    the step (a fresh attempt), optionally after a corrector tweaks the prompt."""


class ToolFailureError(HealableError):
    """A tool invoked through the hook failed (``ToolResult.ok == False``).
    Correction: a bounded re-plan of the step's logic, then re-launch."""

    def __init__(self, message: str, *, tool: str = "", call_id: int = -1) -> None:
        super().__init__(message)
        self.tool = tool
        self.call_id = call_id


class InvalidStepError(HealableError):
    """A step completed but produced a LOGICALLY-INVALID result (e.g. empty /
    schema-violating output) — a third failure mode the runtime self-heals.

    Distinct from :class:`MalformedOutputError` (the *planner's* JSON could not be
    parsed) and :class:`ToolFailureError` (a *tool* raised): here the phi call
    returned, but a result validator rejected the value. Correction: re-launch
    the step (bounded), and if node-level retries are exhausted the runtime
    re-derives a corrective sub-graph for just that node (re-plan)."""

    def __init__(self, message: str, *, node_id: str = "", reason: str = "") -> None:
        super().__init__(message)
        self.node_id = node_id
        self.reason = reason


# --------------------------------------------------------------------------- #
# Heal trace
# --------------------------------------------------------------------------- #


@dataclass
class HealAttempt:
    """One detected-and-corrected failure within a :class:`SelfHeal.run`."""

    attempt: int
    failure_type: str
    error: str
    correction: str
    relaunched: bool = True


@dataclass
class HealLog:
    """The full self-heal trace for one wrapped logic call.

    ``attempts`` is empty when the logic succeeded on the first try. ``healed``
    is True when at least one failure was detected, corrected, and the eventual
    re-launch succeeded. ``exhausted`` is True when the bound was hit without
    success."""

    label: str = ""
    attempts: list[HealAttempt] = field(default_factory=list)
    healed: bool = False
    exhausted: bool = False
    final_error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "attempts": [a.__dict__ for a in self.attempts],
            "healed": self.healed,
            "exhausted": self.exhausted,
            "final_error": self.final_error,
        }


# A corrector is called with (error, attempt_number) BEFORE the next re-launch.
# It may mutate shared state (e.g. tweak a prompt, pick a different tool) and
# returns a one-line human description of the correction it made (for the log).
Corrector = Callable[[HealableError, int], str]


class SelfHeal:
    """Run an async logic callable with bounded, classified self-healing.

    Parameters
    ----------
    max_heals:
        Maximum number of corrected re-launches before giving up (the original
        attempt is not counted, so ``max_heals=2`` means up to 3 total runs).
    on_malformed / on_tool_error:
        Optional correctors invoked before re-launching after the matching
        failure class. Each returns a one-line description recorded on the log.
        If omitted, a default "re-launch unchanged" / "bounded re-plan" note is
        recorded and the logic is simply retried.
    """

    def __init__(
        self,
        *,
        max_heals: int = 2,
        on_malformed: Optional[Corrector] = None,
        on_tool_error: Optional[Corrector] = None,
        on_invalid_step: Optional[Corrector] = None,
    ) -> None:
        self.max_heals = max_heals
        self.on_malformed = on_malformed
        self.on_tool_error = on_tool_error
        self.on_invalid_step = on_invalid_step

    def _classify(self, exc: HealableError, attempt: int) -> tuple[str, str]:
        """Return ``(failure_type, correction)`` for one healable failure.

        Picks the matching corrector (or a sensible default note) per failure
        class. Keeping this in one place means every healable class — malformed
        JSON, tool error, invalid step — gets the SAME bounded re-launch
        treatment and a uniform give-up path."""
        if isinstance(exc, MalformedOutputError):
            corrector, default, label = (
                self.on_malformed,
                "re-launch step from scratch (malformed phi JSON)",
                "malformed_json",
            )
        elif isinstance(exc, ToolFailureError):
            corrector, default, label = (
                self.on_tool_error,
                f"bounded re-plan of step after tool {exc.tool!r} failed",
                "tool_error",
            )
        elif isinstance(exc, InvalidStepError):
            corrector, default, label = (
                self.on_invalid_step,
                f"re-launch step after logically-invalid result ({exc.reason})",
                "invalid_step",
            )
        else:  # pragma: no cover - defensive; HealableError subclasses are enumerated
            corrector, default, label = (None, "re-launch step", "healable")
        correction = corrector(exc, attempt) if corrector else default
        return label, correction

    async def run(
        self,
        logic: Callable[[], Awaitable[Any]],
        *,
        label: str = "",
        log: Optional[HealLog] = None,
    ) -> Any:
        """Run ``logic()``; on a :class:`HealableError`, correct and re-launch.

        Returns the logic's result on success. Re-raises the last error (and
        marks the log ``exhausted``) once ``max_heals`` corrected re-launches
        have failed. Non-:class:`HealableError` exceptions are NOT caught — only
        the failure classes the runtime is meant to self-heal are recovered."""
        hl = log if log is not None else HealLog(label=label)
        hl.label = hl.label or label
        attempt = 0
        while True:
            try:
                result = await logic()
                if attempt > 0:
                    hl.healed = True
                return result
            except HealableError as exc:
                if attempt >= self.max_heals:
                    hl.exhausted = True
                    hl.final_error = f"{type(exc).__name__}: {exc}"
                    raise
                attempt += 1
                failure_type, correction = self._classify(exc, attempt)
                hl.attempts.append(
                    HealAttempt(attempt, failure_type, str(exc), correction)
                )


__all__ = [
    "HealableError",
    "MalformedOutputError",
    "ToolFailureError",
    "InvalidStepError",
    "HealAttempt",
    "HealLog",
    "SelfHeal",
    "Corrector",
]
