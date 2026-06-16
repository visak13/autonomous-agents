"""Agent-facing reactive-lambda TOOLS — the first-class capability the LLM uses.

These wrap a :class:`~reactive_tools.subscriptions.LambdaRegistry` as ordinary
tools registered in the GLOBAL tool registry (d12: tool availability is global —
every agent / every LLM call can reach them). Through them the agent itself
CREATES, COMPOSES, LISTS, and CLOSES reactive subscriptions at scale, as suits a
task. The ``list_subscriptions`` tool doubles as the programmatic read-only
surface; the USER's observe-only view is the registry's meta-plane (d15).

The tools are ``async`` because creating a lambda spawns its driver task on the
running event loop (:meth:`LambdaRegistry.create`) — the hook awaits async tools
directly on the loop, so ``asyncio.create_task`` is valid inside them.

Every call still flows on the event plane like any other tool (the hook emits
``tool_call`` / ``tool_result``) — so the act of an agent creating a lambda is
itself observable, and a lambda could even observe lambda-management activity.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from .subscriptions import LambdaInputError, LambdaRegistry


def register_lambda_tools(
    hook: Any, registry: LambdaRegistry, *, source_plane: Any = None
) -> Any:
    """Register the reactive-lambda tools onto ``hook`` (a :class:`ToolHook`).

    ``registry`` is the shared :class:`LambdaRegistry` bound to the same plane the
    run engine + tools publish on. Returns the hook for chaining. Attaches the
    registry as ``hook.subscriptions`` so a host (e.g. the chat app's read-only
    UI endpoints) can reach the live-subscriptions surface without re-plumbing.

    ``source_plane`` (s9/a2) — when given, the ``create_subscription`` /
    ``compose_subscriptions`` tools OBSERVE that plane instead of the registry's
    own. A host serving per-run streams (the agent runtime's per-chat plane) wires
    this so an agent that creates a lambda mid-run observes THIS run's live
    events, while the lambda is still recorded on the shared registry the UI
    reads. ``None`` keeps the default (observe the registry's own plane)."""

    async def create_subscription(
        kinds: Sequence[str],
        label: str = "",
        reducer: str = "each",
        reaction: str = "advisory",
        owner: Optional[Mapping[str, Any]] = None,
        max_fires: Optional[int] = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Create a reactive lambda that OBSERVES ``kinds`` on the event plane.

        ``kinds``: event kinds to watch (e.g. ``["agent_node_done"]``; ``[]`` =
        all). ``reducer``: ``each`` | ``every:N`` | ``sample:N`` |
        ``distinct:<field>`` | ``match:<field>=<value>`` — a data-plane kind
        (``tool_call``/``tool_result``) REQUIRES a non-``each`` reducer.
        ``reaction``: ``advisory`` (emit a governed observation per fire) or
        ``count``. ``max_fires``: auto-close after N fires. Returns the created
        lambda's read-only view."""
        rec = registry.create(
            kinds,
            label=label,
            reducer=reducer,
            reaction=reaction,
            owner=owner,
            max_fires=max_fires,
            note=note,
            source_plane=source_plane,
        )
        return rec.as_view()

    async def compose_subscriptions(
        sub_ids: Sequence[str],
        label: str = "",
        reducer: str = "each",
        reaction: str = "advisory",
        owner: Optional[Mapping[str, Any]] = None,
        max_fires: Optional[int] = None,
    ) -> dict[str, Any]:
        """Compose existing lambdas into a NEW one over the UNION of their kinds.

        The merge combinator: fuse several lambdas the agent created into one
        higher-order lambda (reduced as a single stream). Returns the new lambda's
        read-only view (with ``composed_from`` lineage)."""
        rec = registry.compose(
            sub_ids,
            label=label,
            reducer=reducer,
            reaction=reaction,
            owner=owner,
            max_fires=max_fires,
            source_plane=source_plane,
        )
        return rec.as_view()

    async def list_subscriptions(include_closed: bool = True) -> dict[str, Any]:
        """READ-ONLY: list every agent-created lambda (the live-subscriptions view).

        Returns ``{"active": <n>, "total": <n>, "subscriptions": [...]}`` — each
        entry the observe-only projection the UI lambda-tab renders (sub_id, what
        it observes, owner, status, fire counters). Pure read; never authors."""
        subs = registry.snapshot(include_closed=include_closed)
        return {
            "active": registry.active_count,
            "total": len(registry.snapshot(include_closed=True)),
            "subscriptions": subs,
        }

    async def close_subscription(sub_id: str) -> dict[str, Any]:
        """Close ONE agent lambda by id (clean teardown; idempotent)."""
        ok = await registry.close(sub_id)
        if not ok:
            raise LambdaInputError(f"unknown subscription {sub_id!r}")
        rec = registry.get(sub_id)
        return rec.as_view() if rec is not None else {"sub_id": sub_id, "status": "closed"}

    hook.register(
        "create_subscription", create_subscription,
        description="Create a reactive lambda the agent uses: observe event kinds, "
                    "reduced (each|every:N|sample:N|distinct:F|match:F=V), advisory reaction.",
    )
    hook.register(
        "compose_subscriptions", compose_subscriptions,
        description="Compose existing lambdas into one over the union of their kinds (merge).",
    )
    hook.register(
        "list_subscriptions", list_subscriptions,
        description="Read-only: list the agent-created reactive lambdas (the live-subscriptions view).",
    )
    hook.register(
        "close_subscription", close_subscription,
        description="Close one agent reactive lambda by id (clean teardown).",
    )

    # Expose the registry on the hook so the host's read-only UI surface can read
    # the live-subscriptions snapshot + meta-plane without re-plumbing.
    hook.subscriptions = registry  # type: ignore[attr-defined]
    return hook


__all__ = ["register_lambda_tools"]
