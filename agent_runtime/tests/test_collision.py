"""DAG spec COLLISION + HITL escalation test (s3/Stage-B a5, d11) — fully OFFLINE.

a4 composes a node's N specs autonomously. a5 adds the other half of d11: when two
specs on a node GENUINELY CONFLICT (declare different values on the SAME shaping
axis), the node PAUSES in-flight and asks the user which spec wins — and only a
COMPATIBLE pair (different axes, or the same value) is composed autonomously with
no escalation. This test locks that, end to end, with zero GPU/network:

- DETECTION is deterministic from declarable ``{{directive:axis=value}}`` tags
  (regex, not NLP): verbose-vs-terse on axis ``length`` collides; markdown (no
  directive) + verbose, or two ``length=verbose`` specs, do NOT.
- ESCALATION reuses the awaitable-approver pattern via :class:`CollisionGate`: a
  forced collision FIRES the gate (a pending resolution appears) and the node is
  PAUSED (``RUNNING``, NOT failed) until a real out-of-band ``resolve`` supplies the
  pick; the node then PROCEEDS with the resolved composition. The verify gate +
  CODER=REVIEWER inline-fix lifecycle stays intact over the resolved composition.
- A genuine collision with NO resolver wired FAILS the node cleanly (never a silent
  auto-pick); a compatible 2-spec node never escalates.

Everything is in-process + offline: FakeTransport, no Ollama / network / GPU
(d2/d7/d8). The real produce path runs through ``AgentRuntime`` so the runtime's
``_scopes_for`` escalation seam is exercised end to end.
"""
from __future__ import annotations

import asyncio

from agent_runtime.collision import (
    Collision,
    CollisionGate,
    CollisionGateError,
    CollisionResolution,
    CollisionResolutionError,
    ConflictAxis,
    Directive,
    apply_resolution,
    detect_collision,
    parse_directives,
    strip_directives,
)
from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.runtime import (
    EVENT_NODE_COLLISION,
    EVENT_NODE_COLLISION_RESOLVED,
    EVENT_NODE_FAILED,
    _REVIEWER_FRAMING,
    AgentRuntime,
)
from agent_runtime.scope import ScopedSpec
from agent_runtime.status import NodeStatus
from llm_framework import FakeTransport
from reactive_tools import EventPlane
from specialization.loader import SpecLoader
from specialization.registry import SpecRegistry
from specialization.seed import MARKDOWN_WRITER_RULESET, seed_ruleset_spec


# Two shaping rulesets that COLLIDE on axis ``length`` (verbose vs terse) — each
# carries a tell-tale string (absent from the other) so the test can prove WHICH
# layer landed after resolution, plus the declarable directive tag.
VERBOSE_RULESET = (
    "OUTPUT-SHAPING RULESET. Expand each finding into full, richly detailed "
    "paragraphs with background and context. {{directive:length=verbose}} "
    "TELLTALE-VERBOSE-LAYER."
)
TERSE_RULESET = (
    "OUTPUT-SHAPING RULESET. Compress the answer to the few shortest possible "
    "bullet points; cut every non-essential word. {{directive:length=terse}} "
    "TELLTALE-TERSE-LAYER."
)
# A SECOND verbose ruleset — same axis, SAME value as VERBOSE_RULESET → they AGREE
# (compatible), so they must NOT escalate.
VERBOSE_TWIN_RULESET = (
    "OUTPUT-SHAPING RULESET. Be expansive and thorough throughout. "
    "{{directive:length=verbose}} TELLTALE-VERBOSE-TWIN-LAYER."
)


def _registry(tmp_path):
    reg = SpecRegistry(tmp_path / "specs")
    seed_ruleset_spec(reg, "verbose-writer", "Expand into full detail.", VERBOSE_RULESET)
    seed_ruleset_spec(reg, "terse-editor", "Compress to the minimum.", TERSE_RULESET)
    seed_ruleset_spec(reg, "verbose-twin", "Also expansive.", VERBOSE_TWIN_RULESET)
    seed_ruleset_spec(
        reg, "markdown-writer", "Shape into clean GFM.", MARKDOWN_WRITER_RULESET
    )
    return reg, SpecLoader(reg)


class RecordingPlane(EventPlane):
    """An EventPlane that also records every published ``(kind, payload)`` so the
    test can assert the collision lifecycle fired (or did not)."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, object]] = []

    async def publish(self, kind, payload=None, *, source=None):  # type: ignore[override]
        self.events.append((kind, payload))
        return await super().publish(kind, payload, source=source)

    def kinds(self) -> list[str]:
        return [k for k, _ in self.events]


def _produce_system(transport: FakeTransport, call_index: int = 0) -> str:
    return next(
        m["content"]
        for m in transport.calls[call_index]["messages"]
        if m["role"] == "system"
    )


# --------------------------------------------------------------------------- #
# (1) THE DETERMINISTIC CONFLICT MODEL — pure functions, no runtime.
# --------------------------------------------------------------------------- #
def test_parse_and_strip_directives():
    # parse pulls the declared directive; strip removes the tag but keeps prose.
    assert parse_directives(VERBOSE_RULESET) == (Directive("length", "verbose"),)
    assert parse_directives(TERSE_RULESET) == (Directive("length", "terse"),)
    assert parse_directives("no tags here") == ()
    # case-insensitive + whitespace-tolerant
    assert parse_directives("{{directive: FORMAT = Markdown }}") == (
        Directive("format", "markdown"),
    )

    stripped = strip_directives(VERBOSE_RULESET)
    assert "{{directive" not in stripped
    assert "TELLTALE-VERBOSE-LAYER" in stripped and "richly detailed" in stripped
    # An untagged body is returned BYTE-FOR-BYTE unchanged (a4 back-compat).
    assert strip_directives(MARKDOWN_WRITER_RULESET) == MARKDOWN_WRITER_RULESET


def test_detect_collision_distinguishes_conflict_from_compatible():
    verbose = ScopedSpec.of("verbose-writer", VERBOSE_RULESET)
    terse = ScopedSpec.of("terse-editor", TERSE_RULESET)
    twin = ScopedSpec.of("verbose-twin", VERBOSE_TWIN_RULESET)
    md = ScopedSpec.of("markdown-writer", MARKDOWN_WRITER_RULESET)

    # GENUINE conflict: same axis 'length', different values.
    col = detect_collision("n1", [verbose, terse])
    assert col is not None
    assert [a.axis for a in col.axes] == ["length"]
    opts = dict(col.axes[0].options)
    assert opts == {"verbose-writer": "verbose", "terse-editor": "terse"}
    assert col.spec_names == ("verbose-writer", "terse-editor")
    assert col.challenge.startswith("collision-n1-")

    # COMPATIBLE: same axis, SAME value (they agree) → no collision.
    assert detect_collision("n1", [verbose, twin]) is None
    # COMPATIBLE: different axes (markdown declares none) → no collision.
    assert detect_collision("n1", [md, verbose]) is None
    # Fewer than two specs can never collide.
    assert detect_collision("n1", [verbose]) is None


def test_challenge_is_deterministic():
    a = detect_collision("n1", [ScopedSpec.of("verbose-writer", VERBOSE_RULESET),
                                ScopedSpec.of("terse-editor", TERSE_RULESET)])
    b = detect_collision("n1", [ScopedSpec.of("verbose-writer", VERBOSE_RULESET),
                                ScopedSpec.of("terse-editor", TERSE_RULESET)])
    assert a.challenge == b.challenge  # same inputs → same key (no clock/randomness)


def test_apply_resolution_filters_and_reorders():
    verbose = ScopedSpec.of("verbose-writer", VERBOSE_RULESET)
    terse = ScopedSpec.of("terse-editor", TERSE_RULESET)
    # "which wins" — drop the loser.
    kept = apply_resolution([verbose, terse], CollisionResolution(order=("terse-editor",)))
    assert [s.name for s in kept] == ["terse-editor"]
    # "ordering" — reorder both.
    re = apply_resolution([verbose, terse], CollisionResolution(order=("terse-editor", "verbose-writer")))
    assert [s.name for s in re] == ["terse-editor", "verbose-writer"]
    # A resolution that names no spec on the node is an error.
    try:
        apply_resolution([verbose, terse], CollisionResolution(order=("ghost",)))
        assert False, "expected CollisionResolutionError"
    except CollisionResolutionError:
        pass


# --------------------------------------------------------------------------- #
# (2) ESCALATION via the awaitable CollisionGate: forced collision → pause →
#     real resolution → node proceeds with the resolved composition.
# --------------------------------------------------------------------------- #
def test_forced_collision_escalates_via_gate_then_resumes(tmp_path):
    reg, loader = _registry(tmp_path)
    transport = FakeTransport(["# Report\n\n- terse bullet."])
    plane = RecordingPlane()
    gate = CollisionGate()

    node = PlanNode(
        id="n1",
        task="Research the topic and report the findings.",
        specs=("verbose-writer", "terse-editor"),  # COLLIDE on axis 'length'
    )
    runtime = AgentRuntime(
        transport=transport, loader=loader, plane=plane,
        collision_resolver=gate.resolver,
    )

    async def drive():
        return await runtime.run(PlanDAG(nodes=[node]))

    async def resolve_when_pending():
        # Poll until the gate shows the parked collision — proves escalation FIRED.
        pend = []
        for _ in range(2000):
            pend = await gate.pending()
            if pend:
                break
            await asyncio.sleep(0)  # yield to the suspended node coroutine
        assert pend, "the forced collision never escalated to the gate"
        # The node is PAUSED in-flight: RUNNING, not failed, awaiting the decision.
        assert runtime.states["n1"].status == NodeStatus.RUNNING
        c = pend[0]
        assert c["node_id"] == "n1"
        assert c["axes"][0]["axis"] == "length"
        # The user picks: terse wins (drop verbose).
        receipt = await gate.resolve(c["challenge"], order=["terse-editor"], note="prefer brevity")
        assert receipt["order"] == ["terse-editor"]
        return c

    async def scenario():
        return await asyncio.gather(drive(), resolve_when_pending())

    result, pend = asyncio.run(scenario())

    # The node completed (it did NOT fail while paused) with the RESOLVED comp.
    assert result.states["n1"]["status"] == NodeStatus.DONE.value
    assert transport.call_count == 1
    system = _produce_system(transport)
    # Resolved to terse-only: the terse layer landed, the verbose layer did NOT.
    assert "TELLTALE-TERSE-LAYER" in system
    assert "TELLTALE-VERBOSE-LAYER" not in system
    # Directive metadata tags are stripped from what the model sees.
    assert "{{directive" not in system
    # The result records the resolved composition.
    assert result.results["n1"].specs == ("terse-editor",)
    # Lifecycle observability: the collision pause + resume fired; no FAILED event.
    assert EVENT_NODE_COLLISION in plane.kinds()
    assert EVENT_NODE_COLLISION_RESOLVED in plane.kinds()
    assert EVENT_NODE_FAILED not in plane.kinds()


# --------------------------------------------------------------------------- #
# (3) A COMPATIBLE 2-spec node composes AUTONOMOUSLY — no escalation.
# --------------------------------------------------------------------------- #
def test_compatible_two_specs_do_not_escalate(tmp_path):
    reg, loader = _registry(tmp_path)

    calls = {"n": 0}

    async def recording_resolver(collision: Collision) -> CollisionResolution:
        calls["n"] += 1  # must NEVER be hit for a compatible pair
        return CollisionResolution(order=collision.spec_names)

    # (a) different axes: markdown (no directive) + verbose → compatible.
    transport_a = FakeTransport(["# ok"])
    plane_a = RecordingPlane()
    asyncio.run(
        AgentRuntime(
            transport=transport_a, loader=loader, plane=plane_a,
            collision_resolver=recording_resolver,
        ).run(PlanDAG(nodes=[PlanNode(id="n1", task="t.", specs=("markdown-writer", "verbose-writer"))]))
    )
    sys_a = _produce_system(transport_a)
    assert "## Sources" in sys_a and "TELLTALE-VERBOSE-LAYER" in sys_a  # BOTH layered
    assert "{{directive" not in sys_a                                   # tags stripped
    assert EVENT_NODE_COLLISION not in plane_a.kinds()

    # (b) same axis, SAME value: verbose + verbose-twin → they agree → compatible.
    transport_b = FakeTransport(["# ok"])
    plane_b = RecordingPlane()
    asyncio.run(
        AgentRuntime(
            transport=transport_b, loader=loader, plane=plane_b,
            collision_resolver=recording_resolver,
        ).run(PlanDAG(nodes=[PlanNode(id="n1", task="t.", specs=("verbose-writer", "verbose-twin"))]))
    )
    sys_b = _produce_system(transport_b)
    assert "TELLTALE-VERBOSE-LAYER" in sys_b and "TELLTALE-VERBOSE-TWIN-LAYER" in sys_b
    assert EVENT_NODE_COLLISION not in plane_b.kinds()

    assert calls["n"] == 0  # the resolver was never consulted for a compatible node


# --------------------------------------------------------------------------- #
# (4) A genuine collision with NO resolver wired FAILS cleanly (no silent pick).
# --------------------------------------------------------------------------- #
def test_genuine_collision_without_resolver_fails_clean(tmp_path):
    reg, loader = _registry(tmp_path)
    transport = FakeTransport(["should never be produced"])
    plane = RecordingPlane()
    node = PlanNode(id="n1", task="t.", specs=("verbose-writer", "terse-editor"))
    result = asyncio.run(
        AgentRuntime(transport=transport, loader=loader, plane=plane).run(PlanDAG(nodes=[node]))
    )
    assert result.states["n1"]["status"] == NodeStatus.FAILED.value
    assert "n1" in result.failed
    assert "collision" in result.failed["n1"].lower() or "conflict" in result.failed["n1"].lower()
    # The conflict was caught BEFORE any produce call — nothing was auto-composed.
    assert transport.call_count == 0
    assert EVENT_NODE_FAILED in plane.kinds()
    assert EVENT_NODE_COLLISION not in plane.kinds()  # no channel → no escalation


# --------------------------------------------------------------------------- #
# (5) VERIFY-GATE + CODER=REVIEWER lifecycle stays intact over the RESOLVED comp:
#     the inline reviewer re-uses the SAME resolved composition (escalation is NOT
#     re-asked).
# --------------------------------------------------------------------------- #
def test_inline_review_reuses_resolved_composition(tmp_path):
    reg, loader = _registry(tmp_path)
    # produce -> rejected (no '# ' heading); inline review -> accepted (has heading).
    transport = FakeTransport(
        [
            "no heading; terse bullets only.",
            "# Fixed report\n\n- terse bullet.",
        ],
        strict=True,
    )
    calls = {"n": 0}

    async def resolver(collision: Collision) -> CollisionResolution:
        calls["n"] += 1
        return CollisionResolution(order=("terse-editor",))  # terse wins

    def verifier(node, result):
        out = (result.output or "").lstrip()
        return True if out.startswith("# ") else (False, "must open with a '# ' heading")

    node = PlanNode(id="n1", task="research and report.", specs=("verbose-writer", "terse-editor"))
    runtime = AgentRuntime(
        transport=transport, loader=loader, collision_resolver=resolver,
        verifier=verifier, max_inline_fixes=1,
    )
    result = asyncio.run(runtime.run(PlanDAG(nodes=[node])))

    assert result.states["n1"]["status"] == NodeStatus.DONE.value
    assert result.states["n1"]["inline_fixed"] is True
    assert transport.call_count == 2

    produce_sys = _produce_system(transport, 0)
    review_sys = _produce_system(transport, 1)
    # BOTH the producer and the reviewer ran the SAME resolved (terse-only) comp.
    for s in (produce_sys, review_sys):
        assert "TELLTALE-TERSE-LAYER" in s
        assert "TELLTALE-VERBOSE-LAYER" not in s
        assert "{{directive" not in s
    assert _REVIEWER_FRAMING in review_sys and _REVIEWER_FRAMING not in produce_sys
    # The collision was escalated EXACTLY ONCE (cached) — the reviewer did NOT re-ask.
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# (6) The CollisionGate validates real decisions (the HITL surface contract).
# --------------------------------------------------------------------------- #
def test_collision_gate_validates_decisions():
    gate = CollisionGate()
    collision = Collision(
        node_id="n1",
        spec_names=("verbose-writer", "terse-editor"),
        axes=(ConflictAxis(axis="length",
                           options=(("verbose-writer", "verbose"), ("terse-editor", "terse"))),),
        challenge="collision-n1-deadbeef",
    )

    async def scenario():
        # Resolving an unknown challenge is rejected.
        try:
            await gate.resolve("nope", order=["terse-editor"])
            assert False, "expected CollisionGateError for unknown challenge"
        except CollisionGateError:
            pass

        # Park the collision on its awaitable resolver, then drive real decisions.
        task = asyncio.create_task(gate.resolver(collision))
        for _ in range(2000):
            if await gate.has_pending(collision.challenge):
                break
            await asyncio.sleep(0)
        assert await gate.has_pending(collision.challenge)

        # Empty pick and an unknown-spec pick are both rejected (the wait stays live).
        try:
            await gate.resolve(collision.challenge, order=[])
            assert False, "expected error for empty order"
        except CollisionGateError:
            pass
        try:
            await gate.resolve(collision.challenge, order=["ghost-spec"])
            assert False, "expected error for unknown spec"
        except CollisionGateError:
            pass
        assert not task.done()  # the node is still paused, the bad picks didn't resolve it

        # A valid pick un-blocks the awaiting resolver with the chosen composition.
        receipt = await gate.resolve(collision.challenge, order=["terse-editor"])
        assert receipt["order"] == ["terse-editor"]
        resolution = await task
        assert resolution.order == ("terse-editor",)

        # It is no longer pending; a second decision is rejected.
        assert not await gate.has_pending(collision.challenge)
        try:
            await gate.resolve(collision.challenge, order=["terse-editor"])
            assert False, "expected error for an already-resolved collision"
        except CollisionGateError:
            pass

    asyncio.run(scenario())


def test_collision_gate_cancel_all_unblocks_pending():
    gate = CollisionGate()
    collision = Collision(
        node_id="n1", spec_names=("a", "b"),
        axes=(ConflictAxis(axis="x", options=(("a", "1"), ("b", "2"))),),
        challenge="collision-n1-cancel",
    )

    async def scenario():
        task = asyncio.create_task(gate.resolver(collision))
        for _ in range(2000):
            if await gate.has_pending(collision.challenge):
                break
            await asyncio.sleep(0)
        n = await gate.cancel_all()
        assert n == 1
        try:
            await task
            assert False, "cancel_all should raise CancelledError into the awaiter"
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())
