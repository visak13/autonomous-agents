"""DAG SPEC-COMPOSITION test (s3/Stage-B a4) — 1 task -> N specs, fully OFFLINE.

a4 lets ONE DAG node carry MULTIPLE specs (``specs: list[str]``) and composes
their ruleset bodies into the produce-step SYSTEM in a deterministic, documented
LAYERED order (eda-base3 ``assemble_ruleset`` style). This test locks that
behaviour and proves it did not regress the single-spec / bare-node paths:

- A node with TWO specs -> the produce SYSTEM carries the single ``_SHAPING_
  FRAMING`` ONCE, then BOTH ruleset bodies, each under its labelled
  ``_RULESET_LAYER_HEADER`` separator, in the node's ``specs`` ORDER; the USER
  turn carries the REAL task (the ruleset never leaks into the user turn).
- COMPOSITION ORDER is the node's spec-list order (reversing ``specs`` reverses
  the layer order in the system) — deterministic + documented.
- The CODER=REVIEWER inline fix re-uses the SAME composed (2-spec) system.
- BACK-COMPAT: a single-spec node composes to EXACTLY ``{framing}\n\n{body}``
  with NO layer header; a bare node still gets NO system shaping layer at all.

Everything is in-process + offline: FakeTransport, no Ollama / network / GPU
(d2/d7/d8). Drives the REAL produce path through ``AgentRuntime`` (which exercises
the runtime's ``_resolve_scopes`` -> ``SubAgent`` composition seam end to end).
"""
from __future__ import annotations

import asyncio

from agent_runtime.factory import PlanDAG, PlanNode
from agent_runtime.runtime import (
    _RULESET_LAYER_HEADER,
    _REVIEWER_FRAMING,
    _SHAPING_FRAMING,
    AgentRuntime,
    SubAgent,
)
from agent_runtime.status import NodeStatus
from llm_framework import FakeTransport
from specialization.loader import SpecLoader
from specialization.registry import SpecRegistry
from specialization.seed import MARKDOWN_WRITER_RULESET, seed_ruleset_spec


# A second, distinctive shaping ruleset to compose ALONGSIDE markdown-writer —
# its body has tell-tale strings absent from the markdown ruleset so the test can
# prove BOTH layers landed and in which order.
BREVITY_RULESET = (
    "You are an OUTPUT-SHAPING ruleset. After applying any other shaping rules, "
    "ENFORCE BREVITY: keep the whole answer under 120 words, delete hedging and "
    "filler, and prefer the shortest phrasing that preserves every fact. "
    "TELLTALE-BREVITY-LAYER."
)


def _two_spec_registry(tmp_path):
    reg = SpecRegistry(tmp_path / "specs")
    seed_ruleset_spec(
        reg, "markdown-writer",
        "Shape findings into clean GFM (headings, lists, summary, sources).",
        MARKDOWN_WRITER_RULESET,
    )
    seed_ruleset_spec(
        reg, "brevity-editor",
        "Tighten the answer: enforce a hard word budget, cut filler.",
        BREVITY_RULESET,
    )
    return reg, SpecLoader(reg)


def _produce_system(transport: FakeTransport, call_index: int = 0) -> str:
    return next(
        m["content"]
        for m in transport.calls[call_index]["messages"]
        if m["role"] == "system"
    )


# --------------------------------------------------------------------------- #
# N-SPEC: both rulesets land, layered, in the documented (specs-list) order.
# --------------------------------------------------------------------------- #
def test_two_specs_compose_both_rulesets_layered_in_order(tmp_path):
    reg, loader = _two_spec_registry(tmp_path)
    transport = FakeTransport(["# Report\n\n**Summary.** tight findings."])

    node = PlanNode(
        id="n1",
        task="Research the topic and report the findings.",
        specs=("markdown-writer", "brevity-editor"),  # composition order
    )
    dag = PlanDAG(nodes=[node])
    runtime = AgentRuntime(transport=transport, loader=loader)
    result = asyncio.run(runtime.run(dag))

    assert result.states["n1"]["status"] == NodeStatus.DONE.value
    assert transport.call_count == 1
    system = _produce_system(transport)
    user = next(m["content"] for m in transport.calls[0]["messages"] if m["role"] == "user")

    # ONE shaping framing wraps the WHOLE stack (not repeated per layer).
    assert _SHAPING_FRAMING in system
    assert system.count(_SHAPING_FRAMING) == 1
    # BOTH ruleset bodies are present...
    assert MARKDOWN_WRITER_RULESET.strip() in system
    assert BREVITY_RULESET.strip() in system
    assert "## Sources" in system and "TELLTALE-BREVITY-LAYER" in system
    # ...each under its labelled layer header, in the specs-list ORDER.
    h1 = _RULESET_LAYER_HEADER.format(i=1, n=2, name="markdown-writer")
    h2 = _RULESET_LAYER_HEADER.format(i=2, n=2, name="brevity-editor")
    assert h1 in system and h2 in system
    assert system.index(h1) < system.index(h2)
    assert system.index(MARKDOWN_WRITER_RULESET.strip()) < system.index(BREVITY_RULESET.strip())

    # The USER turn is the REAL task; the rulesets never leak into it.
    assert "report the findings" in user
    assert "## Sources" not in user and "TELLTALE-BREVITY-LAYER" not in user

    # The result records the FULL ordered composition + a primary spec.
    assert result.results["n1"].specs == ("markdown-writer", "brevity-editor")
    assert result.results["n1"].spec == "markdown-writer"


def test_composition_order_follows_the_spec_list(tmp_path):
    """Reversing the node's ``specs`` reverses the layer order — deterministic."""
    reg, loader = _two_spec_registry(tmp_path)
    transport = FakeTransport(["# ok"])
    node = PlanNode(
        id="n1",
        task="do the task.",
        specs=("brevity-editor", "markdown-writer"),  # REVERSED
    )
    runtime = AgentRuntime(transport=transport, loader=loader)
    asyncio.run(runtime.run(PlanDAG(nodes=[node])))
    system = _produce_system(transport)
    h_brevity = _RULESET_LAYER_HEADER.format(i=1, n=2, name="brevity-editor")
    h_md = _RULESET_LAYER_HEADER.format(i=2, n=2, name="markdown-writer")
    assert system.index(h_brevity) < system.index(h_md)
    # Layer order tracks the list, not the alphabet.
    assert system.index(BREVITY_RULESET.strip()) < system.index(MARKDOWN_WRITER_RULESET.strip())


# --------------------------------------------------------------------------- #
# CODER=REVIEWER: the inline fix re-uses the SAME composed 2-spec system.
# --------------------------------------------------------------------------- #
def test_inline_review_reuses_the_same_composed_two_spec_system(tmp_path):
    reg, loader = _two_spec_registry(tmp_path)
    # produce -> rejected (no heading); inline review -> accepted (has heading).
    transport = FakeTransport(
        [
            "Summary only, no level-1 heading. TELLTALE-BREVITY-LAYER ignored.",
            "# Fixed report\n\n**Summary.** tight.",
        ],
        strict=True,
    )

    def verifier(node, result):
        out = (result.output or "").lstrip()
        return True if out.startswith("# ") else (False, "must open with a '# ' heading")

    node = PlanNode(
        id="n1",
        task="research and report.",
        specs=("markdown-writer", "brevity-editor"),
    )
    runtime = AgentRuntime(
        transport=transport, loader=loader, verifier=verifier, max_inline_fixes=1
    )
    result = asyncio.run(runtime.run(PlanDAG(nodes=[node])))

    assert result.states["n1"]["status"] == NodeStatus.DONE.value
    assert result.states["n1"]["inline_fixed"] is True
    assert transport.call_count == 2

    produce_sys = _produce_system(transport, 0)
    review_sys = _produce_system(transport, 1)
    # Produce carries BOTH rulesets + framing, NOT the reviewer framing.
    assert MARKDOWN_WRITER_RULESET.strip() in produce_sys
    assert BREVITY_RULESET.strip() in produce_sys
    assert _REVIEWER_FRAMING not in produce_sys
    # Inline review re-uses the SAME composed 2-spec stack + the reviewer framing.
    assert _SHAPING_FRAMING in review_sys
    assert MARKDOWN_WRITER_RULESET.strip() in review_sys
    assert BREVITY_RULESET.strip() in review_sys
    assert _REVIEWER_FRAMING in review_sys


# --------------------------------------------------------------------------- #
# BACK-COMPAT: single-spec composes to EXACTLY {framing}\n\n{body} (no header);
# the scalar `spec` form and a 1-element `specs` list are equivalent.
# --------------------------------------------------------------------------- #
def test_single_spec_back_compat_no_layer_header(tmp_path):
    reg, loader = _two_spec_registry(tmp_path)

    # (a) legacy scalar spec
    t_scalar = FakeTransport(["# r"])
    asyncio.run(
        AgentRuntime(transport=t_scalar, loader=loader).run(
            PlanDAG(nodes=[PlanNode(id="n1", task="t.", spec="markdown-writer")])
        )
    )
    sys_scalar = _produce_system(t_scalar)

    # (b) a one-element specs list — must be IDENTICAL
    t_list = FakeTransport(["# r"])
    asyncio.run(
        AgentRuntime(transport=t_list, loader=loader).run(
            PlanDAG(nodes=[PlanNode(id="n1", task="t.", specs=("markdown-writer",))])
        )
    )
    sys_list = _produce_system(t_list)

    expected = f"{_SHAPING_FRAMING}\n\n{MARKDOWN_WRITER_RULESET.strip()}"
    assert sys_scalar == expected           # exactly framing + the one body
    assert sys_list == expected             # the list form is equivalent
    assert _RULESET_LAYER_HEADER.format(i=1, n=1, name="markdown-writer") not in sys_scalar
    assert "=====" not in sys_scalar         # no layer separators for a single spec


def test_bare_node_has_no_system_shaping_layer(tmp_path):
    reg, loader = _two_spec_registry(tmp_path)
    transport = FakeTransport(["plain answer"])
    asyncio.run(
        AgentRuntime(transport=transport, loader=loader).run(
            PlanDAG(nodes=[PlanNode(id="n1", task="Summarise.")])  # no spec / specs
        )
    )
    roles = [m["role"] for m in transport.calls[0]["messages"]]
    assert "system" not in roles
    assert _SHAPING_FRAMING not in str(transport.calls[0]["messages"])


# --------------------------------------------------------------------------- #
# UNIT: SubAgent composition seam directly (scopes handed in, no loader).
# --------------------------------------------------------------------------- #
def test_subagent_compose_system_unit(tmp_path):
    from agent_runtime.scope import ScopedSpec

    scopes = [
        ScopedSpec.of("markdown-writer", MARKDOWN_WRITER_RULESET.strip()),
        ScopedSpec.of("brevity-editor", BREVITY_RULESET.strip()),
    ]
    node = PlanNode(id="n1", task="t.", specs=("markdown-writer", "brevity-editor"))
    agent = SubAgent(node, transport=FakeTransport(["x"]), scopes=scopes)
    system = agent._compose_system()
    assert system.startswith(_SHAPING_FRAMING)
    assert _RULESET_LAYER_HEADER.format(i=1, n=2, name="markdown-writer") in system
    assert _RULESET_LAYER_HEADER.format(i=2, n=2, name="brevity-editor") in system
    assert agent.spec_names == ("markdown-writer", "brevity-editor")
    # The by-construction d10 proof still holds with a multi-scope set.
    from agent_runtime.scope import ScopedSpec as _SS
    _SS.assert_no_loader(agent)
