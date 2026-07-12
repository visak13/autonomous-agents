"""SB-RR (d292/d293) — SELF-POLICING grep gate for the ROLE_RESEARCHER retirement.

The recipe's anti-fabrication lens (d240) requires that, after SB-RR, the engine's DISPATCH
shows ZERO role branching for gather/worker — research is a SELF-SELECTED specialization, every
spawned node is a WORKER (d273), and gather/trivial/research-follow-up all flow through ONE
unified worker loop. The ONLY role conditional that may survive in routing is the terminal
SYNTHESIZER delivery (d215). These tests read the actual engine source and FAIL if a role
branch creeps back into the dispatch — a guard against a regression re-introducing the role."""
from __future__ import annotations

import inspect
import re

from agent_runtime.research_tree import DagGrower
from agent_runtime.runtime import SubAgent


def test_dispatch_only_role_branch_is_terminal_synthesizer():
    """``SubAgent.run`` (the per-node DISPATCH) branches on a role ONLY for the terminal
    synthesizer delivery (d215). NO ``role == ROLE_RESEARCHER`` (retired), NO ``role ==
    ROLE_WORKER`` routing (every node is a worker — you don't branch on worker-ness, d273), and
    NO ``role is None`` producer branch (the trivial producer folds into the unified loop)."""
    src = inspect.getsource(SubAgent.run)
    # EVERY role comparison in the dispatch must reference ROLE_SYNTHESIZER (== or != exclusion).
    role_cmps = re.findall(r"role\s*[!=]=\s*(ROLE_\w+|None)", src)
    assert role_cmps, "expected at least the terminal-synthesizer role check in run()"
    assert set(role_cmps) == {"ROLE_SYNTHESIZER"}, (
        f"a non-synthesizer role branch crept back into the dispatch: {sorted(set(role_cmps))}"
    )
    # explicit token guards (belt-and-braces; the routing CODE these would form is gone):
    assert "role == ROLE_RESEARCHER" not in src and "ROLE_RESEARCHER ==" not in src
    assert "role == ROLE_WORKER" not in src and "ROLE_WORKER ==" not in src
    assert "role is None" not in src and "role == None" not in src


def test_grower_recognizes_gather_by_memory_handle_not_role():
    """The grower's ingest recognizer ``DagGrower._is_research_node`` folds a completed gather
    node SOURCE-AGNOSTICALLY by the research-MEMORY HANDLE it binds on (or a legacy web_search
    tool) — it NO LONGER reads the node ROLE. This is the grower's own bookkeeping, not a runtime
    routing discriminator, and it must not resurrect a role check."""
    src = inspect.getsource(DagGrower._is_research_node)
    assert 'getattr(node, "role"' not in src, "ingest recognizer must not read the node role"
    assert not re.search(r"role\s*[!=]=\s*ROLE_", src), "no role comparison in the recognizer"
    assert "research_memory_handle" in src, "folds by the source-agnostic research-memory handle"
