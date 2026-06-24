"""agent_runtime — the autonomous planner + in-process agent runtime + self-heal.

Step 6 of the build (outcome O6). This is the abstract-factory dynamic-plan core
that ties the prior subsystems together:

- the **planner** (:class:`Planner`) reasons over ONLY the abstract factory + the
  specialization LOOKUP (d10) and has phi EMIT a custom DAG (model-derived, no
  hard-coded task prompt — d6);
- the **in-process runtime** (:class:`AgentRuntime`) launches each DAG node as a
  tracked ``asyncio`` task respecting ``depends_on`` (d2 — no shell forking),
  each launched :class:`SubAgent` scoped to ONLY its task + the ONE compiled spec
  it loads (d10);
- the **self-heal** layer (:class:`SelfHeal`) detects a failed logical step,
  corrects it, and re-launches it in-process — covering malformed-phi-JSON
  repair and a bounded re-plan on tool error.

The phi transport is PLUGGABLE (the llm_framework ``Transport`` protocol): the
deterministic stubs in :mod:`agent_runtime.stub` drive the whole harness offline
with zero GPU (d7/d8); the live ``OllamaTransport`` swaps in when the shared GPU
frees, with no other code change.

Workspace member (d11): resolves into the single shared root .venv so it imports
llm_framework / reactive_tools / specialization in the SAME in-process
interpreter (d2 — one process).
"""
from __future__ import annotations

from .factory import (
    FACTORY_DESCRIPTION,
    NODE_SCHEMA,
    VALID_ROLES,
    AbstractPlanFactory,
    PlanDAG,
    PlanError,
    PlanNode,
)
from .planner import (
    HEAL_ACTIONS,
    AmbiguityDecision,
    HealDecision,
    Planner,
    PlanResult,
)
from .incremental import (
    DEFAULT_MAX_NODES,
    DEFAULT_NODE_NUM_PREDICT,
    IncrementalPlanner,
)
from .clarification import (
    CLARIFICATION_KIND,
    EVENT_NEEDS_CLARIFICATION,
    clarification_payload,
)
from .heal_router import (
    EVENT_HEAL_ROUTED,
    EVENT_NODE_FAILURE_DETECTED,
    HEAL_RULE_KINDS,
    HealRoute,
    HealRouter,
    register_heal_rule,
)
from .reactor import (
    EVENT_NODE_CLARIFICATION,
    REACTOR_KINDS,
    PlannerReactor,
)
from .review_injection import (
    FINAL_REVIEW_ID,
    REVIEW_SUFFIX,
    inject_reviews,
)
from .runtime import (
    EVENT_NODE_CANCELLED,
    EVENT_NODE_DONE,
    EVENT_NODE_FAILED,
    EVENT_NODE_HEALED,
    EVENT_NODE_INLINE_FIXED,
    EVENT_NODE_LAUNCHED,
    EVENT_NODE_REPLANNED,
    EVENT_NODE_REVIEW,
    EVENT_NODE_SKIPPED,
    EVENT_NODE_VERIFIABLE,
    EVENT_NODE_VERIFY_FAILED,
    EVENT_NODE_COLLISION,
    EVENT_NODE_COLLISION_RESOLVED,
    AgentRuntime,
    NodeVerifier,
    Replanner,
    ResultValidator,
    RuntimeResult,
    SubAgent,
    SubAgentResult,
    ToolArgEmitter,
)
from .collision import (
    Collision,
    CollisionGate,
    CollisionGateError,
    CollisionResolution,
    CollisionResolutionError,
    CollisionResolver,
    CollisionUnresolved,
    ConflictAxis,
    Directive,
    apply_resolution,
    detect_collision,
    parse_directives,
    strip_directives,
)
from .toolargs import (
    TOOL_ARG_SCHEMAS,
    SchemaToolArgEmitter,
    ToolArgEmission,
    default_fallback,
)
from .scope import PlannerScope, ScopedSpec, ScopeViolation
from .selfheal import (
    HealAttempt,
    HealableError,
    HealLog,
    InvalidStepError,
    MalformedOutputError,
    SelfHeal,
    ToolFailureError,
)
from .status import IllegalTransition, NodeState, NodeStatus
from .roles import (
    ROLE_SYNTHESIZER,
    ROLE_WORKER,
    ROLE_FRAMINGS,
    ROLE_SCHEMAS,
    ROLE_VERDICTS,
    JUDGMENT_ROLES,
    POSITION_FRAMINGS,
    position_framing,
    is_judgment_role,
    legal_verdict,
    role_framing,
    role_num_predict_floor,
    role_schema,
)
from .shapes import (
    DEEP_RESEARCH,
    SHAPES_DIR,
    VALID_EXECUTION,
    VALID_POSITIONS,
    ShapeError,
    ShapeSpec,
    load_shape,
    load_shapes,
    shape_names,
    unroll_shape,
)
from .scheduler import (
    Dispatch,
    ExecutionMode,
    execution_mode_for,
    first_ready_action,
    is_complete,
    next_dispatch,
    ready_wave,
)
from .shape_selector import (
    ESCALATE,
    ShapeSelection,
    ShapeSelector,
    build_selection_schema,
)
from .shape_author import (
    ShapeAuthor,
    build_shape_schema,
    shape_to_toml,
    write_shape,
)
from .verify import default_node_verifier
from .missing_spec import (
    CHOICE_DEFINE_AND_RESUME,
    CHOICE_SSE_FALLBACK,
    EVENT_MISSING_SPECIALIST,
    MISSING_SPEC_CHOICES,
    MissingSpecialist,
    apply_resolution as apply_missing_spec_resolution,
    detect_missing_specialists,
    missing_from_requested,
    missing_specialist_payload,
)
from .research_tree import (
    Branch,
    DagGrower,
    DecisionResult,
    LeafResult,
    N4_TREE_DEPTH_CEILING,
    ResearchState,
    Tree,
    TreeConfig,
    TREE_TOOLS,
    parse_tree_call,
    run_decision_node,
)
from .discovery_tools import (
    GET_SHAPES_TOOL,
    GET_SPECS_TOOL,
    GetShapesArgs,
    GetSpecsArgs,
    make_get_shapes,
    make_get_specs,
    register_discovery_tools,
)
from . import stub

__all__ = [
    # factory + DAG
    "AbstractPlanFactory",
    "PlanDAG",
    "PlanNode",
    "PlanError",
    "FACTORY_DESCRIPTION",
    "NODE_SCHEMA",
    "VALID_ROLES",
    # node roles (d48: worker|synthesizer only) + deep-research positions
    "ROLE_WORKER",
    "ROLE_SYNTHESIZER",
    "ROLE_FRAMINGS",
    "ROLE_SCHEMAS",
    "ROLE_VERDICTS",
    "JUDGMENT_ROLES",
    "POSITION_FRAMINGS",
    "position_framing",
    "is_judgment_role",
    "legal_verdict",
    "role_framing",
    "role_num_predict_floor",
    "role_schema",
    # declarative plan shapes (text-file defined) + the generic cyclic unroll
    "ShapeSpec",
    "ShapeError",
    "SHAPES_DIR",
    "DEEP_RESEARCH",
    "VALID_EXECUTION",
    "VALID_POSITIONS",
    "unroll_shape",
    "load_shape",
    "load_shapes",
    "shape_names",
    # deterministic shape dispatch scheduler (ported from eda-base3 plan FSM)
    "ExecutionMode",
    "execution_mode_for",
    "first_ready_action",
    "ready_wave",
    "Dispatch",
    "next_dispatch",
    "is_complete",
    # Gemma shape-SELECTION (native structured enum call, d1)
    "ShapeSelector",
    "ShapeSelection",
    "build_selection_schema",
    "ESCALATE",
    # Gemma shape-AUTHORING from an NL description (native structured call, d14(2))
    "ShapeAuthor",
    "build_shape_schema",
    "shape_to_toml",
    "write_shape",
    # planner
    "Planner",
    "PlanResult",
    "HealDecision",
    "HEAL_ACTIONS",
    # incremental seed-then-fill authoring (the eda-base3 port, d3)
    "IncrementalPlanner",
    "DEFAULT_MAX_NODES",
    "DEFAULT_NODE_NUM_PREDICT",
    # reactive self-heal routing (b4, §2e, d1)
    "HealRouter",
    "HealRoute",
    "register_heal_rule",
    "HEAL_RULE_KINDS",
    "EVENT_NODE_FAILURE_DETECTED",
    "EVENT_HEAL_ROUTED",
    # event-driven planner reaction (P2.2, d129.2)
    "PlannerReactor",
    "EVENT_NODE_CLARIFICATION",
    "REACTOR_KINDS",
    # framework-injected review (P2.2, d129.3)
    "inject_reviews",
    "REVIEW_SUFFIX",
    "FINAL_REVIEW_ID",
    # runtime
    "AgentRuntime",
    "RuntimeResult",
    "SubAgent",
    "SubAgentResult",
    "Replanner",
    "ResultValidator",
    "NodeVerifier",
    "default_node_verifier",
    "ToolArgEmitter",
    # missing-specialist detection + notify/CHOICE surface (s4 M1, RC8)
    "EVENT_MISSING_SPECIALIST",
    "MISSING_SPEC_CHOICES",
    "AmbiguityDecision",
    "EVENT_NEEDS_CLARIFICATION",
    "CLARIFICATION_KIND",
    "clarification_payload",
    "CHOICE_SSE_FALLBACK",
    "CHOICE_DEFINE_AND_RESUME",
    "MissingSpecialist",
    "detect_missing_specialists",
    "missing_from_requested",
    "missing_specialist_payload",
    "apply_missing_spec_resolution",
    # schema-constrained tool-arg emission (s8/b1 phi hardening)
    "SchemaToolArgEmitter",
    "ToolArgEmission",
    "TOOL_ARG_SCHEMAS",
    "default_fallback",
    "EVENT_NODE_LAUNCHED",
    "EVENT_NODE_DONE",
    "EVENT_NODE_FAILED",
    "EVENT_NODE_HEALED",
    "EVENT_NODE_CANCELLED",
    "EVENT_NODE_REPLANNED",
    "EVENT_NODE_SKIPPED",
    "EVENT_NODE_VERIFIABLE",
    "EVENT_NODE_REVIEW",
    "EVENT_NODE_INLINE_FIXED",
    "EVENT_NODE_VERIFY_FAILED",
    "EVENT_NODE_COLLISION",
    "EVENT_NODE_COLLISION_RESOLVED",
    # DAG spec-collision detection + HITL escalation (d11)
    "Directive",
    "parse_directives",
    "strip_directives",
    "ConflictAxis",
    "Collision",
    "detect_collision",
    "CollisionResolution",
    "apply_resolution",
    "CollisionResolver",
    "CollisionGate",
    "CollisionGateError",
    "CollisionResolutionError",
    "CollisionUnresolved",
    # context-scoping by construction (d10)
    "ScopedSpec",
    "PlannerScope",
    "ScopeViolation",
    # per-node status state machine
    "NodeStatus",
    "NodeState",
    "IllegalTransition",
    # self-heal
    "SelfHeal",
    "HealLog",
    "HealAttempt",
    "HealableError",
    "MalformedOutputError",
    "ToolFailureError",
    "InvalidStepError",
    # offline stubs (pluggable transport)
    "stub",
    # s9/N4 — TREE-shaped research with pruning + persisted-state decision node
    "Branch",
    "DagGrower",
    "DecisionResult",
    "LeafResult",
    "N4_TREE_DEPTH_CEILING",
    "ResearchState",
    "Tree",
    "TreeConfig",
    "TREE_TOOLS",
    "parse_tree_call",
    "run_decision_node",
    # discovery tools (get_shapes / get_specs) — queryable shape/spec catalog
    "GET_SHAPES_TOOL",
    "GET_SPECS_TOOL",
    "GetShapesArgs",
    "GetSpecsArgs",
    "make_get_shapes",
    "make_get_specs",
    "register_discovery_tools",
]
