/**
 * Type-safe model of the ReactiveAgents backend surface (s7/b3).
 *
 * Every enum/status from the backend is modeled as a discriminated union or a
 * string-literal union and consumed via an EXHAUSTIVE switch with a `never`
 * default — so a new backend value becomes a COMPILE error here, never a silent
 * fall-through in the UI (spec [required]). Raw JSON is validated into these
 * typed models at the edge (see ./client.ts), so nothing past this boundary
 * touches an untyped `any`.
 *
 * Backend references:
 *  - chat_app/chat_app/routes.py     (chats / message / runs / stream / artifacts)
 *  - chat_app/chat_app/runs.py       (RunRecord)
 *  - chat_app/chat_app/persistence.py(ChatRecord / TurnRecord / ArtifactRef)
 *  - agent_runtime/.../status.py     (NodeStatus)
 *  - agent_runtime/.../runtime.py    (EVENT_NODE_* lifecycle kinds)
 */

// --------------------------------------------------------------------------- //
// node lifecycle — the eda-base3 lifecycle (d9): pending -> in-progress(running)
// -> verifiable -> done, plus the terminal/edge states.
// --------------------------------------------------------------------------- //
export type NodeStatus =
  | "pending"
  | "running"
  | "verifiable"
  | "done"
  | "failed"
  | "skipped"
  | "cancelled";

/**
 * SSE event kinds the per-chat stream relays (routes.py RUNTIME_EVENT_KINDS +
 * the run-engine / collision lifecycle kinds from runtime.py). `connected` is the
 * stream handshake; `tool_call`/`tool_result` ride the same plane (consumed by
 * the a2 CoT overlay later, surfaced here for completeness).
 */
export type StreamEventKind =
  | "connected"
  | "agent_node_launched"
  | "agent_node_done"
  | "agent_node_failed"
  | "agent_node_healed"
  | "agent_node_cancelled"
  | "agent_node_replanned"
  | "agent_node_skipped"
  | "agent_node_verifiable"
  | "agent_node_review"
  | "agent_node_inline_fixed"
  | "agent_node_verify_failed"
  | "agent_node_collision"
  | "agent_node_collision_resolved"
  | "tool_call"
  | "tool_result";

/** Every node-scoped lifecycle event carries (at least) a node id. */
export interface NodeEventPayload {
  node_id: string;
  spec?: string | null;
  specs?: string[];
  error?: string | null;
  reason?: string | null;
  attempt?: number;
  via?: string;
  blocked_by?: string;
}

/**
 * The inner JSON envelope every non-`connected` SSE event carries
 * (routes.py event_source(): {kind, seq, source, payload}).
 */
export interface StreamEnvelope {
  kind: StreamEventKind;
  seq: number;
  source: string;
  payload: unknown;
}

// --------------------------------------------------------------------------- //
// REST resources
// --------------------------------------------------------------------------- //
export interface TurnRecord {
  chat_id: string;
  turn_index: number;
  user_request: string;
  events: Array<Record<string, unknown>>;
  final_response: string;
  created_at: string;
}

export interface ArtifactRef {
  artifact_id: string;
  chat_id: string;
  filename: string;
  mime: string;
  path: string;
  size: number;
  created_at: string;
}

export interface ChatRecord {
  chat_id: string;
  title: string;
  created_at: string;
  turns: TurnRecord[];
  artifacts: ArtifactRef[];
}

export interface NodeStateOut {
  node_id: string;
  status: NodeStatus;
  attempts: number;
  error: string | null;
}

/** An artifact ref as surfaced in a run's final summary (routes.py ArtifactOut). */
export interface ArtifactOut {
  artifact_id: string;
  filename: string;
  mime: string;
  size: number;
  node_id: string | null;
}

// --------------------------------------------------------------------------- //
// MISSING-SPECIALIST PAUSE (s4 M1, RC8 / s10-a8) — a run can PAUSE instead of
// completing when the plan needs a specialist no registered spec provides. The
// terminal MessageResponse then carries `missing_specialist=true` and a `pending`
// CHOICE payload the UI surfaces; the user picks a resolution and the client
// echoes the `resume_token` back to POST /chats/{id}/resume. (missing_spec.py
// missing_specialist_payload + MissingSpecialist.as_dict; routes.py MessageResponse.)
// --------------------------------------------------------------------------- //

/** The two resolutions offered alongside the missing-specialist notify
 * (missing_spec.py MISSING_SPEC_CHOICES). `sse_fallback` runs the unmet node(s)
 * spec-less and streams a best-effort answer; `define_and_resume` stamps a
 * now-defined specialization onto them and resumes. Consumed via an EXHAUSTIVE
 * switch with a `never` default so a new choice is a COMPILE error (spec [required]). */
export type MissingSpecChoice = "sse_fallback" | "define_and_resume";

/** One node that needs a specialist no registered spec provides
 * (missing_spec.py MissingSpecialist.as_dict). `needs` is the planner's free-text
 * descriptor of (or the user-requested name for) the required specialist. */
export interface MissingSpecialistNode {
  node_id: string;
  task: string;
  needs: string;
  role: string | null;
}

/** The `pending` CHOICE payload on a missing-specialist pause
 * (missing_spec.py missing_specialist_payload). The opaque `resume_token` is
 * echoed back to POST /chats/{id}/resume; `choices` is the offered set; `missing`
 * lists the unmet node(s). */
export interface MissingSpecialistPending {
  resume_token: string;
  choices: MissingSpecChoice[];
  missing: MissingSpecialistNode[];
}

// --------------------------------------------------------------------------- //
// AMBIGUITY-CLARIFICATION PAUSE (scenario-2) — a run can PAUSE instead of acting
// when the planner judges the request too underspecified and asks ONE clarifying
// question. The terminal MessageResponse then carries `needs_clarification=true`
// and a `pending` CLARIFICATION payload the UI surfaces; the user types an
// `answer` and the client echoes it (with the `resume_token`) to POST
// /chats/{id}/resume, which re-drives the plan on the clarified intent — instead
// of the UI re-sending the answer as a FRESH /message (which the gate re-asks).
// (clarification.py clarification_payload; routes.py MessageResponse / resume.)
// --------------------------------------------------------------------------- //

/** The `pending` payload on an ambiguity-clarification pause
 * (clarification.py clarification_payload). The `kind` discriminator tells it
 * apart from a MissingSpecialistPending; the opaque `resume_token` is echoed back
 * (with the user's `answer`) to POST /chats/{id}/resume; `question` is the
 * planner's single clarifying question; `original_query` is the request the
 * answer refines. */
export interface ClarificationPending {
  kind: "clarification";
  resume_token: string;
  question: string;
  original_query: string;
}

/** The two pause payload shapes a run can carry. Discriminated by the
 * `kind:"clarification"` field present ONLY on ClarificationPending — narrow on
 * `"kind" in pending` (or on the response's `needs_clarification` /
 * `missing_specialist` flags) before consuming. */
export type PendingPause = MissingSpecialistPending | ClarificationPending;

/** The terminal summary of one driven message run (routes.py MessageResponse).
 * On a missing-specialist PAUSE, `ok` is false, the run produced nothing
 * (`node_states`/`outputs`/`artifacts` empty), `missing_specialist` is true and
 * `pending` carries the CHOICE. On an ambiguity-clarification PAUSE, `ok` is
 * false, `needs_clarification` is true and `pending` carries the clarification
 * payload (the question + resume_token). For a normal run both flags are false
 * and `pending` is null (back-compat: pre-existing clients ignore the new
 * fields). */
export interface MessageResponse {
  chat_id: string;
  turn_index: number;
  ok: boolean;
  launch_order: string[];
  node_states: NodeStateOut[];
  outputs: Record<string, string>;
  artifacts: ArtifactOut[];
  /** True when the run paused because a needed specialist is unavailable. */
  missing_specialist?: boolean;
  /** True when the run paused on a planner clarifying question (scenario-2). */
  needs_clarification?: boolean;
  /** The pause payload when paused (CHOICE or clarification), else null/absent. */
  pending?: PendingPause | null;
}

/** POST /chats/{id}/resume body (routes.py ResumeRequest) — resolve a paused run.
 * ONE request type serves both pauses (validated per pause kind server-side):
 *  - MISSING-SPECIALIST: `choice` is required; `define_and_resume` also needs
 *    `spec_name` (the registered spec to apply to every unmet node).
 *  - CLARIFICATION (scenario-2): `answer` carries the user's reply to the
 *    planner's question; `choice` is omitted. */
export interface ResumeRequest {
  resume_token: string;
  choice?: MissingSpecChoice;
  spec_name?: string;
  /** The user's answer to the planner's clarifying question (clarification pause). */
  answer?: string;
}

/** The 202 body of the decoupled run-start (routes.py start_run). */
export interface StartRunResponse {
  run_id: string;
  chat_id: string;
  status: RunStatus;
  stream: string;
  status_url: string;
}

export type RunStatus = "running" | "done" | "failed" | "cancelled";

/** A background run's polled state (runs.py RunRecord.to_dict). */
export interface RunRecord {
  run_id: string;
  chat_id: string;
  status: RunStatus;
  /** Present once `done`: the run body's MessageResponse. */
  result: MessageResponse | null;
  error: string | null;
  started: number;
  ended: number | null;
  duration_s: number | null;
}

export interface HealthResponse {
  status: string;
  transport: string;
  components: Record<string, unknown>;
}

// =========================================================================== //
// s7/a2 — engagement surfaces
//
// New backend shapes for the three a2 features. As with the a1 models above,
// every enum is a string-literal union consumed via an exhaustive `switch`
// (spec [required]), and the raw SSE/JSON is validated into these typed models
// at the client edge — nothing past this boundary is `any`.
// =========================================================================== //

// --------------------------------------------------------------------------- //
// (a) CHAIN-OF-THOUGHT — the tool-call/result plane the brain-icon overlay reads.
// These ride the SAME per-chat SSE stream as the node lifecycle (routes.py
// RUNTIME_EVENT_KINDS includes "tool_call"/"tool_result"); the payload shapes
// are reactive_tools/tool_hook.py ToolHook.invoke / invoke_sync.
// --------------------------------------------------------------------------- //

/** `tool_call` payload — emitted the instant a tool is invoked.
 * NOTE: tool_hook.py emits call_id as an INTEGER sequence (_next_call_id), so
 * the wire type is string|number; the overlay normalizes it via String(). */
export interface ToolCallPayload {
  call_id: string | number;
  name: string;
  args: Record<string, unknown>;
}

/** `tool_result` payload — emitted when the tool returns (ok) or raises (error). */
export interface ToolResultPayload {
  call_id: string | number;
  name: string;
  ok: boolean;
  /** The tool's return value (shape is tool-specific — kept opaque on purpose). */
  value: unknown;
  error: string | null;
}

// --------------------------------------------------------------------------- //
// (b) LAMBDA TAB — the read-only live view of agent-created reactive lambdas.
// Snapshot: GET /lambda/subscriptions. Live channel: SSE GET /lambda/stream
// (the META plane). Shapes are reactive_tools/subscriptions.py LambdaRecord.as_view
// and the META_LAMBDA_* publish payloads.
// --------------------------------------------------------------------------- //

/** A lambda's lifecycle status in the registry. */
export type LambdaStatus = "active" | "closed";

/** The observe-only projection of ONE agent-created reactive lambda
 * (LambdaRecord.as_view). The user reads this; never authors it (d15). */
export interface LambdaSubscriptionView {
  sub_id: string;
  label: string;
  /** One-line "what it watches" string, e.g. "agent_node_done [count]". */
  observes: string;
  kinds: string[];
  reducer: string;
  reaction: string;
  owner: Record<string, unknown>;
  status: LambdaStatus;
  created_seq: number;
  seen_count: number;
  fire_count: number;
  last_event_kind: string | null;
  last_fired_seq: number | null;
  closed_seq: number | null;
  composed_from: string[];
}

/** GET /lambda/subscriptions body. */
export interface LambdaSnapshot {
  active: number;
  total: number;
  subscriptions: LambdaSubscriptionView[];
}

/** Named events on the lambda META plane (SSE /lambda/stream). `connected` is the
 * handshake; the rest are the live-subscriptions channel. */
export type LambdaMetaKind =
  | "connected"
  | "lambda_registered"
  | "lambda_fired"
  | "lambda_closed"
  | "lambda_observation";

/** `lambda_fired` payload — a live counter tick for one lambda. */
export interface LambdaFiredPayload {
  sub_id: string;
  label: string;
  source_seq: number;
  source_kind: string;
  fire_count: number;
  seen_count: number;
}

/** `lambda_closed` payload — a lambda reached a terminal state. */
export interface LambdaClosedPayload {
  sub_id: string;
  label: string;
  reason: string;
  fire_count: number;
  seen_count: number;
}

/** A lambda-meta SSE envelope, same {kind,seq,source,payload} shape as the chat
 * stream's StreamEnvelope but over the LambdaMetaKind set. */
export interface LambdaMetaEnvelope {
  kind: LambdaMetaKind;
  seq: number;
  source: string;
  payload: unknown;
}

// --------------------------------------------------------------------------- //
// (c) SPEC-DEFINITION CHAT — the distinct interactive spec-authoring surface
// (spec_chat.py). The user states intent, reads the drafted ruleset, critiques,
// watches it re-author, and approves to compile + register a planner-loadable
// spec. States: SpecConversation STATE_* (open|approved|denied|cancelled).
// --------------------------------------------------------------------------- //

export type SpecChatState = "open" | "approved" | "denied" | "cancelled";

/** One recorded conversation turn (spec_chat.py TurnOut). */
export interface SpecTurn {
  role: "user" | "agent";
  text: string;
}

/** The working draft after a turn — the ruleset body plus the EXACT compiled
 * markdown doc the user would approve (spec_chat.py DraftOut). */
export interface SpecDraft {
  name: string;
  description: string;
  body: string;
  markdown: string;
  /** How many author/refine rounds produced this body (1-based). */
  turn: number;
}

/** POST /spec-chats (201) — a freshly opened session, before any turn. */
export interface OpenSpecChatResponse {
  session_id: string;
  name: string;
  description: string;
  state: SpecChatState;
  started: boolean;
}

/** POST /spec-chats/{id}/message — one turn's outcome. */
export interface SpecChatMessageResponse {
  session_id: string;
  state: SpecChatState;
  draft: SpecDraft;
  turns: SpecTurn[];
}

/** GET /spec-chats/{id} — the full reopen transcript. */
export interface SpecChatView {
  session_id: string;
  name: string;
  description: string;
  state: SpecChatState;
  started: boolean;
  draft: SpecDraft | null;
  turns: SpecTurn[];
}

/** POST /spec-chats/{id}/approve — compile + register receipt. */
export interface ApproveSpecResponse {
  session_id: string;
  state: SpecChatState;
  name: string;
  source: string;
  registered: boolean;
}

/** POST /spec-chats/{id}/deny — discard receipt. */
export interface DenySpecResponse {
  session_id: string;
  state: SpecChatState;
  discarded: boolean;
}

// --------------------------------------------------------------------------- //
// (s4/RC7) RE-EDITABLE specialization surface — re-open an EXISTING registered
// spec to view + edit, persisted + effective on the next run. Backed by the a2
// store/API (spec_chat.py): list (body-free rows) / fetch-one-by-id (full body
// + provenance) / direct PUT update / re-open-into-chat. The edit persists
// through the runtime's authoritative SpecRegistry, NOT SQLite (d10).
// --------------------------------------------------------------------------- //

/** GET /spec-chats/registered — one body-free identity row of the
 * "pick a spec to re-open" list (spec_chat.py RegisteredSpecRow). */
export interface RegisteredSpecRow {
  name: string;
  description: string;
  source: string;
}

/** GET /spec-chats/registered/{name} — the FULL persisted spec for the re-open
 * VIEW: body + provenance included (spec_chat.py RegisteredSpecOut). */
export interface RegisteredSpec {
  name: string;
  description: string;
  source: string;
  body: string;
  research_trace_ref: string;
  created_at: string;
}

/** PUT /spec-chats/registered/{name} body — overlay edits (at least one of the
 * two must be present; the backend 422s otherwise). Identity + provenance are
 * preserved server-side (spec_chat.py UpdateSpecRequest). */
export interface UpdateSpecRequest {
  description?: string;
  body?: string;
}

// --------------------------------------------------------------------------- //
// (s4/a5, d5/d9; REDESIGNED s17, d247/d248/d249) SHAPES screen — the DEDICATED
// view over the TEXT-FILE-defined plan shapes. A shape is an execution DISCIPLINE
// + DOCTRINE — it declares NO fixed node topology (the planner/grower AUTHORS the
// topology at runtime by reasoning). The screen lists every shape, surfaces its
// discipline, doctrine and declared phase flow, and lets the user set a per-shape
// MAX_ITER override (persisted to the shared SQLite via the a4 backend — the
// deep-research grow loop's depth ceiling). Backend: chat_app/shape_config.py —
// GET /shapes, GET /shapes/{name}, PUT /shapes/{name}/max_iter. A shape's view =
// ShapeSpec.as_dict() PLUS the stored override + the effective ceiling.
// The old round_roles/final_roles bounded-unroll preview (and its transitional
// empty-[] API shim) is REMOVED — no fixed round topology exists anymore.
// --------------------------------------------------------------------------- //

/** A shape's declared execution discipline (agent_runtime/shapes.py). `sequential`
 * = strict single-file; `concurrent` = wave fan-out; `deep-research` = iterative
 * deepening research GROWN at runtime by the planner/grower (no fixed rounds).
 * Consumed via an EXHAUSTIVE switch with a `never` default so a new discipline is
 * a COMPILE error here (spec [required]). */
export type ShapeExecution = "sequential" | "concurrent" | "deep-research";

/** One declared phase of a shape's flow (RP-6b d359/d361): the phase KIND
 * (research/write/…) + the spec_role the engine routes that phase's specs by. */
export interface ShapePhase {
  kind: string;
  spec_role: string;
}

/** One text-file shape merged with its stored override (shape_config.py
 * ShapeConfigService._view = ShapeSpec.as_dict() + max_iter_override +
 * effective_max_iter). The shape carries DISCIPLINE (`execution`), DOCTRINE
 * (`completeness_stop`, `decompose_methodology`), declared `phases` flow and the
 * deep-research growth bounds (`fan_out`/`max_layers`/`max_sources` — safety
 * ceilings, not a topology). `max_iter` is the file DEFAULT depth ceiling,
 * `hard_cap` the absolute safety bound, `max_iter_override` the UI-set value (or
 * null), `effective_max_iter` the override clamped to hard_cap. */
export interface ShapeView {
  name: string;
  description: string;
  max_iter: number;
  hard_cap: number;
  edges: Record<string, unknown>;
  source: string;
  execution: ShapeExecution;
  completeness_stop: string;
  decompose_methodology: string;
  deny_domains: string[];
  expand_on_gaps: boolean;
  fan_out: number;
  max_layers: number;
  max_sources: number;
  phases: ShapePhase[];
  max_iter_override: number | null;
  effective_max_iter: number;
}

/** GET /shapes body. */
export interface ShapeListResponse {
  shapes: ShapeView[];
}

/** PUT /shapes/{name}/max_iter body — the ONE field the Shapes screen edits
 * (shape_config.py SetMaxIterRequest; backend enforces 1..1000, 422 otherwise). */
export interface SetShapeMaxIterRequest {
  max_iter: number;
}

/** POST /shapes/author body (s9/b1, d14(2)) — the user DESCRIBES a shape and the
 * live Gemma model authors the declarative file (shape_authoring.py
 * AuthorShapeRequest). `description` is the only required field; `name_hint` is an
 * optional slug nudge. The response is the authored shape's full ShapeView. */
export interface AuthorShapeRequest {
  description: string;
  name_hint?: string;
}

/** POST /shapes/{name}/refine body (s8/b6, d18a) — the user REFINES an existing
 * shape in plain language and the live Gemma model authors the next version
 * BUILDING ON the current one (shape_authoring.py RefineShapeRequest). The shape
 * name is in the path; `instruction` is the change to apply. The response is the
 * refined shape's full ShapeView (the file is overwritten in place). */
export interface RefineShapeRequest {
  instruction: string;
}

// --------------------------------------------------------------------------- //
// (s17, d18a/d249 parity) SHAPE CHAT — the conversational, draft-based shape
// authoring surface (chat_app/shape_chat.py). Mirrors the spec chat: each message
// drives one live authoring turn over an IN-SESSION draft; nothing persists until
// approve (create-collision → 409); deny discards. `mode` is "create" for a fresh
// draft, "refine" when the session was opened on an existing shape (refine_of).
// --------------------------------------------------------------------------- //

/** One transcript turn of a shape chat. */
export interface ShapeChatTurn {
  role: string;
  text: string;
}

/** The compact draft preview the chat renders (null before the first turn). */
export interface ShapeChatDraft {
  name: string;
  description: string;
  execution: ShapeExecution;
  max_iter: number;
}

/** GET/POST /shape-chat views (shape_chat.py ShapeChatView). */
export interface ShapeChatView {
  session_id: string;
  mode: "create" | "refine";
  state: "open" | "approved" | "denied";
  refine_of: string | null;
  turns: ShapeChatTurn[];
  draft: ShapeChatDraft | null;
}

/** POST /shape-chat/{sid}/approve body response. */
export interface ApproveShapeChatResponse {
  approved: boolean;
  shape: ShapeView;
}
