/**
 * The HTTP client for the ReactiveAgents backend.
 *
 * Same-origin in production (the backend serves the SPA at GET / and mounts
 * /static), so all paths are root-relative. Every outbound call carries a
 * timeout via AbortSignal (universal [required]: timeouts on every outbound
 * call). Responses are validated into the typed models at this boundary, so no
 * `any` leaks past it.
 */
import type {
  ApproveSpecResponse,
  AuthorShapeRequest,
  ChatRecord,
  DenySpecResponse,
  HealthResponse,
  LambdaSnapshot,
  MessageResponse,
  OpenSpecChatResponse,
  RefineShapeRequest,
  RegisteredSpec,
  RegisteredSpecRow,
  ResumeRequest,
  RunRecord,
  SetShapeMaxIterRequest,
  ShapeListResponse,
  ShapeView,
  SpecChatMessageResponse,
  SpecChatView,
  StartRunResponse,
  UpdateSpecRequest,
} from "./types";

/** Default per-request timeout. The SSE stream is the live channel; these REST
 * calls are all short control-plane operations, so a generous-but-bounded ceiling
 * is correct. */
const DEFAULT_TIMEOUT_MS = 15_000;

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "DELETE";
  body?: unknown;
  timeoutMs?: number;
  signal?: AbortSignal | undefined;
}

async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, timeoutMs = DEFAULT_TIMEOUT_MS, signal } = opts;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new DOMException("timeout", "TimeoutError")), timeoutMs);
  // Honor an externally-supplied signal (e.g. React Query cancellation) as well
  // as our own timeout.
  if (signal) {
    if (signal.aborted) controller.abort(signal.reason);
    else signal.addEventListener("abort", () => controller.abort(signal.reason), { once: true });
  }
  try {
    const init: RequestInit = { method, signal: controller.signal };
    if (body !== undefined) {
      init.headers = { "Content-Type": "application/json" };
      init.body = JSON.stringify(body);
    }
    const res = await fetch(path, init);
    if (!res.ok) {
      const detail = await safeDetail(res);
      throw new ApiError(res.status, detail ?? `${method} ${path} failed (${res.status})`);
    }
    if (res.status === 204) return undefined as T;
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

async function safeDetail(res: Response): Promise<string | null> {
  try {
    const data: unknown = await res.json();
    if (data && typeof data === "object" && "detail" in data) {
      const d = (data as { detail: unknown }).detail;
      return typeof d === "string" ? d : JSON.stringify(d);
    }
  } catch {
    // non-JSON error body; fall through
  }
  return null;
}

// --------------------------------------------------------------------------- //
// chats
// --------------------------------------------------------------------------- //
export function listChats(signal?: AbortSignal): Promise<{ chats: ChatRecord[] }> {
  return request<{ chats: ChatRecord[] }>("/chats", { signal });
}

export function getChat(chatId: string, signal?: AbortSignal): Promise<ChatRecord> {
  return request<ChatRecord>(`/chats/${encodeURIComponent(chatId)}`, { signal });
}

export function createChat(title?: string): Promise<ChatRecord> {
  return request<ChatRecord>("/chats", {
    method: "POST",
    body: title === undefined ? {} : { title },
  });
}

// --------------------------------------------------------------------------- //
// runs (the decoupled, non-freezing path)
// --------------------------------------------------------------------------- //
export function startRun(chatId: string, message: string, topic?: string): Promise<StartRunResponse> {
  return request<StartRunResponse>(`/chats/${encodeURIComponent(chatId)}/runs`, {
    method: "POST",
    body: topic === undefined ? { message } : { message, topic },
  });
}

export function getRun(runId: string, signal?: AbortSignal): Promise<RunRecord> {
  return request<RunRecord>(`/runs/${encodeURIComponent(runId)}`, { signal });
}

/** The synchronous back-compat submit (kept available; the UI prefers startRun). */
export function postMessage(chatId: string, message: string, topic?: string): Promise<MessageResponse> {
  return request<MessageResponse>(`/chats/${encodeURIComponent(chatId)}/message`, {
    method: "POST",
    body: topic === undefined ? { message } : { message, topic },
    timeoutMs: 120_000,
  });
}

/**
 * Resolve a MISSING-SPECIALIST pause (s10-a8): echo the pause's `resume_token`
 * back with the chosen resolution. `sse_fallback` runs the unmet node(s) spec-less
 * and streams a best-effort answer; `define_and_resume` stamps the now-registered
 * `spec_name` onto them. The backend re-drives the paused plan, so this is a slow
 * planner+runtime round-trip — it carries the same generous ceiling as postMessage.
 * Returns the resumed run's full MessageResponse (a normal terminal summary).
 */
export function resumeRun(chatId: string, req: ResumeRequest): Promise<MessageResponse> {
  return request<MessageResponse>(`/chats/${encodeURIComponent(chatId)}/resume`, {
    method: "POST",
    body: req,
    timeoutMs: 120_000,
  });
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>("/health", { signal });
}

// --------------------------------------------------------------------------- //
// artifacts — URLs (the browser does the actual download/inline render)
// --------------------------------------------------------------------------- //
export function artifactDownloadUrl(artifactId: string): string {
  return `/artifacts/${encodeURIComponent(artifactId)}`;
}

export function artifactInlineUrl(artifactId: string): string {
  return `/artifacts/${encodeURIComponent(artifactId)}?inline=1`;
}

/** The per-chat live SSE stream URL. */
export function chatStreamUrl(chatId: string): string {
  return `/chats/${encodeURIComponent(chatId)}/stream`;
}

// --------------------------------------------------------------------------- //
// (s7/a2) reactive-lambda surface — READ-ONLY (d15: user observes, never authors)
// --------------------------------------------------------------------------- //
export function getLambdaSubscriptions(
  includeClosed = true,
  signal?: AbortSignal,
): Promise<LambdaSnapshot> {
  const q = includeClosed ? "" : "?include_closed=false";
  return request<LambdaSnapshot>(`/lambda/subscriptions${q}`, { signal });
}

/** The live SSE meta-plane channel for the lambda tab (lambda_registered/fired/
 * closed/observation). */
export function lambdaStreamUrl(): string {
  return "/lambda/stream";
}

// --------------------------------------------------------------------------- //
// (s7/a2) spec-definition chat — the DISTINCT interactive spec-authoring surface.
// The redraft (/message) and compile-on-approve (/approve) are blocking phi/file
// round-trips on the live path (the backend offloads them off its event loop),
// so they carry a generous timeout like the synchronous postMessage above.
// --------------------------------------------------------------------------- //
const SPEC_CHAT_TURN_TIMEOUT_MS = 120_000;

export function openSpecChat(name: string, description?: string): Promise<OpenSpecChatResponse> {
  return request<OpenSpecChatResponse>("/spec-chats", {
    method: "POST",
    body: description === undefined ? { name } : { name, description },
  });
}

export function sendSpecChatMessage(
  sessionId: string,
  message: string,
): Promise<SpecChatMessageResponse> {
  return request<SpecChatMessageResponse>(
    `/spec-chats/${encodeURIComponent(sessionId)}/message`,
    { method: "POST", body: { message }, timeoutMs: SPEC_CHAT_TURN_TIMEOUT_MS },
  );
}

export function getSpecChat(sessionId: string, signal?: AbortSignal): Promise<SpecChatView> {
  return request<SpecChatView>(`/spec-chats/${encodeURIComponent(sessionId)}`, { signal });
}

export function approveSpecChat(sessionId: string): Promise<ApproveSpecResponse> {
  return request<ApproveSpecResponse>(
    `/spec-chats/${encodeURIComponent(sessionId)}/approve`,
    { method: "POST", timeoutMs: SPEC_CHAT_TURN_TIMEOUT_MS },
  );
}

export function denySpecChat(sessionId: string): Promise<DenySpecResponse> {
  return request<DenySpecResponse>(`/spec-chats/${encodeURIComponent(sessionId)}/deny`, {
    method: "POST",
  });
}

// --------------------------------------------------------------------------- //
// (s4/RC7) RE-EDITABLE specialization surface — list / fetch-by-id / direct
// PUT update / re-open-into-chat. The list + fetch are short reads; the PUT is a
// registry FILE write the backend offloads off its loop, so it carries the same
// generous ceiling as the redraft turns above.
// --------------------------------------------------------------------------- //
export function listRegisteredSpecs(signal?: AbortSignal): Promise<RegisteredSpecRow[]> {
  return request<RegisteredSpecRow[]>("/spec-chats/registered", { signal });
}

export function getRegisteredSpec(name: string, signal?: AbortSignal): Promise<RegisteredSpec> {
  return request<RegisteredSpec>(
    `/spec-chats/registered/${encodeURIComponent(name)}`,
    { signal },
  );
}

export function updateRegisteredSpec(
  name: string,
  edit: UpdateSpecRequest,
): Promise<RegisteredSpec> {
  return request<RegisteredSpec>(
    `/spec-chats/registered/${encodeURIComponent(name)}`,
    { method: "PUT", body: edit, timeoutMs: SPEC_CHAT_TURN_TIMEOUT_MS },
  );
}

/**
 * DELETE a registered specialization (s4/a4, d13 UI). Unlinks its doc through the
 * stateless SpecRegistry — additive, leaving create/list/select/update/author
 * untouched. The backend returns `{ok, deleted}` (200) or 404 if absent; the
 * caller's mutation invalidates the list so the row clears. Specs have no
 * built-ins, so any registered spec is deletable.
 */
export function deleteRegisteredSpec(name: string): Promise<void> {
  return request<void>(`/spec-chats/registered/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

/** Re-open an existing registered spec into an EDITABLE chat session (201). The
 * returned view begins already-started with the existing body as its draft; the
 * caller then refines via /message and re-registers via /approve. */
export function reopenSpecChat(name: string): Promise<SpecChatView> {
  return request<SpecChatView>("/spec-chats/reopen", {
    method: "POST",
    body: { name },
    timeoutMs: SPEC_CHAT_TURN_TIMEOUT_MS,
  });
}

// --------------------------------------------------------------------------- //
// (s4/a5, d5/d9) SHAPES screen — list the text-file shapes + their structure,
// view one, and SET a per-shape max_iter override. The list/view are short reads;
// the PUT writes one row to the shared SQLite (a4 ShapeConfigStore), so it is a
// short control-plane op on the default timeout.
// --------------------------------------------------------------------------- //
export function listShapes(signal?: AbortSignal): Promise<ShapeView[]> {
  return request<ShapeListResponse>("/shapes", { signal }).then((r) => r.shapes);
}

export function getShape(name: string, signal?: AbortSignal): Promise<ShapeView> {
  return request<ShapeView>(`/shapes/${encodeURIComponent(name)}`, { signal });
}

export function setShapeMaxIter(name: string, maxIter: number): Promise<ShapeView> {
  const body: SetShapeMaxIterRequest = { max_iter: maxIter };
  return request<ShapeView>(`/shapes/${encodeURIComponent(name)}/max_iter`, {
    method: "PUT",
    body,
  });
}

/**
 * DELETE a USER-AUTHORED shape (s4/a4, d13 UI): unlinks its `.toml` and drops its
 * `max_iter` override row server-side. The backend REFUSES a shipped built-in with
 * 409 (only the user's own shapes are deletable) and 404s an unknown name; on
 * success it returns `{ok, deleted}` (200). The caller's mutation invalidates the
 * catalog so the row clears. The UI hides this control on built-ins, but a 409
 * surfaces gracefully as the real guard.
 */
export function deleteShape(name: string): Promise<void> {
  return request<void>(`/shapes/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

// --------------------------------------------------------------------------- //
// (s9/b1, d14(2)/d9) DESCRIBE-A-SHAPE — the user describes a shape and the live
// Gemma model authors the declarative file. This is a blocking native-structured
// model round-trip the backend offloads off its event loop, so it carries the same
// generous ceiling as the spec-chat authoring turns above. Returns the authored
// shape's full view (so the catalog lists it at once).
// --------------------------------------------------------------------------- //
const SHAPE_AUTHOR_TIMEOUT_MS = 120_000;

export function authorShape(description: string, nameHint?: string): Promise<ShapeView> {
  const body: AuthorShapeRequest =
    nameHint && nameHint.trim() ? { description, name_hint: nameHint.trim() } : { description };
  return request<ShapeView>("/shapes/author", {
    method: "POST",
    body,
    timeoutMs: SHAPE_AUTHOR_TIMEOUT_MS,
  });
}

// (s8/b6, d18a) REFINE-A-SHAPE — the user edits an EXISTING shape in plain language
// and the live Gemma model authors the next version building on the current one.
// Same blocking model round-trip + generous ceiling as authoring; the file is
// overwritten in place. Returns the refined shape's full view.
export function refineShape(name: string, instruction: string): Promise<ShapeView> {
  const body: RefineShapeRequest = { instruction };
  return request<ShapeView>(`/shapes/${encodeURIComponent(name)}/refine`, {
    method: "POST",
    body,
    timeoutMs: SHAPE_AUTHOR_TIMEOUT_MS,
  });
}
