/**
 * TanStack Query hooks — the server cache is the source of truth for server
 * state (spec [required]); no server data is mirrored into useState+useEffect.
 *
 * Run status is POLLED via `refetchInterval` until the run reaches a terminal
 * state, at which point polling stops itself — the canonical Query way to track
 * an async backend job without a fetch-in-effect.
 */
import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseQueryResult,
} from "@tanstack/react-query";
import {
  approveShapeChat,
  approveSpecChat,
  authorShape,
  createChat,
  deleteRegisteredSpec,
  deleteShape,
  denyShapeChat,
  denySpecChat,
  getChat,
  getLambdaSubscriptions,
  getRegisteredSpec,
  getRun,
  getSpecChat,
  getShape,
  listChats,
  listRegisteredSpecs,
  listShapes,
  openShapeChat,
  openSpecChat,
  refineShape,
  reopenSpecChat,
  resumeRun,
  sendShapeChatMessage,
  sendSpecChatMessage,
  setShapeMaxIter,
  startRun,
  updateRegisteredSpec,
} from "./client";
import type {
  ApproveSpecResponse,
  ChatRecord,
  DenySpecResponse,
  LambdaSnapshot,
  MessageResponse,
  OpenSpecChatResponse,
  RegisteredSpec,
  RegisteredSpecRow,
  ResumeRequest,
  RunRecord,
  ApproveShapeChatResponse,
  ShapeChatView,
  ShapeView,
  SpecChatMessageResponse,
  SpecChatView,
  StartRunResponse,
  UpdateSpecRequest,
} from "./types";

const RUN_POLL_INTERVAL_MS = 800;

export const queryKeys = {
  chats: ["chats"] as const,
  chat: (id: string) => ["chat", id] as const,
  run: (id: string) => ["run", id] as const,
  lambdaSubscriptions: ["lambda", "subscriptions"] as const,
  specChat: (id: string) => ["spec-chat", id] as const,
  registeredSpecs: ["spec-chats", "registered"] as const,
  registeredSpec: (name: string) => ["spec-chats", "registered", name] as const,
  shapes: ["shapes"] as const,
  shape: (name: string) => ["shapes", name] as const,
};

export function useChats(): UseQueryResult<ChatRecord[]> {
  return useQuery({
    queryKey: queryKeys.chats,
    queryFn: async ({ signal }) => (await listChats(signal)).chats,
  });
}

export function useChat(chatId: string | null): UseQueryResult<ChatRecord> {
  return useQuery({
    queryKey: chatId ? queryKeys.chat(chatId) : ["chat", "none"],
    queryFn: ({ signal }) => getChat(chatId as string, signal),
    enabled: chatId !== null,
  });
}

export function useCreateChat() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (title?: string) => createChat(title),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queryKeys.chats });
    },
  });
}

export function useStartRun() {
  return useMutation<StartRunResponse, Error, { chatId: string; message: string; topic?: string }>({
    mutationFn: ({ chatId, message, topic }) => startRun(chatId, message, topic),
  });
}

/**
 * Resolve a missing-specialist PAUSE (s10-a8): POST the chosen resolution to
 * /chats/{id}/resume and, on success, refresh the chat history + index so the
 * resumed answer renders as a new turn (and the artifacts panel reloads). The
 * resumed MessageResponse is returned to the caller (which reconciles the DAG).
 */
export function useResumeRun() {
  const qc = useQueryClient();
  return useMutation<MessageResponse, Error, { chatId: string; req: ResumeRequest }>({
    mutationFn: ({ chatId, req }) => resumeRun(chatId, req),
    onSuccess: (_resp, { chatId }) => {
      void qc.invalidateQueries({ queryKey: queryKeys.chat(chatId) });
      void qc.invalidateQueries({ queryKey: queryKeys.chats });
    },
  });
}

/**
 * Poll one run until it is terminal. Returns the live RunRecord; polling stops
 * automatically once `status` leaves "running".
 */
export function useRunPolling(runId: string | null): UseQueryResult<RunRecord> {
  return useQuery({
    queryKey: runId ? queryKeys.run(runId) : ["run", "none"],
    queryFn: ({ signal }) => getRun(runId as string, signal),
    enabled: runId !== null,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data && data.status !== "running") return false;
      return RUN_POLL_INTERVAL_MS;
    },
  });
}

// --------------------------------------------------------------------------- //
// (s7/a2) lambda subscriptions — the snapshot is the source of truth; the live
// SSE meta-plane (see useLambdaStream) pushes incremental updates into this cache,
// so the tab stays live without a fetch-in-effect.
// --------------------------------------------------------------------------- //
export function useLambdaSubscriptions(enabled = true): UseQueryResult<LambdaSnapshot> {
  return useQuery({
    queryKey: queryKeys.lambdaSubscriptions,
    queryFn: ({ signal }) => getLambdaSubscriptions(true, signal),
    enabled,
  });
}

// --------------------------------------------------------------------------- //
// (s7/a2) spec-definition chat — the distinct interactive spec-authoring surface.
// The transcript (GET) is server state in the cache; opening/messaging/approving/
// denying are mutations that write the fresh view back into that cache.
// --------------------------------------------------------------------------- //
export function useSpecChat(sessionId: string | null): UseQueryResult<SpecChatView> {
  return useQuery({
    queryKey: sessionId ? queryKeys.specChat(sessionId) : ["spec-chat", "none"],
    queryFn: ({ signal }) => getSpecChat(sessionId as string, signal),
    enabled: sessionId !== null,
  });
}

export function useOpenSpecChat() {
  return useMutation<OpenSpecChatResponse, Error, { name: string; description?: string }>({
    mutationFn: ({ name, description }) => openSpecChat(name, description),
  });
}

export function useSendSpecChatMessage() {
  const qc = useQueryClient();
  return useMutation<SpecChatMessageResponse, Error, { sessionId: string; message: string }>({
    mutationFn: ({ sessionId, message }) => sendSpecChatMessage(sessionId, message),
    onSuccess: (resp) => {
      // Reconcile the transcript cache from the authoritative GET view so the
      // surface reflects every recorded turn (not just the optimistic draft).
      void qc.invalidateQueries({ queryKey: queryKeys.specChat(resp.session_id) });
    },
  });
}

export function useApproveSpecChat() {
  const qc = useQueryClient();
  return useMutation<ApproveSpecResponse, Error, { sessionId: string }>({
    mutationFn: ({ sessionId }) => approveSpecChat(sessionId),
    onSuccess: (resp) => {
      void qc.invalidateQueries({ queryKey: queryKeys.specChat(resp.session_id) });
    },
  });
}

export function useDenySpecChat() {
  const qc = useQueryClient();
  return useMutation<DenySpecResponse, Error, { sessionId: string }>({
    mutationFn: ({ sessionId }) => denySpecChat(sessionId),
    onSuccess: (resp) => {
      void qc.invalidateQueries({ queryKey: queryKeys.specChat(resp.session_id) });
    },
  });
}

// --------------------------------------------------------------------------- //
// (s4/RC7) RE-EDITABLE specialization surface — the registered-spec index + the
// per-spec view are server state in the cache; the direct edit (PUT) and the
// re-open-into-chat are mutations. The PUT writes the fresh persisted spec back
// into the cache AND invalidates the index, so the edited body is reflected the
// instant it lands (no fetch-in-effect).
// --------------------------------------------------------------------------- //
export function useRegisteredSpecs(enabled = true): UseQueryResult<RegisteredSpecRow[]> {
  return useQuery({
    queryKey: queryKeys.registeredSpecs,
    queryFn: ({ signal }) => listRegisteredSpecs(signal),
    enabled,
  });
}

export function useRegisteredSpec(name: string | null): UseQueryResult<RegisteredSpec> {
  return useQuery({
    queryKey: name ? queryKeys.registeredSpec(name) : ["spec-chats", "registered", "none"],
    queryFn: ({ signal }) => getRegisteredSpec(name as string, signal),
    enabled: name !== null,
  });
}

export function useUpdateRegisteredSpec() {
  const qc = useQueryClient();
  return useMutation<RegisteredSpec, Error, { name: string; edit: UpdateSpecRequest }>({
    mutationFn: ({ name, edit }) => updateRegisteredSpec(name, edit),
    onSuccess: (spec) => {
      // The PUT returns the freshly-persisted spec — seed it straight into the
      // per-spec cache so the editor reflects the saved body, then refresh the
      // body-free index (description may have changed).
      qc.setQueryData(queryKeys.registeredSpec(spec.name), spec);
      void qc.invalidateQueries({ queryKey: queryKeys.registeredSpecs });
    },
  });
}

/**
 * DELETE a registered specialization (s4/a4, d13 UI). On success, drop the
 * per-spec cache entry and invalidate the body-free index so the row clears from
 * the visible list immediately (no fetch-in-effect). Any registered spec is
 * deletable (specs have no built-ins).
 */
export function useDeleteRegisteredSpec() {
  const qc = useQueryClient();
  return useMutation<void, Error, { name: string }>({
    mutationFn: ({ name }) => deleteRegisteredSpec(name),
    onSuccess: (_void, { name }) => {
      qc.removeQueries({ queryKey: queryKeys.registeredSpec(name) });
      void qc.invalidateQueries({ queryKey: queryKeys.registeredSpecs });
    },
  });
}

export function useReopenSpecChat() {
  const qc = useQueryClient();
  return useMutation<SpecChatView, Error, { name: string }>({
    mutationFn: ({ name }) => reopenSpecChat(name),
    onSuccess: (view) => {
      // The reopen response IS the started transcript — prime the spec-chat cache
      // so the conversation surface renders the seeded draft without a re-GET.
      qc.setQueryData(queryKeys.specChat(view.session_id), view);
    },
  });
}

// --------------------------------------------------------------------------- //
// (s4/a5, d5/d9) SHAPES screen — the text-file shape catalog + per-shape view are
// server state in the cache; setting a shape's max_iter is a mutation. The PUT
// returns the freshly-merged shape view (override + effective), so it is seeded
// straight into both the list and the per-shape cache — the new effective round
// count is reflected the instant it lands (no fetch-in-effect).
// --------------------------------------------------------------------------- //
export function useShapes(enabled = true): UseQueryResult<ShapeView[]> {
  return useQuery({
    queryKey: queryKeys.shapes,
    queryFn: ({ signal }) => listShapes(signal),
    enabled,
  });
}

export function useShape(name: string | null): UseQueryResult<ShapeView> {
  return useQuery({
    queryKey: name ? queryKeys.shape(name) : ["shapes", "none"],
    queryFn: ({ signal }) => getShape(name as string, signal),
    enabled: name !== null,
  });
}

export function useSetShapeMaxIter() {
  const qc = useQueryClient();
  return useMutation<ShapeView, Error, { name: string; maxIter: number }>({
    mutationFn: ({ name, maxIter }) => setShapeMaxIter(name, maxIter),
    onSuccess: (shape) => {
      // The PUT returns the updated view (override + effective_max_iter clamped to
      // hard_cap) — seed it into the per-shape cache and patch the same row in the
      // list cache so both reflect the persisted override without a re-GET.
      qc.setQueryData(queryKeys.shape(shape.name), shape);
      qc.setQueryData<ShapeView[]>(queryKeys.shapes, (prev) =>
        prev?.map((s) => (s.name === shape.name ? shape : s)),
      );
    },
  });
}

/**
 * DELETE a USER-AUTHORED shape (s4/a4, d13 UI). On success, drop the per-shape
 * cache entry and invalidate the catalog so the row clears from the visible list
 * immediately. The backend 409s a shipped built-in (the UI hides the control on
 * built-ins, but a 409 still surfaces via the mutation error as the real guard).
 */
export function useDeleteShape() {
  const qc = useQueryClient();
  return useMutation<void, Error, { name: string }>({
    mutationFn: ({ name }) => deleteShape(name),
    onSuccess: (_void, { name }) => {
      qc.removeQueries({ queryKey: queryKeys.shape(name) });
      void qc.invalidateQueries({ queryKey: queryKeys.shapes });
    },
  });
}

// --------------------------------------------------------------------------- //
// (s9/b1, d14(2)/d9) DESCRIBE-A-SHAPE — author a new shape from an NL description
// (POST /shapes/author). The response IS the authored shape's full view; seed it
// into the per-shape cache and invalidate the catalog so the new shape appears in
// the list immediately (no fetch-in-effect).
// --------------------------------------------------------------------------- //
export function useAuthorShape() {
  const qc = useQueryClient();
  return useMutation<ShapeView, Error, { description: string; nameHint?: string }>({
    mutationFn: ({ description, nameHint }) => authorShape(description, nameHint),
    onSuccess: (shape) => {
      qc.setQueryData(queryKeys.shape(shape.name), shape);
      void qc.invalidateQueries({ queryKey: queryKeys.shapes });
    },
  });
}

// (s8/b6, d18a) REFINE-A-SHAPE — edit an existing shape in plain language; the live
// Gemma model authors the next version building on the current one
// (POST /shapes/{name}/refine). The response IS the refined shape's full view; seed
// it into the per-shape cache and patch the same row in the catalog so the edited
// structure (e.g. a sequential→concurrent posture flip) shows immediately.
export function useRefineShape() {
  const qc = useQueryClient();
  return useMutation<ShapeView, Error, { name: string; instruction: string }>({
    mutationFn: ({ name, instruction }) => refineShape(name, instruction),
    onSuccess: (shape) => {
      qc.setQueryData(queryKeys.shape(shape.name), shape);
      qc.setQueryData<ShapeView[]>(queryKeys.shapes, (prev) =>
        prev?.map((s) => (s.name === shape.name ? shape : s)),
      );
      void qc.invalidateQueries({ queryKey: queryKeys.shapes });
    },
  });
}

// --------------------------------------------------------------------------- //
// (s17, d18a/d249 parity) SHAPE CHAT — the conversational draft-based authoring
// hooks. The session view is returned by every drive, so the panel holds it as
// mutation state; approve lands the new/updated shape in the catalog caches.
// --------------------------------------------------------------------------- //
export function useOpenShapeChat() {
  return useMutation<ShapeChatView, Error, { refineOf?: string | undefined }>({
    mutationFn: ({ refineOf }) => openShapeChat(refineOf),
  });
}

export function useSendShapeChatMessage() {
  return useMutation<ShapeChatView, Error, { sessionId: string; message: string }>({
    mutationFn: ({ sessionId, message }) => sendShapeChatMessage(sessionId, message),
  });
}

export function useApproveShapeChat() {
  const qc = useQueryClient();
  return useMutation<ApproveShapeChatResponse, Error, { sessionId: string }>({
    mutationFn: ({ sessionId }) => approveShapeChat(sessionId),
    onSuccess: (res) => {
      qc.setQueryData(queryKeys.shape(res.shape.name), res.shape);
      void qc.invalidateQueries({ queryKey: queryKeys.shapes });
    },
  });
}

export function useDenyShapeChat() {
  return useMutation<{ denied: boolean }, Error, { sessionId: string }>({
    mutationFn: ({ sessionId }) => denyShapeChat(sessionId),
  });
}
