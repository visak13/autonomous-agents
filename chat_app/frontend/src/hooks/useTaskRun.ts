/**
 * Drive ONE task run end to end and expose a live PLAN/DAG view.
 *
 * Flow (the decoupled, non-freezing path — d4):
 *   1. the chat's SSE stream is already open (subscribe-before-publish);
 *   2. `send()` POSTs /chats/{id}/runs and gets a run id back immediately (202);
 *   3. per-node lifecycle events arrive live on the SSE stream and build the DAG;
 *   4. the run id is polled until terminal; the final MessageResponse's
 *      node_states are the AUTHORITATIVE reconciliation (in case any live event
 *      was missed across a reconnect gap), and carry the artifacts.
 *
 * Server state (the run record, chat history) lives in the Query cache; only the
 * live, not-yet-persisted DAG accumulation is local reducer state.
 */
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { useResumeRun, useRunPolling, useStartRun } from "../api/queries";
import { statusForEvent } from "../api/lifecycle";
import type {
  ArtifactOut,
  MissingSpecChoice,
  MissingSpecialistPending,
  NodeStateOut,
  NodeStatus,
  StreamEnvelope,
} from "../api/types";
import { useChatStream, type StreamStatus } from "./useChatStream";

export interface DagNode {
  id: string;
  status: NodeStatus;
  attempts: number;
  error: string | null;
  /** Most recent spec(s) composed onto this node, if the event carried them. */
  specs: string[];
}

interface DagState {
  /** insertion-ordered node ids (the order events first mention them). */
  order: string[];
  byId: Record<string, DagNode>;
}

type DagAction =
  | { type: "reset" }
  | { type: "event"; env: StreamEnvelope }
  | { type: "reconcile"; states: NodeStateOut[] };

const RANK: Record<NodeStatus, number> = {
  pending: 0,
  running: 1,
  verifiable: 2,
  done: 3,
  // terminal-but-unsuccessful states rank above done so a reconcile can't be
  // overwritten by a stale live "verifiable", but they are not "more complete".
  failed: 3,
  skipped: 3,
  cancelled: 3,
};

const TERMINAL_STATES: ReadonlySet<NodeStatus> = new Set<NodeStatus>([
  "done",
  "failed",
  "skipped",
  "cancelled",
]);

function nodeIdOf(payload: unknown): string | null {
  if (payload && typeof payload === "object" && "node_id" in payload) {
    const v = (payload as { node_id: unknown }).node_id;
    return typeof v === "string" ? v : null;
  }
  return null;
}

function specsOf(payload: unknown): string[] | null {
  if (payload && typeof payload === "object" && "specs" in payload) {
    const v = (payload as { specs: unknown }).specs;
    if (Array.isArray(v)) return v.filter((s): s is string => typeof s === "string");
  }
  return null;
}

function errorOf(payload: unknown): string | null {
  if (payload && typeof payload === "object") {
    const o = payload as Record<string, unknown>;
    const e = o["error"] ?? o["reason"];
    if (typeof e === "string") return e;
  }
  return null;
}

function ensureNode(state: DagState, id: string): DagState {
  if (state.byId[id]) return state;
  return {
    order: [...state.order, id],
    byId: { ...state.byId, [id]: { id, status: "pending", attempts: 0, error: null, specs: [] } },
  };
}

function dagReducer(state: DagState, action: DagAction): DagState {
  switch (action.type) {
    case "reset":
      return { order: [], byId: {} };
    case "event": {
      const id = nodeIdOf(action.env.payload);
      if (id === null) return state;
      const next = ensureNode(state, id);
      const current = next.byId[id];
      if (!current) return next;
      const incoming = statusForEvent(action.env.kind);
      // Never regress a node that already reached a terminal state.
      if (incoming === null) {
        // event carried node-scoped metadata (specs) but no state change
        const specs = specsOf(action.env.payload);
        if (!specs) return next;
        return { ...next, byId: { ...next.byId, [id]: { ...current, specs } } };
      }
      if (TERMINAL_STATES.has(current.status) && RANK[incoming] <= RANK[current.status]) {
        return next;
      }
      const specs = specsOf(action.env.payload) ?? current.specs;
      const error = errorOf(action.env.payload) ?? (incoming === "failed" ? current.error : null);
      return {
        ...next,
        byId: { ...next.byId, [id]: { ...current, status: incoming, specs, error } },
      };
    }
    case "reconcile": {
      let next = state;
      for (const st of action.states) {
        next = ensureNode(next, st.node_id);
        const cur = next.byId[st.node_id];
        if (!cur) continue;
        next = {
          ...next,
          byId: {
            ...next.byId,
            [st.node_id]: { ...cur, status: st.status, attempts: st.attempts, error: st.error },
          },
        };
      }
      return next;
    }
    default: {
      const _exhaustive: never = action;
      return _exhaustive;
    }
  }
}

export interface UseTaskRunResult {
  nodes: DagNode[];
  artifacts: ArtifactOut[];
  runStatus: "idle" | "running" | "done" | "failed" | "cancelled";
  runError: string | null;
  streamStatus: StreamStatus;
  busy: boolean;
  send: (message: string, topic?: string) => void;
  /** Set when the just-finished run persisted a new turn (so callers can refresh history). */
  lastCompletedRunId: string | null;
  /** The user request of the run in flight FOR THIS CHAT (optimistic echo), else null. */
  pendingMessage: string | null;
  /** The CHOICE payload when the run PAUSED needing an unavailable specialist
   * (s10-a8), else null. Surfaced so the UI can render the resolution buttons. */
  pendingResolution: MissingSpecialistPending | null;
  /** Resolve the active missing-specialist pause: `sse_fallback` runs the unmet
   * node(s) spec-less; `define_and_resume` stamps `specName` onto them. */
  resume: (choice: MissingSpecChoice, specName?: string) => void;
  /** True while a resume round-trip is in flight. */
  resuming: boolean;
  /** A resume error message, else null. */
  resumeError: string | null;
}

export function useTaskRun(chatId: string | null): UseTaskRunResult {
  const [dag, dispatch] = useReducer(dagReducer, { order: [], byId: {} });
  const [runId, setRunId] = useState<string | null>(null);
  const [lastCompletedRunId, setLastCompletedRunId] = useState<string | null>(null);
  // The user request currently in flight + the chat whose start request is still
  // pending. Both are scoped to a single chat so a run started in one thread can
  // never drive the busy/echo state of another after a mid-run switch (no
  // cross-bleed — o4 isolation requirement).
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);
  const [pendingChatId, setPendingChatId] = useState<string | null>(null);
  // The resume_token of a pause the user has already RESOLVED — once set, the
  // resolution surface for that token hides even though the original run result
  // still reports missing_specialist=true (the resumed run is a separate POST,
  // not a new polled run). Cleared when a fresh run/chat starts.
  const [resolvedToken, setResolvedToken] = useState<string | null>(null);
  // Latest active chat, read inside the async mutation callback so a run whose
  // start resolves AFTER the user switched threads is dropped rather than
  // adopted by the now-current thread.
  const currentChatIdRef = useRef(chatId);
  currentChatIdRef.current = chatId;

  const onEvent = useCallback((env: StreamEnvelope) => {
    dispatch({ type: "event", env });
  }, []);

  const { status: streamStatus } = useChatStream(chatId, onEvent, chatId !== null);
  const startRun = useStartRun();
  const resumeRunMut = useResumeRun();
  const runQuery = useRunPolling(runId);

  // Reconcile against the authoritative terminal summary the instant the run is
  // done. Guarded by run id so it applies exactly once per run.
  const result = runQuery.data?.status === "done" ? runQuery.data.result : null;
  useEffect(() => {
    if (result && runId) {
      dispatch({ type: "reconcile", states: result.node_states });
      setLastCompletedRunId(runId);
      setPendingMessage(null); // the turn is now persisted; the real bubble takes over
    }
  }, [result, runId]);

  // Clear ALL per-run state when switching chats so the previous thread's live
  // run (DAG, run id, optimistic echo, any resolved pause) never bleeds into the
  // newly-selected one.
  useEffect(() => {
    dispatch({ type: "reset" });
    setRunId(null);
    setPendingMessage(null);
    setPendingChatId(null);
    setResolvedToken(null);
    resumeRunMut.reset();
  }, [chatId]);

  const send = useCallback(
    (message: string, topic?: string) => {
      if (!chatId) return;
      const startedFor = chatId;
      dispatch({ type: "reset" });
      setRunId(null);
      setPendingMessage(message);
      setPendingChatId(startedFor);
      setResolvedToken(null);
      resumeRunMut.reset();
      startRun.mutate(
        { chatId, message, ...(topic !== undefined ? { topic } : {}) },
        {
          // Only adopt the run id if the user is still on the chat it was started
          // for; otherwise the run completes server-side and is picked up via the
          // chat-history refetch on switch-back (never shown in another thread).
          onSuccess: (resp) => {
            if (currentChatIdRef.current === startedFor) setRunId(resp.run_id);
          },
          onSettled: () => {
            setPendingChatId((p) => (p === startedFor ? null : p));
          },
        },
      );
    },
    [chatId, startRun],
  );

  // The active missing-specialist pause for THIS chat: the just-finished run
  // reported missing_specialist with a CHOICE payload, AND the user has not yet
  // resolved that token. (Scoped to this chat — `result` is keyed on this chat's
  // run id; a chat switch resets resolvedToken and the run id above.)
  const pendingResolution: MissingSpecialistPending | null =
    result?.missing_specialist && result.pending && result.pending.resume_token !== resolvedToken
      ? result.pending
      : null;

  const resume = useCallback(
    (choice: MissingSpecChoice, specName?: string) => {
      if (!chatId || !pendingResolution || resumeRunMut.isPending) return;
      const token = pendingResolution.resume_token;
      const req =
        choice === "define_and_resume" && specName
          ? { resume_token: token, choice, spec_name: specName }
          : { resume_token: token, choice };
      // The resumed plan runs on THIS chat's plane, so its node lifecycle streams
      // live over the already-open SSE — start from a clean DAG, then take the
      // resumed summary's node_states as the authoritative reconciliation.
      dispatch({ type: "reset" });
      resumeRunMut.mutate(
        { chatId, req },
        {
          onSuccess: (resp) => {
            dispatch({ type: "reconcile", states: resp.node_states });
            setResolvedToken(token);
          },
        },
      );
    },
    [chatId, pendingResolution, resumeRunMut],
  );

  const nodes = useMemo(
    () => dag.order.map((id) => dag.byId[id]).filter((n): n is DagNode => n !== undefined),
    [dag],
  );

  // The start mutation is a single shared object; gate its pending/error on the
  // chat its start belongs to so it cannot mark a different thread busy. Once the
  // run id is adopted (same chat), runQuery (keyed on that id) drives the state.
  const startingThisChat = startRun.isPending && pendingChatId === chatId;
  const rawRunStatus = runQuery.data?.status ?? null;
  const runStatus: UseTaskRunResult["runStatus"] = startingThisChat
    ? "running"
    : rawRunStatus ?? "idle";
  const busy = startingThisChat || rawRunStatus === "running";
  const startError = pendingChatId === chatId ? startRun.error?.message ?? null : null;
  const runError = startError ?? runQuery.data?.error ?? null;

  return {
    nodes,
    artifacts: result?.artifacts ?? [],
    runStatus,
    runError,
    streamStatus,
    busy,
    send,
    lastCompletedRunId,
    pendingMessage,
    pendingResolution,
    resume,
    resuming: resumeRunMut.isPending,
    resumeError: resumeRunMut.error?.message ?? null,
  };
}
