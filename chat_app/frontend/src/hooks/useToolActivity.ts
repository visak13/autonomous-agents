/**
 * The chain-of-thought feed: the agent's live tool activity for ONE chat.
 *
 * Subscribes to the SAME per-chat SSE stream the DAG view uses (routes.py
 * /chats/{id}/stream), but reads only the `tool_call` / `tool_result` events
 * (reactive_tools/tool_hook.py). This is an INDEPENDENT EventSource from a1's
 * useTaskRun — the in-process plane supports many subscribers (subscribe-before-
 * publish, one Subscription per connection), so the CoT overlay is purely
 * additive and never perturbs the DAG run logic.
 *
 * A tool call and its result are correlated by `call_id`: a `tool_call` opens a
 * `pending` entry; the matching `tool_result` resolves it to `ok` / `error`.
 * State is a local reducer (live, not-yet-persisted UI state — not server state),
 * keyed on chatId so switching chats clears it. No fetch-in-effect, no phantom
 * effects: the only effect is the SSE lifecycle owned by useChatStream.
 */
import { useCallback, useEffect, useMemo, useReducer } from "react";
import type {
  StreamEnvelope,
  ToolCallPayload,
  ToolResultPayload,
} from "../api/types";
import { useChatStream, type StreamStatus } from "./useChatStream";

export type ToolActivityStatus = "pending" | "ok" | "error";

export interface ToolActivity {
  callId: string;
  name: string;
  args: Record<string, unknown>;
  status: ToolActivityStatus;
  /** Set once the matching tool_result lands (ok→value summary, error→message). */
  result: string | null;
  /** Monotonic insertion index, for stable ordering + keys. */
  seq: number;
}

interface ActivityState {
  order: string[]; // call_ids in arrival order
  byId: Record<string, ToolActivity>;
  counter: number;
}

type ActivityAction =
  | { type: "reset" }
  | { type: "call"; payload: ToolCallPayload }
  | { type: "result"; payload: ToolResultPayload };

const EMPTY: ActivityState = { order: [], byId: {}, counter: 0 };

/** Compact a tool's return value into a single short line for the overlay. */
function summarizeValue(value: unknown): string {
  if (value === null || value === undefined) return "ok";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    const json = JSON.stringify(value);
    return json.length > 240 ? `${json.slice(0, 240)}…` : json;
  } catch {
    return "ok";
  }
}

// call_id is emitted as an INTEGER by tool_hook.py (_next_call_id), so accept
// string|number here and normalize to a string key in the reducer — a "string"
// only guard silently dropped every live tool frame (s7/a4 finding).
function isCallId(v: unknown): v is string | number {
  return typeof v === "string" || typeof v === "number";
}

function isToolCall(payload: unknown): payload is ToolCallPayload {
  return (
    !!payload &&
    typeof payload === "object" &&
    isCallId((payload as ToolCallPayload).call_id) &&
    typeof (payload as ToolCallPayload).name === "string"
  );
}

function isToolResult(payload: unknown): payload is ToolResultPayload {
  return (
    !!payload &&
    typeof payload === "object" &&
    isCallId((payload as ToolResultPayload).call_id) &&
    typeof (payload as ToolResultPayload).ok === "boolean"
  );
}

function activityReducer(state: ActivityState, action: ActivityAction): ActivityState {
  switch (action.type) {
    case "reset":
      return EMPTY;
    case "call": {
      const { name, args } = action.payload;
      const call_id = String(action.payload.call_id); // normalize int|string -> key
      if (state.byId[call_id]) return state; // duplicate frame — ignore
      const entry: ToolActivity = {
        callId: call_id,
        name,
        args: args ?? {},
        status: "pending",
        result: null,
        seq: state.counter,
      };
      return {
        order: [...state.order, call_id],
        byId: { ...state.byId, [call_id]: entry },
        counter: state.counter + 1,
      };
    }
    case "result": {
      const { name, ok, value, error } = action.payload;
      const call_id = String(action.payload.call_id); // normalize int|string -> key
      const existing = state.byId[call_id];
      // A result may arrive for a call we never saw the open frame of (reconnect
      // gap) — synthesize the entry so nothing is dropped.
      const base: ToolActivity = existing ?? {
        callId: call_id,
        name,
        args: {},
        status: "pending",
        result: null,
        seq: state.counter,
      };
      const resolved: ToolActivity = {
        ...base,
        status: ok ? "ok" : "error",
        result: ok ? summarizeValue(value) : error ?? "failed",
      };
      const order = existing ? state.order : [...state.order, call_id];
      const counter = existing ? state.counter : state.counter + 1;
      return { order, byId: { ...state.byId, [call_id]: resolved }, counter };
    }
    default: {
      const _exhaustive: never = action;
      return _exhaustive;
    }
  }
}

export interface UseToolActivityResult {
  activities: ToolActivity[];
  /** Count of calls still awaiting a result — drives the "thinking" pulse. */
  pendingCount: number;
  streamStatus: StreamStatus;
}

export function useToolActivity(
  chatId: string | null,
  enabled: boolean,
): UseToolActivityResult {
  const [state, dispatch] = useReducer(activityReducer, EMPTY);

  // Clear the feed when switching chats — the tool activity is per-chat live UI
  // state, and the SSE subscription re-keys on chatId, so without this reset the
  // previous chat's tool calls would leak into the next chat's overlay.
  useEffect(() => {
    dispatch({ type: "reset" });
  }, [chatId]);

  const onEvent = useCallback((env: StreamEnvelope) => {
    if (env.kind === "tool_call" && isToolCall(env.payload)) {
      dispatch({ type: "call", payload: env.payload });
    } else if (env.kind === "tool_result" && isToolResult(env.payload)) {
      dispatch({ type: "result", payload: env.payload });
    }
    // All other kinds (node lifecycle, handshake) are the DAG's concern, not the
    // CoT overlay's — ignored here without a state change.
  }, []);

  const { status } = useChatStream(chatId, onEvent, enabled && chatId !== null);

  const activities = useMemo(
    () =>
      state.order
        .map((id) => state.byId[id])
        .filter((a): a is ToolActivity => a !== undefined),
    [state],
  );

  const pendingCount = useMemo(
    () => activities.filter((a) => a.status === "pending").length,
    [activities],
  );

  return { activities, pendingCount, streamStatus: status };
}
