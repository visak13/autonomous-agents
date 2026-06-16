/**
 * Subscribe to one chat's live SSE stream (routes.py GET /chats/{id}/stream).
 *
 * Uses the browser-native `EventSource`, which RECONNECTS automatically on a
 * dropped connection (the action's "reconnect on drop"); the server has no
 * replay buffer, so a reconnect simply resumes the live feed. The server emits
 * NAMED events (`event: <kind>`), so we register a listener per known kind plus
 * the `connected` handshake — `onmessage` alone would miss them all.
 *
 * No phantom effects: the single effect owns the EventSource lifecycle keyed on
 * `chatId`+`enabled`, and reads the latest `onEvent` through a ref so a changing
 * callback never tears down and rebuilds the connection.
 */
import { useEffect, useRef, useState } from "react";
import { chatStreamUrl } from "../api/client";
import type { StreamEnvelope, StreamEventKind } from "../api/types";

const NAMED_EVENT_KINDS: readonly StreamEventKind[] = [
  "agent_node_launched",
  "agent_node_done",
  "agent_node_failed",
  "agent_node_healed",
  "agent_node_cancelled",
  "agent_node_replanned",
  "agent_node_skipped",
  "agent_node_verifiable",
  "agent_node_review",
  "agent_node_inline_fixed",
  "agent_node_verify_failed",
  "agent_node_collision",
  "agent_node_collision_resolved",
  "tool_call",
  "tool_result",
];

export type StreamStatus = "connecting" | "open" | "reconnecting" | "idle";

export interface UseChatStreamResult {
  status: StreamStatus;
}

export function useChatStream(
  chatId: string | null,
  onEvent: (env: StreamEnvelope) => void,
  enabled: boolean,
): UseChatStreamResult {
  const [status, setStatus] = useState<StreamStatus>("idle");
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  useEffect(() => {
    if (!chatId || !enabled) {
      setStatus("idle");
      return;
    }
    setStatus("connecting");
    const es = new EventSource(chatStreamUrl(chatId));
    let opened = false;

    const dispatch = (kind: StreamEventKind) => (ev: MessageEvent<string>) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(ev.data);
      } catch {
        return; // malformed frame — skip rather than crash the stream
      }
      if (kind === "connected") return; // handshake only
      if (parsed && typeof parsed === "object") {
        const env = parsed as Partial<StreamEnvelope>;
        onEventRef.current({
          kind,
          seq: typeof env.seq === "number" ? env.seq : 0,
          source: typeof env.source === "string" ? env.source : "",
          payload: env.payload,
        });
      }
    };

    es.addEventListener("connected", () => {
      opened = true;
      setStatus("open");
    });
    for (const kind of NAMED_EVENT_KINDS) {
      es.addEventListener(kind, dispatch(kind) as EventListener);
    }
    es.onopen = () => {
      opened = true;
      setStatus("open");
    };
    es.onerror = () => {
      // EventSource transitions to CONNECTING and retries on its own unless
      // it has been closed. Reflect that in the UI; do not close it ourselves.
      setStatus(opened ? "reconnecting" : "connecting");
    };

    return () => {
      es.close();
      setStatus("idle");
    };
  }, [chatId, enabled]);

  return { status };
}
