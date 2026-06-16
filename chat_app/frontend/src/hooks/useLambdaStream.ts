/**
 * Subscribe to the reactive-lambda META plane (app.py GET /lambda/stream).
 *
 * This is the READ-ONLY live channel behind the lambda tab (d15): the agents
 * author + use the lambdas; the user only observes. Like the chat stream it uses
 * the browser-native `EventSource` (auto-reconnect on drop; the plane has no
 * replay buffer, so a reconnect resumes the live feed and the snapshot query is
 * the reconciliation point). The server emits NAMED events, so we register a
 * listener per known meta kind plus the `connected` handshake.
 *
 * No phantom effects: one effect owns the EventSource lifecycle keyed on
 * `enabled`, reading the latest `onEvent` through a ref so a changing callback
 * never tears down the connection.
 */
import { useEffect, useRef, useState } from "react";
import { lambdaStreamUrl } from "../api/client";
import type { LambdaMetaEnvelope, LambdaMetaKind } from "../api/types";
import type { StreamStatus } from "./useChatStream";

const LAMBDA_META_KINDS: readonly Exclude<LambdaMetaKind, "connected">[] = [
  "lambda_registered",
  "lambda_fired",
  "lambda_closed",
  "lambda_observation",
];

export function useLambdaStream(
  enabled: boolean,
  onEvent: (env: LambdaMetaEnvelope) => void,
): { status: StreamStatus } {
  const [status, setStatus] = useState<StreamStatus>("idle");
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  useEffect(() => {
    if (!enabled) {
      setStatus("idle");
      return;
    }
    setStatus("connecting");
    const es = new EventSource(lambdaStreamUrl());
    let opened = false;

    const dispatch = (kind: LambdaMetaKind) => (ev: MessageEvent<string>) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(ev.data);
      } catch {
        return; // malformed frame — skip rather than crash the stream
      }
      if (kind === "connected") return; // handshake only
      if (parsed && typeof parsed === "object") {
        const env = parsed as Partial<LambdaMetaEnvelope>;
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
    for (const kind of LAMBDA_META_KINDS) {
      es.addEventListener(kind, dispatch(kind) as EventListener);
    }
    es.onopen = () => {
      opened = true;
      setStatus("open");
    };
    es.onerror = () => {
      // EventSource retries on its own unless closed; reflect, don't close.
      setStatus(opened ? "reconnecting" : "connecting");
    };

    return () => {
      es.close();
      setStatus("idle");
    };
  }, [enabled]);

  return { status };
}
