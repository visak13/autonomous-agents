/**
 * (a) CHAIN-OF-THOUGHT brain-icon overlay (s7/a2, feature a; d13).
 *
 * A brain icon pinned in the RIGHT corner that COLLAPSES into itself and EXPANDS
 * into a small pop-up widget streaming, in real time, the agent's live tool calls
 * and what it is doing — driven off the per-chat SSE tool_call/tool_result plane
 * (useToolActivity). The icon pulses while a call is in flight.
 *
 * Self-contained: it owns its own collapsed state and its own stream subscription
 * (additive to a1's DAG stream). Keyboard-accessible: the launcher is a real
 * <button> with aria-expanded; the panel is a labelled region.
 */
import { useState } from "react";
import { useToolActivity, type ToolActivity } from "../../hooks/useToolActivity";
import "./CotOverlay.css";

interface CotOverlayProps {
  chatId: string | null;
}

export function CotOverlay({ chatId }: CotOverlayProps) {
  const [open, setOpen] = useState(false);
  const { activities, pendingCount } = useToolActivity(chatId, chatId !== null);

  // Nothing to think about until a chat is selected.
  if (chatId === null) return null;

  const thinking = pendingCount > 0;
  const total = activities.length;

  return (
    <div className="cot-overlay" data-open={open}>
      {open && (
        <section className="cot-panel" aria-label="Agent chain of thought">
          <header className="cot-panel-head">
            <span className="cot-panel-title">
              <span className="cot-brain" aria-hidden="true">🧠</span>
              Chain of thought
            </span>
            <button
              type="button"
              className="cot-collapse"
              onClick={() => setOpen(false)}
              aria-label="Collapse chain-of-thought overlay"
              title="Collapse"
            >
              ×
            </button>
          </header>
          <div className="cot-panel-status" aria-live="polite">
            {thinking ? (
              <span className="cot-live">
                <span className="cot-live-dot" aria-hidden="true" />
                working… {pendingCount} tool{pendingCount === 1 ? "" : "s"} in flight
              </span>
            ) : (
              <span className="cot-idle">
                {total === 0 ? "no tool activity yet" : `idle · ${total} tool call${total === 1 ? "" : "s"}`}
              </span>
            )}
          </div>
          <ol className="cot-feed">
            {activities.length === 0 ? (
              <li className="cot-empty">
                The agent's live tool calls stream here as it works.
              </li>
            ) : (
              activities.map((a) => <ActivityRow key={a.callId} activity={a} />)
            )}
          </ol>
        </section>
      )}

      <button
        type="button"
        className={`cot-launcher${thinking ? " cot-launcher-active" : ""}`}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label={open ? "Collapse chain-of-thought overlay" : "Expand chain-of-thought overlay"}
        title="Chain of thought"
      >
        <span className="cot-launcher-icon" aria-hidden="true">🧠</span>
        {thinking && <span className="cot-launcher-pulse" aria-hidden="true" />}
        {!open && total > 0 && <span className="cot-launcher-count" aria-hidden="true">{total}</span>}
      </button>
    </div>
  );
}

function ActivityRow({ activity }: { activity: ToolActivity }) {
  const argKeys = Object.keys(activity.args);
  const argSummary =
    argKeys.length === 0 ? null : argKeys.map((k) => `${k}=${formatArg(activity.args[k])}`).join(", ");

  return (
    <li className="cot-item" data-status={activity.status}>
      <div className="cot-item-head">
        <span className="cot-item-icon" aria-hidden="true">{statusIcon(activity.status)}</span>
        <span className="cot-item-name">{activity.name}</span>
        <span className={`cot-item-badge cot-${activity.status}`}>{statusText(activity.status)}</span>
      </div>
      {argSummary && <div className="cot-item-args">{argSummary}</div>}
      {activity.result && (
        <div className="cot-item-result" data-status={activity.status}>
          {activity.result}
        </div>
      )}
    </li>
  );
}

function statusIcon(status: ToolActivity["status"]): string {
  switch (status) {
    case "pending":
      return "◐";
    case "ok":
      return "✓";
    case "error":
      return "✗";
    default: {
      const _exhaustive: never = status;
      return _exhaustive;
    }
  }
}

function statusText(status: ToolActivity["status"]): string {
  switch (status) {
    case "pending":
      return "running";
    case "ok":
      return "done";
    case "error":
      return "error";
    default: {
      const _exhaustive: never = status;
      return _exhaustive;
    }
  }
}

function formatArg(value: unknown): string {
  if (typeof value === "string") return value.length > 48 ? `"${value.slice(0, 48)}…"` : `"${value}"`;
  if (value === null || value === undefined) return "null";
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    const json = JSON.stringify(value);
    return json.length > 48 ? `${json.slice(0, 48)}…` : json;
  } catch {
    return "…";
  }
}
