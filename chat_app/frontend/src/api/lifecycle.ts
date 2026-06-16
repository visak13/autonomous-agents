/**
 * Lifecycle reducers over the backend's node-event stream.
 *
 * The single source of truth for "what does this SSE event do to a node's
 * state". Both the exhaustive `switch` below MUST handle every StreamEventKind:
 * the `const _exhaustive: never = kind` default turns a NEW backend event kind
 * into a COMPILE error here (spec [required]), instead of a node silently
 * sticking in the wrong state.
 */
import type { NodeStatus, StreamEventKind } from "./types";

/** Ordering used to render lifecycle progress and to never regress a node. */
export const NODE_STATUS_ORDER: readonly NodeStatus[] = [
  "pending",
  "running",
  "verifiable",
  "done",
  "failed",
  "skipped",
  "cancelled",
];

/** Terminal states a node never leaves. */
const TERMINAL: ReadonlySet<NodeStatus> = new Set<NodeStatus>([
  "done",
  "failed",
  "skipped",
  "cancelled",
]);

export function isTerminal(status: NodeStatus): boolean {
  return TERMINAL.has(status);
}

/**
 * Map an incoming stream event kind to the node status it implies, or `null`
 * when the event does not change a node's lifecycle state (e.g. the stream
 * handshake, tool traffic, or a transient heal/review notice that does not by
 * itself move the node forward).
 *
 * Exhaustive by construction: the `never` default fails to compile if a new
 * StreamEventKind is added without a decision here.
 */
export function statusForEvent(kind: StreamEventKind): NodeStatus | null {
  switch (kind) {
    case "agent_node_launched":
      return "running";
    case "agent_node_verifiable":
      return "verifiable";
    case "agent_node_done":
    case "agent_node_inline_fixed":
      return "done";
    case "agent_node_failed":
    case "agent_node_verify_failed":
      return "failed";
    case "agent_node_skipped":
      return "skipped";
    case "agent_node_cancelled":
      return "cancelled";
    // Transient notices: the node stays in its current phase. A heal/replan
    // re-runs the produce step (back to running); a review is the verify gate
    // working; a collision pauses but does not change the discrete status set.
    case "agent_node_healed":
    case "agent_node_replanned":
      return "running";
    case "agent_node_review":
    case "agent_node_collision":
    case "agent_node_collision_resolved":
      return "verifiable";
    // Not node-state transitions.
    case "connected":
    case "tool_call":
    case "tool_result":
      return null;
    default: {
      const _exhaustive: never = kind;
      return _exhaustive;
    }
  }
}

/** Human label for a node status (single source for UI copy). */
export function statusLabel(status: NodeStatus): string {
  switch (status) {
    case "pending":
      return "Pending";
    case "running":
      return "In progress";
    case "verifiable":
      return "Verifying";
    case "done":
      return "Done";
    case "failed":
      return "Failed";
    case "skipped":
      return "Skipped";
    case "cancelled":
      return "Cancelled";
    default: {
      const _exhaustive: never = status;
      return _exhaustive;
    }
  }
}
