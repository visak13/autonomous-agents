import { NODE_STATUS_ORDER, isTerminal, statusLabel } from "../../api/lifecycle";
import type { NodeStatus } from "../../api/types";
import type { DagNode } from "../../hooks/useTaskRun";
import "./DagView.css";

interface DagViewProps {
  nodes: DagNode[];
  runStatus: "idle" | "running" | "done" | "failed" | "cancelled";
}

// The lifecycle lane each node walks (eda-base3 lifecycle, d9). Terminal-but-not
// "done" states are shown as a distinct end marker rather than a lane step.
const LIFECYCLE_LANE: readonly NodeStatus[] = ["pending", "running", "verifiable", "done"];

export function DagView({ nodes, runStatus }: DagViewProps) {
  const activeId = currentNodeId(nodes);

  return (
    <section className="dag" aria-label="Plan / DAG">
      <header className="dag-header">
        <h2 className="dag-title">Plan</h2>
        <span className={`dag-run-status dag-run-${runStatus}`}>{runStatusLabel(runStatus)}</span>
      </header>

      <div className="dag-body">
        {nodes.length === 0 ? (
          <p className="dag-empty">
            The plan appears here when a run starts — each task node moves through
            pending → in progress → verifying → done.
          </p>
        ) : (
          <ol className="dag-nodes">
            {nodes.map((node) => (
              <NodeCard key={node.id} node={node} current={node.id === activeId} />
            ))}
          </ol>
        )}
      </div>
    </section>
  );
}

/** The node the run is "on" right now: the first non-terminal node, preferring
 * a verifiable/running one. Drives the current-task highlight. */
function currentNodeId(nodes: DagNode[]): string | null {
  const active = nodes.find((n) => n.status === "running" || n.status === "verifiable");
  if (active) return active.id;
  const pending = nodes.find((n) => !isTerminal(n.status));
  return pending?.id ?? null;
}

function NodeCard({ node, current }: { node: DagNode; current: boolean }) {
  const laneIndex = LIFECYCLE_LANE.indexOf(node.status);
  const terminalBad = isTerminal(node.status) && node.status !== "done";

  return (
    <li
      className={`node-card${current ? " node-current" : ""}`}
      data-status={node.status}
      aria-current={current ? "step" : undefined}
    >
      <div className="node-head">
        <span className="node-id">{node.id}</span>
        <span className={`node-badge status-${node.status}`}>{statusLabel(node.status)}</span>
      </div>

      <div className="node-lane" role="presentation">
        {LIFECYCLE_LANE.map((stage, i) => {
          const reached = laneIndex >= 0 && i <= laneIndex;
          return (
            <span
              key={stage}
              className={`lane-step${reached ? " lane-reached" : ""}`}
              title={statusLabel(stage)}
            />
          );
        })}
      </div>

      {node.specs.length > 0 && (
        <div className="node-specs">
          {node.specs.map((s) => (
            <span className="spec-chip" key={s}>
              {s}
            </span>
          ))}
        </div>
      )}

      {node.attempts > 1 && <div className="node-attempts">attempt {node.attempts}</div>}
      {terminalBad && node.error && <div className="node-error">{node.error}</div>}
    </li>
  );
}

function runStatusLabel(status: DagViewProps["runStatus"]): string {
  switch (status) {
    case "idle":
      return "Idle";
    case "running":
      return "Running";
    case "done":
      return "Complete";
    case "failed":
      return "Failed";
    case "cancelled":
      return "Cancelled";
    default: {
      const _exhaustive: never = status;
      return _exhaustive;
    }
  }
}

// Re-exported for any consumer that wants the canonical lifecycle ordering.
export { NODE_STATUS_ORDER };
