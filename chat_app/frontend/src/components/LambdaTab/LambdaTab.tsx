/**
 * (b) LAMBDA TAB (s7/a2, feature b; d15).
 *
 * A collapsible tab in the chat-input header that is a STRICTLY READ-ONLY live
 * view of the reactive subscriptions the AGENTS created and are using. The user
 * OBSERVES ONLY — there is deliberately no author/edit/close control here (d15:
 * agent = author+consumer; user = passive observer).
 *
 * Data flow (spec [required]: server state lives in the Query cache, not
 * useState): the GET /lambda/subscriptions snapshot is the source of truth; the
 * /lambda/stream META plane (useLambdaStream) pushes incremental
 * registered/fired/closed updates straight into that cache via setQueryData, so
 * the tab stays live with no fetch-in-effect and no second copy of the data.
 */
import { useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useLambdaSubscriptions, queryKeys } from "../../api/queries";
import { useLambdaStream } from "../../hooks/useLambdaStream";
import type {
  LambdaFiredPayload,
  LambdaMetaEnvelope,
  LambdaSnapshot,
  LambdaSubscriptionView,
} from "../../api/types";
import "./LambdaTab.css";

interface LambdaTabProps {
  open: boolean;
  onToggle: () => void;
}

const EMPTY_SNAPSHOT: LambdaSnapshot = { active: 0, total: 0, subscriptions: [] };

export function LambdaTab({ open, onToggle }: LambdaTabProps) {
  const qc = useQueryClient();
  // Keep the snapshot warm even when collapsed so the badge count is live; the
  // SSE channel only runs while expanded (no point streaming into a hidden view).
  const { data, isLoading, isError } = useLambdaSubscriptions(true);
  const snapshot = data ?? EMPTY_SNAPSHOT;

  const onMeta = useCallback(
    (env: LambdaMetaEnvelope) => {
      qc.setQueryData<LambdaSnapshot>(queryKeys.lambdaSubscriptions, (prev) =>
        applyMetaEvent(prev ?? EMPTY_SNAPSHOT, env),
      );
    },
    [qc],
  );
  useLambdaStream(open, onMeta);

  const activeCount = snapshot.subscriptions.filter((s) => s.status === "active").length;

  return (
    <div className="lambda-tab" data-open={open}>
      <button
        type="button"
        className="lambda-tab-toggle"
        onClick={onToggle}
        aria-expanded={open}
        aria-controls="lambda-tab-panel"
      >
        <span className="lambda-tab-glyph" aria-hidden="true">λ</span>
        <span className="lambda-tab-label">Reactive subscriptions</span>
        <span className="lambda-tab-count" aria-label={`${activeCount} active`}>
          {activeCount} active
        </span>
        <span className="lambda-tab-chevron" aria-hidden="true">{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <div className="lambda-tab-panel" id="lambda-tab-panel" role="region" aria-label="Agent reactive subscriptions (read-only)">
          <p className="lambda-tab-note">
            Read-only — these lambdas are created and used by the agents. You observe; you never author.
          </p>
          {isLoading && <p className="lambda-tab-hint">Loading subscriptions…</p>}
          {isError && <p className="lambda-tab-error" role="alert">Could not load subscriptions.</p>}
          {!isLoading && !isError && snapshot.subscriptions.length === 0 && (
            <p className="lambda-tab-hint">No reactive subscriptions yet — they appear as the agents create them.</p>
          )}
          <ul className="lambda-list">
            {snapshot.subscriptions.map((sub) => (
              <LambdaRow key={sub.sub_id} sub={sub} />
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function LambdaRow({ sub }: { sub: LambdaSubscriptionView }) {
  const ownerLabel = formatOwner(sub.owner);
  return (
    <li className="lambda-row" data-status={sub.status}>
      <div className="lambda-row-head">
        <span className="lambda-row-label">{sub.label || sub.sub_id}</span>
        <span className={`lambda-row-status lambda-${sub.status}`}>{sub.status}</span>
      </div>
      <div className="lambda-row-observes">
        <code>{sub.observes}</code>
      </div>
      <div className="lambda-row-meta">
        {ownerLabel && <span className="lambda-meta-chip">owner: {ownerLabel}</span>}
        <span className="lambda-meta-chip">seen {sub.seen_count}</span>
        <span className="lambda-meta-chip">fired {sub.fire_count}</span>
        {sub.composed_from.length > 0 && (
          <span className="lambda-meta-chip">⊕ {sub.composed_from.length}</span>
        )}
      </div>
    </li>
  );
}

/** Reduce one meta-plane event into the cached snapshot — never mutating in place
 * (returns a fresh object so React Query notifies subscribers). */
function applyMetaEvent(prev: LambdaSnapshot, env: LambdaMetaEnvelope): LambdaSnapshot {
  switch (env.kind) {
    case "connected":
    case "lambda_observation":
      // Handshake / advisory reaction — no change to the subscription list.
      return prev;
    case "lambda_registered": {
      const view = env.payload as LambdaSubscriptionView;
      if (!view || typeof view.sub_id !== "string") return prev;
      const without = prev.subscriptions.filter((s) => s.sub_id !== view.sub_id);
      return recount({ ...prev, subscriptions: [...without, view] });
    }
    case "lambda_fired": {
      const p = env.payload as LambdaFiredPayload;
      if (!p || typeof p.sub_id !== "string") return prev;
      return recount({
        ...prev,
        subscriptions: prev.subscriptions.map((s) =>
          s.sub_id === p.sub_id
            ? {
                ...s,
                fire_count: p.fire_count,
                seen_count: p.seen_count,
                last_event_kind: p.source_kind,
                last_fired_seq: p.source_seq,
              }
            : s,
        ),
      });
    }
    case "lambda_closed": {
      const p = env.payload as { sub_id?: unknown };
      if (!p || typeof p.sub_id !== "string") return prev;
      const subId = p.sub_id;
      return recount({
        ...prev,
        subscriptions: prev.subscriptions.map((s) =>
          s.sub_id === subId ? { ...s, status: "closed" as const } : s,
        ),
      });
    }
    default: {
      const _exhaustive: never = env.kind;
      return _exhaustive;
    }
  }
}

function recount(snap: LambdaSnapshot): LambdaSnapshot {
  return {
    ...snap,
    total: snap.subscriptions.length,
    active: snap.subscriptions.filter((s) => s.status === "active").length,
  };
}

function formatOwner(owner: Record<string, unknown>): string | null {
  const node = owner["node_id"] ?? owner["node"] ?? owner["run_id"] ?? owner["run"];
  if (typeof node === "string") return node;
  const keys = Object.keys(owner);
  return keys.length > 0 ? String(owner[keys[0] as string]) : null;
}
