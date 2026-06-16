/**
 * MISSING-SPECIALIST RESOLUTION SURFACE (s10-a8, closes d15 scenario-3 "via the
 * UI" + o3 "surface the choice to the user").
 *
 * When a run PAUSES because the plan needs a specialist no registered spec
 * provides, the backend returns `missing_specialist=true` with a `pending` CHOICE
 * payload (resume_token + the unmet node(s)). This surface renders that choice in
 * the conversation and wires each resolution back to POST /chats/{id}/resume:
 *
 *  - "Continue anyway" → `sse_fallback`: run the unmet node(s) spec-less and
 *    stream a best-effort answer.
 *  - "Define a specialist" → `define_and_resume`: define + register the missing
 *    specialization RIGHT HERE (reusing the real spec-authoring chat —
 *    SpecConversationHost — not a clone), then resume the SAME paused plan with
 *    the freshly-registered spec stamped onto the node(s).
 *
 * The user can never reach a dead-end pause: every pause is actionable.
 */
import { useState } from "react";
import { useOpenSpecChat } from "../../api/queries";
import type { MissingSpecChoice, MissingSpecialistPending } from "../../api/types";
import { SpecConversationHost } from "../SpecChat/SpecChatSurface";
import "./MissingSpecialistPause.css";

interface MissingSpecialistPauseProps {
  pending: MissingSpecialistPending;
  resuming: boolean;
  resumeError: string | null;
  onResume: (choice: MissingSpecChoice, specName?: string) => void;
}

export function MissingSpecialistPause({
  pending,
  resuming,
  resumeError,
  onResume,
}: MissingSpecialistPauseProps) {
  // `choices` is the offered set; we render a button per resolution but switch the
  // "Define a specialist" button into the inline authoring flow when picked.
  const [defining, setDefining] = useState(false);

  // The free-text descriptor(s) of what's needed — surfaced so the user knows what
  // capability to define or knowingly waive. Distinct values only.
  const needs = Array.from(
    new Set(pending.missing.map((m) => m.needs).filter((n) => n.trim() !== "")),
  );

  return (
    <div className="ms-pause" role="group" aria-label="Resolve missing specialist">
      <div className="ms-pause-head">
        <span className="ms-pause-badge" aria-hidden="true">
          ⏸
        </span>
        <div>
          <h2 className="ms-pause-title">A needed specialist isn’t available</h2>
          <p className="ms-pause-sub">
            {needs.length > 0 ? (
              <>
                This request asked for{" "}
                <strong>{needs.join(", ")}</strong>, which isn’t registered yet.
                Pick how to continue — nothing ran, so you lose no progress.
              </>
            ) : (
              <>
                The plan needs a specialist no registered specialization provides.
                Pick how to continue.
              </>
            )}
          </p>
        </div>
      </div>

      {pending.missing.length > 0 && (
        <ul className="ms-pause-nodes">
          {pending.missing.map((m) => (
            <li key={m.node_id} className="ms-pause-node">
              <span className="ms-pause-node-task">{m.task || m.node_id}</span>
              <span className="ms-pause-node-needs">needs: {m.needs}</span>
            </li>
          ))}
        </ul>
      )}

      {!defining ? (
        <div className="ms-pause-actions">
          {pending.choices.includes("sse_fallback") && (
            <button
              type="button"
              className="ms-pause-secondary"
              onClick={() => onResume("sse_fallback")}
              disabled={resuming}
            >
              {resuming ? "Resuming…" : "Continue anyway (stream a best-effort answer)"}
            </button>
          )}
          {pending.choices.includes("define_and_resume") && (
            <button
              type="button"
              className="ms-pause-primary"
              onClick={() => setDefining(true)}
              disabled={resuming}
            >
              Define a specialist
            </button>
          )}
        </div>
      ) : (
        <DefineAndResume
          resuming={resuming}
          onCancel={() => setDefining(false)}
          onApprovedResume={(specName) => onResume("define_and_resume", specName)}
        />
      )}

      {resumeError && (
        <p className="ms-pause-error" role="alert">
          Resume failed: {resumeError}
        </p>
      )}
    </div>
  );
}

/**
 * The inline "Define a specialist" path. Step 1 names the specialization and
 * opens a real spec-authoring session; step 2 hands off to the SAME
 * SpecConversationHost the dedicated Specializations screen uses (define →
 * critique → approve & register). When it registers, we resume the paused plan
 * with that spec stamped on (define_and_resume), so the user defines and
 * continues without leaving the conversation.
 */
function DefineAndResume({
  resuming,
  onCancel,
  onApprovedResume,
}: {
  resuming: boolean;
  onCancel: () => void;
  onApprovedResume: (specName: string) => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [session, setSession] = useState<{ id: string; name: string } | null>(null);
  const [resumed, setResumed] = useState(false);
  const openChat = useOpenSpecChat();

  const open = () => {
    const trimmed = name.trim();
    if (!trimmed || openChat.isPending) return;
    openChat.mutate(
      { name: trimmed, ...(description.trim() ? { description: description.trim() } : {}) },
      { onSuccess: (resp) => setSession({ id: resp.session_id, name: trimmed }) },
    );
  };

  // Once the spec registers, resume the paused plan with it — exactly once.
  const handleApproved = () => {
    if (!session || resumed) return;
    setResumed(true);
    onApprovedResume(session.name);
  };

  if (!session) {
    return (
      <form
        className="ms-define"
        onSubmit={(e) => {
          e.preventDefault();
          open();
        }}
      >
        <p className="ms-define-lead">
          Name the specialization, then author its ruleset. When you approve it, the
          paused plan resumes with this specialist applied.
        </p>
        <label className="ms-define-field">
          <span className="ms-define-label">Specialization name</span>
          <input
            className="ms-define-input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. forensic-accountant"
            maxLength={120}
            autoFocus
            required
          />
        </label>
        <label className="ms-define-field">
          <span className="ms-define-label">Description (optional)</span>
          <input
            className="ms-define-input"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="What this specialist is for"
            maxLength={400}
          />
        </label>
        {openChat.isError && (
          <p className="ms-pause-error" role="alert">{openChat.error.message}</p>
        )}
        <div className="ms-pause-actions">
          <button type="button" className="ms-pause-ghost" onClick={onCancel}>
            ← Back to choices
          </button>
          <button
            type="submit"
            className="ms-pause-primary"
            disabled={openChat.isPending || name.trim() === ""}
          >
            {openChat.isPending ? "Opening…" : "Start authoring"}
          </button>
        </div>
      </form>
    );
  }

  return (
    <div className="ms-define-host">
      {resumed && (
        <p className="ms-pause-banner" role="status">
          {resuming
            ? `✓ “${session.name}” registered — resuming the plan with it…`
            : `✓ “${session.name}” registered and applied — the resumed answer is below.`}
        </p>
      )}
      <SpecConversationHost
        sessionId={session.id}
        onReset={onCancel}
        onApproved={handleApproved}
      />
    </div>
  );
}
