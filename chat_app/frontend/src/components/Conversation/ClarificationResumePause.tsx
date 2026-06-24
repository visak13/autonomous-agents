/**
 * AMBIGUITY-CLARIFICATION RESUME SURFACE (scenario-2, the d5 loop fix).
 *
 * When a run PAUSES because the planner judged the request too underspecified and
 * asked ONE clarifying question, the backend returns `needs_clarification=true`
 * with a `pending` CLARIFICATION payload (resume_token + question). This surface
 * renders the question and an answer box, and wires the answer back to
 * POST /chats/{id}/resume — which re-drives the SAME plan on the clarified intent.
 *
 * THE LOOP FIX: the answer goes to /resume (carrying the resume_token), NOT a
 * fresh /message. Re-sending it as a new message is exactly what made the
 * ambiguity gate re-ask the same question every turn (the o2/d5 loop). Here the
 * already-given answer reaches the paused run, which resolves instead of re-asking.
 */
import { useState } from "react";
import type { ClarificationPending } from "../../api/types";
import "./MissingSpecialistPause.css";

interface ClarificationResumePauseProps {
  pending: ClarificationPending;
  resuming: boolean;
  resumeError: string | null;
  onResolve: (answer: string) => void;
}

export function ClarificationResumePause({
  pending,
  resuming,
  resumeError,
  onResolve,
}: ClarificationResumePauseProps) {
  const [answer, setAnswer] = useState("");

  const submit = () => {
    const trimmed = answer.trim();
    if (!trimmed || resuming) return;
    onResolve(trimmed);
  };

  return (
    <div className="ms-pause" role="group" aria-label="Answer the clarifying question">
      <div className="ms-pause-head">
        <span className="ms-pause-badge" aria-hidden="true">
          ⏸
        </span>
        <div>
          <h2 className="ms-pause-title">One quick clarification</h2>
          <p className="ms-pause-sub">
            The agent needs a detail before it plans. Answer below — your reply
            resumes this same task (nothing ran yet, so you lose no progress).
          </p>
        </div>
      </div>

      <p className="ms-pause-question">{pending.question}</p>

      <form
        className="ms-clarify"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <textarea
          className="ms-clarify-input"
          value={answer}
          rows={3}
          aria-label="Your answer"
          placeholder="Type your answer…"
          maxLength={4000}
          autoFocus
          disabled={resuming}
          onChange={(e) => setAnswer(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <div className="ms-pause-actions">
          <button
            type="submit"
            className="ms-pause-primary"
            disabled={resuming || answer.trim() === ""}
          >
            {resuming ? "Resuming…" : "Answer & continue"}
          </button>
        </div>
      </form>

      {resumeError && (
        <p className="ms-pause-error" role="alert">
          Resume failed: {resumeError}
        </p>
      )}
    </div>
  );
}
