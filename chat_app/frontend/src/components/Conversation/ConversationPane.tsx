import { useEffect, useRef, useState } from "react";
import type {
  ChatRecord,
  MissingSpecChoice,
  MissingSpecialistPending,
} from "../../api/types";
import { LambdaTab } from "../LambdaTab/LambdaTab";
import { MissingSpecialistPause } from "./MissingSpecialistPause";
import "./ConversationPane.css";

interface ConversationPaneProps {
  chat: ChatRecord | null;
  loading: boolean;
  busy: boolean;
  runError: string | null;
  canSend: boolean;
  onSend: (message: string, topic?: string) => void;
  onNewChat: () => void;
  /** The user request of the in-flight run for this chat, echoed optimistically
   * as the newest turn while the agent works (visible multi-turn flow). */
  pendingMessage: string | null;
  /** The missing-specialist CHOICE payload when the run paused (s10-a8), else null. */
  pendingResolution: MissingSpecialistPending | null;
  /** Resolve the active pause (sse_fallback / define_and_resume). */
  onResume: (choice: MissingSpecChoice, specName?: string) => void;
  /** True while a resume round-trip is in flight. */
  resuming: boolean;
  /** A resume error message, else null. */
  resumeError: string | null;
}

export function ConversationPane({
  chat,
  loading,
  busy,
  runError,
  canSend,
  onSend,
  onNewChat,
  pendingMessage,
  pendingResolution,
  onResume,
  resuming,
  resumeError,
}: ConversationPaneProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const turnCount = chat?.turns.length ?? 0;

  // Keep the latest turn / thinking indicator / resolution surface in view as the
  // conversation grows.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turnCount, busy, pendingResolution, resuming]);

  if (!chat && !loading) {
    return (
      <div className="conversation conversation-empty">
        <div className="empty-card">
          <div className="empty-logo" aria-hidden="true">
            λ
          </div>
          <h1>Start a task</h1>
          <p>
            Open a chat and describe a task. The agent plans a DAG, runs it live, and
            surfaces artifacts here — without ever freezing the app.
          </p>
          <button type="button" className="empty-cta" onClick={onNewChat}>
            New chat
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="conversation">
      <div className="conversation-scroll" ref={scrollRef}>
        {loading && <p className="conversation-hint">Loading conversation…</p>}
        {chat?.turns.map((turn) => (
          <div className="turn" key={`${turn.chat_id}-${turn.turn_index}`}>
            <div className="bubble bubble-user">
              <div className="bubble-role">You</div>
              <div className="bubble-body">{turn.user_request}</div>
            </div>
            <div className="bubble bubble-agent">
              <div className="bubble-role">Agent</div>
              <div className="bubble-body">{turn.final_response || <em>(no text response — see artifacts)</em>}</div>
            </div>
          </div>
        ))}
        {busy && pendingMessage && (
          <div className="turn" key="pending-turn">
            <div className="bubble bubble-user">
              <div className="bubble-role">You</div>
              <div className="bubble-body">{pendingMessage}</div>
            </div>
          </div>
        )}
        {busy && (
          <div className="bubble bubble-agent thinking" aria-live="polite">
            <div className="bubble-role">Agent</div>
            <div className="bubble-body">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
              <span className="thinking-text">running the plan…</span>
            </div>
          </div>
        )}
        {pendingResolution && (
          <MissingSpecialistPause
            pending={pendingResolution}
            resuming={resuming}
            resumeError={resumeError}
            onResume={onResume}
          />
        )}
        {runError && <p className="conversation-error" role="alert">Run error: {runError}</p>}
      </div>

      <MessageInput canSend={canSend} busy={busy} onSend={onSend} />
    </div>
  );
}

/** The chat-input header: the read-only lambda tab sits here (d15), directly
 * above the composer, collapsible so it never crowds the input. */
function MessageInput({
  canSend,
  busy,
  onSend,
}: {
  canSend: boolean;
  busy: boolean;
  onSend: (message: string, topic?: string) => void;
}) {
  const [value, setValue] = useState("");
  const [lambdaOpen, setLambdaOpen] = useState(false);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || !canSend) return;
    onSend(trimmed);
    setValue("");
  };

  return (
    <div className="composer-shell">
      <LambdaTab open={lambdaOpen} onToggle={() => setLambdaOpen((v) => !v)} />
      <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <textarea
        className="composer-input"
        placeholder={canSend ? "Describe a task…" : busy ? "Run in progress…" : "Open a chat to start"}
        value={value}
        rows={1}
        aria-label="Message"
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
      />
      <button type="submit" className="composer-send" disabled={!canSend || value.trim() === ""}>
        {busy ? "Running…" : "Send"}
      </button>
      </form>
    </div>
  );
}
