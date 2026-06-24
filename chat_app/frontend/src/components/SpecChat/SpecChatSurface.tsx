/**
 * (c) INTERACTIVE SPEC-DEFINITION CHAT SURFACE (s7/a2, feature e; d11)
 *     + (s4/RC7) RE-EDITABLE specialization surface.
 *
 * Two jobs share this one workspace view, behind a small mode machine:
 *
 *  - DEFINE A NEW SPEC (the original a2 flow): state intent (turn 1 authors
 *    draft 1), read the drafted ruleset + its EXACT compiled markdown, critique
 *    it (each later turn RE-DRAFTS the body), then approve to compile + register
 *    a planner-loadable spec — or discard.
 *  - RE-OPEN AN EXISTING SPEC to edit (s4/RC7): the landing lists every
 *    registered spec; picking one opens an EDITOR that loads the full persisted
 *    body + provenance, edits it, and SAVES through the a2 PUT API so the edit
 *    PERSISTS and is EFFECTIVE ON THE NEXT RUN (d10 — persisted through the
 *    runtime's authoritative SpecRegistry, not SQLite). The same re-opened spec
 *    can alternatively be REFINED VIA CHAT (POST /spec-chats/reopen), which
 *    seeds the existing body into the SAME SpecConversation surface used to
 *    author new specs (component reuse, not a clone).
 *
 * Server state (the transcript, the registered index, a spec's persisted body)
 * lives in the Query cache; the only local state is which mode is open, the
 * draft input, and a re-open editor's working text. Lifecycle state is a
 * discriminated union consumed exhaustively (spec [required]).
 */
import { useEffect, useRef, useState } from "react";
import {
  useApproveSpecChat,
  useDeleteRegisteredSpec,
  useDenySpecChat,
  useOpenSpecChat,
  useRegisteredSpec,
  useRegisteredSpecs,
  useReopenSpecChat,
  useSendSpecChatMessage,
  useSpecChat,
  useUpdateRegisteredSpec,
} from "../../api/queries";
import type {
  RegisteredSpec,
  RegisteredSpecRow,
  SpecChatState,
  SpecDraft,
  SpecTurn,
} from "../../api/types";
import "./SpecChatSurface.css";

interface SpecChatSurfaceProps {
  onClose: () => void;
}

/**
 * Which sub-surface is showing. `landing` is the picker (define-new + the
 * re-open list); `creating`/`refining` both render the SpecConversation over a
 * session id (refining is seeded from an existing spec); `editing` is the
 * direct-PUT editor for a registered spec.
 */
type Mode =
  | { kind: "landing" }
  | { kind: "creating"; sessionId: string }
  | { kind: "refining"; sessionId: string; name: string }
  | { kind: "editing"; name: string };

export function SpecChatSurface({ onClose }: SpecChatSurfaceProps) {
  const [mode, setMode] = useState<Mode>({ kind: "landing" });
  const toLanding = () => setMode({ kind: "landing" });

  const subtitle =
    mode.kind === "editing"
      ? "Re-open a registered specialization to view and edit it — saved edits take effect on the next run."
      : mode.kind === "refining"
        ? "Refine an existing ruleset by chat — approve to re-register it under the same name."
        : "Chat a ruleset into shape (or re-open an existing one), then register it for the planner to load.";

  return (
    <section className="spec-surface" aria-label="Specializations">
      <header className="spec-surface-head">
        <div className="spec-surface-titles">
          <h1 className="spec-surface-title">Specializations</h1>
          <p className="spec-surface-sub">{subtitle}</p>
        </div>
        <button
          type="button"
          className="spec-surface-back"
          onClick={mode.kind === "landing" ? onClose : toLanding}
        >
          {mode.kind === "landing" ? "← Back to tasks" : "← All specializations"}
        </button>
      </header>

      {mode.kind === "landing" && (
        <SpecLanding
          onCreated={(sessionId) => setMode({ kind: "creating", sessionId })}
          onEdit={(name) => setMode({ kind: "editing", name })}
        />
      )}

      {(mode.kind === "creating" || mode.kind === "refining") && (
        <SpecConversationHost sessionId={mode.sessionId} onReset={toLanding} />
      )}

      {mode.kind === "editing" && (
        <SpecEditor
          name={mode.name}
          onBack={toLanding}
          onRefineInChat={(sessionId, name) =>
            setMode({ kind: "refining", sessionId, name })
          }
        />
      )}
    </section>
  );
}

// =========================================================================== //
// landing: define-a-new-spec form + the re-open-an-existing list (RC7)
// =========================================================================== //
function SpecLanding({
  onCreated,
  onEdit,
}: {
  onCreated: (sessionId: string) => void;
  onEdit: (name: string) => void;
}) {
  return (
    <div className="spec-landing">
      <OpenSessionForm onOpened={onCreated} />
      <RegisteredSpecList onEdit={onEdit} />
    </div>
  );
}

function OpenSessionForm({ onOpened }: { onOpened: (id: string) => void }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const openChat = useOpenSpecChat();

  const submit = () => {
    const trimmed = name.trim();
    if (!trimmed || openChat.isPending) return;
    openChat.mutate(
      { name: trimmed, ...(description.trim() ? { description: description.trim() } : {}) },
      { onSuccess: (resp) => onOpened(resp.session_id) },
    );
  };

  return (
    <form
      className="spec-open"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <h2 className="spec-open-title">Define a new specialization</h2>
      <label className="spec-field">
        <span className="spec-field-label">Spec name</span>
        <input
          className="spec-field-input"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. concise-markdown-brief"
          maxLength={120}
          required
        />
      </label>
      <label className="spec-field">
        <span className="spec-field-label">Description (optional)</span>
        <input
          className="spec-field-input"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="What the planner looks this spec up by"
          maxLength={400}
        />
      </label>
      {openChat.isError && (
        <p className="spec-error" role="alert">{openChat.error.message}</p>
      )}
      <button type="submit" className="spec-primary" disabled={openChat.isPending || name.trim() === ""}>
        {openChat.isPending ? "Opening…" : "Start authoring"}
      </button>
    </form>
  );
}

/** The "pick an existing spec to re-open" list (GET /spec-chats/registered). */
function RegisteredSpecList({ onEdit }: { onEdit: (name: string) => void }) {
  const specs = useRegisteredSpecs();

  return (
    <section className="spec-registered" aria-label="Registered specializations">
      <h2 className="spec-open-title">Re-open an existing specialization</h2>
      {specs.isLoading && <p className="spec-hint">Loading registered specs…</p>}
      {specs.isError && (
        <p className="spec-error" role="alert">{specs.error.message}</p>
      )}
      {specs.data && specs.data.length === 0 && (
        <p className="spec-hint">
          No specializations registered yet. Define one above — it will appear here to re-open and edit.
        </p>
      )}
      {specs.data && specs.data.length > 0 && (
        <ul className="spec-reg-list">
          {specs.data.map((row: RegisteredSpecRow) => (
            <li key={row.name} className="spec-reg-item">
              <button
                type="button"
                className="spec-reg-row"
                onClick={() => onEdit(row.name)}
                aria-label={`Re-open ${row.name}`}
              >
                <span className="spec-reg-name">{row.name}</span>
                {row.description && (
                  <span className="spec-reg-desc">{row.description}</span>
                )}
                <span className="spec-reg-source">{row.source}</span>
              </button>
              <DeleteSpecButton name={row.name} />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/**
 * Per-row DELETE control for a registered specialization (s4/a4, d13 UI). A REAL
 * hit-testable button (its own row sibling — never nested inside the re-open
 * button, which would be invalid + un-clickable). Click → confirm → delete; the
 * mutation invalidates the index so the row clears. Double-delete is guarded by
 * disabling while the mutation is pending. Any registered spec is deletable
 * (specs have no built-ins). */
function DeleteSpecButton({ name }: { name: string }) {
  const del = useDeleteRegisteredSpec();
  const onDelete = () => {
    if (del.isPending) return;
    if (!window.confirm(`Delete specialization "${name}"? This cannot be undone.`)) {
      return;
    }
    del.mutate({ name });
  };
  return (
    <span className="spec-reg-delete-wrap">
      <button
        type="button"
        className="spec-reg-delete"
        onClick={onDelete}
        disabled={del.isPending}
        aria-label={`Delete ${name}`}
        title={`Delete ${name}`}
      >
        {del.isPending ? "Deleting…" : "Delete"}
      </button>
      {del.isError && (
        // A failed delete is shown plainly, never a silent no-op (e.g. a 404 if the
        // spec vanished concurrently).
        <span className="spec-reg-delete-err" role="alert">
          {del.error.message}
        </span>
      )}
    </span>
  );
}

// =========================================================================== //
// re-open EDITOR: load the full persisted spec, edit body/description, SAVE via
// the a2 PUT API (persists + effective next run). Also offers "Refine in chat".
// =========================================================================== //
function SpecEditor({
  name,
  onBack,
  onRefineInChat,
}: {
  name: string;
  onBack: () => void;
  onRefineInChat: (sessionId: string, name: string) => void;
}) {
  const spec = useRegisteredSpec(name);
  const reopen = useReopenSpecChat();

  const refineInChat = () => {
    if (reopen.isPending) return;
    reopen.mutate(
      { name },
      { onSuccess: (view) => onRefineInChat(view.session_id, name) },
    );
  };

  return (
    <div className="spec-editor">
      {spec.isLoading && <p className="spec-hint">Loading {name}…</p>}
      {spec.isError && (
        <div className="spec-editor-error">
          <p className="spec-error" role="alert">{spec.error.message}</p>
          <button type="button" className="spec-secondary" onClick={onBack}>
            ← Back
          </button>
        </div>
      )}
      {spec.data && (
        // Re-mount the form when the loaded spec identity changes so its local
        // working text is seeded ONCE from the persisted body (no server→state
        // mirroring effect — the React-idiomatic "reset via key").
        <SpecEditForm
          key={`${spec.data.name}@${spec.data.created_at}`}
          loaded={spec.data}
          onRefineInChat={reopen.isPending ? undefined : refineInChat}
          reopenError={reopen.isError ? reopen.error.message : null}
        />
      )}
    </div>
  );
}

function SpecEditForm({
  loaded,
  onRefineInChat,
  reopenError,
}: {
  loaded: RegisteredSpec;
  onRefineInChat?: (() => void) | undefined;
  reopenError: string | null;
}) {
  const [description, setDescription] = useState(loaded.description);
  const [body, setBody] = useState(loaded.body);
  const update = useUpdateRegisteredSpec();
  const [savedAt, setSavedAt] = useState<number | null>(null);

  // The fields the user has actually changed vs. the last-persisted values.
  const descChanged = description !== loaded.description;
  const bodyChanged = body !== loaded.body;
  const dirty = descChanged || bodyChanged;
  const bodyBlank = body.trim() === "";

  const save = () => {
    if (!dirty || update.isPending || bodyBlank) return;
    const edit: { description?: string; body?: string } = {};
    if (descChanged) edit.description = description;
    if (bodyChanged) edit.body = body;
    update.mutate(
      { name: loaded.name, edit },
      { onSuccess: () => setSavedAt(Date.now()) },
    );
  };

  return (
    <div className="spec-editor-body">
      <header className="spec-editor-head">
        <div>
          <h2 className="spec-draft-name">{loaded.name}</h2>
          <p className="spec-editor-meta">
            source: {loaded.source} · created {formatTs(loaded.created_at)}
          </p>
        </div>
        {onRefineInChat && (
          <button
            type="button"
            className="spec-secondary"
            onClick={onRefineInChat}
            title="Re-open this ruleset in the authoring chat to refine it conversationally"
          >
            Refine in chat ↗
          </button>
        )}
      </header>

      <label className="spec-field">
        <span className="spec-field-label">Description</span>
        <input
          className="spec-field-input"
          value={description}
          onChange={(e) => {
            setDescription(e.target.value);
            setSavedAt(null);
          }}
          placeholder="What the planner looks this spec up by"
          maxLength={400}
          aria-label="Spec description"
        />
      </label>

      <label className="spec-field spec-field-grow">
        <span className="spec-field-label">Ruleset body</span>
        <textarea
          className="spec-editor-textarea"
          value={body}
          onChange={(e) => {
            setBody(e.target.value);
            setSavedAt(null);
          }}
          aria-label="Spec body"
          spellCheck={false}
        />
      </label>

      {bodyBlank && (
        <p className="spec-error" role="alert">The ruleset body can’t be empty.</p>
      )}
      {update.isError && (
        <p className="spec-error" role="alert">{update.error.message}</p>
      )}
      {reopenError && <p className="spec-error" role="alert">{reopenError}</p>}
      {savedAt !== null && !dirty && (
        <p className="spec-banner spec-banner-ok" role="status">
          ✓ Saved — persisted to the spec registry; effective on the next run.
        </p>
      )}

      <div className="spec-editor-actions">
        <button
          type="button"
          className="spec-primary"
          onClick={save}
          disabled={!dirty || update.isPending || bodyBlank}
        >
          {update.isPending ? "Saving…" : "Save changes"}
        </button>
        <span className="spec-editor-dirty" aria-live="polite">
          {dirty ? "Unsaved changes" : "Up to date"}
        </span>
      </div>
    </div>
  );
}

function formatTs(iso: string): string {
  // created_at is an ISO string from the registry; show a compact local form,
  // tolerating a non-parseable value rather than throwing.
  const t = Date.parse(iso);
  return Number.isNaN(t) ? iso : new Date(t).toLocaleString();
}

// =========================================================================== //
// SpecConversation host — drives one session (create OR re-opened refine). The
// transcript is server state in the cache (useSpecChat); a re-opened session is
// primed into that cache by useReopenSpecChat so it renders the seeded draft.
// =========================================================================== //
export function SpecConversationHost({
  sessionId,
  onReset,
  onApproved,
}: {
  sessionId: string;
  onReset: () => void;
  /** Fired once when the session reaches `approved` (the spec is compiled +
   * registered). Lets an embedding surface — e.g. the missing-specialist
   * resolution flow — resume the paused plan with the freshly-defined spec. */
  onApproved?: () => void;
}) {
  const view = useSpecChat(sessionId);
  const state = view.data?.state ?? "open";

  // Notify the embedder exactly once when the spec becomes registered.
  const notifiedRef = useRef(false);
  useEffect(() => {
    if (state === "approved" && !notifiedRef.current) {
      notifiedRef.current = true;
      onApproved?.();
    }
  }, [state, onApproved]);

  return (
    <SpecConversation
      sessionId={sessionId}
      state={state}
      started={view.data?.started ?? false}
      turns={view.data?.turns ?? []}
      draft={view.data?.draft ?? null}
      loading={view.isLoading}
      onReset={onReset}
    />
  );
}

function SpecConversation({
  sessionId,
  state,
  started,
  turns,
  draft,
  loading,
  onReset,
}: {
  sessionId: string;
  state: SpecChatState;
  started: boolean;
  turns: SpecTurn[];
  draft: SpecDraft | null;
  loading: boolean;
  onReset: () => void;
}) {
  const [message, setMessage] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const sendMessage = useSendSpecChatMessage();
  const approve = useApproveSpecChat();
  const deny = useDenySpecChat();

  const terminal = state === "approved" || state === "denied" || state === "cancelled";
  const busy = sendMessage.isPending || approve.isPending || deny.isPending;

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns.length, busy]);

  const submit = () => {
    const trimmed = message.trim();
    if (!trimmed || busy || terminal) return;
    sendMessage.mutate({ sessionId, message: trimmed });
    setMessage("");
  };

  const placeholder = started
    ? "Critique the draft — it re-authors against your feedback…"
    : "State what this spec should do (this authors the first draft)…";

  return (
    <div className="spec-convo">
      <div className="spec-convo-main">
        <div className="spec-transcript" ref={scrollRef}>
          {loading && turns.length === 0 && <p className="spec-hint">Loading…</p>}
          {turns.length === 0 && !loading && (
            <p className="spec-hint">
              No turns yet. Your first message states the intent and authors draft 1.
            </p>
          )}
          {turns.map((turn, i) => (
            <div className={`spec-bubble spec-${turn.role}`} key={`${turn.role}-${i}`}>
              <div className="spec-bubble-role">{turn.role === "user" ? "You" : "Author"}</div>
              <div className="spec-bubble-body">{turn.text}</div>
            </div>
          ))}
          {busy && !terminal && (
            <div className="spec-bubble spec-agent thinking" aria-live="polite">
              <div className="spec-bubble-role">Author</div>
              <div className="spec-bubble-body">re-drafting the ruleset…</div>
            </div>
          )}
          {sendMessage.isError && (
            <p className="spec-error" role="alert">{sendMessage.error.message}</p>
          )}
        </div>

        <StateBanner state={state} />

        {!terminal && (
          <form
            className="spec-composer"
            onSubmit={(e) => {
              e.preventDefault();
              submit();
            }}
          >
            <textarea
              className="spec-composer-input"
              value={message}
              rows={2}
              placeholder={placeholder}
              aria-label="Spec message"
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit();
                }
              }}
            />
            <button type="submit" className="spec-primary" disabled={busy || message.trim() === ""}>
              {sendMessage.isPending ? "Sending…" : started ? "Refine" : "Author"}
            </button>
          </form>
        )}
        {terminal && (
          <button type="button" className="spec-secondary" onClick={onReset}>
            ← All specializations
          </button>
        )}
      </div>

      <aside className="spec-draft" aria-label="Working draft preview">
        <DraftPreview draft={draft} />
        {draft && !terminal && (
          <div className="spec-draft-actions">
            <button
              type="button"
              className="spec-primary"
              onClick={() => approve.mutate({ sessionId })}
              disabled={busy}
            >
              {approve.isPending ? "Compiling…" : "Approve & register"}
            </button>
            <button
              type="button"
              className="spec-danger"
              onClick={() => deny.mutate({ sessionId })}
              disabled={busy}
            >
              {deny.isPending ? "Discarding…" : "Discard"}
            </button>
          </div>
        )}
        {approve.isError && <p className="spec-error" role="alert">{approve.error.message}</p>}
        {deny.isError && <p className="spec-error" role="alert">{deny.error.message}</p>}
      </aside>
    </div>
  );
}

function DraftPreview({ draft }: { draft: SpecDraft | null }) {
  if (!draft) {
    return (
      <div className="spec-draft-empty">
        The compiled ruleset preview appears here after your first message.
      </div>
    );
  }
  return (
    <div className="spec-draft-body">
      <header className="spec-draft-head">
        <h2 className="spec-draft-name">{draft.name}</h2>
        <span className="spec-draft-round">round {draft.turn}</span>
      </header>
      {draft.description && <p className="spec-draft-desc">{draft.description}</p>}
      <h3 className="spec-draft-section">Compiled spec</h3>
      <pre className="spec-draft-markdown">{draft.markdown}</pre>
    </div>
  );
}

function StateBanner({ state }: { state: SpecChatState }) {
  switch (state) {
    case "open":
      return null;
    case "approved":
      return (
        <p className="spec-banner spec-banner-ok" role="status">
          ✓ Approved — the spec is compiled and registered. The planner can now load it.
        </p>
      );
    case "denied":
      return (
        <p className="spec-banner spec-banner-muted" role="status">
          Discarded — nothing was compiled.
        </p>
      );
    case "cancelled":
      return (
        <p className="spec-banner spec-banner-muted" role="status">
          Cancelled.
        </p>
      );
    default: {
      const _exhaustive: never = state;
      return _exhaustive;
    }
  }
}
