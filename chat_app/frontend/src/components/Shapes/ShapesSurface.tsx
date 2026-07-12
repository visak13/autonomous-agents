/**
 * (s4/a5, d5/d9) DEDICATED SHAPES SCREEN.
 *
 * Its OWN screen — distinct from the specialization chat. It MIRRORS the spec
 * surface's look/feel/interaction QUALITY (the same design tokens, the same
 * header/back chrome, the same buttons + banners) so the app feels consistent,
 * but its FLOW + BEHAVIOR are GENUINE to the shapes implementation, NOT a clone
 * of the spec chat:
 *
 *  - the spec chat is a TRANSCRIPT + draft-preview conversation that AUTHORS a
 *    ruleset; this is a MASTER-DETAIL CATALOG that READS the text-file-defined
 *    plan shapes and surfaces what a shape REALLY is (s17 redesign, d247/d248):
 *    an execution DISCIPLINE + DOCTRINE. A shape declares NO fixed node topology
 *    — the PLANNER/GROWER AUTHORS the topology at runtime by reasoning — so the
 *    old round-by-round unroll preview is RETIRED; the detail pane renders the
 *    discipline, the declared phase flow, the doctrine text and (for
 *    deep-research) the growth safety bounds instead.
 *  - the only thing the user EDITS here is a shape's per-shape MAX_ITER override
 *    (d5), saved through the a4 backend (PUT /shapes/{name}/max_iter) so it
 *    persists to the shared SQLite — the deep-research grow loop's depth CEILING
 *    (a non-deciding safety net, d240: the model's stop_research decides).
 *
 * Server state (the shape catalog, a shape's view) is the Query cache; the only
 * local state is which shape is selected and the editor's working value. The
 * execution discipline is a discriminated union consumed exhaustively (spec
 * [required]) so a new backend discipline is a compile error, never a silent
 * fall-through.
 */
import { useState } from "react";
import {
  useApproveShapeChat,
  useDeleteShape,
  useDenyShapeChat,
  useOpenShapeChat,
  useSendShapeChatMessage,
  useSetShapeMaxIter,
  useShapes,
} from "../../api/queries";
import type { ShapeChatView, ShapeExecution, ShapeView } from "../../api/types";
import "./ShapesSurface.css";

interface ShapesSurfaceProps {
  onClose: () => void;
}

/**
 * The 6 shipped BUILT-IN shapes (s4/a4, d13 no-regression).
 *
 * SOURCE OF TRUTH: chat_app/chat_app/shape_config.py:62-71 (`BUILTIN_SHAPES`).
 * This is a deliberate MIRROR of that backend guard — the backend DELETE route
 * 409s these names (only the user's OWN authored shapes are deletable), which is
 * the load-bearing safety floor; this frontend copy only HIDES the delete control
 * cosmetically. If a built-in is ever added/removed, update BOTH this set and the
 * backend set above (keep them in lockstep). Built-ins and user-authored shapes
 * share the same SHAPES_DIR, so `ShapeView.source` (an identical-shaped path for
 * both) can't distinguish them — the NAME is the reliable detector. A 409 is still
 * handled gracefully (visible error) as the real safety net if a built-in delete
 * is ever attempted. */
const BUILTIN_SHAPES: ReadonlySet<string> = new Set([
  "linear",
  "modular-parallel",
  "concurrent-multi-topic-gathering",
  "deep-research",
  "iterative-deep-research",
  "iterative-writing-improvement",
]);

export function ShapesSurface({ onClose }: ShapesSurfaceProps) {
  const shapes = useShapes();
  const [selected, setSelected] = useState<string | null>(null);
  // s17 (d18a parity): which shape a CONVERSATIONAL refine session targets (null =
  // the chat panel drafts a NEW shape). Keyed into the panel so switching target
  // cleanly restarts the conversation.
  const [refineTarget, setRefineTarget] = useState<string | null>(null);

  // Auto-select the first shape once the catalog loads so the detail pane is
  // never empty on entry (derived from server data, not mirrored into an effect).
  const list = shapes.data ?? [];
  const activeName =
    selected && list.some((s) => s.name === selected)
      ? selected
      : (list[0]?.name ?? null);
  const active = list.find((s) => s.name === activeName) ?? null;

  return (
    <section className="shapes-surface" aria-label="Shapes">
      <header className="shapes-surface-head">
        <div className="shapes-surface-titles">
          <h1 className="shapes-surface-title">Shapes</h1>
          <p className="shapes-surface-sub">
            The plan shapes the planner selects per query — defined in text files,
            with a per-shape iteration ceiling you can set here.
          </p>
        </div>
        <button type="button" className="shapes-surface-back" onClick={onClose}>
          ← Back to tasks
        </button>
      </header>
      <hr className="orn-divider orn-divider-feather" aria-hidden="true" />

      <div className="shapes-body">
        <div className="shapes-sidebar">
          <ShapeChatPanel
            key={refineTarget ?? "«create»"}
            refineOf={refineTarget}
            onAuthored={(name) => {
              setSelected(name);
              setRefineTarget(null);
            }}
            onCancelRefine={() => setRefineTarget(null)}
          />
          <nav className="shapes-list" aria-label="Plan shapes">
          {shapes.isLoading && <p className="shapes-hint">Loading shapes…</p>}
          {shapes.isError && (
            <p className="shapes-error" role="alert">{shapes.error.message}</p>
          )}
          {shapes.data && list.length === 0 && (
            <p className="shapes-hint">No text-file shapes are defined.</p>
          )}
          {list.map((shape) => (
            <div key={shape.name} className="shapes-row-item">
              <button
                type="button"
                className={`shapes-row${shape.name === activeName ? " shapes-row-active" : ""}`}
                onClick={() => setSelected(shape.name)}
                aria-current={shape.name === activeName}
                aria-label={`View ${shape.name}`}
              >
                <span className="shapes-row-name">{shape.name}</span>
                <ExecutionBadge execution={shape.execution} />
                <span className="shapes-row-iter">{iterSummary(shape)}</span>
              </button>
              {!BUILTIN_SHAPES.has(shape.name) && (
                <ShapeDeleteButton name={shape.name} />
              )}
            </div>
          ))}
          </nav>
        </div>

        <div className="shapes-detail">
          {active ? (
            <ShapeDetail
              key={active.name}
              shape={active}
              onRefineInChat={() => setRefineTarget(active.name)}
            />
          ) : (
            !shapes.isLoading && (
              <p className="shapes-hint shapes-detail-empty">
                Select a shape to view its structure.
              </p>
            )
          )}
        </div>
      </div>
    </section>
  );
}

// =========================================================================== //
// SHAPE CHAT (s17, d18a/d249 parity): the CONVERSATIONAL shape-authoring panel —
// the same free-flowing iterative flow the spec chat gives rulesets. Each message
// drives one live authoring turn over an IN-SESSION DRAFT (nothing touches disk
// mid-conversation); the draft preview updates every turn; Approve persists it
// through the backend's round-trip guard, Discard throws it away. Opened plain it
// DRAFTS A NEW shape; opened with `refineOf` it edits that shape conversationally
// (seeded from its real on-disk definition).
// =========================================================================== //
function ShapeChatPanel({
  refineOf,
  onAuthored,
  onCancelRefine,
}: {
  refineOf: string | null;
  onAuthored: (name: string) => void;
  onCancelRefine: () => void;
}) {
  const [message, setMessage] = useState("");
  // The session view is DRIVE output (every route returns the fresh view), held as
  // local state keyed by the panel's `key` — not a cached server query, because an
  // open draft is ephemeral by design (deny/restart drops it).
  const [view, setView] = useState<ShapeChatView | null>(null);
  const open = useOpenShapeChat();
  const send = useSendShapeChatMessage();
  const approve = useApproveShapeChat();
  const deny = useDenyShapeChat();
  const busy = open.isPending || send.isPending || approve.isPending;

  const submit = async () => {
    const text = message.trim();
    if (!text || busy) return;
    try {
      let sid = view?.session_id;
      if (!sid) {
        const opened = await open.mutateAsync({ refineOf: refineOf ?? undefined });
        sid = opened.session_id;
        setView(opened);
      }
      const next = await send.mutateAsync({ sessionId: sid, message: text });
      setView(next);
      setMessage("");
    } catch {
      // the mutation's own isError/error render below — nothing else to do
    }
  };

  const onApprove = () => {
    if (!view || busy) return;
    approve.mutate(
      { sessionId: view.session_id },
      {
        onSuccess: (res) => {
          setView(null);
          onAuthored(res.shape.name);
        },
      },
    );
  };

  const onDiscard = () => {
    if (view) deny.mutate({ sessionId: view.session_id });
    setView(null);
    setMessage("");
    if (refineOf) onCancelRefine();
  };

  const err = send.error ?? open.error ?? approve.error;

  return (
    <section className="shapes-author" aria-label="Shape chat">
      <h2 className="shapes-author-title">
        {refineOf ? `Refine “${refineOf}” in chat` : "Draft a new shape in chat"}
      </h2>
      <p className="shapes-author-hint">
        Talk the shape into existence — each message revises the draft below,
        building on the previous version. Nothing is saved until you approve.
      </p>

      {view && view.turns.length > 0 && (
        <ol className="shapes-chat-transcript" aria-label="Shape chat transcript">
          {view.turns.map((t, i) => (
            <li key={i} className={`shapes-chat-turn shapes-chat-${t.role}`}>
              {t.text}
            </li>
          ))}
        </ol>
      )}

      {view?.draft && (
        <div className="shapes-doctrine" aria-label="Current draft">
          <strong>Draft:</strong> {view.draft.name}{" "}
          <ExecutionBadge execution={view.draft.execution} /> —{" "}
          {view.draft.description}
        </div>
      )}

      <label className="shapes-field">
        <span className="shapes-field-label">
          {view?.draft ? "Revise the draft" : "Describe the shape"}
        </span>
        <textarea
          className="shapes-field-input shapes-author-input"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder={
            view?.draft
              ? "e.g. run the second phase in parallel instead of in sequence"
              : "e.g. a foundation phase, then independent avenues explored in parallel, then combine"
          }
          maxLength={4000}
          rows={3}
          aria-label="Shape chat message"
        />
      </label>
      {err && <p className="shapes-error" role="alert">{err.message}</p>}
      <div className="shapes-chat-actions">
        <button
          type="button"
          className="shapes-primary"
          onClick={() => void submit()}
          disabled={busy || message.trim() === ""}
        >
          {send.isPending || open.isPending ? "Drafting…" : "Send"}
        </button>
        {view?.draft && (
          <>
            <button
              type="button"
              className="shapes-primary"
              onClick={onApprove}
              disabled={busy}
            >
              {approve.isPending ? "Saving…" : "Approve & save"}
            </button>
            <button
              type="button"
              className="shapes-row-delete"
              onClick={onDiscard}
              disabled={busy}
            >
              Discard
            </button>
          </>
        )}
        {refineOf && !view?.draft && (
          <button type="button" className="shapes-row-delete" onClick={onDiscard}>
            Cancel
          </button>
        )}
      </div>
    </section>
  );
}

/**
 * Per-row DELETE control for a USER-AUTHORED shape (s4/a4, d13 UI). A REAL
 * hit-testable button rendered as a sibling of the row's view button (never
 * nested inside it — that would be invalid HTML + un-clickable). Only rendered for
 * non-built-in shapes (the caller gates on BUILTIN_SHAPES); a 409 from the backend
 * guard still surfaces gracefully via the mutation error as the real safety net.
 * Click → confirm → delete; the mutation invalidates the catalog so the row
 * clears. Double-delete is guarded by disabling while the mutation is pending. */
function ShapeDeleteButton({ name }: { name: string }) {
  const del = useDeleteShape();
  const onDelete = () => {
    if (del.isPending) return;
    if (!window.confirm(`Delete shape "${name}"? This cannot be undone.`)) {
      return;
    }
    del.mutate({ name });
  };
  return (
    <span className="shapes-row-delete-wrap">
      <button
        type="button"
        className="shapes-row-delete"
        onClick={onDelete}
        disabled={del.isPending}
        aria-label={`Delete ${name}`}
        title={`Delete ${name}`}
      >
        {del.isPending ? "Deleting…" : "Delete"}
      </button>
      {del.isError && (
        // A failed delete is shown plainly, never a silent no-op — e.g. a 409 if a
        // built-in were ever attempted, or a 404 if the shape vanished concurrently.
        <span className="shapes-row-delete-err" role="alert">
          {del.error.message}
        </span>
      )}
    </span>
  );
}

/** A short "ceiling" summary for the list row (genuine: only the deep-research
 * discipline consumes max_iter — as its grow-loop DEPTH ceiling). */
function iterSummary(shape: ShapeView): string {
  if (!isIterative(shape)) return "single pass";
  const over = shape.max_iter_override;
  return over != null && over !== shape.max_iter
    ? `depth ≤ ${shape.effective_max_iter} (set)`
    : `depth ≤ ${shape.effective_max_iter}`;
}

/** A shape iterates (consumes max_iter) iff its discipline is deep-research —
 * the grow loop honors max_iter as its depth ceiling. The dispatch-discipline
 * shapes (sequential/concurrent) run their authored nodes once. (s17: keyed off
 * the execution discipline — the retired round_roles topology no longer exists.) */
function isIterative(shape: ShapeView): boolean {
  return shape.execution === "deep-research";
}

// =========================================================================== //
// detail: the shape's real structure + the max_iter editor
// =========================================================================== //
function ShapeDetail({
  shape,
  onRefineInChat,
}: {
  shape: ShapeView;
  onRefineInChat: () => void;
}) {
  return (
    <article className="shapes-card">
      <header className="shapes-card-head">
        <div className="shapes-card-titles">
          <h2 className="shapes-card-name">{shape.name}</h2>
          <ExecutionBadge execution={shape.execution} />
        </div>
        <code className="shapes-card-source" title={shape.source}>
          {fileName(shape.source)}
        </code>
      </header>

      {shape.description && (
        <p className="shapes-card-desc">{shape.description}</p>
      )}

      <ShapeStructure shape={shape} />

      {/* s17 (d18a parity): refinement is CONVERSATIONAL — this hands the shape to
          the chat panel, which seeds a draft from its on-disk definition and only
          persists on approve (the one-shot rewrite-the-file form is retired). */}
      <section className="shapes-refine" aria-label="Refine this shape">
        <h3 className="shapes-struct-title">Refine this shape</h3>
        <p className="shapes-struct-note">
          Evolve it conversationally — each turn builds on the draft, and nothing is
          saved until you approve.
        </p>
        <button type="button" className="shapes-primary" onClick={onRefineInChat}>
          Refine in chat
        </button>
      </section>

      <MaxIterEditor shape={shape} />
    </article>
  );
}

// s17 (d18a parity): the ONE-SHOT refine form is RETIRED — refinement now runs
// through the conversational ShapeChatPanel (draft turns + approve gate). The
// backend /shapes/{name}/refine one-shot route remains for API callers.

// --------------------------------------------------------------------------- //
// structure — GENUINE per execution discipline (the heart of "not a copy-pasta")
// --------------------------------------------------------------------------- //
function ShapeStructure({ shape }: { shape: ShapeView }) {
  switch (shape.execution) {
    case "deep-research":
      return <DeepResearchStructure shape={shape} />;
    case "sequential":
      return <SequentialStructure />;
    case "concurrent":
      return <ConcurrentStructure />;
    default: {
      const _exhaustive: ShapeExecution = shape.execution;
      return _exhaustive;
    }
  }
}

/** Deep-research = discipline + doctrine (s17 redesign, d247/d248). There is NO
 * fixed round topology to preview — the planner/grower AUTHORS the research
 * topology at runtime by reasoning (decompose into facets, deepen on note gaps,
 * prune settled leads, stop when every concern is settled). What the shape
 * genuinely declares is its doctrine text + the growth SAFETY bounds, so that is
 * what this pane renders. */
function DeepResearchStructure({ shape }: { shape: ShapeView }) {
  return (
    <section className="shapes-struct" aria-label="Deep-research discipline">
      <h3 className="shapes-struct-title">Discipline &amp; doctrine</h3>
      <p className="shapes-struct-note">
        Iterative deepening research. The shape declares <em>no fixed topology</em>{" "}
        — the planner authors the research tree at runtime by reasoning: decompose
        the goal into facets, gather and take notes, expand on the gaps the notes
        leave, prune settled leads, and stop when every concern is settled. The
        model&apos;s own stop decision is primary; the bounds below are safety
        ceilings, not the plan.
      </p>

      <PhaseFlow phases={shape.phases} />

      {shape.decompose_methodology && (
        <p className="shapes-doctrine">
          <strong>Decompose:</strong> {shape.decompose_methodology}
        </p>
      )}
      {shape.completeness_stop && (
        <p className="shapes-doctrine">
          <strong>Stop when:</strong> {shape.completeness_stop}
        </p>
      )}

      <dl className="shapes-meta">
        <div className="shapes-meta-row">
          <dt>Depth ceiling</dt>
          <dd>{shape.effective_max_iter}</dd>
        </div>
        <div className="shapes-meta-row">
          <dt>File default</dt>
          <dd>{shape.max_iter}</dd>
        </div>
        <div className="shapes-meta-row">
          <dt>Hard cap</dt>
          <dd>{shape.hard_cap}</dd>
        </div>
        {shape.fan_out > 0 && (
          <div className="shapes-meta-row">
            <dt>Fan-out / layer</dt>
            <dd>{shape.fan_out}</dd>
          </div>
        )}
        {shape.max_layers > 0 && (
          <div className="shapes-meta-row">
            <dt>Max layers</dt>
            <dd>{shape.max_layers}</dd>
          </div>
        )}
        {shape.max_sources > 0 && (
          <div className="shapes-meta-row">
            <dt>Max sources</dt>
            <dd>{shape.max_sources}</dd>
          </div>
        )}
      </dl>
    </section>
  );
}

/** The shape's declared PHASE flow (research → write → …), when present — the
 * follow-up-plan vocabulary the planner reasons over, NOT a node graph. */
function PhaseFlow({ phases }: { phases: ShapeView["phases"] }) {
  if (!phases || phases.length === 0) return null;
  return (
    <div
      className="shapes-flow shapes-flow-seq"
      role="img"
      aria-label={`phase flow: ${phases.map((p) => p.kind).join(", then ")}`}
    >
      {phases.map((p, i) => (
        <span key={`${p.kind}-${i}`} className="shapes-role-chain-item">
          {i > 0 && <span className="shapes-arrow" aria-hidden="true">→</span>}
          <RoleChip role={p.kind} />
        </span>
      ))}
      <span className="shapes-arrow" aria-hidden="true">→</span>
      <span className="shapes-flow-node">done</span>
    </div>
  );
}

/** Sequential discipline — strict single-file dispatch (at most one node in
 * flight). The planner authors the nodes; this shape only governs HOW they run. */
function SequentialStructure() {
  return (
    <section className="shapes-struct" aria-label="Sequential structure">
      <h3 className="shapes-struct-title">Execution discipline</h3>
      <p className="shapes-struct-note">
        Strict single-file: at most one node runs at a time, each starting only
        once the previous has finished. The planner authors the nodes; this shape
        governs the dispatch order.
      </p>
      <div className="shapes-flow shapes-flow-seq" role="img" aria-label="node then node then node, one at a time">
        <span className="shapes-flow-node">node</span>
        <span className="shapes-arrow" aria-hidden="true">→</span>
        <span className="shapes-flow-node">node</span>
        <span className="shapes-arrow" aria-hidden="true">→</span>
        <span className="shapes-flow-node">node</span>
      </div>
    </section>
  );
}

/** Concurrent discipline — wave fan-out; every ready node launches together,
 * `depends_on` edges still ordering dependent steps. */
function ConcurrentStructure() {
  return (
    <section className="shapes-struct" aria-label="Concurrent structure">
      <h3 className="shapes-struct-title">Execution discipline</h3>
      <p className="shapes-struct-note">
        Wave fan-out: every independent ready node launches at once and runs
        together (bounded by the runtime's max concurrency), while{" "}
        <code>depends_on</code> edges still order dependent steps.
      </p>
      <div className="shapes-flow shapes-flow-par" role="img" aria-label="three nodes in parallel feeding a join node">
        <span className="shapes-flow-fan">
          <span className="shapes-flow-node">node A</span>
          <span className="shapes-flow-node">node B</span>
          <span className="shapes-flow-node">node C</span>
        </span>
        <span className="shapes-arrow" aria-hidden="true">→</span>
        <span className="shapes-flow-node">join</span>
      </div>
    </section>
  );
}

// --------------------------------------------------------------------------- //
// the per-shape max_iter editor — the ONE thing the screen writes (d5)
// --------------------------------------------------------------------------- //
function MaxIterEditor({ shape }: { shape: ShapeView }) {
  // Seed the working value from the persisted override (or the file default when
  // none is set). The form is re-mounted via `key` on the shape name in the parent
  // so this local state is seeded ONCE per shape (no server→state mirroring).
  const [value, setValue] = useState<string>(
    String(shape.max_iter_override ?? shape.max_iter),
  );
  const setMaxIter = useSetShapeMaxIter();
  const [savedAt, setSavedAt] = useState<number | null>(null);

  const parsed = Number.parseInt(value, 10);
  const valid = Number.isFinite(parsed) && parsed >= 1 && parsed <= 1000;
  const current = shape.max_iter_override ?? shape.max_iter;
  const dirty = valid && parsed !== current;
  const iterative = isIterative(shape);

  const save = () => {
    if (!dirty || setMaxIter.isPending || !valid) return;
    setMaxIter.mutate(
      { name: shape.name, maxIter: parsed },
      { onSuccess: () => setSavedAt(Date.now()) },
    );
  };

  // What the runtime WILL run if the typed value is saved — preview the clamp to
  // hard_cap so the user sees a value above the cap will be bounded.
  const previewEffective = valid ? Math.min(parsed, shape.hard_cap) : null;
  const willClamp = previewEffective != null && parsed > shape.hard_cap;

  return (
    <section className="shapes-editor" aria-label="Max iterations">
      <div className="shapes-editor-head">
        <h3 className="shapes-struct-title">Iteration ceiling</h3>
        {shape.max_iter_override != null && (
          <span className="shapes-override-pill">override set</span>
        )}
      </div>

      {!iterative && (
        <p className="shapes-struct-note">
          This shape dispatches its nodes once under its execution discipline, so
          the runtime does not unroll rounds — the ceiling below is stored per the
          per-shape setting but takes effect on iterative shapes (deep-research).
        </p>
      )}

      <div className="shapes-editor-row">
        <label className="shapes-field">
          <span className="shapes-field-label">Max iterations</span>
          <input
            className="shapes-field-input"
            type="number"
            min={1}
            max={1000}
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
              setSavedAt(null);
            }}
            aria-label={`Max iterations for ${shape.name}`}
          />
        </label>
        <button
          type="button"
          className="shapes-primary"
          onClick={save}
          disabled={!dirty || setMaxIter.isPending || !valid}
        >
          {setMaxIter.isPending ? "Saving…" : "Save"}
        </button>
        <span className="shapes-editor-dirty" aria-live="polite">
          {dirty ? "Unsaved changes" : "Up to date"}
        </span>
      </div>

      {!valid && value.trim() !== "" && (
        <p className="shapes-error" role="alert">
          Enter a whole number between 1 and 1000.
        </p>
      )}
      {willClamp && (
        <p className="shapes-clamp-note">
          Above the hard cap ({shape.hard_cap}); the runtime will run{" "}
          {previewEffective} {iterative ? "rounds" : ""}.
        </p>
      )}
      {setMaxIter.isError && (
        <p className="shapes-error" role="alert">{setMaxIter.error.message}</p>
      )}
      {savedAt !== null && !dirty && (
        <p className="shapes-banner shapes-banner-ok" role="status">
          ✓ Saved — persisted to the shared store; the runtime honors it on the
          next {iterative ? "deep-research run" : "iterative run"}.
        </p>
      )}
    </section>
  );
}

// =========================================================================== //
// small presentational helpers
// =========================================================================== //
function ExecutionBadge({ execution }: { execution: ShapeExecution }) {
  const label =
    execution === "deep-research"
      ? "iterative"
      : execution === "sequential"
        ? "sequential"
        : "parallel";
  return (
    <span className={`shapes-badge shapes-badge-${execution}`} title={execution}>
      {label}
    </span>
  );
}

function RoleChip({ role }: { role: string }) {
  return <span className={`shapes-role-chip shapes-role-${role}`}>{role}</span>;
}

/** The shape's basename for the source code reference, tolerating either path
 * separator (the backend stores an absolute OS path). */
function fileName(source: string): string {
  if (!source) return "shape";
  const parts = source.split(/[\\/]/);
  return parts[parts.length - 1] || source;
}
