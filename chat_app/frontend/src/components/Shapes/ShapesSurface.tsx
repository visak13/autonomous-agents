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
 *    plan shapes and surfaces each shape's REAL structure — the execution
 *    discipline (sequential / concurrent) and, for the bounded cyclic
 *    deep-research shape, the round_roles/final_roles UNROLL.
 *  - the only thing the user EDITS here is a shape's per-shape MAX_ITER override
 *    (d5), saved through the a4 backend (PUT /shapes/{name}/max_iter) so it
 *    persists to the shared SQLite and is honored by the s3 deep-research unroll.
 *    For an iterative shape the structure preview reflects the EFFECTIVE round
 *    count live; for a dispatch-discipline shape the editor is shown (the mandate
 *    is per-shape) with an honest note that the ceiling applies to iterative
 *    unrolls.
 *
 * Server state (the shape catalog, a shape's view) is the Query cache; the only
 * local state is which shape is selected and the editor's working value. The
 * execution discipline is a discriminated union consumed exhaustively (spec
 * [required]) so a new backend discipline is a compile error, never a silent
 * fall-through.
 */
import { useState } from "react";
import {
  useAuthorShape,
  useSetShapeMaxIter,
  useShapes,
} from "../../api/queries";
import type { ShapeExecution, ShapeView } from "../../api/types";
import "./ShapesSurface.css";

interface ShapesSurfaceProps {
  onClose: () => void;
}

export function ShapesSurface({ onClose }: ShapesSurfaceProps) {
  const shapes = useShapes();
  const [selected, setSelected] = useState<string | null>(null);

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

      <div className="shapes-body">
        <div className="shapes-sidebar">
          <ShapeAuthorForm onAuthored={(name) => setSelected(name)} />
          <nav className="shapes-list" aria-label="Plan shapes">
          {shapes.isLoading && <p className="shapes-hint">Loading shapes…</p>}
          {shapes.isError && (
            <p className="shapes-error" role="alert">{shapes.error.message}</p>
          )}
          {shapes.data && list.length === 0 && (
            <p className="shapes-hint">No text-file shapes are defined.</p>
          )}
          {list.map((shape) => (
            <button
              key={shape.name}
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
          ))}
          </nav>
        </div>

        <div className="shapes-detail">
          {active ? (
            <ShapeDetail key={active.name} shape={active} />
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
// describe-a-shape: the user DESCRIBES a shape and the live Gemma model authors
// the declarative file (s9/b1, d14(2)). Mirrors the spec screen's describe→author
// UX (a plain-language box + an author button), but it is the GENUINE one-shot
// shapes flow — a shape is a single small structured decision, not a multi-turn
// ruleset conversation. On success the catalog refreshes and the new shape is
// auto-selected so the user immediately sees its authored structure.
// =========================================================================== //
function ShapeAuthorForm({ onAuthored }: { onAuthored: (name: string) => void }) {
  const [description, setDescription] = useState("");
  const author = useAuthorShape();

  const submit = () => {
    const trimmed = description.trim();
    if (!trimmed || author.isPending) return;
    author.mutate(
      { description: trimmed },
      {
        onSuccess: (shape) => {
          setDescription("");
          onAuthored(shape.name);
        },
      },
    );
  };

  return (
    <form
      className="shapes-author"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <h2 className="shapes-author-title">Describe a new shape</h2>
      <p className="shapes-author-hint">
        Describe the execution posture in plain language — the local model authors
        the declarative shape file. You never hand-write shapes.
      </p>
      <label className="shapes-field">
        <span className="shapes-field-label">Description</span>
        <textarea
          className="shapes-field-input shapes-author-input"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="e.g. iteratively research a topic in depth, a critic checking each round, then synthesize and verify"
          maxLength={4000}
          rows={3}
          aria-label="Describe the shape to author"
        />
      </label>
      {author.isError && (
        <p className="shapes-error" role="alert">{author.error.message}</p>
      )}
      <button
        type="submit"
        className="shapes-primary"
        disabled={author.isPending || description.trim() === ""}
      >
        {author.isPending ? "Authoring…" : "Author shape"}
      </button>
    </form>
  );
}

/** A short "ceiling" summary for the list row (genuine: only iterative shapes
 * iterate, so others read as single-pass). */
function iterSummary(shape: ShapeView): string {
  if (!isIterative(shape)) return "single pass";
  const over = shape.max_iter_override;
  return over != null && over !== shape.max_iter
    ? `${shape.effective_max_iter} rounds (set)`
    : `${shape.effective_max_iter} rounds`;
}

/** A shape iterates (consumes max_iter) iff it declares per-round roles — only the
 * deep-research bounded-unroll shape does. The dispatch-discipline shapes
 * (sequential/concurrent) run their nodes once under a launch discipline. */
function isIterative(shape: ShapeView): boolean {
  return shape.round_roles.length > 0;
}

// =========================================================================== //
// detail: the shape's real structure + the max_iter editor
// =========================================================================== //
function ShapeDetail({ shape }: { shape: ShapeView }) {
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

      <MaxIterEditor shape={shape} />
    </article>
  );
}

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

/** The bounded cyclic unroll, rendered round-by-round to the EFFECTIVE round count
 * the runtime will actually run (override clamped to hard_cap). Non-final rounds
 * emit `round_roles` ({research, critic}); the single final round emits
 * `final_roles` ({research, synthesis, verify}). Each round depends on the prior
 * round's tail → growing visibility into every earlier layer. */
function DeepResearchStructure({ shape }: { shape: ShapeView }) {
  const total = shape.effective_max_iter;
  // Render every round when the count is small; otherwise show the first two
  // non-final rounds, an elision for the middle, and ALWAYS the distinct final
  // round so both the {research+critic} pattern and the final
  // {research+synthesis+verify} round are visible without a huge list.
  const ELIDE_OVER = 6;
  const nonFinalCount = Math.max(0, total - 1); // rounds 1..total-1 use round_roles
  const elide = total > ELIDE_OVER;
  const headCount = elide ? 2 : nonFinalCount;

  // Build an explicit render list so the elision is its OWN item between the head
  // rounds and the final round — never standing in for a real round.
  type RoundItem =
    | { kind: "round"; index: number; final: boolean; roles: string[] }
    | { kind: "elide"; from: number; to: number };

  const items: RoundItem[] = [];
  for (let n = 1; n <= headCount; n++) {
    items.push({ kind: "round", index: n, final: false, roles: shape.round_roles });
  }
  if (elide && nonFinalCount > headCount) {
    items.push({ kind: "elide", from: headCount + 1, to: nonFinalCount });
  }
  if (total >= 1) {
    items.push({ kind: "round", index: total, final: true, roles: shape.final_roles });
  }

  return (
    <section className="shapes-struct" aria-label="Deep-research structure">
      <h3 className="shapes-struct-title">Bounded unroll</h3>
      <p className="shapes-struct-note">
        The same specialization runs every round — only the node <em>role</em>
        differs. {nonFinalCount} {nonFinalCount === 1 ? "round" : "rounds"} of{" "}
        {roleList(shape.round_roles)}, then 1 final round of{" "}
        {roleList(shape.final_roles)}. Each round sees all prior researched layers.
      </p>

      <ol className="shapes-rounds">
        {items.map((item) =>
          item.kind === "elide" ? (
            <li
              key="elide"
              className="shapes-round shapes-round-elide"
              aria-hidden="true"
            >
              <span className="shapes-round-label">⋮</span>
              <span className="shapes-round-elide-text">
                rounds {item.from}–{item.to} · {roleList(shape.round_roles)}
              </span>
            </li>
          ) : (
            <li
              key={item.index}
              className={`shapes-round${item.final ? " shapes-round-final" : ""}`}
            >
              <span className="shapes-round-label">
                {item.final ? "final round" : `round ${item.index}`}
              </span>
              <span className="shapes-role-chain">
                {item.roles.map((role, j) => (
                  <span key={role} className="shapes-role-chain-item">
                    {j > 0 && <span className="shapes-arrow" aria-hidden="true">→</span>}
                    <RoleChip role={role} />
                  </span>
                ))}
              </span>
            </li>
          ),
        )}
      </ol>

      <dl className="shapes-meta">
        <div className="shapes-meta-row">
          <dt>Effective rounds</dt>
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
      </dl>
    </section>
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

function roleList(roles: string[]): string {
  if (roles.length === 0) return "no roles";
  if (roles.length === 1) return `{${roles[0]}}`;
  return `{${roles.join(" + ")}}`;
}

/** The shape's basename for the source code reference, tolerating either path
 * separator (the backend stores an absolute OS path). */
function fileName(source: string): string {
  if (!source) return "shape";
  const parts = source.split(/[\\/]/);
  return parts[parts.length - 1] || source;
}
