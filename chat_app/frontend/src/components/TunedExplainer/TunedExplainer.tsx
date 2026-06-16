/**
 * "How your model was tuned for your PC" — a STANDALONE, PERSISTENT, RE-OPENABLE
 * explainer view (user-requested deliverable G; recipe s8/b2).
 *
 * AUDIENCE = a non-expert with NO ML/Ollama background. The single most important
 * rule for this file: every technical term is defined in ONE plain sentence the
 * first time it appears, and there are NO undefined acronyms. The content is plain
 * data here (TERMS / SETTINGS / RESULTS), rendered presentationally — every number
 * is a MEASURED value from the real s8 runs (artifacts/s8/optimization_measurement_
 * table.md + agentic_fitness_report.md), not a guess.
 *
 * It is reached via a TopBar entry and rendered as a full-width single-pane view,
 * so it is re-openable any time (not a one-shot toast). Styling is token-only.
 */
import "./TunedExplainer.css";

interface TunedExplainerProps {
  onClose: () => void;
}

/** A term + its one-plain-sentence definition (the "no undefined jargon" rule). */
interface Term {
  readonly term: string;
  readonly plain: string;
}

/** "QUALITY" = changes how smart the model is. "FIT/SPEED" = quality-neutral. */
type SettingCategory = "quality" | "fit-speed";

interface Setting {
  readonly name: string;
  readonly value: string;
  readonly category: SettingCategory;
  /** One plain sentence: what this does FOR YOU. */
  readonly forYou: string;
  /** The measured effect from the real runs. */
  readonly measured: string;
}

interface Measured {
  readonly label: string;
  readonly value: string;
  readonly plain: string;
}

const ANALOGIES: readonly Term[] = [
  {
    term: "Weights = the model's brain",
    plain:
      "The “weights” are the billions of numbers the model learned during training — think of them as its brain. More of them, kept more precisely, means a bigger, heavier brain.",
  },
  {
    term: "Quantization = how much you compress the brain",
    plain:
      "“Quantization” means storing each of those brain-numbers using fewer digits so the whole model takes up less space — like saving a photo at a smaller file size. “Q8” keeps 8 bits per number (lightly compressed); “Q4” / “int4” keeps only 4 bits (heavily compressed, about half the size again).",
  },
  {
    term: "QAT = trained to stay sharp while compressed",
    plain:
      "“QAT” (Quantization-Aware Training) means the model was TAUGHT to work well in its compressed int4 form from the start — so the heavy compression is not a lossy afterthought. The result is much smaller — its weights take about 3.1 GiB on disk instead of the ~7.6 GiB of the 8-bit version — at near-original quality.",
  },
  {
    term: "KV cache = the scratchpad",
    plain:
      "The “KV cache” is the short-term scratchpad the model uses to remember the words it has already read in the current conversation, so it doesn't re-read them every time.",
  },
  {
    term: "Flash attention = a faster way to do the SAME math",
    plain:
      "“Flash attention” is a smarter way to run the exact same calculation more quickly. It does NOT skip layers and does NOT make the model dumber — same answer, less time.",
  },
  {
    term: "Context window (num_ctx) = how much it can read at once",
    plain:
      "The “context window” (the num_ctx setting) is the maximum amount of text the model can hold in view at one time — measured in “tokens”.",
  },
  {
    term: "Token = a chunk of text",
    plain:
      "A “token” is a small piece of text — very roughly ¾ of a word. “Tokens per second” is just the model's typing speed.",
  },
  {
    term: "VRAM = the graphics card's memory",
    plain:
      "“VRAM” is the dedicated memory on your graphics card (GPU). The whole model has to fit inside it to run fast. Your card has 6 GB.",
  },
  {
    term: "Temperature / top_p / top_k = the randomness dials",
    plain:
      "“Temperature”, “top_p” and “top_k” control how random vs. predictable the model's wording is. Turned all the way down (temperature 0) it gives the same dependable answer every time — which is what we want for planning.",
  },
  {
    term: "TTFT = the wait before the first word",
    plain:
      "“TTFT” (time-to-first-token) is how long you wait after hitting send before the model starts replying.",
  },
  {
    term: "Thinking model = it reasons privately first",
    plain:
      "This is a “thinking model”: by default it writes out a long private train-of-thought before its real answer. The think=false setting tells it to skip that and answer directly (see the last section for why that matters here).",
  },
];

const SETTINGS: readonly Setting[] = [
  {
    name: "Brain compression: int4 QAT",
    value: "int4 QAT (Q4)",
    category: "quality",
    forYou:
      "This is the ONLY setting that changes how smart the model is. We use the int4 QAT brain — heavily compressed but trained to stay sharp — so its weights take ~3.1 GiB on disk (vs ~7.6 GiB for the 8-bit version) at near-original quality.",
    measured: "Model fits in 1.44 GiB of VRAM instead of multiple GB.",
  },
  {
    name: "Answer directly: think = false",
    value: "think = false",
    category: "fit-speed",
    forYou:
      "Tells the thinking model to give the answer straight away instead of writing a long private reasoning trace first. For planning this is the make-or-break setting.",
    measured: "Plans came back valid 24 out of 24 times; with it off, plans silently failed.",
  },
  {
    name: "Structured replies: format = json",
    value: "format = json",
    category: "fit-speed",
    forYou:
      "Asks the model to reply as clean, structured data the app can act on reliably (rather than free-form prose).",
    measured: "100% of replies parsed correctly on the first try.",
  },
  {
    name: "Predictable wording: temperature 0",
    value: "temperature 0 · top_p 0.95 · top_k 64",
    category: "fit-speed",
    forYou:
      "Turns the randomness all the way down so you get the same dependable plan every time you ask the same thing.",
    measured: "Deterministic: identical, valid plan on every repeat.",
  },
  {
    name: "Memory span: context window 8K (up to 128K)",
    value: "num_ctx 8192 (max 131072)",
    category: "fit-speed",
    forYou:
      "How much text it can read at once. We set a generous 8,000 tokens, and it can stretch to 128,000 — far more than a normal chat needs.",
    measured: "Even the full 128K span costs only ~1.56 GiB of VRAM.",
  },
  {
    name: "Scratchpad precision: KV cache left full (f16)",
    value: "KV cache f16 (not compressed)",
    category: "fit-speed",
    forYou:
      "We keep the model's scratchpad at full precision instead of compressing it, because on this model the scratchpad is already tiny — only 3 of its 35 layers keep one.",
    measured: "Compressing it saved no memory here and actually cost ~5% speed.",
  },
  {
    name: "Faster math: flash attention on",
    value: "OLLAMA_FLASH_ATTENTION = 1",
    category: "fit-speed",
    forYou:
      "Runs the exact same calculation faster. Same answer — just less waiting, especially on long inputs.",
    measured: "No speed penalty on short replies, a clear win on long ones.",
  },
  {
    name: "One request at a time: num_parallel 1",
    value: "OLLAMA_NUM_PARALLEL = 1",
    category: "fit-speed",
    forYou:
      "Devotes the whole graphics card to your single request instead of splitting it, which is both faster and gives your request the full memory span.",
    measured: "Typing speed rose from ~89 to ~94 tokens/sec.",
  },
  {
    name: "Stay warm: keep_alive 30m + warm-up",
    value: 'keep_alive "30m" + startup warm-up',
    category: "fit-speed",
    forYou:
      "Keeps the model loaded and ready so your first message doesn't wait for it to wake up from cold.",
    measured: "First-word wait drops from ~8–10 s (cold) to ~0.38 s (warm).",
  },
];

const RESULTS: readonly Measured[] = [
  { label: "Typing speed", value: "~82 tokens/sec", plain: "How fast it writes the answer." },
  { label: "Memory used", value: "1.44 GiB VRAM", plain: "Out of your 6 GB card — leaving ~4.4 GiB free." },
  { label: "Reading span", value: "128K tokens", plain: "How much it can take in at once." },
  { label: "First-word wait (warm)", value: "~0.38 s", plain: "The pause before it starts replying." },
];

export function TunedExplainer({ onClose }: TunedExplainerProps) {
  return (
    <section className="tuned" aria-label="How your model was tuned for your PC">
      <header className="tuned-head">
        <div className="tuned-titles">
          <h1 className="tuned-title">How your model was tuned for your PC</h1>
          <p className="tuned-sub">
            The plain-English story of how we made the AI fit and run fast on your own
            graphics card — every term explained, no jargon left hanging.
          </p>
        </div>
        <button type="button" className="tuned-back" onClick={onClose}>
          ← Back to tasks
        </button>
      </header>

      <div className="tuned-body">
        {/* The one big idea: two categories of knobs. */}
        <article className="tuned-card tuned-bigidea">
          <h2 className="tuned-h2">The one big idea: two kinds of knobs</h2>
          <p>
            Tuning a model means turning a lot of little knobs. The good news is they fall
            into just <strong>two groups</strong>, and only one of them can change how
            smart the model is:
          </p>
          <div className="tuned-twocol">
            <div className="tuned-cat tuned-cat-quality">
              <span className="tuned-cat-tag">Quality knob</span>
              <p>
                <strong>Just one knob is in here:</strong> how much we compress the model's
                “brain” (its <em>weights</em>). This is the only setting that can affect how
                good the answers are.
              </p>
            </div>
            <div className="tuned-cat tuned-cat-fit">
              <span className="tuned-cat-tag">Fit &amp; speed knobs</span>
              <p>
                <strong>Everything else.</strong> These decide how well the model fits in
                your graphics card and how fast it runs. They are{" "}
                <em>quality-neutral</em> — they don't make the model smarter or dumber.
              </p>
            </div>
          </div>
        </article>

        {/* Plain-language glossary / analogies. */}
        <article className="tuned-card">
          <h2 className="tuned-h2">The words you'll see, in plain language</h2>
          <p className="tuned-lead">
            A few terms come up below. Here's each one in a single plain sentence (with the
            picture we find helpful):
          </p>
          <dl className="tuned-glossary">
            {ANALOGIES.map((t) => (
              <div className="tuned-term" key={t.term}>
                <dt className="tuned-term-name">{t.term}</dt>
                <dd className="tuned-term-def">{t.plain}</dd>
              </div>
            ))}
          </dl>
        </article>

        {/* The VRAM budget math. */}
        <article className="tuned-card">
          <h2 className="tuned-h2">Why it fits: the 6 GB memory budget</h2>
          <p>
            Your graphics card has <strong>6 GB of VRAM</strong>, and the whole model has to
            fit inside it to run fast. That budget is what drove the single quality choice:
          </p>
          <ul className="tuned-budget">
            <li>
              <span className="tuned-budget-pick tuned-pick-yes">Chosen ✓</span>
              <span>
                <strong>int4 QAT — about 3.1 GiB on disk, and only 1.44 GiB of VRAM once
                loaded.</strong> Fits easily, leaving <strong>~4.4 GiB free</strong> as
                headroom. Because it was trained to stay sharp while compressed (QAT), you get
                near-original quality at roughly half the size of the 8-bit build.
              </span>
            </li>
            <li>
              <span className="tuned-budget-pick tuned-pick-no">Rejected ✗</span>
              <span>
                <strong>The standard build (Q4_K_M) — about 6.7 GiB.</strong> Still larger
                than the entire 6 GB card, so it would spill out of VRAM and crawl.
              </span>
            </li>
            <li>
              <span className="tuned-budget-pick tuned-pick-no">Rejected ✗</span>
              <span>
                <strong>The lightly-compressed “Q8” version — about 7.6 GiB.</strong> Also too
                big for the 6 GB card. The lighter compression bought no quality worth the
                fit it loses here.
              </span>
            </li>
          </ul>
          <p className="tuned-note">
            A surprise we measured: this model uses “sliding-window” attention, so only{" "}
            <strong>3 of its 35 layers</strong> keep a scratchpad (KV cache). That makes the
            scratchpad tiny — even a full 128K reading span costs only ~1.56 GiB. So a bigger
            memory span is essentially free here, and there's no need to compress the
            scratchpad.
          </p>
        </article>

        {/* The actual chosen settings table. */}
        <article className="tuned-card">
          <h2 className="tuned-h2">The actual settings we chose</h2>
          <p className="tuned-lead">
            Each row is a real setting we landed on, what it does for you, and the effect we
            measured. The colored tag shows which of the two groups it belongs to.
          </p>
          <div className="tuned-table-wrap" role="region" aria-label="Chosen settings" tabIndex={0}>
            <table className="tuned-table">
              <thead>
                <tr>
                  <th scope="col">Setting</th>
                  <th scope="col">Group</th>
                  <th scope="col">What this does for you</th>
                  <th scope="col">Measured effect</th>
                </tr>
              </thead>
              <tbody>
                {SETTINGS.map((s) => (
                  <tr key={s.name}>
                    <th scope="row" className="tuned-setting-name">
                      {s.name}
                      <code className="tuned-setting-value">{s.value}</code>
                    </th>
                    <td>
                      <span
                        className={
                          s.category === "quality"
                            ? "tuned-pill tuned-pill-quality"
                            : "tuned-pill tuned-pill-fit"
                        }
                      >
                        {s.category === "quality" ? "Quality" : "Fit & speed"}
                      </span>
                    </td>
                    <td>{s.forYou}</td>
                    <td className="tuned-measured">{s.measured}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        {/* The headline measured results. */}
        <article className="tuned-card">
          <h2 className="tuned-h2">The result, measured on your PC</h2>
          <div className="tuned-results">
            {RESULTS.map((r) => (
              <div className="tuned-result" key={r.label}>
                <span className="tuned-result-value">{r.value}</span>
                <span className="tuned-result-label">{r.label}</span>
                <span className="tuned-result-plain">{r.plain}</span>
              </div>
            ))}
          </div>
        </article>

        {/* The thinking-model insight. */}
        <article className="tuned-card">
          <h2 className="tuned-h2">The “thinking model” insight</h2>
          <p>
            This model is a <strong>thinking model</strong>: left to itself, it writes out a
            long private train-of-thought before giving its real answer. That's often useful
            — but when the app asks it to <em>plan</em> the steps of a task, that private
            reasoning ran so long it used up the model's reply budget{" "}
            <strong>before it ever wrote the plan</strong> — so the plan came back empty.
          </p>
          <p>
            The fix is one setting: <code>think = false</code>. It tells the model to skip the
            private monologue and write the plan directly. With it on, planning failed; with
            it off, the model produced a valid plan <strong>24 times out of 24</strong>. That
            single switch is why your model can reliably drive the app.
          </p>
        </article>
      </div>
    </section>
  );
}
