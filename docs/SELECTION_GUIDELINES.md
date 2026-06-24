# Selection Guidelines — how the planner picks specializations & shapes

**Audience:** the model making a selection call (planner DAG authoring,
incremental per-node authoring, and the shape selector), and the reviewer who
checks that selection behaves. This is the single shared criteria doc the
selection prompts point at. It is **selection-side only** — how to CHOOSE among
existing candidates. How a spec/shape is *authored* (its body, its description)
is covered elsewhere; this doc only states the **quality bar** a description must
clear to be selectable.

Grounded in the s2 audit (`.audit-s2/`) and the s4/a1 diagnosis
(`.s4-findings/SELECTION-AND-LIFECYCLE.md`). Both selection prompts ALREADY show
the model each candidate's **name + description** — the gap this doc closes is
not visibility, it is the absence of documented match-criteria, precedence, and a
description quality bar.

---

## 1. The two independent selection calls

Selection is two separate native-structured Gemma calls; neither sees the other's
output:

| Call | Picks | Where | Catalog shown |
|---|---|---|---|
| **Shape selector** | the ONE plan shape (how the work runs) | `shape_selector.py` `_system_prompt` | every shape's name + one-line description, plus `escalate` |
| **Plan factory** | per-node `spec`/`specs`/`needs_spec` (which expert runs a step) | `factory.py` `FACTORY_DESCRIPTION` + `NODE_SCHEMA` | registered specializations: name + description (body-free) |

Shape = the **topology** of the work (one pass, parallel branches, multi-round
research…). Specialization = the **expertise / output-style** applied to a single
step. They are orthogonal: choose the shape for the whole goal, choose a spec per
node.

---

## 2. SPEC selection — what makes a step MATCH a specialization

A step matches a specialization when **the step's actual work is the work the
spec's description says it does** — judged on intent, not surface wording. Apply,
in order:

1. **Match on the WORK, not keywords.** Compare what the node must PRODUCE to
   what each spec description says it produces/binds to. A node that "writes the
   final report as Markdown" matches `markdown-writer`
   ("Bind to the node that PRODUCES a written report… when the user wants
   Markdown"); a node that "reads the sources and reports concrete figures"
   matches `research-analyst`. Do not match on a shared word alone.
2. **Bind a spec to the node it actually shapes.** An output-style spec
   (`markdown-writer`, `html-writer`, `terse-emoji-brief`, `pirate-speak-brief`)
   belongs on the node that PRODUCES the user-visible deliverable — usually the
   final synthesis/write node — NOT on an upstream research node. An
   analysis/role spec (`research-analyst`, `forensic-accountant`) belongs on the
   node doing that reasoning.
3. **Match the requested OUTPUT FORMAT.** When the user names a format for the
   deliverable, bind the output-style spec FOR THAT FORMAT and no other: an HTML
   request (an HTML report / a web page / a `.html` file) → `html-writer`; a
   Markdown request (a `.md` document / readable Markdown) → `markdown-writer`.
   The two are mutually exclusive — never substitute one for the other, and never
   stack both. This is the s8/b2 fix: an HTML request must NOT route to
   `markdown-writer` just because that spec's description was the more specific.
4. **No confident match → leave it UNSPECIALIZED.** A plain producer step with
   no clear specialist is correct as `spec: null`. Do not stretch a loosely
   related spec onto a node "just in case" — a wrong spec injects the wrong
   output contract and degrades the step. Specialization is opt-in, not required.
5. **A real expert is needed but none is registered → `needs_spec`.** If the step
   genuinely REQUIRES a specialist no listed spec covers, leave `spec`/`specs`
   empty and describe the needed capability in `needs_spec` in plain terms
   (e.g. "forensic accountant report"). **Never invent a spec name**, and never
   silently run a step that needed an expert as unspecialized.

### 2b. Precedence & tie-breaks (resolve in this order)

1. **User-named beats planner-authored.** A specialization the USER explicitly
   asked for — surfaced by the shape selector as `requested_specs` (enum-locked to
   registered names) — takes precedence over a spec the planner would otherwise
   pick for that role. The user naming `research-analyst` wins over a default.
2. **`spec` vs `specs`.** Use the scalar `spec` for exactly ONE specialization.
   Use the list `specs` ONLY when **two or more genuinely apply and compose**
   (e.g. `research-analyst` for the analysis contract + `markdown-writer` for the
   output format on the same final node); they apply in listed order. Never put a
   single name in `specs`, and never set both `spec` and `specs`.
3. **Cap on layering.** Keep `specs` to the **minimum that compose cleanly —
   prefer 1, at most 2–3.** Each layer adds tokens and can conflict (two
   output-style specs on one node contradict each other). If two candidates are
   output styles, pick ONE; do not stack `markdown-writer` + `html-writer`.
4. **Two plausible specs, only one slot → pick the one whose description most
   specifically names this node's work.** A spec that says "Bind to the node
   that PRODUCES a written report" beats a vague "writes documents" for the
   final-report node. If still tied, prefer the narrower/more specific spec; if
   neither clearly fits, leave it unspecialized rather than guess.

---

## 3. SHAPE selection

Shape selection is already well-guided in `_system_prompt`; the rules below are
the shared restatement (keep them and the prompt consistent):

1. **Choose by the WORK, not the phrasing.** A question, a "describe…" and an
   imperative that ask for the SAME result route to the SAME shape.
   - one straight sequence of steps → `linear`
   - independent parts to gather then combine → `modular-parallel` /
     `concurrent-multi-topic-gathering`
   - a sequential baseline THEN parallel exploration → `linear-plus-modular-parallel`
   - an exhaustive multi-round survey of ONE topic with critique →
     `deep-research` / `iterative-deep-research`
   - draft → review → improve a piece of writing → `iterative-writing-improvement`
2. **Do not over-escalate.** A simple informational request phrased as a question
   is NOT a reason to pick a heavy multi-round research shape. Match the shape's
   weight to the work's weight.
3. **`escalate` only when genuinely unsure** that any shape fits — never as a lazy
   default, never to invent a shape.

The same call also reports intent signals (`search_allowed`, `requested_specs`,
`unmet_specs`, `wants_file`). Report what the user's intent actually is; do not
guess a need the user did not express (`unmet_specs`/`requested_specs` = `[]` when
none).

---

## 4. The description QUALITY BAR (the selection lever)

Selection can only be as good as the candidate descriptions the model is shown —
a spec/shape is **only as selectable as its description is clear.** This is the
lever: when authoring or editing a spec/shape description, make it
**selection-grade**. A selection-grade description:

- **States WHAT the spec/shape does** in concrete terms (the work, the output
  contract), not a vague restatement of its name.
- **States WHEN to bind it** — the kind of node/step or request it fits (the
  strong specs do this explicitly: *"Bind to the node that PRODUCES a written
  report… when the user wants Markdown"*).
- **Is distinguishable** from its neighbours, so the model can tell two
  candidates apart rather than coin-flip.
- **Is concise** — one to three tight sentences; selection reads the whole
  catalog every call, so verbosity costs tokens on every plan.

Grounding from the live catalog:

- **Selection-grade (good):** `markdown-writer`, `html-writer`, `research-analyst`,
  `forensic-accountant`, `terse-emoji-brief` — each states the work AND when to
  bind. (`html-writer` was promoted to a canonical seed in s8/b5 with a
  selection-grade, format-discriminating description — *"…when the user wants HTML
  / a web page / a .html file — NOT Markdown"* — paired symmetrically with
  `markdown-writer`'s *"…readable Markdown output (a .md document) — NOT HTML"*, so
  an HTML request no longer loses the tie to `markdown-writer`.)
- **Below the bar (fix on next edit):** `Deep-research` ("Do a detailed research
  on the given topic") — true but generic; it gives the selector nothing to bind
  ON beyond the name. Test artifacts like `a3-reedit-proof` are noise and should
  be deleted, not matched.

A description that fails this bar is a **selection bug at the source** — fix the
description rather than adding selection heuristics to compensate.

---

## 5. Quick checklist

- [ ] Matched each node's spec on the WORK it produces, not a shared keyword.
- [ ] Output-style spec is on the PRODUCER node; role spec on the reasoning node.
- [ ] No-confident-match nodes left `spec: null` (not force-fit).
- [ ] A required-but-unregistered expert is in `needs_spec`, no invented name.
- [ ] User-requested spec honored over a planner default.
- [ ] `spec` for one, `specs` only for genuine composition (≤2–3, no conflicts).
- [ ] Shape chosen by the work's weight; `escalate` only when truly unsure.
- [ ] Any description below the quality bar flagged to fix at the source.
