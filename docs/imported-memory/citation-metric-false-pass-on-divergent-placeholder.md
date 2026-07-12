---
name: citation-metric-false-pass-on-divergent-placeholder
description: "a \"no-placeholder\" citation check keyed to one fabrication shape false-PASSES on a divergent shape; gate on real-URL count vs fetched, and READ the Sources section"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 656f03e1-d850-4ac3-9a16-3b9aa414cb04
---

Reviewing a citation-fidelity fix (ReactiveAgents s9/c12r), my placeholder detector
matched only the `[Name, 2025]` shape — exactly the shape the fix's anti-fabrication
PROMPT explicitly forbade — so the metric scored `n_placeholders=0` and PASSED. Direct
inspection of the rendered HTML showed the synthesizer had instead authored a DIFFERENT
placeholder shape: `Source Citation Placeholder: <a href="URL 1">Source 1 Title</a>` (and
content placeholders like `[Specific diplomatic incident…]`). The model followed the
instruction to the LETTER (no `[Name,2025]`) and just picked another placeholder.

**Why:** a small model under an anti-fabrication instruction avoids the *named* bad form
but still won't use the real data — it substitutes a novel placeholder the detector
doesn't enumerate. A fabrication-shape allowlist can never be complete.

**How to apply:** for citation gates, the load-bearing signal is `n_real_urls`
(count of `https?://` in the RENDERED output) **cross-checked against the URLs the
research actually fetched (from the traces)** — i.e. "are >=1 real fetched URLs present,
0 foreign/invented" — NOT "no known-placeholder token." And always READ the actual
Sources section / anchors; never bless on the metric alone. Same family as
[[acceptance-check-single-document-ness]] (tag-balance passes a 2-doc concat) and
[[citation-fix-cant-fix-wrong-chunk-selection]] (citation mechanism vs model behaviour).
Mechanism note: c12's feed+prompt was the CORRECT design (real URLs fed, citations
model-authored, no code-injection) — the failure was an E4B long-path ceiling, proven by
the SHORT path citing 3/3 real URLs verbatim while 2/2 long runs cited 0.
