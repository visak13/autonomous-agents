---
name: spa-rebuild-masked-by-cached-html
description: A rebuilt Vite/SPA bundle can be masked by a cached index.html; verify the loaded asset hash before trusting any UI proof
metadata: 
  node_type: memory
  type: feedback
  originSessionId: ba66dca7-30db-4da4-b88c-4ad177d965cd
---

When re-proving a freshly-built SPA change via Playwright, a rebuilt Vite bundle can be MASKED by a cached `index.html`: the browser keeps running the OLD hashed asset (e.g. `index-lAdv_w3d.js`) even though `curl /` shows the NEW hash, so the new feature "doesn't render" — a false negative that looks like a code bug. In s10-a9 (ReactiveAgents missing-specialist resolution UI) this cost a full debug cycle chasing a phantom logic bug; the card code was correct and the backend payload was correct — only the tab ran a pre-change bundle.

**Why:** `browser_navigate` to the same SPA root URL can serve a cached HTML doc that references a stale asset hash; content-hashed JS alone doesn't help if the HTML pointing at it is cached.

**How to apply:** before trusting a UI proof, (1) cache-bust the HTML — navigate to `/?v=N`; (2) ASSERT `document.querySelectorAll('script')[].src` ends with the exact hash your `vite build` just emitted; (3) only then drive the flow. Temporary in-component `window.__dbg = {...}` is a fast way to confirm the new bundle is actually executing (remove + rebuild before shipping).

Relates to [[runtime-proof-needs-manual-evidence-review]] and [[programmatic-click-masks-pointer-events]] — runtime UI proofs need an explicit "am I really testing the artifact I think I am" check.
