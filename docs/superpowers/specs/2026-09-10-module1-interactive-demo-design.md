# Module 1 Interactive Demo — Design

## Purpose

Module 1 ("How AI Works") currently teaches the tokenization →
embeddings → attention → autoregressive prediction pipeline entirely
in prose. This adds a hands-on, in-browser demo that runs a real small
language model and lets the learner watch their own sentence move
through each of those stages, reinforcing the README's mechanics with
something they did, not just read — consistent with this repo's stated
goal that every module "end with something you did, not just read."

## Scope

One new page/demo attached to Module 1 only. No changes to other
modules. No backend, no server, no build pipeline, no new
repo-wide tooling.

## Tech approach

- Plain HTML + CSS + a small number of hand-written ES modules under
  `modules/01-how-ai-works/demo/`. No bundler, no `npm`/`node_modules`
  checked into the repo, no lockfile.
- [transformers.js](https://github.com/xenova/transformers.js) is
  loaded from a CDN (jsDelivr) at runtime via a `<script type="module">`
  import — not vendored into the repo.
- Hosting via GitHub Pages, enabled on the repository (serving from
  root or `/docs`, whichever proves simplest at enablement time — see
  Open Questions). The demo is reachable at a stable Pages URL and
  linked from `modules/01-how-ai-works/README.md`.

Rejected alternative: a Vite/React (or similar) build. This would be
the only build-tooled piece of an otherwise pure-Markdown repo,
introducing dependency and maintenance overhead disproportionate to a
single demo page, and inconsistent with the "no assumed coding
background" spirit of the repo (a contributor extending this module
shouldn't need to learn a JS toolchain to edit it).

## Model

- **Model:** `Xenova/distilgpt2` — a real, small GPT-2 variant already
  published in transformers.js-compatible ONNX format (no conversion
  work needed). Its architecture matches what the README already
  teaches (token embeddings, multi-head attention, autoregressive
  next-token prediction), so what the demo shows is literally the
  mechanism the prose describes, not an analogy for it.
- **Loading:** lazy — nothing downloads until the learner clicks
  "Load the model." A visible progress bar tracks download percentage.
  A plain-language notice appears before the download starts
  (approximate size, one-time cost, cached by the browser after).
- **Inference:** runs entirely client-side (WASM, or WebGPU where
  available) via transformers.js. No data leaves the learner's
  browser.

## Page structure & UX

Single page, single flow:

1. **Input** — a text box pre-filled with the README's own example
   ("What color is the sky?") so the page works immediately without
   the learner having to think of something to type; fully editable.
2. **Load / Run controls** — "Load the model" (see above), then "Run"
   once loaded. Run is disabled while input is empty or the model
   isn't ready.
3. **Tokens** — chips rendering each token of the input alongside its
   integer id. Directly mirrors the README's step-1 example.
4. **Embeddings** — each token's embedding rendered as a small
   color-coded heatmap strip (communicates "this is a vector of
   numbers," not intended to teach vector math), plus one simple 2D
   plot positioning tokens by similarity to each other.
5. **Attention** — a token-by-token heatmap for a selected
   layer/head (a sensible default layer is picked to try to surface a
   pronoun-to-referent pattern when the input has one), directly
   demonstrating the README's "it" → noun tracking example.
6. **Prediction** — top-5 next-token candidates as probability bars, a
   temperature slider (wired to the same 0–1 range and behavior the
   README describes) that live-recomputes the bars, and a "step"
   button that accepts a token, appends it to the sequence, and
   re-runs — so the learner can watch autoregressive generation happen
   one token at a time.

## Error handling & fallbacks

| Condition | Behavior |
|---|---|
| Browser lacks WASM/WebGPU support needed | Detected on page load; plain message shown instead of a silent failure or broken UI |
| Model download fails or times out | Visible error state with a retry button; no partial/broken render |
| Inference slow on a given device | Run button shows a spinner/disabled state so it doesn't read as hung |
| Empty input | Run stays disabled until there's text |

## Testing & verification

No automated test framework for a static demo page like this.
Verification is manual, via a local static server (ES modules require
`http(s)://`, not `file://`):

- Golden path: type a sentence → load model → run → confirm all four
  sections (tokens, embeddings, attention, prediction) populate →
  drag the temperature slider and confirm bars update → click step a
  few times and confirm generation continues sensibly.
- Edge cases: empty input (Run stays disabled), a sentence with a
  pronoun referring to an earlier noun (sanity-check the attention
  heatmap shows a plausible pattern), a simulated slow/failed load
  (error + retry path).

This will be reported as manual verification, not automated coverage.

## Docs

Add a short "Try it" section to `modules/01-how-ai-works/README.md`
linking to the GitHub Pages URL, following the repo's existing
pattern of the README being the source of truth for a module.

## Open questions

- Exact GitHub Pages source (repo root vs. `/docs`) will be settled
  during implementation based on what's simplest given the repo's
  current structure (all content currently lives under `modules/`).
