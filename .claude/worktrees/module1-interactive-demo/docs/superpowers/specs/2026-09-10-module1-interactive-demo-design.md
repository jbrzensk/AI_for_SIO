# Module 1 Interactive Demo — Design

## Purpose

Module 1 ("How AI Works") currently teaches the tokenization →
embeddings → attention → autoregressive prediction pipeline entirely
in prose. This adds a hands-on, in-browser demo that runs a real small
language model and lets the learner watch their own sentence move
through each of those stages, reinforcing the README's mechanics with
something they did, not just read — consistent with this repo's stated
goal that every module "end with something you did, not just read."
(Tokenization and next-token prediction are the model's genuine
output; embeddings and attention are clearly-labeled illustrative
approximations — see "What's real vs. illustrative" under Model
below, a correction made after inspecting the actual model file.)

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
  work needed).
- **Loading:** lazy — nothing downloads until the learner clicks
  "Load the model." A visible progress bar tracks download percentage.
  A plain-language notice appears before the download starts
  (approximate size, one-time cost, cached by the browser after).
- **Inference:** runs entirely client-side (WASM, or WebGPU where
  available) via transformers.js. No data leaves the learner's
  browser.

### What's real vs. illustrative (important correction from initial brainstorm)

The published `Xenova/distilgpt2` ONNX graph (verified by inspecting
its declared graph outputs directly) only exposes two things:
`logits` and the `present.*.key`/`present.*.value` KV-cache tensors
used internally for fast generation. It does **not** expose attention
weights or hidden-state/embedding vectors — those were never wired up
as graph outputs at export time, and there is no runtime flag that
makes a static ONNX graph emit tensors it wasn't built to emit.

This means, concretely:

- **Tokenization** — 100% real. Happens in JS via the model's real
  tokenizer before any model call.
- **Autoregressive prediction** — 100% real. `logits` is a genuine
  model output, so the top-5 next-token probabilities and the
  temperature slider reflect the model's actual behavior.
- **Embeddings** and **Attention** — the real tensors are not
  obtainable from this model file, full stop (not merely
  difficult — the graph doesn't produce them). These two panels are
  therefore **illustrative, hand-built approximations**, clearly
  labeled as such in the UI, rather than the model's real internals:
  - *Embeddings panel:* each token gets a small deterministic vector
    derived from a character-n-gram hash of its text (a legitimate,
    simplified feature-hashing technique — not random noise, but not
    distilgpt2's real embedding either). The heatmap strip visualizes
    "a token becomes a vector of numbers." The accompanying 2D plot
    positions tokens by similarity *of these hashed vectors*, and is
    captioned to say so explicitly — it will tend to cluster tokens
    that share spelling/substrings, not necessarily meaning, and the
    UI must not claim otherwise.
  - *Attention panel:* weights are computed by a distance +
    repetition heuristic (tokens closer together, and tokens that
    are repeats of an earlier token, get higher weight), rendered as
    the same token-by-token heatmap the real-data version would have
    used. Captioned explicitly as an illustrative pattern, not
    distilgpt2's real attention weights.

  Both panels carry a visible, permanent caption (not a dismissible
  tooltip) making this distinction, so the page never implies these
  two panels are measuring the model itself.

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
4. **Embeddings (illustrative — see "What's real vs. illustrative"
   above)** — each token's hashed vector rendered as a small
   color-coded heatmap strip, plus one simple 2D plot positioning
   tokens by similarity of those hashed vectors. Permanently captioned
   as a simplified illustration, not the model's real embeddings.
5. **Attention (illustrative — see above)** — a token-by-token
   heatmap driven by the distance + repetition heuristic. Permanently
   captioned as an illustrative pattern, not the model's real
   attention weights. No layer/head selector, since there's no real
   per-layer/per-head data to switch between.
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
  repeated word (sanity-check the illustrative attention heatmap
  reflects the repetition heuristic as designed), a simulated
  slow/failed load (error + retry path).

This will be reported as manual verification, not automated coverage.

## Docs

Add a short "Try it" section to `modules/01-how-ai-works/README.md`
linking to the GitHub Pages URL, following the repo's existing
pattern of the README being the source of truth for a module.

## Open questions

- Exact GitHub Pages source (repo root vs. `/docs`) will be settled
  during implementation based on what's simplest given the repo's
  current structure (all content currently lives under `modules/`).
