# Module 1 Interactive Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an in-browser, GitHub Pages–hosted demo to Module 1 that runs a real small language model (`Xenova/distilgpt2` via transformers.js) so a learner can type a sentence and watch it move through tokenization, (illustrative) embeddings, (illustrative) attention, and real autoregressive next-token prediction.

**Architecture:** A single static page (`modules/01-how-ai-works/demo/index.html`) with plain CSS and hand-written ES modules — no bundler, no npm dependency tree in the repo. `transformers.js` is imported lazily from a pinned CDN URL only when the learner clicks "Load the model." Pure computation (hashing, softmax, similarity, the illustrative-attention heuristic) lives in a dependency-free `math.js` module with real Node unit tests; everything that touches the DOM or the model is verified manually in a browser, per the spec.

**Tech Stack:** Vanilla JavaScript (ES modules), HTML, CSS, [`@huggingface/transformers`](https://www.npmjs.com/package/@huggingface/transformers) v4.2.0 loaded from jsDelivr, Node.js built-in test runner (`node --test`) for pure-logic unit tests, GitHub Pages for hosting.

**Spec:** [docs/superpowers/specs/2026-09-10-module1-interactive-demo-design.md](../specs/2026-09-10-module1-interactive-demo-design.md)

## Global Constraints

- No bundler, no build step. Every file under `modules/01-how-ai-works/demo/` is served as-is.
- No `node_modules`, no dependency lockfile committed to the repo. The only Node-related file is a 1-line `package.json` (`{"type": "module"}`) so `node --test` treats `.js` files as ES modules for local test runs — this declares zero dependencies.
- `transformers.js` is loaded from `https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0` (pinned exact version), and only via a **lazy dynamic `import()`** inside `loadModel()` — never a static top-level import — so nothing downloads before the learner clicks "Load the model."
- Model: `Xenova/distilgpt2`.
- Per the corrected spec: tokenization and next-token prediction must use the model's real output. Embeddings and Attention panels are illustrative heuristics and **must** carry a permanent, visible caption saying so — never implied to be the model's real internals.
- GitHub Pages source: root of the `main` branch (confirmed — no `/docs` restructuring needed). Final URL: `https://jbrzensk.github.io/AI_for_SIO/modules/01-how-ai-works/demo/`.
- Verification is manual, in a real browser, via a local static server (ES modules require `http(s)://`, not `file://`) — e.g. `cd modules/01-how-ai-works/demo && python3 -m http.server 8000`, then open `http://localhost:8000/`. The one exception is `math.js`, which is pure and has real automated tests via `node --test`.

---

### Task 1: Pure pipeline math (`math.js`)

**Files:**
- Create: `modules/01-how-ai-works/demo/package.json`
- Create: `modules/01-how-ai-works/demo/js/math.js`
- Test: `modules/01-how-ai-works/demo/js/math.test.js`

**Interfaces:**
- Produces (used by every later task):
  - `hashEmbedding(text: string, dims?: number = 16): number[]` — deterministic, unit-length pseudo-embedding.
  - `cosineSimilarity(a: number[], b: number[]): number` — throws if lengths differ.
  - `project2D(vectors: number[][]): [number, number][]` — one `[x, y]` pair per input vector.
  - `softmaxWithTemperature(logits: ArrayLike<number>, temperature?: number = 1): number[]`.
  - `topK(values: ArrayLike<number>, k: number): { index: number, value: number }[]` — descending by value.
  - `attentionWeights(tokens: string[]): number[][]` — causally-masked (row `i`, col `j`, zero for `j > i`), each row sums to 1.

- [x] **Step 1: Create the Node module config**

`modules/01-how-ai-works/demo/package.json`:
```json
{
  "type": "module",
  "private": true
}
```

- [x] **Step 2: Write the failing tests**

`modules/01-how-ai-works/demo/js/math.test.js`:
```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  hashEmbedding,
  cosineSimilarity,
  project2D,
  softmaxWithTemperature,
  topK,
  attentionWeights,
} from './math.js';

test('hashEmbedding is deterministic', () => {
  assert.deepEqual(hashEmbedding('hello'), hashEmbedding('hello'));
});

test('hashEmbedding returns a unit-length vector of the requested dims', () => {
  const v = hashEmbedding('research', 16);
  assert.equal(v.length, 16);
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  assert.ok(Math.abs(norm - 1) < 1e-6);
});

test('hashEmbedding of an empty string is the zero vector', () => {
  assert.deepEqual(hashEmbedding('', 8), new Array(8).fill(0));
});

test('cosineSimilarity of a vector with itself is 1', () => {
  const v = hashEmbedding('sky');
  assert.ok(Math.abs(cosineSimilarity(v, v) - 1) < 1e-9);
});

test('cosineSimilarity throws on mismatched lengths', () => {
  assert.throws(() => cosineSimilarity([1, 2], [1, 2, 3]));
});

test('project2D returns one [x, y] pair per input vector, deterministically', () => {
  const vectors = [hashEmbedding('cat'), hashEmbedding('dog')];
  const points = project2D(vectors);
  assert.equal(points.length, 2);
  for (const p of points) assert.equal(p.length, 2);
  assert.deepEqual(project2D(vectors), points);
});

test('softmaxWithTemperature sums to 1', () => {
  const probs = softmaxWithTemperature([1, 2, 3], 1);
  assert.ok(Math.abs(probs.reduce((a, b) => a + b, 0) - 1) < 1e-9);
});

test('softmaxWithTemperature: lower temperature sharpens the distribution', () => {
  const logits = [1, 2, 3];
  const sharp = softmaxWithTemperature(logits, 0.1);
  const flat = softmaxWithTemperature(logits, 2);
  assert.ok(Math.max(...sharp) > Math.max(...flat));
});

test('topK returns the k largest values in descending order with original indices', () => {
  const result = topK([5, 1, 9, 3], 2);
  assert.deepEqual(result.map((r) => r.value), [9, 5]);
  assert.deepEqual(result.map((r) => r.index), [2, 0]);
});

test('attentionWeights: each row sums to 1 and is causally masked', () => {
  const matrix = attentionWeights(['the', 'cat', 'sat']);
  for (let i = 0; i < matrix.length; i++) {
    const row = matrix[i];
    assert.ok(Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-9);
    for (let j = i + 1; j < row.length; j++) assert.equal(row[j], 0);
  }
});

test('attentionWeights: a repeated token gets boosted weight from its earlier occurrence', () => {
  const matrix = attentionWeights(['cat', 'sat', 'on', 'the', 'cat']);
  const lastRow = matrix[4];
  // token 0 ("cat") repeats token 4 ("cat"); token 1 ("sat") does not.
  assert.ok(lastRow[0] > lastRow[1]);
});
```

- [x] **Step 3: Run the tests to verify they fail**

Run: `node --test modules/01-how-ai-works/demo/js/math.test.js`
Expected: FAIL — `Cannot find module './math.js'` (or similar), since `math.js` doesn't exist yet.

- [x] **Step 4: Implement `math.js`**

`modules/01-how-ai-works/demo/js/math.js`:
```js
export function hashEmbedding(text, dims = 16) {
  const vector = new Array(dims).fill(0);
  const normalized = text.trim().toLowerCase();
  if (normalized.length === 0) return vector;

  const n = 3;
  const padded = `  ${normalized}  `;
  for (let i = 0; i < padded.length - n + 1; i++) {
    const gram = padded.slice(i, i + n);
    let hash = 2166136261; // FNV-1a offset basis
    for (let j = 0; j < gram.length; j++) {
      hash ^= gram.charCodeAt(j);
      hash = Math.imul(hash, 16777619);
    }
    const index = Math.abs(hash) % dims;
    const sign = (hash & 1) === 0 ? 1 : -1;
    vector[index] += sign;
  }

  const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
  if (norm === 0) return vector;
  return vector.map((v) => v / norm);
}

export function cosineSimilarity(a, b) {
  if (a.length !== b.length) {
    throw new Error('cosineSimilarity: vectors must be the same length');
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function projectionAxes(dims) {
  const axisA = [];
  const axisB = [];
  for (let i = 0; i < dims; i++) {
    axisA.push(Math.sin(i * 12.9898));
    axisB.push(Math.sin(i * 78.233 + 1));
  }
  return [axisA, axisB];
}

export function project2D(vectors) {
  if (vectors.length === 0) return [];
  const dims = vectors[0].length;
  const [axisA, axisB] = projectionAxes(dims);
  return vectors.map((v) => {
    let x = 0;
    let y = 0;
    for (let i = 0; i < dims; i++) {
      x += v[i] * axisA[i];
      y += v[i] * axisB[i];
    }
    return [x, y];
  });
}

export function softmaxWithTemperature(logits, temperature = 1) {
  const t = Math.max(temperature, 1e-4);
  const scaled = Array.from(logits, (v) => v / t);
  const max = Math.max(...scaled);
  const exps = scaled.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export function topK(values, k) {
  return Array.from(values)
    .map((value, index) => ({ index, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, k);
}

export function attentionWeights(tokens) {
  const n = tokens.length;
  const matrix = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(n).fill(0);
    let sum = 0;
    for (let j = 0; j <= i; j++) {
      const distanceScore = 1 / (1 + (i - j));
      const isRepeat = j !== i && tokens[j].trim().toLowerCase() === tokens[i].trim().toLowerCase();
      row[j] = distanceScore * (isRepeat ? 4 : 1);
      sum += row[j];
    }
    for (let j = 0; j <= i; j++) row[j] = row[j] / sum;
    matrix.push(row);
  }
  return matrix;
}
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `node --test modules/01-how-ai-works/demo/js/math.test.js`
Expected: PASS — all 11 tests green.

- [x] **Step 6: Commit**

```bash
git add modules/01-how-ai-works/demo/package.json modules/01-how-ai-works/demo/js/math.js modules/01-how-ai-works/demo/js/math.test.js
git commit -m "Add pure pipeline math module for module 1 demo, with unit tests"
```

---

### Task 2: Page shell (`index.html`, `style.css`, bootstrap `main.js`)

**Files:**
- Create: `modules/01-how-ai-works/demo/index.html`
- Create: `modules/01-how-ai-works/demo/style.css`
- Create: `modules/01-how-ai-works/demo/js/main.js`

**Interfaces:**
- Consumes: none yet (model.js doesn't exist until Task 3).
- Produces: the DOM element ids every later task's `main.js` additions and render modules rely on: `sentence-input`, `load-button`, `run-button`, `load-progress`, `load-status`, `retry-button`, `error-banner`, `unsupported-banner`, `tokens-section`, `embeddings-section`, `attention-section`, `prediction-section`.

- [x] **Step 1: Write `index.html`**

`modules/01-how-ai-works/demo/index.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>How AI Works — Interactive Pipeline Demo</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <main>
    <h1>How AI Works: Watch a Real Model Think</h1>
    <p>
      Type a sentence and watch it move through a real small language model
      (<code>distilgpt2</code>). Tokenization and next-token prediction below
      are the model's genuine output. Embeddings and Attention are
      illustrative approximations — the published model file doesn't expose
      those real internal values — see the caption on each of those panels.
    </p>

    <div id="unsupported-banner" class="banner banner-error" hidden>
      This demo needs a browser with WebAssembly support. Please try a
      recent version of Chrome, Firefox, Edge, or Safari.
    </div>

    <div id="error-banner" class="banner banner-error" hidden></div>

    <section class="controls">
      <label for="sentence-input">Your sentence</label>
      <input id="sentence-input" type="text" value="What color is the sky?" />

      <div class="load-row">
        <button id="load-button" type="button">Load the model (~100-300MB, one-time)</button>
        <progress id="load-progress" value="0" max="100" hidden></progress>
        <span id="load-status"></span>
        <button id="retry-button" type="button" hidden>Retry</button>
      </div>

      <button id="run-button" type="button" disabled>Run</button>
    </section>

    <section>
      <h2>Tokens</h2>
      <div id="tokens-section">Load the model, then click Run to see tokens.</div>
    </section>

    <section>
      <h2>Embeddings</h2>
      <div id="embeddings-section"></div>
    </section>

    <section>
      <h2>Attention</h2>
      <div id="attention-section"></div>
    </section>

    <section>
      <h2>Prediction</h2>
      <div id="prediction-section">Click Run to see next-token predictions.</div>
    </section>
  </main>

  <script type="module" src="js/main.js"></script>
</body>
</html>
```

- [x] **Step 2: Write `style.css`**

`modules/01-how-ai-works/demo/style.css`:
```css
:root {
  color-scheme: light dark;
  --fg: #1a1a1a;
  --bg: #ffffff;
  --border: #d0d0d0;
  --accent: #2f6feb;
  --error-bg: #fde8e8;
  --error-fg: #8a1f1f;
  --caption-fg: #666666;
}

@media (prefers-color-scheme: dark) {
  :root {
    --fg: #e8e8e8;
    --bg: #1a1a1a;
    --border: #444444;
    --error-bg: #3a1f1f;
    --error-fg: #ffb3b3;
    --caption-fg: #aaaaaa;
  }
}

body {
  background: var(--bg);
  color: var(--fg);
  font-family: system-ui, sans-serif;
  margin: 0;
}

main {
  max-width: 780px;
  margin: 0 auto;
  padding: 1.5rem;
}

section {
  margin-top: 1.5rem;
}

.controls {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
}

.controls input[type='text'] {
  font-size: 1rem;
  padding: 0.5rem;
}

.load-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.banner {
  padding: 0.75rem 1rem;
  border-radius: 6px;
  margin-bottom: 1rem;
}

.banner-error {
  background: var(--error-bg);
  color: var(--error-fg);
}

.illustrative-caption {
  color: var(--caption-fg);
  font-size: 0.85rem;
  font-style: italic;
}

.token-chip {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.25rem 0.5rem;
  margin: 0.2rem;
}

.token-chip-text {
  font-weight: bold;
}

.token-chip-id {
  font-size: 0.75rem;
  color: var(--caption-fg);
}

.embedding-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0.15rem 0;
}

.embedding-row-label {
  min-width: 4rem;
}

.embedding-strip {
  display: flex;
}

.embedding-cell {
  width: 10px;
  height: 18px;
}

.embedding-plot {
  width: 100%;
  max-width: 320px;
  height: 320px;
  border: 1px solid var(--border);
  margin-top: 0.5rem;
}

.embedding-plot-label {
  font-size: 8px;
  fill: var(--fg);
}

.attention-table {
  border-collapse: collapse;
  font-size: 0.8rem;
}

.attention-table th,
.attention-table td {
  border: 1px solid var(--border);
  padding: 0.2rem 0.4rem;
  text-align: center;
}

.prediction-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
  margin-bottom: 0.75rem;
}

.prediction-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0.25rem 0;
}

.prediction-token {
  min-width: 5rem;
  font-family: monospace;
}

.prediction-bar-track {
  flex: 1;
  background: var(--border);
  border-radius: 4px;
  overflow: hidden;
  height: 16px;
}

.prediction-bar-fill {
  background: var(--accent);
  height: 100%;
}

.prediction-pct {
  min-width: 3.5rem;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
```

- [x] **Step 3: Write the bootstrap `main.js`**

`modules/01-how-ai-works/demo/js/main.js`:
```js
const els = {
  input: document.getElementById('sentence-input'),
  loadButton: document.getElementById('load-button'),
  runButton: document.getElementById('run-button'),
  loadProgress: document.getElementById('load-progress'),
  loadStatus: document.getElementById('load-status'),
  errorBanner: document.getElementById('error-banner'),
  retryButton: document.getElementById('retry-button'),
  unsupportedBanner: document.getElementById('unsupported-banner'),
  tokensSection: document.getElementById('tokens-section'),
  embeddingsSection: document.getElementById('embeddings-section'),
  attentionSection: document.getElementById('attention-section'),
  predictionSection: document.getElementById('prediction-section'),
};

function updateRunButtonState() {
  els.runButton.disabled = els.input.value.trim().length === 0;
}

if (typeof WebAssembly !== 'object') {
  els.unsupportedBanner.hidden = false;
  els.loadButton.disabled = true;
} else {
  els.input.addEventListener('input', updateRunButtonState);
}
```

- [x] **Step 4: Manually verify in a browser**

Run: `cd modules/01-how-ai-works/demo && python3 -m http.server 8000`, open `http://localhost:8000/`.

Expected:
- Page loads with no console errors.
- Input box is pre-filled with "What color is the sky?".
- "Load the model" button is enabled; "Run" is disabled (Run stays disabled because the model isn't loaded — Task 3 wires that dependency in; for now it's disabled because `runButton.disabled` starts `true` in the HTML and `updateRunButtonState` only re-checks input text, not model state).
- Progress bar, error banner, retry button, and unsupported banner are all hidden.
- In devtools console, run `document.getElementById('unsupported-banner').hidden = false` — banner becomes visible and styled correctly (manual check of the render path, since real WASM-lacking browsers aren't available to test against directly).

- [x] **Step 5: Commit**

```bash
git add modules/01-how-ai-works/demo/index.html modules/01-how-ai-works/demo/style.css modules/01-how-ai-works/demo/js/main.js
git commit -m "Add module 1 demo page shell"
```

---

### Task 3: Model loading + real tokenization

**Files:**
- Create: `modules/01-how-ai-works/demo/js/model.js`
- Create: `modules/01-how-ai-works/demo/js/render-tokens.js`
- Modify: `modules/01-how-ai-works/demo/js/main.js`

**Interfaces:**
- Consumes: DOM element ids from Task 2.
- Produces (used by Tasks 4-6):
  - `model.js`: `isReady(): boolean`, `loadModel(onProgress: (percent: number) => void): Promise<void>`, `tokenize(text: string): { tokens: string[], ids: number[] }`, `predictNextTokenLogits(text: string): Promise<Float32Array>`, `decodeTokenId(id: number): string`.
  - `render-tokens.js`: `renderTokens(container: HTMLElement, tokens: string[], ids: number[]): void`.

- [x] **Step 1: Implement `model.js`**

`modules/01-how-ai-works/demo/js/model.js`:
```js
const MODEL_ID = 'Xenova/distilgpt2';
const TRANSFORMERS_CDN_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

let tokenizer = null;
let model = null;

export function isReady() {
  return tokenizer !== null && model !== null;
}

export async function loadModel(onProgress) {
  const { AutoTokenizer, AutoModelForCausalLM } = await import(TRANSFORMERS_CDN_URL);
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    progress_callback: (info) => {
      if (info.status === 'progress_total' && typeof onProgress === 'function') {
        onProgress(info.progress);
      }
    },
  });
}

function assertReady() {
  if (!isReady()) {
    throw new Error('Model is not loaded yet — call loadModel() first');
  }
}

export function tokenize(text) {
  assertReady();
  const ids = tokenizer.encode(text, { add_special_tokens: false });
  const tokens = ids.map((id) => tokenizer.decode([id]));
  return { tokens, ids };
}

export async function predictNextTokenLogits(text) {
  assertReady();
  const inputs = await tokenizer(text);
  const { logits } = await model(inputs);
  const [, seqLen, vocabSize] = logits.dims;
  const start = (seqLen - 1) * vocabSize;
  return logits.data.slice(start, start + vocabSize);
}

export function decodeTokenId(id) {
  assertReady();
  return tokenizer.decode([id]);
}
```

- [x] **Step 2: Implement `render-tokens.js`**

`modules/01-how-ai-works/demo/js/render-tokens.js`:
```js
export function renderTokens(container, tokens, ids) {
  container.innerHTML = '';
  if (tokens.length === 0) {
    container.textContent = 'No tokens yet — type something and click Run.';
    return;
  }
  tokens.forEach((token, i) => {
    const chip = document.createElement('span');
    chip.className = 'token-chip';

    const label = document.createElement('span');
    label.className = 'token-chip-text';
    label.textContent = token;

    const idLabel = document.createElement('span');
    idLabel.className = 'token-chip-id';
    idLabel.textContent = String(ids[i]);

    chip.append(label, idLabel);
    container.appendChild(chip);
  });
}
```

- [x] **Step 3: Wire loading + tokenize-on-run into `main.js`**

Replace the full contents of `modules/01-how-ai-works/demo/js/main.js` with:
```js
import * as modelApi from './model.js';
import { renderTokens } from './render-tokens.js';

const els = {
  input: document.getElementById('sentence-input'),
  loadButton: document.getElementById('load-button'),
  runButton: document.getElementById('run-button'),
  loadProgress: document.getElementById('load-progress'),
  loadStatus: document.getElementById('load-status'),
  errorBanner: document.getElementById('error-banner'),
  retryButton: document.getElementById('retry-button'),
  unsupportedBanner: document.getElementById('unsupported-banner'),
  tokensSection: document.getElementById('tokens-section'),
  embeddingsSection: document.getElementById('embeddings-section'),
  attentionSection: document.getElementById('attention-section'),
  predictionSection: document.getElementById('prediction-section'),
};

function showError(message) {
  els.errorBanner.textContent = message;
  els.errorBanner.hidden = false;
}

function clearError() {
  els.errorBanner.hidden = true;
  els.errorBanner.textContent = '';
}

function updateRunButtonState() {
  els.runButton.disabled = !modelApi.isReady() || els.input.value.trim().length === 0;
}

async function handleLoad() {
  clearError();
  els.loadButton.disabled = true;
  els.retryButton.hidden = true;
  els.loadStatus.textContent = 'Downloading model (one-time, ~100-300MB, cached after)...';
  els.loadProgress.hidden = false;
  els.loadProgress.value = 0;
  try {
    await modelApi.loadModel((percent) => {
      els.loadProgress.value = percent;
    });
    els.loadStatus.textContent = 'Model ready.';
    els.loadButton.hidden = true;
    updateRunButtonState();
  } catch (err) {
    console.error(err);
    showError('Could not download the model. Check your connection and retry.');
    els.loadButton.disabled = false;
    els.retryButton.hidden = false;
  }
}

async function handleRun() {
  clearError();
  const text = els.input.value;
  els.runButton.disabled = true;
  els.runButton.textContent = 'Running…';
  try {
    const { tokens, ids } = modelApi.tokenize(text);
    renderTokens(els.tokensSection, tokens, ids);
  } catch (err) {
    console.error(err);
    showError('Something went wrong running the model. See the console for details.');
  } finally {
    els.runButton.textContent = 'Run';
    updateRunButtonState();
  }
}

if (typeof WebAssembly !== 'object') {
  els.unsupportedBanner.hidden = false;
  els.loadButton.disabled = true;
} else {
  els.input.addEventListener('input', updateRunButtonState);
  els.loadButton.addEventListener('click', handleLoad);
  els.retryButton.addEventListener('click', handleLoad);
  els.runButton.addEventListener('click', handleRun);
}
```

- [x] **Step 4: Manually verify in a browser**

Run: `cd modules/01-how-ai-works/demo && python3 -m http.server 8000`, open `http://localhost:8000/`.

Expected:
- Click "Load the model" → progress bar animates from 0 to 100, status changes to "Model ready.", Load button disappears, Run button becomes enabled (since input isn't empty).
- Type over the input so it's empty → Run becomes disabled again; type something back in → Run re-enables.
- Click "Run" with "What color is the sky?" in the box → token chips appear, each showing real BPE token text (e.g. `What`, ` color`, ` is`, ` the`, ` sky`, `?`) and its real integer id.
- Open devtools Network tab, throttle to "Offline", reload, click "Load the model" → error banner appears with the retry message, Retry button is visible; switch back online and click Retry → succeeds.

- [x] **Step 5: Commit**

```bash
git add modules/01-how-ai-works/demo/js/model.js modules/01-how-ai-works/demo/js/render-tokens.js modules/01-how-ai-works/demo/js/main.js
git commit -m "Load distilgpt2 via transformers.js and render real tokens"
```

---

### Task 4: Real next-token prediction (temperature + step)

**Files:**
- Create: `modules/01-how-ai-works/demo/js/render-prediction.js`
- Modify: `modules/01-how-ai-works/demo/js/main.js`

**Interfaces:**
- Consumes: `math.js`'s `softmaxWithTemperature`/`topK` (Task 1), `model.js`'s `predictNextTokenLogits`/`decodeTokenId` (Task 3).
- Produces: `render-prediction.js`: `renderPrediction(container: HTMLElement, { logits: Float32Array|null, decodeTokenId: (id:number)=>string, temperature: number, onTemperatureChange: (t:number)=>void, onStep: ()=>void }): void`.

- [x] **Step 1: Implement `render-prediction.js`**

`modules/01-how-ai-works/demo/js/render-prediction.js`:
```js
import { softmaxWithTemperature, topK } from './math.js';

export function renderPrediction(container, { logits, decodeTokenId, temperature, onTemperatureChange, onStep }) {
  container.innerHTML = '';

  if (!logits) {
    container.textContent = 'Click Run to see next-token predictions.';
    return;
  }

  const controls = document.createElement('div');
  controls.className = 'prediction-controls';

  const sliderLabel = document.createElement('label');
  sliderLabel.textContent = `Temperature: ${temperature.toFixed(2)} `;
  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = '0.1';
  slider.max = '1.5';
  slider.step = '0.05';
  slider.value = String(temperature);
  slider.addEventListener('input', () => {
    onTemperatureChange(Number(slider.value));
  });
  sliderLabel.appendChild(slider);
  controls.appendChild(sliderLabel);

  const stepButton = document.createElement('button');
  stepButton.type = 'button';
  stepButton.textContent = 'Step: accept top token and continue';
  stepButton.addEventListener('click', onStep);
  controls.appendChild(stepButton);

  container.appendChild(controls);

  const probs = softmaxWithTemperature(logits, temperature);
  const top5 = topK(probs, 5);

  const list = document.createElement('div');
  list.className = 'prediction-bars';
  top5.forEach(({ index, value }) => {
    const row = document.createElement('div');
    row.className = 'prediction-row';

    const label = document.createElement('span');
    label.className = 'prediction-token';
    label.textContent = decodeTokenId(index);

    const track = document.createElement('div');
    track.className = 'prediction-bar-track';
    const fill = document.createElement('div');
    fill.className = 'prediction-bar-fill';
    fill.style.width = `${Math.round(value * 100)}%`;
    track.appendChild(fill);

    const pct = document.createElement('span');
    pct.className = 'prediction-pct';
    pct.textContent = `${(value * 100).toFixed(1)}%`;

    row.append(label, track, pct);
    list.appendChild(row);
  });
  container.appendChild(list);
}
```

- [x] **Step 2: Wire prediction + temperature + step into `main.js`**

In `modules/01-how-ai-works/demo/js/main.js`:

Add to the imports at the top:
```js
import { renderPrediction } from './render-prediction.js';
import { softmaxWithTemperature, topK } from './math.js';
```

Add a `state` object right after the `els = {...}` block:
```js
const state = {
  temperature: 0.7,
  currentText: '',
  lastLogits: null,
};
```

Add a new function, and call it from `handleRun`:
```js
function renderPredictionSection() {
  renderPrediction(els.predictionSection, {
    logits: state.lastLogits,
    decodeTokenId: modelApi.decodeTokenId,
    temperature: state.temperature,
    onTemperatureChange: (value) => {
      state.temperature = value;
      renderPredictionSection();
    },
    onStep: async () => {
      if (!state.lastLogits) return;
      const probs = softmaxWithTemperature(state.lastLogits, state.temperature);
      const [{ index }] = topK(probs, 1);
      const nextText = modelApi.decodeTokenId(index);
      els.input.value = state.currentText + nextText;
      await handleRun();
    },
  });
}
```

Modify `handleRun`'s try block to also record `state.currentText`, run the real prediction, and render it — the full function becomes:
```js
async function handleRun() {
  clearError();
  const text = els.input.value;
  els.runButton.disabled = true;
  els.runButton.textContent = 'Running…';
  try {
    const { tokens, ids } = modelApi.tokenize(text);
    renderTokens(els.tokensSection, tokens, ids);

    state.currentText = text;
    state.lastLogits = await modelApi.predictNextTokenLogits(text);
    renderPredictionSection();
  } catch (err) {
    console.error(err);
    showError('Something went wrong running the model. See the console for details.');
  } finally {
    els.runButton.textContent = 'Run';
    updateRunButtonState();
  }
}
```

- [x] **Step 3: Manually verify in a browser**

Run: `cd modules/01-how-ai-works/demo && python3 -m http.server 8000`, open `http://localhost:8000/`.

Expected:
- Load the model, click Run with "What color is the sky?" → Prediction panel shows a temperature slider (starting at 0.70), a Step button, and 5 probability bars with real tokens and percentages that sum to roughly 100%. A word like " blue" should plausibly be among the top candidates.
- Drag the temperature slider → bars re-render live (higher temperature flattens the bars, lower temperature sharpens them), without re-running the model (should be instant, no network/compute delay).
- Click "Step: accept top token and continue" repeatedly → the input box grows with each accepted token, tokens/prediction panels refresh each time, generation stays coherent for at least a few steps.

- [x] **Step 4: Commit**

```bash
git add modules/01-how-ai-works/demo/js/render-prediction.js modules/01-how-ai-works/demo/js/main.js
git commit -m "Add real autoregressive next-token prediction with temperature and step"
```

---

### Task 5: Illustrative embeddings panel

**Files:**
- Create: `modules/01-how-ai-works/demo/js/render-embeddings.js`
- Modify: `modules/01-how-ai-works/demo/js/main.js`

**Interfaces:**
- Consumes: `math.js`'s `hashEmbedding`/`project2D` (Task 1).
- Produces: `render-embeddings.js`: `renderEmbeddings(container: HTMLElement, tokens: string[]): void`.

- [x] **Step 1: Implement `render-embeddings.js`**

`modules/01-how-ai-works/demo/js/render-embeddings.js`:
```js
import { hashEmbedding, project2D, cosineSimilarity } from './math.js';

export function renderEmbeddings(container, tokens) {
  container.innerHTML = '';

  const caption = document.createElement('p');
  caption.className = 'illustrative-caption';
  caption.textContent =
    "Illustrative — these vectors are a simplified hash of each token's spelling, " +
    "not distilgpt2's real embeddings. Tokens with similar spelling cluster together " +
    'here; real embeddings cluster by meaning.';
  container.appendChild(caption);

  if (tokens.length === 0) return;

  const vectors = tokens.map((t) => hashEmbedding(t));

  const strips = document.createElement('div');
  strips.className = 'embedding-strips';
  tokens.forEach((token, i) => {
    const row = document.createElement('div');
    row.className = 'embedding-row';

    const label = document.createElement('span');
    label.className = 'embedding-row-label';
    label.textContent = token;

    const strip = document.createElement('div');
    strip.className = 'embedding-strip';
    vectors[i].forEach((value) => {
      const cell = document.createElement('span');
      cell.className = 'embedding-cell';
      const intensity = Math.round(((value + 1) / 2) * 255);
      cell.style.backgroundColor = `rgb(${255 - intensity}, ${intensity}, 160)`;
      strip.appendChild(cell);
    });

    row.append(label, strip);
    strips.appendChild(row);
  });
  container.appendChild(strips);

  const points = project2D(vectors);
  const svgNS = 'http://www.w3.org/2000/svg';
  const plot = document.createElementNS(svgNS, 'svg');
  plot.setAttribute('class', 'embedding-plot');
  plot.setAttribute('viewBox', '0 0 200 200');

  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const scaleX = (x) => (maxX === minX ? 100 : 10 + ((x - minX) / (maxX - minX)) * 180);
  const scaleY = (y) => (maxY === minY ? 100 : 10 + ((y - minY) / (maxY - minY)) * 180);

  points.forEach(([x, y], i) => {
    const label = document.createElementNS(svgNS, 'text');
    label.setAttribute('x', String(scaleX(x)));
    label.setAttribute('y', String(scaleY(y)));
    label.setAttribute('class', 'embedding-plot-label');
    label.textContent = tokens[i].trim() || '·';
    plot.appendChild(label);
  });
  container.appendChild(plot);

  if (tokens.length > 1) {
    let best = { i: 0, j: 1, score: -Infinity };
    for (let i = 0; i < vectors.length; i++) {
      for (let j = i + 1; j < vectors.length; j++) {
        const score = cosineSimilarity(vectors[i], vectors[j]);
        if (score > best.score) best = { i, j, score };
      }
    }
    const summary = document.createElement('p');
    summary.className = 'illustrative-caption';
    summary.textContent = `Most similar pair in this simplified space: "${tokens[best.i]}" and "${tokens[best.j]}" (cosine similarity ${best.score.toFixed(2)}).`;
    container.appendChild(summary);
  }
}
```

- [x] **Step 2: Wire it into `main.js`**

Add to the imports at the top of `modules/01-how-ai-works/demo/js/main.js`:
```js
import { renderEmbeddings } from './render-embeddings.js';
```

In `handleRun`, add the call right after the existing `renderTokens(...)` line:
```js
    const { tokens, ids } = modelApi.tokenize(text);
    renderTokens(els.tokensSection, tokens, ids);
    renderEmbeddings(els.embeddingsSection, tokens);
```

- [x] **Step 3: Manually verify in a browser**

Run: `cd modules/01-how-ai-works/demo && python3 -m http.server 8000`, open `http://localhost:8000/`.

Expected:
- Click Run → Embeddings panel shows the italic illustrative caption, one color-coded strip per token, an SVG scatter plot with each token's text positioned in it, and a "Most similar pair" line underneath.
- Type a sentence with a repeated word (e.g. "the cat sat near the cat") and click Run → the two occurrences of "cat" and "the" land at or very near the same point in the plot, and the "Most similar pair" line calls out one of those repeated-word pairs with a similarity score near 1.00 (since `hashEmbedding` is deterministic per exact token text).

- [x] **Step 4: Commit**

```bash
git add modules/01-how-ai-works/demo/js/render-embeddings.js modules/01-how-ai-works/demo/js/main.js
git commit -m "Add illustrative embeddings panel to module 1 demo"
```

---

### Task 6: Illustrative attention panel

**Files:**
- Create: `modules/01-how-ai-works/demo/js/render-attention.js`
- Modify: `modules/01-how-ai-works/demo/js/main.js`

**Interfaces:**
- Consumes: `math.js`'s `attentionWeights` (Task 1).
- Produces: `render-attention.js`: `renderAttention(container: HTMLElement, tokens: string[]): void`.

- [x] **Step 1: Implement `render-attention.js`**

`modules/01-how-ai-works/demo/js/render-attention.js`:
```js
import { attentionWeights } from './math.js';

export function renderAttention(container, tokens) {
  container.innerHTML = '';

  const caption = document.createElement('p');
  caption.className = 'illustrative-caption';
  caption.textContent =
    "Illustrative — this heatmap is a distance + repetition heuristic, not distilgpt2's " +
    'real attention weights.';
  container.appendChild(caption);

  if (tokens.length === 0) return;

  const matrix = attentionWeights(tokens);
  const table = document.createElement('table');
  table.className = 'attention-table';

  const headerRow = document.createElement('tr');
  headerRow.appendChild(document.createElement('th'));
  tokens.forEach((token) => {
    const th = document.createElement('th');
    th.textContent = token;
    headerRow.appendChild(th);
  });
  table.appendChild(headerRow);

  matrix.forEach((row, i) => {
    const tr = document.createElement('tr');
    const rowHeader = document.createElement('th');
    rowHeader.textContent = tokens[i];
    tr.appendChild(rowHeader);

    row.forEach((weight) => {
      const td = document.createElement('td');
      const intensity = Math.round(weight * 255);
      td.style.backgroundColor = `rgb(${255 - intensity}, ${255 - Math.round(intensity / 2)}, 255)`;
      td.textContent = weight > 0 ? weight.toFixed(2) : '';
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });

  container.appendChild(table);
}
```

- [x] **Step 2: Wire it into `main.js`**

Add to the imports at the top of `modules/01-how-ai-works/demo/js/main.js`:
```js
import { renderAttention } from './render-attention.js';
```

In `handleRun`, add the call right after `renderEmbeddings(...)`:
```js
    renderEmbeddings(els.embeddingsSection, tokens);
    renderAttention(els.attentionSection, tokens);
```

- [x] **Step 3: Manually verify in a browser**

Run: `cd modules/01-how-ai-works/demo && python3 -m http.server 8000`, open `http://localhost:8000/`.

Expected:
- Click Run → Attention panel shows the italic illustrative caption and a token-by-token heatmap table; cells above the diagonal (later tokens attending to earlier ones is fine, but earlier tokens should show nothing for later columns) are blank, confirming the causal mask.
- Type "the cat sat near the cat" and click Run → the last row ("cat") shows a visibly darker cell under the earlier "cat" column than under neighboring non-repeated columns, reflecting the repetition heuristic.

- [x] **Step 4: Commit**

```bash
git add modules/01-how-ai-works/demo/js/render-attention.js modules/01-how-ai-works/demo/js/main.js
git commit -m "Add illustrative attention panel to module 1 demo"
```

---

### Task 7: Full error-handling & edge-case pass

**Files:**
- Modify: `modules/01-how-ai-works/demo/js/main.js` (only if a gap is found in Step 1-4 below — see Step 5)

**Interfaces:** none new — this task exercises the full assembled app from Tasks 2-6.

- [x] **Step 1: Verify empty-input handling**

In the browser, load the model, then clear the input box entirely.
Expected: Run button becomes disabled immediately (no need to click anything else).

- [x] **Step 2: Verify slow/in-flight state doesn't look hung**

With the model loaded, click Run.
Expected: Run button text changes to "Running…" and the button is disabled for the (brief) duration of tokenize + predict, then returns to "Run" and re-enables.

- [x] **Step 3: Verify download failure + retry**

In devtools, open the Network tab, set throttling to "Offline". Reload the page and click "Load the model".
Expected: error banner reads "Could not download the model. Check your connection and retry.", the Retry button is visible, the Load button is re-enabled (not stuck disabled).
Then set throttling back to "No throttling" (or "Online") and click Retry.
Expected: loads successfully, error banner clears, progress bar completes, Run enables.

- [x] **Step 4: Verify the unsupported-browser path renders correctly**

In devtools console: `document.getElementById('unsupported-banner').hidden = false`.
Expected: banner is visible with readable styling in both light and dark OS theme (toggle OS/browser dark mode and reload to check the dark-mode CSS variables from Task 2's `style.css`).

- [x] **Step 5: Fix anything broken**

If any of Steps 1-4 fail, fix the relevant code in `main.js` (the logic for all four cases was written in Tasks 2-4; this task is the first point at which they're exercised against the fully-assembled app). Re-run the failing step until it passes. If no fixes were needed, skip the commit in Step 6.

> **Done 2026-09-30:** Step 3 failed — Retry could never recover in-page, because the browser caches a failed `import()` of the transformers.js URL and transformers.js 4.2.0 memoizes a network failure as "model file doesn't exist" for the life of the page. Fixed in `main.js`: Retry now reloads the page and resumes the download automatically, keeping the typed sentence. Steps 1–4 re-verified in headless Chrome (offline, and model-host-only blocked), along with the dark-mode banner.

- [x] **Step 6: Commit (only if Step 5 required changes)**

```bash
git add modules/01-how-ai-works/demo/js/main.js
git commit -m "Fix edge-case handling found during module 1 demo verification pass"
```

---

### Task 8: README link, GitHub Pages, final walkthrough

**Files:**
- Modify: `modules/01-how-ai-works/README.md`

**Interfaces:** none — documentation and hosting only.

- [x] **Step 1: Add a "Try it" section to the module README**

In `modules/01-how-ai-works/README.md`, add this section right after the "## Hands-on exercise" section (before "## Key takeaways"):
```markdown
## Try it: interactive pipeline demo

Want to see the mechanism above happen for real? There's a small in-browser
demo that runs an actual small language model (`distilgpt2`) on a sentence
you type, showing real tokenization and real next-token prediction (the
embeddings and attention panels are clearly-labeled illustrative
approximations — see the caption on each):

**[Open the interactive demo](https://jbrzensk.github.io/AI_for_SIO/modules/01-how-ai-works/demo/)**

First load takes a one-time ~100-300MB download (cached by your browser
afterward). No data leaves your browser — everything runs locally.
```

- [x] **Step 2: Enable GitHub Pages — requires the repo owner's confirmation**

This is a repository-visibility setting change (it makes the repo's static files reachable at a public URL) and should be done by the repo owner, not automated silently. In the GitHub web UI: **Settings → Pages → Source: "Deploy from a branch" → Branch: `main`, folder: `/ (root)` → Save.** GitHub will publish at `https://jbrzensk.github.io/AI_for_SIO/` within a few minutes; confirm the demo is reachable at `https://jbrzensk.github.io/AI_for_SIO/modules/01-how-ai-works/demo/`.

- [x] **Step 3: Full golden-path walkthrough on the published Pages URL**

Once Pages is live, open the real published URL (not localhost) and repeat:
- Load the model → progress completes → Run enables.
- Type "The trophy didn't fit in the suitcase because it was too big." and click Run → tokens, illustrative embeddings, illustrative attention, and real top-5 predictions all populate.
- Drag the temperature slider → bars update live.
- Click Step three or four times → generation continues and the input box grows accordingly.
- Confirm both illustrative panels show their captions on the live page (not just localhost).

- [x] **Step 4: Commit**

```bash
git add modules/01-how-ai-works/README.md
git commit -m "Link the interactive pipeline demo from module 1's README"
```
