import * as modelApi from './model.js';
import { renderTokens } from './render-tokens.js';
import { renderPrediction } from './render-prediction.js';
import { softmaxWithTemperature, topK, sampleWeighted } from './math.js';
import { renderEmbeddings } from './render-embeddings.js';
import { renderAttention } from './render-attention.js';

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

const state = {
  temperature: 0.7,
  currentText: '',
  lastLogits: null,
};

let isRunning = false;

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
    els.loadProgress.hidden = true;
  } catch (err) {
    console.error(err);
    showError('Could not download the model. Check your connection and retry.');
    els.loadStatus.textContent = '';
    els.loadProgress.hidden = true;
    els.loadButton.disabled = false;
    els.retryButton.hidden = false;
  }
}

// Grow the textarea to fit its content, so tokens appended by Step or a
// candidate pick (including whitespace-only ones like a newline) are always
// visible rather than scrolled out of a fixed two-row box.
function autoGrowInput() {
  const input = els.input;
  input.style.height = 'auto';
  input.style.height = `${input.scrollHeight + input.offsetHeight - input.clientHeight}px`;
}

// Disable/enable every button in the prediction panel (Step + the candidate
// rows) so rapid clicks can't launch overlapping inference calls.
function setPredictionButtonsDisabled(disabled) {
  els.predictionSection.querySelectorAll('button').forEach((button) => {
    button.disabled = disabled;
  });
}

// Shared by Step (sampled index) and clicking a candidate (chosen index):
// append that token's real text to the sequence and re-run.
async function appendTokenAndRun(index) {
  if (isRunning || !state.lastLogits) return;
  els.input.value = state.currentText + modelApi.decodeTokenId(index);
  autoGrowInput();
  await handleRun();
}

function renderPredictionSection() {
  renderPrediction(els.predictionSection, {
    logits: state.lastLogits,
    decodeTokenId: modelApi.decodeTokenId,
    temperature: state.temperature,
    disabled: isRunning,
    onTemperatureChange: (value) => {
      state.temperature = value;
      renderPredictionSection();
    },
    onStep: () => {
      if (!state.lastLogits) return;
      // Sample from the same top-5 candidates shown in the bars, weighted by
      // their temperature-adjusted probability — never always the single top
      // token. Deterministic argmax has no way out of a repetition loop (a
      // small model's next most likely token after one newline is often
      // another newline); weighted sampling gives lower-ranked candidates a
      // real (temperature-controlled) chance, which is also what the
      // temperature slider is teaching in the first place. When the model is
      // near-certain (e.g. 99.8% on another newline), sampling alone can't
      // escape — that's what picking a candidate directly is for.
      const probs = softmaxWithTemperature(state.lastLogits, state.temperature);
      const candidates = topK(probs, 5);
      const { index } = sampleWeighted(candidates);
      return appendTokenAndRun(index);
    },
    onPick: (index) => appendTokenAndRun(index),
  });
}

async function handleRun() {
  if (isRunning) return;
  isRunning = true;
  clearError();
  const text = els.input.value;
  els.runButton.disabled = true;
  els.runButton.textContent = 'Running…';
  // Disable Step and the candidate rows (if they exist from a prior render)
  // for the duration of this run, so rapid clicks can't launch overlapping
  // inference calls while we're awaiting the model below.
  setPredictionButtonsDisabled(true);
  try {
    const { tokens, ids } = modelApi.tokenize(text);
    renderTokens(els.tokensSection, tokens, ids);
    renderEmbeddings(els.embeddingsSection, tokens);
    renderAttention(els.attentionSection, tokens);

    state.currentText = text;
    state.lastLogits = await modelApi.predictNextTokenLogits(text);
    renderPredictionSection();
  } catch (err) {
    console.error(err);
    showError('Something went wrong running the model. See the console for details.');
  } finally {
    isRunning = false;
    els.runButton.textContent = 'Run';
    updateRunButtonState();
    // Re-enable Step and the candidate rows — both the fresh ones a successful
    // run just rendered (built disabled, since isRunning was still true) and,
    // on failure, the old ones disabled above that never got rebuilt.
    setPredictionButtonsDisabled(false);
  }
}

function handleInputChange() {
  autoGrowInput();
  updateRunButtonState();
  // If the text no longer matches what the current predictions were computed
  // for, drop the stale predictions rather than let a Step silently discard
  // the user's edit (see final review Important #3). Clearing lastLogits makes
  // renderPrediction fall back to its "Click Run..." placeholder, which also
  // removes the Step button, so Step is naturally unavailable until re-run.
  if (state.lastLogits && els.input.value !== state.currentText) {
    state.lastLogits = null;
    renderPredictionSection();
  }
}

if (typeof WebAssembly !== 'object') {
  els.unsupportedBanner.hidden = false;
  els.loadButton.disabled = true;
} else {
  els.input.addEventListener('input', handleInputChange);
  els.loadButton.addEventListener('click', handleLoad);
  els.retryButton.addEventListener('click', handleLoad);
  els.runButton.addEventListener('click', handleRun);
}
