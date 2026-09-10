import * as modelApi from './model.js';
import { renderTokens } from './render-tokens.js';
import { renderPrediction } from './render-prediction.js';
import { softmaxWithTemperature, topK } from './math.js';
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

async function handleRun() {
  if (isRunning) return;
  isRunning = true;
  clearError();
  const text = els.input.value;
  els.runButton.disabled = true;
  els.runButton.textContent = 'Running…';
  // Disable the Step button (if it currently exists from a prior render) for
  // the duration of this run, so rapid Step clicks can't launch overlapping
  // inference calls while we're awaiting the model below.
  const existingStepButton = els.predictionSection.querySelector('.prediction-step-button');
  if (existingStepButton) existingStepButton.disabled = true;
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
    // Re-enable the Step button. On success this is a no-op (renderPredictionSection
    // already built a fresh, enabled button); on failure this restores the button
    // that was disabled above and never got rebuilt.
    const stepButton = els.predictionSection.querySelector('.prediction-step-button');
    if (stepButton) stepButton.disabled = false;
  }
}

function handleInputChange() {
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
