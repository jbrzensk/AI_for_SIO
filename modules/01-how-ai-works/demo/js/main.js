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
