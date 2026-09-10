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
