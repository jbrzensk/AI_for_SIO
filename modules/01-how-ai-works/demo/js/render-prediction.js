import { softmaxWithTemperature, topK } from './math.js';
import { visibleLabel } from './visible-label.js';

export function renderPrediction(container, { logits, decodeTokenId, temperature, disabled, onTemperatureChange, onStep, onPick }) {
  if (!logits) {
    container.innerHTML = '';
    container._predictionLogitsRef = null;
    container.textContent = 'Click Run to see next-token predictions.';
    return;
  }

  // Only rebuild the controls subtree (slider + Step button) when the container
  // is otherwise empty (first render after a Run) or `logits` is a new object
  // (a new Run happened). A pure temperature change re-renders with the SAME
  // logits reference, so this branch is skipped and the slider/button elements
  // the user may be actively interacting with are never removed/recreated.
  const isFreshRender = container._predictionLogitsRef !== logits;

  if (isFreshRender) {
    container.innerHTML = '';
    container._predictionLogitsRef = logits;

    const controls = document.createElement('div');
    controls.className = 'prediction-controls';

    const sliderLabel = document.createElement('label');
    const sliderLabelText = document.createElement('span');
    sliderLabelText.className = 'prediction-temp-label';
    sliderLabelText.textContent = `Temperature: ${temperature.toFixed(2)} `;
    sliderLabel.appendChild(sliderLabelText);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.className = 'prediction-temp-slider';
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
    stepButton.className = 'prediction-step-button';
    stepButton.textContent = 'Step: sample a token and continue';
    stepButton.addEventListener('click', onStep);
    controls.appendChild(stepButton);

    container.appendChild(controls);

    const hint = document.createElement('p');
    hint.className = 'prediction-hint';
    hint.textContent =
      'Click a candidate to choose the next token yourself, or press Step to let the model sample one.';
    container.appendChild(hint);

    const list = document.createElement('div');
    list.className = 'prediction-bars';
    container.appendChild(list);
  } else {
    // Temperature-only change: update the label text in place. Do NOT touch
    // the slider's value here — it already reflects what the user is dragging,
    // and this whole branch is reached specifically to avoid recreating it.
    const sliderLabelText = container.querySelector('.prediction-temp-label');
    if (sliderLabelText) {
      sliderLabelText.textContent = `Temperature: ${temperature.toFixed(2)} `;
    }
  }

  // The bars subtree always gets torn down and rebuilt — this is cheap and
  // never affects the slider or Step button elements.
  const list = container.querySelector('.prediction-bars');
  list.innerHTML = '';

  const probs = softmaxWithTemperature(logits, temperature);
  const top5 = topK(probs, 5);

  top5.forEach(({ index, value }) => {
    // Each candidate is a real button so the learner can pick the next token
    // themselves (by mouse or keyboard) — e.g. to escape a run of newlines
    // that sampling alone would keep choosing.
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'prediction-row';
    row.disabled = Boolean(disabled);
    row.addEventListener('click', () => onPick(index));

    const label = document.createElement('span');
    label.className = 'prediction-token';
    label.textContent = visibleLabel(decodeTokenId(index));

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
}
