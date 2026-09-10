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
