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

  const wrapper = document.createElement('div');
  wrapper.className = 'attention-table-wrapper';
  wrapper.appendChild(table);
  container.appendChild(wrapper);
}
