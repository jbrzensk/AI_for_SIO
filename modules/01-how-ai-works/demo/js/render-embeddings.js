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
