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
