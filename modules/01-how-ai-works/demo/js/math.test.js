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
