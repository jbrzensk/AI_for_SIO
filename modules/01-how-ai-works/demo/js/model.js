const MODEL_ID = 'Xenova/distilgpt2';
const TRANSFORMERS_CDN_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

let tokenizer = null;
let model = null;

export function isReady() {
  return tokenizer !== null && model !== null;
}

export async function loadModel(onProgress) {
  const { AutoTokenizer, AutoModelForCausalLM } = await import(TRANSFORMERS_CDN_URL);
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    progress_callback: (info) => {
      if (info.status === 'progress_total' && typeof onProgress === 'function') {
        onProgress(info.progress);
      }
    },
  });
}

function assertReady() {
  if (!isReady()) {
    throw new Error('Model is not loaded yet — call loadModel() first');
  }
}

export function tokenize(text) {
  assertReady();
  const ids = tokenizer.encode(text, { add_special_tokens: false });
  const tokens = ids.map((id) => tokenizer.decode([id]));
  return { tokens, ids };
}

export async function predictNextTokenLogits(text) {
  assertReady();
  const inputs = await tokenizer(text);
  const { logits } = await model(inputs);
  const [, seqLen, vocabSize] = logits.dims;
  const start = (seqLen - 1) * vocabSize;
  return logits.data.slice(start, start + vocabSize);
}

export function decodeTokenId(id) {
  assertReady();
  return tokenizer.decode([id]);
}
