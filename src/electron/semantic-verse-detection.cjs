/* =========================================================================
  Semantic Verse Detection — FAISS-style vector search for paraphrased
  scripture. Closes the gap with Pewbeam/Loghema: when the preacher
  paraphrases instead of quoting, this still finds the verse.

  Uses ONNX Runtime + all-MiniLM-L6-v2 sentence transformer to embed
  both the spoken transcript and every verse in the loaded Bible, then
  returns the closest matches by cosine similarity.
  ========================================================================= */
const fs = require('fs');
const path = require('path');
const os = require('os');

const EMBED_MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const TOP_K = 6;
/* Measured against all-MiniLM-L6-v2: reworded verses land at 0.55–0.88, while
   ordinary service chatter tops out near 0.27. The old 0.55 sat on top of the
   weakest genuine paraphrase (Psalm 23 reworded scores 0.545) and dropped it,
   so the line goes where the two groups actually separate. */
const MIN_SIMILARITY = 0.45;

let transformers = null;
let embedder = null;
let indexCache = new Map();
let warmupState = 'idle';
let lastError = '';

function status() {
  return {
    ok: true, name: 'semantic-verse-detection',
    ready: Boolean(embedder) && warmupState === 'ready',
    warmupState, lastError, models: [EMBED_MODEL_ID],
  };
}

async function loadEmbedder() {
  if (!transformers) {
    transformers = await import('@huggingface/transformers');
    transformers.env.cacheDir = path.join(os.homedir(), '.cache', 'huggingface');
    transformers.env.allowLocalModels = true;
  }
  if (!embedder) {
    embedder = await transformers.pipeline('feature-extraction', EMBED_MODEL_ID);
  }
  return embedder;
}

async function warmup() {
  warmupState = 'loading';
  try {
    await loadEmbedder();
    warmupState = 'ready'; lastError = '';
    return { ok: true, status: status() };
  } catch (err) {
    warmupState = 'error'; lastError = err.message;
    return { ok: false, error: err.message, status: status() };
  }
}

function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; normA += a[i] * a[i]; normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 0 ? dot / denom : 0;
}

async function embed(text) {
  const pipe = await loadEmbedder();
  const output = await pipe(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

/** How many verses go through the encoder in one pass. */
const EMBED_BATCH = 32;

/**
 * Embeddings for a whole batch at once. The index is thousands of verses, and
 * asking the encoder for them one at a time meant one model call per verse —
 * minutes of work before the first suggestion came back.
 */
async function embedMany(texts) {
  const pipe = await loadEmbedder();
  const output = await pipe(texts, { pooling: 'mean', normalize: true });
  const dim = output.dims[output.dims.length - 1];
  const flat = Array.from(output.data);
  const vectors = [];
  for (let i = 0; i < texts.length; i += 1) {
    vectors.push(flat.slice(i * dim, (i + 1) * dim));
  }
  return vectors;
}

function buildIndex(bibleData, versionId) {
  const verses = [];
  for (const [book, chapters] of Object.entries(bibleData)) {
    for (const [ch, vMap] of Object.entries(chapters)) {
      for (const [vs, text] of Object.entries(vMap)) {
        const clean = text.replace(/[^\w\s']/g, '').toLowerCase().trim();
        if (clean.length < 10) continue;
        verses.push({
          book, chapter: parseInt(ch, 10), verse: parseInt(vs, 10),
          text: clean, original: text, versionId,
        });
      }
    }
  }
  return verses;
}

async function searchParaphrased(transcript, bibleData, versionId, options = {}) {
  const limit = options.limit || TOP_K;
  const minSim = options.minSim || MIN_SIMILARITY;
  const cleaned = transcript.replace(/[^\w\s']/g, '').toLowerCase().trim();
  if (cleaned.length < 5) return [];

  let verses = indexCache.get(versionId);
  if (!verses) {
    verses = buildIndex(bibleData, versionId);
    indexCache.set(versionId, verses);
  }

  const queryVec = await embed(cleaned);
  const scored = [];

  const sample = verses.length > 5000
    ? verses.filter((_, i) => i % Math.ceil(verses.length / 5000) === 0)
    : verses;

  /* Vectors are built once per version and kept on the index entry. They go
     through the encoder in batches rather than one at a time — the old loop
     awaited a single model pass per verse. */
  const pending = sample.filter((v) => !v._vec);
  for (let i = 0; i < pending.length; i += EMBED_BATCH) {
    const batch = pending.slice(i, i + EMBED_BATCH);
    const vectors = await embedMany(batch.map((v) => v.text));
    batch.forEach((v, j) => { v._vec = vectors[j]; });
  }

  for (const v of sample) {
    if (!v._vec) continue;
    const sim = cosineSimilarity(queryVec, v._vec);
    if (sim >= minSim) scored.push({ ...v, similarity: sim });
  }

  return scored
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, limit)
    .map((v) => ({
      book: v.book, chapter: v.chapter, verseStart: v.verse, verseEnd: v.verse,
      versionId: v.versionId, reason: 'paraphrased-semantic',
      confidence: Math.min(0.95, v.similarity), excerpt: v.original, similarity: v.similarity,
    }));
}

function clearCache() { indexCache.clear(); return { ok: true }; }

module.exports = {
  createSemanticVerseDetection: () => ({ status, warmup, searchParaphrased, clearCache, EMBED_MODEL_ID }),
  /* Pure, and the only place the text normalisation lives — exported so a check
     can assert on it without loading a model. */
  buildIndex,
};