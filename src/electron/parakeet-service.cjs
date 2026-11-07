/* =========================================================================
  NVIDIA Parakeet TDT 0.6B v3 — ONNX speech recognition
  -------------------------------------------------------------------------
  Most accurate open-weight model (6.32% WER, Hugging Face Open ASR
  Leaderboard Oct 2026). Beats Whisper Large V3 Turbo on 7/8 English
  test sets. Runs ~20x real time on CPU. 25 European languages.

  Uses int8 ONNX export — 409 MB, plain ONNX Runtime, no PyTorch.
  Model: efederici/parakeet-tdt-0.6b-v3-onnx-int4
  ========================================================================= */
const fs = require('fs');
const path = require('path');
const os = require('os');

const MODEL_ID = 'efederici/parakeet-tdt-0.6b-v3-onnx-int4';
const MODEL_LABEL = 'Parakeet TDT 0.6B v3';
const APPROX_SIZE = '409 MB';
const SUPPORTED_LANGUAGES = ['en','de','es','fr','it','pt','pl','nl','uk','sv','cs','da','fi','el','hu','ro','bg','hr','sk','lt','sl','lv','et','ca'];

let transformers = null;
let pipe = null;
let warmupState = 'idle';
let lastError = '';
let downloadProgress = 0;
const cacheDir = path.join(os.homedir(), '.cache', 'parakeet');

/* Same test the ONNX service uses: the weights live under the cache dir as a
   folder named after the model id. Without it Settings could not tell "download
   the 409 MB" from "load what is already on disk" after a restart. */
function isCached() {
  try {
    return fs.existsSync(path.join(cacheDir, MODEL_ID));
  } catch {
    return false;
  }
}

function status() {
  return {
    ok: true, name: 'parakeet', modelId: MODEL_ID, modelLabel: MODEL_LABEL,
    family: 'parakeet', available: true, ready: warmupState === 'ready' && Boolean(pipe),
    modelInstalled: isCached(),
    warmupState, downloadProgress, cacheDir, lastError,
    backends: ['onnxruntime-node'], multilingual: true,
    supportedLanguages: SUPPORTED_LANGUAGES, approxSize: APPROX_SIZE,
  };
}

async function loadTransformers() {
  if (!transformers) {
    transformers = await import('@huggingface/transformers');
    transformers.env.cacheDir = cacheDir;
    transformers.env.allowLocalModels = true;
  }
  return transformers;
}

async function warmup(payload = {}) {
  if (pipe) return { ok: true, warmedUp: true, status: status() };
  warmupState = 'loading'; downloadProgress = 0;
  try {
    const { pipeline } = await loadTransformers();
    pipe = await pipeline('automatic-speech-recognition', MODEL_ID, {
      dtype: 'q8',
      progress_callback: (progress) => {
        if (progress.status === 'progress') { warmupState = 'downloading'; downloadProgress = Math.round(progress.progress || 0); }
        else if (progress.status === 'done') { downloadProgress = 100; }
      },
    });
    warmupState = 'ready'; lastError = '';
    return { ok: true, warmedUp: true, status: status() };
  } catch (err) {
    warmupState = 'error'; lastError = err.message; pipe = null;
    return { ok: false, error: err.message, status: status() };
  }
}

async function transcribe(payload = {}) {
  try {
    if (!pipe) { const w = await warmup(payload); if (!w.ok) return w; }
    let audioInput;
    if (payload.filePath) {
      if (!fs.existsSync(payload.filePath)) return { ok: false, error: 'Audio file not found: ' + payload.filePath };
      audioInput = payload.filePath;
    } else if (payload.audioData) {
      const arr = payload.audioData;
      audioInput = arr instanceof Float32Array ? arr : Float32Array.from(arr);
    } else if (payload.base64) {
      /* Read through a DataView: a Buffer is a window into a shared pool, so
         an odd byteOffset made `new Int16Array(buffer.buffer, ...)` throw
         outright, and an odd byte count made the length fractional. */
      const buffer = Buffer.from(payload.base64, 'base64');
      const samples = Math.floor(buffer.byteLength / 2);
      const float32 = new Float32Array(samples);
      for (let i = 0; i < samples; i++) float32[i] = buffer.readInt16LE(i * 2) / 32768.0;
      audioInput = float32;
    } else if (payload.text) {
      return { ok: true, provider: 'parakeet', text: payload.text, confidence: 0.85, isPassthrough: true, status: status() };
    } else {
      return { ok: false, error: 'No audio input provided.' };
    }
    /* 'auto' is the absence of a language, not a name for one. Parakeet v3
       detects it from the audio when it is not told; passing "auto" through
       would ask the model for a language by that name. */
    const options = { task: 'transcribe', return_timestamps: true };
    if (payload.language && payload.language !== 'auto') options.language = payload.language;
    const result = await pipe(audioInput, options);
    const segments = (result.chunks || []).map((chunk) => ({
      start: chunk.timestamp?.[0] || 0, end: chunk.timestamp?.[1] || 0, text: chunk.text || '',
    }));
    return {
      ok: true, provider: 'parakeet', modelId: MODEL_ID,
      text: result.text || '', segments,
      confidence: (result.text || '').trim() ? 0.94 : 0,
      duration: segments.length > 0 ? segments[segments.length - 1].end : 0,
      status: status(),
    };
  } catch (err) {
    return { ok: false, error: err.message, status: status() };
  }
}

async function dispose() {
  pipe = null; warmupState = 'idle'; downloadProgress = 0;
  return { ok: true, status: status() };
}

module.exports = { createParakeetService: () => ({ status, warmup, transcribe, dispose }) };