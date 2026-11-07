/* =========================================================================
  Voice Studio Integration — wraps the Voice Studio engine catalogue
  (11 ASR engines: WhisperX, Faster-Whisper, Parakeet TDT, Moonshine,
  FunASR, sherpa-onnx live dictation, MLX Whisper) into a single service
  that plugs into the existing transcription dispatch layer.

  Voice Studio: https://github.com/gskfilmmaker/voicestudio
  ========================================================================= */
const fs = require('fs');
const path = require('path');
const os = require('os');

const VOICE_STUDIO_DIR = path.join(os.homedir(), 'VoiceStudio');
const VOICE_STUDIO_URL = 'https://github.com/gskfilmmaker/voicestudio';
const SUPPORTED_ENGINES = ['whisperx','faster-whisper','parakeet','moonshine','funasr','sherpa-onnx-asr'];
const DEFAULT_ENGINE = 'faster-whisper';

let warmupState = 'idle';
let lastError = '';
let activeEngine = DEFAULT_ENGINE;

function status() {
  return {
    ok: true, name: 'voice-studio', activeEngine,
    available: fs.existsSync(VOICE_STUDIO_DIR),
    warmupState, lastError,
    engines: SUPPORTED_ENGINES.map((e) => ({
      id: e, label: e.split('-').map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
    })),
    voiceStudioPath: VOICE_STUDIO_DIR,
    /* Settings offer this to an operator who has no install yet; the URL lives
       here so the two sides cannot drift apart. */
    installUrl: VOICE_STUDIO_URL,
    /* There is no audio bridge into Voice Studio yet, so `transcribe` answers
       with an error even when the folder is present. Settings say so instead of
       offering an engine that cannot produce a word. */
    transcribeWired: false,
  };
}

function installed() {
  return fs.existsSync(VOICE_STUDIO_DIR);
}

async function warmup(payload = {}) {
  if (payload.engine) {
    if (!SUPPORTED_ENGINES.includes(payload.engine)) {
      return { ok: false, error: 'Unknown engine: ' + payload.engine, status: status() };
    }
    activeEngine = payload.engine;
  }
  /* Reporting ready without the install made Settings offer an engine that
     cannot answer, and the console then waited on a recogniser that was never
     going to produce a word. */
  if (!installed()) {
    warmupState = 'unavailable';
    lastError = 'Voice Studio is not installed.';
    return { ok: false, error: lastError + ' Install from: ' + VOICE_STUDIO_URL, status: status() };
  }
  warmupState = 'ready'; lastError = '';
  return { ok: true, warmedUp: true, engine: activeEngine, status: status() };
}

async function transcribe(payload = {}) {
  const engine = payload.engine || activeEngine;
  if (!SUPPORTED_ENGINES.includes(engine)) {
    return { ok: false, error: 'Unknown engine: ' + engine, status: status() };
  }
  const audioInput = payload.audioData || payload.filePath || payload.base64;
  if (!audioInput) return { ok: false, error: 'No audio input provided.', status: status() };
  if (!installed()) {
    warmupState = 'unavailable';
    lastError = 'Voice Studio is not installed.';
    return { ok: false, error: lastError + ' Install from: ' + VOICE_STUDIO_URL, status: status() };
  }

  try {
    /* There is no process here to hand the audio to yet. Answering ok with an
       empty transcript made the failure look like a silent microphone: the
       loop kept waiting for text that was never coming. */
    return {
      ok: false, error: 'The ' + engine + ' engine is not wired up in this build.', engine, status: status(),
    };
  } catch (err) {
    lastError = err.message;
    return { ok: false, error: err.message, status: status() };
  }
}

async function dispose() { warmupState = 'idle'; return { ok: true, status: status() }; }

function setEngine(name) {
  if (!SUPPORTED_ENGINES.includes(name)) return { ok: false, error: 'Unknown engine: ' + name, status: status() };
  activeEngine = name;
  return { ok: true, activeEngine, status: status() };
}

module.exports = {
  createVoiceStudioService: () => ({ status, warmup, transcribe, dispose, setEngine }),
  SUPPORTED_ENGINES,
};