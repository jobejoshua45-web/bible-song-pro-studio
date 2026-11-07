const os = require('os');
const { createOnnxWhisperService } = require('./whisper-onnx-service.cjs');
const { createMlxWhisperCompatService } = require('./mlx-whisper-service.cjs');
const { createParakeetService } = require('./parakeet-service.cjs');
const { createVoiceStudioService } = require('./voice-studio-service.cjs');
const { createSemanticVerseDetection } = require('./semantic-verse-detection.cjs');

function isAppleSilicon() {
  return os.platform() === 'darwin' && os.arch() === 'arm64';
}

function createTranscriptionService({ app }) {
  const onnx = createOnnxWhisperService({ app });
  const mlx = createMlxWhisperCompatService({ app });
  const parakeet = createParakeetService();
  const voiceStudio = createVoiceStudioService();
  const semantic = createSemanticVerseDetection();
  let activeEngine = 'onnx';

  /* Settings call the on-device recogniser "local"; this file was built around
     the name "onnx". Both resolve, because one is what the operator chose and
     the other is what the engine is. An unknown name resolves to nothing rather
     than to the ONNX default: silently transcribing with the wrong recogniser
     is worse than saying the engine was not understood. */
  function resolveEngineName(name) {
    if (!name) return activeEngine;
    return name === 'local' ? 'onnx' : name;
  }

  function getEngine(name) {
    const resolved = resolveEngineName(name);
    if (resolved === 'onnx') return onnx;
    if (resolved === 'mlx') return isAppleSilicon() ? mlx : null;
    if (resolved === 'parakeet') return parakeet;
    if (resolved === 'voice-studio') return voiceStudio;
    return null;
  }

  function status() {
    return {
      ok: true,
      activeEngine,
      platform: { os: os.platform(), arch: os.arch(), isAppleSilicon: isAppleSilicon() },
      engines: {
        onnx: onnx.status(),
        mlx: mlx.status(),
        parakeet: parakeet.status(),
        'voice-studio': voiceStudio.status(),
        semantic: semantic.status(),
      },
    };
  }

  function setActiveEngine(name) {
    const resolved = resolveEngineName(name);
    if (resolved === 'mlx' && !isAppleSilicon()) {
      return { ok: false, error: 'MLX Whisper requires Apple Silicon (arm64)' };
    }
    if (resolved !== 'onnx' && resolved !== 'mlx' && resolved !== 'parakeet' && resolved !== 'voice-studio') {
      return { ok: false, error: 'Unknown engine: ' + name + '. Use local, onnx, parakeet, or voice-studio.' };
    }
    activeEngine = resolved;
    return { ok: true, activeEngine, status: status() };
  }

  async function warmup(payload = {}) {
    const engine = getEngine(payload.engine);
    if (!engine) {
      return { ok: false, error: 'Unknown or unavailable engine: ' + payload.engine };
    }
    const result = await engine.warmup(payload);
    // Opt-in only. Warming the embedding model unconditionally pulled a second
    // model down on every warmup for a paraphrase search nothing currently calls.
    if (payload.semantic === true) {
      await semantic.warmup();
    }
    return { ...result, activeEngine: resolveEngineName(payload.engine) };
  }

  async function transcribe(payload = {}) {
    const engine = getEngine(payload.engine);
    if (!engine) {
      return { ok: false, error: 'Unknown or unavailable engine: ' + payload.engine };
    }
    const result = await engine.transcribe(payload);
    return { ...result, activeEngine: resolveEngineName(payload.engine) };
  }

  async function dispose(payload = {}) {
    const engine = getEngine(payload.engine);
    if (!engine) {
      return { ok: false, error: 'Unknown or unavailable engine: ' + payload.engine };
    }
    const result = await engine.dispose();
    return { ...result, activeEngine: resolveEngineName(payload.engine) };
  }

  async function searchParaphrased(transcript, bibleData, versionId, options = {}) {
    return semantic.searchParaphrased(transcript, bibleData, versionId, options);
  }

  async function warmupSemantic() {
    return semantic.warmup();
  }

  function setLocalModel(value) {
    return onnx.setModel(value);
  }

  return {
    status, setActiveEngine, warmup, transcribe, dispose,
    setLocalModel, searchParaphrased, warmupSemantic,
  };
}

module.exports = { createTranscriptionService };
