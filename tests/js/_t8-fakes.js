// Test doubles for the T8 client audio & link suites: an injectable clock with timers,
// a worklet MessagePort stand-in, and fake WebCodecs / speechSynthesis classes.
// Not a test file (no .test.js suffix); imported by the suites.

/** A clock with setTimer/clearTimer that only moves when advance() is called. */
export function fakeClock(start = 0) {
  let t = start;
  let id = 0;
  const timers = new Map();
  return {
    now: () => t,
    setTimer: (fn, ms) => { const h = ++id; timers.set(h, { at: t + Math.max(0, Number(ms) || 0), fn }); return h; },
    clearTimer: (h) => { timers.delete(h); },
    advance(ms) {
      const end = t + ms;
      for (;;) {
        let next = null;
        for (const [h, x] of timers) if (x.at <= end && (!next || x.at < next[1].at)) next = [h, x];
        if (!next) break;
        timers.delete(next[0]);
        t = next[1].at;
        next[1].fn();
      }
      t = end;
    },
    get pending() { return timers.size; },
  };
}

/** The worklet side of a MessagePort: records what the processor posts. */
export function fakePort() {
  const port = {
    messages: [],
    transfers: [],
    onmessage: null,
    postMessage(m, tr) { port.messages.push(m); port.transfers.push(tr || []); },
    take(pred = () => true) { const out = port.messages.filter(pred); port.messages = port.messages.filter((m) => !pred(m)); return out; },
    of(type) { return port.messages.filter((m) => m && m.type === type); },
  };
  return port;
}

/** Deterministic noise in [-1, 1). */
export function lcg(seed = 1) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 2 ** 31 - 1; };
}

/** A sine (amplitude a) of n samples at `rate`. */
export function sine(n, rate, hz, a = 0.5, phase0 = 0) {
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = a * Math.sin(2 * Math.PI * hz * (i + phase0) / rate);
  return x;
}

export const dbToAmp = (db) => 10 ** (db / 20);

// ---- WebCodecs fakes -------------------------------------------------------

/** A 20 ms SILK-WB Opus TOC byte (config 9, code 0) followed by payload bytes. */
export const opusPacket = (...payload) => new Uint8Array([0x48, ...payload]);

export function fakeCodecs({ encSupported = true, decSupported = true, failEncodeAfter = Infinity, failDecode = false } = {}) {
  const log = { configs: [], encoded: [], decoded: [], encoders: [], decoders: [] };
  class AudioData {
    constructor(init) { Object.assign(this, init); this.closed = false; }
    close() { this.closed = true; }
  }
  class EncodedAudioChunk {
    constructor(init) { Object.assign(this, init); this.byteLength = init.data.byteLength ?? init.data.length; }
  }
  class AudioEncoder {
    static async isConfigSupported(cfg) { log.configs.push(['enc', cfg]); return { supported: encSupported, config: cfg }; }
    constructor({ output, error }) { this.output = output; this.error = error; this.state = 'unconfigured'; this.queue = []; this.count = 0; log.encoders.push(this); }
    configure(cfg) { this.cfg = cfg; this.state = 'configured'; log.configs.push(['enc-configure', cfg]); }
    encode(data) {
      if (this.state !== 'configured') throw new Error('not configured');
      log.encoded.push(data);
      // Output arrives later (like the real thing): on a microtask.
      queueMicrotask(() => {
        if (this.state !== 'configured') return;
        this.count += 1;
        if (this.count > failEncodeAfter) { this.state = 'closed'; this.error(new Error('encoder exploded')); return; }
        const i16 = data.data;
        const bytes = opusPacket(i16[0] & 0xff, (i16[0] >> 8) & 0xff, this.count & 0xff);
        this.output({ byteLength: bytes.length, timestamp: data.timestamp, copyTo(dst) { dst.set(bytes); } });
      });
    }
    async flush() { await new Promise((r) => setTimeout(r, 0)); }
    close() { this.state = 'closed'; }
  }
  class AudioDecoder {
    static async isConfigSupported(cfg) { log.configs.push(['dec', cfg]); return { supported: decSupported, config: cfg }; }
    constructor({ output, error }) { this.output = output; this.error = error; this.state = 'unconfigured'; log.decoders.push(this); }
    configure(cfg) { this.cfg = cfg; this.state = 'configured'; }
    decode(chunk) {
      log.decoded.push(chunk);
      queueMicrotask(() => {
        if (this.state !== 'configured') return;
        if (failDecode) { this.state = 'closed'; this.error(new Error('decoder exploded')); return; }
        const n = Math.round((chunk.duration / 1e6) * 48000);
        const first = chunk.data[1] ?? 0;
        this.output({
          numberOfFrames: n, sampleRate: 48000, timestamp: chunk.timestamp, closed: false,
          copyTo(dst) { for (let i = 0; i < n; i++) dst[i] = first / 256; },
          close() { this.closed = true; },
        });
      });
    }
    reset() { this.state = 'unconfigured'; }
    async flush() {}
    close() { this.state = 'closed'; }
  }
  return { codecs: { AudioEncoder, AudioDecoder, AudioData, EncodedAudioChunk }, log };
}

/** Let queued microtasks (fake codec outputs) run. */
export const settle = () => new Promise((r) => setTimeout(r, 0));

// ---- speechSynthesis fake --------------------------------------------------

export class FakeUtterance {
  constructor(text) { this.text = text; this.voice = null; this.lang = ''; this.rate = 1; this.pitch = 1; }
}

/** speechSynthesis with manual control: start()/end()/error() drive the utterance in flight. */
export function fakeSynth(voices = []) {
  const synth = {
    spoken: [],
    cancels: 0,
    current: null,
    voices,
    getVoices() { return synth.voices; },
    speak(u) { synth.spoken.push(u); synth.current = u; },
    cancel() {
      synth.cancels += 1;
      const u = synth.current;
      synth.current = null;
      // Chrome fires an error (interrupted) or end on the cancelled utterance.
      if (u && u.onerror) u.onerror({ error: 'interrupted' });
    },
    start() { synth.current?.onstart?.({}); },
    end() { const u = synth.current; synth.current = null; u?.onend?.({}); },
    error(error = 'synthesis-failed') { const u = synth.current; synth.current = null; u?.onerror?.({ error }); },
  };
  return synth;
}
