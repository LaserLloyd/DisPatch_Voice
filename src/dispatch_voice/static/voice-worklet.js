// DisPatch Voice: voice-worklet.js — mic capture (AudioWorklet processor `drive-mic`)
// with the energy gate, sample offsets and speech edges (v3 contracts C-§E.2, SPEC §4.6,
// §5.6). Runs on the audio thread; the pure helpers at the bottom run on the main thread.
//
// Output: Int16 mono at 16 kHz in 20 ms frames (320 samples), the format /ws/voice takes.
//   worklet → main  {off, buf}                 one frame; buf (ArrayBuffer, 640 B) is transferred;
//                                              off = 16 kHz index of its first sample since the
//                                              stream started (u32, wraps)
//                   {level}                    RMS 0..1 of the last 100 ms (the ring meter)
//                   {gate: 'open'|'closed', off}  gate edges; on 'open' the 300 ms pre-roll
//                                              frames follow at once with their own offsets;
//                                              'closed' carries the offset just past the last
//                                              frame sent, so the uplink flushes its encoder
//                   {speech: true|false}       energy edges (the optimistic duck listens)
//   main → worklet  {enabled}                  false: nothing is posted, the clock keeps running
//                   {gate: {on, preRollMs, hangoverS}}   on with `ready.gate`; off for v2
//                   {localSpeaking}            the phone's voice is playing → echoBiasDb applies
//                   {echoBiasDb}               extra dB the speech threshold needs during local speech
//                   {hold}                     (amendment T8-1) the server says hearing/transcribing:
//                                              the gate must not close
//                   {restart: true}            (amendment T8-1) a new stream: off back to 0
//
// Rates: ≥ 16 kHz → a 31-tap windowed-sinc low-pass (cutoff 7 kHz) evaluated only at
// the output instants, then decimation. < 16 kHz (Bluetooth SCO/HFP at 8 kHz) → linear
// upsampling and no decimation (feasibility M3); `off` still counts 16 kHz samples.
//
// The gate (SPEC §5.6): frame energy in dBFS against a noise floor (initialised from
// the first 300 ms, then a slow EWMA, τ ≈ 2 s, of frames under floor + 6 dB). Speech
// starts after 60 ms (3 frames) above floor + 9 dB (+ echoBiasDb while localSpeaking),
// ends after 200 ms under floor + 6 dB (+ bias) — the −3 dB hysteresis. The gate opens
// on a speech start, sends the 300 ms pre-roll, and closes `hangoverS` after the last
// frame over the lower threshold, never while `hold` is set. The hangover comes from
// the server (`ready.gate.hangover_s`), so the server always hears the silence its
// end-of-turn decision needs, and its offset tracker zero-fills the rest.
//
// Owner: track T8.

export const MIC_PROCESSOR = 'drive-mic';
export const OUT_RATE = 16000;
export const FRAME = 320;                 // 20 ms at 16 kHz
const FRAME_MS = 20;
const TAPS = 31;
const CUTOFF_HZ = 7000;

/** Gate and energy constants (dB relative to the noise floor). */
export const GATE = Object.freeze({
  openDb: 9,            // speech: this far over the floor …
  openFrames: 3,        // … for 60 ms
  closeDb: 6,           // hysteresis: under this for …
  closeFrames: 10,      // … 200 ms ends the speech edge
  floorInitFrames: 15,  // the first 300 ms set the floor
  floorTauMs: 2000,     // slow EWMA of quiet frames
  floorMinDb: -70,      // digital silence must not make every hiss "speech"
  floorStuckMs: 5000,   // no quiet frame this long (the cabin got louder): …
  floorWindowMs: 3000,  // … the floor climbs toward the quietest frame of the last 3 s
  preRollMs: 300,
  hangoverS: 1.5,       // until `ready.gate` says otherwise
  maxEchoBiasDb: 9,
});

/** Blackman-windowed sinc low-pass, unity DC gain. */
export function lowpass(rate, cutoff = CUTOFF_HZ, taps = TAPS) {
  const fc = Math.min(0.49, cutoff / rate);
  const h = new Float32Array(taps);
  const m = (taps - 1) / 2;
  let sum = 0;
  for (let i = 0; i < taps; i++) {
    const x = i - m;
    const sinc = x === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * x) / (Math.PI * x);
    const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (taps - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (taps - 1));
    h[i] = sinc * w;
    sum += h[i];
  }
  for (let i = 0; i < taps; i++) h[i] /= sum;
  return h;
}

/**
 * The processor's logic without the AudioWorklet base class, so node --test drives it
 * with a fake port. `port` needs postMessage(msg, transfer?); `rate` is the context rate.
 */
export class MicCore {
  constructor(port, rate, opts = {}) {
    this.port = port;
    this.rate = rate;
    this.up = rate < OUT_RATE;          // SCO: linear upsampling
    this.ratio = rate / OUT_RATE;       // input samples per output sample
    this.h = this.up ? null : lowpass(rate);
    this.hist = new Float32Array(TAPS);
    this.pos = 0;
    this.acc = 0;                       // decimation: fractional input position
    this.phase = 0;                     // upsampling: position between prev and the next input
    this.prev = 0;
    this.out = new Int16Array(FRAME);
    this.len = 0;
    this.off = 0;                       // 16 kHz index of the frame being filled
    this.sq = 0;
    this.sqN = 0;
    this.enabled = opts.enabled !== undefined ? !!opts.enabled : true;
    this.localSpeaking = false;
    this.echoBiasDb = Number(opts.echoBiasDb) || 0;
    this.hold = false;
    this.gateOn = false;
    this.preRollMs = GATE.preRollMs;
    this.hangoverS = GATE.hangoverS;
    if (opts.gate) this.setGate(opts.gate);
    this.resetEnergy();
  }

  resetEnergy() {
    this.floor = null;
    this.initDb = [];
    this.loudRun = 0;                   // consecutive frames over the open threshold
    this.quietRun = 0;                  // consecutive frames under the close threshold
    this.speech = false;
    this.open = false;
    this.sinceLoud = Infinity;          // ms since the last frame over the close threshold
    this.preRoll = [];                  // [{off, buf}] while closed
    this.recent = new Float32Array(GATE.floorWindowMs / FRAME_MS).fill(Infinity);
    this.recentAt = 0;
    this.sinceQuiet = 0;                // ms since a frame counted as quiet for the floor
  }

  setGate(g) {
    if (!g || typeof g !== 'object') return;
    if (typeof g.preRollMs === 'number' && g.preRollMs >= 0) this.preRollMs = Math.min(2000, g.preRollMs);
    if (typeof g.hangoverS === 'number' && g.hangoverS > 0) this.hangoverS = Math.min(30, g.hangoverS);
    if ('on' in g) {
      const on = !!g.on;
      if (on && !this.gateOn) {
        // Turning the gate on mid-stream errs toward sending: start open, let the hangover close it.
        this.open = true;
        this.sinceLoud = 0;
      }
      this.gateOn = on;
    }
  }

  onMessage(d) {
    if (!d || typeof d !== 'object') return;
    if ('enabled' in d) {
      const en = !!d.enabled;
      if (en && !this.enabled) { this.preRoll = []; this.open = !this.gateOn; this.loudRun = 0; this.speech = false; }
      this.enabled = en;
    }
    if ('gate' in d) this.setGate(d.gate);
    if ('localSpeaking' in d) this.localSpeaking = !!d.localSpeaking;
    if ('echoBiasDb' in d) this.echoBiasDb = Math.max(0, Math.min(GATE.maxEchoBiasDb, Number(d.echoBiasDb) || 0));
    if ('hold' in d) this.hold = !!d.hold;
    if (d.restart) { this.off = 0; this.len = 0; this.resetEnergy(); }
  }

  /** One render quantum of input (Float32Array, any length). */
  process(ch) {
    if (!ch) return;
    if (this.up) this.upsample(ch); else this.decimate(ch);
  }

  decimate(ch) {
    const h = this.h;
    const hist = this.hist;
    for (let i = 0; i < ch.length; i++) {
      hist[this.pos] = ch[i];
      this.pos = (this.pos + 1) % TAPS;
      this.acc += 1;
      if (this.acc >= this.ratio) {
        this.acc -= this.ratio;
        let y = 0;
        let j = this.pos;              // oldest sample first
        for (let k = 0; k < TAPS; k++) {
          y += h[k] * hist[j];
          j = j + 1 === TAPS ? 0 : j + 1;
        }
        this.sample(y);
      }
    }
  }

  upsample(ch) {
    // Output instants every `ratio` input samples (ratio < 1), interpolated between
    // the previous input sample and the current one.
    for (let i = 0; i < ch.length; i++) {
      const x = ch[i];
      while (this.phase < 1) {
        this.sample(this.prev + (x - this.prev) * this.phase);
        this.phase += this.ratio;
      }
      this.phase -= 1;
      this.prev = x;
    }
  }

  sample(y) {
    const v = y > 1 ? 1 : y < -1 ? -1 : y;
    if (this.enabled) { this.sq += v * v; this.sqN += 1; }
    this.out[this.len++] = v < 0 ? v * 0x8000 : v * 0x7fff;
    if (this.len === FRAME) {
      this.frame();
      this.len = 0;
      this.off = (this.off + FRAME) >>> 0;
    }
  }

  frame() {
    if (!this.enabled) return;          // the clock runs; nothing leaves the worklet
    const out = this.out;
    let s = 0;
    for (let i = 0; i < FRAME; i++) { const x = out[i] / 32768; s += x * x; }
    const db = 10 * Math.log10(s / FRAME + 1e-12);
    this.energy(db);
    const off = this.off;
    if (!this.gateOn) {
      this.post(off);
    } else if (this.open) {
      this.post(off);
      if (!this.hold && this.sinceLoud >= this.hangoverS * 1000) {
        this.open = false;
        this.preRoll = [];
        this.port.postMessage({ gate: 'closed', off: (off + FRAME) >>> 0 });
      }
    } else {
      this.preRoll.push({ off, buf: out.buffer.slice(0) });
      const keep = Math.max(0, Math.round(this.preRollMs / FRAME_MS));
      while (this.preRoll.length > keep) this.preRoll.shift();
      if (this.speech) {
        this.open = true;
        const pre = this.preRoll;
        this.preRoll = [];
        this.port.postMessage({ gate: 'open', off: pre.length ? pre[0].off : off });
        for (const f of pre) this.port.postMessage({ off: f.off, buf: f.buf }, [f.buf]);
      }
    }
    if (this.sqN >= 1600) {             // 100 ms at 16 kHz
      this.port.postMessage({ level: Math.sqrt(this.sq / this.sqN) });
      this.sq = 0;
      this.sqN = 0;
    }
  }

  post(off) {
    const buf = this.out.buffer.slice(0);
    this.port.postMessage({ off, buf }, [buf]);
  }

  energy(db) {
    if (this.floor === null) {
      this.initDb.push(db);
      if (this.initDb.length < GATE.floorInitFrames) return;
      this.floor = Math.max(GATE.floorMinDb, Math.min(...this.initDb));
      this.initDb = [];
    }
    const bias = this.localSpeaking ? this.echoBiasDb : 0;
    const openAt = this.floor + GATE.openDb + bias;
    const closeAt = this.floor + GATE.closeDb + bias;
    if (db > openAt) this.loudRun += 1; else this.loudRun = 0;
    if (db < closeAt) this.quietRun += 1; else this.quietRun = 0;
    if (db >= closeAt) this.sinceLoud = 0; else this.sinceLoud += FRAME_MS;
    if (!this.speech && this.loudRun >= GATE.openFrames) {
      this.speech = true;
      this.port.postMessage({ speech: true });
    } else if (this.speech && this.quietRun >= GATE.closeFrames) {
      this.speech = false;
      this.port.postMessage({ speech: false });
    }
    // The floor follows quiet frames only: τ ≈ 2 s, faster downwards. If nothing has
    // been quiet for 5 s the noise itself rose (engine, fan, window down): climb toward
    // the quietest frame of the last 3 s at the same pace, or the gate never closes.
    this.recent[this.recentAt] = db;
    this.recentAt = (this.recentAt + 1) % this.recent.length;
    const slow = 1 - Math.exp(-FRAME_MS / GATE.floorTauMs);
    if (db < this.floor + GATE.closeDb) {
      this.sinceQuiet = 0;
      const a = db < this.floor ? 0.05 : slow;
      this.floor = Math.max(GATE.floorMinDb, this.floor + a * (db - this.floor));
    } else if ((this.sinceQuiet += FRAME_MS) >= GATE.floorStuckMs) {
      let min = Infinity;
      for (let i = 0; i < this.recent.length; i++) if (this.recent[i] < min) min = this.recent[i];
      if (min > this.floor && min !== Infinity) this.floor += slow * (min - this.floor);
    }
  }
}

// Registered only inside AudioWorkletGlobalScope (where registerProcessor exists).
// Via globalThis: these are worklet-scope globals the app's source scanner rightly
// does not know about, and importing this file on the main thread must not throw.
if (typeof globalThis.registerProcessor === 'function' && typeof globalThis.AudioWorkletProcessor === 'function') {
  class DriveMicProcessor extends globalThis.AudioWorkletProcessor {
    constructor(options) {
      super();
      this.core = new MicCore(this.port, globalThis.sampleRate, (options && options.processorOptions) || {});
      this.port.onmessage = (e) => this.core.onMessage(e.data);
    }

    process(inputs) {
      const ch = inputs[0] && inputs[0][0];
      if (ch) this.core.process(ch);
      return true;
    }
  }
  globalThis.registerProcessor(MIC_PROCESSOR, DriveMicProcessor);
}

// ------------------------------------------------------- main-thread helpers

const BT_LABEL = /bluetooth|hands-?free|headset|\bhfp\b|\bsco\b|\bbt\b|car ?kit|airpods|buds/i;

/**
 * True when the output is (probably) Bluetooth: a device label that says so, or an
 * AudioContext.outputLatency over 100 ms (car kits run 150–300 ms). Either input may
 * be missing (labels are empty without a mic permission; outputLatency is 0 on some).
 */
export function isBluetoothOutput({ label = '', outputLatency = 0 } = {}) {
  return BT_LABEL.test(String(label || '')) || (Number(outputLatency) || 0) > 0.1;
}

/**
 * The phone-voice echo bias (SPEC §5.4): starts at +6 dB on Bluetooth output (else 0),
 * +3 dB per `echo_ignored` during local speech, −1 dB per clean turn, between 0 and +9.
 * Post `{echoBiasDb: bias.value}` to the worklet after each change.
 */
export function createEchoBias({ bluetooth = false } = {}) {
  let v = bluetooth ? 6 : 0;
  const clamp = (x) => Math.max(0, Math.min(GATE.maxEchoBiasDb, x));
  return {
    get value() { return v; },
    onEchoIgnored(duringLocalSpeech = true) { if (duringLocalSpeech) v = clamp(v + 3); return v; },
    onCleanTurn() { v = clamp(v - 1); return v; },
    setBluetooth(bt) { if (bt) v = Math.max(v, 6); return v; },
  };
}

/** The worklet gate settings from `ready.gate` (C.4): {pre_roll_ms, hangover_s} → the main→worklet message. */
export function gateFromReady(gate) {
  if (!gate || typeof gate !== 'object') return { gate: { on: false } };
  return {
    gate: {
      on: true,
      preRollMs: Number.isFinite(gate.pre_roll_ms) ? gate.pre_roll_ms : GATE.preRollMs,
      hangoverS: Number.isFinite(gate.hangover_s) ? gate.hangover_s : GATE.hangoverS,
    },
  };
}
