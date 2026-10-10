// DisPatch Voice: voice-player-worklet.js — the reply player (AudioWorklet processor
// `drive-player`) and its main-thread client (v3 contracts C-§E.2, SPEC §4.6, §4.8).
//
// Replaces the createBufferSource scheduler: one ring of sentences on the audio
// thread, so pause / resume / rewind are sample-accurate and `played` is exactly
// what left the worklet (the heard position, before output latency).
//
//   main → worklet  {type:'push', gen, seq, kind, rate, pcm: Float32Array}   (pcm transferred)
//                       rate 24000 (PCM) or 48000 (decoded Opus), resampled here to the
//                       context rate; optional `totalMs` (the clip's full length, for acks)
//                   {type:'pause'}               20 ms ramp down, then the clock stops
//                   {type:'resume', rewind}      20 ms ramp up; rewind = restart the current
//                                                seq from its first sample. A `resume {rewind:
//                                                true}` that lands just after an in-place
//                                                resume of the same sentence (≤ 1.5 s: the
//                                                server's `duck off` arrived first) still
//                                                rewinds — the order of the two never matters
//                   {type:'flush', gen}          drop everything older than gen (100 ms fade);
//                                                also ends a pause (a barge-in ends the duck)
//                   {type:'gain', value, rampMs} master gain
//                   {type:'prebuffer', ms}       hold a new sentence until this much is buffered
//                                                ("Wait" on a slow link: the whole sentence)
//                   {type:'end', gen, seq}       (amendment T8-1) no more audio for this seq:
//                                                a drain after it is the end, not an underrun
//                   {type:'reset'}               (amendment T8-1) a new socket: gens start over
//   worklet → main  {type:'played', gen, seq, ms, leadMs}  every 100 ms while a sentence is current
//                   {type:'underrun', gen, seq, gapMs}     once per hole inside one sentence, when
//                                                          the audio resumes (the exact gap)
//                   {type:'idle', gen, seq, ended}         the queue drained (ended: the seq was
//                                                          complete — `end` or a later seq)
//                   {type:'level', v}                      output RMS over the same 100 ms
//
// Priorities notice > reply > backchannel > ack. Fillers (ack, backchannel) never
// queue behind or interrupt a reply/notice — they are dropped. A reply never cuts an
// ack of ≤ 1.3 s (it queues behind it); it cuts a longer ack (a tier line), a notice
// line or a backchannel with a 50 ms fade (SPEC §6.2, §10; experience L7). The rule
// lives here, in the one place that owns the queue (amendment T8-1).
//
// `gen` is the wire's u16 (C.4): compared with serial arithmetic, so a wrap never
// strands the player.
//
// Owner: track T8.

export const PLAYER_PROCESSOR = 'drive-player';
export const PRIORITY = Object.freeze({ ack: 0, backchannel: 1, reply: 2, notice: 3 });
export const ACK_UNCUT_MS = 1300;
export const CUT_FADE_MS = 50;
export const FLUSH_FADE_MS = 100;
export const PAUSE_RAMP_MS = 20;
export const POST_EVERY_MS = 100;
export const LATE_REWIND_MS = 1500;
export const DONE_KEEP = 64;           // finished (gen, seq) pairs remembered against stragglers

const genNewer = (a, b) => a !== b && ((a - b) & 0xffff) < 0x8000;
const genOlder = (a, b) => a !== b && !genNewer(a, b);
const rankOf = (kind) => (kind in PRIORITY ? PRIORITY[kind] : PRIORITY.reply);
const doneKey = (gen, seq) => gen * 4294967296 + seq;   // u16 gen, u32 seq: exact in a double
const doneGen = (key) => Math.floor(key / 4294967296);

// ------------------------------------------------------------- resampler

const TABLE_RES = 512;   // kernel table points per input sample
const HALF_TAPS = 16;    // half-width in output-rate terms (32 taps when upsampling)
const tables = new Map();

function kernelTable(inRate, outRate) {
  const key = `${inRate}>${outRate}`;
  let t = tables.get(key);
  if (t) return t;
  const scale = Math.min(1, outRate / inRate);
  const fc = 0.5 * scale * 0.92;           // cycles per input sample (8 % guard band)
  const W = Math.ceil(HALF_TAPS / scale);  // half-width in input samples
  const n = W * TABLE_RES;
  const h = new Float32Array(n + 2);
  for (let i = 0; i <= n + 1; i++) {
    const x = i / TABLE_RES;
    const sinc = x === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * x) / (Math.PI * x);
    const r = Math.min(1, x / W);
    const w = 0.42 + 0.5 * Math.cos(Math.PI * r) + 0.08 * Math.cos(2 * Math.PI * r); // Blackman (centred)
    h[i] = x >= W ? 0 : sinc * w;
  }
  t = { h, W, gain: 1 };
  // Normalise to unity DC gain at the integer phase.
  let s = 0;
  for (let k = -W + 1; k <= W; k++) s += h[Math.abs(k) * TABLE_RES];
  t.gain = 1 / s;
  tables.set(key, t);
  return t;
}

/**
 * Streaming windowed-sinc resampler with no start-up delay: output sample j sits at
 * input position j × inRate / outRate. push() returns what can be computed now;
 * flush() pads with zeros and returns the tail.
 */
export class StreamResampler {
  constructor(inRate, outRate) {
    this.inRate = inRate;
    this.outRate = outRate;
    this.pass = inRate === outRate;
    if (this.pass) return;
    const t = kernelTable(inRate, outRate);
    this.h = t.h; this.W = t.W; this.g = t.gain;
    this.step = inRate / outRate;
    // Input history: W zeros before the first real sample, so position W is sample 0.
    this.buf = new Float32Array(4096);
    this.n = this.W;
    this.p = this.W;
    this.realEnd = this.W;
  }

  _append(x) {
    if (this.n + x.length > this.buf.length) {
      const b = new Float32Array(Math.max(this.buf.length * 2, this.n + x.length));
      b.set(this.buf.subarray(0, this.n));
      this.buf = b;
    }
    this.buf.set(x, this.n);
    this.n += x.length;
  }

  _run(limit) {
    const { h, W, buf, step } = this;
    const g = this.g;
    const stop = Math.min(limit, this.n - W);
    const out = new Float32Array(Math.max(0, Math.ceil((stop - this.p) / step) + 1));
    let o = 0;
    while (this.p < limit && Math.floor(this.p) + W < this.n) {
      const p = this.p;
      const i0 = Math.floor(p);
      let y = 0;
      for (let k = i0 - W + 1; k <= i0 + W; k++) {
        const d = Math.abs(p - k) * TABLE_RES;   // the table is in input-sample units
        const di = d | 0;
        if (di >= h.length - 1) continue;
        const v = h[di] + (h[di + 1] - h[di]) * (d - di);
        y += buf[k] * v;
      }
      out[o++] = y * g;
      this.p += step;
    }
    // Drop history nobody will read again.
    const keep = Math.floor(this.p) - W;
    if (keep > 2048) {
      this.buf.copyWithin(0, keep, this.n);
      this.n -= keep; this.p -= keep; this.realEnd -= keep;
    }
    return o === out.length ? out : out.slice(0, o);
  }

  push(x) {
    if (this.pass) return x;
    this._append(x);
    this.realEnd += x.length;
    return this._run(Infinity);
  }

  flush() {
    if (this.pass) return new Float32Array(0);
    this._append(new Float32Array(this.W + 1));
    return this._run(this.realEnd);
  }
}

// ------------------------------------------------------------- player core

/**
 * The processor's logic without the AudioWorklet base class: node --test drives it
 * with a fake port and calls render(out) per quantum.
 */
export class PlayerCore {
  constructor(port, rate) {
    this.port = port;
    this.rate = rate;
    this.items = [];
    this.minGen = null;
    this.paused = false;
    this.env = 1;                      // pause envelope 0..1
    this.envStep = 1 / Math.max(1, Math.round(PAUSE_RAMP_MS / 1000 * rate));
    this.gain = 1;
    this.gainTarget = 1;
    this.gainStep = 0;
    this.prebufferMs = 0;
    this.t = 0;                        // samples rendered since start
    this.lastPost = 0;
    this.sq = 0;
    this.sqN = 0;
    this.drained = null;               // {item, at} while the current sentence waits for audio
    this.resumed = null;               // {item, at}: the last in-place resume from a pause
    this.idleSent = true;
    this.done = new Set();             // finished (gen, seq) keys: a straggler for one is dropped
    this.postEvery = Math.round(POST_EVERY_MS / 1000 * rate);
  }

  onMessage(d) {
    if (!d || typeof d !== 'object') return;
    switch (d.type) {
      case 'push': this.push(d); break;
      case 'pause': this.paused = true; break;
      case 'resume': this.resume(!!d.rewind); break;
      case 'flush': this.flush(d.gen); break;
      case 'gain': this.setGain(d.value, d.rampMs); break;
      case 'prebuffer': this.prebufferMs = Math.max(0, Math.min(10000, Number(d.ms) || 0)); break;
      case 'end': this.end(d.gen, d.seq); break;
      case 'reset': this.reset(); break;
      default: break;
    }
  }

  // ---- queue

  find(gen, seq) { return this.items.find((it) => it.gen === gen && it.seq === seq && !it.cut) || null; }

  newItem(d, gen) {
    const kind = typeof d.kind === 'string' ? d.kind : 'reply';
    const rate = Number(d.rate) || this.rate;
    return {
      gen, seq: d.seq >>> 0, kind, rank: rankOf(kind), chunks: [], len: 0, read: 0, ci: 0, co: 0,
      ended: false, started: false, cut: null, totalMs: Number.isFinite(d.totalMs) ? d.totalMs : null,
      rs: new StreamResampler(rate, this.rate),
    };
  }

  append(it, x) {
    if (!x || !x.length) return;
    it.chunks.push(x);
    it.len += x.length;
    this.idleSent = false;             // something to play again: the next drain is reported
    if (this.drained && this.drained.item === it) {
      const gapMs = (this.t - this.drained.at) / this.rate * 1000;
      this.drained = null;
      this.port.postMessage({ type: 'underrun', gen: it.gen, seq: it.seq, gapMs: Math.round(gapMs) });
    }
  }

  cutItem(it, ms) {
    if (it.cut) return;
    // Not started, finished, or paused into silence (a barge-in during a duck): gone now —
    // the old reply must never fade back in.
    if (!it.started || it.read >= it.len || (this.paused && this.env === 0)) { this.remove(it); return; }
    const n = Math.max(1, Math.round(ms / 1000 * this.rate));
    it.cut = { left: n, total: n };
  }

  remove(it) {
    const i = this.items.indexOf(it);
    if (i >= 0) this.items.splice(i, 1);
    if (this.drained && this.drained.item === it) this.drained = null;
    // Gone for good (played out, cut, or dropped): a late chunk for it must not start it
    // again after the next sentence (review T8 r2 L5). Bounded; insertion order = age.
    this.done.add(doneKey(it.gen, it.seq));
    if (this.done.size > DONE_KEEP) this.done.delete(this.done.values().next().value);
  }

  push(d) {
    let pcm = d.pcm;
    if (pcm instanceof Int16Array) pcm = Float32Array.from(pcm, (v) => v / 32768);
    if (!(pcm instanceof Float32Array)) return;
    if (!pcm.length) return;   // nothing to play: never create an empty item (review T8 r2 L2)
    const gen = (Number(d.gen) || 0) & 0xffff;
    if (this.minGen !== null && genOlder(gen, this.minGen)) return;   // from before a barge-in
    if (this.minGen === null || genNewer(gen, this.minGen)) this.flush(gen);
    const seq = d.seq >>> 0;
    // A straggler for a finished sentence, or for one fading out after a cut.
    if (this.done.has(doneKey(gen, seq)) || this.items.some((x) => x.cut && x.gen === gen && x.seq === seq)) return;
    let it = this.find(gen, seq);
    if (!it) {
      it = this.newItem(d, gen);
      if (!this.admit(it)) return;
      this.items.push(it);
      this.idleSent = false;
    } else if (Number.isFinite(d.totalMs)) {
      it.totalMs = d.totalMs;
    }
    this.append(it, it.rs.push(pcm));
  }

  /** The priority rule. Returns false when the newcomer is dropped. */
  admit(it) {
    const live = this.items.filter((x) => !x.cut);
    const filler = it.rank <= PRIORITY.backchannel;
    if (filler && live.some((x) => x.rank > it.rank)) return false;
    if (filler) return true;
    // A reply or notice arrives: fillers that have not started are dropped.
    for (const x of live) if (!x.started && x.rank < it.rank && x.rank <= PRIORITY.backchannel) this.remove(x);
    const cur = this.items[0];
    if (!cur || cur.cut || !cur.started) return true;
    if (cur.kind === 'ack') {
      const ms = cur.totalMs ?? (cur.len / this.rate * 1000);
      if (ms > ACK_UNCUT_MS) this.cutItem(cur, CUT_FADE_MS);
    } else if (cur.kind === 'backchannel') {
      this.cutItem(cur, CUT_FADE_MS);
    } else if (cur.kind === 'notice' && it.kind === 'reply') {
      this.cutItem(cur, CUT_FADE_MS);
    }
    return true;
  }

  end(gen, seq) {
    const it = this.find((Number(gen) || 0) & 0xffff, seq >>> 0);
    if (!it || it.ended) return;
    it.ended = true;
    // A drain followed by `end` is the end of the sentence, not a hole: no underrun for the
    // resampler tail, and `idle {ended: true}` must still follow the earlier `ended: false`.
    if (this.drained && this.drained.item === it) { this.drained = null; this.idleSent = false; }
    this.append(it, it.rs.flush());
  }

  flush(gen) {
    const g = (Number(gen) || 0) & 0xffff;
    this.minGen = g;
    for (const it of [...this.items]) if (genOlder(it.gen, g)) this.cutItem(it, FLUSH_FADE_MS);
    // Older gens are dropped by minGen now; only this gen's finished sentences still matter.
    for (const k of [...this.done]) if (doneGen(k) !== g) this.done.delete(k);
    // A barge-in ends any duck: the next reply must play.
    this.paused = false;
  }

  reset() {
    this.items = [];
    this.done = new Set();
    this.minGen = null;
    this.paused = false;
    this.env = 1;
    this.drained = null;
    this.resumed = null;
    this.idleSent = true;
  }

  resume(rewind) {
    const cur = this.items[0];
    const live = cur && cur.started && !cur.cut;
    if (!this.paused) {
      // Not paused: only a late rewind for the sentence that was just resumed in place.
      const r = this.resumed;
      if (rewind && live && r && r.item === cur && this.t - r.at <= LATE_REWIND_MS / 1000 * this.rate) {
        cur.read = 0; cur.ci = 0; cur.co = 0;
        this.resumed = null;
      }
      return;
    }
    this.paused = false;
    if (rewind && live) { cur.read = 0; cur.ci = 0; cur.co = 0; this.resumed = null; }
    else this.resumed = live ? { item: cur, at: this.t } : null;
  }

  setGain(value, rampMs) {
    const v = Math.max(0, Math.min(2, Number(value)));
    if (!Number.isFinite(v)) return;
    this.gainTarget = v;
    const n = Math.round((Number(rampMs) || 0) / 1000 * this.rate);
    if (n <= 0) { this.gain = v; this.gainStep = 0; } else this.gainStep = Math.abs(v - this.gain) / n;
  }

  bufferedMs(it) { return (it.len - it.read) / this.rate * 1000; }

  canStart(it) {
    if (!this.prebufferMs || it.ended || this.items.length > 1) return it.len > 0;
    return this.bufferedMs(it) >= this.prebufferMs;
  }

  /** Finish the current sentence; post idle when nothing follows. */
  finish(it) {
    this.remove(it);
    if (!this.items.length && !this.idleSent) {
      this.idleSent = true;
      this.port.postMessage({ type: 'idle', gen: it.gen, seq: it.seq, ended: true });
    }
  }

  /** One output sample from the queue (0 when nothing is playable). Advances the clock. */
  next() {
    for (;;) {
      const it = this.items[0];
      if (!it) return 0;
      // `end` for a sentence that never produced a sample: done, it must not hold the queue.
      if (it.ended && it.len === 0) { this.finish(it); continue; }
      if (!it.started) {
        if (!this.canStart(it)) return 0;
        it.started = true;
      }
      if (it.read < it.len) {
        let c = it.chunks[it.ci];
        while (it.co >= c.length) { it.ci += 1; it.co = 0; c = it.chunks[it.ci]; }
        let s = c[it.co++];
        it.read += 1;
        if (it.cut) {
          s *= it.cut.left / it.cut.total;
          if (--it.cut.left <= 0) this.finish(it);
        }
        return s;
      }
      // The current sentence has nothing left to play.
      if (it.cut || it.ended || this.items.length > 1) { this.finish(it); continue; }
      if (!this.drained) {
        this.drained = { item: it, at: this.t };
        if (!this.idleSent) {
          this.idleSent = true;
          this.port.postMessage({ type: 'idle', gen: it.gen, seq: it.seq, ended: false });
        }
      }
      return 0;
    }
  }

  /** Fill one render quantum. `out` is the output channel (Float32Array); `more` any extra channels. */
  render(out, more = []) {
    for (let i = 0; i < out.length; i++) {
      // Pause envelope: ramp, and once silent the clock stops (nothing is consumed).
      if (this.paused) this.env = Math.max(0, this.env - this.envStep);
      else this.env = Math.min(1, this.env + this.envStep);
      if (this.gain !== this.gainTarget) {
        if (this.gainStep === 0 || Math.abs(this.gainTarget - this.gain) <= this.gainStep) this.gain = this.gainTarget;
        else this.gain += this.gain < this.gainTarget ? this.gainStep : -this.gainStep;
      }
      let s = 0;
      if (!(this.paused && this.env === 0)) s = this.next() * this.env * this.gain;
      out[i] = s;
      this.sq += s * s;
      this.sqN += 1;
      this.t += 1;
    }
    for (const ch of more) ch.set(out);
    if (this.t - this.lastPost >= this.postEvery) this.report();
  }

  report() {
    // Keep the 100 ms cadence on the sample clock (quanta do not divide it evenly).
    this.lastPost += this.postEvery;
    if (this.t - this.lastPost >= this.postEvery) this.lastPost = this.t;
    const cur = this.items.find((x) => !x.cut) || null;
    if (cur) {
      let lead = 0;
      for (const x of this.items) if (!x.cut) lead += x.len - x.read;
      this.port.postMessage({
        type: 'played', gen: cur.gen, seq: cur.seq,
        ms: Math.round(cur.read / this.rate * 1000), leadMs: Math.round(lead / this.rate * 1000),
      });
      this.port.postMessage({ type: 'level', v: Math.sqrt(this.sq / Math.max(1, this.sqN)) });
    }
    this.sq = 0;
    this.sqN = 0;
  }
}

if (typeof globalThis.registerProcessor === 'function' && typeof globalThis.AudioWorkletProcessor === 'function') {
  class DrivePlayerProcessor extends globalThis.AudioWorkletProcessor {
    constructor() {
      super();
      this.core = new PlayerCore(this.port, globalThis.sampleRate);
      this.port.onmessage = (e) => this.core.onMessage(e.data);
    }

    process(_inputs, outputs) {
      const o = outputs[0];
      if (o && o[0]) this.core.render(o[0], o.slice(1));
      return true;
    }
  }
  globalThis.registerProcessor(PLAYER_PROCESSOR, DrivePlayerProcessor);
}

// ------------------------------------------------------- main-thread helpers

/**
 * The main-thread side of the player node. `port` is node.port. Push transfers the
 * PCM buffer. Keeps the last `played` report so `heard` frames can be built at once.
 *   new AudioWorkletNode(ctx, 'drive-player', {numberOfInputs: 0, outputChannelCount: [1]})
 */
export function createPlayerClient(port, { onPlayed, onUnderrun, onIdle, onLevel } = {}) {
  let last = null;
  let paused = false;
  port.onmessage = (e) => {
    const d = e.data;
    if (!d) return;
    if (d.type === 'played') { last = d; onPlayed?.(d); }
    else if (d.type === 'underrun') onUnderrun?.(d);
    else if (d.type === 'idle') onIdle?.(d);
    else if (d.type === 'level') onLevel?.(d.v);
  };
  const post = (m, tr) => port.postMessage(m, tr || []);
  return {
    /** `pcm`: a Float32Array in [-1, 1], or an Int16Array (scaled by the worklet), or numbers. */
    push({ gen, seq, kind = 'reply', rate, pcm, totalMs }) {
      // An Int16Array goes through as is — the worklet divides by 32768; copying it into a
      // Float32Array here would play full-scale noise (review T8 r2 L1).
      let p = pcm instanceof Float32Array || pcm instanceof Int16Array ? pcm : Float32Array.from(pcm);
      if (p.byteOffset !== 0 || p.byteLength !== p.buffer.byteLength) p = p.slice();
      const m = { type: 'push', gen: gen & 0xffff, seq: seq >>> 0, kind, rate, pcm: p };
      if (Number.isFinite(totalMs)) m.totalMs = totalMs;
      post(m, [p.buffer]);
    },
    pause() { paused = true; post({ type: 'pause' }); },
    resume(rewind = false) { paused = false; post({ type: 'resume', rewind: !!rewind }); },
    flush(gen) { paused = false; last = null; post({ type: 'flush', gen: gen & 0xffff }); },
    gain(value, rampMs = 0) { post({ type: 'gain', value, rampMs }); },
    prebuffer(ms) { post({ type: 'prebuffer', ms }); },
    end(gen, seq) { post({ type: 'end', gen: gen & 0xffff, seq: seq >>> 0 }); },
    reset() { paused = false; last = null; post({ type: 'reset' }); },
    get paused() { return paused; },
    /** The last `played` report ({gen, seq, ms, leadMs}) or null. */
    get last() { return last; },
  };
}

/** The optimistic-duck restore delay: max(700, 400 + RTT EWMA) ms (hardening L11). */
export function duckRestoreMs(rttEwmaMs) {
  const r = Number(rttEwmaMs);
  return Math.max(700, 400 + (Number.isFinite(r) && r > 0 ? r : 0));
}

/**
 * `heard.played_ms` for `source: "server"` (C.3): what the LISTENER heard — the
 * worklet's played ms minus AudioContext.outputLatency (fallback baseLatency, else 0),
 * clamped to [0, sentenceMs]. Over car Bluetooth the speaker is 150–300 ms behind.
 */
export function heardPlayedMs(playedMs, ctx = {}, sentenceMs = Infinity) {
  const lat = Number(ctx.outputLatency) > 0 ? Number(ctx.outputLatency) : (Number(ctx.baseLatency) > 0 ? Number(ctx.baseLatency) : 0);
  const ms = Math.round((Number(playedMs) || 0) - lat * 1000);
  return Math.max(0, Math.min(Number.isFinite(sentenceMs) ? Math.round(sentenceMs) : ms, ms));
}

/**
 * The phone-side optimistic duck (SPEC §4.6 layer 1). On a mic speech edge while a
 * reply plays: `apply(true)` (the caller ramps the duck gain to 0.15 over 40 ms and
 * pauses the player) and `notify(true)` (send `duck_local {on: true}`); restore after
 * duckRestoreMs(rtt()) unless the server sent `duck`/`barge_in`. Never in the first
 * `echoGuardMs` of a reply, at most `maxPerReply` per reply, off for the voice chat
 * after two `echo_ignored` verdicts in a row. Not for the phone voice (it cannot pause).
 * `apply(false, {rewind})` restores (player resume + gain back to 1).
 *
 * The rewind rule (> 800 ms paused → restart the sentence) is judged by THIS
 * side's own pause when the phone ducked first: the server measures from its
 * later `duck` and under-counts by the detection delay, so a 1.08 s pause came
 * back as `resume {rewind: false}` mid-word (FR-works 9). `rewind` from the
 * server is honoured as well; a pause the phone did not start is the server's call.
 */
export const REWIND_AFTER_MS = 800;
export function createOptimisticDuck({
  apply, notify = () => {}, rtt = () => 0, now = () => Date.now(),
  setTimer = (fn, ms) => setTimeout(fn, ms), clearTimer = (h) => clearTimeout(h),
  echoGuardMs = 1500, maxPerReply = 2, enabled = () => true,
} = {}) {
  if (typeof apply !== 'function') throw new TypeError('apply is required');
  let replyAt = null;
  let count = 0;
  let echoStreak = 0;
  let disabled = false;
  let state = 'none'; // none | local | server
  let timer = null;
  let localAt = null;  // when THIS side paused the reply (null: the server ducked first, or nothing)
  const stopTimer = () => { if (timer !== null) clearTimer(timer); timer = null; };
  const restore = (info = {}) => {
    stopTimer();
    localAt = null;
    if (state === 'none') return false;
    const was = state;
    state = 'none';
    apply(false, info);
    if (was === 'local') notify(false);
    return true;
  };
  return {
    get state() { return state; },
    get disabled() { return disabled; },
    onReplyStart() { replyAt = now(); count = 0; },
    onReplyEnd() { replyAt = null; if (state === 'local') restore(); },
    /** The mic worklet's {speech} edge. Returns true when it ducked. */
    onSpeech(on) {
      if (!on || disabled || state !== 'none' || replyAt === null || !enabled()) return false;
      if (now() - replyAt < echoGuardMs || count >= maxPerReply) return false;
      count += 1;
      state = 'local';
      localAt = now();
      apply(true, {});
      notify(true);
      // The same > 800 ms rule as the server's `resume` (C.4): a longer pause restarts the sentence.
      const ms = duckRestoreMs(rtt());
      timer = setTimer(() => { timer = null; if (state === 'local') restore({ rewind: ms > REWIND_AFTER_MS, timeout: true }); }, ms);
      return true;
    },
    /** Server `duck {on}` (server audio only; with the phone voice it is informational). */
    onServerDuck(on) {
      if (on) {
        stopTimer();
        if (state === 'none') apply(true, { server: true });
        state = 'server';
      } else {
        // `duck off` comes just before the server's `resume`: a pause this side
        // started > 800 ms ago restarts the sentence here (FR-works 9).
        restore({ rewind: localAt !== null && now() - localAt > REWIND_AFTER_MS });
      }
    },
    /**
     * Server `resume {gen, rewind}` (the caller checks gen). Always reaches the player, even
     * when a `duck off` already restored it: the worklet honours a late `rewind` for the
     * sentence it just resumed, so `duck off` → `resume` and `resume` → `duck off` agree.
     * Returns true when a duck was still in force.
     */
    onResume(rewind = false) {
      const pausedHere = localAt !== null && now() - localAt > REWIND_AFTER_MS;
      const info = { rewind: !!rewind || pausedHere };
      if (restore(info)) return true;
      apply(false, info);
      return false;
    },
    /** Server `barge_in`: the player is flushed by the caller; the duck is over. */
    onBargeIn() { stopTimer(); localAt = null; const was = state; state = 'none'; if (was !== 'none') apply(false, { flushed: true }); },
    /**
     * A new socket (`ready`): whatever duck was in force is over — the new session will never
     * send `duck off` for it. Restores the gain/player (`apply(false, {flushed: true})`); the
     * echo streak and `disabled` stand (they are about the car, not the socket). Call it beside
     * `player.reset()`, `claims.reset()` and `localVoice.reset()` (review T8 r2 M3).
     */
    reset() {
      stopTimer();
      localAt = null;
      const was = state;
      state = 'none';
      replyAt = null;
      count = 0;
      if (was !== 'none') apply(false, { flushed: true });
    },
    onEchoIgnored() { echoStreak += 1; if (echoStreak >= 2) disabled = true; },
    /** A turn that was not an echo resets the streak. */
    onTurnAccepted() { echoStreak = 0; },
    dispose() { stopTimer(); },
  };
}
