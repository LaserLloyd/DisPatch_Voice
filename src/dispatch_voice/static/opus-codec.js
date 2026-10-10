// DisPatch Voice: opus-codec.js — v3 binary framing + the browser's Opus (WebCodecs),
// with a PCM fallback both ways (v3 contracts C-§C.1, C.2, C-§E.2).
//
// What it does
//   wire3.*          pack/unpack the v3 binary frames (DataView, pure). The Python
//                    side (transport/wire3.py, track T6) reads the same fixture file.
//   opusSupport()    asks WebCodecs whether the two measured configurations work:
//                    encode 16 kHz mono "voip"/"voice" complexity 5, decode at 48 kHz.
//   createUplink()   mic frames (320 Int16 @ 16 kHz from the worklet, with `off`) →
//                    Opus packets, 5 per message (100 ms) → wire3 uplink frames, into
//                    the replay ring and onto the socket. PCM: 1 × 640 B per message.
//                    An encoder error falls back to PCM without losing a frame.
//   createDownlink() v3 downlink frames → decoded Float32 → the player worklet. Opus
//                    decodes at 48 kHz on every browser; PCM is 24 kHz.
//
// Measured (design prototype, Chrome 151 headless): every config above supported; the
// server's libopus decodes the browser's 16 kHz packets with WER 0.0 at 16 and
// 12 kbps; encode ≈ 272 µs per 20 ms frame, decode ≈ 126–155 µs per packet. A 48 kHz
// encoder config is NOT used: the uplink is 16 kHz end to end.
//
// AudioEncoder/AudioDecoder exist on the main thread only (not in
// AudioWorkletGlobalScope), so this module runs on the main thread. Every
// WebCodecs class is injectable (`codecs`) so node --test drives it with fakes.
//
// Owner: track T8.

// ---------------------------------------------------------------- wire3 (C.2)

export const MAGIC = 0x03;
export const CODEC = Object.freeze({ pcm16: 0, opus: 1 });
export const CODEC_NAMES = Object.freeze(['pcm16', 'opus']);
/** Downlink `kind` byte. */
export const KIND = Object.freeze({ reply: 0, ack: 1, notice: 2, probe: 3, backchannel: 4 });
export const KIND_NAMES = Object.freeze(['reply', 'ack', 'notice', 'probe', 'backchannel']);
/** Uplink `flags` bits. */
export const UP_FLAGS = Object.freeze({ localSpeaking: 0x01, replay: 0x02 });
export const UP_HEADER = 8;       // magic, codec, flags, off u32, count
export const DOWN_HEADER = 14;    // magic, codec, kind, gen u16, seq u32, t u32, count
export const MAX_UP_PACKETS = 50; // server: more → error {code: bad_frame}
export const MAX_FRAME_BYTES = 65536; // server: a larger binary message is dropped pre-enqueue
export const LIVE_OPUS_PACKETS = 5;   // 100 ms per live uplink message
export const PCM_UP_BYTES = 640;      // 20 ms of 16 kHz PCM16
export const DOWN_PCM_RATE = 24000;
export const DOWN_OPUS_RATE = 48000;

/** A malformed v3 frame. `name` is 'WireError' so callers can tell it from a bug. */
export class WireError extends Error {
  constructor(message) { super(message); this.name = 'WireError'; }
}

function bytesOf(x) {
  if (x instanceof Uint8Array) return x;
  if (x instanceof ArrayBuffer) return new Uint8Array(x);
  if (ArrayBuffer.isView(x)) return new Uint8Array(x.buffer, x.byteOffset, x.byteLength);
  throw new TypeError('expected an ArrayBuffer or a typed array');
}

function u8(name, v) {
  if (!Number.isInteger(v) || v < 0 || v > 0xff) throw new RangeError(`${name} must be a u8, got ${v}`);
  return v;
}

function knownCodec(codec) {
  if (codec !== CODEC.pcm16 && codec !== CODEC.opus) throw new RangeError(`unknown codec ${codec}`);
  return codec;
}

function packPackets(out, at, packets) {
  for (const p of packets) {
    out[at] = p.length & 0xff;
    out[at + 1] = p.length >> 8;
    out.set(p, at + 2);
    at += 2 + p.length;
  }
  return at;
}

// The rules match transport/wire3.py exactly (the shared fixture holds both to them):
// 1..max packets, each 1..65535 B, PCM16 packets even-length (a probe's random bytes excepted).
function normPackets(packets, max, pcmEven) {
  if (!Array.isArray(packets)) throw new TypeError('packets must be an array');
  if (packets.length < 1 || packets.length > max) throw new RangeError(`packet count ${packets.length} outside 1..${max}`);
  return packets.map((p) => {
    const b = bytesOf(p);
    if (b.length < 1 || b.length > 0xffff) throw new RangeError(`packet length ${b.length} outside 1..65535`);
    if (pcmEven && b.length % 2) throw new RangeError('PCM16 packet with an odd byte count');
    return b;
  });
}

/**
 * Uplink frame: 0x03 | codec u8 | flags u8 | off u32 LE | count u8 | count × (len u16 LE + bytes).
 * `off` is taken modulo 2^32 (it wraps). Returns a Uint8Array that owns its buffer.
 */
export function packUp(codec, flags, off, packets) {
  knownCodec(codec);
  u8('flags', flags);
  if (!Number.isFinite(off)) throw new RangeError('off must be a number');
  const ps = normPackets(packets, MAX_UP_PACKETS, codec === CODEC.pcm16);
  const n = UP_HEADER + ps.reduce((a, p) => a + 2 + p.length, 0);
  if (n > MAX_FRAME_BYTES) throw new RangeError(`uplink frame of ${n} B exceeds ${MAX_FRAME_BYTES}`);
  const out = new Uint8Array(n);
  const dv = new DataView(out.buffer);
  out[0] = MAGIC; out[1] = codec; out[2] = flags;
  dv.setUint32(3, Math.trunc(off) >>> 0, true);
  out[7] = ps.length;
  packPackets(out, UP_HEADER, ps);
  return out;
}

function readPackets(b, at, count, pcmEven) {
  const packets = [];
  for (let i = 0; i < count; i++) {
    if (at + 2 > b.length) throw new WireError(`truncated: packet ${i} has no length`);
    const len = b[at] | (b[at + 1] << 8);
    at += 2;
    if (len === 0) throw new WireError(`packet ${i} is empty`);
    if (at + len > b.length) throw new WireError(`truncated: packet ${i} wants ${len} B`);
    if (pcmEven && len % 2) throw new WireError(`PCM16 packet ${i} with an odd byte count`);
    packets.push(b.slice(at, at + len));
    at += len;
  }
  if (at !== b.length) throw new WireError(`${b.length - at} trailing bytes`);
  return packets;
}

/** → {codec, flags, off, packets}; throws WireError on a bad frame (as unpack_up raises ValueError). */
export function unpackUp(buf) {
  const b = bytesOf(buf);
  if (b.length > MAX_FRAME_BYTES) throw new WireError(`uplink frame of ${b.length} B exceeds ${MAX_FRAME_BYTES}`);
  if (b.length < UP_HEADER) throw new WireError(`short frame (${b.length} B)`);
  if (b[0] !== MAGIC) throw new WireError(`bad magic 0x${b[0].toString(16)}`);
  const codec = b[1];
  if (codec > 1) throw new WireError(`unknown codec ${codec}`);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const count = b[7];
  if (count < 1 || count > MAX_UP_PACKETS) throw new WireError(`packet count ${count} outside 1..${MAX_UP_PACKETS}`);
  return { codec, flags: b[2], off: dv.getUint32(3, true), packets: readPackets(b, UP_HEADER, count, codec === CODEC.pcm16) };
}

/**
 * Downlink frame: 0x03 | codec u8 | kind u8 | gen u16 LE | seq u32 LE | t u32 LE | count u8 |
 * count × (len u16 LE + bytes). `gen` is taken modulo 2^16, `seq`/`t` modulo 2^32.
 */
export function packDown(codec, kind, gen, seq, t, packets) {
  knownCodec(codec);
  if (!Number.isInteger(kind) || kind < 0 || kind >= KIND_NAMES.length) throw new RangeError(`unknown kind ${kind}`);
  const ps = normPackets(packets, 0xff, codec === CODEC.pcm16 && kind !== KIND.probe);
  const n = DOWN_HEADER + ps.reduce((a, p) => a + 2 + p.length, 0);
  const out = new Uint8Array(n);
  const dv = new DataView(out.buffer);
  out[0] = MAGIC; out[1] = codec; out[2] = kind;
  dv.setUint16(3, Math.trunc(gen) & 0xffff, true);
  dv.setUint32(5, Math.trunc(seq) >>> 0, true);
  dv.setUint32(9, Math.trunc(t) >>> 0, true);
  out[13] = ps.length;
  packPackets(out, DOWN_HEADER, ps);
  return out;
}

/** → {codec, kind, gen, seq, t, packets}; throws WireError on a bad frame. */
export function unpackDown(buf) {
  const b = bytesOf(buf);
  if (b.length < DOWN_HEADER) throw new WireError(`short frame (${b.length} B)`);
  if (b[0] !== MAGIC) throw new WireError(`bad magic 0x${b[0].toString(16)}`);
  const codec = b[1];
  if (codec > 1) throw new WireError(`unknown codec ${codec}`);
  const kind = b[2];
  if (kind >= KIND_NAMES.length) throw new WireError(`unknown kind ${kind}`);
  const count = b[13];
  if (count < 1) throw new WireError('no packets');
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  return {
    codec,
    kind,
    gen: dv.getUint16(3, true),
    seq: dv.getUint32(5, true),
    t: dv.getUint32(9, true),
    packets: readPackets(b, DOWN_HEADER, count, codec === CODEC.pcm16 && kind !== KIND.probe),
  };
}

/** True when a binary message is a v3 frame (a v2 frame is raw PCM16 and never starts with 0x03 by contract). */
export function isV3(buf) {
  const b = bytesOf(buf);
  return b.length > 0 && b[0] === MAGIC;
}

export const wire3 = Object.freeze({ packUp, unpackUp, packDown, unpackDown, isV3 });

// ------------------------------------------------------- Opus packet duration

// RFC 6716 §3.1: TOC byte → frame size; frame-count code → frames per packet.
const SILK_MS = [10, 20, 40, 60];
const CELT_MS = [2.5, 5, 10, 20];

/** Duration of one Opus packet in ms, from its TOC byte (0 for an empty packet). */
export function opusPacketMs(packet) {
  const p = bytesOf(packet);
  if (!p.length) return 0;
  const config = p[0] >> 3;
  const code = p[0] & 3;
  let frameMs;
  if (config < 12) frameMs = SILK_MS[config & 3];
  else if (config < 16) frameMs = (config & 1) ? 20 : 10;
  else frameMs = CELT_MS[config & 3];
  let frames = 1;
  if (code === 1 || code === 2) frames = 2;
  else if (code === 3) frames = p.length > 1 ? (p[1] & 0x3f) : 0;
  return frames * frameMs;
}

/** Samples at 16 kHz an uplink packet covers (Opus by TOC, PCM16 by length). */
export function packetSamples16k(codec, packet) {
  const p = bytesOf(packet);
  return codec === CODEC.opus ? Math.round(opusPacketMs(p) * 16) : p.length >> 1;
}

// ------------------------------------------------------------- WebCodecs support

/** The uplink encoder configuration (measured matrix: 16 kHz, voip/voice, complexity 5). */
export function upConfig(kbps = 16) {
  return {
    codec: 'opus', sampleRate: 16000, numberOfChannels: 1, bitrate: Math.round(kbps * 1000),
    opus: { frameDuration: 20000, application: 'voip', signal: 'voice', complexity: 5, format: 'opus' },
  };
}

/** The downlink decoder configuration (Opus decodes at 48 kHz on every browser). */
export function downConfig() {
  return { codec: 'opus', sampleRate: DOWN_OPUS_RATE, numberOfChannels: 1 };
}

function codecsOf(codecs) {
  const g = globalThis;
  return {
    AudioEncoder: codecs?.AudioEncoder ?? g.AudioEncoder,
    AudioDecoder: codecs?.AudioDecoder ?? g.AudioDecoder,
    AudioData: codecs?.AudioData ?? g.AudioData,
    EncodedAudioChunk: codecs?.EncodedAudioChunk ?? g.EncodedAudioChunk,
  };
}

/**
 * {up, down}: whether WebCodecs can do each direction. Called once before the
 * socket URL is built; `up=opus` / `down=opus` go in the query only when true (C.1).
 */
export async function opusSupport({ codecs } = {}) {
  const c = codecsOf(codecs);
  const ask = async (Cls, cfg) => {
    if (typeof Cls?.isConfigSupported !== 'function') return false;
    try { return !!(await Cls.isConfigSupported(cfg))?.supported; } catch { return false; }
  };
  const [up, down] = await Promise.all([
    c.AudioData ? ask(c.AudioEncoder, upConfig(16)) : false,
    c.EncodedAudioChunk ? ask(c.AudioDecoder, downConfig()) : false,
  ]);
  return { up, down };
}

/** The `up=`/`down=` query values (best first), from opusSupport()'s answer. */
export function codecQuery({ up, down } = {}) {
  return { up: up ? 'opus,pcm16' : 'pcm16', down: down ? 'opus,pcm16' : 'pcm16' };
}

// ------------------------------------------------------------------ uplink

function int16Of(frame) {
  if (frame instanceof Int16Array) return frame;
  if (frame instanceof ArrayBuffer) return new Int16Array(frame);
  if (ArrayBuffer.isView(frame)) return new Int16Array(frame.buffer, frame.byteOffset, frame.byteLength >> 1);
  throw new TypeError('mic frame must be an Int16Array or ArrayBuffer');
}

const sdiff = (a, b) => ((a - b) | 0); // signed distance on the u32 sample clock

/**
 * The one default clock of the T8 modules: monotonic epoch-ms. The same function as
 * `link-monitor.js`'s `monoNow` (kept separate so neither module imports the other);
 * the downlink's `arrival` feeds `monitor.onChunk`, so both must read one clock — a
 * caller that injects `now` injects the same one into both (review T8 r2 L3).
 */
export const monoNow = () => (globalThis.performance && typeof globalThis.performance.now === 'function'
  ? globalThis.performance.timeOrigin + globalThis.performance.now() : Date.now());

/**
 * The uplink. `codec` 'opus' or 'pcm16'; `kbps` 16 or 12 (Opus only).
 *   push(frame, off)   one 20 ms mic frame (320 Int16) with its 16 kHz offset
 *   flush()            the gate closed: emit what is pending (→ Promise)
 *   setKbps(kbps)      the codec ladder rung (applies from the next frame)
 *   close()
 * Every message goes to `ring.put(off, buf)` (when a ring is given) and `onMessage(buf, info)`.
 * `flags()` is read per frame as it is pushed (bit0 local_speaking); a message carries the
 * OR of its frames. On an encoder error the
 * uplink switches to PCM16 for good, re-sends the frames the encoder still held,
 * and calls `onError('encoder', err)` once (the client then sends `link {why: "encoder"}`).
 * An encoder that stops answering without an error (a WebCodecs stall after a reconfigure
 * or backgrounding) is treated the same: it holds at least `minStallFrames` frames (a
 * healthy encoder holds one or two) and the oldest waited more than `stallMs` of wall time,
 * or it holds `maxInFlight` frames, or `flush()` did not finish within `stallMs`. Both a
 * count and wall time, so a main-thread hitch that delivers a burst of queued mic frames
 * at once (their outputs queued in between) is not mistaken for a stall.
 * NOTE: `onError` may run synchronously inside this call (no WebCodecs at all) — the
 * handler must not use the returned uplink.
 */
export const ENCODER_STALL_MS = 2000;
export const ENCODER_MIN_STALL_FRAMES = 25; // 500 ms held
export const ENCODER_MAX_IN_FLIGHT = 500;   // 10 s of 20 ms frames: a memory backstop

export function createUplink({
  codec = 'opus', kbps = 16, onMessage, ring = null, flags = () => 0, onError = () => {}, codecs,
  now = monoNow,
  setTimer = (f, ms) => setTimeout(f, ms), clearTimer = (h) => clearTimeout(h),
  stallMs = ENCODER_STALL_MS, maxInFlight = ENCODER_MAX_IN_FLIGHT, minStallFrames = ENCODER_MIN_STALL_FRAMES,
} = {}) {
  if (typeof onMessage !== 'function') throw new TypeError('onMessage is required');
  const c = codecsOf(codecs);
  let mode = codec === 'opus' ? 'opus' : 'pcm16';
  let rate = kbps;
  let enc = null;
  let closed = false;
  /** Frames handed to the encoder and not yet output: [{off, frame}]. */
  let inFlight = [];
  /** Opus packets waiting for the next message. */
  let pend = null; // {off, next, flags, packets}
  let tsBase = null; // unwrapped µs timestamps for AudioData
  let lastOff = null;
  let sent = 0;

  const frameFlags = () => (Number(flags()) || 0) & 0xff & ~UP_FLAGS.replay;
  const emit = (codecId, off, packets, fl) => {
    const buf = packUp(codecId, fl, off, packets);
    if (ring) ring.put(off, buf);
    sent += 1;
    onMessage(buf, { off, codec: CODEC_NAMES[codecId], packets: packets.length });
  };
  const emitPend = () => {
    if (pend && pend.packets.length) emit(CODEC.opus, pend.off, pend.packets, pend.flags);
    pend = null;
  };
  const pcmFrame = (off, frame, fl = frameFlags()) => {
    const i16 = int16Of(frame);
    emit(CODEC.pcm16, off, [new Uint8Array(i16.buffer.slice(i16.byteOffset, i16.byteOffset + i16.byteLength))], fl);
  };

  const fail = (err) => {
    if (mode !== 'opus') return;
    mode = 'pcm16';
    emitPend();
    const held = inFlight;
    inFlight = [];
    try { enc && enc.state !== 'closed' && enc.close(); } catch { /* already gone */ }
    enc = null;
    for (const f of held) pcmFrame(f.off, f.frame, f.flags);
    onError('encoder', err);
  };

  const onOutput = (chunk) => {
    const f = inFlight.shift();
    if (!f) return; // nothing we sent (cannot happen with 20 ms frames)
    const bytes = new Uint8Array(chunk.byteLength);
    chunk.copyTo(bytes);
    if (pend && pend.next !== f.off) emitPend(); // a gate gap: a message is contiguous
    if (!pend) pend = { off: f.off, next: f.off, flags: 0, packets: [] };
    pend.flags |= f.flags; // local_speaking during any frame of the message
    pend.packets.push(bytes);
    pend.next = (f.off + 320) >>> 0;
    if (pend.packets.length >= LIVE_OPUS_PACKETS) emitPend();
  };

  const open = () => {
    if (mode !== 'opus') return;
    if (!c.AudioEncoder || !c.AudioData) { fail(new Error('WebCodecs AudioEncoder unavailable')); return; }
    try {
      enc = new c.AudioEncoder({ output: onOutput, error: (e) => fail(e) });
      enc.configure(upConfig(rate));
    } catch (e) { fail(e); }
  };
  open();

  return {
    get codec() { return mode; },
    get kbps() { return mode === 'opus' ? rate : null; },
    get sent() { return sent; },
    push(frame, off) {
      if (closed) return;
      off >>>= 0;
      if (mode !== 'opus') { pcmFrame(off, frame); return; }
      const i16 = int16Of(frame);
      if (lastOff === null) tsBase = 0;
      else tsBase += sdiff(off, lastOff) * 62.5;
      lastOff = off;
      const copy = new Int16Array(i16); // the worklet's buffer was transferred to us; keep our own
      const at = now();
      // A silent stall: the encoder holds frames it never returns. The frames go out as PCM.
      if (inFlight.length >= maxInFlight
          || (inFlight.length >= minStallFrames && at - inFlight[0].at > stallMs)) {
        fail(new Error('encoder stalled'));
        pcmFrame(off, copy);
        return;
      }
      inFlight.push({ off, frame: copy, flags: frameFlags(), at }); // flags as of capture, not of output
      try {
        const data = new c.AudioData({
          format: 's16', sampleRate: 16000, numberOfFrames: copy.length, numberOfChannels: 1,
          timestamp: Math.round(tsBase), data: copy,
        });
        enc.encode(data);
        data.close?.();
      } catch (e) { fail(e); }
    },
    async flush() {
      if (closed) return;
      if (mode === 'opus' && enc) {
        let timer = null;
        const stalled = new Promise((resolve) => { timer = setTimer(() => resolve('stalled'), stallMs); });
        try {
          const r = await Promise.race([enc.flush().then(() => 'ok'), stalled]);
          if (r === 'stalled') fail(new Error('encoder flush stalled'));
        } catch (e) { fail(e); } finally { clearTimer(timer); }
      }
      emitPend();
    },
    setKbps(kbps2) {
      if (kbps2 === rate) return;
      rate = kbps2;
      if (mode === 'opus' && enc) {
        // Chrome reconfigures in place; queued frames keep their order.
        try { enc.configure(upConfig(rate)); } catch (e) { fail(e); }
      }
    },
    close() {
      closed = true;
      try { enc && enc.state !== 'closed' && enc.close(); } catch { /* gone */ }
      enc = null; inFlight = []; pend = null;
    },
  };
}

/** C-§E.2 name: the Opus uplink (falls back to PCM16 itself). */
export function createOpusUplink(opts = {}) { return createUplink({ ...opts, codec: 'opus' }); }

// ---------------------------------------------------------------- downlink

function pcm16ToFloat(bytes) {
  const n = bytes.length >> 1;
  const out = new Float32Array(n);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, n * 2);
  for (let i = 0; i < n; i++) out[i] = dv.getInt16(i * 2, true) / 32768;
  return out;
}

/**
 * The downlink. `feed(buf, arrivalMs)` takes one binary v3 frame.
 *   kind probe  → onProbe({bytes, arrival, gen, seq}) (never decoded)
 *   PCM16       → player.push({type:'push', gen, seq, kind, rate: 24000, pcm})
 *   Opus        → decoded at 48 kHz by AudioDecoder, then the same push
 * `onChunk({gen, seq, kind, t, arrival, durationMs, bytes, codec})` fires on receipt
 * (before decoding) for the link monitor. A decoder failure calls
 * `onError('decoder', err)` once and drops Opus frames from then on (the client
 * reconnects with `down=pcm16`, C.3 `link {why: "decoder"}`).
 * A decoder that stops answering without an error (the twin of the encoder stall) is the
 * same failure: when it holds at least `minStallPackets` packets and the oldest was fed more
 * than `stallMs` ago, or it holds `maxPending`, the downlink waits `confirmMs` more (so
 * decoder outputs queued behind a main-thread hitch can run) and fails when none came; a
 * `flush()` that does not settle within `stallMs` fails at once (review T8 r2 M2).
 * `player` is anything with push(msg) — the player worklet client.
 * `now` defaults to `monoNow`, the link monitor's clock: `arrival` goes to `onChunk`.
 */
export const DECODER_STALL_MS = 2000;
export const DECODER_MIN_STALL_PACKETS = 25;  // 500 ms of 20 ms packets held
export const DECODER_MAX_PENDING = 500;       // 10 s held: a memory backstop
export const DECODER_CONFIRM_MS = 250;

export function createDownlink({
  player, onChunk = () => {}, onProbe = () => {}, onError = () => {}, codecs, now = monoNow,
  setTimer = (f, ms) => setTimeout(f, ms), clearTimer = (h) => clearTimeout(h),
  stallMs = DECODER_STALL_MS, minStallPackets = DECODER_MIN_STALL_PACKETS,
  maxPending = DECODER_MAX_PENDING, confirmMs = DECODER_CONFIRM_MS,
} = {}) {
  if (!player || typeof player.push !== 'function') throw new TypeError('player.push is required');
  const c = codecsOf(codecs);
  let dec = null;
  let broken = false;
  let ts = 0;
  /** Metadata per packet in decode order: [{ts, gen, seq, kind, at}]. */
  let metas = [];
  let outputs = 0;      // decoder outputs seen (progress, for the stall watchdog)
  let confirm = null;   // the stall confirmation timer

  const stopConfirm = () => { if (confirm !== null) clearTimer(confirm); confirm = null; };
  const fail = (err) => {
    if (broken) return;
    broken = true;
    stopConfirm();
    metas = [];
    try { dec && dec.state !== 'closed' && dec.close(); } catch { /* gone */ }
    dec = null;
    onError('decoder', err);
  };
  /** Suspect a stall: confirm it only if no output arrives within confirmMs. */
  const suspect = () => {
    if (confirm !== null || broken) return;
    const seen = outputs;
    confirm = setTimer(() => {
      confirm = null;
      if (!broken && outputs === seen && metas.length) fail(new Error('decoder stalled'));
    }, confirmMs);
  };
  const watchdog = (at) => {
    if (metas.length >= maxPending || (metas.length >= minStallPackets && at - metas[0].at > stallMs)) suspect();
  };
  const onOutput = (ad) => {
    outputs += 1;
    // Outputs come in decode order; match by timestamp, tolerate a packet that produced none.
    if (typeof ad.timestamp === 'number' && metas.some((x) => x.ts === ad.timestamp)) {
      while (metas[0].ts !== ad.timestamp) metas.shift();
    }
    const m = metas.shift();
    try {
      if (m) {
        const pcm = new Float32Array(ad.numberOfFrames);
        ad.copyTo(pcm, { planeIndex: 0, format: 'f32-planar' });
        player.push({ type: 'push', gen: m.gen, seq: m.seq, kind: m.kind, rate: ad.sampleRate, pcm });
      }
    } finally { ad.close?.(); }
  };
  const decoder = () => {
    if (dec || broken) return dec;
    if (!c.AudioDecoder || !c.EncodedAudioChunk) { fail(new Error('WebCodecs AudioDecoder unavailable')); return null; }
    try {
      dec = new c.AudioDecoder({ output: onOutput, error: (e) => fail(e) });
      dec.configure(downConfig());
    } catch (e) { fail(e); }
    return dec;
  };

  return {
    get broken() { return broken; },
    /** Packets handed to the decoder and not yet output. */
    get pending() { return metas.length; },
    feed(buf, arrival = now()) {
      const f = unpackDown(buf);
      const bytes = bytesOf(buf).length;
      const kindName = KIND_NAMES[f.kind] || String(f.kind);
      if (f.kind === KIND.probe) { onProbe({ bytes, arrival, gen: f.gen, seq: f.seq }); return f; }
      let durationMs = 0;
      if (f.codec === CODEC.opus) for (const p of f.packets) durationMs += opusPacketMs(p);
      else for (const p of f.packets) durationMs += (p.length >> 1) / (DOWN_PCM_RATE / 1000);
      onChunk({ gen: f.gen, seq: f.seq, kind: kindName, t: f.t, arrival, durationMs, bytes, codec: CODEC_NAMES[f.codec] });
      if (f.codec === CODEC.pcm16) {
        for (const p of f.packets) {
          player.push({ type: 'push', gen: f.gen, seq: f.seq, kind: kindName, rate: DOWN_PCM_RATE, pcm: pcm16ToFloat(p) });
        }
        return f;
      }
      const d = decoder();
      if (!d) return f;
      const at = now();
      watchdog(at);
      for (const p of f.packets) {
        const pms = opusPacketMs(p) || 20;
        metas.push({ ts, gen: f.gen, seq: f.seq, kind: kindName, at });
        try {
          d.decode(new c.EncodedAudioChunk({ type: 'key', timestamp: ts, duration: pms * 1000, data: p }));
        } catch (e) { fail(e); return f; }
        ts += pms * 1000;
      }
      watchdog(at);
      return f;
    },
    /** Resolves when every packet fed so far has been decoded and pushed to the player. */
    async flush() {
      if (!dec || broken) return;
      let timer = null;
      const stalled = new Promise((resolve) => { timer = setTimer(() => resolve('stalled'), stallMs); });
      try {
        const r = await Promise.race([dec.flush().then(() => 'ok'), stalled]);
        if (r === 'stalled') fail(new Error('decoder flush stalled'));
      } catch (e) { fail(e); } finally { clearTimer(timer); }
    },
    /** Drop what is queued in the decoder (a barge-in); the player's flush does the rest. */
    reset() {
      metas = [];
      stopConfirm();
      if (dec && !broken) { try { dec.reset(); dec.configure(downConfig()); } catch (e) { fail(e); } }
    },
    close() {
      stopConfirm();
      try { dec && dec.state !== 'closed' && dec.close(); } catch { /* gone */ }
      dec = null; metas = [];
    },
  };
}

/** C-§E.2 name: the Opus downlink (PCM16 frames pass straight through). */
export function createOpusDownlink(opts = {}) { return createDownlink(opts); }
