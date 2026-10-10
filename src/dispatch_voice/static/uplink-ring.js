// DisPatch Voice: uplink-ring.js — the last 30 s of what the mic sent, for replay
// after a reconnect (v3 SPEC §5.5, contracts C-§E.2, C.5 #10).
//
// Every uplink message (a wire3 frame from opus-codec.js) is kept here, keyed by
// its 16 kHz sample offset. On reconnect the client sends
// `start.cursor {stream, from: cursorFrom(...)}`; the server answers `cursor {from}`
// (= max(client_from, committed)) and the client replays the ring from there,
// re-packed with the `replay` flag (bit1) and up to 50 Opus packets per message.
// Staleness is judged per burst (a run of offset-contiguous packets, i.e. one stretch
// the gate let through) by the burst's LAST sample (amendment T6-1): a burst that ended
// more than 20 s before the live mic clock is not replayed — that speech is stale, and
// the UI says "That got cut off. Say it again." — and a burst is never cut in half, so
// the server never gets the tail of a sentence without its head. The server's
// `cursor {from}` answer may lie BELOW the client's own `from` (an utterance the old
// socket had only half heard): replay from the answer; trim only what the server has
// committed (a transcript / end-of-turn offset), never up to the last rx offset.
//
// Budget: 30 s of Opus ≈ 59 KB; PCM keeps 10 s (320 KB) — pass `seconds: 10` and
// `maxBytes` for it. The ring is per stream id (`s` in the socket URL).
//
// Owner: track T8. Pure, no DOM; node --test-able.

// Siblings load through this module's own query (`?v=<assets hash>`), so the browser keeps
// ONE instance of opus-codec.js (amendment T7a-3); in node the query is empty.
const Q = new URL(import.meta.url).search;
const {
  CODEC, MAX_UP_PACKETS, UP_FLAGS, isV3, packUp, packetSamples16k, unpackUp,
} = await import(`./opus-codec.js${Q}`);

const RATE = 16000;
const sdiff = (a, b) => ((a - b) | 0); // signed distance on the u32 sample clock

/**
 * The cursor a reconnect sends: max(lastRxOff, lastTranscriptUpto, lastEotUpto)
 * (hardening M3: the two transcript offsets stop a restart inside the 2 s pong gap
 * from replaying an utterance the server already submitted). Missing values are
 * ignored; compared on the wrapping u32 clock. All missing → 0.
 */
export function cursorFrom({ lastRxOff, lastTranscriptUpto, lastEotUpto } = {}) {
  let best = null;
  for (const v of [lastRxOff, lastTranscriptUpto, lastEotUpto]) {
    if (typeof v !== 'number' || !Number.isFinite(v)) continue;
    const u = Math.trunc(v) >>> 0;
    if (best === null || sdiff(u, best) > 0) best = u;
  }
  return best === null ? 0 : best;
}

/**
 * @param {{seconds?: number, maxBytes?: number, stream?: string|null}} [opts]
 */
export function createUplinkRing({ seconds = 30, maxBytes = Infinity, stream = null } = {}) {
  /** [{off, codec, flags, packets: Uint8Array[], lens: number[], samples, bytes}] in offset order. */
  let entries = [];
  let bytes = 0;
  let cutOff = false;
  const span = Math.round(seconds * RATE);
  const endOf = (e) => (e.off + e.samples) >>> 0;

  const evict = () => {
    if (!entries.length) return;
    const newest = endOf(entries[entries.length - 1]);
    while (entries.length > 1 && sdiff(newest, entries[0].off) > span) bytes -= entries.shift().bytes;
    while (entries.length > 1 && bytes > maxBytes) bytes -= entries.shift().bytes;
  };

  return {
    stream,
    /** Keep one sent message. `buf` is a wire3 uplink frame (a raw PCM16 buffer is kept as codec 0). */
    put(off, buf) {
      let codec = CODEC.pcm16;
      let flags = 0;
      let packets;
      let at = off >>> 0;
      if (isV3(buf)) {
        const f = unpackUp(buf);
        ({ codec, flags, packets } = f);
        at = f.off;
      } else {
        packets = [new Uint8Array(buf instanceof ArrayBuffer ? buf.slice(0) : buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))];
      }
      const lens = packets.map((p) => packetSamples16k(codec, p));
      const samples = lens.reduce((a, b) => a + b, 0);
      const n = packets.reduce((a, p) => a + p.length, 0);
      // A stream that went backwards restarted: what we held belongs to the old clock.
      if (entries.length && sdiff(at, entries[entries.length - 1].off) < 0) { entries = []; bytes = 0; }
      entries.push({ off: at, codec, flags: flags & ~UP_FLAGS.replay, packets, lens, samples, bytes: n });
      bytes += n;
      evict();
    },
    /**
     * Messages to replay everything from `off` on, re-packed with the replay flag.
     * Packet granularity: a packet that contains `off` is included whole (the
     * server drops the overlap). With `nowOff` (the live mic clock), a burst whose
     * last sample is more than `maxAgeS` before it is skipped whole and `cutOff`
     * turns true; a burst that ended inside the window is replayed whole from `off`,
     * even if it began earlier (T6-1: never half a burst).
     * Opus: ≤ 50 packets per message; PCM: 1 × 640 B per message (C.2).
     */
    replayFrom(off, { nowOff = null, maxAgeS = 20 } = {}) {
      off >>>= 0;
      cutOff = false;
      const staleBefore = nowOff == null ? null : ((nowOff >>> 0) - Math.round(maxAgeS * RATE)) >>> 0;
      // 1. Bursts: maximal runs of offset-contiguous packets (codec/flag changes do not split one).
      const bursts = [];
      let b = null;
      for (const e of entries) {
        let pOff = e.off;
        for (let i = 0; i < e.packets.length; i++) {
          const start = pOff;
          const pEnd = (pOff + e.lens[i]) >>> 0;
          pOff = pEnd;
          if (!b || b.end !== start) { b = { end: pEnd, packets: [] }; bursts.push(b); }
          b.packets.push({ start, end: pEnd, codec: e.codec, flags: e.flags, data: e.packets[i] });
          b.end = pEnd;
        }
      }
      // 2. Re-pack what the server lacks, burst by burst.
      const out = [];
      let run = null; // {codec, flags, off, next, packets}
      const close = () => { if (run) out.push(packUp(run.codec, run.flags | UP_FLAGS.replay, run.off, run.packets)); run = null; };
      for (const burst of bursts) {
        if (sdiff(burst.end, off) <= 0) continue; // all of it already received
        if (staleBefore !== null && sdiff(burst.end, staleBefore) < 0) { cutOff = true; continue; }
        for (const p of burst.packets) {
          if (sdiff(p.end, off) <= 0) continue; // already received
          const max = p.codec === CODEC.opus ? MAX_UP_PACKETS : 1;
          if (!run || run.codec !== p.codec || run.flags !== p.flags || run.next !== p.start || run.packets.length >= max) {
            close();
            run = { codec: p.codec, flags: p.flags, off: p.start, next: p.start, packets: [] };
          }
          run.packets.push(p.data);
          run.next = p.end;
        }
      }
      close();
      return out;
    },
    /** True when the last replayFrom() skipped speech for being older than maxAgeS. */
    get cutOff() { return cutOff; },
    /** Offset of the oldest sample held, or null. */
    oldest() { return entries.length ? entries[0].off : null; },
    /** Offset just past the newest sample held, or null. */
    newest() { return entries.length ? endOf(entries[entries.length - 1]) : null; },
    /** Forget every message that ends at or before `off` (acknowledged by the server). */
    trim(off) {
      off >>>= 0;
      while (entries.length && sdiff(endOf(entries[0]), off) <= 0) bytes -= entries.shift().bytes;
    },
    clear() { entries = []; bytes = 0; cutOff = false; },
    get bytes() { return bytes; },
    get size() { return entries.length; },
    /** Seconds of audio held (from the oldest sample to the newest). */
    get seconds() { return entries.length ? sdiff(endOf(entries[entries.length - 1]), entries[0].off) / RATE : 0; },
  };
}

/**
 * Sends replay messages paced at `factor` × realtime so the server's byte-bounded
 * audio queue (64,000 B) is never flooded (10 s of PCM is 320 KB). Opus replays
 * (≈ 59 KB for 30 s) go out in a few messages either way.
 *   start(messages) / stop(); onDone() after the last one.
 * Timers are injected (setTimer/clearTimer) for tests.
 */
export function createReplayPump({
  send, factor = 4, onDone = () => {},
  setTimer = (fn, ms) => setTimeout(fn, ms), clearTimer = (h) => clearTimeout(h),
} = {}) {
  if (typeof send !== 'function') throw new TypeError('send is required');
  let queue = [];
  let timer = null;
  const msOf = (buf) => {
    try {
      const f = unpackUp(buf);
      return f.packets.reduce((a, p) => a + packetSamples16k(f.codec, p), 0) / RATE * 1000;
    } catch { return 0; }
  };
  const step = () => {
    timer = null;
    const buf = queue.shift();
    if (buf === undefined) { onDone(); return; }
    send(buf);
    if (!queue.length) { onDone(); return; }
    timer = setTimer(step, msOf(buf) / factor);
  };
  return {
    start(messages) {
      this.stop();
      queue = [...messages];
      if (!queue.length) { onDone(); return; }
      step();
    },
    stop() { if (timer !== null) clearTimer(timer); timer = null; queue = []; },
    get pending() { return queue.length; },
  };
}
