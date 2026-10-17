// DisPatch Voice: link-monitor.js — is the network keeping up? (v3 SPEC §5.3,
// contracts C-§E.2, C.3 `link`). Pure; every clock is injected.
//
// Inputs (fed by the client):
//   onPong(rttMs, rxOff, sentOff)     every 2 s ping's answer: RTT EWMA/p90 and the uplink
//                                     backlog_s = max(0, (sentOff − rxOff)/16000 − rtt/2)
//   onSay(seq, tServer, kind)         arms the first-audio deadline for a REPLY seq (server audio
//                                     only): max(5 s, p90(first_chunk_ms) + 2 s)
//   onChunk(seq, arrival, tServer, durationMs, leadMs, bytes)
//                                     clears the deadline; skew = arrival − tServer against its
//                                     minimum tells link delay from server (TTS) delay
//   onUnderrun(seq, gapMs)            the player's hole report
//   onStall()                         a pong more than 5 s late
//   onReady() / onClose()             socket edges
//   onSayEnd()                        the reply is over: a held predictive switch goes now; the
//                                     reply's deadlines and goodput window are cleared
//   onReplyCut()                      the reply is over WITHOUT a `say_end` (the server sends none for
//                                     a cut reply): a barge-in, a tap-stop, an undo, a failed turn, or
//                                     a new turn. Same effect as onSayEnd(); without it `replyActive`
//                                     (and a held predictive change) would block recovery for good
//   onTurnError(seq?)                 `turn_error` (e.g. tts_failed): no audio is coming for seq
//                                     (or for any armed seq) — a TTS fault is not the network's
//   onProbe(bytes, elapsedMs)         a 16 KB probe came back (elapsed = request → last byte)
//   probeFailed()                     the probe was refused (`error {rate_limited}`); an
//                                     unanswered probe also expires after probeTimeoutMs
//   onMetrics({first_chunk_ms})       calibrates the deadline
//   tick()                            evaluate timers (call every ~250 ms while open)
// Output: onChange({mode, downKbps, upKbps, why}) — send it as `link {mode, down_kbps,
// up_kbps, why}` — and onChange({backlogS, slow}).
//
// Rules (SPEC §5.3). Degraded when any holds:
//   · ≥ 2 underruns in one reply, or one gap > 400 ms, AND the skew has grown (the link) —
//     a TTS hole (≈ 1.1–1.35 s inside a long Chatterbox sentence even on a direct link)
//     has a flat skew and is never blamed on the network;
//   · RTT EWMA > 1 s on 3 consecutive pongs, or a pong > 5 s late;
//   · goodput < 1.3 × needed for 3 s while streaming AND the skew still rising inside the
//     window (a realtime TTS such as Chatterbox never delivers 1.3 × the need, so the ratio
//     alone means nothing; a link that cannot keep up shows as a rising skew);
//   · a first-audio deadline missed.
// "Skew grown" is measured against the best skew of THIS reply, never the socket's
// lifetime: a settled offset (a cell handover, an NTP step) is not a link falling behind.
// `arrival` should come from a monotonic clock (performance.timeOrigin + performance.now()).
// Recovery when all hold: healthy ≥ 30 s, between replies, RTT EWMA < 500 ms, the dwell
// over, and a probe at ≥ 2 × needed (≤ 1 per 20 s). Dwell 60 s, doubling per flap within
// 5 min (cap 10 min). The codec ladder (24→16 down / 16→12 up) is the first step down when
// Opus is on; a deadline miss or a stall goes straight to text (a lower bitrate cannot
// rescue a sentence that is already 5 s late). A predictive cause (rtt, goodput) that fires
// mid-reply is held until `say_end`, so the voice never changes mid-answer (experience
// M12); a reactive one (underrun, deadline, stall) is emitted at once.
//
// Never `bufferedAmount` (it read 0 while 5.4 s of audio sat in the kernel).
//
// Owner: track T8.

export const LINK_THRESHOLDS = Object.freeze({
  pingEveryMs: 2000,
  rttAlpha: 0.3,
  rttWindow: 10,              // pongs in the p90
  rttDegradedMs: 1000,
  rttDegradedPongs: 3,
  stallMs: 5000,              // a pong later than this = stall
  deadAfterMs: 8000,          // heartbeat: dead after the later of 3 missed pongs or 8 s …
  deadRttFactor: 4,           // … scaled to 4 × EWMA once the EWMA is over …
  deadScaleAboveMs: 1500,     // … 1.5 s (slow is not dead)
  underrunsPerReply: 2,
  bigGapMs: 400,
  skewGrowMs: 300,            // arrival − t_server this far over this reply's minimum: the link is behind
  skewRiseMs: 100,            // … and still rising this much inside the goodput window
  goodputFactor: 1.3,
  goodputWindowMs: 2000,
  goodputHoldMs: 3000,
  firstAudioMinMs: 5000,
  firstAudioPadMs: 2000,
  firstChunkPriorMs: 4000,    // Chatterbox, measured 3.9–4.1 s, until this voice chat has samples
  healthyMs: 30000,
  rttHealthyMs: 500,
  probeBytes: 16384,
  probeFactor: 2,
  probeEveryMs: 20000,
  probeTimeoutMs: 30000,      // an unanswered probe is written off after this
  dwellMs: 60000,
  dwellMaxMs: 600000,
  flapWindowMs: 300000,
  ladderGraceMs: 10000,       // a rung change applies at the next sentence: give it time
  audioGraceMinMs: 1500,      // after a switch back to audio, a `say` this soon (or within 2 × RTT)
                              // may still be a text-mode one: no deadline for it
  backlogSlowS: 2,
  backlogClearS: 0.3,
  /** On-the-wire kbps a mode needs (downlink + uplink, measured ≈ 28 + 30 at 24/16). */
  neededKbps: Object.freeze({ opusTop: 30, opusLow: 20, pcm16: 470 }),
  rungs: Object.freeze({ down: Object.freeze([24, 16]), up: Object.freeze([16, 12]) }),
  // Not '3g': Chrome reports it for RTT ≥ 270 ms, where the measured LTE/relay medians sit,
  // and 28 kbps Opus carries fine there (review T8 L3).
  slowTypes: Object.freeze(['slow-2g', '2g']),
});

const REACTIVE = new Set(['underrun', 'deadline', 'stall']);
const sdiff = (a, b) => ((a - b) | 0);

function p90(xs) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.ceil(0.9 * s.length) - 1)];
}

/** Heartbeat: ms without a pong before the socket is judged dead (SPEC §5.5). */
export function heartbeatDeadMs(rttEwmaMs, th = LINK_THRESHOLDS) {
  const r = Number(rttEwmaMs) || 0;
  return r > th.deadScaleAboveMs ? Math.max(th.deadAfterMs, th.deadRttFactor * r) : th.deadAfterMs;
}

/**
 * The one clock every T8 module defaults to (link monitor, downlink, uplink): monotonic
 * epoch-ms, so a wall-clock step (NTP after a flight-mode spell) never splits a window in
 * two. `opus-codec.js` exports the same function under the same name; a caller that injects
 * its own `now` must inject the SAME one into `createDownlink` and `createLinkMonitor`
 * (review T8 r2 L3).
 */
export const monoNow = () => (globalThis.performance && typeof globalThis.performance.now === 'function'
  ? globalThis.performance.timeOrigin + globalThis.performance.now() : Date.now());

/**
 * @param {{now?: () => number, thresholds?: object, onChange?: (e: object) => void,
 *          codec?: {down?: 'opus'|'pcm16', up?: 'opus'|'pcm16'}}} [opts]
 */

export function createLinkMonitor({
  now = monoNow, thresholds = LINK_THRESHOLDS, onChange = () => {}, codec = {},
} = {}) {
  const th = { ...LINK_THRESHOLDS, ...thresholds };
  let downCodec = codec.down === 'pcm16' ? 'pcm16' : 'opus';
  let upCodec = codec.up === 'pcm16' ? 'pcm16' : 'opus';

  // The link state the server should be in.
  let mode = 'audio';
  let down = downCodec === 'opus' ? th.rungs.down[0] : null;
  let up = upCodec === 'opus' ? th.rungs.up[0] : null;
  let lastWhy = null;
  let pinned = null;            // 'text' when the user chose "Always use the phone's voice"

  // Hysteresis.
  let degradedAt = null;        // when the current degraded step began
  let laddered = null;          // when the codec ladder stepped down (grace)
  let dwell = th.dwellMs;
  let recoveredAt = -Infinity;
  let lastBadAt = now();
  let pending = null;           // a predictive change held until say_end
  let lastProbeAt = -Infinity;
  let probeOut = false;
  let audioSince = -Infinity;   // when the mode last went (back) to audio

  // Per connection.
  let rttEwma = null;
  let rtts = [];
  let highPongs = 0;
  let minSkew = null;           // the best skew of the current reply
  let lastSkew = null;
  let chunks = [];              // [{arrival, bytes, durationMs, skew}] (goodput window, this reply)
  let goodputBadSince = null;
  let deadlines = new Map();    // seq → due
  let firstChunk = [];
  let replyActive = false;
  let underruns = 0;
  let backlogS = 0;
  let backlogSlow = false;

  const atTop = () => mode === 'audio' && (down === null || down === th.rungs.down[0]);
  const state = () => ({ mode, downKbps: down, upKbps: up });
  const isDefault = () => mode === 'audio'
    && down === (downCodec === 'opus' ? th.rungs.down[0] : null)
    && up === (upCodec === 'opus' ? th.rungs.up[0] : null);

  const emit = (why) => { lastWhy = why; onChange({ ...state(), why }); };
  const bad = () => { lastBadAt = now(); };

  const enterDegraded = () => {
    const t = now();
    dwell = t - recoveredAt < th.flapWindowMs ? Math.min(th.dwellMaxMs, dwell * 2) : th.dwellMs;
    degradedAt = t;
  };

  /**
   * Switch to text. A new degraded spell (from the top rung, from a rung no spell put us on,
   * or after a recovery step — text → 16 kbps that fails again is a flap) doubles the dwell
   * within the flap window; the ladder → text step inside one spell only restarts it
   * (review T8 r2 M1: a text ↔ 16 kbps flap never doubled).
   */
  const toText = (next, why) => {
    if (mode === 'text') return;
    if (atTop() || degradedAt === null || recoveredAt > degradedAt) enterDegraded(); else degradedAt = now();
    commit(next, why);
  };

  /** Apply a held or immediate change. */
  const commit = (next, why) => {
    if (next.mode === 'audio' && mode !== 'audio') audioSince = now();
    ({ mode, downKbps: down, upKbps: up } = { mode: next.mode, downKbps: next.downKbps, upKbps: next.upKbps });
    emit(why);
  };

  /** A new reply (or the end of one): the skew baseline and the goodput window start over. */
  const resetReply = () => {
    minSkew = null; lastSkew = null; chunks = []; goodputBadSince = null;
  };
  const trimWindow = (t) => {
    const cut = t - th.goodputWindowMs;
    while (chunks.length && chunks[0].arrival < cut) chunks.shift();
  };

  const degrade = (why) => {
    bad();
    if (pinned) return;
    if (mode === 'text') return;
    const t = now();
    const reactive = REACTIVE.has(why);
    const severe = why === 'deadline' || why === 'stall';
    // First step: the codec ladder (no voice change → never held).
    if (!severe && downCodec === 'opus' && atTop() && !pending) {
      enterDegraded();
      laddered = t;
      commit({ mode: 'audio', downKbps: th.rungs.down[1], upKbps: upCodec === 'opus' ? th.rungs.up[1] : null }, why);
      return;
    }
    if (!severe && laddered !== null && t - laddered < th.ladderGraceMs) return; // the rung has not applied yet
    const next = { mode: 'text', downKbps: down, upKbps: up };
    if (!reactive && replyActive) { pending = { next, why }; return; }
    pending = null;
    toText(next, why);
  };

  const linkBehind = () => minSkew !== null && lastSkew !== null && lastSkew - minSkew >= th.skewGrowMs;
  /** The skew is still climbing inside the goodput window (not a settled step). */
  const skewRising = () => {
    let lo = Infinity;
    for (const c of chunks) if (Number.isFinite(c.skew) && c.skew < lo) lo = c.skew;
    return lastSkew !== null && Number.isFinite(lo) && lastSkew - lo >= th.skewRiseMs;
  };

  const neededFor = (target) => {
    if (target.mode === 'text') return 1;
    if (downCodec !== 'opus') return th.neededKbps.pcm16;
    return target.downKbps === th.rungs.down[0] ? th.neededKbps.opusTop : th.neededKbps.opusLow;
  };

  /** The next step up from here, or null when already at the top. */
  const stepUp = () => {
    if (mode === 'text') {
      return downCodec === 'opus'
        ? { mode: 'audio', downKbps: th.rungs.down[1], upKbps: upCodec === 'opus' ? th.rungs.up[1] : null }
        : { mode: 'audio', downKbps: null, upKbps: upCodec === 'opus' ? th.rungs.up[0] : null };
    }
    if (!atTop()) return { mode: 'audio', downKbps: th.rungs.down[0], upKbps: upCodec === 'opus' ? th.rungs.up[0] : null };
    return null;
  };

  const recoverable = () => {
    if (pinned || isDefault() || replyActive || pending) return false;
    const t = now();
    if (rttEwma === null || rttEwma >= th.rttHealthyMs) return false;
    if (degradedAt !== null && t - degradedAt < dwell) return false;
    return t - lastBadAt >= th.healthyMs;
  };

  /** The reply is over (a `say_end`, or a cut with none): a held predictive switch goes now. */
  const replyOver = () => {
    replyActive = false;
    underruns = 0;
    // A sentence that never got audio (tts_failed, no_speech_in_reply) is not a link fault.
    deadlines = new Map();
    resetReply();
    if (pending) {
      const p = pending;
      pending = null;
      toText(p.next, p.why);
    }
  };

  const api = {
    get thresholds() { return th; },
    // ---- inputs
    onPong(rttMs, rxOff, sentOff) {
      const rtt = Math.max(0, Number(rttMs) || 0);
      rttEwma = rttEwma === null ? rtt : th.rttAlpha * rtt + (1 - th.rttAlpha) * rttEwma;
      rtts.push(rtt);
      if (rtts.length > th.rttWindow) rtts.shift();
      if (rttEwma >= th.rttHealthyMs) bad();
      if (rtt > th.stallMs) degrade('stall');
      highPongs = rttEwma > th.rttDegradedMs ? highPongs + 1 : 0;
      if (highPongs >= th.rttDegradedPongs) { highPongs = 0; degrade('rtt'); }
      if (typeof rxOff === 'number' && typeof sentOff === 'number') {
        backlogS = Math.max(0, sdiff(sentOff >>> 0, rxOff >>> 0) / 16000 - rtt / 2000);
        if (!backlogSlow && backlogS > th.backlogSlowS) backlogSlow = true;
        else if (backlogSlow && backlogS < th.backlogClearS) backlogSlow = false;
        onChange({ backlogS: Math.round(backlogS * 100) / 100, slow: backlogSlow });
      }
    },
    onSay(seq, _tServer, kind = 'reply') {
      if (kind === 'reply' && !replyActive) { replyActive = true; underruns = 0; resetReply(); }
      // Deadlines are for reply sentences only: an ack/tier line is one pre-rendered burst and a
      // notice ("Still there?") has no `say_end` — a missed one is a TTS fault, never worth a
      // mode switch (review T8 r2 L4).
      if (mode !== 'audio' || kind !== 'reply') return;
      // Just back in audio: the server may still be finishing a text-mode sentence.
      const t = now();
      if (t - audioSince < Math.max(th.audioGraceMinMs, 2 * (rttEwma || 0))) return;
      const pr = p90(firstChunk);
      const ms = Math.max(th.firstAudioMinMs, (pr === null ? th.firstChunkPriorMs : pr) + th.firstAudioPadMs);
      deadlines.set(seq >>> 0, t + ms);
    },
    onSayEnd() { replyOver(); },
    /** The reply ended without a `say_end` (barge-in, tap-stop, undo, failed turn, new turn). */
    onReplyCut() { replyOver(); },
    onChunk(seq, arrival, tServer, durationMs = 0, _leadMs = 0, bytes = 0) {
      deadlines.delete(seq >>> 0);
      const a = Number(arrival);
      const skew = a - Number(tServer);
      if (Number.isFinite(skew)) {
        lastSkew = skew;
        if (minSkew === null || skew < minSkew) minSkew = skew;
      }
      chunks.push({ arrival: a, bytes: Number(bytes) || 0, durationMs: Number(durationMs) || 0, skew });
      trimWindow(a);
    },
    /** `turn_error`: no audio is coming for `seq` (or, without a seq, for any armed one). */
    onTurnError(seq) {
      if (seq === undefined || seq === null) deadlines = new Map();
      else deadlines.delete(seq >>> 0);
    },
    onUnderrun(_seq, gapMs) {
      if (!linkBehind()) return;      // the TTS left the hole, not the network
      bad();
      underruns += 1;
      if (underruns >= th.underrunsPerReply || Number(gapMs) > th.bigGapMs) degrade('underrun');
    },
    onStall() { degrade('stall'); },
    onReady() {
      // A new socket: per-connection measures start over; the mode stands (resend it).
      minSkew = null; lastSkew = null; chunks = []; goodputBadSince = null;
      deadlines = new Map(); highPongs = 0; probeOut = false;
    },
    onClose() {
      deadlines = new Map();
      replyActive = false;
      if (pending) { const p = pending; pending = null; toText(p.next, p.why); }
    },
    onMetrics(m = {}) {
      const v = Number(m.first_chunk_ms);
      if (Number.isFinite(v) && v > 0) { firstChunk.push(v); if (firstChunk.length > 20) firstChunk.shift(); }
    },
    /** Probe result. Returns the measured kbps. Recovers one step when it passes. */
    onProbe(bytes, elapsedMs) {
      const answered = probeOut;           // false: a late answer to a probe already written off
      probeOut = false;
      const transfer = Math.max(1, (Number(elapsedMs) || 0) - (rttEwma || 0));
      const kbps = (Number(bytes) || 0) * 8 / transfer;
      const next = stepUp();
      if (answered && next && recoverable() && kbps >= th.probeFactor * neededFor(next)) {
        recoveredAt = now();
        // The dwell gates leaving the degraded spell, not each rung of the climb: an upward
        // step keeps degradedAt, so the next probe (≥ 20 s later) can finish it (review L2).
        if (next.mode === 'audio' && next.downKbps === th.rungs.down[0]) { degradedAt = null; laddered = null; }
        commit(next, 'recovered');
      }
      return kbps;
    },
    tick() {
      const t = now();
      for (const [seq, due] of deadlines) {
        if (t >= due) { deadlines.delete(seq); if (mode === 'audio') degrade('deadline'); }
      }
      // Goodput while streaming: audio (or bytes) per wall time over the last 2 s of THIS
      // reply, only when the link is behind and still falling further behind.
      trimWindow(t);
      if (probeOut && t - lastProbeAt >= th.probeTimeoutMs) probeOut = false;
      if (replyActive && chunks.length >= 2 && mode === 'audio') {
        const span = Math.max(1, t - chunks[0].arrival);
        const bytes = chunks.reduce((s, c) => s + c.bytes, 0);
        const audio = chunks.reduce((s, c) => s + c.durationMs, 0);
        const needed = neededFor(state());
        const ratio = bytes > 0 ? (bytes * 8 / span) / needed : audio / span;
        if (ratio < th.goodputFactor && linkBehind() && skewRising()) {
          if (goodputBadSince === null) goodputBadSince = t;
          bad();
          if (t - goodputBadSince >= th.goodputHoldMs) { goodputBadSince = null; degrade('goodput'); }
        } else goodputBadSince = null;
      } else goodputBadSince = null;
    },
    // ---- control
    /** The starting mode of a voice chat from navigator.connection.effectiveType. */
    startMode(effectiveType, { bluetooth = false } = {}) {
      if (th.slowTypes.includes(String(effectiveType || ''))) {
        mode = 'text';
        degradedAt = now() - th.dwellMs; // start degraded, probe up after a healthy spell
        lastWhy = 'rtt';
      }
      if (bluetooth && downCodec === 'opus') down = th.rungs.down[1]; // SCO: start at the 16 kbps rung
      return state();
    },
    /** True when a probe should go now (≤ 1 per 20 s, only when recovery is otherwise due). */
    wantProbe() {
      const t = now();
      if (probeOut && t - lastProbeAt >= th.probeTimeoutMs) probeOut = false; // never answered
      if (probeOut || t - lastProbeAt < th.probeEveryMs || !stepUp() || !recoverable()) return false;
      lastProbeAt = t;
      probeOut = true;
      return true;
    },
    /** The probe was refused (`error {rate_limited}`) or lost: the next one may go after 20 s. */
    probeFailed() { probeOut = false; },
    /** "Always use the phone's voice": pin text (why user); null unpins. */
    setPinned(m) {
      pinned = m === 'text' ? 'text' : null;
      if (pinned && mode !== 'text') { pending = null; mode = 'text'; emit('user'); }
      else if (!pinned && mode === 'text' && lastWhy === 'user') {
        mode = 'audio'; down = downCodec === 'opus' ? th.rungs.down[0] : null; up = upCodec === 'opus' ? th.rungs.up[0] : null;
        degradedAt = null; laddered = null; audioSince = now(); emit('user');
      }
    },
    /** The codecs the socket negotiated (`ready.codec_in/out`) or a decoder/encoder fallback. */
    setCodec({ down: d, up: u } = {}) {
      if (d) { downCodec = d === 'opus' ? 'opus' : 'pcm16'; if (downCodec !== 'opus') down = null; else if (down === null) down = th.rungs.down[0]; }
      if (u) { upCodec = u === 'opus' ? 'opus' : 'pcm16'; if (upCodec !== 'opus') up = null; else if (up === null) up = th.rungs.up[0]; }
    },
    /** The `link` frame to re-send after a `ready` (null when the defaults stand). */
    resendFrame() {
      if (isDefault()) return null;
      const f = { type: 'link', mode, why: lastWhy || 'user' };
      if (down !== null) f.down_kbps = down;
      if (up !== null) f.up_kbps = up;
      return f;
    },
    current() { return state(); },
    get rttEwma() { return rttEwma; },
    get deadAfterMs() { return heartbeatDeadMs(rttEwma, th); },
    /** For the Voice lab line. */
    stats() {
      return {
        mode, down_kbps: down, up_kbps: up, rtt_ms: rttEwma === null ? null : Math.round(rttEwma),
        rtt_p90_ms: p90(rtts), backlog_s: Math.round(backlogS * 100) / 100, underruns,
        skew_ms: minSkew === null || lastSkew === null ? null : Math.round(lastSkew - minSkew),
        dwell_ms: dwell, pending: pending ? pending.why : null, pinned: !!pinned,
      };
    },
  };
  return api;
}

/** A `link` frame (C.3) from an onChange link event. */
export function linkFrame(e) {
  const f = { type: 'link', mode: e.mode, why: e.why };
  if (e.downKbps != null) f.down_kbps = e.downKbps;
  if (e.upKbps != null) f.up_kbps = e.upKbps;
  return f;
}
