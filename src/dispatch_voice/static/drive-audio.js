// DisPatch Voice: drive-audio.js — the voice chat's audio and link engine.
// Owner: track T7 (part 2). Contracts: C-§C (the client half of the wire:
// every frame it sends, `cursorFrom`, deferred predictive `link`, the
// post-`barge_in` `heard`), C-§E.2 (the T8 modules it drives, all reached
// through bridge.js), C.5 invariants 1, 4, 9–12, amendments T6-1/T6-2,
// T8-1..T8-3, T7a-7. SPEC §4.6 (duck/resume/barge-in), §5.2–§5.6.
//
// No DOM. drive.js owns the screen and the socket's life; this engine owns
// what moves through it:
//
//   mic worklet ─► uplink (Opus | PCM, wire3) ─► socket        (v3)
//              └─► uplink ring (30 s) ─► replay after a reconnect
//   socket ─► downlink (wire3; WebCodecs Opus | PCM) ─► claims ─► player worklet
//   `say` ─► claims ─► server audio | phone voice | caption only
//   pongs, chunks, underruns, says ─► link monitor ─► `link` frames + the notice
//
// A v2 server (no `ready.proto`) gets today's wire: raw PCM16 up, JSON
// `audio` header + PCM down, `playback_done` on every drain; nothing v3-only
// (heard, link, probe, duck_local) is ever sent to it.
//
// Every browser object (worklet port, WebCodecs, speechSynthesis) and every
// clock is injected, so tests/js/drive-audio.test.js runs it under node with
// the real T8 modules and T8's fakes.

const Q = new URL(import.meta.url).search;
const B = await import(`./bridge.js${Q}`);

/** u16 serial-number compare for `gen` (C.4: the client compares gen & 0xFFFF). */
export const genNewer = (a, b) => a !== b && (((a - b) & 0xffff) < 0x8000);
const sdiff = (a, b) => ((a - b) | 0);

/** Mic audio captured before the first `ready` (20 ms frames → 10 s). */
export const PRE_READY_FRAMES = 500;
/** How long a resuming socket waits for the server's `cursor {from}` before replaying from its own. */
export const CURSOR_WAIT_MS = 3000;
/** `heard` goes out alone this often while a reply plays (C.3). */
export const HEARD_EVERY_MS = 1000;
/** After a reconnect the player keeps the old session's audio at most this long. */
export const OLD_AUDIO_MAX_MS = 15000;
/**
 * …and the phone voice the old session's queue at most this long. Each phone
 * sentence has its own watchdog (local-voice.js), so the queue always drains;
 * this only bounds a long one.
 */
export const OLD_PHONE_MAX_MS = 60000;
/** The PCM16 uplink ring holds 10 s (SPEC §5.5: 30 s of Opus ≈ 59 KB, 10 s of PCM ≈ 320 KB). */
export const RING_MAX_BYTES = 320000;
/** "Wait" (prebuffer the whole sentence): the player worklet's prebuffer cap. */
export const WAIT_PREBUFFER_MS = 10000;
/** The duck gain (SPEC §4.6 layer 1: ramp to 0.15 over 40 ms). */
export const DUCK_GAIN = 0.15;
export const DUCK_RAMP_MS = 40;
/**
 * Half-duplex on the device: while the phone voice speaks (speechSynthesis has no
 * echo-cancellation reference, so its own speaker reaches the mic), and for this long
 * after it stops (the room's tail), every mic frame is zeroed before it leaves
 * `micMessage`. Zeroed, not dropped: the uplink offsets stay contiguous.
 */
export const PHONE_VOICE_MUTE_TAIL_MS = 400;

/** Which voice a sentence goes to when the link is degraded (§5.4): 'phone' | 'text'. */
export function localPath({ slowLink = 'phone', phoneOk = false } = {}) {
  if (slowLink === 'text') return 'text';
  return phoneOk ? 'phone' : 'text';
}

/**
 * The `link` frame to put on the wire for a monitor event, given the device's
 * "When the connection is slow" choice. `wait` never asks for text: it stays
 * in the server's voice at the lowest rung and prebuffers whole sentences.
 * Returns {frame, wait, text}.
 */
export function wireLink(e, slowLink, th = B.LINK_THRESHOLDS) {
  const frame = B.linkFrame ? B.linkFrame(e) : { type: 'link', mode: e.mode, why: e.why };
  if (e.mode === 'text' && slowLink === 'wait') {
    const low = { type: 'link', mode: 'audio', why: e.why };
    const rungs = (th && th.rungs) || { down: [24, 16], up: [16, 12] };
    // Always the lowest downlink rung: a `text` event (a missed deadline, an
    // underrun) carries no rate of its own, and Wait's whole point is to ask
    // for the cheapest audio the server can send.
    low.down_kbps = rungs.down[rungs.down.length - 1];
    if (e.upKbps != null || frame.up_kbps != null) low.up_kbps = rungs.up[rungs.up.length - 1];
    return { frame: low, wait: true, text: false };
  }
  return { frame, wait: false, text: e.mode === 'text' };
}

/**
 * `start.after_partial.sentences` (T6-2): how many leading sentences of reply
 * `mid` the listener WILL hear in full — the ones whose audio arrived complete
 * (`audio_end`, not "played": the player keeps playing across a reconnect),
 * that the phone holds (it keeps speaking across a reconnect, review-T7b M2),
 * or that were shown as text — counting on from
 * `base` (what an earlier resume already skipped). A cut sentence stops the
 * count: it comes again from its start.
 */
export function completeSentences(records, mid, base = 0) {
  // `mid` may be a set of ids: a reply announced under its provisional `run:…`
  // id goes on under its row id once `stream_done` names it (T3s2-1).
  const has = mid instanceof Set ? (x) => mid.has(x) : (x) => x === mid;
  let n = base;
  for (const r of records) {
    if (r.kind !== 'reply' || !has(r.mid)) continue;
    if ((r.complete || r.path === 'phone') && !r.cut && r.path) n += 1;
    else break;
  }
  return n;
}

/**
 * A stand-in for the player worklet when AudioWorklet is missing (very old
 * browsers): the v2 gap-free buffer scheduler behind the player client's
 * interface. No pause (the duck only lowers the volume), no rewind, `played`
 * once per sentence (so captions still appear), `idle` when the queue drains.
 */
export function createBufferPlayerClient(ac, out, { onPlayed, onIdle } = {}) {
  let nextAt = 0;
  let minGen = null;
  let last = null;
  const live = new Set();
  const started = new Set();
  const stopAll = () => {
    for (const s of live) { try { s.onended = null; s.stop(); } catch { /* ended */ } }
    live.clear();
    nextAt = 0;
  };
  return {
    push({ gen = 0, seq = 0, kind = 'reply', rate = 24000, pcm }) {
      const g = gen & 0xffff;
      if (minGen !== null && genNewer(minGen, g)) return;
      if (minGen === null || genNewer(g, minGen)) { if (minGen !== null) stopAll(); minGen = g; }
      const f = pcm instanceof Float32Array ? pcm : Float32Array.from(pcm, (v) => v / 32768);
      if (!f.length) return;
      const buf = ac.createBuffer(1, f.length, rate);
      buf.getChannelData(0).set(f);
      const src = ac.createBufferSource();
      src.buffer = buf;
      src.connect(out || ac.destination);
      const at = Math.max(ac.currentTime + 0.02, nextAt);
      nextAt = at + f.length / rate;
      live.add(src);
      src.onended = () => {
        live.delete(src);
        if (!live.size && onIdle) onIdle({ type: 'idle', gen: g, seq, ended: true });
      };
      src.start(at);
      const k = `${g}:${seq}`;
      if (!started.has(k)) {
        started.add(k);
        if (started.size > 256) started.delete(started.values().next().value);
        const rep = { type: 'played', gen: g, seq, ms: 1, leadMs: 0, kind };
        last = rep;
        if (onPlayed) setTimeout(() => { if (minGen === g) onPlayed(rep); }, Math.max(0, (at - ac.currentTime) * 1000));
      }
    },
    pause() {}, resume() {}, gain() {}, prebuffer() {}, end() {},
    flush(gen) { minGen = gen & 0xffff; last = null; stopAll(); },
    reset() { minGen = null; last = null; stopAll(); },
    get paused() { return false; },
    get last() { return last; },
  };
}

/**
 * The engine. Options (all injectable):
 *   send(obj)                 a JSON frame on the open socket (dropped when closed)
 *   sendBinary(buf)           a binary frame on the open socket
 *   isOpen() -> bool          the socket is open
 *   micPost(msg)              a message to the mic worklet (drive-mic)
 *   setDuckGain(value, ms)    ramp the main-thread duck gain node
 *   settings() -> blob        the device settings (drive-settings.js F.3 blob)
 *   lang() -> 'en'            the UI language (phone voice choice)
 *   android, bluetooth        device facts (phone-voice picker semantics, echo bias, rung)
 *   effectiveType             navigator.connection.effectiveType (the starting mode)
 *   synth, Utterance          speechSynthesis (null: no phone voice)
 *   codecs                    WebCodecs classes (tests)
 *   latency() -> {outputLatency, baseLatency}   the AudioContext's, for `heard`
 *   stream                    this voice chat's stream id (the ring key, C.1 `s`)
 *   now, setTimer, clearTimer one monotonic clock for everything (T8-3)
 *   ui                        callbacks: onSay, onSentence, onPlaying, onLevel, onDuck,
 *                             onLink, onCutOff, onSpeech, onMicLevel, onStats, onPhoneFailed
 */
export function createDriveAudio({
  send = () => {}, sendBinary = () => {}, isOpen = () => false,
  micPost = () => {}, setDuckGain = () => {},
  settings = () => ({}), lang = () => 'en', android = false, bluetooth = false, effectiveType = '',
  synth = globalThis.speechSynthesis || null, Utterance = globalThis.SpeechSynthesisUtterance,
  codecs, latency = () => ({}), stream = '',
  now = B.monoNow, setTimer = (f, ms) => setTimeout(f, ms), clearTimer = (h) => clearTimeout(h),
  ui = {},
} = {}) {
  const th = B.LINK_THRESHOLDS || { rungs: { down: [24, 16], up: [16, 12] }, probeBytes: 16384 };
  const cb = (name, ...a) => { const f = ui[name]; if (typeof f === 'function') { try { f(...a); } catch { /* UI must not break audio */ } } };
  const cfg = () => settings() || {};

  // ---- per voice chat --------------------------------------------------------
  let disposed = false;
  let proto = 2;               // this socket's protocol (from `ready`)
  let sayCap = false;          // the server sends `say`/`say_end` and reads `heard` (ready.say, or v3)
  let sessions = 0;            // a per-`ready` id for servers that send no sid
  let everReady = false;
  let everV3 = false;
  let sid = '';                // this socket's session id (`ready.sid`)
  let open = false;            // socket open
  let ready = false;           // `ready` seen on this socket
  let live = false;            // v3: uplink messages go out as they are made
  let resuming = false;        // this socket resumes an earlier v3 session
  let socketExtra = {};        // start.cursor / start.after_partial for this socket
  let curGen = 0;              // the newest gen seen
  const preReady = [];         // [{off, buf}] before the first `ready`
  let micClock = null;         // {off, at}: the mic's 16 kHz clock at a wall instant
  let lastRxOff = null;
  let lastTranscriptUpto = null;
  let lastEotUpto = null;
  let sentEnd = 0;             // the uplink offset just past the last live message
  let codecIn = 'pcm16';
  let codecOut = 'pcm16';
  let upKbps = 16;             // the server's starting uplink rung (ready.opus_up_kbps)
  let upFailed = false;
  let downFailed = false;
  let support = { up: false, down: false };

  // sentences: `${sid}:${seq}` → {sid, gen, seq, kind, text, role, mid, intent, path, complete, cut, shown, ms}
  const says = new Map();
  const order = [];            // records in arrival order (after_partial counts them)
  const keyOf = (s, seq) => `${s}:${seq >>> 0}`;
  // The reply in flight: {mid (newest id), ids (every id it went by: the provisional
  // `run:…` and then the row id, T3s2-1), sid, gen, lastSeq|null, ended}. `ended`
  // means "nothing of it is left to resume": said in full (`say_end`), cut
  // (`barge_in`, a tap-stop), failed (`turn_error`), or followed by a new turn.
  // `turn` is the user row the reply answers (`transcript.message_id`): a reconnect
  // that resumes the reply names it as `after` (review-T7b H1). `abandoned` = cut,
  // failed or followed by a new turn; `ended` alone (say_end) = the server said it all.
  let reply = null;
  let lastTurnRow = null;      // the newest `transcript.message_id` (a user row)
  const partialBase = new Map();  // id → {sid, base, n}: what earlier resumes already skipped
  let resumeFrom = null;       // {ids, entry, sid}: the reply this socket's after_partial named
  const isProvisional = (id) => typeof id === 'string' && id.startsWith('run:');
  let resumeAnchor = null;     // {after, skip}: this socket resumes a cut reply (startExtra)
  function endReply() { if (reply) { reply.ended = true; reply.abandoned = true; } resumeFrom = null; resumeAnchor = null; }
  /** Nothing of the reply is left to hear after a drop: abandoned, or said in full and all here. */
  function replySettled(rp) {
    if (!rp || rp.abandoned || !rp.mid) return true;
    if (!rp.ended) return false;
    // say_end came: every one of its sentences must be complete (audio all here, the
    // phone has it, or it was captioned) — the last one may have been cut by the drop.
    for (const r of order) {
      if (r.sid !== rp.sid || r.kind !== 'reply' || !rp.ids.has(r.mid)) continue;
      if (!(r.complete || r.path === 'phone') || r.cut || !r.path) return false;
    }
    return true;
  }
  let lastSayKey = null;       // the previous sentence on this socket (player `end` on the next say)

  // ---- the player (worklet client, or the buffer fallback) --------------------
  let player = null;
  let playerSid = '';          // whose sentences the player holds
  let playerBusy = false;      // something pushed since the last `idle`
  let held = [];               // player messages waiting for the player / the deferred reset
  let resetPending = false;
  let resetTimer = null;
  let playing = null;          // {gen, seq, kind} of the sentence the player is playing
  // The sentence the player holds ran dry mid-way (`idle {ended: false}`) and waits for
  // more audio: `played` keeps repeating its position, but nothing is audible. The screen
  // must not show a playing reply then (a drop mid-reply read "Waiting for signal" over
  // "Tap to interrupt"). `playing` itself stays (what was heard is still reported).
  let starved = null;          // {gen, seq, ms|null}
  let replyActive = false;
  let lastHeardAt = -Infinity;
  let stopHeard = null;        // {h, at}: what a tap-stop reported, for the barge_in that follows
  let doResetFn = null;

  // ---- phone voice state (declared before the link monitor can route to it) -------
  let localVoice = null;
  let phoneVoice = null;
  let phoneFailed = false;
  let voices = [];
  const phoneRecs = new Map();         // `${gen}:${seq}` → record
  let phoneShown = null;
  let wireMode = 'audio';              // what the server was last asked for
  // After a reconnect the phone goes on speaking what the old session gave it
  // (after_partial counted it, review-T7b M2); the new session's sentences wait
  // behind it, as the player's do (T6-2). [['say', record] | ['end', {gen, last_seq}]]
  let phoneResetPending = false;
  let phoneHeld = [];
  let phoneResetTimer = null;

  const playerCallbacks = {
    onPlayed: (d) => onPlayed(d),
    onUnderrun: (d) => {
      if (starved && starved.gen === d.gen && starved.seq === d.seq) { starved = null; cb('onPlaying', playing && playing.kind); }
      if (monitor && proto >= 3) monitor.onUnderrun(d.seq, d.gapMs);
    },
    onIdle: (d) => onIdle(d),
    onLevel: (v) => cb('onLevel', v),
  };

  function playerDo(op, ...a) {
    if (!player || resetPending) { held.push([op, a]); return; }
    try { player[op](...a); } catch { /* a closed context */ }
  }
  function flushHeld() {
    const q = held;
    held = [];
    for (const [op, a] of q) playerDo(op, ...a);
  }
  /** A new session: the player forgets gens — once what the old session left has played. */
  function resetPlayer({ now: immediate = false } = {}) {
    clearTimer(resetTimer); resetTimer = null;
    const doReset = () => {
      clearTimer(resetTimer); resetTimer = null;
      doResetFn = null;
      resetPending = false;
      playerBusy = false;
      playing = null;
      if (player) { try { player.reset(); } catch { /* gone */ } }
      playerSid = sid;
      flushHeld();
    };
    if (immediate || !player || !playerBusy) { doReset(); return; }
    // T6-2: the sentences that arrived complete keep playing across a reconnect
    // (after_partial skipped them); the new session's audio waits behind them.
    resetPending = true;
    doResetFn = doReset;
    resetTimer = setTimer(doReset, OLD_AUDIO_MAX_MS);
  }

  // ---- T8 modules ---------------------------------------------------------------
  const claims = B.createSeqClaims();
  // 30 s of Opus fits well inside the byte cap; PCM16 (640 B per 20 ms) stops at
  // 10 s (SPEC §5.5, review-T7b L1). The codec can change mid-chat, so one ring
  // with both bounds.
  const ring = B.createUplinkRing ? B.createUplinkRing({ seconds: 30, maxBytes: RING_MAX_BYTES, stream }) : null;
  const pump = B.createReplayPump && ring
    ? B.createReplayPump({ send: (buf) => sendBinary(buf), setTimer, clearTimer, onDone: () => onReplayDone() })
    : null;
  let replayTo = null;
  let cursorTimer = null;
  let uplink = null;
  let uplinkCodec = null;

  const bias = B.createEchoBias ? B.createEchoBias({ bluetooth }) : null;
  let localSpeaking = false;
  let phoneMutedUntil = 0;     // the mic stays zeroed this long after the phone voice stops
  /** True while the phone voice speaks and for PHONE_VOICE_MUTE_TAIL_MS after (half-duplex). */
  const micHalfDuplex = () => localSpeaking || now() < phoneMutedUntil;

  const duck = B.createOptimisticDuck ? B.createOptimisticDuck({
    apply: (on, info = {}) => {
      if (on) {
        setDuckGain(DUCK_GAIN, DUCK_RAMP_MS);
        playerDo('pause');
      } else {
        setDuckGain(1, DUCK_RAMP_MS);
        // After a flush the worklet is already unpaused (resume is then a no-op); after a
        // socket drop (`reset`) the old audio must not stay paused behind the new session.
        playerDo('resume', !!info.rewind && !info.flushed);
      }
      cb('onDuck', !!on);
    },
    notify: (on) => { if (proto >= 3) send({ type: 'duck_local', on: !!on }); },
    rtt: () => (monitor && monitor.rttEwma) || 0,
    now, setTimer, clearTimer,
    enabled: () => proto >= 3 && !localSpeaking && cfg().optimisticDuck !== false && cfg().bargeIn !== false,
  }) : null;

  let monitor = null;
  let textMode = false;        // the server is asked for `say` only (phone voice / text)
  let waitHold = false;        // "Wait": prebuffering whole sentences in the server voice
  let pinned = false;          // "Always use the phone's voice"
  let probeSentAt = null;
  if (B.createLinkMonitor) {
    monitor = B.createLinkMonitor({ now, onChange: (e) => onLinkEvent(e), codec: { down: 'opus', up: 'opus' } });
    const st = monitor.startMode(effectiveType, { bluetooth });
    if (st && st.mode === 'text') applyLinkMode('text', st);
  }

  const downlink = B.createDownlink ? B.createDownlink({
    player: { push: (m) => serverPush(m) },
    onChunk: (c) => {
      if (monitor && c.kind !== 'probe') {
        monitor.onChunk(c.seq, c.arrival, c.t, c.durationMs, (player && player.last && player.last.leadMs) || 0, c.bytes);
      }
    },
    onProbe: (p) => {
      if (monitor && probeSentAt !== null) monitor.onProbe(p.bytes, Math.max(1, p.arrival - probeSentAt));
      probeSentAt = null;
    },
    onError: (kind) => {
      if (kind !== 'decoder' || downFailed) return;
      downFailed = true;
      if (monitor) monitor.setCodec({ down: 'pcm16' });
      // The server switches to PCM16 from the next sentence (LinkState pcm_fallback);
      // the next socket negotiates down=pcm16 from the start.
      if (proto >= 3) send({ type: 'link', mode: textMode ? 'text' : 'audio', down_kbps: null, why: 'decoder' });
    },
    codecs, now, setTimer, clearTimer,
  }) : null;
  const pairer = { pending: null };   // v2: JSON `audio` header + the next binary frame

  // ---- phone voice ----------------------------------------------------------------
  function phoneOk() {
    return !!(localVoice && !phoneFailed && phoneVoice);
  }
  function pickPhoneVoice() {
    const s = cfg();
    phoneVoice = B.pickVoice(voices, lang(), s.phoneVoice || '', { android }) || null;
    if (!phoneVoice || !synth || !B.createLocalVoice || typeof Utterance !== 'function') {
      if (localVoice && !phoneVoice) { localVoice.cancel(); }
      return;
    }
    if (localVoice) { localVoice.setVoice(phoneVoice); return; }
    try {
      localVoice = B.createLocalVoice({
        synth, Utterance, voice: phoneVoice, lang: lang(), now, setTimer, clearTimer,
        onDone: (d) => onPhoneDone(d),
        onFail: () => onPhoneFail(),
        onSpeaking: (on) => onLocalSpeaking(on),
      });
    } catch { localVoice = null; }
  }

  function onLocalSpeaking(on) {
    localSpeaking = !!on;
    if (!localSpeaking) phoneMutedUntil = now() + PHONE_VOICE_MUTE_TAIL_MS;
    micPost({ localSpeaking: localSpeaking });
    if (bias) micPost({ echoBiasDb: bias.value });
    if (on) showPhoneCaption();
    cb('onPlaying', on ? 'reply' : null);
  }
  function showPhoneCaption() {
    if (!localVoice) return;
    const c = localVoice.current();
    if (!c) return;
    const k = `${c.gen}:${c.seq}`;
    if (phoneShown === k) return;
    phoneShown = k;
    const r = phoneRecs.get(k);
    if (r) {
      show(r);
      if (r.kind === 'reply') {
        cb('onNow', r.key);                    // the sentence being read is highlighted (§7.4)
        cb('onActivity');
        if (!replyActive) replyActive = true;
      }
    }
  }
  function onPhoneDone(d) {
    const r = phoneRecs.get(`${d.gen}:${d.seq}`);
    if (r && d.spoken) r.complete = true;
    if (phoneResetPending) {
      // The old session's sentence: never reported to the new session. Once its
      // queue has drained, the new session's sentences may start.
      if (!d.final && localVoice && !localVoice.pending) {
        setTimer(() => { if (phoneResetPending && localVoice && !localVoice.pending) doPhoneReset(); }, 0);
      }
      if (d.final) cb('onPlaying', null);
      return;
    }
    if (d.final && sayCap && ready && (!r || r.sid === sid)) send({ type: 'playback_done', gen: d.gen, seq: d.seq });
    if (d.final) { replyActive = false; cb('onPlaying', null); }
  }
  /** A new session: the phone forgets gens — once what the old session gave it is spoken. */
  function resetPhone({ now: immediate = false } = {}) {
    if (phoneResetPending) {
      // A session that came and went while the phone was still busy: what it gave
      // the phone was counted as heard (after_partial), so it is shown as text.
      for (const [op, r] of phoneHeld) if (op === 'say') { r.path = 'text'; r.complete = true; show(r); }
      phoneHeld = [];
    }
    if (immediate || !localVoice || !localVoice.pending) { doPhoneReset(); return; }
    phoneResetPending = true;
    clearTimer(phoneResetTimer);
    phoneResetTimer = setTimer(() => doPhoneReset(), OLD_PHONE_MAX_MS);
  }
  function doPhoneReset() {
    clearTimer(phoneResetTimer); phoneResetTimer = null;
    phoneResetPending = false;
    if (localVoice) localVoice.reset();
    phoneRecs.clear();
    phoneShown = null;
    const q = phoneHeld;
    phoneHeld = [];
    for (const [op, x] of q) {
      if (op === 'say') phoneEnqueue(x);
      else if (op === 'end' && localVoice) localVoice.sayEnd(x);
    }
  }
  /** Drop the new session's held phone sentences that `pred` names (a stop, a barge-in); then reset now. */
  function dropHeldPhone(pred) {
    phoneHeld = phoneHeld.filter(([op, r]) => op !== 'say' || !pred(r) || ((r.cut = true), false));
    doPhoneReset();
  }
  /** Hand one sentence to the phone (it is claimed for the phone already). */
  function phoneEnqueue(r) {
    r.path = 'phone';
    if (phoneResetPending) { phoneHeld.push(['say', r]); return; }
    phoneRecs.set(`${r.gen}:${r.seq}`, r);
    if (phoneRecs.size > 256) phoneRecs.delete(phoneRecs.keys().next().value);
    if (localVoice && localVoice.enqueue({ gen: r.gen, seq: r.seq, text: r.text, kind: r.kind, message_id: r.mid })) return;
    // The phone refused it (gave up, or an older gen): the caption carries it.
    r.path = 'text'; r.complete = true; show(r);
    if (r.kind === 'notice' && sayCap && ready && r.sid === sid) send({ type: 'playback_done', gen: r.gen, seq: r.seq });
  }
  function onPhoneFail() {
    phoneFailed = true;
    // Two failures: text for the rest of this voice chat (§5.4); what the phone
    // still held is shown as text.
    const moved = [];
    if (phoneResetPending) {
      clearTimer(phoneResetTimer); phoneResetTimer = null;
      phoneResetPending = false;
      for (const [op, r] of phoneHeld) if (op === 'say' && r.sid === sid) { r.path = 'text'; r.complete = true; show(r); moved.push(r); }
      phoneHeld = [];
    }
    for (const r of phoneRecs.values()) {
      if (!r.complete && r.sid === sid) { r.path = 'text'; r.complete = true; show(r); moved.push(r); }
    }
    // Nothing will play what moved: the server's `speaking` must still end (C.3).
    // A notice is done at once; the reply when its last sentence moved (a later
    // `say_end` finds that sentence on the text path and reports it then).
    if (sayCap && ready) {
      for (const r of moved) if (r.kind === 'notice') send({ type: 'playback_done', gen: r.gen, seq: r.seq });
      if (reply && reply.ended && reply.sid === sid && reply.lastSeq !== null) {
        const last = says.get(keyOf(sid, reply.lastSeq));
        if (last && moved.includes(last)) send({ type: 'playback_done', gen: last.gen, seq: last.seq });
      }
    }
    replyActive = false;
    cb('onPhoneFailed');
    cb('onLink', linkState());
  }

  // ---- link -------------------------------------------------------------------------
  function linkState() {
    const s = cfg();
    const path = textMode ? localPath({ slowLink: s.slowLink, phoneOk: phoneOk() }) : 'server';
    let notice = null;
    if (waitHold) notice = 'wait';
    else if (textMode && !pinned) notice = path;          // 'phone' | 'text'
    else if (textMode && pinned && path === 'text') notice = 'text';  // pinned but no phone voice
    return { degraded: !!notice, notice, path, mode: textMode ? 'text' : 'audio', pinned };
  }

  function applyLinkMode(mode, e = {}) {
    const s = cfg();
    if (mode === 'text' && s.slowLink === 'wait' && !pinned) {
      if (!waitHold) { waitHold = true; playerDo('prebuffer', WAIT_PREBUFFER_MS); }
      textMode = false;
    } else if (mode === 'text') {
      if (waitHold) { waitHold = false; playerDo('prebuffer', 0); }
      if (!textMode) { textMode = true; handPendingToLocal(); }
    } else {
      textMode = false;
      if (waitHold) { waitHold = false; playerDo('prebuffer', 0); }
    }
    return e;
  }

  function onLinkEvent(e) {
    if (!e) return;
    if ('backlogS' in e) { cb('onStats', api.stats()); return; }
    // The uplink rung is this client's own encoder (LinkState: "nothing to wait for").
    if (uplink && Number.isFinite(e.upKbps)) { try { uplink.setKbps(e.upKbps); } catch { /* closed */ } }
    applyLinkMode(e.mode, e);
    sendWire(e);
    cb('onLink', linkState());
  }
  /** Put a monitor state on the wire, mapped through the slow-link choice. */
  function sendWire(e) {
    const w = wireLink(e, pinned ? 'phone' : cfg().slowLink, th);
    wireMode = w.frame.mode;
    if (proto >= 3 && open && ready) send(w.frame);
  }

  /**
   * Sentences of this session still waiting for their first audio: the phone (or
   * the caption) takes them. Only the trailing run of such sentences counts —
   * anything before a sentence that got audio, the phone or a caption is past
   * (its TTS failed, or its reply was cut) — and only the reply in flight's.
   */
  function handPendingToLocal() {
    const pend = [];
    for (let i = order.length - 1; i >= 0; i--) {
      const r = order[i];
      if (r.sid !== sid || r.path) break;
      pend.push(r);
    }
    for (const r of pend.reverse()) {
      if (r.cut || r.dead) continue;
      if (r.kind === 'reply') {
        if (r.mid && !(reply && reply.sid === sid && reply.ids.has(r.mid))) continue;
        routeLocal(r);
      } else if (r.kind === 'notice') routeLocal(r);
    }
  }

  // ---- sentences --------------------------------------------------------------------
  function record(m, s = sid) {
    const k = keyOf(s, m.seq);
    let r = says.get(k);
    if (r) return { r, fresh: false };
    r = {
      key: k, sid: s, gen: (Number(m.gen) || 0) & 0xffff, seq: m.seq >>> 0, kind: m.kind || 'reply',
      text: String(m.text || ''), role: m.role === 'narrator' ? 'narrator' : 'character',
      mid: m.message_id || null, intent: m.intent || null,
      path: null, complete: false, cut: false, dead: false, shown: false, ms: 0,
    };
    says.set(k, r);
    order.push(r);
    while (order.length > 256) { const old = order.shift(); says.delete(old.key); }
    return { r, fresh: true };
  }
  function show(r) {
    if (!r || r.shown) return;
    r.shown = true;
    if (r.kind === 'reply') cb('onSentence', { key: r.key, text: r.text, role: r.role, mid: r.mid, seq: r.seq, path: r.path });
  }
  function routeLocal(r) {
    if (r.kind === 'backchannel' || r.kind === 'ack') {
      // Fillers are never spoken by the phone (the `ack` earcon is), C.5 #12.
      claims.claim(r.sid, r.seq, 'text');
      r.path = 'text'; r.complete = true;
      return;
    }
    const path = localPath({ slowLink: pinned ? 'phone' : cfg().slowLink, phoneOk: phoneOk() });
    if (path === 'phone') {
      if (!claims.claim(r.sid, r.seq, 'phone')) return;      // its audio already started
      phoneEnqueue(r);
      return;
    }
    if (!claims.claim(r.sid, r.seq, 'text')) return;
    r.path = 'text'; r.complete = true;
    show(r);
    // Nothing will play a notice: tell the server at once so `speaking` can end (C.3).
    if (r.kind === 'notice' && sayCap && ready) send({ type: 'playback_done', gen: r.gen, seq: r.seq });
  }

  /** The next `say` (or `say_end`) means the previous sentence's audio is all here (T8-1 `end`). */
  function endPrevious() {
    if (!lastSayKey) return;
    const p = says.get(lastSayKey);
    lastSayKey = null;
    if (p && p.path === 'server') { p.complete = true; playerDo('end', p.gen, p.seq); }
  }

  function onSay(m) {
    if (typeof m.seq !== 'number') return;
    const { r, fresh } = record(m);
    if (!fresh) return;                      // the same session re-sent it (C.5 #1)
    if (genNewer(r.gen, curGen)) curGen = r.gen;
    endPrevious();
    lastSayKey = r.key;
    // The link monitor needs the v3 wire (offsets, server send times, wire3 chunks):
    // fed a v2 server's says it would miss every first-audio deadline.
    if (monitor && proto >= 3) monitor.onSay(r.seq, m.t, r.kind);
    if (r.kind === 'reply' && r.mid) {
      const cur = reply && reply.sid === sid ? reply : null;
      if (cur && !cur.ended && cur.gen === r.gen && !cur.ids.has(r.mid) && isProvisional(cur.mid)) {
        // `stream_done` named the row: the same reply goes on under its row id.
        cur.ids.add(r.mid);
        cur.mid = r.mid;
      } else if (!cur || !cur.ids.has(r.mid)) {
        reply = { mid: r.mid, ids: new Set([r.mid]), sid, gen: r.gen, lastSeq: null, ended: false, abandoned: false, turn: lastTurnRow };
        if (resumeFrom && resumeFrom.sid !== sid) {
          // The first reply of a resumed session goes on with the one the reconnect
          // named. Announced then under its provisional id only, it now carries its
          // row id: what the earlier sessions skipped counts under that id too.
          const fr = resumeFrom;
          resumeFrom = null;
          if (!fr.ids.has(r.mid) && [...fr.ids].every(isProvisional)) {
            partialBase.set(r.mid, fr.entry);
            for (const id of fr.ids) reply.ids.add(id);
            reply.mid = r.mid;
          }
        }
      }
    }
    cb('onSay', m);
    if (textMode && sayCap) routeLocal(r);
  }

  function onSayEnd(m) {
    // v3: the session sends `say_end` right after the last sentence's `say`, often
    // while its audio still streams; only `audio_end` completes a sentence there
    // (T6-2). Ending it early would make the player drop its later audio and the
    // resume count it as heard. v2 has no `audio_end`: the next frame ends it.
    if (proto < 3) endPrevious();
    // The resumed reply was said to its end on this socket: `after` goes back to
    // the newest reply heard (a later settings re-send must not resume it again).
    if (ready) resumeAnchor = null;
    const gen = (Number(m.gen) || 0) & 0xffff;
    // `say_end.message_id` is the row id when known, while the reply may have been
    // announced under its provisional id (T3s2-1): the same gen is the same reply.
    if (reply && reply.sid === sid && !reply.ended
        && (!m.message_id || reply.ids.has(m.message_id) || reply.gen === gen)) {
      reply.ended = true;
      reply.lastSeq = m.last_seq >>> 0;
      if (m.message_id && !reply.ids.has(m.message_id)) { reply.ids.add(m.message_id); reply.mid = m.message_id; }
      // The player may already have drained the last sentence (its `idle` came before
      // this `say_end`): the reply is over, so speech now is a new turn, not a duck.
      if (!playing && !playerBusy && !localSpeaking) {
        replyActive = false;
        if (duck) duck.onReplyEnd();
      }
    }
    const last = says.get(keyOf(sid, m.last_seq));
    if (proto < 3 && last && last.path === 'server') { last.complete = true; playerDo('end', gen, last.seq); }
    if (phoneResetPending) phoneHeld.push(['end', { gen, last_seq: m.last_seq }]);
    else if (localVoice) localVoice.sayEnd({ gen, last_seq: m.last_seq });
    if (monitor && proto >= 3) monitor.onSayEnd();
    if (last && last.path === 'text' && sayCap && ready) {
      send({ type: 'playback_done', gen, seq: last.seq });
      replyActive = false;
    }
  }

  function onAudioEnd(m) {
    const r = says.get(keyOf(sid, m.seq));
    if (r) r.complete = true;
    playerDo('end', (Number(m.gen) || 0) & 0xffff, m.seq >>> 0);
  }

  /** Server audio for one sentence (from the downlink, or a v2 pair). */
  function serverPush(m) {
    const seq = m.seq >>> 0;
    const s = sid;
    if (proto >= 3 && sid && !claims.claim(sid, seq, 'server')) return;   // the phone/caption has it
    let r = says.get(keyOf(s, seq));
    if (!r && m.kind !== 'probe') r = record({ seq, gen: m.gen, kind: m.kind, text: '' }, s).r;
    if (r) {
      r.path = 'server';
      const n = m.pcm ? m.pcm.length : 0;
      if (n && m.rate) r.ms += n / m.rate * 1000;
    }
    const g = (Number(m.gen) || 0) & 0xffff;
    if (genNewer(g, curGen)) curGen = g;
    playerBusy = true;
    if (!resetPending) playerSid = s;
    playerDo('push', { gen: g, seq, kind: m.kind || 'reply', rate: m.rate, pcm: m.pcm, totalMs: m.totalMs });
  }

  // ---- player events -------------------------------------------------------------
  const isStarved = () => !!(starved && playing && starved.gen === playing.gen && starved.seq === playing.seq);
  function onPlayed(d) {
    if (starved) {
      if (starved.gen !== d.gen || starved.seq !== d.seq) starved = null;
      else if (starved.ms === null) starved.ms = d.ms;     // the first report after the drain: where it stopped
      else if (d.ms > starved.ms) starved = null;           // audio again
    }
    const r = says.get(keyOf(playerSid, d.seq));
    const kind = (r && r.kind) || d.kind || 'reply';
    if (!playing || playing.seq !== d.seq || playing.gen !== d.gen) playing = { gen: d.gen, seq: d.seq, kind };
    if (d.ms > 0 && r) {
      show(r);
      if (kind === 'reply') {
        cb('onNow', r.key);
        if (!replyActive) { replyActive = true; if (duck) duck.onReplyStart(); }
        cb('onActivity');
      }
    }
    cb('onPlaying', kind);
  }

  function onIdle(d) {
    playing = null;
    starved = d.ended ? null : { gen: d.gen, seq: d.seq, ms: null };
    if (resetPending) {
      // The old session's audio has played out: now the new session's may start.
      if (doResetFn) doResetFn();
      cb('onPlaying', null);
      return;
    }
    playerBusy = false;
    if (proto < 3) {
      // v2: every drain is reported (the server ignores the stale ones, B6).
      const done = { type: 'playback_done', gen: d.gen };
      if (typeof d.seq === 'number') done.seq = d.seq;
      if (ready) send(done);
    } else if (d.ended && ready && playerSid === sid) {
      send({ type: 'playback_done', gen: d.gen, seq: d.seq });
    }
    const r = says.get(keyOf(playerSid, d.seq));
    if (r && r.kind === 'reply' && (!reply || reply.ended)) {
      replyActive = false;
      if (duck) duck.onReplyEnd();
    }
    cb('onPlaying', null);
  }

  // ---- uplink -----------------------------------------------------------------------
  function ensureUplink() {
    if (!B.createUplink) return null;
    const want = upFailed ? 'pcm16' : codecIn;
    if (uplink && uplinkCodec === want) return uplink;
    if (uplink) { try { uplink.close(); } catch { /* gone */ } }
    uplinkCodec = want;
    const u = B.createUplink({
      codec: want, kbps: upKbps, ring, codecs, now, setTimer, clearTimer,
      flags: () => (localSpeaking ? 1 : 0),
      onMessage: (buf, info) => {
        if (!live || !open) return;           // the ring keeps it for the replay
        sendBinary(buf);
        const n = info && info.packets ? info.packets : 1;
        sentEnd = ((info ? info.off : 0) + n * 320) >>> 0;
      },
      onError: (kind) => {
        if (kind !== 'encoder' || upFailed) return;
        upFailed = true;
        uplinkCodec = 'pcm16';                // the uplink switched itself to PCM16 for good
        if (monitor) monitor.setCodec({ up: 'pcm16' });
        if (proto >= 3 && open) send({ type: 'link', mode: textMode ? 'text' : 'audio', why: 'encoder' });
      },
    });
    uplink = u;
    return u;
  }

  function micNowOff() {
    if (!micClock) return null;
    return (micClock.off + Math.round((now() - micClock.at) * 16)) >>> 0;
  }

  function goLive(after = null) {
    live = true;
    if (after !== null && ring) {
      // What the mic produced while the replay was going out.
      for (const m of ring.replayFrom(after)) sendBinary(m);
    }
  }
  function startReplay(from) {
    clearTimer(cursorTimer); cursorTimer = null;
    if (!ring || live) return;
    const msgs = ring.replayFrom(from >>> 0, { nowOff: micNowOff() });
    if (ring.cutOff) cb('onCutOff');
    replayTo = ring.newest();
    if (!msgs.length) { onReplayDone(); return; }
    if (pump) pump.start(msgs); else { for (const m of msgs) sendBinary(m); onReplayDone(); }
  }
  function onReplayDone() {
    if (live || !open) return;
    const after = replayTo;
    replayTo = null;
    goLive(after);
  }

  // ---- public ------------------------------------------------------------------------
  const api = {
    /** Await opusSupport() once, before the first socket URL (C.1). */
    async probeSupport() {
      try { support = await B.opusSupport({ codecs }); } catch { support = { up: false, down: false }; }
      return support;
    },
    /** The `up=`/`down=` lists for the socket URL, best first. */
    codecLists() {
      return {
        up: support.up && !upFailed ? ['opus', 'pcm16'] : ['pcm16'],
        down: support.down && !downFailed ? ['opus', 'pcm16'] : ['pcm16'],
      };
    },
    /** The player worklet's port (or a buffer-player client) is ready. */
    setPlayer(p) {
      if (!p) return;
      player = typeof p.postMessage === 'function' && B.createPlayerClient
        ? B.createPlayerClient(p, playerCallbacks) : p;
      flushHeld();
    },
    playerCallbacks,
    /** The phone's voices arrived (speechSynthesis.getVoices, async on Chrome). */
    setVoices(list) { voices = Array.isArray(list) ? list : []; pickPhoneVoice(); cb('onLink', linkState()); },

    /** The socket opened: the per-socket `start` extras (C.3 cursor, after_partial). */
    opened() {
      open = true;
      ready = false;
      live = false;
      socketExtra = {};
      resumeAnchor = null;
      resuming = everV3 && !!ring;
      if (resuming) {
        const from = B.ringCursorFrom
          ? B.ringCursorFrom({ lastRxOff, lastTranscriptUpto, lastEotUpto })
          : Math.max(lastRxOff || 0, lastTranscriptUpto || 0, lastEotUpto || 0);
        socketExtra.cursor = { stream, from };
        if (!replySettled(reply)) {
          // Count the newest session's sentences on from what the sessions before it
          // skipped — and never count one session twice (a socket that opened and
          // dropped before `ready` leaves the same session in place). The reply is
          // named by its newest id (the row, once known); every id it went by counts.
          const ids = [...reply.ids].reverse();
          const saved = ids.map((id) => partialBase.get(id)).find(Boolean);
          const base = saved ? (saved.sid === reply.sid ? saved.base : saved.n) : 0;
          const n = completeSentences(order.filter((r) => r.sid === reply.sid), reply.ids, base);
          socketExtra.after_partial = { message_id: reply.mid, sentences: n };
          const entry = { sid: reply.sid, base, n };
          for (const id of ids) partialBase.set(id, entry);
          while (partialBase.size > 64) partialBase.delete(partialBase.keys().next().value);
          resumeFrom = { ids: new Set(reply.ids), entry, sid: reply.sid };
          resumeAnchor = { after: reply.turn || null, skip: new Set(reply.ids) };
        } else resumeFrom = null;
      }
      return api.startExtra();
    },
    /**
     * What every `start` on this socket carries besides the settings. While the
     * reply named by `after_partial` is still unsettled, `resume` says how to
     * anchor `after`: the real session speaks only replies stored AFTER `after`,
     * so naming the cut reply there would lose the rest of it (review-T7b H1).
     * `resume.after` is the user row it answers (null: unknown); `resume.skip`
     * are the cut reply's ids, never sent as `after`/`after_ids`.
     */
    startExtra() {
      const out = { ...socketExtra };
      if (socketExtra.after_partial && resumeAnchor) {
        const skip = new Set(resumeAnchor.skip);
        if (reply && reply.sid === sid && !reply.abandoned) for (const id of reply.ids) skip.add(id);
        out.resume = { after: resumeAnchor.after, skip: [...skip] };
      }
      return out;
    },

    ready(m) {
      ready = true;
      const wasReady = everReady;
      everReady = true;
      proto = Number(m.proto) >= 3 ? 3 : 2;
      sayCap = proto >= 3 || m.say === true;
      sessions += 1;
      // Sentence keys are per session (C.5 #9): a server without `sid` gets a local one.
      sid = typeof m.sid === 'string' && m.sid ? m.sid : `s${sessions}`;
      lastSayKey = null;
      // A new session numbers from seq 1 / gen 0 again (C.5 #9, T8-2): the old
      // session's gen must not gate this one's `resume`, nor raise the floor a
      // tap-stop flushes to (review-T7b H3).
      curGen = 0;
      stopHeard = null;
      claims.reset();
      if (duck) duck.reset();
      replyActive = false;
      // v2: no carry-over. v3: what the old session left plays out first (T6-2).
      resetPhone({ now: proto < 3 });
      resetPlayer({ now: proto < 3 });
      if (proto >= 3) {
        everV3 = true;
        codecIn = m.codec_in === 'opus' && !upFailed ? 'opus' : 'pcm16';
        codecOut = m.codec_out === 'opus' && !downFailed ? 'opus' : 'pcm16';
        // The operator's uplink rung (LinkState up_kbps: 16 or 12) starts this encoder.
        const uk = Number(m.opus_up_kbps);
        upKbps = uk === 12 || uk === 16 ? uk : 16;
        if (uplink) { try { uplink.setKbps(upKbps); } catch { /* closed */ } }
        if (monitor) {
          monitor.setCodec({ down: codecOut, up: codecIn });
          monitor.onReady();
          wireMode = 'audio';           // a new session starts in audio (T6 notes)
          pinned = cfg().slowLink === 'always_phone';
          monitor.setPinned(pinned ? 'text' : null);
          // Re-send a degraded state the new session does not know (T8 notes, T6 notes).
          const again = monitor.resendFrame();
          if (again) sendWire({ mode: again.mode, why: again.why, downKbps: again.down_kbps, upKbps: again.up_kbps });
        }
        if (B.gateFromReady) micPost(B.gateFromReady(m.gate));
        ensureUplink();
        const pre = !wasReady ? preReady.splice(0) : [];
        if (resuming) {
          // Replay what the server lacks (C.5 #10): from its `cursor {from}` answer, or
          // our own cursor when no answer comes.
          cursorTimer = setTimer(() => { cursorTimer = null; startReplay(socketExtra.cursor ? socketExtra.cursor.from : 0); }, CURSOR_WAIT_MS);
        } else if (pre.length && uplink && ring && pump) {
          // Speech from before `ready` (≤ 10 s) goes out like a ring replay, paced at 4×
          // realtime: as one burst, 10 s of PCM16 (≈ 325 kB) overflows the server's
          // 64,000 B audio queue (C.2) and the start of what was said is lost. The
          // uplink keeps each message in the ring while not live; Opus output that is
          // still being encoded when the replay ends goes out live.
          for (const f of pre) uplink.push(new Int16Array(f.buf), f.off);
          startReplay(pre[0].off);
        } else {
          goLive(null);
          if (uplink) for (const f of pre) uplink.push(new Int16Array(f.buf), f.off);
        }
      } else {
        micPost({ gate: { on: false } });
        live = true;
        if (!wasReady) for (const f of preReady.splice(0)) sendBinary(f.buf);
        // No link monitoring without the v3 wire. A `say`-capable server still honours
        // "Always use the phone's voice" (start.slow_link pins its text mode).
        pinned = sayCap && cfg().slowLink === 'always_phone';
        waitHold = false;
        textMode = pinned;
      }
      preReady.length = 0;
      cb('onLink', linkState());
    },

    /** The socket closed (any reason). The player keeps what it holds (T6-2). */
    closed() {
      open = false;
      ready = false;
      live = false;
      clearTimer(cursorTimer); cursorTimer = null;
      if (pump) pump.stop();
      replayTo = null;
      micPost({ hold: false });
      if (downlink) downlink.reset();
      if (monitor) monitor.onClose();
      pairer.pending = null;
      // A sentence whose audio had not all arrived is cut: it comes again in full.
      for (const r of order) if (r.sid === sid && r.path === 'server' && !r.complete) r.cut = true;
      if (proto < 3) { resetPlayer({ now: true }); replyActive = false; }
    },

    /** One JSON frame from the server (drive.js still paints the screen from it). */
    frame(m) {
      if (!m || typeof m !== 'object') return;
      switch (m.type) {
        case 'ready': api.ready(m); break;
        case 'audio': {               // v2: the next binary frame is this header's PCM
          pairer.pending = m;
          break;
        }
        case 'say': onSay(m); break;
        case 'say_end': onSayEnd(m); break;
        case 'audio_end': onAudioEnd(m); break;
        case 'duck': {
          // The phone voice cannot pause: a server duck is informational then (§4.6, H4).
          if (localSpeaking) break;
          if (duck) duck.onServerDuck(!!m.on);
          else { setDuckGain(m.on ? DUCK_GAIN : 1, DUCK_RAMP_MS); playerDo(m.on ? 'pause' : 'resume', false); cb('onDuck', !!m.on); }
          break;
        }
        case 'resume': {
          const g = (Number(m.gen) || 0) & 0xffff;
          if (g !== curGen) break;      // only ever for the current gen (C.4)
          if (localSpeaking || (localVoice && localVoice.current())) {
            if (m.rewind && localVoice && !phoneResetPending) localVoice.respeak();
            break;
          }
          if (duck) duck.onResume(!!m.rewind);
          else { setDuckGain(1, DUCK_RAMP_MS); playerDo('resume', !!m.rewind); cb('onDuck', false); }
          break;
        }
        case 'barge_in': api.bargeIn(m.gen); break;
        case 'cursor': if (resuming && !live) startReplay(Number(m.from) || 0); break;
        case 'transcript': {
          if (Number.isFinite(m.upto)) {
            lastTranscriptUpto = m.upto >>> 0;
            if (ring) ring.trim(lastTranscriptUpto);
          }
          if (m.text) {
            // A new turn: the reply before it was said in full or cut (a backchannel
            // or a cough over a reply never becomes a transcript).
            endReply();
            if (monitor && proto >= 3) monitor.onReplyCut();   // the server sends no say_end for a cut reply
            if (typeof m.message_id === 'string' && m.message_id) lastTurnRow = m.message_id;
            if (duck) duck.onTurnAccepted();
            if (bias) { bias.onCleanTurn(); micPost({ echoBiasDb: bias.value }); }
          }
          break;
        }
        case 'eot': if (Number.isFinite(m.upto)) lastEotUpto = m.upto >>> 0; break;
        case 'echo_ignored':
          if (duck) duck.onEchoIgnored();
          if (bias) { bias.onEchoIgnored(localSpeaking); micPost({ echoBiasDb: bias.value }); }
          break;
        case 'turn_error': {
          // The sentence whose speech failed is dead: never handed to the phone later.
          const dead = Number.isFinite(m.seq) ? says.get(keyOf(sid, m.seq))
            : (m.reason === 'tts_failed' && lastSayKey ? says.get(lastSayKey) : null);
          if (dead && !dead.path) dead.dead = true;
          // Any other reason fails the whole turn: nothing of the reply is left to resume.
          if (m.reason !== 'tts_failed' && reply && reply.sid === sid
              && (!m.message_id || reply.ids.has(m.message_id))) endReply();
          if (m.reason !== 'tts_failed' && monitor && proto >= 3) monitor.onReplyCut();   // no say_end is coming
          if (monitor && proto >= 3) monitor.onTurnError(Number.isFinite(m.seq) ? m.seq : undefined);
          break;
        }
        case 'error': if (m.code === 'rate_limited' && monitor && probeSentAt !== null) { monitor.probeFailed(); probeSentAt = null; } break;
        case 'metrics': if (monitor && proto >= 3 && Number.isFinite(m.first_chunk_ms)) monitor.onMetrics({ first_chunk_ms: m.first_chunk_ms }); break;
        case 'state':
          // The gate must not close while the server is still hearing (§5.6, T8-1).
          micPost({ hold: m.state === 'hearing' || m.state === 'transcribing' });
          break;
        default: break;
      }
    },

    /** One binary frame from the server. */
    binary(buf) {
      if (!(buf instanceof ArrayBuffer)) return;
      if (proto >= 3 && downlink && B.wire3 && B.wire3.isV3(buf)) {
        try { downlink.feed(buf, now()); } catch { /* a bad frame: dropped (C.2) */ }
        return;
      }
      const h = pairer.pending;
      pairer.pending = null;
      if (!h) return;
      // v2: the header carries the sentence (no `say`).
      const kind = h.kind || 'reply';
      if (typeof h.seq === 'number') {
        const { r, fresh } = record({ ...h, kind }, sid);
        if (fresh || (h.chunk === 0 && !r.text && h.text)) { r.text = String(h.text || r.text); cb('onSay', { ...h, kind, type: 'say' }); }
      }
      serverPush({ gen: h.gen || 0, seq: typeof h.seq === 'number' ? h.seq : 0, kind, rate: h.sample_rate || 24000, pcm: new Int16Array(buf) });
    },

    /** One message from the mic worklet. */
    micMessage(d) {
      if (d instanceof ArrayBuffer) d = { buf: d, off: micClock ? micClock.off : 0 };
      if (!d || typeof d !== 'object') return;
      if (d.buf instanceof ArrayBuffer) {
        const off = Number.isFinite(d.off) ? d.off >>> 0 : (micClock ? micClock.off : 0);
        micClock = { off: (off + 320) >>> 0, at: now() };
        // Half-duplex: the phone voice has no echo reference, so its own speaker would be
        // heard, transcribed and barged in on. Same length, same `off`: the offsets stay whole.
        if (micHalfDuplex()) d = { ...d, buf: new ArrayBuffer(d.buf.byteLength) };
        if (!everReady) {
          preReady.push({ off, buf: d.buf });
          if (preReady.length > PRE_READY_FRAMES) preReady.splice(0, preReady.length - PRE_READY_FRAMES);
          return;
        }
        if (everV3 && (proto >= 3 || !ready)) {
          // v3 (and between sockets after one): into the uplink — offline, its messages
          // wait in the ring for the replay.
          const u = uplink || ensureUplink();
          if (u) u.push(new Int16Array(d.buf), off);
          return;
        }
        if (ready && open) sendBinary(d.buf);   // a v2 server: raw PCM16, as today
        return;
      }
      if (d.gate === 'closed' && uplink) { uplink.flush(); return; }
      if (typeof d.speech === 'boolean') {
        if (duck && ready) duck.onSpeech(d.speech);
        if (!micHalfDuplex()) cb('onSpeech', d.speech);   // never "hearing you" for the phone's own voice
        return;
      }
      if (typeof d.level === 'number') cb('onMicLevel', d.level);
    },

    /** The listener paused (or resumed) the microphone. */
    micPaused(paused) {
      if (paused && uplink) uplink.flush();
    },

    /** `pong` (C.4): the RTT and what the server has received. */
    pong(m, rttMs) {
      if (Number.isFinite(m.rx_off)) lastRxOff = m.rx_off >>> 0;
      if (monitor && Number.isFinite(rttMs) && proto >= 3) monitor.onPong(rttMs, lastRxOff, sentEnd);
    },

    /** What the listener has heard of the sentence playing now (C.3 `heard`), or null. */
    heardNow() {
      if (localVoice && localVoice.current()) {
        // The old session's sentence (deferred reset): nothing this session can place.
        if (phoneResetPending) return null;
        const c = localVoice.current();
        return { gen: c.gen, seq: c.seq, played_ms: localVoice.playedMs(), source: 'phone' };
      }
      const last = player && player.last;
      if (!last || !playing) return null;
      // The old session's audio still playing out after a reconnect (review-T7b M3):
      // its (gen, seq) mean nothing to the new session.
      if (resetPending || playerSid !== sid) return null;
      const r = says.get(keyOf(playerSid, last.seq));
      const lim = r && r.ms > 0 ? r.ms : Infinity;
      const ms = B.heardPlayedMs ? B.heardPlayedMs(last.ms, latency() || {}, lim) : last.ms;
      return { gen: last.gen, seq: last.seq, played_ms: ms, source: 'server' };
    },

    /** Server `barge_in {gen}`: drop older audio, end any duck, report what was heard (C.3). */
    bargeIn(gen) {
      const g = (Number(gen) || 0) & 0xffff;
      // Before the flush clears `last` (T8 notes). A tap-stop already silenced the
      // player: report what it heard then, so the server need not wait 1.5 s (C.3).
      let h = api.heardNow();
      if (!h && stopHeard && now() - stopHeard.at < 2000) h = stopHeard.h;
      stopHeard = null;
      if (resetPending && doResetFn) { held = []; doResetFn(); }
      if (phoneResetPending) dropHeldPhone((r) => genNewer(g, r.gen));
      if (!genNewer(curGen, g)) curGen = g;
      playerDo('flush', g);
      if (downlink) downlink.reset();
      if (localVoice) localVoice.dropOlderThan(g);
      if (duck) duck.onBargeIn(); else { setDuckGain(1, DUCK_RAMP_MS); cb('onDuck', false); }
      // The cut reply is abandoned: a reconnect must not ask the server to resume it.
      // (A reply of an earlier session is older whatever its gen: gens restart per session.)
      if (reply && (reply.sid !== sid || genNewer(g, reply.gen))) endReply();
      for (const r of order) if (r.sid === sid && genNewer(g, r.gen) && !r.shown) r.cut = true;
      replyActive = false;
      if (monitor && proto >= 3) monitor.onReplyCut();   // the server sends no say_end for a cut reply
      playing = null;
      if (sayCap && ready && h) send({ type: 'heard', gen: g, seq: h.seq, played_ms: h.played_ms, source: h.source });
      cb('onPlaying', null);
    },

    /**
     * The listener stopped the reply here (tap, cancel, undo): silence now and drop
     * this gen's chunks still on the wire. Returns the `heard` sub-object (C.3a).
     */
    stopHere() {
      const h = api.heardNow();
      if (h) stopHeard = { h, at: now() };
      const g = (curGen + 1) & 0xffff;
      // The old session's audio still playing out (deferred reset) stops now too —
      // not one round trip later, when the server's barge_in comes (review-T7b M3).
      if (resetPending && doResetFn) { held = []; doResetFn(); }
      if (phoneResetPending) dropHeldPhone(() => true);
      playerDo('flush', g);
      if (downlink) downlink.reset();
      if (localVoice) localVoice.dropOlderThan(g);
      if (duck) duck.onBargeIn();
      endReply();               // the listener stopped it: nothing to resume after a drop
      replyActive = false;
      if (monitor && proto >= 3) monitor.onReplyCut();
      playing = null;
      cb('onPlaying', null);
      return sayCap && h ? h : null;
    },

    /**
     * The server took the turn back (`undone`). Its `barge_in` (sent just before)
     * already flushed this gen and ended the duck; flushing again at gen + 1 would
     * drop the NEXT reply, which the server sends at that same gen (review-T7b H2).
     */
    undone() {
      endReply();
      replyActive = false;
      if (monitor && proto >= 3) monitor.onReplyCut();
      stopHeard = null;
    },

    /** Every ~250 ms: the monitor's clock, probes, `heard` while a reply plays, phone captions. */
    tick() {
      if (disposed) return;
      showPhoneCaption();
      if (!sayCap || !open || !ready) return;
      if (monitor && proto >= 3) {
        monitor.tick();
        if (monitor.wantProbe()) {
          probeSentAt = now();
          send({ type: 'probe', bytes: th.probeBytes || 16384 });
        }
      }
      if (replyActive || (playing && playing.kind === 'reply') || localSpeaking) {
        const t = now();
        if (t - lastHeardAt >= HEARD_EVERY_MS) {
          const h = api.heardNow();
          if (h) { lastHeardAt = t; send({ type: 'heard', ...h }); }
        }
      }
    },

    /** The device settings changed (the slow-link choice may pin the phone voice). */
    settingsChanged() {
      pickPhoneVoice();
      const want = cfg().slowLink === 'always_phone';
      if (proto < 3) {
        // No link monitoring here; the next `start.slow_link` re-pins the server.
        pinned = sayCap && want;
        textMode = pinned;
        waitHold = false;
        cb('onLink', linkState());
        return;
      }
      if (monitor && want !== pinned) { pinned = want; monitor.setPinned(want ? 'text' : null); }
      // A degraded link under a different choice (Wait ↔ Phone voice / Text): re-map it.
      if (monitor) {
        const cur = monitor.current();
        applyLinkMode(cur.mode, cur);
        const w = wireLink({ ...cur, why: 'user' }, pinned ? 'phone' : cfg().slowLink, th);
        if (w.frame.mode !== wireMode) sendWire({ ...cur, why: 'user' });
      }
      cb('onLink', linkState());
    },

    stats() {
      const m = monitor ? monitor.stats() : {};
      return { ...m, path: linkState().path, proto, codec_in: codecIn, codec_out: codecOut, phone: phoneOk() };
    },

    get proto() { return proto; },
    /** The server sends `say` and reads `heard` (v3, or a v2 server whose `ready.say` is true). */
    get sayCap() { return sayCap; },
    get sid() { return sid; },
    get live() { return live; },
    get link() { return linkState(); },
    /** A reply is playing here (server audio or the phone's voice). */
    get replyPlaying() { return (!!playing && playing.kind === 'reply' && !isStarved()) || localSpeaking; },
    get playingKind() { return localSpeaking ? 'reply' : (playing && !isStarved() ? playing.kind : null); },
    get localSpeaking() { return localSpeaking; },
    /** The phone voice speaks (or just stopped): the mic is being zeroed. */
    get micHalfDuplex() { return micHalfDuplex(); },
    get phoneOk() { return phoneOk(); },
    get ring() { return ring; },
    /** For tests: the sentence records in arrival order. */
    get records() { return order.slice(); },

    dispose() {
      disposed = true;
      clearTimer(cursorTimer); clearTimer(resetTimer);
      if (pump) pump.stop();
      try { uplink && uplink.close(); } catch { /* gone */ }
      try { downlink && downlink.close(); } catch { /* gone */ }
      try { localVoice && localVoice.cancel(); } catch { /* gone */ }
      if (duck) duck.dispose();
      micPost({ hold: false });
      try { player && player.reset(); } catch { /* gone */ }
    },
  };
  return api;
}
