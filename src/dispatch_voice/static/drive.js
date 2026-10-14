// Voice chat — hands-free voice for the open thread (unlocked tier only).
// Owner: track T7. Contracts: C-§C (the client side of the wire protocol),
// C-§E.2 (the modules it consumes, through bridge.js), SPEC §5.5, §7.3, §7.4.
//
// One full-screen view built around the bot's own avatar in a state ring:
// the ring's colour, shape and motion say what is happening, a word under it
// names the state, and a hint line says what a tap on the picture will do.
// Errors are written ON the overlay (a notice strip) and played as a tone,
// never only as a toast. A call screen's row at the bottom (1.2.0): round
// icon buttons with a word under each — Mute/Unmute (the microphone), Type
// (a text box for a headset or a quiet room: the message goes over the voice
// socket as `typed`, the reply is spoken as usual) and Hang up. The gear opens
// a quick sheet of the settings people change mid-conversation
// (drive-settings.js); "Type instead of talking" there opens every chat muted
// with the text box out.
//
// v3 additions (SPEC §7.4): Mute mutes only the microphone — a reply that
// is playing finishes ("Muted — still speaking"), the picture tap is the
// interrupt (the control was "Pause" until FR-ux M5: a pause glyph promised a
// media pause it never did); one sound per turn end (the `eot` tone waits
// 800 ms and is dropped when an acknowledgement or the first reply sentence
// arrives);
// live captions (committed words full colour, the unsettled tail dimmed);
// "Got it — go on?" while the server holds the turn ≥ 600 ms; narrator
// sentences in italics; a first-run hint (or the English-only line) on the
// first three opens; `waiting` after 60 s offline, retrying up to 15 min;
// Hang up when quiet; All settings… opens over this screen without hanging up.
//
// v3 part 2 (the audio path, drive-audio.js): the player worklet (pause,
// rewind, `played`/`idle`), the mic gate and offsets, Opus both ways with a
// PCM16 fallback, the link monitor → phone voice (or captions, or Wait) with
// the standing notice, the uplink ring + `start.cursor` replay and
// `after_partial`, `(sid, seq)` claims, `heard` frames, the optimistic duck.
// Captions show a sentence when it starts PLAYING, so text the listener never
// heard (a barge-in) is never appended. A v2 server still gets today's wire.
//
// The browser only moves audio. Mic → AudioWorklet (16 kHz PCM16) →
// /ws/voice/<thread>; the server does turn detection, speech-to-text, sends
// the words through the thread's NORMAL send path, and streams the reply back.
// Voice chat runs with the screen on: a hidden page loses the mic, so it
// pauses itself on hide and says so on return (feasibility M6).
//
// Pure helpers are exported for tests/js/drive.test.js; the DOM half only
// runs inside openDrive().

const Q = new URL(import.meta.url).search;
const { el, railIcon, iconPaths, t, language, sentences, acquireInert, releaseInert, activity: shellActivity } = await import(`./shell.js${Q}`);
const {
  PATIENCE_PRESETS, loadDriveSettings, saveDriveSettings, onDriveSettings, startPayload,
  buildQuickSheet, MIC_ICON, phoneVoices, isAndroid, canPickSpeaker, deviceOptions,
} = await import(`./drive-settings.js${Q}`);
const { assetUrl, isBluetoothOutput } = await import(`./bridge.js${Q}`);
const { createDriveAudio, createBufferPlayerClient } = await import(`./drive-audio.js${Q}`);

export { PATIENCE_PRESETS };

// ---------------------------------------------------------------- pure ----

/** Reconnect delay: 0.5 s doubling to 8 s; every 15 s once `waiting` (§5.5). */
export function nextBackoff(prev, { waiting = false } = {}) {
  if (waiting) return WAITING_RETRY_MS;
  return Math.min(8000, prev ? prev * 2 : 500);
}

/** After this long offline the screen says "Waiting for signal" (§5.5). */
export const WAITING_AFTER_MS = 60000;
/** Retry period while waiting. */
export const WAITING_RETRY_MS = 15000;
/** A dropped connection retries for this long before voice chat stops and asks. */
export const GIVE_UP_MS = 15 * 60000;
/** Heartbeat (C.3 `ping`): every 2 s; dead after 8 s of silence, or 4 × the RTT above 1.5 s. */
export const PING_EVERY_MS = 2000;
export const PONG_TIMEOUT_MS = 8000;
/** Before `ready` the server is loading models and answers nothing: allow this long. */
export const PRE_READY_MS = 60000;
/** The audio engine's clock tick (link monitor, probes, `heard` cadence). */
export const AUDIO_TICK_MS = 250;
/** The first socket waits at most this long for the WebCodecs check (C.1 codec lists). */
export const SUPPORT_WAIT_MS = 1000;
/** How long Undo is offered after a transcript (contract: 8 s). */
export const UNDO_MS = 8000;
/**
 * The `eot` tone waits this long for an ack or the first reply sentence (experience M1).
 * The server sends the ack line 0.55 s after `eot` (host config `ack_after_s`); the hold
 * leaves 250 ms for the network, so an acknowledged turn makes ONE sound (FR-works 5:
 * at 650 ms against a 0.7 s ack both played, 50 ms apart). A test pins the pair.
 */
export const EOT_TONE_HOLD_MS = 800;
/** "Got it — go on?" only for holds at least this long (shorter ones flicker, L10). */
export const GOTIT_MIN_HOLD_MS = 600;
/** The first-run hint shows on this many opens per device (L12). */
export const FIRST_RUN_OPENS = 3;
/** After "Still there?", this much more silence hangs up (A9). */
export const QUIET_GRACE_MS = 20000;
/** A `typed` message (1.2.0): the server's TYPED_MAX_CHARS, checked here first so an over-long one is never sent — and never lost. */
export const TYPED_MAX_CHARS = 4000;
/** A stopped reply can be continued by a tap for this long (design-realtime). */
export const RESUME_AFTER_STOP_MS = 3000;
/**
 * While a voice chat is open with the microphone on and the page visible, DisPatch's
 * idle auto-lock is told the person is active at least this often (`shell.activity`):
 * talking makes no touches, and the shell would otherwise lock mid-conversation
 * (FR-works 1). The same policy as the server's session touch (SPEC §5.6); the
 * server's own quiet bound (`rt.quiet_max_s`) still ends a chat nobody uses.
 */
export const ACTIVITY_EVERY_MS = 20000;
/** After the listener scrolls the captions by hand, they stop following the speech this long (FR-ux H1). */
export const FOLLOW_PAUSE_MS = 4000;
/** Frames that are conversation, not plumbing: each one is activity for the idle lock. */
const ACTIVITY_FRAMES = new Set(['state', 'partial', 'transcript', 'say', 'eot', 'barge_in']);

/**
 * What to do after the voice socket closed (C-§C.4 close codes):
 *   1008  refused (locked / not allowed)      → stop, say why
 *   1013  unavailable or busy                 → stop, say why (no retry loop);
 *         the reason names which: the switch is off, the server is busy
 *         (too many voice chats), or voice is not set up
 *   4001  replaced by a newer socket          → close quietly
 *   4002  the server ended an idle chat       → hang up ("Voice chat ended."), no retry
 *         (nothing said either way for `rt.quiet_max_s`; Amendment FINAL-FIX-2)
 *   1011  a server-side worker died           → retry with backoff
 *   1001  heartbeat lost / else the network   → retry with backoff
 */
export function reconnectDecision(code, why = '') {
  if (code === 1008) return { retry: false, reason: 'refused' };
  if (code === 4002) return { retry: false, reason: 'quiet' };
  const reason = String(why || '');
  if (code === 1013 && /^Voice is turned off/i.test(reason)) return { retry: false, reason: 'voice_off' };
  if (code === 1013 && /busy/i.test(reason)) return { retry: false, reason: 'busy' };
  if (code === 1013) return { retry: false, reason: 'unavailable' };
  if (code === 4001) return { retry: false, reason: 'replaced' };
  if (code === 1011) return { retry: true, reason: 'internal' };
  return { retry: true, reason: 'network' };
}

const fmtMs = (ms) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);
const MODE_KEYS = { audio: 'drive.mode_server', phone: 'drive.mode_phone', text: 'drive.mode_text' };

/**
 * The Voice lab "Show timings" line from the last `metrics` frames, as sentences:
 * "Heard in 420 ms. Reply started in 1.6 s.", plus the link sentence
 * "420 ms, 31 kbps, phone voice." when the server reported one; '' when nothing yet.
 */
export function timingsLine(times) {
  const parts = [];
  if (times && typeof times.heard === 'number') parts.push(t('drive.timing_heard', { t: fmtMs(times.heard) }));
  if (times && typeof times.first === 'number') parts.push(t('drive.timing_first', { t: fmtMs(times.first) }));
  const link = times && times.link;
  if (link && typeof link.rtt_ms === 'number') {
    const kbps = typeof link.down_kbps === 'number' ? link.down_kbps : '–';
    parts.push(t('drive.timing_link', { rtt: Math.round(link.rtt_ms), kbps, mode: t(MODE_KEYS[link.mode] || MODE_KEYS.audio) }));
  }
  return sentences(parts);
}

/**
 * ws(s)://host/ws/voice/<thread> for the page's own origin (C-§C.1). `opts`
 * adds the v3 negotiation (`v=3`, stream id, codec lists) once the client
 * speaks v3; without it the server keeps today's v2 wire.
 */
export function voiceUrl(loc, threadId, device = '', opts = null) {
  const proto = loc.protocol === 'https:' ? 'wss' : 'ws';
  const q = new URLSearchParams();
  if (device) q.set('d', device);
  if (opts && opts.v3) {
    if (opts.stream) q.set('s', opts.stream);
    q.set('v', '3');
    q.set('up', (opts.up || ['pcm16']).join(','));
    q.set('down', (opts.down || ['pcm16']).join(','));
  }
  const qs = q.toString();
  return `${proto}://${loc.host}/ws/voice/${encodeURIComponent(threadId)}${qs ? `?${qs.replace(/%2C/g, ',')}` : ''}`;
}

export const DEVICE_KEY = 'dispatch-drive-device';

/** A fresh stream id for one voice chat (C.1 `s`, the replay ring's key): [A-Za-z0-9-]{8,64}. */
export function newStreamId() {
  const c = globalThis.crypto;
  if (c && typeof c.randomUUID === 'function') return c.randomUUID();
  return `s${Date.now().toString(36)}${Math.random().toString(36).slice(2, 12)}`;
}

/**
 * This browser's voice device id: random, kept in localStorage. On an install
 * WITHOUT a PIN there is no session to tell devices apart, so the server
 * counts voice sockets per device id instead. Not a secret and not an
 * identity: with a PIN set the server ignores it.
 */
export function driveDeviceId(storage = globalThis.localStorage) {
  try {
    let id = storage && storage.getItem(DEVICE_KEY);
    if (id && /^[A-Za-z0-9-]{8,64}$/.test(id)) return id;
    const c = globalThis.crypto;
    id = c && c.randomUUID ? c.randomUUID() : `d${Date.now().toString(36)}${Math.random().toString(36).slice(2, 12)}`;
    if (storage) storage.setItem(DEVICE_KEY, id);
    return id;
  } catch { return ''; }
}

/** Can voice chat open for this thread at all? (The server re-checks.) */
export function driveAllowed({ locked, thread, bot, secure, voiceOff = false }) {
  const no = (whyKey) => ({ ok: false, whyKey, why: t(whyKey) });
  if (locked) return no('drive.err_locked_device');
  if (voiceOff) return no('drive.button_voice_off');
  if (!thread) return no('drive.err_no_chat');
  // 1.3.1: a `safe` bot (one locked devices may also talk to) gets voice like
  // any other — voice itself stays unlocked-only (the device must hold the PIN).
  if (!secure) return no('drive.err_https');
  return { ok: true, whyKey: '', why: '' };
}

/**
 * What a tap on the picture does in each state. The hint line under the
 * state word always says this out loud, so no mode is a silent surprise.
 * `playing` = a REPLY is playing (not an ack or a spoken notice).
 * `resumable` = a reply was stopped < 3 s ago and the server can replay it.
 */
export function tapAction(state, { playing = false, resumable = false } = {}) {
  if (playing || state === 'speaking') return 'interrupt';
  if (resumable && (state === 'listening' || state === 'paused')) return 'continue';
  switch (state) {
    case 'listening': case 'hearing': return 'pause';
    case 'paused': return 'resume';
    case 'thinking': case 'transcribing': return 'cancel';
    case 'stopped': return 'retry';
    default: return 'none';
  }
}

/** Thinking feedback: a plain word, then a counter at 4 s, then "still working" at 25 s. */
export function thinkingPhase(ms) {
  if (ms >= 25000) return 'long';
  if (ms >= 4000) return 'counting';
  return 'plain';
}

/**
 * The hint-line key for a state (null = no hint). `sounds` false: no tone will
 * play, so "start talking after the tone" becomes "…when it says Listening" (FR-ux L2).
 */
export function hintKey(state, { playing = false, thinkingMs = 0, autoListen = true, resumable = false, sounds = true } = {}) {
  switch (tapAction(state, { playing, resumable })) {
    case 'interrupt': return 'drive.hint_interrupt';
    case 'continue': return 'drive.hint_resume_after_stop';
    case 'pause': return state === 'hearing' ? 'drive.hint_keep_talking' : (autoListen ? 'drive.hint_pause' : 'drive.hint_pause_ptt');
    case 'resume': return 'drive.hint_resume';
    case 'cancel': return thinkingPhase(thinkingMs) === 'long' ? 'drive.hint_still_working' : 'drive.hint_cancel';
    case 'retry': return null;
    default: break;
  }
  if (state === 'connecting') return sounds ? 'drive.hint_wait_tone' : 'drive.hint_wait_listening';
  if (state === 'offline') return 'drive.hint_offline';
  if (state === 'waiting') return 'drive.hint_waiting';
  return null;
}

/**
 * The ring/word state to SHOW. Pause pauses only the microphone (experience
 * M7): while the listener has paused, every live server state reads as
 * `paused`; the word says "Paused — still speaking" while a reply plays.
 */
export function displayState(state, { listening = true, playing = false, ducked = false } = {}) {
  const live = ['listening', 'hearing', 'transcribing', 'thinking', 'speaking', 'paused'];
  if (!listening && live.includes(state)) {
    return { state: 'paused', wordKey: (playing || state === 'speaking') ? 'drive.paused_speaking' : 'drive.state_paused' };
  }
  // The reply is held because the listener started talking (a duck, §4.6): the
  // picture says it is hearing them; a word-confirmed interruption follows, or
  // the reply resumes.
  if (ducked && (state === 'speaking' || playing)) return { state: 'hearing', wordKey: null };
  return { state, wordKey: null };
}

/** States in which the first-run tip is hidden (it comes back with listening). */
const TIPLESS = new Set(['offline', 'waiting', 'stopped', 'locked']);

/**
 * The first-run tip for this open (L12, H3): null once the device has seen it 3 times.
 * With "Interrupt by talking" off the tip says how to stop a reply instead
 * (FR-ux L2: "talk over me" was the one thing that would not work).
 */
export function firstRunTipKey(opens, lang = 'en', { bargeIn = true } = {}) {
  if (!(opens < FIRST_RUN_OPENS)) return null;
  if (opens === 0 && String(lang || 'en').split('-')[0] !== 'en') return 'drive.english_only';
  return bargeIn ? 'drive.first_run_hint' : 'drive.first_run_hint_tap';
}

/**
 * The persistent notice while the link is degraded (§7.4): which voice the
 * replies come in now. `degraded` false → null. `phoneOk` = this device has a
 * working phone voice.
 */
export function linkNoticeKey(degraded, slowLink, phoneOk = true) {
  if (!degraded) return null;
  if (slowLink === 'wait') return 'drive.link_wait';
  if (slowLink === 'text') return 'drive.link_text';
  return phoneOk ? 'drive.link_phone' : 'drive.link_text';
}

/**
 * The notice key for the audio engine's link state ({degraded, notice: 'phone'|'text'|'wait',
 * pinned}). "Always use the phone's voice" on a device that has none is not a slow
 * link: it gets its own line (FR-ux L3: it used to say "Low signal" on a healthy link).
 */
export function linkStateNoticeKey(l) {
  if (!l || !l.degraded) return null;
  if (l.pinned && l.notice === 'text') return 'drive.link_no_phone_voice';
  return linkNoticeKey(true, l.notice === 'wait' ? 'wait' : (l.notice === 'text' ? 'text' : 'phone'), l.notice === 'phone');
}

/** Short earcons (~150–250 ms). `ready` rises, `err` falls. */
export const EARCONS = {
  eot: [660, 880],
  ack: [520, 520],
  ready: [523, 659, 784],
  err: [440, 330],
};

export function playEarcon(ctx, kind = 'eot') {
  const tones = EARCONS[kind] || EARCONS.err;
  const t0 = ctx.currentTime + 0.01;
  tones.forEach((f, i) => {
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, t0 + i * 0.075);
    g.gain.exponentialRampToValueAtTime(0.25, t0 + i * 0.075 + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + i * 0.075 + 0.07);
    o.connect(g); g.connect(ctx.destination);
    o.start(t0 + i * 0.075); o.stop(t0 + i * 0.075 + 0.08);
  });
}

/**
 * One sound per turn end (experience M1): `eot()` arms the turn-end tone,
 * which plays after EOT_TONE_HOLD_MS unless `suppress()` (an acknowledgement
 * or the first reply sentence arrived) came first. Timers injectable.
 */
export function createEotTone({ play, hold = EOT_TONE_HOLD_MS, setT = (fn, ms) => globalThis.setTimeout(fn, ms), clearT = (h) => globalThis.clearTimeout(h) } = {}) {
  let timer = null;
  return {
    eot() {
      if (timer != null) clearT(timer);
      timer = setT(() => { timer = null; play && play(); }, hold);
    },
    suppress() { if (timer != null) { clearT(timer); timer = null; } },
    get pending() { return timer != null; },
    cancel() { if (timer != null) clearT(timer); timer = null; },
  };
}

/**
 * Hang up when quiet (experience M9, C-§C.3 `quiet_check`): after `minutes`
 * with no speech either way, `onCheck()` (the client sends `quiet_check`; the
 * server says "Still there?"); QUIET_GRACE_MS more without the LISTENER
 * answering → `onHangup()`. 0 minutes = never.
 *   activity()  the bot spoke (a reply, `state: speaking`): resets the idle
 *               clock, but NOT a pending check — the server's own "Still
 *               there?" is spoken audio too, and must not cancel itself.
 *   heard()     the listener answered (`hearing`, a partial, a transcript, a
 *               button): resets everything.
 * `tick({ canAsk })` is driven by the overlay's 1 s timer (and by tests).
 * `canAsk: false` (the microphone is paused, so nobody can answer by voice):
 * at the deadline it hangs up without asking.
 */
export function createQuietTimer({ minutes = 15, now = () => Date.now(), onCheck, onHangup, grace = QUIET_GRACE_MS } = {}) {
  let last = now();
  let checkedAt = null;
  let mins = minutes;
  return {
    activity() { if (checkedAt == null) last = now(); },
    heard() { last = now(); checkedAt = null; },
    setMinutes(m) { mins = m; last = now(); checkedAt = null; },
    get checking() { return checkedAt != null; },
    tick({ canAsk = true } = {}) {
      if (!mins) return null;
      const n = now();
      if (checkedAt != null) {
        if (n - checkedAt >= grace) { checkedAt = null; last = n; onHangup && onHangup(); return 'hangup'; }
        return null;
      }
      if (n - last >= mins * 60000) {
        if (!canAsk) { last = n; onHangup && onHangup(); return 'hangup'; }
        checkedAt = n; onCheck && onCheck(); return 'check';
      }
      return null;
    },
  };
}

/** Undo is offered for UNDO_MS after a transcript lands. */
export function createUndoWindow(ms = UNDO_MS, now = () => Date.now()) {
  let openedAt = null;
  return {
    open() { openedAt = now(); },
    close() { openedAt = null; },
    canUndo() { return openedAt != null && now() - openedAt < ms; },
    remaining() { return openedAt == null ? 0 : Math.max(0, Math.ceil((ms - (now() - openedAt)) / 1000)); },
  };
}

/**
 * Heartbeat (C.3/C.4): ping every PING_EVERY_MS from the moment the socket
 * opens; `check()` reports dead when nothing came back for the later of
 * PONG_TIMEOUT_MS and 4 × the RTT EWMA once that is above 1.5 s (a slow
 * cellular link must not be mistaken for a dead one). `pong(t)` feeds the RTT.
 * Until `arm()` (called on `ready`) the deadline is PRE_READY_MS: the server
 * accepts first and loads the speech models before it answers anything, and a
 * cold load must not look like a dead link. Timers are injected for tests.
 */
export function createHeartbeat({ send, onDead, now = () => Date.now(), every = PING_EVERY_MS, timeout = PONG_TIMEOUT_MS,
  preReady = PRE_READY_MS,
  setInt = (fn, ms) => globalThis.setInterval(fn, ms), clearInt = (h) => globalThis.clearInterval(h) } = {}) {
  let last = 0;
  let timer = null;
  let rtt = null;
  let armed = true;
  const hb = {
    /** `armed: false` = the socket is open but the server has not said `ready`. */
    start({ armed: a = true } = {}) { last = now(); armed = a; hb.stop(); timer = setInt(() => hb.tick(), every); },
    arm() { if (!armed) { armed = true; last = now(); } },
    stop() { if (timer != null) clearInt(timer); timer = null; },
    alive() { last = now(); },
    pong(sentAt) {
      const n = now();
      last = n;
      if (typeof sentAt === 'number' && sentAt <= n) {
        const r = n - sentAt;
        rtt = rtt == null ? r : rtt * 0.8 + r * 0.2;
      }
    },
    get rtt() { return rtt; },
    get armed() { return armed; },
    deadline() {
      const d = rtt != null && rtt > 1500 ? Math.max(timeout, 4 * rtt) : timeout;
      return armed ? d : Math.max(d, preReady);
    },
    dead() { return now() - last >= hb.deadline(); },
    tick() {
      if (hb.dead()) { hb.stop(); onDead && onDead(); return; }
      send && send({ type: 'ping', t: now() });
    },
  };
  return hb;
}

/**
 * The replies the listener has heard, newest first. A streamed reply's audio
 * carries its PROVISIONAL id until the stream completes, so the reconnect
 * `start` sends the newest id as `after` and the recent ones as `after_ids`.
 */
export function createHeardTracker(max = 8) {
  let ids = [];
  return {
    note(id) {
      if (!id) return;
      if (ids[0] === id) return;
      ids = [id, ...ids.filter((x) => x !== id)].slice(0, max);
    },
    get after() { return ids[0] || null; },
    get ids() { return ids.slice(); },
  };
}

/** Plain words for a `turn_error` reason (contract). */
export function turnErrorKey(reason) {
  switch (reason) {
    case 'no_speech_in_reply': return 'drive.err_nothing_to_say';
    case 'timeout': return 'drive.err_timeout';
    case 'tts_failed': return 'drive.err_tts';
    default: return 'drive.err_agent';
  }
}

/**
 * Plain words for an `error` frame (null = nothing to show). New servers send
 * a `code`; a 2.1.0 server sends only an English `message` ("Drive mode hit
 * an internal error…"), matched here so its errors still read well.
 */
export function errorFrameKey(m) {
  const code = m && m.code;
  const msg = String((m && m.message) || '');
  if (code === 'undo_expired') return 'drive.err_undo_expired';
  if (code === 'stt_failed' || /recognition/i.test(msg)) return 'drive.err_stt';
  if (code === 'rate_limited') return 'drive.err_rate';
  if (code === 'unavailable' || /internal error/i.test(msg)) return 'drive.err_internal_retry';
  if (code === 'bad_frame') return null;
  if (/synthesis/i.test(msg)) return 'drive.err_tts';
  if (/could not send/i.test(msg)) return 'drive.err_send';
  return 'drive.err_generic';
}

/** The last `metrics` this device saw (the Voice lab shows them). */
let lastMetrics = null;
export function getLastMetrics() { return lastMetrics; }
/** Merge one `metrics` frame into the remembered set (exported for tests). */
export function noteMetrics(m, into = lastMetrics) {
  const out = { ...(into || {}) };
  for (const [k, v] of Object.entries(m || {})) if (k !== 'type') out[k] = v;
  out.at = Date.now();
  lastMetrics = out;
  return out;
}

/**
 * Warm the HTTP cache with the two audio worklet modules (served immutable under
 * the entry's `?v=`), so the first voice chat on a slow link does not fetch them
 * after the tap — on EDGE that started the microphone ~3 s late (browser e2e).
 * Once per page; a failed fetch may be retried by a later call.
 */
let workletsPrefetched = false;
export function prefetchAudioWorklets(fetchFn = globalThis.fetch) {
  if (workletsPrefetched || typeof fetchFn !== 'function') return false;
  workletsPrefetched = true;
  for (const n of ['voice-worklet.js', 'voice-player-worklet.js']) {
    try {
      Promise.resolve(fetchFn(assetUrl(n), { credentials: 'same-origin' })).catch(() => { workletsPrefetched = false; });
    } catch { workletsPrefetched = false; }
  }
  return true;
}

// ----------------------------------------------------------------- DOM ----

let active = null;    // the one open voice chat
let opening = false;  // set synchronously by openDrive: a double tap opens one

/**
 * Wire the header button. `ctx` supplies the app's view of the world:
 *   getThread() -> {id, bot_id, title} | null, getBot(id) -> bot | null,
 *   isLocked() -> bool, toast(msg, isError), getAvatar(thread, bot) ->
 *   {src, letter}, voiceStatus() -> /api/voice/status | null,
 *   openDriveSettings() -> Settings → Voice, setBotVoice(botId, voice),
 *   reloadBots() -> Promise (optional; re-read the roster before the quick sheet),
 *   history: {push(tag, onBack) -> pop()} (optional).
 * Call refresh() whenever the open thread or the lock state changes.
 */
export function initDrive(button, ctx) {
  if (!button) return { refresh() {}, close() {} };
  const verdict = () => {
    const thread = ctx.getThread();
    return {
      thread,
      v: driveAllowed({
        locked: ctx.isLocked(), thread, bot: thread && ctx.getBot(thread.bot_id),
        secure: !!globalThis.isSecureContext,
        voiceOff: !!(ctx.isVoiceOff && ctx.isVoiceOff()),
      }),
    };
  };
  const refresh = () => {
    const { thread, v } = verdict();
    // Hidden when locked (or voice is off); shown-but-explaining on http.
    button.hidden = ctx.isLocked() || !!(ctx.isVoiceOff && ctx.isVoiceOff()) || !thread;
    const label = v.ok ? t('drive.button_aria') : v.why;
    button.title = label;
    button.setAttribute('aria-label', v.ok ? t('drive.button_aria') : `${t('drive.button')}: ${v.why}`);
    button.dataset.ready = v.ok ? '1' : '0';
  };
  const onClick = () => {
    const { thread, v } = verdict();
    if (!v.ok) { ctx.toast(v.why, true); return; }
    openDrive(thread, { ...ctx, returnFocus: button });
  };
  button.addEventListener('click', onClick);
  refresh();
  return {
    refresh,
    close: (opts) => active && active.close(opts),
    dispose() { button.removeEventListener('click', onClick); if (active) active.close({ quiet: true }); },
    get active() { return active; },
  };
}

let silentUrl = null;
function silentWavUrl() {
  if (silentUrl) return silentUrl;
  // 1 s of 8 kHz 8-bit silence.
  const n = 8000;
  const b = new Uint8Array(44 + n);
  const dv = new DataView(b.buffer);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) b[o + i] = s.charCodeAt(i); };
  w(0, 'RIFF'); dv.setUint32(4, 36 + n, true); w(8, 'WAVE'); w(12, 'fmt ');
  dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, 8000, true); dv.setUint32(28, 8000, true); dv.setUint16(32, 1, true);
  dv.setUint16(34, 8, true); w(36, 'data'); dv.setUint32(40, n, true);
  b.fill(128, 44);
  silentUrl = URL.createObjectURL(new Blob([b], { type: 'audio/wav' }));
  return silentUrl;
}

/** Mic-off (the Mute button while the microphone is live). */
const MIC_OFF_ICON = [
  'M2 2l20 20',
  'M18.89 13.23A7 7 0 0 0 19 12v-2',
  'M5 10v2a7 7 0 0 0 12 5',
  'M15 9.34V5a3 3 0 0 0-5.68-1.33',
  'M9 9v3a3 3 0 0 0 5.12 2.12',
  'M12 19v3',
];
/** A keyboard (the Type button). */
const KEYBOARD_ICON = [
  'M4 6h16a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2Z',
  'M6 10h.01', 'M10 10h.01', 'M14 10h.01', 'M18 10h.01', 'M6 14h.01', 'M18 14h.01', 'M9 14h6',
];
/** A handset; voice.css turns it face down for Hang up. */
const HANDSET_ICON = [
  'M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92Z',
];
/** Try again (a stopped chat). */
const RETRY_ICON = ['M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8', 'M21 3v5h-5'];
/** Send (the text box). */
const SEND_ICON = ['M22 2 11 13', 'M22 2 15 22l-4-9-9-4Z'];
/** Output (where replies play): a speaker, a headset, Bluetooth — by route. */
const SPEAKER_ICON = ['M11 5 6 9H2v6h4l5 4V5Z', 'M15.5 8.5a5 5 0 0 1 0 7', 'M19 5a9 9 0 0 1 0 14'];
const HEADSET_ICON = ['M3 14v-3a9 9 0 0 1 18 0v3', 'M21 15v2a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3Z', 'M3 15v2a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3Z'];
const BLUETOOTH_ICON = ['m7 7 10 10-5 5V2l5 5L7 17'];
const ROUTE_ICONS = { speaker: SPEAKER_ICON, headset: HEADSET_ICON, bluetooth: BLUETOOTH_ICON, default: SPEAKER_ICON };

/**
 * The kind of output a device label names: 'bluetooth', 'headset' or 'speaker'
 * (the fallback). Labels are the OS's own words ("AirPods Pro", "Headphones
 * (Bluetooth)", "Speakers (Realtek)"), so this is a word hunt, kept short.
 */
export function outputRoute(label) {
  const l = String(label || '').toLowerCase();
  if (/bluetooth|\bbt\b|airpods|buds|wh-|wf-/.test(l)) return 'bluetooth';
  if (/headset|headphone|earphone|earbud|wired|jack/.test(l)) return 'headset';
  return 'speaker';
}

/**
 * The Output button on the call screen (1.3.0): where replies play, picked like
 * a phone's call-screen route — a speaker, a headset, a Bluetooth device. Only
 * browsers whose AudioContext has setSinkId can do this (desktop Chrome and
 * Edge); on a phone the OS routes audio itself and the button stays hidden.
 * `options` are deviceOptions('audiooutput') rows; `sinkId` is the saved one.
 */
export function outputMenuModel(options, sinkId) {
  const rows = (options || []).map((o) => ({
    value: o.value, label: o.label, route: o.value ? outputRoute(o.label) : 'default',
    current: (sinkId || '') === (o.value || ''),
  }));
  if (rows.length && !rows.some((r) => r.current)) rows[0].current = true;   // a remembered device that is gone
  return rows;
}
const reducedMotion = () => !!(globalThis.matchMedia && globalThis.matchMedia('(prefers-reduced-motion: reduce)').matches);

/** A round call-screen button: an icon in a circle with a word under it. */
function roadButton(cls) {
  return el('button', { type: 'button', class: `drive-road ${cls}` }, [
    el('span', { class: 'drive-road-icon', 'aria-hidden': 'true' }),
    el('span', { class: 'drive-road-label' }),
  ]);
}
function setRoad(btn, paths, label) {
  btn.firstChild.replaceChildren(railIcon(paths));
  btn.lastChild.textContent = label;
}

/**
 * Open the voice chat. MUST be called from inside the tap's click handler:
 * the silent keep-alive <audio> and the AudioContext are started before the
 * first await, while the browser still counts this as a user gesture.
 */
export function openDrive(thread, ctx) {
  if (active) return active;
  if (opening) return null;
  opening = true;

  const bot = ctx.getBot(thread.bot_id);
  let settings = loadDriveSettings(globalThis.localStorage);
  // First-run hint (L12) — counted before the settings listener exists.
  const tipKey = firstRunTipKey(settings.firstRunOpens, language(), { bargeIn: settings.bargeIn !== false });
  if (settings.firstRunOpens < FIRST_RUN_OPENS) {
    settings = saveDriveSettings(globalThis.localStorage, { firstRunOpens: settings.firstRunOpens + 1 });
  }

  // ---- inside the gesture: audio session, keep-alive, context ----------
  try { if (navigator.audioSession) navigator.audioSession.type = 'play-and-record'; } catch { /* Safari 17+ only */ }
  let keepAlive = null;
  try {
    keepAlive = new Audio(silentWavUrl());
    keepAlive.loop = true;
    keepAlive.volume = 0.0001;
    keepAlive.play().catch(() => {});
  } catch { keepAlive = null; }
  const AC = globalThis.AudioContext || globalThis.webkitAudioContext;
  const ac = new AC();
  ac.resume().catch(() => {});
  if (settings.sinkId && typeof ac.setSinkId === 'function') ac.setSinkId(settings.sinkId).catch(() => {});
  // Reply audio (player worklet) → duck gain → out. The ring's speaking level
  // comes from the player's own `level` reports.
  let duckGain = null;
  try {
    duckGain = ac.createGain();
    duckGain.connect(ac.destination);
  } catch { duckGain = null; }

  // ---- DOM --------------------------------------------------------------
  const avatar = (ctx.getAvatar && ctx.getAvatar(thread, bot)) || {};
  // The dialog's name says what it is, not only whose (FR-ux L8: "Voice Test, dialog").
  const root = el('div', {
    class: 'drive-overlay', role: 'dialog', 'aria-modal': 'true',
    'aria-label': t('drive.dialog_label', { bot: bot ? bot.name : '' }),
    dataset: { state: 'connecting' },
  });
  const name = el('div', { class: 'drive-name', id: 'drive-name', dir: 'auto', text: bot ? bot.name : t('drive.title') });
  const sub = el('div', { class: 'drive-sub', dir: 'auto', text: (thread.title || '').trim() });
  const gear = el('button', {
    type: 'button', class: 'icon-btn ghost drive-gear', title: t('drive.sheet_title'),
    'aria-label': t('drive.sheet_title'), 'aria-haspopup': 'dialog',
  }, [railIcon(iconPaths('gear'))]);
  const head = el('header', { class: 'drive-head' }, [
    el('div', { class: 'drive-id' }, [name, sub.textContent ? sub : null]), gear,
  ]);

  const letter = avatar.letter || (bot && bot.name ? bot.name[0] : '•');
  const face = avatar.src
    ? el('img', { class: 'drive-face', src: avatar.src, alt: '' })
    : el('span', { class: 'drive-face drive-letter', 'aria-hidden': 'true', text: letter });
  if (face.tagName === 'IMG') {
    face.addEventListener('error', () => {
      face.replaceWith(el('span', { class: 'drive-face drive-letter', 'aria-hidden': 'true', text: letter }));
    }, { once: true });
  }
  const ring = el('button', { type: 'button', class: 'drive-ring' }, [face, el('span', { class: 'drive-dot', 'aria-hidden': 'true' })]);
  const stateWord = el('div', { class: 'drive-state', 'aria-hidden': 'true' });
  const hint = el('div', { class: 'drive-hint', 'aria-hidden': 'true' });
  const tip = el('div', { class: 'drive-tip', hidden: '' });
  const notice = el('div', { class: 'drive-notice', role: 'alert', hidden: '' });
  // A standing info line while the link is degraded (§7.4); errors keep `notice`.
  const linkLine = el('div', { class: 'drive-notice drive-link', role: 'status', hidden: '', dataset: { kind: 'info' } });
  const timings = el('div', { class: 'drive-timings', 'aria-hidden': 'true', hidden: '' });
  // Undo lives under the state word, not in the captions: with Captions off it
  // must still be seen and tapped (FR-ux M3: it was announced but 0×0).
  const undoBtn = el('button', { type: 'button', class: 'btn-secondary drive-undo', hidden: '' });
  const hero = el('div', { class: 'drive-hero' }, [ring, stateWord, hint, tip, notice, linkLine, undoBtn, timings]);

  const you = el('p', { class: 'drive-you', dir: 'auto' });
  const them = el('p', { class: 'drive-them', dir: 'auto' });
  const talk = el('section', { class: 'drive-talk', 'aria-label': t('drive.transcript') }, [you, them]);

  const pauseBtn = roadButton('drive-pause');
  const typeBtn = roadButton('drive-type');
  const endBtn = roadButton('drive-end');
  const outputBtn = roadButton('drive-output');
  // The text box (1.2.0): shown by the Type button, or from the start with "Type
  // instead of talking". A message goes over the voice socket as `typed` and is
  // answered like a spoken one (the reply is spoken, Undo works). Enter sends,
  // Shift+Enter is a new line; an IME's Enter (Japanese) only commits the text.
  const composeInput = el('textarea', {
    class: 'drive-compose-input', rows: '1', placeholder: t('drive.compose_placeholder'),
    'aria-label': t('drive.compose_placeholder'), enterkeyhint: 'send', autocomplete: 'off',
  });
  const composeSend = el('button', { type: 'submit', class: 'btn-primary drive-compose-send', 'aria-label': t('drive.send') },
    [railIcon(SEND_ICON)]);
  const compose = el('form', { class: 'drive-compose', hidden: '' }, [composeInput, composeSend]);
  const bar = el('div', { class: 'drive-bar' }, [compose, el('div', { class: 'drive-buttons' }, [pauseBtn, typeBtn, outputBtn, endBtn])]);
  const status = el('div', { class: 'sr-only', role: 'status', 'aria-live': 'polite' });

  const col = el('div', { class: 'drive-col' }, [head, hero, talk, bar]);
  root.append(col, status);
  document.body.append(root);
  document.body.classList.add('drive-open');
  const inertToken = acquireInert();
  // The first-run tip stands until the first transcript, but never next to a
  // connection or error state, where "talk over me" would mislead (V2).
  let tipOn = !!tipKey;
  if (tipKey) { tip.textContent = t(tipKey); tip.hidden = false; }

  // ---- state -------------------------------------------------------------
  const undo = createUndoWindow();
  const heard = createHeardTracker();
  let stream = null;
  let node = null;
  let micQueue = [];             // mic worklet messages sent before the worklet exists
  let ws = null;
  let ready = false;
  let everReady = false;
  let serverProto = 2;           // ready.proto (absent = v2)
  let replayLast = false;        // the server says it answers `replay_last` (Amendment T7a-6)
  let backoff = 0;
  let reconnectTimer = null;
  let offlineSince = 0;
  let offlineTicker = null;
  let closed = false;
  let listening = !settings.typedInput;   // the listener wants the mic on ("Type instead of talking" opens muted)
  let pausedByHide = false;
  let wakeLock = null;
  let state = 'connecting';
  let stopReason = null;
  let thinkingAt = 0;
  let thinkingTicker = null;
  let undoTicker = null;
  let noticeTimer = null;
  let level = 0;
  let sheet = null;
  let popHistory = null;
  let gotIt = false;             // "Got it — go on?" (eot_soft with a long hold)
  let stoppedAt = 0;             // a reply was stopped at this time (tap = continue for 3 s)
  let linkForced = null;         // active.setLink() (automation): {degraded}
  let link = { degraded: false, notice: null, path: 'server' };
  let ducked = false;            // the reply is held: the listener may be talking (§4.6)
  let offlineSaid = false;       // "No signal. I'll send that…" shown in this offline spell
  let replyStarted = false;      // the first reply sentence of this turn arrived
  const sentences = new Map();   // engine sentence key (`sid:seq`) → <span> in the reply line; cleared per turn
  const spanText = new WeakMap(); // <span> → the sentence text it shows
  let nowKey = null;             // the sentence highlighted as playing
  let followUpSent = false;       // this session was sent a boolean `follow_up`
  let micLive = false;           // the mic worklet is connected and capturing
  let greetPending = false;      // `ready` came; the tone waits for the mic
  let supportDone = false;       // the WebCodecs check (codec lists) has answered
  const streamId = newStreamId();
  const replyPlaying = () => audio.replyPlaying;
  // A reply is under way: audio playing here, or the server still speaking
  // (between chunks, or with the mic paused) — the picture tap interrupts it.
  const replyOn = () => replyPlaying() || state === 'speaking';
  // "Tap to continue" only where the server can do it: `replay_last` is optional
  // (C.3), and a server without it ignores the frame (review-T7b M1).
  const resumable = () => replayLast && stoppedAt > 0 && Date.now() - stoppedAt < RESUME_AFTER_STOP_MS;
  const hb = createHeartbeat({ send: (f) => send(f), onDead: () => dropSocket() });
  const eotTone = createEotTone({ play: () => earcon('eot') });
  const quiet = createQuietTimer({
    minutes: settings.hangupQuietMin,
    onCheck: () => send({ type: 'quiet_check' }),
    onHangup: () => close(),
  });
  // With the microphone paused nobody can answer "Still there?" by voice, so
  // the timer hangs up without asking (review L5).
  const quietTicker = setInterval(() => { if (!closed && ready) quiet.tick({ canAsk: listening }); }, 1000);
  // DisPatch's idle auto-lock counts touches only: an open chat with the mic on is
  // activity while this page is visible (FR-works 1).
  const visible = () => !globalThis.document || document.visibilityState !== 'hidden';
  const activityTicker = setInterval(() => {
    if (!closed && ready && listening && visible()) shellActivity();
  }, ACTIVITY_EVERY_MS);

  const can = () => !closed;

  function send(obj) {
    if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
  }
  function sendBinary(buf) {
    if (ws && ws.readyState === 1) ws.send(buf);
  }
  function micPost(m) {
    if (node) { try { node.port.postMessage(m); } catch { /* gone */ } } else micQueue.push(m);
  }

  // ---- the audio & link engine (drive-audio.js) ----------------------------
  let bluetooth = false;
  try { bluetooth = isBluetoothOutput ? isBluetoothOutput({ outputLatency: ac.outputLatency }) : false; } catch { bluetooth = false; }
  const audio = createDriveAudio({
    send, sendBinary,
    isOpen: () => !!(ws && ws.readyState === 1),
    micPost,
    setDuckGain: (v, ms) => setDuck(v, ms),
    settings: () => settings,
    lang: () => language(),
    android: isAndroid(),
    bluetooth,
    effectiveType: (navigator.connection && navigator.connection.effectiveType) || '',
    latency: () => ({ outputLatency: ac.outputLatency, baseLatency: ac.baseLatency }),
    stream: streamId,
    ui: {
      onSay: (m) => onSayUi(m),
      onSentence: (x) => addSentence(x),
      onNow: (key) => markNow(key),
      onPlaying: () => { if (!closed) paintIfChanged(); },
      onActivity: () => quiet.activity(),
      onLevel: (v) => { if (replyOn() && !reducedMotion()) setLevel(Math.min(1, v * 4)); },
      onDuck: (on) => { ducked = !!on; if (!closed) paint(); },
      // A stopped or locked chat has no link to report: a late monitor event must not
      // bring "Low signal" back onto that screen (audit M3).
      onLink: (l) => { if (state === 'stopped' || state === 'locked') return; link = l; if (!closed) paintLink(); },
      onCutOff: () => showNotice(t('drive.cut_off'), { kind: 'info', ms: 6000 }),
      onSpeech: (on) => {
        if (on && (state === 'offline' || state === 'waiting') && !offlineSaid) {
          offlineSaid = true;
          showNotice(t('drive.offline_said'), { kind: 'info', ms: 6000 });
        }
      },
      onPhoneFailed: () => showNotice(t('drive.link_text'), { kind: 'info', ms: 6000 }),
    },
  });
  const supportP = Promise.race([
    audio.probeSupport().catch(() => null),
    new Promise((r) => setTimeout(r, SUPPORT_WAIT_MS)),
  ]).then(() => { supportDone = true; });
  phoneVoices().then((v) => { if (!closed) audio.setVoices(v); }).catch(() => {});
  const audioTicker = setInterval(() => { if (!closed) { audio.tick(); paintClientLink(); } }, AUDIO_TICK_MS);

  function sendStart() {
    const frame = startPayload(settings, heard.after, heard.ids, audio.startExtra());
    // Back to "the bot's default" after this session was told a choice: absent
    // keys keep their value on the server, so say so with null (Amendment T7a-7).
    if (frame.follow_up === undefined && followUpSent) frame.follow_up = null;
    followUpSent = typeof frame.follow_up === 'boolean';
    send(frame);
    // The Voice lab's Reset is one-shot: once the server has been told, clear it.
    if (frame.rt && frame.rt.learned_reset) settings = saveDriveSettings(globalThis.localStorage, { learnedReset: false });
  }
  function earcon(kind) {
    if (settings.sounds) { try { playEarcon(ac, kind); } catch { /* closed context */ } }
  }
  let localWord = null;           // the word setListening() announced itself (L10)
  let lastSaid = { text: '', at: 0 };
  function announce(text) {
    status.textContent = '';
    if (text) {
      lastSaid = { text, at: Date.now() };
      setTimeout(() => { status.textContent = text; }, 30);
    }
  }

  // ---- notice strip ------------------------------------------------------
  function showNotice(text, { kind = 'error', ms = 0 } = {}) {
    clearTimeout(noticeTimer);
    if (!text) { notice.hidden = true; notice.textContent = ''; return; }
    notice.dataset.kind = kind;
    notice.textContent = text;
    notice.hidden = false;
    if (ms) noticeTimer = setTimeout(() => { notice.hidden = true; }, ms);
  }
  function paintLink() {
    const key = linkForced ? linkNoticeKey(!!linkForced.degraded, settings.slowLink, audio.phoneOk)
      : linkStateNoticeKey(link);
    linkLine.textContent = key ? t(key) : '';
    linkLine.hidden = !key;
    syncTalk();
  }
  /** The chat ended (stopped, locked): nothing is being read by anything any more. */
  function resetLink() {
    link = { degraded: false, notice: null, path: 'server' };
    paintLink();
  }

  // ---- captions: where they show, and following the speech (FR-ux H1, M2) --
  /**
   * The captions area shows with Captions on, and ALSO whenever replies take the
   * text path (a slow link on "Text only", no phone voice): then the text on this
   * screen is the reply, whatever the Captions setting says (FR-ux M2). The area
   * keeps a few lines' room from the start and grows with its text (voice.css), so
   * the picture stays put at the first word; `drive-captioned` only marks that
   * there is text.
   */
  function syncTalk() {
    const textPath = link.path === 'text' || !!them.querySelector('.drive-text-path');
    talk.hidden = !settings.captions && !textPath;
    const has = !talk.hidden && (!!you.textContent.trim() || them.childElementCount > 0);
    col.classList.toggle('drive-captioned', has);
  }
  let followPausedUntil = 0;
  const pauseFollow = () => { followPausedUntil = Date.now() + FOLLOW_PAUSE_MS; };
  for (const ev of ['wheel', 'touchmove', 'pointerdown']) talk.addEventListener(ev, pauseFollow, { passive: true });
  talk.addEventListener('scroll', () => talk.classList.toggle('drive-scrolled', talk.scrollTop > 2), { passive: true });
  /** Keep the sentence being spoken in view: its top in the upper part of the area. */
  function follow(span) {
    if (!span || talk.hidden || Date.now() < followPausedUntil || typeof talk.getBoundingClientRect !== 'function') return;
    const box = talk.getBoundingClientRect();
    const r = span.getBoundingClientRect();
    const top = r.top - box.top;
    const h = talk.clientHeight || box.height;
    if (!(h > 0) || (top >= 0 && r.bottom <= box.bottom && top <= h * 0.6)) return;
    const to = Math.max(0, talk.scrollTop + top - h / 3);
    try { talk.scrollTo({ top: to, behavior: reducedMotion() ? 'auto' : 'smooth' }); } catch { talk.scrollTop = to; }
  }

  // ---- state painting ----------------------------------------------------
  let paintedKind;
  function paintIfChanged() {
    const k = audio.playingKind;
    if (k !== paintedKind) { paintedKind = k; paint(); }
  }
  function paint() {
    const shown = displayState(state, { listening, playing: replyOn(), ducked });
    const s = shown.state;
    root.dataset.state = s;
    let word;
    if (shown.wordKey) word = t(shown.wordKey);
    else if (gotIt && (s === 'hearing' || s === 'transcribing' || s === 'listening')) word = t('drive.state_gotit_goon');
    else if (s === 'thinking') {
      const ms = Date.now() - thinkingAt;
      word = thinkingPhase(ms) === 'plain' ? t('drive.state_thinking')
        : t('drive.state_thinking_elapsed', { s: Math.floor(ms / 1000) });
    } else {
      word = t(`drive.state_${s}`);
    }
    stateWord.textContent = word;
    const opts = {
      playing: replyOn(), thinkingMs: s === 'thinking' ? Date.now() - thinkingAt : 0,
      autoListen: settings.autoListen, resumable: resumable(), sounds: settings.sounds !== false,
    };
    const hk = hintKey(s, opts);
    let hintText = hk ? t(hk) : '';
    if (s === 'offline') {
      const left = reconnectTimer ? Math.max(1, Math.ceil(backoffLeft() / 1000)) : 0;
      hintText = left ? t('drive.hint_offline', { s: left }) : t('drive.hint_reconnecting');
    }
    // The text box is out and nothing plays: say so, instead of "tap to unmute".
    if (composeOpen && s === 'paused' && !replyOn()) hintText = t('drive.hint_type');
    hint.textContent = hintText;
    tip.hidden = !tipOn || TIPLESS.has(s);
    ring.setAttribute('aria-label', hintText ? `${word}. ${hintText}` : word);
    // aria-disabled, not `disabled`: the picture keeps focus (it is where focus
    // starts when the voice chat opens, while it is still connecting).
    ring.setAttribute('aria-disabled', String(tapAction(s, opts) === 'none'));
    paintBar();
  }
  let atScheduled = 0;
  function backoffLeft() { return atScheduled - Date.now(); }

  function paintBar() {
    if (state === 'stopped') {
      setRoad(pauseBtn, RETRY_ICON, t('drive.try_again'));
      pauseBtn.hidden = stopReason === 'refused';
      pauseBtn.dataset.muted = '0';
      typeBtn.hidden = true;
      outputBtn.hidden = true;
      setRoad(endBtn, iconPaths('back'), t('drive.back_to_chat'));
      endBtn.classList.add('drive-end-plain');
      return;
    }
    endBtn.classList.remove('drive-end-plain');
    pauseBtn.hidden = false;
    // The label says what a tap does ("Mute" / "Unmute"), so no aria-pressed: a
    // swapped label plus a pressed state read "Unmute, toggle button, pressed" (FR-ux M5).
    // `data-muted` only lights the circle while the microphone is off.
    setRoad(pauseBtn, listening ? MIC_OFF_ICON : MIC_ICON, listening ? t('drive.pause') : t('drive.resume'));
    pauseBtn.dataset.muted = listening ? '0' : '1';
    pauseBtn.removeAttribute('aria-pressed');
    typeBtn.hidden = false;
    setRoad(typeBtn, KEYBOARD_ICON, t('drive.type_input'));
    typeBtn.setAttribute('aria-label', t('drive.type_input_aria'));
    typeBtn.setAttribute('aria-pressed', String(composeOpen));
    setRoad(endBtn, HANDSET_ICON, t('drive.end'));
    paintOutput();
  }

  const ANNOUNCE = new Set(['listening', 'paused', 'offline', 'waiting', 'stopped', 'locked']);
  function setState(s) {
    const prev = state;
    state = s;
    if (s === 'thinking' && prev !== 'thinking') {
      thinkingAt = Date.now();
      clearInterval(thinkingTicker);
      thinkingTicker = setInterval(() => { if (state === 'thinking') paint(); }, 1000);
    } else if (s !== 'thinking') {
      clearInterval(thinkingTicker); thinkingTicker = null;
    }
    if (s !== 'offline' && s !== 'waiting') { clearInterval(offlineTicker); offlineTicker = null; }
    if (s !== 'hearing' && s !== 'transcribing') gotIt = false;
    // Only the listener answers a pending "Still there?": the server reports
    // `speaking` while it says it (review M3).
    if (s === 'hearing') quiet.heard();
    else if (s === 'speaking') quiet.activity();
    paint();
    if (prev !== s && ANNOUNCE.has(s)) {
      // Mute/Unmute was already announced on the tap; the server's echo
      // (`state: paused`) must not say it a second time.
      if (stateWord.textContent !== localWord) announce(stateWord.textContent);
      localWord = null;
    }
    // Push-to-talk: a finished reply leaves the mic paused.
    if (prev === 'speaking' && s === 'listening' && !settings.autoListen && listening) setListening(false);
  }

  function stop(reason) {
    stopReason = reason;
    hb.stop();
    eotTone.cancel();
    clearTimeout(reconnectTimer); reconnectTimer = null;
    if (ws) { const old = ws; ws = null; old.onclose = null; old.onmessage = null; try { old.close(); } catch { /* gone */ } }
    ready = false;
    // A stopped voice chat must not keep the microphone open.
    stopMic();
    const msg = {
      refused: t('drive.err_refused'),
      unavailable: t('drive.err_unavailable'),
      busy: t('drive.err_busy'),
      voice_off: t('drive.err_voice_off'),
      internal: t('drive.err_internal'),
      network: t('drive.err_network'),
      mic_denied: t('drive.err_mic_denied'),
      mic_missing: t('drive.err_mic_missing'),
      https: t('drive.err_https'),
    }[reason] || t('drive.err_internal');
    audio.stopHere();
    earcon('err');
    setState('stopped');
    resetLink();
    showNotice(msg);
  }

  // ---- timings (Voice lab → "Show timings on the voice chat screen") -----
  const lastTimes = { heard: null, first: null, link: null };
  let serverLink = false;        // the server reported its own link line (v3 metrics)
  function paintTimings(m) {
    if (typeof m.eot_to_transcript_ms === 'number') { lastTimes.heard = m.eot_to_transcript_ms; lastTimes.first = null; }
    if (typeof m.eot_to_first_audio_ms === 'number') lastTimes.first = m.eot_to_first_audio_ms;
    if (m.link && typeof m.link === 'object') { lastTimes.link = m.link; serverLink = true; }
    renderTimings();
  }
  function renderTimings() {
    const line = timingsLine(lastTimes);
    timings.textContent = line;
    timings.hidden = !settings.showTimings || !line;
  }
  /**
   * Without a server link line: this device's own RTT, rung and which voice
   * speaks — on the timings line and in the Voice lab's "Last voice chat"
   * Connection row (the remembered metrics).
   */
  let clientLinkAt = 0;
  function paintClientLink() {
    if (serverLink || audio.proto < 3) return;
    const n = Date.now();
    if (n - clientLinkAt < 2000) return;
    clientLinkAt = n;
    const st = audio.stats();
    if (typeof st.rtt_ms !== 'number') return;
    const clientLink = { rtt_ms: st.rtt_ms, down_kbps: st.down_kbps, mode: st.path === 'server' ? 'audio' : st.path };
    lastTimes.link = clientLink;
    noteMetrics({ link: clientLink });
    renderTimings();
  }

  // ---- duck: the listener may be talking over the reply ------------------
  function setDuck(value, ms = 40) {
    if (!duckGain) return;
    try {
      const g = duckGain.gain;
      g.cancelScheduledValues(ac.currentTime);
      g.setTargetAtTime(value, ac.currentTime, Math.max(0.005, ms / 3000));
    } catch { /* closed context */ }
  }

  // ---- ring level (the one moving thing) --------------------------------
  function setLevel(v) {
    level = Math.max(0, Math.min(1, v));
    ring.style.setProperty('--lvl', level.toFixed(3));
  }

  // ---- mic ---------------------------------------------------------------
  async function startMic() {
    const want = { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true };
    if (settings.micId) want.deviceId = { exact: settings.micId };
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: want });
    } catch (err) {
      // A remembered mic that is gone (the car is not connected today) falls
      // back to the default instead of failing the voice chat.
      if (settings.micId && err && (err.name === 'OverconstrainedError' || err.name === 'NotFoundError')) {
        delete want.deviceId;
        stream = await navigator.mediaDevices.getUserMedia({ audio: want });
      } else {
        throw err;
      }
    }
    if (closed || state === 'stopped') { stream.getTracks().forEach((tr) => tr.stop()); stream = null; return; }
    if (!node) {
      await ac.audioWorklet.addModule(assetUrl('voice-worklet.js'));
      if (closed) return;
      const n = new AudioWorkletNode(ac, 'drive-mic');
      n.port.onmessage = (e) => {
        const d = e.data;
        if (d && typeof d.level === 'number') {
          if ((state === 'hearing' || state === 'listening') && !reducedMotion()) setLevel(Math.min(1, d.level * 6));
          return;
        }
        // Frames (with their 16 kHz offsets), gate and speech edges: the engine's.
        if (listening || !(d instanceof ArrayBuffer || (d && d.buf))) audio.micMessage(d);
      };
      node = n;
      // What the engine said before the worklet existed (a `ready` gate, hold, bias).
      const q = micQueue;
      micQueue = [];
      for (const m of q) { try { node.port.postMessage(m); } catch { /* gone */ } }
    }
    const srcNode = ac.createMediaStreamSource(stream);
    srcNode.connect(node);
    node._src = srcNode;
    node.port.postMessage({ enabled: listening });
    // Not connected to the destination: the mic must never play back.
    micLive = true;
    maybeGreet();
  }
  /** `ready` + a capturing mic: the ready tone, and Listening (or Paused). */
  function maybeGreet() {
    if (!greetPending || !ready || !micLive || closed) return;
    greetPending = false;
    earcon('ready');
    quiet.heard();     // a fresh session: start the idle clock again
    if (listening) { if (state === 'connecting' || state === 'offline' || state === 'waiting') setState('listening'); }
    else setState('paused');
    // The Output word names a remembered device from the start (the labels are
    // readable now that the microphone permission is in), not only after a tap.
    if (outputPossible && settings.sinkId) refreshOutputs().catch(() => {});
  }
  function stopMic() {
    try { stream && stream.getTracks().forEach((tr) => tr.stop()); } catch { /* gone */ }
    try { node && node._src && node._src.disconnect(); } catch { /* gone */ }
    stream = null;
    micLive = false;
  }
  /** The reply player (C-§E.2): the worklet, or the buffer scheduler where AudioWorklet is missing. */
  async function startPlayer() {
    try {
      await ac.audioWorklet.addModule(assetUrl('voice-player-worklet.js'));
      if (closed) return;
      const pn = new AudioWorkletNode(ac, 'drive-player', { numberOfInputs: 0, outputChannelCount: [1] });
      pn.connect(duckGain || ac.destination);
      audio.setPlayer(pn.port);
    } catch {
      if (closed) return;
      audio.setPlayer(createBufferPlayerClient(ac, duckGain || ac.destination, audio.playerCallbacks));
    }
  }
  /**
   * Why the microphone could not start. Only a refusal is "allow the
   * microphone"; a missing device is its own line; anything else (the audio
   * worklet failed to load, a stale add-on asset) is not the listener's to fix
   * in the browser's site settings, so it is an internal error with a retry.
   */
  function micError(err) {
    const n = err && err.name;
    if (!globalThis.isSecureContext) return 'https';
    if (n === 'NotFoundError' || n === 'OverconstrainedError') return 'mic_missing';
    if (n === 'NotAllowedError' || n === 'SecurityError' || n === 'PermissionDeniedError') return 'mic_denied';
    try { console.warn('voice chat: the microphone path failed to start', err); } catch { /* no console */ }
    return 'internal';
  }

  // ---- socket ------------------------------------------------------------
  function connect() {
    if (closed) return;
    clearTimeout(reconnectTimer); reconnectTimer = null;
    // The codec lists go in the URL (C.1): the first socket waits for the WebCodecs check.
    if (!supportDone) { supportP.then(() => { if (!closed && !ws && !reconnectTimer) connect(); }); return; }
    ready = false;
    followUpSent = false;          // a new socket is a new session
    const sock = new WebSocket(voiceUrl(location, thread.id, driveDeviceId(), { v3: true, stream: streamId, ...audio.codecLists() }));
    ws = sock;
    sock.binaryType = 'arraybuffer';
    sock.onopen = () => {
      if (ws !== sock) return;
      backoff = 0;
      audio.opened();
      hb.start({ armed: false });   // dead-man timer counts from `ready` (review L2)
      if (listening) sendStart(); else { sendStart(); send({ type: 'pause' }); }
    };
    sock.onmessage = (e) => {
      if (ws !== sock) return;
      hb.alive();
      if (typeof e.data !== 'string') { audio.binary(e.data); return; }
      let m; try { m = JSON.parse(e.data); } catch { return; }
      if (m && ACTIVITY_FRAMES.has(m.type) && visible()) shellActivity();
      const wasPlaying = replyPlaying();
      audio.frame(m);
      onFrame(m, { wasPlaying });
    };
    sock.onclose = (e) => { if (ws === sock) onSocketClosed(e.code, e.reason); };
  }

  /** A heartbeat miss: the socket is half-open. Abandon it and reconnect now. */
  function dropSocket() {
    const old = ws;
    ws = null;
    if (old) { old.onclose = null; old.onmessage = null; try { old.close(); } catch { /* gone */ } }
    onSocketClosed(1006);
  }

  function onSocketClosed(code, why = '') {
    hb.stop();
    ready = false;
    audio.closed();
    eotTone.cancel();
    if (closed || state === 'stopped' || state === 'locked') return;
    const d = reconnectDecision(code, why);
    if (d.reason === 'replaced') {
      close({ quiet: true });
      if (ctx.toast) ctx.toast(t('drive.replaced'));
      return;
    }
    // The server ended a chat nobody spoke in for a long time: hang up the way
    // "Hang up when quiet" does (the toast says "Voice chat ended.").
    if (d.reason === 'quiet') { close(); return; }
    if (!d.retry) { stop(d.reason); return; }
    if (d.reason === 'internal') showNotice(t('drive.err_internal_retry'), { ms: 8000 });
    if (!offlineSince) offlineSince = Date.now();
    const away = Date.now() - offlineSince;
    if (away >= GIVE_UP_MS) { stop('network'); return; }
    const waiting = away >= WAITING_AFTER_MS;
    scheduleReconnect(nextBackoff(backoff, { waiting }));
    // v3: the player keeps playing what arrived complete (T6-2); the engine
    // already silenced a v2 reply.
    setState(waiting ? 'waiting' : 'offline');
    if (!offlineTicker) offlineTicker = setInterval(() => { if (state === 'offline' || state === 'waiting') paint(); }, 1000);
  }
  function scheduleReconnect(ms) {
    clearTimeout(reconnectTimer);
    backoff = ms;
    atScheduled = Date.now() + ms;
    reconnectTimer = setTimeout(() => { reconnectTimer = null; connect(); paint(); }, ms);
  }
  /** Back online or visible again: try now instead of waiting out the backoff. */
  function retryNow() {
    if (closed || (state !== 'offline' && state !== 'waiting') || !reconnectTimer) return;
    scheduleReconnect(0);
  }

  // ---- incoming -----------------------------------------------------------
  /**
   * A sentence was announced (`say`, or the v2 audio header): the reply has
   * started — the turn-end tone is dropped, Undo closes, the reply's id is
   * remembered for the reconnect `start`. Its caption waits until it PLAYS.
   */
  function onSayUi(m) {
    const kind = m.kind || 'reply';
    if (kind === 'reply') {
      if (m.message_id) heard.note(m.message_id);
      if (!undoBtn.hidden) hideUndo();
      if (!replyStarted) { replyStarted = true; eotTone.suppress(); }
      quiet.activity();
      return;
    }
    if (kind === 'ack') { eotTone.suppress(); return; }
    if (kind === 'notice' && m.text) showNotice(m.text, { kind: 'info', ms: 6000 });
  }
  /** A reply sentence started playing (or was spoken by the phone, or shown as text). */
  function addSentence({ key, text, role, path }) {
    // A sentence on the text path IS the reply: shown whatever Captions says (FR-ux M2).
    const textPath = path === 'text';
    if (!text || (!settings.captions && !textPath) || sentences.has(key)) return;
    // A reply resumed on a new socket (`after_partial`, §5.5) starts again with the
    // sentence the drop cut mid-way, under the new session's key: that sentence's
    // line is already on screen from the earlier session — it is the same line, not
    // a second one (only an earlier session's span qualifies: a reply may say the
    // same words twice on purpose).
    const sid = String(key).slice(0, String(key).lastIndexOf(':'));
    for (const [k, old] of sentences) {
      if (String(k).slice(0, String(k).lastIndexOf(':')) !== sid && spanText.get(old) === text) {
        sentences.set(key, old);
        return;
      }
    }
    const span = el('span', {
      class: 'drive-sentence' + (role === 'narrator' ? ' drive-narrator' : '') + (textPath ? ' drive-text-path' : ''),
      text: `${text} `,
    });
    spanText.set(span, text);
    them.append(span);
    sentences.set(key, span);
    syncTalk();
    // Shown as text (nothing will play it): it is "now" at once.
    if (textPath) markNow(key);
  }
  function markNow(key) {
    if (key === nowKey) return;
    const span = sentences.get(key);
    if (!span) return;
    nowKey = key;
    them.querySelectorAll('.drive-sentence.now').forEach((n) => n.classList.remove('now'));
    span.classList.add('now');
    follow(span);
  }

  /** Live captions while the listener talks (C.4 `partial`). */
  function onPartial(m) {
    quiet.heard();
    if (!settings.captions) return;
    const committed = String(m.committed || '');
    const tail = String(m.tail || '');
    you.classList.remove('drive-empty');
    you.replaceChildren(
      committed ? el('span', { class: 'drive-committed', text: committed }) : null,
      committed && tail ? ' ' : null,
      tail ? el('span', { class: 'drive-tail', text: tail }) : null,
    );
    syncTalk();
  }

  function onFrame(m, { wasPlaying = false } = {}) {
    switch (m.type) {
      case 'ready': {
        ready = true;
        offlineSince = 0;
        offlineSaid = false;
        serverProto = Number(m.proto) >= 3 ? 3 : 2;
        replayLast = serverProto >= 3 && (m.replay_last === true || !!(m.rt && m.rt.replay_last === true));
        // The engine keys sentences by (sid, seq) (C.5 #9, review L1): a new session's
        // seq 1 is a new caption. The spans already shown stay on screen.
        hb.arm();
        const learned = m.rt && m.rt.learned;
        if (learned && Number.isFinite(learned.probe_ms) && Number.isFinite(learned.hold_ms)) {
          saveDriveSettings(globalThis.localStorage, { learned: { probeMs: learned.probe_ms, holdMs: learned.hold_ms } });
        }
        const reconnected = everReady;
        everReady = true;
        if (stopReason || reconnected) { stopReason = null; showNotice(''); }
        // The tone means "talk now": it waits until the microphone is really
        // capturing too (a cold open on a slow link fetches the worklet after
        // the server is ready — words spoken before that were lost).
        greetPending = true;
        maybeGreet();
        break;
      }
      case 'pong': {
        const sentAt = typeof m.t === 'number' ? m.t : null;
        hb.pong(sentAt);
        audio.pong(m, sentAt !== null ? Date.now() - sentAt : NaN);
        break;
      }
      case 'state':
        // `listening` waits for the tone: the server ready AND the mic capturing.
        if ((!ready || greetPending) && m.state === 'listening') break;
        setState(m.state);
        break;
      case 'partial': onPartial(m); break;
      case 'eot_soft':
        gotIt = Number(m.hold_ms) >= GOTIT_MIN_HOLD_MS;
        paint();
        break;
      case 'eot':
        gotIt = false;
        replyStarted = false;
        eotTone.eot();
        paint();
        break;
      case 'ack': eotTone.suppress(); earcon('ack'); break;
      case 'transcript': onTranscript(m); break;
      case 'undone': onUndone(); break;
      case 'barge_in':
        // The engine flushed the player and sent `heard`; a reply that was playing
        // can be continued with a tap for 3 s.
        if (wasPlaying) stoppedAt = Date.now();
        ducked = false;
        paint();
        break;
      case 'turn_error':
        earcon('err');
        showNotice(t(turnErrorKey(m.reason)), { ms: 8000 });
        break;
      case 'metrics':
        noteMetrics(m);
        paintTimings(m);
        break;
      case 'locked':
        showNotice(t('drive.locked_notice'));
        setState('locked');
        resetLink();
        setTimeout(() => close({ quiet: true }), 2000);
        break;
      case 'error': {
        const key = errorFrameKey(m);
        if (!key) break;
        earcon('err');
        showNotice(t(key), { ms: 8000 });
        if (key === 'drive.err_undo_expired') hideUndo();
        break;
      }
      default: break;
    }
  }

  function onTranscript(m) {
    quiet.heard();
    gotIt = false;
    tipOn = false;
    tip.hidden = true;
    // `message_id` is the user row Undo would take back. null = heard but
    // nothing was sent; absent = a 2.1.0 server, which still accepts `undo`.
    const rowId = m.message_id === undefined ? undefined : (m.message_id || null);
    if (!settings.captions) { if (m.text && rowId !== null) openUndo(rowId); return; }
    if (!m.text) {
      you.textContent = t('drive.you_empty');
      you.classList.add('drive-empty');
      earcon('err');
      return;
    }
    you.classList.remove('drive-empty');
    you.textContent = t('drive.you_said', { text: m.text });
    if (!m.merged) { them.replaceChildren(); sentences.clear(); nowKey = null; followPausedUntil = 0; }
    try { talk.scrollTop = 0; } catch { /* detached */ }
    syncTalk();
    showNotice('');
    if (rowId !== null) openUndo(rowId);
  }

  let undoFor = null;
  function openUndo(rowId) {
    undoFor = rowId || null;
    undo.open();
    undoBtn.hidden = false;
    const tick = () => {
      if (!undo.canUndo()) { hideUndo(); return; }
      undoBtn.textContent = t('drive.undo_countdown', { s: undo.remaining() });
    };
    clearInterval(undoTicker);
    undoTicker = setInterval(tick, 250);
    tick();
    announce(t('drive.undo_available'));
  }
  function hideUndo() {
    undo.close();
    clearInterval(undoTicker); undoTicker = null;
    undoBtn.hidden = true;
  }
  function doUndo() {
    if (!undo.canUndo()) { hideUndo(); return; }
    const frame = { type: 'undo' };
    if (undoFor) frame.message_id = undoFor;
    const h = audio.heardNow();
    if (h && audio.proto >= 3) frame.heard = h;
    send(frame);
    hideUndo();
  }
  function onUndone() {
    hideUndo();
    // The server's barge_in (sent just before `undone`) already moved gen on and
    // flushed: stopping again here would drop the next reply (review-T7b H2).
    audio.undone();
    you.textContent = '';
    them.replaceChildren(); sentences.clear(); nowKey = null;
    syncTalk();
    showNotice(t('drive.undone'), { kind: 'info', ms: 4000 });
  }

  // ---- actions -------------------------------------------------------------
  /**
   * Mute = the microphone only (experience M7): a reply that is playing
   * finishes; the screen says "Muted — still speaking". Sends ONLY `pause`.
   */
  function setListening(on) {
    listening = on;
    if (node) node.port.postMessage({ enabled: on });
    audio.micPaused(!on);
    if (!on) {
      send({ type: 'pause' });
      paint();
    } else {
      // Back to talking: a text box opened by the Type button goes away; with
      // "Type instead of talking" on it stays (both ways of asking are wanted).
      if (composeOpen && !settings.typedInput) closeCompose();
      sendStart();
      if (ready && state === 'paused') setState('listening'); else paint();
    }
    // Announce here, not on the server's echo: that is a round trip later on a weak
    // link, and never on a server that does not echo (audit L10).
    const word = stateWord.textContent;
    if (!(lastSaid.text === word && Date.now() - lastSaid.at < 1500)) {
      announce(word);
      localWord = word;
    }
  }

  function interrupt() {
    stoppedAt = Date.now();
    const h = audio.stopHere();
    send(h ? { type: 'stop_playback', heard: h } : { type: 'stop_playback' });
    paint();
    setTimeout(() => { if (!closed) paint(); }, RESUME_AFTER_STOP_MS + 50);
  }

  function tap() {
    quiet.heard();       // a tap, a button, a key: the listener is there
    const shown = displayState(state, { listening, playing: replyOn() });
    const act = tapAction(shown.state, { playing: replyOn(), resumable: resumable() });
    if (act === 'interrupt') { interrupt(); return; }
    if (act === 'continue') { stoppedAt = 0; send({ type: 'replay_last' }); paint(); return; }
    if (act === 'pause') { setListening(false); return; }
    if (act === 'resume') { setListening(true); return; }
    // cancel → the server barges in (gen + 1) and abandons the turn.
    if (act === 'cancel') {
      const h = audio.stopHere();
      send(h ? { type: 'cancel', heard: h } : { type: 'cancel' });
      hideUndo();
      return;
    }
    if (act === 'retry') retry();
  }

  async function retry() {
    showNotice('');
    stopReason = null;
    offlineSince = 0;
    backoff = 0;
    setState('connecting');
    if (!stream) {
      try { await startMic(); } catch (err) { stop(micError(err)); return; }
    }
    connect();
  }

  // ---- the text box (Type) ---------------------------------------------------
  let composeOpen = false;
  /**
   * The on-screen keyboard: Android (with DisPatch's `interactive-widget=
   * resizes-content`) shrinks the layout viewport, iOS keeps it at full height and
   * moves only the visual viewport, which would leave the box under the keys.
   * While the text box is out the fixed overlay is sized to the visual viewport.
   */
  const fitViewport = (() => {
    const vv = globalThis.visualViewport;
    let on = false;
    const apply = () => {
      if (!on || !vv) return;
      root.style.height = `${Math.round(vv.height)}px`;
      root.style.top = `${Math.round(vv.offsetTop)}px`;
      root.style.bottom = 'auto';
    };
    return {
      start() { if (!vv || on) return; on = true; vv.addEventListener('resize', apply); vv.addEventListener('scroll', apply); apply(); },
      stop() {
        if (!on) return;
        on = false;
        if (vv) { vv.removeEventListener('resize', apply); vv.removeEventListener('scroll', apply); }
        root.style.height = ''; root.style.top = ''; root.style.bottom = '';
      },
    };
  })();
  function autoGrow() {
    composeInput.style.height = 'auto';
    composeInput.style.height = `${Math.min(composeInput.scrollHeight || 0, 120)}px`;
  }
  function openCompose({ focus = true } = {}) {
    if (!composeOpen) {
      composeOpen = true;
      compose.hidden = false;
      col.classList.add('drive-composing');
      fitViewport.start();
      paint();
    }
    if (focus) { try { composeInput.focus({ preventScroll: true }); } catch { composeInput.focus(); } }
  }
  function closeCompose() {
    if (!composeOpen) return;
    composeOpen = false;
    compose.hidden = true;
    col.classList.remove('drive-composing');
    fitViewport.stop();
    if (document.activeElement === composeInput) ring.focus();
    paint();
  }
  /** The Type button: mute the microphone if it is on, and show the box (again: hide it). */
  function toggleCompose() {
    quiet.heard();
    if (composeOpen && !settings.typedInput) { closeCompose(); return; }
    if (listening) setListening(false);
    openCompose();
  }
  function sendTyped() {
    const text = composeInput.value.trim();
    if (!text) return;
    // The server answers an over-long `typed` with a bad_frame and the words
    // would already be gone from the box: refuse here, keep them, say why.
    if (text.length > TYPED_MAX_CHARS) { showNotice(t('drive.compose_too_long', { n: TYPED_MAX_CHARS }), { kind: 'info', ms: 5000 }); return; }
    if (!(ws && ws.readyState === 1 && ready)) { showNotice(t('drive.compose_offline'), { kind: 'info', ms: 4000 }); return; }
    quiet.heard();
    tipOn = false; tip.hidden = true;
    send({ type: 'typed', text });
    composeInput.value = '';
    autoGrow();
    composeInput.focus();
  }

  // ---- output route (1.3.0) ---------------------------------------------------
  const outputPossible = canPickSpeaker();
  let outputRows = [];                    // the last device list, for the label
  let outputMenu = null;
  function currentOutput() {
    const rows = outputMenuModel(outputRows, settings.sinkId);
    return rows.find((r) => r.current) || { route: 'default', label: '' };
  }
  function paintOutput() {
    outputBtn.hidden = !outputPossible || state === 'stopped';
    if (outputBtn.hidden) return;
    const cur = currentOutput();
    const word = cur.route === 'default' ? t('drive.output') : t(`drive.output_${cur.route}`);
    setRoad(outputBtn, ROUTE_ICONS[cur.route] || SPEAKER_ICON, word);
    outputBtn.setAttribute('aria-label', cur.label ? t('drive.output_aria_now', { name: cur.label }) : t('drive.output_aria'));
    outputBtn.setAttribute('aria-haspopup', 'menu');
    outputBtn.setAttribute('aria-expanded', String(!!outputMenu));
  }
  /** The output devices as of now, and the button's word with them. */
  async function refreshOutputs() {
    try { outputRows = await deviceOptions('audiooutput'); } catch { outputRows = []; }
    if (!closed) paintOutput();
  }
  async function openOutputMenu() {
    if (outputMenu || !outputPossible) return;
    await refreshOutputs();
    if (closed || outputMenu) return;
    const rows = outputMenuModel(outputRows, settings.sinkId);
    const items = rows.map((r) => {
      const b = el('button', { type: 'button', role: 'menuitemradio', class: 'drive-output-item', 'aria-checked': String(r.current), dataset: { value: r.value } }, [
        railIcon(ROUTE_ICONS[r.route] || SPEAKER_ICON),
        el('span', { class: 'drive-output-label', text: r.label }),
        el('span', { class: 'drive-output-check', 'aria-hidden': 'true', text: r.current ? '✓' : '' }),
      ]);
      b.addEventListener('click', () => {
        settings = saveDriveSettings(globalThis.localStorage, { sinkId: r.value });   // onDriveSettings applies setSinkId
        closeOutputMenu();
        announce(t('drive.output_now', { name: r.label }));
      });
      return b;
    });
    outputMenu = el('div', { class: 'drive-output-menu', role: 'menu', 'aria-label': t('drive.output_aria') }, [
      el('div', { class: 'drive-output-head', text: t('drive.output_aria') }), ...items,
    ]);
    outputMenu.addEventListener('keydown', (e) => {
      const list = [...outputMenu.querySelectorAll('[role="menuitemradio"]')];
      const i = list.indexOf(document.activeElement);
      if (e.key === 'ArrowDown') { e.preventDefault(); list[(i + 1) % list.length]?.focus(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); list[(i - 1 + list.length) % list.length]?.focus(); }
      else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeOutputMenu(); }
    });
    bar.append(outputMenu);
    paintOutput();
    (items.find((b) => b.getAttribute('aria-checked') === 'true') || items[0])?.focus();
  }
  function closeOutputMenu() {
    if (!outputMenu) return;
    outputMenu.remove();
    outputMenu = null;
    paintOutput();
    outputBtn.focus();
  }
  function toggleOutputMenu() { quiet.heard(); if (outputMenu) closeOutputMenu(); else openOutputMenu(); }
  root.addEventListener('click', (e) => {
    if (outputMenu && !e.target.closest('.drive-output-menu, .drive-output')) closeOutputMenu();
  });

  // ---- quick sheet ----------------------------------------------------------
  async function openSheet() {
    if (sheet) return;
    const scrim = el('div', { class: 'drive-scrim' });
    // The roster may have changed elsewhere (a Bots-tab edit, another device):
    // re-read it, but never hold the sheet up for long (review L6).
    const reload = ctx.reloadBots
      ? Promise.race([Promise.resolve().then(() => ctx.reloadBots()).catch(() => {}), new Promise((r) => setTimeout(r, 1500))])
      : null;
    const [voices] = await Promise.all([phoneVoices().catch(() => []), reload]);
    if (sheet || closed) return;
    const panel = buildQuickSheet({
      status: ctx.voiceStatus ? ctx.voiceStatus() : null,
      bot: (ctx.getBot && ctx.getBot(thread.bot_id)) || bot,
      voices, android: isAndroid(), lang: language(),
      setBotVoice: ctx.setBotVoice,
      onClose: () => closeSheet(),
      // Settings opens OVER this screen with the microphone paused; nothing
      // hangs up (experience L8). A tap on the picture resumes afterwards.
      onAllSettings: () => {
        closeSheet();
        if (listening) setListening(false);
        if (ctx.openDriveSettings) ctx.openDriveSettings();
      },
      notify: (msg) => showNotice(msg, { ms: 6000 }),
    });
    scrim.addEventListener('click', () => closeSheet());
    sheet = { scrim, panel };
    root.append(scrim, panel);
    col.inert = true;
    const first = panel.querySelector('select, button, input');
    if (first) first.focus();
  }
  function closeSheet() {
    if (!sheet) return;
    sheet.scrim.remove(); sheet.panel.remove();
    sheet = null;
    col.inert = false;
    gear.focus();
  }

  // ---- lifecycle -------------------------------------------------------------
  async function holdWakeLock() {
    if (!settings.screenOn) return;
    try {
      if ('wakeLock' in navigator && document.visibilityState === 'visible') {
        wakeLock = await navigator.wakeLock.request('screen');
      }
    } catch { /* battery saver or unsupported: the screen may sleep */ }
  }

  function setupMediaSession() {
    if (!('mediaSession' in navigator)) return;
    try {
      const meta = { title: (thread.title || '').trim() || t('drive.title'), artist: bot ? bot.name : 'DisPatch', album: 'DisPatch' };
      if (avatar.src) meta.artwork = [{ src: new URL(avatar.src, location.href).href, sizes: '256x256' }];
      navigator.mediaSession.metadata = new MediaMetadata(meta);
      navigator.mediaSession.setActionHandler('play', () => tap());
      navigator.mediaSession.setActionHandler('pause', () => tap());
      try { navigator.mediaSession.setActionHandler('stop', () => close()); } catch { /* optional */ }
    } catch { /* not fatal */ }
  }

  function onVisibility() {
    if (document.visibilityState === 'hidden') {
      // The mic dies with a hidden page anyway; say so to the server so a
      // half-heard turn is not submitted, and to the listener on return.
      if (listening && state !== 'stopped') { pausedByHide = true; setListening(false); }
      return;
    }
    if (settings.screenOn) holdWakeLock();
    ac.resume().catch(() => {});
    retryNow();
    if (pausedByHide) {
      pausedByHide = false;
      setListening(true);
      showNotice(t('drive.paused_screen_off'), { kind: 'info', ms: 4000 });
    }
  }
  function onOnline() { retryNow(); }

  ac.onstatechange = () => {
    if (closed) return;
    if (ac.state === 'interrupted' || ac.state === 'suspended') {
      ac.resume().catch(() => {
        if (document.visibilityState === 'visible') showNotice(t('drive.audio_interrupted'), { kind: 'info' });
      });
    }
  };

  const unsubscribe = onDriveSettings((next) => {
    const prev = settings;
    settings = next;
    if (listening) sendStart();
    syncTalk();
    if (!settings.showTimings) timings.hidden = true;
    if (prev.hangupQuietMin !== next.hangupQuietMin) quiet.setMinutes(next.hangupQuietMin);
    if (prev.typedInput !== next.typedInput) {
      // Switched from the quick sheet, which is still open: the box comes out
      // without taking the focus from under it.
      if (next.typedInput) { if (listening) setListening(false); openCompose({ focus: false }); } else closeCompose();
    }
    if (prev.screenOn !== next.screenOn) {
      if (next.screenOn) holdWakeLock(); else { try { wakeLock && wakeLock.release(); } catch { /* gone */ } wakeLock = null; }
    }
    if (prev.micId !== next.micId && stream) {
      stopMic();
      startMic().catch((err) => stop(micError(err)));
    }
    if (prev.sinkId !== next.sinkId && typeof ac.setSinkId === 'function') ac.setSinkId(next.sinkId || '').catch(() => {});
    if (prev.sinkId !== next.sinkId) paintOutput();
    if (prev.slowLink !== next.slowLink || prev.phoneVoice !== next.phoneVoice) audio.settingsChanged();
    paintLink();
    paint();
  });

  function onKey(e) {
    if (e.key === 'Escape') {
      e.preventDefault();
      if (sheet) closeSheet();
      else if (outputMenu) closeOutputMenu();
      // Esc in the text box leaves the box, never the chat: it closes a box the
      // Type button opened, and only moves the focus to the picture with "Type
      // instead of talking" (the box stays; a blur would put the focus on the
      // body, outside the overlay's key handling).
      else if (composeOpen && e.target === composeInput) { if (settings.typedInput) ring.focus(); else closeCompose(); }
      else close();
      return;
    }
    if (e.key === 'Tab') { trapFocus(e); return; }
    const tag = (e.target && e.target.tagName) || '';
    // The picture is a button and holds the focus from the start, so the
    // screen's shortcuts must apply on it too: Space = the microphone (never
    // the picture's tap, which interrupts a playing reply), U = Undo. Enter
    // on the picture stays its tap. Other buttons keep their own Space.
    const onRing = e.target === ring;
    if ((!onRing && /^(BUTTON|SELECT|INPUT|TEXTAREA)$/.test(tag)) || sheet) return;
    if (e.key === ' ' || e.code === 'Space') {
      if (state === 'stopped') return;          // the picture's tap is Try again here
      e.preventDefault(); quiet.heard(); setListening(!listening);
    }
    else if ((e.key === 'u' || e.key === 'U') && !undoBtn.hidden) { e.preventDefault(); doUndo(); }
  }
  function trapFocus(e) {
    const scope = sheet ? sheet.panel : root;
    const items = [...scope.querySelectorAll('button, select, input, textarea, [tabindex]:not([tabindex="-1"])')]
      .filter((n) => !n.disabled && !n.hidden && n.offsetParent !== null);
    if (!items.length) return;
    const first = items[0];
    const last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }

  function close({ quiet: silent = false } = {}) {
    if (closed) return;
    closed = true;
    opening = false;
    hb.stop();
    eotTone.cancel();
    clearInterval(quietTicker);
    clearInterval(activityTicker);
    clearTimeout(reconnectTimer);
    clearInterval(thinkingTicker); clearInterval(undoTicker); clearInterval(offlineTicker);
    clearTimeout(noticeTimer);
    clearInterval(audioTicker);
    fitViewport.stop();
    unsubscribe();
    try { if (ws) { ws.onclose = null; ws.close(); } } catch { /* gone */ }
    stopMic();
    try { audio.dispose(); } catch { /* gone */ }
    try { ac.close(); } catch { /* gone */ }
    try { wakeLock && wakeLock.release(); } catch { /* gone */ }
    try { keepAlive && keepAlive.pause(); } catch { /* gone */ }
    if ('mediaSession' in navigator) {
      for (const a of ['play', 'pause', 'stop']) { try { navigator.mediaSession.setActionHandler(a, null); } catch { /* ok */ } }
      try { navigator.mediaSession.metadata = null; } catch { /* ok */ }
    }
    document.removeEventListener('visibilitychange', onVisibility);
    globalThis.removeEventListener && globalThis.removeEventListener('online', onOnline);
    root.remove();
    document.body.classList.remove('drive-open');
    releaseInert(inertToken);
    active = null;
    if (popHistory) { const p = popHistory; popHistory = null; try { p(); } catch { /* best effort */ } }
    const back = ctx.returnFocus;
    if (back && back.isConnected) { try { back.focus(); } catch { /* best effort */ } }
    if (!silent && ctx.toast) ctx.toast(t('drive.ended'));
  }

  ring.addEventListener('click', tap);
  pauseBtn.addEventListener('click', () => { quiet.heard(); if (state === 'stopped') retry(); else setListening(!listening); });
  typeBtn.addEventListener('click', toggleCompose);
  outputBtn.addEventListener('click', (e) => { e.stopPropagation(); toggleOutputMenu(); });
  endBtn.addEventListener('click', () => close());
  compose.addEventListener('submit', (e) => { e.preventDefault(); sendTyped(); });
  composeInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.keyCode !== 229) { e.preventDefault(); sendTyped(); }
  });
  composeInput.addEventListener('input', autoGrow);
  undoBtn.addEventListener('click', doUndo);
  gear.addEventListener('click', openSheet);
  root.addEventListener('keydown', onKey);
  // A button activates on Space's keyup: keep the picture's tap out of it (onKey).
  root.addEventListener('keyup', (e) => {
    if (e.target === ring && (e.key === ' ' || e.code === 'Space') && state !== 'stopped') e.preventDefault();
  });
  document.addEventListener('visibilitychange', onVisibility);
  if (globalThis.addEventListener) globalThis.addEventListener('online', onOnline);
  // One history entry: a phone's Back closes voice chat instead of leaving the app.
  popHistory = pushHistory(ctx.history, () => close());

  syncTalk();
  setState('connecting');
  paintLink();
  active = {
    close,
    get state() { return state; },
    /** Automation hook: force the link notice ({degraded}) or hand it back to the monitor (null). */
    setLink(l) { linkForced = l ? { degraded: !!l.degraded } : null; paintLink(); },
    /** The audio engine (tests, the Voice lab). */
    get audio() { return audio; },
    /** The text box is out (the Type button, or "Type instead of talking"). */
    get composing() { return composeOpen; },
    /** Test/automation hook: feed one server frame. */
    _frame: (m) => onFrame(m),
  };
  if (settings.typedInput) openCompose({ focus: false });
  opening = false;
  setupMediaSession();
  ring.focus();

  // Socket and mic start together: the mic prompt can take seconds, and the
  // server loads its models meanwhile. Audio is held until `ready` either way.
  connect();
  startPlayer();
  startMic().then(() => holdWakeLock()).catch((err) => { if (can()) stop(micError(err)); });
  return active;
}

/**
 * One history entry for the open voice chat (a phone's Back hangs up). The
 * shell's `history.push(tag, onBack)` when present (C-§E.1); otherwise the
 * browser's own, ignoring a pop that lands back on our entry (Settings opened
 * over the voice chat pushes its own entry and pops it).
 */
export function pushHistory(shellHistory, onBack) {
  if (shellHistory && typeof shellHistory.push === 'function') {
    try { return shellHistory.push('voice', onBack); } catch { /* fall through */ }
  }
  try {
    globalThis.history.pushState({ drive: 1 }, '');
    let mine = true;
    const onPop = (e) => {
      if (e.state && e.state.drive) return;
      mine = false;
      globalThis.removeEventListener('popstate', onPop);
      onBack();
    };
    globalThis.addEventListener('popstate', onPop);
    return () => {
      globalThis.removeEventListener('popstate', onPop);
      if (mine) { mine = false; try { globalThis.history.back(); } catch { /* best effort */ } }
    };
  } catch { return null; }
}
