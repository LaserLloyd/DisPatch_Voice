// The voice chat screen (owner: track T7): the pure half of drive.js (ported
// from the pre-split suite, DisPatch e8a5f75 frontend/tests/drive.test.js) plus the v3
// behaviours of SPEC §7.4 driven through the real overlay under jsdom with a
// fake AudioContext, microphone and WebSocket:
//   Pause pauses only the mic ("Paused — still speaking"), one sound per turn
//   end (800 ms), first-run hint ×3 / English-only, "Got it — go on?" ≥ 600 ms,
//   live captions, narrator italics, All settings… keeps the socket, Hang up
//   when quiet, waiting after 60 s offline, busy 1013, Esc / Space.

import test from 'node:test';
import assert from 'node:assert/strict';
import { domSkip } from './_require-dom.js';
import { HAS_DOM, fakeShell, loadBundle, settle, withDom } from './_shell-fake.js';

import {
  driveAllowed, nextBackoff, voiceUrl, driveDeviceId, newStreamId, linkStateNoticeKey,
  reconnectDecision, tapAction, thinkingPhase, hintKey, createUndoWindow,
  createHeartbeat, createHeardTracker, turnErrorKey, errorFrameKey, EARCONS, timingsLine,
  displayState, firstRunTipKey, linkNoticeKey, createEotTone, createQuietTimer,
  pushHistory, noteMetrics, getLastMetrics, initDrive, outputRoute, outputMenuModel,
  GIVE_UP_MS, WAITING_AFTER_MS, WAITING_RETRY_MS, UNDO_MS, PING_EVERY_MS, PONG_TIMEOUT_MS,
  EOT_TONE_HOLD_MS, GOTIT_MIN_HOLD_MS, FIRST_RUN_OPENS, QUIET_GRACE_MS, PRE_READY_MS,
  ACTIVITY_EVERY_MS, FOLLOW_PAUSE_MS,
} from '../../src/dispatch_voice/static/drive.js';
import { bindShell } from '../../src/dispatch_voice/static/shell.js';
import { saveDriveSettings } from '../../src/dispatch_voice/static/drive-settings.js';
import { PlayerCore } from '../../src/dispatch_voice/static/voice-player-worklet.js';
import { packDown, CODEC, KIND } from '../../src/dispatch_voice/static/opus-codec.js';
import { fakeSynth, FakeUtterance } from './_t8-fakes.js';

const dom = { skip: domSkip(HAS_DOM ? false : 'jsdom is not installed (npm install)') };
const EN = loadBundle('en');

// ------------------------------------------------------------- basics ---

test('driveDeviceId: stable per browser, well-formed, survives no storage', () => {
  const m = new Map();
  const st = { getItem: (k) => m.get(k) ?? null, setItem: (k, v) => m.set(k, v) };
  const a = driveDeviceId(st);
  assert.match(a, /^[A-Za-z0-9-]{8,64}$/);
  assert.equal(driveDeviceId(st), a);
  assert.equal(typeof driveDeviceId(null), 'string');
});

test('voiceUrl: page scheme, escaped thread id; the v3 negotiation only when asked (C.1)', () => {
  assert.equal(voiceUrl({ protocol: 'https:', host: 'box:8443' }, 't/1'), 'wss://box:8443/ws/voice/t%2F1');
  assert.equal(voiceUrl({ protocol: 'http:', host: 'h' }, 't', 'dev-12345'), 'ws://h/ws/voice/t?d=dev-12345');
  assert.equal(voiceUrl({ protocol: 'https:', host: 'h' }, 't', 'dev-12345', { v3: true, stream: 'str-0001', up: ['opus', 'pcm16'], down: ['pcm16'] }),
    'wss://h/ws/voice/t?d=dev-12345&s=str-0001&v=3&up=opus,pcm16&down=pcm16');
});

test('driveAllowed: locked, no thread, safe bot and insecure origin all refuse, each with its own reason', () => {
  const thread = { id: 't', bot_id: 'b' };
  assert.equal(driveAllowed({ locked: true, thread, bot: {}, secure: true }).whyKey, 'drive.err_locked_device');
  assert.equal(driveAllowed({ locked: false, thread: null, bot: {}, secure: true }).whyKey, 'drive.err_no_chat');
  assert.equal(driveAllowed({ locked: false, thread, bot: { safe: true }, secure: true }).whyKey, 'drive.err_safe_bot');
  assert.equal(driveAllowed({ locked: false, thread, bot: {}, secure: false }).whyKey, 'drive.err_https');
  assert.equal(driveAllowed({ locked: false, thread, bot: { safe: false }, secure: true }).ok, true);
});

test('backoff: 0.5 s doubling to 8 s, then every 15 s once waiting; give up after 15 min (§5.5)', () => {
  const seq = [];
  let b = 0;
  for (let i = 0; i < 7; i++) { b = nextBackoff(b); seq.push(b); }
  assert.deepEqual(seq, [500, 1000, 2000, 4000, 8000, 8000, 8000]);
  assert.equal(nextBackoff(8000, { waiting: true }), WAITING_RETRY_MS);
  assert.equal(WAITING_AFTER_MS, 60000);
  assert.equal(WAITING_RETRY_MS, 15000);
  assert.equal(GIVE_UP_MS, 15 * 60000);
});

test('stream id: random per voice chat, the shape C.1 allows for `s`', () => {
  const a = newStreamId();
  const b = newStreamId();
  assert.match(a, /^[A-Za-z0-9-]{8,64}$/);
  assert.notEqual(a, b);
});

test('earcons: ready rises, err falls, all four exist', () => {
  assert.deepEqual(Object.keys(EARCONS).sort(), ['ack', 'eot', 'err', 'ready']);
  assert.ok(EARCONS.ready.every((f, i) => i === 0 || f > EARCONS.ready[i - 1]));
  assert.ok(EARCONS.err[1] < EARCONS.err[0]);
});

test('reconnectDecision (C.4): 1008/1013 stop — busy and switch-off named — 4001 quiet, 1011 and the rest retry', () => {
  assert.deepEqual(reconnectDecision(1008), { retry: false, reason: 'refused' });
  assert.deepEqual(reconnectDecision(1013), { retry: false, reason: 'unavailable' });
  assert.deepEqual(reconnectDecision(1013, 'No extension provides voice'), { retry: false, reason: 'unavailable' });
  assert.deepEqual(reconnectDecision(1013, 'Voice is busy on this server (3 chats open)'), { retry: false, reason: 'busy' });
  assert.deepEqual(reconnectDecision(1013, 'Voice is turned off in Settings'), { retry: false, reason: 'voice_off' });
  assert.deepEqual(reconnectDecision(4001), { retry: false, reason: 'replaced' });
  assert.deepEqual(reconnectDecision(1011), { retry: true, reason: 'internal' });
  for (const code of [1006, 1001, 1000, 1012]) assert.deepEqual(reconnectDecision(code), { retry: true, reason: 'network' });
});

test('heartbeat: a ping every 2 s; dead after 8 s, or 4 × the RTT once it is above 1.5 s (C.3)', () => {
  let now = 0;
  const sent = [];
  let dead = 0;
  const hb = createHeartbeat({ send: (f) => sent.push(f), onDead: () => { dead += 1; }, now: () => now, setInt: () => 1, clearInt: () => {} });
  assert.equal(PING_EVERY_MS, 2000);
  assert.equal(PONG_TIMEOUT_MS, 8000);
  hb.start();
  now = 2000; hb.tick();
  assert.equal(sent[0].type, 'ping');
  hb.pong(1000);                 // RTT 1 s
  now = 9000; hb.tick();
  assert.equal(dead, 0);
  now = 10000; hb.tick();        // 8 s since the pong
  assert.equal(dead, 1);
  // A slow cellular link: RTT 3 s → the deadline stretches to 12 s.
  now = 20000; hb.start();
  for (let i = 0; i < 20; i++) hb.pong(now - 3000);
  assert.ok(hb.rtt > 2900);
  assert.ok(hb.deadline() >= 11600);
  now = 29000; hb.tick();
  assert.equal(dead, 1, 'not dead at 9 s on a 3 s RTT');
});

test('heartbeat: before `ready` the server is loading models — no drop until PRE_READY_MS (review L2)', () => {
  let now = 0;
  const sent = [];
  let dead = 0;
  const hb = createHeartbeat({ send: (f) => sent.push(f), onDead: () => { dead += 1; }, now: () => now, setInt: () => 1, clearInt: () => {} });
  assert.equal(PRE_READY_MS, 60000);
  hb.start({ armed: false });
  assert.equal(hb.armed, false);
  now = 2000; hb.tick();
  assert.equal(sent.length, 1, 'pings go out from open (the server marks the socket as pinging)');
  now = 12000; hb.tick();
  assert.equal(dead, 0, 'a 12 s cold load is not a dead link');
  now = 20000; hb.arm();                     // ready
  now = 27000; hb.tick();
  assert.equal(dead, 0);
  now = 28000; hb.tick();
  assert.equal(dead, 1, 'armed: the normal 8 s deadline from `ready`');
  // A server that never says ready is given up on after a minute.
  now = 100000; hb.start({ armed: false });
  now = 159000; hb.tick(); assert.equal(dead, 1);
  now = 160000; hb.tick(); assert.equal(dead, 2);
});

// ------------------------------------------------------- state machine ---

test('tapAction: one meaning per state; playback means interrupt; a fresh stop can be continued (v3)', () => {
  assert.equal(tapAction('listening'), 'pause');
  assert.equal(tapAction('paused'), 'resume');
  assert.equal(tapAction('thinking'), 'cancel');
  assert.equal(tapAction('speaking'), 'interrupt');
  assert.equal(tapAction('paused', { playing: true }), 'interrupt', 'paused mic, reply playing: the tap interrupts');
  assert.equal(tapAction('listening', { resumable: true }), 'continue');
  assert.equal(tapAction('stopped'), 'retry');
  for (const s of ['connecting', 'offline', 'waiting', 'locked']) assert.equal(tapAction(s), 'none', s);
});

test('hints: every tappable state says what the tap does; waiting and tap-to-continue have theirs', () => {
  assert.equal(hintKey('listening'), 'drive.hint_pause');
  assert.equal(hintKey('listening', { autoListen: false }), 'drive.hint_pause_ptt');
  assert.equal(hintKey('hearing'), 'drive.hint_keep_talking');
  assert.equal(hintKey('paused'), 'drive.hint_resume');
  assert.equal(hintKey('speaking'), 'drive.hint_interrupt');
  assert.equal(hintKey('waiting'), 'drive.hint_waiting');
  assert.equal(hintKey('listening', { resumable: true }), 'drive.hint_resume_after_stop');
  assert.equal(hintKey('thinking', { thinkingMs: 26000 }), 'drive.hint_still_working');
  assert.equal(thinkingPhase(4000), 'counting');
  assert.equal(hintKey('stopped'), null);
});

test('displayState: Pause pauses only the mic — every live state reads "paused", "still speaking" while a reply plays', () => {
  assert.deepEqual(displayState('speaking', { listening: false }), { state: 'paused', wordKey: 'drive.paused_speaking' });
  assert.deepEqual(displayState('listening', { listening: false, playing: true }), { state: 'paused', wordKey: 'drive.paused_speaking' });
  assert.deepEqual(displayState('thinking', { listening: false }), { state: 'paused', wordKey: 'drive.state_paused' });
  assert.deepEqual(displayState('offline', { listening: false }), { state: 'offline', wordKey: null });
  assert.deepEqual(displayState('speaking', { listening: true }), { state: 'speaking', wordKey: null });
});

test('first-run tip: three opens per device; the first open in another UI language is the English-only line', () => {
  assert.equal(FIRST_RUN_OPENS, 3);
  assert.equal(firstRunTipKey(0, 'en'), 'drive.first_run_hint');
  assert.equal(firstRunTipKey(2, 'en'), 'drive.first_run_hint');
  assert.equal(firstRunTipKey(3, 'en'), null);
  assert.equal(firstRunTipKey(0, 'de'), 'drive.english_only');
  assert.equal(firstRunTipKey(1, 'de'), 'drive.first_run_hint');
  assert.equal(firstRunTipKey(0, 'en-GB'), 'drive.first_run_hint');
});

test('link notice: which voice replies come in while degraded (§7.4)', () => {
  assert.equal(linkNoticeKey(false, 'phone'), null);
  assert.equal(linkNoticeKey(true, 'phone', true), 'drive.link_phone');
  assert.equal(linkNoticeKey(true, 'phone', false), 'drive.link_text');
  assert.equal(linkNoticeKey(true, 'wait'), 'drive.link_wait');
  assert.equal(linkNoticeKey(true, 'text'), 'drive.link_text');
});

test('one sound per turn end: the eot tone waits 800 ms and an ack or first sentence drops it (M1, FR-works 5)', () => {
  const timers = new Map();
  let id = 0;
  const setT = (fn, ms) => { timers.set(++id, { fn, ms }); return id; };
  const clearT = (h) => timers.delete(h);
  let played = 0;
  const tone = createEotTone({ play: () => { played += 1; }, setT, clearT });
  assert.equal(EOT_TONE_HOLD_MS, 800);
  tone.eot();
  assert.equal([...timers.values()][0].ms, 800);
  tone.suppress();
  assert.equal(timers.size, 0);
  assert.equal(played, 0);
  tone.eot();
  [...timers.values()][0].fn();
  assert.equal(played, 1);
});

test('hang up when quiet: quiet_check after N minutes, hang-up 20 s later unless someone speaks (A9)', () => {
  let now = 0;
  const events = [];
  const q = createQuietTimer({ minutes: 5, now: () => now, onCheck: () => events.push('check'), onHangup: () => events.push('hangup') });
  assert.equal(QUIET_GRACE_MS, 20000);
  now = 299000; assert.equal(q.tick(), null);
  now = 300000; assert.equal(q.tick(), 'check');
  now = 310000; assert.equal(q.tick(), null);
  q.activity();                                    // the server speaking "Still there?" itself
  assert.equal(q.checking, true, 'bot-side activity never answers the check (review M3)');
  q.heard();                                       // "Still there?" — "Yes."
  assert.equal(q.checking, false);
  now = 330000; assert.equal(q.tick(), null);
  now = 630000; assert.equal(q.tick(), 'check');
  now = 650000; assert.equal(q.tick(), 'hangup');
  assert.deepEqual(events, ['check', 'check', 'hangup']);
  // Outside a check, a reply resets the idle clock.
  now = 900000; q.activity();
  now = 900000 + 299000; assert.equal(q.tick(), null);
  const never = createQuietTimer({ minutes: 0, now: () => 1e12 });
  assert.equal(never.tick(), null);
});

test('hang up when quiet: with the microphone paused it hangs up without asking (review L5)', () => {
  let now = 0;
  const events = [];
  const q = createQuietTimer({ minutes: 5, now: () => now, onCheck: () => events.push('check'), onHangup: () => events.push('hangup') });
  now = 299000; assert.equal(q.tick({ canAsk: false }), null);
  now = 300000; assert.equal(q.tick({ canAsk: false }), 'hangup');
  assert.deepEqual(events, ['hangup']);
});

test('undo window, heard tracker', () => {
  let now = 1000;
  const u = createUndoWindow(UNDO_MS, () => now);
  u.open();
  now += 7999; assert.equal(u.canUndo(), true);
  now += 1; assert.equal(u.canUndo(), false);
  const h = createHeardTracker(3);
  h.note('a'); h.note('b'); h.note('a'); h.note('c'); h.note('d');
  assert.deepEqual(h.ids, ['d', 'c', 'a']);
});

test('error words: turn errors, codes, and 2.1.0 English messages', () => {
  assert.equal(turnErrorKey('timeout'), 'drive.err_timeout');
  assert.equal(turnErrorKey('x'), 'drive.err_agent');
  assert.equal(errorFrameKey({ code: 'unavailable' }), 'drive.err_internal_retry');
  assert.equal(errorFrameKey({ code: 'bad_frame' }), null);
  assert.equal(errorFrameKey({ message: 'Drive mode hit an internal error and stopped.' }), 'drive.err_internal_retry');
  assert.equal(errorFrameKey({ message: 'Speech synthesis failed.' }), 'drive.err_tts');
});

test('timings line reads as sentences in the UI language, no middle dots (1.0.1, audit L12)', dom, () => withDom(async () => {
  const times = { heard: 420, first: 1600, link: { rtt_ms: 420.4, down_kbps: 31, mode: 'phone' } };
  for (const [lang, want] of [
    ['en', 'Heard in 420 ms. Reply started in 1.6 s. 420 ms, 31 kbps, phone voice.'],
    ['ja', null],
  ]) {
    const { shell } = fakeShell({ lang });
    const prev = bindShell(shell);
    try {
      const line = timingsLine(times);
      assert.doesNotMatch(line, /·/);
      if (want) assert.equal(line, want);
      else assert.match(line, /^[^。]+。[^。]+。[^。]+。$/, `${lang}: three sentences ending in 。, no spaces between (${line})`);
    } finally { bindShell(prev); }
  }
}));

test('locales: the stock voice and the default voice are two different words in every language (1.0.1, audit L14)', () => {
  for (const lang of ['en', 'de', 'es', 'fr', 'pt', 'ja', 'zh', 'ar']) {
    const b = loadBundle(lang);
    assert.notEqual(b.drive.default_voice, b.drive.voice_stock, `${lang}: drive.default_voice vs drive.voice_stock`);
    assert.notEqual(b.voices.default_badge, b.voices.stock_badge, `${lang}: voices.default_badge vs voices.stock_badge`);
    assert.doesNotMatch(b.drive.timing_link, /·/, `${lang}: drive.timing_link`);
  }
  const DE = loadBundle('de');
  assert.equal(DE.drive.voice_stock, 'Mitgelieferte Stimme');
  assert.equal(DE.voices.stock_badge, 'Mitgeliefert');
});

test('timings line: heard / first audio, plus the link line when the server sent one (Voice lab)', () => {
  const { shell } = { shell: { i18n: { t: (k, v) => `${k}:${JSON.stringify(v || {})}` } } };
  const prev = bindShell(shell);
  try {
    assert.equal(timingsLine({}), '');
    const line = timingsLine({ heard: 420, first: 1600, link: { rtt_ms: 420.4, down_kbps: 31, mode: 'phone' } });
    assert.match(line, /drive\.timing_heard:\{"t":"420 ms"\}/);
    assert.match(line, /drive\.timing_first:\{"t":"1\.6 s"\}/);
    assert.match(line, /drive\.timing_link:\{"rtt":420,"kbps":31,"mode":"drive\.mode_phone/);
  } finally { bindShell(prev); }
  const m = noteMetrics({ type: 'metrics', speculation: { started: 2 } });
  assert.equal(m.speculation.started, 2);
  noteMetrics({ type: 'metrics', link: { rtt_ms: 9 } });
  assert.equal(getLastMetrics().speculation.started, 2, 'merged, not replaced');
});

test('history: the shell\'s history.push when present (C-§E.1), else the browser\'s', () => {
  const calls = [];
  const pop = pushHistory({ push: (tag, onBack) => { calls.push(tag); return () => calls.push('pop'); } }, () => {});
  assert.deepEqual(calls, ['voice']);
  pop();
  assert.deepEqual(calls, ['voice', 'pop']);
});

// ------------------------------------------------- the overlay under jsdom ---

/** Fake Web Audio + mic + socket, recording what the overlay does. */
function fakeAudioWorld() {
  const world = { sockets: [], oscillators: [], sources: [], micStarted: 0 };
  class FakeParam { constructor() { this.value = 1; } cancelScheduledValues() {} setTargetAtTime(v) { this.value = v; } setValueAtTime() {} exponentialRampToValueAtTime() {} }
  class FakeAC {
    constructor() {
      this.currentTime = 0; this.state = 'running'; this.destination = {}; this.sampleRate = 48000;
      this.audioWorklet = { addModule: async (url) => { world.workletUrl = url; } };
    }
    resume() { return Promise.resolve(); }
    close() { this.state = 'closed'; return Promise.resolve(); }
    createGain() { return { gain: new FakeParam(), connect() {} }; }
    createAnalyser() { return { fftSize: 512, connect() {}, getFloatTimeDomainData(b) { b.fill(0); } }; }
    createBuffer(ch, len, sr) { const d = new Float32Array(len); return { duration: len / sr, getChannelData: () => d }; }
    createBufferSource() {
      const s = { onended: null, connect() {}, start() {}, stop() {} };
      world.sources.push(s);
      return s;
    }
    createOscillator() {
      const o = { frequency: { value: 0 }, connect() {}, start() {}, stop() {} };
      world.oscillators.push(o);
      return o;
    }
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
  }
  // The mic node records what it is told; the player node runs the REAL PlayerCore
  // (voice-player-worklet.js) so `played`/`idle` come back as in a browser.
  world.players = [];
  class FakeNode {
    constructor(ac, name) {
      this.name = name;
      this.port = { posted: [], postMessage: (m) => { this.port.posted.push(m); if (this.core) this.core.onMessage(m); }, onmessage: null };
      if (name === 'drive-player') {
        this.core = new PlayerCore({ postMessage: (m) => this.port.onmessage && this.port.onmessage({ data: m }) }, 48000);
        world.players.push(this);
      }
      if (name === 'drive-mic') world.mic = this;
    }
    connect() {}
  }
  /** Play `ms` of reply audio on every player node. */
  world.render = (ms) => {
    const out = new Float32Array(128);
    for (let done = 0; done < ms; done += 128 / 48) for (const p of world.players) p.core.render(out);
  };
  class FakeWS {
    constructor(url) { this.url = url; this.readyState = 0; this.sent = []; this.closed = null; world.sockets.push(this); }
    send(d) {
      const f = typeof d === 'string' ? JSON.parse(d) : d;
      this.sent.push(f);
      // A live server answers every ping at once (C.4 pong).
      if (f && f.type === 'ping' && this.readyState === 1) this.msg({ type: 'pong', t: f.t, rx_off: 0, srv_t: 0 });
    }
    close(code) { this.closed = code || 1000; this.readyState = 3; }
    open() { this.readyState = 1; this.onopen && this.onopen(); }
    msg(obj) { this.onmessage && this.onmessage({ data: typeof obj === 'string' || obj instanceof ArrayBuffer ? obj : JSON.stringify(obj) }); }
    drop(code, reason = '') { this.readyState = 3; this.onclose && this.onclose({ code, reason }); }
    frames(type) { return this.sent.filter((f) => f && f.type === type); }
  }
  class FakeAudio { constructor() { this.loop = false; } play() { return Promise.resolve(); } pause() {} }
  world.globals = { AudioContext: FakeAC, AudioWorkletNode: FakeNode, WebSocket: FakeWS, Audio: FakeAudio };
  world.ws = () => world.sockets.at(-1);
  world.freqs = () => world.oscillators.map((o) => o.frequency.value);
  return world;
}

async function openOverlay(world, { lang = 'en', toast = [], openSettings = null, ctx: extra = {} } = {}) {
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: async () => { world.micStarted += 1; return { getTracks: () => [{ stop() {} }] }; }, enumerateDevices: async () => [] },
  });
  const fs = fakeShell({ lang });
  const prev = bindShell(fs.shell);
  const button = document.createElement('button');
  document.body.append(button);
  const thread = { id: 'th-1', bot_id: 'nova', title: 'Evening' };
  const bot = { id: 'nova', name: 'Nova', voice: '' };
  const d = initDrive(button, {
    getThread: () => thread, getBot: () => bot, isLocked: () => false,
    toast: (m) => toast.push(m), getAvatar: () => ({ src: '', letter: 'S' }),
    voiceStatus: () => ({ enabled: true, profiles: [], builtin_voices: [] }),
    openDriveSettings: openSettings || (() => {}), history: fs.shell.history,
    ...extra,
  });
  button.click();
  await settle();
  const ws = world.ws();
  ws.open();
  ws.msg({ type: 'ready', sample_rate_in: 16000 });
  await settle();
  const overlay = document.querySelector('.drive-overlay');
  return { d, ws, overlay, fs, button, restore: () => bindShell(prev), word: () => overlay.querySelector('.drive-state').textContent };
}

/**
 * Advance mocked time in small steps: node's MockTimers moves Date to the
 * end of a large tick() before it fires the intervals inside it, which no
 * real clock does (a 2 s heartbeat would see 5 minutes of silence at once).
 */
function advance(t, ms, step = 500) {
  for (let left = ms; left > 0; left -= step) t.mock.timers.tick(Math.min(step, left));
}

function replyAudio(ws, { seq = 1, gen = 0, text = 'Hello there.', kind = 'reply', chunk = 0 } = {}) {
  ws.msg({ type: 'audio', gen, seq, chunk, sample_rate: 24000, text, message_id: 'm-1', kind });
  ws.msg(new Int16Array(2400).fill(3000).buffer);
}

/** v3: a sentence's `say`, then `ms` of its PCM16 audio as wire3 binary frames. */
function sayWithAudio(ws, { seq = 1, gen = 0, text = 'Hello.', role = 'character', mid = 'm-1', ms = 200, kind = 'reply' } = {}) {
  ws.msg({ type: 'say', gen, seq, text, kind, role, message_id: mid, t: 1 });
  for (let left = ms; left > 0; left -= 40) {
    const b = new Uint8Array(40 * 24 * 2).fill(16);
    ws.msg(packDown(CODEC.pcm16, KIND[kind], gen, seq, 0, [b]).buffer);
  }
}

test('overlay: Pause sends ONLY `pause`; a playing reply finishes under "Paused — still speaking"; the tap interrupts (M7)', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    const o = await openOverlay(world);
    try {
      assert.equal(o.overlay.dataset.state, 'listening');
      replyAudio(o.ws);
      o.ws.msg({ type: 'state', state: 'speaking' });
      const before = o.ws.sent.length;
      o.overlay.querySelector('.drive-pause').click();
      const after = o.ws.sent.slice(before);
      assert.deepEqual(after.map((f) => f.type), ['pause']);
      assert.equal(o.overlay.dataset.state, 'paused');
      assert.equal(o.word(), EN.drive.paused_speaking);
      assert.match(o.overlay.querySelector('.drive-pause').textContent, new RegExp(EN.drive.resume));
      assert.equal(o.overlay.querySelector('.drive-hint').textContent, EN.drive.hint_interrupt, 'the hint names the interrupt, not "listen"');
      // The picture tap while the reply plays is the interrupt.
      o.overlay.querySelector('.drive-ring').click();
      assert.equal(o.ws.frames('stop_playback').length, 1);
      // Reply over, server says paused: the plain word.
      o.ws.msg({ type: 'state', state: 'paused' });
      assert.equal(o.word(), EN.drive.state_paused);
      o.overlay.querySelector('.drive-pause').click();          // Resume
      assert.equal(o.ws.sent.at(-1).type, 'start');
    } finally { o.d.close(); o.restore(); }
  }, world.globals);
});

test('overlay: one sound per turn end — the eot tone is dropped when an ack or the first sentence arrives within 800 ms', dom, async (t) => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    const o = await openOverlay(world);
    t.mock.timers.enable({ apis: ['setTimeout'] });
    try {
      const EOT = EARCONS.eot[0];
      const count = (f) => world.freqs().filter((x) => x === f).length;
      o.ws.msg({ type: 'eot', reason: 'silence', upto: 16000 });
      t.mock.timers.tick(300);
      o.ws.msg({ type: 'ack' });
      t.mock.timers.tick(1000);
      assert.equal(count(EOT), 0, 'ack in the window: no eot tone');
      assert.ok(count(EARCONS.ack[0]) >= 1, 'the ack earcon is the one sound');
      o.ws.msg({ type: 'eot', reason: 'silence', upto: 32000 });
      t.mock.timers.tick(200);
      o.ws.msg({ type: 'say', gen: 0, seq: 2, text: 'Sure.', kind: 'reply', role: 'character', message_id: 'm-2', t: 1 });
      t.mock.timers.tick(1000);
      assert.equal(count(EOT), 0, 'first reply sentence in the window: no eot tone');
      o.ws.msg({ type: 'eot', reason: 'silence', upto: 48000 });
      t.mock.timers.tick(EOT_TONE_HOLD_MS + 1);
      assert.equal(count(EOT), 1, 'nothing came: the tone plays');
    } finally { t.mock.timers.reset(); o.d.close(); o.restore(); }
  }, world.globals);
});

test('overlay: first-run hint on the first three opens per device, then gone; English-only line first in another language', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    const tips = [];
    for (let i = 0; i < 4; i++) {
      const o = await openOverlay(world);
      const tip = o.overlay.querySelector('.drive-tip');
      tips.push(tip.hidden ? null : tip.textContent);
      o.d.close({ quiet: true }); o.restore();
    }
    assert.deepEqual(tips, [EN.drive.first_run_hint, EN.drive.first_run_hint, EN.drive.first_run_hint, null]);
  }, world.globals);
  await withDom(async () => {
    const DE = loadBundle('de');
    const o = await openOverlay(world, { lang: 'de' });
    try { assert.equal(o.overlay.querySelector('.drive-tip').textContent, DE.drive.english_only); }
    finally { o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: "Got it — go on?" only for holds ≥ 600 ms; live captions dim the tail; narrator sentences italic', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    const o = await openOverlay(world);
    try {
      assert.equal(GOTIT_MIN_HOLD_MS, 600);
      o.ws.msg({ type: 'state', state: 'hearing' });
      o.ws.msg({ type: 'eot_soft', p: 0.6, hold_ms: 300 });
      assert.equal(o.word(), EN.drive.state_hearing);
      o.ws.msg({ type: 'eot_soft', p: 0.6, hold_ms: 800 });
      assert.equal(o.word(), EN.drive.state_gotit_goon);
      o.ws.msg({ type: 'partial', committed: 'What is on my', tail: 'calendar', stable: 4, t: 1.2 });
      const you = o.overlay.querySelector('.drive-you');
      assert.equal(you.querySelector('.drive-committed').textContent, 'What is on my');
      assert.equal(you.querySelector('.drive-tail').textContent, 'calendar');
      o.ws.msg({ type: 'transcript', text: 'What is on my calendar', merged: false, message_id: 'u-1', upto: 32000 });
      assert.equal(you.querySelector('.drive-tail'), null);
      o.ws.msg({ type: 'ready', proto: 3, sid: 'aaaa', sample_rate_in: 16000 });
      sayWithAudio(o.ws, { seq: 3, text: 'She leans in.', role: 'narrator', mid: 'm-3' });
      sayWithAudio(o.ws, { seq: 4, text: 'Tell me more.', mid: 'm-3' });
      assert.equal(o.overlay.querySelectorAll('.drive-them .drive-sentence').length, 0, 'a caption waits for its audio to play');
      world.render(450);
      o.ws.msg({ type: 'say', gen: 0, seq: 4, text: 'Tell me more.', kind: 'reply', message_id: 'm-3', t: 2 });  // re-sent: no duplicate
      const spans = [...o.overlay.querySelectorAll('.drive-them .drive-sentence')];
      assert.deepEqual(spans.map((s) => s.textContent.trim()), ['She leans in.', 'Tell me more.']);
      assert.ok(spans[0].classList.contains('drive-narrator'));
      assert.ok(!spans[1].classList.contains('drive-narrator'));
    } finally { o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: All settings… opens Settings over the voice chat with the mic paused — the socket stays (L8)', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    let opened = 0;
    const o = await openOverlay(world, { openSettings: () => { opened += 1; } });
    try {
      o.overlay.querySelector('.drive-gear').click();
      await settle(20);
      const all = document.querySelector('#drive-sheet-all');
      assert.ok(all, 'the quick sheet is open');
      all.click();
      assert.equal(opened, 1);
      assert.equal(o.ws.closed, null, 'never hangs up');
      assert.ok(document.querySelector('.drive-overlay'), 'the voice chat is still there');
      assert.equal(o.ws.sent.at(-1).type, 'pause');
      assert.equal(document.querySelector('.drive-sheet'), null, 'the sheet closed');
    } finally { o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: Hang up when quiet → quiet_check → hang-up 20 s later with "Voice chat ended."', dom, async (t) => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, hangupQuietMin: 5, firstRunOpens: 3 }));
    t.mock.timers.enable({ apis: ['setInterval', 'setTimeout', 'Date'], now: 1_000_000 });
    const toasts = [];
    const o = await openOverlay(world, { toast: toasts });
    try {
      advance(t, 5 * 60000);
      assert.equal(o.ws.frames('quiet_check').length, 1);
      o.ws.msg({ type: 'say', gen: 0, seq: 9, text: 'Still there?', kind: 'notice', t: 1 });
      advance(t, QUIET_GRACE_MS + 1000);
      assert.equal(document.querySelector('.drive-overlay'), null, 'hung up');
      assert.deepEqual(toasts, [EN.drive.ended]);
      assert.equal(o.ws.closed, 1000);
    } finally { t.mock.timers.reset(); o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: speaking after "Still there?" keeps the voice chat open', dom, async (t) => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, hangupQuietMin: 5, firstRunOpens: 3 }));
    t.mock.timers.enable({ apis: ['setInterval', 'setTimeout', 'Date'], now: 1_000_000 });
    const o = await openOverlay(world);
    try {
      advance(t, 5 * 60000);
      assert.equal(o.ws.frames('quiet_check').length, 1);
      advance(t, 5000);
      o.ws.msg({ type: 'state', state: 'hearing' });
      advance(t, QUIET_GRACE_MS + 1000);
      assert.ok(document.querySelector('.drive-overlay'), 'still open');
    } finally { t.mock.timers.reset(); o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: the server speaking "Still there?" does not cancel its own check (review M3)', dom, async (t) => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, hangupQuietMin: 5, firstRunOpens: 3 }));
    t.mock.timers.enable({ apis: ['setInterval', 'setTimeout', 'Date'], now: 1_000_000 });
    const toasts = [];
    const o = await openOverlay(world, { toast: toasts });
    try {
      advance(t, 5 * 60000);
      assert.equal(o.ws.frames('quiet_check').length, 1);
      // What the real TTS worker reports while it plays the notice.
      o.ws.msg({ type: 'say', gen: 0, seq: 9, text: 'Still there?', kind: 'notice', t: 1 });
      o.ws.msg({ type: 'state', state: 'speaking' });
      replyAudio(o.ws, { seq: 9, text: 'Still there?', kind: 'notice' });
      advance(t, 2000);
      o.ws.msg({ type: 'state', state: 'listening' });
      advance(t, QUIET_GRACE_MS);
      assert.equal(document.querySelector('.drive-overlay'), null, 'hung up');
      assert.deepEqual(toasts, [EN.drive.ended]);
    } finally { t.mock.timers.reset(); o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: with the microphone paused, Hang up when quiet ends the chat without asking (review L5)', dom, async (t) => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, hangupQuietMin: 5, firstRunOpens: 3 }));
    t.mock.timers.enable({ apis: ['setInterval', 'setTimeout', 'Date'], now: 1_000_000 });
    const toasts = [];
    const o = await openOverlay(world, { toast: toasts });
    try {
      o.overlay.querySelector('.drive-pause').click();
      assert.equal(o.ws.frames('pause').length, 1);
      advance(t, 4 * 60000);
      assert.ok(document.querySelector('.drive-overlay'), 'still open at 4 min');
      advance(t, 60000 + 1000);
      assert.equal(o.ws.frames('quiet_check').length, 0, 'never asks someone who cannot answer');
      assert.equal(document.querySelector('.drive-overlay'), null, 'hung up');
      assert.deepEqual(toasts, [EN.drive.ended]);
    } finally { t.mock.timers.reset(); o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: a reply after a reconnect shows its captions although its seq repeats (review L1)', dom, async (t) => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, firstRunOpens: 3 }));
    t.mock.timers.enable({ apis: ['setInterval', 'setTimeout', 'Date'], now: 3_000_000 });
    const o = await openOverlay(world);
    try {
      o.ws.msg({ type: 'ready', proto: 3, sid: 'aaaa', sample_rate_in: 16000 });
      sayWithAudio(o.ws, { seq: 1, text: 'Alpha one.', mid: 'm1' });
      o.ws.msg({ type: 'say_end', gen: 0, message_id: 'm1', last_seq: 1 });
      world.render(300);
      o.ws.drop(1006);
      advance(t, 3000);
      const ws = world.ws();
      assert.notEqual(ws, o.ws, 'reconnected');
      ws.open();
      ws.msg({ type: 'ready', proto: 3, sid: 'bbbb', sample_rate_in: 16000 });
      sayWithAudio(ws, { seq: 1, text: 'Beta two.', mid: 'm2' });
      world.render(150);
      const lines = [...document.querySelectorAll('.drive-sentence')].map((n) => n.textContent.trim());
      assert.deepEqual(lines, ['Alpha one.', 'Beta two.']);
      // The same session re-sending a sentence is still shown once.
      ws.msg({ type: 'say', gen: 0, seq: 1, text: 'Beta two.', kind: 'reply', message_id: 'm2', t: 1 });
      assert.equal(document.querySelectorAll('.drive-sentence').length, 2);
    } finally { t.mock.timers.reset(); o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: a reply resumed after a drop restarts the cut sentence on the SAME caption line (§5.5, no repeat on screen)', dom, async (t) => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, firstRunOpens: 3 }));
    t.mock.timers.enable({ apis: ['setInterval', 'setTimeout', 'Date'], now: 3_000_000 });
    const o = await openOverlay(world);
    try {
      o.ws.msg({ type: 'ready', proto: 3, sid: 'aaaa', sample_rate_in: 16000 });
      o.ws.msg({ type: 'transcript', text: 'What is on today?', merged: false, message_id: 'u1' });
      sayWithAudio(o.ws, { seq: 1, text: 'Point one.', mid: 'r1' });
      o.ws.msg({ type: 'audio_end', gen: 0, seq: 1, kind: 'reply' });
      sayWithAudio(o.ws, { seq: 2, text: 'Point two is cut.', mid: 'r1', ms: 120 });   // no audio_end: cut mid-way
      world.render(300);
      const lines = () => [...document.querySelectorAll('.drive-sentence')].map((n) => n.textContent.trim());
      assert.deepEqual(lines(), ['Point one.', 'Point two is cut.']);
      o.ws.drop(1006);
      advance(t, 3000);
      const ws = world.ws();
      assert.notEqual(ws, o.ws, 'reconnected');
      ws.open();
      ws.msg({ type: 'ready', proto: 3, sid: 'bbbb', sample_rate_in: 16000 });
      // The server resumes from the cut sentence (after_partial = 1): it is seq 1 of the new session.
      sayWithAudio(ws, { seq: 1, text: 'Point two is cut.', mid: 'r1', ms: 400 });
      sayWithAudio(ws, { seq: 2, text: 'Point three.', mid: 'r1', ms: 200 });
      world.render(700);
      assert.deepEqual(lines(), ['Point one.', 'Point two is cut.', 'Point three.'], 'the restarted sentence is not shown twice');
      // A different reply in the SAME session may repeat words on purpose: shown again.
      ws.msg({ type: 'transcript', text: 'Again?', merged: false, message_id: 'u2' });
      sayWithAudio(ws, { seq: 3, text: 'Point three.', mid: 'r2', ms: 200 });
      sayWithAudio(ws, { seq: 4, text: 'Point three.', mid: 'r2', ms: 200 });
      world.render(500);
      assert.deepEqual(lines(), ['Point three.', 'Point three.'], 'same-session repeats are two sentences');
    } finally { t.mock.timers.reset(); o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: a slow model load (no `ready` for 12 s) is not a dead link (review L2)', dom, async (t) => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    t.mock.timers.enable({ apis: ['setInterval', 'setTimeout', 'Date'], now: 4_000_000 });
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: async () => ({ getTracks: () => [{ stop() {} }] }), enumerateDevices: async () => [] },
    });
    const fs = fakeShell({ lang: 'en' });
    const prev = bindShell(fs.shell);
    const button = document.createElement('button');
    document.body.append(button);
    const d = initDrive(button, {
      getThread: () => ({ id: 'th-1', bot_id: 'nova', title: 'Evening' }), getBot: () => ({ id: 'nova', name: 'Nova' }),
      isLocked: () => false, toast: () => {}, getAvatar: () => ({ src: '', letter: 'S' }),
      voiceStatus: () => ({ enabled: true, profiles: [], builtin_voices: [] }), openDriveSettings: () => {}, history: fs.shell.history,
    });
    try {
      button.click();
      await settle();
      const ws = world.ws();
      ws.readyState = 1;
      ws.send = (x) => { ws.sent.push(typeof x === 'string' ? JSON.parse(x) : x); };   // loading: no pongs
      ws.onopen();
      advance(t, 12000);
      assert.equal(world.sockets.length, 1, 'no reconnect while the models load');
      assert.ok(ws.frames('ping').length >= 5, 'pings still go out');
      assert.equal(ws.closed, null);
      const overlay = document.querySelector('.drive-overlay');
      assert.equal(overlay.dataset.state, 'connecting');
      ws.msg({ type: 'ready', sample_rate_in: 16000 });
      assert.equal(overlay.dataset.state, 'listening');
      advance(t, 9000);
      assert.equal(world.sockets.length, 2, 'after `ready`, 8 s of silence is a dead link again');
    } finally { t.mock.timers.reset(); d.close({ quiet: true }); bindShell(prev); }
  }, world.globals);
});

test('overlay: Follow-up back to "the bot\'s default" tells the open session with null, once (review L4, Amendment T7a-7)', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, firstRunOpens: 3 }));
    const o = await openOverlay(world);
    try {
      assert.equal('follow_up' in o.ws.frames('start').at(-1), false, 'a fresh device sends no follow_up');
      saveDriveSettings(localStorage, { followUp: false });
      assert.equal(o.ws.frames('start').at(-1).follow_up, false);
      saveDriveSettings(localStorage, { followUp: null });
      assert.equal(o.ws.frames('start').at(-1).follow_up, null, 'the session drops its explicit value');
      saveDriveSettings(localStorage, { patience: 'patient' });
      assert.equal('follow_up' in o.ws.frames('start').at(-1), false, 'then absent again');
    } finally { o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: the Voice lab\'s Reset reaches the open session once as start.rt.learned_reset (T7a-6)', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, firstRunOpens: 3, learned: { probeMs: 300, holdMs: 700 } }));
    const o = await openOverlay(world);
    try {
      assert.deepEqual(o.ws.frames('start').at(-1).rt.learned, { probe_ms: 300, hold_ms: 700 }, 'the blob seeds the server');
      saveDriveSettings(localStorage, { learned: null, learnedReset: true });     // the lab's Reset
      const resets = o.ws.frames('start').filter((f) => f.rt && f.rt.learned_reset === true);
      assert.equal(resets.length, 1, 'sent once');
      assert.equal(JSON.parse(localStorage.getItem('dispatch-drive')).learnedReset, false, 'then cleared');
      saveDriveSettings(localStorage, { patience: 'patient' });
      assert.equal('learned_reset' in o.ws.frames('start').at(-1).rt, false, 'never again');
    } finally { o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: the quick sheet re-reads the roster before it shows "{Bot}\'s voice" (review L6)', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, firstRunOpens: 3 }));
    let current = { id: 'nova', name: 'Nova', voice: '' };
    let reloads = 0;
    const o = await openOverlay(world, { ctx: {
      getBot: () => current,
      voiceStatus: () => ({ enabled: true, default_voice: 'p1', profiles: [{ id: 'p1', name: 'Robin' }, { id: 'p2', name: 'Kit' }], builtin_voices: [] }),
      reloadBots: async () => { reloads += 1; current = { ...current, voice: 'p2' }; },
    } });
    try {
      o.overlay.querySelector('.drive-gear').click();
      await settle(20);
      assert.equal(reloads, 1);
      assert.equal(document.querySelector('#drive-sheet-voice').value, 'p2', 'the value saved elsewhere, not the stale one');
    } finally { o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: a dropped link retries, says "Waiting for signal" after 60 s, and gets back to listening (§5.5)', dom, async (t) => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    t.mock.timers.enable({ apis: ['setInterval', 'setTimeout', 'Date'], now: 2_000_000 });
    const o = await openOverlay(world);
    try {
      o.ws.drop(1006);
      assert.equal(o.overlay.dataset.state, 'offline');
      // Keep failing for a minute.
      for (let i = 0; i < 20 && Date.now() < 2_000_000 + WAITING_AFTER_MS + 1000; i++) {
        advance(t, 8000);
        world.ws().drop(1006);
      }
      assert.equal(o.overlay.dataset.state, 'waiting');
      assert.equal(o.word(), EN.drive.state_waiting);
      assert.match(o.overlay.querySelector('.drive-hint').textContent, new RegExp(EN.drive.hint_waiting.slice(0, 9)));
      const n = world.sockets.length;
      advance(t, WAITING_RETRY_MS);
      assert.equal(world.sockets.length, n + 1, 'retries every 15 s while waiting');
      const ws = world.ws();
      ws.open();
      ws.msg({ type: 'ready' });
      assert.equal(o.overlay.dataset.state, 'listening');
    } finally { t.mock.timers.reset(); o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: the first-run tip is hidden while offline or waiting and back with listening (V2)', dom, async (t) => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    t.mock.timers.enable({ apis: ['setInterval', 'setTimeout', 'Date'], now: 2_000_000 });
    const o = await openOverlay(world);
    const tip = () => o.overlay.querySelector('.drive-tip');
    try {
      assert.equal(tip().hidden, false, 'shown on a first open');
      o.ws.drop(1006);
      assert.equal(o.overlay.dataset.state, 'offline');
      assert.equal(tip().hidden, true, '"talk over me" is no help while offline');
      for (let i = 0; i < 20 && Date.now() < 2_000_000 + WAITING_AFTER_MS + 1000; i++) {
        advance(t, 8000);
        world.ws().drop(1006);
      }
      assert.equal(o.overlay.dataset.state, 'waiting');
      assert.equal(tip().hidden, true, 'nor while waiting for signal');
      advance(t, WAITING_RETRY_MS);
      const ws = world.ws();
      ws.open();
      ws.msg({ type: 'ready' });
      assert.equal(o.overlay.dataset.state, 'listening');
      assert.equal(tip().hidden, false, 'back with listening until the first transcript');
      ws.msg({ type: 'transcript', text: 'Hello there', message_id: 'u-1' });
      assert.equal(tip().hidden, true, 'gone after the first transcript');
      ws.drop(1006);
      advance(t, 1000);
      world.ws().open();
      world.ws().msg({ type: 'ready' });
      assert.equal(tip().hidden, true, 'and stays gone');
    } finally { t.mock.timers.reset(); o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: 1013 "busy" stops with its own words, not a retry loop', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    const o = await openOverlay(world);
    try {
      const n = world.sockets.length;
      o.ws.drop(1013, 'Voice is busy on this server (3 chats open)');
      await settle();
      assert.equal(o.overlay.dataset.state, 'stopped');
      assert.equal(o.overlay.querySelector('.drive-notice').textContent, EN.drive.err_busy);
      assert.equal(world.sockets.length, n, 'no reconnect');
    } finally { o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: Esc hangs up, Space toggles the mic, "Hang up" is the button, the shell history entry is popped', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    const toasts = [];
    const o = await openOverlay(world, { toast: toasts });
    try {
      assert.equal(o.overlay.querySelector('.drive-end').textContent, EN.drive.end);
      assert.equal(document.activeElement, o.overlay.querySelector('.drive-ring'), 'focus starts on the picture');
      assert.equal(o.fs.log.history.length, 1);
      o.overlay.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true }));
      assert.equal(o.ws.sent.at(-1).type, 'pause');
      o.overlay.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true }));
      assert.equal(o.ws.sent.at(-1).type, 'start');
      o.overlay.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
      assert.equal(document.querySelector('.drive-overlay'), null);
      assert.deepEqual(toasts, [EN.drive.ended]);
      assert.equal(document.activeElement, o.button, 'focus returns to the microphone button');
      assert.equal(o.fs.log.history[0].popped, true);
      assert.match(world.workletUrl, /voice-worklet\.js$/, 'the worklet comes from the add-on, not /static/');
    } finally { o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: with the focus on the picture (where it starts) Space is the microphone, never the tap; U is Undo', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    const o = await openOverlay(world);
    try {
      const ring = o.overlay.querySelector('.drive-ring');
      assert.equal(document.activeElement, ring);
      o.ws.msg({ type: 'ready', proto: 3, sid: 'aaaa', sample_rate_in: 16000 });
      o.ws.msg({ type: 'state', state: 'speaking' });
      sayWithAudio(o.ws, { seq: 1, text: 'A reply that is playing.', ms: 2000 });
      world.render(200);
      const before = o.ws.sent.length;
      const down = new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true, cancelable: true });
      ring.dispatchEvent(down);
      const up = new KeyboardEvent('keyup', { key: ' ', code: 'Space', bubbles: true, cancelable: true });
      ring.dispatchEvent(up);
      assert.equal(down.defaultPrevented, true, 'no native button activation on keydown…');
      assert.equal(up.defaultPrevented, true, '…nor on keyup (the tap would interrupt the reply)');
      assert.deepEqual(o.ws.sent.slice(before).map((f) => f.type).filter((x) => x !== 'heard'), ['pause']);
      assert.equal(o.ws.frames('stop_playback').length, 0);
      assert.equal(o.word(), EN.drive.paused_speaking);
      ring.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true, cancelable: true }));
      assert.equal(o.ws.sent.at(-1).type, 'start');
      // U on the picture takes the last transcript back.
      o.ws.msg({ type: 'transcript', text: 'Never mind that.', merged: false, message_id: 'u-9' });
      assert.equal(o.overlay.querySelector('.drive-undo').hidden, false);
      ring.dispatchEvent(new KeyboardEvent('keydown', { key: 'u', bubbles: true, cancelable: true }));
      const undo = o.ws.frames('undo').at(-1);
      assert.ok(undo, 'U sends undo');
      assert.equal(undo.message_id, 'u-9');
      // Another button keeps its own Space (Hang up is not a shortcut target).
      const end = o.overlay.querySelector('.drive-end');
      const n = o.ws.sent.length;
      const ev = new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true, cancelable: true });
      end.dispatchEvent(ev);
      assert.equal(ev.defaultPrevented, false);
      assert.equal(o.ws.sent.length, n);
    } finally { o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

// ------------------------------------------- part 2: the audio path (drive-audio.js) ---

/** The engine's monotonic clock follows the mocked Date (performance is not mocked by node). */
const mockPerf = () => ({ timeOrigin: 0, now: () => Date.now() });

test('overlay: the socket negotiates v3 (C.1): v=3, the voice chat\'s stream id, codec lists best first', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    const o = await openOverlay(world);
    try {
      const u = new URL(o.ws.url);
      assert.equal(u.searchParams.get('v'), '3');
      assert.match(u.searchParams.get('s'), /^[A-Za-z0-9-]{8,64}$/);
      assert.equal(u.searchParams.get('up'), 'pcm16', 'no WebCodecs here: PCM16 only');
      assert.equal(u.searchParams.get('down'), 'pcm16');
      assert.match(u.searchParams.get('d'), /^[A-Za-z0-9-]{8,64}$/);
      assert.ok(world.players.length === 1, 'the player worklet node exists');
    } finally { o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: a slow link hands the reply to the phone voice and says so on the screen (§5.4, §7.4)', dom, async (t) => {
  const world = fakeAudioWorld();
  const synth = fakeSynth([{ voiceURI: 'v-en', name: 'English', lang: 'en-US', localService: true, default: true }]);
  await withDom(async () => {
    localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, firstRunOpens: 3 }));
    t.mock.timers.enable({ apis: ['setInterval', 'setTimeout', 'Date'], now: 5_000_000 });
    const o = await openOverlay(world);
    try {
      o.ws.msg({ type: 'ready', proto: 3, sid: 'aaaa', sample_rate_in: 16000, codec_in: 'pcm16', codec_out: 'pcm16' });
      const line = o.overlay.querySelector('.drive-link');
      assert.equal(line.hidden, true);
      o.ws.msg({ type: 'state', state: 'speaking' });
      o.ws.msg({ type: 'say', gen: 0, seq: 1, text: 'This sentence is late.', kind: 'reply', message_id: 'm1', t: 1 });
      advance(t, 6500, 250);
      const link = o.ws.frames('link');
      assert.equal(link.length, 1);
      assert.equal(link[0].mode, 'text');
      assert.equal(link[0].why, 'deadline');
      assert.equal(line.hidden, false);
      assert.equal(line.textContent, EN.drive.link_phone);
      assert.deepEqual(synth.spoken.map((u) => u.text), ['This sentence is late.']);
      synth.start();
      advance(t, 300, 250);
      assert.deepEqual([...document.querySelectorAll('.drive-sentence')].map((n) => n.textContent.trim()), ['This sentence is late.'],
        'the caption appears when the phone starts it');
      // The tap interrupts the phone voice too, and reports what was heard.
      o.overlay.querySelector('.drive-ring').click();
      const stop = o.ws.frames('stop_playback').at(-1);
      assert.equal(stop.heard.source, 'phone');
    } finally { t.mock.timers.reset(); o.d.close({ quiet: true }); o.restore(); }
  }, { ...world.globals, speechSynthesis: synth, SpeechSynthesisUtterance: FakeUtterance, performance: mockPerf() });
});

test('overlay: a duck shows "Hearing you" over a held reply; resume brings "Speaking" back (§4.6)', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    const o = await openOverlay(world);
    try {
      o.ws.msg({ type: 'ready', proto: 3, sid: 'aaaa', sample_rate_in: 16000 });
      o.ws.msg({ type: 'state', state: 'speaking' });
      sayWithAudio(o.ws, { seq: 1, text: 'A long answer.', ms: 2000 });
      world.render(200);
      assert.equal(o.word(), EN.drive.state_speaking);
      o.ws.msg({ type: 'duck', on: true, gen: 0 });
      assert.equal(o.word(), EN.drive.state_hearing);
      assert.equal(o.overlay.dataset.state, 'hearing');
      assert.equal(world.players[0].core.paused, true, 'the player is held');
      o.ws.msg({ type: 'resume', gen: 0, rewind: false });
      assert.equal(o.word(), EN.drive.state_speaking);
      assert.equal(world.players[0].core.paused, false);
    } finally { o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: the picture tap stops a playing reply and the frame carries what was heard (C.3a)', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    const o = await openOverlay(world);
    try {
      o.ws.msg({ type: 'ready', proto: 3, sid: 'aaaa', sample_rate_in: 16000 });
      sayWithAudio(o.ws, { seq: 1, text: 'Stop me here.', ms: 1000 });
      world.render(400);
      o.overlay.querySelector('.drive-ring').click();
      const stop = o.ws.frames('stop_playback').at(-1);
      assert.equal(stop.heard.gen, 0);
      assert.equal(stop.heard.seq, 1);
      assert.equal(stop.heard.source, 'server');
      assert.ok(stop.heard.played_ms > 200, `played_ms ${stop.heard.played_ms}`);
    } finally { o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: speech while waiting for signal gets "No signal. I\'ll send that…"; the reconnect start carries the cursor (§5.5)', dom, async (t) => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, firstRunOpens: 3 }));
    t.mock.timers.enable({ apis: ['setInterval', 'setTimeout', 'Date'], now: 6_000_000 });
    const o = await openOverlay(world);
    try {
      o.ws.msg({ type: 'ready', proto: 3, sid: 'aaaa', sample_rate_in: 16000, gate: { pre_roll_ms: 300, hangover_s: 2.5 } });
      assert.ok(world.mic, 'the mic worklet exists');
      assert.ok(world.mic.port.posted.some((m) => m.gate && m.gate.on), 'the gate from ready reached the worklet');
      o.ws.msg({ type: 'transcript', text: 'hello', merged: false, message_id: 'u1', upto: 48000 });
      const stream = new URL(o.ws.url).searchParams.get('s');
      // The link dies; after 60 s offline the screen says "Waiting for signal".
      world.sockets.forEach((w) => { w.readyState = 3; });
      o.ws.drop(1006);
      for (let i = 0; i < 6; i++) { advance(t, 10000, 500); const w = world.ws(); if (w.readyState === 0) w.drop(1006); }
      advance(t, 2000, 500);
      assert.equal(o.overlay.dataset.state, 'waiting');
      world.mic.port.onmessage({ data: { speech: true } });
      const n = o.overlay.querySelector('.drive-notice:not(.drive-link)');
      assert.equal(n.hidden, false);
      assert.equal(n.textContent, EN.drive.offline_said);
      // Signal is back: the new socket's start resumes the stream from what the server has.
      advance(t, 16000, 500);
      const ws = world.ws();
      ws.open();
      const start = ws.frames('start')[0];
      assert.deepEqual(start.cursor, { stream, from: 48000 });
      ws.msg({ type: 'ready', proto: 3, sid: 'bbbb', sample_rate_in: 16000 });
      assert.equal(o.overlay.dataset.state, 'listening');
    } finally { t.mock.timers.reset(); o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: the ready tone and "Listening" wait until the microphone is capturing (a slow worklet load on cellular)', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, firstRunOpens: 3 }));
    let release;
    const gate = new Promise((r) => { release = r; });
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: async () => { await gate; return { getTracks: () => [{ stop() {} }] }; }, enumerateDevices: async () => [] },
    });
    const fs = fakeShell({ lang: 'en' });
    const prev = bindShell(fs.shell);
    const button = document.createElement('button');
    document.body.append(button);
    const d = initDrive(button, {
      getThread: () => ({ id: 'th-1', bot_id: 'nova', title: 'Evening' }), getBot: () => ({ id: 'nova', name: 'Nova' }),
      isLocked: () => false, toast: () => {}, getAvatar: () => ({ src: '', letter: 'S' }),
      voiceStatus: () => ({ enabled: true, profiles: [], builtin_voices: [] }), openDriveSettings: () => {}, history: fs.shell.history,
    });
    try {
      button.click();
      await settle();
      const ws = world.ws();
      ws.open();
      const tones = world.oscillators.length;
      ws.msg({ type: 'ready', proto: 3, sid: 'aaaa', sample_rate_in: 16000 });
      ws.msg({ type: 'state', state: 'listening' });
      const overlay = document.querySelector('.drive-overlay');
      assert.equal(overlay.dataset.state, 'connecting', 'not "Listening" while the mic is not capturing');
      assert.equal(world.oscillators.length, tones, 'no ready tone yet');
      release();
      await settle();
      assert.equal(overlay.dataset.state, 'listening');
      assert.deepEqual(world.freqs().slice(tones), EARCONS.ready, 'the ready tone, once the mic is live');
    } finally { d.close({ quiet: true }); bindShell(prev); }
  }, world.globals);
});

test('prefetchAudioWorklets: both worklet modules once, with the entry\'s ?v= (warm cache for a slow first open)', async () => {
  const urls = [];
  const fakeFetch = (u) => { urls.push(String(u)); return Promise.resolve({ ok: true }); };
  const { prefetchAudioWorklets } = await import('../../src/dispatch_voice/static/drive.js');
  assert.equal(prefetchAudioWorklets(fakeFetch), true);
  assert.equal(prefetchAudioWorklets(fakeFetch), false, 'once per page');
  assert.deepEqual(urls.map((u) => u.split('/').pop().split('?')[0]), ['voice-worklet.js', 'voice-player-worklet.js']);
});

// ------------------------------------------ review T7b round 2: overlay regressions ---

for (const variant of ['v3', 'v2']) {
  test(`overlay (${variant}): after an Undo the next reply plays, is captioned and reports playback_done (review-T7b H2)`, dom, async (t) => {
    const world = fakeAudioWorld();
    await withDom(async () => {
      localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, firstRunOpens: 3 }));
      t.mock.timers.enable({ apis: ['setInterval', 'setTimeout', 'Date'], now: 5_000_000 });
      const o = await openOverlay(world);
      try {
        if (variant === 'v3') o.ws.msg({ type: 'ready', proto: 3, sid: 'aaaa', sample_rate_in: 16000 });
        o.ws.msg({ type: 'transcript', text: 'Wrong question', merged: false, message_id: 'u1' });
        const undoBtn = o.overlay.querySelector('.drive-undo');
        assert.equal(undoBtn.hidden, false);
        undoBtn.click();
        assert.equal(o.ws.frames('undo').length, 1);
        // The session's _undo: barge_in {gen + 1}, then `undone`.
        o.ws.msg({ type: 'barge_in', gen: 1 });
        o.ws.msg({ type: 'undone', message_id: 'u1' });
        // The next question; its answer comes at gen 1 (gen moves only on barge_in).
        o.ws.msg({ type: 'transcript', text: 'Right question', merged: false, message_id: 'u2' });
        o.ws.msg({ type: 'state', state: 'speaking' });
        if (variant === 'v3') {
          sayWithAudio(o.ws, { seq: 1, gen: 1, text: 'The answer.', mid: 'r2', ms: 400 });
          o.ws.msg({ type: 'audio_end', gen: 1, seq: 1, kind: 'reply' });
          o.ws.msg({ type: 'say_end', gen: 1, message_id: 'r2', last_seq: 1 });
        } else {
          replyAudio(o.ws, { seq: 1, gen: 1, text: 'The answer.' });
        }
        world.render(800);
        await settle();
        const flushes = world.players[0].port.posted.filter((m) => m.type === 'flush').map((m) => m.gen);
        assert.ok(flushes.every((g) => g <= 1), `never flushed past the server's gen (${flushes})`);
        const lines = [...document.querySelectorAll('.drive-sentence')].map((n) => n.textContent.trim());
        assert.deepEqual(lines, ['The answer.'], 'played, so captioned');
        assert.ok(o.ws.frames('playback_done').some((f) => f.gen === 1), 'playback_done for gen 1');
      } finally { t.mock.timers.reset(); o.d.close({ quiet: true }); o.restore(); }
    }, world.globals);
  });
}

test('overlay: "Tap to continue" only when the server says it answers replay_last (review-T7b M1, Amendment T7a-6)', dom, async (t) => {
  for (const cap of [false, true]) {
    const world = fakeAudioWorld();
    await withDom(async () => {
      localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, firstRunOpens: 3 }));
      t.mock.timers.enable({ apis: ['setInterval', 'setTimeout', 'Date'], now: 7_000_000 });
      const o = await openOverlay(world);
      try {
        const ready = { type: 'ready', proto: 3, sid: 'aaaa', sample_rate_in: 16000 };
        if (cap) ready.replay_last = true;
        o.ws.msg(ready);
        await settle();
        sayWithAudio(o.ws, { seq: 1, text: 'A long answer.', mid: 'r1', ms: 2000 });
        o.ws.msg({ type: 'state', state: 'speaking' });
        world.render(400);
        o.overlay.querySelector('.drive-ring').click();           // tap = stop
        o.ws.msg({ type: 'barge_in', gen: 1 });
        o.ws.msg({ type: 'state', state: 'listening' });
        const hint = o.overlay.querySelector('.drive-hint').textContent;
        o.overlay.querySelector('.drive-ring').click();
        const sent = o.ws.sent.filter((f) => f && f.type && !['ping', 'heard'].includes(f.type)).map((f) => f.type);
        if (cap) {
          assert.equal(hint, EN.drive.hint_resume_after_stop);
          assert.equal(sent.at(-1), 'replay_last');
        } else {
          assert.notEqual(hint, EN.drive.hint_resume_after_stop, 'no promise the server cannot keep');
          assert.ok(!sent.includes('replay_last'), `the tap does its normal thing (${sent.slice(-3)})`);
          assert.equal(sent.at(-1), 'pause');
        }
      } finally { t.mock.timers.reset(); o.d.close({ quiet: true }); o.restore(); }
    }, world.globals);
  }
});


// ------------------------------------------------ final-review fixes ---

test('pure: the server ending an idle chat (4002) hangs up without a retry (FR-hardening L3)', () => {
  assert.deepEqual(reconnectDecision(4002, 'quiet'), { retry: false, reason: 'quiet' });
});

test('pure: hints follow the device settings (FR-ux L2) and the pinned-without-voice notice (L3)', () => {
  assert.equal(hintKey('connecting', {}), 'drive.hint_wait_tone');
  assert.equal(hintKey('connecting', { sounds: false }), 'drive.hint_wait_listening');
  assert.equal(firstRunTipKey(0, 'en'), 'drive.first_run_hint');
  assert.equal(firstRunTipKey(1, 'en', { bargeIn: false }), 'drive.first_run_hint_tap');
  assert.equal(firstRunTipKey(0, 'de', { bargeIn: false }), 'drive.english_only');
  assert.equal(linkStateNoticeKey({ degraded: true, notice: 'text', pinned: true }), 'drive.link_no_phone_voice');
  assert.equal(linkStateNoticeKey({ degraded: true, notice: 'text', pinned: false }), 'drive.link_text');
  assert.equal(driveAllowed({ locked: false, voiceOff: true, thread: null, bot: null, secure: true }).whyKey,
    'drive.button_voice_off', 'voice switched off says so, not "unlock DisPatch" (V3 #6)');
});

test('overlay: named "Voice chat with <bot>"; Mute / Unmute with no pressed state (FR-ux L8, M5)', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    const o = await openOverlay(world);
    try {
      assert.equal(o.overlay.getAttribute('aria-label'), 'Voice chat with Nova');
      assert.equal(o.overlay.hasAttribute('aria-labelledby'), false);
      const btn = o.overlay.querySelector('.drive-pause');
      assert.match(btn.textContent, new RegExp(EN.drive.pause));
      assert.equal(EN.drive.pause, 'Mute');
      assert.equal(btn.hasAttribute('aria-pressed'), false);
      btn.click();
      assert.match(btn.textContent, new RegExp(EN.drive.resume));
      assert.equal(EN.drive.resume, 'Unmute');
      assert.equal(btn.hasAttribute('aria-pressed'), false, 'a swapped label plus aria-pressed read "Unmute, pressed"');
      assert.equal(o.word(), 'Muted');
    } finally { o.d.close(); o.restore(); }
  }, world.globals);
});

test('overlay: with Captions off Undo is still on screen, under the state word (FR-ux M3)', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    saveDriveSettings(globalThis.localStorage, { captions: false });
    const o = await openOverlay(world);
    try {
      o.ws.msg({ type: 'transcript', text: 'Remind me at five', message_id: 'u-1' });
      const undo = o.overlay.querySelector('.drive-undo');
      assert.equal(undo.hidden, false);
      assert.equal(undo.closest('[hidden]'), null, 'no hidden ancestor (it was inside the captions)');
      assert.ok(undo.closest('.drive-hero'));
      assert.equal(o.overlay.querySelector('.drive-talk').hidden, true, 'the captions themselves stay off');
      undo.click();
      assert.equal(o.ws.frames('undo').length, 1);
    } finally { o.d.close(); o.restore(); }
  }, world.globals);
});

for (const ending of ['busy', 'locked']) {
  test(`overlay: "Low signal" does not survive the end of the chat (${ending}) (1.0.1, audit M3)`, dom, async (t) => {
    const world = fakeAudioWorld();
    const synth = fakeSynth([{ voiceURI: 'v-en', name: 'English', lang: 'en-US', localService: true, default: true }]);
    await withDom(async () => {
      localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, firstRunOpens: 3 }));
      t.mock.timers.enable({ apis: ['setInterval', 'setTimeout', 'Date'], now: 5_000_000 });
      const o = await openOverlay(world);
      try {
        o.ws.msg({ type: 'ready', proto: 3, sid: 'aaaa', sample_rate_in: 16000, codec_in: 'pcm16', codec_out: 'pcm16' });
        o.ws.msg({ type: 'state', state: 'speaking' });
        o.ws.msg({ type: 'say', gen: 0, seq: 1, text: 'This sentence is late.', kind: 'reply', message_id: 'm1', t: 1 });
        advance(t, 6500, 250);
        const line = o.overlay.querySelector('.drive-link');
        assert.equal(line.hidden, false, 'the slow link handed the reply to the phone voice');
        assert.equal(line.textContent, EN.drive.link_phone);
        if (ending === 'busy') {
          o.ws.drop(1013, 'Voice is busy on this server (3 chats open)');
          advance(t, 300, 100);
          assert.equal(o.overlay.dataset.state, 'stopped');
          assert.equal(o.overlay.querySelector('.drive-notice:not(.drive-link)').textContent, EN.drive.err_busy);
        } else {
          o.ws.msg({ type: 'locked' });
          advance(t, 300, 100);
          assert.equal(o.overlay.dataset.state, 'locked');
        }
        assert.equal(line.hidden, true, 'the stopped screen says one thing, not two');
        assert.equal(line.textContent, '');
      } finally { t.mock.timers.reset(); o.d.close({ quiet: true }); o.restore(); }
    }, { ...world.globals, speechSynthesis: synth, SpeechSynthesisUtterance: FakeUtterance, performance: mockPerf() });
  });
}

test('overlay: Mute and Unmute are announced on the tap, once, without waiting for the server (1.0.1, audit L10)', dom, async (t) => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, firstRunOpens: 3 }));
    t.mock.timers.enable({ apis: ['setInterval', 'setTimeout', 'Date'], now: 5_000_000 });
    const o = await openOverlay(world);
    try {
      o.ws.msg({ type: 'ready', proto: 3, sid: 'aaaa', sample_rate_in: 16000, codec_in: 'pcm16', codec_out: 'pcm16' });
      o.ws.msg({ type: 'state', state: 'listening' });
      advance(t, 100, 50);
      const status = o.overlay.querySelector('.sr-only[role=status]');
      const said = [];
      const seen = () => { const x = status.textContent; if (x && said.at(-1) !== x) said.push(x); };
      const mo = new MutationObserver(seen);
      mo.observe(status, { childList: true, characterData: true, subtree: true });
      const pause = o.overlay.querySelector('.drive-pause');
      pause.click();                                     // Mute
      advance(t, 100, 50); await settle(); seen();
      assert.deepEqual(said, [EN.drive.state_paused], 'Muted is said on the tap, before any echo');
      status.textContent = '';
      o.ws.msg({ type: 'state', state: 'paused' });       // the server's echo
      advance(t, 100, 50); await settle(); seen();
      assert.equal(status.textContent, '', 'the echo does not say it again');
      pause.click();                                     // Unmute
      advance(t, 100, 50); await settle(); seen();
      assert.equal(said.at(-1), EN.drive.state_listening);
      const n = said.length;
      o.ws.msg({ type: 'state', state: 'listening' });
      advance(t, 100, 50); await settle(); seen();
      assert.equal(said.length, n, 'one announcement per tap');
      mo.disconnect();
    } finally { t.mock.timers.reset(); o.d.close({ quiet: true }); o.restore(); }
  }, world.globals);
});

test('overlay: replies shown as text appear even with Captions off; no "Low signal" for a pinned phone voice (FR-ux M2, L3)', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    // "Always use the phone's voice" on a device that has none: every reply is text.
    saveDriveSettings(globalThis.localStorage, { captions: false, slowLink: 'always_phone' });
    const o = await openOverlay(world);
    try {
      o.ws.msg({ type: 'ready', proto: 3, sid: 'aaaa', sample_rate_in: 16000 });
      await settle();
      o.ws.msg({ type: 'transcript', text: 'What is on today?', message_id: 'u-1' });
      o.ws.msg({ type: 'say', gen: 0, seq: 1, text: 'You have the dentist at three.', kind: 'reply', role: 'character', message_id: 'm-1', t: 1 });
      await settle();
      const talk = o.overlay.querySelector('.drive-talk');
      assert.equal(talk.hidden, false, 'the text IS the reply here');
      assert.match(o.overlay.querySelector('.drive-them').textContent, /dentist at three/);
      assert.ok(o.overlay.querySelector('.drive-sentence.drive-text-path.now'));
      const link = o.overlay.querySelector('.drive-link');
      assert.equal(link.textContent, EN.drive.link_no_phone_voice);
    } finally { o.d.close(); o.restore(); }
  }, world.globals);
});

test('overlay: a voice chat is activity for DisPatch\'s idle lock, plumbing is not (FR-works 1)', dom, async (t) => {
  const world = fakeAudioWorld();
  t.mock.timers.enable({ apis: ['setInterval'] });
  await withDom(async () => {
    const o = await openOverlay(world);
    let n = 0;
    bindShell({ ...o.fs.shell, activity: () => { n += 1; } });
    try {
      o.ws.msg({ type: 'state', state: 'hearing' });
      assert.equal(n, 1, 'a conversation frame is activity');
      o.ws.msg({ type: 'metrics', eot_to_transcript_ms: 300 });
      assert.equal(n, 1, 'metrics are not');
      t.mock.timers.tick(ACTIVITY_EVERY_MS + 1);
      assert.ok(n >= 2, 'an open, listening chat keeps the lock away while nobody speaks');
      o.overlay.querySelector('.drive-pause').click();           // muted: no more ticks
      const muted = n;
      t.mock.timers.tick(ACTIVITY_EVERY_MS * 2 + 1);
      assert.equal(n, muted);
    } finally { t.mock.timers.reset(); o.d.close(); o.restore(); }
  }, world.globals);
});

test('overlay: the server ending a quiet chat (4002) hangs up with "Voice chat ended" (FR-hardening L3)', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    const toast = [];
    const o = await openOverlay(world, { toast });
    try {
      const sockets = world.sockets.length;
      o.ws.drop(4002, 'quiet');
      await settle();
      assert.equal(document.querySelector('.drive-overlay'), null);
      assert.deepEqual(toast, [EN.drive.ended]);
      assert.equal(world.sockets.length, sockets, 'no reconnect');
    } finally { o.restore(); }
  }, world.globals);
});

test('overlay: the captions follow the sentence being spoken and pause after a manual scroll (FR-ux H1)', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    const proto = window.HTMLElement.prototype;
    const saved = { rect: proto.getBoundingClientRect, ch: Object.getOwnPropertyDescriptor(proto, 'clientHeight') };
    // A 100 px captions area; every sentence lies below it (as on a phone after 6 sentences).
    proto.getBoundingClientRect = function rect() {
      if (this.classList.contains('drive-talk')) return { top: 0, bottom: 100, height: 100, left: 0, right: 300 };
      if (this.classList.contains('drive-sentence')) return { top: 160, bottom: 180, height: 20, left: 0, right: 300 };
      return { top: 0, bottom: 0, height: 0, left: 0, right: 0 };
    };
    Object.defineProperty(proto, 'clientHeight', { configurable: true, get() { return this.classList.contains('drive-talk') ? 100 : 0; } });
    const o = await openOverlay(world);
    try {
      const talk = o.overlay.querySelector('.drive-talk');
      const tops = [];
      talk.scrollTo = ({ top }) => { tops.push(top); talk.scrollTop = top; };
      o.ws.msg({ type: 'ready', proto: 3, sid: 'aaaa', sample_rate_in: 16000 });
      await settle();
      o.ws.msg({ type: 'transcript', text: 'Tell me a story', message_id: 'u-1' });
      sayWithAudio(o.ws, { seq: 1, text: 'Once upon a time.', ms: 400 });
      world.render(200);
      await settle();
      assert.ok(tops.length >= 1, 'the highlighted sentence was scrolled into view');
      assert.ok(Math.abs(tops[0] - (160 - 100 / 3)) < 1, `scrolled to put it a third down, got ${tops[0]}`);
      // The listener scrolls by hand: no following for FOLLOW_PAUSE_MS.
      talk.dispatchEvent(new window.Event('wheel'));
      const before = tops.length;
      sayWithAudio(o.ws, { seq: 2, text: 'There was a dragon.', ms: 400 });
      world.render(600);
      await settle();
      assert.equal(tops.length, before, 'a hand scroll pauses the follow');
      assert.ok(FOLLOW_PAUSE_MS >= 3000);
    } finally {
      proto.getBoundingClientRect = saved.rect;
      if (saved.ch) Object.defineProperty(proto, 'clientHeight', saved.ch); else delete proto.clientHeight;
      o.d.close(); o.restore();
    }
  }, world.globals);
});

// ------------------------------------------------ the Type button (1.2.0) ---

test('overlay: Type mutes the microphone, shows the box; Enter sends `typed`; the transcript shows it; Esc closes only the box', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    const o = await openOverlay(world);
    try {
      const typeBtn = o.overlay.querySelector('.drive-type');
      const compose = o.overlay.querySelector('.drive-compose');
      assert.ok(typeBtn && compose.hidden, 'the box is hidden until asked for');
      assert.equal(o.overlay.querySelector('.drive-end').textContent, EN.drive.end);
      assert.equal(typeBtn.getAttribute('aria-pressed'), 'false');
      typeBtn.click();
      assert.equal(compose.hidden, false);
      assert.equal(o.d.active.composing, true);
      assert.equal(o.ws.sent.at(-1).type, 'pause', 'the microphone is muted first');
      assert.equal(o.overlay.dataset.state, 'paused');
      assert.equal(o.overlay.querySelector('.drive-hint').textContent, EN.drive.hint_type);
      assert.equal(typeBtn.getAttribute('aria-pressed'), 'true');
      assert.equal(o.overlay.querySelector('.drive-pause').dataset.muted, '1');
      const input = o.overlay.querySelector('.drive-compose-input');
      assert.equal(document.activeElement, input, 'the box takes the focus');
      input.value = '  what time is it?  ';
      // Shift+Enter is a new line, an IME Enter commits text: neither sends.
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', shiftKey: true, bubbles: true, cancelable: true }));
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', keyCode: 229, bubbles: true, cancelable: true }));
      assert.equal(o.ws.frames('typed').length, 0);
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
      assert.deepEqual(o.ws.frames('typed'), [{ type: 'typed', text: 'what time is it?' }]);
      assert.equal(input.value, '', 'the box is cleared for the next message');
      assert.equal(document.activeElement, input, 'and keeps the focus (a headset: keep typing)');
      // An empty box sends nothing.
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
      assert.equal(o.ws.frames('typed').length, 1);
      // The server echoes the typed text as a transcript: the caption shows it, Undo opens.
      o.ws.msg({ type: 'transcript', text: 'what time is it?', typed: true, message_id: 'u-1', merged: false });
      o.ws.msg({ type: 'state', state: 'thinking' });
      assert.match(o.overlay.querySelector('.drive-you').textContent, /what time is it\?/);
      assert.equal(o.overlay.querySelector('.drive-undo').hidden, false);
      assert.equal(o.overlay.dataset.state, 'paused', 'the mic is still muted: every live state reads paused');
      // Esc in the box closes the box, not the chat.
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
      assert.equal(compose.hidden, true);
      assert.ok(document.querySelector('.drive-overlay'), 'the chat is still open');
      assert.equal(document.activeElement, o.overlay.querySelector('.drive-ring'));
      // Unmute: the microphone is back, the box stays away.
      o.overlay.querySelector('.drive-pause').click();
      assert.equal(o.ws.sent.at(-1).type, 'start');
      assert.equal(compose.hidden, true);
      // Type again while listening: muted + box; Unmute then closes the box it opened.
      typeBtn.click();
      assert.equal(compose.hidden, false);
      o.overlay.querySelector('.drive-pause').click();
      assert.equal(compose.hidden, true);
    } finally { o.d.close(); o.restore(); }
  }, world.globals);
});

test('overlay: "Type instead of talking" opens the chat muted with the box out; Unmute keeps the box; the sheet switch works mid-chat', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    saveDriveSettings(globalThis.localStorage, { typedInput: true, firstRunOpens: 3 });
    const o = await openOverlay(world);
    try {
      const compose = o.overlay.querySelector('.drive-compose');
      assert.equal(compose.hidden, false);
      assert.equal(o.overlay.dataset.state, 'paused');
      assert.deepEqual(o.ws.sent.filter((f) => f.type === 'start' || f.type === 'pause').map((f) => f.type), ['start', 'pause']);
      assert.equal(document.activeElement, o.overlay.querySelector('.drive-ring'), 'no keyboard pops up on open');
      assert.equal(world.mic.port.posted.some((m) => m && m.enabled === false), true, 'the mic worklet is told it is off');
      o.overlay.querySelector('.drive-pause').click();          // Unmute
      assert.equal(o.overlay.dataset.state, 'listening');
      assert.equal(compose.hidden, false, 'both ways of asking stay available');
      // Esc in the box only drops the focus here.
      const input = o.overlay.querySelector('.drive-compose-input');
      input.focus();
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
      assert.equal(compose.hidden, false);
      assert.ok(document.querySelector('.drive-overlay'));
      // The switch off (from the sheet, say): the box goes away; on again: muted + box.
      saveDriveSettings(globalThis.localStorage, { typedInput: false });
      assert.equal(compose.hidden, true);
      saveDriveSettings(globalThis.localStorage, { typedInput: true });
      assert.equal(compose.hidden, false);
      assert.equal(o.overlay.dataset.state, 'paused');
    } finally { o.d.close(); o.restore(); saveDriveSettings(globalThis.localStorage, { typedInput: false }); }
  }, world.globals);
});

test('overlay: a stopped chat shows Try again / Back to chat and no Type button; a typed message before `ready` is kept with a notice', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    const o = await openOverlay(world);
    try {
      o.overlay.querySelector('.drive-type').click();
      o.ws.drop(1013, 'busy');
      assert.equal(o.overlay.dataset.state, 'stopped');
      assert.equal(o.overlay.querySelector('.drive-type').hidden, true);
      assert.equal(o.overlay.querySelector('.drive-pause').textContent, EN.drive.try_again);
      assert.equal(o.overlay.querySelector('.drive-end').textContent, EN.drive.back_to_chat);
      const input = o.overlay.querySelector('.drive-compose-input');
      input.value = 'hello';
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
      assert.equal(input.value, 'hello', 'not sent, not lost');
      assert.equal(o.overlay.querySelector('.drive-notice').textContent, EN.drive.compose_offline);
    } finally { o.d.close(); o.restore(); }
  }, world.globals);
});

// ------------------------------------------------ the Output button (1.3.0) ---

test('outputRoute / outputMenuModel: labels become routes; the saved sink is current, a gone one falls back to the first', () => {
  assert.equal(outputRoute('Headphones (Bluetooth)'), 'bluetooth');
  assert.equal(outputRoute('AirPods Pro'), 'bluetooth');
  assert.equal(outputRoute('USB Headset'), 'headset');
  assert.equal(outputRoute('Speakers (Realtek)'), 'speaker');
  const opts = [{ value: '', label: 'System default' }, { value: 'bt1', label: 'Buds' }, { value: 'sp', label: 'Speakers' }];
  const rows = outputMenuModel(opts, 'bt1');
  assert.deepEqual(rows.map((r) => [r.route, r.current]), [['default', false], ['bluetooth', true], ['speaker', false]]);
  assert.equal(outputMenuModel(opts, 'gone')[0].current, true);
});

test('overlay: no Output button where AudioContext has no setSinkId (a phone: the OS routes audio)', dom, async () => {
  const world = fakeAudioWorld();
  await withDom(async () => {
    const o = await openOverlay(world);
    try {
      assert.equal(o.overlay.querySelector('.drive-output').hidden, true);
    } finally { o.d.close(); o.restore(); }
  }, world.globals);
});

test('overlay: with setSinkId the Output button lists the outputs; picking one saves sinkId, applies it and relabels the button', dom, async () => {
  const world = fakeAudioWorld();
  const sinks = [];
  world.globals.AudioContext.prototype.setSinkId = async function (id) { sinks.push(id); };
  await withDom(async () => {
    const o = await openOverlay(world);
    // openOverlay installs its own mediaDevices; the menu enumerates on the tap.
    navigator.mediaDevices.enumerateDevices = async () => [
      { kind: 'audiooutput', deviceId: 'default', label: 'Default' },
      { kind: 'audiooutput', deviceId: 'bt1', label: 'Headphones (Bluetooth)' },
      { kind: 'audiooutput', deviceId: 'sp1', label: 'Speakers (Realtek)' },
      { kind: 'audioinput', deviceId: 'm1', label: 'Mic' },
    ];
    try {
      const btn = o.overlay.querySelector('.drive-output');
      assert.equal(btn.hidden, false);
      assert.equal(btn.textContent, EN.drive.output, 'the system default: the plain word');
      btn.click();
      await settle();
      const items = [...o.overlay.querySelectorAll('.drive-output-item')];
      assert.deepEqual(items.map((b) => b.textContent.replace('✓', '').trim()), [EN.drive.device_default, 'Headphones (Bluetooth)', 'Speakers (Realtek)']);
      assert.equal(items[0].getAttribute('aria-checked'), 'true');
      items[1].click();
      await settle();
      assert.equal(JSON.parse(globalThis.localStorage.getItem('dispatch-drive')).sinkId, 'bt1');
      assert.deepEqual(sinks, ['bt1'], 'applied to the AudioContext through the settings listener');
      assert.equal(o.overlay.querySelector('.drive-output-menu'), null, 'the menu closed');
      assert.equal(btn.textContent, EN.drive.output_bluetooth);
      // Esc closes an open menu without hanging up.
      btn.click(); await settle();
      o.overlay.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
      assert.equal(o.overlay.querySelector('.drive-output-menu'), null);
      assert.ok(document.querySelector('.drive-overlay'));
    } finally { o.d.close(); o.restore(); saveDriveSettings(globalThis.localStorage, { sinkId: '' }); }
  }, world.globals);
});
