// The voice chat's audio & link engine (owner: track T7, part 2): drive-audio.js
// driving the REAL T8 modules — the player worklet's PlayerCore behind a fake
// port pair, the link monitor, the downlink/uplink framing, the ring, the
// claims, the phone voice — under node with T8's fake clock, codecs and
// speechSynthesis. Contracts C-§C (the client half of the wire), C.5 #1/#4/
// #9–#12, T6-1/T6-2, T8-1..T8-3, SPEC §4.6, §5.2–§5.6, §7.4.

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  createDriveAudio, wireLink, localPath, completeSentences, genNewer, CURSOR_WAIT_MS, OLD_AUDIO_MAX_MS,
  DUCK_GAIN, WAIT_PREBUFFER_MS, RING_MAX_BYTES,
} from '../../src/dispatch_voice/static/drive-audio.js';
import { PlayerCore } from '../../src/dispatch_voice/static/voice-player-worklet.js';
import { packDown, unpackUp, CODEC, KIND, UP_FLAGS } from '../../src/dispatch_voice/static/opus-codec.js';
import { LINK_THRESHOLDS } from '../../src/dispatch_voice/static/link-monitor.js';
import { fakeClock, fakeCodecs, fakeSynth, FakeUtterance, settle } from './_t8-fakes.js';

const RATE = 48000;
const VOICES = [{ voiceURI: 'local-en', name: 'English', lang: 'en-US', localService: true, default: true }];

/** 40 ms of 24 kHz PCM16 (one v3 PCM downlink chunk) at a constant level. */
function pcmChunk(v = 8000, ms = 40) {
  const n = ms * 24;
  const b = new Uint8Array(n * 2);
  const dv = new DataView(b.buffer);
  for (let i = 0; i < n; i++) dv.setInt16(i * 2, v, true);
  return b;
}
const down = (seq, { gen = 0, kind = KIND.reply, t = 0, ms = 40 } = {}) =>
  packDown(CODEC.pcm16, kind, gen, seq, t, [pcmChunk(8000, ms)]).buffer;

/**
 * One engine on a fake world. The player is the real PlayerCore: what the engine
 * posts reaches the core synchronously, and render(ms) plays that much audio,
 * delivering `played`/`idle`/`underrun`/`level` back through the client.
 */
function world({ settings = {}, voices = VOICES, synth = fakeSynth(VOICES), effectiveType = '', codecsOpt = {} } = {}) {
  const clock = fakeClock(1_000_000);
  const sent = [];
  const bins = [];
  const mic = [];
  const gains = [];
  const ui = { sentences: [], now: [], links: [], ducks: [], cutoffs: 0, says: [], speech: [], phoneFailed: 0 };
  let open = false;
  const s = { slowLink: 'phone', optimisticDuck: true, bargeIn: true, phoneVoice: '', ...settings };
  const back = { onmessage: null };
  const core = new PlayerCore({ postMessage: (m) => back.onmessage && back.onmessage({ data: m }) }, RATE);
  const port = {
    set onmessage(fn) { back.onmessage = fn; },
    get onmessage() { return back.onmessage; },
    posted: [],
    postMessage(m) { port.posted.push(m); core.onMessage(m); },
  };
  const { codecs, log: codecLog } = fakeCodecs(codecsOpt);
  const a = createDriveAudio({
    send: (o) => { if (open) sent.push(o); },
    sendBinary: (b) => { if (open) bins.push(b); },
    isOpen: () => open,
    micPost: (m) => mic.push(m),
    setDuckGain: (v) => gains.push(v),
    settings: () => s,
    lang: () => 'en', android: false, effectiveType,
    synth, Utterance: FakeUtterance, codecs,
    latency: () => ({ outputLatency: 0.05 }),
    stream: 'str-test-0001',
    now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer,
    ui: {
      onSentence: (x) => ui.sentences.push(x.text),
      onNow: (k) => ui.now.push(k),
      onLink: (l) => ui.links.push(l),
      onDuck: (on) => ui.ducks.push(on),
      onCutOff: () => { ui.cutoffs += 1; },
      onSay: (m) => ui.says.push(m),
      onSpeech: (on) => ui.speech.push(on),
      onPhoneFailed: () => { ui.phoneFailed += 1; },
    },
  });
  a.setPlayer(port);
  if (voices) a.setVoices(voices);
  const out = new Float32Array(128);
  const w = {
    a, clock, sent, bins, mic, gains, ui, core, port, synth, settings: s, codecLog,
    open() { open = true; return a.opened(); },
    close() { open = false; a.closed(); },
    ready(extra = {}) { a.frame({ type: 'ready', sample_rate_in: 16000, ...extra }); },
    readyV3(extra = {}) { w.ready({ proto: 3, sid: 'sid-1', codec_in: 'pcm16', codec_out: 'pcm16', gate: { pre_roll_ms: 300, hangover_s: 2.5 }, ...extra }); },
    /** Play `ms` of audio (render quanta), moving the fake clock with it and ticking the engine. */
    play(ms) {
      for (let done = 0; done < ms; done += 128 / RATE * 1000) {
        core.render(out);
        clock.advance(128 / RATE * 1000);
      }
    },
    tick(ms, step = 250) { for (let left = ms; left > 0; left -= step) { clock.advance(Math.min(step, left)); a.tick(); } },
    frames: (type) => sent.filter((f) => f.type === type),
    playerOps: (type) => port.posted.filter((m) => m.type === type),
    micFrame: (off, v = 0) => a.micMessage({ off, buf: new Int16Array(320).fill(v).buffer }),
  };
  return w;
}

// ------------------------------------------------------------------ pure ---

test('wireLink: Wait never asks the server for text — the lowest rung instead; other choices pass through', () => {
  const e = { mode: 'text', downKbps: 24, upKbps: 16, why: 'deadline' };
  assert.deepEqual(wireLink(e, 'phone').frame, { type: 'link', mode: 'text', why: 'deadline', down_kbps: 24, up_kbps: 16 });
  const w = wireLink(e, 'wait', LINK_THRESHOLDS);
  assert.equal(w.wait, true);
  assert.deepEqual(w.frame, { type: 'link', mode: 'audio', why: 'deadline', down_kbps: 16, up_kbps: 12 });
  assert.deepEqual(wireLink({ mode: 'audio', downKbps: 16, upKbps: 12, why: 'rtt' }, 'wait').frame,
    { type: 'link', mode: 'audio', why: 'rtt', down_kbps: 16, up_kbps: 12 });
  assert.equal(localPath({ slowLink: 'phone', phoneOk: true }), 'phone');
  assert.equal(localPath({ slowLink: 'phone', phoneOk: false }), 'text', 'no phone voice → text (§5.4)');
  assert.equal(localPath({ slowLink: 'text', phoneOk: true }), 'text');
  assert.ok(genNewer(0, 65535) && !genNewer(65535, 0) && !genNewer(3, 3));
});

test('completeSentences (T6-2): leading complete, uncut sentences of the reply, on from a base', () => {
  const r = (seq, x) => ({ kind: 'reply', mid: 'm', seq, complete: true, cut: false, path: 'server', ...x });
  assert.equal(completeSentences([r(1), r(2), r(3, { complete: false })], 'm'), 2);
  assert.equal(completeSentences([r(1), r(2, { cut: true }), r(3)], 'm'), 1, 'a cut sentence stops the count');
  assert.equal(completeSentences([r(1), { ...r(2), kind: 'ack' }, r(3)], 'm', 4), 6, 'acks are not sentences of the reply');
  assert.equal(completeSentences([r(1, { path: null })], 'm'), 0, 'a say whose audio never came is not heard');
});

// ------------------------------------------------------------------- v2 ---

test('v2 server: raw PCM up (pre-ready frames flushed once), JSON header + PCM down, playback_done on every drain, nothing v3 sent', () => {
  const w = world();
  w.open();
  w.micFrame(0); w.micFrame(320);
  assert.equal(w.bins.length, 0, 'nothing before ready');
  w.ready();
  assert.equal(w.a.proto, 2);
  assert.equal(w.bins.length, 2, 'the pre-ready buffer goes out raw');
  assert.equal(w.bins[0].byteLength, 640);
  assert.deepEqual(w.mic.at(-1), { gate: { on: false } }, 'no gate on v2');
  w.micFrame(640);
  assert.equal(w.bins.length, 3);
  // Down: header, then its PCM.
  w.a.frame({ type: 'audio', gen: 0, seq: 1, chunk: 0, sample_rate: 24000, text: 'Hello there.', message_id: 'm1', kind: 'reply' });
  w.a.binary(new Int16Array(4800).fill(4000).buffer);
  assert.equal(w.ui.says.length, 1, 'v2: the first chunk header is the sentence');
  assert.deepEqual(w.ui.sentences, [], 'not on screen before it plays');
  w.play(150);
  assert.deepEqual(w.ui.sentences, ['Hello there.']);
  assert.equal(w.a.replyPlaying, true);
  w.play(250);
  assert.deepEqual(w.frames('playback_done'), [{ type: 'playback_done', gen: 0, seq: 1 }]);
  assert.equal(w.frames('heard').length + w.frames('link').length + w.frames('probe').length, 0);
});

// ------------------------------------------------------------------- v3 ---

test('v3 ready: gate from ready.gate, pre-ready frames go out as wire3 with their offsets (paced like a replay), live afterwards', () => {
  const w = world();
  w.open();
  for (let k = 0; k < 3; k++) w.micFrame(k * 320, 100);
  w.readyV3();
  assert.equal(w.a.proto, 3);
  assert.deepEqual(w.mic.find((m) => m.gate && m.gate.on), { gate: { on: true, preRollMs: 300, hangoverS: 2.5 } });
  w.clock.advance(50);                                     // 4× realtime: 5 ms per 20 ms frame
  assert.equal(w.bins.length, 3);
  const f = unpackUp(w.bins[0]);
  assert.equal(f.codec, CODEC.pcm16);
  assert.equal(f.off, 0);
  assert.ok(f.flags & UP_FLAGS.replay, 'sent from the ring');
  assert.equal(unpackUp(w.bins[2]).off, 640);
  assert.equal(w.a.live, true);
  w.micFrame(960, 100);
  assert.equal(unpackUp(w.bins.at(-1)).off, 960);
  assert.equal(unpackUp(w.bins.at(-1)).flags & UP_FLAGS.replay, 0, 'then live');
  // The gate must not close while the server is still hearing (§5.6).
  w.a.frame({ type: 'state', state: 'hearing' });
  assert.deepEqual(w.mic.at(-1), { hold: true });
  w.a.frame({ type: 'state', state: 'thinking' });
  assert.deepEqual(w.mic.at(-1), { hold: false });
});

test('v3 reply: captions only for what plays (unheard text is never appended); audio_end ends the sentence; playback_done after the last', () => {
  const w = world();
  w.open(); w.readyV3();
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'First.', kind: 'reply', message_id: 'm1', t: 1 });
  w.a.frame({ type: 'say', gen: 0, seq: 2, text: 'Second.', kind: 'reply', message_id: 'm1', t: 2 });
  assert.deepEqual(w.ui.sentences, [], 'a say is not a caption until its audio plays');
  w.a.binary(down(1, { ms: 40 }));
  for (let i = 0; i < 4; i++) w.a.binary(down(1));
  w.a.frame({ type: 'audio_end', gen: 0, seq: 1, kind: 'reply' });
  assert.ok(w.playerOps('end').some((m) => m.seq === 1), 'audio_end → player end');
  w.play(120);
  assert.deepEqual(w.ui.sentences, ['First.']);
  // A barge-in before "Second." plays: it never appears.
  w.a.frame({ type: 'barge_in', gen: 1 });
  w.a.binary(down(2));
  w.play(200);
  assert.deepEqual(w.ui.sentences, ['First.'], 'the unheard sentence is not appended');
  const heard = w.frames('heard');
  assert.equal(heard.length, 1, 'one heard right after the barge_in (C.3)');
  assert.equal(heard[0].gen, 1, 'with the NEW gen');
  assert.equal(heard[0].seq, 1);
  assert.equal(heard[0].source, 'server');
  assert.ok(heard[0].played_ms >= 0 && heard[0].played_ms <= 200, `played_ms ${heard[0].played_ms} minus outputLatency, clamped`);
});

test('v3 reply: audio_end → end + playback_done {gen, seq} for the last seq once it has played; never on a mid-sentence drain, never on say_end alone (it comes right after the last `say`, before that sentence\'s audio is all sent)', () => {
  const w = world();
  w.open(); w.readyV3();
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'Only.', kind: 'reply', message_id: 'm1', t: 1 });
  w.a.binary(down(1));
  w.play(100);                                    // drained mid-sentence: idle {ended: false}
  assert.equal(w.frames('playback_done').length, 0);
  w.a.frame({ type: 'say_end', gen: 0, message_id: 'm1', last_seq: 1 });
  w.play(200);
  assert.equal(w.frames('playback_done').length, 0, 'say_end does not end the sentence\'s audio');
  assert.equal(w.a.records[0].complete, false);
  w.a.binary(down(1));                            // the rest of its audio still plays
  w.a.frame({ type: 'audio_end', gen: 0, seq: 1, kind: 'reply' });
  w.play(200);
  assert.deepEqual(w.frames('playback_done'), [{ type: 'playback_done', gen: 0, seq: 1 }]);
  assert.equal(w.playerOps('push').filter((m) => m.seq === 1).length, 2, 'both chunks reached the player');
});

test('heard goes out alone every 1 s while a reply plays (C.3), from the worklet clock minus output latency', () => {
  const w = world();
  w.open(); w.readyV3();
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'A long one.', kind: 'reply', message_id: 'm1', t: 1 });
  for (let i = 0; i < 100; i++) w.a.binary(down(1));   // 4 s
  for (let i = 0; i < 10; i++) { w.play(250); w.a.tick(); }
  const h = w.frames('heard');
  assert.ok(h.length >= 2 && h.length <= 3, `about one per second (${h.length} in 2.5 s)`);
  assert.ok(h.at(-1).played_ms > h[0].played_ms, 'played_ms advances');
  assert.ok(h.every((x) => x.source === 'server' && x.seq === 1 && x.gen === 0));
});

test('optimistic duck (§4.6 layer 1): a speech edge 1.5 s into a reply ducks to 0.15 and pauses, tells the server, restores after max(700, 400 + rtt)', () => {
  const w = world();
  w.open(); w.readyV3();
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'Talk talk talk.', kind: 'reply', message_id: 'm1', t: 1 });
  for (let i = 0; i < 100; i++) w.a.binary(down(1));
  w.play(400);
  w.a.micMessage({ speech: true });
  assert.deepEqual(w.ui.ducks, [], 'never in the first 1.5 s of a reply (echo guard)');
  w.play(1300);
  w.a.micMessage({ speech: true });
  assert.equal(w.gains.at(-1), DUCK_GAIN);
  assert.equal(w.playerOps('pause').length, 1);
  assert.deepEqual(w.frames('duck_local'), [{ type: 'duck_local', on: true }]);
  assert.equal(w.core.paused, true);
  // The server confirms nothing: the restore fires at 700 ms (rtt unknown → 0).
  w.clock.advance(699);
  assert.equal(w.core.paused, true);
  w.clock.advance(2);
  assert.equal(w.core.paused, false);
  assert.equal(w.gains.at(-1), 1);
  assert.deepEqual(w.frames('duck_local').at(-1), { type: 'duck_local', on: false });
});

test('server duck → pause; resume {rewind} restarts the sentence; a resume for another gen is ignored; barge_in ends the duck', () => {
  const w = world();
  w.open(); w.readyV3();
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'Hello.', kind: 'reply', message_id: 'm1', t: 1 });
  for (let i = 0; i < 50; i++) w.a.binary(down(1));
  w.play(500);
  w.a.frame({ type: 'duck', on: true, gen: 0 });
  assert.equal(w.core.paused, true);
  w.a.frame({ type: 'resume', gen: 7, rewind: true });
  assert.equal(w.core.paused, true, 'only ever for the current gen (C.4)');
  w.a.frame({ type: 'resume', gen: 0, rewind: true });
  assert.equal(w.core.paused, false);
  assert.ok(w.playerOps('resume').some((m) => m.rewind === true), 'rewind reaches the player');
  w.play(200);
  w.a.frame({ type: 'duck', on: true, gen: 0 });
  w.a.frame({ type: 'barge_in', gen: 1 });
  assert.equal(w.core.paused, false, 'a barge-in ends any duck');
  assert.equal(w.gains.at(-1), 1);
});

test('a slow link (missed first-audio deadline) hands the reply to the phone voice: link text, the phone speaks each say once, late server audio is dropped (claims)', () => {
  const w = world();
  w.open(); w.readyV3();
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'This is late.', kind: 'reply', message_id: 'm1', t: 1 });
  w.tick(6500);
  const link = w.frames('link');
  assert.equal(link.length, 1);
  assert.equal(link[0].mode, 'text');
  assert.equal(link[0].why, 'deadline');
  assert.deepEqual(w.synth.spoken.map((u) => u.text), ['This is late.'], 'the waiting sentence goes to the phone');
  assert.equal(w.ui.links.at(-1).notice, 'phone', 'the notice says the phone voice is reading');
  // Its server audio turns up after all: the phone owns the seq, nothing is scheduled.
  const pushes = w.playerOps('push').length;
  w.a.binary(down(1));
  assert.equal(w.playerOps('push').length, pushes, 'no audio scheduled for a claimed seq');
  // The next sentence comes as text (server in text mode): the phone again, in order.
  w.a.frame({ type: 'say', gen: 0, seq: 2, text: 'And this.', kind: 'reply', message_id: 'm1', t: 2 });
  w.synth.start(); w.a.tick();
  assert.deepEqual(w.ui.sentences, ['This is late.']);
  assert.equal(w.ui.now.at(-1), 'sid-1:1', 'the sentence the phone reads is the highlighted one');
  w.synth.end();
  w.clock.advance(100);
  w.synth.start();
  w.a.frame({ type: 'say_end', gen: 0, message_id: 'm1', last_seq: 2 });
  w.synth.end();
  assert.deepEqual(w.synth.spoken.map((u) => u.text), ['This is late.', 'And this.']);
  assert.deepEqual(w.frames('playback_done').at(-1), { type: 'playback_done', gen: 0, seq: 2 }, 'the phone reports the drain');
  const local = w.mic.filter((m) => 'localSpeaking' in m).map((m) => m.localSpeaking);
  assert.ok(local.includes(true) && local.at(-1) === false, 'local_speaking follows the phone');
});

test('"Text only": a degraded link shows replies as captions at once and reports playback_done at say_end', () => {
  const w = world({ settings: { slowLink: 'text' } });
  w.open(); w.readyV3();
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'Stuck.', kind: 'reply', message_id: 'm1', t: 1 });
  w.tick(6500);
  assert.equal(w.frames('link')[0].mode, 'text');
  assert.deepEqual(w.ui.sentences, ['Stuck.']);
  assert.equal(w.synth.spoken.length, 0);
  assert.equal(w.ui.links.at(-1).notice, 'text');
  w.a.frame({ type: 'say', gen: 0, seq: 2, text: 'Next.', kind: 'reply', message_id: 'm1', t: 2 });
  w.a.frame({ type: 'say_end', gen: 0, message_id: 'm1', last_seq: 2 });
  assert.deepEqual(w.ui.sentences, ['Stuck.', 'Next.']);
  assert.deepEqual(w.frames('playback_done').at(-1), { type: 'playback_done', gen: 0, seq: 2 });
});

test('"Wait": a degraded link stays in the server voice at the lowest rung and prebuffers whole sentences', () => {
  const w = world({ settings: { slowLink: 'wait' } });
  w.open(); w.readyV3({ codec_out: 'opus', codec_in: 'opus' });
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'Hold on.', kind: 'reply', message_id: 'm1', t: 1 });
  w.tick(6500);
  const link = w.frames('link');
  assert.equal(link.length, 1);
  assert.deepEqual(link[0], { type: 'link', mode: 'audio', why: 'deadline', down_kbps: 16, up_kbps: 12 });
  assert.ok(w.playerOps('prebuffer').some((m) => m.ms === WAIT_PREBUFFER_MS));
  assert.equal(w.ui.links.at(-1).notice, 'wait');
  assert.equal(w.synth.spoken.length, 0, 'never the phone voice');
});

test('"Always use the phone\'s voice": pinned text from ready, every reply spoken by the phone, no low-signal notice', () => {
  const w = world({ settings: { slowLink: 'always_phone' } });
  w.open(); w.readyV3();
  assert.equal(w.frames('link')[0].mode, 'text');
  assert.equal(w.frames('link')[0].why, 'user');
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'Phone only.', kind: 'reply', message_id: 'm1', t: 1 });
  w.clock.advance(100);                                    // speak() waits 60 ms after a cancel (ready reset)
  assert.deepEqual(w.synth.spoken.map((u) => u.text), ['Phone only.']);
  assert.equal(w.ui.links.at(-1).degraded, false, 'a choice, not a low signal');
  assert.equal(w.ui.links.at(-1).path, 'phone');
});

test('no phone voice on this device: a degraded "Phone voice" link falls back to captions (§5.4)', () => {
  const w = world({ voices: [], synth: fakeSynth([]) });
  w.open(); w.readyV3();
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'Nobody to read me.', kind: 'reply', message_id: 'm1', t: 1 });
  w.tick(6500);
  assert.equal(w.ui.links.at(-1).notice, 'text');
  assert.deepEqual(w.ui.sentences, ['Nobody to read me.']);
});

test('the phone voice failing twice turns the voice chat to text with the notice (§5.4)', () => {
  const w = world({ settings: { slowLink: 'always_phone' } });
  w.open(); w.readyV3();
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'One.', kind: 'reply', message_id: 'm1', t: 1 });
  w.clock.advance(1600);                                  // no onstart in 1.5 s
  w.a.frame({ type: 'say', gen: 0, seq: 2, text: 'Two.', kind: 'reply', message_id: 'm1', t: 2 });
  w.clock.advance(1700);
  assert.equal(w.ui.phoneFailed, 1);
  assert.equal(w.a.phoneOk, false);
  assert.equal(w.ui.links.at(-1).path, 'text');
  w.a.frame({ type: 'say', gen: 0, seq: 3, text: 'Three.', kind: 'reply', message_id: 'm1', t: 3 });
  assert.ok(w.ui.sentences.includes('Three.'));
});

test('server duck while the phone speaks is informational: the phone keeps speaking; only barge_in cancels (H4)', () => {
  const w = world({ settings: { slowLink: 'always_phone' } });
  w.open(); w.readyV3();
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'Keep going.', kind: 'reply', message_id: 'm1', t: 1 });
  w.clock.advance(100);
  w.synth.start();
  assert.equal(w.a.localSpeaking, true);
  const cancels = w.synth.cancels;
  w.a.frame({ type: 'duck', on: true, gen: 0 });
  assert.equal(w.synth.cancels, cancels);
  assert.deepEqual(w.ui.ducks, []);
  w.clock.advance(300);
  w.a.frame({ type: 'barge_in', gen: 1 });
  assert.equal(w.synth.cancels, cancels + 1);
  const h = w.frames('heard').at(-1);
  assert.equal(h.source, 'phone');
  assert.equal(h.gen, 1);
  assert.ok(h.played_ms > 0);
});

test('echo: echo_ignored during local speech raises the gate bias; a clean turn lowers it (§5.4)', () => {
  const w = world({ settings: { slowLink: 'always_phone' } });
  w.open(); w.readyV3();
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'Echo me.', kind: 'reply', message_id: 'm1', t: 1 });
  w.clock.advance(100);
  w.synth.start();
  w.a.frame({ type: 'echo_ignored', text: 'echo me' });
  const biases = () => w.mic.filter((m) => 'echoBiasDb' in m).map((m) => m.echoBiasDb);
  assert.equal(biases().at(-1), 3);
  w.a.frame({ type: 'transcript', text: 'a real question', merged: false, message_id: 'u1', upto: 32000 });
  assert.equal(biases().at(-1), 2);
});

test('reconnect (v3): start.cursor = max(rx_off, transcript, eot); after_partial counts sentences whose audio is complete; offline speech waits in the ring and replays from the server answer, then live', () => {
  const w = world();
  w.open(); w.readyV3();
  for (let k = 0; k < 10; k++) w.micFrame(k * 320, 500);
  w.a.pong({ type: 'pong', t: 0, rx_off: 3200 }, 50);
  w.a.frame({ type: 'eot', upto: 2880 });
  w.a.frame({ type: 'transcript', text: 'hi', merged: false, message_id: 'u1', upto: 2880 });
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'One.', kind: 'reply', message_id: 'm9', t: 1 });
  w.a.binary(down(1));
  w.a.frame({ type: 'audio_end', gen: 0, seq: 1, kind: 'reply' });
  w.a.frame({ type: 'say', gen: 0, seq: 2, text: 'Two.', kind: 'reply', message_id: 'm9', t: 2 });
  w.a.binary(down(2));                                     // cut: no audio_end
  w.close();
  // Speech while offline: into the ring, not onto a dead socket.
  for (let k = 10; k < 20; k++) w.micFrame(k * 320, 500);
  const extra = w.open();
  assert.deepEqual(extra.cursor, { stream: 'str-test-0001', from: 3200 });
  assert.deepEqual(extra.after_partial, { message_id: 'm9', sentences: 1 }, 'sentence 1 arrived complete; 2 was cut');
  w.bins.length = 0;
  w.readyV3({ sid: 'sid-2' });
  w.micFrame(20 * 320, 500);
  assert.equal(w.bins.length, 0, 'live audio waits for the replay');
  w.a.frame({ type: 'cursor', from: 2880 });               // below our own from (T6-1)
  w.clock.advance(2000);                                   // the pump paces it at 4× realtime
  const replayed = w.bins.map(unpackUp).filter((f) => f.flags & UP_FLAGS.replay);
  assert.ok(replayed.length >= 1);
  assert.ok(unpackUp(w.bins[0]).flags & UP_FLAGS.replay, 'the replay goes first, flagged');
  assert.equal(replayed[0].off, 2880, 'from the server answer, even below our own from');
  assert.equal(w.a.live, true, 'live after the replay');
  const offs = w.bins.map((b) => unpackUp(b).off);
  assert.ok(offs.includes(20 * 320), 'the frame captured during the replay follows it');
  w.micFrame(21 * 320, 500);
  assert.equal(unpackUp(w.bins.at(-1)).flags & UP_FLAGS.replay, 0, 'then plain live frames');
});

test('reconnect: no cursor answer within 3 s → replay from our own cursor; speech older than 20 s is "cut off"', () => {
  const w = world();
  w.open(); w.readyV3();
  for (let k = 0; k < 5; k++) w.micFrame(k * 320, 500);
  w.close();
  w.clock.advance(25000);                                 // a long blackout; the mic clock moves on
  w.open();
  w.readyV3({ sid: 'sid-2' });
  w.clock.advance(CURSOR_WAIT_MS + 1);
  assert.equal(w.ui.cutoffs, 1, 'That got cut off. Say it again.');
  assert.equal(w.a.live, true);
});

test('reconnect: the old session\'s complete audio keeps playing; the new session\'s waits behind it, then plays (T6-2)', () => {
  const w = world();
  w.open(); w.readyV3();
  w.a.frame({ type: 'say', gen: 3, seq: 4, text: 'Old tail.', kind: 'reply', message_id: 'm1', t: 1 });
  for (let i = 0; i < 25; i++) w.a.binary(down(4, { gen: 3 }));   // 1 s
  w.a.frame({ type: 'audio_end', gen: 3, seq: 4, kind: 'reply' });
  w.play(100);
  const resets = w.playerOps('reset').length;
  w.close();
  w.open();
  w.readyV3({ sid: 'sid-2' });
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'New one.', kind: 'reply', message_id: 'm1', t: 1 });
  w.a.binary(down(1, { gen: 0 }));
  assert.equal(w.playerOps('reset').length, resets, 'no reset while the old audio plays');
  w.play(1000);
  assert.ok(w.ui.sentences.includes('Old tail.'));
  assert.equal(w.playerOps('reset').length, resets + 1, 'reset once it drained');
  w.play(200);
  assert.ok(w.ui.sentences.includes('New one.'), 'then the new session plays (gen 0 after gen 3)');
});

test('reconnect: a stuck old sentence cannot hold the new session forever', () => {
  const w = world();
  w.open(); w.readyV3();
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'Stuck.', kind: 'reply', message_id: 'm1', t: 1 });
  w.a.binary(down(1));
  const resets = w.playerOps('reset').length;
  w.close(); w.open(); w.readyV3({ sid: 'sid-2' });
  assert.equal(w.playerOps('reset').length, resets);
  w.clock.advance(OLD_AUDIO_MAX_MS + 1);
  assert.equal(w.playerOps('reset').length, resets + 1);
});

test('a tap-stop flushes now and returns `heard`; the barge_in that follows reports it again with the new gen', () => {
  const w = world();
  w.open(); w.readyV3();
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'Stop me.', kind: 'reply', message_id: 'm1', t: 1 });
  for (let i = 0; i < 25; i++) w.a.binary(down(1));
  w.play(400);
  const h = w.a.stopHere();
  assert.equal(h.seq, 1);
  assert.equal(h.source, 'server');
  assert.ok(w.playerOps('flush').some((m) => m.gen === 1));
  w.a.frame({ type: 'barge_in', gen: 1 });
  assert.deepEqual(w.frames('heard').at(-1), { type: 'heard', gen: 1, seq: 1, played_ms: h.played_ms, source: 'server' });
});

test('decoder failure: link {why: decoder, down_kbps: null}; the next socket asks for PCM16 down', async () => {
  const w = world({ codecsOpt: { failDecode: true } });
  await w.a.probeSupport();
  assert.deepEqual(w.a.codecLists(), { up: ['opus', 'pcm16'], down: ['opus', 'pcm16'] });
  w.open(); w.readyV3({ codec_out: 'opus' });
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'x', kind: 'reply', message_id: 'm1', t: 1 });
  w.a.binary(packDown(CODEC.opus, KIND.reply, 0, 1, 0, [new Uint8Array([0x48, 1, 2])]).buffer);
  await settle();
  assert.deepEqual(w.frames('link').at(-1), { type: 'link', mode: 'audio', down_kbps: null, why: 'decoder' });
  assert.deepEqual(w.a.codecLists().down, ['pcm16']);
});

test('probes: sent when the monitor wants one, measured on the probe frame', () => {
  const w = world({ effectiveType: '2g' });                // starts degraded (text) → probes to climb
  w.open(); w.readyV3();
  assert.equal(w.frames('link')[0].mode, 'text', 'a 2g start is told to the server at ready');
  for (let i = 0; i < 40 && !w.frames('probe').length; i++) { w.tick(1000); w.a.pong({ type: 'pong', t: 0, rx_off: 0 }, 50); }
  const probes = w.frames('probe');
  assert.ok(probes.length >= 1, 'a probe once recovery is due');
  assert.equal(probes[0].bytes, LINK_THRESHOLDS.probeBytes);
  w.clock.advance(100);
  w.a.binary(packDown(CODEC.pcm16, KIND.probe, 0, 0, 0, [new Uint8Array(16384)]).buffer);
  assert.equal(w.frames('link').at(-1).why, 'recovered', 'a fast probe climbs one step');
});

test('speech edges reach the UI (the "No signal" line while waiting) and the gate-closed edge flushes the uplink', () => {
  const w = world();
  w.open(); w.readyV3();
  w.a.micMessage({ speech: true });
  assert.deepEqual(w.ui.speech, [true]);
  w.a.micMessage({ gate: 'closed', off: 3200 });           // no throw, nothing pending
  w.a.dispose();
  assert.deepEqual(w.mic.at(-1), { hold: false });
});

test('the link ladder\'s uplink rung reconfigures this client\'s own Opus encoder (LinkState: nothing to wait for)', async () => {
  const w = world();
  w.open(); w.readyV3({ codec_in: 'opus', codec_out: 'opus' });
  for (let k = 0; k < 6; k++) w.micFrame(k * 320, 800);
  await settle();
  // Three slow pongs (RTT EWMA > 1 s), no reply playing: the codec ladder steps down now.
  for (let i = 0; i < 4; i++) { w.clock.advance(2000); w.a.pong({ type: 'pong', t: 0, rx_off: 0 }, 2500); }
  const link = w.frames('link').at(-1);
  assert.equal(link.up_kbps, 12, 'the ladder asks for the 12 kbps uplink rung');
  for (let k = 6; k < 12; k++) w.micFrame(k * 320, 800);
  await settle();
  const enc = w.codecLog.configs.filter(([k]) => k === 'enc-configure').map(([, c]) => c.bitrate);
  assert.equal(enc.at(-1), 12000, `the encoder was reconfigured (${enc})`);
});

test('after_partial: a socket that opened and dropped before ready does not count the old session twice; a second resume counts on from the first', () => {
  const w = world();
  w.open(); w.readyV3();
  for (const seq of [1, 2, 3]) {
    w.a.frame({ type: 'say', gen: 0, seq, text: `S${seq}.`, kind: 'reply', message_id: 'm7', t: seq });
    w.a.binary(down(seq));
    if (seq < 3) w.a.frame({ type: 'audio_end', gen: 0, seq, kind: 'reply' });
  }
  w.close();
  assert.deepEqual(w.open().after_partial, { message_id: 'm7', sentences: 2 });
  w.close();                                    // dropped before ready
  assert.deepEqual(w.open().after_partial, { message_id: 'm7', sentences: 2 }, 'still 2, not 4');
  w.readyV3({ sid: 'sid-2' });                  // the new session goes on from sentence 3
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'S3.', kind: 'reply', message_id: 'm7', t: 1 });
  w.a.binary(down(1));
  w.a.frame({ type: 'audio_end', gen: 0, seq: 1, kind: 'reply' });
  w.close();
  assert.deepEqual(w.open().after_partial, { message_id: 'm7', sentences: 3 }, '2 skipped before + 1 complete now');
});

test('a server that comes back as v2 after a v3 session gets raw PCM again, never wire3', () => {
  const w = world();
  w.open(); w.readyV3();
  w.close();
  w.open();
  w.bins.length = 0;
  w.ready();                                    // no proto: v2
  w.micFrame(9600, 300);
  assert.equal(w.bins.length, 1);
  assert.equal(w.bins[0].byteLength, 640);
  assert.equal(new Uint8Array(w.bins[0])[0] === 3 && w.bins[0].byteLength !== 640, false, 'raw PCM16, not a wire3 frame');
});

// ------------------------------------------------- the slice-2 server: v2 audio + `say` ---

test('a v2 server that sends `say` (ready.say, no proto): its audio plays once, the phone never joins in, captions follow play', () => {
  const w = world();
  w.open();
  w.ready({ sid: 'abcd', say: true });
  assert.equal(w.a.proto, 2);
  assert.equal(w.a.sayCap, true);
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'Hybrid one.', kind: 'reply', message_id: 'm1', t: 1 });
  w.a.frame({ type: 'audio', gen: 0, seq: 1, chunk: 0, sample_rate: 24000, text: 'Hybrid one.', message_id: 'm1', kind: 'reply' });
  w.a.binary(new Int16Array(4800).fill(3000).buffer);
  w.play(150);
  assert.deepEqual(w.ui.sentences, ['Hybrid one.'], 'one caption: the say and the header are the same sentence');
  w.tick(8000);                                 // past any first-audio deadline
  w.a.frame({ type: 'say', gen: 0, seq: 2, text: 'Hybrid two.', kind: 'reply', message_id: 'm1', t: 2 });
  w.clock.advance(100);
  w.a.frame({ type: 'audio', gen: 0, seq: 2, chunk: 0, sample_rate: 24000, text: 'Hybrid two.', message_id: 'm1', kind: 'reply' });
  w.a.binary(new Int16Array(4800).fill(3000).buffer);
  w.play(300);
  assert.equal(w.synth.spoken.length, 0, 'the link monitor is not fed without the v3 wire: the phone never speaks');
  assert.deepEqual(w.ui.sentences, ['Hybrid one.', 'Hybrid two.']);
  assert.equal(w.frames('link').length + w.frames('probe').length, 0);
});

test('a v2 server that sends `say`: "Always use the phone\'s voice" speaks its say-only replies and reports the drain', () => {
  const w = world({ settings: { slowLink: 'always_phone' } });
  w.open();
  w.ready({ sid: 'abcd', say: true });
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'Read me aloud.', kind: 'reply', message_id: 'm1', t: 1 });
  w.clock.advance(100);
  assert.deepEqual(w.synth.spoken.map((u) => u.text), ['Read me aloud.']);
  w.synth.start();
  w.a.frame({ type: 'say_end', gen: 0, message_id: 'm1', last_seq: 1 });
  w.synth.end();
  assert.deepEqual(w.frames('playback_done').at(-1), { type: 'playback_done', gen: 0, seq: 1 });
  assert.equal(w.frames('link').length, 0, 'no link frames to a v2 server (start.slow_link pins it)');
});

test('a server without `sid`: each session keys its sentences apart, so seq 1 after a reconnect is a new caption', () => {
  const w = world();
  w.open(); w.ready();
  w.a.frame({ type: 'audio', gen: 0, seq: 1, chunk: 0, sample_rate: 24000, text: 'Before.', message_id: 'm1', kind: 'reply' });
  w.a.binary(new Int16Array(4800).fill(3000).buffer);
  w.play(400);
  w.close(); w.open(); w.ready();
  w.a.frame({ type: 'audio', gen: 0, seq: 1, chunk: 0, sample_rate: 24000, text: 'After.', message_id: 'm2', kind: 'reply' });
  w.a.binary(new Int16Array(4800).fill(3000).buffer);
  w.play(150);
  assert.deepEqual(w.ui.sentences, ['Before.', 'After.']);
});

test('heard goes to a `say`-capable v2 server after a barge-in, never to a plain v2 server', () => {
  for (const say of [true, false]) {
    const w = world();
    w.open();
    w.ready(say ? { sid: 'abcd', say: true } : {});
    w.a.frame({ type: 'audio', gen: 0, seq: 1, chunk: 0, sample_rate: 24000, text: 'Long.', message_id: 'm1', kind: 'reply' });
    w.a.binary(new Int16Array(24000).fill(3000).buffer);
    w.play(300);
    w.a.frame({ type: 'barge_in', gen: 1 });
    assert.equal(w.frames('heard').length, say ? 1 : 0, `say=${say}`);
  }
});

// ------------------------------------------------- review T7b: reply lifetime ---

const sayR = (w, seq, text, mid = 'm1', gen = 0) =>
  w.a.frame({ type: 'say', gen, seq, text, kind: 'reply', message_id: mid, t: 1 });
const audioFor = (w, seq, n = 25, gen = 0) => {
  for (let i = 0; i < n; i++) w.a.binary(down(seq, { gen }));
  w.a.frame({ type: 'audio_end', gen, seq, kind: 'reply' });
};

test('after_partial: a reply cut by barge_in, stopped by a tap, or followed by a new turn is never resumed on a reconnect', () => {
  // barge_in
  let w = world();
  w.open(); w.readyV3();
  sayR(w, 1, 'One.'); audioFor(w, 1);
  sayR(w, 2, 'Two.'); audioFor(w, 2);
  w.play(1200);
  w.a.frame({ type: 'barge_in', gen: 1 });
  w.close();
  assert.equal(w.open().after_partial, undefined, 'barge_in');
  // tap-stop
  w = world();
  w.open(); w.readyV3();
  sayR(w, 1, 'One.'); audioFor(w, 1);
  w.play(300);
  w.a.stopHere();
  w.close();
  assert.equal(w.open().after_partial, undefined, 'tap-stop');
  // tts_failed mid-reply (no say_end), then a new turn
  w = world();
  w.open(); w.readyV3();
  sayR(w, 1, 'One.'); audioFor(w, 1);
  sayR(w, 2, 'Two.');
  w.a.frame({ type: 'turn_error', reason: 'tts_failed', message_id: 'm1' });
  w.play(1200);
  w.close();
  assert.deepEqual(w.open().after_partial, { message_id: 'm1', sentences: 1 }, 'a failed sentence alone does not end the reply');
  w.readyV3({ sid: 'sid-2' });
  w.a.frame({ type: 'transcript', text: 'next question', merged: false, message_id: 'u2', upto: 64000 });
  w.close();
  assert.equal(w.open().after_partial, undefined, 'a new turn ends it');
  // agent_failed ends the turn at once
  w = world();
  w.open(); w.readyV3();
  sayR(w, 1, 'One.'); audioFor(w, 1);
  w.a.frame({ type: 'turn_error', reason: 'agent_failed', message_id: 'm1' });
  w.close();
  assert.equal(w.open().after_partial, undefined, 'agent_failed');
});

test('a provisional id that becomes the row id mid-reply (T3s2-1) is ONE reply: after_partial counts every sentence and names the row; say_end under the row id ends it', () => {
  const w = world();
  w.open(); w.readyV3();
  sayR(w, 1, 'A.', 'run:abc'); audioFor(w, 1);
  sayR(w, 2, 'B.', 'run:abc'); audioFor(w, 2);
  sayR(w, 3, 'C.', 'row-9'); audioFor(w, 3);
  sayR(w, 4, 'D.', 'row-9'); for (let i = 0; i < 25; i++) w.a.binary(down(4));
  w.play(500);
  w.close();
  assert.deepEqual(w.open().after_partial, { message_id: 'row-9', sentences: 3 });

  const v = world();
  v.open(); v.readyV3();
  sayR(v, 1, 'A.', 'run:abc'); audioFor(v, 1);
  v.play(1200);                                     // drained before say_end arrives
  v.a.frame({ type: 'say_end', gen: 0, message_id: 'row-9', last_seq: 1 });
  v.tick(3000);
  assert.equal(v.a.replyPlaying, false);
  v.a.micMessage({ speech: true });
  assert.equal(v.frames('duck_local').length, 0, 'speech after the reply is a new turn, not a duck');
  assert.equal(v.playerOps('pause').length, 0);
  v.close();
  assert.equal(v.open().after_partial, undefined, 'the reply ended under its row id');
});

test('a degrade hands the phone only the reply in flight — never an old reply\'s sentence whose speech failed', () => {
  const w = world();
  w.open(); w.readyV3();
  // An old notice whose audio never came (no turn_error is sent for notices).
  w.a.frame({ type: 'say', gen: 0, seq: 9, text: 'Still there?', kind: 'notice', message_id: null, t: 1 });
  sayR(w, 1, 'Old sentence whose audio never came.', 'm1');
  w.a.frame({ type: 'turn_error', reason: 'tts_failed', message_id: 'm1' });
  w.a.frame({ type: 'say_end', gen: 0, message_id: 'm1', last_seq: 1 });
  w.a.frame({ type: 'transcript', text: 'again', merged: false, message_id: 'u2', upto: 32000 });
  sayR(w, 2, 'Fresh reply.', 'm2'); audioFor(w, 2); w.play(1200);
  w.a.frame({ type: 'say_end', gen: 0, message_id: 'm2', last_seq: 2 });
  w.a.frame({ type: 'transcript', text: 'third', merged: false, message_id: 'u3', upto: 48000 });
  sayR(w, 3, 'Third reply, slow.', 'm3');
  w.tick(6000);                                     // its first-audio deadline is missed
  assert.ok(w.frames('link').some((f) => f.mode === 'text'));
  assert.deepEqual(w.synth.spoken.map((u) => u.text), ['Third reply, slow.']);
});

test('the phone voice gives up while it holds the reply\'s last sentence: the caption shows it and playback_done still reaches the server', () => {
  const w = world();
  w.open(); w.readyV3();
  sayR(w, 1, 'First.'); w.tick(6000);              // degraded → the phone
  assert.equal(w.a.link.path, 'phone');
  sayR(w, 2, 'Second and last.');
  w.a.frame({ type: 'say_end', gen: 0, message_id: 'm1', last_seq: 2 });
  w.tick(12000);                                    // the fake engine never starts: two failures
  assert.equal(w.ui.phoneFailed, 1);
  assert.equal(w.a.link.path, 'text');
  assert.ok(w.ui.sentences.includes('Second and last.'));
  assert.ok(w.frames('playback_done').some((f) => f.seq === 2), JSON.stringify(w.frames('playback_done')));
});

test('Wait chosen while degraded asks for the lowest downlink rung even when the event carried no rate', () => {
  const w = world();
  w.open(); w.readyV3({ codec_out: 'opus' });
  sayR(w, 1, 'One.');
  w.tick(6000);
  assert.equal(w.frames('link').at(-1).mode, 'text');
  w.settings.slowLink = 'wait'; w.a.settingsChanged();
  const l = w.frames('link').at(-1);
  assert.deepEqual([l.mode, l.why, l.down_kbps], ['audio', 'user', LINK_THRESHOLDS.rungs.down.at(-1)]);
  assert.equal(w.playerOps('prebuffer').at(-1).ms, WAIT_PREBUFFER_MS);
  assert.deepEqual(wireLink({ mode: 'text', why: 'underrun' }, 'wait', LINK_THRESHOLDS).frame.down_kbps, LINK_THRESHOLDS.rungs.down.at(-1));
});

test('10 s of PCM16 captured before ready never reaches the server as one burst (its audio queue holds 64,000 B)', () => {
  const w = world();
  w.open();
  for (let k = 0; k < 500; k++) w.micFrame(k * 320, 1000);
  w.readyV3({ codec_in: 'pcm16' });
  const bytes = () => w.bins.reduce((n, b) => n + b.byteLength, 0);
  assert.ok(bytes() < 64000, `${bytes()} B at once`);
  w.micFrame(500 * 320, 1000);                            // live speech waits behind the flush
  w.clock.advance(1000);
  assert.ok(bytes() < 140000, `${bytes()} B in the first second (4× realtime)`);
  w.clock.advance(2000);
  const offs = w.bins.map((b) => unpackUp(b).off);
  assert.equal(offs.length, 501);
  assert.deepEqual(offs.slice(0, 3), [0, 320, 640]);
  assert.equal(offs.at(-1), 500 * 320, 'the live frame follows the flush, in order');
  assert.equal(w.a.live, true);
});

test('a sentence that ran dry mid-way (a drop mid-reply) is not "playing" for the screen; what was heard is still reported; audio again → playing', () => {
  const w = world();
  w.open(); w.readyV3();
  w.a.frame({ type: 'say', gen: 0, seq: 1, text: 'Cut in the middle.', kind: 'reply', message_id: 'm1', t: 1 });
  for (let i = 0; i < 5; i++) w.a.binary(down(1));         // 200 ms, no audio_end: the link went
  w.play(120);
  assert.equal(w.a.replyPlaying, true, 'audible');
  assert.equal(w.a.playingKind, 'reply');
  w.play(800);                                                // drained; `played` keeps repeating its position
  assert.equal(w.a.replyPlaying, false, 'silent while it waits for audio');
  assert.equal(w.a.playingKind, null);
  const h = w.a.heardNow();
  assert.ok(h && h.seq === 1 && h.played_ms > 0, 'a tap now still reports what was heard');
  for (let i = 0; i < 5; i++) w.a.binary(down(1));          // the audio comes back
  w.play(120);
  assert.equal(w.a.replyPlaying, true, 'audible again');
  w.a.frame({ type: 'audio_end', gen: 0, seq: 1, kind: 'reply' });
  w.play(400);
  assert.equal(w.a.replyPlaying, false, 'ended');
});

// ---------------------------------------- review T7b round 2: across sessions ---

const { startPayload } = await import('../../src/dispatch_voice/static/drive-settings.js');
const sayG = (w, seq, text, { gen = 0, mid = 'm1', kind = 'reply' } = {}) =>
  w.a.frame({ type: 'say', gen, seq, text, kind, message_id: mid, t: 1 });

test('H1: a reconnect mid-reply anchors `after` on the user row the reply answers, never on the cut reply (the session speaks only replies stored AFTER `after`)', () => {
  const w = world();
  w.open(); w.readyV3();
  w.a.frame({ type: 'transcript', text: 'Tell me a story', merged: false, message_id: 'u7', upto: 3200 });
  sayR(w, 1, 'One.', 'r7'); audioFor(w, 1);
  sayR(w, 2, 'Two.', 'r7'); audioFor(w, 2);
  sayR(w, 3, 'Three.', 'r7'); for (let i = 0; i < 5; i++) w.a.binary(down(3));   // cut mid-sentence
  w.close();
  const extra = w.open();
  assert.deepEqual(extra.after_partial, { message_id: 'r7', sentences: 2 });
  assert.deepEqual(extra.resume, { after: 'u7', skip: ['r7'] });
  // drive.js's heard tracker named the cut reply first; the start frame must not.
  const f = startPayload({}, 'r7', ['r7', 'r6', 'r5'], extra);
  assert.equal(f.after, 'u7', 'after = the user row');
  assert.deepEqual(f.after_ids, ['r6', 'r5'], 'the cut reply is not among after_ids');
  assert.deepEqual(f.after_partial, { message_id: 'r7', sentences: 2 });
  // No user row known: the newest reply heard in full.
  assert.equal(startPayload({}, 'r7', ['r7', 'r6'], { ...extra, resume: { after: null, skip: ['r7'] } }).after, 'r6');
  // The new session speaks the rest of r7 (sentence 3 on); once said to its end,
  // a settings re-send names the reply again (it must not be resumed twice).
  w.readyV3({ sid: 'sid-2' });
  sayR(w, 1, 'Three.', 'r7'); audioFor(w, 1);
  assert.ok(w.a.startExtra().resume, 'still resuming while r7 goes on');
  w.a.frame({ type: 'say_end', gen: 0, message_id: 'r7', last_seq: 1 });
  assert.equal(w.a.startExtra().resume, undefined, 'said to its end: no anchor any more');
  assert.equal(startPayload({}, 'r7', ['r7'], w.a.startExtra()).after, 'r7');
});

test('H1: a reply whose say_end came but whose last sentence was cut by the drop is still resumed; one whose audio all arrived is not', () => {
  let w = world();
  w.open(); w.readyV3();
  w.a.frame({ type: 'transcript', text: 'q', merged: false, message_id: 'u1', upto: 3200 });
  sayR(w, 1, 'One.', 'r1'); audioFor(w, 1);
  sayR(w, 2, 'Two.', 'r1'); for (let i = 0; i < 5; i++) w.a.binary(down(2));
  w.a.frame({ type: 'say_end', gen: 0, message_id: 'r1', last_seq: 2 });
  w.close();
  let extra = w.open();
  assert.deepEqual(extra.after_partial, { message_id: 'r1', sentences: 1 }, 'sentence 2 comes again');
  assert.equal(extra.resume.after, 'u1');
  w = world();
  w.open(); w.readyV3();
  sayR(w, 1, 'One.', 'r1'); audioFor(w, 1);
  w.a.frame({ type: 'say_end', gen: 0, message_id: 'r1', last_seq: 1 });
  w.close();
  extra = w.open();
  assert.equal(extra.after_partial, undefined);
  assert.equal(extra.resume, undefined);
});

test('H1: a reply cut while still under its provisional id goes on under its row id in the next session; a second drop counts on from the first', () => {
  const w = world();
  w.open(); w.readyV3();
  w.a.frame({ type: 'transcript', text: 'q', merged: false, message_id: 'u1', upto: 3200 });
  sayR(w, 1, 'A.', 'run:x1'); audioFor(w, 1);
  sayR(w, 2, 'B.', 'run:x1'); audioFor(w, 2);
  sayR(w, 3, 'C.', 'run:x1');
  w.close();
  assert.deepEqual(w.open().after_partial, { message_id: 'run:x1', sentences: 2 });
  w.readyV3({ sid: 'sid-2' });
  sayR(w, 1, 'C.', 'row-5'); audioFor(w, 1);       // the session resumed it by its stored row
  sayR(w, 2, 'D.', 'row-5');
  w.close();
  const extra = w.open();
  assert.deepEqual(extra.after_partial, { message_id: 'row-5', sentences: 3 }, '2 skipped before + 1 complete now');
  assert.equal(extra.resume.after, 'u1');
  assert.ok(extra.resume.skip.includes('row-5') && extra.resume.skip.includes('run:x1'));
});

/** Session 1 with two barge-ins (gen → 2), then a drop and a new session (sid-2). */
function afterTwoBargeIns(opts) {
  const w = world(opts);
  w.open(); w.readyV3();
  sayG(w, 1, 'First.'); for (let i = 0; i < 10; i++) w.a.binary(down(1));
  w.play(100);
  w.a.frame({ type: 'barge_in', gen: 1 });
  sayG(w, 2, 'Second.', { gen: 1, mid: 'm2' }); for (let i = 0; i < 10; i++) w.a.binary(down(2, { gen: 1 }));
  w.play(100);
  w.a.frame({ type: 'barge_in', gen: 2 });
  w.play(300);
  w.close();
  w.open();
  w.readyV3({ sid: 'sid-2' });
  return w;
}

test('H3: a new session restarts gen at 0 — its `resume {gen 0, rewind}` rewinds the sentence', () => {
  const w = afterTwoBargeIns();
  sayG(w, 1, 'New session sentence.', { mid: 'm3' });
  for (let i = 0; i < 50; i++) w.a.binary(down(1));
  w.play(600);
  w.a.frame({ type: 'duck', on: true, gen: 0 });
  w.play(1000);
  w.a.frame({ type: 'duck', on: false, gen: 0 });
  w.a.frame({ type: 'resume', gen: 0, rewind: true });
  assert.ok(w.playerOps('resume').some((m) => m.rewind === true), 'the rewind reaches the player');
  assert.equal(w.core.items[0].read, 0, 'the sentence restarts from its first sample');
});

test('H3: after a reconnect, a tap-stop in phone-voice mode does not mute the phone for the rest of the chat', () => {
  const w = afterTwoBargeIns({ settings: { slowLink: 'always_phone' } });
  sayG(w, 1, 'Phone one.', { mid: 'm3' });
  w.clock.advance(100);
  w.synth.start();
  w.a.stopHere();
  assert.ok(w.playerOps('flush').some((m) => m.gen === 1), 'the stop flushes to the NEW session\'s gen + 1');
  w.a.frame({ type: 'barge_in', gen: 1 });
  w.clock.advance(100);
  w.a.frame({ type: 'transcript', text: 'next', merged: false, message_id: 'u9', upto: 1000 });
  sayG(w, 2, 'Phone two, after the stop.', { gen: 1, mid: 'm4' });
  w.clock.advance(200);
  assert.ok(w.synth.spoken.map((u) => u.text).includes('Phone two, after the stop.'), 'the phone speaks the next reply');
  assert.equal(w.a.records.find((r) => r.text === 'Phone two, after the stop.').path, 'phone');
});

test('H2: Undo (barge_in {g} then `undone`) leaves gen at g: the next reply, sent at g, plays', () => {
  const w = world();
  w.open(); w.readyV3();
  w.a.frame({ type: 'transcript', text: 'Wrong question', merged: false, message_id: 'u1', upto: 3200 });
  w.a.frame({ type: 'barge_in', gen: 1 });
  w.a.undone();
  w.a.frame({ type: 'transcript', text: 'Right question', merged: false, message_id: 'u2', upto: 6400 });
  sayG(w, 1, 'The answer.', { gen: 1, mid: 'r2' });
  audioFor(w, 1, 10, 1);
  w.a.frame({ type: 'say_end', gen: 1, message_id: 'r2', last_seq: 1 });
  w.play(800);
  assert.deepEqual(w.playerOps('flush').map((m) => m.gen), [1], 'one flush, at the barge_in gen');
  assert.ok(w.ui.sentences.includes('The answer.'), 'played and captioned');
  assert.ok(w.frames('playback_done').some((f) => f.gen === 1 && f.seq === 1), 'playback_done reaches the server');
});

test('M2: phone voice across a reconnect — what the phone holds counts as heard, keeps speaking, and the new session\'s sentences wait behind it (nothing spoken twice)', () => {
  const w = world({ settings: { slowLink: 'always_phone' } });
  w.open(); w.readyV3();
  sayG(w, 1, 'Sentence one.'); sayG(w, 2, 'Sentence two.'); sayG(w, 3, 'Sentence three.');
  sayG(w, 4, 'Sentence four.');
  w.clock.advance(100);
  w.synth.start(); w.synth.end();                 // #1 spoken
  w.clock.advance(100);
  w.synth.start();                                // #2 speaking
  w.close();
  const extra = w.open();
  assert.deepEqual(extra.after_partial, { message_id: 'm1', sentences: 4 }, 'every sentence handed to the phone counts');
  w.synth.end();                                  // #2 finishes before `ready`
  w.clock.advance(100);
  w.synth.start();                                // #3 starts
  const sentBefore = w.frames('playback_done').length;
  w.readyV3({ sid: 'sid-2' });
  assert.equal(w.synth.spoken.filter((u) => u.text === 'Sentence three.').length, 1, '#3 is not cancelled by ready');
  assert.equal(w.a.heardNow(), null, 'the old session\'s sentence is not reported to the new one');
  // The new session's next reply arrives while the phone still speaks the old one.
  w.a.frame({ type: 'transcript', text: 'and then?', merged: false, message_id: 'u2', upto: 1000 });
  sayG(w, 1, 'New reply.', { mid: 'm2' });
  w.a.frame({ type: 'say_end', gen: 0, message_id: 'm2', last_seq: 1 });
  assert.ok(!w.synth.spoken.some((u) => u.text === 'New reply.'), 'waits behind the old queue');
  w.synth.end(); w.clock.advance(100);            // #3 done
  w.synth.start(); w.synth.end(); w.clock.advance(100);   // #4
  assert.equal(w.frames('playback_done').length, sentBefore, 'the old session\'s sentences are never reported to the new session');
  w.clock.advance(10);                            // the deferred reset
  w.synth.start(); w.synth.end(); w.clock.advance(500);
  const spoken = w.synth.spoken.map((u) => u.text);
  assert.deepEqual(spoken, ['Sentence one.', 'Sentence two.', 'Sentence three.', 'Sentence four.', 'New reply.']);
  assert.ok(w.frames('playback_done').some((f) => f.gen === 0 && f.seq === 1), 'the new reply\'s playback_done');
});

/** Session 1 leaves ~3 s of complete audio in the player; the link drops; the new session is ready. */
function oldAudioPlaying() {
  const w = world();
  w.open(); w.readyV3();
  sayG(w, 1, 'A long first sentence.');
  for (let i = 0; i < 75; i++) w.a.binary(down(1));
  w.a.frame({ type: 'audio_end', gen: 0, seq: 1, kind: 'reply' });
  w.play(300);
  w.close();
  w.open();
  w.readyV3({ sid: 'sid-2' });
  w.play(300);
  return w;
}

test('M3: a tap-stop while the old session\'s audio plays out (deferred reset) silences it at once', () => {
  const w = oldAudioPlaying();
  assert.ok(w.core.items.length, 'old audio still playing');
  assert.equal(w.a.stopHere(), null, 'nothing of the old session is reported as heard');
  w.play(500);
  assert.equal(w.core.items.length, 0, 'nothing more of the old audio plays');
});

test('M3: during the deferred reset no `heard` names the old session\'s (gen, seq)', () => {
  const w = oldAudioPlaying();
  const n0 = w.frames('heard').length;
  w.tick(2500);
  assert.equal(w.frames('heard').length, n0);
});

test('L1: the uplink ring is byte-capped (10 s of PCM16) and the encoder starts at the server\'s uplink rung', async () => {
  assert.equal(RING_MAX_BYTES, 320000);
  const w = world();
  w.open(); w.readyV3();
  for (let k = 0; k < 1000; k++) w.micFrame(k * 320, 500);   // 20 s of PCM16
  const r = w.a.ring.replayFrom(0);
  const bytes = r.reduce((a, b) => a + b.byteLength, 0);
  assert.ok(bytes <= RING_MAX_BYTES * 1.05, `${bytes} B kept (wire headers included)`);
  assert.ok(bytes >= 300000, 'about 10 s kept');
  const v = world();
  v.open(); v.readyV3({ codec_in: 'opus', codec_out: 'opus', opus_up_kbps: 12 });
  for (let k = 0; k < 4; k++) v.micFrame(k * 320, 800);
  await settle();
  const enc = v.codecLog.configs.filter(([k]) => k === 'enc-configure').map(([, c]) => c.bitrate);
  assert.equal(enc[0], 12000, `the encoder starts at 12 kbps (${enc})`);
});

test('M2: the old session\'s reply all with the phone (say_end came) is not resumed, and its last sentence\'s final is never reported to the new session', () => {
  const w = world({ settings: { slowLink: 'always_phone' } });
  w.open(); w.readyV3();
  sayG(w, 1, 'One.'); sayG(w, 2, 'Two.');
  w.a.frame({ type: 'say_end', gen: 0, message_id: 'm1', last_seq: 2 });
  w.clock.advance(100);
  w.synth.start(); w.synth.end(); w.clock.advance(100);
  w.synth.start();                                  // #2 speaking when the link drops
  w.close();
  const extra = w.open();
  assert.equal(extra.after_partial, undefined, 'the phone has all of it: nothing to resume');
  w.readyV3({ sid: 'sid-2' });
  const n0 = w.frames('playback_done').length;
  w.synth.end();                                    // the old reply's last sentence ends
  w.clock.advance(1000);
  assert.equal(w.frames('playback_done').length, n0, 'not reported to the new session');
  assert.deepEqual(w.synth.spoken.map((u) => u.text), ['One.', 'Two.']);
});

test('a barge_in in the new session ends a reply of the old session whatever its gen (gens restart per session)', () => {
  const w = world();
  w.open(); w.readyV3();
  w.a.frame({ type: 'barge_in', gen: 1 });
  w.a.frame({ type: 'barge_in', gen: 2 });
  sayG(w, 1, 'Old one.', { gen: 2, mid: 'r9' });
  for (let i = 0; i < 75; i++) w.a.binary(down(1, { gen: 2 }));
  w.a.frame({ type: 'audio_end', gen: 2, seq: 1, kind: 'reply' });
  sayG(w, 2, 'Old two.', { gen: 2, mid: 'r9' });
  w.play(200);
  w.close();
  assert.deepEqual(w.open().after_partial, { message_id: 'r9', sentences: 1 });
  w.readyV3({ sid: 'sid-2' });
  w.play(200);                                      // the old audio still plays out
  w.a.frame({ type: 'barge_in', gen: 1 });          // the listener talks over it
  w.close();
  assert.equal(w.open().after_partial, undefined, 'the cut reply is never resumed');
});
