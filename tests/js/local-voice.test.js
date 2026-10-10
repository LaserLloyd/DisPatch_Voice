// T8: local-voice.js — the phone's voice: pickVoice on desktop- and Android-shaped lists,
// labels, one utterance per `say`, watchdog, cancel-not-pause, respeak, gen drops,
// playedMs, playback_done timing (SPEC §5.4, contracts C-§E.2, C.3; feasibility H2, H4).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  pickVoice, voiceOptions, normLang, primaryTag, langLabel, isAndroid, splitUtterance, waitForVoices,
  createLocalVoice, LOCAL_VOICE,
} from '../../src/dispatch_voice/static/local-voice.js';
import { fakeClock, fakeSynth, FakeUtterance } from './_t8-fakes.js';

// Chrome desktop (Linux/Windows/macOS): named voices, network ones with localService false.
const DESKTOP = [
  { name: 'Google US English', lang: 'en-US', localService: false, default: false, voiceURI: 'Google US English' },
  { name: 'Google UK English Female', lang: 'en-GB', localService: false, default: false, voiceURI: 'Google UK English Female' },
  { name: 'Microsoft Zira - English (United States)', lang: 'en-US', localService: true, default: true, voiceURI: 'Microsoft Zira - English (United States)' },
  { name: 'Microsoft George - English (United Kingdom)', lang: 'en-GB', localService: true, default: false, voiceURI: 'Microsoft George - English (United Kingdom)' },
  { name: 'Microsoft Hedda - German (Germany)', lang: 'de-DE', localService: true, default: false, voiceURI: 'Microsoft Hedda - German (Germany)' },
];
// Chrome on Android (TtsPlatformImpl): one entry per locale, named by language, lang with "_",
// localService always true; the real voice is the one chosen in Android settings.
const ANDROID = [
  { name: 'Deutsch', lang: 'de_DE', localService: true, default: false, voiceURI: 'Deutsch' },
  { name: 'English', lang: 'en_US', localService: true, default: true, voiceURI: 'English' },
  { name: 'English', lang: 'en_GB', localService: true, default: false, voiceURI: 'English' },
  { name: 'English', lang: 'en_IN', localService: true, default: false, voiceURI: 'English' },
  { name: 'español', lang: 'es_ES', localService: true, default: false, voiceURI: 'español' },
  { name: '日本語', lang: 'ja_JP', localService: true, default: false, voiceURI: '日本語' },
];

test('normLang / primaryTag / isAndroid', () => {
  assert.equal(normLang('en_US'), 'en-US');
  assert.equal(normLang('EN-gb'), 'en-GB');
  assert.equal(normLang('zh_Hant_TW'), 'zh-Hant-TW');
  assert.equal(normLang(''), '');
  assert.equal(primaryTag('pt_BR'), 'pt');
  assert.equal(isAndroid('Mozilla/5.0 (Linux; Android 16; Pixel 9) Chrome/151'), true);
  assert.equal(isAndroid('Mozilla/5.0 (X11; Linux x86_64) Chrome/151'), false);
});

test('pickVoice, desktop-shaped: local voices only, exact region preferred, stored URI honoured', () => {
  assert.equal(pickVoice(DESKTOP, 'en-GB').name, 'Microsoft George - English (United Kingdom)');
  assert.equal(pickVoice(DESKTOP, 'en-US').name, 'Microsoft Zira - English (United States)');
  assert.equal(pickVoice(DESKTOP, 'en').name, 'Microsoft Zira - English (United States)', 'the default among en');
  assert.equal(pickVoice(DESKTOP, 'de').lang, 'de-DE');
  assert.equal(pickVoice(DESKTOP, 'en', 'Microsoft George - English (United Kingdom)').lang, 'en-GB');
  assert.equal(pickVoice(DESKTOP, 'en', 'Google US English').localService, true, 'a stored network voice is not honoured on desktop');
  assert.equal(pickVoice(DESKTOP, 'ja'), null, 'no voice for the language: the device falls back to Text');
  assert.equal(pickVoice(DESKTOP.filter((v) => !v.localService), 'en'), null, 'network-only desktop: no phone voice');
  assert.equal(pickVoice([], 'en'), null);
  assert.equal(pickVoice(undefined, 'en'), null);
});

test('pickVoice, Android-shaped: per-locale entries, no localService filter, exact lang wins', () => {
  const opt = { android: true };
  assert.equal(pickVoice(ANDROID, 'en-GB', null, opt).lang, 'en_GB');
  assert.equal(pickVoice(ANDROID, 'en_IN', null, opt).lang, 'en_IN');
  assert.equal(pickVoice(ANDROID, 'en', null, opt).lang, 'en_US', 'the default locale');
  assert.equal(pickVoice(ANDROID, 'es-MX', null, opt).lang, 'es_ES', 'primary subtag match');
  assert.equal(pickVoice(ANDROID, 'ja', null, opt).name, '日本語');
  assert.equal(pickVoice(ANDROID, 'fr', null, opt), null);
  // Even an engine that reported localService false is usable on Android (it cannot tell).
  const odd = ANDROID.map((v) => ({ ...v, localService: false }));
  assert.equal(pickVoice(odd, 'en-GB', null, opt).lang, 'en_GB');
  // A "-network" variant (some engines) is avoided when an offline one exists.
  const net = [{ name: 'en-us-x-iol-network', lang: 'en-US', localService: true, voiceURI: 'en-us-x-iol-network' },
    { name: 'en-us-x-iol-local', lang: 'en-US', localService: true, voiceURI: 'en-us-x-iol-local' }];
  assert.equal(pickVoice(net, 'en-US', null, opt).name, 'en-us-x-iol-local');
});

test('labels come from lang ("English (United States)"), names only where two share a lang', () => {
  assert.equal(langLabel('en_US'), 'English (United States)');
  assert.equal(langLabel('de-DE', 'de'), 'Deutsch (Deutschland)');
  const opts = voiceOptions(ANDROID, 'en', { android: true });
  assert.deepEqual(opts.map((o) => o.label), ['English (United States)', 'English (United Kingdom)', 'English (India)']);
  const two = [...DESKTOP, { name: 'Microsoft David - English (United States)', lang: 'en-US', localService: true, voiceURI: 'David' }];
  const d = voiceOptions(two, 'en');
  assert.deepEqual(d.map((o) => o.label), [
    'English (United States) — Microsoft Zira - English (United States)',
    'English (United Kingdom)',
    'English (United States) — Microsoft David - English (United States)',
  ]);
});

test('waitForVoices waits for voiceschanged (headless and fresh Android start with 0 voices)', async () => {
  const synth = fakeSynth([]);
  const listeners = [];
  synth.addEventListener = (ev, fn) => listeners.push([ev, fn]);
  synth.removeEventListener = () => {};
  const p = waitForVoices(synth, { timeoutMs: 1000 });
  synth.voices = ANDROID;
  listeners[0][1]();
  assert.equal((await p).length, ANDROID.length);
  const clock = fakeClock();
  const empty = fakeSynth([]);
  const q = waitForVoices(empty, { timeoutMs: 1000, setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  clock.advance(1000);
  assert.deepEqual(await q, []);
  assert.equal((await waitForVoices(fakeSynth(DESKTOP))).length, 5);
});

test('splitUtterance keeps every piece ≤ 260 chars, at sentence, clause, then word boundaries', () => {
  assert.deepEqual(splitUtterance('Short one.'), ['Short one.']);
  assert.deepEqual(splitUtterance('   '), []);
  const long = 'This is a sentence about the dentist at three. '.repeat(12);
  const parts = splitUtterance(long);
  assert.ok(parts.length >= 3);
  assert.ok(parts.every((p) => p.length <= LOCAL_VOICE.maxChars && p.endsWith('.')));
  assert.equal(parts.join(' ').replace(/\s+/g, ' '), long.trim().replace(/\s+/g, ' '));
  const words = 'word '.repeat(100);
  assert.ok(splitUtterance(words).every((p) => p.length <= 260 && !p.includes('wor d')));
  assert.ok(splitUtterance('x'.repeat(600)).every((p) => p.length <= 260));
});

function voice(extra = {}) {
  const clock = fakeClock();
  const synth = fakeSynth(ANDROID);
  const done = [];
  const fails = [];
  const speaking = [];
  const lv = createLocalVoice({
    synth, Utterance: FakeUtterance, rate: 1, voice: ANDROID[2], lang: 'en-GB',
    onDone: (d) => done.push(d), onFail: (r) => fails.push(r), onSpeaking: (s) => speaking.push(s),
    now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer, ...extra,
  });
  return { clock, synth, lv, done, fails, speaking };
}
const say = (seq, text = `Sentence ${seq}.`, gen = 0, kind = 'reply') => ({ type: 'say', gen, seq, text, kind, message_id: 'm1', role: 'character' });

test('one utterance per say, in order, with the picked voice, rate and pitch 1', () => {
  const v = voice({ rate: 1.15 });
  v.lv.enqueue(say(1));
  v.lv.enqueue(say(2));
  assert.equal(v.synth.spoken.length, 1, 'our own queue: one at a time');
  const u = v.synth.spoken[0];
  // u.lang is BCP 47 even though Android lists the voice as 'en_GB' (review L6).
  assert.deepEqual([u.text, u.voice.lang, u.lang, u.rate, u.pitch], ['Sentence 1.', 'en_GB', 'en-GB', 1.15, 1]);
  v.synth.start();
  assert.deepEqual(v.speaking, [true]);
  v.synth.end();
  assert.equal(v.synth.spoken.length, 2);
  v.synth.start();
  v.synth.end();
  assert.deepEqual(v.done.map((d) => [d.seq, d.final, d.spoken]), [[1, false, true], [2, false, true]]);
  assert.deepEqual(v.speaking, [true, false], 'local_speaking ends when the queue drains');
});

test('playback_done: on the last sentence\'s end after say_end, else 400 ms after the queue drains', () => {
  const v = voice();
  v.lv.enqueue(say(1));
  v.lv.enqueue(say(2));
  v.lv.sayEnd({ gen: 0, last_seq: 2 });
  v.synth.start(); v.synth.end();
  v.synth.start(); v.synth.end();
  assert.deepEqual(v.done.filter((d) => d.final).map((d) => [d.seq, d.why]), [[2, 'say_end']]);
  v.clock.advance(1000);
  assert.equal(v.done.filter((d) => d.final).length, 1, 'never twice');
  // Drain fallback, and a say inside the 400 ms cancels it.
  const w = voice();
  w.lv.enqueue(say(1));
  w.synth.start(); w.synth.end();
  w.clock.advance(300);
  w.lv.enqueue(say(2));
  w.clock.advance(300);
  assert.equal(w.done.filter((d) => d.final).length, 0);
  w.synth.start(); w.synth.end();
  w.clock.advance(400);
  assert.deepEqual(w.done.filter((d) => d.final).map((d) => [d.seq, d.why]), [[2, 'drained']]);
  // say_end that arrives after the last sentence already ended.
  const x = voice();
  x.lv.enqueue(say(4));
  x.synth.start(); x.synth.end();
  x.lv.sayEnd({ gen: 0, last_seq: 4 });
  assert.deepEqual(x.done.filter((d) => d.final).map((d) => d.why), ['say_end']);
  x.clock.advance(1000);
  assert.equal(x.done.filter((d) => d.final).length, 1);
});

test('watchdog: no onstart in 1.5 s → cancel and count; two failures → onFail, text only', () => {
  const v = voice();
  v.lv.enqueue(say(1));
  v.lv.enqueue(say(2));
  v.clock.advance(1499);
  assert.equal(v.synth.cancels, 0);
  v.clock.advance(1);
  assert.equal(v.synth.cancels, 1);
  assert.equal(v.fails.length, 0);
  v.clock.advance(LOCAL_VOICE.afterCancelMs);
  assert.equal(v.synth.spoken.at(-1).text, 'Sentence 2.', 'moved on to the next sentence');
  v.clock.advance(1500);
  assert.deepEqual(v.fails, ['no-start']);
  assert.equal(v.lv.failed, true);
  assert.equal(v.lv.enqueue(say(3)), false, 'text only from now on');
});

test('three clean sentences forgive a failure: two cold starts far apart are not a broken voice (review L5)', () => {
  const v = voice();
  v.lv.enqueue(say(1));
  v.clock.advance(1500);                       // cold engine: no onstart in 1.5 s
  for (let i = 2; i < 52; i++) { v.clock.advance(LOCAL_VOICE.afterCancelMs); v.lv.enqueue(say(i)); v.synth.start(); v.synth.end(); }
  v.lv.enqueue(say(60));
  v.clock.advance(1500);                       // evicted while backgrounded: slow once more
  assert.deepEqual(v.fails, []);
  assert.equal(v.lv.failed, false);
  // Two failures with fewer than three clean sentences between them still give up.
  const w = voice();
  w.lv.enqueue(say(1));
  w.clock.advance(1500);
  w.clock.advance(LOCAL_VOICE.afterCancelMs);
  for (let i = 2; i < 4; i++) { w.lv.enqueue(say(i)); w.synth.start(); w.synth.end(); }
  w.lv.enqueue(say(9));
  w.clock.advance(1500);
  assert.deepEqual(w.fails, ['no-start']);
});

test('reset() on a new socket forgets say_end, finals and gens: no premature final (review M5)', () => {
  const v = voice();
  v.lv.enqueue({ ...say(1, 'Session one.'), message_id: 'a' });
  v.synth.start(); v.synth.end();
  v.lv.sayEnd({ gen: 0, last_seq: 1 });
  assert.deepEqual(v.done.filter((d) => d.final).map((d) => `${d.messageId}:${d.seq}`), ['a:1']);
  v.lv.dropOlderThan(3);
  v.lv.reset();                                // ready: seq restarts at 1, gen at 0
  assert.equal(v.lv.enqueue({ ...say(1, 'Two, first.'), message_id: 'b' }), true, 'gen 0 accepted again');
  v.lv.enqueue({ ...say(2, 'Two, second.'), message_id: 'b' });
  v.clock.advance(LOCAL_VOICE.afterCancelMs);
  v.synth.start(); v.synth.end();
  assert.deepEqual(v.done.filter((d) => d.final).map((d) => `${d.messageId}:${d.seq}`), ['a:1'], 'no final after sentence 1 of 2');
  v.synth.start(); v.synth.end();
  v.lv.sayEnd({ gen: 0, last_seq: 2 });
  assert.deepEqual(v.done.filter((d) => d.final).map((d) => `${d.messageId}:${d.seq}`), ['a:1', 'b:2']);
  // The same (gen, seq) key as session 1 gets its own final in session 2.
  const w = voice();
  w.lv.enqueue(say(1)); w.synth.start(); w.synth.end(); w.lv.sayEnd({ gen: 0, last_seq: 1 });
  w.lv.reset();
  w.clock.advance(LOCAL_VOICE.afterCancelMs);
  w.lv.enqueue(say(1)); w.synth.start(); w.synth.end(); w.lv.sayEnd({ gen: 0, last_seq: 1 });
  assert.equal(w.done.filter((d) => d.final).length, 2);
  // reset() cancels what was speaking.
  const x = voice();
  x.lv.enqueue(say(1)); x.synth.start();
  const before = x.synth.cancels;
  x.lv.reset();
  assert.equal(x.synth.cancels, before + 1);
  assert.equal(x.lv.pending, 0);
  assert.deepEqual(x.speaking, [true, false]);
});

test('watchdog: no onend in chars/12 + 3 s → cancel and count', () => {
  const v = voice();
  const text = 'x'.repeat(120); // 10 s + 3 s
  v.lv.enqueue(say(1, text));
  v.synth.start();
  v.clock.advance(12999);
  assert.equal(v.synth.cancels, 0);
  v.clock.advance(1);
  assert.equal(v.synth.cancels, 1);
  assert.equal(v.lv.failed, false);
});

test('engine errors count; interrupted/canceled (our own cancel) do not', () => {
  const v = voice();
  v.lv.enqueue(say(1));
  v.lv.enqueue(say(2));
  v.lv.enqueue(say(3));
  v.synth.start();
  v.synth.error('interrupted');
  assert.equal(v.lv.failed, false);
  v.synth.start();
  v.synth.error('synthesis-failed');
  v.synth.start();
  v.synth.error('audio-busy');
  assert.deepEqual(v.fails, ['audio-busy']);
});

test('cancel-not-pause: barge_in drops older gens and cancels the one speaking', () => {
  const v = voice();
  v.lv.enqueue(say(1, 'Old one.', 3));
  v.lv.enqueue(say(2, 'Old two.', 3));
  v.synth.start();
  v.lv.enqueue(say(1, 'New one.', 4)); // may arrive before barge_in's effect
  v.lv.dropOlderThan(4);
  assert.equal(v.synth.cancels, 1);
  v.clock.advance(LOCAL_VOICE.afterCancelMs);
  assert.equal(v.synth.spoken.at(-1).text, 'New one.');
  assert.equal(v.lv.enqueue(say(3, 'Late old.', 3)), false, 'an older gen is refused');
  assert.ok(!v.synth.spoken.some((u) => u.text === 'Old two.'));
  // gen is u16: 0 after 65535 is newer.
  const w = voice();
  w.lv.dropOlderThan(65535);
  assert.equal(w.lv.enqueue(say(1, 'wrapped', 0)), true);
});

test('respeak() restarts the current sentence (resume {rewind}); cancel() forgets everything', () => {
  const v = voice();
  v.lv.enqueue(say(1, 'Tuesday at three.'));
  v.lv.enqueue(say(2));
  v.synth.start();
  v.clock.advance(700);
  assert.ok(v.lv.playedMs() > 0);
  assert.equal(v.lv.respeak(), true);
  v.clock.advance(LOCAL_VOICE.afterCancelMs);
  assert.equal(v.synth.spoken.at(-1).text, 'Tuesday at three.');
  assert.equal(v.lv.playedMs(), 0, 'heard position restarts with it');
  v.lv.cancel();
  assert.equal(v.lv.pending, 0);
  assert.equal(v.lv.current(), null);
  assert.equal(v.lv.respeak(), false);
});

test('playedMs: ms since onstart, clamped to the sentence\'s length at rate × 15 chars/s', () => {
  const v = voice({ rate: 1 });
  v.lv.enqueue(say(1, 'x'.repeat(30))); // 2 s at 15 chars/s
  assert.equal(v.lv.playedMs(), 0, 'not started');
  v.synth.start();
  v.clock.advance(1200);
  assert.equal(v.lv.playedMs(), 1200);
  v.clock.advance(1500); // 2.7 s: past the estimate, inside the watchdog (30/12 + 3 = 5.5 s)
  assert.equal(v.lv.playedMs(), 2000);
  assert.deepEqual(v.lv.current(), { gen: 0, seq: 1, messageId: 'm1' });
});

test('a long say is spoken in ≤ 260-char pieces as one sentence', () => {
  const v = voice();
  const long = 'This is a sentence about the dentist at three. '.repeat(8);
  v.lv.enqueue(say(1, long));
  let n = 0;
  while (v.synth.current) { n += 1; v.synth.start(); v.synth.end(); }
  assert.ok(n >= 2);
  assert.ok(v.synth.spoken.every((u) => u.text.length <= 260));
  assert.equal(v.done.length, 1);
});

test('requires a synth and an Utterance class', () => {
  assert.throws(() => createLocalVoice({ Utterance: FakeUtterance }), TypeError);
  assert.throws(() => createLocalVoice({ synth: fakeSynth(), Utterance: undefined }), TypeError);
});
