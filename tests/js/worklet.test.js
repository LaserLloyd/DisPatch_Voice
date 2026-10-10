// T8: voice-worklet.js — the mic processor `drive-mic`: rates (incl. 8 kHz SCO), offsets,
// the energy gate with pre-roll and hangover, speech edges, echo bias (contracts C-§E.2,
// SPEC §4.6, §5.6; feasibility M3). Driven through MicCore with a fake port, plus once
// through the registered AudioWorkletProcessor with fake worklet globals.
import test from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  MicCore, FRAME, GATE, MIC_PROCESSOR, isBluetoothOutput, createEchoBias, gateFromReady,
} from '../../src/dispatch_voice/static/voice-worklet.js';
import { fakePort, lcg, sine, dbToAmp } from './_t8-fakes.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = join(HERE, '..', '..', 'src', 'dispatch_voice', 'static', 'voice-worklet.js');

/** Feed `x` in 128-sample render quanta. */
function feed(core, x) { for (let i = 0; i < x.length; i += 128) core.process(x.subarray(i, Math.min(x.length, i + 128))); }
const frames = (port) => port.messages.filter((m) => m && m.buf);
const pcmOf = (fs) => { const out = new Int16Array(fs.length * FRAME); fs.forEach((f, i) => out.set(new Int16Array(f.buf), i * FRAME)); return out; };
function zeroCrossHz(i16, rate = 16000) {
  let z = 0;
  for (let i = 1; i < i16.length; i++) if ((i16[i - 1] < 0) !== (i16[i] < 0)) z += 1;
  return z / 2 / (i16.length / rate);
}
const rms = (i16) => Math.sqrt(i16.reduce((a, v) => a + (v / 32768) ** 2, 0) / i16.length);

for (const rate of [48000, 44100, 16000]) {
  test(`${rate} Hz context: 16 kHz frames of 320 samples with contiguous offsets`, () => {
    const port = fakePort();
    const core = new MicCore(port, rate);
    assert.equal(core.up, false);
    feed(core, sine(rate, rate, 1000, 0.5)); // 1 s of 1 kHz
    const fs = frames(port);
    assert.ok(fs.length >= 49 && fs.length <= 50, `${fs.length} frames`);
    fs.forEach((f, i) => { assert.equal(f.off, i * FRAME); assert.equal(f.buf.byteLength, 640); });
    const pcm = pcmOf(fs).subarray(320); // past the filter's start-up
    assert.ok(Math.abs(zeroCrossHz(pcm) - 1000) < 15, `${zeroCrossHz(pcm)} Hz`);
    assert.ok(Math.abs(rms(pcm) - 0.5 / Math.SQRT2) < 0.02, `rms ${rms(pcm)}`);
    // Levels every 100 ms.
    assert.ok(port.messages.filter((m) => typeof m.level === 'number').length >= 9);
  });
}

test('48 kHz: the anti-alias filter stops a 12 kHz tone folding into the speech band', () => {
  const port = fakePort();
  const core = new MicCore(port, 48000);
  feed(core, sine(48000, 48000, 12000, 0.5));
  const pcm = pcmOf(frames(port)).subarray(320);
  assert.ok(rms(pcm) < 0.005, `alias rms ${rms(pcm)}`);
});

test('8 kHz SCO context: linear upsampling, no decimation, offsets still count 16 kHz samples', () => {
  const port = fakePort();
  const core = new MicCore(port, 8000);
  assert.equal(core.up, true);
  assert.equal(core.h, null); // no FIR decimator built
  feed(core, sine(8000, 8000, 1000, 0.5)); // 1 s
  const fs = frames(port);
  assert.equal(fs.length, 50);
  fs.forEach((f, i) => assert.equal(f.off, i * FRAME));
  const pcm = pcmOf(fs);
  assert.ok(Math.abs(zeroCrossHz(pcm) - 1000) < 10, `${zeroCrossHz(pcm)} Hz`);
  assert.ok(Math.abs(rms(pcm) - 0.5 / Math.SQRT2) < 0.03, `rms ${rms(pcm)}`);
  // 2 s more: offsets keep counting at 16 kHz.
  feed(core, sine(16000, 8000, 300, 0.5));
  assert.equal(frames(port).at(-1).off, 149 * FRAME);
});

test('11.025 kHz context upsamples at the right average rate', () => {
  const port = fakePort();
  const core = new MicCore(port, 11025);
  feed(core, sine(11025 * 2, 11025, 440, 0.3));
  assert.ok(Math.abs(frames(port).length - 100) <= 1);
});

/** A gated core at 16 kHz with a learned floor of ≈ −60 dBFS noise. */
function gatedCore({ hangoverS = 1, preRollMs = 300, rate = 16000 } = {}) {
  const port = fakePort();
  const core = new MicCore(port, rate);
  const rnd = lcg(7);
  const noise = (s, db = -60) => Float32Array.from({ length: Math.round(s * rate) }, () => rnd() * dbToAmp(db) * Math.SQRT2 * 1.22);
  feed(core, noise(0.5));
  core.onMessage({ gate: { on: true, preRollMs, hangoverS } });
  feed(core, noise(hangoverS + 0.2)); // the gate starts open (errs toward sending), then closes
  port.messages.length = 0;
  return { port, core, noise, speech: (s, db = -20) => sine(Math.round(s * rate), rate, 300, dbToAmp(db) * Math.SQRT2) };
}

test('gate: closed in silence; speech opens it with the 300 ms pre-roll at its own offsets', () => {
  const { port, core, noise, speech } = gatedCore();
  const before = core.off;
  feed(core, noise(1));
  assert.equal(frames(port).length, 0, 'nothing leaves while closed');
  const onset = core.off;
  feed(core, speech(0.5));
  const open = port.messages.find((m) => m.gate === 'open');
  assert.ok(open, 'gate opened');
  const fs = frames(port);
  // The pre-roll is 15 frames (300 ms) ending at the frame that opened the gate, which is
  // at most 4 frames (60 ms of loud frames + a partial frame) after the onset: so at least
  // 220 ms of audio from before the speech reaches the server and no word onset is clipped.
  const openFrame = fs[14];
  assert.ok(openFrame.off - onset <= 3 * FRAME, `opened ${(openFrame.off - onset) / 16} ms after onset`);
  assert.ok(onset - open.off >= 220 * 16, `pre-roll reaches ${(onset - open.off) / 16} ms before onset`);
  assert.ok(onset - open.off <= 300 * 16, 'and no more than the pre-roll');
  assert.equal(fs[0].off, open.off);
  for (let i = 1; i < fs.length; i++) assert.equal(fs[i].off, fs[i - 1].off + FRAME, 'contiguous');
  assert.ok(before < onset);
  // The gate message comes before the pre-roll frames.
  assert.ok(port.messages.indexOf(open) < port.messages.indexOf(fs[0]));
});

test('gate: speech edges after 60 ms over floor + 9 dB, off after 200 ms under floor + 6 dB', () => {
  const { port, core, noise, speech } = gatedCore();
  const onset = core.off;
  feed(core, speech(0.5));
  const msgs = port.messages;
  const on = msgs.findIndex((m) => m.speech === true);
  assert.ok(on >= 0);
  // 3 loud frames: the edge is posted within the 3rd frame of speech.
  const framesBefore = msgs.slice(0, on).filter((m) => m.buf && m.off >= onset).length;
  assert.ok(framesBefore <= 3, `${framesBefore} speech frames before the edge`);
  port.messages.length = 0;
  feed(core, noise(0.18));
  assert.equal(port.messages.filter((m) => m.speech === false).length, 0, 'not before 200 ms');
  feed(core, noise(0.05));
  assert.equal(port.messages.filter((m) => m.speech === false).length, 1);
});

test('gate: closes hangoverS after the last loud frame, carrying the next offset', () => {
  const { port, core, noise, speech } = gatedCore({ hangoverS: 1 });
  feed(core, speech(0.4));
  const speechEnd = core.off;
  port.messages.length = 0;
  feed(core, noise(0.9));
  assert.equal(port.messages.filter((m) => m.gate === 'closed').length, 0);
  feed(core, noise(0.3));
  const closed = port.messages.find((m) => m.gate === 'closed');
  assert.ok(closed, 'closed after the hangover');
  const last = frames(port).at(-1);
  assert.equal(closed.off, last.off + FRAME, 'the offset just past the last frame sent');
  // Silence after speech was sent for ≈ the hangover (the server hears what its end-of-turn needs).
  const sentSilenceMs = (last.off + FRAME - speechEnd) / 16;
  assert.ok(sentSilenceMs >= 960 && sentSilenceMs <= 1060, `${sentSilenceMs} ms of trailing audio`);
});

test('gate: never closes while the server says hearing/transcribing (hold)', () => {
  const { port, core, noise, speech } = gatedCore({ hangoverS: 0.5 });
  feed(core, speech(0.3));
  core.onMessage({ hold: true });
  feed(core, noise(2));
  assert.equal(port.messages.filter((m) => m.gate === 'closed').length, 0);
  core.onMessage({ hold: false });
  feed(core, noise(0.05));
  assert.equal(port.messages.filter((m) => m.gate === 'closed').length, 1);
});

test('gate off (a v2 server): every frame is posted', () => {
  const port = fakePort();
  const core = new MicCore(port, 16000);
  feed(core, new Float32Array(16000));
  assert.equal(frames(port).length, 50);
  assert.deepEqual(gateFromReady(undefined), { gate: { on: false } });
  assert.deepEqual(gateFromReady({ pre_roll_ms: 300, hangover_s: 5.3 }), { gate: { on: true, preRollMs: 300, hangoverS: 5.3 } });
});

test('echo bias raises the speech threshold only while the phone voice speaks', () => {
  const run = (localSpeaking, biasDb) => {
    const { port, core, speech } = gatedCore();
    core.onMessage({ localSpeaking, echoBiasDb: biasDb });
    feed(core, speech(0.4, -48)); // ≈ floor + 12 dB
    return port.messages.some((m) => m.speech === true);
  };
  assert.equal(run(false, 9), true, 'no local speech: +12 dB is speech');
  assert.equal(run(true, 0), true, 'local speech, no bias');
  assert.equal(run(true, 6), false, 'local speech + 6 dB Bluetooth bias: the echo stays out');
  const { core } = gatedCore();
  core.onMessage({ echoBiasDb: 40 });
  assert.equal(core.echoBiasDb, GATE.maxEchoBiasDb, 'clamped to +9');
});

test('a louder cabin (no quiet frame for 5 s) lets the floor climb so the gate can close', () => {
  const { port, core, noise } = gatedCore({ hangoverS: 1 });
  feed(core, noise(20, -35)); // engine noise 25 dB over the learned floor, for good
  assert.equal(port.messages.filter((m) => m.gate === 'closed').length, 1, 'closed again within 20 s');
});

test('enabled:false sends nothing but the 16 kHz clock keeps running; restart zeroes it', () => {
  const port = fakePort();
  const core = new MicCore(port, 48000);
  feed(core, new Float32Array(4800)); // 100 ms → 5 frames
  core.onMessage({ enabled: false });
  feed(core, new Float32Array(48000));
  assert.equal(frames(port).length, 5);
  core.onMessage({ enabled: true });
  feed(core, new Float32Array(960));
  assert.equal(frames(port).at(-1).off, 55 * FRAME, 'a 1 s pause is a 1 s gap in offsets (the server zero-fills)');
  core.onMessage({ restart: true });
  port.messages.length = 0;
  feed(core, new Float32Array(960));
  assert.equal(frames(port)[0].off, 0);
});

test('processorOptions carry the gate; the processor registers as drive-mic', async () => {
  const registered = {};
  const g = globalThis;
  const saved = { registerProcessor: g.registerProcessor, AudioWorkletProcessor: g.AudioWorkletProcessor, sampleRate: g.sampleRate };
  g.sampleRate = 8000;
  g.AudioWorkletProcessor = class { constructor() { this.port = fakePort(); } };
  g.registerProcessor = (name, cls) => { registered[name] = cls; };
  try {
    await import(`${pathToFileURL(SRC).href}?registered=1`);
    assert.equal(typeof registered[MIC_PROCESSOR], 'function');
    const p = new registered[MIC_PROCESSOR]({ processorOptions: { gate: { on: false } } });
    assert.equal(p.core.rate, 8000);
    assert.equal(p.process([[new Float32Array(128)]]), true);
    p.port.onmessage({ data: { enabled: false } });
    assert.equal(p.core.enabled, false);
    assert.equal(p.process([[]]), true, 'no input channel (a disconnected mic) is fine');
  } finally {
    Object.assign(g, saved);
    for (const k of Object.keys(saved)) if (saved[k] === undefined) delete g[k];
  }
});

test('Bluetooth output detection and the echo bias policy (+6 start, +3 per echo, −1 per clean turn, 0..9)', () => {
  assert.equal(isBluetoothOutput({ label: 'Galaxy Buds2 (Bluetooth)' }), true);
  assert.equal(isBluetoothOutput({ label: 'Car Hands-Free' }), true);
  assert.equal(isBluetoothOutput({ label: 'Speaker', outputLatency: 0.22 }), true);
  assert.equal(isBluetoothOutput({ label: 'Speaker', outputLatency: 0.02 }), false);
  assert.equal(isBluetoothOutput(), false);
  const b = createEchoBias({ bluetooth: true });
  assert.equal(b.value, 6);
  assert.equal(b.onEchoIgnored(), 9);
  assert.equal(b.onEchoIgnored(), 9);
  assert.equal(b.onCleanTurn(), 8);
  assert.equal(b.onEchoIgnored(false), 8, 'an echo verdict outside local speech does not count');
  const w = createEchoBias();
  assert.equal(w.value, 0);
  assert.equal(w.onCleanTurn(), 0);
  assert.equal(w.setBluetooth(true), 6);
});
