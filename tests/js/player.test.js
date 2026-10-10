// T8: voice-player-worklet.js — the `drive-player` ring: resampling, sample-accurate
// pause/resume/rewind, played/underrun/idle, the priority + ack-cut rule, gens, prebuffer,
// and the main-thread helpers (optimistic duck restore, heard ms) (contracts C-§E.2, C.3,
// C.4 `resume {rewind}`, SPEC §4.6, §6.2, §10; hardening L11, feasibility M2).
import test from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';
import {
  PlayerCore, StreamResampler, PLAYER_PROCESSOR, ACK_UNCUT_MS, createPlayerClient, duckRestoreMs,
  heardPlayedMs, createOptimisticDuck,
} from '../../src/dispatch_voice/static/voice-player-worklet.js';
import { fakePort, fakeClock, sine } from './_t8-fakes.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = join(HERE, '..', '..', 'src', 'dispatch_voice', 'static', 'voice-player-worklet.js');
const RATE = 48000;
const Q = 128;
const STEP = 1 / (0.02 * RATE); // 20 ms pause ramp

/** Render n samples (whole quanta) and return them. */
function render(core, n) {
  const out = new Float32Array(Math.ceil(n / Q) * Q);
  for (let i = 0; i < out.length; i += Q) core.render(out.subarray(i, i + Q));
  return out;
}
/** pcm[i] = (i + 1) × 1e-5: every sample identifies its own index. */
const ramp = (n) => Float32Array.from({ length: n }, (_, i) => (i + 1) * 1e-5);
const idx = (v) => Math.round(v / 1e-5) - 1;
const push = (core, o) => core.onMessage({ type: 'push', gen: 0, seq: 1, kind: 'reply', rate: RATE, ...o });
const firstNonZero = (x) => x.findIndex((v) => v !== 0);

test('24 kHz PCM and 48 kHz Opus both come out at the context rate, gap-free across chunks', () => {
  const port = fakePort();
  const core = new PlayerCore(port, RATE);
  const x = sine(24000, 24000, 440, 0.5); // 1 s
  for (let i = 0; i < x.length; i += 960) push(core, { rate: 24000, pcm: x.slice(i, i + 960) }); // 40 ms chunks
  core.onMessage({ type: 'end', gen: 0, seq: 1 });
  const y = render(core, 48000 + 1024);
  let err = 0;
  // (The first and last 2 ms are the filter meeting silence; real speech starts and ends quiet.)
  for (let j = 96; j < 48000 - 96; j++) err = Math.max(err, Math.abs(y[j] - 0.5 * Math.sin(2 * Math.PI * 440 * j / 48000)));
  assert.ok(err < 1e-3, `max error ${err}`);
  assert.ok(y.subarray(48000 + 40).every((v) => Math.abs(v) < 1e-6), 'then silence');
  const idle = port.of('idle');
  assert.deepEqual(idle.map((m) => [m.gen, m.seq, m.ended]), [[0, 1, true]]);
  // At a 44.1 kHz context the same second lasts 44 100 samples.
  const c2 = new PlayerCore(fakePort(), 44100);
  c2.onMessage({ type: 'push', gen: 0, seq: 1, kind: 'reply', rate: 48000, pcm: new Float32Array(48000).fill(0.25) });
  c2.onMessage({ type: 'end', gen: 0, seq: 1 });
  const z = new Float32Array(44100 + 256);
  for (let i = 0; i < z.length; i += Q) c2.render(z.subarray(i, i + Q));
  const nz = z.filter((v) => Math.abs(v) > 0.2).length;
  assert.ok(Math.abs(nz - 44100) <= 40, `${nz} samples`);
  assert.ok(new StreamResampler(48000, 48000).pass);
});

test('pause: 20 ms ramp, then the clock stops; resume continues at the very next sample', () => {
  const port = fakePort();
  const core = new PlayerCore(port, RATE);
  push(core, { pcm: ramp(48000) });
  const a = render(core, 9600); // 200 ms
  assert.equal(idx(a.at(-1)), 9599);
  core.onMessage({ type: 'pause' });
  const b = render(core, 2048);
  // The ramp-down: sample k plays at env 1 − (k+1)·STEP, until the envelope reaches 0.
  for (let k = 0; k < 900; k++) assert.ok(Math.abs(b[k] - (9600 + k + 1) * 1e-5 * (1 - (k + 1) * STEP)) < 1e-6, `ramp ${k}`);
  const it = core.items[0];
  const stoppedAt = it.read;
  assert.ok(stoppedAt >= 9600 + 959 && stoppedAt <= 9600 + 961, `stopped at ${stoppedAt}`);
  render(core, 48000); // a whole second paused
  assert.equal(it.read, stoppedAt, 'nothing consumed while paused');
  assert.ok(render(core, 256).every((v) => v === 0));
  core.onMessage({ type: 'resume', rewind: false });
  const c = render(core, 1024);
  // The next sample after the pause, ramped up over 20 ms.
  assert.ok(Math.abs(c[0] - (stoppedAt + 1) * 1e-5 * STEP) < 1e-7);
  for (let k = 0; k < 900; k++) assert.ok(Math.abs(c[k] - (stoppedAt + k + 1) * 1e-5 * Math.min(1, (k + 1) * STEP)) < 1e-6, `resume ${k}`);
});

test('resume {rewind: true} restarts the current seq from its first sample', () => {
  const core = new PlayerCore(fakePort(), RATE);
  push(core, { pcm: ramp(48000) });
  render(core, 24000);
  core.onMessage({ type: 'pause' });
  render(core, 4800);
  core.onMessage({ type: 'resume', rewind: true });
  const c = render(core, 2048);
  for (let k = 0; k < 1500; k++) assert.ok(Math.abs(c[k] - (k + 1) * 1e-5 * Math.min(1, (k + 1) * STEP)) < 1e-6, `rewind ${k}`);
  assert.equal(idx(c[2000]), 2000);
});

test('resume without a pause is ignored (never rewinds a sentence that is playing)', () => {
  const core = new PlayerCore(fakePort(), RATE);
  push(core, { pcm: ramp(48000) });
  render(core, 4096);
  core.onMessage({ type: 'resume', rewind: true });
  assert.equal(idx(render(core, 128)[0]), 4096);
});

test('a late resume {rewind} right after an in-place resume still rewinds; never later, never another seq (review M3)', () => {
  // duck off (→ resume in place) then resume {rewind: true}
  const core = new PlayerCore(fakePort(), RATE);
  push(core, { pcm: ramp(96000) });
  render(core, 19200);
  core.onMessage({ type: 'pause' });
  render(core, 43200);                          // 900 ms paused
  core.onMessage({ type: 'resume', rewind: false });
  render(core, 4800);                           // 100 ms of the in-place resume
  core.onMessage({ type: 'resume', rewind: true });
  assert.equal(idx(render(core, 128)[0]), 0, 'rewound to the first sample of the seq');
  // More than 1.5 s after the in-place resume: ignored.
  const late = new PlayerCore(fakePort(), RATE);
  push(late, { pcm: ramp(192000) });
  render(late, 9600);
  late.onMessage({ type: 'pause' }); render(late, 4800);
  late.onMessage({ type: 'resume', rewind: false });
  render(late, 76800);                          // 1.6 s
  late.onMessage({ type: 'resume', rewind: true });
  assert.ok(idx(render(late, 128)[0]) > 80000, 'not rewound');
  // The resumed sentence ended and the next one plays: the late rewind does not touch it.
  const next = new PlayerCore(fakePort(), RATE);
  push(next, { pcm: ramp(9600) });
  render(next, 4800);
  next.onMessage({ type: 'pause' }); render(next, 2400);
  next.onMessage({ type: 'resume', rewind: false });
  push(next, { seq: 2, pcm: ramp(48000) });
  next.onMessage({ type: 'end', gen: 0, seq: 1 });
  render(next, 9600);
  next.onMessage({ type: 'resume', rewind: true });
  assert.ok(idx(render(next, 128)[0]) > 4000, 'seq 2 keeps playing');
});

test('end after a drain is the end, not a hole: no underrun, idle {ended: true} follows (review M2)', () => {
  const cases = [[24000, 48000], [48000, 44100], [48000, 48000]];
  for (const [inRate, ctxRate] of cases) {
    const port = fakePort();
    const core = new PlayerCore(port, ctxRate);
    core.onMessage({ type: 'push', gen: 1, seq: 1, kind: 'reply', rate: inRate, pcm: new Float32Array(inRate / 2).fill(0.3) });
    let n = Math.round(1.3 * ctxRate);
    while (n > 0) { const q = new Float32Array(Math.min(Q, n)); core.render(q); n -= q.length; }
    core.onMessage({ type: 'end', gen: 1, seq: 1 });   // the LLM's next sentence was 800 ms late
    n = Math.round(0.1 * ctxRate);
    while (n > 0) { const q = new Float32Array(Math.min(Q, n)); core.render(q); n -= q.length; }
    const tag = `${inRate}→${ctxRate}`;
    assert.deepEqual(port.of('underrun'), [], tag);
    assert.deepEqual(port.of('idle').map((i) => i.ended), [false, true], tag);
  }
  // A real hole inside a sentence is still one underrun with its gap.
  const port = fakePort();
  const core = new PlayerCore(port, RATE);
  core.onMessage({ type: 'push', gen: 1, seq: 1, kind: 'reply', rate: 24000, pcm: new Float32Array(12000).fill(0.3) });
  render(core, 48000);
  core.onMessage({ type: 'push', gen: 1, seq: 1, kind: 'reply', rate: 24000, pcm: new Float32Array(12000).fill(0.3) });
  assert.equal(port.of('underrun').length, 1);
});

test('end on audio_end and again on the next say: the second end is harmless (amendment T6-2)', () => {
  const port = fakePort();
  const core = new PlayerCore(port, RATE);
  core.onMessage({ type: 'push', gen: 1, seq: 1, kind: 'reply', rate: 24000, pcm: new Float32Array(12000).fill(0.3) });
  core.onMessage({ type: 'end', gen: 1, seq: 1 });      // audio_end
  render(core, 9600);
  core.onMessage({ type: 'push', gen: 1, seq: 2, kind: 'reply', rate: 24000, pcm: new Float32Array(12000).fill(0.3) });
  core.onMessage({ type: 'end', gen: 1, seq: 1 });      // the later say: seq 1 already ended
  render(core, 38400 - 1280);                           // seq 1 done, seq 2 nearly so
  core.onMessage({ type: 'end', gen: 1, seq: 2 });
  render(core, 4800);
  assert.deepEqual(port.of('underrun'), []);
  const idle = port.of('idle');
  assert.deepEqual(idle.map((i) => [i.seq, i.ended]), [[2, true]]);
});

test('played every 100 ms with ms into the seq and the lead; frozen while paused', () => {
  const port = fakePort();
  const core = new PlayerCore(port, RATE);
  push(core, { pcm: new Float32Array(48000).fill(0.1) });
  render(core, 14400); // 300 ms
  const played = port.of('played');
  assert.equal(played.length, 3);
  played.forEach((m, i) => {
    assert.deepEqual([m.gen, m.seq], [0, 1]);
    assert.ok(Math.abs(m.ms - (i + 1) * 100) <= 3, `ms ${m.ms}`);
    assert.ok(Math.abs(m.leadMs - (1000 - m.ms)) <= 1, `lead ${m.leadMs}`);
  });
  assert.ok(port.of('level').length >= 3);
  assert.ok(Math.abs(port.of('level')[0].v - 0.1) < 1e-3);
  core.onMessage({ type: 'pause' });
  render(core, 4800);
  port.messages.length = 0;
  render(core, 9600);
  const frozen = port.of('played');
  assert.equal(frozen.length, 2);
  assert.equal(frozen[0].ms, frozen[1].ms);
});

test('a hole inside one sentence: idle (not ended), then one underrun with the exact gap', () => {
  const port = fakePort();
  const core = new PlayerCore(port, RATE);
  push(core, { pcm: new Float32Array(4800).fill(0.1) }); // 100 ms
  render(core, 4800 + 9600); // play it, then 200 ms of nothing
  assert.deepEqual(port.of('idle').map((m) => m.ended), [false]);
  push(core, { pcm: new Float32Array(4800).fill(0.1) });
  const u = port.of('underrun');
  assert.equal(u.length, 1);
  assert.deepEqual([u[0].gen, u[0].seq], [0, 1]);
  assert.ok(Math.abs(u[0].gapMs - 200) <= 3, `gap ${u[0].gapMs}`);
  core.onMessage({ type: 'end', gen: 0, seq: 1 });
  render(core, 9600);
  assert.deepEqual(port.of('idle').map((m) => m.ended), [false, true]);
  assert.equal(port.of('underrun').length, 1, 'the end is not an underrun');
});

test('the next sentence after a drained one is not an underrun (between-sentence gaps are normal)', () => {
  const port = fakePort();
  const core = new PlayerCore(port, RATE);
  push(core, { seq: 1, pcm: new Float32Array(4800).fill(0.1) });
  render(core, 9600);
  push(core, { seq: 2, pcm: new Float32Array(4800).fill(0.2) });
  const y = render(core, 9600);
  assert.equal(port.of('underrun').length, 0);
  assert.ok(Math.abs(y[0] - 0.2) < 1e-6, 'seq 2 starts at once');
});

test('ack ≤ 1.3 s finishes and the reply queues behind it (ACK_UNCUT_MS)', () => {
  assert.equal(ACK_UNCUT_MS, 1300);
  const core = new PlayerCore(fakePort(), RATE);
  push(core, { seq: 1, kind: 'ack', pcm: new Float32Array(48000).fill(0.3) }); // 1.0 s
  core.onMessage({ type: 'end', gen: 0, seq: 1 });
  render(core, 12032);
  push(core, { seq: 2, kind: 'reply', pcm: new Float32Array(48000).fill(0.7) });
  const y = render(core, 48000);
  const switchAt = y.findIndex((v) => Math.abs(v - 0.7) < 1e-6);
  assert.equal(switchAt, 48000 - 12032, 'every ack sample played, then the reply');
  assert.ok(y.subarray(0, switchAt).every((v) => Math.abs(v - 0.3) < 1e-6));
});

test('a tier line (ack > 1.3 s) is cut by the reply with a 50 ms fade', () => {
  const core = new PlayerCore(fakePort(), RATE);
  push(core, { seq: 1, kind: 'ack', pcm: new Float32Array(96000).fill(0.3) }); // 2 s
  render(core, 12000);
  push(core, { seq: 2, kind: 'reply', pcm: new Float32Array(48000).fill(0.7) });
  const y = render(core, 4800);
  assert.ok(y[0] <= Math.fround(0.3) && y[1] < Math.fround(0.3), 'fading');
  assert.ok(Math.abs(y[1199] - 0.3 * (2400 - 1199) / 2400) < 1e-6, 'half way down at 25 ms');
  assert.ok(Math.abs(y[2400] - 0.7) < 1e-6, 'reply right after 50 ms');
});

test('an ack announced at 1.0 s (totalMs) is not cut even when only part has arrived', () => {
  const core = new PlayerCore(fakePort(), RATE);
  push(core, { seq: 1, kind: 'ack', totalMs: 1000, pcm: new Float32Array(14400).fill(0.3) });
  render(core, 4864);
  push(core, { seq: 1, kind: 'ack', pcm: new Float32Array(33600).fill(0.3) });
  push(core, { seq: 2, kind: 'reply', pcm: new Float32Array(4800).fill(0.7) });
  const y = render(core, 48000);
  assert.equal(y.findIndex((v) => Math.abs(v - 0.7) < 1e-6), 48000 - 4864);
});

test('a notice line and a backchannel are cut by a reply; fillers never interrupt or queue behind a reply', () => {
  for (const kind of ['notice', 'backchannel']) {
    const core = new PlayerCore(fakePort(), RATE);
    push(core, { seq: 1, kind, pcm: new Float32Array(48000).fill(0.3) });
    render(core, 4800);
    push(core, { seq: 2, kind: 'reply', pcm: new Float32Array(4800).fill(0.7) });
    const y = render(core, 4800);
    assert.ok(Math.abs(y[2400] - 0.7) < 1e-6, `${kind} cut after 50 ms`);
  }
  const core = new PlayerCore(fakePort(), RATE);
  push(core, { seq: 1, kind: 'reply', pcm: new Float32Array(9600).fill(0.7) });
  render(core, 128);
  push(core, { seq: 2, kind: 'ack', pcm: new Float32Array(4800).fill(0.3) });
  push(core, { seq: 3, kind: 'backchannel', pcm: new Float32Array(4800).fill(0.4) });
  core.onMessage({ type: 'end', gen: 0, seq: 1 });
  const y = render(core, 19200);
  assert.ok(!y.some((v) => Math.abs(v - 0.3) < 1e-6 || Math.abs(v - 0.4) < 1e-6), 'no filler was heard');
  assert.equal(core.items.length, 0);
});

test('a notice that arrives during a reply waits its turn (never cuts the sentence)', () => {
  const core = new PlayerCore(fakePort(), RATE);
  push(core, { seq: 1, kind: 'reply', pcm: new Float32Array(9600).fill(0.7) });
  render(core, 128);
  push(core, { seq: 2, kind: 'notice', pcm: new Float32Array(4800).fill(0.2) });
  core.onMessage({ type: 'end', gen: 0, seq: 1 });
  const y = render(core, 19200);
  assert.equal(y.findIndex((v) => Math.abs(v - 0.2) < 1e-6), 9600 - 128);
});

test('flush {gen}: older audio fades in 100 ms and never comes back; a newer gen plays', () => {
  const port = fakePort();
  const core = new PlayerCore(port, RATE);
  push(core, { gen: 5, pcm: new Float32Array(48000).fill(0.5) });
  render(core, 4800);
  core.onMessage({ type: 'flush', gen: 6 });
  const y = render(core, 9600);
  assert.ok(y[0] > 0.49 && y[2400] < 0.26 && y[2400] > 0.24, 'fade');
  assert.ok(y.subarray(4800).every((v) => v === 0), 'silent after 100 ms');
  push(core, { gen: 5, seq: 2, pcm: new Float32Array(4800).fill(0.5) });
  assert.equal(core.items.length, 0, 'a late chunk of the old gen is dropped');
  push(core, { gen: 6, seq: 1, pcm: new Float32Array(4800).fill(0.6) });
  assert.ok(Math.abs(render(core, 128)[0] - 0.6) < 1e-6);
});

test('flush ends a pause (barge-in ends the duck); a newer gen in a push flushes the older', () => {
  const core = new PlayerCore(fakePort(), RATE);
  push(core, { gen: 1, pcm: new Float32Array(48000).fill(0.5) });
  render(core, 4800);
  core.onMessage({ type: 'pause' });
  render(core, 4800);
  core.onMessage({ type: 'flush', gen: 2 });
  assert.equal(core.paused, false);
  push(core, { gen: 3, seq: 1, pcm: new Float32Array(9600).fill(0.6) });
  const y = render(core, 9600);
  assert.ok(Math.abs(y[1000] - 0.6) < 1e-6);
});

test('gen is the wire u16: 0 after 65535 is newer', () => {
  const core = new PlayerCore(fakePort(), RATE);
  push(core, { gen: 65535, pcm: new Float32Array(4800).fill(0.5) });
  push(core, { gen: 0, seq: 1, pcm: new Float32Array(4800).fill(0.6) });
  const y = render(core, 9600);
  assert.ok(Math.abs(y[4799] - 0.6) < 1e-6 || Math.abs(y[6000] - 0.6) < 1e-6, 'gen 0 plays');
  push(core, { gen: 65535, seq: 9, pcm: new Float32Array(4800).fill(0.5) });
  assert.ok(!core.items.some((i) => i.gen === 65535), 'the old gen is refused');
});

test('prebuffer ("Wait"): a sentence starts once enough is buffered, or when it ends', () => {
  const core = new PlayerCore(fakePort(), RATE);
  core.onMessage({ type: 'prebuffer', ms: 500 });
  push(core, { pcm: new Float32Array(9600).fill(0.5) }); // 200 ms
  assert.ok(render(core, 9600).every((v) => v === 0), 'held');
  push(core, { pcm: new Float32Array(19200).fill(0.5) }); // 600 ms buffered now
  assert.ok(Math.abs(render(core, 128)[0] - 0.5) < 1e-6, 'starts');
  const c2 = new PlayerCore(fakePort(), RATE);
  c2.onMessage({ type: 'prebuffer', ms: 10000 });
  c2.onMessage({ type: 'push', gen: 0, seq: 1, kind: 'reply', rate: RATE, pcm: new Float32Array(4800).fill(0.5) });
  assert.equal(firstNonZero(render(c2, 4800)), -1);
  c2.onMessage({ type: 'end', gen: 0, seq: 1 });
  assert.ok(Math.abs(render(c2, 128)[0] - 0.5) < 1e-6, 'a complete short sentence plays');
});

test('reset (a new socket) forgets the gens; gain ramps the master level', () => {
  const core = new PlayerCore(fakePort(), RATE);
  push(core, { gen: 9, pcm: new Float32Array(4800).fill(0.5) });
  core.onMessage({ type: 'reset' });
  assert.equal(core.items.length, 0);
  push(core, { gen: 0, pcm: new Float32Array(48000).fill(0.5) });
  core.onMessage({ type: 'gain', value: 0.15, rampMs: 40 });
  const y = render(core, 4096);
  assert.ok(y[0] < 0.5 && y[0] > 0.49);
  assert.ok(Math.abs(y[1920] - 0.075) < 1e-3, 'at 0.15 after 40 ms');
  core.onMessage({ type: 'gain', value: 1, rampMs: 0 });
  assert.ok(Math.abs(render(core, 128)[0] - 0.5) < 1e-6);
});

test('Int16 PCM is accepted; garbage messages are ignored', () => {
  const core = new PlayerCore(fakePort(), RATE);
  core.onMessage(null);
  core.onMessage({ type: 'nonsense' });
  core.onMessage({ type: 'push', gen: 0, seq: 1, rate: RATE, pcm: 'nope' });
  assert.equal(core.items.length, 0);
  core.onMessage({ type: 'push', gen: 0, seq: 1, rate: RATE, pcm: new Int16Array(128).fill(16384) });
  assert.ok(Math.abs(render(core, 128)[0] - 0.5) < 1e-6);
});

test('the player client posts the frozen formats and transfers the PCM', () => {
  const sent = [];
  const port = { postMessage: (m, tr) => sent.push([m, tr]), onmessage: null };
  const played = [];
  const client = createPlayerClient(port, { onPlayed: (d) => played.push(d) });
  const pcm = new Float32Array(10);
  client.push({ gen: 65537, seq: 3, kind: 'ack', rate: 24000, pcm, totalMs: 900 });
  assert.deepEqual(sent[0][0].type, 'push');
  assert.deepEqual([sent[0][0].gen, sent[0][0].seq, sent[0][0].kind, sent[0][0].rate, sent[0][0].totalMs], [1, 3, 'ack', 24000, 900]);
  assert.equal(sent[0][1][0], sent[0][0].pcm.buffer);
  const big = new Float32Array(100);
  client.push({ gen: 0, seq: 1, rate: 48000, pcm: big.subarray(10, 20) });
  assert.equal(sent[1][0].pcm.length, 10);
  assert.equal(sent[1][0].pcm.buffer.byteLength, 40, 'a view is copied before transfer');
  client.pause(); assert.equal(client.paused, true);
  client.resume(true); assert.equal(client.paused, false);
  client.flush(7); client.gain(0.15, 40); client.prebuffer(500); client.end(2, 4); client.reset();
  assert.deepEqual(sent.slice(2).map(([m]) => m), [
    { type: 'pause' }, { type: 'resume', rewind: true }, { type: 'flush', gen: 7 }, { type: 'gain', value: 0.15, rampMs: 40 },
    { type: 'prebuffer', ms: 500 }, { type: 'end', gen: 2, seq: 4 }, { type: 'reset' },
  ]);
  port.onmessage({ data: { type: 'played', gen: 1, seq: 2, ms: 300, leadMs: 700 } });
  assert.deepEqual(client.last, { type: 'played', gen: 1, seq: 2, ms: 300, leadMs: 700 });
  assert.equal(played.length, 1);
});

test('heard.played_ms subtracts outputLatency (fallback baseLatency) and clamps to the sentence', () => {
  assert.equal(heardPlayedMs(1000, { outputLatency: 0.2 }), 800);
  assert.equal(heardPlayedMs(1000, { outputLatency: 0, baseLatency: 0.01 }), 990);
  assert.equal(heardPlayedMs(1000, {}), 1000);
  assert.equal(heardPlayedMs(100, { outputLatency: 0.25 }), 0);
  assert.equal(heardPlayedMs(5000, { outputLatency: 0.1 }, 3000), 3000);
});

test('duck restore waits max(700, 400 + RTT EWMA) ms (hardening L11)', () => {
  assert.equal(duckRestoreMs(0), 700);
  assert.equal(duckRestoreMs(214), 700);
  assert.equal(duckRestoreMs(566), 966);
  assert.equal(duckRestoreMs(undefined), 700);
});

test('optimistic duck: echo guard, restore timer from RTT, at most twice per reply, server takes over', () => {
  const clock = fakeClock();
  const calls = [];
  let rtt = 566;
  const d = createOptimisticDuck({ apply: (on, info) => calls.push(['apply', on, info]), notify: (on) => calls.push(['notify', on]), rtt: () => rtt, now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  assert.equal(d.onSpeech(true), false, 'no reply playing');
  d.onReplyStart();
  clock.advance(1000);
  assert.equal(d.onSpeech(true), false, 'first 1.5 s of a reply: echo guard');
  clock.advance(1000);
  assert.equal(d.onSpeech(true), true);
  assert.deepEqual(calls.splice(0), [['apply', true, {}], ['notify', true]]);
  clock.advance(965);
  assert.equal(d.state, 'local');
  clock.advance(1);
  assert.equal(d.state, 'none');
  // 966 ms > 800 ms: the restore rewinds, like the server's `resume` would (review L4).
  assert.deepEqual(calls.splice(0), [['apply', false, { rewind: true, timeout: true }], ['notify', false]]);
  // Second duck; the server confirms with `duck {on}` → the timer no longer restores.
  assert.equal(d.onSpeech(true), true);
  d.onServerDuck(true);
  clock.advance(5000);
  assert.equal(d.state, 'server');
  assert.equal(d.onResume(true), true);
  assert.deepEqual(calls.splice(0).at(-1), ['apply', false, { rewind: true }]);
  assert.equal(d.onSpeech(true), false, 'at most twice per reply');
  d.onReplyEnd();
  d.onReplyStart();
  clock.advance(2000);
  rtt = 0;
  assert.equal(d.onSpeech(true), true, 'a new reply resets the count');
  d.onBargeIn();
  assert.equal(d.state, 'none');
  assert.deepEqual(calls.splice(0).at(-1), ['apply', false, { flushed: true }]);
  clock.advance(1000);
  assert.equal(calls.length, 0, 'the cancelled timer never fires');
});

test('optimistic duck: two echo_ignored verdicts in a row turn it off; a real turn resets the streak', () => {
  const clock = fakeClock();
  const d = createOptimisticDuck({ apply: () => {}, now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  d.onEchoIgnored();
  d.onTurnAccepted();
  d.onEchoIgnored();
  assert.equal(d.disabled, false);
  d.onEchoIgnored();
  assert.equal(d.disabled, true);
  d.onReplyStart();
  clock.advance(2000);
  assert.equal(d.onSpeech(true), false);
  const off = createOptimisticDuck({ apply: () => {}, enabled: () => false, now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  off.onReplyStart();
  clock.advance(2000);
  assert.equal(off.onSpeech(true), false, 'phone voice / interrupt-by-talking off');
  // Server duck while nothing was ducked locally pauses; duck off restores.
  const calls = [];
  const s = createOptimisticDuck({ apply: (on) => calls.push(on), now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  s.onServerDuck(true);
  s.onServerDuck(false);
  assert.deepEqual(calls, [true, false]);
});

test('optimistic duck: `duck off` then `resume {rewind}` still reaches the player with the rewind (review M3)', () => {
  const clock = fakeClock();
  const calls = [];
  const d = createOptimisticDuck({ apply: (on, info) => calls.push([on, info]), rtt: () => 300, now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  d.onReplyStart();
  clock.advance(2000);
  d.onSpeech(true);
  clock.advance(250);
  d.onServerDuck(true);
  clock.advance(1200);
  d.onServerDuck(false);
  assert.equal(d.onResume(true), false, 'the duck was already over');
  // This side paused 1.45 s ago (> 800 ms): `duck off` already rewinds (FR-works 9),
  // and the server's late `resume {rewind}` is a no-op on top of it.
  assert.deepEqual(calls, [[true, {}], [false, { rewind: true }], [false, { rewind: true }]]);
  // Through the real player core: the late rewind restarts the sentence.
  const core = new PlayerCore(fakePort(), RATE);
  push(core, { pcm: ramp(96000) });
  render(core, 9600);
  for (const [on, info] of calls) core.onMessage(on ? { type: 'pause' } : { type: 'resume', rewind: !!info.rewind });
  assert.equal(idx(render(core, 128)[0]), 0);
});

test('registers as drive-player and renders through process()', async () => {
  const registered = {};
  const g = globalThis;
  const saved = { registerProcessor: g.registerProcessor, AudioWorkletProcessor: g.AudioWorkletProcessor, sampleRate: g.sampleRate };
  g.sampleRate = 48000;
  g.AudioWorkletProcessor = class { constructor() { this.port = fakePort(); } };
  g.registerProcessor = (name, cls) => { registered[name] = cls; };
  try {
    await import(`${pathToFileURL(SRC).href}?registered=1`);
    const P = registered[PLAYER_PROCESSOR];
    assert.equal(typeof P, 'function');
    const p = new P();
    p.port.onmessage({ data: { type: 'push', gen: 0, seq: 1, kind: 'reply', rate: 24000, pcm: new Float32Array(2400).fill(0.5) } });
    const left = new Float32Array(128);
    const right = new Float32Array(128);
    assert.equal(p.process([], [[left, right]]), true);
    assert.ok(left.some((v) => v !== 0));
    assert.deepEqual(Array.from(right), Array.from(left), 'every output channel carries the voice');
  } finally {
    Object.assign(g, saved);
    for (const k of Object.keys(saved)) if (saved[k] === undefined) delete g[k];
  }
});

// ---- review T8 round 2

test('the client passes an Int16Array through (the worklet scales it); end to end it plays at ±0.5 (review r2 L1)', () => {
  const sent = [];
  const port = { postMessage: (m, tr) => sent.push([m, tr]), onmessage: null };
  const client = createPlayerClient(port);
  client.push({ gen: 1, seq: 1, rate: RATE, pcm: new Int16Array([16384, -16384, 0]) });
  const [m, tr] = sent[0];
  assert.ok(m.pcm instanceof Int16Array);
  assert.equal(tr[0], m.pcm.buffer, 'transferred');
  const big = new Int16Array(64);
  client.push({ gen: 1, seq: 1, rate: RATE, pcm: big.subarray(8, 16) });
  assert.equal(sent[1][0].pcm.buffer.byteLength, 16, 'an Int16 view is copied before transfer');
  // What the worklet does with the first message.
  const core = new PlayerCore(fakePort(), RATE);
  core.onMessage({ ...m, pcm: new Int16Array(128).fill(16384) });
  core.onMessage({ type: 'end', gen: 1, seq: 1 });
  const y = render(core, 128);
  assert.ok(Math.abs(y[0] - 0.5) < 1e-6, `got ${y[0]}`);
  // A plain number array is still taken as float samples.
  client.push({ gen: 1, seq: 2, rate: RATE, pcm: [0.25, -0.25] });
  assert.ok(sent[2][0].pcm instanceof Float32Array);
  assert.deepEqual(Array.from(sent[2][0].pcm), [0.25, -0.25]);
});

test('a zero-length push creates nothing and never wedges the queue (review r2 L2)', () => {
  const port = fakePort();
  const core = new PlayerCore(port, RATE);
  push(core, { seq: 1, pcm: new Float32Array(0) });
  core.onMessage({ type: 'end', gen: 0, seq: 1 });
  assert.equal(core.items.length, 0);
  push(core, { seq: 2, pcm: new Float32Array(14400).fill(0.9) });
  core.onMessage({ type: 'end', gen: 0, seq: 2 });
  const y = render(core, RATE);
  assert.ok(y[0] > 0.89, 'seq 2 plays at once');
  assert.deepEqual(port.of('idle').map((i) => [i.seq, i.ended]), [[2, true]]);
  assert.equal(core.items.length, 0);
});

test('an ended item that never got a sample is finished, not waited on (review r2 L2)', () => {
  const port = fakePort();
  const core = new PlayerCore(port, RATE);
  // Build the state directly: same-rate items have no resampler tail, so `end` adds nothing.
  const it = core.newItem({ seq: 1, kind: 'reply', rate: RATE }, 0);
  core.items.push(it);
  core.minGen = 0;
  core.idleSent = false;
  push(core, { seq: 2, pcm: new Float32Array(4800).fill(0.7) });
  core.onMessage({ type: 'end', gen: 0, seq: 1 });
  core.onMessage({ type: 'end', gen: 0, seq: 2 });
  const y = render(core, 4800);
  assert.ok(Math.abs(y[0] - 0.7) < 1e-6, 'seq 2 is not held behind the empty seq 1');
  assert.deepEqual(port.of('idle').map((i) => [i.seq, i.ended]), [[2, true]]);
});

test('a straggler chunk for a finished or fading sentence is dropped, never played after the next one (review r2 L5)', () => {
  const port = fakePort();
  const core = new PlayerCore(port, RATE);
  push(core, { seq: 1, pcm: new Float32Array(9600).fill(0.3) });
  core.onMessage({ type: 'end', gen: 0, seq: 1 });
  render(core, 14400);                                     // seq 1 is over
  push(core, { seq: 2, pcm: new Float32Array(9600).fill(0.9) });
  push(core, { seq: 1, pcm: new Float32Array(9600).fill(0.3) }); // late
  core.onMessage({ type: 'end', gen: 0, seq: 2 });
  const y = render(core, RATE);
  assert.ok(!y.some((v) => Math.abs(v - 0.3) < 1e-6), 'the seq-1 straggler never plays');
  assert.deepEqual(core.items.map((i) => i.seq), []);
  // A notice cut by a reply: chunks still arriving for it during the 50 ms fade are dropped too.
  const c2 = new PlayerCore(fakePort(), RATE);
  c2.onMessage({ type: 'push', gen: 0, seq: 5, kind: 'notice', rate: RATE, pcm: new Float32Array(48000).fill(0.2) });
  render(c2, 256);
  c2.onMessage({ type: 'push', gen: 0, seq: 6, kind: 'reply', rate: RATE, pcm: new Float32Array(48000).fill(0.6) });
  c2.onMessage({ type: 'push', gen: 0, seq: 5, kind: 'notice', rate: RATE, pcm: new Float32Array(4800).fill(0.2) });
  assert.deepEqual(c2.items.map((i) => [i.seq, !!i.cut]), [[5, true], [6, false]]);
  render(c2, 4800);
  c2.onMessage({ type: 'push', gen: 0, seq: 5, kind: 'notice', rate: RATE, pcm: new Float32Array(4800).fill(0.2) });
  assert.deepEqual(c2.items.map((i) => i.seq), [6]);
});

test('the finished-sentence memory is bounded, forgets older gens, and clears on reset (review r2 L5)', () => {
  const core = new PlayerCore(fakePort(), RATE);
  for (let s = 1; s <= 100; s++) {
    push(core, { seq: s, pcm: new Float32Array(16).fill(0.1) });
    core.onMessage({ type: 'end', gen: 0, seq: s });
    render(core, 128);
  }
  assert.ok(core.done.size <= 64);
  push(core, { gen: 1, seq: 1, pcm: new Float32Array(16).fill(0.1) }); // a newer gen: seq 1 again is new
  assert.equal(core.done.size, 0);
  assert.deepEqual(core.items.map((i) => [i.gen, i.seq]), [[1, 1]]);
  core.onMessage({ type: 'end', gen: 1, seq: 1 });
  render(core, 128);
  core.onMessage({ type: 'reset' });
  assert.equal(core.done.size, 0);
  push(core, { gen: 1, seq: 1, pcm: new Float32Array(16).fill(0.1) }); // a new socket reuses (gen, seq)
  assert.equal(core.items.length, 1);
});

test('optimistic duck: reset() on a new socket ends a server duck, restores the gain, keeps the echo verdicts (review r2 M3)', () => {
  const clock = fakeClock();
  const calls = [];
  const d = createOptimisticDuck({ apply: (on, info) => calls.push([on, info]), notify: (on) => calls.push(['notify', on]), rtt: () => 200, now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  d.onReplyStart();
  clock.advance(2000);
  d.onServerDuck(true);
  assert.equal(d.state, 'server');
  // The socket drops (1006); the new socket's `ready`:
  d.reset();
  assert.equal(d.state, 'none');
  assert.deepEqual(calls.splice(0), [[true, { server: true }], [false, { flushed: true }]]);
  d.onReplyStart();
  clock.advance(2000);
  assert.equal(d.onSpeech(true), true, 'the next reply ducks optimistically again');
  // A local duck in force at the drop: its restore timer is cancelled by reset.
  d.reset();
  calls.splice(0);
  clock.advance(5000);
  assert.deepEqual(calls, [], 'the cancelled timer never fires');
  // Nothing in force: reset touches nothing. The echo streak survives a socket.
  d.onEchoIgnored(); d.onEchoIgnored();
  d.reset();
  assert.deepEqual(calls, []);
  assert.equal(d.disabled, true);
});

test('optimistic duck: the rewind follows THIS side\'s pause, not the server\'s later duck (FR-works 9)', () => {
  // "Uh-huh" over a reply: the phone paused at +185 ms, the server's duck came at
  // +500 ms, and the server resumed at +1269 ms with rewind false (it saw 0.77 s).
  // The listener's player was paused 1.08 s, over the 800 ms rule: restart the sentence.
  const clock = fakeClock();
  const calls = [];
  const d = createOptimisticDuck({ apply: (on, info) => calls.push([on, info]), rtt: () => 100, now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  d.onReplyStart();
  clock.advance(2000);
  assert.equal(d.onSpeech(true), true);
  clock.advance(315);
  d.onServerDuck(true);
  clock.advance(769);
  d.onServerDuck(false);
  d.onResume(false);
  assert.deepEqual(calls[1], [false, { rewind: true }], 'duck off after a 1.08 s local pause rewinds');
  // A short local pause does not.
  const calls2 = [];
  const d2 = createOptimisticDuck({ apply: (on, info) => calls2.push([on, info]), rtt: () => 100, now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  d2.onReplyStart();
  clock.advance(2000);
  d2.onSpeech(true);
  clock.advance(300);
  d2.onServerDuck(true);
  clock.advance(200);
  d2.onServerDuck(false);
  d2.onResume(false);
  assert.deepEqual(calls2.slice(1), [[false, { rewind: false }], [false, { rewind: false }]]);
  // The server ducked first (no local pause): its own rewind decision stands.
  const calls3 = [];
  const d3 = createOptimisticDuck({ apply: (on, info) => calls3.push([on, info]), rtt: () => 100, now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  d3.onServerDuck(true);
  clock.advance(1500);
  d3.onServerDuck(false);
  d3.onResume(true);
  assert.deepEqual(calls3, [[true, { server: true }], [false, { rewind: false }], [false, { rewind: true }]]);
});
