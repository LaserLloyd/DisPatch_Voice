// T8: opus-codec.js — WebCodecs Opus up/down with the PCM fallback (contracts C-§E.2, C.1, C.2).
// WebCodecs does not exist in node; these fakes behave like Chrome's (asynchronous outputs, one
// packet per 20 ms frame, 48 kHz decode). The real browser run is tests/js/browser/ (see
// docs/voice-chat/audio-modules.md) against the server's libopus.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  opusSupport, codecQuery, upConfig, downConfig, createUplink, createOpusUplink, createDownlink,
  createOpusDownlink, packDown, unpackUp, UP_FLAGS, KIND,
} from '../../src/dispatch_voice/static/opus-codec.js';
import { createUplinkRing } from '../../src/dispatch_voice/static/uplink-ring.js';
import { fakeCodecs, settle, opusPacket, fakeClock } from './_t8-fakes.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(join(HERE, '..', 'fixtures', 'wire3.json'), 'utf8'));
const frame = (v = 0) => new Int16Array(320).fill(v).buffer; // what the mic worklet transfers

test('configs are the measured matrix: 16 kHz voip/voice complexity 5 up, 48 kHz down', () => {
  assert.deepEqual(upConfig(16), {
    codec: 'opus', sampleRate: 16000, numberOfChannels: 1, bitrate: 16000,
    opus: { frameDuration: 20000, application: 'voip', signal: 'voice', complexity: 5, format: 'opus' },
  });
  assert.equal(upConfig(12).bitrate, 12000);
  assert.deepEqual(downConfig(), { codec: 'opus', sampleRate: 48000, numberOfChannels: 1 });
});

test('opusSupport asks both directions and never throws', async () => {
  const yes = fakeCodecs();
  assert.deepEqual(await opusSupport({ codecs: yes.codecs }), { up: true, down: true });
  assert.deepEqual(yes.log.configs.map((c) => c[0]).sort(), ['dec', 'enc']);
  const noUp = fakeCodecs({ encSupported: false });
  assert.deepEqual(await opusSupport({ codecs: noUp.codecs }), { up: false, down: true });
  const throwing = fakeCodecs();
  throwing.codecs.AudioDecoder.isConfigSupported = async () => { throw new Error('boom'); };
  assert.deepEqual(await opusSupport({ codecs: throwing.codecs }), { up: true, down: false });
  // No WebCodecs at all (an old browser): PCM both ways.
  assert.deepEqual(await opusSupport({ codecs: { AudioEncoder: null, AudioDecoder: null, AudioData: null, EncodedAudioChunk: null } }), { up: false, down: false });
  assert.deepEqual(codecQuery({ up: true, down: false }), { up: 'opus,pcm16', down: 'pcm16' });
  assert.deepEqual(codecQuery({ up: true, down: true }), { up: 'opus,pcm16', down: 'opus,pcm16' });
});

test('uplink: 5 Opus packets (100 ms) per message, offsets from the worklet, into the ring', async () => {
  const { codecs, log } = fakeCodecs();
  const ring = createUplinkRing();
  const sent = [];
  let speaking = false;
  const up = createOpusUplink({ kbps: 16, codecs, ring, onMessage: (b, info) => sent.push([b, info]), flags: () => (speaking ? UP_FLAGS.localSpeaking : 0) });
  assert.equal(up.codec, 'opus');
  for (let i = 0; i < 12; i++) { if (i === 5) speaking = true; up.push(frame(i), i * 320); }
  await settle();
  assert.equal(sent.length, 2);
  await up.flush(); // the gate closed: the last 2 packets go now
  assert.equal(sent.length, 3);
  const f = sent.map(([b]) => unpackUp(b));
  assert.deepEqual(f.map((x) => [x.codec, x.off, x.packets.length]), [[1, 0, 5], [1, 1600, 5], [1, 3200, 2]]);
  assert.deepEqual(f.map((x) => x.flags), [0, UP_FLAGS.localSpeaking, UP_FLAGS.localSpeaking]);
  // Each packet came from the right frame (the fake encoder echoes the first sample).
  assert.deepEqual(f.flatMap((x) => x.packets.map((p) => p[1])), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
  // AudioData: s16 mono 16 kHz, 320 frames, timestamps 20 ms apart.
  assert.deepEqual(log.encoded.map((d) => d.timestamp).slice(0, 3), [0, 20000, 40000]);
  assert.ok(log.encoded.every((d) => d.format === 's16' && d.sampleRate === 16000 && d.numberOfFrames === 320 && d.closed));
  assert.equal(ring.size, 3);
  assert.equal(ring.newest(), 12 * 320);
});

test('uplink: a gate gap starts a new message (a message is contiguous)', async () => {
  const { codecs } = fakeCodecs();
  const sent = [];
  const up = createUplink({ codec: 'opus', codecs, onMessage: (b) => sent.push(unpackUp(b)) });
  for (let i = 0; i < 3; i++) up.push(frame(), i * 320);
  for (let i = 0; i < 3; i++) up.push(frame(), 32000 + i * 320); // 2 s later: the gate reopened
  await up.flush();
  assert.deepEqual(sent.map((f) => [f.off, f.packets.length]), [[0, 3], [32000, 3]]);
});

test('uplink: an encoder error falls back to PCM16 without losing a frame', async () => {
  const { codecs } = fakeCodecs({ failEncodeAfter: 3 });
  const sent = [];
  const errors = [];
  const up = createUplink({ codec: 'opus', codecs, onMessage: (b) => sent.push(unpackUp(b)), onError: (why) => errors.push(why) });
  for (let i = 0; i < 6; i++) up.push(frame(i), i * 320);
  await settle();
  assert.deepEqual(errors, ['encoder']);
  assert.equal(up.codec, 'pcm16');
  up.push(frame(6), 6 * 320);
  await up.flush();
  // 3 Opus packets in one message, then frames 3..6 as PCM, 1 × 640 B each.
  assert.deepEqual(sent.map((f) => [f.codec, f.off, f.packets.length]), [[1, 0, 3], [0, 960, 1], [0, 1280, 1], [0, 1600, 1], [0, 1920, 1]]);
  assert.equal(sent[1].packets[0].length, 640);
  assert.equal(new Int16Array(sent[4].packets[0].buffer)[0], 6);
});

test('uplink: an encoder that goes silent (no output, no error) falls back to PCM within 2 s (review M6)', async () => {
  const { codecs } = fakeCodecs();
  class Mute extends codecs.AudioEncoder { encode() { if (this.state !== 'configured') throw new Error('nc'); } }
  const clock = fakeClock(0);
  const sent = [];
  const errors = [];
  const up = createUplink({
    codec: 'opus', codecs: { ...codecs, AudioEncoder: Mute }, onMessage: (b) => sent.push(unpackUp(b)),
    onError: (why, e) => errors.push([why, String(e.message)]), now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer,
  });
  for (let i = 0; i < 150; i++) { up.push(frame(i), i * 320); clock.advance(20); }   // 3 s of mic
  await settle();
  assert.deepEqual(errors, [['encoder', 'encoder stalled']]);
  assert.equal(up.codec, 'pcm16');
  // Every frame reached the server, in order, none twice.
  assert.deepEqual(sent.map((f) => f.off), Array.from({ length: 150 }, (_, i) => i * 320));
  assert.ok(sent.every((f) => f.codec === 0));
});

test('uplink: a main-thread hitch (a burst of queued frames) is not a stall', async () => {
  const { codecs } = fakeCodecs();
  const clock = fakeClock(0);
  const sent = [];
  const errors = [];
  const up = createUplink({ codec: 'opus', codecs, onMessage: (b) => sent.push(unpackUp(b)), onError: (w) => errors.push(w), now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  // Frames 0..1 went in just before a 3 s hitch; their outputs are queued behind it.
  up.push(frame(0), 0); up.push(frame(1), 320);
  clock.advance(3000);
  // The burst: the queued frames are delivered at once (outputs interleave on microtasks).
  for (let i = 2; i < 152; i++) { up.push(frame(i), i * 320); if (i % 2 === 0) await settle(); }
  await settle();
  await up.flush();
  assert.deepEqual(errors, []);
  assert.equal(up.codec, 'opus');
  assert.equal(sent.reduce((n, f) => n + f.packets.length, 0), 152);
});

test('uplink: flush() on an encoder whose flush never settles gives up after 2 s and sends PCM (review M6)', async () => {
  const { codecs } = fakeCodecs();
  class Hang extends codecs.AudioEncoder { encode() {} flush() { return new Promise(() => {}); } }
  const clock = fakeClock(0);
  const sent = [];
  const errors = [];
  const up = createUplink({ codec: 'opus', codecs: { ...codecs, AudioEncoder: Hang }, onMessage: (b) => sent.push(unpackUp(b)), onError: (w) => errors.push(w), now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  for (let i = 0; i < 3; i++) up.push(frame(i), i * 320);
  let done = false;
  const p = up.flush().then(() => { done = true; });
  await settle();
  assert.equal(done, false);
  clock.advance(2000);
  await p;
  assert.deepEqual(errors, ['encoder']);
  assert.deepEqual(sent.map((f) => [f.codec, f.off]), [[0, 0], [0, 320], [0, 640]]);
  assert.equal(clock.pending, 0, 'the race timer is cleared');
});

test('uplink: PCM16 mode and no WebCodecs both send 1 × 640 B per 20 ms', async () => {
  const sent = [];
  const pcm = createUplink({ codec: 'pcm16', onMessage: (b) => sent.push(unpackUp(b)) });
  pcm.push(frame(7), 320);
  assert.deepEqual(sent.map((f) => [f.codec, f.off, f.packets[0].length]), [[0, 320, 640]]);
  const errs = [];
  const none = createUplink({ codec: 'opus', codecs: { AudioEncoder: null, AudioData: null }, onMessage: (b) => sent.push(unpackUp(b)), onError: (w) => errs.push(w) });
  assert.equal(none.codec, 'pcm16');
  assert.deepEqual(errs, ['encoder']);
  none.push(frame(1), 640);
  assert.equal(sent.at(-1).codec, 0);
});

test('uplink: the codec ladder reconfigures the encoder in place', async () => {
  const { codecs, log } = fakeCodecs();
  const up = createUplink({ codec: 'opus', kbps: 16, codecs, onMessage: () => {} });
  up.setKbps(12);
  assert.equal(up.kbps, 12);
  const cfgs = log.configs.filter((c) => c[0] === 'enc-configure').map((c) => c[1].bitrate);
  assert.deepEqual(cfgs, [16000, 12000]);
});

test('downlink: Opus decodes at 48 kHz into player pushes tagged gen/seq/kind', async () => {
  const { codecs, log } = fakeCodecs();
  const pushes = [];
  const chunks = [];
  const down = createOpusDownlink({ player: { push: (m) => pushes.push(m) }, codecs, onChunk: (c) => chunks.push(c), now: () => 1234 });
  down.feed(packDown(1, KIND.reply, 3, 9, 5000, [opusPacket(10), opusPacket(20), opusPacket(30)]));
  down.feed(packDown(1, KIND.ack, 3, 10, 5100, [opusPacket(40)]));
  await settle();
  assert.equal(pushes.length, 4);
  assert.deepEqual(pushes.map((p) => [p.type, p.gen, p.seq, p.kind, p.rate, p.pcm.length]), [
    ['push', 3, 9, 'reply', 48000, 960], ['push', 3, 9, 'reply', 48000, 960], ['push', 3, 9, 'reply', 48000, 960], ['push', 3, 10, 'ack', 48000, 960],
  ]);
  assert.equal(pushes[1].pcm[0], 20 / 256);
  assert.deepEqual(chunks.map((c) => [c.seq, c.kind, c.t, c.arrival, c.durationMs, c.codec]), [[9, 'reply', 5000, 1234, 60, 'opus'], [10, 'ack', 5100, 1234, 20, 'opus']]);
  assert.ok(log.decoded.every((c) => c.type === 'key' && c.duration === 20000));
  assert.deepEqual(log.decoders[0].cfg, downConfig());
});

test('downlink: PCM16 frames (24 kHz) need no decoder', () => {
  const pushes = [];
  const down = createDownlink({ player: { push: (m) => pushes.push(m) }, codecs: { AudioDecoder: null, EncodedAudioChunk: null } });
  const pcm = new Int16Array(960);
  pcm[0] = 16384; pcm[1] = -32768;
  down.feed(packDown(0, KIND.notice, 1, 2, 0, [new Uint8Array(pcm.buffer)]));
  assert.equal(pushes.length, 1);
  assert.deepEqual([pushes[0].rate, pushes[0].kind, pushes[0].pcm.length, pushes[0].pcm[0], pushes[0].pcm[1]], [24000, 'notice', 960, 0.5, -1]);
});

test('downlink: a probe is reported, never played', () => {
  const pushes = [];
  const probes = [];
  const down = createDownlink({ player: { push: (m) => pushes.push(m) }, onProbe: (p) => probes.push(p), now: () => 99 });
  const vec = FIXTURE.down.find((v) => v.name === 'probe-odd-length');
  down.feed(new Uint8Array(Buffer.from(vec.hex, 'hex')));
  assert.equal(pushes.length, 0);
  assert.deepEqual(probes, [{ bytes: vec.hex.length / 2, arrival: 99, gen: 0, seq: 0 }]);
});

test('downlink: the fixture\'s real server frames decode with the right durations', async () => {
  const { codecs } = fakeCodecs();
  const chunks = [];
  const pushes = [];
  const down = createDownlink({ player: { push: (m) => pushes.push(m) }, codecs, onChunk: (c) => chunks.push(c) });
  for (const name of ['opus-reply-200ms', 'pcm16-reply-40ms', 'opus-ack', 'opus-backchannel']) {
    down.feed(new Uint8Array(Buffer.from(FIXTURE.down.find((v) => v.name === name).hex, 'hex')));
  }
  await settle();
  assert.deepEqual(chunks.map((c) => [c.kind, Math.round(c.durationMs)]), [['reply', 200], ['reply', 40], ['ack', 80], ['backchannel', 20]]);
  assert.equal(pushes.filter((p) => p.rate === 48000).length, 15);
  assert.equal(pushes.filter((p) => p.rate === 24000).length, 1);
});

test('downlink: a decoder error is reported once and later Opus frames are dropped', async () => {
  const { codecs } = fakeCodecs({ failDecode: true });
  const errors = [];
  const pushes = [];
  const down = createDownlink({ player: { push: (m) => pushes.push(m) }, codecs, onError: (w) => errors.push(w) });
  down.feed(packDown(1, 0, 0, 1, 0, [opusPacket(1)]));
  await settle();
  down.feed(packDown(1, 0, 0, 1, 0, [opusPacket(2)]));
  await settle();
  assert.deepEqual(errors, ['decoder']);
  assert.equal(down.broken, true);
  assert.equal(pushes.length, 0);
  // PCM still plays (the client reconnects with down=pcm16).
  down.feed(packDown(0, 0, 0, 2, 0, [new Uint8Array(4)]));
  assert.equal(pushes.length, 1);
});

test('downlink: reset() after a barge-in drops what the decoder still holds', async () => {
  const { codecs } = fakeCodecs();
  const pushes = [];
  const down = createDownlink({ player: { push: (m) => pushes.push(m) }, codecs });
  down.feed(packDown(1, 0, 4, 1, 0, [opusPacket(1), opusPacket(2)]));
  down.reset();
  await settle();
  assert.equal(pushes.length, 0);
  down.feed(packDown(1, 0, 5, 1, 0, [opusPacket(3)]));
  await settle();
  assert.deepEqual(pushes.map((p) => p.gen), [5]);
});

// ---- review T8 round 2

test('downlink: a decoder that goes silent (no output, no error) is reported once within ~2.25 s (review r2 M2)', async () => {
  const { codecs } = fakeCodecs();
  class Mute extends codecs.AudioDecoder { decode() { if (this.state !== 'configured') throw new Error('nc'); } }
  const clock = fakeClock(0);
  const errors = [];
  const pushes = [];
  const down = createDownlink({
    player: { push: (m) => pushes.push(m) }, codecs: { ...codecs, AudioDecoder: Mute },
    onError: (w, e) => errors.push([w, String(e.message)]), now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer,
  });
  let failedAt = null;
  for (let i = 0; i < 50 * 10 && failedAt === null; i++) {   // up to 10 s of 20 ms packets
    down.feed(packDown(1, KIND.reply, 0, 1, i * 20, [opusPacket(1)]));
    clock.advance(20);
    if (errors.length) failedAt = clock.now();
  }
  await settle();
  assert.deepEqual(errors, [['decoder', 'decoder stalled']]);
  assert.ok(failedAt >= 2000 && failedAt <= 2400, `failed at ${failedAt} ms`);
  assert.equal(down.broken, true);
  assert.equal(down.pending, 0, 'the pending metadata is released');
  assert.equal(clock.pending, 0, 'no timer left behind');
  // PCM still plays (the client reconnects with down=pcm16).
  down.feed(packDown(0, 0, 0, 2, 0, [new Uint8Array(4)]));
  assert.equal(pushes.length, 1);
});

test('downlink: a burst bigger than the backstop, or a main-thread hitch, is not a decoder stall', async () => {
  const { codecs } = fakeCodecs();
  const clock = fakeClock(0);
  const errors = [];
  const pushes = [];
  const down = createDownlink({ player: { push: (m) => pushes.push(m) }, codecs, onError: (w) => errors.push(w), now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  // A pre-rendered 12 s line in one go: 600 packets fed before any output can run.
  for (let i = 0; i < 20; i++) down.feed(packDown(1, KIND.reply, 0, 1, 0, Array.from({ length: 30 }, () => opusPacket(2))));
  assert.ok(down.pending >= 500);
  await settle();
  clock.advance(1000);
  assert.deepEqual(errors, []);
  assert.equal(pushes.length, 600);
  // A hitch: 30 packets fed, then 3 s pass before their outputs run; more arrive meanwhile.
  for (let i = 0; i < 30; i++) down.feed(packDown(1, KIND.reply, 0, 2, 0, [opusPacket(3)]));
  clock.advance(3000);
  down.feed(packDown(1, KIND.reply, 0, 2, 0, [opusPacket(3)]));   // suspects: oldest held 3 s
  await settle();                                                   // the queued outputs run
  clock.advance(240);
  // Streaming goes on: a packet is in flight (fed, output not yet run) when the check fires.
  down.feed(packDown(1, KIND.reply, 0, 2, 0, [opusPacket(3)]));
  clock.advance(1000);
  assert.deepEqual(errors, [], 'outputs arrived since the suspicion: progress, not a stall');
  assert.equal(down.broken, false);
  await settle();
  assert.equal(pushes.length, 632);
});

test('downlink: flush() on a decoder whose flush never settles gives up after 2 s (review r2 M2)', async () => {
  const { codecs } = fakeCodecs();
  class Hang extends codecs.AudioDecoder { decode() {} flush() { return new Promise(() => {}); } }
  const clock = fakeClock(0);
  const errors = [];
  const down = createDownlink({ player: { push: () => {} }, codecs: { ...codecs, AudioDecoder: Hang }, onError: (w) => errors.push(w), now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  down.feed(packDown(1, KIND.reply, 0, 1, 0, [opusPacket(1)]));
  let done = false;
  const p = down.flush().then(() => { done = true; });
  await settle();
  assert.equal(done, false);
  clock.advance(2000);
  await p;
  assert.deepEqual(errors, ['decoder']);
  assert.equal(clock.pending, 0);
});

test('downlink: reset() cancels a pending stall confirmation', async () => {
  const { codecs } = fakeCodecs();
  class Mute extends codecs.AudioDecoder { decode() {} }
  const clock = fakeClock(0);
  const errors = [];
  const down = createDownlink({ player: { push: () => {} }, codecs: { ...codecs, AudioDecoder: Mute }, onError: (w) => errors.push(w), now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  for (let i = 0; i < 30; i++) down.feed(packDown(1, KIND.reply, 0, 1, 0, [opusPacket(1)]));
  clock.advance(2100);
  down.feed(packDown(1, KIND.reply, 0, 1, 0, [opusPacket(1)]));   // suspected
  assert.equal(clock.pending, 1);
  down.reset();                                                     // barge-in
  clock.advance(1000);
  assert.deepEqual(errors, []);
  assert.equal(clock.pending, 0);
});
