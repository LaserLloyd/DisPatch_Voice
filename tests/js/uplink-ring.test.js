// T8: uplink-ring.js — 30 s replay ring, replay from an offset, the stale rule, cursorFrom()
// (SPEC §5.5, contracts C-§E.2, C.2, C.5 #10).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createUplinkRing, cursorFrom, createReplayPump } from '../../src/dispatch_voice/static/uplink-ring.js';
import { packUp, unpackUp, UP_FLAGS, MAX_UP_PACKETS } from '../../src/dispatch_voice/static/opus-codec.js';
import { fakeClock, opusPacket } from './_t8-fakes.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(join(HERE, '..', 'fixtures', 'wire3.json'), 'utf8'));

/** One live Opus message: 5 × 20 ms packets starting at `off`; payload byte = packet index. */
const opusMsg = (off, flags = 0) => packUp(1, flags, off, Array.from({ length: 5 }, (_, i) => opusPacket(((off / 320) + i) & 0xff)));
const pcmMsg = (off) => packUp(0, 0, off, [new Uint8Array(640).fill((off / 320) & 0xff)]);
/** Fill the ring with `seconds` of contiguous Opus from `start`. */
function fill(ring, seconds, start = 0) {
  const n = Math.round(seconds * 10);
  for (let i = 0; i < n; i++) ring.put(start + i * 1600, opusMsg(start + i * 1600));
  return start + n * 1600;
}
const offsOf = (msgs) => msgs.map((m) => unpackUp(m)).flatMap((f) => f.packets.map((_, i) => f.off + i * 320));

test('cursorFrom = max(lastRxOff, lastTranscriptUpto, lastEotUpto), wrap-aware', () => {
  assert.equal(cursorFrom({ lastRxOff: 16000, lastTranscriptUpto: 24000, lastEotUpto: 20000 }), 24000);
  assert.equal(cursorFrom({ lastRxOff: 48000, lastTranscriptUpto: 24000 }), 48000);
  assert.equal(cursorFrom({ lastEotUpto: 320 }), 320);
  assert.equal(cursorFrom({}), 0);
  assert.equal(cursorFrom({ lastRxOff: null, lastTranscriptUpto: undefined, lastEotUpto: NaN }), 0);
  // Across the u32 wrap the newer value is the "smaller" number.
  assert.equal(cursorFrom({ lastRxOff: 0xfffff000, lastTranscriptUpto: 0x100 }), 0x100);
  assert.equal(cursorFrom({ lastRxOff: 2 ** 32 + 7 }), 7);
});

test('keeps 30 s of Opus and evicts the oldest', () => {
  const ring = createUplinkRing({ seconds: 30, stream: 'abcd1234' });
  const end = fill(ring, 40);
  assert.equal(ring.stream, 'abcd1234');
  assert.ok(ring.seconds <= 30 && ring.seconds >= 29.9, `held ${ring.seconds} s`);
  assert.equal(ring.newest(), end);
  assert.equal(ring.oldest(), end - 30 * 16000);
  // ≈ 59 KB for 30 s at 16 kbps in the SPEC; these test packets are tiny, so just bounded.
  assert.ok(ring.bytes > 0);
});

test('maxBytes bounds a PCM ring (10 s = 320 KB)', () => {
  const ring = createUplinkRing({ seconds: 10, maxBytes: 320000 });
  for (let i = 0; i < 1000; i++) ring.put(i * 320, pcmMsg(i * 320)); // 20 s
  assert.ok(ring.bytes <= 320000);
  assert.ok(ring.seconds <= 10);
  assert.equal(ring.newest(), 1000 * 320);
});

test('replayFrom(off): from the packet containing off, replay flag set, contiguous, ≤ 50 packets per message', () => {
  const ring = createUplinkRing();
  fill(ring, 3); // 0 .. 48000
  const msgs = ring.replayFrom(16000 + 100); // inside the packet starting at 16000
  const frames = msgs.map((m) => unpackUp(m));
  assert.ok(frames.every((f) => f.flags & UP_FLAGS.replay), 'replay bit');
  assert.ok(frames.every((f) => f.packets.length <= MAX_UP_PACKETS));
  assert.equal(frames[0].off, 16000);
  const offs = offsOf(msgs);
  assert.equal(offs.length, (48000 - 16000) / 320);
  for (let i = 1; i < offs.length; i++) assert.equal(offs[i], offs[i - 1] + 320);
  // The packets are the originals, byte for byte.
  assert.deepEqual(frames[0].packets[0], opusPacket(50));
  // 100 packets → two messages of 50.
  assert.deepEqual(frames.map((f) => f.packets.length), [50, 50]);
  // Every replay message is under the 65,536 B the server accepts.
  assert.ok(msgs.every((m) => m.length <= 65536));
});

test('replay keeps local_speaking runs and gate gaps apart', () => {
  const ring = createUplinkRing();
  ring.put(0, opusMsg(0));
  ring.put(1600, opusMsg(1600, UP_FLAGS.localSpeaking));
  ring.put(16000, opusMsg(16000)); // the gate was closed in between
  const frames = ring.replayFrom(0).map((m) => unpackUp(m));
  assert.deepEqual(frames.map((f) => [f.off, f.flags, f.packets.length]), [
    [0, UP_FLAGS.replay, 5],
    [1600, UP_FLAGS.replay | UP_FLAGS.localSpeaking, 5],
    [16000, UP_FLAGS.replay, 5],
  ]);
});

test('PCM replays one 640 B packet per message (C.2)', () => {
  const ring = createUplinkRing({ seconds: 10, maxBytes: 320000 });
  for (let i = 0; i < 10; i++) ring.put(i * 320, pcmMsg(i * 320));
  const frames = ring.replayFrom(960).map((m) => unpackUp(m));
  assert.equal(frames.length, 7);
  assert.ok(frames.every((f) => f.codec === 0 && f.packets.length === 1 && f.packets[0].length === 640));
  assert.deepEqual(frames.map((f) => f.off), [960, 1280, 1600, 1920, 2240, 2560, 2880]);
});

test('a burst that ended more than 20 s ago is not replayed: cutOff says so', () => {
  const ring = createUplinkRing();
  // Three utterances (gate gaps between): 0–2 s, 10–12 s, 24–26 s; the mic clock is at 35 s.
  for (const s0 of [0, 10, 24]) fill(ring, 2, s0 * 16000);
  const nowOff = 35 * 16000;
  const offs = offsOf(ring.replayFrom(0, { nowOff, maxAgeS: 20 }));
  assert.equal(offs[0], 24 * 16000, 'the two stale bursts are skipped, the fresh one replays');
  assert.equal(offs.length, 100);
  assert.equal(ring.cutOff, true);
  ring.replayFrom(24 * 16000, { nowOff });
  assert.equal(ring.cutOff, false, 'nothing stale after the cursor');
});

test('a burst is judged by its LAST sample and never cut in half (amendment T6-1)', () => {
  const ring = createUplinkRing();
  // One utterance from 0 to 8 s; the mic clock is at 25 s: its head is 25 s old, its end 17 s.
  const end = fill(ring, 8);
  const nowOff = 25 * 16000;
  const offs = offsOf(ring.replayFrom(0, { nowOff, maxAgeS: 20 }));
  assert.equal(offs[0], 0, 'replayed from its first sample');
  assert.equal(offs.at(-1), end - 320);
  assert.equal(offs.length, 400);
  assert.equal(ring.cutOff, false);
  // The same burst, ended 21 s ago: skipped whole, never its fresh-looking tail alone.
  assert.deepEqual(ring.replayFrom(0, { nowOff: end + 21 * 16000 }), []);
  assert.equal(ring.cutOff, true);
});

test('the server may answer a cursor BELOW the client\'s own from: replay from the answer (T6-1)', () => {
  const ring = createUplinkRing();
  fill(ring, 2.4); // one utterance, 0–2.4 s; the old socket had received up to 1.2 s
  const clientFrom = cursorFrom({ lastRxOff: 19200 });
  assert.equal(clientFrom, 19200);
  const answer = 0; // the server: the unfinished utterance started at 0
  const offs = offsOf(ring.replayFrom(answer, { nowOff: 3 * 16000 }));
  assert.equal(offs[0], 0, 'the head of the sentence comes back too');
  assert.equal(offs.length, 120);
});

test('nothing after the cursor → nothing to replay; trim() forgets what the server has', () => {
  const ring = createUplinkRing();
  const end = fill(ring, 2);
  assert.deepEqual(ring.replayFrom(end), []);
  ring.trim(16000);
  assert.equal(ring.oldest(), 16000);
  ring.trim(end);
  assert.equal(ring.size, 0);
  assert.equal(ring.oldest(), null);
});

test('a stream that restarts at a lower offset drops the old clock', () => {
  const ring = createUplinkRing();
  fill(ring, 2, 160000);
  ring.put(0, opusMsg(0));
  assert.equal(ring.size, 1);
  assert.equal(ring.oldest(), 0);
});

test('offsets near the u32 wrap replay in order', () => {
  const ring = createUplinkRing();
  const start = 0xffffffff - 1599; // the next message starts at 0 + ...
  for (let i = 0; i < 4; i++) ring.put((start + i * 1600) >>> 0, packUp(1, 0, (start + i * 1600) >>> 0, Array.from({ length: 5 }, () => opusPacket(1))));
  const offs = offsOf(ring.replayFrom(start));
  assert.equal(offs.length, 20);
  for (let i = 1; i < offs.length; i++) assert.equal((offs[i] - offs[i - 1]) >>> 0, 320);
});

test('the shared fixture\'s real libopus uplink messages go through the ring unchanged', () => {
  const live = FIXTURE.up.find((v) => v.name === 'opus-live-100ms');
  assert.ok(live, 'fixture vector opus-live-100ms');
  const ring = createUplinkRing();
  ring.put(live.off, new Uint8Array(Buffer.from(live.hex, 'hex')));
  const [msg] = ring.replayFrom(live.off);
  const f = unpackUp(msg);
  assert.equal(f.flags, UP_FLAGS.replay);
  assert.deepEqual(f.packets.map((p) => Buffer.from(p).toString('hex')), live.packets);
  assert.equal(ring.seconds, 0.1); // 5 real 20 ms packets, read from their TOC bytes
});

test('replay pump paces at 4× realtime and reports done', () => {
  const clock = fakeClock();
  const sent = [];
  let done = 0;
  const pump = createReplayPump({ send: (b) => sent.push([clock.now(), b]), onDone: () => { done += 1; }, setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  const ring = createUplinkRing({ seconds: 10, maxBytes: 320000 });
  for (let i = 0; i < 50; i++) ring.put(i * 320, pcmMsg(i * 320)); // 1 s of PCM
  pump.start(ring.replayFrom(0));
  assert.equal(sent.length, 1);
  clock.advance(1000);
  assert.equal(sent.length, 50);
  assert.equal(done, 1);
  // 20 ms per message at 4× → 5 ms apart: 1 s of audio in ≈ 250 ms.
  assert.equal(sent[49][0], 49 * 5);
  // stop() mid-way sends nothing more.
  sent.length = 0;
  pump.start(ring.replayFrom(0));
  pump.stop();
  clock.advance(1000);
  assert.equal(sent.length, 1);
  assert.equal(pump.pending, 0);
});
