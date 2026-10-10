// T8: the v3 binary frames in JS (contracts C.2, §G). The hand-built vectors below are
// read straight off the C.2 byte layout; the shared fixture (tests/fixtures/wire3.json,
// owned by T6) is round-tripped too, so Python and JS are held to the same bytes.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  packUp, unpackUp, packDown, unpackDown, isV3, wire3, WireError, MAX_UP_PACKETS, MAX_FRAME_BYTES,
  opusPacketMs, packetSamples16k,
} from '../../src/dispatch_voice/static/opus-codec.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(join(HERE, '..', 'fixtures', 'wire3.json'), 'utf8'));
const hex = (u8) => Buffer.from(u8).toString('hex');
const unhex = (h) => new Uint8Array(Buffer.from(h, 'hex'));

test('uplink: the C.2 layout byte for byte', () => {
  const f = packUp(1, 0b01, 0x01020304, [unhex('aabb'), unhex('cc')]);
  //            magic codec flags off(LE)   count len   bytes  len   bytes
  assert.equal(hex(f), '03' + '01' + '01' + '04030201' + '02' + '0200aabb' + '0100cc');
  assert.deepEqual(unpackUp(f), { codec: 1, flags: 1, off: 0x01020304, packets: [unhex('aabb'), unhex('cc')] });
  // PCM: one 640 B packet per message.
  const pcm = new Uint8Array(640).map((_, i) => i & 0xff);
  const p = packUp(0, 0, 320, [pcm]);
  assert.equal(p.length, 8 + 2 + 640);
  assert.equal(hex(p.subarray(0, 10)), '03' + '00' + '00' + '40010000' + '01' + '8002');
  assert.deepEqual(unpackUp(p).packets[0], pcm);
});

test('downlink: the C.2 layout byte for byte', () => {
  const f = packDown(1, 4, 0x1234, 7, 0xdeadbeef, [unhex('48aa')]);
  //            magic codec kind gen   seq       t         count len  bytes
  assert.equal(hex(f), '03' + '01' + '04' + '3412' + '07000000' + 'efbeadde' + '01' + '0200' + '48aa');
  assert.deepEqual(unpackDown(f), { codec: 1, kind: 4, gen: 0x1234, seq: 7, t: 0xdeadbeef, packets: [unhex('48aa')] });
});

test('offsets, gen, seq and t wrap like the wire (u32 / u16)', () => {
  const pk = [unhex('0000')];
  assert.equal(unpackUp(packUp(0, 0, 2 ** 32 + 5, pk)).off, 5);
  assert.equal(unpackUp(packUp(0, 0, -1, pk)).off, 0xffffffff);
  assert.equal(unpackUp(packUp(0, 0, 0xffffffff, pk)).off, 0xffffffff);
  const d = unpackDown(packDown(0, 0, 0x10001, 2 ** 32 + 2, 2 ** 32 + 3, pk));
  assert.deepEqual([d.gen, d.seq, d.t], [1, 2, 3]);
});

test('bad frames throw WireError (as unpack_up/unpack_down raise ValueError)', () => {
  const good = packUp(1, 0, 0, [unhex('48')]);
  const bad = [
    ['bad magic', Uint8Array.from([0x02, ...good.subarray(1)])],
    ['short header', good.subarray(0, 7)],
    ['truncated packet', good.subarray(0, good.length - 1)],
    ['missing length', Uint8Array.from([...good.subarray(0, 7), 1, 5])],
    ['trailing bytes', Uint8Array.from([...good, 0])],
    ['unknown codec', Uint8Array.from([3, 7, ...good.subarray(2)])],
  ];
  for (const [why, b] of bad) assert.throws(() => unpackUp(b), WireError, why);
  const over = new Uint8Array(8 + 51 * 3);
  over.set([3, 1, 0, 0, 0, 0, 0, 51]);
  for (let i = 0; i < 51; i++) over.set([1, 0, 0x48], 8 + i * 3);
  assert.throws(() => unpackUp(over), WireError, '> 50 packets');
  const dgood = packDown(1, 0, 1, 1, 1, [unhex('48')]);
  assert.throws(() => unpackDown(dgood.subarray(0, 13)), WireError);
  assert.throws(() => unpackDown(Uint8Array.from([9, ...dgood.subarray(1)])), WireError);
  assert.throws(() => unpackDown(dgood.subarray(0, dgood.length - 1)), WireError);
  // The rules transport/wire3.py also enforces (one fixture holds both sides to them).
  assert.throws(() => unpackUp(unhex('0301004006000000')), WireError, 'count zero');
  assert.throws(() => unpackUp(unhex('03010040060000010000')), WireError, 'zero-length packet');
  assert.throws(() => unpackUp(unhex('03000000000000010300010203')), WireError, 'odd PCM');
  assert.throws(() => unpackUp(new Uint8Array(MAX_FRAME_BYTES + 1).fill(3)), WireError, 'over 64 KiB');
  assert.throws(() => unpackDown(unhex('03010501000100000005000000010100fc')), WireError, 'unknown kind');
  assert.throws(() => unpackDown(unhex('0301000100010000000500000000')), WireError, 'count zero');
  // A probe carries random bytes: odd length is fine there.
  assert.equal(unpackDown(unhex('03000300000000000000fa00000107009f03c4e2b17a5d')).packets[0].length, 7);
});

test('packing refuses what the server would drop', () => {
  assert.throws(() => packUp(1, 0, 0, Array.from({ length: MAX_UP_PACKETS + 1 }, () => unhex('48'))), RangeError);
  assert.throws(() => packUp(0, 0, 0, [new Uint8Array(MAX_FRAME_BYTES)]), RangeError);
  assert.throws(() => packUp(2.5, 0, 0, []), RangeError);
  assert.throws(() => packDown(0, 300, 0, 0, 0, []), RangeError);
  assert.throws(() => packUp(0, 0, 0, [new Uint8Array(70000)]), RangeError);
  assert.throws(() => packUp(0, 0, 0, []), RangeError, 'no packets');
  assert.throws(() => packUp(1, 0, 0, [new Uint8Array(0)]), RangeError, 'empty packet');
  assert.throws(() => packUp(0, 0, 0, [new Uint8Array(3)]), RangeError, 'odd PCM');
  assert.throws(() => packDown(0, 5, 0, 0, 0, [unhex('00')]), RangeError, 'unknown kind');
  assert.equal(packDown(0, 3, 0, 0, 0, [unhex('010203')]).length, 14 + 2 + 3); // probe: odd is fine
});

test('isV3 tells a v3 frame from v2 raw PCM, and wire3 bundles the four functions', () => {
  assert.equal(isV3(packUp(0, 0, 0, [unhex('0000')])), true);
  assert.equal(isV3(new Uint8Array([0, 0, 1, 2])), false);
  assert.equal(isV3(new ArrayBuffer(0)), false);
  assert.deepEqual(Object.keys(wire3).sort(), ['isV3', 'packDown', 'packUp', 'unpackDown', 'unpackUp']);
});

test('Opus TOC durations (RFC 6716 §3.1) and 16 kHz sample counts', () => {
  assert.equal(opusPacketMs(unhex('48')), 20);       // SILK WB 20 ms, code 0
  assert.equal(opusPacketMs(unhex('40')), 10);       // SILK WB 10 ms
  assert.equal(opusPacketMs(unhex('f8')), 20);       // CELT FB 20 ms
  assert.equal(opusPacketMs(unhex('f9')), 40);       // code 1: two frames
  assert.equal(opusPacketMs(unhex('fb03')), 60);     // code 3: three frames
  assert.equal(opusPacketMs(unhex('60')), 10);       // hybrid SWB 10 ms (config 12)
  assert.equal(opusPacketMs(unhex('68')), 20);       // hybrid SWB 20 ms (config 13)
  assert.equal(opusPacketMs(new Uint8Array(0)), 0);
  assert.equal(packetSamples16k(1, unhex('48')), 320);
  assert.equal(packetSamples16k(0, new Uint8Array(640)), 320);
});

const fixtureFilled = FIXTURE.status !== 'placeholder'
  && ['up', 'down', 'bad_up', 'bad_down'].some((k) => (FIXTURE[k] || []).length);

test('shared fixture wire3.json: JS packs the same bytes and round-trips them', {
  skip: !fixtureFilled && 'tests/fixtures/wire3.json is still the day-0 placeholder (T6 fills it)',
}, () => {
  for (const v of FIXTURE.up) {
    const f = packUp(v.codec, v.flags, v.off, v.packets.map(unhex));
    assert.equal(hex(f), v.hex.toLowerCase(), `up ${v.name}`);
    const u = unpackUp(unhex(v.hex));
    assert.deepEqual([u.codec, u.flags, u.off, u.packets.map(hex)], [v.codec, v.flags, v.off >>> 0, v.packets.map((p) => p.toLowerCase())], `up ${v.name}`);
  }
  for (const v of FIXTURE.down) {
    const f = packDown(v.codec, v.kind, v.gen, v.seq, v.t, v.packets.map(unhex));
    assert.equal(hex(f), v.hex.toLowerCase(), `down ${v.name}`);
    const d = unpackDown(unhex(v.hex));
    assert.deepEqual([d.codec, d.kind, d.gen, d.seq, d.t, d.packets.map(hex)],
      [v.codec, v.kind, v.gen & 0xffff, v.seq >>> 0, v.t >>> 0, v.packets.map((p) => p.toLowerCase())], `down ${v.name}`);
  }
  for (const v of FIXTURE.bad_up) assert.throws(() => unpackUp(unhex(v.hex)), WireError, `bad_up ${v.name}: ${v.why}`);
  for (const v of FIXTURE.bad_down) assert.throws(() => unpackDown(unhex(v.hex)), WireError, `bad_down ${v.name}: ${v.why}`);
});
