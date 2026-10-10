// T8: link-monitor.js — degraded / recovered on the SPEC §5.3 rules with hysteresis and dwell,
// predictive vs reactive timing (experience M12), the codec ladder, TTS-vs-link cause split
// (contracts C-§E.2, C.3 `link`). Fake clock throughout.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createLinkMonitor, LINK_THRESHOLDS, heartbeatDeadMs, linkFrame } from '../../src/dispatch_voice/static/link-monitor.js';
import { fakeClock } from './_t8-fakes.js';

function setup(opts = {}) {
  const clock = fakeClock(1_000_000);
  const links = [];
  const backlogs = [];
  const m = createLinkMonitor({
    now: clock.now,
    onChange: (e) => ('mode' in e ? links : backlogs).push(e),
    ...opts,
  });
  return { clock, m, links, backlogs };
}
/** n pongs 2 s apart at `rtt` ms. */
function pongs(ctx, n, rtt) { for (let i = 0; i < n; i++) { ctx.clock.advance(2000); ctx.m.onPong(rtt); ctx.m.tick(); } }
/** Stream `ms` of reply audio for seq: 100 ms chunks, `skewGrow` ms of extra delay per chunk. */
function stream(ctx, seq, ms, { skewGrow = 0, tServer0 = 0 } = {}) {
  let tS = tServer0;
  for (let t = 0; t < ms; t += 100) {
    ctx.clock.advance(100 + skewGrow);
    tS += 100;
    ctx.m.onChunk(seq, ctx.clock.now(), tS, 100, 200, 300);
    ctx.m.tick();
  }
}

test('LINK_THRESHOLDS carries the SPEC §5.3 numbers in one frozen object', () => {
  assert.ok(Object.isFrozen(LINK_THRESHOLDS));
  const t = LINK_THRESHOLDS;
  assert.deepEqual([t.pingEveryMs, t.rttDegradedMs, t.rttDegradedPongs, t.stallMs, t.underrunsPerReply, t.bigGapMs],
    [2000, 1000, 3, 5000, 2, 400]);
  assert.deepEqual([t.goodputFactor, t.goodputHoldMs, t.firstAudioMinMs, t.firstAudioPadMs, t.healthyMs, t.rttHealthyMs],
    [1.3, 3000, 5000, 2000, 30000, 500]);
  assert.deepEqual([t.probeBytes, t.probeFactor, t.probeEveryMs, t.dwellMs, t.dwellMaxMs], [16384, 2, 20000, 60000, 600000]);
  assert.deepEqual([...t.rungs.down], [24, 16]);
  assert.deepEqual([...t.rungs.up], [16, 12]);
});

test('RTT EWMA > 1 s on 3 consecutive pongs: the ladder first (24→16 / 16→12), then text', () => {
  const ctx = setup();
  pongs(ctx, 5, 250);
  assert.equal(ctx.links.length, 0, 'an LTE/DERP median never trips');
  pongs(ctx, 1, 1600); // a single spike (EWMA ≈ 655 ms)
  assert.equal(ctx.links.length, 0, 'one spike is not a slow link');
  pongs(ctx, 5, 1600);
  assert.deepEqual(ctx.links[0], { mode: 'audio', downKbps: 16, upKbps: 12, why: 'rtt' });
  pongs(ctx, 6, 1600); // past the ladder grace, still slow
  assert.deepEqual(ctx.links.at(-1), { mode: 'text', downKbps: 16, upKbps: 12, why: 'rtt' });
  assert.equal(ctx.m.current().mode, 'text');
});

test('predictive switches wait for say_end; reactive ones go at once', () => {
  const ctx = setup();
  pongs(ctx, 4, 1600); // → ladder rung (never held: the voice does not change)
  assert.equal(ctx.links.length, 1);
  ctx.clock.advance(11000);
  ctx.m.onSay(1, 0, 'reply');
  ctx.m.onChunk(1, ctx.clock.now(), 0, 100);
  pongs(ctx, 3, 1600);
  assert.equal(ctx.links.length, 1, 'rtt mid-reply: held');
  assert.equal(ctx.m.stats().pending, 'rtt');
  ctx.m.onSayEnd();
  assert.deepEqual(ctx.links.at(-1), { mode: 'text', downKbps: 16, upKbps: 12, why: 'rtt' });
  // Reactive: a stall mid-reply switches immediately.
  const r = setup();
  r.m.onSay(1, 0, 'reply');
  r.m.onStall();
  assert.deepEqual(r.links, [{ mode: 'text', downKbps: 24, upKbps: 16, why: 'stall' }]);
});

test('a pong more than 5 s late is a stall (straight to text)', () => {
  const ctx = setup();
  ctx.m.onPong(6200);
  assert.deepEqual(ctx.links, [{ mode: 'text', downKbps: 24, upKbps: 16, why: 'stall' }]);
});

test('underruns count only when the skew grows (the link), never for a TTS hole', () => {
  // TTS hole: chunks are late at the server too (t_server jumps with arrival) → flat skew.
  const tts = setup();
  tts.m.onSay(1, 0, 'reply');
  stream(tts, 1, 2000);
  tts.clock.advance(1300);
  tts.m.onChunk(1, tts.clock.now(), 2000 + 1300, 100);
  tts.m.onUnderrun(1, 1300);
  tts.m.onUnderrun(1, 1200);
  assert.equal(tts.links.length, 0, 'Chatterbox\'s ~1.2 s hole is not the network');
  // Link: arrival drifts behind t_server.
  const link = setup();
  link.m.onSay(1, 0, 'reply');
  stream(link, 1, 1500, { skewGrow: 30 });
  assert.ok(link.m.stats().skew_ms >= 300);
  link.m.onUnderrun(1, 120);
  assert.equal(link.links.length, 0, 'one short underrun is not enough');
  link.m.onUnderrun(1, 150);
  assert.deepEqual(link.links[0], { mode: 'audio', downKbps: 16, upKbps: 12, why: 'underrun' });
});

test('one gap > 400 ms with a growing skew trips at once', () => {
  const ctx = setup();
  ctx.m.onSay(1, 0, 'reply');
  stream(ctx, 1, 1500, { skewGrow: 30 });
  ctx.m.onUnderrun(1, 450);
  assert.equal(ctx.links[0].why, 'underrun');
});

test('a missed first-audio deadline (max(5 s, p90 + 2 s)) goes straight to text', () => {
  const ctx = setup();
  ctx.m.onSay(3, 0, 'reply');
  ctx.clock.advance(5900);
  ctx.m.tick();
  assert.equal(ctx.links.length, 0, 'prior p90 4 s + 2 s = 6 s');
  ctx.clock.advance(200);
  ctx.m.tick();
  assert.deepEqual(ctx.links, [{ mode: 'text', downKbps: 24, upKbps: 16, why: 'deadline' }]);
  // Calibrated by first_chunk_ms; the first chunk clears the deadline.
  const c = setup();
  for (const ms of [1000, 1100, 1200]) c.m.onMetrics({ first_chunk_ms: ms });
  c.m.onSay(1, 0, 'reply');
  c.clock.advance(4900);
  c.m.onChunk(1, c.clock.now(), 0, 100);
  c.clock.advance(5000);
  c.m.tick();
  assert.equal(c.links.length, 0);
  c.m.onSay(2, 0, 'reply');
  c.clock.advance(5001);
  c.m.tick();
  assert.equal(c.links.at(-1).why, 'deadline', 'max(5 s, 1.2 + 2 s) = 5 s');
});

test('no deadline in text mode (no audio is coming)', () => {
  const ctx = setup();
  ctx.m.startMode('2g');
  ctx.m.onSay(1, 0, 'reply');
  ctx.clock.advance(20000);
  ctx.m.tick();
  assert.equal(ctx.links.length, 0);
});

test('goodput < 1.3 × needed for 3 s while streaming and falling behind', () => {
  const ctx = setup();
  ctx.m.onSay(1, 0, 'reply');
  // 100 ms of audio per 160 ms of wall time, 300 B each ≈ 15 kbps against 30 needed.
  let tS = 0;
  for (let i = 0; i < 40; i++) {
    ctx.clock.advance(160);
    tS += 100;
    ctx.m.onChunk(1, ctx.clock.now(), tS, 100, 0, 300);
    ctx.m.tick();
  }
  assert.ok(ctx.links.some((l) => l.why === 'goodput'), JSON.stringify(ctx.links));
  // Fast server, fast link: never.
  const ok = setup();
  ok.m.onSay(1, 0, 'reply');
  stream(ok, 1, 6000);
  assert.equal(ok.links.length, 0);
});

test('recovery: dwell, 30 s healthy, between replies, RTT < 500 ms, then a probe at ≥ 2 × needed', () => {
  const ctx = setup();
  ctx.m.onStall();
  assert.equal(ctx.m.current().mode, 'text');
  pongs(ctx, 10, 200); // 20 s: inside the 60 s dwell
  assert.equal(ctx.m.wantProbe(), false);
  pongs(ctx, 21, 200); // 62 s since the stall, ≥ 30 s healthy
  ctx.m.onSay(5, 0, 'reply');
  assert.equal(ctx.m.wantProbe(), false, 'never mid-reply');
  ctx.m.onSayEnd();
  assert.equal(ctx.m.wantProbe(), true);
  assert.equal(ctx.m.wantProbe(), false, 'one probe in flight');
  // 16 KB in 200 ms + RTT ≈ 650 kbps — passes 2 × 20 kbps for the 16 kbps rung.
  ctx.m.onProbe(16384, 400);
  assert.deepEqual(ctx.links.at(-1), { mode: 'audio', downKbps: 16, upKbps: 12, why: 'recovered' });
  // The next step up needs another dwell + probe.
  assert.equal(ctx.m.wantProbe(), false);
  pongs(ctx, 31, 200);
  assert.equal(ctx.m.wantProbe(), true);
  ctx.m.onProbe(16384, 400);
  assert.deepEqual(ctx.links.at(-1), { mode: 'audio', downKbps: 24, upKbps: 16, why: 'recovered' });
  assert.equal(ctx.m.resendFrame(), null, 'back to the defaults');
});

test('a probe that is too slow does not recover; probes at most every 20 s', () => {
  const ctx = setup();
  ctx.m.onStall();
  pongs(ctx, 31, 200);
  assert.equal(ctx.m.wantProbe(), true);
  ctx.m.onProbe(16384, 6000); // ≈ 22 kbps < 40
  assert.equal(ctx.m.current().mode, 'text');
  ctx.clock.advance(10000);
  assert.equal(ctx.m.wantProbe(), false, '≤ 1 probe / 20 s');
  ctx.clock.advance(10000);
  ctx.m.onPong(200);
  assert.equal(ctx.m.wantProbe(), true);
});

test('a bad spell restarts the healthy clock', () => {
  const ctx = setup();
  ctx.m.onStall();
  pongs(ctx, 25, 200);
  ctx.m.onPong(2000); // EWMA 0.3 × 2000 + 0.7 × 200 = 740 ≥ 500 → not healthy (and the next pong too)
  pongs(ctx, 10, 200);
  assert.equal(ctx.m.wantProbe(), false);
  pongs(ctx, 15, 200);
  assert.equal(ctx.m.wantProbe(), true);
});

test('dwell doubles per flap within 5 min, capped at 10 min; resets after a calm spell', () => {
  const ctx = setup();
  const cycle = () => {
    ctx.m.onStall();
    const dwell = ctx.m.stats().dwell_ms;
    pongs(ctx, Math.ceil(Math.max(dwell, 30000) / 2000) + 1, 100);
    assert.equal(ctx.m.wantProbe(), true);
    ctx.m.onProbe(16384, 300); // → audio low rung
    pongs(ctx, Math.ceil(Math.max(dwell, 30000) / 2000) + 1, 100);
    if (ctx.m.wantProbe()) ctx.m.onProbe(16384, 300);
    return dwell;
  };
  assert.equal(cycle(), 60000);
  assert.equal(cycle(), 120000);
  assert.equal(cycle(), 240000);
  assert.equal(cycle(), 480000);
  assert.equal(cycle(), 600000);
  assert.equal(cycle(), 600000, 'cap');
  ctx.clock.advance(400000); // a calm 6+ minutes since the last recovery
  ctx.m.onStall();
  assert.equal(ctx.m.stats().dwell_ms, 60000);
});

test('startMode: a 2g-or-worse start is degraded, probes up later; 3g starts in audio; SCO on the 16 kbps rung', () => {
  const slow = setup();
  assert.deepEqual(slow.m.startMode('2g'), { mode: 'text', downKbps: 24, upKbps: 16 });
  assert.deepEqual(slow.m.resendFrame(), { type: 'link', mode: 'text', why: 'rtt', down_kbps: 24, up_kbps: 16 });
  pongs(slow, 16, 150);
  assert.equal(slow.m.wantProbe(), true);
  const fast = setup();
  assert.deepEqual(fast.m.startMode('4g'), { mode: 'audio', downKbps: 24, upKbps: 16 });
  assert.equal(fast.m.resendFrame(), null);
  const sco = setup();
  assert.deepEqual(sco.m.startMode('4g', { bluetooth: true }), { mode: 'audio', downKbps: 16, upKbps: 16 });
  for (const t of ['slow-2g', '2g']) assert.equal(setup().m.startMode(t).mode, 'text');
  // Chrome says '3g' for RTT ≥ 270 ms — the measured LTE/relay median; 28 kbps Opus carries.
  assert.equal(setup().m.startMode('3g').mode, 'audio');
});

// ---- review T8 fixes (H1, M1, M4, L2)

test('a settled +300 ms skew step and a 3.5 s first-chunk wait never degrade (review H1)', () => {
  const ctx = setup();
  ctx.m.onReady();
  // Reply 1: server faster than realtime; a one-time +300 ms step (cell handover) 1 s in.
  ctx.m.onSay(1, 0, 'reply');
  for (let i = 0; i < 20; i++) {
    const skew = i < 10 ? 100 : 400;
    ctx.m.onChunk(1, ctx.clock.now(), ctx.clock.now() - skew, 200, 1000, 700);
    ctx.clock.advance(100); ctx.m.tick();
  }
  ctx.m.onSayEnd();
  pongs(ctx, 5, 250);
  // Reply 2: Chatterbox takes 3.5 s for the first chunk, then streams at realtime with the
  // settled +400 ms offset (wire bytes/s < 1.3 × need — the ratio alone would trip).
  ctx.m.onSay(2, 0, 'reply');
  for (let i = 0; i < 14; i++) { ctx.clock.advance(250); ctx.m.tick(); }
  let tS = 50_000;
  for (let i = 0; i < 60; i++) {
    tS += 100;
    ctx.clock.advance(100);
    ctx.m.onChunk(2, ctx.clock.now(), tS + (i % 3) * 40, 100, 200, 350); // ± jitter
    ctx.m.tick();
  }
  ctx.m.onSayEnd();
  assert.deepEqual(ctx.links, []);
  // A step inside one reply, then flat: the goodput rule does not fire either (skew not rising).
  const step = setup();
  step.m.onSay(1, 0, 'reply');
  let s2 = 0;
  for (let i = 0; i < 80; i++) {
    s2 += 100;
    step.clock.advance(100);
    step.m.onChunk(1, step.clock.now(), s2 - (i < 10 ? 0 : 400), 100, 200, 350);
    step.m.tick();
  }
  assert.deepEqual(step.links, []);
});

test('a slow, steady deficit (skew rising every window) still trips goodput (review H1)', () => {
  const ctx = setup();
  ctx.m.onSay(1, 0, 'reply');
  // The link carries 0.8 × what the server sends: 100 ms of audio per 125 ms of wall time.
  let tS = 0;
  for (let i = 0; i < 100; i++) {
    tS += 100;
    ctx.clock.advance(125);
    ctx.m.onChunk(1, ctx.clock.now(), tS, 100, 0, 350);
    ctx.m.tick();
  }
  assert.equal(ctx.links[0]?.why, 'goodput', JSON.stringify(ctx.links));
});

test('say_end and turn_error clear first-audio deadlines (review M1)', () => {
  const end = setup();
  end.m.onSay(1, 0, 'reply');
  end.clock.advance(300);
  end.m.onSayEnd();                       // tts_failed / no_speech_in_reply: the reply closed
  for (let i = 0; i < 40; i++) { end.clock.advance(250); end.m.tick(); }
  assert.deepEqual(end.links, []);
  const err = setup();
  err.m.onSay(1, 0, 'reply');
  err.m.onSay(2, 0, 'reply');
  err.m.onTurnError(1);
  err.clock.advance(6100); err.m.tick();
  assert.equal(err.links.length, 1, 'seq 2 still armed');
  const all = setup();
  all.m.onSay(1, 0, 'reply'); all.m.onSay(2, 0, 'reply');
  all.m.onTurnError();
  all.clock.advance(10_000); all.m.tick();
  assert.deepEqual(all.links, []);
});

test('no deadline for a say landing right after the switch back to audio (text-mode race)', () => {
  const ctx = setup();
  ctx.m.onStall();
  pongs(ctx, 31, 200);
  assert.equal(ctx.m.wantProbe(), true);
  ctx.m.onProbe(16384, 400);
  assert.equal(ctx.m.current().mode, 'audio');
  ctx.m.onSay(9, 0, 'reply');             // the server's text-mode say crossing our link frame
  ctx.clock.advance(7000); ctx.m.tick();
  assert.equal(ctx.m.current().mode, 'audio');
  ctx.m.onSayEnd();
  ctx.m.onSay(10, 0, 'reply');            // later: armed as usual
  ctx.clock.advance(6100); ctx.m.tick();
  assert.equal(ctx.links.at(-1).why, 'deadline');
});

test('an unanswered or refused probe does not block recovery for good (review M4)', () => {
  const lost = setup();
  lost.m.onStall();
  pongs(lost, 31, 80);
  assert.equal(lost.m.wantProbe(), true);
  pongs(lost, 10, 80);                    // 20 s, no answer: still out
  assert.equal(lost.m.wantProbe(), false);
  pongs(lost, 6, 80);                     // 32 s: written off
  assert.equal(lost.m.wantProbe(), true);
  // A late answer to a written-off probe is measured but does not recover.
  const late = setup();
  late.m.onStall();
  pongs(late, 31, 80);
  assert.equal(late.m.wantProbe(), true);
  pongs(late, 16, 80);
  assert.ok(late.m.onProbe(16384, 300) > 100);
  assert.equal(late.m.current().mode, 'text');
  const refused = setup();
  refused.m.onStall();
  pongs(refused, 31, 80);
  assert.equal(refused.m.wantProbe(), true);
  refused.m.probeFailed();                // error {rate_limited}
  assert.equal(refused.m.wantProbe(), false, 'still ≤ 1 per 20 s');
  pongs(refused, 10, 80);
  assert.equal(refused.m.wantProbe(), true);
});

test('the climb back is one dwell: text → 16 → 24 kbps on consecutive probes (review L2)', () => {
  const ctx = setup();
  ctx.m.onPong(80);
  ctx.m.onStall();
  const t0 = ctx.clock.now();
  let top = null;
  for (let s = 0; s < 600 && top === null; s++) {
    ctx.clock.advance(1000);
    if (s % 2 === 0) ctx.m.onPong(80);
    ctx.m.tick();
    if (ctx.m.wantProbe()) ctx.m.onProbe(16384, 100);
    const st = ctx.m.current();
    if (st.mode === 'audio' && st.downKbps === 24) top = ctx.clock.now() - t0;
  }
  assert.ok(top !== null && top <= 61_000 + 20_000, `top after ${top} ms`);
  assert.deepEqual(ctx.links.map((l) => [l.mode, l.downKbps, l.why]),
    [['text', 24, 'stall'], ['audio', 16, 'recovered'], ['audio', 24, 'recovered']]);
});

test('PCM downlink: no ladder, the mode switch is the first step; kbps fields omitted', () => {
  const ctx = setup({ codec: { down: 'pcm16', up: 'pcm16' } });
  pongs(ctx, 4, 1600);
  assert.deepEqual(ctx.links, [{ mode: 'text', downKbps: null, upKbps: null, why: 'rtt' }]);
  assert.deepEqual(linkFrame(ctx.links[0]), { type: 'link', mode: 'text', why: 'rtt' });
  // A decoder failure moves the downlink to PCM; the ladder then skips the down rung.
  const d = setup();
  d.m.setCodec({ down: 'pcm16' });
  assert.deepEqual(d.m.current(), { mode: 'audio', downKbps: null, upKbps: 16 });
});

test('backlog_s from pong.rx_off: shown when > 2 s, cleared under 0.3 s; never a mode switch', () => {
  const ctx = setup();
  ctx.m.onPong(400, 0, 16000 * 3);
  assert.deepEqual(ctx.backlogs.at(-1), { backlogS: 2.8, slow: true });
  ctx.m.onPong(400, 16000 * 2, 16000 * 3);
  assert.deepEqual(ctx.backlogs.at(-1), { backlogS: 0.8, slow: true });
  ctx.m.onPong(400, 16000 * 3, 16000 * 3);
  assert.deepEqual(ctx.backlogs.at(-1), { backlogS: 0, slow: false });
  ctx.m.onPong(400, 0xffffff00, 0x100); // across the wrap
  assert.ok(ctx.backlogs.at(-1).backlogS < 0.1);
  assert.equal(ctx.links.length, 0);
});

test('"Always use the phone\'s voice" pins text (why user) and blocks recovery', () => {
  const ctx = setup();
  ctx.m.setPinned('text');
  assert.deepEqual(ctx.links, [{ mode: 'text', downKbps: 24, upKbps: 16, why: 'user' }]);
  pongs(ctx, 40, 100);
  assert.equal(ctx.m.wantProbe(), false);
  ctx.m.onStall();
  assert.equal(ctx.links.length, 1);
  ctx.m.setPinned(null);
  assert.deepEqual(ctx.links.at(-1), { mode: 'audio', downKbps: 24, upKbps: 16, why: 'user' });
});

test('a held switch is applied on close; onReady keeps the mode for resendFrame()', () => {
  const ctx = setup();
  pongs(ctx, 4, 1600);
  ctx.clock.advance(11000);
  ctx.m.onSay(1, 0, 'reply');
  ctx.m.onChunk(1, ctx.clock.now(), 0, 100); // audio is flowing: no deadline
  pongs(ctx, 3, 1600);
  ctx.m.onClose();
  assert.equal(ctx.links.at(-1).mode, 'text');
  ctx.m.onReady();
  assert.deepEqual(ctx.m.resendFrame(), { type: 'link', mode: 'text', why: 'rtt', down_kbps: 16, up_kbps: 12 });
});

test('heartbeat: dead after 8 s, or 4 × the RTT EWMA once it is over 1.5 s', () => {
  assert.equal(heartbeatDeadMs(null), 8000);
  assert.equal(heartbeatDeadMs(1400), 8000);
  assert.equal(heartbeatDeadMs(2500), 10000);
  const ctx = setup();
  pongs(ctx, 6, 3000);
  assert.ok(ctx.m.deadAfterMs > 8000);
});

// ---- review T8 round 2

test('a text ↔ 16 kbps flap doubles the dwell like any other (review r2 M1)', () => {
  const ctx = setup();
  pongs(ctx, 1, 80);
  ctx.m.onStall();                                  // → text
  const dwells = [ctx.m.stats().dwell_ms];
  for (let flap = 0; flap < 5; flap++) {
    let stepped = false;
    for (let s = 0; s < 1200 && !stepped; s++) {
      ctx.clock.advance(1000);
      if (s % 2 === 0) ctx.m.onPong(80);
      ctx.m.tick();
      if (ctx.m.wantProbe()) ctx.m.onProbe(16384, 100);
      stepped = ctx.m.current().mode === 'audio';
    }
    assert.ok(stepped, `recovered to the low rung (flap ${flap})`);
    assert.equal(ctx.m.current().downKbps, 16);
    ctx.clock.advance(2000);
    ctx.m.onStall();                                // the 16 kbps rung fails again
    assert.equal(ctx.m.current().mode, 'text');
    dwells.push(ctx.m.stats().dwell_ms);
  }
  assert.deepEqual(dwells, [60000, 120000, 240000, 480000, 600000, 600000]);
});

test('the ladder → text step inside one degraded spell does not double the dwell', () => {
  const ctx = setup();
  pongs(ctx, 3, 1500);                              // rtt → the 16 kbps rung (spell starts, dwell 60 s)
  assert.equal(ctx.m.current().downKbps, 16);
  ctx.clock.advance(11000);
  pongs(ctx, 3, 1500);                              // still bad after the grace → text
  assert.equal(ctx.m.current().mode, 'text');
  assert.equal(ctx.m.stats().dwell_ms, 60000);
});

test('a notice or ack say arms no first-audio deadline; a reply still does (review r2 L4)', () => {
  const ctx = setup();
  ctx.m.onReady();
  ctx.m.onSay(7, 0, 'notice');                      // "Still there?" whose TTS failed: no audio, no say_end
  ctx.m.onSay(8, 0, 'ack');
  for (let i = 0; i < 40; i++) { ctx.clock.advance(250); ctx.m.onPong(80); ctx.m.tick(); }
  assert.deepEqual(ctx.links, []);
  ctx.m.onSay(9, 0, 'reply');
  for (let i = 0; i < 40; i++) { ctx.clock.advance(250); ctx.m.tick(); }
  assert.deepEqual(ctx.links.map((e) => [e.mode, e.why]), [['text', 'deadline']]);
});

test('the default clock is monotonic epoch-ms and is the same function as the downlink\'s (review r2 L3)', async () => {
  const { monoNow } = await import('../../src/dispatch_voice/static/link-monitor.js');
  const codec = await import('../../src/dispatch_voice/static/opus-codec.js');
  const a = monoNow();
  const b = codec.monoNow();
  assert.ok(Math.abs(a - Date.now()) < 5000, 'epoch-ms');
  assert.ok(b >= a && b - a < 50);
  assert.equal(monoNow.toString(), codec.monoNow.toString(), 'one clock definition in both modules');
});
