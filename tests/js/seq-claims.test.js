// T8: seq-claims.js — (sid, seq) keyed claims, reset on every `ready` (contracts C-§E.2, C.5 #1, #9).
import test from 'node:test';
import assert from 'node:assert/strict';
import { createSeqClaims, CLAIM_PATHS } from '../../src/dispatch_voice/static/seq-claims.js';

test('first path to claim a sentence wins; the same path keeps re-claiming it', () => {
  const c = createSeqClaims();
  assert.equal(c.claim('aa11', 1, 'server'), true);   // first audio message of seq 1
  assert.equal(c.claim('aa11', 1, 'server'), true);   // its later chunks
  assert.equal(c.claim('aa11', 1, 'phone'), false);   // the phone must not speak it too
  assert.equal(c.claim('aa11', 1, 'text'), false);
  assert.equal(c.owner('aa11', 1), 'server');
  assert.equal(c.claim('aa11', 2, 'phone'), true);
  assert.equal(c.claim('aa11', 2, 'server'), false);  // late audio for a sentence the phone spoke
  assert.equal(c.owner('aa11', 3), null);
});

test('claims are keyed (sid, seq): a new session\'s seq 1 is not the old session\'s', () => {
  const c = createSeqClaims();
  assert.equal(c.claim('old-sid', 1, 'server'), true);
  assert.equal(c.claim('old-sid', 2, 'server'), true);
  assert.equal(c.claim('old-sid', 3, 'server'), true);
  // Reconnect after seq 3: the server's new VoiceSession starts at seq 1 with a new sid.
  // Even without the reset the key differs, so the new first sentence plays (hardening M2).
  assert.equal(c.claim('new-sid', 1, 'phone'), true);
  assert.equal(c.owner('old-sid', 1), 'server');
});

test('reset() on ready drops every claim (simulated reconnect)', () => {
  const c = createSeqClaims();
  const ready1 = 'aaaaaaaaaaaaaaaa';
  for (let s = 1; s <= 3; s++) assert.equal(c.claim(ready1, s, 'server'), true);
  assert.equal(c.size, 3);
  c.reset(); // `ready` from the new socket
  assert.equal(c.size, 0);
  const ready2 = 'bbbbbbbbbbbbbbbb';
  assert.equal(c.claim(ready2, 1, 'text'), true);
  assert.equal(c.claim(ready2, 1, 'server'), false); // the audio of a sentence shown as text is dropped
  // Even the same sid (a server that reused it) starts clean after reset.
  assert.equal(c.claim(ready1, 1, 'phone'), true);
});

test('seq is compared as a u32 and an unknown path is a programming error', () => {
  const c = createSeqClaims();
  assert.equal(c.claim('s', 2 ** 32 + 5, 'server'), true);
  assert.equal(c.claim('s', 5, 'phone'), false);
  assert.throws(() => c.claim('s', 1, 'speaker'), TypeError);
  assert.deepEqual([...CLAIM_PATHS], ['server', 'phone', 'text']);
});

test('memory is bounded: the oldest claims are forgotten first', () => {
  const c = createSeqClaims({ limit: 4 });
  for (let s = 1; s <= 6; s++) c.claim('s', s, 'server');
  assert.equal(c.size, 4);
  assert.equal(c.owner('s', 1), null);
  assert.equal(c.owner('s', 2), null);
  assert.equal(c.owner('s', 6), 'server');
});
