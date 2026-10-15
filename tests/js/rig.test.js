// The GPU rig in the client (ADDON-ENGINE §5): Settings › Voice "Use the GPU rig
// when available" with its status line, and the Voice lab's rig line and fields.
// Every state renders; the token never reaches the page (the server never sends it).

import test from 'node:test';
import assert from 'node:assert/strict';
import { domSkip } from './_require-dom.js';
import { HAS_DOM, fakeShell, settle, withDom } from './_shell-fake.js';

import { rigStatusLine, rigLabLine, rigRow, rigClipRow, rigReason, mountDriveSettings } from '../../src/dispatch_voice/static/drive-settings.js';
import { rigLabRow, fmtMs } from '../../src/dispatch_voice/static/voice-tools.js';
import { bindShell } from '../../src/dispatch_voice/static/shell.js';

const dom = { skip: domSkip(HAS_DOM ? false : 'jsdom is not installed (npm install)') };

const READY = {
  configured: true, enabled: true, active: true, state: 'ready', host: '192.0.2.9:1236', token: 'file',
  gpu: { index: 3, name: 'RTX 3090', sm: 86 }, lease: 'held',
  engines: { stt: { name: 'parakeet', sha8: '1a2b3c4d' }, tts: { name: 'chatterbox-turbo', sha8: '5e6f7a8b' } },
  limits: { sessions: 3, contexts: 2, ahead_ms_max: 10000, max_text: 2000 },
  deadlines: { stt_final_ms: 300, tts_start_ms: 400, ahead_ms: 3000, first_chunk_tokens: 15 },
  metrics: { turns: 25, stt: 23, ttfa_p50: 140, partial_lag_p90: 190, share: 92, ping_rtt: 4.2, last_fallback_reason: null },
};

async function inShell(fn) {
  return withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    try { return await fn(); } finally { bindShell(prev); }
  });
}

test('rig: the status line and the lab line for every state', dom, () => inShell(async () => {
  assert.equal(rigStatusLine({ state: 'not_configured' }), 'Not set up on this computer');
  assert.equal(rigStatusLine({ state: 'off' }), null);
  assert.equal(rigStatusLine({ state: 'idle' }), 'Used from the next voice chat.');
  assert.equal(rigStatusLine(READY), 'Using the rig now (RTX 3090).');
  assert.match(rigStatusLine({ state: 'busy' }), /busy, so this computer is speaking/);
  assert.match(rigStatusLine({ state: 'down' }), /isn't answering/);
  assert.match(rigStatusLine({ state: 'cooling' }), /another try shortly/);
  assert.match(rigStatusLine({ state: 'misconfigured' }), /refused the token/);
  assert.equal(rigLabLine(READY, (ms) => `${ms} ms`),
    'GPU rig: ready on RTX 3090 (GPU 3). First audio p50 140 ms. Partial lag p90 190 ms. 92 % of turns on the rig. Last fallback: none.');
  assert.equal(rigLabLine({ state: 'busy', metrics: {} }), 'GPU rig: busy, using this computer.');
  assert.equal(rigLabLine({ state: 'misconfigured', metrics: READY.metrics }), 'GPU rig: misconfigured (token).');
  assert.equal(rigLabLine({ state: 'off', metrics: {} }), 'GPU rig: off.');
  assert.equal(rigLabLine(null), 'GPU rig: not set up');
  assert.match(rigLabLine({ state: 'idle', metrics: { turns: 3, last_fallback_reason: 'missed_pongs' } }), /Last fallback: the rig stopped answering\./);
  assert.doesNotMatch(rigLabLine(READY), /·/);
  // An unleased rig serves, and says so; a rig loading its models leaves the CPU speaking.
  const unleased = { ...READY, lease: 'unleased' };
  assert.equal(rigStatusLine(unleased), 'Using the rig now (RTX 3090). It runs without a GPU lease.');
  assert.match(rigLabLine(unleased, (ms) => `${ms} ms`), /^GPU rig: ready on RTX 3090 \(GPU 3\)\. Running without a GPU lease\. First audio/);
  assert.match(rigStatusLine({ state: 'loading' }), /loading its models, so this computer is speaking/);
  assert.equal(rigLabLine({ state: 'loading', metrics: {} }), 'GPU rig: loading its models, using this computer.');
  assert.equal(rigReason('tts_late'), 'the rig was too slow');
  assert.equal(rigReason('something_new'), 'a rig error');
}));

test('rig: Settings › Voice row is disabled with "Not set up" until the server has an address and a token', dom, () => inShell(async () => {
  const saves = [];
  const r = rigRow({ rig: { configured: false, enabled: false, state: 'not_configured' } }, async (p) => { saves.push(p); return true; });
  const cb = r.querySelector('#drive-rig-input');
  assert.equal(r.id, 'drive-rig');
  assert.match(r.textContent, /Use the GPU rig when available/);
  assert.match(r.textContent, /This computer takes over whenever it isn't/);
  assert.equal(cb.disabled, true);
  assert.equal(cb.checked, false);
  assert.equal(r.querySelector('#drive-rig-status').textContent, 'Not set up on this computer');
  assert.equal(rigRow({}, async () => true), null, 'an older server: no row');
}));

test('rig: switching it on saves rig.enabled and the status line follows', dom, () => inShell(async () => {
  const saves = [];
  let ok = true;
  const st = { rig: { configured: true, enabled: false, state: 'off' } };
  const r = rigRow(st, async (p) => { saves.push(p); return ok; });
  document.body.append(r);
  const cb = r.querySelector('#drive-rig-input');
  const line = r.querySelector('#drive-rig-status');
  assert.equal(cb.disabled, false);
  assert.equal(line.hidden, true);
  cb.checked = true;
  cb.dispatchEvent(new Event('change'));
  await settle();
  assert.deepEqual(saves, [{ rig: { enabled: true } }]);
  assert.equal(line.textContent, 'Used from the next voice chat.');
  ok = false;                                     // a refused save puts the switch back
  cb.checked = false;
  cb.dispatchEvent(new Event('change'));
  await settle();
  assert.equal(cb.checked, true);
}));

test('rig: the row sits in the Voices group of Settings › Voice', dom, () => inShell(async () => {
  const host = document.createElement('div');
  document.body.append(host);
  mountDriveSettings(host, {
    status: { enabled: true, switch: true, ready: true, missing: [], stt: 'parakeet', tts: 'auto', smart_turn: true,
      profiles: [], builtin_voices: [], rig: READY },
    bots: () => [], voices: [], android: false,
  });
  const voices = host.querySelector('#drive-group-voices');
  assert.ok(voices.querySelector('#drive-rig'));
  assert.equal(host.querySelector('#drive-rig-input').checked, true);
  assert.match(host.querySelector('#drive-rig-status').textContent, /Using the rig now/);
}));

test('rig: the Voice lab row shows the fields, the setup lines, and never a token', dom, () => inShell(async () => {
  const r = rigLabRow({ ...READY });
  assert.equal(r.id, 'vtool-rig');
  const text = r.textContent;
  assert.match(text, /GPU rig: ready on RTX 3090 \(GPU 3\)/);
  assert.match(text, /Address: 192\.0\.2\.9:1236/);
  assert.match(text, /stt parakeet 1a2b3c4d, tts chatterbox-turbo 5e6f7a8b/);
  assert.match(text, /Limits: 3 voice chats, 2 replies at once, 10 s ahead/);
  assert.match(text, /Deadlines: final words 300 ms, first audio 400 ms/);
  assert.doesNotMatch(text, /·/);
  assert.match(text, new RegExp(`Ping: ${fmtMs(4.2)}`));
  assert.match(text, /GPU lease: held/);
  assert.match(text, /rig\.url/);
  assert.match(text, /SPEECHFORGE_TOKEN=…/);
  const idle = rigLabRow({ state: 'idle', configured: true, enabled: true, metrics: {}, last: { gpu: { index: 3, name: 'RTX 3090' } } });
  assert.match(idle.textContent, /standing by for the next voice chat/);
  assert.match(rigLabRow(null).textContent, /GPU rig: not set up/);
  assert.match(rigLabRow({ ...READY, lease: 'unleased' }).textContent, /GPU lease: none, the rig runs without one/);
}));

const CLIP_OFF = "Replies stay on this computer: the rig's voice engine needs the recording and sending it is off";

test('rig: a voice the rig cannot make without the recording says why, in Settings and the Voice lab', dom, () => inShell(async () => {
  assert.equal(rigStatusLine({ ...READY, voice_note: 'clip_off' }), `Using the rig now (RTX 3090). ${CLIP_OFF}.`);
  assert.equal(rigStatusLine({ state: 'idle', voice_note: 'clip_off' }), `Used from the next voice chat. ${CLIP_OFF}.`);
  assert.equal(rigStatusLine({ ...READY, lease: 'unleased', voice_note: 'clip_off' }),
    `Using the rig now (RTX 3090). It runs without a GPU lease. ${CLIP_OFF}.`);
  assert.equal(rigStatusLine({ state: 'off', voice_note: 'clip_off' }), null);
  assert.equal(rigStatusLine({ ...READY, voice_note: 'something_new' }), 'Using the rig now (RTX 3090).');
  assert.ok(rigLabLine({ ...READY, voice_note: 'clip_off' }, (ms) => `${ms} ms`)
    .startsWith(`GPU rig: ready on RTX 3090 (GPU 3). ${CLIP_OFF}. First audio`));
  assert.match(rigStatusLine({ ...READY, voice_note: 'clip_refused' }), /the rig does not accept recordings/);
  assert.match(rigLabLine({ ...READY, voice_note: 'bad_clip' }), /the rig could not use the voice recording/);
}));

test('rig: "Send voice recordings to the rig" saves rig.send_reference_clip, off by default', dom, () => inShell(async () => {
  const saves = [];
  let ok = true;
  assert.equal(rigClipRow({ rig: { configured: true, enabled: true, state: 'idle' } }, async () => true), null,
    'an older server: no row');
  const off = rigClipRow({ rig: { configured: false, enabled: false, state: 'not_configured', send_reference_clip: false } },
    async () => true);
  assert.equal(off.querySelector('#drive-rig-clip-input').disabled, true);
  const st = { rig: { configured: true, enabled: true, state: 'idle', send_reference_clip: false } };
  const r = rigClipRow(st, async (p) => { saves.push(p); return ok; });
  document.body.append(r);
  assert.equal(r.id, 'drive-rig-clip');
  assert.match(r.textContent, /Send voice recordings to the rig/);
  assert.match(r.textContent, /reference recording/);
  assert.match(r.textContent, /The rig has its own setting/);
  const cb = r.querySelector('#drive-rig-clip-input');
  assert.equal(cb.checked, false);
  assert.equal(cb.disabled, false);
  cb.checked = true;
  cb.dispatchEvent(new Event('change'));
  await settle();
  assert.deepEqual(saves, [{ rig: { send_reference_clip: true } }]);
  assert.equal(st.rig.send_reference_clip, true);
  ok = false;                                     // a refused save puts the switch back
  cb.checked = false;
  cb.dispatchEvent(new Event('change'));
  await settle();
  assert.equal(cb.checked, true);
}));

test('rig: the recordings switch sits under the rig switch in Settings › Voice', dom, () => inShell(async () => {
  const host = document.createElement('div');
  document.body.append(host);
  mountDriveSettings(host, {
    status: { enabled: true, switch: true, ready: true, missing: [], stt: 'parakeet', tts: 'auto', smart_turn: true,
      profiles: [], builtin_voices: [], rig: { ...READY, send_reference_clip: false } },
    bots: () => [], voices: [], android: false,
  });
  const rows = [...host.querySelectorAll('#drive-group-voices .drive-row')].map((r) => r.id);
  assert.equal(rows.indexOf('drive-rig-clip'), rows.indexOf('drive-rig') + 1);
  assert.equal(host.querySelector('#drive-rig-clip-input').checked, false);
}));
