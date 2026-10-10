// The voice tools (owner: track T7): Voice cloning (the Voices manager on
// /api/voice/profiles), Built-in voices and the Voice lab with its v3 rows
// (engines, acknowledgements + Rebuild, learned pause + Reset, the last voice
// chat's speculation/link numbers, sessions and the power hold). Ported from
// the pre-split suite (DisPatch e8a5f75 frontend/tests/voice-tab.test.js) onto the add-on’s
// layout; the DisPatch wiring pins of that suite moved to DisPatch's own
// extension tests (track T1).

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { domSkip } from './_require-dom.js';
import { HAS_DOM, STATIC, fakeFetch, fakeShell, settle, withDom } from './_shell-fake.js';

import {
  cloningState, builtinState, mountVoiceTool, unmountVoiceTool, mountedVoiceTool, VOICE_TOOLS, VOICE_TOOL_TILES,
  INSTALL, fmtMs, builtinPlugin, ackRows, lastChatRows, STT_ENGINES, DECODER_PROMPT_S,
} from '../../src/dispatch_voice/static/voice-tools.js';
import { createVoicesPane, isStock, canBuild, chipClass, voiceMetaLine } from '../../src/dispatch_voice/static/voices.js';
import { noteMetrics } from '../../src/dispatch_voice/static/drive.js';
import { bindShell } from '../../src/dispatch_voice/static/shell.js';

const dom = { skip: domSkip(HAS_DOM ? false : 'jsdom is not installed (npm install)') };
const STOCK = 'stock-host';

async function inShell(fn, fetch) {
  return withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    const host = document.createElement('div');
    host.append(document.createElement('p'));
    host.firstChild.textContent = 'old';
    document.body.append(host);
    try { return await fn(host); } finally { unmountVoiceTool(); bindShell(prev); }
  }, { fetch });
}

// ------------------------------------------------------------------ pure ---

test('cloningState / builtinState read the status plugins block', () => {
  assert.equal(cloningState(null), 'missing');
  assert.equal(cloningState({ plugins: { cloning: { installed: true, ready: false } } }), 'models');
  assert.equal(cloningState({ plugins: { cloning: { installed: true, ready: true } } }), 'ready');
  assert.equal(builtinState(null), 'missing');
  assert.equal(builtinState({ plugins: { builtin: { installed: false, package: true } } }), 'not-enabled');
  assert.equal(builtinState({ plugins: { builtin: { installed: true, missing: ['model kokoro'] } } }), 'models');
  assert.equal(builtinState({ plugins: { builtin: { installed: true, missing: [] } } }), 'ready');
});

test('install steps are the add-on\'s own (moving v1 tag, fetch-models, doctor); the built-in plugin names itself', () => {
  const cmds = INSTALL.cloning.map((s) => s.cmd);
  assert.match(cmds[0], /dispatch-voice @ git\+https:\/\/github\.com\/LaserLloyd\/DisPatch_Voice@v1"$/);
  assert.ok(cmds.includes('.venv/bin/dispatch-voice fetch-models'));
  assert.ok(cmds.includes('.venv/bin/dispatch-voice doctor'));
  for (const c of cmds) assert.doesNotMatch(c, /--group voice|uvicorn|DISPATCH_VOICE_PKG/);
  assert.doesNotMatch(readFileSync(join(STATIC, 'voice-tools.js'), 'utf8').replace(/\/\/.*$/gm, ''), /kokoro/i);
  const st = { plugins: { builtin: { installed: true, name: 'acme-voices', licence: 'MIT', models: [{ where: 'plugin', cmd: 'acme fetch' }],
    suggested: { name: 'dispatch-voice-kokoro', licence: 'GPL-3.0-or-later', install: [], enable: [] } } } };
  assert.deepEqual([builtinPlugin(st).name, builtinPlugin(st).licence], ['acme-voices', 'MIT']);
});

test('fmtMs: SI units, no locale', () => {
  assert.equal(fmtMs(420.4), '420 ms');
  assert.equal(fmtMs(1340), '1.3 s');
  assert.equal(fmtMs(12500), '13 s');
  assert.equal(fmtMs(null), '–');
});

test('tool tiles for shell.tools.register (C-§E.1): ids, feature keys, icons, name/aria keys', () => {
  assert.deepEqual(VOICE_TOOLS, { 'voice-cloning': 'voice_cloning', 'voice-builtin': 'voice_builtin', 'voice-lab': 'voice_lab' });
  assert.deepEqual(VOICE_TOOL_TILES.map((x) => [x.id, x.feature, x.nameKey, x.ariaKey]), [
    ['voice-cloning', 'voice_cloning', 'vtools.cloning_name', 'vtools.cloning_aria'],
    ['voice-builtin', 'voice_builtin', 'vtools.builtin_name', 'vtools.builtin_aria'],
    ['voice-lab', 'voice_lab', 'vtools.lab_name', 'vtools.lab_aria'],
  ]);
  for (const tile of VOICE_TOOL_TILES) assert.ok(Array.isArray(tile.iconPaths) && tile.iconPaths.length, tile.id);
  assert.deepEqual(STT_ENGINES, ['parakeet', 'moonshine']);
  assert.ok(DECODER_PROMPT_S.includes(4), 'the default decoder prompt (4 s) is offered');
});

test('voices helpers: build needs a usable clip + a name + consent; chips; the stock voice', () => {
  assert.equal(canBuild({ clip: {}, analysis: { ok: true }, name: 'A', consent: 'self' }), true);
  assert.equal(canBuild({ clip: {}, analysis: { ok: true }, name: ' ', consent: 'self' }), false);
  assert.equal(canBuild({ clip: {}, analysis: { ok: false }, name: 'A', consent: 'self' }), false);
  assert.equal(chipClass('fail'), 'drive-chip fail');
  assert.equal(voiceMetaLine({ consent: { kind: 'self' }, reference_s: 14.6 }, () => ''), 'voices.kind_self, voices.clip_len');
  assert.ok(isStock({ id: STOCK }, STOCK) && !isStock({ id: 'alex' }, STOCK) && !isStock({ id: STOCK }, null));
});

// ------------------------------------------------------------------ DOM ---

test('Voice cloning tool: not installed → the add-on\'s commands; installed → the manager on /api/voice/profiles', dom, async () => {
  const status = { enabled: true, switch: true, stock_voice: STOCK, builtin_voices: [], plugins: { cloning: { installed: false, ready: false } } };
  const fetch = fakeFetch({
    'GET /api/voice/status': () => status,
    'GET /api/voice/profiles': { voices: [{ id: STOCK, name: 'Stock', consent: { kind: 'synthetic' } }, { id: 'alex', name: 'Alex', consent: { kind: 'self' } }], default: 'alex' },
  });
  await inShell(async (host) => {
    await mountVoiceTool(host, 'voice-cloning', {});
    assert.ok(host.querySelector('#vtool-install-group'));
    const code = [...host.querySelectorAll('.voice-cmd code')].map((c) => c.textContent).join('\n');
    assert.match(code, /DisPatch_Voice@v1"/);
    assert.match(code, /dispatch-voice fetch-models/);
    unmountVoiceTool();

    status.plugins.cloning = { installed: true, ready: true };
    await mountVoiceTool(host, 'voice-cloning', {});
    await settle(10);
    assert.equal(mountedVoiceTool(), 'voice-cloning');
    assert.ok(host.querySelector('.voices-pane'));
    assert.equal(host.querySelector('.voices-back'), null, 'embedded: the tool header is the way back');
    const rows = [...host.querySelectorAll('.voices-row')];
    assert.deepEqual(rows.map((r) => r.dataset.id), ['alex', STOCK], 'the stock voice last');
    assert.ok(rows[1].querySelector('.voices-stock'));
    assert.equal(rows[1].querySelector('.voices-delete'), null, 'no Delete on the stock voice');
    assert.ok(fetch.calls.some((c) => c.path === '/api/voice/profiles'), 'the moved profile store');
    assert.ok(!fetch.calls.some((c) => c.path.startsWith('/api/voices')), 'never the old /api/voices');
  }, fetch);
});

test('Built-in voices tool: plugin absent / installed-not-enabled / ready', dom, async () => {
  const SUG = { name: 'dispatch-voice-kokoro', licence: 'GPL-3.0-or-later',
    install: [{ where: 'plugin', cmd: 'uv run dispatch-voice-kokoro fetch-models' }],
    enable: [{ where: 'env', cmd: 'DISPATCH_VOICE_EXTENSIONS=dispatch_voice_kokoro:setup' }] };
  const status = { enabled: false, switch: false, default_voice: 'builtin:am_michael', plugins: { builtin: { installed: false, package: false, missing: [], suggested: SUG } } };
  const fetch = fakeFetch({ 'GET /api/voice/status': () => status });
  await inShell(async (host) => {
    await mountVoiceTool(host, 'voice-builtin', {});
    assert.match(host.textContent, /dispatch-voice-kokoro fetch-models/);
    status.plugins.builtin = { installed: false, package: true, missing: [], suggested: SUG };
    await mountVoiceTool(host, 'voice-builtin', {});
    assert.deepEqual([...host.querySelectorAll('.voice-cmd code')].map((c) => c.textContent), ['DISPATCH_VOICE_EXTENSIONS=dispatch_voice_kokoro:setup']);
    status.plugins.builtin = { installed: true, package: true, missing: [], suggested: SUG, name: 'dispatch-voice-kokoro', licence: 'GPL-3.0-or-later' };
    status.builtin_voices = [{ id: 'builtin:af_heart', name: 'Heart' }, { id: 'builtin:am_michael', name: 'Michael' }];
    await mountVoiceTool(host, 'voice-builtin', {});
    const rows = [...host.querySelectorAll('.vtool-voice-row')];
    assert.deepEqual(rows.map((r) => r.dataset.voice), ['builtin:af_heart', 'builtin:am_michael']);
    assert.ok(rows[1].querySelector('.voices-default'));
    assert.ok(rows.every((r) => r.querySelector('.drive-preview').disabled), 'no samples while voice is off');
  }, fetch);
});

const LAB = () => ({
  switch: true, forced: null, enabled: true, ready: true, missing: [], stt: 'parakeet', smart_turn: true,
  tts: 'auto', tts_resolved: 'chatterbox', fallback: null, loaded: true, warm: 'done', warm_seconds: 4,
  minimax: { selected: false, key_set: false }, hq: { configured: false },
  thinking: 'off', thinking_levels: ['off', 'low', 'medium'],
  timings: { hard_silence_s: { value: 1.4, default: 1.4, min: 0.8, max: 3 } },
});
const LAB_STATUS = () => ({
  enabled: true, switch: true, profiles: [], builtin_voices: [], stock_voice: STOCK,
  rt: { stt: 'parakeet', speculate: true, decoder_prompt_s: 4, decoder_sessions: 1, addendum_s: 3 },
  acks: { default: { ready: 50, total: 50 }, sage: { ready: 7, total: 19, name: 'Sage' } },
  sessions: { open: 1, max: 3 }, power_hold: 'held',
});

test('Voice lab: engines, acknowledgements + Rebuild, sessions + power, timing, show-timings, quick test', dom, async () => {
  const lab = LAB();
  const status = LAB_STATUS();
  const fetch = fakeFetch({
    'GET /api/voice/status': () => status,
    'GET /api/voice/lab': () => lab,
    'PATCH /api/voice/settings': (req) => req.body,
    'POST /api/voice/acks/rebuild': {},
  });
  await inShell(async (host) => {
    await mountVoiceTool(host, 'voice-lab', {});
    // Engines (server): STT engine + voice sample length, decoder sessions read-only.
    const stt = host.querySelector('#vtool-stt select');
    assert.deepEqual([...stt.options].map((o) => o.value), ['parakeet', 'moonshine']);
    const dp = host.querySelector('#vtool-decoder-prompt select');
    assert.equal(dp.value, '4');
    dp.value = '3'; dp.dispatchEvent(new Event('change'));
    await settle();
    assert.deepEqual(fetch.calls.find((c) => c.method === 'PATCH').body, { rt: { decoder_prompt_s: 3 } });
    assert.match(host.querySelector('#vtool-decoder-prompt').textContent, /Decoder sessions: 1/);
    assert.match(host.querySelector('#vtool-sessions').textContent, /1 of 3 open/);
    // Acknowledgements: one row per voice, Rebuild posts for that voice.
    const ackRowsEl = [...host.querySelectorAll('.vtool-acks-row')];
    assert.deepEqual(ackRowsEl.map((r) => r.dataset.voice), ['default', 'sage']);
    assert.match(ackRowsEl[1].textContent, /Sage: 7 of 19 ready/);
    ackRowsEl[1].querySelector('button').click();
    await settle();
    assert.ok(fetch.calls.some((c) => c.method === 'POST' && c.path === '/api/voice/acks/rebuild' && c.search === '?voice=sage'));
    // The rest of the lab survives: timing, show timings, quick test enabled.
    assert.ok(host.querySelector('#vtool-hard-input'));
    assert.ok(host.querySelector('#vtool-show-timings input'));
    assert.equal(host.querySelector('#vtool-test-btn').disabled, false);
    // Cloud and rig engines are status only (keys stay in the server's environment).
    assert.equal(host.querySelector('#vtool-hq input, #vtool-minimax input'), null);
  }, fetch);
});

test('Voice lab: the learned pause shows and Reset clears it on this device', dom, async () => {
  const fetch = fakeFetch({ 'GET /api/voice/status': LAB_STATUS(), 'GET /api/voice/lab': LAB() });
  await inShell(async (host) => {
    localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, learned: { probeMs: 250, holdMs: 700 } }));
    await mountVoiceTool(host, 'voice-lab', {});
    const r = host.querySelector('#vtool-learned');
    assert.match(r.textContent, /Waits 700 ms after a pause \(checks at 250 ms\)/);
    r.querySelector('button').click();
    await settle(10);
    assert.equal(JSON.parse(localStorage.getItem('dispatch-drive')).learned, null);
    assert.equal(JSON.parse(localStorage.getItem('dispatch-drive')).learnedReset, true, 'the server is told at the next start');
    assert.match(host.querySelector('#vtool-learned').textContent, /Nothing learned yet/);
    assert.equal(host.querySelector('#vtool-learned button').disabled, true);
  }, fetch);
});

test('Voice lab: the last voice chat\'s speculation and link numbers (from metrics), or "Nothing yet"', dom, () => withDom(async () => {
  const { shell } = fakeShell();
  const prev = bindShell(shell);
  try {
    const none = lastChatRows(null);
    assert.equal(none.length, 1);
    assert.equal(none[0].id, 'vtool-last-none');
    const m = noteMetrics({ type: 'metrics', speculation: { started: 2, adopted: 1, discarded: 1, stale: 0 },
      link: { rtt_ms: 420.2, down_kbps: 31, mode: 'phone', relay: true }, eot_to_first_audio_ms: 2400, platform_profile: 'performance', max_mhz: 5090 });
    const rows = lastChatRows(m);
    const by = Object.fromEntries(rows.map((r) => [r.id, r.textContent]));
    assert.match(by['vtool-spec'], /2 started, 1 used, 1 dropped, 0 too late/);
    assert.match(by['vtool-link'], /420 ms · 31 kbps · phone voice/);
    assert.match(by['vtool-link'], /Tailscale relay/);
    assert.match(by['vtool-last-timing'], /Reply started in 2\.4 s/);
    assert.match(by['vtool-power'], /performance, up to 5090 MHz/);
    const acks = ackRows({ default: { ready: 3, total: 50 } }, () => {});
    assert.match(acks[0].textContent, /Default voice: 3 of 50 ready/);
    assert.deepEqual(ackRows(null, () => {}), []);
  } finally { bindShell(prev); }
}));

test('an unknown tool id mounts nothing', dom, async () => {
  await inShell(async (host) => {
    await mountVoiceTool(host, 'voice-nope', {});
    assert.equal(host.textContent, 'old');
  }, fakeFetch({}));
});

test('createVoicesPane without embedded still takes over its pane with a back row', dom, async () => {
  const fetch = fakeFetch({ 'GET /api/voice/profiles': { voices: [] }, 'GET /api/voice/status': { builtin_voices: [] } });
  await inShell(async (host) => {
    const p = createVoicesPane(host, { toast() {}, onChanged() {} });
    p.open();
    await settle(10);
    assert.ok(host.querySelector('.voices-back'));
    assert.equal(host.querySelector('p').hidden, true);
    p.close();
    assert.equal(host.querySelector('p').hidden, false);
  }, fetch);
});

test('Voice lab: the keep-talking window is offered only without streaming recognition (FR-ux L5)', dom, async () => {
  const merge = { value: 1.8, default: 1.8, min: 0.5, max: 5 };
  for (const [stt, shown] of [['parakeet', false], [null, true]]) {
    const lab = { ...LAB(), stt, timings: { ...LAB().timings, merge_window_s: merge } };
    const status = { ...LAB_STATUS(), rt: { ...LAB_STATUS().rt, stt } };
    const fetch = fakeFetch({ 'GET /api/voice/status': () => status, 'GET /api/voice/lab': () => lab });
    await inShell(async (host) => {
      await mountVoiceTool(host, 'voice-lab', {});
      assert.equal(!!host.querySelector('#vtool-merge'), shown, `stt ${stt}`);
      assert.ok(host.querySelector('#vtool-hard-input'), 'end of turn stays');
    }, fetch);
  }
});
