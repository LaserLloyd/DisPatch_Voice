// Voice chat settings (owner: track T7): the device blob and its migration
// (C-§F.3), the `start` frame (C-§C.3), Settings → Voice (SPEC §7.1: 19 rows
// in 6 groups + a collapsed Status), the quick sheet (§7.3), the Bots-tab
// popover (§7.4) and the add-on's locale bundles (C-§E.3, the rename).

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { domSkip } from './_require-dom.js';
import { HAS_DOM, LOCALES, STATIC, fakeFetch, fakeShell, loadBundle, settle, withDom } from './_shell-fake.js';

import {
  DRIVE_KEY, LEGACY_PATIENCE_KEY, DRIVE_DEFAULTS, BLOB_VERSION, loadDriveSettings, saveDriveSettings,
  normalizeDriveSettings, onDriveSettings, startPayload, cursorFrom, clampSpeed, parseVoice, voiceName,
  voiceOptions, engineLine, voiceTabState, setupSteps, phoneVoiceRowMode, hasPhoneVoice, rpOptions,
  narratorOptions, sessionLines, mountDriveSettings, buildQuickSheet, botVoicePill, setBotVoice,
  RP_MODES, VOICE_REPO_URL, mergeInto, saveErrorText, placePopover, followUpValue, POPOVER_GUTTER,
} from '../../src/dispatch_voice/static/drive-settings.js';
import { bindShell } from '../../src/dispatch_voice/static/shell.js';

const dom = { skip: domSkip(HAS_DOM ? false : 'jsdom is not installed (npm install)') };

function memStorage(init = {}) {
  const m = new Map(Object.entries(init));
  return {
    map: m,
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => m.set(k, String(v)),
    removeItem: (k) => m.delete(k),
  };
}

// ------------------------------------------------------------ the blob ---

test('blob: defaults on a fresh device carry the v3 keys (F.3)', () => {
  const s = loadDriveSettings(memStorage());
  for (const [k, v] of Object.entries(DRIVE_DEFAULTS)) assert.deepEqual(s[k], v, k);
  assert.equal(s.reactions, 'sometimes');
  assert.equal(s.slowLink, 'phone');
  assert.equal(s.hangupQuietMin, 15);
  assert.equal(s.firstRunOpens, 0);
  assert.equal(s.followUp, null, "null = the bot's own default");
  assert.equal(s.v, BLOB_VERSION);
});

test('blob: an r1/2.1 blob migrates — acks → reactions, backchannels → often, partials folded, followUp:true → bot default', () => {
  const st = memStorage({ [DRIVE_KEY]: JSON.stringify({ acks: 'often', patience: 'patient', partials: false, followUp: true, captions: true }) });
  const s = loadDriveSettings(st);
  assert.equal(s.reactions, 'often');
  assert.equal(s.patience, 'patient');
  assert.equal(s.followUp, null);
  assert.equal('partials' in s, false);
  assert.equal('acks' in s, false);
  // Rewritten once in the v3 layout, so a later explicit true survives.
  const saved = JSON.parse(st.getItem(DRIVE_KEY));
  assert.equal(saved.v, BLOB_VERSION);
  assert.equal(saveDriveSettings(st, { followUp: true }).followUp, true);
  assert.equal(loadDriveSettings(st).followUp, true);

  assert.equal(normalizeDriveSettings({ acks: 'sometimes', backchannels: true }).reactions, 'often');
  assert.equal(normalizeDriveSettings({ acks: 'off', backchannels: true }).reactions, 'off');
  assert.equal(normalizeDriveSettings({ followUp: false }).followUp, false, 'an old "off" is a real choice');
});

test('blob: the 2.1.0 patience key migrates into the blob once and is removed', () => {
  const st = memStorage({ [LEGACY_PATIENCE_KEY]: 'quick' });
  assert.equal(loadDriveSettings(st).patience, 'quick');
  assert.equal(st.getItem(LEGACY_PATIENCE_KEY), null);
  assert.equal(JSON.parse(st.getItem(DRIVE_KEY)).patience, 'quick');
});

test('blob: junk falls back per key; a throwing storage still loads', () => {
  const s = normalizeDriveSettings({
    patience: 'forever', reactions: 'always', slowLink: 'pigeon', hangupQuietMin: 7, firstRunOpens: -3,
    learned: { probeMs: 'x' }, phoneVoice: 'v'.repeat(300), bargeIn: 'yes',
  });
  assert.equal(s.patience, 'normal');
  assert.equal(s.reactions, 'sometimes');
  assert.equal(s.slowLink, 'phone');
  assert.equal(s.hangupQuietMin, 15);
  assert.equal(s.firstRunOpens, 0);
  assert.equal(s.learned, null);
  assert.equal(s.phoneVoice, '');
  assert.equal(s.bargeIn, true);
  assert.deepEqual(normalizeDriveSettings({ learned: { probeMs: 250.4, holdMs: 900 } }).learned, { probeMs: 250, holdMs: 900 });
  const bad = { getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); } };
  assert.equal(loadDriveSettings(bad).patience, 'normal');
});

test('blob: a change notifies listeners (the open voice chat re-sends start)', () => {
  const st = memStorage();
  const seen = [];
  const off = onDriveSettings((s) => seen.push(s.reactions));
  saveDriveSettings(st, { reactions: 'often' });
  off();
  saveDriveSettings(st, { reactions: 'off' });
  assert.deepEqual(seen, ['often']);
});

// ------------------------------------------------------------ start ---

test('start: Reactions maps to start.acks and rt.backchannels (only "often")', () => {
  for (const [reactions, bc] of [['off', false], ['sometimes', false], ['often', true]]) {
    const f = startPayload({ reactions });
    assert.equal(f.acks, reactions);
    assert.equal(f.rt.backchannels, bc, reactions);
  }
});

test('start: carries exactly the C.3 fields; follow_up only when this device chose', () => {
  const f = startPayload({ ...DRIVE_DEFAULTS, captions: false, slowLink: 'text' }, 'm-2', ['m-2', 'm-1']);
  assert.deepEqual(Object.keys(f).sort(),
    ['acks', 'after', 'after_ids', 'barge_in', 'patience', 'reply_style', 'rt', 'slow_link', 'type'].sort());
  assert.equal(f.type, 'start');
  assert.equal(f.slow_link, 'text');
  assert.deepEqual(f.rt, { partials: false, backchannels: false, optimistic_duck: true });
  assert.equal(startPayload({ followUp: false }).follow_up, false);
  assert.equal(startPayload({ followUp: true }).follow_up, true);
  assert.equal('follow_up' in startPayload({}), false, "absent = the bot's default applies");
  const g = startPayload({}, null, null, { cursor: { stream: 's-1', from: 3200 }, after_partial: { message_id: 'm', sentences: 2 }, junk: 1 });
  assert.deepEqual(g.cursor, { stream: 's-1', from: 3200 });
  assert.deepEqual(g.after_partial, { message_id: 'm', sentences: 2 });
  assert.equal('junk' in g, false);
  assert.equal(startPayload({}, null, Array.from({ length: 12 }, (_, i) => `m${i}`)).after_ids.length, 8);
});

test('start: the learned pause seeds the server; the lab\'s Reset is a one-shot learned_reset (T7a-6, T3s3-1)', () => {
  assert.equal('learned' in startPayload({}).rt, false, 'nothing learned: nothing sent');
  assert.deepEqual(startPayload({ learned: { probeMs: 300, holdMs: 750 } }).rt.learned, { probe_ms: 300, hold_ms: 750 });
  const r = startPayload({ learned: { probeMs: 300, holdMs: 750 }, learnedReset: true }).rt;
  assert.equal(r.learned_reset, true);
  assert.equal('learned' in r, false, 'a reset never re-seeds the value it forgets');
  assert.equal(normalizeDriveSettings({ learnedReset: true }).learnedReset, true);
  assert.equal(normalizeDriveSettings({ learnedReset: 'yes' }).learnedReset, false);
});

test('cursorFrom = max(lastRxOff, lastTranscriptUpto, lastEotUpto) (C.5 #10)', () => {
  assert.equal(cursorFrom({ lastRxOff: 16000, lastTranscriptUpto: 24000, lastEotUpto: 20000 }), 24000);
  assert.equal(cursorFrom({ lastRxOff: 32000 }), 32000);
  assert.equal(cursorFrom({ lastRxOff: NaN }), 0);
  assert.equal(cursorFrom(), 0);
  // The offsets are a u32 sample clock: 100 samples after the wrap is later than 2³² − 296.
  assert.equal(cursorFrom({ lastRxOff: 4294967000, lastEotUpto: 100 }), 100);
});

// ------------------------------------------------------------ helpers ---

test('speed clamps to 0.8–1.3 on a 0.05 grid; voice values keep the saved one visible', () => {
  assert.equal(clampSpeed(2), 1.3);
  assert.equal(clampSpeed(0.1), 0.8);
  assert.equal(clampSpeed('1.07'), 1.05);
  assert.equal(clampSpeed('x'), 1);
  assert.deepEqual(parseVoice('builtin:af'), { kind: 'builtin', id: 'af' });
  const st = { profiles: [{ id: 'p1', name: 'Robin' }, { id: 'stock', name: 'x' }], stock_voice: 'stock', builtin_voices: [{ id: 'af', name: 'Heart' }], default_voice: 'p1' };
  const opts = voiceOptions(st, { withDefault: true, current: 'gone' });
  assert.equal(opts[0].value, '');
  assert.ok(opts.some((o) => o.value === 'gone' && o.group === 'missing'));
  assert.equal(voiceName(st, 'stock'), 'drive.voice_stock');
  assert.ok(narratorOptions(st, 'gone').find((o) => o.value === 'gone').label.endsWith('drive.narrator_missing'));
  assert.equal(narratorOptions(st)[0].value, '');
  assert.deepEqual(rpOptions().map((o) => o.value), RP_MODES);
  assert.deepEqual(rpOptions({ withServerDefault: true }).map((o) => o.value), ['', ...RP_MODES]);
});

test('voiceTabState: none / forced / off / setup / on', () => {
  assert.equal(voiceTabState(null), 'none');
  assert.equal(voiceTabState({ forced: 'off' }), 'forced');
  assert.equal(voiceTabState({ switch: false, enabled: false }), 'off');
  assert.equal(voiceTabState({ switch: true, enabled: false, missing: ['model x'] }), 'setup');
  assert.equal(voiceTabState({ switch: true, enabled: true, missing: [] }), 'on');
});

test('setupSteps: the add-on\'s own commands (reinstall at the moving v1 tag, fetch-models, doctor)', () => {
  const cmds = (m) => setupSteps(m).map((s) => s.cmd);
  assert.deepEqual(cmds([]), []);
  assert.deepEqual(cmds(['model parakeet-tdt-0.6b-v2']), ['.venv/bin/dispatch-voice fetch-models', '.venv/bin/dispatch-voice doctor']);
  const pkg = cmds(['python package onnx-asr']);
  assert.equal(pkg[0], `uv pip install --python .venv/bin/python "dispatch-voice @ git+${VOICE_REPO_URL}@v1"`);
  assert.equal(VOICE_REPO_URL, 'https://github.com/LaserLloyd/DisPatch_Voice');
  assert.ok(cmds(['MINIMAX_API_KEY missing']).includes('MINIMAX_API_KEY=…'));
  for (const s of setupSteps(['python package x', 'model y'])) assert.doesNotMatch(s.cmd, /--group voice|app\.voice/);
});

test('phone voice row: hidden unless the slow-link choice is a phone one; Android gets the hint, desktop a picker', () => {
  const desktop = [{ name: 'Alex', lang: 'en-US', localService: true, voiceURI: 'a' }, { name: 'Cloud', lang: 'en-US', localService: false, voiceURI: 'c' }];
  const android = [{ name: 'English', lang: 'en_US', localService: true, voiceURI: 'x' }];
  assert.equal(phoneVoiceRowMode('wait', { voices: desktop }), 'hidden');
  assert.equal(phoneVoiceRowMode('text', { voices: desktop }), 'hidden');
  assert.equal(phoneVoiceRowMode('phone', { voices: desktop, lang: 'en' }), 'picker');
  assert.equal(phoneVoiceRowMode('always_phone', { voices: android, lang: 'en', android: true }), 'android');
  assert.equal(phoneVoiceRowMode('phone', { voices: [], lang: 'en' }), 'none');
  assert.equal(phoneVoiceRowMode('phone', { voices: [desktop[1]], lang: 'en' }), 'none', 'desktop: only offline voices count');
  assert.equal(hasPhoneVoice(android, 'en', { android: true }), true);
  assert.equal(hasPhoneVoice(desktop, 'de'), false);
});

test('Status lines: sessions open and the power hold, in words', () => {
  assert.deepEqual(sessionLines({ sessions: { open: 1, max: 3 }, power_hold: 'held' }),
    ['drive.sessions_open', 'drive.power_held']);
  assert.deepEqual(sessionLines({}), []);
  assert.match(engineLine({ tts: 'auto', stock_voice: 's', smart_turn: true, rt: { stt: 'parakeet' } }), /drive\.engine_stt/);
});

// ---------------------------------------------------- the rename / locales ---

const PLURAL = new Set(['zero', 'one', 'two', 'few', 'many', 'other']);
function flat(o, p = '') {
  return Object.entries(o).flatMap(([k, v]) => {
    if (k === '$meta') return [];
    const isPlural = v && typeof v === 'object' && Object.keys(v).every((c) => PLURAL.has(c));
    return v && typeof v === 'object' && !isPlural ? flat(v, `${p}${k}.`) : [[`${p}${k}`, v]];
  });
}

test('rename: no en value says "Drive mode" or uses "drive" as a noun; namespaces are drive/voices/vtools only', () => {
  const en = loadBundle('en');
  assert.deepEqual(Object.keys(en).filter((k) => k !== '$meta').sort(), ['drive', 'voices', 'vtools']);
  for (const [k, v] of flat(en)) {
    assert.doesNotMatch(String(v), /Drive mode/i, k);
    assert.doesNotMatch(String(v), /\bdrive\b/i, k);
  }
  assert.equal(en.drive.title, 'Voice chat');
  assert.equal(en.drive.end, 'Hang up');
  assert.equal(en.drive.ended, 'Voice chat ended.');
  assert.equal(en.drive.button, 'Voice chat');
  assert.equal(en.drive.button_aria, 'Voice chat (hands-free)');
});

test('rename: no locale keeps the old localised mode names (C-§E.3)', () => {
  const OLD = {
    de: /Fahrmodus|Fahrt|Fahrbildschirm|Fahreinstellung/, es: /conducción|trayecto/i, fr: /conduite|trajet/i,
    pt: /condução|viagem/i, ja: /ドライブ/, zh: /驾驶/, ar: /القيادة/,
  };
  const NEW = { de: 'Sprachchat', es: 'Chat de voz', fr: 'Chat vocal', pt: 'Chat de voz', ja: '音声チャット', zh: '语音聊天', ar: 'الدردشة الصوتية' };
  for (const [lang, re] of Object.entries(OLD)) {
    const b = loadBundle(lang);
    for (const [k, v] of flat(b)) assert.doesNotMatch(String(v), re, `${lang}:${k}`);
    assert.equal(b.drive.title, NEW[lang], lang);
  }
});

test('every literal key the client uses exists in all 8 bundles (dynamic families too)', () => {
  const langs = readdirSync(LOCALES).filter((n) => n.endsWith('.json')).map((n) => n.slice(0, -5));
  assert.equal(langs.length, 8);
  const used = new Set();
  for (const f of readdirSync(STATIC).filter((n) => n.endsWith('.js'))) {
    const src = readFileSync(join(STATIC, f), 'utf8');
    for (const m of src.matchAll(/'((?:drive|vtools|voices)\.[a-z0-9_]+)'/g)) used.add(m[1]);
  }
  for (const s of ['connecting', 'listening', 'hearing', 'transcribing', 'thinking', 'speaking', 'paused', 'offline', 'waiting', 'stopped', 'locked']) used.add(`drive.state_${s}`);
  for (const s of ['off', 'low', 'medium']) used.add(`drive.thinking_${s}`);
  for (const s of ['quick', 'normal', 'patient']) used.add(`drive.patience_${s}`);
  for (const s of ['short', 'normal', 'longer']) used.add(`drive.reply_${s}`);
  for (const s of ['off', 'sometimes', 'often']) used.add(`drive.acks_${s}`);
  for (const s of RP_MODES) used.add(`drive.rp_${s}`);
  for (const s of ['parakeet', 'moonshine']) used.add(`vtools.stt_${s}`);
  for (const lang of langs) {
    const keys = new Set(flat(loadBundle(lang)).map(([k]) => k));
    const missing = [...used].filter((k) => !keys.has(k));
    assert.deepEqual(missing, [], `${lang} is missing keys`);
  }
});

// -------------------------------------------------------- Settings → Voice ---

const ON_STATUS = () => ({
  enabled: true, switch: true, forced: null, missing: [],
  default_voice: 'p1', speed: 1, stock_voice: 'stock', tts: 'auto', smart_turn: true,
  profiles: [{ id: 'p1', name: 'Robin' }, { id: 'stock', name: 'Stock' }],
  builtin_voices: [{ id: 'af_heart', name: 'Heart' }],
  plugins: { cloning: { installed: true, ready: true } },
  rp_speech: 'dialogue_narrator', narrator_voice: '', expressive_tags: true, voice_turn_thinking: 'off',
  sessions: { open: 1, max: 3 }, power_hold: 'held', rt: { stt: 'parakeet' },
});

function rowsOf(host, groupSel = 'section.dev-group') {
  return [...host.querySelectorAll(groupSel)].map((g) => ({
    id: g.id, rows: [...g.querySelectorAll(':scope > .dev-group-body > .dev-row')].filter((r) => !r.hidden).map((r) => r.id),
  }));
}

test('Settings → Voice: 20 rows in 6 groups + a collapsed Status (SPEC §7.1, 1.2.0: + Type instead of talking)', dom, () => withDom(async () => {
  const { shell } = fakeShell();
  const prev = bindShell(shell);
  try {
    localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, slowLink: 'phone' }));
    const host = document.createElement('div');
    document.body.append(host);
    const voices = [{ name: 'Alex', lang: 'en-US', localService: true, voiceURI: 'alex' }];
    mountDriveSettings(host, { status: ON_STATUS(), cloning: true, bots: () => [], voices, android: false, setEnabled: async () => true });
    const groups = rowsOf(host);
    assert.equal(groups.length, 6, groups.map((g) => g.id).join());
    assert.deepEqual(groups.map((g) => g.id), ['drive-group-switch', 'drive-group-voices', 'drive-group-listening',
      'drive-group-replies', 'drive-group-connection', 'drive-group-device']);
    const total = groups.reduce((n, g) => n + g.rows.length, 0);
    assert.equal(total, 20, JSON.stringify(groups));
    assert.deepEqual(groups[1].rows, ['drive-default-voice', 'drive-speed', 'drive-rp', 'drive-narrator', 'drive-manage-voices']);
    assert.deepEqual(groups[3].rows, ['drive-reply-length', 'drive-follow-up', 'drive-reactions', 'drive-tones']);
    assert.deepEqual(groups[4].rows, ['drive-slow', 'drive-phone-voice']);
    assert.deepEqual(groups[5].rows, ['drive-screen-on', 'drive-captions', 'drive-hangup-quiet']);
    const status = host.querySelector('details#drive-group-status');
    assert.ok(status, 'Status is a <details>');
    assert.equal(status.open, false, 'collapsed');
    const statusRows = [...status.querySelectorAll('.dev-row')].map((r) => r.id);
    for (const id of ['drive-engine', 'drive-thinking', 'drive-show-timings', 'drive-expressive', 'drive-sessions']) assert.ok(statusRows.includes(id), id);
    assert.match(host.querySelector('#drive-sessions').textContent, /1 of 3 open/);
    // The switch row says the engines are English-only (H3).
    assert.match(host.querySelector('#drive-switch').textContent, /English only/);
    // No literal "Drive" anywhere on the page.
    assert.doesNotMatch(host.textContent, /\bDrive\b/);
  } finally { bindShell(prev); }
}));

test('Settings → Voice: device rows save to the blob and change what start sends', dom, () => withDom(async () => {
  const { shell } = fakeShell();
  const prev = bindShell(shell);
  try {
    const host = document.createElement('div');
    document.body.append(host);
    mountDriveSettings(host, { status: ON_STATUS(), bots: () => [], voices: [], android: false });
    host.querySelector('#drive-reactions .drive-seg-btn[data-value="often"]').click();
    let s = JSON.parse(localStorage.getItem('dispatch-drive'));
    assert.equal(s.reactions, 'often');
    let f = startPayload(s);
    assert.equal(f.acks, 'often');
    assert.equal(f.rt.backchannels, true);
    host.querySelector('#drive-reactions .drive-seg-btn[data-value="off"]').click();
    f = startPayload(JSON.parse(localStorage.getItem('dispatch-drive')));
    assert.equal(f.acks, 'off');
    assert.equal(f.rt.backchannels, false);
    host.querySelector('#drive-hangup-quiet .drive-seg-btn[data-value="30"]').click();
    assert.equal(JSON.parse(localStorage.getItem('dispatch-drive')).hangupQuietMin, 30);
    host.querySelector('#drive-hangup-quiet .drive-seg-btn[data-value="0"]').click();
    assert.equal(JSON.parse(localStorage.getItem('dispatch-drive')).hangupQuietMin, 0);
    const cap = host.querySelector('#drive-captions input');
    cap.checked = false; cap.dispatchEvent(new Event('change'));
    assert.equal(startPayload(JSON.parse(localStorage.getItem('dispatch-drive'))).rt.partials, false, 'captions carry the partials');
  } finally { bindShell(prev); }
}));

test('Settings → Voice: the Phone voice row follows the slow-link choice; Android shows the hint, not a picker', dom, () => withDom(async () => {
  const { shell } = fakeShell();
  const prev = bindShell(shell);
  try {
    const host = document.createElement('div');
    document.body.append(host);
    const desk = [{ name: 'Alex', lang: 'en-US', localService: true, voiceURI: 'alex' }];
    mountDriveSettings(host, { status: ON_STATUS(), bots: () => [], voices: desk, android: false });
    assert.ok(host.querySelector('#drive-phone-voice select'), 'desktop: a picker');
    assert.equal(host.querySelector('#drive-phone-voice select option[value="alex"]').textContent,
      'Alex — English (United States)', 'no nested parentheses (review-T7b L2)');
    host.querySelector('#drive-slow .drive-seg-btn[data-value="wait"]').click();
    assert.equal(host.querySelector('#drive-phone-voice'), null, 'Wait: no phone voice row');
    host.querySelector('#drive-slow .drive-seg-btn[data-value="always_phone"]').click();
    assert.ok(host.querySelector('#drive-phone-voice'));
    assert.match(host.querySelector('#drive-slow').textContent, /Fastest replies/);
    const sel = host.querySelector('#drive-phone-voice select');
    sel.value = 'alex'; sel.dispatchEvent(new Event('change'));
    assert.equal(JSON.parse(localStorage.getItem('dispatch-drive')).phoneVoice, 'alex');

    const host2 = document.createElement('div');
    document.body.append(host2);
    localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, slowLink: 'phone' }));
    mountDriveSettings(host2, { status: ON_STATUS(), bots: () => [], voices: [{ name: 'English', lang: 'en_US', localService: true, voiceURI: 'x' }], android: true });
    const pr = host2.querySelector('#drive-phone-voice');
    assert.equal(pr.querySelector('select'), null, 'Android: no picker');
    assert.match(pr.textContent, /Text-to-speech output/);

    const host3 = document.createElement('div');
    document.body.append(host3);
    mountDriveSettings(host3, { status: ON_STATUS(), bots: () => [], voices: [], android: false });
    assert.match(host3.querySelector('#drive-phone-voice').textContent, /no offline voice/);
  } finally { bindShell(prev); }
}));

test('Settings → Voice: the narrator row shows only for "dialogue + narrator"; role-play saves to the server', dom, async () => {
  const fetch = fakeFetch({ 'PATCH /api/voice/settings': (req) => req.body });
  await withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    try {
      const host = document.createElement('div');
      document.body.append(host);
      mountDriveSettings(host, { status: { ...ON_STATUS(), rp_speech: 'all' }, bots: () => [], voices: [] });
      assert.equal(host.querySelector('#drive-narrator').hidden, true);
      const rp = host.querySelector('#drive-rp select');
      rp.value = 'dialogue_narrator'; rp.dispatchEvent(new Event('change'));
      await settle();
      assert.deepEqual(fetch.calls.at(-1), { method: 'PATCH', path: '/api/voice/settings', search: '', body: { rp_speech: 'dialogue_narrator' } });
      assert.equal(host.querySelector('#drive-narrator').hidden, false);
    } finally { bindShell(prev); }
  }, { fetch });
});

test('Settings → Voice: off shows the switch alone; setup lists what is missing with the commands; failure offers Try again', dom, () => withDom(async () => {
  const { shell } = fakeShell();
  const prev = bindShell(shell);
  try {
    const host = document.createElement('div');
    document.body.append(host);
    mountDriveSettings(host, { status: { enabled: false, switch: false, missing: [] } });
    assert.equal(host.querySelectorAll('section.dev-group').length, 1);
    assert.ok(host.querySelector('#drive-switch-input'));
    mountDriveSettings(host, { status: { enabled: false, switch: true, missing: ['model parakeet-tdt-0.6b-v2'] } });
    assert.match(host.querySelector('#drive-setup').textContent, /fetch-models/);
    mountDriveSettings(host, { status: { forced: 'off' } });
    assert.equal(host.querySelector('#drive-switch-input'), null);
    let retried = 0;
    mountDriveSettings(host, { status: null, retry: () => { retried += 1; } });
    host.querySelector('#drive-status-retry').click();
    assert.equal(retried, 1);
    mountDriveSettings(host, { status: null, loading: true });
    assert.ok(host.querySelector('.vtool-loading'));
  } finally { bindShell(prev); }
}));

// ----------------------------------------------------------- quick sheet ---

test('quick sheet: "{bot}\'s voice" PUTs this bot in the full roster (order kept), never the server default', dom, async () => {
  const roster = [{ id: 'a', name: 'A', voice: '' }, { id: 'nova', name: 'Nova', voice: '' }, { id: 'c', name: 'C', voice: 'p1' }];
  const fetch = fakeFetch({
    'GET /api/bots/all': { bots: roster },
    'PUT /api/bots/order': (req) => ({ bots: roster.map((b) => ({ ...b, voice: b.id === 'nova' ? 'p1' : b.voice })) }),
  });
  await withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    try {
      const bot = { id: 'nova', name: 'Nova', voice: '' };
      const sheet = buildQuickSheet({ status: ON_STATUS(), bot, voices: [], android: false, lang: 'en', notify: () => {} });
      document.body.append(sheet);
      assert.match(sheet.textContent, /Nova’s voice/);
      const sel = sheet.querySelector('#drive-sheet-voice');
      sel.value = 'p1'; sel.dispatchEvent(new Event('change'));
      await settle(10);
      const put = fetch.calls.find((c) => c.method === 'PUT');
      assert.ok(put, 'PUT /api/bots/order');
      assert.deepEqual(put.body, { bots: [{ id: 'a', order: 0 }, { id: 'nova', order: 1, voice: 'p1' }, { id: 'c', order: 2 }] });
      assert.equal(fetch.calls.some((c) => c.path === '/api/voice/settings'), false, 'the server default is untouched');
      assert.equal(bot.voice, 'p1');
      // Six road rows (Type instead of talking since 1.2.0) + the voice row + All settings.
      assert.equal(sheet.querySelectorAll('.dev-group-body > .dev-row').length, 8);
      assert.ok(sheet.querySelector('#drive-sheet-typed input.dev-switch'));
      assert.ok(sheet.querySelector('#drive-sheet-all'));
    } finally { bindShell(prev); }
  }, { fetch });
});

test('quick sheet on a locked device (1.3.3): device settings only — no bot voice, no All settings', dom, () => withDom(async () => {
  const { shell } = fakeShell({ locked: true });
  const prev = bindShell(shell);
  try {
    const bot = { id: 'alpha', name: 'Alpha', voice: '', safe: true };
    const sheet = buildQuickSheet({ status: ON_STATUS(), bot, voices: [], android: false, lang: 'en',
      locked: true, onAllSettings: () => {} });
    assert.equal(sheet.querySelector('#drive-sheet-voice'), null, 'the bot voice writes server state');
    assert.equal(sheet.querySelector('#drive-sheet-all'), null, 'Settings are unlocked-only');
    assert.ok(sheet.querySelector('#drive-sheet-typed input.dev-switch'));
    assert.equal(sheet.querySelectorAll('.dev-group-body > .dev-row').length, 6);
  } finally { bindShell(prev); }
}));

test('quick sheet: "Phone voice" is offered only when this device has one', dom, () => withDom(async () => {
  const { shell } = fakeShell();
  const prev = bindShell(shell);
  try {
    const vals = (sheet) => [...sheet.querySelectorAll('#drive-sheet-slow .drive-seg-btn')].map((b) => b.dataset.value);
    const none = buildQuickSheet({ status: ON_STATUS(), bot: null, voices: [], android: false, lang: 'en' });
    assert.deepEqual(vals(none), ['wait', 'text']);
    const some = buildQuickSheet({ status: ON_STATUS(), bot: null, voices: [{ name: 'A', lang: 'en-US', localService: true, voiceURI: 'a' }], android: false, lang: 'en' });
    assert.deepEqual(vals(some), ['phone', 'wait', 'text']);
  } finally { bindShell(prev); }
}));

test('setBotVoice refuses an unknown bot', async () => {
  const saved = globalThis.fetch;
  globalThis.fetch = fakeFetch({ 'GET /api/bots/all': { bots: [{ id: 'a' }] } });
  try { await assert.rejects(setBotVoice('zz', 'p1')); } finally { globalThis.fetch = saved; }
});

// ------------------------------------------------------------ Bots popover ---

test('Bots popover: voice and role-play text wait for Save (onChange); the two switches save at once to voice.yaml', dom, async () => {
  const fetch = fakeFetch({ 'PATCH /api/voice/settings': (req) => req.body });
  await withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    try {
      const changes = [];
      const st = { ...ON_STATUS(), bots: { atlas: { backchannel: false, follow_up: true } } };
      const pill = botVoicePill({ id: 'atlas', name: 'Atlas', voice: '', voice_rp: '' }, st, (p) => changes.push(p));
      document.body.append(pill);
      const btn = pill.querySelector('.bm-voice-pill');
      assert.equal(btn.getAttribute('aria-expanded'), 'false');
      btn.click();
      const pop = pill.querySelector('.bm-voice-pop');
      assert.ok(pop);
      const [voiceSel, rpSel] = pop.querySelectorAll('select');
      voiceSel.value = 'p1'; voiceSel.dispatchEvent(new Event('change'));
      rpSel.value = 'skip_actions'; rpSel.dispatchEvent(new Event('change'));
      assert.deepEqual(changes, [{ voice: 'p1' }, { voice_rp: 'skip_actions' }]);
      assert.equal(fetch.calls.length, 0, 'nothing saved yet: the Bots tab Save sends them');
      assert.match(btn.textContent, /Robin/);
      const [bc] = pop.querySelectorAll('input.dev-switch');
      bc.checked = true; bc.dispatchEvent(new Event('change'));
      await settle();
      assert.deepEqual(fetch.calls[0].body, { bots: { atlas: { backchannel: true } } });
      // Escape closes it.
      pop.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
      assert.equal(pill.querySelector('.bm-voice-pop'), null);
    } finally { bindShell(prev); }
  }, { fetch });
});

// ------------------------------------------------------ review fixes ---

const EN = loadBundle('en');
const EN_LOCKED_SAME = 'Same as standard';

test('Settings → Voice: Follow-up is three-state — the bot\'s default, On, Off (review L4)', dom, () => withDom(async () => {
  const { shell } = fakeShell();
  const prev = bindShell(shell);
  try {
    const host = document.createElement('div');
    document.body.append(host);
    mountDriveSettings(host, { status: ON_STATUS(), bots: () => [], voices: [] });
    const btn = (v) => host.querySelector(`#drive-follow-up .drive-seg-btn[data-value="${v}"]`);
    assert.deepEqual([...host.querySelectorAll('#drive-follow-up .drive-seg-btn')].map((b) => b.textContent),
      [EN.drive.follow_up_bot, EN.drive.follow_up_on, EN.drive.follow_up_off]);
    assert.equal(btn('bot').getAttribute('aria-checked'), 'true', "a fresh device follows the bot");
    btn('off').click();
    assert.equal(JSON.parse(localStorage.getItem('dispatch-drive')).followUp, false);
    assert.equal(startPayload(JSON.parse(localStorage.getItem('dispatch-drive'))).follow_up, false);
    btn('bot').click();
    assert.equal(JSON.parse(localStorage.getItem('dispatch-drive')).followUp, null, 'back to the bot\'s default');
    assert.equal('follow_up' in startPayload(JSON.parse(localStorage.getItem('dispatch-drive'))), false);
    assert.equal(followUpValue(true), 'on');
    assert.equal(followUpValue(undefined), 'bot');
  } finally { bindShell(prev); }
}));

test('Settings → Voice: a save merges only what was sent into the status (deep), then asks for a re-read (review L3)', dom, async () => {
  // The real PATCH answers with the settings view, not the status shape.
  const fetch = fakeFetch({ 'PATCH /api/voice/settings': () => ({ enabled: true, rt: { power_hold: 'performance' }, opus: 'auto', rp_speech: 'dialogue_narrator' }) });
  await withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    try {
      const host = document.createElement('div');
      document.body.append(host);
      const st = { ...ON_STATUS(), rp_speech: 'all', opus: true, rt: { stt: 'parakeet', decoder_prompt_s: 6 } };
      let saved = 0;
      mountDriveSettings(host, { status: st, bots: () => [], voices: [], saved: () => { saved += 1; } });
      const rp = host.querySelector('#drive-rp select');
      rp.value = 'dialogue_narrator'; rp.dispatchEvent(new Event('change'));
      await settle();
      assert.equal(st.rp_speech, 'dialogue_narrator');
      assert.equal(st.opus, true, 'the settings-view opus did not overwrite the status one');
      assert.deepEqual(st.rt, { stt: 'parakeet', decoder_prompt_s: 6 });
      assert.equal(saved, 1);
    } finally { bindShell(prev); }
  }, { fetch });
  const t = { rt: { stt: 'parakeet', decoder_prompt_s: 6 }, a: 1 };
  mergeInto(t, { rt: { stt_engine: 'moonshine' }, b: [1], __proto__: { x: 1 } });
  assert.deepEqual(t, { rt: { stt: 'parakeet', decoder_prompt_s: 6, stt_engine: 'moonshine' }, a: 1, b: [1] });
});

test('save errors name the server\'s reason for a refusal, cut to length (review L7)', dom, () => withDom(async () => {
  const { shell } = fakeShell();
  const prev = bindShell(shell);
  try {
    const err = (status, message) => Object.assign(new Error(message), { status });
    assert.equal(saveErrorText(err(422, 'Unknown setting: rp_speech')), `${EN.drive.err_save} (Unknown setting: rp_speech)`);
    assert.equal(saveErrorText(err(500, 'boom')), EN.drive.err_save, 'a crash is not explained');
    assert.equal(saveErrorText(err(422, '422')), EN.drive.err_save);
    assert.equal(saveErrorText(null), EN.drive.err_save);
    assert.ok(saveErrorText(err(400, 'x'.repeat(500))).length < EN.drive.err_save.length + 150);
  } finally { bindShell(prev); }
}));

test('Bots popover: kept inside the scrolling pane — shifted right at phone width, narrowed when needed (review M2)', dom, () => withDom(async () => {
  const { shell } = fakeShell();
  const prev = bindShell(shell);
  try {
    // jsdom has no layout: give the pane, the pill and the popover the boxes
    // the real shell measured at 390 px (popover −246…94, pane 0…390).
    const pane = document.createElement('div');
    pane.style.overflowY = 'auto'; pane.style.overflowX = 'auto';
    document.body.append(pane);
    const pill = botVoicePill({ id: 'atlas', name: 'Atlas', voice: '', voice_rp: '' }, ON_STATUS(), () => {});
    pane.append(pill);
    const box = (l, r) => () => ({ left: l, right: r, width: r - l, top: 0, bottom: 10, height: 10, x: l, y: 0 });
    pane.getBoundingClientRect = box(0, 390);
    pill.getBoundingClientRect = box(60, 94);
    let pop = null;
    const origAppend = pill.append.bind(pill);
    pill.append = (...nodes) => {
      for (const n of nodes) if (n.classList && n.classList.contains('bm-voice-pop')) {
        pop = n;
        n.getBoundingClientRect = () => {
          const maxW = parseFloat(n.style.maxWidth) || 340;
          const w = Math.min(340, maxW);
          const a = pill.getBoundingClientRect();
          const left = n.style.left ? a.left + parseFloat(n.style.left) : a.right - w;
          return { left, right: left + w, width: w, top: 0, bottom: 300, height: 300, x: left, y: 0 };
        };
      }
      return origAppend(...nodes);
    };
    pill.querySelector('.bm-voice-pill').click();
    assert.ok(pop, 'opened');
    const r = pop.getBoundingClientRect();
    assert.equal(r.left, POPOVER_GUTTER, 'pinned to the pane\'s left gutter');
    assert.ok(r.right <= 390 - POPOVER_GUTTER, `right ${r.right}`);
    assert.equal(pop.style.right, 'auto');
    // A pane narrower than the popover: it narrows to fit.
    pill.querySelector('.bm-voice-pill').click();
    pane.getBoundingClientRect = box(0, 300);
    pill.querySelector('.bm-voice-pill').click();
    const r2 = pop.getBoundingClientRect();
    assert.equal(r2.width, 300 - 2 * POPOVER_GUTTER);
    assert.equal(r2.left, POPOVER_GUTTER);
    // Plenty of room (desktop): left where the stylesheet put it.
    pill.querySelector('.bm-voice-pill').click();
    pane.getBoundingClientRect = box(0, 1366);
    pill.getBoundingClientRect = box(900, 934);
    pop = null;
    pill.querySelector('.bm-voice-pill').click();
    assert.equal(pop.style.left, '');
    assert.equal(typeof placePopover, 'function');
  } finally { bindShell(prev); }
}));

test('phone voice labels spell the region out: "English (United States)" (SPEC §5.4)', async () => {
  const { langLabel } = await import('../../src/dispatch_voice/static/drive-settings.js');
  assert.equal(langLabel('en_US', 'en'), 'English (United States)');
  assert.equal(langLabel('en-GB', 'en'), 'English (United Kingdom)');
});

// ------------------------------------------------ final-review fixes ---

test('engine line: short sentences, a named engine, no middle dots (FR-ux L6)', () => {
  const line = engineLine({ tts: 'auto', stock_voice: 's', smart_turn: true, rt: { stt: 'parakeet' } });
  assert.doesNotMatch(line, /·/);
  assert.match(line, /drive\.engine_turn_on drive\.engine_stt/);
});

test('Settings → Voice: every control is described by its hint; server-wide groups say so (FR-ux L7, L8)', dom, () => withDom(async () => {
  const { shell } = fakeShell();
  const prev = bindShell(shell);
  try {
    const host = document.createElement('div');
    document.body.append(host);
    mountDriveSettings(host, { status: ON_STATUS(), cloning: true, bots: () => [], voices: [], android: false, setEnabled: async () => true });
    const controls = [...host.querySelectorAll('input, select, [role="radiogroup"]')];
    assert.ok(controls.length >= 15);
    for (const c of controls) {
      const row = c.closest('.dev-row');
      const hint = row && row.querySelector('.dev-row-hint');
      if (!hint) continue;
      assert.equal(c.getAttribute('aria-describedby'), hint.id, `${row.id}: described by its hint`);
      assert.ok(document.getElementById(hint.id) === hint);
    }
    const groupHint = (id) => host.querySelector(`#${id} > .dev-group-hint`);
    const EN = loadBundle('en');
    assert.equal(groupHint('drive-group-switch').textContent, EN.drive.group_server_hint);
    assert.equal(groupHint('drive-group-voices').textContent, EN.drive.group_server_hint);
    assert.equal(groupHint('drive-group-listening'), null, 'device groups carry no server hint');
  } finally { bindShell(prev); }
}));

test('quick sheet: without a phone voice the slow-link row shows what will really happen (FR-ux L4)', dom, () => withDom(async () => {
  const { shell } = fakeShell();
  const prev = bindShell(shell);
  try {
    const chosen = (sheet) => sheet.querySelector('#drive-sheet-slow .drive-seg-btn[aria-checked="true"]').dataset.value;
    for (const stored of ['phone', 'always_phone', 'text']) {
      localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, slowLink: stored }));
      const sheet = buildQuickSheet({ status: ON_STATUS(), bot: null, voices: [], android: false, lang: 'en' });
      assert.equal(chosen(sheet), 'text', `${stored} with no phone voice routes to text`);
      const note = [...sheet.querySelectorAll('#drive-sheet-slow .dev-row-hint')].map((n) => n.textContent);
      assert.equal(note.includes(loadBundle('en').drive.link_no_phone_voice), stored !== 'text');
    }
    localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, slowLink: 'wait' }));
    assert.equal(chosen(buildQuickSheet({ status: ON_STATUS(), bot: null, voices: [], android: false, lang: 'en' })), 'wait');
  } finally { bindShell(prev); }
}));

test('Bots popover: "Voice on locked devices" starts on Same as standard, saves at once to voice.yaml, and clears with ""', dom, async () => {
  const fetch = fakeFetch({ 'PATCH /api/voice/settings': (req) => req.body });
  await withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    try {
      const changes = [];
      const st = { ...ON_STATUS(), bots: { atlas: { backchannel: false, follow_up: true } } };
      const pill = botVoicePill({ id: 'atlas', name: 'Atlas', voice: '', voice_rp: '' }, st, (p) => changes.push(p));
      document.body.append(pill);
      pill.querySelector('.bm-voice-pill').click();
      const pop = pill.querySelector('.bm-voice-pop');
      const sel = pop.querySelector('#bm-locked-voice-atlas select');
      assert.ok(sel, 'the picker is in the popover');
      assert.equal(sel.value, '', 'default: Same as standard');
      assert.equal(sel.options[0].value, '');
      assert.equal(sel.options[0].textContent, EN_LOCKED_SAME);
      assert.ok(sel.options.length > 2, 'then the normal voice list');
      assert.ok(pop.querySelector('#bm-locked-voice-atlas .drive-preview'), 'with a preview button');
      sel.value = 'p1'; sel.dispatchEvent(new Event('change'));
      await settle();
      assert.deepEqual(fetch.calls[0].body, { bots: { atlas: { locked_voice: 'p1' } } });
      assert.deepEqual(changes, [], 'not a DisPatch bot field: it never waits for the Bots Save');
      sel.value = ''; sel.dispatchEvent(new Event('change'));
      await settle();
      assert.deepEqual(fetch.calls[1].body, { bots: { atlas: { locked_voice: '' } } });
    } finally { bindShell(prev); }
  }, { fetch });
});

test('Bots popover: a saved locked voice is shown selected', dom, () => withDom(async () => {
  const { shell } = fakeShell();
  const prev = bindShell(shell);
  try {
    const st = { ...ON_STATUS(), bots: { atlas: { locked_voice: 'p1' } } };
    const pill = botVoicePill({ id: 'atlas', name: 'Atlas', voice: '', voice_rp: '' }, st, () => {});
    document.body.append(pill);
    pill.querySelector('.bm-voice-pill').click();
    assert.equal(pill.querySelector('#bm-locked-voice-atlas select').value, 'p1');
  } finally { bindShell(prev); }
}));
