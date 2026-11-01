// Per-profile cloned voices in the client (add-on 1.6.0): the server tells each row what the
// caller may do (`household`, `mine`, `can_manage`, `can_download`); the pane shows only that, and
// a profile without `voice_lab` gets no clone UI. A shell without `can` is the old manager.

import test from 'node:test';
import assert from 'node:assert/strict';
import { domSkip } from './_require-dom.js';
import { HAS_DOM, fakeFetch, fakeShell, settle, withDom } from './_shell-fake.js';
import { createVoicesPane, voiceRights } from '../../src/dispatch_voice/static/voices.js';
import { voiceOptions } from '../../src/dispatch_voice/static/drive-settings.js';
import { mount } from '../../src/dispatch_voice/static/index.js';
import { bindShell } from '../../src/dispatch_voice/static/shell.js';

const dom = { skip: domSkip(HAS_DOM ? false : 'jsdom is not installed (npm install)') };

const USER = ['use_chat', 'use_voice', 'voice_lab'];
const BASIC = ['use_chat', 'use_voice'];
const ADMIN = [...USER, 'manage_plugins', 'manage_characters', 'admin_system'];

const row = (id, name, over = {}) => ({
  id, name, consent: { kind: 'self' }, created: 1, household: false, mine: true,
  can_manage: true, can_download: true, ...over,
});
const ROWS = () => [
  row('mine-1', 'My voice'),
  row('house-1', 'House voice', { household: true, mine: false, can_manage: false, can_download: false }),
];

async function inPane(perms, voices, fn, routes = {}, { noProfiles = false } = {}) {
  const fetch = fakeFetch({
    'GET /api/voice/profiles': { voices, default: null },
    'GET /api/voice/status': { builtin_voices: [], stock_voice: null, host_profiles: !!perms && !noProfiles },
    ...routes,
  });
  return withDom(async () => {
    const { shell } = fakeShell(perms ? { perms } : {});
    const prev = bindShell(shell);
    const host = document.createElement('div');
    document.body.append(host);
    const calls = { confirm: [], prompt: [], toasts: [] };
    try {
      const pane = createVoicesPane(host, {
        embedded: true,
        ...(perms ? { can: (p) => perms.includes(p) } : {}),
        toast: (m, e) => calls.toasts.push({ m, e }),
        confirm: async (m) => { calls.confirm.push(m); return true; },
        prompt: async (m, d) => { calls.prompt.push(m); return `${d} 2`; },
        onChanged() {},
      });
      pane.open();
      await settle(15);
      return await fn(host, fetch, calls);
    } finally { bindShell(prev); }
  }, { fetch });
}

const rowOf = (host, id) => host.querySelector(`.voices-row[data-id="${id}"]`);
const btn = (r, cls) => r.querySelector(cls);

test('voiceRights: the server flags win; a row without them is the old single-tenant manager', () => {
  assert.deepEqual(voiceRights({ id: 'x' }, { admin: true }),
    { tenant: false, household: false, mine: false, manage: true, download: true, setDefault: true, publish: false });
  const own = voiceRights(row('a', 'A'), { admin: false });
  assert.deepEqual([own.manage, own.download, own.setDefault, own.publish, own.mine], [true, true, false, false, true]);
  const hh = voiceRights(ROWS()[1], { admin: false });
  assert.deepEqual([hh.manage, hh.download, hh.setDefault, hh.publish], [false, false, false, false]);
  const adminHh = voiceRights({ ...ROWS()[1], can_manage: true, can_download: true }, { admin: true });
  assert.deepEqual([adminHh.manage, adminHh.setDefault, adminHh.publish], [true, true, true]);
  const adminOwn = voiceRights(row('a', 'A'), { admin: true });
  assert.deepEqual([adminOwn.setDefault, adminOwn.publish], [false, true], 'a private voice is never the default');
});

test('a profile with voice_lab: "My voices", own clone fully editable, household voice labelled and read-only', dom, async () => {
  await inPane(USER, ROWS(), async (host) => {
    assert.equal(host.querySelector('#voices-list-title').textContent, 'My voices');
    const own = rowOf(host, 'mine-1');
    const hh = rowOf(host, 'house-1');
    assert.ok(own.querySelector('.voices-mine') && !own.querySelector('.voices-household'));
    assert.ok(hh.querySelector('.voices-household'));
    for (const cls of ['.voices-delete', '.voices-download']) assert.ok(btn(own, cls), `own ${cls}`);
    assert.equal(btn(own, '.voices-publish'), null, 'only an Admin can publish');
    for (const cls of ['.voices-delete', '.voices-download', '.voices-publish']) assert.equal(btn(hh, cls), null, `household ${cls}`);
    assert.equal(hh.querySelectorAll('.voices-actions button').length, 1, 'just the sample');
    assert.ok(host.querySelector('#voices-add'), 'may add a voice');
    assert.equal(host.querySelector('#voices-builtin'), null, 'built-in voices and the default are the Admin\'s');
    assert.equal(rowOf(host, 'mine-1').textContent.includes('Set as default'), false);
  });
});

test('the sample of a voice you cannot download plays the server preview, not the recording', dom, async () => {
  const played = [];
  await withDom(async () => {
    globalThis.Audio = class { constructor(u) { played.push(u); } play() { return Promise.resolve(); } pause() {} };
    await inPane(USER, ROWS(), async (host) => {
      const sample = (id) => rowOf(host, id).querySelector('.voices-actions button');
      sample('house-1').click();
      sample('mine-1').click();
    });
  });
  assert.deepEqual(played, ['/api/voice/preview?voice=house-1', '/api/voice/profiles/mine-1/files/audition.wav']);
});

test('without voice_lab there is no clone UI', dom, async () => {
  await inPane(BASIC, ROWS(), async (host) => {
    assert.equal(host.querySelector('#voices-add'), null);
    assert.equal(host.querySelector('.voices-record'), null);
  });
});

test('an Admin: household checkbox on the add form, Make household / Make private, Set as default for household voices', dom, async () => {
  const rows = [
    row('adm-1', 'Admin voice'),
    row('house-1', 'House voice', { household: true, mine: false }),
  ];
  const patches = [];
  await inPane(ADMIN, rows, async (host, fetch, calls) => {
    const own = rowOf(host, 'adm-1');
    const hh = rowOf(host, 'house-1');
    assert.equal(btn(own, '.voices-publish').textContent, 'Make household');
    assert.equal(btn(hh, '.voices-publish').textContent, 'Make private');
    assert.ok([...hh.querySelectorAll('button')].some((b) => b.textContent === 'Set as default'));
    assert.ok(![...own.querySelectorAll('button')].some((b) => b.textContent === 'Set as default'), 'a private voice is not a default candidate');
    assert.ok(host.querySelector('#voices-builtin'), 'server-wide controls stay with the Admin');
    btn(own, '.voices-publish').click();
    await settle(15);
    assert.equal(calls.confirm.length, 1);
    assert.deepEqual(patches, [{ household: true }]);
  }, { 'PATCH /api/voice/profiles/adm-1': (req) => { patches.push(req.body); return {}; } });
});

test('the household checkbox exists for an Admin only', dom, async () => {
  // The add form is built when a clip is chosen; drive it through the file input.
  for (const [perms, expected] of [[USER, false], [ADMIN, true]]) {
    await inPane(perms, [], async (host) => {
      const input = host.querySelector('#voices-add input[type=file]');
      Object.defineProperty(input, 'files', { value: [new Blob([new Uint8Array(10)], { type: 'audio/wav' })] });
      input.dispatchEvent(new Event('change'));
      await settle(15);
      assert.equal(!!host.querySelector('.voices-household-opt'), expected);
    }, { 'POST /api/voice/profiles/analyse': { ok: true, checks: [] } });
  }
});

test('a DisPatch 3.x Admin (no voice_lab permission) still gets the add form, without the household option', dom, async () => {
  const admin3 = ADMIN.filter((p) => p !== 'voice_lab');
  await inPane(admin3, [], async (host) => {
    assert.ok(host.querySelector('#voices-add'), 'the Add-voice section stays');
    const input = host.querySelector('#voices-add input[type=file]');
    Object.defineProperty(input, 'files', { value: [new Blob([new Uint8Array(10)], { type: 'audio/wav' })] });
    input.dispatchEvent(new Event('change'));
    await settle(15);
    assert.equal(host.querySelector('.voices-household-opt'), null, 'no profiles on this host: no household option');
  }, { 'POST /api/voice/profiles/analyse': { ok: true, checks: [] } }, { noProfiles: true });
});

test('on a profiles host an Admin without voice_lab gets no add form', dom, async () => {
  await inPane(ADMIN.filter((p) => p !== 'voice_lab'), [], async (host) => {
    assert.equal(host.querySelector('#voices-add'), null);
  });
});

test('rename and delete go to the server for an own clone', dom, async () => {
  const seen = [];
  await inPane(USER, ROWS(), async (host) => {
    const own = rowOf(host, 'mine-1');
    [...own.querySelectorAll('button')].find((b) => b.textContent === 'Rename').click();
    await settle(15);
    btn(own, '.voices-delete').click();
    await settle(15);
  }, {
    'PATCH /api/voice/profiles/mine-1': (req) => { seen.push(['PATCH', req.body]); return {}; },
    'DELETE /api/voice/profiles/mine-1': () => { seen.push(['DELETE']); return {}; },
  });
  assert.deepEqual(seen, [['PATCH', { name: 'My voice 2' }], ['DELETE']]);
});

test('a host without shell.can and a server without the flags: the old manager, unchanged', dom, async () => {
  const old = [{ id: 'alex', name: 'Alex', consent: { kind: 'self' } }];
  await inPane(null, old, async (host) => {
    assert.equal(host.querySelector('#voices-list-title').textContent, 'Your voices');
    const r = rowOf(host, 'alex');
    assert.equal(r.querySelector('.voices-scope'), null, 'no badges');
    assert.ok(r.querySelector('.voices-delete'));
    assert.equal(r.querySelector('.voices-publish'), null);
    assert.equal(r.querySelector('.voices-download'), null);
    assert.ok([...r.querySelectorAll('button')].some((b) => b.textContent === 'Set as default'));
    assert.ok(host.querySelector('#voices-add') && host.querySelector('#voices-builtin'));
  });
});

test('the household pickers never offer a personal clone', () => {
  const st = { profiles: [
    { id: 'h', name: 'House', household: true, mine: false },
    { id: 'p', name: 'Mine', household: false, mine: true },
    { id: 'old', name: 'Old' },
  ], builtin_voices: [] };
  assert.deepEqual(voiceOptions(st).map((o) => o.value), ['h', 'old']);
  assert.ok(voiceOptions(st, { current: 'p' }).some((o) => o.value === 'p' && o.group === 'missing'), 'a saved voice still shows');
});

// ---------------------------------------------------------------- Settings -> Voice

const STATUS = () => ({
  enabled: true, switch: true, forced: null, missing: [], default_voice: '', stock_voice: null,
  profiles: [], builtin_voices: [], plugins: { cloning: { installed: true, ready: true } },
  rp_speech: 'all', tts: 'auto', smart_turn: true,
});
function app(perms) {
  return fakeFetch({
    'GET /api/voice/status': () => STATUS(),
    'GET /api/bots/all': { bots: [] },
    'GET /api/tools': { tools: [] },
    'GET /api/voice/profiles': { voices: ROWS(), default: null },
  });
}

test('Settings -> Voice: a profile that may clone gets My voices; one that may not, and a 2.x shell, do not', dom, async () => {
  for (const [perms, expected] of [[USER, true], [BASIC, false], [ADMIN, false], [undefined, false]]) {
    const fetch = app(perms);
    await withDom(async () => {
      const fs = fakeShell({ thread: { id: 't1', bot_id: 'nova', title: 'x' }, ...(perms ? { perms } : {}) });
      const h = await mount(fs.shell, {});
      await settle(25);
      try {
        const group = fs.paneHost.querySelector('#drive-group-my-voices');
        assert.equal(!!group, expected, JSON.stringify(perms));
        const listed = fetch.calls.some((c) => c.path === '/api/voice/profiles');
        assert.equal(listed, expected, 'the voice list is only fetched when it is shown');
        if (expected) assert.ok(group.querySelector('.voices-row[data-id="mine-1"]'));
      } finally { h.unmount(); }
    }, { fetch });
  }
});
