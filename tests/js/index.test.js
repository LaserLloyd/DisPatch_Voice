// The extension entry (owner: track T7): C-§A (the JS handshake), C-§E.1
// (mount(shell, {assetsVersion}) against the frozen shell, every slot), and
// SPEC §3.5–§3.6 (what main.js did for voice now lives here).

import test from 'node:test';
import assert from 'node:assert/strict';
import { domSkip } from './_require-dom.js';
import { HAS_DOM, fakeFetch, fakeShell, settle, withDom } from './_shell-fake.js';
import { EXT, mount, versionQuery, createVoiceApp, loadModules } from '../../src/dispatch_voice/static/index.js';

const dom = { skip: domSkip(HAS_DOM ? false : 'jsdom is not installed (npm install)') };

const STATUS_ON = () => ({
  enabled: true, switch: true, forced: null, missing: [], default_voice: 'p1', stock_voice: 'stock',
  profiles: [{ id: 'p1', name: 'Robin' }, { id: 'stock', name: 'Stock' }], builtin_voices: [],
  plugins: { cloning: { installed: true, ready: true } }, rp_speech: 'all', tts: 'auto', smart_turn: true,
});
const BOTS = [
  { id: 'nova', name: 'Nova', voice: '', safe: false, visible: true },
  { id: 'alpha', name: 'Alpha', voice: '', safe: true, visible: true },
];
function routes(status = STATUS_ON(), extra = {}) {
  return fakeFetch({
    'GET /api/voice/status': () => status,
    'GET /api/bots/all': { bots: BOTS },
    'GET /api/tools': { tools: [{ id: 'voice-cloning', enabled: true }, { id: 'voice-lab', enabled: true }] },
    'PATCH /api/voice/settings': (req) => req.body,
    ...extra,
  });
}

test('EXT: id voice, accepts extension API 1..1, carries the add-on version', () => {
  assert.equal(EXT.id, 'voice');
  assert.deepEqual(EXT.api, [1, 1]);
  assert.match(EXT.version, /^\d+\.\d+\.\d+(\.dev\d+)?$/);
});

test('versionQuery: siblings carry the server\'s assets hash (?v=), else the entry\'s own query (A1)', () => {
  assert.equal(versionQuery({ assetsVersion: 'abc123' }), '?v=abc123');
  assert.equal(versionQuery({ assetsVersion: 'a b' }), '?v=a%20b');
  assert.equal(versionQuery({}, 'https://h/ext/voice/index.js?v=zz'), '?v=zz');
  assert.equal(versionQuery(null, 'https://h/ext/voice/index.js'), '');
});

test('mount refuses a shell outside the API range (the loader shows its own page)', dom, () => withDom(async () => {
  for (const api of [0, 2]) {
    const fs = fakeShell({ extApi: api });
    await assert.rejects(mount(fs.shell, {}), /extension API/);
    assert.equal(fs.log.panes.length, 0);
    assert.equal(fs.log.actions.length, 0);
  }
}));

test('mount wires every slot: header button before the thread menu, Settings → Voice pane, Bots pill, three tool tiles, lock', dom, async () => {
  await withDom(async () => {
    const fs = fakeShell({ thread: { id: 't1', bot_id: 'nova', title: 'Evening' }, bots: BOTS });
    const h = await mount(fs.shell, {});
    await settle(10);
    try {
      // Header: the mic button, just before #thread-menu-btn.
      const btn = fs.header.querySelector('#drive-btn');
      assert.ok(btn);
      assert.equal(btn.nextElementSibling.id, 'thread-menu-btn');
      assert.equal(btn.hidden, false, 'voice on + a normal bot → visible');
      assert.equal(btn.getAttribute('aria-label'), 'Voice chat (hands-free)');
      // Settings → Voice: the add-on's page replaced DisPatch's absent page.
      assert.equal(fs.paneHost.querySelector('.voice-absent'), null);
      assert.ok(fs.paneHost.querySelector('#drive-group-switch'));
      assert.ok(fs.paneHost.querySelector('#drive-group-voices'));
      assert.ok(fs.paneHost.querySelector('#drive-manage-voices'), 'cloning installed + tool on → Manage voices');
      // Bots: a pill for every bot, the Safe-Mode one included (1.3.1).
      const extra = fs.log.rowExtras[0].fn;
      assert.ok(extra(BOTS[0], () => {}).querySelector('.bm-voice-pill'));
      assert.ok(extra(BOTS[1], () => {}).querySelector('.bm-voice-pill'), 'a safe bot gets the Voice pill too');
      // Tools: three tiles.
      assert.deepEqual(fs.log.tools[0].tiles.map((x) => x.id), ['voice-cloning', 'voice-builtin', 'voice-lab']);
      assert.ok(fs.log.tools[0].tiles.every((x) => typeof x.open === 'function'));
      assert.equal(fs.lockedListeners, 1);
      // A safe bot's thread keeps the button (1.3.1: the tier is the device's,
      // not the bot's); shell.onThread still repaints it (Amendment T7a-2).
      fs.setThread({ id: 't2', bot_id: 'alpha', title: 'Kids' });
      assert.equal(btn.hidden, false);
      assert.equal(btn.dataset.ready, '1');
    } finally { h.unmount(); }
  }, { fetch: routes() });
});

test('voice off: the button hides, the pane shows the switch, no pill; turning it on refreshes everything', dom, async () => {
  const status = { enabled: false, switch: false, forced: null, missing: [] };
  const fetch = routes(status, {
    'PATCH /api/voice/settings': (req) => { Object.assign(status, STATUS_ON(), { enabled: !!req.body.enabled, switch: !!req.body.enabled }); return status; },
  });
  await withDom(async () => {
    const fs = fakeShell({ thread: { id: 't1', bot_id: 'nova' }, bots: BOTS });
    const h = await mount(fs.shell, {});
    await settle(10);
    try {
      const btn = fs.header.querySelector('#drive-btn');
      assert.equal(btn.hidden, true);
      // The Bots tab rendered while voice is off: an empty, hidden slot (the
      // shell does not re-render its roster on a tab switch, review M1).
      const slot = fs.log.rowExtras[0].fn(BOTS[0], () => {});
      assert.ok(slot && slot.classList.contains('bm-voice-host'));
      assert.equal(slot.hidden, true);
      assert.equal(slot.querySelector('.bm-voice-pill'), null, 'no pill while voice is off');
      document.body.append(slot);
      const sw = fs.paneHost.querySelector('#drive-switch-input');
      assert.ok(sw);
      sw.checked = true; sw.dispatchEvent(new Event('change'));
      await settle(20);
      assert.deepEqual(fetch.calls.find((c) => c.method === 'PATCH').body, { enabled: true });
      assert.ok(fs.log.toasts.some((x) => /Voice is on/.test(x.text)));
      assert.equal(btn.hidden, false, 'voice came on: the button shows');
      assert.ok(fs.paneHost.querySelector('#drive-group-listening'), 'the everyday settings appear');
      assert.equal(slot.hidden, false, 'the same Settings session: the slot shows');
      assert.ok(slot.querySelector('.bm-voice-pill'), 'the pill is there without a re-render');
      // And back off: the pill goes away again.
      sw.checked = false;
      fs.paneHost.querySelector('#drive-switch-input').checked = false;
      fs.paneHost.querySelector('#drive-switch-input').dispatchEvent(new Event('change'));
      await settle(20);
      assert.equal(slot.hidden, true);
      assert.equal(slot.querySelector('.bm-voice-pill'), null);
    } finally { h.unmount(); }
  }, { fetch });
});

test('a failing /api/voice/status does not fail the mount: the pane says so and Try again recovers', dom, async () => {
  let fail = true;
  const fetch = routes(null, { 'GET /api/voice/status': () => (fail ? { __status: 500, body: { detail: 'boom' } } : STATUS_ON()) });
  await withDom(async () => {
    const fs = fakeShell({ bots: BOTS });
    const h = await mount(fs.shell, {});
    await settle(10);
    try {
      assert.ok(fs.paneHost.querySelector('#drive-status-failed'));
      fail = false;
      fs.paneHost.querySelector('#drive-status-retry').click();
      await settle(20);
      assert.ok(fs.paneHost.querySelector('#drive-group-voices'));
    } finally { h.unmount(); }
  }, { fetch });
});

test('a tool tile opens the shell\'s tool pane with the tool inside; a new tile replaces it', dom, async () => {
  const fetch = routes(STATUS_ON(), {
    'GET /api/voice/lab': { switch: true, enabled: true, ready: true, missing: [], thinking: 'off', timings: {}, minimax: {}, hq: {} },
  });
  await withDom(async () => {
    const fs = fakeShell({ bots: BOTS });
    const h = await mount(fs.shell, {});
    await settle(10);
    try {
      const lab = fs.log.tools[0].tiles.find((x) => x.id === 'voice-lab');
      lab.open('voice-lab');
      await settle(20);
      const pane = fs.log.openPanes[0];
      assert.equal(pane.spec.id, 'voice-lab');
      assert.equal(pane.spec.title, 'Voice lab');
      assert.ok(Array.isArray(pane.spec.iconPaths));
      assert.ok(fs.toolHost.querySelector('#vtool-show-timings'), 'the Voice lab rendered into the pane');
      fs.log.tools[0].tiles.find((x) => x.id === 'voice-builtin').open('voice-builtin');
      await settle(10);
      assert.equal(pane.closed, true, 'the previous pane closed');
      // Manage voices opens the Voice cloning tool.
      fs.paneHost.querySelector('#dev-voices-open').click();
      await settle(10);
      assert.equal(fs.log.openPanes.at(-1).spec.id, 'voice-cloning');
    } finally { h.unmount(); }
  }, { fetch });
});

test('locking the device closes an open voice chat at once (onLocked) and hides the button', dom, async () => {
  class WS { constructor() { this.readyState = 0; this.sent = []; } send(d) { this.sent.push(d); } close() { this.readyState = 3; } }
  class AC { constructor() { this.currentTime = 0; this.audioWorklet = { addModule: async () => {} }; this.destination = {}; } resume() { return Promise.resolve(); } close() {} createGain() { return { gain: { cancelScheduledValues() {}, setTargetAtTime() {} }, connect() {} }; } createAnalyser() { return { connect() {}, fftSize: 512 }; } createOscillator() { return { frequency: {}, connect() {}, start() {}, stop() {} }; } }
  class Audio { play() { return Promise.resolve(); } pause() {} }
  await withDom(async () => {
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: () => new Promise(() => {}), enumerateDevices: async () => [] } });
    const fs = fakeShell({ thread: { id: 't1', bot_id: 'nova', title: 'x' }, bots: BOTS });
    const h = await mount(fs.shell, {});
    await settle(10);
    try {
      const btn = fs.header.querySelector('#drive-btn');
      btn.click();
      assert.ok(document.querySelector('.drive-overlay'), 'voice chat open');
      fs.emitLocked();
      assert.equal(document.querySelector('.drive-overlay'), null, 'closed on lock');
      assert.equal(btn.hidden, true);
      assert.equal(fs.log.toasts.length, 0, 'quietly');
    } finally { h.unmount(); }
  }, { fetch: routes(), WebSocket: WS, AudioContext: AC, Audio });
});

test('unmount gives every slot back: button, pane (absent page returns), pill, tiles, lock listener, the shell binding', dom, async () => {
  await withDom(async () => {
    const fs = fakeShell({ bots: BOTS });
    const shellMod = await import('../../src/dispatch_voice/static/shell.js?v=h1');
    const h = await mount(fs.shell, { assetsVersion: 'h1' });
    await settle(10);
    assert.equal(shellMod.getShell(), fs.shell, 'siblings share the ?v=h1 instance the entry bound');
    const bridge = await import('../../src/dispatch_voice/static/bridge.js?v=h1');
    assert.match(bridge.assetUrl('voice-worklet.js'), /voice-worklet\.js\?v=h1$/);
    h.unmount();
    assert.equal(fs.header.querySelector('#drive-btn'), null);
    assert.ok(fs.paneHost.querySelector('.voice-absent'), 'pane disposed → DisPatch\'s absent page');
    assert.equal(fs.log.rowExtras[0].disposed, true);
    assert.equal(fs.log.tools[0].disposed, true);
    assert.equal(fs.lockedListeners, 0);
    assert.notEqual(shellMod.getShell(), fs.shell);
    h.unmount();   // idempotent
  }, { fetch: routes() });
});

test('onLocale re-renders the pane and the button in the new words', dom, async () => {
  await withDom(async () => {
    const fs = fakeShell({ bots: BOTS });
    const h = await mount(fs.shell, {});
    await settle(10);
    try {
      const before = fs.paneHost.firstChild;
      h.onLocale();
      assert.notEqual(fs.paneHost.firstChild, before, 'rebuilt');
      assert.ok(fs.paneHost.querySelector('#drive-group-voices'));
    } finally { h.unmount(); }
  }, { fetch: routes() });
});

test('Bots pill: an unsaved popover edit survives a status repaint; a forced refresh re-reads the roster (review M1/L6)', dom, async () => {
  const fetch = routes();
  await withDom(async () => {
    const fs = fakeShell({ thread: { id: 't1', bot_id: 'nova', title: 'Evening' }, bots: BOTS });
    const app = createVoiceApp(fs.shell, await loadModules(''));
    await app.start();
    await settle(10);
    try {
      const rosterReads = () => fetch.calls.filter((c) => c.method === 'GET' && c.path === '/api/bots/all').length;
      const n0 = rosterReads();
      const changes = [];
      const slot = fs.log.rowExtras[0].fn({ ...BOTS[0] }, (p) => changes.push(p));
      document.body.append(slot);
      slot.querySelector('.bm-voice-pill').click();
      const sel = slot.querySelector('.bm-voice-pop select');
      sel.value = 'p1'; sel.dispatchEvent(new Event('change'));
      assert.deepEqual(changes, [{ voice: 'p1' }], 'handed to the shell (waits for Save)');
      slot.querySelector('.bm-voice-pill').click();      // close the popover
      await app.refreshStatus({ force: true });
      await settle(5);
      assert.ok(rosterReads() > n0, 'a forced refresh re-reads /api/bots/all');
      assert.match(slot.querySelector('.bm-voice-pill').textContent, /Robin/, 'the unsaved choice is still shown after the repaint');
      // An open popover is never torn down by a background repaint.
      slot.querySelector('.bm-voice-pill').click();
      await app.refreshStatus({ force: true });
      assert.ok(slot.querySelector('.bm-voice-pop'), 'still open');
    } finally { app.stop(); }
  }, { fetch });
});

test('voice off: the hidden button names the real reason, not "unlock" (V3 #6)', dom, async () => {
  const status = { enabled: false, switch: false, forced: null, missing: [] };
  await withDom(async () => {
    const fs = fakeShell({ thread: { id: 't1', bot_id: 'nova' }, bots: BOTS });
    const h = await mount(fs.shell, {});
    await settle(10);
    try {
      const btn = fs.header.querySelector('#drive-btn');
      assert.equal(btn.hidden, true);
      assert.match(btn.getAttribute('aria-label'), /turned off in Settings › Voice/);
      assert.doesNotMatch(btn.getAttribute('aria-label'), /Unlock/);
    } finally { h.unmount(); }
  }, { fetch: routes(status) });
});
