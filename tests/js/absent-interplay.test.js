// Pane ownership between DisPatch's "add-on absent" page and the add-on's
// Settings → Voice (owner: track T7; the rule is C-§E.1 / A16, the absent
// page itself is T1's). The fake shell implements the rule as T1 must:
// settings.pane() replaces the absent page, its dispose re-applies it.
//
//   - mount replaces the absent page; unmount brings it back
//   - "Check again" (the loader mounting again after an unmount, without a
//     page reload) gives exactly one of everything, never two
//   - a mount that fails (handshake) leaves the absent page untouched and
//     registers nothing
//   - a lock between mount and unmount leaves no voice chat behind

import test from 'node:test';
import assert from 'node:assert/strict';
import { domSkip } from './_require-dom.js';
import { HAS_DOM, fakeFetch, fakeShell, settle, withDom } from './_shell-fake.js';
import { mount } from '../../src/dispatch_voice/static/index.js';

const dom = { skip: domSkip(HAS_DOM ? false : 'jsdom is not installed (npm install)') };

const fetchOn = () => fakeFetch({
  'GET /api/voice/status': { enabled: true, switch: true, missing: [], profiles: [], builtin_voices: [], stock_voice: null },
  'GET /api/bots/all': { bots: [{ id: 'b', name: 'B', voice: '' }] },
  'GET /api/tools': { tools: [] },
});

test('mount replaces the absent page; unmount brings it back', dom, () => withDom(async () => {
  const fs = fakeShell();
  assert.ok(fs.paneHost.querySelector('.voice-absent'), 'before: DisPatch\'s page');
  const h = await mount(fs.shell, {});
  await settle(10);
  assert.equal(fs.paneHost.querySelector('.voice-absent'), null);
  assert.ok(fs.paneHost.querySelector('#drive-group-switch'));
  h.unmount();
  assert.ok(fs.paneHost.querySelector('.voice-absent'));
  assert.equal(fs.paneHost.querySelector('#drive-group-switch'), null);
}, { fetch: fetchOn() }));

test('"Check again" remounts cleanly: one button, one pill source, one set of tiles, one lock listener', dom, () => withDom(async () => {
  const fs = fakeShell({ thread: { id: 't', bot_id: 'b' }, bots: [{ id: 'b', name: 'B' }] });
  const h1 = await mount(fs.shell, {});
  await settle(10);
  h1.unmount();
  const h2 = await mount(fs.shell, {});
  await settle(10);
  try {
    assert.equal(fs.header.querySelectorAll('#drive-btn').length, 1);
    assert.equal(fs.log.rowExtras.filter((x) => !x.disposed).length, 1);
    assert.equal(fs.log.tools.filter((x) => !x.disposed).length, 1);
    assert.equal(fs.log.panes.filter((x) => !x.disposed).length, 1);
    assert.equal(fs.lockedListeners, 1);
    assert.ok(fs.paneHost.querySelector('#drive-group-switch'));
  } finally { h2.unmount(); }
  assert.ok(fs.paneHost.querySelector('.voice-absent'));
}, { fetch: fetchOn() }));

test('a failed mount (handshake) leaves the absent page and registers nothing', dom, () => withDom(async () => {
  const fs = fakeShell({ extApi: 2 });
  await assert.rejects(mount(fs.shell, {}));
  assert.ok(fs.paneHost.querySelector('.voice-absent'));
  assert.equal(fs.header.querySelector('#drive-btn'), null);
  assert.equal(fs.log.rowExtras.length + fs.log.tools.length + fs.log.panes.length, 0);
}, { fetch: fetchOn() }));

test('mounted on a locked shell: nothing is fetched and the button stays hidden', dom, async () => {
  const fetch = fetchOn();
  await withDom(async () => {
    const fs = fakeShell({ locked: true, thread: { id: 't', bot_id: 'b' } });
    const h = await mount(fs.shell, {});
    await settle(10);
    try {
      assert.equal(fetch.calls.filter((c) => c.path === '/api/voice/status').length, 0);
      assert.equal(fs.header.querySelector('#drive-btn').hidden, true);
    } finally { h.unmount(); }
  }, { fetch });
});
