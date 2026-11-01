// Settings → Voice, accessibility floor (issue #9): the checks a script can make against the
// rendered pane. Not a full WCAG audit (contrast, screen-reader reading order and zoom need a
// person and a browser); this keeps the mechanical failures out: every control has an accessible
// name, ids are unique, nothing steals the tab order, images have alternative text, and every
// aria-labelledby / aria-controls / label-for reference resolves.

import test from 'node:test';
import assert from 'node:assert/strict';
import { domSkip } from './_require-dom.js';
import { HAS_DOM, fakeFetch, fakeShell, withDom } from './_shell-fake.js';
import { mountDriveSettings } from '../../src/dispatch_voice/static/drive-settings.js';
import { bindShell } from '../../src/dispatch_voice/static/shell.js';

const dom = { skip: domSkip(HAS_DOM ? false : 'jsdom is not installed (npm install)') };

const PROVIDERS = (stt, tts) => ({
  providers: [
    { kind: 'stt', id: 'cpu', name: 'This computer (Parakeet)' }, { kind: 'stt', id: 'rig', name: 'Speech rig (GPU)' },
    { kind: 'stt', id: 'openai', name: 'OpenAI-compatible server' },
    { kind: 'tts', id: 'cpu', name: 'This computer' }, { kind: 'tts', id: 'rig', name: 'Speech rig (GPU)' },
    { kind: 'tts', id: 'openai', name: 'OpenAI-compatible server' },
  ],
  plugins: [{ name: 'kokoro', target: 'dispatch_voice_kokoro:setup', enabled: false }],
  selected: { stt, tts, vad: 'cpu' },
});

const STATUS = (stt, tts, rig) => ({
  enabled: true, switch: true, forced: null, missing: [], ready: true,
  default_voice: 'p1', speed: 1, stock_voice: 'stock', tts: 'auto', smart_turn: true,
  profiles: [{ id: 'p1', name: 'Robin' }, { id: 'stock', name: 'Stock' }],
  builtin_voices: [{ id: 'af_heart', name: 'Heart' }],
  plugins: { cloning: { installed: true, ready: true } },
  rp_speech: 'all', narrator_voice: '', expressive_tags: true, voice_turn_thinking: 'off',
  sessions: { open: 0, max: 3 }, power_hold: 'off', rt: { stt: 'parakeet' },
  providers: PROVIDERS(stt, tts), health: [], system: [], models: { state: 'idle', percent: 0 },
  settings_env: {}, tts_engines: ['auto', 'chatterbox', 'minimax'],
  config: { rig_url: 'wss://rig.example.org/v1/realtime', stt_url: 'http://127.0.0.1:8000', stt_model: 'whisper-1',
    hq_url: 'http://127.0.0.1:8001', hq_model: 'tts-1', extensions: [] },
  rig: { configured: rig, enabled: rig, url_set: rig, token: rig ? { set: true, source: 'file' } : null },
});

function accessibleName(el, doc) {
  const aria = (el.getAttribute('aria-label') || '').trim();
  if (aria) return aria;
  const by = el.getAttribute('aria-labelledby');
  if (by) {
    const t = by.split(/\s+/).map((id) => doc.getElementById(id)?.textContent || '').join(' ').trim();
    if (t) return t;
  }
  if (el.id) {
    const lab = [...doc.querySelectorAll('label')].find((l) => l.htmlFor === el.id);
    if (lab && lab.textContent.trim()) return lab.textContent.trim();
  }
  const wrap = el.closest('label');
  if (wrap && wrap.textContent.trim()) return wrap.textContent.trim();
  if (el.tagName === 'BUTTON' && el.textContent.trim()) return el.textContent.trim();
  return (el.getAttribute('title') || '').trim();
}

for (const [label, stt, tts, rig] of [['cpu', 'cpu', 'cpu', false], ['server', 'openai', 'openai', false], ['rig', 'rig', 'rig', true]]) {
  test(`Settings > Voice (${label} providers): controls are named, ids unique, tab order untouched`, dom, async () => {
    await withDom(async () => {
      const { shell } = fakeShell();
      const prev = bindShell(shell);
      try {
        const host = document.createElement('div');
        document.body.append(host);
        mountDriveSettings(host, { status: STATUS(stt, tts, rig), bots: () => [], voices: [], android: false });
        const controls = [...host.querySelectorAll('input:not([type=hidden]), select, textarea, button, [role=button], [role=switch], [role=checkbox], [role=radio]')];
        assert.ok(controls.length > 5, 'the pane rendered controls');
        const unnamed = controls.filter((c) => !accessibleName(c, document)).map((c) => c.outerHTML.slice(0, 120));
        assert.deepEqual(unnamed, [], 'controls without an accessible name');
        const ids = [...host.querySelectorAll('[id]')].map((e) => e.id);
        assert.deepEqual(ids.filter((id, i) => ids.indexOf(id) !== i), [], 'duplicate ids');
        assert.deepEqual([...host.querySelectorAll('[tabindex]')].filter((e) => Number(e.getAttribute('tabindex')) > 0).map((e) => e.outerHTML.slice(0, 80)), [], 'positive tabindex');
        assert.deepEqual([...host.querySelectorAll('img:not([alt])')].map((e) => e.outerHTML.slice(0, 80)), [], 'images without alt');
        for (const attr of ['aria-labelledby', 'aria-controls', 'aria-describedby']) {
          for (const e of host.querySelectorAll(`[${attr}]`)) {
            for (const id of e.getAttribute(attr).split(/\s+/)) assert.ok(document.getElementById(id), `${attr}="${id}" points at nothing`);
          }
        }
        for (const l of host.querySelectorAll('label[for]')) assert.ok(document.getElementById(l.htmlFor), `label for="${l.htmlFor}" points at nothing`);
      } finally { bindShell(prev); }
    }, { fetch: fakeFetch({}) });
  });
}
