// Settings → Voice, 1.6.0 (static/voice-sections.js): the providers, the rig's address and
// write-only token, "set by environment", plugins, the model download, the input mode and the
// microphone test with its latency readout.

import test from 'node:test';
import assert from 'node:assert/strict';
import { domSkip } from './_require-dom.js';
import { HAS_DOM, fakeFetch, fakeShell, settle, withDom } from './_shell-fake.js';
import { mountDriveSettings } from '../../src/dispatch_voice/static/drive-settings.js';
import { bindShell } from '../../src/dispatch_voice/static/shell.js';
import {
  inputMode, inputModePatch, providerOptions, selectedProvider, envOf, healthOf,
} from '../../src/dispatch_voice/static/voice-sections.js';

const dom = { skip: domSkip(HAS_DOM ? false : 'jsdom is not installed (npm install)') };

const PROVIDERS = {
  providers: [
    { kind: 'stt', id: 'cpu', name: 'This computer (Parakeet)' }, { kind: 'stt', id: 'rig', name: 'Speech rig (GPU)' },
    { kind: 'stt', id: 'openai', name: 'OpenAI-compatible server' }, { kind: 'stt', id: 'faster', name: 'faster-whisper' },
    { kind: 'tts', id: 'cpu', name: 'This computer' }, { kind: 'tts', id: 'rig', name: 'Speech rig (GPU)' },
    { kind: 'tts', id: 'openai', name: 'OpenAI-compatible server' },
  ],
  plugins: [{ name: 'kokoro', target: 'dispatch_voice_kokoro:setup', enabled: false }],
  selected: { stt: 'cpu', tts: 'cpu', vad: 'cpu' },
};

const STATUS = (over = {}) => ({
  enabled: true, switch: true, forced: null, missing: [], ready: true,
  default_voice: 'p1', speed: 1, stock_voice: 'stock', tts: 'auto', smart_turn: true,
  profiles: [{ id: 'p1', name: 'Robin' }, { id: 'stock', name: 'Stock' }],
  builtin_voices: [{ id: 'af_heart', name: 'Heart' }],
  plugins: { cloning: { installed: true, ready: true } },
  rp_speech: 'all', narrator_voice: '', expressive_tags: true, voice_turn_thinking: 'off',
  sessions: { open: 0, max: 3 }, power_hold: 'off', rt: { stt: 'parakeet' },
  providers: PROVIDERS, health: [], system: [], models: { state: 'idle', percent: 0 },
  settings_env: {}, tts_engines: ['auto', 'chatterbox', 'minimax'],
  config: { rig_url: '', stt_url: '', stt_model: '', hq_url: '', hq_model: '', extensions: [] },
  rig: { configured: false, enabled: false, url_set: false, token: null },
  ...over,
});

function mount(status, ctx = {}) {
  const host = document.createElement('div');
  document.body.append(host);
  mountDriveSettings(host, { status, bots: () => [], voices: [], android: false, ...ctx });
  return host;
}

const ok = { 'PATCH /api/voice/settings': (req) => req.body };
const change = (el, value) => { el.value = value; el.dispatchEvent(new Event('change')); };

// ------------------------------------------------------------- pure helpers --

test('input mode is two device switches seen as one choice', () => {
  assert.equal(inputMode({ typedInput: false, autoListen: true }), 'hands_free');
  assert.equal(inputMode({ typedInput: false, autoListen: false }), 'push_to_talk');
  assert.equal(inputMode({ typedInput: true, autoListen: true }), 'type');
  assert.deepEqual(inputModePatch('type'), { typedInput: true });
  assert.deepEqual(inputModePatch('push_to_talk'), { typedInput: false, autoListen: false });
  assert.deepEqual(inputModePatch('hands_free'), { typedInput: false, autoListen: true });
});

test('provider options: the three built-ins first, then what plugins registered', () => {
  assert.deepEqual(providerOptions(STATUS(), 'stt').map((o) => o.value), ['cpu', 'rig', 'openai', 'faster']);
  assert.equal(providerOptions(STATUS(), 'stt').find((o) => o.value === 'faster').label, 'faster-whisper');
  assert.deepEqual(providerOptions(STATUS(), 'tts').map((o) => o.value), ['cpu', 'rig', 'openai']);
  assert.equal(selectedProvider(STATUS(), 'stt'), 'cpu');
  assert.equal(selectedProvider({}, 'tts'), 'cpu');
  assert.equal(envOf(STATUS({ settings_env: { speed: 'DISPATCH_VOICE_SPEED' } }), 'speed'), 'DISPATCH_VOICE_SPEED');
  assert.equal(envOf(STATUS(), 'speed'), null);
  assert.equal(healthOf(STATUS({ health: [{ id: 'rig', state: 'ok' }] }), 'rig'), null);
  assert.equal(healthOf(STATUS({ health: [{ id: 'rig', state: 'down', reason: 'r', fix: 'f' }] }), 'rig').fix, 'f');
});

// ------------------------------------------------------- 1 speech to text --

test('speech to text: the provider select saves; the rig switches rig.enabled, a server turns the rig off', dom, async () => {
  const fetch = fakeFetch(ok);
  await withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    try {
      const off = () => STATUS({ rig: { configured: true, enabled: false, url_set: true, token: 'file' } });
      const host = mount(off());
      const sel = host.querySelector('#drive-stt-provider');
      assert.deepEqual([...sel.options].map((o) => o.value), ['cpu', 'rig', 'openai', 'faster']);
      change(sel, 'rig');
      await settle();
      assert.deepEqual(fetch.calls[0].body, { rig: { enabled: true } });
      change(mount(off()).querySelector('#drive-stt-provider'), 'openai');
      await settle();
      assert.deepEqual(fetch.calls[1].body, { stt_provider: 'openai' }, 'the rig was off: nothing to turn off');
      const on = STATUS({ rig: { configured: true, enabled: true, url_set: true, token: 'file' },
        providers: { ...PROVIDERS, selected: { stt: 'rig', tts: 'rig', vad: 'cpu' } } });
      assert.equal(mount(on).querySelector('#drive-stt-provider').value, 'rig');
      change(mount(on).querySelector('#drive-stt-provider'), 'openai');
      await settle();
      assert.deepEqual(fetch.calls[2].body, { stt_provider: 'openai', rig: { enabled: false } });
    } finally { bindShell(prev); }
  }, { fetch });
});

test('speech to text: the engine select and the server fields follow the provider', dom, async () => {
  const fetch = fakeFetch(ok);
  await withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    try {
      const cpu = mount(STATUS());
      change(cpu.querySelector('#drive-stt-engine'), 'moonshine');
      await settle();
      assert.deepEqual(fetch.calls[0].body, { rt: { stt_engine: 'moonshine' } });
      assert.equal(cpu.querySelector('#drive-stt-url'), null, 'no server fields for the CPU');
      const remote = mount(STATUS({
        providers: { ...PROVIDERS, selected: { ...PROVIDERS.selected, stt: 'openai' } },
        config: { rig_url: '', stt_url: 'http://127.0.0.1:8000', stt_model: 'whisper-1', hq_url: '', hq_model: '', extensions: [] },
      }));
      assert.equal(remote.querySelector('#drive-stt-engine'), null, 'no local engine for a remote server');
      const url = remote.querySelector('#drive-stt-url-input');
      assert.equal(url.value, 'http://127.0.0.1:8000');
      url.value = 'http://127.0.0.1:9000';
      remote.querySelector('#drive-stt-url-save').click();
      await settle();
      assert.deepEqual(fetch.calls[1].body, { stt_url: 'http://127.0.0.1:9000' });
    } finally { bindShell(prev); }
  }, { fetch });
});

test('the rig: address saves through the schema, the token is write-only, a refused address shows the server reason', dom, async () => {
  const fetch = fakeFetch({
    'PATCH /api/voice/settings': (req) => (req.body.rig && req.body.rig.url === 'ws://169.254.169.254/x'
      ? { __status: 422, body: { detail: 'rig.url: That is a cloud metadata address.' } } : req.body),  // scrub-ok: metadata fixture
    'PUT /api/voice/rig/token': { ok: true },
    'DELETE /api/voice/rig/token': { ok: true },
  });
  await withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    const toasts = [];
    try {
      const st = STATUS({ rig: { configured: true, enabled: false, url_set: true, token: null },
        health: [{ id: 'rig', state: 'unconfigured', reason: 'The rig token is not set.', fix: 'Paste the token.', code: 'rig_token_missing' }] });
      const host = mount(st, { toast: (m, e) => toasts.push([m, e]) });
      assert.equal(host.querySelector('#drive-rig-details').open, true, 'an address was set: the rig block is open');
      assert.match(host.querySelector('#drive-rig-url').textContent, /The rig token is not set\./);
      assert.match(host.querySelector('#drive-rig-url').textContent, /Fix: Paste the token\./);
      const url = host.querySelector('#drive-rig-url-input');
      url.value = 'ws://169.254.169.254/x';  // scrub-ok: metadata fixture
      host.querySelector('#drive-rig-url-save').click();
      await settle();
      assert.ok(toasts.some(([m, e]) => e && /metadata/.test(m)), 'the server\'s reason reaches the toast');
      url.value = 'wss://rig.example.test/v1/realtime';
      host.querySelector('#drive-rig-url-save').click();
      await settle();
      assert.deepEqual(fetch.calls.filter((c) => c.method === 'PATCH').pop().body, { rig: { url: 'wss://rig.example.test/v1/realtime' } });
      // the token
      const tok = host.querySelector('#drive-rig-token-input');
      assert.equal(tok.type, 'password');
      assert.match(host.querySelector('#drive-rig-token-state').textContent, /No token yet/);
      tok.value = 'tok-abcdefghij-123';
      host.querySelector('#drive-rig-token-save').click();
      await settle();
      const put = fetch.calls.find((c) => c.method === 'PUT');
      assert.deepEqual(put.body, { token: 'tok-abcdefghij-123' });  // scrub-ok: test fixture
      assert.equal(tok.value, '', 'the field is emptied: the browser keeps nothing');
      assert.doesNotMatch(document.body.innerHTML, /tok-abcdefghij-123/, 'the token is nowhere on the page');
      assert.match(host.querySelector('#drive-rig-token-state').textContent, /A token is saved/);
      assert.equal(host.querySelector('#drive-rig-token-remove').hidden, false);
      host.querySelector('#drive-rig-token-remove').click();
      await settle();
      assert.ok(fetch.calls.some((c) => c.method === 'DELETE' && c.path === '/api/voice/rig/token'));
    } finally { bindShell(prev); }
  }, { fetch });
});

test('set by environment: the badge names the variable and the control is disabled', dom, async () => {
  const fetch = fakeFetch(ok);
  await withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    try {
      const st = STATUS({
        settings_env: { 'rig.url': 'DISPATCH_VOICE_RIG_URL', tts_engine: 'DISPATCH_VOICE_TTS', 'rt.stt_engine': 'DISPATCH_VOICE_STT' },
        config: { rig_url: 'ws://127.0.0.1:1236/v1/realtime', stt_url: '', stt_model: '', hq_url: '', hq_model: '', extensions: [] },
        rig: { configured: true, enabled: true, url_set: true, token: 'env' },
        providers: { ...PROVIDERS, selected: { stt: 'rig', tts: 'rig', vad: 'cpu' } },
      });
      const host = mount(st);
      const urlRow = host.querySelector('#drive-rig-url');
      assert.match(urlRow.querySelector('.drive-env-badge').textContent, /Set by environment \(DISPATCH_VOICE_RIG_URL\)/);
      assert.equal(urlRow.querySelector('input').disabled, true);
      assert.equal(urlRow.querySelector('button').disabled, true);
      assert.equal(host.querySelector('#drive-stt-engine').disabled, true);
      assert.match(host.querySelector('#drive-rig-token-state').textContent, /environment provides the token/);
      assert.equal(host.querySelector('#drive-rig-token-input').disabled, true);
      assert.equal(host.querySelector('#drive-stt-provider').disabled, false, 'a setting the environment does not decide stays editable');
      const tts = mount(STATUS({ settings_env: { tts_engine: 'DISPATCH_VOICE_TTS' } }));
      assert.equal(tts.querySelector('#drive-tts-engine').disabled, true);
      assert.match(tts.querySelector('#drive-tts-engine-row .drive-env-badge').textContent, /DISPATCH_VOICE_TTS/);
      assert.equal(fetch.calls.length, 0, 'nothing is saved by rendering a locked page');
    } finally { bindShell(prev); }
  }, { fetch });
});

// -------------------------------------------------------- 2 text to speech --

test('text to speech: engine, the remote server fields, and provider plugins that apply after a restart', dom, async () => {
  const fetch = fakeFetch(ok);
  await withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    try {
      const host = mount(STATUS());
      change(host.querySelector('#drive-tts-engine'), 'minimax');
      await settle();
      assert.deepEqual(fetch.calls[0].body, { tts_engine: 'minimax' });
      const cb = host.querySelector('#drive-plugin-kokoro');
      assert.ok(cb, 'the installed plugin is listed');
      assert.match(host.querySelector('#drive-plugin-row-kokoro').textContent, /GPL-3\.0/);
      const note = host.querySelector('.drive-plugins-restart');
      assert.equal(note.hidden, true);
      cb.checked = true; cb.dispatchEvent(new Event('change'));
      await settle();
      assert.deepEqual(fetch.calls[1].body, { extensions: ['kokoro'] });
      assert.equal(note.hidden, false, 'a restart applies it');
      cb.checked = false; cb.dispatchEvent(new Event('change'));
      await settle();
      assert.deepEqual(fetch.calls[2].body, { extensions: [] });

      const remote = mount(STATUS({
        providers: { ...PROVIDERS, selected: { ...PROVIDERS.selected, tts: 'openai' } },
        config: { rig_url: '', stt_url: '', stt_model: '', hq_url: 'http://127.0.0.1:8001', hq_model: 'tts-1', extensions: [] },
      }));
      assert.equal(remote.querySelector('#drive-tts-engine'), null);
      assert.equal(remote.querySelector('#drive-tts-url-input').value, 'http://127.0.0.1:8001');
      change(remote.querySelector('#drive-tts-provider'), 'cpu');
      await settle();
      assert.deepEqual(fetch.calls.at(-1).body, { tts_provider: 'cpu' });
    } finally { bindShell(prev); }
  }, { fetch });
});

test('no plugins installed: the list says so', dom, async () => {
  await withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    try {
      const host = mount(STATUS({ providers: { ...PROVIDERS, plugins: [] } }));
      assert.match(host.querySelector('#drive-plugins').textContent, /No provider plugins are installed/);
    } finally { bindShell(prev); }
  });
});

// ------------------------------------------------------- 3 input and devices --

test('input and devices: one input-mode choice writes the two device switches', dom, async () => {
  await withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    try {
      localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3 }));
      const host = mount(STATUS());
      const seg = (v) => host.querySelector(`#drive-input-mode .drive-seg-btn[data-value="${v}"]`);
      assert.equal(seg('hands_free').getAttribute('aria-checked'), 'true');
      seg('push_to_talk').click();
      let s = JSON.parse(localStorage.getItem('dispatch-drive'));
      assert.equal(s.autoListen, false); assert.equal(s.typedInput, false);
      seg('type').click();
      s = JSON.parse(localStorage.getItem('dispatch-drive'));
      assert.equal(s.typedInput, true);
      seg('hands_free').click();
      s = JSON.parse(localStorage.getItem('dispatch-drive'));
      assert.deepEqual([s.autoListen, s.typedInput], [true, false]);
    } finally { bindShell(prev); }
  });
});

test('microphone test: a level bar, a latency readout, and the microphone is released', dom, async () => {
  const stopped = [];
  class FakeAudioContext {
    constructor() { this.baseLatency = 0.01; this.outputLatency = 0.04; }
    createMediaStreamSource() { return { connect() {} }; }
    createAnalyser() {
      return { fftSize: 1024, connect() {}, getFloatTimeDomainData(buf) { buf.fill(0.2); } };
    }
    close() { return Promise.resolve(); }
  }
  const fetch = fakeFetch({ 'GET /api/voice/settings/schema': { settings: [] } });
  await withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    try {
      Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: {
        getUserMedia: async (c) => ({ getTracks: () => [{ stop: () => stopped.push(c) }] }),
      } });
      const host = mount(STATUS());
      const btn = host.querySelector('#drive-mic-test');
      btn.click();
      await settle(10);
      assert.match(host.querySelector('#drive-mic-latency').textContent, /Audio delay 50 ms/);
      assert.match(host.querySelector('#drive-mic-out').textContent, /Hearing you/);
      assert.ok(Number(host.querySelector('#drive-mic-meter').value) > 0);
      await settle(10);
      assert.match(host.querySelector('#drive-mic-latency').textContent, /server round trip \d+ ms/);
      btn.click();                                   // stop
      assert.equal(stopped.length, 1, 'the tracks were stopped');
      assert.equal(btn.textContent, 'Test microphone');
    } finally { bindShell(prev); delete navigator.mediaDevices; }
  }, { fetch, AudioContext: FakeAudioContext });
});

test('microphone test: a browser that refuses the microphone says so', dom, async () => {
  await withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    try {
      Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: {
        getUserMedia: async () => { throw new Error('denied'); },
      } });
      const host = mount(STATUS());
      host.querySelector('#drive-mic-test').click();
      await settle(5);
      assert.match(host.querySelector('#drive-mic-out').textContent, /microphone/i);
    } finally { bindShell(prev); delete navigator.mediaDevices; }
  });
});

// ---------------------------------------------------------------- models --

test('models: Download starts the job and shows progress; Cancel is offered while it runs', dom, async () => {
  const running = { state: 'running', percent: 40, file: 'encoder-model.int8.onnx', cancellable: true };
  const fetch = fakeFetch({
    'POST /api/voice/models/fetch': running,
    'GET /api/voice/models/status': running,
    'POST /api/voice/models/cancel': { state: 'cancelled', percent: 40 },
  });
  await withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    let host;
    try {
      host = mount(STATUS({ health: [{ id: 'models', state: 'unconfigured', reason: 'x', fix: 'y', code: 'models_missing' }] }));
      assert.match(host.querySelector('#drive-models-line').textContent, /Some models are missing/);
      assert.equal(host.querySelector('#drive-models-cancel').hidden, true);
      host.querySelector('#drive-models-download').click();
      await settle(5);
      assert.ok(fetch.calls.some((c) => c.method === 'POST' && c.path === '/api/voice/models/fetch'));
      assert.match(host.querySelector('#drive-models-line').textContent, /Downloading encoder-model\.int8\.onnx: 40%/);
      assert.equal(host.querySelector('#drive-models-progress').value, 40);
      assert.equal(host.querySelector('#drive-models-cancel').hidden, false);
      assert.equal(host.querySelector('#drive-models-download').hidden, true);
      host.querySelector('#drive-models-cancel').click();
      await settle(5);
      assert.match(host.querySelector('#drive-models-line').textContent, /Download cancelled/);
      assert.equal(host.querySelector('#drive-models-download').hidden, false);
    } finally { if (host && host.paneDispose) host.paneDispose(); bindShell(prev); }
  }, { fetch });
});

test('models: the old two-folder layout offers a one-click move, and a missing system program shows its install command', dom, async () => {
  const fetch = fakeFetch({ 'POST /api/voice/models/migrate': { moved: 4 } });
  await withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    let host;
    try {
      host = mount(STATUS({
        models: { state: 'idle', percent: 0, layout: { legacy: true } },
        system: [{ id: 'ffmpeg', ok: false, optional: false, required_for: 'cloning a voice', fix: 'winget install Gyan.FFmpeg' }],
      }));
      assert.match(host.querySelector('#drive-models-legacy').textContent, /old folders/);
      host.querySelector('#drive-models-move').click();
      await settle(5);
      assert.deepEqual(fetch.calls[0].body, { dry_run: false });
      const g = host.querySelector('#drive-group-models');
      assert.match(g.textContent, /ffmpeg is not installed \(cloning a voice needs it\)/);
      assert.match(g.querySelector('code').textContent, /winget install Gyan\.FFmpeg/);
    } finally { if (host && host.paneDispose) host.paneDispose(); bindShell(prev); }
  }, { fetch });
});

test('while voice is off or not ready the page still offers the model download', dom, async () => {
  await withDom(async () => {
    const { shell } = fakeShell();
    const prev = bindShell(shell);
    let host;
    try {
      host = mount(STATUS({ enabled: false, switch: true, ready: false, missing: ['model silero_vad.onnx'] }));
      assert.ok(host.querySelector('#drive-group-models'), 'download from the page, not a terminal');
      assert.ok(host.querySelector('#drive-group-stt') && host.querySelector('#drive-group-tts'), 'a remote provider can be chosen before any download');
      assert.equal(host.querySelector('#drive-group-input'), null, 'the device sections wait until voice runs');
    } finally { if (host && host.paneDispose) host.paneDispose(); bindShell(prev); }
  });
});
