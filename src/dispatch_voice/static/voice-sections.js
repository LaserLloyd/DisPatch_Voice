// Settings → Voice, 1.6.0: the sections in the order the page presents them.
//
//   1  Speech to text       provider, engine, remote server, the GPU rig's address and token
//   2  Text to speech       provider, voice engine, default voice and speed, provider plugins
//   3  Input and devices    input mode, patience, barge-in, microphone and speaker, a mic test
//                           with a latency readout
//   4  Voices per character each character's voice, role-play speech, locked-device voice, flags
//   +  Models               download with progress and cancel, the one-time folder move
//
// Everything here is built from /api/voice/status (providers, health, system, models and
// `settings_env`: the settings the server's ENVIRONMENT decides, shown as "set by environment"
// and disabled) and saved through PATCH /api/voice/settings. The rig token has its own
// write-only route (PUT/DELETE /api/voice/rig/token): this file never reads one back.
//
// The builders get the page's own row/switch/select helpers as `H`, so there is one look and no
// import cycle with drive-settings.js (which imports this file).

const Q = new URL(import.meta.url).search;
const { el, t, has, listFormat } = await import(`./shell.js${Q}`);

export const SECTION_ORDER = Object.freeze(['stt', 'tts', 'input', 'characters']);

/** Literal keys, so the i18n scan sees them. */
const PROVIDER_NAME = {
  cpu: 'drive.provider_cpu',
  rig: 'drive.provider_rig',
  openai: 'drive.provider_openai',
};

// ---------------------------------------------------------------- helpers --

/** The non-secret values the server reports for the fields edited in place. */
const cfg = (st) => (st && st.config) || {};

/** The environment variable deciding setting `id`, or null. */
export function envOf(st, id) {
  const e = st && st.settings_env;
  return e && typeof e[id] === 'string' ? e[id] : null;
}

/** "Set by environment (DISPATCH_VOICE_SPEED)" next to a control the environment decides. */
export function envBadge(st, id) {
  const name = envOf(st, id);
  if (!name) return null;
  return el('span', {
    class: 'drive-env-badge', title: t('drive.set_by_env_hint'),
    'data-env': name, text: t('drive.set_by_env', { name }),
  });
}

/** Add the badge to a row and disable its controls when the environment decides `id`. */
function guard(st, id, rowEl, controls) {
  const badge = envBadge(st, id);
  if (!badge) return rowEl;
  for (const c of controls) { if (c) c.disabled = true; }
  const text = rowEl.querySelector('.dev-row-text');
  (text || rowEl).append(badge);
  rowEl.classList.add('drive-env-locked');
  return rowEl;
}

/** One provider name for a select: our three, or the plugin's own. */
function providerLabel(id, catalog, kind) {
  if (PROVIDER_NAME[id]) return t(PROVIDER_NAME[id]);
  const hit = (catalog || []).find((p) => p.kind === kind && p.id === id);
  return (hit && hit.name) || id;
}

/** [{value,label}] for the provider select of `kind` ('stt' | 'tts'). */
export function providerOptions(st, kind) {
  const cat = (st && st.providers && st.providers.providers) || [];
  const ids = [];
  for (const id of ['cpu', 'rig', 'openai']) ids.push(id);
  for (const p of cat) if (p.kind === kind && !ids.includes(p.id)) ids.push(p.id);
  return ids.map((id) => ({ value: id, label: providerLabel(id, cat, kind) }));
}

/** The provider the server uses now for `kind`. */
export function selectedProvider(st, kind) {
  const sel = st && st.providers && st.providers.selected;
  return (sel && sel[kind]) || 'cpu';
}

/** The health entry for a part of voice (`rig`, `stt_server`, ...), or null while all is fine. */
export function healthOf(st, id) {
  const h = (st && st.health) || [];
  const e = h.find((x) => x && x.id === id);
  return e && e.state !== 'ok' ? e : null;
}

/** A health entry as a note: the reason, then what to do about it. */
export function healthNote(entry) {
  if (!entry) return null;
  const p = el('p', { class: `dev-row-hint drive-health drive-health-${entry.state}`, role: 'note' });
  p.append(el('span', { text: entry.reason || '' }));
  if (entry.fix) p.append(' ', el('strong', { text: t('drive.health_fix', { fix: entry.fix }) }));
  return p;
}

function textInput(label, value, { type = 'text', placeholder = '', id = null, maxlength = 256 } = {}) {
  return el('input', {
    type, class: 'drive-text', 'aria-label': label, value: value || '', placeholder, id, maxlength: String(maxlength),
    autocomplete: 'off', spellcheck: 'false', dir: 'ltr',
  });
}

function saveButton(label, onClick, id = null) {
  const b = el('button', { type: 'button', class: 'btn-secondary drive-save', id, text: label });
  b.addEventListener('click', onClick);
  return b;
}

/** A text field that saves on the Save button / Enter. `save(value) -> Promise<bool>`. */
function savedField(H, st, { id, label, hint, settingId, value, placeholder, save, type = 'text' }) {
  const input = textInput(label, value, { type, placeholder, id: `${id}-input` });
  const btn = saveButton(t('drive.save'), async () => {
    btn.disabled = true;
    const ok = await save(input.value.trim());
    btn.disabled = false;
    if (!ok) input.value = value || '';
    else value = input.value.trim();
  }, `${id}-save`);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); btn.click(); } });
  const r = H.row(label, hint, [input, btn], { id });
  r.classList.add('dev-row-action');
  return guard(st, settingId, r, [input, btn]);
}

// ------------------------------------------------------- 1 speech to text --

/** The GPU rig's address, token and state: shared by both provider sections. */
export function rigBlock(H, st, save, ctx) {
  const rg = (st && st.rig) || {};
  const rows = [];
  const urlRow = savedField(H, st, {
    id: 'drive-rig-url', label: t('drive.rig_url'), hint: t('drive.rig_url_hint'), settingId: 'rig.url',
    value: cfg(st).rig_url || '', placeholder: 'wss://', save: async (v) => save({ rig: { url: v } }),
  });
  rows.push(urlRow);
  rows.push(tokenRow(H, st, ctx));
  const hn = healthNote(healthOf(st, 'rig'));
  if (hn) urlRow.append(hn);
  const toggle = H.rigRow(st, save);
  if (toggle) rows.push(guard(st, 'rig.enabled', toggle, [toggle.querySelector('input')]));
  rows.push(el('p', { class: 'dev-row-hint drive-rig-shared', text: t('drive.rig_shared_note') }));
  return rows;
}

/** Rig token: write-only. Shows whether one is set and where from, never the value. */
function tokenRow(H, st, ctx) {
  const rg = (st && st.rig) || {};
  const src = rg.token || (ctx.rigToken && ctx.rigToken.source) || null;
  const input = textInput(t('drive.rig_token'), '', { type: 'password', placeholder: t('drive.rig_token_placeholder'),
    id: 'drive-rig-token-input', maxlength: 512 });
  input.autocomplete = 'new-password';
  const state = el('span', { class: 'dev-row-hint drive-note', id: 'drive-rig-token-state' });
  const paint = (source) => {
    state.textContent = source === 'env' ? t('drive.rig_token_env') : source ? t('drive.rig_token_set') : t('drive.rig_token_none');
    del.hidden = source !== 'file';
  };
  let del = null;
  const put = saveButton(t('drive.save'), async () => {
    const v = input.value.trim();
    if (!v) return;
    put.disabled = true;
    try {
      await H.voiceJson('/api/voice/rig/token', { method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: v }) });
      input.value = '';                       // the browser keeps nothing
      paint('file');
      if (ctx.toast) ctx.toast(t('drive.rig_token_saved'));
      if (ctx.saved) ctx.saved();
    } catch (e) {
      if (ctx.toast) ctx.toast(H.saveErrorText(e), true);
    }
    put.disabled = false;
  }, 'drive-rig-token-save');
  del = saveButton(t('drive.rig_token_remove'), async () => {
    del.disabled = true;
    try {
      await H.voiceJson('/api/voice/rig/token', { method: 'DELETE' });
      paint(null);
      if (ctx.saved) ctx.saved();
    } catch (e) {
      if (ctx.toast) ctx.toast(H.saveErrorText(e), true);
    }
    del.disabled = false;
  }, 'drive-rig-token-remove');
  paint(src);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); put.click(); } });
  const r = H.row(t('drive.rig_token'), t('drive.rig_token_hint'), [input, put, del], { id: 'drive-rig-token' });
  r.classList.add('dev-row-action');
  r.querySelector('.dev-row-text').append(state);
  if (src === 'env') { input.disabled = true; put.disabled = true; del.hidden = true; }
  return r;
}

export function sttSection(H, st, save, ctx) {
  const rows = [];
  const cur = selectedProvider(st, 'stt');
  const sel = H.selectInput(t('drive.provider'), providerOptions(st, 'stt'), cur, async (v) => {
    const patch = v === 'rig' ? { rig: { enabled: true } }
      : { stt_provider: v, ...((st.rig && st.rig.enabled) ? { rig: { enabled: false } } : {}) };
    if (!(await save(patch))) sel.value = cur;
    else ctx.rebuild && ctx.rebuild();
  });
  sel.id = 'drive-stt-provider';
  const provRow = H.row(t('drive.provider'), t('drive.provider_hint_stt'), [sel], { id: 'drive-stt-provider-row' });
  const phn = healthNote(healthOf(st, 'stt_provider'));     // a provider voice chats do not use (yet)
  if (phn) provRow.append(phn);
  rows.push(guard(st, 'stt_provider', provRow, [sel]));
  if (cur === 'cpu' || cur === 'rig') {
    const engines = [{ value: 'parakeet', label: t('drive.stt_parakeet') }, { value: 'moonshine', label: t('drive.stt_moonshine') }];
    const eng = H.selectInput(t('drive.stt_engine'), engines, (st.rt && st.rt.stt) || 'parakeet', async (v) => {
      if (!(await save({ rt: { stt_engine: v } }))) eng.value = (st.rt && st.rt.stt) || 'parakeet';
    });
    eng.id = 'drive-stt-engine';
    rows.push(guard(st, 'rt.stt_engine', H.row(t('drive.stt_engine'), t('drive.stt_engine_hint'), [eng], { id: 'drive-stt-engine-row' }), [eng]));
  }
  if (cur === 'openai') {
    rows.push(savedField(H, st, { id: 'drive-stt-url', label: t('drive.server_url'), hint: t('drive.stt_url_hint'),
      settingId: 'stt_url', value: cfg(st).stt_url || '', placeholder: 'http://', save: (v) => save({ stt_url: v }) }));
    rows.push(savedField(H, st, { id: 'drive-stt-model', label: t('drive.server_model'), hint: '', settingId: 'stt_model',
      value: cfg(st).stt_model || '', placeholder: 'whisper-1', save: (v) => save({ stt_model: v }) }));
    const hn = healthNote(healthOf(st, 'stt_server'));
    if (hn) rows[rows.length - 2].append(hn);
  }
  if (cur === 'rig') rows.push(...rigBlock(H, st, save, ctx));
  else rows.push(rigOffer(H, st, save, ctx));
  return H.group('drive.group_stt', rows, { id: 'drive-group-stt', hint: t('drive.group_stt_hint') });
}

/** With the rig not chosen, a one-line way to set it up (address + token) without leaving the page. */
function rigOffer(H, st, save, ctx) {
  const d = el('details', { class: 'drive-rig-details', id: 'drive-rig-details' }, [
    el('summary', { text: t('drive.rig_setup') }),
    el('div', { class: 'dev-group-body' }, rigBlock(H, st, save, ctx)),
  ]);
  if ((st.rig && (st.rig.url_set || st.rig.enabled))) d.open = true;
  return d;
}

// ------------------------------------------------------- 2 text to speech --

export function ttsSection(H, st, save, ctx) {
  const rows = [];
  const cur = selectedProvider(st, 'tts');
  const sel = H.selectInput(t('drive.provider'), providerOptions(st, 'tts'), cur, async (v) => {
    const patch = v === 'rig' ? { rig: { enabled: true } }
      : { tts_provider: v, ...((st.rig && st.rig.enabled) ? { rig: { enabled: false } } : {}) };
    if (!(await save(patch))) sel.value = cur;
    else ctx.rebuild && ctx.rebuild();
  });
  sel.id = 'drive-tts-provider';
  const tprov = H.row(t('drive.provider'), t('drive.provider_hint_tts'), [sel], { id: 'drive-tts-provider-row' });
  const thn = healthNote(healthOf(st, 'tts_provider'));
  if (thn) tprov.append(thn);
  rows.push(guard(st, 'tts_provider', tprov, [sel]));
  if (cur === 'cpu') {
    const engines = ['auto', 'chatterbox', 'minimax', ...((st.tts_engines || []).filter((e) => !['auto', 'chatterbox', 'minimax'].includes(e)))];
    const labelOf = (e) => (has(`drive.tts_${e}`) ? t(`drive.tts_${e}`) : e);
    const eng = H.selectInput(t('drive.tts_engine'), engines.map((e) => ({ value: e, label: labelOf(e) })), st.tts || 'auto', async (v) => {
      if (!(await save({ tts_engine: v }))) eng.value = st.tts || 'auto';
    });
    eng.id = 'drive-tts-engine';
    rows.push(guard(st, 'tts_engine', H.row(t('drive.tts_engine'), t('drive.tts_engine_hint'), [eng], { id: 'drive-tts-engine-row' }), [eng]));
  }
  if (cur === 'openai') {
    rows.push(savedField(H, st, { id: 'drive-tts-url', label: t('drive.server_url'), hint: t('drive.hq_url_hint'),
      settingId: 'hq_url', value: cfg(st).hq_url || '', placeholder: 'http://', save: (v) => save({ hq_url: v }) }));
    rows.push(savedField(H, st, { id: 'drive-tts-model', label: t('drive.server_model'), hint: '', settingId: 'hq_model',
      value: cfg(st).hq_model || '', placeholder: 'tts-1', save: (v) => save({ hq_model: v }) }));
    const hn = healthNote(healthOf(st, 'tts_server'));
    if (hn) rows[rows.length - 2].append(hn);
  }
  if (cur === 'rig') rows.push(el('p', { class: 'dev-row-hint drive-rig-shared', text: t('drive.rig_shared_tts') }));
  rows.push(...(ctx.voiceRows || []));
  rows.push(pluginsRow(H, st, save, ctx));
  return H.group('drive.group_tts', rows, { id: 'drive-group-tts', hint: t('drive.group_tts_hint') });
}

/** Provider plugins found on the server: switch each on or off (applies after a restart). */
export function pluginsRow(H, st, save, ctx) {
  const list = (st.providers && st.providers.plugins) || [];
  if (!ctx.extensions) ctx.extensions = [...(cfg(st).extensions || [])];
  const enabled = new Set(ctx.extensions);
  const box = el('div', { class: 'drive-plugins', id: 'drive-plugins' });
  if (!list.length) {
    box.append(el('p', { class: 'dev-row-hint', text: t('drive.plugin_none') }));
  }
  const note = el('p', { class: 'dev-row-hint drive-plugins-restart', text: t('drive.plugins_restart'), hidden: '' });
  for (const p of list) {
    const on = enabled.has(p.name) || !!p.enabled;
    const cb = H.switchInput(on, t('drive.plugin_enable', { name: p.name }), async (v) => {
      cb.disabled = true;
      const next = new Set(ctx.extensions || []);
      if (v) next.add(p.name); else next.delete(p.name);
      const ok = await save({ extensions: [...next] });
      cb.disabled = false;
      if (!ok) { cb.checked = !v; return; }
      ctx.extensions = [...next];
      note.hidden = false;
    });
    cb.id = `drive-plugin-${p.name}`;
    box.append(H.row(p.name, p.name === 'kokoro' ? t('drive.plugin_gpl') : '', [cb], { id: `drive-plugin-row-${p.name}` }));
  }
  box.append(note);
  const r = H.row(t('drive.plugins_title'), t('drive.plugins_hint'), [], { id: 'drive-plugins-row', block: true });
  r.append(box);
  return guard(st, 'extensions', r, [...box.querySelectorAll('input')]);
}

// ----------------------------------------------------------------- models --

/** Download / progress / cancel and the one-time folder move. `api` = {get, post}. */
export function modelsSection(H, st, ctx) {
  const m = (st && st.models) || {};
  const rows = [];
  const bar = el('progress', { class: 'drive-progress', max: '100', value: String(m.percent || 0), id: 'drive-models-progress',
    'aria-label': t('drive.models_title') });
  const line = el('p', { class: 'dev-row-hint drive-note', id: 'drive-models-line', role: 'status' });
  const go = saveButton(t('drive.models_download'), () => start(), 'drive-models-download');
  const stop = saveButton(t('drive.models_cancel'), () => cancel(), 'drive-models-cancel');
  let timer = null;
  const paint = (s) => {
    const running = s.state === 'running';
    bar.hidden = !running;
    bar.value = s.percent || 0;
    stop.hidden = !running;
    go.hidden = running;
    if (running) line.textContent = t('drive.models_progress', { file: s.file || '…', pct: Math.round(s.percent || 0) });
    else if (s.state === 'done') line.textContent = s.restart_required ? `${t('drive.models_done')} ${t('drive.models_restart')}` : t('drive.models_done');
    else if (s.state === 'cancelled') line.textContent = t('drive.models_cancelled');
    else if (s.state === 'failed') line.textContent = t('drive.models_failed', { error: s.error || '' });
    else line.textContent = (st.health || []).some((h) => h.id === 'models' && h.state !== 'ok') ? t('drive.models_missing') : t('drive.models_ready');
  };
  const poll = async () => {
    try {
      const s = await H.voiceJson('/api/voice/models/status');
      paint(s);
      if (s.state === 'running') timer = setTimeout(poll, 700);
      else { timer = null; if (s.state === 'done' && ctx.saved) ctx.saved(); }
    } catch { timer = null; }
  };
  async function start() {
    go.disabled = true;
    try {
      const s = await H.voiceJson('/api/voice/models/fetch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      paint(s);
      timer = setTimeout(poll, 700);
    } catch (e) {
      if (ctx.toast) ctx.toast(H.saveErrorText(e), true);
    }
    go.disabled = false;
  }
  async function cancel() {
    stop.disabled = true;
    try { paint(await H.voiceJson('/api/voice/models/cancel', { method: 'POST' })); } catch { /* finished meanwhile */ }
    stop.disabled = false;
  }
  paint(m);
  if (m.state === 'running') timer = setTimeout(poll, 400);
  const r = H.row(t('drive.models_title'), t('drive.models_hint'), [go, stop], { id: 'drive-models' });
  r.classList.add('dev-row-action');
  r.querySelector('.dev-row-text').append(line, bar);
  rows.push(r);
  const lay = m.layout || {};
  if (lay.legacy) {
    const mv = saveButton(t('drive.models_move'), async () => {
      mv.disabled = true;
      try {
        await H.voiceJson('/api/voice/models/migrate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{"dry_run":false}' });
        if (ctx.toast) ctx.toast(t('drive.models_moved'));
        if (ctx.saved) ctx.saved();
      } catch (e) { if (ctx.toast) ctx.toast(H.saveErrorText(e), true); }
      mv.disabled = false;
    }, 'drive-models-move');
    const mr = H.row(t('drive.models_legacy'), t('drive.models_legacy_hint'), [mv], { id: 'drive-models-legacy' });
    mr.classList.add('dev-row-action');
    rows.push(mr);
  }
  const sys = ((st && st.system) || []).filter((x) => !x.ok);
  for (const s of sys) {
    rows.push(el('p', { class: 'dev-row-hint drive-health drive-health-degraded', role: 'note' }, [
      el('span', { text: t('drive.system_missing', { name: s.id, use: s.required_for }) }), ' ',
      el('code', { dir: 'ltr', text: s.fix }),
    ]));
  }
  const g = H.group('drive.group_models', rows, { id: 'drive-group-models' });
  g.dispose = () => { if (timer) clearTimeout(timer); };
  return g;
}

// -------------------------------------------------- 3 input and devices --

/** The three ways to talk, over two device switches: autoListen (off = push to talk) and typedInput. */
export function inputMode(s) {
  if (s.typedInput) return 'type';
  return s.autoListen ? 'hands_free' : 'push_to_talk';
}
export function inputModePatch(mode) {
  if (mode === 'type') return { typedInput: true };
  return { typedInput: false, autoListen: mode === 'hands_free' };
}

/** Level meter + latency readout for the chosen microphone. Pure DOM + Web Audio. */
export function micTest(H, s, ctx) {
  const meter = el('meter', { class: 'drive-meter', min: '0', max: '1', value: '0', id: 'drive-mic-meter', 'aria-label': t('drive.mic_test') });
  const out = el('p', { class: 'dev-row-hint drive-note', id: 'drive-mic-out', role: 'status' });
  const lat = el('p', { class: 'dev-row-hint drive-note', id: 'drive-mic-latency' });
  const btn = saveButton(t('drive.mic_test_start'), () => (running ? stopTest() : startTest()), 'drive-mic-test');
  let running = false;
  let stream = null;
  let ac = null;
  let raf = 0;
  let peak = 0;
  async function startTest() {
    const md = globalThis.navigator && navigator.mediaDevices;
    if (!md || !md.getUserMedia) { out.textContent = t('drive.err_https'); return; }
    try {
      stream = await md.getUserMedia({ audio: s.micId ? { deviceId: { exact: s.micId } } : true });
    } catch { out.textContent = t('drive.err_mic_denied'); return; }
    const AC = globalThis.AudioContext || globalThis.webkitAudioContext;
    if (!AC) { out.textContent = t('drive.err_https'); stopTest(); return; }
    ac = new AC();
    const src = ac.createMediaStreamSource(stream);
    const an = ac.createAnalyser();
    an.fftSize = 1024;
    src.connect(an);
    const buf = new Float32Array(an.fftSize);
    running = true;
    peak = 0;
    btn.textContent = t('drive.mic_test_stop');
    const base = Number.isFinite(ac.baseLatency) ? ac.baseLatency : 0;
    const outl = Number.isFinite(ac.outputLatency) ? ac.outputLatency : 0;
    lat.textContent = t('drive.mic_latency', { ms: Math.round((base + outl) * 1000) });
    const t0 = Date.now();
    const tick = () => {
      if (!running) return;
      an.getFloatTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      const rms = Math.sqrt(sum / buf.length);
      const level = Math.min(1, rms * 6);
      meter.value = level;
      peak = Math.max(peak, level);
      out.textContent = peak > 0.08 ? t('drive.mic_test_hearing') : t('drive.mic_test_silent');
      if (Date.now() - t0 > 15000) { stopTest(); return; }   // never leave the microphone open
      raf = requestAnimationFrame ? requestAnimationFrame(tick) : setTimeout(tick, 50);
    };
    tick();
    measureServer();
  }
  function stopTest() {
    running = false;
    if (raf && globalThis.cancelAnimationFrame) cancelAnimationFrame(raf);
    if (stream) for (const tr of stream.getTracks()) tr.stop();
    if (ac && ac.close) ac.close().catch(() => {});
    stream = null; ac = null;
    meter.value = 0;
    btn.textContent = t('drive.mic_test_start');
  }
  async function measureServer() {
    const t0 = performance.now();
    try {
      await H.voiceJson('/api/voice/settings/schema');
      const ms = Math.round(performance.now() - t0);
      lat.textContent = `${lat.textContent} · ${t('drive.mic_server_rtt', { ms })}`.replace(/^ · /, '');
    } catch { /* the readout stays audio-only */ }
  }
  const r = H.row(t('drive.mic_test'), t('drive.mic_test_hint'), [btn, meter], { id: 'drive-mic-test-row' });
  r.classList.add('dev-row-action');
  r.querySelector('.dev-row-text').append(out, lat);
  r.dispose = stopTest;
  return r;
}

export function inputSection(H, st, s, set, ctx) {
  const rows = [];
  const modes = [{ value: 'hands_free', label: t('drive.mode_hands_free') }, { value: 'push_to_talk', label: t('drive.mode_push_to_talk') },
    { value: 'type', label: t('drive.mode_type') }];
  rows.push(H.row(t('drive.input_mode'), t('drive.input_mode_hint'),
    [H.segmented(t('drive.input_mode'), modes, inputMode(s), (v) => set(inputModePatch(v)))], { block: true, id: 'drive-input-mode' }));
  rows.push(...(ctx.listeningRows || []));
  const test = micTest(H, s, ctx);
  rows.push(test);
  const g = H.group('drive.group_input', rows, { id: 'drive-group-input', hint: t('drive.group_input_hint') });
  g.dispose = test.dispose;
  return g;
}

// ------------------------------------------------ 4 voices per character --

/** Immediate save of a bot's voice / role-play mode through DisPatch's bot order route. */
export async function setBotFields(H, botId, fields) {
  const r = await H.voiceJson('/api/bots/all');
  const bots = Array.isArray(r) ? r : (r && r.bots) || [];
  if (!bots.some((b) => b && b.id === botId)) throw new Error('unknown bot');
  const payload = bots.map((b, i) => (b.id === botId ? { id: b.id, order: i, ...fields } : { id: b.id, order: i }));
  const out = await H.voiceJson('/api/bots/order', { method: 'PUT', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bots: payload }) });
  return (out && out.bots) || [];
}

export function characterRow(H, st, bot, ctx) {
  const per = (st.bots && st.bots[bot.id]) || {};
  const id = `drive-char-${bot.id}`;
  const notify = (m) => ctx.toast && ctx.toast(m, true);
  let voice = typeof bot.voice === 'string' ? bot.voice : '';
  let rp = typeof bot.voice_rp === 'string' ? bot.voice_rp : '';
  const saveField = async (fields, revert) => {
    try {
      const bots = await setBotFields(H, bot.id, fields);
      if (bots.length && ctx.setRoster) ctx.setRoster(bots);
      return true;
    } catch (e) { revert(); notify(H.saveErrorText(e)); return false; }
  };
  const voiceSel = H.selectInput(t('drive.voice'), H.voiceOptions(st, { withDefault: true, current: voice }), voice, async (v) => {
    const prev = voice;
    if (await saveField({ voice: v }, () => { voiceSel.value = prev; })) voice = v;
  });
  const rpSel = H.selectInput(t('drive.rp_speech'), H.rpOptions({ withServerDefault: true }), rp, async (v) => {
    const prev = rp;
    if (await saveField({ voice_rp: v }, () => { rpSel.value = prev; })) rp = v;
  });
  const flag = async (key, cb, v) => {
    cb.disabled = true;
    try { await H.patchVoiceSettings({ bots: { [bot.id]: { [key]: v } } }); per[key] = v; if (ctx.saved) ctx.saved(); }
    catch (e) { cb.checked = !v; notify(H.saveErrorText(e)); }
    cb.disabled = false;
  };
  const bc = H.switchInput(per.backchannel === true, t('drive.bot_backchannel'), (v) => flag('backchannel', bc, v));
  const fu = H.switchInput(per.follow_up !== false, t('drive.bot_follow_up'), (v) => flag('follow_up', fu, v));
  let locked = typeof per.locked_voice === 'string' ? per.locked_voice : '';
  const lockedSel = H.selectInput(t('drive.bot_locked_voice'),
    [{ value: '', label: t('drive.bot_locked_voice_same') }, ...H.voiceOptions(st, { current: locked })], locked, async (v) => {
      lockedSel.disabled = true;
      try { await H.patchVoiceSettings({ bots: { [bot.id]: { locked_voice: v } } }); locked = v; per.locked_voice = v; }
      catch (e) { lockedSel.value = locked; notify(H.saveErrorText(e)); }
      lockedSel.disabled = false;
    });
  const body = el('div', { class: 'dev-group-body drive-char-body' }, [
    H.row(t('drive.voice'), '', [voiceSel, H.previewButton(() => voiceSel.value || st.default_voice || '', notify)], { id: `${id}-voice` }),
    H.row(t('drive.rp_speech'), '', [rpSel], { id: `${id}-rp` }),
    H.row(t('drive.bot_locked_voice'), t('drive.bot_locked_voice_hint'),
      [lockedSel, H.previewButton(() => lockedSel.value || voiceSel.value || st.default_voice || '', notify)], { id: `${id}-locked` }),
    H.row(t('drive.bot_backchannel'), t('drive.bot_backchannel_hint'), [bc], { id: `${id}-bc` }),
    H.row(t('drive.bot_follow_up'), t('drive.bot_follow_up_hint'), [fu], { id: `${id}-fu` }),
  ]);
  return el('details', { class: 'drive-char', id }, [
    el('summary', { class: 'drive-char-title', text: bot.name || bot.id }), body,
  ]);
}

export function charactersSection(H, st, bots, ctx) {
  const list = (bots || []).filter((b) => b && b.visible !== false && b.id);
  const rows = list.length ? list.map((b) => characterRow(H, st, b, ctx))
    : [el('p', { class: 'dev-row-hint', text: t('drive.character_none') })];
  const g = H.group('drive.group_characters', rows, { id: 'drive-group-characters', hint: t('drive.group_characters_hint') });
  return g;
}

/** Open (and scroll to) one character in the section: the Bots tab's pill lands here. */
export function focusCharacter(root, botId) {
  if (!root) return false;
  const d = root.querySelector(`[id="drive-char-${botId}"]`);
  if (!d) return false;
  d.open = true;
  try { d.scrollIntoView({ block: 'center' }); } catch { /* jsdom */ }
  const sel = d.querySelector('select');
  if (sel && sel.focus) { try { sel.focus({ preventScroll: true }); } catch { sel.focus(); } }
  return true;
}

export { listFormat };
