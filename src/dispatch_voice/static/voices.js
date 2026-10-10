// The Voice cloning tool: the cloned-voice manager.
// Owner: track T7. Contract: C-§D (`/api/voice/profiles`, moved from
// /api/voices in v3 with no alias).
//
// It lives in the opt-in Voice cloning tool (voice-tools.js mounts it with
// `embedded: true`); Settings → Voice links to it with "Manage voices".
// Without `embedded` it still takes over a pane in place, with its own
// ‹ back row. Same JSON API and rules as dispatch-voice's standalone page: a
// voice is built on THIS computer from a 10–20 s clip, and whose voice it is
// (consent) is a required choice.
//
//   GET    /api/voice/profiles                → {voices, default, engine, watermarking}
//   POST   /api/voice/profiles/analyse        multipart file → quality report
//   POST   /api/voice/profiles                multipart file,name,consent,consent_note
//   PATCH  /api/voice/profiles/{id}           {default: true} | {name}
//   DELETE /api/voice/profiles/{id}
//   GET    /api/voice/profiles/{id}/files/audition.wav
//
// Built-in voices (from an optional plugin, which names itself) are listed
// too, with ▶ and "Set as default" through /api/voice/settings, so one screen
// answers "what will I hear". The stock voice ships with the add-on: it
// carries a "Stock" badge and can't be renamed or deleted here.
//
// Rows are the Settings .dev-group/.dev-row markup; nothing here has a look
// of its own beyond the quality chips and the recording script box.

const Q = new URL(import.meta.url).search;
const { el, railIcon, iconPaths, t, metaList } = await import(`./shell.js${Q}`);
const {
  voiceJson, voiceStatus, patchVoiceSettings, previewButton, selectInput, row, MIC_ICON, PLAY_ICON,
  builtinValue, PROFILES_API,
} = await import(`./drive-settings.js${Q}`);

const API = PROFILES_API;
const RECORD_LIMIT_S = 40;

/** Quality checks → chip level classes (server levels ok | warn | fail). */
export function chipClass(level) {
  return level === 'fail' ? 'drive-chip fail' : level === 'warn' ? 'drive-chip warn' : 'drive-chip ok';
}

/** Can this clip be built? Mirrors the server's own refusal rules. */
export function canBuild({ clip, analysis, name, consent }) {
  return !!(clip && analysis && analysis.ok && String(name || '').trim() && consent);
}

/** One line under a voice's name: who it is, when it was added, how long the clip. */
export function voiceMetaLine(v, fmtDate) {
  const parts = [];
  const kind = v && v.consent && v.consent.kind;
  if (kind === 'self') parts.push(t('voices.kind_self'));
  else if (kind === 'permission') parts.push(t('voices.kind_permission'));
  else if (kind === 'synthetic') parts.push(t('voices.kind_synthetic'));
  if (v && v.created) parts.push(t('voices.added', { date: fmtDate(v.created * 1000) }));
  if (v && v.reference_s) parts.push(t('voices.clip_len', { s: Math.round(v.reference_s) }));
  return metaList(parts);
}

/** {name, licence} of the built-in voices plugin the server suggests. */
function builtinSuggestion(st) {
  const sug = (st && st.plugins && st.plugins.builtin && st.plugins.builtin.suggested) || {};
  return { name: sug.name || '', licence: sug.licence || '' };
}

/** Is `v` dispatch-voice's stock voice? (`stockId` from /api/voice/status.) */
export function isStock(v, stockId) {
  return !!(v && stockId && v.id === stockId);
}

/**
 * Mount the manager. ctx: { toast, confirm(msg, opts), prompt(title, value),
 * onChanged() — repaint whatever shows the voice status, embedded }.
 *
 * embedded: false — `pane` is a settings pane; the manager takes the place of
 *   its contents until ‹ Voice is pressed.
 * embedded: true — `pane` is the manager's own (empty) host, e.g. the Voice
 *   cloning tool's body. No back row or title (the tool's header has them).
 */
export function createVoicesPane(pane, ctx) {
  let host = null;
  let pollTimer = null;
  let recorder = null;
  let recTimer = null;
  let clip = null;
  let analysis = null;
  let audio = null;

  const fmtDate = (ms) => {
    try { return new Intl.DateTimeFormat(document.documentElement.lang || undefined, { day: 'numeric', month: 'short' }).format(new Date(ms)); }
    catch { return new Date(ms).toDateString(); }
  };

  function stopAudio() {
    if (audio) { try { audio.pause(); } catch { /* gone */ } audio = null; }
  }
  function play(url) {
    stopAudio();
    audio = new Audio(url);
    audio.play().catch(() => ctx.toast(t('drive.err_preview'), true));
  }

  function open() {
    if (host) return;
    if (!ctx.embedded) for (const c of pane.children) { c.dataset.voicesHidden = c.hidden ? '1' : '0'; c.hidden = true; }
    host = el('div', { class: 'voices-pane', id: 'voices-pane' });
    pane.append(host);
    pane.scrollTop = 0;
    render();
  }

  function close() {
    if (!host) return;
    clearTimeout(pollTimer); pollTimer = null;
    stopRecording(true);
    stopAudio();
    host.remove(); host = null;
    for (const c of pane.children) { if (c.dataset.voicesHidden != null) { c.hidden = c.dataset.voicesHidden === '1'; delete c.dataset.voicesHidden; } }
    clip = null; analysis = null;
    if (ctx.embedded) return;
    ctx.onChanged && ctx.onChanged();
    const btn = document.getElementById('dev-voices-open');
    if (btn) { try { btn.focus(); btn.scrollIntoView({ block: 'center' }); } catch { /* ignore */ } }
  }

  async function render() {
    if (!host) return;
    const back = el('button', { type: 'button', class: 'voices-back' }, [
      railIcon(iconPaths('back')), document.createTextNode(` ${t('voices.back')}`),
    ]);
    back.addEventListener('click', close);
    const title = el('h3', { class: 'voices-title', text: t('voices.title') });
    const listBody = el('div', { class: 'dev-group-body', id: 'voices-list' }, [
      el('div', { class: 'dev-row' }, [el('span', { class: 'dev-row-text' }, [el('span', { class: 'dev-row-hint', text: t('drive.loading') })])]),
    ]);
    const builtinBody = el('div', { class: 'dev-group-body', id: 'voices-builtin' });
    const addBody = el('div', { class: 'dev-group-body', id: 'voices-add' });
    const engine = el('p', { class: 'dev-group-hint voices-engine' });
    host.replaceChildren(...[
      ctx.embedded ? null : el('div', { class: 'voices-top' }, [back, title]),
      el('p', { class: 'dev-group-hint voices-intro', text: t('voices.intro') }),
      el('section', { class: 'dev-group' }, [el('h3', { class: 'dev-group-title', text: t('voices.yours') }), listBody]),
      el('section', { class: 'dev-group' }, [el('h3', { class: 'dev-group-title', text: t('voices.builtin') }), builtinBody]),
      el('section', { class: 'dev-group' }, [el('h3', { class: 'dev-group-title', text: t('voices.add') }), addBody]),
      engine,
    ].filter(Boolean));
    renderAdd(addBody);
    if (!ctx.embedded) back.focus();
    await refresh();
  }

  async function refresh() {
    if (!host) return;
    const listBody = host.querySelector('#voices-list');
    const builtinBody = host.querySelector('#voices-builtin');
    let data; let st = null;
    try {
      [data, st] = await Promise.all([voiceJson(API), voiceStatus().catch(() => null)]);
    } catch (e) {
      listBody.replaceChildren(el('div', { class: 'dev-row' }, [el('span', { class: 'dev-row-text' }, [
        el('span', { class: 'dev-row-hint voices-err', text: t('voices.load_failed', { error: e.message }) }),
      ])]));
      return;
    }
    if (!host) return;
    const defaultVoice = (st && st.default_voice) || (data.default ? data.default : '');
    const stockId = (st && st.stock_voice) || null;
    const voices = data.voices || [];
    // The stock voice last: it is the fallback, not one of "yours".
    const ordered = voices.filter((v) => !isStock(v, stockId)).concat(voices.filter((v) => isStock(v, stockId)));
    if (!ordered.some((v) => !isStock(v, stockId))) {
      const builtin = !!(st && (st.builtin_voices || []).length);
      ordered.unshift(null);
      listBody.replaceChildren(...ordered.map((v) => (v ? voiceRow(v, defaultVoice, stockId)
        : el('div', { class: 'dev-row' }, [el('span', { class: 'dev-row-text' }, [
          el('strong', { text: t('voices.none') }),
          el('span', { class: 'dev-row-hint', text: builtin ? t('voices.none_hint') : t('voices.none_hint_stock') }),
        ])]))));
    } else {
      listBody.replaceChildren(...ordered.map((v) => voiceRow(v, defaultVoice, stockId)));
    }
    renderBuiltin(builtinBody, st, defaultVoice);
    const eng = host.querySelector('.voices-engine');
    if (eng) {
      eng.textContent = data.engine
        ? (data.watermarking ? t('voices.engine_wm', { engine: data.engine }) : t('voices.engine', { engine: data.engine }))
        : t('voices.engine_off');
    }
    clearTimeout(pollTimer);
  }

  function voiceRow(v, defaultVoice, stockId) {
    const isDefault = defaultVoice ? defaultVoice === v.id : !!v.default;
    const stock = isStock(v, stockId);
    const base = `${API}/${encodeURIComponent(v.id)}`;
    const text = el('span', { class: 'dev-row-text' }, [
      el('strong', { dir: 'auto' }, [
        document.createTextNode(stock ? t('drive.voice_stock') : v.name),
        stock ? el('span', { class: 'pin-toggle voices-stock', text: t('voices.stock_badge') }) : null,
        isDefault ? el('span', { class: 'pin-toggle pinned voices-default', text: t('voices.default_badge') }) : null,
      ]),
      el('span', { class: 'dev-row-hint', text: stock ? t('voices.stock_hint') : voiceMetaLine(v, fmtDate) }),
    ]);
    const test = el('button', { type: 'button', class: 'btn-secondary' }, [railIcon(PLAY_ICON), document.createTextNode(` ${t('voices.test')}`)]);
    test.addEventListener('click', () => play(`${base}/files/audition.wav`));
    const actions = el('div', { class: 'voices-actions' }, [test]);
    if (!isDefault) {
      const def = el('button', { type: 'button', class: 'btn-secondary', text: t('voices.set_default') });
      def.addEventListener('click', () => setDefault(v.id, true));
      actions.append(def);
    }
    const ren = el('button', { type: 'button', class: 'btn-secondary', text: t('voices.rename') });
    ren.addEventListener('click', async () => {
      const n = await ctx.prompt(t('voices.rename_title'), v.name);
      if (n && n.trim() && n.trim() !== v.name) {
        try {
          await voiceJson(base, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: n.trim() }) });
          refresh();
        } catch (e) { ctx.toast(e.message, true); }
      }
    });
    const del = el('button', { type: 'button', class: 'btn-secondary voices-delete', text: t('voices.delete') });
    del.addEventListener('click', async () => {
      if (!await ctx.confirm(t('voices.delete_confirm', { name: v.name }), { danger: true, okText: t('voices.delete') })) return;
      try { await voiceJson(base, { method: 'DELETE' }); refresh(); ctx.onChanged && ctx.onChanged(); }
      catch (e) { ctx.toast(e.message, true); }
    });
    // The stock voice ships with dispatch-voice: renaming it would not stick
    // and deleting it only makes the package build it again.
    if (!stock) actions.append(ren, del);
    return el('div', { class: 'dev-row dev-row-block voices-row' + (stock ? ' voices-row-stock' : ''), dataset: { id: v.id } }, [text, actions]);
  }

  async function setDefault(value, isProfile) {
    try {
      try {
        await patchVoiceSettings({ default_voice: value });
      } catch (e) {
        // A server without /api/voice/settings yet: the profile store's own default.
        if (isProfile && e.status === 404) {
          await voiceJson(`${API}/${encodeURIComponent(value)}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ default: true }) });
        } else throw e;
      }
      ctx.toast(t('voices.default_set'));
      refresh();
      ctx.onChanged && ctx.onChanged();
    } catch (e) { ctx.toast(t('drive.err_save') + ' ' + e.message, true); }
  }

  function renderBuiltin(body, st, defaultVoice) {
    const list = (st && st.builtin_voices) || [];
    if (!list.length) {
      body.replaceChildren(el('div', { class: 'dev-row' }, [el('span', { class: 'dev-row-text' }, [
        el('span', { class: 'dev-row-hint', text: t('voices.builtin_plugin', builtinSuggestion(st)) }),
      ])]));
      return;
    }
    const opts = list.map((b) => ({ value: builtinValue(b.id), label: b.name || b.id }));
    const cur = defaultVoice && defaultVoice.startsWith('builtin:') ? defaultVoice : opts[0].value;
    const sel = selectInput(t('voices.builtin'), opts, cur, () => paintDef());
    const def = el('button', { type: 'button', class: 'btn-secondary', text: t('voices.set_default') });
    def.addEventListener('click', () => setDefault(sel.value, false));
    const r = row(t('voices.builtin_pick'), '', [sel, previewButton(() => sel.value, (m) => ctx.toast(m, true)), def]);
    r.classList.add('voices-builtin-row');
    const hint = el('span', { class: 'dev-row-hint' });
    r.querySelector('.dev-row-text').append(hint);
    function paintDef() {
      const on = sel.value === defaultVoice;
      def.hidden = on;
      hint.textContent = on ? t('voices.builtin_is_default') : t('voices.builtin_hint');
    }
    paintDef();
    body.replaceChildren(r);
  }

  // ---- add a voice -------------------------------------------------------
  function renderAdd(body) {
    const status = el('p', { class: 'voices-status', role: 'status', 'aria-live': 'polite' });
    const recBtn = el('button', { type: 'button', class: 'btn-secondary' }, [railIcon(MIC_ICON), document.createTextNode(` ${t('voices.record')}`)]);
    const upBtn = el('button', { type: 'button', class: 'btn-secondary' }, [railIcon(iconPaths('upload')), document.createTextNode(` ${t('voices.upload')}`)]);
    const file = el('input', { type: 'file', accept: 'audio/*,.wav,.mp3,.m4a,.ogg,.opus,.webm,.flac,.aac,.mp4', hidden: '' });
    const start = el('div', { class: 'dev-row dev-row-block' }, [
      el('span', { class: 'dev-row-text' }, [
        el('strong', { text: t('voices.add_title') }),
        el('span', { class: 'dev-row-hint', text: t('voices.add_hint') }),
      ]),
      el('div', { class: 'voices-actions' }, [recBtn, upBtn, file]),
    ]);
    const recPane = el('div', { class: 'dev-row dev-row-block voices-rec', hidden: '' });
    const review = el('div', { class: 'dev-row dev-row-block voices-review', hidden: '' });
    body.replaceChildren(start, recPane, review, el('div', { class: 'dev-row dev-row-block voices-status-row' }, [status]));

    const say = (msg, isErr = false) => { status.textContent = msg || ''; status.classList.toggle('err', !!isErr); };

    upBtn.addEventListener('click', () => file.click());
    file.addEventListener('change', () => { const f = file.files && file.files[0]; if (f) useClip(f, f.name); });

    recBtn.addEventListener('click', () => {
      recPane.hidden = false; review.hidden = true;
      const timer = el('span', { class: 'voices-rec-time', 'aria-live': 'off' });
      const go = el('button', { type: 'button', class: 'btn-primary', text: t('voices.rec_start') });
      const halt = el('button', { type: 'button', class: 'btn-secondary', text: t('voices.rec_stop'), disabled: '' });
      recPane.replaceChildren(
        el('span', { class: 'dev-row-text' }, [
          el('strong', { text: t('voices.rec_title') }),
          el('span', { class: 'dev-row-hint', text: t('voices.rec_hint') }),
        ]),
        el('blockquote', { class: 'voices-script', text: t('voices.script') }),
        el('div', { class: 'voices-actions' }, [go, halt, timer]),
      );
      go.addEventListener('click', async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
          });
          const chunks = [];
          recorder = new MediaRecorder(stream);
          recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
          recorder.onstop = () => {
            stream.getTracks().forEach((tr) => tr.stop());
            clearInterval(recTimer); recTimer = null;
            go.disabled = false; halt.disabled = true;
            timer.textContent = '';
            const discard = recorder && recorder._discard;
            const type = (recorder && recorder.mimeType) || 'audio/webm';
            recorder = null;
            if (discard) return;
            recPane.hidden = true;
            useClip(new Blob(chunks, { type }), `recording.${type.includes('ogg') ? 'ogg' : 'webm'}`);
          };
          recorder.start();
          const t0 = Date.now();
          go.disabled = true; halt.disabled = false;
          const tick = () => {
            const s = Math.round((Date.now() - t0) / 1000);
            timer.textContent = t('voices.rec_seconds', { s });
            if (s >= RECORD_LIMIT_S) stopRecording(false);
          };
          tick();
          recTimer = setInterval(tick, 500);
          halt.focus();
        } catch {
          say(t('drive.err_mic_denied'), true);
        }
      });
      halt.addEventListener('click', () => stopRecording(false));
      go.focus();
    });

    async function useClip(blob, name) {
      clip = { blob, name };
      analysis = null;
      review.hidden = false;
      const player = el('audio', { controls: '', src: URL.createObjectURL(blob), class: 'voices-clip' });
      const chips = el('div', { class: 'voices-chips' });
      const nameIn = el('input', { type: 'text', class: 'voices-name', maxlength: '60', autocomplete: 'off', placeholder: t('voices.name_ph'), 'aria-label': t('voices.name') });
      const consent = el('fieldset', { class: 'voices-consent' }, [el('legend', { text: t('voices.consent') })]);
      const note = el('input', { type: 'text', class: 'voices-name', maxlength: '300', autocomplete: 'off', placeholder: t('voices.note_ph'), 'aria-label': t('voices.note') });
      const build = el('button', { type: 'button', class: 'btn-primary', text: t('voices.build'), disabled: '' });
      const discard = el('button', { type: 'button', class: 'btn-secondary', text: t('voices.discard') });
      const kinds = [['self', 'voices.consent_self'], ['permission', 'voices.consent_permission'], ['synthetic', 'voices.consent_synthetic']];
      for (const [value, key] of kinds) {
        const r = el('input', { type: 'radio', name: 'voices-consent', value });
        r.addEventListener('change', update);
        consent.append(el('label', { class: 'voices-consent-opt' }, [r, el('span', { text: t(key) })]));
      }
      review.replaceChildren(
        el('span', { class: 'dev-row-text' }, [el('strong', { text: t('voices.review') })]),
        player, chips,
        el('label', { class: 'voices-field' }, [el('span', { text: t('voices.name') }), nameIn]),
        consent,
        el('label', { class: 'voices-field' }, [el('span', { text: t('voices.note') }), note]),
        el('div', { class: 'voices-actions' }, [build, discard]),
      );
      nameIn.addEventListener('input', update);
      function consentValue() { const c = consent.querySelector('input:checked'); return c ? c.value : ''; }
      function update() { build.disabled = !canBuild({ clip, analysis, name: nameIn.value, consent: consentValue() }); }
      discard.addEventListener('click', () => { clip = null; analysis = null; review.hidden = true; file.value = ''; say(''); });
      build.addEventListener('click', async () => {
        build.disabled = true;
        say(t('voices.building'));
        const fd = new FormData();
        fd.append('file', clip.blob, clip.name);
        fd.append('name', nameIn.value.trim());
        fd.append('consent', consentValue());
        fd.append('consent_note', note.value.trim());
        try {
          const p = await voiceJson(API, { method: 'POST', body: fd });
          say(t('voices.built', { name: p.name }));
          clip = null; analysis = null; review.hidden = true; file.value = '';
          await refresh();
          ctx.onChanged && ctx.onChanged();
        } catch (e) {
          say(e.message, true);
          update();
        }
      });
      say(t('voices.checking'));
      const fd = new FormData();
      fd.append('file', blob, name);
      try {
        analysis = await voiceJson(`${API}/analyse`, { method: 'POST', body: fd });
        chips.replaceChildren(...(analysis.checks || []).map((c) => el('span', { class: chipClass(c.level), text: c.message })));
        say(analysis.ok ? t('voices.clip_ok') : t('voices.clip_bad'), !analysis.ok);
      } catch (e) {
        analysis = null;
        chips.replaceChildren();
        say(e.message, true);
      }
      update();
      if (analysis && analysis.ok) nameIn.focus();
    }
  }

  function stopRecording(discard) {
    if (recorder && recorder.state === 'recording') {
      recorder._discard = !!discard;
      recorder.stop();
    }
    clearInterval(recTimer); recTimer = null;
  }

  return { open, close, isOpen: () => !!host };
}
