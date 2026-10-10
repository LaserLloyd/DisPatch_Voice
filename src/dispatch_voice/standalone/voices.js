// Voices panel. Served next to its API: <prefix>/ui loads this, the API lives at <prefix>.
(() => {
  'use strict';
  const API = location.pathname.replace(/\/ui\/?$/, '');
  const $ = (id) => document.getElementById(id);
  let clip = null;          // {blob, name}
  let analysis = null;
  let recorder = null;
  let recChunks = [];
  let recTimer = null;

  async function api(path, opts = {}) {
    const res = await fetch(API + path, { credentials: 'same-origin', ...opts });
    if (!res.ok) {
      let msg = res.statusText;
      try { const j = await res.json(); msg = j.detail || msg; } catch (_) { /* not json */ }
      throw new Error(msg);
    }
    const type = res.headers.get('content-type') || '';
    return type.includes('application/json') ? res.json() : res.blob();
  }

  function status(el, text, isErr) {
    el.textContent = text || '';
    el.classList.toggle('err', !!isErr);
  }

  function chips(container, checks) {
    container.replaceChildren(...checks.map((c) => {
      const s = document.createElement('span');
      s.className = 'chip ' + c.level;
      s.textContent = c.message;
      return s;
    }));
  }

  // ---------------------------------------------------------------- tabs
  function showTab(which) {
    $('tab-upload').setAttribute('aria-pressed', String(which === 'upload'));
    $('tab-record').setAttribute('aria-pressed', String(which === 'record'));
    $('pane-upload').hidden = which !== 'upload';
    $('pane-record').hidden = which !== 'record';
  }
  $('tab-upload').addEventListener('click', () => showTab('upload'));
  $('tab-record').addEventListener('click', () => showTab('record'));

  // -------------------------------------------------------------- upload
  $('pick').addEventListener('click', () => $('file').click());
  $('file').addEventListener('change', (e) => { if (e.target.files[0]) useClip(e.target.files[0], e.target.files[0].name); });
  const drop = $('drop');
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault(); drop.classList.remove('over');
    const f = e.dataTransfer.files[0];
    if (f) useClip(f, f.name);
  });

  // -------------------------------------------------------------- record
  $('rec-start').addEventListener('click', async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
      });
      recChunks = [];
      recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => { if (e.data.size) recChunks.push(e.data); };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        clearInterval(recTimer);
        $('rec-state').textContent = '';
        const blob = new Blob(recChunks, { type: recorder.mimeType || 'audio/webm' });
        useClip(blob, 'recording.' + ((recorder.mimeType || '').includes('ogg') ? 'ogg' : 'webm'));
      };
      recorder.start();
      const t0 = Date.now();
      $('rec-state').innerHTML = '<span class="rec-dot"></span>0 s';
      recTimer = setInterval(() => {
        const s = Math.round((Date.now() - t0) / 1000);
        $('rec-state').innerHTML = '<span class="rec-dot"></span>' + s + ' s';
        if (s >= 40) recorder.stop();
      }, 500);
      $('rec-start').disabled = true; $('rec-stop').disabled = false;
    } catch (err) {
      status($('add-status'), 'Microphone not available: ' + err.message, true);
    }
  });
  $('rec-stop').addEventListener('click', () => {
    if (recorder && recorder.state === 'recording') recorder.stop();
    $('rec-start').disabled = false; $('rec-stop').disabled = true;
  });

  // -------------------------------------------------------------- review
  async function useClip(blob, name) {
    clip = { blob, name };
    $('clip-audio').src = URL.createObjectURL(blob);
    $('review').hidden = false;
    $('build').disabled = true;
    status($('add-status'), 'Checking the clip…');
    const fd = new FormData(); fd.append('file', blob, name);
    try {
      analysis = await api('/analyse', { method: 'POST', body: fd });
      chips($('chips'), analysis.checks);
      status($('add-status'), analysis.ok ? 'Looks usable. Name it, confirm whose voice it is, then build.' : 'This clip can\'t be used. See the red notes above.', !analysis.ok);
    } catch (err) {
      analysis = null;
      $('chips').replaceChildren();
      status($('add-status'), err.message, true);
    }
    updateBuild();
  }

  function consentValue() {
    const c = document.querySelector('input[name=consent]:checked');
    return c ? c.value : '';
  }

  function updateBuild() {
    $('build').disabled = !(clip && analysis && analysis.ok && $('name').value.trim() && consentValue());
  }
  $('name').addEventListener('input', updateBuild);

  $('discard').addEventListener('click', () => {
    clip = null; analysis = null;
    $('review').hidden = true; $('file').value = '';
    status($('add-status'), '');
  });

  $('build').addEventListener('click', async () => {
    $('build').disabled = true;
    status($('add-status'), 'Building the voice. This takes a few seconds on this computer…');
    const fd = new FormData();
    fd.append('file', clip.blob, clip.name);
    fd.append('name', $('name').value.trim());
    fd.append('consent', consentValue());
    fd.append('consent_note', $('note').value.trim());
    try {
      const p = await api('', { method: 'POST', body: fd });
      status($('add-status'), 'Built "' + p.name + '" in ' + p.timings.build_s + ' s. Listen to its test line below.');
      $('review').hidden = true; clip = null; analysis = null; $('name').value = ''; $('note').value = '';
      await refresh();
    } catch (err) {
      status($('add-status'), err.message, true);
      updateBuild();
    }
  });

  // ---------------------------------------------------------------- list
  function play(card, url) {
    const a = card.querySelector('.v-audio');
    a.hidden = false; a.src = url; a.play().catch(() => {});
  }

  function renderCard(v) {
    const card = document.getElementById('card-tpl').content.firstElementChild.cloneNode(true);
    card.dataset.id = v.id;
    card.querySelector('.v-name').textContent = v.name;
    card.querySelector('.v-default').hidden = !v.default;
    card.querySelector('.v-consent').textContent = 'Consent: ' + v.consent.statement + (v.consent.note ? ' — ' + v.consent.note : '');
    const m = v.quality && v.quality.match;
    if (m != null) {
      const pct = Math.max(0, Math.min(100, Math.round(m * 100)));
      card.querySelector('.v-match').innerHTML = 'Match <span class="meter"><i style="width:' + pct + '%"></i></span> ' + pct + '%';
    }
    chips(card.querySelector('.v-chips'), (v.quality && v.quality.checks || []).filter((c) => c.level !== 'ok'));
    card.querySelector('.v-fillers').textContent = v.fillers === 'ready' ? 'Reaction sounds ready.' : 'Rendering reaction sounds in the background…';
    const base = '/' + encodeURIComponent(v.id);
    card.querySelector('.v-play-ref').onclick = () => play(card, API + base + '/files/reference.wav');
    card.querySelector('.v-play-aud').onclick = () => play(card, API + base + '/files/audition.wav');
    card.querySelector('.v-make-default').hidden = !!v.default;
    card.querySelector('.v-make-default').onclick = async () => {
      await api(base, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ default: true }) });
      refresh();
    };
    card.querySelector('.v-rename').onclick = async () => {
      const n = prompt('New name', v.name);
      if (n && n.trim()) {
        await api(base, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: n.trim() }) });
        refresh();
      }
    };
    card.querySelector('.v-delete').onclick = async () => {
      if (confirm('Delete "' + v.name + '"? This removes the clip and the voice from this computer.')) {
        await api(base, { method: 'DELETE' });
        refresh();
      }
    };
    const sayBtn = card.querySelector('.v-say');
    sayBtn.onclick = async () => {
      const text = card.querySelector('.v-say-text').value.trim();
      if (!text) return;
      sayBtn.disabled = true; sayBtn.textContent = 'Generating…';
      try {
        const blob = await api(base + '/say', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
        play(card, URL.createObjectURL(blob));
      } catch (err) {
        alert(err.message);
      } finally {
        sayBtn.disabled = false; sayBtn.textContent = 'Say it';
      }
    };
    return card;
  }

  let pollTimer = null;
  let shown = '';           // signature of what the cards currently show
  const signature = (vs) => JSON.stringify(vs.map((v) => [v.id, v.name, v.default]));

  async function refresh() {
    try {
      const data = await api('');
      const list = $('list');
      const sig = signature(data.voices);
      if (!data.voices.length) {
        list.innerHTML = '<p class="muted">No voices yet. Add one above.</p>';
      } else if (sig === shown) {
        // Only background state changed: update in place so playing audio and
        // an in-flight "Say it" aren't wiped by the poll.
        data.voices.forEach((v) => {
          const card = list.querySelector('[data-id="' + CSS.escape(v.id) + '"]');
          if (card) card.querySelector('.v-fillers').textContent = v.fillers === 'ready' ? 'Reaction sounds ready.' : 'Rendering reaction sounds in the background…';
        });
      } else {
        list.replaceChildren(...data.voices.map(renderCard));
      }
      shown = data.voices.length ? sig : '';
      $('engine-line').textContent = 'Engine: Chatterbox-Turbo (MIT, runs on this computer, ' + data.engine + ').' +
        (data.watermarking ? ' Output carries an inaudible AI watermark.' : ' Watermarking is off (install the watermark extra to enable it).');
      clearTimeout(pollTimer);
      if (data.voices.some((v) => v.fillers !== 'ready')) pollTimer = setTimeout(refresh, 4000);
    } catch (err) {
      $('list').innerHTML = '';
      status($('list'), 'Could not load voices: ' + err.message, true);
    }
  }

  async function init() {
    const kinds = await api('/consent-kinds');
    $('consent').replaceChildren(...Object.entries(kinds).map(([k, text]) => {
      const l = document.createElement('label');
      const r = document.createElement('input');
      r.type = 'radio'; r.name = 'consent'; r.value = k; r.addEventListener('change', updateBuild);
      l.append(r, document.createTextNode(text));
      return l;
    }));
    refresh();
  }
  init();
})();
