// Voice chat settings: what can be changed, and where it is stored.
// Owner: track T7. Contracts: C-§E.2 (`drive-settings.js`), C-§F.3 (the
// device blob and its migration), C-§C.3 (`start`), SPEC §7.1–§7.3.
//
// Three scopes, kept apart on purpose (docs/voice-chat/client.md, "Settings"):
//   device  one JSON blob in localStorage (`dispatch-drive`), sent to the
//           server on every WS `start` frame and re-sent when it changes;
//   server  the install's voice settings (the switch, voices, role-play
//           text, thinking), GET/PATCH /api/voice/settings — unlocked tier
//           only, like every /api/voice route;
//   bot     `bots[].voice` and `bots[].voice_rp` (DisPatch stores them, the
//           add-on validates), and the per-bot conversation defaults in
//           voice.yaml `bots.<id>` (backchannel opt-in, follow-up default).
//
// The pure half (load/migrate/normalise, the start payload, voice labels,
// setup steps) is exported for tests/js/settings.test.js. The DOM half builds
// three surfaces that must look like the rest of DisPatch: Settings → Voice
// (19 rows in 6 groups + a collapsed Status, SPEC §7.1), the voice chat
// screen's quick sheet (§7.3) and the Bots-tab voice popover (§7.4). All of
// it is the Settings modal's own .dev-group / .dev-row markup.
//
// Internal names keep the old "drive" word on purpose (the blob key, the
// i18n namespace, CSS classes): renaming them would drop every device's
// settings for no visible gain. Users only ever read "voice chat".

const Q = new URL(import.meta.url).search;
const { el, railIcon, iconPaths, t, has, listFormat, language, isMobile } = await import(`./shell.js${Q}`);
const { pickVoice, ringCursorFrom } = await import(`./bridge.js${Q}`);

export const DRIVE_KEY = 'dispatch-drive';
export const LEGACY_PATIENCE_KEY = 'dispatch-drive-patience';
/** Blob layout version: 3 = the voice chat v3 keys (F.3). Older blobs migrate. */
export const BLOB_VERSION = 3;
export const PATIENCE_PRESETS = ['quick', 'normal', 'patient'];
export const REPLY_STYLES = ['short', 'normal', 'longer'];
export const THINKING_LEVELS = ['off', 'low', 'medium'];
export const REACTIONS = ['off', 'sometimes', 'often'];
export const SLOW_LINK = ['phone', 'wait', 'text', 'always_phone'];
export const HANGUP_QUIET_MIN = [5, 15, 30, 0];
export const RP_MODES = ['all', 'skip_actions', 'dialogue', 'dialogue_narrator'];
export const SPEED_MIN = 0.8;
export const SPEED_MAX = 1.3;
/** Where DisPatch Voice is published (the install hints point here). */
export const VOICE_REPO_URL = 'https://github.com/LaserLloyd/DisPatch_Voice';

export const DRIVE_DEFAULTS = Object.freeze({
  patience: 'normal',      // how long the server waits after you stop talking
  bargeIn: true,           // talking over a reply stops it
  autoListen: true,        // false = push-to-talk: pause after every reply
  sounds: true,            // the Tones row: earcons (heard you / ready / error)
  replyStyle: 'normal',    // short | normal | longer
  followUp: null,          // null = the bot's own default; true/false = this device's choice
  screenOn: true,          // Screen Wake Lock while voice chat is open
  micId: '',               // '' = system default
  sinkId: '',              // '' = system default (setSinkId browsers only)
  captions: true,          // words while you speak + the reply, on the voice chat screen
  showTimings: false,      // Voice lab: latency numbers on the voice chat screen
  reactions: 'sometimes',  // acks (sometimes) + mm-hm for opted-in bots (often)
  slowLink: 'phone',       // when the connection is slow: phone | wait | text | always_phone
  phoneVoice: '',          // voiceURI of the phone voice ('' = pick by language)
  optimisticDuck: true,    // lower the reply the moment you start talking
  learned: null,           // {probeMs, holdMs} the server learned for this device
  learnedReset: false,     // one-shot: the Voice lab's Reset, sent as start.rt.learned_reset
  hangupQuietMin: 15,      // 5 | 15 | 30 | 0 (never)
  firstRunOpens: 0,        // the first-run hint shows while < 3
});

const BOOL_KEYS = ['bargeIn', 'autoListen', 'sounds', 'screenOn', 'captions', 'showTimings', 'optimisticDuck',
  'learnedReset'];

function clampInt(v, lo, hi) {
  const n = Math.round(Number(v));
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : null;
}

/**
 * Any stored value → a complete, valid settings object. Junk falls back per
 * key. Old blobs migrate (F.3): `acks` → `reactions`, `backchannels: true`
 * with acks on → "often", `partials` folds into `captions`, and a pre-v3
 * `followUp: true` (the old default, indistinguishable from a choice) becomes
 * null so the bot's own default applies.
 */
export function normalizeDriveSettings(raw) {
  const r = raw && typeof raw === 'object' ? raw : {};
  const out = { ...DRIVE_DEFAULTS };
  const old = r.v !== BLOB_VERSION;
  if (PATIENCE_PRESETS.includes(r.patience)) out.patience = r.patience;
  if (REPLY_STYLES.includes(r.replyStyle)) out.replyStyle = r.replyStyle;
  for (const k of BOOL_KEYS) if (typeof r[k] === 'boolean') out[k] = r[k];
  for (const k of ['micId', 'sinkId', 'phoneVoice']) {
    if (typeof r[k] === 'string' && r[k].length <= 256) out[k] = r[k];
  }
  if (typeof r.followUp === 'boolean') out.followUp = (old && r.followUp === true) ? null : r.followUp;
  if (REACTIONS.includes(r.reactions)) out.reactions = r.reactions;
  else if (REACTIONS.includes(r.acks)) {
    out.reactions = r.acks;
    if (r.backchannels === true && r.acks !== 'off') out.reactions = 'often';
  }
  if (SLOW_LINK.includes(r.slowLink)) out.slowLink = r.slowLink;
  if (HANGUP_QUIET_MIN.includes(r.hangupQuietMin)) out.hangupQuietMin = r.hangupQuietMin;
  const opens = clampInt(r.firstRunOpens, 0, 1000);
  if (opens != null) out.firstRunOpens = opens;
  const l = r.learned;
  if (l && typeof l === 'object') {
    const probeMs = clampInt(l.probeMs, 0, 10000);
    const holdMs = clampInt(l.holdMs, 0, 10000);
    if (probeMs != null && holdMs != null) out.learned = { probeMs, holdMs };
  }
  out.v = BLOB_VERSION;
  return out;
}

/**
 * Read the device settings. 2.1.0 stored only the patience preset, under its
 * own key; that value is carried into the blob once and the old key removed.
 * An old-layout blob is rewritten in the v3 layout on first read.
 */
export function loadDriveSettings(storage) {
  let parsed = null;
  try {
    const raw = storage && storage.getItem(DRIVE_KEY);
    if (raw) parsed = JSON.parse(raw);
  } catch { parsed = null; }
  if (parsed && typeof parsed === 'object') {
    const s = normalizeDriveSettings(parsed);
    if (parsed.v !== BLOB_VERSION) {
      try { storage.setItem(DRIVE_KEY, JSON.stringify(s)); } catch { /* private mode */ }
    }
    return s;
  }
  let legacy = null;
  try { legacy = storage && storage.getItem(LEGACY_PATIENCE_KEY); } catch { legacy = null; }
  const s = normalizeDriveSettings(PATIENCE_PRESETS.includes(legacy) ? { patience: legacy } : {});
  if (legacy != null) {
    try {
      storage.setItem(DRIVE_KEY, JSON.stringify(s));
      storage.removeItem(LEGACY_PATIENCE_KEY);
    } catch { /* private mode: migrate again next time */ }
  }
  return s;
}

const listeners = new Set();

/** Merge a change into the stored settings, tell every listener, return the result. */
export function saveDriveSettings(storage, patch) {
  const next = normalizeDriveSettings({ ...loadDriveSettings(storage), ...(patch || {}), v: BLOB_VERSION });
  try { storage && storage.setItem(DRIVE_KEY, JSON.stringify(next)); } catch { /* private mode */ }
  for (const fn of [...listeners]) { try { fn(next); } catch { /* a listener's problem */ } }
  return next;
}

/** Subscribe to device-setting changes (the open voice chat re-sends `start`). */
export function onDriveSettings(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/**
 * The WS `start` frame for these settings (C-§C.3). `after` is the newest
 * reply the listener heard; `afterIds` the recent ones, newest first.
 * `extra` carries the per-connection fields (`cursor`, `after_partial`).
 * `follow_up` is sent only when this device chose; otherwise the bot's own
 * default applies (absent keys keep their value on the server).
 */
export function startPayload(s, after = null, afterIds = null, extra = null) {
  const v = normalizeDriveSettings({ ...(s || {}), v: BLOB_VERSION });
  const frame = {
    type: 'start',
    patience: v.patience,
    barge_in: v.bargeIn,
    reply_style: v.replyStyle,
    acks: v.reactions,
    slow_link: v.slowLink,
    rt: {
      partials: v.captions,
      backchannels: v.reactions === 'often',
      optimistic_duck: v.optimisticDuck,
    },
  };
  // Learned endpointing (SPEC §4.3, Amendment T3s3-1): the server keeps the learned
  // pause per device and its value wins; this device's copy seeds a server that has
  // none (a restart). The Voice lab's Reset is a one-shot `learned_reset` that makes
  // the server forget too (Amendment T7a-6); drive.js clears it once it is sent.
  if (v.learnedReset) frame.rt.learned_reset = true;
  else if (v.learned) frame.rt.learned = { probe_ms: v.learned.probeMs, hold_ms: v.learned.holdMs };
  if (typeof v.followUp === 'boolean') frame.follow_up = v.followUp;
  let ids = Array.isArray(afterIds) ? afterIds.slice() : [];
  if (extra && extra.after_partial && extra.resume && typeof extra.resume === 'object') {
    // Resuming a cut reply (review-T7b H1): the session speaks only replies stored
    // AFTER `after`, so `after` is the user row that reply answers — or, unknown,
    // the newest reply heard in full — and the cut reply's ids are left out.
    const skip = new Set(Array.isArray(extra.resume.skip) ? extra.resume.skip : []);
    ids = ids.filter((id) => !skip.has(id));
    after = extra.resume.after || (after && !skip.has(after) ? after : ids[0]) || null;
  }
  if (after) frame.after = after;
  if (ids.length) frame.after_ids = ids.slice(0, 8);
  if (extra && typeof extra === 'object') {
    if (extra.cursor && typeof extra.cursor === 'object') frame.cursor = extra.cursor;
    if (extra.after_partial && typeof extra.after_partial === 'object') frame.after_partial = extra.after_partial;
  }
  return frame;
}

/**
 * The uplink offset a reconnect resumes from (C.5 #10, hardening M3): the
 * highest of what the server acknowledged and the two transcript offsets.
 * Delegates to the ring's wrap-aware version (the offsets are a u32 sample
 * clock, Amendment T8-1); the plain max is the fallback.
 */
export function cursorFrom(offsets = {}) {
  if (ringCursorFrom) return ringCursorFrom(offsets);
  const { lastRxOff = 0, lastTranscriptUpto = 0, lastEotUpto = 0 } = offsets || {};
  const ok = (x) => (Number.isFinite(x) && x >= 0 ? x : 0);
  return Math.max(ok(lastRxOff), ok(lastTranscriptUpto), ok(lastEotUpto));
}

/** Clamp a speaking speed to the server's range, on a 0.05 grid. */
export function clampSpeed(x) {
  const v = Number(x);
  if (!Number.isFinite(v)) return 1;
  return Math.round(Math.min(SPEED_MAX, Math.max(SPEED_MIN, v)) * 20) / 20;
}

// ------------------------------------------------------------ server API --

async function voiceJson(url, opts = {}) {
  const r = await fetch(url, { credentials: 'same-origin', ...opts });
  if (!r.ok) {
    let detail = r.statusText;
    try { const b = await r.json(); detail = (b && (b.detail || b.error)) || detail; } catch { /* not json */ }
    if (detail && typeof detail === 'object') {
      try { detail = JSON.stringify(detail); } catch { detail = String(detail); }
    }
    const err = new Error(String(detail || r.status));
    err.status = r.status;
    throw err;
  }
  return r.status === 204 ? null : r.json();
}

const jsonBody = (method, body) => ({
  method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});

export const voiceStatus = () => voiceJson('/api/voice/status');
export const voiceLab = () => voiceJson('/api/voice/lab');
export const voiceSettings = () => voiceJson('/api/voice/settings');
export const patchVoiceSettings = (patch) => voiceJson('/api/voice/settings', jsonBody('PATCH', patch));
/** The voice profile store (moved from /api/voices in v3, no alias — C-§D). */
export const PROFILES_API = '/api/voice/profiles';
export { voiceJson };

/**
 * Save one bot's voice now (the quick sheet's "{bot}'s voice" row). PUT
 * /api/bots/order keeps the order of the bots it is sent and appends the
 * rest, so the whole roster goes, in its current order, with `voice` set on
 * this bot only (every other field absent = unchanged). Returns the roster
 * the server answered with.
 */
export async function setBotVoice(botId, voice) {
  const r = await voiceJson('/api/bots/all');
  const bots = Array.isArray(r) ? r : (r && r.bots) || [];
  if (!bots.some((b) => b && b.id === botId)) throw new Error('unknown bot');
  const payload = bots.map((b, i) => (b.id === botId ? { id: b.id, order: i, voice: voice || '' } : { id: b.id, order: i }));
  const out = await voiceJson('/api/bots/order', jsonBody('PUT', { bots: payload }));
  return (out && out.bots) || [];
}

// ---------------------------------------------------------- voice labels --

/** "builtin:af_heart" → {kind:'builtin', id:'af_heart'}; anything else is a profile id. */
export function parseVoice(v) {
  const s = String(v || '');
  if (!s) return { kind: 'default', id: '' };
  if (s.startsWith('builtin:')) return { kind: 'builtin', id: s.slice(8) };
  return { kind: 'profile', id: s };
}

/** The select value for a built-in voice; `builtin:<id>` and a bare id mean the same. */
export function builtinValue(id) {
  const s = String(id || '');
  return s.startsWith('builtin:') ? s : `builtin:${s}`;
}

/** Deep-merge plain objects (`{rt: {stt_engine}}` must not drop the rest of `rt`). */
export function mergeInto(target, patch) {
  if (!target || !patch || typeof patch !== 'object') return target;
  for (const [k, v] of Object.entries(patch)) {
    if (k === '__proto__' || k === 'constructor' || k === 'prototype') continue;
    const cur = target[k];
    if (v && typeof v === 'object' && !Array.isArray(v) && cur && typeof cur === 'object' && !Array.isArray(cur)) mergeInto(cur, v);
    else target[k] = v;
  }
  return target;
}

/**
 * "Couldn't save that setting." plus the server's reason for a refusal it
 * explained (a 400/409/422 `detail`), so an older server's "Unknown setting"
 * is visible (review L7). Long validation dumps are cut.
 */
export function saveErrorText(e) {
  const base = t('drive.err_save');
  const st = e && e.status;
  const msg = e && typeof e.message === 'string' ? e.message.trim() : '';
  if (!(st === 400 || st === 409 || st === 422) || !msg || msg === String(st)) return base;
  return `${base} (${msg.length > 140 ? `${msg.slice(0, 139)}…` : msg})`;
}

/** A readable name for a voice value, from /api/voice/status. */
export function voiceName(status, value) {
  const p = parseVoice(value);
  const st = status || {};
  if (p.kind === 'profile') {
    if (st.stock_voice && p.id === st.stock_voice) return t('drive.voice_stock');
    const hit = (st.profiles || []).find((x) => x.id === p.id);
    return hit ? hit.name : p.id;
  }
  if (p.kind === 'builtin') {
    const hit = (st.builtin_voices || []).find((x) => builtinValue(x.id) === builtinValue(p.id));
    return t('drive.voice_builtin', { name: hit ? (hit.name || p.id) : p.id });
  }
  return '';
}

/**
 * Options for a voice <select>: cloned voices first, then the built-in ones.
 * `withDefault` adds the per-bot "Use the default" entry. The current value
 * is always present, even when the server no longer lists it (group
 * 'missing'), so a select never silently shows another voice than the saved one.
 */
export function voiceOptions(status, { withDefault = false, current = '', placeholder = false } = {}) {
  const st = status || {};
  const out = [];
  if (placeholder && !current) out.push({ value: '', label: t('drive.voice_pick'), group: 'none' });
  if (withDefault) {
    const def = st.default_voice ? voiceName(st, st.default_voice) : '';
    out.push({ value: '', label: def ? t('drive.voice_use_default_named', { name: def }) : t('drive.voice_use_default') });
  }
  for (const p of st.profiles || []) {
    const stock = !!st.stock_voice && p.id === st.stock_voice;
    out.push({ value: p.id, label: stock ? t('drive.voice_stock') : p.name, group: stock ? 'stock' : 'profile' });
  }
  for (const b of st.builtin_voices || []) {
    out.push({ value: builtinValue(b.id), label: t('drive.voice_builtin', { name: b.name || b.id }), group: 'builtin' });
  }
  if (current && !out.some((o) => o.value === current)) {
    out.push({ value: current, label: voiceName(st, current) || current, group: 'missing' });
  }
  return out;
}

/** Role-play text options; `withServerDefault` adds the per-bot "Server default" entry. */
export function rpOptions({ withServerDefault = false } = {}) {
  const out = withServerDefault ? [{ value: '', label: t('drive.rp_server_default') }] : [];
  for (const m of RP_MODES) out.push({ value: m, label: t(`drive.rp_${m}`) });
  return out;
}

/** The narrator select: "" = the default narrator, then every voice; a missing one says so. */
export function narratorOptions(status, current = '') {
  const opts = [{ value: '', label: t('drive.narrator_default') }]
    .concat(voiceOptions(status, { current }).map((o) => (
      o.group === 'missing' ? { ...o, label: `${o.label} ${t('drive.narrator_missing')}` } : o)));
  return opts;
}

// --------------------------------------------------------- phone voice ---

/** Android? (Chrome Android lists one speech entry per locale — feasibility H2.) */
export function isAndroid(ua = (globalThis.navigator && navigator.userAgent) || '') {
  return /Android/i.test(String(ua || ''));
}

/** The speech engine's voices, waiting briefly for Chrome's async list. */
export function phoneVoices(synth = globalThis.speechSynthesis, waitMs = 400) {
  return new Promise((resolve) => {
    if (!synth || typeof synth.getVoices !== 'function') { resolve([]); return; }
    let list = [];
    try { list = synth.getVoices() || []; } catch { list = []; }
    if (list.length) { resolve(list); return; }
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      try { synth.removeEventListener && synth.removeEventListener('voiceschanged', finish); } catch { /* old engine */ }
      let l = [];
      try { l = synth.getVoices() || []; } catch { l = []; }
      resolve(l);
    };
    try { synth.addEventListener && synth.addEventListener('voiceschanged', finish); } catch { /* old engine */ }
    setTimeout(finish, waitMs);
  });
}

/** "English (United States)" for "en-US", in the UI language (SPEC §5.4: labels from `lang`). */
export function langLabel(lang, uiLang = language()) {
  const tag = String(lang || '').replace(/_/g, '-');
  try {
    // 'standard' spells the region out ("English (United States)", not "American English").
    const dn = new Intl.DisplayNames([uiLang || 'en'], { type: 'language', languageDisplay: 'standard' });
    return dn.of(tag) || tag;
  } catch { return tag; }
}

/** The phone voices worth offering for the UI language (desktop: local only). */
export function phoneVoiceChoices(voices, lang = language(), { android = isAndroid() } = {}) {
  const primary = String(lang || 'en').toLowerCase().split(/[-_]/)[0];
  return (voices || []).filter((v) => v
    && (android || v.localService === true)
    && String(v.lang || '').toLowerCase().replace(/_/g, '-').split('-')[0] === primary);
}

/**
 * What the Phone voice row shows: 'hidden' (slow-link setting is not a phone
 * one), 'none' (no usable voice: the device falls back to text), 'android'
 * (the hint naming Android's own TTS setting — the page cannot choose) or
 * 'picker' (desktop: a select of offline voices).
 */
export function phoneVoiceRowMode(slowLink, { android = false, voices = [], lang = 'en' } = {}) {
  if (slowLink !== 'phone' && slowLink !== 'always_phone') return 'hidden';
  if (!pickVoice(voices, lang, '', { android })) return 'none';
  return android ? 'android' : 'picker';
}

/** Does this device have a phone voice at all (the quick sheet offers "Phone voice" only then)? */
export function hasPhoneVoice(voices, lang = 'en', { android = false } = {}) {
  return !!pickVoice(voices, lang, '', { android });
}

/** Speak the sample sentence with a phone voice (▶ on the Phone voice row). */
export function speakSample(voice, synth = globalThis.speechSynthesis) {
  if (!synth || typeof globalThis.SpeechSynthesisUtterance !== 'function') return false;
  try {
    synth.cancel();
    const u = new SpeechSynthesisUtterance(t('drive.phone_voice_sample'));
    if (voice) { u.voice = voice; u.lang = voice.lang; }
    synth.speak(u);
    return true;
  } catch { return false; }
}

// ------------------------------------------------------------- preview ---

let previewAudio = null;
let previewUrl = null;
let previewBtn = null;

function stopPreview() {
  if (previewAudio) { try { previewAudio.pause(); } catch { /* gone */ } }
  if (previewUrl) { try { URL.revokeObjectURL(previewUrl); } catch { /* gone */ } }
  if (previewBtn) { previewBtn.setAttribute('aria-pressed', 'false'); previewBtn.classList.remove('busy'); }
  previewAudio = null; previewUrl = null; previewBtn = null;
}

/** ▶ a short sample of `voice` (a profile id or builtin:<id>); a second tap stops it. */
export async function playPreview(voice, btn, onError) {
  if (previewBtn && previewBtn === btn) { stopPreview(); return; }
  stopPreview();
  if (!voice) return;
  previewBtn = btn || null;
  if (btn) { btn.setAttribute('aria-pressed', 'true'); btn.classList.add('busy'); }
  try {
    const r = await fetch(`/api/voice/preview?voice=${encodeURIComponent(voice)}`, { credentials: 'same-origin' });
    if (!r.ok) throw new Error(String(r.status));
    if (previewBtn !== btn) return;
    previewUrl = URL.createObjectURL(await r.blob());
    previewAudio = new Audio(previewUrl);
    if (btn) btn.classList.remove('busy');
    previewAudio.onended = () => { if (previewBtn === btn) stopPreview(); };
    await previewAudio.play();
  } catch {
    if (previewBtn === btn) stopPreview();
    if (onError) onError(t('drive.err_preview'));
  }
}

// ------------------------------------------------------------ controls ---

export const PLAY_ICON = ['M8 5.5v13l10.5-6.5Z'];
export const MIC_ICON = [
  'M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Z',
  'M19 11a7 7 0 0 1-14 0',
  'M12 18v3',
];

let rowSeq = 0;
/**
 * One settings row: name + hint on the start side, controls on the end. Each
 * control is described by the row's hint (`aria-describedby`), so a screen
 * reader hears "Turn this off if…" too, not only the name (FR-ux L8).
 */
export function row(title, hint, controls = [], { id, block = false } = {}) {
  const hintId = hint ? `${id || `drive-row-${++rowSeq}`}-hint` : null;
  const text = el('span', { class: 'dev-row-text' }, [
    el('strong', { text: title, id: id ? `${id}-label` : null }),
    hint ? el('span', { class: 'dev-row-hint', text: hint, id: hintId }) : null,
  ]);
  if (hintId) {
    for (const c of controls) {
      if (!c || typeof c.matches !== 'function' || c.hasAttribute('aria-describedby')) continue;
      if (c.matches('input, select, [role="radiogroup"]')) c.setAttribute('aria-describedby', hintId);
    }
  }
  return el('div', { class: 'dev-row drive-row' + (block ? ' drive-row-block' : ''), id }, [text, ...controls]);
}

/** A second hint line under a row's first. */
function addHint(r, text, extraClass = '') {
  if (!text) return;
  r.querySelector('.dev-row-text').append(el('span', { class: 'dev-row-hint drive-note' + (extraClass ? ` ${extraClass}` : ''), text }));
}

/** On/off switch: the Device tab's switch, as a plain checkbox. */
export function switchInput(checked, label, onChange) {
  const cb = el('input', { type: 'checkbox', class: 'dev-switch', 'aria-label': label });
  cb.checked = !!checked;
  cb.addEventListener('change', () => onChange(cb.checked));
  return cb;
}

/** Segmented control (radio group) — one choice of a few. */
export function segmented(label, options, value, onChange) {
  const box = el('div', { class: 'drive-seg', role: 'radiogroup', 'aria-label': label });
  const btns = options.map((o) => {
    const b = el('button', { type: 'button', role: 'radio', class: 'drive-seg-btn', text: o.label, dataset: { value: String(o.value) } });
    b.addEventListener('click', () => { set(String(o.value)); onChange(o.value); });
    return b;
  });
  function set(v) {
    btns.forEach((b) => {
      const on = b.dataset.value === String(v);
      b.setAttribute('aria-checked', String(on));
      b.tabIndex = on ? 0 : -1;
    });
  }
  box.addEventListener('keydown', (e) => {
    const keys = ['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp'];
    if (!keys.includes(e.key)) return;
    e.preventDefault();
    const rtl = document.documentElement.dir === 'rtl';
    const fwd = e.key === 'ArrowDown' || e.key === (rtl ? 'ArrowLeft' : 'ArrowRight');
    const cur = btns.findIndex((b) => b.getAttribute('aria-checked') === 'true');
    const next = btns[(cur + (fwd ? 1 : btns.length - 1)) % btns.length];
    next.click(); next.focus();
  });
  box.append(...btns);
  set(value);
  box.setValue = set;
  return box;
}

/** A <select> in the Device tab's style. */
export function selectInput(label, options, value, onChange) {
  const sel = el('select', { class: 'lang-select drive-select', 'aria-label': label });
  fillSelect(sel, options, value);
  sel.addEventListener('change', () => onChange(sel.value));
  return sel;
}

export function fillSelect(sel, options, value) {
  sel.replaceChildren(...options.map((o) => {
    const opt = el('option', { value: o.value, text: o.label });
    if (o.value === value) opt.selected = true;
    return opt;
  }));
  sel.value = value;
}

/** ▶ preview button for whatever the paired select shows. */
export function previewButton(getVoice, onError) {
  const b = el('button', {
    type: 'button', class: 'icon-btn drive-preview', title: t('drive.preview'),
    'aria-label': t('drive.preview'), 'aria-pressed': 'false',
  }, [railIcon(PLAY_ICON)]);
  b.addEventListener('click', () => playPreview(getVoice(), b, onError));
  return b;
}

const patienceOptions = () => PATIENCE_PRESETS.map((v) => ({ value: v, label: t(`drive.patience_${v}`) }));
const replyOptions = () => REPLY_STYLES.map((v) => ({ value: v, label: t(`drive.reply_${v}`) }));
const reactionOptions = () => REACTIONS.map((v) => ({ value: v, label: t(`drive.acks_${v}`) }));
/** Follow-up is three-state (Amendment T7a-4): the bot's default, or this device's On/Off. */
const followUpOptions = () => ['bot', 'on', 'off'].map((v) => ({ value: v, label: t(`drive.follow_up_${v}`) }));
export const followUpValue = (f) => (f === true ? 'on' : f === false ? 'off' : 'bot');
const SLOW_KEYS = { phone: 'drive.slow_phone', wait: 'drive.slow_wait', text: 'drive.slow_text', always_phone: 'drive.slow_always' };
export const slowLinkOptions = (values = SLOW_LINK) => values.map((v) => ({ value: v, label: t(SLOW_KEYS[v]) }));
const hangupOptions = () => HANGUP_QUIET_MIN.map((m) => ({
  value: m, label: m ? t('drive.hangup_min', { n: m }) : t('drive.hangup_never'),
}));

/** Microphone / speaker choices. Labels need mic permission; before that they are blank. */
export async function deviceOptions(kind) {
  const out = [{ value: '', label: t('drive.device_default') }];
  try {
    const devs = await navigator.mediaDevices.enumerateDevices();
    let n = 0;
    for (const d of devs) {
      if (d.kind !== kind || !d.deviceId || d.deviceId === 'default' || d.deviceId === 'communications') continue;
      n += 1;
      out.push({ value: d.deviceId, label: d.label || t(kind === 'audioinput' ? 'drive.mic_n' : 'drive.speaker_n', { n }) });
    }
  } catch { /* no mediaDevices (http, old browser) */ }
  return out;
}

export function canPickSpeaker() {
  const AC = globalThis.AudioContext;
  return !!(AC && AC.prototype && 'setSinkId' in AC.prototype);
}

function storage() { return globalThis.localStorage; }

function micRow(s, set) {
  const micSel = selectInput(t('drive.mic'), [{ value: '', label: t('drive.device_default') }], s.micId, (v) => set({ micId: v }));
  deviceOptions('audioinput').then((o) => fillSelect(micSel, o, o.some((x) => x.value === s.micId) ? s.micId : ''));
  return row(t('drive.mic'), t('drive.mic_hint'), [micSel], { id: 'drive-mic' });
}

// --------------------------------------------------------- quick sheet ---

/**
 * The overlay gear's bottom sheet (SPEC §7.3): only what changes on the
 * road, 56 px rows, applied at once (the voice chat re-sends `start` through
 * onDriveSettings). The voice row sets THIS thread's bot voice — never the
 * server-wide default, which stays in Settings › Voices (experience M8).
 *
 * opts: { status, bot, onClose, onAllSettings, notify, setBotVoice(botId, v),
 *         voices: speechSynthesis voices (for "Phone voice"), android, lang }
 */
export function buildQuickSheet({
  status, bot, onClose, onAllSettings, notify, setBotVoice: saveBotVoice = setBotVoice,
  voices = [], android = isAndroid(), lang = language(),
}) {
  const s = loadDriveSettings(storage());
  const set = (patch) => saveDriveSettings(storage(), patch);
  const st = status || {};
  const body = el('div', { class: 'dev-group-body drive-sheet-body' });

  if (bot && bot.id && voiceOptions(st).length) {
    let cur = typeof bot.voice === 'string' ? bot.voice : '';
    const sel = selectInput(t('drive.voice_of', { bot: bot.name || '' }),
      voiceOptions(st, { withDefault: true, current: cur }), cur, async (v) => {
        try {
          await saveBotVoice(bot.id, v);
          cur = v;
          bot.voice = v;
        } catch (e) {
          fillSelect(sel, voiceOptions(st, { withDefault: true, current: cur }), cur);
          if (notify) notify(saveErrorText(e));
        }
      });
    sel.id = 'drive-sheet-voice';
    const preview = previewButton(() => sel.value || st.default_voice || '', notify);
    body.append(row(t('drive.voice_of', { bot: bot.name || '' }), t('drive.voice_sheet_hint'), [sel, preview], { id: 'drive-sheet-voice-row' }));
  }
  body.append(
    row(t('drive.patience'), t('drive.patience_hint'),
      [segmented(t('drive.patience'), patienceOptions(), s.patience, (v) => set({ patience: v }))], { block: true }),
    row(t('drive.barge_in'), t('drive.barge_in_hint'),
      [switchInput(s.bargeIn, t('drive.barge_in'), (v) => set({ bargeIn: v }))]),
    row(t('drive.reply_length'), t('drive.reply_length_hint'),
      [segmented(t('drive.reply_length'), replyOptions(), s.replyStyle, (v) => set({ replyStyle: v }))], { block: true }),
  );
  // "Phone voice" only when this device has one; "Always…" is Settings-only
  // (SPEC §15 #9) unless it is what this device already chose.
  const phone = hasPhoneVoice(voices, lang, { android });
  const values = [phone ? 'phone' : null, 'wait', 'text', phone && s.slowLink === 'always_phone' ? 'always_phone' : null].filter(Boolean);
  // What the device will actually do: without a phone voice, "Phone voice" and
  // "Always use the phone's voice" both fall back to text (drive-audio
  // localPath), so that is what is highlighted, with the reason under it
  // (FR-ux L4: it used to highlight Wait while the engine routed to text).
  const slowVal = values.includes(s.slowLink) ? s.slowLink : (s.slowLink === 'wait' ? 'wait' : 'text');
  const slowRow = row(t('drive.slow_link'), t('drive.slow_link_hint'),
    [segmented(t('drive.slow_link'), slowLinkOptions(values), slowVal, (v) => set({ slowLink: v }))],
    { block: true, id: 'drive-sheet-slow' });
  if (!phone && (s.slowLink === 'phone' || s.slowLink === 'always_phone')) addHint(slowRow, t('drive.link_no_phone_voice'));
  body.append(slowRow);
  body.append(micRow(s, set));

  const all = el('button', { type: 'button', class: 'dev-row drive-row drive-row-link', id: 'drive-sheet-all' }, [
    el('span', { class: 'dev-row-text' }, [
      el('strong', { text: t('drive.all_settings') }),
      el('span', { class: 'dev-row-hint', text: t('drive.all_settings_hint') }),
    ]),
    railIcon(iconPaths('forward')),
  ]);
  all.addEventListener('click', () => onAllSettings && onAllSettings());
  body.append(all);

  const title = el('h2', { class: 'drive-sheet-title', id: 'drive-sheet-title', text: t('drive.sheet_title') });
  const closeBtn = el('button', {
    type: 'button', class: 'icon-btn ghost drive-sheet-close', 'aria-label': t('drive.close'),
  }, [railIcon(iconPaths('close'))]);
  closeBtn.addEventListener('click', () => onClose && onClose());
  return el('div', {
    class: 'drive-sheet', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'drive-sheet-title',
  }, [el('div', { class: 'drive-sheet-head' }, [title, closeBtn]), body]);
}

// ------------------------------------------------------ Settings → Voice --

/**
 * What Settings → Voice shows for this /api/voice/status answer.
 *   'none'    no status (network error): an error row and Try again
 *   'forced'  DISPATCH_VOICE=0 on the server: no switch, one line saying so
 *   'off'     the switch is off (the default on a fresh install)
 *   'setup'   the switch is on, but voice can't run yet: what is missing
 *   'on'      voice works: the switch, then the everyday settings
 */
export function voiceTabState(st) {
  if (!st || typeof st !== 'object') return 'none';
  if (st.forced === 'off') return 'forced';
  const on = typeof st.switch === 'boolean' ? st.switch : !!st.enabled;
  if (!on) return 'off';
  const missing = Array.isArray(st.missing) ? st.missing : [];
  return st.enabled && !missing.length ? 'on' : 'setup';
}

/** Where a setup command runs. Literal keys so the i18n scan sees them. */
const WHERE = {
  backend: 'drive.where_backend',
  start: 'drive.where_start',
  plugin: 'drive.where_plugin',
  env: 'drive.where_env',
};

/** The add-on's own commands (shell, never translated). Run in DisPatch's backend folder. */
export const DV_STEPS = Object.freeze({
  install: { where: 'backend', cmd: `uv pip install --python .venv/bin/python "dispatch-voice @ git+${VOICE_REPO_URL}@v1"` },
  models: { where: 'backend', cmd: '.venv/bin/dispatch-voice fetch-models' },
  doctor: { where: 'backend', cmd: '.venv/bin/dispatch-voice doctor' },
});

/** The commands that fix a /api/voice/status `missing` list, in order: [{where, cmd}]. */
export function setupSteps(missing) {
  const m = (Array.isArray(missing) ? missing : []).map(String);
  const has = (re) => m.some((x) => re.test(x));
  const steps = [];
  // A Python package missing from the add-on's own install: reinstall it
  // (`uv sync` in DisPatch's folder removes the add-on — `doctor` says so).
  if (has(/^python package|^dispatch-voice package|^a text-to-speech engine/)) steps.push(DV_STEPS.install);
  if (has(/^model |^dispatch-voice models|models?\b.*fetch-models/)) steps.push(DV_STEPS.models);
  if (has(/^MINIMAX_API_KEY/)) steps.push({ where: 'env', cmd: 'MINIMAX_API_KEY=…' });
  if (steps.length) steps.push(DV_STEPS.doctor);
  return steps;
}

/** One "what to run" block: a hint line per step and the command as code. */
export function stepsBlock(steps) {
  const box = el('div', { class: 'voice-steps' });
  for (const s of steps || []) {
    box.append(
      el('p', { class: 'dev-row-hint voice-step-where', text: t(WHERE[s.where] || WHERE.backend) }),
      el('pre', { class: 'voice-cmd', dir: 'ltr' }, [el('code', { text: s.cmd })]),
    );
  }
  return box;
}

function missingNote(missing) {
  const list = Array.isArray(missing) ? missing.map(String) : [];
  if (!list.length) return null;
  return el('p', { class: 'drive-missing', role: 'note' }, [
    el('strong', { text: t('drive.missing_title') }), ' ',
    el('span', { text: listFormat(list) }),
  ]);
}

export function group(titleKey, rows, { extraClass = '', id = null, hint = null } = {}) {
  return el('section', { class: 'dev-group drive-group' + (extraClass ? ' ' + extraClass : ''), id }, [
    el('h3', { class: 'dev-group-title', text: t(titleKey) }),
    hint ? el('p', { class: 'dev-group-hint', text: hint }) : null,
    el('div', { class: 'dev-group-body' }, rows.filter(Boolean)),
  ]);
}

/** A collapsed group (the Status group): <details> with the group title as its summary. */
function foldGroup(titleKey, rows, { id = null } = {}) {
  return el('details', { class: 'dev-group drive-group drive-fold', id }, [
    el('summary', { class: 'dev-group-title drive-fold-title', text: t(titleKey) }),
    el('div', { class: 'dev-group-body' }, rows.filter(Boolean)),
  ]);
}

/** The switch on top of Settings → Voice. */
function switchGroup(st, ctx) {
  const forced = st.forced || null;
  let r;
  if (forced === 'off') {
    r = row(t('drive.switch'), t('drive.switch_forced_off'), [], { id: 'drive-switch' });
  } else {
    const on = forced === 'on' ? true : !!st.switch;
    const cb = switchInput(on, t('drive.switch'), async (v) => {
      cb.disabled = true;
      const ok = ctx.setEnabled ? await ctx.setEnabled(v) : false;
      if (!ok) { cb.checked = !v; cb.disabled = false; }
    });
    cb.id = 'drive-switch-input';
    if (forced === 'on') cb.disabled = true;
    r = row(t('drive.switch'), forced === 'on' ? t('drive.switch_forced_on') : t('drive.switch_hint'), [cb], { id: 'drive-switch' });
  }
  // Always shown: the engines understand and speak English only (H3).
  addHint(r, t('drive.english_only'), 'drive-english-only');
  return group('drive.group_switch', [r], { id: 'drive-group-switch', hint: t('drive.group_server_hint') });
}

/** Below the switch while voice is off or not ready: what it needs. */
function needsGroup(st, state) {
  const missing = Array.isArray(st.missing) ? st.missing : [];
  const r = row(
    state === 'off' ? t('drive.needs_title_off') : t('drive.setup_title'),
    state === 'off' ? t('drive.needs_hint_off') : t('drive.setup_hint'),
    [], { id: 'drive-setup' },
  );
  r.classList.add('dev-row-block');
  const note = missingNote(missing);
  if (note) r.append(note);
  const steps = setupSteps(missing);
  if (steps.length) {
    r.append(stepsBlock(steps));
    r.append(el('p', { class: 'dev-row-hint drive-setup-fix', text: t('drive.setup_restart') }));
  } else if (missing.length) {
    r.append(el('p', { class: 'dev-row-hint drive-setup-fix', text: t('drive.setup_docs') }));
  }
  return group('drive.group_status', [r], { id: 'drive-group-needs' });
}

/**
 * The engines in words, as short sentences: "Your voices, with a built-in
 * fallback. Turn detection on. Speech recognition: Parakeet." (the Advanced
 * row and the Voice lab). No middle-dot meta string and no raw engine id
 * (FR-ux L6, SPEC §7.5).
 */
export function engineLine(st) {
  const s = st || {};
  const parts = [];
  const tts = String(s.tts || 'auto');
  const stock = s.stock_voice || null;
  const own = (s.profiles || []).filter((p) => p && p.id !== stock);
  const builtin = (s.builtin_voices || []).length > 0;
  const dv = !!(s.voices_ui || stock || (s.extensions || []).includes('dispatch-voice') || s.extension);
  const cloner = tts === 'auto' || tts === 'chatterbox';
  if (tts === 'minimax') parts.push(t('drive.engine_cloud'));
  else if (dv && cloner && builtin) parts.push(t('drive.engine_cloned'));
  else if (dv && cloner && own.length) parts.push(t('drive.engine_cloned_only'));
  else if (dv && cloner) parts.push(t('drive.engine_stock'));
  else parts.push(t('drive.engine_builtin'));
  parts.push(s.smart_turn ? t('drive.engine_turn_on') : t('drive.engine_turn_off'));
  const stt = s.stt || (s.rt && s.rt.stt);
  if (stt) {
    const key = `vtools.stt_${String(stt).replace(/[^a-z0-9_]/gi, '')}`;
    const name = has(key) ? t(key) : String(stt);
    parts.push(t('drive.engine_stt', { name }));
  }
  return parts.join(' ');
}

/** The read-only "Voice chats" lines: sessions open and the power hold (§4.11). */
export function sessionLines(st) {
  const s = st || {};
  const out = [];
  const ses = s.sessions;
  if (ses && Number.isFinite(ses.open) && Number.isFinite(ses.max)) {
    out.push(t('drive.sessions_open', { open: ses.open, max: ses.max }));
  }
  const ph = s.power_hold;
  if (ph === 'held') out.push(t('drive.power_held'));
  else if (ph === 'off') out.push(t('drive.power_off'));
  else if (ph === 'unavailable') out.push(t('drive.power_unavailable'));
  return out;
}

/**
 * Settings → Voice (SPEC §7.1). `host` is the pane body. Re-callable: it
 * rebuilds from scratch whenever the status changes.
 *
 * ctx: {
 *   status,                         the last /api/voice/status (null = not loaded / failed)
 *   loading: bool,                  a status request is in flight
 *   bots: () => [{id,name,voice?,safe,visible}],
 *   setEnabled(on) → Promise<bool>  save the switch (index.js re-reads status)
 *   cloning: bool, openCloning()    "Manage voices" (the Voice cloning tool)
 *   retry()                         re-read the status (the error row's button)
 *   toast(msg, err)
 *   voices, android                 phone voices (tests inject them)
 * }
 */
export function mountDriveSettings(host, ctx) {
  if (!host) return;
  const st = ctx.status || null;
  const lead = el('p', { class: 'muted modal-hint drive-settings-hint', text: t('drive.settings_hint') });
  const state = voiceTabState(st);
  if (state === 'none') {
    if (ctx.loading) {
      host.replaceChildren(lead, el('p', { class: 'dev-group-hint vtool-loading', text: t('drive.loading') }));
      return;
    }
    const again = el('button', { type: 'button', class: 'btn-secondary', id: 'drive-status-retry', text: t('drive.try_again') });
    again.addEventListener('click', () => ctx.retry && ctx.retry());
    const r = row(t('drive.status_failed'), t('drive.status_failed_hint'), [again], { id: 'drive-status-failed' });
    r.classList.add('dev-row-action');
    host.replaceChildren(lead, group('drive.group_status', [r]));
    return;
  }
  const groups = [switchGroup(st, ctx)];
  if (state !== 'on') {
    const missing = Array.isArray(st.missing) ? st.missing : [];
    if (state === 'setup' || (state === 'off' && missing.length)) groups.push(needsGroup(st, state));
    host.replaceChildren(lead, ...groups);
    return;
  }
  const s = loadDriveSettings(storage());
  const set = (patch) => saveDriveSettings(storage(), patch);
  const notify = (m) => ctx.toast && ctx.toast(m, true);
  const save = async (patch) => {
    try {
      await patchVoiceSettings(patch);
    } catch (e) {
      notify(e && e.status === 404 ? t('drive.err_settings_missing') : saveErrorText(e));
      return false;
    }
    // Only what was sent goes into the status copy: the PATCH answer is the
    // settings view (`enabled` = the bare switch, `rt.power_hold`, `opus:
    // auto|off`), not the status shape (review L3). The owner then re-reads
    // the status in the background.
    mergeInto(st, patch);
    if (ctx.saved) ctx.saved();
    return true;
  };

  // ---- Voices (server)
  const voiceRows = [];
  const builtins = Array.isArray(st.builtin_voices) ? st.builtin_voices : [];
  const stock = st.stock_voice || null;
  const own = (st.profiles || []).filter((p) => p.id !== stock);
  if (voiceOptions(st).length) {
    const opts = () => voiceOptions(st, { current: st.default_voice || '', placeholder: true });
    const sel = selectInput(t('drive.default_voice'), opts(), st.default_voice || '', async (v) => {
      if (!v) return;
      if (await save({ default_voice: v })) {
        fillSelect(sel, opts(), st.default_voice || v);
        paintSpeed();
      } else fillSelect(sel, opts(), st.default_voice || '');
    });
    const r = row(t('drive.default_voice'), t('drive.default_voice_hint'), [sel, previewButton(() => sel.value, notify)], { id: 'drive-default-voice' });
    // ONE hint line, written from the roster when it is known: who uses the
    // default, or that nobody does right now.
    const hintEl = r.querySelector('.dev-row-hint');
    const bots = (ctx.bots ? ctx.bots() : []).filter((b) => b && !b.safe && b.visible !== false);
    if (bots.length && bots.every((b) => typeof b.voice === 'string')) {
      const uses = bots.filter((b) => !b.voice).map((b) => b.name);
      hintEl.textContent = uses.length ? t('drive.voice_users', { names: listFormat(uses) }) : t('drive.voice_users_none');
    }
    const note = !own.length ? (builtins.length ? 'drive.voice_builtin_only' : (stock ? 'drive.voice_stock_only' : null)) : null;
    if (note) addHint(r, t(note));
    voiceRows.push(r);

    // Speaking speed: built-in voices only (a cloned or the stock voice
    // speaks at its own pace), so no row at all without them.
    const speed = el('input', {
      type: 'range', class: 'drive-range', min: String(SPEED_MIN), max: String(SPEED_MAX), step: '0.05',
      'aria-label': t('drive.speed'),
    });
    const out = el('output', { class: 'drive-range-value' });
    const showSpeed = () => {
      const v = clampSpeed(speed.value);
      out.textContent = Math.abs(v - 1) < 0.001 ? t('drive.speed_normal') : t('drive.speed_value', { x: v.toFixed(2) });
    };
    speed.value = String(clampSpeed(st.speed != null ? st.speed : 1));
    showSpeed();
    speed.addEventListener('input', showSpeed);
    speed.addEventListener('change', () => save({ speed: clampSpeed(speed.value) }));
    const speedRow = row(t('drive.speed'), t('drive.speed_hint'), [speed, out], { id: 'drive-speed' });
    function paintSpeed() {
      const cloned = parseVoice(sel.value).kind === 'profile';
      speedRow.querySelector('.dev-row-hint').textContent = cloned ? t('drive.speed_hint_cloned') : t('drive.speed_hint');
    }
    paintSpeed();
    if (builtins.length) voiceRows.push(speedRow);
  } else {
    voiceRows.push(row(t('drive.voice_none'), t('drive.voice_none_hint'), [], { id: 'drive-no-voice' }));
  }

  // Role-play text + the narrator voice (shown only for the narrator mode).
  let rpMode = RP_MODES.includes(st.rp_speech) ? st.rp_speech : 'all';
  const narratorSel = selectInput(t('drive.narrator_voice'), narratorOptions(st, st.narrator_voice || ''), st.narrator_voice || '', async (v) => {
    if (!(await save({ narrator_voice: v }))) fillSelect(narratorSel, narratorOptions(st, st.narrator_voice || ''), st.narrator_voice || '');
  });
  const narratorRow = row(t('drive.narrator_voice'), t('drive.narrator_voice_hint'),
    [narratorSel, previewButton(() => narratorSel.value || stock || '', notify)], { id: 'drive-narrator' });
  const rpSel = selectInput(t('drive.rp_speech'), rpOptions(), rpMode, async (v) => {
    if (await save({ rp_speech: v })) { rpMode = v; narratorRow.hidden = rpMode !== 'dialogue_narrator'; }
    else rpSel.value = rpMode;
  });
  narratorRow.hidden = rpMode !== 'dialogue_narrator';
  rpSel.classList.add('drive-select-wide');   // its longest choice is a sentence
  voiceRows.push(row(t('drive.rp_speech'), t('drive.rp_speech_hint'), [rpSel], { id: 'drive-rp' }), narratorRow);

  if (ctx.cloning) {
    const open = el('button', { type: 'button', class: 'btn-secondary', id: 'dev-voices-open' }, [
      railIcon(MIC_ICON), document.createTextNode(` ${t('drive.manage_voices_open')}`),
    ]);
    open.addEventListener('click', () => ctx.openCloning && ctx.openCloning());
    const r = row(t('drive.manage_voices'), t('drive.manage_voices_hint'), [open], { id: 'drive-manage-voices' });
    r.classList.add('dev-row-action');
    voiceRows.push(r);
  }
  groups.push(group('drive.group_voices', voiceRows, { id: 'drive-group-voices', hint: t('drive.group_server_hint') }));

  // ---- Listening (device)
  groups.push(group('drive.group_listening', [
    row(t('drive.patience'), t('drive.patience_hint'),
      [segmented(t('drive.patience'), patienceOptions(), s.patience, (v) => set({ patience: v }))], { block: true, id: 'drive-patience' }),
    row(t('drive.barge_in'), t('drive.barge_in_hint'),
      [switchInput(s.bargeIn, t('drive.barge_in'), (v) => set({ bargeIn: v }))], { id: 'drive-barge-in' }),
    row(t('drive.auto_listen'), t('drive.auto_listen_hint'),
      [switchInput(s.autoListen, t('drive.auto_listen'), (v) => set({ autoListen: v }))], { id: 'drive-auto-listen' }),
    micRow(s, set),
  ], { id: 'drive-group-listening' }));

  // ---- Replies (device)
  groups.push(group('drive.group_replies', [
    row(t('drive.reply_length'), t('drive.reply_length_hint'),
      [segmented(t('drive.reply_length'), replyOptions(), s.replyStyle, (v) => set({ replyStyle: v }))], { block: true, id: 'drive-reply-length' }),
    row(t('drive.follow_up'), t('drive.follow_up_hint'),
      [segmented(t('drive.follow_up'), followUpOptions(), followUpValue(s.followUp),
        (v) => set({ followUp: v === 'on' ? true : v === 'off' ? false : null }))], { block: true, id: 'drive-follow-up' }),
    row(t('drive.reactions'), t('drive.reactions_hint'),
      [segmented(t('drive.reactions'), reactionOptions(), s.reactions, (v) => set({ reactions: v }))], { block: true, id: 'drive-reactions' }),
    row(t('drive.tones'), t('drive.sounds_hint'),
      [switchInput(s.sounds, t('drive.tones'), (v) => set({ sounds: v }))], { id: 'drive-tones' }),
  ], { id: 'drive-group-replies' }));

  // ---- Connection (device)
  groups.push(connectionGroup(s, set, ctx));

  // ---- This device
  groups.push(group('drive.group_device', [
    row(t('drive.screen_on'), t('drive.screen_on_hint'),
      [switchInput(s.screenOn, t('drive.screen_on'), (v) => set({ screenOn: v }))], { id: 'drive-screen-on' }),
    row(t('drive.captions'), t('drive.captions_hint'),
      [switchInput(s.captions, t('drive.captions'), (v) => set({ captions: v }))], { id: 'drive-captions' }),
    row(t('drive.hangup_quiet'), t('drive.hangup_quiet_hint'),
      [segmented(t('drive.hangup_quiet'), hangupOptions(), s.hangupQuietMin, (v) => set({ hangupQuietMin: Number(v) }))],
      { block: true, id: 'drive-hangup-quiet' }),
  ], { id: 'drive-group-device' }));

  // ---- Status (collapsed): everything an owner reads rather than changes.
  const statusRows = [row(t('vtools.lab_engine'), engineLine(st), [], { id: 'drive-engine' })];
  if (canPickSpeaker() && !isMobile()) {
    const spk = selectInput(t('drive.speaker'), [{ value: '', label: t('drive.device_default') }], s.sinkId, (v) => set({ sinkId: v }));
    deviceOptions('audiooutput').then((o) => fillSelect(spk, o, o.some((x) => x.value === s.sinkId) ? s.sinkId : ''));
    statusRows.push(row(t('drive.speaker'), t('drive.speaker_hint'), [spk], { id: 'drive-speaker' }));
  }
  const levels = Array.isArray(st.thinking_levels) && st.thinking_levels.length ? st.thinking_levels : THINKING_LEVELS;
  const thinkSel = selectInput(t('drive.thinking'), levels.map((v) => ({ value: v, label: t(`drive.thinking_${v}`) })),
    THINKING_LEVELS.includes(st.voice_turn_thinking) ? st.voice_turn_thinking : 'off', (v) => save({ voice_turn_thinking: v }));
  const thinkRow = row(t('drive.thinking'), t('drive.thinking_hint'), [thinkSel], { id: 'drive-thinking' });
  addHint(thinkRow, t('drive.group_server_hint'));
  statusRows.push(thinkRow);
  statusRows.push(row(t('vtools.show_timings'), t('vtools.show_timings_hint'),
    [switchInput(s.showTimings, t('vtools.show_timings'), (v) => set({ showTimings: v }))], { id: 'drive-show-timings' }));
  if (typeof st.expressive_tags === 'boolean') {
    const tagRow = row(t('vtools.expressive_tags'), t('vtools.expressive_tags_hint'),
      [switchInput(st.expressive_tags, t('vtools.expressive_tags'), (v) => save({ expressive_tags: v }))], { id: 'drive-expressive' });
    addHint(tagRow, t('drive.group_server_hint'));
    statusRows.push(tagRow);
  }
  const lines = sessionLines(st);
  if (lines.length) {
    const r = row(t('drive.status_sessions'), lines[0], [], { id: 'drive-sessions' });
    for (const l of lines.slice(1)) addHint(r, l);
    statusRows.push(r);
  }
  groups.push(foldGroup('drive.group_status', statusRows, { id: 'drive-group-status' }));

  host.replaceChildren(lead, ...groups);
}

/** Connection: when the link is slow, and the phone voice (§5.4, §7.1). */
function connectionGroup(s, set, ctx) {
  const body = el('div', { class: 'dev-group-body' });
  const box = el('section', { class: 'dev-group drive-group', id: 'drive-group-connection' }, [
    el('h3', { class: 'dev-group-title', text: t('drive.group_connection') }), body,
  ]);
  const android = ctx.android != null ? !!ctx.android : isAndroid();
  let voices = Array.isArray(ctx.voices) ? ctx.voices : [];
  let cur = s;
  const paint = () => {
    const slow = segmented(t('drive.slow_link'), slowLinkOptions(), cur.slowLink, (v) => {
      cur = set({ slowLink: v });
      paint();
      const seg = body.querySelector('#drive-slow .drive-seg-btn[aria-checked="true"]');
      if (seg) seg.focus();
    });
    const slowRow = row(t('drive.slow_link'), t('drive.slow_link_hint'), [slow], { block: true, id: 'drive-slow' });
    if (cur.slowLink === 'always_phone') addHint(slowRow, t('drive.slow_always_hint'));
    body.replaceChildren(...[slowRow, phoneVoiceRow(cur, set, voices, android)].filter(Boolean));
  };
  paint();
  if (!Array.isArray(ctx.voices)) {
    phoneVoices().then((v) => { if (v.length) { voices = v; paint(); } });
  }
  return box;
}

function phoneVoiceRow(s, set, voices, android) {
  const lang = language();
  const mode = phoneVoiceRowMode(s.slowLink, { android, voices, lang });
  if (mode === 'hidden') return null;
  if (mode === 'none') return row(t('drive.phone_voice'), t('drive.phone_voice_none'), [], { id: 'drive-phone-voice' });
  const sample = el('button', {
    type: 'button', class: 'icon-btn drive-preview', title: t('drive.preview'), 'aria-label': t('drive.preview'),
  }, [railIcon(PLAY_ICON)]);
  if (mode === 'android') {
    sample.addEventListener('click', () => speakSample(pickVoice(voices, lang, s.phoneVoice, { android: true })));
    return row(t('drive.phone_voice'), t('drive.phone_voice_android_hint'), [sample], { id: 'drive-phone-voice' });
  }
  const choices = phoneVoiceChoices(voices, lang, { android: false });
  const chosen = pickVoice(voices, lang, s.phoneVoice, { android: false });
  // "Alex — English (United States)": the region label has its own parentheses.
  const opts = choices.map((v) => ({ value: v.voiceURI, label: `${v.name} — ${langLabel(v.lang)}` }));
  const sel = selectInput(t('drive.phone_voice'), opts, chosen ? chosen.voiceURI : '', (v) => set({ phoneVoice: v }));
  sample.addEventListener('click', () => speakSample(choices.find((v) => v.voiceURI === sel.value) || chosen));
  return row(t('drive.phone_voice'), t('drive.phone_voice_hint'), [sel, sample], { id: 'drive-phone-voice' });
}

// --------------------------------------------------- Bots tab popover ---

/**
 * The Voice pill on a Bots-tab card, opening a small popover (SPEC §7.4):
 * Voice, Role-play text (Server default + 4), "Reactions while I talk" (the
 * per-bot backchannel opt-in) and "Asks a follow-up" (the per-bot default).
 *
 * Voice and role-play text are DisPatch bot fields: they go through
 * `onChange(patch)` and wait for the Bots tab's Save, like the card's other
 * pills. The two switches are the add-on's own per-bot settings
 * (voice.yaml `bots.<id>`, never new DisPatch fields) and save at once; the
 * popover's footnote says which is which.
 */
/**
 * Keep an anchored popover inside the box that clips it (review M2): the
 * nearest ancestor that scrolls or clips horizontally — the Bots tab's pane
 * scrolls, so anything past its edge is unreachable — intersected with the
 * viewport, less an 8 px gutter. The popover hangs from the anchor's end
 * edge by default (voice.css); this shifts it sideways, and narrows it when
 * the box is narrower than the popover. Physical left/right in the inline
 * style beat the stylesheet's logical inset in both writing directions.
 */
export const POPOVER_GUTTER = 8;
export function placePopover(pop, anchor) {
  if (!pop || !anchor || !pop.isConnected) return;
  const doc = anchor.ownerDocument;
  const win = doc.defaultView;
  if (!win || typeof pop.getBoundingClientRect !== 'function') return;
  pop.style.left = ''; pop.style.right = ''; pop.style.maxWidth = '';
  let lo = 0;
  let hi = win.innerWidth || doc.documentElement.clientWidth || 0;
  for (let n = anchor.parentElement; n && n !== doc.body && n !== doc.documentElement; n = n.parentElement) {
    const ox = win.getComputedStyle(n).overflowX;
    if (ox && ox !== 'visible') {
      const r = n.getBoundingClientRect();
      if (r.width > 0) { lo = Math.max(lo, r.left); hi = Math.min(hi, r.right); }
      break;
    }
  }
  const avail = hi - lo - 2 * POPOVER_GUTTER;
  if (!(avail > 0)) return;
  let pr = pop.getBoundingClientRect();
  if (pr.width > avail) { pop.style.maxWidth = `${avail}px`; pr = pop.getBoundingClientRect(); }
  let x = pr.left;
  if (pr.right > hi - POPOVER_GUTTER) x -= pr.right - (hi - POPOVER_GUTTER);
  if (x < lo + POPOVER_GUTTER) x = lo + POPOVER_GUTTER;
  if (Math.abs(x - pr.left) < 0.5) return;
  const ar = anchor.getBoundingClientRect();
  pop.style.left = `${Math.round(x - ar.left)}px`;
  pop.style.right = 'auto';
}

export function botVoicePill(bot, status, onChange, { saveBotSettings = (id, patch) => patchVoiceSettings({ bots: { [id]: patch } }), notify = null } = {}) {
  const st = status || {};
  let voice = typeof bot.voice === 'string' ? bot.voice : '';
  let rp = typeof bot.voice_rp === 'string' ? bot.voice_rp : '';
  const per = (st.bots && st.bots[bot.id]) || {};
  const wrap = el('span', { class: 'bm-voice-wrap' });
  const word = el('span', { class: 'bm-voice-word' });
  const pill = el('button', {
    type: 'button', class: 'bm-safe-btn bm-voice-pill', title: t('drive.bot_voice_title'),
    'aria-label': t('drive.bot_voice_aria', { name: bot.name }), 'aria-haspopup': 'dialog', 'aria-expanded': 'false',
  }, [railIcon(MIC_ICON), word]);
  const paint = () => {
    word.textContent = voice ? voiceName(st, voice) : t('drive.bot_voice_badge');
    pill.classList.toggle('on', !!voice || !!rp);
  };
  paint();
  let pop = null;
  const onDoc = (e) => { if (pop && !wrap.contains(e.target)) close(); };
  const onKey = (e) => { if (e.key === 'Escape' && pop) { e.stopPropagation(); close(); pill.focus(); } };
  const onResize = () => { if (pop) placePopover(pop, wrap); };
  function close() {
    if (!pop) return;
    pop.remove(); pop = null;
    pill.setAttribute('aria-expanded', 'false');
    document.removeEventListener('pointerdown', onDoc, true);
    globalThis.removeEventListener && globalThis.removeEventListener('resize', onResize);
  }
  function open() {
    const voiceSel = selectInput(t('drive.voice'), voiceOptions(st, { withDefault: true, current: voice }), voice, (v) => {
      voice = v; paint(); onChange({ voice: v });
    });
    const rpSel = selectInput(t('drive.rp_speech'), rpOptions({ withServerDefault: true }), rp, (v) => {
      rp = v; paint(); onChange({ voice_rp: v });
    });
    const saveFlag = (key, cb) => async (v) => {
      cb.disabled = true;
      try { await saveBotSettings(bot.id, { [key]: v }); per[key] = v; }
      catch (e) { cb.checked = !v; if (notify) notify(saveErrorText(e)); }
      cb.disabled = false;
    };
    const bc = switchInput(per.backchannel === true, t('drive.bot_backchannel'), () => {});
    bc.addEventListener('change', () => saveFlag('backchannel', bc)(bc.checked));
    const fu = switchInput(per.follow_up !== false, t('drive.bot_follow_up'), () => {});
    fu.addEventListener('change', () => saveFlag('follow_up', fu)(fu.checked));
    pop = el('div', {
      class: 'menu bm-voice-pop', role: 'dialog', 'aria-label': t('drive.bot_voice_aria', { name: bot.name }),
    }, [el('div', { class: 'dev-group-body' }, [
      row(t('drive.voice'), '', [voiceSel, previewButton(() => voiceSel.value || st.default_voice || '', notify)], { id: `bm-voice-${bot.id}` }),
      row(t('drive.rp_speech'), '', [rpSel], { id: `bm-rp-${bot.id}` }),
      row(t('drive.bot_backchannel'), t('drive.bot_backchannel_hint'), [bc]),
      row(t('drive.bot_follow_up'), t('drive.bot_follow_up_hint'), [fu]),
    ]), el('p', { class: 'dev-group-hint bm-voice-note', text: t('drive.pop_save_note') })]);
    pop.addEventListener('keydown', onKey);
    pop.addEventListener('click', (e) => e.stopPropagation());
    wrap.append(pop);
    placePopover(pop, wrap);
    pill.setAttribute('aria-expanded', 'true');
    document.addEventListener('pointerdown', onDoc, true);
    globalThis.addEventListener && globalThis.addEventListener('resize', onResize);
    try { voiceSel.focus({ preventScroll: true }); } catch { voiceSel.focus(); }
    try { pop.scrollIntoView({ block: 'nearest', inline: 'nearest' }); } catch { /* jsdom */ }
  }
  pill.addEventListener('click', (e) => { e.stopPropagation(); if (pop) close(); else open(); });
  wrap.append(pill);
  wrap.closePopover = close;
  return wrap;
}
