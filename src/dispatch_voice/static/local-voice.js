// DisPatch Voice: local-voice.js — the phone's own voice (speechSynthesis) for a slow link
// (v3 SPEC §5.4, contracts C-§E.2, C.3 `heard`/`playback_done`, C.5 #12). Pure apart from
// the injected `synth` (window.speechSynthesis) and `Utterance` (SpeechSynthesisUtterance).
//
// What Chrome on Android actually exposes (feasibility H2, Chromium TtsPlatformImpl):
// one entry per available LOCALE, named like "English", `lang` "en_US", `localService`
// always true; the voice is whatever the user picked under Android Settings ›
// Text-to-speech output, and the page cannot tell an offline voice from a network one.
// So pickVoice matches the UI language's primary subtag, prefers an exact `lang`, labels
// entries from `lang` ("English (United States)"), applies the `localService` filter on
// desktop only, and honours a stored voice URI when it is still offered.
//
// Speaking: one utterance per `say` (the splitter keeps sentences ≤ 260 chars, so
// Chromium's long-utterance halt never applies; a longer text is split here anyway), our
// own queue, a watchdog (no `onstart` in 1.5 s, or no `onend` in chars/12 + 3 s → cancel
// and count), two failures → `onFail()` and text only for this voice chat. Three clean
// sentences in a row forgive a failure (a cold Android engine is slow to start once after
// being evicted; two such starts 50 sentences apart are not a broken voice).
// Cancel, never pause: Android's pause() is stop. `barge_in` → dropOlderThan(gen);
// a server `duck` is informational while the phone speaks (it keeps speaking);
// `resume {rewind: true}` → respeak(); nothing is re-spoken after an `echo_ignored`.
//
// Owner: track T8.

export const LOCAL_VOICE = Object.freeze({
  maxChars: 260,
  startTimeoutMs: 1500,
  endCharsPerSec: 12,         // watchdog: chars / 12 + 3 s
  endPadMs: 3000,
  failuresToText: 2,
  cleanToForgive: 3,          // clean sentences in a row that reset the failure count
  drainMs: 400,               // playback_done 400 ms after the queue drains with no say pending
  heardCharsPerSec: 15,       // the server maps elapsed → chars at rate × 15 chars/s
  afterCancelMs: 60,          // Chrome can drop a speak() issued in the same task as cancel()
});

const genNewer = (a, b) => a !== b && ((a - b) & 0xffff) < 0x8000;
const genOlder = (a, b) => a !== b && !genNewer(a, b);

/** 'en_US' / 'EN-us' → 'en-US'; '' → ''. */
export function normLang(lang) {
  const parts = String(lang || '').trim().replace(/_/g, '-').split('-').filter(Boolean);
  if (!parts.length) return '';
  return [parts[0].toLowerCase(), ...parts.slice(1).map((p) => (p.length === 2 ? p.toUpperCase() : p))].join('-');
}

export const primaryTag = (lang) => normLang(lang).split('-')[0] || '';

/** True on Android (Chrome's voice list is per locale there). */
export function isAndroid(ua = globalThis.navigator?.userAgent || '') { return /Android/i.test(ua); }

/** "English (United States)" from a voice's lang, in the UI language. */
export function langLabel(lang, uiLang = 'en') {
  const l = normLang(lang);
  try {
    // 'standard' gives "English (United States)", not the dialect form "American English".
    const dn = new Intl.DisplayNames([uiLang || 'en'], { type: 'language', languageDisplay: 'standard' });
    return dn.of(l) || l;
  } catch { return l; }
}

/**
 * The voice the phone reads replies with, or null (the device's fallback is then Text).
 * @param {SpeechSynthesisVoice[]} voices  speechSynthesis.getVoices()
 * @param {string} lang                     the UI language (optionally with a region, 'en-GB')
 * @param {string|null} preferredURI        the stored `phoneVoice` (honoured when offered)
 * @param {{android?: boolean}} [opts]
 */
export function pickVoice(voices, lang, preferredURI = null, { android = false } = {}) {
  const want = normLang(lang || 'en');
  const prim = primaryTag(want);
  let cands = Array.from(voices || []).filter((v) => v && v.lang && primaryTag(v.lang) === prim);
  if (!android) cands = cands.filter((v) => v.localService === true);
  if (!cands.length) return null;
  if (preferredURI) {
    const p = cands.find((v) => v.voiceURI === preferredURI);
    if (p) return p;
  }
  const exact = cands.filter((v) => normLang(v.lang) === want);
  let pool = exact.length ? exact : cands;
  const offline = pool.filter((v) => !/network|online/i.test(`${v.name || ''} ${v.voiceURI || ''}`));
  if (offline.length) pool = offline;
  return pool.find((v) => v.default) || pool[0];
}

/**
 * Picker rows for Settings (desktop; Android shows the system-settings hint instead):
 * [{uri, label, lang}] in the UI language, the voice name added only where two share a lang.
 */
export function voiceOptions(voices, lang, { android = false, uiLang = 'en' } = {}) {
  const prim = primaryTag(lang || 'en');
  let list = Array.from(voices || []).filter((v) => v && v.lang && primaryTag(v.lang) === prim);
  if (!android) list = list.filter((v) => v.localService === true);
  const count = new Map();
  for (const v of list) count.set(normLang(v.lang), (count.get(normLang(v.lang)) || 0) + 1);
  return list.map((v) => {
    const base = langLabel(v.lang, uiLang);
    return { uri: v.voiceURI, lang: normLang(v.lang), label: count.get(normLang(v.lang)) > 1 && v.name ? `${base} — ${v.name}` : base };
  });
}

/** getVoices(), waiting up to `timeoutMs` for `voiceschanged` when the list starts empty. */
export function waitForVoices(synth, { timeoutMs = 1000, setTimer = (f, ms) => setTimeout(f, ms), clearTimer = (h) => clearTimeout(h) } = {}) {
  const now = () => (synth && typeof synth.getVoices === 'function' ? synth.getVoices() : []);
  const first = now();
  if (first.length || !synth) return Promise.resolve(first);
  return new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimer(timer);
      try { synth.removeEventListener?.('voiceschanged', finish); } catch { /* old engines */ }
      resolve(now());
    };
    const timer = setTimer(finish, timeoutMs);
    if (typeof synth.addEventListener === 'function') synth.addEventListener('voiceschanged', finish);
    else synth.onvoiceschanged = finish;
  });
}

/** Split a text into ≤ max-char pieces at sentence, clause, then word boundaries. */
export function splitUtterance(text, max = LOCAL_VOICE.maxChars) {
  const s = String(text || '').trim();
  if (s.length <= max) return s ? [s] : [];
  const out = [];
  let rest = s;
  while (rest.length > max) {
    const win = rest.slice(0, max + 1);
    let cut = -1;
    for (const re of [/[.!?…。！？]\s/g, /[,;:—–、，]\s/g, /\s/g]) {
      let m;
      let last = -1;
      while ((m = re.exec(win))) last = m.index + 1;
      if (last > max * 0.4) { cut = last; break; }
      if (last > cut) cut = last;     // a short boundary is still better than a mid-word cut
    }
    if (cut <= 0) cut = max;
    out.push(rest.slice(0, cut).trim());
    rest = rest.slice(cut).trim();
  }
  if (rest) out.push(rest);
  return out.filter(Boolean);
}

/**
 * The phone-voice speaker.
 *   enqueue(say)          one `say` frame ({gen, seq, text, kind, message_id}); false if dropped
 *   sayEnd({gen, last_seq})  the reply is complete
 *   cancel()              stop now, forget the queue
 *   reset()               a new socket (`ready`): cancel and forget every gen/seq-keyed state
 *   respeak()             restart the current sentence from its start (resume {rewind})
 *   dropOlderThan(gen)    barge_in: cancel and drop every older gen
 *   playedMs()            ms since the current sentence's onstart, clamped to its estimated length
 *   current()             {gen, seq, messageId} of the sentence speaking, or null
 * Callbacks: onDone({gen, seq, kind, messageId, final, why}) after each sentence — `final`
 * means "send playback_done now" (the last sentence after `say_end`, or the queue drained
 * 400 ms ago with no say pending); onFail(reason) once, when the phone's voice gave up
 * (two failures); onSpeaking(bool) for the uplink `local_speaking` flag and the worklet.
 */
export function createLocalVoice({
  synth, rate = 1, voice = null, lang = '', onDone = () => {}, onFail = () => {}, onSpeaking = () => {},
  now = () => Date.now(), setTimer = (f, ms) => setTimeout(f, ms), clearTimer = (h) => clearTimeout(h),
  Utterance = globalThis.SpeechSynthesisUtterance, config = {},
} = {}) {
  if (!synth || typeof synth.speak !== 'function') throw new TypeError('synth (speechSynthesis) is required');
  if (typeof Utterance !== 'function') throw new TypeError('SpeechSynthesisUtterance is required');
  const cfg = { ...LOCAL_VOICE, ...config };
  let queue = [];
  let cur = null;             // {item, piece, token, startedAt}
  let token = 0;
  let failures = 0;
  let cleanRun = 0;
  let failed = false;
  let minGen = null;
  let speaking = false;
  let startTimer = null;
  let endTimer = null;
  let drainTimer = null;
  let deferTimer = null;
  let lastCancelAt = -Infinity;
  let ended = null;           // {gen, lastSeq} from say_end
  let lastDone = null;        // {gen, seq, kind, messageId}
  const finalSent = new Set();

  const setSpeaking = (on) => { if (on !== speaking) { speaking = on; onSpeaking(on); } };
  const clearWatch = () => {
    if (startTimer !== null) clearTimer(startTimer);
    if (endTimer !== null) clearTimer(endTimer);
    startTimer = null; endTimer = null;
  };
  const clearDrain = () => { if (drainTimer !== null) clearTimer(drainTimer); drainTimer = null; };
  const hardCancel = () => {
    token += 1;               // every event of the utterance in flight is now stale
    clearWatch();
    if (deferTimer !== null) { clearTimer(deferTimer); deferTimer = null; }
    try { synth.cancel(); } catch { /* engine gone */ }
    lastCancelAt = now();
  };
  const keyOf = (gen, seq) => `${gen}:${seq}`;
  const sendFinal = (d, why) => {
    const k = keyOf(d.gen, d.seq);
    if (finalSent.has(k)) return;
    finalSent.add(k);
    if (finalSent.size > 256) finalSent.delete(finalSent.values().next().value);
    onDone({ ...d, final: true, why });
  };

  const fail = (reason) => {
    failures += 1;
    cleanRun = 0;
    if (failures >= cfg.failuresToText && !failed) {
      failed = true;
      hardCancel();
      queue = [];
      cur = null;
      setSpeaking(false);
      onFail(reason);
      return true;
    }
    return false;
  };

  const finishItem = (item, spoken) => {
    cur = null;
    const d = { gen: item.gen, seq: item.seq, kind: item.kind, messageId: item.messageId };
    if (spoken) lastDone = d;
    const isLast = ended && ended.gen === item.gen && ended.lastSeq === item.seq;
    onDone({ ...d, final: false, spoken });
    if (isLast) sendFinal(d, 'say_end');
    pump();
  };

  function speakPiece() {
    if (!cur || failed) return;
    const item = cur.item;
    const text = item.pieces[cur.piece];
    const my = ++token;
    cur.token = my;
    const u = new Utterance(text);
    // BCP 47 for the Web Speech API: Android lists its voices as 'en_US'.
    if (voice) { u.voice = voice; u.lang = normLang(voice.lang) || lang; } else if (lang) u.lang = lang;
    u.rate = rate;
    u.pitch = 1;
    u.onstart = () => {
      if (my !== token || !cur) return;
      if (startTimer !== null) { clearTimer(startTimer); startTimer = null; }
      if (cur.piece === 0 && cur.startedAt === null) cur.startedAt = now();
      setSpeaking(true);
      const ms = text.length / cfg.endCharsPerSec / Math.max(0.5, rate) * 1000 + cfg.endPadMs;
      endTimer = setTimer(() => {
        endTimer = null;
        if (my !== token) return;
        hardCancel();
        if (!fail('timeout')) finishItem(item, false);
      }, ms);
    };
    u.onend = () => {
      if (my !== token || !cur) return;
      clearWatch();
      if (cur.piece + 1 < item.pieces.length) { cur.piece += 1; speakPiece(); return; }
      if (failures > 0 && ++cleanRun >= cfg.cleanToForgive) { failures = 0; cleanRun = 0; }
      finishItem(item, true);
    };
    u.onerror = (e) => {
      if (my !== token || !cur) return;
      clearWatch();
      const err = (e && e.error) || 'synthesis-failed';
      if (err === 'interrupted' || err === 'canceled') { finishItem(item, false); return; }
      token += 1;
      if (!fail(err)) finishItem(item, false);
    };
    startTimer = setTimer(() => {
      startTimer = null;
      if (my !== token) return;
      hardCancel();
      if (!fail('no-start')) finishItem(item, false);
    }, cfg.startTimeoutMs);
    const go = () => { deferTimer = null; if (my === token) synth.speak(u); };
    if (now() - lastCancelAt < cfg.afterCancelMs) deferTimer = setTimer(go, cfg.afterCancelMs);
    else go();
  }

  function pump() {
    if (cur || failed) return;
    const item = queue.shift();
    if (!item) {
      setSpeaking(false);
      clearDrain();
      if (lastDone) {
        const d = lastDone;
        drainTimer = setTimer(() => { drainTimer = null; if (!cur && !queue.length) sendFinal(d, 'drained'); }, cfg.drainMs);
      }
      return;
    }
    clearDrain();
    cur = { item, piece: 0, token: 0, startedAt: null };
    speakPiece();
  }

  return {
    get failed() { return failed; },
    get speaking() { return speaking; },
    get pending() { return queue.length + (cur ? 1 : 0); },
    setRate(r) { if (Number.isFinite(r) && r > 0) rate = r; },
    setVoice(v) { voice = v || null; },
    enqueue(say) {
      if (failed || !say) return false;
      const gen = (Number(say.gen) || 0) & 0xffff;
      if (minGen !== null && genOlder(gen, minGen)) return false;
      const pieces = splitUtterance(say.text, cfg.maxChars);
      if (!pieces.length) return false;
      queue.push({ gen, seq: say.seq >>> 0, kind: say.kind || 'reply', messageId: say.message_id ?? null, text: String(say.text), pieces });
      clearDrain();
      pump();
      return true;
    },
    sayEnd({ gen, last_seq: lastSeq } = {}) {
      ended = { gen: (Number(gen) || 0) & 0xffff, lastSeq: lastSeq >>> 0 };
      // The reply's last sentence may already be done.
      if (!cur && !queue.length && lastDone && lastDone.gen === ended.gen && lastDone.seq === ended.lastSeq) {
        clearDrain();
        sendFinal(lastDone, 'say_end');
      }
    },
    cancel() {
      hardCancel();
      queue = [];
      cur = null;
      clearDrain();
      setSpeaking(false);
    },
    /**
     * A new socket (`ready`): seq restarts at 1 and gen at 0, so everything keyed on them
     * — the reply's `say_end`, the last spoken sentence, the sent finals, the barge-in gen —
     * is forgotten, and the queue is cancelled. Call it beside claims.reset()/player.reset().
     * The failure count stays: it is about the phone's engine, not the socket.
     */
    reset() {
      hardCancel();
      queue = [];
      cur = null;
      clearDrain();
      ended = null;
      lastDone = null;
      finalSent.clear();
      minGen = null;
      setSpeaking(false);
    },
    respeak() {
      if (!cur || failed) return false;
      hardCancel();
      cur.piece = 0;
      cur.startedAt = null;
      speakPiece();
      return true;
    },
    dropOlderThan(gen) {
      const g = (Number(gen) || 0) & 0xffff;
      minGen = minGen === null || genNewer(g, minGen) ? g : minGen;
      queue = queue.filter((it) => !genOlder(it.gen, minGen));
      if (cur && genOlder(cur.item.gen, minGen)) {
        hardCancel();
        cur = null;
        pump();
      }
    },
    playedMs() {
      if (!cur || cur.startedAt === null) return 0;
      const est = cur.item.text.length / (cfg.heardCharsPerSec * Math.max(0.5, rate)) * 1000;
      return Math.round(Math.max(0, Math.min(est, now() - cur.startedAt)));
    },
    current() { return cur ? { gen: cur.item.gen, seq: cur.item.seq, messageId: cur.item.messageId } : null; },
  };
}
