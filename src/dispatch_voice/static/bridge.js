// DisPatch Voice: the client app's view of the audio & link modules.
// Owner: track T7 (this file); the modules it reaches are track T8's.
// Contract: C-§E.2 (module APIs and worklet message formats, frozen).
//
// T7 builds the screens against the E.2 APIs; T8 builds the modules. This
// bridge is the one seam between them: it imports each T8 module (with the
// same `?v=` the entry was loaded with, so every file is one immutable
// fetch and one module instance) and hands out the contract names. While a
// module is still a day-0 placeholder, or a browser cannot load it, the
// names resolve to small fallbacks that keep today's behaviour (or to null,
// meaning "the feature is not there yet" — callers check). Nothing here
// changes a message format; nothing outside this file imports a T8 module.
//
// The two worklet files are imported here too, for their main-thread helpers
// (createPlayerClient, createOptimisticDuck, heardPlayedMs, createEchoBias,
// isBluetoothOutput, gateFromReady): they register their processor only
// where `registerProcessor` exists, so the main-thread import has no side
// effect (Amendment T8-1). addModule() gets their URLs from assetUrl().

const Q = new URL(import.meta.url).search;

async function load(name) {
  try { return await import(`./${name}${Q}`); } catch { return {}; }
}

const [linkMod, localMod, opusMod, ringMod, claimsMod, playerMod, micMod] = await Promise.all([
  load('link-monitor.js'), load('local-voice.js'), load('opus-codec.js'),
  load('uplink-ring.js'), load('seq-claims.js'),
  load('voice-player-worklet.js'), load('voice-worklet.js'),
]);
const fn = (mod, name) => (typeof mod[name] === 'function' ? mod[name] : null);

/** URL of a sibling asset (worklet modules), carrying the entry's `?v=`. */
export function assetUrl(name) {
  return new URL(`./${name}${Q}`, import.meta.url).href;
}

// ------------------------------------------------------------ seq claims ---

/**
 * Fallback for seq-claims.js (E.2): the first of server | phone | text to
 * claim a (sid, seq) wins; reset() on every `ready` (C.5 #9).
 */
export function fallbackSeqClaims() {
  let map = new Map();
  return {
    claim(sid, seq, path) {
      const k = `${sid || ''}:${seq}`;
      if (map.has(k)) return map.get(k) === path;
      map.set(k, path);
      if (map.size > 4096) map = new Map([...map].slice(-2048));
      return true;
    },
    reset() { map = new Map(); },
  };
}

export const createSeqClaims = typeof claimsMod.createSeqClaims === 'function'
  ? claimsMod.createSeqClaims : fallbackSeqClaims;

// ----------------------------------------------------------- phone voice ---

/** "en-US" / "en_US" → "en-US"; junk → "". */
function normLang(l) {
  return String(l || '').replace(/_/g, '-').trim();
}

/**
 * Fallback for local-voice.js pickVoice (E.2): match the UI language's
 * primary subtag, prefer an exact `lang`, honour `preferredURI`; the
 * `localService` filter applies on desktop only (Chrome Android lists one
 * entry per locale and marks every one local — feasibility H2).
 */
export function fallbackPickVoice(voices, lang, preferredURI = '', { android = false } = {}) {
  const list = Array.isArray(voices) ? voices.filter(Boolean) : [];
  const usable = android ? list : list.filter((v) => v.localService === true);
  if (preferredURI) {
    const hit = usable.find((v) => v.voiceURI === preferredURI);
    if (hit) return hit;
  }
  const want = normLang(lang).toLowerCase();
  const primary = want.split('-')[0];
  if (!primary) return null;
  const same = usable.filter((v) => normLang(v.lang).toLowerCase().split('-')[0] === primary);
  return same.find((v) => normLang(v.lang).toLowerCase() === want)
    || same.find((v) => v.default) || same[0] || null;
}

export const pickVoice = typeof localMod.pickVoice === 'function' ? localMod.pickVoice : fallbackPickVoice;

/** createLocalVoice from T8, or null until it lands (no phone voice yet). */
export const createLocalVoice = typeof localMod.createLocalVoice === 'function' ? localMod.createLocalVoice : null;

// ------------------------------------------------------------------ link ---

/** createLinkMonitor from T8, or null (no link monitoring yet). */
export const createLinkMonitor = typeof linkMod.createLinkMonitor === 'function' ? linkMod.createLinkMonitor : null;
export const LINK_THRESHOLDS = linkMod.LINK_THRESHOLDS || null;

// ------------------------------------------------------------- transport ---

export const opusSupport = typeof opusMod.opusSupport === 'function'
  ? opusMod.opusSupport : () => ({ up: false, down: false });
export const createOpusUplink = typeof opusMod.createOpusUplink === 'function' ? opusMod.createOpusUplink : null;
export const createOpusDownlink = typeof opusMod.createOpusDownlink === 'function' ? opusMod.createOpusDownlink : null;
export const wire3 = opusMod.wire3 || null;
export const createUplinkRing = typeof ringMod.createUplinkRing === 'function' ? ringMod.createUplinkRing : null;
/** The wrap-aware reconnect cursor (u32 sample clock), when T8's ring provides it. */
export const ringCursorFrom = typeof ringMod.cursorFrom === 'function' ? ringMod.cursorFrom : null;

/** createUplink/createDownlink take the negotiated codec ('opus' | 'pcm16'); the E.2 names force Opus. */
export const createUplink = fn(opusMod, 'createUplink');
export const createDownlink = fn(opusMod, 'createDownlink');
export const codecQuery = fn(opusMod, 'codecQuery');
export const createReplayPump = fn(ringMod, 'createReplayPump');
export const linkFrame = fn(linkMod, 'linkFrame');
/** The one monotonic clock the downlink and the link monitor share (Amendment T8-3). */
export const monoNow = fn(linkMod, 'monoNow') || fn(opusMod, 'monoNow')
  || (() => (globalThis.performance ? globalThis.performance.timeOrigin + globalThis.performance.now() : Date.now()));

// ------------------------------------------------- worklet main-thread helpers ---

export const createPlayerClient = fn(playerMod, 'createPlayerClient');
export const createOptimisticDuck = fn(playerMod, 'createOptimisticDuck');
export const heardPlayedMs = fn(playerMod, 'heardPlayedMs');
export const createEchoBias = fn(micMod, 'createEchoBias');
export const isBluetoothOutput = fn(micMod, 'isBluetoothOutput');
export const gateFromReady = fn(micMod, 'gateFromReady');

/** Which T8 pieces are real (the Voice lab and tests read this). */
export const available = Object.freeze({
  seqClaims: typeof claimsMod.createSeqClaims === 'function',
  pickVoice: typeof localMod.pickVoice === 'function',
  localVoice: !!createLocalVoice,
  linkMonitor: !!createLinkMonitor,
  opus: !!(createOpusUplink && createOpusDownlink && wire3),
  ring: !!createUplinkRing,
  player: !!createPlayerClient,
  duck: !!createOptimisticDuck,
});
