// DisPatch Voice: seq-claims.js — who plays each sentence (v3 contracts C-§E.2, C.5 #1, #9).
//
// A sentence (`seq`) can reach the listener on three paths: the server's audio,
// the phone's own voice (speechSynthesis) or captions only. The first path to
// claim a sentence wins and every later arrival for it is dropped, so a sentence
// is never heard twice (a reply replayed after a reconnect, a `say` that the
// phone already spoke when the late audio lands, ...).
//
// Claims are keyed by (sid, seq). `seq` restarts at 1 in every server
// VoiceSession, and `ready.sid` names the session, so a new session's seq 1 is
// never mistaken for the old session's (hardening M2: every reconnect lost its
// first sentence otherwise). The client calls reset() on every `ready`;
// cross-session dedup is `start.after_partial`'s job, never this module's.
//
// Owner: track T8. Pure, no DOM; node --test-able.

/** The three paths a sentence can take. */
export const CLAIM_PATHS = Object.freeze(['server', 'phone', 'text']);

/** Claims kept before the oldest are forgotten (a reply rarely has 50 sentences). */
const DEFAULT_LIMIT = 512;

/**
 * @param {{limit?: number}} [opts]
 * @returns {{
 *   claim(sid: string, seq: number, path: 'server'|'phone'|'text'): boolean,
 *   owner(sid: string, seq: number): string|null,
 *   reset(): void,
 *   readonly size: number,
 * }}
 */
export function createSeqClaims({ limit = DEFAULT_LIMIT } = {}) {
  /** Map preserves insertion order, which is what the bound evicts by. */
  let claims = new Map();
  const key = (sid, seq) => `${sid == null ? '' : String(sid)}\u0000${Number(seq) >>> 0}`;

  return {
    /**
     * True when `path` owns (sid, seq): either it just claimed it, or it had
     * already (every chunk of one sentence re-claims on the same path).
     * False when another path got there first: drop the arrival.
     */
    claim(sid, seq, path) {
      if (!CLAIM_PATHS.includes(path)) throw new TypeError(`unknown claim path: ${path}`);
      const k = key(sid, seq);
      const held = claims.get(k);
      if (held !== undefined) return held === path;
      claims.set(k, path);
      if (claims.size > limit) {
        // Forget the oldest; a claim that old belongs to a long-finished reply.
        const first = claims.keys().next().value;
        claims.delete(first);
      }
      return true;
    },
    /** The path that owns (sid, seq), or null. */
    owner(sid, seq) {
      const held = claims.get(key(sid, seq));
      return held === undefined ? null : held;
    },
    /** Called on every `ready` (C.4): a new server session starts its own seq count. */
    reset() { claims = new Map(); },
    get size() { return claims.size; },
  };
}
