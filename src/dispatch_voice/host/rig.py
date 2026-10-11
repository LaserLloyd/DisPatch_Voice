"""The optional GPU rig: a SpeechForge server over its ``sf.1`` session protocol.

**The box decides, the rig computes.** Turn-taking, speculation, the barge-in
classifier, acknowledgements and heard tracking stay on the box; the rig only
recognises speech, scores end-of-turn and renders speech. The two adapters that
use it sit behind the session's existing seams: ``rt/stt_rig.py``
(``RigStreamingSTT``, a ``StreamingSTT``) and ``host/engines/rig_tts.py``
(``ReplyEngine``, a ``TTSEngine`` with one rig context per reply).

**Off unless set up.** ``rig.enabled`` in ``<data>/voice.yaml`` only takes
effect when ``rig.url`` is set AND the ``SPEECHFORGE_TOKEN`` is present (the
environment, else ``~/.config/local-chat/speechforge.env``, mode 0600). With no
rig, :func:`open_for_session` returns None and the session is exactly the CPU
one. The token is read at each voice-chat open (a rotation needs no restart),
sent only in the upgrade header, never logged, never shown to the browser and
never stored in ``voice.yaml``.

**Never slower than the CPU.** A turn never STARTS on a rig that is not proven:
``ready`` received, a pong answered after the last failure, the lease held (or the
rig serving unleased: SpeechForge loaded without a StudioForge lease), the models
not loading, no lease being vacated, and no failure in the last 30 s. While the rig is down, unproven or cooling down,
every turn runs the CPU path with no deadline and no hedge. The rig is declared
dead in under a second (3 missed pongs at 250 ms, or 1.0 s without any frame,
or partial lag over 800 ms for 1 s while speech flows, or a close, a fatal
error or a lost lease); what was in flight fails over at once (the mic ring
replays into the CPU recogniser; the current sentence re-renders on the CPU
and the rest of that reply stays there). Re-promotion is a probe BETWEEN turns
(never a polled cache), and never within 30 s of the last failure.

Wire contract: SpeechForge ``docs/PROTOCOL.md`` (sf.1). The constants below
mirror ``speechforge/protocol.py`` (read, not imported: the rig's package is not
a dependency of the add-on); ``tests/rig/test_sf1_contract.py`` pins them.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import queue
import re
import secrets
import stat
import struct
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

log = logging.getLogger("dispatch.voice")

# --------------------------------------------------------------------------- #
# sf.1 constants (mirrors speechforge/protocol.py; pinned by the contract test)
# --------------------------------------------------------------------------- #

PROTO = "sf.1"
MAGIC = 0x53
KIND_MIC = 0x01
KIND_TTS = 0x02
HEADER = struct.Struct("<BBHHIIH")
HEADER_SIZE = HEADER.size                 # 16
MIC_RATE = 16000
TTS_RATE = 24000
MIC_FRAME_SAMPLES = 320                   # 20 ms
TTS_FRAME_SAMPLES = 960                   # 40 ms
MAX_MESSAGE_BYTES = 64 * 1024
U16 = 0xFFFF
U32 = 0xFFFFFFFF

# --------------------------------------------------------------------------- #
# Health and fallback numbers (ADDON-ENGINE §2-§3, SPEC-speechforge §4.3/§6)
# --------------------------------------------------------------------------- #

PING_S = 0.25                 # ping{t} cadence
MISSED_PONGS = 3              # this many unanswered pings: down
SILENCE_S = 1.0               # this long without ANY frame from the rig: down
LAG_MS = 800                  # stt.partial.lag_ms above this ...
LAG_FOR_S = 1.0               # ... for this long while speech flows: down
SPEECH_FLOWING_S = 0.3        # mic audio sent this recently = speech is flowing
PONG_FRESH_S = 1.0            # "pongs current": the last pong is at most this old
COOLDOWN_S = 30.0             # never promoted within this long of the last failure
RESUME_WINDOW_S = 10.0        # the rig keeps a dropped session's slot this long
RESUME_ATTEMPTS = 2           # failed reconnects before waiting for the between-turn probe
READY_WAIT_S = 25.0           # hello -> ready (the rig waits up to 20 s for its models)
OPEN_TIMEOUT_S = 5.0
RATE_LIMIT_BACKOFF_S = 60.0   # a 429
IDLE_PROBE_EVERY_S = 1.0      # the between-turn probe check while nothing is going on
#: A turn that started on the CPU while the rig was still connecting takes the power
#: hold; so does a session whose rig is not ready this long after the open.
HOLD_GRACE_S = 2.0
#: ``credit{played_ms}`` cadence while a context is open (the ping cadence). The rig
#: freezes ``played_ms`` at the last report, so the report must stay fresh.
CREDIT_S = 0.25
#: The late clock's allowance on top of the rig's own credit rule: the rig gates
#: every LM step (40 ms of audio per token) and the first audio of a unit needs a
#: whole first chunk, plus the 3 lookahead tokens of the unit before it.
TOKEN_MS = 40
LOOKAHEAD_TOKENS = 3
CREDIT_MARGIN_MS = 100
#: Two ``busy`` errors in a row on a context: the rig is treated as down (cooldown).
BUSY_STRIKES = 2
#: ``ready.limits.max_text`` when the rig does not say (SpeechForge's default).
MAX_TEXT_DEFAULT = 2000

TOKEN_ENV = "SPEECHFORGE_TOKEN"

#: Down / fallback reasons (machine codes; the Voice lab translates them).
REASONS = ("missed_pongs", "silence", "lag", "closed", "lease_lost", "fatal_error",
           "final_timeout", "tts_late", "tts_failed", "unknown_voice", "bad_voice",
           "busy", "rate_limited", "not_ready", "unreachable", "token", "vacating")
#: Lease states a NEW turn may start on. ``unleased``: the rig loaded without a
#: StudioForge lease (its lease mode is off, or StudioForge was down and the card was
#: free) and serves; the Voice lab says so.
SERVING_LEASES = ("held", "unleased")


# --------------------------------------------------------------------------- #
# Settings (voice.yaml `rig:`)
# --------------------------------------------------------------------------- #

#: (lo, hi) for the numeric keys of ``rig:``.
RIG_RANGES: dict[str, tuple[int, int]] = {
    "stt_final_deadline_ms": (100, 1000),
    "tts_start_deadline_ms": (200, 2000),
    "ahead_ms": (1000, 10000),
    "first_chunk_tokens": (4, 100),
}
URL_MAX = 256


@dataclass(frozen=True)
class RigSettings:
    """``voice.yaml`` ``rig:`` (all optional; the defaults are ADDON-ENGINE §1's).

    ``url`` has no default: an address is a fact about one install, not a
    default of a public package. The switch is off until the owner turns it on."""
    enabled: bool = False
    url: str = ""                         # ws://<rig>:1236/v1/realtime
    stt_final_deadline_ms: int = 350      # stt.flush -> stt.final, then the CPU tail (rig p99 172 + RTT 80 + margin)
    tts_start_deadline_ms: int = 800      # tts.text -> first audio, then that reply on the CPU (rig TTFA max ~520 + RTT + margin)
    ahead_ms: int = 3000                  # credit: audio rendered past what the phone played
    first_chunk_tokens: int = 15          # the rig's first TTS chunk

    @classmethod
    def from_mapping(cls, raw) -> RigSettings:
        if not isinstance(raw, dict):
            return cls()
        return replace(cls(), **clean_rig(raw))

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def active(self, token: str) -> bool:
        """The rig is in play for a voice chat: switched on, an address, a token."""
        return bool(self.enabled and self.url and valid_url(self.url) and token)


def valid_url(url: str) -> bool:
    """``ws://`` / ``wss://`` with a host, no credentials, no query or fragment (the
    token travels in a header, never in a URL that lands in logs)."""
    if not isinstance(url, str) or not url or len(url) > URL_MAX or any(c.isspace() for c in url):
        return False
    try:
        u = urlsplit(url)
        _ = u.port                                   # raises on a bad port
    except ValueError:
        return False
    return (u.scheme in ("ws", "wss") and bool(u.hostname) and not u.username and not u.password
            and not u.query and not u.fragment)


def clean_rig(raw) -> dict:
    """The ``rig:`` subtree, validated: unknown keys dropped, a number outside its
    range clamped, a wrong type or a bad URL dropped (the default applies)."""
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    if isinstance(raw.get("enabled"), bool):
        out["enabled"] = raw["enabled"]
    url = raw.get("url")
    if url == "" or url is None and "url" in raw:
        out["url"] = ""
    elif isinstance(url, str) and valid_url(url.strip()):
        out["url"] = url.strip()
    for k, (lo, hi) in RIG_RANGES.items():
        v = raw.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = int(min(hi, max(lo, round(float(v)))))
    return out


def http_base(url: str) -> str:
    """``ws://h:p/v1/realtime`` -> ``http://h:p`` (the voice routes of the same server)."""
    u = urlsplit(url)
    return f"{'https' if u.scheme == 'wss' else 'http'}://{u.netloc}"


def host_of(url: str) -> str:
    try:
        return urlsplit(url).netloc if url else ""
    except ValueError:
        return ""


# --------------------------------------------------------------------------- #
# The token (never logged, never sent to the browser, never in voice.yaml)
# --------------------------------------------------------------------------- #

def token_path() -> Path:
    return Path.home() / ".config" / "local-chat" / "speechforge.env"


@dataclass(frozen=True)
class Token:
    value: str = ""
    source: str | None = None       # "env" | "file" | None
    problem: str | None = None      # why the file was not used (no value in it)

    def __repr__(self) -> str:      # never the value
        return f"Token(source={self.source!r}, set={bool(self.value)}, problem={self.problem!r})"


_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def read_token() -> Token:
    """``SPEECHFORGE_TOKEN`` from the environment, else from the env file. A file
    that other users can read is refused (and named, by path, in the status)."""
    env = (os.environ.get(TOKEN_ENV) or "").strip()
    if env:
        return Token(env, "env")
    p = token_path()
    try:
        st = p.stat()
    except OSError:
        return Token()
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        log.warning("voice: %s is readable by other users; the rig token in it is ignored "
                    "(chmod 600 it)", p)
        return Token(problem="permissions")
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return Token(problem="unreadable")
    for line in text.splitlines():
        m = _ENV_LINE.match(line)
        if m and m.group(1) == TOKEN_ENV:
            v = m.group(2)
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            v = v.strip()
            if v:
                return Token(v, "file")
    return Token(problem="no_token")


def _fingerprint(url: str, token: str) -> str:
    return hashlib.sha256(f"{url}\0{token}".encode()).hexdigest()[:16]


#: url+token fingerprints the rig refused with 401: no retry until either changes.
_MISCONFIGURED: set[str] = set()


# --------------------------------------------------------------------------- #
# Binary frames
# --------------------------------------------------------------------------- #

def pack_audio(kind: int, ctx: int, gen: int, seq: int, pts: int, pcm16: np.ndarray) -> bytes:
    pcm = np.asarray(pcm16, dtype="<i2")
    n = int(pcm.shape[0])
    if n > U16:
        raise ValueError("too many samples for one frame")
    return HEADER.pack(MAGIC, kind, ctx & U16, gen & U16, seq & U32, pts & U32, n) + pcm.tobytes()


def unpack_audio(buf: bytes) -> tuple[int, int, int, int, int, np.ndarray]:
    """(kind, ctx, gen, seq, pts, pcm16). ValueError on a malformed frame."""
    if len(buf) < HEADER_SIZE:
        raise ValueError("binary frame shorter than the 16-byte header")
    magic, kind, ctx, gen, seq, pts, n = HEADER.unpack_from(buf)
    if magic != MAGIC:
        raise ValueError("bad magic")
    if kind not in (KIND_MIC, KIND_TTS):
        raise ValueError(f"unknown binary kind {kind}")
    if len(buf) != HEADER_SIZE + 2 * n:
        raise ValueError("sample count does not match the frame length")
    return kind, ctx, gen, seq, pts, np.frombuffer(buf, dtype="<i2", offset=HEADER_SIZE, count=n)


def dumps(msg: dict) -> str:
    return json.dumps(msg, separators=(",", ":"), allow_nan=False)


def to_int16(pcm) -> np.ndarray:
    a = np.asarray(pcm)
    if a.dtype == np.int16:
        return a.reshape(-1)
    a = np.clip(a.astype(np.float32).reshape(-1), -1.0, 1.0)
    return (a * 32767.0).astype("<i2")


# --------------------------------------------------------------------------- #
# Counters (the Voice lab's rig line)
# --------------------------------------------------------------------------- #

def _pct(xs, q: float) -> int | None:
    if not xs:
        return None
    s = sorted(xs)
    return int(round(s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]))


class RigMetrics:
    """``rig {turns, stt, tts, fell_back, last_fallback_reason, ttfa_p50/p90,
    partial_lag_p50/p90, ping_rtt}`` — process-wide, since the add-on started."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.turns = 0            # utterances that started (rig or CPU) while the rig was in play
        self.stt = 0              # ... of them recognised on the rig to the end
        self.replies = 0          # replies that started while the rig was in play
        self.tts = 0              # ... of them spoken on the rig
        self.fell_back = 0
        self.last_fallback_reason: str | None = None
        self.last_fallback_at: float | None = None
        self.ttfa: deque[float] = deque(maxlen=200)
        self.lag: deque[float] = deque(maxlen=400)
        self.ping_rtt: float | None = None

    def add(self, name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + n)

    def fallback(self, reason: str) -> None:
        with self._lock:
            self.fell_back += 1
            self.last_fallback_reason = reason
            self.last_fallback_at = time.time()

    def observe(self, name: str, v: float) -> None:
        with self._lock:
            getattr(self, name).append(float(v))

    def rtt(self, ms: float) -> None:
        with self._lock:
            self.ping_rtt = ms if self.ping_rtt is None else round(0.8 * self.ping_rtt + 0.2 * ms, 1)

    def snapshot(self) -> dict:
        with self._lock:
            return {"turns": self.turns, "stt": self.stt, "replies": self.replies, "tts": self.tts,
                    "fell_back": self.fell_back, "last_fallback_reason": self.last_fallback_reason,
                    "last_fallback_at": self.last_fallback_at,
                    "ttfa_p50": _pct(self.ttfa, 0.5), "ttfa_p90": _pct(self.ttfa, 0.9),
                    "partial_lag_p50": _pct(self.lag, 0.5), "partial_lag_p90": _pct(self.lag, 0.9),
                    "ping_rtt": None if self.ping_rtt is None else round(self.ping_rtt, 1),
                    "share": (round(100 * self.stt / self.turns) if self.turns else None)}

    def reset(self) -> None:
        self.__init__()


METRICS = RigMetrics()


# --------------------------------------------------------------------------- #
# The power hold, rig-aware (ADDON-ENGINE §3 "Power hold")
# --------------------------------------------------------------------------- #

class RigPowerGate:
    """The session's power-profile hold when a rig is in play: NOT taken while the
    rig serves; taken at the first CPU failover (or a turn that starts on the CPU,
    or a rig that is not ready ``HOLD_GRACE_S`` after the open); released when the
    rig is promoted again. Same ``start()`` / ``release()`` shape as ws._Hold
    (``release`` is blocking: the clean-up runs it off the loop)."""

    def __init__(self, settings, *, power=None) -> None:
        self._settings = settings
        self._power = power
        self._lock = threading.Lock()
        self._held = False
        self._closed = False
        self._pending: threading.Thread | None = None

    def _mod(self):
        if self._power is None:
            from . import power
            self._power = power
        return self._power

    @property
    def held(self) -> bool:
        return self._held

    def start(self) -> None:          # ws.serve calls it where the CPU hold would start
        return None

    def need(self) -> None:
        """The session is (or is about to be) on the CPU: hold, on a thread."""
        self._switch(True)

    def not_needed(self) -> None:
        """The rig serves again: give the hold back, on a thread."""
        self._switch(False)

    def _switch(self, on: bool) -> None:
        with self._lock:
            if self._closed or self._held == on:
                return
            self._held = on
            prev = self._pending

            def run() -> None:
                if prev is not None:
                    prev.join(10.0)
                try:
                    if on:
                        self._mod().acquire(self._settings)
                    else:
                        self._mod().release()
                except Exception:
                    log.exception("voice: power hold %s failed", "acquire" if on else "release")

            t = threading.Thread(target=run, name="voice-rig-power", daemon=True)
            self._pending = t
        t.start()

    def release(self) -> None:
        with self._lock:
            self._closed = True
            held, self._held = self._held, False
            pending = self._pending
        if pending is not None:
            pending.join(10.0)
        if held:
            try:
                self._mod().release()
            except Exception:
                log.exception("voice: power hold release failed")


# --------------------------------------------------------------------------- #
# Voices: conditionals only, never the reference clip (D3)
# --------------------------------------------------------------------------- #

_SHA_CACHE: dict[str, tuple[float, int, str]] = {}


def file_sha256(path: Path) -> str:
    st = path.stat()
    key = str(path)
    hit = _SHA_CACHE.get(key)
    if hit is not None and hit[0] == st.st_mtime and hit[1] == st.st_size:
        return hit[2]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    _SHA_CACHE[key] = (st.st_mtime, st.st_size, h.hexdigest())
    return h.hexdigest()


def voice_file(profile_id: str | None) -> Path | None:
    """The ``voice.npz`` (Chatterbox conditionals) a profile id speaks with on the CPU,
    or None when the rig cannot speak it (a built-in voice, no profile store, no
    chosen and no default profile). Never the reference clip."""
    try:
        from .tts import builtin_id
        if builtin_id(profile_id):
            return None
        from . import profiles_bridge as pb
        from .engines import chatterbox
        if not chatterbox.available():
            return None
        st = pb.store()
        pid = profile_id or st.default_id()
        if not pid:
            return None
        st.get(pid)                                  # KeyError / ProfileError: unknown
        p = Path(st.root) / pid / "voice.npz"
        return p if p.is_file() else None
    except Exception:
        return None


#: Test seam: the httpx transport VoiceSync uses when none is passed (None = the network).
HTTP_TRANSPORT = None


class VoiceSync:
    """``HEAD /v1/voices/<sha>`` then ``PUT`` on a 404, over HTTP to the same server
    (tailnet), with the same token; out of band, so it never competes with audio.
    ``transport`` is a test seam (an ``httpx`` transport)."""

    def __init__(self, url: str, token: str, *, transport=None, timeout_s: float = 15.0) -> None:
        self.base = http_base(url)
        self._token = token
        self._transport = transport
        self.timeout_s = timeout_s
        self.state: dict[str, str] = {}          # sha -> ready | syncing | bad | error
        self._locks: dict[str, asyncio.Lock] = {}

    def _client(self):
        import httpx
        return httpx.AsyncClient(timeout=self.timeout_s, trust_env=False,
                                 transport=self._transport if self._transport is not None else HTTP_TRANSPORT,
                                 headers={"Authorization": f"Bearer {self._token}"})

    async def ensure(self, sha: str, path: Path, *, force: bool = False) -> str:
        """``ready`` | ``bad`` (the rig refused the conditionals: CPU for the session) |
        ``token`` (401) | ``error``."""
        lock = self._locks.setdefault(sha, asyncio.Lock())
        async with lock:
            cur = self.state.get(sha)
            if cur in ("ready", "bad") and not force:
                return cur
            self.state[sha] = "syncing"
            try:
                async with self._client() as c:
                    if not force:
                        r = await c.head(f"{self.base}/v1/voices/{sha}")
                        if r.status_code == 200:
                            self.state[sha] = "ready"
                            return "ready"
                        if r.status_code == 401:
                            self.state[sha] = "error"
                            return "token"
                    body = await asyncio.to_thread(path.read_bytes)
                    if hashlib.sha256(body).hexdigest() != sha:
                        self.state[sha] = "error"          # the file changed under us: next session
                        return "error"
                    r = await c.put(f"{self.base}/v1/voices/{sha}", content=body,
                                    headers={"Content-Type": "application/octet-stream"})
                    if r.status_code in (200, 201, 204):
                        self.state[sha] = "ready"
                        log.info("voice: rig voice %s uploaded (%d bytes of conditionals)", sha[:8], len(body))
                    elif r.status_code == 401:
                        self.state[sha] = "error"
                        return "token"
                    elif r.status_code in (400, 413, 422):
                        self.state[sha] = "bad"
                    else:
                        self.state[sha] = "error"
            except Exception as e:  # noqa: BLE001 - a failed sync keeps that voice on the CPU
                log.info("voice: rig voice sync of %s failed (%s)", sha[:8], type(e).__name__)
                self.state[sha] = "error"
            log.info("voice: rig voice %s: %s", sha[:8], self.state[sha])
            return self.state[sha]


def _delete_target() -> tuple[str, str] | None:
    """(url, token) when a rig address and token are set, else None."""
    try:
        from .config import load_settings
        rs = getattr(load_settings(), "rig", None)
        tok = read_token()
        if rs is None or not rs.url or not valid_url(rs.url) or not tok.value:
            return None
        return rs.url, tok.value
    except Exception:
        return None


def sha_for_delete(store, pid: str) -> str | None:
    """Before a profile is deleted: the sha256 of its conditionals, when a rig is set
    up (else None, and nothing is read)."""
    if _delete_target() is None:
        return None
    try:
        store.get(pid)
        p = Path(store.root) / pid / "voice.npz"
        return file_sha256(p) if p.is_file() else None
    except Exception:
        return None


def delete_voice_async(sha: str) -> None:
    """Profile deleted on the box -> ``DELETE /v1/voices/<sha>`` (deletion sync,
    SPEC §8.4). Whenever a rig address and token are set (even with the switch off:
    an earlier session may have stored the voice there). Fire and forget."""
    target = _delete_target()
    if target is None:
        return
    url, token = target

    def run() -> None:
        try:
            import httpx
            r = httpx.delete(f"{http_base(url)}/v1/voices/{sha}", timeout=10.0, trust_env=False,
                             headers={"Authorization": f"Bearer {token}"})
            log.info("voice: rig voice %s deleted (HTTP %s)", sha[:8], r.status_code)
        except Exception as e:  # noqa: BLE001
            log.info("voice: rig voice %s delete failed (%s)", sha[:8], type(e).__name__)

    threading.Thread(target=run, name="voice-rig-delete", daemon=True).start()


# --------------------------------------------------------------------------- #
# The connection: one sf.1 socket per voice chat
# --------------------------------------------------------------------------- #

class RigRefused(Exception):
    def __init__(self, reason: str, retry_after_s: float = 0.0) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after_s = retry_after_s


@dataclass
class TtsCtx:
    """One rig TTS context (one reply, one voice) as the box sees it."""
    ctx: int
    gen: int
    voice: str
    epoch: int
    q: queue.Queue = field(default_factory=queue.Queue)
    samples: int = 0                  # ctx audio received (24 kHz samples)
    started: bool = False
    first_frame_at: float | None = None
    closed: bool = False
    cancelled: bool = False
    paused_since: float | None = None
    paused_total: float = 0.0
    played_ms: float | None = None   # the phone's last report, in the ctx clock
    played_at: float | None = None
    paused_at_report: float = 0.0     # paused_s() when that report came
    last_credit_at: float = float("-inf")
    #: The last ``played_ms`` SENT to the rig (None: none yet, the rig assumes real time
    #: from its first frame). The rig freezes its idea of playback at this value until
    #: the next credit, so the late clock reasons from it, never from an estimate.
    sent_played_ms: float | None = None
    text_chars: int = 0               # tts.text characters sent on this ctx (limits.max_text)
    first_chunk_tokens: int = 15

    @property
    def audio_ms(self) -> float:
        return self.samples * 1000.0 / TTS_RATE

    def paused_s(self, now: float) -> float:
        return self.paused_total + (now - self.paused_since if self.paused_since is not None else 0.0)

    def played_est_ms(self, now: float) -> float:
        """How much of this ctx the phone has played: its last report plus real time
        since (minus pauses), never past the audio it has; without a report, real time
        from the first frame (the rig's own assumption)."""
        if self.played_ms is not None and self.played_at is not None:
            run = max(0.0, (now - self.played_at) - (self.paused_s(now) - self.paused_at_report))
            return min(self.played_ms + run * 1000.0, max(self.played_ms, self.audio_ms))
        if self.first_frame_at is None:
            return 0.0
        return max(0.0, (now - self.first_frame_at - self.paused_s(now)) * 1000.0)

    def rig_played_ms(self, now: float) -> float:
        """The rig's idea of playback, by ITS rule (SpeechForge ``assumed_played_ms``):
        the last ``played_ms`` it was sent, frozen; else real time since its first frame
        minus pauses (our first frame arrives after the rig sent it, so this is never
        ahead of the rig)."""
        if self.sent_played_ms is not None:
            return self.sent_played_ms
        if self.first_frame_at is None:
            return 0.0
        return max(0.0, (now - self.first_frame_at - self.paused_s(now)) * 1000.0)

    def rig_may_produce(self, now: float, ahead_ms: float) -> bool:
        """Could the rig be producing audio for this ctx right now, by its own credit
        rule (``produced < played + ahead``, produced counted in 40 ms tokens with the
        3-token lookahead), with a margin for the first chunk the next unit needs before
        any audio? False = it may be holding back on purpose: no late clock."""
        if self.paused_since is not None:
            return False
        margin = (self.first_chunk_tokens + LOOKAHEAD_TOKENS) * TOKEN_MS + CREDIT_MARGIN_MS
        return self.audio_ms + margin < self.rig_played_ms(now) + ahead_ms


_LIVE: "weakref.WeakSet[RigConnection]" = weakref.WeakSet()
_LAST: dict = {}


class RigConnection:
    """The sf.1 socket of one voice chat, its health, and the frames it carries.

    Lives on the session's event loop. The STT and TTS adapters call it from worker
    threads: everything they use is thread-safe (``send_threadsafe``, ``usable``,
    ``alive``, the STT condition, the per-ctx queues)."""

    def __init__(self, settings: RigSettings, token: str, *, version: str = "",
                 clock: Callable[[], float] = time.monotonic, connect=None,
                 metrics: RigMetrics | None = None, power: RigPowerGate | None = None) -> None:
        self.settings = settings
        self._token = token
        self.version = version
        self.clock = clock
        self._connect_fn = connect
        self.metrics = metrics or METRICS
        self.power = power
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ws = None
        self.state = "idle"               # idle connecting ready down misconfigured closed
        self.reason: str | None = None
        self.proven = False
        self.lease: str | None = None
        self.vacating = False
        self.models_loading = False       # state{models:"loading"}: not usable until ready
        self.ready_info: dict = {}
        self.session_id: str | None = None
        self._resume_token: str | None = None
        self.epoch = 0                    # bumps on every failure that moves in-flight work
        self.cooldown_s = COOLDOWN_S      # (a test shortens it; 30 s in use)
        self.down_until = float("-inf")   # no promotion before this (30 s cooldown)
        self.retry_at = float("-inf")     # no reconnect before this (Retry-After / 429)
        self.failed_at: float | None = None
        self._opened_at = clock()
        self._closing = False
        self._dropped_at: float | None = None
        self._resume_attempts = 0
        self._tasks: set[asyncio.Task] = set()
        self._send_q: asyncio.Queue | None = None
        self._connect_task: asyncio.Task | None = None
        # liveness
        self.last_rx = float("-inf")
        self.last_pong = float("-inf")
        self._outstanding: deque[float] = deque()
        self._proof_after = float("inf")  # a pong for a ping sent after this proves the rig
        self._lag_since: float | None = None
        self._mic_at = float("-inf")
        # STT state (guarded by `cond`)
        self.cond = threading.Condition()
        self.mic_pts = 0
        self._mic_seq = 0
        self._mic_rest = np.zeros(0, "<i2")
        self.partials: deque[tuple[int, dict]] = deque(maxlen=32)   # (n, frame)
        self.partial_n = 0
        self.eous: deque[tuple[int, dict]] = deque(maxlen=32)
        self.eou_n = 0
        self.finals: dict[Any, dict] = {}
        # TTS contexts
        self.ctxs: dict[int, TtsCtx] = {}
        self._next_ctx = 0
        self._lock = threading.Lock()
        self.busy_check: Callable[[], bool] | None = None   # the session: a turn/reply is going on
        self.on_promote: list[Callable[[], None]] = []
        _LIVE.add(self)

    # ------------------------------------------------------------ public state
    def usable(self) -> bool:
        """May a NEW turn (utterance or reply) start on the rig?"""
        now = self.clock()
        return (self.state == "ready" and self.proven and self.lease in SERVING_LEASES
                and not self.vacating and not self.models_loading
                and now >= self.down_until and now - self.last_pong <= PONG_FRESH_S)

    def max_text(self) -> int:
        """``ready.limits.max_text``: tts.text characters one context takes in all."""
        lim = (self.ready_info or {}).get("limits")
        v = lim.get("max_text") if isinstance(lim, dict) else None
        return int(v) if isinstance(v, int) and not isinstance(v, bool) and v > 0 else MAX_TEXT_DEFAULT

    def first_chunk_tokens(self) -> int:
        """The larger of the first chunk we ask for and the rig's default (a context's
        later units use the rig's): what a unit needs before its first audio."""
        ours = int(self.settings.first_chunk_tokens)
        tts = ((self.ready_info or {}).get("engines") or {}).get("tts")
        theirs = tts.get("first_chunk_tokens") if isinstance(tts, dict) else None
        if isinstance(theirs, int) and not isinstance(theirs, bool) and 0 < theirs <= 200:
            return max(ours, theirs)
        return ours

    def alive(self, epoch: int) -> bool:
        """Is work started at ``epoch`` still good on the rig?"""
        return self.state == "ready" and epoch == self.epoch

    def summary(self) -> dict:
        now = self.clock()
        r = self.ready_info or {}
        state = self.state
        if state == "ready" and not self.usable():
            state = "busy" if (self.vacating or self.lease not in (None, *SERVING_LEASES)) else (
                "loading" if self.models_loading else
                "cooling" if now < self.down_until else "connecting")
        elif state == "down":
            state = "busy" if self.reason in ("busy", "rate_limited", "not_ready", "vacating") else "down"
        return {"state": state, "reason": self.reason,
                "cooldown_s": max(0, round(self.down_until - now)) if now < self.down_until else 0,
                "version": r.get("version"), "proto": r.get("proto"),
                "engines": {k: {"name": (v or {}).get("name"), "sha8": (v or {}).get("sha8"),
                                "provider": (v or {}).get("provider")}
                            for k, v in (r.get("engines") or {}).items() if isinstance(v, dict)},
                "gpu": r.get("gpu") if isinstance(r.get("gpu"), dict) else None,
                "limits": r.get("limits") if isinstance(r.get("limits"), dict) else None,
                "lease": self.lease}

    def _publish(self) -> None:
        _LAST.clear()
        _LAST.update(self.summary(), at=time.time())

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Open the socket in the background (the rig's warm-up: lease + model load).
        Call on the session's loop. Until ``ready`` + a pong, the session is ``cpu``."""
        self.loop = asyncio.get_running_loop()
        self._opened_at = self.clock()
        fp = _fingerprint(self.settings.url, self._token)
        if fp in _MISCONFIGURED:
            self._set_misconfigured()
            return
        self._spawn(self._health())
        self._begin_connect(resume=False)

    def _spawn(self, coro) -> asyncio.Task:
        t = asyncio.ensure_future(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return t

    def _begin_connect(self, *, resume: bool) -> None:
        if self._closing or self.state in ("misconfigured", "closed"):
            return
        if self._connect_task is not None and not self._connect_task.done():
            return
        self._connect_task = self._spawn(self._connect(resume=resume))

    async def _open_socket(self):
        if self._connect_fn is not None:
            return await self._connect_fn(self.settings.url, self._token)
        from websockets.asyncio.client import connect
        return await connect(self.settings.url, subprotocols=[PROTO],
                             additional_headers={"Authorization": f"Bearer {self._token}"},
                             max_size=MAX_MESSAGE_BYTES, compression=None,
                             open_timeout=OPEN_TIMEOUT_S, ping_interval=None, close_timeout=1.0,
                             user_agent_header=f"dispatch-voice/{self.version or '1'}")

    async def _connect(self, *, resume: bool) -> None:
        self.state = "connecting"
        self.proven = False
        self._publish()
        try:
            try:
                ws = await self._open_socket()
            except Exception as e:      # noqa: BLE001 - classified below
                raise self._refusal(e) from None
            self.ws = ws
            hello: dict = {"type": "hello", "proto": PROTO,
                           "client": {"name": "dispatch-voice", "version": self.version or "1"},
                           "caps": {"opus_out": False}}
            if resume and self.session_id and self._resume_token:
                hello["resume"] = {"session": self.session_id, "token": self._resume_token}
            else:
                with self.cond:
                    self.mic_pts = 0
                    self._mic_seq = 0
                    self._mic_rest = np.zeros(0, "<i2")
            await ws.send(dumps(hello))
            ready = await asyncio.wait_for(self._await_ready(ws), READY_WAIT_S)
        except RigRefused as r:
            await self._drop_socket()
            self._refused(r)
            return
        except Exception as e:  # noqa: BLE001 - timeout, a close during hello
            await self._drop_socket()
            self._refused(RigRefused("unreachable" if not isinstance(e, asyncio.TimeoutError) else "not_ready",
                                     0.0))
            return
        self._on_ready(ready)
        self._send_q = asyncio.Queue()
        self._spawn(self._reader(self.ws))
        self._spawn(self._sender(self.ws, self._send_q))
        self.send({"type": "stt.config", "lang": "en", "partial_ms": 160,
                   "eou": {"enabled": True, "min_pause_ms": 150}})
        self._ping_now()

    def _refusal(self, e: Exception) -> RigRefused:
        resp = getattr(e, "response", None)
        code = getattr(resp, "status_code", None)
        if code is None:
            return RigRefused("unreachable")
        retry = 0.0
        with contextlib.suppress(Exception):
            retry = float(resp.headers.get("Retry-After") or 0)
        if code == 401:
            return RigRefused("token")
        if code == 429:
            return RigRefused("rate_limited", max(retry, RATE_LIMIT_BACKOFF_S))
        if code == 503:
            return RigRefused("busy", retry)
        return RigRefused("unreachable", retry)

    async def _await_ready(self, ws) -> dict:
        """Frames until ``ready`` (``state{models: loading}`` may come first). An
        ``error`` here is a refusal (``not_ready``/``busy`` + ``retry_after_s``)."""
        while True:
            msg = await ws.recv()
            self.last_rx = self.clock()
            if isinstance(msg, bytes):
                continue
            try:
                m = json.loads(msg)
            except ValueError:
                continue
            if not isinstance(m, dict):
                continue
            kind = m.get("type")
            if kind == "ready":
                return m
            if kind == "error":
                if m.get("fatal"):
                    code = str(m.get("code") or "")
                    raise RigRefused("not_ready" if code in ("not_ready", "busy", "vacated") else "fatal_error",
                                     float(m.get("retry_after_s") or 0))
                continue            # a refused resume: a fresh session's ready follows
            if kind == "state":
                if m.get("lease"):
                    self.lease = m.get("lease")
                if m.get("models") == "loading" and not self.models_loading:
                    self.models_loading = True      # the rig is loading: the CPU serves meanwhile
                    self._publish()

    def _on_ready(self, ready: dict) -> None:
        now = self.clock()
        if ready.get("session") != self.session_id:
            with self.cond:              # a fresh session (or a refused resume): new clocks
                self.mic_pts = 0
                self._mic_seq = 0
                self._mic_rest = np.zeros(0, "<i2")
                # The old session's transcripts are in its mic clock: never this one's.
                self.partials.clear()
                self.eous.clear()
                self.finals.clear()
        self.ready_info = ready
        self.session_id = ready.get("session")
        self._resume_token = ready.get("resume_token")
        lease = ready.get("lease") or {}
        self.lease = lease.get("state") if isinstance(lease, dict) else None
        self.vacating = self.lease == "vacating"
        self.models_loading = False
        self.state = "ready"
        self.reason = None
        self.last_rx = now
        self.last_pong = float("-inf")
        self._outstanding.clear()
        self._dropped_at = None
        self._resume_attempts = 0
        # The open counts as the probe; a resume inside a cooldown waits for the
        # between-turn probe after it.
        self._proof_after = now if now >= self.down_until else float("inf")
        self.proven = False
        log.info("voice: rig ready (%s, GPU %s)", (ready.get("version") or "?"),
                 ((ready.get("gpu") or {}).get("name") or "?"))
        self._publish()

    def _refused(self, r: RigRefused) -> None:
        now = self.clock()
        if r.reason == "token":
            _MISCONFIGURED.add(_fingerprint(self.settings.url, self._token))
            self._set_misconfigured()
            return
        self.state = "down"
        self.reason = r.reason
        self.proven = False
        # Retry after the rig's Retry-After, else after the cooldown (an unreachable rig
        # is not dialled every second); either way only between turns (`_probe`).
        self.retry_at = now + (r.retry_after_s if r.retry_after_s > 0 else self.cooldown_s)
        if self._dropped_at is not None and now - self._dropped_at <= RESUME_WINDOW_S:
            self._resume_attempts += 1
            if self._resume_attempts < RESUME_ATTEMPTS and r.reason == "unreachable":
                self._spawn(self._retry_resume())
        log.info("voice: rig unavailable (%s); this computer speaks", r.reason)
        self._need_cpu()
        self._publish()

    async def _retry_resume(self) -> None:
        await asyncio.sleep(0.5)
        self._begin_connect(resume=True)

    def _set_misconfigured(self) -> None:
        self.state = "misconfigured"
        self.reason = "token"
        self.proven = False
        log.warning("voice: the rig refused the token (401); not retried until rig.url or "
                    "%s changes", TOKEN_ENV)
        self._need_cpu()
        self._publish()

    async def _drop_socket(self) -> None:
        ws, self.ws = self.ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()

    async def close(self) -> None:
        """``bye`` at the end of a voice chat; the rig's idle clock releases the card."""
        self._closing = True
        ws = self.ws
        if ws is not None and self.state == "ready":
            with contextlib.suppress(Exception):
                await asyncio.wait_for(ws.send(dumps({"type": "bye", "reason": "stop"})), 1.0)
        self.state = "closed"
        for c in list(self.ctxs.values()):
            c.q.put(("failed", "closed"))
        with self.cond:
            self.cond.notify_all()
        await self._drop_socket()
        for t in list(self._tasks):
            if t is not asyncio.current_task():
                t.cancel()
        await asyncio.gather(*[t for t in self._tasks if t is not asyncio.current_task()],
                             return_exceptions=True)
        _LAST.update(state="idle", at=time.time())

    # ------------------------------------------------------------ failure
    def fail(self, reason: str, *, retry_after_s: float = 0.0, move: bool = True) -> None:
        """A failure: no NEW turn on the rig for ``max(30 s, retry_after_s)``; with
        ``move`` the work in flight fails over now (the epoch changes)."""
        now = self.clock()
        self.down_until = max(self.down_until, now + max(self.cooldown_s, retry_after_s))
        self.failed_at = now
        self.proven = False
        self.reason = reason
        self._proof_after = float("inf")          # proof needs a between-turn probe now
        if move:
            self.epoch += 1
            with self._lock:
                hit = list(self.ctxs.values())
            for c in hit:
                c.q.put(("failed", reason))
            with self.cond:
                self.cond.notify_all()
        self.metrics.fallback(reason)
        log.info("voice: rig %s; this computer takes over", reason)
        self._need_cpu()
        self._publish()

    def dead(self, reason: str, *, retry_after_s: float = 0.0) -> None:
        """The rig is down (socket gone or unusable): fail over, drop the socket, and
        re-attach in the background (a resume keeps the slot and the warm caches;
        promotion still waits for the cooldown and a between-turn probe)."""
        if self.state in ("closed", "misconfigured"):
            return
        was_ready = self.state == "ready"
        self.state = "down"
        self.fail(reason, retry_after_s=retry_after_s)
        self.state = "down"
        self.reason = reason
        if self.loop is not None:
            with contextlib.suppress(RuntimeError):
                self.loop.call_soon_threadsafe(lambda: self._spawn(self._drop_socket()))
        if was_ready and not self._closing and reason in ("closed", "silence", "missed_pongs"):
            self._dropped_at = self.clock()
            self._resume_attempts = 0
            if self.loop is not None:
                with contextlib.suppress(RuntimeError):
                    self.loop.call_soon_threadsafe(lambda: self._begin_connect(resume=True))
        self._publish()

    def _need_cpu(self) -> None:
        if self.power is not None:
            self.power.need()

    def _maybe_release_power(self) -> None:
        """The rig serves again: give the power hold back, but never while a turn or a
        reply is still in flight (it may be CPU work: the hold stays until it ends)."""
        if self.power is None or not self.power.held or not self.usable():
            return
        busy = False
        if self.busy_check is not None:
            with contextlib.suppress(Exception):
                busy = bool(self.busy_check())
        if not busy:
            self.power.not_needed()

    def note_turn(self, on_rig: bool) -> None:
        """An utterance started (on the rig or the CPU) while the rig is in play."""
        self.metrics.add("turns")
        if not on_rig:
            self._need_cpu()

    # ------------------------------------------------------------ probe (between turns)
    def maybe_probe(self) -> None:
        """Between turns: re-promote a rig that cooled down. A dead socket is
        reconnected (``ready`` + one ping round trip); a live one needs one fresh
        pong. Never within the cooldown or before a Retry-After. Thread-safe."""
        if self.loop is None or self._closing:
            return
        with contextlib.suppress(RuntimeError):
            self.loop.call_soon_threadsafe(self._probe)

    def _probe(self) -> None:
        now = self.clock()
        if self.state in ("misconfigured", "closed", "connecting") or now < self.down_until:
            return
        if self.state in ("down", "idle"):
            if now >= self.retry_at:
                self._begin_connect(resume=bool(self._dropped_at and now - self._dropped_at <= RESUME_WINDOW_S))
            return
        if self.state == "ready" and not self.proven and self._proof_after == float("inf"):
            self._proof_after = now          # the next pong proves the rig
            self._ping_now()

    # ------------------------------------------------------------ io
    def send(self, msg: dict | bytes) -> None:
        """Queue a frame (call on the loop)."""
        q = self._send_q
        if q is None or self.state != "ready":
            return
        q.put_nowait(msg if isinstance(msg, bytes) else dumps(msg))

    def send_threadsafe(self, msg: dict | bytes) -> None:
        loop = self.loop
        if loop is None:
            return
        data = msg if isinstance(msg, bytes) else dumps(msg)
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self._enqueue_raw, data)

    def _enqueue_raw(self, data) -> None:
        q = self._send_q
        if q is not None and self.state == "ready":
            q.put_nowait(data)

    async def _sender(self, ws, q: asyncio.Queue) -> None:
        try:
            while True:
                item = await q.get()
                await ws.send(item)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the reader sees the close
            return

    def send_mic(self, pcm16: np.ndarray) -> int:
        """Mic audio (16 kHz int16) -> 20 ms sf.1 frames with the session's mic clock.
        Thread-safe. Returns the mic clock after it (samples sent)."""
        frames: list[bytes] = []
        with self.cond:
            a = to_int16(pcm16)
            buf = np.concatenate([self._mic_rest, a]) if len(self._mic_rest) else a
            n_full = len(buf) // MIC_FRAME_SAMPLES * MIC_FRAME_SAMPLES
            for i in range(0, n_full, MIC_FRAME_SAMPLES):
                frames.append(pack_audio(KIND_MIC, 0, 0, self._mic_seq, self.mic_pts,
                                         buf[i:i + MIC_FRAME_SAMPLES]))
                self._mic_seq += 1
                self.mic_pts += MIC_FRAME_SAMPLES
            self._mic_rest = np.array(buf[n_full:], copy=True)
            end = self.mic_pts + len(self._mic_rest)
        self._mic_at = self.clock()
        for f in frames:
            self.send_threadsafe(f)
        return end

    def flush_mic(self) -> int:
        """Send the < 20 ms remainder (before ``stt.flush``: the final covers it all)."""
        with self.cond:
            rest, self._mic_rest = self._mic_rest, np.zeros(0, "<i2")
            if not len(rest):
                return self.mic_pts
            f = pack_audio(KIND_MIC, 0, 0, self._mic_seq, self.mic_pts, rest)
            self._mic_seq += 1
            self.mic_pts += len(rest)
            end = self.mic_pts
        self.send_threadsafe(f)
        return end

    def drop_mic_rest(self) -> None:
        with self.cond:
            self._mic_rest = np.zeros(0, "<i2")

    # ------------------------------------------------------------ health
    def _ping_now(self) -> None:
        if self.state != "ready":
            return
        t = self.clock()
        self._outstanding.append(t)
        load = None
        with contextlib.suppress(Exception):
            load = round(os.getloadavg()[0] / max(1, os.cpu_count() or 1), 2)
        msg: dict = {"type": "ping", "t": t}
        if load is not None:
            msg["box_load"] = load
        self.send(msg)

    async def _health(self) -> None:
        """Pings every 250 ms and the sub-second down rules; the idle probe."""
        last_ping = float("-inf")
        last_idle = self.clock()
        while not self._closing:
            await asyncio.sleep(0.05)
            now = self.clock()
            if self.state == "ready":
                if now - self.last_rx > SILENCE_S:
                    self.dead("silence")
                    continue
                if now - last_ping >= PING_S:
                    if len(self._outstanding) >= MISSED_PONGS:
                        self.dead("missed_pongs")
                        continue
                    last_ping = now
                    self._ping_now()
                if (self._lag_since is not None and now - self._lag_since >= LAG_FOR_S
                        and now - self._mic_at <= SPEECH_FLOWING_S):
                    self._lag_since = None
                    self.fail("lag")
                self._credit_tick(now)
                if self.proven and self.power is not None and self.power.held:
                    self._maybe_release_power()
            if (self.power is not None and not self.power.held and not self.usable()
                    and now - self._opened_at >= HOLD_GRACE_S):
                self._need_cpu()          # not ready this long after the open: the CPU serves
            if now - last_idle >= IDLE_PROBE_EVERY_S:
                last_idle = now
                busy = False
                if self.busy_check is not None:
                    with contextlib.suppress(Exception):
                        busy = bool(self.busy_check())
                if not busy:
                    self._probe()

    async def _reader(self, ws) -> None:
        try:
            async for msg in ws:
                self.last_rx = self.clock()
                if isinstance(msg, bytes):
                    self._on_audio(msg)
                    continue
                try:
                    m = json.loads(msg)
                except ValueError:
                    continue
                if isinstance(m, dict):
                    self._on_frame(m)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a close
            pass
        if ws is self.ws and not self._closing and self.state == "ready":
            self.dead("closed")

    def _on_audio(self, buf: bytes) -> None:
        try:
            kind, ctx, _gen, _seq, _pts, pcm = unpack_audio(buf)
        except ValueError:
            return
        if kind != KIND_TTS:
            return
        c = self.ctxs.get(ctx)
        if c is None or c.cancelled:
            return
        if c.first_frame_at is None:
            c.first_frame_at = self.clock()
        c.samples += len(pcm)
        c.q.put(("audio", np.array(pcm, copy=True)))

    def _on_frame(self, m: dict) -> None:
        kind = m.get("type")
        now = self.clock()
        if kind == "pong":
            t = m.get("t")
            if isinstance(t, (int, float)):
                while self._outstanding and self._outstanding[0] <= t + 1e-9:
                    self._outstanding.popleft()
                self.last_pong = now
                self.metrics.rtt((now - float(t)) * 1000.0)
                if (self.state == "ready" and not self.proven and t >= self._proof_after
                        and now >= self.down_until and self.lease in SERVING_LEASES
                        and not self.vacating and not self.models_loading):
                    self.proven = True
                    self.reason = None
                    log.info("voice: rig proven (lease %s); new turns use it", self.lease)
                    self._maybe_release_power()
                    for fn in list(self.on_promote):
                        with contextlib.suppress(Exception):
                            fn()
                    self._publish()
            return
        if kind == "state":
            models = m.get("models")
            if models == "loading":
                self.models_loading = True
            elif models == "ready":
                self.models_loading = False
            lease = m.get("lease")
            if lease == "lost":
                self.lease = "lost"
                self.dead("lease_lost")
            elif lease == "vacating":
                self.lease = "vacating"
                self.vacating = True
                self.fail("vacating", retry_after_s=float(m.get("deadline_s") or 0), move=False)
            elif isinstance(lease, str):
                self.lease = lease
                if lease in SERVING_LEASES:
                    self.vacating = False
            self._publish()
            return
        if kind == "stt.partial":
            lag = m.get("lag_ms")
            if isinstance(lag, (int, float)):
                self.metrics.observe("lag", lag)
                if lag > LAG_MS:
                    if self._lag_since is None:
                        self._lag_since = now
                else:
                    self._lag_since = None
            with self.cond:
                self.partial_n += 1
                self.partials.append((self.partial_n, m))
                self.cond.notify_all()
            return
        if kind == "eou":
            with self.cond:
                self.eou_n += 1
                self.eous.append((self.eou_n, m))
                self.cond.notify_all()
            return
        if kind == "stt.final":
            with self.cond:
                self.finals[m.get("id")] = m
                while len(self.finals) > 16:
                    self.finals.pop(next(iter(self.finals)))
                self.cond.notify_all()
            return
        if kind == "error":
            code = str(m.get("code") or "internal")
            if m.get("fatal"):
                self.dead("vacating" if code == "vacated" else
                          "busy" if code in ("not_ready", "busy") else "fatal_error",
                          retry_after_s=float(m.get("retry_after_s") or 0))
                return
            ctx = m.get("ctx")
            c = self.ctxs.get(ctx) if isinstance(ctx, int) else None
            if c is not None:
                c.q.put(("error", code))
            return
        if kind in ("tts.started", "tts.words", "tts.flushed", "tts.done", "cancelled"):
            if kind == "cancelled":
                for ctx in m.get("ctxs") or []:
                    with self._lock:
                        c = self.ctxs.pop(ctx, None) if isinstance(ctx, int) else None
                    if c is not None:
                        c.q.put(("cancelled", None))
                return
            c = self.ctxs.get(m.get("ctx"))
            if c is None:
                return
            if kind == "tts.started":
                c.started = True
                if isinstance(m.get("ttfa_ms"), (int, float)):
                    self.metrics.observe("ttfa", m["ttfa_ms"])
                c.q.put(("started", m.get("ttfa_ms")))
            elif kind == "tts.words":
                c.q.put(("words", m.get("words") or []))
            elif kind == "tts.flushed":
                c.q.put(("flushed", m.get("id")))
            elif kind == "tts.done":
                c.q.put(("done", None))
                c.closed = True
                with self._lock:
                    self.ctxs.pop(c.ctx, None)
            return
        # stt.vad, ready (after a resume), voice.* answers, unknown frames: tolerated

    # ------------------------------------------------------------ TTS contexts
    def open_ctx(self, *, gen: int, voice: str, seed: int | None, turn_id: str | None = None,
                 speculative: bool = False) -> TtsCtx | None:
        """``tts.open`` (+ the credit). Thread-safe; None when the rig is not alive."""
        with self._lock:
            if self.state != "ready" or self.vacating or self.models_loading:
                return None                    # vacating: the current ctx finishes, no new one
            self._next_ctx = (self._next_ctx % U16) + 1
            c = TtsCtx(ctx=self._next_ctx, gen=gen & U16, voice=voice, epoch=self.epoch,
                       first_chunk_tokens=self.first_chunk_tokens())
            self.ctxs[c.ctx] = c
        msg: dict = {"type": "tts.open", "ctx": c.ctx, "gen": c.gen, "voice": voice,
                     "first_chunk_tokens": int(self.settings.first_chunk_tokens), "speculative": bool(speculative)}
        if seed is not None:
            msg["seed"] = int(seed)
        if turn_id:
            msg["turn_id"] = str(turn_id)[:64]
        self.send_threadsafe(msg)
        self.send_threadsafe({"type": "credit", "ctx": c.ctx, "ahead_ms": int(self.settings.ahead_ms)})
        return c

    def close_ctx(self, c: TtsCtx) -> None:
        if not c.closed and not c.cancelled:
            c.closed = True
            self.send_threadsafe({"type": "tts.close", "ctx": c.ctx})

    def cancel_ctx(self, c: TtsCtx) -> None:
        if not c.cancelled:
            c.cancelled = True
            self.send_threadsafe({"type": "cancel", "ctx": c.ctx})

    def cancel_gen(self, gen: int) -> None:
        """``cancel{gen}``: every ctx with gen <= ``gen`` stops (idempotent). Called on
        the loop BEFORE abort_turn / Speculator.start (L8 order)."""
        g = gen & U16
        with self._lock:
            hit = [c for c in self.ctxs.values() if c.gen <= g]
        for c in hit:
            c.cancelled = True
            c.q.put(("cancelled", None))
        if self.state == "ready":
            self.send_threadsafe({"type": "cancel", "gen": g})

    def credit(self, c: TtsCtx, played_ms: float, *, force: bool = False) -> None:
        """The phone's report (ctx clock). Sent now when it is the first one, when
        ``force``, or when the last credit is ``CREDIT_S`` old; between reports the
        health loop keeps the rig's copy fresh (:meth:`_credit_tick`)."""
        now = self.clock()
        c.played_ms = max(0.0, float(played_ms))
        c.played_at = now
        c.paused_at_report = c.paused_s(now)
        if force or c.sent_played_ms is None or now - c.last_credit_at >= CREDIT_S:
            self._send_credit(c, now)

    def _send_credit(self, c: TtsCtx, now: float) -> None:
        """``credit{ctx, ahead_ms, played_ms}`` with the phone's playback as of now (its
        last report run on in real time): the rig freezes the value until the next one."""
        v = float(int(max(0.0, c.played_est_ms(now))))
        c.last_credit_at = now
        c.sent_played_ms = v
        self.send_threadsafe({"type": "credit", "ctx": c.ctx, "ahead_ms": int(self.settings.ahead_ms),
                              "played_ms": int(v)})

    def _credit_tick(self, now: float) -> None:
        """Every ``CREDIT_S`` while a context is open and the phone has reported on it."""
        with self._lock:
            open_ = [c for c in self.ctxs.values() if not c.cancelled and c.played_ms is not None]
        for c in open_:
            if now - c.last_credit_at >= CREDIT_S:
                self._send_credit(c, now)

    def pause_ctx(self, c: TtsCtx) -> None:
        if c.paused_since is None and not c.cancelled and not c.closed:
            now = self.clock()
            if c.played_ms is not None:
                self._send_credit(c, now)        # the playback point the pause freezes
            c.paused_since = now
            self.send_threadsafe({"type": "tts.pause", "ctx": c.ctx})

    def resume_ctx(self, c: TtsCtx) -> None:
        if c.paused_since is not None:
            now = self.clock()
            c.paused_total += now - c.paused_since
            c.paused_since = None
            if not c.cancelled:
                self.send_threadsafe({"type": "tts.resume", "ctx": c.ctx})
                if c.played_ms is not None:
                    self._send_credit(c, now)


# --------------------------------------------------------------------------- #
# Status (Settings › Voice, the Voice lab, doctor)
# --------------------------------------------------------------------------- #

def status(settings) -> dict:
    """What the browser may know: no token, the address reduced to its host."""
    rs: RigSettings = getattr(settings, "rig", None) or RigSettings()
    tok = read_token()
    configured = bool(rs.url and valid_url(rs.url) and tok.value)
    live = [c for c in list(_LIVE) if c.state != "closed"]
    out: dict = {"configured": configured, "enabled": rs.enabled, "url_set": bool(rs.url),
                 "token": tok.source, "token_problem": tok.problem, "host": host_of(rs.url),
                 "active": bool(configured and rs.enabled), "sessions": len(live),
                 "metrics": METRICS.snapshot()}
    if not configured:
        out["state"] = "not_configured"
    elif not rs.enabled:
        out["state"] = "off"
    elif _fingerprint(rs.url, tok.value) in _MISCONFIGURED:
        out["state"] = "misconfigured"
        out["reason"] = "token"
    elif live:
        out.update(live[-1].summary())
    else:
        out["state"] = "idle"
        if _LAST:
            out["last"] = {k: _LAST.get(k) for k in ("version", "gpu", "engines", "limits", "at")}
    return out


# --------------------------------------------------------------------------- #
# Wiring a voice chat (ws.serve)
# --------------------------------------------------------------------------- #

@dataclass
class RigSession:
    """What ws.serve gets for one voice chat with a rig in play."""
    conn: RigConnection
    sync: VoiceSync
    power: RigPowerGate
    #: profile id (None = the default voice) -> "sha256:<hex>" once stored on the rig
    voices: dict = field(default_factory=dict)
    bad: set = field(default_factory=set)
    _pending: dict = field(default_factory=dict)

    def voice_ref(self, profile_id: str | None) -> str | None:
        """The rig voice id for a profile, or None (not synced, refused, not a cloned
        voice): that reply runs on the CPU."""
        return self.voices.get(profile_id)

    async def sync_voices(self, profile_ids) -> None:
        for pid in dict.fromkeys(profile_ids):
            await self.sync_one(pid)

    async def sync_one(self, profile_id: str | None, *, force: bool = False) -> None:
        path = await asyncio.to_thread(voice_file, profile_id)
        if path is None:
            return
        try:
            sha = await asyncio.to_thread(file_sha256, path)
        except OSError:
            return
        if sha in self.bad:
            return
        res = await self.sync.ensure(sha, path, force=force)
        if res == "ready":
            self.voices[profile_id] = f"sha256:{sha}"
        elif res == "bad":
            self.bad.add(sha)
            self.voices.pop(profile_id, None)
        elif res == "token":
            self.voices.pop(profile_id, None)

    def resync_threadsafe(self, profile_id: str | None) -> None:
        """``unknown_voice`` on ``tts.open``: PUT again (once), in the background."""
        self.voices.pop(profile_id, None)
        if self._pending.get(profile_id):
            return
        self._pending[profile_id] = True
        loop = self.conn.loop
        if loop is None:
            return

        async def run() -> None:
            try:
                await self.sync_one(profile_id, force=True)
            finally:
                self._pending.pop(profile_id, None)

        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(lambda: self.conn._spawn(run()))

    async def close(self) -> None:
        await self.conn.close()


def open_for_session(settings, *, token: str | None = None, connect=None,
                     http_transport=None) -> RigSession | None:
    """The rig for one voice chat, or None when it is not in play (switch off, no
    address, no token, a malformed address): the session is then exactly the CPU
    one. Starts the socket in the background (call on the session's loop)."""
    rs: RigSettings = getattr(settings, "rig", None) or RigSettings()
    if not rs.enabled or not rs.url:
        return None
    if not valid_url(rs.url):
        log.warning("voice: rig.url is not a ws:// or wss:// address; the rig is not used")
        return None
    value = token if token is not None else read_token().value
    if not value:
        log.info("voice: rig.enabled is on but %s is not set; the rig is not used", TOKEN_ENV)
        return None
    from dispatch_voice import __version__
    power = RigPowerGate(settings)
    conn = RigConnection(rs, value, version=__version__, connect=connect, power=power)
    sync = VoiceSync(rs.url, value, transport=http_transport)
    conn.start()
    return RigSession(conn, sync, power)


def random_seed() -> int:
    return secrets.randbelow(2**31 - 1)
