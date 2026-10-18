#!/usr/bin/env python
"""Low-bandwidth end-to-end harness (owner: T6; SPEC §12 "Simulated bad networks").

A wire client that behaves like the v3 phone client — gated mic with sample
offsets, Opus or PCM16 uplink in 100 ms messages, a 30 s uplink ring with
`start.cursor` replay, the modelled player (drive.js's gap-free scheduler with a
20 ms lead) that reports `playback_done`, seq claims keyed `(sid, seq)`, a link
monitor that switches to the phone voice (`link {mode: "text"}`; reactive causes at
once, predictive ones at `say_end`), the phone voice itself (one utterance per
`say`, chars/15 s), 2 s pings with the dead-socket rule (the later of 3 missed
pongs or 8 s; 4x RTT above 1.5 s), reconnect with backoff plus an immediate try on
`online`, and `start.after_partial` — talks through ``tests/shaper.py`` to a server.

Profiles (tests/shaper.py PROFILES): baseline, lte-edge, derp-busy, 3g-weak, edge,
blackout-20s, stall-5s; each over permessage-deflate and raw. Assertions (§12):

* every run: exactly one user row, carrying the whole utterance (voiced length
  within 10 % of the clip's); every reply sentence fully heard exactly once, in
  order, the reply complete; the transcript reached the client;
* 3g-weak with the PCM downlink (what a client without WebCodecs gets): text mode
  entered and every `say` heard once (with Opus the audio fits and it stays audio);
* edge: the first `say` arrives <= `eot` + 6 s;
* blackout-20s (mid-utterance): the client reconnected and still one row with the
  full sentence;
* stall-5s (mid-reply, then the old path is gone): the client rejoined while the
  LONG reply was still streaming (the server says so), and only unplayed sentences
  followed, none twice.

A sentence counts as complete when its ``audio_end`` (Amendment T6-2) or a later
``say``/``say_end`` arrived; ``after_partial.sentences`` counts the leading complete
sentences of the reply (the player keeps playing them across a reconnect), so a
sentence cut by a drop comes again from its start and is heard in full once.

Two judges read the outcome (review-T6 M1):

* ``ref`` (the default with the reference server): its ``/rows/{thread}`` debug view
  (voiced length of the row, the reply's sentence list, the server's resume log).
* ``dispatch``: only what a real DisPatch exposes. It unlocks with the PIN
  (``POST /api/auth/unlock``; the PIN comes from ``--pin-file`` or ``DISPATCH_PIN``,
  never argv) and sends the ``lc_session`` cookie on the socket, creates one thread
  per run (``POST /api/threads {bot_id}``) or uses ``--thread`` (rows that existed
  before the run are ignored), and reads ``GET /api/threads/{id}/messages``.
  ``full_sentence`` = the user row's text against the clip's known words
  (``fixtures/transcripts.json``: WER <= 0.15 and the first word present, which is
  what a cut-off head loses); ``reply_complete_once_in_order`` = the heard ``say``
  texts, in order, spell exactly the assistant row's words after T5's sentence
  splitter and ``normalise`` (the ``say`` text is the speakable text, C.4); and
  ``rejoined_mid_reply`` is judged by the client (the drop came mid-reply and the
  reply's heard sentences came from two sessions) because a real server keeps no
  resume log.

Servers:

* default: the reference v3 server (tests/transport/refserver.py, built from the
  production transport modules), ``ref`` judge.
* ``--ref-dispatch``: the reference server in its DisPatch-shaped mode (PIN, cookie
  gate, threads/messages API, oracle STT), ``dispatch`` judge: proves the judge.
* ``--server ws://host:port`` (+ ``--base``, ``--bot`` or ``--thread``, the PIN): a
  running DisPatch with the add-on, ``dispatch`` judge.
* ``--start-dispatch DIR``: start DisPatch from that backend checkout on
  ``--server-port`` (scratch data, one API bot on the stub LLM's long reply, a PIN set
  through ``/api/auth/setup``), ``dispatch`` judge. Needs T3 slice 4 for the v3 wire.

    uv run python tests/e2e_lowbw.py                      # the full matrix, ~15 min
    uv run python tests/e2e_lowbw.py baseline --codecs opus --compression deflate
    uv run python tests/e2e_lowbw.py --ref-dispatch blackout-20s stall-5s
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import http.cookies
import json
import os
import re
import secrets
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import shaper

from dispatch_voice.transport import opus, wire3

SR = 16000
FRAME = 320
SR_OUT = 24000
PING_S = 2.0
LEAD_S = 0.02
RING_S = 30.0
STALE_S = 20.0
PHONE_CHARS_PER_S = 15.0
DEFAULT_PORTS = (19055, 19056)                 # reference server, shaper (T6 range 19055-19064)
CLIP = HERE / "fixtures" / "clips" / "driver1b.wav"
TRANSCRIPTS = HERE / "fixtures" / "transcripts.json"
WER_MAX = 0.15
#: ``--fixed-reply split``: a reply whose FIRST sentence is long with a comma after 20
#: characters — the splitter cuts it there while it streams (the first words matter for
#: latency). A rejoin must still skip exactly the pieces heard (review-T3s4 #1: 3 of 3
#: such stall-5s runs lost "It should arrive on Thursday before noon.").
SPLIT_REPLY = ("The dentist is at three this afternoon, and the school call is at four thirty. "
               "Traffic looks light on the way there. I can remind you ten minutes before each one. "
               "The package from the hardware store shipped this morning. It should arrive on "
               "Thursday before noon. The weather stays dry until the evening. There is a small "
               "chance of rain after seven. Your next free slot is tomorrow at ten. I moved the "
               "grocery list to the shared note. The garage quote came in a little under budget. "
               "The electrician can start the week after next. Nothing else needs your attention today.")

# link-monitor thresholds (a subset of LINK_THRESHOLDS in link-monitor.js, SPEC §5.3)
RTT_EWMA_BAD_MS = 1000
RTT_BAD_COUNT = 3
PONG_LATE_S = 5.0
UNDERRUNS_PER_REPLY = 2
GAP_BAD_S = 0.4
DEADLINE_MIN_S = 5.0


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


# --------------------------------------------------------------------------- #
# the DisPatch judge (review-T6 M1)
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower().replace("\u2019", "'"))


def wer(hyp: str, ref: str) -> float:
    h, r = words(hyp), words(ref)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / len(r)


def known_text(clip: Path = CLIP) -> str:
    return json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))["16k"][clip.stem]


def spoken_words(content: str) -> list[str]:
    """The words a reply row is spoken as: T5's splitter + normalise (what `say.text` carries)."""
    try:
        from dispatch_voice.text.speech_text import SentenceSplitter, normalise
    except ImportError:                               # pragma: no cover - the add-on is in-tree
        return words(content)
    out: list[str] = []
    for span in SentenceSplitter().split(content or ""):
        out += words(normalise(span.text, tags_ok=False))
    return out


class DispatchJudge:
    """Talks to a DisPatch (or the reference server's DisPatch-shaped mode) the way a
    browser would: PIN unlock -> ``lc_session`` cookie on every request and the socket."""

    def __init__(self, base: str, *, pin: str | None = None, cookie: str | None = None,
                 bot: str | None = None, thread: str | None = None) -> None:
        self.base = base.rstrip("/")
        self.pin = pin
        self.cookie = cookie
        self.bot = bot
        self.thread = thread

    def _req(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict, object]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self.cookie:
            req.add_header("Cookie", f"lc_session={self.cookie}")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, dict(r.headers), json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), None

    def login(self) -> None:
        if self.cookie or not self.pin:
            return
        code, headers, _ = self._req("POST", "/api/auth/unlock", {"pin": self.pin})
        if code != 200:
            raise RuntimeError(f"unlock refused ({code})")
        jar = http.cookies.SimpleCookie()
        for k, v in headers.items():
            if k.lower() == "set-cookie":
                jar.load(v)
        if "lc_session" not in jar:
            raise RuntimeError("unlock answered without an lc_session cookie")
        self.cookie = jar["lc_session"].value

    def ws_headers(self) -> dict:
        return {"Cookie": f"lc_session={self.cookie}"} if self.cookie else {}

    def new_thread(self) -> str:
        if self.thread:
            return self.thread
        if not self.bot:
            raise RuntimeError("the dispatch judge needs --bot (a thread per run) or --thread")
        code, _, body = self._req("POST", "/api/threads", {"bot_id": self.bot})
        if code != 200 or not isinstance(body, dict) or not body.get("id"):
            raise RuntimeError(f"could not create a thread ({code})")
        return body["id"]

    def messages(self, thread: str) -> list[dict]:
        code, _, body = self._req("GET", f"/api/threads/{thread}/messages?limit=500")
        if code != 200 or not isinstance(body, dict):
            raise RuntimeError(f"messages of {thread}: HTTP {code}")
        return [m for m in body.get("messages") or [] if isinstance(m, dict)]

    def snapshot(self, thread: str) -> set[str]:
        return {m.get("id") for m in self.messages(thread)}

    def outcome(self, thread: str, before: set[str]) -> dict:
        new = [m for m in self.messages(thread) if m.get("id") not in before]
        users = [m for m in new if m.get("role") == "user"]
        bots = [m for m in new if m.get("role") == "assistant"]
        return {"users": users, "reply": bots[-1] if bots else None, "assistant_rows": len(bots)}


def voiced_seconds(pcm: np.ndarray, frame: int = FRAME, db: float = -40.0) -> float:
    n = len(pcm) // frame
    f = pcm[:n * frame].astype(np.float64).reshape(n, frame) / 32768
    return float(((20 * np.log10(np.sqrt((f ** 2).mean(1)) + 1e-9)) > db).sum() * frame / SR)


def load_clip(path: Path = CLIP) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == SR, sr
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


# --------------------------------------------------------------------------- #
# uplink ring
# --------------------------------------------------------------------------- #
@dataclass
class RingEntry:
    off: int
    n: int                    # samples
    codec: int
    packets: list[bytes]
    burst: int
    t: float                  # capture time (monotonic)


class UplinkRing:
    """30 s of sent messages. Trims only while idle; replay is per burst and a burst is
    stale only when its LAST sample is older than 20 s (never cut a burst in half)."""

    def __init__(self, seconds: float = RING_S) -> None:
        self.max_samples = int(seconds * SR)
        self.entries: list[RingEntry] = []

    def put(self, e: RingEntry) -> None:
        self.entries.append(e)
        while self.entries and wire3.u32_diff(e.off + e.n, self.entries[0].off) > self.max_samples:
            self.entries.pop(0)

    def trim(self, upto: int) -> None:
        self.entries = [e for e in self.entries if wire3.u32_diff(e.off + e.n, upto) > 0]

    def replay(self, frm: int, now: float) -> tuple[list[bytes], int]:
        """Messages (replay flag, <= 50 packets, contiguous) from offset ``frm``; and how many
        stale bursts were skipped."""
        bursts: dict[int, list[RingEntry]] = {}
        for e in self.entries:
            if wire3.u32_diff(e.off + e.n, frm) > 0:
                bursts.setdefault(e.burst, []).append(e)
        out, stale = [], 0

        def flush(run: list[RingEntry]) -> None:
            if run:
                out.append(wire3.pack_up(run[0].codec, wire3.FLAG_REPLAY, run[0].off,
                                         [p for x in run for p in x.packets]))
                run.clear()

        for es in bursts.values():
            if now - es[-1].t > STALE_S:
                stale += 1
                continue
            run: list[RingEntry] = []
            for e in es:
                if run and (e.codec != run[0].codec or wire3.u32_diff(e.off, run[-1].off + run[-1].n) != 0
                            or sum(len(x.packets) for x in run) + len(e.packets) > wire3.MAX_UP_PACKETS):
                    flush(run)
                run.append(e)
            flush(run)
        return out, stale


# --------------------------------------------------------------------------- #
# downlink: player + phone voice + claims
# --------------------------------------------------------------------------- #
@dataclass
class Sentence:
    sid: str
    gen: int
    seq: int
    text: str
    mid: str | None
    kind: str
    say_at: float
    path: str | None = None           # "server" | "phone"
    first_audio: float | None = None
    chunks: int = 0
    play_start: float | None = None
    play_end: float = 0.0
    complete: bool = False            # no more audio will come (a later say / say_end)
    underruns: int = 0
    gap_max: float = 0.0
    heard_at: float | None = None
    cut: bool = False


class Downlink:
    def __init__(self, t0: float) -> None:
        self.t0 = t0
        self.next_at = 0.0
        self.sentences: list[Sentence] = []
        self.by_key: dict[tuple[str, int], Sentence] = {}
        self.claims: dict[tuple[str, int], str] = {}
        self.dropped_claimed = 0
        self.phone_queue: list[Sentence] = []
        self.phone_busy_until = 0.0
        self.reply_underruns = 0
        self.decoders: dict[tuple[str, int], opus.Decoder] = {}

    def claim(self, sid: str, seq: int, path: str) -> bool:
        k = (sid, seq)
        if k in self.claims:
            return self.claims[k] == path
        self.claims[k] = path
        return True

    def on_say(self, f: dict, sid: str, now: float, text_mode: bool) -> Sentence:
        for s in self.sentences:                     # everything before this one is complete
            if s.sid == sid and s.seq < f["seq"]:
                s.complete = True
        s = Sentence(sid, f.get("gen", 0), f["seq"], f.get("text", ""), f.get("message_id"),
                     f.get("kind", "reply"), now)
        self.sentences.append(s)
        self.by_key[(sid, s.seq)] = s
        if text_mode:
            self.to_phone(s, now)
        return s

    def to_phone(self, s: Sentence, now: float) -> None:
        if s.path is None and self.claim(s.sid, s.seq, "phone"):
            s.path = "phone"
            s.complete = True
            start = max(now, self.phone_busy_until, self.next_at)
            s.play_start = start
            s.play_end = start + len(s.text) / PHONE_CHARS_PER_S
            self.phone_busy_until = s.play_end

    def on_audio(self, sid: str, codec: int, kind: int, seq: int, packets: list[bytes], now: float) -> float:
        if kind == wire3.KIND_PROBE:
            return 0.0
        s = self.by_key.get((sid, seq))
        if s is None or not self.claim(sid, seq, "server"):
            self.dropped_claimed += 1
            return 0.0
        s.path = "server"
        if codec == wire3.CODEC_OPUS:
            dec = self.decoders.setdefault((sid, seq), opus.Decoder(SR_OUT))
            n = sum(len(dec.decode(p)) for p in packets)
        else:
            n = sum(len(p) // 2 for p in packets)
        dur = n / SR_OUT
        start = max(now + LEAD_S, self.next_at)
        if s.chunks and start > self.next_at + 0.001 and self.next_at > 0:
            gap = start - self.next_at
            s.underruns += 1
            s.gap_max = max(s.gap_max, gap)
            self.reply_underruns += 1
        if s.first_audio is None:
            s.first_audio = now
            s.play_start = start
        s.chunks += 1
        self.next_at = start + dur
        s.play_end = self.next_at
        return dur

    def settle(self, now: float, connected: bool) -> None:
        for s in self.sentences:
            if s.heard_at is None and s.complete and s.play_end and now >= s.play_end:
                s.heard_at = s.play_end

    def mark_cut(self) -> None:
        """The socket dropped: a sentence not known complete (no later `say`/`say_end`) is cut —
        its received audio still plays, then it stops; it comes again in full. A `say` that
        never got audio or a phone claim is forgotten (the server sends it again)."""
        for s in self.sentences:
            if s.heard_at is None and not s.complete:
                s.cut = True
                s.complete = True
                if s.path is None:
                    s.path = "lost"

    def fully_played(self, mid: str | None, base: int) -> int:
        """`after_partial.sentences`: leading sentences of ``mid`` the listener WILL hear in full.

        A sentence whose audio arrived complete is counted even while it still waits in
        the player: the player keeps playing across a reconnect, so asking for it again
        would play it twice (cross-session dedup is after_partial's job, C.5 #9). A
        sentence cut by the drop is not counted: it comes again from its start."""
        n = base
        for s in [x for x in self.sentences if x.mid == mid and x.kind == "reply"]:
            if s.complete and not s.cut and s.path is not None:
                n += 1
            else:
                break
        return n


# --------------------------------------------------------------------------- #
# the client
# --------------------------------------------------------------------------- #
class WireClient:
    def __init__(self, ws_base: str, thread: str, *, up: str, down: str, compression: str | None,
                 slow_link: str = "phone", headers: dict | None = None, send_after: bool = True) -> None:
        self.ws_base = ws_base
        # drive.js anchors a reconnect's `after` on the user row the cut reply answers
        # (Amendment T7b-3 / review-T7b H1); `send_after=False` models a client without it
        self.send_after = send_after
        self.turn_id: str | None = None
        self.headers = dict(headers or {})
        self.thread = thread
        self.up = up
        self.down = down
        self.compression = compression
        self.slow_link = slow_link
        self.device = "dev-" + secrets.token_hex(6)
        self.stream = "str-" + secrets.token_hex(8)
        self.t0 = time.monotonic()
        self.dl = Downlink(self.t0)
        self.ring = UplinkRing()
        self.enc = opus.Encoder(SR, 16000, application="voip") if up == "opus" else None
        self.burst = 0
        self.gate_open = False
        self.pending: list[tuple[int, bytes]] = []
        # connection state
        self.ws = None
        self.sid: str | None = None
        self.connected = asyncio.Event()
        self.live = False
        self.outq: asyncio.Queue[bytes] = asyncio.Queue()
        self.ready: dict | None = None
        self.online_now = asyncio.Event()
        self.closed_for_good = False
        # cursors
        self.last_rx_off = 0
        self.last_transcript_upto = 0
        self.last_eot_upto = 0
        self.cursor_answers: list[int] = []
        # link monitor
        self.mode = "audio"
        self.link_sent: list[dict] = []
        self.deferred: dict | None = None
        self.rtt_ewma: float | None = None
        self.rtt_bad = 0
        self.rtts: list[float] = []
        self.last_ping_t: float | None = None
        self.last_pong_at = time.monotonic()
        self.unanswered = 0
        # bookkeeping
        self.events: list[tuple[float, str, object]] = []
        self.transcripts: list[dict] = []
        self.reconnects = 0
        self.connects = 0
        self.refused = 0
        self.replayed_msgs = 0
        self.stale_bursts = 0
        self.say_end: dict[str, float] = {}
        self.current_mid: str | None = None
        self.mid_base: dict[str, int] = {}
        self.sent_done: set = set()
        self.bytes_up = 0
        self.bytes_down = 0
        self.errors: list[dict] = []
        self.marks: dict[str, float] = {}
        self.drops_mid_reply = 0

    # ------------------------------------------------------------------ log #
    def rel(self, t: float | None = None) -> float:
        return round((time.monotonic() if t is None else t) - self.t0, 3)

    def ev(self, what: str, data: object = None) -> None:
        self.events.append((self.rel(), what, data))

    def mark(self, k: str) -> None:
        self.marks.setdefault(k, time.monotonic())

    # ---------------------------------------------------------------- uplink #
    def capture(self, k: int, pcm: np.ndarray, gate: bool) -> None:
        """Frame ``k`` (20 ms at offset k*320) from the microphone."""
        off = (k * FRAME) & wire3.U32
        if gate and not self.gate_open:
            self.gate_open = True
            self.burst += 1
        if not gate:
            if self.gate_open:
                self.gate_open = False
                self._flush_pending()
            return
        if self.enc is not None:
            self.pending.append((off, self.enc.encode(pcm)))
            if len(self.pending) >= 5:
                self._flush_pending()
        else:
            self._emit(RingEntry(off, FRAME, wire3.CODEC_PCM16, [pcm.astype("<i2").tobytes()], self.burst,
                                 time.monotonic()))

    def _flush_pending(self) -> None:
        if not self.pending:
            return
        off = self.pending[0][0]
        e = RingEntry(off, FRAME * len(self.pending), wire3.CODEC_OPUS, [p for _, p in self.pending],
                      self.burst, time.monotonic())
        self.pending = []
        self._emit(e)

    def _emit(self, e: RingEntry) -> None:
        self.ring.put(e)
        if self.live:
            self.outq.put_nowait(wire3.pack_up(e.codec, 0, e.off, e.packets))

    def cursor_from(self) -> int:
        """C.5 #10: max(lastRxOff, lastTranscriptUpto, lastEotUpto) on the u32 ring."""
        best = self.last_rx_off
        for v in (self.last_transcript_upto, self.last_eot_upto):
            if wire3.u32_diff(v, best) > 0:
                best = v
        return best

    # ------------------------------------------------------------ connection #
    def url(self) -> str:
        return (f"{self.ws_base}/ws/voice/{self.thread}?d={self.device}&s={self.stream}&v=3"
                f"&up={self.up},pcm16&down={self.down},pcm16")

    async def connection_loop(self) -> None:
        import websockets
        backoff = 0.5
        first = True
        while not self.closed_for_good:
            try:
                ws = await websockets.connect(self.url(), compression=self.compression, max_size=None,
                                              ping_interval=None, open_timeout=10, close_timeout=1,
                                              proxy=None, additional_headers=self.headers or None)
            except (TimeoutError, OSError, websockets.exceptions.InvalidHandshake, websockets.exceptions.ConnectionClosed) as e:
                self.refused += 1
                self.ev("connect_failed", repr(e)[:80])
                self.online_now.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self.online_now.wait(), backoff)
                backoff = min(8.0, backoff * 2)
                continue
            self.connects += 1
            if not first:
                self.reconnects += 1
            first = False
            got_ready = False
            try:
                got_ready = await self.session(ws)
            finally:
                if got_ready and self.current_mid is not None and self.current_mid not in self.say_end:
                    self.drops_mid_reply += 1
                self.connected.clear()
                self.live = False
                while not self.outq.empty():
                    self.outq.get_nowait()
                self.dl.mark_cut()
                with contextlib.suppress(Exception):
                    ws.transport.abort()
                self.ev("disconnected")
            if got_ready:
                backoff = 0.5
            else:
                # No `ready` (review-T3s4 #6: the socket opened but the session never came,
                # on a link still choked by the dropped one): a failed connection — back off
                # and try again, like a refused one; never the end of the client.
                self.refused += 1
                self.online_now.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self.online_now.wait(), backoff)
                backoff = min(8.0, backoff * 2)

    async def session(self, ws) -> bool:
        """One socket's life. False when it never got `ready` (the caller retries)."""
        import websockets
        self.ws = ws
        self.ev("connected", ws.response.headers.get("Sec-WebSocket-Extensions"))
        try:
            ready = json.loads(await asyncio.wait_for(ws.recv(), 15))
        except (TimeoutError, websockets.exceptions.ConnectionClosed, ValueError) as e:
            self.ev("no_ready", repr(e)[:80])
            return False
        if not isinstance(ready, dict) or ready.get("type") != "ready":
            self.ev("no_ready", str(ready)[:80])
            return False
        self.ready = ready
        self.sid = ready.get("sid") or secrets.token_hex(8)
        self.mark("ready")
        resuming = self.connects > 1
        start = {"type": "start", "patience": "normal", "barge_in": True, "reply_style": "normal",
                 "follow_up": False, "acks": "off", "slow_link": self.slow_link}
        if resuming:
            start["cursor"] = {"stream": self.stream, "from": self.cursor_from()}
            mid = self.current_mid
            if mid is not None and mid not in self.say_end:
                n = self.dl.fully_played(mid, self.mid_base.get(mid, 0))
                start["after_partial"] = {"message_id": mid, "sentences": n}
                if self.send_after and self.turn_id:
                    start["after"] = self.turn_id
                self.mid_base[mid] = n
                for s in self.dl.sentences:      # the new session's sentences of `mid` count from n
                    if s.mid == mid:
                        s.mid = mid + "#" + (s.sid or "")[:6]
        self.ev("start", {k: v for k, v in start.items() if k in ("cursor", "after_partial", "after")})
        await ws.send(json.dumps(start))
        if self.mode == "text":                   # a new session starts in audio: say it again
            why = self.link_sent[-1].get("why", "stall") if self.link_sent else "stall"
            await ws.send(json.dumps({"type": "link", "mode": "text", "why": why}))
        reader = asyncio.create_task(self.reader(ws))
        pinger = asyncio.create_task(self.pinger(ws))
        sender = asyncio.create_task(self.sender(ws))
        tasks = {reader, pinger, sender}
        try:
            if resuming:
                frm = await self.wait_cursor(3.0)
                msgs, stale = self.ring.replay(frm if frm is not None else self.cursor_from(), time.monotonic())
                self.stale_bursts += stale
                for m in msgs:
                    await ws.send(m)
                    self.bytes_up += len(m)
                    self.replayed_msgs += 1
                self.ev("replayed", {"from": frm, "messages": len(msgs), "stale_bursts": stale})
            # go live: anything captured from now on is sent as it comes
            last = self.ring.entries[-1].off if self.ring.entries else None
            self.live = True
            if resuming and last is not None:
                # entries that arrived during the replay itself
                for e in self.ring.entries:
                    if wire3.u32_diff(e.off, last) > 0:
                        self.outq.put_nowait(wire3.pack_up(e.codec, 0, e.off, e.packets))
            self.connected.set()
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return True

    async def wait_cursor(self, timeout: float) -> int | None:
        n = len(self.cursor_answers)
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            if len(self.cursor_answers) > n:
                return self.cursor_answers[-1]
            await asyncio.sleep(0.02)
        return None

    async def sender(self, ws) -> None:
        while True:
            m = await self.outq.get()
            await ws.send(m)
            self.bytes_up += len(m)

    async def pinger(self, ws) -> None:
        self.last_pong_at = time.monotonic()
        self.unanswered = 0
        while True:
            now = time.monotonic()
            dead_after = max(8.0, 4 * self.rtt_ewma / 1000 if self.rtt_ewma and self.rtt_ewma > 1500 else 0.0)
            if self.unanswered >= 3 and now - self.last_pong_at > dead_after:
                self.ev("dead", round(now - self.last_pong_at, 2))
                return
            if self.last_ping_t is not None and now - self.last_pong_at > PONG_LATE_S \
                    and self.mode == "audio" and self.unanswered >= 2:
                await self.degrade("rtt", reactive=False)
            self.unanswered += 1
            self.last_ping_t = now
            with contextlib.suppress(Exception):
                await asyncio.wait_for(ws.send(json.dumps({"type": "ping", "t": round(now - self.t0, 4)})), 1.0)
            await asyncio.sleep(PING_S)

    # --------------------------------------------------------------- downlink #
    async def reader(self, ws) -> None:
        async for m in ws:
            now = time.monotonic()
            if isinstance(m, bytes):
                self.bytes_down += len(m)
                try:
                    codec, kind, _gen, seq, _t, packets = wire3.unpack_down(m)
                except ValueError as e:
                    self.errors.append({"bad_down": str(e)})
                    continue
                if kind == wire3.KIND_REPLY:
                    self.mark("first_reply_audio")
                s = self.dl.by_key.get((self.sid, seq))
                self.dl.on_audio(self.sid, codec, kind, seq, packets, now)
                if s is not None and s.path == "server" and s.underruns and \
                        (s.underruns >= UNDERRUNS_PER_REPLY or s.gap_max > GAP_BAD_S):
                    await self.degrade("underrun", reactive=True)
                continue
            self.bytes_down += len(m)
            f = json.loads(m)
            typ = f.get("type")
            if typ == "pong":
                rtt = (now - self.t0 - float(f["t"])) * 1000
                self.rtts.append(rtt)
                self.rtt_ewma = rtt if self.rtt_ewma is None else 0.7 * self.rtt_ewma + 0.3 * rtt
                self.rtt_bad = self.rtt_bad + 1 if self.rtt_ewma > RTT_EWMA_BAD_MS else 0
                self.last_pong_at = now
                self.unanswered = 0
                if isinstance(f.get("rx_off"), int) and wire3.u32_diff(f["rx_off"], self.last_rx_off) > 0:
                    self.last_rx_off = f["rx_off"]
                if self.rtt_bad >= RTT_BAD_COUNT:
                    await self.degrade("rtt", reactive=False)
            elif typ == "say":
                self.mark("first_say")
                s = self.dl.on_say(f, self.sid, now, self.mode == "text")
                if s.kind == "reply" and s.mid:
                    self.current_mid = s.mid
                asyncio.create_task(self.deadline(s))
            elif typ == "audio_end":           # Amendment T6-2: this sentence's audio is complete
                s = self.dl.by_key.get((self.sid, f.get("seq")))
                if s is not None:
                    s.complete = True
            elif typ == "say_end":
                for s in self.dl.sentences:
                    if s.sid == self.sid:
                        s.complete = True
                self.say_end[f.get("message_id")] = now
                # The real session's `say_end.message_id` is the ROW id once the stream is
                # done, while its `say` frames carried the stream's provisional id (C.4,
                # Amendment T3s2-1): the reply it closes is the one its `last_seq` belongs
                # to (or, when every sentence was heard before a reconnect, the current one).
                last = self.dl.by_key.get((self.sid, f.get("last_seq")))
                if last is not None and last.mid:
                    self.say_end.setdefault(last.mid, now)
                elif self.current_mid is not None:
                    self.say_end.setdefault(self.current_mid, now)
                if self.deferred is not None:
                    d, self.deferred = self.deferred, None
                    await self.send_link(d)
            elif typ == "eot":
                self.mark("eot")
                if isinstance(f.get("upto"), int):
                    self.last_eot_upto = f["upto"]
            elif typ == "transcript":
                self.mark("transcript")
                if isinstance(f.get("message_id"), str) and f["message_id"]:
                    self.turn_id = f["message_id"]
                self.transcripts.append({"t": self.rel(now), "text": f.get("text"),
                                         "replayed": bool(f.get("replayed")), "upto": f.get("upto")})
                if isinstance(f.get("upto"), int):
                    self.last_transcript_upto = f["upto"]
                if isinstance(f.get("upto"), int):    # committed: the ring below it is not needed
                    self.ring.trim(f["upto"])
            elif typ == "cursor":
                self.cursor_answers.append(int(f["from"]))
            elif typ in ("error", "turn_error"):
                self.errors.append(f)
            elif typ == "state":
                self.ev("state", f.get("state"))

    async def deadline(self, s: Sentence) -> None:
        firsts = [x.first_audio - x.say_at for x in self.dl.sentences if x.first_audio]
        p90 = sorted(firsts)[int(0.9 * (len(firsts) - 1))] if firsts else 0.0
        await asyncio.sleep(max(DEADLINE_MIN_S, p90 + 2.0))
        if s.path is None and s.sid == self.sid and self.mode == "audio" and self.connected.is_set():
            await self.degrade("deadline", reactive=True)

    async def degrade(self, why: str, *, reactive: bool) -> None:
        if self.mode == "text" or self.slow_link not in ("phone", "text"):
            return
        frame = {"type": "link", "mode": "text", "why": why}
        if reactive:
            await self.send_link(frame)
        elif self.deferred is None:
            self.deferred = frame
            # nothing playing: no reply to finish first
            if not any(s.sid == self.sid and not s.complete for s in self.dl.sentences):
                d, self.deferred = self.deferred, None
                await self.send_link(d)

    async def send_link(self, frame: dict) -> None:
        if self.mode == "text":
            return
        self.mode = "text"
        self.link_sent.append({**frame, "t": self.rel()})
        self.ev("link", frame)
        now = time.monotonic()
        for s in self.dl.sentences:              # sentences waiting for their audio: the phone speaks them
            if s.path is None and s.sid == self.sid:
                self.dl.to_phone(s, now)
        if self.ws is not None:
            with contextlib.suppress(Exception):
                await self.ws.send(json.dumps(frame))

    async def reporter(self) -> None:
        """playback_done when the modelled queue drains; settles 'heard'."""
        while True:
            await asyncio.sleep(0.05)
            now = time.monotonic()
            self.dl.settle(now, self.connected.is_set())
            if not self.connected.is_set() or self.ws is None:
                continue
            mine = [s for s in self.dl.sentences if s.sid == self.sid and s.path]
            if not mine:
                continue
            last = mine[-1]
            if now >= max(self.dl.next_at, self.dl.phone_busy_until) + 0.05 and last.complete \
                    and (self.sid, last.gen, last.seq) not in self.sent_done:
                self.sent_done.add((self.sid, last.gen, last.seq))
                with contextlib.suppress(Exception):
                    await self.ws.send(json.dumps({"type": "playback_done", "gen": last.gen, "seq": last.seq}))


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
async def mic(client: WireClient, clip: np.ndarray, start_speech: asyncio.Event, speech_marks: dict,
              hangover_s: float, stop: asyncio.Event) -> None:
    """Real-time 20 ms frames; the gate opens 300 ms before the speech and closes hangover_s after."""
    loop = asyncio.get_running_loop()
    t_base = loop.time()
    k = 0
    seg = None            # (first frame index of the gated segment, pcm with pre-roll + clip + hangover)
    rng = np.random.default_rng(7)
    while not stop.is_set():
        due = t_base + k * 0.02
        delay = due - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        if seg is None and start_speech.is_set() and "start" not in speech_marks:
            pre = (rng.standard_normal(int(0.3 * SR)) * 30).astype(np.int16)
            post = (rng.standard_normal(int(hangover_s * SR)) * 30).astype(np.int16)
            seg = (k, np.concatenate([pre, clip, post]))
            speech_marks["start"] = time.monotonic() + 0.3
            speech_marks["start_off"] = (k * FRAME + int(0.3 * SR))
            speech_marks["end_at_frame"] = k + (len(pre) + len(clip)) // FRAME
        if seg is not None:
            i = k - seg[0]
            frame = seg[1][i * FRAME:(i + 1) * FRAME]
            if len(frame) == FRAME:
                client.capture(k, frame, True)
                if k == speech_marks["end_at_frame"]:
                    speech_marks["end"] = time.monotonic()
            else:
                client.capture(k, np.zeros(FRAME, np.int16), False)
                seg = None
                speech_marks["gate_closed"] = time.monotonic()
        else:
            client.capture(k, np.zeros(FRAME, np.int16), False)
        k += 1


async def run_one(profile: str, *, up: str, down: str, compression: str | None, server_port: int,
                  proxy_port: int, thread: str | None, server_http: str | None, clip: np.ndarray,
                  timeout_s: float = 120.0, judge: DispatchJudge | None = None, send_after: bool = True) -> dict:
    spec = shaper.PROFILES[profile]
    before: set[str] = set()
    if judge is not None:
        await asyncio.to_thread(judge.login)
        thread = await asyncio.to_thread(judge.new_thread)
        before = await asyncio.to_thread(judge.snapshot, thread)
    proxy = shaper.make_proxy(profile, proxy_port, server_port)
    await proxy.start()
    client = WireClient(f"ws://127.0.0.1:{proxy_port}", thread, up=up, down=down, compression=compression,
                        headers=judge.ws_headers() if judge is not None else None, send_after=send_after)
    start_speech = asyncio.Event()
    stop = asyncio.Event()
    speech: dict = {}
    res: dict = {"profile": profile, "up": up, "down": down, "compression": compression or "raw",
                 "judge": "dispatch" if judge is not None else "ref", "thread": thread,
                 "client_after": send_after}
    conn = asyncio.create_task(client.connection_loop())
    rep = asyncio.create_task(client.reporter())
    mic_task = None
    fired: list[tuple] = []

    async def scheduler():
        pending = list(spec["events"])
        while pending:
            await asyncio.sleep(0.02)
            now = time.monotonic()
            for evt in list(pending):
                anchor, after, action, arg = evt
                t_anchor = speech.get("start") if anchor == "speech1" else \
                    min([client.marks[k] for k in ("first_say", "first_reply_audio") if k in client.marks],
                        default=None)
                if t_anchor is not None and now >= t_anchor + after:
                    proxy.apply(action, arg)
                    fired.append((client.rel(now), action, arg))
                    pending.remove(evt)
                    if action in ("blackout", "stall"):
                        asyncio.create_task(online_after(float(arg)))

    async def online_after(s: float):
        await asyncio.sleep(s + 0.05)
        client.online_now.set()          # navigator 'online' when the radio is back

    sched = asyncio.create_task(scheduler())
    try:
        await asyncio.wait_for(client.connected.wait(), 30)
        res["connect_s"] = round(client.marks["ready"] - client.t0, 3)
        hang = float((client.ready or {}).get("gate", {}).get("hangover_s", 2.6))
        mic_task = asyncio.create_task(mic(client, clip, start_speech, speech, hang, stop))
        await asyncio.sleep(1.0)
        start_speech.set()
        t_end = time.monotonic() + timeout_s

        def finished() -> bool:
            mid = client.current_mid
            if not sched.done() or mid is None or mid not in client.say_end:
                return False
            if time.monotonic() < max(client.dl.next_at, client.dl.phone_busy_until) + 0.3:
                return False
            mine = [s for s in client.dl.sentences if s.mid == mid and s.kind == "reply"]
            return bool(mine) and all(s.heard_at or s.cut for s in mine)

        while time.monotonic() < t_end and not finished():
            await asyncio.sleep(0.1)
        res["timed_out"] = time.monotonic() >= t_end
    finally:
        stop.set()
        client.closed_for_good = True
        for t in (sched, rep, mic_task, conn):
            if t is not None:
                t.cancel()
        await asyncio.gather(*(t for t in (sched, rep, mic_task, conn) if t), return_exceptions=True)
        with contextlib.suppress(Exception):
            if client.ws is not None:
                client.ws.transport.abort()
        res["proxy"] = proxy.stats()
        await proxy.stop()

    # ------------------------------------------------------------- results #
    heard = [s for s in client.dl.sentences if s.kind == "reply" and s.heard_at and not s.cut]
    res["heard"] = [{"text": s.text[:40], "path": s.path, "sid": s.sid[:6], "underruns": s.underruns} for s in heard]
    texts = [s.text for s in heard]
    res["heard_twice"] = sorted({t[:40] for t in texts if texts.count(t) > 1})
    rows: dict = {}
    if judge is not None:
        got = [w for t in texts for w in words(t)]
        for _ in range(6):        # a real server persists the assistant row as it streams: let it settle
            out = await asyncio.to_thread(judge.outcome, thread, before)
            reply = out["reply"]
            expect = spoken_words(reply.get("content") or "") if reply else []
            if expect == got:
                break
            await asyncio.sleep(1.0)
        res["rows"] = [{"text": m.get("content"), "id": m.get("id")} for m in out["users"]]
        res["reply_row"] = {"id": reply.get("id"), "chars": len(reply.get("content") or "")} if reply else None
        res["reply_sentences"] = len(reply_texts_of(reply))
        res["reply_words"] = {"expected": len(expect), "heard": len(got)}
        res["heard_all"] = bool(expect) and got == expect
        res["heard_in_order"] = res["heard_all"]
        if expect and got != expect:
            i = next((k for k, (a, b) in enumerate(zip(got, expect, strict=False)) if a != b), min(len(got), len(expect)))
            res["reply_mismatch"] = {"at_word": i, "heard": got[max(0, i - 3):i + 5],
                                     "expected": expect[max(0, i - 3):i + 5]}
        res["known_text"] = known_text()
        res["server_resumes"] = []
        mids = {(s.mid or "").split("#")[0] for s in heard}
        res["reply_sids"] = len({s.sid for s in heard if s.mid and s.mid.split("#")[0] in mids})
    else:
        if server_http:
            with contextlib.suppress(Exception):
                rows = await asyncio.to_thread(fetch_json, f"{server_http}/rows/{thread}")
        res["rows"] = [{k: r.get(k) for k in ("text", "voiced_s", "dur_s", "upto", "start")}
                       for r in rows.get("rows", [])]
        reply_texts = []
        for r in rows.get("replies", {}).values():
            reply_texts = r["sentences"]
        res["reply_sentences"] = len(reply_texts)
        res["heard_in_order"] = texts == [t for t in reply_texts if t in texts]
        res["heard_all"] = bool(reply_texts) and set(reply_texts) <= set(texts)
        res["server_resumes"] = [x["resume"] for x in rows.get("log", []) if "resume" in x]
    res["cut_sentences"] = sum(1 for s in client.dl.sentences if s.cut)
    res["paths"] = {p: sum(1 for s in heard if s.path == p) for p in ("server", "phone")}
    res["underruns"] = sum(s.underruns for s in heard)
    res["gap_max_s"] = round(max([s.gap_max for s in heard] or [0.0]), 3)
    res["dropped_claimed"] = client.dl.dropped_claimed
    res["link_sent"] = client.link_sent
    res["mode_end"] = client.mode
    res["reconnects"] = client.reconnects
    res["drops_mid_reply"] = client.drops_mid_reply
    res["refused_connects"] = client.refused
    res["replayed_msgs"] = client.replayed_msgs
    res["stale_bursts"] = client.stale_bursts
    res["cursor_answers"] = client.cursor_answers
    res["transcripts"] = client.transcripts
    res["errors"] = client.errors[:5]
    res["events_fired"] = fired
    res["rtt_ms"] = {"n": len(client.rtts), "median": round(statistics.median(client.rtts)) if client.rtts else None,
                     "max": round(max(client.rtts)) if client.rtts else None}
    res["bytes"] = {"up": client.bytes_up, "down": client.bytes_down}
    m = client.marks
    lat = {}
    if "end" in speech:
        for k in ("eot", "transcript", "first_say", "first_reply_audio"):
            if k in m:
                lat[f"speech_end_to_{k}"] = round(m[k] - speech["end"], 3)
        if "eot" in m and "first_say" in m:
            lat["eot_to_first_say"] = round(m["first_say"] - m["eot"], 3)
        heard_end = max([s.heard_at for s in heard] or [0.0])
        if heard_end:
            lat["speech_end_to_reply_heard"] = round(heard_end - speech["end"], 2)
    res["latency_s"] = lat
    res["clip_voiced_s"] = round(voiced_seconds(clip), 3)
    res["events"] = client.events[-60:]
    res["assertions"] = check(res)
    return res


def reply_texts_of(reply: dict | None) -> list[str]:
    if not reply:
        return []
    try:
        from dispatch_voice.host.session import split_reply
    except ImportError:                               # pragma: no cover
        return [reply.get("content") or ""]
    return [sp.text for sp in split_reply(reply.get("content") or "")]


def full_sentence_text(text: str, ref: str) -> bool:
    """The row carries the whole utterance: WER <= 0.15 and the first word present (a cut
    head loses the first words; one mid-sentence slip of the recogniser is tolerated)."""
    rw, hw = words(ref), words(text)
    return bool(rw) and wer(text, ref) <= WER_MAX and bool(hw) and hw[0] == rw[0]


def check(r: dict) -> dict:
    out = {}
    rows = r["rows"]
    out["one_user_row"] = len(rows) == 1
    if not rows:
        out["full_sentence"] = False
    elif r.get("judge") == "dispatch":
        out["full_sentence"] = full_sentence_text(rows[0].get("text") or "", r["known_text"])
    else:
        v = rows[0].get("voiced_s") or 0
        out["full_sentence"] = abs(v - r["clip_voiced_s"]) <= max(0.1 * r["clip_voiced_s"], 0.15)
    out["transcript_reached_client"] = len(r["transcripts"]) >= 1
    out["reply_complete_once_in_order"] = r["heard_all"] and not r["heard_twice"] and r["heard_in_order"]
    out["no_timeout"] = not r.get("timed_out")
    p = r["profile"]
    if p == "3g-weak" and r["down"] == "pcm16":
        out["text_mode_entered"] = any(x.get("mode") == "text" for x in r["link_sent"])
    if p == "edge":
        out["first_say_within_eot_plus_6s"] = r["latency_s"].get("eot_to_first_say", 99) <= 6.0
    if p == "blackout-20s":
        out["reconnected"] = r["reconnects"] >= 1
    if p == "stall-5s":
        if r["drops_mid_reply"] < 1:
            # The whole reply arrived before the stall (the client was on the phone voice, or
            # the reply was short): the rejoin was not exercised. Reported, never a pass and
            # never a failure (review-T3s4 #6) — the summary says so and the run is counted apart.
            r["not_exercised"] = ["rejoined_mid_reply"]
        elif r.get("judge") == "dispatch":
            # judged by the client (a real server keeps no resume log): the drop came
            # mid-reply and the reply's heard sentences came from two sessions
            out["rejoined_mid_reply"] = r.get("reply_sids", 0) >= 2
        else:
            # the late-joiner case must not pass by accident: the drop came mid-reply and the
            # server resumed it while the agent's stream was still open
            out["rejoined_mid_reply"] = any(x.get("stream_open") for x in r.get("server_resumes", []))
    return out


# --------------------------------------------------------------------------- #
# matrix
# --------------------------------------------------------------------------- #
def start_refserver(port: int, state: Path, reply: str, stt: str, log: Path,
                    extra: tuple[str, ...] = ()) -> subprocess.Popen:
    state.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, str(HERE / "transport" / "refserver.py"), "--port", str(port), "--state", str(state),
            "--reply", reply, "--stt", stt, *extra]
    with open(log, "ab") as f:          # the child keeps its own copy of the descriptor
        proc = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
    for _ in range(200):
        with contextlib.suppress(Exception):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"reference server exited ({proc.returncode}); see {log}")
        time.sleep(0.1)
    proc.terminate()
    raise RuntimeError("reference server did not start")


def write_pin(path: Path) -> str:
    """A fresh scratch PIN in a 0600 file (passed by path, never on argv)."""
    pin = "".join(secrets.choice("0123456789") for _ in range(8))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(pin)
    from dispatch_voice import fsperm
    fsperm.ensure_private(path)            # Windows: the ACL (the mode above means nothing there)
    return pin


def start_scratch_dispatch(a, out_dir: Path):
    """DisPatch from ``a.start_dispatch`` on ``a.server_port`` with the add-on, one API bot on
    the stub LLM's long reply (streamed slowly, so a mid-reply drop finds it open), and a
    scratch PIN set through /api/auth/setup. Returns (proc, llm, judge)."""
    from e2e_voice_roundtrip import start_dispatch
    from stub_llm import StubLLM
    fixed = None
    if a.fixed_reply == "split":
        fixed = SPLIT_REPLY
    elif a.fixed_reply:
        fixed = Path(a.fixed_reply[1:]).read_text(encoding="utf-8").strip() if a.fixed_reply.startswith("@") else a.fixed_reply
    llm = StubLLM(port=a.llm_port, delay_s=1.2, reply="long", tokens_per_s=8.0, fixed_reply=fixed).start()
    data = out_dir / "dispatch-data"
    proc = start_dispatch(a.server_port, data, a.start_dispatch, llm.base_url, online=a.online, kokoro=a.kokoro,
                          power_hold=a.power_hold, bot_kind=a.bot_kind)
    pin = write_pin(out_dir / "pin")
    judge = DispatchJudge(f"http://127.0.0.1:{a.server_port}", bot="voicetest")
    code, _h, _ = judge._req("POST", "/api/auth/setup", {"new_pin": pin})
    if code != 200:
        raise RuntimeError(f"could not set the scratch PIN ({code})")
    judge.pin = pin                      # every run logs in again, like a phone would
    settle_scratch(f"http://127.0.0.1:{a.server_port}", pin)
    return proc, llm, judge


def settle_scratch(base: str, pin: str, timeout_s: float = 240.0) -> dict:
    """Wait until the fresh server's start-up work is over: the prewarm (``warm``) and
    the acknowledgement renders it queues (every ``acks`` row ``ready + failed >= total``).
    Without this the first run of a matrix shares the CPU (and the GIL: the phonemizer
    is Python) with that work, its pongs come back ~2 s late, the client's link monitor
    goes to text mode, and the run is "not exercised" (V1). The live unit prewarms at
    boot, long before anyone talks to it, so this is the realistic state."""
    j = DispatchJudge(base, pin=pin)
    j.login()
    t0 = time.monotonic()
    st: dict = {}
    while time.monotonic() - t0 < timeout_s:
        code, _h, body = j._req("GET", "/api/voice/status")
        st = body if code == 200 and isinstance(body, dict) else {}
        acks = st.get("acks") or {}
        rendered = all((r.get("ready", 0) + r.get("failed", 0)) >= r.get("total", 0)
                       for r in acks.values() if isinstance(r, dict))
        if st.get("warm") is True and rendered:
            print(f"scratch server settled in {time.monotonic() - t0:.1f} s (warm, acks {acks})", flush=True)
            return st
        time.sleep(0.5)
    print(f"scratch server NOT settled after {timeout_s:.0f} s: warm={st.get('warm')} acks={st.get('acks')}",
          flush=True)
    return st


def heard_of(r: dict) -> str:
    if r.get("judge") == "dispatch":       # sentences heard; words heard / in the assistant row
        w = r.get("reply_words") or {}
        return f"{len(r['heard'])} ({w.get('heard', 0)}/{w.get('expected', 0)} words)"
    return f"{len(r['heard'])}/{r['reply_sentences']}"


def summary_line(r: dict) -> str:
    a = r["assertions"]
    ok = all(a.values())
    lat = r["latency_s"]
    verdict = ("FAIL" if not ok else "N/E " if r.get("not_exercised") else "PASS")
    return (f"{verdict}  {r['profile']:<12} {r['up']}/{r['down']:<6} {r['compression']:<7} "
            f"eot {lat.get('speech_end_to_eot', '-')!s:>6}  say {lat.get('eot_to_first_say', '-')!s:>6}  "
            f"audio {lat.get('speech_end_to_first_reply_audio', '-')!s:>6}  heard {heard_of(r)} "
            f"{r['paths']}  mode {r['mode_end']}  reconn {r['reconnects']}  replay {r['replayed_msgs']}  "
            f"under {r['underruns']}  rtt {r['rtt_ms']['median']}/{r['rtt_ms']['max']}  "
            f"up {r['bytes']['up'] // 1024} KB down {r['bytes']['down'] // 1024} KB"
            + ("" if ok else "  failed: " + ",".join(k for k, v in a.items() if not v))
            + ("  not exercised: " + ",".join(r["not_exercised"]) if r.get("not_exercised") else ""))


def matrix(profiles, codecs, compressions) -> list[tuple[str, str, str, str | None]]:
    runs = []
    for p in profiles:
        for c in codecs:
            for comp in compressions:
                runs.append((p, c, c, None if comp == "raw" else "deflate"))
    # the PCM downlink on the weak links: the case text mode exists for
    if "opus" in codecs and "pcm16" not in codecs:
        for p in ("3g-weak", "edge"):
            if p in profiles:
                for comp in compressions:
                    runs.append((p, "opus", "pcm16", None if comp == "raw" else "deflate"))
    return runs


async def main_async(a) -> int:
    clip = load_clip()
    out_dir = a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    proc = None
    llm = None
    server_http = None
    judge: DispatchJudge | None = None
    try:
        if a.start_dispatch:
            server_port = a.server_port
            proc, llm, judge = await asyncio.to_thread(start_scratch_dispatch, a, out_dir)
        elif a.server:
            server_port = int(a.server.rsplit(":", 1)[1].split("/")[0])
            server_http = a.server_http
            if a.judge != "ref":
                base = a.base or ("http" + a.server[2:]).split("/ws/")[0]     # ws://h:p -> http://h:p
                judge = DispatchJudge(base.rstrip("/"), pin=read_pin(a), cookie=os.environ.get("DISPATCH_COOKIE"),
                                      bot=a.bot, thread=a.thread)
        elif a.ref_dispatch:
            server_port = a.server_port
            pin_file = out_dir / "pin"
            pin = write_pin(pin_file)
            proc = start_refserver(server_port, out_dir / "state", a.reply, "oracle", out_dir / "refserver.log",
                                   ("--dispatch-api", "--pin-file", str(pin_file), "--oracle-text", known_text(),
                                    "--oracle-voiced", str(round(voiced_seconds(clip), 3)),
                                    *(("--no-open-mark",) if a.no_open_mark else ())))
            judge = DispatchJudge(f"http://127.0.0.1:{server_port}", pin=pin, bot="refbot")
        else:
            server_port = a.server_port
            proc = start_refserver(server_port, out_dir / "state", a.reply, a.stt, out_dir / "refserver.log",
                                   ("--no-open-mark",) if a.no_open_mark else ())
            server_http = f"http://127.0.0.1:{server_port}"
        for p, up, down, comp in matrix(a.profiles, a.codecs, a.compression):
            thread = None if judge is not None else (a.thread or f"t-{secrets.token_hex(4)}")
            r = await run_one(p, up=up, down=down, compression=comp, server_port=server_port,
                              proxy_port=a.proxy_port, thread=thread, server_http=server_http, clip=clip,
                              timeout_s=a.timeout, judge=judge, send_after=not a.no_after)
            results.append(r)
            print(summary_line(r), flush=True)
            (out_dir / "e2e_lowbw.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
            await asyncio.sleep(1.0)
    finally:
        if proc is not None:
            proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(10)
        if llm is not None:
            llm.stop()
    failed = [r for r in results if not all(r["assertions"].values())]
    ne = [r for r in results if r.get("not_exercised") and r not in failed]
    print(f"\n{len(results) - len(failed) - len(ne)}/{len(results)} runs passed"
          + (f", {len(ne)} not exercised (re-run them)" if ne else "")
          + f"; results in {out_dir / 'e2e_lowbw.json'}")
    return 1 if failed else 0


def read_pin(a) -> str | None:
    if a.pin_file:
        return Path(a.pin_file).read_text(encoding="utf-8").strip() or None
    return os.environ.get("DISPATCH_PIN") or None


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Low-bandwidth e2e over the shaper")
    ap.add_argument("profiles", nargs="*", default=shaper.ORDER)
    ap.add_argument("--codecs", default="opus", help="uplink/downlink codec list: opus,pcm16")
    ap.add_argument("--compression", default="deflate,raw")
    ap.add_argument("--server", help="ws://host:port of a running server (default: start the reference server)")
    ap.add_argument("--server-http", help="http://host:port exposing /rows/{thread} (reference server, --judge ref)")
    ap.add_argument("--judge", choices=["dispatch", "ref"], default="dispatch",
                    help="with --server: read the outcome through DisPatch's API (default) or a /rows view")
    ap.add_argument("--base", help="--server's HTTP base (default: the ws URL as http)")
    ap.add_argument("--pin-file", help="a file holding the DisPatch PIN (or DISPATCH_PIN; DISPATCH_COOKIE "
                                       "= an lc_session value instead)")
    ap.add_argument("--bot", help="--server: create a thread with this bot for every run")
    ap.add_argument("--thread", help="thread id on a real server (rows that existed before a run are ignored)")
    ap.add_argument("--ref-dispatch", action="store_true",
                    help="the reference server in its DisPatch-shaped mode, judged like DisPatch")
    ap.add_argument("--start-dispatch", type=Path, metavar="BACKEND",
                    help="start DisPatch from this backend checkout (scratch data, stub LLM, PIN) and judge it")
    ap.add_argument("--kokoro", type=Path, help="--start-dispatch: also overlay this Kokoro plugin checkout "
                                               "(the planned live unit, SPEC §11 step 2)")
    ap.add_argument("--online", action="store_true", help="--start-dispatch: let uv reach the network")
    ap.add_argument("--power-hold", default="off", choices=["off", "performance"],
                    help="--start-dispatch: the scratch server's rt.power_hold (off: a network run must "
                         "not hold the box's performance profile)")
    ap.add_argument("--no-after", action="store_true",
                    help="the client sends no `after` anchor on a reconnect (drive.js does: the user row "
                         "the cut reply answers); the server must resume from `after_partial` alone")
    ap.add_argument("--bot-kind", default="api", choices=["api", "advisor"],
                    help="--start-dispatch: the scratch bot. api: the reply is stored, then streamed "
                         "(simulated, ~2.5 s); advisor: streamed as the stub LLM writes it (8 words/s), "
                         "so a rejoin mid-reply meets an OPEN stream (the late-joiner case)")
    ap.add_argument("--llm-port", type=int, default=0,
                    help="--start-dispatch: the stub LLM's loopback port (0 = any free port)")
    ap.add_argument("--fixed-reply", metavar="TEXT|@FILE",
                    help="--start-dispatch: the stub LLM answers with this reply (after its "
                         "\"Answer N.\"), e.g. one whose first sentence is clause-cut while it "
                         "streams — the resume must still skip exactly what was heard "
                         "(review-T3s4 #1); `split` = SPLIT_REPLY")
    ap.add_argument("--server-port", type=int, default=DEFAULT_PORTS[0])
    ap.add_argument("--proxy-port", type=int, default=DEFAULT_PORTS[1])
    ap.add_argument("--reply", default="long", choices=["short", "long"])
    ap.add_argument("--stt", default="fake", choices=["fake", "parakeet"])
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--no-open-mark", action="store_true",
                    help="reference server without Amendment T6-1 (shows the half-sentence row)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / ".e2e-out" / "lowbw")
    a = ap.parse_args(argv)
    bad = [p for p in a.profiles if p not in shaper.PROFILES]
    if bad:
        ap.error(f"unknown profiles {bad}; known: {shaper.ORDER}")
    if a.server and a.judge == "dispatch" and not (a.bot or a.thread):
        ap.error("--server with the dispatch judge needs --bot or --thread")
    a.codecs = [c.strip() for c in a.codecs.split(",") if c.strip()]
    a.compression = [c.strip() for c in a.compression.split(",") if c.strip()]
    return a


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
