"""A reference v3 voice server for the low-bandwidth harness (owner: T6; a TEST DOUBLE).

It speaks the v3 wire (contracts §C) using only the production transport modules —
:class:`~dispatch_voice.transport.Transport`, ``OffsetTracker``, ``CommitCursor``,
``LinkState`` — around deliberately simple stand-ins for everything else:

* turn detection: 20 ms energy frames (speech starts after 0.2 s voiced, ends after
  0.8 s unvoiced); the detector's start offset is ``mark_open``-ed in the cursor;
* STT: a fake that reports the utterance's voiced and total seconds (``--stt parakeet``
  uses the real model when ``onnx_asr`` and the models are present);
* the agent turn: a thread-level reply that outlives the socket (1.2 s to the first
  sentence; ``--reply long`` = 10 sentences, one every 1.5 s, so the stream is still
  open when a client rejoins mid-reply; ``short`` = 3, one every 0.35 s), spoken
  from the reply's full sentence list, so a session that joins mid-stream speaks it
  once (the late-joiner rule, C.5 #11) minus ``start.after_partial``; a FINISHED reply is
  resumed only when ``start.after`` names something before it, as in the real session;
* TTS: a 24 kHz tone, 0.32 s per word, produced at 0.6x realtime in 0.5 s chunks.

``--dispatch-api --pin PIN`` makes it look like DisPatch from the outside, so the
harness's DisPatch judge (``e2e_lowbw.py --judge dispatch``) can be proven end to end
before T3 slice 4: ``POST /api/auth/unlock`` issues the ``lc_session`` cookie, the
socket refuses a missing cookie or a foreign Origin with 1008 BEFORE accept, threads
are created with ``POST /api/threads`` and read back with
``GET /api/threads/{id}/messages`` (user + assistant rows, cookie required), and the
``say`` text is the normalised speakable text (``text.speech_text.normalise``) as in
the real session. ``--stt oracle`` answers the clip's known words, minus the head in
proportion to the voiced audio that was lost (what a cut-off utterance would read).

The commit cursor is keyed by the device and the client's stream id; a socket
without a valid ``s`` (v2, or a malformed id) gets NO cursor at all (review-T6 M2:
a shared fallback key would let one client's submit swallow another's utterance).

It exists so ``tests/e2e_lowbw.py`` can prove the transport end to end over the
shaper before T3's session (slice 4) lands, and as a reading aid for T3: every
transport call is where the real session makes it. ``e2e_lowbw.py --server URL``
runs the same client against a real DisPatch scratch instance.

    python tests/transport/refserver.py --port 19055 --state DIR [--reply long]
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import secrets
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from dispatch_voice.text.speech_text import normalise
from dispatch_voice.transport import (
    CommitCursor,
    GapTooLarge,
    Transport,
    wire3,
)

log = logging.getLogger("refserver")

SR_IN = 16000
SR_OUT = 24000
FRAME = 320
VOICED_DB = -40.0
START_FRAMES = 10            # 0.2 s voiced -> speech start
END_FRAMES = 40              # 0.8 s unvoiced -> end of turn
PRE_ROLL = 10                # frames kept before the start
TTS_RTF = 0.6
TTS_CHUNK_S = 0.5
WORD_S = 0.32
LLM_FIRST_S = 1.2
SENTENCE_EVERY_S = {"short": 0.35, "long": 1.5}    # long: the stream is still open mid-reply
MAX_QUEUE_BYTES = 64_000

REPLIES = {
    "short": ["Sure.", "Your afternoon is clear until four.", "Then you have the dentist."],
    "long": [f"Point {i}: " + s for i, s in enumerate([
        "the plumber comes at two and needs the side gate open",
        "the dentist moved your cleaning to Thursday at nine",
        "your sister asked whether Sunday lunch is still on",
        "the car is due for its inspection next week",
        "the library books are due back on Friday",
        "the electrician sent the revised quote this morning",
        "the school pickup changes to three fifteen tomorrow",
        "there is a parcel waiting at the post office",
        "the gym class you booked starts at six",
        "rain is expected in the evening so take a coat",
    ], 1)],
}


@dataclass
class Reply:
    message_id: str
    sentences: list[str] = field(default_factory=list)
    done: bool = False
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    def bump(self) -> None:
        self.changed.set()
        self.changed = asyncio.Event()


@dataclass
class Thread:
    id: str
    rows: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)       # the DisPatch-shaped rows (--dispatch-api)
    replies: dict[str, Reply] = field(default_factory=dict)
    latest: Reply | None = None
    new_reply: asyncio.Event = field(default_factory=asyncio.Event)


class World:
    def __init__(self, state_dir: Path, reply: str, stt: str, open_mark: bool = True, *,
                 dispatch_api: bool = False, pin: str | None = None, oracle_text: str = "",
                 oracle_voiced_s: float = 0.0) -> None:
        self.cursor = CommitCursor(state_dir)
        self.open_mark = open_mark          # False: the pre-Amendment-T6-1 behaviour (for comparison)
        self.threads: dict[str, Thread] = {}
        self.slots: dict[str, RefSession] = {}
        self.reply = reply
        self.stt = None
        self.stt_mode = stt
        self.oracle_words = oracle_text.split()
        self.oracle_voiced_s = oracle_voiced_s
        self.dispatch_api = dispatch_api
        self.pin = pin
        self.sessions: set[str] = set()     # lc_session tokens issued by /api/auth/unlock
        self.log: list[dict] = []
        if stt == "parakeet":
            import onnx_asr
            models = Path.home() / ".local/share/dispatch-voice-models/parakeet-tdt-0.6b-v2"
            self.stt = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", str(models), quantization="int8")

    def thread(self, tid: str) -> Thread:
        return self.threads.setdefault(tid, Thread(tid))

    async def run_reply(self, th: Thread, rep: Reply) -> None:
        row = None
        if self.dispatch_api:
            row = {"id": rep.message_id, "thread_id": th.id, "role": "assistant", "content": "",
                   "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "media_url": None, "metadata": None}
            th.messages.append(row)
        await asyncio.sleep(LLM_FIRST_S)
        for s in REPLIES[self.reply]:
            rep.sentences.append(s)
            if row is not None:
                row["content"] = " ".join(rep.sentences)
            rep.bump()
            await asyncio.sleep(SENTENCE_EVERY_S[self.reply])
        rep.done = True
        rep.bump()

    def oracle(self, voiced_s: float) -> str:
        """The clip's words, minus the head in proportion to the voiced audio lost."""
        n = len(self.oracle_words)
        if not n or self.oracle_voiced_s <= 0:
            return ""
        keep = max(0, min(n, round(n * voiced_s / self.oracle_voiced_s)))
        return " ".join(self.oracle_words[n - keep:])


WORLD: World | None = None


class Detector:
    """Energy turn detector over the continuous (offset-corrected) uplink."""

    def __init__(self) -> None:
        self.reset(0)

    def reset(self, pos: int) -> None:
        self.pos = pos                  # stream offset of the next sample
        self.buf = np.zeros(0, np.int16)
        self.voiced_run = 0
        self.silent_run = 0
        self.in_speech = False
        self.start = 0
        self.last_voiced_end = 0
        self.frames: list[np.ndarray] = []
        self.history: list[np.ndarray] = []

    def feed(self, pcm: np.ndarray) -> list[tuple]:
        ev = []
        self.buf = np.concatenate([self.buf, pcm]) if len(self.buf) else pcm
        while len(self.buf) >= FRAME:
            f, self.buf = self.buf[:FRAME], self.buf[FRAME:]
            rms = np.sqrt(np.mean((f.astype(np.float64) / 32768) ** 2)) + 1e-9
            voiced = 20 * np.log10(rms) > VOICED_DB
            off = self.pos
            self.pos = (self.pos + FRAME) & wire3.U32
            if not self.in_speech:
                self.history = (self.history + [f])[-PRE_ROLL - START_FRAMES:]
                self.voiced_run = self.voiced_run + 1 if voiced else 0
                if self.voiced_run >= START_FRAMES:
                    self.in_speech = True
                    self.frames = list(self.history)
                    self.start = (off - (len(self.history) - 1) * FRAME) & wire3.U32
                    self.silent_run = 0
                    self.last_voiced_end = self.pos
                    ev.append(("start", self.start))
            else:
                self.frames.append(f)
                if voiced:
                    self.silent_run = 0
                    self.last_voiced_end = self.pos
                else:
                    self.silent_run += 1
                    if self.silent_run >= END_FRAMES:
                        audio = np.concatenate(self.frames)
                        ev.append(("end", self.start, self.last_voiced_end, audio))
                        self.in_speech = False
                        self.voiced_run = 0
                        self.frames = []
                        self.history = []
        return ev


class RefSession:
    def __init__(self, ws: WebSocket, th: Thread, tr: Transport, key: tuple[str, str] | None) -> None:
        self.ws = ws
        self.th = th
        self.tr = tr
        self.key = key
        self.sid = secrets.token_hex(8)
        self.gen = 0
        self.seq = 0
        self.det = Detector()
        self.anchored = False
        self.speaker: asyncio.Task | None = None
        self.speak_q: asyncio.Queue[tuple[Reply, int]] = asyncio.Queue()
        self.closed = asyncio.Event()
        self.close_code = 1000
        self.after_partial: dict | None = None
        self.stats = {"bad_frames": 0, "too_big": 0, "replay_msgs": 0, "gap_resets": 0, "submits": 0,
                      "skipped_committed": 0, "say": 0, "late_audio_dropped": 0}

    async def send(self, obj: dict) -> None:
        with contextlib.suppress(Exception):
            await self.tr.send(obj)

    # ------------------------------------------------------------- speaking #
    async def speak_loop(self) -> None:
        while True:
            rep, start = await self.speak_q.get()
            i = start
            while True:
                if i < len(rep.sentences):
                    await self.speak(rep, rep.sentences[i])
                    i += 1
                    continue
                if rep.done:
                    await self.send({"type": "say_end", "gen": self.gen, "message_id": rep.message_id,
                                     "last_seq": self.seq})
                    break
                await rep.changed.wait()

    async def speak(self, rep: Reply, text: str) -> None:
        self.tr.at_sentence_boundary()
        self.seq += 1
        seq, gen = self.seq, self.gen
        if WORLD.dispatch_api:                       # `say.text` = exactly the speakable text (C.4)
            text = normalise(text, tags_ok=False)
        await self.tr.send_say({"gen": gen, "seq": seq, "text": text, "message_id": rep.message_id,
                                "kind": "reply", "role": "character"})
        self.stats["say"] += 1
        if self.tr.link.text_mode:
            return
        total = int(len(text.split()) * WORD_S * SR_OUT)
        t = np.arange(total) / SR_OUT
        pcm = (0.25 * np.sin(2 * np.pi * (180 + 7 * (seq % 9)) * t) * 32767).astype(np.int16)
        step = int(TTS_CHUNK_S * SR_OUT)
        for n, i in enumerate(range(0, total, step)):
            await asyncio.sleep(min(step, total - i) / SR_OUT * TTS_RTF)
            await self.tr.send_audio(kind="reply", gen=gen, seq=seq, pcm16=pcm[i:i + step],
                                     t_ms=self.tr.now_ms(), chunk=n)
        await self.tr.end_sentence(kind="reply", gen=gen, seq=seq)

    def queue_reply(self, rep: Reply, start: int = 0) -> None:
        self.speak_q.put_nowait((rep, start))

    # -------------------------------------------------------------- hearing #
    async def on_audio(self, buf: bytes) -> None:
        if Transport.too_big(buf):
            self.stats["too_big"] += 1
            await self.send({"type": "error", "code": "bad_frame", "message": "Audio message too large."})
            return
        try:
            got = self.tr.decode_up(buf)
        except ValueError:
            self.stats["bad_frames"] += 1
            await self.send({"type": "error", "code": "bad_frame", "message": "That audio wasn't understood."})
            return
        if got is None:              # v2 raw PCM: no offsets
            off, flags, pcm = self.det.pos, 0, np.frombuffer(buf, "<i2").astype(np.int16)
        else:
            off, flags, pcm = got
        if flags & wire3.FLAG_REPLAY:
            self.stats["replay_msgs"] += 1
        if not self.anchored:
            self.anchored = True
            self.det.reset(off)
        try:
            parts = self.tr.offsets.feed(off, pcm)
        except GapTooLarge as e:
            self.stats["gap_resets"] += 1
            if self.det.in_speech and self.key is not None:
                WORLD.cursor.clear_open(self.key)
            self.det.reset((off) & wire3.U32)
            parts = e.frames
        for p in parts:
            for ev in self.det.feed(p):
                await self.on_event(ev)

    async def on_event(self, ev: tuple) -> None:
        if ev[0] == "start":
            if WORLD.open_mark and self.key is not None:
                WORLD.cursor.mark_open(self.key, ev[1])
            await self.send({"type": "state", "state": "hearing"})
            return
        _, start, end, audio = ev
        committed = WORLD.cursor.get(self.key) if self.key is not None else None
        if committed is not None and wire3.u32_diff(end, committed) <= 0:
            self.stats["skipped_committed"] += 1
            return
        voiced = self._voiced_s(audio)
        if WORLD.stt is not None:
            text = await asyncio.to_thread(WORLD.stt.recognize, audio.astype(np.float32) / 32768,
                                           sample_rate=SR_IN)
        elif WORLD.stt_mode == "oracle":
            text = WORLD.oracle(voiced)
        else:
            text = f"utterance of {voiced:.2f} s voiced"
        mid = "u-" + secrets.token_hex(6)
        row = {"id": mid, "text": text, "start": start, "upto": end, "voiced_s": round(voiced, 3),
               "dur_s": round(len(audio) / SR_IN, 3), "sid": self.sid, "t": time.time()}
        self.th.rows.append(row)
        if WORLD.dispatch_api:
            self.th.messages.append({"id": mid, "thread_id": self.th.id, "role": "user", "content": text,
                                     "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "media_url": None,
                                     "metadata": {"voice": True}})
        if self.key is not None:
            WORLD.cursor.set(self.key, end, meta={"text": text, "message_id": mid})
        self.stats["submits"] += 1
        await self.send({"type": "eot", "reason": "silence", "upto": end})
        await self.send({"type": "transcript", "text": text, "merged": False, "message_id": mid, "upto": end})
        rep = Reply("r-" + secrets.token_hex(6))
        rep.user_id = mid  # type: ignore[attr-defined]
        self.th.replies[rep.message_id] = rep
        self.th.latest = rep
        asyncio.create_task(WORLD.run_reply(self.th, rep))
        self.queue_reply(rep, 0)

    @staticmethod
    def _voiced_s(audio: np.ndarray) -> float:
        n = len(audio) // FRAME
        f = audio[:n * FRAME].astype(np.float64).reshape(n, FRAME) / 32768
        db = 20 * np.log10(np.sqrt((f ** 2).mean(1)) + 1e-9)
        return float((db > VOICED_DB).sum() * FRAME / SR_IN)

    def _after_covers(self, rep: Reply, msg: dict) -> bool:
        """The real session's reading of ``start.after`` for a STORED reply (review-T7b H1):
        ``_resume_after`` speaks only replies stored after the first id of ``[after,
        *after_ids]`` it knows. So a finished reply is resumed only when that id comes
        before it in the thread (its user row, an older reply); naming the reply itself,
        or nothing known, speaks nothing. An open stream is resumed whatever ``after`` says
        (the snapshot path), so this is asked only for a done reply. A client that sends no
        ``after`` at all (``e2e_lowbw.py``'s) keeps the old lenient reading: the reply named
        by ``after_partial`` is resumed."""
        order: list[str] = []
        for r in self.th.replies.values():
            uid = getattr(r, "user_id", None)
            if uid:
                order.append(uid)
            order.append(r.message_id)
        pos = {mid: i for i, mid in enumerate(order)}
        ids = msg.get("after_ids") if isinstance(msg.get("after_ids"), list) else []
        cands = [c for c in [msg.get("after"), *ids] if isinstance(c, str) and c]
        if not cands:
            return False
        for cand in cands:
            if cand in pos:
                return pos[cand] >= pos.get(rep.message_id, -1)
        return True

    # ------------------------------------------------------------- control #
    async def on_control(self, msg: dict) -> None:
        t = msg.get("type")
        if t == "ping":
            await self.send(self.tr.pong(msg.get("t")))          # pong() validates `t`
        elif t == "start":
            self.tr.link.set_slow_link(msg.get("slow_link"))
            cur = msg.get("cursor")
            if self.key is not None and isinstance(cur, dict) and cur.get("stream") == self.key[1] \
                    and isinstance(cur.get("from"), int) and not isinstance(cur.get("from"), bool):
                frm = WORLD.cursor.resume_from(self.key, cur["from"])
                await self.send({"type": "cursor", "from": frm})
                e = WORLD.cursor.entry(self.key)
                if e and e.upto is not None and wire3.u32_diff(e.upto, cur["from"]) > 0 and e.meta:
                    await self.send({"type": "transcript", "text": e.meta.get("text", ""), "merged": False,
                                     "message_id": e.meta.get("message_id"), "upto": e.upto, "replayed": True})
            ap = msg.get("after_partial")
            rep = self.th.latest
            if rep is not None and not getattr(rep, "_resumed_" + self.sid, False):
                setattr(rep, "_resumed_" + self.sid, True)
                if isinstance(ap, dict) and ap.get("message_id") == rep.message_id:
                    n = max(0, int(ap.get("sentences") or 0))
                    named = rep.done and self._after_covers(rep, msg)
                    WORLD.log.append({"sid": self.sid, "thread": self.th.id, "t": time.time(), "resume": {
                        "message_id": rep.message_id, "from_sentence": n, "stream_open": not rep.done,
                        "sentences_so_far": len(rep.sentences), "after_names_it": named}})
                    if named:
                        pass        # the real session thinks the client heard it: nothing is spoken
                    elif n < len(rep.sentences) or not rep.done:
                        self.queue_reply(rep, n)
                    elif rep.done:
                        await self.send({"type": "say_end", "gen": self.gen, "message_id": rep.message_id,
                                         "last_seq": self.seq})
                elif isinstance(cur, dict) and not rep.done:
                    # a reply the client never heard a sentence of (it started during the drop)
                    self.queue_reply(rep, 0)
        elif t == "link":
            self.tr.link.request(msg)
            WORLD.log.append({"sid": self.sid, "thread": self.th.id, "link": msg, "t": time.time()})
        elif t == "probe":
            await self.tr.send_probe(msg.get("bytes", 0))
        elif t == "playback_done":
            pass

    # ----------------------------------------------------------------- run #
    async def run(self) -> None:
        await self.send({"type": "ready", "sample_rate_in": SR_IN, "sample_rate_out": SR_OUT,
                         "sample_format": "pcm_s16le", "streaming_tts": True, "smart_turn": False,
                         "sid": self.sid, "langs": ["en"], **self.tr.ready_fields()})
        await self.send({"type": "state", "state": "listening"})
        self.speaker = asyncio.create_task(self.speak_loop())
        try:
            while True:
                m = await self.ws.receive()
                if m["type"] == "websocket.disconnect":
                    break
                if m.get("bytes") is not None:
                    await self.on_audio(m["bytes"])
                elif m.get("text") is not None:
                    try:
                        msg = json.loads(m["text"])
                    except ValueError:
                        continue
                    if isinstance(msg, dict):
                        await self.on_control(msg)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            self.closed.set()
            if self.speaker:
                # review-T6 H1 (belt): let the speaking task unwind before the codecs go;
                # Transport.close() also waits for in-flight audio work itself
                self.speaker.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self.speaker
            await self.tr.close()
            self.stats["late_audio_dropped"] = self.tr.stats.get("late_audio_dropped", 0)


def _origin_ok(ws: WebSocket) -> bool:
    origin = ws.headers.get("origin")
    if origin is None:
        return True
    netloc = urlparse(origin).netloc
    return bool(netloc) and netloc == ws.headers.get("host", "")


async def voice_ws(ws: WebSocket) -> None:
    tid = ws.path_params["thread_id"]
    # --dispatch-api: the DisPatch gate, pre-accept (SPEC §8.1): Origin == Host, a live
    # unlocked session, a thread that exists; anything else is 1008 before accept
    if WORLD.dispatch_api and (not _origin_ok(ws) or ws.cookies.get("lc_session") not in WORLD.sessions
                               or tid not in WORLD.threads):
        await ws.close(code=1008)
        return
    tr = Transport.negotiate(ws, ws.query_params, rate_out=SR_OUT)
    await ws.accept()
    # the cursor needs BOTH ids; without a valid stream id there is no cursor (M2)
    key = (tr.device or ws.cookies.get("lc_session") or secrets.token_hex(8), tr.stream) \
        if tr.stream is not None else None
    slot = tr.device or ws.cookies.get("lc_session") or ("anon-" + secrets.token_hex(8))
    old = WORLD.slots.get(slot)
    sess = RefSession(ws, WORLD.thread(tid), tr, key)
    WORLD.slots[slot] = sess
    if old is not None:
        with contextlib.suppress(Exception):
            await old.ws.close(4001, "replaced")
    WORLD.log.append({"sid": sess.sid, "thread": tid, "open": True, "proto": tr.proto, "codec_in": tr.codec_in,
                      "codec_out": tr.codec_out, "t": time.time()})
    try:
        await sess.run()
    finally:
        WORLD.log.append({"sid": sess.sid, "thread": tid, "closed": True, "stats": sess.stats, "tx": tr.stats,
                          "t": time.time()})
        if WORLD.slots.get(slot) is sess:
            WORLD.slots.pop(slot, None)


async def rows(request):
    th = WORLD.threads.get(request.path_params["thread_id"])
    return JSONResponse({"rows": th.rows if th else [],
                         "replies": {k: {"sentences": v.sentences, "done": v.done}
                                     for k, v in (th.replies.items() if th else [])},
                         "log": [x for x in WORLD.log if x.get("thread") == request.path_params["thread_id"]]})


async def health(request):
    return JSONResponse({"ok": True, "reply": WORLD.reply, "stt": WORLD.stt_mode,
                         "dispatch_api": WORLD.dispatch_api})


# ------------------------------------------------- DisPatch-shaped API (--dispatch-api) #
def _authed(request: Request) -> bool:
    return request.cookies.get("lc_session") in WORLD.sessions


async def api_unlock(request: Request):
    body = await request.json()
    if not WORLD.pin or not isinstance(body, dict) or str(body.get("pin") or "") != WORLD.pin:
        return JSONResponse({"detail": "Incorrect PIN"}, status_code=401)
    tok = secrets.token_urlsafe(24)
    WORLD.sessions.add(tok)
    resp = JSONResponse({"ok": True, "remembered": False})
    resp.set_cookie("lc_session", tok, httponly=True, samesite="strict", path="/")
    return resp


async def api_create_thread(request: Request):
    if not _authed(request):
        return JSONResponse({"detail": "Unlock for full access"}, status_code=403)
    tid = "t-" + secrets.token_hex(6)
    WORLD.thread(tid)
    return JSONResponse({"id": tid, "bot_id": "refbot"})


async def api_messages(request: Request):
    if not _authed(request):
        return JSONResponse({"detail": "Unlock for full access"}, status_code=403)
    th = WORLD.threads.get(request.path_params["thread_id"])
    if th is None:
        return JSONResponse({"detail": "Thread not found"}, status_code=404)
    return JSONResponse({"thread_id": th.id, "messages": [dict(m) for m in th.messages], "has_more": False})


def make_app(state_dir: Path, reply: str = "long", stt: str = "fake", open_mark: bool = True, *,
             dispatch_api: bool = False, pin: str | None = None, oracle_text: str = "",
             oracle_voiced_s: float = 0.0) -> Starlette:
    global WORLD
    WORLD = World(state_dir, reply, stt, open_mark, dispatch_api=dispatch_api, pin=pin,
                  oracle_text=oracle_text, oracle_voiced_s=oracle_voiced_s)
    routes = [WebSocketRoute("/ws/voice/{thread_id}", voice_ws), Route("/health", health)]
    if dispatch_api:
        routes += [Route("/api/auth/unlock", api_unlock, methods=["POST"]),
                   Route("/api/threads", api_create_thread, methods=["POST"]),
                   Route("/api/threads/{thread_id}/messages", api_messages)]
    else:
        routes.append(Route("/rows/{thread_id}", rows))
    return Starlette(routes=routes)


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--reply", choices=sorted(REPLIES), default="long")
    ap.add_argument("--stt", choices=["fake", "parakeet", "oracle"], default="fake")
    ap.add_argument("--oracle-text", default="", help="--stt oracle: the clip's known words")
    ap.add_argument("--oracle-voiced", type=float, default=0.0, help="--stt oracle: the clip's voiced seconds")
    ap.add_argument("--no-open-mark", action="store_true", help="disable Amendment T6-1 (comparison runs)")
    ap.add_argument("--dispatch-api", action="store_true",
                    help="look like DisPatch: PIN unlock + lc_session cookie gate + threads/messages API")
    ap.add_argument("--pin-file", type=Path, help="--dispatch-api: a file holding the PIN (never on argv)")
    a = ap.parse_args()
    pin = a.pin_file.read_text().strip() if a.pin_file else None
    if a.dispatch_api and not pin:
        ap.error("--dispatch-api needs --pin-file")
    app = make_app(a.state, a.reply, a.stt, open_mark=not a.no_open_mark, dispatch_api=a.dispatch_api,
                   pin=pin, oracle_text=a.oracle_text, oracle_voiced_s=a.oracle_voiced)
    # the live unit's flags (SPEC §11): 256 KB frames, 60 s keepalive timeout
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning", ws_max_size=262144,
                ws_ping_interval=20, ws_ping_timeout=60)


if __name__ == "__main__":
    main()
