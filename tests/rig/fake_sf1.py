"""A fake SpeechForge rig speaking sf.1: an in-process asyncio websocket on
127.0.0.1 (ports 19100-19109). No model, no GPU, no network beyond loopback.

It validates every box->rig frame against the golden ``client_frames`` table
(``fixtures/sf1/protocol.json``, transcribed from SpeechForge), answers like the
real session does (``ready`` from the golden frame, ``pong``, ``stt.final`` for a
flush, TTS frames per flush, ``cancelled``), and can be told to misbehave:
refuse the upgrade, stop answering pings, go silent, stall TTS, drop finals, die.

**Credit, as SpeechForge does it** (``tts_ctx.py``, unchanged in sf.1): before each
LM step the rig checks ``produced < assumed_played + ahead_ms``, where ``produced``
counts 40 ms tokens (a finished unit counts its 3 lookahead tokens too) and
``assumed_played`` is the LAST ``credit.played_ms`` received, frozen until the next
one (real time since the first frame, minus pauses, when none came). A unit's first
audio waits for a whole first chunk of tokens. ``tts.pause`` stops it.
``limits.max_text`` is per context (``too_long``), and a second open context waits
for the first one's ``tts.done``.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import struct
import time
from pathlib import Path

import numpy as np

GOLDEN = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "sf1" / "protocol.json").read_text())
C = GOLDEN["constants"]
HEADER = struct.Struct(C["HEADER"])
PORTS = range(19100, 19110)
TOKEN = "test-token-not-real"  # scrub-ok: test fixture
_TYPES = {"str": str, "int": int, "float": float}


def check_client_frame(msg) -> str | None:
    """The sf.1 rule the real server applies (protocol.check_control). None = valid."""
    if not isinstance(msg, dict):
        return "not an object"
    kind = msg.get("type")
    spec = GOLDEN["client_frames"].get(kind)
    if spec is None:
        return f"unknown type {kind!r}"
    for f, names in spec.items():
        v = msg.get(f)
        if v is None or isinstance(v, bool) or not isinstance(v, tuple(_TYPES[n] for n in names)):
            return f"{kind}.{f} missing or wrong type"
    for f in ("ctx", "gen"):
        v = msg.get(f)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= C["U16"]):
            return f"{kind}.{f} out of range"
    if kind == "cancel" and msg.get("gen") is None and msg.get("ctx") is None:
        return "cancel needs gen or ctx"
    return None


def tts_frame(ctx: int, gen: int, seq: int, pts: int, value: int, n: int = C["TTS_FRAME_SAMPLES"]) -> bytes:
    pcm = np.full(n, value, "<i2")
    return HEADER.pack(C["MAGIC"], C["KIND_TTS"], ctx, gen, seq, pts, n) + pcm.tobytes()


TOKEN_MS = 40
LOOKAHEAD = 3
DEFAULT_FIRST_CHUNK = 10       # the rig's own first chunk (later units of a ctx)


class _Ctx:
    def __init__(self, msg: dict, ahead_ms: int = 3000) -> None:
        self.ctx = msg["ctx"]
        self.gen = msg["gen"]
        self.voice = msg["voice"]
        self.first_chunk = msg.get("first_chunk_tokens") or DEFAULT_FIRST_CHUNK
        self.text = ""
        self.text_chars = 0
        self.samples = 0
        self.seq = 0
        self.started = False
        self.cancelled = False
        self.closing = False
        self.done = False
        self.units = 0
        self.task: asyncio.Task | None = None
        self.gate_open: asyncio.Event = asyncio.Event()   # set: may start (not queued)
        # credit (SpeechForge tts_ctx)
        self.ahead_ms = ahead_ms
        self.played_ms: float | None = None
        self.tokens_total = 0
        self.first_frame_at: float | None = None
        self.paused_since: float | None = None
        self.paused_total = 0.0
        self.stalls = 0
        self.stalled = False

    def assumed_played_ms(self) -> float:
        if self.played_ms is not None:
            return self.played_ms
        if self.first_frame_at is None:
            return 0.0
        now = time.monotonic()
        paused = self.paused_total + (now - self.paused_since if self.paused_since is not None else 0.0)
        return max(0.0, (now - self.first_frame_at - paused) * 1000)

    def may_run(self) -> bool:
        if self.paused_since is not None:
            return False
        return self.tokens_total * TOKEN_MS < self.assumed_played_ms() + self.ahead_ms


class FakeRig:
    def __init__(self, *, token: str = TOKEN) -> None:
        self.token = token
        self.port: int | None = None
        self.server = None
        self.conns: list = []
        self.ws = None                      # the current connection
        # knobs
        self.refuse: tuple[int, dict] | None = None   # (status, headers) before the upgrade
        self.ready_delay = 0.0
        self.ready_error: dict | None = None          # an error frame instead of ready
        self.answer_pings = True
        self.silent = False                 # send nothing at all
        self.final_text = "what is the weather"
        self.drop_finals = False
        self.final_delay = 0.0
        self.tts_delay = 0.0                # before a sentence's first frame
        self.tts_delays: list[float] = []   # per flush, consumed in order (then tts_delay)
        self.tts_frames = 3                 # frames per sentence
        self.tts_value = 1000               # PCM value of the rig's audio (CPU fakes use another)
        self.tts_stall = False              # never render
        self.voices: set[str] | None = None  # None = every voice is known
        self.credit_model = True            # gate rendering like SpeechForge (see the docstring)
        self.lease_state = "held"           # ready.lease.state ("unleased": loaded without a lease)
        self.loading_first = False          # state{models:"loading"} before ready (a cold rig)
        self.max_text = GOLDEN["server_frames"]["ready"]["limits"]["max_text"]
        self.tts_errors: list[str] = []     # per flush: an error code instead of audio (consumed)
        self.ticker: asyncio.Task | None = None
        # records
        self.received: list[dict] = []      # every control frame, in order
        self.errors: list[str] = []         # box->rig frames that broke sf.1
        self.headers: list[dict] = []       # upgrade request headers per attempt
        self.mic: list[tuple[int, int]] = []  # (pts, n) per mic frame
        self.ctxs: dict[int, _Ctx] = {}
        self.sessions = 0
        self.resumed = 0

    # ------------------------------------------------------------ lifecycle
    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/v1/realtime"

    async def start(self) -> FakeRig:
        from websockets.asyncio.server import serve
        last = None
        for port in PORTS:
            try:
                self.server = await serve(self._handler, "127.0.0.1", port, subprotocols=["sf.1"],
                                          process_request=self._process_request, max_size=65536,
                                          compression=None, ping_interval=None)
                self.port = port
                return self
            except OSError as e:
                last = e
        raise RuntimeError(f"no free port in 19100-19109: {last}")

    async def stop(self) -> None:
        if self.ticker is not None:
            self.ticker.cancel()
        if self.server is not None:
            self.server.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.server.wait_closed(), 2)

    async def kill(self) -> None:
        """The rig dies: every socket is cut (no close handshake from our side)."""
        for ws in list(self.conns):
            with contextlib.suppress(Exception):
                ws.transport.abort()
        self.conns.clear()

    def chatter(self, every_s: float = 0.1) -> None:
        """Keep sending harmless frames (stt.vad) so only the pong rule can fire."""
        async def run():
            while True:
                await asyncio.sleep(every_s)
                if self.ws is not None and not self.silent:
                    with contextlib.suppress(Exception):
                        await self.ws.send(json.dumps({"type": "stt.vad", "speech": False, "at": 0}))
        self.ticker = asyncio.ensure_future(run())

    async def send(self, frame) -> None:
        if self.ws is not None:
            await self.ws.send(frame if isinstance(frame, (bytes, str)) else json.dumps(frame))

    def of(self, kind: str) -> list[dict]:
        return [m for m in self.received if m.get("type") == kind]

    def mic_samples(self) -> int:
        return sum(n for _, n in self.mic)

    # ------------------------------------------------------------ server
    async def _process_request(self, connection, request):
        h = {k: v for k, v in request.headers.raw_items()}
        self.headers.append(h)
        if request.headers.get("Origin"):
            return connection.respond(403, '{"error":{"code":"forbidden"}}')
        if self.refuse is not None:
            status, extra = self.refuse
            resp = connection.respond(status, '{"error":{"code":"refused"}}')
            for k, v in extra.items():
                resp.headers[k] = v
            return resp
        if request.headers.get("Authorization") != f"Bearer {self.token}":
            return connection.respond(401, '{"error":{"code":"invalid_token"}}')
        if "sf.1" not in (request.headers.get("Sec-WebSocket-Protocol") or ""):
            return connection.respond(400, '{"error":{"code":"invalid_request"}}')
        return None

    async def _out(self, ws, frame) -> None:
        if self.silent:
            return
        with contextlib.suppress(Exception):
            await ws.send(frame if isinstance(frame, (bytes, str)) else json.dumps(frame))

    async def _handler(self, ws) -> None:
        self.conns.append(ws)
        self.ws = ws
        try:
            first = json.loads(await ws.recv())
            self.received.append(first)
            err = check_client_frame(first)
            if err or first.get("type") != "hello" or first.get("proto") != "sf.1":
                self.errors.append(err or "first frame is not hello{sf.1}")
                return
            if self.loading_first:
                await self._out(ws, {"type": "state", "lease": self.lease_state, "models": "loading"})
            if self.ready_delay:
                await asyncio.sleep(self.ready_delay)
            if self.ready_error is not None:
                await self._out(ws, self.ready_error)
                await ws.close(1013)
                return
            ready = json.loads(json.dumps(GOLDEN["server_frames"]["ready"]))
            ready["limits"]["max_text"] = self.max_text
            ready["lease"]["state"] = self.lease_state
            res = first.get("resume")
            if isinstance(res, dict) and res.get("session"):
                self.resumed += 1
                ready["session"] = res["session"]
            else:
                self.sessions += 1
                ready["session"] = f"sess-{self.sessions}"
            await self._out(ws, ready)
            async for msg in ws:
                if isinstance(msg, bytes):
                    self._on_mic(msg)
                    continue
                m = json.loads(msg)
                self.received.append(m)
                err = check_client_frame(m)
                if err:
                    self.errors.append(err)
                    await self._out(ws, {"type": "error", "code": "bad_frame", "fatal": False, "message": err})
                    continue
                await self._on_control(ws, m)
        except Exception:  # noqa: BLE001 - a client that went away
            pass
        finally:
            for c in self.ctxs.values():
                if c.task is not None:
                    c.task.cancel()
            with contextlib.suppress(ValueError):
                self.conns.remove(ws)

    def _on_mic(self, buf: bytes) -> None:
        magic, kind, ctx, gen, seq, pts, n = HEADER.unpack_from(buf)
        if magic != C["MAGIC"] or kind != C["KIND_MIC"] or len(buf) != C["HEADER_SIZE"] + 2 * n or ctx or gen:
            self.errors.append("bad mic frame")
            return
        self.mic.append((pts, n))

    async def _on_control(self, ws, m: dict) -> None:
        kind = m["type"]
        if kind == "ping":
            if self.answer_pings:
                pong = dict(GOLDEN["server_frames"]["pong"], t=m["t"], srv_t=time.time())
                await self._out(ws, pong)
        elif kind == "stt.flush":
            if not self.drop_finals:
                async def answer():
                    if self.final_delay:
                        await asyncio.sleep(self.final_delay)
                    end = max((p + n for p, n in self.mic), default=0)
                    await self._out(ws, {"type": "stt.final", "id": m.get("id"), "text": self.final_text,
                                         "words": [], "upto": end, "ms": 20})
                asyncio.ensure_future(answer())
        elif kind == "tts.open":
            if self.voices is not None and m["voice"] not in self.voices:
                await self._out(ws, {"type": "error", "code": "unknown_voice", "fatal": False, "ctx": m["ctx"],
                                     "message": "voice.put first"})
                return
            c = _Ctx(m)
            if not any(not o.done and not o.cancelled for o in self.ctxs.values()):
                c.gate_open.set()
            self.ctxs[m["ctx"]] = c
        elif kind == "tts.text":
            c = self.ctxs.get(m["ctx"])
            if c is not None:
                if c.text_chars + len(m["text"]) > self.max_text:
                    await self._out(ws, {"type": "error", "code": "too_long", "fatal": False, "ctx": c.ctx,
                                         "message": f"over {self.max_text} chars in this ctx"})
                    return
                c.text_chars += len(m["text"])
                c.text += m["text"]
        elif kind == "credit":
            c = self.ctxs.get(m["ctx"])
            if c is not None:
                c.ahead_ms = int(m["ahead_ms"])
                if isinstance(m.get("played_ms"), (int, float)):
                    c.played_ms = max(0.0, float(m["played_ms"]))
        elif kind == "tts.pause":
            c = self.ctxs.get(m["ctx"])
            if c is not None and c.paused_since is None:
                c.paused_since = time.monotonic()
        elif kind == "tts.resume":
            c = self.ctxs.get(m["ctx"])
            if c is not None and c.paused_since is not None:
                c.paused_total += time.monotonic() - c.paused_since
                c.paused_since = None
        elif kind == "tts.flush":
            c = self.ctxs.get(m["ctx"])
            if c is not None and not c.cancelled:
                delay = self.tts_delays.pop(0) if self.tts_delays else self.tts_delay
                err = self.tts_errors.pop(0) if self.tts_errors else None
                prev = c.task
                c.task = asyncio.ensure_future(self._render(ws, c, m["id"], delay, prev, err))
        elif kind == "tts.close":
            c = self.ctxs.get(m["ctx"])
            if c is not None:
                c.closing = True

                async def done(c=c):
                    await c.gate_open.wait()
                    if c.task is not None:
                        with contextlib.suppress(Exception):
                            await c.task
                    if not c.cancelled:
                        await self._out(ws, {"type": "tts.done", "ctx": c.ctx, "audio_ms": c.samples // 24,
                                             "rtf": 0.1, "gpu_ms": 1})
                    c.done = True
                    self.ctxs.pop(c.ctx, None)
                    self._start_next()
                asyncio.ensure_future(done())
        elif kind == "cancel":
            gen, ctx = m.get("gen"), m.get("ctx")
            hit = [c for c in self.ctxs.values() if (ctx is not None and c.ctx == ctx) or (gen is not None and c.gen <= gen)]
            for c in hit:
                c.cancelled = True
                c.gate_open.set()
                if c.task is not None:
                    c.task.cancel()
                self.ctxs.pop(c.ctx, None)
            self._start_next()
            out = {"type": "cancelled", "stop_ms": 1.0, "ctxs": sorted(c.ctx for c in hit)}
            if gen is not None:
                out["gen"] = gen
            if ctx is not None:
                out["ctx"] = ctx
            await self._out(ws, out)
        elif kind == "bye":
            await ws.close(1000)

    def _start_next(self) -> None:
        """A context ended: the next queued one may render (one active per session)."""
        if any(o.gate_open.is_set() and not o.done and not o.cancelled for o in self.ctxs.values()):
            return
        for o in self.ctxs.values():
            if not o.done and not o.cancelled:
                o.gate_open.set()
                return

    async def _gate(self, c: _Ctx) -> bool:
        """SpeechForge's LM-step gate (5 ms polls). False: cancelled."""
        while self.credit_model and not c.may_run():
            if not c.stalled and c.paused_since is None:
                c.stalls += 1
            c.stalled = True
            await asyncio.sleep(0.005)
            if c.cancelled:
                return False
        c.stalled = False
        return not c.cancelled

    async def _frame(self, ws, c: _Ctx, delay: float) -> None:
        if c.first_frame_at is None:
            c.first_frame_at = time.monotonic()
        if not c.started:
            c.started = True
            await self._out(ws, {"type": "tts.started", "ctx": c.ctx, "ttfa_ms": round(delay * 1000) + 5,
                                 "turn_id": None})
        await self._out(ws, tts_frame(c.ctx, c.gen, c.seq, c.samples, self.tts_value))
        c.seq += 1
        c.samples += C["TTS_FRAME_SAMPLES"]

    async def _render(self, ws, c: _Ctx, fid, delay: float, prev, err: str | None = None) -> None:
        if prev is not None:
            with contextlib.suppress(Exception):
                await prev
        await c.gate_open.wait()
        if self.tts_stall:
            await asyncio.sleep(3600)
        if delay:
            await asyncio.sleep(delay)
        if c.cancelled:
            return
        if err is not None:
            c.text = ""
            await self._out(ws, {"type": "error", "code": err, "fatal": False, "ctx": c.ctx, "message": err})
            return
        text, c.text = c.text, ""
        # A unit of n frames: n - 3 LM tokens (each gated), then 3 lookahead frames.
        n_frames = self.tts_frames
        n_tok = max(1, n_frames - LOOKAHEAD)
        first_chunk = c.first_chunk if c.units == 0 else DEFAULT_FIRST_CHUNK
        c.units += 1
        dur = n_frames * 40
        start_ms = c.samples // 24
        base = c.tokens_total
        pending = 0
        toks = text.split()
        for count in range(n_tok):
            c.tokens_total = base + count
            if not await self._gate(c):
                return
            pending += 1
            if count + 1 == n_tok and toks:       # on_lm_done: the words go before the rest
                step = dur / len(toks)
                words = [[t.strip(".,!?"), round(start_ms + i * step), round(start_ms + (i + 1) * step)]
                         for i, t in enumerate(toks)]
                await self._out(ws, {"type": "tts.words", "ctx": c.ctx, "words": words, "unit_ms": dur,
                                     "estimated": True})
            if count + 1 >= min(first_chunk, n_tok):  # a unit's first audio needs a whole chunk
                for _ in range(pending):
                    if c.cancelled:
                        return
                    await self._frame(ws, c, delay)
                pending = 0
            await asyncio.sleep(0)
        for _ in range(pending + (n_frames - n_tok)):
            if c.cancelled:
                return
            await self._frame(ws, c, delay)
        c.tokens_total = base + n_frames
        await self._out(ws, {"type": "tts.flushed", "ctx": c.ctx, "id": fid, "audio_ms": c.samples // 24})
