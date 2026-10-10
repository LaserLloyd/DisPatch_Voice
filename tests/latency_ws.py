#!/usr/bin/env python3
"""Latency harness, ``ws`` mode on a SCRATCH DisPatch (SPEC §12 "Latency harness", §4.9).

T4's ``tests/latency_harness.py`` times the realtime engines directly; this drives the
REAL session hub end to end: a scratch DisPatch (never the live one) started like the
live unit (``uv run --offline --with-editable <this checkout> uvicorn app.main:app``,
``--ws-max-size 262144``), an advisor bot whose provider AND embeddings are the 1.2 s stub
(``tests/stub_llm.py``), the real models (Silero, Smart Turn, Parakeet, Chatterbox or the
Kokoro plugin), and a v3 client written to C-§C (PCM16 both ways, real-time uplink pacing,
a modelled player that answers `barge_in` with the exact `heard`, sends one every second
while a reply plays, and `playback_done` when it drains).

Scenarios (each on a fresh thread):

* ``turn:<clip>`` — speech end (Silero on the clip, threshold 0.5, the engines-mode rule)
  → `eot`, `transcript`, first `say`, first reply audio; the server's own `metrics`
  (speculation, p_done, hold, first chunk) beside it.
* ``barge`` — a question, then ``interrupt.wav`` ("Wait, stop, that's not what I meant.")
  0.8 s into the reply: speech onset → `duck`, → `barge_in`, → the interrupting words'
  `transcript`; the cut reply's row must carry ``voice_heard.upto`` (abort keep_partial
  → exact `heard` → `note_heard`).
* ``false`` — a question, then ``backchannel.wav`` ("Uh-huh", 0.87 s) over the reply:
  onset → `duck` → `resume {rewind}` within 2.0 s, no `barge_in`, no new user row.
* ``addendum`` — ``addendum.wav`` ("What's on my calendar this afternoon" + 2.0 s + "oh
  and tomorrow"): exactly one user row carrying both parts, `transcript {merged: true}`,
  one reply.

Passes: idle, then under ``tests/load.py`` (``--load-n``, 8). Every row carries the
platform profile and max MHz (the §4.9 column: the scratch server holds ``performance``
while the chat is open unless ``--power-hold off``; ``--hold power-saver`` pins the 2 GHz
car state for the run — use the two together for the power-saver column).

    flock "$VOICE_LOCK" uv run python tests/latency_ws.py \\
        --backend ~/Projects/dispatch-chat/backend --port 19032 --out <dir> [--kokoro <plugin>]

Thresholds (SPEC §12): speech end → first reply audio ≤ 3.8 s cloned / ≤ 2.8 s Kokoro in
power-saver, ≤ 3.0 / ≤ 2.3 s under the hold, idle and under load; onset → `barge_in` ≤ 1.0 s;
false interruption resumed ≤ 2.0 s (rewound when the pause was > 0.8 s); the addendum → one
row + one reply; ≤ 2 speculations per turn. (Onset → duck ≤ 0.3 s is the PHONE's optimistic
duck — the browser e2e's; the server's `duck` is reported for reference.)
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ADDON = HERE.parent
sys.path.insert(0, str(HERE))

from latency_harness import (  # tests/ is on the path
    THRESHOLDS,
    column_of,
    load1,
    max_mhz,
    platform_profile,
    speech_models_dir,
)

SR = 16000
FRAME = 512
UP_MS = 20                                  # uplink pacing: 20 ms of PCM per message
OUT_RATE = 24000                            # downlink PCM16 rate (C.2)
BOT = "adv"


# --------------------------------------------------------------------------- #
# clips and their speech marks (Silero, the engines-mode rule)
# --------------------------------------------------------------------------- #

def load_clip(name: str) -> np.ndarray:
    from fakes import load_clip as _lc
    return _lc(name)


class Marks:
    def __init__(self) -> None:
        from dispatch_voice.host.vad_turn import SileroVAD
        self.vad = SileroVAD(speech_models_dir() / "silero_vad.onnx", threads=1)

    def probs(self, x: np.ndarray) -> list[float]:
        self.vad.reset()
        return [self.vad(x[i:i + FRAME]) for i in range(0, len(x) - FRAME + 1, FRAME)]

    def onset_s(self, x: np.ndarray) -> float:
        for i, p in enumerate(self.probs(x)):
            if p >= 0.5:
                return i * FRAME / SR
        return 0.0

    def end_s(self, x: np.ndarray) -> float:
        last = 0
        for i, p in enumerate(self.probs(x)):
            if p >= 0.5:
                last = i
        return (last + 1) * FRAME / SR


# --------------------------------------------------------------------------- #
# the scratch server
# --------------------------------------------------------------------------- #

def start_scratch(port: int, data: Path, backend: Path, llm_base: str, *, kokoro: Path | None,
                  power_hold: str, speculate: str = "advisor") -> subprocess.Popen:
    import httpx
    shutil.rmtree(data, ignore_errors=True)
    know = data / "knowledge"
    (know / "notes").mkdir(parents=True)
    (know / "_profile.md").write_text("# Profile\nThe owner keeps a family calendar and a garden.\n")
    (know / "notes" / "calendar.md").write_text("The dentist is on Thursdays. The school calls on Fridays.\n")
    prov = {"provider": "openai", "base_url": llm_base, "model": "stub", "api_key": "not-a-key"}
    cfg = {"bots": [
        {"id": BOT, "name": "Adviser", "emoji": "A", "order": 0, "visible": True, "safe": False,
         "advisor": {"owner": "Sam", "providers": [prov], "knowledge_dir": str(know), "agents": ["worker"],
                     "research_agent": "worker", "action_agent": "worker",
                     "embeddings": {"base_url": llm_base, "model": "stub-embed", "api_key": "not-a-key"}}},
        {"id": "apibot", "name": "Api Bot", "emoji": "B", "order": 1, "visible": True, "safe": False, "api": prov},
    ]}
    data.mkdir(parents=True, exist_ok=True)
    (data / "config.yaml").write_text(json.dumps(cfg))
    os.chmod(data / "config.yaml", 0o600)
    voice = {"enabled": True, "rt": {"power_hold": power_hold, "speculate": speculate}}
    (data / "voice.yaml").write_text(json.dumps(voice))
    os.chmod(data / "voice.yaml", 0o600)
    env = {k: v for k, v in os.environ.items() if k not in ("DISPATCH_VOICE_PKG", "DISPATCH_EXTENSIONS", "DISPATCH_VOICE")}
    env.update(DISPATCH_DATA_DIR=str(data), DISPATCH_MIRROR="0", DISPATCH_GATEWAY_WS="0",
               OPENCLAW_BIN="/nonexistent/openclaw", DISPATCH_SPEECH_MODELS=str(speech_models_dir()))
    overlays = ["--with-editable", str(ADDON)]
    if kokoro is not None:
        overlays += ["--with-editable", str(kokoro)]
        env["DISPATCH_VOICE_EXTENSIONS"] = "dispatch_voice_kokoro:setup"
        env["DISPATCH_VOICE_TTS"] = "kokoro"
    uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    argv = [uv, "run", "--quiet", "--offline", "--no-sync", *overlays, "uvicorn", "app.main:app",
            "--host", "127.0.0.1", "--port", str(port), "--no-access-log",
            "--ws-ping-timeout", "60", "--ws-max-size", "262144"]
    with open(data / "server.log", "wb") as log:     # the child keeps its own copy of the fd
        proc = subprocess.Popen(argv, cwd=backend, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"scratch server died; see {data / 'server.log'}")
        with contextlib.suppress(httpx.HTTPError):
            if httpx.get(base + "/api/voice/status", timeout=1).status_code == 200:
                return proc
        time.sleep(0.5)
    proc.terminate()
    raise SystemExit("scratch server did not warm up in 240 s")


def wait_warm(base: str, timeout: float = 240) -> dict:
    import httpx
    t0 = time.monotonic()
    st: dict = {}
    while time.monotonic() - t0 < timeout:
        st = httpx.get(base + "/api/voice/status", timeout=5).json()
        acks = st.get("acks") or {}
        rendered = all((r.get("ready", 0) + r.get("failed", 0)) >= r.get("total", 0)
                       for r in acks.values() if isinstance(r, dict))
        # the start-up prewarm finished (models loaded) AND the acknowledgement lines it
        # queued are rendered: renders share the TTS engine (its lock) and the CPU, so a
        # turn timed during them measures start-up, not a voice chat (V1)
        if st.get("warm") is True and rendered:
            return st
        time.sleep(0.5)
    return st


# --------------------------------------------------------------------------- #
# a v3 client with a modelled player
# --------------------------------------------------------------------------- #

def pack_up(off: int, pcm16: bytes) -> bytes:
    from dispatch_voice.transport import wire3
    return wire3.pack_up(wire3.CODEC_PCM16, 0, off, [pcm16])


@dataclass
class Row:
    scenario: str
    clip: str
    load: str
    profile: str
    max_mhz: int
    load1: str
    column: str = ""
    speech_end_s: float = 0.0
    eot_at: float | None = None
    eot_soft: bool = False
    transcript_at: float | None = None
    transcript: str = ""
    first_say_at: float | None = None
    first_audio_at: float | None = None
    server_eot_to_first_audio_ms: int | None = None
    first_chunk_ms: int | None = None
    p_done: float | None = None
    hold_ms: int | None = None
    speculated: bool | None = None
    speculation_hit: bool | None = None
    speculation: dict | None = None
    power_hold: str | None = None
    duck_at: float | None = None
    barge_in_at: float | None = None
    resume_at: float | None = None
    resume_rewind: bool | None = None
    interrupt_transcript_at: float | None = None
    interrupt_to_submit_ms: int | None = None
    heard_noted: bool | None = None
    ack_played: bool | None = None          # a rendered acknowledgement line played this turn
    ack_eta_ms: int | None = None           # the gate's ETA of first audio when it decided
    ack_after_eot_s: float | None = None    # when the line's `say` went out, after `eot`
    earcon: bool = False                    # the `ack` earcon frame instead
    user_rows: int | None = None
    merged: bool | None = None
    replies: int | None = None
    ok: bool = True
    notes: list[str] = field(default_factory=list)


class Client:
    """One voice chat over ``/ws/voice/<thread>?v=3``: real-time uplink, a player model."""

    def __init__(self, ws, t0: float) -> None:
        self.ws = ws
        self.t0 = t0
        self.off = 0
        self.frames: list[tuple[float, dict]] = []
        self.gen = 0
        self.sid = ""
        self.seq_start: dict[int, float] = {}      # seq -> wall time its playback starts
        self.seq_ms: dict[int, float] = {}         # seq -> ms of audio received
        self.play_end = 0.0                        # wall time the queued audio ends
        self.say_end = False
        self.first_reply_audio: float | None = None
        self.reply_audio_after: float = 0.0
        self._last_heard = 0.0
        self.done_sent = False
        self.timed_out = False

    def now(self) -> float:
        return time.perf_counter()

    async def send_pcm(self, pcm: bytes) -> None:
        await self.ws.send(pack_up(self.off, pcm))
        self.off = (self.off + len(pcm) // 2) & 0xFFFFFFFF

    def first(self, t: str, after: float = 0.0, **match) -> tuple[float, dict] | None:
        for at, f in self.frames:
            if f.get("type") == t and at >= after and all(f.get(k) == v for k, v in match.items()):
                return at, f
        return None

    def last_before(self, t: str, before: float) -> tuple[float, dict] | None:
        """The last frame of type ``t`` received at or before ``before``."""
        found = None
        for at, f in self.frames:
            if at > before:
                break
            if f.get("type") == t:
                found = (at, f)
        return found

    def playing(self) -> tuple[int, int] | None:
        now = self.now()
        for seq in sorted(self.seq_start):
            st = self.seq_start[seq]
            if st <= now < st + self.seq_ms.get(seq, 0) / 1000:
                return seq, int((now - st) * 1000)
        return None

    async def on_message(self, m) -> None:
        from dispatch_voice.transport import wire3
        now = self.now()
        if isinstance(m, (bytes, bytearray)):
            if not wire3.is_v3(m):
                return
            _codec, kind, gen, seq, _t_ms, packets = wire3.unpack_down(bytes(m))
            if kind == 3 or (gen & 0xFFFF) != (self.gen & 0xFFFF):
                return
            ms = sum(len(p) for p in packets) / 2 / OUT_RATE * 1000
            if kind == 0:
                if self.first_reply_audio is None and now >= self.reply_audio_after:
                    self.first_reply_audio = now
                if seq not in self.seq_start:
                    self.seq_start[seq] = max(now, self.play_end)
                self.seq_ms[seq] = self.seq_ms.get(seq, 0.0) + ms
                self.play_end = max(self.play_end, self.seq_start[seq]) + ms / 1000
            return
        f = json.loads(m)
        self.frames.append((now, f))
        t = f.get("type")
        if t == "ready":
            self.sid = f.get("sid", "")
        elif t == "barge_in":
            self.gen = int(f.get("gen", self.gen + 1))
            cur = self.playing()
            # C.3: one exact `heard` for the NEW gen at once (output latency 0 here).
            if cur is not None:
                await self.ws.send(json.dumps({"type": "heard", "gen": self.gen, "seq": cur[0],
                                               "played_ms": cur[1], "source": "server"}))
            self.seq_start.clear(); self.seq_ms.clear(); self.play_end = now
        elif t == "say_end":
            self.say_end = True
        elif t == "resume" and f.get("rewind"):
            cur = self.playing()
            if cur is not None:                     # the player restarts the sentence
                self.seq_start[cur[0]] = now
                self.play_end = now + sum(self.seq_ms[q] / 1000 for q in self.seq_ms if q >= cur[0])

    async def tick(self) -> None:
        """Periodic `heard` while a reply plays; `playback_done` once it drained."""
        now = self.now()
        cur = self.playing()
        if cur is not None and now - self._last_heard >= 1.0:
            self._last_heard = now
            await self.ws.send(json.dumps({"type": "heard", "gen": self.gen, "seq": cur[0],
                                           "played_ms": cur[1], "source": "server"}))
        if self.say_end and self.seq_ms and not self.done_sent and now >= self.play_end:
            self.done_sent = True
            await self.ws.send(json.dumps({"type": "playback_done", "gen": self.gen,
                                           "seq": max(self.seq_ms)}))


async def chat(url: str, script, *, max_s: float = 90.0) -> Client:
    """Open a chat, run ``script(client, stream)`` while a reader records every frame."""
    import websockets
    t0 = time.perf_counter()
    async with websockets.connect(url, max_size=1 << 20, open_timeout=60) as ws:
        c = Client(ws, t0)
        while True:
            m = await asyncio.wait_for(ws.recv(), 60)
            if isinstance(m, str) and json.loads(m).get("type") == "ready":
                await c.on_message(m)
                break
        await ws.send(json.dumps({"type": "start", "patience": "normal", "barge_in": True,
                                  "reply_style": "normal", "acks": "sometimes", "slow_link": "phone",
                                  "rt": {"partials": True, "speculate": True, "backchannels": False,
                                         "optimistic_duck": True}}))

        async def reader():
            async for m in ws:
                await c.on_message(m)

        rd = asyncio.create_task(reader())
        silence = b"\x00\x00" * (SR * UP_MS // 1000)

        async def stream(x: np.ndarray | None = None, seconds: float = 0.0, until=None) -> float:
            """Send ``x`` (or silence) as a real-time mic would: each 20 ms message when its
            LAST sample has been "captured". Returns the wall time of the first sample, so
            sample k was spoken at ``start + k / 16000`` (the latency reference)."""
            pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes() if x is not None else b""
            step = SR * UP_MS // 1000 * 2
            n = len(pcm) // step if x is not None else int(seconds * 1000 / UP_MS)
            start = time.perf_counter()
            for i in range(n):
                delay = start + (i + 1) * UP_MS / 1000 - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                await c.send_pcm(pcm[i * step:(i + 1) * step] if x is not None else silence)
                await c.tick()
                if until is not None and until():
                    break
            return start

        try:
            await asyncio.wait_for(script(c, stream), max_s)
        except TimeoutError:
            c.timed_out = True
        finally:
            rd.cancel()
            with contextlib.suppress(Exception):
                await ws.send(json.dumps({"type": "cancel"}))
    return c


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #

def new_thread(base: str) -> str:
    import httpx
    return httpx.post(base + "/api/threads", json={"bot_id": BOT}, timeout=10).json()["id"]


def ws_url(base: str, tid: str) -> str:
    return (base.replace("http", "ws") + f"/ws/voice/{tid}?d=dev-{uuid.uuid4().hex[:12]}"
            f"&s=str-{uuid.uuid4().hex[:12]}&v=3&up=pcm16&down=pcm16")


def thread_rows(base: str, tid: str) -> list[dict]:
    import httpx
    msgs = httpx.get(base + f"/api/threads/{tid}/messages", timeout=10).json()
    return msgs["messages"] if isinstance(msgs, dict) else msgs


def fill_metrics(row: Row, c: Client, after: float = 0.0) -> None:
    if c.timed_out:
        row.notes.append("timed out")
    a = c.first("say", kind="ack")
    # The line is timed from the `eot` of the turn it covers: the LAST `eot` before it. A
    # turn that ended early and was then resumed (eot -> more speech -> eot, one merged
    # submit) has two; timing from the first made a correct +0.7 s line read +5 s (V1).
    e = c.last_before("eot", a[0]) if a else None
    if e and a:
        row.ack_after_eot_s = round(a[0] - e[0], 3)
    row.earcon = c.first("ack") is not None
    for _, f in c.frames:
        if f.get("type") != "metrics":
            continue
        if "eot_to_first_audio_ms" in f and row.server_eot_to_first_audio_ms is None:
            row.server_eot_to_first_audio_ms = f.get("eot_to_first_audio_ms")
            row.first_chunk_ms = f.get("first_chunk_ms")
            row.ack_played = f.get("ack_played")
            row.ack_eta_ms = f.get("ack_eta_ms")
        if "eot_to_transcript_ms" in f and row.p_done is None:
            row.p_done = f.get("p_done")
            row.hold_ms = f.get("hold_ms")
            row.speculated = f.get("speculated")
            row.speculation_hit = f.get("speculation_hit")
            row.speculation = f.get("speculation")
        if "interrupt_to_submit_ms" in f:
            row.interrupt_to_submit_ms = f.get("interrupt_to_submit_ms")


def base_row(scenario: str, clip: str, load: str) -> Row:
    prof, mhz = platform_profile(), max_mhz()
    return Row(scenario=scenario, clip=clip, load=load, profile=prof, max_mhz=mhz, load1=load1(),
               column=column_of(prof, mhz))


async def sc_turn(base: str, marks: Marks, clip: str, load: str) -> Row:
    x = load_clip(clip)
    row = base_row("turn", clip, load)
    end_s = marks.end_s(x)
    row.speech_end_s = round(end_s, 3)
    mark = {}

    async def script(c: Client, stream):
        mark["t"] = await stream(x)
        await stream(seconds=25, until=lambda: c.first_reply_audio is not None)
        await stream(seconds=20, until=lambda: c.done_sent)

    c = await chat(ws_url(base, new_thread(base)), script)
    t_end = mark["t"] + end_s
    rel = lambda t: None if t is None else round(t - t_end, 3)
    e = c.first("eot")
    row.eot_at = rel(e[0]) if e else None
    row.eot_soft = c.first("eot_soft") is not None
    tr = c.first("transcript")
    row.transcript_at, row.transcript = (rel(tr[0]), tr[1].get("text", "")) if tr else (None, "")
    sy = c.first("say", kind="reply")
    row.first_say_at = rel(sy[0]) if sy else None
    row.first_audio_at = rel(c.first_reply_audio)
    fill_metrics(row, c)
    row.ok = row.first_audio_at is not None and bool(row.transcript)
    return row


async def sc_barge(base: str, marks: Marks, load: str) -> Row:
    q, intr = load_clip("short"), load_clip("interrupt")
    row = base_row("barge", "interrupt", load)
    onset = marks.onset_s(intr)
    tid = new_thread(base)
    mark = {}

    async def script(c: Client, stream):
        await stream(q)
        await stream(seconds=25, until=lambda: c.first_reply_audio is not None)
        await stream(seconds=0.8)                       # 0.8 s into the reply
        mark["t"] = await stream(intr)
        mark["n_tr"] = 1
        await stream(seconds=20, until=lambda: sum(1 for _, f in c.frames
                                                   if f.get("type") == "transcript" and f.get("text")) >= 2)
        await stream(seconds=1.0)

    c = await chat(ws_url(base, tid), script, max_s=90)
    t_on = mark["t"] + onset
    rel = lambda t: None if t is None else round(t - t_on, 3)
    d = c.first("duck", after=mark["t"], on=True)
    b = c.first("barge_in", after=mark["t"])
    trs = [(at, f) for at, f in c.frames if f.get("type") == "transcript" and f.get("text")]
    row.duck_at = rel(d[0]) if d else None
    row.barge_in_at = rel(b[0]) if b else None
    row.interrupt_transcript_at = rel(trs[1][0]) if len(trs) > 1 else None
    row.transcript = trs[1][1].get("text", "") if len(trs) > 1 else ""
    fill_metrics(row, c)
    await asyncio.sleep(1.5)                            # the note lands on the row
    rows = thread_rows(base, tid)
    row.heard_noted = any(isinstance((m.get("metadata") or {}).get("voice_heard"), dict)
                          and "upto" in m["metadata"]["voice_heard"] for m in rows if m["role"] == "assistant")
    row.user_rows = sum(1 for m in rows if m["role"] == "user")
    row.ok = bool(b) and row.heard_noted and row.user_rows == 2
    return row


async def sc_false(base: str, marks: Marks, load: str) -> Row:
    q, bc = load_clip("short"), load_clip("backchannel")
    row = base_row("false", "backchannel", load)
    onset = marks.onset_s(bc)
    tid = new_thread(base)
    mark = {}

    async def script(c: Client, stream):
        await stream(q)
        await stream(seconds=25, until=lambda: c.first_reply_audio is not None)
        await stream(seconds=0.6)
        mark["t"] = await stream(bc)
        await stream(seconds=4.0, until=lambda: c.first("resume", after=mark["t"]) is not None
                     or c.first("barge_in", after=mark["t"]) is not None)
        await stream(seconds=1.0)

    c = await chat(ws_url(base, tid), script)
    t_on = mark["t"] + onset
    rel = lambda t: None if t is None else round(t - t_on, 3)
    d = c.first("duck", after=mark["t"], on=True)
    r = c.first("resume", after=mark["t"])
    b = c.first("barge_in", after=mark["t"])
    row.duck_at = rel(d[0]) if d else None
    row.resume_at = rel(r[0]) if r else None
    row.resume_rewind = r[1].get("rewind") if r else None
    row.barge_in_at = rel(b[0]) if b else None
    fill_metrics(row, c)
    rows = thread_rows(base, tid)
    row.user_rows = sum(1 for m in rows if m["role"] == "user")
    echo = c.first("echo_ignored", after=mark["t"])
    if echo:
        row.notes.append(f"echo_ignored: {echo[1].get('text')!r}")
    row.ok = bool(r) and not b and row.user_rows == 1
    return row


async def sc_addendum(base: str, marks: Marks, load: str) -> Row:
    x = load_clip("addendum")
    row = base_row("addendum", "addendum", load)
    row.speech_end_s = round(marks.end_s(x), 3)
    tid = new_thread(base)
    mark = {}

    async def script(c: Client, stream):
        mark["t"] = await stream(x)
        await stream(seconds=25, until=lambda: c.first_reply_audio is not None)
        await stream(seconds=20, until=lambda: c.done_sent)

    c = await chat(ws_url(base, tid), script)
    t_end = mark["t"] + row.speech_end_s
    trs = [f for _, f in c.frames if f.get("type") == "transcript" and f.get("text")]
    row.merged = bool(trs) and trs[-1].get("merged") is True
    row.transcript = trs[-1].get("text", "") if trs else ""
    row.first_audio_at = None if c.first_reply_audio is None else round(c.first_reply_audio - t_end, 3)
    fill_metrics(row, c)
    await asyncio.sleep(1.0)
    rows = thread_rows(base, tid)
    users = [m for m in rows if m["role"] == "user"]
    row.user_rows = len(users)
    row.replies = sum(1 for m in rows if m["role"] == "assistant")
    # Both parts in the one row: Parakeet hears the second as "oh, and tomorrow" or "Oh, in to
    # morrow," — the merge is what is checked, not the recogniser's spelling.
    last = users[-1]["content"].lower() if users else ""
    both = "calendar" in last and "morrow" in last
    row.ok = row.user_rows == 1 and both and row.replies == 1
    if not both:
        row.notes.append(f"user rows: {[u['content'] for u in users]}")
    return row


# --------------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------------- #

def verdicts(rows: list[Row], tts: str) -> list[str]:
    out: list[str] = []
    groups: dict[tuple[str, str], list[Row]] = {}
    for r in rows:
        if r.scenario == "turn" and r.first_audio_at is not None and not r.load.startswith("embed"):
            groups.setdefault((r.load, r.column), []).append(r)
    for (load, col), rs in sorted(groups.items()):
        thr = THRESHOLDS[(tts, col)]
        vals = [r.first_audio_at for r in rs]
        med = statistics.median(vals)
        profs = sorted({f"{r.profile}/{r.max_mhz}" for r in rs})
        out.append(f"{'PASS' if med <= thr else 'FAIL'} first audio {load} [{tts}, {col} column, {len(rs)} rows, "
                   f"{'|'.join(profs)}]: median {med:.2f} s (max {max(vals):.2f}) vs ≤ {thr} s")
        out.append(f"INFO {load} [{col}] medians: eot {statistics.median(r.eot_at for r in rs if r.eot_at is not None):.2f} s, "
                   f"transcript {statistics.median(r.transcript_at for r in rs if r.transcript_at is not None):.2f} s, "
                   f"first say {statistics.median(r.first_say_at for r in rs if r.first_say_at is not None):.2f} s, "
                   f"server eot→audio {statistics.median(r.server_eot_to_first_audio_ms for r in rs if r.server_eot_to_first_audio_ms is not None):.0f} ms, "
                   f"first chunk {statistics.median(r.first_chunk_ms for r in rs if r.first_chunk_ms is not None):.0f} ms; "
                   f"speculated {sum(bool(r.speculated) for r in rs)}/{len(rs)}, adopted {sum(bool(r.speculation_hit) for r in rs)}/{len(rs)}")
    base = [r.first_audio_at for r in rows if r.scenario == "turn" and r.clip == "short" and r.load == "idle"
            and r.first_audio_at is not None]
    for load in sorted({r.load for r in rows if r.load.startswith("embed")}):
        vals = [r.first_audio_at for r in rows if r.load == load and r.first_audio_at is not None]
        if not vals or not base:
            continue
        extra = statistics.median(vals) - statistics.median(base)
        out.append(f"{'PASS' if extra <= 0.3 + 0.15 else 'FAIL'} advisor retrieval budget ({load} s embedder): "
                   f"first audio {statistics.median(vals):.2f} s, {extra:+.2f} s vs the idle short turns "
                   f"(budget +0.3 s, +0.15 s noise)")
    turns = [r for r in rows if r.scenario == "turn" and r.ack_eta_ms is not None]
    if turns:
        played = [r for r in turns if r.ack_played]
        bad = [r for r in played if (r.ack_eta_ms or 0) < 1200 or (r.ack_after_eot_s is not None
                                                                   and not 0.5 <= r.ack_after_eot_s <= 0.9)]
        out.append(f"{'PASS' if not bad else 'FAIL'} acknowledgement gate: {len(played)}/{len(turns)} turns acked "
                   f"(ETA {sorted({r.ack_eta_ms for r in turns})} ms; line after eot "
                   f"{[r.ack_after_eot_s for r in played]} s), earcon {sum(r.earcon for r in turns)}; "
                   f"a line only at +0.55 s (ack_after_s) and only when the ETA ≥ 1.2 s")
    starts = [r.speculation.get("started", 0) for r in rows if isinstance(r.speculation, dict)]
    if starts:
        out.append(f"{'PASS' if max(starts) <= 2 else 'FAIL'} speculation ≤ 2 starts per turn (max {max(starts)} over a chat)")
    for load in sorted({r.load for r in rows}):
        b = [r for r in rows if r.scenario == "barge" and r.load == load]
        if b:
            vals = [r.barge_in_at for r in b if r.barge_in_at is not None]
            ok = len(vals) == len(b) and max(vals) <= 1.0
            out.append(f"{'PASS' if ok else 'FAIL'} onset → barge_in ≤ 1.0 s ({load}): {vals}; server duck {[r.duck_at for r in b]}; "
                       f"heard noted {[r.heard_noted for r in b]}; interrupt→submit ms {[r.interrupt_to_submit_ms for r in b]}")
        f = [r for r in rows if r.scenario == "false" and r.load == load]
        if f:
            vals = [r.resume_at for r in f if r.resume_at is not None]
            ok = len(vals) == len(f) and max(vals) <= 2.0 and not any(r.barge_in_at for r in f) and all(r.ok for r in f)
            out.append(f"{'PASS' if ok else 'FAIL'} false interruption resumed ≤ 2.0 s ({load}): {vals}, rewind {[r.resume_rewind for r in f]}")
        a = [r for r in rows if r.scenario == "addendum" and r.load == load]
        if a:
            ok = all(r.ok for r in a)
            out.append(f"{'PASS' if ok else 'FAIL'} addendum → one row + one reply ({load}): rows {[r.user_rows for r in a]}, "
                       f"replies {[r.replies for r in a]}, merged {[r.merged for r in a]}")
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

async def one_pass(base: str, marks: Marks, load: str, clips: list[str], runs: int, scenarios: set[str]) -> list[Row]:
    rows: list[Row] = []
    for _ in range(runs):
        if "turn" in scenarios:
            for clip in clips:
                r = await sc_turn(base, marks, clip, load)
                print(json.dumps(asdict(r)), flush=True)
                rows.append(r)
                await asyncio.sleep(1.0)
        for name, fn in (("barge", sc_barge), ("false", sc_false), ("addendum", sc_addendum)):
            if name in scenarios:
                r = await fn(base, marks, load)
                print(json.dumps(asdict(r)), flush=True)
                rows.append(r)
                await asyncio.sleep(1.0)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, type=Path, help="DisPatch backend dir (2.2.0+, extension host)")
    ap.add_argument("--port", type=int, default=19032)
    ap.add_argument("--llm-port", type=int, default=0,
                    help="the stub LLM's loopback port (default 0: any free port)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kokoro", type=Path, default=None, help="the Kokoro plugin checkout (speak with Kokoro)")
    ap.add_argument("--clips", default="short,medium,long,pause")
    ap.add_argument("--scenarios", default="turn,barge,false,addendum")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--load-n", type=int, default=8)
    ap.add_argument("--passes", default="idle,load", help="idle, load or both")
    ap.add_argument("--power-hold", default="performance", choices=("performance", "off"),
                    help="the scratch server's rt.power_hold")
    ap.add_argument("--hold", default=None, choices=("performance", "power-saver"),
                    help="hold this profile for the whole run (PowerHold, released at the end)")
    ap.add_argument("--llm-delay", type=float, default=1.2)
    ap.add_argument("--embed-delays", default="",
                    help="e.g. 0,1.7,2.5: the stub's /embeddings delay for extra advisor turns (the 0.3 s voice "
                         "retrieval budget, feasibility C1): each must stay within +0.3 s of the idle turns")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    a = ap.parse_args(argv)

    from stub_llm import StubLLM
    a.out.mkdir(parents=True, exist_ok=True)
    tts = "kokoro" if a.kokoro else "chatterbox"
    hold = None
    hold_state = "off"
    if a.hold:
        from dispatch_voice.host.power import PowerHold
        hold = PowerHold(profile=a.hold)
        hold_state = f"{a.hold}:{hold.acquire()}"
        time.sleep(2.0)
    report: dict = {"header": {"tts": tts, "power_hold_server": a.power_hold, "hold": hold_state,
                               "platform_profile": platform_profile(), "max_mhz": max_mhz(), "load1": load1(),
                               "llm_delay_s": a.llm_delay, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    print(json.dumps(report["header"]), flush=True)
    rows: list[Row] = []
    proc = None
    try:
        with StubLLM(a.llm_port, delay_s=a.llm_delay, reply="normal") as llm:
            proc = start_scratch(a.port, a.out / "data", a.backend.expanduser(), llm.base_url,
                                 kokoro=a.kokoro.expanduser() if a.kokoro else None, power_hold=a.power_hold)
            base = f"http://127.0.0.1:{a.port}"
            st = wait_warm(base)
            report["status"] = {k: st.get(k) for k in ("warm", "ready", "default_voice", "proto", "rt", "power_hold")}
            print(json.dumps({"status": report["status"]}), flush=True)
            marks = Marks()
            clips = [c for c in a.clips.split(",") if c]
            scen = {s for s in a.scenarios.split(",") if s}
            # one warm-up chat (the first sentence of a voice prepares its prefix)
            asyncio.run(sc_turn(base, marks, "short", "warmup"))
            for d in [float(v) for v in a.embed_delays.split(",") if v.strip()]:
                llm.configure(embed_delay_s=d)
                for _ in range(a.runs):
                    r = asyncio.run(sc_turn(base, marks, "short", f"embed{d:g}"))
                    print(json.dumps(asdict(r)), flush=True)
                    rows.append(r)
                    time.sleep(1.0)
            llm.configure(embed_delay_s=0.0)
            for p in a.passes.split(","):
                if p == "idle":
                    rows += asyncio.run(one_pass(base, marks, "idle", clips, a.runs, scen))
                elif p == "load":
                    from load import cpu_load
                    with cpu_load(a.load_n):
                        time.sleep(1.0)
                        rows += asyncio.run(one_pass(base, marks, f"load{a.load_n}", clips, a.runs, scen))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(20)
            except subprocess.TimeoutExpired:
                proc.kill()
        if hold is not None:
            hold.release_all()
    vs = verdicts(rows, tts)
    print("\n".join(vs))
    report["rows"] = [asdict(r) for r in rows]
    report["verdicts"] = vs
    (a.out / "latency_ws.json").write_text(json.dumps(report, indent=1))
    if a.do_assert and any(v.startswith("FAIL") for v in vs):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
