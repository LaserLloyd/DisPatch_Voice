#!/usr/bin/env python3
"""Voice chat end to end on a SCRATCH server (never a live one).

    uv run python tests/e2e_voice_roundtrip.py --port 19020                       # fake host (no DisPatch)
    uv run python tests/e2e_voice_roundtrip.py --port 19020 --backend <dispatch-chat>/backend
    uv run --with playwright python tests/e2e_voice_roundtrip.py --backend ... --browser

Two hosts:

* ``--host fake`` (the default without ``--backend``): ``tests/host/fake_host.py``,
  the contract double of DisPatch (tests/fakes.py) with the add-on's own
  ``setup(ctx)``, served by uvicorn from this environment. Everything on the
  add-on's side is real: the models, ``ws.serve``, the session, ``/api/voice``.
* ``--host dispatch`` (``--backend DIR`` or ``DISPATCH_BACKEND``): the real
  DisPatch (2.2.0+, with the extension host) started as the live unit starts
  it — ``uv run --offline --with-editable <this checkout> uvicorn app.main:app``
  — with ONE API bot pointed at a stub LLM (tests/stub_llm.py) and
  ``DISPATCH_EXTENSIONS=dispatch_voice:EXTENSION``. No PIN is set, which
  DisPatch treats as unlocked (the PIN/cookie gate is the contract test's).

What it does:

1. Starts the server on ``--port`` with a throwaway data dir (``DISPATCH_DATA_DIR``),
   ``DISPATCH_VOICE=1`` and the speech models from ``DISPATCH_SPEECH_MODELS``.
2. Takes the driver's two utterances from ``tests/fixtures/clips`` (the stock
   voice saying "What's on my calendar for this afternoon?" and "Actually,
   wait, stop. Just tell me the first one."), or ``--wav`` (16 kHz mono 16-bit).
3. Wire mode: opens /ws/voice/<thread>, streams the WAV in REAL TIME (20 ms
   frames) + silence, and records eot / transcript / first reply audio. Then
   speaks again while the reply is "playing" and times the duck and barge-in.
   Browser mode (``--browser``, DisPatch host only): Chromium with a fake mic
   opens the app, presses the mic button and waits for the transcript and the
   reply in the voice-chat view.
   With ``--v3`` the socket speaks protocol v3 like the real client (binary frames with
   offsets, ``say``/``audio_end``, a device and stream id, the reconnect with the same
   ids) and two more phases run: "Tap to continue" (a tap-stop in the reply's second
   piece, then ``replay_last``: the reply goes on from that piece, its audio at once) and
   a reconnect of the same voice chat while the old socket is still open (the old
   session is replaced: 4001, one session counted).
4. Checks the thread through the HTTP API: the transcript is a user row with
   metadata.voice, the bot's reply follows it, the agent received the spoken
   instructions (the stub LLM's system prompt / the fake host's call log), and
   ``/api/voice/status`` read ``power_hold: held`` and ``sessions.open >= 1``
   while the chat was open and ``power_hold: off`` after it (``--allow-no-power-hold``
   for a box without power-profiles-daemon).

5. ``--rig ws://<rig-tailnet-ip>:1236/v1/realtime`` (``--host dispatch``; the token is
   ``SPEECHFORGE_TOKEN`` in the environment, never on the command line): the scratch
   ``voice.yaml`` turns the GPU rig on, the wire test waits until the rig is proven before
   it speaks, and the checks add: the rig was proven, the first utterance was recognised
   and the reply spoken on the rig (``/api/voice/status`` rig counters), no reply fell back
   for being late, and the power hold was NOT taken while the rig served. With
   ``--voice-clip`` the clone reaches the rig by HEAD 404 → PUT; without it the stock voice.

Prints a JSON summary (latencies in ms) and exits non-zero on any failure.
Needs the models (``dispatch-voice fetch-models``) and, for ``--host dispatch``,
a DisPatch checkout with ``app/extensions.py``.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

ADDON = Path(__file__).resolve().parents[1]
TESTS = ADDON / "tests"
sys.path.insert(0, str(TESTS))
REPLY = ("You have two things this afternoon. A dentist appointment at three, and a call "
         "with the school at four thirty. Want me to remind you before each one?")
CLIP_1 = TESTS / "fixtures" / "clips" / "driver1b.wav"     # "What's on my calendar for this afternoon?"
CLIP_2 = TESTS / "fixtures" / "clips" / "driver2.wav"      # "Actually, wait, stop. Just tell me the first one."
#: ``--rig``: the rig URL when the GPU rig is in play (set by main()).
RIG_URL: str | None = None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #

def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def read_wav16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, \
            f"{path} must be 16 kHz mono 16-bit"
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768


# --------------------------------------------------------------------------- #
# Servers
# --------------------------------------------------------------------------- #

def _wait_up(port: int, proc: subprocess.Popen, log: Path, timeout_s: float = 180) -> None:
    import httpx
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        with contextlib.suppress(httpx.HTTPError, OSError):
            if httpx.get(f"http://127.0.0.1:{port}/api/voice/status", timeout=1).status_code == 200:
                return
        time.sleep(0.5)
    proc.terminate()
    raise SystemExit(f"server did not come up; see {log}")


def start_fake_host(port: int, data: Path, *, power_hold: str | None = None) -> subprocess.Popen:
    data.mkdir(parents=True, exist_ok=True)
    if power_hold is not None:
        (data / "voice.yaml").write_text(json.dumps({"rt": {"power_hold": power_hold}}) + "\n", encoding="utf-8")
        os.chmod(data / "voice.yaml", 0o600)
    env = dict(os.environ, DISPATCH_DATA_DIR=str(data), DISPATCH_VOICE="1")
    log = data / "server.log"
    argv = [sys.executable, "-m", "uvicorn", "--app-dir", str(TESTS / "host"), "fake_host:app",
            "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
    with open(log, "wb") as f:                   # the child keeps its own copy of the descriptor
        proc = subprocess.Popen(argv, cwd=ADDON, env=env, stdout=f, stderr=subprocess.STDOUT)
    _wait_up(port, proc, log)
    return proc


#: The live unit's uvicorn flags (SPEC §11 step 2): ``--ws-max-size`` and the ping
#: timeout are load-bearing (hardening M5, the keepalive over a slow link).
UNIT_UVICORN_FLAGS = ("--no-access-log", "--ws-ping-timeout", "60", "--ws-max-size", "262144")


def start_dispatch(port: int, data: Path, backend: Path, llm_base_url: str, *, online: bool,
                   kokoro: Path | None = None, power_hold: str | None = None,
                   bot_kind: str = "api", rig_url: str | None = None) -> subprocess.Popen:
    """The real DisPatch, started as the live unit starts it. With ``kokoro`` (the plugin
    checkout) it is the planned SPEC §11 step 2 unit exactly: both overlays
    ``--with-editable``, ``DISPATCH_VOICE_EXTENSIONS=dispatch_voice_kokoro:setup``, the add-on
    found through its entry point (no ``DISPATCH_EXTENSIONS``) and switched on by
    ``voice.yaml: enabled: true`` (step 3), not by the ``DISPATCH_VOICE`` override.
    ``power_hold`` (``off`` | ``performance``) goes into the scratch ``voice.yaml`` as
    ``rt.power_hold``: a scratch run that is not measuring the hold must not hold the
    box's performance profile (it only heats the CPU). ``bot_kind``: ``api`` (the reply
    is generated whole, stored, then streamed — simulated, ~2.5 s) or ``advisor`` (streamed
    as the LLM writes it, under a provisional ``run:`` id, stored at the end)."""
    data.mkdir(parents=True, exist_ok=True)
    prov = {"provider": "openai", "base_url": llm_base_url, "model": "stub", "api_key": "not-a-key"}
    bot = {"id": "voicetest", "name": "Voice Test", "emoji": "🎙", "order": 0, "visible": True, "safe": False}
    if bot_kind == "advisor":
        know = data / "knowledge"
        (know / "notes").mkdir(parents=True, exist_ok=True)
        (know / "_profile.md").write_text("# Profile\nThe owner keeps a family calendar.\n", encoding="utf-8")
        bot["advisor"] = {"owner": "Sam", "providers": [prov], "knowledge_dir": str(know),
                          "agents": ["worker"], "research_agent": "worker", "action_agent": "worker",
                          "embeddings": {"base_url": llm_base_url, "model": "stub-embed",
                                         "api_key": "not-a-key"}}
    else:
        bot["api"] = prov
    (data / "config.yaml").write_text(json.dumps({"bots": [bot]}, ensure_ascii=False), encoding="utf-8")
                                        # JSON is YAML; ascii escapes of an emoji
                                        # would parse as lone surrogates
    os.chmod(data / "config.yaml", 0o600)
    (data / "tools.yaml").write_text(
        "tools:\n- {id: voice-cloning, kind: builtin, enabled: true}\n", encoding="utf-8")
    env = dict(os.environ, DISPATCH_DATA_DIR=str(data),
               DISPATCH_MIRROR="0", DISPATCH_GATEWAY_WS="0",
               OPENCLAW_BIN="/nonexistent/openclaw")
    overlays = ["--with-editable", str(ADDON)]
    voice_yaml: dict = {}
    if kokoro is not None:
        overlays += ["--with-editable", str(kokoro)]
        env["DISPATCH_VOICE_EXTENSIONS"] = "dispatch_voice_kokoro:setup"
        env.pop("DISPATCH_EXTENSIONS", None)
        env.pop("DISPATCH_VOICE", None)
        voice_yaml["enabled"] = True
    else:
        env.update(DISPATCH_VOICE="1", DISPATCH_EXTENSIONS="dispatch_voice:EXTENSION")
    if power_hold is not None:
        voice_yaml["rt"] = {"power_hold": power_hold}
    if rig_url:
        voice_yaml["rig"] = {"enabled": True, "url": rig_url}   # the token stays in the environment
    if voice_yaml:
        (data / "voice.yaml").write_text(json.dumps(voice_yaml) + "\n", encoding="utf-8")     # JSON is YAML
        os.chmod(data / "voice.yaml", 0o600)
    uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    argv = [uv, "run", "--quiet", "--no-sync", *([] if online else ["--offline"]),
            *overlays, "uvicorn", "app.main:app",
            "--host", "127.0.0.1", "--port", str(port), *UNIT_UVICORN_FLAGS]
    log = data / "server.log"
    with open(log, "wb") as f:                   # the child keeps its own copy of the descriptor
        proc = subprocess.Popen(argv, cwd=backend, env=env, stdout=f, stderr=subprocess.STDOUT)
    _wait_up(port, proc, log)
    return proc


# --------------------------------------------------------------------------- #
# Wire test
# --------------------------------------------------------------------------- #

def v3_query(device: str, stream: str) -> str:
    """The v3 socket query (C.1) — PCM16 both ways: the harness has no WebCodecs."""
    return f"?d={device}&s={stream}&v=3&up=pcm16&down=pcm16"


def new_id(prefix: str) -> str:
    import secrets
    return f"{prefix}-{secrets.token_hex(6)}"


class Uplink:
    """The client's mic: 20 ms frames in real time — raw PCM16 (v2) or v3 binary frames
    carrying their 16 kHz sample offset (C.2), one counter for the whole socket."""

    def __init__(self, ws, v3: bool) -> None:
        self.ws, self.v3, self.off = ws, v3, 0

    async def stream(self, x: np.ndarray, trailing_s: float = 0.0) -> None:
        from dispatch_voice.transport import wire3
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes() + b"\x00\x00" * int(16000 * trailing_s)
        step = 640                                   # 20 ms
        nxt = time.monotonic()
        for i in range(0, len(pcm), step):
            frame = pcm[i:i + step]
            if self.v3:
                await self.ws.send(wire3.pack_up(wire3.CODEC_PCM16, 0, self.off, [frame]))
            else:
                await self.ws.send(frame)
            self.off = (self.off + len(frame) // 2) & 0xFFFFFFFF
            nxt += 0.02
            await asyncio.sleep(max(0, nxt - time.monotonic()))


def down_audio(m: bytes) -> tuple[int, int, int, bytes] | None:
    """A v3 downlink message -> (kind, gen, seq, pcm16 bytes); None for anything else."""
    from dispatch_voice.transport import wire3
    if not wire3.is_v3(m):
        return None
    codec, kind, gen, seq, _t, packets = wire3.unpack_down(m)
    return kind, gen, seq, b"".join(packets) if codec == wire3.CODEC_PCM16 else b""


async def wire_test(base: str, tid: str, speech1: np.ndarray, speech2: np.ndarray, out: Path, *,
                    v3: bool = False) -> dict:
    import websockets
    device, stream_id = new_id("dev"), new_id("str")
    url = base.replace("http", "ws") + f"/ws/voice/{tid}" + (v3_query(device, stream_id) if v3 else "")
    res: dict = {"events": [], "v3": v3}
    reply_pcm: list[bytes] = []
    t0 = time.monotonic()

    def ms(t):
        return round((t - t0) * 1000)

    async with websockets.connect(url, max_size=None, open_timeout=180) as ws:
        res["connect_ms"] = ms(time.monotonic())
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready", ready
        res["ready"] = ready
        res["connect_to_ready_ms"] = ms(time.monotonic()) - res["connect_ms"]
        # The session is open now. The power hold is taken beside it, off the
        # ready path (~0.3 s on this box), so poll the status route for up to
        # 2 s: what it says WHILE a chat is live is the §4.11 / §11 proof.
        import httpx
        st: dict = {}
        t_poll = time.monotonic()
        while True:
            r = await asyncio.to_thread(httpx.get, base + "/api/voice/status", timeout=5)
            st = r.json() if r.status_code == 200 else {}
            if st.get("power_hold") in ("held", "unavailable") or time.monotonic() - t_poll > 2.0:
                break
            await asyncio.sleep(0.1)
        res["status_during"] = {"power_hold": st.get("power_hold"), "sessions": st.get("sessions"),
                                "hold_seen_after_ready_ms": round((time.monotonic() - t_poll) * 1000)}
        await ws.send(json.dumps({"type": "start", "patience": "normal"}))
        pending_hdr = None
        marks: dict = {}
        if RIG_URL:
            # The rig's socket opens with the chat; speak once it is proven (silence meanwhile).
            t_rig = time.monotonic()
            mic0 = Uplink(ws, v3)
            rig_st: dict = {}
            while time.monotonic() - t_rig < 45:
                await mic0.stream(np.zeros(1600, np.float32), 0)
                rig_st = (await asyncio.to_thread(httpx.get, base + "/api/voice/status", timeout=5)).json().get("rig") or {}
                if rig_st.get("state") == "ready":
                    break
            res["rig_wait_ms"] = round((time.monotonic() - t_rig) * 1000)
            res["rig_at_start"] = {k: rig_st.get(k) for k in ("state", "lease", "version", "reason")}
            await asyncio.sleep(0.3)                  # the power gate releases off the loop
            st_on = (await asyncio.to_thread(httpx.get, base + "/api/voice/status", timeout=5)).json()
            res["power_hold_on_rig"] = st_on.get("power_hold")

        async def reader():
            nonlocal pending_hdr
            async for m in ws:
                now = time.monotonic()
                if isinstance(m, bytes) and v3:
                    got = down_audio(m)
                    if got is not None and got[0] == 0:          # kind 0: reply audio
                        reply_pcm.append(got[3])
                        audio_seqs.setdefault(got[2], now)
                        marks.setdefault("first_reply_audio", now)
                        marks["last_reply_audio"] = now
                    elif got is not None and got[0] == 1:
                        marks.setdefault("ack_audio", now)
                    continue
                if isinstance(m, bytes):
                    if pending_hdr and pending_hdr.get("kind") == "reply":
                        reply_pcm.append(m)
                        marks.setdefault("first_reply_audio", now)
                        marks["last_reply_audio"] = now
                    if pending_hdr and pending_hdr.get("kind") == "ack":
                        marks.setdefault("ack_audio", now)
                    pending_hdr = None
                    continue
                f = json.loads(m)
                res["events"].append({"t_ms": ms(now), **{k: v for k, v in f.items() if k != "type"}, "type": f["type"]})
                if f["type"] == "audio":
                    pending_hdr = f
                    if f.get("kind") == "reply":
                        audio_seqs.setdefault(f["seq"], now)
                elif f["type"] == "say":
                    says.append({**f, "_phase": len(res.get("transcript_frames", []))})
                    say_at.setdefault(f["seq"], now)
                    if f.get("kind") == "reply":
                        marks.setdefault("first_say", now)
                    elif f.get("kind") == "ack":
                        marks.setdefault("ack_say", now)
                elif f["type"] == "say_end":
                    marks.setdefault("say_end", now)
                elif f["type"] == "barge_in":
                    # C.3: the client answers a barge_in with one exact `heard` for the new gen.
                    marks.setdefault("barge_in", now)
                    last = max(audio_seqs) if audio_seqs else 1
                    await ws.send(json.dumps({"type": "heard", "gen": f["gen"], "seq": last,
                                              "played_ms": 400, "source": "server"}))
                elif f["type"] == "duck" and f.get("on"):
                    marks.setdefault("duck", now)
                elif f["type"] in ("eot", "transcript", "barge_in", "turn_error"):
                    marks.setdefault(f["type"], now)
                    if f["type"] == "transcript":
                        res.setdefault("transcripts", []).append(f["text"])
                        if f.get("text"):
                            res.setdefault("transcript_frames", []).append(f)
                elif f["type"] == "metrics":
                    res.setdefault("server_metrics", []).append(f)

        says: list = []
        say_at: dict = {}
        audio_seqs: dict = {}
        rt = asyncio.create_task(reader())
        mic = Uplink(ws, v3)
        if RIG_URL:
            mic.off = mic0.off                        # one sample clock for the socket
        stream = mic.stream

        t_start = time.monotonic()
        await stream(speech1, 0.0)
        speech_end_1 = time.monotonic()
        silence_task = asyncio.create_task(stream(np.zeros(16000 * 8, np.float32), 0))
        for _ in range(400):                             # wait for reply audio
            if "first_reply_audio" in marks:
                break
            await asyncio.sleep(0.05)
        silence_task.cancel()
        await asyncio.sleep(0.3)
        # Barge-in: speak while the reply is still "playing" (we never send
        # playback_done, so the server believes the client is still playing).
        speech2_start = time.monotonic()
        await stream(speech2, 1.0)
        # The words that cut the reply become the next turn: wait for its reply's SECOND
        # sentence (the first one's audio is then complete), then drop the socket.
        def reply2():
            return [x for x in says if x.get("kind") == "reply" and x["_phase"] >= 2]
        for _ in range(400):
            if len(reply2()) >= 2:
                break
            await asyncio.sleep(0.05)
        rt.cancel()
        r2 = reply2()
        frames = res.get("transcript_frames", [])
        res["resume"] = {"after": frames[1].get("message_id") if len(frames) >= 2 else None,
                         "message_id": r2[0].get("message_id") if r2 else None,
                         "complete_before": [x["text"] for x in r2[:-1]]}

    if RIG_URL:
        # The first socket waited for a proven rig: its turns and replies are the rig's.
        res["rig_metrics_main"] = ((await asyncio.to_thread(httpx.get, base + "/api/voice/status", timeout=5))
                                   .json().get("rig") or {}).get("metrics") or {}

    # Late joiner / resume (T3 acceptance): a NEW socket, as after a drop mid-reply. The
    # client names the reply it was in (`after_partial`: the sentences whose audio was
    # complete) and the user row it answers (`after`); the server must say only the rest.
    rs = res["resume"]
    if rs["after"] and rs["message_id"]:
        said: list[str] = []
        # (v3: the same device and stream ids, as drive.js reconnects; the old socket is closed)
        async with websockets.connect(url, max_size=None, open_timeout=60) as ws2:
            ready2 = json.loads(await ws2.recv())
            rs["new_sid"] = ready2.get("sid") != ready.get("sid")
            await ws2.send(json.dumps({"type": "start", "patience": "normal", "after": rs["after"],
                                       "after_partial": {"message_id": rs["message_id"],
                                                         "sentences": len(rs["complete_before"])}}))
            t_end = time.monotonic() + 20
            while time.monotonic() < t_end:
                try:
                    m = await asyncio.wait_for(ws2.recv(), max(0.1, t_end - time.monotonic()))
                except (TimeoutError, websockets.exceptions.ConnectionClosed):
                    break
                if isinstance(m, bytes):
                    continue
                f = json.loads(m)
                if f["type"] == "say" and f.get("kind") == "reply":
                    said.append(f["text"])
                elif f["type"] == "say_end":
                    rs["say_end"] = True
                    break
        rs["said_after"] = said

    res["timeline_ms"] = {k: ms(v) for k, v in sorted(marks.items(), key=lambda kv: kv[1])}
    res["say"] = {"sid": ready.get("sid"), "say_flag": ready.get("say"),
                  "count": len([x for x in says if x.get("kind") == "reply"]),
                  "kinds": sorted({x.get("kind") for x in says}),
                  "texts": [x["text"] for x in says if x.get("kind") == "reply"][:3],
                  "say_before_audio": all(say_at.get(q, float("inf")) <= t for q, t in audio_seqs.items()),
                  "say_end": "say_end" in marks}
    res["speech1_ms"] = {"start": ms(t_start), "end": ms(speech_end_1)}
    if "eot" in marks:
        res["latency_ms"] = {
            "speech_end_to_eot": round((marks["eot"] - speech_end_1) * 1000),
            "speech_end_to_transcript": round((marks.get("transcript", marks["eot"]) - speech_end_1) * 1000),
            "speech_end_to_ack_audio": round((marks["ack_audio"] - speech_end_1) * 1000) if "ack_audio" in marks else None,
            "speech_end_to_first_reply_audio": round((marks["first_reply_audio"] - speech_end_1) * 1000) if "first_reply_audio" in marks else None,
        }
    if "barge_in" in marks:
        res["latency_ms"]["speech2_onset_to_barge_in"] = round((marks["barge_in"] - speech2_start) * 1000)
    if "duck" in marks:
        # Since v2 the first reaction to talking over the reply is a DUCK
        # (playback lowered/paused); the hard barge_in waits for the words.
        res["latency_ms"]["speech2_onset_to_duck"] = round((marks["duck"] - speech2_start) * 1000)
    if reply_pcm:
        a = np.frombuffer(b"".join(reply_pcm), "<i2")
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(a.tobytes())
        res["reply_audio_s"] = round(len(a) / 24000, 2)
    return res


async def continue_test(base: str, tid: str, speech1: np.ndarray) -> dict:
    """v3: "Tap to continue" (C.3 `replay_last`, review-T3s4 #1/#2). The driver asks, the
    reply plays; once its second piece is playing the listener taps (`stop_playback` with
    the exact `heard`), then taps "continue": the session must go on from the START of the
    piece they were in, with its audio at once (the send-ahead clock does not wait for the
    audio the tap threw away)."""
    import websockets
    url = base.replace("http", "ws") + f"/ws/voice/{tid}" + v3_query(new_id("dev"), new_id("str"))
    out: dict = {}
    async with websockets.connect(url, max_size=None, open_timeout=60) as ws:
        ready = json.loads(await ws.recv())
        out["proto"] = ready.get("proto")
        out["replay_last_offered"] = ready.get("replay_last") is True
        await ws.send(json.dumps({"type": "start", "patience": "normal", "acks": "off"}))
        mic = Uplink(ws, True)
        says: list[dict] = []
        audio: dict[tuple[int, int], float] = {}         # (gen, seq) -> first audio
        gens: list[int] = []
        state = {"gen": 0}

        async def reader():
            async for m in ws:
                now = time.monotonic()
                if isinstance(m, bytes):
                    got = down_audio(m)
                    if got is not None and got[0] == 0:
                        audio.setdefault((got[1], got[2]), now)
                    continue
                f = json.loads(m)
                if f["type"] == "say" and f.get("kind") == "reply":
                    says.append({**f, "_at": now})
                elif f["type"] == "barge_in":
                    gens.append(f["gen"])
                    state["gen"] = f["gen"]

        rt = asyncio.create_task(reader())
        talk = asyncio.create_task(mic.stream(speech1, 2.0))
        try:
            for _ in range(600):                          # the reply's second piece is playing
                if len(says) >= 2 and (says[1]["gen"], says[1]["seq"]) in audio:
                    break
                await asyncio.sleep(0.05)
            if len(says) < 2:
                out["error"] = "the reply never reached its second piece"
                return out
            cut = says[1]
            await asyncio.sleep(0.4)
            await ws.send(json.dumps({"type": "stop_playback", "heard": {
                "gen": cut["gen"], "seq": cut["seq"], "played_ms": 300, "source": "server"}}))
            for _ in range(100):
                if gens:
                    break
                await asyncio.sleep(0.02)
            if not gens:
                out["error"] = "no barge_in after the tap"
                return out
            # the client's exact `heard` for the new gen (C.3), then "Tap to continue"
            await ws.send(json.dumps({"type": "heard", "gen": gens[-1], "seq": cut["seq"],
                                      "played_ms": 300, "source": "server"}))
            await asyncio.sleep(0.5)
            n0 = len(says)
            t_tap = time.monotonic()
            await ws.send(json.dumps({"type": "replay_last"}))
            for _ in range(200):
                cont = [x for x in says[n0:] if x["gen"] == gens[-1]]
                if cont and (cont[0]["gen"], cont[0]["seq"]) in audio:
                    break
                await asyncio.sleep(0.05)
            cont = [x for x in says[n0:] if x["gen"] == gens[-1]]
            out["cut_piece"] = cut["text"]
            out["continued_with"] = [x["text"] for x in cont][:3]
            if cont:
                out["tap_to_say_ms"] = round((cont[0]["_at"] - t_tap) * 1000)
                first = audio.get((cont[0]["gen"], cont[0]["seq"]))
                if first is not None:
                    out["tap_to_first_audio_ms"] = round((first - t_tap) * 1000)
        finally:
            talk.cancel()
            rt.cancel()
    return out


async def twin_test(base: str, tid: str) -> dict:
    """v3: a reconnect of the SAME voice chat (device + stream ids) while the old socket is
    still open (half-open after a network change) replaces the old session at once — 4001
    "replaced" on the old socket, one session counted (review-T3s4 #4)."""
    import httpx
    import websockets
    url = base.replace("http", "ws") + f"/ws/voice/{tid}" + v3_query(new_id("dev"), new_id("str"))
    out: dict = {}
    async with websockets.connect(url, max_size=None, open_timeout=60) as a:
        json.loads(await a.recv())
        await a.send(json.dumps({"type": "start", "acks": "off"}))
        async with websockets.connect(url, max_size=None, open_timeout=60) as b:
            ready_b = json.loads(await b.recv())
            out["b_ready"] = ready_b.get("type") == "ready"
            await b.send(json.dumps({"type": "start", "acks": "off"}))
            t0 = time.monotonic()
            try:
                while True:                               # A's backlog, then its close frame
                    await asyncio.wait_for(a.recv(), max(0.1, 5 - (time.monotonic() - t0)))
            except websockets.exceptions.ConnectionClosed as e:
                out["a_close_code"] = e.rcvd.code if e.rcvd else None
                out["a_closed_after_ms"] = round((time.monotonic() - t0) * 1000)
            except TimeoutError:
                out["a_close_code"] = None
            for _ in range(20):
                st = (await asyncio.to_thread(httpx.get, base + "/api/voice/status", timeout=5)).json()
                out["sessions_open"] = (st.get("sessions") or {}).get("open")
                if out["sessions_open"] == 1:
                    break
                await asyncio.sleep(0.1)
    return out


def resume_ok(rs: dict, bots: list[dict]) -> bool:
    """The resumed socket said exactly the reply's sentences after the complete ones."""
    sys.path.insert(0, str(ADDON / "src"))
    from dispatch_voice.host.session import split_reply
    from dispatch_voice.text.speech_text import normalise
    if not rs.get("said_after") or not bots:
        return False
    content = bots[-1].get("content") or ""
    sentences = [normalise(sp.text, tags_ok=False) for sp in split_reply(content)]
    sentences = [t for t in sentences if t]
    n = len(rs.get("complete_before") or [])
    rs["expected_after"] = sentences[n:]
    return bool(rs.get("new_sid")) and rs["said_after"] == sentences[n:] and n >= 1


def build_voice(base: str, clip: Path) -> dict:
    """POST the clip to the profiles API (consent: synthetic), set it as the
    default, and wait for its fillers. Returns timings."""
    import httpx
    api = base + "/api/voice/profiles"
    t0 = time.monotonic()
    with clip.open("rb") as f:
        r = httpx.post(api, timeout=300,
                       data={"name": "E2E host", "consent": "synthetic",
                             "consent_note": "e2e test clip"},
                       files={"file": (clip.name, f, "audio/wav")})
    r.raise_for_status()
    prof = r.json()
    built = time.monotonic() - t0
    httpx.patch(f"{api}/{prof['id']}", json={"default": True}, timeout=30).raise_for_status()
    for _ in range(240):                     # fillers render in the background
        p = httpx.get(f"{api}/{prof['id']}", timeout=30).json()
        if p.get("fillers") == "ready":
            break
        time.sleep(0.5)
    return {"id": prof["id"], "build_s": round(built, 2),
            "fillers_wait_s": round(time.monotonic() - t0 - built, 2),
            "profile_keys": sorted(p.keys())}


# --------------------------------------------------------------------------- #
# Browser test (headless Chromium, fake mic; DisPatch host only)
# --------------------------------------------------------------------------- #

def browser_test(base: str, tid: str, wav: Path, shots: Path, chromium: str | None = None) -> dict:
    from playwright.sync_api import sync_playwright
    out: dict = {}
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=chromium or None, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={wav}", "--autoplay-policy=no-user-gesture-required"])
        page = b.new_page(viewport={"width": 1280, "height": 900})
        logs: list[str] = []
        page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        page.goto(base + f"/?popout=1&bot=voicetest&thread={tid}")
        page.wait_for_selector("#drive-btn:not([hidden])", timeout=30000)
        page.screenshot(path=str(shots / "voice-header.png"))
        t0 = time.monotonic()
        page.click("#drive-btn")
        page.wait_for_selector(".drive-overlay", timeout=10000)
        import re
        page.locator(".drive-you").filter(has_text=re.compile(r"\w{3}")).wait_for(timeout=60000)
        out["transcript_ui"] = page.inner_text(".drive-you")
        out["click_to_transcript_ms"] = round((time.monotonic() - t0) * 1000)
        page.locator(".drive-them").filter(has_text=re.compile(r"\w{3}")).wait_for(timeout=60000)
        out["reply_ui"] = page.inner_text(".drive-them")
        out["click_to_reply_text_ms"] = round((time.monotonic() - t0) * 1000)
        out["state"] = page.get_attribute(".drive-overlay", "data-state")
        page.screenshot(path=str(shots / "voice-view.png"))
        page.click(".drive-exit")
        out["console_errors"] = [line for line in logs if line.startswith("error")]
        b.close()
    return out


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", choices=("fake", "dispatch"), default=None,
                    help="fake (default without --backend) or dispatch (a real checkout)")
    ap.add_argument("--backend", type=Path, default=Path(os.environ["DISPATCH_BACKEND"]) if os.environ.get("DISPATCH_BACKEND") else None,
                    help="DisPatch's backend directory (implies --host dispatch)")
    ap.add_argument("--online", action="store_true", help="let uv reach the network for the overlay (default: --offline)")
    ap.add_argument("--kokoro", type=Path, default=None,
                    help="dispatch host: also overlay this Kokoro plugin checkout, exactly like the planned "
                         "live unit (SPEC §11 step 2: entry-point discovery, voice.yaml enabled)")
    ap.add_argument("--power-hold", choices=("off", "performance"), default=None,
                    help="the scratch server's rt.power_hold (default: the server's own, performance). "
                         "With off the hold checks expect 'off' during and after the chat (a run that is "
                         "not measuring the hold must not hold the box's performance profile)")
    ap.add_argument("--allow-no-power-hold", action="store_true",
                    help="accept status.power_hold 'unavailable' during the chat (a box without "
                         "power-profiles-daemon); by default it must read 'held'")
    ap.add_argument("--port", type=int, default=8798)
    ap.add_argument("--llm-port", type=int, default=0,
                    help="dispatch host: the stub LLM's loopback port (default 0: any free port)")
    ap.add_argument("--data", type=Path, default=None, help="scratch data dir (default: a new temp dir)")
    ap.add_argument("--work", type=Path, default=None, help="scratch work dir (default: a new temp dir)")
    ap.add_argument("--wav", type=Path, default=None, help="16 kHz mono 16-bit driver speech (default: a fixture clip)")
    ap.add_argument("--llm-delay", type=float, default=1.2, help="stub LLM seconds to the first token (dispatch host)")
    ap.add_argument("--browser", action="store_true",
                    help="also run the headless Chromium fake-mic test (dispatch host; needs playwright)")
    ap.add_argument("--chromium", default=None, help="Chromium executable for --browser (default: Playwright's own)")
    ap.add_argument("--wait-warm", action="store_true",
                    help="wait for the start-up model prewarm before the first session")
    ap.add_argument("--voice-clip", type=Path, default=None,
                    help="build a cloned-voice profile from this clip through /api/voice/profiles and make it the default")
    ap.add_argument("--v3", action="store_true",
                    help="the wire test speaks protocol v3 (binary frames, offsets, `say`/`audio_end`, the "
                         "stream id) like the real client, and adds the \"Tap to continue\" and "
                         "reconnect-replaces phases (review-T3s4 #11: v2 alone left them uncovered)")
    ap.add_argument("--rig", default=None, metavar="WS_URL",
                    help="use a SpeechForge rig (ws://<rig>:1236/v1/realtime; --host dispatch); the token is "
                         "SPEECHFORGE_TOKEN in the environment")
    args = ap.parse_args()
    host = args.host or ("dispatch" if args.backend else "fake")
    global RIG_URL
    RIG_URL = args.rig
    if args.rig and (host != "dispatch" or not os.environ.get("SPEECHFORGE_TOKEN")):
        raise SystemExit("--rig needs --host dispatch and SPEECHFORGE_TOKEN in the environment")
    if host == "dispatch" and not args.backend:
        raise SystemExit("--host dispatch needs --backend DIR (or DISPATCH_BACKEND)")

    work = args.work or Path(tempfile.mkdtemp(prefix="dispatch-voice-e2e-"))
    work.mkdir(parents=True, exist_ok=True)
    data = args.data or (work / "data")
    base = f"http://127.0.0.1:{args.port}"
    s1 = read_wav16k(args.wav or CLIP_1)
    s2 = read_wav16k(CLIP_2)
    llm = None
    if host == "dispatch":
        from stub_llm import StubLLM
        llm = StubLLM(args.llm_port, delay_s=args.llm_delay, fixed_reply=REPLY).start()
        proc = start_dispatch(args.port, data, args.backend, llm.base_url, online=args.online,
                              kokoro=args.kokoro, power_hold=args.power_hold, rig_url=args.rig)
    else:
        proc = start_fake_host(args.port, data, power_hold=args.power_hold)
    ok = True
    summary: dict = {"host": host, "work_dir": str(work)}
    try:
        import httpx
        st = httpx.get(base + "/api/voice/status").json()
        if args.wait_warm:
            t_w = time.monotonic()
            for _ in range(480):
                st = httpx.get(base + "/api/voice/status").json()
                if st.get("warm"):
                    break
                time.sleep(0.25)
            summary["warm_wait_s"] = round(time.monotonic() - t_w, 1)
        summary["status"] = {k: st.get(k) for k in ("enabled", "ready", "missing", "warm", "extension",
                                                     "power_hold", "sessions", "stock_voice", "default_voice")}
        if args.voice_clip:
            summary["voice_profile"] = build_voice(base, args.voice_clip)
        r = httpx.post(base + "/api/threads", json={"bot_id": "voicetest"})
        r.raise_for_status()
        tid = r.json()["id"]
        summary["wire"] = asyncio.run(wire_test(base, tid, s1, s2, work / "reply.wav", v3=args.v3))
        if args.rig:
            # The first socket waited for a proven rig: every turn and reply on it is the
            # rig's. (Later sockets may start before their rig is proven, which by design
            # runs those turns here.)
            summary["rig_wire_metrics"] = summary["wire"].pop("rig_metrics_main", None) or {}
        summary["status_during"] = summary["wire"].pop("status_during", None) or {}
        msgs = httpx.get(base + f"/api/threads/{tid}/messages").json()
        rows = msgs["messages"] if isinstance(msgs, dict) else msgs
        users = [m for m in rows if m["role"] == "user"]
        bots = [m for m in rows if m["role"] == "assistant"]
        summary["thread"] = {"user_rows": [(m["content"], (m.get("metadata") or {}).get("voice")) for m in users],
                             "assistant_rows": len(bots)}
        if host == "dispatch":
            seen = llm.chat_requests()
            summary["instructions_seen"] = any("Spoken replies" in (r.get("system") or "") for r in seen)
        else:
            calls = httpx.get(base + "/_fake/calls").json()
            summary["instructions_seen"] = any("Spoken replies" in ((c.get("instructions") or {}).get("system") or "")
                                               for c in calls["submit_turn"])
            summary["power_hold_after"] = calls["power_hold"]
        # The hold is released by the session's clean-up, a moment after the socket closed.
        for _ in range(50):
            summary["status_after"] = httpx.get(base + "/api/voice/status").json().get("power_hold")
            if summary["status_after"] != "held":
                break
            time.sleep(0.1)
        checks = {
            "transcript_heard": bool(summary["wire"].get("transcripts")) and "calendar" in summary["wire"]["transcripts"][0].lower(),
            "voice_row_persisted": any(v for _, v in summary["thread"]["user_rows"]),
            "assistant_replied": summary["thread"]["assistant_rows"] >= 1,
            "reply_audio_received": summary["wire"].get("reply_audio_s", 0) > 1,
            "barge_in_seen": "barge_in" in summary["wire"]["timeline_ms"],
            "instructions_sent": bool(summary["instructions_seen"]),
            "power_hold_released": summary["status_after"] in ("off", "unavailable"),
            # Held while the chat was open (SPEC §4.11; runbook §11 step 6b). A box
            # without power-profiles-daemon reads "unavailable": --allow-no-power-hold.
            "power_hold_held_during": (summary["status_during"].get("power_hold") == "off"
                                       if args.power_hold == "off" else
                                       summary["status_during"].get("power_hold") == "held"
                                       or (args.allow_no_power_hold
                                           and summary["status_during"].get("power_hold") == "unavailable")),
            "session_counted_during": (summary["status_during"].get("sessions") or {}).get("open", 0) >= 1,
            # Slice 2: `say` precedes every reply sentence's audio; a 16-hex sid; say_end (or a cut).
            "say_before_audio": summary["wire"]["say"]["count"] >= 1 and summary["wire"]["say"]["say_before_audio"],
            "sid_in_ready": isinstance(summary["wire"]["say"]["sid"], str) and len(summary["wire"]["say"]["sid"]) == 16,
            "heard_noted": any((m.get("metadata") or {}).get("voice_heard", {}).get("upto") is not None for m in bots),
            # T3 acceptance: a new socket resumes the reply it was cut from with only the
            # sentences it had not heard, each once (the late-joiner / after_partial rules).
            "resume_rest_spoken": resume_ok(summary["wire"].get("resume") or {}, bots),
        }
        if args.rig:
            # With the rig serving, the CPU power hold is not taken (ADDON-ENGINE §3).
            checks["power_hold_held_during"] = True
            checks["rig_proven"] = (summary["wire"].get("rig_at_start") or {}).get("state") == "ready"
            checks["no_power_hold_while_on_rig"] = summary["wire"].get("power_hold_on_rig") in ("off", "unavailable")
        if args.v3:
            checks["proto_3"] = (summary["wire"].get("ready") or {}).get("proto") == 3
            th3 = httpx.post(base + "/api/threads", json={"bot_id": "voicetest"}).json()["id"]
            summary["continue"] = c = asyncio.run(continue_test(base, th3, s1))
            checks["continue_from_the_cut_piece"] = bool(c.get("continued_with")) and \
                c["continued_with"][0] == c.get("cut_piece")
            # the kept audio goes out at once (pacing, review-T3s4 #2): the client's 5 s
            # first-audio deadline would flip it to the phone voice
            checks["continue_audio_prompt"] = c.get("tap_to_first_audio_ms") is not None and \
                c["tap_to_first_audio_ms"] <= 1500
            th4 = httpx.post(base + "/api/threads", json={"bot_id": "voicetest"}).json()["id"]
            summary["twin"] = t = asyncio.run(twin_test(base, th4))
            checks["reconnect_replaces_the_old_session"] = t.get("a_close_code") == 4001 and \
                t.get("sessions_open") == 1
        if args.browser:
            th2 = httpx.post(base + "/api/threads", json={"bot_id": "voicetest"}).json()["id"]
            summary["browser_thread"] = th2
            mic = work / "driver1-mic.wav"
            write_wav(mic, np.concatenate([np.zeros(16000, np.float32), s1,
                                           np.zeros(16000 * 40, np.float32)]), 16000)
            summary["browser"] = browser_test(base, th2, mic, work, args.chromium)
            checks["browser_transcript"] = "calendar" in summary["browser"].get("transcript_ui", "").lower()
            checks["browser_reply"] = len(summary["browser"].get("reply_ui", "")) > 3
        if args.rig:
            # The rig's counters after EVERY phase (the continue and reconnect phases too).
            rg = httpx.get(base + "/api/voice/status").json().get("rig") or {}
            m = rg.get("metrics") or {}
            summary["rig"] = {"wait_ms": summary["wire"].get("rig_wait_ms"), "at_start": summary["wire"].get("rig_at_start"),
                              "power_hold_on_rig": summary["wire"].get("power_hold_on_rig"), "metrics": m}
            wm = summary.get("rig_wire_metrics") or {}
            checks["rig_recognised_the_turns"] = (wm.get("stt") or 0) >= 1 and wm.get("stt") == wm.get("turns")
            checks["rig_spoke_the_replies"] = (wm.get("tts") or 0) >= 1 and wm.get("tts") == wm.get("replies")
            checks["rig_no_fallback"] = not m.get("fell_back")       # over the whole run
        summary["checks"] = checks
        ok = all(checks.values())
    except Exception as e:  # noqa: BLE001 - a harness reports every failure in its summary
        summary["error"] = repr(e)
        ok = False
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(15)
        if llm is not None:
            llm.stop()
    summary["ok"] = ok
    print(json.dumps(summary, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
