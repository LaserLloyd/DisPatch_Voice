"""It still speaks: the whole pipeline on the new seam with the REAL engines
and no DisPatch — Silero + Smart Turn end the turn, Parakeet transcribes a
fixture clip, the FakeContext persists the row and streams a scripted
advisor-style reply, Chatterbox (the stock voice) speaks it, and the
power-profile hold is held for the session and released after.

Model-backed: skips unless ``DISPATCH_SPEECH_MODELS`` names the speech models
and the Chatterbox models are fetched. Run it under the box-wide lock
(``flock <lockfile> uv run pytest ...``); it loads ~1.5 GB.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from fakes import FIXTURES, FakeWS, load_clip

from dispatch_voice.host import context, power, registry, routes
from dispatch_voice.host.config import load_settings
from dispatch_voice.host.engines import chatterbox

pytestmark = pytest.mark.models

REPLY = ("You have two things this afternoon. A dentist appointment at three, and a call "
         "with the school at four thirty. Want me to remind you before each one?")


@pytest.fixture
def real_env(fake_ctx, monkeypatch):
    if not os.environ.get("DISPATCH_SPEECH_MODELS"):
        pytest.skip("DISPATCH_SPEECH_MODELS not set (speech models)")
    for k in list(os.environ):
        if k.startswith("DISPATCH_VOICE_") and k != "DISPATCH_VOICE_MODELS":
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DISPATCH_VOICE", "1")
    monkeypatch.setenv("DISPATCH_VOICE_MERGE_WINDOW", "0.3")
    monkeypatch.delenv("DISPATCH_VOICE_EXTENSIONS", raising=False)
    context.set_context(fake_ctx)
    registry.reset()
    routes._LOADED[:] = []
    routes._APPLIED["on"] = None
    missing = load_settings().missing()
    if missing or not chatterbox.available():
        pytest.skip(f"models missing: {missing or 'Chatterbox'}")
    from dispatch_voice.host import setup
    setup(fake_ctx)
    yield fake_ctx
    registry.reset()
    context.clear_context()
    power.release_all()


def pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


async def test_a_fixture_clip_is_heard_answered_and_spoken(real_env):
    ctx = real_env
    ctx.reply_for = lambda text: REPLY
    ws = FakeWS()
    task = asyncio.create_task(ctx.connect("voice", "t-voice-1", ws))
    t0 = time.monotonic()
    while not ws.json_of("ready") and time.monotonic() - t0 < 120:
        await asyncio.sleep(0.05)
    ready = ws.json_of("ready")
    assert ready and ready[0]["smart_turn"] is True and ready[0]["sample_rate_out"] == 24000
    # The hold is taken beside the session, off the ready path: it lands a few
    # hundred ms after `ready` (the child's bus call), so poll for it.
    t_hold = time.monotonic()
    while power.state(load_settings()) == "off" and time.monotonic() - t_hold < 5:
        await asyncio.sleep(0.05)
    held_during = power.state(load_settings())
    hold_after_ready_ms = round((time.monotonic() - t_hold) * 1000)
    ws.push_json({"type": "start"})
    clip = np.concatenate([load_clip("short"), np.zeros(16000 * 2, np.float32)])
    data = pcm16(clip)
    for i in range(0, len(data), 640):                     # 20 ms chunks, as the browser sends them
        ws.push_audio(data[i:i + 640])
        await asyncio.sleep(0)
    def reply_seconds() -> float:
        return sum(len(b) for i, b in enumerate(ws.sent)
                   if isinstance(b, bytes) and isinstance(ws.sent[i - 1], dict)
                   and ws.sent[i - 1].get("kind") == "reply") / 2 / 24000

    t1 = time.monotonic()
    while time.monotonic() - t1 < 90:
        audio = [m for m in ws.json_of("audio") if m["kind"] == "reply"]
        speaking = any(m["type"] == "state" and m["state"] == "speaking" for m in ws.sent if isinstance(m, dict))
        # The pipelined engine streams 40 ms frames as they are decoded: wait for a
        # second of reply audio (or the reply's end), not just its first frame.
        if audio and speaking and (reply_seconds() >= 1.0 or ws.json_of("say_end")):
            break
        await asyncio.sleep(0.1)
    await ctx.drain()
    transcripts = ws.json_of("transcript")
    baseline = json.loads((FIXTURES / "transcripts.json").read_text())["16k"]["short"]
    assert transcripts and transcripts[0]["text"] == baseline
    rows = ctx.db().messages("t-voice-1")
    assert rows[0].role == "user" and rows[0].metadata == {"voice": True} and rows[0].content == baseline
    assert rows[1].role == "assistant" and rows[1].content == REPLY
    hdrs = [m for m in ws.json_of("audio") if m["kind"] == "reply"]
    assert hdrs and hdrs[0]["text"].startswith("You have two things")
    # Slice 2: every reply sentence is announced by a `say` before its first audio.
    says = [m for m in ws.json_of("say") if m["kind"] == "reply"]
    assert says and says[0]["text"].startswith("You have two things") and says[0]["seq"] == hdrs[0]["seq"]
    order = [(m["type"], m["seq"]) for m in ws.sent if isinstance(m, dict) and m.get("type") in ("say", "audio")]
    with_audio = [m for m in says if ("audio", m["seq"]) in order]     # later sentences may still be synthesising
    assert with_audio
    for m in with_audio:
        assert order.index(("say", m["seq"])) < order.index(("audio", m["seq"]))
    assert len(ready[0]["sid"]) == 16 and ready[0]["say"] is True
    pcm = b"".join(b for i, b in enumerate(ws.sent)
                   if isinstance(b, bytes) and isinstance(ws.sent[i - 1], dict)
                   and ws.sent[i - 1].get("kind") == "reply")
    seconds = len(pcm) / 2 / 24000
    assert seconds >= 1.0, f"only {seconds:.2f} s of reply audio"
    assert np.abs(np.frombuffer(pcm, "<i2")).max() > 500          # not silence
    metrics = [m for m in ws.json_of("metrics") if "eot_to_first_audio_ms" in m]
    assert metrics and metrics[0]["eot_to_first_audio_ms"] < 60000
    assert held_during in ("held", "unavailable")
    ws.hang_up()
    await asyncio.wait_for(task, 30)
    assert power.state(load_settings()) == "off"
    print(json.dumps({"transcript": transcripts[0]["text"], "reply_audio_s": round(seconds, 2),
                      "eot_to_first_audio_ms": metrics[0]["eot_to_first_audio_ms"],
                      "power_hold_during": held_during, "hold_after_ready_ms": hold_after_ready_ms,
                      **power.platform_facts()}))
    assert Path(ctx.paths.voices_dir, "stock-host", "profile.json").exists()   # the stock voice spoke
