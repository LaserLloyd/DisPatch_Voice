"""The rig inside a real voice chat (ws.serve + VoiceSession + the FakeContext):
with no rig configured the session is exactly today's; with one, the reply is spoken
by the rig, the power hold is not taken, and a barge-in cancels on the rig before
the host abort."""
from __future__ import annotations

import asyncio
import hashlib

import httpx
import numpy as np
import pytest
from rig.conftest import wait_for
from rig.fake_sf1 import TOKEN, FakeRig

from dispatch_voice.host import config as vconfig
from dispatch_voice.host import power, rig, routes, vad_turn
from dispatch_voice.host import ws as ws_mod
from dispatch_voice.host.engines import chatterbox
from dispatch_voice.host.engines.rig_tts import ReplyEngine
from settling import pause, settle, wait_until

THREAD = "t-voice-1"


@pytest.fixture(autouse=True)
def _clean_rig_state(monkeypatch, tmp_path):
    rig._MISCONFIGURED.clear()
    rig._LAST.clear()
    rig.METRICS.reset()
    monkeypatch.delenv(rig.TOKEN_ENV, raising=False)
    monkeypatch.setattr(rig, "token_path", lambda: tmp_path / "speechforge.env")
    yield
    rig._MISCONFIGURED.clear()


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setenv(rig.TOKEN_ENV, TOKEN)
    return TOKEN


@pytest.fixture
async def fake_rig():
    r = await FakeRig().start()
    try:
        yield r
    finally:
        await r.stop()



@pytest.fixture
def served(host_env, monkeypatch, fake_engines, no_models, tmp_path):
    from dispatch_voice.host import setup
    monkeypatch.setattr(chatterbox, "available", lambda: False)
    monkeypatch.setenv("DISPATCH_VOICE", "1")
    monkeypatch.setenv("DISPATCH_VOICE_MERGE_WINDOW", "0")
    # The acknowledgement lines render on a background thread through the same fake engine
    # as soon as a session builds its kit, before the session counts as "a chat is open";
    # on a slow runner all of them landed in `said` and broke "the CPU engine said nothing".
    monkeypatch.setenv("DISPATCH_VOICE_ACK_RENDER", "0")

    async def fake_warm(settings):
        return fake_engines.stt, fake_engines.tts

    monkeypatch.setattr(routes, "warm_models", fake_warm)
    monkeypatch.setattr(vad_turn, "build_detector", lambda s: fake_engines.OneTurnDetector())
    monkeypatch.setattr(routes, "voice_for_bot", lambda s, e, b: None)
    holds: list[str] = []
    monkeypatch.setattr(power, "acquire", lambda settings: holds.append("acquire") or "held")
    monkeypatch.setattr(power, "release", lambda: holds.append("release") or "off")
    npz = tmp_path / "voice.npz"
    npz.write_bytes(b"conditionals")
    monkeypatch.setattr(rig, "voice_file", lambda pid: npz)
    monkeypatch.setattr(rig, "HTTP_TRANSPORT", httpx.MockTransport(lambda r: httpx.Response(200)))
    setup(host_env)
    return host_env, holds, npz


def _session():
    live = [s for s in ws_mod._SESSIONS.values() if hasattr(s, "tts")]
    return live[-1] if live else None


async def test_no_rig_configured_is_byte_for_byte_the_cpu_session(served, fake_engines, fake_rig, token):
    ctx, holds, _ = served
    vconfig.save_prefs({"rig": {"enabled": False, "url": fake_rig.url}})     # set up, switched off
    ws = fake_engines.FakeWS()
    task = asyncio.create_task(ctx.connect("voice", THREAD, ws))
    assert await wait_until(lambda: _session() is not None, 10.0)    # a cold first connect is slow
    s = _session()
    assert s.tts is fake_engines.tts and s.rig is None and s.heard.words_of is None
    assert fake_rig.headers == []                                           # never dialled
    assert holds == ["acquire"]                                             # today's hold
    ws.hang_up()
    await asyncio.wait_for(task, 5)


async def test_with_the_rig_the_reply_is_spoken_by_it_and_no_hold_is_taken(served, fake_engines, fake_rig, token,
                                                                             monkeypatch):
    ctx, holds, npz = served
    from dispatch_voice.host import session as session_mod
    from dispatch_voice.rt.stt_rig import RigStreamingSTT
    from dispatch_voice.rt.stt_stream import ParakeetTailSTT
    seen: dict = {}
    orig = session_mod.VoiceSession.__init__

    def init(self, *a, **kw):
        seen.update(kw)
        orig(self, *a, **kw)

    monkeypatch.setattr(session_mod.VoiceSession, "__init__", init)
    vconfig.save_prefs({"rig": {"enabled": True, "url": fake_rig.url}})
    ctx.reply_for = lambda text: "Two meetings today. Want details?"
    ws = fake_engines.FakeWS()
    task = asyncio.create_task(ctx.connect("voice", THREAD, ws))
    assert await wait_until(lambda: _session() is not None, 10.0)    # a cold first connect is slow
    s = _session()
    assert isinstance(s.tts, ReplyEngine) and s.rig is s.tts and s.heard.words_of is not None
    # The streaming STT handed to the session is the rig's, with the CPU recogniser behind it.
    assert isinstance(seen["sstt"], RigStreamingSTT) and isinstance(seen["sstt"].cpu, ParakeetTailSTT)
    conn = s.tts.conn
    sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    assert await wait_for(lambda: conn.usable() and s.tts.rig.voice_ref(None) == f"sha256:{sha}")
    ws.push_json({"type": "start"})
    ws.push_audio()
    await settle(40)
    await ctx.drain()
    assert await wait_for(lambda: ws.json_of("say_end"), 3.0)
    texts = [m["text"] for m in fake_rig.of("tts.text")]
    assert texts and texts[0].startswith("Two meetings today")
    pcm = [np.frombuffer(m, "<i2") for m in ws.sent if isinstance(m, bytes)]
    assert pcm and any((p == fake_rig.tts_value).all() for p in pcm if len(p))
    assert fake_engines.tts.said == []                  # the CPU engine said nothing
    assert holds == []                                  # the rig serves: no performance hold
    ws.hang_up()
    await asyncio.wait_for(task, 5)
    assert await wait_for(lambda: fake_rig.of("bye"))
    assert fake_engines.tts.said == []
    st = rig.status(vconfig.load_settings())
    assert TOKEN not in repr(st) and st["host"] == f"127.0.0.1:{fake_rig.port}"


async def test_a_refused_rig_runs_the_cpu_path_and_takes_the_hold(served, fake_engines, fake_rig, token):
    ctx, holds, _ = served
    fake_rig.refuse = (503, {"Retry-After": "30"})
    vconfig.save_prefs({"rig": {"enabled": True, "url": fake_rig.url}})
    ctx.reply_for = lambda text: "Hello."
    ws = fake_engines.FakeWS()
    task = asyncio.create_task(ctx.connect("voice", THREAD, ws))
    assert await wait_for(lambda: holds == ["acquire"], 3.0)
    ws.push_json({"type": "start"})
    ws.push_audio()
    await settle(40)
    await ctx.drain()
    assert await wait_for(lambda: fake_engines.tts.said, 3.0)
    assert fake_engines.tts.said[0].startswith("Hello") and fake_rig.of("tts.open") == []
    ws.hang_up()
    await asyncio.wait_for(task, 5)
    assert holds == ["acquire", "release"]


async def test_barge_in_cancels_on_the_rig_before_the_host_abort(served, fake_engines, fake_rig, token):
    ctx, holds, _ = served
    vconfig.save_prefs({"rig": {"enabled": True, "url": fake_rig.url}})
    ws = fake_engines.FakeWS()
    task = asyncio.create_task(ctx.connect("voice", THREAD, ws))
    assert await wait_until(lambda: _session() is not None, 10.0)    # a cold first connect is slow
    s = _session()
    order: list = []
    real_cancel = s.rig.cancel_gen
    s.rig.cancel_gen = lambda g: (order.append(("cancel", g)), real_cancel(g))

    async def abort(thread_id, keep_partial=False):
        order.append(("abort", keep_partial))
        return None

    s.host.abort = abort
    gen0 = s.gen
    await s.barge_in(by_user_voice=True, abort=True)
    assert order == [("cancel", gen0), ("abort", False)]
    assert await wait_for(lambda: {"type": "cancel", "gen": gen0} in fake_rig.of("cancel"))
    # The duck pauses the rig's context; the false-interruption resume resumes it.
    calls = []
    s.rig.pause, s.rig.resume = (lambda: calls.append("pause")), (lambda: calls.append("resume"))
    await s._set_duck(True)
    await s._set_duck(False)
    assert calls == ["pause", "resume"]
    ws.hang_up()
    await asyncio.wait_for(task, 5)
