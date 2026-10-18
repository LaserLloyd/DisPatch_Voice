"""Rig states the add-on must cope with as SpeechForge sends them: an unleased rig,
models still loading, a lease being vacated, a 503 with Retry-After before the
upgrade; per-context engine errors; the power hold never released mid-CPU-work."""
from __future__ import annotations

import asyncio

import pytest

from forge_voice_core import rig
from forge_voice_core.engines.rig_tts import ReplyEngine
from forge_voice_core.rt import stt_rig

from .conftest import FakeCpuTTS, RecordingPower, make_conn, proven, wait_for

VOICE = "sha256:" + "ef" * 32


def engine(conn, fake):
    sess = rig.RigSession(conn, rig.VoiceSync(fake.url, fake.token), rig.RigPowerGate(None))
    sess.voices[None] = VOICE
    cpu = FakeCpuTTS()
    return cpu, ReplyEngine(cpu, sess)


def say(eng, text, *, gen=0, seq=1, rid="r1"):
    return {int(f[0]) for f in eng.stream_reply(text, None, None, gen=gen, seq=seq, rid=rid)}


async def test_an_unleased_rig_serves_and_says_so(fake_rig):
    fake_rig.lease_state = "unleased"
    conn = make_conn(fake_rig)
    conn.start()
    assert await proven(conn)
    s = conn.summary()
    assert s["state"] == "ready" and s["lease"] == "unleased"
    cpu, eng = engine(conn, fake_rig)
    assert await asyncio.to_thread(say, eng, "Hello.") == {fake_rig.tts_value} and cpu.said == []
    await conn.close()


async def test_models_loading_stays_on_the_cpu_until_ready(fake_rig):
    fake_rig.loading_first = True
    fake_rig.ready_delay = 0.6
    conn = make_conn(fake_rig)
    conn.start()
    assert await wait_for(lambda: conn.models_loading)
    assert not conn.usable() and conn.state == "connecting"
    cpu, eng = engine(conn, fake_rig)
    assert await asyncio.to_thread(say, eng, "Early.", rid="early") == {FakeCpuTTS.VALUE}
    assert await proven(conn)
    assert not conn.models_loading
    assert await asyncio.to_thread(say, eng, "Later.", rid="later") == {fake_rig.tts_value}
    # A live session told the models are loading again: no new turn on the rig.
    await fake_rig.send({"type": "state", "lease": "held", "models": "loading"})
    assert await wait_for(lambda: not conn.usable())
    assert conn.summary()["state"] == "loading"
    await fake_rig.send({"type": "state", "lease": "held", "models": "ready"})
    assert await wait_for(conn.usable)
    await conn.close()


async def test_vacating_finishes_the_current_reply_and_opens_nothing_new(fake_rig):
    conn = make_conn(fake_rig)
    conn.start()
    assert await proven(conn)
    cpu, eng = engine(conn, fake_rig)
    assert await asyncio.to_thread(say, eng, "First.", seq=1) == {fake_rig.tts_value}
    await fake_rig.send({"type": "state", "lease": "vacating", "deadline_s": 45})
    assert await wait_for(lambda: conn.vacating)
    # The reply in progress keeps its context on the rig ...
    assert await asyncio.to_thread(say, eng, "Second.", seq=2) == {fake_rig.tts_value}
    # ... a new reply is the CPU's, no context is opened, the rig waits out the deadline.
    assert await asyncio.to_thread(say, eng, "New reply.", seq=3, rid="r2") == {FakeCpuTTS.VALUE}
    assert len(fake_rig.of("tts.open")) == 1
    assert conn.open_ctx(gen=1, voice=VOICE, seed=1) is None
    assert not conn.usable() and conn.down_until - conn.clock() > 40
    assert conn.summary()["state"] == "busy"
    # The rig unloads: a fatal `vacated` with Retry-After -> down, no reconnect before it.
    await fake_rig.send({"type": "error", "code": "vacated", "fatal": True, "retry_after_s": 120,
                         "message": "unloaded"})
    assert await wait_for(lambda: conn.state == "down")
    assert conn.reason == "vacating" and conn.down_until - conn.clock() > 100
    await conn.close()


async def test_a_503_with_retry_after_before_the_upgrade(fake_rig):
    fake_rig.refuse = (503, {"Retry-After": "9"})
    conn = make_conn(fake_rig)
    conn.start()
    assert await wait_for(lambda: conn.state == "down")
    assert conn.reason == "busy" and 8 <= conn.retry_at - conn.clock() <= 9.5
    assert conn.summary()["state"] == "busy"
    n = len(fake_rig.headers)
    conn._probe()
    await asyncio.sleep(0.1)
    assert len(fake_rig.headers) == n                       # not before Retry-After
    await conn.close()


@pytest.mark.parametrize("code,reason", [("engine_failed", "tts_failed"), ("not_ready", "not_ready")])
async def test_an_engine_error_is_a_rig_failure_not_just_a_cpu_reply(fake_rig, code, reason):
    conn = make_conn(fake_rig)
    conn.start()
    assert await proven(conn)
    cpu, eng = engine(conn, fake_rig)
    fake_rig.tts_errors = [code]
    assert await asyncio.to_thread(say, eng, "Hello.") == {FakeCpuTTS.VALUE}
    assert not conn.usable() and conn.reason == reason
    # The next reply does not pay the round trip again: CPU, no tts.open.
    assert await asyncio.to_thread(say, eng, "Next.", rid="r2") == {FakeCpuTTS.VALUE}
    assert len(fake_rig.of("tts.open")) == 1
    await conn.close()


async def test_busy_twice_in_a_row_cools_the_rig_once_is_tolerated(fake_rig):
    conn = make_conn(fake_rig)
    conn.start()
    assert await proven(conn)
    cpu, eng = engine(conn, fake_rig)
    fake_rig.tts_errors = ["busy"]
    assert await asyncio.to_thread(say, eng, "One.", rid="a") == {FakeCpuTTS.VALUE}
    assert conn.usable()                                    # once: that reply only
    fake_rig.tts_errors = ["busy"]
    assert await asyncio.to_thread(say, eng, "Two.", rid="b") == {FakeCpuTTS.VALUE}
    assert not conn.usable() and conn.reason == "busy"
    await conn.close()


async def test_the_proving_pong_never_releases_the_hold_mid_cpu_work(fake_rig):
    power = RecordingPower()
    conn = make_conn(fake_rig, power=power)
    busy = [True]
    conn.busy_check = lambda: busy[0]
    conn.power.need()                                       # a CPU turn is running
    await wait_for(lambda: power.calls == ["acquire"])
    conn.start()
    assert await proven(conn)
    await asyncio.sleep(0.3)
    assert power.calls == ["acquire"] and conn.power.held   # proven, but CPU work in flight
    busy[0] = False
    assert await wait_for(lambda: power.calls == ["acquire", "release"])
    await conn.close()


def test_the_pause_wait_for_the_rigs_eou_is_short():
    assert stt_rig.PAUSE_WAIT_S <= 0.12
