"""RigConnection against the fake sf.1 rig: hello/ready, the proof, refusals, the
sub-second down rules, the 30 s rule, resume, unknown-frame tolerance, bye."""
from __future__ import annotations

import asyncio
import time

import pytest

from forge_voice_core import rig

from .conftest import RecordingPower, make_conn, proven, wait_for
from .fake_sf1 import GOLDEN


async def test_hello_ready_and_the_first_pong_prove_the_rig(fake_rig):
    conn = make_conn(fake_rig)
    conn.start()
    try:
        assert not conn.usable()                       # until ready + a pong: the CPU
        assert await proven(conn)
        hello = fake_rig.received[0]
        assert hello["type"] == "hello" and hello["proto"] == "sf.1" and "resume" not in hello
        assert hello["client"] == {"name": "forge-voice-core", "version": "test"}      # the standalone host link; DisPatch says dispatch-voice
        h = fake_rig.headers[0]
        assert h["Authorization"] == f"Bearer {fake_rig.token}"
        assert h["Sec-WebSocket-Protocol"] == "sf.1" and "Origin" not in h
        assert "permessage-deflate" not in (h.get("Sec-WebSocket-Extensions") or "")
        cfg = fake_rig.of("stt.config")[0]
        assert cfg["partial_ms"] == 160 and cfg["eou"] == {"enabled": True, "min_pause_ms": 150}
        s = conn.summary()
        assert s["state"] == "ready" and s["gpu"]["name"] == "NVIDIA GeForce RTX 3090"
        assert s["engines"]["tts"]["sha8"] == "5e6f7a8b" and s["limits"]["contexts"] == 2
        await asyncio.sleep(0.6)
        pings = fake_rig.of("ping")
        assert len(pings) >= 2 and all("box_load" in p for p in pings)   # every 250 ms
        assert fake_rig.errors == []
    finally:
        await conn.close()
    assert fake_rig.of("bye") and fake_rig.of("bye")[0]["reason"] == "stop"


async def test_a_401_is_misconfigured_and_never_retried(fake_rig):
    wrong = "wrong-token-not-real"  # scrub-ok: test fixture
    power = RecordingPower()
    conn = make_conn(fake_rig, power=power, token=wrong)
    conn.start()
    assert await wait_for(lambda: conn.state == "misconfigured")
    assert conn.reason == "token" and not conn.usable()
    attempts = len(fake_rig.headers)
    conn._probe()                                     # a between-turn probe does nothing
    await asyncio.sleep(0.2)
    assert len(fake_rig.headers) == attempts
    await wait_for(lambda: power.calls)
    assert power.calls == ["acquire"]                 # the CPU serves: the hold is taken
    await conn.close()
    # A new voice chat with the same address and token does not dial again.
    again = make_conn(fake_rig, token=wrong)
    again.start()
    assert again.state == "misconfigured" and len(fake_rig.headers) == attempts
    await again.close()


@pytest.mark.parametrize("status,headers,reason,retry", [
    (503, {"Retry-After": "7"}, "busy", 7.0),
    (429, {}, "rate_limited", rig.RATE_LIMIT_BACKOFF_S),
])
async def test_refusals_retry_after_retry_after_and_only_between_turns(fake_rig, status, headers, reason, retry):
    fake_rig.refuse = (status, headers)
    conn = make_conn(fake_rig)
    conn.start()
    assert await wait_for(lambda: conn.state == "down")
    assert conn.reason == reason and not conn.usable()
    assert conn.retry_at - conn.clock() == pytest.approx(retry, abs=0.5)
    n = len(fake_rig.headers)
    conn._probe()                                     # before Retry-After: no dial
    await asyncio.sleep(0.1)
    assert len(fake_rig.headers) == n
    fake_rig.refuse = None
    conn.retry_at = conn.clock() - 1                  # Retry-After passed
    conn._probe()
    assert await proven(conn)
    await conn.close()


async def test_not_ready_error_after_hello_is_a_503(fake_rig):
    fake_rig.ready_error = dict(GOLDEN["server_frames"]["error_not_ready"])
    conn = make_conn(fake_rig)
    conn.start()
    assert await wait_for(lambda: conn.state == "down")
    assert conn.reason == "not_ready" and conn.retry_at - conn.clock() == pytest.approx(20, abs=0.5)
    assert conn.summary()["state"] == "busy"
    await conn.close()


async def test_three_missed_pongs_are_down_within_a_second(fake_rig):
    conn = make_conn(fake_rig)
    conn.start()
    assert await proven(conn)
    fake_rig.chatter(0.1)                             # frames keep coming: only pongs stop
    fake_rig.answer_pings = False
    t0 = time.monotonic()
    epoch = conn.epoch
    assert await wait_for(lambda: conn.epoch == epoch + 1, 2.0, 0.005)
    dt = time.monotonic() - t0
    assert rig.METRICS.last_fallback_reason == "missed_pongs" and dt < 1.0, dt
    assert not conn.usable()
    await conn.close()


async def test_one_second_of_silence_is_down(fake_rig):
    conn = make_conn(fake_rig)
    conn.start()
    assert await proven(conn)
    fake_rig.silent = True
    t0 = time.monotonic()
    assert await wait_for(lambda: conn.epoch == 1, 2.0, 0.005)
    dt = time.monotonic() - t0
    # Missed pongs (3 x 250 ms) usually fire first; silence is the 1.0 s backstop.
    assert rig.METRICS.last_fallback_reason in ("missed_pongs", "silence") and dt <= 1.1, dt
    assert not conn.usable()
    await conn.close()


async def test_a_dead_socket_is_down_at_once_and_reattached_with_resume(fake_rig):
    conn = make_conn(fake_rig)
    conn.start()
    assert await proven(conn)
    t0 = time.monotonic()
    await fake_rig.kill()
    assert await wait_for(lambda: conn.epoch == 1, 1.0, 0.005)
    assert time.monotonic() - t0 < 1.0
    # Re-attached in the background with hello{resume}; the same session comes back ...
    assert await wait_for(lambda: fake_rig.resumed == 1 and conn.state == "ready", 3.0)
    hello = [m for m in fake_rig.received if m["type"] == "hello"][-1]
    assert hello["resume"] == {"session": "sess-1", "token": GOLDEN["server_frames"]["ready"]["resume_token"]}
    # ... but no turn starts on it: the 30 s rule, and then a between-turn probe.
    await asyncio.sleep(0.4)
    assert not conn.usable() and conn.summary()["state"] == "cooling"
    await conn.close()


async def test_re_promotion_needs_the_cooldown_and_a_probe(fake_rig):
    power = RecordingPower()
    conn = make_conn(fake_rig, power=power)
    conn.cooldown_s = 0.5
    conn.start()
    assert await proven(conn)
    conn.fail("tts_late", move=False)                 # a failure that keeps the socket
    assert power.calls == [] or power.calls[-1] == "acquire"
    await asyncio.sleep(0.3)
    conn._probe()                                     # inside the cooldown: nothing
    await asyncio.sleep(0.1)
    assert not conn.usable()
    await asyncio.sleep(0.4)
    assert not conn.usable()                          # pongs alone never re-promote
    conn._probe()                                     # between turns, after the cooldown
    assert await proven(conn, 1.0)
    await wait_for(lambda: power.calls[-1:] == ["release"])
    assert power.calls == ["acquire", "release"]
    await conn.close()


async def test_partial_lag_over_800_ms_for_a_second_while_speaking_is_down(fake_rig):
    conn = make_conn(fake_rig)
    conn.start()
    assert await proven(conn)
    lagging = dict(GOLDEN["server_frames"]["stt.partial"], lag_ms=950)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.6 and conn.epoch == 0:
        conn.send_mic(__import__("numpy").zeros(320, "<i2"))     # speech is flowing
        await fake_rig.send(lagging)
        await asyncio.sleep(0.1)
    assert conn.epoch == 1 and conn.reason == "lag" and not conn.usable()
    assert 0.9 <= time.monotonic() - t0 <= 1.4
    await conn.close()


async def test_lease_lost_is_down_and_vacating_stops_new_turns(fake_rig):
    conn = make_conn(fake_rig)
    conn.start()
    assert await proven(conn)
    await fake_rig.send(GOLDEN["server_frames"]["state_vacating"])
    assert await wait_for(lambda: not conn.usable())
    assert conn.state == "ready" and conn.epoch == 0  # in-flight work stays on the rig
    assert conn.summary()["state"] == "busy"
    await fake_rig.send(GOLDEN["server_frames"]["state_lost"])
    assert await wait_for(lambda: conn.epoch == 1)
    assert conn.reason == "lease_lost"
    await conn.close()


async def test_unknown_and_malformed_frames_are_tolerated(fake_rig):
    conn = make_conn(fake_rig)
    conn.start()
    assert await proven(conn)
    await fake_rig.send({"type": "something.new", "x": 1})
    await fake_rig.send("not json")
    await fake_rig.send(b"\x00\x01garbage")
    await fake_rig.send({"type": "voice.list", "ids": []})
    await asyncio.sleep(0.3)
    assert conn.usable() and conn.epoch == 0
    await conn.close()


async def test_not_set_up_means_no_rig_at_all(monkeypatch, tmp_path):
    from forge_voice_core.engine_settings import EngineSettings as VoiceSettings
    s = VoiceSettings()
    assert rig.open_for_session(s) is None                       # switch off by default
    from dataclasses import replace
    on = replace(s, rig=rig.RigSettings(enabled=True, url="ws://127.0.0.1:19109/v1/realtime"))
    assert rig.open_for_session(on) is None                      # no token
    bad = replace(s, rig=rig.RigSettings(enabled=True, url="http://x/"))
    monkeypatch.setenv(rig.TOKEN_ENV, "t")
    assert rig.open_for_session(bad) is None                     # not a ws:// address


async def test_silence_alone_is_down_at_one_second(fake_rig, monkeypatch):
    """The 1.0 s backstop on its own (the pong rule out of the way)."""
    monkeypatch.setattr(rig, "MISSED_PONGS", 1000)
    conn = make_conn(fake_rig)
    conn.start()
    assert await proven(conn)
    fake_rig.silent = True
    t0 = time.monotonic()
    assert await wait_for(lambda: conn.epoch == 1, 2.0, 0.005)
    dt = time.monotonic() - t0
    assert rig.METRICS.last_fallback_reason == "silence" and 0.9 <= dt <= 1.15, dt
    await conn.close()
