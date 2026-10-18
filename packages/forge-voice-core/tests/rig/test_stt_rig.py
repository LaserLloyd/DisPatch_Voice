"""RigStreamingSTT: the utterance on the rig (mic frames, partials, eou, final), the
failovers (rig killed mid-utterance, a late final) with no lost or doubled words,
and the "never slower" property when the rig is not proven."""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from forge_voice_core import rig
from forge_voice_core.rt.stt_rig import RigStreamingSTT

from .conftest import FakeCpuSTT, frames, make_conn, proven, wait_for
from .fake_sf1 import GOLDEN


@pytest.fixture
async def live(fake_rig):
    conn = make_conn(fake_rig)
    conn.start()
    assert await proven(conn)
    cpu = FakeCpuSTT()
    stt = RigStreamingSTT(cpu, conn)
    yield fake_rig, conn, cpu, stt
    await conn.close()


async def _push(stt, fs, prob=0.9):
    for f in fs:
        await asyncio.to_thread(stt.push, f, vad_prob=prob)


async def test_an_utterance_on_the_rig_keeps_the_cpu_idle(live):
    fake, conn, cpu, stt = live
    await _push(stt, frames(20))                      # 20 x 512 = 10240 samples
    assert stt.mode == "rig" and cpu.pushed == []
    text = await asyncio.to_thread(stt.final, timeout_s=1.2)
    assert text == "what is the weather" and cpu.calls == []
    # Every sample reached the rig exactly once, in 20 ms frames on one clock.
    assert fake.mic_samples() == 10240
    pts = [p for p, _ in fake.mic]
    assert pts == sorted(pts) and all(b - a == 320 for a, b in zip(pts, pts[1:], strict=False))
    assert all(n == 320 for _, n in fake.mic[:-1])
    flush = fake.of("stt.flush")[0]
    assert isinstance(flush["id"], int)
    await asyncio.to_thread(stt.reset)
    assert await wait_for(lambda: fake.of("stt.reset"))
    assert rig.METRICS.snapshot()["stt"] == 1 and rig.METRICS.snapshot()["turns"] == 1
    assert fake.errors == []


async def test_at_pause_returns_the_partial_covering_the_pause_and_tick_only_new_ones(live):
    fake, conn, cpu, stt = live
    await _push(stt, frames(10))
    end = conn.mic_pts
    await fake.send(dict(GOLDEN["server_frames"]["stt.partial"], tail="what is", upto=end - 3000))
    t0 = time.monotonic()
    p_task = asyncio.ensure_future(asyncio.to_thread(stt.at_pause))
    await asyncio.sleep(0.05)
    await fake.send(dict(GOLDEN["server_frames"]["stt.partial"], committed="", tail="what is the weather", upto=end))
    p = await p_task
    assert p.tail == "what is the weather" and time.monotonic() - t0 < 0.25
    assert p.t_audio == pytest.approx(end / 16000, abs=0.01)
    assert stt.tick() is None                         # nothing newer than what at_pause gave
    await fake.send(dict(GOLDEN["server_frames"]["stt.partial"], tail="what is the weather today", upto=end + 1))
    await asyncio.sleep(0.05)
    stt.tick_interval_s = 0.0
    t = stt.tick()
    assert t is not None and t.tail.endswith("today") and stt.tick() is None
    assert cpu.calls == []


async def test_the_rigs_eou_replaces_the_local_smart_turn(live):
    fake, conn, cpu, stt = live
    local_calls = []

    def local(audio):
        local_calls.append(1)
        return 0.11

    await _push(stt, frames(8))
    speech_end = conn.mic_pts
    await _push(stt, frames(3), prob=0.05)            # the pause
    await fake.send(dict(GOLDEN["server_frames"]["eou"], p=0.97, at=speech_end))
    score = stt.turn_fn_for(local)
    assert await asyncio.to_thread(score, None) == pytest.approx(0.97) and local_calls == []
    # A re-probe of the same pause (the rig scores once per pause): local, at once.
    t0 = time.monotonic()
    assert await asyncio.to_thread(score, None) == pytest.approx(0.11)
    assert local_calls == [1] and time.monotonic() - t0 < 0.1


async def test_no_eou_in_time_falls_back_to_local_smart_turn(live):
    fake, conn, cpu, stt = live
    await _push(stt, frames(4))
    t0 = time.monotonic()
    assert await asyncio.to_thread(stt.turn_fn_for(lambda a: 0.4), None) == 0.4
    assert time.monotonic() - t0 < 0.4
    assert await asyncio.to_thread(stt.turn_fn_for(None), None) is None


async def test_rig_killed_mid_utterance_replays_the_ring_into_the_cpu(live):
    fake, conn, cpu, stt = live
    fs = frames(40)                                   # 1.28 s, distinct values
    await _push(stt, fs[:25])
    t_kill = time.monotonic()
    await fake.kill()
    assert await wait_for(lambda: conn.epoch == 1, 1.0, 0.005)
    await _push(stt, fs[25:])                         # the next frame moves the utterance
    assert stt.mode == "cpu" and time.monotonic() - t_kill < 1.0
    heard = cpu.samples()
    # No lost and no doubled words: the CPU heard every frame exactly once, in order.
    assert np.array_equal(heard, np.concatenate(fs))
    assert await asyncio.to_thread(stt.final, timeout_s=1.2) == "cpu words"
    assert rig.METRICS.snapshot()["stt"] == 0 and rig.METRICS.snapshot()["fell_back"] >= 1


async def test_a_late_final_is_300_ms_then_the_cpu_tail_and_the_rig_is_down(live):
    fake, conn, cpu, stt = live
    fake.drop_finals = True
    fs = frames(15)
    await _push(stt, fs)
    t0 = time.monotonic()
    text = await asyncio.to_thread(stt.final, timeout_s=1.2)
    dt = time.monotonic() - t0
    assert text == "cpu words" and 0.28 <= dt < 0.5, dt
    assert np.array_equal(cpu.samples(), np.concatenate(fs))   # the CPU heard the whole ring
    assert not conn.usable() and rig.METRICS.last_fallback_reason == "final_timeout"
    await asyncio.to_thread(stt.reset)
    await _push(stt, frames(2))
    assert stt.mode == "cpu"                          # the next turn does not start on it


async def test_never_slower_an_unproven_rig_is_the_cpu_path_with_no_waits(fake_rig):
    fake_rig.refuse = (503, {"Retry-After": "30"})
    conn = make_conn(fake_rig)
    conn.start()
    assert await wait_for(lambda: conn.state == "down")
    cpu = FakeCpuSTT()
    stt = RigStreamingSTT(cpu, conn)
    ref = FakeCpuSTT()                                # what the CPU-only session would do
    fs = frames(12)
    timeline, ref_timeline = [], []
    t0 = time.monotonic()
    for f in fs:
        stt.push(f, vad_prob=0.9)
        ref.push(f, vad_prob=0.9)
    timeline.append(("at_pause", stt.at_pause().text))
    ref_timeline.append(("at_pause", ref.at_pause().text))
    timeline.append(("eou", stt.turn_fn_for(lambda a: 0.8)(None)))
    ref_timeline.append(("eou", 0.8))
    timeline.append(("tick", stt.tick()))
    ref_timeline.append(("tick", ref.tick()))
    timeline.append(("final", stt.final(timeout_s=1.2)))
    ref_timeline.append(("final", ref.final(timeout_s=1.2)))
    dt = time.monotonic() - t0
    assert timeline == ref_timeline and cpu.calls == ref.calls
    assert np.array_equal(cpu.samples(), ref.samples())
    assert dt < 0.05, dt                              # no deadline waited anywhere
    assert fake_rig.mic == [] and not fake_rig.of("stt.flush")
    await conn.close()


async def test_a_turn_never_starts_on_the_rig_mid_utterance(fake_rig):
    """The rig becoming ready while an utterance runs on the CPU does not move it."""
    fake_rig.ready_delay = 0.3
    conn = make_conn(fake_rig)
    conn.start()
    cpu = FakeCpuSTT()
    stt = RigStreamingSTT(cpu, conn)
    stt.push(frames(1)[0], vad_prob=0.9)
    assert stt.mode == "cpu"
    assert await proven(conn)
    await _push(stt, frames(5))
    assert stt.mode == "cpu" and fake_rig.mic == [] and len(cpu.pushed) == 6
    await asyncio.to_thread(stt.reset)
    await _push(stt, frames(2))
    assert stt.mode == "rig"                          # the NEXT utterance uses it
    await conn.close()


async def test_a_new_rig_session_drops_the_old_sessions_transcripts(live):
    conn = live[1]
    old = conn.session_id
    conn.partials.append((999, {"upto": 160000, "committed": "stale words"}))
    conn.eous.append((999, {"p": 0.9, "at": 160000}))
    conn.finals["stale"] = {"text": "stale"}
    conn._on_ready(dict(conn.ready_info or {}, session=f"{old}-new"))
    assert not conn.partials and not conn.eous and not conn.finals
