"""Credit and context limits against a fake rig that gates like SpeechForge.

The rig freezes ``played_ms`` at the last ``credit`` and counts produced audio in
40 ms tokens (lookahead included); a unit's first audio waits for a whole chunk. The
box must (a) keep the rig's copy fresh (every 250 ms while a context is open) and
(b) never run the 400 ms late clock while the rig, by that rule, may be holding back.
Mirrors the rig-compat review's S5x(b)/(d): a 4 s first sentence, a phone that
reports once per second at (or behind) real time -> the reply stays on the rig.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from dispatch_voice.host import rig
from dispatch_voice.host.engines.rig_tts import ReplyEngine

from .conftest import FakeCpuTTS, make_conn, proven, wait_for

VOICE = "sha256:" + "cd" * 32


@pytest.fixture
async def live(fake_rig):
    conn = make_conn(fake_rig, tts_start_deadline_ms=400)       # the late rule at 400 ms here
    conn.start()
    assert await proven(conn)
    sess = rig.RigSession(conn, rig.VoiceSync(fake_rig.url, fake_rig.token), rig.RigPowerGate(None))
    sess.voices[None] = VOICE
    cpu = FakeCpuTTS()
    eng = ReplyEngine(cpu, sess)
    yield fake_rig, conn, cpu, eng
    await conn.close()


def speak_with_phone(eng, sentences, *, frames_per, fake, gen=0, rid="r1", lag_s=0.0, every_s=1.0):
    """The session's TTS thread speaks `sentences` back to back while a phone plays
    the audio in real time (``lag_s`` behind) and sends ``heard`` every ``every_s``.
    Returns (values per sentence, the phone's reports)."""
    got: list[list[int]] = []
    recv_ms: list[float] = []                 # audio received per sentence (ms)
    lock = threading.Lock()
    t_first: list[float] = []
    done = threading.Event()
    reports: list[tuple[int, int]] = []

    def phone():
        k = 1
        while not done.is_set():
            if not t_first:
                time.sleep(0.005)
                continue
            due = t_first[0] + k * every_s
            if done.wait(max(0.0, due - time.monotonic())):
                return
            k += 1
            with lock:
                have = list(recv_ms)
            pos = min(max(0.0, (time.monotonic() - t_first[0] - lag_s) * 1000.0), sum(have))
            start = 0.0
            for i, ms in enumerate(have):
                if pos <= start + ms or i == len(have) - 1:
                    reports.append((i + 1, int(pos - start)))
                    eng.heard(gen, i + 1, int(pos - start))
                    break
                start += ms

    th = threading.Thread(target=phone, daemon=True)
    th.start()
    try:
        for i, text in enumerate(sentences):
            fake.tts_frames = frames_per[i]
            vals: list[int] = []
            with lock:
                recv_ms.append(0.0)
            for f in eng.stream_reply(text, None, None, gen=gen, seq=i + 1, rid=rid):
                if not t_first:
                    t_first.append(time.monotonic())
                vals.append(int(f[0]))
                with lock:
                    recv_ms[-1] += len(f) / 24.0
            got.append(vals)
    finally:
        done.set()
        th.join(2)
    return got, reports


@pytest.mark.parametrize("lag_s", [0.0, 0.5], ids=["phone-at-real-time", "phone-half-a-second-behind"])
async def test_a_4s_first_sentence_heard_once_a_second_stays_on_the_rig(live, lag_s):
    fake, conn, cpu, eng = live
    got, reports = await asyncio.to_thread(
        speak_with_phone, eng, ["A first sentence that runs for four whole seconds.", "Then a second.",
                                "And a third one."],
        frames_per=[100, 30, 30], fake=fake, lag_s=lag_s)
    assert all(v == [fake.tts_value] * len(v) and v for v in got), [sorted(set(v)) for v in got]
    assert cpu.said == [] and rig.METRICS.last_fallback_reason is None
    assert conn.usable()                                       # no cooldown either
    ctx = fake.of("tts.open")[0]["ctx"]
    stalls = fake.ctxs[ctx].stalls if ctx in fake.ctxs else None
    assert stalls is None or stalls >= 1                       # the credit really held it back
    credits = [c for c in fake.of("credit") if "played_ms" in c]
    assert len(credits) >= 3 * len(reports) - 2                 # kept fresh between reports
    played = [c["played_ms"] for c in credits]
    assert played[-1] >= played[0]


async def test_credit_is_refreshed_every_250_ms_and_frozen_by_a_pause(live):
    fake, conn, cpu, eng = live
    fake.tts_frames = 60                                        # 2.4 s
    await asyncio.to_thread(lambda: list(eng.stream_reply("Long enough.", None, None, gen=2, seq=1, rid="r")))
    eng.heard(2, 1, 100)
    await asyncio.sleep(0.62)
    credits = [c for c in fake.of("credit") if "played_ms" in c]
    assert 3 <= len(credits) <= 4, credits
    assert credits[0]["played_ms"] == 100
    assert all(b["played_ms"] > a["played_ms"] for a, b in zip(credits, credits[1:]))
    assert credits[-1]["played_ms"] <= 100 + 650
    eng.pause()
    await asyncio.sleep(0.05)
    frozen = [c for c in fake.of("credit") if "played_ms" in c][-1]["played_ms"]
    await asyncio.sleep(0.6)
    paused = [c["played_ms"] for c in fake.of("credit") if "played_ms" in c]
    assert all(v == frozen for v in paused[paused.index(frozen):])      # a pause freezes playback
    eng.resume()
    # The next 250 ms credit tick carries the advanced clock: wait for it, not for a fixed time.
    assert await wait_for(lambda: any(c.get("played_ms", 0) > frozen for c in fake.of("credit")), timeout=3)
    after = [c["played_ms"] for c in fake.of("credit") if "played_ms" in c]
    assert after[-1] > frozen
    # Never past the audio the phone has.
    await asyncio.sleep(2.5)
    assert [c for c in fake.of("credit") if "played_ms" in c][-1]["played_ms"] <= 2400
    eng.cancel_gen(2)
    await wait_for(lambda: fake.of("cancel"))
    n = len(fake.of("credit"))
    await asyncio.sleep(0.4)
    assert len(fake.of("credit")) == n                          # a cancelled ctx gets no credit


async def test_a_rig_that_could_produce_and_does_not_is_still_late(live):
    """The credit rule must not hide a stuck rig: short first sentence, credit wide open,
    the second sentence never starts -> CPU after ~400 ms, rig cooled down."""
    fake, conn, cpu, eng = live
    await asyncio.to_thread(lambda: list(eng.stream_reply("Hi.", None, None, gen=0, seq=1, rid="r")))
    fake.tts_delays = [5.0]
    t0 = time.monotonic()
    out = await asyncio.to_thread(lambda: list(eng.stream_reply("Stuck.", None, None, gen=0, seq=2, rid="r")))
    assert {int(f[0]) for f in out} == {FakeCpuTTS.VALUE} and 0.38 <= time.monotonic() - t0 < 0.8
    assert rig.METRICS.last_fallback_reason == "tts_late" and not conn.usable()


def test_rig_may_produce_mirrors_the_rig_rule():
    c = rig.TtsCtx(ctx=1, gen=0, voice=VOICE, epoch=0, first_chunk_tokens=15)
    c.samples = 4000 * 24                                       # 4 s received
    c.first_frame_at = 0.0
    margin = (15 + 3) * 40 + rig.CREDIT_MARGIN_MS
    assert not c.rig_may_produce(0.5, 3000)                     # real time: 500 + 3000 < 4000 + margin
    assert c.rig_may_produce(10.0, 3000)
    c.sent_played_ms = 1000.0                                   # the rig froze at 1000
    assert not c.rig_may_produce(10.0, 3000)                    # an estimate never opens the clock
    c.sent_played_ms = 4000.0 + margin - 3000 + 1
    assert c.rig_may_produce(10.0, 3000)
    c.paused_since = 9.0
    assert not c.rig_may_produce(10.0, 3000)


async def test_a_reply_past_max_text_continues_in_a_fresh_context(live):
    fake, conn, cpu, eng = live
    fake.max_text = 40
    await asyncio.sleep(0)
    await conn.close()
    conn2 = make_conn(fake)
    conn2.start()
    assert await proven(conn2)
    assert conn2.max_text() == 40
    sess = rig.RigSession(conn2, rig.VoiceSync(fake.url, fake.token), rig.RigPowerGate(None))
    sess.voices[None] = VOICE
    eng2 = ReplyEngine(cpu, sess)
    sentences = ["Fifteen chars a.", "Fifteen chars b.", "Fifteen chars c.", "Fifteen chars d.", "Short."]
    vals = []
    for i, s in enumerate(sentences):
        out = await asyncio.to_thread(lambda s=s, i=i: list(eng2.stream_reply(s, None, None, gen=1, seq=i + 1,
                                                                              rid="long")))
        vals.append({int(f[0]) for f in out})
    assert all(v == {fake.tts_value} for v in vals) and cpu.said == []
    opens = fake.of("tts.open")
    assert len(opens) == 2
    assert {o["voice"] for o in opens} == {VOICE} and len({o["seed"] for o in opens}) == 1
    texts = fake.of("tts.text")
    per_ctx: dict = {}
    for t in texts:
        per_ctx[t["ctx"]] = per_ctx.get(t["ctx"], 0) + len(t["text"])
    assert all(n <= 40 for n in per_ctx.values()) and len(per_ctx) == 2
    assert [m for m in fake.received if m.get("type") == "tts.close"]   # the full ctxs were closed
    assert rig.METRICS.last_fallback_reason is None
    await conn2.close()
