"""ReplyEngine: one rig context per reply, flush-per-sentence framing, the 400 ms
start rule, a rig dying mid-sentence, credit, cancel/pause/resume, word timings,
unknown voices, and the "never slower" property."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from dispatch_voice.host import rig
from dispatch_voice.host.engines.rig_tts import ReplyEngine

from .conftest import FakeCpuTTS, make_conn, proven, wait_for

VOICE = "sha256:" + "ab" * 32


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


def say(eng, text, *, gen=0, seq=1, rid="r1", stop=None, kind="reply", voice=None):
    """Run one sentence like the session's TTS thread does: (frames, values)."""
    out = list(eng.stream_reply(text, voice, stop, gen=gen, seq=seq, rid=rid, kind=kind))
    return out, sorted({int(f[0]) for f in out})


async def test_one_ctx_per_reply_and_frames_per_flush(live):
    fake, conn, cpu, eng = live
    out1, v1 = await asyncio.to_thread(say, eng, "Two meetings today.", seq=1)
    out2, v2 = await asyncio.to_thread(say, eng, "Want details?", seq=2)
    assert v1 == v2 == [fake.tts_value] and len(out1) == len(out2) == fake.tts_frames
    assert all(len(f) == 960 for f in out1 + out2) and cpu.said == []
    opens = fake.of("tts.open")
    assert len(opens) == 1
    o = opens[0]
    assert o["voice"] == VOICE and o["gen"] == 0 and o["first_chunk_tokens"] == 15
    assert isinstance(o["seed"], int) and o["speculative"] is False
    texts = fake.of("tts.text")
    assert [t["text"] for t in texts] == ["Two meetings today.", "Want details?"]
    assert all(t["final"] is True and t["ctx"] == o["ctx"] for t in texts)
    flushes = fake.of("tts.flush")
    assert len(flushes) == 2 and flushes[0]["id"] != flushes[1]["id"]
    assert fake.of("credit")[0] == {"type": "credit", "ctx": o["ctx"], "ahead_ms": 3000}
    # The next reply closes the previous context and opens its own.
    await asyncio.to_thread(say, eng, "Next.", seq=3, rid="r2")
    assert await wait_for(lambda: fake.of("tts.close"))
    assert fake.of("tts.close")[0]["ctx"] == o["ctx"] and len(fake.of("tts.open")) == 2
    assert fake.errors == []
    snap = rig.METRICS.snapshot()
    assert snap["replies"] == 2 and snap["tts"] == 2 and snap["ttfa_p50"] is not None


async def test_words_become_sentence_relative_timings(live):
    fake, conn, cpu, eng = live
    await asyncio.to_thread(say, eng, "Two meetings today.", seq=1)
    await asyncio.to_thread(say, eng, "Want the details?", seq=2)
    w1, w2 = eng.words_for(1), eng.words_for(2)
    assert [w[0] for w in w1] == ["Two", "meetings", "today"] and w1[0][1] == 0
    assert [w[0] for w in w2] == ["Want", "the", "details"] and w2[0][1] == 0   # not the ctx clock
    assert w2[-1][2] == 120 and eng.words_for(99) is None


async def test_a_late_first_sentence_moves_the_whole_reply_to_the_cpu(live):
    fake, conn, cpu, eng = live
    fake.tts_delay = 0.9
    t0 = time.monotonic()
    out, values = await asyncio.to_thread(say, eng, "Hello there.", seq=1)
    dt = time.monotonic() - t0
    assert values == [FakeCpuTTS.VALUE] and 0.38 <= dt < 0.7, dt     # <= 400 ms, once
    assert cpu.said == ["Hello there."]
    assert await wait_for(lambda: any(c.get("ctx") is not None for c in fake.of("cancel")))  # it stops rendering
    fake.tts_delay = 0.0
    out, values = await asyncio.to_thread(say, eng, "And more.", seq=2)
    assert values == [FakeCpuTTS.VALUE] and cpu.said[-1] == "And more."   # one reply, one engine
    assert len(fake.of("tts.text")) == 1
    assert rig.METRICS.last_fallback_reason == "tts_late" and not conn.usable()


async def test_credit_held_audio_is_not_late(live):
    """The rig 3 s ahead of the phone is holding back on purpose: no fallback."""
    fake, conn, cpu, eng = live
    fake.tts_frames = 100                               # 4 s of audio in the first sentence
    await asyncio.to_thread(say, eng, "A long first sentence.", seq=1)
    fake.tts_frames = 3
    fake.tts_delays = [0.7]                             # held back by the credit, then renders
    out, values = await asyncio.to_thread(say, eng, "Second.", seq=2)
    assert values == [fake.tts_value] and cpu.said == []


async def test_rig_dying_mid_sentence_rerenders_it_on_the_cpu(live):
    fake, conn, cpu, eng = live
    await asyncio.to_thread(say, eng, "First.", seq=1)
    fake.tts_frames = 50
    got: list = []

    def run():
        for f in eng.stream_reply("Second sentence here.", None, None, gen=0, seq=2, rid="r1"):
            got.append(int(f[0]))
            if len(got) == 3:
                asyncio.run_coroutine_threadsafe(fake.kill(), loop).result(2)

    loop = asyncio.get_running_loop()
    fake.tts_delays = [0.0]
    t0 = time.monotonic()
    await asyncio.to_thread(run)
    assert time.monotonic() - t0 < 1.5
    rig_part = [v for v in got if v == fake.tts_value]
    cpu_part = [v for v in got if v == FakeCpuTTS.VALUE]
    assert len(rig_part) >= 3 and cpu_part == [FakeCpuTTS.VALUE] * cpu.frames
    assert got.index(FakeCpuTTS.VALUE) == len(rig_part)          # the CPU render follows, whole
    assert cpu.said == ["Second sentence here."]
    assert eng.words_for(2) is None                              # no rig timings for that seq
    out, values = await asyncio.to_thread(say, eng, "Third.", seq=3)
    assert values == [FakeCpuTTS.VALUE]                          # the rest of the reply: CPU


async def test_cancel_pause_resume_and_credit(live):
    fake, conn, cpu, eng = live
    await asyncio.to_thread(say, eng, "One.", seq=1, gen=4)
    ctx = fake.of("tts.open")[0]["ctx"]
    eng.pause()
    eng.resume()
    eng.heard(4, 1, 60)
    await asyncio.sleep(0.1)
    eng.heard(4, 1, 90)                                          # throttled (<= 1/s)
    eng.cancel_gen(4)
    assert await wait_for(lambda: fake.of("cancel"))
    assert fake.of("tts.pause") == [{"type": "tts.pause", "ctx": ctx}]
    assert fake.of("tts.resume") == [{"type": "tts.resume", "ctx": ctx}]
    credits = [c for c in fake.of("credit") if "played_ms" in c]
    assert credits == [{"type": "credit", "ctx": ctx, "ahead_ms": 3000, "played_ms": 60}]
    assert fake.of("cancel")[-1] == {"type": "cancel", "gen": 4}
    # Order on the socket: pause, resume, credit, cancel — as the session called them.
    kinds = [m["type"] for m in fake.received if m["type"] in ("tts.pause", "tts.resume", "credit", "cancel")]
    assert kinds[-4:] == ["tts.pause", "tts.resume", "credit", "cancel"]


async def test_a_stop_cancels_the_ctx_and_never_falls_back(live):
    fake, conn, cpu, eng = live
    fake.tts_stall = True
    stop = threading.Event()
    threading.Timer(0.1, stop.set).start()
    out, values = await asyncio.to_thread(say, eng, "Cut me off.", seq=1, stop=stop)
    assert out == [] and cpu.said == []
    assert await wait_for(lambda: any("ctx" in c for c in fake.of("cancel")))


async def test_unknown_voice_is_a_cpu_reply_and_a_resync(live, monkeypatch):
    fake, conn, cpu, eng = live
    fake.voices = set()
    resynced = []
    monkeypatch.setattr(eng.rig, "resync_threadsafe", lambda pid: resynced.append(pid))
    out, values = await asyncio.to_thread(say, eng, "Hello.", seq=1)
    assert values == [FakeCpuTTS.VALUE] and resynced == [None]
    assert rig.METRICS.last_fallback_reason == "unknown_voice"


async def test_acks_notices_and_voices_not_on_the_rig_stay_on_the_cpu(live):
    fake, conn, cpu, eng = live
    out, values = await asyncio.to_thread(say, eng, "Mm-hm.", kind="ack")
    assert values == [FakeCpuTTS.VALUE]
    out, values = await asyncio.to_thread(say, eng, "Narrated.", voice="builtin:x", rid="r9")
    assert values == [FakeCpuTTS.VALUE]
    assert list(eng.stream("plain", None)) and eng.cache_tag(None) == "cb-cpu-tag"
    assert fake.of("tts.open") == []


async def test_never_slower_with_the_rig_down_a_reply_is_exactly_the_cpu(fake_rig):
    fake_rig.refuse = (503, {"Retry-After": "30"})
    conn = make_conn(fake_rig)
    conn.start()
    assert await wait_for(lambda: conn.state == "down")
    sess = rig.RigSession(conn, rig.VoiceSync(fake_rig.url, fake_rig.token), rig.RigPowerGate(None))
    sess.voices[None] = VOICE
    cpu, ref = FakeCpuTTS(), FakeCpuTTS()
    eng = ReplyEngine(cpu, sess)
    t0 = time.monotonic()
    got = [list(eng.stream_reply(s, None, None, gen=0, seq=i, rid="r1")) for i, s in enumerate(["A.", "B."])]
    dt = time.monotonic() - t0
    want = [list(ref.stream(s, None, None)) for s in ["A.", "B."]]
    assert [[f.tolist() for f in g] for g in got] == [[f.tolist() for f in w] for w in want]
    assert dt < 0.05 and fake_rig.of("tts.open") == []
    await conn.close()


async def test_voice_sync_heads_then_puts_only_the_conditionals(tmp_path):
    import hashlib

    import httpx
    npz = tmp_path / "voice.npz"
    npz.write_bytes(b"conditionals-only")
    sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    seen = []

    def handler(req: httpx.Request):
        seen.append((req.method, req.url.path, req.headers.get("authorization"), req.content))
        if req.method == "HEAD":
            return httpx.Response(404)
        return httpx.Response(201, json={"id": sha})

    sync = rig.VoiceSync("ws://127.0.0.1:19101/v1/realtime", "tok", transport=httpx.MockTransport(handler))
    assert await sync.ensure(sha, npz) == "ready"
    assert [s[:2] for s in seen] == [("HEAD", f"/v1/voices/{sha}"), ("PUT", f"/v1/voices/{sha}")]
    assert all(s[2] == "Bearer tok" for s in seen) and seen[1][3] == b"conditionals-only"
    assert await sync.ensure(sha, npz) == "ready" and len(seen) == 2          # cached
    bad = rig.VoiceSync("ws://h:1/x", "tok", transport=httpx.MockTransport(
        lambda r: httpx.Response(404) if r.method == "HEAD" else httpx.Response(400)))
    assert await bad.ensure(sha, npz) == "bad"


async def test_reply_done_closes_the_ctx_at_once_and_only_its_own(live):
    fake, conn, cpu, eng = live
    await asyncio.to_thread(say, eng, "All done.", seq=1, rid="r1")
    eng.reply_done("other")                       # another reply's end: not this ctx
    await asyncio.sleep(0.1)
    assert fake.of("tts.close") == []
    eng.reply_done("r1")
    assert await wait_for(lambda: fake.of("tts.close"))
    assert fake.of("tts.close")[0]["ctx"] == fake.of("tts.open")[0]["ctx"]
    # The next reply opens its own ctx and closes nothing twice.
    await asyncio.to_thread(say, eng, "Next.", seq=2, rid="r2")
    assert len(fake.of("tts.open")) == 2 and len(fake.of("tts.close")) == 1
    assert fake.errors == []
