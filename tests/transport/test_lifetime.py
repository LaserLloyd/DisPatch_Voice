"""Transport lifetime and the review-T6 fixes: close() vs a worker-thread encode (H1),
late audio for an ended sentence (M3), decoded-length bound (L1), the batching timer on a
dead socket (L6), pong's echoed `t` (L7), the bounded relay call log (L2)."""
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from dispatch_voice import transport
from dispatch_voice.transport import RelayIndicator, Transport, gen_older, opus, wire3
from settling import pause

needs_opus = pytest.mark.skipif(not opus.available(), reason="libopus not installed")
RACE = Path(__file__).with_name("race_close.py")


def tone(seconds, rate=24000, hz=200.0):
    t = np.arange(int(rate * seconds)) / rate
    return (0.3 * np.sin(2 * np.pi * hz * t) * 32767).astype(np.int16)


class RecWS:
    def __init__(self):
        self.out = []

    async def send_text(self, s):
        self.out.append(("text", json.loads(s)))

    async def send_bytes(self, b):
        self.out.append(("bytes", wire3.unpack_down(bytes(b))))

    def kinds(self):
        return [v["type"] if k == "text" else f"bin:{v[3]}" for k, v in self.out]


def _race(mode: str, n: int) -> dict:
    """Run the stress in a child: a heap abort / SIGSEGV must fail the test, not kill pytest."""
    p = subprocess.run([sys.executable, str(RACE), mode, str(n)], capture_output=True, text=True,
                       timeout=240, check=False)
    assert p.returncode == 0, f"{mode}: exit {p.returncode}\n{p.stderr[-2000:]}"
    return json.loads(p.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------ H1 #
@needs_opus
def test_close_racing_a_threaded_encode_never_crashes_200x():
    r = _race("natural", 200)
    assert r["results"] == {"ok": 200}, r


@needs_opus
def test_close_waits_for_an_encode_inside_the_foreign_call():
    r = _race("inner", 3)
    assert r["results"] == {"ok": 3}, r


@needs_opus
def test_close_that_stops_waiting_still_never_overlaps_the_foreign_call():
    r = _race("stuck", 3)
    assert r["results"] == {"ok": 3}, r


@needs_opus
def test_close_racing_a_threaded_decode_never_crashes():
    r = _race("decode", 30)
    assert set(r["results"]) <= {"ok", "ValueError"} and sum(r["results"].values()) == 30, r


@needs_opus
def test_codec_close_is_idempotent_and_later_calls_raise():
    enc = opus.Encoder(24000, 24000)
    enc.close()
    enc.close()
    assert enc.closed
    with pytest.raises(opus.OpusError):
        enc.encode(np.zeros(enc.frame, np.int16))
    with pytest.raises(opus.OpusError):
        _ = enc.lookahead
    dec = opus.Decoder(16000)
    dec.close()
    dec.close()
    with pytest.raises(opus.OpusError):
        dec.decode(b"\xf8\xff\xfe")
    with pytest.raises(opus.OpusError):
        dec.reset()


async def test_close_is_bounded_when_a_send_is_stuck(monkeypatch):
    monkeypatch.setattr(transport, "CLOSE_WAIT_S", 0.1)
    gate = asyncio.Event()

    class StuckWS(RecWS):
        async def send_bytes(self, b):
            await gate.wait()                         # a stalled downlink: never drains

    ws = StuckWS()
    t = Transport(ws, proto=3, codec_out="pcm16")
    sender = asyncio.create_task(t.send_audio(kind="reply", gen=0, seq=1, pcm16=tone(0.2), t_ms=0))
    await pause(0.02)
    t0 = time.monotonic()
    await t.close()
    assert time.monotonic() - t0 < 1.0
    assert t._streams == {} and t.closed
    gate.set()
    await asyncio.wait_for(sender, 2)                 # the stuck sender finds it closed and stops
    await t.close()                                   # idempotent


async def test_nothing_reopens_after_close():
    ws = RecWS()
    t = Transport(ws, proto=3, codec_out="pcm16")
    await t.close()
    await t.send_audio(kind="reply", gen=0, seq=1, pcm16=tone(0.1), t_ms=0)
    await t.end_sentence(kind="reply", gen=0, seq=1)
    assert await t.send_say({"gen": 0, "seq": 2, "text": "x"}) is False
    assert await t.send_probe(1000) == 0
    assert t._streams == {} and ws.out == []
    with pytest.raises(ValueError):
        t.decode_up(wire3.pack_up(wire3.CODEC_PCM16, 0, 0, [b"\x00\x00" * 160]))


# ------------------------------------------------------------------ M3 #
@needs_opus
async def test_late_chunk_after_its_sentence_ended_is_dropped_not_reopened():
    """The reviewer's probe: say(2) before end_sentence(1), then a late chunk of 1."""
    ws = RecWS()
    t = Transport(ws, proto=3, codec_out="opus")
    await t.send_say({"gen": 0, "seq": 1, "text": "One.", "kind": "reply"})
    await t.send_audio(kind="reply", gen=0, seq=1, pcm16=tone(0.5), t_ms=0)
    await t.send_say({"gen": 0, "seq": 2, "text": "Two.", "kind": "reply"})   # ends seq 1
    await t.send_audio(kind="reply", gen=0, seq=1, pcm16=tone(0.5), t_ms=0)    # late: dropped
    await t.end_sentence(kind="reply", gen=0, seq=1)                            # no-op
    await t.send_audio(kind="reply", gen=0, seq=2, pcm16=tone(0.3), t_ms=0, last=True)
    k = ws.kinds()
    ends = [v for kind, v in ws.out if kind == "text" and v["type"] == "audio_end"]
    assert [e["seq"] for e in ends] == [1, 2]
    i_say2 = k.index("say", k.index("say") + 1)
    assert all(x != "bin:1" for x in k[i_say2:])
    assert t.stats["late_audio_dropped"] == 1
    await t.close()


async def test_audio_for_a_dropped_gen_is_ignored_and_gen_compare_wraps():
    ws = RecWS()
    t = Transport(ws, proto=3, codec_out="pcm16")
    await t.drop_older_gens(5)
    await t.send_audio(kind="reply", gen=4, seq=1, pcm16=tone(0.1), t_ms=0)
    assert ws.out == [] and t.stats["late_audio_dropped"] == 1
    await t.send_audio(kind="reply", gen=5, seq=1, pcm16=tone(0.04), t_ms=0)
    assert ws.kinds() == ["bin:1"]
    assert gen_older(4, 5) and not gen_older(5, 5) and not gen_older(6, 5)
    assert gen_older(0xFFFF, 0) and not gen_older(0, 0xFFFF)          # across the u16 wrap
    await t.drop_older_gens(3)                                         # a lower gen never lowers the floor
    await t.send_audio(kind="reply", gen=4, seq=2, pcm16=tone(0.04), t_ms=0)
    assert t.stats["late_audio_dropped"] == 2
    await t.close()


# ------------------------------------------------------------------ L1 #
@needs_opus
def test_uplink_message_decoding_to_more_than_2s_is_refused():
    t = Transport(None, proto=3, codec_in="opus")
    enc = opus.Encoder(16000, 16000, application="voip", frame_ms=60)
    x = (np.sin(np.arange(16000 * 4) / 5) * 8000).astype(np.int16)
    packets = [enc.encode(x[i:i + enc.frame]) for i in range(0, 50 * enc.frame, enc.frame)]   # 50 x 60 ms = 3 s
    with pytest.raises(ValueError, match="more than 2 s"):
        t.decode_up(wire3.pack_up(wire3.CODEC_OPUS, 0, 0, packets))
    _off, _f, pcm = t.decode_up(wire3.pack_up(wire3.CODEC_OPUS, 0, 0, packets[:30]))        # 1.8 s is fine
    assert len(pcm) == 30 * 960
    assert t.stats["bad_frames"] == 1


def test_pcm_uplink_message_over_2s_is_refused():
    t = Transport(None, proto=3, codec_in="pcm16")
    pk = [b"\x01\x00" * 650] * 50                       # 32,500 samples = 2.03 s, 65,000 B
    with pytest.raises(ValueError):
        t.decode_up(wire3.pack_up(wire3.CODEC_PCM16, 0, 0, pk))
    _o, _f, pcm = t.decode_up(wire3.pack_up(wire3.CODEC_PCM16, 0, 0, pk[:49]))
    assert len(pcm) == 49 * 650


# ------------------------------------------------------------------ L6 #
@needs_opus
async def test_batching_timer_ends_quietly_on_a_dead_socket():
    class DyingWS(RecWS):
        n = 0

        async def send_bytes(self, b):
            DyingWS.n += 1
            if DyingWS.n > 1:
                raise RuntimeError("socket closed")
            await super().send_bytes(b)

    t = Transport(DyingWS(), proto=3, codec_out="opus")
    await t.send_audio(kind="reply", gen=0, seq=1, pcm16=tone(0.02), t_ms=0)   # first message at once
    await t.send_audio(kind="reply", gen=0, seq=1, pcm16=tone(0.06), t_ms=0)   # held -> timer
    timer = t._timer
    assert timer is not None
    await asyncio.sleep(0.2)
    assert timer.done() and not timer.cancelled() and timer.exception() is None
    await t.close()


# ------------------------------------------------------------------ L7 #
def test_pong_echoes_only_a_finite_number():
    t = Transport(None, proto=3)
    assert t.pong(12.5)["t"] == 12.5 and t.pong(7)["t"] == 7
    for bad in ({"x": 1}, [1, 2], "123", None, True, float("nan"), float("inf")):
        assert t.pong(bad)["t"] is None


# ------------------------------------------------------------------ L2 #
async def test_relay_call_log_is_bounded():
    async def run(argv, timeout):
        return 1, ""
    ind = RelayIndicator(runner=run, tailscale="/bin/false")
    for _ in range(100):
        ind._status_at = float("-inf")
        await ind.lookup("100.64.0.9")  # scrub-ok: fictional fixture address
    assert len(ind.calls) == 32
