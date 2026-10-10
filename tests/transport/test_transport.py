"""Transport: negotiation, v2 compatibility, v3 Opus/PCM framing and batching, say ordering, uplink decode."""
import asyncio
import json
import time

import numpy as np
import pytest

from dispatch_voice.transport import Transport, opus, parse_query, wire3

needs_opus = pytest.mark.skipif(not opus.available(), reason="libopus not installed")
Q3 = {"d": "device-0001", "s": "stream-0001", "v": "3", "up": "opus,pcm16", "down": "opus,pcm16"}


class RecWS:
    def __init__(self, delay=0.0):
        self.out = []
        self.delay = delay

    async def send_text(self, s):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.out.append(("text", json.loads(s), time.monotonic()))

    async def send_bytes(self, b):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.out.append(("bytes", bytes(b), time.monotonic()))

    def binary(self):
        return [wire3.unpack_down(b) for k, b, _ in self.out if k == "bytes"]

    def kinds(self):
        out = []
        for k, v, _ in self.out:
            if k == "text":
                out.append(v["type"])
            else:
                _c, _kind, _gen, seq, _t, packets = wire3.unpack_down(v)
                out.append(f"bin:{seq}:{len(packets)}")
        return out


def tone(seconds, rate=24000, hz=200.0):
    t = np.arange(int(rate * seconds)) / rate
    return (0.3 * np.sin(2 * np.pi * hz * t) * 32767).astype(np.int16)


def decode_seq(msgs, seq, rate=24000):
    dec = opus.Decoder(rate)
    return np.concatenate([dec.decode(p) for c, k, g, s, t, packets in msgs if s == seq for p in packets])


# --------------------------------------------------------------- negotiation #
def test_parse_query_validates_everything():
    n = parse_query({"d": "short", "s": "x" * 65, "v": "3", "up": "opus,flac,PCM16,opus", "down": 5})
    assert (n.proto, n.device, n.stream, n.up, n.down) == (3, None, None, ("opus", "pcm16"), ())
    assert parse_query({}).proto == 2
    assert parse_query({"v": "4"}).proto == 2
    n = parse_query(Q3)
    assert (n.device, n.stream) == ("device-0001", "stream-0001")


@needs_opus
def test_negotiate_v3_opus_and_preferences():
    t = Transport.negotiate(None, Q3)
    assert (t.proto, t.codec_in, t.codec_out, t.stream) == (3, "opus", "opus", "stream-0001")
    t = Transport.negotiate(None, {**Q3, "up": "pcm16,opus", "down": "pcm16"})
    assert (t.codec_in, t.codec_out) == ("pcm16", "pcm16")          # client order is "best first"
    t = Transport.negotiate(None, {**Q3, "up": "", "down": "flac"})
    assert (t.codec_in, t.codec_out) == ("pcm16", "pcm16")
    t = Transport.negotiate(None, Q3, opus_mode="off")
    assert (t.codec_in, t.codec_out) == ("pcm16", "pcm16")
    t = Transport.negotiate(None, Q3, rate_out=22050)
    assert t.codec_out == "pcm16"                                   # Opus cannot take 22.05 kHz


def test_v2_socket_has_no_v3_fields():
    t = Transport.negotiate(None, {"d": "device-0001"})
    assert t.proto == 2 and t.ready_fields() == {}
    assert (t.codec_in, t.codec_out) == ("pcm16", "pcm16")


@needs_opus
def test_ready_fields_and_gate():
    t = Transport.negotiate(None, Q3, opus_down_kbps=16, opus_up_kbps=12, hard_silence_s=1.4, hold_max_s=1.2)
    r = t.ready_fields()
    assert r == {"proto": 3, "codec_in": "opus", "codec_out": "opus", "opus_up_kbps": 12, "opus_down_kbps": 16,
                 "rx_off": 0, "gate": {"pre_roll_ms": 300, "hangover_s": 2.6}, "say": True}
    t = Transport.negotiate(None, Q3, hard_silence_s=3.0, hold_max_s=1.2)
    assert t.ready_fields()["gate"]["hangover_s"] == 5.0            # max(4.5, 1.8) + 0.5


# ------------------------------------------------------------------ v2 wire #
async def test_v2_audio_is_header_plus_pcm_adjacent_under_concurrency():
    ws = RecWS(delay=0.001)
    t = Transport(ws, proto=2)
    pcm = tone(0.04)

    async def audio(i):
        await t.send_audio(kind="reply", gen=0, seq=i, pcm16=pcm, t_ms=0, text="Hi." if i == 1 else None,
                           message_id="m1", role="character", chunk=0)

    await asyncio.gather(*(audio(i) for i in range(1, 6)), *(t.send({"type": "state", "state": "x"}) for _ in range(5)))
    kinds = [k for k, _, _ in ws.out]
    for i, (k, v, _) in enumerate(ws.out):
        if k == "bytes":
            assert ws.out[i - 1][0] == "text" and ws.out[i - 1][1]["type"] == "audio"
            assert len(v) == pcm.nbytes
    assert kinds.count("bytes") == 5
    head = next(v for k, v, _ in ws.out if k == "text" and v["type"] == "audio" and v["seq"] == 1)
    assert head == {"type": "audio", "gen": 0, "seq": 1, "chunk": 0, "sample_rate": 24000, "text": "Hi.",
                    "message_id": "m1", "kind": "reply", "role": "character"}
    assert await t.send_say({"gen": 0, "seq": 1, "text": "Hi."}) is False      # no `say` on v2
    assert await t.send_probe(1000) == 0


async def test_v2_uplink_is_raw():
    t = Transport(None, proto=2)
    assert t.decode_up(b"\x10\x00" * 320) is None


# ------------------------------------------------------------------ v3 Opus #
@needs_opus
async def test_v3_opus_first_message_at_once_then_full_batches_and_padded_tail():
    ws = RecWS()
    t = Transport(ws, proto=3, codec_out="opus")
    x = tone(1.03)                                                   # 51.5 frames
    await t.send_say({"gen": 1, "seq": 1, "text": "Hello there.", "message_id": "m1", "kind": "reply",
                      "role": "character"})
    await t.send_audio(kind="reply", gen=1, seq=1, pcm16=x[:2400], t_ms=0)     # 100 ms -> 5 packets at once
    assert ws.kinds() == ["say", "bin:1:5"]
    await t.send_audio(kind="reply", gen=1, seq=1, pcm16=x[2400:], t_ms=0)     # 46.5 frames more
    msgs = ws.binary()
    assert [len(m[5]) for m in msgs] == [5, 10, 10, 10, 10]                    # 6 packets held, rem 240
    await t.end_sentence(kind="reply", gen=1, seq=1)
    msgs = ws.binary()
    sizes = [len(m[5]) for m in msgs]
    la = opus.Encoder(24000, 24000).lookahead
    total_frames = -(-(len(x) + la) // 480)
    assert sum(sizes) == total_frames and max(sizes) <= 10
    assert all(m[1] == wire3.KIND_REPLY and m[2] == 1 and m[3] == 1 for m in msgs)
    y = decode_seq(msgs, 1)
    n = min(len(x), len(y) - la)
    assert n == len(x)                                                         # no tail lost
    assert np.corrcoef(x[:n].astype(float), y[la:la + n].astype(float))[0, 1] > 0.95
    assert ws.out[0][1]["t"] <= msgs[0][4] <= t.now_ms()                       # send-time stamps


@needs_opus
async def test_partial_batch_waits_at_most_100ms():
    ws = RecWS()
    t = Transport(ws, proto=3, codec_out="opus")
    await t.send_audio(kind="reply", gen=0, seq=1, pcm16=tone(0.02), t_ms=0)   # first: at once
    t0 = time.monotonic()
    await t.send_audio(kind="reply", gen=0, seq=1, pcm16=tone(0.06), t_ms=0)   # 3 packets: held
    assert len(ws.binary()) == 1
    await asyncio.sleep(0.2)
    msgs = ws.binary()
    assert [len(m[5]) for m in msgs] == [1, 3]
    waited = ws.out[-1][2] - t0
    assert 0.08 <= waited <= 0.16, waited
    await t.close()


@needs_opus
async def test_new_sentence_or_say_ends_the_previous_one_in_order():
    ws = RecWS()
    t = Transport(ws, proto=3, codec_out="opus")
    await t.send_audio(kind="reply", gen=0, seq=1, pcm16=tone(0.02), t_ms=0)
    await t.send_audio(kind="reply", gen=0, seq=1, pcm16=tone(0.07), t_ms=0)   # held + remainder
    await t.send_say({"gen": 0, "seq": 2, "text": "Next.", "kind": "reply"})
    await t.send_audio(kind="reply", gen=0, seq=2, pcm16=tone(0.1), t_ms=0)
    k = ws.kinds()
    assert k.index("say") > max(i for i, s in enumerate(k) if s.startswith("bin:1:"))
    assert all(s.startswith("bin:2:") for s in k[k.index("say") + 1:])
    seq1 = sum(len(m[5]) for m in ws.binary() if m[3] == 1)
    assert seq1 * 480 >= int(0.09 * 24000) + opus.Encoder(24000, 24000).lookahead
    await t.close()


@needs_opus
async def test_ack_last_flag_and_rung_change_lands_on_the_next_sentence():
    ws = RecWS()
    t = Transport(ws, proto=3, codec_out="opus")
    speech = (np.random.default_rng(1).standard_normal(24000) * 6000).astype(np.int16)
    await t.send_audio(kind="ack", gen=0, seq=1, pcm16=speech, t_ms=0, last=True)
    a = [m for m in ws.binary() if m[3] == 1]
    assert all(m[1] == wire3.KIND_ACK for m in a)
    t.link.request({"mode": "audio", "down_kbps": 16, "why": "goodput"})
    assert t.at_sentence_boundary() == {"down_kbps": 16}
    await t.send_audio(kind="reply", gen=0, seq=2, pcm16=speech, t_ms=0, last=True)
    b = [m for m in ws.binary() if m[3] == 2]
    size = lambda ms: sum(len(p) for m in ms for p in m[5]) / sum(len(m[5]) for m in ms)
    assert size(b) < size(a) * 0.8                                   # 16 kbps packets are smaller


@needs_opus
async def test_decoder_failure_switches_the_downlink_to_pcm():
    ws = RecWS()
    t = Transport(ws, proto=3, codec_out="opus")
    t.link.request({"mode": "audio", "why": "decoder"})
    assert t.at_sentence_boundary() == {"pcm_fallback": True}
    assert t.codec_out == "pcm16"
    await t.send_audio(kind="reply", gen=0, seq=1, pcm16=tone(0.1), t_ms=0, last=True)
    assert {m[0] for m in ws.binary()} == {wire3.CODEC_PCM16}


async def test_v3_pcm_is_40ms_per_message_without_a_header():
    ws = RecWS()
    t = Transport(ws, proto=3, codec_out="pcm16")
    x = tone(0.1)
    await t.send_audio(kind="reply", gen=2, seq=5, pcm16=x, t_ms=0)
    msgs = ws.binary()
    assert [len(m[5][0]) for m in msgs] == [1920, 1920, 960]
    assert all(len(m[5]) == 1 and m[0] == wire3.CODEC_PCM16 and m[2] == 2 and m[3] == 5 for m in msgs)
    assert np.frombuffer(b"".join(m[5][0] for m in msgs), "<i2").tolist() == x.tolist()
    assert not any(k == "text" for k, _, _ in ws.out)                # no JSON audio header on v3


async def test_float_input_is_converted():
    ws = RecWS()
    t = Transport(ws, proto=3, codec_out="pcm16")
    await t.send_audio(kind="reply", gen=0, seq=1, pcm16=np.array([0.5, -2.0], np.float32), t_ms=0)
    assert np.frombuffer(ws.binary()[0][5][0], "<i2").tolist() == [16383, -32767]


@needs_opus
async def test_drop_older_gens_discards_pending_audio():
    ws = RecWS()
    t = Transport(ws, proto=3, codec_out="opus")
    await t.send_audio(kind="reply", gen=0, seq=1, pcm16=tone(0.02), t_ms=0)
    await t.send_audio(kind="reply", gen=0, seq=1, pcm16=tone(0.06), t_ms=0)  # held
    await t.drop_older_gens(1)
    await asyncio.sleep(0.15)
    assert [len(m[5]) for m in ws.binary()] == [1]                    # the held packets never went out
    await t.close()


async def test_say_validation_and_probe():
    ws = RecWS()
    t = Transport(ws, proto=3, codec_out="pcm16")
    with pytest.raises(TypeError):
        await t.send_say({"gen": 0, "text": "x"})
    assert await t.send_say({"gen": 0, "seq": 1, "text": "Hi.", "kind": "reply"}) is True
    say = ws.out[-1][1]
    assert say["type"] == "say" and isinstance(say["t"], int)
    n = await t.send_probe(20000)
    assert n == 20000
    _c, kind, _gen, _seq, _tt, packets = ws.binary()[-1]
    assert kind == wire3.KIND_PROBE and len(packets[0]) == 20000
    assert await t.send_probe(20000) == 0                              # one per 20 s


# ------------------------------------------------------------------- uplink #
@needs_opus
def test_decode_up_opus_and_pcm():
    t = Transport.negotiate(None, Q3)
    enc = opus.Encoder(16000, 16000, application="voip")
    x = (np.sin(np.arange(1600) / 5) * 8000).astype(np.int16)
    packets = [enc.encode(x[i:i + 320]) for i in range(0, 1600, 320)]
    off, flags, pcm = t.decode_up(wire3.pack_up(1, wire3.FLAG_LOCAL_SPEAKING, 3200, packets))
    assert (off, flags, len(pcm), pcm.dtype) == (3200, 1, 1600, np.int16)
    off, flags, pcm = t.decode_up(wire3.pack_up(0, 0, 4800, [x[:320].tobytes()]))
    assert pcm.tolist() == x[:320].tolist()
    assert t.decode_up(x.tobytes()) is None                            # a raw frame on a v3 socket is v2 PCM
    with pytest.raises(ValueError):
        t.decode_up(b"\x03\x01\x00")
    with pytest.raises(ValueError):
        t.decode_up(wire3.pack_up(1, 0, 0, [b"\xff\xff\xff\xff\xff"]))   # corrupt Opus -> bad frame
    assert t.stats["bad_frames"] == 2 and t.stats["rx_opus_packets"] == 5


def test_too_big_and_pong():
    assert Transport.too_big(b"\x03" * 65537) and not Transport.too_big(b"\x03" * 65536)
    t = Transport.negotiate(None, Q3)
    t.offsets.feed(0, np.zeros(320, np.int16))
    p = t.pong(123.5)
    assert p["type"] == "pong" and p["t"] == 123.5 and p["rx_off"] == 320 and isinstance(p["srv_t"], int)


async def test_link_metrics_include_relay_only_when_known():
    t = Transport.negotiate(None, Q3)
    m = t.link_metrics({"path": "relay", "region": "tok"})
    assert m["relay"] is True and m["mode"] == "audio" and "out_ms" in m
    assert "relay" not in t.link_metrics({"path": "idle"})


async def test_closed_transport_sends_nothing():
    ws = RecWS()
    t = Transport(ws, proto=3, codec_out="pcm16")
    await t.close()
    await t.send({"type": "state"})
    await t.send_audio(kind="reply", gen=0, seq=1, pcm16=tone(0.1), t_ms=0)
    assert ws.out == []


@needs_opus
async def test_audio_end_follows_the_last_audio_of_a_sentence():
    """Amendment T6-2: one `audio_end` per sentence that sent audio, after its padded tail."""
    ws = RecWS()
    t = Transport(ws, proto=3, codec_out="opus")
    await t.send_say({"gen": 3, "seq": 1, "text": "One.", "kind": "reply"})
    await t.send_audio(kind="reply", gen=3, seq=1, pcm16=tone(0.5), t_ms=0)
    await t.end_sentence(kind="reply", gen=3, seq=1)
    await t.send_say({"gen": 3, "seq": 2, "text": "Two.", "kind": "reply"})
    await t.send_audio(kind="reply", gen=3, seq=2, pcm16=tone(0.3), t_ms=0)
    await t.send_say({"gen": 3, "seq": 3, "text": "Three.", "kind": "reply"})      # ends seq 2 implicitly
    await t.send_audio(kind="ack", gen=3, seq=4, pcm16=tone(0.2), t_ms=0, last=True)
    k = ws.kinds()
    ends = [v for kind, v, _ in ws.out if kind == "text" and v["type"] == "audio_end"]
    assert ends == [{"type": "audio_end", "gen": 3, "seq": 1, "kind": "reply"},
                    {"type": "audio_end", "gen": 3, "seq": 2, "kind": "reply"},
                    {"type": "audio_end", "gen": 3, "seq": 4, "kind": "ack"}]
    i1 = k.index("audio_end")
    assert all(not x.startswith("bin:1:") for x in k[i1 + 1:])          # nothing of seq 1 after its end
    assert k.index("say", i1) > i1                                       # and before the next say
    await t.end_sentence(kind="reply", gen=3, seq=3)                       # seq 3 sent no audio: no end
    assert len([1 for kind, v, _ in ws.out if kind == "text" and v["type"] == "audio_end"]) == 3


async def test_audio_end_on_pcm_and_never_on_v2():
    ws = RecWS()
    t = Transport(ws, proto=3, codec_out="pcm16")
    await t.send_audio(kind="reply", gen=0, seq=1, pcm16=tone(0.1), t_ms=0, last=True)
    assert ws.kinds()[-1] == "audio_end"
    ws2 = RecWS()
    t2 = Transport(ws2, proto=2)
    await t2.send_audio(kind="reply", gen=0, seq=1, pcm16=tone(0.1), t_ms=0, last=True)
    await t2.end_sentence(kind="reply", gen=0, seq=1)
    assert [v["type"] for kind, v, _ in ws2.out if kind == "text"] == ["audio"]
