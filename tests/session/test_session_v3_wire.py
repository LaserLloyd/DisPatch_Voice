"""Session hub, slice 4 (SPEC §5, §14 T3 (4)): the v3 wire through the session —
negotiation and `ready`, `say` + binary audio + `audio_end`, Opus both ways, uplink
offsets (zero-fill, gaps, `eot.upto`, `pong.rx_off`), the commit cursor (commit before
the host call, the open mark, `cursor {from}` + the replayed transcript), the
`after_partial` skip and the late-joiner belt (C.5 #11), the byte-bounded uplink queue,
`link` rungs / decoder fallback at the sentence boundary, probes, the `local_speaking`
flag, `drop_older_gens` on a barge-in — all with fakes (no models, no DisPatch)."""
from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import FakeSTT, FakeTTS, FakeWS, ScriptedDetector, utterance

from dispatch_voice.host import session as vs
from dispatch_voice.host.session import VoiceSession
from dispatch_voice.host.vad_turn import Discarded, SpeechStart
from dispatch_voice.transport import CommitCursor, Transport, opus, wire3
from session.test_session import THREAD, settings, settle
from session.test_session_s2 import Host, frame, reply

REPLY = "You have two things this afternoon. A dentist appointment at three. Want a reminder?"
STREAM = "str-0123456789ab"
DEVICE = "dev-abcdef12"
needs_opus = pytest.mark.skipif(not opus.available(), reason="libopus is not installed")


def query(codec: str = "pcm16", **over) -> dict:
    q = {"v": "3", "d": DEVICE, "s": STREAM, "up": f"{codec},pcm16", "down": f"{codec},pcm16"}
    q.update(over)
    return q


class RecordingDetector(ScriptedDetector):
    """A scripted detector that also records how many samples it was fed and keeps the
    attributes the session reads for the utterance start (``ep._frames``, ``_buf``)."""

    def __init__(self, script):
        super().__init__(script)
        self.fed = 0
        self.ep._frames = []
        self._buf = np.zeros(0, np.float32)

    def feed(self, pcm):
        self.fed += len(pcm) // 2
        return super().feed(pcm)


def make(script=(), *, codec="pcm16", q=None, cursor=None, host=None, tts=None, det=None,
         stt="what's the weather", **kw):
    ws = FakeWS(query=q if q is not None else query(codec))
    tts = tts or FakeTTS()
    tr = Transport.negotiate(ws, ws.query_params, rate_out=tts.sample_rate)
    host = host or Host()
    det = det or RecordingDetector(list(script))
    key = (DEVICE, STREAM) if cursor is not None else None
    s = VoiceSession(ws, thread_id=THREAD, settings=settings(merge_window_s=0.0), detector=det,
                     stt_engine=FakeSTT(stt), tts_engine=tts, host=host.build(),
                     notice=lambda text: np.ones(48, "<i2"), heard_wait_s=0.3,
                     transport=tr, cursor=cursor, cursor_key=key, **kw)
    return s, ws, tts, host, tr


async def start(s, ws, **extra):
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start", **extra})
    await settle(5)
    return task


def binaries(ws) -> list[tuple]:
    """Every v3 downlink message as (codec, kind, gen, seq, t, packets)."""
    return [wire3.unpack_down(b) for b in ws.sent if isinstance(b, bytes) and wire3.is_v3(b)]


def up_pcm(off: int, samples: int, *, flags: int = 0, value: int = 0) -> bytes:
    return wire3.pack_up(wire3.CODEC_PCM16, flags, off, [np.full(samples, value, "<i2").tobytes()])


# --------------------------------------------------------------------------- #
# negotiation, ready, downlink
# --------------------------------------------------------------------------- #

async def test_v3_ready_carries_the_wire_fields_and_the_session_gate():
    s, ws, _, _, tr = make()
    task = await start(s, ws)
    r = ws.json_of("ready")[0]
    assert r["proto"] == 3 and r["codec_in"] == "pcm16" and r["codec_out"] == "pcm16"
    assert r["opus_down_kbps"] == 24 and r["opus_up_kbps"] == 16 and r["rx_off"] == 0
    assert r["say"] is True and set(r["gate"]) == {"pre_roll_ms", "hangover_s"}
    ws.hang_up(); await task
    assert tr.closed                              # the session freed the codecs on the way out


async def test_v2_socket_gets_no_wire_fields_and_keeps_the_json_audio_header():
    s, ws, _, _, tr = make(q={})
    assert not tr.v3
    task = await start(s, ws)
    assert "proto" not in ws.json_of("ready")[0]
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": "Hello there."}))
    await settle(20)
    hdrs = ws.json_of("audio")
    assert hdrs and all(isinstance(ws.sent[ws.sent.index(h) + 1], bytes) for h in hdrs)
    assert not ws.json_of("audio_end") and not binaries(ws)
    ws.hang_up(); await task


async def test_v3_reply_is_say_then_binary_audio_then_audio_end_per_sentence():
    s, ws, tts, _, _ = make()
    task = await start(s, ws)
    for f in reply("m1", REPLY, prov="run:1"):
        s.on_app_frame(f)
    await settle(60)
    assert not ws.json_of("audio")                # no JSON audio header on v3
    says = ws.json_of("say")
    assert [m["seq"] for m in says] == [1, 2, 3]
    ends = ws.json_of("audio_end")
    assert [(m["seq"], m["kind"]) for m in ends] == [(1, "reply"), (2, "reply"), (3, "reply")]
    # wire order: say(n) < first binary of n < audio_end(n) < say(n+1)
    order = []
    for m in ws.sent:
        if isinstance(m, bytes):
            order.append(("bin", wire3.unpack_down(m)[3]))
        elif m.get("type") in ("say", "audio_end"):
            order.append((m["type"], m["seq"]))
    for q in (1, 2, 3):
        assert order.index(("say", q)) < order.index(("bin", q)) < order.index(("audio_end", q))
        if q < 3:
            assert order.index(("audio_end", q)) < order.index(("say", q + 1))
    pcm = [b for b in binaries(ws) if b[3] == 1]
    assert pcm and all(b[0] == wire3.CODEC_PCM16 and b[1] == wire3.KIND_REPLY for b in pcm)
    assert sum(len(p) // 2 for b in pcm for p in b[5]) == tts.chunks * tts.chunk_samples
    # `say.t` is the transport's clock: the same one the binary `t` carries
    assert says[0]["t"] <= pcm[0][4] + 5
    ws.hang_up(); await task


@needs_opus
async def test_v3_opus_downlink_decodes_to_the_sentence_and_ack_lines_are_opus_too():
    tts = FakeTTS(chunks=6, chunk_samples=960)           # 6 × 40 ms
    s, ws, _, _, tr = make(codec="opus", tts=tts)
    assert tr.codec_out == "opus" and tr.codec_in == "opus"
    task = await start(s, ws)
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": "Hello there."}))
    await settle(60)
    msgs = [b for b in binaries(ws) if b[3] == 1]
    assert msgs and all(b[0] == wire3.CODEC_OPUS for b in msgs)
    dec = opus.Decoder(24000)
    n = sum(len(dec.decode(p)) for b in msgs for p in b[5])
    # the padded tail: whole 20 ms frames covering the audio plus the encoder lookahead
    assert 6 * 960 <= n <= 6 * 960 + 480 + 480
    assert ws.json_of("audio_end") and ws.json_of("audio_end")[0]["seq"] == 1
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# uplink: offsets, eot.upto, pong.rx_off, gaps
# --------------------------------------------------------------------------- #

async def test_uplink_offsets_zero_fill_a_gate_gap_and_eot_upto_uses_them():
    det = RecordingDetector([[], [], [SpeechStart(), utterance()]])
    s, ws, _, _host, _tr = make(det=det)
    task = await start(s, ws)
    ws.push_audio(up_pcm(0, 320)); await settle(5)
    ws.push_audio(up_pcm(320 * 4, 320)); await settle(5)      # the mic gate left 3 frames out
    ws.push_audio(up_pcm(320 * 5, 320)); await settle(20)
    assert det.fed == 320 * 6                                  # 3 × 320 zeros were fed in between
    eot = ws.json_of("eot")[0]
    assert eot["upto"] == 320 * 6                              # utterance() has no trailing silence
    assert ws.json_of("transcript")[0]["upto"] == eot["upto"]
    ws.push_json({"type": "ping", "t": 7}); await settle(5)
    pong = ws.json_of("pong")[0]
    assert pong["rx_off"] == 320 * 6 and pong["t"] == 7
    ws.hang_up(); await task


async def test_overlapping_replay_is_dropped_and_a_large_gap_resets_the_detector():
    det = RecordingDetector([])
    s, ws, _, _, _tr = make(det=det)
    task = await start(s, ws)
    ws.push_audio(up_pcm(0, 640)); await settle(5)
    ws.push_audio(up_pcm(320, 640, flags=wire3.FLAG_REPLAY)); await settle(5)   # half already fed
    assert det.fed == 960
    resets = det.resets
    ws.push_audio(up_pcm(16000 * 60, 320)); await settle(5)                       # a minute later
    assert det.resets == resets + 1 and det.fed == 960 + 320
    ws.hang_up(); await task


async def test_a_bad_v3_message_is_a_rate_limited_bad_frame_not_a_crash():
    s, ws, _, _, _tr = make()
    task = await start(s, ws)
    ws.push_audio(b"\x03\x00\x00\x00\x00\x00\x00\x09"); await settle(5)    # count 9, no packets
    ws.push_audio(b"\x03\x07"); await settle(5)
    errs = ws.json_of("error")
    assert len(errs) == 1 and errs[0]["code"] == "bad_frame"
    assert not s._fatal.is_set()
    ws.hang_up(); await task


async def test_an_oversized_message_is_dropped_before_the_queue():
    s, ws, _, _, tr = make()
    task = await start(s, ws)
    ws.push_audio(b"\x03" + b"\x00" * 70_000); await settle(5)
    assert ws.json_of("error")[0]["code"] == "bad_frame" and s._audio_q_bytes == 0
    assert tr.stats["bad_frames"] >= 1
    ws.hang_up(); await task


async def test_the_uplink_queue_is_byte_bounded_and_a_replay_has_its_own_allowance():
    """C.2/A11: live audio that finds the 64 KB queue full is dropped at once; a replay
    (flags bit1) is queued without waiting, within REPLAY_QUEUE_BYTES (review-T3s4 #8: the
    old ≤ 5 s wait for room blocked the receive loop)."""
    s, _ws, _, _, _ = make()
    s._audio_q_bytes = vs.AUDIO_QUEUE_BYTES - 100          # as if the worker were behind
    live = up_pcm(0, 320)
    await s._enqueue_audio(live)
    assert s._dropped_audio == 1 and s._audio_q.empty()     # live audio: dropped at once
    replay = up_pcm(0, 320, flags=wire3.FLAG_REPLAY)
    await asyncio.wait_for(s._enqueue_audio(replay), 0.05)  # never waits
    assert s._audio_q.qsize() == 1 and s._dropped_audio == 1 and s._replay_q_bytes == len(replay)


async def test_local_speaking_flag_raises_the_vad_threshold():
    det = RecordingDetector([])
    det.ep.cfg.threshold = 0.5
    s, ws, _, _, _ = make(det=det)
    task = await start(s, ws)
    ws.push_audio(up_pcm(0, 320, flags=wire3.FLAG_LOCAL_SPEAKING)); await settle(5)
    assert det.ep.cfg.threshold == pytest.approx(0.65)
    ws.push_audio(up_pcm(320, 320)); await settle(5)
    assert det.ep.cfg.threshold == pytest.approx(0.5)
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# the commit cursor
# --------------------------------------------------------------------------- #

async def test_the_cursor_commits_before_the_host_call_and_answers_a_reconnect(tmp_path):
    cur = CommitCursor(tmp_path)
    seen_at_submit = []
    host = Host()
    orig = host.build

    def build(**kw):
        h = orig(**kw)
        inner = h.submit

        async def submit(tid, text, opts):
            seen_at_submit.append(cur.get((DEVICE, STREAM)))
            return await inner(tid, text, opts)
        h.submit = submit
        return h
    host.build = build
    det = RecordingDetector([[], [SpeechStart(), utterance()]])
    s, ws, _, _, _ = make(det=det, cursor=cur, host=host)
    task = await start(s, ws)
    ws.push_audio(up_pcm(1000, 320)); await settle(5)
    ws.push_audio(up_pcm(1320, 320)); await settle(20)
    upto = ws.json_of("eot")[0]["upto"]
    assert upto == 1640 and seen_at_submit == [1640]           # committed BEFORE the host call
    e = cur.entry((DEVICE, STREAM))
    assert e.upto == 1640 and e.meta == {"text": "what's the weather", "message_id": "u1"}
    assert e.open_from is None                                  # the submit covered the open mark
    ws.hang_up(); await task
    # The phone reconnects without having seen the transcript: its own cursor is older.
    s2, ws2, _, host2, _ = make(cursor=cur)
    task2 = await start(s2, ws2, cursor={"stream": STREAM, "from": 1320})
    assert ws2.json_of("cursor") == [{"type": "cursor", "from": 1640}]
    tr = ws2.json_of("transcript")
    assert tr == [{"type": "transcript", "text": "what's the weather", "merged": False,
                   "message_id": "u1", "upto": 1640, "replayed": True}]
    # A re-sent start with the same cursor is a settings re-send: no second answer (B4).
    ws2.push_json({"type": "start", "cursor": {"stream": STREAM, "from": 1320}}); await settle(5)
    assert len(ws2.json_of("cursor")) == 1 and not host2.of("submit")
    ws2.hang_up(); await task2


async def test_a_drop_mid_utterance_resumes_from_the_utterance_start(tmp_path):
    cur = CommitCursor(tmp_path)
    det = RecordingDetector([[], [], [SpeechStart()]])
    s, ws, _, _, _ = make(det=det, cursor=cur)
    task = await start(s, ws)
    for i in range(3):
        ws.push_audio(up_pcm(5000 + 320 * i, 320)); await settle(5)
    det.ep._frames = [np.zeros(512, np.float32)] * 1            # one VAD frame of utterance so far
    # (the SpeechStart above was handled before the frames were set: mark again as the
    # session would on a later SpeechStart; the stored mark must not move backwards)
    s._cursor_open()
    e = cur.entry((DEVICE, STREAM))
    assert e.open_from is not None and e.upto is None
    ws.hang_up(); await task
    # the new session is told to replay from the open utterance's start, below the client's from
    assert cur.resume_from((DEVICE, STREAM), 5000 + 960) == e.open_from


async def test_discarded_speech_clears_the_open_mark(tmp_path):
    cur = CommitCursor(tmp_path)
    det = RecordingDetector([[SpeechStart()], [Discarded(0.1)]])
    s, ws, _, _, _ = make(det=det, cursor=cur)
    task = await start(s, ws)
    ws.push_audio(up_pcm(0, 320)); await settle(5)
    assert cur.entry((DEVICE, STREAM)).open_from is not None
    ws.push_audio(up_pcm(320, 320)); await settle(5)
    assert cur.entry((DEVICE, STREAM)).open_from is None
    ws.hang_up(); await task


async def test_no_stream_id_means_no_cursor_and_no_cursor_answer(tmp_path):
    s, ws, _, _, tr = make(q=query(s="bad"))
    assert tr.stream is None and s.cursor is None
    task = await start(s, ws, cursor={"stream": "bad", "from": 0})
    assert not ws.json_of("cursor")
    ws.hang_up(); await task


def test_wire_options_keys_the_cursor_by_device_or_a_session_hash_never_the_token(tmp_path, monkeypatch):
    from dispatch_voice.host import context
    from dispatch_voice.host import ws as ws_mod
    paths = SimpleNamespace(state_dir=tmp_path / "state")
    monkeypatch.setattr(context, "get_context", lambda: SimpleNamespace(paths=paths))
    st = settings()
    sock = SimpleNamespace(token="fake-test-token")  # scrub-ok: test fixture, not a credential
    out = ws_mod.wire_options(FakeWS(query=query()), sock, st, FakeTTS())
    assert out["transport"].v3 and out["cursor_key"] == (DEVICE, STREAM) and callable(out["relay"])
    out = ws_mod.wire_options(FakeWS(query=query(d="x")), sock, st, FakeTTS())   # invalid device id
    slot, stream = out["cursor_key"]
    assert stream == STREAM and "secret" not in slot
    assert slot == "s-" + hashlib.sha256(b"fake-test-token").hexdigest()[:24]
    out = ws_mod.wire_options(FakeWS(query={}), sock, st, FakeTTS())             # v2
    assert not out["transport"].v3 and "cursor" not in out and "relay" not in out
    out = ws_mod.wire_options(FakeWS(query=query(s="!")), sock, st, FakeTTS())   # no stream id
    assert "cursor" not in out


# --------------------------------------------------------------------------- #
# reconnect mid-reply: after_partial and the late-joiner belt (C.5 #11)
# --------------------------------------------------------------------------- #

async def test_after_partial_skips_the_pieces_heard_on_the_old_socket():
    s, ws, tts, _, _ = make()
    task = await start(s, ws, after_partial={"message_id": "run:1", "sentences": 2})
    # the subscribe snapshot of the still-open stream, then the rest of it
    s.on_app_frame(frame(type="stream_start", message_id="run:1"))
    s.on_app_frame(frame(type="stream_chunk", message_id="run:1", replace=True,
                         text="You have two things this afternoon. A dentist appointment at three. Want"))
    s.on_app_frame(frame(type="stream_chunk", message_id="run:1", text=" a reminder?"))
    s.on_app_frame(frame(type="stream_done", message_id="m1", provisional_id="run:1",
                         message={"id": "m1", "role": "assistant", "content": REPLY}))
    await settle(40)
    assert tts.said == ["Want a reminder?"]
    assert [m["text"] for m in ws.json_of("say")] == ["Want a reminder?"]
    assert ws.json_of("say_end")[0]["last_seq"] == 1
    ws.hang_up(); await task


async def test_a_reply_heard_in_full_before_the_drop_still_gets_its_say_end():
    s, ws, tts, _, _ = make()
    task = await start(s, ws, after_partial={"message_id": "run:1", "sentences": 3})
    for f in reply("m1", REPLY, prov="run:1"):
        s.on_app_frame(f)
    await settle(30)
    assert tts.said == [] and not ws.json_of("say")
    assert ws.json_of("say_end") and ws.json_of("say_end")[0]["message_id"] == "m1"
    ws.hang_up(); await task


async def test_late_joiner_without_a_snapshot_speaks_the_reply_once_from_stream_done():
    s, ws, tts, _, _ = make()
    task = await start(s, ws)
    # deltas of a stream this session never saw start: a tail from mid-sentence
    s.on_app_frame(frame(type="stream_chunk", message_id="run:7", text="ntment at three. Want"))
    s.on_app_frame(frame(type="stream_chunk", message_id="run:7", text=" a reminder?"))
    await settle(20)
    assert tts.said == [] and not ws.json_of("say")
    s.on_app_frame(frame(type="stream_done", message_id="m7", provisional_id="run:7",
                         message={"id": "m7", "role": "assistant", "content": REPLY}))
    await settle(40)
    assert tts.said == ["You have two things this afternoon.", "A dentist appointment at three.",
                        "Want a reminder?"]
    ws.hang_up(); await task


async def test_a_snapshot_catches_a_late_reply_up_and_its_deltas_count_again():
    s, ws, tts, _, _ = make()
    task = await start(s, ws)
    s.on_app_frame(frame(type="stream_chunk", message_id="run:8", text="ignored tail"))
    s.on_app_frame(frame(type="stream_chunk", message_id="run:8", replace=True,
                         text="You have two things this afternoon. A dent"))
    s.on_app_frame(frame(type="stream_chunk", message_id="run:8", text="ist appointment at three. "))
    await settle(40)
    assert tts.said == ["You have two things this afternoon.", "A dentist appointment at three."]
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# link, probes, barge-in
# --------------------------------------------------------------------------- #

@needs_opus
async def test_link_rung_and_decoder_fallback_apply_at_the_next_sentence():
    s, ws, _, _, tr = make(codec="opus")
    task = await start(s, ws)
    ws.push_json({"type": "link", "mode": "audio", "down_kbps": 16, "why": "goodput"}); await settle(3)
    assert tr.link.down_kbps == 24                       # pending until a sentence boundary
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": "Hello there."}))
    await settle(30)
    assert tr.link.down_kbps == 16 and tr.codec_out == "opus"
    ws.push_json({"type": "link", "mode": "audio", "down_kbps": None, "why": "decoder"}); await settle(3)
    s.on_app_frame(frame(type="message", message={"id": "m2", "role": "assistant", "content": "Bye now."}))
    await settle(30)
    assert tr.codec_out == "pcm16"
    last = [b for b in binaries(ws) if b[3] == 2]
    assert last and all(b[0] == wire3.CODEC_PCM16 for b in last)
    ws.hang_up(); await task


async def test_a_probe_is_answered_with_one_kind_3_message_off_the_receive_loop():
    s, ws, _, _, _ = make()
    task = await start(s, ws)
    ws.push_json({"type": "probe", "bytes": 4096}); await settle(10)
    probes = [b for b in binaries(ws) if b[1] == wire3.KIND_PROBE]
    assert len(probes) == 1 and len(probes[0][5][0]) == 4096
    ws.push_json({"type": "probe", "bytes": 4096}); await settle(10)     # ≤ 1 per 20 s
    assert len([b for b in binaries(ws) if b[1] == wire3.KIND_PROBE]) == 1
    ws.hang_up(); await task


async def test_barge_in_drops_the_cut_gens_audio_on_the_wire():
    s, ws, _, _, _tr = make(tts=FakeTTS(chunks=400))
    task = await start(s, ws)
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": REPLY}))
    while not binaries(ws):
        await asyncio.sleep(0.002)
    ws.push_json({"type": "stop_playback"}); await settle(20)
    gen = ws.json_of("barge_in")[0]["gen"]
    assert gen == 1
    i = next(i for i, m in enumerate(ws.sent) if isinstance(m, dict) and m.get("type") == "barge_in")
    late = [wire3.unpack_down(b) for b in ws.sent[i + 1:] if isinstance(b, bytes)]
    assert all(b[2] >= gen for b in late)                 # nothing of gen 0 after the barge_in
    ws.hang_up(); await task


async def test_first_audio_metrics_carry_the_link_view_on_v3():
    s, ws, _, _, _ = make()
    s._eot_at = s.clock()
    s._first_audio_logged = False
    task = await start(s, ws)
    s._first_audio_logged = False
    s._eot_at = s.clock()
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": "Hello there."}))
    await settle(30)
    m = [x for x in ws.json_of("metrics") if "eot_to_first_audio_ms" in x]
    assert m and m[0]["link"]["mode"] == "audio" and m[0]["link"]["down_kbps"] == 24
    ws.hang_up(); await task


async def test_the_open_mark_is_the_real_detectors_utterance_start_with_its_pre_roll(tmp_path):
    """The real TurnDetector + Endpointer (an energy stand-in for Silero): speech
    starting at offset 16 000 + 3 200 is marked open at its start minus the 0.3 s pre-roll
    (rounded to the 512-sample VAD frame), never after the speech began."""
    from dispatch_voice.host.vad_turn import Endpointer, EndpointerConfig, TurnDetector
    cfg = EndpointerConfig(min_speech_s=0.1, pre_roll_s=0.3)
    det = TurnDetector(lambda frame: 1.0 if float(np.abs(frame).mean()) > 0.05 else 0.0, Endpointer(cfg))
    cur = CommitCursor(tmp_path)
    s, ws, _, _, _ = make(det=det, cursor=cur)
    task = await start(s, ws)
    base = 16000
    ws.push_audio(up_pcm(base, 3200)); await settle(5)                       # 0.2 s of silence
    for i in range(10):                                                       # 0.4 s of "speech"
        ws.push_audio(up_pcm(base + 3200 + 640 * i, 640, value=8000)); await settle(3)
    e = cur.entry((DEVICE, STREAM))
    speech_at = base + 3200
    assert e is not None and e.open_from is not None
    assert speech_at - 4800 - 512 <= e.open_from <= speech_at
    ws.hang_up(); await task
