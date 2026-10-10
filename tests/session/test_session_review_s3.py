"""Regression tests for the slice-3 review (impl/review-T3s3.md), fixed in T3 slice 4:
the reviewer's probes inverted (each must now merge and speak), plus the per-utterance
snapshot and the streaming-STT boundary they asked for. Fakes only (no models)."""
from __future__ import annotations

import asyncio

import numpy as np
from fakes import FakeSTT, FakeTTS, FakeWS

from dispatch_voice.host import session as vs
from dispatch_voice.host.session import VoiceSession
from dispatch_voice.host.vad_turn import (
    Endpointer,
    EndpointerConfig,
    Pause,
    SpeechStart,
)
from dispatch_voice.rt.endpoint import TurnPolicy
from dispatch_voice.rt.settings import RtSettings
from dispatch_voice.rt.stt_stream import ParakeetTailSTT
from session.test_session import THREAD, kit_with_lines, settings, settle
from session.test_session_s2 import frame, reply
from session.test_session_s3 import (
    QUESTION,
    REPLY,
    Clock,
    FakeStreamingSTT,
    Host,
    ProbeDetector,
    addendum_session,
    make,
    over_reply,
    partial,
    pause,
    play,
    speak,
    start,
)

A = "What's on my calendar this afternoon?"
B = "and tomorrow."


async def _until(cond, timeout: float = 6.0) -> bool:
    t = asyncio.get_running_loop().time()
    while not cond():
        if asyncio.get_running_loop().time() - t > timeout:
            return False
        await asyncio.sleep(0.01)
    return True


# --------------------------------------------------------------------------- #
# #1 (HIGH): speech that starts during a turn end and ends before it finishes
# --------------------------------------------------------------------------- #

async def test_an_addendum_spoken_and_ended_during_a_slow_submit_merges_and_its_reply_is_spoken():
    sstt = FakeStreamingSTT([partial(A), partial(B)])
    host = Host(submit_delay_s=1.0)
    s, ws, _tts, host, _det = make([[SpeechStart()], [pause()], [SpeechStart()], [pause()]],
                                   p=(0.95, 0.95), sstt=sstt, host=host, backend="gateway")
    task = await start(s, ws)
    await speak(ws, 2, wait=20)                       # A ends; its submit takes 1 s
    assert await _until(lambda: len(host.of("submit")) == 1)
    await speak(ws, 2, wait=20)                       # B starts and ends inside that second
    assert await _until(lambda: len(host.of("submit")) == 2, timeout=8)
    await asyncio.sleep(1.2)
    merged = host.of("submit")[1][1]
    assert merged.startswith("What's on my calendar this afternoon") and merged.endswith(B)
    assert not s._hold                                # nothing left to hold for
    s.on_app_frame(frame(type="message", message={"id": "u2", "role": "user", "content": merged}))
    s.on_app_frame(frame(type="thinking", status="started"))
    for f in reply("m9", "Two things today. Nothing tomorrow.", prov="run:9"):
        s.on_app_frame(f)
    await asyncio.sleep(0.5)
    assert " ".join(m["text"] for m in ws.json_of("say")) == "Two things today. Nothing tomorrow."
    ws.hang_up(); await task


async def test_speech_begun_before_the_submit_and_ended_during_it_merges_instead_of_replacing():
    sstt = FakeStreamingSTT([partial(A), partial(B)], final_delay_s=0.4)
    host = Host(submit_delay_s=1.0)
    s, ws, _tts, host, _det = make([[SpeechStart()], [pause()], [SpeechStart()], [pause()]],
                                   p=(0.95, 0.95), sstt=sstt, host=host, backend="gateway")
    task = await start(s, ws)
    await speak(ws, 2, wait=5)
    assert await _until(sstt.final_started.is_set)
    await speak(ws, 1, wait=5)                        # B starts inside A's final()
    assert await _until(lambda: len(host.of("submit")) == 1)
    await speak(ws, 1, wait=20)                       # ... and pauses during A's submit
    assert await _until(lambda: len(host.of("submit")) == 2, timeout=8)
    await asyncio.sleep(1.2)
    assert ws.json_of("transcript")[-1]["merged"] is True
    assert ("retract", "u1") in host.calls and not s._hold
    ws.hang_up(); await task


async def test_a_queued_turn_end_keeps_its_own_stop_cut_and_aborts_with_the_partial():
    """#1(c): the stop-cut utterance's end is still running (a slow `final()`) when the
    next utterance begins; that SpeechStart resets the OPEN utterance's flags — the
    ending one keeps its own, so its cut run is aborted keeping the spoken prefix
    (keep_partial=True), never abandoned without it."""
    clock = Clock()
    sstt = FakeStreamingSTT([], ticks=[partial("wait stop")], final="wait stop no the other one",
                            final_delay_s=0.4)
    _s, ws, _tts, host, _det, task = await over_reply([[SpeechStart()], [], [pause()], [SpeechStart()]],
                                                   words="x", sstt=sstt, clock=clock, backend="gateway")
    ws.push_audio(); await settle(5)                  # A: speech over the reply
    clock.t += 0.6
    ws.push_audio(); await settle(20)                 # the tick reads "wait stop": the stop cut
    assert ws.json_of("barge_in") and not host.of("abort")
    ws.push_audio(); await settle(5)                  # A's pause → end → final() (0.4 s)
    assert await _until(sstt.final_started.is_set)
    ws.push_audio(); await settle(5)                  # B begins during A's end
    assert await _until(lambda: bool(host.of("submit")))
    assert ("abort", True) in host.calls and ("abort", False) not in host.calls
    assert host.of("submit")[0][1] == "wait stop no the other one"
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# #2 (MEDIUM): the next utterance's words while a turn end reads the stream
# --------------------------------------------------------------------------- #

WORDS = {1000: "wait", 2000: "tuesday", 3000: "and", 4000: "thursday", 5000: "afternoon"}


class WordRecognizer:
    """Reads 'words' from constant sample values (the review's SlowRecognizer)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, audio: np.ndarray) -> str:
        v = np.round(np.asarray(audio) * 32768).astype(int)
        out: list[str] = []
        for x in v[np.nonzero(v)]:
            w = WORDS.get(int(x))
            if w and (not out or out[-1] != w):
                out.append(w)
        text = (" ".join(out) + ".") if out else ""
        self.calls.append(text)
        return text


class MarkingDetector(ProbeDetector):
    """ProbeDetector that hands every frame of an open utterance to the frame sink and
    marks the utterance's END in it (``None``), as the real Endpointer does."""

    def feed(self, pcm):
        events = super().feed(pcm)
        if self.active or any(isinstance(e, SpeechStart) for e in events):
            a = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0
            if self.frame_sink is not None and a.size:
                self.frame_sink(a, 0.9 if np.any(a) else 0.0)
        return events

    def end_turn(self, reason, prob=None):
        te = super().end_turn(reason, prob)
        if te is not None and self.frame_sink is not None:
            self.frame_sink(None, -1.0)
        return te


def word(v: int, n: int = 1) -> bytes:
    return np.full(320 * n, v, "<i2").tobytes()


async def test_words_spoken_while_a_turn_end_reads_the_stream_land_in_the_next_turn_whole():
    """The review's probe with the end mark: "wait tuesday" (a tick stop cut), its end runs
    a 1 s gateway abort; "and thursday" is spoken during it, "thursday afternoon" after.
    Nothing is lost and nothing is filed under the first turn."""
    rec = WordRecognizer()
    sstt = ParakeetTailSTT(rec)
    script = [[SpeechStart()], [], [], [Pause(np.zeros(16000, np.float32), 0.25, 1.0)],
              [SpeechStart()], [], [], [Pause(np.zeros(16000, np.float32), 0.25, 1.0)]]
    det = MarkingDetector(script, p=(0.95,))
    host = Host()
    vh = host.build()
    orig_abort = vh.abort
    abort_started = asyncio.Event()

    async def slow_abort(tid, keep_partial=False):
        abort_started.set()
        await asyncio.sleep(1.0)
        return await orig_abort(tid, keep_partial=keep_partial)

    vh.abort = slow_abort
    ws = FakeWS()
    rts = RtSettings.from_mapping({})
    s = VoiceSession(ws, thread_id=THREAD, settings=settings(merge_window_s=0.0, ack_after_s=0.6),
                     detector=det, stt_engine=FakeSTT("unused"), tts_engine=FakeTTS(), host=vh,
                     notice=lambda text: np.ones(48, "<i2"), heard_wait_s=0.3, sstt=sstt,
                     policy=TurnPolicy(rts), rt=rts, bot_backend="gateway")
    s.watch_tick_s = 0.01
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start", "acks": "off"}); await settle(5)
    s._current = vs._Turn(msg_id="u0", text="before", at=s.clock()); s._owner = s._current
    await play(s, ws, REPLY)
    for v in (1000, 2000):
        ws.push_audio(word(v, 4)); await settle(4)
    await asyncio.sleep(0.6); await settle(4)
    ws.push_audio(word(2000, 2)); await settle(20)          # the tick reads "wait tuesday."
    assert s._stop_cut
    ws.push_audio(b"\x00\x00" * 320); await settle(10)      # A's pause → end (abort 1 s)
    await asyncio.wait_for(abort_started.wait(), 3)
    ws.push_audio(word(3000, 4)); await settle(10)          # B during A's end: "and"
    ws.push_audio(word(4000, 4)); await settle(10)          # "thursday"
    await asyncio.sleep(1.2); await settle(20)
    ws.push_audio(word(4000, 4)); await settle(10)          # B goes on: "thursday"
    ws.push_audio(word(5000, 4)); await settle(10)          # "afternoon" + its pause → end
    assert await _until(lambda: len(host.of("submit")) >= 2, timeout=6)
    await asyncio.sleep(0.3)
    subs = [c[1] for c in host.of("submit")]
    assert subs[0] == "wait tuesday."
    assert subs[-1] == "wait tuesday and thursday afternoon."
    assert ws.json_of("transcript")[-1]["merged"] is True
    ws.hang_up(); await task


def test_the_endpointer_marks_an_utterance_end_in_the_frame_sink_in_order():
    seen: list = []
    ep = Endpointer(EndpointerConfig(min_speech_s=0.064, turn_hard_silence_s=0.128, probe_events=True))
    ep.frame_sink = lambda fr, p: seen.append(None if fr is None else round(float(p), 2))
    f = np.zeros(512, np.float32)
    events = []
    for prob in (0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.9):
        events += ep.process(f, prob)
    names = [type(e).__name__ for e in events]
    assert "TurnEnd" in names
    i = seen.index(None)
    assert seen[:i] == [0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0][:i]      # the utterance's frames first
    assert seen.count(None) == 1                                   # one mark per end
    # a discarded blip and an external reset never mark an end
    seen.clear()
    ep.reset()
    for prob in (0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0):
        ep.process(f, prob)
    assert None not in seen


async def test_a_discard_while_a_turn_end_reads_the_stream_keeps_the_ending_turns_frames():
    """An echo verdict on the NEW utterance during the previous end drops only the new
    utterance's waiting frames; the ending turn's own stream is untouched."""
    sstt = FakeStreamingSTT([partial(A)])
    s, _ws, _tts, _host, _det = make([], sstt=sstt)
    pushed = []
    s.sstt = type("Rec", (), {"push": lambda self, fr, vad_prob: pushed.append(fr), "reset": lambda self: pushed.clear()})()
    fr = np.ones(512, np.float32)
    s._on_frame(fr, 0.9)                     # A's frame → the stream
    s._on_frame(None, -1.0)                  # A ends (its turn end now reads the stream)
    s._on_frame(fr * 2, 0.9)                 # B's frame waits
    assert len(pushed) == 1 and s._sink_busy()
    s._sstt_drop_open()                      # B is dropped (an echo verdict)
    assert len(pushed) == 1                  # A's frame stays for A's final()
    s._sink_release()                        # A's end done: reset, nothing of B follows
    assert pushed == [] and not s._sink_busy()


# --------------------------------------------------------------------------- #
# #3 (MEDIUM): an acknowledgement over a listener who is still talking
# --------------------------------------------------------------------------- #

async def test_no_acknowledgement_while_the_listener_is_still_talking():
    kit = kit_with_lines({"question": ("Let me see.",)}, first_chunk_s=2.0)
    sstt = FakeStreamingSTT([partial(QUESTION)])
    s, ws, _tts, host, det = make([[SpeechStart()], [pause()], [SpeechStart()], []], p=(0.95,),
                                  sstt=sstt, acks=kit, backend="gateway")
    task = await start(s, ws, acks="often")
    await speak(ws, 2, wait=20)                       # A ends at the comma → submit, acks armed
    assert host.of("submit")
    await asyncio.sleep(0.15)
    await speak(ws, 1, wait=5)                        # the listener goes on (addendum candidate)
    assert s._hold and det.active
    await asyncio.sleep(0.6); await settle(10)        # past ack_after_s
    assert not [m for m in ws.json_of("say") if m.get("kind") == "ack"]
    assert not ws.json_of("ack")
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# #4 (MEDIUM): a tick echo verdict never throws away the listener's utterance
# --------------------------------------------------------------------------- #

async def test_a_tick_echo_resumes_the_reply_but_keeps_the_listeners_utterance():
    """The listener starts by quoting the bot: the first tick reads only the quote (an
    echo) — the reply resumes, but the utterance stays open; the words at the pause are
    a real interruption and become the turn."""
    words = "A dentist appointment at three? Can we move it to four?"
    sstt = FakeStreamingSTT([partial(words)], ticks=[partial("A dentist appointment")], final=words)
    _s, ws, _tts, host, det, task = await over_reply([[SpeechStart()], [], [], [pause()]],
                                                   words="x", sstt=sstt, backend="gateway")
    for _ in range(3):
        ws.push_audio(); await settle(10)
    assert det.discards == 0 and ws.json_of("resume")            # the bleed resumed the reply ...
    assert not ws.json_of("echo_ignored") and det.active         # ... the utterance goes on
    ws.push_audio(); await settle(80)                            # its pause: the full words decide
    assert ws.json_of("barge_in")
    assert [c[1] for c in host.of("submit")] == [words]
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# #5, #7, #9, #10 (LOW)
# --------------------------------------------------------------------------- #

async def test_a_long_addendum_over_early_audio_still_merges():
    clock = Clock()
    s, ws, _tts, _host, _det = addendum_session(clock=clock)
    task = await start(s, ws)
    await speak(ws, 2, wait=40)
    s.on_app_frame(frame(type="message", message={"id": "u1", "role": "user", "content": QUESTION}))
    clock.t += 1.9
    await play(s, ws)
    clock.t += 0.15
    ws.push_audio(); await settle(10)
    assert s._hold and ws.json_of("duck")[-1]["on"] is True
    clock.t += 2.1                                     # the addendum is still being spoken
    await settle(20)
    assert s._hold and not ws.json_of("resume")
    ws.push_audio(); await settle(60)
    assert ws.json_of("transcript")[-1]["merged"] is True
    ws.hang_up(); await task


async def test_a_malformed_frame_never_ends_the_chat():
    s, ws, _tts, _host, _det = make([])
    task = asyncio.create_task(s.run())
    ws.inbox.put_nowait({"type": "websocket.receive",
                         "text": '{"type": "start", "rt": {"learned": {"probe_ms": Infinity, "hold_ms": 600}}}'})
    await settle(20)
    assert not task.done()
    ws.push_json({"type": "ping", "t": 1}); await settle(5)
    assert ws.json_of("pong")
    ws.hang_up(); await task


async def test_a_hang_up_discards_a_speculative_run_still_going():
    sstt = FakeStreamingSTT([partial(QUESTION)], final_delay_s=0.5)
    s, ws, _tts, host, _det = make([[SpeechStart()], [pause()]], p=(0.95,), sstt=sstt)
    task = await start(s, ws)
    await speak(ws, 2, wait=5)
    assert await _until(lambda: bool(host.runs))
    ws.hang_up(); await task
    assert host.runs[0].discarded


async def test_no_speculation_once_the_session_is_no_longer_live():
    sstt = FakeStreamingSTT([partial(QUESTION)])
    live = {"ok": True}

    class LiveHost(Host):
        def build(self, **kw):
            return super().build(session_live=lambda: live["ok"], **kw)

    s, ws, _tts, host, _det = make([[SpeechStart()], [pause()]], p=(0.95,), sstt=sstt, host=LiveHost())
    task = await start(s, ws)
    live["ok"] = False
    await speak(ws, 2, wait=40)
    assert not host.of("speculate") and not host.of("submit")
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# #8: what an interruption cost
# --------------------------------------------------------------------------- #

async def test_interruption_metrics_measure_the_settle_and_the_wait_after_the_words():
    FakeStreamingSTT([partial("Actually, how about Thursday instead?")])
    _s, ws, _tts, _host, _det, task = await over_reply([[SpeechStart()], [pause()]],
                                                   words="Actually, how about Thursday instead?")
    ws.push_audio(); await settle(5)
    ws.push_audio(); await settle(80)
    m = next(x for x in ws.json_of("metrics") if "eot_to_transcript_ms" in x)
    assert "cut_settle_ms" in m and "interrupt_to_submit_ms" in m
    assert 0 <= m["interrupt_to_submit_ms"] < 2000
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# pins for the individual pieces of #1 and #2
# --------------------------------------------------------------------------- #

async def test_a_turn_end_never_rearms_the_hold_for_speech_that_is_gone():
    """#1(a): speech resumed during a slow submit, then the listener paused the mic (the
    detector reset, the speech is gone). The end's clean-up must not hold the reply."""
    sstt = FakeStreamingSTT([partial(A)])
    host = Host(submit_delay_s=0.6)
    s, ws, _tts, host, det = make([[SpeechStart()], [pause()], [SpeechStart()]], p=(0.95,),
                                  sstt=sstt, host=host, backend="gateway")
    task = await start(s, ws)
    await speak(ws, 2, wait=10)                         # A ends; its submit takes 0.6 s
    assert await _until(lambda: bool(host.of("submit")))
    await speak(ws, 1, wait=5)                          # B starts during the submit
    assert s._resumed_during_end and det.active
    ws.push_json({"type": "pause"}); await settle(5)    # the mic is paused: B is gone
    assert await _until(lambda: not s._ending, timeout=3)
    assert not s._hold                                  # at once, not after the watchdog
    ws.hang_up(); await task


async def test_a_turn_end_reads_its_own_stop_cut_after_final_even_if_new_speech_began():
    """#1(c): "Wait." over the reply (a tick's stop cut), its end runs a slow final(), and
    the listener starts again meanwhile — that SpeechStart resets the OPEN utterance's
    flags. The ending one is still "only a stop": nothing is submitted."""
    clock = Clock()
    sstt = FakeStreamingSTT([], ticks=[partial("wait")], final="Wait.", final_delay_s=0.4)
    s, ws, _tts, host, _det, task = await over_reply([[SpeechStart()], [], [pause()], [SpeechStart()]],
                                                   words="x", sstt=sstt, clock=clock, backend="gateway")
    ws.push_audio(); await settle(5)
    clock.t += 0.6
    ws.push_audio(); await settle(20)                   # the tick: stop cut
    assert ws.json_of("barge_in") and s._stop_cut
    ws.push_audio(); await settle(5)                    # its pause → end → final() (0.4 s)
    assert await _until(sstt.final_started.is_set)
    ws.push_audio(); await settle(5)                    # new speech during that final()
    await asyncio.sleep(0.6); await settle(20)
    assert not host.of("submit")                        # "Wait." alone was only the stop
    ws.hang_up(); await task


async def test_a_stop_cut_runs_its_abort_beside_final():
    """#2: the stop cut's run is aborted while final() reads the words, not before it
    (an inline gateway abort of up to 1.5 s made the next words land in the wrong turn)."""
    clock = Clock()
    sstt = FakeStreamingSTT([partial("wait stop, the other one")], ticks=[partial("wait stop")],
                            final="wait stop, the other one", final_delay_s=0.5)
    host = Host()
    vh_build = host.build

    def build(**kw):
        vh = vh_build(**kw)
        inner = vh.abort

        async def slow_abort(tid, keep_partial=False):
            await asyncio.sleep(0.5)
            return await inner(tid, keep_partial=keep_partial)
        vh.abort = slow_abort
        return vh
    host.build = build
    s, ws, _tts, host, _det, task = await over_reply([[SpeechStart()], [], [pause()]], words="x",
                                                   sstt=sstt, clock=clock, host=host, backend="gateway")
    ws.push_audio(); await settle(5)
    clock.t += 0.6
    ws.push_audio(); await settle(20)                   # the tick's stop cut
    assert s._stop_cut
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    ws.push_audio(); await settle(2)                    # the pause → end
    assert await _until(lambda: bool(host.of("submit")), timeout=3)
    took = loop.time() - t0
    assert took < 0.9, took                             # 0.5 s abort ‖ 0.5 s final, not 1.0 s
    assert ("abort", True) in host.calls
    ws.hang_up(); await task
