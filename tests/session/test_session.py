"""Voice chat: the voice session, with fakes.

No models, no DisPatch: the session is driven through a fake socket, a
scripted turn detector and fake STT/TTS engines (tests/fakes.py). The gate
(Origin, PIN, safe bot, slots) is DisPatch's; the add-on's serve() lives in
tests/host/test_ws_serve.py. Ported from DisPatch's test_voice_session.py.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest
from fakes import FakeSTT, FakeTTS, FakeWS, ScriptedDetector, utterance

from dispatch_voice.host.acks import AckKit
from dispatch_voice.host.config import VoiceSettings
from dispatch_voice.host.session import VoiceHost, VoiceSession
from dispatch_voice.host.vad_turn import SpeechStart
from dispatch_voice.text.acks import AckSelector, EtaEstimator, Pools
from settling import pause, settle, wait_until

THREAD = "t-1"


class _NoSkip:
    """An rng whose skip roll never skips and whose shuffle keeps order."""

    def random(self):
        return 0.99

    def shuffle(self, x):
        pass


class FakeRenderQueue:
    """Every line is rendered: `samples` int16 at 24 kHz; a `prune`/`ensure` no-op."""

    def __init__(self, voice: str, lines, *, samples: int = 100):
        self.voice = voice
        self.lines = {t: np.ones(samples, "<i2") for t in lines}
        self.rebuilt: list[str] = []

    def get(self, voice, text):
        return self.lines.get(text) if voice == self.voice else None

    def duration_s(self, voice, text):
        pcm = self.get(voice, text)
        return None if pcm is None else len(pcm) / 24000

    def status(self):
        return {self.voice: {"ready": len(self.lines), "total": len(self.lines), "failed": 0}}

    def rebuild(self, voice):
        self.rebuilt.append(voice)

    def ensure(self, voice, lines):
        pass

    def prune(self, voice, keep=None):
        return 0


def kit_with_lines(pools: dict, *, voice: str = "", mode: str = "often", samples: int = 100,
                   bot_id: str | None = None, first_chunk_s: float = 1.2) -> AckKit:
    """An acknowledgement kit whose every pool line is already rendered."""
    p = Pools(default={k: tuple(v) for k, v in pools.items()})
    lines = [t for v in pools.values() for t in v]
    return AckKit(AckSelector(p, mode=mode, rng=_NoSkip()), FakeRenderQueue(voice, lines, samples=samples),
                  voice, bot_id, EtaEstimator(first_chunk_s=first_chunk_s))


def settings(**kw):
    base = {"flag": "1", "ack_after_s": 0.05, "merge_window_s": 0.15}
    base.update(kw)
    return VoiceSettings(**base)


def make_session(script, stt_text="what's the weather", **host_kw):
    ws = FakeWS()
    submitted = []

    async def submit(tid, text):
        submitted.append((tid, text))
        return "m-user"

    host = VoiceHost(submit=submit, **host_kw)
    tts = FakeTTS()
    s = VoiceSession(ws, thread_id=THREAD, settings=settings(),
                     detector=ScriptedDetector(script), stt_engine=FakeSTT(stt_text),
                     tts_engine=tts, host=host)
    return s, ws, tts, submitted



# --------------------------------------------------------------------------- #
# Session behaviour
# --------------------------------------------------------------------------- #

async def test_turn_end_submits_transcript_through_host():
    s, ws, tts, submitted = make_session([[SpeechStart(), utterance()]])
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    ws.push_audio(); ws.push_audio()
    await settle()
    assert submitted == [(THREAD, "what's the weather")]
    assert [m["type"] for m in ws.sent if isinstance(m, dict)][:1] == ["ready"]
    assert ws.json_of("eot") and ws.json_of("transcript")[0]["text"] == "what's the weather"
    assert {"type": "state", "state": "thinking"} in ws.sent
    ws.hang_up(); await task


async def test_audio_is_ignored_until_start():
    s, ws, tts, submitted = make_session([[utterance()]])
    task = asyncio.create_task(s.run())
    ws.push_audio()
    await settle()
    assert submitted == [] and not ws.json_of("eot")
    ws.hang_up(); await task


async def test_empty_transcript_is_not_sent():
    s, ws, tts, submitted = make_session([[utterance()]], stt_text="")
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); ws.push_audio()
    await settle()
    assert submitted == [] and ws.json_of("transcript")[0]["text"] == ""
    ws.hang_up(); await task


async def test_ack_fires_when_reply_is_slow():
    s, ws, tts, submitted = make_session([[utterance()]])
    s.acks.selector.rng = _NoSkip()        # the Reactions roll is the selector's; pin it
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); ws.push_audio()
    await asyncio.sleep(0.15)
    assert ws.json_of("ack")
    ws.hang_up(); await task


async def test_streamed_reply_is_spoken_sentence_by_sentence_with_paired_binary():
    s, ws, tts, _ = make_session([])
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    await settle(5)
    for chunk in ("Sure. It is sun", "ny today. Bring", " a hat."):
        s.on_app_frame({"type": "stream_chunk", "thread_id": THREAD,
                        "message_id": "run:1", "text": chunk})
    s.on_app_frame({"type": "stream_done", "thread_id": THREAD, "provisional_id": "run:1",
                    "message_id": "m-9", "message": {"id": "m-9", "role": "assistant",
                    "content": "Sure. It is sunny today. Bring a hat."}})
    # The same reply re-announced as a plain message must not be spoken twice.
    s.on_app_frame({"type": "message", "thread_id": THREAD,
                    "message": {"id": "m-9", "role": "assistant", "content": "x"}})
    await settle()
    assert tts.said == ["Sure. It is sunny today.", "Bring a hat."]
    # Every audio header is immediately followed by its bytes.
    for i, m in enumerate(ws.sent):
        if isinstance(m, dict) and m.get("type") == "audio":
            assert isinstance(ws.sent[i + 1], bytes)
    assert ws.json_of("audio")[-1]["message_id"] == "m-9"
    ws.hang_up(); await task


async def test_other_threads_users_and_sub_rows_are_not_spoken():
    s, ws, tts, _ = make_session([])
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    s.on_app_frame({"type": "message", "thread_id": "other",
                    "message": {"id": "a", "role": "assistant", "content": "Nope."}})
    s.on_app_frame({"type": "message", "thread_id": THREAD,
                    "message": {"id": "b", "role": "user", "content": "Mine."}})
    s.on_app_frame({"type": "message", "thread_id": THREAD,
                    "message": {"id": "c", "role": "assistant", "content": "Working.",
                                "metadata": {"sub": True}}})
    s.on_app_frame({"type": "message", "thread_id": THREAD,
                    "message": {"id": "d", "role": "assistant", "content": "Yes."}})
    await settle()
    assert tts.said == ["Yes."]
    ws.hang_up(); await task


async def test_progress_frames_are_ignored_for_now():
    """B.1a passes `progress` through the tap; until the tool-status lines
    (slice 2) it is neither spoken nor an error."""
    s, ws, tts, _ = make_session([])
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    s.on_app_frame({"type": "progress", "thread_id": THREAD, "item": {"kind": "tool", "name": "brave_search"}})
    await settle()
    assert tts.said == [] and not ws.json_of("error") and not ws.json_of("turn_error")
    ws.hang_up(); await task


async def test_replace_chunks_do_not_duplicate_speech():
    s, ws, tts, _ = make_session([])
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    s.on_app_frame({"type": "stream_chunk", "thread_id": THREAD, "message_id": "r",
                    "text": "One. Two"})
    s.on_app_frame({"type": "stream_chunk", "thread_id": THREAD, "message_id": "r",
                    "text": "One. Two. Three.", "replace": True})
    s.on_app_frame({"type": "stream_done", "thread_id": THREAD, "message_id": "r",
                    "message": {"id": "r", "role": "assistant", "content": "One. Two. Three."}})
    await settle()
    assert tts.said == ["One. Two. Three."]
    ws.hang_up(); await task


async def test_user_speech_while_speaking_ducks_then_barges_in_once_words_are_real():
    """H3: speech over the reply only DUCKS it; the reply is cut (and the run
    aborted) once the transcript proves it is the driver, not our echo."""
    aborted = []

    async def abort(tid):
        aborted.append(tid)

    s, ws, tts, submitted = make_session([[SpeechStart()], [utterance()]],
                                         stt_text="stop, what about tomorrow", abort=abort)
    s.settings = settings(barge_abort_turn=True)
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    s.on_app_frame({"type": "message", "thread_id": THREAD,
                    "message": {"id": "m1", "role": "assistant", "content": "A long answer."}})
    await settle()
    assert s.speaking
    gen_before = s.gen
    ws.push_audio()                                  # speech starts over the reply
    await settle()
    assert ws.json_of("duck") == [{"type": "duck", "on": True, "gen": gen_before}]
    assert not ws.json_of("barge_in") and aborted == []      # nothing cut yet
    ws.push_audio()                                  # ...and it was real words
    await settle()
    barge = ws.json_of("barge_in")
    assert barge and barge[0]["gen"] == gen_before + 1
    assert aborted == [THREAD] and not s.speaking
    assert submitted == [(THREAD, "stop, what about tomorrow")]
    # Late chunks of the barged reply stay silent.
    n = len(tts.said)
    s.on_app_frame({"type": "stream_done", "thread_id": THREAD, "message_id": "m1",
                    "message": {"id": "m1", "role": "assistant", "content": "A long answer. More."}})
    await settle()
    assert len(tts.said) == n
    ws.hang_up(); await task


async def test_echo_over_the_reply_unducks_and_never_aborts():
    """H3: our own voice leaking into the mic must not kill the reply."""
    aborted = []

    async def abort(tid):
        aborted.append(tid)

    s, ws, tts, submitted = make_session([[SpeechStart()], [utterance()]],
                                         stt_text="A long answer about the weather", abort=abort)
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    s.on_app_frame({"type": "message", "thread_id": THREAD, "message": {
        "id": "m1", "role": "assistant",
        "content": "A long answer about the weather. And then a second sentence follows."}})
    await settle()
    ws.push_audio(); await settle()
    assert ws.json_of("duck")[-1]["on"] is True
    ws.push_audio(); await settle()
    assert ws.json_of("echo_ignored")
    assert ws.json_of("duck")[-1] == {"type": "duck", "on": False, "gen": 0}
    assert not ws.json_of("barge_in") and aborted == [] and submitted == []
    assert {"type": "state", "state": "speaking"} in ws.sent[-3:]
    ws.hang_up(); await task


async def test_barge_in_off_ignores_speech_while_speaking():
    s, ws, tts, submitted = make_session([[SpeechStart()], [utterance()]])
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start", "barge_in": False})
    s.on_app_frame({"type": "message", "thread_id": THREAD,
                    "message": {"id": "m1", "role": "assistant", "content": "A long answer."}})
    await settle()
    ws.push_audio(); ws.push_audio(); await settle()
    assert not ws.json_of("duck") and not ws.json_of("barge_in") and submitted == []
    assert len(s.detector.script) == 2               # the mic was not even fed
    ws.hang_up(); await task


async def test_speaking_raises_min_speech_for_barge_in_and_restores_it():
    s, ws, tts, _ = make_session([])
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    normal = s.detector.ep.cfg.min_speech_s
    s.on_app_frame({"type": "message", "thread_id": THREAD,
                    "message": {"id": "m1", "role": "assistant", "content": "Hi."}})
    await settle()
    assert s.detector.ep.cfg.min_speech_s == s.settings.barge_in_speech_s
    ws.push_json({"type": "playback_done", "gen": s.gen})
    await settle()
    assert s.detector.ep.cfg.min_speech_s == normal and not s.speaking
    ws.hang_up(); await task


async def test_own_voice_echo_is_not_submitted():
    s, ws, tts, submitted = make_session([[utterance()]], stt_text="It is sunny today")
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    s.on_app_frame({"type": "message", "thread_id": THREAD,
                    "message": {"id": "m1", "role": "assistant", "content": "It is sunny today."}})
    await settle()
    ws.push_audio()
    await settle()
    assert submitted == [] and ws.json_of("echo_ignored")
    ws.hang_up(); await task


async def test_lapsed_session_locks_and_ends():
    live = {"ok": True}
    s, ws, tts, _ = make_session([], session_live=lambda: live["ok"])
    t = {"now": 0.0}
    s.clock = lambda: t["now"]
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); await settle(5)
    live["ok"] = False
    t["now"] = 5.0
    ws.push_audio()
    await asyncio.wait_for(task, 1.0)
    assert ws.json_of("locked")


async def test_resume_after_reconnect_speaks_missed_replies_once():
    async def replies_after(tid, after):
        assert (tid, after) == (THREAD, "m-old")
        return [("m-a", "First."), ("m-b", "Second.")]

    s, ws, tts, _ = make_session([], replies_after=replies_after)
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start", "after": "m-old"})
    await settle()
    s.on_app_frame({"type": "message", "thread_id": THREAD,
                    "message": {"id": "m-b", "role": "assistant", "content": "Second."}})
    await settle()
    assert tts.said == ["First.", "Second."]
    ws.hang_up(); await task


async def test_stop_playback_and_pause():
    s, ws, tts, _ = make_session([[utterance()]])
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); ws.push_json({"type": "pause"})
    ws.push_audio(); await settle()
    assert not ws.json_of("eot") and {"type": "state", "state": "paused"} in ws.sent
    ws.push_json({"type": "stop_playback"}); await settle()
    assert ws.json_of("barge_in")
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# Turn merge, patience, acks, voice
# --------------------------------------------------------------------------- #

class SeqSTT:
    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, audio):
        return self.texts.pop(0) if self.texts else ""


def merge_session(texts, script, **kw):
    ws = FakeWS()
    calls = []

    async def submit(tid, text):
        calls.append(("submit", text))
        return f"u{sum(1 for c in calls if c[0] == 'submit')}"

    async def abort(tid):
        calls.append(("abort", tid))

    async def retract(tid, mid):
        calls.append(("retract", mid))

    host = VoiceHost(submit=submit, abort=abort, retract=retract)
    tts = FakeTTS()
    s = VoiceSession(ws, thread_id=THREAD, settings=settings(merge_window_s=1.0, **kw),
                     detector=ScriptedDetector(script), stt_engine=SeqSTT(texts),
                     tts_engine=tts, host=host)
    return s, ws, tts, calls


async def test_resume_within_window_merges_into_one_message():
    s, ws, tts, calls = merge_session(
        ["Remind me to call the garage", "about the brakes"],
        [[utterance()], [SpeechStart()], [utterance()]])
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    ws.push_audio(); await settle()
    # The early turn's reply starts arriving while we are still in the window...
    s.on_app_frame({"type": "message", "thread_id": THREAD,
                    "message": {"id": "u1", "role": "user", "content": "x"}})
    s.on_app_frame({"type": "stream_chunk", "thread_id": THREAD, "message_id": "run:a",
                    "text": "Sure, which garage do you mean? "})
    ws.push_audio(); await settle()            # driver resumes
    ws.push_audio(); await settle()            # ...and finishes
    assert calls == [("submit", "Remind me to call the garage"), ("abort", THREAD),
                     ("retract", "u1"), ("submit", "Remind me to call the garage about the brakes")]
    last = ws.json_of("transcript")[-1]
    assert isinstance(last.pop("upto"), int)          # C.4: the utterance end's uplink offset
    assert last == {"type": "transcript", "merged": True, "message_id": "u2",
                    "text": "Remind me to call the garage about the brakes"}
    # The abandoned turn never got a word out, even when it finishes late.
    s.on_app_frame({"type": "stream_done", "thread_id": THREAD, "provisional_id": "run:a",
                    "message_id": "a1", "message": {"id": "a1", "role": "assistant",
                    "content": "Sure, which garage do you mean? Tell me."}})
    # Then the merged message echoes and ITS reply is spoken.
    s.on_app_frame({"type": "message", "thread_id": THREAD,
                    "message": {"id": "u2", "role": "user", "content": "merged"}})
    s.on_app_frame({"type": "message", "thread_id": THREAD,
                    "message": {"id": "a2", "role": "assistant", "content": "Done, brakes it is."}})
    await asyncio.sleep(1.2); await settle()
    # (M1 may have STARTED synthesising the early reply during the hold; it
    # must never have been sent.)
    assert [m["text"] for m in ws.json_of("audio") if m["kind"] == "reply" and m["chunk"] == 0] \
        == ["Done, brakes it is."]
    ws.hang_up(); await task


async def test_reply_is_held_during_merge_window_then_spoken():
    s, ws, tts, calls = merge_session(["What time is it"], [[utterance()]])
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); ws.push_audio(); await settle()
    s.on_app_frame({"type": "message", "thread_id": THREAD,
                    "message": {"id": "a1", "role": "assistant", "content": "It is three o'clock."}})
    await settle()
    # M1: synthesis starts at once, but nothing is SENT inside the 1 s window.
    assert tts.said == ["It is three o'clock."]
    assert not [m for m in ws.json_of("audio") if m["kind"] == "reply"]
    await asyncio.sleep(1.1); await settle()
    assert [m["text"] for m in ws.json_of("audio") if m["kind"] == "reply" and m["chunk"] == 0] \
        == ["It is three o'clock."]
    ws.hang_up(); await task


async def test_resume_that_is_only_noise_releases_the_hold():
    s, ws, tts, calls = merge_session(["What time is it", ""],
                                      [[utterance()], [SpeechStart()], [utterance()]])
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); ws.push_audio(); await settle()
    ws.push_audio(); await settle()
    assert s._hold
    ws.push_audio(); await settle()
    assert [c[0] for c in calls] == ["submit"]        # no abort, no retract
    assert not s._hold
    ws.hang_up(); await task


async def test_patience_preset_scales_endpointing():
    s, ws, tts, _ = make_session([])
    base = s.detector.ep.cfg.turn_hard_silence_s
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start", "patience": "patient"}); await settle()
    assert s.patience == 1.5
    assert s.detector.ep.cfg.turn_hard_silence_s == pytest.approx(base * 1.5)
    assert s.settings.merge_window(s.patience) == pytest.approx(0.15 * 1.5)
    ws.push_json({"type": "start", "patience": "nonsense"}); await settle()
    assert s.patience == s.settings.patience
    ws.hang_up(); await task


async def test_voice_ack_plays_at_submit_in_the_session_voice():
    ws = FakeWS()

    async def submit(tid, text):
        return "u1"

    tts = FakeTTS()
    s = VoiceSession(ws, thread_id=THREAD, settings=settings(),
                     detector=ScriptedDetector([[utterance()]]), stt_engine=FakeSTT(),
                     tts_engine=tts, host=VoiceHost(submit=submit),
                     profile_id="host-voice",
                     acks=kit_with_lines({"question": ("Let me see.",)}, voice="host-voice"))
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); ws.push_audio(); await settle()
    acks = [m for m in ws.json_of("audio") if m["kind"] == "ack"]
    assert len(acks) == 1 and not ws.json_of("ack")    # spoken ack replaces the cue
    say = [m for m in ws.json_of("say") if m["kind"] == "ack"]
    assert say == [{**say[0], "text": "Let me see.", "seq": acks[0]["seq"], "intent": "question"}]
    assert "Let me see." in s._recent_said                # its echo is never a turn (C.5 #5)
    s._merge_until = 0
    s.on_app_frame({"type": "message", "thread_id": THREAD,
                    "message": {"id": "a1", "role": "assistant", "content": "Here you go then."}})
    await settle()
    assert tts.voices == ["host-voice"]
    ws.hang_up(); await task


async def test_reply_audio_streams_in_chunks_with_one_seq():
    s, ws, tts, _ = make_session([])
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    s.on_app_frame({"type": "message", "thread_id": THREAD,
                    "message": {"id": "a1", "role": "assistant", "content": "One full sentence here."}})
    await settle()
    hdrs = [m for m in ws.json_of("audio") if m["kind"] == "reply"]
    assert [h["chunk"] for h in hdrs] == [0, 1, 2]
    assert len({h["seq"] for h in hdrs}) == 1 and hdrs[0]["text"] and hdrs[1]["text"] is None
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# 2.1.0 RC review regressions
# --------------------------------------------------------------------------- #

async def test_audio_and_control_frames_slide_the_session_idle_window():
    """H1: a chat longer than the idle lock must not lock mid-sentence."""
    touches = []
    clock = {"t": 0.0}
    s, ws, tts, _ = make_session([], touch=lambda: touches.append(clock["t"]))
    s.clock = lambda: clock["t"]
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    await settle(5)
    assert len(touches) == 1                     # a control frame is activity
    ws.push_json({"type": "ping"})               # a ping is not
    ws.push_audio(); ws.push_audio()             # throttled: same instant
    await settle(5)
    assert len(touches) == 1
    clock["t"] = 31.0
    ws.push_audio()                              # real mic audio, 31 s later
    await settle(5)
    assert len(touches) == 2
    ws.hang_up(); await task


async def test_speaking_survives_playback_done_while_next_sentence_synthesises():
    """M6: the client drains sentence 1 while sentence 2 is still streaming in.
    `speaking` must stay on, or our own voice counts as a barge-in."""
    s, ws, tts, _ = make_session([])
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    await settle(5)
    s.on_app_frame({"type": "stream_start", "thread_id": THREAD, "message_id": "run:1"})
    s.on_app_frame({"type": "stream_chunk", "thread_id": THREAD,
                    "message_id": "run:1", "text": "This is the first sentence. And the"})
    await settle()
    assert s.speaking and tts.said == ["This is the first sentence."]
    ws.push_json({"type": "playback_done", "gen": s.gen})
    await settle()
    assert s.speaking, "reply still streaming: must stay speaking"
    s.on_app_frame({"type": "stream_done", "thread_id": THREAD, "provisional_id": "run:1",
                    "message_id": "m-1", "message": {"id": "m-1", "role": "assistant",
                    "content": "This is the first sentence. And the second one follows."}})
    await settle()
    assert s.speaking                            # sentence 2 sent, not yet played
    ws.push_json({"type": "playback_done", "gen": s.gen})
    await settle()
    assert not s.speaking
    ws.hang_up(); await task


async def test_crashed_worker_sends_error_and_ends_session():
    """M7: a worker exception must not leave a zombie "Listening" session."""
    class Boom(ScriptedDetector):
        def feed(self, pcm):
            raise RuntimeError("detector exploded")

    ws = FakeWS()

    async def submit(tid, text):
        return "m"

    s = VoiceSession(ws, thread_id=THREAD, settings=settings(), detector=Boom([]),
                     stt_engine=FakeSTT(), tts_engine=FakeTTS(), host=VoiceHost(submit=submit))
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); ws.push_audio()
    await asyncio.wait_for(task, 2)              # run() returns on its own
    err = ws.json_of("error")
    assert err and "Voice chat" in err[0]["message"] and "Drive" not in err[0]["message"]


async def test_merged_utterance_is_capped():
    from dispatch_voice.host import session as vs
    s, ws, tts, calls = merge_session(
        ["a " * 3000, "b " * 3000], [[utterance()], [SpeechStart()], [utterance()]])
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    for _ in range(3):
        ws.push_audio(); await settle()
    submits = [c[1] for c in calls if c[0] == "submit"]
    assert len(submits) == 2 and len(submits[1]) <= vs.MAX_UTTERANCE_CHARS
    ws.hang_up(); await task
