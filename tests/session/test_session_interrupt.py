"""Interrupt sensitivity in the session (1.5.0): background talk and noise over a reply
neither pause nor cut it at "normal"; the listener's own voice and clear stop words do.

A real TurnDetector runs over synthetic PCM (a tone at speech level for the listener, the
same tone 20 dB down for someone across the room, low noise for the room); its VAD calls
all of them speech, as Silero does with chatter. The words come from FakeSTT."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest
from fakes import FakeSTT, FakeTTS, FakeWS, ScriptedDetector, utterance

from dispatch_voice.host.config import VoiceSettings
from dispatch_voice.host.session import VoiceHost, VoiceSession
from dispatch_voice.host.vad_turn import Endpointer, EndpointerConfig, SpeechStart, TurnDetector
from settling import pause, settle

pytestmark = pytest.mark.interrupt_default
THREAD = "t-int"
SR = 16000


def settings(**kw):
    base = {"flag": "1", "ack_after_s": 0.05, "merge_window_s": 0.15}
    base.update(kw)
    return VoiceSettings(**base)


def pcm(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def tone(seconds: float, amp: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def room(seconds: float, amp: float = 0.0008, seed: int = 5) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(int(seconds * SR)) * amp).astype(np.float32)


def real_detector() -> TurnDetector:
    cfg = EndpointerConfig(min_speech_s=0.25, pre_roll_s=0.3, turn_hard_silence_s=0.8,
                           no_smart_turn_silence_s=0.6)
    # Anything over the room's hiss is "speech" (chatter fools Silero the same way).
    return TurnDetector(lambda f: 1.0 if float(np.abs(f).mean()) > 0.004 else 0.0, Endpointer(cfg))


def make(detector, stt_text="what's the weather"):
    ws = FakeWS()
    submitted: list[str] = []
    aborted: list[str] = []

    async def submit(tid, text):
        submitted.append(text)
        return f"m-user-{len(submitted)}"

    async def abort(tid):
        aborted.append(tid)

    host = VoiceHost(submit=submit, abort=abort)
    stt = FakeSTT(stt_text)
    s = VoiceSession(ws, thread_id=THREAD, settings=settings(barge_abort_turn=True), detector=detector,
                     stt_engine=stt, tts_engine=FakeTTS(), host=host)
    return s, ws, stt, submitted, aborted



async def push(ws, x: np.ndarray, chunk_s: float = 0.1):
    step = int(chunk_s * SR)
    for i in range(0, len(x), step):
        ws.push_audio(pcm(x[i:i + step]))
        await pause(0.002)
    await settle(20)


async def reply_playing(s, ws, mid="m-a1"):
    s.on_app_frame({"type": "message", "thread_id": THREAD,
                    "message": {"id": mid, "role": "assistant", "content": "A long answer about fruit bats."}})
    await settle()
    assert s.speaking


async def learn_voice(s, ws, stt, n=2):
    """The listener's level as two ordinary turns would teach it (see the test below)."""
    from dispatch_voice.rt.interrupt import speech_level_db
    for _ in range(n):
        s.levels.note_voice(speech_level_db(tone(1.0, 0.3)))


async def test_ordinary_turns_teach_the_listener_level_turns_over_a_reply_do_not():
    s, ws, stt, submitted, aborted = make(real_detector())
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    await push(ws, room(1.0))
    assert s.levels.floor_db is not None and s.levels.floor_db < -55, "the room's hiss is the floor"
    for i in range(2):
        stt.text = f"tell me something new number {i}"
        await push(ws, np.concatenate([tone(1.0, 0.3), room(1.0)]))
    assert len(submitted) == 2
    assert s.levels.voice_db == pytest.approx(20 * np.log10(0.3 / np.sqrt(2)), abs=1.0)
    ws.hang_up(); await task


async def test_the_default_sensitivity_is_normal_and_start_changes_it():
    s, ws, *_ = make(ScriptedDetector([]))
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    await settle()
    assert s.interrupt_sensitivity == "normal" and s.interrupt.tick_min_words == 4
    ws.push_json({"type": "start", "interrupt_sensitivity": "low"})
    await settle()
    assert s.interrupt.name == "low"
    ws.push_json({"type": "start", "interrupt_sensitivity": "high"})
    await settle()
    assert s.interrupt.name == "high"
    ws.push_json({"type": "start", "interrupt_sensitivity": "deafening"})
    await settle()
    assert s.interrupt.name == "normal"
    ws.push_json({"type": "start", "patience": "normal"})       # absent key: unchanged
    await settle()
    assert s.interrupt.name == "normal"
    ws.hang_up(); await task


async def test_background_talk_over_a_reply_neither_pauses_nor_cuts_it():
    s, ws, stt, submitted, aborted = make(real_detector())
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    await push(ws, room(1.0))
    await learn_voice(s, ws, stt)
    n_turns = len(submitted)
    await reply_playing(s, ws)
    gen = s.gen
    # Someone across the room talks for 1.2 s, 20 dB under the listener: real words.
    stt.text = "why is he talking to the computer again"
    await push(ws, np.concatenate([tone(1.2, 0.03), room(1.2)]))
    assert ws.json_of("duck") == [], "the reply never paused"
    assert ws.json_of("barge_in") == [] and s.gen == gen, "nor was it cut"
    assert aborted == [] and len(submitted) == n_turns, "and the words did not become a turn"
    assert s.speaking
    ws.hang_up(); await task


async def test_room_noise_over_a_reply_does_not_pause_it():
    s, ws, stt, submitted, aborted = make(real_detector())
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    await push(ws, room(1.0))
    await learn_voice(s, ws, stt)
    await reply_playing(s, ws)
    stt.text = ""                                   # a clatter: no words at all
    await push(ws, np.concatenate([room(1.0, amp=0.01, seed=9), room(1.0)]))
    assert ws.json_of("duck") == [] and ws.json_of("barge_in") == []
    ws.hang_up(); await task


async def test_your_own_voice_over_a_reply_pauses_it_after_sustained_speech_then_cuts_it():
    s, ws, stt, submitted, aborted = make(real_detector())
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    await push(ws, room(1.0))
    await learn_voice(s, ws, stt)
    n_turns = len(submitted)
    await reply_playing(s, ws)
    # 0.3 s of the listener: past 1.4.0's 0.25 s duck, short of normal's 0.45 s.
    await push(ws, tone(0.3, 0.3), chunk_s=0.05)
    assert ws.json_of("duck") == [], "not yet: normal waits for 0.45 s of speech"
    await push(ws, tone(0.4, 0.3), chunk_s=0.05)
    assert ws.json_of("duck")[:1] == [{"type": "duck", "on": True, "gen": 0}]
    stt.text = "actually tell me about penguins instead"
    await push(ws, room(1.2))
    assert ws.json_of("barge_in"), "real words in the listener's voice cut the reply"
    assert aborted == [THREAD]
    assert submitted[n_turns:] == ["actually tell me about penguins instead"]
    ws.hang_up(); await task


async def test_high_keeps_the_1_4_0_rules_background_talk_cuts_the_reply():
    s, ws, stt, submitted, aborted = make(real_detector())
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start", "interrupt_sensitivity": "high"})
    await push(ws, room(1.0))
    await learn_voice(s, ws, stt)
    await reply_playing(s, ws)
    stt.text = "why is he talking to the computer again"
    await push(ws, np.concatenate([tone(1.2, 0.03), room(1.2)]))
    assert ws.json_of("duck"), "high ducks at the first 0.25 s, at any level"
    assert ws.json_of("barge_in") and aborted == [THREAD]
    ws.hang_up(); await task


# --------------------------------------------- stop words (scripted detector)


def scripted(stt_text):
    return make(ScriptedDetector([[SpeechStart()], [utterance()]]), stt_text=stt_text)


@pytest.mark.parametrize(("sens", "words", "cut"), [
    ("normal", "What?", False), ("normal", "Thank you.", False), ("normal", "Sorry?", False),
    ("normal", "Stop.", True), ("normal", "Okay, wait.", True), ("normal", "Hold on.", True),
    ("low", "What?", False), ("low", "Wait.", True),
    ("high", "What?", True), ("high", "Thank you.", True),
])
async def test_only_clear_stop_words_stop_a_reply_below_high(sens, words, cut):
    s, ws, stt, submitted, aborted = scripted(words)
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start", "interrupt_sensitivity": sens})
    await reply_playing(s, ws)
    ws.push_audio()          # speech starts over the reply
    await settle()
    ws.push_audio()          # ... and ends (zero audio: level unknown)
    await settle(60)
    assert bool(ws.json_of("barge_in")) is cut, (sens, words)
    assert submitted == [], "a stop or a backchannel never becomes a turn"
    ws.hang_up(); await task


async def test_mid_speech_early_stop_takes_only_clear_stop_words_below_high():
    s, ws, *_ = make(ScriptedDetector([]))
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"})
    await settle()
    assert not s._early_stop("what")
    assert not s._early_stop("sorry what")
    assert s._early_stop("actually wait")
    assert s._early_stop("stop")
    ws.push_json({"type": "start", "interrupt_sensitivity": "high"})
    await settle()
    assert s._early_stop("what") and s._early_stop("actually wait")
    ws.hang_up(); await task
