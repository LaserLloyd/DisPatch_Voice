"""Interrupt sensitivity on the realtime path (1.5.0): the probe flow with a streaming STT,
as live runs it (local Parakeet or the rig). Tick partials, pause verdicts and the level
gate at "normal" against the same scenes at "high" (the 1.4.0 rules)."""
from __future__ import annotations

import numpy as np
import pytest
from session.test_session_s3 import FakeStreamingSTT, over_reply, partial, speak
from session.test_session_s3 import pause as _pause

from dispatch_voice.host.vad_turn import Pause, SpeechStart
from dispatch_voice.rt.interrupt import speech_level_db

pytestmark = pytest.mark.interrupt_default
SR = 16000


def tone(seconds: float, amp: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def pause_with(audio: np.ndarray, silence_s: float = 0.25) -> Pause:
    return Pause(audio=audio, silence_s=silence_s, speech_s=len(audio) / SR - silence_s)


async def scene(sens: str, script, *, ticks=(), partials=("x",)):
    sstt = FakeStreamingSTT([partial(p) for p in partials], ticks=[partial(t) for t in ticks])
    s, ws, tts, host, det, task = await over_reply(script, words="x", sstt=sstt)
    ws.push_json({"type": "start", "acks": "off", "interrupt_sensitivity": sens})
    await speak(ws, 0, wait=5)
    assert s.interrupt.name == sens
    return s, ws, host, task


@pytest.mark.parametrize(("sens", "tick", "cut"), [
    ("normal", "What?", False), ("normal", "Thank you.", False), ("normal", "Sorry, what?", False),
    ("normal", "Wait", True), ("normal", "Okay, stop.", True), ("normal", "Hold on.", True),
    ("high", "What?", True), ("high", "Thank you.", True),
])
async def test_a_tick_cuts_mid_speech_only_on_a_clear_stop_word_below_high(sens, tick, cut):
    s, ws, host, task = await scene(sens, [[SpeechStart()], []], ticks=[tick])
    await speak(ws, 2, wait=40)
    assert bool(ws.json_of("barge_in")) is cut, (sens, tick)
    assert not host.of("submit")
    ws.hang_up(); await task


@pytest.mark.parametrize(("sens", "tick", "cut"), [
    ("normal", "tell me more", False), ("normal", "tell me more please", True),
    ("low", "tell me more please", False), ("low", "tell me more about it", True),
    ("high", "tell me more", True),
])
async def test_a_tick_needs_the_profile_s_word_count_to_cut(sens, tick, cut):
    s, ws, host, task = await scene(sens, [[SpeechStart()], []], ticks=[tick])
    await speak(ws, 2, wait=40)
    assert bool(ws.json_of("barge_in")) is cut, (sens, tick)
    ws.hang_up(); await task


async def test_normal_never_ducks_at_speech_start_high_does():
    for sens, ducked in (("normal", False), ("high", True)):
        s, ws, host, task = await scene(sens, [[SpeechStart()]])
        await speak(ws, 1, wait=20)
        assert bool([d for d in ws.json_of("duck") if d["on"]]) is ducked, sens
        ws.hang_up(); await task


async def test_a_faint_talker_at_a_pause_is_the_room_the_reply_goes_on():
    """Someone across the room (20 dB under the listener's learned voice) says real words
    over the reply: at normal nothing is cut and no turn is made; at high (1.4.0) the
    same words cut the reply and become the next turn."""
    faint = np.concatenate([tone(1.2, 0.03), np.zeros(int(0.25 * SR), np.float32)])
    words = "why is he talking to the computer again"
    for sens, cut in (("normal", False), ("high", True)):
        s, ws, host, task = await scene(sens, [[SpeechStart()], [pause_with(faint)]], partials=[words])
        for _ in range(2):
            s.levels.note_voice(speech_level_db(tone(1.0, 0.3)))
        await speak(ws, 2, wait=60)
        assert bool(ws.json_of("barge_in")) is cut, sens
        assert bool(host.of("abort")) is cut, sens
        if not cut:
            assert not host.of("submit")
            assert s.speaking, "the reply went on"
        ws.hang_up(); await task


async def test_the_listener_at_a_pause_still_cuts_the_reply_at_normal():
    near = np.concatenate([tone(1.2, 0.25), np.zeros(int(0.25 * SR), np.float32)])
    s, ws, host, task = await scene("normal", [[SpeechStart()], [pause_with(near)]],
                                    partials=["actually tell me about penguins"])
    for _ in range(2):
        s.levels.note_voice(speech_level_db(tone(1.0, 0.3)))
    await speak(ws, 2, wait=60)
    assert ws.json_of("barge_in") and host.of("abort")
    ws.hang_up(); await task


async def test_an_ambiguous_stop_word_said_near_the_phone_still_stops_at_normal():
    """"Thank you." as a whole utterance in the listener's own voice is a stop (they mean
    it); the same words faint, or with no level known, are a backchannel."""
    near = np.concatenate([tone(0.6, 0.25), np.zeros(int(0.25 * SR), np.float32)])
    s, ws, host, task = await scene("normal", [[SpeechStart()], [pause_with(near)]], partials=["Thank you."])
    for _ in range(2):
        s.levels.note_voice(speech_level_db(tone(1.0, 0.3)))
    await speak(ws, 2, wait=60)
    assert ws.json_of("barge_in") and not host.of("submit")
    ws.hang_up(); await task
    s, ws, host, task = await scene("normal", [[SpeechStart()], [_pause(speech_s=0.6)]], partials=["Thank you."])
    await speak(ws, 2, wait=60)
    assert not ws.json_of("barge_in"), "zero audio: no level known, no ambiguous stop"
    ws.hang_up(); await task
