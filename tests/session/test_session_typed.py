"""The `typed` control frame (C.3, 1.2.0): a message typed on the voice chat screen
becomes the same turn a transcript makes — submitted through the host with the voice
persona, its reply spoken — with no audio, STT, hold or speculation. Fakes only."""
from __future__ import annotations

import asyncio

from fakes import FakeSTT, FakeTTS, FakeWS, ScriptedDetector

from dispatch_voice.host.session import TYPED_MAX_CHARS, VoiceHost, VoiceSession

from .test_session import THREAD, make_session, settings, settle, wait_until


async def test_a_typed_message_is_submitted_and_echoed_as_a_typed_transcript():
    s, ws, _tts, submitted = make_session([])
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); await settle(5)
    ws.push_json({"type": "pause"}); await settle(5)                 # a headset: mic muted
    ws.push_json({"type": "typed", "text": "  what's the weather in Fussa?  "})
    assert await wait_until(lambda: submitted)
    assert submitted == [(THREAD, "what's the weather in Fussa?")]
    tr = ws.json_of("transcript")
    assert tr and tr[-1] == {"type": "transcript", "text": "what's the weather in Fussa?",
                             "merged": False, "typed": True, "message_id": "m-user"}
    assert not ws.json_of("eot"), "nothing was heard: no end-of-turn"
    assert {"type": "state", "state": "thinking"} in ws.sent
    assert s.listening is False, "the microphone's state is left alone"
    ws.hang_up(); await task


async def test_the_reply_to_a_typed_message_is_spoken():
    s, ws, tts, submitted = make_session([])
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); await settle(5)
    ws.push_json({"type": "typed", "text": "tell me a joke"})
    assert await wait_until(lambda: submitted)
    s.on_app_frame({"type": "message", "thread_id": THREAD,
                    "message": {"id": "m-user", "role": "user", "content": "tell me a joke"}})
    s.on_app_frame({"type": "thinking", "thread_id": THREAD, "state": "started"})
    s.on_app_frame({"type": "stream_chunk", "thread_id": THREAD, "message_id": "run:1",
                    "text": "Why did the cat sit on the laptop? For the warmth."})
    s.on_app_frame({"type": "stream_done", "thread_id": THREAD, "provisional_id": "run:1",
                    "message_id": "m-9", "message": {"id": "m-9", "role": "assistant",
                    "content": "Why did the cat sit on the laptop? For the warmth."}})
    assert await wait_until(lambda: len(tts.said) >= 2)
    assert tts.said == ["Why did the cat sit on the laptop?", "For the warmth."]
    ws.hang_up(); await task


async def test_a_bad_typed_frame_submits_nothing():
    s, ws, _tts, submitted = make_session([])
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); await settle(5)
    for bad in ({"type": "typed"}, {"type": "typed", "text": "   "}, {"type": "typed", "text": 7},
                {"type": "typed", "text": "x" * (TYPED_MAX_CHARS + 1)}):
        ws.push_json(bad)
    await settle(10)
    assert submitted == []
    errs = ws.json_of("error")
    assert errs and errs[0]["code"] == "bad_frame"
    assert not ws.json_of("transcript")
    ws.hang_up(); await task


async def test_the_host_is_told_the_message_was_typed():
    seen = []

    async def submit(tid, text, opts=None):
        seen.append((text, dict(opts or {})))
        return "m-user"

    ws = FakeWS()
    s = VoiceSession(ws, thread_id=THREAD, settings=settings(), detector=ScriptedDetector([]),
                     stt_engine=FakeSTT("unused"), tts_engine=FakeTTS(), host=VoiceHost(submit=submit))
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); await settle(5)
    ws.push_json({"type": "typed", "text": "hello"})
    assert await wait_until(lambda: seen)
    assert seen[0][0] == "hello" and seen[0][1]["typed"] is True
    assert seen[0][1]["reply_style"] == "normal"      # the voice persona's options as usual
    ws.hang_up(); await task


async def test_a_message_typed_over_a_playing_reply_cuts_it_first():
    aborted = []

    async def abort(tid, keep_partial=True):
        aborted.append(tid)
        return None

    s, ws, tts, submitted = make_session([], abort=abort)
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); await settle(5)
    s.on_app_frame({"type": "stream_start", "thread_id": THREAD, "message_id": "run:1"})
    s.on_app_frame({"type": "stream_chunk", "thread_id": THREAD, "message_id": "run:1",
                    "text": "One sentence here. Two sentences here. Three sentences here."})
    assert await wait_until(lambda: tts.said and ws.json_of("audio"))
    assert s.speaking or s._talking()
    gen_before = s.gen
    ws.push_json({"type": "typed", "text": "wait, something else"})
    assert await wait_until(lambda: submitted)
    assert ws.json_of("barge_in") and s.gen == gen_before + 1
    assert submitted == [(THREAD, "wait, something else")]
    ws.hang_up(); await task


async def test_a_submit_failure_is_a_turn_error_not_a_crash():
    async def submit(tid, text):
        raise RuntimeError("down")

    ws = FakeWS()
    s = VoiceSession(ws, thread_id=THREAD, settings=settings(), detector=ScriptedDetector([]),
                     stt_engine=FakeSTT("unused"), tts_engine=FakeTTS(), host=VoiceHost(submit=submit))
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); await settle(5)
    ws.push_json({"type": "typed", "text": "hello"})
    assert await wait_until(lambda: ws.json_of("turn_error"))
    assert ws.json_of("turn_error")[0]["reason"] == "agent_failed"
    assert ws.json_of("transcript")[-1]["typed"] is True
    ws.push_json({"type": "ping", "t": 1}); await settle(5)
    assert ws.json_of("pong"), "the session is still alive"
    ws.hang_up(); await task
