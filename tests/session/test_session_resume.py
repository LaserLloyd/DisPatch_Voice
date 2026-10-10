"""Session hub, slice 4 (SPEC §14 T3 (4), §12.1 T3 rows): resuming a voice chat —
the late joiner (hardening H4: `test_late_joiner_speaks_full_reply_once`), a reconnect's
new `sid` and seq 1, a restart between `transcript` and `pong` (one row, C.5 #8 — the
cursor answer and the server's belt for replayed audio), `replay_last` ("Tap to
continue" after a cut, "say that again" after a finished reply; SPEC §4.6 item 4, C.3),
and the merge-hold fixes an uplink reset needs (an open utterance dropped by
`GapTooLarge`, a hold no speech is left for). All with fakes: no models, no DisPatch."""
from __future__ import annotations

import asyncio

import numpy as np
from fakes import FakeSTT, FakeTTS, FakeWS, ScriptedDetector, utterance

from dispatch_voice.host import session as vs
from dispatch_voice.host.session import VoiceSession
from dispatch_voice.host.vad_turn import SpeechStart
from dispatch_voice.transport import CommitCursor, Transport, wire3
from session.test_session import THREAD, settings, settle
from session.test_session_s2 import Host, frame, reply
from session.test_session_s3 import Clock, FakeStreamingSTT, partial, pause, speak
from session.test_session_s3 import make as make_rt
from session.test_session_s3 import start as start_rt
from session.test_session_v3_wire import (
    DEVICE,
    STREAM,
    RecordingDetector,
    binaries,
    query,
    up_pcm,
)

# the late-joiner case from the hardening review (H4), as the session speaks it
FULL = "The dentist is at three on Tuesday. Bring the insurance card. Park on the left side."
SENTENCES = ["The dentist is at three on Tuesday.", "Bring the insurance card.", "Park on the left side."]
REPLY = "You have two things this afternoon. A dentist appointment at three. Want a reminder?"
R1, R2, R3 = "You have two things this afternoon.", "A dentist appointment at three.", "Want a reminder?"


def make(script=(), *, q=None, cursor=None, host: Host | None = None, tts=None, det=None, stt="hello there",
         clock=None, host_kw: dict | None = None, **kw):
    """A v3 session (PCM both ways unless ``q`` says otherwise) over fakes."""
    ws = FakeWS(query=q if q is not None else query("pcm16"))
    tts = tts or FakeTTS(chunks=2, chunk_samples=2400)          # 0.2 s per sentence
    tr = Transport.negotiate(ws, ws.query_params, rate_out=tts.sample_rate)
    host = host or Host()
    det = det or RecordingDetector(list(script))
    key = (DEVICE, STREAM) if cursor is not None else None
    extra = {"clock": clock} if clock is not None else {}
    s = VoiceSession(ws, thread_id=THREAD, settings=settings(merge_window_s=0.0), detector=det,
                     stt_engine=FakeSTT(stt), tts_engine=tts, host=host.build(**(host_kw or {})),
                     notice=lambda text: np.ones(48, "<i2"), heard_wait_s=0.3,
                     transport=tr, cursor=cursor, cursor_key=key, **extra, **kw)
    s.watch_tick_s = 0.01
    return s, ws, tts, host


async def start(s, ws, **extra):
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start", "acks": "off", **extra})
    await settle(5)
    return task


def chunks(rid: str, text: str, *, start: int = 0, step: int = 17) -> list[dict]:
    return [frame(type="stream_chunk", message_id=rid, text=text[i:i + step]) for i in range(start, len(text), step)]


def done(rid: str, mid: str, text: str) -> dict:
    return frame(type="stream_done", message_id=mid, provisional_id=rid,
                 message={"id": mid, "role": "assistant", "content": text})


def says(ws) -> list[dict]:
    return ws.json_of("say")


def as_turn(s) -> None:
    """The reply that follows belongs to a voice turn of this session (as a submit makes it)."""
    s._current = vs._Turn(msg_id="u0", text="before", at=s.clock())
    s._owner = s._current


# --------------------------------------------------------------------------- #
# the late joiner (hardening H4, C.5 #11)
# --------------------------------------------------------------------------- #

async def test_late_joiner_speaks_full_reply_once():
    """The prototype's three sessions through the real session: one present from the start,
    one that joined after 40 characters had streamed with NO snapshot (the belt: its
    deltas are a tail and are dropped, the reply is spoken from `stream_done`), and one
    that got the subscribe snapshot (`stream_start` + `stream_chunk {replace}`). Each speaks
    the three sentences once, whole, in order — never "g the insurance card." or a splice."""
    runs = {}
    for name in ("from_start", "late", "snapshot"):
        s, ws, tts, _host = make()
        task = await start(s, ws)
        if name == "from_start":
            frames = [frame(type="stream_start", message_id="run:a"), *chunks("run:a", FULL)]
        elif name == "late":
            frames = chunks("run:a", FULL, start=40)
        else:
            frames = [frame(type="stream_start", message_id="run:a"),
                      frame(type="stream_chunk", message_id="run:a", text=FULL[:40], replace=True),
                      *chunks("run:a", FULL, start=40)]
        for f in [*frames, done("run:a", "m1", FULL)]:
            s.on_app_frame(f)
        await settle(60)
        runs[name] = (list(tts.said), [m["text"] for m in says(ws)])
        assert ws.json_of("say_end")[-1]["last_seq"] == says(ws)[-1]["seq"]
        ws.hang_up(); await task
    for name, (said, texts) in runs.items():
        assert said == SENTENCES, (name, said)
        assert texts == SENTENCES, (name, texts)


async def test_a_late_joiner_skips_the_sentences_the_old_socket_heard():
    s, ws, tts, _host = make()
    task = await start(s, ws, after_partial={"message_id": "run:a", "sentences": 1})
    for f in [*chunks("run:a", FULL, start=40), done("run:a", "m1", FULL)]:
        s.on_app_frame(f)
    await settle(60)
    assert tts.said == SENTENCES[1:]
    assert [m["seq"] for m in says(ws)] == [1, 2]
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# reconnect and restart
# --------------------------------------------------------------------------- #

async def test_a_reconnect_is_a_new_sid_and_its_first_sentence_is_seq_1():
    """C.5 #9 / hardening M2: the old session got to seq 3; the new one starts at seq 1
    under a NEW sid, so the client's `(sid, seq)` claims never drop its first sentence."""
    s1, ws1, _tts1, _ = make()
    t1 = await start(s1, ws1)
    for f in reply("m1", REPLY, prov="run:1"):
        s1.on_app_frame(f)
    await settle(60)
    assert [m["seq"] for m in says(ws1)] == [1, 2, 3]
    ws1.hang_up(); await t1
    s2, ws2, _tts2, _ = make()
    t2 = await start(s2, ws2)
    for f in reply("m2", "Here is the next one.", prov="run:2"):
        s2.on_app_frame(f)
    await settle(40)
    assert says(ws2)[0]["seq"] == 1
    assert ws2.json_of("ready")[0]["sid"] != ws1.json_of("ready")[0]["sid"]
    ws2.hang_up(); await t2


async def test_a_restart_between_transcript_and_pong_submits_the_utterance_once(tmp_path):
    """The process restarts after the submit and before the client's next `pong`: the
    cursor is persisted (a fresh CommitCursor reads it back), the reconnect is answered
    with the committed offset AND the transcript again — and a client that replays from
    below that answer anyway (one that gave up waiting for it on a slow link) still makes
    no second row: an utterance ending inside replayed audio at or below the committed
    offset is dropped (C.5 #8). A new utterance after it is a turn as usual."""
    det1 = RecordingDetector([[], [SpeechStart(), utterance()]])
    s1, ws1, _, host1 = make(det=det1, cursor=CommitCursor(tmp_path), stt="call the garage")
    t1 = await start(s1, ws1)
    ws1.push_audio(up_pcm(0, 320)); await settle(5)
    ws1.push_audio(up_pcm(320, 320)); await settle(20)
    assert [c[1] for c in host1.of("submit")] == ["call the garage"]
    assert ws1.json_of("eot")[0]["upto"] == 640
    ws1.hang_up(); await t1
    # --- "restart": a new process, a new cursor object over the same file
    det2 = RecordingDetector([[], [SpeechStart(), utterance()], [], [SpeechStart(), utterance()]])
    s2, ws2, _, host2 = make(det=det2, cursor=CommitCursor(tmp_path), stt="call the garage")
    t2 = await start(s2, ws2, cursor={"stream": STREAM, "from": 320})
    assert ws2.json_of("cursor") == [{"type": "cursor", "from": 640}]
    assert ws2.json_of("transcript")[0]["replayed"] is True
    # the client replays from 0 (below the answer) with the replay flag
    ws2.push_audio(up_pcm(0, 320, flags=wire3.FLAG_REPLAY)); await settle(5)
    ws2.push_audio(up_pcm(320, 320, flags=wire3.FLAG_REPLAY)); await settle(20)
    assert not host2.of("submit") and not ws2.json_of("eot")
    # live audio after it: a new utterance, a new row
    ws2.push_audio(up_pcm(640, 320)); await settle(5)
    ws2.push_audio(up_pcm(960, 320)); await settle(20)
    assert [c[1] for c in host2.of("submit")] == ["call the garage"]
    assert ws2.json_of("eot")[0]["upto"] == 1280
    ws2.hang_up(); await t2


async def test_after_partial_alone_resumes_the_stored_reply_from_the_cursor_anchor(tmp_path):
    """A reply that was STORED before the drop (an API bot's row exists before its stream)
    is resumed only from a stored row, and a stored row is found only after something.
    A client that names the cut reply (`after_partial`) without an `after` anchor gets the
    rest of exactly that reply: the cursor's record of the last submitted utterance (its
    user row) is the anchor. Nothing else after the anchor is read out."""
    cur = CommitCursor(tmp_path)
    det1 = RecordingDetector([[], [SpeechStart(), utterance()]])
    s1, ws1, _, _host1 = make(det=det1, cursor=cur, stt="what's on today")
    t1 = await start(s1, ws1)
    ws1.push_audio(up_pcm(0, 320)); await settle(5)
    ws1.push_audio(up_pcm(320, 320)); await settle(20)
    assert cur.entry((DEVICE, STREAM)).meta["message_id"] == "u1"
    ws1.hang_up(); await t1
    asked = []

    async def replies_after(tid, after_id):
        asked.append(after_id)
        return [("m1", REPLY), ("m2", "Another reply nobody asked to resume.")] if after_id == "u1" else None

    s2, ws2, tts2, _ = make(cursor=cur, host_kw={"replies_after": replies_after})
    t2 = await start(s2, ws2, cursor={"stream": STREAM, "from": 640},
                     after_partial={"message_id": "m1", "sentences": 2})
    await settle(40)
    assert asked == ["u1"]
    assert tts2.said == [R3] and [m["text"] for m in says(ws2)] == [R3]
    assert says(ws2)[0]["message_id"] == "m1" and ws2.json_of("say_end")[0]["last_seq"] == 1
    ws2.hang_up(); await t2


async def test_live_audio_below_the_committed_offset_is_still_a_turn(tmp_path):
    """The belt judges REPLAYED audio only: a client whose offset clock restarted (live
    audio below the committed offset) must never go deaf."""
    cur = CommitCursor(tmp_path)
    cur.set((DEVICE, STREAM), 16000 * 60, meta={"text": "earlier", "message_id": "u0"})
    s, ws, _, host = make(det=RecordingDetector([[], [SpeechStart(), utterance()]]), cursor=cur)
    task = await start(s, ws)
    ws.push_audio(up_pcm(0, 320)); await settle(5)
    ws.push_audio(up_pcm(320, 320)); await settle(20)
    assert [c[1] for c in host.of("submit")] == ["hello there"]
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# replay_last (SPEC §4.6 item 4, C.3): "Tap to continue" and "say that again"
# --------------------------------------------------------------------------- #

async def _played(s, ws, *, frames=None):
    """A voice turn's reply arrives and is sent in full (FakeTTS: instantly)."""
    as_turn(s)
    for f in frames if frames is not None else reply("m1", REPLY, prov="run:1"):
        s.on_app_frame(f)
    await settle(60)
    assert s.speaking


async def test_ready_says_the_session_answers_replay_last():
    s, ws, _, _ = make()
    task = await start(s, ws)
    assert ws.json_of("ready")[0]["replay_last"] is True
    ws.hang_up(); await task


async def test_tap_to_continue_resumes_the_cut_reply_from_the_sentence_the_listener_was_in():
    s, ws, tts, host = make()
    task = await start(s, ws)
    await _played(s, ws)
    assert tts.said == [R1, R2, R3]
    # the tap, two thirds into sentence 2 (its `heard` rides on the frame)
    ws.push_json({"type": "stop_playback", "heard": {"gen": 0, "seq": 2, "played_ms": 130}})
    await settle(5)
    ws.push_json({"type": "heard", "gen": 1, "seq": 2, "played_ms": 140}); await settle(20)
    assert ws.json_of("barge_in") == [{"type": "barge_in", "gen": 1}]
    assert host.of("note_heard") and s._pending_note is not None
    before = len(binaries(ws))
    ws.push_json({"type": "replay_last"}); await settle(60)
    new = says(ws)[3:]
    assert [(m["text"], m["gen"], m["seq"]) for m in new] == [(R2, 1, 4), (R3, 1, 5)]
    assert all(m["message_id"] == "m1" for m in new)
    assert tts.said == [R1, R2, R3]                       # from the kept audio: no engine call
    sent = binaries(ws)[before:]
    assert {b[3] for b in sent} == {4, 5} and all(b[2] == 1 for b in sent)
    assert [m["seq"] for m in ws.json_of("audio_end")][-2:] == [4, 5]
    assert ws.json_of("say_end")[-1]["last_seq"] == 5
    assert s._pending_note is None                        # the listener hears the rest now
    assert s.speaking
    # heard to its end: the row is noted as heard in full (history is not cut at the tap)
    ws.push_json({"type": "playback_done", "gen": 1, "seq": 5}); await settle(20)
    last = host.of("note_heard")[-1]
    assert last[1] == "m1" and last[2] == len(REPLY) and last[4] == ""
    assert not s.speaking
    ws.hang_up(); await task


async def test_a_sentence_heard_to_its_end_is_not_said_again_on_continue():
    s, ws, _tts, _ = make()
    task = await start(s, ws)
    await _played(s, ws)
    ws.push_json({"type": "stop_playback", "heard": {"gen": 0, "seq": 2, "played_ms": 200}})   # all of it
    await settle(20)
    ws.push_json({"type": "replay_last"}); await settle(60)
    assert [m["text"] for m in says(ws)[3:]] == [R3]
    ws.hang_up(); await task


async def test_continue_follows_a_run_still_streaming_after_the_tap():
    """A tap never aborts the run: the rest of the answer streams on into the chat
    (muted for voice). "Tap to continue" picks the reply up from the cut sentence and
    follows the stream again, so the sentences that arrived after the tap — and those
    still to come — are spoken too, once, and `say_end` closes it."""
    s, ws, tts, host = make()
    task = await start(s, ws)
    head = REPLY[:50]                                     # sentence 1 and half of sentence 2
    await _played(s, ws, frames=[frame(type="stream_start", message_id="run:1"),
                                 frame(type="stream_chunk", message_id="run:1", text=head)])
    assert tts.said == [R1]
    ws.push_json({"type": "stop_playback", "heard": {"gen": 0, "seq": 1, "played_ms": 60}})
    await settle(10)
    s.on_app_frame(frame(type="stream_chunk", message_id="run:1", text=REPLY[50:70]))   # muted
    await settle(10)
    assert tts.said == [R1] and len(says(ws)) == 1
    ws.push_json({"type": "replay_last"}); await settle(40)
    s.on_app_frame(frame(type="stream_chunk", message_id="run:1", text=REPLY[70:]))
    s.on_app_frame(done("run:1", "m1", REPLY))
    await settle(60)
    assert [m["text"] for m in says(ws)[1:]] == [R1, R2, R3]
    assert tts.said == [R1, R2, R3]                       # R1 from the kept audio
    assert ws.json_of("say_end")[-1]["last_seq"] == says(ws)[-1]["seq"]
    assert not host.of("submit") and ("abort", True) not in host.calls
    ws.hang_up(); await task


async def test_replay_last_with_no_cut_says_the_last_reply_again():
    s, ws, tts, _ = make()
    task = await start(s, ws)
    await _played(s, ws)
    ws.push_json({"type": "playback_done", "gen": 0, "seq": 3}); await settle(10)
    assert not s.speaking
    ws.push_json({"type": "replay_last"}); await settle(60)
    assert [(m["text"], m["seq"], m["gen"]) for m in says(ws)[3:]] == [(R1, 4, 0), (R2, 5, 0), (R3, 6, 0)]
    assert tts.said == [R1, R2, R3]
    assert ws.json_of("say_end")[-1]["last_seq"] == 6
    ws.hang_up(); await task


async def test_replay_last_is_ignored_while_speaking_thinking_or_with_nothing_kept():
    clock = Clock()
    s, ws, _tts, _ = make(clock=clock)
    task = await start(s, ws)
    ws.push_json({"type": "replay_last"}); await settle(10)       # nothing said yet
    assert not says(ws)
    await _played(s, ws)
    ws.push_json({"type": "replay_last"}); await settle(20)       # still speaking
    assert len(says(ws)) == 3
    ws.push_json({"type": "playback_done", "gen": 0, "seq": 3}); await settle(10)
    s._awaiting_reply = True                                       # a turn is pending
    ws.push_json({"type": "replay_last"}); await settle(20)
    assert len(says(ws)) == 3
    s._awaiting_reply = False
    clock.t += vs.REPLAY_KEEP_S + 1                                # kept no longer
    ws.push_json({"type": "replay_last"}); await settle(20)
    assert len(says(ws)) == 3
    ws.hang_up(); await task


async def test_replay_last_in_phone_voice_mode_is_say_frames_alone():
    s, ws, tts, _ = make()
    task = await start(s, ws, slow_link="always_phone")
    await _played(s, ws)
    assert not binaries(ws) and tts.said == []
    ws.push_json({"type": "playback_done", "gen": 0, "seq": 3}); await settle(10)
    ws.push_json({"type": "replay_last"}); await settle(40)
    assert [m["text"] for m in says(ws)[3:]] == [R1, R2, R3]
    assert not binaries(ws) and tts.said == []
    ws.hang_up(); await task


async def test_a_replay_is_cut_like_any_reply_and_the_note_names_the_right_sentence():
    """The replayed pieces have new seqs; the heard map holds ONE copy of the reply, so a
    tap during the replay notes the sentence it was in (never a doubled heard text)."""
    s, ws, _tts, host = make()
    task = await start(s, ws)
    await _played(s, ws)
    ws.push_json({"type": "playback_done", "gen": 0, "seq": 3}); await settle(10)
    ws.push_json({"type": "replay_last"}); await settle(60)
    ws.push_json({"type": "stop_playback", "heard": {"gen": 0, "seq": 5, "played_ms": 100}})
    await settle(10)
    ws.push_json({"type": "heard", "gen": 1, "seq": 5, "played_ms": 100}); await settle(30)
    note = host.of("note_heard")[-1]
    assert note[1] == "m1"
    assert note[3].startswith(R1 + " A dentist") and R1 not in note[3][len(R1):]
    assert note[4].endswith(R3)
    ws.hang_up(); await task


async def test_say_that_again_replays_the_last_reply_without_a_turn():
    """The whole-utterance path: "Say that again." after a reply finished is not a turn
    (no row, no LLM call); the reply goes out again from its kept audio."""
    ws = FakeWS(query=query("pcm16"))
    tts = FakeTTS(chunks=2, chunk_samples=2400)
    host = Host()
    s = VoiceSession(ws, thread_id=THREAD, settings=settings(merge_window_s=0.0),
                     detector=ScriptedDetector([[SpeechStart(), utterance()]]),
                     stt_engine=FakeSTT("Sorry, say that again please."), tts_engine=tts,
                     host=host.build(), notice=lambda text: np.ones(48, "<i2"), heard_wait_s=0.3,
                     transport=Transport.negotiate(ws, ws.query_params, rate_out=24000))
    task = await start(s, ws)
    for f in reply("m1", REPLY, prov="run:1"):
        s.on_app_frame(f)
    await settle(60)
    ws.push_json({"type": "playback_done", "gen": 0, "seq": 3}); await settle(10)
    ws.push_audio(); await settle(60)
    assert not host.of("submit") and not ws.json_of("transcript")
    assert [m["text"] for m in says(ws)[3:]] == [R1, R2, R3]
    assert tts.said == [R1, R2, R3]
    ws.hang_up(); await task


async def test_say_that_again_with_nothing_kept_is_an_ordinary_turn():
    s, ws, _tts, host = make(det=RecordingDetector([[SpeechStart(), utterance()]]), stt="Say that again.")
    task = await start(s, ws)
    ws.push_audio(up_pcm(0, 320)); await settle(30)
    assert [c[1] for c in host.of("submit")] == ["Say that again."]
    ws.hang_up(); await task


async def test_say_that_again_on_the_streaming_path_discards_the_speculation():
    sstt = FakeStreamingSTT([partial("Say that again.")], final="Say that again.")
    s, ws, tts, host, _det = make_rt([[SpeechStart()], [pause()]], p=(0.95,), sstt=sstt)
    task = await start_rt(s, ws)
    for f in reply("m1", REPLY, prov="run:1"):
        s.on_app_frame(f)
    await settle(60)
    ws.push_json({"type": "playback_done", "gen": 0, "seq": 3}); await settle(10)
    await speak(ws, 2, wait=60)
    assert not host.of("submit")
    assert all(run.discarded for run in host.runs)          # a speculative run is never adopted
    assert [m["text"] for m in ws.json_of("say")[3:]] == [R1, R2, R3]
    assert tts.said == [R1, R2, R3]
    ws.hang_up(); await task


def test_replay_commands_are_whole_utterances_only():
    yes = ["Say that again.", "say it again", "Could you repeat that, please?", "Pardon?",
           "Sorry, what did you say?", "Repeat.", "One more time."]
    no = ["Say that again about the dentist", "repeat the list of groceries", "what time is it",
          "say something", "", "again"]
    assert all(VoiceSession._is_replay_command(t) for t in yes)
    assert not any(VoiceSession._is_replay_command(t) for t in no)


# --------------------------------------------------------------------------- #
# holds an uplink reset must not leave behind
# --------------------------------------------------------------------------- #

async def test_an_uplink_gap_mid_utterance_drops_it_releases_the_hold_and_the_open_mark(tmp_path):
    cur = CommitCursor(tmp_path)
    det = RecordingDetector([[SpeechStart()]])
    s, ws, _, host = make(det=det, cursor=cur)
    task = await start(s, ws)
    ws.push_audio(up_pcm(0, 320)); await settle(10)
    assert cur.entry((DEVICE, STREAM)).open_from is not None
    s._hold = True                                       # as a merge window would have set it
    pid = s._pause_id
    ws.push_audio(up_pcm(16000 * 90, 320)); await settle(10)   # 90 s later: the detector resets
    assert det.resets >= 1 and not s._hold and s._pause_id > pid
    assert cur.entry((DEVICE, STREAM)).open_from is None
    assert not host.of("submit")
    ws.hang_up(); await task


async def test_a_hold_with_no_speech_left_is_released_by_the_watchdog():
    clock = Clock()
    s, ws, _tts, _host, det = make_rt([], clock=clock)
    task = await start_rt(s, ws)
    s._hold = True
    det.active = False
    await settle(10)
    assert s._hold                                        # not yet
    clock.t += vs.HOLD_STUCK_S + 1
    await settle(10)
    assert not s._hold
    ws.hang_up(); await task


async def test_speech_that_resumed_and_ended_during_a_turn_end_leaves_no_hold_behind():
    """The resumed speech's own turn end (serialised behind the first) merges it; the first
    turn end's clean-up must not then re-arm the hold for speech that is already over —
    that held the merged reply until the next utterance."""
    clock = Clock()

    class TwoFinals(FakeStreamingSTT):
        """One final text per utterance, the first one slow (a recogniser call in flight)."""

        def __init__(self):
            super().__init__([partial("Call the shop."), partial("and the bank.")], final_delay_s=0.25)
            self.finals_text = ["Call the shop.", "and the bank."]

        def final(self, *, timeout_s=None):
            super().final(timeout_s=timeout_s)
            self.final_delay_s = 0.0
            return self.finals_text.pop(0)

    sstt = TwoFinals()
    s, ws, _tts, host, _det = make_rt([[SpeechStart()], [pause()], [SpeechStart()], [pause()]],
                                      p=(0.95, 0.95), sstt=sstt, clock=clock)
    task = await start_rt(s, ws)
    await speak(ws, 2, wait=3)
    await asyncio.to_thread(sstt.final_started.wait, 2)
    clock.t += 0.4
    ws.push_audio(); await settle(2)                      # speech resumes during final() ...
    ws.push_audio(); await settle(2)                      # ... and ends at once (its own probe)
    for _ in range(100):
        await settle(5)
        if len(host.of("submit")) >= 2 and not s._ending:
            break
    assert [x[1] for x in host.of("submit")] == ["Call the shop.", "Call the shop and the bank."]
    assert not s._hold and not s._ending
    # the merged row echoes back through the tap, then its reply streams
    s.on_app_frame(frame(type="message", message={"id": "u2", "role": "user",
                                                  "content": "Call the shop and the bank."}))
    for f in reply("m1", REPLY, prov="run:1"):
        s.on_app_frame(f)
    await settle(60)
    assert [m["text"] for m in ws.json_of("say")] == [R1, R2, R3]     # the merged reply is spoken
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# send-ahead pacing (v3): the reply's audio never runs far ahead of playback
# --------------------------------------------------------------------------- #

def _audio_s(ws, *, rate: int = 24000) -> float:
    return sum(len(p) // 2 for b in binaries(ws) if b[1] == wire3.KIND_REPLY for p in b[5]) / rate


async def test_reply_audio_is_paced_to_real_time_plus_the_send_ahead():
    """Unpaced, a fast engine wrote a whole reply into the socket at once; on a slow link
    everything sent after it (the phone voice's `say`, `pong`, `barge_in`) queued behind
    it. Paced, the audio sent stays within real time + `send_ahead_s` of the first audio."""
    tts = FakeTTS(chunks=10, chunk_samples=4800)              # 2 s per sentence, instantly
    s, ws, _tts, _ = make(tts=tts, send_ahead_s=0.5)
    task = await start(s, ws)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    for f in reply("m1", REPLY, prov="run:1"):
        s.on_app_frame(f)
    await asyncio.sleep(0.6)
    elapsed = loop.time() - t0
    sent = _audio_s(ws)
    assert 0.5 <= sent <= elapsed + 0.5 + 0.2 + 0.05, (sent, elapsed)    # + one chunk
    assert len(says(ws)) == 1                                 # sentence 2 has not started
    # a barge-in is never stuck behind the pacing wait
    ws.push_json({"type": "stop_playback"}); await settle(20)
    assert ws.json_of("barge_in")
    after = _audio_s(ws)
    await asyncio.sleep(0.3)
    assert _audio_s(ws) == after                              # nothing more of the cut reply
    ws.hang_up(); await task


async def test_a_drained_client_restarts_the_pacing_clock_and_v2_is_never_paced():
    s, ws, _tts, _ = make(tts=FakeTTS(chunks=10, chunk_samples=4800), send_ahead_s=0.5)
    assert s.send_ahead_s == 0.5
    v2, _ws2, _t2, _h2 = make(q={})
    assert v2.send_ahead_s is None                            # the v2 wire is unchanged
    task = await start(s, ws)
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": "Short one."}))
    await asyncio.sleep(1.9)                                   # 2 s of audio: paced over ~1.5 s
    assert ws.json_of("audio_end")
    ws.push_json({"type": "playback_done", "gen": 0, "seq": 1}); await settle(5)
    t0 = asyncio.get_running_loop().time()
    s.on_app_frame(frame(type="message", message={"id": "m2", "role": "assistant", "content": "Next one."}))
    await settle(10)
    # the drained client starts a fresh budget: the next reply's first chunks go out at once
    first = [b for b in binaries(ws) if b[3] == 2]
    assert first and asyncio.get_running_loop().time() - t0 < 0.3
    ws.hang_up(); await task


async def test_a_replay_burst_is_fed_to_the_detector_a_second_at_a_time():
    """review-T3s3 #13: the detector holds its lock for a whole feed call, and the session's
    decision calls take it on the event loop — a coalesced replay burst is fed ≤ 1 s a call."""
    class SizedDetector(RecordingDetector):
        def __init__(self, script):
            super().__init__(script)
            self.calls: list[int] = []

        def feed(self, pcm):
            self.calls.append(len(pcm))
            return super().feed(pcm)

    det = SizedDetector([])
    s, ws, _, _ = make(det=det)
    task = await start(s, ws)
    await s._feed_detector([b"\x00\x00" * 16000] * 3)      # a coalesced batch: 3 s of audio
    assert det.fed == 48000 and len(det.calls) == 3 and max(det.calls) <= vs.FEED_MAX_BYTES
    ws.hang_up(); await task


async def test_the_subscribe_snapshot_waits_for_the_first_start_and_its_after_partial():
    """What `ws.serve` does on a reconnect: it subscribes (the snapshot of the reply still
    streaming is queued) BEFORE the session runs, and the client's `start` — with
    `after_partial` — is read only after `ready`. The snapshot must wait for it: spoken
    any earlier, the reconnect said the cut reply from its start (measured: advisor,
    stall-5s, 122/100 words heard)."""
    s, ws, tts, _ = make()
    s.on_app_frame(frame(type="stream_start", message_id="run:1"))
    s.on_app_frame(frame(type="stream_chunk", message_id="run:1", replace=True, text=REPLY[:70]))
    ws.push_json({"type": "start", "acks": "off", "after_partial": {"message_id": "run:1", "sentences": 1}})
    task = asyncio.create_task(s.run())
    await settle(20)
    s.on_app_frame(frame(type="stream_chunk", message_id="run:1", text=REPLY[70:]))
    s.on_app_frame(done("run:1", "m1", REPLY))
    await settle(40)
    assert tts.said == [R2, R3]
    assert [m["seq"] for m in says(ws)] == [1, 2] and ws.json_of("say_end")[0]["last_seq"] == 2
    ws.hang_up(); await task


async def test_a_reply_streaming_at_subscribe_and_stored_by_the_start_is_spoken_once():
    """The stream finished between the subscribe (snapshot) and the `start` (its row is
    stored): the queued snapshot is handled before the stored replies are looked up, so
    the row is recognised as the same reply and spoken once, from the stream."""
    async def replies_after(tid, after_id):
        return [("m1", REPLY)] if after_id == "u0" else None

    s, ws, tts, _ = make(host_kw={"replies_after": replies_after})
    s.on_app_frame(frame(type="stream_start", message_id="run:1"))
    s.on_app_frame(frame(type="stream_chunk", message_id="run:1", replace=True, text=REPLY[:70]))
    ws.push_json({"type": "start", "acks": "off", "after": "u0",
                  "after_partial": {"message_id": "run:1", "sentences": 1}})
    task = asyncio.create_task(s.run())
    await settle(20)
    s.on_app_frame(frame(type="stream_chunk", message_id="run:1", text=REPLY[70:]))
    s.on_app_frame(done("run:1", "m1", REPLY))
    await settle(40)
    assert tts.said == [R2, R3]
    ws.hang_up(); await task
