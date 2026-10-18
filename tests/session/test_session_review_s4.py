"""Session hub, the review of slice 4 (`impl/review-T3s4.md`), pinned with fakes (no models,
no DisPatch):

* #1 a resume loses no sentence: the reply's pieces depend on its TEXT alone (the session
  feeds the splitter canonically), so `after_partial.sentences` names the same pieces in a
  new session that split the snapshot / the stored row whole; "Tap to continue" skips by
  SOURCE OFFSET; "repeat the last part" says the last sentence only;
* #2 the send-ahead pacing clock never survives a `barge_in` (or a continue), and a dropped
  piece never counts as audio the client plays; #7 the PCM16 downlink's lead is capped and
  paced audio goes out in slices;
* #3 `eot.upto`, the commit and the open mark come from the detector's own position, never
  the end of a coalesced batch: a second utterance of a replay burst is not dropped;
* #5 a reply whose text is REWRITTEN (`replace`, a final content that differs) is never
  spoken as a splice;
* #8 a full queue never blocks the receive loop (replays have their own allowance); #9 the
  per-session `after_partial` state is bounded; #10 a big Opus batch is decoded off the
  event loop; #11 `replay_last` only while listening and never during a turn end;
* a session that ends during the host submit never cuts it in half.
"""
from __future__ import annotations

import asyncio
import random
import threading
import time

import numpy as np
import pytest
from fakes import FakeTTS

from dispatch_voice.host import session as vs
from dispatch_voice.host.session import feed_canonical, split_reply
from dispatch_voice.host.vad_turn import SpeechStart, TurnEnd
from dispatch_voice.text.speech_text import SentenceSplitter
from dispatch_voice.transport import CommitCursor, opus, wire3
from settling import pause as real_pause
from session.test_session import settle, wait_until
from session.test_session_resume import as_turn, done, make, says, start
from session.test_session_s2 import frame
from session.test_session_v3_wire import (
    binaries,
    query,
    up_pcm,
)

# A reply whose FIRST sentence is long and has a comma after 20 characters: while it
# streams the splitter cuts it at the comma (the first words matter for latency); split
# whole it used not to be (review-T3s4 #1, measured on the real stack).
COMMA_REPLY = ("Answer 3. The dentist is at three, and the school call is at four thirty. "
               "Traffic looks light on the way there. I can remind you ten minutes before each one. "
               "Your next free slot is tomorrow at ten.")

GOLDEN = [
    COMMA_REPLY,
    ("The dentist is at three this afternoon, and the school call is at four thirty. "
     "Traffic looks light on the way there. I can remind you ten minutes before each one. "
     "The package from the hardware store shipped this morning. It should arrive on Thursday before noon."),
    "You have two things this afternoon. A dentist appointment at three. Want a reminder?",
    "Yes. No. Maybe so. Sure thing, that works for me, and also for the others who asked earlier today.",
    ("Dr. Smith arrived at 3 p.m. on Tuesday, e.g. right after lunch. No. 5 is next; no, not that one. "
     "Well... maybe later?! Fine."),
    ("\"Hello there,\" she said, smiling at the crowd that had gathered by the old stone gate. "
     "*She waves slowly and steps back into the shade of the tree.* Then she turned away."),
    "Here is the list:\n\n- milk\n- eggs and bread\n- 2 apples\n\nThat's all for today, I think.",
    ("Run this:\n```\nls -la /tmp\necho done\n```\nThen check the output carefully before you go on, "
     "because the next step depends on it."),
    ("This is a very long first sentence without any full stop that keeps going on and on about the "
     "weather and the traffic and the shopping and the school run and the dentist and everything else "
     "that might come up during a long drive across the whole county on a rainy afternoon in autumn so "
     "that it passes every limit the splitter has for one sentence and then some more words follow"),
    "Okay! [[pic:a cat in a hat|cat]] Here it is :react:happy: and that's the picture. 🎉 Enjoy.",
    "  Leading spaces and\ttabs.  Two  spaces between words.\nA new line here. ",
]


def chunkings(text: str, rng: random.Random, n: int = 12):
    yield [text]
    yield list(text)
    for _ in range(n):
        out, i = [], 0
        while i < len(text):
            k = rng.randint(1, 13)
            out.append(text[i:i + k])
            i += k
        yield out


def streamed(chunks: list[str]):
    sp, pending, out = SentenceSplitter(), "", []
    for c in chunks:
        got, pending = feed_canonical(sp, pending, c, final=False)
        out += got
    got, _ = feed_canonical(sp, pending, "", final=True)
    return out + got


# --------------------------------------------------------------------------- #
# #1 the split depends on the text alone
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text", GOLDEN, ids=range(len(GOLDEN)))
def test_the_split_depends_on_the_text_alone(text):
    """Property: however the deltas are cut (whole, one character at a time, random sizes),
    the session's canonical feed yields the same spans — text AND source offsets — as
    `split_reply` (what a resume splits the stored row / snapshot with)."""
    want = [(s.text, s.start, s.end) for s in split_reply(text)]
    assert want                                            # every golden text says something
    rng = random.Random(len(text))
    for ch in chunkings(text, rng):
        got = [(s.text, s.start, s.end) for s in streamed(ch)]
        assert got == want, (ch[:6], got, want)


def test_the_comma_reply_streamed_raw_was_not_its_whole_split():
    """The defect the canonical feed removes, kept visible: T5's raw `push()` cuts the first
    sentence at the comma when the full stop arrives late, a whole `split()` does not."""
    sp = SentenceSplitter()
    raw = []
    for i in range(0, len(COMMA_REPLY), 6):
        raw += [s.text for s in sp.push(COMMA_REPLY[i:i + 6])]
    raw += [s.text for s in sp.flush()]
    assert raw != [s.text for s in SentenceSplitter().split(COMMA_REPLY)]
    assert [s.text for s in split_reply(COMMA_REPLY)] == [s.text for s in streamed(
        [COMMA_REPLY[i:i + 6] for i in range(0, len(COMMA_REPLY), 6)])]


async def _old_session_heard(text: str, *, upto: int, n_heard: int, step: int = 6) -> list[str]:
    """An old session that received the reply as deltas up to ``upto`` characters; the
    listener heard its first ``n_heard`` pieces in full."""
    s1, ws1, _t1, _h1 = make()
    task1 = await start(s1, ws1)
    s1.on_app_frame(frame(type="stream_start", message_id="run:a"))
    for i in range(0, upto, step):
        s1.on_app_frame(frame(type="stream_chunk", message_id="run:a", text=text[i:min(i + step, upto)]))
    await settle(60)
    old = [m["text"] for m in says(ws1)]
    ws1.hang_up(); await task1
    assert len(old) >= n_heard
    return old[:n_heard]


async def test_a_resume_from_the_snapshot_loses_nothing_and_repeats_nothing():
    """review-T3s4 #1, the advisor case (an open stream at the rejoin): the new session gets
    the subscribe snapshot (the text so far, whole) and the rest as deltas."""
    heard = await _old_session_heard(COMMA_REPLY, upto=90, n_heard=2)
    assert heard[1].startswith("and the school call")       # the first sentence WAS clause-cut
    s2, ws2, _t2, _h2 = make()
    task2 = await start(s2, ws2, after_partial={"message_id": "run:a", "sentences": len(heard)})
    for f in [frame(type="stream_start", message_id="run:a"),
              frame(type="stream_chunk", message_id="run:a", text=COMMA_REPLY[:90], replace=True),
              frame(type="stream_chunk", message_id="run:a", text=COMMA_REPLY[90:]),
              done("run:a", "m1", COMMA_REPLY)]:
        s2.on_app_frame(f)
    await settle(60)
    new = [m["text"] for m in says(ws2)]
    ws2.hang_up(); await task2
    assert " ".join(heard + new).split() == COMMA_REPLY.split()
    assert new[0] == "Traffic looks light on the way there."


async def test_a_resume_from_the_stored_row_loses_nothing_and_repeats_nothing():
    """review-T3s4 #1, the API-bot case (the row is stored before its stream): the new
    session finds the reply through `replies_after` and splits the row whole."""
    heard = await _old_session_heard(COMMA_REPLY, upto=120, n_heard=2)

    async def replies_after(tid, after):
        return [("m1", COMMA_REPLY)] if after == "u0" else None
    s2, ws2, _t2, _h2 = make(host_kw={"replies_after": replies_after})
    task2 = await start(s2, ws2, after="u0", after_partial={"message_id": "run:a", "sentences": 2})
    await settle(60)
    new = [m["text"] for m in says(ws2)]
    ws2.hang_up(); await task2
    assert " ".join(heard + new).split() == COMMA_REPLY.split()


async def test_a_late_joiner_without_a_snapshot_skips_exactly_what_was_heard():
    """The C.5 #11 belt: the new session dropped the tail's deltas (no stream_start, no
    snapshot) and speaks the reply from `stream_done` minus `after_partial`."""
    heard = await _old_session_heard(COMMA_REPLY, upto=90, n_heard=2)
    s2, ws2, _t2, _h2 = make()
    task2 = await start(s2, ws2, after_partial={"message_id": "run:a", "sentences": 2})
    for i in range(90, len(COMMA_REPLY), 7):
        s2.on_app_frame(frame(type="stream_chunk", message_id="run:a", text=COMMA_REPLY[i:i + 7]))
    s2.on_app_frame(done("run:a", "m1", COMMA_REPLY))
    await settle(60)
    new = [m["text"] for m in says(ws2)]
    ws2.hang_up(); await task2
    assert " ".join(heard + new).split() == COMMA_REPLY.split()


async def _spoken_then_cut(text: str, *, piece: int, played_ms: int):
    """A reply streamed in as deltas and said; the listener taps while in ``piece``
    (0-based) after ``played_ms`` of it."""
    s, ws, _tts, _host = make(tts=FakeTTS(chunks=2, chunk_samples=2400))
    task = await start(s, ws)
    as_turn(s)
    s.on_app_frame(frame(type="stream_start", message_id="run:a"))
    for i in range(0, len(text), 6):
        s.on_app_frame(frame(type="stream_chunk", message_id="run:a", text=text[i:i + 6]))
    s.on_app_frame(done("run:a", "m1", text))
    await settle(80)
    spoken = [(m["seq"], m["text"]) for m in says(ws)]
    seq, words = spoken[piece]
    ws.push_json({"type": "stop_playback",
                  "heard": {"gen": 0, "seq": seq, "played_ms": played_ms, "source": "server"}})
    await settle(30)
    return s, ws, task, spoken, words


async def test_tap_to_continue_starts_at_the_sentence_the_listener_was_in():
    """review-T3s4 #1(b): the continue skips by source offset; it begins with the very piece
    the listener was in (the reply streamed with its first sentence clause-cut)."""
    _s, ws, task, spoken, words = await _spoken_then_cut(COMMA_REPLY, piece=2, played_ms=100)
    assert spoken[1][1].startswith("and the school call")
    n0 = len(says(ws))
    ws.push_json({"type": "replay_last"})
    await settle(60)
    cont = [m["text"] for m in says(ws)[n0:]]
    ws.hang_up(); await task
    assert cont and cont[0] == words
    assert cont == [t for _q, t in spoken[2:]]


async def test_tap_to_continue_after_a_sentence_heard_to_its_end_starts_at_the_next():
    _s, ws, task, spoken, _words = await _spoken_then_cut(COMMA_REPLY, piece=1, played_ms=100_000)
    n0 = len(says(ws))
    ws.push_json({"type": "replay_last"})
    await settle(60)
    cont = [m["text"] for m in says(ws)[n0:]]
    ws.hang_up(); await task
    assert cont == [t for _q, t in spoken[2:]]


async def test_repeat_the_last_part_says_the_last_sentence_only():
    """review-T3s4 #11: "repeat the last part" replayed the whole reply."""
    from session.test_session_s3 import FakeStreamingSTT, partial, pause, speak
    from session.test_session_s3 import make as make_rt
    from session.test_session_s3 import start as start_rt
    sstt = FakeStreamingSTT([partial("Repeat the last part.")], final="Repeat the last part.")
    s, ws, _tts, host, _det = make_rt([[SpeechStart()], [pause()]], p=(0.95,), sstt=sstt)
    task = await start_rt(s, ws)
    text = "You have two things this afternoon. A dentist appointment at three. Want a reminder?"
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": text}))
    await settle(60)
    ws.push_json({"type": "playback_done", "gen": 0, "seq": 3}); await settle(10)
    await speak(ws, 2, wait=60)
    ws.hang_up(); await task
    assert not host.of("submit")
    assert [m["text"] for m in ws.json_of("say")[3:]] == ["Want a reminder?"]


# --------------------------------------------------------------------------- #
# #2 / #7 pacing
# --------------------------------------------------------------------------- #

LONG = ("This is the first sentence of the reply, long enough. "
        "Here is a second sentence that also runs a while. "
        "A third sentence follows the second one closely. "
        "And a fourth sentence closes the whole reply now.")


async def _first_audio_after(ws, gen: int, t0: float, limit: float = 6.0) -> float | None:
    loop = asyncio.get_running_loop()
    while loop.time() - t0 < limit:
        if [b for b in binaries(ws) if b[2] == gen]:
            return loop.time() - t0
        await real_pause(0.01)
    return None


async def test_the_reply_after_a_tap_is_not_held_behind_the_cut_audio(monkeypatch):
    """review-T3s4 #2 (B2): Kokoro-like whole-sentence chunks, a tap-stop 1 s into the reply,
    the next reply 0.5 s later: its first audio goes out at once (it waited 1.9 s). The lead
    and the slice are set as the review measured them (2 s, whole chunks)."""
    monkeypatch.setattr(vs, "PCM_SEND_AHEAD_S", 2.0)
    monkeypatch.setattr(vs, "PACE_SLICE_S", 2.0)
    s, ws, _tts, _host = make(tts=FakeTTS(chunks=1, chunk_samples=24000 * 2), send_ahead_s=2.0)
    task = await start(s, ws)
    as_turn(s)
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": LONG}))
    await asyncio.sleep(1.0)
    ws.push_json({"type": "stop_playback"}); await settle(20)
    gen = ws.json_of("barge_in")[-1]["gen"]
    await asyncio.sleep(0.5)
    loop = asyncio.get_running_loop()
    as_turn(s)
    t0 = loop.time()
    s.on_app_frame(frame(type="message", message={"id": "m2", "role": "assistant", "content": "New answer here."}))
    first = await _first_audio_after(ws, gen, t0)
    ws.hang_up(); await task
    assert first is not None and first < 0.3, first


async def test_tap_to_continue_sends_the_kept_audio_at_once(monkeypatch):
    """review-T3s4 #2 (B3): the continue's audio is not held behind the cut reply's."""
    monkeypatch.setattr(vs, "PCM_SEND_AHEAD_S", 2.0)
    monkeypatch.setattr(vs, "PACE_SLICE_S", 2.0)
    s, ws, _tts, _host = make(tts=FakeTTS(chunks=1, chunk_samples=24000 * 2), send_ahead_s=2.0)
    task = await start(s, ws)
    as_turn(s)
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": LONG}))
    await asyncio.sleep(1.0)
    ws.push_json({"type": "stop_playback", "heard": {"gen": 0, "seq": 1, "played_ms": 900, "source": "server"}})
    await settle(20)
    gen = ws.json_of("barge_in")[-1]["gen"]
    await asyncio.sleep(0.3)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    ws.push_json({"type": "replay_last"})
    first = await _first_audio_after(ws, gen, t0)
    ws.hang_up(); await task
    assert first is not None and first < 0.3, first


async def test_a_piece_dropped_while_paced_never_counts_as_played():
    s, ws, _tts, _host = make(send_ahead_s=0.5)
    task = await start(s, ws)
    s._client_drained = False
    s._pace_t0, s._pace_sent = time.monotonic(), 5.0                     # 4.5 s too far ahead
    waiter = asyncio.create_task(s._paced(s.gen, 1.0))
    await settle(5)
    s.gen += 1                                       # a barge-in while it waits
    assert await asyncio.wait_for(waiter, 1.0) is False
    assert s._pace_sent == 5.0
    ws.hang_up(); await task


async def test_a_pcm_downlink_keeps_at_most_the_capped_lead_in_flight(monkeypatch):
    """review-T3s4 #7: 384 kbps of PCM16 — the lead is PCM_SEND_AHEAD_S (not 4 s) and a
    whole-sentence chunk goes out in PACE_SLICE_S slices, each paced."""
    sent: list[tuple[float, int]] = []
    s, ws, _tts, _host = make(tts=FakeTTS(chunks=1, chunk_samples=24000 * 3))   # one 3 s chunk
    assert s._send_ahead_now() == vs.PCM_SEND_AHEAD_S
    loop = asyncio.get_running_loop()
    real = s.tr.send_audio

    async def spy(**kw):
        sent.append((loop.time(), len(kw["pcm16"])))
        return await real(**kw)
    monkeypatch.setattr(s.tr, "send_audio", spy)
    task = await start(s, ws)
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": "One long sentence here."}))
    assert await wait_until(lambda: sum(n for _t, n in sent) >= 24000 * 3, timeout=8.0)
    t0 = sent[0][0]
    audio = 0.0
    for t, n in sent:
        ahead = audio - (t - t0)                     # seconds already sent beyond real time
        assert ahead <= vs.PCM_SEND_AHEAD_S + 0.1, (round(t - t0, 3), round(audio, 3))
        audio += n / 24000
    assert max(n for _t, n in sent) <= int(vs.PACE_SLICE_S * 24000)
    assert len(ws.json_of("audio_end")) == 1         # still ONE sentence on the wire
    ws.hang_up(); await task


@pytest.mark.skipif(not opus.available(), reason="libopus is not installed")
def test_an_opus_downlink_keeps_the_full_lead():
    s, _ws, _tts, _host = make(q=query("opus"))
    assert s.tr.codec_out == "opus" and s._send_ahead_now() == vs.SEND_AHEAD_S


# --------------------------------------------------------------------------- #
# #3 positions inside a coalesced batch
# --------------------------------------------------------------------------- #

def _te(silence_s: float) -> TurnEnd:
    return TurnEnd(audio=np.zeros(16000, np.float32), reason="silence", turn_prob=0.9, silence_s=silence_s)


async def test_eot_upto_inside_a_two_second_message_is_the_detectors_position():
    """review-T3s4 #3 (A): one 2 s message, the utterance ends in its FIRST second with 0.5 s
    of trailing silence: `eot.upto` is 8 000, not the batch end minus the silence (24 000)."""
    s, ws, _tts, _host = make([[SpeechStart(), _te(0.5)], []])
    task = await start(s, ws)
    ws.push_audio(up_pcm(0, 32000))
    assert await wait_until(lambda: ws.json_of("eot"))
    ws.hang_up(); await task
    assert ws.json_of("eot")[0]["upto"] == 8000
    assert s._pos == 32000                           # the batch end is restored after the feed


async def test_a_second_utterance_in_one_replay_message_is_not_dropped(tmp_path):
    """review-T3s4 #3 (A2): the first utterance committed the cursor at its real end (8 000),
    so the second (ends at 22 400) is not "already submitted" by the C.5 #8 belt."""
    cur = CommitCursor(tmp_path)
    s, ws, _tts, host = make([[SpeechStart(), _te(0.5)], [SpeechStart(), _te(0.6)]], cursor=cur)
    task = await start(s, ws)
    ws.push_audio(up_pcm(0, 32000, flags=wire3.FLAG_REPLAY))
    assert await wait_until(lambda: len(ws.json_of("eot")) == 2)
    await settle(20)
    ws.hang_up(); await task
    assert [e["upto"] for e in ws.json_of("eot")] == [8000, 22400]
    assert len(host.of("submit")) == 2


async def test_the_real_detectors_positions_mark_the_eot_and_the_open_utterance(tmp_path):
    """With the real TurnDetector (an energy stand-in for Silero) fed three 1 s messages:
    0.5 s silence, 1 s of speech, 1.5 s silence. `eot.upto` is the end of the speech (to a
    32 ms frame) and the open mark is its start minus the pre-roll — positions INSIDE the
    messages (the end is decided 0.8 s into the third one), not their ends."""
    from dispatch_voice.host.vad_turn import Endpointer, EndpointerConfig, TurnDetector
    cfg = EndpointerConfig(min_speech_s=0.1, pre_roll_s=0.3, turn_hard_silence_s=0.8,
                           no_smart_turn_silence_s=0.8)
    det = TurnDetector(lambda f: 1.0 if float(np.abs(f).mean()) > 0.05 else 0.0, Endpointer(cfg))
    marks: list[int] = []
    cur = CommitCursor(tmp_path)
    s, ws, _tts, _host = make(det=det, cursor=cur)
    real_open = s._cursor_open

    def spy(start=None):
        marks.append(start)
        real_open(start)
    s._cursor_open = spy
    task = await start(s, ws)
    pcm = np.zeros(48000, "<i2")
    pcm[8000:24000] = 8000                            # speech from 0.5 s to 1.5 s
    for i in range(0, 48000, 16000):                  # three 1 s messages (the 64 KB cap)
        ws.push_audio(wire3.pack_up(wire3.CODEC_PCM16, 0, 1000 + i, [pcm[i:i + 16000].tobytes()]))
        await wait_until(lambda: s._audio_q.empty() and not s._audio_q_bytes)
        await settle(10)
    assert await wait_until(lambda: ws.json_of("eot"))
    ws.hang_up(); await task
    upto = ws.json_of("eot")[0]["upto"]
    assert 1000 + 24000 - 512 <= upto <= 1000 + 24000 + 512, upto
    assert marks and marks[0] is not None
    assert 1000 + 8000 - 4800 - 512 <= marks[0] <= 1000 + 8000, marks


# --------------------------------------------------------------------------- #
# #5 a rewritten reply is never spliced
# --------------------------------------------------------------------------- #

async def test_a_rewrite_by_replace_is_said_from_the_changed_sentence_never_spliced():
    """review-T3s4 #5 (E): the advisor's provider fallback restarts the text. The sentence
    already said stays said; the new text is spoken from its start; no "firstsit"."""
    s, ws, _tts, _host = make()
    task = await start(s, ws)
    as_turn(s)
    s.on_app_frame(frame(type="stream_start", message_id="run:a"))
    s.on_app_frame(frame(type="stream_chunk", message_id="run:a", text="Let me check the calendar for you. The first"))
    await settle(30)
    new = "Sorry, that provider failed. Your dentist visit is at three tomorrow."
    s.on_app_frame(frame(type="stream_chunk", message_id="run:a", replace=True, text=new))
    s.on_app_frame(done("run:a", "m1", new))
    await settle(40)
    ws.hang_up(); await task
    assert [m["text"] for m in says(ws)] == ["Let me check the calendar for you.",
                                             "Sorry, that provider failed.",
                                             "Your dentist visit is at three tomorrow."]
    assert ws.json_of("say_end") and ws.json_of("say_end")[-1]["last_seq"] == says(ws)[-1]["seq"]


async def test_a_rewrite_before_anything_was_said_replaces_the_queued_pieces():
    """Nothing of the old text was said yet (the turn's merge hold): its queued pieces are
    dropped and only the new text is spoken."""
    s, ws, _tts, _host = make()
    task = await start(s, ws)
    as_turn(s)
    s._hold = True
    s.on_app_frame(frame(type="stream_start", message_id="run:a"))
    s.on_app_frame(frame(type="stream_chunk", message_id="run:a", text="Old words here today. More old words follow. "))
    await settle(30)
    assert not says(ws)
    new = "Fresh words come instead now. And a second one follows."
    s.on_app_frame(frame(type="stream_chunk", message_id="run:a", replace=True, text=new))
    s.on_app_frame(done("run:a", "m1", new))
    await settle(20)
    s._hold = False
    await settle(60)
    ws.hang_up(); await task
    assert [m["text"] for m in says(ws)] == ["Fresh words come instead now.", "And a second one follows."]
    assert {b[3] for b in binaries(ws)} == {m["seq"] for m in says(ws)}    # no audio of the old pieces


async def test_a_final_content_that_differs_only_in_markup_is_not_said_twice():
    """The stored row drops a directive and a link's URL: the spoken words are the same, so
    nothing is said again — the heard map's spans move to the stored content's offsets (a
    later cut notes the right `upto`)."""
    s, ws, _tts, _host = make()
    task = await start(s, ws)
    as_turn(s)
    streamed_text = "[[reply_to_current]] See [the map](https://example.org/x). It is close by. Turn left."
    stored = "See the map. It is close by. Turn left."
    s.on_app_frame(frame(type="stream_start", message_id="run:a"))
    for i in range(0, len(streamed_text), 9):
        s.on_app_frame(frame(type="stream_chunk", message_id="run:a", text=streamed_text[i:i + 9]))
    await settle(30)
    s.on_app_frame(done("run:a", "m1", stored))
    await settle(60)
    texts = [m["text"] for m in says(ws)]
    assert texts == ["See the map.", "It is close by.", "Turn left."]
    second = says(ws)[1]["seq"]
    span = s.heard.span_of(second)                   # (a cut there notes an offset into `stored`)
    assert stored[span.start:span.end].strip() == "It is close by."
    ws.hang_up(); await task


async def test_a_final_content_without_the_trailing_whitespace_changes_nothing():
    s, ws, _tts, _host = make()
    task = await start(s, ws)
    as_turn(s)
    text = "One sentence is here now. And another one follows it.\n\n"
    s.on_app_frame(frame(type="stream_start", message_id="run:a"))
    s.on_app_frame(frame(type="stream_chunk", message_id="run:a", text=text))
    s.on_app_frame(done("run:a", "m1", text.strip()))
    await settle(60)
    ws.hang_up(); await task
    assert [m["text"] for m in says(ws)] == ["One sentence is here now.", "And another one follows it."]


async def test_a_truncating_replace_never_says_a_sentence_twice():
    """A `replace` that takes back the unfinished tail: the sentence already said is
    expected again from the new text and not repeated when its whitespace arrives."""
    s, ws, _tts, _host = make()
    task = await start(s, ws)
    as_turn(s)
    s.on_app_frame(frame(type="stream_start", message_id="run:a"))
    s.on_app_frame(frame(type="stream_chunk", message_id="run:a", text="One two three four. Five six"))
    await settle(30)
    s.on_app_frame(frame(type="stream_chunk", message_id="run:a", replace=True, text="One two three four."))
    s.on_app_frame(frame(type="stream_chunk", message_id="run:a", text=" Seven eight nine ten."))
    s.on_app_frame(done("run:a", "m1", "One two three four. Seven eight nine ten."))
    await settle(60)
    ws.hang_up(); await task
    assert [m["text"] for m in says(ws)] == ["One two three four.", "Seven eight nine ten."]


# --------------------------------------------------------------------------- #
# #8 #9 #10 #11 and the shielded submit
# --------------------------------------------------------------------------- #

async def test_a_replay_burst_never_blocks_the_receive_loop():
    """review-T3s4 #8: with the queue past 64 KB, a replayed message is queued at once
    (its own allowance), live audio behind it too (no hole mid-utterance), and a ping sent
    in between gets its pong — nothing waits on the receive loop."""
    s, _ws, _tts, _host = make()
    s._audio_q_bytes = vs.AUDIO_QUEUE_BYTES - 100           # as if the worker were behind
    await s._enqueue_audio(up_pcm(0, 320))                   # live: dropped
    assert s._dropped_audio == 1
    for i in range(40):                                      # ~26 KB of replay past the bound
        await s._enqueue_audio(up_pcm(320 * i, 320, flags=wire3.FLAG_REPLAY))
    await s._enqueue_audio(up_pcm(320 * 40, 320))            # live, behind the replay
    assert s._audio_q.qsize() == 41 and s._dropped_audio == 1
    s._audio_q_bytes = vs.REPLAY_QUEUE_BYTES - 10
    await s._enqueue_audio(up_pcm(0, 320, flags=wire3.FLAG_REPLAY))
    assert s._dropped_audio == 2                             # even a replay has a bound
    # on the socket: a ping in the middle of a queue that is full is answered at once
    s2, ws2, _t2, _h2 = make()
    s2._audio_q_bytes = vs.AUDIO_QUEUE_BYTES - 100
    task = await start(s2, ws2)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    ws2.push_audio(up_pcm(0, 320, flags=wire3.FLAG_REPLAY))
    ws2.push_json({"type": "ping", "t": 5})
    assert await wait_until(lambda: ws2.json_of("pong"), timeout=2.0)
    assert loop.time() - t0 < 0.5
    ws2.hang_up(); await task


async def test_after_partial_state_is_bounded():
    s, ws, _tts, _host = make()
    task = await start(s, ws)
    for i in range(30):
        ws.push_json({"type": "start", "after_partial": {"message_id": f"run:{i:04d}", "sentences": 2}})
    ws.push_json({"type": "start", "after_partial": {"message_id": "x" * 200, "sentences": 2}})
    await settle(30)
    ws.hang_up(); await task
    assert len(s._after_partial) == vs.AFTER_PARTIAL_KEEP
    assert "run:0029" in s._after_partial and "x" * 200 not in s._after_partial


async def test_a_big_uplink_batch_is_decoded_off_the_event_loop():
    """review-T3s4 #10: a coalesced replay batch is decoded in a worker thread."""
    s, ws, _tts, _host = make()
    threads: list[bool] = []
    real = s._decode_up

    def spy(buf):
        threads.append(threading.current_thread() is threading.main_thread())
        return real(buf)
    s._decode_up = spy
    for i in range(20):                                      # queued before the worker runs
        await s._enqueue_audio(up_pcm(320 * i, 320, flags=wire3.FLAG_REPLAY))
    task = await start(s, ws)
    assert await wait_until(lambda: len(threads) >= 20)
    ws.push_audio(up_pcm(320 * 20, 320))                     # one live message: inline
    assert await wait_until(lambda: len(threads) >= 21)
    ws.hang_up(); await task
    assert not any(threads[:20]) and threads[-1] is True


async def test_replay_last_is_ignored_while_paused_and_during_a_turn_end():
    """review-T3s4 #11: C.3 says "while listening"; a turn end in progress is not idle."""
    s, ws, _tts, _host = make(tts=FakeTTS(chunks=2, chunk_samples=2400))
    task = await start(s, ws)
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": "Hello there friend."}))
    await settle(40)
    ws.push_json({"type": "playback_done", "gen": 0, "seq": 1}); await settle(10)
    ws.push_json({"type": "pause"}); await settle(5)
    ws.push_json({"type": "replay_last"}); await settle(30)
    assert len(says(ws)) == 1                                # paused: nothing
    ws.push_json({"type": "start"}); await settle(5)
    s._ending = 1
    await s._do_replay("whole")
    await settle(30)
    assert len(says(ws)) == 1 and s._say_q.empty()          # a turn end in progress: nothing
    s._ending = 0
    ws.push_json({"type": "replay_last"}); await settle(40)
    assert [m["text"] for m in says(ws)] == ["Hello there friend.", "Hello there friend."]
    ws.hang_up(); await task


async def test_a_session_that_ends_during_the_submit_never_cuts_it_in_half():
    """The cursor commits before the host submit (C.5 #8): a hang-up (or a reconnect that
    replaces the session) while the submit runs lets it finish — a row and its turn."""
    finished = asyncio.Event()
    started = asyncio.Event()

    async def slow_submit(tid, text, opts):
        started.set()
        await asyncio.sleep(0.3)
        finished.set()
        return "u1"
    from fakes import FakeSTT, ScriptedDetector, utterance  # noqa: F401
    s, ws, _tts, _host = make([[SpeechStart(), utterance()]])
    s.host.submit = slow_submit
    task = await start(s, ws)
    ws.push_audio(up_pcm(0, 320))
    await asyncio.wait_for(started.wait(), 2.0)
    ws.hang_up(); await task                               # the session ends mid-submit
    await asyncio.wait_for(finished.wait(), 2.0)           # ...and the submit still completed


async def test_a_rewrite_that_extends_a_queued_sentence_says_it_once():
    """The new text moves a sentence boundary over a piece that was queued but not said: the
    new, longer piece takes its place (never the old piece AND the new one)."""
    s, ws, _tts, _host = make()
    task = await start(s, ws)
    as_turn(s)
    s._hold = True                                   # nothing is said yet
    s.on_app_frame(frame(type="stream_start", message_id="run:a"))
    s.on_app_frame(frame(type="stream_chunk", message_id="run:a",
                         text="The first thing is here. It is close by. "))
    await settle(20)
    new = "The first thing is here. It is close by, two streets down. Turn left there."
    s.on_app_frame(done("run:a", "m1", new))
    await settle(20)
    s._hold = False
    await settle(60)
    ws.hang_up(); await task
    assert [m["text"] for m in says(ws)] == ["The first thing is here.",
                                             "It is close by, two streets down.", "Turn left there."]


async def test_tap_to_continue_after_a_markup_rewrite_loses_nothing():
    """review-T3s4 #1(b), the belt: the continue skips by SOURCE OFFSET. Here the stored row
    differs from the streamed text in markup only, which moves a sentence boundary (the
    directive made the first sentence long enough to stand alone): a count of pieces would
    skip the new first piece and lose "It is close by."; by offset nothing is lost."""
    s, ws, _tts, _host = make(tts=FakeTTS(chunks=2, chunk_samples=2400))
    task = await start(s, ws)
    as_turn(s)
    streamed_text = "[[reply_to_current]] See [the map](https://example.org/x). It is close by. Turn left."
    stored = "See the map. It is close by. Turn left."
    s.on_app_frame(frame(type="stream_start", message_id="run:a"))
    for i in range(0, len(streamed_text), 9):
        s.on_app_frame(frame(type="stream_chunk", message_id="run:a", text=streamed_text[i:i + 9]))
    s.on_app_frame(done("run:a", "m1", stored))
    await settle(80)
    spoken = [(m["seq"], m["text"]) for m in says(ws)]
    assert [t for _q, t in spoken] == ["See the map.", "It is close by.", "Turn left."]
    ws.push_json({"type": "stop_playback",
                  "heard": {"gen": 0, "seq": spoken[1][0], "played_ms": 50, "source": "server"}})
    await settle(30)
    n0 = len(says(ws))
    ws.push_json({"type": "replay_last"})
    await settle(60)
    cont = " ".join(m["text"] for m in says(ws)[n0:])
    ws.hang_up(); await task
    assert "It is close by." in cont and cont.endswith("Turn left.")


async def test_tap_to_continue_resends_the_cut_piece_from_its_kept_audio():
    """A piece the engine finished (Kokoro yields a whole sentence at once) but the tap cut
    while its paced slices were going out is still kept whole: "Tap to continue" starts with
    it from the kept audio — no second engine call, audio at once."""
    tts = FakeTTS(chunks=1, chunk_samples=24000 * 2)     # 2 s per sentence, one chunk each
    s, ws, _tts, _host = make(tts=tts)
    task = await start(s, ws)
    as_turn(s)
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": LONG}))
    # piece 2 is being sent: its first slices are out, the rest paced
    assert await wait_until(lambda: len(says(ws)) >= 2 and [b for b in binaries(ws) if b[3] == says(ws)[1]["seq"]])
    second = says(ws)[1]
    ws.push_json({"type": "stop_playback", "heard": {"gen": 0, "seq": second["seq"], "played_ms": 300,
                                                      "source": "server"}})
    await settle(20)
    gen = ws.json_of("barge_in")[-1]["gen"]
    said_before = list(tts.said)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    ws.push_json({"type": "replay_last"})
    first = await _first_audio_after(ws, gen, t0)
    await settle(20)
    ws.hang_up(); await task
    assert first is not None and first < 0.3, first
    cont = [m for m in says(ws) if m["gen"] == gen]
    assert cont[0]["text"] == second["text"]
    assert tts.said[len(said_before):].count(second["text"]) == 0      # from the kept audio


async def test_tap_and_continue_at_once_still_use_the_kept_audio():
    """The continue is rebuilt while the cut piece is still winding down (its audio is kept
    only when its synthesis loop ends): the kept audio is looked up again when the item is
    played, so the cut piece still goes out from it."""
    tts = FakeTTS(chunks=1, chunk_samples=24000 * 2)
    s, ws, _tts, _host = make(tts=tts)
    task = await start(s, ws)
    as_turn(s)
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": LONG}))
    assert await wait_until(lambda: len(says(ws)) >= 2 and [b for b in binaries(ws) if b[3] == says(ws)[1]["seq"]])
    second = says(ws)[1]
    said_before = list(tts.said)
    ws.push_json({"type": "stop_playback", "heard": {"gen": 0, "seq": second["seq"], "played_ms": 300,
                                                      "source": "server"}})
    ws.push_json({"type": "replay_last"})                 # the same instant
    assert await wait_until(lambda: [m for m in says(ws) if m["gen"] == 1])
    await settle(40)
    ws.hang_up(); await task
    cont = [m for m in says(ws) if m["gen"] == 1]
    assert cont[0]["text"] == second["text"]
    assert tts.said[len(said_before):].count(second["text"]) == 0      # from the kept audio
