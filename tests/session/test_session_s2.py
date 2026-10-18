"""Session hub, slice 2 (SPEC §14 T3 (2)): `seq` at enqueue, `sid`, `say` /
`say_end`, text mode, the acknowledgement timing gate and tool lines, the
role-play rendering with the narrator voice, heard bookkeeping with the
abort → `note_heard` order, the interruption note in the next instructions,
`follow_up: null`, and the phone-voice `ack` earcon — all with fakes (no
models, no DisPatch)."""
from __future__ import annotations

import asyncio

import numpy as np
from fakes import FakeSTT, FakeTTS, FakeWS, ScriptedDetector, utterance

from dispatch_voice.host import session as vs
from dispatch_voice.host.session import VoiceHost, VoiceSession
from dispatch_voice.host.vad_turn import SpeechStart
from settling import pause
from session.test_session import (
    THREAD,
    _NoSkip,
    kit_with_lines,
    settings,
    settle,
    wait_until,
)

REPLY = "You have two things this afternoon. A dentist appointment at three. Want a reminder?"


class PacedTTS(FakeTTS):
    """A long sentence that takes real time to synthesise (2 ms a chunk), so a cut
    "inside the first sentence" stays inside it on a loaded box."""

    def stream(self, text, voice=None, stop=None):
        import time
        for chunk in super().stream(text, voice, stop):
            time.sleep(0.002)
            yield chunk


class Host:
    """A recording VoiceHost whose abort returns a row id and whose note_heard
    records the call order."""

    def __init__(self, *, abort_row: str | None = "a-partial", note_ok: bool = True) -> None:
        self.calls: list[tuple] = []
        self.abort_row = abort_row
        self.note_ok = note_ok
        self.n = 0

    def build(self, **kw) -> VoiceHost:
        async def submit(tid, text, opts):
            self.n += 1
            self.calls.append(("submit", text, opts))
            return f"u{self.n}"

        async def abort(tid, keep_partial=False):
            self.calls.append(("abort", keep_partial))
            return self.abort_row if keep_partial else None

        async def note_heard(tid, mid, *, heard_upto, heard_text, unheard_text):
            self.calls.append(("note_heard", mid, heard_upto, heard_text, unheard_text))
            return self.note_ok

        async def retract(tid, mid):
            self.calls.append(("retract", mid))

        return VoiceHost(submit=submit, abort=abort, note_heard=note_heard, retract=retract, **kw)

    def of(self, name: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == name]


def make(script, *, stt="what's the weather", host: Host | None = None, tts: FakeTTS | None = None, **kw):
    ws = FakeWS()
    host = host or Host()
    tts = tts or FakeTTS()
    s = VoiceSession(ws, thread_id=THREAD, settings=settings(merge_window_s=0.0),
                     detector=ScriptedDetector(script), stt_engine=FakeSTT(stt),
                     tts_engine=tts, host=host.build(), notice=lambda text: np.ones(48, "<i2"),
                     heard_wait_s=0.3, **kw)
    return s, ws, tts, host


def frame(**kw):
    return {"thread_id": THREAD, **kw}


def reply(mid: str, text: str, *, prov: str | None = None):
    """A whole reply as DisPatch streams it: stream_start, 4-word chunks, stream_done."""
    rid = prov or mid
    out = [frame(type="stream_start", message_id=rid)]
    words = text.split(" ")
    for i in range(0, len(words), 4):
        out.append(frame(type="stream_chunk", message_id=rid, text=(" " if i else "") + " ".join(words[i:i + 4])))
    done = frame(type="stream_done", message_id=mid, message={"id": mid, "role": "assistant", "content": text})
    if prov:
        done["provisional_id"] = prov
    out.append(done)
    return out


async def start(s, ws, **extra):
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start", **extra})
    await settle(5)
    return task


def seq_order(ws) -> list[tuple[str, int]]:
    """(type, seq) of every say/audio frame, in wire order."""
    return [(m["type"], m["seq"]) for m in ws.sent
            if isinstance(m, dict) and m.get("type") in ("say", "audio")]


# --------------------------------------------------------------------------- #
# ready, say, say_end, seq
# --------------------------------------------------------------------------- #

async def test_ready_carries_sid_say_langs_acks_and_power_hold():
    s, ws, _, _ = make([], power_state=lambda: "held")
    task = await start(s, ws, acks="often", follow_up=None)
    r = ws.json_of("ready")[0]
    assert len(r["sid"]) == 16 and int(r["sid"], 16) >= 0 and r["sid"] == s.sid
    # `ready` goes out before the first `start` is read: it reports the defaults.
    assert r["say"] is True and r["langs"] == ["en"] and r["acks"] == "sometimes" and r["power_hold"] == "held"
    assert r["follow_up"] is True                    # the bot's default (bot_follow_up)
    await settle(3)
    assert s.acks.mode == "often"                    # start.acks applied
    ws.hang_up(); await task
    s2, ws2, _, _ = make([])
    assert s2.sid != s.sid


async def test_say_precedes_audio_per_seq_and_say_end_closes_the_reply():
    s, ws, tts, _ = make([])
    task = await start(s, ws)
    for f in reply("m1", REPLY, prov="run:1"):
        s.on_app_frame(f)
    await settle(40)
    says = ws.json_of("say")
    assert [m["text"] for m in says] == ["You have two things this afternoon.",
                                         "A dentist appointment at three.", "Want a reminder?"]
    assert [m["seq"] for m in says] == [1, 2, 3] and all(m["kind"] == "reply" and m["role"] == "character"
                                                         for m in says)
    # The first sentences carry the provisional id (what the stream frames
    # carry, like the v2 audio headers); the one cut by stream_done the row id.
    assert [m["message_id"] for m in says] == ["run:1", "run:1", "m1"]
    # C.5 #1: the say of a seq comes before that seq's first audio; audio of
    # seq n is finished before the say of seq n+1 (seq is assigned at enqueue,
    # the say goes out when the sentence's hold releases).
    order = seq_order(ws)
    for q in (1, 2, 3):
        first_say = order.index(("say", q))
        first_audio = order.index(("audio", q))
        assert first_say < first_audio
    assert order.index(("say", 2)) > max(i for i, x in enumerate(order) if x == ("audio", 1))
    assert ws.json_of("say_end") == [{"type": "say_end", "gen": 0, "message_id": "m1", "last_seq": 3}]
    assert all("t" in m for m in says)
    ws.hang_up(); await task


async def test_seq_is_assigned_at_enqueue_before_synthesis():
    """Three sentences queued at once all have their seq before the first is spoken."""
    s, ws, _, _ = make([])
    task = await start(s, ws)
    s._merge_until = s.clock() + 10            # hold everything: nothing is spoken yet
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": REPLY}))
    await settle(5)
    assert s.seq == 3 and s._say_q.qsize() + (1 if s._tts_busy else 0) == 3 and not ws.json_of("say")
    assert s.heard.spans_of("m1") and [q for q, _ in s.heard.spans_of("m1")] == [1, 2, 3]
    s._merge_until = 0
    await settle(30)
    assert [m["seq"] for m in ws.json_of("say")] == [1, 2, 3]
    ws.hang_up(); await task


async def test_a_held_say_is_never_sent_for_an_abandoned_gen():
    """Sentences queued behind a hold (merge window) and cut before it releases
    never get a `say` (C.5 #2): the next reply starts at the next seq, gen + 1."""
    s, ws, tts, _ = make([])
    task = await start(s, ws)
    s._merge_until = s.clock() + 10
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": REPLY}))
    await settle(5)
    assert s.seq == 3 and not ws.json_of("say")
    await s.barge_in(by_user_voice=False, abort=False)
    s._merge_until = 0
    await settle(20)
    assert not ws.json_of("say") and ws.json_of("barge_in")[0]["gen"] == 1
    s.on_app_frame(frame(type="message", message={"id": "m2", "role": "assistant", "content": "Okay then, sure."}))
    await settle(20)
    says = ws.json_of("say")
    assert [(m["gen"], m["seq"], m["text"]) for m in says] == [(1, 4, "Okay then, sure.")]
    ws.hang_up(); await task


async def test_a_cut_reply_gets_no_say_end_and_a_new_gen():
    s, ws, _, _ = make([], tts=FakeTTS(chunks=3000))
    task = await start(s, ws)
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": REPLY}))
    while not ws.json_of("audio"):
        await pause(0.002)
    ws.push_json({"type": "stop_playback"})
    await settle(20)
    assert ws.json_of("barge_in") and not ws.json_of("say_end")
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# text mode (phone voice) and the link frame
# --------------------------------------------------------------------------- #

async def test_text_mode_sends_say_only_and_never_calls_the_engine():
    s, ws, tts, _ = make([])
    task = await start(s, ws)
    ws.push_json({"type": "link", "mode": "text", "why": "underrun"})
    await settle(3)
    for f in reply("m1", REPLY):
        s.on_app_frame(f)
    await settle(20)
    says = ws.json_of("say")
    assert [m["seq"] for m in says] == [1, 2, 3] and tts.said == [] and not ws.json_of("audio")
    assert ws.json_of("say_end")[0]["last_seq"] == 3
    assert list(s._recent_said)[-3:] == [m["text"] for m in says]     # the phone said them
    assert s.speaking and ws.json_of("state")[-1]["state"] == "speaking"
    ws.push_json({"type": "playback_done", "gen": 0, "seq": 3})          # the phone's onend
    await settle(5)
    assert not s.speaking and ws.json_of("state")[-1]["state"] == "listening"
    ws.hang_up(); await task


async def test_mode_change_applies_at_the_next_sentence():
    s, ws, tts, _ = make([], tts=FakeTTS(chunks=3000))
    task = await start(s, ws)
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": REPLY}))
    while not ws.json_of("audio"):
        await pause(0.002)
    ws.push_json({"type": "link", "mode": "text", "why": "stall"})      # mid-sentence
    # (3000 chunks: on a loaded box sentence 1 takes a while to finish — wait for the
    # result, not a fixed 0.6 s: review-T3s3 #13)
    assert await wait_until(lambda: len(ws.json_of("say")) >= 3 and s._say_q.empty() and not s._tts_busy)
    await settle(10)
    audio_seqs = {m["seq"] for m in ws.json_of("audio")}
    assert audio_seqs == {1}                      # sentence 1 finished on its path (C.5 #6)
    assert [m["seq"] for m in ws.json_of("say")] == [1, 2, 3] and tts.said == [REPLY.split(". ")[0] + "."]
    ws.hang_up(); await task


async def test_always_phone_means_text_mode_from_the_start():
    s, ws, tts, _ = make([])
    task = await start(s, ws, slow_link="always_phone")
    assert s.mode == "text"
    ws.push_json({"type": "link", "mode": "audio", "why": "recovered"})   # Settings wins
    await settle(3)
    assert s.mode == "text"
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": "Sure thing, done."}))
    await settle(10)
    assert tts.said == [] and ws.json_of("say")
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# role-play rendering and the narrator voice
# --------------------------------------------------------------------------- #

RP = '*She leans in.* "Tell me more about it." She waits for an answer.'


async def test_narrator_segments_use_the_narrator_voice_and_both_texts_are_recent():
    s, ws, tts, _ = make([], rp_mode="dialogue_narrator", narrator_id="narrator-voice", profile_id="char-voice")
    task = await start(s, ws)
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": RP}))
    await settle(40)
    says = ws.json_of("say")
    roles = [(m["role"], m["text"]) for m in says]
    assert ("narrator", "She leans in.") in roles and ("character", "Tell me more about it.") in roles
    assert ("narrator", "She waits for an answer.") in roles
    voices = dict(zip([m["text"] for m in says], tts.voices, strict=True))
    assert voices["She leans in."] == "narrator-voice" and voices["Tell me more about it."] == "char-voice"
    assert "She leans in." in s._recent_said and "Tell me more about it." in s._recent_said
    # Every piece keeps a source span into the stored markdown (heard tracking).
    spans = [sp for _, sp in s.heard.spans_of("m1")]
    assert [sp.text for sp in spans] == [t for _, t in roles]
    assert all(0 <= sp.start <= sp.end <= len(RP) for sp in spans)
    assert spans[0].start == 0 and spans[-1].end == len(RP)
    ws.hang_up(); await task


async def test_skip_actions_drops_the_action_and_a_missing_narrator_voice_uses_the_character():
    s, ws, tts, _ = make([], rp_mode="skip_actions", profile_id="char-voice")
    task = await start(s, ws)
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": RP}))
    await settle(30)
    assert "She leans in." not in tts.said and "Tell me more about it." in tts.said
    assert set(tts.voices) == {"char-voice"}
    ws.hang_up(); await task
    s, ws, tts, _ = make([], rp_mode="dialogue_narrator", narrator_id=None, profile_id="char-voice")
    task = await start(s, ws)
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant", "content": RP}))
    await settle(30)
    assert set(tts.voices) == {"char-voice"} and any(m["role"] == "narrator" for m in ws.json_of("say"))
    ws.hang_up(); await task


async def test_speech_text_reads_numbers_and_drops_markdown():
    s, ws, tts, _ = make([])
    task = await start(s, ws)
    s.on_app_frame(frame(type="message", message={"id": "m1", "role": "assistant",
                                                  "content": "**Tuesday at 3pm** costs $4.50, see [the map](https://x.io/m)."}))
    await settle(20)
    said = " ".join(tts.said)
    assert "three" in said and "four dollars" in said and "**" not in said and "https" not in said
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# acknowledgements: the gate, text-mode earcon, off, tool lines
# --------------------------------------------------------------------------- #

async def test_ack_line_plays_once_at_the_gate_and_the_reply_queues_behind_it():
    s, ws, tts, host = make([[utterance()]], acks=kit_with_lines({"question": ("Let me see.", "Hmm.")}))
    task = await start(s, ws)
    ws.push_audio()
    # The line is due ack_after_s (50 ms) after the transcript: wait for it, not for a fixed time.
    assert await wait_until(lambda: [m for m in ws.json_of("audio") if m["kind"] == "ack"])
    ack_say = [m for m in ws.json_of("say") if m["kind"] == "ack"]
    ack_audio = [m for m in ws.json_of("audio") if m["kind"] == "ack"]
    assert len(ack_say) == 1 and len(ack_audio) == 1 and ack_say[0]["seq"] == ack_audio[0]["seq"] == 1
    assert ack_say[0]["text"] == "Let me see." and not ws.json_of("ack")   # a line, not the earcon
    for f in reply("m1", "It is sunny all day today."):
        s.on_app_frame(f)
    await settle(20)
    rs = [m for m in ws.json_of("say") if m["kind"] == "reply"]
    assert rs and rs[0]["seq"] == 2                          # the reply queued behind the ack
    metrics = [m for m in ws.json_of("metrics") if "eot_to_first_audio_ms" in m]
    assert metrics and metrics[0]["ack_played"] is True and metrics[0]["ack_eta_ms"] >= 1200
    ws.hang_up(); await task


async def test_no_ack_when_a_reply_is_already_queued_or_the_eta_is_short():
    s, ws, tts, host = make([[utterance()]], acks=kit_with_lines({"question": ("Let me see.",)}, first_chunk_s=0.0))
    s.acks.estimator.prior["ttft"] = 0.5          # ETA 0.5 s < 1.2 s: no ack
    task = await start(s, ws)
    ws.push_audio(); await settle(10)
    assert not [m for m in ws.json_of("say") if m["kind"] == "ack"] and not ws.json_of("ack")
    ws.hang_up(); await task


async def test_reactions_off_means_no_ack_line_and_no_earcon_ever():
    s, ws, _, _ = make([[utterance()]], acks=kit_with_lines({"question": ("Let me see.",)}))
    task = await start(s, ws, acks="off")
    ws.push_audio(); await settle(15)
    assert not ws.json_of("ack") and not [m for m in ws.json_of("say") if m["kind"] == "ack"]
    ws.hang_up(); await task


async def test_no_rendered_line_falls_back_to_the_earcon():
    s, ws, _, _ = make([[utterance()]])              # the silent kit: nothing rendered
    s.acks.selector.rng = _NoSkip()                  # the Reactions roll did not skip
    task = await start(s, ws)
    # The earcon is due ack_after_s (50 ms) after the transcript; a fixed settle(10)
    # (~50 ms) had no margin and failed on a loaded box. Wait for it instead.
    ws.push_audio()
    assert await wait_until(lambda: ws.json_of("ack"), timeout=5.0)
    assert not ws.json_of("say")
    ws.hang_up(); await task


async def test_phone_voice_mode_acks_with_the_earcon_never_a_spoken_line():
    s, ws, tts, _ = make([[utterance()]], acks=kit_with_lines({"question": ("Let me see.",)}))
    task = await start(s, ws, slow_link="always_phone")
    ws.push_audio()
    assert await wait_until(lambda: ws.json_of("ack"))
    assert ws.json_of("ack") and not [m for m in ws.json_of("say") if m["kind"] == "ack"]
    assert not ws.json_of("audio") and tts.said == []
    ws.hang_up(); await task


async def test_tool_lines_once_per_distinct_tool_at_most_two_as_notices():
    kit = kit_with_lines({"question": ("Let me see.",)})
    kit.queue.lines.update({t: np.ones(100, "<i2") for t in ("Searching", "Checking your calendar", "Checking your cards")})
    s, ws, _, _ = make([[utterance()]], acks=kit)
    s.settings = settings(merge_window_s=0.0, ack_after_s=0.02)
    task = await start(s, ws, acks="often")
    ws.push_audio()
    assert await wait_until(lambda: [m for m in ws.json_of("audio") if m["kind"] == "ack"])   # the ack goes first
    s.on_app_frame(frame(type="message", message={"id": "u1", "role": "user", "content": "x"}))
    s.on_app_frame(frame(type="thinking", status="started"))
    for name in ("brave_search", "brave_search", "google_calendar", "flashcards_list"):
        s.on_app_frame(frame(type="progress", item={"kind": "tool", "name": name, "text": "…"}))
    # Tool lines come ≥ 3 s apart (MIN_GAP_S): let the first one go, then the second.
    s._turn_acks.min_gap_s = 0.01
    await settle(80)
    notices = [m for m in ws.json_of("say") if m["kind"] == "notice"]
    assert [m["text"] for m in notices] == ["Searching", "Checking your calendar"]   # once per tool, ≤ 2
    assert [m["kind"] for m in ws.json_of("audio")] == ["ack", "notice", "notice"]
    assert "Searching" in s._recent_said
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# heard tracking: abort → heard → note_heard, the note in the next turn
# --------------------------------------------------------------------------- #

async def _speak_then_interrupt(s, ws, host, *, heard_frame: dict | None):
    """A reply plays; the user talks over it; words confirm the interruption."""
    for f in reply("m1", REPLY, prov="run:1"):
        s.on_app_frame(f)
    while len(ws.json_of("audio")) < 2:
        await pause(0.002)
    ws.push_audio()                                   # SpeechStart: duck
    await settle(5)
    assert ws.json_of("duck") and not ws.json_of("barge_in")
    if heard_frame is not None:
        ws.push_json({"type": "heard", **heard_frame})
        await settle(3)
    ws.push_audio()                                   # TurnEnd: real words
    await settle(10)


async def test_interruption_runs_abort_then_the_exact_heard_then_note_heard():
    tts = PacedTTS(chunks=200)                         # a long first sentence: the cut lands inside it
    host = Host(abort_row="a-partial")
    s, ws, _, _ = make([[SpeechStart()], [utterance()]], stt="wait, what about tomorrow",
                       host=host, tts=tts)
    task = await start(s, ws)
    # A periodic heard first (so the session knows the client sends them)...
    await _speak_then_interrupt(s, ws, host, heard_frame={"gen": 0, "seq": 1, "played_ms": 100, "source": "server"})
    barge = ws.json_of("barge_in")[0]
    assert barge["gen"] == 1
    # ...the session now waits (≤ heard_wait_s) for the exact post-barge_in heard.
    assert host.of("abort") == [("abort", True)] and not host.of("note_heard")
    ws.push_json({"type": "heard", "gen": 1, "seq": 1, "played_ms": 400, "source": "server"})
    await settle(12)
    (n,) = host.of("note_heard")
    assert n[1] == "a-partial"                        # the row abort_turn persisted
    # The cut is inside sentence 1 (a word boundary of the stored text); how far
    # depends on how much audio had gone out, so only the shape is pinned.
    assert 0 <= n[2] <= len("You have two things this afternoon.") and n[4]
    assert "dentist" in n[4] and (n[3] + " " + n[4]).strip().startswith("You have two things")
    # Order: abort before note_heard; the new turn carries the note once.
    names = [c[0] for c in host.calls]
    assert names.index("abort") < names.index("note_heard") < names.index("submit")
    sub = host.of("submit")
    assert sub[0][1] == "wait, what about tomorrow"
    assert sub[0][2]["interrupted"] == {"heard": n[3], "unheard": n[4]}
    assert ws.json_of("transcript")[0]["message_id"] == "u1"
    # The next turn carries no note.
    s.detector.script.append([utterance()])
    ws.push_audio(); await settle(10)
    assert host.of("submit")[1][2]["interrupted"] is None
    ws.hang_up(); await task


async def test_without_any_heard_frame_the_sentence_being_sent_is_the_cut():
    """A 2.1 client never sends `heard`: no wait, the piece being streamed is
    assumed unheard (everything before it heard) — SPEC §10."""
    host = Host(abort_row="a-partial")
    s, ws, _, _ = make([[SpeechStart()], [utterance()]], stt="stop, what about tomorrow",
                       host=host, tts=PacedTTS(chunks=200))
    s.heard_wait_s = 5.0                              # a wait would show (robust under load)
    task = await start(s, ws)
    t0 = s.clock()
    await _speak_then_interrupt(s, ws, host, heard_frame=None)
    assert s.clock() - t0 < 2.0                       # no heard wait for an old client
    (n,) = host.of("note_heard")
    assert n[1] == "a-partial" and n[2] == 0 and n[3] == "" and "You have two things" in n[4]
    assert host.of("submit")[0][2]["interrupted"] == {"heard": "", "unheard": n[4]}
    ws.hang_up(); await task


async def test_note_heard_is_deferred_until_the_row_has_an_id():
    """A gateway bot: abort returns None and the reply's id is provisional until
    stream_done maps it (A7)."""
    host = Host(abort_row=None)
    s, ws, _, _ = make([[SpeechStart()], [utterance()]], stt="hold on, what about tomorrow",
                       host=host, tts=PacedTTS(chunks=200))
    task = await start(s, ws)
    s.on_app_frame(frame(type="stream_start", message_id="run:9"))
    s.on_app_frame(frame(type="stream_chunk", message_id="run:9", text=REPLY))
    while len(ws.json_of("audio")) < 2:
        await pause(0.002)
    ws.push_audio(); await settle(5)
    ws.push_json({"type": "heard", "gen": 0, "seq": 1, "played_ms": 300, "source": "server"})
    await settle(3)
    ws.push_audio(); await asyncio.sleep(0.45)        # past the 0.3 s heard wait of this test
    assert host.of("abort") == [("abort", True)] and not host.of("note_heard")
    assert s._deferred_note is not None
    s.on_app_frame(frame(type="stream_done", message_id="m9", provisional_id="run:9",
                         message={"id": "m9", "role": "assistant", "content": REPLY}))
    await settle(5)
    (n,) = host.of("note_heard")
    assert n[1] == "m9" and s._deferred_note is None
    ws.hang_up(); await task


async def test_a_merge_persists_nothing_and_writes_no_note():
    """The driver resumes inside the merge window: one message; abort with
    keep_partial=False, retract the first row, never note_heard."""
    from session.test_session import SeqSTT
    host = Host(abort_row="a-partial")
    ws = FakeWS()
    s = VoiceSession(ws, thread_id=THREAD, settings=settings(merge_window_s=0.5),
                     detector=ScriptedDetector([[utterance()], [SpeechStart()], [utterance()]]),
                     stt_engine=SeqSTT(["what's on my", "calendar today"]), tts_engine=FakeTTS(),
                     host=host.build(), heard_wait_s=0.3)
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); ws.push_audio(); await settle(5)
    ws.push_audio(); await settle(3)
    ws.push_audio(); await settle(20)
    assert host.of("abort") == [("abort", False)] and not host.of("note_heard")
    assert host.of("retract") == [("retract", "u1")]
    assert host.of("submit")[-1][1] == "what's on my calendar today"
    assert host.of("submit")[-1][2]["interrupted"] is None
    ws.hang_up(); await task


async def test_tap_to_stop_with_a_heard_sub_object_notes_the_cut():
    host = Host(abort_row="a-partial")
    s, ws, _, _ = make([], host=host, tts=PacedTTS(chunks=200))
    task = await start(s, ws)
    for f in reply("m1", REPLY, prov="run:1"):
        s.on_app_frame(f)
    while len(ws.json_of("audio")) < 2:
        await pause(0.002)
    ws.push_json({"type": "stop_playback", "heard": {"gen": 0, "seq": 1, "played_ms": 60, "source": "server"}})
    await asyncio.sleep(0.45)
    assert host.of("abort") == []                     # a tap never aborts the turn
    (n,) = host.of("note_heard")
    assert n[1] == "m1" and "dentist" in n[4]            # the row id stream_done mapped; the rest unheard
    assert ws.json_of("barge_in") and not s.speaking
    assert host.of("submit") == []
    ws.hang_up(); await task


async def test_cancel_while_thinking_persists_nothing():
    host = Host(abort_row="a-partial")
    s, ws, _, _ = make([[utterance()]], host=host)
    task = await start(s, ws)
    ws.push_audio(); await settle(5)
    ws.push_json({"type": "cancel"}); await settle(5)
    assert host.of("abort") == [("abort", False)] and not host.of("note_heard")
    ws.hang_up(); await task


async def test_the_h3_partial_row_is_never_spoken_again():
    s, ws, tts, _ = make([])
    task = await start(s, ws)
    s.on_app_frame(frame(type="message", message={"id": "a-partial", "role": "assistant", "content": "You have two",
                                                  "metadata": {"voice_heard": {"interrupted": True}}}))
    await settle(10)
    assert tts.said == [] and not ws.json_of("say")
    ws.hang_up(); await task


async def test_follow_up_null_means_the_bots_default_in_the_turn_opts():
    host = Host()
    s, ws, _, _ = make([[utterance()], [utterance()]], host=host, bot_follow_up=False)
    task = await start(s, ws, follow_up=None)
    assert ws.json_of("ready")[0]["follow_up"] is False
    ws.push_audio(); await settle(5)
    assert host.of("submit")[0][2]["follow_up"] is None          # the host builds the bot's default
    ws.push_json({"type": "start", "follow_up": True}); await settle(3)
    ws.push_audio(); await settle(5)
    assert host.of("submit")[1][2]["follow_up"] is True and s.effective_follow_up is True
    ws.hang_up(); await task


async def test_heard_frames_are_validated_and_do_not_crash():
    s, ws, _, _ = make([])
    task = await start(s, ws)
    for bad in ({"type": "heard"}, {"type": "heard", "gen": "x", "seq": 1, "played_ms": 1},
                {"type": "heard", "gen": 0, "seq": 1, "played_ms": -5}, {"type": "heard", "gen": 0, "seq": 1,
                                                                        "played_ms": 10, "source": "mars"}):
        ws.push_json(bad)
    await settle(5)
    assert s._last_heard is not None and s._last_heard.source == "server"   # the last one, source defaulted
    assert not ws.json_of("error")
    ws.hang_up(); await task


def test_the_heard_wait_is_the_contract_value():
    # The render queue's gate is "no voice chat open" (host/acks._busy reads
    # ws._SESSIONS); the session's own is_busy() was dead code and is gone
    # (review-T3s2 #8).
    assert vs.HEARD_WAIT_S == 1.5
    assert not hasattr(VoiceSession, "is_busy")
