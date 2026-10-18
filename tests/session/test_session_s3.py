"""Session hub, slice 3 (SPEC §14 T3 (3)): the realtime turn through the session —
TurnPolicy at every probed pause (end / hold + `eot_soft` / reprobe / listen),
StreamingSTT partials (`partial` frames) and `final()`, the Speculator (advisor
only by default, caps, the thread-head stale rule, DisPatch refusing adoption),
the "oh, and…" addendum merge, the word-gated barge-in (echo / backchannel / stop /
real; `resume {gen, rewind}`; the false-interruption timeout), the phone-voice rules
(an informational duck, 0.6 s of speech, never a rewind), learned endpointing,
backchannels, `quiet_check`, the idle-touch policy, and the review-T3s2 fixes
(#1 hang-up during the heard wait, #2 stale `heard`, #3 heard across the id switch,
#4 the exact `heard` after the fallback, #5 earcon vs skip, #6 text-mode ack,
#9 `interrupt_to_submit_ms`) — all with fakes (no models, no DisPatch)."""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import FakeSTT, FakeTTS, FakeWS

from dispatch_voice.host import session as vs
from dispatch_voice.host.session import VoiceHost, VoiceSession
from dispatch_voice.host.vad_turn import (
    Discarded,
    EndpointerConfig,
    Pause,
    SpeechStart,
    TurnEnd,
)
from dispatch_voice.rt.endpoint import Learned, TurnPolicy
from dispatch_voice.rt.settings import RtSettings
from dispatch_voice.rt.stt_stream import Partial
from dispatch_voice.transport import wire3
from session.test_session import THREAD, _NoSkip, kit_with_lines, settings, settle
from session.test_session_s2 import frame, reply
from settling import pause as real_pause
from settling import wait_until

QUESTION = "What's on my calendar this afternoon?"
REPLY = "You have two things this afternoon. A dentist appointment at three. Want a reminder?"


# --------------------------------------------------------------------------- #
# doubles
# --------------------------------------------------------------------------- #

def partial(text: str, *, final: bool | None = None) -> Partial:
    t = text.strip()
    return Partial(committed="", tail=t, stable_words=max(0, len(t.split()) - 2),
                   punct_final=t.endswith((".", "?", "!")) if final is None else final,
                   t_audio=2.0, cost_ms=300)


def pause(silence_s: float = 0.25, speech_s: float = 2.0) -> Pause:
    return Pause(audio=np.zeros(int(16000 * (speech_s + silence_s)), np.float32),
                 silence_s=silence_s, speech_s=speech_s)


def turn_end(seconds: float = 2.0, reason: str = "silence", silence_s: float = 1.4) -> TurnEnd:
    return TurnEnd(audio=np.zeros(int(16000 * seconds), np.float32), reason=reason, silence_s=silence_s)


class ProbeDetector:
    """The TurnDetector surface the probe flow uses: ``feed`` returns the next scripted
    event list (one per audio message), Smart Turn answers from ``p`` (one value per
    probe), and the session's decisions are recorded (``end_turn``, ``request_reprobe``,
    ``discard_utterance``)."""

    def __init__(self, script, *, p=(0.95,)):
        self.script = list(script)
        self.p = list(p)
        self.ep = SimpleNamespace(cfg=EndpointerConfig(), turn_fn=self._smart_turn)
        self.frame_sink = None
        self.active = False
        self.ended: list[tuple[str, float | None]] = []
        self.reprobes: list[float] = []
        self.discards = 0
        self.resets = 0
        self.speech_s = 2.0

    def _smart_turn(self, audio):
        return self.p.pop(0) if len(self.p) > 1 else self.p[0]

    def feed(self, pcm):
        events = self.script.pop(0) if self.script else []
        for ev in events:
            if isinstance(ev, SpeechStart):
                self.active = True
            elif isinstance(ev, (TurnEnd, Discarded)):
                self.active = False
        return events

    def end_turn(self, reason, prob=None):
        if not self.active:
            return None
        self.active = False
        self.ended.append((reason, prob))
        return TurnEnd(audio=np.zeros(int(16000 * self.speech_s), np.float32), reason=reason,
                       turn_prob=prob, silence_s=0.25)

    def request_reprobe(self, at_silence_s):
        self.reprobes.append(round(at_silence_s, 3))

    def discard_utterance(self):
        self.discards += 1
        self.active = False

    def reset(self):
        self.resets += 1
        self.active = False


class FakeStreamingSTT:
    """``at_pause`` answers the scripted partials in order, ``final`` the scripted final
    text (optionally after ``final_delay_s`` — a recogniser call in flight), ``tick``
    the scripted tick partials."""

    def __init__(self, partials=(), final: str | None = None, *, ticks=(), final_delay_s: float = 0.0):
        self.partials = list(partials)
        self.final_text = final
        self.ticks = list(ticks)
        self.final_delay_s = final_delay_s
        self.pushed = 0
        self.resets = 0
        self.finals = 0
        self.final_started = threading.Event()
        self._last = ""

    def push(self, pcm16, *, vad_prob):
        self.pushed += 1

    def at_pause(self):
        p = self.partials.pop(0) if self.partials else partial(self._last or "")
        self._last = p.text
        return p

    def tick(self):
        return self.ticks.pop(0) if self.ticks else None

    def final(self, *, timeout_s=None):
        self.finals += 1
        self.final_started.set()
        if self.final_delay_s:
            import time
            time.sleep(self.final_delay_s)
        return self.final_text if self.final_text is not None else self._last

    def reset(self):
        self.resets += 1
        self._last = ""


class Run:
    """A SpeculativeRun double (C-§B.1): ``stale`` is what DisPatch sets when it
    refuses adoption."""

    def __init__(self, text: str, head_id: str | None):
        self.text_in = text
        self.text = ""
        self.done = False
        self.head_id = head_id
        self.prompt_chars = 1500
        self.discarded = False
        self.stale = False
        self.adopted = False

    async def discard(self):
        self.discarded = True

    def usable(self):
        return not (self.discarded or self.adopted)


class Host:
    """A recording VoiceHost with speculation: every call lands in ``calls`` in order."""

    def __init__(self, *, abort_row: str | None = "a-partial", head: str | None = "h0",
                 refuse_adoption: bool = False, speculate: bool = True, submit_delay_s: float = 0.0):
        self.calls: list[tuple] = []
        self.abort_row = abort_row
        self.head = head
        self.refuse = refuse_adoption
        self.runs: list[Run] = []
        self.n = 0
        self.speculate_on = speculate
        self.submit_delay_s = submit_delay_s

    def build(self, **kw) -> VoiceHost:
        async def submit(tid, text, opts):
            self.n += 1
            run = (opts or {}).get("prefetched")
            if run is not None and self.refuse:
                run.stale = True
            elif run is not None:
                run.adopted = True
            self.calls.append(("submit", text, dict(opts or {})))
            if self.submit_delay_s:
                await asyncio.sleep(self.submit_delay_s)
            return f"u{self.n}"

        async def abort(tid, keep_partial=False):
            self.calls.append(("abort", keep_partial))
            return self.abort_row if keep_partial else None

        async def note_heard(tid, mid, *, heard_upto, heard_text, unheard_text):
            self.calls.append(("note_heard", mid, heard_upto, heard_text, unheard_text))
            return True

        async def retract(tid, mid):
            self.calls.append(("retract", mid))

        async def speculate(tid, text, opts):
            self.calls.append(("speculate", text))
            run = Run(text, self.head)
            self.runs.append(run)
            return run

        touches = kw.pop("touches", None)

        def touch():
            if touches is not None:
                touches.append(1)

        return VoiceHost(submit=submit, abort=abort, note_heard=note_heard, retract=retract,
                         speculate=speculate if self.speculate_on else None, touch=touch, **kw)

    def of(self, name: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == name]

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


class GatedTTS(FakeTTS):
    """A TTS whose sentences wait on ``gate`` (the engine still synthesising)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.gate = threading.Event()

    def stream(self, text, voice=None, stop=None):
        self.said.append(text)
        while not self.gate.wait(0.01):
            if stop is not None and stop.is_set():
                return
        for _ in range(self.chunks):
            if stop is not None and stop.is_set():
                return
            yield np.zeros(self.chunk_samples, "<i2")


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def make(script=(), *, p=(0.95,), sstt=None, host: Host | None = None, tts=None, backend="advisor",
         rt: dict | None = None, clock=None, acks=None, ws=None, **kw):
    ws = ws or FakeWS()
    host = host or Host()
    tts = tts or FakeTTS()
    det = ProbeDetector(script, p=p)
    rts = RtSettings.from_mapping(rt or {})
    sstt = sstt if sstt is not None else FakeStreamingSTT([partial(QUESTION)])
    extra = {"clock": clock} if clock is not None else {}
    s = VoiceSession(ws, thread_id=THREAD, settings=settings(merge_window_s=0.0, ack_after_s=0.6),
                     detector=det, stt_engine=FakeSTT("unused"), tts_engine=tts, host=host.build(),
                     notice=lambda text: np.ones(48, "<i2"), heard_wait_s=0.3, sstt=sstt,
                     policy=TurnPolicy(rts), rt=rts, bot_backend=backend,
                     acks=acks, **extra, **kw)
    s.watch_tick_s = 0.01            # the watchdog's timers run on the session clock
    return s, ws, tts, host, det


async def start(s, ws, **extra):
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start", "acks": "off", **extra})
    await settle(5)
    return task


def types(ws) -> list[str]:
    return [m["type"] for m in ws.sent if isinstance(m, dict)]


async def speak(ws, n: int = 1, wait: int = 20):
    for _ in range(n):
        ws.push_audio()
    await settle(wait)


async def play(s, ws, text=REPLY, mid="m1", prov="run:1", wait=40):
    """A reply arrives and is spoken (FakeTTS: instantly); `speaking` stays on
    until a `playback_done`."""
    for f in reply(mid, text, prov=prov):
        s.on_app_frame(f)
    await settle(wait)


# --------------------------------------------------------------------------- #
# the decision table at a probed pause (SPEC §4.3)
# --------------------------------------------------------------------------- #

async def test_confident_end_speculates_for_the_advisor_and_adopts_the_run():
    s, ws, _tts, host, det = make([[SpeechStart()], [pause()]], p=(0.95,))
    task = await start(s, ws)
    await speak(ws, 2, wait=40)
    assert det.ended == [("p_end", 0.95)]
    assert "eot_soft" not in types(ws) and ws.json_of("eot")
    # The speculative run started with the partial (abort/merge before it: none), then the
    # final text matched and the submit ADOPTED it (C-§B.1 rule 4 is DisPatch's).
    assert host.names()[:2] == ["speculate", "submit"]
    assert host.of("speculate")[0][1] == QUESTION
    submit = host.of("submit")[0]
    assert submit[1] == QUESTION and submit[2]["prefetched"] is host.runs[0]
    m = next(x for x in ws.json_of("metrics") if "eot_to_transcript_ms" in x)
    assert m["speculated"] is True and m["speculation_hit"] is True
    assert m["speculation"]["started"] == 1 and m["speculation"]["adopted"] == 1
    assert m["hold_ms"] == 0 and s._merge_until <= s.clock()     # no hold on a confident end
    # The reply is spoken at once (nothing holds it).
    await play(s, ws)
    assert [x["text"] for x in ws.json_of("say")][:1] == ["You have two things this afternoon."]
    ws.hang_up(); await task


async def test_unsure_row_sends_eot_soft_holds_the_reply_and_merges_speech_inside_the_hold():
    sstt = FakeStreamingSTT([partial("I was thinking that maybe we could."),
                             partial("go to the beach this weekend.")],
                            final=None)
    s, ws, _tts, host, _det = make([[SpeechStart()], [pause()], [SpeechStart()], [pause()]],
                                 p=(0.7, 0.95), sstt=sstt)
    task = await start(s, ws)
    await speak(ws, 2, wait=40)
    soft = ws.json_of("eot_soft")
    assert soft and soft[0]["p"] == 0.7 and soft[0]["hold_ms"] >= 300
    order = types(ws)
    assert order.index("eot_soft") < order.index("eot")       # eot_soft precedes eot (C.4)
    assert host.of("submit")[0][1] == "I was thinking that maybe we could."
    # The reply arrives inside the hold: synthesised, never said.
    s._merge_until = s.clock() + 5.0                         # a long learned hold, for the test
    await play(s, ws, "Sure, which day?", mid="m1", prov="run:1", wait=20)
    assert not ws.json_of("say")
    # The driver goes on inside the hold: ONE merged message, the first row taken back,
    # the first run aborted without a partial.
    ws.push_audio(); await settle(10)
    assert s._hold
    ws.push_audio(); await settle(40)
    submits = host.of("submit")
    assert len(submits) == 2
    assert submits[1][1] == "I was thinking that maybe we could go to the beach this weekend."
    assert ("retract", "u1") in host.calls and ("abort", False) in host.calls
    assert ws.json_of("transcript")[-1]["merged"] is True
    assert not [x for x in ws.json_of("say") if x["text"] == "Sure, which day?"]
    ws.hang_up(); await task


async def test_reprobe_row_asks_again_and_never_speculates_then_a_confident_probe_ends():
    sstt = FakeStreamingSTT([partial("Can you call the dentist."), partial("Can you call the dentist.")])
    s, ws, _tts, host, det = make([[SpeechStart()], [pause(0.25)], [pause(0.65)]], p=(0.3, 0.92), sstt=sstt)
    task = await start(s, ws)
    await speak(ws, 2, wait=30)
    assert det.reprobes == [0.65] and not ws.json_of("eot") and not host.of("speculate")
    await speak(ws, 1, wait=40)
    assert det.ended and det.ended[0][0] == "p_end"
    assert host.of("speculate") and host.of("submit")
    ws.hang_up(); await task


async def test_listen_row_waits_for_the_hard_silence_which_ends_without_speculation():
    sstt = FakeStreamingSTT([partial("So I was looking at the plan and")], final="So I was looking at the plan and")
    s, ws, _tts, host, det = make([[SpeechStart()], [pause()], [turn_end()]], p=(0.2,), sstt=sstt)
    task = await start(s, ws)
    await speak(ws, 2, wait=30)
    assert not ws.json_of("eot") and not det.ended
    await speak(ws, 1, wait=40)
    assert ws.json_of("eot")[0]["reason"] == "silence"
    assert not host.of("speculate")                        # no partial at the hard silence: no text
    assert host.of("submit")[0][1] == "So I was looking at the plan and"
    ws.hang_up(); await task


async def test_partial_frames_carry_no_message_id_and_follow_start_rt_partials():
    s, ws, _, _, _ = make([[SpeechStart()], [pause()]], p=(0.95,))
    task = await start(s, ws)
    await speak(ws, 2, wait=40)
    parts = ws.json_of("partial")
    assert parts and set(parts[0]) == {"type", "committed", "tail", "stable", "t"}
    assert parts[0]["tail"] == QUESTION and "message_id" not in parts[0]
    ws.hang_up(); await task
    s2, ws2, _, _, _ = make([[SpeechStart()], [pause()]], p=(0.95,))
    task2 = await start(s2, ws2, rt={"partials": False})
    await speak(ws2, 2, wait=40)
    assert not ws2.json_of("partial") and ws2.json_of("transcript")
    ws2.hang_up(); await task2


async def test_ticks_run_only_for_captions_or_over_a_reply():
    sstt = FakeStreamingSTT([], ticks=[partial("what is the")])
    s, ws, _, _, _ = make([[SpeechStart()], []], sstt=sstt)
    task = await start(s, ws, rt={"partials": False})
    await speak(ws, 2, wait=20)
    assert sstt.ticks                                       # captions off, nothing playing: no recogniser call
    ws.push_json({"type": "start", "rt": {"partials": True}}); await settle(5)
    await speak(ws, 1, wait=20)
    assert not sstt.ticks and ws.json_of("partial")[0]["tail"] == "what is the"
    ws.hang_up(); await task


async def test_learned_values_travel_in_ready_a_false_end_raises_them_and_reset_forgets():
    learned = []
    s, ws, _, host, det = make([[SpeechStart()], [pause()], [SpeechStart()], [pause()]], p=(0.95,),
                               sstt=FakeStreamingSTT([partial("Call the shop."), partial("and the bank.")]),
                               on_learned=learned.append)
    r0 = None
    task = asyncio.create_task(s.run())
    await settle(3)
    r0 = ws.json_of("ready")[0]["rt"]["learned"]
    # Nothing learned for this device on this server yet: null, so the client keeps its
    # own blob (review-T3s3 #6 — the defaults used to overwrite it after every restart).
    assert r0 is None
    ws.push_json({"type": "start", "acks": "off"}); await settle(5)
    await speak(ws, 2, wait=40)
    # Speech back 0.4 s after the eot (inside the 1 s false-end window, past the 0 s hold):
    # the next utterance merges and the policy learns a false end.
    s._eot_at = s.clock() - 0.4
    s._merge_until = 0.0
    ws.push_audio(); await settle(10)
    ws.push_audio(); await settle(40)
    assert host.of("submit")[-1][1] == "Call the shop and the bank."
    assert learned and learned[-1].probe_ms == 300 and s.policy.false_ends == 1
    assert det.ep.cfg.turn_probe_silence_s == pytest.approx(0.30)
    ws.push_json({"type": "start", "rt": {"learned_reset": True}}); await settle(5)
    assert s.policy.learned.as_dict() == {"probe_ms": 250, "hold_ms": 600}
    assert det.ep.cfg.turn_probe_silence_s == pytest.approx(0.25)
    ws.hang_up(); await task


async def test_the_device_blob_applies_only_when_the_server_has_no_learned_values():
    s, ws, _, _, det = make([])
    task = await start(s, ws, rt={"learned": {"probeMs": 400, "holdMs": 900}})
    assert s.policy.learned.as_dict() == {"probe_ms": 400, "hold_ms": 900}
    assert det.ep.cfg.turn_probe_silence_s == pytest.approx(0.40)
    ws.hang_up(); await task
    s2, ws2, _, _, _ = make([])
    s2.policy = TurnPolicy(s2.rt, Learned(probe_ms=300, hold_ms=500))
    s2._learned_from_server = True
    task2 = await start(s2, ws2, rt={"learned": {"probe_ms": 450, "hold_ms": 1000}})
    assert s2.policy.learned.as_dict() == {"probe_ms": 300, "hold_ms": 500}    # the server's wins
    assert ws2.json_of("ready")[0]["rt"]["learned"] == {"probe_ms": 300, "hold_ms": 500}
    ws2.hang_up(); await task2


# --------------------------------------------------------------------------- #
# speculation (SPEC §4.4)
# --------------------------------------------------------------------------- #

async def test_a_final_text_that_differs_discards_the_run_and_submits_normally():
    sstt = FakeStreamingSTT([partial("Remind me to call the dentist")],
                            final="Remind me to call the dentist tomorrow morning at nine please")
    s, ws, _, host, _ = make([[SpeechStart()], [pause()]], p=(0.95,), sstt=sstt)
    task = await start(s, ws)
    await speak(ws, 2, wait=40)
    assert host.runs[0].discarded
    assert "prefetched" not in host.of("submit")[0][2]
    m = next(x for x in ws.json_of("metrics") if "eot_to_transcript_ms" in x)
    assert m["speculated"] is True and m["speculation_hit"] is False
    assert m["speculation"]["discarded"] == 1 and m["speculation"]["adopted"] == 0
    ws.hang_up(); await task


async def test_a_thread_head_that_moved_under_the_speculation_makes_it_stale():
    sstt = FakeStreamingSTT([partial(QUESTION)], final=QUESTION, final_delay_s=0.15)
    s, ws, _, host, _ = make([[SpeechStart()], [pause()]], p=(0.95,), sstt=sstt, host=Host(head="h0"))
    task = await start(s, ws)
    ws.push_audio(); await settle(3)
    ws.push_audio()
    # A typed message lands while final() is still running (hardening M1).
    await asyncio.to_thread(sstt.final_started.wait, 2)
    s.on_app_frame(frame(type="message", message={"id": "typed-1", "role": "user", "content": "hi"}))
    await settle(60)
    assert host.runs and host.runs[0].discarded
    assert "prefetched" not in host.of("submit")[0][2]
    assert s.spec.stats["stale"] == 1 and s.spec.stats["adopted"] == 0
    ws.hang_up(); await task


async def test_dispatch_refusing_adoption_is_counted_stale_not_a_hit():
    s, ws, _, host, _ = make([[SpeechStart()], [pause()]], p=(0.95,), host=Host(refuse_adoption=True))
    task = await start(s, ws)
    await speak(ws, 2, wait=40)
    assert host.of("submit")[0][2].get("prefetched") is host.runs[0]
    m = next(x for x in ws.json_of("metrics") if "eot_to_transcript_ms" in x)
    assert m["speculation_hit"] is False
    assert m["speculation"]["stale"] == 1 and m["speculation"]["adopted"] == 0
    ws.hang_up(); await task


@pytest.mark.parametrize(("backend", "mode", "device", "expect"), [
    ("advisor", "advisor", True, True),
    ("api", "advisor", True, False),          # metered API bots are opt-in (§15 #2)
    ("api", "all", True, True),
    ("gateway", "all", True, False),          # tools and memory: never a shadow run
    ("advisor", "off", True, False),
    ("advisor", "advisor", False, False),     # start.rt.speculate: false
])
async def test_who_speculates(backend, mode, device, expect):
    s, ws, _, host, _ = make([[SpeechStart()], [pause()]], p=(0.95,), backend=backend, rt={"speculate": mode})
    task = await start(s, ws, rt={"speculate": device})
    await speak(ws, 2, wait=40)
    assert bool(host.of("speculate")) is expect
    assert ws.json_of("ready")[0]["rt"]["speculate"] is (backend in ("advisor", "api") and mode != "off"
                                                        and (mode == "all" or backend == "advisor"))
    assert host.of("submit")
    ws.hang_up(); await task


async def test_abort_runs_before_the_speculative_start_on_a_real_interruption():
    sstt = FakeStreamingSTT([partial("Actually tell me about tomorrow instead.")])
    s, ws, _tts, host, _det = make([[SpeechStart()], [pause()]], p=(0.95,), sstt=sstt)
    task = await start(s, ws)
    s._current = vs._Turn(msg_id="u0", text="before", at=s.clock())
    s._owner = s._current
    await play(s, ws)
    assert s.speaking
    await speak(ws, 2, wait=60)
    names = host.names()
    assert ("abort", True) in host.calls
    assert names.index("abort") < names.index("speculate") < names.index("submit")
    assert ws.json_of("barge_in")
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# the "oh, and…" addendum (SPEC §4.3, experience H1)
# --------------------------------------------------------------------------- #

def addendum_session(*, tts=None, clock=None):
    sstt = FakeStreamingSTT([partial(QUESTION), partial("Oh and tomorrow.")])
    return make([[SpeechStart()], [pause()], [SpeechStart()], [pause()]], p=(0.95, 0.95),
                sstt=sstt, tts=tts, clock=clock)


async def test_addendum_within_the_window_while_nothing_was_said_merges_into_one_row():
    clock = Clock()
    s, ws, _tts, host, _det = addendum_session(clock=clock)
    task = await start(s, ws)
    await speak(ws, 2, wait=40)
    clock.t += 2.0                                        # "…this afternoon?" — 2 s — "oh, and tomorrow"
    ws.push_audio(); await settle(10)
    assert s._hold
    ws.push_audio(); await settle(40)
    submits = host.of("submit")
    assert [x[1] for x in submits] == [QUESTION, "What's on my calendar this afternoon Oh and tomorrow."]
    assert ("retract", "u1") in host.calls and ("abort", False) in host.calls
    assert ws.json_of("transcript")[-1]["merged"] is True
    # The first reply landing late is never spoken; the merged turn's reply is.
    s.on_app_frame(frame(type="message", message={"id": "u2", "role": "user", "content": "x"}))
    await play(s, ws, "Only the dentist.", mid="m2", prov="run:2")
    assert [x["text"] for x in ws.json_of("say")] == ["Only the dentist."]
    ws.hang_up(); await task


async def test_addendum_merges_even_while_the_first_reply_is_still_being_synthesised():
    clock = Clock()
    tts = GatedTTS()
    s, ws, tts, host, _det = addendum_session(tts=tts, clock=clock)
    task = await start(s, ws)
    await speak(ws, 2, wait=40)
    await play(s, ws, "You have two things.", mid="m1", prov="run:1", wait=20)
    # Synthesising: the sentence's `say` may be out, but no audio — nothing is audible yet.
    assert s._tts_busy and not ws.json_of("audio") and not s._current.said
    clock.t += 1.5
    ws.push_audio(); await settle(10)
    assert s._hold
    ws.push_audio(); await settle(40)
    # Not an interruption (nothing was heard): one merged row, nothing persisted for the first.
    assert not host.of("note_heard") and ("abort", True) not in host.calls
    assert host.of("submit")[-1][1] == "What's on my calendar this afternoon Oh and tomorrow."
    assert ws.json_of("barge_in")                          # the client drops the first reply's say
    tts.gate.set(); await settle(20)
    assert not ws.json_of("audio")
    ws.hang_up(); await task


async def test_after_the_window_new_words_replace_the_pending_answer():
    clock = Clock()
    s, ws, _tts, host, _det = addendum_session(clock=clock)
    task = await start(s, ws)
    await speak(ws, 2, wait=40)
    clock.t += 3.5                                        # past rt.addendum_s × patience (3 s)
    ws.push_audio(); await settle(10)
    assert not s._hold
    ws.push_audio(); await settle(40)
    assert [x[1] for x in host.of("submit")] == [QUESTION, "Oh and tomorrow."]
    assert ("abort", False) in host.calls and not host.of("retract")
    assert ws.json_of("transcript")[-1]["merged"] is False
    ws.hang_up(); await task


async def test_once_a_sentence_was_said_more_speech_is_an_interruption_not_an_addendum():
    clock = Clock()
    s, ws, _tts, host, _det = addendum_session(clock=clock)
    task = await start(s, ws)
    await speak(ws, 2, wait=40)
    s.on_app_frame(frame(type="message", message={"id": "u1", "role": "user", "content": QUESTION}))
    await play(s, ws)
    assert s.speaking and s._current.said
    clock.t += 1.0
    ws.push_audio(); await settle(10)
    assert not s._hold and ws.json_of("duck")[-1]["on"] is True
    ws.push_audio(); await settle(60)
    assert ws.json_of("barge_in") and ("abort", True) in host.calls
    assert host.of("submit")[-1][1] == "Oh and tomorrow." and ws.json_of("transcript")[-1]["merged"] is False
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# word-gated barge-in (SPEC §4.6)
# --------------------------------------------------------------------------- #

async def over_reply(script, *, words, p=(0.95,), clock=None, mode="audio", sstt=None, **kw):
    if sstt is None:
        sstt = FakeStreamingSTT([partial(w) for w in (words if isinstance(words, list) else [words])])
    s, ws, tts, host, det = make(script, p=p, sstt=sstt, clock=clock, **kw)
    task = await start(s, ws, **({"slow_link": "always_phone"} if mode == "text" else {}))
    s._current = vs._Turn(msg_id="u0", text="before", at=s.clock())
    s._owner = s._current
    await play(s, ws)
    assert s.speaking
    return s, ws, tts, host, det, task


async def test_echo_resumes_the_reply_in_the_same_gen_and_rewinds_after_a_long_pause():
    clock = Clock()
    s, ws, _tts, host, det, task = await over_reply([[SpeechStart()], [pause()]],
                                                   words="A dentist appointment at three.", clock=clock)
    gen = s.gen
    ws.push_audio(); await settle(10)
    assert ws.json_of("duck")[-1] == {"type": "duck", "on": True, "gen": gen}
    clock.t += 0.9                                        # the pause lasted > 0.8 s
    ws.push_audio(); await settle(30)
    assert ws.json_of("echo_ignored") and not ws.json_of("barge_in")
    assert ws.json_of("resume") == [{"type": "resume", "gen": gen, "rewind": True}]
    assert s.gen == gen and not host.of("submit") and det.discards == 1
    ws.hang_up(); await task


async def test_a_short_echo_resumes_in_place():
    clock = Clock()
    _s, ws, _tts, _host, _det, task = await over_reply([[SpeechStart()], [pause()]],
                                                   words="A dentist appointment at three.", clock=clock)
    ws.push_audio(); await settle(10)
    clock.t += 0.3
    ws.push_audio(); await settle(30)
    assert ws.json_of("resume")[-1]["rewind"] is False
    ws.hang_up(); await task


async def test_a_backchannel_resumes_and_never_bumps_gen_two_in_three_seconds_stop():
    clock = Clock()
    script = [[SpeechStart()], [Pause(np.zeros(8000, np.float32), 0.25, 0.5)],
              [SpeechStart()], [Pause(np.zeros(8000, np.float32), 0.25, 0.5)]]
    s, ws, _tts, host, _det, task = await over_reply(script, words=["Uh-huh.", "Okay."], clock=clock)
    gen = s.gen
    await speak(ws, 2, wait=30)
    assert ws.json_of("resume") and s.gen == gen and not ws.json_of("barge_in")
    clock.t += 1.5                                        # a second backchannel within 3 s = "you can stop"
    await speak(ws, 2, wait=40)
    assert ws.json_of("barge_in") and ws.json_of("barge_in")[0]["gen"] == gen + 1
    assert not host.of("submit") and ("abort", True) not in host.calls
    ws.hang_up(); await task


async def test_a_stop_word_cuts_the_reply_and_nothing_becomes_a_turn():
    s, ws, _tts, host, _det, task = await over_reply([[SpeechStart()], [pause(speech_s=0.6)]], words="Stop.")
    gen = s.gen
    await speak(ws, 2, wait=40)
    assert ws.json_of("barge_in") == [{"type": "barge_in", "gen": gen + 1}]
    assert not host.of("submit") and ("abort", True) not in host.calls    # a stop never aborts the run
    assert not s.speaking and ws.json_of("state")[-1]["state"] == "listening"
    ws.hang_up(); await task


async def test_a_real_interruption_cuts_notes_and_becomes_the_next_turn():
    s, ws, _tts, host, _det, task = await over_reply([[SpeechStart()], [pause()]],
                                                   words="Wait, which dentist was that?")
    ws.push_json({"type": "heard", "gen": s.gen, "seq": 2, "played_ms": 400, "source": "server"})
    await settle(3)
    await speak(ws, 2, wait=60)
    assert await wait_until(lambda: "submit" in host.names())      # the chain: abort, note_heard, submit
    assert ws.json_of("barge_in")
    assert host.names().index("abort") < host.names().index("note_heard") < host.names().index("submit")
    submit = host.of("submit")[0]
    assert submit[1] == "Wait, which dentist was that?" and submit[2]["interrupted"]["unheard"]
    m = next(x for x in ws.json_of("metrics") if "eot_to_transcript_ms" in x)
    assert isinstance(m["interrupt_to_submit_ms"], int) and m["interrupt_to_submit_ms"] >= 0   # review-T3s2 #9
    ws.hang_up(); await task


async def test_a_duck_with_no_verdict_resumes_after_the_false_interruption_timeout():
    clock = Clock()
    s, ws, _tts, _host, _det, task = await over_reply([[SpeechStart()]], words="x", clock=clock)
    s.watch_tick_s = 0.01
    ws.push_audio(); await settle(10)
    assert ws.json_of("duck")[-1]["on"] is True
    clock.t += 1.5; await settle(10)
    assert not ws.json_of("resume")
    clock.t += 0.6; await settle(10)                      # > rt.false_interrupt_timeout_s (2.0 s)
    assert ws.json_of("resume") == [{"type": "resume", "gen": s.gen, "rewind": True}]
    assert ws.json_of("duck")[-1]["on"] is False
    ws.hang_up(); await task


async def test_an_echo_ended_by_the_hard_silence_is_never_a_turn():
    sstt = FakeStreamingSTT([], final="a dentist appointment at three")
    s, ws, _tts, host, _det = make([[SpeechStart()], [turn_end()]], sstt=sstt)
    task = await start(s, ws)
    s._current = vs._Turn(msg_id="u0", text="before", at=s.clock())
    s._owner = s._current
    await play(s, ws)
    await speak(ws, 2, wait=40)
    assert ws.json_of("echo_ignored") and not host.of("submit") and not ws.json_of("barge_in")
    ws.hang_up(); await task


async def test_words_over_a_reply_nobody_heard_yet_replace_it_without_a_partial_row():
    tts = GatedTTS()
    clock = Clock()
    sstt = FakeStreamingSTT([partial("No, I meant the other calendar.")])
    s, ws, tts, host, _det = make([[SpeechStart()], [pause()]], p=(0.95,), sstt=sstt, tts=tts, clock=clock)
    task = await start(s, ws)
    s._current = vs._Turn(msg_id="u0", text="before", at=clock.t - 10, echoed=True)
    s._last_submit = ("before", "u0")
    await play(s, ws, "You have two things.", wait=20)
    assert s._tts_busy and not ws.json_of("audio")
    await speak(ws, 2, wait=40)
    assert ("abort", False) in host.calls and ("abort", True) not in host.calls
    assert not host.of("note_heard") and host.of("submit")[-1][1] == "No, I meant the other calendar."
    tts.gate.set(); await settle(20)
    assert not ws.json_of("audio")                         # the cut sentence is never heard
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# the phone's voice (text mode): VAD alone never cancels, nothing re-spoken (H4)
# --------------------------------------------------------------------------- #

async def test_phone_voice_duck_is_informational_holds_nothing_and_never_rewinds():
    clock = Clock()
    s, ws, tts, _host, det, task = await over_reply([[SpeechStart()], [pause()]], mode="text",
                                                   words="You have two things this afternoon.", clock=clock)
    assert tts.said == []                                 # the phone reads the `say` frames
    assert det.ep.cfg.min_speech_s == pytest.approx(vs.PHONE_VOICE_BARGE_IN_S)
    assert det.ep.cfg.threshold == pytest.approx(vs.SPEAKING_VAD_THRESHOLD)
    ws.push_audio(); await settle(10)
    assert ws.json_of("duck")[-1]["on"] is True and s._duck_informational and not s._held()
    n_say = len(ws.json_of("say"))
    await play(s, ws, "And one more thing.", mid="m2", prov="run:2", wait=20)
    assert len(ws.json_of("say")) == n_say + 1            # the duck held nothing back
    clock.t += 1.2
    ws.push_audio(); await settle(30)
    assert ws.json_of("echo_ignored") and not ws.json_of("barge_in")
    assert ws.json_of("resume")[-1]["rewind"] is False    # the phone never re-speaks
    ws.hang_up(); await task


async def test_phone_voice_noise_or_a_hard_silence_end_without_words_never_cancels():
    sstt = FakeStreamingSTT([partial("")], final="")
    _s, ws, _tts, host, _det, task = await over_reply([[SpeechStart()], [Discarded(0.1)], [SpeechStart()],
                                                    [pause()], [], [turn_end()]],
                                                   mode="text", words="", sstt=sstt)
    await speak(ws, 6, wait=40)
    assert not ws.json_of("barge_in") and not host.of("submit") and ("abort", True) not in host.calls
    ws.hang_up(); await task


async def test_phone_voice_confirmed_stop_cancels_and_nothing_is_respoken():
    s, ws, _tts, _host, _det, task = await over_reply([[SpeechStart()], [pause(speech_s=0.5)]],
                                                   mode="text", words="Stop.")
    said_before = len(ws.json_of("say"))
    await speak(ws, 2, wait=40)
    assert ws.json_of("barge_in") and not ws.json_of("resume")
    for f in reply("m1", REPLY + " More words after the cut.", prov="run:1"):
        s.on_app_frame(f)
    await settle(30)
    assert len(ws.json_of("say")) == said_before
    ws.hang_up(); await task


async def test_server_audio_barge_in_needs_a_quarter_second_and_the_phone_voice_more():
    s, ws, _tts, _host, det, task = await over_reply([], words="x")
    assert det.ep.cfg.min_speech_s == pytest.approx(vs.SERVER_AUDIO_BARGE_IN_S)
    assert s.settings.barge_in_speech_s == pytest.approx(0.25)
    ws.push_json({"type": "playback_done", "gen": s.gen}); await settle(10)
    assert not s.speaking and det.ep.cfg.min_speech_s == pytest.approx(0.25)
    assert det.ep.cfg.threshold == pytest.approx(0.5)
    ws.hang_up(); await task


async def test_the_local_speaking_flag_means_the_phone_voice_threshold():
    from session.test_session_v3_wire import make as make_v3
    from session.test_session_v3_wire import up_pcm
    s, ws, _, _, _ = make_v3()
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); await settle(5)
    ws.push_audio(up_pcm(0, 320, flags=wire3.FLAG_LOCAL_SPEAKING)); await settle(5)
    assert s.detector.ep.cfg.min_speech_s == pytest.approx(vs.PHONE_VOICE_BARGE_IN_S)
    ws.push_audio(up_pcm(320, 320)); await settle(5)
    assert not s._local_speaking
    assert s.detector.ep.cfg.min_speech_s == pytest.approx(s._base[0])     # nothing of ours playing
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# backchannels (SPEC §4.7)
# --------------------------------------------------------------------------- #

def bc_session(*, bot_backchannel=True, clock=None):
    kit = kit_with_lines({"statement": ("Mm-hm.",)}, samples=200)
    kit.queue.lines["Mm-hm."] = np.full(200, 1000, "<i2")
    sstt = FakeStreamingSTT([partial("So I was looking at the plan and")])
    script = [[SpeechStart()], [Pause(np.zeros(16000 * 5, np.float32), 0.6, 4.5)]]
    return make(script, p=(0.2,), sstt=sstt, acks=kit, bot_backchannel=bot_backchannel, clock=clock)


async def test_a_listen_row_plays_one_quiet_backchannel_without_bumping_gen():
    clock = Clock()
    s, ws, _tts, host, _det = bc_session(clock=clock)
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start", "acks": "often", "rt": {"backchannels": True}}); await settle(5)
    assert ws.json_of("ready")[0]["rt"]["backchannels"] is True
    ws.push_audio(); await settle(5)
    clock.t += 4.5
    ws.push_audio(); await settle(30)
    bc = [m for m in ws.json_of("say") if m["kind"] == "backchannel"]
    assert [m["text"] for m in bc] == ["Mm-hm."] and s.gen == 0 and not host.of("submit")
    hdr = next(m for m in ws.json_of("audio") if m["kind"] == "backchannel")
    pcm = ws.sent[ws.sent.index(hdr) + 1]
    assert np.frombuffer(pcm, "<i2").max() == 500        # the rendered line at −6 dB
    assert "Mm-hm." in s._recent_said
    ws.hang_up(); await task


@pytest.mark.parametrize("why", ["bot_off", "device_off", "barge_in_off"])
async def test_no_backchannel_unless_the_bot_the_device_and_barge_in_all_say_so(why):
    clock = Clock()
    s, ws, _tts, _host, _det = bc_session(bot_backchannel=(why != "bot_off"), clock=clock)
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start", "acks": "often", "barge_in": why != "barge_in_off",
                  "rt": {"backchannels": why != "device_off"}}); await settle(5)
    ws.push_audio(); await settle(5)
    clock.t += 4.5
    ws.push_audio(); await settle(30)
    assert not [m for m in ws.json_of("say") if m["kind"] == "backchannel"]
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# quiet_check, idle touch (C.3, SPEC §5.6)
# --------------------------------------------------------------------------- #

async def test_quiet_check_speaks_still_there_as_a_notice_once_a_minute():
    clock = Clock()
    s, ws, _tts, _host, _det = make([], clock=clock)
    task = await start(s, ws)
    ws.push_json({"type": "quiet_check"}); await settle(10)
    notes = [m for m in ws.json_of("say") if m["kind"] == "notice"]
    assert [m["text"] for m in notes] == ["Still there?"] and not s.speaking
    ws.push_json({"type": "quiet_check"}); await settle(10)
    assert len([m for m in ws.json_of("say") if m["kind"] == "notice"]) == 1
    clock.t += 61
    ws.push_json({"type": "quiet_check"}); await settle(10)
    assert len([m for m in ws.json_of("say") if m["kind"] == "notice"]) == 2
    ws.hang_up(); await task


async def test_quiet_check_in_phone_voice_mode_is_a_say_alone():
    s, ws, tts, _host, _det = make([])
    task = await start(s, ws, slow_link="always_phone")
    ws.push_json({"type": "quiet_check"}); await settle(10)
    assert [m["text"] for m in ws.json_of("say")] == ["Still there?"]
    assert not ws.json_of("audio") and tts.said == []
    ws.hang_up(); await task


async def test_a_gated_listener_stays_unlocked_but_a_silent_socket_extends_nothing():
    clock = Clock()
    touches: list = []
    host = Host()
    ws = FakeWS()
    det = ProbeDetector([])
    s = VoiceSession(ws, thread_id=THREAD, settings=settings(), detector=det, stt_engine=FakeSTT("x"),
                     tts_engine=FakeTTS(), host=host.build(touches=touches), clock=clock)
    s.watch_tick_s = 0.005
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); await settle(5)
    n0 = len(touches)
    # 11 minutes of a gated mic (no audio at all), pinging every 2 s: touched every 30 s.
    for _ in range(330):
        clock.t += 2.0
        ws.push_json({"type": "ping", "t": 1}); await real_pause(0.002)
    await settle(5)
    assert len(touches) - n0 >= 20 and not ws.json_of("locked")
    # The client goes silent (half-open): within 10 s nothing is touched any more.
    n1 = len(touches)
    for _ in range(4):
        clock.t += 5.0; await settle(3)
    assert len(touches) - n1 <= 1
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# review-T3s2 fixes
# --------------------------------------------------------------------------- #

async def test_hang_up_during_the_heard_wait_ends_the_session_and_submits_nothing():
    """#1: a CancelledError inside the ≤ 1.5 s wait used to be swallowed: run() never
    returned (power hold, subscription, slot leaked) and the cut words were submitted
    after the socket was gone."""
    s, ws, _tts, host, _det, task = await over_reply([[SpeechStart()], [pause()]],
                                                   words="Wait, which dentist was that?")
    s.heard_wait_s = 1.5
    ws.push_json({"type": "heard", "gen": s.gen, "seq": 1, "played_ms": 100})   # a heard-capable client
    await settle(3)
    await speak(ws, 2, wait=10)
    assert s._heard_fut is not None and not s._heard_fut.done()         # waiting for the exact frame
    ws.hang_up()
    await asyncio.wait_for(task, 1.0)                                    # run() returns at once
    assert not host.of("submit")


async def test_a_stale_heard_of_the_previous_reply_never_annotates_the_next():
    """#2: reply A's periodic `heard`, then reply B cut before any `heard` for B: the
    note must not carry A's words or offsets."""
    s, ws, _tts, host, _det = make([[SpeechStart()], [pause()]], p=(0.95,),
                                 sstt=FakeStreamingSTT([partial("Wait, what about Friday?")]))
    task = await start(s, ws)
    s._current = vs._Turn(msg_id="u0", text="before", at=s.clock())
    s._owner = s._current
    await play(s, ws, "Alpha one. Alpha two.", mid="a1", prov="run:a")
    ws.push_json({"type": "heard", "gen": s.gen, "seq": 1, "played_ms": 300}); await settle(3)
    ws.push_json({"type": "playback_done", "gen": s.gen, "seq": 2}); await settle(5)
    await play(s, ws, "Bravo one. Bravo two.", mid="b1", prov="run:b")
    assert s.speaking
    await speak(ws, 2, wait=60)                                          # cut B: no exact heard comes
    assert await wait_until(lambda: host.of("note_heard"))
    notes = host.of("note_heard")
    assert notes and all("Alpha" not in n[3] and "Alpha" not in n[4] for n in notes)
    assert "Bravo" in notes[-1][4]
    ws.hang_up(); await task


async def test_heard_text_spans_the_provisional_to_row_id_switch():
    """#3: sentences 1-2 streamed under `run:1`, sentence 3 flushed at stream_done under
    the row id — a cut in sentence 3 must still report sentences 1-2 as heard."""
    s, ws, _tts, host, _det = make([[SpeechStart()], [pause()]], p=(0.95,), tts=FakeTTS(chunk_samples=24000),
                                 sstt=FakeStreamingSTT([partial("Hold on, which one?")]))
    task = await start(s, ws)
    s._current = vs._Turn(msg_id="u0", text="before", at=s.clock())
    s._owner = s._current
    await play(s, ws)
    says = ws.json_of("say")
    assert [m["message_id"] for m in says] == ["run:1", "run:1", "m1"]
    ws.push_json({"type": "heard", "gen": s.gen, "seq": 3, "played_ms": 200}); await settle(3)
    await speak(ws, 2, wait=60)
    assert await wait_until(lambda: host.of("note_heard"))
    note = host.of("note_heard")[0]
    assert note[1] == "a-partial"
    assert note[3].startswith("You have two things this afternoon. A dentist appointment at three.")
    assert "Want a reminder" in note[4]
    ws.hang_up(); await task


async def test_the_exact_heard_after_the_fallback_renotes_the_row():
    """#4: a client whose FIRST `heard` is the post-barge_in one: the fallback note goes
    out at once, the exact frame lands a moment later and re-notes the row."""
    s, ws, _tts, host, _det = make([[SpeechStart()], [pause()]], p=(0.95,), tts=FakeTTS(chunk_samples=24000),
                                 sstt=FakeStreamingSTT([partial("Hold on, which one?")]))
    task = await start(s, ws)
    s._current = vs._Turn(msg_id="u0", text="before", at=s.clock())
    s._owner = s._current
    await play(s, ws)
    await speak(ws, 2, wait=40)
    notes = host.of("note_heard")
    assert len(notes) == 1                    # the fallback: the last piece sent is the cut
    assert notes[0][4] == "Want a reminder?"
    # The exact answer to the barge_in (the NEW gen): the listener was 0.6 s into sentence 2.
    ws.push_json({"type": "heard", "gen": s.gen, "seq": 2, "played_ms": 600}); await settle(20)
    notes = host.of("note_heard")
    assert len(notes) == 2 and notes[1][1] == "a-partial"
    assert notes[1][2] < notes[0][2]
    assert notes[1][3].startswith("You have two things this afternoon. A")
    assert notes[1][4].endswith("Want a reminder?") and notes[1][4] != "Want a reminder?"
    # ... and the next turn's instructions carry the exact note.
    assert s._pending_note is None or s._pending_note["unheard"] == notes[1][4]
    ws.hang_up(); await task


class _AlwaysSkip(_NoSkip):
    def random(self):
        return 0.0


async def test_the_earcon_stands_in_when_no_line_fits_but_never_for_a_skip():
    """#5: one rendered line in ANOTHER pool used to silence the gate; and a Reactions
    roll that skipped used to beep."""
    kit = kit_with_lines({"statement": ("Okay.",)})                     # rendered, but not a question line
    kit.selector.pools.default["question"] = ("Let me see.",)          # a question line, never rendered
    s, ws, _tts, _host, _det = make([[SpeechStart()], [pause()]], p=(0.95,), acks=kit)
    task = await start(s, ws, acks="often")
    await speak(ws, 2, wait=60)
    await asyncio.sleep(0.7); await settle(5)
    assert ws.json_of("ack") and not [m for m in ws.json_of("say") if m["kind"] == "ack"]
    ws.hang_up(); await task
    kit2 = kit_with_lines({"question": ("Let me see.",)})
    kit2.selector.rng = _AlwaysSkip()
    s2, ws2, _, _, _ = make([[SpeechStart()], [pause()]], p=(0.95,), acks=kit2)
    task2 = await start(s2, ws2, acks="sometimes")
    await speak(ws2, 2, wait=60)
    await asyncio.sleep(0.7); await settle(5)
    assert not ws2.json_of("ack") and not ws2.json_of("say")
    assert s2._turn_acks.ack_outcome == "skipped"
    ws2.hang_up(); await task2


async def test_text_mode_ack_is_never_recent_and_a_notice_is_recorded_once():
    """#6: the earcon is not speech; a notice line goes into `_recent_said` once."""
    kit = kit_with_lines({"question": ("Let me think.",)})
    s, ws, _tts, _host, _det = make([[SpeechStart()], [pause()]], p=(0.95,), acks=kit)
    task = await start(s, ws, acks="often", slow_link="always_phone")
    await speak(ws, 2, wait=60)
    await asyncio.sleep(0.7); await settle(5)
    assert ws.json_of("ack") and "Let me think." not in s._recent_said
    ws.hang_up(); await task
    s2, ws2, _, _, _ = make([])
    task2 = await start(s2, ws2, slow_link="always_phone")
    ws2.push_json({"type": "quiet_check"}); await settle(10)
    assert [m["text"] for m in ws2.json_of("say")] == ["Still there?"]
    assert list(s2._recent_said).count("Still there?") == 1
    ws2.hang_up(); await task2


async def test_end_all_sessions_counts_sessions_not_chats_being_built(monkeypatch):
    from dispatch_voice.host import ws as ws_mod
    ended = []
    sess = SimpleNamespace(end=lambda code, reason: ended.append(code))
    res = ws_mod._Reservation()
    monkeypatch.setitem(ws_mod._SESSIONS, 1, sess)
    monkeypatch.setitem(ws_mod._SESSIONS, 2, res)
    assert ws_mod.open_count() == 1
    assert await ws_mod.end_all_sessions(1013, "off") == 1                 # #7
    assert ended == [1013] and res.ended == (1013, "off")


# --------------------------------------------------------------------------- #
# turn ends off the audio worker; speech during a turn end (the false end)
# --------------------------------------------------------------------------- #

async def test_the_mic_keeps_flowing_while_a_hard_silence_turn_end_is_being_sent():
    sstt = FakeStreamingSTT([], final=QUESTION, final_delay_s=0.3)
    s, ws, _tts, host, det = make([[SpeechStart()], [turn_end()], [], [], []], sstt=sstt)
    task = await start(s, ws)
    await speak(ws, 2, wait=3)
    await asyncio.to_thread(sstt.final_started.wait, 2)
    left = len(det.script)
    ws.push_audio(); await settle(3)
    ws.push_audio(); await settle(3)
    assert len(det.script) < left and not host.of("submit")    # fed while final() still runs
    await settle(60)
    assert host.of("submit")[0][1] == QUESTION
    ws.hang_up(); await task


async def test_speech_that_starts_while_the_turn_is_being_sent_joins_that_turn():
    learned = []
    clock = Clock()
    sstt = FakeStreamingSTT([partial("Call the shop."), partial("and the bank.")], final_delay_s=0.25)
    s, ws, _tts, host, _det = make([[SpeechStart()], [pause()], [SpeechStart()], [pause()]], p=(0.95, 0.95),
                                 sstt=sstt, on_learned=learned.append, clock=clock)
    task = await start(s, ws)
    await speak(ws, 2, wait=3)
    await asyncio.to_thread(sstt.final_started.wait, 2)
    clock.t += 0.4                                              # 0.4 s after the eot ...
    ws.push_audio(); await settle(3)                            # ... the driver goes on, during final()
    assert s._hold and s._ending
    await settle(80)
    assert host.of("submit")[0][1] == "Call the shop." and s._hold      # the hold survives the submit
    sstt.final_delay_s = 0.0
    ws.push_audio(); await settle(60)
    assert [x[1] for x in host.of("submit")] == ["Call the shop.", "Call the shop and the bank."]
    assert ("retract", "u1") in host.calls and ws.json_of("transcript")[-1]["merged"] is True
    assert s.policy.false_ends == 1 and learned                         # learned: a false end
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# control frames never block the receive loop on the heard wait
# --------------------------------------------------------------------------- #

async def test_a_tap_stop_uses_the_exact_heard_the_receive_loop_reads_after_barge_in():
    """stop_playback arrives on the receive loop, which is also what reads the `heard`
    answering the `barge_in`: the wait must not run inline (it would always time out)."""
    s, ws, _tts, host, _det, task = await over_reply([], words="x", tts=FakeTTS(chunk_samples=24000))
    s.heard_wait_s = 5.0                                   # a blocked loop would show as a 5 s stall
    ws.push_json({"type": "heard", "gen": 0, "seq": 1, "played_ms": 100}); await settle(3)
    t0 = asyncio.get_running_loop().time()
    ws.push_json({"type": "stop_playback", "heard": {"gen": 0, "seq": 1, "played_ms": 300}})
    await settle(5)
    assert ws.json_of("barge_in") == [{"type": "barge_in", "gen": 1}]
    ws.push_json({"type": "heard", "gen": 1, "seq": 2, "played_ms": 500})   # the exact answer
    ws.push_json({"type": "ping", "t": 7})
    await settle(10)
    assert ws.json_of("pong") and asyncio.get_running_loop().time() - t0 < 1.0
    notes = host.of("note_heard")
    assert len(notes) == 1 and notes[0][1] == "m1"          # the tap never aborts: the row's own id
    assert notes[0][3].startswith("You have two things this afternoon. A")   # inside sentence 2
    assert ("abort", True) not in host.calls and not host.of("submit")
    ws.hang_up(); await task


async def test_cancel_while_a_reply_plays_keeps_the_partial_and_reads_frames_meanwhile():
    s, ws, _tts, host, _det, task = await over_reply([], words="x", tts=FakeTTS(chunk_samples=24000))
    s.heard_wait_s = 5.0
    ws.push_json({"type": "heard", "gen": 0, "seq": 1, "played_ms": 100}); await settle(3)
    ws.push_json({"type": "cancel"}); await settle(5)
    assert ws.json_of("barge_in") and ("abort", True) in host.calls
    ws.push_json({"type": "heard", "gen": 1, "seq": 1, "played_ms": 900}); await settle(10)
    notes = host.of("note_heard")
    assert len(notes) == 1 and notes[0][1] == "a-partial"
    assert ws.json_of("state")[-1]["state"] == "listening"
    ws.hang_up(); await task


async def test_undo_answers_at_once_and_retracts_after_the_abort():
    host = Host(submit_delay_s=0.0)
    s, ws, _tts, host, _det = make([[SpeechStart()], [pause()]], p=(0.95,), host=host)
    task = await start(s, ws)
    await speak(ws, 2, wait=40)
    ws.push_json({"type": "undo"}); await settle(10)
    assert ws.json_of("undone") == [{"type": "undone", "message_id": "u1"}]
    names = host.names()
    assert names.index("abort") < names.index("retract")
    ws.hang_up(); await task


async def test_hang_up_during_the_heard_wait_on_the_whole_utterance_path_never_leaves_a_zombie():
    """#1 on the 2.1 whole-utterance path (no streaming STT): the interruption runs
    inside the audio worker; a swallowed CancelledError left it alive forever, so
    run() never returned (the review's probe_zombie)."""
    from fakes import ScriptedDetector, utterance
    host = Host()
    ws = FakeWS()
    s = VoiceSession(ws, thread_id=THREAD, settings=settings(merge_window_s=0.0),
                     detector=ScriptedDetector([[SpeechStart()], [utterance()]]),
                     stt_engine=FakeSTT("wait, what about tomorrow instead"), tts_engine=FakeTTS(),
                     host=host.build(), notice=lambda text: np.ones(48, "<i2"), heard_wait_s=1.5)
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start", "acks": "off"}); await settle(5)
    s._current = vs._Turn(msg_id="u0", text="before", at=s.clock())
    s._owner = s._current
    await play(s, ws)
    ws.push_json({"type": "heard", "gen": 0, "seq": 1, "played_ms": 10}); await settle(3)
    ws.push_audio(); await settle(5)
    ws.push_audio(); await settle(10)
    assert s._heard_fut is not None and not s._heard_fut.done()
    ws.hang_up()
    await asyncio.wait_for(task, 1.0)
    await asyncio.sleep(0.05)
    assert not host.of("submit")


async def test_hang_up_during_the_heard_wait_of_a_hard_silence_end_submits_nothing():
    sstt = FakeStreamingSTT([], final="wait, what about tomorrow instead")
    s, ws, _tts, host, _det, task = await over_reply([[SpeechStart()], [turn_end()]], words="x", sstt=sstt)
    s.heard_wait_s = 1.5
    ws.push_json({"type": "heard", "gen": 0, "seq": 1, "played_ms": 10}); await settle(3)
    await speak(ws, 2, wait=15)
    assert s._heard_fut is not None and not s._heard_fut.done()
    ws.hang_up()
    await asyncio.wait_for(task, 1.0)
    await asyncio.sleep(0.05)
    assert not host.of("submit")


async def test_backchannels_are_capped_per_utterance_and_the_cap_resets_with_the_next():
    clock = Clock()
    kit = kit_with_lines({"statement": ("Mm-hm.", "Right.", "I see.")}, samples=200)
    sstt = FakeStreamingSTT([partial("So I was looking at the plan and")] * 6)
    listen = lambda: Pause(np.zeros(16000 * 5, np.float32), 0.6, 4.5)
    script = [[SpeechStart()], [listen()], [listen()], [listen()], [listen()], [Discarded(0.1)],
              [SpeechStart()], [listen()]]
    s, ws, _tts, _host, _det = make(script, p=(0.2,), sstt=sstt, acks=kit, bot_backchannel=True, clock=clock)
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start", "acks": "often", "rt": {"backchannels": True}}); await settle(5)
    ws.push_audio(); await settle(5)
    clock.t += 4.5
    for _ in range(4):                                     # four listen rows, 7 s apart, one utterance
        ws.push_audio(); await settle(20)
        clock.t += 7.0
    bcs = lambda: [m for m in ws.json_of("say") if m["kind"] == "backchannel"]
    assert len(bcs()) == 3                                 # ≤ 3 per utterance (SPEC §4.7)
    ws.push_audio(); await settle(5)                       # that utterance ends (noise)
    ws.push_audio(); await settle(5)                       # a new one starts ...
    clock.t += 4.5
    ws.push_audio(); await settle(20)
    assert len(bcs()) == 4                                 # ... and may have its own
    ws.hang_up(); await task


async def test_say_end_follows_the_last_sentences_audio_even_when_stream_done_lands_mid_sentence():
    """Amendment T8-1: `say_end` only after the reply's last audio message."""
    tts = GatedTTS(chunks=3)
    s, ws, tts, _host, _det = make([], tts=tts)
    task = await start(s, ws)
    s.on_app_frame(frame(type="stream_start", message_id="run:1"))
    # The chunk ends with a newline: the splitter emits the sentence before stream_done.
    s.on_app_frame(frame(type="stream_chunk", message_id="run:1", text="Only one sentence here.\n"))
    await settle(10)
    assert ws.json_of("say") and not ws.json_of("say_end")       # the say is out, audio is not
    s.on_app_frame(frame(type="stream_done", message_id="m1", provisional_id="run:1",
                         message={"id": "m1", "role": "assistant", "content": "Only one sentence here."}))
    await settle(10)
    assert not ws.json_of("say_end")                             # stream_done mid-sentence: not yet
    tts.gate.set(); await settle(20)
    order = [m.get("type") for m in ws.sent if isinstance(m, dict) and m.get("type") in ("audio", "say_end")]
    assert order[-1] == "say_end" and order.count("audio") == 3
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# the tick verdict: a stop word cuts at once; ticks run faster over a reply
# --------------------------------------------------------------------------- #

async def test_a_stop_word_in_a_tick_cuts_at_once_and_the_rest_becomes_a_turn():
    """interrupt.wav: "Wait, stop, that's not what I meant." has no 0.25 s pause in its
    first second; the tick partial "Wait" cuts the reply, the utterance goes on, and at
    its end the run is aborted (keep_partial) and the words are the next turn."""
    sstt = FakeStreamingSTT([partial("Wait, stop, that's not what I meant.")],
                            ticks=[partial("Wait")])
    _s, ws, _tts, host, _det, task = await over_reply([[SpeechStart()], [], [pause()]], words="x",
                                                   sstt=sstt, tts=FakeTTS(chunk_samples=24000))
    ws.push_json({"type": "heard", "gen": 0, "seq": 1, "played_ms": 700}); await settle(3)
    await speak(ws, 2, wait=40)
    assert ws.json_of("barge_in") == [{"type": "barge_in", "gen": 1}]
    assert not host.of("abort") and not host.of("submit")              # cut, not aborted, no turn yet
    await speak(ws, 1, wait=60)                                         # the pause: real words
    names = host.names()
    assert ("abort", True) in host.calls and names.index("abort") < names.index("submit")
    assert host.of("submit")[0][1] == "Wait, stop, that's not what I meant."
    # The reply had landed whole (its row is m1): the cut's note is on it at once.
    (note,) = host.of("note_heard")
    assert note[1] == "m1" and note[3] == "You" and note[4].startswith("have two things")
    m = next(x for x in ws.json_of("metrics") if "eot_to_transcript_ms" in x)
    assert m["interrupt_to_submit_ms"] >= 0
    ws.hang_up(); await task


async def test_a_tick_stop_whose_utterance_was_only_stop_makes_no_turn():
    sstt = FakeStreamingSTT([partial("Wait, stop.")], ticks=[partial("Wait")])
    _s, ws, _tts, host, _det, task = await over_reply([[SpeechStart()], [], [pause(speech_s=0.8)]],
                                                   words="x", sstt=sstt)
    await speak(ws, 2, wait=40)
    assert ws.json_of("barge_in")
    await speak(ws, 1, wait=40)
    assert not host.of("submit") and ("abort", True) not in host.calls
    assert ws.json_of("state")[-1]["state"] == "listening"
    ws.hang_up(); await task


async def test_ticks_run_every_half_second_while_speech_runs_over_a_reply():
    class Rec(FakeStreamingSTT):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.tick_interval_s = 1.0
            self.seen: list = []

        def tick(self):
            self.seen.append(self.tick_interval_s)

    sstt = Rec([])
    s, ws, _tts, _host, _det, task = await over_reply([[SpeechStart()], []], words="x", sstt=sstt)
    await speak(ws, 2, wait=20)
    assert sstt.seen and sstt.seen[-1] == vs.TICK_OVER_REPLY_S
    ws.push_json({"type": "playback_done", "gen": s.gen}); await settle(5)
    s._duck = False
    s.sstt.seen.clear()
    await s._tick()
    assert sstt.seen == [1.0]                                           # captions alone: the STT's own cadence
    ws.hang_up(); await task


async def test_a_tick_stop_on_a_reply_still_streaming_notes_the_row_the_later_abort_persists():
    sstt = FakeStreamingSTT([partial("Wait, stop, that's not what I meant.")], ticks=[partial("Wait")])
    s, ws, _tts, host, _det = make([[SpeechStart()], [], [pause()]], p=(0.95,), sstt=sstt,
                                 tts=FakeTTS(chunk_samples=24000))
    task = await start(s, ws)
    s._current = vs._Turn(msg_id="u0", text="before", at=s.clock())
    s._owner = s._current
    s.on_app_frame(frame(type="stream_start", message_id="run:1"))
    s.on_app_frame(frame(type="stream_chunk", message_id="run:1", text=REPLY + " "))
    await settle(40)
    assert s.speaking
    ws.push_json({"type": "heard", "gen": 0, "seq": 2, "played_ms": 500}); await settle(3)
    await speak(ws, 2, wait=40)
    assert ws.json_of("barge_in") and not host.of("note_heard")        # no row yet: deferred
    await speak(ws, 1, wait=60)
    (note,) = host.of("note_heard")
    assert note[1] == "a-partial" and note[3].startswith("You have two things this afternoon. A")
    assert host.of("submit")[0][2]["interrupted"]["unheard"] == note[4]
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# our clips count as our voice; tick verdicts stay out of the backchannel rule
# --------------------------------------------------------------------------- #

async def test_an_ack_clip_playing_counts_as_our_voice_and_its_echo_is_never_a_turn():
    sstt = FakeStreamingSTT([], final="let me see")
    s, ws, _tts, host, det = make([[SpeechStart()], [turn_end()]], sstt=sstt)
    task = await start(s, ws)
    s._recent_said.append("Let me see.")                  # an ack line just went out ...
    s._line_until = s.clock() + 1.0                       # ... and is still playing
    s._apply_threshold()
    assert det.ep.cfg.threshold == pytest.approx(vs.SPEAKING_VAD_THRESHOLD)
    assert det.ep.cfg.min_speech_s == pytest.approx(vs.SERVER_AUDIO_BARGE_IN_S)
    await speak(ws, 2, wait=40)
    assert ws.json_of("echo_ignored") and not host.of("submit")
    ws.hang_up(); await task


async def test_a_tick_verdict_never_arms_the_two_backchannels_rule():
    """The first 0.3 s of "Actually…" transcribe as "Mm-hmm." on a tick; the pause's
    "Okay." must still be a backchannel (resume), not a second one in 3 s (= stop)."""
    sstt = FakeStreamingSTT([Partial("", "Okay.", 0, True, 0.5, 100)], ticks=[partial("Mm-hmm.")])
    s, ws, _tts, _host, _det, task = await over_reply(
        [[SpeechStart()], [], [Pause(np.zeros(8000, np.float32), 0.25, 0.5)]], words="x", sstt=sstt)
    gen = s.gen
    for _ in range(3):                                    # one detector feed each (no coalescing)
        ws.push_audio(); await settle(10)
    await settle(30)
    assert ws.json_of("resume") and not ws.json_of("barge_in") and s.gen == gen
    ws.hang_up(); await task


async def test_first_audio_metrics_carry_the_platform_facts(monkeypatch):
    from dispatch_voice.host import power
    monkeypatch.setattr(power, "platform_facts", lambda: {"platform_profile": "performance", "max_mhz": 5090})
    s, ws, _tts, _host, _det = make([[SpeechStart()], [pause()]], p=(0.95,))
    task = await start(s, ws)
    await speak(ws, 2, wait=40)
    await play(s, ws)
    m = next(x for x in ws.json_of("metrics") if "eot_to_first_audio_ms" in x)
    assert m["platform_profile"] == "performance" and m["max_mhz"] == 5090
    ws.hang_up(); await task


# --------------------------------------------------------------------------- #
# reconnect races (review T7b-fix notes for the hub)
# --------------------------------------------------------------------------- #

async def test_a_reply_cut_under_its_provisional_id_resumes_from_the_stored_row_without_repeats():
    """after_partial {run:7, 2}: the old socket heard two sentences of a reply it only knew as
    run:7; the new session only sees the stored row r7 (replies_after(user row))."""
    stored = [("r7", REPLY)]

    async def replies_after(tid, after):
        return stored if after == "u7" else None

    s, ws, _tts, _host, _det = make([])
    s.host.replies_after = replies_after
    task = await start(s, ws, after="u7", after_partial={"message_id": "run:7", "sentences": 2})
    await settle(30)
    assert [m["text"] for m in ws.json_of("say")] == ["Want a reminder?"]
    assert ws.json_of("say_end")[-1]["message_id"] == "r7"
    ws.hang_up(); await task


async def test_a_reply_still_streaming_at_the_reconnect_is_never_also_spoken_from_its_row():
    stored: list = []

    async def replies_after(tid, after):
        return stored

    s, ws, _tts, _host, _det = make([])
    s.host.replies_after = replies_after
    task = asyncio.create_task(s.run())
    await settle(3)
    # The subscribe snapshot: the reply is streaming (provisional id) ...
    s.on_app_frame(frame(type="stream_start", message_id="run:9"))
    s.on_app_frame(frame(type="stream_chunk", message_id="run:9", replace=True,
                         text="You have two things this afternoon. A dentist"))
    await settle(10)
    stored.append(("r9", REPLY))                         # ... and lands in the db meanwhile
    ws.push_json({"type": "start", "acks": "off", "after": "u9"}); await settle(20)
    s.on_app_frame(frame(type="stream_done", message_id="r9", provisional_id="run:9",
                         message={"id": "r9", "role": "assistant", "content": REPLY}))
    await settle(40)
    texts = [m["text"] for m in ws.json_of("say")]
    assert texts == ["You have two things this afternoon.", "A dentist appointment at three.", "Want a reminder?"]
    ws.hang_up(); await task


async def test_a_reply_resumed_from_its_row_keeps_a_late_stream_done_silent():
    stored = [("r5", REPLY)]

    async def replies_after(tid, after):
        return stored

    s, ws, _tts, _host, _det = make([])
    s.host.replies_after = replies_after
    task = await start(s, ws, after="u5")
    await settle(30)
    n = len(ws.json_of("say"))
    assert n == 3
    s.on_app_frame(frame(type="stream_done", message_id="r5", provisional_id="run:5",
                         message={"id": "r5", "role": "assistant", "content": REPLY}))
    await settle(30)
    assert len(ws.json_of("say")) == n
    ws.hang_up(); await task


async def test_without_speculation_the_submit_itself_waits_for_the_cut_note():
    """A gateway bot (no speculation hook to settle it): the tick's cut is still noting
    (the exact `heard` has not come) when the utterance ends — the submit waits for it."""
    sstt = FakeStreamingSTT([partial("Wait, stop, that's not what I meant.")], ticks=[partial("Wait")])
    s, ws, _tts, host, _det = make([[SpeechStart()], [], [pause()]], p=(0.95,), sstt=sstt,
                                 tts=FakeTTS(chunk_samples=24000), backend="gateway")
    task = await start(s, ws)
    s._current = vs._Turn(msg_id="u0", text="before", at=s.clock())
    s._owner = s._current
    s.on_app_frame(frame(type="stream_start", message_id="run:1"))
    s.on_app_frame(frame(type="stream_chunk", message_id="run:1", text=REPLY + " "))
    await settle(40)
    ws.push_json({"type": "heard", "gen": 0, "seq": 2, "played_ms": 500}); await settle(3)
    await speak(ws, 2, wait=5)
    await speak(ws, 1, wait=80)
    assert not host.of("speculate")
    submit = host.of("submit")[0]
    assert submit[2]["interrupted"] is not None and submit[2]["interrupted"]["unheard"]
    ws.hang_up(); await task


async def test_a_short_tick_with_a_command_word_cuts_early_unless_the_bot_said_it():
    """driver2: "Actually, wait, stop. Just tell me the first one." — the tick reads
    "Actually, wait." 1.3 s in; "stop" only completes 0.6 s later."""
    sstt = FakeStreamingSTT([partial("Actually, wait, stop. Just tell me the first one.")],
                            ticks=[partial("Actually, wait.")])
    _s, ws, _tts, host, _det, task = await over_reply([[SpeechStart()], [], [pause()]], words="x", sstt=sstt)
    await speak(ws, 2, wait=40)
    assert ws.json_of("barge_in") and not host.of("submit")              # cut at the tick
    await speak(ws, 1, wait=60)
    assert host.of("submit")[0][1] == "Actually, wait, stop. Just tell me the first one."
    ws.hang_up(); await task
    # The same words while the bot itself is saying "wait": possibly our own echo.
    sstt2 = FakeStreamingSTT([partial("x")], ticks=[partial("Wait until")])
    s2, ws2, _tts2, _host2, _det2, task2 = await over_reply([[SpeechStart()], []], words="x", sstt=sstt2)
    s2._recent_said.append("Wait until three before you leave.")
    await speak(ws2, 2, wait=40)
    assert not ws2.json_of("barge_in")
    ws2.hang_up(); await task2


async def test_a_turn_end_that_fails_is_reported_and_the_chat_goes_on():
    class Boom(FakeStreamingSTT):
        def reset(self):
            super().reset()
            if self.resets == 2:                                       # the turn end's, not start's
                raise RuntimeError("recogniser exploded")

    sstt = Boom([partial(QUESTION)], final=QUESTION)
    s, ws, _tts, host, _det = make([[SpeechStart()], [turn_end()]], sstt=sstt)
    task = await start(s, ws)
    await speak(ws, 2, wait=40)
    assert not host.of("submit")
    assert ws.json_of("error")[-1]["code"] == "stt_failed"             # said, not swallowed
    assert ws.json_of("state")[-1]["state"] == "listening"
    assert not task.done()                                             # the chat goes on
    ws.hang_up(); await task


async def test_a_failing_background_step_is_logged_not_left_unretrieved(caplog):
    s, ws, _tts, _host, _det = make([])
    task = await start(s, ws)

    async def boom():
        raise RuntimeError("background step exploded")

    s._spawn(boom())
    await settle(5)
    assert any("background step failed" in r.getMessage() for r in caplog.records)
    ws.hang_up(); await task


async def test_an_addendum_begun_before_the_reply_became_audible_still_merges():
    """The VAD confirms speech 0.25 s after its onset: a reply whose first audio went out
    inside that quarter second did not interrupt the driver — "oh, and tomorrow" merges."""
    clock = Clock()
    s, ws, _tts, host, _det = addendum_session(clock=clock)
    task = await start(s, ws)
    await speak(ws, 2, wait=40)
    s.on_app_frame(frame(type="message", message={"id": "u1", "role": "user", "content": QUESTION}))
    clock.t += 1.9
    await play(s, ws)                                     # first audio at +1.9 s ...
    assert s._current.said
    clock.t += 0.15                                       # ... the speech, begun at +1.8 s, confirmed at +2.05 s
    ws.push_audio(); await settle(10)
    assert s._hold and ws.json_of("duck")[-1]["on"] is True
    ws.push_audio(); await settle(60)
    assert host.of("submit")[-1][1] == "What's on my calendar this afternoon Oh and tomorrow."
    assert ws.json_of("transcript")[-1]["merged"] is True and ("abort", True) not in host.calls
    ws.hang_up(); await task


async def test_speech_begun_after_the_reply_became_audible_is_an_interruption():
    clock = Clock()
    s, ws, _tts, host, _det = addendum_session(clock=clock)
    task = await start(s, ws)
    await speak(ws, 2, wait=40)
    s.on_app_frame(frame(type="message", message={"id": "u1", "role": "user", "content": QUESTION}))
    clock.t += 1.0
    await play(s, ws)                                     # audible at +1.0 s
    clock.t += 0.6                                        # speech began at +1.35 s: after it
    ws.push_audio(); await settle(10)
    assert not s._hold
    ws.push_audio(); await settle(60)
    assert ws.json_of("transcript")[-1]["merged"] is False and ("abort", True) in host.calls
    ws.hang_up(); await task


async def test_a_chat_nobody_speaks_in_is_ended_by_the_server_and_stops_touching():
    """FR-hardening L3: the client's "Hang up when quiet" is a device setting
    ("Never" is a choice), so the server bounds an idle chat itself: with no
    utterance and no reply sentence for rt.quiet_max_s it stops sliding the unlocked
    session and ends the chat (4002 "quiet"). Speech resets the clock."""
    clock = Clock()
    touches: list = []
    host = Host()
    ws = FakeWS()
    det = ProbeDetector([])
    s = VoiceSession(ws, thread_id=THREAD, settings=settings(), detector=det, stt_engine=FakeSTT("x"),
                     tts_engine=FakeTTS(), host=host.build(touches=touches), clock=clock,
                     rt=RtSettings.from_mapping({"quiet_max_s": 600}))
    s.watch_tick_s = 0.005
    task = asyncio.create_task(s.run())
    ws.push_json({"type": "start"}); await settle(5)

    async def ping_for(seconds: float) -> None:
        for _ in range(int(seconds / 2)):
            clock.t += 2.0
            ws.push_json({"type": "ping", "t": 1}); await real_pause(0.002)

    await ping_for(500)
    assert not task.done() and touches
    await s.state("hearing")                       # the listener speaks: the clock restarts
    await ping_for(500)
    await settle(5)
    assert not task.done(), "500 s after speech is not quiet yet"
    n = len(touches)
    await ping_for(120)
    await asyncio.wait_for(task, 2.0)
    assert s.close_code == (vs.QUIET_CLOSE, vs.QUIET_REASON)
    assert len(touches) - n <= 4                   # nothing touched once it was over
    assert RtSettings.from_mapping({"quiet_max_s": 5}).quiet_max_s == 600      # clamped
    assert RtSettings().quiet_max_s == 2400         # above the longest device choice (30 min)
