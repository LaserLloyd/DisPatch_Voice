"""ParakeetTailSTT commit logic with a scripted recogniser; final() after an in-flight at_pause;
tick rate; stable words; the spinning config entry; build_streaming_stt selection and fallback."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from forge_voice_core.rt import RtSettings
from forge_voice_core.rt.stt_stream import (
    COMMIT_AFTER_S,
    ParakeetTailSTT,
    Partial,
    StreamingSTT,
    build_streaming_stt,
    ort_options,
)
from rt.conftest import ScriptedRecognizer, silence_frames, speech_frames


def feed(stt, frames):
    for fr, p in frames:
        stt.push(fr, vad_prob=p)


def test_partial_shape_and_protocol(recognizer, clock):
    stt = ParakeetTailSTT(recognizer, clock=clock)
    assert isinstance(stt, StreamingSTT)
    feed(stt, speech_frames(2.0))
    p = stt.at_pause()
    assert isinstance(p, Partial) and p.committed == "" and p.tail == "2.0s" and p.t_audio == pytest.approx(2.0, abs=0.05)
    assert p.punct_final is False and p.stable_words == 0 and p.cost_ms == 0
    assert p.text == "2.0s"


def test_stable_words_is_agreement_minus_two():
    texts = iter(["what's on my calendar this", "what's on my calendar this afternoon?",
                  "what's on my calendar this afternoon?"])
    stt = ParakeetTailSTT(ScriptedRecognizer(lambda a: next(texts)))
    feed(stt, speech_frames(1.0))
    assert stt.at_pause().stable_words == 0          # nothing to agree with yet
    feed(stt, speech_frames(1.0))
    p = stt.at_pause()
    assert p.stable_words == 3 and p.punct_final      # 5 agree, minus 2
    assert stt.at_pause() is not None and len(stt._prev_tail_words) == 6   # no new speech: reused, no call
    feed(stt, speech_frames(0.1))
    p = stt.at_pause()
    assert p.stable_words == 4                        # 6 agree, minus 2


def test_tail_closes_at_an_old_silence_once_longer_than_five_seconds():
    seen = []

    def rec(a):
        seen.append(round(len(a) / 16000, 1))
        return f"seg{len(seen)}"

    stt = ParakeetTailSTT(ScriptedRecognizer(rec))
    feed(stt, speech_frames(2.0))
    feed(stt, silence_frames(0.4))        # a ≥ 300 ms silence at 2.0–2.4 s
    feed(stt, speech_frames(2.0))
    p = stt.at_pause()                     # 4.4 s: under the limit, nothing closes
    assert p.committed == "" and p.tail == "seg1" and seen == [4.4]
    feed(stt, speech_frames(1.5))          # 5.9 s: closes at the silence (ended 3.5 s ago)
    p = stt.at_pause()
    assert p.committed == "seg2" and p.tail == "seg3"
    assert seen[1] == pytest.approx(2.4, abs=0.1) and seen[2] == pytest.approx(3.5, abs=0.1)
    # The committed text never changes afterwards, whatever the tail does.
    feed(stt, speech_frames(1.0))
    p2 = stt.at_pause()
    assert p2.committed == "seg2" and p2.tail == "seg4"
    assert stt.final().startswith("seg2 ")


def test_a_young_silence_does_not_close_the_tail():
    stt = ParakeetTailSTT(ScriptedRecognizer())
    feed(stt, speech_frames(5.5))
    feed(stt, silence_frames(0.4))        # the only silence is < 1 s old
    p = stt.at_pause()
    assert p.committed == "" and p.t_audio == pytest.approx(5.9, abs=0.05)


def test_final_reuses_a_fresh_partial_else_one_more_call(recognizer):
    stt = ParakeetTailSTT(recognizer)
    feed(stt, speech_frames(2.0))
    stt.at_pause()
    assert len(recognizer.calls) == 1
    feed(stt, speech_frames(0.2))          # < 0.3 s of new audio since the partial
    assert stt.final() == "2.0s" and len(recognizer.calls) == 1
    feed(stt, speech_frames(0.5))          # now 0.7 s of new audio: one more tail call
    assert stt.final() == "2.7s" and len(recognizer.calls) == 2


def test_final_waits_for_the_inflight_pause_call():
    rec = ScriptedRecognizer(delay_s=0.3)
    stt = ParakeetTailSTT(rec)
    feed(stt, speech_frames(2.0))
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("p", stt.at_pause()))
    th.start()
    time.sleep(0.05)
    assert stt.inflight
    t0 = time.perf_counter()
    text = stt.final()                      # blocks until the pause call returns, then reuses it
    dt = time.perf_counter() - t0
    th.join()
    assert text == "2.0s" and len(rec.calls) == 1 and 0.15 < dt < 0.6


def test_final_reuses_a_partial_followed_only_by_silence(recognizer):
    stt = ParakeetTailSTT(recognizer)
    feed(stt, speech_frames(2.0))
    stt.at_pause()
    feed(stt, silence_frames(1.2))         # a long hold: silence only → the words cannot change
    assert stt.final() == "2.0s" and len(recognizer.calls) == 1 and stt.primed_hits == 1


# ---- prime(): the tail call started before the probe -------------------------- #

def test_prime_then_silence_is_reused_at_the_probe():
    rec = ScriptedRecognizer(delay_s=0.2)
    stt = ParakeetTailSTT(rec)
    feed(stt, speech_frames(2.0))
    feed(stt, silence_frames(0.15))
    assert stt.prime() is True
    assert stt.inflight and stt.prime() is False       # one in flight: no second
    assert stt.tick() is None                           # ticks never race a primed call
    feed(stt, silence_frames(0.10))                     # the probe fires at 0.25 s
    t0 = time.perf_counter()
    p = stt.at_pause()                                  # waits for the primed call, reuses it
    dt = time.perf_counter() - t0
    assert p.tail == "2.1s" and p.t_audio == pytest.approx(2.25, abs=0.05)   # covers 2.0 + 0.15 of audio
    assert len(rec.calls) == 1 and stt.primed_hits == 1 and dt < 0.25
    assert stt.final() == "2.1s" and len(rec.calls) == 1
    # Once reusable, a second prime() is a no-op too.
    assert stt.prime() is False


def test_prime_then_speech_is_transcribed_again():
    rec = ScriptedRecognizer(delay_s=0.05)
    stt = ParakeetTailSTT(rec)
    feed(stt, speech_frames(2.0))
    feed(stt, silence_frames(0.15))
    assert stt.prime() is True
    feed(stt, speech_frames(0.5))                       # a false pause: the user went on
    feed(stt, silence_frames(0.25))
    p = stt.at_pause()
    assert p.tail == "2.9s" and len(rec.calls) == 2 and stt.primed_hits == 0


def test_prime_needs_audio_and_survives_reset():
    rec = ScriptedRecognizer(delay_s=0.1)
    stt = ParakeetTailSTT(rec)
    assert stt.prime() is False                         # nothing to hear
    feed(stt, speech_frames(1.0))
    assert stt.prime() is True
    stt.reset()                                         # the primed result belongs to nobody now
    stt._prime_thread.join(1.0)
    assert stt.last_partial is None or stt.last_partial.t_audio == 0.0
    feed(stt, speech_frames(0.5))
    assert stt.at_pause().tail == "0.5s"


def test_tick_rate_limit_and_inflight(recognizer, clock):
    stt = ParakeetTailSTT(recognizer, clock=clock)
    feed(stt, speech_frames(1.0))
    assert stt.tick() is not None
    assert stt.tick() is None               # < 1 s since the last tick
    clock.advance(1.0)
    feed(stt, speech_frames(0.5))
    p = stt.tick()
    assert p is not None and p.tail == "1.5s"
    stt._inflight.set()
    clock.advance(1.0)
    assert stt.tick() is None               # a pause call is in flight
    stt._inflight.clear()
    stt.reset()
    assert stt.t_audio == 0 and stt.committed == ""
    clock.advance(1.0)
    assert stt.tick() is None               # < 0.1 s of audio: nothing to hear


def test_push_accepts_int16_and_reframes(recognizer):
    stt = ParakeetTailSTT(recognizer)
    stt.push(np.zeros(1600, np.int16), vad_prob=0.9)       # 100 ms, not a multiple of 512
    stt.push(np.zeros(0, np.int16), vad_prob=0.9)
    assert stt.t_audio == pytest.approx(0.1)


def test_recognizer_failure_is_an_empty_partial_not_a_crash():
    def boom(a):
        raise RuntimeError("model gone")
    stt = ParakeetTailSTT(ScriptedRecognizer(boom))
    feed(stt, speech_frames(1.0))
    assert stt.at_pause().tail == "" and stt.final() == ""


# ---- sessions and the factory -------------------------------------------------- #

def test_ort_options_turn_spinning_off_for_the_bursty_sessions():
    pytest.importorskip("onnxruntime")
    so = ort_options(6)
    assert so.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    assert so.intra_op_num_threads == 6 and so.inter_op_num_threads == 1
    assert ort_options(2, spinning=True).get_session_config_entry("session.intra_op.allow_spinning") == "1"


def test_build_streaming_stt_wraps_the_shared_engine_and_falls_back_from_moonshine(monkeypatch):
    shared = SimpleNamespace(transcribe=lambda a: "shared")
    settings = SimpleNamespace(rt=RtSettings(), threads=6, parakeet_dir=None, models_dir=None)
    stt = build_streaming_stt(settings, shared)
    assert isinstance(stt, ParakeetTailSTT)
    feed(stt, speech_frames(1.0))
    assert stt.final() == "shared"
    # Moonshine asked for but unavailable → Parakeet tail mode, not an error.
    monkeypatch.setattr("forge_voice_core.rt.stt_stream.moonshine_model",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no model")))
    stt2 = build_streaming_stt(SimpleNamespace(rt=RtSettings(stt_engine="moonshine"), threads=6,
                                               parakeet_dir=None, models_dir=None), shared)
    assert isinstance(stt2, ParakeetTailSTT)
    with pytest.raises(RuntimeError):
        build_streaming_stt(SimpleNamespace(rt=RtSettings(), threads=6, parakeet_dir=None), None)


def test_commit_after_default_is_five_seconds():
    assert COMMIT_AFTER_S == 5.0


# ---- review T4 #1: reset() during an in-flight call never leaks into the next turn ---- #

class GatedRecognizer:
    """The first call blocks on ``gate`` (a 'slow' recogniser), later ones answer at once;
    every answer names the audio length so a leaked result is recognisable."""

    def __init__(self, first_text="TURN1-WORDS"):
        self.gate = threading.Event()
        self.n = 0
        self.first_text = first_text
        self.sizes: list[float] = []

    def __call__(self, audio):
        self.n += 1
        self.sizes.append(round(audio.size / 16000, 2))
        if self.n == 1:
            self.gate.wait(5.0)
            return self.first_text
        return f"turn2-{audio.size / 16000:.1f}s"


def _in_thread(fn):
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("v", fn()))
    th.start()
    time.sleep(0.05)
    return th, out


def test_reset_during_a_plain_pause_call_discards_its_result():
    rec = GatedRecognizer()
    stt = ParakeetTailSTT(rec)
    feed(stt, speech_frames(6.0))
    feed(stt, silence_frames(0.25))
    th, out = _in_thread(stt.at_pause)
    assert stt.inflight
    stt.reset()                                         # cancel / hang-up / new turn mid-call
    rec.gate.set()
    th.join(5.0)
    assert out["v"].text == "" and out["v"].t_audio == 0.0   # the cancelled call's answer is empty
    assert stt.last_partial is None and stt.committed == ""
    feed(stt, speech_frames(1.0))                       # the NEXT turn, shorter than the old partial
    feed(stt, silence_frames(0.25))
    p = stt.at_pause()
    assert p.text == "turn2-1.2s" and rec.n == 2 and stt.primed_hits == 0
    assert stt.final() == "turn2-1.2s" and rec.n == 2


def test_reset_during_a_closing_call_never_leaks_into_committed():
    rec = GatedRecognizer(first_text="OLD-SEGMENT")
    stt = ParakeetTailSTT(rec)
    feed(stt, speech_frames(3.0))
    feed(stt, silence_frames(0.5))                      # a ≥ 0.3 s silence …
    feed(stt, speech_frames(3.0))                       # … ≥ 1 s old: the tail (6.5 s) closes there
    feed(stt, silence_frames(0.15))
    assert stt.prime() is True                          # the primed path closes the segment too
    stt.reset()
    rec.gate.set()
    stt._prime_thread.join(5.0)
    assert stt.committed == "" and stt.last_partial is None
    feed(stt, speech_frames(1.0))
    p = stt.at_pause()
    assert p.committed == "" and p.text.startswith("turn2-") and "OLD" not in stt.final()


def test_reset_during_tick_and_final_is_discarded(clock):
    rec = GatedRecognizer()
    stt = ParakeetTailSTT(rec, clock=clock)
    feed(stt, speech_frames(2.0))
    th, out = _in_thread(stt.tick)
    stt.reset()
    rec.gate.set()
    th.join(5.0)
    assert out["v"] is None and stt.last_partial is None
    rec2 = GatedRecognizer()
    stt2 = ParakeetTailSTT(rec2, clock=clock)
    feed(stt2, speech_frames(2.0))
    th, out = _in_thread(stt2.final)
    stt2.reset()
    rec2.gate.set()
    th.join(5.0)
    assert out["v"] == "" and stt2.last_partial is None     # nobody's words


# ---- review T4 #3: the tail is capped even without a qualifying silence ------------ #

def _sizes_recognizer():
    sizes: list[float] = []
    return sizes, ScriptedRecognizer(lambda a: (sizes.append(round(a.size / 16000, 2)) or f"w{len(sizes)}"))


def test_tail_cap_bounds_every_call_without_silence():
    sizes, rec = _sizes_recognizer()
    stt = ParakeetTailSTT(rec)
    feed(stt, speech_frames(30.0))                      # road noise: Silero never drops below 0.35
    p = stt.at_pause()
    assert max(sizes) <= stt.tail_max_s + 0.05, sizes   # no call sees more than the cap
    assert p.committed and stt._tail_seconds() <= stt.tail_max_s
    assert sum(sizes) == pytest.approx(30.0, abs=0.1)   # … and nothing was dropped
    feed(stt, speech_frames(30.0))
    stt.at_pause()
    assert max(sizes) <= stt.tail_max_s + 0.05 and stt._tail_seconds() <= stt.tail_max_s
    assert stt.final().startswith("w1 w2")              # committed text accumulates in order


def test_short_pauses_still_bound_the_tail_and_cut_at_the_quietest_frame():
    sizes, rec = _sizes_recognizer()
    stt = ParakeetTailSTT(rec)
    for _ in range(10):
        feed(stt, speech_frames(2.0))
        feed(stt, silence_frames(0.25))                 # the probe pauses; never a ≥ 0.3 s silence
    probs_before = list(stt._probs)
    stt.at_pause()
    assert max(sizes) <= stt.tail_max_s + 0.05, sizes
    assert stt._tail_seconds() <= stt.tail_max_s
    # Every forced cut lands inside a 0.25 s pause (the quietest frames), never mid-speech.
    closed = 0
    for size in sizes[:-1]:                             # the closing segments, in order
        closed += round(size * 16000 / 512)
        assert probs_before[closed - 1] < 0.35, (size, closed)


def test_tick_closes_a_long_tail_too(clock):
    sizes, rec = _sizes_recognizer()
    stt = ParakeetTailSTT(rec, clock=clock)
    for _ in range(14):                                 # live captions on a 14 s monologue
        feed(stt, speech_frames(1.0))
        clock.advance(1.0)
        stt.tick()
    assert max(sizes) <= stt.tail_max_s + 0.05, sizes
    assert stt.committed


def test_tail_max_default_and_floor():
    assert ParakeetTailSTT(ScriptedRecognizer()).tail_max_s == 10.0
    assert ParakeetTailSTT(ScriptedRecognizer(), commit_after_s=5.0, tail_max_s=2.0).tail_max_s == 5.0


# ---- review T4 #11: final() can be bounded ------------------------------------------ #

def test_final_with_a_timeout_returns_the_best_known_text():
    rec = ScriptedRecognizer(delay_s=0.4)
    stt = ParakeetTailSTT(rec)
    feed(stt, speech_frames(2.0))
    stt.at_pause()                                      # "2.0s" known
    feed(stt, speech_frames(1.0))
    th, _ = _in_thread(stt.at_pause)                    # a slow call in flight
    t0 = time.perf_counter()
    text = stt.final(timeout_s=0.05)
    assert text == "2.0s" and time.perf_counter() - t0 < 0.3
    th.join(2.0)
    assert stt.final(timeout_s=1.0) == "3.0s"           # the finished call is reused


# ---- review T4 #7: Moonshine push() never waits on the transcriber ------------------ #

def test_moonshine_push_does_not_block_on_update_transcription(monkeypatch):
    import sys
    import types

    class FakeTranscriber:
        MOONSHINE_FLAG_FORCE_UPDATE = 1
        inst = None

        def __init__(self, model_path, model_arch, update_interval=0.5):
            FakeTranscriber.inst = self
            self.audio: list[int] = []
            self.lock = threading.Lock()
            self.slow = threading.Event()

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

        def add_audio(self, samples, rate):
            self.audio.append(len(samples))

        def update_transcription(self, flags=0):
            self.slow.wait(1.0)                         # the 76–522 ms call
            return types.SimpleNamespace(lines=[types.SimpleNamespace(text=f"{sum(self.audio)}")])

    api = types.ModuleType("moonshine_voice.moonshine_api")
    api.ModelArch = types.SimpleNamespace(MEDIUM_STREAMING="medium")
    tr = types.ModuleType("moonshine_voice.transcriber")
    tr.Transcriber = FakeTranscriber
    pkg = types.ModuleType("moonshine_voice")
    pkg.moonshine_api, pkg.transcriber = api, tr
    monkeypatch.setitem(sys.modules, "moonshine_voice", pkg)
    monkeypatch.setitem(sys.modules, "moonshine_voice.moonshine_api", api)
    monkeypatch.setitem(sys.modules, "moonshine_voice.transcriber", tr)
    from forge_voice_core.rt.stt_stream import MoonshineSTT
    stt = MoonshineSTT("/nonexistent", "medium")
    stt.push(np.zeros(512, np.float32), vad_prob=0.9)
    th, out = _in_thread(stt.at_pause)                  # the STT thread, inside update_transcription
    t0 = time.perf_counter()
    for _ in range(10):
        stt.push(np.zeros(512, np.float32), vad_prob=0.9)   # the event loop keeps pushing
    assert time.perf_counter() - t0 < 0.05
    assert stt.t_audio == pytest.approx(11 * 512 / 16000)
    FakeTranscriber.inst.slow.set()
    th.join(2.0)
    assert out["v"].tail == "512"                       # that call saw the one frame pushed before it
    assert stt.at_pause().tail == str(11 * 512)         # the next feeds the buffered ten
