"""Voice chat: endpointing on synthetic audio (no models needed).

The state machine is fed with the EnergyVAD test double so the tests pin the
LOGIC — debounce, blip rejection, Smart Turn probe, hard-silence fallback,
max length — independent of any model. Model-backed checks at the bottom run
only when DISPATCH_SPEECH_MODELS points at downloaded models.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from dispatch_voice.host.vad_turn import (
    FRAME,
    SAMPLE_RATE,
    Discarded,
    Endpointer,
    EndpointerConfig,
    EnergyVAD,
    SpeechStart,
    TurnDetector,
    TurnEnd,
    mel_filters,
    whisper_features,
)


def tone(seconds: float, freq: float = 220.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), np.float32)


def pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def run(audio: np.ndarray, turn_fn=None, chunk: int = 320, **cfg) -> list:
    det = TurnDetector(EnergyVAD(), Endpointer(EndpointerConfig(**cfg), turn_fn))
    data = pcm16(audio)
    events = []
    for i in range(0, len(data), chunk * 2):    # 20 ms chunks, like the browser
        events.extend(det.feed(data[i:i + chunk * 2]))
    return events


def kinds(events) -> list[str]:
    return [type(e).__name__ for e in events]


def test_silence_produces_nothing():
    assert run(silence(3.0)) == []


def test_speech_then_silence_ends_turn_without_smart_turn():
    ev = run(np.concatenate([silence(0.5), tone(1.0), silence(1.5)]))
    assert kinds(ev) == ["SpeechStart", "TurnEnd"]
    end = ev[-1]
    assert end.reason == "silence"
    # Utterance ≈ pre-roll + 1 s of tone + ≤0.2 s kept tail, not the 1.5 s of silence.
    assert 0.9 <= len(end.audio) / SAMPLE_RATE <= 1.6


def test_end_waits_for_configured_silence():
    # 0.5 s of silence is less than the 0.8 s fallback: still mid-turn.
    ev = run(np.concatenate([tone(1.0), silence(0.5)]))
    assert kinds(ev) == ["SpeechStart"]


def test_short_blip_is_discarded_not_a_turn():
    ev = run(np.concatenate([silence(0.3), tone(0.1), silence(1.0)]))
    assert kinds(ev) == ["Discarded"]
    assert isinstance(ev[0], Discarded)


def test_pause_inside_sentence_does_not_split_turn():
    ev = run(np.concatenate([tone(0.8), silence(0.4), tone(0.8), silence(1.2)]))
    assert kinds(ev) == ["SpeechStart", "TurnEnd"]
    assert len(ev[-1].audio) / SAMPLE_RATE > 1.8


def test_smart_turn_complete_ends_early():
    calls = []

    def finished(audio):
        calls.append(len(audio))
        return 0.95

    audio = np.concatenate([tone(1.0), silence(1.5)])
    ev = run(audio, turn_fn=finished)
    assert kinds(ev) == ["SpeechStart", "TurnEnd"]
    assert ev[-1].reason == "smart_turn" and ev[-1].turn_prob == pytest.approx(0.95)
    assert len(calls) == 1          # probed once per pause, not every frame


def test_smart_turn_incomplete_falls_back_to_hard_silence():
    ev = run(np.concatenate([tone(1.0), silence(2.0)]), turn_fn=lambda a: 0.1)
    assert kinds(ev) == ["SpeechStart", "TurnEnd"]
    assert ev[-1].reason == "silence"


def test_smart_turn_incomplete_then_more_speech_reprobes():
    probes = []

    def fn(audio):
        probes.append(len(audio))
        return 0.1 if len(probes) == 1 else 0.9

    ev = run(np.concatenate([tone(0.6), silence(0.4), tone(0.6), silence(1.0)]), turn_fn=fn)
    assert kinds(ev) == ["SpeechStart", "TurnEnd"]
    assert ev[-1].reason == "smart_turn" and len(probes) == 2


def test_smart_turn_exception_does_not_break_endpointing():
    def boom(audio):
        raise RuntimeError("model gone")
    ev = run(np.concatenate([tone(1.0), silence(2.0)]), turn_fn=boom)
    assert ev[-1].reason == "silence"


def test_max_length_cuts_a_monologue():
    ev = run(tone(4.0), max_utterance_s=2.0)
    ends = [e for e in ev if isinstance(e, TurnEnd)]
    assert ends and ends[0].reason == "max_length"
    assert abs(len(ends[0].audio) / SAMPLE_RATE - 2.0) < 0.1
    # Speech that keeps going after the cut simply starts the next turn.
    assert kinds(ev)[-1] == "SpeechStart"


def test_odd_byte_chunks_are_tolerated():
    ev = run(np.concatenate([tone(1.0), silence(1.5)]), chunk=333)
    assert kinds(ev) == ["SpeechStart", "TurnEnd"]


def test_mel_filters_shape_and_coverage():
    fb = mel_filters()
    assert fb.shape == (201, 80)
    assert (fb >= 0).all() and (fb.sum(axis=0) > 0).all()


def test_whisper_features_shape_and_range():
    f = whisper_features(tone(2.0))
    assert f.shape == (1, 80, 800) and f.dtype == np.float32
    assert np.isfinite(f).all()
    # Whisper's (log10 + 4) / 4 scaling with an 8-decade floor.
    assert f.max() - f.min() <= 2.0 + 1e-5


# --------------------------------------------------------------------------- #
# Real models (skipped unless downloaded)
# --------------------------------------------------------------------------- #

_MODELS = Path(os.environ.get("DISPATCH_SPEECH_MODELS", "/nonexistent"))


@pytest.mark.skipif(not (_MODELS / "silero_vad.onnx").exists(), reason="voice models not downloaded")
def test_silero_ignores_silence_and_steady_tone_quietly():
    from dispatch_voice.host.vad_turn import SileroVAD
    vad = SileroVAD(_MODELS / "silero_vad.onnx")
    probs = [vad(silence(0.032)[:FRAME]) for _ in range(30)]
    assert max(probs) < 0.2


@pytest.mark.skipif(not (_MODELS / "smart-turn-v3.2-cpu.onnx").exists(), reason="voice models not downloaded")
def test_smart_turn_returns_probability():
    from dispatch_voice.host.vad_turn import SmartTurn
    p = SmartTurn(_MODELS / "smart-turn-v3.2-cpu.onnx")(tone(1.0))
    assert 0.0 <= p <= 1.0


def test_event_types_are_exported():
    assert SpeechStart and TurnEnd


def test_feed_with_end_annotates_every_event_with_its_uplink_position():
    """review-T3s4 #3: with `end` (the uplink offset just past the fed audio) each event
    carries `pos` (just past the frame that produced it) and a SpeechStart its utterance
    `start` (pre-roll included); `end_turn()`'s TurnEnd carries the processed position."""
    from dispatch_voice.host.vad_turn import (
        FRAME,
        Endpointer,
        EndpointerConfig,
        SpeechStart,
        TurnDetector,
        TurnEnd,
    )
    cfg = EndpointerConfig(min_speech_s=0.064, pre_roll_s=0.096, probe_events=True,
                           turn_hard_silence_s=0.5)
    det = TurnDetector(lambda f: 1.0 if float(abs(f).mean()) > 0.05 else 0.0, Endpointer(cfg))
    pcm = np.zeros(24000, "<i2")                      # 1.5 s: the hard silence ends the turn
    pcm[FRAME * 10:FRAME * 20] = 8000                 # speech in frames 10..19
    base = 100_000
    ev = det.feed(pcm.tobytes(), end=base + 24000)
    ss = [e for e in ev if isinstance(e, SpeechStart)]
    assert ss and ss[0].pos == base + FRAME * 12      # confirmed after 2 frames of speech (64 ms)
    assert ss[0].start == base + FRAME * 7            # 3 pre-roll frames before frame 10
    ends = [e for e in ev if isinstance(e, TurnEnd)]
    assert ends and ends[0].pos - round(ends[0].silence_s * 16000) == base + FRAME * 20
    assert det.pos == base + 24000 - len(det._buf)
    # a decision's end carries the processed position
    pcm2 = np.zeros(4096, "<i2")
    pcm2[:FRAME * 4] = 8000
    det.feed(pcm2.tobytes(), end=base + 24000 + 4096)
    te = det.end_turn("policy", 0.9)
    assert te is not None and te.pos == det.pos == base + 24000 + 4096 - len(det._buf)
    # without `end`: nothing annotated
    plain = TurnDetector(lambda f: 1.0 if float(abs(f).mean()) > 0.05 else 0.0, Endpointer(EndpointerConfig(
        min_speech_s=0.064, pre_roll_s=0.096)))
    ev2 = plain.feed(pcm.tobytes())
    assert all(getattr(e, "pos", None) is None for e in ev2) and plain.pos is None
