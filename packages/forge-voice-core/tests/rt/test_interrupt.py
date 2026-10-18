"""Interrupt sensitivity (1.5.0): the profiles, the level meter and the stricter stop rule,
with synthetic noise and speech levels (no models)."""
from __future__ import annotations

import numpy as np
import pytest

from forge_voice_core.rt.backchannel import classify_interruption, has_strong_stop
from forge_voice_core.rt.interrupt import (
    AMBIENT_MIN_FRAMES,
    DEFAULT_SENSITIVITY,
    PROFILES,
    LevelMeter,
    frame_db,
    profile,
    sensitivity_value,
    speech_level_db,
)

SR = 16000


def tone(seconds: float, amp: float, f: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def noise(seconds: float, amp: float, seed: int = 3) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(int(seconds * SR)) * amp).astype(np.float32)


def db_of(amp_sine: float) -> float:
    return 20 * np.log10(amp_sine / np.sqrt(2))


# ------------------------------------------------------------------ profiles


def test_default_is_normal_and_high_is_the_1_4_0_rules():
    assert DEFAULT_SENSITIVITY == "normal"
    hi = PROFILES["high"]
    assert (hi.duck_speech_s, hi.tick_min_words, hi.floor_margin_db, hi.voice_margin_db) == (None, 3, None, None)
    assert hi.weak_stops and hi.early_any_command and hi.optimistic_duck


@pytest.mark.parametrize("name", ["normal", "low"])
def test_normal_and_low_are_less_eager_than_high_on_every_knob(name):
    p = PROFILES[name]
    assert p.duck_speech_s is not None and p.duck_speech_s > 0.25
    assert p.tick_min_words > 3
    assert p.floor_margin_db is not None and p.voice_margin_db is not None
    assert not p.weak_stops and not p.early_any_command and not p.optimistic_duck


def test_low_is_stricter_than_normal():
    lo, no = PROFILES["low"], PROFILES["normal"]
    assert lo.duck_speech_s > no.duck_speech_s
    assert lo.tick_min_words > no.tick_min_words
    assert lo.floor_margin_db > no.floor_margin_db
    assert lo.voice_margin_db < no.voice_margin_db      # closer to your own level


def test_sensitivity_values_fall_back_to_the_default():
    for v in ("low", "normal", "high"):
        assert sensitivity_value(v) == v and profile(v).name == v
    for junk in ("loud", "", None, 3, True, ["high"]):
        assert sensitivity_value(junk) == "normal"
        assert profile(junk).name == "normal"


# ------------------------------------------------------------------ levels


def test_speech_level_reads_the_loud_part_and_ignores_digital_silence():
    # 0.3 s of zero-fill pre-roll, then speech-level tone: the level is the tone's.
    a = np.concatenate([np.zeros(int(0.3 * SR), np.float32), tone(0.6, 0.3)])
    assert speech_level_db(a) == pytest.approx(db_of(0.3), abs=0.5)
    assert speech_level_db(np.zeros(SR, np.float32)) is None
    assert speech_level_db(tone(0.05, 0.3)) is None           # under three frames
    assert speech_level_db(None) is None
    assert frame_db(tone(0.032, 0.3)) == pytest.approx(db_of(0.3), abs=0.5)


def feed_room(m: LevelMeter, amp: float, frames: int = 200, seed: int = 1):
    x = noise(frames * 512 / SR, amp, seed)
    for i in range(frames):
        m.observe(frame_db(x[i * 512:(i + 1) * 512]), 0.05)


def test_the_floor_follows_non_speech_frames_only():
    m = LevelMeter()
    assert m.floor_db is None
    feed_room(m, 0.001, frames=AMBIENT_MIN_FRAMES - 1)
    assert m.floor_db is None, "not trusted before ~1 s of room"
    feed_room(m, 0.001, frames=50)
    floor = m.floor_db
    assert floor == pytest.approx(-60, abs=2)
    for _ in range(300):
        m.observe(-15.0, 0.95)        # speech frames never lift the floor
        m.observe(-120.0, 0.0)        # nor does digital silence lower it
    assert m.floor_db == pytest.approx(floor, abs=0.01)


def test_the_floor_rises_with_a_louder_room():
    m = LevelMeter()
    feed_room(m, 0.001, frames=320)
    quiet = m.floor_db
    feed_room(m, 0.02, frames=320, seed=2)      # a TV comes on: ~26 dB louder
    assert m.floor_db > quiet + 20


def test_voice_level_is_the_median_of_recent_turns():
    m = LevelMeter()
    m.note_voice(-18.0)
    assert m.voice_db is None, "one turn is not enough to know your voice"
    m.note_voice(-22.0)
    m.note_voice(None)
    m.note_voice(-20.0)
    assert m.voice_db == pytest.approx(-20.0)


def learned_meter(voice_amp=0.3, room_amp=0.001) -> LevelMeter:
    m = LevelMeter()
    feed_room(m, room_amp)
    for _ in range(3):
        m.note_voice(speech_level_db(tone(1.0, voice_amp)))
    return m


def test_gate_noise_and_background_talk_fail_your_own_voice_passes():
    m = learned_meter()                       # your voice ≈ −13.5 dB, room ≈ −60 dB
    normal, low, high = PROFILES["normal"], PROFILES["low"], PROFILES["high"]
    near = speech_level_db(tone(0.6, 0.25))   # you, a little softer than usual
    across = speech_level_db(tone(0.6, 0.03))  # someone across the room: −20 dB
    hiss = speech_level_db(noise(0.6, 0.0015))  # the room itself, Silero-ish "speech"
    assert m.judge(near, normal)[0] and m.judge(near, low)[0]
    ok, why = m.judge(across, normal)
    assert not ok and "your voice" in why
    assert not m.judge(across, low)[0]
    ok, why = m.judge(hiss, normal)
    assert not ok and "room floor" in why
    # "high" is the 1.4.0 rules: no level gate at all.
    assert m.judge(across, high) == (True, "") and m.judge(hiss, high) == (True, "")


def test_gate_margins_normal_vs_low():
    m = LevelMeter()
    for _ in range(2):
        m.note_voice(-20.0)
    normal, low = PROFILES["normal"], PROFILES["low"]
    assert m.judge(-28.0, normal)[0]          # 8 dB under your voice: still you at normal
    assert not m.judge(-28.0, low)[0]         # ... but low wants you within 6 dB
    assert not m.judge(-31.0, normal)[0]


def test_unknown_levels_never_block():
    m = LevelMeter()
    for p in PROFILES.values():
        assert m.judge(-50.0, p)[0], "nothing learned yet: no gate"
        assert m.judge(None, p)[0], "no audio: no gate"
    assert m.snapshot() == {"floor_db": None, "voice_db": None}


# ------------------------------------------------------------------ stop words


@pytest.mark.parametrize("words", ["What?", "Thank you.", "Sorry?", "No no.", "Okay, okay.", "Got it, thanks."])
def test_ambiguous_stop_words_are_not_stops_without_weak_stops(words):
    assert classify_interruption(words, dur_s=0.6, recent_said=[], prev_verdict=None, now=0.0) == "stop"
    assert classify_interruption(words, dur_s=0.6, recent_said=[], prev_verdict=None, now=0.0,
                                 weak_stops=False) == "backchannel"


@pytest.mark.parametrize("words", ["Stop.", "Wait.", "Okay, wait.", "Hold on.", "No, stop.", "That's enough.",
                                   "Pause.", "Hang on a second.", "uh, stop"])
def test_clear_stop_words_always_stop(words):
    assert has_strong_stop(words)
    for weak in (True, False):
        assert classify_interruption(words, dur_s=0.6, recent_said=[], prev_verdict=None, now=0.0,
                                     weak_stops=weak) == "stop", (words, weak)


def test_two_backchannels_make_a_stop_only_with_weak_stops():
    prev = ("backchannel", 10.0)
    assert classify_interruption("Mm.", dur_s=0.4, recent_said=[], prev_verdict=prev, now=11.0) == "stop"
    assert classify_interruption("Mm.", dur_s=0.4, recent_said=[], prev_verdict=prev, now=11.0,
                                 weak_stops=False) == "backchannel"


def test_real_words_and_echo_are_unchanged_by_weak_stops():
    said = ["Fruit bats are big bats that mostly live in warm places."]
    for weak in (True, False):
        assert classify_interruption("big bats that mostly live", dur_s=1.2, recent_said=said,
                                     prev_verdict=None, now=0.0, weak_stops=weak) == "echo"
        assert classify_interruption("tell me about penguins instead", dur_s=1.5, recent_said=said,
                                     prev_verdict=None, now=0.0, weak_stops=weak) == "real"
    assert not has_strong_stop("what about tomorrow")
