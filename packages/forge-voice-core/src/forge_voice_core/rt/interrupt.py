"""Interrupt sensitivity: how sure the session must be before speech over a reply
pauses or cuts it (1.5.0).

Up to 1.4.0 every interruption rule ran at one setting, tuned for a quiet car: 0.25 s of
voice-like sound over the reply paused it, a lone "what" or "thank you" cut it, three
words cut it and the run. In a family room that is far too eager: other people talking,
a child, the TV and the bot's own echo kept pausing and cutting replies (the 2026-10-17
journal: stop words "What?", "Thank you.", "Okay, wait." from the room, pauses resumed
as echo or noise, replies cut a few words in).

Three profiles (a device setting, ``start.interrupt_sensitivity``):

* ``high`` — exactly the 1.4.0 behaviour;
* ``normal`` (the default) — the reply pauses only after 0.45 s of sustained speech that
  is loud enough to be the listener (see the level gate), a cut needs four words of the
  listener's own (three before), a mid-speech stop needs a clear stop word ("stop",
  "wait", "hold on", "pause", "enough"…); "what", "sorry", "no", "thanks" and the like
  stop a reply only as a whole, near-field utterance;
* ``low`` — 0.7 s, five words, a stricter level gate, clear stop words only.

The level gate is adaptive: :class:`LevelMeter` follows the room's ambient floor (the
non-speech frames the server receives, our own echo residue included) and the
listener's own voice level (the speech of turns they made while nothing played). Speech
over a reply counts only when it is ``floor_margin_db`` over the floor AND no more than
``voice_margin_db`` under the listener's usual level: someone across the room is quieter
at the mic than the person holding it. Unknown levels never block (a fresh session, a
test double without audio). Clear stop words always work, whatever the level.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

SENSITIVITIES = ("low", "normal", "high")
DEFAULT_SENSITIVITY = "normal"

#: A 512-sample frame (32 ms at 16 kHz) quieter than this is digital silence (the
#: uplink's zero-fill, a muted mic): it says nothing about the room.
SILENT_DB = -85.0
#: Frames under this speech probability are the room, not a talker.
AMBIENT_PROB = 0.3
#: The ambient window: about 10 s of received non-speech frames.
AMBIENT_FRAMES = 320
#: The floor is this percentile of the ambient window (robust to a clatter).
AMBIENT_PERCENTILE = 20.0
#: ... and needs at least this many frames (≈ 1 s) before it is trusted.
AMBIENT_MIN_FRAMES = 30
#: An utterance's level is this percentile of its frames' energy (its loud part: the
#: pre-roll and the gaps between words sit below it).
SPEECH_PERCENTILE = 80.0
#: The listener's voice level is the median of their last few turns.
VOICE_KEEP = 8
VOICE_MIN_TURNS = 2
FRAME = 512


@dataclass(frozen=True)
class InterruptProfile:
    name: str
    #: Sustained speech before the reply pauses (None = the configured barge_in_speech_s).
    duck_speech_s: float | None
    #: Words a mid-speech partial needs to cut the reply and the run.
    tick_min_words: int
    #: The level gate (None = off).
    floor_margin_db: float | None
    voice_margin_db: float | None
    #: Ambiguous stop words ("what", "sorry", "no", "thanks", "okay okay", two
    #: backchannels in 3 s) are stops at any level and mid-speech.
    weak_stops: bool
    #: A mid-speech partial with any command word cuts at once (else clear stop words only).
    early_any_command: bool
    #: The phone's own instant duck on a mic energy edge (client).
    optimistic_duck: bool


PROFILES: dict[str, InterruptProfile] = {
    "high": InterruptProfile("high", None, 3, None, None, True, True, True),
    "normal": InterruptProfile("normal", 0.45, 4, 10.0, 10.0, False, False, False),
    "low": InterruptProfile("low", 0.7, 5, 14.0, 6.0, False, False, False),
}


def sensitivity_value(v, default: str = DEFAULT_SENSITIVITY) -> str:
    return v if isinstance(v, str) and v in PROFILES else default


def profile(name) -> InterruptProfile:
    return PROFILES[sensitivity_value(name)]


def frame_db(frame: np.ndarray) -> float:
    """Energy of float32 samples (−1..1) in dBFS."""
    x = np.asarray(frame, dtype=np.float32)
    if not x.size:
        return -120.0
    return float(10.0 * np.log10(float(np.mean(x * x)) + 1e-12))


def speech_level_db(audio) -> float | None:
    """The loud part of an utterance in dBFS (80th percentile of its 32 ms frames, digital
    silence left out); None for less than three frames of sound."""
    if audio is None:
        return None
    x = np.asarray(audio, dtype=np.float32).ravel()
    n = len(x) // FRAME
    if n < 3:
        return None
    sq = (x[: n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1)
    db = 10.0 * np.log10(sq + 1e-12)
    db = db[db > SILENT_DB]
    if len(db) < 3:
        return None
    return float(np.percentile(db, SPEECH_PERCENTILE))


class LevelMeter:
    """The room's floor and the listener's voice level, for the level gate. ``observe``
    runs on the audio thread (every frame the detector sees), the rest on the event loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ambient: deque[float] = deque(maxlen=AMBIENT_FRAMES)
        self._voice: deque[float] = deque(maxlen=VOICE_KEEP)

    def observe(self, db: float, prob: float) -> None:
        if db <= SILENT_DB or prob >= AMBIENT_PROB:
            return
        with self._lock:
            self._ambient.append(float(db))

    def note_voice(self, level_db: float | None) -> None:
        """A turn the listener made while nothing played: their own voice level."""
        if level_db is None:
            return
        with self._lock:
            self._voice.append(float(level_db))

    @property
    def floor_db(self) -> float | None:
        with self._lock:
            if len(self._ambient) < AMBIENT_MIN_FRAMES:
                return None
            vals = list(self._ambient)
        return float(np.percentile(vals, AMBIENT_PERCENTILE))

    @property
    def voice_db(self) -> float | None:
        with self._lock:
            if len(self._voice) < VOICE_MIN_TURNS:
                return None
            vals = list(self._voice)
        return float(np.median(vals))

    def judge(self, level_db: float | None, prof: InterruptProfile) -> tuple[bool, str]:
        """Is speech at ``level_db`` the listener (True) or the room (False, with why)?"""
        if level_db is None or (prof.floor_margin_db is None and prof.voice_margin_db is None):
            return True, ""
        floor, voice = self.floor_db, self.voice_db
        if prof.floor_margin_db is not None and floor is not None \
                and level_db < floor + prof.floor_margin_db:
            return False, f"{level_db:.0f} dB, room floor {floor:.0f} dB"
        if prof.voice_margin_db is not None and voice is not None \
                and level_db < voice - prof.voice_margin_db:
            return False, f"{level_db:.0f} dB, your voice {voice:.0f} dB"
        return True, ""

    def snapshot(self) -> dict:
        f, v = self.floor_db, self.voice_db
        return {"floor_db": None if f is None else round(f, 1),
                "voice_db": None if v is None else round(v, 1)}
