"""Voice activity + end-of-turn detection (Silero VAD + Smart Turn v3.2, CPU).

Three layers, so the interesting logic is testable without a model:

* :class:`Endpointer` — a pure state machine over (frame, speech-probability)
  pairs. It decides when speech started, when a turn ended and when a blip was
  only noise. No numpy-free promise, but no onnxruntime either.
* :class:`SileroVAD` / :class:`SmartTurn` — thin onnxruntime wrappers that
  turn audio into the probabilities the state machine consumes.
* :class:`TurnDetector` — glue: raw PCM16 bytes in, events out.

Endpointing, in words. A frame is 512 samples (32 ms at 16 kHz). Speech has to
last ``min_speech_s`` before it counts (a door slam or a turn signal click is
shorter). Once someone is talking, a short pause (``turn_probe_silence_s``)
asks Smart Turn "does that sound finished?" — it hears intonation, so "yes."
ends quickly and "I was thinking that…" does not. If Smart Turn says
unfinished, the turn still ends after ``turn_hard_silence_s`` of silence. With
no Smart Turn model, a plain ``no_smart_turn_silence_s`` silence ends it.

Probe events (``EndpointerConfig.probe_events``, SPEC §4.1/§4.3). The v3
session does not want the state machine to call Smart Turn itself: at a pause
it runs Smart Turn AND the streaming transcript in parallel, then asks the
``TurnPolicy`` (``rt/endpoint.py``) what the two mean — end, hold, reprobe or
listen. In that mode the endpointer emits a :class:`Pause` event at every probe
mark (the first at ``turn_probe_silence_s``, later ones where the session asks
with ``TurnDetector.request_reprobe``), a :class:`Resumed` event when speech
returns after a probed pause (so a decision still in flight is stale), and
ends the turn by itself only on the hard silence or the maximum length; the
session ends it from a decision with ``TurnDetector.end_turn``. Every frame of
the utterance (pre-roll included) is also handed to ``TurnDetector.frame_sink``
with its speech probability — that is how the streaming STT gets its audio.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _log

log = _log.get()

SAMPLE_RATE = 16000
FRAME = 512                       # Silero's native window at 16 kHz
FRAME_S = FRAME / SAMPLE_RATE     # 0.032 s


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #

@dataclass
class SpeechStart:
    """Real speech began (after the min-speech debounce). Used for barge-in.
    ``pos``/``start``: uplink offsets, see :class:`TurnDetector` (None without them)."""
    pos: int | None = None        # just past the frame that confirmed the speech
    start: int | None = None      # the utterance's first frame (pre-roll included)


@dataclass
class TurnEnd:
    audio: np.ndarray             # float32 mono 16 kHz, the whole utterance
    reason: str                   # smart_turn | silence | max_length | a policy row's reason
    turn_prob: float | None = None
    silence_s: float = 0.0        # trailing silence at the end (sample offsets, metrics)
    pos: int | None = None        # uplink offset just past the last frame (TurnDetector)


@dataclass
class Discarded:
    """Speech-like noise that never reached ``min_speech_s``."""
    seconds: float
    pos: int | None = None


@dataclass
class Pause:
    """A pause inside an utterance reached a probe mark (``probe_events`` mode):
    the session runs Smart Turn on ``audio`` (the utterance so far, silence
    included) and the streaming transcript, and decides. ``silence_s`` is the
    pause so far, ``speech_s`` the speech heard in this utterance."""
    audio: np.ndarray
    silence_s: float
    speech_s: float
    pos: int | None = None


@dataclass
class Resumed:
    """Speech returned after a pause that was already probed: whatever the
    session decided (or is still deciding) about that pause is stale."""
    pause_s: float
    pos: int | None = None


# --------------------------------------------------------------------------- #
# The state machine
# --------------------------------------------------------------------------- #

@dataclass
class EndpointerConfig:
    threshold: float = 0.5
    min_speech_s: float = 0.25
    turn_probe_silence_s: float = 0.25
    turn_hard_silence_s: float = 1.4
    no_smart_turn_silence_s: float = 0.8
    max_utterance_s: float = 30.0
    smart_turn_threshold: float = 0.5
    pre_roll_s: float = 0.3       # audio kept from before speech was detected
    #: v3 session mode: emit Pause/Resumed events at the probe marks instead of
    #: calling Smart Turn here; only the hard silence / max length end a turn
    #: from inside (the session ends it from a decision).
    probe_events: bool = False


@dataclass
class Endpointer:
    cfg: EndpointerConfig = field(default_factory=EndpointerConfig)
    #: audio(float32) -> P(turn complete). None = no Smart Turn model.
    turn_fn: Callable[[np.ndarray], float] | None = None
    #: Every utterance frame (pre-roll included) with its speech probability:
    #: the streaming STT's feed (``TurnDetector.frame_sink`` sets it). A ``None``
    #: frame marks the END of an utterance (``_finish``): what follows is the next one.
    frame_sink: Callable[[np.ndarray | None, float], None] | None = None

    def __post_init__(self) -> None:
        self._pre: deque[np.ndarray] = deque(
            maxlen=max(1, round(self.cfg.pre_roll_s / FRAME_S)))
        self.reset()

    def reset(self) -> None:
        self._frames: list[np.ndarray] = []
        self._in_utterance = False   # some speech seen, collecting
        self._confirmed = False      # passed min_speech (SpeechStart sent)
        self._speech_frames = 0
        self._silence_frames = 0
        self._probed = False         # Smart Turn already asked this pause
        self._next_mark: float | None = None   # probe_events: silence (s) at which the next Pause fires
        self._fired_mark: float | None = None  # probe_events: the mark that fired in this pause
        self._pre.clear()

    @property
    def active(self) -> bool:
        return self._in_utterance

    @property
    def confirmed(self) -> bool:
        return self._confirmed

    @property
    def silence_s(self) -> float:
        return self._silence_frames * FRAME_S

    @property
    def speech_s(self) -> float:
        return self._speech_frames * FRAME_S

    def request_reprobe(self, at_silence_s: float) -> None:
        """``probe_events``: ask for another :class:`Pause` once the current pause
        reaches ``at_silence_s`` (or at the next frame when it already has)."""
        self._next_mark = max(0.0, float(at_silence_s))

    def _sink(self, frame: np.ndarray | None, prob: float) -> None:
        fn = self.frame_sink
        if fn is None:
            return
        try:
            fn(frame, prob)
        except Exception:
            log.exception("voice: frame sink failed")
            self.frame_sink = None

    def process(self, frame: np.ndarray, prob: float) -> list:
        """Feed one FRAME of float32 audio with its speech probability."""
        c = self.cfg
        events: list = []
        # Hysteresis: once talking, a frame has to drop clearly below the
        # threshold to count as silence, so breathy word endings don't split.
        is_speech = prob >= (c.threshold - 0.15 if self._in_utterance else c.threshold)

        if not self._in_utterance:
            if is_speech:
                self._in_utterance = True
                self._frames = [*self._pre, frame]
                self._speech_frames = 1
                self._silence_frames = 0
                self._next_mark = c.turn_probe_silence_s
                self._fired_mark = None
                if self.frame_sink is not None:
                    for fr in self._pre:
                        self._sink(fr, 0.0)
                    self._sink(frame, prob)
                self._pre.clear()
            else:
                self._pre.append(frame)
            return events

        self._frames.append(frame)
        self._sink(frame, prob)
        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
            self._probed = False
            if c.probe_events and self._fired_mark is not None:
                events.append(Resumed(self._fired_mark))
            if c.probe_events:
                self._next_mark = c.turn_probe_silence_s
                self._fired_mark = None
            if not self._confirmed and self._speech_frames * FRAME_S >= c.min_speech_s:
                self._confirmed = True
                events.append(SpeechStart())
        else:
            self._silence_frames += 1

        silence_s = self._silence_frames * FRAME_S
        length_s = len(self._frames) * FRAME_S

        if not self._confirmed:
            # A blip that went quiet before it ever counted as speech.
            if silence_s >= c.turn_probe_silence_s:
                events.append(Discarded(self._speech_frames * FRAME_S))
                self.reset()
            return events

        if length_s >= c.max_utterance_s:
            events.append(self._finish("max_length"))
            return events

        if c.probe_events:
            if silence_s >= c.turn_hard_silence_s:
                events.append(self._finish("silence"))
            elif (not is_speech and self._next_mark is not None and silence_s >= self._next_mark
                  and self._fired_mark != self._next_mark):
                self._fired_mark = self._next_mark
                self._next_mark = None
                events.append(Pause(self._audio(), silence_s, self._speech_frames * FRAME_S))
            return events

        if self.turn_fn is None:
            if silence_s >= c.no_smart_turn_silence_s:
                events.append(self._finish("silence"))
            return events

        if silence_s >= c.turn_hard_silence_s:
            events.append(self._finish("silence"))
        elif silence_s >= c.turn_probe_silence_s and not self._probed:
            self._probed = True
            try:
                p = float(self.turn_fn(self._audio()))
            except Exception:
                log.exception("smart turn failed; falling back to silence")
                p = None
            if p is not None and p >= c.smart_turn_threshold:
                events.append(self._finish("smart_turn", p))
        return events

    def finish(self, reason: str, prob: float | None = None) -> TurnEnd | None:
        """End the utterance now (a session decision in ``probe_events`` mode).
        None when no confirmed utterance is open."""
        if not self._in_utterance or not self._confirmed:
            return None
        return self._finish(reason, prob)

    def _audio(self) -> np.ndarray:
        return np.concatenate(self._frames) if self._frames else np.zeros(0, np.float32)

    def _finish(self, reason: str, prob: float | None = None) -> TurnEnd:
        # Trim the trailing silence (keep ~0.2 s) — STT does not need it.
        silence_s = self._silence_frames * FRAME_S
        keep_tail = max(0, self._silence_frames - round(0.2 / FRAME_S))
        frames = self._frames[: len(self._frames) - keep_tail] if keep_tail else self._frames
        audio = np.concatenate(frames) if frames else np.zeros(0, np.float32)
        self.reset()
        # The end mark, in order with the frames (the caller holds the detector lock): the
        # sink's next frames are the NEXT utterance's (review-T3s3 #2 — the session keeps
        # them out of the stream the ending turn is still reading).
        self._sink(None, -1.0)
        return TurnEnd(audio=audio, reason=reason, turn_prob=prob, silence_s=silence_s)


# --------------------------------------------------------------------------- #
# Model wrappers
# --------------------------------------------------------------------------- #

def _session(path: Path, threads: int, *, spinning: bool = False):
    """A bursty session (Silero, Smart Turn): ``session.intra_op.allow_spinning``
    off unless asked (``rt.stt_spinning``; lead-measured, SPEC §0.2: faster under
    load and no 0.5 core-second burn after every call). The one place these
    options come from is ``rt/stt_stream.ort_options`` (Amendment T4-1 (f))."""
    import onnxruntime as ort

    from .rt.stt_stream import ort_options
    so = ort_options(threads, spinning=spinning)
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


class SileroVAD:
    """Silero VAD v5/v6 ONNX, one 512-sample frame at a time, stateful."""

    CONTEXT = 64  # samples of the previous frame the v5 graph expects

    def __init__(self, path: Path, threads: int = 1, *, spinning: bool = False) -> None:
        self._sess = _session(path, threads, spinning=spinning)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), np.float32)
        self._context = np.zeros(self.CONTEXT, np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._context, frame.astype(np.float32)])[None, :]
        out, self._state = self._sess.run(
            None, {"input": x, "state": self._state, "sr": np.array(SAMPLE_RATE, np.int64)})
        self._context = frame[-self.CONTEXT:].astype(np.float32)
        return float(out[0][0])


class EnergyVAD:
    """Crude RMS gate. A TEST DOUBLE and a no-model fallback — not for the car."""

    def __init__(self, threshold_db: float = -35.0) -> None:
        self.threshold_db = threshold_db

    def reset(self) -> None:
        pass

    def __call__(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)) + 1e-12)
        return 1.0 if 20 * np.log10(rms) > self.threshold_db else 0.0


# ---- Whisper log-mel, numpy only (what Smart Turn's encoder was trained on) #

def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, np.float64)
    mel = f / (200.0 / 3)
    log_region = f >= 1000.0
    mel = np.where(log_region, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), mel)
    return mel


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, np.float64)
    f = m * (200.0 / 3)
    log_region = m >= 15.0
    return np.where(log_region, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), f)


def mel_filters(n_mels: int = 80, n_fft: int = 400, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Slaney-scale, slaney-normalised filterbank, shape (n_freqs, n_mels)."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, n_freqs)
    mel_pts = np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    diff = np.diff(hz_pts)
    slopes = hz_pts[None, :] - fft_freqs[:, None]           # (n_freqs, n_mels+2)
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    fb = np.maximum(0, np.minimum(down, up))
    enorm = 2.0 / (hz_pts[2: n_mels + 2] - hz_pts[:n_mels])
    return fb * enorm[None, :]


_MEL = None


def whisper_features(audio: np.ndarray, seconds: int = 8) -> np.ndarray:
    """(1, 80, seconds*100) float32 — WhisperFeatureExtractor(chunk_length=8)
    as Smart Turn calls it: keep the last 8 s, LEFT-pad with zeros, normalise
    to zero mean / unit variance, log-mel."""
    global _MEL
    if _MEL is None:
        _MEL = mel_filters()
    n = seconds * SAMPLE_RATE
    x = np.asarray(audio, np.float64)[-n:]
    if len(x) < n:
        x = np.pad(x, (n - len(x), 0))
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    n_fft, hop = 400, 160
    xp = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    window = np.hanning(n_fft + 1)[:-1]
    spec = np.fft.rfft(xp[idx] * window, axis=1)
    power = (np.abs(spec) ** 2)[:-1]                        # whisper drops the last frame
    mel = power @ _MEL                                        # (frames, 80)
    log_spec = np.log10(np.maximum(mel, 1e-10)).T
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)[None, :, :]


class SmartTurn:
    """Smart Turn v3.x ONNX: P(the speaker has finished)."""

    def __init__(self, path: Path, threads: int = 1, *, spinning: bool = False) -> None:
        self._sess = _session(path, threads, spinning=spinning)

    def __call__(self, audio: np.ndarray) -> float:
        out = self._sess.run(None, {"input_features": whisper_features(audio)})
        return float(np.asarray(out[0]).reshape(-1)[0])


# --------------------------------------------------------------------------- #
# Glue
# --------------------------------------------------------------------------- #

class TurnDetector:
    """PCM16 little-endian mono 16 kHz bytes in → events out. One per voice
    session. ``feed`` runs on the session's audio thread; the decision calls
    (``end_turn``, ``request_reprobe``, ``discard_utterance``, ``reset``) come
    from the event loop, so the state machine is behind one lock.

    Uplink positions (review-T3s4 #3): when ``feed`` is given ``end`` (the uplink
    sample offset, u32, just past the fed audio), every event it returns carries
    ``pos`` — the offset just past the 32 ms frame that produced it — and a
    :class:`SpeechStart` also carries ``start``, the offset of the utterance's first
    frame (pre-roll included). :attr:`pos` is the offset past the last frame
    processed, and a :class:`TurnEnd` from :meth:`end_turn` carries it. Without
    ``end`` nothing is annotated (``pos`` stays None)."""

    def __init__(self, vad: Callable[[np.ndarray], float],
                 endpointer: Endpointer) -> None:
        self.vad = vad
        self.ep = endpointer
        self._buf = np.zeros(0, np.float32)
        self.last_prob = 0.0
        self._lock = threading.Lock()
        #: The uplink offset just past the last processed frame (None until a feed with ``end``).
        self.pos: int | None = None
        #: Called with ``(frame_db, prob)`` for EVERY frame (1.5.0: the session's level
        #: meter follows the room's floor from it); None = nobody listens.
        self.frame_observer: Callable[[float, float], None] | None = None

    @property
    def frame_sink(self) -> Callable[[np.ndarray, float], None] | None:
        return self.ep.frame_sink

    @frame_sink.setter
    def frame_sink(self, fn: Callable[[np.ndarray, float], None] | None) -> None:
        self.ep.frame_sink = fn

    @property
    def active(self) -> bool:
        return self.ep.active

    def reset(self) -> None:
        with self._lock:
            self._buf = np.zeros(0, np.float32)
            self.ep.reset()
            if hasattr(self.vad, "reset"):
                self.vad.reset()

    def discard_utterance(self) -> None:
        """Drop the open utterance (an echo, a backchannel, a "stop"): the next
        speech starts a new one. The VAD keeps its state."""
        with self._lock:
            self.ep.reset()

    def end_turn(self, reason: str, prob: float | None = None) -> TurnEnd | None:
        """``probe_events``: end the open utterance from a policy decision. The
        :class:`TurnEnd` carries ``pos`` (see the class docstring) when known."""
        with self._lock:
            te = self.ep.finish(reason, prob)
            if te is not None:
                te.pos = self.pos
            return te

    def request_reprobe(self, at_silence_s: float) -> None:
        with self._lock:
            self.ep.request_reprobe(at_silence_s)

    def feed(self, pcm16: bytes, end: int | None = None) -> list:
        """Run the state machine over ``pcm16``; ``end``: the uplink offset just past it
        (annotates the events with their positions, see the class docstring)."""
        if len(pcm16) % 2:
            pcm16 = pcm16[:-1]
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        events: list = []
        with self._lock:
            self._buf = np.concatenate([self._buf, samples])
            while len(self._buf) >= FRAME:
                frame, self._buf = self._buf[:FRAME], self._buf[FRAME:]
                self.last_prob = self.vad(frame)
                obs = self.frame_observer
                if obs is not None:
                    try:
                        obs(float(10.0 * np.log10(float(np.mean(frame * frame)) + 1e-12)), self.last_prob)
                    except Exception:
                        log.exception("voice: frame observer failed")
                        self.frame_observer = None
                got = self.ep.process(frame, self.last_prob)
                if end is not None:
                    # Everything still in `_buf` came after this frame.
                    at = (int(end) - len(self._buf)) & 0xFFFFFFFF
                    self.pos = at
                    for ev in got:
                        ev.pos = at
                        if isinstance(ev, SpeechStart):
                            ev.start = (at - len(self.ep._frames) * FRAME) & 0xFFFFFFFF
                events.extend(got)
        return events


def _rt_spinning(settings) -> bool:
    rt = getattr(settings, "rt", None)
    return bool(getattr(rt, "stt_spinning", False))


def build_detector(settings) -> TurnDetector:
    """Production detector from VoiceSettings (Silero + Smart Turn if present)."""
    cfg = EndpointerConfig(
        threshold=settings.vad_threshold,
        min_speech_s=settings.min_speech_s,
        turn_probe_silence_s=settings.turn_probe_silence_s,
        turn_hard_silence_s=settings.turn_hard_silence_s,
        no_smart_turn_silence_s=settings.no_smart_turn_silence_s,
        max_utterance_s=settings.max_utterance_s,
        smart_turn_threshold=settings.smart_turn_threshold,
    )
    turn_fn = _shared_smart_turn(settings)
    return TurnDetector(SileroVAD(settings.silero_path, spinning=_rt_spinning(settings)),
                        Endpointer(cfg, turn_fn))


_SMART_TURN: dict[str, SmartTurn] = {}
#: The start-up prewarm and a first session can both ask at once; one load.
_SMART_TURN_LOCK = threading.Lock()


def _shared_smart_turn(settings) -> SmartTurn | None:
    p = settings.smart_turn_path
    if not p.exists():
        return None
    key = str(p)
    with _SMART_TURN_LOCK:
        if key not in _SMART_TURN:
            _SMART_TURN[key] = SmartTurn(p, threads=2, spinning=_rt_spinning(settings))
        return _SMART_TURN[key]
