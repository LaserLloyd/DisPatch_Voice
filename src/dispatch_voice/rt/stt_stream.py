"""Streaming STT (SPEC §4.2, C-§B.3): Parakeet as a tail re-transcriber, Moonshine optional.

Parakeet TDT is a whole-utterance model, but its partials are stable: in 44
measured steps the leading words matched the final transcript in every step
except the last one or two. ``ParakeetTailSTT`` turns that into a streaming
recogniser without a new model:

* the utterance is closed segments (text fixed, ``committed``) plus one open
  tail; ``at_pause()`` transcribes only the tail (430–1250 ms measured; 253–
  577 ms for the 2–3.5 s tails a real pause leaves at 2 GHz);
* once the tail passes ``commit_after_s`` (5 s) it is first closed at the last
  silence ≥ 300 ms that is ≥ 1 s old, so every call stays bounded on a
  monologue and the committed text never changes; when no such silence exists
  (road noise holding Silero up, a speaker chaining clauses with < 0.3 s
  pauses) a tail longer than ``tail_max_s`` (10 s) is closed anyway at its
  quietest frame that is ≥ 1 s old, so no call ever transcribes more than
  ≈ 10 s and the open tail cannot grow without bound (review T4 #3);
* ``tick()`` runs at most once per second and only when no pause call is in
  flight (live captions never starve the pause path);
* ``final()`` returns committed + the last partial when < 0.3 s of audio
  arrived since that partial (the common case: the turn ends at the pause
  that was just transcribed) or when every frame since it was silence;
  otherwise one more tail call. It waits for an in-flight ``at_pause()``
  first — at most ``timeout_s`` when given, answering with the best text it
  has past that (the session bounds the total at ``hold_max_s``, A8);
* ``reset()`` from any thread is safe at any moment: every call captures the
  buffer generation when it takes its audio and drops its result (committed
  join, partial, timings) when a ``reset()`` happened meanwhile, so a
  cancelled turn's words never become the next turn's transcript (review T4
  #1);
* ``prime()`` (optional, the session's call) starts the tail call at the first
  ≈ 0.15 s of silence instead of at the 0.25 s probe, in a worker thread; the
  probe's ``at_pause()`` then reuses it when only silence arrived in between.
  The transcript is the same with 0.10 and 0.25 s of trailing silence on every
  fixture clip (measured, ``probe_early_tail.json``), so the reuse is exact,
  and the decision lands ≈ 0.1 s earlier on every turn. A false pause (speech
  resumes before the probe) wastes that call — 0–1 per clip at 0.15 s, 0–4 at
  0.10 s (measured) — and the next ``at_pause()`` waits for it, which bounds
  the worst case at one call;
* ``stable_words`` = leading agreement with the previous partial minus two.

Threading: ``push()`` is called on the event loop with every 32 ms frame;
``at_pause``/``tick``/``final`` are blocking and run in the session's STT
thread. The recogniser itself (``onnx_asr``) runs one transcription at a
time behind its own lock.

Sessions (:func:`ort_options`) are created with
``session.intra_op.allow_spinning = "0"`` (lead-measured, SPEC §0.2: −9…−23 %
under load, and the 46–59 % of a core burnt after every call → 0 %). The
decoder keeps spinning; that is ``engine.py``'s business.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

log = logging.getLogger("dispatch.voice")

SAMPLE_RATE = 16000
FRAME = 512
FRAME_S = FRAME / SAMPLE_RATE
COMMIT_AFTER_S = 5.0        # close the tail once it is longer than this
COMMIT_SILENCE_S = 0.3      # … at a silence at least this long
COMMIT_MIN_AGE_S = 1.0      # … that ended at least this long ago
TAIL_MAX_S = 10.0           # hard cap: close the tail at its quietest old frame past this
SILENCE_PROB = 0.35         # Silero probability below which a frame is silence
TICK_INTERVAL_S = 1.0
FINAL_FRESH_S = 0.3         # final() reuses the last partial below this much new audio
FINAL_DEADLINE_S = 0.3
CHURN_WORDS = 2             # the measured churn never exceeded the last two words
MIN_AUDIO_SAMPLES = 1600    # < 0.1 s: nothing to hear

_WORD = re.compile(r"[A-Za-z0-9']+")
_TERMINAL = re.compile(r"[.?!]+[\"'”’)\]]*\s*$")


@dataclass
class Partial:
    committed: str        # never changes again
    tail: str             # open segment, may change
    stable_words: int     # words of `tail` agreed across the last two calls
    punct_final: bool     # tail ends with . ? !
    t_audio: float        # seconds of audio covered
    cost_ms: int

    @property
    def text(self) -> str:
        return f"{self.committed} {self.tail}".strip()


@runtime_checkable
class StreamingSTT(Protocol):
    def push(self, pcm16: np.ndarray, *, vad_prob: float) -> None: ...
    def at_pause(self) -> Partial: ...
    def tick(self) -> Partial | None: ...
    def final(self) -> str: ...
    def reset(self) -> None: ...


def ort_options(threads: int, *, spinning: bool = False, log_severity: int = 3):
    """SessionOptions for the bursty sessions (STT, Smart Turn): spinning OFF by default."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, int(threads))
    so.inter_op_num_threads = 1
    so.log_severity_level = log_severity
    so.add_session_config_entry("session.intra_op.allow_spinning", "1" if spinning else "0")
    return so


def words_of(s: str) -> list[str]:
    return _WORD.findall((s or "").lower().replace("’", "'"))


def _agree(prev: list[str], cur: list[str]) -> int:
    n = 0
    for a, b in zip(prev, cur, strict=False):     # the shorter list bounds the agreement
        if a != b:
            break
        n += 1
    return n


def _join(a: str, b: str) -> str:
    a, b = (a or "").strip(), (b or "").strip()
    if not a:
        return b
    if not b:
        return a
    return f"{a} {b}"


def _to_float(pcm16: np.ndarray) -> np.ndarray:
    a = np.asarray(pcm16)
    if a.dtype == np.int16:
        return a.astype(np.float32) / 32768.0
    if a.dtype != np.float32:
        return a.astype(np.float32)
    return a


class ParakeetRecognizer:
    """onnx-asr Parakeet TDT 0.6B v2 (int8) with the bursty-session options.

    ``shared_engine`` for :func:`build_streaming_stt` may be one of these, or
    the host's ``ParakeetSTT`` (anything with ``transcribe(float32) -> str``).
    """

    def __init__(self, model_dir: Path, threads: int = 6, *, spinning: bool = False) -> None:
        import onnx_asr
        self._model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v2", str(model_dir), quantization="int8",
            sess_options=ort_options(threads, spinning=spinning), providers=["CPUExecutionProvider"])
        self._lock = threading.Lock()
        self.threads = threads
        self.spinning = spinning

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size < MIN_AUDIO_SAMPLES:
            return ""
        with self._lock:
            text = self._model.recognize(np.ascontiguousarray(audio, dtype=np.float32), sample_rate=SAMPLE_RATE)
        return (text or "").strip()


class ParakeetTailSTT:
    """Closed segments + one open tail; see the module docstring."""

    def __init__(self, recognize: Callable[[np.ndarray], str], *,
                 commit_after_s: float = COMMIT_AFTER_S, commit_silence_s: float = COMMIT_SILENCE_S,
                 commit_min_age_s: float = COMMIT_MIN_AGE_S, tick_interval_s: float = TICK_INTERVAL_S,
                 final_fresh_s: float = FINAL_FRESH_S, tail_max_s: float = TAIL_MAX_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._recognize = recognize
        self.commit_after_s = commit_after_s
        self.commit_silence_s = commit_silence_s
        self.commit_min_age_s = commit_min_age_s
        self.tail_max_s = max(commit_after_s, float(tail_max_s))
        self.tick_interval_s = tick_interval_s
        self.final_fresh_s = final_fresh_s
        self._clock = clock
        self._lock = threading.Lock()          # guards the buffers
        self._call_lock = threading.Lock()     # one pause/tick/final call at a time
        self._inflight = threading.Event()     # set while at_pause()/final()/prime() transcribes
        self.calls = 0                          # recogniser calls (tests, metrics)
        self.primed_hits = 0                    # at_pause()/final() answered by a primed call
        self._generation = 0
        self._prime_thread: threading.Thread | None = None
        self.reset()

    # ------------------------------------------------------------------ buffers
    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self._frames: list[np.ndarray] = []   # open tail, 512-sample frames
            self._probs: list[float] = []
            self._committed = ""
            self._tail_text = ""
            self._prev_tail_words: list[str] = []
            self._t_total = 0.0                  # audio pushed since reset
            self._t_closed = 0.0                 # audio in closed segments
            self._t_partial = -1.0               # t_total covered by the last partial
            self._t_speech = 0.0                 # t_total at the end of the last speech frame
            self._last_tick = -1e9
            self._last_partial: Partial | None = None

    def push(self, pcm16: np.ndarray, *, vad_prob: float) -> None:
        a = _to_float(pcm16)
        if a.size == 0:
            return
        with self._lock:
            # Split into 512-sample frames (the session already pushes whole frames;
            # anything else is re-framed so the silence map stays per frame).
            n = len(a)
            speech = float(vad_prob) >= SILENCE_PROB
            for i in range(0, n, FRAME):
                fr = a[i:i + FRAME]
                self._frames.append(fr)
                self._probs.append(float(vad_prob))
                self._t_total += len(fr) / SAMPLE_RATE
            if speech:
                self._t_speech = self._t_total

    @property
    def t_audio(self) -> float:
        return self._t_total

    @property
    def committed(self) -> str:
        return self._committed

    @property
    def inflight(self) -> bool:
        return self._inflight.is_set()

    # ------------------------------------------------------------------ helpers
    def _tail_seconds(self) -> float:
        return self._t_total - self._t_closed

    def _commit_boundary(self) -> int | None:
        """Frame index to close the tail at: the END of the last silence run ≥
        commit_silence_s that lies ≥ commit_min_age_s before the newest audio (so the
        new tail starts where speech resumed). None = keep the tail."""
        probs = self._probs
        n = len(probs)
        need = max(1, round(self.commit_silence_s / FRAME_S))
        age = max(0, round(self.commit_min_age_s / FRAME_S))
        limit = n - age                       # frames at or beyond this are too young
        best_end = None
        run_start = None
        for i in range(limit):
            if probs[i] < SILENCE_PROB:
                if run_start is None:
                    run_start = i
                if i - run_start + 1 >= need:
                    best_end = i + 1          # cut after this silent frame
            else:
                run_start = None
        # Never close an (almost) empty segment.
        if best_end is None or best_end < int(0.5 / FRAME_S):
            return None
        return best_end

    def _forced_boundary(self, limit: int | None = None) -> int | None:
        """The cap's cut when no qualifying silence exists: after the quietest frame
        (lowest Silero probability; the latest one on a tie, so the segment closed is
        the longest the cap allows) that is ≥ commit_min_age_s old, ≥ 0.5 s into the
        tail and, with ``limit``, before that frame index."""
        probs = self._probs
        age = max(0, round(self.commit_min_age_s / FRAME_S))
        lo = int(0.5 / FRAME_S)
        hi = len(probs) - age
        if limit is not None:
            hi = min(hi, limit)
        if hi <= lo:
            return None
        best, best_p = None, None
        for i in range(lo, hi):
            if best_p is None or probs[i] <= best_p:
                best, best_p = i, probs[i]
        return None if best is None else best + 1

    def _close_tail_if_long(self) -> list[np.ndarray]:
        """With the buffer lock HELD: split off closed segments when the tail is long.
        Returns the audio of the segments to transcribe into ``committed``, in order.
        Normally one segment, cut at the last qualifying silence; past ``tail_max_s``
        (no such silence, or a long stretch left unprobed) the cuts are forced at the
        quietest old frames so that every segment AND the tail left behind are at most
        ``tail_max_s`` long — no recogniser call ever sees more than that."""
        segs: list[np.ndarray] = []
        max_frames = max(1, round(self.tail_max_s / FRAME_S))
        while self._tail_seconds() > self.commit_after_s:
            b = self._commit_boundary()
            if b is None and self._tail_seconds() > self.tail_max_s:
                b = self._forced_boundary()
            if b is None:
                break
            if b > max_frames:                       # a segment longer than the cap: split it
                b = self._forced_boundary(limit=max_frames) or max_frames
            seg = np.concatenate(self._frames[:b]) if b else np.zeros(0, np.float32)
            del self._frames[:b]
            del self._probs[:b]
            self._t_closed += len(seg) / SAMPLE_RATE
            segs.append(seg)
            if len(segs) >= 8:                       # ≥ 80 s unprobed: enough for one call
                break
        return segs

    def _take_locked(self, *, close: bool) -> tuple[int, list[np.ndarray], np.ndarray, float]:
        """With the buffer lock HELD: (generation, closing segments, tail audio, t_covered)
        — one consistent snapshot of what a call is about to transcribe."""
        closing = self._close_tail_if_long() if close else []
        audio = np.concatenate(self._frames) if self._frames else np.zeros(0, np.float32)
        return self._generation, closing, audio, self._t_total

    def _snapshot(self) -> np.ndarray:
        with self._lock:
            return np.concatenate(self._frames) if self._frames else np.zeros(0, np.float32)

    def _transcribe(self, audio: np.ndarray) -> str:
        self.calls += 1
        try:
            return (self._recognize(audio) or "").strip()
        except Exception:  # fail soft: an empty partial, never a dead session
            log.exception("voice: streaming STT call failed")
            return ""

    def _partial_from_locked(self, tail_text: str, t_covered: float, cost_ms: int) -> Partial:
        """With the buffer lock HELD and the generation already checked."""
        cur = words_of(tail_text)
        stable = max(0, _agree(self._prev_tail_words, cur) - CHURN_WORDS)
        self._prev_tail_words = cur
        self._tail_text = tail_text
        p = Partial(committed=self._committed, tail=tail_text, stable_words=stable,
                    punct_final=bool(_TERMINAL.search(tail_text)), t_audio=round(t_covered, 3),
                    cost_ms=cost_ms)
        self._last_partial = p
        self._t_partial = t_covered
        return p

    def _reusable_locked(self) -> Partial | None:
        """With the buffer lock HELD: the last partial, when nothing but silence was
        pushed after the audio it covered (a longer silence leaves the words unchanged —
        measured on every fixture clip between 0.10 and 0.25 s)."""
        p = self._last_partial
        if p is not None and self._t_partial >= 0 and self._t_speech <= self._t_partial + 1e-6:
            return p
        return None

    # ------------------------------------------------------------------ calls
    def _tail_call(self, *, close: bool = True) -> Partial | None:
        """One recogniser pass over the open tail (closing a long tail first), with
        ``_call_lock`` held by the caller. The audio is taken and the result stored
        under the buffer lock; a ``reset()`` in between makes the result nobody's and
        None comes back (the caller answers with the empty/new state). None also when
        there is nothing to hear."""
        t0 = self._clock()
        with self._lock:
            gen, closing, audio, t_covered = self._take_locked(close=close)
        if not closing and audio.size < MIN_AUDIO_SAMPLES:
            return None
        texts = [self._transcribe(seg) for seg in closing]
        tail_text = self._transcribe(audio) if audio.size >= MIN_AUDIO_SAMPLES else ""
        cost = round((self._clock() - t0) * 1000)
        with self._lock:
            if gen != self._generation:
                return None                   # reset() ran meanwhile: discard everything
            for text in texts:
                self._committed = _join(self._committed, text)
                self._prev_tail_words = []    # the tail's words moved into committed
            return self._partial_from_locked(tail_text, t_covered, cost)

    def _empty_partial(self) -> Partial:
        with self._lock:
            return Partial(self._committed, self._tail_text, 0, False, round(self._t_total, 3), 0)

    def at_pause(self) -> Partial:
        with self._call_lock:                   # waits for a primed call in flight
            self._inflight.set()
            try:
                with self._lock:
                    p = self._reusable_locked()
                    t_now = self._t_total
                if p is not None:
                    self.primed_hits += 1
                    return Partial(p.committed, p.tail, p.stable_words, p.punct_final, round(t_now, 3), p.cost_ms)
                return self._tail_call() or self._empty_partial()
            finally:
                self._inflight.clear()

    def prime(self) -> bool:
        """Start the tail transcription now, in a worker thread, so the probe's
        ``at_pause()`` finds it done (or nearly). Returns True when a call was started;
        False when one is already in flight, when the last partial already covers every
        speech frame, or when there is nothing to hear yet. Never blocks."""
        with self._lock:
            if self._inflight.is_set() or self._reusable_locked() is not None:
                return False
            if (self._t_total - self._t_closed) * SAMPLE_RATE < MIN_AUDIO_SAMPLES:
                return False
            if not self._call_lock.acquire(blocking=False):
                return False
            self._inflight.set()

        def run() -> None:
            try:
                self._tail_call()
            except Exception:
                log.exception("voice: primed STT call failed")
            finally:
                self._inflight.clear()
                self._call_lock.release()

        self._prime_thread = threading.Thread(target=run, name="voice-stt-prime", daemon=True)
        self._prime_thread.start()
        return True

    def tick(self) -> Partial | None:
        if self._inflight.is_set():
            return None
        now = self._clock()
        if now - self._last_tick < self.tick_interval_s:
            return None
        if not self._call_lock.acquire(blocking=False):
            return None
        try:
            self._last_tick = now
            return self._tail_call()          # closes a long tail too: captions never grow unbounded
        finally:
            self._call_lock.release()

    def _known_text_locked(self) -> str:
        return _join(self._committed, self._tail_text)

    def final(self, *, timeout_s: float | None = None) -> str:
        """The turn's text. Waits for an in-flight pause call (its partial is usually
        the answer) — at most ``timeout_s`` when given, else as long as it takes; past
        the deadline the best known text (committed + last tail) is returned without a
        new call, which is what the session's ``hold_max_s`` wants (A8)."""
        if timeout_s is None:
            acquired = self._call_lock.acquire()
        else:
            acquired = self._call_lock.acquire(timeout=max(0.0, float(timeout_s)))
        if not acquired:
            with self._lock:
                return self._known_text_locked()
        try:
            with self._lock:
                fresh = self._t_total - self._t_partial
                reusable = self._reusable_locked() is not None
                if self._last_partial is not None and (reusable or 0 <= fresh < self.final_fresh_s):
                    if reusable and not (0 <= fresh < self.final_fresh_s):
                        self.primed_hits += 1
                    return self._known_text_locked()
            self._inflight.set()
            try:
                p = self._tail_call()
                if p is None:                 # nothing to hear, or reset() ran: nobody's words
                    with self._lock:
                        return self._known_text_locked()
                return _join(p.committed, p.tail)
            finally:
                self._inflight.clear()
        finally:
            self._call_lock.release()

    @property
    def last_partial(self) -> Partial | None:
        return self._last_partial


# --------------------------------------------------------------------------- #
# Moonshine (optional, MIT): continuous words from a sliding-window encoder
# --------------------------------------------------------------------------- #

MOONSHINE_LAG_S = 0.6       # partial text runs 0.4–0.6 s behind the audio (measured)


def moonshine_available() -> bool:
    """Is the optional ``moonshine_voice`` package installed? A metadata probe
    (``find_spec``), not an import: nothing heavy loads until it is used, and a
    broken install still falls back to Parakeet where it is built (V1 #8)."""
    import importlib.util
    try:
        return importlib.util.find_spec("moonshine_voice") is not None
    except (ImportError, ValueError):
        return False


def moonshine_model(models_dir: Path | None = None, *, fetch: bool = False):
    """(model_path, arch) for ``medium-streaming-en``: ``<models_dir>/moonshine`` when it
    holds a model, else the package's own cache. ``fetch=True`` lets the package download
    it (never at boot; only an explicit fetch)."""
    from moonshine_voice import get_model_for_language
    from moonshine_voice.moonshine_api import ModelArch
    arch = ModelArch.MEDIUM_STREAMING
    local = (Path(models_dir) / "moonshine") if models_dir else None
    if local is not None and local.exists() and any(local.iterdir()):
        return str(local), arch
    if not fetch:
        # The package resolves the cached copy without a download when it is present.
        import moonshine_voice as mv
        cache = Path(mv.get_assets_path()) if hasattr(mv, "get_assets_path") else None
        if cache is None or not cache.exists():
            raise FileNotFoundError("Moonshine model not downloaded (moonshine_voice cache is empty)")
    path, arch = get_model_for_language("en", arch)
    return str(path), arch


class MoonshineSTT:
    """``moonshine_voice`` (MIT) streaming transcriber behind the same shape.

    Single-thread mode always (``MOONSHINE_ORT_SINGLE_THREAD=1``): the default
    thread mode pins every core (17–22 CPU-s per audio second measured); single
    thread costs 0.7–0.9 CPU-s per audio second with the same accuracy. Partials
    lag the audio by 0.4–0.6 s, so ``at_pause()``/``tick()`` return the current
    hypothesis at no extra cost and ``final()`` forces an update, then polls until
    the text settles or the 300 ms deadline passes. Its own line segmentation
    events are ignored; our endpointer stays the authority.

    ``push()`` (the event loop, every 32 ms) only appends to a buffer under a
    short lock; the STT thread feeds the buffered frames to the transcriber
    right before each ``update_transcription`` (76–522 ms measured), so a pause
    probe never stalls the socket handler (review T4 #7).
    """

    def __init__(self, model_path=None, arch=None, *, clock: Callable[[], float] = time.monotonic,
                 deadline_s: float = FINAL_DEADLINE_S, update_interval: float = 0.5) -> None:
        os.environ["MOONSHINE_ORT_SINGLE_THREAD"] = "1"
        from moonshine_voice.moonshine_api import ModelArch
        from moonshine_voice.transcriber import Transcriber
        if model_path is None:
            model_path, arch = moonshine_model()
        self._tr = Transcriber(model_path=str(model_path), model_arch=arch or ModelArch.MEDIUM_STREAMING,
                               update_interval=update_interval)
        self._force = int(getattr(Transcriber, "MOONSHINE_FLAG_FORCE_UPDATE", 1))
        self._clock = clock
        self.deadline_s = deadline_s
        self._lock = threading.Lock()          # the transcriber (STT thread)
        self._buf_lock = threading.Lock()      # the frame buffer (event loop ↔ STT thread)
        self._buf: list[np.ndarray] = []
        self.calls = 0
        self._prev_words: list[str] = []
        self._t_total = 0.0
        self._started = False
        self._start()

    def _start(self) -> None:
        if not self._started:
            self._tr.start()
            self._started = True

    def reset(self) -> None:
        with self._lock:
            try:
                if self._started:
                    self._tr.stop()
            except Exception as e:
                log.debug("voice: moonshine stop failed (%s)", e)
            self._started = False
            self._start()
            with self._buf_lock:
                self._buf = []
                self._t_total = 0.0
            self._prev_words = []

    def close(self) -> None:
        with self._lock:
            try:
                if self._started:
                    self._tr.stop()
                self._tr.close()
            except Exception as e:
                log.debug("voice: moonshine close failed (%s)", e)
            self._started = False

    def push(self, pcm16: np.ndarray, *, vad_prob: float) -> None:
        a = _to_float(pcm16)
        if a.size == 0:
            return
        with self._buf_lock:                   # never waits on the transcriber
            self._buf.append(a)
            self._t_total += len(a) / SAMPLE_RATE

    def _feed_locked(self) -> None:
        """With ``_lock`` HELD: hand the buffered frames to the transcriber."""
        with self._buf_lock:
            buf, self._buf = self._buf, []
        for a in buf:
            self._tr.add_audio(a.tolist(), SAMPLE_RATE)

    def _text(self, force: bool = False) -> str:
        with self._lock:
            self._feed_locked()
            self.calls += 1
            tx = self._tr.update_transcription(self._force if force else 0)
        lines = getattr(tx, "lines", None) or []
        return " ".join((getattr(ln, "text", "") or "").strip() for ln in lines).strip()

    def _partial(self, text: str, cost_ms: int) -> Partial:
        cur = words_of(text)
        stable = max(0, _agree(self._prev_words, cur) - CHURN_WORDS)
        self._prev_words = cur
        return Partial(committed="", tail=text, stable_words=stable,
                       punct_final=bool(_TERMINAL.search(text)), t_audio=round(self._t_total, 3), cost_ms=cost_ms)

    def at_pause(self) -> Partial:
        t0 = self._clock()
        text = self._text()
        return self._partial(text, int((self._clock() - t0) * 1000))

    def tick(self) -> Partial | None:
        return self.at_pause()

    def final(self) -> str:
        t0 = self._clock()
        text = self._text(force=True)
        while self._clock() - t0 < self.deadline_s:
            time.sleep(0.05)
            new = self._text(force=True)
            if new == text:
                break
            text = new
        return text

    @property
    def t_audio(self) -> float:
        return self._t_total

    @property
    def inflight(self) -> bool:
        return False


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def _rt(settings):
    return getattr(settings, "rt", None)


def build_streaming_stt(settings, shared_engine=None) -> StreamingSTT:
    """``rt.stt_engine``: ``parakeet`` (default) or ``moonshine`` (optional extra; falls back
    to Parakeet tail mode when the package or its model is missing). ``shared_engine`` is
    the host's loaded recogniser (anything with ``transcribe(audio) -> str``); without it
    a :class:`ParakeetRecognizer` is loaded from ``settings.parakeet_dir`` with spinning
    per ``rt.stt_spinning`` (default off)."""
    rt = _rt(settings)
    engine = (getattr(rt, "stt_engine", None) or getattr(settings, "stt_engine", None)
              or os.environ.get("DISPATCH_VOICE_STT", "parakeet") or "parakeet").strip().lower()
    if engine == "moonshine":
        try:
            path, arch = moonshine_model(getattr(settings, "models_dir", None))
            return MoonshineSTT(path, arch)
        except Exception as e:
            log.warning("voice: Moonshine STT unavailable (%s); using Parakeet tail mode", e)
    if shared_engine is not None and hasattr(shared_engine, "transcribe"):
        return ParakeetTailSTT(shared_engine.transcribe)
    threads = int(getattr(settings, "threads", 6) or 6)
    spinning = bool(getattr(rt, "stt_spinning", False))
    model_dir = getattr(settings, "parakeet_dir", None)
    if model_dir is None:
        raise RuntimeError("build_streaming_stt: no shared engine and settings has no parakeet_dir")
    return ParakeetTailSTT(ParakeetRecognizer(Path(model_dir), threads, spinning=spinning).transcribe)
