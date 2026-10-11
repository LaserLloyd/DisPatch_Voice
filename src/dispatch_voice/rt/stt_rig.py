"""``RigStreamingSTT``: the ``StreamingSTT`` seam (C-§B.3) on a SpeechForge rig, with
the CPU recogniser behind it (ADDON-ENGINE §3).

Per utterance, never per call: at the utterance's first frame the adapter asks
the connection whether a NEW turn may start on the rig (``link.usable()``: ready,
proven by a pong, lease held, no failure in the last 30 s). If not, the whole
utterance is the CPU recogniser's, exactly as without a rig — no deadline, no
hedge. If so:

* ``push`` forwards each frame to the rig (20 ms sf.1 frames on the session's mic
  clock) and keeps it in the mic ring; the CPU recogniser stays idle;
* ``at_pause`` returns the latest ``stt.partial`` covering the pause (waiting at
  most ``PAUSE_WAIT_S`` for it); ``tick`` returns a new partial when one came;
* ``turn_fn_for(local)`` gives the session ``eou{p}`` (the rig's Smart Turn at its
  150 ms pause) instead of the local Smart Turn — the local one runs only when the
  rig has none for this pause within ``PAUSE_WAIT_S`` (120 ms: a pause the rig's
  VAD did not see costs at most that over the CPU path);
* ``final`` sends ``stt.flush{id}`` and waits for ``stt.final{id}`` up to
  ``stt_final_deadline_ms`` (350: the rig measured p50 115 / p99 172 ms under
  continuous partials, plus up to 80 ms of tailnet RTT); past it the rig is marked
  down and the CPU transcribes the ring (once: <= 350 ms + a CPU transcription of
  the WHOLE utterance, since the idle CPU recogniser committed nothing meanwhile);
* the rig dying mid-utterance (the connection's epoch moves) replays the ring
  into the CPU recogniser, which then continues live: no lost and no doubled
  words, because the CPU hears the whole utterance exactly once.

The ring holds the open utterance (frames since the last reset), bounded at
``RING_S`` — the session's longest utterance — so a failover never drops the
start of a long one.

This module never imports ``dispatch_voice.host``: the connection is handed in
(``host/rig.py``'s ``RigConnection``) and used through a small duck-typed API.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from .stt_stream import _TERMINAL, Partial

log = logging.getLogger("dispatch.voice")

MIC_RATE = 16000
#: The ring covers the session's longest utterance (``max_utterance_s``, 30 s) plus slack.
RING_S = 32.0
#: ``at_pause`` / the EOU score wait at most this long for the rig's frame for this pause.
PAUSE_WAIT_S = 0.12
#: A frame counts as speech at this Silero probability (the detector's own threshold).
SPEECH_PROB = 0.5
#: The rig's VAD and the box's may disagree on where the pause began by this much.
PAUSE_TOLERANCE = int(0.3 * MIC_RATE)
POLL_S = 0.05


class RigStreamingSTT:
    """See the module docstring. ``cpu`` is the session's own ``StreamingSTT``."""

    def __init__(self, cpu, link, *, final_deadline_ms: int | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.cpu = cpu
        self.link = link
        self.final_deadline_s = (final_deadline_ms if final_deadline_ms is not None
                                 else int(getattr(getattr(link, "settings", None), "stt_final_deadline_ms", 350))) / 1000.0
        self.clock = clock
        self._lock = threading.RLock()
        self._mode = "idle"                  # idle | rig | cpu (this utterance)
        self._epoch = -1
        self._ring: deque[tuple[np.ndarray, float]] = deque()
        self._ring_n = 0
        self._utt_start = 0                  # mic clock at this utterance's first frame
        self._last_speech = 0                # mic clock after the last speech frame
        self._partial_seen = 0               # the last stt.partial n handed out
        self._eou_seen = 0
        self._eou_used_at = -1               # `at` of the last eou handed out (this utterance)
        self._last_tick = float("-inf")
        self._fid = 0
        self._tick_interval = float(getattr(cpu, "tick_interval_s", 1.0) or 1.0)
        self.failovers = 0
        self.rig_utterances = 0

    # ---- the session reads/sets a few of the CPU recogniser's attributes
    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name in ("cpu",):
            raise AttributeError(name)
        return getattr(self.cpu, name)

    @property
    def tick_interval_s(self) -> float:
        return self._tick_interval

    @tick_interval_s.setter
    def tick_interval_s(self, v: float) -> None:
        self._tick_interval = float(v)
        if hasattr(self.cpu, "tick_interval_s"):
            self.cpu.tick_interval_s = v

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def on_rig(self) -> bool:
        return self._mode == "rig"

    # ------------------------------------------------------------------ helpers
    def _alive(self) -> bool:
        return self.link.alive(self._epoch)

    def _failover_locked(self, reason: str) -> None:
        """With the lock held: this utterance moves to the CPU recogniser, which hears
        the ring first (the whole utterance so far), then continues live."""
        if self._mode != "rig":
            return
        self._mode = "cpu"
        self.failovers += 1
        log.info("voice: rig STT lost mid-utterance (%s); replaying %.1f s into the CPU recogniser",
                 reason, self._ring_n / MIC_RATE)
        for pcm, prob in self._ring:
            self.cpu.push(pcm, vad_prob=prob)

    def _check_locked(self) -> None:
        if self._mode == "rig" and not self._alive():
            self._failover_locked(getattr(self.link, "reason", None) or "down")

    def _wait(self, pred: Callable[[], Any], timeout: float) -> Any:
        """Wait on the connection's condition until ``pred()`` is truthy, the rig dies,
        or ``timeout`` passes. Returns pred()'s value or None."""
        end = self.clock() + max(0.0, timeout)
        cond = self.link.cond
        with cond:
            while True:
                got = pred()
                if got:
                    return got
                if not self._alive():
                    return None
                left = end - self.clock()
                if left <= 0:
                    return None
                cond.wait(min(POLL_S, left))

    def _partial_obj(self, frame: dict) -> Partial:
        tail = str(frame.get("tail") or "")
        upto = int(frame.get("upto") or 0)
        return Partial(committed=str(frame.get("committed") or ""), tail=tail,
                       stable_words=int(frame.get("stable_words") or 0),
                       punct_final=bool(frame.get("punct_final", bool(_TERMINAL.search(tail)))),
                       t_audio=round(max(0, upto - self._utt_start) / MIC_RATE, 3),
                       cost_ms=int(frame.get("ms") or 0))

    def _latest_partial(self, *, newer_than: int = 0) -> tuple[int, dict] | None:
        """The newest stt.partial of THIS utterance (call with the cond held)."""
        for n, f in reversed(self.link.partials):
            if n <= newer_than:
                return None
            if int(f.get("upto") or 0) > self._utt_start:
                return n, f
        return None

    # ------------------------------------------------------------------ StreamingSTT
    def push(self, pcm16: np.ndarray, *, vad_prob: float) -> None:
        with self._lock:
            if self._mode == "idle":
                on_rig = bool(self.link.usable())
                self._mode = "rig" if on_rig else "cpu"
                self._epoch = self.link.epoch
                self._utt_start = int(self.link.mic_pts)
                self._last_speech = self._utt_start
                with self.link.cond:
                    self._partial_seen = self.link.partial_n
                    self._eou_seen = self.link.eou_n
                self._eou_used_at = -1
                self.link.note_turn(on_rig)
            # A rig that died since the last frame: the ring (without this frame) goes to
            # the CPU first, then this frame below — each frame reaches the CPU once.
            self._check_locked()
            if self._mode == "rig":
                self._ring.append((pcm16, float(vad_prob)))
                self._ring_n += len(pcm16)
                while self._ring and self._ring_n - len(self._ring[0][0]) >= RING_S * MIC_RATE:
                    self._ring_n -= len(self._ring.popleft()[0])
                end = self.link.send_mic(pcm16)
                if float(vad_prob) >= SPEECH_PROB:
                    self._last_speech = end
                return
        self.cpu.push(pcm16, vad_prob=vad_prob)

    def at_pause(self) -> Partial:
        with self._lock:
            self._check_locked()
            if self._mode != "rig":
                return self.cpu.at_pause()
            target = self._last_speech
        got = self._wait(lambda: (lambda p: p if p and int(p[1].get("upto") or 0) >= target else None)(
            self._latest_partial()), PAUSE_WAIT_S)
        with self._lock:
            self._check_locked()
            if self._mode != "rig":
                return self.cpu.at_pause()
            if got is None:
                with self.link.cond:
                    got = self._latest_partial()
            if got is None:
                return Partial("", "", 0, False, round(max(0, target - self._utt_start) / MIC_RATE, 3), 0)
            self._partial_seen = max(self._partial_seen, got[0])
            return self._partial_obj(got[1])

    def tick(self) -> Partial | None:
        with self._lock:
            self._check_locked()
            if self._mode != "rig":
                return self.cpu.tick() if self._mode == "cpu" else None
            now = self.clock()
            if now - self._last_tick < self._tick_interval:
                return None
            with self.link.cond:
                got = self._latest_partial(newer_than=self._partial_seen)
            if got is None:
                return None
            self._last_tick = now
            self._partial_seen = got[0]
            return self._partial_obj(got[1])

    def prime(self) -> bool:
        with self._lock:
            if self._mode == "rig":
                return False                 # the rig decodes the tail itself
        fn = getattr(self.cpu, "prime", None)
        return bool(fn()) if callable(fn) else False

    def turn_fn_for(self, local: Callable[[np.ndarray], float] | None):
        """The session's Smart Turn for a probe: the rig's ``eou{p}`` for this pause while
        the utterance is on the rig, else (or when it has none in time) ``local``."""
        def score(audio):
            with self._lock:
                self._check_locked()
                on_rig = self._mode == "rig"
                floor = self._last_speech - PAUSE_TOLERANCE
                seen = self._eou_seen
                # A re-probe of a pause the rig already scored (it scores once per pause):
                # the local model rescores it with the longer silence, at once.
                rescore = self._eou_used_at >= 0 and self._eou_used_at >= floor
            if on_rig and not rescore:
                def pred():
                    for n, f in reversed(self.link.eous):
                        if n <= seen:
                            return None
                        if int(f.get("at") or 0) >= floor and isinstance(f.get("p"), (int, float)):
                            return n, f
                    return None
                got = self._wait(pred, PAUSE_WAIT_S)
                if got is not None:
                    with self._lock:
                        self._eou_seen = max(self._eou_seen, got[0])
                        self._eou_used_at = int(got[1].get("at") or 0)
                    return float(got[1]["p"])
                with self._lock:
                    self._check_locked()
            return local(audio) if local is not None else None
        return score

    def final(self, *, timeout_s: float | None = None) -> str:
        with self._lock:
            self._check_locked()
            mode = self._mode
        if mode != "rig":
            return self._cpu_final(timeout_s)
        t0 = self.clock()
        with self._lock:
            self._fid += 1
            fid = self._fid
            self.link.flush_mic()
            self.link.send_threadsafe({"type": "stt.flush", "id": fid})
        deadline = self.final_deadline_s if timeout_s is None else min(self.final_deadline_s, float(timeout_s))
        got = self._wait(lambda: self.link.finals.get(fid), deadline)
        if got is not None:
            with self._lock:
                if self._mode == "rig":
                    self.rig_utterances += 1
                    self.link.metrics.add("stt")
                    return str(got.get("text") or "").strip()
        # Late or dead: the rig is marked down and the CPU transcribes the ring.
        with self._lock:
            if self._mode == "rig":
                if self._alive():
                    self.link.fail("final_timeout", move=False)
                self._failover_locked("final_timeout")
        left = None if timeout_s is None else max(0.0, float(timeout_s) - (self.clock() - t0))
        return self._cpu_final(left)

    def _cpu_final(self, timeout_s: float | None) -> str:
        if timeout_s is None:
            return self.cpu.final()
        try:
            return self.cpu.final(timeout_s=timeout_s)
        except TypeError:
            return self.cpu.final()

    def reset(self) -> None:
        with self._lock:
            if self._mode == "rig" and self._alive():
                self.link.drop_mic_rest()
                self.link.send_threadsafe({"type": "stt.reset"})
            self._mode = "idle"
            self._ring.clear()
            self._ring_n = 0
        self.cpu.reset()
        probe = getattr(self.link, "maybe_probe", None)
        if callable(probe):
            probe()                          # between turns: the place to re-promote
