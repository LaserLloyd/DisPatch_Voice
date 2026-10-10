"""Uplink sample-offset accounting (pure; contracts C-§B.3, A11; SPEC §5.5).

Every v3 uplink message carries ``off``, the 16 kHz sample index of its first
sample since the stream started (u32, wraps). The client's mic gate leaves holes
while nobody talks, and a reconnect replays its ring from the server's cursor, so
messages can skip ahead or overlap. :class:`OffsetTracker` turns them back into
the continuous signal the turn detector expects:

* in order -> the samples, as they are;
* a gap of at most :data:`MAX_GAP_S` -> that many zeros first (the end-of-turn
  logic needs to *hear* the silence; 5 s costs <= 75 ms of Silero);
* a larger gap, a jump far backwards, or a gap the zero-fill budget cannot pay
  for -> :class:`GapTooLarge`: the caller resets the detector and then feeds
  ``err.frames`` (the tracker has already re-anchored on this message);
* an overlap (``off`` below what was already fed: a replay) -> the overlapping
  samples are dropped, possibly the whole message.

Zero-fill is a token bucket refilled at :data:`FILL_RATE` x realtime (2x) and
holding at most :data:`FILL_BURST_S` (10 s), so a client whose ``off`` races
ahead (5 s jumps every 100 ms would be 50x realtime of synthetic silence into
Silero) gets resets instead (hardening M5).
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .wire3 import U32, u32_diff

RATE = 16_000
MAX_GAP_S = 5.0
FILL_RATE = 2.0
FILL_BURST_S = 10.0
#: Further back than this is not a ring replay (the ring holds 30 s) but a client
#: that restarted its counter: re-anchor instead of dropping audio for hours.
MAX_REWIND_S = 60.0


class GapTooLarge(Exception):
    """Reset the detector, then feed :attr:`frames` (already re-anchored)."""

    def __init__(self, reason: str, gap: int, frames: list[np.ndarray]) -> None:
        super().__init__(reason)
        self.reason = reason            # "gap" | "rate" | "rewind"
        self.gap = gap                  # samples (negative for a rewind)
        self.frames = frames


class OffsetTracker:
    """Per-socket uplink cursor. ``now_off`` (optional, per call) is the realtime
    clock in samples; by default the tracker reads ``clock()`` x 16 000."""

    def __init__(self, *, rate: int = RATE, max_gap_s: float = MAX_GAP_S, fill_rate: float = FILL_RATE,
                 fill_burst_s: float = FILL_BURST_S, max_rewind_s: float = MAX_REWIND_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.rate = rate
        self.max_gap = int(max_gap_s * rate)
        self.fill_rate = fill_rate
        self.fill_burst = int(fill_burst_s * rate)
        self.max_rewind = int(max_rewind_s * rate)
        self._clock = clock
        self._anchored = False
        #: the next sample offset the detector expects (u32)
        self.expected = 0
        #: end (exclusive) of the furthest uplink message received (u32) — `pong.rx_off`
        self.rx_off = 0
        self._credit = float(self.max_gap)      # one maximal gap may be filled at once
        self._credit_at: int | None = None
        # stats
        self.zero_filled = 0
        self.dropped = 0
        self.resets = 0

    @property
    def anchored(self) -> bool:
        return self._anchored

    def _now(self, now_off: int | None) -> int:
        return int(now_off) if now_off is not None else int(self._clock() * self.rate)

    def _refill(self, now: int) -> None:
        if self._credit_at is not None and now > self._credit_at:
            self._credit = min(float(self.fill_burst), self._credit + (now - self._credit_at) * self.fill_rate)
        self._credit_at = now if self._credit_at is None else max(self._credit_at, now)

    def _note_rx(self, end: int) -> None:
        if not self._anchored or u32_diff(end, self.rx_off) > 0:
            self.rx_off = end & U32

    def feed(self, off: int, pcm16: np.ndarray, *, now_off: int | None = None) -> list[np.ndarray]:
        """Account one message (``off`` = its first sample) -> the arrays to feed, in order."""
        pcm = np.asarray(pcm16, dtype=np.int16).reshape(-1)
        n = len(pcm)
        off = int(off) & U32
        now = self._now(now_off)
        self._refill(now)
        end = (off + n) & U32
        if not self._anchored:
            self._anchored = True
            self.rx_off = end
            self.expected = end
            return [pcm] if n else []
        self._note_rx(end)
        d = u32_diff(off, self.expected)
        if d == 0:
            self.expected = end
            return [pcm] if n else []
        if d < 0:
            if -d > self.max_rewind:
                return self._reanchor("rewind", d, pcm, end)
            skip = min(-d, n)
            self.dropped += skip
            if skip >= n:
                return []
            self.expected = end
            return [pcm[skip:]]
        # d > 0: a gap
        if d > self.max_gap:
            return self._reanchor("gap", d, pcm, end)
        if d > self._credit:
            return self._reanchor("rate", d, pcm, end)
        self._credit -= d
        self.zero_filled += d
        self.expected = end
        out = [np.zeros(d, np.int16)]
        if n:
            out.append(pcm)
        return out

    def _reanchor(self, reason: str, gap: int, pcm: np.ndarray, end: int):
        self.resets += 1
        self.expected = end
        if reason == "rewind":
            self.rx_off = end
        raise GapTooLarge(reason, gap, [pcm] if len(pcm) else [])

    def reset(self) -> None:
        """Forget the anchor (a new stream); the next message anchors again. Stats and
        the zero-fill budget are kept: a client cannot refill it by forcing resets."""
        self._anchored = False
