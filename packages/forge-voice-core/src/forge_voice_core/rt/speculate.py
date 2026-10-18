"""Speculative LLM start with cancel/merge (SPEC §4.4, C-§B.3 ``Speculator``).

The session starts a run on the first ``end``/``hold`` decision taken AFTER
``at_pause()`` returned (the Speculator needs text — feasibility C2), never
on a ``reprobe`` row (the policy's ``Decision.speculate`` is False there). The
host hook (``start``) runs the same advisor/API call the real turn would,
persists and broadcasts nothing, and returns a ``SpeculativeRun`` or ``None``
(gateway bots, or a real turn holding the thread).

Spend cap (hardening M7): at most ``max_per_turn`` (2) runs per turn, none
within ``min_gap_s`` (3 s) of the previous attempt, nothing on fewer than four
words, and a run older than ``timeout_s`` (20 s) is discarded unread.

``start()`` is fail-soft: a host hook that raises or takes longer than
``start_timeout_s`` (5 s) counts as ``refused`` and the turn goes on without
speculation — an optimisation never fails the turn-end path (review T4 #8).

Commit (``commit(final_text)``): the run is handed back when
``similar(final, started) ≥ threshold`` (0.95, end-weighted), else discarded.
Pass ``head_id`` (the thread's current last message id) when known: a run
whose ``head_id`` moved is stale (hardening M1) — discarded and counted
separately, because DisPatch would refuse to adopt it anyway.

``stats`` feeds ``metrics.speculation`` on the Voice lab line
(``{started, adopted, discarded, stale, prompt_chars}``) so the cost is
visible before ``rt.speculate: all`` widens the default.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from difflib import SequenceMatcher
from typing import Any

from .. import _log

log = _log.get()

_WORD = re.compile(r"[a-z0-9']+")
END_WEIGHTED_WORDS = 5
END_WEIGHT = 2.0
MIN_WORDS = 4


def _words(s: str) -> list[str]:
    return _WORD.findall((s or "").lower().replace("’", "'"))


def similar(a: str, b: str) -> float:
    """End-weighted word similarity in [0, 1]: the last five words of each side
    weigh 2×, the rest 1×; matched words (longest common blocks) count on both
    sides. 1.0 for the same words in the same order; a changed last word on a
    seven-word sentence scores ≈ 0.8, a changed article mid-way through 25
    words ≈ 0.97 (the measured churn class never exceeded the last two words,
    so a committed partial whose tail settled scores 1.0)."""
    wa, wb = _words(a), _words(b)
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    if wa == wb:
        return 1.0

    def weights(n: int) -> list[float]:
        return [END_WEIGHT if i >= n - END_WEIGHTED_WORDS else 1.0 for i in range(n)]

    ta, tb = weights(len(wa)), weights(len(wb))
    sm = SequenceMatcher(a=wa, b=wb, autojunk=False)
    matched = 0.0
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            matched += ta[i + k] + tb[j + k]
    return matched / (sum(ta) + sum(tb))


class Speculator:
    def __init__(self, start: Callable[[str], Awaitable[Any | None]], *,
                 timeout_s: float = 20.0, threshold: float = 0.95,
                 max_per_turn: int = 2, min_gap_s: float = 3.0, start_timeout_s: float = 5.0,
                 clock: Callable[[], float] = time.monotonic):
        self._start_fn = start
        self.timeout_s = float(timeout_s)
        self.start_timeout_s = max(0.05, float(start_timeout_s))
        self.threshold = float(threshold)
        self.max_per_turn = max(1, int(max_per_turn))
        self.min_gap_s = max(0.0, float(min_gap_s))
        self._clock = clock
        self._run: Any | None = None
        self._run_text = ""
        self._run_at = 0.0
        self._orphans: list[Any] = []
        self._started_this_turn = 0
        self._last_attempt_at: float | None = None
        self.stats: dict[str, int] = {"started": 0, "adopted": 0, "discarded": 0, "stale": 0,
                                      "prompt_chars": 0, "refused": 0}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ state
    @property
    def live(self) -> bool:
        return self._run is not None

    @property
    def started_text(self) -> str:
        return self._run_text if self._run is not None else ""

    def new_turn(self) -> None:
        """Reset the per-turn counters. A run left over from the previous turn is
        stale by definition; it is discarded on the next async call."""
        if self._run is not None:
            self._orphans.append(self._run)
            self._run = None
            self._run_text = ""
        self._started_this_turn = 0
        self._last_attempt_at = None

    async def _reap(self) -> None:
        while self._orphans:
            run = self._orphans.pop()
            await self._discard_run(run)

    async def _discard_run(self, run: Any, *, count: str = "discarded") -> None:
        self.stats[count] += 1
        fn = getattr(run, "discard", None)
        if fn is None:
            return
        try:
            res = fn()
            if asyncio.iscoroutine(res):
                await res
        except Exception as e:  # noqa: BLE001 - a run that fails to discard is already dead
            log.debug("voice: speculative run discard failed (%s)", e)

    def _expired(self) -> bool:
        return self._run is not None and (self._clock() - self._run_at) > self.timeout_s

    # ------------------------------------------------------------------ actions
    async def start(self, text: str) -> bool:
        """Start a run and return True; False (nothing started) when the turn's cap
        or gap blocks it, when ``text`` has fewer than four words, or when the
        host declined, raised or did not answer within ``start_timeout_s`` (all
        counted as ``refused``). Starting discards the previous run first (A8)."""
        async with self._lock:
            await self._reap()
            now = self._clock()
            if len(_words(text)) < MIN_WORDS:
                return False
            if self._started_this_turn >= self.max_per_turn:
                return False
            if self._last_attempt_at is not None and (now - self._last_attempt_at) < self.min_gap_s:
                return False
            if self._run is not None:
                old, self._run, self._run_text = self._run, None, ""
                await self._discard_run(old)
            self._last_attempt_at = now
            try:
                run = await asyncio.wait_for(self._start_fn(text), timeout=self.start_timeout_s)
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as e:  # noqa: BLE001 - whatever the host did, speculation is optional
                log.warning("voice: speculative start refused (%s: %s)", type(e).__name__, e)
                run = None
            if run is None:
                self.stats["refused"] += 1
                return False
            self._run, self._run_text, self._run_at = run, text, now
            self._started_this_turn += 1
            self.stats["started"] += 1
            chars = getattr(run, "prompt_chars", None)
            self.stats["prompt_chars"] += int(chars) if isinstance(chars, int) and chars > 0 else len(text)
            return True

    async def commit(self, final_text: str, *, head_id: str | None = None) -> Any | None:
        """The run when ``similar(final, started) ≥ threshold`` (and the thread head
        is unchanged when ``head_id`` is given), else discard → None."""
        async with self._lock:
            await self._reap()
            if self._run is None:
                return None
            run, text = self._run, self._run_text
            self._run, self._run_text = None, ""
            if self._expired_at(run):
                await self._discard_run(run)
                return None
            run_head = getattr(run, "head_id", None)
            if head_id is not None and run_head is not None and head_id != run_head:
                await self._discard_run(run, count="stale")   # counted on its own, not as discarded
                return None
            if similar(final_text, text) >= self.threshold:
                self.stats["adopted"] += 1
                return run
            await self._discard_run(run)
            return None

    def _expired_at(self, run: Any) -> bool:
        return (self._clock() - self._run_at) > self.timeout_s

    async def discard(self) -> None:
        """Idempotent; used by merges/addenda and on a real interruption."""
        async with self._lock:
            await self._reap()
            if self._run is None:
                return
            run, self._run, self._run_text = self._run, None, ""
            await self._discard_run(run)

    def note_stale(self) -> None:
        """DisPatch refused adoption after a commit (its own head check); counted here
        so the lab line shows it next to the Speculator's own stale count."""
        self.stats["stale"] += 1
        if self.stats["adopted"] > 0:
            self.stats["adopted"] -= 1
