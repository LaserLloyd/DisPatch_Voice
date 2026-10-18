"""Semantic end-of-turn: ``TurnPolicy`` + per-device ``Learned`` endpointing (SPEC §4.3, C-§B.3).

The flat 1.8 s merge hold retires. At every pause ≥ ``probe_silence_s`` the
session gathers two signals — Smart Turn's P(done) on the last 8 s and the
streaming partial (does it end in terminal punctuation, does it end on a
continuation word) — and asks :meth:`TurnPolicy.decide` once with both in.
The table (measured basis in SPEC §4.3: Smart Turn scores complete clauses
0.76–0.99 and mid-clause 0.01–0.2; the one misfire class, "…maybe we could"
at 0.58, has no terminal punctuation / ends on a continuation word):

====================  ========================================  =============================
P(done)               partial ends with                         decision
====================  ========================================  =============================
≥ 0.90                anything                                  ``end``, hold 0
0.50–0.90             ``. ? !`` and not a continuation word     ``end``, hold 0
0.50–0.90             no terminal punctuation / continuation    ``hold`` h = clamp(EMA, min, max) × patience
< 0.50                ``. ? !``                                 ``reprobe`` at +0.4 s; ``end`` after 1.0 s
< 0.50                not terminal                              ``listen``; hard silence × patience ends it
====================  ========================================  =============================

Parakeet appends a full stop to almost every utterance ("I was thinking that
maybe we could." in the fixture baseline), so the continuation-word test is
what actually catches the misfire class; punctuation alone would not.

``partial=None`` (feasibility M1): Smart Turn is in but the tail transcript
is still in flight. P ≥ 0.90 ends the turn at once (no text, so no
speculation); anything else is ``listen`` — wait for the partial.

Learned endpointing (LiveKit's dynamic rule, re-implemented): an EMA of the
driver's intra-turn resume gaps sets the hold; every false end (speech
resumed within 1.0 s of an ``eot``) raises the hold floor by 0.15 s (max
``hold_max_s``) and the probe silence by 50 ms (max 0.5 s); clean turns decay
both. ``learned`` travels in ``ready.rt.learned`` and the device blob; the
server's value wins when both exist (the host passes the blob's only when it
has none of its own).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .settings import RtSettings
from .stt_stream import Partial

Action = Literal["end", "hold", "reprobe", "listen"]

#: Words a clause does not end on. A partial whose last word is one of these
#: is "mid-thought" even when Parakeet closed it with a full stop.
CONTINUATION_WORDS = frozenset("""
and or but so because to the a an that maybe could would should can if with for of
at in on by from as than then while unless until whether which who whom whose
also plus either neither nor like about into onto over under through
i'm i've i'd we're we've they're you're it's he's she's
""".split())

P_END = 0.90          # ≥ → end regardless of text
P_UNSURE = 0.50       # [0.5, 0.9) → text decides
REPROBE_AFTER_S = 0.4  # a `reprobe` row asks again after this much more silence
REPROBE_END_S = 1.0    # … and ends the turn after this much total silence
FALSE_END_STEP_S = 0.15
PROBE_STEP_S = 0.05
PROBE_MAX_S = 0.5
EMA_ALPHA = 0.9        # retention: ema = 0.9 × ema + 0.1 × gap
EMA_INITIAL_S = 0.6    # before the first observed gap: the middle of the clamp range
DECAY_STEPS = 3        # clean turns to undo one false-end step

_TERMINAL = re.compile(r"[.?!]+[\"'”’)\]]*\s*$")
_WORD = re.compile(r"[A-Za-z0-9']+")


@dataclass
class Signals:
    p_done: float | None          # Smart Turn on the last 8 s + 0.25 s silence (None = no model)
    partial: Partial | None
    pause_s: float                # silence so far
    patience: float               # the driver's multiplier


@dataclass
class Decision:
    action: Action
    hold_s: float = 0.0           # for "hold": clamp(EMA of resume gaps, hold_min, hold_max) × patience
    speculate: bool = False       # start the LLM now — end/hold rows only, NEVER on reprobe (A8)
    backchannel_ok: bool = False  # "listen" row only; the session applies the §4.6 limits
    reason: str = ""              # which table row fired (metrics/tests)


@dataclass(frozen=True)
class Learned:
    probe_ms: int
    hold_ms: int

    def as_dict(self) -> dict:
        return {"probe_ms": int(self.probe_ms), "hold_ms": int(self.hold_ms)}

    @classmethod
    def from_dict(cls, d) -> Learned | None:
        if not isinstance(d, dict):
            return None
        try:
            p, h = int(d.get("probe_ms")), int(d.get("hold_ms"))
        except (TypeError, ValueError, OverflowError):    # JSON's Infinity: int(inf) overflows
            return None
        if p <= 0 or h <= 0:
            return None
        return cls(probe_ms=p, hold_ms=h)


def last_word(text: str) -> str:
    words = _WORD.findall((text or "").lower())
    return words[-1] if words else ""


def ends_terminal(text: str) -> bool:
    return bool(_TERMINAL.search(text or ""))


def text_is_unsure(partial: Partial) -> bool:
    """No terminal punctuation, or the last word is a continuation word."""
    tail = partial.tail if partial.tail.strip() else partial.committed
    if not tail.strip():
        return True
    terminal = partial.punct_final or ends_terminal(tail)
    return (not terminal) or last_word(tail) in CONTINUATION_WORDS


class TurnPolicy:
    def __init__(self, cfg: RtSettings | None = None, learned: Learned | None = None):
        self.cfg = cfg or RtSettings()
        lo, hi = self.cfg.hold_min_s, self.cfg.hold_max_s
        self._ema = min(hi, max(lo, EMA_INITIAL_S))
        self._floor = lo                         # raised by false ends, decays on clean turns
        self._probe = self.cfg.probe_silence_s   # raised by false ends, decays on clean turns
        self._false_ends = 0
        self._clean_streak = 0
        if learned is not None:
            self._probe = min(PROBE_MAX_S, max(self.cfg.probe_silence_s, learned.probe_ms / 1000.0))
            self._ema = min(hi, max(lo, learned.hold_ms / 1000.0))

    # ------------------------------------------------------------------ decide
    @property
    def probe_silence_s(self) -> float:
        """The pause that should trigger the next probe (learned)."""
        return self._probe

    @property
    def hold_s(self) -> float:
        """The learned hold before the patience multiplier."""
        return min(self.cfg.hold_max_s, max(self.cfg.hold_min_s, max(self._ema, self._floor)))

    def decide(self, s: Signals) -> Decision:
        patience = max(0.1, float(s.patience or 1.0))
        hard = self.cfg.hard_silence_s * patience
        p = s.p_done
        partial = s.partial
        has_text = partial is not None and bool((partial.committed + " " + partial.tail).strip())

        if partial is None:
            # Smart Turn only: a confident end needs no words to end; anything else waits.
            if p is not None and p >= P_END:
                return Decision("end", reason="p_end_no_partial")
            if s.pause_s >= hard:
                return Decision("end", reason="hard_silence")
            return Decision("listen", reason="awaiting_partial")

        if s.pause_s >= hard:
            return Decision("end", speculate=has_text, reason="hard_silence")

        if p is None:
            # No Smart Turn model: the text alone decides, as a 0.5–0.9 row would.
            if text_is_unsure(partial):
                return Decision("hold", hold_s=self.hold_s * patience, speculate=has_text, reason="no_model_unsure")
            return Decision("end", speculate=has_text, reason="no_model_terminal")

        if p >= P_END:
            return Decision("end", speculate=has_text, reason="p_end")

        unsure = text_is_unsure(partial)
        if p >= P_UNSURE:
            if unsure:
                return Decision("hold", hold_s=self.hold_s * patience, speculate=has_text, reason="unsure_hold")
            return Decision("end", speculate=has_text, reason="terminal_end")

        # p < 0.5
        if not unsure:
            if s.pause_s >= REPROBE_END_S:
                return Decision("end", speculate=has_text, reason="reprobe_timeout_end")
            return Decision("reprobe", hold_s=REPROBE_AFTER_S, speculate=False, reason="reprobe")
        return Decision("listen", backchannel_ok=True, reason="listen")

    # ----------------------------------------------------------------- learning
    def observe(self, *, false_end: bool, resume_gap_s: float | None) -> None:
        """Feed the outcome of a turn: ``false_end`` when speech resumed within 1.0 s of
        an ``eot`` (the gap is ``resume_gap_s``); a merged hold also reports its gap with
        ``false_end=False``; a clean turn reports ``(False, None)``."""
        lo, hi = self.cfg.hold_min_s, self.cfg.hold_max_s
        if resume_gap_s is not None and resume_gap_s > 0:
            gap = min(hi, max(0.0, float(resume_gap_s)))
            self._ema = EMA_ALPHA * self._ema + (1.0 - EMA_ALPHA) * gap
        if false_end:
            self._false_ends += 1
            self._clean_streak = 0
            self._floor = min(hi, self._floor + FALSE_END_STEP_S)
            self._probe = min(PROBE_MAX_S, self._probe + PROBE_STEP_S)
            return
        self._clean_streak += 1
        if self._clean_streak % DECAY_STEPS == 0:
            self._floor = max(lo, self._floor - FALSE_END_STEP_S)
            self._probe = max(self.cfg.probe_silence_s, self._probe - PROBE_STEP_S)

    @property
    def false_ends(self) -> int:
        return self._false_ends

    @property
    def learned(self) -> Learned:
        return Learned(probe_ms=round(self._probe * 1000), hold_ms=round(self.hold_s * 1000))

    def reset(self) -> None:
        """Back to the configured defaults (the Voice lab's Reset)."""
        self.__init__(self.cfg, None)
