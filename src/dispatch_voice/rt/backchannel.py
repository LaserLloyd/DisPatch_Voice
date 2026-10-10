"""Backchannel timing and the interruption classifier (SPEC §4.6–4.7, C-§B.3).

``BackchannelGate`` is pure timing: a backchannel ("Mm-hm.") may go out on a
``listen`` row when the turn is ≥ 4 s old, the pause ≥ 0.5 s, ≥ 6 s since the
last one and fewer than three this turn. The session owns the other gates
(Interrupt-by-talking on, Reactions "Often", the bot opted in) and calls
``sent(now)`` when it actually plays one.

``classify_interruption`` decides what the first pause partial of speech over
a reply means, in this order:

* ``echo`` — its (non-digit) words are a run the bot just said (Amendment
  T4-3): at least three tokens, found **in order and contiguously inside one
  sentence** of ``recent_said``, starting and ending on matched tokens, with
  at most one interior token unmatched and ≥ 80 % matched overall. Loudspeaker
  bleed is a contiguous run of one sentence; words of two sentences mixed, or
  a run with a new last word ("I was looking at the budget"), is the user. A
  single command word is never an echo, so "stop" still stops a reply that
  happened to contain the word; one or two tokens go to the backchannel rule;
* ``stop`` — a command word or stop-intent phrase on its own ("stop", "wait",
  "no", "hold on", "okay okay", "got it, thanks", "that's enough", "never
  mind" …), or a second backchannel verdict within 3 s of the previous one
  (experience M4: "okay okay / got it, thanks" over a long answer means *you
  can stop*). Plain "okay" / "yeah" / "right" once is a backchannel, not a
  stop (design §7);
* ``backchannel`` — under 1.0 s, at most two tokens, none a command word.
  Neither Parakeet nor Moonshine transcribes "uh-huh" (the fixture gives
  "Odd High."), so this keys on duration and token count, never on the word;
* ``real`` — everything else: "Wait, stop, that's not what I meant" carries
  content after the command and becomes the next turn.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Verdict = Literal["echo", "backchannel", "stop", "real"]

ECHO_OVERLAP = 0.80
ECHO_MIN_TOKENS = 3
ECHO_MAX_GAPS = 1
BACKCHANNEL_MAX_S = 1.0
BACKCHANNEL_MAX_TOKENS = 2
SECOND_BACKCHANNEL_S = 3.0
#: A stop utterance is at most this many tokens (fillers aside): "okay, okay, got it,
#: thanks" is five, "alright that's enough, thank you" six (FR-works 7).
STOP_MAX_TOKENS = 8

#: Single words that mean "stop" on their own or in a short run ("no no", "wait wait").
COMMAND_WORDS = frozenset("""
stop wait no pause skip what sorry enough quiet hush shush
""".split())
#: Whole-utterance phrases that mean "stop" (normalised: lowercase, no punctuation).
STOP_PHRASES = frozenset("""
hold on|hang on|okay okay|ok ok|got it|thanks|thank you|that's enough|thats enough|never mind|nevermind
quiet|be quiet|shut up|got it thanks|okay thanks|ok thanks|okay got it|ok got it|all right that's enough
alright that's enough|stop it|stop stop|no no|wait wait|hold on a second|hang on a second|enough enough
thanks so much|thank you so much|thanks a lot|thank you very much|thanks very much|thank you thank you
""".replace("\n", "|").split("|"))
#: One of these alone is a backchannel; repeated ("okay okay"), or next to a stop
#: phrase ("perfect, thanks"), it is a stop.
ACK_WORDS = frozenset("okay ok yeah yes right sure mhm mm uh-huh uhhuh yep yup aha alright perfect great cool".split())
#: Filler tokens ignored before the phrase match ("uh, stop" → "stop").
FILLERS = frozenset("uh um er ah hmm hm oh".split())

_TOKEN = re.compile(r"[a-z0-9']+")
_DIGIT = re.compile(r"^\d+(st|nd|rd|th|s)?$")      # numbers and ordinals: STT hears them loosely


@dataclass
class BackchannelGate:
    min_turn_s: float = 4.0
    min_pause_s: float = 0.5
    min_gap_s: float = 6.0
    max_per_turn: int = 3

    def __post_init__(self) -> None:
        self._last_at: float | None = None
        self._count = 0

    def ok(self, *, turn_s: float, pause_s: float, now: float) -> bool:
        if turn_s < self.min_turn_s or pause_s < self.min_pause_s:
            return False
        if self._count >= self.max_per_turn:
            return False
        return self._last_at is None or (now - self._last_at) >= self.min_gap_s

    def sent(self, now: float) -> None:
        self._last_at = now
        self._count += 1

    def new_turn(self) -> None:
        self._count = 0

    @property
    def count(self) -> int:
        return self._count


def tokens(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower().replace("’", "'"))


def _run_match(mine: list[str], said: list[str], j: int) -> int:
    """Tokens of ``mine`` matched against ``said[j:]`` as one contiguous run, allowing
    ``ECHO_MAX_GAPS`` interior gaps (one token substituted, or inserted on either side).
    0 unless the run covers ALL of ``mine`` — an unmatched first or last token means the
    user said something of their own."""
    matched, gaps, k, l = 0, 0, 0, j
    n, m = len(mine), len(said)
    while k < n and l < m:
        if mine[k] == said[l]:
            matched, k, l = matched + 1, k + 1, l + 1
            continue
        if gaps >= ECHO_MAX_GAPS:
            break
        gaps += 1
        if k + 1 < n and l + 1 < m and mine[k + 1] == said[l + 1]:
            k, l = k + 1, l + 1                      # substitution
        elif k + 1 < n and mine[k + 1] == said[l]:
            k += 1                                   # an extra token in mine
        elif l + 1 < m and mine[k] == said[l + 1]:
            l += 1                                   # an extra token in said
        else:
            break
    return matched if k == n and mine[-1] == said[l - 1] else 0


def _echo_fraction(words: list[str], recent_said: Sequence[str]) -> float:
    """Best fraction of the user's non-digit tokens found as one contiguous run inside
    a single recent sentence (0.0 below ``ECHO_MIN_TOKENS``)."""
    mine = [w for w in words if not _DIGIT.match(w)]
    if len(mine) < ECHO_MIN_TOKENS:
        return 0.0
    best = 0
    for s in recent_said or ():
        said = tokens(s)
        for j, w in enumerate(said):
            if w == mine[0]:
                best = max(best, _run_match(mine, said, j))
            if best == len(mine):
                return 1.0
    return best / len(mine)


def _overlap(words: list[str], recent_said: Sequence[str]) -> float:
    """Kept for callers of the old name: the contiguous-run fraction."""
    return _echo_fraction(words, recent_said)


#: The longest stop phrase, in tokens (the greedy segmentation tries these first).
_PHRASE_MAX = max(len(p.split()) for p in STOP_PHRASES)


def _stop_units(core: list[str]) -> tuple[int, int] | None:
    """Segment ``core`` into stop phrases, command words and acknowledgement words.
    ``(acks, others)`` for the segmentation that uses EVERY token with the most
    phrases/commands, else None — one content word ("okay, got it, and the
    weather?") makes it a real turn. Dynamic programming, not greedy: "got it
    thanks so much" is "got it" + "thanks so much", not "got it thanks" + "so much"."""
    n = len(core)
    best: list[tuple[int, int] | None] = [None] * (n + 1)   # suffix i → (others, acks)
    best[n] = (0, 0)
    for i in range(n - 1, -1, -1):
        cands = []
        for k in range(1, min(_PHRASE_MAX, n - i) + 1):
            rest = best[i + k]
            if rest is None:
                continue
            piece = " ".join(core[i:i + k])
            if piece in STOP_PHRASES or (k == 1 and piece in COMMAND_WORDS):
                cands.append((rest[0] + 1, rest[1]))
            elif k == 1 and piece in ACK_WORDS:
                cands.append((rest[0], rest[1] + 1))
        best[i] = max(cands) if cands else None
    if best[0] is None:
        return None
    others, acks = best[0]
    return acks, others


def _is_stop(words: list[str]) -> bool:
    """A stop: the utterance is made ONLY of stop phrases, command words and
    acknowledgement words (fillers aside), at most :data:`STOP_MAX_TOKENS` long, with
    a stop phrase or command word in it, or one acknowledgement word repeated —
    "okay" once is a backchannel; "okay okay", "okay, okay, got it, thanks" and
    "yeah no, that's enough" are stops (experience M4; FR-works 7: anything over
    four tokens used to become a new turn)."""
    core = [w for w in words if w not in FILLERS]
    if not core or len(core) > STOP_MAX_TOKENS:
        return False
    units = _stop_units(core)
    if units is None:
        return False
    acks, others = units
    # A stop phrase or command word anywhere in it ("okay, okay, got it, thanks"), or
    # one acknowledgement word repeated ("okay okay"). Different acks alone ("okay
    # right sure") stay what they were: not a stop.
    return others >= 1 or (acks >= 2 and len(set(core)) == 1)


def classify_interruption(words: str, *, dur_s: float, recent_said: Sequence[str],
                          prev_verdict: tuple[str, float] | None, now: float) -> Verdict:
    toks = tokens(words)
    if not toks:
        # Nothing recognisable (a cough, road noise): the session's false-interruption
        # timer resumes the reply; treat it like a short backchannel for the two-in-3-s rule.
        return "backchannel" if dur_s < BACKCHANNEL_MAX_S else "real"
    core = [w for w in toks if w not in FILLERS] or toks
    single_command = len(core) == 1 and core[0] in COMMAND_WORDS
    if not single_command and _echo_fraction(core, recent_said) >= ECHO_OVERLAP:
        return "echo"
    if _is_stop(toks):
        return "stop"
    if dur_s < BACKCHANNEL_MAX_S and len(core) <= BACKCHANNEL_MAX_TOKENS \
            and not any(w in COMMAND_WORDS for w in core):
        if prev_verdict is not None and prev_verdict[0] == "backchannel" \
                and (now - prev_verdict[1]) <= SECOND_BACKCHANNEL_S:
            return "stop"
        return "backchannel"
    return "real"
