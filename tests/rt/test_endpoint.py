"""TurnPolicy decision table (SPEC §4.3) incl. partial=None, learned EMA, false-end counting."""
import pytest

from dispatch_voice.rt import RtSettings
from dispatch_voice.rt.endpoint import (
    CONTINUATION_WORDS,
    Learned,
    Signals,
    TurnPolicy,
    last_word,
    text_is_unsure,
)
from dispatch_voice.rt.stt_stream import Partial


def partial(tail: str, committed: str = "", punct: bool | None = None) -> Partial:
    if punct is None:
        punct = tail.rstrip().endswith((".", "?", "!"))
    return Partial(committed=committed, tail=tail, stable_words=max(0, len(tail.split()) - 2),
                   punct_final=punct, t_audio=2.0, cost_ms=250)


def sig(p, text, pause=0.3, patience=1.0, committed=""):
    return Signals(p_done=p, partial=None if text is None else partial(text, committed), pause_s=pause,
                   patience=patience)


# ---- the measured rows ----------------------------------------------------- #

def test_confident_end_regardless_of_text():
    d = TurnPolicy().decide(sig(0.99, "What's on my calendar this afternoon?"))
    assert d.action == "end" and d.hold_s == 0 and d.speculate
    d = TurnPolicy().decide(sig(0.95, "and also check whether"))     # P wins over the text
    assert d.action == "end" and d.speculate


def test_unsure_band_with_terminal_punctuation_ends():
    d = TurnPolicy().decide(sig(0.76, "Can you remind me to call the dentist."))
    assert d.action == "end" and d.speculate and d.reason == "terminal_end"


def test_the_misfire_class_holds_even_with_parakeets_full_stop():
    # "I was thinking that maybe we could." scores 0.58 and Parakeet closes it with "."
    d = TurnPolicy().decide(sig(0.58, "I was thinking that maybe we could."))
    assert d.action == "hold" and d.speculate and 0.3 <= d.hold_s <= 1.2
    d = TurnPolicy().decide(sig(0.70, "so maybe we should push the start date back a month and"))
    assert d.action == "hold"


def test_hold_is_scaled_by_patience_and_clamped():
    p = TurnPolicy(RtSettings())
    quick = p.decide(sig(0.6, "we could", patience=0.7)).hold_s
    patient = p.decide(sig(0.6, "we could", patience=1.5)).hold_s
    assert quick == pytest.approx(p.hold_s * 0.7) and patient == pytest.approx(p.hold_s * 1.5)


def test_low_p_with_terminal_text_reprobes_then_ends():
    p = TurnPolicy()
    d = p.decide(sig(0.21, "and also check whether the package has shipped yet?", pause=0.3))
    assert d.action == "reprobe" and d.hold_s == 0.4 and d.speculate is False
    d = p.decide(sig(0.21, "and also check whether the package has shipped yet?", pause=1.0))
    assert d.action == "end"


def test_low_p_mid_clause_listens_and_may_backchannel():
    d = TurnPolicy().decide(sig(0.013, "So I was looking at the plan for the garage conversion again and"))
    assert d.action == "listen" and d.backchannel_ok and not d.speculate


def test_hard_silence_ends_a_listen_row():
    p = TurnPolicy(RtSettings(hard_silence_s=1.4))
    assert p.decide(sig(0.01, "and the", pause=1.39)).action == "listen"
    assert p.decide(sig(0.01, "and the", pause=1.4)).action == "end"
    assert p.decide(sig(0.01, "and the", pause=1.5, patience=1.5)).action == "listen"   # 1.4 × 1.5 = 2.1
    assert p.decide(sig(0.01, "and the", pause=2.1, patience=1.5)).action == "end"


# ---- partial=None (feasibility M1) ------------------------------------------ #

def test_partial_none_rows():
    p = TurnPolicy()
    d = p.decide(sig(0.95, None))
    assert d.action == "end" and d.speculate is False          # no text → nothing to speculate on
    d = p.decide(sig(0.80, None))
    assert d.action == "listen" and d.speculate is False and d.backchannel_ok is False
    d = p.decide(sig(0.2, None))
    assert d.action == "listen"
    assert p.decide(sig(None, None, pause=1.4)).action == "end"   # hard silence still ends it


def test_never_speculates_on_reprobe_or_without_text():
    p = TurnPolicy()
    assert p.decide(sig(0.3, "Yes?")).speculate is False
    assert p.decide(sig(0.99, "")).speculate is False
    assert p.decide(sig(0.99, "   ")).speculate is False


# ---- no Smart Turn model ----------------------------------------------------- #

def test_no_model_text_alone_decides():
    p = TurnPolicy()
    assert p.decide(sig(None, "Call the dentist tomorrow.")).action == "end"
    assert p.decide(sig(None, "Call the dentist tomorrow and")).action == "hold"


# ---- text helpers -------------------------------------------------------------- #

def test_continuation_words_and_last_word():
    assert last_word("we could.") == "could" and "could" in CONTINUATION_WORDS
    assert last_word("the dentist?") == "dentist"
    assert last_word("") == ""
    assert text_is_unsure(partial("we could."))
    assert not text_is_unsure(partial("Thanks, that's all!"))
    assert text_is_unsure(partial("that's all", punct=False))
    assert text_is_unsure(partial("", committed="Hello there"))    # empty tail → committed text decides
    assert not text_is_unsure(partial("", committed="Hello there."))


# ---- learned endpointing ------------------------------------------------------ #

def test_learned_defaults_and_false_end_steps():
    p = TurnPolicy(RtSettings())
    assert p.learned == Learned(probe_ms=250, hold_ms=600)
    p.observe(false_end=True, resume_gap_s=0.8)
    assert p.learned.probe_ms == 300
    assert p.false_ends == 1
    # the floor rose by 0.15 to 0.45; the EMA moved a tenth of the way to 0.8
    assert p.learned.hold_ms == pytest.approx(620, abs=1)
    for _ in range(10):
        p.observe(false_end=True, resume_gap_s=1.2)
    assert p.learned.hold_ms == 1200 and p.learned.probe_ms == 500       # both capped


def test_clean_turns_decay_the_learned_values():
    p = TurnPolicy(RtSettings())
    p.observe(false_end=True, resume_gap_s=None)
    p.observe(false_end=True, resume_gap_s=None)
    hold_up, probe_up = p.learned.hold_ms, p.learned.probe_ms
    assert hold_up == 600 and probe_up == 350      # EMA (0.6) still above the raised floor (0.6)
    for _ in range(3):
        p.observe(false_end=False, resume_gap_s=None)
    assert p.learned.probe_ms == 300
    for _ in range(3):
        p.observe(false_end=False, resume_gap_s=None)
    assert p.learned.probe_ms == 250               # never below the configured probe


def test_ema_of_resume_gaps_moves_the_hold():
    p = TurnPolicy(RtSettings())
    for _ in range(40):
        p.observe(false_end=False, resume_gap_s=1.1)
    assert 1050 <= p.learned.hold_ms <= 1100
    assert p.decide(sig(0.6, "we could")).hold_s == pytest.approx(p.learned.hold_ms / 1000, abs=0.001)


def test_learned_round_trip_and_reset():
    p = TurnPolicy(RtSettings(), learned=Learned(probe_ms=400, hold_ms=900))
    assert p.learned == Learned(probe_ms=400, hold_ms=900)
    assert p.probe_silence_s == 0.4
    p.reset()
    assert p.learned == Learned(probe_ms=250, hold_ms=600)
    assert Learned.from_dict({"probe_ms": 300, "hold_ms": 700}) == Learned(300, 700)
    assert Learned.from_dict({"probe_ms": "x"}) is None and Learned.from_dict(None) is None
    assert Learned.from_dict({"probe_ms": 0, "hold_ms": 5}) is None
    assert Learned(250, 600).as_dict() == {"probe_ms": 250, "hold_ms": 600}
