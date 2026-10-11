"""Heard text from the rig's word timings vs the linear map (ADDON-ENGINE §4)."""
from __future__ import annotations

from dispatch_voice.rt.heard import HeardMap, Span, heard_prefix, heard_prefix_words

TEXT = "Your dentist appointment moved to Thursday afternoon."
# A long word early: the linear map and the word timings disagree there.
WORDS = [["Your", 0, 200], ["dentist", 200, 900], ["appointment", 900, 1700], ["moved", 1700, 1950],
         ["to", 1950, 2050], ["Thursday", 2050, 2600], ["afternoon", 2600, 3200]]


def test_the_cut_lands_after_the_last_word_that_finished():
    assert heard_prefix_words(TEXT, WORDS, 0) == 0.0
    assert heard_prefix_words(TEXT, WORDS, 199) == 0.0             # "Your" still playing
    f = heard_prefix_words(TEXT, WORDS, 1000)                      # inside "appointment"
    assert TEXT[:round(f * len(TEXT))].rstrip() == "Your dentist"
    f = heard_prefix_words(TEXT, WORDS, 1960)
    assert TEXT[:round(f * len(TEXT))].rstrip() == "Your dentist appointment moved"
    assert heard_prefix_words(TEXT, WORDS, 3200) == 1.0
    # The linear map at 1000 ms (31 % of the audio) cuts somewhere else, mid-word.
    _, lin = heard_prefix([Span(TEXT, 0, len(TEXT))], 3200, 1000)
    assert TEXT[:round(lin * len(TEXT))].rstrip() != "Your dentist"


def test_timings_that_do_not_line_up_fall_back_to_the_linear_map():
    assert heard_prefix_words(TEXT, WORDS[:3], 1000) is None
    assert heard_prefix_words(TEXT, [["x", "a", "b"]] * 7, 1000) is None
    assert heard_prefix_words("", WORDS, 10) is None


def test_heard_map_uses_word_timings_when_the_engine_has_them():
    hm = HeardMap()
    hm.note(1, message_id="m1", span=Span(TEXT, 0, len(TEXT), source=TEXT), audio_ms=3200)
    linear = hm.locate(1, 1000)
    hm.words_of = lambda seq: WORDS if seq == 1 else None
    worded = hm.locate(1, 1000)
    assert worded.heard == "Your dentist" and worded.upto == len("Your dentist ")
    # At 800 ms "dentist" is still playing: the words say "Your"; the linear map says more.
    assert hm.locate(1, 800).heard == "Your"
    hm.words_of = None
    assert hm.locate(1, 800).heard == "Your dentist" == linear.heard
    assert hm.locate(1, 1000, source="phone").heard == HeardMap(_seqs=hm._seqs).locate(1, 1000, source="phone").heard
