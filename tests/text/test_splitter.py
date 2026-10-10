"""SentenceSplitter: when a sentence is ready to speak, and where it sits in the source.

The first block is the pre-split suite (DisPatch tests/test_voice_tts.py) ported
onto Span results; the rest covers v3: role-play holds, caps and source spans."""
from __future__ import annotations

import random

import pytest

from dispatch_voice.text import Span
from dispatch_voice.text.speech_text import SentenceSplitter, normalise


def texts(spans: list[Span]) -> list[str]:
    return [s.text for s in spans]


def stream(text: str, step: int = 3, **kw) -> list[str]:
    """Feed `text` a few characters at a time, like a token stream."""
    sp = SentenceSplitter(min_words=1, **kw)
    out: list[Span] = []
    for i in range(0, len(text), step):
        out += sp.push(text[i:i + step])
    return texts(out + sp.flush())


def assert_spans_cover(source: str, spans: list[Span]) -> None:
    """Each span is its own source slice, in order, with only whitespace between."""
    pos = 0
    for s in spans:
        assert source[s.start:s.end] == s.text and len(s) == s.end - s.start
        assert s.start >= pos and not source[pos:s.start].strip(), (pos, s)
        assert s.text == s.text.strip() and s.text
        pos = s.end
    assert not source[pos:].strip()


# ---- the pre-split behaviour ----------------------------------------------- #

def test_sentences_come_out_as_soon_as_complete():
    sp = SentenceSplitter(min_words=1)
    assert texts(sp.push("Sure. I can")) == ["Sure."]
    assert texts(sp.push(" do that! Next")) == ["I can do that!"]
    assert texts(sp.flush()) == ["Next"]


def test_streamed_and_whole_text_split_identically():
    text = "First one. Second one? Third one! And a tail"
    want = ["First one.", "Second one?", "Third one!", "And a tail"]
    assert stream(text, 1) == want and stream(text, 5) == want


def test_abbreviations_and_initials_do_not_end_a_sentence():
    assert stream("Ask Dr. Smith about it e.g. tomorrow. Then rest.") == [
        "Ask Dr. Smith about it e.g. tomorrow.", "Then rest."]
    assert stream("J. R. R. Tolkien wrote it. Yes.") == ["J. R. R. Tolkien wrote it.", "Yes."]


def test_a_quoted_no_ends_a_sentence_but_no_dot_number_does_not():
    """review-T5 #12: "No." closed by a quotation mark is a word, not "No. 5"."""
    text = 'She said: "Yes." Then: "No." Then nothing happened. See No. 5 please. And no. Okay.'
    want = ['She said: "Yes."', 'Then: "No."', "Then nothing happened.", "See No. 5 please.",
            "And no.", "Okay."]
    assert stream(text, 1) == want and stream(text, 4) == want and stream(text, 200) == want


def test_decimals_stay_whole():
    assert stream("It costs 3.50 today. Okay.") == ["It costs 3.50 today.", "Okay."]


def test_open_code_fence_is_held_back():
    sp = SentenceSplitter(min_words=1)
    assert texts(sp.push("Run this. ```\nx = 1. y = 2. ")) == ["Run this."]
    assert sp.push("z = 3.\n") == []
    assert texts(sp.push("```\nThat is all. ")) == ["```\nx = 1. y = 2. z = 3.\n```\nThat is all."]


def test_open_directive_is_held_back():
    sp = SentenceSplitter(min_words=1)
    assert texts(sp.push("Look. [[media:/a. b")) == ["Look."]
    assert texts(sp.push(".png|x]] Here it is. ")) == ["[[media:/a. b.png|x]] Here it is."]


def test_long_first_clause_is_cut_at_a_comma_for_latency():
    text = ("Well, I looked at the calendar for the whole week and the traffic report too, "
            "and everything is fine")
    first = texts(SentenceSplitter(first_soft_chars=80).push(text))
    assert first and first[0].endswith(",") and len(first[0]) <= 80


def test_paragraph_and_list_breaks_end_sentences():
    assert stream("Shopping list\n\nmilk\n- eggs\n- bread") == ["Shopping list", "milk", "- eggs", "- bread"]


def test_tiny_fragments_wait_for_the_next_sentence():
    sp = SentenceSplitter()            # min_words=4
    assert sp.push("Sure. ") == []
    got = sp.push("I can do that today. Ok")
    assert texts(got) == ["Sure. I can do that today."] and got[0].start == 0
    assert texts(sp.flush()) == ["Ok"]


def test_a_short_whole_reply_still_comes_out_on_flush():
    sp = SentenceSplitter()
    assert sp.push("Yes.") == [] and texts(sp.flush()) == ["Yes."]


def test_flush_on_empty_is_empty():
    assert SentenceSplitter().flush() == []


def test_a_long_first_sentence_starts_with_its_first_clause():
    sp = SentenceSplitter()
    out = texts(sp.push("You have two things this afternoon: a dentist appointment at three, and a "
                        "call with the school at four thirty. Want me to remind you? ") + sp.flush())
    assert out[0] == "You have two things this afternoon:"
    assert out[1].startswith("a dentist appointment") and out[1].endswith("thirty.")
    sp = SentenceSplitter()
    got: list[Span] = []
    for w in ["You", "have", "two", "things", "this", "afternoon:", "a", "dentist", "appointment", "at", "three"]:
        got += sp.push(w + " ")
    assert got and got[0].text == "You have two things this afternoon:"
    assert texts(SentenceSplitter().push("Sure, it is sunny. ")) == ["Sure, it is sunny."]


# ---- role-play holds (v3) -------------------------------------------------- #

def test_an_open_action_is_held_until_it_closes():
    sp = SentenceSplitter(min_words=1)
    assert texts(sp.push("Okay. *She walks to the door. Then")) == ["Okay."]
    assert texts(sp.push(" stops.* Right. ")) == ["*She walks to the door. Then stops.*", "Right."]


def test_an_open_quote_is_held_until_it_closes():
    sp = SentenceSplitter(min_words=1)
    assert texts(sp.push('She says, "Today is gym day. You know it.')) == []
    assert texts(sp.push(' No excuses." Fine. ')) == [
        'She says, "Today is gym day. You know it. No excuses."', "Fine."]


def test_a_sentence_never_ends_inside_a_closed_quote_but_does_at_its_closer():
    src = ('"Missed four days," Rae says. "Today is gym day. You know it. No excuses." '
           '*She slides the plan across.* Then: "One session. That\'s all."')
    want = ['"Missed four days," Rae says.', '"Today is gym day. You know it. No excuses."',
            "*She slides the plan across.*", 'Then: "One session. That\'s all."']
    for step in (1, 2, 5, 1000):
        assert stream(src, step) == want


def test_an_inch_mark_or_a_spaced_star_never_holds():
    assert stream('The waist is 42.9" now. Bench went up. 3 * 4 is twelve. Done.') == [
        'The waist is 42.9" now.', "Bench went up.", "3 * 4 is twelve.", "Done."]


def test_an_unclosed_action_stops_holding_at_its_cap():
    sp = SentenceSplitter(min_words=1, action_cap=40)
    out = sp.push("Fine. *She walks to the door. She opens it slowly and looks out. Then more. ")
    # past the cap the open star no longer holds: sentences come out again
    assert texts(out)[0] == "Fine." and len(out) >= 2


def test_an_unclosed_quote_holds_up_to_800_chars():
    sp = SentenceSplitter(min_words=1)
    body = "One more line of the speech. " * 20             # ~580 chars, still under the cap
    assert texts(sp.push('Fine. "' + body)) == ["Fine."]
    more = sp.push("And more of it now. " * 15)                # past 800: released
    assert more and more[0].text.startswith('"One more line')


# ---- source spans (A7) ----------------------------------------------------- #

SOURCES = [
    "**Two** things: see [the guide](https://example.com/x). Then rest, and a **bold** close.",
    "## Plan\n- milk\n- eggs\n\n1. call mum\n2. done\n\nThat's it!",
    "*wags tail* Yes ma'am. *pants* On it. \"Where do you want it, boss?\" she asks, *tilting her head*.",
    "Run this:\n```\nx = 1. y = 2.\n```\nThen [[media:/a. b.png|x]] here. Ok",
    "Sure.  \n\n  I can do that today.   Ok   ",
    ("A very long first sentence that keeps going without any full stop at all, really, "
    "it goes on and on with commas, clauses, and more words until the hard cap must cut it "
    "somewhere sensible because nobody wants to wait for the whole thing to arrive before "
    "hearing anything at all from the bot, which would be awful"),
]


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("seed", range(6))
def test_spans_reconstruct_the_source_for_any_chunking(source, seed):
    rng = random.Random(seed)
    sp = SentenceSplitter()
    out: list[Span] = []
    i = 0
    while i < len(source):
        n = rng.randint(1, 12)
        out += sp.push(source[i:i + n])
        i += n
    out += sp.flush()
    assert_spans_cover(source, out)
    assert sp.source == source


@pytest.mark.parametrize("source", SOURCES)
def test_split_equals_streaming(source):
    whole = SentenceSplitter().split(source)
    assert_spans_cover(source, whole)
    sp = SentenceSplitter()
    streamed = [s for ch in source for s in sp.push(ch)] + sp.flush()
    assert [(s.start, s.end) for s in streamed] == [(s.start, s.end) for s in whole]


def test_markdown_row_spans_point_into_the_stored_markdown():
    """Heard tracking cuts the STORED markdown: a span covers the raw markup, so a
    cut at a span boundary never lands inside `**…**` or a link."""
    src = "**Two** things: see [the guide](https://example.com/x). Then rest."
    spans = SentenceSplitter(min_words=1).split(src)
    assert texts(spans) == ["**Two** things: see [the guide](https://example.com/x).", "Then rest."]
    first = spans[0]
    assert src[:first.end].endswith("(https://example.com/x).")
    assert normalise(first.text, tags_ok=False) == "Two things: see the guide."
    assert len(normalise(first.text, tags_ok=False)) < len(first)   # why heard maps by fraction
    assert src[spans[1].start:] == "Then rest."


def test_split_resets_the_splitter():
    sp = SentenceSplitter(min_words=1)
    sp.push("Leftover text without end")
    assert texts(sp.split("New one. Two.")) == ["New one.", "Two."]
