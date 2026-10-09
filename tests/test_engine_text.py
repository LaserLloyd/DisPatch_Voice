from dispatch_voice.engine import normalise_text, split_speakable


def test_normalise_text():
    assert normalise_text("  hello   there ") == "Hello there."
    assert normalise_text("what?") == "What?"
    assert normalise_text("ha [laugh]") == "Ha [laugh]"
    assert normalise_text("it’s — fine") == "It's, fine."
    assert normalise_text("") == ""


def test_first_sentence_goes_out_alone():
    ready, rest = split_speakable("Sure. I can do that for you right now. And")
    assert ready[0] == "Sure."
    assert ready[1] == "I can do that for you right now."
    assert rest == "And"


def test_later_short_sentences_merge():
    ready, rest = split_speakable("Okay. Yes. Right. That is a longer sentence here.", first=False)
    assert ready == ["Okay. Yes. Right. That is a longer sentence here."]
    assert rest == ""


def test_incomplete_text_waits_until_final():
    ready, rest = split_speakable("no punctuation yet", first=False)
    assert ready == [] and rest == "no punctuation yet"
    ready, rest = split_speakable("no punctuation yet", final=True, first=False)
    assert ready == ["no punctuation yet"] and rest == ""
