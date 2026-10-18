"""heard_prefix → (index, fraction); source_offset on the markdown golden row; HeardMap."""
import pytest

from forge_voice_core.rt.heard import (
    HeardMap,
    Span,
    heard_prefix,
    heard_texts,
    source_offset,
)

GOLDEN = "**Tuesday at 3pm** — see [the map](https://example.com/directions?x=1) for the route. Then call me."
SPEAKABLE_1 = "Tuesday at 3pm, see the map for the route."
SPAN_1 = Span(SPEAKABLE_1, 0, GOLDEN.index(" Then"), GOLDEN[:GOLDEN.index(" Then")])
SPAN_2 = Span("Then call me.", GOLDEN.index("Then"), len(GOLDEN), "Then call me.")


def test_heard_prefix_linear_map():
    spans = [Span("one two three", 0, 13), Span("four five six seven", 14, 33)]
    assert heard_prefix(spans, 1000, 0) == (0, 0.0)
    assert heard_prefix(spans, 1000, 1000) == (1, 1.0)
    assert heard_prefix(spans, 1000, 5000) == (1, 1.0)
    i, f = heard_prefix(spans, 1000, 200)
    assert i == 0 and f == pytest.approx(0.49, abs=0.02)     # 6.4 of 13 chars
    i, f = heard_prefix(spans, 1000, 600)
    assert i == 1 and f == pytest.approx((19.2 - 13) / 19, abs=0.01)
    assert heard_prefix([], 1000, 500) == (0, 0.0)
    assert heard_prefix(spans, 0, 500) == (0, 0.0)


def test_source_offset_lands_on_word_boundaries_outside_markup():
    # Inside the URL of the link → back to the "[" of the link.
    url_at = SPAN_1.source.index("https") + 10
    frac = url_at / len(SPAN_1)
    off = source_offset(SPAN_1, frac, content=GOLDEN)
    assert GOLDEN[off:].startswith("[the map]")
    # Inside the bold run → back to the "**".
    frac = (GOLDEN.index("at 3pm") + 3) / len(SPAN_1)
    off = source_offset(SPAN_1, frac, content=GOLDEN)
    assert off == 0
    # Mid-word in plain text → the start of that word.
    frac = (GOLDEN.index("route") + 2) / len(SPAN_1)
    off = source_offset(SPAN_1, frac, content=GOLDEN)
    assert GOLDEN[off:].startswith("route")
    # Fractions 0 and 1 are the span's ends.
    assert source_offset(SPAN_1, 0.0, content=GOLDEN) == 0
    assert source_offset(SPAN_1, 1.0, content=GOLDEN) == SPAN_1.end


def test_source_offset_uses_span_source_without_content_and_approximates_without_either():
    frac = (SPAN_1.source.index("https") + 10) / len(SPAN_1)
    assert GOLDEN[source_offset(SPAN_1, frac):].startswith("[the map]")
    bare = Span(SPEAKABLE_1, 0, SPAN_1.end)
    off = source_offset(bare, 0.5)
    assert 0 < off < bare.end
    assert source_offset(Span("", 5, 5), 0.5) == 5


def test_heard_texts_caps_and_boundaries():
    heard, unheard = heard_texts([SPAN_1, SPAN_2], 0, 0.5)   # 21 of 42 chars lands inside "the"
    assert heard == "Tuesday at 3pm, see" and unheard == "the map for the route. Then call me."
    long = Span("word " * 100, 0, 500)
    heard, unheard = heard_texts([long, long], 1, 0.5)
    assert len(heard) <= 160 and heard.startswith("…") and len(unheard) <= 240 and unheard.endswith("…")
    assert heard_texts([], 0, 0.5) == ("", "")


def test_heard_map_locate_server_and_phone():
    m = HeardMap()
    m.note(1, message_id="m-9", span=SPAN_1, audio_ms=0, content=GOLDEN)
    m.add_audio(1, 1500)
    m.add_audio(1, 1500)
    m.note(2, message_id="m-9", span=SPAN_2, audio_ms=1000, content=GOLDEN)
    pos = m.locate(1, 1500)
    assert pos.message_id == "m-9" and pos.index == 0 and pos.fraction == pytest.approx(0.5)
    assert GOLDEN[pos.upto:].startswith(("[the map]", "see", "for"))
    assert pos.heard.startswith("Tuesday") and "Then call me." in pos.unheard and not pos.complete
    done = m.locate(2, 5000)
    assert done.complete and done.unheard == "" and done.upto == len(GOLDEN)
    # The phone voice reports ms since onstart; characters at 15/s.
    ph = m.locate(2, 400, source="phone")          # 6 chars of "Then call me." (13 chars → 867 ms)
    assert ph.index == 1 and 0.4 <= ph.fraction <= 0.5
    assert m.locate(7, 100) is None
    assert m.last_seq("m-9") == 2 and [q for q, _ in m.spans_of("m-9")] == [1, 2]
    m.forget(1)
    assert m.last_seq("m-9") == 2 and len(m.spans_of("m-9")) == 1
    m.clear()
    assert m.locate(2, 10) is None


# ---- review T4 #4: markdown-aware openers ------------------------------------------- #

@pytest.mark.parametrize("src, frac, word", [
    ("The file user_data.csv has four hundred rows and we should load it before the meeting starts.", 0.9, "meeting"),
    ("Check if a < b holds first, then run the rest of the pipeline before Tuesday.", 0.9, "Tuesday"),
    ("Run `setup_env` first, then start the server and open the dashboard on your phone.", 0.95, "phone"),
    ("Prices are 5 * 3 * 2 dollars total which is thirty dollars for the three of them.", 0.9, "of them"),
    ("See [the map](https://example.com/a_very/long_path?x=1) and then turn left at the bakery.", 0.5, "[the map]"),
    ("Call <b>Tuesday</b> at three and then <i>wait</i> for the call back from them.", 0.17, "<b>"),
    ("Run `setup env now` first, then start the server and open the dashboard on your phone.", 0.12, "`setup"),
    ("He said **not** before Tuesday, so the plan is to wait and see what happens then.", 0.11, "**not**"),
    ("Use snake_case_names and *never* a lone * in a sentence like this one here.", 0.05, "snake_case_names"),
])
def test_source_offset_markdown_aware_openers(src, frac, word):
    sp = Span(text=src, start=0, end=len(src), source=src)
    off = source_offset(sp, frac)
    assert src[off:].startswith(word), (off, src[off:])
    assert off == 0 or src[off - 1].isspace()


def test_source_offset_still_steps_out_of_open_emphasis():
    src = "He said **not before Tuesday**, so wait."
    sp = Span(text=src, start=0, end=len(src), source=src)
    frac = (src.index("Tuesday") + 2) / len(src)
    assert src[source_offset(sp, frac):].startswith("**not")
    src2 = "Use _emphasis here_ and go on with the rest of the sentence."
    sp2 = Span(text=src2, start=0, end=len(src2), source=src2)
    frac2 = (src2.index("here") + 1) / len(src2)
    assert src2[source_offset(sp2, frac2):].startswith("_emphasis")
