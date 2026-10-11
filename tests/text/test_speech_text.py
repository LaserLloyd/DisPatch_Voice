"""Speech normalisation and role-play: the golden table, the four ordering traps,
each number rule on its own, the emphasis rule and the four RP modes."""
from __future__ import annotations

import pytest

from dispatch_voice.text.speech_text import (
    RP_MODES,
    RpRenderer,
    expand_numbers,
    filter_tags,
    is_cjk_heavy,
    normalise,
    rp_render,
    segment,
    strip_markdown,
    strip_symbols,
)

MODES = RP_MODES

# --------------------------------------------------------------------------- #
# Golden table: the 13 hard sentences of the design prototype (normalise, with
# a tag-capable voice) — plus the rows the experience review added.
# --------------------------------------------------------------------------- #

GOLDEN = [
    ("You have **two things** this afternoon: a dentist at 3:45pm and a call at 16:30.",
     ("You have two things this afternoon: a dentist at three forty-five p m and a call at four "
     "thirty p m.")),
    (("That's $1,250.50, due 2026-10-15 — about 15% of the $8,400 total. Call 090-1234-5678 if it "
     "bounces."),
     ("That's one thousand two hundred fifty dollars and fifty cents, due October fifteenth, twenty "
     "twenty-six — about fifteen percent of the eight thousand four hundred dollars total. Call zero "
     "nine zero, one two three four, five six seven eight if it bounces.")),
    ("Oct 9th was the 3rd session this week; the waist is 42.9 in and bench went up 10 lb to 185 lbs.",
     ("October ninth was the third session this week; the waist is forty-two point nine in and bench "
     "went up ten pounds to one hundred eighty-five pounds.")),
    ("The deploy ran in 1.8 s per turn (1,000 turns, 2.5 GB RAM) on the Ryzen AI 9 365 at 35°C.",
     ("The deploy ran in one point eight seconds per turn (one thousand turns, two point five "
     "gigabytes RAM) on the Ryzen AI 9 365 at thirty-five degrees.")),
    (("Here's the plan:\n- check the SOFA dates\n- email HR\n- see https://example.com/offer-letter "
     "and alex@example.com"),
     ("Here's the plan: check the SOFA dates. email HR. see example dot com and alex at example dot "
     "com")),
    ("Hmm 🤔 good question!! 🎉 Let me think… ✨", "Hmm good question!! Let me think..."),
    ("The API returns JSON; NASA and the USAF use it. OK?",
     "The API returns JSON; NASA and the USAF use it. OK?"),
    (("*wags tail* Yes ma'am. *pants* On it. \"Where do you want it, boss?\" she asks, *tilting her "
     "head*."),
     ("*wags tail* Yes ma'am. *pants* On it. \"Where do you want it, boss?\" she asks, *tilting her "
     "head*.")),
    (("\"Missed four days,\" Rae says. \"Today is gym day.\" *She slides the plan across.* Then: "
     "\"One session. That's all.\""),
     ("\"Missed four days,\" Rae says. \"Today is gym day.\" *She slides the plan across.* Then: "
     "\"One session. That's all.\"")),
    ("[chuckle] Octopuses have three hearts. [laugh] Anyway. [whisper] secret [sigh]",
     "[chuckle] Octopuses have three hearts. [laugh] Anyway. secret [sigh]"),
    ("会議は3時45分に始まります。¥12,500です。", "会議は3時45分に始まります。¥12,500です。"),
    ("Sam said: ```python\nprint('hi')\n``` and that's it. Range 10-20 minutes, in the 1990s.",
     ("Sam said: The code is in the chat. and that's it. Range 10 to 20 minutes, in the nineteen "
     "nineties.")),
    (("*She pauses, considering* — \"Honestly? Take the deload.\" — then *grins* and the rest is "
     "unclosed *"),
     ("*She pauses, considering* — \"Honestly? Take the deload.\" — then *grins* and the rest is "
     "unclosed *")),
]


@pytest.mark.parametrize("raw,spoken", GOLDEN, ids=[f"g{i:02d}" for i in range(len(GOLDEN))])
def test_golden_table(raw, spoken):
    assert normalise(raw, tags_ok=True) == spoken


def test_golden_rp_rows_under_every_mode():
    doxy = normalise(GOLDEN[7][0], tags_ok=False)
    assert rp_render(doxy, "all") == [
        ("character", ("wags tail. Yes ma'am. pants. On it. Where do you want it, boss? she asks, "
                      "tilting her head."))]
    assert rp_render(doxy, "skip_actions") == [
        ("character", "Yes ma'am. On it. Where do you want it, boss? she asks.")]
    assert rp_render(doxy, "dialogue") == [("character", "Where do you want it, boss?")]
    assert rp_render(doxy, "dialogue_narrator") == [
        ("narrator", "wags tail. Yes ma'am. pants. On it."),
        ("character", "Where do you want it, boss?"),
        ("narrator", "she asks, tilting her head.")]

    trainer = normalise(GOLDEN[8][0], tags_ok=False)
    assert rp_render(trainer, "skip_actions") == [
        ("character", "Missed four days, Rae says. Today is gym day. Then: One session. That's all.")]
    assert rp_render(trainer, "dialogue") == [
        ("character", "Missed four days. Today is gym day. One session. That's all.")]
    assert rp_render(trainer, "dialogue_narrator") == [
        ("character", "Missed four days."), ("narrator", "Rae says."),
        ("character", "Today is gym day."), ("narrator", "She slides the plan across. Then."),
        ("character", "One session. That's all.")]

    unclosed = normalise(GOLDEN[12][0], tags_ok=False)
    assert rp_render(unclosed, "dialogue") == [("character", "Honestly? Take the deload.")]
    assert rp_render(unclosed, "dialogue_narrator")[0] == ("narrator", "She pauses, considering.")


# ---- the emphasis rows (experience H2) ------------------------------------ #

@pytest.mark.parametrize("mode", MODES)
def test_emphasis_is_identical_under_all_four_modes(mode):
    assert rp_render(normalise("It's *not* about the money.", tags_ok=False), mode) == [
        ("character", "It's not about the money.")]


@pytest.mark.parametrize("text,spoken", [
    ("That's *mine*.", "That's mine."),                 # one word at the end: emphasis
    ("un*believ*able, honestly.", "unbelievable, honestly."),   # word chars on both sides
    ("*Not* again.", "Not again."),                     # capitalised word running on
    ("I *really* do.", "I really do."),
])
@pytest.mark.parametrize("mode", MODES)
def test_one_word_emphasis_stays_in_the_sentence(text, spoken, mode):
    assert rp_render(text, mode) == [("character", spoken)]


# review-T5 #5: the H2 failure one comma to the left. A single word after a
# comma or dash is emphasis unless it is a known gesture.
@pytest.mark.parametrize("text,spoken", [
    ("Well, *that* was fun.", "Well, that was fun."),
    ("Honestly, *everything* went wrong.", "Honestly, everything went wrong."),
    ("I mean, *wow*.", "I mean, wow."),
    ("It was — *amazing*.", "It was — amazing."),
])
@pytest.mark.parametrize("mode", MODES)
def test_one_word_emphasis_after_a_comma_stays_in_the_sentence(text, spoken, mode):
    assert rp_render(normalise(text, tags_ok=False), mode) == [("character", spoken)]


@pytest.mark.parametrize("text,want", [
    ("*laughs* That's a *great* idea, *really*.", {
        "all": [("character", "laughs. That's a great idea, really.")],
        "skip_actions": [("character", "That's a great idea, really.")],
        "dialogue": [("character", "That's a great idea, really.")],
        "dialogue_narrator": [("narrator", "laughs."), ("character", "That's a great idea, really.")]}),
    ("*stretches* Okay, *now* we're talking.", {
        "all": [("character", "stretches. Okay, now we're talking.")],
        "skip_actions": [("character", "Okay, now we're talking.")],
        "dialogue": [("character", "Okay, now we're talking.")],
        "dialogue_narrator": [("narrator", "stretches."), ("character", "Okay, now we're talking.")]}),
    ("Okay, *sighs*, fine.", {                           # a known gesture is still an action
        "all": [("character", "Okay, sighs, fine.")],
        "skip_actions": [("character", "Okay, fine.")],
        "dialogue": [("character", "Okay, fine.")],
        "dialogue_narrator": [("character", "Okay."), ("narrator", "sighs."), ("character", "fine.")]}),
])
@pytest.mark.parametrize("mode", MODES)
def test_actions_and_emphasis_in_one_sentence_under_every_mode(text, want, mode):
    assert rp_render(normalise(text, tags_ok=False), mode) == want[mode]


def test_a_two_word_star_span_is_still_an_action():
    src = '*She leans in.* "Tell me more."'
    assert segment(src) == [("action", "She leans in."), ("quote", "Tell me more.")]
    assert rp_render(src, "skip_actions") == [("character", "Tell me more.")]
    assert rp_render(src, "dialogue_narrator") == [("narrator", "She leans in."),
                                                   ("character", "Tell me more.")]


def test_a_one_word_action_at_a_sentence_start_is_an_action():
    assert segment("*sighs* Fine.") == [("action", "sighs"), ("prose", "Fine.")]
    assert segment("Okay. *pants* On it.") == [("prose", "Okay."), ("action", "pants"),
                                               ("prose", "On it.")]


# ---- the four ordering traps (design §0 row 5) ---------------------------- #

def test_trap_url_needs_an_alphabetic_tld_so_decimals_survive():
    assert normalise("It used 2.5 GB, see example.com.", tags_ok=False) == \
        "It used two point five gigabytes, see example dot com."
    assert "dot" not in normalise("Version 3.12 is out.", tags_ok=False)


def test_trap_single_letter_units_need_a_space():
    assert normalise("Back in the 1990s it took 5 s.", tags_ok=False) == \
        "Back in the nineteen nineties it took five seconds."
    assert normalise("A 2m cable.", tags_ok=False) == "A 2m cable."


def test_trap_money_does_not_swallow_the_next_space():
    assert normalise("It was $20 and change.", tags_ok=False) == "It was twenty dollars and change."
    assert normalise("$5 more", tags_ok=False) == "five dollars more"


def test_trap_cjk_heavy_text_keeps_its_digits():
    assert normalise("会議は3時45分に始まります。", tags_ok=False) == "会議は3時45分に始まります。"
    assert is_cjk_heavy("会議は3時") and not is_cjk_heavy("Meeting at 3 (会議)")


# ---- each number rule on its own ------------------------------------------ #

@pytest.mark.parametrize("raw,spoken", [
    ("$5", "five dollars"), ("$1", "one dollar"), ("$0.99", "ninety-nine cents"),
    ("$2k budget", "two thousand dollars budget"), ("$1.5 million", "one point five million dollars"),
    ("€5", "five euros"), ("£3.50", "three pounds fifty"), ("¥12,500", "twelve thousand five hundred yen"),
    ("15%", "fifteen percent"), ("2.5%", "two point five percent"),
    ("3:45pm", "three forty-five p m"), ("16:30", "four thirty p m"), ("12:00", "twelve o'clock"),
    ("9:00 a.m.", "nine a m."), ("at 9 a.m. today", "at nine a m today"), ("7:05", "seven oh five"), ("3pm", "three p m"), ("25:00", "25:00"),
    ("2026-10-15", "October fifteenth, twenty twenty-six"), ("Oct 9th", "October ninth"),
    ("October 9, 2026", "October ninth, twenty twenty-six"), ("9 October", "the ninth of October"),
    ("090-1234-5678", "zero nine zero, one two three four, five six seven eight"),
    ("3rd", "third"), ("21st", "twenty-first"), ("112th", "one hundred twelfth"),
    ("10 km", "ten kilometres"), ("1 km", "one kilometre"), ("5 m away", "five metres away"),
    ("1.8 s", "one point eight seconds"), ("-5°C", "minus five degrees"),
    ("72°F", "seventy-two degrees Fahrenheit"), ("100 mph", "one hundred miles an hour"),
    ("10-20", "10 to 20"), ("1,000", "one thousand"), ("42.9", "forty-two point nine"),
    ("80s", "eighties"), ("2020s", "twenty twenties"), ("2026", "twenty twenty-six"),
    ("2005", "two thousand five"), ("1905", "nineteen oh five"), ("12345", "twelve thousand three hundred forty-five"),
    ("RTX 5090", "RTX fifty ninety"), ("10", "10"), ("Room 101", "Room 101"),
    ("v2.1.0", "v2.1.0"), ("203.0.113.4", "203.0.113.4"), ("#1 fan", "number 1 fan"),
    ("~5 minutes", "about 5 minutes"), ("A & B", "A and B"), ("24/7", "twenty-four seven"),
    ("2x faster", "2 times faster"), ("it was -3", "it was minus 3"),
    ("Mar. 5", "March fifth"), ("5 Sept 2024", "the fifth of September, twenty twenty-four"),
    ("Sep 3", "September third"), ("1,234.5 dollars", "one thousand two hundred thirty-four point five dollars"),
    ("1234.5", "one thousand two hundred thirty-four point five"),
    # review-T5 #4: a name that starts like a month is never a date
    ("I told Marcus 5 times.", "I told Marcus 5 times."), ("Tell Julia 3 things.", "Tell Julia 3 things."),
    ("He led 5 Marines into town.", "He led 5 Marines into town."), ("Nova 5 is out.", "Nova 5 is out."),
    ("Junior 5 left.", "Junior 5 left."), ("Decent 2 options.", "Decent 2 options."),
    ("Augustus 4 won.", "Augustus 4 won."), ("Octavia 2 days.", "Octavia 2 days."),
])
def test_number_rules(raw, spoken):
    assert normalise(raw, tags_ok=False) == spoken


def test_expand_numbers_leaves_acronyms_and_small_integers():
    assert expand_numbers("The API on 3 boxes") == "The API on 3 boxes"


# ---- markdown, tags, symbols ------------------------------------------------ #

def test_markdown_and_directives():
    s = normalise("Nice! :react:happy: 🎉 [[media:/tmp/x.png|cap]] ![a](/media/1) [the guide](https://x.io/a)",
                  tags_ok=False)
    assert s == "Nice! the guide"
    assert normalise("## Plan\n- milk\n- eggs\n1. call mum", tags_ok=False) == "Plan. milk. eggs. call mum"
    assert normalise("| Day | Time |\n|---|---|\n| Mon | 9am |", tags_ok=False) == \
        "Day, Time. Mon, nine a m."
    assert normalise("Use `uv sync` and <b>this</b> _now_.", tags_ok=False) == "Use uv sync and this now."
    assert normalise("Run:\n```\nrm -rf x\n```", tags_ok=False) == "Run: The code is in the chat."


def test_all_emoji_or_punctuation_is_nothing_to_say():
    assert normalise("👍🎉", tags_ok=False) == ""
    assert normalise("— … —", tags_ok=False) == ""
    assert normalise("", tags_ok=True) == ""


@pytest.mark.parametrize("tags_ok", [True, False])
def test_tags_kept_only_for_a_tag_capable_voice(tags_ok):
    out = filter_tags("[Laugh] ok [whisper] then [clear throat] [neutral]", tags_ok=tags_ok)
    words = out.split()
    assert ("[laugh]" in out) is tags_ok and ("[clear throat]" in out) is tags_ok
    assert "[whisper]" not in out and "[neutral]" not in out and "ok" in words


def test_strip_symbols_keeps_scripts_and_drops_pictographs():
    assert strip_symbols("नमस्ते مرحبا").split() == ["नमस्ते", "مرحبا"]   # combining marks kept
    assert strip_symbols("👍🏽 ok ➡️ 1️⃣").split() == ["ok", "1"]
    assert strip_markdown("**bold** and __under__ and ~~gone~~") == "bold and under and gone"


# ---- segment ---------------------------------------------------------------- #

def test_segment_kinds_and_unclosed_openers():
    assert segment('He said "hi there') == [("prose", "He said"), ("quote", "hi there")]
    assert segment("She *grins and waves") == [("prose", "She"), ("action", "grins and waves")]
    assert segment("“Typographic” and «guillemets» and „German“") == [
        ("quote", "Typographic"), ("prose", "and"), ("quote", "guillemets"), ("prose", "and"),
        ("quote", "German")]
    # outermost quotes only; ** is markdown, never an action
    assert segment('"He said “no” to me," she said.') == [
        ("quote", "He said “no” to me,"), ("prose", "she said.")]
    assert segment("**Bold** words here.") == [("prose", "**Bold** words here.".replace("*", ""))]
    # an inch mark never opens a quotation; a spaced star is a literal
    assert segment('The waist is 42.9" now and 3 * 4 is twelve.') == [
        ("prose", 'The waist is 42.9" now and 3 4 is twelve.')]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("text", [
    "Plain prose, nothing special. Two sentences.",
    "Numbers: three forty-five p m.",
    "Isn't it fine? Yes.",
])
def test_non_rp_text_is_the_identity_under_every_mode(text, mode):
    assert rp_render(text, mode) == [("character", text)]


def test_mode_names_from_the_prototype_map_to_the_contract_names():
    t = '*nods* "Fine." she says.'
    assert rp_render(t, "quotes") == rp_render(t, "dialogue") == [("character", "Fine.")]
    assert rp_render(t, "quotes+narrator") == rp_render(t, "dialogue_narrator")
    assert rp_render(t, "nonsense") == rp_render(t, "all")


def test_dialogue_without_quotes_drops_actions_but_speaks_the_rest():
    assert rp_render("*nods* Sure, let's go.", "dialogue") == [("character", "Sure, let's go.")]
    assert rp_render("*nods* Sure, let's go.", "dialogue_narrator") == [
        ("narrator", "nods."), ("character", "Sure, let's go.")]


def test_rp_renderer_remembers_dialogue_across_sentences():
    r = RpRenderer("dialogue_narrator")
    assert r.render('"Ready?" she asks.') == [("character", "Ready?"), ("narrator", "she asks.")]
    # a later sentence without quotes is narration in a reply that has dialogue
    assert r.render("She picks up the bag.") == [("narrator", "She picks up the bag.")]
    r2 = RpRenderer("dialogue")
    r2.render('"Ready?" she asks.')
    assert r2.render("She picks up the bag.") == []


def test_an_all_action_sentence_says_nothing_when_actions_are_skipped():
    assert rp_render("*She walks away.*", "skip_actions") == []
    assert rp_render("*She walks away.*", "dialogue_narrator") == [("narrator", "She walks away.")]


def test_an_echoed_app_note_is_never_read_out():
    from dispatch_voice.text.persona import build
    note = build(None, backend="gateway", profile_tags=False, style="normal", follow_up=True,
                 interrupted=None, extension_persona=None).trailer
    assert normalise(note + " Anyway, boss, ready?", tags_ok=True) == "Anyway, boss, ready?"
    assert normalise("Sure. [voice chat] Let's go.", tags_ok=False) == "Sure. Let's go."
    assert normalise("(app note, not from the user: the reply is read", tags_ok=False) == ""


def test_an_echoed_interruption_note_with_brackets_inside_is_dropped_whole():
    """review-T5 #14: the quoted heard text may itself contain ")"."""
    from dispatch_voice.text.persona import build
    note = build(None, backend="gateway", profile_tags=False, style="normal", follow_up=True,
                 interrupted={"heard": "the plan (part one) is", "unheard": "then (maybe) we go"},
                 extension_persona=None).trailer
    assert normalise(note + " Anyway, boss, ready?", tags_ok=True) == "Anyway, boss, ready?"
    assert normalise("Sure thing (really).", tags_ok=False) == "Sure thing (really)."


@pytest.mark.parametrize("src,want", [
    ("The party is on 12/24/2025.", "The party is on December twenty-fourth, twenty twenty-five."),
    ("Due 1/5/26 at 3pm.", "Due January fifth, twenty twenty-six at three p m."),
    ("See you 24/12/2025!", "See you December twenty-fourth, twenty twenty-five!"),     # day/month
    ("It shipped 2025/12/24.", "It shipped December twenty-fourth, twenty twenty-five."),
    ("Card expires 12/2025.", "Card expires December twenty twenty-five."),
    ("Add 3/4 cup of flour.", "Add 3/4 cup of flour."),                                   # a fraction
    ("Open 24/7.", "Open twenty-four seven."),
    ("Pay 13/45/2025.", "Pay 13/45/2025."),                       # not a date: left whole, not half read
])
def test_numeric_slash_dates(src, want):
    """FR-works 8: "12/24/2025" became "12/24/twenty twenty-five", which Kokoro read
    as "12 slash 24 slash 2025"."""
    assert normalise(src, tags_ok=False) == want


# --- 1.0.1: a local phone number, ASCII emoticons, a conjunction left by a dropped action ---

@pytest.mark.parametrize("raw, spoken", [
    ("Call 555-0123.", "Call five five five, zero one two three."),
    ("Call 555-1234 now.", "Call five five five, one two three four now."),
    ("My number is 555-1230.", "My number is five five five, one two three zero."),
    ("Between 500-1000 people.", "Between 500 to one thousand people."),
    ("Pages 10-20.", "Pages 10 to 20."),
    ("Call 555-123-4567.", "Call five five five, one two three, four five six seven."),
])
def test_a_two_group_phone_number_is_read_digit_by_digit_not_as_a_range(raw, spoken):
    assert normalise(raw, tags_ok=False) == spoken


@pytest.mark.parametrize("raw, spoken", [
    ("3km run :)", "three kilometres run"),
    ("Nice one ;-)", "Nice one"),
    ("See you :D.", "See you."),
    ("Wow :P that was close.", "Wow that was close."),
    ("Love you <3", "Love you"),
    ("Hmm :/ not sure.", "Hmm not sure."),
    # look-alikes that are not emoticons
    ("Time is 10:30.", "Time is ten thirty."),
    ("Ratio 8:3 is fine.", "Ratio 8:3 is fine."),
    ("Note:D is next.", "Note:D is next."),
])
def test_ascii_emoticons_are_not_spoken(raw, spoken):
    assert normalise(raw, tags_ok=False) == spoken


@pytest.mark.parametrize("src, want", [
    ('Then she says "let\'s go" and *smiles warmly*.', "Then she says let's go."),
    ("She laughs but *looks away*.", "She laughs."),
    ("He nods, then *walks to the door*.", "He nods."),
    ("Go. And *she walks off slowly*.", "Go."),
    ("Wait, *she sighs heavily*!", "Wait!"),
    ("He waits, *tilting his head*.", "He waits."),
])
def test_skipping_an_action_leaves_no_dangling_conjunction(src, want):
    assert rp_render(normalise(src, tags_ok=False), "skip_actions") == [("character", want)]


def test_speak_all_keeps_the_conjunction_with_the_action():
    src = 'Then she says "let\'s go" and *smiles warmly*.'
    assert rp_render(normalise(src, tags_ok=False), "all") == [
        ("character", "Then she says let's go and smiles warmly.")]
