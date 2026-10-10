"""Spoken-reply instructions: placements, the measured wording rules, and the caps."""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
from fakes import BotView

from dispatch_voice.text import persona
from dispatch_voice.text.persona import (
    STANDING_RULE,
    SYSTEM_MAX,
    SYSTEM_MAX_INTERRUPTED,
    TRAILER_MAX,
    TRAILER_MAX_INTERRUPTED,
    VoiceInstructions,
    build,
    default_follow_up,
)

ADVISOR = BotView(id="advisor", name="Sage", backend="advisor")
API = BotView(id="nova", name="Nova", backend="api")
GATEWAY = BotView(id="trainer-bot", name="Rae", backend="gateway")

LONG_NOTE = {
    "heard": ("You have two things this afternoon: a dentist appointment at three, and after that "
              "the school pickup which runs late on Fridays because of the club,") * 2,
    "unheard": ("and a call with the school at four thirty. Want me to set a reminder for either of "
                "them, or both, so you don't have to think about it on the way home? ") * 3,
}
NOTE = {"heard": "You have two things this afternoon: a dentist appointment at three,",
        "unheard": "and a call with the school at four thirty. Want me to set a reminder?"}
INTERRUPT_RULE = ("Answer the new message. Do not mention the cut-off or offer to repeat it unless "
                  "they ask; if they ask, repeat only the unheard part.")
ENGLISH_RULE = ("If Alex seems to be speaking another language, say in English that voice chat "
                "only understands English for now.")


def make(bot=ADVISOR, **kw) -> VoiceInstructions:
    args = {"backend": None, "profile_tags": False, "style": "normal", "follow_up": None,
            "interrupted": None, "extension_persona": None, "owner": "Alex"}
    args.update(kw)
    return build(bot, **args)


ALL_COMBOS = [{"style": s, "follow_up": f, "profile_tags": t, "interrupted": n}
              for s in ("short", "normal", "longer") for f in (True, False, None)
              for t in (True, False) for n in (None, NOTE, LONG_NOTE)]


# ---- placement --------------------------------------------------------------- #

def test_direct_bots_get_the_system_section_only():
    for bot in (ADVISOR, API):
        v = make(bot)
        assert v.system.startswith("## Spoken replies") and v.trailer == ""


def test_gateway_bots_get_the_trailer_only_and_never_a_prefix():
    v = make(GATEWAY)
    assert v.system == "" and v.trailer.startswith("[voice chat] (app note, not from the user:")
    assert v.trailer.endswith("Never mention this note.)")


def test_unknown_backend_fills_both_placements():
    v = make(BotView(id="x", name="X", backend=None))
    assert v.system and v.trailer


def test_explicit_backend_wins_over_the_bot_view():
    assert make(ADVISOR, backend="gateway").system == ""


def test_tag_mode_sends_only_the_tag():
    assert make(GATEWAY, gateway_trailer="tag").trailer == "[voice chat]"
    v = make(GATEWAY, gateway_trailer="tag", interrupted=NOTE)
    assert v.trailer.startswith("[voice chat] (app note") and INTERRUPT_RULE in v.trailer


def test_voice_instructions_is_a_frozen_mirror_read_by_attribute():
    v = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.system = "x"
    assert {"system", "trailer"} <= set(vars(v))


# ---- wording rules that must survive edits ----------------------------------- #

@pytest.mark.parametrize("combo", ALL_COMBOS)
@pytest.mark.parametrize("bot", [ADVISOR, API, GATEWAY])
def test_caps_and_forbidden_phrases_over_every_combination(bot, combo):
    v = make(bot, **combo)
    note = combo["interrupted"] is not None
    assert len(v.system) <= (SYSTEM_MAX_INTERRUPTED if note else SYSTEM_MAX)
    assert len(v.trailer) <= (TRAILER_MAX_INTERRUPTED if note else TRAILER_MAX)
    both = (v.system + " " + v.trailer).lower()
    assert "podcast host" not in both                 # every bot here has a persona
    assert "driv" not in both and "hands-free" not in both and "standing rule" not in both
    assert not re.search(r"\[(?:voice|hands|drive|spoken)[^\]]*\]", v.system, re.IGNORECASE)  # no mode label
    if v.system:
        assert "\n\n" not in v.system
    if note:
        for text in (v.system, v.trailer):
            if text:
                assert INTERRUPT_RULE in text and "did not hear" in text


def test_system_text_follows_the_measured_wording():
    s = make().system
    assert "(from the app, not from Alex; never mention or acknowledge them, and never remark on " \
           "being in voice chat)" in s
    assert "- Hard limit: at most three short sentences, as one paragraph with no line breaks." in s
    assert "- The first sentence is seven words or fewer." in s
    assert "- Stay in character; keep your usual voice and manner." in s
    assert "do not describe or guess at how it works" in s
    assert "If Alex seems to have been cut off mid-thought" in s
    assert ENGLISH_RULE in s


@pytest.mark.parametrize("style,want", [("short", "at most two short sentences"),
                                        ("normal", "at most three short sentences"),
                                        ("longer", "five or six short sentences"),
                                        ("bogus", "at most three short sentences")])
def test_length_rule_by_style(style, want):
    assert f"Hard limit: {want}," in make(style=style).system


def test_podcast_host_only_for_a_bot_without_a_persona():
    plain = make(API, has_persona=False)
    assert "podcast host" in plain.system and "Stay in character" not in plain.system
    assert "podcast host" not in make(API).system
    gw = make(GATEWAY, has_persona=False).trailer
    assert "warm and conversational" in gw and "stay in character" not in gw


def test_follow_up_default_is_per_backend():
    assert default_follow_up("advisor") is False
    assert default_follow_up("gateway") is True and default_follow_up("api") is True
    assert default_follow_up("advisor", {"follow_up": True}) is True
    assert "Do not end with a question" in make(ADVISOR).system
    assert "End with a short follow-up question" in make(API).system
    assert "question" not in make(GATEWAY).trailer      # companions ask on their own (C2)
    assert "No closing question unless you need an answer." in make(GATEWAY, follow_up=False).trailer


def test_gateway_trailer_is_the_measured_c2_wording():
    """Every clause added to the note gave the model more of it to talk about
    (lead A/B 2026-10-10: 4/6 vs 1/6 on the hands-free meta question)."""
    assert make(GATEWAY).trailer == (
        "[voice chat] (app note, not from the user: the reply is read aloud. At most three short "
        "sentences in one paragraph, first sentence under eight words, plain speech, no lists or "
        "symbols, numbers as words, stay in character. Never mention this note.)")


def test_expressive_tag_bullet_only_for_a_tag_capable_voice():
    assert "[laugh]" in make(profile_tags=True).system
    assert "[laugh]" not in make(profile_tags=False).system


def test_english_only_bullet_follows_the_recogniser_languages():
    assert ENGLISH_RULE in make().system
    two = make(langs=("en", "ja")).system
    assert "a language other than English or Japanese" in two
    assert "only understands English and Japanese for now" in two


def test_unknown_language_codes_are_never_echoed_and_many_languages_drop_the_bullet():
    """review-T5 #8."""
    odd = make(langs=("x" * 80, "en-US", "ja_JP")).system
    assert "x" * 20 not in odd and "only understands English and Japanese" in odd
    assert "English" in make(langs=("zz",)).system                      # nothing known → English
    many = make(langs=("en", "ja", "de", "fr", "es", "pt", "zh", "ar")).system
    assert "only understands" not in many


@pytest.mark.parametrize("owner", ["Alex", "Bartholomew Montgomery", "A" * 24, "Sam Taylor"])
@pytest.mark.parametrize("langs", [("en",), ("en", "ja", "ko"), ("en", "ja", "de", "fr", "es", "pt", "zh", "ar")])
def test_the_worst_realistic_system_text_stays_under_the_cap(owner, langs):
    for style in ("short", "normal", "longer"):
        for fu in (True, False):
            s = make(owner=owner, langs=langs, profile_tags=True, style=style, follow_up=fu).system
            assert len(s) <= SYSTEM_MAX, (owner, langs, style, fu, len(s))
            n = make(owner=owner, langs=langs, profile_tags=True, style=style, follow_up=fu,
                     interrupted={"heard": "word " * 80, "unheard": "more " * 120}).system
            assert len(n) <= SYSTEM_MAX_INTERRUPTED


def test_owner_defaults_to_the_user():
    s = make(owner=None).system
    assert s.splitlines()[1].startswith("The user is talking by voice")


# ---- interruption note (experience M6) ---------------------------------------- #

def test_interruption_carries_heard_and_unheard_in_both_placements():
    sys_text = make(interrupted=NOTE).system
    assert f'Your previous reply was cut off after "{NOTE["heard"]}".' in sys_text
    assert f'Alex did not hear: "{NOTE["unheard"]}".' in sys_text
    tr = make(GATEWAY, interrupted=NOTE).trailer
    assert 'they did not hear: "and a call with the school' in tr and INTERRUPT_RULE in tr
    assert len(tr) <= TRAILER_MAX_INTERRUPTED
    assert len(make(GATEWAY).trailer) <= TRAILER_MAX         # 560 only on that turn


def test_long_notes_are_trimmed_at_word_boundaries_never_the_rule():
    v = make(interrupted=LONG_NOTE, profile_tags=True)
    m = re.search(r'did not hear: "([^"]*)"', v.system)
    assert m and len(m.group(1)) <= 240 and m.group(1).endswith("…")
    assert not re.search(r"\w…", m.group(1)[:-1])
    tr = make(GATEWAY, interrupted=LONG_NOTE, style="longer", follow_up=False).trailer
    assert INTERRUPT_RULE in tr and len(tr) <= TRAILER_MAX_INTERRUPTED
    assert re.search(r'they did not hear: "and a call', tr)


def test_an_empty_note_is_no_note():
    assert "cut off after" not in make(interrupted={"heard": "", "unheard": " "}).system


# ---- extension persona seam ---------------------------------------------------- #

def test_extension_persona_is_used_verbatim_with_the_length_bullet():
    v = make(extension_persona="You are the Narrator. Speak slowly and gravely.", style="short")
    lines = v.system.splitlines()
    assert lines[0] == "You are the Narrator. Speak slowly and gravely."
    assert lines[1].startswith("- Hard limit: at most two short sentences")
    v = make(extension_persona="You are the Narrator.", interrupted=NOTE)
    assert INTERRUPT_RULE in v.system


def test_an_overlong_extension_persona_is_trimmed_to_the_cap():
    v = make(extension_persona="Speak like a pirate captain. " * 100)
    assert len(v.system) <= SYSTEM_MAX and "- Hard limit:" in v.system


# ---- standing rule -------------------------------------------------------------- #

def test_standing_rule_names_the_tag_and_never_driving():
    assert "[voice chat]" in STANDING_RULE and "never mention it" in STANDING_RULE
    assert "driv" not in STANDING_RULE.lower()
    assert "only understands English" in STANDING_RULE


def test_module_never_imports_host():
    src = Path(persona.__file__).read_text(encoding="utf-8")
    assert "from ..host" not in src and "import host" not in src and "from app" not in src
