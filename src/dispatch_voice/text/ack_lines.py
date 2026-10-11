"""The shipped acknowledgement pools (engine-neutral text, no model).

An acknowledgement is the short reaction a bot says while it works out the
real reply ("Let me see.", "On it."). It is chosen by the INTENT of what the
person just said, so a question never gets "on it" and a statement never gets
"let me check"; and it is only spoken when there is a real wait.

Rules every line follows (``acks.validate_line`` enforces the mechanical ones,
``tests/text/test_acks.py`` the rest):

* one breath: 1–7 words (the wait tiers up to 9), ≤ 60 characters;
* no praise of any kind ("good question" after a non-question is the tic this
  replaces) and nothing that agrees or disagrees before the reply has (no
  "makes sense", "fair enough", "great");
* nothing that promises a result ("I'll have that in a second", "nearly there");
* nothing that invites the person to keep talking ("go on"): the bot is about
  to answer, and an invitation makes people talk over the reply;
* "Yeah." and "Got it." are not statement or emotional reactions (a reply that
  then opens with "Yeah, …" sounds like a stutter);
* no digits, emoji or markdown: these lines go to the speech engine verbatim.

Operators override any list in ``<data>/voice-acks.yaml`` (contracts F.2).
"""
from __future__ import annotations

#: The pools by intent. ``followup`` is the reaction after a barge-in (the
#: person cut the reply off and said something new); ``wait_5s``/``wait_15s``
#: are said at most once per turn when the reply is that late.
DEFAULT: dict[str, tuple[str, ...]] = {
    # after a question — neutral "thinking" lines: they fit "what's on my
    # calendar?" and "do you like jazz?" alike
    "question": (
        "Let me see.",
        "Let's see.",
        "Hmm, let me think.",
        "One second.",
        "Okay, let me see.",
        "Let me think about that.",
        "Hmm.",
        "Give me a second.",
        "Mm, let me see.",
        "Hang on.",
        "Right, let me see.",
        "Okay, one second.",
        "Let me think.",
        "Hmm, one second.",
    ),
    # after a request or instruction (remind me, call, navigate, play, find …)
    "request": (
        "On it.",
        "Sure.",
        "Okay.",
        "Sure thing.",
        "Got it.",
        "Will do.",
        "Okay, one sec.",
        "Right, on it.",
        "Yep.",
        "Let me get that.",
        "Yep, on it.",
        "Sure, one moment.",
        "Okay, on it.",
        "Sure, give me a second.",
    ),
    # after a statement or a story — listening noises, never a verdict
    "statement": (
        "Mm-hm.",
        "Right.",
        "Okay.",
        "I see.",
        "Mm.",
        "Got you.",
        "Ah, okay.",
        "Oh, okay.",
        "Mm, okay.",
        "Right, right.",
        "Oh, right.",
        "Ah, I see.",
        "Huh, okay.",
    ),
    # after something hard (a rough day, worry, frustration)
    "emotional": (
        "Mm.",
        "I hear you.",
        "Oh, man.",
        "That sounds like a lot.",
        "Mm, I get it.",
        "Ah, I hear you.",
        "That's a lot.",
        "That's tough.",
        "Oof.",
        "Oh no.",
        "Ugh, that's rough.",
    ),
    # after good news said with a "!" ("We got the house!") — surprise only, so a misread
    # piece of bad news never gets a glad answer
    "upbeat": (
        "Oh wow.",
        "Whoa.",
        "Oh, wow!",
        "No way!",
        "Wow.",
        "Oh!",
    ),
    # after a barge-in: the person stopped the reply and said something new
    "followup": (
        "Okay.",
        "Sure.",
        "Right.",
        "Oh, okay.",
        "Got it.",
        "Okay, sure.",
        "Right, okay.",
        "Ah, okay.",
    ),
    "wait_5s": (
        "Still on it.",
        "Still here, just a moment.",
        "Hang on, still working on it.",
        "Bear with me a second.",
        "Still thinking.",
        "Just a moment longer.",
    ),
    "wait_15s": (
        "This is taking longer than I'd like. Hang on.",
        "Still working on it, sorry for the wait.",
        "Bear with me, it's a slow one.",
        "Sorry, this one's taking a while.",
    ),
}

#: Character flavour pools, by speaking style. A bot's pool is drawn before the
#: default pool for the same intent; a missing intent falls back to the default.
#: A bot gets a style through ``voice-acks.yaml: bots.<bot id>.flavour: <style>``
#: (or :data:`ALIASES`); ``bots.<bot id>.<intent>`` overrides single intents on
#: top. The styles are generic on purpose: a box's own characters are named in
#: its own data dir, never in this public package.
FLAVOUR: dict[str, dict[str, tuple[str, ...]]] = {
    # wry: dry, measured, a little amused (an advisor)
    "wry": {
        "question": ("Let me think.", "Right. Let me look.", "Give me a second on that.",
                     "Let me check my notes.", "Let me turn that over."),
        "request": ("On it.", "Sure.", "Right.", "Fine, let me look."),
        "statement": ("Mm-hm.", "Noted.", "Right.", "I see."),
        "emotional": ("Mm.", "Right.", "I hear you.", "That's a lot."),
        "followup": ("Okay.", "Right.", "Understood."),
        "wait_5s": ("Still on it.", "Bear with me."),
        "wait_15s": ("This one's slow. Bear with me.",),
    },
    # bright: crisp, quick, playful
    "bright": {
        "question": ("Ooh. Let me think.", "One sec.", "Ooh, okay.", "Hmm, hang on.",
                     "Okay, okay, let me see."),
        "request": ("On it.", "Yep.", "Sure thing.", "Got it.", "Easy.", "On it, on it."),
        "statement": ("Mm-hm.", "Right.", "Okay.", "Got you.", "Ooh, okay."),
        "emotional": ("Mm.", "Oof.", "Oh no.", "I hear you."),
        "followup": ("Okay.", "Yep.", "Oh, okay."),
        "wait_5s": ("Still on it.", "Hang on, still going."),
        "wait_15s": ("Slow one, sorry. Still going.",),
    },
    # trainer: calm, action first, calls you "boss"
    "trainer": {
        "question": ("One second.", "Let me think, boss.", "Okay, let me see."),
        "request": ("On it, boss.", "Got it.", "Okay.", "Will do."),
        "statement": ("Okay.", "Noted.", "Right.", "Heard."),
        "emotional": ("Heard.", "I hear you, boss."),
        "followup": ("Okay.", "Right."),
        "wait_5s": ("Still on it.",),
        "wait_15s": ("Hang tight, boss. Still working.",),
    },
    # gruff: low, few words, eager
    "gruff": {
        "question": ("Hmm.", "Let me dig into that.", "One sec."),
        "request": ("On it.", "Okay.", "Got it.", "Right away."),
        "statement": ("Mm-hm.", "Okay.", "Mm."),
        "emotional": ("Mm.", "Aw."),
        "followup": ("Okay.",),
        "wait_5s": ("Still digging.",),
        "wait_15s": ("Still digging. Hang on.",),
    },
}

#: Bot ids that get a style without any configuration. DisPatch's built-in
#: advisor bot is ``advisor``.
ALIASES: dict[str, str] = {"advisor": "wry"}

#: Non-lexical tokens. A Kokoro voice reads them as words ("Hmm." came out as a
#: word on af_heart), so word-only voices skip any line that contains one.
NON_LEXICAL = frozenset({"mm", "mmm", "mm-hm", "mm-hmm", "hmm", "hm", "mhm", "uh-huh",
                         "oof", "ooh", "aw", "ugh", "huh"})

#: Phrases no shipped pool may contain (the tests check them; operators decide for their own).
FORBIDDEN = ("good question", "great question", "good one", "great point", "good point",
             "love that", "makes sense", "fair enough", "go on", "great", "nearly there",
             "almost done", "almost there", "i'll have", "in a second i'll")
