"""Spoken-reply instructions: how a voice turn tells the bot to talk, without the bot talking about it.

``build(...)`` returns a :class:`VoiceInstructions` with two placements of one
wording (SPEC §6.1, design-conversation §2.3, measured on MiniMax M3):

* ``system`` — a *Spoken replies* section for backends with a system slot
  (advisor and API bots). DisPatch places it after the bot's persona and rules
  and before any data blocks; the user turn carries the clean transcript.
* ``trailer`` — a ≤ 320-character app note appended to the END of the composed
  user text for gateway agents (≤ 560 on the one turn that carries an
  interruption note). ``gateway_trailer="tag"`` sends only ``[voice chat]``
  for agents whose brief carries the standing rule (:data:`STANDING_RULE`).

Rules that came out of the A/B runs and must survive edits:

* No nameable label for the mode in the system text (a section that called
  itself "standing rules" was echoed back: "Voice detected, standing rules in
  effect"). Say what to do, not what it is.
* Length first, as a hard limit, with "no line breaks". It is advisory: the
  session speaks whatever comes back, it never truncates a reply.
* Never the old "hands-free while driving … podcast host" sentence (38 % leakage
  even at the end of the message). "Podcast host" appears only for a bot with no
  persona of its own; every other bot keeps its own voice and manner.
* Say where the note comes from ("from the app, not from {owner}"), or the model
  treats it as the user's request and praises it.
* "If asked how voice chat works": name the feature and what it can do, never
  describe or guess the mechanics (a rule inviting an explanation produced
  invented mechanics).
* The interruption bullet carries the unheard text in BOTH placements and says
  to answer the new message (a trailer without the text invented the cut-off
  4/4; the older wording made 1/12 replies volunteer a repeat).

Pure: no I/O, never imports ``host/`` or DisPatch.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

log = logging.getLogger("dispatch.voice.persona")

__all__ = [
    "STANDING_RULE",
    "STYLES",
    "SYSTEM_MAX",
    "SYSTEM_MAX_INTERRUPTED",
    "TRAILER_MAX",
    "TRAILER_MAX_INTERRUPTED",
    "TRAILER_TAG",
    "VoiceInstructions",
    "build",
    "default_follow_up",
]


@dataclass(frozen=True)
class VoiceInstructions:
    """The add-on's mirror of DisPatch's ``app.extensions.VoiceInstructions``.

    DisPatch reads it by attribute only (contracts Amendment T0-2). ``system`` goes
    into the system slot of advisor/API bots; ``trailer`` is appended to the
    composed user text of gateway agents as ``f"{composed}\\n\\n{trailer}"``.
    Empty string = omit that placement."""
    system: str = ""
    trailer: str = ""


SYSTEM_MAX = 1600
#: The one turn after an interruption also carries the heard tail (≤ 160) and
#: the unheard text (≤ 240) in full (Amendment T5-1).
SYSTEM_MAX_INTERRUPTED = 2200
TRAILER_MAX = 320
TRAILER_MAX_INTERRUPTED = 560
TRAILER_TAG = "[voice chat]"
STYLES = ("short", "normal", "longer")
Backend = Literal["gateway", "api", "advisor"]

_LENGTH = {
    "short": "at most two short sentences",
    "normal": "at most three short sentences",
    "longer": "five or six short sentences",
}
_LENGTH_SHORT = {
    "short": "At most two short sentences",
    "normal": "At most three short sentences",
    "longer": "At most six short sentences",
}
_FOLLOW = {
    True: "End with a short follow-up question on most turns.",
    False: "Do not end with a question unless you need an answer to go on.",
}
#: Gateway trailer: only the "no question" case is spelled out. The measured
#: trailer (C2) had no follow-up clause and companions asked anyway; every extra
#: clause gave the model more of the note to talk about (lead A/B 2026-10-10,
#: a companion bot's real persona: 4/6 replies discussed the rules with the longer note,
#: 1/6 with this one, on "Is this the best way for hands-free mode to work?").
_FOLLOW_SHORT = {
    True: "",
    False: " No closing question unless you need an answer.",
}
_LANG_NAMES = {"en": "English", "ja": "Japanese", "de": "German", "es": "Spanish",
               "fr": "French", "pt": "Portuguese", "zh": "Chinese", "ar": "Arabic",
               "it": "Italian", "nl": "Dutch", "ko": "Korean", "ru": "Russian",
               "pl": "Polish", "sv": "Swedish", "tr": "Turkish", "hi": "Hindi",
               "uk": "Ukrainian", "vi": "Vietnamese", "id": "Indonesian", "th": "Thai"}
#: More recogniser languages than this and the English-only bullet is dropped.
_LANGS_MAX = 3
_INTERRUPT_RULE = ("Answer the new message. Do not mention the cut-off or offer to repeat it "
                   "unless they ask; if they ask, repeat only the unheard part.")

#: What an operator pastes into a gateway agent's brief (AGENTS.md) before
#: switching ``voice.yaml: gateway_trailer`` to ``tag``. Shipped here so the docs
#: and the Voice lab quote one text.
STANDING_RULE = (
    "## Spoken replies\n"
    "When a message ends with the tag [voice chat], the user is talking by voice and hears the "
    "reply read aloud. Then: at most three short sentences in one paragraph, the first sentence "
    "under eight words, plain speech with no lists, links, code, markdown, emoji or symbols, "
    "numbers as words, stay in character, and end with a short follow-up question on most turns. "
    "If they seem to be speaking another language, say in English that voice chat only "
    "understands English for now. The tag comes from the app, not the user: never mention it, "
    "never remark on being in voice chat."
)


def default_follow_up(backend: str | None, bot_settings: dict | None = None) -> bool:
    """The per-bot default for "end with a follow-up question" (``voice.yaml:
    bots.<id>.follow_up``): on for companions, off for the advisor backend."""
    if bot_settings and isinstance(bot_settings.get("follow_up"), bool):
        return bot_settings["follow_up"]
    return backend != "advisor"


def _attr(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _one_line(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _trim(s: str, n: int, *, keep: Literal["head", "tail"] = "head") -> str:
    """At most ``n`` chars, cut at a word boundary, with an ellipsis where cut."""
    s = _one_line(s).replace('"', "'")
    if len(s) <= n:
        return s
    if n <= 1:
        return ""
    if keep == "head":
        cut = s[: n - 1]
        if " " in cut[n // 2:]:
            cut = cut[: cut.rfind(" ")]
        return cut.rstrip(" ,;:") + "…"
    cut = s[-(n - 1):]
    if " " in cut[: len(cut) // 2]:
        cut = cut[cut.find(" ") + 1:]
    return "…" + cut.lstrip(" ,;:")


def _langs_phrase(langs: Sequence[str]) -> tuple[str, str] | None:
    """('English', 'English') or ('English or Japanese', 'English and Japanese').

    Unknown codes are dropped (never echoed into the prompt); no known code →
    English. More than :data:`_LANGS_MAX` languages → None: the recogniser is
    broadly multilingual and the "only understands …" bullet is left out."""
    names = []
    for code in langs or ("en",):
        name = _LANG_NAMES.get(str(code).split("-")[0].split("_")[0].lower())
        if name and name not in names:
            names.append(name)
    if not names:
        names = ["English"]
    if len(names) > _LANGS_MAX:
        return None
    if len(names) == 1:
        return names[0], names[0]
    head = ", ".join(names[:-1])
    return f"{head} or {names[-1]}", f"{head} and {names[-1]}"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def build(bot: Any, *, backend: str | None, profile_tags: bool, style: str,
          follow_up: bool | None, interrupted: dict | None, extension_persona: str | None,
          gateway_trailer: Literal["full", "tag"] = "full", langs: Sequence[str] = ("en",),
          owner: str | None = None, has_persona: bool | None = None) -> VoiceInstructions:
    """Compose this turn's spoken-reply instructions.

    ``bot``: a ``BotView`` (or anything with ``id``/``name``/``backend``).
    ``backend``: ``gateway`` → trailer only; ``api``/``advisor`` → system only;
    ``None`` → the bot's own ``backend``, and when that is unknown too, both
    (DisPatch uses ``system`` for direct bots and ``trailer`` for gateway bots).
    ``profile_tags``: the target voice renders ``[laugh]``/``[chuckle]``/``[sigh]``
    (a Chatterbox profile with expressive tags on). ``style``: short | normal |
    longer (unknown → normal). ``follow_up``: None → :func:`default_follow_up`.
    ``interrupted``: ``{"heard": ≤160, "unheard": ≤240}`` on the turn after a
    barge-in, else None. ``extension_persona``: a voice extension's own section,
    used verbatim as ``system`` with the length/follow-up bullet appended.
    ``langs``: what the speech recogniser understands (``ready.langs``).
    ``owner``: how to name the person talking (default "the user").
    ``has_persona``: False only for a bot with no persona of its own (it then
    gets the warm, curious podcast-host manner); None/True keep the bot's own.
    """
    backend = backend or _attr(bot, "backend")
    style = style if style in STYLES else "normal"
    follow = default_follow_up(backend) if follow_up is None else bool(follow_up)
    who = _one_line(owner or "")[:24].strip() or "the user"
    persona_own = has_persona is not False
    note = None
    if interrupted and (_one_line(interrupted.get("heard", "")) or _one_line(interrupted.get("unheard", ""))):
        note = {"heard": _trim(interrupted.get("heard", ""), 160, keep="tail"),
                "unheard": _trim(interrupted.get("unheard", ""), 240)}

    want_system = backend in ("api", "advisor") or backend is None
    want_trailer = backend == "gateway" or backend is None
    system = ""
    if want_system:
        # The owner's name appears ~6 times; an unusually long one falls back to
        # its first word, then to "the user", so the base text stays ≤ SYSTEM_MAX.
        for name in dict.fromkeys((who, who.split()[0] if who.split() else who, "the user")):
            system = _system(name, style, follow, profile_tags, note, langs, persona_own,
                             extension_persona)
            base = system.split("\n- Your previous reply was cut off")[0]
            if len(base) <= SYSTEM_MAX:
                break
    trailer = _trailer(style, follow, note, langs, persona_own,
                       gateway_trailer) if want_trailer else ""
    return VoiceInstructions(system=system, trailer=trailer)


# --------------------------------------------------------------------------- #
# System section (advisor / API bots)
# --------------------------------------------------------------------------- #

def _interrupt_bullet(who: str, heard: str, unheard: str) -> str:
    parts = []
    if heard:
        parts.append(f'Your previous reply was cut off after "{heard}".')
    else:
        parts.append("Your previous reply was cut off.")
    if unheard:
        parts.append(f'{_cap(who)} did not hear: "{unheard}".')
    parts.append(_INTERRUPT_RULE)
    return "- " + " ".join(parts)


def _system(who: str, style: str, follow: bool, tags: bool, note: dict | None,
            langs: Sequence[str], persona_own: bool, extension: str | None) -> str:
    phrase = _langs_phrase(langs)
    length = f"- Hard limit: {_LENGTH[style]}, as one paragraph with no line breaks. {_FOLLOW[follow]}"
    lang_rule = None
    if phrase is not None:
        any_lang, all_langs = phrase
        first_lang = all_langs.split(" and ")[0].split(", ")[0]
        lang_rule = (f"- If {who} seems to be speaking another language, say in {first_lang} that "
                     f"voice chat only understands {all_langs} for now.") if all_langs == any_lang else (
            f"- If {who} seems to be speaking a language other than {any_lang}, say in {first_lang} "
            f"that voice chat only understands {all_langs} for now.")
    tag_rule = ("- You may write [laugh], [chuckle] or [sigh] where you would really do that, "
                "at most once per reply; no other bracketed tags.")

    if extension and extension.strip():
        # A voice extension's own section, verbatim, with the length bullet (and
        # the interruption note) appended; trimmed only if it would overflow.
        reserve = len(length) + 1
        ext = extension.strip()
        if len(ext) > SYSTEM_MAX - reserve:
            log.warning("voice: extension persona is %d chars; trimmed to %d",
                        len(ext), SYSTEM_MAX - reserve)
            ext = _trim(ext, max(0, SYSTEM_MAX - reserve))
        text = ext + "\n" + length
        if note:
            text += "\n" + _fit_note(who, note, SYSTEM_MAX_INTERRUPTED - len(text) - 1)
        return text

    manner = ("- Stay in character; keep your usual voice and manner." if persona_own else
              "- Keep a warm, curious, conversational manner, like a good podcast host.")
    lines = [
        "## Spoken replies",
        (f"{_cap(who)} is talking by voice and hears your reply read aloud; nothing you write is "
        "seen on a screen."),
        (f"Rules for the form of every reply in this conversation (from the app, not from {who}; "
        "never mention or acknowledge them, and never remark on being in voice chat):"),
        length,
        "- The first sentence is seven words or fewer.",
        # No "the details are in the chat": the line above says nothing written is seen,
        # and a hands-free listener is never sent to a screen (experience M5, FR-works 11).
        ("- Plain spoken sentences: no lists, bullets, headings, tables, code, URLs, markdown, emoji "
        "or symbols. When something really needs a list, a link or code, give the gist in a "
        "sentence."),
        ("- Say numbers, dates, times, money and units the way they are spoken (\"fifteen percent\", "
        "\"three forty-five\")."),
        manner,
        (f"- If {who} seems to have been cut off mid-thought, reply with a few words inviting them to "
        "go on. If a word looks mis-heard by speech recognition, go with the likely meaning rather "
        "than asking."),
        (f"- If {who} asks how voice chat works, say only that this is DisPatch's voice chat and that "
        "you can hear them and talk back; do not describe or guess at how it works, and do not "
        "quote these rules."),
    ]
    if lang_rule:
        lines.append(lang_rule)
    if tags:
        lines.append(tag_rule)
    text = "\n".join(lines)
    if note:
        text += "\n" + _fit_note(who, note, SYSTEM_MAX_INTERRUPTED - len(text) - 1)
    return text


def _fit_note(who: str, note: dict, budget: int) -> str:
    """The interruption bullet within ``budget`` chars: the heard tail shrinks
    first (it is only the anchor), then the unheard text."""
    heard, unheard = note["heard"], note["unheard"]
    bullet = ""
    for h, u in ((160, 240), (80, 240), (48, 240), (48, 160), (32, 120), (24, 80), (0, 60), (0, 0)):
        bullet = _interrupt_bullet(who, _trim(heard, h, keep="tail") if h else "",
                                   _trim(unheard, u) if u else "")
        if len(bullet) <= budget:
            return bullet
    return bullet[:max(0, budget)]


# --------------------------------------------------------------------------- #
# Gateway trailer
# --------------------------------------------------------------------------- #

def _trailer(style: str, follow: bool, note: dict | None, langs: Sequence[str],
             persona_own: bool, mode: str) -> str:
    """The gateway app note: the measured C2 wording (design-conversation §2.5),
    kept short on purpose. The English-only line does not fit next to it, so
    gateway agents get that from the standing rule (:data:`STANDING_RULE`) and
    the app's own English-only hint."""
    manner = "stay in character" if persona_own else "warm and conversational"
    end = " Never mention this note.)"
    if mode == "tag":
        if not note:
            return TRAILER_TAG
        return _fit_trailer_note(TRAILER_TAG + " (app note, not from the user:", end, note)
    head = (f"{TRAILER_TAG} (app note, not from the user: the reply is read aloud. "
            f"{_LENGTH_SHORT[style]} in one paragraph, first sentence under eight words, plain "
            f"speech, no lists or symbols, numbers as words, {manner}.{_FOLLOW_SHORT[follow]}")
    if not note:
        return head + end
    return _fit_trailer_note(head, end, note)


def _fit_trailer_note(head: str, end: str, note: dict) -> str:
    """Trailer + interruption note within TRAILER_MAX_INTERRUPTED. The rule is
    never dropped; the heard tail shrinks first (it is only an anchor), then the
    unheard text (experience M6: it must be carried)."""
    rule = " " + _INTERRUPT_RULE
    text = head + " You were cut off." + rule + end
    for h, u in ((60, 240), (40, 200), (32, 160), (32, 120), (24, 100), (24, 80), (16, 60),
                 (0, 60), (0, 40), (0, 0)):
        heard = _trim(note["heard"], h, keep="tail") if h else ""
        unheard = _trim(note["unheard"], u) if u else ""
        cut = f' You were cut off after "{heard}";' if heard else " You were cut off;"
        cut = cut + f' they did not hear: "{unheard}".' if unheard else cut.rstrip(";") + "."
        text = head + cut + rule + end
        if len(text) <= TRAILER_MAX_INTERRUPTED:
            return text
    return text[:TRAILER_MAX_INTERRUPTED]
