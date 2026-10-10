"""Acknowledgements: what the bot says while it works out the reply, and when.

Pieces (the session — host/ — is the only caller):

* :func:`intent_of` — the intent of what the person just said (cheap regexes).
* :class:`AckSelector` — picks a line: flavour pool first, shuffle bag per
  intent and voice, never one of the last four spoken, never one opening with
  the speculative reply's first word, skipped by the Reactions setting
  (Off/Sometimes/Often = 100/35/10 %), never on two consecutive fast turns.
* :class:`TurnAcks` + :class:`EtaEstimator` — the per-turn timing gate: an ack
  only after a real wait (``ACK_AFTER_S`` after the transcript: the host sets
  0.55 s, inside the client's turn-end tone hold; dropped when a reply is
  already queued or first audio is under 1.2 s away or the clip would run into
  it), the 5 s / 15 s wait tiers once each, tool status lines (≤ 2, once per
  distinct tool, replacing the 5 s tier), nothing while speaking or after an
  error. Pure: the session feeds it events and a clock.
* :func:`tool_status_line` — one spoken status per gateway tool call.
* :class:`RenderQueue` — renders the pools in the background, one line at a
  time, only while no reply is being synthesised and no session is waiting for
  one, into a disk cache with a fixed seed per line (output = f(voice, text)).
* :func:`load_pools` / :class:`PoolsFile` — ``<data>/voice-acks.yaml``.

Never imports ``host/``; engine access is injected.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import random
import re
import threading
import time
import unicodedata
import zlib
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from . import ack_lines

log = logging.getLogger("dispatch.voice.acks")

__all__ = [
    "INTENTS",
    "POOL_KEYS",
    "RENDER_ORDER",
    "TIERS",
    "AckSelector",
    "EtaEstimator",
    "Pools",
    "PoolsFile",
    "RenderQueue",
    "TurnAcks",
    "cache_key",
    "default_pools",
    "estimate_clip_s",
    "intent_of",
    "is_word_line",
    "line_seed",
    "load_pools",
    "tool_status_line",
    "validate_line",
]

Intent = Literal["question", "request", "statement", "emotional", "followup", "none"]
INTENTS = ("question", "request", "statement", "emotional", "followup")
TIERS = ("wait_5s", "wait_15s")
POOL_KEYS = INTENTS + TIERS
#: Background render order: the commonest reactions first, then the tiers.
RENDER_ORDER = ("statement", "request", "question", "followup", "emotional", "wait_5s", "wait_15s")
SKIP_RATE = {"off": 1.0, "sometimes": 0.35, "often": 0.10}
RECENT_WINDOW = 4
FAST_TURN_S = 4.0
MAX_WORDS = 9
MAX_CHARS = 60
MAX_LINES = 20

# --------------------------------------------------------------------------- #
# Lines and validation
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_MD_RE = re.compile(r"[*_`#\[\]<>|~]|\]\(|^\s*[-+•]\s")


def _first_word(text: str | None) -> str:
    m = _WORD_RE.search(text or "")
    return m.group(0).lower().replace("’", "'") if m else ""


def is_word_line(text: str) -> bool:
    """False when the line holds a non-lexical sound ("Mm-hm.", "Hmm, okay.")."""
    return not any(w.lower() in ack_lines.NON_LEXICAL for w in _WORD_RE.findall(text))


def validate_line(text: object) -> str | None:
    """The cleaned line, or None when it breaks a rule (contracts F.2): 1–9 words,
    ≤ 60 characters, no digits, no emoji/symbols, no markdown."""
    if not isinstance(text, str):
        return None
    t = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    if not t or len(t) > MAX_CHARS:
        return None
    words = t.split()
    if not 1 <= len(words) <= MAX_WORDS:
        return None
    if any(ch.isdigit() for ch in t) or _MD_RE.search(t):
        return None
    for ch in t:
        cat = unicodedata.category(ch)
        if not (cat[0] in "LP" or ch == " "):
            return None
    if not _WORD_RE.search(t):
        return None
    return t


def estimate_clip_s(text: str) -> float:
    """Expected spoken length: ≈ 0.4 s + 0.35 s per word (the ack bench)."""
    return 0.4 + 0.35 * len(text.split())


def line_seed(text: str) -> int:
    """The fixed render seed of a line: output is a pure function of (voice, text)."""
    return zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF


def cache_key(text: str) -> str:
    """Disk-cache file name of a line (the same key as the host's render cache)."""
    return hashlib.sha256(text.encode()).hexdigest()[:24] + ".npy"


# --------------------------------------------------------------------------- #
# Pools
# --------------------------------------------------------------------------- #

def _norm_key(s: str) -> str:
    return str(s or "").strip().casefold()


@dataclass(frozen=True)
class Pools:
    """Default lines per pool key plus per-bot overrides.

    ``flavour``: the shipped style pools (``wry``, ``bright``, ``trainer``,
    ``gruff``). ``bots``: operator pools keyed by bot id
    (``voice-acks.yaml: bots.<id>.<intent>``). ``bot_flavour``: bot id → style
    (``voice-acks.yaml: bots.<id>.flavour``). ``names``: bot id → display name
    (from :meth:`bind`); a bot whose id or name equals a style key or an
    :data:`~.ack_lines.ALIASES` key gets that style. Lookup per intent: the
    operator's lines, else the bot's style, else nothing (the default pool is
    always the second source)."""
    default: Mapping[str, tuple[str, ...]]
    flavour: Mapping[str, Mapping[str, tuple[str, ...]]] = field(default_factory=dict)
    bots: Mapping[str, Mapping[str, tuple[str, ...]]] = field(default_factory=dict)
    names: Mapping[str, str] = field(default_factory=dict)
    word_only_prefixes: tuple[str, ...] = ("builtin:",)
    bot_flavour: Mapping[str, str] = field(default_factory=dict)
    #: Operator tool-status rules (``voice-acks.yaml: tools``), tried before the
    #: shipped generic table: ``(compiled pattern, spoken line)``.
    tool_rules: tuple[tuple[re.Pattern, str], ...] = ()

    def bind(self, bots: Iterable[tuple[str, str]] | Iterable[object]) -> Pools:
        """A copy that knows each bot's display name (``(id, name)`` pairs or
        objects with ``id``/``name``, e.g. DisPatch ``BotView`` s)."""
        names = dict(self.names)
        for b in bots:
            bid, name = (b if isinstance(b, tuple) else (getattr(b, "id", ""), getattr(b, "name", "")))
            if bid:
                names[str(bid)] = str(name or "")
        return Pools(self.default, self.flavour, self.bots, names, self.word_only_prefixes,
                     self.bot_flavour, self.tool_rules)

    def _flavour_key(self, bot_id: str | None) -> str | None:
        if not bot_id:
            return None
        chosen = _norm_key(self.bot_flavour.get(bot_id, ""))
        if chosen in self.flavour:
            return chosen
        for cand in (bot_id, self.names.get(bot_id, "")):
            k = _norm_key(cand)
            k = ack_lines.ALIASES.get(k, k)
            if k in self.flavour:
                return k
        return None

    def word_only(self, voice: str | None) -> bool:
        return bool(voice) and any(str(voice).startswith(p) for p in self.word_only_prefixes)

    def lines(self, key: str, *, bot_id: str | None, voice: str | None
              ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """(flavour lines, default lines) for one pool key, filtered for the voice."""
        own: tuple[str, ...] = ()
        op = self.bots.get(bot_id or "", {})
        if key in op:
            own = tuple(op[key])
        else:
            fk = self._flavour_key(bot_id)
            if fk:
                own = tuple(self.flavour[fk].get(key, ()))
        base = tuple(x for x in self.default.get(key, ()) if x not in own)
        if self.word_only(voice):
            own = tuple(x for x in own if is_word_line(x))
            base = tuple(x for x in base if is_word_line(x))
        return own, base

    def render_lines(self, *, bot_id: str | None, voice: str | None) -> list[str]:
        """Every line this bot can say in this voice, in render priority order
        (statement → request → question → follow-up → emotional → tiers; flavour
        before default)."""
        out: list[str] = []
        seen: set[str] = set()
        for key in RENDER_ORDER:
            own, base = self.lines(key, bot_id=bot_id, voice=voice)
            for line in own + base:
                if line not in seen:
                    seen.add(line)
                    out.append(line)
        return out

    def all_texts(self) -> set[str]:
        """Every line any bot could say (for cache pruning)."""
        out: set[str] = set()
        for pool in (self.default, *self.flavour.values(), *self.bots.values()):
            for lines in pool.values():
                out.update(lines)
        out.update(self.tool_texts())
        return out

    def tool_texts(self) -> list[str]:
        """The operator's tool-status lines (rendered like the shipped ones)."""
        out: list[str] = []
        for _rx, line in self.tool_rules:
            if line not in out:
                out.append(line)
        return out


def default_pools() -> Pools:
    return Pools(default={k: tuple(v) for k, v in ack_lines.DEFAULT.items()},
                 flavour={k: {i: tuple(v) for i, v in p.items()} for k, p in ack_lines.FLAVOUR.items()})


def _clean_pool(raw: object, where: str) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    if raw is None:
        return out
    if not isinstance(raw, dict):
        log.warning("voice-acks: %s is not a mapping; ignored", where)
        return out
    for key, lines in raw.items():
        if key not in POOL_KEYS:
            log.warning("voice-acks: %s.%s is not a known intent; ignored", where, key)
            continue
        if not isinstance(lines, list):
            log.warning("voice-acks: %s.%s is not a list; ignored", where, key)
            continue
        good: list[str] = []
        for line in lines:
            v = validate_line(line)
            if v is None:
                log.warning("voice-acks: %s.%s: skipped %r (1–9 words, ≤ 60 chars, no digits, "
                            "emoji or markdown)", where, key, str(line)[:80])
                continue
            if v not in good:
                good.append(v)
        if len(good) > MAX_LINES:
            log.warning("voice-acks: %s.%s has %d lines; keeping the first %d", where, key,
                        len(good), MAX_LINES)
            good = good[:MAX_LINES]
        if good:
            out[key] = tuple(good)
    return out


def load_pools(path: Path | str | None) -> Pools:
    """Shipped pools with ``<data>/voice-acks.yaml`` applied.

    ``default.<intent>`` replaces that intent's list; ``bots.<bot id>.flavour``
    picks a shipped style (wry, bright, trainer, gruff) for one bot, and
    ``bots.<bot id>.<intent>`` overrides single intents on top. Validation per line; a bad line, intent or file is
    logged and skipped, never fatal."""
    pools = default_pools()
    if path is None:
        return pools
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return pools
    except OSError as e:
        log.warning("voice-acks: cannot read %s: %s", p.name, e)
        return pools
    try:
        import yaml
        data = yaml.safe_load(text) or {}
    except Exception as e:  # noqa: BLE001 — ImportError or any YAML error: keep the shipped pools
        log.warning("voice-acks: %s not loaded (%s); using the shipped pools", p.name,
                    type(e).__name__)
        return pools
    if not isinstance(data, dict):
        log.warning("voice-acks: %s is not a mapping; using the shipped pools", p.name)
        return pools
    default = dict(pools.default)
    default.update(_clean_pool(data.get("default"), "default"))
    bots: dict[str, dict[str, tuple[str, ...]]] = {}
    bot_flavour: dict[str, str] = {}
    raw_bots = data.get("bots") or {}
    if isinstance(raw_bots, dict):
        for bid, pool in raw_bots.items():
            if isinstance(pool, dict) and "flavour" in pool:
                pool = dict(pool)
                style = _norm_key(pool.pop("flavour"))
                if style in pools.flavour:
                    bot_flavour[str(bid)] = style
                else:
                    log.warning("voice-acks: bots.%s.flavour %r is not one of %s; ignored", bid,
                                style, ", ".join(sorted(pools.flavour)))
            cleaned = _clean_pool(pool, f"bots.{bid}")
            if cleaned:
                bots[str(bid)] = cleaned
    else:
        log.warning("voice-acks: bots is not a mapping; ignored")
    return Pools(default, pools.flavour, bots, pools.names, pools.word_only_prefixes, bot_flavour,
                 _clean_tool_rules(data.get("tools")))


#: ``voice-acks.yaml: tools`` bounds: rules, and the length of one pattern.
MAX_TOOL_RULES = 32
TOOL_PATTERN_MAX = 120


def _clean_tool_rules(raw: object) -> tuple[tuple[re.Pattern, str], ...]:
    """``tools: [{match: <regex on the tool name>, say: <line>}, …]`` — install-specific
    tool names (an MCP server's prefix, a private agent tool) map to a spoken status
    here instead of in the shipped table, which stays generic. Bad rules are logged
    and skipped; the line follows the acknowledgement-line rules."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        log.warning("voice-acks: tools is not a list; ignored")
        return ()
    out: list[tuple[re.Pattern, str]] = []
    for i, rule in enumerate(raw):
        if len(out) >= MAX_TOOL_RULES:
            log.warning("voice-acks: tools has more than %d rules; the rest are ignored", MAX_TOOL_RULES)
            break
        if not isinstance(rule, dict):
            log.warning("voice-acks: tools[%d] is not a mapping; ignored", i)
            continue
        pat, say = rule.get("match"), rule.get("say")
        if not isinstance(pat, str) or not pat or len(pat) > TOOL_PATTERN_MAX:
            log.warning("voice-acks: tools[%d].match must be a pattern of 1-%d characters; ignored",
                        i, TOOL_PATTERN_MAX)
            continue
        try:
            rx = re.compile(pat, re.IGNORECASE)
        except re.error as e:
            log.warning("voice-acks: tools[%d].match is not a valid pattern (%s); ignored", i, e)
            continue
        line = validate_line(say)
        if line is None:
            log.warning("voice-acks: tools[%d].say %r is not a valid line (1-9 words, <= 60 chars, "
                        "no digits, emoji or markdown); ignored", i, str(say)[:80])
            continue
        out.append((rx, line))
    return tuple(out)


class PoolsFile:
    """``load_pools`` hot-reloaded on mtime (checked at most once a second)."""

    def __init__(self, path: Path | str | None, *, check_every_s: float = 1.0) -> None:
        self.path = Path(path) if path else None
        self.check_every_s = check_every_s
        self._mtime: float | None = None
        self._checked = 0.0
        self._pools = load_pools(self.path)
        self._names: dict[str, str] = {}
        self._mtime = self._stat()
        self._lock = threading.Lock()

    def _stat(self) -> float | None:
        try:
            return self.path.stat().st_mtime if self.path else None
        except OSError:
            return None

    def bind(self, bots) -> None:
        with self._lock:
            self._pools = self._pools.bind(bots)
            self._names = dict(self._pools.names)

    def get(self) -> Pools:
        now = time.monotonic()
        with self._lock:
            if now - self._checked >= self.check_every_s:
                self._checked = now
                m = self._stat()
                if m != self._mtime:
                    self._mtime = m
                    self._pools = load_pools(self.path).bind(self._names.items())
                    log.info("voice-acks: reloaded")
            return self._pools


# --------------------------------------------------------------------------- #
# Intent
# --------------------------------------------------------------------------- #

_LEAD_FILLER = re.compile(
    r"^(?:(?:so|okay|ok|hey|hi|um+|uh+|er+|well|and|oh|alright|all right|right|yeah|yes|"
    r"hmm+|now|listen|look|actually|anyway|also|then|but)\b[\s,.!-]*)+", re.IGNORECASE)
# "hey Sage, …" / "ok nova tell me …": the name after a greeting is dropped
# whatever its case (Parakeet usually, not always, capitalises names), unless
# the word is itself the start of the request or question.
_HEY_NAME = re.compile(r"^(?:hey|hi|hello|ok|okay|yo)\s+([A-Za-z][a-z'’]+)[,.!]?\s+", re.IGNORECASE)
_HEY_THERE = re.compile(r"^(?:hey|hi|hello|hiya)\s+there\b[\s,.!]*", re.IGNORECASE)
_WH = re.compile(
    r"^(?:who|whom|whose|what|what's|whats|when|when's|where|where's|why|how|how's|which)\b",
    re.IGNORECASE)
_AUX = re.compile(
    r"^(?:is|isn't|are|aren't|am|was|wasn't|were|weren't|do|does|doesn't|don't|did|didn't|"
    r"can|can't|could|couldn't|would|wouldn't|should|shouldn't|will|won't|have|haven't|"
    r"has|hasn't|had|any|anything|shall|may|might)\b", re.IGNORECASE)
# "What a day!" / "How nice." are exclamations, not questions.
_EXCLAIM = re.compile(r"^(?:what\s+an?\b|how\s+(?:nice|lovely|cool|sweet|great|funny|weird|"
                      r"strange|odd|sad|awful|annoying|cute|kind|rude|typical)\b)", re.IGNORECASE)
_IMPERATIVE = (
    "remind|tell|give|show|send|find|look|check|book|set|add|make|call|text|play|open|start|"
    "stop|turn|put|get|write|draft|schedule|cancel|change|update|read|search|take|help|"
    "navigate|skip|pause|resume|reply|message|email|order|note|save|delete|remove|move|"
    "list|summarise|summarize|explain|describe|translate|pick|choose|plan|bring|go ahead|"
    "let's|let us|try|repeat|say|sing|count|guess|recommend|suggest|keep|clear|mark|"
    "forward|share|buy|pay|ring|dial|route")
_NOT_A_NAME = re.compile(
    rf"^(?:{_IMPERATIVE}|please|so|um|uh|well|and|but|then|now|there|you|i|i'm|it|it's|we|"
    r"that|this|the|a|an|my|can|could|would|will|do|does|did|is|are|was|were|what|when|where|"
    r"why|how|who|which|thanks|thank|good|morning|listen|look|quick|just|sorry|yes|no|yeah)$",
    re.IGNORECASE)
_REQUEST = re.compile(
    rf"^(?:please\s+)?(?:{_IMPERATIVE})\b"
    r"|^(?:can|could|would|will)\s+(?:you|ya)\s+(?:please\s+)?"
    r"|^(?:i\s+(?:need|want|'d like|would like)\s+(?:you\s+)?to)\b"
    r"|^(?:i'd\s+like\s+you\s+to)\b"
    r"|\bplease[.!]?\s*$", re.IGNORECASE)
# "Can you believe it?" / "Will you be here tomorrow?" / "Would you say that's
# normal?" ask for an opinion or a fact; they are questions, not requests.
_MODAL_QUESTION = re.compile(
    r"^(?:can|could|would|will|won't|wouldn't)\s+(?:you|ya)\s+(?:ever\s+|even\s+|really\s+|still\s+)?"
    r"(?:believe|think|imagine|say|feel|know|guess|recall|rather|be(?!\s+able\b)|"
    r"remember(?!\s+to\b))\b", re.IGNORECASE)
_RHETORICAL = re.compile(r"^(?:can|could|would)\s+(?:you|ya)\s+(?:even\s+)?(?:believe|imagine)\b",
                         re.IGNORECASE)
# Greetings, thanks, goodbyes, affection and small talk: the reply is quick and
# social, and a listening noise ("I see.") after "I love you." is exactly the
# misfit a reaction must never be. Stripped from the front; when nothing of
# substance is left the intent is ``none``.
_PHATIC = re.compile(
    r"^(?:"
    r"good\s+(?:morning|afternoon|evening|night)|morning|evening|hello|hi|hey|hiya|howdy|yo|"
    r"greetings|welcome back|"
    r"(?:thanks|thank\s+(?:you|u)|cheers|ta|much appreciated|appreciate\s+(?:it|that|you)|"
    r"i\s+(?:really\s+)?appreciate\s+(?:it|that|you|this))"
    r"(?:\s+(?:a lot|a bunch|so much|very much|again|anyway|heaps|loads|"
    r"for\s+(?:that|this|everything|the help|your help|helping|asking|listening)))*|"
    r"(?:good\s*)?bye(?:\s*bye)?|goodbye|night night|see\s+(?:you|ya)(?:\s+(?:later|soon|"
    r"tomorrow|tonight|then|around|in a bit))?|talk\s+(?:to\s+you\s+)?(?:later|soon|tomorrow)|"
    r"catch\s+you\s+later|later|take care|"
    r"have\s+a\s+(?:good|nice|great|lovely|safe)\s+(?:day|night|evening|one|weekend|trip|time|"
    r"drive|flight|sleep)|sleep\s+(?:well|tight)|sweet dreams|"
    r"(?:i\s+)?love\s+(?:you|ya)(?:\s+(?:too|so much|lots))?|(?:i\s+)?miss\s+(?:you|ya)(?:\s+too)?|"
    r"(?:sounds|that\s+sounds)\s+(?:good|great|perfect|fine|lovely|nice|fun|about right)"
    r"(?:\s+to\s+me)?|that\s+works(?:\s+for\s+me)?|"
    r"that(?:'s|\s+is|\s+was)\s+(?:really\s+|very\s+|so\s+|super\s+)?(?:fine|great|perfect|good|"
    r"okay|ok|helpful|useful|kind|sweet|lovely|nice|amazing|awesome|brilliant)"
    r"(?:\s+(?:with|for|of)\s+(?:me|you))?|"
    r"(?:that|this|it)\s+(?:really\s+|totally\s+|definitely\s+|actually\s+)?help(?:s|ed)"
    r"(?:\s+(?:a lot|a bunch|so much|loads|heaps|a ton))?|"
    r"perfect|great|awesome|cool|nice|lovely|brilliant|no worries|no problem|all good|"
    r"you're welcome|never\s*mind|"
    r"how\s+(?:are|r)\s+(?:you|ya|things)(?:\s+(?:doing|today|tonight|going))?|how're you|"
    r"how(?:'s|\s+is)\s+(?:it|everything|life|things)\s+going|how\s+have\s+you\s+been|"
    r"how(?:'s|\s+is|\s+was)\s+your\s+(?:day|night|evening|morning|weekend|week)"
    r"(?:\s+(?:going|been))?|how\s+(?:you|ya)\s+doing|what's\s+up|whats\s+up|what's\s+new|sup|"
    r"(?:are\s+)?you\s+(?:still\s+)?there|nice\s+to\s+(?:meet|see|hear)\s+you|"
    r"good\s+to\s+(?:see|hear\s+from)\s+you"
    r")\b[\s,.!?;:-]*", re.IGNORECASE)
# Hard or heavy feelings only: an upbeat share is a statement (a neutral
# reaction), never "that sounds like a lot".
_EMOTIONAL = re.compile(
    r"\b(?:rough|tired|exhausted|wiped|drained|burn(?:ed|t)\s+out|stress(?:ed|ful)?|"
    r"frustrat\w*|annoy\w*|upset|sad|depress\w*|worried|worry(?:ing)?|anxious|anxiety|"
    r"nervous|scared|afraid|angry|mad at|furious|lonely|alone|miss (?:him|her|them|my)|"
    r"hurts?|hurting|pain|sick|ill|awful|terrible|horrible|crappy|lousy|overwhelm\w*|"
    r"struggl\w*|hate(?!\s+to\s+(?:say|admit|break|tell|ask|bother|interrupt))|ugh|sucks|"
    r"crying|cried|heartbroken|grief|griev\w*|died|passed away|lost my|fired|laid off|dumped|"
    r"broke up|panic\w*|can't sleep|couldn't sleep|long (?:day|week|night|shift))\b", re.IGNORECASE)
# A trigger is negated when one of these sits in the three words before it:
# "No pain today", "I'm not tired at all", "I finally beat my anxiety".
_NEGATOR = re.compile(r"\b(?:not|no|never|without|beat|beaten|over|gone|free|nothing|"
                      r"less|stopped|quit|cured)\b|n't\b", re.IGNORECASE)
_UPBEAT = re.compile(r"\b(?:great|fine|happy|better|good news|amazing|love it|awesome|"
                     r"wonderful|excited|glad|relieved|fantastic|brilliant|perfect)\b",
                     re.IGNORECASE)
_CLAUSE_SPLIT = re.compile(r"[,;:.!?—–]|\s-\s|\bbut\b", re.IGNORECASE)


def _strip_name(t: str) -> str:
    m = _HEY_NAME.match(t)
    if m and not _NOT_A_NAME.match(m.group(1)):
        return t[m.end():]
    return t


def _core(text: str) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    t = _HEY_THERE.sub("", t)
    t = _strip_name(t)
    prev = None
    while prev != t:
        prev = t
        t = _LEAD_FILLER.sub("", t).lstrip(" ,.-")
    return t


def _strip_phatic(t: str) -> str:
    """Drop leading social phrases, and a name right after one: "Good morning
    Sage, what's on today?" → "what's on today?"; "Thanks Mom." → ""."""
    prev = None
    while prev != t:
        prev = t
        m = _PHATIC.match(t)
        if m and m.end() > 0:
            t = t[m.end():]
            n = re.match(r"([A-Za-z][a-z'’]+)\b[\s,.!?]*", t)
            # A capitalised word (or a lone trailing word) after the phrase is a
            # name: "Morning Sage, …", "thanks mom".
            if n and not _NOT_A_NAME.match(n.group(1)) and (
                    n.group(1)[:1].isupper() or not t[n.end():].strip()):
                t = t[n.end():]
        t = _LEAD_FILLER.sub("", t).lstrip(" ,.-")
    return t


def _emotional(raw: str) -> bool:
    """A hard feeling that is not negated ("no pain today") and not outweighed
    by an upbeat first clause ("Great news, the pain is gone")."""
    hits = []
    for m in _EMOTIONAL.finditer(raw):
        window = " ".join(raw[max(0, m.start() - 48): m.start()].split()[-3:])
        if not _NEGATOR.search(window):
            hits.append(m)
    if not hits:
        return False
    if _UPBEAT.search(raw):
        first = _CLAUSE_SPLIT.split(raw, maxsplit=1)[0]
        if not any(h.start() < len(first) for h in hits):
            return False
    return True


def intent_of(text: str, *, after_barge_in: bool = False) -> Intent:
    """question | request | statement | emotional | followup | none.

    ≤ 2-word utterances ("yes", "okay thanks") get none: the reply is quick and
    a reaction to "yes" sounds silly. After a barge-in the person has moved on:
    ``followup``. Greetings, thanks, goodbyes, affection and small talk ("I love
    you.", "How are you?", "Sounds good to me.") get none: a listening noise
    after them is a misfit. A hard feeling (not negated, not outweighed by good
    news) without a question is ``emotional``; an imperative (car verbs included:
    navigate, call, text, play, skip, pause, set, start, open) or "can you …" is
    a ``request`` unless it asks for an opinion ("can you believe it?"); a
    wh-start, an auxiliary start (only with a "?" when the transcript is
    punctuated) or a trailing "?" is a ``question``; everything else is a
    ``statement``."""
    raw = re.sub(r"\s+", " ", text or "").strip()
    if len(_WORD_RE.findall(raw)) <= 2:
        return "none"
    if after_barge_in:
        return "followup"
    core = _strip_phatic(_core(raw) or raw)
    if len(_WORD_RE.findall(core)) <= 1:
        return "none"
    ends_q = raw.endswith("?")
    punctuated = raw[-1:] in ".!?…"
    is_q = ends_q or (bool(_WH.search(core)) and not (raw.endswith("!") or _EXCLAIM.search(core))) \
        or (bool(_AUX.search(core)) and not punctuated)
    if not is_q and _emotional(core):
        return "emotional"
    if _MODAL_QUESTION.search(core):
        # "Can you believe it?" / "Can you imagine?" are exclamations; no
        # thinking line fits them, so a short one gets nothing.
        if _RHETORICAL.search(core) and len(_WORD_RE.findall(core)) <= 5:
            return "none"
        return "question"
    if _REQUEST.search(core):
        return "request"
    if is_q:
        return "question"
    return "statement"


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

class AckSelector:
    """Picks acknowledgement lines for one voice session.

    A shuffle bag per (bot, voice, pool key): each cycle draws the bot's own
    flavour lines first (shuffled), then the default lines (shuffled), so every
    line is used before any repeats; no line among the last
    :data:`RECENT_WINDOW` spoken in this session (across intents and tiers)."""

    def __init__(self, pools: Pools, *, mode: Literal["off", "sometimes", "often"] = "sometimes",
                 rng: random.Random | None = None) -> None:
        self.pools = pools
        self.mode = mode if mode in SKIP_RATE else "sometimes"
        self.rng = rng or random.Random()
        self.recent: deque[str] = deque(maxlen=RECENT_WINDOW)
        self._bags: dict[tuple[str, str, str], list[str]] = {}
        #: Why the last :meth:`pick`/:meth:`tier` answered as it did (Amendment T3s3-1):
        #: ``picked`` · ``skipped`` (the Reactions roll) · ``fast_turn`` · ``off`` ·
        #: ``no_intent`` · ``no_lines`` (the pool is empty) · ``none_fit`` (lines exist but
        #: none is eligible: not rendered yet, too long for the time left, or opening
        #: with the reply's first word). The session sends the ``ack`` earcon only for
        #: ``none_fit`` — a roll that skipped stays silent, as designed.
        self.last_outcome = ""
        self._draw_outcome = ""

    def set_mode(self, mode: str) -> None:
        self.mode = mode if mode in SKIP_RATE else "sometimes"

    def note_said(self, text: str) -> None:
        """Count a line spoken by other means (a tool status line) as recent."""
        if text:
            self.recent.append(text)

    def _draw(self, key: str, bot_id: str, voice: str, reply_first_word: str | None,
              fits: Callable[[str], bool] | None) -> str | None:
        own, base = self.pools.lines(key, bot_id=bot_id, voice=voice)
        if not own and not base:
            self._draw_outcome = "no_lines"
            return None
        self._draw_outcome = "none_fit"
        bag_key = (bot_id or "", voice or "", key)
        bag = self._bags.get(bag_key)
        valid = set(own) | set(base)
        if bag is not None:
            bag[:] = [x for x in bag if x in valid]   # pools may have reloaded
        avoid_word = _first_word(reply_first_word) if reply_first_word else ""

        def hard_ok(line: str) -> bool:
            return (not (avoid_word and _first_word(line) == avoid_word)
                    and (fits is None or fits(line)))

        # The no-repeat window shrinks to (eligible lines − 1) when the pool is
        # small (an operator's 3-line pool, a Kokoro voice, a voice still
        # rendering), so it ages instead of starving: a small pool never goes
        # silent just because every line was said recently, and with two or more
        # eligible lines the one said last is still never picked again.
        eligible = sum(1 for line in valid if hard_ok(line))
        window = min(RECENT_WINDOW, max(0, eligible - 1))
        recent = set(list(self.recent)[-window:]) if window else set()

        def ok(line: str) -> bool:
            return line not in recent and hard_ok(line)

        for attempt in range(2):
            if not bag:
                a, b = list(own), list(base)
                self.rng.shuffle(a)
                self.rng.shuffle(b)
                bag = self._bags[bag_key] = a + b
            for i, line in enumerate(bag):
                if ok(line):
                    return bag.pop(i)
            if attempt == 0:
                # Everything left in this cycle is excluded; a fresh cycle may
                # still have an eligible line (only when the bag was a remnant).
                if len(bag) >= len(valid):
                    break
                bag = []
        return None

    def pick(self, intent: str, *, bot_id: str, voice: str, now: float,
             last_turn_at: float | None, last_acked: bool,
             reply_first_word: str | None = None,
             fits: Callable[[str], bool] | None = None) -> str | None:
        """A line for this turn, or None to stay quiet.

        None when: the intent is none/unknown; Reactions are off; the previous
        turn was < 4 s ago and was acknowledged (never two in a row on a fast
        exchange); the Reactions skip roll says skip; or no eligible line (all
        recent, all opening with ``reply_first_word``, none ``fits``)."""
        if self.mode == "off":
            self.last_outcome = "off"
            return None
        if intent not in INTENTS:
            self.last_outcome = "no_intent"
            return None
        if last_turn_at is not None and last_acked and now - last_turn_at < FAST_TURN_S:
            self.last_outcome = "fast_turn"
            return None
        if self.rng.random() < SKIP_RATE[self.mode]:
            self.last_outcome = "skipped"
            return None
        line = self._draw(intent, bot_id, voice, reply_first_word, fits)
        if line:
            self.recent.append(line)
        self.last_outcome = "picked" if line else self._draw_outcome
        return line

    def tier(self, which: Literal["wait_5s", "wait_15s"], *, bot_id: str, voice: str,
             fits: Callable[[str], bool] | None = None) -> str | None:
        """A wait line (no skip roll); None when Reactions are off or none is eligible."""
        if which not in TIERS or self.mode == "off":
            self.last_outcome = "off" if self.mode == "off" else "no_intent"
            return None
        line = self._draw(which, bot_id, voice, None, fits)
        if line:
            self.recent.append(line)
        self.last_outcome = "picked" if line else self._draw_outcome
        return line


# --------------------------------------------------------------------------- #
# Tool status lines
# --------------------------------------------------------------------------- #

#: The shipped table is GENERIC on purpose (it ships in a public package): tool
#: names a particular install uses — an MCP server's prefix, a private agent
#: tool — go in the operator's ``voice-acks.yaml: tools`` rules, which are tried
#: first (FR-hardening M1).
_TOOL_LINES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"flash_?cards?|(?:^|_)cards?(?:_|$)|(?:^|_)decks?(?:_|$)", re.IGNORECASE),
     "Checking your cards"),
    (re.compile(r"calendar", re.IGNORECASE), "Checking your calendar"),
    (re.compile(r"e?mail|inbox", re.IGNORECASE), "Checking your email"),
    (re.compile(r"^memory|wiki|knowledge|(?:^|_)notes?(?:_|$)", re.IGNORECASE), "Checking my notes"),
    (re.compile(r"web_?search|(?:^|_)search", re.IGNORECASE), "Searching"),
    (re.compile(r"weather|forecast", re.IGNORECASE), "Checking the weather"),
    (re.compile(r"image|picture|photo", re.IGNORECASE), "Working on the picture"),
    (re.compile(r"spawn|delegate|(?:^|_)agents?(?:_|$)|^agent", re.IGNORECASE), "Checking with the team"),
    (re.compile(r"^(?:exec|bash|shell|process|read|write|edit|apply_patch)$", re.IGNORECASE), "Working on it"),
)


def tool_status_line(tool_name: str | None,
                     rules: tuple[tuple[re.Pattern, str], ...] = ()) -> str | None:
    """One spoken status for a gateway tool call: the operator's ``rules`` first
    (``voice-acks.yaml: tools``), then the generic table — ``*search*`` →
    "Searching", ``*calendar*`` → "Checking your calendar", … — anything else →
    "Looking that up"; None for an empty name (not a tool)."""
    name = (tool_name or "").strip()
    if not name:
        return None
    for rx, line in (*rules, *_TOOL_LINES):
        if rx.search(name):
            return line
    return "Looking that up"


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #

class EtaEstimator:
    """Per-session estimate of "transcript → first reply audio", in seconds.

    EMA of the LLM's time to first token and the voice's first-chunk synthesis
    time; each component uses its prior (the lab's measured ``first_chunk_ms``,
    1.2 s TTFT) until it has two samples."""

    def __init__(self, *, first_chunk_s: float, ttft_s: float = 1.2, alpha: float = 0.3) -> None:
        self.prior = {"ttft": float(ttft_s), "chunk": float(first_chunk_s)}
        self.ema: dict[str, float | None] = {"ttft": None, "chunk": None}
        self.n = {"ttft": 0, "chunk": 0}
        self.alpha = alpha

    def observe(self, *, ttft_s: float | None = None, first_chunk_s: float | None = None) -> None:
        for k, v in (("ttft", ttft_s), ("chunk", first_chunk_s)):
            if v is None or v < 0:
                continue
            cur = self.ema[k]
            self.ema[k] = float(v) if cur is None else cur + self.alpha * (float(v) - cur)
            self.n[k] += 1

    def _part(self, k: str) -> float:
        return self.ema[k] if self.n[k] >= 2 and self.ema[k] is not None else self.prior[k]

    @property
    def eta_s(self) -> float:
        return self._part("ttft") + self._part("chunk")


class TurnAcks:
    """The acknowledgement plan of ONE turn. Pure: feed it events, poll it with
    the session clock, play what :meth:`poll` returns.

    * ``ack`` at ``t0 + ACK_AFTER_S`` (0.6 s here; the host sets ``ack_after_s``,
      0.55 s) — skipped when a reply sentence is already
      queued/synthesising, when the ETA of first audio is under 1.2 s, or when
      the clip would not end 0.3 s before that ETA;
    * ``tier`` ``wait_5s`` at 5 s and ``wait_15s`` at 15 s, once each, only while
      no reply audio has started, never while anything plays, never after an
      error; a tool status line replaces the 5 s tier;
    * ``tool`` lines: once per distinct tool (and distinct phrase), ≤ 2 per turn,
      at least ``min_gap_s`` after the previous line of this turn.
    ``t0`` is when the final transcript was taken (end of turn)."""

    ACK_AFTER_S = 0.6
    MIN_ETA_S = 1.2
    END_MARGIN_S = 0.3
    TIER_5S = 5.0
    TIER_15S = 15.0
    MAX_TOOL_LINES = 2
    MIN_GAP_S = 3.0

    def __init__(self, selector: AckSelector, *, bot_id: str, voice: str, intent: str,
                 t0: float, eta_s: float, last_turn_at: float | None = None,
                 last_acked: bool = False, clip_s: Callable[[str], float | None] | None = None,
                 min_gap_s: float | None = None) -> None:
        self.sel = selector
        self.bot_id, self.voice, self.intent = bot_id, voice, intent
        self.t0, self.eta_s = t0, eta_s
        self.last_turn_at, self.last_acked = last_turn_at, last_acked
        self._clip = clip_s
        self.min_gap_s = self.MIN_GAP_S if min_gap_s is None else min_gap_s
        self.reply_queued = False
        self.reply_started = False
        self.is_speaking = False
        self.errored = False
        self.first_word: str | None = None
        self.ack_decided = False
        self.acked = False
        #: Why the ``ACK_AFTER_S`` decision came out as it did: ``gate`` (a reply was queued,
        #: something was playing or the ETA was short — no line wanted) or the
        #: selector's ``last_outcome`` (``picked``, ``skipped``, ``none_fit`` …).
        self.ack_outcome = ""
        self.tier5_done = False
        self.tier15_done = False
        self.tools_seen: set[str] = set()
        self.tool_lines: list[str] = []
        self._pending_tools: deque[str] = deque()
        self._last_line_at: float | None = None
        self.said: list[tuple[str, str]] = []

    # ---- events ------------------------------------------------------------ #

    def on_reply_queued(self) -> None:
        """A reply sentence is queued or being synthesised."""
        self.reply_queued = True

    def on_reply_audio(self) -> None:
        """The first reply audio started playing: no tiers or tool lines after this."""
        self.reply_queued = self.reply_started = True
        self._pending_tools.clear()

    def on_speaking(self, speaking: bool) -> None:
        self.is_speaking = bool(speaking)

    def on_error(self) -> None:
        self.errored = True
        self._pending_tools.clear()

    def on_reply_text(self, text: str) -> None:
        """Speculative/streamed reply text known so far (its first word is avoided)."""
        if text and not self.first_word:
            w = _first_word(text)
            self.first_word = w or None

    def on_tool(self, tool_name: str | None) -> None:
        if self.reply_started or self.errored:
            return
        line = tool_status_line(tool_name, getattr(self.sel.pools, "tool_rules", ()))
        name = (tool_name or "").strip().lower()
        if not line or name in self.tools_seen:
            return
        self.tools_seen.add(name)
        if line in self.tool_lines or line in self._pending_tools:
            return
        if len(self.tool_lines) + len(self._pending_tools) >= self.MAX_TOOL_LINES:
            return
        self._pending_tools.append(line)

    # ---- clock ------------------------------------------------------------- #

    def _clip_s(self, text: str) -> float | None:
        """Length of the line's clip. With a ``clip_s`` callable (the session passes
        ``RenderQueue.duration_s``) None means "not rendered in this voice yet", and
        such a line is never picked (the earcon covers it); without one, the
        estimate is used."""
        if self._clip is None:
            return estimate_clip_s(text)
        v = self._clip(text)
        return None if v is None else float(v)

    def _fits(self, now: float, deadline: float | None) -> Callable[[str], bool]:
        def ok(text: str) -> bool:
            c = self._clip_s(text)
            return c is not None and (deadline is None or now + c <= deadline)
        return ok

    def _gap_ok(self, now: float) -> bool:
        return self._last_line_at is None or now - self._last_line_at >= self.min_gap_s

    def _spoke(self, kind: str, text: str, now: float) -> tuple[str, str]:
        self._last_line_at = now
        self.said.append((kind, text))
        return kind, text

    def poll(self, now: float) -> tuple[Literal["ack", "tier", "tool"], str] | None:
        """The line due now, if any (call it whenever an event lands and at
        :meth:`next_due`)."""
        if self.errored:
            return None
        if not self.ack_decided and now >= self.t0 + self.ACK_AFTER_S:
            self.ack_decided = True
            if not self.reply_queued and not self.is_speaking and self.eta_s >= self.MIN_ETA_S:
                deadline = self.t0 + self.eta_s - self.END_MARGIN_S
                line = self.sel.pick(
                    self.intent, bot_id=self.bot_id, voice=self.voice, now=self.t0,
                    last_turn_at=self.last_turn_at, last_acked=self.last_acked,
                    reply_first_word=self.first_word, fits=self._fits(now, deadline))
                self.ack_outcome = getattr(self.sel, "last_outcome", "") or ("picked" if line else "")
                if line:
                    self.acked = True
                    return self._spoke("ack", line, now)
            else:
                self.ack_outcome = "gate"
        if self.reply_started or self.is_speaking:
            return None
        if self._pending_tools and self._gap_ok(now):
            line = self._pending_tools.popleft()
            self.tool_lines.append(line)
            self.sel.note_said(line)
            return self._spoke("tool", line, now)
        if (not self.tier5_done and not self.tool_lines and now >= self.t0 + self.TIER_5S
                and self._gap_ok(now)):
            self.tier5_done = True
            line = self.sel.tier("wait_5s", bot_id=self.bot_id, voice=self.voice,
                                 fits=self._fits(now, None))
            if line:
                return self._spoke("tier", line, now)
        if not self.tier15_done and now >= self.t0 + self.TIER_15S and self._gap_ok(now):
            self.tier15_done = True
            line = self.sel.tier("wait_15s", bot_id=self.bot_id, voice=self.voice,
                                 fits=self._fits(now, None))
            if line:
                return self._spoke("tier", line, now)
        return None

    def next_due(self, now: float) -> float | None:
        """When :meth:`poll` should be called next (None = nothing scheduled)."""
        if self.errored or self.reply_started:
            return None
        due = []
        if not self.ack_decided:
            due.append(self.t0 + self.ACK_AFTER_S)
        gap_at = (self._last_line_at + self.min_gap_s) if self._last_line_at is not None else now
        if self._pending_tools:
            due.append(max(now, gap_at))
        if not self.tier5_done and not self.tool_lines:
            due.append(max(self.t0 + self.TIER_5S, gap_at))
        if not self.tier15_done:
            due.append(max(self.t0 + self.TIER_15S, gap_at))
        return min(due) if due else None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

RenderFn = Callable[[str, str, int], "tuple[object, int]"]


class RenderQueue:
    """Background renders of acknowledgement lines, one engine call at a time.

    ``render(voice, text, seed) -> (int16 samples, sample_rate)`` does the
    synthesis (the host wires the engine's ``render_line``/``synth``);
    ``busy()`` is True while a reply is being synthesised or any session is
    thinking/speaking — the queue waits, it never competes with a reply.
    ``cache_dir(voice)`` is the voice's disk-cache directory (the engine's
    cache tag, e.g. ``cb-<model sha8>-<profile>-<created>``), or None for no
    disk cache. Files are ``cache_key(text)``, written after the sanity check
    (expected ≈ 0.4 + 0.35 × words s; > 2.5× or < 0.3 s → one re-render with
    ``seed + 1``, else the line is dropped for that voice)."""

    MAX_FILES_PER_TAG = 2000
    LRU_VOICES = 8
    POLL_S = 0.25

    def __init__(self, render: RenderFn, *, busy: Callable[[], bool] = lambda: False,
                 cache_dir: Callable[[str], Path | None] | None = None,
                 keep: Callable[[], Iterable[str]] | None = None, nice: int = 15,
                 autostart: bool = True, default_rate: int = 24000) -> None:
        self._render = render
        self._busy = busy
        self._cache_dir = cache_dir or (lambda voice: None)
        self._keep = keep
        self._nice = nice
        self._autostart = autostart
        self._cv = threading.Condition()
        self._queue: deque[tuple[str, str]] = deque()
        self._queued: set[tuple[str, str]] = set()
        self._wanted: dict[str, list[str]] = {}
        self._failed: set[tuple[str, str]] = set()
        self._mem: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._thread: threading.Thread | None = None
        self._stop = False
        self._working: tuple[str, str] | None = None
        self._rates: dict[str, int] = {}
        self.default_rate = default_rate
        self.renders = 0
        self._bad_dirs: set[str] = set()
        self._busy_failing = False

    # ---- public API (contracts B.3) ---------------------------------------- #

    def ensure(self, voice: str, lines: Iterable[str]) -> None:
        """Enqueue every (text, voice) pair that is neither cached nor queued."""
        voice = voice or ""
        broken = self._dir_broken(voice)
        with self._cv:
            wanted = self._wanted.setdefault(voice, [])
            for text in lines:
                if not text or not isinstance(text, str):
                    continue
                if text not in wanted:
                    wanted.append(text)
                pair = (voice, text)
                if broken:                      # a bad profile tag: no acks for it, no raise
                    self._failed.add(pair)
                    continue
                if pair in self._queued or pair in self._failed or self._have(voice, text):
                    continue
                self._queue.append(pair)
                self._queued.add(pair)
            self._cv.notify_all()
        if self._autostart:
            self.start()

    def get(self, voice: str, text: str):
        """The rendered int16 samples, or None (not rendered yet / dropped)."""
        voice = voice or ""
        with self._cv:
            mem = self._mem.get(voice)
            if mem is not None and text in mem:
                self._mem.move_to_end(voice)
                return mem[text]
        pcm = self._load(voice, text)
        if pcm is not None:
            with self._cv:
                self._remember(voice, text, pcm)
        return pcm

    def duration_s(self, voice: str, text: str) -> float | None:
        """Seconds of the rendered clip, or None when it is not rendered (yet)."""
        pcm = self.get(voice, text)
        if pcm is None:
            return None
        return len(pcm) / float(self._rates.get(voice or "", self.default_rate))

    def status(self) -> dict:
        """``{voice: {"ready": n, "total": m, "failed": k}}`` for the Voice lab
        row. ``failed`` lines were dropped (runaway clip, engine error, bad cache
        directory) and stay out of ``ready``, so the row can read "61 of 62
        (1 skipped)" instead of waiting for a line that will never come."""
        with self._cv:
            wanted = {v: list(t) for v, t in self._wanted.items()}
        out = {}
        for voice, texts in wanted.items():
            with self._cv:
                ready = sum(1 for t in texts if self._have(voice, t))
                failed = sum(1 for t in texts if (voice, t) in self._failed
                             and not self._have(voice, t))
            out[voice] = {"ready": ready, "total": len(texts), "failed": failed}
        return out

    def rebuild(self, voice: str) -> None:
        """Forget and delete this voice's rendered lines, prune its cache, re-enqueue."""
        voice = voice or ""
        with self._cv:
            texts = list(self._wanted.get(voice, []))
            self._mem.pop(voice, None)
            self._failed = {p for p in self._failed if p[0] != voice}
            self._queue = deque(p for p in self._queue if p[0] != voice)
            self._queued = {p for p in self._queued if p[0] != voice}
        self._bad_dirs.discard(voice)
        d = self._dir(voice)
        if d is not None:
            for t in texts:
                with contextlib.suppress(OSError):
                    (Path(d) / cache_key(t)).unlink()
        self.prune(voice)
        self.ensure(voice, texts)

    # ---- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        with self._cv:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = False
            self._thread = threading.Thread(target=self._run, name="voice-ack-render", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
            t = self._thread
        if t is not None:
            t.join(timeout)

    def clear(self) -> None:
        """Drop the in-memory clips (``release_models``); the disk cache stays."""
        with self._cv:
            self._mem.clear()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until nothing is queued or rendering (tests, warm-once)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while self._queue or self._working is not None:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    return False
                self._cv.wait(0.05 if left is None else min(0.05, left))
        return True

    TMP_GRACE_S = 60.0

    def prune(self, voice: str, keep: Iterable[str] | None = None) -> int:
        """Delete cached ``.npy`` files of this voice that no current pool (or
        the host's ``keep()`` list, e.g. notice lines) names; then cap the tag at
        :data:`MAX_FILES_PER_TAG`, oldest first. Returns the number removed.

        A no-op (with a warning) before :meth:`ensure` has named this voice's
        lines — pruning then would wipe the tag. Another writer's ``*.tmp.npy``
        younger than :data:`TMP_GRACE_S` is left alone (it is mid-write)."""
        with self._cv:
            wanted = list(self._wanted.get(voice or "", []))
        if not wanted:
            log.warning("voice: ack prune for %s skipped: call ensure() first", voice or "default")
            return 0
        d = self._dir(voice or "")
        if d is None:
            return 0
        d = Path(d)
        names: set[str] = {cache_key(t) for t in wanted}
        now = time.time()
        for src in (keep, self._keep() if self._keep else None):
            for t in src or ():
                names.add(cache_key(t))
        removed = 0
        try:
            files = [p for p in d.iterdir() if p.suffix == ".npy" and p.is_file()]
        except OSError:
            return 0
        survivors = []
        for p in files:
            if p.name.endswith(".tmp.npy"):
                try:
                    young = now - p.stat().st_mtime < self.TMP_GRACE_S
                except OSError:
                    continue
                if young:
                    continue
            if p.name not in names:
                with contextlib.suppress(OSError):
                    p.unlink()
                    removed += 1
            else:
                survivors.append(p)
        if len(survivors) > self.MAX_FILES_PER_TAG:
            def mtime(p: Path) -> float:
                try:
                    return p.stat().st_mtime
                except OSError:
                    return 0.0
            survivors.sort(key=mtime)
            for p in survivors[: len(survivors) - self.MAX_FILES_PER_TAG]:
                with contextlib.suppress(OSError):
                    p.unlink()
                    removed += 1
        return removed

    # ---- internals --------------------------------------------------------- #

    def _dir(self, voice: str) -> Path | None:
        """``cache_dir(voice)`` that never raises: a bad profile tag is logged
        once and means "no disk cache" (and :meth:`ensure` marks it failed)."""
        try:
            d = self._cache_dir(voice)
        except Exception:
            if voice not in self._bad_dirs:
                self._bad_dirs.add(voice)
                log.exception("voice: ack cache directory for %s failed; no acks for it",
                              voice or "default")
            return None
        return None if d is None else Path(d)

    def _dir_broken(self, voice: str) -> bool:
        self._dir(voice)
        return voice in self._bad_dirs

    def _is_busy(self) -> bool:
        """``busy()`` that never kills the render thread: an exception counts as
        busy (never compete with a reply) and is logged once per failing streak."""
        try:
            busy = bool(self._busy())
        except Exception:
            if not self._busy_failing:
                self._busy_failing = True
                log.exception("voice: ack render busy() check failed; treating as busy")
            return True
        self._busy_failing = False
        return busy

    def _have(self, voice: str, text: str) -> bool:
        mem = self._mem.get(voice)
        if mem is not None and text in mem:
            return True
        d = self._dir(voice)
        return d is not None and (d / cache_key(text)).is_file()

    def _remember(self, voice: str, text: str, pcm) -> None:
        mem = self._mem.setdefault(voice, {})
        mem[text] = pcm
        self._mem.move_to_end(voice)
        while len(self._mem) > self.LRU_VOICES:
            self._mem.popitem(last=False)

    def _load(self, voice: str, text: str):
        d = self._dir(voice)
        if d is None:
            return None
        f = Path(d) / cache_key(text)
        if not f.is_file():
            return None
        try:
            import numpy as np
            return np.load(f).astype("<i2")
        except (OSError, ValueError, EOFError):   # a torn or foreign file: render again
            return None

    def _save(self, voice: str, text: str, pcm) -> None:
        d = self._dir(voice)
        if d is None:
            return
        import numpy as np
        d = Path(d)
        f = d / cache_key(text)
        try:
            d.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f.stem + f".{os.getpid()}.tmp.npy")
            np.save(tmp, np.asarray(pcm, "<i2"))
            tmp.replace(f)
        except OSError as e:
            log.warning("voice: ack cache write failed: %s", e)

    @staticmethod
    def _sane(text: str, n: int, rate: int) -> bool:
        if rate <= 0:
            return False
        dur = n / rate
        return 0.3 <= dur <= 2.5 * estimate_clip_s(text)

    def _render_one(self, voice: str, text: str):
        import numpy as np
        seed = line_seed(text)
        for s in (seed, seed + 1):
            pcm, rate = self._render(voice, text, s)
            pcm = np.asarray(pcm, "<i2").reshape(-1)
            self.renders += 1
            if self._sane(text, len(pcm), int(rate)):
                self._rates[voice] = int(rate)
                return pcm
            log.info("voice: ack %r rendered %.2f s in %s (expected ≈ %.2f s); %s",
                     text, len(pcm) / max(1, int(rate)), voice or "default",
                     estimate_clip_s(text), "retrying" if s == seed else "dropped")
        return None

    def _run(self) -> None:
        with contextlib.suppress(Exception):
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), self._nice)
        while True:
            with self._cv:
                while not self._stop and not self._queue:
                    self._cv.wait()
                if self._stop:
                    return
            while not self._stop and self._is_busy():
                time.sleep(self.POLL_S)
            with self._cv:
                if self._stop:
                    return
                if not self._queue:
                    continue
                pair = self._queue.popleft()
                self._queued.discard(pair)
                self._working = pair
            voice, text = pair
            try:
                with self._cv:
                    have = self._have(voice, text)
                if not have:
                    pcm = self._render_one(voice, text)
                    if pcm is None:
                        with self._cv:
                            self._failed.add(pair)
                    else:
                        self._save(voice, text, pcm)
                        with self._cv:
                            self._remember(voice, text, pcm)
            except Exception:
                log.exception("voice: ack render failed (%s)", voice or "default")
                with self._cv:
                    self._failed.add(pair)
            finally:
                with self._cv:
                    self._working = None
                    self._cv.notify_all()
