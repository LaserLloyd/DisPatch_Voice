"""What a reply sounds like: sentence spans, speech normalisation and role-play segments.

Per reply, in this order (SPEC §6.3; the order matters, see the four traps below):

1. :class:`SentenceSplitter` cuts the streamed markdown into sentences as soon as
   each is complete. It holds back while a code fence, a ``[[…]]`` directive, a
   ``*…*`` action or a quotation is still open (actions up to 200 characters,
   quotations up to 800) and every sentence carries its :class:`Span` in the SOURCE
   content, so heard-text tracking can cut the stored markdown at a word boundary.
2. :func:`normalise` turns one sentence into plain words:
   :func:`strip_markdown` → tag filter → :func:`strip_symbols` → :func:`expand_numbers`.
3. :func:`rp_render` (built on :func:`segment`) splits the normalised sentence into
   ``(role, text)`` pieces for the character and narrator voices.

The four ordering traps (regression tests in ``tests/text/test_speech_text.py``):
a URL needs an alphabetic top-level domain or it eats decimals ("2.5 GB"); the
single-letter units s/m/g/h need a space or "1990s" becomes "seconds"; money must
not swallow the space after it; CJK-heavy text gets no English number words.

Pure Python, no dependencies, no I/O. Never imports ``host/``.
"""
from __future__ import annotations

import dataclasses
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "CHATTERBOX_TAGS",
    "RP_MODES",
    "RpRenderer",
    "SentenceSplitter",
    "Span",
    "expand_numbers",
    "filter_tags",
    "is_cjk_heavy",
    "normalise",
    "number_words",
    "rp_render",
    "segment",
    "strip_markdown",
    "strip_symbols",
]

# --------------------------------------------------------------------------- #
# Spans
# --------------------------------------------------------------------------- #

# ONE Span class for the whole add-on: rt.heard defines it (contracts B.3, T4's
# heard map consumes it), the splitter fills it. Amendment T5-2: the splitter
# sets ``text`` (and ``source``) to the raw source slice, ``source[start:end]``;
# the session swaps ``text`` for the normalised speech (``dataclasses.replace``)
# before it registers the span in the heard map, so the time map runs over what
# the engine really said and ``source_offset`` still sees the markup.
from ..rt.heard import Span

_SPAN_HAS_SOURCE = "source" in {f.name for f in dataclasses.fields(Span)}


def _span(text: str, start: int, end: int) -> Span:
    if _SPAN_HAS_SOURCE:
        return Span(text=text, start=start, end=end, source=text)
    return Span(text=text, start=start, end=end)


# --------------------------------------------------------------------------- #
# Role-play markup scanner (shared by the splitter and segment())
# --------------------------------------------------------------------------- #

#: Quote pairs (outermost only). „ closes with “ (German), so “ is both an opener
#: (English) and a closer (after „); the stack decides.
QUOTE_PAIRS = {'"': '"', "\u201c": "\u201d", "\u00ab": "\u00bb", "\u300c": "\u300d",
               "\u300e": "\u300f", "\u201e": "\u201c"}
_OPEN_PUNCT = set("([{\u2014\u2013-:,;/")
_FENCE_OR_DIRECTIVE = re.compile(r"```.*?(?:```|$)|\[\[.*?(?:\]\]|$)|`[^`\n]*`", re.DOTALL)

ACTION_CAP = 200     # an open *action* holds the splitter this long, then it is prose
QUOTE_CAP = 800      # an open quotation holds it this long


def _is_word(ch: str) -> bool:
    return bool(ch) and (ch.isalnum() or ch == "_")


def _straight_quote_opens(text: str, i: int) -> bool:
    prev = text[i - 1] if i > 0 else ""
    nxt = text[i + 1] if i + 1 < len(text) else ""
    if not nxt or nxt.isspace():
        return False
    return (not prev) or prev.isspace() or prev in _OPEN_PUNCT or prev in QUOTE_PAIRS


def _star_opens(text: str, i: int) -> bool:
    nxt = text[i + 1] if i + 1 < len(text) else ""
    return bool(nxt) and not nxt.isspace() and nxt != "*"


def _star_closes(text: str, i: int) -> bool:
    prev = text[i - 1] if i > 0 else ""
    return bool(prev) and not prev.isspace()


@dataclass
class _Mark:
    kind: Literal["action", "quote"]
    open: int            # index of the opening delimiter
    close: int | None    # index of the closing delimiter, None while unclosed
    closer: str = ""


def _scan_marks(text: str) -> list[_Mark]:
    """Outermost ``*actions*`` and quotations in ``text``.

    Code fences, inline code and ``[[…]]`` directives are skipped. ``**``/``__``
    runs are markdown emphasis, never an action delimiter. A ``*`` with spaces on
    both sides is a literal (a bullet, a multiplication). A straight ``"`` opens
    only after a space/opening punctuation and before a non-space (so an inch
    mark never opens a quotation); typographic openers always open.
    """
    skip: list[tuple[int, int]] = [m.span() for m in _FENCE_OR_DIRECTIVE.finditer(text)]
    marks: list[_Mark] = []
    cur: _Mark | None = None
    i, n, k = 0, len(text), 0
    while i < n:
        while k < len(skip) and skip[k][1] <= i:
            k += 1
        if k < len(skip) and skip[k][0] <= i < skip[k][1]:
            i = skip[k][1]
            continue
        ch = text[i]
        if ch == "*":
            j = i
            while j < n and text[j] == "*":
                j += 1
            run = j - i
            if run >= 2:                      # ** or *** : markdown emphasis, not RP
                i = j
                continue
            if cur is None:
                if _star_opens(text, i):
                    cur = _Mark("action", i, None, "*")
            elif cur.kind == "action" and _star_closes(text, i):
                cur.close = i
                marks.append(cur)
                cur = None
            i += 1
            continue
        if cur is None and ch in QUOTE_PAIRS:
            if ch != '"' or _straight_quote_opens(text, i):
                cur = _Mark("quote", i, None, QUOTE_PAIRS[ch])
            i += 1
            continue
        if cur is not None and cur.kind == "quote" and ch == cur.closer:
            cur.close = i
            marks.append(cur)
            cur = None
        i += 1
    if cur is not None:
        marks.append(cur)
    return marks


# --------------------------------------------------------------------------- #
# Sentence splitting
# --------------------------------------------------------------------------- #

_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
           "approx", "no", "fig", "inc", "ltd", "co", "jan", "feb", "mar", "apr", "jun",
           "jul", "aug", "sep", "sept", "oct", "nov", "dec", "a.m", "p.m", "u.s"}
# A sentence end: terminal punctuation plus any closing quotes/brackets/asterisk,
# then whitespace; a blank line; or a line break before a list item.
_END_RE = re.compile(
    r"([.!?…]+[\"')\]*\u201d\u00bb\u300d\u300f]*)(\s+)|(\n\s*\n)|(\n)(?=\s*(?:[-*+•]|\d+[.)])\s)")
_CLOSE_QUOTE = re.compile("[\"'\u201d\u00bb\u300d\u300f]")
_CODE_SPAN_RE = (re.compile(r"```.*?```", re.DOTALL), re.compile(r"\[\[.*?\]\]", re.DOTALL))


class SentenceSplitter:
    """Push text as it streams; get back complete sentences as :class:`Span` s.

    Holds back while a code fence, a ``[[…]]`` directive, a ``*…*`` action
    (≤ ``action_cap`` chars) or a quotation (≤ ``quote_cap`` chars) is still
    open, so a sentence never ends inside one, and never cuts inside a closed
    action or quotation either. A first chunk that runs long without a full stop
    is cut at a comma — the first words matter most for latency. A fragment of
    fewer than ``min_words`` words waits and is spoken with the next sentence.

    Offsets are into the concatenation of everything pushed since construction
    (= the stored reply content when the session pushes every delta).
    """

    def __init__(self, first_soft_chars: int = 48, hard_chars: int = 260,
                 min_words: int = 4, first_max_chars: int = 80, *,
                 action_cap: int = ACTION_CAP, quote_cap: int = QUOTE_CAP) -> None:
        self._src = ""
        self._pos = 0                      # start of the unconsumed buffer in _src
        self._held: tuple[int, int] | None = None
        self._first = True
        self.min_words = min_words
        self.first_soft_chars = first_soft_chars
        self.first_max_chars = max(first_max_chars, first_soft_chars)
        self.hard_chars = hard_chars
        self.action_cap = action_cap
        self.quote_cap = quote_cap

    # ---- public ----------------------------------------------------------- #

    def push(self, text: str) -> list[Span]:
        self._src += text
        return self._drain(final=False)

    def flush(self) -> list[Span]:
        out = self._drain(final=True)
        buf = self._src[self._pos:]
        b0 = self._pos + (len(buf) - len(buf.lstrip()))
        b1 = self._pos + len(buf.rstrip())
        start = self._held[0] if self._held else (b0 if b1 > b0 else None)
        end = b1 if b1 > b0 else (self._held[1] if self._held else None)
        self._held = None
        self._pos = len(self._src)
        if start is not None and end is not None and end > start:
            out.append(_span(self._src[start:end], start, end))
        return out

    def split(self, content: str) -> list[Span]:
        """Every sentence of a whole reply (resets the splitter first)."""
        self.__init__(self.first_soft_chars, self.hard_chars, self.min_words,
                      self.first_max_chars, action_cap=self.action_cap,
                      quote_cap=self.quote_cap)
        return self.push(content) + self.flush()

    @property
    def source(self) -> str:
        return self._src

    # ---- internals -------------------------------------------------------- #

    def _open_construct(self, s: str) -> int | None:
        """Index where an unclosed fence/directive/action/quote starts, else None."""
        fences = [m.start() for m in re.finditer(r"```", s)]
        if len(fences) % 2:
            return fences[-1]
        i = s.rfind("[[")
        if i != -1 and s.find("]]", i) == -1:
            return i
        for m in _scan_marks(s):
            if m.close is None:
                cap = self.action_cap if m.kind == "action" else self.quote_cap
                if len(s) - m.open <= cap:
                    return m.open
        return None

    def _drain(self, final: bool) -> list[Span]:
        out: list[Span] = []
        while True:
            buf = self._src[self._pos:]
            hold = None if final else self._open_construct(buf)
            scan = buf if hold is None else buf[:hold]
            cut = self._find_cut(scan, final)
            if cut is None:
                break
            raw = buf[:cut]
            s0 = self._pos + (len(raw) - len(raw.lstrip()))
            s1 = self._pos + len(raw.rstrip())
            rest = buf[cut:]
            self._pos += cut + (len(rest) - len(rest.lstrip()))
            if s1 <= s0:
                continue
            start = self._held[0] if self._held else s0
            joined = self._src[start:s1]
            # Prosody: a two-word fragment on its own sounds clipped. Hold it
            # and say it together with the next sentence.
            if len(joined.split()) < self.min_words and not final:
                self._held = (start, s1)
                continue
            self._held = None
            out.append(_span(joined, start, s1))
            self._first = False
        return out

    def _protected(self, s: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """(code spans, role-play spans) a sentence must not end inside."""
        code = [m.span() for rx in _CODE_SPAN_RE for m in rx.finditer(s)]
        rp = []
        for m in _scan_marks(s):
            if m.close is None:
                continue
            cap = self.action_cap if m.kind == "action" else self.quote_cap
            if m.close - m.open <= cap:
                rp.append((m.open, m.close + 1))
        return code, rp

    def _clause_cut(self, s: str, spans) -> int | None:
        for limit in (self.first_soft_chars, self.first_max_chars):
            w = s[:limit]
            j = max(w.rfind(", "), w.rfind("; "), w.rfind(": "), w.rfind(" — "))
            if j > 20 and not any(a <= j < b for a, b in spans):
                return j + 1
        return None

    def _find_cut(self, s: str, final: bool) -> int | None:
        code, rp = self._protected(s)
        spans = code + rp
        for m in _END_RE.finditer(s):
            if any(a <= m.start() < b for a, b in code):
                continue
            # Inside a closed action/quotation only when the match ends inside it;
            # 'gym day."' ends the sentence together with its closing quote.
            if any(a <= m.start() < b and m.end(1 if m.group(1) else 0) < b for a, b in rp):
                continue
            if m.group(1):
                end_word = s[: m.start(1)].rsplit(None, 1)
                word = end_word[-1].lower().rstrip(".") if end_word else ""
                word = word.lstrip("*\"'(\u201c")
                # '"No."' closed by a quotation mark ends the sentence; "No." is
                # the number abbreviation only before a digit ("No. 5").
                quoted = bool(_CLOSE_QUOTE.search(m.group(1)))
                nxt = s[m.end():m.end() + 1]
                abbrev = word in _ABBREV and not (word == "no" and nxt and not nxt.isdigit())
                if (m.group(1).startswith(".") and not quoted
                        and (abbrev or (len(word) == 1 and word.isalpha()))):
                    continue
                if self._first and m.end(1) > self.first_max_chars:
                    j = self._clause_cut(s, spans)
                    if j is not None:
                        return j
                return m.end(1)
            if m.group(3):
                return m.end(3)
            if m.group(4):
                return m.end(4)
        limit = self.first_soft_chars if self._first else self.hard_chars
        if code and len(s) > limit and not final:
            a, b = code[0]
            if a < limit:
                return b if len(s) > b > 0 and a == 0 else (a if a > 20 else None)
        if len(s) > limit and not final:
            def clause_end(window: str) -> int:
                j = max(window.rfind(", "), window.rfind("; "), window.rfind(": "),
                        window.rfind(" — "))
                while j > 20 and any(a <= j < b for a, b in rp):
                    j = max(window.rfind(", ", 0, j), window.rfind("; ", 0, j),
                            window.rfind(": ", 0, j), window.rfind(" — ", 0, j))
                return j
            j = clause_end(s[:limit])
            if j > 20:
                return j + 1
            if self._first:
                if len(s) <= self.first_max_chars:
                    return None
                j = clause_end(s[:self.first_max_chars])
                if j > 20:
                    return j + 1
            if len(s) > self.hard_chars:
                window = s[:limit]
                k = window.rfind(" ")
                for a, b in rp:               # never inside a closed action/quote
                    if a < k < b and a > 0:
                        k = s.rfind(" ", 0, a) if s.rfind(" ", 0, a) > 0 else a
                return k if k > 0 else limit
        return None


# --------------------------------------------------------------------------- #
# Normalisation: markdown
# --------------------------------------------------------------------------- #

CHATTERBOX_TAGS = frozenset({"[laugh]", "[chuckle]", "[sigh]", "[gasp]", "[cough]",
                             "[clear throat]", "[sniff]", "[groan]", "[shush]"})

_FENCE = re.compile(r"```.*?(?:```|$)", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]*)\)")
# A bare domain needs an alphabetic, lowercase TLD so decimals ("2.5 GB") and
# abbreviations ("e.g.") never read as links (trap 1).
_URL = re.compile(
    r"(?i:\b(?:https?://(?:www\.)?|www\.)?)"
    r"((?i:[a-z0-9-]+)(?:\.(?i:[a-z0-9-]+))*\.(?-i:[a-z]{2,}))(?:[/?#]\S*)?(?=[\s,;:!?)\]]|\.(?:\s|$)|$)")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_HEADING = re.compile(r"^\s{0,3}#{1,6}(?:\s+|$)", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^\s*>\s?", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE)
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$", re.MULTILINE)
_EMPH = re.compile(r"(\*\*\*|\*\*|__|~~)(?=\S)(.+?)(?<=\S)\1")
_UNDERSCORE_EMPH = re.compile(r"(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)")
_HTML = re.compile(r"</?[a-zA-Z][^>]{0,200}>")
_DIRECTIVE = re.compile(r"\[\[[^\]]*(?:\]\]|$)", re.DOTALL)
_REACT = re.compile(r":react:[A-Za-z0-9_\-]+:", re.IGNORECASE)
_TAG = re.compile(r"\[([a-zA-Z][a-zA-Z ]{0,20})\]")
_ABBREV_WORDS = [(re.compile(r"\be\.g\.,?", re.IGNORECASE), "for example,"),
                 (re.compile(r"\bi\.e\.,?", re.IGNORECASE), "that is,"),
                 (re.compile(r"\betc\.(?=\s+[a-z])"), "et cetera"),
                 (re.compile(r"\betc\."), "et cetera."),
                 (re.compile(r"\bvs\.?(?=\s)", re.IGNORECASE), "versus"),
                 (re.compile(r"\bapprox\.(?=\s)", re.IGNORECASE), "about")]


def _table_line(line: str) -> str:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    cells = [c for c in cells if c]
    if not cells:
        return ""
    body = ", ".join(cells)
    return body if body[-1] in ".!?" else body + "."


# A gateway agent that echoes the app note back (it happened 1 in 24 on "how does
# this work?") must never read it out: the chat keeps it, the voice drops it.
# The note ends "Never mention this note.)"; the quoted heard text inside it may
# itself contain ")", so the end marker wins over the first ")".
_NOTE_ECHO = re.compile(
    r"\[voice chat\]\s*(?:\(app note(?:.*?never mention this note\.?\)|[^)]*(?:\)|$)))?"
    r"|\(app note, not from the user(?:.*?never mention this note\.?\)|[^)]*(?:\)|$))",
    re.IGNORECASE | re.DOTALL)


def strip_markdown(t: str) -> str:
    """Markdown/HTML/links/directives → plain text (the RP ``*`` and quotes stay)."""
    t = _NOTE_ECHO.sub(" ", t)
    t = _FENCE.sub(" The code is in the chat. ", t)
    t = _DIRECTIVE.sub(" ", t)
    t = _REACT.sub(" ", t)
    t = _IMG.sub(" ", t)
    t = _LINK.sub(r"\1", t)
    t = _EMAIL.sub(lambda m: m.group(0).replace("@", " at ").replace(".", " dot "), t)
    t = _URL.sub(lambda m: " " + m.group(1).replace(".", " dot ") + " ", t)
    t = _INLINE_CODE.sub(r"\1", t)
    t = _TABLE_RULE.sub(" ", t)
    lines = []
    for line in t.split("\n"):
        if line.strip().startswith("|") and line.count("|") >= 2:
            line = _table_line(line)
        lines.append(line)
    t = "\n".join(lines)
    t = _HEADING.sub("", t)
    t = _BLOCKQUOTE.sub("", t)
    t = _BULLET.sub("", t)
    for _ in range(2):
        t = _EMPH.sub(r"\2", t)
    t = _UNDERSCORE_EMPH.sub(r"\1", t)
    t = _HTML.sub(" ", t)
    for rx, rep in _ABBREV_WORDS:
        t = rx.sub(rep, t)
    # A line break is a pause: give a line that has no end punctuation one.
    parts = [p.strip() for p in t.split("\n")]
    parts = [p for p in parts if p]
    joined = []
    for i, p in enumerate(parts):
        if i < len(parts) - 1 and p[-1] not in ".!?:;,…" and not p.endswith(("*", '"', "\u201d")):
            p += "."
        joined.append(p)
    return " ".join(joined).replace("|", ", ")


def filter_tags(t: str, *, tags_ok: bool) -> str:
    """Keep the nine Chatterbox tags for a tag-capable voice; drop every other ``[tag]``."""
    def rep(m: re.Match) -> str:
        tag = m.group(0).lower()
        return tag if tags_ok and tag in CHATTERBOX_TAGS else " "
    return _TAG.sub(rep, t)


# --------------------------------------------------------------------------- #
# Normalisation: symbols
# --------------------------------------------------------------------------- #

_KEEP_SYMBOLS = set("$¥€£%°+=~&#")       # expanded or mapped to words later
_RP_MARKERS = set("*\"'\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f\u00ab\u00bb\u201e")
_DROP_FORMAT = {"\u200d", "\ufe0e", "\ufe0f", "\u20e3", "\u00ad", "\u2060"}


# ASCII emoticons at a word's edge (":)", ";-)", ":D", "=P", "<3"): the eyes need
# a space or the start before them, so "http://", "10:30" and "Note:D" are untouched.
_EMOTICON = re.compile(r"(?:(?<=\s)|^)(?:[:;=][-^']?[)(\]\[DPpOo3/\\|]+|<3+)(?=\s|$|[.,!?;:])")


def strip_symbols(t: str) -> str:
    """NFKC, then keep letters, numbers, punctuation, marks and whitespace.

    Every emoji, pictograph, arrow and math symbol is Unicode category S* and
    goes, except ``$ ¥ € £ % °`` (spoken by :func:`expand_numbers`) and
    ``+ = ~ & #`` (mapped to words). Combining marks stay (Devanagari, Arabic),
    except the emoji modifiers (variation selectors, ZWJ, keycap). ASCII
    emoticons (``:)``, ``;-)``, ``<3``) go too."""
    t = _EMOTICON.sub(" ", t)
    t = t.replace("\u00bd", " one half").replace("\u00bc", " a quarter").replace("\u00be", " three quarters")
    t = unicodedata.normalize("NFKC", t)
    out = []
    for ch in t:
        if ch in _DROP_FORMAT or "\U000e0000" <= ch <= "\U000e007f" or "\ufe00" <= ch <= "\ufe0f":
            out.append(" " if ch == "\u20e3" else "")
            continue
        cat = unicodedata.category(ch)
        if ch.isspace() or ch in _RP_MARKERS or ch in _KEEP_SYMBOLS:
            out.append(ch)
        elif cat[0] in "LNPM":
            out.append(" " if ch in "\\_" else ch)
        else:
            out.append(" ")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Normalisation: numbers (English-shaped text only)
# --------------------------------------------------------------------------- #

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORD_IRREG = {"one": "first", "two": "second", "three": "third", "five": "fifth",
              "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def number_words(n: int) -> str:
    """1250 → 'one thousand two hundred fifty' (US style, no 'and')."""
    if n < 0:
        return "minus " + number_words(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    for div, name in ((10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"),
                      (1000, "thousand")):
        if n >= div:
            head, rest = divmod(n, div)
            return number_words(head) + f" {name}" + ("" if rest == 0 else " " + number_words(rest))
    return str(n)


def ordinal_words(n: int) -> str:
    w = number_words(n)
    head, sep, last = w.rpartition("-") if "-" in w.split()[-1] else w.rpartition(" ")
    if last in _ORD_IRREG:
        last = _ORD_IRREG[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + sep + last


def _decimal(s: str) -> str:
    neg = s.startswith("-")
    s = s.lstrip("-")
    whole, _, frac = s.partition(".")
    out = number_words(int(whole.replace(",", ""))) if whole else "zero"
    if frac:
        out += " point " + " ".join(ONES[int(c)] for c in frac)
    return ("minus " if neg else "") + out


def _year(n: int) -> str:
    if 2000 <= n <= 2009:
        return number_words(n)
    a, b = divmod(n, 100)
    if b == 0:
        return number_words(a) + " hundred"
    if b < 10:
        return number_words(a) + " oh " + number_words(b)
    return number_words(a) + " " + number_words(b)


_CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"), "€": ("euro", "euros", "cent", "cents"),
             "£": ("pound", "pounds", "penny", "pence"), "¥": ("yen", "yen", None, None)}
_MULT = {"k": 1000, "K": 1000, "m": 10 ** 6, "M": 10 ** 6, "bn": 10 ** 9, "B": 10 ** 9}
_MULT_WORDS = ("thousand", "million", "billion", "trillion")

# Money: the multiplier is attached or a word; the space AFTER the amount is never
# consumed (trap 3: "$20 and" must stay "twenty dollars and").
_MONEY = re.compile(
    r"(-)?([$¥€£])\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
    r"(?:(k|K|M|bn|B|m)\b|\s(thousand|million|billion|trillion)\b)?")
_PERCENT = re.compile(r"(-?\d+(?:\.\d+)?)\s?%")
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?::\d{2})?(?:\s?([AaPp])\.?\s?[Mm]\b\.?)?")
_HOUR_AMPM = re.compile(r"\b(\d{1,2})\s?([AaPp])\.?\s?[Mm]\b\.?")
_ISO_DATE = re.compile(r"\b(1\d{3}|20\d{2})-(\d{2})-(\d{2})\b")
# Numeric dates with a year: 12/24/2025, 12/24/25 (en-US month/day; a first field
# over 12 is read day/month: 24/12/2025), and 2025/12/24. Without a year "3/4" is a
# fraction, never a date (FR-works 8: "12/24/2025" became "12/24/twenty twenty-five").
_SLASH_DATE = re.compile(r"(?<![\w/.])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\w/]|\.\d)")
_MY_SLASH = re.compile(r"(?<![\w/.])(\d{1,2})/(1\d{3}|20\d{2})(?![\w/]|\.\d)")   # expires 12/2025
_YMD_SLASH = re.compile(r"(?<![\w/.])(1\d{3}|20\d{2})/(\d{1,2})/(\d{1,2})(?![\w/]|\.\d)")
# The month forms are enumerated and anchored: "Marcus 5", "Julia 3", "5 Marines"
# and "Nova 5" are names, never dates.
_MONTH_RX = (r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
             r"Sep(?:tember|t)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b\.?")
_MD_DATE = re.compile(r"\b" + _MONTH_RX + r"\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(1\d{3}|20\d{2})\b)?")
_DM_DATE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + _MONTH_RX + r"(?:,?\s+(1\d{3}|20\d{2})\b)?")
_PHONE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[-\s])?(?:\(\d{2,4}\)\s?|\d{2,4}[-\s])\d{3,4}[-\s]\d{3,4}(?![\w])")
# A local number ("555-0123") has two groups, so _PHONE misses it and the range rule
# would read "555 to one hundred twenty-three". It is read digit by digit unless it
# looks like a range: ascending, no leading zero, a round right side, no phone word
# just before it ("500-1000 people" stays a range).
_PHONE_LOCAL = re.compile(r"(?<![\w.,-])(\d{3})-(\d{4})(?![\w-]|[.,]\d)")
_PHONE_WORDS = re.compile(r"\b(?:call|phone|dial|text|number|tel|fax|ring|reach|contact|line|ext)\w*"
                          r"(?:\W+\w+){0,2}\W*\Z",
                          re.IGNORECASE)
_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
_UNITS = {"km": ("kilometre", "kilometres"), "cm": ("centimetre", "centimetres"),
          "mm": ("millimetre", "millimetres"), "kg": ("kilo", "kilos"), "lb": ("pound", "pounds"),
          "lbs": ("pound", "pounds"), "mi": ("mile", "miles"), "ft": ("foot", "feet"),
          "GB": ("gigabyte", "gigabytes"), "MB": ("megabyte", "megabytes"),
          "TB": ("terabyte", "terabytes"), "KB": ("kilobyte", "kilobytes"),
          "ms": ("millisecond", "milliseconds"), "min": ("minute", "minutes"),
          "mins": ("minute", "minutes"), "hr": ("hour", "hours"), "hrs": ("hour", "hours"),
          "mph": ("mile an hour", "miles an hour"), "kph": ("kilometre an hour", "kilometres an hour"),
          "km/h": ("kilometre an hour", "kilometres an hour"), "kWh": ("kilowatt hour", "kilowatt hours"),
          "kW": ("kilowatt", "kilowatts"), "GHz": ("gigahertz", "gigahertz"),
          "MHz": ("megahertz", "megahertz"), "ml": ("millilitre", "millilitres"),
          "mL": ("millilitre", "millilitres"), "L": ("litre", "litres"),
          "s": ("second", "seconds"), "m": ("metre", "metres"), "g": ("gram", "grams"),
          "h": ("hour", "hours"), "°C": ("degree", "degrees"), "°F": ("degree Fahrenheit", "degrees Fahrenheit"),
          "°": ("degree", "degrees")}
# Single-letter units need a space ("5 m", "1.8 s"); attached they are something
# else ("1990s", "5G", "2m" stays) — trap 2.
_UNIT = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?)(?:\s?(km/h|kWh|kph|mph|GHz|MHz|km|cm|mm|kg|lbs|lb|mi|ft|GB|MB|TB|KB|"
    r"ms|mins|min|hrs|hr|kW|ml|mL)\b|\s(h|m|g|s|L)\b(?![-'])|\s?(°C|°F|°)(?!\w))")
_RANGE = re.compile(r"(?<![\w.,-])(\d+)\s?[-–]\s?(\d+)(?![\d\w]|[.,]\d|-\d)")
_THOUSANDS = re.compile(r"(?<![\w.,])-?\d{1,3}(?:,\d{3})+(?![\d,]|\.\d)")
_DECIMAL = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d+(?![\w]|\.\d)")
_DECADE = re.compile(r"(?<![\w'])(1[1-9]|20)?(\d)0['’]?s\b|(?<=\s)['’](\d)0s\b")
# Never a year inside a slash date (an unparsable one is left alone, not half read).
_YEAR = re.compile(r"(?<![\w.,/])(1[1-9]\d{2}|20\d{2})(?![\w.,]\d|\w)")
_MODEL = re.compile(r"\b([A-Z]{2,5})\s(\d{2})(\d{2})\b")
_BIGINT = re.compile(r"(?<![\w.,/])\d{4,}(?![\w]|[.,]\d)")
_TIMES_X = re.compile(r"\b(\d+(?:\.\d+)?)x\b")
_NEGATIVE = re.compile(r"(?<![\w.,)-])-(?=\d)")
_TWENTY_FOUR_SEVEN = re.compile(r"\b24/7\b")


def _money(m: re.Match) -> str:
    neg, sym, amount, frac, mult, mult_word = m.groups()
    one, many, sub_one, sub_many = _CURRENCY[sym]
    whole = int(amount.replace(",", ""))
    pre = "minus " if neg else ""
    if mult or mult_word:
        num = _decimal(f"{whole}.{frac}") if frac else number_words(whole)
        scale = mult_word or {1000: "thousand", 10 ** 6: "million", 10 ** 9: "billion"}[_MULT[mult]]
        return f"{pre}{num} {scale} {many}"
    if frac and len(frac) != 2:
        return f"{pre}{_decimal(f'{whole}.{frac}')} {many}"
    cents = int(frac) if frac else 0
    if whole == 0 and cents and sub_one:
        return f"{pre}{number_words(cents)} {sub_one if cents == 1 else sub_many}"
    s = f"{pre}{number_words(whole)} {one if whole == 1 else many}"
    if cents and sub_one:
        s += f" {number_words(cents)}" if sym == "£" else \
            f" and {number_words(cents)} {sub_one if cents == 1 else sub_many}"
    return s


def _clock(h: int, mm: int, ap: str) -> str | None:
    if h > 23 or mm > 59:
        return None
    if ap:
        if h == 0 or h > 12:
            return None
    else:
        if h > 12:
            h, ap = h - 12, "p"
        elif h == 0:
            h, ap = 12, "a"
    if mm == 0:
        core = number_words(h) if ap else f"{number_words(h)} o'clock"
    elif mm < 10:
        core = f"{number_words(h)} oh {number_words(mm)}"
    else:
        core = f"{number_words(h)} {number_words(mm)}"
    return f"{core} {ap.lower()} m" if ap else core


def _keep_stop(m: re.Match, spoken: str | None) -> str:
    """'9 a.m.' ending a sentence: the abbreviation's dot is also the full stop."""
    if spoken is None:
        return m.group(0)
    rest = m.string[m.end():]
    if m.group(0).endswith(".") and (not rest.strip() or re.match(r"\s+[A-Z\"\u201c]", rest)):
        return spoken + "."
    return spoken


def _time(m: re.Match) -> str:
    return _keep_stop(m, _clock(int(m.group(1)), int(m.group(2)), m.group(3) or ""))


def _hour_ampm(m: re.Match) -> str:
    return _keep_stop(m, _clock(int(m.group(1)), 0, m.group(2)))


def _month(name: str) -> str:
    return next(x for x in MONTHS if x.lower().startswith(name[:3].lower()))


def _iso_date(m: re.Match) -> str:
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return m.group(0)
    return f"{MONTHS[mo - 1]} {ordinal_words(d)}, {_year(y)}"


def _slash_year(y: str) -> int:
    n = int(y)
    return n if len(y) == 4 else 2000 + n


def _slash_date(m: re.Match) -> str:
    a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
    mo, d = (a, b) if a <= 12 else (b, a)          # en-US month/day; 24/12 is day/month
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return m.group(0)
    return f"{MONTHS[mo - 1]} {ordinal_words(d)}, {_year(_slash_year(y))}"


def _my_slash(m: re.Match) -> str:
    mo, y = int(m.group(1)), int(m.group(2))
    if not 1 <= mo <= 12:
        return m.group(0)
    return f"{MONTHS[mo - 1]} {_year(y)}"


def _ymd_slash(m: re.Match) -> str:
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return m.group(0)
    return f"{MONTHS[mo - 1]} {ordinal_words(d)}, {_year(y)}"


def _md_date(m: re.Match) -> str:
    d = int(m.group(2))
    if not 1 <= d <= 31:
        return m.group(0)
    y = m.group(3)
    return f"{_month(m.group(1))} {ordinal_words(d)}" + (f", {_year(int(y))}" if y else "")


def _dm_date(m: re.Match) -> str:
    d = int(m.group(1))
    if not 1 <= d <= 31:
        return m.group(0)
    y = m.group(3)
    return f"the {ordinal_words(d)} of {_month(m.group(2))}" + (f", {_year(int(y))}" if y else "")


def _phone_local(m: re.Match) -> str:
    a, b = m.group(1), m.group(2)
    before = m.string[max(0, m.start() - 24):m.start()]
    if (b[0] != "0" and int(b) > int(a) and int(b) % 10 == 0
            and not _PHONE_WORDS.search(before)):
        return m.group(0)                      # "500-1000": a range, said later
    return _phone(m)


def _phone(m: re.Match) -> str:
    groups = re.findall(r"\+?\d+", m.group(0))
    spoken = []
    for g in groups:
        pre = "plus " if g.startswith("+") else ""
        spoken.append(pre + " ".join(ONES[int(c)] for c in g.lstrip("+")))
    return ", ".join(spoken)


def _unit(m: re.Match) -> str:
    num, u1, u2, u3 = m.groups()
    unit = u1 or u2 or u3
    one, many = _UNITS[unit]
    value = float(num)
    return f"{_decimal(num)} {one if abs(value) == 1 and '.' not in num else many}"


def _decade(m: re.Match) -> str:
    if m.group(3):                                   # '80s
        return TENS[int(m.group(3))][:-1] + "ies" if int(m.group(3)) >= 2 else m.group(0)
    century, tens = m.group(1), int(m.group(2))
    if century is None:                              # 80s, 90s
        return TENS[tens][:-1] + "ies" if tens >= 2 else m.group(0)
    if tens == 0:
        return number_words(int(century)) + " hundreds" if century != "20" else "two thousands"
    if tens == 1:
        return f"{number_words(int(century))} tens"
    return f"{number_words(int(century))} {TENS[tens][:-1]}ies"


def _big(m: re.Match) -> str:
    s = m.group(0)
    if len(s) > 15:
        return " ".join(ONES[int(c)] for c in s)
    return number_words(int(s))


def expand_numbers(t: str) -> str:
    """Money, percent, times, dates, phone numbers, ordinals, units, ranges,
    thousands, decimals, decades, years and integers ≥ 1000 → words. Small plain
    integers stay digits (every engine reads "10" right); acronyms are untouched."""
    t = _MONEY.sub(_money, t)
    t = _PERCENT.sub(lambda m: f"{_decimal(m.group(1))} percent", t)
    t = _TIME.sub(_time, t)
    t = _HOUR_AMPM.sub(_hour_ampm, t)
    t = _ISO_DATE.sub(_iso_date, t)
    t = _YMD_SLASH.sub(_ymd_slash, t)
    t = _SLASH_DATE.sub(_slash_date, t)
    t = _MY_SLASH.sub(_my_slash, t)
    t = _MD_DATE.sub(_md_date, t)
    t = _DM_DATE.sub(_dm_date, t)
    t = _PHONE.sub(_phone, t)
    t = _PHONE_LOCAL.sub(_phone_local, t)
    t = _ORDINAL.sub(lambda m: ordinal_words(int(m.group(1))), t)
    t = _UNIT.sub(_unit, t)
    t = _TIMES_X.sub(lambda m: f"{m.group(1)} times", t)
    t = _TWENTY_FOUR_SEVEN.sub("twenty-four seven", t)
    t = _RANGE.sub(lambda m: f"{m.group(1)} to {m.group(2)}", t)
    t = _THOUSANDS.sub(lambda m: number_words(int(m.group(0).replace(",", ""))), t)
    t = _DECIMAL.sub(lambda m: _decimal(m.group(0)), t)
    t = _DECADE.sub(_decade, t)
    t = _MODEL.sub(lambda m: f"{m.group(1)} {number_words(int(m.group(2)))} "
                   f"{number_words(int(m.group(3))) if int(m.group(3)) >= 10 else 'oh ' + number_words(int(m.group(3)))}"
                   if int(m.group(3)) else f"{m.group(1)} {number_words(int(m.group(2)))} hundred", t)
    t = _YEAR.sub(lambda m: _year(int(m.group(0))), t)
    t = _BIGINT.sub(_big, t)
    t = _NEGATIVE.sub("minus ", t)
    return t


_SYMBOL_WORDS = [
    (re.compile(r"\s*&\s*"), " and "),
    (re.compile(r"(?<=\w)\+\+"), " plus plus"),
    (re.compile(r"(?<=[\w)])\s*\+\s*(?=[\w(])"), " plus "),
    (re.compile(r"(?<=[\w)])\s*=\s*(?=[\w(])"), " equals "),
    (re.compile(r"~\s*(?=\d)"), "about "),
    (re.compile(r"#\s*(?=\d)"), "number "),
    (re.compile(r"(?<=[A-Za-z])/(?=[A-Za-z])"), " "),
    (re.compile(r"[+=~#&$¥€£%°]"), " "),
]


def is_cjk_heavy(t: str) -> bool:
    """True when ≥ 25 % of the non-space characters are kana, kanji or hangul."""
    chars = [c for c in t if not c.isspace()]
    if not chars:
        return False
    cjk = sum(1 for c in chars if "\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff"
              or "\uac00" <= c <= "\ud7af" or "\u3400" <= c <= "\u4dbf")
    return cjk * 4 >= len(chars)


def normalise(text: str, *, tags_ok: bool) -> str:
    """One sentence of markdown → the words the TTS engine should say ('' = nothing).

    strip_markdown → tag filter (the nine Chatterbox tags only when ``tags_ok``)
    → strip_symbols → expand_numbers (English-shaped text only). The role-play
    ``*`` and quotation marks survive for :func:`rp_render`."""
    if not text:
        return ""
    t = strip_markdown(text)
    t = filter_tags(t, tags_ok=tags_ok)
    t = strip_symbols(t)
    if not is_cjk_heavy(t):                  # trap 4: a ja/zh reply keeps its digits
        t = expand_numbers(t)
    cjk = is_cjk_heavy(t)
    for rx, rep in (_SYMBOL_WORDS[:-1] if cjk else _SYMBOL_WORDS):
        t = rx.sub(rep, t)
    if cjk:
        t = re.sub(r"[+=~#&]", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if _speakable(t) else ""


# --------------------------------------------------------------------------- #
# Role-play segmentation
# --------------------------------------------------------------------------- #

RP_MODES = ("all", "skip_actions", "dialogue", "dialogue_narrator")
_MODE_ALIASES = {"quotes": "dialogue", "quotes+narrator": "dialogue_narrator",
                 "quotes_narrator": "dialogue_narrator", "speak_all": "all"}
# What may precede a one-word action: the start of the text or of a sentence
# (terminal punctuation, a closing quote/bracket, another action's "*", an
# opening bracket or quote, a line break).
_SENTENCE_END = set(".!?…([*\"\u201c\u201d\u00bb\u300d\u300f\u300c\u300e)]\n")
_CLAUSE_MARK = set(",;:\u2014\u2013-")
#: One-word gestures that read as an action even mid-sentence after a comma or
#: dash ("Okay, *sighs*, fine."); any other single word there is emphasis
#: ("Well, *that* was fun.").
GESTURES = frozenset({
    "nods", "nodding", "sighs", "sighing", "laughs", "laughing", "chuckles", "chuckling",
    "giggles", "giggling", "grins", "grinning", "smiles", "smiling", "smirks", "smirking",
    "shrugs", "shrugging", "winks", "winking", "blushes", "blushing", "gasps", "gasping",
    "groans", "groaning", "yawns", "yawning", "coughs", "coughing", "sniffs", "sniffing",
    "sniffles", "snorts", "snorting", "pauses", "pausing", "frowns", "frowning", "beams",
    "beaming", "waves", "waving", "hums", "humming", "whistles", "whistling", "pouts",
    "pouting", "sobs", "sobbing", "cries", "crying", "stretches", "stretching", "facepalms",
    "gulps", "gulping", "scoffs", "scoffing", "snickers", "snickering", "hugs", "winces",
    "wincing", "squeals", "giggle", "laugh", "sigh", "nod", "shrug", "smile", "grin", "wink",
    "chuckle", "gasp", "groan", "yawn", "cough", "sniff", "snort", "pause", "frown", "pout",
})


def _sentence_start(text: str, i: int) -> bool:
    before = text[:i].rstrip(" \t")
    return not before or before[-1] in _SENTENCE_END


def _clause_start(text: str, i: int) -> bool:
    before = text[:i].rstrip()
    return not before or before[-1] in _SENTENCE_END or before[-1] in _CLAUSE_MARK


def _is_action(text: str, m: _Mark) -> bool:
    """The emphasis rule (contracts B.3, A7): an asterisk span is an action only
    when it is ≥ 2 words or starts a sentence; a span with word characters on
    both sides, or a one-word span elsewhere, is emphasis. After a comma or dash
    only a known gesture (:data:`GESTURES`) is an action: "Well, *that* was
    fun." keeps "that" (experience H2)."""
    end = m.close if m.close is not None else len(text)
    inner = text[m.open + 1:end]
    before = text[m.open - 1] if m.open > 0 else ""
    after = text[end + 1] if m.close is not None and end + 1 < len(text) else ""
    if _is_word(before) and _is_word(after):
        return False
    words = inner.split()
    if len(words) >= 2:
        return True
    if not words:
        return False
    word = words[0].strip(".,!?…;:").lower()
    if not _sentence_start(text, m.open):
        return word in GESTURES and _clause_start(text, m.open)
    # "*Not* again." — a capitalised word that runs on into a lowercase
    # continuation of the same sentence is emphasis, not "*sighs* Fine."
    rest = text[end + 1:].lstrip() if m.close is not None else ""
    return not (words[0][:1].isupper() and rest[:1].islower())


def segment(text: str) -> list[tuple[str, str]]:
    """``[(kind, text)]`` with kind ∈ {prose, quote, action}, delimiters removed.

    Outermost quotations only; an unclosed opener runs to the end of the text.
    Emphasis spans (see :func:`_is_action`) stay inside the surrounding piece
    with their asterisks removed. Whitespace inside each piece is collapsed."""
    if not text:
        return []
    out: list[tuple[str, str]] = []
    pos = 0
    pending = ""

    def emit(kind: str, s: str) -> None:
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            return
        if not _speakable(s.replace("*", "")):
            # "*tilting her head*." — bare punctuation ends the piece before it
            end = re.sub(r"[^.!?…]", "", s)[:1]
            if end and out and out[-1][1][-1:] not in ".!?…":
                out[-1] = (out[-1][0], out[-1][1] + end)
            return
        if out and out[-1][0] == kind == "prose":
            out[-1] = (kind, out[-1][1] + " " + s)
        else:
            out.append((kind, s))

    for m in _scan_marks(text):
        end = m.close if m.close is not None else len(text)
        if m.kind == "action" and not _is_action(text, m):
            pending += text[pos:m.open] + text[m.open + 1:end]
            pos = end + 1 if m.close is not None else len(text)
            continue
        pending += text[pos:m.open]
        emit("prose", pending)
        pending = ""
        emit(m.kind, text[m.open + 1:end])
        pos = end + 1 if m.close is not None else len(text)
    pending += text[pos:]
    emit("prose", pending)
    # A lone "*" (a literal, an unopenable star) or a bare dash is not speech.
    return [(k, re.sub(r"\s+", " ", s.replace("*", "")).strip()) for k, s in out
            if _speakable(s.replace("*", ""))]


def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:!?…])", r"\1", s)
    s = re.sub(r"([,;:\u2014\u2013-])\s*([.!?])", r"\2", s)
    s = re.sub(r"([,;:])(\s*[,;:\u2014\u2013])+", r"\1", s)
    s = s.strip(" \u2014\u2013-")
    s = re.sub(r"^[,;:.!?…]+\s*", "", s)
    return s.strip()


def _speakable(s: str) -> bool:
    return any(ch.isalnum() for ch in s)


def _canonical_mode(mode: str | None) -> str:
    m = (mode or "all").strip().lower()
    m = _MODE_ALIASES.get(m, m)
    return m if m in RP_MODES else "all"


# A word that only joins prose to the action that followed it.
_DANGLING = re.compile(r"(?:\s+(?:and|but|or|so|then|while|as|with))+\s*$", re.IGNORECASE)


def rp_render(text: str, mode: str = "all", *, has_dialogue: bool | None = None
              ) -> list[tuple[str, str]]:
    """``[(role, text)]`` with role ∈ {character, narrator} for one normalised sentence.

    ``all`` (default): everything in the character voice, asterisks removed.
    ``skip_actions``: ``*actions*`` dropped. ``dialogue``: quotations only — a
    text without quotations is spoken without its actions. ``dialogue_narrator``:
    quotations in the character voice, prose and actions in the narrator voice;
    without quotations, prose stays with the character and actions go to the
    narrator. Non-RP text (no actions, no quotations) is the identity under every
    mode. ``has_dialogue`` overrides "does this reply contain quotations" (the
    streaming caller knows it from earlier sentences; see :class:`RpRenderer`).
    The prototype names ``quotes``/``quotes+narrator`` map onto ``dialogue``/
    ``dialogue_narrator``; an unknown mode is ``all``."""
    mode = _canonical_mode(mode)
    segs = segment(text)
    if not segs:
        return []
    kinds = {k for k, _ in segs}
    if kinds == {"prose"} and not (has_dialogue and mode in ("dialogue", "dialogue_narrator")):
        s = _clean(segs[0][1]) if len(segs) == 1 else _clean(" ".join(x for _, x in segs))
        return [("character", s)] if _speakable(s) else []
    quotes = has_dialogue if has_dialogue is not None else ("quote" in kinds)
    pieces: list[tuple[str, str, int, str]] = []
    for idx, (kind, s) in enumerate(segs):
        role: str | None
        if mode == "all":
            role = "character"
        elif mode == "skip_actions":
            role = None if kind == "action" else "character"
        elif mode == "dialogue":
            if quotes:
                role = "character" if kind == "quote" else None
            else:
                role = None if kind == "action" else "character"
        else:  # dialogue_narrator
            if quotes:
                role = "character" if kind == "quote" else "narrator"
            else:
                role = "narrator" if kind == "action" else "character"
        if role is not None:
            pieces.append((role, s, idx, kind))
        elif pieces and pieces[-1][2] == idx - 1 and (s[-1:] in ".!?…" or idx == len(segs) - 1):
            # "she asks, *tilting her head*." without the action: keep the full stop;
            # "she says hi and *waves*." without it: drop the "and" it leaves behind
            r0, s0, i0, k0 = pieces[-1]
            s0 = re.sub(r"[\s,;:\u2014\u2013-]+$", "", s0)
            if s0 and s0[-1] not in ".!?…":
                end = s[-1] if s[-1:] in ".!?…" else ""
                trimmed = re.sub(r"[\s,;:\u2014\u2013-]+$", "", _DANGLING.sub("", " " + s0)).strip()
                if _speakable(trimmed):
                    pieces[-1] = (r0, trimmed + ("" if trimmed[-1] in ".!?…" else end), i0, k0)
                elif _DANGLING.fullmatch(" " + s0) and len(pieces) > 1:
                    # '… "let's go" and *smiles*.': the bare "and" goes, the stop stays
                    pieces.pop()
                    r1, s1, i1, k1 = pieces[-1]
                    if end and s1.rstrip()[-1:] not in ".!?…":
                        pieces[-1] = (r1, s1.rstrip() + end, i1, k1)
                else:
                    pieces[-1] = (r0, s0 + end, i0, k0)
    merged: list[list] = []          # [role, text, last idx, last kind]
    for role, s, idx, kind in pieces:
        if merged and merged[-1][0] == role:
            prev = _clean(merged[-1][1])
            starts_upper = s[:1].isupper() or s[:1] in "\"\u201c"
            if starts_upper and prev and (merged[-1][2] != idx - 1 or merged[-1][3] == "action"):
                # Something was dropped in between, or an action ran into a new
                # sentence: give the joint a full stop so it is not read on.
                prev = re.sub(r"[,;:\u2014\u2013-]+$", "", prev).rstrip()
                if prev and prev[-1] not in ".!?…":
                    prev += "."
            merged[-1] = [role, prev + " " + s, idx, kind]
        else:
            merged.append([role, s, idx, kind])
    out = []
    for role, s, _, _ in merged:
        s = re.sub(r"[,;:\u2014\u2013-]+$", "", _clean(s)).rstrip()
        if s and s[-1] not in ".!?…" and len(merged) > 1:
            s += "."             # each voice piece is said as a whole sentence
        if _speakable(s):
            out.append((role, s))
    return out


class RpRenderer:
    """Per-reply role-play state: once a sentence of this reply contained a
    quotation, later sentences without one are narration, not dialogue."""

    def __init__(self, mode: str = "all") -> None:
        self.mode = _canonical_mode(mode)
        self.seen_dialogue = False

    def render(self, text: str) -> list[tuple[str, str]]:
        if any(k == "quote" for k, _ in segment(text)):
            self.seen_dialogue = True
        return rp_render(text, self.mode, has_dialogue=True if self.seen_dialogue else None)


def render_all(sentences: Iterable[str], mode: str, *, tags_ok: bool = False) -> list[tuple[str, str]]:
    """Convenience: normalise + role-play a whole sequence of sentences."""
    r = RpRenderer(mode)
    out: list[tuple[str, str]] = []
    for s in sentences:
        out.extend(r.render(normalise(s, tags_ok=tags_ok)))
    return out
