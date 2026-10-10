"""Heard-text tracking: where in a reply the listener was when it was cut (SPEC §4.8, C-§B.3).

Chatterbox and Kokoro give no word timings, so the map is linear in time over
the sentence's speakable text (≤ 15 % error at worst, which "cut off after…"
tolerates; sentence-level accuracy is exact because every sentence is its own
``seq``). The pieces:

* :class:`Span` — a speakable sentence and its ``[start, end)`` in the SOURCE
  content (the stored markdown). T5's ``SentenceSplitter.split()`` returns one
  per sentence. ``source`` (optional, Amendment T4-1) is the source slice
  itself, so :func:`source_offset` can find a word boundary without being
  handed the whole content.
* :func:`heard_prefix` — ``(sentence_index, fraction)`` for a played position
  over a sequence of spans.
* :func:`source_offset` — that fraction turned into an offset into the stored
  markdown, cut back to a word boundary and out of any open markup
  (hardening M6: "**Tuesday at 3pm** — see [the map](…)" is spoken as 20
  different characters; a speakable-text offset lands mid-markup or inside a
  URL). The session stores the result as ``voice_heard.upto``. Openers are
  markdown-aware (review T4 #4): ``*``/``_``/``**``/``__``/``~~`` count only
  where CommonMark would open a run (preceded by start/whitespace/punctuation
  and followed by a non-space — ``user_data.csv`` and ``5 * 3`` are literal),
  a backtick only when a closing backtick follows in the span, and ``<`` only
  when it starts a tag (``<b``, ``</b``, ``<!--``), so ``a < b`` is literal.
* :class:`HeardMap` — the session's per-``seq`` bookkeeping (message id,
  span, audio length) and :meth:`HeardMap.locate`, which turns the client's
  ``heard {seq, played_ms}`` into the ``note_heard`` arguments (``upto``,
  ``heard`` ≤ 160 chars, ``unheard`` ≤ 240 chars).
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

HEARD_MAX = 160
UNHEARD_MAX = 240


@dataclass(frozen=True)
class Span:
    text: str          # speakable sentence (what the TTS was given)
    start: int         # [start, end) in the SOURCE content (stored markdown)
    end: int
    source: str = ""   # optional: content[start:end] (lets source_offset see the markup)

    def __len__(self) -> int:
        return max(0, self.end - self.start)


def heard_prefix(spans: Sequence[Span], total_ms: int, played_ms: int) -> tuple[int, float]:
    """Linear time map over the spans' speakable text → (sentence_index, fraction)."""
    if not spans:
        return 0, 0.0
    total_ms = max(0, int(total_ms))
    played_ms = max(0, int(played_ms))
    if total_ms <= 0:
        return 0, 0.0
    if played_ms >= total_ms:
        return len(spans) - 1, 1.0
    lengths = [max(1, len(s.text)) for s in spans]
    chars = sum(lengths)
    pos = (played_ms / total_ms) * chars
    acc = 0.0
    for i, n in enumerate(lengths):
        if pos < acc + n or i == len(lengths) - 1:
            frac = (pos - acc) / n
            return i, min(1.0, max(0.0, frac))
        acc += n
    return len(spans) - 1, 1.0


_EMPH = re.compile(r"\*\*|__|~~|\*|_")
_TAG_START = re.compile(r"<(?=[A-Za-z/!])")


def _word_boundary(text: str, off: int) -> int:
    """Move ``off`` left to the start of the word it is in (whitespace boundary); the
    end of the text is a boundary too."""
    off = max(0, min(len(text), off))
    if off >= len(text):
        return len(text)
    while off > 0 and not text[off - 1].isspace():
        off -= 1
    return off


def _inside_link(prefix: str) -> int | None:
    """Index of the "[" of a link construct that ``prefix`` ends inside of, else None."""
    lb = prefix.rfind("[")
    if lb < 0:
        return None
    after = prefix[lb:]
    if "](" in after:
        url_part = after.split("](", 1)[1]
        return lb if ")" not in url_part else None     # inside the (url) part
    return lb if "]" not in after else None             # inside the [text] part


def _can_open(text: str, i: int, j: int) -> bool:
    """A delimiter run ``text[i:j]`` opens emphasis when it is left-flanking (CommonMark
    §6.2, simplified): preceded by start/whitespace/punctuation and followed by a
    non-space. Intraword ``_`` (``snake_case``) and ``5 * 3`` never open."""
    before = text[i - 1] if i > 0 else " "
    after = text[j] if j < len(text) else " "
    if after.isspace():
        return False
    if text[i] == "_" and before.isalnum():
        return False                                     # intraword underscore is literal
    return before.isspace() or not before.isalnum()


def _can_close(text: str, i: int, j: int) -> bool:
    """Right-flanking: preceded by a non-space, followed by end/whitespace/punctuation."""
    before = text[i - 1] if i > 0 else " "
    after = text[j] if j < len(text) else " "
    if before.isspace():
        return False
    if text[i] == "_" and after.isalnum():
        return False
    return after.isspace() or not after.isalnum()


def _open_emphasis(text: str, off: int) -> int | None:
    """Index of the earliest emphasis/strike marker still open at ``off``, else None."""
    stack: list[tuple[str, int]] = []
    for m in _EMPH.finditer(text, 0, off):
        mark, i, j = m.group(0), m.start(), m.end()
        if stack and stack[-1][0] == mark and _can_close(text, i, j):
            stack.pop()
        elif _can_open(text, i, j):
            stack.append((mark, i))
    return stack[0][1] if stack else None


def _open_code(text: str, off: int) -> int | None:
    """Index of a backtick that opens a code span ``off`` lies inside of: an odd number
    of backticks before ``off`` AND a closing backtick somewhere after it."""
    prefix = text[:off]
    if prefix.count("`") % 2 == 0 or "`" not in text[off:]:
        return None
    return prefix.rfind("`")


def _open_tag(text: str, off: int) -> int | None:
    """Index of the ``<`` of a tag ``off`` lies inside of (``<b``, ``</b``, ``<!--``);
    a bare ``<`` (``a < b``) is literal."""
    prefix = text[:off]
    lt = prefix.rfind("<")
    if lt < 0 or ">" in prefix[lt:] or not _TAG_START.match(text, lt):
        return None
    return lt


def _outside_markup(text: str, off: int) -> int:
    """Step back out of an open link ``[…](…)``, an open emphasis/code marker or an
    open ``<tag>``, so the offset never lands inside markup."""
    for _ in range(8):   # nested/adjacent constructs
        prefix = text[:off]
        candidates = [c for c in (_inside_link(prefix), _open_tag(text, off), _open_code(text, off),
                                  _open_emphasis(text, off)) if c is not None and c < off]
        if not candidates:
            break
        off = _word_boundary(text, min(candidates))
    return off


def source_offset(span: Span, fraction: float, *, content: str | None = None) -> int:
    """``span.start + round(fraction × len(span))`` cut back to a word boundary in the
    source, and out of any open markup (A7). With ``content`` (the stored markdown)
    the boundary is exact; with ``span.source`` likewise; with neither the cut uses
    the speakable text's word boundaries as an approximation."""
    fraction = min(1.0, max(0.0, float(fraction)))
    n = len(span)
    if n <= 0:
        return span.start
    if fraction >= 1.0:
        return span.end
    rel = round(fraction * n)
    if content is not None and len(content) >= span.end:
        src = content[span.start:span.end]
    elif span.source:
        src = span.source
    else:
        # Approximate: word boundary of the speakable text, scaled onto the source range.
        t = span.text or ""
        if not t:
            return span.start + rel
        k = _word_boundary(t, round(fraction * len(t)))
        return span.start + round(k / max(1, len(t)) * n)
    off = _outside_markup(src, _word_boundary(src, rel))
    return span.start + off


def heard_texts(spans: Sequence[Span], index: int, fraction: float) -> tuple[str, str]:
    """(heard ≤ 160 chars, keeping the END; unheard ≤ 240 chars, keeping the START)."""
    if not spans:
        return "", ""
    index = min(len(spans) - 1, max(0, index))
    fraction = min(1.0, max(0.0, float(fraction)))
    cur = spans[index].text or ""
    cut = _word_boundary(cur, round(fraction * len(cur)))
    heard = " ".join([s.text for s in spans[:index]] + [cur[:cut].rstrip()]).strip()
    unheard = " ".join([cur[cut:].lstrip()] + [s.text for s in spans[index + 1:]]).strip()
    if len(heard) > HEARD_MAX:
        heard = "…" + heard[-(HEARD_MAX - 1):].lstrip()
    if len(unheard) > UNHEARD_MAX:
        unheard = unheard[:UNHEARD_MAX - 1].rstrip() + "…"
    return heard, unheard


@dataclass(frozen=True)
class HeardPosition:
    message_id: str | None
    seq: int
    index: int            # sentence index within the message
    fraction: float       # of that sentence
    upto: int             # source offset (voice_heard.upto)
    heard: str
    unheard: str
    complete: bool        # the whole reply was heard (nothing to annotate)


@dataclass
class _Seq:
    message_id: str | None
    span: Span
    audio_ms: int
    content: str | None = None


@dataclass
class HeardMap:
    """Per-seq bookkeeping for one voice session; ``note`` on every spoken sentence."""
    _seqs: dict[int, _Seq] = field(default_factory=dict)

    def note(self, seq: int, *, message_id: str | None, span: Span, audio_ms: int,
             content: str | None = None) -> None:
        self._seqs[int(seq)] = _Seq(message_id, span, max(0, int(audio_ms)), content)

    def add_audio(self, seq: int, ms: int) -> None:
        """Audio length is only known as chunks arrive; add to the seq's total."""
        s = self._seqs.get(int(seq))
        if s is not None:
            s.audio_ms += max(0, int(ms))

    def forget(self, seq: int) -> None:
        self._seqs.pop(int(seq), None)

    def span_of(self, seq: int) -> Span | None:
        """The source span a seq was noted with (None when unknown)."""
        s = self._seqs.get(int(seq))
        return s.span if s is not None else None

    def respan(self, seq: int, span: Span) -> None:
        """The same spoken piece now sits at ``span`` of a rewritten reply text (its audio
        length is kept): a later ``locate`` maps into the new text."""
        s = self._seqs.get(int(seq))
        if s is not None:
            s.span = span

    def clear(self) -> None:
        self._seqs.clear()

    def spans_of(self, message_id: str | None) -> list[tuple[int, Span]]:
        return sorted(((q, s.span) for q, s in self._seqs.items() if s.message_id == message_id),
                      key=lambda t: t[0])

    def last_seq(self, message_id: str | None) -> int | None:
        qs = [q for q, s in self._seqs.items() if s.message_id == message_id]
        return max(qs) if qs else None

    def locate(self, seq: int, played_ms: int, *, source: str = "server",
               rate_chars_per_s: float = 15.0) -> HeardPosition | None:
        """Where the listener was. ``source: phone``: ``played_ms`` is time since the
        phone's ``onstart`` and maps to characters at ``rate × 15 chars/s`` (C13 C2)."""
        s = self._seqs.get(int(seq))
        if s is None:
            return None
        ordered = self.spans_of(s.message_id)
        index = next((i for i, (q, _) in enumerate(ordered) if q == int(seq)), 0)
        spans = [sp for _, sp in ordered]
        if source == "phone":
            total_ms = int(len(s.span.text) / max(1.0, rate_chars_per_s) * 1000)
        else:
            total_ms = s.audio_ms
        if total_ms <= 0:
            frac = 0.0 if played_ms <= 0 else 1.0
        else:
            _, frac = heard_prefix([s.span], total_ms, played_ms)
        upto = source_offset(s.span, frac, content=s.content)
        heard, unheard = heard_texts(spans, index, frac)
        complete = index == len(spans) - 1 and frac >= 1.0
        return HeardPosition(s.message_id, int(seq), index, frac, upto, heard, unheard, complete)
