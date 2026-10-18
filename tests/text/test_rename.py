"""Rename: "Drive mode" is "Voice chat" everywhere a person reads it (SPEC §6.4).

Internal identifiers keep their names (``drive.js``, ``drive.*`` keys,
``.drive-*``, ``dispatch-drive``, ``/ws/voice``), so they are skipped: a match
glued to ``. - _ /`` or inside backticks is an identifier, not the noun. The
pre-split document under ``docs/voice-chat/legacy/`` is deleted before 1.0.0
(T3) and is not scanned; a CHANGELOG line announcing the rename is allowed."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from dispatch_voice.text import ack_lines, persona

ROOT = Path(__file__).resolve().parents[2]

_DRIVE = re.compile(r"\bdrive\b", re.IGNORECASE)
_DRIVE_MODE = re.compile(r"\bdrive[- ]mode\b", re.IGNORECASE)
_CODE = re.compile(r"`[^`\n]*`")
_ALLOWED_LINE = re.compile(r"now called voice chat", re.IGNORECASE)


def scanned_files() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "CHANGELOG.md",
             ROOT / "src/dispatch_voice/standalone/voices.html",
             ROOT / "packages/forge-voice-core/src/forge_voice_core/speak.py"]
    files += sorted((ROOT / "src/dispatch_voice/text").glob("*.py"))
    files += sorted((ROOT / "packages/forge-voice-core/src/forge_voice_core/text").glob("*.py"))
    files += sorted(p for p in (ROOT / "docs").rglob("*.md") if "legacy" not in p.parts)
    return [p for p in files if p.is_file()]


def noun_hits(text: str) -> list[str]:
    hits = []
    for line in text.splitlines():
        if _ALLOWED_LINE.search(line):
            continue
        prose = _CODE.sub(" ", line)
        for m in _DRIVE.finditer(prose):
            before = prose[m.start() - 1] if m.start() else ""
            after = prose[m.end():m.end() + 2]
            if before in ".-_/'\"#[" or re.match(r"[.\-_/][\w*]", after):
                continue                    # drive.js, drive.*, .drive-x, dispatch-drive
            hits.append(line.strip())
        if _DRIVE_MODE.search(prose):
            hits.append(line.strip())
    return hits


@pytest.mark.parametrize("path", scanned_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_drive_as_a_noun(path):
    assert noun_hits(path.read_text(encoding="utf-8")) == []


def test_the_scanner_catches_the_old_words_and_spares_identifiers():
    assert noun_hits("Use it for Drive mode.")
    assert noun_hits("Applies from the next drive.")
    assert noun_hits("End drive")
    assert not noun_hits("See `drive.js` and drive.js, the drive.* keys, .drive-open, dispatch-drive.")
    assert not noun_hits("The driver's seat; driving is one example.")
    assert not noun_hits("Drive mode is now called Voice chat.")


def test_persona_text_and_pools_never_mention_driving():
    for text in (persona.STANDING_RULE,
                 persona.build(None, backend=None, profile_tags=True, style="normal",
                               follow_up=True, interrupted=None, extension_persona=None).system,
                 persona.build(None, backend="gateway", profile_tags=True, style="normal",
                               follow_up=True, interrupted=None, extension_persona=None).trailer):
        assert not re.search(r"driv|hands-free|podcast", text, re.IGNORECASE)
    lines = [x for pool in [ack_lines.DEFAULT, *ack_lines.FLAVOUR.values()]
             for v in pool.values() for x in v]
    assert not any(re.search(r"driv|\broad\b|\bcar\b", x, re.IGNORECASE) for x in lines)


def test_standalone_voices_page_uses_the_new_copy():
    html = (ROOT / "src/dispatch_voice/standalone/voices.html").read_text(encoding="utf-8")
    assert ">Set as default<" in html and ">Default<" in html
    assert "Drive" not in html
