"""System programs the add-on needs that pip cannot install.

``pip install dispatch-voice`` brings every Python dependency, but two pieces are programs or
libraries of the operating system:

* **ffmpeg** decodes the clip you upload when you clone a voice (any browser-recorded format);
* **libopus** is the Opus codec of the low-bandwidth voice wire (without it the wire stays
  PCM16: it works, it just uses more data).

They are declared in ``pyproject.toml`` (``[tool.dispatch-voice.system-requirements]``, a drift
test keeps the two lists equal), shown in the README, and checked here for ``dispatch-voice
doctor`` and ``GET /api/voice/status`` with an install command for the operating system the
server runs on.
"""
from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Requirement:
    id: str
    what: str                         # what it is for, one sentence
    required_for: str                 # the feature that does not work without it
    optional: bool                    # True: voice chat itself works without it
    check: Callable[[], str | None] = field(compare=False, repr=False)   # a description of what was found, or None
    hints: dict = field(default_factory=dict)  # package manager -> command


def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _libopus() -> str | None:
    from . import opus
    if opus._switched_off():
        return None
    try:
        return "libopus" if opus._load() is not None else None
    except Exception:       # noqa: BLE001 - "cannot tell" is "not found"
        return None


REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement("ffmpeg", "decodes the voice clip you upload", "cloning a voice from an uploaded recording", False,
                _ffmpeg,
                {"apt": "sudo apt install ffmpeg", "dnf": "sudo dnf install ffmpeg-free", "pacman": "sudo pacman -S ffmpeg",
                 "zypper": "sudo zypper install ffmpeg", "apk": "sudo apk add ffmpeg", "brew": "brew install ffmpeg",
                 "winget": "winget install Gyan.FFmpeg", "choco": "choco install ffmpeg"}),
    Requirement("libopus", "the Opus audio codec of the low-bandwidth voice link", "compact voice over a slow link", True,
                _libopus,
                {"apt": "sudo apt install libopus0", "dnf": "sudo dnf install opus", "pacman": "sudo pacman -S opus",
                 "zypper": "sudo zypper install libopus0", "apk": "sudo apk add opus", "brew": "brew install opus",
                 "winget": "No winget package: download opus.dll (an Opus 1.4+ build, for example from vcpkg: "
                           "`vcpkg install opus`) and put it next to python.exe or on PATH",
                 "choco": "choco install opus-tools, then copy opus.dll next to python.exe"}),
)


def package_manager(platform: str | None = None, os_release: str | None = None) -> str:
    """The package manager key for this machine: ``winget`` (Windows), ``brew`` (macOS), or from
    ``/etc/os-release`` ``ID`` / ``ID_LIKE`` on Linux (``apt`` when nothing matches)."""
    plat = platform or sys.platform
    if plat == "win32":
        return "winget"
    if plat == "darwin":
        return "brew"
    text = os_release
    if text is None:
        try:
            text = Path("/etc/os-release").read_text(encoding="utf-8")
        except OSError:
            text = ""
    ids: list[str] = []
    for line in text.splitlines():
        k, _, v = line.partition("=")
        if k in ("ID", "ID_LIKE"):
            ids += v.strip().strip('"').lower().split()
    table = (("debian", "apt"), ("ubuntu", "apt"), ("fedora", "dnf"), ("rhel", "dnf"), ("centos", "dnf"),
             ("arch", "pacman"), ("suse", "zypper"), ("opensuse", "zypper"), ("alpine", "apk"))
    for needle, mgr in table:
        if needle in ids:
            return mgr
    return "apt"


def hint(req: Requirement, manager: str | None = None) -> str:
    mgr = manager or package_manager()
    return req.hints.get(mgr) or next(iter(req.hints.values()), "")


def check(manager: str | None = None) -> list[dict]:
    """``[{id, ok, what, required_for, optional, fix, package_manager, all_fixes}]`` for every
    requirement (no paths: the status page may show it to any operator)."""
    mgr = manager or package_manager()
    out = []
    for r in REQUIREMENTS:
        try:
            found = r.check()
        except Exception:       # noqa: BLE001
            found = None
        out.append({"id": r.id, "ok": bool(found),
                    "what": r.what, "required_for": r.required_for, "optional": r.optional,
                    "fix": "" if found else hint(r, mgr), "package_manager": mgr,
                    "all_fixes": dict(r.hints)})
    return out
