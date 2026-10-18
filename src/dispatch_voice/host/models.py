"""One models folder with subfolders, and the one-time move from the old two.

Before 1.6.0 the models lived in TWO unrelated places with two environment variables:
the speech models (Silero, Smart Turn, Parakeet) flat in ``DISPATCH_SPEECH_MODELS``
(``<data>/speech-models``) and Chatterbox in ``DISPATCH_VOICE_MODELS``
(``~/.local/share/dispatch-voice/models``). Now there is one ``models_dir`` (the folder
DisPatch hands the add-on as ``paths.models_dir``) with a subfolder per family::

    <models_dir>/speech/        silero_vad.onnx, smart-turn-v3.2-cpu.onnx, parakeet-tdt-0.6b-v2/
    <models_dir>/chatterbox/    the Chatterbox-Turbo ONNX graphs, tokenizer.json, config.json

Nothing is moved behind anybody's back. The resolvers below return the OLD location while
models still sit there (so an existing install keeps working untouched), the new one for a
fresh install, and :func:`migrate` moves the old files (``dry_run`` by default, idempotent,
refuses to overwrite). An explicit ``DISPATCH_SPEECH_MODELS`` / ``DISPATCH_VOICE_MODELS`` is
an operator's choice of place: it is honoured and never moved.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger("dispatch.voice")

SPEECH_SUB = "speech"
CHATTERBOX_SUB = "chatterbox"
SPEECH_ENV = "DISPATCH_SPEECH_MODELS"
CHATTERBOX_ENV = "DISPATCH_VOICE_MODELS"
#: Top-level entries ``dispatch-voice fetch-models`` puts in the speech folder.
SPEECH_ENTRIES = ("silero_vad.onnx", "smart-turn-v3.2-cpu.onnx", "parakeet-tdt-0.6b-v2")
MARKER = ".layout.json"
LAYOUT = 2


def root() -> Path:
    """The unified folder: DisPatch's ``paths.models_dir`` (else the environment's)."""
    from . import context
    return context.paths().models_dir


def _explicit(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else None


def legacy_chatterbox_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "dispatch-voice" / "models"


def _has_speech(d: Path) -> bool:
    return any((d / e).exists() for e in SPEECH_ENTRIES)


def _has_chatterbox(d: Path) -> bool:
    try:
        return d.is_dir() and any((d / n).exists() for n in ("tokenizer.json", "onnx", "config.json"))
    except OSError:
        return False


def speech_dir() -> Path:
    """Where the speech models are: the explicit env folder, else the old flat location
    while models still sit there, else ``<models_dir>/speech``."""
    exp = _explicit(SPEECH_ENV)
    if exp is not None:
        return exp
    r = root()
    new = r / SPEECH_SUB
    if _has_speech(new):
        return new
    if _has_speech(r):
        return r                         # not migrated yet: keep using it
    return new


def chatterbox_dir() -> Path:
    exp = _explicit(CHATTERBOX_ENV)
    if exp is not None:
        return exp
    new = root() / CHATTERBOX_SUB
    if _has_chatterbox(new):
        return new
    old = legacy_chatterbox_dir()
    if _has_chatterbox(old):
        return old
    return new


def plan() -> list[dict]:
    """The moves :func:`migrate` would make: ``[{family, src, dst, state}]`` where state is
    ``move`` (do it), ``skip_exists`` (the destination already has it), ``explicit`` (an env
    folder: left alone)."""
    out: list[dict] = []
    r = root()
    sp_new = r / SPEECH_SUB
    if _explicit(SPEECH_ENV) is not None:
        out.append({"family": "speech", "src": str(r), "dst": str(sp_new), "state": "explicit"})
    else:
        for e in SPEECH_ENTRIES:
            src = r / e
            if src.exists():
                dst = sp_new / e
                out.append({"family": "speech", "src": str(src), "dst": str(dst),
                            "state": "skip_exists" if dst.exists() else "move"})
    cb_new = r / CHATTERBOX_SUB
    old = legacy_chatterbox_dir()
    if _explicit(CHATTERBOX_ENV) is not None:
        out.append({"family": "chatterbox", "src": str(old), "dst": str(cb_new), "state": "explicit"})
    elif _has_chatterbox(old) and old.resolve() != cb_new.resolve():
        out.append({"family": "chatterbox", "src": str(old), "dst": str(cb_new),
                    "state": "skip_exists" if _has_chatterbox(cb_new) else "move"})
    return out


def _move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def migrate(*, dry_run: bool = True) -> dict:
    """Move the old model locations under :func:`root`. Idempotent: a second run finds
    nothing to do. ``dry_run`` (the default) only reports. Returns
    ``{"dry_run", "moves": [...], "moved": n, "errors": [...]}``."""
    moves = plan()
    report = {"dry_run": dry_run, "moves": moves, "moved": 0, "errors": []}
    if dry_run:
        return report
    for m in moves:
        if m["state"] != "move":
            continue
        src, dst = Path(m["src"]), Path(m["dst"])
        try:
            if m["family"] == "chatterbox":
                dst.mkdir(parents=True, exist_ok=True)
                for child in list(src.iterdir()):
                    target = dst / child.name
                    if target.exists():
                        continue
                    shutil.move(str(child), str(target))
                with contextlib.suppress(OSError):
                    src.rmdir()             # only when now empty
            else:
                _move(src, dst)
            m["state"] = "moved"
            report["moved"] += 1
        except OSError as e:
            m["state"] = "error"
            report["errors"].append(f"{src.name}: {e}")
            log.warning("voice: could not move %s to %s (%s)", src, dst, e)
    if report["moved"]:
        with contextlib.suppress(OSError):
            root().mkdir(parents=True, exist_ok=True)
            (root() / MARKER).write_text(json.dumps({"layout": LAYOUT, "moved": report["moved"]}), encoding="utf-8")
    return report


def legacy_present() -> bool:
    """Are models still at an old location (and movable)? For the status page."""
    return any(m["state"] == "move" for m in plan())
