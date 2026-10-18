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

DisPatch 3.0 names its folder ``<data>/models`` (it was ``<data>/speech-models``, a name that
held Chatterbox too, review F8) and moves the old folder across once. If that move has not
happened (a host that predates it, or a move that failed) the add-on still finds, and can move,
what sits in ``<data>/speech-models`` (:func:`legacy_host_root`).
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
from pathlib import Path

from . import _log, hostlink

log = _log.get()

SPEECH_SUB = "speech"
CHATTERBOX_SUB = "chatterbox"
SPEECH_ENV = "DISPATCH_SPEECH_MODELS"
CHATTERBOX_ENV = "DISPATCH_VOICE_MODELS"
#: Top-level entries ``dispatch-voice fetch-models`` puts in the speech folder.
SPEECH_ENTRIES = ("silero_vad.onnx", "smart-turn-v3.2-cpu.onnx", "parakeet-tdt-0.6b-v2")
#: Every file of the Parakeet folder, in the order the download writes them. The encoder is the
#: fourth of five, so "the encoder exists" is true while the decoder is still coming down.
PARAKEET_FILES = ("config.json", "vocab.txt", "nemo128.onnx", "encoder-model.int8.onnx",
                  "decoder_joint-model.int8.onnx")
MARKER = ".layout.json"
LAYOUT = 2


def root() -> Path:
    """The unified folder: the host's ``models_root()`` (DisPatch: ``paths.models_dir``; standalone: the
    platform data folder or ``FORGE_VOICE_MODELS_DIR``)."""
    return hostlink.get().models_root()


def _explicit(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else None


def legacy_chatterbox_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "dispatch-voice" / "models"


LEGACY_HOST_NAME = "speech-models"


def legacy_host_root() -> Path | None:
    """The pre-3.0 host folder ``<data>/speech-models`` while it still exists next to the
    current models folder (and is not the models folder itself), else None."""
    try:
        r = root()
        if r.name == LEGACY_HOST_NAME:
            return None
        cand = r.parent / LEGACY_HOST_NAME
        if cand.is_dir() and cand.resolve() != r.resolve():
            return cand
    except OSError:
        pass
    return None


def _has_speech(d: Path) -> bool:
    return any((d / e).exists() for e in SPEECH_ENTRIES)


def _has_chatterbox(d: Path) -> bool:
    try:
        return d.is_dir() and any((d / n).exists() for n in ("tokenizer.json", "onnx", "config.json"))
    except OSError:
        return False


def parakeet_complete(d: Path) -> bool:
    """Is the whole Parakeet set in ``d``? (``status.ready`` and the STT provider's health use this,
    so a download in progress is not "ready", review F6.)"""
    try:
        return all((Path(d) / n).exists() for n in PARAKEET_FILES)
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
    old = legacy_host_root()
    if old is not None:
        if _has_speech(old / SPEECH_SUB):
            return old / SPEECH_SUB
        if _has_speech(old):
            return old
    return new


def _chatterbox_complete(d: Path) -> bool:
    try:
        from .engines.chatterbox import complete
        return complete(d)
    except Exception:       # noqa: BLE001 - "cannot tell" is "not complete"
        return False


def chatterbox_dir() -> Path:
    """The explicit env folder, else the first COMPLETE set (new, old host folder, old home
    folder), else the first folder holding part of one (a download in progress), else the new
    one. Completeness first: a move that stopped half way leaves a few files in the new folder
    and the rest in the old one, and the engine must keep loading from the whole set."""
    exp = _explicit(CHATTERBOX_ENV)
    if exp is not None:
        return exp
    new = root() / CHATTERBOX_SUB
    old_host = legacy_host_root()
    cands = [new] + ([old_host / CHATTERBOX_SUB] if old_host is not None else []) + [legacy_chatterbox_dir()]
    for d in cands:
        if _chatterbox_complete(d):
            return d
    for d in cands:
        if _has_chatterbox(d):
            return d
    return new


def plan() -> list[dict]:
    """The moves :func:`migrate` would make: ``[{family, src, dst, state}]`` where state is
    ``move`` (do it), ``skip_exists`` (the destination already has it), ``explicit`` (an env
    folder: left alone)."""
    out: list[dict] = []
    r = root()
    sp_new = r / SPEECH_SUB
    old_host = legacy_host_root()
    if _explicit(SPEECH_ENV) is not None:
        out.append({"family": "speech", "src": str(r), "dst": str(sp_new), "state": "explicit"})
    else:
        seen: set[str] = set()
        sources = [r] + ([old_host / SPEECH_SUB, old_host] if old_host is not None else [])
        for base in sources:
            for e in SPEECH_ENTRIES:
                src = base / e
                if not src.exists():
                    continue
                dst = sp_new / e
                taken = e in seen or dst.exists()
                seen.add(e)
                out.append({"family": "speech", "src": str(src), "dst": str(dst),
                            "state": "skip_exists" if taken else "move"})
    cb_new = r / CHATTERBOX_SUB
    olds = [legacy_chatterbox_dir()] + ([old_host / CHATTERBOX_SUB] if old_host is not None else [])
    if _explicit(CHATTERBOX_ENV) is not None:
        out.append({"family": "chatterbox", "src": str(olds[0]), "dst": str(cb_new), "state": "explicit"})
    else:
        # "Taken" only by a WHOLE set: a move that stopped half way (a file in use on Windows, a full
        # disk) must be resumable, not reported as done with half the graphs left behind.
        taken = _chatterbox_complete(cb_new)
        for old in olds:
            if _has_chatterbox(old) and old.resolve() != cb_new.resolve():
                out.append({"family": "chatterbox", "src": str(old), "dst": str(cb_new),
                            "state": "skip_exists" if taken else "move"})
                taken = True
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
                # Folders first, then files, each by name: directory order differs between file
                # systems, and the engine's graphs (the folder Windows holds open) must be the
                # first thing tried, so a refusal leaves the old set whole and loadable.
                for child in sorted(src.iterdir(), key=lambda c: (not c.is_dir(), c.name)):
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
    old_host = legacy_host_root()
    if old_host is not None:
        for d in (old_host / SPEECH_SUB, old_host / CHATTERBOX_SUB, old_host):
            with contextlib.suppress(OSError):
                d.rmdir()                   # only when now empty
    if report["moved"]:
        with contextlib.suppress(OSError):
            root().mkdir(parents=True, exist_ok=True)
            (root() / MARKER).write_text(json.dumps({"layout": LAYOUT, "moved": report["moved"]}), encoding="utf-8")
    return report


def legacy_present() -> bool:
    """Are models still at an old location (and movable)? For the status page."""
    return any(m["state"] == "move" for m in plan())
