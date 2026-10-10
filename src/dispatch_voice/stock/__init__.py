"""The stock voice: what speaks when nobody has cloned a voice yet.

Chatterbox has no built-in voice; every voice is a profile built from a
reference clip. So that a fresh install can talk before anyone records
anything, this package ships ONE short reference clip of fully synthetic
speech (``synthetic-host.wav``, made with Kokoro-82M, Apache-2.0 weights; not a
real person) and turns it into an ordinary profile, ``stock-host``, in the
profile store:

* ``dispatch-voice fetch-models`` builds the voice tensors once and caches
  them next to the models (``<models>/stock/stock-host.npz``; never committed).
* :func:`ensure_stock` copies them into a profile directory. With
  ``build=True`` it builds them with the store's engine when the cache is
  missing (an install that fetched its models before this existed).
* :func:`render_extras` renders the audition line and the fillers, like a
  cloned profile gets after its build (slow; run it in the background).

The stock profile is never made the default by itself: a host app uses it
only when it has no other voice to speak with.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from importlib import resources
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import audio
from ..engine import VoiceTensors, default_model_dir

STOCK_ID = "stock-host"
STOCK_NAME = "Stock voice (synthetic)"
REFERENCE_NAME = "synthetic-host.wav"

_LOCK = threading.Lock()


def reference_path() -> Path:
    return Path(str(resources.files(__package__).joinpath(REFERENCE_NAME)))


def reference_audio() -> np.ndarray:
    """The packaged clip as mono float32 at audio.SAMPLE_RATE (no ffmpeg needed)."""
    a, sr = sf.read(reference_path(), dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != audio.SAMPLE_RATE:
        n = round(len(a) * audio.SAMPLE_RATE / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
    return a.astype(np.float32)


def cached_tensors_path(model_dir: Path | None = None) -> Path:
    return Path(model_dir or default_model_dir()) / "stock" / f"{STOCK_ID}.npz"


def build_tensors(engine, model_dir: Path | None = None) -> Path:
    """Build the stock voice tensors with ``engine`` and cache them."""
    voice = engine.build_voice(audio.prepare_reference(reference_audio()))
    path = cached_tensors_path(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    voice.save(path)
    return path


def ensure_stock(store, *, build: bool = True, model_dir: Path | None = None) -> str | None:
    """Make sure the stock profile exists in ``store``; its id, or None when
    it does not exist and could not be made (no cached tensors and
    ``build=False``, or no engine)."""
    d = Path(store.root) / STOCK_ID
    if (d / "profile.json").exists():
        return STOCK_ID
    # A cheap (build=False) caller, e.g. a status page, never waits behind a
    # build that is loading the engine: it just reports "not yet".
    if not _LOCK.acquire(blocking=build):
        return None
    try:
        return _ensure_locked(store, d, build, model_dir)
    finally:
        _LOCK.release()


def _ensure_locked(store, d: Path, build: bool, model_dir: Path | None) -> str | None:
    if (d / "profile.json").exists():
        return STOCK_ID
    cache = cached_tensors_path(model_dir)
    voice = None
    if cache.exists():
        try:
            voice = VoiceTensors.load(cache)
        except (OSError, ValueError, KeyError):   # a corrupt cache: rebuild
            voice = None
    if voice is None:
        if not build or store.engine is None:
            return None
        voice = store.engine.build_voice(audio.prepare_reference(reference_audio()))
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            voice.save(cache)
        except OSError:
            pass                       # read-only models dir: the profile still works
    raw = reference_audio()
    ref = audio.prepare_reference(raw)
    tmp = Path(store.root) / f".{STOCK_ID}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "fillers").mkdir(parents=True)
    try:
        sf.write(tmp / "reference.wav", ref, audio.SAMPLE_RATE, subtype="PCM_16")
        voice.save(tmp / "voice.npz")
        now = int(time.time())
        meta = {
            "name": STOCK_NAME,
            "created": now,
            "engine": "chatterbox-turbo-onnx",
            "stock": True,
            "source": {"filename": REFERENCE_NAME, "stock": True},
            "reference_s": round(len(ref) / audio.SAMPLE_RATE, 2),
            "quality": {**audio.analyse(raw).as_dict(), "match": None},
            "consent": {"kind": "synthetic",
                        "statement": "This is a synthetic or stock voice, not a real person.",
                        "note": "Bundled stock voice (Kokoro-82M synthetic speech).", "at": now},
            "watermarked": None,
            "fillers": "pending",
            "default": False,
        }
        (tmp / "profile.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
        os.replace(tmp, d)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return STOCK_ID


def render_extras(store, pid: str = STOCK_ID, *, fillers: bool = True) -> None:
    """The audition line and (``fillers``) the fillers for the stock profile
    (slow). Inside DisPatch only the audition is wanted: the acknowledgement
    lines are the host's render queue's (SPEC C12)."""
    from ..profiles import AUDITION_LINE
    d = Path(store.root) / pid
    if not (d / "audition.wav").exists():
        wav = store.engine.synthesize(AUDITION_LINE, store.voice(pid), seed=0)
        tmp = d / ".audition.wav.tmp"           # tmp + rename: never a half-written file
        sf.write(tmp, wav, audio.SAMPLE_RATE, subtype="PCM_16", format="WAV")
        os.replace(tmp, d / "audition.wav")
    if not fillers:
        return
    meta = json.loads((d / "profile.json").read_text())
    if meta.get("fillers") != "ready":
        store.render_fillers(pid)
