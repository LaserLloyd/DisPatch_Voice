"""The interface voice chat calls: speak(text_chunks, profile_id) -> PCM frames.

Frames are 16-bit little-endian mono PCM at 24 kHz, FRAME_MS long, so the
caller can push them straight down a WebSocket and stop between any two of them
on barge-in. Set `stop` to abort mid-sentence (the engine checks it every token).
"""
from __future__ import annotations

import random
import threading
from typing import Iterable, Iterator

import numpy as np
import soundfile as sf

from . import audio
from .profiles import FILLERS, ProfileStore

FRAME_MS = 40
FRAME_SAMPLES = audio.SAMPLE_RATE * FRAME_MS // 1000


def frames(a: np.ndarray) -> Iterator[bytes]:
    pcm = audio.to_pcm16(a)
    step = FRAME_SAMPLES * 2
    for i in range(0, len(pcm), step):
        yield pcm[i:i + step]


def resolve(store: ProfileStore, profile_id: str | None) -> str:
    pid = profile_id or store.default_id()
    if not pid:
        raise LookupError("no voice profile chosen and no default set")
    store.get(pid)          # raises KeyError / ProfileError for unknown ids
    return pid


def speak(store: ProfileStore, text_chunks: Iterable[str], profile_id: str | None = None,
          stop: threading.Event | None = None) -> Iterator[bytes]:
    pid = resolve(store, profile_id)
    voice = store.voice(pid)
    for wav in store.engine.stream(text_chunks, voice, stop):
        for fr in frames(wav):
            if stop is not None and stop.is_set():
                return
            yield fr


def filler(store: ProfileStore, profile_id: str | None = None, rng: random.Random | None = None) -> bytes | None:
    """A pre-rendered acknowledgement in the profile's voice, or None if not rendered yet.

    Play it the moment the person's turn ends, so the voice "reacts" while the LLM works.
    """
    pid = resolve(store, profile_id)
    picks = list(range(len(FILLERS)))
    (rng or random).shuffle(picks)
    for i in picks:
        try:
            a, _ = sf.read(store.file(pid, f"fillers/{i}.wav"), dtype="float32")
        except KeyError:
            continue
        return audio.to_pcm16(a)
    return None
