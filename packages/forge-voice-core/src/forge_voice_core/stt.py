"""Speech-to-text: NVIDIA Parakeet TDT 0.6B v2 (English) through onnx-asr, CPU.

Why this runtime: onnx-asr runs the NeMo export with nothing but numpy and
onnxruntime — no torch, no NeMo, no CUDA, which matters on a ROCm/CPU box and
keeps the uv env small. Parakeet is an offline (whole-utterance) model here:
the Endpointer hands it one finished turn and it returns in a small fraction
of the turn's length (see docs/voice-chat/turn-taking.md for measured numbers), so a
streaming partial adds CPU cost for no audible gain in a car.

The interface still matches the plan's ``feed → partial/final``: ``feed`` only
buffers, ``final`` transcribes. A streaming engine can drop in behind it.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

import numpy as np

from . import _log

log = _log.get()


class ParakeetSTT:
    def __init__(self, model_dir: Path, threads: int = 4, *, spinning: bool = False) -> None:
        import onnx_asr

        from .rt.stt_stream import ort_options
        # A bursty session (the tail transcript at every pause, the final at the
        # end): ORT intra-op spinning OFF unless ``rt.stt_spinning`` (lead-measured,
        # SPEC §0.2: −9…−23 % under load, and no 0.5 core-second burn after each
        # call stealing time from the decoder; contracts §B.3 / Amendment T4-1 (f)).
        so = ort_options(threads, spinning=spinning)
        self.threads = max(1, threads)
        self.spinning = spinning
        self._model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v2", str(model_dir), quantization="int8",
            sess_options=so, providers=["CPUExecutionProvider"])
        # onnxruntime sessions are thread-safe for run(), but one transcription
        # at a time keeps a second driver from halving the first one's speed.
        self._lock = threading.Lock()

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size < 1600:          # <0.1 s: nothing to hear
            return ""
        with self._lock:
            text = self._model.recognize(audio.astype(np.float32), sample_rate=16000)
        return (text or "").strip()


class Transcriber:
    """Per-session façade: ``feed`` buffers, ``final`` transcribes off-loop."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self._chunks: list[np.ndarray] = []

    def feed(self, audio: np.ndarray) -> None:
        self._chunks.append(audio)

    async def final(self, audio: np.ndarray | None = None) -> str:
        if audio is None:
            audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        self._chunks = []
        return await asyncio.to_thread(self.engine.transcribe, audio)


def remote_engine(settings):
    """``stt_provider = openai``: the same ``transcribe(audio) -> str`` seam, over
    ``POST {stt_url}/v1/audio/transcriptions`` (providers.OpenAISTTProvider). Not cached: it
    holds no model, and a changed address or model applies to the next chat."""
    import os

    from . import providers
    return providers.OpenAISTTEngine(providers.OpenAISTTProvider(
        settings.stt_url, settings.stt_model, api_key=os.environ.get("DISPATCH_VOICE_STT_API_KEY", "").strip(),
        timeout_s=float(settings.stt_timeout_s), allow=tuple(getattr(settings, "rig_allow", ()) or ())))


_ENGINES: dict[str, object] = {}
_ENGINES_LOCK = threading.Lock()


def shared_engine(settings):
    """Load once per process (the encoder is ~650 MB; loading takes seconds)."""
    if getattr(settings, "stt_provider", "cpu") == "openai":
        return remote_engine(settings)
    key = f"{settings.stt_engine}:{settings.parakeet_dir}"
    with _ENGINES_LOCK:
        if key not in _ENGINES:
            if settings.stt_engine != "parakeet":
                raise RuntimeError(f"unknown STT engine {settings.stt_engine!r}")
            rt = getattr(settings, "rt", None)
            _ENGINES[key] = ParakeetSTT(settings.parakeet_dir, settings.threads,
                                        spinning=bool(getattr(rt, "stt_spinning", False)))
        return _ENGINES[key]
