"""Fixtures for tests/rt/ (owner: track T4). The shared doubles are in the root
conftest (fake_ctx, fake_engines, b1a_frames, fixtures_dir) and tests/fakes.py.

Here: a settable fake clock, a scripted recogniser for the tail STT, a fake
Chatterbox engine for the pipelined TTS (tokens with a programmable delay, a
decoder that returns 960 synthetic samples per token so chunk arithmetic can
be checked exactly), and the model-dir probe the model-backed tests skip on.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from forge_voice_core.engine import SAMPLES_PER_TOKEN, VoiceTensors


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


class ScriptedRecognizer:
    """``transcribe(audio) -> text`` driven by a script keyed on the audio's length in
    seconds (rounded to 0.1 s) or by a default function; records every call."""

    def __init__(self, fn=None, delay_s: float = 0.0) -> None:
        self.fn = fn or (lambda a: f"{len(a) / 16000:.1f}s")
        self.delay_s = delay_s
        self.calls: list[int] = []

    def __call__(self, audio: np.ndarray) -> str:
        self.calls.append(int(audio.size))
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.fn(audio)


@pytest.fixture
def recognizer() -> ScriptedRecognizer:
    return ScriptedRecognizer()


def speech_frames(seconds: float, *, prob: float = 0.9, seed: int = 0) -> list[tuple[np.ndarray, float]]:
    """(frame, vad_prob) pairs of 512 samples of low-level noise."""
    rng = np.random.default_rng(seed)
    n = round(seconds * 16000 / 512)
    return [((rng.standard_normal(512) * 0.1).astype(np.float32), prob) for _ in range(n)]


def silence_frames(seconds: float, *, prob: float = 0.05) -> list[tuple[np.ndarray, float]]:
    n = round(seconds * 16000 / 512)
    return [(np.zeros(512, np.float32), prob) for _ in range(n)]


class FakeVoiceStore:
    def __init__(self, created: str = "1700000000") -> None:
        self.created = created
        self.voices: dict[str, VoiceTensors] = {}

    def add(self, pid: str, prompt_s: float = 10.0) -> VoiceTensors:
        n = int(prompt_s * 25)
        v = VoiceTensors(np.zeros((1, 151, 1024), np.float32), np.arange(n, dtype=np.int64)[None],
                         np.zeros((1, 192), np.float32), np.zeros((1, n * 2, 80), np.float32))
        self.voices[pid] = v
        return v

    def voice(self, pid: str) -> VoiceTensors:
        return self.voices[pid]

    def get(self, pid: str) -> dict:
        if pid not in self.voices:
            raise KeyError(pid)
        return {"id": pid, "created": self.created}

    def default_id(self) -> str | None:
        return next(iter(self.voices), None)


class FakeChatterbox:
    """Stands in for ChatterboxOnnx's pipelined surface.

    ``stream_tokens`` yields ``n_tokens`` tokens, ``token_delay_s`` apart (after
    ``prefill_s``); ``decode_chunk`` returns a deterministic 440 Hz tone whose phase
    is continuous in the token index (so a correct seam is seamless and a wrong one
    is a visible jump) after ``decode_s``, honouring left_context/trailing_silence
    exactly like the real decoder (960 samples per kept token).
    """

    def __init__(self, n_tokens: int = 60, token_delay_s: float = 0.0, prefill_s: float = 0.0,
                 decode_s: float = 0.0, decoder_sessions: int = 1) -> None:
        self.n_tokens = n_tokens
        self.token_delay_s = token_delay_s
        self.prefill_s = prefill_s
        self.decode_s = decode_s
        self.decoder_sessions = decoder_sessions
        self.decoder_threads = 6
        self.decode_calls: list[dict] = []
        self.synth_calls: list[tuple] = []
        self.sessions_set: list[tuple[int, int | None]] = []
        self.prefix_used: list[bool] = []          # per stream_tokens call (PrefixedFakeChatterbox)
        self._lock = threading.Lock()

    def slice_prompt(self, voice, prompt_s):
        from forge_voice_core.engine import ChatterboxOnnx
        return ChatterboxOnnx.slice_prompt(voice, prompt_s)

    def set_decoder_sessions(self, n, threads=None):
        self.sessions_set.append((n, threads))
        self.decoder_sessions = n

    def model_tag(self) -> str:
        return "deadbeef"

    def stream_tokens(self, text, voice, stop=None, seed=None):
        if self.prefill_s:
            time.sleep(self.prefill_s)
        for i in range(self.n_tokens):
            if stop is not None and stop.is_set():
                return
            if self.token_delay_s:
                time.sleep(self.token_delay_s)
            yield 100 + i

    def decode_chunk(self, tokens, voice, *, left_context=8, prompt_s=4.0, session=0,
                     trailing_silence=3, keep_silence=False, context_tail=0, right_context=0):
        if self.decode_s:
            time.sleep(self.decode_s)
        toks = list(tokens)
        with self._lock:
            self.decode_calls.append({"tokens": toks, "left": left_context, "right": right_context,
                                      "session": session, "keep_silence": keep_silence,
                                      "context_tail": context_tail,
                                      "prompt_tokens": int(voice.prompt_token.shape[1]),
                                      "t": time.perf_counter()})
        first_index = toks[0] - 100
        n_all = len(toks) + trailing_silence
        t = (np.arange(n_all * SAMPLES_PER_TOKEN) + first_index * SAMPLES_PER_TOKEN) / 24000.0
        wav = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        start = max(0, left_context * SAMPLES_PER_TOKEN - context_tail) if left_context else 0
        if keep_silence and not right_context:
            end = len(wav)
        else:
            end = len(wav) - (trailing_silence + right_context) * SAMPLES_PER_TOKEN
        return wav[start:end]

    def synthesize(self, text, voice, stop=None, seed=None):
        self.synth_calls.append((text, seed))
        return np.full(2400, 0.25, np.float32)


class PrefixedFakeChatterbox(FakeChatterbox):
    """The same fake with the LM prefix cache surface (``prefix_cache`` /
    ``stream_tokens(prefix=)``): records how many prefixes were built and whether each
    sentence used one."""

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.prefix_calls: list[int] = []

    def prefix_cache(self, voice):
        from forge_voice_core.engine import PrefixKV
        self.prefix_calls.append(int(voice.cond_emb.shape[1]))
        return PrefixKV(length=int(voice.cond_emb.shape[1]), kv={"k": np.zeros((1, 16, 151, 64), np.float32)})

    def stream_tokens(self, text, voice, stop=None, seed=None, *, prefix=None):
        self.prefix_used.append(prefix is not None)
        yield from super().stream_tokens(text, voice, stop, seed)


@pytest.fixture
def fake_store() -> FakeVoiceStore:
    s = FakeVoiceStore()
    s.add("stock")
    return s


@pytest.fixture
def fake_engine() -> FakeChatterbox:
    return FakeChatterbox()


# --------------------------------------------------------------------------- #
# Model directories (read-only use of the live model dirs is allowed)
# --------------------------------------------------------------------------- #

def speech_models_dir() -> Path:
    env = os.environ.get("DISPATCH_SPEECH_MODELS")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".local/share/dispatch-voice-models"


def have_speech_models() -> bool:
    d = speech_models_dir()
    return (d / "silero_vad.onnx").exists() and (d / "parakeet-tdt-0.6b-v2" / "encoder-model.int8.onnx").exists()


def have_smart_turn() -> bool:
    return (speech_models_dir() / "smart-turn-v3.2-cpu.onnx").exists()


def have_chatterbox() -> bool:
    from forge_voice_core.engine import COMPONENTS, _filename, default_model_dir
    d = default_model_dir()
    return all((d / "onnx" / _filename(c, "q4")).exists() for c in COMPONENTS) and \
        (d / "stock" / "stock-host.npz").exists()


def have_onnx_asr() -> bool:
    import importlib.util
    return importlib.util.find_spec("onnx_asr") is not None
