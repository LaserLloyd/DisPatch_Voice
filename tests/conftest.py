import io
import threading

import numpy as np
import pytest
import soundfile as sf

from dispatch_voice import audio
from dispatch_voice.engine import VoiceTensors

SR = audio.SAMPLE_RATE


def speechlike(seconds=12.0, noise_db=-60.0, seed=1, clip=False):
    """Syllable-ish bursts of a harmonic tone with pauses, plus optional noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.3 * t)
    voice = sum(np.sin(2 * np.pi * k * np.cumsum(f0) / SR) / k for k in range(1, 6))
    env = (np.sin(2 * np.pi * 3.5 * t) > -0.2).astype(float)
    env *= (np.sin(2 * np.pi * 0.25 * t) > -0.7)          # occasional pauses
    x = 0.3 * voice * env
    x += rng.standard_normal(len(t)) * (10 ** (noise_db / 20))
    if clip:
        x = np.clip(x * 6, -1, 1)
    return x.astype(np.float32)


def wav_bytes(a, sr=SR):
    buf = io.BytesIO()
    sf.write(buf, a, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class FakeEngine:
    """Stands in for ChatterboxOnnx: deterministic, instant."""
    dtype = "fake"
    sample_rate = SR
    watermarking = False

    def __init__(self):
        self.calls = []
        self.last_stats = {}

    def build_voice(self, ref):
        v = float(np.mean(np.abs(ref)))
        return VoiceTensors(np.full((1, 4, 8), v, np.float32), np.arange(10, dtype=np.int64)[None],
                            np.full((1, 192), v, np.float32), np.ones((1, 5, 80), np.float32))

    def synthesize(self, text, voice, stop=None, seed=None):
        self.calls.append(text)
        n = int(0.06 * SR * max(1, len(text.split())))
        return (0.1 * np.sin(np.arange(n) / 10)).astype(np.float32)

    def similarity(self, a, b):
        return 0.9

    def stream(self, chunks, voice, stop=None):
        from dispatch_voice.engine import ChatterboxOnnx
        return ChatterboxOnnx.stream(self, chunks, voice, stop)


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def store(tmp_path, engine):
    from dispatch_voice.profiles import ProfileStore
    return ProfileStore(tmp_path / "voices", engine)
