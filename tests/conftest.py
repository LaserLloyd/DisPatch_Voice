import io
import os
import sys
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


# --------------------------------------------------------------------------- #
# v3 shared fixtures (C-§G; owned by T0 — tracks add their own tests/<area>/conftest.py
# and never edit this file without a note to T0). The doubles live in tests/fakes.py.
# --------------------------------------------------------------------------- #

# Each track's legacy copies (byte-identical from DisPatch @ e8a5f75, still importing
# `app.voice.*`) sit in tests/<area>/legacy/ until the owner ports them; never collected.
collect_ignore_glob = ["*/legacy/*"]


@pytest.fixture
def fake_ctx(tmp_path):
    """A FakeContext (ExtensionContext stand-in) over tmp_path, with four bots:
    sage (advisor), swift (api), nova (gateway), alpha (safe) and thread `t-voice-1` on sage."""
    from fakes import FakeContext

    ctx = FakeContext(tmp_path / "data")
    ctx.add_thread("t-voice-1", "sage", "Voice test")
    return ctx


@pytest.fixture
def fake_engines():
    """FakeSTT / FakeTTS / OneTurnDetector instances plus the FakeWS and ScriptedDetector classes."""
    from fakes import FakeEngines

    return FakeEngines()


@pytest.fixture
def b1a_frames():
    """The recorded B.1a frame sequences (tests/fixtures/frames_b1a.json)."""
    from fakes import recorded_frames

    return recorded_frames()


@pytest.fixture
def fixtures_dir():
    from fakes import FIXTURES

    return FIXTURES


# --------------------------------------------------------------------------- #
# Model-backed runs: a clean exit code
# --------------------------------------------------------------------------- #
# ONNX Runtime's native thread pools can be torn down by the interpreter's own
# exit in an order that aborts ("terminate called without an active exception",
# exit 134) AFTER every test passed and the report was printed (V1 #6): a CI that
# keys on the exit code saw red. Once pytest is completely done, a run that
# loaded models leaves with os._exit(pytest's own status) instead.
# DISPATCH_VOICE_TEST_HARD_EXIT=0 turns this off (to debug that teardown).
_MODELS_RAN = False


def pytest_runtest_setup(item):
    global _MODELS_RAN
    if item.get_closest_marker("models") is not None:
        _MODELS_RAN = True


def pytest_sessionfinish(session, exitstatus):
    session.config._voice_exitstatus = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    if _MODELS_RAN and os.environ.get("DISPATCH_VOICE_TEST_HARD_EXIT", "1") != "0":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(getattr(config, "_voice_exitstatus", 0))
