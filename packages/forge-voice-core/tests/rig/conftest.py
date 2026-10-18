"""Fixtures for tests/rig/: the GPU rig engine against a fake sf.1 rig (no rig,
no model, no network beyond 127.0.0.1:19100-19109)."""
from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest
from settling import pause

from forge_voice_core import rig as rig_mod
from forge_voice_core.rt.stt_stream import Partial

from .fake_sf1 import TOKEN, FakeRig


@pytest.fixture(autouse=True)
def _clean_rig_state(monkeypatch, tmp_path):
    rig_mod._MISCONFIGURED.clear()
    rig_mod._LAST.clear()
    rig_mod.METRICS.reset()
    monkeypatch.setattr(rig_mod, "HTTP_TRANSPORT", None)
    monkeypatch.delenv(rig_mod.TOKEN_ENV, raising=False)
    monkeypatch.setattr(rig_mod, "token_path", lambda: tmp_path / "speechforge.env")
    yield
    rig_mod._MISCONFIGURED.clear()


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setenv(rig_mod.TOKEN_ENV, TOKEN)
    return TOKEN


@pytest.fixture
async def fake_rig():
    r = await FakeRig().start()
    try:
        yield r
    finally:
        await r.stop()


class RecordingPower:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def acquire(self, settings):
        self.calls.append("acquire")
        return "held"

    def release(self):
        self.calls.append("release")
        return "off"


async def wait_for(pred, timeout: float = 3.0, step: float = 0.01):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        await pause(step)           # real time: asyncio.sleep(0.01) returns at once on Windows
    return bool(pred())


def make_conn(fake, *, power=None, token: str | None = None, **settings) -> rig_mod.RigConnection:
    rs = rig_mod.RigSettings(enabled=True, url=fake.url, **settings)
    gate = rig_mod.RigPowerGate(None, power=power) if power is not None else None
    return rig_mod.RigConnection(rs, token or fake.token, version="test", power=gate)


async def proven(conn, timeout: float = 3.0) -> bool:
    return await wait_for(conn.usable, timeout)


class FakeCpuSTT:
    """The CPU recogniser behind the rig: records every frame it hears."""

    def __init__(self, text: str = "cpu words") -> None:
        self.text = text
        self.pushed: list[np.ndarray] = []
        self.calls: list[str] = []
        self.resets = 0
        self.tick_interval_s = 1.0
        self._lock = threading.Lock()

    def push(self, pcm16, *, vad_prob):
        with self._lock:
            self.pushed.append(np.array(pcm16, copy=True))

    def at_pause(self):
        self.calls.append("at_pause")
        return Partial("", self.text, 0, False, 0.0, 1)

    def tick(self):
        self.calls.append("tick")
        return None

    def final(self, *, timeout_s=None):
        self.calls.append("final")
        return self.text

    def reset(self):
        self.resets += 1
        with self._lock:
            self.pushed.clear()

    def samples(self) -> np.ndarray:
        with self._lock:
            return np.concatenate(self.pushed) if self.pushed else np.zeros(0, "<i2")


class FakeCpuTTS:
    """The CPU engine behind the reply engine: 40 ms frames of a CPU-only value."""
    sample_rate = 24000
    sample_format = "pcm_s16le"
    channels = 1
    streaming = True
    VALUE = 7

    def __init__(self, frames: int = 2) -> None:
        self.frames = frames
        self.said: list[str] = []

    def stream(self, text, profile_id=None, stop=None):
        self.said.append(text)
        for _ in range(self.frames):
            if stop is not None and stop.is_set():
                return
            yield np.full(960, self.VALUE, "<i2")

    def synth(self, text, profile_id=None):
        return np.concatenate(list(self.stream(text, profile_id))), self.sample_rate

    def cache_tag(self, profile_id):
        return "cb-cpu-tag"


def frames(n: int, value: int = 100, size: int = 512) -> list[np.ndarray]:
    """n mic frames (int16, 32 ms) with distinct values so order is checkable."""
    return [np.full(size, value + i, "<i2") for i in range(n)]
