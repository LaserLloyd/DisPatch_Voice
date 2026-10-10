"""Fixtures for tests/host/ (owner: track T3). The shared doubles are in the root
conftest (fake_ctx, fake_engines, b1a_frames, fixtures_dir) and tests/fakes.py.

* ``host_env``: the host wired to ``fake_ctx`` (context set, every registry and
  model cache clean, no DISPATCH_VOICE* in the environment, the data dir under
  tmp_path). Yields the FakeContext.
* ``no_models``: ``VoiceSettings.missing()`` answers [] (the switch decides, not
  the files).
* ``fake_tts``: ``tts.shared_engine`` returns the FakeTTS of ``fake_engines``.
* ``voice_app``: ``host.setup(ctx)`` with the Chatterbox models absent, plus a
  TestClient on ``ctx.app`` and a ``/ws/voice/{thread_id}`` route that runs the
  FakeContext gate (so the socket tests go through a real ASGI websocket).
* ``live_app``: ``voice_app`` with voice forced on, fake engines, a one-turn
  detector and the power hold stubbed: a session reaches ``ready`` and speaks.
"""
from __future__ import annotations

import os

import pytest
from fastapi import WebSocket


def _clear_caches():
    from dispatch_voice.host import (
        profiles_bridge,
        registry,
        routes,
        stt,
        tts,
        vad_turn,
        ws,
    )
    registry.reset()
    routes._LOADED[:] = []
    routes._APPLIED["on"] = None
    routes._PREWARM.clear()
    routes._PREWARM["state"] = "idle"
    routes._PREVIEWS.clear()
    routes._NOTICES.clear()
    from dispatch_voice.host import acks
    acks.reset()
    with stt._ENGINES_LOCK:
        stt._ENGINES.clear()
    with tts._ENGINES_LOCK:
        tts._ENGINES.clear()
    with vad_turn._SMART_TURN_LOCK:
        vad_turn._SMART_TURN.clear()
    profiles_bridge._STORE = None
    profiles_bridge._EXTRAS_STARTED.clear()
    ws._SESSIONS.clear()


@pytest.fixture
def host_env(fake_ctx, monkeypatch):
    from dispatch_voice.host import config as vconfig
    from dispatch_voice.host import context
    for k in list(os.environ):
        if k.startswith("DISPATCH_VOICE") or k in ("DISPATCH_SPEECH_MODELS", "DISPATCH_DATA_DIR"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(vconfig, "_PREFS_CACHE", None)
    _clear_caches()
    context.set_context(fake_ctx)
    yield fake_ctx
    _clear_caches()
    context.clear_context()


@pytest.fixture
def no_models(monkeypatch):
    from dispatch_voice.host.config import VoiceSettings
    monkeypatch.setattr(VoiceSettings, "missing", lambda self: [])


@pytest.fixture
def fake_tts(monkeypatch, fake_engines):
    from dispatch_voice.host import tts
    eng = fake_engines.tts
    monkeypatch.setattr(tts, "shared_engine", lambda settings: eng)
    return eng


def mount_ws_route(ctx) -> None:
    """``/ws/voice/{thread_id}`` on ctx.app: the FakeContext gate + the registered handler."""

    @ctx.app.websocket("/ws/voice/{thread_id}")
    async def voice_socket(websocket: WebSocket, thread_id: str):
        await ctx.connect("voice", thread_id, websocket)


@pytest.fixture
def voice_app(host_env, monkeypatch):
    from fastapi.testclient import TestClient

    from dispatch_voice.host import setup
    from dispatch_voice.host.engines import chatterbox
    monkeypatch.setattr(chatterbox, "available", lambda: False)
    setup(host_env)
    mount_ws_route(host_env)
    with TestClient(host_env.app) as c:
        yield c, host_env


@pytest.fixture
def live_app(voice_app, no_models, fake_engines, monkeypatch):
    """Voice on, fake engines, every audio chunk ends a turn, no power hold."""
    from dispatch_voice.host import power, routes, vad_turn
    monkeypatch.setenv("DISPATCH_VOICE", "1")
    monkeypatch.setenv("DISPATCH_VOICE_MERGE_WINDOW", "0")

    async def fake_warm(settings):
        return fake_engines.stt, fake_engines.tts

    monkeypatch.setattr(routes, "warm_models", fake_warm)
    monkeypatch.setattr(routes, "warm_models", fake_warm)
    monkeypatch.setattr(vad_turn, "build_detector", lambda s: fake_engines.OneTurnDetector())
    monkeypatch.setattr(routes, "voice_for_bot", lambda s, e, b: None)
    holds: list[str] = []
    monkeypatch.setattr(power, "acquire", lambda settings: holds.append("acquire") or "off")
    monkeypatch.setattr(power, "release", lambda: holds.append("release") or "off")
    c, ctx = voice_app
    ctx.power_calls = holds          # type: ignore[attr-defined]
    return c, ctx
