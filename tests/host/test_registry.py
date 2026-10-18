"""The add-on's own plugin seam (host/registry.py), as the Kokoro plugin uses it.

A plugin plugs in through DISPATCH_VOICE_EXTENSIONS. These tests pin what the
add-on promises it: its router is mounted under /api/voice behind DisPatch's
gate whatever it declares, its engine is selectable by name, a broken plugin
cannot stop the add-on, and the resolver/ack/persona hooks fail soft.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from dispatch_voice.host import persona, registry, routes, tts
from dispatch_voice.host.config import VoiceSettings


@pytest.fixture(autouse=True)
def _clean_registry(host_env):
    yield
    for name in [m for m in sys.modules if m.startswith("fake_voice_ext")]:
        del sys.modules[name]


def install(name: str, setup_fn) -> None:
    mod = types.ModuleType(name)
    mod.setup = setup_fn
    sys.modules[name] = mod


def ext_router():
    r = APIRouter(prefix="/profiles-ext")

    @r.get("")
    async def list_profiles():          # declares NO auth of its own
        return {"profiles": ["host"]}

    return r


def test_extension_router_is_unlocked_only_even_without_its_own_gate(host_env):
    install("fake_voice_ext_a", lambda api: api.include_router(ext_router()))
    assert registry.mount_extensions(host_env, "fake_voice_ext_a:setup") == ["fake_voice_ext_a:setup"]
    c = TestClient(host_env.app)
    assert c.get("/api/voice/profiles-ext").json() == {"profiles": ["host"]}
    host_env.operator = False
    assert c.get("/api/voice/profiles-ext").status_code == 403
    assert [p for _, p, _ in host_env.routers] == ["/api/voice"]


def test_extension_router_with_a_prefix_argument(host_env):
    r = APIRouter()

    @r.get("/x")
    async def x():
        return {"x": 1}

    registry.ExtensionAPI(host_env).include_router(r, prefix="plug")
    assert [p for _, p, _ in host_env.routers] == ["/api/voice/plug"]
    assert TestClient(host_env.app).get("/api/voice/plug/x").json() == {"x": 1}


def test_an_absolute_api_voice_router_is_refused(host_env):
    r = APIRouter(prefix="/api/voice/old")

    @r.get("")
    async def old():
        return {}

    with pytest.raises(ValueError, match="relative"):
        registry.ExtensionAPI(host_env).include_router(r)
    with pytest.raises(RuntimeError, match="context"):
        registry.ExtensionAPI(None).include_router(ext_router())


def test_broken_extension_is_skipped_and_others_still_load(host_env):
    def boom(api):
        raise RuntimeError("bad extension")

    install("fake_voice_ext_bad", boom)
    install("fake_voice_ext_ok", lambda api: api.include_router(ext_router()))
    loaded = registry.mount_extensions(
        host_env, "fake_voice_ext_bad:setup, nonexistent_pkg_xyz:setup ,fake_voice_ext_ok:setup")
    assert loaded == ["fake_voice_ext_ok:setup"]


def test_registered_engine_is_selected_by_name(host_env):
    class Eng:
        sample_rate = 22050
        sample_format = "pcm_s16le"
        channels = 1
        streaming = True

        def stream(self, text, profile_id=None, stop=None):
            yield np.zeros(10, "<i2")

        def synth(self, text, profile_id=None):
            return np.zeros(10, "<i2"), 22050

    install("fake_voice_ext_eng", lambda api: api.register_engine("cloner", lambda s: Eng()))
    registry.mount_extensions(None, "fake_voice_ext_eng:setup")     # the CLI path: no context
    eng = tts.build_engine(VoiceSettings(tts_engine="cloner"))
    assert isinstance(eng, Eng) and eng.sample_rate == 22050


def test_unknown_engine_is_an_error_not_a_silent_default():
    with pytest.raises(RuntimeError, match="unknown TTS engine"):
        tts.build_engine(VoiceSettings(tts_engine="nope"))


def test_hq_fallback_wraps_any_engine():
    class Local:
        sample_rate = 24000
        streaming = False

        def stream(self, text, profile_id=None, stop=None):
            yield np.ones(5, "<i2")

    registry.register_engine("local", lambda s: Local())
    eng = tts.build_engine(VoiceSettings(tts_engine="local", hq_url="http://127.0.0.1:9/", hq_timeout_s=0.2))
    assert isinstance(eng, tts.FallbackTTS)
    out = list(eng.stream("hi"))                 # HQ unreachable -> local voice
    assert len(out) == 1 and out[0].tolist() == [1] * 5
    assert eng._down_until > 0                   # and it stays on local for a while


def test_profile_ack_and_persona_hooks_fail_soft():
    assert registry.resolve_profile("sage") is None
    base = persona.instructions_for("sage", None, backend="api", tags_ok=False)
    assert base.system.startswith("## Spoken replies") and base.trailer == ""
    registry.set_profile_resolver(lambda bot: {"sage": "host"}.get(bot))
    registry.set_ack_provider(lambda pid: [np.zeros(3, "<i2")] if pid == "host" else [])
    registry.set_persona_provider(lambda bot, pid: "Be brief." if pid == "host" else None)
    assert registry.resolve_profile("sage") == "host" and registry.resolve_profile("x") is None
    assert len(registry.acks_for("host")) == 1 and registry.acks_for(None) is None
    # An extension's persona is used verbatim as the system section, with the
    # driver's length/follow-up choice appended (T5's builder).
    instr = persona.instructions_for("sage", "host", style="short", follow_up=False, backend="api", tags_ok=False)
    assert instr.system.startswith("Be brief.") and "two short sentences" in instr.system
    assert "Do not end with a question" in instr.system and instr.trailer == ""
    assert persona.instructions_for("x", None, backend="api", tags_ok=False).system == base.system

    def boom(*a):
        raise RuntimeError("down")

    registry.set_profile_resolver(boom)
    registry.set_ack_provider(boom)
    registry.set_persona_provider(boom)
    assert registry.resolve_profile("sage") is None and registry.acks_for("host") is None
    assert persona.instructions_for("sage", "host", backend="api", tags_ok=False).system == base.system


def test_voice_for_bot_is_the_resolver_and_acks_come_from_the_render_queue(host_env):
    from dispatch_voice.host import acks as hacks

    class Eng:
        sample_rate = 24000

        def __init__(self):
            self.calls = []

        def synth(self, text, profile_id=None):
            self.calls.append((text, profile_id))
            return np.zeros(24000, "<i2"), 24000          # 1 s: passes the sanity check

    registry.set_profile_resolver(lambda bot: "host" if bot == "sage" else None)
    registry.set_ack_provider(lambda pid: [np.ones(2, "<i2")] if pid == "host" else None)
    eng = Eng()
    assert routes.voice_for_bot(VoiceSettings(), eng, "sage") == "host"
    assert routes.voice_for_bot(VoiceSettings(), eng, "other") is None
    # The kit queues this voice's lines (pools + notices + tool lines) in the
    # background and reads them back; a plugin's old-style acks are not used.
    kit = hacks.kit_for(VoiceSettings(), eng, bot_id="sage", profile_id="host", mode="often", name="Host")
    assert kit.queue is not None and kit.queue.wait_idle(10)
    said = {t for t, _ in eng.calls}
    assert "Sorry, that didn't go through." in said and "Searching" in said
    assert all(pid == "host" for _, pid in eng.calls)
    assert kit.has_rendered() and kit.pcm("Searching") is not None and kit.clip_s("Searching") == 1.0
    st = hacks.status()
    assert st["host"]["ready"] == st["host"]["total"] > 10 and st["host"]["failed"] == 0 and st["host"]["name"] == "Host"
    assert hacks.rebuild("host") is True and hacks.rebuild("nope") is False
    hacks.release()
    assert hacks.status() == {}


def test_status_route_is_unlocked_only(voice_app):
    c, ctx = voice_app
    ctx.operator = False
    assert c.get("/api/voice/status").status_code == 403
    ctx.operator = True
    st = c.get("/api/voice/status").json()
    assert "enabled" in st and "missing" in st and "extensions" in st


def test_a_module_target_in_voice_yaml_is_never_imported(host_env, monkeypatch):
    """The `extensions` setting is written by any operator (PATCH): a module:setup target found
    there must not run (`sys:exit` would stop DisPatch at every boot). Only the environment may
    name one; installed entry-point plugins load from either."""
    from dispatch_voice.host import config as vconfig
    monkeypatch.delenv("DISPATCH_VOICE_EXTENSIONS", raising=False)
    ran = []
    install("fake_voice_ext_file", lambda api: ran.append("ran"))
    vconfig.save_prefs({"extensions": ["sys:exit", "fake_voice_ext_file:setup", "fake_voice_ext_file"]})
    assert registry.mount_extensions(host_env, eps=[]) == [] and ran == []
    monkeypatch.setenv("DISPATCH_VOICE_EXTENSIONS", "fake_voice_ext_file:setup")
    assert registry.mount_extensions(host_env, eps=[]) == ["fake_voice_ext_file:setup"] and ran == ["ran"]


def test_a_target_that_exits_does_not_stop_the_boot(host_env):
    assert registry.mount_extensions(host_env, "sys:exit", eps=[]) == []


def test_patch_refuses_a_new_module_target(voice_app, monkeypatch):
    c, _ = voice_app
    r = c.patch("/api/voice/settings", json={"extensions": ["sys:exit"]})
    assert r.status_code == 422 and "DISPATCH_VOICE_EXTENSIONS" in r.text
    assert c.patch("/api/voice/settings", json={"extensions": ["kokoro"]}).status_code == 200
