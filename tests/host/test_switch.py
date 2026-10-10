"""Voice as an optional feature: the Settings › Voice switch, the server
preferences (voice.yaml) and the Voice lab, on the add-on's own routes.

* The switch (``enabled`` in <data>/voice.yaml) is OFF on a fresh install,
  even with every package and model present. ``DISPATCH_VOICE=1`` / ``0``
  overrides it, and then the API refuses to move it (409).
* It applies live: off ends open voice chats with 1013 and drops the loaded
  speech models; on starts the prewarm. Nothing is preloaded while off.
* /ws/voice refuses with 1013 and a reason that names the switch.
* The Voice lab's settings (merge window, hard silence) and the v3 server
  settings (max_sessions, rt.power_hold) are validated, persisted and resettable.

Ported from DisPatch's test_voice_optional.py onto the FakeContext; the
tools.yaml rows and the per-session slots are DisPatch's and stay there.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time

import numpy as np
import pytest
import yaml
from starlette.websockets import WebSocketDisconnect

from dispatch_voice.host import config as vconfig
from dispatch_voice.host import profiles_bridge, registry, routes
from dispatch_voice.host import ws as ws_mod
from dispatch_voice.host.config import VoiceSettings

THREAD = "t-voice-1"



def rt_settings_out(**over):
    """The ``rt`` tree GET/PATCH /api/voice/settings returns: every C-§F.1 key (defaults)
    plus the hold, with ``over`` applied."""
    from dispatch_voice.rt.settings import RtSettings
    d = RtSettings().to_dict()
    d.pop("bots", None)
    d.pop("hard_silence_s", None)
    d.update(over)
    return d


def rt_status_out(power_hold="performance"):
    """The ``rt`` block of /api/voice/status (C-§D)."""
    from dispatch_voice.rt.settings import RtSettings
    r = RtSettings()
    return {"stt": r.stt_engine, "speculate": r.speculate, "decoder_prompt_s": r.decoder_prompt_s,
            "decoder_sessions": r.decoder_sessions, "addendum_s": r.addendum_s, "power_hold": power_hold}

def voice_yaml(ctx) -> dict:
    p = ctx.paths.voice_yaml
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def write_yaml(ctx, text: str) -> None:
    ctx.paths.voice_yaml.parent.mkdir(parents=True, exist_ok=True)
    ctx.paths.voice_yaml.write_text(text)


async def _noop_apply(was_on, settings):
    return None


# --------------------------------------------------------------------------- #
# The switch: default, env override, persistence
# --------------------------------------------------------------------------- #

def test_fresh_install_is_off_even_when_everything_is_installed(host_env, no_models):
    st = vconfig.load_settings()
    assert (st.forced, st.switch, st.switched_on, st.enabled) == (None, False, False, False)
    s = st.status()
    assert s["enabled"] is False and s["switch"] is False and s["ready"] is True
    assert s["max_sessions"] == 3 and s["rt"] == rt_status_out()


@pytest.mark.parametrize("flag,forced,on", [
    ("1", "on", True), ("true", "on", True), ("yes", "on", True), ("ON", "on", True),
    ("0", "off", False), ("false", "off", False), ("", "off", False), ("maybe", "off", False),
    ("auto", None, False),
])
def test_env_flag_forces_the_switch(host_env, no_models, monkeypatch, flag, forced, on):
    monkeypatch.setenv("DISPATCH_VOICE", flag)
    write_yaml(host_env, "enabled: true\n")
    st = vconfig.load_settings()
    assert st.forced == forced
    assert st.switched_on is (True if forced is None else on)
    assert st.enabled is (True if forced is None else on)


def test_switch_on_but_not_ready_is_not_enabled(host_env, monkeypatch):
    write_yaml(host_env, "enabled: true\n")
    monkeypatch.setattr(VoiceSettings, "missing", lambda self: ["model silero_vad.onnx"])
    st = vconfig.load_settings()
    assert st.switched_on is True and st.enabled is False
    assert st.status()["switch"] is True and st.status()["enabled"] is False


def test_existing_install_migrates_by_adding_enabled_true(host_env, no_models):
    """The live box's voice.yaml predates the switch (2.1 names). Adding one
    line turns voice back on and keeps the old preference."""
    write_yaml(host_env, "kokoro_speed: 1.3\n")
    assert vconfig.load_settings().enabled is False
    write_yaml(host_env, "kokoro_speed: 1.3\nenabled: true\n")
    st = vconfig.load_settings()
    assert st.enabled is True and st.speed == 1.3


@pytest.mark.parametrize("raw", ["enabled: yes-please\n", "enabled: 1\n", "enabled: null\n"])
def test_a_junk_switch_value_reads_as_off(host_env, raw):
    write_yaml(host_env, raw)
    assert vconfig.load_settings().switch is False


def test_v3_prefs_read_and_clean(host_env):
    # A hand edit writes `off`, which YAML reads as boolean false: it still means off.
    write_yaml(host_env, "max_sessions: 6\nrt:\n  power_hold: off\n  bogus: 1\n")
    st = vconfig.load_settings()
    assert st.max_sessions == 6 and st.power_hold == "off"
    write_yaml(host_env, "rt:\n  power_hold: 'off'\n")
    assert vconfig.load_settings().power_hold == "off"
    write_yaml(host_env, "rt:\n  power_hold: on\n")
    assert vconfig.load_settings().power_hold == "performance"
    assert vconfig.clean_prefs({"max_sessions": 0, "rt": {"power_hold": "turbo"}}) == {}
    assert vconfig.clean_prefs({"max_sessions": 8.0, "rt": {"power_hold": "OFF"}}) == {
        "max_sessions": 8, "rt": {"power_hold": "off"}}
    assert vconfig.clean_prefs({"max_sessions": True}) == {}


def test_power_hold_env_default(host_env, monkeypatch):
    monkeypatch.setenv("DISPATCH_VOICE_POWER_HOLD", "off")
    assert vconfig.load_settings().power_hold == "off"
    monkeypatch.setenv("DISPATCH_VOICE_POWER_HOLD", "nonsense")
    assert vconfig.load_settings().power_hold == "performance"
    monkeypatch.setenv("DISPATCH_VOICE_MAX_SESSIONS", "99")
    assert vconfig.load_settings().max_sessions == 8


def test_patch_enabled_persists_and_status_follows(voice_app, no_models, monkeypatch):
    c, ctx = voice_app
    monkeypatch.setattr(routes, "apply_switch", _noop_apply)
    st = c.get("/api/voice/status").json()
    assert (st["enabled"], st["switch"], st["forced"]) == (False, False, None)
    r = c.patch("/api/voice/settings", json={"enabled": True})
    assert r.status_code == 200, r.text
    assert r.json()["enabled"] is True and r.json()["forced"] is None
    assert voice_yaml(ctx) == {"enabled": True}
    st = c.get("/api/voice/status").json()
    assert (st["enabled"], st["switch"]) == (True, True)
    assert c.patch("/api/voice/settings", json={"enabled": False}).json()["enabled"] is False
    assert voice_yaml(ctx) == {"enabled": False}


@pytest.mark.parametrize("bad", ["yes", 1, None, "true"])
def test_patch_enabled_must_be_a_bool(voice_app, bad):
    c, ctx = voice_app
    assert c.patch("/api/voice/settings", json={"enabled": bad}).status_code == 422
    assert voice_yaml(ctx) == {}


def test_patch_enabled_refused_while_the_env_decides(voice_app, no_models, monkeypatch):
    c, ctx = voice_app
    monkeypatch.setenv("DISPATCH_VOICE", "1")
    assert c.patch("/api/voice/settings", json={"enabled": False}).status_code == 409
    # Saying what is already true is not a change.
    assert c.patch("/api/voice/settings", json={"enabled": True}).status_code == 200
    monkeypatch.setenv("DISPATCH_VOICE", "0")
    assert c.patch("/api/voice/settings", json={"enabled": True}).status_code == 409
    assert c.get("/api/voice/status").json()["forced"] == "off"
    assert "enabled" not in voice_yaml(ctx)


# --------------------------------------------------------------------------- #
# Live: refusal, preload, release
# --------------------------------------------------------------------------- #

def _close(c, path):
    with pytest.raises(WebSocketDisconnect) as e, c.websocket_connect(path) as ws:
        ws.receive_text()
    return e.value


def test_ws_refuses_with_1013_naming_the_switch_when_off(voice_app, no_models):
    c, _ = voice_app
    e = _close(c, f"/ws/voice/{THREAD}")
    assert e.code == 1013
    assert e.reason == "Voice is turned off in Settings › Voice"


def test_ws_refusal_names_the_server_when_forced_off(voice_app, no_models, monkeypatch):
    c, _ = voice_app
    monkeypatch.setenv("DISPATCH_VOICE", "0")
    e = _close(c, f"/ws/voice/{THREAD}")
    assert e.code == 1013 and "on this server" in e.reason


def test_ws_refuses_a_safe_bot_pre_accept(voice_app, no_models):
    """The gate is DisPatch's; the FakeContext mirrors it: 1008 before accept."""
    c, ctx = voice_app
    ctx.add_thread("t-safe", "alpha")
    assert _close(c, "/ws/voice/t-safe").code == 1008


def test_ws_works_after_switching_on(live_app, monkeypatch):
    c, ctx = live_app
    monkeypatch.delenv("DISPATCH_VOICE", raising=False)
    monkeypatch.setattr(routes, "apply_switch", _noop_apply)
    assert _close(c, f"/ws/voice/{THREAD}").code == 1013
    assert c.patch("/api/voice/settings", json={"enabled": True}).status_code == 200
    with c.websocket_connect(f"/ws/voice/{THREAD}") as ws:
        ws.send_json({"type": "start"})
        seen = set()
        for _ in range(20):
            m = ws.receive()
            if m.get("text"):
                seen.add(json.loads(m["text"]).get("type"))
            if "ready" in seen:
                break
        assert "ready" in seen


def test_spoken_turn_through_the_asgi_socket(live_app):
    c, ctx = live_app
    ctx.reply_for = lambda text: "Two meetings today. Want details?"
    with c.websocket_connect(f"/ws/voice/{THREAD}", headers={"origin": "http://testserver"}) as ws:
        assert json.loads(ws.receive_text())["type"] == "ready"
        ws.send_text(json.dumps({"type": "start"}))
        ws.send_bytes(b"\x00\x00" * 320)
        seen, audio = [], 0
        for _ in range(60):
            m = ws.receive()
            if m.get("bytes") is not None:
                audio += 1
                break
            seen.append(json.loads(m["text"]))
        types = [f["type"] for f in seen]
        assert "eot" in types and "transcript" in types and audio >= 1
        hdr = [f for f in seen if f["type"] == "audio" and f.get("kind") == "reply"]
        assert hdr and hdr[0]["text"].startswith("Two meetings today")
    rows = ctx.db().messages(THREAD)
    assert rows[0].role == "user" and rows[0].content == "what's the weather"
    assert rows[0].metadata == {"voice": True}
    for _ in range(100):                      # the server's finally runs after the client closed
        if "release" in ctx.power_calls:
            break
        time.sleep(0.02)
    assert ctx.power_calls == ["acquire", "release"]


def test_no_preload_while_off(host_env, no_models, monkeypatch):
    monkeypatch.setenv("DISPATCH_VOICE_PREWARM", "1")
    warmed = []
    monkeypatch.setattr(routes, "prewarm", lambda st: warmed.append(st.switched_on))
    routes.prewarm_start()
    assert warmed == [] and routes._APPLIED["on"] is False
    # Switched on: the start-up hook does warm (DisPatch runs it in a thread).
    write_yaml(host_env, "enabled: true\n")
    routes.prewarm_start()
    assert warmed == [True] and routes._APPLIED["on"] is True


def test_release_models_drops_every_cache(host_env, monkeypatch):
    from dispatch_voice.host import stt, tts, vad_turn
    stt._ENGINES["x"] = object()
    tts._ENGINES["x"] = object()
    vad_turn._SMART_TURN["x"] = object()
    routes._NOTICES[("x",)] = np.zeros(1)
    routes._PREVIEWS[("x",)] = b""
    routes._PREWARM.update(state="done", seconds=1.0)
    gen = routes._GEN[0]
    routes.release_models()
    assert not stt._ENGINES and not tts._ENGINES and not vad_turn._SMART_TURN
    assert not routes._NOTICES and not routes._PREVIEWS
    assert routes._PREWARM == {"state": "idle"}
    assert routes._GEN[0] == gen + 1


def test_a_prewarm_overtaken_by_a_release_gives_its_models_back(host_env, monkeypatch):
    from dispatch_voice.host import stt, tts, vad_turn

    class STT:
        def transcribe(self, a):
            routes.release_models()          # the switch went off mid-load
            stt._ENGINES["late"] = object()  # ...and this load finished after
            return ""

    class K:
        sample_rate = 24000

        def synth(self, text, profile_id=None):
            return np.zeros(4, "<i2"), 24000

    monkeypatch.setattr(VoiceSettings, "missing", lambda self: [])
    monkeypatch.setattr(stt, "shared_engine", lambda s: STT())
    monkeypatch.setattr(tts, "shared_engine", lambda s: K())
    monkeypatch.setattr(vad_turn, "_shared_smart_turn", lambda s: None)
    monkeypatch.setattr(vad_turn, "build_detector", lambda s: None)
    assert routes.prewarm(VoiceSettings(flag="1")) is False
    assert not stt._ENGINES
    assert routes._PREWARM["state"] == "idle"


def test_switching_off_ends_open_sessions_and_releases(host_env, monkeypatch):
    import asyncio
    ended, released = [], []

    class Sess:
        def end(self, code, reason=""):
            ended.append((code, reason))

    monkeypatch.setitem(ws_mod._SESSIONS, 1, Sess())
    monkeypatch.setattr(routes, "release_models", lambda: released.append(1))
    on = VoiceSettings(switch=True)
    off = VoiceSettings(switch=False)
    asyncio.run(routes.apply_switch(True, off))
    assert ended == [(1013, routes.OFF_SETTINGS)] and released == [1]
    # off → on starts the prewarm in the background (when voice can run).
    started = []
    monkeypatch.setattr(VoiceSettings, "missing", lambda self: [])
    monkeypatch.setattr(threading.Thread, "start", lambda self: started.append(self.name))
    asyncio.run(routes.apply_switch(False, on))
    assert started == ["voice-prewarm"]
    # No change, nothing happens.
    asyncio.run(routes.apply_switch(True, on))
    assert started == ["voice-prewarm"] and released == [1]


def test_patch_off_runs_the_live_switch(voice_app, no_models, monkeypatch):
    c, _ = voice_app
    seen = []

    async def spy(was_on, settings):
        seen.append((was_on, settings.switched_on))

    monkeypatch.setattr(routes, "apply_switch", spy)
    c.patch("/api/voice/settings", json={"enabled": True})
    c.patch("/api/voice/settings", json={"enabled": False})
    assert seen == [(False, True), (True, False)]


# --------------------------------------------------------------------------- #
# Settings keys: the Voice lab's and the v3 server settings
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("body", [
    {"merge_window_s": 0.2}, {"merge_window_s": 9}, {"merge_window_s": "long"},
    {"merge_window_s": True}, {"hard_silence_s": 0.5}, {"hard_silence_s": 3.5},
    {"hard_silence_s": None}, {"reset": ["speed"]}, {"reset": "merge_window_s"},
    {"reset": [1]}, {"patience": 2},
    {"max_sessions": 0}, {"max_sessions": 9}, {"max_sessions": "3"}, {"max_sessions": True},
    {"rt": "off"}, {"rt": {"power_hold": "turbo"}}, {"rt": {"bogus": 1}},
])
def test_settings_reject_bad_values(voice_app, body):
    c, ctx = voice_app
    assert c.patch("/api/voice/settings", json=body).status_code == 422
    assert voice_yaml(ctx) == {}


def test_lab_settings_persist_apply_and_reset(voice_app):
    c, ctx = voice_app
    r = c.patch("/api/voice/settings", json={"merge_window_s": 2.5, "hard_silence_s": 1.1,
                                             "voice_turn_thinking": "low"})
    assert r.status_code == 200, r.text
    assert (r.json()["merge_window_s"], r.json()["hard_silence_s"]) == (2.5, 1.1)
    st = vconfig.load_settings()                       # what the next session reads
    assert (st.merge_window_s, st.turn_hard_silence_s, st.voice_turn_thinking) == (2.5, 1.1, "low")
    assert voice_yaml(ctx) == {"merge_window_s": 2.5, "hard_silence_s": 1.1,
                               "voice_turn_thinking": "low"}
    r = c.patch("/api/voice/settings", json={"reset": ["merge_window_s", "hard_silence_s"]})
    assert (r.json()["merge_window_s"], r.json()["hard_silence_s"]) == (1.8, 1.4)
    assert voice_yaml(ctx) == {"voice_turn_thinking": "low"}


def test_max_sessions_and_power_hold_persist_and_apply(voice_app):
    c, ctx = voice_app
    r = c.patch("/api/voice/settings", json={"max_sessions": 5, "rt": {"power_hold": "off"}})
    assert r.status_code == 200, r.text
    assert r.json()["max_sessions"] == 5 and r.json()["rt"] == rt_settings_out(power_hold="off")
    assert voice_yaml(ctx) == {"max_sessions": 5, "rt": {"power_hold": "off"}}
    st = vconfig.load_settings()
    assert st.max_sessions == 5 and st.power_hold == "off"
    status = c.get("/api/voice/status").json()
    assert status["max_sessions"] == 5 and status["rt"] == rt_status_out("off")
    assert status["power_hold"] == "off"
    # The rt subtree merges key by key (a second PATCH never drops a sibling).
    assert c.patch("/api/voice/settings", json={"rt": {"power_hold": "performance"}}).json()["rt"] == rt_settings_out()
    assert c.patch("/api/voice/settings", json={"rt": {}}).status_code == 200


def test_lab_timings_reach_the_detector(host_env, monkeypatch):
    write_yaml(host_env, "hard_silence_s: 2.2\nmerge_window_s: 3.0\n")
    from dispatch_voice.host import vad_turn
    monkeypatch.setattr(vad_turn, "SileroVAD", lambda p, **kw: None)
    monkeypatch.setattr(vad_turn, "_shared_smart_turn", lambda s: None)
    st = vconfig.load_settings()
    det = vad_turn.build_detector(st)
    assert det.ep.cfg.turn_hard_silence_s == 2.2
    assert st.merge_window(1.0) == 3.0


# --------------------------------------------------------------------------- #
# Voice lab quick test
# --------------------------------------------------------------------------- #

@pytest.fixture
def lab_on(live_app, monkeypatch):
    c, ctx = live_app
    monkeypatch.delenv("DISPATCH_VOICE", raising=False)
    write_yaml(ctx, "enabled: true\n")
    return c


def test_lab_test_needs_voice_on(voice_app, no_models):
    c, _ = voice_app
    assert c.post("/api/voice/lab/test", content=b"\0" * 32000).status_code == 503


@pytest.mark.parametrize("size,code", [(100, 422), (32001, 422), (16000 * 2 * 10 + 2, 413)])
def test_lab_test_checks_the_clip(lab_on, size, code):
    assert lab_on.post("/api/voice/lab/test", content=b"\0" * size).status_code == code


def test_lab_test_transcribes_and_speaks_back_with_timings(lab_on, fake_engines):
    pcm = (np.sin(np.linspace(0, 400, 48000)) * 8000).astype("<i2").tobytes()
    r = lab_on.post("/api/voice/lab/test", content=pcm,
                    headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["transcript"] == fake_engines.stt.text and j["said"] == fake_engines.stt.text
    assert j["heard_ms"] == 3000
    for k in ("load_ms", "stt_ms", "tts_first_ms", "tts_total_ms", "spoken_ms"):
        assert isinstance(j[k], int) and j[k] >= 0
    wav = base64.b64decode(j["audio_wav_b64"])
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"


def test_lab_test_with_nothing_heard_says_so(lab_on, fake_engines):
    fake_engines.stt.text = ""
    j = lab_on.post("/api/voice/lab/test", content=b"\0" * 32000).json()
    assert j["transcript"] == "" and j["said"] == routes.LAB_TEST_FALLBACK


def test_lab_info_lists_the_power_hold_and_platform(lab_on):
    j = lab_on.get("/api/voice/lab").json()
    assert j["switch"] is True and j["timings"]["merge_window_s"]["min"] == 0.5
    assert j["timings"]["hard_silence_s"] == {"value": 1.4, "default": 1.4, "min": 0.8, "max": 3.0}
    assert "key" not in str(j["minimax"]).lower().replace("key_set", "")
    assert j["power_hold"] in ("off", "held", "unavailable")
    assert set(j["platform"]) == {"platform_profile", "max_mhz"}


def test_lab_routes_honour_a_tool_switch_when_the_host_offers_one(lab_on, live_app):
    c, ctx = live_app
    ctx.tool_enabled = lambda tid: False         # a DisPatch that exposes the switch
    assert c.get("/api/voice/lab").status_code == 404
    assert c.post("/api/voice/lab/test", content=b"\0" * 32000).status_code == 404
    ctx.tool_enabled = lambda tid: True
    assert c.get("/api/voice/lab").status_code == 200
    del ctx.tool_enabled                          # without it: open to operators
    assert c.get("/api/voice/lab").status_code == 200


# --------------------------------------------------------------------------- #
# Release, caches, the hand-edited switch, the provider description
# --------------------------------------------------------------------------- #

def test_release_gives_the_heap_back_and_rearms_stock_fillers(host_env, monkeypatch):
    """Measured on the reference box: without malloc_trim the process kept
    ~700 MB after a release and grew by a model set per off/on cycle; and the
    stock voice's filler render, queued before the release, loaded Chatterbox
    straight back in. Both are pinned here."""
    trimmed = []
    monkeypatch.setattr(routes, "_trim_heap", lambda: trimmed.append(1))
    profiles_bridge._EXTRAS_STARTED.add(str(profiles_bridge.voices_dir()))
    routes.release_models()
    assert trimmed == [1]
    assert str(profiles_bridge.voices_dir()) not in profiles_bridge._EXTRAS_STARTED


def test_trim_heap_never_raises():
    routes._trim_heap()


def test_voice_for_bot_is_the_resolved_profile_and_renders_nothing_itself(host_env, monkeypatch, fake_engines):
    """Acknowledgements are the render queue's (host/acks.py), never a second
    per-process set: looking a bot's voice up starts no render thread."""
    started = []
    monkeypatch.setattr(threading.Thread, "start", lambda self: started.append(self.name))
    monkeypatch.setattr(registry, "_profile_resolver", lambda bot_id: "stock-host")
    assert routes.voice_for_bot(VoiceSettings(), fake_engines.tts, None) == "stock-host"
    monkeypatch.setattr(registry, "_profile_resolver", lambda bot_id: "builtin:af_heart")
    assert routes.voice_for_bot(VoiceSettings(), fake_engines.tts, None) == "builtin:af_heart"
    assert started == []


def test_a_hand_edited_switch_applies_live(host_env, monkeypatch):
    """Review M4: editing `enabled` in voice.yaml must have the same live
    effect as the app's switch (chats end + models released / prewarm)."""
    import asyncio
    seen = []

    async def spy(was, settings):
        seen.append((was, settings.switched_on))

    monkeypatch.setattr(routes, "apply_switch", spy)
    write_yaml(host_env, "enabled: true\n")
    asyncio.run(routes.sync_switch())          # first look: baseline only
    assert seen == []
    write_yaml(host_env, "enabled: false\n")
    p = host_env.paths.voice_yaml
    os.utime(p, (time.time() + 5, time.time() + 5))
    asyncio.run(routes.sync_switch())
    asyncio.run(routes.sync_switch())          # no change, no second call
    assert seen == [(True, False)]


def test_the_status_route_starts_the_switch_watch(voice_app):
    c, _ = voice_app
    c.get("/api/voice/status")
    task = routes._WATCH.get("task")
    assert task is not None


def test_a_builtin_provider_describes_itself_and_releases(host_env, monkeypatch):
    """Review L6/L7: the Built-in voices tool shows the provider's own name,
    licence and steps (a permissive provider is never labelled GPL), and
    switching voice off reaches the plugin's own model cache."""
    st = routes._plugins(VoiceSettings())["builtin"]
    assert st["installed"] is False and st["name"] is None
    assert st["suggested"]["licence"] == "GPL-3.0-or-later"          # only a suggestion
    released = []
    registry.set_builtin_voices_provider(
        lambda s: [{"id": "builtin:x", "name": "X"}],
        info={"name": "acme-voices", "licence": "MIT", "bogus": 1,
              "models": [{"where": "plugin", "cmd": "acme fetch"}, {"where": "/etc", "cmd": "rm"}]})
    registry.add_release_hook(lambda: released.append(1))
    st = routes._plugins(VoiceSettings())["builtin"]
    assert (st["installed"], st["name"], st["licence"]) == (True, "acme-voices", "MIT")
    assert st["models"] == [{"where": "plugin", "cmd": "acme fetch"}]
    assert "bogus" not in registry.builtin_info()
    routes.release_models()
    assert released == [1]
    assert registry.ExtensionAPI(None, None).api_version == 3
    assert registry.ExtensionAPI().api_version == 3


def test_cache_tags_are_plain_names_and_old_ones_are_pruned(host_env):
    class Eng:
        def __init__(self, tag):
            self.tag = tag

        def cache_tag(self, pid):
            return self.tag

    for bad in ("../x", ".hidden", "a/b", "a b", "", "x" * 200):
        assert routes._cache_dir(None, Eng(bad), None) is None
    d = routes._cache_dir(None, Eng("kokoro-fp32-af_heart-1.050"), None)
    assert d.name == "kokoro-fp32-af_heart-1.050" and d.parent == host_env.paths.cache_dir
    root = host_env.paths.cache_dir
    for i in range(routes.CACHE_KEEP + 3):
        (root / f"t{i}").mkdir(parents=True)
        os.utime(root / f"t{i}", (i, i))
    routes._prune_cache(root)
    kept = sorted(p.name for p in root.iterdir())
    assert len(kept) == routes.CACHE_KEEP and "t0" not in kept and f"t{routes.CACHE_KEEP + 2}" in kept
