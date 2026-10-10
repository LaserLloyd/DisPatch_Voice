"""The settings/preview/status routes, the per-bot voice validator and the
engine-side rules (opt-in disk cache, M2, L6, H5), ported from DisPatch's
test_voice_drive_v2_app.py onto the FakeContext. The abort hook and the
gateway seat belong to DisPatch (T1/T2) and stay there."""
from __future__ import annotations

import io
import threading
import time
import wave
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from fakes import BotView

from dispatch_voice.host import config as vconfig
from dispatch_voice.host import registry, routes
from dispatch_voice.host import tts as vtts
from dispatch_voice.host import ws as ws_mod

BUILTINS = [{"id": "builtin:af_heart", "name": "Heart (US, female)", "lang": "en-us"},
            {"id": "builtin:am_michael", "name": "Michael (US, male)", "lang": "en-us"}]
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

@pytest.fixture
def app_voice(voice_app, no_models, monkeypatch, fake_engines):
    c, ctx = voice_app
    monkeypatch.setenv("DISPATCH_VOICE", "1")
    monkeypatch.setattr(registry, "_builtin_list", lambda settings: BUILTINS)
    eng = fake_engines.tts
    eng.calls = []
    orig_synth = eng.synth

    def synth(text, voice=None):
        eng.calls.append((text, voice))
        return orig_synth(text, voice)

    eng.synth = synth
    monkeypatch.setattr(vtts, "shared_engine", lambda settings: eng)
    routes._PREVIEWS.clear()
    return c, ctx, eng


# --------------------------------------------------------------------------- #
# /api/voice/settings
# --------------------------------------------------------------------------- #

def test_settings_round_trip_persist_and_apply_live(app_voice):
    c, ctx, _ = app_voice
    r = c.get("/api/voice/settings")
    assert r.status_code == 200
    assert r.json() == {"enabled": True, "forced": "on",
                        "default_voice": "builtin:af_heart", "speed": 1.05,
                        "voice_turn_thinking": "off", "merge_window_s": 1.8,
                        "hard_silence_s": 1.4, "max_sessions": 3, "max_sessions_live": 3,
                        "restart_required": False, "rt": rt_settings_out(), "rp_speech": "all", "narrator_voice": "", "expressive_tags": True,
                     "gateway_trailer": "full", "opus": "auto", "opus_down_kbps": 24, "opus_up_kbps": 16,
                     "bots": {"nova": {"backchannel": False, "follow_up": True},
                              "sage": {"backchannel": False, "follow_up": False},
                              "swift": {"backchannel": False, "follow_up": True}}}
    r = c.patch("/api/voice/settings", json={"speed": 1.2, "voice_turn_thinking": "low",
                                             "default_voice": "builtin:am_michael"})
    assert r.status_code == 200, r.text
    assert r.json() == {"enabled": True, "forced": "on",
                        "default_voice": "builtin:am_michael", "speed": 1.2,
                        "voice_turn_thinking": "low", "merge_window_s": 1.8,
                        "hard_silence_s": 1.4, "max_sessions": 3, "max_sessions_live": 3,
                        "restart_required": False, "rt": rt_settings_out(), "rp_speech": "all", "narrator_voice": "", "expressive_tags": True,
                     "gateway_trailer": "full", "opus": "auto", "opus_down_kbps": 24, "opus_up_kbps": 16,
                     "bots": {"nova": {"backchannel": False, "follow_up": True},
                              "sage": {"backchannel": False, "follow_up": False},
                              "swift": {"backchannel": False, "follow_up": True}}}
    f = ctx.paths.voice_yaml
    assert f.exists() and (f.stat().st_mode & 0o777) == 0o600
    st = vconfig.load_settings()                       # what the next turn reads
    assert (st.builtin_voice, st.speed, st.voice_turn_thinking) == ("am_michael", 1.2, "low")
    assert yaml.safe_load(f.read_text()) == {"builtin_voice": "am_michael", "speed": 1.2,
                                             "voice_turn_thinking": "low"}
    assert c.get("/api/voice/settings").json()["speed"] == 1.2


def test_changing_max_sessions_is_saved_but_says_restart_required(app_voice):
    """The gate's `max_total` is fixed at setup (DisPatch seals the context):
    a new value persists, the live gate keeps the old one, and both responses
    say so, so the Settings pane can show "takes effect after a restart"."""
    c, ctx, _ = app_voice
    r = c.patch("/api/voice/settings", json={"max_sessions": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["max_sessions"] == 5 and body["max_sessions_live"] == 3 and body["restart_required"] is True
    assert ctx.socket_count("voice") == (0, 3)          # the gate did not change
    assert c.get("/api/voice/status").json()["sessions"] == {"open": 0, "max": 3}
    assert c.get("/api/voice/settings").json()["restart_required"] is True
    r = c.patch("/api/voice/settings", json={"max_sessions": 3})
    assert r.json()["restart_required"] is False


async def test_settings_reach_the_next_voice_turn(app_voice):
    c, ctx, _ = app_voice
    assert c.patch("/api/voice/settings", json={"voice_turn_thinking": "medium"}).status_code == 200
    host = ws_mod.build_host(ctx, ctx.thread_socket(THREAD), bot_id="sage")
    await host.submit(THREAD, "hello there", {"reply_style": "short", "follow_up": False})
    (_, kw), = ctx.calls_to("submit_turn")
    assert kw["thinking"] == "medium"
    assert "two short sentences" in kw["instructions"].system
    assert "Do not end with a question" in kw["instructions"].system
    assert kw["instructions"].trailer == ""          # sage is the advisor: system slot only


@pytest.mark.parametrize("body", [
    {"speed": 2.0}, {"speed": "fast"}, {"speed": True}, {"voice_turn_thinking": "high"},
    {"default_voice": "builtin:nope"}, {"default_voice": "no-such-profile"},
    {"default_voice": ""}, {"volume": 3},
])
def test_settings_reject_bad_values(app_voice, body):
    c, ctx, _ = app_voice
    assert c.patch("/api/voice/settings", json=body).status_code == 422
    assert not ctx.paths.voice_yaml.exists()


def test_settings_and_preview_are_unlocked_only(app_voice):
    c, ctx, _ = app_voice
    ctx.operator = False
    assert c.get("/api/voice/settings").status_code == 403
    assert c.patch("/api/voice/settings", json={"speed": 1.0}).status_code == 403
    assert c.get("/api/voice/preview", params={"voice": "builtin:af_heart"}).status_code == 403


# --------------------------------------------------------------------------- #
# /api/voice/preview and /api/voice/status
# --------------------------------------------------------------------------- #

def test_preview_returns_a_wav_and_caches_it(app_voice):
    c, _, eng = app_voice
    r = c.get("/api/voice/preview", params={"voice": "builtin:am_michael"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(r.content)) as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1
    assert eng.calls == [(routes.PREVIEW_TEXT, "builtin:am_michael")]
    assert "voice chat" in routes.PREVIEW_TEXT and "Drive" not in routes.PREVIEW_TEXT
    c.get("/api/voice/preview", params={"voice": "builtin:am_michael"})
    assert len(eng.calls) == 1                         # cached per voice + speed


def test_preview_validates_voice_and_text(app_voice):
    c, _, _ = app_voice
    assert c.get("/api/voice/preview", params={"voice": "builtin:nope"}).status_code == 422
    assert c.get("/api/voice/preview", params={"voice": ""}).status_code == 422
    assert c.get("/api/voice/preview", params={"voice": "builtin:af_heart",
                                               "text": "x" * 121}).status_code == 422
    assert c.get("/api/voice/preview", params={"voice": "builtin:af_heart",
                                               "text": "Short line."}).status_code == 200


def test_status_lists_what_settings_need(app_voice):
    c, _, _ = app_voice
    st = c.get("/api/voice/status").json()
    for k in ("default_voice", "builtin_voices", "profiles", "speed", "voice_turn_thinking",
              "ready", "missing", "enabled"):
        assert k in st, k
    assert st["builtin_voices"] == BUILTINS and st["default_voice"] == "builtin:af_heart"


# --------------------------------------------------------------------------- #
# Per-bot voice: the validator DisPatch calls on the Bots save
# --------------------------------------------------------------------------- #

def test_bot_voice_validator(app_voice):
    c, ctx, _ = app_voice
    ctx.validate_bot_field("voice", "builtin:am_michael")
    ctx.validate_bot_field("voice", "")                       # back to the default voice
    with pytest.raises(ValueError, match="Unknown voice: who-knows"):
        ctx.validate_bot_field("voice", "who-knows")
    with pytest.raises(ValueError):
        ctx.validate_bot_field("voice", "x" * 81)


def test_a_bots_builtin_voice_resolves_without_a_profile_store(app_voice, monkeypatch):
    c, ctx, _ = app_voice
    ctx.bots._bots["sage"] = BotView("sage", "Sage", voice="builtin:am_michael", backend="advisor")
    assert registry._profile_resolver is None               # no Chatterbox models mounted
    assert registry.resolve_profile("sage") == "builtin:am_michael"
    assert registry.resolve_profile("swift") is None            # no voice saved
    monkeypatch.setattr(registry, "_builtin_list", None)
    assert registry.resolve_profile("sage") is None         # provider gone: default voice


# --------------------------------------------------------------------------- #
# Engines: downloader, background renders yield (M2), opt-in cache, L6, H5
# --------------------------------------------------------------------------- #

def test_downloader_no_longer_fetches_kokoro():
    from dispatch_voice.host import download
    names = [rel for rel, _, _ in download.files_for()]
    assert names and not any("kokoro" in n for n in names)
    assert {"silero_vad.onnx", "smart-turn-v3.2-cpu.onnx"} <= set(names)


def test_background_renders_wait_for_reply_synthesis():
    act = vtts._Activity()
    act.enter()
    done = threading.Event()

    def bg():
        act.wait_idle(quiet_s=0.05)
        done.set()

    threading.Thread(target=bg, daemon=True).start()
    time.sleep(0.15)
    assert not done.is_set()                 # a reply is being synthesised
    act.exit()
    assert done.wait(1.0)


class CachedEngine:
    """An engine that opts in to the disk cache, like the Kokoro plugin."""
    sample_rate = 24000

    def __init__(self):
        self.n = 0

    def cache_tag(self, profile_id):
        if profile_id and not vtts.builtin_id(profile_id):
            return None
        return f"fake-{vtts.builtin_id(profile_id) or 'default'}-1.050"

    def synth(self, text, profile_id=None):
        self.n += 1
        return np.full(10, 7, "<i2"), 24000


def test_opt_in_engine_renders_are_cached_on_disk(host_env):
    k = CachedEngine()
    st = vconfig.VoiceSettings(flag="1")
    a = routes._render_cached(st, k, None, "Got it.", wait_idle=False)
    b = routes._render_cached(st, k, None, "Got it.", wait_idle=False)
    assert k.n == 1 and list(a) == list(b) == [7] * 10
    d = routes._cache_dir(st, k, None)
    assert d.name == "fake-default-1.050"
    assert d.is_relative_to(host_env.paths.cache_dir) and list(d.glob("*.npy"))
    # Found through a FallbackTTS chain too; a cloned profile is never cached.
    chain = vtts.FallbackTTS(SimpleNamespace(sample_rate=24000), k)
    assert routes._cache_dir(st, chain, "builtin:am_michael").name == "fake-am_michael-1.050"
    assert routes._cache_dir(st, chain, "some-cloned-profile") is None


def test_engines_without_cache_tag_are_not_cached(host_env):
    class Plain:
        sample_rate = 24000
        n = 0

        def synth(self, text, profile_id=None):
            Plain.n += 1
            return np.zeros(4, "<i2"), 24000

    st = vconfig.VoiceSettings(flag="1")
    assert routes._cache_dir(st, Plain(), None) is None
    routes._render_cached(st, Plain(), None, "Got it.", wait_idle=False)
    routes._render_cached(st, Plain(), None, "Got it.", wait_idle=False)
    assert Plain.n == 2 and not host_env.paths.cache_dir.exists()


def test_fallback_with_builtin_or_unknown_voice_has_no_cooldown():
    """L6: a voice the cloned engine does not know is not an engine failure."""
    class Primary:
        sample_rate = 24000
        streaming = True

        def stream(self, text, profile_id=None, stop=None):
            raise LookupError("not mine")
            yield  # pragma: no cover

    class Kok:
        sample_rate = 24000

        def stream(self, text, profile_id=None, stop=None):
            yield np.ones(3, "<i2")

    f = vtts.FallbackTTS(Primary(), Kok(), cooldown_s=30.0)
    assert len(list(f.stream("Hi.", "builtin:am_michael"))) == 1
    assert f._down_until == 0.0


def test_prewarm_is_a_no_op_when_voice_is_off(host_env):
    assert routes.prewarm(vconfig.VoiceSettings(flag="0")) is False
    assert routes.prewarm(vconfig.VoiceSettings(flag="1", prewarm=False)) is False


def test_prewarm_loads_engines_and_runs_a_warm_synthesis(host_env, monkeypatch):
    calls = []

    class STT:
        def transcribe(self, a):
            calls.append("stt")
            return ""

    class K:
        sample_rate = 24000

        def warm(self):
            self.synth("Okay.")

        def synth(self, text, profile_id=None):
            calls.append(("tts", text))
            return np.zeros(4, "<i2"), 24000

    from dispatch_voice.host import stt as vstt
    from dispatch_voice.host import vad_turn
    monkeypatch.setattr(vconfig.VoiceSettings, "missing", lambda self: [])
    monkeypatch.setattr(vstt, "shared_engine", lambda s: STT())
    monkeypatch.setattr(vtts, "shared_engine", lambda s: K())
    monkeypatch.setattr(vad_turn, "_shared_smart_turn", lambda s: None)
    monkeypatch.setattr(vad_turn, "build_detector", lambda s: None)
    monkeypatch.setattr(routes, "voice_for_bot", lambda s, e, b: None)
    assert routes.prewarm(vconfig.VoiceSettings(flag="1")) is True
    assert "stt" in calls and ("tts", "Okay.") in calls


def test_profile_to_warm_reads_the_bots_through_the_context(host_env, monkeypatch):
    """The bot roster comes from ctx.bots (never app.config); safe bots are skipped."""
    host_env.bots._bots["alpha"] = BotView("alpha", "Alpha", safe=True, voice="secret-voice", backend="gateway")
    host_env.bots._bots["nova"] = BotView("nova", "Nova", voice="nova-voice", backend="gateway")
    monkeypatch.setattr(registry, "_profile_resolver",
                        lambda bot_id: {"nova": "nova-voice", "alpha": "secret-voice"}.get(bot_id))
    assert routes._profile_to_warm(registry, vtts) == "nova-voice"


# --------------------------------------------------------------------------- #
# Slice 2: role-play / narrator / tags / trailer / per-bot settings, acks status
# --------------------------------------------------------------------------- #

def test_slice2_settings_round_trip_and_validation(app_voice):
    c, ctx, _ = app_voice
    r = c.patch("/api/voice/settings", json={"rp_speech": "dialogue_narrator", "narrator_voice": "builtin:af_heart",
                                             "expressive_tags": False, "gateway_trailer": "tag",
                                             "bots": {"nova": {"backchannel": True}}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["rp_speech"], body["narrator_voice"], body["expressive_tags"], body["gateway_trailer"]) == \
        ("dialogue_narrator", "builtin:af_heart", False, "tag")
    assert body["bots"]["nova"] == {"backchannel": True, "follow_up": True}
    assert body["bots"]["sage"] == {"backchannel": False, "follow_up": False}     # the advisor's default
    # A second PATCH merges per bot per key: nova keeps backchannel, sage gains follow_up.
    r = c.patch("/api/voice/settings", json={"bots": {"sage": {"follow_up": True}, "nova": {"follow_up": False}}})
    assert r.status_code == 200
    assert r.json()["bots"]["nova"] == {"backchannel": True, "follow_up": False}
    assert r.json()["bots"]["sage"] == {"backchannel": False, "follow_up": True}
    st = vconfig.load_settings()
    assert st.rp_speech == "dialogue_narrator" and st.narrator_voice == "builtin:af_heart"
    assert st.bots == {"nova": {"backchannel": True, "follow_up": False}, "sage": {"follow_up": True}}
    assert st.bot("nova", backend="gateway") == {"backchannel": True, "follow_up": False}
    # Clearing the narrator voice.
    assert c.patch("/api/voice/settings", json={"narrator_voice": ""}).json()["narrator_voice"] == ""
    status = c.get("/api/voice/status").json()
    assert status["rp_speech"] == "dialogue_narrator" and status["narrator_voice"] == ""
    assert status["expressive_tags"] is False and status["bots"]["nova"]["backchannel"] is True
    assert "alpha" not in status["bots"] and isinstance(status["acks"], dict)


@pytest.mark.parametrize("body", [
    {"rp_speech": "quotes"}, {"rp_speech": 3}, {"narrator_voice": "builtin:nope"}, {"narrator_voice": "no such"},
    {"expressive_tags": "yes"}, {"gateway_trailer": "short"}, {"bots": []}, {"bots": {"ghost": {"follow_up": True}}},
    {"bots": {"alpha": {"backchannel": True}}}, {"bots": {"nova": {"flavour": "wry"}}},
    {"bots": {"nova": {"follow_up": "on"}}}, {"bots": {"": {"follow_up": True}}},
])
def test_slice2_settings_reject_bad_values(app_voice, body):
    c, _, _ = app_voice
    r = c.patch("/api/voice/settings", json=body)
    assert r.status_code == 422, (body, r.text)


def test_acks_rebuild_route(app_voice, monkeypatch):
    from dispatch_voice.host import acks as hacks
    c, _, _ = app_voice
    assert c.post("/api/voice/acks/rebuild", params={"voice": "ghost"}).status_code == 404
    assert c.post("/api/voice/acks/rebuild", params={"voice": "bad id!"}).status_code == 422
    monkeypatch.setattr(hacks, "rebuild", lambda voice: voice == "host")
    monkeypatch.setattr(hacks, "status", lambda: {"host": {"ready": 0, "total": 5, "failed": 0}})
    r = c.post("/api/voice/acks/rebuild", params={"voice": "host"})
    assert r.status_code == 200 and r.json()["acks"]["host"]["total"] == 5
    ctx_off = c.app
    assert ctx_off is not None
