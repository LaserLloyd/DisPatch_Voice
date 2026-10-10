"""The licence split: built-in voices are a plugin, never part of the add-on.

The add-on must work, and say clearly what is missing, with NO built-in voice
provider installed; and a plugin that registers an engine with
``role="fallback"`` plus a built-in voice provider through
DISPATCH_VOICE_EXTENSIONS must get back exactly the 2.1 behaviour (the
built-in voice behind the cloned one, its list, its preview). The fake
plugin below stands in for dispatch-voice-kokoro.
"""
from __future__ import annotations

import io
import itertools
import sys
import types
import wave

import numpy as np
import pytest
import yaml
from fakes import BotView, FakeContext
from fastapi.testclient import TestClient

from dispatch_voice.host import config as vconfig
from dispatch_voice.host import context, registry, routes, setup, tts
from dispatch_voice.host.config import VoiceSettings
from dispatch_voice.host.engines import chatterbox

FAKE_VOICES = [{"id": "builtin:af_heart", "name": "Heart (US, female)", "lang": "en-us", "default": True},
               {"id": "builtin:am_michael", "name": "Michael (US, male)", "lang": "en-us"}]
_n = itertools.count()



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

@pytest.fixture(autouse=True)
def _clean(host_env, monkeypatch):
    monkeypatch.setattr(chatterbox, "available", lambda: False)      # no Chatterbox models here
    yield
    for name in [m for m in sys.modules if m.startswith("fake_builtin_ext")]:
        del sys.modules[name]


class Eng:
    def __init__(self, name, rate=24000):
        self.name, self.sample_rate = name, rate
        self.sample_format, self.channels, self.streaming = "pcm_s16le", 1, False
        self.has_fallback = False
        self.calls = []
        self.applied = []

    def stream(self, text, profile_id=None, stop=None):
        self.calls.append(profile_id)
        if self.name == "primary" and (profile_id is None or tts.builtin_id(profile_id)):
            raise LookupError("not mine")
        yield np.full(4, 1 if self.name == "primary" else 2, "<i2")

    def synth(self, text, profile_id=None):
        parts = list(self.stream(text, profile_id))
        return np.concatenate(parts), self.sample_rate

    def apply_settings(self, settings):
        self.applied.append((settings.builtin_voice, settings.speed))


def fake_plugin(engine, previews):
    """What dispatch-voice-kokoro's setup(api) does."""
    def setup_plugin(api):
        assert api.api_version >= 2
        api.register_engine("fakebuiltin", lambda s: engine, role="fallback",
                            missing=lambda s: [], sample_rate=24000)

        def preview(settings, voice, text):
            previews.append((voice, text, settings.speed))
            return np.full(240, 3, "<i2"), 24000

        api.set_builtin_voices_provider(lambda s: FAKE_VOICES, preview=preview)

    mod = types.ModuleType("fake_builtin_ext")
    mod.setup = setup_plugin
    sys.modules["fake_builtin_ext"] = mod


def mounted(tmp_path, monkeypatch, spec: str = "") -> tuple[TestClient, FakeContext]:
    """setup(ctx) on a fresh FakeContext with DISPATCH_VOICE_EXTENSIONS=spec."""
    ctx = FakeContext(tmp_path / f"ctx{next(_n)}")
    ctx.add_thread("t-voice-1", "sage")
    context.set_context(ctx)
    registry.reset()
    routes._LOADED[:] = []
    monkeypatch.setenv("DISPATCH_VOICE_EXTENSIONS", spec)
    setup(ctx)
    return TestClient(ctx.app), ctx


# --------------------------------------------------------------------------- #
# No built-in voice provider installed
# --------------------------------------------------------------------------- #

def test_no_provider_means_no_builtin_voices_and_a_clear_missing_line(tmp_path, monkeypatch):
    c, _ = mounted(tmp_path, monkeypatch)
    st = c.get("/api/voice/status").json()
    assert st["builtin_voices"] == [] and st["default_voice"] == ""
    assert st["profiles"] == []
    # Nothing can speak: the line points at fetch-models (the pane's `models` step),
    # never at a "python package" (the pane would read that as "voice is off").
    assert any(m.startswith("dispatch-voice models") and "fetch-models" in m for m in st["missing"])
    assert not any(m.startswith("python package") for m in st["missing"])
    assert "kokoro" not in " ".join(k for k in st if k != "missing")


def test_no_provider_rejects_builtin_voices_without_crashing(tmp_path, monkeypatch):
    c, ctx = mounted(tmp_path, monkeypatch)
    assert c.patch("/api/voice/settings", json={"default_voice": "builtin:af_heart"}).status_code == 422
    assert c.get("/api/voice/preview", params={"voice": "builtin:af_heart"}).status_code == 422
    # A saved 2.1 choice degrades: the setting is kept, but nothing reports it.
    ctx.paths.voice_yaml.write_text("kokoro_voice: am_michael\nkokoro_speed: 1.2\n")
    r = c.get("/api/voice/settings").json()
    assert r == {"enabled": False, "forced": None, "default_voice": "", "speed": 1.2,
                 "voice_turn_thinking": "off", "merge_window_s": 1.8, "hard_silence_s": 1.4,
                 "max_sessions": 3, "max_sessions_live": 3, "restart_required": False,
                 "rt": rt_settings_out(), "rp_speech": "all", "narrator_voice": "", "expressive_tags": True,
                     "gateway_trailer": "full", "opus": "auto", "opus_down_kbps": 24, "opus_up_kbps": 16,
                     "bots": {"nova": {"backchannel": False, "follow_up": True},
                              "sage": {"backchannel": False, "follow_up": False},
                              "swift": {"backchannel": False, "follow_up": True}}}


def test_no_provider_bot_builtin_voice_resolves_to_the_default(host_env):
    host_env.bots._bots["sage"] = BotView("sage", "Sage", voice="builtin:am_michael", backend="advisor")
    assert registry.resolve_profile("sage") is None
    registry.set_builtin_voices_provider(lambda s: FAKE_VOICES)
    assert registry.resolve_profile("sage") == "builtin:am_michael"


def test_named_engine_that_is_not_installed_says_where_it_comes_from():
    miss = VoiceSettings(tts_engine="kokoro").missing()
    assert any("not installed" in m and "dispatch-voice-kokoro" in m for m in miss)
    with pytest.raises(RuntimeError, match="unknown TTS engine"):
        tts.build_engine(VoiceSettings(tts_engine="kokoro"))


def test_auto_with_nothing_installed_is_a_clear_error_not_a_crash():
    with pytest.raises(RuntimeError, match="no TTS engine"):
        tts.build_engine(VoiceSettings(tts_engine="auto"))


def test_chatterbox_named_but_not_fetched_says_fetch_models():
    miss = VoiceSettings(tts_engine="chatterbox").missing()
    assert miss[-1] == "dispatch-voice models (run `dispatch-voice fetch-models`)"


# --------------------------------------------------------------------------- #
# Engine roles
# --------------------------------------------------------------------------- #

def test_fallback_engine_goes_behind_the_primary():
    p, f = Eng("primary"), Eng("fallback")
    registry.register_engine("prim", lambda s: p)
    registry.register_engine("fb", lambda s: f, role="fallback", sample_rate=24000)
    eng = tts.build_engine(VoiceSettings(tts_engine="auto"))
    assert isinstance(eng, tts.FallbackTTS) and eng.primary is p and eng.fallback is f
    assert p.has_fallback is True
    assert [c.tolist() for c in eng.stream("Hi.", "a-profile")] == [[1] * 4]
    assert [c.tolist() for c in eng.stream("Hi.", "builtin:af_heart")] == [[2] * 4]
    assert eng._down_until == 0.0


def test_fallback_alone_when_it_is_the_only_engine():
    f = Eng("fallback")
    registry.register_engine("fb", lambda s: f, role="fallback")
    assert tts.build_engine(VoiceSettings(tts_engine="auto")) is f


def test_fallback_with_another_sample_rate_is_not_paired():
    p, f = Eng("primary", 22050), Eng("fallback", 24000)
    registry.register_engine("prim", lambda s: p)
    registry.register_engine("fb", lambda s: f, role="fallback", sample_rate=24000)
    assert tts.build_engine(VoiceSettings(tts_engine="prim")) is p


def test_unready_fallback_is_skipped_and_reported_only_when_chosen():
    p = Eng("primary")
    registry.register_engine("prim", lambda s: p)
    registry.register_engine("fb", lambda s: Eng("fallback"), role="fallback",
                             missing=lambda s: ["model fb-1"])
    assert tts.build_engine(VoiceSettings(tts_engine="prim")) is p
    assert VoiceSettings(tts_engine="fb").missing()[-1] == "model fb-1"
    assert not any("fb-1" in m for m in VoiceSettings(tts_engine="auto").missing())


def test_bad_role_is_refused():
    with pytest.raises(ValueError):
        registry.register_engine("x", lambda s: None, role="backup")


# --------------------------------------------------------------------------- #
# Provider installed through DISPATCH_VOICE_EXTENSIONS (the plugin path)
# --------------------------------------------------------------------------- #

def test_plugin_restores_the_builtin_voice_list_default_and_preview(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPATCH_VOICE", "1")
    monkeypatch.setattr(VoiceSettings, "missing", lambda self: [])      # no speech models here
    fb, previews = Eng("fallback"), []
    fake_plugin(fb, previews)
    c, _ = mounted(tmp_path, monkeypatch, "fake_builtin_ext:setup")
    st = c.get("/api/voice/status").json()
    assert st["extensions"] == ["fake_builtin_ext:setup"]
    assert st["builtin_voices"] == FAKE_VOICES
    assert st["default_voice"] == "builtin:af_heart"          # the provider's default
    assert st["plugins"]["builtin"]["installed"] is True
    r = c.get("/api/voice/preview", params={"voice": "builtin:am_michael"})
    assert r.status_code == 200
    with wave.open(io.BytesIO(r.content)) as w:
        assert w.getframerate() == 24000 and w.getnframes() == 240
    assert previews == [("builtin:am_michael", routes.PREVIEW_TEXT, 1.05)]


def test_plugin_choice_is_saved_with_the_new_names_and_applied_live(tmp_path, monkeypatch):
    fb, previews = Eng("fallback"), []
    fake_plugin(fb, previews)
    c, ctx = mounted(tmp_path, monkeypatch, "fake_builtin_ext:setup")
    tts.shared_engine(vconfig.load_settings())                 # a loaded engine to update
    r = c.patch("/api/voice/settings", json={"default_voice": "builtin:am_michael", "speed": 0.9})
    assert r.status_code == 200, r.text
    assert r.json()["default_voice"] == "builtin:am_michael"
    assert yaml.safe_load(ctx.paths.voice_yaml.read_text()) == {"builtin_voice": "am_michael", "speed": 0.9}
    assert fb.applied[-1] == ("am_michael", 0.9)


def test_saved_builtin_voice_the_provider_no_longer_lists_falls_back_to_its_default(tmp_path, monkeypatch):
    fake_plugin(Eng("fallback"), [])
    c, ctx = mounted(tmp_path, monkeypatch, "fake_builtin_ext:setup")
    ctx.paths.voice_yaml.write_text("builtin_voice: zz_gone\n")
    assert c.get("/api/voice/settings").json()["default_voice"] == "builtin:af_heart"


def test_a_failing_provider_is_an_empty_list_not_a_500(tmp_path, monkeypatch):
    def boom(settings):
        raise RuntimeError("voices file corrupt")

    c, _ = mounted(tmp_path, monkeypatch)
    registry.set_builtin_voices_provider(boom)
    st = c.get("/api/voice/status")
    assert st.status_code == 200 and st.json()["builtin_voices"] == []


# --------------------------------------------------------------------------- #
# Setting names: generic, with the 2.1 names as aliases
# --------------------------------------------------------------------------- #

def test_env_aliases(monkeypatch):
    monkeypatch.setenv("DISPATCH_VOICE_KOKORO_VOICE", "bf_emma")
    monkeypatch.setenv("DISPATCH_VOICE_KOKORO_SPEED", "1.15")
    s = vconfig.load_settings()
    assert (s.builtin_voice, s.speed) == ("bf_emma", 1.15)
    assert (s.kokoro_voice, s.kokoro_speed) == ("bf_emma", 1.15)       # read-only aliases
    monkeypatch.setenv("DISPATCH_VOICE_BUILTIN_VOICE", "am_adam")
    monkeypatch.setenv("DISPATCH_VOICE_SPEED", "0.95")
    s = vconfig.load_settings()
    assert (s.builtin_voice, s.speed) == ("am_adam", 0.95)               # new names win


def test_voice_yaml_aliases_and_new_names_win():
    assert vconfig.clean_prefs({"kokoro_voice": "af_bella", "kokoro_speed": 1.1}) == {
        "builtin_voice": "af_bella", "speed": 1.1}
    assert vconfig.clean_prefs({"kokoro_voice": "af_bella", "builtin_voice": "am_adam",
                                "kokoro_speed": 1.1, "speed": 0.9}) == {
        "builtin_voice": "am_adam", "speed": 0.9}
    assert vconfig.clean_prefs({"kokoro_speed": 5}) == {}


def test_saving_migrates_a_2_1_voice_yaml(host_env):
    p = host_env.paths.voice_yaml
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("kokoro_voice: af_bella\nkokoro_speed: 1.3\n")
    assert vconfig.load_settings().speed == 1.3
    vconfig.save_prefs({"voice_turn_thinking": "low"})
    assert yaml.safe_load(p.read_text()) == {"builtin_voice": "af_bella", "speed": 1.3,
                                             "voice_turn_thinking": "low"}


def test_no_kokoro_or_dispatch_import_anywhere_in_the_add_on():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "dispatch_voice"
    hits = []
    for f in root.rglob("*.py"):
        text = f.read_text()
        for needle in ("import kokoro", "from kokoro", "from app import", "from app.", "import app.", "import app\n"):
            if needle in text:
                hits.append((str(f.relative_to(root)), needle))
    assert hits == []
