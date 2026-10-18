"""Cloned voices in the host: the profile store, the ``chatterbox`` engine, the
resolver, the fillers, the stock voice and the ``/api/voice/profiles`` router
(the two halves ``host/dv.py`` dissolved into). The Chatterbox engine is a
fake, so no model is loaded; the real engine's own tests are tests/test_models.py."""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import BotView
from fastapi.testclient import TestClient

from dispatch_voice.host import profiles_bridge as pb
from dispatch_voice.host import registry, routes, setup, tts
from dispatch_voice.host.config import VoiceSettings
from dispatch_voice.host.engines import chatterbox


class FakeChatterbox:
    sample_rate = 24000
    dtype = "fake"
    watermarking = False

    def __init__(self):
        self.calls = []

    def stream(self, text_chunks, voice, stop=None):
        self.calls.append(("".join(text_chunks), voice))
        yield np.zeros(2400, np.float32)            # 0.1 s -> 3 frames of 40 ms...
        yield np.zeros(2400, np.float32)


@pytest.fixture
def dv_env(host_env, monkeypatch):
    fake = FakeChatterbox()
    monkeypatch.setattr(chatterbox.LazyEngine, "_get", lambda self: fake)
    monkeypatch.setattr(chatterbox, "available", lambda: True)
    # Hermetic: never see a stock voice cached in the real models dir.
    monkeypatch.setenv("DISPATCH_VOICE_MODELS", str(host_env.paths.models_dir / "dv-models"))
    return SimpleNamespace(ctx=host_env, engine=fake)


def make_profile(st, pid: str, default: bool = False) -> None:
    """A profile directory as the store writes it, minus the clip."""
    d = st.root / pid
    d.mkdir(parents=True)
    from dispatch_voice.engine import VoiceTensors
    VoiceTensors(np.zeros((1, 2, 4), np.float32), np.zeros((1, 3), np.int64),
                 np.zeros((1, 192), np.float32), np.zeros((1, 5, 80), np.float32)).save(d / "voice.npz")
    (d / "profile.json").write_text(json.dumps({"id": pid, "name": pid, "default": default}), encoding="utf-8")
    (d / "fillers").mkdir()
    import soundfile as sf
    sf.write(d / "fillers" / "0.wav", np.full(480, 0.1, np.float32), 24000)


def set_voice(ctx, bot_id: str, voice: str) -> None:
    ctx.bots._bots[bot_id] = BotView(bot_id, bot_id.title(), voice=voice, backend="gateway")


@pytest.fixture
def client(dv_env):
    setup(dv_env.ctx)
    assert routes._LOADED == ["dispatch-voice"]
    return TestClient(dv_env.ctx.app)


@pytest.mark.parametrize("path", ["/api/voice/profiles", "/api/voice/profiles/consent-kinds"])
def test_profiles_api_is_unlocked_only(client, dv_env, path):
    dv_env.ctx.operator = False
    assert client.get(path).status_code == 403
    dv_env.ctx.operator = True
    assert client.get(path).status_code == 200


def test_no_page_under_api_voice(client):
    assert client.get("/api/voice/profiles/ui").status_code == 404
    assert client.get("/api/voice/profiles/ui/voices.js").status_code == 404


def test_profiles_writes_are_unlocked_only(client, dv_env):
    dv_env.ctx.operator = False
    assert client.post("/api/voice/profiles/analyse",
                       files={"file": ("a.wav", b"RIFF", "audio/wav")}).status_code == 403
    assert client.post("/api/voice/profiles", data={"name": "x", "consent": "own"},
                       files={"file": ("a.wav", b"RIFF", "audio/wav")}).status_code == 403


def test_profile_resolution_bot_key_then_default(dv_env):
    st = pb.store()
    assert st.root == dv_env.ctx.paths.voices_dir
    assert pb.resolve_profile("sage") == "stock-host"         # the roster's sage names the stock voice
    set_voice(dv_env.ctx, "sage", "")
    assert pb.resolve_profile("sage") is None                 # nothing yet
    make_profile(st, "host", default=True)
    make_profile(st, "pirate")
    set_voice(dv_env.ctx, "sage", "pirate")
    set_voice(dv_env.ctx, "nova", "gone")
    set_voice(dv_env.ctx, "swift", "")
    assert pb.resolve_profile("sage") == "pirate"
    assert pb.resolve_profile("nova") == "host"               # missing profile -> default
    assert pb.resolve_profile("swift") == "host"
    assert pb.resolve_profile("no-such-bot") == "host"


def test_engine_streams_pcm_frames_and_falls_back_without_a_voice(dv_env):
    eng = chatterbox.ChatterboxTTS()

    class Fallback:
        sample_rate = 24000

        def stream(self, text, profile_id=None, stop=None):
            yield np.ones(10, "<i2")

    chain = tts.FallbackTTS(eng, Fallback(), cooldown_s=30)
    out = list(chain.stream("Hello there, how are you?"))     # no profile at all
    assert len(out) == 1 and out[0][0] == 1 and chain._down_until == 0.0
    make_profile(pb.store(), "host", default=True)
    out = list(chain.stream("Hello there, how are you?"))
    assert out and all(f.dtype == np.dtype("<i2") for f in out)
    assert sum(len(f) for f in out) == 4800                   # 2 x 0.1 s at 24 kHz
    assert all(len(f) <= 960 for f in out)                    # 40 ms frames


def test_stop_event_ends_the_stream(dv_env):
    make_profile(pb.store(), "host", default=True)
    stop = threading.Event()
    stop.set()
    assert list(chatterbox.ChatterboxTTS().stream("Hello there, how are you?", None, stop)) == []


def test_fillers_come_from_the_profile(dv_env):
    st = pb.store()
    make_profile(st, "host", default=True)
    f = pb.fillers("host")
    assert len(f) == 1 and f[0].dtype == np.dtype("<i2") and len(f[0]) == 480
    assert pb.fillers(None) is None and pb.fillers("nope") is None


def test_install_registers_engine_and_resolver_but_no_filler_acks(client, dv_env):
    assert registry.engine_factory("chatterbox") is not None
    make_profile(pb.store(), "host", default=True)
    assert registry.resolve_profile("anyone") == "host"
    assert registry.acks_for("host") is None          # fillers retired as acks (SPEC §6.2)
    assert pb.fillers("host")                          # the store still renders them
    eng = tts.build_engine(VoiceSettings(tts_engine="auto"))
    inner = eng.primary if isinstance(eng, tts.FallbackTTS) else eng
    assert isinstance(inner, chatterbox.ChatterboxTTS)


def test_not_available_means_nothing_mounted(host_env, monkeypatch):
    monkeypatch.setattr(chatterbox, "available", lambda: False)
    assert pb.install() is None
    assert registry.engine_factory("chatterbox") is None and registry._profile_resolver is None
    assert chatterbox.missing() == ["dispatch-voice models (run `dispatch-voice fetch-models`)"]


def test_builtin_or_unknown_voice_falls_back_without_cooldown(dv_env):
    """L6: a built-in voice id, or a profile id that does not exist, is "not
    mine" (LookupError) for the cloned engine — never a 30 s engine cooldown."""
    class Kok:
        sample_rate = 24000

        def stream(self, text, profile_id=None, stop=None):
            yield np.ones(3, "<i2")

    st = pb.store()
    make_profile(st, "real-voice", default=True)
    f = tts.FallbackTTS(chatterbox.ChatterboxTTS(), Kok(), cooldown_s=30.0)
    for pid in ("builtin:am_michael", "no-such-voice"):
        out = list(f.stream("Hello there.", pid))
        assert len(out) == 1 and f._down_until == 0.0, pid
    assert dv_env.engine.calls == []


def test_bot_with_a_builtin_voice_resolves_to_it(dv_env):
    st = pb.store()
    make_profile(st, "real-voice", default=True)
    registry.set_builtin_voices_provider(lambda s: [{"id": "builtin:am_michael", "name": "Michael"}])
    set_voice(dv_env.ctx, "sage", "builtin:am_michael")
    assert pb.resolve_profile("sage") == "builtin:am_michael"
    assert pb.fillers("builtin:am_michael") is None


def test_saved_builtin_voice_without_a_provider_degrades_to_the_default(dv_env):
    """The Kokoro plugin was uninstalled: a bot saved with a built-in voice
    speaks in the default profile instead of failing."""
    st = pb.store()
    make_profile(st, "real-voice", default=True)
    set_voice(dv_env.ctx, "sage", "builtin:am_michael")
    assert pb.resolve_profile("sage") == "real-voice"
    eng = chatterbox.ChatterboxTTS()                      # last resort: no fallback behind it
    assert len(list(eng.stream("Hello there.", "builtin:am_michael"))) > 0
    assert dv_env.engine.calls and dv_env.engine.calls[-1][1] is st.voice("real-voice")


# --------------------------------------------------------------------------- #
# The stock voice (no cloned profile, no built-in voice plugin)
# --------------------------------------------------------------------------- #

from dispatch_voice import stock as dv_stock


class BuildingChatterbox(FakeChatterbox):
    def build_voice(self, ref):
        from dispatch_voice.engine import VoiceTensors
        return VoiceTensors(np.zeros((1, 2, 4), np.float32), np.zeros((1, 3), np.int64),
                            np.zeros((1, 192), np.float32), np.zeros((1, 5, 80), np.float32))

    def synthesize(self, text, voice, stop=None, seed=None):
        return np.zeros(2400, np.float32)


@pytest.fixture
def stock_env(dv_env, monkeypatch):
    fake = BuildingChatterbox()
    monkeypatch.setattr(chatterbox.LazyEngine, "_get", lambda self: fake)
    monkeypatch.setattr(pb, "_render_stock_extras", lambda pid: None)
    dv_env.engine = fake
    return dv_env


def test_last_resort_engine_speaks_the_stock_voice(stock_env):
    eng = chatterbox.ChatterboxTTS()
    assert not eng.has_fallback
    for pid in (None, "builtin:af_heart", "no-such-voice"):
        out = list(eng.stream("Hello there, how are you?", pid))
        assert out, pid
    st = pb.store()
    assert st.get(dv_stock.STOCK_ID)["stock"] is True
    assert pb.default_profile() is None                     # never made the default by itself
    assert pb.resolve_profile(None) == dv_stock.STOCK_ID    # now that it exists


def test_with_a_fallback_behind_it_unknown_voices_still_go_to_the_fallback(stock_env):
    class Kok:
        sample_rate = 24000

        def stream(self, text, profile_id=None, stop=None):
            yield np.ones(3, "<i2")

    f = tts.FallbackTTS(chatterbox.ChatterboxTTS(), Kok(), cooldown_s=30.0)
    assert f.primary.has_fallback
    for pid in (None, "builtin:af_heart", "no-such-voice"):
        out = list(f.stream("Hello there.", pid))
        assert len(out) == 1 and out[0].tolist() == [1, 1, 1], pid
    assert not (pb.store().root / dv_stock.STOCK_ID).exists()   # 2.1 behaviour, no stock build


def test_stock_voice_can_be_chosen_and_made_default(stock_env):
    assert pb.profile_exists(dv_stock.STOCK_ID)          # built on first use
    pb.set_default_profile(dv_stock.STOCK_ID)
    assert pb.default_profile() == dv_stock.STOCK_ID
    assert dv_stock.STOCK_ID in {p["id"] for p in pb.profiles()}


def test_stock_profile_without_build_needs_the_fetched_tensors(stock_env):
    assert pb.stock_profile(build=False) is None          # nothing cached, nothing built
    dv_stock.build_tensors(stock_env.engine)              # what fetch-models does
    assert pb.stock_profile(build=False) == dv_stock.STOCK_ID


def test_default_profile_can_be_set_and_cleared(dv_env):
    st = pb.store()
    make_profile(st, "one")
    make_profile(st, "two", default=True)
    pb.set_default_profile("one")
    assert pb.default_profile() == "one"
    pb.set_default_profile(None)
    assert pb.default_profile() is None
    assert {p["id"] for p in pb.profiles()} == {"one", "two"}


def test_release_drops_the_loaded_engine_and_rearms_extras(dv_env):
    st = pb.store()
    eng = st.engine
    assert isinstance(eng, chatterbox.LazyEngine)
    eng._eng = object()
    pb._EXTRAS_STARTED.add(str(pb.voices_dir()))
    chatterbox.release()
    assert eng._eng is None and not pb._EXTRAS_STARTED


def test_the_store_follows_the_data_dir(dv_env, tmp_path):
    from fakes import FakeContext

    from dispatch_voice.host import context
    a = pb.store()
    other = FakeContext(tmp_path / "other")
    context.set_context(other)
    try:
        assert pb.store() is not a and pb.store().root == other.paths.voices_dir
    finally:
        context.set_context(dv_env.ctx)


def test_cache_tag_and_render_line_of_the_cloned_voice(client, dv_env, monkeypatch):
    """The cloned voice opts into the disk cache (its acknowledgement lines render
    once): the tag names the pinned model, the profile and its `created` stamp, is
    the same for every call, and never loads the engine or builds the stock voice
    (that is `render_line`'s, on the render thread)."""
    eng = chatterbox.ChatterboxTTS()
    loads = []
    monkeypatch.setattr(pb, "stock_profile", lambda build=False: loads.append(build) or None)
    assert eng.cache_tag(None) == eng.cache_tag("stock-host")            # no profile yet: the stock id
    assert eng.cache_tag(None).startswith("cb-") and "stock-host" in eng.cache_tag(None)
    make_profile(pb.store(), "host", default=True)
    tag = eng.cache_tag(None)
    assert tag == eng.cache_tag("host") and "-host-" in tag and tag == eng.cache_tag(None)
    assert eng.cache_tag("builtin:af_heart") == tag                      # no provider: degrades to ours
    eng.has_fallback = True
    assert eng.cache_tag("builtin:af_heart") is None                      # a plugin's voice: its tag
    assert loads == []                                                     # nothing was built or loaded
    assert eng.supports_tags is True
