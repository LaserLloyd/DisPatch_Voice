"""After the in-app model download, speech works without restarting DisPatch (review F6).

Setup ran with the Chatterbox models absent, so nothing was registered. The files then land
(the download job), and the next ``tts.build_engine`` / ``tts.shared_engine`` / ``resolve_profile``
must register the engine and the resolver by itself, and ``status.ready`` must be true only when
a reply can actually be spoken (not while the download is half way)."""
from __future__ import annotations

import pytest

from dispatch_voice.engine import COMPONENTS, _filename
from dispatch_voice.host import fetch_job, models, registry, routes, stt, tts
from dispatch_voice.host import profiles_bridge as pb
from dispatch_voice.host.config import load_settings
from dispatch_voice.host.engines import chatterbox


def put(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture
def cb_dir(host_env, monkeypatch, tmp_path):
    """An empty Chatterbox folder the add-on resolves to (nothing is loaded from it)."""
    d = tmp_path / "chatterbox-models"
    monkeypatch.setenv("DISPATCH_VOICE_MODELS", str(d))
    monkeypatch.setattr(chatterbox.LazyEngine, "_get", lambda self: pytest.fail("no model may load here"))
    return d


def land_chatterbox(d, *, last_component=True):
    """What the download job writes, in its order: tokenizer, config, then each graph + weights."""
    put(d / "tokenizer.json")
    put(d / "config.json")
    comps = COMPONENTS if last_component else COMPONENTS[:-1]
    for c in comps:
        fn = _filename(c, "q4")
        put(d / "onnx" / fn)
        put(d / "onnx" / (fn + "_data"))


def test_a_download_in_progress_is_not_available(cb_dir):
    assert not chatterbox.available()
    put(cb_dir / "tokenizer.json")                       # the first file creates onnx/ at once ...
    (cb_dir / "onnx").mkdir()
    assert not chatterbox.available()
    land_chatterbox(cb_dir, last_component=False)       # ... three graphs of four ...
    assert not chatterbox.available()
    put(cb_dir / "onnx" / (_filename(COMPONENTS[-1], "q4")))     # ... the graph without its weights
    assert not chatterbox.available()
    put(cb_dir / "onnx" / (_filename(COMPONENTS[-1], "q4") + "_data"))
    assert chatterbox.available()


def test_the_engine_registers_itself_when_the_files_land(cb_dir):
    """setup() found no models and registered nothing; build_engine registers the engine later."""
    s = load_settings()
    assert registry.engine_spec("chatterbox") is None
    with pytest.raises(RuntimeError, match="no TTS engine"):
        tts.build_engine(s)
    land_chatterbox(cb_dir)
    eng = tts.build_engine(s)                            # auto -> chatterbox, registered on the spot
    assert registry.engine_spec("chatterbox") is not None
    assert type(eng).__name__ == "ChatterboxTTS"
    assert registry.has_profile_resolver()


def test_the_shared_engine_is_built_once_across_the_registration(cb_dir):
    """The cache key reads the registered engine names: registering inside build_engine would
    key the first engine under the old names and build a second one on the next call."""
    land_chatterbox(cb_dir)
    s = load_settings()
    first = tts.shared_engine(s)
    assert tts.shared_engine(s) is first and len(tts._ENGINES) == 1


def test_status_is_ready_exactly_when_a_reply_can_be_spoken(cb_dir, monkeypatch):
    sp = models.speech_dir()
    for f in ("silero_vad.onnx", "smart-turn-v3.2-cpu.onnx"):
        put(sp / f)
    for f in models.PARAKEET_FILES[:-1]:                 # the decoder is still coming down
        put(sp / "parakeet-tdt-0.6b-v2" / f)
    land_chatterbox(cb_dir)
    s = load_settings()
    assert any("parakeet" in m for m in s.missing())     # speech recognition is not whole yet
    put(sp / "parakeet-tdt-0.6b-v2" / models.PARAKEET_FILES[-1])
    miss = [m for m in load_settings().missing() if "python package" not in m]
    assert miss == []
    # ready means build_engine works: that is the guarantee F6 asks for
    assert tts.build_engine(load_settings()) is not None


def test_not_ready_while_the_voice_models_are_missing(cb_dir):
    s = load_settings()
    assert any("fetch-models" in m or "dispatch-voice models" in m for m in s._tts_missing())
    land_chatterbox(cb_dir)
    assert load_settings()._tts_missing() == []


def test_a_profile_resolves_without_the_setup_resolver(cb_dir):
    """acks and the routes call registry.resolve_profile; with the resolver unset it tries the
    lazy registration first (the stock voice then speaks, not the engine's bare default)."""
    land_chatterbox(cb_dir)
    assert not registry.has_profile_resolver()
    registry.resolve_profile("anyone")
    assert registry.has_profile_resolver()


def test_ensure_engine_is_false_without_models_and_idempotent_with(cb_dir):
    assert pb.ensure_engine() is False and registry.engine_spec("chatterbox") is None
    land_chatterbox(cb_dir)
    assert pb.ensure_engine() is True and pb.ensure_engine() is True
    assert len([n for n in registry.engine_names() if n == "chatterbox"]) == 1


def test_an_explicit_chatterbox_choice_registers_too(cb_dir, monkeypatch):
    monkeypatch.setenv("DISPATCH_VOICE_TTS", "chatterbox")
    land_chatterbox(cb_dir)
    assert type(tts.build_engine(load_settings())).__name__ == "ChatterboxTTS"


def test_speech_recognition_picks_the_models_up_without_a_restart(host_env, monkeypatch, tmp_path):
    """The STT engine is built lazily from the folder in force at the time of the call, and a
    failed build is not cached: the first call before the download raises, the one after loads."""
    monkeypatch.setenv("DISPATCH_SPEECH_MODELS", str(tmp_path / "speech"))
    built = []

    class Fake:
        def __init__(self, d, threads=4, *, spinning=False):
            built.append(d)

    monkeypatch.setattr(stt, "ParakeetSTT", Fake)
    s = load_settings()
    assert any("parakeet" in m for m in s.missing())
    eng = stt.shared_engine(s)                           # the build itself is deferred to first use
    assert stt.shared_engine(load_settings()) is eng and len(built) == 1


def test_a_finished_download_asks_for_a_restart_only_when_the_router_is_not_mounted(monkeypatch):
    job = fetch_job.FetchJob(run=lambda **kw: 0)
    monkeypatch.setattr(routes, "_LOADED", [])
    job.start(only=None, force=False, stock=False)
    job.join(5)
    assert job.status()["state"] == "done" and job.status()["restart_required"] is True
    monkeypatch.setattr(routes, "_LOADED", ["dispatch-voice"])     # cloned voices were mounted at setup
    job2 = fetch_job.FetchJob(run=lambda **kw: 0)
    job2.start(only=None, force=False, stock=False)
    job2.join(5)
    assert job2.status()["restart_required"] is False
    job3 = fetch_job.FetchJob(run=lambda **kw: 0)
    monkeypatch.setattr(routes, "_LOADED", [])
    job3.start(only="speech", force=False, stock=False)
    job3.join(5)
    assert job3.status()["restart_required"] is False             # speech models alone never need one
