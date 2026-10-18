"""The stock voice: a profile built from the bundled synthetic clip."""
import json

import numpy as np
import pytest

from forge_voice_core import audio, stock


@pytest.fixture
def models(tmp_path, monkeypatch):
    d = tmp_path / "models"
    monkeypatch.setenv("DISPATCH_VOICE_MODELS", str(d))
    return d


def test_reference_is_packaged_synthetic_audio():
    a = stock.reference_audio()
    assert a.dtype == np.float32 and a.ndim == 1
    assert 10.0 < len(a) / audio.SAMPLE_RATE < 15.0
    assert audio.analyse(a).ok


def test_without_cache_or_build_nothing_is_created(store, models):
    assert stock.ensure_stock(store, build=False) is None
    assert not (store.root / stock.STOCK_ID).exists()
    assert store.list() == []


def test_build_creates_an_ordinary_non_default_profile(store, models, engine):
    assert stock.ensure_stock(store, build=True) == stock.STOCK_ID
    p = store.get(stock.STOCK_ID)
    assert p["name"] == stock.STOCK_NAME and p["stock"] is True
    assert p["consent"]["kind"] == "synthetic" and p["default"] is False
    assert store.default_id() is None                   # never the default by itself
    assert (store.root / stock.STOCK_ID / "voice.npz").exists()
    assert stock.cached_tensors_path().exists()         # cached for the next store
    assert store.voice(stock.STOCK_ID).speaker_embeddings.shape == (1, 192)
    # idempotent
    assert stock.ensure_stock(store, build=True) == stock.STOCK_ID


def test_cached_tensors_make_the_profile_without_an_engine(tmp_path, models, engine):
    from forge_voice_core.profiles import ProfileStore
    stock.build_tensors(engine)
    bare = ProfileStore(tmp_path / "other", None)       # no engine at all
    assert stock.ensure_stock(bare, build=False) == stock.STOCK_ID
    assert json.loads((bare.root / stock.STOCK_ID / "profile.json").read_text(encoding="utf-8"))["stock"] is True


def test_extras_render_audition_and_fillers(store, models, engine):
    from forge_voice_core.profiles import FILLERS
    stock.ensure_stock(store)
    stock.render_extras(store)
    d = store.root / stock.STOCK_ID
    assert (d / "audition.wav").exists()
    assert len(list((d / "fillers").glob("*.wav"))) == len(FILLERS)
    assert store.get(stock.STOCK_ID)["fillers"] == "ready"


def test_stock_speaks_through_speak(store, models):
    from forge_voice_core import speak
    stock.ensure_stock(store)
    frames = list(speak.speak(store, ["Hello there, this is the stock voice."], stock.STOCK_ID))
    assert frames and all(isinstance(f, bytes) for f in frames)


def test_a_cheap_lookup_never_waits_behind_a_build(store, models):
    """A status page asking build=False must not block while another thread
    holds the lock to load the engine and build the voice."""
    with stock._LOCK:
        assert stock.ensure_stock(store, build=False) is None
