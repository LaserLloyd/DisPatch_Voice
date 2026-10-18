import threading

from forge_voice_core import speak
from forge_voice_core.speak import FRAME_SAMPLES
from test_profiles import make


def test_speak_streams_frames_in_order(store, engine):
    p = make(store)
    store.update(p["id"], default=True)
    engine.calls.clear()
    out = list(speak.speak(store, iter(["Sure. ", "I can do that for you right ", "now. And then", " more."])))
    assert engine.calls == ["Sure.", "I can do that for you right now.", "And then more."]
    # Each sentence ends on a short frame; every frame is whole 16-bit samples.
    assert out and all(0 < len(f) <= FRAME_SAMPLES * 2 and len(f) % 2 == 0 for f in out)
    expected = sum(int(0.06 * 24000 * max(1, len(c.split()))) for c in engine.calls) * 2
    assert sum(map(len, out)) == expected


def test_stop_halts_output(store, engine):
    p = make(store)
    stop = threading.Event()
    gen = speak.speak(store, iter(["First sentence here. ", "Second sentence goes on and on."]), p["id"], stop)
    next(gen)
    stop.set()
    assert list(gen) == []


def test_no_default_is_an_error(store):
    import pytest
    with pytest.raises(LookupError):
        list(speak.speak(store, iter(["hi."])))


def test_filler_after_render(store):
    p = make(store)
    assert speak.filler(store, p["id"]) is None
    store.render_fillers(p["id"])
    assert speak.filler(store, p["id"])
