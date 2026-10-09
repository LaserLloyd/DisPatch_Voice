"""Real-engine tests. Slow (~30 s on a laptop CPU). Skipped when the models aren't downloaded."""
from pathlib import Path

import numpy as np
import pytest

from dispatch_voice import audio
from dispatch_voice.engine import COMPONENTS, _filename, default_model_dir

FIXTURE = Path(__file__).parents[1] / "src" / "dispatch_voice" / "stock" / "synthetic-host.wav"
have_models = all((default_model_dir() / "onnx" / _filename(c, "q4")).exists() for c in COMPONENTS)
pytestmark = [pytest.mark.models, pytest.mark.skipif(not have_models, reason="run `dispatch-voice fetch-models`")]


@pytest.fixture(scope="module")
def real():
    from dispatch_voice.engine import ChatterboxOnnx
    return ChatterboxOnnx(watermark=False)


def test_clone_and_speak(real, tmp_path):
    from dispatch_voice.profiles import ProfileStore
    store = ProfileStore(tmp_path / "voices", real)
    p = store.create("Synthetic host", FIXTURE.read_bytes(), FIXTURE.name, "synthetic")
    assert p["timings"]["build_s"] < 5
    # the clone should sound more like the reference than an unrelated voice does
    assert p["quality"]["match"] is not None and p["quality"]["match"] > 0.6
    wav = store.speak_line(p["id"], "Sure thing, I am on it.")
    assert 0.5 < len(wav) / audio.SAMPLE_RATE < 4
    assert np.isfinite(wav).all() and np.abs(wav).max() > 0.01
    assert real.last_stats["rtf"] < 3


def test_stop_is_honoured(real):
    import threading
    ref = audio.prepare_reference(audio.decode(FIXTURE.read_bytes()))
    v = real.build_voice(ref)
    stop = threading.Event(); stop.set()
    assert len(real.synthesize("This should never be spoken.", v, stop)) == 0


def test_no_runaway_or_reference_replay(real):
    """Greedy decoding used to loop the reference clip's words for ~10 s. Guard the length."""
    from dispatch_voice.profiles import AUDITION_LINE
    ref = audio.prepare_reference(audio.decode(FIXTURE.read_bytes()))
    v = real.build_voice(ref)
    for seed in (1, 2):
        aud = real.synthesize(AUDITION_LINE, v, seed=seed)
        assert 4.0 < len(aud) / audio.SAMPLE_RATE < 10.0
        short = real.synthesize("Mm-hmm.", v, seed=seed)
        assert len(short) / audio.SAMPLE_RATE < 2.5
