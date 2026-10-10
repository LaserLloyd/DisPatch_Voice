"""Parakeet transcribes the Opus uplink like PCM (SPEC §5.1: WER 0.0 at 16 and 12 kbps).

Model-backed: skips without ``onnx_asr`` or the Parakeet models. Run it under the
box-wide models lock::

    flock "$VOICE_LOCK" uv run --no-sync --with onnx-asr \\
        pytest tests/transport/test_models_opus.py -m models
"""
import os
import re
from pathlib import Path

import numpy as np
import pytest

from dispatch_voice.transport import Transport, opus, wire3

pytestmark = pytest.mark.models

MODELS = Path(os.environ.get("DISPATCH_SPEECH_MODELS", Path.home() / ".local/share/dispatch-voice-models"))
PARAKEET = MODELS / "parakeet-tdt-0.6b-v2"
CLIPS = ["short", "medium", "long", "pause", "interrupt", "driver1b", "driver2"]


@pytest.fixture(scope="module")
def asr():
    onnx_asr = pytest.importorskip("onnx_asr")
    if not opus.available():
        pytest.skip("libopus not installed")
    if not PARAKEET.exists():
        pytest.skip(f"no Parakeet models at {PARAKEET}")
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.log_severity_level = 3
    return onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", str(PARAKEET), quantization="int8",
                               sess_options=so, providers=["CPUExecutionProvider"])


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(1, len(ref))


def through_the_uplink(x16: np.ndarray, kbps: int) -> np.ndarray:
    """What the server's detector gets: browser-style voip Opus in 100 ms v3 messages, decoded by
    the transport and stitched by its OffsetTracker (the real server path, end to end)."""
    enc = opus.Encoder(16000, kbps * 1000, application="voip")
    t = Transport.negotiate(None, {"v": "3", "s": "stream-0001", "up": "opus", "down": "opus"})
    out = []
    frames = len(x16) // 320
    for i in range(0, frames, 5):
        packets = [enc.encode(x16[(i + j) * 320:(i + j + 1) * 320]) for j in range(min(5, frames - i))]
        off, _flags, pcm = t.decode_up(wire3.pack_up(1, 0, i * 320, packets))
        out += t.offsets.feed(off, pcm, now_off=i * 320)
    y = np.concatenate(out)
    return y[enc.lookahead:]


def test_parakeet_on_opus_matches_pcm(asr, fixtures_dir):
    from fakes import load_clip
    rows = []
    for name in CLIPS:
        x = np.clip(load_clip(name), -1, 1)
        x16 = (x * 32767).astype(np.int16)
        ref = asr.recognize(x.astype(np.float32), sample_rate=16000)
        for kbps in (16, 12):
            y = through_the_uplink(x16, kbps).astype(np.float32) / 32768
            hyp = asr.recognize(y, sample_rate=16000)
            rows.append((name, kbps, wer(words(ref), words(hyp)), ref, hyp))
    for name, kbps, w, ref, hyp in rows:
        print(f"{name:10} {kbps} kbps WER {w:.3f}  pcm={ref!r}  opus={hyp!r}")
    by_rate = {k: np.mean([w for _, kk, w, _, _ in rows if kk == k]) for k in (16, 12)}
    assert by_rate[16] == 0.0, by_rate
    assert by_rate[12] <= 0.02, by_rate
