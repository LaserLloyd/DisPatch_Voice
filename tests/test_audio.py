import numpy as np
import pytest

from dispatch_voice import audio
from conftest import SR, speechlike, wav_bytes


def test_decode_roundtrip_and_resample():
    a = speechlike(6.0)
    out = audio.decode(wav_bytes(a[::2], sr=SR // 2))   # 12 kHz in, 24 kHz out
    assert abs(len(out) / SR - 6.0) < 0.05


@pytest.mark.parametrize("data", [b"", b"not audio at all" * 50])
def test_decode_rejects_garbage(data):
    with pytest.raises(audio.ClipError):
        audio.decode(data)


def test_decode_rejects_oversize():
    with pytest.raises(audio.ClipError, match="larger"):
        audio.decode(b"0" * (audio.MAX_UPLOAD_BYTES + 1))


def levels(q):
    return {c.key: c.level for c in q.checks}


def test_clean_clip_passes():
    q = audio.analyse(speechlike(24.0))          # the bursts are ~50% duty, so ~12 s of "speech"
    assert q.ok and levels(q)["length"] == "ok" and levels(q)["noise"] == "ok"


def test_short_clip_fails():
    q = audio.analyse(speechlike(3.0))
    assert not q.ok and levels(q)["length"] == "fail"


def test_noisy_clip_fails():
    q = audio.analyse(speechlike(14.0, noise_db=-12))
    assert levels(q)["noise"] in ("warn", "fail")


def test_clipped_clip_flagged():
    q = audio.analyse(speechlike(14.0, clip=True))
    assert levels(q)["clipping"] in ("warn", "fail")


def test_prepare_reference_trims_cuts_and_normalises():
    a = np.concatenate([np.zeros(SR * 2, np.float32), speechlike(30.0) * 0.1, np.zeros(SR * 2, np.float32)])
    ref = audio.prepare_reference(a)
    assert len(ref) / SR <= audio.MAX_KEEP_S + 0.01
    assert np.abs(ref).max() <= 0.981
    assert np.sqrt(np.mean(ref ** 2)) > np.sqrt(np.mean((a[a != 0]) ** 2))   # louder than the quiet input


def test_decode_caps_duration(monkeypatch):
    monkeypatch.setattr(audio, "MAX_DECODE_S", 2)
    out = audio.decode(wav_bytes(speechlike(6.0)))
    assert abs(len(out) / SR - 2.0) < 0.05


def test_decode_cannot_be_steered_to_read_a_local_file(tmp_path):
    """An HLS playlist upload must not make ffmpeg open file:// (or any
    protocol other than the stdin pipe)."""
    secret = tmp_path / "clip.wav"
    secret.write_bytes(wav_bytes(speechlike(6.0)))
    playlist = (b"#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:6.0,\n"
                + f"file://{secret}\n".encode() + b"#EXT-X-ENDLIST\n")
    with pytest.raises(audio.ClipError):
        audio.decode(playlist)


def test_decode_argv_restricts_input_protocols_and_duration(monkeypatch):
    """Defence in depth: whatever the local ffmpeg build's own demuxer
    safety, the argv pins input to the stdin pipe and caps the duration."""
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        raise audio.subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    with pytest.raises(audio.ClipError):
        audio.decode(b"x" * 100, ffmpeg="ffmpeg")
    argv = seen["argv"]
    i = argv.index("-protocol_whitelist")
    assert argv[i + 1] == "pipe" and i < argv.index("-i")
    assert argv[argv.index("-t") + 1] == str(audio.MAX_DECODE_S)
