"""ctypes libopus binding: round trips, bitrate rungs, fail-soft (C-§B.3; SPEC §5.1)."""
import ctypes
import ctypes.util

import numpy as np
import pytest

from dispatch_voice.transport import Transport, opus, wire3

_LIBOPUS_PRESENT = opus.available()
needs_opus = pytest.mark.skipif(not _LIBOPUS_PRESENT, reason="libopus not installed")


def lag_corr(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    n = min(len(x), len(y) - lag)
    return float(np.corrcoef(x[:n].astype(float), y[lag:lag + n].astype(float))[0, 1])


def roundtrip(x: np.ndarray, rate: int, kbps: int, app: str, dec_rate: int | None = None):
    enc = opus.Encoder(rate, kbps * 1000, application=app)
    dec = opus.Decoder(dec_rate or rate)
    n = len(x) // enc.frame
    packets = [enc.encode(x[i * enc.frame:(i + 1) * enc.frame]) for i in range(n)]
    y = np.concatenate([dec.decode(p) for p in packets])
    return enc, packets, y


def speech(fixtures_dir, name="medium"):
    from fakes import load_clip
    return (np.clip(load_clip(name), -1, 1) * 32767).astype(np.int16)


@needs_opus
def test_version_and_available():
    assert opus.available()
    assert opus.version().startswith("libopus")


def envelope_corr(x: np.ndarray, y: np.ndarray, lag: int, hop: int = 320) -> float:
    a, b = x[:len(y) - lag].astype(float), y[lag:].astype(float)
    m = min(len(a), len(b)) // hop
    ea = np.sqrt((a[:m * hop].reshape(m, hop) ** 2).mean(1))
    eb = np.sqrt((b[:m * hop].reshape(m, hop) ** 2).mean(1))
    return float(np.corrcoef(ea, eb)[0, 1])


@needs_opus
@pytest.mark.parametrize("rate,kbps,app", [(24000, 24, "audio"), (24000, 16, "audio"),
                                           (16000, 16, "audio"), (16000, 12, "audio"),
                                           (16000, 16, "voip"), (16000, 12, "voip")])
def test_round_trip_speech_correlates(fixtures_dir, rate, kbps, app):
    x16 = speech(fixtures_dir)
    if rate == 24000:   # the downlink: an engine-rate signal (linear 16 -> 24 kHz is fine for a correlation test)
        t = np.arange(int(len(x16) * 1.5)) / 1.5
        x = np.interp(t, np.arange(len(x16)), x16).astype(np.int16)
    else:
        x = x16
    enc, packets, y = roundtrip(x, rate, kbps, app)
    assert len(y) == len(packets) * enc.frame
    if app == "audio":
        # the server's downlink mode (CELT/hybrid) is waveform-matching: corr >= 0.9 after the lookahead
        assert lag_corr(x, y, enc.lookahead) >= 0.9
    else:
        # the browser's uplink mode ("voip", what isConfigSupported normalises to) is SILK, which
        # keeps the spectrum and envelope but not the waveform (measured 0.83-0.87 sample corr at
        # 12-16 kbps); the uplink's real acceptance is Parakeet's transcript (test_models_opus.py)
        assert lag_corr(x, y, enc.lookahead) >= 0.75
        assert envelope_corr(x, y, enc.lookahead) >= 0.98
    mean_bytes = np.mean([len(p) for p in packets])
    # measured on this box: ~59 B (24 kbps), ~40 B (16 kbps), ~29 B (12 kbps); allow VBR spread
    assert mean_bytes == pytest.approx(kbps * 1000 / 8 / 50, rel=0.35)


@needs_opus
def test_decoder_rate_is_independent_of_the_encoder_rate(fixtures_dir):
    x = speech(fixtures_dir, "short")
    _enc, packets, y = roundtrip(x, 16000, 16, "voip", dec_rate=48000)
    assert len(y) == len(packets) * 960          # browsers decode to 48 kHz; libopus can too
    assert opus.packet_samples(packets[0], 48000) == 960
    assert opus.packet_samples(packets[0], 16000) == 320


@needs_opus
def test_set_bitrate_changes_packet_sizes(fixtures_dir):
    x = speech(fixtures_dir)
    enc = opus.Encoder(16000, 16000, application="voip")
    frames = [x[i * 320:(i + 1) * 320] for i in range(len(x) // 320)]
    half = len(frames) // 2
    a = [len(enc.encode(f)) for f in frames[:half]]
    enc.set_bitrate(12000)
    assert enc.bitrate == 12000
    b = [len(enc.encode(f)) for f in frames[half:]]
    assert np.mean(b) < np.mean(a) * 0.9
    enc.set_bitrate(1)                     # clamped, never an error
    assert enc.bitrate == opus.MIN_BITRATE


@needs_opus
def test_bad_inputs_raise_cleanly():
    enc = opus.Encoder(24000, 24000)
    with pytest.raises(ValueError):
        enc.encode(np.zeros(100, np.int16))
    with pytest.raises(ValueError):
        enc.encode(np.zeros(480, np.float32))
    with pytest.raises(ValueError):
        opus.Encoder(22050, 24000)
    with pytest.raises(ValueError):
        opus.Encoder(24000, 24000, frame_ms=7)
    dec = opus.Decoder(16000)
    with pytest.raises(opus.OpusError):
        dec.decode(b"")
    with pytest.raises(opus.OpusError):
        dec.decode(bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF]))       # code-3 packet with an absurd frame count
    assert len(dec.decode(None)) == 320                           # PLC: one 20 ms frame
    enc.close()
    enc.close()                                                   # idempotent
    with pytest.raises(opus.OpusError):
        enc.encode(np.zeros(480, np.int16))


@needs_opus
def test_reset_starts_an_independent_stream(fixtures_dir):
    x = speech(fixtures_dir, "short")[:4800]
    enc = opus.Encoder(24000, 24000)
    first = [enc.encode(x[i:i + 480]) for i in range(0, 4800, 480)]
    enc.reset()
    again = [enc.encode(x[i:i + 480]) for i in range(0, 4800, 480)]
    assert first == again


@pytest.fixture
def no_libopus(monkeypatch):
    """Make libopus unloadable for the duration of a test."""
    real_cdll = ctypes.CDLL
    real_find = ctypes.util.find_library

    def refuse(name, *a, **kw):
        if "opus" in str(name):
            raise OSError(f"{name}: cannot open shared object file")
        return real_cdll(name, *a, **kw)

    monkeypatch.setattr(ctypes.util, "find_library", lambda n: None if n == "opus" else real_find(n))
    monkeypatch.setattr(ctypes, "CDLL", refuse)
    opus._reset_for_tests()
    yield
    monkeypatch.undo()
    opus._reset_for_tests()


def test_fail_soft_without_libopus(no_libopus):
    assert opus.available() is False
    assert opus.version() == ""
    with pytest.raises(opus.OpusUnavailable):
        opus.Encoder(24000, 24000)
    with pytest.raises(opus.OpusUnavailable):
        opus.Decoder(16000)
    t = Transport.negotiate(None, {"v": "3", "s": "stream-0001", "up": "opus,pcm16", "down": "opus,pcm16"})
    assert (t.proto, t.codec_in, t.codec_out) == (3, "pcm16", "pcm16")
    assert t.ready_fields()["codec_out"] == "pcm16"
    # a client that sends Opus anyway gets a bad frame, never a crash
    with pytest.raises(ValueError):
        t.decode_up(wire3.pack_up(1, 0, 0, [b"\x08\x00"]))
    # PCM still works
    _off, _flags, pcm = t.decode_up(wire3.pack_up(0, 0, 0, [b"\x01\x00\x02\x00"]))
    assert pcm.tolist() == [1, 2]


def test_switched_off_by_env(monkeypatch):
    monkeypatch.setenv("DISPATCH_VOICE_OPUS", "off")
    assert opus.available() is False
    t = Transport.negotiate(None, {"v": "3", "up": "opus", "down": "opus"})
    assert (t.codec_in, t.codec_out) == ("pcm16", "pcm16")
    monkeypatch.setenv("DISPATCH_VOICE_OPUS", "auto")
    if not _LIBOPUS_PRESENT:
        pytest.skip("libopus not installed: cannot check that 'auto' re-enables it")
    assert opus.available() is True
