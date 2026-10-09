"""Reference-clip decoding, clean-up and quality checks.

Everything here works on mono float32 at SAMPLE_RATE. Decoding shells out to
ffmpeg with a fixed argv (never a shell) and reads the PCM from a pipe, so any
container ffmpeg understands (wav, mp3, m4a, ogg, webm from a browser recorder)
is accepted without touching the filesystem.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass, field

import numpy as np

SAMPLE_RATE = 24000          # Chatterbox S3Gen rate
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_SPEECH_S = 5.0           # the model refuses shorter prompts
IDEAL_SPEECH_S = (10.0, 20.0)
MAX_KEEP_S = 20.0            # longer clips are cut to the best window
TARGET_LUFS = -20.0
FFMPEG_TIMEOUT_S = 60
#: Never decode more than this many seconds, whatever the container claims.
#: Only MAX_KEEP_S is ever kept; the headroom is for analysis of long takes.
MAX_DECODE_S = 300


class ClipError(ValueError):
    """The upload can't be used. The message is safe to show to the user."""


def decode(data: bytes, ffmpeg: str | None = None) -> np.ndarray:
    if not data:
        raise ClipError("The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ClipError(f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    exe = ffmpeg or shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg is not installed")
    argv = [exe, "-nostdin", "-hide_banner", "-loglevel", "error",
            # The upload is untrusted: ffmpeg may open ONLY the stdin pipe, so
            # a playlist-style container (HLS, concat, ...) cannot make it
            # read a local file or fetch a URL.
            "-protocol_whitelist", "pipe",
            "-i", "pipe:0", "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-t", str(MAX_DECODE_S),
            "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    try:
        proc = subprocess.run(argv, input=data, capture_output=True, timeout=FFMPEG_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as e:
        raise ClipError("Decoding took too long.") from e
    if proc.returncode != 0 or not proc.stdout:
        raise ClipError("That doesn't look like an audio file ffmpeg can read.")
    audio = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if not np.all(np.isfinite(audio)):
        raise ClipError("The audio contains invalid samples.")
    return audio


def _frames(a: np.ndarray, sr: int, ms: int = 20) -> np.ndarray:
    n = int(sr * ms / 1000)
    usable = len(a) // n * n
    return a[:usable].reshape(-1, n) if usable else np.zeros((0, n), np.float32)


def speech_mask(a: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """Energy-based speech/non-speech per 20 ms frame. Returns (mask, frame_db)."""
    fr = _frames(a, sr)
    if not len(fr):
        return np.zeros(0, bool), np.zeros(0)
    db = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    floor = np.percentile(db, 10)
    peak = np.percentile(db, 95)
    # Speech if well above the noise floor and not far below the loud parts.
    thresh = max(floor + 10.0, peak - 35.0)
    mask = db > thresh
    # Close 200 ms gaps so syllables aren't split into silence.
    k = 10
    padded = np.convolve(mask.astype(float), np.ones(k), "same") > 0
    return padded & (db > floor + 6.0), db


def trim(a: np.ndarray, sr: int = SAMPLE_RATE, keep_pad_s: float = 0.15) -> np.ndarray:
    mask, _ = speech_mask(a, sr)
    if not mask.any():
        return a
    hop = int(sr * 0.02)
    start = max(0, np.argmax(mask) * hop - int(keep_pad_s * sr))
    end = min(len(a), (len(mask) - np.argmax(mask[::-1])) * hop + int(keep_pad_s * sr))
    return a[start:end]


def best_window(a: np.ndarray, sr: int = SAMPLE_RATE, seconds: float = MAX_KEEP_S) -> np.ndarray:
    """Pick the window with the most speech when a clip is too long."""
    n = int(seconds * sr)
    if len(a) <= n:
        return a
    mask, _ = speech_mask(a, sr)
    per = int(seconds / 0.02)
    if len(mask) <= per:
        return a[:n]
    score = np.convolve(mask.astype(float), np.ones(per), "valid")
    i = int(np.argmax(score)) * int(sr * 0.02)
    return a[i:i + n]


def normalise_loudness(a: np.ndarray, sr: int = SAMPLE_RATE, target: float = TARGET_LUFS) -> np.ndarray:
    import pyloudnorm as pyln
    if len(a) < int(0.5 * sr):
        return a
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(a.astype(np.float64))
    if not np.isfinite(lufs):
        return a
    out = a * (10 ** ((target - lufs) / 20))
    peak = np.abs(out).max()
    if peak > 0.98:                       # never clip while normalising
        out = out * (0.98 / peak)
    return out.astype(np.float32)


@dataclass
class Check:
    key: str
    level: str          # ok | warn | fail
    message: str


@dataclass
class Quality:
    duration_s: float
    speech_s: float
    snr_db: float
    clipping_pct: float
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(c.level == "fail" for c in self.checks)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def analyse(a: np.ndarray, sr: int = SAMPLE_RATE) -> Quality:
    """Score a raw (decoded, not yet trimmed) reference clip."""
    duration = len(a) / sr
    mask, db = speech_mask(a, sr)
    speech_s = float(mask.sum() * 0.02)
    if mask.any() and (~mask).any():
        snr = float(np.median(db[mask]) - np.percentile(db[~mask], 50))
    elif mask.any():
        snr = 40.0
    else:
        snr = 0.0
    clipping = float(np.mean(np.abs(a) >= 0.999) * 100) if len(a) else 0.0
    q = Quality(round(duration, 2), round(speech_s, 2), round(snr, 1), round(clipping, 3))

    if speech_s < MIN_SPEECH_S:
        q.checks.append(Check("length", "fail",
                              f"Only {speech_s:.1f} s of speech. At least {MIN_SPEECH_S:.0f} s is needed, 10–20 s is best."))
    elif speech_s < IDEAL_SPEECH_S[0]:
        q.checks.append(Check("length", "warn", f"{speech_s:.1f} s of speech works, but 10–20 s clones better."))
    elif duration > MAX_KEEP_S + 2:
        q.checks.append(Check("length", "ok", f"Long clip: the best {MAX_KEEP_S:.0f} s of speech will be used."))
    else:
        q.checks.append(Check("length", "ok", f"{speech_s:.1f} s of speech."))

    if snr < 12:
        q.checks.append(Check("noise", "fail", "Too much background noise. Record somewhere quiet, close to the mic."))
    elif snr < 20:
        q.checks.append(Check("noise", "warn", "Some background noise. It will be cloned into the voice."))
    else:
        q.checks.append(Check("noise", "ok", "Clean background."))

    if clipping > 0.5:
        q.checks.append(Check("clipping", "fail", "The recording is distorted (clipping). Record a little quieter."))
    elif clipping > 0.1:
        q.checks.append(Check("clipping", "warn", "A few clipped peaks. Fine, but quieter is better."))
    else:
        q.checks.append(Check("clipping", "ok", "No clipping."))

    if len(db) and mask.any() and float(np.percentile(db[mask], 95) - np.percentile(db[mask], 5)) < 6:
        q.checks.append(Check("dynamics", "warn",
                              "Very flat level. This can mean music or heavy compression. Plain speech clones best."))
    return q


def prepare_reference(a: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Trim, cut to the best window and normalise. The result is what gets cloned."""
    a = trim(a, sr)
    a = best_window(a, sr)
    return normalise_loudness(a, sr)


def to_pcm16(a: np.ndarray) -> bytes:
    return (np.clip(a, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def to_wav_bytes(a: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, a, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
