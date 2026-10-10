#!/usr/bin/env python3
"""Rebuild the derived audio fixtures (the checked-in files are the reference).

    python tests/fixtures/make_fixtures.py derive      # clips-8k/ from clips/ (+ addendum)
    python tests/fixtures/make_fixtures.py addendum    # addendum.wav (needs kokoro_onnx + the Kokoro model)

`derive` is pure numpy and deterministic. It models the Bluetooth SCO/HFP path
of a car kit: an 8 kHz capture (anti-alias low-pass at 3.6 kHz, decimate by 2),
then the mic worklet's linear upsampling back to 16 kHz (v3 contracts §E.2: a
context below 16 kHz is upsampled, never decimated). Output stays 16 kHz mono
16-bit so every consumer reads one format.

`addendum` synthesises the two halves with Kokoro (voice am_michael, the voice
the other clips use) and joins them with exactly 2.0 s of silence. Kokoro is a
separate, optional plugin; nothing in the test suite imports it.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SR = 16000
ADDENDUM = ("What's on my calendar this afternoon", "Oh, and tomorrow.")
ADDENDUM_GAP_S = 2.0


def read(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0


def write(path: Path, x: np.ndarray, sr: int = SR) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).round().astype("<i2").tobytes())


def lowpass(x: np.ndarray, cutoff_hz: float, sr: int, taps: int = 127) -> np.ndarray:
    fc = cutoff_hz / sr
    m = (taps - 1) / 2
    n = np.arange(taps) - m
    h = 2 * fc * np.sinc(2 * fc * n) * np.blackman(taps)
    h /= h.sum()
    return np.convolve(x, h, mode="same").astype(np.float32)


def via_8k(x: np.ndarray) -> np.ndarray:
    """16 kHz -> 8 kHz capture -> linear upsample to 16 kHz (the worklet's path)."""
    x8 = lowpass(x, 3600.0, SR)[::2]
    t16 = np.arange(len(x8) * 2) / 2.0
    return np.interp(t16, np.arange(len(x8)), x8).astype(np.float32)


def derive() -> None:
    src, dst = HERE / "clips", HERE / "clips-8k"
    dst.mkdir(exist_ok=True)
    meta = {}
    sources = sorted(src.glob("*.wav"))
    if (HERE / "addendum.wav").exists():
        sources.append(HERE / "addendum.wav")
    for p in sources:
        y = via_8k(read(p))
        write(dst / p.name, y)
        meta[p.stem] = {"from": f"{p.parent.name}/{p.name}" if p.parent != HERE else p.name,
                        "seconds": round(len(y) / SR, 2)}
    (dst / "meta.json").write_text(json.dumps({
        "$about": "Band-limited copies: 8 kHz capture (low-pass 3.6 kHz, decimate) then linear "
                  "upsampling to 16 kHz, as the mic worklet does for an SCO context. "
                  "Built by make_fixtures.py derive.",
        "clips": meta}, indent=1) + "\n")
    print(f"clips-8k: {len(meta)} files")


def addendum() -> None:
    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    md = Path.home() / ".local/share/dispatch-voice-models/kokoro"
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(md / "kokoro-v1.0.onnx"), sess_options=so,
                                providers=["CPUExecutionProvider"])
    k = Kokoro.from_session(sess, str(md / "voices-v1.0.bin"))
    parts = []
    for text in ADDENDUM:
        a, sr = k.create(text, voice="am_michael", speed=1.0, lang="en-us")
        a = np.asarray(a, np.float32)
        n = round(len(a) * SR / sr)
        a = lowpass(a, 7000.0, sr)
        parts.append(np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32))
    gap = np.zeros(int(ADDENDUM_GAP_S * SR), np.float32)
    x = np.concatenate([parts[0], gap, parts[1]])
    write(HERE / "addendum.wav", x)
    info = {"text": f"{ADDENDUM[0]} {ADDENDUM[1]}", "first": ADDENDUM[0], "second": ADDENDUM[1],
            "first_end_sample": len(parts[0]), "second_start_sample": len(parts[0]) + len(gap),
            "first_end_s": round(len(parts[0]) / SR, 3),
            "second_start_s": round((len(parts[0]) + len(gap)) / SR, 3),
            "gap_s": ADDENDUM_GAP_S, "seconds": round(len(x) / SR, 2), "voice": "kokoro am_michael"}
    (HERE / "addendum.json").write_text(json.dumps(info, indent=1) + "\n")
    print("addendum.wav", info)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["derive", "addendum"])
    a = ap.parse_args(argv)
    {"derive": derive, "addendum": addendum}[a.what]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
