"""sf.1 is pinned: the add-on's constants and frames match the golden table
(transcribed from SpeechForge v0.2.0), and — when a SpeechForge checkout is on
this machine — the golden table still matches SpeechForge's source (read with
``ast``, never imported)."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest

from forge_voice_core import rig

from .fake_sf1 import GOLDEN, check_client_frame

SPEECHFORGE = Path(os.environ.get("SPEECHFORGE_CHECKOUT", Path.home() / "Projects" / "speechforge"))


def test_proto_and_constants_are_pinned():
    assert rig.PROTO == GOLDEN["proto"] == "sf.1"
    c = GOLDEN["constants"]
    assert (rig.MAGIC, rig.KIND_MIC, rig.KIND_TTS) == (c["MAGIC"], c["KIND_MIC"], c["KIND_TTS"])
    assert rig.HEADER.format == c["HEADER"] and rig.HEADER_SIZE == c["HEADER_SIZE"] == 16
    assert (rig.MIC_RATE, rig.TTS_RATE) == (c["MIC_RATE"], c["TTS_RATE"])
    assert (rig.MIC_FRAME_SAMPLES, rig.TTS_FRAME_SAMPLES) == (c["MIC_FRAME_SAMPLES"], c["TTS_FRAME_SAMPLES"])
    assert rig.MAX_MESSAGE_BYTES == c["MAX_MESSAGE_BYTES"]


def test_audio_frames_round_trip_and_reject_garbage():
    pcm = np.arange(-160, 160, dtype="<i2")
    buf = rig.pack_audio(rig.KIND_MIC, 0, 0, 7, 70000, pcm)
    assert len(buf) == 16 + 2 * 320 and buf[0] == 0x53 and buf[1] == 0x01
    kind, ctx, gen, seq, pts, out = rig.unpack_audio(buf)
    assert (kind, ctx, gen, seq, pts) == (1, 0, 0, 7, 70000) and np.array_equal(out, pcm)
    for bad in (b"", b"S\x02" + b"\x00" * 10, b"X" + buf[1:], buf[:-2], buf[:1] + b"\x09" + buf[2:]):
        with pytest.raises(ValueError):
            rig.unpack_audio(bad)


def test_every_frame_the_client_builds_is_valid_sf1():
    # The fake validates every live frame too; this pins the shapes on their own.
    frames = [
        {"type": "hello", "proto": "sf.1", "client": {"name": "dispatch-voice", "version": "1"}, "caps": {"opus_out": False}},
        {"type": "stt.config", "lang": "en", "partial_ms": 160, "eou": {"enabled": True, "min_pause_ms": 150}},
        {"type": "stt.flush", "id": 1}, {"type": "stt.reset"},
        {"type": "tts.open", "ctx": 1, "gen": 0, "voice": "sha256:" + "0" * 64, "first_chunk_tokens": 15,
         "seed": 3, "speculative": False},
        {"type": "tts.text", "ctx": 1, "text": "Hello.", "final": True}, {"type": "tts.flush", "ctx": 1, "id": 2},
        {"type": "tts.close", "ctx": 1}, {"type": "tts.pause", "ctx": 1}, {"type": "tts.resume", "ctx": 1},
        {"type": "cancel", "gen": 4}, {"type": "cancel", "ctx": 1},
        {"type": "credit", "ctx": 1, "ahead_ms": 3000, "played_ms": 1200},
        {"type": "ping", "t": 12.5, "box_load": 0.3}, {"type": "bye", "reason": "stop"},
    ]
    for f in frames:
        assert check_client_frame(json.loads(rig.dumps(f))) is None, f
    assert check_client_frame({"type": "cancel"}) is not None
    assert check_client_frame({"type": "tts.open", "ctx": 70000, "gen": 0, "voice": "x"}) is not None


def _literal(tree: ast.Module, name: str):
    for node in tree.body:
        target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
        if isinstance(target, ast.Name) and target.id == name:
            return node.value
    raise KeyError(name)


@pytest.mark.skipif(not (SPEECHFORGE / "src" / "speechforge" / "protocol.py").exists(),
                    reason="no SpeechForge checkout on this machine")
def test_golden_table_matches_the_speechforge_source():
    """Read SpeechForge's protocol.py as text: its constants, error codes and the
    client-frame table must equal the golden data the add-on is tested against."""
    tree = ast.parse((SPEECHFORGE / "src" / "speechforge" / "protocol.py").read_text(encoding="utf-8"))
    c = GOLDEN["constants"]
    for name in ("MAGIC", "KIND_MIC", "KIND_TTS", "MIC_RATE", "TTS_RATE", "MIC_FRAME_SAMPLES",
                 "TTS_FRAME_SAMPLES", "U16", "U32"):
        assert ast.literal_eval(_literal(tree, name)) == c[name], name
    assert ast.literal_eval(_literal(tree, "PROTO")) == GOLDEN["proto"]
    assert ast.literal_eval(_literal(tree, "ERROR_CODES")) == tuple(GOLDEN["error_codes"])
    header = _literal(tree, "HEADER")
    assert isinstance(header, ast.Call) and ast.literal_eval(header.args[0]) == c["HEADER"]
    frames = _literal(tree, "CLIENT_FRAMES")
    got: dict = {}
    for k, v in zip(frames.keys, frames.values, strict=True):
        fields = {}
        for fk, fv in zip(v.keys, v.values, strict=True):
            if isinstance(fv, ast.Name):            # _NUM = (int, float)
                fields[ast.literal_eval(fk)] = ["int", "float"]
            else:
                fields[ast.literal_eval(fk)] = [e.id for e in fv.elts]
        got[ast.literal_eval(k)] = fields
    assert got == GOLDEN["client_frames"]
