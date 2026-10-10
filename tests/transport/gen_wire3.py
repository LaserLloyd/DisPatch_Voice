"""Regenerate tests/fixtures/wire3.json (owner: T6). Dev tool, not collected.

    uv run python tests/transport/gen_wire3.py            # rewrite the fixture
    uv run python tests/transport/gen_wire3.py --check    # exit 1 if it would change

The Opus payloads are real libopus packets (16 kHz/16 kbps voip uplink, 24 kHz/24 kbps
audio downlink) of a deterministic tone, so the JS side can also try decoding them;
the frame tests never re-encode, they only check the framing bytes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dispatch_voice.transport import opus, wire3

OUT = ROOT / "tests" / "fixtures" / "wire3.json"

SCHEMA = {
    "up": "[{name, codec: 0|1, flags, off, packets: [hex], hex}] - pack_up(codec, flags, off, packets) == bytes.fromhex(hex) and unpack_up round-trips",
    "down": "[{name, codec: 0|1, kind: 0..4, gen, seq, t, packets: [hex], hex}] - pack_down / unpack_down",
    "bad_up": "[{name, hex, why}] - unpack_up raises ValueError (bad magic, short length, > 50 packets, ...)",
    "bad_down": "[{name, hex, why}] - unpack_down raises ValueError",
}


def tone(rate: int, seconds: float, hz: float) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (0.25 * np.sin(2 * np.pi * hz * t) * 32767).astype(np.int16)


def opus_packets(rate: int, kbps: int, app: str, n: int, hz: float) -> list[bytes]:
    enc = opus.Encoder(rate, kbps * 1000, application=app)
    x = tone(rate, n * 0.02, hz)
    return [enc.encode(x[i * enc.frame:(i + 1) * enc.frame]) for i in range(n)]


def pcm(n_samples: int, start: int = 0) -> bytes:
    return (np.arange(start, start + n_samples, dtype=np.int64) * 97 % 65536 - 32768).astype("<i2").tobytes()


def up(name, codec, flags, off, packets):
    return {"name": name, "codec": codec, "flags": flags, "off": off,
            "packets": [p.hex() for p in packets], "hex": wire3.pack_up(codec, flags, off, packets).hex()}


def down(name, codec, kind, gen, seq, t, packets):
    return {"name": name, "codec": codec, "kind": kind, "gen": gen, "seq": seq, "t": t,
            "packets": [p.hex() for p in packets],
            "hex": wire3.pack_down(codec, kind, gen, seq, t, packets).hex()}


def build() -> dict:
    up_opus = opus_packets(16000, 16, "voip", 12, 300.0)
    down_opus = opus_packets(24000, 24, "audio", 10, 220.0)
    ok_up_hdr = bytes([3, 1, 0]) + (1600).to_bytes(4, "little")
    one_pkt = len(up_opus[0]).to_bytes(2, "little") + up_opus[0]
    down_hdr = bytes([3, 1, 0]) + (1).to_bytes(2, "little") + (1).to_bytes(4, "little") + (5).to_bytes(4, "little")
    return {
        "$about": "Shared v3 binary-frame vectors (v3 contracts §C.2, §G), filled by T6 from "
                  "dispatch_voice.transport.wire3 (tests/transport/gen_wire3.py). T6's Python tests and "
                  "T8's JS tests read this same file. All integers little-endian. Uplink header 8 B: "
                  "0x03 codec flags off:u32 count; downlink header 14 B: 0x03 codec kind gen:u16 seq:u32 t:u32 "
                  "count; then count x (len:u16 + bytes). Packers mask off/seq/t to u32 and gen to u16.",
        "status": "filled",
        "libopus": opus.version(),
        "schema": SCHEMA,
        "up": [
            up("pcm16-live-20ms", 0, 0, 0, [pcm(320)]),
            up("pcm16-short", 0, 0, 320, [pcm(4)]),
            up("opus-live-100ms", 1, 0, 1600, up_opus[:5]),
            up("opus-local-speaking", 1, wire3.FLAG_LOCAL_SPEAKING, 3200, up_opus[5:10]),
            up("opus-replay", 1, wire3.FLAG_REPLAY, 48000, up_opus),
            up("opus-replay-local-speaking", 1, wire3.FLAG_REPLAY | wire3.FLAG_LOCAL_SPEAKING, 64000, up_opus[:2]),
            up("opus-reserved-flag-bits-kept", 1, 0xF0, 80000, up_opus[:1]),
            up("off-near-wrap", 1, 0, 0xFFFFFF00, up_opus[:5]),
            up("off-max", 0, 0, 0xFFFFFFFF, [pcm(2)]),
            up("replay-50-packets", 1, wire3.FLAG_REPLAY, 160000, (up_opus * 5)[:50]),
        ],
        "down": [
            down("opus-reply-200ms", 1, wire3.KIND_REPLY, 1, 1, 5, down_opus),
            down("opus-reply-first-message", 1, wire3.KIND_REPLY, 1, 2, 1210, down_opus[:3]),
            down("pcm16-reply-40ms", 0, wire3.KIND_REPLY, 2, 7, 60000, [pcm(960, 5)]),
            down("opus-ack", 1, wire3.KIND_ACK, 2, 8, 61000, down_opus[:4]),
            down("opus-notice", 1, wire3.KIND_NOTICE, 3, 9, 62000, down_opus[:2]),
            down("opus-backchannel", 1, wire3.KIND_BACKCHANNEL, 3, 10, 63000, down_opus[:1]),
            down("probe-odd-length", 0, wire3.KIND_PROBE, 0, 0, 64000, [bytes.fromhex("9f03c4e2b17a5d")]),
            down("counters-at-wrap", 1, wire3.KIND_REPLY, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, down_opus[:1]),
        ],
        "bad_up": [
            {"name": "bad-magic", "hex": (bytes([2]) + ok_up_hdr[1:] + b"\x01" + one_pkt).hex(),
             "why": "first byte is not 0x03 (that is a v2 raw PCM frame; unpack_up refuses it)"},
            {"name": "short-header", "hex": ok_up_hdr[:5].hex(), "why": "shorter than the 8-byte header"},
            {"name": "count-zero", "hex": (ok_up_hdr + b"\x00").hex(), "why": "no packets"},
            {"name": "count-51", "hex": (ok_up_hdr + bytes([51]) + (b"\x01\x00\xaa" * 51)).hex(),
             "why": "more than 50 packets"},
            {"name": "unknown-codec", "hex": (bytes([3, 2, 0]) + ok_up_hdr[3:] + b"\x01" + one_pkt).hex(),
             "why": "codec 2 does not exist"},
            {"name": "missing-packet", "hex": (ok_up_hdr + b"\x02" + one_pkt).hex(),
             "why": "count says 2, only 1 packet present"},
            {"name": "truncated-length", "hex": (ok_up_hdr + b"\x01" + b"\x05").hex(),
             "why": "the u16 length is cut"},
            {"name": "truncated-payload", "hex": (ok_up_hdr + b"\x01" + one_pkt[:-3]).hex(),
             "why": "the packet is shorter than its length"},
            {"name": "trailing-bytes", "hex": (ok_up_hdr + b"\x01" + one_pkt + b"\x00").hex(),
             "why": "bytes after the last packet"},
            {"name": "zero-length-packet", "hex": (ok_up_hdr + b"\x01" + b"\x00\x00").hex(),
             "why": "a packet of 0 bytes"},
            {"name": "pcm-odd-length", "hex": (bytes([3, 0, 0]) + (0).to_bytes(4, "little") + b"\x01"
                                               + b"\x03\x00\x01\x02\x03").hex(),
             "why": "PCM16 packet with an odd byte count"},
        ],
        "bad_down": [
            {"name": "bad-magic", "hex": (bytes([4]) + down_hdr[1:] + b"\x01\x01\x00\xfc").hex(),
             "why": "first byte is not 0x03"},
            {"name": "short-header", "hex": down_hdr[:10].hex(), "why": "shorter than the 14-byte header"},
            {"name": "count-zero", "hex": (down_hdr + b"\x00").hex(), "why": "no packets"},
            {"name": "unknown-kind", "hex": (bytes([3, 1, 5]) + down_hdr[3:] + b"\x01\x01\x00\xfc").hex(),
             "why": "kind 5 does not exist"},
            {"name": "unknown-codec", "hex": (bytes([3, 7, 0]) + down_hdr[3:] + b"\x01\x01\x00\xfc").hex(),
             "why": "codec 7 does not exist"},
            {"name": "truncated-payload", "hex": (down_hdr + b"\x01\x08\x00\xfc\xfd").hex(),
             "why": "the packet is shorter than its length"},
            {"name": "trailing-bytes", "hex": (down_hdr + b"\x01\x01\x00\xfc\xff").hex(),
             "why": "bytes after the last packet"},
            {"name": "pcm-odd-length", "hex": (bytes([3, 0, 0]) + down_hdr[3:] + b"\x01\x03\x00\x01\x02\x03").hex(),
             "why": "PCM16 (non-probe) packet with an odd byte count"},
            {"name": "zero-length-packet", "hex": (down_hdr + b"\x01\x00\x00").hex(), "why": "a packet of 0 bytes"},
        ],
    }


def main() -> int:
    data = build()
    text = json.dumps(data, indent=1) + "\n"
    if "--check" in sys.argv:
        cur = json.loads(OUT.read_text())
        same = {k: v for k, v in cur.items() if k not in ("up", "down", "libopus")} == \
               {k: v for k, v in data.items() if k not in ("up", "down", "libopus")}
        print("framing vectors unchanged" if same else "wire3.json would change")
        return 0 if same else 1
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text)} chars, {len(data['up'])} up, {len(data['down'])} down, "
          f"{len(data['bad_up'])} bad up, {len(data['bad_down'])} bad down)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
