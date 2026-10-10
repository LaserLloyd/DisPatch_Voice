"""Voice wire protocol v3: binary framing (pure; contracts C-§C.2, B.3).

Uplink (client -> server)::

    0x03 | codec u8 | flags u8 | off u32 LE | count u8 | count x (len u16 LE + bytes)

Downlink (server -> client)::

    0x03 | codec u8 | kind u8 | gen u16 LE | seq u32 LE | t u32 LE | count u8 | count x (len u16 LE + bytes)

``codec``: 0 = PCM16 LE mono, 1 = Opus. Uplink ``off`` is the 16 kHz sample index of
the first sample, counted from stream start; it wraps at 2**32, as do ``seq`` and
``t`` (server monotonic ms). ``gen`` is carried as u16 (the client compares
``gen & 0xFFFF``). Packers mask those four counters instead of raising, so the
session can pass its unbounded Python ints; every other out-of-range value raises
``ValueError``. Unpackers raise ``ValueError`` on anything malformed: the session
answers ``error {code: "bad_frame"}`` and drops the message.

A binary message whose first byte is not ``0x03`` is a v2 raw PCM16 frame; use
:func:`is_v3` to tell them apart before calling :func:`unpack_up`.
"""
from __future__ import annotations

import struct

MAGIC = 0x03

CODEC_PCM16 = 0
CODEC_OPUS = 1
CODECS = {"pcm16": CODEC_PCM16, "opus": CODEC_OPUS}
CODEC_NAMES = {v: k for k, v in CODECS.items()}

KIND_REPLY = 0
KIND_ACK = 1
KIND_NOTICE = 2
KIND_PROBE = 3
KIND_BACKCHANNEL = 4
KINDS = {"reply": KIND_REPLY, "ack": KIND_ACK, "notice": KIND_NOTICE,
         "probe": KIND_PROBE, "backchannel": KIND_BACKCHANNEL}
KIND_NAMES = {v: k for k, v in KINDS.items()}

FLAG_LOCAL_SPEAKING = 0x01      # the phone's own voice is playing
FLAG_REPLAY = 0x02              # the message comes from the client's uplink ring

#: An uplink binary message larger than this is dropped BEFORE it is queued (A11).
MAX_MESSAGE = 65_536
#: Packets per uplink message: 5 live (100 ms of Opus), up to 50 when replaying.
MAX_UP_PACKETS = 50
#: Packets per downlink message the server sends: 10 Opus packets (200 ms).
MAX_DOWN_PACKETS = 10
MAX_PACKET = 0xFFFF

_UP = struct.Struct("<BBBIB")           # magic, codec, flags, off, count  (8 B)
_DOWN = struct.Struct("<BBBHIIB")       # magic, codec, kind, gen, seq, t, count  (14 B)
_LEN = struct.Struct("<H")
UP_HEADER = _UP.size
DOWN_HEADER = _DOWN.size

U32 = 0xFFFFFFFF
U16 = 0xFFFF


def is_v3(buf: bytes | bytearray | memoryview) -> bool:
    """True when ``buf`` starts with the v3 magic byte (else it is v2 raw PCM16)."""
    return len(buf) > 0 and buf[0] == MAGIC


def _pack_packets(codec: int, packets: list[bytes], max_count: int, *, pcm_even: bool) -> bytes:
    if not isinstance(packets, (list, tuple)):
        raise TypeError("packets must be a list of bytes")
    if not 1 <= len(packets) <= max_count:
        raise ValueError(f"packet count {len(packets)} outside 1..{max_count}")
    out = []
    for p in packets:
        n = len(p)
        if not 1 <= n <= MAX_PACKET:
            raise ValueError(f"packet length {n} outside 1..{MAX_PACKET}")
        if pcm_even and codec == CODEC_PCM16 and n % 2:
            raise ValueError("PCM16 packet with an odd byte count")
        out.append(_LEN.pack(n))
        out.append(bytes(p))
    return b"".join(out)


def _unpack_packets(buf: memoryview, pos: int, count: int, *, pcm_even: bool) -> list[bytes]:
    end = len(buf)
    packets = []
    for i in range(count):
        if pos + 2 > end:
            raise ValueError(f"truncated: packet {i} length missing")
        (n,) = _LEN.unpack_from(buf, pos)
        pos += 2
        if n == 0:
            raise ValueError(f"packet {i} is empty")
        if pos + n > end:
            raise ValueError(f"truncated: packet {i} wants {n} B, {end - pos} left")
        if pcm_even and n % 2:
            raise ValueError(f"PCM16 packet {i} with an odd byte count")
        packets.append(bytes(buf[pos:pos + n]))
        pos += n
    if pos != end:
        raise ValueError(f"{end - pos} trailing bytes after {count} packets")
    return packets


def _check_codec(codec: int) -> None:
    if codec not in CODEC_NAMES:
        raise ValueError(f"unknown codec {codec!r}")


def pack_up(codec: int, flags: int, off: int, packets: list[bytes]) -> bytes:
    """Build one uplink message. ``off`` is masked to u32."""
    _check_codec(codec)
    if not 0 <= int(flags) <= 0xFF:
        raise ValueError(f"flags {flags!r} outside u8")
    body = _pack_packets(codec, packets, MAX_UP_PACKETS, pcm_even=True)
    msg = _UP.pack(MAGIC, codec, int(flags), int(off) & U32, len(packets)) + body
    if len(msg) > MAX_MESSAGE:
        raise ValueError(f"uplink message of {len(msg)} B exceeds {MAX_MESSAGE}")
    return msg


def unpack_up(buf: bytes | bytearray | memoryview) -> tuple[int, int, int, list[bytes]]:
    """Parse one uplink message -> ``(codec, flags, off, packets)``; ``ValueError`` when bad.

    Unknown flag bits are kept (reserved, forward compatible); unknown codecs,
    an empty or over-long packet list, zero-length or truncated packets, odd PCM
    lengths, trailing bytes and messages over :data:`MAX_MESSAGE` are refused."""
    mv = memoryview(buf)
    if len(mv) > MAX_MESSAGE:
        raise ValueError(f"uplink message of {len(mv)} B exceeds {MAX_MESSAGE}")
    if len(mv) < UP_HEADER:
        raise ValueError(f"truncated: {len(mv)} B is shorter than the {UP_HEADER} B header")
    magic, codec, flags, off, count = _UP.unpack_from(mv, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:02x}")
    _check_codec(codec)
    if not 1 <= count <= MAX_UP_PACKETS:
        raise ValueError(f"packet count {count} outside 1..{MAX_UP_PACKETS}")
    packets = _unpack_packets(mv, UP_HEADER, count, pcm_even=codec == CODEC_PCM16)
    return codec, flags, off, packets


def pack_down(codec: int, kind: int, gen: int, seq: int, t_ms: int, packets: list[bytes]) -> bytes:
    """Build one downlink message. ``gen`` is masked to u16, ``seq``/``t_ms`` to u32."""
    _check_codec(codec)
    if kind not in KIND_NAMES:
        raise ValueError(f"unknown kind {kind!r}")
    body = _pack_packets(codec, packets, 0xFF, pcm_even=kind != KIND_PROBE)
    return _DOWN.pack(MAGIC, codec, kind, int(gen) & U16, int(seq) & U32, int(t_ms) & U32,
                      len(packets)) + body


def unpack_down(buf: bytes | bytearray | memoryview) -> tuple[int, int, int, int, int, list[bytes]]:
    """Parse one downlink message -> ``(codec, kind, gen, seq, t_ms, packets)``; ``ValueError`` when bad.

    A probe (kind 3) carries random bytes, so the PCM even-length rule does not apply to it."""
    mv = memoryview(buf)
    if len(mv) < DOWN_HEADER:
        raise ValueError(f"truncated: {len(mv)} B is shorter than the {DOWN_HEADER} B header")
    magic, codec, kind, gen, seq, t_ms, count = _DOWN.unpack_from(mv, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:02x}")
    _check_codec(codec)
    if kind not in KIND_NAMES:
        raise ValueError(f"unknown kind {kind}")
    if count < 1:
        raise ValueError("no packets")
    packets = _unpack_packets(mv, DOWN_HEADER, count,
                              pcm_even=codec == CODEC_PCM16 and kind != KIND_PROBE)
    return codec, kind, gen, seq, t_ms, packets


def u32_diff(a: int, b: int) -> int:
    """``a - b`` on the u32 ring, as a signed 32-bit value (wrap-aware ordering)."""
    return ((int(a) - int(b) + 0x8000_0000) & U32) - 0x8000_0000
