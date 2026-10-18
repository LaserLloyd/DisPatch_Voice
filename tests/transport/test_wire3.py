"""wire3 framing against the shared fixture (tests/fixtures/wire3.json, also read by T8's JS tests)."""
import json
import struct

import pytest

from dispatch_voice.transport import wire3


@pytest.fixture(scope="module")
def vectors():
    from fakes import FIXTURES
    return json.loads((FIXTURES / "wire3.json").read_text(encoding="utf-8"))


def manual_up(codec, flags, off, packets):
    """Independent layout (no wire3 code): the bytes the contract describes."""
    out = bytes([0x03, codec, flags]) + (off & 0xFFFFFFFF).to_bytes(4, "little") + bytes([len(packets)])
    for p in packets:
        out += len(p).to_bytes(2, "little") + p
    return out


def manual_down(codec, kind, gen, seq, t, packets):
    out = bytes([0x03, codec, kind]) + (gen & 0xFFFF).to_bytes(2, "little") \
        + (seq & 0xFFFFFFFF).to_bytes(4, "little") + (t & 0xFFFFFFFF).to_bytes(4, "little") + bytes([len(packets)])
    for p in packets:
        out += len(p).to_bytes(2, "little") + p
    return out


def test_fixture_is_filled_and_covers_the_contract(vectors):
    assert vectors["status"] == "filled"
    assert set(vectors["schema"]) == {"up", "down", "bad_up", "bad_down"}
    assert {v["codec"] for v in vectors["up"]} == {0, 1}
    assert {v["codec"] for v in vectors["down"]} == {0, 1}
    assert {v["kind"] for v in vectors["down"]} == {0, 1, 2, 3, 4}
    assert any(v["off"] >= 0xFFFFFF00 for v in vectors["up"]), "an offset at the wrap"
    assert any(v["gen"] == 0xFFFF and v["seq"] == 0xFFFFFFFF for v in vectors["down"])
    assert any(len(v["packets"]) == 50 for v in vectors["up"]), "a full replay message"
    assert {"bad-magic", "short-header", "count-51"} <= {v["name"] for v in vectors["bad_up"]}
    assert {"bad-magic", "short-header"} <= {v["name"] for v in vectors["bad_down"]}
    assert all(v["why"] for v in vectors["bad_up"] + vectors["bad_down"])


def test_up_vectors_pack_and_unpack(vectors):
    for v in vectors["up"]:
        packets = [bytes.fromhex(p) for p in v["packets"]]
        want = bytes.fromhex(v["hex"])
        assert want == manual_up(v["codec"], v["flags"], v["off"], packets), v["name"]
        assert wire3.pack_up(v["codec"], v["flags"], v["off"], packets) == want, v["name"]
        assert wire3.unpack_up(want) == (v["codec"], v["flags"], v["off"], packets), v["name"]
        assert wire3.is_v3(want)


def test_down_vectors_pack_and_unpack(vectors):
    for v in vectors["down"]:
        packets = [bytes.fromhex(p) for p in v["packets"]]
        want = bytes.fromhex(v["hex"])
        assert want == manual_down(v["codec"], v["kind"], v["gen"], v["seq"], v["t"], packets), v["name"]
        assert wire3.pack_down(v["codec"], v["kind"], v["gen"], v["seq"], v["t"], packets) == want
        assert wire3.unpack_down(want) == (v["codec"], v["kind"], v["gen"], v["seq"], v["t"], packets)


def test_bad_vectors_raise(vectors):
    for v in vectors["bad_up"]:
        with pytest.raises(ValueError):
            wire3.unpack_up(bytes.fromhex(v["hex"]))
    for v in vectors["bad_down"]:
        with pytest.raises(ValueError):
            wire3.unpack_down(bytes.fromhex(v["hex"]))


def test_counters_are_masked_not_refused():
    msg = wire3.pack_up(1, 0, 2**32 + 5, [b"\x01"])
    assert wire3.unpack_up(msg)[2] == 5
    _codec, _kind, gen, seq, t, _ = wire3.unpack_down(wire3.pack_down(1, 0, 0x1_0002, 2**32 + 7, 2**33 + 9, [b"\x01"]))
    assert (gen, seq, t) == (2, 7, 9)


@pytest.mark.parametrize("args", [
    (2, 0, 0, [b"\x00\x00"]),            # unknown codec
    (0, 256, 0, [b"\x00\x00"]),          # flags beyond u8
    (0, 0, 0, []),                       # no packets
    (1, 0, 0, [b"\x01"] * 51),           # > 50 packets
    (1, 0, 0, [b""]),                    # empty packet
    (0, 0, 0, [b"\x00"]),                # odd PCM
    (1, 0, 0, [b"\x00" * 65536]),        # packet over u16
])
def test_pack_up_refuses(args):
    with pytest.raises(ValueError):
        wire3.pack_up(*args)


def test_pack_up_refuses_a_message_over_the_server_limit():
    # 50 x 1400 B = 70 KB > 65,536: the packer never builds what the server drops
    with pytest.raises(ValueError):
        wire3.pack_up(1, wire3.FLAG_REPLAY, 0, [b"\x01" * 1400] * 50)


def test_unpack_up_refuses_oversize_and_v2_raw():
    big = wire3.pack_up(0, 0, 0, [b"\x00\x00" * 100]) + b"\x00" * (wire3.MAX_MESSAGE)
    with pytest.raises(ValueError):
        wire3.unpack_up(big)
    raw_v2 = b"\x10\x00" * 320            # first byte 0x10: a v2 PCM frame
    assert not wire3.is_v3(raw_v2)
    with pytest.raises(ValueError):
        wire3.unpack_up(raw_v2)
    assert not wire3.is_v3(b"")


def test_pack_down_refuses():
    with pytest.raises(ValueError):
        wire3.pack_down(1, 5, 0, 0, 0, [b"\x01"])
    with pytest.raises(ValueError):
        wire3.pack_down(0, 0, 0, 0, 0, [b"\x01"])          # odd PCM reply
    assert wire3.pack_down(0, wire3.KIND_PROBE, 0, 0, 0, [b"\x01"])   # a probe may be odd
    with pytest.raises(ValueError):
        wire3.pack_down(1, 0, 0, 0, 0, [b"\x01"] * 256)


def test_header_sizes_and_overheads_match_the_budget():
    assert wire3.UP_HEADER == 8 and wire3.DOWN_HEADER == 14
    # 5 Opus packets of ~40 B per 100 ms: 8 + 5 x 2 framing bytes
    msg = wire3.pack_up(1, 0, 0, [b"\x00" * 40] * 5)
    assert len(msg) == 8 + 5 * 42
    assert struct.unpack_from("<I", msg, 3)[0] == 0


def test_accepts_memoryview_and_bytearray():
    msg = wire3.pack_up(1, 1, 99, [b"\x01\x02"])
    assert wire3.unpack_up(memoryview(msg)) == wire3.unpack_up(bytearray(msg)) == (1, 1, 99, [b"\x01\x02"])


def test_u32_diff_orders_across_the_wrap():
    assert wire3.u32_diff(5, 0xFFFFFFFB) == 10
    assert wire3.u32_diff(0xFFFFFFFB, 5) == -10
    assert wire3.u32_diff(7, 7) == 0
