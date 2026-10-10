"""The shaping proxy behaves like the link it claims to be (rate, delay, stall, cut, blackout)."""
import asyncio
import contextlib
import time

import pytest
import shaper

from .conftest import free_ports


class Sink:
    """A TCP server that records (arrival time, bytes) and can send to the client."""

    def __init__(self):
        self.chunks: list[tuple[float, int]] = []
        self.writers: list[asyncio.StreamWriter] = []
        self.closed = asyncio.Event()

    async def handle(self, reader, writer):
        self.writers.append(writer)
        try:
            while data := await reader.read(65536):
                self.chunks.append((time.monotonic(), len(data)))
        except ConnectionError:
            pass
        self.closed.set()

    @property
    def total(self):
        return sum(n for _, n in self.chunks)


@contextlib.asynccontextmanager
async def rig(profile_or_links):
    sink = Sink()
    sport, pport = free_ports(2)
    server = await asyncio.start_server(sink.handle, "127.0.0.1", sport)
    if isinstance(profile_or_links, str):
        proxy = shaper.make_proxy(profile_or_links, pport, sport)
    else:
        up, down = profile_or_links
        proxy = shaper.Proxy(pport, sport, up, down)
    await proxy.start()
    try:
        yield proxy, sink, pport
    finally:
        await proxy.stop()
        server.close()


def test_profiles_are_the_seven_of_the_contract():
    assert shaper.ORDER == ["baseline", "lte-edge", "derp-busy", "3g-weak", "edge", "blackout-20s", "stall-5s"]
    assert set(shaper.PROFILES) == set(shaper.ORDER)
    assert all(p["about"] for p in shaper.PROFILES.values())


async def test_rate_and_delay_are_honoured():
    async with rig((shaper.Link(200, 100), shaper.Link(None))) as (_proxy, sink, port):
        _, w = await asyncio.open_connection("127.0.0.1", port)
        t0 = time.monotonic()
        w.write(b"\x00" * 25_000)                       # 25 KB at 200 kbps + 134 B/1280 B overhead
        await w.drain()
        while sink.total < 25_000:
            await asyncio.sleep(0.01)
        took = time.monotonic() - t0
        expect = (25_000 + 134 * 20) * 8 / 200_000 + 0.1
        assert took == pytest.approx(expect, rel=0.12), took
        first = sink.chunks[0][0] - t0
        assert first >= 0.1                              # the one-way delay
        w.close()


async def test_stall_holds_bytes_then_delivers_them():
    async with rig("baseline") as (proxy, sink, port):
        _, w = await asyncio.open_connection("127.0.0.1", port)
        proxy.stall(0.6)
        t0 = time.monotonic()
        w.write(b"x" * 1000)
        await w.drain()
        while sink.total < 1000:
            await asyncio.sleep(0.01)
        assert time.monotonic() - t0 >= 0.55            # held for the stall, then delivered
        w.close()


async def test_cut_drops_both_sides():
    async with rig("baseline") as (proxy, sink, port):
        r, w = await asyncio.open_connection("127.0.0.1", port)
        w.write(b"hello")
        await w.drain()
        await asyncio.sleep(0.05)
        proxy.cut()
        await asyncio.wait_for(sink.closed.wait(), 2)
        with pytest.raises((ConnectionError, asyncio.IncompleteReadError)):
            data = await asyncio.wait_for(r.read(10), 2)
            if not data:
                raise ConnectionResetError
        assert proxy.stats()["events"][-1][1] == "cut"


async def test_blackout_refuses_new_connections_and_loses_what_a_closed_socket_queued():
    async with rig("baseline") as (proxy, sink, port):
        _, w = await asyncio.open_connection("127.0.0.1", port)
        proxy.blackout(0.8)
        w.write(b"y" * 4000)                            # sent into a dead radio ...
        await w.drain()
        await asyncio.sleep(0.2)
        w.transport.abort()                              # ... and the phone gives up on the socket
        # a new connection during the blackout is refused
        r2, _w2 = await asyncio.open_connection("127.0.0.1", port)
        assert await asyncio.wait_for(r2.read(10), 2) == b""
        assert proxy.refused == 1
        await asyncio.sleep(0.8)
        assert sink.total == 0                           # nothing of the dead socket ever arrived
        # after the blackout connections work again
        _, w3 = await asyncio.open_connection("127.0.0.1", port)
        w3.write(b"z" * 10)
        await w3.drain()
        for _ in range(100):
            if sink.total >= 10:
                break
            await asyncio.sleep(0.01)
        assert sink.total == 10
        w3.close()


async def test_an_aborted_full_connection_does_not_starve_the_next_one():
    """review-T3s4 #6: the client gives up on a connection whose downlink queue is full (its
    dead-socket rule) and reconnects. The old connection's unsent bytes and booked airtime
    leave the SHARED link with it: the reconnect gets the server's upgrade answer and its
    next line promptly (it got one line after 5 s, then nothing, and the run timed out)."""
    conns: list[asyncio.StreamWriter] = []

    async def handle(reader, writer):
        n = len(conns)
        conns.append(writer)
        if n == 0:                                    # the old session: far faster than the link
            with contextlib.suppress(Exception):
                for _ in range(200):
                    writer.write(b"A" * 4096)
                    await writer.drain()
                    await asyncio.sleep(0.01)
        else:
            writer.write(b"UPGRADE\n")
            await writer.drain()
            await asyncio.sleep(0.2)
            writer.write(b"READY\n")
            await writer.drain()
            await asyncio.sleep(5)

    sport, pport = free_ports(2)
    server = await asyncio.start_server(handle, "127.0.0.1", sport)
    proxy = shaper.Proxy(pport, sport, shaper.Link(None), shaper.Link(120, 50))   # EDGE downlink
    await proxy.start()
    try:
        r1, w1 = await asyncio.open_connection("127.0.0.1", pport)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.5:            # read slowly, then give up on the socket
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(r1.read(1024), 0.3)
        assert proxy.down.queued > 0
        w1.transport.abort()
        for _ in range(100):
            if proxy.down.queued == 0:
                break
            await asyncio.sleep(0.02)
        assert proxy.down.queued == 0                 # nothing phantom left on the shared link
        r2, w2 = await asyncio.open_connection("127.0.0.1", pport)
        t1 = time.monotonic()
        assert await asyncio.wait_for(r2.readline(), 5) == b"UPGRADE\n"
        assert await asyncio.wait_for(r2.readline(), 5) == b"READY\n"
        assert time.monotonic() - t1 < 2.0
        w2.transport.abort()
    finally:
        await proxy.stop()
        server.close()
