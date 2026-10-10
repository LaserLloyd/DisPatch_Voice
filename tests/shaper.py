"""Userspace link shaper: a TCP proxy that models a thin, slow, stalling pipe (owner: T6).

The single source of the low-bandwidth e2e matrix (contracts §G): :data:`PROFILES`
names ``baseline, lte-edge, derp-busy, 3g-weak, edge, blackout-20s, stall-5s``.

Root-free (no tc/netem): it runs in CI and on boxes where automation has no sudo.
CDP throttling is NOT a substitute — it let a WebSocket run at 19.6 Mbps under a
250 kbps setting (measured in the design prototype). Sits between a client (browser or test script) and
a server; each direction is a :class:`Link` with:

  kbps        capacity (None = unlimited). Each segment (<= ``seg`` bytes) costs
              (len + overhead) x 8 / kbps of serialisation, which is how a WebSocket
              byte stream behaves on a real link: TCP/IP + TLS + WireGuard headers
              ride along (134 B per packet, inner MTU 1280).
  delay_ms    one-way propagation delay, plus uniform jitter.
  queue_cap   bytes the link holds before it stops reading from the sender (a modem
              buffer); past it TCP flow control pushes back into the sender's
              socket, exactly like a phone on a weak link.

Events (:class:`Proxy`):

  stall(s)    freeze departures both ways (a handover, a tunnel). Bytes wait, like
              TCP data waiting for retransmission; they arrive when it ends.
  cut()       drop every connection now (RST both ways; queued bytes are lost).
  blackout(s) no signal for ``s``: a stall, new connections are refused, and a
              connection that either end gives up on during it is torn down
              WITHOUT delivering what it had queued (a closed socket's unsent data
              never leaves a phone with no signal).

    proxy = make_proxy("3g-weak", listen=19056, target=19055)
    await proxy.start(); ...; await proxy.stop()

CLI (for manual and browser runs; commands on stdin: ``stall N``, ``cut``,
``blackout N``, ``stats``)::

    python tests/shaper.py --profile edge --listen 19056 --target 19055
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import random
import socket
import sys
from dataclasses import dataclass, field

# name: link (up kbps, down kbps, one-way delay ms, jitter ms) + scripted events.
# Event anchors (used by tests/e2e_lowbw.py): "speech1" = the first utterance starts,
# "reply" = the first reply audio/say arrives. (anchor, seconds after it, action, arg)
PROFILES: dict[str, dict] = {
    "baseline":     {"up": None, "down": None, "delay_ms": 0, "jitter_ms": 0, "events": [],
                     "about": "the proxy with no shaping (the harness itself)"},
    "lte-edge":     {"up": 600, "down": 1500, "delay_ms": 125, "jitter_ms": 20, "events": [],
                     "about": "LTE at a cell edge"},
    "derp-busy":    {"up": 1000, "down": 1000, "delay_ms": 75, "jitter_ms": 50, "events": [],
                     "about": "relayed through a busy Tailscale DERP"},
    "3g-weak":      {"up": 250, "down": 250, "delay_ms": 250, "jitter_ms": 40, "events": [],
                     "about": "weak 3G: PCM cannot keep up; Opus fits, the server voice may not"},
    "edge":         {"up": 120, "down": 120, "delay_ms": 450, "jitter_ms": 80, "events": [],
                     "about": "EDGE: raw PCM loses the socket within a turn"},
    "blackout-20s": {"up": 600, "down": 1500, "delay_ms": 125, "jitter_ms": 20,
                     "events": [("speech1", 1.2, "blackout", 20.0)],
                     "about": "lte-edge, then 20 s with no signal in the middle of an utterance"},
    "stall-5s":     {"up": 600, "down": 1500, "delay_ms": 125, "jitter_ms": 20,
                     "events": [("reply", 4.5, "stall", 5.0), ("reply", 9.5, "cut", None)],
                     "about": "lte-edge, then a 5 s handover stall mid-reply after which the old path "
                              "is gone: the client rejoins while the reply still streams"},
}
ORDER = ["baseline", "lte-edge", "derp-busy", "3g-weak", "edge", "blackout-20s", "stall-5s"]

MAX_SAMPLES = 20_000


@dataclass
class Link:
    kbps: float | None = None
    delay_ms: float = 0.0
    jitter_ms: float = 0.0
    overhead: int = 134
    seg: int = 1280
    queue_cap: int = 48 * 1024
    name: str = ""
    # stats
    bytes: int = 0
    max_queued: int = 0
    max_queue_delay_s: float = 0.0
    samples: list = field(default_factory=list)      # (t, queued_bytes, queue_delay_s)
    # state (shared by every connection of the proxy, like one radio)
    queued: int = 0
    last_dep: float = 0.0
    stall_until: float = 0.0

    def stalled(self, now: float) -> bool:
        return now < self.stall_until

    async def run(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, t0: float,
                  pair: _Pair) -> None:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[tuple[float, bytes, float] | None] = asyncio.Queue()
        space = asyncio.Event()
        space.set()
        mine = [0]                                    # bytes of this connection still queued
        mine_air = [0.0]                              # ... and the airtime they were given

        async def drain():
            try:
                while True:
                    item = await q.get()
                    if item is None:
                        break
                    due, data, air = item
                    while True:                       # a stall that started later still holds it
                        now = loop.time()
                        wait = max(due, self.stall_until) - now
                        if wait <= 0:
                            break
                        await asyncio.sleep(wait)
                    if pair.dead:
                        break
                    writer.write(data)
                    await writer.drain()
                    self.queued -= len(data)
                    mine[0] -= len(data)
                    mine_air[0] -= air
                    if self.queued < self.queue_cap // 2:
                        space.set()
            except (ConnectionError, OSError):
                pass
            finally:
                # Whatever never left is gone — its bytes AND the airtime it was booked for
                # (review-T3s4 #6: left in the SHARED `queued`/`last_dep`, a connection the
                # client aborted with a full queue starved the next one, its reconnect).
                self.queued -= mine[0]
                mine[0] = 0
                if mine_air[0] > 0:
                    self.last_dep = max(loop.time(), self.last_dep - mine_air[0])
                mine_air[0] = 0.0
                space.set()
                if pair.dead:
                    pair.abort()
                else:
                    with contextlib.suppress(Exception):
                        writer.close()

        dt = asyncio.create_task(drain())
        try:
            while not dt.done():
                data = await reader.read(16384)
                if not data:
                    # the sender closed during a blackout: nothing it queued ever gets out (a
                    # plain stall delivers it afterwards, as TCP would after a handover)
                    if pair.proxy.blackout_active():
                        pair.dead = True
                    break
                for i in range(0, len(data), self.seg):
                    if dt.done():
                        break                         # the far side is gone: nothing can be sent
                    seg = data[i:i + self.seg]
                    now = loop.time()
                    dep = max(now, self.last_dep, self.stall_until)
                    air = (len(seg) + self.overhead) * 8 / (self.kbps * 1000) if self.kbps else 0.0
                    dep += air
                    self.last_dep = dep
                    due = dep + self.delay_ms / 1000 + (random.uniform(0, self.jitter_ms) / 1000
                                                        if self.jitter_ms else 0)
                    self.bytes += len(seg)
                    self.queued += len(seg)
                    mine[0] += len(seg)
                    mine_air[0] += air
                    qd = due - now
                    self.max_queued = max(self.max_queued, self.queued)
                    self.max_queue_delay_s = max(self.max_queue_delay_s, qd)
                    if len(self.samples) < MAX_SAMPLES:
                        self.samples.append((round(now - t0, 3), self.queued, round(qd, 3)))
                    q.put_nowait((due, seg, air))
                    if self.queued >= self.queue_cap:
                        space.clear()
                        await space.wait()           # backpressure: stop reading
        except (ConnectionError, asyncio.CancelledError, OSError):
            pass
        finally:
            q.put_nowait(None)
            if pair.dead:
                dt.cancel()
            with contextlib.suppress(BaseException):
                await dt

    def stats(self) -> dict:
        return {"name": self.name, "kbps": self.kbps, "delay_ms": self.delay_ms, "bytes": self.bytes,
                "max_queued_bytes": self.max_queued, "max_queue_delay_s": round(self.max_queue_delay_s, 3)}


@dataclass(eq=False)                       # identity hash: kept in a set
class _Pair:
    proxy: Proxy
    client: asyncio.StreamWriter
    server: asyncio.StreamWriter
    dead: bool = False

    def abort(self) -> None:
        self.dead = True
        for w in (self.client, self.server):
            with contextlib.suppress(Exception):
                w.transport.abort()


class Proxy:
    def __init__(self, listen: int, target: int, up: Link, down: Link, rcvbuf: int = 32768,
                 target_host: str = "127.0.0.1") -> None:
        self.listen = listen
        self.target = target
        self.target_host = target_host
        self.up = up
        self.down = down
        self.up.name = self.up.name or "up"
        self.down.name = self.down.name or "down"
        self.rcvbuf = rcvbuf
        self._server = None
        self._pairs: set[_Pair] = set()
        self._tasks: set[asyncio.Task] = set()
        self._refuse_until = 0.0
        self.t0 = 0.0
        self.connections = 0
        self.refused = 0
        self.events: list[tuple[float, str, float | None]] = []

    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    def blackout_active(self) -> bool:
        try:
            now = self._now()
        except RuntimeError:
            return False
        return now < self._refuse_until

    async def _handle(self, reader, writer):
        if self._now() < self._refuse_until:
            self.refused += 1
            with contextlib.suppress(Exception):
                writer.transport.abort()
            return
        try:
            tr, tw = await asyncio.open_connection(self.target_host, self.target)
        except OSError:
            writer.close()
            return
        self.connections += 1
        pair = _Pair(self, writer, tw)
        self._pairs.add(pair)
        a = asyncio.create_task(self.up.run(reader, tw, self.t0, pair))
        b = asyncio.create_task(self.down.run(tr, writer, self.t0, pair))
        self._tasks.update({a, b})
        try:
            await asyncio.gather(a, b, return_exceptions=True)
        finally:
            self._tasks.difference_update({a, b})
            self._pairs.discard(pair)

    def _log(self, what: str, arg: float | None = None) -> None:
        self.events.append((round(self._now() - self.t0, 3), what, arg))

    def cut(self) -> None:
        """Drop every connection now (RST both ways). The client sees a close/1006."""
        self._log("cut")
        for p in list(self._pairs):
            p.abort()
        self._pairs.clear()

    def stall(self, seconds: float) -> None:
        """Freeze departures in both directions for ``seconds`` (queued bytes wait)."""
        self._log("stall", seconds)
        until = self._now() + seconds
        self.up.stall_until = max(self.up.stall_until, until)
        self.down.stall_until = max(self.down.stall_until, until)

    def blackout(self, seconds: float) -> None:
        """No signal for ``seconds``: stall + refuse new connections + lose what a closed socket queued."""
        self._log("blackout", seconds)
        until = self._now() + seconds
        self._refuse_until = max(self._refuse_until, until)
        self.up.stall_until = max(self.up.stall_until, until)
        self.down.stall_until = max(self.down.stall_until, until)

    def apply(self, action: str, arg: float | None) -> None:
        if action == "stall":
            self.stall(float(arg))
        elif action == "cut":
            self.cut()
        elif action == "blackout":
            self.blackout(float(arg))
        else:
            raise ValueError(f"unknown shaper action {action!r}")

    async def start(self) -> None:
        self.t0 = self._now()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # A small receive window on the client-facing side, so the client's kernel cannot
        # hide seconds of audio in its buffers (the backlog shows up as RTT, as on a phone).
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf)
        sock.bind(("127.0.0.1", self.listen))
        sock.listen(16)
        sock.setblocking(False)
        self._server = await asyncio.start_server(self._handle, sock=sock)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            for p in list(self._pairs):
                p.abort()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._server.wait_closed(), 2.0)
            for t in list(self._tasks):
                t.cancel()

    def stats(self) -> dict:
        return {"up": self.up.stats(), "down": self.down.stats(), "connections": self.connections,
                "refused": self.refused, "events": self.events}


def make_proxy(name: str, listen: int, target: int) -> Proxy:
    p = PROFILES[name]
    return Proxy(listen, target,
                 Link(p["up"], p["delay_ms"], p["jitter_ms"], name="up"),
                 Link(p["down"], p["delay_ms"], p["jitter_ms"], name="down"))


async def _cli(args) -> None:
    proxy = make_proxy(args.profile, args.listen, args.target)
    await proxy.start()
    print(f"shaper {args.profile}: 127.0.0.1:{args.listen} -> 127.0.0.1:{args.target} "
          f"({PROFILES[args.profile]['about']}); stdin: stall N | cut | blackout N | stats", flush=True)
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    try:
        while True:
            line = (await reader.readline()).decode().strip()
            if not line:
                if reader.at_eof():
                    await asyncio.Event().wait()
                continue
            cmd, *rest = line.split()
            try:
                if cmd == "stats":
                    print(json.dumps(proxy.stats()), flush=True)
                else:
                    proxy.apply(cmd, float(rest[0]) if rest else None)
                    print("ok", cmd, *rest, flush=True)
            except (ValueError, IndexError) as e:
                print("error:", e, flush=True)
    finally:
        await proxy.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", choices=ORDER, default="baseline")
    ap.add_argument("--listen", type=int, required=True)
    ap.add_argument("--target", type=int, required=True)
    args = ap.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_cli(args))


if __name__ == "__main__":
    main()
