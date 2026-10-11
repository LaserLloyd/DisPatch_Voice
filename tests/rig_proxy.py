"""A loopback TCP forwarder in front of a SpeechForge rig, for the harnesses' forced
drop: ``cut()`` aborts every connection through it (the sf.1 socket and any voice
upload) and refuses new ones for a while, exactly what a rig that vanishes looks
like from the box. Bytes are forwarded untouched (the WebSocket upgrade, the token
header and the frames stay end to end).

    with RigProxy("<rig-host>", 1236) as p:
        url = f"ws://127.0.0.1:{p.port}/v1/realtime"
        ...
        p.cut(refuse_s=3.0)
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from typing import Self


class RigProxy:
    def __init__(self, host: str, port: int, *, listen_port: int = 0) -> None:
        self.host, self.target_port = host, port
        self.port = listen_port
        self.loop = asyncio.new_event_loop()
        self._writers: set[asyncio.StreamWriter] = set()
        self._refuse_until = 0.0
        self.cuts: list[float] = []
        self.accepted = 0
        self.refused = 0
        self._thread = threading.Thread(target=self.loop.run_forever, name="rig-proxy", daemon=True)
        self._server = None

    def __enter__(self) -> Self:
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._start(), self.loop)
        fut.result(10)
        return self

    def __exit__(self, *exc) -> None:
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(self._stop(), self.loop).result(5)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(5)

    async def _start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", self.port)
        self.port = self._server.sockets[0].getsockname()[1]

    async def _stop(self) -> None:
        self._abort_all()
        if self._server is not None:
            self._server.close()

    def _abort_all(self) -> None:
        for w in list(self._writers):
            with contextlib.suppress(Exception):
                w.transport.abort()
        self._writers.clear()

    def cut(self, refuse_s: float = 3.0) -> None:
        """Drop every connection now; refuse new ones for ``refuse_s``. Thread-safe."""
        self.cuts.append(time.monotonic())

        def do() -> None:
            self._refuse_until = time.monotonic() + refuse_s
            self._abort_all()
        self.loop.call_soon_threadsafe(do)

    async def _handle(self, r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        if time.monotonic() < self._refuse_until:
            self.refused += 1
            w.transport.abort()
            return
        try:
            ur, uw = await asyncio.wait_for(asyncio.open_connection(self.host, self.target_port), 5)
        except Exception:  # noqa: BLE001 - the rig itself is unreachable
            w.transport.abort()
            return
        self.accepted += 1
        self._writers.update((w, uw))

        async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
            try:
                with contextlib.suppress(Exception):          # either side went away
                    while data := await src.read(65536):
                        dst.write(data)
                        await dst.drain()
            finally:
                with contextlib.suppress(Exception):
                    dst.transport.abort()

        await asyncio.gather(pipe(r, uw), pipe(ur, w))
        self._writers.discard(w)
        self._writers.discard(uw)
