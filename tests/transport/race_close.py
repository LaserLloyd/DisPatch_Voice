"""Stress: Transport.close() racing a worker-thread Opus encode / decode (review-T6 H1).

Run as a child process by ``test_lifetime.py`` so that a heap corruption (glibc
abort) or a SIGSEGV fails the test instead of killing pytest. Before the fix,
``natural 200`` aborted with ``malloc(): mismatching next->prev_size`` within the
first iterations and ``inner`` segfaulted.

    python tests/transport/race_close.py natural 200   # 2 s chunk || close() after 0.5 ms
    python tests/transport/race_close.py inner 3        # the destroy forced into the C-call window
    python tests/transport/race_close.py stuck 3        # ... and close() gives up waiting (CLOSE_WAIT_S)
    python tests/transport/race_close.py decode 50      # close() while decode_up runs in a thread

Prints one JSON line ``{"mode", "n", "results": {outcome: count}}`` and exits 0.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dispatch_voice import transport
from dispatch_voice.transport import Transport, opus, wire3


class WS:
    def __init__(self) -> None:
        self.after_close = 0
        self.tr: Transport | None = None

    async def send_text(self, s):
        if self.tr is not None and self.tr.closed and '"audio_end"' in s:
            self.after_close += 1
        await asyncio.sleep(0)

    async def send_bytes(self, b):
        await asyncio.sleep(0)


def tone(seconds: float, rate: int = 24000) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (0.3 * np.sin(2 * np.pi * 200 * t) * 32767).astype(np.int16)


async def one_encode(delay: float, seconds: float) -> str:
    ws = WS()
    tr = Transport(ws, proto=3, codec_out="opus")
    ws.tr = tr
    task = asyncio.create_task(tr.send_audio(kind="reply", gen=0, seq=1, pcm16=tone(seconds), t_ms=0))
    await asyncio.sleep(delay)
    await tr.close()
    try:
        await task
    except Exception as e:  # noqa: BLE001 - the outcome is the point
        return f"{type(e).__name__}:{e}"
    if ws.after_close:
        return "audio_end_after_close"
    # nothing may be (re)opened after close
    await tr.send_audio(kind="reply", gen=0, seq=2, pcm16=tone(0.5), t_ms=0)
    return "ok" if not tr._streams else "stream_after_close"


async def one_decode(delay: float) -> str:
    tr = Transport(None, proto=3, codec_in="opus")
    enc = opus.Encoder(16000, 16000, application="voip")
    x = (np.sin(np.arange(16000) / 5) * 8000).astype(np.int16)
    packets = [enc.encode(x[i:i + 320]) for i in range(0, 16000, 320)][:50]
    enc.close()
    msg = wire3.pack_up(wire3.CODEC_OPUS, 0, 0, packets)
    tr.decode_up(msg)                             # creates the decoder
    task = asyncio.create_task(asyncio.to_thread(tr.decode_up, msg))
    await asyncio.sleep(delay)
    await tr.close()
    try:
        await task
    except ValueError as e:
        return "ValueError" if "closed" in str(e) else f"ValueError:{e}"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}:{e}"
    return "ok"


async def main(mode: str, n: int) -> dict:
    results: dict[str, int] = {}
    if mode in ("inner", "stuck", "decode"):
        opus.available()
        lib = opus._LIB
        name = "opus_decode" if mode == "decode" else "opus_encode"
        orig = getattr(lib, name)
        pause = 0.001 if mode == "decode" else 0.01

        def wrapped(*a):
            time.sleep(pause)                     # widen the window inside the foreign call
            return orig(*a)
        setattr(lib, name, wrapped)
    if mode == "stuck":
        transport.CLOSE_WAIT_S = 0.05             # close() stops waiting for the lock mid-encode
    for _ in range(n):
        if mode == "natural":
            r = await one_encode(0.0005, 2.0)
        elif mode in ("inner", "stuck"):
            r = await one_encode(0.05, 1.0)
        else:
            r = await one_decode(0.01)
        results[r] = results.get(r, 0) + 1
    return {"mode": mode, "n": n, "results": results}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(main(sys.argv[1], int(sys.argv[2])))), flush=True)
