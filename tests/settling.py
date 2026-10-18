"""Waiting in tests that is the same on every OS.

``asyncio.sleep(0.005)`` is NOT a 5 ms wait on Windows: the event loop fires every timer
that is due within its clock resolution (``time.get_clock_info("monotonic")`` is 15.6 ms
there) at once, so ``for _ in range(30): await asyncio.sleep(0.005)`` took ~0 s instead of
150 ms and the work it was waiting for (worker threads, the ack and merge timers) had not
happened yet. That was the cause of the session suites' Windows failures, not a slow box.

* :func:`pause` waits REAL time in a worker thread (the loop keeps running meanwhile),
  exact to the OS timer;
* :func:`settle` is the old "let the session work" pause: ``n`` x 5 ms of real time;
* :func:`wait_until` polls a condition with the same real-time step - prefer it wherever the
  test can say WHAT it is waiting for.
"""
from __future__ import annotations

import asyncio
import time


async def pause(seconds: float) -> None:
    """Really wait ``seconds`` (the loop stays free to run callbacks and timers)."""
    if seconds <= 0:
        await asyncio.sleep(0)
        return
    await asyncio.to_thread(time.sleep, seconds)


async def settle(n: int = 30, per: float = 0.005) -> None:
    """Give the code under test ``n * per`` seconds (150 ms by default) of real time. It is a
    pause, not a quiescence wait: several tests look at the session WHILE a slow step (a
    transcription, a submit) is still running, so waiting for it to finish would change what
    they test. A test that needs an outcome waits for it with :func:`wait_until`."""
    await pause(n * per)


async def wait_until(cond, timeout: float = 5.0, step: float = 0.01) -> bool:
    """Poll ``cond()`` until it is true or ``timeout`` passes; returns its last value."""
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() >= end:
            return bool(cond())
        await pause(step)
    return True
