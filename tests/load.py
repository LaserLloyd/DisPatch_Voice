#!/usr/bin/env python3
"""CPU load generator for the latency harness's "under load" column.

    python tests/load.py 8 60          # 8 busy processes for 60 s (the measured setup)

    from load import cpu_load
    with cpu_load(8):                  # busy until the block exits
        run_the_measurement()

Each process spins pure-Python arithmetic, i.e. one fully busy core-thread,
the way a second voice chat or a browser tab competes for the box. Copied from
the feasibility prototype every "under load" number in the spec was taken with,
so the two stay comparable.
"""
from __future__ import annotations

import contextlib
import math
import multiprocessing as mp
import sys
import time


def spin(seconds: float) -> None:
    end = time.perf_counter() + seconds
    x = 0.0
    while time.perf_counter() < end:
        for i in range(20000):
            x += math.sqrt(i)


@contextlib.contextmanager
def cpu_load(n: int = 8, max_seconds: float = 3600.0):
    """Keep `n` processes busy for the duration of the block (capped at max_seconds)."""
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=spin, args=(max_seconds,), daemon=True) for _ in range(n)]
    for p in procs:
        p.start()
    try:
        yield procs
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(5)


def main(argv: list[str]) -> int:
    n = int(argv[1]) if len(argv) > 1 else 8
    t = float(argv[2]) if len(argv) > 2 else 60.0
    procs = [mp.Process(target=spin, args=(t,)) for _ in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
