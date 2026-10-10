"""Fixtures for tests/transport/ (owner: track T6). The shared doubles are in the root
conftest (fake_ctx, fake_engines, b1a_frames, fixtures_dir) and tests/fakes.py."""
import contextlib
import socket

import pytest


def free_ports(n: int, lo: int = 19059, hi: int = 19064) -> list[int]:
    """Loopback ports in this track's range when free (the box's port rule), else ephemeral ones (CI)."""
    got: list[int] = []
    for p in [*range(lo, hi + 1), *([0] * n)]:
        with contextlib.suppress(OSError), socket.socket() as s:
            s.bind(("127.0.0.1", p))
            got.append(s.getsockname()[1])
        if len(got) == n:
            return got
    raise RuntimeError("no free ports")


@pytest.fixture
def port_pair() -> list[int]:
    return free_ports(2)
