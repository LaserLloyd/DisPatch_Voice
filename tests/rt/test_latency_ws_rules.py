"""``tests/latency_ws.py`` row rules (V1): the acknowledgement line is timed from the
`eot` of the turn it covers, i.e. the LAST `eot` before it, so a turn that ended early
and was resumed (two `eot`s, one merged submit) is not reported as a +5 s ack."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # tests/: latency_ws
latency_ws = pytest.importorskip("latency_ws")


def _client(frames):
    c = latency_ws.Client(None, 0.0)
    c.frames = frames
    return c


def _row():
    return latency_ws.Row(scenario="turn", clip="medium", load="idle", profile="balanced", max_mhz=5090, load1="1")


def test_ack_is_timed_from_the_last_eot_before_it():
    frames = [(1.0, {"type": "eot"}), (1.1, {"type": "transcript", "text": "Can you remind me"}),
              (5.3, {"type": "eot"}), (5.32, {"type": "transcript", "text": "…tomorrow?", "merged": True}),
              (6.0, {"type": "say", "kind": "ack", "seq": 1}), (7.5, {"type": "eot"})]
    row = _row()
    latency_ws.fill_metrics(row, _client(frames))
    assert row.ack_after_eot_s == pytest.approx(0.7)


def test_one_eot_and_no_ack():
    row = _row()
    latency_ws.fill_metrics(row, _client([(1.0, {"type": "eot"}), (2.0, {"type": "say", "kind": "reply"})]))
    assert row.ack_after_eot_s is None
    row = _row()
    latency_ws.fill_metrics(row, _client([(1.0, {"type": "eot"}), (1.6, {"type": "say", "kind": "ack"})]))
    assert row.ack_after_eot_s == pytest.approx(0.6)
