"""OffsetTracker: gaps, overlaps, the 5 s cap, the 2x-realtime zero-fill budget, the u32 wrap (C-§B.3, A11)."""
import numpy as np
import pytest

from dispatch_voice.transport.offsets import GapTooLarge, OffsetTracker

SR = 16000
F = 320                                   # one 20 ms frame


def frame(v=1, n=F):
    return np.full(n, v, np.int16)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def joined(parts):
    return np.concatenate(parts) if parts else np.zeros(0, np.int16)


def test_in_order_passes_through_and_tracks_rx_off():
    tr = OffsetTracker(clock=Clock())
    assert joined(tr.feed(1000, frame(1))).tolist() == [1] * F       # anchors anywhere
    assert joined(tr.feed(1000 + F, frame(2))).tolist() == [2] * F
    assert tr.expected == 1000 + 2 * F and tr.rx_off == 1000 + 2 * F
    assert tr.zero_filled == 0 and tr.dropped == 0


def test_gap_up_to_5s_is_zero_filled():
    clk = Clock()
    tr = OffsetTracker(clock=clk)
    tr.feed(0, frame(1))
    clk.t = 3.0                                                       # 3 s of real time pass (gated mic)
    out = tr.feed(F + 2 * SR, frame(2))                               # a 2 s gap
    assert len(out) == 2
    assert out[0].dtype == np.int16 and len(out[0]) == 2 * SR and not out[0].any()
    assert out[1].tolist() == [2] * F
    assert tr.zero_filled == 2 * SR
    # exactly 5 s is still filled
    clk.t = 10.0
    out = tr.feed(tr.expected + 5 * SR, frame(3))
    assert len(out[0]) == 5 * SR


def test_gap_over_5s_raises_with_the_frames_and_reanchors():
    tr = OffsetTracker(clock=Clock())
    tr.feed(0, frame(1))
    with pytest.raises(GapTooLarge) as ei:
        tr.feed(F + 5 * SR + 1, frame(7))
    err = ei.value
    assert err.reason == "gap" and err.gap == 5 * SR + 1
    assert joined(err.frames).tolist() == [7] * F                     # feed these after detector.reset()
    assert tr.resets == 1
    nxt = F + 5 * SR + 1 + F
    assert tr.expected == nxt and tr.rx_off == nxt
    assert joined(tr.feed(nxt, frame(8))).tolist() == [8] * F         # continues normally


def test_overlap_is_dropped_partly_or_wholly():
    tr = OffsetTracker(clock=Clock())
    tr.feed(0, frame(1, 4 * F))                                       # 0 .. 1280
    out = tr.feed(2 * F, np.arange(4 * F, dtype=np.int16))            # 640 .. 1920: first 640 overlap
    assert joined(out).tolist() == list(range(2 * F, 4 * F))
    assert tr.dropped == 2 * F and tr.expected == 6 * F
    assert tr.feed(F, frame(9)) == []                                 # a replay entirely before expected
    assert tr.dropped == 3 * F and tr.expected == 6 * F
    assert tr.rx_off == 6 * F                                         # rx_off never goes backwards


def test_replay_of_a_ring_then_live_yields_each_sample_once():
    """A reconnect replays the ring from `cursor.from`; overlapping live frames are dropped."""
    tr = OffsetTracker(clock=Clock())
    sig = np.arange(50 * F, dtype=np.int16)
    for i in range(20):                                            # live: frames 0..19
        tr.feed(i * F, sig[i * F:(i + 1) * F])
    got = []
    for i in range(15, 50, 5):                                        # replay from frame 15 in 100 ms messages
        got += tr.feed(i * F, sig[i * F:(i + 5) * F])
    assert joined(got).tolist() == sig[20 * F:].tolist()


def test_zero_fill_is_capped_at_2x_realtime():
    """off jumping 5 s every 100 ms (50x realtime of silence) must hit reset, not Silero."""
    clk = Clock()
    tr = OffsetTracker(clock=clk)
    tr.feed(0, frame())
    filled_ok, resets = 0, 0
    off = F
    for _ in range(100):                                              # 10 s of wall time
        clk.t += 0.1
        off += 5 * SR
        try:
            tr.feed(off, frame())
            filled_ok += 1
        except GapTooLarge as e:
            assert e.reason == "rate"
            resets += 1
        off += F
    assert resets > 0
    # budget: 5 s initial + 2x realtime over 10 s = 25 s of zero-fill at most
    assert tr.zero_filled <= (5 + 2 * 10) * SR
    assert filled_ok <= 5


def test_honest_gated_mic_never_runs_out_of_budget():
    """Speech bursts with real pauses (off and wall clock advance together) always fill."""
    clk = Clock()
    tr = OffsetTracker(clock=clk)
    off = 0
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(5, 50)) * F                              # 0.1 - 1 s burst
        tr.feed(off, frame(1, n))
        off += n
        clk.t += n / SR
        pause = int(rng.integers(1, 250)) * F                         # up to 5 s pause
        off += pause
        clk.t += pause / SR
    assert tr.resets == 0


def test_now_off_overrides_the_clock():
    tr = OffsetTracker(clock=lambda: 0.0)
    tr.feed(0, frame(), now_off=0)
    tr.feed(F + 4 * SR, frame(), now_off=0)                           # 4 s gap: the initial 5 s budget
    with pytest.raises(GapTooLarge):                                  # 4 s more, but no time passed
        tr.feed(tr.expected + 4 * SR, frame(), now_off=0)
    tr.feed(tr.expected + 4 * SR, frame(), now_off=3 * SR)            # 3 s later: 6 s refilled


def test_u32_wrap_is_continuous():
    tr = OffsetTracker(clock=Clock())
    start = 2**32 - 2 * F
    tr.feed(start, frame(1))
    out = tr.feed(start + F, frame(2))                                # crosses the wrap
    assert joined(out).tolist() == [2] * F
    assert tr.expected == 0 and tr.rx_off == 0
    assert joined(tr.feed(0, frame(3))).tolist() == [3] * F
    assert tr.expected == F


def test_far_rewind_reanchors_instead_of_going_deaf():
    tr = OffsetTracker(clock=Clock())
    tr.feed(10 * 60 * SR, frame())                                    # 10 min into the stream
    with pytest.raises(GapTooLarge) as ei:
        tr.feed(0, frame(4))                                          # the client restarted its counter
    assert ei.value.reason == "rewind"
    assert tr.expected == F and tr.rx_off == F
    assert joined(tr.feed(F, frame(5))).tolist() == [5] * F


def test_reset_keeps_the_budget():
    clk = Clock()
    tr = OffsetTracker(clock=clk)
    tr.feed(0, frame())
    tr.feed(F + 5 * SR, frame())                                      # spends the initial 5 s
    tr.reset()
    tr.feed(10**6, frame())                                           # re-anchors silently
    with pytest.raises(GapTooLarge):
        tr.feed(10**6 + F + 4 * SR, frame())                          # nothing refilled: no time passed


def test_empty_message_moves_nothing():
    tr = OffsetTracker(clock=Clock())
    tr.feed(0, frame())
    assert tr.feed(F, np.zeros(0, np.int16)) == []
    assert tr.expected == F
