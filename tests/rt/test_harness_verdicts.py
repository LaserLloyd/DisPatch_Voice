"""The latency harness scores every row against the §4.9 column it RAN in (review T4 #6):
a pass whose profile flipped midway is flagged MIXED and each part gets its own verdict."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # tests/: latency_harness
latency_harness = pytest.importorskip("latency_harness")
TurnResult, verdicts, column_of = latency_harness.TurnResult, latency_harness.verdicts, latency_harness.column_of


def row(load, profile, mhz, first_audio, clip="short"):
    return TurnResult(clip=clip, load=load, profile=profile, max_mhz=mhz, load1="1", utterance_s=2.0,
                      speech_end_s=1.8, first_audio_at=first_audio, decision_at=0.8, first_sentence_at=2.0,
                      tts_first_chunk_ms=900, speculation_starts=1)


def test_column_of():
    assert column_of("low-power", 2000) == "power-saver"
    assert column_of("performance", 5090) == "performance"
    assert column_of("balanced", 5090) == "performance"
    assert column_of("performance", 2000) == "power-saver"       # a hold that did not take


def test_verdicts_group_rows_by_the_profile_they_ran_in():
    rows = [row("idle", "low-power", 2000, 3.5)] * 3 + [row("idle", "performance", 5090, 2.9)] * 3
    out = verdicts(rows, "chatterbox")
    assert any(v.startswith("MIXED idle") for v in out)
    ps = [v for v in out if v.endswith("MIXED pass]") and "first audio idle" in v]
    assert len(ps) == 2
    assert any("PASS" in v and "power-saver column, 3 rows" in v and "3.50 s" in v for v in ps)
    assert any("PASS" in v and "performance column, 3 rows" in v and "2.90 s" in v for v in ps)
    # The same 3.5 s rows judged in the performance column would fail — and are not.
    assert not any(v.startswith("FAIL") for v in out)
    assert sum(v.startswith("INFO stages idle") for v in out) == 2
    # A clean pass: one verdict, no MIXED line.
    out2 = verdicts([row("load8", "low-power", 2000, 3.9)] * 2, "chatterbox")
    assert not any(v.startswith("MIXED") for v in out2)
    assert any(v.startswith("FAIL first audio load8: median 3.90 s") and "power-saver column, 2 rows" in v for v in out2)
