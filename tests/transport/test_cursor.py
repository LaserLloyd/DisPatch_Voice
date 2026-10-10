"""CommitCursor: LRU/TTL, never backwards, persisted across a process restart (C-§B.3, A10)."""
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dispatch_voice.transport.cursor import FILENAME, CommitCursor

SRC = str(Path(__file__).resolve().parents[2] / "src")


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_set_get_and_resume_from(tmp_path):
    c = CommitCursor(tmp_path)
    key = ("slot-1", "stream-0001")
    assert c.get(key) is None
    assert c.resume_from(key, 500) == 500
    c.set(key, 32000, meta={"text": "what's on today", "message_id": "m1"})
    assert c.get(key) == 32000
    assert c.entry(key).meta == {"text": "what's on today", "message_id": "m1"}
    assert c.resume_from(key, 16000) == 32000          # max(client_from, committed)
    assert c.resume_from(key, 48000) == 48000
    assert c.persistent and (tmp_path / FILENAME).exists()


def test_never_moves_backwards_and_handles_the_wrap(tmp_path):
    c = CommitCursor(tmp_path)
    c.set("k", 5000)
    c.set("k", 4000)                                    # older: ignored
    assert c.get("k") == 5000
    c.set("k", 2**32 - 100)                             # 5100 samples BEHIND 5000 on the ring
    assert c.get("k") == 5000
    c.set("w", 2**32 - 100)
    c.set("w", 50)                                      # 150 samples later, across the wrap
    assert c.get("w") == 50
    c.set("w", 2**32 - 10)                              # behind 50 on the ring: ignored
    assert c.get("w") == 50
    assert c.resume_from("w", 2**32 - 10) == 50
    assert c.resume_from("w", 70) == 70


def test_ttl_expires_entries():
    clk = Clock()
    c = CommitCursor(None, clock=clk)
    c.set(("s", "a"), 100)
    clk.t += 599
    assert c.get(("s", "a")) == 100
    clk.t += 2                                          # 601 s after the set
    assert c.get(("s", "a")) is None
    assert len(c) == 0


def test_expired_entry_does_not_block_a_lower_offset():
    clk = Clock()
    c = CommitCursor(None, clock=clk)
    c.set("k", 90000)
    clk.t += 700
    c.set("k", 10)                                      # a new chat after the TTL may start low
    assert c.get("k") == 10


def test_lru_keeps_256_most_recently_used():
    clk = Clock()
    c = CommitCursor(None, clock=clk)
    for i in range(256):
        clk.t += 0.01
        c.set(("slot", f"s{i}"), i)
    clk.t += 0.01
    assert c.get(("slot", "s0")) == 0                   # touch the oldest: now recently used
    clk.t += 0.01
    c.set(("slot", "new"), 1)
    assert len(c) == 256
    assert c.get(("slot", "s0")) == 0
    assert c.get(("slot", "s1")) is None                # the least recently used went
    assert c.get(("slot", "new")) == 1


def test_keys_are_distinct_per_slot_and_stream():
    c = CommitCursor(None)
    c.set(("a", "x"), 1)
    c.set(("a", "y"), 2)
    c.set(("b", "x"), 3)
    assert (c.get(("a", "x")), c.get(("a", "y")), c.get(("b", "x"))) == (1, 2, 3)
    with pytest.raises(TypeError):
        c.set(object(), 1)


def test_file_is_private(tmp_path):
    state = tmp_path / "ext" / "voice"
    c = CommitCursor(state)
    c.set("k", 1)
    assert stat.S_IMODE(os.stat(state).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(state / FILENAME).st_mode) == 0o600


def test_set_costs_at_most_a_millisecond(tmp_path):
    c = CommitCursor(tmp_path)
    c.set("warm", 1)
    t = time.perf_counter()
    for i in range(200):
        c.set(("slot", f"s{i % 50}"), i * 1000)
    per = (time.perf_counter() - t) / 200
    assert per < 0.001, f"{per * 1000:.3f} ms per set"


_CHILD = r"""
import sys
sys.path.insert(0, {src!r})
from dispatch_voice.transport.cursor import CommitCursor
c = CommitCursor({path!r})
if sys.argv[1] == "write":
    c.set(("dev-12345678", "stream-0001"), 123456, meta={{"text": "remind me at five", "message_id": "m42"}})
    import os
    os._exit(0)           # no close(), no atexit: the process just dies after the set
else:
    e = c.entry(("dev-12345678", "stream-0001"))
    print(e.upto if e else None, (e.meta or {{}}).get("message_id") if e else None)
"""


def test_survives_a_process_restart(tmp_path):
    """The deploy case: a set, the process dies without closing, a new process reads it back."""
    code = _CHILD.format(src=SRC, path=str(tmp_path))
    w = subprocess.run([sys.executable, "-c", code, "write"], capture_output=True, text=True, timeout=30,
                       check=False)
    assert w.returncode == 0, w.stderr
    r = subprocess.run([sys.executable, "-c", code, "read"], capture_output=True, text=True, timeout=30,
                       check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["123456", "m42"]


def test_corrupt_file_is_moved_aside(tmp_path, caplog):
    (tmp_path / FILENAME).write_bytes(b"this is not a sqlite database" * 100)
    c = CommitCursor(tmp_path)
    c.set("k", 7)
    assert c.get("k") == 7 and c.persistent
    assert list(tmp_path.glob(FILENAME + ".corrupt-*"))


def test_unwritable_location_falls_back_to_memory(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    c = CommitCursor(blocker / "sub")                   # parent is a file: cannot mkdir
    c.set("k", 9)
    assert c.get("k") == 9 and not c.persistent


def test_oversized_meta_is_dropped_not_fatal():
    c = CommitCursor(None)
    c.set("k", 5, meta={"text": "x" * 10000})
    e = c.entry("k")
    assert e.upto == 5 and e.meta is None


# --------------------------------------------------- unfinished utterances (T6-1) #
def test_open_utterance_rewinds_below_the_clients_rx_off():
    """Drop mid-utterance: the old session heard 0.5 s of it (rx_off past its start); the new
    session must hear it from its start, not from rx_off."""
    c = CommitCursor(None)
    key = ("dev", "stream")
    c.set(key, 16000)                                   # utterance 1 submitted, ends at 1.0 s
    c.mark_open(key, 40000)                             # utterance 2 starts at 2.5 s ...
    assert c.resume_from(key, 48000) == 40000           # ... the client last heard rx_off 3.0 s
    assert c.resume_from(key, 30000) == 30000           # a client that is further back keeps its from
    assert c.resume_from(key, 8000) == 16000            # never before what was committed
    c.set(key, 60000)                                   # utterance 2 submitted: the mark goes
    assert c.entry(key).open_from is None
    assert c.resume_from(key, 48000) == 60000


def test_open_mark_without_any_commit_and_clear():
    c = CommitCursor(None)
    c.mark_open("k", 3200)
    assert c.get("k") is None and c.entry("k").open_from == 3200
    assert c.resume_from("k", 9000) == 3200
    c.clear_open("k")                                   # it was noise
    assert c.resume_from("k", 9000) == 9000


def test_open_mark_is_clamped_to_committed_and_persists(tmp_path):
    c = CommitCursor(tmp_path)
    c.set("k", 50000)
    c.mark_open("k", 40000)                             # a stale start before the commit
    assert c.entry("k").open_from == 50000
    c.close()
    again = CommitCursor(tmp_path)
    assert again.entry("k").open_from == 50000 and again.get("k") == 50000


def test_old_schema_file_is_migrated(tmp_path):
    import sqlite3
    db = sqlite3.connect(tmp_path / FILENAME)
    db.execute("CREATE TABLE cursor (k TEXT PRIMARY KEY, upto INTEGER NOT NULL, meta TEXT, "
               "updated REAL NOT NULL, used REAL NOT NULL)")
    db.execute("INSERT INTO cursor VALUES (?, ?, NULL, ?, ?)", ('["k"]', 7, time.time(), time.time()))
    db.commit()
    db.close()
    c = CommitCursor(tmp_path)
    assert c.get("k") == 7
    c.mark_open("k", 9)
    assert c.resume_from("k", 20) == 9
