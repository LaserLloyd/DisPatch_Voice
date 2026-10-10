"""The commit cursor: how far each uplink stream has been submitted (contracts C-§B.3, A10).

Keyed by ``(slot_key, stream)``: the slot (device/thread) and the client's random
per-voice-chat stream id. ``upto`` is the uplink sample offset of the end of the
last utterance that became a turn. On reconnect the server answers
``cursor {from: max(client_from, committed)}`` and the client replays its ring
from there, so an utterance is never submitted twice and a transcript the client
missed can be re-sent with ``replayed: true`` (the optional ``meta`` dict holds it).

**Unfinished utterances** (Amendment T6-1): the client's ``from`` includes
``lastRxOff``, but audio the old socket received for an utterance that had not
ended yet lived only in that session's detector and dies with it; replaying from
``rxOff`` would hand the new session the tail of a sentence without its head. So
the session calls :meth:`mark_open` with the utterance's start offset when the
detector starts hearing speech, and :meth:`clear_open` when that speech is
discarded; :meth:`set` (a submit) clears it too. :meth:`resume_from` then answers
the start of the unfinished utterance when it lies between ``committed`` and the
client's ``from`` — the client's ring still holds it (it trims only while idle).

Persisted in ``ctx.paths.state_dir / "cursor.sqlite"`` (WAL, ``synchronous=NORMAL``,
one row write per ``set``) because every backend deploy restarts the process and a
restart between ``transcript`` and the next ``pong`` must not replay an utterance
(hardening M3). LRU of 256 entries, TTL 600 s since the last ``set``. Fail-soft: if
the file cannot be opened the cursor keeps working in memory and logs once.
Thread-safe; one instance per process, shared by every session.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .wire3 import U32, u32_diff

log = logging.getLogger("dispatch_voice.transport.cursor")

FILENAME = "cursor.sqlite"
MAX_ENTRIES = 256
TTL_S = 600.0
_META_MAX = 4096

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cursor (
    k       TEXT PRIMARY KEY,
    upto    INTEGER,
    open_from INTEGER,
    meta    TEXT,
    updated REAL NOT NULL,
    used    REAL NOT NULL
)"""


@dataclass(frozen=True)
class CursorEntry:
    upto: int | None            # None: nothing submitted yet on this stream
    meta: dict | None
    updated: float
    open_from: int | None = None   # start of an utterance heard but not yet submitted


def _key(key) -> str:
    if isinstance(key, str):
        return json.dumps([key])
    if isinstance(key, (tuple, list)) and all(isinstance(p, (str, int)) for p in key):
        return json.dumps([str(p) for p in key])
    raise TypeError(f"cursor key must be a str or a tuple of str, got {key!r}")


class CommitCursor:
    def __init__(self, path: str | os.PathLike | None = None, *, max_entries: int = MAX_ENTRIES,
                 ttl_s: float = TTL_S, clock: Callable[[], float] = time.time) -> None:
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self.path: Path | None = None
        self.persistent = False
        if path is not None:
            p = Path(path)
            if p.is_dir() or not p.suffix:
                p = p / FILENAME
            self.path = p
        self._db = self._open()

    # ------------------------------------------------------------- storage #
    def _connect(self, target: str) -> sqlite3.Connection:
        db = sqlite3.connect(target, check_same_thread=False, isolation_level=None, timeout=2.0)
        if target != ":memory:":
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
        db.execute(_SCHEMA)
        cols = {r[1] for r in db.execute("PRAGMA table_info(cursor)")}
        if "open_from" not in cols:                 # a file from before Amendment T6-1
            db.execute("ALTER TABLE cursor ADD COLUMN open_from INTEGER")
        db.execute("SELECT count(*) FROM cursor").fetchone()
        return db

    def _open(self) -> sqlite3.Connection:
        if self.path is None:
            return self._connect(":memory:")
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fresh = not self.path.exists()
            if fresh:
                # create it 0600 before sqlite does (its WAL/SHM files copy these bits)
                os.close(os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600))
            try:
                db = self._connect(str(self.path))
            except sqlite3.DatabaseError as e:
                aside = self.path.with_name(f"{self.path.name}.corrupt-{int(self._clock())}")
                log.warning("voice: cursor store unreadable (%s); moved aside to %s", e, aside.name)
                self.path.replace(aside)
                for sfx in ("-wal", "-shm"):
                    side = self.path.with_name(self.path.name + sfx)
                    if side.exists():
                        side.unlink()
                os.close(os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600))
                db = self._connect(str(self.path))
            self.persistent = True
            return db
        except (OSError, sqlite3.Error) as e:
            log.warning("voice: cursor store %s unavailable (%s); keeping the cursor in memory "
                        "(a restart may replay one utterance)", self.path.name, e)
            return self._connect(":memory:")

    def close(self) -> None:
        with self._lock:
            try:
                self._db.close()
            except sqlite3.Error:
                pass

    # --------------------------------------------------------------- api #
    def set(self, key, upto: int, *, meta: dict | None = None) -> None:
        """Record that the stream is committed up to ``upto`` (u32 sample offset).

        Never moves backwards (wrap-aware): an older ``upto`` is ignored. ``meta``
        (JSON, <= 4 KB) is what a reconnect may need to re-send, e.g. the transcript.
        Clears an :meth:`mark_open` mark that this commit covers."""
        k = _key(key)
        upto = int(upto) & U32
        m = None
        if meta is not None:
            m = json.dumps(meta, separators=(",", ":"), ensure_ascii=False)
            if len(m) > _META_MAX:
                m = None
                log.warning("voice: cursor meta over %d chars dropped", _META_MAX)
        now = self._clock()
        with self._lock:
            try:
                row = self._db.execute("SELECT upto, updated, open_from FROM cursor WHERE k=?", (k,)).fetchone()
                live = row is not None and now - row[1] <= self.ttl_s
                if live and row[0] is not None and u32_diff(upto, row[0]) < 0:
                    return
                open_from = row[2] if live else None
                if open_from is not None and u32_diff(open_from, upto) < 0:
                    open_from = None              # the submit covers the open utterance
                self._db.execute(
                    "INSERT INTO cursor(k, upto, open_from, meta, updated, used) VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(k) DO UPDATE SET upto=excluded.upto, open_from=excluded.open_from, "
                    "meta=excluded.meta, updated=excluded.updated, used=excluded.used",
                    (k, upto, open_from, m, now, now))
                self._prune(now)
            except sqlite3.Error as e:
                log.warning("voice: cursor write failed (%s)", e)

    def mark_open(self, key, start: int) -> None:
        """The detector started hearing an utterance at sample offset ``start`` (not submitted yet)."""
        k = _key(key)
        start = int(start) & U32
        now = self._clock()
        with self._lock:
            try:
                row = self._db.execute("SELECT upto, updated FROM cursor WHERE k=?", (k,)).fetchone()
                live = row is not None and now - row[1] <= self.ttl_s
                if live and row[0] is not None and u32_diff(start, row[0]) < 0:
                    start = row[0]                # never before what is already committed
                if live:
                    self._db.execute("UPDATE cursor SET open_from=?, updated=?, used=? WHERE k=?",
                                     (start, now, now, k))
                else:
                    self._db.execute(
                        "INSERT INTO cursor(k, upto, open_from, meta, updated, used) VALUES(?,?,?,?,?,?) "
                        "ON CONFLICT(k) DO UPDATE SET upto=NULL, open_from=excluded.open_from, meta=NULL, "
                        "updated=excluded.updated, used=excluded.used", (k, None, start, None, now, now))
                self._prune(now)
            except sqlite3.Error as e:
                log.warning("voice: cursor write failed (%s)", e)

    def clear_open(self, key) -> None:
        """The heard speech was discarded (noise, echo): nothing to rewind to."""
        with self._lock:
            try:
                self._db.execute("UPDATE cursor SET open_from=NULL WHERE k=?", (_key(key),))
            except sqlite3.Error as e:
                log.warning("voice: cursor write failed (%s)", e)

    def _prune(self, now: float) -> None:
        self._db.execute("DELETE FROM cursor WHERE updated < ?", (now - self.ttl_s,))
        self._db.execute("DELETE FROM cursor WHERE k NOT IN "
                         "(SELECT k FROM cursor ORDER BY used DESC, rowid DESC LIMIT ?)", (self.max_entries,))

    def entry(self, key) -> CursorEntry | None:
        k = _key(key)
        now = self._clock()
        with self._lock:
            try:
                row = self._db.execute("SELECT upto, meta, updated, open_from FROM cursor WHERE k=?",
                                       (k,)).fetchone()
                if row is None:
                    return None
                if now - row[2] > self.ttl_s:
                    self._db.execute("DELETE FROM cursor WHERE k=?", (k,))
                    return None
                self._db.execute("UPDATE cursor SET used=? WHERE k=?", (now, k))
            except sqlite3.Error as e:
                log.warning("voice: cursor read failed (%s)", e)
                return None
        meta = None
        if row[1]:
            try:
                meta = json.loads(row[1])
            except ValueError:
                meta = None
        return CursorEntry(None if row[0] is None else int(row[0]), meta, float(row[2]),
                           None if row[3] is None else int(row[3]))

    def get(self, key) -> int | None:
        """The committed offset for ``key``, or None (unknown or older than the TTL)."""
        e = self.entry(key)
        return None if e is None else e.upto

    def resume_from(self, key, client_from: int) -> int:
        """The ``cursor {from}`` answer: ``max(client_from, committed)`` on the u32 ring —
        except when an unfinished utterance started at ``o`` with ``committed <= o < client_from``:
        then ``o``, so the new session hears that utterance from its start (Amendment T6-1)."""
        e = self.entry(key)
        client_from = int(client_from) & U32
        if e is None:
            return client_from
        frm = client_from
        if e.upto is not None and u32_diff(e.upto, frm) > 0:
            frm = e.upto
        if e.open_from is not None and u32_diff(e.open_from, frm) < 0 and \
                (e.upto is None or u32_diff(e.open_from, e.upto) >= 0):
            frm = e.open_from
        return frm

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT count(*) FROM cursor").fetchone()[0]
