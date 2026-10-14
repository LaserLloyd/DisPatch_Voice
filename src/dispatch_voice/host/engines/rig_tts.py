"""``ReplyEngine``: the session's TTS engine when a SpeechForge rig is in play
(ADDON-ENGINE §4). It has the ``host.tts.TTSEngine`` shape and wraps the CPU engine
(``primary``), which keeps every job it had: acknowledgements, notices, previews,
``render_line`` and its disk cache stay on the CPU (background, cached per voice).

Live replies go through :meth:`ReplyEngine.stream_reply`, which the session calls
with the reply's identity (stream id + generation), one sentence at a time:

* **One reply, one engine** (voice parity). The choice is made at the reply's FIRST
  sentence from session-level health: the rig only when it is proven (``usable``)
  and the voice's conditionals are stored there; else the whole reply is the CPU's,
  with no deadline paid. Once a reply moves to the CPU it stays there.
* **One rig context per reply** (per voice: a narrator piece opens its own).
  ``tts.open{ctx, gen, voice, seed, first_chunk_tokens}``, then per sentence
  ``tts.text{final}`` + ``tts.flush{id}``: the audio frames before ``tts.flushed{id}``
  are that sentence's (24 kHz, 40 ms int16 — the host's own frame shape).
* **The start rule** (800 ms by default). A sentence with no audio within
  ``tts_start_deadline_ms`` of its text is rendered on the CPU and the rest of the
  reply stays there; the rig context is cancelled. The clock runs only while the rig
  could be producing by ITS credit rule (``TtsCtx.rig_may_produce``: the last
  ``played_ms`` actually sent, frozen as the rig freezes it, with a margin for the
  first chunk): a paused context or one held back by the credit is not late.
* **``limits.max_text``** is per context: a reply whose text would pass it continues
  in a fresh context (same voice, same seed), which the rig queues behind the first.
* **Engine errors** (``engine_failed``, ``not_ready``, ``vacated``, a second ``busy``
  in a row) move that reply to the CPU AND count as a rig failure (cooldown, then the
  between-turn probe), so the next replies do not pay the same error round trip.
* **A rig that dies mid-sentence**: the sentence re-renders on the CPU from its first
  sample (a sentence boundary, never mid-word) and the rest of the reply stays on
  the CPU.
* **Credit / barge-in / false interruption**: :meth:`heard` turns the phone's
  ``heard`` reports into ``credit{played_ms}`` (kept fresh every 250 ms by the
  connection while the context is open); :meth:`cancel_gen` sends ``cancel{gen}``
  (the session calls it BEFORE the host abort); :meth:`pause` / :meth:`resume` follow
  the duck.
* **Heard text**: the rig's ``tts.words`` (estimates for Chatterbox) become each
  sentence's word timings (:meth:`words_for`); the session's heard map cuts at a word
  instead of its linear time map.

``cache_tag`` is the CPU engine's (rig renders are never cached, so no rig tag mixes
with CPU tags).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from .. import rig as _rig

log = logging.getLogger("dispatch.voice")

POLL_S = 0.02
REPLIES_KEEP = 16
SEQ_KEEP = 256
RESYNC_WAIT_S = 1.5     # a re-PUT of the voice conditionals (~0.8 MB over the tailnet)


class _Fallback(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _Stopped(Exception):
    pass


@dataclass
class _ReplyState:
    key: tuple
    choice: str                      # rig | cpu
    seed: int
    ctx: _rig.TtsCtx | None = None
    reason: str | None = None
    resynced: bool = False           # the voice was re-registered once for this reply


class ReplyEngine:
    sample_format = "pcm_s16le"
    channels = 1
    streaming = True

    def __init__(self, cpu, rig_session: _rig.RigSession, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.cpu = cpu
        self.primary = cpu                  # tts.walk() reaches the CPU chain through it
        self.rig = rig_session
        self.conn = rig_session.conn
        self.clock = clock
        self.sample_rate = int(getattr(cpu, "sample_rate", 24000) or 24000)
        self._lock = threading.Lock()
        self._replies: OrderedDict[tuple, _ReplyState] = OrderedDict()
        self._open: _ReplyState | None = None
        self._fid = 0
        self._words: OrderedDict[int, list] = OrderedDict()
        self._seq_ctx: OrderedDict[int, tuple[_rig.TtsCtx, float]] = OrderedDict()
        self._busy_strikes = 0

    def __getattr__(self, name):
        if name.startswith("__") or name in ("cpu", "rig", "conn"):
            raise AttributeError(name)
        return getattr(self.cpu, name)

    # ---- everything that is not a live reply: the CPU engine, unchanged
    def stream(self, text: str, profile_id: str | None = None, stop: threading.Event | None = None):
        return self.cpu.stream(text, profile_id, stop)

    def synth(self, text: str, profile_id: str | None = None):
        return self.cpu.synth(text, profile_id)

    # ------------------------------------------------------------------ replies
    def stream_reply(self, text: str, profile_id: str | None = None,
                     stop: threading.Event | None = None, *, gen: int = 0, seq: int = 0,
                     rid: str | None = None, kind: str = "reply") -> Iterator[np.ndarray]:
        if kind != "reply" or self.sample_rate != _rig.TTS_RATE:
            yield from self.cpu.stream(text, profile_id, stop)
            return
        st = self._state_for((rid or "", int(gen)), profile_id)
        if st.choice == "rig" and self.rig.local_only(profile_id):
            # A voice the rig never speaks (the role-play narrator's built-in voice):
            # THIS piece is the CPU's, the reply stays on the rig for its cloned
            # pieces. Moving the whole reply here is what changed the speaker mid-reply.
            self.conn._need_cpu()
            yield from self.cpu.stream(text, profile_id, stop)
            return
        if st.choice == "rig":
            try:
                yield from self._rig_sentence(st, text, profile_id, stop, int(seq))
                return
            except _Stopped:
                return
            except _Fallback as f:
                if f.reason == "unknown_voice" and not st.resynced and self._reregister(st, profile_id):
                    # The rig lost the voice (or it was never stored): re-registered, so
                    # the sentence is retried once on the rig, not the CPU's other timbre.
                    try:
                        yield from self._rig_sentence(st, text, profile_id, stop, int(seq))
                        return
                    except _Stopped:
                        return
                    except _Fallback as f2:
                        f = f2
                self._to_cpu(st, f.reason)
            if stop is not None and stop.is_set():
                return
        yield from self.cpu.stream(text, profile_id, stop)

    def _reregister(self, st: _ReplyState, profile_id: str | None) -> bool:
        """Once per reply: drop the failed context and wait (bounded) for the voice
        to be PUT again. False = still not stored: the reply goes to the CPU."""
        st.resynced = True
        c, st.ctx = st.ctx, None
        if c is not None:
            self.conn.cancel_ctx(c)
        # (the rig's own `unknown_voice` already started the re-PUT: _rig_sentence)
        if not self.rig.syncing(profile_id):
            return False                               # nothing is on its way: no pointless wait
        return self.rig.wait_voice(profile_id, RESYNC_WAIT_S)

    def _state_for(self, key: tuple, profile_id: str | None) -> _ReplyState:
        with self._lock:
            st = self._replies.get(key)
            if st is not None:
                return st
            prev = self._open
            if prev is not None and prev.ctx is not None:
                self.conn.close_ctx(prev.ctx)          # the previous reply is complete
            # A reply that opens with narration (a built-in voice the rig never speaks)
            # still belongs on the rig when the session's cloned voice is stored there.
            on_rig = bool(self.conn.usable()) and (
                self.rig.voice_ref(profile_id) is not None
                or (self.rig.local_only(profile_id) and self.rig.has_rig_voice()))
            st = _ReplyState(key, "rig" if on_rig else "cpu", _rig.random_seed())
            self._replies[key] = st
            while len(self._replies) > REPLIES_KEEP:
                self._replies.popitem(last=False)
            self._open = st
        self.conn.metrics.add("replies")
        if on_rig:
            self.conn.metrics.add("tts")
        else:
            self.conn._need_cpu()                     # a reply on the CPU: the hold
        return st

    def reply_done(self, rid: str | None) -> None:
        """The reply's last sentence went out: close its rig context now (contract §4),
        not when the next reply starts. An open idle context holds up the rig's vacate
        for its whole drain window."""
        with self._lock:
            st = self._open
            c = st.ctx if st is not None and st.key[0] == (rid or "") else None
        if c is not None:
            self.conn.close_ctx(c)

    def _to_cpu(self, st: _ReplyState, reason: str) -> None:
        with self._lock:
            st.choice = "cpu"
            st.reason = reason
            c, st.ctx = st.ctx, None
        if c is not None:
            self.conn.cancel_ctx(c)
        self.conn.metrics.fallback(reason)
        self.conn._need_cpu()
        log.info("voice: reply moves to this computer (%s)", reason)

    def _ctx_for(self, st: _ReplyState, profile_id: str | None, text: str = "") -> _rig.TtsCtx:
        voice = self.rig.voice_ref(profile_id)
        if voice is None:
            raise _Fallback("unknown_voice")
        c = st.ctx
        if c is not None and c.voice == voice and not (c.cancelled or c.closed) \
                and c.text_chars and c.text_chars + len(text) > self.conn.max_text():
            # limits.max_text is per context: continue in a fresh one (same voice and
            # seed: the same speaker); the rig queues it behind this one.
            self.conn.close_ctx(c)
            c = None
        if c is not None and (c.voice != voice or c.cancelled or c.closed):
            if c.voice != voice:
                self.conn.close_ctx(c)
            c = None
        if c is None:
            c = self.conn.open_ctx(gen=st.key[1], voice=voice, seed=st.seed)
            if c is None:
                raise _Fallback(self.conn.reason or "closed")
            st.ctx = c
        return c

    def _rig_sentence(self, st: _ReplyState, text: str, profile_id: str | None,
                      stop: threading.Event | None, seq: int) -> Iterator[np.ndarray]:
        c = self._ctx_for(st, profile_id, text)
        with self._lock:
            self._fid += 1
            fid = self._fid
        c.text_chars += len(text)
        start_ms = c.audio_ms
        self._seq_ctx[seq] = (c, start_ms)
        while len(self._seq_ctx) > SEQ_KEEP:
            self._seq_ctx.popitem(last=False)
        self.conn.send_threadsafe({"type": "tts.text", "ctx": c.ctx, "text": text, "final": True})
        self.conn.send_threadsafe({"type": "tts.flush", "ctx": c.ctx, "id": fid})
        start_s = self.conn.settings.tts_start_deadline_ms / 1000.0
        ahead = float(self.conn.settings.ahead_ms)
        t_text = self.clock()
        gate_open_at: float | None = t_text if c.rig_may_produce(t_text, ahead) else None
        got = 0
        words: list = []
        while True:
            if stop is not None and stop.is_set():
                self.conn.cancel_ctx(c)
                raise _Stopped()
            if not self.conn.alive(c.epoch):
                raise _Fallback(self.conn.reason or "closed")
            try:
                kind, val = c.q.get(timeout=POLL_S)
            except queue.Empty:
                if got:
                    continue
                now = self.clock()
                # The rig may be holding back on purpose: a paused context, or one
                # `ahead` past the playback point it was last told (credit). The clock
                # runs only while the rig could produce by its own rule.
                if not c.rig_may_produce(now, ahead):
                    gate_open_at = None
                elif gate_open_at is None:
                    gate_open_at = now
                if gate_open_at is not None and now - gate_open_at > start_s:
                    self.conn.cancel_ctx(c)
                    self.conn.fail("tts_late", move=False)
                    raise _Fallback("tts_late") from None
                continue
            if kind == "audio":
                if not got:
                    self._busy_strikes = 0
                got += len(val)
                yield val
            elif kind == "words":
                words.extend(w for w in val if isinstance(w, (list, tuple)) and len(w) == 3)
            elif kind == "flushed":
                if val == fid:
                    self._note_words(seq, words, start_ms)
                    return
            elif kind == "started":
                continue
            elif kind == "error":
                if val == "unknown_voice":
                    self.rig.resync_threadsafe(profile_id)
                elif val in ("engine_failed", "not_ready", "vacated"):
                    # The rig cannot speak right now: cool it down so the next replies
                    # do not pay the same round trip (and its error flood limit).
                    self.conn.fail("tts_failed" if val == "engine_failed" else
                                   "vacating" if val == "vacated" else "not_ready", move=False)
                elif val == "busy":
                    self._busy_strikes += 1
                    if self._busy_strikes >= _rig.BUSY_STRIKES:
                        self._busy_strikes = 0
                        self.conn.fail("busy", move=False)
                raise _Fallback(str(val))
            elif kind == "cancelled":
                if stop is not None and stop.is_set():
                    raise _Stopped()
                raise _Fallback("cancelled")
            elif kind in ("failed", "done"):
                raise _Fallback(str(val or "closed"))

    def _note_words(self, seq: int, words: list, start_ms: float) -> None:
        out = []
        for w, s, e in words:
            try:
                out.append([str(w), max(0, int(round(float(s) - start_ms))), max(0, int(round(float(e) - start_ms)))])
            except (TypeError, ValueError):
                continue
        if out:
            self._words[seq] = out
            while len(self._words) > SEQ_KEEP:
                self._words.popitem(last=False)

    def words_for(self, seq: int) -> list | None:
        """The rig's word timings for a sentence (ms from the sentence's first sample),
        or None (spoken on the CPU: the linear map applies)."""
        return self._words.get(int(seq))

    # ------------------------------------------------------------------ session hooks
    def cancel_gen(self, gen: int) -> None:
        """``cancel{gen}``: the rig stops every reply of this generation and older."""
        self.conn.cancel_gen(gen)

    def pause(self) -> None:
        st = self._open
        if st is not None and st.ctx is not None:
            self.conn.pause_ctx(st.ctx)

    def resume(self) -> None:
        st = self._open
        if st is not None and st.ctx is not None:
            self.conn.resume_ctx(st.ctx)

    def heard(self, gen: int, seq: int, played_ms: int) -> None:
        """The phone's ``heard`` -> ``credit{ctx, ahead_ms, played_ms}`` (at most every
        250 ms; the connection re-sends it, run on in real time, between reports)."""
        hit = self._seq_ctx.get(int(seq))
        if hit is None:
            return
        c, start_ms = hit
        if c.cancelled or c.closed or c.gen != (int(gen) & _rig.U16):
            return
        self.conn.credit(c, start_ms + max(0, int(played_ms)))
