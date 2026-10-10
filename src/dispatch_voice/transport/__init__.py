"""Voice wire transport: the session's only send path (contracts C-§B.3, C.1, C.2).

One :class:`Transport` per socket, built by :meth:`Transport.negotiate` from the
URL query (``?d=&s=&v=3&up=opus,pcm16&down=opus,pcm16``):

* **v2** (no ``v=3``): exactly today's wire — a JSON ``audio`` header and the raw
  PCM16 frame, adjacent under one lock; no ``say``; uplink is raw PCM16.
* **v3**: ``say`` frames, binary framing (:mod:`.wire3`) with sample offsets, and
  Opus both ways when both sides can (else PCM16 without the JSON header).

Downlink Opus: every sentence (``kind``, ``gen``, ``seq``) is its own Opus stream
(a fresh encoder at the current ``down_kbps`` rung, so a rung change lands at the
next sentence by construction); 20 ms packets, at most 10 per message. The first
message of a sentence goes out at once with whatever packets exist (first audio is
never delayed); later packets go out in full 10-packet messages, and a partial
batch waits at most :data:`BATCH_MS` (100 ms). Sentences go out one after
another: audio for another sentence, or a ``say``, ends every open sentence first
(pending packets plus the padded tail), and a probe flushes pending packets, so
the client receives each sentence's audio contiguously and in order.
:meth:`end_sentence` (or ``last=True``) pads the final partial frame plus the
encoder lookahead with silence, so no tail is lost; the decoded sentence starts
with the encoder lookahead (≈ 6.5 ms) of near-silence. Ending a sentence that sent
audio also sends ``audio_end {gen, seq, kind}`` (Amendment T6-2): the only way a
client can know a sentence's audio is complete when the next ``say`` is held up
by a stall, which ``start.after_partial`` needs after a drop. The ``t`` stamp of a binary message is this
transport's monotonic clock in ms at the moment it is written (:meth:`now_ms`): the
client's "arrival − t growing = link, flat = server TTS" test needs send time.

Uplink: :meth:`decode_up` turns a v3 message into ``(off, flags, pcm16 @ 16 kHz)``
(``None`` for a v2 raw frame; ``ValueError`` for a bad one). The session feeds
:attr:`offsets` (an :class:`~.offsets.OffsetTracker`) with it, which also yields
``pong.rx_off``. Messages over :data:`~.wire3.MAX_MESSAGE` must be dropped by the
caller before they are queued (:meth:`too_big`).

libopus is optional: without it (or with ``opus: off``) both directions negotiate
``pcm16`` and nothing else changes.

Lifetime (review-T6 H1): :meth:`close` takes the audio lock before it frees any
codec, so it waits for an encode running in a worker thread (bounded by
:data:`CLOSE_WAIT_S`; the codecs' own locks make the free safe even past that),
and every call after ``close`` is a no-op. Audio for a sentence that already
ended (its ``audio_end`` went out) is dropped with one warning instead of opening
a second stream for the same ``seq``: call ``end_sentence(N)`` before
``send_say(N+1)``.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from . import opus, wire3
from .cursor import CommitCursor, CursorEntry
from .link import LinkState, ProbeLimiter, RelayIndicator, peer_address, probe_payload
from .offsets import GapTooLarge, OffsetTracker

__all__ = [
    "CommitCursor",
    "CursorEntry",
    "GapTooLarge",
    "LinkState",
    "Negotiated",
    "OffsetTracker",
    "ProbeLimiter",
    "RelayIndicator",
    "Transport",
    "gen_older",
    "opus",
    "parse_query",
    "peer_address",
    "probe_payload",
    "wire3",
]

log = logging.getLogger("dispatch_voice.transport")

PROTO = 3
RATE_IN = 16_000
ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")
CODEC_LIST_MAX = 8
BATCH_MS = 100
PACKETS_PER_MESSAGE = wire3.MAX_DOWN_PACKETS
PCM_CHUNK_MS = 40
OPUS_FRAME_MS = 20
MAX_OPEN_STREAMS = 4
#: encode in a worker thread when a call brings more than this many frames (~2.8 ms of CPU)
THREAD_FRAMES = 10
PRE_ROLL_MS = 300
#: how long close() waits for in-flight audio work (a send blocked on a stalled socket)
CLOSE_WAIT_S = 2.0
#: one uplink message may decode to at most this much audio (review-T6 L1: a 64 KB
#: message of 50 x 120 ms packets would otherwise be 6 s of PCM)
MAX_UP_DECODED_S = 2.0
#: sentences remembered as finished, so a late chunk cannot re-open one
FINISHED_KEEP = 64


@dataclass(frozen=True)
class Negotiated:
    proto: int
    device: str | None
    stream: str | None
    up: tuple[str, ...]
    down: tuple[str, ...]


def _codec_list(raw) -> tuple[str, ...]:
    if not isinstance(raw, str) or len(raw) > 64:
        return ()
    out = []
    for c in raw.split(",")[:CODEC_LIST_MAX]:
        c = c.strip().lower()
        if c in wire3.CODECS and c not in out:
            out.append(c)
    return tuple(out)


def parse_query(query: Mapping) -> Negotiated:
    """Validate the socket's query (unknown or malformed values -> defaults, silently)."""
    get = query.get
    v = get("v")
    proto = PROTO if str(v) == "3" else 2
    d = get("d")
    s = get("s")
    device = d if isinstance(d, str) and ID_RE.match(d) else None
    stream = s if isinstance(s, str) and ID_RE.match(s) else None
    return Negotiated(proto, device, stream, _codec_list(get("up")), _codec_list(get("down")))


def gen_older(g: int, than: int) -> bool:
    """``g`` is an older gen than ``than`` on the u16 wire ring (a half-ring window)."""
    return 0 < ((int(than) - int(g)) & wire3.U16) < 0x8000


def _pick(client: tuple[str, ...], opus_ok: bool) -> str:
    for c in client:
        if c == "opus" and opus_ok:
            return "opus"
        if c == "pcm16":
            return "pcm16"
    return "pcm16"


@dataclass
class _Stream:
    key: tuple[int, int, int]            # (kind, gen, seq)
    codec: int
    enc: opus.Encoder | None
    rem: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int16))
    packets: list = field(default_factory=list)
    first_pending_at: float | None = None
    sent_messages: int = 0
    samples_in: int = 0


class Transport:
    """Per-socket wire. ``ws`` needs ``send_text(str)`` and ``send_bytes(bytes)``."""

    def __init__(self, ws, *, proto: int = 2, codec_in: str = "pcm16", codec_out: str = "pcm16",
                 device: str | None = None, stream: str | None = None, rate_out: int = 24_000,
                 opus_down_kbps: int = 24, opus_up_kbps: int = 16,
                 gate: dict | None = None, clock: Callable[[], float] = time.monotonic,
                 offsets: OffsetTracker | None = None) -> None:
        self.ws = ws
        self.proto = PROTO if proto == PROTO else 2
        self.codec_in = codec_in if self.proto == PROTO else "pcm16"
        self.codec_out = codec_out if self.proto == PROTO else "pcm16"
        self.device = device
        self.stream = stream
        self.rate_out = int(rate_out)
        self.clock = clock
        self.link = LinkState(down_kbps=opus_down_kbps if opus_down_kbps in (24, 16) else 24,
                              up_kbps=opus_up_kbps if opus_up_kbps in (16, 12) else 16)
        self.offsets = offsets or OffsetTracker(clock=clock)
        self.probes = ProbeLimiter(clock=clock)
        self.gate = gate or {"pre_roll_ms": PRE_ROLL_MS, "hangover_s": 2.6}
        self._t0 = clock()
        self._send_lock = asyncio.Lock()       # one ws write at a time (v2 header+PCM adjacency)
        self._audio_lock = asyncio.Lock()      # encoder state + batching
        self._streams: dict[tuple[int, int, int], _Stream] = {}
        self._order: list[tuple[int, int, int]] = []
        self._timer: asyncio.Task | None = None
        self._decoder: opus.Decoder | None = None
        self._closed = False
        self._finished: OrderedDict[tuple[int, int, int], None] = OrderedDict()
        self._gen_floor: int | None = None
        self._late_warned = False
        self.stats = {"tx_bytes": 0, "tx_messages": 0, "tx_audio_messages": 0, "tx_audio_ms": 0.0,
                      "rx_bytes": 0, "rx_messages": 0, "rx_opus_packets": 0, "say": 0,
                      "probes": 0, "max_send_ms": 0.0, "bad_frames": 0, "late_audio_dropped": 0}
        if self.codec_out == "opus" and self.rate_out not in opus.RATES:
            log.info("voice: engine rate %d Hz cannot go into Opus; downlink stays PCM16", self.rate_out)
            self.codec_out = "pcm16"

    # ------------------------------------------------------------ negotiation #
    @classmethod
    def negotiate(cls, ws, query: Mapping, *, opus_mode: str = "auto", opus_down_kbps: int = 24,
                  opus_up_kbps: int = 16, rate_out: int = 24_000, hard_silence_s: float = 1.4,
                  hold_max_s: float = 1.2, pre_roll_ms: int = PRE_ROLL_MS,
                  clock: Callable[[], float] = time.monotonic) -> Transport:
        """Build the transport for this socket from its query and the server settings.

        ``opus_mode``: ``voice.yaml: opus`` (``auto`` | ``off``). The gate hangover
        follows SPEC §5.6: ``max(hard_silence × 1.5, hold_max × 1.5) + 0.5``."""
        n = parse_query(query)
        opus_ok = opus_mode != "off" and opus.available()
        gate = {"pre_roll_ms": int(pre_roll_ms),
                "hangover_s": round(max(hard_silence_s * 1.5, hold_max_s * 1.5) + 0.5, 2)}
        if n.proto != PROTO:
            return cls(ws, proto=2, device=n.device, stream=n.stream, rate_out=rate_out, gate=gate, clock=clock)
        return cls(ws, proto=PROTO, codec_in=_pick(n.up, opus_ok), codec_out=_pick(n.down, opus_ok),
                   device=n.device, stream=n.stream, rate_out=rate_out, opus_down_kbps=opus_down_kbps,
                   opus_up_kbps=opus_up_kbps, gate=gate, clock=clock)

    @property
    def v3(self) -> bool:
        return self.proto == PROTO

    def now_ms(self) -> int:
        """This socket's server clock in ms (u32), the `t` of binary frames and `say`, and `pong.srv_t`."""
        return int((self.clock() - self._t0) * 1000) & wire3.U32

    def ready_fields(self) -> dict:
        """The transport's part of `ready` (C.4); empty for a v2 socket."""
        if not self.v3:
            return {}
        return {"proto": PROTO, "codec_in": self.codec_in, "codec_out": self.codec_out,
                "opus_up_kbps": self.link.up_kbps, "opus_down_kbps": self.link.down_kbps,
                "rx_off": self.offsets.rx_off, "gate": dict(self.gate), "say": True}

    @property
    def closed(self) -> bool:
        return self._closed

    def pong(self, t) -> dict:
        """The answer to `ping {t}`: `{type: pong, t, rx_off, srv_t}` (extra fields are harmless on v2).

        ``t`` is echoed only when it is a finite number (else ``None``): it comes from the client."""
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
            t = None
        return {"type": "pong", "t": t, "rx_off": self.offsets.rx_off, "srv_t": self.now_ms()}

    # ---------------------------------------------------------------- writes #
    async def _write_text(self, s: str) -> None:
        t = time.perf_counter()
        await self.ws.send_text(s)
        self._account(len(s.encode()), t)

    async def _write_bytes(self, b: bytes) -> None:
        t = time.perf_counter()
        await self.ws.send_bytes(b)
        self._account(len(b), t)

    def _account(self, n: int, t: float) -> None:
        self.stats["tx_bytes"] += n
        self.stats["tx_messages"] += 1
        ms = (time.perf_counter() - t) * 1000
        if ms > self.stats["max_send_ms"]:
            self.stats["max_send_ms"] = round(ms, 2)

    async def send(self, obj: dict) -> None:
        """Any JSON frame. Audio still pending for a sentence is NOT flushed by this
        (control frames such as `state`/`pong` may overtake held audio)."""
        if self._closed:
            return
        data = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        async with self._send_lock:
            await self._write_text(data)

    async def send_say(self, frame: dict) -> bool:
        """`say {gen, seq, text, message_id, kind, role, t, ...}` — v3 only (returns False on v2).

        Ends every open sentence first (its pending audio and padded tail go out before
        the `say`), so a sentence's audio is never split by the next one's `say`."""
        if self._closed or not self.v3:
            return False
        f = dict(frame)
        f["type"] = "say"
        f.setdefault("t", self.now_ms())
        if not isinstance(f.get("gen"), int) or not isinstance(f.get("seq"), int) \
                or not isinstance(f.get("text"), str):
            raise TypeError("say needs int gen, int seq and str text")
        async with self._audio_lock:
            if self._closed:
                return False
            await self._flush_all(pad=True)
        await self.send(f)
        self.stats["say"] += 1
        return True

    async def send_audio(self, *, kind: str, gen: int, seq: int, pcm16: np.ndarray, t_ms: int | None = None,
                         text: str | None = None, message_id: str | None = None, role: str = "character",
                         chunk: int = 0, intent: str | None = None, last: bool = False) -> None:
        """Ship one chunk of a sentence (int16 at :attr:`rate_out`).

        v2: the JSON ``audio`` header (`text` as given — the session passes it on
        chunk 0 only) and the raw PCM, adjacent. v3: Opus or PCM16 binary messages
        (`text`/`message_id`/`role` travel in the preceding `say`, not here).
        ``last=True`` ends the sentence (same as :meth:`end_sentence`). ``t_ms`` is
        accepted for the contract's shape; v3 stamps send time (:meth:`now_ms`)."""
        if self._closed:
            return
        pcm = np.asarray(pcm16)
        if pcm.dtype.kind == "f":
            pcm = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
        elif pcm.dtype != np.int16:
            pcm = pcm.astype(np.int16)
        pcm = pcm.reshape(-1)
        if not self.v3:
            header = {"type": "audio", "gen": gen, "seq": seq, "chunk": chunk, "sample_rate": self.rate_out,
                      "text": text, "message_id": message_id, "kind": kind, "role": role}
            if intent:
                header["intent"] = intent
            data = json.dumps(header, separators=(",", ":"), ensure_ascii=False)
            async with self._send_lock:
                await self._write_text(data)
                await self._write_bytes(pcm.astype("<i2", copy=False).tobytes())
            self.stats["tx_audio_messages"] += 1
            self.stats["tx_audio_ms"] += len(pcm) * 1000 / self.rate_out
            return
        k = wire3.KINDS.get(kind)
        if k is None:
            raise ValueError(f"unknown audio kind {kind!r}")
        key = (k, int(gen), int(seq))
        async with self._audio_lock:
            if self._closed:
                return
            if self._gen_floor is not None and gen_older(key[1], self._gen_floor):
                # audio of a gen a barge-in dropped (TTS still finishing): expected, silent
                self.stats["late_audio_dropped"] += 1
                return
            if key in self._finished:
                # review-T6 M3: a late chunk for a sentence that already ended must not open
                # a second stream with a second audio_end for the same seq
                self.stats["late_audio_dropped"] += 1
                if not self._late_warned:
                    self._late_warned = True
                    log.warning("voice: audio for an ended sentence %s dropped; call end_sentence(N) "
                                "before send_say(N+1)", key)
                return
            st = await self._stream_for(key)
            if st is None:
                return
            if st.codec == wire3.CODEC_PCM16:
                await self._send_pcm(st, pcm)
            else:
                await self._encode(st, pcm)
                await self._ship(st, force=st.sent_messages == 0)
            if last:
                await self._finish(st)

    async def end_sentence(self, *, kind: str = "reply", gen: int, seq: int) -> None:
        """No more audio for this sentence: pad and send its tail now."""
        k = wire3.KINDS.get(kind)
        async with self._audio_lock:
            if self._closed:
                return
            st = self._streams.get((k, int(gen), int(seq)))
            if st is not None:
                await self._finish(st)

    async def drop_older_gens(self, gen: int) -> None:
        """A barge-in: forget pending audio of every gen below ``gen`` (never sent); later
        audio for those gens is dropped too."""
        async with self._audio_lock:
            gen = int(gen)
            if self._gen_floor is None or gen_older(self._gen_floor, gen):
                self._gen_floor = gen
            for key in [k for k in self._order if gen_older(k[1], gen)]:
                self._close_stream(key)

    async def send_probe(self, n: int) -> int:
        """Answer `probe {bytes}` with one kind-3 binary message of random bytes.

        Returns the payload size, or 0 when refused (v2, rate limit, empty)."""
        if self._closed or not self.v3 or not self.probes.allow():
            return 0
        payload = probe_payload(n)
        if not payload:
            return 0
        msg = wire3.pack_down(wire3.CODEC_PCM16, wire3.KIND_PROBE, 0, 0, self.now_ms(), [payload])
        async with self._audio_lock:
            if self._closed:
                return 0
            await self._flush_all(pad=False)
            async with self._send_lock:
                await self._write_bytes(msg)
        self.stats["probes"] += 1
        return len(payload)

    # ------------------------------------------------------- downlink internals #
    async def _stream_for(self, key: tuple[int, int, int]) -> _Stream | None:
        """The open stream for ``key`` (created on first audio); ``None`` once closed."""
        if self._closed:
            return None
        st = self._streams.get(key)
        if st is not None:
            return st
        # sentences go out one after another: starting one ends every other (padded), so
        # the client receives each sentence's audio contiguously and in order
        await self._flush_all(pad=True)
        codec = wire3.CODEC_OPUS if self.codec_out == "opus" else wire3.CODEC_PCM16
        enc = None
        if codec == wire3.CODEC_OPUS:
            try:
                enc = opus.Encoder(self.rate_out, self.link.down_kbps * 1000, application="audio",
                                   complexity=5, signal="voice", frame_ms=OPUS_FRAME_MS)
            except (opus.OpusUnavailable, opus.OpusError, ValueError) as e:
                log.warning("voice: Opus encoder failed (%s); downlink falls back to PCM16", e)
                self.codec_out = "pcm16"
                codec = wire3.CODEC_PCM16
        if self._closed:                     # closed while the previous sentence was flushing
            if enc is not None:
                enc.close()
            return None
        st = _Stream(key, codec, enc)
        self._streams[key] = st
        self._order.append(key)
        while len(self._order) > MAX_OPEN_STREAMS:
            await self._finish(self._streams[self._order[0]])
        return st

    def _close_stream(self, key) -> None:
        st = self._streams.pop(key, None)
        if key in self._order:
            self._order.remove(key)
        if st is not None:
            self._finished[key] = None
            while len(self._finished) > FINISHED_KEEP:
                self._finished.popitem(last=False)
            if st.enc is not None:
                st.enc.close()

    async def _send_pcm(self, st: _Stream, pcm: np.ndarray) -> None:
        step = self.rate_out * PCM_CHUNK_MS // 1000
        kind, gen, seq = st.key
        for i in range(0, len(pcm), step):
            if self._closed:
                return
            piece = pcm[i:i + step]
            msg = wire3.pack_down(wire3.CODEC_PCM16, kind, gen, seq, self.now_ms(),
                                  [piece.astype("<i2", copy=False).tobytes()])
            async with self._send_lock:
                await self._write_bytes(msg)
            st.sent_messages += 1
            self.stats["tx_audio_messages"] += 1
            self.stats["tx_audio_ms"] += len(piece) * 1000 / self.rate_out

    async def _encode(self, st: _Stream, pcm: np.ndarray, *, pad_tail: bool = False) -> None:
        enc = st.enc
        buf = np.concatenate([st.rem, pcm]) if len(st.rem) else pcm
        st.samples_in += len(pcm)
        if pad_tail:
            need = len(buf) + enc.lookahead
            total = -(-need // enc.frame) * enc.frame
            buf = np.concatenate([buf, np.zeros(total - len(buf), np.int16)])
        nf = len(buf) // enc.frame
        frames = [buf[i * enc.frame:(i + 1) * enc.frame] for i in range(nf)]
        st.rem = buf[nf * enc.frame:].copy()
        if not frames:
            return
        try:
            if len(frames) > THREAD_FRAMES:
                packets = await asyncio.to_thread(lambda: [enc.encode(f) for f in frames])
            else:
                packets = [enc.encode(f) for f in frames]
        except opus.OpusError:
            if enc.closed:                   # close() past CLOSE_WAIT_S: the audio is discarded anyway
                return
            raise
        if st.first_pending_at is None:
            st.first_pending_at = self.clock()
        st.packets.extend(packets)

    async def _ship(self, st: _Stream, *, force: bool = False) -> None:
        """Send full messages; send a partial one when ``force`` (first audio, end, deadline)."""
        kind, gen, seq = st.key
        sent = 0
        if self._closed:
            st.packets = []
            st.first_pending_at = None
            return
        while len(st.packets) >= PACKETS_PER_MESSAGE or (force and st.packets):
            sent += 1
            batch, st.packets = st.packets[:PACKETS_PER_MESSAGE], st.packets[PACKETS_PER_MESSAGE:]
            msg = wire3.pack_down(wire3.CODEC_OPUS, kind, gen, seq, self.now_ms(), batch)
            async with self._send_lock:
                await self._write_bytes(msg)
            st.sent_messages += 1
            self.stats["tx_audio_messages"] += 1
            self.stats["tx_audio_ms"] += len(batch) * OPUS_FRAME_MS
        if not st.packets:
            st.first_pending_at = None
        else:
            if sent:
                # what is left was encoded last (FIFO): its wait starts now
                st.first_pending_at = self.clock()
            self._arm_timer()

    def _arm_timer(self) -> None:
        if self._timer is not None and not self._timer.done():
            return
        self._timer = asyncio.create_task(self._deadline())

    async def _deadline(self) -> None:
        try:
            while not self._closed:
                due = [st.first_pending_at for st in self._streams.values() if st.packets and st.first_pending_at]
                if not due:
                    return
                wait = min(due) + BATCH_MS / 1000 - self.clock()
                if wait > 0:
                    await asyncio.sleep(wait)
                    continue
                async with self._audio_lock:
                    if self._closed:
                        return
                    now = self.clock()
                    for key in list(self._order):
                        st = self._streams.get(key)
                        if st and st.packets and st.first_pending_at is not None \
                                and now - st.first_pending_at >= BATCH_MS / 1000 - 0.001:
                            await self._ship(st, force=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - review-T6 L6: a dead socket; the next _ship re-arms
            log.debug("voice: batching timer stopped (%s: %s)", type(e).__name__, e)

    async def _finish(self, st: _Stream) -> None:
        if st.codec == wire3.CODEC_OPUS and st.enc is not None and (len(st.rem) or st.samples_in):
            await self._encode(st, np.zeros(0, np.int16), pad_tail=True)
        await self._ship(st, force=True)
        self._close_stream(st.key)
        if st.sent_messages and not self._closed:
            # Amendment T6-2: the client can only know a sentence's audio is complete from
            # this (a later `say` may be held up by a stall), and `after_partial` needs it
            kind, gen, seq = st.key
            await self.send({"type": "audio_end", "gen": gen, "seq": seq, "kind": wire3.KIND_NAMES[kind]})

    async def _flush_all(self, *, pad: bool) -> None:
        for key in list(self._order):
            st = self._streams.get(key)
            if st is None:
                continue
            if pad:
                await self._finish(st)
            elif st.packets:
                await self._ship(st, force=True)

    # ------------------------------------------------------- link + boundaries #
    def at_sentence_boundary(self) -> dict:
        """Apply a pending `link` request (call before each new sentence). Returns what changed.

        A `down_kbps` change needs nothing else: each sentence opens a fresh encoder at the
        current rung. `pcm_fallback` switches the downlink codec to PCM16 for good."""
        changed = self.link.at_boundary(self.clock())
        if changed.get("pcm_fallback") and self.codec_out == "opus":
            self.codec_out = "pcm16"
            log.info("voice: the client's Opus decoder failed; downlink is PCM16 from now on")
        return changed

    def link_metrics(self, relay: Mapping | None = None) -> dict:
        """`metrics.link` from the server's side: mode, rungs, why, degraded time, relay, out_ms."""
        m = self.link.metrics(self.clock())
        m["out_ms"] = self.stats["max_send_ms"]
        if relay is not None:
            r = RelayIndicator.relayed(relay)
            if r is not None:
                m["relay"] = r
        return m

    # ----------------------------------------------------------------- uplink #
    @staticmethod
    def too_big(buf) -> bool:
        """True for a binary message the session must drop BEFORE queueing it (A11)."""
        return len(buf) > wire3.MAX_MESSAGE

    def decode_up(self, buf: bytes) -> tuple[int, int, np.ndarray] | None:
        """A v3 uplink message -> ``(off, flags, pcm16 @ 16 kHz)``; ``None`` for a v2 raw
        PCM16 frame (the caller uses ``buf`` as is); ``ValueError`` when malformed."""
        self.stats["rx_messages"] += 1
        self.stats["rx_bytes"] += len(buf)
        if not wire3.is_v3(buf):
            return None
        if self._closed:
            raise ValueError("transport closed")
        limit = int(MAX_UP_DECODED_S * RATE_IN)
        try:
            codec, flags, off, packets = wire3.unpack_up(buf)
            if codec == wire3.CODEC_PCM16:
                if sum(len(p) for p in packets) // 2 > limit:
                    raise ValueError(f"message carries more than {MAX_UP_DECODED_S:g} s of audio")
                pcm = np.frombuffer(b"".join(packets), dtype="<i2").astype(np.int16)
            else:
                dec = self._decoder
                if dec is None:
                    if not opus.available():
                        raise ValueError("Opus uplink but libopus is unavailable")
                    dec = self._decoder = opus.Decoder(RATE_IN)
                # review-T6 L1: bound the DECODED length before decoding anything
                if sum(opus.packet_samples(p, RATE_IN) for p in packets) > limit:
                    raise ValueError(f"message decodes to more than {MAX_UP_DECODED_S:g} s of audio")
                parts = [dec.decode(p) for p in packets]
                self.stats["rx_opus_packets"] += len(packets)
                pcm = np.concatenate(parts) if parts else np.zeros(0, np.int16)
        except (ValueError, opus.OpusError) as e:
            self.stats["bad_frames"] += 1
            raise ValueError(str(e)) from e
        return off, flags, pcm

    # ------------------------------------------------------------------ close #
    async def close(self) -> None:
        """Stop the batching timer and free the codecs (pending audio is discarded).

        Idempotent. Waits (up to :data:`CLOSE_WAIT_S`) for the audio lock, so an encode
        in a worker thread finishes before its encoder is freed; past that (a send stuck
        on a stalled socket) the codecs' own locks still keep the free from overlapping
        a foreign call, and the stuck sender finds the transport closed."""
        self._closed = True
        if self._timer is not None and self._timer is not asyncio.current_task():
            self._timer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._timer
        acquired = False
        try:
            await asyncio.wait_for(self._audio_lock.acquire(), CLOSE_WAIT_S)
            acquired = True
        except TimeoutError:
            log.info("voice: transport closed while a send was still blocked")
        try:
            for key in list(self._order):
                self._close_stream(key)
            dec, self._decoder = self._decoder, None
            if dec is not None:
                dec.close()
        finally:
            if acquired:
                self._audio_lock.release()
