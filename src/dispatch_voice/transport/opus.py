"""ctypes binding to the system libopus (BSD-3-Clause), fail-soft (contracts C-§B.3).

libopus is loaded dynamically, like libsndfile through soundfile: nothing is
compiled or vendored. When it cannot be found (or ``DISPATCH_VOICE_OPUS`` is
``off``/``0``) :func:`available` returns False and the transport negotiates PCM16
both ways; nothing else changes.

    enc = Encoder(24000, 24000)            # 24 kHz in, 24 kbps, 20 ms frames
    pkt = enc.encode(frame_int16)          # exactly enc.frame samples -> one packet
    dec = Decoder(16000)
    pcm = dec.decode(pkt)                  # int16 ndarray at 16 kHz

Measured on this box (libopus 1.6): encode 282 us per 20 ms frame at 24 kHz /
24 kbps, decode 64 us; 16 kHz / 16 kbps encode 216 us, decode 31 us. Foreign
calls release the GIL, so long buffers can be coded in a worker thread.

Each Encoder/Decoder holds a per-instance lock across every foreign call that
touches its state (encode/decode/ctl/reset) AND across ``close()``: a destroy can
never overlap a call running in another thread (review-T6 H1 - that overlap
corrupted the heap of the host process). After ``close()`` every call raises
:class:`OpusError` ("closed"). One instance per stream is still the intended use;
the lock makes misuse safe, not concurrent.
"""
from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import logging
import os
import threading

import numpy as np

log = logging.getLogger("dispatch_voice.transport.opus")

APPLICATION = {"voip": 2048, "audio": 2049, "lowdelay": 2051}
SIGNAL = {"auto": -1000, "voice": 3001, "music": 3002}
RATES = (8000, 12000, 16000, 24000, 48000)
FRAME_MS = (10, 20, 40, 60)

_SET_BITRATE = 4002
_GET_BITRATE = 4003
_SET_VBR = 4006
_SET_COMPLEXITY = 4010
_SET_INBAND_FEC = 4012
_SET_DTX = 4016
_SET_SIGNAL = 4024
_GET_LOOKAHEAD = 4027
_RESET_STATE = 4028
_MAX_PACKET = 1275 * 3          # the largest legal packet (three 20 ms frames of 1275 B)
MIN_BITRATE = 6000
MAX_BITRATE = 510_000


class OpusUnavailable(OSError):
    """libopus could not be loaded (or was switched off)."""


class OpusError(RuntimeError):
    """libopus returned an error code."""


_CANDIDATES = ("libopus.so.0", "libopus.so", "libopus.0.dylib", "libopus.dylib", "opus.dll", "libopus-0.dll")
_lock = threading.Lock()
_LIB = None
_LOAD_ERROR: str | None = None


def _switched_off() -> bool:
    return os.environ.get("DISPATCH_VOICE_OPUS", "auto").strip().lower() in ("off", "0", "false", "no")


def _bind(lib):
    c = ctypes
    lib.opus_get_version_string.restype = c.c_char_p
    lib.opus_get_version_string.argtypes = []
    lib.opus_strerror.restype = c.c_char_p
    lib.opus_strerror.argtypes = [c.c_int]
    lib.opus_encoder_create.restype = c.c_void_p
    lib.opus_encoder_create.argtypes = [c.c_int32, c.c_int, c.c_int, c.POINTER(c.c_int)]
    lib.opus_encoder_destroy.restype = None
    lib.opus_encoder_destroy.argtypes = [c.c_void_p]
    lib.opus_encode.restype = c.c_int32
    lib.opus_encode.argtypes = [c.c_void_p, c.POINTER(c.c_int16), c.c_int, c.POINTER(c.c_ubyte), c.c_int32]
    lib.opus_encoder_ctl.restype = c.c_int          # varargs: the extra argument is typed per call
    lib.opus_decoder_create.restype = c.c_void_p
    lib.opus_decoder_create.argtypes = [c.c_int32, c.c_int, c.POINTER(c.c_int)]
    lib.opus_decoder_destroy.restype = None
    lib.opus_decoder_destroy.argtypes = [c.c_void_p]
    lib.opus_decode.restype = c.c_int
    lib.opus_decode.argtypes = [c.c_void_p, c.POINTER(c.c_ubyte), c.c_int32, c.POINTER(c.c_int16), c.c_int, c.c_int]
    lib.opus_packet_get_nb_samples.restype = c.c_int
    lib.opus_packet_get_nb_samples.argtypes = [c.POINTER(c.c_ubyte), c.c_int32, c.c_int32]
    return lib


def _load():
    global _LIB, _LOAD_ERROR
    with _lock:
        if _LIB is not None or _LOAD_ERROR is not None:
            return _LIB
        names = []
        found = ctypes.util.find_library("opus")
        if found:
            names.append(found)
        names.extend(n for n in _CANDIDATES if n not in names)
        errors = []
        for name in names:
            try:
                _LIB = _bind(ctypes.CDLL(name))
                log.info("voice: libopus %s loaded (%s)", _LIB.opus_get_version_string().decode(), name)
                return _LIB
            except (OSError, AttributeError) as e:
                errors.append(f"{name}: {e}")
        _LOAD_ERROR = "; ".join(errors) or "libopus not found"
        log.warning("voice: libopus unavailable, the voice wire stays PCM16 (%s)", _LOAD_ERROR[:300])
        return None


def available() -> bool:
    """True when libopus is loadable and not switched off by ``DISPATCH_VOICE_OPUS``. Never raises."""
    if _switched_off():
        return False
    try:
        return _load() is not None
    except (OSError, AttributeError, ValueError, TypeError):   # pragma: no cover - fail-soft belt
        return False


def version() -> str:
    """The libopus version string (e.g. ``"libopus 1.6"``), or ``""`` when unavailable."""
    if not available():
        return ""
    return _LIB.opus_get_version_string().decode()


def _lib():
    if not available():
        raise OpusUnavailable(_LOAD_ERROR or "libopus switched off (DISPATCH_VOICE_OPUS)")
    return _LIB


def _check(code: int, what: str) -> int:
    if code < 0:
        msg = _LIB.opus_strerror(code)
        raise OpusError(f"{what}: {msg.decode() if msg else 'error'} ({code})")
    return code


class Encoder:
    """One mono Opus encoder. ``encode`` takes exactly :attr:`frame` int16 samples."""

    def __init__(self, rate: int, bitrate: int, application: str = "audio", complexity: int = 5,
                 signal: str = "voice", frame_ms: int = 20) -> None:
        lib = _lib()
        if rate not in RATES:
            raise ValueError(f"Opus cannot take {rate} Hz input (one of {RATES})")
        if frame_ms not in FRAME_MS:
            raise ValueError(f"frame_ms {frame_ms} not one of {FRAME_MS}")
        if application not in APPLICATION or signal not in SIGNAL:
            raise ValueError("unknown application or signal")
        self.rate = rate
        self.frame_ms = frame_ms
        self.frame = rate * frame_ms // 1000
        self._lk = threading.Lock()
        self._st = None
        err = ctypes.c_int(0)
        st = lib.opus_encoder_create(rate, 1, APPLICATION[application], ctypes.byref(err))
        if err.value < 0 and st:
            lib.opus_encoder_destroy(st)
        _check(err.value, "opus_encoder_create")
        if not st:
            raise OpusError("opus_encoder_create returned NULL")
        self._st = ctypes.c_void_p(st)
        self._buf = (ctypes.c_ubyte * _MAX_PACKET)()
        self.bitrate = 0
        self.set_bitrate(bitrate)
        self._ctl(_SET_COMPLEXITY, max(0, min(10, int(complexity))))
        self._ctl(_SET_SIGNAL, SIGNAL[signal])
        self._ctl(_SET_VBR, 1)
        self._ctl(_SET_INBAND_FEC, 0)       # TCP: no loss to conceal
        self._ctl(_SET_DTX, 0)

    def _live(self):
        """The state pointer; call with ``self._lk`` held."""
        if self._st is None:
            raise OpusError("encoder closed")
        return self._st

    def _ctl(self, request: int, value: int) -> None:
        with self._lk:
            _check(_LIB.opus_encoder_ctl(self._live(), ctypes.c_int(request), ctypes.c_int32(int(value))),
                   f"opus_encoder_ctl({request})")

    def set_bitrate(self, bps: int) -> None:
        """Change the target bitrate (clamped to 6..510 kbps); applies from the next frame."""
        bps = max(MIN_BITRATE, min(MAX_BITRATE, int(bps)))
        self._ctl(_SET_BITRATE, bps)
        self.bitrate = bps

    @property
    def lookahead(self) -> int:
        """Encoder delay in samples at :attr:`rate` (the decoder output lags the input by this)."""
        v = ctypes.c_int32(0)
        with self._lk:
            _check(_LIB.opus_encoder_ctl(self._live(), ctypes.c_int(_GET_LOOKAHEAD), ctypes.byref(v)),
                   "lookahead")
        return v.value

    def reset(self) -> None:
        """Back to the freshly created state (a new, independent stream)."""
        with self._lk:
            _check(_LIB.opus_encoder_ctl(self._live(), ctypes.c_int(_RESET_STATE)), "reset")

    def encode(self, int16_frame: np.ndarray) -> bytes:
        """One frame of int16 samples (exactly :attr:`frame` long) -> one Opus packet."""
        pcm = np.asarray(int16_frame)
        if pcm.dtype != np.int16 or pcm.ndim != 1 or len(pcm) != self.frame:
            raise ValueError(f"need one 1-D int16 frame of {self.frame} samples, got {pcm.dtype} {pcm.shape}")
        pcm = np.ascontiguousarray(pcm)
        with self._lk:
            n = _check(_LIB.opus_encode(self._live(), pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                        self.frame, self._buf, _MAX_PACKET), "opus_encode")
            return bytes(self._buf[:n])

    @property
    def closed(self) -> bool:
        return getattr(self, "_st", None) is None

    def close(self) -> None:
        """Free the encoder; waits for a call in progress in another thread (never overlaps it)."""
        lk = getattr(self, "_lk", None)
        if lk is None:                       # __init__ failed before the lock existed
            return
        with lk:
            st, self._st = self._st, None
            if st and _LIB is not None:
                _LIB.opus_encoder_destroy(st)

    def __del__(self):
        with contextlib.suppress(Exception):     # interpreter shutdown: the library may be gone
            self.close()


class Decoder:
    """One mono Opus decoder producing int16 at ``rate`` (any Opus rate; the packet's own rate is irrelevant)."""

    def __init__(self, rate: int, max_frame_ms: int = 120) -> None:
        lib = _lib()
        if rate not in RATES:
            raise ValueError(f"Opus cannot decode to {rate} Hz (one of {RATES})")
        self.rate = rate
        self._lk = threading.Lock()
        self._st = None
        err = ctypes.c_int(0)
        st = lib.opus_decoder_create(rate, 1, ctypes.byref(err))
        if err.value < 0 and st:
            lib.opus_decoder_destroy(st)
        _check(err.value, "opus_decoder_create")
        if not st:
            raise OpusError("opus_decoder_create returned NULL")
        self._st = ctypes.c_void_p(st)
        self._max = rate * max_frame_ms // 1000
        self._out = np.zeros(self._max, np.int16)

    def decode(self, packet: bytes | None) -> np.ndarray:
        """One packet -> int16 samples at :attr:`rate`. ``None`` = a lost packet (PLC, 20 ms).

        Raises :class:`OpusError` on a corrupt packet (the caller treats it as a bad frame)."""
        buf = None
        if packet is not None:
            if not packet:
                raise OpusError("empty packet")
            buf = (ctypes.c_ubyte * len(packet)).from_buffer_copy(packet)
        with self._lk:
            if self._st is None:
                raise OpusError("decoder closed")
            out = self._out
            optr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
            if buf is None:
                n = _check(_LIB.opus_decode(self._st, None, 0, optr, self.rate // 50, 0), "opus_decode(plc)")
            else:
                n = _check(_LIB.opus_decode(self._st, buf, len(packet), optr, self._max, 0), "opus_decode")
            return out[:n].copy()

    def reset(self) -> None:
        with self._lk:
            if self._st is None:
                raise OpusError("decoder closed")
            _check(_LIB.opus_decoder_ctl(self._st, ctypes.c_int(_RESET_STATE)), "reset")

    @property
    def closed(self) -> bool:
        return getattr(self, "_st", None) is None

    def close(self) -> None:
        """Free the decoder; waits for a call in progress in another thread (never overlaps it)."""
        lk = getattr(self, "_lk", None)
        if lk is None:
            return
        with lk:
            st, self._st = self._st, None
            if st and _LIB is not None:
                _LIB.opus_decoder_destroy(st)

    def __del__(self):
        with contextlib.suppress(Exception):     # interpreter shutdown: the library may be gone
            self.close()


def packet_samples(packet: bytes, rate: int) -> int:
    """How many samples ``packet`` decodes to at ``rate`` (raises on a corrupt packet)."""
    lib = _lib()
    buf = (ctypes.c_ubyte * len(packet)).from_buffer_copy(packet)
    return _check(lib.opus_packet_get_nb_samples(buf, len(packet), rate), "opus_packet_get_nb_samples")


def _reset_for_tests() -> None:
    """Forget the load result (tests that monkeypatch the loader)."""
    global _LIB, _LOAD_ERROR
    with _lock:
        _LIB = None
        _LOAD_ERROR = None
