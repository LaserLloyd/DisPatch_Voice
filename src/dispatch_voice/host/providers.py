"""Speech providers: STT, TTS and turn-taking behind one interface.

Until 1.6.0 the session was welded to concrete classes: ``StreamingSTT`` (pull-style, tied
to the VAD probability), ``TTSEngine`` (blocking iterators, an opaque ``profile_id``) and
``TurnDetector``. This module is the seam a settings page, a plugin and a test can talk to
WITHOUT touching the session engine:

* :class:`STTProvider` ``.transcribe(stream)``: audio in, an async stream of :class:`Partial`
  then one :class:`Final` out;
* :class:`TTSProvider` ``.synthesize(text, voice)``: an async stream of PCM16 mono bytes;
* :class:`VADProvider` ``.detect(stream)`` / ``.build()``: speech and end-of-turn events, or the
  session's own :class:`~dispatch_voice.host.vad_turn.TurnDetector`;
* every provider has ``health()`` (state + reason + fix), ``capabilities()`` and
  ``settings_schema()`` (the schema entries it reads, for the Settings pane).

The built-in adapters WRAP the existing classes, they do not replace them:

========  ====================================  ==========================================
id        STT                                   TTS
========  ====================================  ==========================================
``cpu``   Parakeet / Moonshine tail STT         the configured engine chain (Chatterbox,
          (:class:`StreamingSTTProvider`)       MiniMax, a plugin's) (:class:`EngineTTSProvider`)
``rig``   :class:`RigSTTProvider` over          :class:`RigTTSProvider` over ``ReplyEngine``
          ``RigStreamingSTT``                   (both share one :class:`RigLink` = one
                                                ``RigConnection``)
``openai``  :class:`OpenAISTTProvider`          :class:`EngineTTSProvider` over ``HQTTS``
          (``POST /v1/audio/transcriptions``)   (``POST /v1/audio/speech``)
========  ====================================  ==========================================

Plugins register through the ``dispatch_voice.providers`` entry-point group (a callable
``setup(api)`` that calls ``api.register_provider(...)``) or, deprecated, through
``DISPATCH_VOICE_EXTENSIONS``; see :mod:`host.registry`.
"""
from __future__ import annotations

import asyncio
import io
import logging
import threading
import wave
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import settings_schema as schema

log = logging.getLogger("dispatch.voice")

MIC_RATE = 16000
TTS_RATE = 24000
FRAME = 512                       # samples per VAD frame at 16 kHz (32 ms)

STATES = ("ok", "degraded", "unconfigured", "down")


# --------------------------------------------------------------------------- #
# Value types
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Health:
    """What a provider can do right now: ``ok`` | ``degraded`` (works, with a caveat) |
    ``unconfigured`` (needs a setting or a download) | ``down``. ``reason`` says why in a
    sentence, ``fix`` what to do about it, ``code`` is stable for scripts."""
    state: str = "ok"
    reason: str = ""
    fix: str = ""
    code: str = ""

    @property
    def ok(self) -> bool:
        return self.state in ("ok", "degraded")

    def to_dict(self) -> dict:
        return {"state": self.state, "reason": self.reason, "fix": self.fix, "code": self.code}


@dataclass(frozen=True)
class Partial:
    """An interim transcript: the words so far, which may still change."""
    text: str
    stable_words: int = 0
    t_audio: float = 0.0
    final: bool = False


@dataclass(frozen=True)
class Final:
    """The finished transcript of one utterance."""
    text: str
    t_audio: float = 0.0
    final: bool = True


@dataclass(frozen=True)
class VadEvent:
    """``speech_start`` (a person started talking) or ``turn_end`` (they finished: ``reason``
    says why, ``prob`` is the turn model's confidence). ``at_s`` is seconds into the stream."""
    kind: str
    at_s: float = 0.0
    prob: float | None = None
    reason: str = ""


AudioIn = AsyncIterable  # of bytes (PCM16 LE mono 16 kHz) or int16/float32 numpy arrays


# --------------------------------------------------------------------------- #
# The interfaces
# --------------------------------------------------------------------------- #

class Provider(ABC):
    """Common surface. ``id`` is unique per ``kind``; ``name`` is what the Settings pane shows."""
    kind: str = ""
    id: str = ""
    name: str = ""
    #: Setting ids from the schema this provider reads (``settings_schema`` filters by them).
    setting_ids: tuple[str, ...] = ()

    @abstractmethod
    def health(self) -> Health: ...

    @abstractmethod
    def capabilities(self) -> dict: ...

    def settings_schema(self) -> list[dict]:
        """The schema entries this provider is configured by (``GET /api/voice/settings/schema``
        rows), secrets without values."""
        want = set(self.setting_ids)
        return [s for s in schema.schema_json()["settings"] if s["id"] in want]

    def describe(self) -> dict:
        return {"kind": self.kind, "id": self.id, "name": self.name, "health": self.health().to_dict(),
                "capabilities": self.capabilities(), "settings": [s["id"] for s in self.settings_schema()]}


class STTProvider(Provider):
    kind = "stt"

    @abstractmethod
    def transcribe(self, stream: AudioIn) -> AsyncIterator[Partial | Final]:
        """Consume ``stream`` (one utterance's audio); yield zero or more :class:`Partial`
        and then exactly one :class:`Final`."""


class TTSProvider(Provider):
    kind = "tts"

    @abstractmethod
    def synthesize(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
        """Yield PCM16 LE mono audio at ``capabilities()["sample_rate"]`` for ``text`` in
        ``voice`` (a profile id, ``builtin:<id>``, or None for the default)."""


class VADProvider(Provider):
    kind = "vad"

    @abstractmethod
    def detect(self, stream: AudioIn) -> AsyncIterator[VadEvent]:
        """Speech-start and end-of-turn events over a PCM16 16 kHz mono stream."""

    @abstractmethod
    def build(self):
        """The session's detector (a ``TurnDetector``-shaped object), one per voice chat."""


# --------------------------------------------------------------------------- #
# Audio plumbing shared by the adapters
# --------------------------------------------------------------------------- #

def to_int16(chunk) -> np.ndarray:
    """``bytes`` (PCM16 LE) / int16 / float32 array -> an int16 array."""
    if isinstance(chunk, (bytes, bytearray, memoryview)):
        b = bytes(chunk)
        return np.frombuffer(b[: len(b) - (len(b) % 2)], dtype="<i2").astype(np.int16, copy=False)
    a = np.asarray(chunk)
    if a.dtype == np.int16:
        return a.reshape(-1)
    if a.dtype.kind == "f":
        return (np.clip(a.reshape(-1), -1.0, 1.0) * 32767.0).astype("<i2")
    return a.astype(np.int16).reshape(-1)


def to_float(a16: np.ndarray) -> np.ndarray:
    return a16.astype(np.float32) / 32768.0


def wav_bytes(pcm16: np.ndarray, rate: int = MIC_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(pcm16, "<i2").tobytes())
    return buf.getvalue()


async def collect(stream: AudioIn, *, max_s: float = 600.0) -> np.ndarray:
    """All of ``stream`` as one int16 array (an utterance; ``max_s`` bounds memory)."""
    parts: list[np.ndarray] = []
    n = 0
    async for chunk in stream:
        a = to_int16(chunk)
        parts.append(a)
        n += len(a)
        if n > max_s * MIC_RATE:
            raise ValueError(f"more than {max_s:.0f} s of audio in one utterance")
    return np.concatenate(parts) if parts else np.zeros(0, np.int16)


def default_vad(settings=None):
    """The frame VAD the CPU adapters use: Silero when its model is on disk, else the energy
    gate (a no-model fallback; fine for a short utterance)."""
    from .vad_turn import EnergyVAD, SileroVAD
    path = getattr(settings, "silero_path", None)
    if path is not None and path.exists():
        try:
            return SileroVAD(path)
        except Exception as e:      # noqa: BLE001
            log.warning("voice: Silero VAD unavailable (%s); using the energy gate", e)
    return EnergyVAD()


# --------------------------------------------------------------------------- #
# STT adapters
# --------------------------------------------------------------------------- #

class StreamingSTTProvider(STTProvider):
    """Any ``rt.stt_stream.StreamingSTT`` (``push/tick/final``) as an :class:`STTProvider`.

    ``factory()`` returns a FRESH streaming recogniser for one utterance; ``vad`` (a
    ``frame -> probability`` callable, or a factory of one) feeds its ``vad_prob``."""

    id = "cpu"
    name = "This computer"
    setting_ids = ("rt.stt_engine", "rt.stt_spinning", "rt.stt_threads_while_speaking", "threads")

    def __init__(self, factory: Callable[[], Any], *, vad: Callable | None = None, settings=None,
                 id: str | None = None, name: str | None = None, remote: bool = False,
                 health_fn: Callable[[], Health] | None = None, languages: tuple[str, ...] = ("en",)) -> None:
        self._factory = factory
        self._vad = vad
        self.settings = settings
        if id:
            self.id = id
        if name:
            self.name = name
        self.remote = remote
        self._health_fn = health_fn
        self.languages = languages

    def health(self) -> Health:
        if self._health_fn is not None:
            return self._health_fn()
        s = self.settings
        if s is not None and getattr(s, "stt_engine", "parakeet") in ("parakeet", "moonshine"):
            if not (s.parakeet_dir / "encoder-model.int8.onnx").exists():
                return Health("unconfigured", "The Parakeet speech-recognition model is not downloaded.",
                              "Settings > Voice > Models > Download, or run `dispatch-voice fetch-models`.",
                              "model_missing")
        return Health("ok")

    def capabilities(self) -> dict:
        return {"kind": "stt", "streaming": True, "partials": True, "languages": list(self.languages),
                "sample_rate": MIC_RATE, "formats": ["pcm_s16le"], "remote": self.remote,
                "max_utterance_s": 30.0}

    async def transcribe(self, stream: AudioIn) -> AsyncIterator[Partial | Final]:
        sstt = await asyncio.to_thread(self._factory)
        vad = self._vad() if callable(self._vad) and not hasattr(self._vad, "reset") else self._vad
        if vad is None:
            vad = await asyncio.to_thread(default_vad, self.settings)
        if hasattr(vad, "reset"):
            vad.reset()
        pending = np.zeros(0, np.int16)
        pushed = 0
        try:
            async for chunk in stream:
                pending = np.concatenate([pending, to_int16(chunk)])
                n = len(pending) // FRAME * FRAME
                if n:
                    frames, pending = pending[:n], pending[n:]
                    pushed += n
                    await asyncio.to_thread(self._push_frames, sstt, vad, frames)
                    part = await asyncio.to_thread(sstt.tick)
                    if part is not None and getattr(part, "text", ""):
                        yield Partial(part.text, int(getattr(part, "stable_words", 0)),
                                      float(getattr(part, "t_audio", pushed / MIC_RATE)))
            if len(pending):
                pad = np.zeros(FRAME - len(pending), np.int16)
                await asyncio.to_thread(self._push_frames, sstt, vad, np.concatenate([pending, pad]))
                pushed += len(pending)
            text = await asyncio.to_thread(sstt.final)
            yield Final(str(text or "").strip(), pushed / MIC_RATE)
        finally:
            await asyncio.to_thread(sstt.reset)

    @staticmethod
    def _push_frames(sstt, vad, frames: np.ndarray) -> None:
        for i in range(0, len(frames), FRAME):
            fr = frames[i:i + FRAME]
            sstt.push(fr, vad_prob=float(vad(to_float(fr))))


class OpenAISTTProvider(STTProvider):
    """Any server that speaks OpenAI's ``POST /v1/audio/transcriptions`` (whisper.cpp,
    faster-whisper-server, vLLM, OpenAI). Whole-utterance: no partials, one :class:`Final`."""

    id = "openai"
    name = "OpenAI-compatible server"
    setting_ids = ("stt_url", "stt_model", "stt_timeout_s", "stt_api_key")
    MAX_BYTES = 25 * 1024 * 1024

    def __init__(self, url: str, model: str = "", *, api_key: str = "", timeout_s: float = 20.0,
                 language: str | None = "en", transport=None, allow: tuple[str, ...] = ()) -> None:
        self.url = (url or "").rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.language = language
        self.transport = transport
        self.allow = allow
        self._last: Health | None = None

    @property
    def endpoint(self) -> str:
        base = self.url[:-3] if self.url.endswith("/v1") else self.url
        return base + "/v1/audio/transcriptions"

    def health(self) -> Health:
        if not self.url:
            return Health("unconfigured", "No server address is set.", "Enter the server's address (stt_url).",
                          "stt_url_missing")
        from . import netpolicy
        v = netpolicy.check_url_policy(self.url, self.allow, schemes=("http", "https"))
        if not v.ok:
            return Health("down", v.reason, v.fix, v.code)
        if self._last is not None and not self._last.ok:
            return self._last
        return Health("degraded", v.warning, "Use https:// or a VPN.", v.code) if v.warning else Health("ok")

    def capabilities(self) -> dict:
        return {"kind": "stt", "streaming": False, "partials": False, "languages": [self.language or "auto"],
                "sample_rate": MIC_RATE, "formats": ["pcm_s16le"], "remote": True, "max_utterance_s": 600.0}

    def _request_parts(self, pcm16: np.ndarray) -> tuple[dict, dict, dict]:
        data: dict = {"response_format": "json"}
        if self.model:
            data["model"] = self.model
        if self.language:
            data["language"] = self.language
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return data, {"file": ("speech.wav", wav_bytes(pcm16), "audio/wav")}, headers

    def transcribe_sync(self, pcm16: np.ndarray) -> str:
        import httpx
        data, files, headers = self._request_parts(pcm16)
        with httpx.Client(transport=self.transport, timeout=self.timeout_s, trust_env=False) as c:
            r = c.post(self.endpoint, data=data, files=files, headers=headers)
        return self._text_of(r)

    def _text_of(self, r) -> str:
        try:
            r.raise_for_status()
            body = r.json()
        except Exception as e:
            self._last = Health("down", f"The server answered with an error ({type(e).__name__}).",
                                "Check the address and that the server is running.", "stt_http_error")
            raise
        self._last = None
        return str((body or {}).get("text") or "").strip()

    async def transcribe(self, stream: AudioIn) -> AsyncIterator[Partial | Final]:
        import httpx
        pcm = await collect(stream)
        if len(pcm) * 2 > self.MAX_BYTES:
            raise ValueError("the utterance is larger than the server accepts")
        if len(pcm) < MIC_RATE // 10:
            yield Final("", len(pcm) / MIC_RATE)
            return
        data, files, headers = self._request_parts(pcm)
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout_s, trust_env=False) as c:
            r = await c.post(self.endpoint, data=data, files=files, headers=headers)
        yield Final(self._text_of(r), len(pcm) / MIC_RATE)


class OpenAISTTEngine:
    """The ``transcribe(float32 audio) -> str`` shape ``rt.stt_stream.ParakeetTailSTT`` takes,
    over an :class:`OpenAISTTProvider`: how a remote recogniser plugs into the unchanged
    session (``stt_provider = openai``)."""

    def __init__(self, provider: OpenAISTTProvider) -> None:
        self.provider = provider

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size < 1600:
            return ""
        return self.provider.transcribe_sync(to_int16(audio))


# --------------------------------------------------------------------------- #
# TTS adapters
# --------------------------------------------------------------------------- #

class EngineTTSProvider(TTSProvider):
    """Any ``host.tts.TTSEngine`` (``stream(text, profile_id, stop)`` of int16 chunks) as a
    :class:`TTSProvider`; ``engine`` may be an engine or a zero-argument factory (built on
    first use, off the loop)."""

    id = "cpu"
    name = "This computer"
    setting_ids = ("tts_engine", "builtin_voice", "speed", "tts_threads", "expressive_tags")

    def __init__(self, engine, *, id: str | None = None, name: str | None = None, remote: bool = False,
                 sample_rate: int | None = None, cloning: bool = False, builtin_voices: bool = False,
                 health_fn: Callable[[], Health] | None = None, setting_ids: tuple[str, ...] | None = None,
                 max_text: int | None = None) -> None:
        self._engine_or_factory = engine
        self._engine = None if callable(engine) and not hasattr(engine, "stream") else engine
        self._lock = threading.Lock()
        if id:
            self.id = id
        if name:
            self.name = name
        self.remote = remote
        self._rate = sample_rate
        self.cloning = cloning
        self.builtin_voices = builtin_voices
        self._health_fn = health_fn
        self.max_text = max_text
        if setting_ids is not None:
            self.setting_ids = setting_ids
        self._last: Health | None = None

    def _get(self):
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    self._engine = self._engine_or_factory()
        return self._engine

    def health(self) -> Health:
        if self._health_fn is not None:
            h = self._health_fn()
            if not h.ok:
                return h
        if self._last is not None and not self._last.ok:
            return self._last
        return Health("ok")

    def capabilities(self) -> dict:
        eng = self._engine
        rate = self._rate or int(getattr(eng, "sample_rate", TTS_RATE) or TTS_RATE)
        return {"kind": "tts", "streaming": bool(getattr(eng, "streaming", True)), "sample_rate": rate,
                "format": "pcm_s16le", "channels": 1, "cloning": self.cloning,
                "builtin_voices": self.builtin_voices, "remote": self.remote, "max_text": self.max_text,
                "languages": ["en"]}

    async def synthesize(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
        engine = await asyncio.to_thread(self._get)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def produce() -> None:
            try:
                for c in engine.stream(text, voice, stop):
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(q.put_nowait, to_int16(c).astype("<i2").tobytes())
            except BaseException as e:      # noqa: BLE001 - handed to the consumer
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        fut = loop.run_in_executor(None, produce)
        try:
            while (item := await q.get()) is not None:
                if isinstance(item, BaseException):
                    self._last = Health("down", f"The voice engine failed ({type(item).__name__}).",
                                        "Check the engine's server and settings.", "tts_failed")
                    raise item
                self._last = None
                yield item
        finally:
            stop.set()
            await fut


def openai_tts_provider(url: str, model: str = "", *, timeout_s: float = 4.0, sample_rate: int = TTS_RATE,
                        api_key: str = "", transport=None, allow: tuple[str, ...] = ()) -> EngineTTSProvider:
    """``POST {url}/v1/audio/speech`` as a provider, wrapping the existing ``HQTTS``."""
    from .tts import HQTTS

    def health() -> Health:
        if not url:
            return Health("unconfigured", "No server address is set.", "Enter the server's address (hq_url).",
                          "hq_url_missing")
        from . import netpolicy
        v = netpolicy.check_url_policy(url, allow, schemes=("http", "https"))
        if not v.ok:
            return Health("down", v.reason, v.fix, v.code)
        if v.warning:
            return Health("degraded", v.warning, "Use https:// or a VPN.", v.code)
        return Health("ok")

    eng = HQTTS(url, model, timeout_s, sample_rate, api_key=api_key, transport=transport)
    return EngineTTSProvider(eng, id="openai", name="OpenAI-compatible server", remote=True,
                             sample_rate=sample_rate, health_fn=health,
                             setting_ids=("hq_url", "hq_model", "hq_timeout_s", "hq_api_key"))


# --------------------------------------------------------------------------- #
# The rig: ONE connection object feeding an STT provider and a TTS provider
# --------------------------------------------------------------------------- #

class RigLink:
    """The SpeechForge connection, owned once and shared by :class:`RigSTTProvider` and
    :class:`RigTTSProvider` (the two used to reach it through one ``RigSession`` inside the
    session; here it is a first-class object). ``conn`` / ``session`` may be injected (tests,
    the session's own); otherwise :meth:`open` dials from the settings on the running loop."""

    def __init__(self, settings=None, *, token: str | None = None, connect=None, http_transport=None,
                 conn=None, session=None) -> None:
        self.settings = settings
        self._token = token
        self._connect = connect
        self._http_transport = http_transport
        self.session = session
        self.conn = conn if conn is not None else getattr(session, "conn", None)

    def _rig_settings(self):
        from . import rig
        return getattr(self.settings, "rig", None) or rig.RigSettings()

    def health(self) -> Health:
        from . import rig
        rs = self._rig_settings()
        token = self._token if self._token is not None else rig.read_token().value
        if not rs.url:
            return Health("unconfigured", "No rig address is set.", "Enter the rig's ws:// address.", "rig_url_missing")
        verdict = rig.url_verdict(rs.url)
        if not verdict.ok:
            return Health("down", verdict.reason, verdict.fix, verdict.code)
        if not token:
            return Health("unconfigured", "The rig token is not set.",
                          "Paste the token in Settings > Voice, or set SPEECHFORGE_TOKEN.", "rig_token_missing")
        if self.conn is None:
            return Health("degraded" if not rs.enabled else "ok",
                          "" if rs.enabled else "The rig is configured but switched off.",
                          "" if rs.enabled else "Turn on 'Use the GPU rig'.", "rig_off" if not rs.enabled else "")
        if self.conn.usable():
            return Health("degraded", verdict.warning, "Use wss:// or a VPN.", verdict.code) if verdict.warning \
                else Health("ok")
        reason = getattr(self.conn, "reason", None) or "not proven yet"
        return Health("degraded", f"The rig is not serving right now ({reason}); the CPU speaks instead.",
                      "Check that the rig is running and the lease is held.", f"rig_{reason}")

    async def open(self):
        """Dial now (on the running loop). Returns the :class:`RigSession` or None."""
        if self.session is not None:
            return self.session
        from . import rig
        self.session = rig.open_for_session(self.settings, token=self._token, connect=self._connect,
                                            http_transport=self._http_transport)
        self.conn = getattr(self.session, "conn", None)
        return self.session

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()
        elif self.conn is not None:
            await self.conn.close()


class RigSTTProvider(StreamingSTTProvider):
    """Speech recognition on the rig, the CPU recogniser behind it (the existing
    ``RigStreamingSTT``: a lost or late rig replays the utterance into the CPU)."""

    id = "rig"
    name = "Speech rig (GPU)"
    setting_ids = ("rig.enabled", "rig.url", "rig.stt_final_deadline_ms", "rig.token")

    def __init__(self, link: RigLink, cpu_factory: Callable[[], Any], *, vad=None, settings=None) -> None:
        from ..rt.stt_rig import RigStreamingSTT
        self.link = link

        def factory():
            return RigStreamingSTT(cpu_factory(), link.conn)

        super().__init__(factory, vad=vad, settings=settings, remote=True, health_fn=link.health)

    def capabilities(self) -> dict:
        c = super().capabilities()
        c["partials"] = True
        c["fallback"] = "cpu"
        return c


class RigTTSProvider(EngineTTSProvider):
    """Speech on the rig with the CPU engine as the fallback (the existing ``ReplyEngine``)."""

    id = "rig"
    name = "Speech rig (GPU)"
    setting_ids = ("rig.enabled", "rig.url", "rig.tts_start_deadline_ms", "rig.ahead_ms",
                   "rig.first_chunk_tokens", "rig.token")

    def __init__(self, link: RigLink, cpu_engine) -> None:
        """``cpu_engine``: the CPU engine behind the rig, or a zero-argument factory of one."""
        from .engines.rig_tts import ReplyEngine
        self.link = link
        self._cpu = cpu_engine
        self._rids = 0

        def build():
            cpu = cpu_engine if hasattr(cpu_engine, "stream") else cpu_engine()
            return ReplyEngine(cpu, link.session)

        super().__init__(build, id="rig", name="Speech rig (GPU)", remote=True, sample_rate=TTS_RATE,
                         cloning=True, health_fn=link.health, setting_ids=self.setting_ids)

    async def synthesize(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
        if self.link.session is None:
            await self.link.open()
        engine = await asyncio.to_thread(self._get)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        self._rids += 1
        rid = f"provider-{self._rids}"

        def produce() -> None:
            try:
                for c in engine.stream_reply(text, voice, stop, gen=0, seq=1, rid=rid):
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(q.put_nowait, to_int16(c).astype("<i2").tobytes())
            except BaseException as e:      # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        fut = loop.run_in_executor(None, produce)
        try:
            while (item := await q.get()) is not None:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            await fut


# --------------------------------------------------------------------------- #
# VAD / turn taking
# --------------------------------------------------------------------------- #

class CpuVADProvider(VADProvider):
    """Silero VAD + Smart Turn v3.2 + the Endpointer, as built by ``vad_turn.build_detector``."""

    id = "cpu"
    name = "This computer (Silero + Smart Turn)"
    setting_ids = ("merge_window_s", "hard_silence_s", "rt.probe_silence_s", "rt.hold_min_s", "rt.hold_max_s")

    def __init__(self, settings=None, *, factory: Callable[[], Any] | None = None) -> None:
        self.settings = settings
        self._factory = factory

    def health(self) -> Health:
        s = self.settings
        if s is not None and not s.silero_path.exists():
            return Health("unconfigured", "The Silero voice-activity model is not downloaded.",
                          "Settings > Voice > Models > Download, or run `dispatch-voice fetch-models`.",
                          "model_missing")
        if s is not None and not s.smart_turn_path.exists():
            return Health("degraded", "Smart Turn is missing; a fixed silence ends each turn instead.",
                          "Download the models to get natural turn-taking.", "smart_turn_missing")
        return Health("ok")

    def capabilities(self) -> dict:
        s = self.settings
        return {"kind": "vad", "sample_rate": MIC_RATE, "frame_samples": FRAME,
                "turn_model": "smart-turn-v3.2" if s is not None and s.smart_turn_path.exists() else None,
                "remote": False}

    def build(self):
        if self._factory is not None:
            return self._factory()
        from .vad_turn import build_detector
        return build_detector(self.settings)

    async def detect(self, stream: AudioIn) -> AsyncIterator[VadEvent]:
        from .vad_turn import SpeechStart, TurnEnd
        det = await asyncio.to_thread(self.build)
        fed = 0
        async for chunk in stream:
            pcm = to_int16(chunk)
            events = await asyncio.to_thread(det.feed, pcm.astype("<i2").tobytes(), None)
            fed += len(pcm)
            for ev in events:
                if isinstance(ev, SpeechStart):
                    yield VadEvent("speech_start", fed / MIC_RATE)
                elif isinstance(ev, TurnEnd):
                    yield VadEvent("turn_end", fed / MIC_RATE, getattr(ev, "turn_prob", None),
                                   str(getattr(ev, "reason", "")))
        te = await asyncio.to_thread(det.end_turn, "end_of_stream")
        if te is not None:
            yield VadEvent("turn_end", fed / MIC_RATE, getattr(te, "turn_prob", None), str(te.reason))


# --------------------------------------------------------------------------- #
# Registry + discovery
# --------------------------------------------------------------------------- #

ENTRY_POINT_GROUP = "dispatch_voice.providers"


class ProviderRegistry:
    """Provider FACTORIES by ``(kind, id)``: nothing heavy is built until asked. Built-ins
    are added by :func:`register_builtins`; a plugin adds its own with
    ``api.register_provider(kind, id, name, factory)``."""

    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], tuple[str, Callable[[Any], Provider], str]] = {}

    def register(self, kind: str, id: str, name: str, factory: Callable[[Any], Provider], *,
                 origin: str = "builtin") -> None:
        if kind not in ("stt", "tts", "vad"):
            raise ValueError(f"provider kind must be stt, tts or vad, not {kind!r}")
        if not id or not id.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"bad provider id {id!r}")
        self._factories[(kind, id)] = (name, factory, origin)

    def drop_plugins(self) -> None:
        """Tests: forget what plugins registered (the built-ins stay)."""
        for key in [k for k, v in self._factories.items() if v[2] != "builtin"]:
            del self._factories[key]

    def ids(self, kind: str) -> list[str]:
        return [i for (k, i) in self._factories if k == kind]

    def build(self, kind: str, id: str, settings=None) -> Provider:
        try:
            _, factory, _ = self._factories[(kind, id)]
        except KeyError:
            raise LookupError(f"no {kind} provider {id!r}") from None
        return factory(settings)

    def catalog(self, settings=None) -> list[dict]:
        """Every registered provider with its health and capabilities (for the Settings pane
        and ``GET /api/voice/status``). A factory that raises is listed ``down``."""
        out = []
        for (kind, id), (name, factory, origin) in self._factories.items():
            try:
                d = factory(settings).describe()
            except Exception as e:      # noqa: BLE001
                d = {"kind": kind, "id": id, "name": name,
                     "health": Health("down", f"Could not start ({type(e).__name__}).", "", "provider_error").to_dict(),
                     "capabilities": {}, "settings": []}
            d["origin"] = origin
            out.append(d)
        return out


REGISTRY = ProviderRegistry()


def discover_entry_points(enabled: set[str] | None = None, *, group: str = ENTRY_POINT_GROUP,
                          eps=None) -> list[dict]:
    """The installed provider plugins: ``[{name, target, enabled}]`` from the
    ``dispatch_voice.providers`` entry-point group (``eps`` overrides discovery, for tests).
    Listing never imports a plugin; :func:`load_entry_points` does, for the enabled ones."""
    if eps is None:
        from importlib.metadata import entry_points
        eps = entry_points(group=group)
    return [{"name": ep.name, "target": ep.value, "enabled": enabled is not None and ep.name in enabled}
            for ep in eps]


def load_entry_points(api, enabled: set[str], *, group: str = ENTRY_POINT_GROUP, eps=None) -> list[str]:
    """Import and run (``setup(api)``) the ENABLED entry-point plugins; returns the names that
    loaded. A plugin that raises is logged and skipped, like the env-var path."""
    if eps is None:
        from importlib.metadata import entry_points
        eps = entry_points(group=group)
    loaded: list[str] = []
    for ep in eps:
        if ep.name not in enabled:
            continue
        try:
            ep.load()(api)
            loaded.append(ep.name)
            log.info("voice: provider plugin %s loaded", ep.name)
        except Exception:
            log.exception("voice: provider plugin %s failed to load; skipped", ep.name)
    return loaded


def register_builtins(registry: ProviderRegistry = REGISTRY) -> None:
    """The providers this package ships. Factories take the ``VoiceSettings`` (or None)."""
    def cpu_stt(settings):
        from ..rt.stt_stream import build_streaming_stt
        from . import stt as host_stt
        return StreamingSTTProvider(
            lambda: build_streaming_stt(settings, host_stt.shared_engine(settings)), settings=settings)

    def rig_stt(settings):
        from ..rt.stt_stream import build_streaming_stt
        from . import stt as host_stt
        link = RigLink(settings)
        return RigSTTProvider(link, lambda: build_streaming_stt(settings, host_stt.shared_engine(settings)),
                              settings=settings)

    def openai_stt(settings):
        from . import netpolicy
        return OpenAISTTProvider(getattr(settings, "stt_url", ""), getattr(settings, "stt_model", ""),
                                 api_key=_env_secret("DISPATCH_VOICE_STT_API_KEY"),
                                 timeout_s=float(getattr(settings, "stt_timeout_s", 20.0)),
                                 allow=netpolicy.parse_allow(getattr(getattr(settings, "rig", None), "allow", ())))

    def cpu_tts(settings):
        from . import tts as host_tts
        return EngineTTSProvider(lambda: host_tts.shared_engine(settings), cloning=True, builtin_voices=True)

    def rig_tts(settings):
        from . import tts as host_tts
        return RigTTSProvider(RigLink(settings), lambda: host_tts.shared_engine(settings))

    def openai_tts(settings):
        return openai_tts_provider(getattr(settings, "hq_url", ""), getattr(settings, "hq_model", ""),
                                   timeout_s=getattr(settings, "hq_timeout_s", 4.0),
                                   api_key=_env_secret("DISPATCH_VOICE_HQ_API_KEY"))

    registry.register("stt", "cpu", "This computer (Parakeet)", cpu_stt)
    registry.register("stt", "rig", "Speech rig (GPU)", rig_stt)
    registry.register("stt", "openai", "OpenAI-compatible server", openai_stt)
    registry.register("tts", "cpu", "This computer", cpu_tts)
    registry.register("tts", "rig", "Speech rig (GPU)", rig_tts)
    registry.register("tts", "openai", "OpenAI-compatible server", openai_tts)
    registry.register("vad", "cpu", "This computer (Silero + Smart Turn)", lambda s: CpuVADProvider(s))


def _env_secret(name: str) -> str:
    import os
    return os.environ.get(name, "").strip()


register_builtins()
