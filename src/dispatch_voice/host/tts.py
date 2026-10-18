"""Text-to-speech for voice chat: sentence splitting + a swappable engine.

The reply arrives as a token stream (``stream_chunk`` frames, or one whole
``message``). Waiting for the end would put the whole reply's generation time
in front of the first word, so :class:`SentenceSplitter` hands out each
sentence as soon as it is complete and the session synthesises it while the
next one is still being written.

What gets spoken is not what is on screen: :func:`speakable` (T5's
``text/speech_text.normalise``) drops code blocks, links, media directives,
reaction markers, markdown punctuation and emoji and reads numbers, money,
dates and times as words. A listener cannot use a code block read aloud;
"the code is in the chat" is the honest substitute. The splitter returns
source :class:`~dispatch_voice.text.Span` s so the session can cut the stored
markdown at a word boundary when a reply is interrupted (SPEC §4.8).

Engines share one small interface (see :class:`TTSEngine`), all blocking —
the session runs them in a worker thread:

* ``sample_rate`` / ``sample_format`` (``"pcm_s16le"``) / ``channels`` (1) /
  ``streaming`` are declared on the engine;
* ``stream(text, profile_id=None, stop=None)`` yields int16 chunks;
* ``synth(text, profile_id=None) -> (int16 ndarray, rate)``.

``profile_id`` is opaque here: it names a voice the engine knows (a cloned
voice profile, or a built-in voice as ``builtin:<id>`` from a plugin's
built-in voice provider). ``None`` = the default. An engine with a fallback
behind it that does not know an id raises ``LookupError`` before any audio,
which makes :class:`FallbackTTS` hand the sentence to the fallback without
putting the primary engine on a cooldown.

Engines:

* ``chatterbox`` (app/voice/dv.py): the ``dispatch-voice`` package,
  Chatterbox-Turbo ONNX (MIT, ungated), cloned voice profiles plus a stock
  synthetic voice. The DEFAULT when installed.
* Plugin engines registered through ``app.voice.registry.register_engine``,
  picked by name with ``DISPATCH_VOICE_TTS``; a ``role="fallback"`` engine
  sits behind the primary one. The optional, separately installed
  ``dispatch-voice-kokoro`` plugin (GPL-3.0, because Kokoro's phonemizer is
  espeak-ng) adds the fast ``kokoro`` engine and its built-in voices this
  way. DisPatch never imports it by name.
* :class:`HQTTS`: optional "HQ" engine behind an OpenAI-compatible
  ``/v1/audio/speech``; always wrapped in :class:`FallbackTTS`.
* :class:`MiniMaxTTS`: MiniMax T2A v2 over HTTPS, opt-in only (network
  dependency; Token Plan coverage of speech minutes NOT verified).
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Protocol

import numpy as np

log = logging.getLogger("dispatch.voice")

# --------------------------------------------------------------------------- #
# Cleaning and sentence splitting: T5's text/speech_text.py (SPEC §6.3)
# --------------------------------------------------------------------------- #

from ..text.speech_text import SentenceSplitter, normalise

__all__ = ["REPLY_ACTIVITY", "SentenceSplitter", "normalise", "speak", "speakable", "to_pcm16"]


def speakable(text: str, *, tags_ok: bool = False) -> str:
    """Markdown reply → plain words worth saying out loud ('' = nothing).

    ``text/speech_text.normalise``: markdown stripped (fences become "The code is
    in the chat."), bracket tags filtered (the nine Chatterbox tags kept only
    when ``tags_ok``), symbols and emoji dropped, numbers/money/dates/times read
    the way people say them (English-shaped text only)."""
    return normalise(text or "", tags_ok=tags_ok)


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #

#: Prefix of a built-in voice id (from a plugin's provider) where a profile id
#: is expected.
BUILTIN = "builtin:"


def builtin_id(profile_id) -> str | None:
    """``builtin:af_heart`` → ``af_heart``; anything else → None."""
    if isinstance(profile_id, str) and profile_id.startswith(BUILTIN):
        return profile_id[len(BUILTIN):] or None
    return None


class _Activity:
    """How many REPLY sentences are being synthesised right now.

    Background renders (acknowledgements, the spoken error lines, start-up
    warm-up) call :meth:`wait_idle` before each line, so they never queue
    ahead of a reply on an engine lock (review M2: six ack renders held the
    engine lock for ~20 s and starved the first reply)."""

    def __init__(self) -> None:
        self._n = 0
        self._last = 0.0
        self._cv = threading.Condition()

    def enter(self) -> None:
        with self._cv:
            self._n += 1

    def exit(self) -> None:
        import time
        with self._cv:
            self._n = max(0, self._n - 1)
            self._last = time.monotonic()
            self._cv.notify_all()

    @property
    def busy(self) -> bool:
        return self._n > 0

    def wait_idle(self, quiet_s: float = 0.3, timeout: float | None = None) -> bool:
        import time
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while True:
                now = time.monotonic()
                gap = quiet_s - (now - self._last)
                if self._n == 0 and gap <= 0:
                    return True
                if deadline is not None and now >= deadline:
                    return False
                wait = max(0.02, gap if self._n == 0 else 0.25)
                if deadline is not None:
                    wait = min(wait, max(0.0, deadline - now))
                self._cv.wait(wait)


REPLY_ACTIVITY = _Activity()


def to_pcm16(audio) -> np.ndarray:
    a = np.clip(np.asarray(audio, np.float32).reshape(-1), -1.0, 1.0)
    return (a * 32767.0).astype("<i2")


class TTSEngine(Protocol):
    """What the session needs from a TTS engine.

    Optional, used when present: ``cache_tag(profile_id) -> str | None``
    (opt-in disk cache: a tag only when the output for that voice is a pure
    function of the text), ``apply_settings(settings)`` (voice/speed
    preferences changed in the app), ``warm()`` (start-up prewarm) and
    ``has_fallback`` (set by :func:`build_engine` when a fallback engine sits
    behind it)."""
    sample_rate: int
    sample_format: str        # always "pcm_s16le" today
    channels: int             # always 1
    streaming: bool           # yields several chunks per sentence

    def stream(self, text: str, profile_id: str | None = None,
               stop: threading.Event | None = None) -> Iterator[np.ndarray]: ...

    def synth(self, text: str, profile_id: str | None = None) -> tuple[np.ndarray, int]: ...


class _Base:
    sample_format = "pcm_s16le"
    channels = 1
    streaming = False

    def stream(self, text, profile_id=None, stop=None):
        if stop is not None and stop.is_set():
            return
        pcm, _ = self.synth(text, profile_id)
        yield pcm


class MiniMaxTTS(_Base):
    """MiniMax T2A v2 (non-streaming per sentence). Untested against the live
    API from this repo — see docs/voice-chat/host.md."""

    sample_rate = 24000

    def __init__(self, api_key: str, model: str, voice: str, base_url: str) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.url = base_url.rstrip("/") + "/v1/t2a_v2"

    def synth(self, text: str, profile_id: str | None = None) -> tuple[np.ndarray, int]:
        import httpx
        body = {
            "model": self.model, "text": text, "stream": False,
            "voice_setting": {"voice_id": self.voice, "speed": 1.0, "vol": 1.0, "pitch": 0},
            "audio_setting": {"sample_rate": self.sample_rate, "format": "pcm", "channel": 1},
        }
        r = httpx.post(self.url, json=body, timeout=20.0,
                       headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        data = r.json()
        base = data.get("base_resp") or {}
        if base.get("status_code") not in (0, None):
            raise RuntimeError(f"MiniMax T2A error {base.get('status_code')}")
        audio_hex = (data.get("data") or {}).get("audio") or ""
        return np.frombuffer(bytes.fromhex(audio_hex), dtype="<i2"), self.sample_rate


class HQTTS(_Base):
    """OpenAI-compatible ``POST {url}/v1/audio/speech`` returning raw PCM16
    mono at ``sample_rate`` (``response_format: "pcm"``), e.g. a GPU engine on
    another machine. NOT verified against a live server — no such server
    exists yet; this is the contract it is wired against."""

    def __init__(self, url: str, model: str, timeout_s: float, sample_rate: int, *,
                 api_key: str = "", transport=None) -> None:
        base = url.rstrip("/")
        base = base[:-3] if base.endswith("/v1") else base          # a base URL that already ends in /v1
        self.url = base + "/v1/audio/speech"
        self.model = model
        self.timeout_s = timeout_s
        self.sample_rate = sample_rate
        self.api_key = api_key                # sent as a bearer token, never logged
        self.transport = transport            # tests; None = httpx's own

    def synth(self, text: str, profile_id: str | None = None) -> tuple[np.ndarray, int]:
        import httpx
        body = {"model": self.model, "input": text, "response_format": "pcm",
                "sample_rate": self.sample_rate}
        if profile_id:
            body["voice"] = profile_id
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        if self.transport is None:
            r = httpx.post(self.url, json=body, timeout=self.timeout_s, headers=headers)
        else:
            with httpx.Client(transport=self.transport, timeout=self.timeout_s) as c:
                r = c.post(self.url, json=body, headers=headers)
        r.raise_for_status()      # 409 (lease) / 503 (priority hold) -> fallback
        return np.frombuffer(r.content, dtype="<i2"), self.sample_rate


class FallbackTTS:
    """Speak with ``primary``; if it fails BEFORE producing audio, speak that
    sentence with ``fallback``.

    Two uses: a remote "HQ" engine with a local fallback (``cooldown_s`` keeps
    a dead remote from costing one timeout per sentence), and the cloned-voice
    engine with a plugin's built-in voice behind it (no voice profile chosen
    yet → ``LookupError`` → built-in voice; no cooldown, so the moment a
    default profile exists it is used). An error after audio has started ends that
    sentence instead — two voices spliced mid-sentence is worse than a cut.
    Both engines must share the sample rate; a primary that does not is
    treated as failed.
    """

    sample_format = "pcm_s16le"
    channels = 1

    def __init__(self, primary, fallback, cooldown_s: float = 120.0) -> None:
        self.primary, self.fallback = primary, fallback
        if hasattr(primary, "has_fallback"):
            # Tell the primary it is not the last resort: a voice it does not
            # know may raise LookupError and the fallback will speak it.
            primary.has_fallback = True
        self.sample_rate = fallback.sample_rate
        self.streaming = bool(getattr(primary, "streaming", False))
        self.cooldown_s = cooldown_s
        self._down_until = 0.0

    def stream(self, text, profile_id=None, stop=None):
        import time
        if time.monotonic() >= self._down_until and \
                getattr(self.primary, "sample_rate", None) == self.sample_rate:
            started = False
            try:
                for chunk in self.primary.stream(text, profile_id, stop):
                    started = True
                    yield chunk
                if started:
                    return
            except LookupError as e:
                log.debug("voice: %s; using the fallback voice", e)
            except Exception as e:
                if started:
                    log.warning("voice: TTS failed mid-sentence: %s", e)
                    return
                log.info("voice: primary TTS unavailable (%s); using fallback", e)
                self._down_until = time.monotonic() + self.cooldown_s
            if stop is not None and stop.is_set():
                return
        yield from self.fallback.stream(text, profile_id, stop)

    def synth(self, text, profile_id=None):
        parts = list(self.stream(text, profile_id))
        return (np.concatenate(parts) if parts else np.zeros(0, "<i2")), self.sample_rate


def walk(engine) -> Iterator:
    """The engine and every engine wrapped inside it (FallbackTTS layers)."""
    seen: set[int] = set()
    stack = [engine]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        yield e
        stack += [getattr(e, "primary", None), getattr(e, "fallback", None)]


def speaker(engine, profile_id):
    """The engine of a chain that actually speaks ``profile_id``.

    Through every :class:`FallbackTTS` layer a built-in voice
    (``builtin:<id>``) goes to the fallback — the cloned-voice engine raises
    ``LookupError`` for it — and anything else to the primary. Callers pass
    the RESOLVED voice (``routes.effective_voice``): "no voice" (None) on a box
    whose default is a built-in voice must be resolved first, or it would land
    on the cloned engine that never speaks it (FR-works 3)."""
    e, seen = engine, set()
    while e is not None and id(e) not in seen and hasattr(e, "primary") and hasattr(e, "fallback"):
        seen.add(id(e))
        e = e.fallback if builtin_id(profile_id) else e.primary
    return e


def cache_tag(engine, profile_id) -> str | None:
    """The disk-cache tag of the engine in this chain that speaks
    ``profile_id`` deterministically, or None (no disk cache). Engines opt in
    with a ``cache_tag(profile_id)`` method."""
    for e in walk(engine):
        fn = getattr(e, "cache_tag", None)
        if callable(fn):
            try:
                tag = fn(profile_id)
            except Exception:
                tag = None
            if tag:
                return str(tag)
    return None


def apply_live(settings) -> None:
    """Push the in-app preferences (built-in voice, speed) into every loaded
    engine that takes them (``apply_settings``): no rebuild, no restart."""
    with _ENGINES_LOCK:
        engines = list(_ENGINES.values())
    for eng in engines:
        for e in walk(eng):
            fn = getattr(e, "apply_settings", None)
            if callable(fn):
                try:
                    fn(settings)
                except Exception:
                    log.exception("voice: applying settings to %s failed", type(e).__name__)


def builtin_voices(settings) -> list[dict]:
    """The installed provider's built-in voices ([] without one)."""
    from . import registry
    return registry.builtin_voices(settings)


def build_engine(settings):
    """The engine DISPATCH_VOICE_TTS names.

    * ``auto`` (default): ``chatterbox`` (registered by app.voice.dv when
      dispatch-voice is installed), else the first ready registered primary
      engine, else the first ready registered fallback engine on its own.
    * a registered name (``chatterbox``, ``kokoro`` from the optional plugin,
      anything an extension registered): that engine.
    * ``minimax``: built in (cloud, opt-in).

    A ready registered ``role="fallback"`` engine with the same sample rate
    goes BEHIND the chosen one. Wrapped with the HQ fallback when
    DISPATCH_VOICE_HQ_URL is set.
    """
    from . import profiles_bridge, registry
    profiles_bridge.ensure_engine()      # models downloaded since setup: no restart needed (review F6)
    name = settings.tts_engine
    if name == "auto":
        name = _auto_engine(settings, registry)
    spec = registry.engine_spec(name)
    if spec is not None:
        eng = spec.factory(settings)
        fb = registry.fallback_for(settings, name, getattr(eng, "sample_rate", None))
        if fb is not None:
            eng = _with_fallback(eng, fb.factory(settings))
    elif name == "minimax":
        eng = MiniMaxTTS(os.environ.get("MINIMAX_API_KEY", ""), settings.minimax_model,
                         settings.minimax_voice, settings.minimax_base_url)
    elif name is None:
        raise RuntimeError("no TTS engine is installed (see /api/voice/status)")
    else:
        raise RuntimeError(f"unknown TTS engine {name!r}")
    if settings.hq_url and getattr(settings, "tts_provider", "auto") != "cpu":
        # The reply text goes to this address, so the network policy is checked HERE too, not only
        # on PATCH: an address from the environment never passed through the route.
        from . import netpolicy
        verdict = netpolicy.check_url_policy(settings.hq_url, tuple(getattr(settings, "rig_allow", ()) or ()),
                                             schemes=("http", "https"))
        if not verdict.ok:
            log.warning("voice: the speech server address (hq_url) is refused by the network policy (%s); "
                        "this computer speaks instead", verdict.code)
        else:
            eng = FallbackTTS(HQTTS(settings.hq_url, settings.hq_model, settings.hq_timeout_s, eng.sample_rate,
                                    api_key=os.environ.get("DISPATCH_VOICE_HQ_API_KEY", "").strip()), eng)
    return eng


def _auto_engine(settings, registry) -> str | None:
    if registry.engine_spec("chatterbox") is not None:
        return "chatterbox"
    for spec in registry.primary_engines():
        if spec.ready(settings):
            return spec.name
    for spec in registry._engines.values():
        if spec.ready(settings):
            return spec.name
    return None


def _with_fallback(primary, fallback):
    if getattr(primary, "sample_rate", None) != getattr(fallback, "sample_rate", None):
        log.warning("voice: fallback engine runs at %s Hz, primary at %s Hz; not paired",
                    getattr(fallback, "sample_rate", None), getattr(primary, "sample_rate", None))
        return primary
    return FallbackTTS(primary, fallback, cooldown_s=30.0)


_ENGINES: dict[str, object] = {}
_ENGINES_LOCK = threading.Lock()


def shared_engine(settings):
    from . import profiles_bridge, registry
    profiles_bridge.ensure_engine()      # before the cache key reads the registered names (review F6)
    # Whether the policy accepts hq_url is part of the key: build_engine decides it once per build, so
    # an allow-list entry removed later must not leave the cached engine sending reply text there.
    hq_ok = None
    if settings.hq_url:
        from . import netpolicy
        hq_ok = netpolicy.check_url_policy(settings.hq_url, tuple(getattr(settings, "rig_allow", ()) or ()),
                                           schemes=("http", "https")).ok
    key = (f"{settings.tts_engine}:{settings.hq_url}:{hq_ok}:{getattr(settings, 'tts_provider', 'auto')}:"
           f"{registry.engine_names()}")
    with _ENGINES_LOCK:
        if key not in _ENGINES:
            _ENGINES[key] = build_engine(settings)
        return _ENGINES[key]


async def speak(text_chunks: AsyncIterator[str] | Iterable[str],
                profile_id: str | None = None, *, engine=None,
                stop: threading.Event | None = None) -> AsyncIterator[bytes]:
    """THE TTS seam: text chunks in → PCM frames out.

    Splits the incoming text on sentences as it arrives, drops what should
    not be read aloud (:func:`speakable`), and synthesises each sentence off
    the event loop in ``profile_id``'s voice, yielding raw frames in the
    engine's declared format (``engine.sample_rate``, ``sample_format``,
    ``channels``). ``engine`` defaults to the configured shared engine. The
    live session uses the same pieces with its own cancellable queue.
    """
    if engine is None:
        from .config import load_settings
        engine = await asyncio.to_thread(shared_engine, load_settings())
    splitter = SentenceSplitter()
    loop = asyncio.get_running_loop()

    async def _say(sentence: str):
        words = speakable(sentence.text if hasattr(sentence, "text") else sentence)
        if not words:
            return
        q: asyncio.Queue = asyncio.Queue()

        def produce():
            try:
                for c in engine.stream(words, profile_id, stop):
                    loop.call_soon_threadsafe(q.put_nowait, c)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        fut = loop.run_in_executor(None, produce)
        while (c := await q.get()) is not None:
            yield np.asarray(c, "<i2").tobytes()
        await fut

    async def _chunks():
        if hasattr(text_chunks, "__aiter__"):
            async for c in text_chunks:
                yield c
        else:
            for c in text_chunks:
                yield c

    async for chunk in _chunks():
        for s in splitter.push(chunk):
            async for frame in _say(s):
                yield frame
    for s in splitter.flush():
        async for frame in _say(s):
            yield frame
