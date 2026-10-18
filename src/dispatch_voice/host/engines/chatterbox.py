"""The ``chatterbox`` TTS engine: this package's Chatterbox-Turbo ONNX, in
``host/tts.py`` terms (the engine half of what used to be ``host/dv.py``).

* :func:`available` — the Chatterbox models are fetched (``dispatch-voice
  fetch-models``). Without them nothing here is registered and
  ``/api/voice/status.missing`` says what to run.
* :class:`LazyEngine` — ``ChatterboxOnnx`` loads on first use (~3 s, ~700 MB
  of sessions), never at import, so an install that never opens voice chat
  pays nothing.
* :class:`ChatterboxTTS` — streams 40 ms PCM16 frames at 24 kHz through T4's
  :class:`~dispatch_voice.rt.tts_stream.PipelinedChatterbox` (SPEC §4.5: the
  first chunk of a sentence is decoded while the language model is still
  writing the rest, so first audio arrives ≈ 1.3–1.5 s after the text at 2 GHz
  instead of after the whole sentence; ``rt.decoder_prompt_s`` /
  ``first_chunk_tokens`` / ``decoder_sessions`` apply live) with the barge-in
  ``stop`` event. Voice resolution: the bot's profile if it exists, else the
  store's default profile; with a fallback engine behind it (a built-in voice
  plugin) an id it does not know raises ``LookupError`` so the fallback speaks;
  as the last resort it speaks the STOCK voice instead of failing.
* :func:`release` — drops the loaded sessions when voice is switched off.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger("dispatch.voice")

SAMPLE_RATE = 24000
#: SPEC §4.10 thread plan: language model 4, decoder 6 (5 each with two sessions).
LM_THREADS = 4
DECODER_THREADS = {1: 6, 2: 5}


def _thread_plan() -> tuple[int, dict[int, int]]:
    """``DISPATCH_VOICE_TTS_THREADS`` (the 2.1 knob) scales the plan when set:
    it is read as the decoder budget, the LM keeps 4 unless the budget is smaller."""
    raw = os.environ.get("DISPATCH_VOICE_TTS_THREADS", "").strip()
    if not raw:
        return LM_THREADS, dict(DECODER_THREADS)
    try:
        n = max(1, int(raw))
    except ValueError:
        return LM_THREADS, dict(DECODER_THREADS)
    return min(LM_THREADS, n), {1: n, 2: max(1, n - 1)}


def _rt():
    try:
        from ..config import load_settings
        return load_settings().rt
    except Exception:  # noqa: BLE001 - outside a configured host (tests, the CLI) the defaults do
        from dispatch_voice.rt.settings import RtSettings
        return RtSettings()


def model_dir() -> Path:
    from dispatch_voice.engine import default_model_dir
    return Path(default_model_dir())


def complete(d: Path) -> bool:
    """Is a whole Chatterbox set in ``d``: the tokenizer and, for one precision, all four graphs
    with their weight files? The in-app download writes ``tokenizer.json`` first and creates
    ``onnx/`` at once, so "the folder exists" is true minutes before the last graph lands; this
    is true only when the engine could load."""
    from dispatch_voice.engine import COMPONENTS, DTYPES, _filename
    if not (d / "tokenizer.json").exists():
        return False
    onnx = d / "onnx"
    if not onnx.is_dir():
        return False
    for dtype in DTYPES:
        names = [_filename(c, dtype) for c in COMPONENTS]
        if all((onnx / n).exists() and (onnx / (n + "_data")).exists() for n in names):
            return True
    return False


def available() -> bool:
    """The Chatterbox models are on disk, all of them (``dispatch-voice fetch-models`` or the
    in-app download)."""
    try:
        return complete(model_dir())
    except Exception:
        return False


def missing() -> list[str]:
    """What ``/api/voice/status.missing`` lists for this engine ([] = ready)."""
    return [] if available() else ["dispatch-voice models (run `dispatch-voice fetch-models`)"]


class LazyEngine:
    """Stands in for ``ChatterboxOnnx`` until something actually needs it."""

    def __init__(self) -> None:
        self._eng = None
        self._lock = threading.Lock()

    def _get(self):
        if self._eng is None:
            with self._lock:
                if self._eng is None:
                    from dispatch_voice.engine import ChatterboxOnnx
                    rt = _rt()
                    lm, dec = _thread_plan()
                    sessions = max(1, min(2, int(getattr(rt, "decoder_sessions", 1) or 1)))
                    self._eng = ChatterboxOnnx(lm_threads=lm, decoder_threads=dec.get(sessions, 6),
                                               decoder_sessions=sessions)
                    log.info("voice: Chatterbox-Turbo ONNX engine loaded (lm %d threads, decoder %d × %d)",
                             lm, sessions, dec.get(sessions, 6))
        return self._eng

    def warm(self) -> None:
        """Load now (start-up prewarm) instead of on the first reply."""
        self._get()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._get(), name)


class ChatterboxTTS:
    """The ``chatterbox`` engine (``host.tts.TTSEngine`` shape).

    ``has_fallback`` is set by ``tts.build_engine`` when another engine (a
    built-in voice plugin) sits behind this one. Then a voice that is not a
    profile here (a ``builtin:`` id, an unknown id, no voice chosen) raises
    LookupError so the fallback speaks it. Without a fallback this engine is
    the last resort and speaks such a sentence in the default profile, else
    the stock voice, instead of failing."""

    sample_rate = SAMPLE_RATE
    sample_format = "pcm_s16le"
    channels = 1
    streaming = True

    def __init__(self, settings=None) -> None:
        self.has_fallback = False
        self._rt = getattr(settings, "rt", None) or _rt()
        self._pipe = None
        self._pipe_lock = threading.Lock()

    # ---- the pipelined engine (T4) ------------------------------------- #
    def _pipeline(self):
        """The :class:`PipelinedChatterbox` over the profile store's lazy engine,
        built on first use (never at import or setup)."""
        if self._pipe is not None:
            return self._pipe
        with self._pipe_lock:
            if self._pipe is None:
                from dispatch_voice.rt.tts_stream import PipelinedChatterbox

                from .. import profiles_bridge as pb
                st = pb.store()
                lm, dec = _thread_plan()
                sessions = max(1, min(2, int(self._rt.decoder_sessions)))
                self._pipe = PipelinedChatterbox(st, engine=st.engine, resolve=self._resolve, rt=self._rt,
                                                 lm_threads=lm, decoder_threads=dec.get(sessions),
                                                 model_dir=model_dir())
                self._pipe.has_fallback = self.has_fallback
        return self._pipe

    @property
    def last_stats(self) -> dict:
        """The last sentence's pipeline stats (first chunk / token timings)."""
        return dict(getattr(self._pipe, "last_stats", None) or {})

    def prepare(self, profile_id: str | None) -> str:
        """A voice's one-off work (sliced prompt + LM prefix, ≈ 0.5 s) ahead of
        its first sentence (Amendment T4-2). Raises LookupError for a voice
        this engine does not speak."""
        return self._pipeline().prepare(profile_id)

    def apply_settings(self, settings) -> None:
        """``rt.decoder_prompt_s`` / ``first_chunk_tokens`` / ``decoder_sessions``
        apply live (a session change is built on a worker, never blocking)."""
        rt = getattr(settings, "rt", None)
        if rt is None:
            return
        self._rt = rt
        if self._pipe is not None:
            self._pipe.apply_settings(settings)

    def sessions_settled(self, timeout: float | None = None) -> bool:
        return self._pipe is None or self._pipe.sessions_settled(timeout)

    def warm(self) -> None:
        """Load the engine and prepare the default voice (the start-up prewarm)."""
        self._pipeline().warm()

    def _resolve(self, profile_id: str | None) -> str:
        from dispatch_voice import speak

        from .. import profiles_bridge as pb
        from ..tts import builtin_id
        st = pb.store()
        sid = pb.stock_id()
        if builtin_id(profile_id):
            if self.has_fallback:
                # A built-in voice was chosen: not ours. LookupError = "use the
                # fallback for this sentence", with no cooldown (review L6).
                raise LookupError(f"{profile_id} is a built-in voice")
            profile_id = None          # no provider installed: degrade
        if profile_id and profile_id == sid:
            if pb.stock_profile(build=True) == sid:
                return sid
            raise LookupError("the stock voice is not available")
        # Resolve up front so "no voice chosen" surfaces as LookupError before
        # any audio, which is what lets FallbackTTS switch engines cleanly. An
        # unknown id is the same case, not an engine failure: a bad id must
        # not put the cloned engine on a 30 s cooldown for everyone.
        try:
            return speak.resolve(st, profile_id)
        except Exception as e:
            if self.has_fallback:
                if isinstance(e, LookupError):
                    raise
                raise LookupError(f"voice {profile_id!r} is not available: {e}") from e
        if profile_id:
            log.warning("voice: voice %r does not exist; using the default voice", profile_id)
        pid = st.default_id() or pb.stock_profile(build=True)
        if not pid:
            raise LookupError("no voice profile and no stock voice available")
        return pid

    def stream(self, text: str, profile_id: str | None = None,
               stop: threading.Event | None = None):
        """40 ms int16 frames at 24 kHz, the first ≈ 1.3–1.5 s after the call at
        2 GHz (pipelined: LM prefill with the cached voice prefix + the first
        25-token chunk). ``LookupError`` before any audio for a voice that is
        not ours (``FallbackTTS`` then speaks it)."""
        yield from self._pipeline().stream(text, profile_id, stop)

    def synth(self, text: str, profile_id: str | None = None):
        parts = list(self.stream(text, profile_id))
        return (np.concatenate(parts) if parts else np.zeros(0, "<i2")), SAMPLE_RATE

    #: A cloned voice renders the Chatterbox expressive tags ([laugh], [sigh], …).
    supports_tags = True

    def render_line(self, text: str, profile_id: str | None = None, *, seed: int) -> np.ndarray:
        """Whole-sentence render with a fixed seed (acknowledgements, notices):
        the LM's token sequence is reproducible per seed, so a line rendered
        once in this voice is reusable (the disk cache under ``cache_tag``)."""
        from .. import profiles_bridge as pb
        from ..tts import to_pcm16
        pid = self._resolve(profile_id)
        st = pb.store()
        wav = st.engine.synthesize(text, st.voice(pid), None, seed=seed)
        return to_pcm16(wav)

    def cache_tag(self, profile_id: str | None = None) -> str | None:
        """``cb-<model sha8>-<profile>-<created>`` (the same tag T4's pipelined
        engine uses, so the rendered lines carry over): a render of ``text`` in
        this voice is reusable. None for ids this engine does not speak. Never
        loads the engine or builds the stock voice (that is ``render_line``'s,
        on the render thread): the voice is resolved by name only — the id, else
        the store's default profile, else the stock id — and the model tag is a
        hash of the pinned revision and the dtype the lazy engine builds."""
        import hashlib

        from dispatch_voice.engine import MODEL_REVISION

        from .. import profiles_bridge as pb
        from ..tts import builtin_id
        if builtin_id(profile_id):
            if self.has_fallback:
                return None                      # a built-in voice: the plugin's tag, not ours
            profile_id = None
        try:
            st = pb.store()
            pid = profile_id or st.default_id() or pb.stock_id()
            if not pid:
                return None
        except Exception as e:  # noqa: BLE001 - a cache lookup never raises; None = not cacheable
            log.debug("voice: no cache tag for %r (%s)", profile_id, e)
            return None
        created = ""
        try:
            created = str(st.get(pid).get("created", "") or "")
        except Exception as e:  # noqa: BLE001 - a stock voice not built yet: the tag is the id alone
            log.debug("voice: no profile metadata for %s (%s)", pid, e)
        tag = hashlib.sha256(f"{MODEL_REVISION}:q4".encode()).hexdigest()[:8]
        return f"cb-{tag}-{pid}-{created}"


_INSTANCES: list = []
_INSTANCES_LOCK = threading.Lock()


def _make(settings) -> ChatterboxTTS:
    eng = ChatterboxTTS(settings)
    with _INSTANCES_LOCK:
        _INSTANCES.append(eng)
        del _INSTANCES[:-4]
    return eng


def register() -> None:
    """Make ``chatterbox`` selectable (``DISPATCH_VOICE_TTS=chatterbox`` / ``auto``)."""
    from .. import registry
    registry.register_engine("chatterbox", _make)


def release() -> None:
    """Drop the loaded Chatterbox sessions, the sliced voices and the LM prefix
    caches (voice switched off). The lazy engine loads them again on next use."""
    from .. import profiles_bridge as pb
    with _INSTANCES_LOCK:
        instances = list(_INSTANCES)
    for eng in instances:
        pipe = getattr(eng, "_pipe", None)
        if pipe is not None:
            try:
                pipe.release()
            except Exception:  # noqa: BLE001 - a release never fails the switch-off
                log.debug("voice: pipelined engine release failed", exc_info=True)
            eng._pipe = None
    pb.release_engine()
