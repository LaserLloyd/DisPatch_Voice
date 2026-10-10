"""``PipelinedChatterbox``: first audio from the first chunk, not the first sentence
(SPEC §4.5, C-§B.3).

Chatterbox-Turbo's language model is fast (≈ 0.3–0.45 s prefill then ≈ 12–13 ms per
speech token; 25 tokens = 1 s of audio) but the conditional decoder re-decodes the
whole reference prompt on every call: 1.43 s for a 25-token chunk with the 10 s
prompt the profile store keeps, 0.67 s with a 4 s slice (measured at 2 GHz). The
pipeline here is the one the design prototype measured:

* an LM thread streams speech tokens into a queue;
* the caller's thread (the session already runs ``stream()`` off-loop) submits a
  decode as soon as ``first_chunk_tokens`` (25; 15 with two sessions) are in, then
  every ``chunk_tokens`` (25) more, each with ``left_context`` (8) already-spoken
  tokens re-decoded and discarded; the decoder is stochastic, so every chunk is a
  fresh draw and the seam is a 20 ms equal-power crossfade over audio both chunks
  decoded for the same tokens;
* with ``decoder_sessions: 2`` chunks alternate sessions and decode concurrently
  (the first chunk can shrink to 15 tokens: ≈ 1.03 s to first audio measured);
  the default is 1 until the §4.10 contention run passes on that box;
* the output is 40 ms int16 frames at 24 kHz, like the whole-sentence engine, so
  the session's ``audio``/``say`` protocol does not change; ``stop`` is checked per
  token and per chunk.

It is the engine the host registers as ``chatterbox``; profile ids resolve through
the injected ``resolve`` (the host's rules for the stock voice and fallbacks), the
reference tensors come from the profile ``store`` and are sliced to
``rt.decoder_prompt_s`` once per profile.

``render_line`` is the whole-sentence path with the full prompt and a seed: the
token sequence is reproducible for a seed, the vocoder's noise is not (13
``RandomNormal`` ops; ``onnxruntime.set_seed`` does not pin them — measured), so
``cache_tag`` means "a render of this text in this voice is reusable", not bit-exact.

The LM's prefill used to re-read the voice's conditioning prefix (151 positions)
on every sentence: 0.45–0.55 s of the first chunk. The prefix's KV cache is now
computed once per voice (``engine.prefix_cache``, ≈ 0.45 s, 30 MB; ``prepare()`` /
``warm()`` do it ahead of the first sentence) and every sentence prefills only its
own text tokens: 55–100 ms (measured; same tokens for the same seed).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np

from ..audio import SAMPLE_RATE, crossfade
from ..engine import VoiceTensors
from .settings import RtSettings

log = logging.getLogger("dispatch.voice")

FRAME_MS = 40
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000      # 960 = one token's worth of audio
CROSSFADE_MS = 20
CROSSFADE_SAMPLES = SAMPLE_RATE * CROSSFADE_MS // 1000
CHUNK_TOKENS = 25
LEFT_CONTEXT = 8
#: Real tokens decoded past each chunk's end (audio dropped) so the chunk's last token
#: keeps its level: with only SILENCE following, the vocoder fades it 10–25 dB; three
#: lookahead tokens hold it within ±12 % (measured, end_effect2.json). Costs ≈ 40 ms of
#: LM wait + 36 ms of decode on the first chunk.
LOOKAHEAD_TOKENS = 3
TRAILING_SILENCE = 3
DEFAULT_LM_THREADS = 4
DEFAULT_DECODER_THREADS = {1: 6, 2: 5}


def to_int16(a: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(a, np.float32), -1.0, 1.0) * 32767).astype("<i2")


def frames_of(pcm: np.ndarray, n: int = FRAME_SAMPLES) -> Iterator[np.ndarray]:
    for i in range(0, len(pcm), n):
        yield pcm[i:i + n]


class _Plan:
    """Chunk schedule over a growing token list: which tokens to decode next and which of
    them to emit. A chunk EMITS [emit_start, emit_end) and DECODES [emit_start − left,
    emit_end + look): ``left`` already-emitted tokens as left context, ``look`` not yet
    emitted ones as right context (both dropped from the audio)."""

    def __init__(self, first: int, chunk: int, left: int, lookahead: int = LOOKAHEAD_TOKENS) -> None:
        self.first, self.chunk, self.left = max(1, first), max(1, chunk), max(0, left)
        self.lookahead = max(0, lookahead)
        self.decoded_upto = 0          # tokens emitted (or scheduled for emission) so far
        self.n_chunks = 0

    def next_chunk(self, n_tokens: int, done: bool) -> tuple[int, int, int, int] | None:
        """(start, end, left_context, lookahead) for the next decode — ``start``..``end`` are
        token indices into the token list, ``end`` includes the lookahead — or None when not
        enough tokens exist yet. A full chunk goes as soon as its tokens plus the lookahead
        exist; once the LM is done the remainder goes as the last chunk (no lookahead)."""
        size = self.first if self.n_chunks == 0 else self.chunk
        avail = n_tokens - self.decoded_upto
        if avail <= 0:
            return None
        if avail >= size + self.lookahead:
            emit_end, look = self.decoded_upto + size, self.lookahead
        elif done:
            emit_end, look = n_tokens, 0
        else:
            return None
        left = min(self.left, self.decoded_upto)
        start = self.decoded_upto - left
        self.decoded_upto = emit_end
        self.n_chunks += 1
        return start, emit_end + look, left, look


class PipelinedChatterbox:
    """The ``chatterbox`` engine (host ``TTSEngine`` shape) with pipelined decoding."""

    sample_rate = SAMPLE_RATE
    sample_format = "pcm_s16le"
    channels = 1
    streaming = True

    def __init__(self, store: Any, *, engine: Any = None, resolve: Callable[[str | None], str] | None = None,
                 rt: RtSettings | None = None, lm_threads: int = DEFAULT_LM_THREADS,
                 decoder_threads: int | None = None, model_dir=None) -> None:
        self.store = store
        self._engine_or_factory = engine
        self._engine = None
        self._engine_lock = threading.Lock()
        self._resolve = resolve
        self.rt = rt or RtSettings()
        self.lm_threads = lm_threads
        self._decoder_threads = decoder_threads
        self.model_dir = model_dir
        self.has_fallback = False
        self._voices: dict[tuple[str, float | None], VoiceTensors] = {}
        self._voices_lock = threading.Lock()
        self._prefixes: dict[str, Any] = {}          # pid → engine.PrefixKV (≈ 30 MB each)
        self._sessions_thread: threading.Thread | None = None
        self._sessions_lock = threading.Lock()
        self.last_stats: dict = {}

    # ----------------------------------------------------------------- engine
    @property
    def decoder_threads(self) -> int:
        return self._decoder_threads or DEFAULT_DECODER_THREADS.get(int(self.rt.decoder_sessions), 6)

    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        with self._engine_lock:
            if self._engine is None:
                src = self._engine_or_factory
                if src is None:
                    from ..engine import ChatterboxOnnx
                    eng = ChatterboxOnnx(self.model_dir, watermark=False, lm_threads=self.lm_threads,
                                         decoder_threads=self.decoder_threads,
                                         decoder_sessions=int(self.rt.decoder_sessions))
                elif callable(src) and not hasattr(src, "stream_tokens"):
                    eng = src()
                else:
                    eng = src
                self._apply_sessions(eng)
                self._engine = eng
                log.info("voice: pipelined Chatterbox ready (lm %d threads, decoder %d × %d threads, prompt %.1f s)",
                         self.lm_threads, int(self.rt.decoder_sessions), self.decoder_threads, self.rt.decoder_prompt_s)
        return self._engine

    def _apply_sessions(self, eng) -> None:
        fn = getattr(eng, "set_decoder_sessions", None)
        if fn is not None and getattr(eng, "decoder_sessions", 1) != int(self.rt.decoder_sessions):
            fn(int(self.rt.decoder_sessions), self.decoder_threads)

    def _apply_sessions_later(self, eng) -> None:
        """Run ``_apply_sessions`` on a worker thread (an ORT session takes ≈ 1 s to build;
        the caller is the settings route on the event loop). One worker at a time; it
        re-reads ``self.rt`` when it runs, so the last setting wins."""
        with self._sessions_lock:
            if self._sessions_thread is not None and self._sessions_thread.is_alive():
                return

            def run() -> None:
                try:
                    self._apply_sessions(eng)
                except Exception:
                    log.exception("voice: changing decoder sessions failed")

            self._sessions_thread = threading.Thread(target=run, name="voice-tts-sessions", daemon=True)
            self._sessions_thread.start()

    def sessions_settled(self, timeout: float | None = None) -> bool:
        """True once no decoder-session change is pending (tests, the lab line)."""
        th = self._sessions_thread
        if th is None:
            return True
        th.join(timeout)
        return not th.is_alive()

    @property
    def engine(self):
        return self._get_engine()

    def warm(self) -> None:
        """Load the engine (and the second decoder session when configured) now, and
        prepare the default voice (sliced prompt + LM prefix)."""
        eng = self._get_engine()
        try:
            self.prepare(None)
        except Exception as e:  # noqa: BLE001 - best effort: no default voice yet is fine
            log.debug("voice: no default voice to prepare (%s)", e)
        warm = getattr(eng, "warm", None)
        if callable(warm):
            warm()

    def prepare(self, profile_id: str | None) -> str:
        """Do a voice's one-off work now (the sliced prompt and the LM prefix, ≈ 0.5 s
        together) so its first sentence pays none of it. The session calls it when a
        thread's bot voice (and narrator voice) is known at ``start``. Returns the
        resolved profile id; raises LookupError for an unknown voice."""
        pid = self._resolve_id(profile_id)
        voice = self._voice_for(pid, self.rt.decoder_prompt_s)
        self._prefix_for(pid, voice)
        return pid

    def release(self) -> None:
        """Drop the engine, the sliced voices and the LM prefixes (voice switched off)."""
        with self._engine_lock:
            self._engine = None
        with self._voices_lock:
            self._voices.clear()
            self._prefixes.clear()

    def apply_settings(self, settings: Any) -> None:
        """``settings.rt`` (an RtSettings or mapping) or a bare RtSettings: picks up
        ``decoder_prompt_s`` / ``first_chunk_tokens`` / ``decoder_sessions`` live. Never
        blocks: a session-count change is built on a worker thread and swapped in
        (``engine.set_decoder_sessions`` builds before it swaps, so a sentence in flight
        finishes on the sessions it has; the next sentence uses the new count — review
        T4 #2); a prompt change drops the sliced-voice cache."""
        rt = getattr(settings, "rt", None)
        if rt is None and isinstance(settings, RtSettings):
            rt = settings
        if isinstance(rt, dict):
            rt = RtSettings.from_mapping({**self.rt.to_dict(), **rt})
        if not isinstance(rt, RtSettings):
            return
        old = self.rt
        self.rt = rt
        if old.decoder_prompt_s != rt.decoder_prompt_s:
            with self._voices_lock:
                self._voices.clear()
        if old.decoder_sessions != rt.decoder_sessions and self._engine is not None:
            self._apply_sessions_later(self._engine)

    # ----------------------------------------------------------------- voices
    def _resolve_id(self, profile_id: str | None) -> str:
        if self._resolve is not None:
            return self._resolve(profile_id)
        from ..speak import resolve
        return resolve(self.store, profile_id)

    def _voice_for(self, pid: str, prompt_s: float | None) -> VoiceTensors:
        key = (pid, prompt_s)
        with self._voices_lock:
            v = self._voices.get(key)
        if v is not None:
            return v
        full = self.store.voice(pid)
        eng = self._get_engine()
        slicer = getattr(eng, "slice_prompt", None)
        v = slicer(full, prompt_s) if (slicer is not None and prompt_s is not None) else full
        with self._voices_lock:
            if len(self._voices) > 16:
                self._voices.clear()
            self._voices[key] = v
        return v

    def _prefix_for(self, pid: str, voice: VoiceTensors):
        """This voice's LM prefix cache, computed once (engines without ``prefix_cache``
        — the fakes, older builds — get None and prefill the whole prefix as before)."""
        with self._voices_lock:
            p = self._prefixes.get(pid)
        if p is not None:
            return p
        eng = self._get_engine()
        fn = getattr(eng, "prefix_cache", None)
        if fn is None:
            return None
        p = fn(voice)
        with self._voices_lock:
            if len(self._prefixes) >= 6:                 # ≈ 30 MB each: a handful, not a library
                self._prefixes.pop(next(iter(self._prefixes)))
            self._prefixes[pid] = p
        return p

    def cache_tag(self, profile_id) -> str | None:
        """``cb-<model sha8>-<profile>-<created>``: a line rendered once in this voice can
        be reused (seeded tokens; the vocoder noise differs per render — see the module
        docstring). None for ids this engine does not speak."""
        try:
            pid = self._resolve_id(profile_id)
        except Exception:  # noqa: BLE001 - a cache lookup never raises; None = not cacheable
            return None
        created = ""
        try:
            meta = self.store.get(pid)
            created = str(meta.get("created", "") or "")
        except Exception as e:  # noqa: BLE001 - the tag degrades to the id alone
            log.debug("voice: no profile metadata for %s (%s)", pid, e)
        eng = self._get_engine()
        tag = eng.model_tag() if hasattr(eng, "model_tag") else "model"
        return f"cb-{tag}-{pid}-{created}"

    # ----------------------------------------------------------------- speaking
    def render_line(self, text: str, profile_id, *, seed: int) -> np.ndarray:
        """Whole-sentence render with the FULL prompt (acks, notices): int16 24 kHz."""
        pid = self._resolve_id(profile_id)
        voice = self.store.voice(pid)
        eng = self._get_engine()
        wav = eng.synthesize(text, voice, None, seed=seed)
        return to_int16(wav)

    def synth(self, text: str, profile_id: str | None = None) -> tuple[np.ndarray, int]:
        parts = list(self.stream(text, profile_id))
        return (np.concatenate(parts) if parts else np.zeros(0, "<i2")), SAMPLE_RATE

    def stream(self, text: str, profile_id: str | None = None,
               stop: threading.Event | None = None) -> Iterator[np.ndarray]:
        """Pipelined synthesis: 40 ms int16 frames, the first ≈ 1.4 s after the call at
        2 GHz with one decoder session (LM prefill + 25 tokens + one 4 s-prompt decode)."""
        pid = self._resolve_id(profile_id)           # LookupError before any audio (FallbackTTS)
        if stop is not None and stop.is_set():
            return
        eng = self._get_engine()
        if not callable(getattr(eng, "stream_tokens", None)):
            # An engine without the LM/decoder split (an older engine build, a stand-in):
            # the whole-sentence path the 2.1 engine took, same frames, same resolution.
            from ..speak import speak
            for frame in speak(self.store, [text], profile_id=pid, stop=stop):
                yield np.frombuffer(frame, dtype="<i2")
            return
        rt = self.rt
        voice = self._voice_for(pid, rt.decoder_prompt_s)
        prefix = self._prefix_for(pid, voice)
        sessions = max(1, int(getattr(eng, "decoder_sessions", 1) or 1))
        yield from self._pipeline(eng, text, voice, stop, sessions=sessions,
                                  first=int(rt.first_chunk_tokens), chunk=CHUNK_TOKENS,
                                  left=LEFT_CONTEXT, prompt_s=rt.decoder_prompt_s, prefix=prefix)

    # ----------------------------------------------------------------- the pipeline
    def _pipeline(self, eng, text: str, voice: VoiceTensors, stop: threading.Event | None, *,
                  sessions: int, first: int, chunk: int, left: int, prompt_s: float | None,
                  prefix: Any = None) -> Iterator[np.ndarray]:
        t_start = time.perf_counter()
        events: queue.Queue[tuple] = queue.Queue()
        tokens: list[int] = []
        lm_error: list[BaseException] = []
        stats: dict = {"chunks": [], "seams": [], "seam_levels": [], "first_chunk_ms": None,
                       "first_token_ms": None, "tokens_ready_ms": None, "prefix": prefix is not None}
        ready_at = first + LOOKAHEAD_TOKENS

        def lm_thread() -> None:
            try:
                gen = (eng.stream_tokens(text, voice, stop, None, prefix=prefix) if prefix is not None
                       else eng.stream_tokens(text, voice, stop, None))
                for n, t in enumerate(gen, start=1):
                    if n == 1:
                        stats["first_token_ms"] = round((time.perf_counter() - t_start) * 1000)
                    elif n == ready_at:
                        stats["tokens_ready_ms"] = round((time.perf_counter() - t_start) * 1000)
                    events.put(("tok", t))
                    if stop is not None and stop.is_set():
                        break
            except Exception as e:  # noqa: BLE001 - surfaced to the consumer, whatever it was
                lm_error.append(e)
            finally:
                if stats["tokens_ready_ms"] is None:     # shorter than a first chunk: ready when the LM is done
                    stats["tokens_ready_ms"] = round((time.perf_counter() - t_start) * 1000)
                events.put(("done", None))

        th = threading.Thread(target=lm_thread, name="voice-tts-lm", daemon=True)
        th.start()
        plan = _Plan(first, chunk, left)
        pool = ThreadPoolExecutor(max_workers=sessions, thread_name_prefix="voice-tts-dec")
        pending: list[tuple[int, Future, float]] = []     # (chunk index, future, submitted at)
        next_session = 0
        held: np.ndarray | None = None                     # last 20 ms of the previous chunk
        emitted = 0
        done = False
        carry = np.zeros(0, "<i2")                         # < 1 frame left over between chunks

        def framed(pcm16: np.ndarray, *, flush: bool = False):
            """Whole 40 ms frames only (the protocol sends one per message); the remainder
            waits for the next chunk, or goes out as the final short frame on flush."""
            nonlocal carry
            buf = np.concatenate([carry, pcm16]) if len(carry) else pcm16
            n_full = len(buf) // FRAME_SAMPLES * FRAME_SAMPLES
            yield from frames_of(buf[:n_full])
            carry = buf[n_full:]
            if flush and len(carry):
                rest, carry = carry, np.zeros(0, "<i2")
                yield rest
        try:
            while True:
                if stop is not None and stop.is_set():
                    break
                # 1. drain token events (block only when nothing is pending)
                try:
                    kind, item = events.get(timeout=0.005 if pending else 0.05)
                    if kind == "tok":
                        tokens.append(item)
                    else:
                        done = True
                    while True:
                        try:
                            kind, item = events.get_nowait()
                        except queue.Empty:
                            break
                        if kind == "tok":
                            tokens.append(item)
                        else:
                            done = True
                except queue.Empty:
                    pass
                if lm_error:
                    raise lm_error[0]
                # 2. submit every chunk the plan allows (one at a time per free session)
                while len(pending) < sessions:
                    nxt = plan.next_chunk(len(tokens), done)
                    if nxt is None:
                        break
                    s, e, lc, look = nxt
                    last = done and e >= len(tokens) and look == 0
                    toks = tokens[s:e]
                    fut = pool.submit(eng.decode_chunk, toks, voice, left_context=lc, prompt_s=prompt_s,
                                      session=next_session % sessions, trailing_silence=TRAILING_SILENCE,
                                      keep_silence=last, context_tail=CROSSFADE_SAMPLES if lc else 0,
                                      right_context=look)
                    next_session += 1
                    pending.append((plan.n_chunks - 1, fut, time.perf_counter()))
                # 3. emit finished chunks in order
                while pending and pending[0][1].done():
                    idx, fut, t_sub = pending.pop(0)
                    wav = fut.result()
                    t_now = time.perf_counter()
                    stats["chunks"].append({"i": idx, "decode_ms": round((t_now - t_sub) * 1000),
                                            "at_ms": round((t_now - t_start) * 1000), "samples": len(wav)})
                    if stop is not None and stop.is_set():
                        break
                    if len(wav) == 0:
                        continue
                    if held is not None:
                        # ``wav`` starts with the re-decoded tail of the previous chunk's last
                        # tokens (context_tail samples): the same 20 ms ``held`` covers.
                        n = min(len(held), len(wav))
                        stats["seams"].append(emitted)
                        stats["seam_levels"].append((float(np.sqrt(np.mean(held[-n:] ** 2))),
                                                     float(np.sqrt(np.mean(wav[:n] ** 2)))))
                        out = np.concatenate([crossfade(held[-n:], wav[:n]), wav[n:]])
                    else:
                        out = wav
                    # keep the last 20 ms back for the next seam (unless this was the last chunk)
                    if not (done and plan.decoded_upto >= len(tokens) and not pending):
                        held, out = out[-CROSSFADE_SAMPLES:], out[:-CROSSFADE_SAMPLES]
                    else:
                        held = None
                    if stats["first_chunk_ms"] is None and len(out):
                        stats["first_chunk_ms"] = round((time.perf_counter() - t_start) * 1000)
                    emitted += len(out)
                    yield from framed(to_int16(out), flush=held is None)
                if done and plan.decoded_upto >= len(tokens) and not pending:
                    break
            if held is not None and not (stop is not None and stop.is_set()):
                emitted += len(held)
                yield from framed(to_int16(held), flush=True)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            th.join(timeout=5.0)
            stats["total_ms"] = round((time.perf_counter() - t_start) * 1000)
            stats["tokens"] = len(tokens)
            stats["audio_ms"] = round(emitted / SAMPLE_RATE * 1000)
            self.last_stats = stats
