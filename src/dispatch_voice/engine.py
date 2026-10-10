"""Chatterbox-Turbo (Resemble AI, MIT) on CPU through ONNX Runtime.

A voice profile is the four tensors the speech encoder produces from a reference
clip: build it once (under a second), save it, and every reply after that skips
the encoder entirely. Those tensors are the "generation parameters".

Measured on a Ryzen AI 9 365 laptop CPU (10 threads, q4): profile build ~0.7 s,
language model 14-19 ms per speech token (25 tokens = 1 s of audio), decoder
~1 s fixed + ~0.25 s per second of audio. Real-time factor ~0.7 on a long
sentence, about 1.7 s to the first audio for a short one.
"""
from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE

MODEL_REPO = "ResembleAI/chatterbox-turbo-ONNX"
# Pinned so a re-download can't silently change the voice.
MODEL_REVISION = os.environ.get("DISPATCH_VOICE_MODEL_REVISION", "d21799bd0354adb85e348b8a0442a8405110a2cf")
START_SPEECH, STOP_SPEECH, SILENCE = 6561, 6562, 4299
NUM_KV_HEADS, HEAD_DIM = 16, 64
TOKENS_PER_SECOND = 25
#: The conditional decoder emits exactly this many 24 kHz samples per speech token
#: (measured: 25 tokens + 3 silence tokens → 26 880 samples), which is what makes
#: chunk slicing and the seam crossfade pure arithmetic.
SAMPLES_PER_TOKEN = SAMPLE_RATE // TOKENS_PER_SECOND     # 960
COMPONENTS = ("speech_encoder", "embed_tokens", "language_model", "conditional_decoder")
DTYPES = ("fp32", "fp16", "q8", "q4", "q4f16")
TAGS = ("[laugh]", "[chuckle]", "[sigh]", "[gasp]", "[cough]", "[clear throat]", "[sniff]", "[groan]", "[shush]")


def default_model_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(os.environ.get("DISPATCH_VOICE_MODELS", os.path.join(base, "dispatch-voice", "models")))


def _filename(component: str, dtype: str) -> str:
    suffix = "" if dtype == "fp32" else "_quantized" if dtype == "q8" else f"_{dtype}"
    return f"{component}{suffix}.onnx"


def fetch_models(model_dir: Path | None = None, dtype: str = "q4", revision: str = MODEL_REVISION) -> Path:
    """Download the ONNX graphs + weights into a plain directory.

    A plain directory, not the HF cache: onnxruntime refuses external-data files
    reached through the cache's symlinks.
    """
    from huggingface_hub import hf_hub_download
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}")
    d = Path(model_dir or default_model_dir())
    d.mkdir(parents=True, exist_ok=True)
    for f in ("tokenizer.json", "config.json"):
        hf_hub_download(MODEL_REPO, f, revision=revision, local_dir=d)
    for c in COMPONENTS:
        fn = _filename(c, dtype)
        hf_hub_download(MODEL_REPO, fn, subfolder="onnx", revision=revision, local_dir=d)
        hf_hub_download(MODEL_REPO, fn + "_data", subfolder="onnx", revision=revision, local_dir=d)
    return d


@dataclass
class PrefixKV:
    """The language model's KV cache after reading a voice's conditioning prefix
    (``cond_emb``, 151 positions for the stock voice), computed once per voice by
    :meth:`ChatterboxOnnx.prefix_cache` and reused by every sentence (``stream_tokens(prefix=)``),
    so a sentence prefills only its own text tokens. Measured: 531 → 98 ms for 24 text
    tokens, 486 → 55 ms for 9; the last-position logits and the text rows of the cache
    are bit-identical to the full prefill, and 40 sampled tokens matched token for token
    under the same seed (``probe_prefix_kv.json``). ≈ 30 MB per voice (48 × fp32)."""
    length: int
    kv: dict[str, np.ndarray]

    @property
    def nbytes(self) -> int:
        return sum(int(v.nbytes) for v in self.kv.values())


@dataclass
class VoiceTensors:
    cond_emb: np.ndarray
    prompt_token: np.ndarray
    speaker_embeddings: np.ndarray
    speaker_features: np.ndarray

    def save(self, path: Path) -> None:
        tmp = Path(str(path) + ".tmp.npz")
        np.savez(tmp, cond_emb=self.cond_emb, prompt_token=self.prompt_token,
                 speaker_embeddings=self.speaker_embeddings, speaker_features=self.speaker_features)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> VoiceTensors:
        with np.load(path, allow_pickle=False) as z:
            return cls(z["cond_emb"], z["prompt_token"], z["speaker_embeddings"], z["speaker_features"])


def normalise_text(text: str) -> str:
    """Same spirit as Chatterbox's punc_norm: tidy spacing, capitalise, end on punctuation."""
    t = text or ""
    for a, b in (("…", "..."), ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"), ("—", ", "), ("–", ", ")):
        t = t.replace(a, b)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    t = re.sub(r",{2,}", ",", t).strip(" ,")
    if not t:
        return t
    t = t[0].upper() + t[1:]
    if t[-1] not in ".!?,;:\"'" and not t.endswith("]"):
        t += "."
    return t


_SENTENCE_END = re.compile(r"([.!?]+[\"')\]]?)(\s+|$)")


def split_speakable(buffer: str, min_words: int = 6, final: bool = False,
                    first: bool = True) -> tuple[list[str], str]:
    """Pull complete sentences off the front of a streaming text buffer.

    Short sentences are merged with the next one until they reach min_words, so
    the voice isn't fed choppy fragments ("Sure." "Okay."). The first sentence of
    a reply (``first=True``) is exempt: speaking it early is the point.
    Returns (ready_chunks, remainder).
    """
    out: list[str] = []
    pending = ""
    pos = 0
    for m in _SENTENCE_END.finditer(buffer):
        sent = buffer[pos:m.end(1)].strip()
        pos = m.end()
        if not sent:
            continue
        pending = f"{pending} {sent}".strip()
        if len(pending.split()) >= min_words or (first and not out):
            out.append(pending)
            pending = ""
    # Keep trailing whitespace: the next streamed piece may start mid-word.
    rest = (pending + " " if pending else "") + buffer[pos:].lstrip()
    if final and rest.strip():
        out.append(rest.strip())
        rest = ""
    return out, rest


class ChatterboxOnnx:
    """Thread-safe (one synthesis at a time) CPU engine."""

    sample_rate = SAMPLE_RATE

    def __init__(self, model_dir: Path | None = None, dtype: str = "q4", threads: int | None = None,
                 watermark: bool | None = None, repetition_penalty: float = 1.2,
                 temperature: float = 0.8, top_k: int = 1000, top_p: float = 0.95,
                 lm_threads: int | None = None, decoder_threads: int | None = None,
                 decoder_sessions: int = 1):
        """``threads`` is the budget for the encoder/embedding sessions (and the default
        for the other two); ``lm_threads`` / ``decoder_threads`` set the language model
        and the conditional decoder separately (SPEC §4.10: LM 4, decoder 6 — or 5 each
        with ``decoder_sessions=2``). Decoder sessions share the weight file; a second
        one costs ≈ 93 MB (measured) and lets two chunks decode at once."""
        from tokenizers import Tokenizer
        self.model_dir = Path(model_dir or default_model_dir())
        self.dtype = dtype
        missing = [c for c in COMPONENTS if not (self.model_dir / "onnx" / _filename(c, dtype)).exists()]
        if missing or not (self.model_dir / "tokenizer.json").exists():
            raise FileNotFoundError(f"models missing in {self.model_dir} ({', '.join(missing) or 'tokenizer'}); "
                                    "run `dispatch-voice fetch-models`")
        base = threads or max(1, min(10, (os.cpu_count() or 2) // 2))
        self.threads = base
        self.lm_threads = lm_threads or base
        self.decoder_threads = decoder_threads or base
        self._s = {}
        for c in ("speech_encoder", "embed_tokens"):
            self._s[c] = self._session(c, base)
        self._s["language_model"] = self._session("language_model", self.lm_threads)
        self._dec: list = []
        self._dec_locks: list[threading.Lock] = []
        self._dec_swap = threading.Lock()    # the decoder-session list (read per chunk, swapped rarely)
        self._dec_build = threading.Lock()   # one set_decoder_sessions() at a time
        self.set_decoder_sessions(max(1, int(decoder_sessions)))   # needs both locks above
        self._kv_inputs = [i for i in self._s["language_model"].get_inputs() if "past_key_values" in i.name]
        self._tok = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self._lock = threading.Lock()        # whole-sentence synthesis, one at a time
        self._lm_lock = threading.Lock()     # the language model (pipelined path)
        self._enc_lock = threading.Lock()    # the speech encoder
        # Resemble's reference sampling settings for Turbo. Greedy decoding (the ONNX
        # example) loops and can replay the reference clip's words instead of the text.
        self.repetition_penalty = repetition_penalty
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self._wm = None
        if watermark is not False:
            try:
                import perth  # optional: needs CPU torch
                cls = perth.PerthImplicitWatermarker   # None when its own imports failed
                self._wm = cls() if cls is not None else None
            except Exception:
                self._wm = None
            if watermark and self._wm is None:
                raise RuntimeError("watermark requested but resemble-perth is unavailable "
                                   "(install the `watermark` extra)")
        self.last_stats: dict = {}

    @property
    def watermarking(self) -> bool:
        return self._wm is not None

    # ------------------------------------------------------------ sessions #
    def _session(self, component: str, threads: int):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(threads))
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        return ort.InferenceSession(str(self.model_dir / "onnx" / _filename(component, self.dtype)), so,
                                    providers=["CPUExecutionProvider"])

    @property
    def decoder_sessions(self) -> int:
        return len(self._dec)

    def set_decoder_sessions(self, n: int, threads: int | None = None) -> None:
        """Keep ``n`` conditional-decoder sessions (1 or 2). Creating the second one
        takes ≈ 1 s and ≈ 93 MB; dropping it frees that. ``self._s["conditional_decoder"]``
        stays the first session for whole-sentence synthesis.

        Safe while chunks decode (review T4 #2): the new sessions are built first, then
        the list is swapped under ``_dec_swap``; a decode in flight keeps the session it
        picked and finishes on it. Session 0 is kept unless ``threads`` differs from the
        count it was built with (the §4.10 plan has 6 threads for one session, 5 each for
        two) — then every session is rebuilt, still before the swap. Calls serialise."""
        n = max(1, min(2, int(n)))
        with self._dec_build:
            want_threads = max(1, int(threads)) if threads is not None else self.decoder_threads
            with self._dec_swap:
                keep = list(self._dec) if want_threads == self.decoder_threads else []
                keep_locks = list(self._dec_locks)[:len(keep)]
            new = keep[:n]
            new_locks = keep_locks[:n]
            while len(new) < n:
                new.append(self._session("conditional_decoder", want_threads))
                new_locks.append(threading.Lock())
            with self._dec_swap:
                self.decoder_threads = want_threads
                self._dec, self._dec_locks = new, new_locks
                self._s["conditional_decoder"] = self._dec[0]

    def _decoder(self, session: int):
        """(session, its lock) for a chunk — taken atomically, so a swap in between leaves
        this decode on a session that still exists."""
        with self._dec_swap:
            i = int(session) % max(1, len(self._dec))
            return self._dec[i], self._dec_locks[i]

    def model_tag(self) -> str:
        """Eight hex chars naming the model files (revision + dtype): part of cache tags."""
        import hashlib
        return hashlib.sha256(f"{MODEL_REVISION}:{self.dtype}".encode()).hexdigest()[:8]

    def build_voice(self, reference: np.ndarray) -> VoiceTensors:
        with self._enc_lock:
            out = self._s["speech_encoder"].run(None, {"audio_values": reference[None].astype(np.float32)})
        return VoiceTensors(*out)

    # ------------------------------------------------------------ pipelined path #
    def _empty_kv(self) -> dict[str, np.ndarray]:
        return {i.name: np.zeros([1, NUM_KV_HEADS, 0, HEAD_DIM],
                                 np.float16 if i.type == "tensor(float16)" else np.float32)
                for i in self._kv_inputs}

    def prefix_cache(self, voice: VoiceTensors) -> PrefixKV:
        """Read the voice's conditioning prefix through the language model once and keep
        the KV cache it leaves (:class:`PrefixKV`): ≈ 0.45 s at 2 GHz, 30 MB, per voice.
        The session's engine caches one per profile; ``stream_tokens(prefix=…)`` then
        prefills only the sentence's text tokens. ``cond_emb`` is untouched by
        :meth:`slice_prompt`, so a sliced and a full voice share the same prefix."""
        n = int(voice.cond_emb.shape[1])
        with self._lm_lock:
            _logits, *present = self._s["language_model"].run(
                None, dict(inputs_embeds=voice.cond_emb, attention_mask=np.ones((1, n), np.int64),
                           position_ids=np.arange(n, dtype=np.int64)[None], **self._empty_kv()))
        return PrefixKV(length=n, kv={i.name: present[j] for j, i in enumerate(self._kv_inputs)})

    def stream_tokens(self, text: str, voice: VoiceTensors, stop: threading.Event | None = None,
                      seed: int | None = None, *, prefix: PrefixKV | None = None) -> Iterator[int]:
        """The language-model loop as a generator of SPEECH tokens (< START_SPEECH),
        in order, as they are sampled (≈ 0.3–0.45 s prefill, then ≈ 12–13 ms per
        token at 2 GHz; 25 tokens = 1 s of audio). ``stop`` is checked per token.
        Holds the LM lock for the whole generation, so two sentences never share it.
        With ``prefix`` (this voice's :class:`PrefixKV`) the prefill covers only the
        text tokens — the same tokens come out for the same seed (measured)."""
        text = normalise_text(text)
        if not text:
            return
        ids = np.array([self._tok.encode(text).ids], dtype=np.int64)
        words = len(text.split()) + 3 * text.count("[")
        max_new = min(1500, 25 + 20 * words)
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        n_out = 0
        t_prefill = None
        with self._lm_lock:
            text_emb = self._s["embed_tokens"].run(None, {"input_ids": ids})[0]
            if prefix is not None and prefix.length == int(voice.cond_emb.shape[1]):
                emb = text_emb
                pkv = dict(prefix.kv)                 # inputs are never written by ORT: shareable
                start = prefix.length
            else:
                emb = np.concatenate([voice.cond_emb, text_emb], axis=1)
                pkv = self._empty_kv()
                start = 0
            mask = np.ones((1, start + emb.shape[1]), np.int64)
            pos = np.arange(start, start + emb.shape[1], dtype=np.int64)[None]
            seen = np.zeros(STOP_SPEECH + 1, bool); seen[START_SPEECH] = True
            for _ in range(max_new):
                if stop is not None and stop.is_set():
                    return
                logits, *present = self._s["language_model"].run(
                    None, dict(inputs_embeds=emb, attention_mask=mask, position_ids=pos, **pkv))
                if t_prefill is None:
                    t_prefill = time.perf_counter() - t0
                row = logits[0, -1, :].astype(np.float32)
                idx = np.nonzero(seen[: row.shape[0]])[0]
                s = row[idx]
                row[idx] = np.where(s < 0, s * self.repetition_penalty, s / self.repetition_penalty)
                nxt = self._sample(row, rng)
                if nxt == STOP_SPEECH:
                    break
                if nxt < seen.shape[0]:
                    seen[nxt] = True
                if nxt < START_SPEECH:
                    n_out += 1
                    yield nxt
                emb = self._s["embed_tokens"].run(None, {"input_ids": np.array([[nxt]], np.int64)})[0]
                mask = np.concatenate([mask, np.ones((1, 1), np.int64)], axis=1)
                pos = pos[:, -1:] + 1
                for j, k in enumerate(pkv):
                    pkv[k] = present[j]
        self.last_stats = dict(self.last_stats, lm_s=round(time.perf_counter() - t0, 3), tokens=n_out,
                               prefill_s=round(t_prefill or 0.0, 3), prefix=prefix is not None)

    @staticmethod
    def slice_prompt(voice: VoiceTensors, prompt_s: float | None) -> VoiceTensors:
        """The decoder re-decodes the whole reference prompt on every call (10 s =
        250 tokens: 1.43 s per 25-token chunk at 2 GHz; 4 s: 0.67 s). Keep the first
        ``prompt_s`` seconds of ``prompt_token`` (25/s) and ``speaker_features`` (50/s);
        ``cond_emb`` (what the LM sees) is untouched. None/≥ full = unchanged."""
        if prompt_s is None:
            return voice
        n_tok = round(float(prompt_s) * TOKENS_PER_SECOND)
        if n_tok <= 0 or n_tok >= voice.prompt_token.shape[1]:
            return voice
        sf = voice.speaker_features
        t_axis = int(np.argmax(sf.shape))          # (1, frames, 80): the time axis is the long one
        sl = [slice(None)] * sf.ndim
        sl[t_axis] = slice(0, n_tok * 2)
        return VoiceTensors(voice.cond_emb, voice.prompt_token[:, :n_tok], voice.speaker_embeddings,
                            np.ascontiguousarray(sf[tuple(sl)]))

    def decode_chunk(self, tokens: Sequence[int], voice: VoiceTensors, *, left_context: int = 8,
                     prompt_s: float | None = 4.0, session: int = 0, trailing_silence: int = 3,
                     keep_silence: bool = False, context_tail: int = 0, right_context: int = 0) -> np.ndarray:
        """Decode ``tokens`` (speech tokens) to float32 24 kHz.

        The first ``left_context`` tokens are context already spoken by the previous chunk:
        re-decoded, audio dropped — except the last ``context_tail`` samples, which the
        pipeline crossfades with the previous chunk's end (both sides then cover the same
        tokens). The last ``right_context`` tokens are lookahead the NEXT chunk will emit:
        re-decoded here so this chunk's own last tokens keep their level (the vocoder fades
        the final 1–2 tokens by 10–25 dB when only silence follows; three real tokens of
        lookahead hold them within ±12 % — measured), audio dropped. ``trailing_silence``
        SILENCE tokens close the sequence; their audio is dropped unless ``keep_silence``
        (the last chunk). The output is exactly 960 samples per kept token (measured).
        ``voice`` may already be sliced; ``prompt_s`` slices it otherwise. ``session`` picks
        the decoder session."""
        toks = [int(t) for t in tokens]
        if not toks:
            return np.zeros(0, np.float32)
        left = max(0, min(int(left_context), len(toks)))
        right = max(0, min(int(right_context), len(toks) - left))
        v = self.slice_prompt(voice, prompt_s)
        seq = np.concatenate([v.prompt_token, np.array([toks], np.int64),
                              np.full((1, max(0, int(trailing_silence))), SILENCE, np.int64)], axis=1)
        sess, lock = self._decoder(session)
        with lock:
            wav = sess.run(None, dict(speech_tokens=seq, speaker_embeddings=v.speaker_embeddings,
                                      speaker_features=v.speaker_features))[0].squeeze(0)
        wav = np.asarray(wav, np.float32)
        per = SAMPLES_PER_TOKEN
        start = max(0, left * per - max(0, int(context_tail)) if left else 0)
        if keep_silence and not right:
            end = len(wav)
        else:
            end = max(start, len(wav) - (int(trailing_silence) + right) * per)
        return wav[start:end]

    def speaker_embedding(self, audio: np.ndarray) -> np.ndarray:
        return self.build_voice(audio).speaker_embeddings.reshape(-1)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity of the model's own speaker embeddings (1.0 = same voice)."""
        ea, eb = self.speaker_embedding(a), self.speaker_embedding(b)
        return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9))

    def _sample(self, row: np.ndarray, rng: np.random.Generator) -> int:
        if self.temperature <= 0:
            return int(np.argmax(row))
        row = row / self.temperature
        k = min(self.top_k, row.shape[0])
        idx = np.argpartition(row, -k)[-k:]
        logits = row[idx]
        order = np.argsort(logits)[::-1]
        idx, logits = idx[order], logits[order]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        keep = np.searchsorted(np.cumsum(p), self.top_p) + 1
        idx, p = idx[:keep], p[:keep] / p[:keep].sum()
        return int(rng.choice(idx, p=p))

    def synthesize(self, text: str, voice: VoiceTensors, stop: threading.Event | None = None,
                   seed: int | None = None) -> np.ndarray:
        text = normalise_text(text)
        if not text:
            return np.zeros(0, np.float32)
        # Speech runs ~10 tokens (0.4 s) per word; stream_tokens caps the run so a bad
        # sample can't run on. Whole-sentence synthesis keeps the FULL reference prompt
        # (quality over latency: acks and notices are rendered in the background).
        with self._lock:
            t0 = time.perf_counter()
            speech = list(self.stream_tokens(text, voice, stop, seed))
            if stop is not None and stop.is_set():
                return np.zeros(0, np.float32)
            t_lm = time.perf_counter() - t0
            t1 = time.perf_counter()
            wav = self.decode_chunk(speech, voice, left_context=0, prompt_s=None, session=0,
                                    trailing_silence=3, keep_silence=True)
            t_dec = time.perf_counter() - t1
        if self._wm is not None:
            wav = np.asarray(self._wm.apply_watermark(wav, sample_rate=self.sample_rate), dtype=np.float32)
        dur = len(wav) / self.sample_rate
        self.last_stats = dict(lm_s=round(t_lm, 3), decode_s=round(t_dec, 3), tokens=len(speech),
                               audio_s=round(dur, 3), rtf=round((t_lm + t_dec) / dur, 3) if dur else None)
        return wav

    def stream(self, text_chunks: Iterable[str], voice: VoiceTensors,
               stop: threading.Event | None = None) -> Iterator[np.ndarray]:
        """Turn an LLM token stream into audio, one speakable chunk at a time."""
        buf, first = "", True
        for piece in text_chunks:
            if stop is not None and stop.is_set():
                return
            buf += piece
            ready, buf = split_speakable(buf, first=first)
            for chunk in ready:
                first = False
                yield self.synthesize(chunk, voice, stop)
        ready, _ = split_speakable(buf, final=True, first=first)
        for chunk in ready:
            if stop is not None and stop.is_set():
                return
            yield self.synthesize(chunk, voice, stop)
