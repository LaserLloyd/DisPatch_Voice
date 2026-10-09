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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .audio import SAMPLE_RATE

MODEL_REPO = "ResembleAI/chatterbox-turbo-ONNX"
# Pinned so a re-download can't silently change the voice.
MODEL_REVISION = os.environ.get("DISPATCH_VOICE_MODEL_REVISION", "d21799bd0354adb85e348b8a0442a8405110a2cf")
START_SPEECH, STOP_SPEECH, SILENCE = 6561, 6562, 4299
NUM_KV_HEADS, HEAD_DIM = 16, 64
TOKENS_PER_SECOND = 25
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
    def load(cls, path: Path) -> "VoiceTensors":
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
                 temperature: float = 0.8, top_k: int = 1000, top_p: float = 0.95):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.model_dir = Path(model_dir or default_model_dir())
        self.dtype = dtype
        missing = [c for c in COMPONENTS if not (self.model_dir / "onnx" / _filename(c, dtype)).exists()]
        if missing or not (self.model_dir / "tokenizer.json").exists():
            raise FileNotFoundError(f"models missing in {self.model_dir} ({', '.join(missing) or 'tokenizer'}); "
                                    "run `dispatch-voice fetch-models`")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or max(1, min(10, (os.cpu_count() or 2) // 2))
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._s = {c: ort.InferenceSession(str(self.model_dir / "onnx" / _filename(c, dtype)), so,
                                           providers=["CPUExecutionProvider"]) for c in COMPONENTS}
        self._kv_inputs = [i for i in self._s["language_model"].get_inputs() if "past_key_values" in i.name]
        self._tok = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self._lock = threading.Lock()
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

    def build_voice(self, reference: np.ndarray) -> VoiceTensors:
        with self._lock:
            out = self._s["speech_encoder"].run(None, {"audio_values": reference[None].astype(np.float32)})
        return VoiceTensors(*out)

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
        ids = np.array([self._tok.encode(text).ids], dtype=np.int64)
        # Speech runs ~10 tokens (0.4 s) per word. Cap at about twice that so a bad
        # sample can't run on; a tag like [laugh] counts as a few words.
        words = len(text.split()) + 3 * text.count("[")
        max_new = min(1500, 25 + 20 * words)
        rng = np.random.default_rng(seed)
        with self._lock:
            t0 = time.perf_counter()
            emb = np.concatenate([voice.cond_emb, self._s["embed_tokens"].run(None, {"input_ids": ids})[0]], axis=1)
            pkv = {i.name: np.zeros([1, NUM_KV_HEADS, 0, HEAD_DIM],
                                    np.float16 if i.type == "tensor(float16)" else np.float32)
                   for i in self._kv_inputs}
            mask = np.ones((1, emb.shape[1]), np.int64)
            pos = np.arange(emb.shape[1], dtype=np.int64)[None]
            generated = [START_SPEECH]
            seen = np.zeros(STOP_SPEECH + 1, bool); seen[START_SPEECH] = True
            for _ in range(max_new):
                if stop is not None and stop.is_set():
                    return np.zeros(0, np.float32)
                logits, *present = self._s["language_model"].run(
                    None, dict(inputs_embeds=emb, attention_mask=mask, position_ids=pos, **pkv))
                row = logits[0, -1, :].astype(np.float32)
                idx = np.nonzero(seen[: row.shape[0]])[0]
                s = row[idx]
                row[idx] = np.where(s < 0, s * self.repetition_penalty, s / self.repetition_penalty)
                nxt = self._sample(row, rng)
                generated.append(nxt)
                if nxt == STOP_SPEECH:
                    break
                if nxt < seen.shape[0]:
                    seen[nxt] = True
                emb = self._s["embed_tokens"].run(None, {"input_ids": np.array([[nxt]], np.int64)})[0]
                mask = np.concatenate([mask, np.ones((1, 1), np.int64)], axis=1)
                pos = pos[:, -1:] + 1
                for j, k in enumerate(pkv):
                    pkv[k] = present[j]
            t_lm = time.perf_counter() - t0
            speech = [t for t in generated[1:] if t < START_SPEECH]
            tokens = np.concatenate([voice.prompt_token, np.array([speech], np.int64),
                                     np.full((1, 3), SILENCE, np.int64)], axis=1)
            t1 = time.perf_counter()
            wav = self._s["conditional_decoder"].run(None, dict(
                speech_tokens=tokens, speaker_embeddings=voice.speaker_embeddings,
                speaker_features=voice.speaker_features))[0].squeeze(0).astype(np.float32)
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
