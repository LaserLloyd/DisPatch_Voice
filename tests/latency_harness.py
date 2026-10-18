#!/usr/bin/env python3
"""Latency harness (T4): end of speech → first reply audio, stage by stage, idle and under load.

Two modes:

* ``engines`` (default) drives the realtime engines directly — Silero → ParakeetTailSTT +
  Smart Turn in parallel at the pause → TurnPolicy → Speculator → the 1.2 s stub LLM →
  first sentence → PipelinedChatterbox (or Kokoro) → first 40 ms frame — on the fixture
  clips, with the §4.9 stage table printed per clip and ``platform_profile`` + ``max_mhz``
  on every run so the power-saver and performance-hold columns are never confused. It runs
  **twice** with ``--both``: idle and under ``tests/load.py 8``. ``--contention`` is the
  20-thread run (STT 4 + Smart Turn 2 + LM 4 + 2 × 5 decoder, all running) that gates
  ``rt.decoder_sessions: 2`` on this box. ``--embed-delays`` exercises the stub's
  ``/v1/embeddings`` delay with the 0.3 s voice budget (a simulation of T2's rule, so the
  stub knob is proven; the real advisor path is measured in ``ws`` mode).
* ``ws`` drives a scratch DisPatch instance over ``/ws/voice/{thread}?v=3`` with the same
  clips and asserts the same thresholds from the frames the server sends (``eot``,
  ``say``, first downlink audio, ``metrics``). It needs T3's session hub (slice 3) on the
  other end; until then it is a protocol driver written to C-§C and unverified.

Thresholds (SPEC §12): first audio ≤ 3.8 s cloned (one session, 4 s prompt) / ≤ 2.8 s
Kokoro in ``power-saver``; ≤ 3.0 s / ≤ 2.3 s under the ``performance`` hold; both in the
idle and the load run; speculation ≤ 2 starts per turn; the addendum clip → one merged
submit. ``--assert`` turns a miss into exit code 1. ``--hold`` takes the ``performance``
hold through the add-on's own ``PowerHold`` (SPEC §4.11) for the run, which is how the
performance column is measured (``--hold power-saver`` pins the 2 GHz car state on a box
someone left in performance); ``--early-tail 0.15`` primes the tail transcript at the first
0.15 s of silence (``ParakeetTailSTT.prime``), the default 0 is the probe-time call the
session makes as specified.

    flock "$VOICE_LOCK" uv run python tests/latency_harness.py --both --json out.json
    flock "$VOICE_LOCK" uv run python tests/latency_harness.py --both --hold --json hold.json
    flock "$VOICE_LOCK" uv run python tests/latency_harness.py --contention
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))          # tests/: stub_llm, load, fakes

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
FRAME = 512
SR = 16000
PROBE_SILENCE_S = 0.25
VOICE_EMBED_TIMEOUT_S = 0.3                       # EmbedConfig.voice_timeout_s (T2)
THRESHOLDS = {                                    # seconds, speech end → first reply audio
    ("chatterbox", "power-saver"): 3.8, ("kokoro", "power-saver"): 2.8,
    ("chatterbox", "performance"): 3.0, ("kokoro", "performance"): 2.3,
}
_SENT_END = re.compile(r"([.!?]+[\"')\]]?)(\s+|$)")


# --------------------------------------------------------------------------- #
# Box facts
# --------------------------------------------------------------------------- #

def platform_profile() -> str:
    try:
        return Path("/sys/firmware/acpi/platform_profile").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def max_mhz() -> int:
    try:
        return int(Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq").read_text(encoding="utf-8")) // 1000
    except OSError:
        return 0


def column_of(profile: str, mhz: int) -> str:
    """Which §4.9 column a row measured at (``profile``, ``mhz``) belongs to."""
    return "performance" if profile in ("performance", "balanced") and mhz > 3000 else "power-saver"


def profile_column() -> str:
    """Which §4.9 column this run belongs to (now)."""
    return column_of(platform_profile(), max_mhz())


def load1() -> str:
    try:
        return Path("/proc/loadavg").read_text(encoding="utf-8").split()[0]
    except OSError:
        return "?"


def speech_models_dir() -> Path:
    env = os.environ.get("DISPATCH_SPEECH_MODELS")
    return Path(env).expanduser() if env else Path.home() / ".local/share/dispatch-voice-models"


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #

class Engines:
    def __init__(self, *, tts: str, prompt_s: float, decoder_sessions: int, first_chunk: int,
                 stt_threads: int = 6, lm_threads: int = 4, decoder_threads: int | None = None) -> None:
        from dispatch_voice.host.vad_turn import SileroVAD, SmartTurn
        from dispatch_voice.rt import RtSettings
        from dispatch_voice.rt.stt_stream import ParakeetRecognizer, ort_options
        md = speech_models_dir()
        t0 = time.perf_counter()
        self.silero = SileroVAD(md / "silero_vad.onnx", threads=1)
        import onnxruntime as ort
        st = SmartTurn.__new__(SmartTurn)                       # the host's class, our session options
        st._sess = ort.InferenceSession(str(md / "smart-turn-v3.2-cpu.onnx"), sess_options=ort_options(2),
                                        providers=["CPUExecutionProvider"])
        self.smart_turn = st
        self.recognizer = ParakeetRecognizer(md / "parakeet-tdt-0.6b-v2", threads=stt_threads, spinning=False)
        self.recognizer.transcribe(np.zeros(SR, np.float32))
        self.smart_turn(np.zeros(SR, np.float32))
        self.rt = RtSettings.from_mapping({"decoder_prompt_s": prompt_s, "decoder_sessions": decoder_sessions,
                                           "first_chunk_tokens": first_chunk})
        self.tts_name = tts
        self.tts = self._build_tts(tts, lm_threads, decoder_threads)
        self.load_s = round(time.perf_counter() - t0, 2)

    def _build_tts(self, name: str, lm_threads: int, decoder_threads: int | None):
        if name == "kokoro":
            return KokoroEngine()
        from dispatch_voice.engine import VoiceTensors, default_model_dir
        from dispatch_voice.rt.tts_stream import PipelinedChatterbox

        class StockStore:
            def __init__(self):
                self._v = VoiceTensors.load(default_model_dir() / "stock" / "stock-host.npz")

            def voice(self, pid):
                return self._v

            def get(self, pid):
                return {"id": pid, "created": 1}

            def default_id(self):
                return "stock"

        t = PipelinedChatterbox(StockStore(), resolve=lambda pid: "stock", rt=self.rt, lm_threads=lm_threads,
                                decoder_threads=decoder_threads)
        t.warm()
        list(t.stream("Okay, one moment.", "stock"))                 # kernels warm
        return t


class KokoroEngine:
    """Kokoro fp32 (the GPL plugin's engine when importable, else kokoro_onnx directly)."""

    sample_rate = 24000
    streaming = False

    def __init__(self) -> None:
        self.last_stats: dict = {}
        try:
            from dispatch_voice_kokoro.engine import KokoroTTS  # type: ignore
            md = speech_models_dir() / "kokoro"
            self._eng = KokoroTTS(md / "kokoro-v1.0.onnx", md / "voices-v1.0.bin", voice="af_heart", threads=8)
            self._mode = "plugin"
        except Exception:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro  # type: ignore
            md = speech_models_dir() / "kokoro"
            so = ort.SessionOptions(); so.intra_op_num_threads = 8; so.log_severity_level = 3
            sess = ort.InferenceSession(str(md / "kokoro-v1.0.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
            self._k = Kokoro.from_session(sess, str(md / "voices-v1.0.bin"))
            self._mode = "kokoro_onnx"
        list(self.stream("Okay."))

    def stream(self, text, profile_id=None, stop=None):
        t0 = time.perf_counter()
        if self._mode == "plugin":
            pcm, _ = self._eng.synth(text, None)
        else:
            a, _sr = self._k.create(text, voice="af_heart", speed=1.05, lang="en-us")
            pcm = (np.clip(np.asarray(a, np.float32), -1, 1) * 32767).astype("<i2")
        self.last_stats = {"first_chunk_ms": round((time.perf_counter() - t0) * 1000), "chunks": [], "tokens": 0,
                           "audio_ms": round(len(pcm) / 24)}
        for i in range(0, len(pcm), 960):
            yield pcm[i:i + 960]


# --------------------------------------------------------------------------- #
# The stub LLM client (OpenAI SSE)
# --------------------------------------------------------------------------- #

class LLMStream:
    """Streams one chat completion from the stub; records first-token and first-sentence times."""

    def __init__(self, base_url: str, text: str, *, embed_delay_s: float | None = None) -> None:
        import httpx
        self.base_url = base_url
        self.text = text
        self.started_at = time.perf_counter()
        self.first_token_at: float | None = None
        self.first_sentence: str | None = None
        self.first_sentence_at: float | None = None
        self.full = ""
        self.embed_wait_s = 0.0
        self.discarded = False
        self._done = threading.Event()
        self._sentence_ready = threading.Event()
        self._client = httpx.Client(timeout=30.0)
        self._embed_delay = embed_delay_s
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _retrieval(self) -> None:
        """T2's rule, simulated: the query embedding gets 0.3 s, then keywords only."""
        if self._embed_delay is None:
            return
        t0 = time.perf_counter()
        try:
            self._client.post(f"{self.base_url}/embeddings", json={"input": self.text, "model": "stub"},
                              timeout=VOICE_EMBED_TIMEOUT_S)
        except Exception:
            pass
        self.embed_wait_s = time.perf_counter() - t0

    def _run(self) -> None:
        try:
            self._retrieval()
            body = {"model": "stub", "stream": True, "messages": [{"role": "user", "content": self.text}]}
            with self._client.stream("POST", f"{self.base_url}/chat/completions", json=body) as r:
                for line in r.iter_lines():
                    if self.discarded:
                        break
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        piece = json.loads(data)["choices"][0]["delta"].get("content") or ""
                    except (KeyError, IndexError, json.JSONDecodeError):
                        continue
                    if not piece:
                        continue
                    if self.first_token_at is None:
                        self.first_token_at = time.perf_counter()
                    self.full += piece
                    if self.first_sentence is None:
                        m = _SENT_END.search(self.full)
                        if m:
                            self.first_sentence = self.full[: m.end(1)].strip()
                            self.first_sentence_at = time.perf_counter()
                            self._sentence_ready.set()
        except Exception as e:
            self.error = e
        finally:
            if self.first_sentence is None and self.full.strip():
                self.first_sentence = self.full.strip()
                self.first_sentence_at = time.perf_counter()
            self._sentence_ready.set()
            self._done.set()
            try:
                self._client.close()
            except Exception:
                pass

    def wait_sentence(self, timeout: float = 20.0) -> str | None:
        self._sentence_ready.wait(timeout)
        return self.first_sentence

    async def discard(self) -> None:
        self.discarded = True


# --------------------------------------------------------------------------- #
# One turn through the engines
# --------------------------------------------------------------------------- #

@dataclass
class TurnResult:
    clip: str
    load: str
    profile: str
    max_mhz: int
    load1: str
    utterance_s: float
    speech_end_s: float
    probe_at: float = 0.0
    smart_turn_ms: int = 0
    at_pause_ms: int = 0
    p_done: float = 0.0
    partial: str = ""
    decision: str = ""
    decision_at: float = 0.0
    hold_s: float = 0.0
    speculated: bool = False
    llm_start_at: float = 0.0
    final_ms: int = 0
    final_text: str = ""
    adopted: bool = False
    submit_at: float = 0.0
    llm_first_token_at: float = 0.0
    first_sentence_at: float = 0.0
    first_sentence: str = ""
    tts_first_chunk_ms: int = 0
    tts_first_token_ms: int = 0
    tts_tokens_ready_ms: int = 0
    first_audio_at: float = 0.0
    underruns: int = 0
    speculation_starts: int = 0
    embed_wait_ms: int = 0
    merged_submits: int = 0
    primed: str = "-"                 # "-" off | "miss" primed but re-transcribed | "hit" reused at the probe
    reprobes: int = 0
    notes: list[str] = field(default_factory=list)


def _vad_probs(eng: Engines, x: np.ndarray) -> list[float]:
    eng.silero.reset()
    return [eng.silero(x[i:i + FRAME]) for i in range(0, len(x) - FRAME + 1, FRAME)]


def _speech_end_frame(probs: list[float]) -> int:
    last = 0
    for i, p in enumerate(probs):
        if p >= 0.5:
            last = i
    return last


def run_turn(eng: Engines, clip_name: str, x: np.ndarray, *, llm_base: str, load_label: str,
             patience: float = 1.0, embed_delay_s: float | None = None, text_override: str | None = None,
             second_part: np.ndarray | None = None, early_tail_s: float = 0.0) -> TurnResult:
    """One utterance: feed frames, (prime the tail at ``early_tail_s`` of silence,) probe at speech
    end + 0.25 s, decide, speculate, final, LLM, TTS. A ``reprobe`` row asks Smart Turn again at
    +0.4 s with the real silence, like the session, and ends the turn at 1.0 s of silence."""
    from dispatch_voice.rt.endpoint import (
        REPROBE_AFTER_S,
        REPROBE_END_S,
        Signals,
        TurnPolicy,
    )
    from dispatch_voice.rt.speculate import Speculator
    from dispatch_voice.rt.stt_stream import ParakeetTailSTT

    probs = _vad_probs(eng, x)
    end_frame = _speech_end_frame(probs)
    res = TurnResult(clip=clip_name, load=load_label, profile=platform_profile(), max_mhz=max_mhz(), load1=load1(),
                     utterance_s=round(len(x) / SR, 2), speech_end_s=round((end_frame + 1) * FRAME / SR, 2))
    stt = ParakeetTailSTT(eng.recognizer.transcribe)
    policy = TurnPolicy(eng.rt)
    starts: list[LLMStream] = []

    async def start_fn(text: str):
        s = LLMStream(llm_base, text, embed_delay_s=embed_delay_s)
        starts.append(s)
        return s

    spec = Speculator(start_fn, max_per_turn=eng.rt.speculate_max_per_turn, min_gap_s=eng.rt.speculate_min_gap_s)
    loop = asyncio.new_event_loop()
    speech = x[: (end_frame + 1) * FRAME]

    # Feed everything up to the end of speech (the box hears it in real time; we only time what
    # happens AFTER speech ends, so the feed itself is instantaneous).
    for i in range(end_frame + 1):
        stt.push(x[i * FRAME:(i + 1) * FRAME], vad_prob=probs[i])
    t_end = time.perf_counter()
    # Trailing silence frames arrive in real time until the probe fires; with early-tail the
    # session primes the tail call once ``early_tail_s`` of them are in.
    rest = list(range(end_frame + 1, len(probs)))
    n_early = int(early_tail_s * SR / FRAME) if early_tail_s > 0 else 0
    for i in rest[:n_early]:
        stt.push(x[i * FRAME:(i + 1) * FRAME], vad_prob=probs[i])
    if early_tail_s > 0:
        time.sleep(max(0.0, early_tail_s - (time.perf_counter() - t_end)))
        res.primed = "miss" if stt.prime() else "none"
    for i in rest[n_early:]:
        stt.push(x[i * FRAME:(i + 1) * FRAME], vad_prob=probs[i])
    time.sleep(max(0.0, PROBE_SILENCE_S - (time.perf_counter() - t_end)))
    res.probe_at = round(time.perf_counter() - t_end, 3)

    # Smart Turn and the tail transcript in parallel (threads A and B).
    out: dict = {}

    def smart(silence_s: float) -> float:
        return float(eng.smart_turn(np.concatenate([speech, np.zeros(int(silence_s * SR), np.float32)])))

    def thread_a():
        t0 = time.perf_counter()
        out["p"] = smart(PROBE_SILENCE_S)
        out["smart_ms"] = round((time.perf_counter() - t0) * 1000)

    def thread_b():
        out["partial"] = stt.at_pause()

    ta, tb = threading.Thread(target=thread_a), threading.Thread(target=thread_b)
    ta.start(); tb.start()
    tb.join()
    partial = out["partial"]
    ta.join()
    if early_tail_s > 0 and stt.primed_hits:
        res.primed = "hit"
    res.at_pause_ms, res.smart_turn_ms, res.p_done = partial.cost_ms, out["smart_ms"], round(float(out["p"]), 3)
    res.partial = partial.text
    pause_s = PROBE_SILENCE_S
    d = policy.decide(Signals(p_done=out["p"], partial=partial, pause_s=pause_s, patience=patience))
    res.decision, res.hold_s = d.action, round(d.hold_s, 3)
    res.decision_at = round(time.perf_counter() - t_end, 3)
    while d.action == "reprobe":
        # The session asks again after REPROBE_AFTER_S more silence (Smart Turn on the longer
        # silence; the transcript cannot have changed) and ends at REPROBE_END_S of silence.
        res.reprobes += 1
        pause_s = min(REPROBE_END_S, pause_s + REPROBE_AFTER_S)
        time.sleep(max(0.0, pause_s - (time.perf_counter() - t_end)))
        p2 = smart(pause_s)
        d = policy.decide(Signals(p_done=p2, partial=partial, pause_s=pause_s, patience=patience))
        res.notes.append(f"reprobe at {pause_s:.2f} s: p={p2:.2f} → {d.action} ({d.reason})")
        res.p_done = round(p2, 3)
        res.decision, res.decision_at = d.action, round(time.perf_counter() - t_end, 3)
    if d.speculate:
        res.speculated = loop.run_until_complete(spec.start(partial.text))
        res.llm_start_at = round(time.perf_counter() - t_end, 3)
    if d.action == "listen":
        res.notes.append("listen row: the session waits for hard silence; harness ends at hard_silence × patience")
        time.sleep(max(0.0, eng.rt.hard_silence_s * patience - (time.perf_counter() - t_end)))
    elif d.action == "hold":
        time.sleep(d.hold_s)

    if second_part is not None:
        # The "oh, and…" addendum arrives inside addendum_s: merge → discard → one merged submit.
        loop.run_until_complete(spec.discard())
        p2 = _vad_probs(eng, second_part)
        for i, p in enumerate(p2):
            stt.push(second_part[i * FRAME:(i + 1) * FRAME], vad_prob=p)
        t_end = time.perf_counter()
        time.sleep(PROBE_SILENCE_S)
        partial = stt.at_pause()
        res.notes.append(f"addendum merged: '{partial.text}'")
        res.merged_submits = 1

    t0 = time.perf_counter()
    final = stt.final()
    res.final_ms = round((time.perf_counter() - t0) * 1000)
    res.final_text = final
    run = loop.run_until_complete(spec.commit(final))
    res.adopted = run is not None
    res.speculation_starts = spec.stats["started"]
    if run is None:
        run = LLMStream(llm_base, text_override or final, embed_delay_s=embed_delay_s)
        starts.append(run)
    res.submit_at = round(time.perf_counter() - t_end, 3)
    sentence = run.wait_sentence(20.0)
    res.embed_wait_ms = round(run.embed_wait_s * 1000)
    if run.first_token_at:
        res.llm_first_token_at = round(run.first_token_at - t_end, 3)
    if run.first_sentence_at:
        res.first_sentence_at = round(run.first_sentence_at - t_end, 3)
    res.first_sentence = sentence or ""
    if not sentence:
        res.notes.append("no sentence from the LLM")
        loop.close()
        return res
    # TTS: first 40 ms frame, and a player model for underruns.
    t_tts = time.perf_counter()
    frames_at: list[tuple[float, int]] = []
    for fr in eng.tts.stream(sentence, "stock"):
        frames_at.append((time.perf_counter(), len(fr)))
    first = frames_at[0][0]
    played = 0
    for t, n in frames_at:
        if t > first + played / 24000 + 0.05:
            res.underruns += 1
        played += n
    res.tts_first_chunk_ms = round((first - t_tts) * 1000)
    st = getattr(eng.tts, "last_stats", {}) or {}
    res.tts_first_token_ms = int(st.get("first_token_ms") or 0)
    res.tts_tokens_ready_ms = int(st.get("tokens_ready_ms") or 0)
    res.first_audio_at = round(first - t_end, 3)
    for s in starts:
        s.discarded = True
    loop.close()
    return res


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #

def load_clip(name: str) -> np.ndarray:
    from fakes import load_clip as _lc
    return _lc(name)


def addendum_parts() -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads((FIXTURES / "addendum.json").read_text(encoding="utf-8"))
    x = load_clip("addendum")
    cut = meta["first_end_sample"] + 3200          # 0.2 s after the first half ends
    return x[:cut], x[cut:]                            # the rest carries the real 2.0 s gap


def engines_pass(eng: Engines, clips: list[str], *, llm_base: str, load_label: str, runs: int,
                 embed_delay_s: float | None, early_tail_s: float = 0.0) -> list[TurnResult]:
    results = []
    for name in clips:
        for _ in range(runs):
            if name == "addendum":
                a, b = addendum_parts()
                results.append(run_turn(eng, name, a, llm_base=llm_base, load_label=load_label,
                                        embed_delay_s=embed_delay_s, second_part=b, early_tail_s=early_tail_s))
            else:
                results.append(run_turn(eng, name, load_clip(name), llm_base=llm_base, load_label=load_label,
                                        embed_delay_s=embed_delay_s, early_tail_s=early_tail_s))
    return results


def table(results: list[TurnResult], tts: str) -> str:
    cols = ["clip", "load", "profile", "mhz", "probe", "prime", "smart", "pause", "p", "decide", "at", "spec",
            "final", "llm1st", "sent", "lm1tok", "tokrdy", "tts1st", "FIRST AUDIO", "under"]
    lines = [" | ".join(cols)]
    for r in results:
        lines.append(" | ".join(str(v) for v in [
            r.clip, r.load, r.profile, r.max_mhz, f"{r.probe_at:.2f}", r.primed, f"{r.smart_turn_ms}ms",
            f"{r.at_pause_ms}ms", f"{r.p_done:.2f}", r.decision + (f"({r.reprobes}r)" if r.reprobes else ""),
            f"{r.decision_at:.2f}", "y" if r.speculated else "n", f"{r.final_ms}ms", f"{r.llm_first_token_at:.2f}",
            f"{r.first_sentence_at:.2f}", f"{r.tts_first_token_ms}ms",
            f"{r.tts_tokens_ready_ms}ms" if r.tts_tokens_ready_ms else "-",
            f"{r.tts_first_chunk_ms}ms", f"{r.first_audio_at:.2f}s", r.underruns]))
    return "\n".join(lines)


def verdicts(results: list[TurnResult], tts: str) -> list[str]:
    """One PASS/FAIL per (load pass, §4.9 column): every row is judged against the
    threshold of the profile it RAN in (``TurnResult.profile``/``max_mhz``), never the
    profile at the end of the run; a pass whose rows span two columns is flagged MIXED
    and each part is scored on its own (review T4 #6)."""
    out = []
    by_load: dict[str, list[TurnResult]] = {}
    for r in results:
        if r.first_audio_at:
            by_load.setdefault(r.load, []).append(r)
    for load, all_rows in by_load.items():
        groups: dict[str, list[TurnResult]] = {}
        for r in all_rows:
            groups.setdefault(column_of(r.profile, r.max_mhz), []).append(r)
        mixed = len(groups) > 1
        if mixed:
            out.append(f"MIXED {load}: the platform profile changed during this pass "
                       f"({', '.join(f'{c}: {len(rs)} rows' for c, rs in groups.items())}); each part is scored alone")
        for col, rows in groups.items():
            vals = [r.first_audio_at for r in rows]
            thr = THRESHOLDS[(tts, col)]
            med = statistics.median(vals)
            ok = med <= thr
            profiles = sorted({f"{r.profile}/{r.max_mhz}" for r in rows})
            out.append(f"{'PASS' if ok else 'FAIL'} first audio {load}: median {med:.2f} s (max {max(vals):.2f}) "
                       f"≤ {thr} s [{tts}, {col} column, {len(rows)} rows, platform_profile={'|'.join(profiles)}"
                       f"{', MIXED pass' if mixed else ''}]")
            out.append(f"INFO stages {load} [{col}] (medians): decision {statistics.median(r.decision_at for r in rows):.2f} s, "
                       f"first sentence {statistics.median(r.first_sentence_at for r in rows):.2f} s, "
                       f"tts first token {statistics.median(r.tts_first_token_ms for r in rows):.0f} ms, "
                       f"tokens ready {statistics.median(r.tts_tokens_ready_ms for r in rows):.0f} ms, "
                       f"tts first chunk {statistics.median(r.tts_first_chunk_ms for r in rows):.0f} ms; "
                       f"primed hit/miss/none {sum(r.primed == 'hit' for r in rows)}/"
                       f"{sum(r.primed == 'miss' for r in rows)}/{sum(r.primed == 'none' for r in rows)}")
    starts = max((r.speculation_starts for r in results), default=0)
    out.append(f"{'PASS' if starts <= 2 else 'FAIL'} speculation ≤ 2 starts per turn (max {starts})")
    under = sum(r.underruns for r in results)
    out.append(f"{'PASS' if under == 0 else 'WARN'} underruns in the first sentence: {under}")
    add = [r for r in results if r.clip == "addendum"]
    if add:
        ok = all(r.merged_submits == 1 and "tomorrow" in r.final_text.lower().replace(" ", "") for r in add)
        out.append(f"{'PASS' if ok else 'FAIL'} addendum → one merged submit with both parts")
    return out


def contention_run(eng: Engines, *, llm_base: str) -> dict:
    """The 20-thread run: STT 4 threads + Smart Turn 2 looping while the pipelined TTS (LM 4 +
    2 × 5 decoder sessions) speaks the 25-word sentence; reports first audio, underruns and the
    STT/Smart Turn costs under contention. Gates rt.decoder_sessions: 2 on this box."""
    from dispatch_voice.rt.stt_stream import ParakeetRecognizer
    md = speech_models_dir()
    rec4 = ParakeetRecognizer(md / "parakeet-tdt-0.6b-v2", threads=4, spinning=False)
    x = load_clip("medium")
    rec4.transcribe(x)
    stop = threading.Event()
    stt_ms: list[float] = []
    st_ms: list[float] = []

    def stt_loop():
        while not stop.is_set():
            t0 = time.perf_counter(); rec4.transcribe(x[:56000]); stt_ms.append((time.perf_counter() - t0) * 1000)

    def st_loop():
        while not stop.is_set():
            t0 = time.perf_counter(); eng.smart_turn(x); st_ms.append((time.perf_counter() - t0) * 1000)
            time.sleep(0.05)

    sentence = ("So I was looking at the plan for the garage conversion again last night and I think the "
                "budget we talked about is a little tight once you add the work.")
    th = [threading.Thread(target=stt_loop, daemon=True), threading.Thread(target=st_loop, daemon=True)]
    for t in th:
        t.start()
    time.sleep(0.5)
    t0 = time.perf_counter()
    frames_at = []
    for fr in eng.tts.stream(sentence, "stock"):
        frames_at.append((time.perf_counter(), len(fr)))
    stop.set()
    for t in th:
        t.join(timeout=10)
    first = frames_at[0][0]
    played, under = 0, 0
    for t, n in frames_at:
        if t > first + played / 24000 + 0.05:
            under += 1
        played += n
    col = profile_column()
    thr = THRESHOLDS[(eng.tts_name, col)]
    first_ms = round((first - t0) * 1000)
    # The gate: first audio inside the §4.9 TTS-first-chunk share of the budget (first audio ≤
    # threshold − (decision 0.8 + LLM 1.2)), no underrun, and final()-class STT calls ≤ 1.2 s.
    tts_budget_s = thr - 2.0
    stt_med = statistics.median(stt_ms) if stt_ms else 0
    reasons = []
    if first_ms / 1000 > tts_budget_s:
        reasons.append(f"first audio {first_ms} ms > {tts_budget_s:.1f} s budget")
    if under:
        reasons.append(f"{under} underrun frames")
    if stt_med > 1200:
        reasons.append(f"STT tail {stt_med:.0f} ms > 1200")
    st = eng.tts.last_stats
    return {"decoder_sessions": eng.rt.decoder_sessions, "first_chunk_tokens": eng.rt.first_chunk_tokens,
            "first_audio_ms": first_ms, "underruns": under, "stt_tail_3.5s_ms_median": round(stt_med),
            "stt_tail_3.5s_ms_max": round(max(stt_ms)) if stt_ms else 0, "stt_calls": len(stt_ms),
            "smart_turn_ms_median": round(statistics.median(st_ms)) if st_ms else 0, "smart_turn_calls": len(st_ms),
            "tts_first_token_ms": st.get("first_token_ms"), "tts_tokens_ready_ms": st.get("tokens_ready_ms"),
            "tts_chunks": st.get("chunks"), "platform_profile": platform_profile(),
            "max_mhz": max_mhz(), "load1": load1(), "tts_budget_s": tts_budget_s,
            "gate_decoder_sessions_2": "PASS" if not reasons else "FAIL", "gate_reasons": reasons}


# --------------------------------------------------------------------------- #
# ws mode (protocol driver for a scratch instance; needs T3 slice 3)
# --------------------------------------------------------------------------- #

def pack_up(off: int, pcm16: bytes, flags: int = 0) -> bytes:
    """C-§C.2 uplink frame, codec 0 (pcm16), one packet."""
    import struct
    return struct.pack("<BBBIB", 0x03, 0, flags, off & 0xFFFFFFFF, 1) + struct.pack("<H", len(pcm16)) + pcm16


async def ws_turn(url: str, x: np.ndarray, clip: str, *, start: dict) -> dict:
    import websockets
    res = {"clip": clip, "profile": platform_profile(), "max_mhz": max_mhz()}
    t_end = None
    async with websockets.connect(url, max_size=1 << 20) as ws:
        ready = None
        while ready is None:
            msg = await asyncio.wait_for(ws.recv(), 30)
            if isinstance(msg, str):
                f = json.loads(msg)
                if f.get("type") == "ready":
                    ready = f
        res["ready"] = {k: ready.get(k) for k in ("proto", "sid", "rt", "power_hold", "codec_out")}
        await ws.send(json.dumps({"type": "start", **start}))
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
        off = 0
        step = 640                                   # 20 ms
        t0 = time.perf_counter()
        for n, i in enumerate(range(0, len(pcm), step), start=1):
            await ws.send(pack_up(off, pcm[i:i + step]))
            off += step // 2
            await asyncio.sleep(max(0.0, t0 + n * 0.02 - time.perf_counter()))
        t_end = time.perf_counter()
        silence = b"\x00" * step
        first_audio = None
        deadline = t_end + 25
        while time.perf_counter() < deadline and first_audio is None:
            await ws.send(pack_up(off, silence)); off += step // 2
            try:
                msg = await asyncio.wait_for(ws.recv(), 0.02)
            except TimeoutError:
                continue
            if isinstance(msg, (bytes, bytearray)):
                if msg and msg[0] == 0x03 and msg[2] == 0:   # downlink kind 0 = reply
                    first_audio = time.perf_counter()
                continue
            f = json.loads(msg)
            t = f.get("type")
            if t in ("eot", "eot_soft", "transcript", "say", "metrics", "state", "partial", "ack"):
                res.setdefault("frames", []).append({"t": round(time.perf_counter() - t_end, 3), **f})
        res["first_audio_at"] = round(first_audio - t_end, 3) if first_audio else None
        await ws.send(json.dumps({"type": "cancel"}))
    return res


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("engines", "ws"), default="engines")
    ap.add_argument("--tts", choices=("chatterbox", "kokoro"), default="chatterbox")
    ap.add_argument("--clips", default="short,medium,long,pause,addendum")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--load", type=int, default=0, help="busy processes during the run (0 = idle)")
    ap.add_argument("--both", action="store_true", help="run idle, then under --load-n processes")
    ap.add_argument("--load-n", type=int, default=8)
    ap.add_argument("--prompt-s", type=float, default=4.0)
    ap.add_argument("--decoder-sessions", type=int, default=1)
    ap.add_argument("--first-chunk", type=int, default=0, help="0 = the RtSettings default for the session count")
    ap.add_argument("--llm-delay", type=float, default=1.2)
    ap.add_argument("--embed-delays", default="", help="e.g. 0,1.7,2.5: simulate the advisor retrieval budget")
    ap.add_argument("--contention", action="store_true")
    ap.add_argument("--hold", nargs="?", const="performance", default=None, choices=("performance", "power-saver"),
                    help="hold this power profile for the run through the add-on's PowerHold (SPEC §4.11): "
                         "`--hold` = performance (the hold column); `--hold power-saver` pins the 2 GHz car "
                         "state on a box someone left in performance. Released at the end either way.")
    ap.add_argument("--early-tail", type=float, default=0.0,
                    help="prime the tail transcript (ParakeetTailSTT.prime) at this much silence; 0 (default) = "
                         "the session as specified, one call at the 0.25 s probe; 0.15 = the measured experiment")
    ap.add_argument("--lm-threads", type=int, default=4)
    ap.add_argument("--decoder-threads", type=int, default=0, help="0 = the default for the session count")
    ap.add_argument("--json", default="")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--url", default="", help="ws mode: ws://127.0.0.1:19040/ws/voice/<thread>?d=…&s=…&v=3&up=pcm16&down=pcm16")
    a = ap.parse_args(argv)

    hold = None
    hold_state = "off"
    if a.hold:
        from dispatch_voice.host.power import PowerHold
        hold = PowerHold(profile=a.hold)
        hold_state = f"{a.hold}:{hold.acquire()}"
        if hold_state.endswith(":held"):
            time.sleep(2.0)                                   # let the governor settle
    try:
        return _main(a, hold_state)
    finally:
        if hold is not None:
            hold.release_all()


def _main(a, hold_state: str) -> int:
    header = {"mode": a.mode, "tts": a.tts, "platform_profile": platform_profile(), "max_mhz": max_mhz(),
              "column": profile_column(), "power_hold": hold_state, "load1": load1(), "prompt_s": a.prompt_s,
              "decoder_sessions": a.decoder_sessions, "llm_delay_s": a.llm_delay, "early_tail_s": a.early_tail,
              "lm_threads": a.lm_threads, "decoder_threads": a.decoder_threads or None,
              "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(json.dumps({"harness": header}))
    report: dict = {"header": header}

    if a.mode == "ws":
        if not a.url:
            print("ws mode needs --url", file=sys.stderr)
            return 2
        start = {"patience": "normal", "barge_in": True, "reply_style": "short", "follow_up": False,
                 "acks": "off", "rt": {"partials": True, "speculate": True, "backchannels": False, "optimistic_duck": True}}
        rows = [asyncio.run(ws_turn(a.url, load_clip(c), c, start=start)) for c in a.clips.split(",")]
        report["ws"] = rows
        for r in rows:
            print(json.dumps(r))
        if a.json:
            Path(a.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
        return 0

    from load import cpu_load
    from stub_llm import StubLLM
    first_chunk = a.first_chunk or (15 if a.decoder_sessions == 2 else 25)
    eng = Engines(tts=a.tts, prompt_s=a.prompt_s, decoder_sessions=a.decoder_sessions, first_chunk=first_chunk,
                  lm_threads=a.lm_threads, decoder_threads=a.decoder_threads or None)
    print(json.dumps({"engines_loaded_s": eng.load_s}))
    results: list[TurnResult] = []
    clips = a.clips.split(",")
    with StubLLM(delay_s=a.llm_delay) as llm:
        if a.contention:
            report["contention"] = contention_run(eng, llm_base=llm.base_url)
            print(json.dumps({"contention": report["contention"]}))
        else:
            passes = [("idle", 0)] + ([("load8", a.load_n)] if a.both else []) if a.load == 0 else [(f"load{a.load}", a.load)]
            for label, n in passes:
                if n:
                    with cpu_load(n):
                        time.sleep(1.0)
                        results += engines_pass(eng, clips, llm_base=llm.base_url, load_label=label, runs=a.runs,
                                                embed_delay_s=None, early_tail_s=a.early_tail)
                else:
                    results += engines_pass(eng, clips, llm_base=llm.base_url, load_label=label, runs=a.runs,
                                            embed_delay_s=None, early_tail_s=a.early_tail)
            if a.embed_delays:
                for d in [float(v) for v in a.embed_delays.split(",") if v.strip()]:
                    llm.configure(embed_delay_s=d)
                    results += engines_pass(eng, ["short"], llm_base=llm.base_url, load_label=f"embed{d}",
                                            runs=1, embed_delay_s=d, early_tail_s=a.early_tail)
                llm.configure(embed_delay_s=0.0)
    if results:
        print(table(results, a.tts))
        vs = verdicts(results, a.tts)
        print("\n".join(vs))
        report["results"] = [asdict(r) for r in results]
        report["verdicts"] = vs
        emb = [r for r in results if r.load.startswith("embed")]
        if emb:
            base = statistics.median([r.first_audio_at for r in results if r.load == "idle" and r.clip == "short"] or [0])
            for r in emb:
                extra = r.first_audio_at - base
                print(f"{'PASS' if extra <= 0.3 + 0.15 else 'FAIL'} {r.load}: embed wait {r.embed_wait_ms} ms, "
                      f"first audio {r.first_audio_at:.2f} s (+{extra:.2f} vs idle short; budget +0.3 s) [simulated]")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    if a.do_assert and any(v.startswith("FAIL") for v in report.get("verdicts", [])):
        return 1
    if a.do_assert and report.get("contention", {}).get("gate_decoder_sessions_2") == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
