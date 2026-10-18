"""Model-backed realtime tests (T4): Parakeet tail mode, Smart Turn on partials, the
pipelined Chatterbox decoder. Marker ``models``; skipped when the speech models are not
on disk. Run them under the box-wide lock — they load ≈ 1.4 GB of models:

    flock "$VOICE_LOCK" uv run --with onnx-asr pytest tests/test_models_rt.py

Every threshold here is the measured one from SPEC §4 / design-realtime §2, taken at the
2 GHz power-saver car state; a faster profile only makes them easier.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from rt.conftest import (
    have_chatterbox,
    have_onnx_asr,
    have_smart_turn,
    have_speech_models,
    speech_models_dir,
)

pytestmark = [pytest.mark.models]

FIXTURES = Path(__file__).parent / "fixtures"
META = json.loads((FIXTURES / "clips" / "meta.json").read_text(encoding="utf-8"))
BASELINE = json.loads((FIXTURES / "transcripts.json").read_text(encoding="utf-8"))
FRAME = 512

need_stt = pytest.mark.skipif(not (have_speech_models() and have_onnx_asr()),
                              reason="Parakeet/Silero models or onnx-asr missing")
need_turn = pytest.mark.skipif(not (have_speech_models() and have_smart_turn()), reason="Smart Turn model missing")
need_tts = pytest.mark.skipif(not have_chatterbox(), reason="run `dispatch-voice fetch-models`")


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower().replace("’", "'"))


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[len(r), len(h)]) / len(r)


def clip(name: str, band: str = "16k") -> np.ndarray:
    from fakes import load_clip
    return load_clip(name, band=band)


def profile_line() -> str:
    try:
        prof = Path("/sys/firmware/acpi/platform_profile").read_text(encoding="utf-8").strip()
        mhz = int(Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq").read_text(encoding="utf-8")) // 1000
    except OSError:
        prof, mhz = "?", 0
    return f"platform_profile={prof} max_mhz={mhz}"


# --------------------------------------------------------------------------- #
# STT: Parakeet tail mode
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def recognizer():
    from dispatch_voice.rt.stt_stream import ParakeetRecognizer
    r = ParakeetRecognizer(speech_models_dir() / "parakeet-tdt-0.6b-v2", threads=6, spinning=False)
    r.transcribe(np.zeros(16000, np.float32))     # kernels warm
    return r


@pytest.fixture(scope="module")
def silero():
    from dispatch_voice.host.vad_turn import SileroVAD
    return SileroVAD(speech_models_dir() / "silero_vad.onnx", threads=1)


def vad_probs(silero, x: np.ndarray) -> list[float]:
    silero.reset()
    return [silero(x[i:i + FRAME]) for i in range(0, len(x) - FRAME + 1, FRAME)]


def run_tail(recognizer, silero, x: np.ndarray, *, pause_min_s: float = 0.3):
    """Feed a clip frame by frame; call at_pause() at every VAD pause ≥ pause_min_s (once per
    pause) and tick() every second; return (partials, final, pause calls)."""
    from dispatch_voice.rt.stt_stream import ParakeetTailSTT
    stt = ParakeetTailSTT(recognizer.transcribe)
    probs = vad_probs(silero, x)
    partials, run, last_tick = [], 0, -1.0
    probed = False
    for i, p in enumerate(probs):
        stt.push(x[i * FRAME:(i + 1) * FRAME], vad_prob=p)
        t = (i + 1) * FRAME / 16000
        if p < 0.35:
            run += 1
        else:
            run, probed = 0, False
        if run * FRAME / 16000 >= pause_min_s and not probed:
            probed = True
            partials.append(("pause", t, stt.at_pause()))
        elif t - last_tick >= 1.0 and p >= 0.5:
            last_tick = t
            pt = stt.tick()
            if pt is not None:
                partials.append(("tick", t, pt))
    t0 = time.perf_counter()
    final = stt.final()
    return partials, final, time.perf_counter() - t0, stt


@need_stt
@pytest.mark.parametrize("name", ["medium", "long", "pause"])
def test_partials_are_stable_except_the_last_two_words(recognizer, silero, name):
    partials, final, final_s, _stt = run_tail(recognizer, silero, clip(name))
    ref = words(BASELINE["16k"][name])
    assert wer(BASELINE["16k"][name], final) <= 0.05, (final, final_s)
    assert partials, "no partial was produced"
    for kind, t, p in partials:
        w = words(p.text)
        # Leading words agree with the final transcript except for the last ≤ 2 of the partial.
        head = w[:-2] if len(w) > 2 else []
        assert head == ref[:len(head)], (kind, t, p.text)
        assert p.stable_words <= max(0, len(words(p.tail)) - 2 + 2)
        assert p.cost_ms < 2500
    # Committed text never changes once set.
    committed = [p.committed for _, _, p in partials if p.committed]
    for a, b in itertools.pairwise(committed):
        assert b.startswith(a)
    print(f"\n{name}: {len(partials)} partials, final {final_s * 1000:.0f} ms, "
          f"pause costs {[p.cost_ms for k, _, p in partials if k == 'pause']} ms, {profile_line()}")


@need_stt
def test_long_monologue_commits_and_bounds_the_tail(recognizer, silero):
    partials, final, _, stt = run_tail(recognizer, silero, clip("long"))
    assert stt.committed, "the 15 s clip should have closed at least one segment"
    assert wer(BASELINE["16k"]["long"], final) <= 0.05
    costs = [p.cost_ms for k, _, p in partials if k == "pause"]
    assert max(costs) < 2000, costs      # bounded: never the whole 15 s at once (≈ 1.1 s one-shot)


@need_stt
def test_tail_cap_bounds_every_call_when_the_vad_never_sees_silence(recognizer):
    """Review T4 #3: with Silero held up (road noise, a radio) no ≥ 0.3 s silence ever
    qualifies, so the 15 s clip reaches the first probe as one tail. The cap closes it at
    its quietest old frame: no recogniser call sees more than tail_max_s (10 s) and the
    words survive the forced cut."""
    from dispatch_voice.rt.stt_stream import ParakeetTailSTT
    sizes: list[float] = []

    def rec(audio):
        sizes.append(audio.size / 16000)
        return recognizer.transcribe(audio)

    x = clip("long")
    stt = ParakeetTailSTT(rec)
    for i in range(0, len(x) - FRAME + 1, FRAME):
        stt.push(x[i:i + FRAME], vad_prob=0.9)           # every frame "speech"
    t0 = time.perf_counter()
    p = stt.at_pause()
    pause_ms = (time.perf_counter() - t0) * 1000
    final = stt.final()
    assert max(sizes) <= stt.tail_max_s + 0.05, sizes
    assert stt.committed and p.committed == stt.committed
    assert sum(sizes) == pytest.approx(len(x) / 16000, abs=0.1)
    w = wer(BASELINE["16k"]["long"], final)
    assert w <= 0.10, (w, final)
    print(f"\nlong, no VAD silence: calls {[round(s, 1) for s in sizes]} s, at_pause {pause_ms:.0f} ms, "
          f"WER {w:.3f}, {profile_line()}")


@need_stt
def test_final_after_a_fresh_pause_partial_is_instant(recognizer, silero):
    from dispatch_voice.rt.stt_stream import ParakeetTailSTT
    x = clip("short")
    stt = ParakeetTailSTT(recognizer.transcribe)
    probs = vad_probs(silero, x)
    for i, p in enumerate(probs):
        stt.push(x[i * FRAME:(i + 1) * FRAME], vad_prob=p)
    t0 = time.perf_counter()
    partial = stt.at_pause()
    pause_ms = (time.perf_counter() - t0) * 1000
    # The turn ends at the pause just transcribed: < 0.3 s of new audio → no second call.
    stt.push(np.zeros(FRAME * 4, np.float32), vad_prob=0.02)
    t0 = time.perf_counter()
    final = stt.final()
    final_ms = (time.perf_counter() - t0) * 1000
    assert final == partial.text and final_ms < 20 and stt.calls == 1
    assert wer(BASELINE["16k"]["short"], final) == 0.0
    print(f"\nshort: at_pause {pause_ms:.0f} ms, final {final_ms:.1f} ms, {profile_line()}")


@need_stt
def test_final_during_an_inflight_pause_returns_within_300ms_of_it(recognizer, silero):
    from dispatch_voice.rt.stt_stream import ParakeetTailSTT
    x = clip("medium")
    stt = ParakeetTailSTT(recognizer.transcribe)
    probs = vad_probs(silero, x)
    for i, p in enumerate(probs):
        stt.push(x[i * FRAME:(i + 1) * FRAME], vad_prob=p)
    got = {}
    th = threading.Thread(target=lambda: got.setdefault("p", stt.at_pause()))
    th.start()
    time.sleep(0.02)
    t_pause_returned = {}

    def watch():
        th.join()
        t_pause_returned["t"] = time.perf_counter()

    w = threading.Thread(target=watch)
    w.start()
    final = stt.final()
    t_final = time.perf_counter()
    w.join()
    assert final == got["p"].text
    assert (t_final - t_pause_returned["t"]) * 1000 < 300


@need_stt
def test_prime_at_015s_of_silence_is_reused_at_the_probe(recognizer, silero):
    """The tail call started at 0.15 s of silence answers the 0.25 s probe: same words as a
    probe-time call (measured identical with 0.10–0.25 s of trailing silence on every clip),
    one recogniser call, and the probe's at_pause() returns in a few ms once it is done."""
    from dispatch_voice.rt.stt_stream import ParakeetTailSTT
    x = clip("medium")
    probs = vad_probs(silero, x)
    end = max(i for i, p in enumerate(probs) if p >= 0.5)
    plain = ParakeetTailSTT(recognizer.transcribe)
    primed = ParakeetTailSTT(recognizer.transcribe)
    for stt in (plain, primed):
        for i in range(end + 1):
            stt.push(x[i * FRAME:(i + 1) * FRAME], vad_prob=probs[i])
    sil = np.zeros(FRAME, np.float32)
    for _ in range(5):                                   # 0.16 s of silence
        primed.push(sil, vad_prob=0.02)
    assert primed.prime() is True
    primed._prime_thread.join(5.0)
    calls_after_prime = primed.calls                     # 2 on this clip: it closes a segment + the tail
    for stt in (plain, primed):
        for _ in range(8):                               # the probe fires at 0.25 s
            stt.push(sil, vad_prob=0.02)
    t0 = time.perf_counter()
    p_plain = plain.at_pause()
    plain_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    p_primed = primed.at_pause()
    primed_ms = (time.perf_counter() - t0) * 1000
    assert p_primed.text == p_plain.text and wer(BASELINE["16k"]["medium"], p_primed.text) == 0.0
    assert primed.calls == calls_after_prime == plain.calls      # the probe made no new call
    assert primed.primed_hits == 1 and primed_ms < 30, primed_ms
    assert primed.final() == p_primed.text and primed.calls == calls_after_prime
    print(f"\nmedium: probe-time at_pause {plain_ms:.0f} ms, primed at_pause {primed_ms:.1f} ms "
          f"(primed call cost {p_primed.cost_ms} ms), {profile_line()}")


@need_stt
def test_8khz_band_limited_clips_wer(recognizer):
    """Feasibility M3: the worklet survives an 8 kHz capture; WER vs the 16 kHz transcripts
    (threshold from the first run: identical except the 0.87 s backchannel)."""
    rows = {}
    for name in ("short", "medium", "long", "pause", "interrupt", "driver1b", "driver2", "addendum"):
        hyp = recognizer.transcribe(clip(name, band="8k"))
        rows[name] = (round(wer(BASELINE["16k"][name], hyp), 3), hyp)
    total = float(np.mean([w for w, _ in rows.values()]))
    print(f"\n8 kHz WER per clip: { {k: v[0] for k, v in rows.items()} } mean {total:.3f}")
    assert total <= 0.05, rows


@need_stt
def test_spinning_is_off_on_the_loaded_recogniser(recognizer):
    assert recognizer.spinning is False and recognizer.threads == 6


# --------------------------------------------------------------------------- #
# Smart Turn on partial audio (design-realtime §2.1 table, ± 0.1)
# --------------------------------------------------------------------------- #

#: (clip, cut at s, P(done) on onnxruntime 1.29, P(done) on onnxruntime 1.31). The design's
#: table was measured on 1.29; the same clips, code and model under 1.31 move two rows
#: (medium@2.0 0.76 → 0.52, the 0.87 s backchannel 0.55 → 0.36; everything else within
#: 0.03). The decision band of the policy is unchanged for the rows that reach it
#: ("…call the dentist" has no terminal punctuation → hold either way). The assertion
#: uses the column of the running runtime, so a runtime bump re-measures rather than
#: silently passing.
TABLE = [
    ("short", None, 0.99, 0.99), ("medium", None, 0.95, 0.93), ("medium", 2.0, 0.76, 0.52),
    ("medium", 3.5, 0.97, 0.97), ("medium", 5.0, 0.21, 0.21), ("medium", 6.5, 0.15, 0.15),
    ("long", 2.0, 0.013, 0.013), ("long", 3.5, 0.021, 0.021), ("long", 5.0, 0.008, 0.008),
    ("long", 6.5, 0.016, 0.016), ("pause_a", None, 0.58, 0.58), ("backchannel", None, 0.55, 0.36),
    ("interrupt", None, 0.99, 0.99),
]


def _ort_minor() -> tuple[int, int]:
    import onnxruntime as ort
    a, b = ort.__version__.split(".")[:2]
    return int(a), int(b)


@pytest.fixture(scope="module")
def smart_turn():
    from dispatch_voice.host.vad_turn import SmartTurn
    return SmartTurn(speech_models_dir() / "smart-turn-v3.2-cpu.onnx", threads=2)


@need_turn
@pytest.mark.parametrize("name,cut,p129,p131", TABLE)
def test_smart_turn_table(smart_turn, name, cut, p129, p131):
    x = clip(name)
    if cut is not None:
        x = x[: int(cut * 16000)]
    p = smart_turn(np.concatenate([x, np.zeros(4000, np.float32)]))     # + 0.25 s of silence
    expected = p131 if _ort_minor() >= (1, 30) else p129
    assert abs(p - expected) <= 0.1, (name, cut, p, expected)


@need_turn
def test_smart_turn_cost(smart_turn):
    x = clip("medium")
    smart_turn(x)
    t0 = time.perf_counter()
    for _ in range(5):
        smart_turn(x)
    ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"\nSmart Turn {ms:.0f} ms/call, {profile_line()}")
    assert ms < 600


# --------------------------------------------------------------------------- #
# TTS: pipelined Chatterbox
# --------------------------------------------------------------------------- #

SENTENCE_25 = ("So I was looking at the plan for the garage conversion again last night and I think the "
               "budget we talked about is a little tight once you add the work.")


class _StockStore:
    def __init__(self):
        from dispatch_voice.engine import VoiceTensors, default_model_dir
        self._v = VoiceTensors.load(default_model_dir() / "stock" / "stock-host.npz")

    def voice(self, pid):
        return self._v

    def get(self, pid):
        return {"id": pid, "created": 1}

    def default_id(self):
        return "stock"


@pytest.fixture(scope="module")
def tts():
    from dispatch_voice.rt import RtSettings
    from dispatch_voice.rt.tts_stream import PipelinedChatterbox
    t = PipelinedChatterbox(_StockStore(), resolve=lambda pid: "stock", rt=RtSettings())
    t.warm()
    list(t.stream("Okay.", "stock"))           # kernels warm
    return t


def play_model(frames_at: list[tuple[float, int]], first_at: float, slack_s: float = 0.05) -> int:
    """A player that starts at the first frame: how many frames arrived after their play time."""
    under, played = 0, 0
    for t, n in frames_at:
        if t > first_at + played / 24000 + slack_s:
            under += 1
        played += n
    return under


@need_tts
def test_first_audio_within_two_seconds_and_no_underrun(tts, idle_cpu):
    t0 = time.perf_counter()
    frames_at: list[tuple[float, int]] = []
    for fr in tts.stream(SENTENCE_25, "stock"):
        frames_at.append((time.perf_counter() - t0, len(fr)))
    first = frames_at[0][0]
    under = play_model(frames_at, first)
    st = tts.last_stats
    print(f"\nfirst audio {first * 1000:.0f} ms, chunks {[(c['decode_ms'], c['at_ms']) for c in st['chunks']]}, "
          f"tokens {st['tokens']}, audio {st['audio_ms']} ms, total {st['total_ms']} ms, {profile_line()}")
    assert first <= 2.0, first
    assert under == 0, f"{under} frames arrived late"
    assert st["tokens"] > 60 and st["audio_ms"] > 2500


@need_tts
def test_seams_have_no_jump_beyond_local_speech_dynamics(tts):
    pcm = np.concatenate(list(tts.stream(SENTENCE_25, "stock"))).astype(np.float32) / 32767.0
    seams = tts.last_stats["seams"]
    levels = tts.last_stats["seam_levels"]
    assert len(seams) >= 2 and len(levels) == len(seams)
    d = np.abs(np.diff(pcm))
    ratios = []
    for s, (a, b) in zip(seams, levels):
        lo, hi = max(1, s - 1200), min(len(d), s + 480 + 1200)
        local = d[lo:hi].max()                       # the speech's own slope around the seam
        seam = d[max(1, s - 2): s + 482].max()
        assert seam <= max(0.1, 1.5 * local), (s, seam, local)
        # The two decodes of the SAME 20 ms (previous chunk's end, decoded with lookahead, vs
        # the next chunk's re-decoded context) should agree in level. The vocoder is a fresh
        # noise draw per call, so single seams vary (±6 dB is routine, more on a transient);
        # the lookahead is judged on the median, each seam only against a gross dip.
        if max(a, b) > 0.003:
            r = (a + 1e-4) / (b + 1e-4)
            ratios.append(r)
            assert 0.05 <= r <= 20.0, (s, a, b)
    assert ratios and 0.5 <= float(np.median(ratios)) <= 2.0, ratios
    print(f"\nseams at {seams}, levels {[(round(a, 4), round(b, 4)) for a, b in levels]}, "
          f"median ratio {np.median(ratios):.2f}")
    out = Path(os.environ.get("VOICE_RT_OUT", ""))     # a render to listen to (not an assertion)
    if os.environ.get("VOICE_RT_OUT") and out.is_dir():
        import wave
        with wave.open(str(out / "seams_25w.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
            w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    assert len(pcm) == (tts.last_stats["tokens"] + 3) * 960


@need_tts
def test_decode_chunk_length_arithmetic_and_lm_determinism(tts):
    eng = tts.engine
    voice = tts.store.voice("stock")
    a = list(eng.stream_tokens("Mm-hm, right.", voice, None, seed=3))
    b = list(eng.stream_tokens("Mm-hm, right.", voice, None, seed=3))
    assert a == b and len(a) > 5                      # the LM sampling is seeded
    w = eng.decode_chunk(a[:10], voice, left_context=0, prompt_s=4.0)
    assert len(w) == 10 * 960
    w2 = eng.decode_chunk(a[:10], voice, left_context=3, prompt_s=4.0, context_tail=480)
    assert len(w2) == 7 * 960 + 480
    w3 = eng.decode_chunk(a[:10], voice, left_context=0, prompt_s=4.0, keep_silence=True)
    assert len(w3) == 13 * 960
    w5 = eng.decode_chunk(a[:10], voice, left_context=0, prompt_s=4.0, right_context=3)
    assert len(w5) == 7 * 960
    # The vocoder is not deterministic (RandomNormal ops): same tokens, different samples.
    w4 = eng.decode_chunk(a[:10], voice, left_context=0, prompt_s=4.0)
    assert not np.array_equal(w, w4)


@need_tts
def test_prefix_cache_gives_the_same_tokens_and_a_short_prefill(tts):
    """The voice's LM prefix (151 positions) read once: the same seed gives the same token
    sequence with and without it, and the per-sentence prefill shrinks to the text tokens."""
    eng = tts.engine
    voice = tts.store.voice("stock")
    text = "So I was looking at the plan for the garage conversion again last night and I think the budget is tight."
    prefix = eng.prefix_cache(voice)
    assert prefix.length == voice.cond_emb.shape[1] and 10e6 < prefix.nbytes < 60e6
    full = list(eng.stream_tokens(text, voice, None, seed=11))
    pre_full = eng.last_stats["prefill_s"]
    cached = list(eng.stream_tokens(text, voice, None, seed=11, prefix=prefix))
    pre_cached = eng.last_stats["prefill_s"]
    assert cached == full and len(full) > 40 and eng.last_stats["prefix"] is True
    print(f"\nprefill: full {pre_full * 1000:.0f} ms, with the voice prefix {pre_cached * 1000:.0f} ms, {profile_line()}")
    assert pre_cached < 0.5 * pre_full
    # The pipeline uses it: prepared voice → the sentence's stats say so and the first token is early.
    tts.prepare("stock")
    list(tts.stream("Nothing on your calendar this afternoon.", "stock"))
    st = tts.last_stats
    assert st["prefix"] is True and st["first_token_ms"] < 400, st


@need_tts
def test_render_line_and_cache_tag(tts):
    line = tts.render_line("Mm-hm.", "stock", seed=1)
    assert line.dtype == np.int16 and 0.3 < len(line) / 24000 < 3.0
    assert tts.cache_tag("stock").startswith("cb-") and tts.cache_tag("stock").endswith("-stock-1")


@need_tts
def test_stop_mid_sentence_is_prompt(tts):
    stop = threading.Event()
    got = 0
    for fr in tts.stream(SENTENCE_25, "stock", stop):
        got += 1
        if got == 2:
            stop.set()
            t_stop = time.perf_counter()
    assert got < 40
    assert time.perf_counter() - t_stop < 1.5       # one decode at most after the stop


@need_tts
def test_decoder_prompt_slice_speeds_the_chunk(tts):
    eng = tts.engine
    voice = tts.store.voice("stock")
    toks = list(eng.stream_tokens("Pushing the start back a month gives you room.", voice, None, seed=1))[:25]
    def t(prompt):
        eng.decode_chunk(toks, voice, left_context=0, prompt_s=prompt)
        ts = []
        for _ in range(2):
            t0 = time.perf_counter(); eng.decode_chunk(toks, voice, left_context=0, prompt_s=prompt)
            ts.append(time.perf_counter() - t0)
        return min(ts)
    t10, t4 = t(10.0), t(4.0)
    print(f"\n25-token chunk: 10 s prompt {t10 * 1000:.0f} ms, 4 s prompt {t4 * 1000:.0f} ms, {profile_line()}")
    assert t4 < 0.75 * t10
