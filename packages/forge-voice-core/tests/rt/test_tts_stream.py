"""PipelinedChatterbox with a fake engine: chunk schedule, seams, frames, stop, two sessions,
cache_tag/render_line/apply_settings/warm/release; prompt slicing; the crossfade helper."""
import threading
import time

import numpy as np
import pytest

from forge_voice_core.audio import crossfade
from forge_voice_core.engine import SAMPLES_PER_TOKEN, ChatterboxOnnx
from forge_voice_core.rt import RtSettings
from forge_voice_core.rt.tts_stream import (
    CROSSFADE_SAMPLES,
    FRAME_SAMPLES,
    PipelinedChatterbox,
    _Plan,
    frames_of,
    to_int16,
)
from rt.conftest import FakeChatterbox, PrefixedFakeChatterbox


def make(store, engine, **rt) -> PipelinedChatterbox:
    return PipelinedChatterbox(store, engine=engine, rt=RtSettings(**rt))


def collect(tts, text="Pushing the start back a month gives you room for the electrical work.", stop=None):
    return list(tts.stream(text, "stock", stop))


def test_plan_schedule():
    p = _Plan(first=25, chunk=25, left=8, lookahead=3)
    assert p.next_chunk(25, False) is None            # the 3 lookahead tokens are not in yet
    assert p.next_chunk(28, False) == (0, 28, 0, 3)   # emits 25, decodes 28
    assert p.next_chunk(45, False) is None            # 20 available, < 25 + 3, LM still running
    assert p.next_chunk(45, True) == (17, 45, 8, 0)   # the remainder once the LM is done
    assert p.next_chunk(45, True) is None
    p = _Plan(first=15, chunk=25, left=8, lookahead=3)
    assert p.next_chunk(60, True) == (0, 18, 0, 3)
    assert p.next_chunk(60, True) == (7, 43, 8, 3)
    assert p.next_chunk(60, True) == (32, 60, 8, 0)   # 20 left < 28: last chunk, no lookahead
    p = _Plan(first=25, chunk=25, left=8, lookahead=0)
    assert p.next_chunk(25, False) == (0, 25, 0, 0)


def test_frames_and_samples_per_token():
    assert SAMPLES_PER_TOKEN == 960 and FRAME_SAMPLES == 960
    out = list(frames_of(np.zeros(2000, "<i2")))
    assert [len(f) for f in out] == [960, 960, 80]


def test_chunk_schedule_and_seams_are_continuous(fake_store):
    eng = FakeChatterbox(n_tokens=60)
    tts = make(fake_store, eng)
    frames = collect(tts)
    pcm = np.concatenate(frames)
    calls = eng.decode_calls
    assert [c["tokens"][0] - 100 for c in calls] == [0, 17, 42]
    assert [len(c["tokens"]) for c in calls] == [28, 36, 18]      # 25+3 | 8+25+3 | 8+10 (remainder)
    assert [c["left"] for c in calls] == [0, 8, 8]
    assert [c["right"] for c in calls] == [3, 3, 0]
    assert [c["context_tail"] for c in calls] == [0, CROSSFADE_SAMPLES, CROSSFADE_SAMPLES]
    assert [c["keep_silence"] for c in calls] == [False, False, True]
    assert all(c["prompt_tokens"] == 100 for c in calls)        # 4 s prompt = 100 tokens
    # 60 tokens + 3 silence tokens kept on the last chunk, nothing lost or doubled at the seams.
    assert len(pcm) == (60 + 3) * SAMPLES_PER_TOKEN
    # The fake decoder is a phase-continuous tone: a correct seam leaves no jump beyond
    # the tone's own slope (0.058/sample) times the √2 an equal-power fade of two
    # identical pieces peaks at; a butted or misaligned seam would show ≈ 0.5–1.0.
    x = pcm.astype(np.float32) / 32767.0
    d = np.abs(np.diff(x))
    assert d.max() < 0.058 * 1.4143 + 0.002
    assert d[: 24 * 960].max() < 0.058 + 0.001          # inside the first chunk: the bare slope
    assert tts.last_stats["seams"] == [25 * 960 - CROSSFADE_SAMPLES, 50 * 960 - CROSSFADE_SAMPLES]
    assert len(tts.last_stats["chunks"]) == 3 and tts.last_stats["tokens"] == 60
    assert all(len(f) == FRAME_SAMPLES for f in frames[:-1]) and frames[0].dtype == np.int16


def test_first_audio_arrives_before_the_lm_finishes(fake_store):
    eng = FakeChatterbox(n_tokens=100, token_delay_s=0.004, decode_s=0.02)
    tts = make(fake_store, eng)
    t0 = time.perf_counter()
    gen = tts.stream("x", "stock")
    first = next(gen)
    t_first = time.perf_counter() - t0
    rest = list(gen)
    total = time.perf_counter() - t0
    assert len(first) == FRAME_SAMPLES
    assert t_first < 0.25 and total >= 0.4          # 25 tokens × 4 ms + decode ≪ 100 × 4 ms
    assert tts.last_stats["first_chunk_ms"] < 250
    assert len(np.concatenate([first, *rest])) == (100 + 3) * SAMPLES_PER_TOKEN


def test_two_sessions_alternate_and_use_the_short_first_chunk(fake_store):
    eng = FakeChatterbox(n_tokens=70, decoder_sessions=2, decode_s=0.03)
    tts = make(fake_store, eng, decoder_sessions=2, first_chunk_tokens=15)
    pcm = np.concatenate(collect(tts))
    # Two decoder threads record their calls in whichever order they start, so the schedule
    # is checked per chunk (keyed on the first token), not in recording order.
    calls = sorted(eng.decode_calls, key=lambda c: c["tokens"][0])
    assert [len(c["tokens"]) for c in calls] == [18, 36, 36, 13]
    assert [c["session"] for c in calls] == [0, 1, 0, 1]
    assert [c["tokens"][0] - 100 for c in calls] == [0, 7, 32, 57]
    assert len(pcm) == (70 + 3) * SAMPLES_PER_TOKEN
    x = pcm.astype(np.float32) / 32767.0
    assert np.abs(np.diff(x)).max() < 0.058 * 1.4143 + 0.002
    assert eng.sessions_set == []                     # the fake already had 2 sessions


def test_engine_sessions_follow_settings(fake_store):
    eng = FakeChatterbox(n_tokens=30, decoder_sessions=1)
    tts = make(fake_store, eng, decoder_sessions=2)
    collect(tts)
    assert eng.sessions_set == [(2, 5)]               # 5 threads each with two sessions
    tts.apply_settings(RtSettings(decoder_sessions=1))
    assert eng.sessions_set[-1] == (1, 6)


def test_stop_cuts_mid_sentence(fake_store):
    eng = FakeChatterbox(n_tokens=200, token_delay_s=0.003, decode_s=0.01)
    tts = make(fake_store, eng)
    stop = threading.Event()
    got = []
    for fr in tts.stream("x", "stock", stop):
        got.append(fr)
        if len(got) == 3:
            stop.set()
    assert 3 <= len(got) <= 30
    assert tts.last_stats["tokens"] < 200             # the LM stopped early too


def test_synth_concatenates_and_render_line_uses_the_full_prompt_with_the_seed(fake_store):
    eng = FakeChatterbox(n_tokens=30)
    tts = make(fake_store, eng)
    pcm, rate = tts.synth("Hello there.", "stock")
    assert rate == 24000 and len(pcm) == 33 * SAMPLES_PER_TOKEN and pcm.dtype == np.int16
    line = tts.render_line("Mm-hm.", "stock", seed=7)
    assert line.dtype == np.int16 and len(line) == 2400 and eng.synth_calls == [("Mm-hm.", 7)]


def test_cache_tag_format_and_unknown_voice(fake_store):
    tts = make(fake_store, FakeChatterbox())
    assert tts.cache_tag("stock") == "cb-deadbeef-stock-1700000000"
    assert tts.cache_tag(None) == "cb-deadbeef-stock-1700000000"     # the store default
    assert tts.cache_tag("nope") is None


def test_unknown_profile_raises_before_any_audio(fake_store):
    tts = make(fake_store, FakeChatterbox())
    with pytest.raises(LookupError):
        next(tts.stream("hi", "nope"))
    calls = []
    tts2 = PipelinedChatterbox(fake_store, engine=FakeChatterbox(), resolve=lambda pid: calls.append(pid) or "stock")
    collect(tts2, "hi")
    assert calls == ["stock"]


def test_apply_settings_drops_the_sliced_voice_cache_and_accepts_mappings(fake_store):
    eng = FakeChatterbox(n_tokens=30)
    tts = make(fake_store, eng)
    collect(tts)
    assert eng.decode_calls[0]["prompt_tokens"] == 100
    tts.apply_settings(type("S", (), {"rt": {"decoder_prompt_s": 3}})())
    assert tts.rt.decoder_prompt_s == 3.0
    eng.decode_calls.clear()
    collect(tts)
    assert eng.decode_calls[0]["prompt_tokens"] == 75
    tts.apply_settings(object())                       # nothing usable: ignored
    assert tts.rt.decoder_prompt_s == 3.0


def test_warm_release_and_lazy_factory(fake_store):
    made = []

    def factory():
        made.append(1)
        return FakeChatterbox(n_tokens=30)

    tts = PipelinedChatterbox(fake_store, engine=factory)
    assert made == []
    tts.warm()
    assert made == [1]
    collect(tts)
    assert made == [1]
    tts.release()
    collect(tts)
    assert made == [1, 1]


def test_lm_error_surfaces(fake_store):
    class Boom(FakeChatterbox):
        def stream_tokens(self, *a, **k):
            raise RuntimeError("lm died")
            yield 0

    tts = make(fake_store, Boom())
    with pytest.raises(RuntimeError):
        collect(tts)


# ---- the LM prefix cache ------------------------------------------------------ #

def test_prefix_is_built_once_per_voice_and_passed_to_every_sentence(fake_store):
    fake_store.add("narrator")
    eng = PrefixedFakeChatterbox(n_tokens=30)
    tts = make(fake_store, eng)
    collect(tts)
    collect(tts)
    list(tts.stream("x", "narrator"))
    assert eng.prefix_calls == [151, 151]             # once for stock, once for the narrator voice
    assert eng.prefix_used == [True, True, True]
    assert tts.last_stats["prefix"] is True
    assert tts.last_stats["first_token_ms"] is not None and tts.last_stats["tokens_ready_ms"] is not None
    tts.release()                                      # voice off: prefixes go with the engine
    collect(tts)
    assert eng.prefix_calls == [151, 151, 151]


def test_prepare_does_the_one_off_work_ahead_and_warm_prepares_the_default(fake_store):
    eng = PrefixedFakeChatterbox(n_tokens=30)
    tts = make(fake_store, eng)
    assert tts.prepare("stock") == "stock"
    assert eng.prefix_calls == [151] and ("stock", 4.0) in tts._voices
    collect(tts)
    assert eng.prefix_calls == [151]                   # nothing rebuilt
    with pytest.raises(LookupError):
        tts.prepare("nope")
    eng2 = PrefixedFakeChatterbox(n_tokens=30)
    tts2 = make(fake_store, eng2)
    tts2.warm()
    assert eng2.prefix_calls == [151]


def test_engines_without_a_prefix_cache_still_work(fake_store):
    eng = FakeChatterbox(n_tokens=30)
    tts = make(fake_store, eng)
    assert tts.prepare("stock") == "stock"
    collect(tts)
    assert tts.last_stats["prefix"] is False and tts.last_stats["tokens"] == 30


# ---- engine helpers (no model) ---------------------------------------------- #

def test_slice_prompt(fake_store):
    v = fake_store.voice("stock")
    s = ChatterboxOnnx.slice_prompt(v, 4.0)
    assert s.prompt_token.shape == (1, 100) and s.speaker_features.shape == (1, 200, 80)
    assert s.cond_emb is v.cond_emb and s.speaker_embeddings is v.speaker_embeddings
    assert ChatterboxOnnx.slice_prompt(v, None) is v
    assert ChatterboxOnnx.slice_prompt(v, 10.0) is v and ChatterboxOnnx.slice_prompt(v, 12.0) is v
    assert ChatterboxOnnx.slice_prompt(v, 0).prompt_token.shape == (1, 250)


def test_crossfade_is_equal_power():
    n = 480
    a = np.ones(n, np.float32)
    b = np.ones(n, np.float32)
    m = crossfade(a, b)
    assert m.shape == (n,) and m.dtype == np.float32
    assert m[0] == pytest.approx(1.0, abs=1e-5) and m[-1] == pytest.approx(1.0, abs=1e-5)
    assert np.all(m >= 0.99) and m.max() <= 1.4143      # cos+sin peaks at √2 for correlated inputs
    assert crossfade(a[:0], b).size == 0
    assert len(crossfade(a[:100], b)) == 100
    assert to_int16(np.array([2.0, -2.0, 0.5])).tolist() == [32767, -32767, 16383]


# ---- review T4 #2: a live decoder_sessions change never breaks a decode in flight ------ #

class SlowSess:
    """A fake ORT session: ``run`` takes ``delay`` and returns 960 samples per token."""

    def __init__(self, delay: float, tag: str) -> None:
        self.delay, self.tag = delay, tag

    def run(self, _outs, feeds):
        time.sleep(self.delay)
        n_tok = feeds["speech_tokens"].shape[1] - feeds["speaker_features"].shape[1] // 2
        return [np.zeros((1, max(1, n_tok) * SAMPLES_PER_TOKEN), np.float32)]


def bare_engine(build_s: float = 0.2, run_s: float = 0.3):
    """A ChatterboxOnnx without models: only the decoder-session surface is live."""
    eng = ChatterboxOnnx.__new__(ChatterboxOnnx)
    eng.decoder_threads = 5
    eng._dec, eng._dec_locks, eng._s = [], [], {}
    eng._dec_swap, eng._dec_build = threading.Lock(), threading.Lock()
    built = []

    def session(component, threads):
        time.sleep(build_s)
        built.append((component, threads))
        return SlowSess(run_s, f"{component}:{threads}:{len(built)}")

    eng._session = session
    return eng, built


def test_set_decoder_sessions_builds_before_it_swaps_and_keeps_in_flight_decodes(fake_store):
    eng, built = bare_engine()
    eng.set_decoder_sessions(2)
    assert eng.decoder_sessions == 2 and len(built) == 2 and eng._s["conditional_decoder"] is eng._dec[0]
    voice = fake_store.voice("stock")
    errors, outs = [], []

    def decode(session):
        try:
            outs.append(len(eng.decode_chunk(list(range(25)), voice, left_context=0, prompt_s=None, session=session)))
        except Exception as e:  # noqa: BLE001 - the probe records whatever happened
            errors.append(f"session {session}: {type(e).__name__}: {e}")

    t1 = threading.Thread(target=decode, args=(1,))
    t1.start()
    time.sleep(0.05)
    tr = threading.Thread(target=eng.set_decoder_sessions, args=(1, 6))   # 2 → 1 with a thread change
    tr.start()
    time.sleep(0.05)
    t2 = threading.Thread(target=decode, args=(0,))                        # a chunk during the rebuild
    t2.start()
    for t in (t1, t2, tr):
        t.join(5.0)
    assert errors == [] and outs == [25 * SAMPLES_PER_TOKEN] * 2
    assert eng.decoder_sessions == 1 and eng.decoder_threads == 6 and built[-1] == ("conditional_decoder", 6)
    # Same thread count: session 0 is kept, only the second is built (and dropped).
    n = len(built)
    eng.set_decoder_sessions(2, 6)
    assert eng.decoder_sessions == 2 and len(built) == n + 1 and eng._dec[0].tag.endswith(f":{n}")
    eng.set_decoder_sessions(1, 6)
    assert eng.decoder_sessions == 1 and len(built) == n + 1


def test_apply_settings_never_blocks_on_a_session_change(fake_store):
    class SlowSessions(FakeChatterbox):
        def set_decoder_sessions(self, n, threads=None):
            time.sleep(0.3)
            super().set_decoder_sessions(n, threads)

    eng = SlowSessions(n_tokens=30, decoder_sessions=1)
    tts = make(fake_store, eng)
    collect(tts)
    t0 = time.perf_counter()
    tts.apply_settings(RtSettings(decoder_sessions=2))
    assert time.perf_counter() - t0 < 0.1                 # the route's thread is not held
    assert eng.decoder_sessions == 1                      # … the change lands on the worker
    frames = collect(tts)                                 # a sentence meanwhile still works
    assert len(frames) > 0
    assert tts.sessions_settled(2.0) and eng.decoder_sessions == 2 and eng.sessions_set[-1] == (2, 5)
    tts.apply_settings(RtSettings(decoder_sessions=2))    # unchanged: no worker
    assert tts.sessions_settled(0.1)


def test_engine_init_creates_the_decoder_locks_before_the_first_session_call():
    """``ChatterboxOnnx.__init__`` calls ``set_decoder_sessions`` while building; the two
    locks it needs must exist by then (caught by the model suite once — keep it caught here)."""
    import ast
    import inspect
    src = inspect.getsource(ChatterboxOnnx.__init__)
    tree = ast.parse("if True:\n" + src)
    order = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr in ("_dec_swap", "_dec_build"):
                    order.append((node.lineno, t.attr))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "set_decoder_sessions":
            order.append((node.lineno, "call"))
    order.sort()
    assert [k for _, k in order] == ["_dec_swap", "_dec_build", "call"], order
