"""The provider interface (host/providers.py): ONE contract test run against every adapter -
the CPU pipeline (STT/TTS/VAD), the OpenAI-compatible endpoints and the SpeechForge rig - with
the same fake input streams, plus discovery and the per-adapter details."""
from __future__ import annotations

import asyncio
import json
import threading

import httpx
import numpy as np
import pytest

from dispatch_voice.host import providers as P
from dispatch_voice.host import registry, rig
from dispatch_voice.host import settings_schema as S
from dispatch_voice.host.vad_turn import EnergyVAD

from .conftest import FakeCpuTTS, make_conn, proven

RATE = 16000
TEXT = "hello world"
SENTENCE = "Two meetings today."


# --------------------------------------------------------------------------- #
# the shared fake streams
# --------------------------------------------------------------------------- #

def tone(seconds: float = 1.2, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (np.sin(2 * np.pi * 440 * t) * amp * 32767).astype("<i2")


async def audio_stream(*, as_numpy: bool = False, seconds: float = 1.2):
    """The same utterance for every STT adapter: 20 ms chunks."""
    pcm = tone(seconds)
    for i in range(0, len(pcm), 320):
        chunk = pcm[i:i + 320]
        yield chunk if as_numpy else chunk.tobytes()
        await asyncio.sleep(0)


class FakeStreaming:
    """A ``StreamingSTT`` double (push / tick / final / reset) for the CPU adapter."""

    def __init__(self) -> None:
        self.pushed = 0
        self.ticks = 0
        self.resets = 0
        self.lock = threading.Lock()

    def push(self, pcm16, *, vad_prob):
        with self.lock:
            assert len(pcm16) == 512 and 0.0 <= vad_prob <= 1.0
            self.pushed += 1

    def tick(self):
        from dispatch_voice.rt.stt_stream import Partial
        with self.lock:
            self.ticks += 1
            return Partial("", "hello", 1, False, self.pushed * 0.032, 1) if self.ticks == 2 else None

    def final(self, *, timeout_s=None):
        return TEXT

    def reset(self):
        self.resets += 1


class FakeEngine:
    """A ``TTSEngine`` double: three 40 ms frames per sentence."""
    sample_rate = 24000
    streaming = True

    def stream(self, text, profile_id=None, stop=None):
        for _ in range(3):
            if stop is not None and stop.is_set():
                return
            yield np.full(960, 11, "<i2")


def openai_transport(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json={"text": TEXT})
        if request.url.path.endswith("/audio/speech"):
            return httpx.Response(200, content=np.full(2400, 5, "<i2").tobytes())
        return httpx.Response(404)
    return httpx.MockTransport(handler)


# --------------------------------------------------------------------------- #
# the adapters, built the way production builds them
# --------------------------------------------------------------------------- #

@pytest.fixture(params=["cpu", "openai", "rig"])
async def stt_provider(request, fake_rig):
    fake_rig.final_text = TEXT
    if request.param == "cpu":
        yield P.StreamingSTTProvider(FakeStreaming, vad=EnergyVAD())
    elif request.param == "openai":
        yield P.OpenAISTTProvider("http://10.0.0.5:8000", "whisper-1", api_key="k", transport=openai_transport([]),  # scrub-ok: address-range fixture
                                  allow=())
    else:
        conn = make_conn(fake_rig)
        conn.start()
        assert await proven(conn)
        link = P.RigLink(None, conn=conn)
        yield P.RigSTTProvider(link, FakeStreaming, vad=EnergyVAD())
        await conn.close()


@pytest.fixture(params=["cpu", "openai", "rig"])
async def tts_provider(request, fake_rig):
    if request.param == "cpu":
        yield P.EngineTTSProvider(FakeEngine())
    elif request.param == "openai":
        yield P.openai_tts_provider("http://10.0.0.5:8000", "tts-1", transport=openai_transport([]))  # scrub-ok: address-range fixture
    else:
        conn = make_conn(fake_rig)
        conn.start()
        assert await proven(conn)
        sess = rig.RigSession(conn, rig.VoiceSync(fake_rig.url, fake_rig.token), rig.RigPowerGate(None))
        sess.voices[None] = "sha256:" + "ab" * 32
        yield P.RigTTSProvider(P.RigLink(None, session=sess), FakeCpuTTS())
        await conn.close()


# --------------------------------------------------------------------------- #
# the contract
# --------------------------------------------------------------------------- #

def check_common(p: P.Provider, kind: str) -> None:
    assert p.kind == kind and p.id and p.name
    h = p.health()
    assert isinstance(h, P.Health) and h.state in P.STATES and (h.ok or (h.reason and h.fix))
    caps = p.capabilities()
    assert caps["kind"] == kind and isinstance(caps["remote"], bool)
    rows = p.settings_schema()
    assert rows and all({"id", "type", "scope"} <= set(r) and "value" not in r for r in rows)
    known = {s.id for s in S.SETTINGS}
    assert {r["id"] for r in rows} <= known
    d = p.describe()
    assert d["id"] == p.id and d["health"]["state"] == h.state
    json.dumps(d)                                           # everything the status page shows is JSON


@pytest.mark.parametrize("as_numpy", [False, True])
async def test_every_stt_provider_yields_partials_then_exactly_one_final(stt_provider, as_numpy):
    check_common(stt_provider, "stt")
    assert stt_provider.capabilities()["sample_rate"] == RATE
    events = [e async for e in stt_provider.transcribe(audio_stream(as_numpy=as_numpy))]
    assert events, "no events"
    *partials, last = events
    assert isinstance(last, P.Final) and last.text == TEXT and last.final is True
    assert all(isinstance(e, P.Partial) and isinstance(e.text, str) and e.final is False for e in partials)
    assert sum(isinstance(e, P.Final) for e in events) == 1
    assert last.t_audio == pytest.approx(1.2, abs=0.05)


async def test_a_second_utterance_on_the_same_provider_is_independent(stt_provider):
    a = [e async for e in stt_provider.transcribe(audio_stream())]
    b = [e async for e in stt_provider.transcribe(audio_stream(seconds=0.6))]
    assert a[-1].text == b[-1].text == TEXT
    assert b[-1].t_audio < a[-1].t_audio


async def test_every_tts_provider_streams_pcm16_mono_and_stops_when_the_consumer_does(tts_provider):
    check_common(tts_provider, "tts")
    caps = tts_provider.capabilities()
    assert caps["format"] == "pcm_s16le" and caps["channels"] == 1 and caps["sample_rate"] == 24000
    chunks = [c async for c in tts_provider.synthesize(SENTENCE, None)]
    assert chunks and all(isinstance(c, bytes) and len(c) % 2 == 0 and len(c) > 0 for c in chunks)
    pcm = np.frombuffer(b"".join(chunks), "<i2")
    assert pcm.size >= 960 and np.abs(pcm).max() > 0
    # abandoning the stream early must neither hang nor leak a worker
    gen = tts_provider.synthesize(SENTENCE, None)
    first = await gen.__anext__()
    assert isinstance(first, bytes)
    await asyncio.wait_for(gen.aclose(), 5)


# --------------------------------------------------------------------------- #
# per-adapter details
# --------------------------------------------------------------------------- #

async def test_cpu_stt_feeds_whole_frames_and_resets_the_recogniser():
    made: list[FakeStreaming] = []

    def factory():
        made.append(FakeStreaming())
        return made[-1]

    p = P.StreamingSTTProvider(factory, vad=EnergyVAD())
    events = [e async for e in p.transcribe(audio_stream(seconds=1.0))]
    sstt = made[0]
    assert sstt.pushed == 32 and sstt.resets == 1            # 16000 samples = 31 full frames + the padded tail
    parts = [e for e in events if isinstance(e, P.Partial)]
    assert parts and parts[0].text == "hello" and parts[0].stable_words == 1


async def test_openai_stt_posts_a_wav_with_the_model_and_the_key():
    seen: list = []
    p = P.OpenAISTTProvider("http://10.0.0.5:8000/v1", "whisper-1", api_key="sekret", transport=openai_transport(seen))  # scrub-ok: address-range fixture
    events = [e async for e in p.transcribe(audio_stream())]
    assert [type(e) for e in events] == [P.Final] and events[0].text == TEXT
    req = seen[0]
    assert req.method == "POST" and req.url.path == "/v1/audio/transcriptions"      # a base ending in /v1 is not doubled
    assert req.headers["authorization"] == "Bearer sekret"
    body = req.read()
    assert b'name="model"' in body and b"whisper-1" in body and b"RIFF" in body and b"speech.wav" in body
    assert b"sekret" not in body


async def test_openai_stt_errors_surface_and_mark_the_provider_down():
    p = P.OpenAISTTProvider("http://10.0.0.5:8000", "m", transport=httpx.MockTransport(lambda r: httpx.Response(500)))  # scrub-ok: address-range fixture
    with pytest.raises(httpx.HTTPStatusError):
        [e async for e in p.transcribe(audio_stream())]
    h = p.health()
    assert h.state == "down" and h.fix and h.code == "stt_http_error"


async def test_openai_stt_silence_is_an_empty_final_without_a_request():
    seen: list = []
    p = P.OpenAISTTProvider("http://10.0.0.5:8000", "m", transport=openai_transport(seen))  # scrub-ok: address-range fixture

    async def nothing():
        yield b"\x00\x00" * 100

    assert [e async for e in p.transcribe(nothing())] == [P.Final("", 100 / RATE)] and seen == []


def test_openai_stt_health_unconfigured_blocked_and_warned():
    assert P.OpenAISTTProvider("").health().state == "unconfigured"
    h = P.OpenAISTTProvider("http://169.254.169.254/").health()
    assert h.state == "down" and h.code == "url_blocked_metadata" and h.fix
    h = P.OpenAISTTProvider("http://stt.example.com/").health()
    assert h.state == "down" and h.code == "url_not_allowed"
    assert P.OpenAISTTProvider("http://stt.example.com/", allow=("stt.example.com",)).health().state == "degraded"
    assert P.OpenAISTTProvider("https://10.0.0.5/").health().state == "ok"  # scrub-ok: address-range fixture


def test_the_openai_engine_fits_the_session_seam():
    p = P.OpenAISTTProvider("http://10.0.0.5:8000", "m", transport=openai_transport([]))  # scrub-ok: address-range fixture
    eng = P.OpenAISTTEngine(p)
    assert eng.transcribe(np.zeros(100, np.float32)) == ""
    assert eng.transcribe(np.ones(3200, np.float32) * 0.1) == TEXT


async def test_openai_tts_sends_the_hq_request_and_reports_failures():
    seen: list = []
    p = P.openai_tts_provider("http://10.0.0.5:8000", "tts-1", api_key="k2", transport=openai_transport(seen))  # scrub-ok: address-range fixture
    pcm = b"".join([c async for c in p.synthesize(SENTENCE, "alloy")])
    assert len(pcm) == 4800
    req = seen[0]
    assert req.url.path == "/v1/audio/speech" and req.headers["authorization"] == "Bearer k2"
    body = json.loads(req.read())
    assert body["input"] == SENTENCE and body["voice"] == "alloy" and body["response_format"] == "pcm"
    bad = P.openai_tts_provider("http://10.0.0.5:8000", "tts-1",  # scrub-ok: address-range fixture
                                transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(httpx.HTTPStatusError):
        [c async for c in bad.synthesize(SENTENCE)]
    assert bad.health().state == "down" and bad.health().code == "tts_failed"
    assert P.openai_tts_provider("").health().state == "unconfigured"
    assert P.openai_tts_provider("http://169.254.169.254/").health().code == "url_blocked_metadata"


async def test_the_rig_providers_share_one_connection_and_fall_back_through_the_cpu(fake_rig):
    fake_rig.final_text = TEXT
    conn = make_conn(fake_rig)
    conn.start()
    assert await proven(conn)
    try:
        link = P.RigLink(None, conn=conn)
        stt = P.RigSTTProvider(link, FakeStreaming, vad=EnergyVAD())
        assert link.conn is conn and stt.link is link
        assert link.health().ok is False or link.health().state in ("ok", "degraded", "unconfigured")
        got = [e async for e in stt.transcribe(audio_stream())]
        assert got[-1].text == TEXT
        assert fake_rig.mic_samples() > 0                      # the audio went to the rig, not the CPU recogniser
        sess = rig.RigSession(conn, rig.VoiceSync(fake_rig.url, fake_rig.token), rig.RigPowerGate(None))
        sess.voices[None] = "sha256:" + "ab" * 32
        cpu = FakeCpuTTS()
        tts = P.RigTTSProvider(P.RigLink(None, session=sess), cpu)
        chunks = [c async for c in tts.synthesize(SENTENCE)]
        assert np.frombuffer(b"".join(chunks), "<i2")[0] == fake_rig.tts_value and cpu.said == []
        assert tts.capabilities()["remote"] is True and tts.capabilities()["cloning"] is True
    finally:
        await conn.close()


def test_the_rig_link_health_names_what_is_wrong(monkeypatch, tmp_path):
    monkeypatch.delenv(rig.TOKEN_ENV, raising=False)
    monkeypatch.setattr(rig, "token_path", lambda: tmp_path / "none.env")
    from dispatch_voice.host.config import VoiceSettings
    h = P.RigLink(VoiceSettings()).health()
    assert h.state == "unconfigured" and h.code == "rig_url_missing" and h.fix
    url = "ws://10.0.0.7:1236/v1/realtime"  # scrub-ok: address-range fixture
    s = VoiceSettings(rig=rig.RigSettings(enabled=True, url=url))
    h = P.RigLink(s).health()
    assert h.state == "unconfigured" and h.code == "rig_token_missing"
    h = P.RigLink(s, token="t").health()
    assert h.state == "degraded" and h.code == "" or h.ok            # configured, plain ws:// to a LAN address
    bad = VoiceSettings(rig=rig.RigSettings(enabled=True, url="ws://169.254.169.254/x"))
    h = P.RigLink(bad, token="t").health()
    assert h.state == "down" and h.code == "url_blocked_metadata"


async def test_cpu_vad_provider_reports_speech_start_and_the_end_of_turn():
    from dispatch_voice.host.vad_turn import Endpointer, EndpointerConfig, TurnDetector

    def build():
        cfg = EndpointerConfig(min_speech_s=0.1, turn_probe_silence_s=0.1, turn_hard_silence_s=0.3,
                               no_smart_turn_silence_s=0.2)
        return TurnDetector(EnergyVAD(), Endpointer(cfg, None))

    p = P.CpuVADProvider(factory=build)
    check_common(p, "vad")
    assert p.capabilities()["frame_samples"] == 512

    async def speech_then_silence():
        pcm = np.concatenate([tone(0.8), np.zeros(int(1.5 * RATE), "<i2")])
        for i in range(0, len(pcm), 640):
            yield pcm[i:i + 640].tobytes()

    kinds = [e.kind async for e in p.detect(speech_then_silence())]
    assert kinds[0] == "speech_start" and "turn_end" in kinds


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #

class FakeEP:
    def __init__(self, name, fn, value="pkg.mod:setup"):
        self.name, self.value, self._fn = name, value, fn

    def load(self):
        return self._fn


def test_entry_point_discovery_lists_without_importing_and_loads_only_the_enabled():
    ran: list[str] = []

    def boom(api):
        raise RuntimeError("broken plugin")

    eps = [FakeEP("kokoro", lambda api: ran.append("kokoro")), FakeEP("other", lambda api: ran.append("other")),
           FakeEP("broken", boom)]
    listed = P.discover_entry_points({"kokoro"}, eps=eps)
    assert listed == [{"name": "kokoro", "target": "pkg.mod:setup", "enabled": True},
                      {"name": "other", "target": "pkg.mod:setup", "enabled": False},
                      {"name": "broken", "target": "pkg.mod:setup", "enabled": False}] and ran == []
    loaded = P.load_entry_points(registry.ExtensionAPI(None), {"kokoro", "broken"}, eps=eps)
    assert loaded == ["kokoro"] and ran == ["kokoro"]            # the broken one is logged and skipped


def test_register_provider_through_the_extension_api_and_the_catalog():
    reg = P.ProviderRegistry()
    reg.register("tts", "demo", "Demo voices", lambda s: P.EngineTTSProvider(FakeEngine(), id="demo", name="Demo voices"))
    reg.register("stt", "bad", "Bad", lambda s: (_ for _ in ()).throw(RuntimeError("no")))
    cat = {(d["kind"], d["id"]): d for d in reg.catalog()}
    assert cat[("tts", "demo")]["health"]["state"] == "ok" and cat[("tts", "demo")]["origin"] == "builtin"
    assert cat[("stt", "bad")]["health"]["state"] == "down"
    with pytest.raises(ValueError):
        reg.register("music", "x", "X", lambda s: None)
    with pytest.raises(LookupError):
        reg.build("tts", "nope")
    api = registry.ExtensionAPI(None)
    before = set(P.REGISTRY.ids("tts"))
    try:
        api.register_provider("tts", "plugin-demo", "Plugin demo", lambda s: P.EngineTTSProvider(FakeEngine()))
        assert "plugin-demo" in P.REGISTRY.ids("tts")
    finally:
        P.REGISTRY._factories.pop(("tts", "plugin-demo"), None)
    assert set(P.REGISTRY.ids("tts")) == before
