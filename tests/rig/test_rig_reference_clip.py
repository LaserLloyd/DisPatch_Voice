"""Rig voices made from the reference recording: a rig whose ``ready`` names a voice
format other than ``chatterbox-npz`` (Fish Audio Drama 3, dots-tts: ``dotstts-ref``)
gets the profile's ``reference.wav`` on ``POST /v1/voices/from-reference``, only with
``rig.send_reference_clip`` on. The voice is kept per profile and per format in
``profile.json`` (``rig_voices``), checked with ``HEAD`` and made again when the rig
lost it or the recording changed. Chatterbox rigs keep the conditionals path."""
from __future__ import annotations

import asyncio
import hashlib
import json

import httpx
import numpy as np
import pytest
import soundfile as sf

from dispatch_voice.host import rig
from dispatch_voice.host.engines.rig_tts import ReplyEngine
from dispatch_voice.profiles import ProfileStore

from .conftest import FakeCpuTTS, make_conn, proven, wait_for
from .fake_sf1 import GOLDEN

URL = "ws://192.0.2.44:1236/v1/realtime"
FMT = "dotstts-ref"
HEARD = "words the rig heard"


def ready_frame(fmt: str | None = None) -> dict:
    r = json.loads(json.dumps(GOLDEN["server_frames"]["ready"]))
    if fmt is not None:
        r["engines"]["tts"]["voice_format"] = fmt
    return r


def write_clip(path, hz: float = 200.0, seconds: float = 4.0) -> None:
    t = np.arange(int(24000 * seconds)) / 24000
    sf.write(path, (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32), 24000, subtype="PCM_16")


class RigHttp:
    """The rig's voice routes over an httpx MockTransport: HEAD/PUT/POST from-reference."""

    def __init__(self) -> None:
        self.seen: list[httpx.Request] = []
        self.voices: set[str] = set()
        self.refuse: tuple[int, str] | None = None     # (status, error code) for the POST

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.seen.append(req)
        path = req.url.path
        if req.method == "POST" and path == "/v1/voices/from-reference":
            if self.refuse is not None:
                status, code = self.refuse
                return httpx.Response(status, json={"error": {"message": code, "type": "x", "code": code}})
            words = req.headers.get("x-reference-transcript")
            vid = hashlib.sha256(req.content + b"|" + (words or "").encode("latin-1")).hexdigest()
            self.voices.add(vid)
            return httpx.Response(201, json={"id": f"sha256:{vid}", "clip": {"prepared": True},
                                             "transcript": words or HEARD,
                                             "transcript_source": "caller" if words else "stt"})
        if path.startswith("/v1/voices/"):
            vid = path.rsplit("/", 1)[1]
            if req.method == "HEAD":
                return httpx.Response(200 if vid in self.voices else 404)
            if req.method == "PUT":
                self.voices.add(vid)
                return httpx.Response(201, json={"id": f"sha256:{vid}"})
        return httpx.Response(404)

    def calls(self) -> list[tuple[str, str]]:
        return [(r.method, r.url.path) for r in self.seen]

    def posts(self) -> list[httpx.Request]:
        return [r for r in self.seen if r.method == "POST"]


@pytest.fixture
def profiles(monkeypatch, tmp_path):
    """A real profile store with one cloned profile ``mine`` (reference.wav, voice.npz)."""
    root = tmp_path / "voices"
    store = ProfileStore(root, None)
    d = root / "mine"
    d.mkdir()
    write_clip(d / "reference.wav")
    (d / "voice.npz").write_bytes(b"conditionals-of-mine")
    (d / "profile.json").write_text(json.dumps({
        "name": "Mine", "created": 1, "default": True,
        "consent": {"kind": "self", "statement": "This is my own voice.", "note": "", "at": 1}}))

    def profile_file(profile_id, name):
        pid = profile_id or "mine"
        if pid.startswith("builtin:"):
            return None
        p = root / pid / name
        return (pid, p) if p.is_file() else None

    monkeypatch.setattr(rig, "_profile_file", profile_file)
    monkeypatch.setattr(rig, "_store", lambda: store)
    rig._FORMATS.clear()
    rig._SHA_CACHE.clear()
    yield store, d
    rig._FORMATS.clear()


@pytest.fixture
async def sessions():
    made: list[rig.RigSession] = []

    def make(http: RigHttp, *, clip: bool = True, url: str = URL) -> rig.RigSession:
        conn = rig.RigConnection(rig.RigSettings(enabled=True, url=url, send_reference_clip=clip), "tok",
                                 version="test")
        s = rig.RigSession(conn, rig.VoiceSync(url, "tok", transport=httpx.MockTransport(http)),
                           rig.RigPowerGate(None))
        made.append(s)
        return s

    yield make
    for s in made:
        await s.conn.close()


def meta(d) -> dict:
    return json.loads((d / "profile.json").read_text())


# ------------------------------------------------------------------ the format
def test_the_voice_format_comes_from_ready_and_defaults_to_chatterbox():
    conn = rig.RigConnection(rig.RigSettings(enabled=True, url=URL), "tok")
    assert conn.voice_format() is None                         # no ready yet
    conn._on_ready(ready_frame())
    assert conn.voice_format() == rig.CHATTERBOX_FORMAT        # the golden frame does not say
    conn._on_ready(ready_frame(FMT))
    assert conn.voice_format() == FMT and conn.summary()["voice_format"] == FMT
    assert rig._FORMATS[URL] == FMT
    rig._FORMATS.clear()


async def test_a_chatterbox_rig_gets_the_conditionals_never_the_recording(profiles, sessions):
    store, d = profiles
    http = RigHttp()
    s = sessions(http)                       # the switch is ON: it changes nothing here
    s.conn._on_ready(ready_frame())
    await s.sync_voices(["mine"])
    sha = hashlib.sha256(b"conditionals-of-mine").hexdigest()
    assert http.calls() == [("HEAD", f"/v1/voices/{sha}"), ("PUT", f"/v1/voices/{sha}")]
    assert s.voice_ref("mine") == f"sha256:{sha}"
    assert "rig_voices" not in meta(d) and s.conn.voice_note is None


# ------------------------------------------------------------------ the opt-in gate
async def test_with_the_switch_off_nothing_is_sent_and_the_status_says_why(profiles, sessions):
    store, d = profiles
    http = RigHttp()
    s = sessions(http, clip=False)
    s.conn._on_ready(ready_frame(FMT))
    await s.sync_voices(["mine", "builtin:af_heart"])
    assert http.seen == []                                     # no HEAD, no PUT, no clip
    assert s.voice_ref("mine") is None and not s.has_rig_voice()
    assert s.conn.voice_note == "clip_off" and s.conn.summary()["voice_note"] == "clip_off"
    assert rig._LAST.get("voice_note") == "clip_off"          # what status() shows between chats


# ------------------------------------------------------------------ the call
async def test_the_recording_is_posted_once_without_a_transcript(profiles, sessions):
    store, d = profiles
    http = RigHttp()
    s = sessions(http)
    s.conn._on_ready(ready_frame(FMT))
    await s.sync_voices(["mine"])
    [post] = http.seen
    wav = (d / "reference.wav").read_bytes()
    assert (post.method, str(post.url)) == ("POST", "http://192.0.2.44:1236/v1/voices/from-reference")
    assert post.headers["authorization"] == "Bearer tok" and post.headers["content-type"] == "audio/wav"
    assert "x-reference-transcript" not in post.headers       # the rig transcribes it
    assert post.content == wav
    info = sf.info(d / "reference.wav")
    assert (info.subtype, info.channels) == ("PCM_16", 1)
    entry = meta(d)["rig_voices"][FMT]
    assert s.voice_ref("mine") == f"sha256:{entry['id']}" and entry["id"] in http.voices
    assert entry["transcript"] == HEARD and entry["transcript_source"] == "stt"
    assert entry["reference_sha256"] == hashlib.sha256(wav).hexdigest()
    assert "rig_voices" not in store.get("mine")              # the API's profile never shows it
    assert s.conn.voice_note is None
    await s.sync_voices(["mine"])                              # stored this chat: nothing more
    assert len(http.seen) == 1


async def test_the_profiles_own_transcript_goes_in_the_header(profiles, sessions):
    store, d = profiles
    m = meta(d)
    m["transcript"] = "  Hello there,\nthis is my voice.  "
    (d / "profile.json").write_text(json.dumps(m))
    http = RigHttp()
    s = sessions(http)
    s.conn._on_ready(ready_frame(FMT))
    await s.sync_voices(["mine"])
    [post] = http.posts()
    assert post.headers["x-reference-transcript"] == "Hello there, this is my voice."
    assert meta(d)["rig_voices"][FMT]["transcript_source"] == "caller"


def test_the_header_transcript():
    assert rig.header_transcript(None) is None and rig.header_transcript("   ") is None
    assert rig.header_transcript(" a\tb\n c ") == b"a b c"
    assert rig.header_transcript("café") == "café".encode("latin-1")
    assert rig.header_transcript("こんにちは") is None             # not Latin-1: the rig's own STT
    assert rig.header_transcript("x" * 2000) == b"x" * 2000
    assert rig.header_transcript("x" * 2001) is None             # never cut: cut words would not match


# ------------------------------------------------------------------ the cache
async def test_a_stored_voice_is_checked_with_head_not_sent_again(profiles, sessions):
    store, d = profiles
    http = RigHttp()
    first = sessions(http)
    first.conn._on_ready(ready_frame(FMT))
    await first.sync_voices(["mine"])
    vid = meta(d)["rig_voices"][FMT]["id"]
    http.seen.clear()
    nxt = sessions(http)                                       # the next voice chat
    nxt.conn._on_ready(ready_frame(FMT))
    await nxt.sync_voices(["mine"])
    assert http.calls() == [("HEAD", f"/v1/voices/{vid}")]
    assert nxt.voice_ref("mine") == f"sha256:{vid}"


async def test_a_404_makes_it_again_with_the_words_the_rig_heard(profiles, sessions):
    store, d = profiles
    http = RigHttp()
    first = sessions(http)
    first.conn._on_ready(ready_frame(FMT))
    await first.sync_voices(["mine"])
    old = meta(d)["rig_voices"][FMT]["id"]
    http.voices.clear()                                        # the rig forgot it
    http.seen.clear()
    nxt = sessions(http)
    nxt.conn._on_ready(ready_frame(FMT))
    await nxt.sync_voices(["mine"])
    assert http.calls() == [("HEAD", f"/v1/voices/{old}"), ("POST", "/v1/voices/from-reference")]
    assert http.posts()[0].headers["x-reference-transcript"] == HEARD   # no second transcription
    entry = meta(d)["rig_voices"][FMT]
    assert nxt.voice_ref("mine") == f"sha256:{entry['id']}" and entry["transcript"] == HEARD


async def test_a_changed_recording_is_sent_again_and_transcribed_again(profiles, sessions):
    store, d = profiles
    http = RigHttp()
    first = sessions(http)
    first.conn._on_ready(ready_frame(FMT))
    await first.sync_voices(["mine"])
    old = meta(d)["rig_voices"][FMT]
    write_clip(d / "reference.wav", hz=330.0, seconds=5.0)     # re-recorded
    rig._SHA_CACHE.clear()
    http.seen.clear()
    nxt = sessions(http)
    nxt.conn._on_ready(ready_frame(FMT))
    await nxt.sync_voices(["mine"])
    assert http.calls() == [("POST", "/v1/voices/from-reference")]        # the old id is not asked about
    assert "x-reference-transcript" not in http.posts()[0].headers       # its words were the old clip's
    new = meta(d)["rig_voices"][FMT]
    assert new["reference_sha256"] == hashlib.sha256((d / "reference.wav").read_bytes()).hexdigest()
    assert new["reference_sha256"] != old["reference_sha256"] and new["id"] != old["id"]


async def test_unknown_voice_posts_the_recording_again_in_the_background(profiles, sessions):
    store, d = profiles
    http = RigHttp()
    s = sessions(http)
    s.conn.loop = asyncio.get_running_loop()
    s.conn._on_ready(ready_frame(FMT))
    await s.sync_voices(["mine"])
    http.voices.clear()
    http.seen.clear()
    s.resync_threadsafe("mine")                                # the rig said unknown_voice
    assert s.voice_ref("mine") is None
    assert await wait_for(lambda: s.voice_ref("mine") is not None)
    assert http.calls() == [("POST", "/v1/voices/from-reference")]        # straight to the POST


async def test_formats_are_kept_apart_in_profile_json(profiles, sessions):
    store, d = profiles
    http = RigHttp()
    for fmt in (FMT, "qwen3tts-ref"):
        s = sessions(http)
        s.conn._on_ready(ready_frame(fmt))
        await s.sync_voices(["mine"])
    rv = meta(d)["rig_voices"]
    assert set(rv) == {FMT, "qwen3tts-ref"}
    store.set_rig_voice("mine", FMT, None)
    assert set(store.rig_voices("mine")) == {"qwen3tts-ref"}


# ------------------------------------------------------------------ refusals
@pytest.mark.parametrize("status,code,note", [(403, "reference_clip_disabled", "clip_refused"),
                                              (400, "bad_audio", "bad_clip")])
async def test_a_refused_recording_stays_here_and_is_not_sent_again(profiles, sessions, status, code, note):
    store, d = profiles
    http = RigHttp()
    http.refuse = (status, code)
    s = sessions(http)
    s.conn._on_ready(ready_frame(FMT))
    await s.sync_voices(["mine"])
    await s.sync_voices(["mine"])
    s.conn._on_ready(ready_frame(FMT))                         # a resume: still not again
    await asyncio.sleep(0.05)
    assert len(http.posts()) == 1
    assert s.voice_ref("mine") is None and s.conn.voice_note == note
    assert "rig_voices" not in meta(d)


# ------------------------------------------------------------------ the engine changing
async def test_a_rig_that_comes_up_with_another_engine_gets_the_recording(profiles, sessions):
    store, d = profiles
    http = RigHttp()
    s = sessions(http)
    await s.sync_voices(["mine"])                              # before any ready: today's guess
    sha = hashlib.sha256(b"conditionals-of-mine").hexdigest()
    assert s.voice_ref("mine") == f"sha256:{sha}"
    s.conn._on_ready(ready_frame(FMT))                         # the rig runs Drama 3
    assert s.voice_ref("mine") is None                         # never the npz on that engine
    assert await wait_for(lambda: s.voice_ref("mine") not in (None, f"sha256:{sha}"))
    assert s.formats["mine"] == FMT and http.calls()[-1] == ("POST", "/v1/voices/from-reference")


async def test_the_recording_never_goes_on_a_guess(profiles, sessions):
    store, d = profiles
    rig._FORMATS[URL] = FMT                                    # the last chat's rig ran Drama 3
    http = RigHttp()
    s = sessions(http)
    await s.sync_voices(["mine"])
    assert http.seen == [] and s.voice_ref("mine") is None     # waits for this rig's ready
    s.conn._on_ready(ready_frame(FMT))
    assert await wait_for(lambda: s.voice_ref("mine") is not None)
    assert http.calls() == [("POST", "/v1/voices/from-reference")]


async def test_a_guess_proved_wrong_by_ready_still_syncs_the_conditionals(profiles, sessions):
    rig._FORMATS[URL] = FMT
    http = RigHttp()
    s = sessions(http)
    await s.sync_voices(["mine"])
    assert http.seen == []
    s.conn._on_ready(ready_frame())                            # back on Chatterbox
    sha = hashlib.sha256(b"conditionals-of-mine").hexdigest()
    assert await wait_for(lambda: s.voice_ref("mine") == f"sha256:{sha}")
    assert ("POST", "/v1/voices/from-reference") not in http.calls()


# ------------------------------------------------------------------ over sf.1
async def test_a_drama_rig_speaks_the_voice_it_made(profiles, fake_rig):
    """End to end against the fake sf.1 rig: its ready names ``dotstts-ref``, the voice
    is made from the recording, and ``tts.open`` names the id the rig returned."""
    store, d = profiles
    fake_rig.voice_format = FMT
    http = RigHttp()
    conn = make_conn(fake_rig, send_reference_clip=True)
    s = rig.RigSession(conn, rig.VoiceSync(fake_rig.url, fake_rig.token, transport=httpx.MockTransport(http)),
                       rig.RigPowerGate(None))
    conn.start()
    try:
        task = asyncio.ensure_future(s.sync_voices(["mine"]))   # as ws.serve does, beside the open
        assert await proven(conn)
        assert await wait_for(lambda: s.voice_ref("mine") is not None)
        await task
        vid = meta(d)["rig_voices"][FMT]["id"]
        assert s.voice_ref("mine") == f"sha256:{vid}"
        fake_rig.voices = {f"sha256:{vid}"}
        cpu = FakeCpuTTS()
        eng = ReplyEngine(cpu, s)
        frames = await asyncio.to_thread(
            lambda: list(eng.stream_reply("Hello there.", "mine", None, gen=0, seq=1, rid="r1")))
        assert [m["voice"] for m in fake_rig.of("tts.open")] == [f"sha256:{vid}"]
        assert frames and all((f == fake_rig.tts_value).all() for f in frames if len(f)) and cpu.said == []
        assert fake_rig.errors == []
    finally:
        await conn.close()
