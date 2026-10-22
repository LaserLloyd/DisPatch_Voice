"""Per-profile cloned voices against a REAL DisPatch 4.0 (the lead runs it at INT-2):

    cd <dispatch-chat worktree>/backend
    DISPATCH_BACKEND=$PWD uv run --offline --with-editable <this repo> pytest -o testpaths= \\
        --rootdir <this repo> <this repo>/tests/contract

Profile A makes a personal clone; profile B must not list it, fetch it, hear it, download its
recordings or speak with it. Written to the WP-CORE semantics (``ctx.caller`` is real; the
bots/voices filters are WP-CORE's), so against the Foundation stubs the cross-profile checks
are skipped with the reason instead of failing. No model is loaded; a clone is a profile
directory written under the data dir (what ``ProfileStore.create`` leaves behind).
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

from .test_dispatch_contract import BACKEND, dispatch  # noqa: F401  (the module-scoped fixture)

pytestmark = pytest.mark.contract


class _FakeEngine:
    sample_rate = 24000
    dtype = "fake"
    watermarking = False


def _clone(root, pid: str, owner: str | None, name: str) -> None:
    import soundfile as sf
    d = root / pid
    (d / "fillers").mkdir(parents=True)
    meta = {"name": name, "created": 1, "default": False}
    if owner:
        meta["owner"] = owner
    (d / "profile.json").write_text(json.dumps(meta), encoding="utf-8")
    for f in ("reference.wav", "audition.wav", "fillers/0.wav"):
        sf.write(d / f, np.full(480, 0.1, np.float32), 24000)


@pytest.fixture
def clones(dispatch, monkeypatch):                                   # noqa: F811
    from fastapi.testclient import TestClient

    from dispatch_voice.host import routes
    from dispatch_voice.host.config import VoiceSettings
    from dispatch_voice.host.engines import chatterbox
    main, extensions, data = dispatch
    import importlib
    try:
        hooks = importlib.import_module("app.ext_profile_hooks")
    except ImportError:
        pytest.skip("this DisPatch predates the per-profile voice hooks (app/ext_profile_hooks.py)")
    monkeypatch.setenv("DISPATCH_VOICE", "1")
    monkeypatch.setattr(VoiceSettings, "missing", lambda self: [])
    monkeypatch.setattr(chatterbox, "available", lambda: True)
    monkeypatch.setattr(chatterbox.LazyEngine, "_get", lambda self: _FakeEngine())    # never load a model
    monkeypatch.setattr(routes, "_preview_wav", lambda settings, voice, text: b"RIFFxxxx")
    clients = []

    def client():
        c = TestClient(main.app, client=("127.0.0.1", 50000))
        c.__enter__()
        c.headers.update({"Origin": "http://testserver"})
        clients.append(c)
        return c

    owner = client()
    auth = importlib.import_module("app.auth")
    auth._sessions.clear()
    auth.set_pin("2468")
    r = owner.post("/api/auth/unlock", json={"pin": "2468"})
    if r.status_code != 200:
        pytest.skip(f"cannot sign in as the owner on this DisPatch: {r.status_code} {r.text[:200]}")
    people = {}
    for name, pin in (("Ann", "1111"), ("Ben", "2222")):
        r = owner.post("/api/people", json={"name": name, "role": "user", "pin": pin})
        if r.status_code == 409:                  # the data dir outlives one test: reuse the profile
            rows = owner.get("/api/people").json()
            rows = rows["profiles"]
            people[name] = next(p for p in rows if p["name"] == name)
            continue
        assert r.status_code == 201, r.text
        people[name] = r.json()
    sessions = {}
    for name, pin in (("Ann", "1111"), ("Ben", "2222")):
        c = client()
        assert c.post("/api/auth/login", json={"name": name, "pin": pin}).status_code == 200
        sessions[name] = c
    yield SimpleNs(owner=owner, ann=sessions["Ann"], ben=sessions["Ben"], people=people,
                   voices=data / "voices", hooks=hooks)
    for c in clients:
        c.__exit__(None, None, None)


class SimpleNs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_the_add_on_declares_per_profile_clones(clones):
    assert clones.hooks.registered_voice_access(), "setup(ctx) did not call register_voice_access"
    # voice_owned_clones_active() is False until WP-CORE replaces the Foundation stub body.
    if not clones.hooks.voice_owned_clones_active():
        pytest.skip("host half (WP-CORE) not merged yet")


def test_profile_b_cannot_list_fetch_hear_or_speak_profile_as_clone(clones):
    a_id = clones.people["Ann"]["id"]
    _clone(clones.voices, "ann-voice", a_id, "Ann voice")
    _clone(clones.voices, "house-voice", None, "House voice")
    if not clones.hooks.voice_owned_clones_active():
        pytest.skip("host half (WP-CORE) not merged yet")
    # A sees her own clone and the household one.
    listed = {v["id"] for v in clones.ann.get("/api/voice/profiles").json()["voices"]}
    assert {"ann-voice", "house-voice"} <= listed
    assert clones.ann.get("/api/voice/profiles/ann-voice").status_code == 200
    assert clones.ann.get("/api/voice/profiles/ann-voice/files/reference.wav").status_code == 200
    assert clones.ann.get("/api/voice/preview", params={"voice": "ann-voice"}).status_code == 200
    # B sees the household voice and nothing of A's.
    listed = {v["id"] for v in clones.ben.get("/api/voice/profiles").json()["voices"]}
    assert "ann-voice" not in listed and "house-voice" in listed
    assert clones.ben.get("/api/voice/profiles/ann-voice").status_code == 404
    assert clones.ben.get("/api/voice/profiles/ann-voice/files/reference.wav").status_code == 404
    assert clones.ben.get("/api/voice/preview", params={"voice": "ann-voice"}).status_code == 404
    assert clones.ben.post("/api/voice/profiles/ann-voice/say", json={"text": "hello"}).status_code == 404
    assert clones.ben.patch("/api/voice/profiles/ann-voice", json={"name": "mine"}).status_code == 404
    assert clones.ben.delete("/api/voice/profiles/ann-voice").status_code == 404
    status = clones.ben.get("/api/voice/status").json()
    assert "ann-voice" not in {p["id"] for p in status["profiles"]}
    # The Owner (an Admin) does not see it either.
    owner_listed = {v["id"] for v in clones.owner.get("/api/voice/profiles").json()["voices"]}
    assert "ann-voice" not in owner_listed
    assert (clones.voices / "ann-voice" / "profile.json").exists()          # nothing was deleted


def test_a_profile_cannot_make_a_household_voice(clones):
    if not clones.hooks.voice_owned_clones_active():
        pytest.skip("host half (WP-CORE) not merged yet")
    r = clones.ann.post("/api/voice/profiles", data={"name": "x", "consent": "self", "household": "true"},
                        files={"file": ("a.wav", b"RIFF....WAVE", "audio/wav")})
    assert r.status_code in (403, 422)
