"""The cross-repo contract (SPEC §12 "Contract"; design-addon §9.3): this add-on
against a REAL DisPatch checkout that carries the extension host
(``app/extensions.py``, DisPatch 2.2.0+).

Run from DisPatch's backend environment with the add-on added:

    cd <dispatch-chat>/backend
    uv run --offline --with-editable <this repo> pytest -o testpaths= --rootdir <this repo> <this repo>/tests/contract

``DISPATCH_BACKEND`` names the backend directory (default: the sibling
checkout ``../dispatch-chat/backend``). Skipped, with the reason, when the
checkout is missing, predates the extension host, or this environment cannot
import it. Throwaway data dir; fake engines (no models); the agent turn is
replaced so no gateway or provider is ever called.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(os.environ.get("DISPATCH_BACKEND",
                              Path(__file__).resolve().parents[2].parent / "dispatch-chat" / "backend"))
pytestmark = pytest.mark.contract


class FakeSTT:
    text = "what is on my calendar"

    def transcribe(self, audio):
        return self.text


class FakeTTS:
    sample_rate = 24000
    sample_format = "pcm_s16le"
    channels = 1
    streaming = True

    def stream(self, text, voice=None, stop=None):
        yield np.zeros(240, "<i2")

    def synth(self, text, voice=None):
        return np.zeros(240, "<i2"), 24000


_ENV_KEYS = ("DISPATCH_DATA_DIR", "DISPATCH_EXTENSIONS", "DISPATCH_VOICE_PREWARM", "DISPATCH_MIRROR",
             "DISPATCH_GATEWAY_WS", "OPENCLAW_BIN")


@pytest.fixture(scope="module")
def dispatch(tmp_path_factory):
    """DisPatch imported in-process. Whatever it changes in this interpreter (``sys.path``,
    the environment, the ``app`` package in ``sys.modules``) is put back afterwards, and
    also when it skips: a skipped contract module must not leave ``app.*`` importable for
    the rest of the add-on suite (``dispatch-voice doctor`` tests read exactly that)."""
    saved_path = list(sys.path)
    saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}
    saved_mods = set(sys.modules)
    try:
        yield from _dispatch(tmp_path_factory)
    finally:
        sys.path[:] = saved_path
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for name in list(sys.modules):
            if name not in saved_mods and (name == "app" or name.startswith("app.")):
                del sys.modules[name]


def _dispatch(tmp_path_factory):
    if not (BACKEND / "app" / "main.py").exists():
        pytest.skip(f"no DisPatch backend at {BACKEND} (set DISPATCH_BACKEND)")
    if not (BACKEND / "app" / "extensions.py").exists():
        pytest.skip("this DisPatch predates the extension host (app/extensions.py)")
    if importlib.util.find_spec("fastapi") is None:
        pytest.skip("not running in DisPatch's environment")
    data = tmp_path_factory.mktemp("data")
    os.environ["DISPATCH_DATA_DIR"] = str(data)
    os.environ["DISPATCH_EXTENSIONS"] = "dispatch_voice:EXTENSION"
    os.environ["DISPATCH_VOICE_PREWARM"] = "0"
    os.environ["DISPATCH_MIRROR"] = "0"
    os.environ["DISPATCH_GATEWAY_WS"] = "0"
    os.environ["OPENCLAW_BIN"] = "/nonexistent/openclaw"
    (data / "config.yaml").write_text(json.dumps({"bots": [
        {"id": "voicetest", "name": "Voice Test", "emoji": "🎙", "order": 0, "visible": True, "safe": False,
         "api": {"provider": "openai", "base_url": "http://127.0.0.1:9/v1", "model": "stub", "api_key": "not-a-key"}},
        {"id": "alpha", "name": "Alpha", "emoji": "✨", "order": 1, "visible": True, "safe": True},
    ]}, ensure_ascii=False))
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    try:
        extensions = importlib.import_module("app.extensions")
        main = importlib.import_module("app.main")
    except Exception as e:  # pragma: no cover - environment-specific
        pytest.skip(f"DisPatch did not import: {type(e).__name__}: {e}")
    yield main, extensions, data


@pytest.fixture
def client(dispatch, monkeypatch):
    from fastapi.testclient import TestClient

    from dispatch_voice.host import routes, vad_turn
    from dispatch_voice.host.config import VoiceSettings
    main, _, _ = dispatch
    auth = importlib.import_module("app.auth")
    auth._sessions.clear()
    monkeypatch.setenv("DISPATCH_VOICE", "1")
    monkeypatch.setenv("DISPATCH_VOICE_MERGE_WINDOW", "0")
    monkeypatch.setattr(VoiceSettings, "missing", lambda self: [])

    async def fake_warm(settings):
        return FakeSTT(), FakeTTS()

    class OneTurnDetector:
        def __init__(self):
            from types import SimpleNamespace
            self.ep = SimpleNamespace(cfg=vad_turn.EndpointerConfig(), turn_fn=None)

        def feed(self, pcm):
            return [vad_turn.TurnEnd(audio=np.zeros(16000, np.float32), reason="silence")]

        def reset(self):
            pass

    monkeypatch.setattr(routes, "warm_models", fake_warm)
    monkeypatch.setattr(vad_turn, "build_detector", lambda s: OneTurnDetector())
    monkeypatch.setattr(routes, "voice_for_bot", lambda s, e, b: None)
    turns = []

    async def fake_turn(thread_id, bot_id, text, opts=None):
        turns.append((thread_id, bot_id, text, opts))
        await main._persist_and_broadcast_message(thread_id, "assistant", "Two meetings today. Want details?")

    monkeypatch.setattr(main, "run_agent_turn", fake_turn)
    with TestClient(main.app) as c:
        yield c, turns, auth


def unlock(c, auth):
    auth.set_pin("1234")
    assert c.post("/api/auth/unlock", json={"pin": "1234"}).status_code == 200


def lock(c, auth):
    auth.set_pin("1234")
    c.cookies.clear()


def thread_for(c, bot_id="voicetest"):
    r = c.post("/api/threads", json={"bot_id": bot_id})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def refused(c, path, **kw):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as e:
        with c.websocket_connect(path, **kw) as ws:
            ws.receive_text()
    return e.value


# --------------------------------------------------------------------------- #

def test_extensions_listing_says_voice_is_ok_with_api_1(client, dispatch):
    c, _, auth = client
    _, extensions, _ = dispatch
    from dispatch_voice import EXTENSION, __version__
    unlock(c, auth)
    r = c.get("/api/extensions")
    assert r.status_code == 200, r.text
    row = next(e for e in r.json()["extensions"] if e["id"] == "voice")
    assert row["state"] == "ok", row
    assert row["api"] == list(EXTENSION["api"]) and row["version"] == __version__
    assert EXTENSION["api"][0] <= extensions.EXT_API <= EXTENSION["api"][1]
    assert row["entry"] == "index.js" and row["css"] == ["voice.css"] and row["locales"] is True
    assert set(row["tools"]) == {"voice-cloning", "voice-builtin", "voice-lab"}
    assert c.get("/api/auth/status").json()["features"]["extensions"] >= 1


def test_voice_api_is_unlocked_only(client):
    c, _, auth = client
    lock(c, auth)
    assert c.get("/api/voice/status").status_code == 403
    unlock(c, auth)
    st = c.get("/api/voice/status")
    assert st.status_code == 200 and st.json()["enabled"] is True
    assert st.json()["extension"]["name"] == "DisPatch Voice"


def test_static_assets_are_operator_only_with_an_etag(client):
    c, _, auth = client
    lock(c, auth)
    assert c.get("/ext/voice/index.js").status_code == 403
    unlock(c, auth)
    r = c.get("/ext/voice/index.js")
    assert r.status_code == 200 and r.headers.get("etag")
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert c.get("/ext/voice/locales/en.json").status_code == 200
    assert c.get("/ext/voice/voices.html").status_code == 404     # allowlist: no documents


def test_bot_voice_is_validated_by_the_add_on(client):
    c, _, auth = client
    unlock(c, auth)
    assert c.put("/api/bots/order", json={"bots": [{"id": "voicetest", "voice": "who-knows"}]}).status_code == 422
    assert c.put("/api/bots/order", json={"bots": [{"id": "voicetest", "voice": ""}]}).status_code == 200
    assert c.put("/api/bots/order", json={"bots": [{"id": "voicetest", "voice_rp": "quotes"}]}).status_code == 422
    assert c.put("/api/bots/order", json={"bots": [{"id": "voicetest", "voice_rp": "dialogue"}]}).status_code == 200


def test_locked_device_is_refused_pre_accept(client):
    c, _, auth = client
    unlock(c, auth)                      # the thread itself needs the unlocked tier
    tid = thread_for(c)
    lock(c, auth)
    assert refused(c, f"/ws/voice/{tid}").code == 1008


def test_cross_origin_handshake_is_refused_even_unlocked(client):
    c, _, auth = client
    unlock(c, auth)
    tid = thread_for(c)
    assert refused(c, f"/ws/voice/{tid}", headers={"origin": "https://evil.example"}).code == 1008


def test_safe_bot_thread_is_refused(client):
    c, _, auth = client
    unlock(c, auth)
    tid = thread_for(c, "alpha")
    assert refused(c, f"/ws/voice/{tid}").code == 1008


def test_feature_off_is_accept_then_1013_with_the_reason(client, monkeypatch):
    c, _, auth = client
    unlock(c, auth)
    tid = thread_for(c)
    monkeypatch.setenv("DISPATCH_VOICE", "0")
    e = refused(c, f"/ws/voice/{tid}")
    assert e.code == 1013 and "on this server" in (e.reason or "")


def test_spoken_turn_lands_in_thread_and_reply_is_spoken(client):
    c, turns, auth = client
    unlock(c, auth)
    tid = thread_for(c)
    with c.websocket_connect(f"/ws/voice/{tid}", headers={"origin": "http://testserver"}) as ws:
        assert json.loads(ws.receive_text())["type"] == "ready"
        ws.send_text(json.dumps({"type": "start"}))
        ws.send_bytes(b"\x00\x00" * 320)
        seen, audio = [], 0
        for _ in range(60):
            m = ws.receive()
            if m.get("bytes") is not None:
                audio += 1
                break
            seen.append(json.loads(m["text"]))
        types = [f["type"] for f in seen]
        assert "eot" in types and "transcript" in types and audio >= 1
        hdr = [f for f in seen if f["type"] == "audio" and f.get("kind") == "reply"]
        assert hdr and hdr[0]["text"].startswith("Two meetings today")
    msgs = c.get(f"/api/threads/{tid}/messages").json()
    rows = msgs["messages"] if isinstance(msgs, dict) else msgs
    user = [m for m in rows if m["role"] == "user"]
    assert user and user[0]["content"] == "what is on my calendar"
    assert user[0]["metadata"]["voice"] is True
    (_, bot_id, text, opts) = turns[0]
    assert bot_id == "voicetest" and text == "what is on my calendar"
    voice = getattr(opts, "voice", None)
    assert voice is not None and voice.system.startswith("## Spoken replies") and opts.thinking == "off"
    assert voice.trailer == ""                       # an API bot: the system slot only


def test_v3_wire_through_the_real_gate(client):
    """Slice 4 (Amendment V1-1): a client asking for `v=3` gets protocol 3 from the add-on
    behind DisPatch's real gate — `ready.proto`, an offset-carrying uplink, the reply as
    `say` + binary v3 frames + `audio_end`, and `pong.rx_off` on the uplink clock."""
    from dispatch_voice.transport import wire3
    c, _, auth = client
    unlock(c, auth)
    tid = thread_for(c)
    q = "d=dev-contract1&s=str-contract-0001&v=3&up=pcm16&down=pcm16"
    with c.websocket_connect(f"/ws/voice/{tid}?{q}", headers={"origin": "http://testserver"}) as ws:
        ready = json.loads(ws.receive_text())
        assert ready["type"] == "ready" and ready["proto"] == 3 and ready["codec_out"] == "pcm16"
        ws.send_text(json.dumps({"type": "start"}))
        ws.send_bytes(wire3.pack_up(wire3.CODEC_PCM16, 0, 4000, [b"\x00\x00" * 320]))
        seen, binary = [], []
        for _ in range(80):
            m = ws.receive()
            if m.get("bytes") is not None:
                binary.append(wire3.unpack_down(m["bytes"]))
                continue
            f = json.loads(m["text"])
            seen.append(f)
            if f["type"] == "audio_end":
                break
        types = [f["type"] for f in seen]
        assert "audio" not in types                          # no JSON audio header on v3
        say = next(f for f in seen if f["type"] == "say" and f["kind"] == "reply")
        assert say["text"].startswith("Two meetings today")
        assert binary and binary[0][1] == wire3.KIND_REPLY and binary[0][3] == say["seq"]
        assert types.index("say") < types.index("audio_end")
        assert next(f for f in seen if f["type"] == "eot")["upto"] == 4320
        ws.send_text(json.dumps({"type": "ping", "t": 1}))
        for _ in range(40):
            f = json.loads(ws.receive_text())
            if f["type"] == "pong":
                assert f["rx_off"] == 4320 and f["t"] == 1
                break
        else:
            raise AssertionError("no pong")
