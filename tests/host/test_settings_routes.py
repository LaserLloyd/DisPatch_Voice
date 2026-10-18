"""1.6.0 settings API: the schema endpoint, "set by environment" sources (env beats voice.yaml),
the new persisted keys, the address policy on PATCH, the write-only rig token, the status health,
the provider catalog and the in-app model download."""
from __future__ import annotations

import logging
import threading

import pytest
import yaml
from fakes import assert_private

from dispatch_voice.host import config as vconfig
from dispatch_voice.host import fetch_job, models, rig, routes
from dispatch_voice.host import settings_schema as S

TOKEN = "tok-9f3c1d7a-not-a-real-token"  # scrub-ok: test fixture


@pytest.fixture(autouse=True)
def _isolated(host_env, monkeypatch, tmp_path):
    rig._MISCONFIGURED.clear()
    monkeypatch.delenv(rig.TOKEN_ENV, raising=False)
    path = tmp_path / "home" / ".config" / "local-chat" / "speechforge.env"
    monkeypatch.setattr(rig, "token_path", lambda: path)
    monkeypatch.setattr(models, "legacy_chatterbox_dir", lambda: tmp_path / "no-old-chatterbox")
    yield path
    rig._MISCONFIGURED.clear()


def yaml_of(ctx):
    p = ctx.paths.voice_yaml
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


# --------------------------------------------------------------------------- #
# schema + sources
# --------------------------------------------------------------------------- #

def test_the_schema_endpoint_serves_the_table(voice_app):
    c, _ = voice_app
    r = c.get("/api/voice/settings/schema")
    assert r.status_code == 200 and r.json() == S.schema_json()
    by = {x["id"]: x for x in r.json()["settings"]}
    assert by["rig.url"]["level"] == "basic" and by["rig.token"]["secret"] and by["tts_engine"]["group"] == "tts"


def test_environment_beats_voice_yaml_and_the_response_says_so(voice_app, monkeypatch):
    c, ctx = voice_app
    ctx.paths.voice_yaml.write_text("speed: 1.2\nmerge_window_s: 2.0\n", encoding="utf-8")
    out = c.get("/api/voice/settings").json()
    assert out["speed"] == 1.2 and out["sources"]["speed"] == "file" and out["env"] == {}
    monkeypatch.setenv("DISPATCH_VOICE_SPEED", "0.9")
    out = c.get("/api/voice/settings").json()
    assert out["speed"] == 0.9                                       # 1.5.0: 1.2 (the file won)
    assert out["sources"]["speed"] == "env" and out["env"]["speed"] == "DISPATCH_VOICE_SPEED"
    assert out["sources"]["merge_window_s"] == "file" and out["sources"]["hard_silence_s"] == "default"
    assert vconfig.load_settings().speed == 0.9


def test_a_patch_to_an_env_held_key_is_saved_but_reported_overridden(voice_app, monkeypatch):
    c, ctx = voice_app
    monkeypatch.setenv("DISPATCH_VOICE_SPEED", "0.9")
    r = c.patch("/api/voice/settings", json={"speed": 1.2, "merge_window_s": 2.5})
    assert r.status_code == 200
    out = r.json()
    assert out["speed"] == 0.9 and out["overridden"] == {"speed": "DISPATCH_VOICE_SPEED"}
    assert yaml_of(ctx)["speed"] == 1.2                              # kept for when the variable goes
    monkeypatch.delenv("DISPATCH_VOICE_SPEED")
    assert c.get("/api/voice/settings").json()["speed"] == 1.2


def test_the_duplicate_keys_have_one_definition_and_their_aliases_work(voice_app):
    c, ctx = voice_app
    r = c.patch("/api/voice/settings", json={"power_hold": "off", "rt": {"stt_engine": "moonshine"}})
    assert r.status_code == 200, r.text
    assert yaml_of(ctx)["rt"] == {"power_hold": "off", "stt_engine": "moonshine"}
    s = vconfig.load_settings()
    assert (s.power_hold, s.rt.power_hold, s.stt_engine, s.rt.stt_engine) == ("off", "off", "moonshine", "moonshine")
    assert c.patch("/api/voice/settings", json={"rt": {"hard_silence_s": 2.0}}).status_code == 200
    s = vconfig.load_settings()
    assert s.turn_hard_silence_s == 2.0 == s.rt.hard_silence_s
    assert yaml_of(ctx)["hard_silence_s"] == 2.0


def test_the_new_persisted_keys_round_trip(voice_app):
    c, ctx = voice_app
    body = {"tts_engine": "chatterbox", "stt_provider": "openai", "stt_url": "http://192.0.2.9:8000/v1",
            "stt_model": "whisper-1", "hq_url": "http://192.0.2.9:8001", "hq_model": "tts-1"}
    r = c.patch("/api/voice/settings", json={**body, "rig": {"allow": ["192.0.2.0/24"]}})
    assert r.status_code == 200, r.text
    out = r.json()
    for k, v in body.items():
        assert out[k] == v
    assert out["rig"]["allow"] == ["192.0.2.0/24"]
    s = vconfig.load_settings()
    assert (s.tts_engine, s.stt_provider, s.stt_url, s.hq_url, s.rig_allow) == (
        "chatterbox", "openai", "http://192.0.2.9:8000/v1", "http://192.0.2.9:8001", ("192.0.2.0/24",))
    assert "a transcription server address" not in " ".join(s.missing())
    miss = vconfig.VoiceSettings(stt_provider="openai", stt_url="").missing()
    assert "a transcription server address (stt_url)" in miss


def test_unknown_engine_and_provider_names_are_refused(voice_app):
    c, _ = voice_app
    r = c.patch("/api/voice/settings", json={"tts_engine": "nonesuch"})
    assert r.status_code == 422 and "Unknown TTS engine" in r.text and "chatterbox" in r.text
    r = c.patch("/api/voice/settings", json={"stt_provider": "nonesuch"})
    assert r.status_code == 422 and "Unknown speech-to-text provider" in r.text
    assert c.patch("/api/voice/settings", json={"extensions": ["kokoro"]}).json()["extensions"] == ["kokoro"]


# --------------------------------------------------------------------------- #
# security: where audio and text may go
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("url, frag", [
    ("ws://169.254.169.254/latest", "metadata"),
    ("ws://169.254.9.9/x", "Link-local"),
    ("ws://0.0.0.0:1236/x", "cannot be connected"),
    ("wss://rig.example.com/v1/realtime", "public address"),
])
def test_rig_url_policy_refuses_forbidden_hosts(voice_app, url, frag):
    c, ctx = voice_app
    r = c.patch("/api/voice/settings", json={"rig": {"url": url}})
    assert r.status_code == 422 and frag in r.text and "rig.url" in r.text, r.text
    assert "rig" not in yaml_of(ctx)                                  # nothing was saved


def test_rig_url_policy_allows_private_hosts_warns_on_plaintext_and_honours_the_admin_list(voice_app):
    c, ctx = voice_app
    r = c.patch("/api/voice/settings", json={"rig": {"url": "ws://127.0.0.1:1236/v1/realtime"}})
    assert r.status_code == 200 and "warnings" not in r.json()
    r = c.patch("/api/voice/settings", json={"rig": {"url": "ws://192.168.7.7:1236/v1/realtime"}})  # scrub-ok: RFC1918 policy test
    assert r.status_code == 200 and any("not encrypted" in w for w in r.json()["warnings"])
    r = c.patch("/api/voice/settings", json={"rig": {"url": "wss://10.1.1.1/v1/realtime"}})  # scrub-ok: RFC1918 policy test
    assert r.status_code == 200 and "warnings" not in r.json()
    # a public host: refused alone, accepted when the same PATCH names it on the allow-list
    pub = {"url": "wss://rig.example.com/v1/realtime"}
    assert c.patch("/api/voice/settings", json={"rig": pub}).status_code == 422
    r = c.patch("/api/voice/settings", json={"rig": {**pub, "allow": ["rig.example.com"]}})
    assert r.status_code == 200 and r.json()["rig"]["url"] == pub["url"]


def test_hq_and_stt_addresses_follow_the_same_policy(voice_app):
    c, _ = voice_app
    for key in ("hq_url", "stt_url"):
        r = c.patch("/api/voice/settings", json={key: "http://169.254.169.254/"})
        assert r.status_code == 422 and key in r.text
        r = c.patch("/api/voice/settings", json={key: "https://api.example.com/"})
        assert r.status_code == 422 and "public address" in r.text
        assert c.patch("/api/voice/settings", json={key: "http://127.0.0.1:8000"}).status_code == 200


def test_an_env_supplied_forbidden_url_is_not_used_and_the_status_says_why(voice_app, monkeypatch):
    c, _ = voice_app
    monkeypatch.setenv("DISPATCH_VOICE_RIG_URL", "ws://169.254.169.254/x")
    monkeypatch.setenv("DISPATCH_VOICE_RIG_ENABLED", "1")
    monkeypatch.setenv(rig.TOKEN_ENV, TOKEN)
    s = vconfig.load_settings()
    assert s.rig.url == "ws://169.254.169.254/x" and not s.rig.active(TOKEN)
    assert rig.open_for_session(s) is None
    st = c.get("/api/voice/status").json()
    assert st["rig"]["configured"] is False and st["rig"]["url_check"]["code"] == "url_blocked_metadata"
    entry = next(h for h in st["health"] if h["id"] == "rig")
    assert entry["state"] == "down" and entry["fix"] and entry["code"] == "url_blocked_metadata"
    assert TOKEN not in c.get("/api/voice/status").text


# --------------------------------------------------------------------------- #
# the token: write-only, private, never logged
# --------------------------------------------------------------------------- #

def test_the_token_is_saved_privately_and_never_comes_back(voice_app, _isolated, caplog):
    c, _ = voice_app
    caplog.set_level(logging.DEBUG)
    r = c.put("/api/voice/rig/token", json={"token": TOKEN})
    assert r.status_code == 200 and r.json()["token"] == "file"
    assert _isolated.read_text(encoding="utf-8") == f"{rig.TOKEN_ENV}={TOKEN}\n"
    assert_private(_isolated)
    assert rig.read_token().value == TOKEN
    for path in ("/api/voice/settings", "/api/voice/status", "/api/voice/lab", "/api/voice/providers"):
        resp = c.get(path)
        assert TOKEN not in resp.text, path
    assert c.get("/api/voice/settings").json()["rig"]["token"] == {"set": True, "source": "file", "problem": None}
    assert TOKEN not in caplog.text and TOKEN not in repr(rig.read_token())
    d = c.delete("/api/voice/rig/token")
    assert d.status_code == 200 and d.json()["token"] is None and not _isolated.exists()


@pytest.mark.parametrize("bad", ["short", "has space in it 12345", "quote'd-token-12345", 'dq"token-12345', "x" * 600,
                                 "line\nbreak-token-123", "", 12345])
def test_a_malformed_token_is_refused_without_echoing_it(voice_app, _isolated, bad):
    c, _ = voice_app
    r = c.put("/api/voice/rig/token", json={"token": bad})
    assert r.status_code == 422 and not _isolated.exists()
    assert str(bad)[:12] not in r.text or bad in ("", 12345, "short")


def test_the_token_endpoint_defers_to_the_environment(voice_app, monkeypatch, _isolated):
    c, _ = voice_app
    monkeypatch.setenv(rig.TOKEN_ENV, TOKEN)
    assert c.put("/api/voice/rig/token", json={"token": "another-token-value"}).status_code == 409
    assert c.delete("/api/voice/rig/token").status_code == 409
    assert not _isolated.exists()
    out = c.get("/api/voice/settings").json()["rig"]["token"]
    assert out == {"set": True, "source": "env", "problem": None}


def test_a_new_token_clears_the_401_latch(voice_app):
    c, _ = voice_app
    rig._MISCONFIGURED.add("x")
    c.put("/api/voice/rig/token", json={"token": TOKEN})
    assert not rig._MISCONFIGURED
    assert c.put("/api/voice/rig/token", json={"token": TOKEN, "extra": 1}).status_code == 422


def test_the_token_never_reaches_the_log_on_a_failed_connection(fake_rig_url=None):
    # the sf.1 client logs outcomes by code, not by header; the header is built in one place
    import inspect
    src = inspect.getsource(rig.RigConnection._open_socket)
    assert "log." not in src.replace("log.warning(\"voice: the rig address resolves", "")


# --------------------------------------------------------------------------- #
# status: health with reason + fix
# --------------------------------------------------------------------------- #

def test_status_health_names_what_is_missing_and_how_to_fix_it(voice_app, monkeypatch):
    c, _ = voice_app
    st = c.get("/api/voice/status").json()
    assert st["health_state"] in ("ok", "degraded", "unconfigured", "down")
    by = {h["id"]: h for h in st["health"]}
    assert by["switch"]["state"] == "unconfigured" and by["switch"]["fix"] and by["switch"]["code"] == "switch_off"
    assert by["models"]["state"] == "unconfigured" and "fetch-models" in by["models"]["fix"]
    assert all({"id", "state", "reason", "fix", "code"} <= set(h) for h in st["health"])
    assert all(h["reason"] and (h["state"] == "ok" or h["fix"] or h["id"] == "engine") for h in st["health"]
               if h["state"] != "ok")
    assert {s["id"] for s in st["system"]} == {"ffmpeg", "libopus"}
    assert "providers" in st and "models" in st and st["models"]["state"] == "idle"
    monkeypatch.setenv("DISPATCH_VOICE", "0")
    by = {h["id"]: h for h in c.get("/api/voice/status").json()["health"]}
    assert by["switch"]["code"] == "forced_off" and "DISPATCH_VOICE" in by["switch"]["fix"]


def test_status_lists_the_environment_overrides(voice_app, monkeypatch):
    c, _ = voice_app
    monkeypatch.setenv("DISPATCH_VOICE_SPEED", "1.1")
    assert c.get("/api/voice/status").json()["settings_env"]["speed"] == "DISPATCH_VOICE_SPEED"


def test_the_provider_catalog_endpoint(voice_app):
    c, _ = voice_app
    r = c.get("/api/voice/providers")
    assert r.status_code == 200
    body = r.json()
    ids = {(p["kind"], p["id"]) for p in body["providers"]}
    assert {("stt", "cpu"), ("stt", "rig"), ("stt", "openai"), ("tts", "cpu"), ("tts", "rig"), ("tts", "openai"),
            ("vad", "cpu")} <= ids
    assert body["selected"] == {"stt": "cpu", "tts": "cpu", "vad": "cpu"}
    assert isinstance(body["plugins"], list)
    c.patch("/api/voice/settings", json={"rig": {"enabled": True}})
    assert c.get("/api/voice/providers").json()["selected"]["stt"] == "rig"


# --------------------------------------------------------------------------- #
# model download job
# --------------------------------------------------------------------------- #

@pytest.fixture
def job(monkeypatch):
    """A fresh job whose body is a fake fetch (never the network)."""
    gate = threading.Event()
    seen: dict = {}

    def fake_run(*, only, force, verify, stock, echo, progress, cancel):
        seen.update(only=only, force=force, stock=stock)
        echo("speech models -> somewhere")
        progress({"phase": "speech", "file": "silero_vad.onnx", "index": 1, "total": 2, "weight_mb": 2.0,
                  "bytes": 1, "bytes_total": 2})
        progress({"phase": "speech", "file": "silero_vad.onnx", "index": 1, "total": 2, "weight_mb": 2.0, "done": True})
        gate.wait(5)
        if cancel.is_set():
            raise __import__("dispatch_voice.host.download", fromlist=["Cancelled"]).Cancelled("x")
        progress({"phase": "chatterbox", "file": "x", "index": 1, "total": 1, "weight_mb": 720.0, "done": True})
        return 0

    j = fetch_job.FetchJob(run=fake_run)
    monkeypatch.setattr(fetch_job, "JOB", j)
    j.gate, j.seen = gate, seen          # type: ignore[attr-defined]
    yield j
    gate.set()
    j.join(5)


def test_fetch_status_and_finish(voice_app, job):
    c, _ = voice_app
    assert c.get("/api/voice/models/status").json()["state"] == "idle"
    r = c.post("/api/voice/models/fetch", json={"only": "speech", "stock": False})
    assert r.status_code == 200 and r.json()["state"] == "running" and r.json()["cancellable"]
    for _ in range(200):                                   # the worker thread reports its first file
        st = c.get("/api/voice/models/status").json()
        if st["file"]:
            break
        threading.Event().wait(0.01)
    assert st["state"] == "running" and st["log"][0].startswith("speech models") and 0 < st["percent"] < 100
    assert st["file"] == "silero_vad.onnx" and job.seen["only"] == "speech" and job.seen["stock"] is False
    assert c.post("/api/voice/models/fetch", json={}).status_code == 409           # one job at a time
    job.gate.set()
    job.join(5)
    st = c.get("/api/voice/models/status").json()
    assert st["state"] == "done" and st["percent"] == 100.0 and st["error"] == "" and "layout" in st
    assert c.post("/api/voice/models/cancel").status_code == 409                   # nothing running


def test_fetch_cancel(voice_app, job):
    c, _ = voice_app
    assert c.post("/api/voice/models/fetch", json={}).status_code == 200
    r = c.post("/api/voice/models/cancel")
    assert r.status_code == 200
    job.gate.set()
    job.join(5)
    st = c.get("/api/voice/models/status").json()
    assert st["state"] == "cancelled" and st["restart_required"] is False
    assert c.post("/api/voice/models/fetch", json={"only": "speech"}).status_code == 200      # can start again


def test_fetch_failure_is_reported(voice_app, monkeypatch):
    c, _ = voice_app

    def boom(**kw):
        raise OSError("network down")

    monkeypatch.setattr(fetch_job, "JOB", fetch_job.FetchJob(run=boom))
    assert c.post("/api/voice/models/fetch", json={}).status_code == 200
    fetch_job.JOB.join(5)
    st = c.get("/api/voice/models/status").json()
    assert st["state"] == "failed" and "network down" in st["error"]


@pytest.mark.parametrize("body", [{"only": "everything"}, {"force": "yes"}, {"stock": 1}])
def test_fetch_validates_its_body(voice_app, job, body):
    c, _ = voice_app
    assert c.post("/api/voice/models/fetch", json=body).status_code == 422
    assert c.get("/api/voice/models/status").json()["state"] == "idle"


def test_migrate_endpoint_dry_run_then_apply(voice_app, host_env):
    c, ctx = voice_app
    root = ctx.paths.models_dir
    (root).mkdir(parents=True, exist_ok=True)
    (root / "silero_vad.onnx").write_bytes(b"s")
    r = c.post("/api/voice/models/migrate", json={})
    assert r.status_code == 200 and r.json()["dry_run"] is True and r.json()["moves"][0]["state"] == "move"
    assert (root / "silero_vad.onnx").exists()
    assert c.get("/api/voice/models/status").json()["layout"]["legacy"] is True
    assert c.post("/api/voice/models/migrate", json={"dry_run": "no"}).status_code == 422
    r = c.post("/api/voice/models/migrate", json={"dry_run": False})
    assert r.json()["moved"] == 1 and (root / "speech" / "silero_vad.onnx").exists()
    assert c.get("/api/voice/models/status").json()["layout"]["legacy"] is False


def test_fetch_all_runs_the_cli_code_with_progress_and_cancel(tmp_path, monkeypatch):
    """The job body is the CLI's: fetch_all with fake downloaders reports every file."""
    from dispatch_voice.host import download
    sha = __import__("hashlib").sha256(b"hello").hexdigest()
    monkeypatch.setattr(download, "FILES", [("a.bin", "http://x.invalid/a", sha), ("b.bin", "http://x.invalid/b", sha)])

    def fake_fetch(url, dest, sha256):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"hello")

    events, lines = [], []
    rc = download.fetch_all(only="speech", speech_dir=tmp_path, echo=lines.append, progress=events.append, fetch=fake_fetch)
    assert rc == 0 and [e["file"] for e in events if e.get("done")] == ["a.bin", "b.bin"]
    assert all(e["phase"] == "speech" and e["total"] == 2 for e in events)

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(download.Cancelled):
        download.fetch_all(only="speech", speech_dir=tmp_path / "c", echo=lines.append, fetch=fake_fetch, cancel=cancel)


def test_the_real_downloader_stops_between_chunks_and_removes_the_partial(tmp_path, monkeypatch):
    import io

    from dispatch_voice.host import download
    cancel = threading.Event()
    reads = {"n": 0}

    class Resp(io.BytesIO):
        headers = {"Content-Length": str(3 << 20)}

        def read(self, n=-1):
            reads["n"] += 1
            if reads["n"] == 2:
                cancel.set()
            return b"x" * (1 << 20) if reads["n"] <= 3 else b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: Resp())
    seen = []
    dest = tmp_path / "f.bin"
    with pytest.raises(download.Cancelled):
        download._fetch("http://x.invalid/f", dest, "0" * 64, progress=lambda d, t: seen.append((d, t)), cancel=cancel)
    assert not dest.exists() and not (tmp_path / "f.bin.part").exists()
    assert seen and seen[0] == (1 << 20, 3 << 20)


def test_tts_provider_keeps_the_hq_address_while_switched_to_cpu(voice_app):
    c, _ = voice_app
    c.patch("/api/voice/settings", json={"hq_url": "http://127.0.0.1:8001", "hq_model": "tts-1"})
    assert c.get("/api/voice/providers").json()["selected"]["tts"] == "openai"           # auto + an address
    c.patch("/api/voice/settings", json={"tts_provider": "cpu"})
    out = c.get("/api/voice/settings").json()
    assert out["hq_url"] == "http://127.0.0.1:8001" and out["tts_provider"] == "cpu"     # the address survives
    assert c.get("/api/voice/providers").json()["selected"]["tts"] == "cpu"
    assert vconfig.load_settings().status()["hq"] is False
    c.patch("/api/voice/settings", json={"tts_provider": "openai"})
    assert c.get("/api/voice/providers").json()["selected"]["tts"] == "openai"
    assert c.patch("/api/voice/settings", json={"tts_provider": "nonesuch"}).status_code == 422
