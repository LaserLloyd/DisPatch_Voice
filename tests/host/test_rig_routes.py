"""The rig's settings (voice.yaml ``rig:``), its token (env or a private env file;
never stored, logged or sent to the browser), the status the Settings row and the
Voice lab draw, the deletion sync and the doctor line."""
from __future__ import annotations

import os

import pytest
import yaml
from fakes import make_private

from dispatch_voice import cli
from dispatch_voice.host import config as vconfig
from dispatch_voice.host import rig

SECRET = "rig-token-value-not-real"  # scrub-ok: test fixture
URL = "ws://192.0.2.7:1236/v1/realtime"


@pytest.fixture(autouse=True)
def _isolated_token(monkeypatch, tmp_path, host_env):
    rig._MISCONFIGURED.clear()
    rig._LAST.clear()
    rig.METRICS.reset()
    monkeypatch.delenv(rig.TOKEN_ENV, raising=False)
    # URL below is a TEST-NET address: a public one, so the admin allow-list has to name it.
    monkeypatch.setenv("DISPATCH_VOICE_RIG_ALLOW", "192.0.2.0/24")
    path = tmp_path / "local-chat" / "speechforge.env"
    monkeypatch.setattr(rig, "token_path", lambda: path)
    yield path
    rig._MISCONFIGURED.clear()


def write_token(path, body: str, mode: int = 0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    make_private(path, private=mode == 0o600)


# ------------------------------------------------------------------ settings
def test_rig_settings_defaults_and_cleaning():
    s = rig.RigSettings()
    assert (s.enabled, s.url, s.stt_final_deadline_ms, s.tts_start_deadline_ms, s.ahead_ms,
            s.first_chunk_tokens) == (False, "", 350, 800, 3000, 15)
    assert rig.clean_rig({"enabled": "yes", "url": "http://x/", "ahead_ms": 99999, "nope": 1}) == {"ahead_ms": 10000}
    assert rig.clean_rig({"url": ""}) == {"url": ""}
    for bad in ("ws://u:p@h:1/x", "wss://h/v1?token=x", "ws:///x", "ftp://h/x", "ws://h:99999/x", "ws://h /x"):
        assert not rig.valid_url(bad), bad
    assert rig.valid_url(URL) and rig.valid_url("wss://rig.example:443/v1/realtime")
    assert rig.http_base("wss://h:9/v1/realtime") == "https://h:9" and rig.host_of(URL) == "192.0.2.7:1236"


def test_voice_yaml_rig_tree_merges_and_loads(host_env):
    assert vconfig.load_settings().rig == rig.RigSettings()          # no file: today's defaults
    vconfig.save_prefs({"rig": {"enabled": True, "url": URL}})
    vconfig.save_prefs({"rig": {"ahead_ms": 4000}})                  # merged key by key
    st = vconfig.load_settings().rig
    assert (st.enabled, st.url, st.ahead_ms, st.stt_final_deadline_ms) == (True, URL, 4000, 350)
    text = host_env.paths.voice_yaml.read_text(encoding="utf-8")
    assert yaml.safe_load(text)["rig"] == {"enabled": True, "url": URL, "ahead_ms": 4000}
    assert "token" not in text.lower()


# ------------------------------------------------------------------ token
def test_token_from_env_then_a_private_file(monkeypatch, _isolated_token):
    assert rig.read_token() == rig.Token()
    write_token(_isolated_token, f"# rig\nexport {rig.TOKEN_ENV}='{SECRET}'\n")
    t = rig.read_token()
    assert t.value == SECRET and t.source == "file" and SECRET not in repr(t)
    monkeypatch.setenv(rig.TOKEN_ENV, "from-env-not-real")  # scrub-ok: test fixture
    assert rig.read_token().source == "env"


def test_a_world_readable_token_file_is_refused(_isolated_token, caplog):
    write_token(_isolated_token, f"{rig.TOKEN_ENV}={SECRET}\n", mode=0o644)
    t = rig.read_token()
    assert t.value == "" and t.problem == "permissions"
    assert SECRET not in caplog.text
    assert ("Everyone" if os.name == "nt" else "chmod 600") in caplog.text


# ------------------------------------------------------------------ routes
def test_settings_route_patches_the_rig_tree(voice_app):
    c, ctx = voice_app
    assert c.get("/api/voice/settings").json()["rig"]["enabled"] is False
    r = c.patch("/api/voice/settings", json={"rig": {"enabled": True, "url": URL}})
    assert r.status_code == 200, r.text
    assert r.json()["rig"]["enabled"] is True and r.json()["rig"]["url"] == URL
    for body, frag in [({"rig": {"url": "http://x/"}}, "ws://"), ({"rig": {"token": "x"}}, "rig.token"),
                       ({"rig": {"ahead_ms": 50}}, "between 1000"), ({"rig": {"enabled": "on"}}, "true or false"),
                       ({"rig": "on"}, "rig takes an object")]:
        r = c.patch("/api/voice/settings", json=body)
        assert r.status_code == 422 and frag in r.text, (body, r.text)
    assert c.patch("/api/voice/settings", json={"rig": {"url": ""}}).json()["rig"]["url"] == ""


def test_status_states_and_no_token_anywhere(voice_app, monkeypatch, _isolated_token):
    c, ctx = voice_app
    st = c.get("/api/voice/status").json()["rig"]
    assert st["state"] == "not_configured" and st["configured"] is False
    c.patch("/api/voice/settings", json={"rig": {"url": URL}})
    write_token(_isolated_token, f"{rig.TOKEN_ENV}={SECRET}\n")
    st = c.get("/api/voice/status").json()["rig"]
    assert st["state"] == "off" and st["configured"] is True and st["token"] == "file"
    c.patch("/api/voice/settings", json={"rig": {"enabled": True}})
    st = c.get("/api/voice/status").json()["rig"]
    assert st["state"] == "idle" and st["host"] == "192.0.2.7:1236" and st["active"] is True
    rig._MISCONFIGURED.add(rig._fingerprint(URL, SECRET))
    assert c.get("/api/voice/status").json()["rig"]["state"] == "misconfigured"
    lab = c.get("/api/voice/lab").json()["rig"]
    assert lab["deadlines"] == {"stt_final_ms": 350, "tts_start_ms": 800, "ahead_ms": 3000, "first_chunk_tokens": 15}
    for path in ("/api/voice/status", "/api/voice/settings", "/api/voice/lab"):
        assert SECRET not in c.get(path).text, path


# ------------------------------------------------------------------ power gate
def test_power_gate_holds_only_while_the_cpu_serves():
    calls = []

    class P:
        def acquire(self, s):
            calls.append("acquire")

        def release(self):
            calls.append("release")

    g = rig.RigPowerGate(None, power=P())
    g.start()                                   # nothing at the open
    assert calls == [] and not g.held
    g.need()
    g.need()                                    # idempotent
    g.not_needed()
    g.need()
    g.release()                                 # the session ends while held
    assert calls == ["acquire", "release", "acquire", "release"]
    g.need()                                    # after the end: nothing
    assert calls[-1] == "release" and len(calls) == 4


# ------------------------------------------------------------------ deletion sync
def test_deleting_a_profile_deletes_its_conditionals_on_the_rig(host_env, monkeypatch, _isolated_token, tmp_path):
    import hashlib
    from types import SimpleNamespace
    root = tmp_path / "voices"
    (root / "host-voice").mkdir(parents=True)
    (root / "host-voice" / "voice.npz").write_bytes(b"tensors")
    store = SimpleNamespace(root=root, get=lambda pid: {"id": pid})
    assert rig.sha_for_delete(store, "host-voice") is None         # no rig set up: nothing read
    vconfig.save_prefs({"rig": {"url": URL}})                # switched off, but an address
    write_token(_isolated_token, f"{rig.TOKEN_ENV}={SECRET}\n")
    sha = rig.sha_for_delete(store, "host-voice")
    assert sha == hashlib.sha256(b"tensors").hexdigest()
    seen = []
    import httpx
    monkeypatch.setattr(httpx, "delete", lambda url, **kw: seen.append((url, kw["headers"])) or
                        SimpleNamespace(status_code=200))
    rig.delete_voice_async(sha)
    for t in __import__("threading").enumerate():
        if t.name == "voice-rig-delete":
            t.join(2)
    assert seen == [(f"http://192.0.2.7:1236/v1/voices/{sha}", {"Authorization": f"Bearer {SECRET}"})]


@pytest.mark.parametrize("url,allow", [
    ("ws://169.254.169.254/x", "192.0.2.0/24"),       # cloud metadata: never, allow-list or not
    (URL, ""),                                        # a public address that is not on the allow-list
])
def test_the_deletion_sync_never_sends_the_token_to_an_address_the_policy_refuses(
        host_env, monkeypatch, _isolated_token, tmp_path, url, allow):
    """rig.url from a 1.5.0 voice.yaml or the environment never passed the PATCH check: the
    deletion sync (fired even with the rig switched off) must apply the policy itself."""
    from types import SimpleNamespace
    import httpx
    monkeypatch.setenv("DISPATCH_VOICE_RIG_ALLOW", allow)
    root = tmp_path / "voices"
    (root / "v").mkdir(parents=True)
    (root / "v" / "voice.npz").write_bytes(b"tensors")
    vconfig.save_prefs({"rig": {"url": url}})
    write_token(_isolated_token, f"{rig.TOKEN_ENV}={SECRET}\n")
    monkeypatch.setattr(httpx, "delete", lambda *a, **k: (_ for _ in ()).throw(AssertionError("token sent")))
    assert rig._delete_target() is None
    assert rig.sha_for_delete(SimpleNamespace(root=root, get=lambda pid: {}), "v") is None
    rig.delete_voice_async("0" * 64)                  # nothing to send it to: no thread, no request


def test_doctor_probe_never_sends_the_token_to_an_address_the_policy_refuses(host_env, monkeypatch, _isolated_token):
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("dialled")))
    write_token(_isolated_token, f"{rig.TOKEN_ENV}={SECRET}\n")
    vconfig.save_prefs({"rig": {"enabled": True, "url": "ws://169.254.169.254/x"}})
    r = cli.probe_rig(vconfig.load_settings())
    assert r["policy"] == "url_blocked_metadata" and "health" not in r
    line, n = cli.rig_line(r)
    assert "network policy" in line and n == 1
    monkeypatch.setenv("DISPATCH_VOICE_RIG_ALLOW", "")
    vconfig.save_prefs({"rig": {"url": URL}})
    assert cli.probe_rig(vconfig.load_settings())["policy"] == "url_not_allowed"


# ------------------------------------------------------------------ doctor
@pytest.mark.parametrize("rep,want,bad", [
    ({"enabled": False, "url_set": False, "url_ok": False}, "not set up", 0),
    ({"enabled": True, "url_set": True, "url_ok": False}, "not a ws://", 1),
    ({"enabled": True, "url_set": True, "url_ok": True, "token": None, "token_problem": "permissions",
      "token_file": "/x/speechforge.env"}, "chmod 600", 1),
    ({"enabled": True, "url_set": True, "url_ok": True, "token": None, "token_file": "/x/s.env"}, "no rig token found", 1),
    ({"enabled": False, "url_set": True, "url_ok": True, "token": "env", "health": None, "detail": "ConnectError"},
     "not reachable", 0),
    ({"enabled": True, "url_set": True, "url_ok": True, "token": "file", "health": 200, "ready": 401}, "refused the token", 1),
    ({"enabled": True, "url_set": True, "url_ok": True, "token": "file", "health": 200, "ready": 200,
      "version": "0.2.0", "gpu": "RTX 3090"}, "SpeechForge 0.2.0 ready on RTX 3090", 0),
    ({"enabled": True, "url_set": True, "url_ok": True, "token": "file", "health": 200, "ready": 503,
      "version": "0.2.0"}, "not ready (HTTP 503)", 0),
])
def test_doctor_rig_line(rep, want, bad):
    rep = {"host": "192.0.2.7:1236", "token_problem": None, **rep}
    line, n = cli.rig_line(rep)
    assert line.startswith("GPU rig") and want in line and n == bad


def test_doctor_probe_never_dials_without_an_address_and_a_token(host_env, monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("dialled")))
    r = cli.probe_rig(vconfig.load_settings())
    assert r["url_set"] is False and "health" not in r
    vconfig.save_prefs({"rig": {"enabled": True, "url": URL}})
    r = cli.probe_rig(vconfig.load_settings())
    assert r["url_ok"] is True and r["token"] is None and "health" not in r


def test_no_profile_id_is_the_default_profile_or_nothing_never_the_stock_voice(monkeypatch, tmp_path):
    """``None`` is the engine's own default voice. With a built-in voice plugin that is the
    plugin's voice (it stays on this computer), so the rig must not speak the stock voice
    for it; without a plugin the session already names the stock voice explicitly."""
    from types import SimpleNamespace

    from dispatch_voice.host import profiles_bridge as pb
    from dispatch_voice.host.engines import chatterbox
    for pid in ("stock-host", "mine"):
        (tmp_path / pid).mkdir()
        (tmp_path / pid / "voice.npz").write_bytes(b"t")
    default = [None]
    store = SimpleNamespace(root=tmp_path, default_id=lambda: default[0], get=lambda pid: {"id": pid})
    monkeypatch.setattr(chatterbox, "available", lambda: True)
    monkeypatch.setattr(pb, "store", lambda: store)
    monkeypatch.setattr(pb, "stock_profile", lambda build=False: "stock-host")
    assert rig.voice_file(None) is None
    assert rig.voice_file("stock-host") == tmp_path / "stock-host" / "voice.npz"
    default[0] = "mine"
    assert rig.voice_file(None) == tmp_path / "mine" / "voice.npz"
