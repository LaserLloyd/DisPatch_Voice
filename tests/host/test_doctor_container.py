"""``dispatch-voice doctor`` inside the DisPatch container, where nothing but the environment says
where the data is: the same data-folder variables DisPatch honours (``DISPATCH_DATA_DIR``, then the
legacy ``LOCAL_CHAT_DATA_DIR``), and the rig token as the host credential store keeps it
(``<data>/ext/voice/secrets.json``). No network: every probe that dials is replaced."""
from __future__ import annotations

import json

import pytest

from dispatch_voice import cli
from dispatch_voice.host import context, rig

TOKEN = "container-token-not-real"  # scrub-ok: test fixture


@pytest.fixture
def bare(monkeypatch, tmp_path):
    """No DisPatch context and no data-folder variable: a plain shell, home under tmp_path."""
    context.clear_context()
    for k in ("DISPATCH_DATA_DIR", "LOCAL_CHAT_DATA_DIR", "DISPATCH_SPEECH_MODELS", "XDG_DATA_HOME",
              rig.TOKEN_ENV):
        monkeypatch.delenv(k, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    rig._MISCONFIGURED.clear()
    yield tmp_path
    context.clear_context()


# ---------------------------------------------------------------- the data folder

def test_legacy_local_chat_data_dir_is_honoured(bare, monkeypatch):
    data = bare / "data"
    monkeypatch.setenv("LOCAL_CHAT_DATA_DIR", str(data))
    assert context.default_data_dir() == data
    p = context.paths()
    assert p.voice_yaml == data / "voice.yaml" and p.state_dir == data / "ext" / "voice"
    assert p.models_dir == data / "models"


def test_dispatch_data_dir_wins_over_the_legacy_name(bare, monkeypatch):
    monkeypatch.setenv("LOCAL_CHAT_DATA_DIR", str(bare / "old"))
    monkeypatch.setenv("DISPATCH_DATA_DIR", str(bare / "new"))
    assert context.default_data_dir() == bare / "new"


def test_a_blank_dispatch_data_dir_falls_through_to_the_legacy_name(bare, monkeypatch):
    monkeypatch.setenv("DISPATCH_DATA_DIR", "   ")
    monkeypatch.setenv("LOCAL_CHAT_DATA_DIR", str(bare / "old"))
    assert context.default_data_dir() == bare / "old"


def test_with_neither_variable_the_xdg_default_is_unchanged(bare, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(bare / "xdg"))
    assert context.default_data_dir() == bare / "xdg" / "local-chat"
    assert context.data_dir_override() == ""
    monkeypatch.delenv("XDG_DATA_HOME")
    assert context.default_data_dir().parts[-3:] == (".local", "share", "local-chat")


def test_the_token_file_follows_the_legacy_variable_too(bare, monkeypatch):
    assert rig.token_path() == rig.legacy_token_path()                 # no variable: the old home file
    monkeypatch.setenv("LOCAL_CHAT_DATA_DIR", str(bare / "data"))
    assert rig.token_path() == bare / "data" / "ext" / "voice" / "speechforge.env"


def test_doctor_reads_the_folder_named_by_local_chat_data_dir(bare, monkeypatch):
    """The container sets only LOCAL_CHAT_DATA_DIR=/data: the doctor must read that folder's
    voice.yaml and models, not ~/.local/share/local-chat."""
    data = bare / "data"
    data.mkdir()
    monkeypatch.setenv("LOCAL_CHAT_DATA_DIR", str(data))
    monkeypatch.setattr(cli, "probe_running", lambda url: {"url": url, "reachable": False})
    rep = cli.doctor_report(backend=None, url="http://127.0.0.1:1", check_update=False)
    assert rep["switch"]["voice_yaml"] == str(data / "voice.yaml")
    assert str(data) in rep["speech_models"]["dir"]
    assert rep["rig"]["token_file"] == str(data / "ext" / "voice" / "speechforge.env")
    assert str(bare / "home") not in json.dumps(rep["switch"]) + rep["speech_models"]["dir"]


# ---------------------------------------------------------------- the rig token

def _secrets(data, payload):
    p = data / "ext" / "voice" / "secrets.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_doctor_finds_the_token_in_the_host_secrets_file(bare, monkeypatch):
    data = bare / "data"
    _secrets(data, {rig.TOKEN_SECRET: TOKEN, "other": "x"})
    monkeypatch.setenv("LOCAL_CHAT_DATA_DIR", str(data))
    assert rig.read_token().source is None                             # the runtime path is unchanged
    t = rig.read_host_secrets_file_token()
    assert t.value == TOKEN and t.source == "secret" and TOKEN not in repr(t)
    from dispatch_voice.host.config import VoiceSettings
    r = cli.probe_rig(VoiceSettings())
    assert r["token"] == "secret" and r["secrets_file"] == str(data / "ext" / "voice" / "secrets.json")
    line, bad = cli.rig_line({**r, "enabled": True, "url_set": True, "url_ok": True, "host": "h:1",
                              "health": None, "detail": "ConnectError"})
    assert "no rig token" not in line and "not reachable" in line and bad == 0
    assert TOKEN not in line


@pytest.mark.parametrize("content", [None, "{not json", "[]", json.dumps({"other": "x"}),
                                     json.dumps({rig.TOKEN_SECRET: "  "}), json.dumps({rig.TOKEN_SECRET: 5})])
def test_a_missing_or_useless_secrets_file_is_no_token(bare, monkeypatch, content):
    data = bare / "data"
    monkeypatch.setenv("DISPATCH_DATA_DIR", str(data))
    if content is not None:
        p = data / "ext" / "voice" / "secrets.json"
        p.parent.mkdir(parents=True)
        p.write_text(content, encoding="utf-8")
    assert rig.read_host_secrets_file_token().value == ""


def test_the_environment_variable_still_wins_over_the_secrets_file(bare, monkeypatch):
    data = bare / "data"
    _secrets(data, {rig.TOKEN_SECRET: TOKEN})
    monkeypatch.setenv("DISPATCH_DATA_DIR", str(data))
    monkeypatch.setenv(rig.TOKEN_ENV, "env-token-not-real")  # scrub-ok: test fixture
    from dispatch_voice.host.config import VoiceSettings
    assert cli.probe_rig(VoiceSettings())["token"] == "env"


def test_the_no_token_line_names_every_place_it_looked():
    line, bad = cli.rig_line({"enabled": True, "url_set": True, "url_ok": True, "token": None, "host": "",
                              "token_problem": None, "token_file": "/d/ext/voice/speechforge.env",
                              "secrets_file": "/d/ext/voice/secrets.json"})
    assert bad == 1
    for place in ("SPEECHFORGE_TOKEN", "/d/ext/voice/secrets.json", "/d/ext/voice/speechforge.env"):
        assert place in line
