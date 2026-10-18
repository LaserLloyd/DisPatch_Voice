"""DisPatch Voice as the first host of forge-voice-core: every moved import path still works (and is the
SAME module object), and DisPatch's answers reach the engines through the host link."""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from dispatch_voice import __version__
from dispatch_voice import _core_bridge
from dispatch_voice.host import context
from forge_voice_core import hostlink

#: old path under dispatch_voice -> new path under forge_voice_core (the shim list; keep in step with the docs)
SHIMS = {
    "audio": "audio", "engine": "engine", "fsperm": "fsperm", "profiles": "profiles", "speak": "speak",
    "stock": "stock",
    "rt.backchannel": "rt.backchannel", "rt.endpoint": "rt.endpoint", "rt.heard": "rt.heard",
    "rt.interrupt": "rt.interrupt", "rt.settings": "rt.settings", "rt.speculate": "rt.speculate",
    "rt.stt_rig": "rt.stt_rig", "rt.stt_stream": "rt.stt_stream", "rt.tts_stream": "rt.tts_stream",
    "host.vad_turn": "vad_turn", "host.stt": "stt", "host.tts": "tts", "host.providers": "providers",
    "host.registry": "registry", "host.health": "health", "host.sysreq": "sysreq",
    "host.netpolicy": "netpolicy", "host.models": "models", "host.download": "download",
    "host.fetch_job": "fetch_job", "host.rig": "rig",
    "host.engines.chatterbox": "engines.chatterbox", "host.engines.rig_tts": "engines.rig_tts",
    "text.speech_text": "text.speech_text", "transport.opus": "opus",
}
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("old,new", sorted(SHIMS.items()))
def test_a_moved_import_path_is_the_core_module_itself(old, new):
    shim = importlib.import_module(f"dispatch_voice.{old}")
    core = importlib.import_module(f"forge_voice_core.{new}")
    assert shim is core                                   # patching either name patches both
    parent, _, leaf = f"dispatch_voice.{old}".rpartition(".")
    assert getattr(importlib.import_module(parent), leaf) is core     # `from dispatch_voice.host import tts`


@pytest.mark.parametrize("old", sorted(SHIMS))
def test_every_old_path_still_exists_as_a_file_marked_deprecated(old):
    parts = old.split(".")
    p = ROOT / "src" / "dispatch_voice" / Path(*parts)
    p = p / "__init__.py" if old == "stock" else p.with_suffix(".py")
    text = p.read_text(encoding="utf-8")
    assert "DEPRECATED" in text and "forge_voice_core." in text and len(text.splitlines()) < 20


def test_dispatch_is_the_installed_host():
    host = hostlink.get()
    assert isinstance(host, _core_bridge.DispatchHostLink)
    assert host.client_name == "dispatch-voice" and host.client_version == __version__


def test_the_core_logs_under_the_names_dispatch_always_used():
    from forge_voice_core import models, opus, rig
    assert models.log.name == "dispatch.voice" and rig.log.name == "dispatch.voice"
    assert opus.log.name == "dispatch_voice.transport.opus"
    assert rig.ws_logger().name == "websockets.client.dispatch_voice"


def test_the_models_folder_and_state_come_from_the_dispatch_context(tmp_path, monkeypatch):
    p = context.env_paths(tmp_path / "data")
    monkeypatch.setattr(context, "paths", lambda: p)
    from forge_voice_core import models, rig
    assert models.root() == p.models_dir
    assert hostlink.get().voices_dir() == p.voices_dir and hostlink.get().state_dir() == p.state_dir
    monkeypatch.setenv("DISPATCH_DATA_DIR", str(tmp_path / "data"))
    assert rig.token_path() == p.state_dir / "speechforge.env"
    assert rig.host_secrets_file() == p.state_dir / "secrets.json"


def test_without_a_context_or_data_dir_the_rig_token_uses_the_home_folder_file(monkeypatch):
    monkeypatch.delenv("DISPATCH_DATA_DIR", raising=False)
    monkeypatch.delenv("LOCAL_CHAT_DATA_DIR", raising=False)
    monkeypatch.setattr(context, "get_context", lambda: None)
    from forge_voice_core import rig
    assert rig.token_path() == rig.legacy_token_path() == Path.home() / ".config" / "local-chat" / "speechforge.env"


def test_the_bot_voice_and_settings_reach_the_engines_through_the_link(fake_ctx):
    context.set_context(fake_ctx)
    try:
        from dispatch_voice.host import config
        assert hostlink.get().load_settings().__class__ is config.VoiceSettings
        assert hostlink.get().bot_voice("sage") == context.bot_voice("sage")
        assert hostlink.get().secret_store() is None or hasattr(hostlink.get().secret_store(), "get")
    finally:
        context.clear_context()


def test_the_settings_table_is_still_one_table_with_the_engine_rows_defined_once():
    from dispatch_voice.host import settings_schema as S
    from forge_voice_core import settings_schema as core
    assert S.rt_ranges() == core.rt_ranges() and S.rig_ranges() == core.rig_ranges()
    engine_ids = set(core.ENGINE_BY_ID)
    assert engine_ids <= set(S.BY_ID)
    assert all(S.BY_ID[i] is core.ENGINE_BY_ID[i] for i in engine_ids)           # the same row objects
    assert len(S.SETTINGS) == 92


def test_python_dash_m_download_still_works_through_the_shim():
    r = subprocess.run([sys.executable, "-m", "dispatch_voice.host.download", "--help"], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0 and "--dest" in r.stdout, r.stderr
