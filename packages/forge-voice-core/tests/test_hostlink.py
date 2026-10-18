"""The host link: the standalone defaults, and a host overriding them."""
from __future__ import annotations

from pathlib import Path

from forge_voice_core import fetch_job, hostlink, models, profile_bridge, rig


class _MyHost(hostlink.HostLink):
    client_name = "chatforge"
    client_version = "9.9"

    def __init__(self, root: Path):
        self.root = root
        self.store = {}

    def models_root(self):
        return self.root / "m"

    def voices_dir(self):
        return self.root / "v"

    def state_dir(self):
        return self.root / "s"

    def secret_store(self):
        outer = self

        class S:
            def get(self, name):
                return outer.store.get(name)

            def set(self, name, value):
                outer.store[name] = value

            def delete(self, name):
                outer.store.pop(name, None)
        return S()

    def router_mounted(self):
        return True


def test_the_standalone_default_lives_under_forge_voice_data_dir(tmp_path):
    h = hostlink.get()
    base = tmp_path / "fvc-data"                                    # the autouse fixture's FORGE_VOICE_DATA_DIR
    assert h.data_dir() == base
    assert h.models_root() == base / "models" and h.voices_dir() == base / "voices"
    assert h.state_dir() == base / "state" and h.has_state_dir()
    assert h.secret_store() is None and h.power() is None and h.router_mounted() is False
    assert h.bot_voice("any") == "" and h.client_name == "forge-voice-core"


def test_explicit_folders_beat_the_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_VOICE_MODELS_DIR", str(tmp_path / "elsewhere"))
    assert hostlink.get().models_root() == tmp_path / "elsewhere"


def test_install_returns_the_previous_link_and_reset_restores_the_default(tmp_path):
    mine = _MyHost(tmp_path)
    prev = hostlink.install(mine)
    assert hostlink.get() is mine and type(prev) is hostlink.HostLink
    hostlink.reset()
    assert type(hostlink.get()) is hostlink.HostLink


def test_models_follow_the_installed_host(tmp_path):
    hostlink.install(_MyHost(tmp_path))
    assert models.root() == tmp_path / "m"
    assert models.speech_dir() == tmp_path / "m" / "speech"
    assert models.chatterbox_dir() == tmp_path / "m" / "chatterbox"


def test_the_profile_bridge_stores_voices_where_the_host_says(tmp_path):
    hostlink.install(_MyHost(tmp_path))
    st = profile_bridge.store()
    assert Path(st.root) == tmp_path / "v"
    assert profile_bridge.store() is st                              # cached while the folder is the same
    hostlink.install(_MyHost(tmp_path / "other"))
    assert Path(profile_bridge.store().root) == tmp_path / "other" / "v"
    assert profile_bridge.stock_id() == "stock-host"


def test_the_rig_token_goes_to_the_hosts_credential_store_first(tmp_path, monkeypatch):
    host = _MyHost(tmp_path)
    hostlink.install(host)
    monkeypatch.delenv(rig.TOKEN_ENV, raising=False)
    assert rig.token_path() == tmp_path / "s" / "speechforge.env"
    rig.write_token("sekret-token-123")
    assert host.store                                                # held by the host's store ...
    assert rig.read_token().value == "sekret-token-123" and rig.read_token().source != "env"
    assert not (tmp_path / "s" / "speechforge.env").exists()         # ... and no token file was written
    assert "sekret-token" not in repr(rig.read_token())
    assert rig.clear_token()
    assert rig.read_token().value == ""


def test_the_download_job_asks_the_host_whether_a_router_needs_a_restart(tmp_path):
    hostlink.install(_MyHost(tmp_path))
    assert fetch_job._router_mounted() is True
    hostlink.reset()
    assert fetch_job._router_mounted() is False


def test_the_default_ensure_engine_is_false_without_models_and_registers_with_them(monkeypatch):
    from forge_voice_core import registry
    from forge_voice_core.engines import chatterbox
    registry.reset()
    assert hostlink.get().ensure_engine() is False
    monkeypatch.setattr(chatterbox, "available", lambda: True)
    assert hostlink.get().ensure_engine() is True
    assert registry.engine_spec("chatterbox") is not None
    registry.reset()


def test_the_default_settings_object_is_complete_enough_for_the_health_report(tmp_path):
    from forge_voice_core import health
    cfg = hostlink.get().load_settings()
    assert cfg.models_dir == tmp_path / "fvc-data" / "models" / "speech"
    rows = health.report(cfg)
    assert "models" in {r["id"] for r in rows}
    assert all({"id", "state", "reason", "fix", "code"} <= set(r) for r in rows)


def test_the_logger_family_can_be_renamed_before_import():
    from forge_voice_core import _log
    assert _log.get().name == _log.NAME
    assert _log.get("opus").name == f"{_log.NAME}.opus"
    saved = (_log.NAME, dict(_log.OVERRIDES))
    try:
        _log.set_name("my.app", {"opus": "my.opus"})
        assert _log.get().name == "my.app" and _log.get("opus").name == "my.opus"
        assert _log.get("x").name == "my.app.x"
    finally:
        _log.set_name(saved[0], saved[1])


async def test_the_rig_session_reports_the_hosts_version_or_the_package_one(monkeypatch):
    import forge_voice_core
    from forge_voice_core.engine_settings import EngineSettings
    monkeypatch.setenv(rig.TOKEN_ENV, "a-test-token-1")
    s = EngineSettings(rig=rig.RigSettings(enabled=True, url="ws://127.0.0.1:19108/v1/realtime"))
    versions = []
    for host_version in ("", "9.9"):
        class H(hostlink.HostLink):
            client_version = host_version
        hostlink.install(H())
        sess = rig.open_for_session(s, connect=_never_connect)
        assert sess is not None
        versions.append(sess.conn.version)
        await sess.conn.close()
    assert versions == [forge_voice_core.__version__, "9.9"]


async def _never_connect(url, token):                                # the connection is built, never dialled
    raise OSError("no network in tests")
