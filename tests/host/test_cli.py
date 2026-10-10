"""``dispatch-voice fetch-models`` (every core model, verify mode) and
``dispatch-voice doctor`` (the handshake, the models, the engines, the
running-vs-checkout warning, the offline-overlay probe, the update check),
with every network call replaced."""
from __future__ import annotations

import hashlib
import json

import pytest

from dispatch_voice import __version__, cli
from dispatch_voice.host import download
from dispatch_voice.host.config import VoiceSettings
from dispatch_voice.host.engines import chatterbox

NOT_REACHABLE = {"url": "http://127.0.0.1:1", "reachable": False, "detail": "ConnectError"}


@pytest.fixture
def env(tmp_path, monkeypatch, host_env):
    monkeypatch.setattr(cli, "probe_running", lambda url: dict(NOT_REACHABLE, url=url))
    monkeypatch.setattr(cli, "_run", lambda argv, cwd, timeout: (1, "stubbed: no uv in tests"))
    monkeypatch.setattr(chatterbox, "available", lambda: False)
    monkeypatch.setenv("DISPATCH_SPEECH_MODELS", str(tmp_path / "speech"))
    return tmp_path


# --------------------------------------------------------------------------- #
# fetch-models
# --------------------------------------------------------------------------- #

def test_verify_reports_every_missing_speech_model(tmp_path):
    lines = []
    assert download.fetch_speech_models(tmp_path, verify=True, echo=lines.append) == 1
    missing = [ln for ln in lines if "MISSING" in ln]
    assert len(missing) == len(download.FILES) and any("silero_vad.onnx" in ln for ln in missing)


def test_fetch_then_have_then_corrupt(tmp_path, monkeypatch):
    sha = hashlib.sha256(b"hello").hexdigest()
    monkeypatch.setattr(download, "FILES", [("a/one.bin", "http://example.invalid/one", sha)])
    fetched = []

    def fake_fetch(url, dest, sha256):
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"hello")

    lines = []
    assert download.fetch_speech_models(tmp_path, fetch=fake_fetch, echo=lines.append) == 0
    assert fetched == ["http://example.invalid/one"] and (tmp_path / "a/one.bin").read_bytes() == b"hello"
    lines.clear()
    assert download.fetch_speech_models(tmp_path, fetch=fake_fetch, echo=lines.append) == 0
    assert any(ln.strip().startswith("have") for ln in lines) and len(fetched) == 1
    (tmp_path / "a/one.bin").write_bytes(b"corrupt")
    lines.clear()
    assert download.fetch_speech_models(tmp_path, verify=True, echo=lines.append) == 1
    assert any("BAD" in ln for ln in lines)
    assert download.fetch_speech_models(tmp_path, force=True, fetch=fake_fetch, echo=lines.append) == 0
    assert (tmp_path / "a/one.bin").read_bytes() == b"hello"


def test_a_failed_download_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "FILES", [("x.bin", "http://example.invalid/x", "0" * 64)])

    def boom(url, dest, sha256):
        raise download.ChecksumError("sha256 mismatch")

    lines = []
    assert download.fetch_speech_models(tmp_path, fetch=boom, echo=lines.append) == 1
    assert any("FAILED x.bin" in ln for ln in lines) and not (tmp_path / "x.bin").exists()


def test_cli_fetch_models_verify_exit_code_and_output(env, capsys):
    rc = cli.main(["fetch-models", "--verify", "--speech-models", str(env / "speech")])
    out = capsys.readouterr().out
    assert rc == 1 and "MISSING" in out and "Chatterbox models" in out and "MISSING" in out.splitlines()[-1]


def test_cli_fetch_models_only_speech_with_a_fake_fetcher(env, monkeypatch, capsys):
    sha = hashlib.sha256(b"hello").hexdigest()
    monkeypatch.setattr(download, "FILES", [("one.bin", "http://example.invalid/one", sha)])
    def fake_fetch(url, dest, sha256):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"hello")

    monkeypatch.setattr(download, "_fetch", fake_fetch)
    rc = cli.main(["fetch-models", "--only", "speech", "--speech-models", str(env / "speech")])
    assert rc == 0 and (env / "speech" / "one.bin").exists()
    assert "fetch one.bin" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #

def test_doctor_report_without_dispatch_lists_what_is_missing(env, capsys):
    rep = cli.doctor_report(backend=None, url="http://127.0.0.1:1", check_update=False)
    assert rep["version"] == __version__ and rep["extension"]["id"] == "voice"
    assert rep["dispatch"]["importable"] is False                 # the add-on's own env
    assert rep["speech_models"]["missing"] and rep["chatterbox_models"]["ok"] is False
    assert rep["running"]["reachable"] is False and "backend" not in rep and "update" not in rep
    assert cli.print_doctor(rep) == 1
    out = capsys.readouterr().out
    assert "MISSING" in out and "not importable" in out and "not reachable" in out
    assert out.strip().endswith("problems: 2")                    # the speech models and Chatterbox


def test_doctor_all_good_is_zero(env, monkeypatch, capsys):
    monkeypatch.setattr(VoiceSettings, "missing", lambda self: [])
    monkeypatch.setattr(chatterbox, "available", lambda: True)
    tensors = env / "stock.npz"
    tensors.write_bytes(b"")
    monkeypatch.setattr("dispatch_voice.stock.cached_tensors_path", lambda *a, **k: tensors)
    monkeypatch.setattr(cli, "probe_running", lambda url: {
        "url": url, "reachable": True, "status_code": 200,
        "voice": {"id": "voice", "version": __version__, "state": "ok", "api": [1, 1]}})
    rep = cli.doctor_report(backend=None, url="http://127.0.0.1:1", check_update=False)
    assert cli.print_doctor(rep) == 0
    out = capsys.readouterr().out
    assert "problems: 0" in out and f"voice {__version__} ok" in out and "restart" not in out
    assert "chatterbox" in rep["engines"]


def test_doctor_reads_ext_api_from_a_backend_dir_without_importing_it(env, capsys):
    backend = env / "backend"
    (backend / "app").mkdir(parents=True)
    (backend / "app" / "extensions.py").write_text("import nothing_importable\nEXT_API = 1   # handshake\n")
    d = cli.probe_dispatch_import(backend)
    assert d == {"importable": False, "ext_api": 1, "compatible": True, "source": "file"}
    (backend / "app" / "extensions.py").write_text("EXT_API = 7\n")
    assert cli.probe_dispatch_import(backend)["compatible"] is False
    assert cli.probe_dispatch_import(env / "nowhere") == {"importable": False, "ext_api": None,
                                                           "compatible": None, "source": None}
    rep = cli.doctor_report(backend=backend, url="http://127.0.0.1:1", check_update=False)
    cli.print_doctor(rep)
    out = capsys.readouterr().out
    assert "EXT_API 7 -> compatible: no" in out and "app/extensions.py" in out


def test_doctor_names_a_backend_that_predates_the_extension_host(env, capsys):
    backend = env / "old-backend"
    (backend / "app").mkdir(parents=True)
    rep = cli.doctor_report(backend=backend, url="http://127.0.0.1:1", check_update=False)
    assert rep["dispatch"]["ext_api"] is None
    cli.print_doctor(rep)
    assert "no app/extensions.py" in capsys.readouterr().out


def test_doctor_warns_when_the_running_version_differs(env, monkeypatch, capsys):
    monkeypatch.setattr(cli, "probe_running", lambda url: {
        "url": url, "reachable": True, "status_code": 200,
        "voice": {"id": "voice", "version": "0.9.0", "state": "ok", "api": [1, 1]}})
    cli.print_doctor(cli.doctor_report(backend=None, url="http://127.0.0.1:1", check_update=False))
    assert "running 0.9.0, checkout" in capsys.readouterr().out


def test_doctor_backend_probes_use_fixed_argv_offline_and_without_sync(env, monkeypatch):
    seen = []

    def fake_run(argv, cwd, timeout):
        seen.append((argv, cwd))
        return (0, "1.0.0\n") if "--with-editable" not in argv else (1, "error: network unreachable")

    monkeypatch.setattr(cli, "_run", fake_run)
    out = cli.probe_backend(env / "backend", env / "checkout")
    assert out["installed_in_venv"] == {"ok": True, "version": "1.0.0", "detail": None}
    assert out["overlay_resolvable_offline"]["ok"] is False
    assert "network unreachable" in out["overlay_resolvable_offline"]["detail"]
    for argv, cwd in seen:
        assert cwd == env / "backend" and argv[0].endswith("uv")
        assert argv[1:5] == ["run", "--offline", "--no-sync"] or argv[1:4] == ["run", "--offline", "--no-sync"]
        assert "python" in argv and "-c" in argv
    assert any("--with-editable" in argv and str(env / "checkout") in argv for argv, _ in seen)


def test_doctor_update_check_is_fail_soft(monkeypatch):
    import urllib.request

    def boom(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert cli.probe_update()["error"] == "OSError"

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            return b"1.2.0\n"

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Resp())
    assert cli.probe_update() == {"checkout": __version__, "latest": "1.2.0"}


def test_cli_doctor_json_and_text(env, monkeypatch, capsys):
    monkeypatch.setattr(cli, "probe_update", lambda: {"checkout": __version__, "latest": __version__})
    assert cli.main(["doctor", "--json", "--url", "http://127.0.0.1:1", "--check-update"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["update"]["latest"] == __version__ and rep["running"]["reachable"] is False
    rc = cli.main(["doctor", "--url", "http://127.0.0.1:1", "--check-update"])
    out = capsys.readouterr().out
    assert rc == 1 and "up to date" in out and out.startswith(f"dispatch-voice {__version__}")


def test_doctor_counts_a_failed_plugin(env, monkeypatch, capsys):
    monkeypatch.setenv("DISPATCH_VOICE_EXTENSIONS", "no_such_plugin_xyz:setup")
    rep = cli.doctor_report(backend=None, url="http://127.0.0.1:1", check_update=False)
    assert rep["plugins"] == {"declared": ["no_such_plugin_xyz:setup"], "loaded": [], "routers": []}
    cli.print_doctor(rep)
    assert "FAILED ['no_such_plugin_xyz:setup']" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Review fixes (review-T3s1 findings 6 and 9)
# --------------------------------------------------------------------------- #

def test_doctor_loads_a_plugin_that_mounts_routes(env, monkeypatch, capsys, tmp_path):
    """Finding 6: `api.include_router` needs DisPatch's context to mount, but
    a plugin that calls it is a plugin that LOADS; the doctor records the
    mount instead of counting the plugin as failed."""
    import sys
    pkg = tmp_path / "doctor_plugin_xyz.py"
    pkg.write_text("from fastapi import APIRouter\n"
                   "def setup(api):\n"
                   "    r = APIRouter()\n"
                   "    @r.get('/ping')\n"
                   "    def ping(): return {'ok': True}\n"
                   "    api.include_router(r, prefix='/xyz')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("doctor_plugin_xyz", None)
    monkeypatch.setenv("DISPATCH_VOICE_EXTENSIONS", "doctor_plugin_xyz:setup")
    rep = cli.doctor_report(backend=None, url="http://127.0.0.1:1", check_update=False)
    assert rep["plugins"] == {"declared": ["doctor_plugin_xyz:setup"], "loaded": ["doctor_plugin_xyz:setup"],
                              "routers": ["/api/voice/xyz"]}
    cli.print_doctor(rep)
    out = capsys.readouterr().out
    assert "FAILED" not in out and "routes under /api/voice/xyz (mounted inside DisPatch)" in out


def test_doctor_hint_fits_the_overlay_deployment(env, monkeypatch, capsys):
    """Finding 9: on an overlay deployment (`--with-editable`, not installed in
    the venv) the "re-run the install line" advice is wrong; say what is true."""
    def fake_run(argv, cwd, timeout):
        return (0, "1.0.0\n") if "--with-editable" in argv else (1, "No module named dispatch_voice")

    monkeypatch.setattr(cli, "_run", fake_run)
    rep = cli.doctor_report(backend=env / "backend", url="http://127.0.0.1:1", check_update=False)
    cli.print_doctor(rep)
    out = capsys.readouterr().out
    assert "installed in its venv: no  (not installed; the --with-editable overlay resolves" in out
    assert "re-run the install line" not in out and "overlay resolvable offline: yes" in out
    # Neither way works: the install hint is the right one.
    monkeypatch.setattr(cli, "_run", lambda argv, cwd, timeout: (1, "no"))
    rep = cli.doctor_report(backend=env / "backend", url="http://127.0.0.1:1", check_update=False)
    cli.print_doctor(rep)
    assert "re-run the install line" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# doctor --unit (V1 #5) and the standalone serve page (FR-hardening L6)
# --------------------------------------------------------------------------- #

def test_doctor_reads_only_the_voice_settings_from_the_unit(env, monkeypatch, capsys):
    """Run without the unit's environment the doctor looked in the default data dir
    and reported the models missing and no plugins; --unit reads them (and the
    backend directory) from `systemctl --user show`, never PATH or a key."""
    import subprocess as sp
    calls = []
    out = ("Environment=PATH=/usr/bin HOME=/srv/user DISPATCH_SPEECH_MODELS=/srv/models "
           "DISPATCH_VOICE_EXTENSIONS=dispatch_voice_kokoro:setup SOME_API_KEY=not-read\n"
           "WorkingDirectory=/srv/dispatch/backend\n")

    def fake_run(argv, **kw):
        calls.append(argv)
        return sp.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    env_, wd = cli.unit_environment("local-chat")
    assert calls[0][:4] == ["systemctl", "--user", "show", "local-chat"]
    assert env_ == {"DISPATCH_SPEECH_MODELS": "/srv/models",
                    "DISPATCH_VOICE_EXTENSIONS": "dispatch_voice_kokoro:setup"}
    assert wd == "/srv/dispatch/backend"
    with pytest.raises(ValueError):
        cli.unit_environment("local-chat; rm -rf /")


def test_serve_refuses_without_its_own_data_dir(host_env, capsys):
    """The standalone page has no login: it never opens DisPatch's voice store."""
    assert cli.main(["serve"]) == 2
    assert "pass --data DIR" in capsys.readouterr().err
    live = host_env.paths.voices_dir
    live.mkdir(parents=True, exist_ok=True)
    assert cli.main(["serve", "--data", str(live)]) == 2
    assert "DisPatch's own voice store" in capsys.readouterr().err


def test_standalone_page_answers_only_its_own_loopback_host(tmp_path):
    """DNS rebinding: a page on a hostile name that resolves to 127.0.0.1 sends its own
    name as Host; it gets 421, even for a GET of a recorded voice."""
    from fastapi.testclient import TestClient

    from dispatch_voice.profiles import ProfileStore
    store = ProfileStore(tmp_path / "voices", None)
    app = cli.standalone_app(store, 8769)
    c = TestClient(app, base_url="http://127.0.0.1:8769")
    assert c.get("/voices", follow_redirects=False).status_code == 200
    assert c.get("/voices", headers={"host": "localhost:8769"}).status_code == 200
    r = c.get("/voices", headers={"host": "evil.example:8769"})
    assert r.status_code == 421
    assert c.get("/voices/someone/files/reference.wav", headers={"host": "evil.example"}).status_code == 421
