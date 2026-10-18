"""System requirements (ffmpeg, libopus): declared in pyproject, checked with an install hint for
the operating system, shown by `dispatch-voice doctor` and the status health."""
from __future__ import annotations

import tomllib
from pathlib import Path

from dispatch_voice import cli
from dispatch_voice.host import sysreq

ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_declares_exactly_the_checked_requirements():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = meta["tool"]["dispatch-voice"]["system-requirements"]
    assert set(declared) == {r.id for r in sysreq.REQUIREMENTS} == {"ffmpeg", "libopus"}
    assert all(isinstance(v, str) and v for v in declared.values())


def test_websockets_is_a_runtime_dependency():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert any(d.startswith("websockets") for d in meta["project"]["dependencies"])


def test_every_requirement_has_a_hint_for_every_supported_package_manager():
    for r in sysreq.REQUIREMENTS:
        assert {"apt", "dnf", "pacman", "brew", "winget"} <= set(r.hints), r.id


def test_package_manager_detection():
    assert sysreq.package_manager("win32") == "winget" and sysreq.package_manager("darwin") == "brew"
    assert sysreq.package_manager("linux", 'ID=ubuntu\nID_LIKE="debian"\n') == "apt"
    assert sysreq.package_manager("linux", "ID=fedora\n") == "dnf"
    assert sysreq.package_manager("linux", 'ID="rocky"\nID_LIKE="rhel centos fedora"\n') == "dnf"
    assert sysreq.package_manager("linux", "ID=arch\n") == "pacman"
    assert sysreq.package_manager("linux", "ID=opensuse-tumbleweed\nID_LIKE=suse\n") == "zypper"
    assert sysreq.package_manager("linux", "ID=alpine\n") == "apk"
    assert sysreq.package_manager("linux", "") == "apt"                  # nothing known: the most common one


def test_check_reports_a_fix_only_for_what_is_missing(monkeypatch):
    reqs = tuple(sysreq.Requirement(r.id, r.what, r.required_for, r.optional,
                                    (lambda: "found") if r.id == "ffmpeg" else (lambda: None), r.hints)
                 for r in sysreq.REQUIREMENTS)
    monkeypatch.setattr(sysreq, "REQUIREMENTS", reqs)
    got = {r["id"]: r for r in sysreq.check("apt")}
    assert got["ffmpeg"]["ok"] and got["ffmpeg"]["fix"] == ""
    assert not got["libopus"]["ok"] and got["libopus"]["fix"] == "sudo apt install libopus0"
    assert got["libopus"]["optional"] and "path" not in str(got)          # no filesystem paths in the report


def test_doctor_prints_the_install_command(monkeypatch, capsys):
    rep = {"version": "1", "checkout": None, "extension": {"id": "voice", "api": [1, 1], "setup": "x"},
           "dispatch": {"importable": False, "ext_api": None, "compatible": None, "source": None},
           "speech_models": {"dir": "d", "missing": []}, "chatterbox_models": {"dir": "d", "ok": True},
           "stock_voice": {"tensors": True}, "python_packages_missing": [],
           "plugins": {"declared": [], "loaded": [], "routers": []}, "engines": [], "builtin_voices": 0,
           "provider_plugins": [{"name": "kokoro", "target": "x:setup", "enabled": False}],
           "switch": {"enabled": True, "forced": None, "voice_yaml": "v"}, "rig": None,
           "running": {"url": "u", "reachable": False},
           "system": [{"id": "ffmpeg", "ok": False, "optional": False, "required_for": "cloning a voice",
                       "fix": "winget install Gyan.FFmpeg", "package_manager": "winget"},
                      {"id": "libopus", "ok": True, "optional": True, "required_for": "x", "fix": "",
                       "package_manager": "winget"}]}
    rc = cli.print_doctor(rep)
    out = capsys.readouterr().out
    assert "system program ffmpeg: MISSING (needed for cloning a voice)" in out
    assert "install (winget): winget install Gyan.FFmpeg" in out and "system program libopus: ok" in out
    assert "warnings: 1" in out and "problems: 0" in out and rc == 0
    assert "provider plugins installed but off: kokoro (switch them on in Settings > Voice)" in out
    assert "plugins: none switched on" in out
