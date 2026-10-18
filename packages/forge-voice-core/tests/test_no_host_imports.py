"""forge_voice_core must import without DisPatch Voice, FastAPI or Starlette.

The guarantee a second host (ChatForge) depends on: the engines never reach for the application they
were extracted from, nor for a web framework. Enforced with an import-blocking fixture: a meta-path
finder that refuses those names, with every ``forge_voice_core`` module imported fresh underneath it.
"""
from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys

import pytest

import forge_voice_core

BLOCKED = ("dispatch_voice", "fastapi", "starlette")


class _Blocker:
    """A meta-path finder that raises ImportError for the blocked top-level names."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is blocked: forge_voice_core must not need it")
        return None


@pytest.fixture
def block_host_imports():
    """Hide dispatch_voice / fastapi / starlette and every already-imported forge_voice_core module, so the
    package is imported from scratch under the block. Everything is put back afterwards."""
    saved = {n: m for n, m in sys.modules.items()
             if n.split(".")[0] in BLOCKED or n == "forge_voice_core" or n.startswith("forge_voice_core.")}
    for n in saved:
        del sys.modules[n]
    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield blocker
    finally:
        sys.meta_path.remove(blocker)
        for n in [n for n in sys.modules if n == "forge_voice_core" or n.startswith("forge_voice_core.")]:
            del sys.modules[n]
        sys.modules.update(saved)


def _all_modules() -> list[str]:
    pkg = importlib.import_module("forge_voice_core")
    return ["forge_voice_core"] + [m.name for m in pkgutil.walk_packages(pkg.__path__, "forge_voice_core.")]


def test_every_module_imports_with_the_host_and_the_web_framework_blocked(block_host_imports):
    names = _all_modules()
    assert len(names) > 30, names                                   # the walk found the package
    for name in names:
        importlib.import_module(name)
    leaked = sorted(n for n in sys.modules if n.split(".")[0] in BLOCKED)
    assert leaked == []


def test_the_blocker_really_blocks(block_host_imports):
    for name in BLOCKED:
        with pytest.raises(ImportError):
            importlib.import_module(name)


def test_importing_the_package_loads_nothing_heavy():
    """``import forge_voice_core`` (and the host link) must stay cheap: no numpy, no onnxruntime."""
    code = ("import sys, forge_voice_core\n"
            "from forge_voice_core import hostlink, _log\n"
            "bad = [m for m in ('numpy', 'onnxruntime', 'onnx_asr', 'httpx', 'websockets', 'fastapi', 'dispatch_voice')"
            " if m in sys.modules]\n"
            "print(','.join(bad)); sys.exit(1 if bad else 0)\n")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_rig_module_imports_without_the_speechforge_extra(block_host_imports, monkeypatch):
    """websockets is imported when the rig connects, never at import: the base install has no rig dependency."""
    monkeypatch.setitem(sys.modules, "websockets", None)             # None in sys.modules = ImportError
    importlib.import_module("forge_voice_core.rig")
    importlib.import_module("forge_voice_core.providers")
