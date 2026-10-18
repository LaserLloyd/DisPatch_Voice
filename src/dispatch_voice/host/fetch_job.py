"""Moved to ``forge_voice_core.fetch_job``: this path is a thin alias so existing imports keep working.

The module object IS the core module (``sys.modules`` aliasing), so ``import dispatch_voice.host.fetch_job`` and
``import forge_voice_core.fetch_job`` are the same object and patching either one patches both.
"""
# DEPRECATED import path (forge-voice-core extraction): import from ``forge_voice_core.fetch_job`` in new code.
import importlib as _importlib
import sys as _sys

_impl = _importlib.import_module("forge_voice_core.fetch_job")
if __name__ != "__main__":
    _sys.modules[__name__] = _impl
