"""Moved to ``forge_voice_core.stock``: this path is a thin alias so existing imports keep working.

The module object IS the core module (``sys.modules`` aliasing), so ``import dispatch_voice.stock`` and
``import forge_voice_core.stock`` are the same object and patching either one patches both.
"""
# DEPRECATED import path (forge-voice-core extraction): import from ``forge_voice_core.stock`` in new code.
import importlib as _importlib
import sys as _sys

_impl = _importlib.import_module("forge_voice_core.stock")
if __name__ != "__main__":
    _sys.modules[__name__] = _impl
