"""Moved to ``forge_voice_core.engines.chatterbox``: this path is a thin alias so existing imports keep working.

The module object IS the core module (``sys.modules`` aliasing), so ``import dispatch_voice.host.engines.chatterbox`` and
``import forge_voice_core.engines.chatterbox`` are the same object and patching either one patches both.
"""
# DEPRECATED import path (forge-voice-core extraction): import from ``forge_voice_core.engines.chatterbox`` in new code.
import importlib as _importlib
import sys as _sys

_impl = _importlib.import_module("forge_voice_core.engines.chatterbox")
if __name__ != "__main__":
    _sys.modules[__name__] = _impl
