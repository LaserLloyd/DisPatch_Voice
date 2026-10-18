"""Moved to ``forge_voice_core.download``: this path is a thin alias so existing imports keep working.

The module object IS the core module (``sys.modules`` aliasing), so ``import dispatch_voice.host.download`` and
``import forge_voice_core.download`` are the same object and patching either one patches both.
"""
# DEPRECATED import path (forge-voice-core extraction): import from ``forge_voice_core.download`` in new code.
import importlib as _importlib
import sys as _sys

_impl = _importlib.import_module("forge_voice_core.download")
if __name__ != "__main__":
    _sys.modules[__name__] = _impl
else:                           # ``python -m dispatch_voice.host.download`` keeps working
    _sys.exit(_impl.main())
