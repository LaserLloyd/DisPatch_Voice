"""The default profile bridge: the voice-profile store under ``hostlink.voices_dir()`` and the stock voice.

The Chatterbox engine speaks with a :class:`~forge_voice_core.profiles.ProfileStore` (cloned voices,
each a folder with ``voice.npz``) and, when nobody has cloned a voice, the bundled stock voice. A host
that stores its profiles somewhere else, or wants other rules for which profile speaks, returns its own
object with the same four functions from ``HostLink.profiles()``.

DisPatch Voice does (``dispatch_voice.host.profiles_bridge``: bots, the rig-aware delete, the router).
"""
from __future__ import annotations

import threading
from pathlib import Path

from . import _log, hostlink

log = _log.get()

_STORE = None
_STORE_LOCK = threading.Lock()


def voices_dir() -> Path:
    return hostlink.get().voices_dir()


def store():
    """The ProfileStore for the folder in force (rebuilt when it moves). The engine behind it loads on
    first use, never here."""
    global _STORE
    with _STORE_LOCK:
        root = voices_dir()
        if _STORE is None or Path(_STORE.root) != root:
            from .engines.chatterbox import LazyEngine
            from .profiles import ProfileStore
            _STORE = ProfileStore(root, LazyEngine())
        return _STORE


def release_engine() -> None:
    """Drop the loaded Chatterbox sessions; the lazy engine reloads on use."""
    from .engines.chatterbox import LazyEngine
    with _STORE_LOCK:
        st = _STORE
    eng = getattr(st, "engine", None) if st is not None else None
    if isinstance(eng, LazyEngine):
        with eng._lock:
            eng._eng = None


def stock_id() -> str | None:
    from . import stock
    return stock.STOCK_ID


def stock_profile(build: bool = False) -> str | None:
    """The stock profile's id once it exists in the store, else None. ``build=False`` only copies the
    tensors ``fetch-models`` pre-built (cheap; never loads the engine); ``build=True`` may load the engine."""
    from . import stock
    try:
        return stock.ensure_stock(store(), build=build)
    except Exception:       # noqa: BLE001 - the stock voice is a convenience, never a crash
        log.exception("voice: the stock voice could not be prepared")
        return None
