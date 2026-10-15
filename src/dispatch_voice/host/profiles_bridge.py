"""Voice profiles in the host: the profile store, the stock voice, which bot
speaks with which profile, the profile fillers, and the Voices API router
(the store half of what used to be ``host/dv.py``).

* The profile store lives at ``ctx.paths.voices_dir`` (``<data>/voices/``).
  Personal data: never in the repo, a deploy or a zip. The engine behind it
  is :class:`engines.chatterbox.LazyEngine` (loads on first use).
* :func:`install` registers the ``chatterbox`` engine and the profile
  resolver with ``registry`` and returns the profile router for
  ``/api/voice/profiles`` (contracts §D: moved from ``/api/voices``,
  no alias; the standalone page's ``/ui`` routes stay in ``dispatch-voice
  serve`` only, amendment A12). With the Chatterbox models missing nothing is
  registered and None is returned: ``/api/voice/status.missing`` says what to
  run, and a restart after ``fetch-models`` mounts it.
* The resolver: the bot's ``voice:`` (config.yaml, read through the context)
  if that profile exists, else the store's default profile; with no profile
  and no built-in voice plugin, the stock voice once it exists.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from . import context

log = logging.getLogger("dispatch.voice")

SAMPLE_RATE = 24000

_STORE = None
_STORE_LOCK = threading.Lock()


def voices_dir() -> Path:
    return context.paths().voices_dir


def store():
    """The ProfileStore for the data dir in force (rebuilt when it moves)."""
    global _STORE
    with _STORE_LOCK:
        root = voices_dir()
        if _STORE is None or Path(_STORE.root) != root:
            from dispatch_voice.profiles import ProfileStore

            from .engines.chatterbox import LazyEngine
            _STORE = ProfileStore(root, LazyEngine())
        return _STORE


class LazyStore:
    """Forwards to :func:`store` on first use, so nothing touches the data dir
    until a request actually needs a profile."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(store(), name)


class _RigAwareStore(LazyStore):
    """The Voices manager's store. Deleting a profile also deletes its voices from the
    GPU rig (SPEC §8.4 deletion sync) when a rig address and token are set: its
    conditionals and every voice the rig made from its recording."""

    def delete(self, pid: str) -> None:
        from . import rig
        ids = rig.ids_for_delete(store(), pid)
        store().delete(pid)
        for vid in ids:
            rig.delete_voice_async(vid)


def release_engine() -> None:
    """Drop the loaded Chatterbox sessions; the lazy engine reloads on use."""
    from .engines.chatterbox import LazyEngine
    with _STORE_LOCK:
        st = _STORE
    eng = getattr(st, "engine", None) if st is not None else None
    if isinstance(eng, LazyEngine):
        with eng._lock:
            eng._eng = None
    # Fillers not rendered yet are rendered on the next use, not now.
    _EXTRAS_STARTED.discard(str(voices_dir()))


# --------------------------------------------------------------------------- #
# The stock voice
# --------------------------------------------------------------------------- #

def _stock_mod():
    from dispatch_voice import stock
    return stock


def stock_id() -> str | None:
    return _stock_mod().STOCK_ID


def stock_profile(build: bool = False) -> str | None:
    """The stock profile's id once it exists in the store, else None.

    ``build=False`` only copies the tensors ``dispatch-voice fetch-models``
    pre-built (cheap; never loads the engine). ``build=True`` may load the
    engine and build them (~1 s plus the engine load)."""
    s = _stock_mod()
    try:
        pid = s.ensure_stock(store(), build=build)
    except Exception:
        log.exception("voice: the stock voice could not be prepared")
        return None
    if pid and build:
        _render_stock_extras(pid)
    return pid


_EXTRAS_STARTED: set[str] = set()


def _render_stock_extras(pid: str) -> None:
    """The stock voice's audition line, once, in the background (its fillers
    are never played inside DisPatch, so they are not rendered here)."""
    s = _stock_mod()
    key = str(voices_dir())
    if key in _EXTRAS_STARTED:
        return
    _EXTRAS_STARTED.add(key)

    def run():
        from .config import load_settings
        from .tts import REPLY_ACTIVITY
        REPLY_ACTIVITY.wait_idle()
        if not load_settings().switched_on:
            # Voice was switched off while this waited: rendering now would
            # load the engine again right after release() gave it back.
            _EXTRAS_STARTED.discard(key)
            return
        try:
            s.render_extras(store(), pid, fillers=False)    # acks are the render queue's (C12)
        except Exception:
            log.exception("voice: rendering the stock voice's fillers failed")

    threading.Thread(target=run, name="voice-stock-extras", daemon=True).start()


# --------------------------------------------------------------------------- #
# Which profile speaks
# --------------------------------------------------------------------------- #

def resolve_profile(bot_id: str | None, voice: str | None = None) -> str | None:
    """bot.voice if that profile exists (or names a built-in voice while a
    provider is installed), else the store default, else, with no built-in
    voice plugin, the stock voice once it exists, else None."""
    from . import registry
    from .tts import builtin_id
    st = store()
    want = voice or context.bot_voice(bot_id)
    if builtin_id(want):
        if registry.has_builtin_voices():
            return want
        want = ""                      # provider gone: the default voice speaks
    if want:
        if want == stock_id():
            return want
        try:
            st.get(want)
            return want
        except Exception:
            log.warning("voice: bot %s names voice %r, which does not exist", bot_id, want)
    default = st.default_id()
    if default:
        return default
    if not registry.has_builtin_voices():
        return stock_profile(build=False)
    return None


def profiles() -> list[dict]:
    """[{id, name}] of every voice profile (newest first)."""
    return [{"id": m["id"], "name": m.get("name") or m["id"]} for m in store().list()]


def profile_exists(pid: str) -> bool:
    if pid and pid == stock_id():
        return True                    # built on first use
    try:
        store().get(pid)
        return True
    except Exception:
        return False


def default_profile() -> str | None:
    return store().default_id()


def set_default_profile(pid: str | None) -> None:
    """Make ``pid`` the default profile, or (None) clear the default so the
    built-in voice speaks."""
    st = store()
    if pid:
        if pid == stock_id() and stock_profile(build=True) != pid:
            raise RuntimeError("the stock voice could not be built")
        st.update(pid, default=True)
        return
    cur = st.default_id()
    if cur:
        st.update(cur, default=False)


def fillers(profile_id: str | None) -> list[np.ndarray] | None:
    """Every pre-rendered filler of this profile (int16, 24 kHz), or None."""
    from .tts import builtin_id
    if not profile_id or builtin_id(profile_id):
        return None
    import soundfile as sf

    from dispatch_voice.profiles import FILLERS
    st = store()
    out = []
    for i in range(len(FILLERS)):
        try:
            a, sr = sf.read(st.file(profile_id, f"fillers/{i}.wav"), dtype="float32")
        except Exception:
            continue
        if sr == SAMPLE_RATE:
            out.append((np.clip(a, -1, 1) * 32767).astype("<i2"))
    return out or None


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #

def install():
    """Register the engine, resolver and fillers; return the profiles router
    (to be included under ``/api/voice/profiles``), or None when the
    Chatterbox models are not fetched (nothing registered).

    The router is built on a lazy store, so building it never creates
    ``<data>/voices`` at setup; a box that never opens the Voices pane never
    gets the directory. The operator gate and the Safe-Mode block are
    DisPatch's (``ctx.include_router``), so the router carries no auth of its
    own (``allow_open``); the standalone ``dispatch-voice serve`` page is the
    only other caller of ``build_router`` and runs on loopback."""
    from dispatch_voice.router import build_router

    from . import registry
    from .engines import chatterbox
    if not chatterbox.available():
        return None
    chatterbox.register()
    registry.set_profile_resolver(resolve_profile)
    # The profile fillers are NOT registered as acknowledgements any more: the
    # session plays the render queue's lines (SPEC §6.2, host/acks.py).
    log.info("voice: cloned voices mounted at /api/voice/profiles (unlocked tier only)")
    return build_router(_RigAwareStore(), allow_open=True, ui=False, fillers=False)
