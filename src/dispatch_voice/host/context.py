"""The DisPatch ``ExtensionContext`` this process was set up with, and the
paths and bot facts the rest of ``host/`` read from it.

``host.setup(ctx)`` stores the context once. Outside DisPatch (the CLI,
``dispatch-voice serve``, unit tests without a fake context) there is no
context: the paths then come from the environment, which is also where a
DisPatch install keeps them (``DISPATCH_DATA_DIR``, ``DISPATCH_SPEECH_MODELS``).

The add-on never imports ``app.*`` (DisPatch): everything it needs from the
host arrives through the context (v3 contracts §B.1), and this module is the
one place that holds it.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("dispatch.voice")


@dataclass(frozen=True)
class Paths:
    """The add-on's files, by name (mirrors contracts §B.1 ``ExtPaths``)."""
    voice_yaml: Path
    acks_yaml: Path
    voices_dir: Path
    cache_dir: Path
    models_dir: Path
    state_dir: Path


_CTX = None
_LOCK = threading.Lock()


def set_context(ctx) -> None:
    """Called once by ``host.setup``. A later call replaces the context (tests)."""
    global _CTX
    with _LOCK:
        _CTX = ctx


def get_context():
    """The ExtensionContext, or None outside DisPatch."""
    return _CTX


def clear_context() -> None:
    """Tests only."""
    set_context(None)


def default_data_dir() -> Path:
    """Where a DisPatch install keeps its data when no context says otherwise."""
    raw = os.environ.get("DISPATCH_DATA_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "local-chat"


def env_paths(data: Path | None = None) -> Paths:
    """The paths derived from the environment (no context): the same layout
    DisPatch's ``ExtPaths`` describes, under ``DISPATCH_DATA_DIR``."""
    data = Path(data) if data is not None else default_data_dir()
    models = os.environ.get("DISPATCH_SPEECH_MODELS", "").strip()
    return Paths(
        voice_yaml=data / "voice.yaml",
        acks_yaml=data / "voice-acks.yaml",
        voices_dir=data / "voices",
        cache_dir=data / "voice-cache",
        models_dir=Path(models).expanduser() if models else data / "models",
        state_dir=data / "ext" / "voice",
    )


def paths() -> Paths:
    """The paths in force: the context's (inside DisPatch), else the environment's."""
    ctx = _CTX
    p = getattr(ctx, "paths", None) if ctx is not None else None
    if p is not None:
        return Paths(voice_yaml=Path(p.voice_yaml), acks_yaml=Path(p.acks_yaml),
                     voices_dir=Path(p.voices_dir), cache_dir=Path(p.cache_dir),
                     models_dir=Path(p.models_dir), state_dir=Path(p.state_dir))
    return env_paths()


# --------------------------------------------------------------------------- #
# Bots (contracts §B.1 ``ctx.bots``: get(id) -> BotView | None, all() -> [BotView])
# --------------------------------------------------------------------------- #

def bots_all() -> list:
    ctx = _CTX
    bots = getattr(ctx, "bots", None) if ctx is not None else None
    if bots is None:
        return []
    try:
        return list(bots.all() or [])
    except Exception:
        log.exception("voice: listing bots failed")
        return []


def bot(bot_id: str | None):
    """The BotView for ``bot_id``, or None (no context, unknown id)."""
    ctx = _CTX
    bots = getattr(ctx, "bots", None) if ctx is not None else None
    if bots is None or not bot_id:
        return None
    try:
        return bots.get(bot_id)
    except Exception:
        log.exception("voice: bot lookup failed for %s", bot_id)
        return None


def bot_voice(bot_id: str | None) -> str:
    """The voice the bot was saved with (``config.yaml bots[].voice``), or ""."""
    b = bot(bot_id)
    return (getattr(b, "voice", "") or "").strip() if b is not None else ""


def bot_backend(bot_id: str | None) -> str | None:
    """"gateway" | "api" | "advisor" | None (unknown bot or no context)."""
    ctx = _CTX
    fn = getattr(ctx, "bot_backend", None) if ctx is not None else None
    if fn is None:
        b = bot(bot_id)
        return getattr(b, "backend", None) if b is not None else None
    try:
        return fn(bot_id)
    except Exception:
        return None
