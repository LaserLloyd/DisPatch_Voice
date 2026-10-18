"""The seam between this package and the application that embeds it (the *host*).

The engines, the model registry, the rig client and the provider layer never import an application.
Whatever they need from one (where the models live, the user's settings, which voice a character
speaks with, a credential store, the power-profile hold) they ask :func:`get` for. The default
:class:`HostLink` answers from the environment and the platform's data folder, so the package works
on its own; an application subclasses it and calls :func:`install` once at start-up.

This is the whole contract in code. ``docs/HOST_API.md`` describes it for a host author.

Stdlib only.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any


def _data_home() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/AppData/Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base)


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else None


class HostLink:
    """What the package asks of its host. Every method has a standalone default."""

    #: The client name the speech rig sees in its ``hello`` and in the User-Agent.
    client_name = "forge-voice-core"
    #: The client version sent to the rig ("" = the package's own).
    client_version = ""

    # ---- folders ------------------------------------------------------------ #
    def data_dir(self) -> Path:
        """The package's own folder (``FORGE_VOICE_DATA_DIR``, else the platform's data folder)."""
        return _env_path("FORGE_VOICE_DATA_DIR") or (_data_home() / "forge-voice")

    def models_root(self) -> Path:
        """The ONE folder the model families live under (``speech/``, ``chatterbox/``)."""
        return _env_path("FORGE_VOICE_MODELS_DIR") or (self.data_dir() / "models")

    def voices_dir(self) -> Path:
        """Where voice profiles (cloned voices) are stored."""
        return _env_path("FORGE_VOICE_VOICES_DIR") or (self.data_dir() / "voices")

    def state_dir(self) -> Path:
        """Where small private state lives (the rig token file, the host credential file)."""
        return _env_path("FORGE_VOICE_STATE_DIR") or (self.data_dir() / "state")

    def has_state_dir(self) -> bool:
        """Is ``state_dir()`` the place for the rig token file? True unless the host still keeps a
        pre-existing home-folder file as the primary (DisPatch outside a configured data folder)."""
        return True

    def legacy_token_path(self) -> Path:
        """The old home-folder rig token file: read as a fallback, never written."""
        return Path.home() / ".config" / "forge-voice" / "speechforge.env"

    # ---- settings and identity -------------------------------------------- #
    def load_settings(self) -> Any:
        """The current settings object (see :class:`forge_voice_core.engine_settings.EngineSettings`
        for the attributes the engines read). Called often; a host should cache."""
        from .engine_settings import EngineSettings
        return EngineSettings()

    def settings_schema_json(self) -> dict:
        """The published schema rows (``Provider.settings_schema`` filters them by id)."""
        from . import settings_schema
        return settings_schema.ENGINE_SCHEMA.schema_json()

    def bot_voice(self, bot_id: str | None) -> str:
        """The voice profile a character was saved with ("" = none)."""
        return ""

    # ---- cloned voices ------------------------------------------------------ #
    def profiles(self) -> Any:
        """The profile bridge: an object with ``store()``, ``stock_id()``, ``stock_profile(build)`` and
        ``release_engine()`` (see :mod:`forge_voice_core.profile_bridge`, the default)."""
        from . import profile_bridge
        return profile_bridge

    def ensure_engine(self) -> bool:
        """Register the ``chatterbox`` TTS engine once its models are on disk (cheap, idempotent;
        called before every engine lookup so a download finished in the app needs no restart)."""
        from . import registry
        from .engines import chatterbox
        if registry.engine_spec("chatterbox") is not None:
            return True
        if not chatterbox.available():
            return False
        chatterbox.register()
        return True

    # ---- credentials, power, router ---------------------------------------- #
    def secret_store(self) -> Any | None:
        """The host's credential store (``get(name)``, ``set(name, value)``, ``delete(name)``) or None."""
        return None

    def power(self) -> Any | None:
        """An object with ``acquire(settings)`` and ``release()`` (hold a performance power profile while
        a voice chat is open), or None."""
        return None

    def router_mounted(self) -> bool:
        """Is the host's cloned-voices HTTP router mounted (it needs a restart after the first model
        download)? Only the model-download job's "restart required" flag reads it."""
        return False


_LOCK = threading.Lock()
_HOST: HostLink = HostLink()


def get() -> HostLink:
    """The installed host link (the standalone default until :func:`install` is called)."""
    return _HOST


def install(host: HostLink) -> HostLink:
    """Install ``host`` and return the previous link (a test restores it)."""
    global _HOST
    with _LOCK:
        prev, _HOST = _HOST, host
    return prev


def reset() -> None:
    """Back to the standalone default (tests)."""
    install(HostLink())
