"""Connects DisPatch Voice to ``forge-voice-core``.

The engines moved into the ``forge_voice_core`` package, which knows no application. It asks its host
(:mod:`forge_voice_core.hostlink`) for everything application-shaped: where the models are, the user's
settings, which voice a bot speaks with, the credential store, the power-profile hold. This module is
DisPatch's answer, installed when ``dispatch_voice`` is imported, BEFORE any engine module is loaded
(they bind their logger names at import, so the DisPatch names are set here first).

Everything is imported lazily inside the methods: ``import dispatch_voice`` stays metadata-only and
fast, which DisPatch's extension discovery relies on.
"""
from __future__ import annotations

from pathlib import Path

from forge_voice_core import _log, hostlink

#: The logger names DisPatch has always used, so its log routing does not change.
_log.set_name("dispatch.voice", overrides={
    "opus": "dispatch_voice.transport.opus",
    "websockets": "websockets.client.dispatch_voice",
})


class DispatchHostLink(hostlink.HostLink):
    client_name = "dispatch-voice"

    @property
    def client_version(self) -> str:                    # type: ignore[override]
        from . import __version__
        return __version__

    # ---- folders ------------------------------------------------------------ #
    def data_dir(self) -> Path:
        from .host import context
        return context.paths().voice_yaml.parent

    def models_root(self) -> Path:
        from .host import context
        return context.paths().models_dir

    def voices_dir(self) -> Path:
        from .host import context
        return context.paths().voices_dir

    def state_dir(self) -> Path:
        from .host import context
        return context.paths().state_dir

    def has_state_dir(self) -> bool:
        from .host import context
        return context.get_context() is not None or bool(context.data_dir_override())

    def legacy_token_path(self) -> Path:
        return Path.home() / ".config" / "local-chat" / "speechforge.env"

    # ---- settings and identity -------------------------------------------- #
    def load_settings(self):
        from .host import config
        return config.load_settings()

    def settings_schema_json(self) -> dict:
        from .host import settings_schema
        return settings_schema.schema_json()

    def bot_voice(self, bot_id: str | None) -> str:
        from .host import context
        return context.bot_voice(bot_id)

    # ---- cloned voices ------------------------------------------------------ #
    def profiles(self):
        from .host import profiles_bridge
        return profiles_bridge

    def ensure_engine(self) -> bool:
        from .host import profiles_bridge
        return profiles_bridge.ensure_engine()

    # ---- credentials, power, router ---------------------------------------- #
    def secret_store(self):
        from .host import context
        ctx = context.get_context()
        store = getattr(ctx, "secrets", None) if ctx is not None else None
        return store if store is not None and all(hasattr(store, m) for m in ("get", "set", "delete")) else None

    def power(self):
        from .host import power
        return power

    def router_mounted(self) -> bool:
        from .host import routes
        return "dispatch-voice" in routes._LOADED


hostlink.install(DispatchHostLink())
