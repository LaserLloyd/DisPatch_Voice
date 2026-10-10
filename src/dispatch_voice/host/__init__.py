"""Voice chat: hands-free voice for DisPatch threads (unlocked tier only).

This package is the host side of the DisPatch Voice add-on: what plugs into
DisPatch's extension host (``app/extensions.py``, contracts §B.1). DisPatch
discovers the add-on through the ``dispatch.extensions`` entry point, checks
the API handshake and calls :func:`setup` with an ``ExtensionContext``.

Speech runs on this machine's CPU:

* Silero VAD and Smart Turn v3.2 handle turn taking, with turn merge and a
  patience setting.
* Parakeet TDT 0.6B turns speech into text.
* A swappable TTS engine speaks the reply (this package's Chatterbox-Turbo
  by default, with optional plugins such as dispatch-voice-kokoro; see tts.py).

The transcript goes through the app's NORMAL send path (``ctx.submit_turn``),
so a spoken message lands in the thread like a typed one and any bot type
can answer.

Voice chat is OFF until the owner turns it on in Settings › Voice
(``enabled`` in ``<data>/voice.yaml``; ``DISPATCH_VOICE=1`` / ``0`` overrides
the switch), and it needs the model files (``dispatch-voice fetch-models``).
See docs/voice-chat/host.md.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("dispatch.voice")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def setup(ctx) -> None:
    """Wire the add-on into DisPatch (contracts §B.4, in this order):
    ``serve_static`` → ``include_router("/api/voice")`` →
    ``thread_websocket("voice", ws.serve, max_total=max_sessions)`` →
    ``set_bot_field_validator("voice")`` → ``set_bot_field_validator("voice_rp")`` →
    ``register_tools`` → ``on_startup(prewarm, thread=True)`` →
    ``on_shutdown(release + power.release_all)`` → the add-on's own plugin
    seam ``registry.mount_extensions`` (``DISPATCH_VOICE_EXTENSIONS``, where
    the optional Kokoro plugin plugs in). A raise here is DisPatch's state
    ``failed`` with the routes rolled back (rule 1); nothing is retried."""
    from . import context, profiles_bridge, registry, routes, ws
    from .config import load_settings
    from .tools import tool_specs

    context.set_context(ctx)
    routes.ASSETS_VERSION = ctx.serve_static(STATIC_DIR, entry="index.js", css=["voice.css"], locales=True)
    settings = load_settings()

    router = routes.build_router()
    profiles_router = profiles_bridge.install()
    if profiles_router is not None:
        router.include_router(profiles_router, prefix="/profiles")
    ctx.include_router(router, prefix="/api/voice", body_limits=routes.body_limits())

    ctx.thread_websocket("voice", ws.serve, max_total=settings.max_sessions)
    ctx.set_bot_field_validator("voice", routes.validate_bot_voice)
    ctx.set_bot_field_validator("voice_rp", routes.validate_bot_voice_rp)
    ctx.register_tools(tool_specs())
    ctx.on_startup(routes.prewarm_start, thread=True)
    ctx.on_shutdown(routes.shutdown)

    loaded = ["dispatch-voice"] if profiles_router is not None else []
    loaded += registry.mount_extensions(ctx)
    routes._LOADED[:] = loaded
    log.info("voice: DisPatch Voice set up (api %s, cloned voices %s, plugins %s, power hold %s)",
             getattr(ctx, "api_version", "?"), "on" if profiles_router is not None else "off (models not fetched)",
             loaded[1:] or "none", settings.power_hold)
